//! Modular arithmetic operations.
//!
//! Provides stateless modular arithmetic functions over Z_q.
//! These are simpler alternatives to the Montgomery-based `ModQ` type
//! when performance is not critical.

use subtle::{Choice, ConditionallySelectable, ConstantTimeGreater};

/// Set when `v < 0`; the sign bit read without a comparison.
#[inline]
pub(crate) fn ct_is_negative(v: i64) -> Choice {
    Choice::from(((v >> 63) & 1) as u8)
}

/// `v - q` when `v >= q`, else `v`.
///
/// `ct_gt(q-1)` rather than `!ct_lt(q)`: subtle builds `ct_lt` out of five
/// barriered ops and this sits under every noise coefficient.
#[inline]
pub(crate) fn ct_sub_if_ge(v: u64, q: u64) -> u64 {
    let ge = v.ct_gt(&q.wrapping_sub(1));
    u64::conditional_select(&v, &v.wrapping_sub(q), ge)
}

/// `a - b`, plus `q` when that borrows: `(a - b) mod q` for `a, b` in `[0, q)`.
///
/// The NTT's correction, on client secrets and in the server's inner loop, so
/// neither a branch nor subtle's volatile barrier. LLVM folds a borrow mask
/// back into a select, and x86-64 cmov conversion turns selects in the
/// Montgomery and Shoup loops into jumps; a mask held opaque by an empty asm
/// block stops that but measured slower on respond. So x86-64 gets its `cmov`
/// written out. Elsewhere it is the borrow mask of the lattice reference code,
/// which wasm32 emits as `select`.
#[inline]
pub(crate) fn sub_mod_branchless(a: u64, b: u64, q: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let mut out = a.wrapping_sub(b);
        let fixed = out.wrapping_add(q);
        // SAFETY: registers only: one compare and one conditional move.
        unsafe {
            core::arch::asm!(
                "cmp {a}, {b}",
                "cmovb {out}, {fixed}",
                a = in(reg) a,
                b = in(reg) b,
                fixed = in(reg) fixed,
                out = inout(reg) out,
                options(pure, nomem, nostack)
            );
        }
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        sub_mod_mask(a, b, q)
    }
}

/// [`sub_mod_branchless`] off x86-64; built in tests so x86-64 checks it too.
#[cfg(any(test, not(target_arch = "x86_64")))]
#[inline]
pub(crate) fn sub_mod_mask(a: u64, b: u64, q: u64) -> u64 {
    let (diff, borrow) = a.overflowing_sub(b);
    let mask = 0u64.wrapping_sub(u64::from(borrow));
    diff.wrapping_add(q & mask)
}

/// `x - q` when `x >= q`, else `x`, through [`sub_mod_branchless`].
#[inline]
pub(crate) fn csubq(x: u64, q: u64) -> u64 {
    sub_mod_branchless(x, q, q)
}

/// Largest modulus whose Shoup remainder, below `2q`, still fits a `u64`.
pub(crate) const SHOUP_NARROW_MAX_MODULUS: u64 = 1 << 63;

/// `floor(b * 2^64 / q)`, the Shoup companion of a public `b < q`.
#[inline]
pub(crate) fn shoup_precompute(b: u64, q: u64) -> u64 {
    ((u128::from(b) << 64) / u128::from(q)) as u64
}

/// `a * b mod q` for a secret `a` and a public `b < q` with companion `b_shoup`.
///
/// `b_shoup` is short of `b * 2^64 / q` by less than one, so for every
/// `a < 2^64` the quotient estimate is at most one short and the remainder
/// lands in `[0, 2q)`: one branch-free subtraction finishes it, where
/// `(a * b) % q` would divide the secret product (a software call on both
/// x86-64 and wasm32 at 128 bits). Needs `q <= SHOUP_NARROW_MAX_MODULUS`.
#[inline]
pub(crate) fn mul_mod_shoup(a: u64, b: u64, b_shoup: u64, q: u64) -> u64 {
    let estimate = ((u128::from(a) * u128::from(b_shoup)) >> 64) as u64;
    csubq(a.wrapping_mul(b).wrapping_sub(estimate.wrapping_mul(q)), q)
}

/// [`mul_mod_shoup`] for any `q`: the remainder is held in 128 bits, where
/// `2q` cannot overflow.
#[inline]
pub(crate) fn mul_mod_shoup_wide(a: u64, b: u64, b_shoup: u64, q: u64) -> u64 {
    let estimate = (u128::from(a) * u128::from(b_shoup)) >> 64;
    let q = u128::from(q);
    let rem = (u128::from(a) * u128::from(b)).wrapping_sub(estimate * q);
    u128::conditional_select(&rem, &rem.wrapping_sub(q), rem.ct_gt(&(q - 1))) as u64
}

/// `a mod q` with the only division taken on the public modulus.
///
/// Barrett with a 2^64 radix. Writing `m = floor((2^64-1)/q)`, the estimate
/// `floor(a*m/2^64)` undershoots the true quotient by at most one for every
/// `a < 2^64`, so one conditional subtraction lands the remainder in [0, q).
/// `a % q` would instead divide by the secret operand, which is variable
/// latency on x86-64; `u64::MAX / q` divides by the public modulus and the
/// widening multiply that replaces it is fixed latency.
#[inline]
pub(crate) fn reduce_by_public_modulus(a: u64, q: u64) -> u64 {
    reduce_by_public_reciprocal(a, q, u64::MAX / q)
}

/// [`reduce_by_public_modulus`] with `recip = u64::MAX / q` hoisted out of a loop.
#[inline]
pub(crate) fn reduce_by_public_reciprocal(a: u64, q: u64, recip: u64) -> u64 {
    let quot = ((u128::from(a) * u128::from(recip)) >> 64) as u64;
    ct_sub_if_ge(a.wrapping_sub(quot.wrapping_mul(q)), q)
}

/// Stateless modular arithmetic operations over Z_q.
///
/// Provides basic modular operations without Montgomery representation.
/// Use this for simple operations; use `mod_q::ModQ` for performance-critical code.
///
/// # Example
///
/// ```
/// use raven_inspire::math::modular::ModQ;
///
/// let q = 1152921504606830593u64;
/// let sum = ModQ::add(100, 200, q);
/// assert_eq!(sum, 300);
/// ```
pub struct ModQ;

impl ModQ {
    /// Adds two values modulo q.
    ///
    /// # Arguments
    ///
    /// * `a` - First operand
    /// * `b` - Second operand
    /// * `q` - The modulus
    ///
    /// # Returns
    ///
    /// `(a + b) mod q`
    #[inline]
    pub fn add(a: u64, b: u64, q: u64) -> u64 {
        let sum = (a as u128) + (b as u128);
        (sum % (q as u128)) as u64
    }

    /// Subtracts two values modulo q.
    ///
    /// # Arguments
    ///
    /// * `a` - First operand (minuend)
    /// * `b` - Second operand (subtrahend)
    /// * `q` - The modulus
    ///
    /// # Returns
    ///
    /// `(a - b) mod q`
    #[inline]
    pub fn sub(a: u64, b: u64, q: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            q - (b - a)
        }
    }

    /// Multiplies two values modulo q.
    ///
    /// # Arguments
    ///
    /// * `a` - First operand
    /// * `b` - Second operand
    /// * `q` - The modulus
    ///
    /// # Returns
    ///
    /// `(a * b) mod q`
    #[inline]
    pub fn mul(a: u64, b: u64, q: u64) -> u64 {
        let prod = (a as u128) * (b as u128);
        (prod % (q as u128)) as u64
    }

    /// Negates a value modulo q.
    ///
    /// # Arguments
    ///
    /// * `a` - The value to negate
    /// * `q` - The modulus
    ///
    /// # Returns
    ///
    /// `(-a) mod q = (q - a) mod q`
    #[inline]
    pub fn negate(a: u64, q: u64) -> u64 {
        if a == 0 {
            0
        } else {
            q - a
        }
    }

    /// Converts a signed integer to its representation in Z_q.
    ///
    /// Negative values are mapped to their positive equivalents modulo q.
    ///
    /// Branch-free and division-free in `val`: this is the first hop out of the
    /// discrete Gaussian, so `val` is secret error and neither its sign nor its
    /// magnitude may steer control flow or a divider.
    ///
    /// # Arguments
    ///
    /// * `val` - The signed value
    /// * `q` - The modulus
    ///
    /// # Returns
    ///
    /// The unsigned representation in [0, q).
    #[inline]
    pub fn from_signed(val: i64, q: u64) -> u64 {
        let sign = val >> 63;
        let magnitude = (val ^ sign).wrapping_sub(sign) as u64;
        let rem = reduce_by_public_modulus(magnitude, q);
        // maps a zero remainder to 0 rather than to q
        let complement = ct_sub_if_ge(q.wrapping_sub(rem), q);
        u64::conditional_select(&rem, &complement, ct_is_negative(val))
    }

    /// Converts from Z_q to signed representation in [-q/2, q/2).
    ///
    /// Values in [0, q/2] remain positive; values in (q/2, q) become negative.
    ///
    /// # Arguments
    ///
    /// * `val` - The unsigned value in [0, q)
    /// * `q` - The modulus
    ///
    /// # Returns
    ///
    /// The signed representation in [-q/2, q/2).
    #[inline]
    pub fn to_signed(val: u64, q: u64) -> i64 {
        if val <= q / 2 {
            val as i64
        } else {
            -((q - val) as i64)
        }
    }

    /// Reduces a value modulo q.
    ///
    /// # Arguments
    ///
    /// * `a` - The value to reduce
    /// * `q` - The modulus
    ///
    /// # Returns
    ///
    /// `a mod q`
    #[inline]
    pub fn reduce(a: u64, q: u64) -> u64 {
        a % q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: u64 = 1152921504606830593;

    #[test]
    fn test_add() {
        assert_eq!(ModQ::add(5, 7, Q), 12);
        assert_eq!(ModQ::add(Q - 1, 2, Q), 1);
    }

    #[test]
    fn test_sub() {
        assert_eq!(ModQ::sub(10, 3, Q), 7);
        assert_eq!(ModQ::sub(3, 10, Q), Q - 7);
    }

    #[test]
    fn test_mul() {
        assert_eq!(ModQ::mul(5, 7, Q), 35);
    }

    #[test]
    fn test_negate() {
        assert_eq!(ModQ::negate(5, Q), Q - 5);
        assert_eq!(ModQ::negate(0, Q), 0);
    }

    #[test]
    fn test_from_signed() {
        assert_eq!(ModQ::from_signed(5, Q), 5);
        assert_eq!(ModQ::from_signed(-5, Q), Q - 5);
        assert_eq!(ModQ::from_signed(0, Q), 0);
    }

    fn from_signed_reference(val: i64, q: u64) -> u64 {
        let r = (i128::from(val)).rem_euclid(i128::from(q));
        r as u64
    }

    /// The branch-free lift must agree with Euclidean remainder everywhere,
    /// including a negative exact multiple of `q`, which must map to `0`, not `q`.
    #[test]
    fn from_signed_matches_euclidean_remainder() {
        let moduli = [Q, 12_289u64, 2u64, 1u64, u64::MAX, (1u64 << 62) + 1];
        let mut vals = vec![
            0i64,
            1,
            -1,
            5,
            -5,
            i64::MAX,
            i64::MIN,
            i64::MIN + 1,
            i64::MAX - 1,
        ];
        for q in moduli {
            let qi = q as i64;
            vals.extend([
                qi,
                -qi,
                qi.wrapping_sub(1),
                qi.wrapping_neg().wrapping_add(1),
            ]);
        }
        for q in moduli {
            for &val in &vals {
                let got = ModQ::from_signed(val, q);
                assert_eq!(got, from_signed_reference(val, q), "val={val} q={q}");
                assert!(got < q, "val={val} q={q} escaped [0, q)");
            }
        }

        let mut state = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..200_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let val = state as i64;
            for q in [Q, 12_289u64, u64::MAX] {
                assert_eq!(
                    ModQ::from_signed(val, q),
                    from_signed_reference(val, q),
                    "val={val} q={q}"
                );
            }
        }
    }

    #[test]
    fn from_signed_lifts_the_whole_gaussian_tailcut() {
        for q in [Q, 12_289u64] {
            for val in -64i64..=64 {
                assert_eq!(ModQ::from_signed(val, q), from_signed_reference(val, q));
            }
        }
    }

    #[test]
    fn test_to_signed() {
        assert_eq!(ModQ::to_signed(5, Q), 5);
        assert_eq!(ModQ::to_signed(Q - 5, Q), -5);
    }
}
