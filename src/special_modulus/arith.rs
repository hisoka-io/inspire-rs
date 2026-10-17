//! Arithmetic modulo a prime below 2^31 on 32-bit words, and its negacyclic transform.
//!
//! Every value is kept fully reduced in `[0, q)`. With `q < 2^31` a sum of two values fits
//! a word, and a Montgomery product `a * b * 2^-32` needs one 64-bit product and one
//! conditional subtraction. The kernels are plain loops marked `#[inline(always)]`, so a
//! caller compiled for a wider vector unit vectorises them without a second source.

use crate::math::primality::is_prime;
use crate::pir::error::{pir_err, Result};

/// A prime `q < 2^31` with its Montgomery constants (`R = 2^32`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Prime32 {
    pub(crate) q: u32,
    /// `-q^-1 mod 2^32`.
    neg_qinv: u32,
    /// `2^64 mod q`, which takes a value into Montgomery form in one product.
    r2: u32,
}

impl Prime32 {
    pub(crate) fn new(q: u64) -> Result<Self> {
        if q >= 1 << 31 || !is_prime(q) {
            return Err(pir_err!("{q} is not a prime below 2^31"));
        }
        let q32 = q as u32;
        // Newton's iteration doubles the correct low bits: 1 -> 2 -> 4 -> ... -> 32.
        let mut inv = 1u32;
        for _ in 0..5 {
            inv = inv.wrapping_mul(2u32.wrapping_sub(q32.wrapping_mul(inv)));
        }
        Ok(Self {
            q: q32,
            neg_qinv: inv.wrapping_neg(),
            r2: ((1u128 << 64) % u128::from(q)) as u32,
        })
    }

    /// `x - q` when `x >= q`, for `x < 2q`. The correction comes from the borrow of
    /// `x - q` (its top bit, as `q < 2^31`), never from a comparison.
    #[inline(always)]
    pub(crate) fn csub(&self, x: u32) -> u32 {
        let r = x.wrapping_sub(self.q);
        r.wrapping_add(self.q & 0u32.wrapping_sub(r >> 31))
    }

    #[inline(always)]
    pub(crate) fn add(&self, a: u32, b: u32) -> u32 {
        self.csub(a + b)
    }

    #[inline(always)]
    pub(crate) fn sub(&self, a: u32, b: u32) -> u32 {
        self.csub(a + self.q - b)
    }

    /// `a * b * 2^-32 mod q` for `a < 2^32` and `b < q`.
    #[inline(always)]
    pub(crate) fn mont(&self, a: u32, b: u32) -> u32 {
        let t = u64::from(a) * u64::from(b);
        let m = (t as u32).wrapping_mul(self.neg_qinv);
        // t and m * q are each below 2^63, so the sum does not wrap; the quotient is below 2q.
        self.csub(((t + u64::from(m) * u64::from(self.q)) >> 32) as u32)
    }

    /// `a * 2^32 mod q`: an operand that [`Self::mont`] multiplies without the `2^-32`.
    pub(crate) fn to_mont(&self, a: u32) -> u32 {
        self.mont(a, self.r2)
    }

    /// Residue of a small signed value, `|x| < q`, folded by its sign bit with no divide.
    #[inline(always)]
    pub(crate) fn small(&self, x: i64) -> u32 {
        (x as u32).wrapping_add(self.q & ((x >> 63) as u32))
    }

    /// Residue of any signed value. Divides: public data only.
    #[inline(always)]
    pub(crate) fn residue_public(&self, x: i64) -> u32 {
        x.rem_euclid(i64::from(self.q)) as u32
    }

    /// Representative in `(-q/2, q/2]`, shifted down by the sign of `q/2 - x`.
    #[inline(always)]
    pub(crate) fn centred(&self, x: u32) -> i64 {
        let above = (i64::from(self.q >> 1) - i64::from(x)) >> 63;
        i64::from(x) - (i64::from(self.q) & above)
    }

    pub(crate) fn pow(&self, base: u64, mut e: u64) -> u32 {
        let q = u64::from(self.q);
        let (mut acc, mut b) = (1u64, base % q);
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * b % q;
            }
            b = b * b % q;
            e >>= 1;
        }
        acc as u32
    }
}

/// Negacyclic transform over `Z_q[X]/(X^n + 1)`: forward Cooley-Tukey with bit-reversed
/// output, inverse Gentleman-Sande. Slot `k` holds the evaluation at an odd power of a
/// primitive `2n`-th root; the slot permutation of an automorphism is read off the
/// transform itself, so no caller depends on which root was chosen.
#[derive(Clone, Debug)]
pub(crate) struct Ntt32 {
    pub(crate) n: usize,
    pub(crate) p: Prime32,
    /// `psi^bitrev(k)` in Montgomery form.
    fwd: Vec<u32>,
    /// `psi^-bitrev(k)` in Montgomery form.
    inv: Vec<u32>,
    /// `n^-1` in Montgomery form.
    n_inv: u32,
    /// `psi^-bitrev(1) * n^-1` in Montgomery form: the last inverse stage's twiddle.
    last: u32,
}

fn bitrev(x: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        x.reverse_bits() >> (usize::BITS - bits)
    }
}

impl Ntt32 {
    pub(crate) fn new(n: usize, q: u64) -> Result<Self> {
        let p = Prime32::new(q)?;
        if !n.is_power_of_two() || n < 2 || !(q - 1).is_multiple_of(2 * n as u64) {
            return Err(pir_err!("{q} has no {}-th root of unity", 2 * n));
        }
        // The first g whose ((q-1)/2n)-th power has order exactly 2n.
        let psi = (2..q)
            .map(|g| p.pow(g, (q - 1) / (2 * n as u64)))
            .find(|&c| p.pow(u64::from(c), n as u64) == p.q - 1)
            .ok_or_else(|| pir_err!("no primitive {}-th root modulo {q}", 2 * n))?;
        let psi_inv = p.pow(u64::from(psi), 2 * n as u64 - 1);
        let bits = n.trailing_zeros();
        let (mut fwd, mut inv) = (vec![0u32; n], vec![0u32; n]);
        let (mut a, mut b) = (1u64, 1u64);
        let mut powers = Vec::with_capacity(n);
        for _ in 0..n {
            powers.push((a as u32, b as u32));
            a = a * u64::from(psi) % q;
            b = b * u64::from(psi_inv) % q;
        }
        for k in 0..n {
            let (f, i) = powers[bitrev(k, bits)];
            fwd[k] = p.to_mont(f);
            inv[k] = p.to_mont(i);
        }
        let n_inv = p.to_mont(p.pow(n as u64, q - 2));
        let last = p.mont(inv[1], n_inv);
        Ok(Self {
            n,
            p,
            fwd,
            inv,
            n_inv,
            last,
        })
    }

    #[inline(always)]
    pub(crate) fn forward(&self, a: &mut [u32]) {
        let n = self.n;
        debug_assert_eq!(a.len(), n);
        let mut t = n;
        let mut m = 1;
        while m < n {
            t >>= 1;
            let tw = &self.fwd[m..2 * m];
            // A fixed-width stage lets the compiler vectorise across blocks too narrow to
            // vectorise within.
            match t {
                1 => self.forward_stage::<1>(a, tw),
                2 => self.forward_stage::<2>(a, tw),
                4 => self.forward_stage::<4>(a, tw),
                8 => self.forward_stage::<8>(a, tw),
                _ => self.forward_wide(a, tw, t),
            }
            m <<= 1;
        }
    }

    #[inline(always)]
    fn forward_butterfly(&self, x: &mut u32, y: &mut u32, s: u32) {
        let p = &self.p;
        let u = *x;
        let v = p.mont(*y, s);
        *x = p.add(u, v);
        *y = p.sub(u, v);
    }

    #[inline(always)]
    fn forward_stage<const T: usize>(&self, a: &mut [u32], tw: &[u32]) {
        // Blocks of 2T as pairs of T-word halves.
        let halves = a.as_chunks_mut::<T>().0.as_chunks_mut::<2>().0;
        for ([lo, hi], &s) in halves.iter_mut().zip(tw) {
            for j in 0..T {
                self.forward_butterfly(&mut lo[j], &mut hi[j], s);
            }
        }
    }

    /// `t` a multiple of 16: the halves are walked in fixed 16-word blocks.
    #[inline(always)]
    fn forward_wide(&self, a: &mut [u32], tw: &[u32], t: usize) {
        for (chunk, &s) in a.chunks_exact_mut(2 * t).zip(tw) {
            let (lo, hi) = chunk.split_at_mut(t);
            let (lo, hi) = (lo.as_chunks_mut::<16>().0, hi.as_chunks_mut::<16>().0);
            for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                for j in 0..16 {
                    self.forward_butterfly(&mut x[j], &mut y[j], s);
                }
            }
        }
    }

    #[inline(always)]
    pub(crate) fn inverse(&self, a: &mut [u32]) {
        let n = self.n;
        debug_assert_eq!(a.len(), n);
        let mut t = 1;
        let mut m = n;
        while m > 2 {
            let h = m >> 1;
            let tw = &self.inv[h..2 * h];
            match t {
                1 => self.inverse_stage::<1>(a, tw),
                2 => self.inverse_stage::<2>(a, tw),
                4 => self.inverse_stage::<4>(a, tw),
                8 => self.inverse_stage::<8>(a, tw),
                _ => self.inverse_wide(a, tw, t),
            }
            t <<= 1;
            m = h;
        }
        // The last stage carries the 1/n scaling on both outputs.
        let p = &self.p;
        let (lo, hi) = a.split_at_mut(t);
        for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
            let (u, v) = (*x, *y);
            *x = p.mont(u + v, self.n_inv);
            *y = p.mont(u + p.q - v, self.last);
        }
    }

    #[inline(always)]
    fn inverse_butterfly(&self, x: &mut u32, y: &mut u32, s: u32) {
        let p = &self.p;
        let (u, v) = (*x, *y);
        *x = p.add(u, v);
        // `u + q - v < 2q` is a valid first operand without its subtraction.
        *y = p.mont(u + p.q - v, s);
    }

    #[inline(always)]
    fn inverse_stage<const T: usize>(&self, a: &mut [u32], tw: &[u32]) {
        // Blocks of 2T as pairs of T-word halves.
        let halves = a.as_chunks_mut::<T>().0.as_chunks_mut::<2>().0;
        for ([lo, hi], &s) in halves.iter_mut().zip(tw) {
            for j in 0..T {
                self.inverse_butterfly(&mut lo[j], &mut hi[j], s);
            }
        }
    }

    /// `t` a multiple of 16, as in [`Self::forward_wide`].
    #[inline(always)]
    fn inverse_wide(&self, a: &mut [u32], tw: &[u32], t: usize) {
        for (chunk, &s) in a.chunks_exact_mut(2 * t).zip(tw) {
            let (lo, hi) = chunk.split_at_mut(t);
            let (lo, hi) = (lo.as_chunks_mut::<16>().0, hi.as_chunks_mut::<16>().0);
            for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                for j in 0..16 {
                    self.inverse_butterfly(&mut x[j], &mut y[j], s);
                }
            }
        }
    }

    /// Slot `k` of `tau_t(f)` is slot `table[k]` of `f`, for every odd `t` in `exps`.
    pub(crate) fn automorphism_tables(&self, exps: &[usize]) -> Result<Vec<Vec<u16>>> {
        let n = self.n;
        if n > 1 << 16 {
            return Err(pir_err!("ring dimension {n} needs wider slot indices"));
        }
        let mut x = vec![0u32; n];
        x[1] = 1;
        self.forward(&mut x);
        let mut slot_of: Vec<(u32, u16)> =
            x.iter().enumerate().map(|(k, &v)| (v, k as u16)).collect();
        slot_of.sort_unstable();
        if slot_of.windows(2).any(|w| w[0].0 == w[1].0) {
            return Err(pir_err!(
                "transform is not an evaluation at {n} distinct roots"
            ));
        }
        exps.iter()
            .map(|&t| {
                let mut mono = vec![0u32; n];
                if t % (2 * n) < n {
                    mono[t % (2 * n)] = 1;
                } else {
                    mono[t % (2 * n) - n] = self.p.q - 1;
                }
                self.forward(&mut mono);
                mono.iter()
                    .map(|v| {
                        slot_of
                            .binary_search_by_key(v, |&(value, _)| value)
                            .map(|at| slot_of[at].1)
                            .map_err(|_| pir_err!("tau_{t} leaves the evaluation points"))
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    const PRIMES: [u64; 3] = [536_813_569, 2_147_389_441, 12_289];

    fn naive_negacyclic(a: &[u32], b: &[u32], q: u64) -> Vec<u32> {
        let n = a.len();
        let mut out = vec![0i128; n];
        for i in 0..n {
            for j in 0..n {
                let prod = i128::from(a[i]) * i128::from(b[j]);
                if i + j < n {
                    out[i + j] += prod;
                } else {
                    out[i + j - n] -= prod;
                }
            }
        }
        out.iter()
            .map(|&v| v.rem_euclid(i128::from(q)) as u32)
            .collect()
    }

    #[test]
    fn montgomery_product_is_exact_at_the_edges() {
        for q in PRIMES {
            let p = Prime32::new(q).unwrap();
            let q32 = q as u32;
            for &(a, b) in &[
                (0, 0),
                (q32 - 1, q32 - 1),
                (1, q32 - 1),
                (q32 / 2, q32 / 2 + 1),
            ] {
                let want = (u64::from(a) * u64::from(b) % q) as u32;
                assert_eq!(p.mont(a, p.to_mont(b)), want, "q {q} a {a} b {b}");
            }
            // The first operand may be any word.
            let a = u32::MAX;
            let want = (u64::from(a) * 5 % q) as u32;
            assert_eq!(p.mont(a, p.to_mont(5)), want);
        }
    }

    #[test]
    fn branch_free_helpers_match_their_reference() {
        for q in PRIMES {
            let p = Prime32::new(q).unwrap();
            let q32 = q as u32;
            for x in [0, 1, q32 - 1, q32, q32 + 1, 2 * q32 - 1] {
                assert_eq!(p.csub(x), x % q32, "csub {x} mod {q}");
            }
            for x in [0i64, 1, -1, q as i64 - 1, 1 - q as i64, 39, -39] {
                assert_eq!(p.small(x), p.residue_public(x), "small {x} mod {q}");
            }
            for x in [0, 1, q32 / 2, q32 / 2 + 1, q32 - 1] {
                let want = if x > q32 / 2 {
                    i64::from(x) - q as i64
                } else {
                    i64::from(x)
                };
                assert_eq!(p.centred(x), want, "centred {x} mod {q}");
            }
        }
    }

    #[test]
    fn a_composite_or_a_wide_modulus_is_refused() {
        assert!(Prime32::new(4097).is_err());
        assert!(Prime32::new((1 << 31) + 11).is_err());
        assert!(
            Ntt32::new(4096, 12_289).is_err(),
            "no 8192-th root mod 12289"
        );
    }

    #[test]
    fn transform_multiplies_negacyclically() {
        let mut rng = ChaCha20Rng::seed_from_u64(3);
        for (n, q) in [(16, 12_289u64), (64, 536_813_569), (256, 2_147_389_441)] {
            let ntt = Ntt32::new(n, q).unwrap();
            let a: Vec<u32> = (0..n).map(|_| rng.gen_range(0..q as u32)).collect();
            let b: Vec<u32> = (0..n).map(|_| rng.gen_range(0..q as u32)).collect();
            let (mut fa, mut fb) = (a.clone(), b.clone());
            ntt.forward(&mut fa);
            ntt.forward(&mut fb);
            let mut prod: Vec<u32> = fa
                .iter()
                .zip(&fb)
                .map(|(&x, &y)| ntt.p.mont(x, ntt.p.to_mont(y)))
                .collect();
            ntt.inverse(&mut prod);
            assert_eq!(prod, naive_negacyclic(&a, &b, q), "n {n} q {q}");
            let mut round = fa.clone();
            ntt.inverse(&mut round);
            assert_eq!(round, a);
        }
    }

    #[test]
    fn automorphism_tables_match_the_coefficient_map() {
        let (n, q) = (64usize, 536_813_569u64);
        let ntt = Ntt32::new(n, q).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(5);
        let f: Vec<u32> = (0..n).map(|_| rng.gen_range(0..q as u32)).collect();
        let exps = [1usize, 3, 5, 9, 127];
        let tables = ntt.automorphism_tables(&exps).unwrap();
        for (&t, table) in exps.iter().zip(&tables) {
            let mut tau = vec![0u32; n];
            for (i, &c) in f.iter().enumerate() {
                let k = (t * i) % (2 * n);
                if k < n {
                    tau[k] = c;
                } else {
                    tau[k - n] = ntt.p.sub(0, c);
                }
            }
            let (mut ff, mut ft) = (f.clone(), tau);
            ntt.forward(&mut ff);
            ntt.forward(&mut ft);
            let gathered: Vec<u32> = table.iter().map(|&s| ff[s as usize]).collect();
            assert_eq!(gathered, ft, "t {t}");
        }
    }
}
