//! IFMA52 Montgomery multiplication for 60-bit DEFAULT_Q.
//!
//! AVX-512-IFMA (Intel) / AVX-512-IFMA52 (AMD Zen 5) intrinsics
//! `_mm512_madd52{lo,hi}_epu64` operate on 52-bit masked operands in
//! 64-bit lanes, producing 104-bit products split as lo52/hi52.
//! DEFAULT_Q = 2^60 − 2^14 + 1 exceeds the 52-bit window by 8 bits, so
//! Montgomery multiplication requires a split-52 decomposition:
//!
//! ```text
//! a = a_hi · 2^52 + a_lo  (a_hi < 2^8, a_lo < 2^52)
//! b = b_hi · 2^52 + b_lo  (b_hi < 2^8, b_lo < 2^52)
//! a · b = p00 + (p10 + p01) · 2^52 + p11 · 2^104
//! where p00 = a_lo · b_lo  (104 bits, via madd52{lo,hi})
//!       p10 = a_hi · b_lo  (60 bits, via madd52{lo,hi})
//!       p01 = a_lo · b_hi  (60 bits, via madd52{lo,hi})
//!       p11 = a_hi · b_hi  (16 bits, via madd52lo)
//! ```
//!
//! The 120-bit product `ab` is assembled from these partial products,
//! then Montgomery-reduced using the standard REDC pattern:
//!
//! ```text
//! m = (ab as u64) · q_inv_neg mod 2^64
//! t = (ab + m · q) >> 64
//! if t >= q { t -= q }
//! ```
//!
//! This module ships two entry points:
//! - `mont_mul_split52`: scalar reference implementation, byte-identical
//!   to `NttContext`'s Montgomery multiply.
//! - `pointwise_mul_ifma52_8wide`: 8-wide SIMD implementation gated via
//!   `#[target_feature(enable = "avx512ifma")]`. Runtime dispatch via
//!   `is_x86_feature_detected!("avx512ifma")`.
//!
//! Scalar fallback (`mont_mul_split52`) preserves correctness on
//! wasm32 and on hardware without IFMA52.
//!
//! # Safety invariants
//!
//! All `unsafe` blocks + `unsafe fn` entry points in the `avx512_ifma`
//! submodule share ONE invariant:
//!
//! > The host CPU must have the `avx512f` + `avx512ifma` features
//! > enabled at runtime. Callers establish this via
//! > `std::is_x86_feature_detected!("avx512ifma")` BEFORE any call
//! > into the public SIMD entry points (`pointwise_mul_ifma52_8wide`,
//! > `mont_mul_split52_x8`). The functions themselves carry
//! > `#[target_feature(enable = "avx512f,avx512ifma")]` which makes
//! > the intrinsic calls well-defined IFF the runtime feature gate is
//! > satisfied.
//!
//!
//! On wasm32 and non-x86_64 targets the `avx512_ifma` submodule is
//! `#[cfg(target_arch = "x86_64")]`-gated off; the scalar
//! `mont_mul_split52` remains the sole entry point. No
//! `is_x86_feature_detected` call fires under wasm32 (the macro is
//! not defined for that target).

const MASK52: u64 = (1u64 << 52) - 1;

/// Split a 60-bit operand into (hi, lo) halves where lo < 2^52 and hi < 2^8.
#[inline]
const fn split52(a: u64) -> (u64, u64) {
    let a_lo = a & MASK52;
    let a_hi = a >> 52;
    (a_hi, a_lo)
}

/// Scalar split-52 Montgomery multiplication, reference implementation
/// for correctness KAT. Not vectorized; returns `a · b · R^{-1} mod q`
/// where `R = 2^64`. Byte-identical to `NttContext`'s Montgomery multiply.
///
/// # Arguments
///
/// * `a`, `b`: operands in `[0, q)`, each < 2^60.
/// * `q`: modulus (in this module's intended use: DEFAULT_Q).
/// * `q_inv_neg`: precomputed `-q^{-1} mod 2^64`, as `NttContext` holds it.
#[inline]
pub fn mont_mul_split52(a: u64, b: u64, q: u64, q_inv_neg: u64) -> u64 {
    let (a_hi, a_lo) = split52(a);
    let (b_hi, b_lo) = split52(b);

    // Partial products emulating the IFMA52 madd52lo/madd52hi outputs.
    //
    // For the SIMD path these four pairs would each be one madd52lo +
    // one madd52hi instruction. The scalar reference uses u128 and hand-
    // extracts the 52-bit halves so the KAT locks the composition math
    // that the SIMD path must reproduce bit-identically.
    let p00 = (a_lo as u128) * (b_lo as u128); // 52b × 52b → 104b
    let p00_lo = (p00 as u64) & MASK52; // bits [51:0]
    let p00_hi = ((p00 >> 52) as u64) & MASK52; // bits [103:52]

    let p10 = (a_hi as u128) * (b_lo as u128); // 8b × 52b → 60b
    let p10_lo = (p10 as u64) & MASK52;
    let p10_hi = ((p10 >> 52) as u64) & MASK52;

    let p01 = (a_lo as u128) * (b_hi as u128); // 52b × 8b → 60b
    let p01_lo = (p01 as u64) & MASK52;
    let p01_hi = ((p01 >> 52) as u64) & MASK52;

    let p11 = (a_hi as u128) * (b_hi as u128); // 8b × 8b → 16b (all in lo)
    let p11_lo = p11 as u64 & MASK52;

    // Assemble 120-bit product `ab` into two u64 halves (ab_lo, ab_hi).
    //
    // ab = p00 + (p10 + p01) · 2^52 + p11 · 2^104
    //
    // Layout in (ab_lo, ab_hi) with ab_lo = bits [63:0], ab_hi = bits [127:64]:
    //
    //   bits [51:0]   : p00_lo
    //   bits [103:52] : p00_hi + p10_lo + p01_lo  (cross-term sum)
    //   bits [111:104]: p10_hi + p01_hi + (p11_lo & 0xFF)
    //   bits [127:112]: (p11_lo >> 8)
    //
    // Widths: cross-term sum is 52b + 53b = 53 bits max (one carry bit
    // into the next stripe). p10_hi + p01_hi is 9 bits; with p11_lo's
    // bottom 8 = 10 bits. p11_lo upper = 8 bits.
    let cross_mid = p00_hi + p10_lo + p01_lo; // up to 53 bits
    let cross_mid_lo52 = cross_mid & MASK52;
    let cross_mid_carry = cross_mid >> 52; // 0 or 1

    let upper_9 = p10_hi + p01_hi + cross_mid_carry; // up to 10 bits
    let p11_low8 = p11_lo & 0xFF; // bits [111:104] contribution
    let p11_high = p11_lo >> 8; // bits [127:112]

    let ab_lo: u64 = p00_lo | (cross_mid_lo52 << 52); // bits [63:0]
    let ab_hi: u64 = (cross_mid_lo52 >> 12)                // top 40b of cross_mid into bits [39:0] of ab_hi
        + ((upper_9 + p11_low8) << 40)                     // bits [111:72] → shifted to [47:8] of ab_hi
        + (p11_high << 48); // bits [127:112] → [63:48] of ab_hi

    // Standard Montgomery REDC: m = (ab & 2^64 - 1) · q_inv_neg mod 2^64.
    // Then t = (ab + m · q) >> 64, producing the reduced result in [0, 2q).
    // Final conditional subtract brings it into [0, q).
    let m = ab_lo.wrapping_mul(q_inv_neg);
    let mq = (m as u128) * (q as u128);

    // ab + mq; we need the upper 64 bits. ab is (ab_lo, ab_hi); mq is 128-bit.
    let sum_lo = (ab_lo as u128) + (mq & ((1u128 << 64) - 1));
    let sum_carry = sum_lo >> 64;
    let t = (ab_hi as u128) + (mq >> 64) + sum_carry;
    let t = t as u64;

    if t >= q {
        t - q
    } else {
        t
    }
}

/// Scalar IFMA52 emulation for composition correctness checking.
///
/// The SIMD IFMA52 path composes four madd52lo + four madd52hi calls
/// (one each for p00, p10, p01, p11 — with p11_hi == 0 since 16b < 52b).
/// This helper makes the composition explicit for the KAT diff.
#[inline]
pub fn ifma52_product_lohi(a: u64, b: u64) -> (u64, u64) {
    // Returns (ab_lo_64, ab_hi_64) of the full 120-bit unsigned product
    // a · b where each operand is assumed to fit in 60 bits. The SIMD
    // path uses IFMA52 intrinsics for the four 52×52 partial products;
    // this scalar path uses u128 and masks to produce byte-identical
    // values.
    let (a_hi, a_lo) = split52(a);
    let (b_hi, b_lo) = split52(b);

    let p00 = (a_lo as u128) * (b_lo as u128);
    let p00_lo = (p00 as u64) & MASK52;
    let p00_hi = ((p00 >> 52) as u64) & MASK52;

    let p10 = (a_hi as u128) * (b_lo as u128);
    let p10_lo = (p10 as u64) & MASK52;
    let p10_hi = ((p10 >> 52) as u64) & MASK52;

    let p01 = (a_lo as u128) * (b_hi as u128);
    let p01_lo = (p01 as u64) & MASK52;
    let p01_hi = ((p01 >> 52) as u64) & MASK52;

    let p11 = (a_hi as u128) * (b_hi as u128);
    let p11_lo = p11 as u64 & MASK52;

    let cross_mid = p00_hi + p10_lo + p01_lo;
    let cross_mid_lo52 = cross_mid & MASK52;
    let cross_mid_carry = cross_mid >> 52;

    let upper_9 = p10_hi + p01_hi + cross_mid_carry;
    let p11_low8 = p11_lo & 0xFF;
    let p11_high = p11_lo >> 8;

    let ab_lo: u64 = p00_lo | (cross_mid_lo52 << 52);
    let ab_hi: u64 = (cross_mid_lo52 >> 12) + ((upper_9 + p11_low8) << 40) + (p11_high << 48);

    (ab_lo, ab_hi)
}

#[cfg(target_arch = "x86_64")]
#[allow(
    clippy::incompatible_msrv,
    clippy::wildcard_imports,
    reason = "AVX-512 intrinsics stabilised in 1.89, which is this crate's MSRV; enumerating ~30 intrinsics per import buys nothing"
)]
pub mod avx512_ifma {
    //! 8-wide SIMD IFMA52 Montgomery via AVX-512-IFMA intrinsics.
    //!
    //! Runtime dispatch: callers MUST check
    //! `is_x86_feature_detected!("avx512ifma")` before invoking these
    //! routines. Scalar fallback at `super::mont_mul_split52`.
    //!
    //! All entry points are `unsafe fn` with `#[target_feature(...)]`.
    //! The single safety invariant for the entire module is: the host
    //! CPU must have the `avx512f` + `avx512ifma` features enabled at
    //! runtime. Intrinsic calls inside these functions do not need their
    //! own `unsafe { }` wrappers — the enclosing `unsafe fn` is the
    //! one safety boundary.

    use std::arch::x86_64::*;

    const MASK52_VEC: u64 = (1u64 << 52) - 1;

    /// 8-wide split-52 Montgomery multiply using AVX-512-IFMA intrinsics.
    ///
    /// Computes `a[i] · b[i] · R^{-1} mod q` for 8 lanes in parallel,
    /// where R = 2^64 and inputs are in [0, q) with q ≤ 2^60.
    ///
    /// # Safety
    ///
    /// Caller MUST ensure `is_x86_feature_detected!("avx512ifma")` is
    /// true on the executing CPU.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub unsafe fn mont_mul_split52_x8(
        a: __m512i,
        b: __m512i,
        q_vec: __m512i,
        q_inv_neg_vec: __m512i,
    ) -> __m512i {
        let mask52 = _mm512_set1_epi64(MASK52_VEC as i64);

        // Split operands.
        let a_lo = _mm512_and_si512(a, mask52);
        let a_hi = _mm512_srli_epi64::<52>(a);
        let b_lo = _mm512_and_si512(b, mask52);
        let b_hi = _mm512_srli_epi64::<52>(b);

        let zero = _mm512_setzero_si512();

        // Partial products via IFMA52. Each madd52{lo,hi} takes acc +
        // (a52 * b52)[_52..52+52 or 52+52..52+52+52].
        let p00_lo = _mm512_madd52lo_epu64(zero, a_lo, b_lo);
        let p00_hi = _mm512_madd52hi_epu64(zero, a_lo, b_lo);
        let p10_lo = _mm512_madd52lo_epu64(zero, a_hi, b_lo);
        let p10_hi = _mm512_madd52hi_epu64(zero, a_hi, b_lo);
        let p01_lo = _mm512_madd52lo_epu64(zero, a_lo, b_hi);
        let p01_hi = _mm512_madd52hi_epu64(zero, a_lo, b_hi);
        let p11_lo = _mm512_madd52lo_epu64(zero, a_hi, b_hi);

        // Assemble 120-bit product per lane into (ab_lo, ab_hi).
        //
        // ab = p00 + (p10 + p01) * 2^52 + p11 * 2^104
        //
        // cross_mid = p00_hi + p10_lo + p01_lo (up to 53 bits per lane)
        // cross_mid_lo52 = cross_mid & mask52
        // cross_mid_carry = cross_mid >> 52
        let cross_mid = _mm512_add_epi64(_mm512_add_epi64(p00_hi, p10_lo), p01_lo);
        let cross_mid_lo52 = _mm512_and_si512(cross_mid, mask52);
        let cross_mid_carry = _mm512_srli_epi64::<52>(cross_mid);

        // upper_9 = p10_hi + p01_hi + cross_mid_carry (up to 10 bits per lane)
        let upper_9 = _mm512_add_epi64(_mm512_add_epi64(p10_hi, p01_hi), cross_mid_carry);
        let mask8 = _mm512_set1_epi64(0xFFi64);
        let p11_low8 = _mm512_and_si512(p11_lo, mask8);
        let p11_high = _mm512_srli_epi64::<8>(p11_lo);

        // ab_lo: bits [63:0] = p00_lo | (cross_mid_lo52 << 52)
        let cross_mid_shifted = _mm512_slli_epi64::<52>(cross_mid_lo52);
        let ab_lo = _mm512_or_si512(p00_lo, cross_mid_shifted);

        // ab_hi: see scalar reference for bit positioning.
        // ab_hi = (cross_mid_lo52 >> 12)
        //       + ((upper_9 + p11_low8) << 40)
        //       + (p11_high << 48)
        let cm_top = _mm512_srli_epi64::<12>(cross_mid_lo52);
        let upper_plus_low = _mm512_add_epi64(upper_9, p11_low8);
        let upper_shifted = _mm512_slli_epi64::<40>(upper_plus_low);
        let p11_hi_shifted = _mm512_slli_epi64::<48>(p11_high);
        let ab_hi = _mm512_add_epi64(_mm512_add_epi64(cm_top, upper_shifted), p11_hi_shifted);

        // Montgomery REDC (scalar per lane): m = ab_lo * q_inv_neg mod 2^64
        //                                    t = (ab + m * q) >> 64
        //                                    if t >= q { t -= q }
        //
        // AVX-512 lacks a full 64×64→128 mul, but `_mm512_madd52{lo,hi}`
        // gives us 52×52 partials. For the reduction we need the HIGH
        // 64 bits of (ab_lo + m*q)[128:64], treating ab_lo as the low
        // 64 of a 128-bit value.
        //
        // Use split-52 multiply for (m * q) as well:
        let m = _mm512_mullo_epi64(ab_lo, q_inv_neg_vec);
        let m_lo = _mm512_and_si512(m, mask52);
        let m_hi = _mm512_srli_epi64::<52>(m);
        let q_lo = _mm512_and_si512(q_vec, mask52);
        let q_hi = _mm512_srli_epi64::<52>(q_vec);

        let mq_p00_lo = _mm512_madd52lo_epu64(zero, m_lo, q_lo);
        let mq_p00_hi = _mm512_madd52hi_epu64(zero, m_lo, q_lo);
        let mq_p10_lo = _mm512_madd52lo_epu64(zero, m_hi, q_lo);
        let mq_p10_hi = _mm512_madd52hi_epu64(zero, m_hi, q_lo);
        let mq_p01_lo = _mm512_madd52lo_epu64(zero, m_lo, q_hi);
        let mq_p01_hi = _mm512_madd52hi_epu64(zero, m_lo, q_hi);
        let mq_p11_lo = _mm512_madd52lo_epu64(zero, m_hi, q_hi);

        let mq_cross_mid = _mm512_add_epi64(_mm512_add_epi64(mq_p00_hi, mq_p10_lo), mq_p01_lo);
        let mq_cross_mid_lo52 = _mm512_and_si512(mq_cross_mid, mask52);
        let mq_cross_mid_carry = _mm512_srli_epi64::<52>(mq_cross_mid);

        let mq_upper_9 =
            _mm512_add_epi64(_mm512_add_epi64(mq_p10_hi, mq_p01_hi), mq_cross_mid_carry);
        let mq_p11_low8 = _mm512_and_si512(mq_p11_lo, mask8);
        let mq_p11_high = _mm512_srli_epi64::<8>(mq_p11_lo);

        let mq_cm_shifted = _mm512_slli_epi64::<52>(mq_cross_mid_lo52);
        let mq_lo = _mm512_or_si512(mq_p00_lo, mq_cm_shifted);

        let mq_cm_top = _mm512_srli_epi64::<12>(mq_cross_mid_lo52);
        let mq_upper_plus_low = _mm512_add_epi64(mq_upper_9, mq_p11_low8);
        let mq_upper_shifted = _mm512_slli_epi64::<40>(mq_upper_plus_low);
        let mq_p11_hi_shifted = _mm512_slli_epi64::<48>(mq_p11_high);
        let mq_hi = _mm512_add_epi64(
            _mm512_add_epi64(mq_cm_top, mq_upper_shifted),
            mq_p11_hi_shifted,
        );

        // Compute (ab_lo + mq_lo) per lane, capture carry into mq_hi.
        // AVX-512 has no direct "add with carry" — emulate via
        // (ab_lo + mq_lo) < ab_lo test for carry.
        let sum_lo = _mm512_add_epi64(ab_lo, mq_lo);
        let carry_mask = _mm512_cmplt_epu64_mask(sum_lo, ab_lo);
        let one = _mm512_set1_epi64(1i64);
        let carry = _mm512_maskz_mov_epi64(carry_mask, one);

        // t = ab_hi + mq_hi + carry
        let t = _mm512_add_epi64(_mm512_add_epi64(ab_hi, mq_hi), carry);

        // Conditional subtract: if t >= q then t -= q
        let ge_mask = _mm512_cmpge_epu64_mask(t, q_vec);
        _mm512_mask_sub_epi64(t, ge_mask, t, q_vec)
    }

    /// Pointwise Montgomery multiply of two standard-form coefficient
    /// arrays using 8-wide IFMA52. Both inputs in [0, q); output also
    /// in [0, q) in Montgomery form (i.e. a * b * R^{-1} mod q).
    ///
    /// # Safety
    ///
    /// Caller MUST ensure `is_x86_feature_detected!("avx512ifma")` is
    /// true on the executing CPU. Array lengths must be multiples of 8.
    #[target_feature(enable = "avx512f,avx512ifma")]
    pub unsafe fn pointwise_mont_mul_x8(
        a: &[u64],
        b: &[u64],
        result: &mut [u64],
        q: u64,
        q_inv_neg: u64,
    ) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());
        assert_eq!(a.len() % 8, 0, "array length must be multiple of 8");

        let q_vec = _mm512_set1_epi64(q as i64);
        let q_inv_neg_vec = _mm512_set1_epi64(q_inv_neg as i64);

        for i in (0..a.len()).step_by(8) {
            let av = _mm512_loadu_si512(a[i..].as_ptr().cast());
            let bv = _mm512_loadu_si512(b[i..].as_ptr().cast());
            let rv = mont_mul_split52_x8(av, bv, q_vec, q_inv_neg_vec);
            _mm512_storeu_si512(result[i..].as_mut_ptr().cast(), rv);
        }
    }
}

/// Shape-compatible stub so non-x86_64 targets still compile the dispatch
/// site. Aborting beats returning wrong residues silently.
#[cfg(not(target_arch = "x86_64"))]
pub mod avx512_ifma {
    /// # Safety
    ///
    /// Unreachable on this target: callers MUST gate on
    /// `is_x86_feature_detected!("avx512ifma")`, which is false everywhere
    /// off x86_64. Calling it anyway aborts.
    #[allow(unused_variables)]
    #[allow(
        clippy::panic,
        reason = "abort is the contract: the scalar fallback at super::mont_mul_split52 is the path off x86_64"
    )]
    pub unsafe fn pointwise_mont_mul_x8(
        a: &[u64],
        b: &[u64],
        result: &mut [u64],
        q: u64,
        q_inv_neg: u64,
    ) {
        panic!("AVX-512-IFMA not available on this target architecture");
    }
}
