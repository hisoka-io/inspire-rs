//! Montgomery reduction specialized to the Solinas prime DEFAULT_Q = 2^60 - 2^14 + 1.
//!
//! The generalized-Mersenne structure expands `m * q` to
//! `(m << 60) - (m << 14) + m`, replacing the dominant multiply of classical
//! REDC with shifts. Output is byte-identical to classical Montgomery at
//! DEFAULT_Q (`tests/solinas_montgomery_kat.rs`) and stays in Montgomery form.

use super::mod_q::DEFAULT_Q;
use super::modular::csubq;

/// `a * b * R^{-1} mod DEFAULT_Q`, R = 2^64, for `a, b` in `[0, DEFAULT_Q)`.
#[inline]
pub fn solinas_mont_mul_default_q(a: u64, b: u64, q_inv_neg: u64) -> u64 {
    const Q: u64 = DEFAULT_Q;
    let ab = (a as u128).wrapping_mul(b as u128);
    let m = (ab as u64).wrapping_mul(q_inv_neg);
    // m * q = (m << 60) - (m << 14) + m
    let mq = ((m as u128) << 60)
        .wrapping_sub((m as u128) << 14)
        .wrapping_add(m as u128);
    csubq((ab.wrapping_add(mq) >> 64) as u64, Q)
}

/// [`solinas_mont_mul_default_q`] over slices; Montgomery form in and out.
///
/// # Panics
///
/// If the lengths differ.
#[inline]
pub fn pointwise_solinas_mont_mul(a: &[u64], b: &[u64], result: &mut [u64], q_inv_neg: u64) {
    assert_eq!(a.len(), b.len(), "pointwise lengths mismatch");
    assert_eq!(a.len(), result.len(), "pointwise result length mismatch");
    for i in 0..a.len() {
        result[i] = solinas_mont_mul_default_q(a[i], b[i], q_inv_neg);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(
    clippy::incompatible_msrv,
    clippy::wildcard_imports,
    reason = "see `math::ifma52::avx512_ifma`: 1.89 intrinsics behind runtime CPUID dispatch"
)]
pub mod avx512_ifma {
    //! 8-wide Solinas-Montgomery via AVX-512-IFMA; split-52 for the `ab` product.
    //!
    //! Callers MUST gate on `is_x86_feature_detected!("avx512ifma")`.

    use super::DEFAULT_Q;
    use std::arch::x86_64::*;

    const MASK52: u64 = (1u64 << 52) - 1;

    /// [`super::solinas_mont_mul_default_q`] across 8 lanes.
    ///
    /// # Safety
    ///
    /// Host MUST have `avx512ifma`.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub unsafe fn solinas_mont_mul_x8(a: __m512i, b: __m512i, q_inv_neg_vec: __m512i) -> __m512i {
        let mask52 = _mm512_set1_epi64(MASK52 as i64);
        let q_vec = _mm512_set1_epi64(DEFAULT_Q as i64);
        let zero = _mm512_setzero_si512();

        let a_lo = _mm512_and_si512(a, mask52);
        let a_hi = _mm512_srli_epi64::<52>(a);
        let b_lo = _mm512_and_si512(b, mask52);
        let b_hi = _mm512_srli_epi64::<52>(b);

        let p00_lo = _mm512_madd52lo_epu64(zero, a_lo, b_lo);
        let p00_hi = _mm512_madd52hi_epu64(zero, a_lo, b_lo);
        let p10_lo = _mm512_madd52lo_epu64(zero, a_hi, b_lo);
        let p10_hi = _mm512_madd52hi_epu64(zero, a_hi, b_lo);
        let p01_lo = _mm512_madd52lo_epu64(zero, a_lo, b_hi);
        let p01_hi = _mm512_madd52hi_epu64(zero, a_lo, b_hi);
        let p11_lo = _mm512_madd52lo_epu64(zero, a_hi, b_hi);

        let cross_mid = _mm512_add_epi64(_mm512_add_epi64(p00_hi, p10_lo), p01_lo);
        let cross_mid_lo52 = _mm512_and_si512(cross_mid, mask52);
        let cross_mid_carry = _mm512_srli_epi64::<52>(cross_mid);

        let upper_9 = _mm512_add_epi64(_mm512_add_epi64(p10_hi, p01_hi), cross_mid_carry);
        let mask8 = _mm512_set1_epi64(0xFFi64);
        let p11_low8 = _mm512_and_si512(p11_lo, mask8);
        let p11_high = _mm512_srli_epi64::<8>(p11_lo);

        let cm_shifted = _mm512_slli_epi64::<52>(cross_mid_lo52);
        let ab_lo = _mm512_or_si512(p00_lo, cm_shifted);

        let cm_top = _mm512_srli_epi64::<12>(cross_mid_lo52);
        let upper_plus_low = _mm512_add_epi64(upper_9, p11_low8);
        let upper_shifted = _mm512_slli_epi64::<40>(upper_plus_low);
        let p11_hi_shifted = _mm512_slli_epi64::<48>(p11_high);
        let ab_hi = _mm512_add_epi64(_mm512_add_epi64(cm_top, upper_shifted), p11_hi_shifted);

        let m = _mm512_mullo_epi64(ab_lo, q_inv_neg_vec);

        // m * q = (m << 60) - (m << 14) + m, carried as a u128 hi/lo pair per lane.
        let mq_60_hi = _mm512_srli_epi64::<4>(m);
        let mq_60_lo = _mm512_slli_epi64::<60>(m);
        let mq_14_hi = _mm512_srli_epi64::<50>(m);
        let mq_14_lo = _mm512_slli_epi64::<14>(m);

        let diff_lo = _mm512_sub_epi64(mq_60_lo, mq_14_lo);
        let borrow_mask = _mm512_cmplt_epu64_mask(mq_60_lo, mq_14_lo);
        let one = _mm512_set1_epi64(1i64);
        let borrow = _mm512_maskz_mov_epi64(borrow_mask, one);
        let diff_hi = _mm512_sub_epi64(_mm512_sub_epi64(mq_60_hi, mq_14_hi), borrow);

        let mq_lo = _mm512_add_epi64(diff_lo, m);
        let carry_mask = _mm512_cmplt_epu64_mask(mq_lo, diff_lo);
        let carry = _mm512_maskz_mov_epi64(carry_mask, one);
        let mq_hi = _mm512_add_epi64(diff_hi, carry);

        let sum_lo = _mm512_add_epi64(ab_lo, mq_lo);
        let sum_carry_mask = _mm512_cmplt_epu64_mask(sum_lo, ab_lo);
        let sum_carry = _mm512_maskz_mov_epi64(sum_carry_mask, one);
        let t = _mm512_add_epi64(_mm512_add_epi64(ab_hi, mq_hi), sum_carry);

        let ge_mask = _mm512_cmpge_epu64_mask(t, q_vec);
        _mm512_mask_sub_epi64(t, ge_mask, t, q_vec)
    }

    /// 8 Cooley-Tukey DIT butterflies: `coeffs[j] = lo + v`, `coeffs[j+t] = lo - v`
    /// with `v = solinas_mul(hi, w)`.
    ///
    /// # Safety
    ///
    /// Host MUST have `avx512ifma`; both pointers MUST address 8 contiguous
    /// u64 slots, non-overlapping (stride `t >= 8`).
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub unsafe fn forward_butterfly_solinas_x8(
        lo_ptr: *mut u64,
        hi_ptr: *mut u64,
        w_vec: __m512i,
        q_inv_neg_vec: __m512i,
    ) {
        let q_vec = _mm512_set1_epi64(DEFAULT_Q as i64);
        let lo = _mm512_loadu_si512(lo_ptr.cast());
        let hi = _mm512_loadu_si512(hi_ptr.cast());
        let v = solinas_mont_mul_x8(hi, w_vec, q_inv_neg_vec);
        let sum = _mm512_add_epi64(lo, v);
        let sum_ge = _mm512_cmpge_epu64_mask(sum, q_vec);
        let new_lo = _mm512_mask_sub_epi64(sum, sum_ge, sum, q_vec);
        let borrow_mask = _mm512_cmplt_epu64_mask(lo, v);
        let diff = _mm512_sub_epi64(lo, v);
        let new_hi = _mm512_mask_add_epi64(diff, borrow_mask, diff, q_vec);
        _mm512_storeu_si512(lo_ptr.cast(), new_lo);
        _mm512_storeu_si512(hi_ptr.cast(), new_hi);
    }

    /// 8 Gentleman-Sande butterflies: `lo = u + v`, `hi = solinas_mul(u - v, w)`.
    ///
    /// # Safety
    ///
    /// Same preconditions as [`forward_butterfly_solinas_x8`].
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    pub unsafe fn inverse_butterfly_solinas_x8(
        lo_ptr: *mut u64,
        hi_ptr: *mut u64,
        w_vec: __m512i,
        q_inv_neg_vec: __m512i,
    ) {
        let q_vec = _mm512_set1_epi64(DEFAULT_Q as i64);
        let u = _mm512_loadu_si512(lo_ptr.cast());
        let v = _mm512_loadu_si512(hi_ptr.cast());
        let sum = _mm512_add_epi64(u, v);
        let sum_ge = _mm512_cmpge_epu64_mask(sum, q_vec);
        let new_lo = _mm512_mask_sub_epi64(sum, sum_ge, sum, q_vec);
        let borrow_mask = _mm512_cmplt_epu64_mask(u, v);
        let raw_diff = _mm512_sub_epi64(u, v);
        let diff_mod = _mm512_mask_add_epi64(raw_diff, borrow_mask, raw_diff, q_vec);
        let new_hi = solinas_mont_mul_x8(diff_mod, w_vec, q_inv_neg_vec);
        _mm512_storeu_si512(lo_ptr.cast(), new_lo);
        _mm512_storeu_si512(hi_ptr.cast(), new_hi);
    }

    /// [`super::pointwise_solinas_mont_mul`] across 8 lanes; lengths must be multiples of 8.
    ///
    /// # Safety
    ///
    /// Host MUST have `avx512ifma`.
    #[target_feature(enable = "avx512f,avx512ifma")]
    pub unsafe fn pointwise_solinas_mont_mul_x8(
        a: &[u64],
        b: &[u64],
        result: &mut [u64],
        q_inv_neg: u64,
    ) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());
        assert_eq!(a.len() % 8, 0, "array length must be multiple of 8");

        let q_inv_neg_vec = _mm512_set1_epi64(q_inv_neg as i64);

        for i in (0..a.len()).step_by(8) {
            let av = _mm512_loadu_si512(a[i..].as_ptr().cast());
            let bv = _mm512_loadu_si512(b[i..].as_ptr().cast());
            let rv = solinas_mont_mul_x8(av, bv, q_inv_neg_vec);
            _mm512_storeu_si512(result[i..].as_mut_ptr().cast(), rv);
        }
    }

    /// `acc[i] = (acc[i] + a[i] * b[i] * R^{-1}) mod DEFAULT_Q` across 8 lanes.
    ///
    /// Both addends are `< q < 2^60`, so one conditional subtract restores `[0, q)`.
    ///
    /// # Safety
    ///
    /// Host MUST have `avx512ifma`; `acc`, `a`, `b` MUST each address `len` u64
    /// with `len` a multiple of 8, must not alias, and must hold Montgomery-form
    /// values in `[0, DEFAULT_Q)`.
    #[target_feature(enable = "avx512f,avx512ifma")]
    pub unsafe fn solinas_mont_mul_acc_x8(
        acc: *mut u64,
        a: *const u64,
        b: *const u64,
        len: usize,
        q_inv_neg: u64,
    ) {
        debug_assert_eq!(len % 8, 0, "len must be a multiple of 8");

        let q_vec = _mm512_set1_epi64(DEFAULT_Q as i64);
        let q_inv_neg_vec = _mm512_set1_epi64(q_inv_neg as i64);

        let mut i = 0usize;
        while i < len {
            let av = _mm512_loadu_si512(a.add(i).cast());
            let bv = _mm512_loadu_si512(b.add(i).cast());
            let accv = _mm512_loadu_si512(acc.add(i).cast());

            let prod = solinas_mont_mul_x8(av, bv, q_inv_neg_vec);

            let sum = _mm512_add_epi64(accv, prod);

            let ge_mask = _mm512_cmpge_epu64_mask(sum, q_vec);
            let reduced = _mm512_mask_sub_epi64(sum, ge_mask, sum, q_vec);

            _mm512_storeu_si512(acc.add(i).cast(), reduced);
            i += 8;
        }
    }
}

/// Shape-compatible stub so non-x86_64 targets still compile the dispatch site.
#[cfg(not(target_arch = "x86_64"))]
#[allow(
    clippy::panic,
    reason = "abort is the contract: solinas_mont_mul_default_q is the scalar path off x86_64"
)]
pub mod avx512_ifma {
    /// # Safety
    ///
    /// Unreachable: callers MUST gate on `avx512ifma`, false off x86_64. Aborts if called.
    #[allow(unused_variables)]
    pub unsafe fn pointwise_solinas_mont_mul_x8(
        a: &[u64],
        b: &[u64],
        result: &mut [u64],
        q_inv_neg: u64,
    ) {
        panic!("AVX-512-IFMA not available on this target architecture");
    }

    /// # Safety
    ///
    /// Unreachable; pointers are never dereferenced. See [`pointwise_solinas_mont_mul_x8`].
    #[allow(unused_variables)]
    pub unsafe fn solinas_mont_mul_acc_x8(
        acc: *mut u64,
        a: *const u64,
        b: *const u64,
        len: usize,
        q_inv_neg: u64,
    ) {
        panic!("AVX-512-IFMA not available on this target architecture");
    }
}
