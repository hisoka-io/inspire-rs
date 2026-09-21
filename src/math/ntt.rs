//! Radix-2 NTT for negacyclic convolution over R_q = Z_q\[X\]/(X^d + 1).
//!
//! Negacyclic wrap needs a primitive 2n-th root psi with psi^n = -1, so q must
//! satisfy q = 1 mod 2n.
//!
//! ```
//! use raven_inspire::math::ntt::NttContext;
//!
//! let ctx = NttContext::with_default_q(256);
//! let mut coeffs = vec![1u64; 256];
//! ctx.forward(&mut coeffs);
//! ctx.inverse(&mut coeffs);
//! assert_eq!(coeffs[0], 1);
//! ```

use super::mod_q::DEFAULT_Q;
use super::primality::is_prime;

/// Far above any prime's least quadratic non-residue, where the root search ends: 11 for
/// the moduli shipped here, O(ln^2 q) under ERH. Reached only if the primality assert is lost.
const ROOT_SEARCH_BOUND: u64 = 1 << 16;

/// Twiddle ladders and Montgomery constants for one (dimension, moduli) pair.
///
/// # Example
///
/// ```
/// use raven_inspire::math::ntt::NttContext;
///
/// let ctx = NttContext::with_default_q(2048);
/// assert_eq!(ctx.dimension(), 2048);
/// ```
#[derive(Clone)]
pub struct NttContext {
    n: usize,
    moduli: Vec<u64>,
    q_inv_neg: Vec<u64>,
    r_squared: Vec<u64>,
    psi_powers: Vec<Vec<u64>>,
    psi_inv_powers: Vec<Vec<u64>>,
    n_inv: Vec<u64>,
    // Shoup twiddles are standard-form, so the Shoup butterflies skip the
    // Montgomery conversions at the NTT boundary entirely.
    psi_powers_std: Vec<Vec<u64>>,
    psi_powers_shoup: Vec<Vec<u64>>,
    psi_inv_powers_std: Vec<Vec<u64>>,
    psi_inv_powers_shoup: Vec<Vec<u64>>,
    n_inv_std: Vec<u64>,
    n_inv_shoup: Vec<u64>,
}

impl NttContext {
    /// # Panics
    ///
    /// If `n` is not a power of two, `q != 1 mod 2n`, or `q` is composite.
    ///
    /// # Example
    ///
    /// ```
    /// use raven_inspire::math::ntt::NttContext;
    /// use raven_inspire::math::mod_q::DEFAULT_Q;
    ///
    /// let ctx = NttContext::new(2048, DEFAULT_Q);
    /// assert_eq!(ctx.dimension(), 2048);
    /// ```
    pub fn new(n: usize, q: u64) -> Self {
        Self::with_moduli(n, &[q])
    }

    /// # Panics
    ///
    /// If `n` is not a power of two, or any modulus is composite or violates `q = 1 mod 2n`.
    pub fn with_moduli(n: usize, moduli: &[u64]) -> Self {
        assert!(n.is_power_of_two(), "n must be a power of two");
        assert!(!moduli.is_empty(), "moduli must be non-empty");

        let mut q_inv_neg = Vec::with_capacity(moduli.len());
        let mut r_squared = Vec::with_capacity(moduli.len());
        let mut psi_powers = Vec::with_capacity(moduli.len());
        let mut psi_inv_powers = Vec::with_capacity(moduli.len());
        let mut n_inv = Vec::with_capacity(moduli.len());
        let mut psi_powers_std = Vec::with_capacity(moduli.len());
        let mut psi_powers_shoup = Vec::with_capacity(moduli.len());
        let mut psi_inv_powers_std = Vec::with_capacity(moduli.len());
        let mut psi_inv_powers_shoup = Vec::with_capacity(moduli.len());
        let mut n_inv_std = Vec::with_capacity(moduli.len());
        let mut n_inv_shoup = Vec::with_capacity(moduli.len());

        for &q in moduli {
            assert!(q % (2 * n as u64) == 1, "q must be 1 mod 2n");
            // On a composite the root search never ends, or accepts a square root of 1 that
            // is not -1 and builds a transform that round-trips while multiplying wrongly.
            assert!(is_prime(q), "modulus {q} must be prime for the NTT");

            let q_inv = Self::compute_q_inv_neg(q);
            let r2 = Self::compute_r_squared(q);

            let psi = Self::find_primitive_root(2 * n as u64, q);
            let psi_mont = Self::to_montgomery(psi, q, r2, q_inv);

            let psi_pow = Self::compute_twiddle_factors(n, psi_mont, q, q_inv, r2);

            let psi_inv = Self::mod_pow(psi, q - 2, q);
            let psi_inv_mont = Self::to_montgomery(psi_inv, q, r2, q_inv);
            let psi_inv_pow = Self::compute_twiddle_factors(n, psi_inv_mont, q, q_inv, r2);

            let n_inv_val = Self::mod_pow(n as u64, q - 2, q);
            let n_inv_mont = Self::to_montgomery(n_inv_val, q, r2, q_inv);

            // Same bit-reversed order as the Montgomery ladder; the butterflies index identically.
            let psi_pow_std = Self::compute_twiddle_factors_std(n, psi, q);
            let psi_pow_shoup = Self::compute_shoup_twins(&psi_pow_std, q);
            let psi_inv_pow_std = Self::compute_twiddle_factors_std(n, psi_inv, q);
            let psi_inv_pow_shoup = Self::compute_shoup_twins(&psi_inv_pow_std, q);
            let n_inv_shoup_val = Self::shoup_precompute(n_inv_val, q);

            q_inv_neg.push(q_inv);
            r_squared.push(r2);
            psi_powers.push(psi_pow);
            psi_inv_powers.push(psi_inv_pow);
            n_inv.push(n_inv_mont);
            psi_powers_std.push(psi_pow_std);
            psi_powers_shoup.push(psi_pow_shoup);
            psi_inv_powers_std.push(psi_inv_pow_std);
            psi_inv_powers_shoup.push(psi_inv_pow_shoup);
            n_inv_std.push(n_inv_val);
            n_inv_shoup.push(n_inv_shoup_val);
        }

        Self {
            n,
            moduli: moduli.to_vec(),
            q_inv_neg,
            r_squared,
            psi_powers,
            psi_inv_powers,
            n_inv,
            psi_powers_std,
            psi_powers_shoup,
            psi_inv_powers_std,
            psi_inv_powers_shoup,
            n_inv_std,
            n_inv_shoup,
        }
    }

    /// [`Self::new`] against `DEFAULT_Q`, which supports `n` up to 8192.
    pub fn with_default_q(n: usize) -> Self {
        Self::new(n, DEFAULT_Q)
    }

    /// Ring dimension n.
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Product of the CRT moduli.
    pub fn modulus(&self) -> u64 {
        self.moduli.iter().copied().fold(1u64, u64::saturating_mul)
    }

    /// The CRT moduli.
    pub fn moduli(&self) -> &[u64] {
        &self.moduli
    }

    /// Number of CRT moduli.
    pub fn crt_count(&self) -> usize {
        self.moduli.len()
    }

    /// `-q^{-1} mod 2^64` for CRT limb `idx`. Not stable public API.
    #[doc(hidden)]
    pub fn q_inv_neg_for_test(&self, idx: usize) -> u64 {
        self.q_inv_neg[idx]
    }

    /// Gates the Solinas-REDC path; its output is byte-identical to the classical one.
    #[inline]
    fn is_solinas_default_q(&self) -> bool {
        self.moduli.len() == 1 && self.moduli[0] == super::mod_q::DEFAULT_Q
    }

    /// `q_inv_neg` of the single Solinas limb, `None` off that path.
    #[cfg(all(feature = "simd-packing-offline", target_arch = "x86_64"))]
    #[inline]
    pub(crate) fn solinas_q_inv_neg(&self) -> Option<u64> {
        if self.is_solinas_default_q() {
            Some(self.q_inv_neg[0])
        } else {
            None
        }
    }

    /// Cooley-Tukey forward NTT in place; lifts standard-form input into Montgomery form.
    ///
    /// # Panics
    ///
    /// If `coeffs.len() != n * crt_count`.
    pub fn forward(&self, coeffs: &mut [u64]) {
        assert_eq!(
            coeffs.len(),
            self.n * self.crt_count(),
            "Input length must match dimension * crt_count"
        );

        if self.is_solinas_default_q() {
            return self.forward_solinas(coeffs);
        }

        for (idx, _) in self.moduli.iter().enumerate() {
            let start = idx * self.n;
            let end = start + self.n;

            for c in &mut coeffs[start..end] {
                *c = Self::to_montgomery_at(
                    *c,
                    self.moduli[idx],
                    self.r_squared[idx],
                    self.q_inv_neg[idx],
                );
            }

            self.forward_inplace_at(&mut coeffs[start..end], idx);
        }
    }

    /// [`Self::forward`] for input already in Montgomery form.
    pub fn forward_inplace(&self, coeffs: &mut [u64]) {
        assert_eq!(
            coeffs.len(),
            self.n * self.crt_count(),
            "Input length must match dimension * crt_count"
        );
        if self.is_solinas_default_q() {
            self.forward_inplace_solinas_at(coeffs, 0);
            return;
        }
        for (idx, _) in self.moduli.iter().enumerate() {
            let start = idx * self.n;
            let end = start + self.n;
            self.forward_inplace_at(&mut coeffs[start..end], idx);
        }
    }

    fn forward_inplace_at(&self, coeffs: &mut [u64], idx: usize) {
        let n = self.n;
        let q = self.moduli[idx];
        let psi_powers = &self.psi_powers[idx];

        let mut t = n;
        let mut m = 1;

        while m < n {
            t >>= 1;
            for i in 0..m {
                let j1 = 2 * i * t;
                let j2 = j1 + t;
                let w = psi_powers[m + i];

                for j in j1..j2 {
                    let u = coeffs[j];
                    let v = self.montgomery_mul_at(coeffs[j + t], w, idx);

                    coeffs[j] = if u + v >= q { u + v - q } else { u + v };
                    coeffs[j + t] = if u >= v { u - v } else { q - v + u };
                }
            }
            m <<= 1;
        }
    }

    /// Gentleman-Sande inverse NTT in place; strips Montgomery form on output.
    ///
    /// # Panics
    ///
    /// If `coeffs.len() != n * crt_count`.
    pub fn inverse(&self, coeffs: &mut [u64]) {
        assert_eq!(
            coeffs.len(),
            self.n * self.crt_count(),
            "Input length must match dimension * crt_count"
        );

        if self.is_solinas_default_q() {
            return self.inverse_solinas(coeffs);
        }

        self.inverse_inplace(coeffs);

        for (idx, _) in self.moduli.iter().enumerate() {
            let start = idx * self.n;
            let end = start + self.n;
            for c in &mut coeffs[start..end] {
                *c = self.montgomery_mul_at(*c, 1, idx);
            }
        }
    }

    /// [`Self::inverse`] leaving output in Montgomery form.
    pub fn inverse_inplace(&self, coeffs: &mut [u64]) {
        assert_eq!(
            coeffs.len(),
            self.n * self.crt_count(),
            "Input length must match dimension * crt_count"
        );
        if self.is_solinas_default_q() {
            self.inverse_inplace_solinas_at(coeffs, 0);
            return;
        }
        for (idx, _) in self.moduli.iter().enumerate() {
            let start = idx * self.n;
            let end = start + self.n;
            self.inverse_inplace_at(&mut coeffs[start..end], idx);
        }
    }

    fn inverse_inplace_at(&self, coeffs: &mut [u64], idx: usize) {
        let n = self.n;
        let q = self.moduli[idx];
        let psi_inv_powers = &self.psi_inv_powers[idx];

        let mut t = 1;
        let mut m = n;

        while m > 1 {
            m >>= 1;
            let j1 = 0;
            for i in 0..m {
                let j2 = j1 + i * 2 * t;
                let w = psi_inv_powers[m + i];

                for j in j2..(j2 + t) {
                    let u = coeffs[j];
                    let v = coeffs[j + t];

                    coeffs[j] = if u + v >= q { u + v - q } else { u + v };
                    let diff = if u >= v { u - v } else { q - v + u };
                    coeffs[j + t] = self.montgomery_mul_at(diff, w, idx);
                }
            }
            t <<= 1;
        }

        for c in coeffs.iter_mut() {
            *c = self.montgomery_mul_at(*c, self.n_inv[idx], idx);
        }
    }

    /// NTT-domain product; both operands must be in Montgomery form.
    ///
    /// # Panics
    ///
    /// If any length differs from `n * crt_count`.
    pub fn pointwise_mul(&self, a: &[u64], b: &[u64], result: &mut [u64]) {
        assert_eq!(
            a.len(),
            self.n * self.crt_count(),
            "Input length must match dimension * crt_count"
        );
        assert_eq!(
            b.len(),
            self.n * self.crt_count(),
            "Input length must match dimension * crt_count"
        );
        assert_eq!(
            result.len(),
            self.n * self.crt_count(),
            "Input length must match dimension * crt_count"
        );

        if self.is_solinas_default_q() {
            let q_inv_neg = self.q_inv_neg[0];
            for i in 0..self.n {
                result[i] = super::solinas_redc::solinas_mont_mul_default_q(a[i], b[i], q_inv_neg);
            }
            return;
        }

        for idx in 0..self.crt_count() {
            let start = idx * self.n;
            for i in 0..self.n {
                result[start + i] = self.montgomery_mul_at(a[start + i], b[start + i], idx);
            }
        }
    }

    /// `(a * b) mod q` on limb 0, Montgomery in and out.
    #[inline]
    pub fn pointwise_mul_single(&self, a: u64, b: u64) -> u64 {
        self.montgomery_mul_at(a, b, 0)
    }

    /// `(a * b) mod q` on limb `idx`, Montgomery in and out.
    #[inline]
    pub fn pointwise_mul_single_at(&self, a: u64, b: u64, idx: usize) -> u64 {
        if self.is_solinas_default_q() {
            return super::solinas_redc::solinas_mont_mul_default_q(a, b, self.q_inv_neg[0]);
        }
        self.montgomery_mul_at(a, b, idx)
    }

    /// Forward NTT with standard-form input and output, no Montgomery boundary.
    ///
    /// # Panics
    ///
    /// If `coeffs.len() != n * crt_count`.
    pub fn forward_shoup(&self, coeffs: &mut [u64]) {
        assert_eq!(
            coeffs.len(),
            self.n * self.crt_count(),
            "Input length must match dimension * crt_count"
        );
        for idx in 0..self.moduli.len() {
            let start = idx * self.n;
            let end = start + self.n;
            self.forward_inplace_shoup_at(&mut coeffs[start..end], idx);
        }
    }

    /// Inverse NTT with standard-form output, already scaled by `n^{-1}` and in `[0, q)`.
    ///
    /// # Panics
    ///
    /// If `coeffs.len() != n * crt_count`.
    pub fn inverse_shoup(&self, coeffs: &mut [u64]) {
        assert_eq!(
            coeffs.len(),
            self.n * self.crt_count(),
            "Input length must match dimension * crt_count"
        );
        for idx in 0..self.moduli.len() {
            let start = idx * self.n;
            let end = start + self.n;
            self.inverse_inplace_shoup_at(&mut coeffs[start..end], idx);
        }
    }

    fn forward_inplace_shoup_at(&self, coeffs: &mut [u64], idx: usize) {
        let n = self.n;
        let q = self.moduli[idx];
        let psi = &self.psi_powers_std[idx];
        let psi_shoup = &self.psi_powers_shoup[idx];

        let mut t = n;
        let mut m = 1;

        while m < n {
            t >>= 1;
            for i in 0..m {
                let j1 = 2 * i * t;
                let j2 = j1 + t;
                let w = psi[m + i];
                let w_shoup = psi_shoup[m + i];

                for j in j1..j2 {
                    let u = coeffs[j];
                    let v = Self::shoup_mul_at(coeffs[j + t], w, w_shoup, q);

                    coeffs[j] = if u + v >= q { u + v - q } else { u + v };
                    coeffs[j + t] = if u >= v { u - v } else { q - v + u };
                }
            }
            m <<= 1;
        }
    }

    fn inverse_inplace_shoup_at(&self, coeffs: &mut [u64], idx: usize) {
        let n = self.n;
        let q = self.moduli[idx];
        let psi_inv = &self.psi_inv_powers_std[idx];
        let psi_inv_shoup = &self.psi_inv_powers_shoup[idx];

        let mut t = 1;
        let mut m = n;

        while m > 1 {
            m >>= 1;
            let j1 = 0;
            for i in 0..m {
                let j2 = j1 + i * 2 * t;
                let w = psi_inv[m + i];
                let w_shoup = psi_inv_shoup[m + i];

                for j in j2..(j2 + t) {
                    let u = coeffs[j];
                    let v = coeffs[j + t];

                    coeffs[j] = if u + v >= q { u + v - q } else { u + v };
                    let diff = if u >= v { u - v } else { q - v + u };
                    coeffs[j + t] = Self::shoup_mul_at(diff, w, w_shoup, q);
                }
            }
            t <<= 1;
        }

        let n_inv = self.n_inv_std[idx];
        let n_inv_shoup = self.n_inv_shoup[idx];
        for c in coeffs.iter_mut() {
            *c = Self::shoup_mul_at(*c, n_inv, n_inv_shoup, q);
        }
    }

    /// NTT-domain product in standard form; caller supplies
    /// `b_shoup[i] = shoup_precompute(b[i], q_of_limb)`.
    ///
    /// # Panics
    ///
    /// If any length differs from `n * crt_count`.
    pub fn pointwise_mul_shoup(&self, a: &[u64], b: &[u64], b_shoup: &[u64], result: &mut [u64]) {
        let total = self.n * self.crt_count();
        assert_eq!(a.len(), total, "a length mismatch");
        assert_eq!(b.len(), total, "b length mismatch");
        assert_eq!(b_shoup.len(), total, "b_shoup length mismatch");
        assert_eq!(result.len(), total, "result length mismatch");

        for idx in 0..self.crt_count() {
            let start = idx * self.n;
            let q = self.moduli[idx];
            for i in 0..self.n {
                result[start + i] =
                    Self::shoup_mul_at(a[start + i], b[start + i], b_shoup[start + i], q);
            }
        }
    }

    /// [`Self::forward`] over Solinas-REDC; requires single-prime DEFAULT_Q.
    ///
    /// # Panics
    ///
    /// If the context is not single-prime DEFAULT_Q or `coeffs.len() != n`.
    pub fn forward_solinas(&self, coeffs: &mut [u64]) {
        assert_eq!(
            self.moduli.len(),
            1,
            "Solinas path is single-prime DEFAULT_Q only"
        );
        assert_eq!(
            self.moduli[0],
            super::mod_q::DEFAULT_Q,
            "Solinas path requires DEFAULT_Q"
        );
        assert_eq!(coeffs.len(), self.n, "Input length must match dimension");

        let q = self.moduli[0];
        let r_squared = self.r_squared[0];
        let q_inv_neg = self.q_inv_neg[0];

        for c in coeffs.iter_mut() {
            *c = Self::to_montgomery(*c, q, r_squared, q_inv_neg);
        }

        self.forward_inplace_solinas_at(coeffs, 0);
    }

    /// [`Self::inverse`] over Solinas-REDC; requires single-prime DEFAULT_Q.
    pub fn inverse_solinas(&self, coeffs: &mut [u64]) {
        assert_eq!(
            self.moduli.len(),
            1,
            "Solinas path is single-prime DEFAULT_Q only"
        );
        assert_eq!(
            self.moduli[0],
            super::mod_q::DEFAULT_Q,
            "Solinas path requires DEFAULT_Q"
        );
        assert_eq!(coeffs.len(), self.n, "Input length must match dimension");

        self.inverse_inplace_solinas_at(coeffs, 0);

        for c in coeffs.iter_mut() {
            *c = self.montgomery_mul_at(*c, 1, 0);
        }
    }

    fn forward_inplace_solinas_at(&self, coeffs: &mut [u64], idx: usize) {
        let n = self.n;
        let q = self.moduli[idx];
        let q_inv_neg = self.q_inv_neg[idx];
        let psi_powers = &self.psi_powers[idx];

        // Scalar, not SIMD: hand-vectorized and unrolled butterflies both measured
        // flat-to-regressed here; LLVM already pipelines the REDC + cmov chain.
        let mut t = n;
        let mut m = 1;
        while m < n {
            t >>= 1;
            for i in 0..m {
                let j1 = 2 * i * t;
                let j2 = j1 + t;
                let w = psi_powers[m + i];

                for j in j1..j2 {
                    let u = coeffs[j];
                    let v = super::solinas_redc::solinas_mont_mul_default_q(
                        coeffs[j + t],
                        w,
                        q_inv_neg,
                    );
                    coeffs[j] = if u + v >= q { u + v - q } else { u + v };
                    coeffs[j + t] = if u >= v { u - v } else { q - v + u };
                }
            }
            m <<= 1;
        }
    }

    fn inverse_inplace_solinas_at(&self, coeffs: &mut [u64], idx: usize) {
        let n = self.n;
        let q = self.moduli[idx];
        let q_inv_neg = self.q_inv_neg[idx];
        let psi_inv_powers = &self.psi_inv_powers[idx];

        let mut t = 1;
        let mut m = n;
        while m > 1 {
            m >>= 1;
            let j1 = 0;
            for i in 0..m {
                let j2 = j1 + i * 2 * t;
                let w = psi_inv_powers[m + i];

                for j in j2..(j2 + t) {
                    let u = coeffs[j];
                    let v = coeffs[j + t];
                    coeffs[j] = if u + v >= q { u + v - q } else { u + v };
                    let diff = if u >= v { u - v } else { q - v + u };
                    coeffs[j + t] =
                        super::solinas_redc::solinas_mont_mul_default_q(diff, w, q_inv_neg);
                }
            }
            t <<= 1;
        }

        for c in coeffs.iter_mut() {
            *c = super::solinas_redc::solinas_mont_mul_default_q(*c, self.n_inv[idx], q_inv_neg);
        }
    }

    /// [`Self::pointwise_mul`] over Solinas-REDC; requires single-prime DEFAULT_Q.
    pub fn pointwise_mul_solinas(&self, a: &[u64], b: &[u64], result: &mut [u64]) {
        assert_eq!(
            self.moduli.len(),
            1,
            "Solinas path is single-prime DEFAULT_Q only"
        );
        assert_eq!(
            self.moduli[0],
            super::mod_q::DEFAULT_Q,
            "Solinas path requires DEFAULT_Q"
        );
        assert_eq!(a.len(), self.n);
        assert_eq!(b.len(), self.n);
        assert_eq!(result.len(), self.n);
        let q_inv_neg = self.q_inv_neg[0];
        for i in 0..self.n {
            result[i] = super::solinas_redc::solinas_mont_mul_default_q(a[i], b[i], q_inv_neg);
        }
    }

    /// Shoup twins for a standard-form operand in `Poly::coeffs()` layout.
    pub fn shoup_precompute_vec(&self, b: &[u64]) -> Vec<u64> {
        let total = self.n * self.crt_count();
        assert_eq!(b.len(), total, "b length mismatch");
        let mut out = Vec::with_capacity(total);
        for idx in 0..self.crt_count() {
            let start = idx * self.n;
            let q = self.moduli[idx];
            for i in 0..self.n {
                out.push(Self::shoup_precompute(b[start + i], q));
            }
        }
        out
    }

    /// Lifts a standard-form value on limb 0 into Montgomery form.
    pub fn to_mont(&self, a: u64) -> u64 {
        Self::to_montgomery(a, self.moduli[0], self.r_squared[0], self.q_inv_neg[0])
    }

    /// Strips Montgomery form on limb 0.
    pub fn from_mont(&self, a: u64) -> u64 {
        self.montgomery_mul_at(a, 1, 0)
    }

    #[inline]
    fn montgomery_mul_at(&self, a: u64, b: u64, idx: usize) -> u64 {
        let q = self.moduli[idx];
        let q_inv_neg = self.q_inv_neg[idx];
        let ab = (a as u128) * (b as u128);
        let m = ((ab as u64).wrapping_mul(q_inv_neg)) as u128;
        let t = ((ab + m * (q as u128)) >> 64) as u64;
        if t >= q {
            t - q
        } else {
            t
        }
    }

    fn to_montgomery(a: u64, q: u64, r_squared: u64, q_inv_neg: u64) -> u64 {
        let ab = (a as u128) * (r_squared as u128);
        let m = ((ab as u64).wrapping_mul(q_inv_neg)) as u128;
        let t = ((ab + m * (q as u128)) >> 64) as u64;
        if t >= q {
            t - q
        } else {
            t
        }
    }

    #[inline]
    fn to_montgomery_at(a: u64, q: u64, r_squared: u64, q_inv_neg: u64) -> u64 {
        Self::to_montgomery(a, q, r_squared, q_inv_neg)
    }

    fn compute_q_inv_neg(q: u64) -> u64 {
        let mut y: u64 = 1;
        for i in 1..64 {
            let yi = y.wrapping_mul(q) & (1u64 << i);
            y |= yi;
        }
        y.wrapping_neg()
    }

    fn compute_r_squared(q: u64) -> u64 {
        let r_mod_q = (1u128 << 64) % (q as u128);
        ((r_mod_q * r_mod_q) % (q as u128)) as u64
    }

    fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
        let mut result = 1u64;
        base %= m;
        while exp > 0 {
            if exp & 1 == 1 {
                result = ((result as u128 * base as u128) % m as u128) as u64;
            }
            exp >>= 1;
            base = ((base as u128 * base as u128) % m as u128) as u64;
        }
        result
    }

    /// Primitive n-th root of unity mod a prime q, for n a power of two dividing q - 1.
    ///
    /// `g^((q-1)/n)` has order exactly n iff g is a quadratic non-residue, so the search
    /// returns at q's least one.
    #[allow(
        clippy::panic,
        reason = "unreachable: with_moduli asserts q prime and q = 1 mod n, and half the residues of a prime are non-residues"
    )]
    fn find_primitive_root(n: u64, q: u64) -> u64 {
        let exp = (q - 1) / n;

        for g in 2..q.min(ROOT_SEARCH_BOUND) {
            let candidate = Self::mod_pow(g, exp, q);
            if Self::mod_pow(candidate, n, q) == 1 && Self::mod_pow(candidate, n / 2, q) != 1 {
                return candidate;
            }
        }
        panic!("no primitive {n}-th root mod {q} among the first {ROOT_SEARCH_BOUND} candidates");
    }

    /// Twiddle ladder in plain mod-q arithmetic, indexed identically to
    /// [`Self::compute_twiddle_factors`] so both paths share bit-reversed order.
    fn compute_twiddle_factors_std(n: usize, psi: u64, q: u64) -> Vec<u64> {
        let mut factors = vec![0u64; n];
        factors[1] = 1;

        for m in 1..n {
            if m.is_power_of_two() {
                let exp = (n / (2 * m)) as u64;
                factors[m] = Self::mod_pow(psi, exp, q);
            } else {
                let prev_idx = m & (m - 1);
                let step_idx = m & (!m + 1);
                let ab = (factors[prev_idx] as u128) * (factors[step_idx] as u128);
                factors[m] = (ab % q as u128) as u64;
            }
        }

        factors
    }

    /// `floor(b * 2^64 / q)` for each entry of a twiddle ladder.
    fn compute_shoup_twins(factors: &[u64], q: u64) -> Vec<u64> {
        factors
            .iter()
            .map(|&b| Self::shoup_precompute(b, q))
            .collect()
    }

    /// `floor(b * 2^64 / q)`, the Shoup twin of a fixed multiplicand.
    #[inline]
    fn shoup_precompute(b: u64, q: u64) -> u64 {
        (((b as u128) << 64) / (q as u128)) as u64
    }

    /// `a * b mod q` in standard form, requiring `a, b` in `[0, q)` and
    /// `b_shoup = shoup_precompute(b, q)`.
    #[inline]
    fn shoup_mul_at(a: u64, b: u64, b_shoup: u64, q: u64) -> u64 {
        let q_est = ((a as u128) * (b_shoup as u128)) >> 64;
        let r = ((a as u128)
            .wrapping_mul(b as u128)
            .wrapping_sub(q_est.wrapping_mul(q as u128))) as u64;
        if r >= q {
            r - q
        } else {
            r
        }
    }

    fn compute_twiddle_factors(
        n: usize,
        psi: u64,
        q: u64,
        q_inv_neg: u64,
        r_squared: u64,
    ) -> Vec<u64> {
        let mut factors = vec![0u64; n];

        factors[1] = Self::to_montgomery(1, q, r_squared, q_inv_neg);

        for m in 1..n {
            if m.is_power_of_two() {
                let exp = n / (2 * m);

                let mut pow = Self::to_montgomery(1, q, r_squared, q_inv_neg);
                for _ in 0..exp {
                    let ab = (pow as u128) * (psi as u128);
                    let mm = ((ab as u64).wrapping_mul(q_inv_neg)) as u128;
                    pow = ((ab + mm * (q as u128)) >> 64) as u64;
                    if pow >= q {
                        pow -= q;
                    }
                }
                factors[m] = pow;
            } else {
                let prev_idx = m & (m - 1);
                let step_idx = m & (!m + 1);

                let ab = (factors[prev_idx] as u128) * (factors[step_idx] as u128);
                let mm = ((ab as u64).wrapping_mul(q_inv_neg)) as u128;
                let t = ((ab + mm * (q as u128)) >> 64) as u64;
                factors[m] = if t >= q { t - q } else { t };
            }
        }

        factors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntt_inverse_roundtrip_small() {
        let n = 16;
        let ctx = NttContext::with_default_q(n);

        let original: Vec<u64> = (0..n as u64).collect();
        let mut coeffs = original.clone();

        ctx.forward(&mut coeffs);
        ctx.inverse(&mut coeffs);

        assert_eq!(coeffs, original);
    }

    #[test]
    fn test_ntt_inverse_roundtrip_1024() {
        let n = 1024;
        let ctx = NttContext::with_default_q(n);

        let original: Vec<u64> = (0..n as u64).collect();
        let mut coeffs = original.clone();

        ctx.forward(&mut coeffs);
        ctx.inverse(&mut coeffs);

        assert_eq!(coeffs, original);
    }

    #[test]
    fn test_ntt_inverse_roundtrip_2048() {
        let n = 2048;
        let ctx = NttContext::with_default_q(n);

        let original: Vec<u64> = (0..n as u64).map(|i| i * 1000 % DEFAULT_Q).collect();
        let mut coeffs = original.clone();

        ctx.forward(&mut coeffs);
        ctx.inverse(&mut coeffs);

        assert_eq!(coeffs, original);
    }

    #[test]
    fn test_ntt_inverse_roundtrip_4096() {
        let n = 4096;
        let ctx = NttContext::with_default_q(n);

        let original: Vec<u64> = (0..n as u64).map(|i| (i * 12345) % DEFAULT_Q).collect();
        let mut coeffs = original.clone();

        ctx.forward(&mut coeffs);
        ctx.inverse(&mut coeffs);

        assert_eq!(coeffs, original);
    }

    #[test]
    fn test_ntt_zero_polynomial() {
        let n = 256;
        let ctx = NttContext::with_default_q(n);

        let mut coeffs = vec![0u64; n];
        ctx.forward(&mut coeffs);

        assert!(coeffs.iter().all(|&c| c == 0));

        ctx.inverse(&mut coeffs);
        assert!(coeffs.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_ntt_constant_polynomial() {
        let n = 256;
        let ctx = NttContext::with_default_q(n);

        let mut coeffs = vec![0u64; n];
        coeffs[0] = 42;
        let original = coeffs.clone();

        ctx.forward(&mut coeffs);
        ctx.inverse(&mut coeffs);

        assert_eq!(coeffs, original);
    }

    #[test]
    fn test_pointwise_multiplication() {
        let n = 256;
        let ctx = NttContext::with_default_q(n);

        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        a[0] = 1;
        b[0] = 1;

        ctx.forward(&mut a);
        ctx.forward(&mut b);

        let mut result = vec![0u64; n];
        ctx.pointwise_mul(&a, &b, &mut result);

        ctx.inverse(&mut result);

        assert_eq!(result[0], 1);
        assert!(result[1..].iter().all(|&c| c == 0));
    }

    #[test]
    fn test_negacyclic_convolution() {
        let n = 256;
        let q = DEFAULT_Q;
        let ctx = NttContext::with_default_q(n);

        let mut a = vec![0u64; n];
        a[1] = 1;

        let mut b = vec![0u64; n];
        b[n - 1] = 1;

        ctx.forward(&mut a);
        ctx.forward(&mut b);

        let mut result = vec![0u64; n];
        ctx.pointwise_mul(&a, &b, &mut result);

        ctx.inverse(&mut result);

        // x * x^(n-1) wraps negacyclically to -1.
        assert_eq!(result[0], q - 1);
        assert!(result[1..].iter().all(|&c| c == 0));
    }

    #[test]
    fn test_linearity() {
        let n = 256;
        let ctx = NttContext::with_default_q(n);
        let q = DEFAULT_Q;

        let a: Vec<u64> = (0..n as u64).collect();
        let b: Vec<u64> = (0..n as u64).map(|i| (i * 2) % q).collect();

        let mut a_ntt = a.clone();
        let mut b_ntt = b.clone();
        ctx.forward(&mut a_ntt);
        ctx.forward(&mut b_ntt);

        let mut sum: Vec<u64> = a.iter().zip(b.iter()).map(|(&x, &y)| (x + y) % q).collect();
        ctx.forward(&mut sum);

        for i in 0..n {
            let expected = (a_ntt[i] + b_ntt[i]) % q;
            assert_eq!(sum[i], expected);
        }
    }
}
