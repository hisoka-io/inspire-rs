//! RGSW ciphertext and gadget types.

use crate::math::gaussian::{os_seeded_chacha, os_seeded_chacha_or_abort, EntropyUnavailable};
use crate::math::{GaussianSampler, NttContext, Poly};
use crate::rlwe::{RlweCiphertext, RlweSecretKey, SeededRlweCiphertext};
use serde::{Deserialize, Serialize};

fn sample_error_poly(dim: usize, moduli: &[u64], sampler: &mut GaussianSampler) -> Poly {
    Poly::sample_gaussian_moduli(dim, moduli, sampler)
}

/// Gadget `[1, z, ..., z^(len-1)]`, bounding decomposed coefficients by z.
///
/// # Example
///
/// ```
/// use raven_inspire::rgsw::GadgetVector;
/// use raven_inspire::math::mod_q::DEFAULT_Q;
///
/// let gadget = GadgetVector::new(1 << 20, 3, DEFAULT_Q);
/// assert_eq!(gadget.len, 3);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GadgetVector {
    /// Base z.
    pub base: u64,
    /// Digit count, `ceil(log_z(q))`.
    pub len: usize,
    /// Ciphertext modulus q.
    pub q: u64,
}

impl GadgetVector {
    /// Gadget with an explicit digit count.
    pub fn new(base: u64, len: usize, q: u64) -> Self {
        debug_assert!(base > 1, "Gadget base must be > 1");
        debug_assert!(len > 0, "Gadget length must be > 0");
        Self { base, len, q }
    }

    /// [`Self::new`] with `len = ceil(log_base(q))`.
    pub fn from_base(base: u64, q: u64) -> Self {
        let len = ((q as f64).log2() / (base as f64).log2()).ceil() as usize;
        Self::new(base, len, q)
    }

    /// `z^i mod q`.
    pub fn power(&self, i: usize) -> u64 {
        let mut result = 1u128;
        let base = self.base as u128;
        let q = self.q as u128;

        for _ in 0..i {
            result = (result * base) % q;
        }
        result as u64
    }

    /// All powers `z^i mod q`.
    pub fn powers(&self) -> Vec<u64> {
        let mut powers = Vec::with_capacity(self.len);
        let mut current = 1u128;
        let base = self.base as u128;
        let q = self.q as u128;

        for _ in 0..self.len {
            powers.push(current as u64);
            current = (current * base) % q;
        }
        powers
    }
}

/// One-sided RGSW ciphertext: row `i` decrypts to `m*z^i`.
///
/// # Example
///
/// ```text
/// [ Row 0..ell-1: RLWE encryptions that decrypt to m*z^i ]
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RgswCiphertext {
    /// One RLWE row per gadget digit.
    pub rows: Vec<RlweCiphertext>,
    /// Decomposition base and length.
    pub gadget: GadgetVector,
}

impl RgswCiphertext {
    /// Pairs rows with the gadget they were built against.
    pub fn from_rows(rows: Vec<RlweCiphertext>, gadget: GadgetVector) -> Self {
        debug_assert_eq!(
            rows.len(),
            gadget.len,
            "one-sided RGSW must have gadget.len rows"
        );
        Self { rows, gadget }
    }

    /// Encrypts a small `message` (constant, bit, or monomial X^k) under `sk`.
    pub fn encrypt(
        sk: &RlweSecretKey,
        message: &Poly,
        gadget: &GadgetVector,
        sampler: &mut GaussianSampler,
        ctx: &NttContext,
    ) -> Self {
        let mut rng = os_seeded_chacha_or_abort("RgswCiphertext::encrypt");
        Self::encrypt_with_rng(sk, message, gadget, sampler, ctx, &mut rng)
    }

    /// [`Self::encrypt`], surfacing entropy failure as an error.
    pub fn try_encrypt(
        sk: &RlweSecretKey,
        message: &Poly,
        gadget: &GadgetVector,
        sampler: &mut GaussianSampler,
        ctx: &NttContext,
    ) -> Result<Self, EntropyUnavailable> {
        let mut rng = os_seeded_chacha("RgswCiphertext::try_encrypt")?;
        Ok(Self::encrypt_with_rng(
            sk, message, gadget, sampler, ctx, &mut rng,
        ))
    }

    /// [`Self::encrypt`] with a caller-provided RNG, which draws only the public
    /// `a` polynomials; the secret randomness stays in `sampler`.
    pub fn encrypt_with_rng<R: rand::RngCore + rand::CryptoRng>(
        sk: &RlweSecretKey,
        message: &Poly,
        gadget: &GadgetVector,
        sampler: &mut GaussianSampler,
        ctx: &NttContext,
        rng: &mut R,
    ) -> Self {
        let d = sk.ring_dim();
        let moduli = sk.poly.moduli();
        let ell = gadget.len;

        let mut rows = Vec::with_capacity(ell);
        let powers = gadget.powers();
        assert!(
            powers.len() >= ell,
            "gadget powers must have at least {} entries, got {}",
            ell,
            powers.len()
        );

        for &power in &powers[..ell] {
            let a = Poly::random_with_rng_moduli(d, moduli, rng);
            let error = sample_error_poly(d, moduli, sampler);

            let a_s = a.mul_ntt(&sk.poly, ctx);
            let b_base = &(-a_s) + &error;

            let scaled_msg = message.scalar_mul(power);
            let b = &b_base + &scaled_msg;

            rows.push(RlweCiphertext::from_parts(a, b));
        }

        Self {
            rows,
            gadget: gadget.clone(),
        }
    }

    /// [`Self::encrypt`] of a constant polynomial.
    pub fn encrypt_scalar(
        sk: &RlweSecretKey,
        message: u64,
        gadget: &GadgetVector,
        sampler: &mut GaussianSampler,
        ctx: &NttContext,
    ) -> Self {
        let msg_poly = Poly::constant_moduli(message, sk.ring_dim(), sk.poly.moduli());
        Self::encrypt(sk, &msg_poly, gadget, sampler, ctx)
    }

    /// Ring dimension d.
    pub fn ring_dim(&self) -> usize {
        self.rows[0].ring_dim()
    }

    /// Modulus q.
    pub fn modulus(&self) -> u64 {
        self.rows[0].modulus()
    }

    /// Gadget length.
    pub fn gadget_len(&self) -> usize {
        self.gadget.len
    }
}

/// [`RgswCiphertext`] carrying each row's `a` as a 32-byte seed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeededRgswCiphertext {
    /// One seeded RLWE row per gadget digit.
    pub rows: Vec<SeededRlweCiphertext>,
    /// Decomposition base and length.
    pub gadget: GadgetVector,
}

impl SeededRgswCiphertext {
    /// Encrypts `message`, drawing row seeds from an OS-seeded ChaCha20 stream
    /// and aborting on entropy failure. Use [`Self::try_encrypt`] to carry it.
    pub fn encrypt(
        sk: &RlweSecretKey,
        message: &Poly,
        gadget: &GadgetVector,
        sampler: &mut GaussianSampler,
        ctx: &NttContext,
    ) -> Self {
        let mut rng = os_seeded_chacha_or_abort("SeededRgswCiphertext::encrypt");
        Self::encrypt_with_rng(sk, message, gadget, sampler, ctx, &mut rng)
    }

    /// [`Self::encrypt`], surfacing entropy failure as an error.
    pub fn try_encrypt(
        sk: &RlweSecretKey,
        message: &Poly,
        gadget: &GadgetVector,
        sampler: &mut GaussianSampler,
        ctx: &NttContext,
    ) -> Result<Self, EntropyUnavailable> {
        let mut rng = os_seeded_chacha("SeededRgswCiphertext::try_encrypt")?;
        Ok(Self::encrypt_with_rng(
            sk, message, gadget, sampler, ctx, &mut rng,
        ))
    }

    /// [`Self::encrypt`] with a caller-supplied RNG.
    ///
    /// `rng` draws only the public row seeds; secret randomness stays in
    /// `sampler`, so a deterministic `rng` costs no IND-CPA as long as the
    /// seeds stay distinct.
    pub fn encrypt_with_rng<R: rand::RngCore + rand::CryptoRng>(
        sk: &RlweSecretKey,
        message: &Poly,
        gadget: &GadgetVector,
        sampler: &mut GaussianSampler,
        ctx: &NttContext,
        rng: &mut R,
    ) -> Self {
        let d = sk.ring_dim();
        let moduli = sk.poly.moduli();
        let ell = gadget.len;

        let mut rows = Vec::with_capacity(ell);
        let powers = gadget.powers();
        assert!(
            powers.len() >= ell,
            "gadget powers must have at least {} entries, got {}",
            ell,
            powers.len()
        );

        for &power in &powers[..ell] {
            let mut seed = [0u8; 32];
            rng.fill_bytes(&mut seed);

            let a = Poly::from_seed_moduli(&seed, d, moduli);
            let error = sample_error_poly(d, moduli, sampler);

            let a_s = a.mul_ntt(&sk.poly, ctx);
            let b_base = &(-a_s) + &error;

            let scaled_msg = message.scalar_mul(power);
            let b = &b_base + &scaled_msg;

            rows.push(SeededRlweCiphertext::new(seed, b));
        }

        Self {
            rows,
            gadget: gadget.clone(),
        }
    }

    /// [`Self::encrypt`] of a constant polynomial.
    pub fn encrypt_scalar(
        sk: &RlweSecretKey,
        message: u64,
        gadget: &GadgetVector,
        sampler: &mut GaussianSampler,
        ctx: &NttContext,
    ) -> Self {
        let msg_poly = Poly::constant_moduli(message, sk.ring_dim(), sk.poly.moduli());
        Self::encrypt(sk, &msg_poly, gadget, sampler, ctx)
    }

    /// Regenerates every row's `a` from its seed.
    pub fn expand(&self) -> RgswCiphertext {
        let rows: Vec<RlweCiphertext> = self
            .rows
            .iter()
            .map(crate::rlwe::SeededRlweCiphertext::expand)
            .collect();
        RgswCiphertext {
            rows,
            gadget: self.gadget.clone(),
        }
    }

    /// Ring dimension d.
    pub fn ring_dim(&self) -> usize {
        self.rows[0].ring_dim()
    }

    /// Modulus q.
    pub fn modulus(&self) -> u64 {
        self.rows[0].modulus()
    }

    /// Gadget length.
    pub fn gadget_len(&self) -> usize {
        self.gadget.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::InspireParams;

    fn test_params() -> InspireParams {
        InspireParams::secure_128_d2048()
    }

    fn make_ctx(params: &InspireParams) -> NttContext {
        params.ntt_context()
    }

    #[test]
    fn test_gadget_vector_creation() {
        let q = 1152921504606830593u64;
        let gadget = GadgetVector::new(1 << 20, 3, q);

        assert_eq!(gadget.base, 1 << 20);
        assert_eq!(gadget.len, 3);
        assert_eq!(gadget.q, q);
    }

    #[test]
    fn test_gadget_powers() {
        let q = 1152921504606830593u64;
        let base = 1 << 20;
        let gadget = GadgetVector::new(base, 3, q);

        let powers = gadget.powers();
        assert_eq!(powers.len(), 3);
        assert_eq!(powers[0], 1);
        assert_eq!(powers[1], base);
        assert_eq!(
            powers[2],
            ((base as u128 * base as u128) % q as u128) as u64
        );
    }

    #[test]
    fn test_gadget_from_base() {
        let q = 1152921504606830593u64;
        let gadget = GadgetVector::from_base(1 << 20, q);

        assert_eq!(gadget.len, 3);
    }

    #[test]
    fn test_rgsw_encryption_structure() {
        let params = test_params();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);
        let gadget = GadgetVector::new(params.gadget_base, params.query_gadget_len, params.q);

        let rgsw = RgswCiphertext::encrypt_scalar(&sk, 1, &gadget, &mut sampler, &ctx);

        assert_eq!(rgsw.rows.len(), params.query_gadget_len);
        assert_eq!(rgsw.ring_dim(), params.ring_dim);
        assert_eq!(rgsw.modulus(), params.q);
    }

    #[test]
    fn test_rgsw_encrypt_zero() {
        let params = test_params();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);
        let gadget = GadgetVector::new(params.gadget_base, params.query_gadget_len, params.q);

        let rgsw = RgswCiphertext::encrypt_scalar(&sk, 0, &gadget, &mut sampler, &ctx);

        assert_eq!(rgsw.rows.len(), 3);
    }

    #[test]
    fn seeded_rgsw_encrypt_with_rng_is_deterministic_when_seeds_match() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let params = test_params();
        let ctx = make_ctx(&params);
        let gadget = GadgetVector::new(params.gadget_base, params.query_gadget_len, params.q);

        let run_once = |sampler_seed: u64, rng_seed: u64| -> Vec<[u8; 32]> {
            let mut sampler = GaussianSampler::with_seed(params.sigma, sampler_seed);
            let sk = RlweSecretKey::generate(&params, &mut sampler);
            let msg = Poly::constant_moduli(1u64, params.ring_dim, params.moduli());
            let mut rng = ChaCha20Rng::seed_from_u64(rng_seed);
            let ct = SeededRgswCiphertext::encrypt_with_rng(
                &sk,
                &msg,
                &gadget,
                &mut sampler,
                &ctx,
                &mut rng,
            );
            ct.rows.iter().map(|r| r.seed).collect()
        };

        let seeds_a = run_once(0xAAAA_BBBB_CCCC_DDDD, 0xDEAD_BEEF_CAFE_F00D);
        let seeds_b = run_once(0xAAAA_BBBB_CCCC_DDDD, 0xDEAD_BEEF_CAFE_F00D);
        assert_eq!(
            seeds_a, seeds_b,
            "matching sampler+RNG seeds must yield identical RGSW row seeds"
        );

        let seeds_diff = run_once(0xAAAA_BBBB_CCCC_DDDD, 0x1234_5678_9ABC_DEF0);
        assert_ne!(
            seeds_a, seeds_diff,
            "different RNG seed must yield different RGSW row seeds"
        );
    }
}
