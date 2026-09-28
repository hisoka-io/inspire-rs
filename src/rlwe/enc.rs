//! RLWE encryption and decryption; `delta = floor(q/p)` is the scaling factor.

use subtle::{ConditionallySelectable, ConstantTimeGreater};

use crate::lwe::LweCiphertext;
use crate::math::modular::reduce_by_public_reciprocal;
use crate::math::{GaussianSampler, NttContext, Poly};
use crate::params::InspireParams;

use super::types::{RlweCiphertext, RlweSecretKey};

/// `floor((v + floor(delta/2)) / delta) mod p` over a secret `v`, dividing only the
/// public `delta` and `p`: a u128 divide is a variable-latency software call on wasm32.
struct PlaintextRounding {
    delta: u128,
    half_delta: u128,
    /// `floor((2^64-1)/delta)`. Estimates at most one short: below 2^64 by the Barrett
    /// bound, and above it the low half is under `delta/2` and adds nothing.
    reciprocal: u128,
    p: u64,
    p_reciprocal: u64,
}

impl PlaintextRounding {
    fn new(delta: u64, p: u64) -> Self {
        Self {
            delta: u128::from(delta),
            half_delta: u128::from(delta / 2),
            reciprocal: u128::from(u64::MAX / delta),
            p,
            p_reciprocal: u64::MAX / p,
        }
    }

    fn round(&self, noisy: u64) -> u64 {
        let numerator = u128::from(noisy) + self.half_delta;
        let (high, low) = (numerator >> 64, numerator & u128::from(u64::MAX));
        let estimate = high * self.reciprocal + ((low * self.reciprocal) >> 64);
        let short = (numerator - estimate * self.delta).ct_gt(&(self.delta - 1));
        let quotient = u128::conditional_select(&estimate, &(estimate + 1), short);
        // Truncating to u64 matches the exact quotient's truncation.
        reduce_by_public_reciprocal(quotient as u64, self.p, self.p_reciprocal)
    }
}

impl RlweSecretKey {
    /// Samples a secret key from the error distribution.
    pub fn generate(params: &InspireParams, sampler: &mut GaussianSampler) -> Self {
        let poly = Poly::sample_gaussian_moduli(params.ring_dim, params.moduli(), sampler);
        Self { poly }
    }
}

impl RlweCiphertext {
    /// `(a, -a*s + e + delta*m)`; `message_poly` coefficients MUST lie in `[0, p)`.
    pub fn encrypt(
        sk: &RlweSecretKey,
        message_poly: &Poly,
        delta: u64,
        a_random: Poly,
        error: &Poly,
        ctx: &NttContext,
    ) -> Self {
        let scaled_msg = message_poly.scalar_mul(delta);
        let neg_a_s = -a_random.mul_ntt(&sk.poly, ctx);
        let b = &(&neg_a_s + error) + &scaled_msg;

        Self { a: a_random, b }
    }

    /// [`Self::encrypt`] against a CRS-derived `a`, which need not be transmitted.
    pub fn encrypt_with_crs(
        sk: &RlweSecretKey,
        message_poly: &Poly,
        delta: u64,
        crs_a: &Poly,
        error: &Poly,
        ctx: &NttContext,
    ) -> Self {
        let scaled_msg = message_poly.scalar_mul(delta);
        let neg_a_s = -crs_a.mul_ntt(&sk.poly, ctx);
        let b = &(&neg_a_s + error) + &scaled_msg;

        Self {
            a: crs_a.clone(),
            b,
        }
    }

    /// `round((a*s + b) / delta) mod p`.
    pub fn decrypt(&self, sk: &RlweSecretKey, delta: u64, p: u64, ctx: &NttContext) -> Poly {
        let d = self.ring_dim();

        let a_s = self.a.mul_ntt(&sk.poly, ctx);
        let noisy_msg = &a_s + &self.b;

        let rounding = PlaintextRounding::new(delta, p);
        let coeffs = noisy_msg
            .coeffs_composed_ct()
            .into_iter()
            .take(d)
            .map(|noisy| rounding.round(noisy))
            .collect();

        // Every rounded value is already below `p`; `from_coeffs` would `%` it again.
        Poly::from_crt_coeffs_reduced(coeffs, &[p])
    }

    /// Componentwise sum, decrypting to `m1 + m2`.
    pub fn add(&self, other: &RlweCiphertext) -> RlweCiphertext {
        RlweCiphertext {
            a: &self.a + &other.a,
            b: &self.b + &other.b,
        }
    }

    /// Componentwise difference, decrypting to `m1 - m2`.
    pub fn sub(&self, other: &RlweCiphertext) -> RlweCiphertext {
        RlweCiphertext {
            a: &self.a - &other.a,
            b: &self.b - &other.b,
        }
    }

    /// Componentwise scalar product, decrypting to `c*m`.
    pub fn scalar_mul(&self, scalar: u64) -> RlweCiphertext {
        RlweCiphertext {
            a: self.a.scalar_mul(scalar),
            b: self.b.scalar_mul(scalar),
        }
    }

    /// Componentwise plaintext product, decrypting to `p(X)*m(X) mod (X^d + 1)`.
    pub fn poly_mul(&self, plaintext_poly: &Poly, ctx: &NttContext) -> RlweCiphertext {
        RlweCiphertext {
            a: self.a.mul_ntt(plaintext_poly, ctx),
            b: self.b.mul_ntt(plaintext_poly, ctx),
        }
    }

    /// `(0, 0)`, the identity for homomorphic addition.
    pub fn zero(params: &InspireParams) -> RlweCiphertext {
        let a = Poly::zero_moduli(params.ring_dim, params.moduli());
        let b = Poly::zero_moduli(params.ring_dim, params.moduli());
        RlweCiphertext { a, b }
    }

    /// `(0, delta*m)`: decrypts under any key and hides nothing, for folding a
    /// known plaintext into a homomorphic chain.
    pub fn trivial_encrypt(
        message_poly: &Poly,
        delta: u64,
        params: &InspireParams,
    ) -> RlweCiphertext {
        let a = Poly::zero_moduli(params.ring_dim, params.moduli());
        let b = message_poly.scalar_mul(delta);
        RlweCiphertext { a, b }
    }

    /// RLWE-to-LWE sample extraction at coefficient 0: `a'_i = a_{d-i mod d}`, `b' = b_0`.
    pub fn sample_extract_coeff0(&self) -> LweCiphertext {
        let d = self.ring_dim();
        let q = self.modulus();

        let mut a_vec = vec![0u64; d];
        a_vec[0] = self.a.coeff(0);
        for (i, a) in a_vec.iter_mut().enumerate().take(d).skip(1) {
            *a = self.a.coeff(d - i);
        }

        let b0 = self.b.coeff(0);

        LweCiphertext { a: a_vec, b: b0, q }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> InspireParams {
        InspireParams::secure_128_d2048()
    }

    fn make_ctx(params: &InspireParams) -> NttContext {
        params.ntt_context()
    }

    fn random_poly(params: &InspireParams) -> Poly {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0x5eed);
        Poly::random_with_rng_moduli(params.ring_dim, params.moduli(), &mut rng)
    }

    fn sample_error_poly(params: &InspireParams, sampler: &mut GaussianSampler) -> Poly {
        Poly::sample_gaussian_moduli(params.ring_dim, params.moduli(), sampler)
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let params = test_params();
        let delta = params.delta();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);

        let msg_coeffs: Vec<u64> = (0..params.ring_dim)
            .map(|i| (i as u64) % params.p)
            .collect();
        let message = Poly::from_coeffs_moduli(msg_coeffs.clone(), params.moduli());

        let a_random = random_poly(&params);
        let error = sample_error_poly(&params, &mut sampler);

        let ct = RlweCiphertext::encrypt(&sk, &message, delta, a_random, &error, &ctx);

        let decrypted = ct.decrypt(&sk, delta, params.p, &ctx);

        for (i, expected) in msg_coeffs.iter().enumerate().take(params.ring_dim) {
            assert_eq!(decrypted.coeff(i), *expected, "Mismatch at coefficient {i}");
        }
    }

    #[test]
    fn test_encrypt_decrypt_zero() {
        let params = test_params();
        let delta = params.delta();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);

        let message = Poly::zero_moduli(params.ring_dim, params.moduli());
        let a_random = random_poly(&params);
        let error = sample_error_poly(&params, &mut sampler);

        let ct = RlweCiphertext::encrypt(&sk, &message, delta, a_random, &error, &ctx);
        let decrypted = ct.decrypt(&sk, delta, params.p, &ctx);

        for i in 0..params.ring_dim {
            assert_eq!(decrypted.coeff(i), 0, "Expected zero at coefficient {i}");
        }
    }

    #[test]
    fn test_homomorphic_addition() {
        let params = test_params();
        let delta = params.delta();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);

        let msg1_coeffs: Vec<u64> = (0..params.ring_dim).map(|i| (i as u64) % 100).collect();
        let msg2_coeffs: Vec<u64> = (0..params.ring_dim)
            .map(|i| ((i + 50) as u64) % 100)
            .collect();

        let msg1 = Poly::from_coeffs_moduli(msg1_coeffs.clone(), params.moduli());
        let msg2 = Poly::from_coeffs_moduli(msg2_coeffs.clone(), params.moduli());

        let a1 = random_poly(&params);
        let e1 = sample_error_poly(&params, &mut sampler);
        let ct1 = RlweCiphertext::encrypt(&sk, &msg1, delta, a1, &e1, &ctx);

        let a2 = random_poly(&params);
        let e2 = sample_error_poly(&params, &mut sampler);
        let ct2 = RlweCiphertext::encrypt(&sk, &msg2, delta, a2, &e2, &ctx);

        let ct_sum = ct1.add(&ct2);
        let decrypted = ct_sum.decrypt(&sk, delta, params.p, &ctx);

        for i in 0..params.ring_dim {
            let expected = (msg1_coeffs[i] + msg2_coeffs[i]) % params.p;
            assert_eq!(decrypted.coeff(i), expected, "Mismatch at coefficient {i}");
        }
    }

    #[test]
    fn test_homomorphic_subtraction() {
        let params = test_params();
        let delta = params.delta();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);

        let msg1_coeffs: Vec<u64> = (0..params.ring_dim)
            .map(|i| 200 + (i as u64) % 100)
            .collect();
        let msg2_coeffs: Vec<u64> = (0..params.ring_dim).map(|i| (i as u64) % 100).collect();

        let msg1 = Poly::from_coeffs_moduli(msg1_coeffs.clone(), params.moduli());
        let msg2 = Poly::from_coeffs_moduli(msg2_coeffs.clone(), params.moduli());

        let a1 = random_poly(&params);
        let e1 = sample_error_poly(&params, &mut sampler);
        let ct1 = RlweCiphertext::encrypt(&sk, &msg1, delta, a1, &e1, &ctx);

        let a2 = random_poly(&params);
        let e2 = sample_error_poly(&params, &mut sampler);
        let ct2 = RlweCiphertext::encrypt(&sk, &msg2, delta, a2, &e2, &ctx);

        let ct_diff = ct1.sub(&ct2);
        let decrypted = ct_diff.decrypt(&sk, delta, params.p, &ctx);

        for i in 0..params.ring_dim {
            let expected = (msg1_coeffs[i] - msg2_coeffs[i]) % params.p;
            assert_eq!(decrypted.coeff(i), expected, "Mismatch at coefficient {i}");
        }
    }

    #[test]
    fn test_scalar_multiplication() {
        let params = test_params();
        let delta = params.delta();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);

        let msg_coeffs: Vec<u64> = (0..params.ring_dim).map(|i| (i as u64) % 50).collect();
        let message = Poly::from_coeffs_moduli(msg_coeffs.clone(), params.moduli());

        let a = random_poly(&params);
        let e = sample_error_poly(&params, &mut sampler);
        let ct = RlweCiphertext::encrypt(&sk, &message, delta, a, &e, &ctx);

        let scalar = 3u64;
        let ct_scaled = ct.scalar_mul(scalar);
        let decrypted = ct_scaled.decrypt(&sk, delta, params.p, &ctx);

        for (i, msg_coeff) in msg_coeffs.iter().enumerate().take(params.ring_dim) {
            let expected = (*msg_coeff * scalar) % params.p;
            assert_eq!(decrypted.coeff(i), expected, "Mismatch at coefficient {i}");
        }
    }

    #[test]
    fn test_zero_ciphertext() {
        let params = test_params();
        let delta = params.delta();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);

        let zero_ct = RlweCiphertext::zero(&params);
        let decrypted = zero_ct.decrypt(&sk, delta, params.p, &ctx);

        for i in 0..params.ring_dim {
            assert_eq!(decrypted.coeff(i), 0);
        }
    }

    #[test]
    fn test_crs_mode_encryption() {
        let params = test_params();
        let delta = params.delta();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);

        let crs_a = random_poly(&params);

        let msg_coeffs: Vec<u64> = (0..params.ring_dim)
            .map(|i| (i as u64) % params.p)
            .collect();
        let message = Poly::from_coeffs_moduli(msg_coeffs.clone(), params.moduli());
        let error = sample_error_poly(&params, &mut sampler);

        let ct = RlweCiphertext::encrypt_with_crs(&sk, &message, delta, &crs_a, &error, &ctx);
        let decrypted = ct.decrypt(&sk, delta, params.p, &ctx);

        for (i, expected) in msg_coeffs.iter().enumerate().take(params.ring_dim) {
            assert_eq!(decrypted.coeff(i), *expected, "Mismatch at coefficient {i}");
        }
    }
}

#[cfg(test)]
mod decrypt_differential {
    use super::*;
    use crate::math::mod_q::DEFAULT_Q;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    const P: u64 = 65_537;
    /// The served mod-switch rung and the checked 45-bit one, spelled out because the
    /// constants live behind a feature this module does not require.
    const SERVED_36_BIT: u64 = 68_718_428_161;
    const CHECKED_45_BIT: u64 = 35_184_372_060_161;
    const UPSTREAM_TWO_CRT: [u64; 2] = [268_369_921, 249_561_089];

    fn reference_round(noisy: u64, delta: u64, p: u64) -> u64 {
        let half_delta = delta / 2;
        let rounded = ((noisy as u128 + half_delta as u128) / delta as u128) as u64;
        rounded % p
    }

    /// The decrypt this module replaced, kept verbatim as the oracle.
    fn reference_decrypt(ct: &RlweCiphertext, sk: &RlweSecretKey, delta: u64, p: u64) -> Poly {
        let ctx = NttContext::with_moduli(ct.ring_dim(), ct.a.moduli());
        let d = ct.ring_dim();
        let noisy_msg = &ct.a.mul_ntt(&sk.poly, &ctx) + &ct.b;
        let coeffs = (0..d)
            .map(|i| reference_round(noisy_msg.coeff(i), delta, p))
            .collect();
        Poly::from_coeffs(coeffs, p)
    }

    fn assert_round_matches(noisy: u64, delta: u64, p: u64) {
        assert_eq!(
            PlaintextRounding::new(delta, p).round(noisy),
            reference_round(noisy, delta, p),
            "noisy={noisy} delta={delta} p={p}"
        );
    }

    /// The quotient only changes at `k*delta - delta/2`, so every value within 16 of
    /// every such step, both ends of `[0, q)`, and a random fill cover the domain.
    fn sweep_quotient_steps(q: u64, p: u64, random_fill: usize) {
        let delta = q / p;
        let rounding = PlaintextRounding::new(delta, p);
        let check = |noisy: u64| {
            assert_eq!(
                rounding.round(noisy),
                reference_round(noisy, delta, p),
                "noisy={noisy} q={q}"
            );
        };
        let steps = q / delta + 2;
        for k in 0..=steps {
            let step = (u128::from(k) * u128::from(delta)).saturating_sub(u128::from(delta / 2));
            for offset in -16i128..=16 {
                let noisy = step as i128 + offset;
                if (0..i128::from(q)).contains(&noisy) {
                    check(noisy as u64);
                }
            }
        }
        for noisy in (0..64).chain(q - 64..q) {
            check(noisy);
        }
        let mut rng = ChaCha20Rng::seed_from_u64(q ^ 0xD0_0D);
        for _ in 0..random_fill {
            check(rng.gen_range(0..q));
        }
    }

    #[test]
    fn rounding_matches_the_divide_at_every_quotient_step_of_the_shipped_moduli() {
        for q in [DEFAULT_Q, SERVED_36_BIT, CHECKED_45_BIT] {
            sweep_quotient_steps(q, P, 1 << 20);
        }
    }

    #[test]
    fn rounding_matches_the_divide_across_the_whole_u64_domain() {
        let edges = [
            0u64,
            1,
            2,
            3,
            (1 << 32) - 1,
            1 << 32,
            (1 << 63) - 1,
            1 << 63,
            u64::MAX - 1,
            u64::MAX,
        ];
        for &noisy in &edges {
            for &delta in &edges[1..] {
                for p in [1, 2, 3, P, u64::MAX] {
                    assert_round_matches(noisy, delta, p);
                }
            }
        }
        let mut rng = ChaCha20Rng::seed_from_u64(0x0DD_BA11);
        for _ in 0..(1 << 20) {
            let noisy: u64 = rng.gen();
            let delta = rng.gen::<u64>() >> rng.gen_range(0..64);
            let p = rng.gen::<u64>() >> rng.gen_range(0..64);
            assert_round_matches(noisy, delta.max(1), p.max(1));
        }
    }

    /// Every value below the served rung, old against new. About 7e10 evaluations, so
    /// run on demand: `cargo test --release --lib -- --ignored exhaustive`.
    #[test]
    #[ignore = "exhaustive over 2^36 values, minutes on all cores. Trigger: changing \
                PlaintextRounding, the decrypt rounding or the served 36-bit modulus."]
    fn rounding_matches_the_divide_exhaustively_below_the_served_rung() {
        let delta = SERVED_36_BIT / P;
        let rounding = PlaintextRounding::new(delta, P);
        let workers = std::thread::available_parallelism().map_or(1, usize::from) as u64;
        let span = SERVED_36_BIT.div_ceil(workers);
        let mismatches: Vec<u64> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|worker| {
                    let rounding = &rounding;
                    scope.spawn(move || {
                        let start = worker * span;
                        let end = (start + span).min(SERVED_36_BIT);
                        (start..end).find(|&noisy| {
                            rounding.round(noisy) != reference_round(noisy, delta, P)
                        })
                    })
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|handle| handle.join().expect("worker completes"))
                .collect()
        });
        assert!(mismatches.is_empty(), "first mismatches: {mismatches:?}");
    }

    fn assert_decrypt_matches(moduli: &[u64], dim: usize, p: u64, seed: u64) {
        let q: u64 = moduli.iter().product();
        let delta = q / p;
        let ctx = NttContext::with_moduli(dim, moduli);
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut sampler = GaussianSampler::with_seed(6.4, seed);
        let sk = RlweSecretKey::from_poly(Poly::sample_gaussian_moduli(dim, moduli, &mut sampler));
        for trial in 0..8 {
            let ct = if trial % 2 == 0 {
                RlweCiphertext::from_parts(
                    Poly::random_with_rng_moduli(dim, moduli, &mut rng),
                    Poly::random_with_rng_moduli(dim, moduli, &mut rng),
                )
            } else {
                let message = Poly::from_coeffs_moduli(
                    (0..dim).map(|_| rng.gen_range(0..p)).collect(),
                    moduli,
                );
                let error = Poly::sample_gaussian_moduli(dim, moduli, &mut sampler);
                let a = Poly::random_with_rng_moduli(dim, moduli, &mut rng);
                RlweCiphertext::encrypt(&sk, &message, delta, a, &error, &ctx)
            };
            let got = ct.decrypt(&sk, delta, p, &ctx);
            let want = reference_decrypt(&ct, &sk, delta, p);
            assert_eq!(
                got.coeffs(),
                want.coeffs(),
                "moduli={moduli:?} trial={trial}"
            );
            assert_eq!(got.moduli(), want.moduli());
        }
    }

    #[test]
    fn decrypt_is_byte_identical_to_the_divide_at_shipped_and_two_crt_moduli() {
        assert_decrypt_matches(&[DEFAULT_Q], 2048, P, 1);
        assert_decrypt_matches(&[SERVED_36_BIT], 2048, P, 2);
        assert_decrypt_matches(&[CHECKED_45_BIT], 2048, P, 3);
        assert_decrypt_matches(&UPSTREAM_TWO_CRT, 2048, P, 4);
        assert_decrypt_matches(&UPSTREAM_TWO_CRT, 2048, 65_536, 5);
    }
}
