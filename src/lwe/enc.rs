//! LWE encryption, decryption, and homomorphic operations.

use super::types::{LweCiphertext, LweSecretKey};
use crate::math::{GaussianSampler, ModQ};
use subtle::{ConditionallySelectable, ConstantTimeEq};

impl LweSecretKey {
    /// Samples a secret key from the error distribution.
    pub fn generate(dim: usize, q: u64, sampler: &mut GaussianSampler) -> Self {
        let coeffs: Vec<u64> = (0..dim)
            .map(|_| {
                let sample = sampler.sample();
                ModQ::from_signed(sample, q)
            })
            .collect();

        Self { coeffs, dim, q }
    }

    /// Wraps existing coefficients.
    pub fn from_coeffs(coeffs: Vec<u64>, q: u64) -> Self {
        let dim = coeffs.len();
        Self { coeffs, dim, q }
    }

    /// Key that `sample_extract_coeff0` output decrypts under: `s\[0\]`, then `-s\[i\]`.
    ///
    /// The negacyclic wrap in `coeff_0(a*s)` puts a minus on every term but the first.
    pub fn from_rlwe(rlwe_sk: &crate::rlwe::RlweSecretKey) -> Self {
        let d = rlwe_sk.ring_dim();
        let q = rlwe_sk.modulus();

        let mut coeffs = vec![0u64; d];
        coeffs[0] = rlwe_sk.poly.coeff(0);
        for (i, coeff) in coeffs.iter_mut().enumerate().take(d).skip(1) {
            let s_i = rlwe_sk.poly.coeff(i);
            let negated = q - s_i;
            *coeff = u64::conditional_select(&negated, &0, s_i.ct_eq(&0));
        }

        Self { coeffs, dim: d, q }
    }
}

impl LweCiphertext {
    /// `(a, -<a, s> + e + delta*m)`.
    pub fn encrypt(sk: &LweSecretKey, message: u64, delta: u64, a: Vec<u64>, error: i64) -> Self {
        let q = sk.q;

        let inner_product = inner_product_mod(&a, &sk.coeffs, q);

        let neg_inner = ModQ::negate(inner_product, q);
        let e_mod = ModQ::from_signed(error, q);
        let delta_m = ModQ::mul(delta, message, q);

        let b = ModQ::add(neg_inner, ModQ::add(e_mod, delta_m, q), q);

        Self { a, b, q }
    }

    /// [`Self::encrypt`] against a CRS-derived `a`, which need not be transmitted.
    pub fn encrypt_with_crs(
        sk: &LweSecretKey,
        message: u64,
        delta: u64,
        crs_a: &[u64],
        error: i64,
    ) -> Self {
        Self::encrypt(sk, message, delta, crs_a.to_vec(), error)
    }

    /// `round((p/q) * (b + <a, s>)) mod p`.
    pub fn decrypt(&self, sk: &LweSecretKey, delta: u64, p: u64) -> u64 {
        let q = self.q;

        let inner_product = inner_product_mod(&self.a, &sk.coeffs, q);

        let noisy_message = ModQ::add(self.b, inner_product, q);

        round_decode(noisy_message, q, p, delta)
    }

    /// Componentwise sum, decrypting to `m1 + m2`.
    pub fn add(&self, other: &LweCiphertext) -> Self {
        debug_assert_eq!(self.q, other.q);
        debug_assert_eq!(self.a.len(), other.a.len());

        let q = self.q;
        let a: Vec<u64> = self
            .a
            .iter()
            .zip(other.a.iter())
            .map(|(&x, &y)| ModQ::add(x, y, q))
            .collect();

        let b = ModQ::add(self.b, other.b, q);

        Self { a, b, q }
    }

    /// Componentwise difference, decrypting to `m1 - m2`.
    pub fn sub(&self, other: &LweCiphertext) -> Self {
        debug_assert_eq!(self.q, other.q);
        debug_assert_eq!(self.a.len(), other.a.len());

        let q = self.q;
        let a: Vec<u64> = self
            .a
            .iter()
            .zip(other.a.iter())
            .map(|(&x, &y)| ModQ::sub(x, y, q))
            .collect();

        let b = ModQ::sub(self.b, other.b, q);

        Self { a, b, q }
    }

    /// Componentwise scalar product, decrypting to `scalar * m`.
    pub fn scalar_mul(&self, scalar: u64) -> Self {
        let q = self.q;
        let a: Vec<u64> = self.a.iter().map(|&x| ModQ::mul(x, scalar, q)).collect();
        let b = ModQ::mul(self.b, scalar, q);

        Self { a, b, q }
    }

    /// All-zero ciphertext, decrypting to 0.
    pub fn zero(dim: usize, q: u64) -> Self {
        Self {
            a: vec![0; dim],
            b: 0,
            q,
        }
    }
}

fn inner_product_mod(a: &[u64], b: &[u64], q: u64) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .fold(0u64, |acc, (&x, &y)| ModQ::add(acc, ModQ::mul(x, y, q), q))
}

/// Recovers m from `noisy = e + delta*m`, requiring `|e| < delta/2`.
fn round_decode(noisy: u64, q: u64, p: u64, _delta: u64) -> u64 {
    let scaled = (noisy as u128) * (p as u128);
    let divided = scaled / (q as u128);
    let remainder = scaled % (q as u128);

    let rounded = if remainder >= (q as u128) / 2 {
        divided + 1
    } else {
        divided
    };

    (rounded % (p as u128)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;

    const DIM: usize = 2048;
    const Q: u64 = 1152921504606830593;
    const P: u64 = 65536;

    fn delta() -> u64 {
        Q / P
    }

    fn gen_small_coeffs<R: Rng>(rng: &mut R, dim: usize, q: u64) -> Vec<u64> {
        (0..dim)
            .map(|_| {
                let val: i64 = (rng.gen::<u8>() % 7) as i64 - 3;
                ModQ::from_signed(val, q)
            })
            .collect()
    }

    fn gen_random_vec<R: Rng>(rng: &mut R, dim: usize, q: u64) -> Vec<u64> {
        (0..dim).map(|_| rng.gen::<u64>() % q).collect()
    }

    fn gen_small_error<R: Rng>(rng: &mut R) -> i64 {
        (rng.gen::<u8>() % 5) as i64 - 2
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(12345);

        let sk = LweSecretKey::from_coeffs(gen_small_coeffs(&mut rng, DIM, Q), Q);
        let a = gen_random_vec(&mut rng, DIM, Q);
        let error = gen_small_error(&mut rng);

        for message in [0, 1, 100, 1000, P - 1] {
            let ct = LweCiphertext::encrypt(&sk, message, delta(), a.clone(), error);
            let decrypted = ct.decrypt(&sk, delta(), P);
            assert_eq!(decrypted, message, "Failed for message {message}");
        }
    }

    #[test]
    fn test_homomorphic_addition() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(54321);

        let sk = LweSecretKey::from_coeffs(gen_small_coeffs(&mut rng, DIM, Q), Q);

        let m1 = 1000u64;
        let m2 = 2000u64;

        let a1 = gen_random_vec(&mut rng, DIM, Q);
        let a2 = gen_random_vec(&mut rng, DIM, Q);
        let e1 = gen_small_error(&mut rng);
        let e2 = gen_small_error(&mut rng);

        let ct1 = LweCiphertext::encrypt(&sk, m1, delta(), a1, e1);
        let ct2 = LweCiphertext::encrypt(&sk, m2, delta(), a2, e2);

        let ct_sum = ct1.add(&ct2);
        let decrypted = ct_sum.decrypt(&sk, delta(), P);

        assert_eq!(decrypted, (m1 + m2) % P);
    }

    #[test]
    fn test_homomorphic_subtraction() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(98765);

        let sk = LweSecretKey::from_coeffs(gen_small_coeffs(&mut rng, DIM, Q), Q);

        let m1 = 5000u64;
        let m2 = 2000u64;

        let a1 = gen_random_vec(&mut rng, DIM, Q);
        let a2 = gen_random_vec(&mut rng, DIM, Q);
        let e1 = gen_small_error(&mut rng);
        let e2 = gen_small_error(&mut rng);

        let ct1 = LweCiphertext::encrypt(&sk, m1, delta(), a1, e1);
        let ct2 = LweCiphertext::encrypt(&sk, m2, delta(), a2, e2);

        let ct_diff = ct1.sub(&ct2);
        let decrypted = ct_diff.decrypt(&sk, delta(), P);

        assert_eq!(decrypted, (m1 - m2) % P);
    }

    #[test]
    fn test_scalar_multiplication() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(11111);

        let sk = LweSecretKey::from_coeffs(gen_small_coeffs(&mut rng, DIM, Q), Q);

        let message = 100u64;
        let scalar = 5u64;

        let a = gen_random_vec(&mut rng, DIM, Q);
        let error: i64 = 1;

        let ct = LweCiphertext::encrypt(&sk, message, delta(), a, error);
        let ct_scaled = ct.scalar_mul(scalar);
        let decrypted = ct_scaled.decrypt(&sk, delta(), P);

        assert_eq!(decrypted, (message * scalar) % P);
    }

    #[test]
    fn test_crs_encryption() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(22222);

        let sk = LweSecretKey::from_coeffs(gen_small_coeffs(&mut rng, DIM, Q), Q);
        let crs_a = gen_random_vec(&mut rng, DIM, Q);

        for message in [42, 100, 1000] {
            let error = gen_small_error(&mut rng);
            let ct = LweCiphertext::encrypt_with_crs(&sk, message, delta(), &crs_a, error);

            assert_eq!(ct.a, crs_a);

            let decrypted = ct.decrypt(&sk, delta(), P);
            assert_eq!(decrypted, message);
        }
    }

    #[test]
    fn test_lwe_extraction_key_consistency() {
        use crate::math::{GaussianSampler, Poly};
        use crate::params::InspireParams;
        use crate::rlwe::{RlweCiphertext, RlweSecretKey};

        let params = InspireParams {
            ring_dim: 256,
            q: 1152921504606830593,
            crt_moduli: vec![1152921504606830593],
            p: 65536,
            sigma: 6.4,
            gadget_base: 1 << 20,
            gadget_len: 3,
            security_level: crate::params::SecurityLevel::Bits128,
        };

        let d = params.ring_dim;
        let q = params.q;
        let delta_val = params.delta();
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let rlwe_sk = RlweSecretKey::generate(&params, &mut sampler);

        let lwe_sk = LweSecretKey::from_rlwe(&rlwe_sk);

        let message = 12345u64;
        let mut msg_coeffs = vec![0u64; d];
        msg_coeffs[0] = message;
        let msg_poly = Poly::from_coeffs(msg_coeffs, q);

        let a = Poly::random(d, q);
        let error_coeffs: Vec<u64> = (0..d)
            .map(|_| ModQ::from_signed(sampler.sample(), q))
            .collect();
        let error = Poly::from_coeffs(error_coeffs, q);
        let rlwe_ct = RlweCiphertext::encrypt(&rlwe_sk, &msg_poly, delta_val, a, &error, &ctx);

        let lwe_ct = rlwe_ct.sample_extract_coeff0();

        let lwe_decrypted = lwe_ct.decrypt(&lwe_sk, delta_val, params.p);

        assert_eq!(
            lwe_decrypted, message,
            "LWE decryption should match: got {lwe_decrypted}, expected {message}"
        );
    }
}
