//! Shift-and-add packing of RLWE ciphertexts that hold their message in coefficient 0
//! ONLY. A PIR response ciphertext holds the whole rotated shard, so it must go through
//! `pack_lwe_to_rlwe` instead.

use crate::ks::KeySwitchingMatrix;
use crate::lwe::LweCiphertext;
use crate::math::{NttContext, Poly};
use crate::params::InspireParams;
use crate::rgsw::{gadget_decompose, GadgetVector};
use crate::rlwe::RlweCiphertext;

/// Shift ciphertext i by X^i and sum, landing message i at coefficient i.
pub fn pack_rlwe_coeffs(
    rlwe_ciphertexts: &[RlweCiphertext],
    params: &InspireParams,
) -> RlweCiphertext {
    let d = params.ring_dim;
    let q = params.q;
    let moduli = params.moduli();

    if rlwe_ciphertexts.is_empty() {
        return RlweCiphertext::from_parts(
            Poly::zero_moduli(d, moduli),
            Poly::zero_moduli(d, moduli),
        );
    }

    let mut result_a = Poly::zero_moduli(d, moduli);
    let mut result_b = Poly::zero_moduli(d, moduli);

    for (slot, rlwe) in rlwe_ciphertexts.iter().enumerate() {
        let shifted = if slot == 0 {
            rlwe.clone()
        } else {
            mul_rlwe_by_monomial(rlwe, slot, q)
        };
        result_a = &result_a + &shifted.a;
        result_b = &result_b + &shifted.b;
    }

    RlweCiphertext::from_parts(result_a, result_b)
}

fn mul_rlwe_by_monomial(ct: &RlweCiphertext, k: usize, q: u64) -> RlweCiphertext {
    let a_shifted = mul_poly_by_monomial(&ct.a, k, q);
    let b_shifted = mul_poly_by_monomial(&ct.b, k, q);
    RlweCiphertext::from_parts(a_shifted, b_shifted)
}

/// Multiply by X^k in R_q = Z_q[X]/(X^d + 1); wrapped coefficients get negated.
fn mul_poly_by_monomial(poly: &Poly, k: usize, q: u64) -> Poly {
    let d = poly.dimension();
    let k = k % (2 * d);

    if k == 0 {
        return poly.clone();
    }

    let mut result_coeffs = vec![0u64; d];

    for i in 0..d {
        let coeff = poly.coeff(i);
        // Ciphertext coefficients are public to the server.
        if coeff == 0 {
            continue;
        }

        let new_idx = i + k;
        if new_idx < d {
            result_coeffs[new_idx] = mod_add(result_coeffs[new_idx], coeff, q);
        } else if new_idx < 2 * d {
            let actual_idx = new_idx - d;
            let neg_coeff = mod_sub(0, coeff, q);
            result_coeffs[actual_idx] = mod_add(result_coeffs[actual_idx], neg_coeff, q);
        } else {
            let actual_idx = new_idx - 2 * d;
            result_coeffs[actual_idx] = mod_add(result_coeffs[actual_idx], coeff, q);
        }
    }

    Poly::from_coeffs_moduli(result_coeffs, poly.moduli())
}

#[inline]
fn mod_add(a: u64, b: u64, q: u64) -> u64 {
    let sum = a as u128 + b as u128;
    (sum % q as u128) as u64
}

#[inline]
fn mod_sub(a: u64, b: u64, q: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        q - b + a
    }
}

/// Collect the LWE b values into a trivial (a=0) RLWE ciphertext, which any RLWE key
/// decrypts - it preserves no encryption under a specific key.
#[allow(dead_code)]
pub fn pack_lwe_trivial(
    lwe_ciphertexts: &[LweCiphertext],
    params: &InspireParams,
) -> RlweCiphertext {
    let d = params.ring_dim;
    let moduli = params.moduli();

    if lwe_ciphertexts.is_empty() {
        return RlweCiphertext::from_parts(
            Poly::zero_moduli(d, moduli),
            Poly::zero_moduli(d, moduli),
        );
    }

    let mut b_coeffs = vec![0u64; d];
    for (slot, lwe) in lwe_ciphertexts.iter().enumerate() {
        if slot < d {
            b_coeffs[slot] = lwe.b;
        }
    }

    RlweCiphertext::from_parts(
        Poly::zero_moduli(d, moduli),
        Poly::from_coeffs_moduli(b_coeffs, moduli),
    )
}

/// Key-switch each LWE from s_lwe to s_rlwe, shift by X^i, and sum.
pub fn pack_lwe_to_rlwe(
    lwe_ciphertexts: &[LweCiphertext],
    ks_matrix: &KeySwitchingMatrix,
    params: &InspireParams,
) -> RlweCiphertext {
    let d = params.ring_dim;
    let q = params.q;
    let ctx = params.ntt_context();

    if lwe_ciphertexts.is_empty() {
        return RlweCiphertext::from_parts(
            Poly::zero_moduli(d, params.moduli()),
            Poly::zero_moduli(d, params.moduli()),
        );
    }

    let mut result_a = Poly::zero_moduli(d, params.moduli());
    let mut result_b = Poly::zero_moduli(d, params.moduli());

    for (slot, lwe) in lwe_ciphertexts.iter().enumerate() {
        let rlwe_switched = lwe_to_rlwe_keyswitch(lwe, ks_matrix, &ctx, params);

        let shifted = if slot == 0 {
            rlwe_switched
        } else {
            mul_rlwe_by_monomial(&rlwe_switched, slot, q)
        };

        result_a = &result_a + &shifted.a;
        result_b = &result_b + &shifted.b;
    }

    RlweCiphertext::from_parts(result_a, result_b)
}

/// Inverse of `sample_extract_coeff0`: [a[0], -a[d-1], ..., -a[1]], which makes
/// <a, s_lwe> equal the constant term of perm(a)(X) * s_rlwe(X).
fn negacyclic_perm(a: &[u64], q: u64) -> Vec<u64> {
    let d = a.len();
    let mut out = vec![0u64; d];

    out[0] = a[0];

    for i in 1..d {
        let val = a[d - i];
        out[i] = if val == 0 { 0 } else { q - val };
    }

    out
}

/// Re-encrypt one LWE ciphertext under s_rlwe, message in coefficient 0.
///
/// Relies on s_lwe being the sample-extract image of s_rlwe, so the a-vector must go
/// through `negacyclic_perm` first.
fn lwe_to_rlwe_keyswitch(
    lwe: &LweCiphertext,
    ks_matrix: &KeySwitchingMatrix,
    ctx: &NttContext,
    params: &InspireParams,
) -> RlweCiphertext {
    let d = params.ring_dim;
    let q = params.q;
    let moduli = params.moduli();

    let a_perm = negacyclic_perm(&lwe.a, q);
    let a_poly = Poly::from_coeffs_moduli(a_perm, moduli);

    let mut b_coeffs = vec![0u64; d];
    b_coeffs[0] = lwe.b;
    let b_poly = Poly::from_coeffs_moduli(b_coeffs, moduli);

    let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, q);
    let a_decomp = gadget_decompose(&a_poly, &gadget);

    let mut result_a = Poly::zero_moduli(d, moduli);
    let mut result_b = b_poly;

    for (i, digit_poly) in a_decomp.iter().enumerate() {
        if i < ks_matrix.len() {
            let ks_row = ks_matrix.get_row(i);

            let term_a = digit_poly.mul_ntt(&ks_row.a, ctx);
            result_a = &result_a + &term_a;

            let term_b = digit_poly.mul_ntt(&ks_row.b, ctx);
            result_b = &result_b + &term_b;
        }
    }

    RlweCiphertext::from_parts(result_a, result_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{GaussianSampler, ModQ};
    use crate::rlwe::RlweSecretKey;

    fn test_params() -> InspireParams {
        InspireParams {
            ring_dim: 256,
            q: 1152921504606830593,
            crt_moduli: vec![1152921504606830593],
            p: 65536,
            sigma: 6.4,
            gadget_base: 1 << 20,
            gadget_len: 3,
            security_level: crate::params::SecurityLevel::Bits128,
        }
    }

    fn sample_error_poly(dim: usize, q: u64, sampler: &mut GaussianSampler) -> Poly {
        let coeffs: Vec<u64> = (0..dim)
            .map(|_| {
                let sample = sampler.sample();
                ModQ::from_signed(sample, q)
            })
            .collect();
        Poly::from_coeffs(coeffs, q)
    }

    #[test]
    fn test_pack_rlwe_coeffs_single() {
        let params = test_params();
        let d = params.ring_dim;
        let q = params.q;
        let delta = params.delta();
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let rlwe_sk = RlweSecretKey::generate(&params, &mut sampler);

        let message = 12345u64;
        let mut msg_coeffs = vec![0u64; d];
        msg_coeffs[0] = message;
        let msg_poly = Poly::from_coeffs(msg_coeffs, q);
        let a = Poly::random(d, q);
        let error = sample_error_poly(d, q, &mut sampler);
        let rlwe_ct = RlweCiphertext::encrypt(&rlwe_sk, &msg_poly, delta, a, &error, &ctx);

        let decrypted = rlwe_ct.decrypt(&rlwe_sk, delta, params.p, &ctx);
        assert_eq!(
            decrypted.coeff(0),
            message,
            "Original RLWE decryption failed"
        );

        let packed = pack_rlwe_coeffs(&[rlwe_ct], &params);
        let packed_decrypted = packed.decrypt(&rlwe_sk, delta, params.p, &ctx);

        assert_eq!(
            packed_decrypted.coeff(0),
            message,
            "Packed RLWE coefficient 0 should contain the message"
        );
    }

    #[test]
    fn test_pack_rlwe_coeffs_multiple() {
        let params = test_params();
        let d = params.ring_dim;
        let q = params.q;
        let delta = params.delta();
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let rlwe_sk = RlweSecretKey::generate(&params, &mut sampler);

        let messages: Vec<u64> = vec![100, 200, 300, 400];
        let rlwe_cts: Vec<_> = messages
            .iter()
            .map(|&msg| {
                let mut msg_coeffs = vec![0u64; d];
                msg_coeffs[0] = msg;
                let msg_poly = Poly::from_coeffs(msg_coeffs, q);
                let a = Poly::random(d, q);
                let error = sample_error_poly(d, q, &mut sampler);
                RlweCiphertext::encrypt(&rlwe_sk, &msg_poly, delta, a, &error, &ctx)
            })
            .collect();

        let packed = pack_rlwe_coeffs(&rlwe_cts, &params);
        let packed_decrypted = packed.decrypt(&rlwe_sk, delta, params.p, &ctx);

        for (i, &expected_msg) in messages.iter().enumerate() {
            assert_eq!(
                packed_decrypted.coeff(i),
                expected_msg,
                "Coefficient {} mismatch: expected {}, got {}",
                i,
                expected_msg,
                packed_decrypted.coeff(i)
            );
        }
    }

    #[test]
    fn test_mul_poly_by_monomial() {
        let d = 8;
        let q = 1152921504606830593u64;

        let mut coeffs = vec![0u64; d];
        coeffs[0] = 1;
        let poly = Poly::from_coeffs(coeffs, q);

        let shifted = mul_poly_by_monomial(&poly, 3, q);
        assert_eq!(shifted.coeff(3), 1);
        assert_eq!(shifted.coeff(0), 0);

        let mut coeffs2 = vec![0u64; d];
        coeffs2[d - 1] = 1;
        let poly2 = Poly::from_coeffs(coeffs2, q);

        let wrapped = mul_poly_by_monomial(&poly2, 1, q);
        assert_eq!(wrapped.coeff(0), q - 1);
    }

    #[test]
    fn test_pack_lwe_to_rlwe_single() {
        use crate::ks::generate_packing_ks_matrix;
        use crate::lwe::LweSecretKey;
        use crate::rgsw::GadgetVector;

        let params = test_params();
        let d = params.ring_dim;
        let q = params.q;
        let delta = params.delta();
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let rlwe_sk = RlweSecretKey::generate(&params, &mut sampler);
        let lwe_sk = LweSecretKey::from_rlwe(&rlwe_sk);

        let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, q);
        let packing_ks = generate_packing_ks_matrix(&lwe_sk, &rlwe_sk, &gadget, &mut sampler, &ctx);

        let message = 12345u64;
        let mut msg_coeffs = vec![0u64; d];
        msg_coeffs[0] = message;
        let msg_poly = Poly::from_coeffs(msg_coeffs, q);

        let a = Poly::random(d, q);
        let error_coeffs: Vec<u64> = (0..d)
            .map(|_| ModQ::from_signed(sampler.sample(), q))
            .collect();
        let error = Poly::from_coeffs(error_coeffs, q);
        let rlwe_ct = RlweCiphertext::encrypt(&rlwe_sk, &msg_poly, delta, a, &error, &ctx);

        let lwe_ct = rlwe_ct.sample_extract_coeff0();

        let lwe_dec = lwe_ct.decrypt(&lwe_sk, delta, params.p);
        assert_eq!(
            lwe_dec, message,
            "LWE decrypt failed: got {lwe_dec}, expected {message}"
        );

        let packed = pack_lwe_to_rlwe(&[lwe_ct], &packing_ks, &params);

        let packed_dec = packed.decrypt(&rlwe_sk, delta, params.p, &ctx);

        assert_eq!(
            packed_dec.coeff(0),
            message,
            "Packed RLWE decrypt failed: got {}, expected {}",
            packed_dec.coeff(0),
            message
        );
    }

    // No multi-LWE case here: a key-switched RLWE carries noise in every coefficient,
    // so shift-and-add mixes it. Multi-LWE packing goes through
    // `automorph_pack::pack_lwes`.
}
