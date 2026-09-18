//! Key-switching operation.

use crate::math::{NttContext, Poly};
use crate::rgsw::gadget_decompose;
use crate::rlwe::RlweCiphertext;

use super::setup::KeySwitchingMatrix;

/// Gadget-decomposes `a` and returns `(0, b) + sum_i a_i * K[i]`, valid under s'
/// and decrypting to the same message.
pub fn key_switch(
    ct: &RlweCiphertext,
    ks_matrix: &KeySwitchingMatrix,
    ctx: &NttContext,
) -> RlweCiphertext {
    let d = ct.ring_dim();
    let moduli = ct.a.moduli();
    let gadget = &ks_matrix.gadget;
    let ell = gadget.len;

    let a_decomp = gadget_decompose(&ct.a, gadget);
    assert!(
        ks_matrix.rows.len() >= ell,
        "key switching matrix has {} rows but gadget length is {}",
        ks_matrix.rows.len(),
        ell
    );
    assert!(
        a_decomp.len() >= ell,
        "gadget decomposition has {} parts but gadget length is {}",
        a_decomp.len(),
        ell
    );

    let mut result_a = Poly::zero_moduli(d, moduli);
    let mut result_b = ct.b.clone();

    for (a_decomp_i, ks_row) in a_decomp[..ell].iter().zip(ks_matrix.rows[..ell].iter()) {
        let term_a = a_decomp_i.mul_ntt(&ks_row.a, ctx);
        result_a += term_a;

        let term_b = a_decomp_i.mul_ntt(&ks_row.b, ctx);
        result_b += term_b;
    }

    RlweCiphertext::from_parts(result_a, result_b)
}

/// [`key_switch`] for rows already in NTT domain.
#[allow(dead_code)]
pub fn key_switch_ntt(
    ct: &RlweCiphertext,
    ks_matrix: &KeySwitchingMatrix,
    ctx: &NttContext,
) -> RlweCiphertext {
    let d = ct.ring_dim();
    let moduli = ct.a.moduli();
    let gadget = &ks_matrix.gadget;
    let ell = gadget.len;

    let a_decomp = gadget_decompose(&ct.a, gadget);

    let a_decomp_ntt: Vec<Poly> = a_decomp
        .into_iter()
        .map(|mut p| {
            p.to_ntt(ctx);
            p
        })
        .collect();
    assert!(
        ks_matrix.rows.len() >= ell,
        "key switching matrix has {} rows but gadget length is {}",
        ks_matrix.rows.len(),
        ell
    );
    assert!(
        a_decomp_ntt.len() >= ell,
        "gadget decomposition has {} parts but gadget length is {}",
        a_decomp_ntt.len(),
        ell
    );

    let mut result_a = Poly::zero_moduli(d, moduli);
    result_a.to_ntt(ctx);

    let mut result_b = ct.b.clone();
    result_b.to_ntt(ctx);

    for (a_decomp_ntt_i, ks_row) in a_decomp_ntt[..ell].iter().zip(ks_matrix.rows[..ell].iter()) {
        let mut ks_a = ks_row.a.clone();
        let mut ks_b = ks_row.b.clone();
        ks_a.to_ntt(ctx);
        ks_b.to_ntt(ctx);

        let term_a = a_decomp_ntt_i.mul_ntt_domain(&ks_a, ctx);
        result_a += term_a;

        let term_b = a_decomp_ntt_i.mul_ntt_domain(&ks_b, ctx);
        result_b += term_b;
    }

    result_a.from_ntt(ctx);
    result_b.from_ntt(ctx);

    RlweCiphertext::from_parts(result_a, result_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ks::generate_ks_matrix;
    use crate::math::GaussianSampler;
    use crate::params::InspireParams;
    use crate::rgsw::GadgetVector;
    use crate::rlwe::RlweSecretKey;

    fn test_params() -> InspireParams {
        InspireParams::secure_128_d2048()
    }

    fn make_ctx(params: &InspireParams) -> NttContext {
        params.ntt_context()
    }

    fn sample_error_poly(dim: usize, moduli: &[u64], sampler: &mut GaussianSampler) -> Poly {
        Poly::sample_gaussian_moduli(dim, moduli, sampler)
    }

    #[test]
    fn test_key_switch_correctness() {
        let params = test_params();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
        let delta = params.delta();

        let sk1 = RlweSecretKey::generate(&params, &mut sampler);
        let sk2 = RlweSecretKey::generate(&params, &mut sampler);

        let gadget = GadgetVector::new(params.gadget_base, params.packing_gadget_len, params.q);
        let ks_matrix = generate_ks_matrix(&sk1, &sk2, &gadget, &mut sampler, &ctx);

        let msg_coeffs: Vec<u64> = (0..params.ring_dim)
            .map(|i| (i as u64) % params.p)
            .collect();
        let msg = Poly::from_coeffs_moduli(msg_coeffs.clone(), params.moduli());
        let a = Poly::random_moduli(params.ring_dim, params.moduli());
        let e = sample_error_poly(params.ring_dim, params.moduli(), &mut sampler);
        let ct1 = RlweCiphertext::encrypt(&sk1, &msg, delta, a, &e, &ctx);

        let dec1 = ct1.decrypt(&sk1, delta, params.p, &ctx);
        for (i, expected) in msg_coeffs.iter().enumerate().take(params.ring_dim) {
            assert_eq!(
                dec1.coeff(i),
                *expected,
                "Original decryption failed at {i}"
            );
        }

        let ct2 = key_switch(&ct1, &ks_matrix, &ctx);

        let dec2 = ct2.decrypt(&sk2, delta, params.p, &ctx);
        for (i, expected) in msg_coeffs.iter().enumerate().take(params.ring_dim) {
            assert_eq!(
                dec2.coeff(i),
                *expected,
                "Key-switched decryption failed at coefficient {i}"
            );
        }
    }

    #[test]
    fn test_key_switch_zero_message() {
        let params = test_params();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
        let delta = params.delta();

        let sk1 = RlweSecretKey::generate(&params, &mut sampler);
        let sk2 = RlweSecretKey::generate(&params, &mut sampler);

        let gadget = GadgetVector::new(params.gadget_base, params.packing_gadget_len, params.q);
        let ks_matrix = generate_ks_matrix(&sk1, &sk2, &gadget, &mut sampler, &ctx);

        let msg = Poly::zero_moduli(params.ring_dim, params.moduli());
        let a = Poly::random_moduli(params.ring_dim, params.moduli());
        let e = sample_error_poly(params.ring_dim, params.moduli(), &mut sampler);
        let ct1 = RlweCiphertext::encrypt(&sk1, &msg, delta, a, &e, &ctx);

        let ct2 = key_switch(&ct1, &ks_matrix, &ctx);

        let decrypted = ct2.decrypt(&sk2, delta, params.p, &ctx);

        for i in 0..params.ring_dim {
            assert_eq!(decrypted.coeff(i), 0, "Expected 0 at coefficient {i}");
        }
    }

    #[test]
    fn test_key_switch_same_key() {
        let params = test_params();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
        let delta = params.delta();

        let sk = RlweSecretKey::generate(&params, &mut sampler);

        let gadget = GadgetVector::new(params.gadget_base, params.packing_gadget_len, params.q);
        let ks_matrix = generate_ks_matrix(&sk, &sk, &gadget, &mut sampler, &ctx);

        let msg_coeffs: Vec<u64> = (0..params.ring_dim).map(|i| (i as u64) % 100).collect();
        let msg = Poly::from_coeffs_moduli(msg_coeffs.clone(), params.moduli());
        let a = Poly::random_moduli(params.ring_dim, params.moduli());
        let e = sample_error_poly(params.ring_dim, params.moduli(), &mut sampler);
        let ct = RlweCiphertext::encrypt(&sk, &msg, delta, a, &e, &ctx);

        let ct_switched = key_switch(&ct, &ks_matrix, &ctx);

        let decrypted = ct_switched.decrypt(&sk, delta, params.p, &ctx);

        for (i, expected) in msg_coeffs.iter().enumerate().take(params.ring_dim) {
            assert_eq!(decrypted.coeff(i), *expected, "Mismatch at coefficient {i}");
        }
    }

    #[test]
    fn test_key_switch_ntt_equivalence() {
        let params = test_params();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
        let delta = params.delta();

        let sk1 = RlweSecretKey::generate(&params, &mut sampler);
        let sk2 = RlweSecretKey::generate(&params, &mut sampler);

        let gadget = GadgetVector::new(params.gadget_base, params.packing_gadget_len, params.q);
        let ks_matrix = generate_ks_matrix(&sk1, &sk2, &gadget, &mut sampler, &ctx);

        let msg_coeffs: Vec<u64> = (0..params.ring_dim).map(|i| (i as u64) % 50).collect();
        let msg = Poly::from_coeffs_moduli(msg_coeffs.clone(), params.moduli());
        let a = Poly::random_moduli(params.ring_dim, params.moduli());
        let e = sample_error_poly(params.ring_dim, params.moduli(), &mut sampler);
        let ct = RlweCiphertext::encrypt(&sk1, &msg, delta, a, &e, &ctx);

        let ct_basic = key_switch(&ct, &ks_matrix, &ctx);
        let ct_ntt = key_switch_ntt(&ct, &ks_matrix, &ctx);

        let dec_basic = ct_basic.decrypt(&sk2, delta, params.p, &ctx);
        let dec_ntt = ct_ntt.decrypt(&sk2, delta, params.p, &ctx);

        for (i, expected) in msg_coeffs.iter().enumerate().take(params.ring_dim) {
            assert_eq!(dec_basic.coeff(i), *expected);
            assert_eq!(dec_ntt.coeff(i), *expected);
        }
    }

    #[test]
    fn test_key_switch_after_homomorphic_add() {
        let params = test_params();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
        let delta = params.delta();

        let sk1 = RlweSecretKey::generate(&params, &mut sampler);
        let sk2 = RlweSecretKey::generate(&params, &mut sampler);

        let gadget = GadgetVector::new(params.gadget_base, params.packing_gadget_len, params.q);
        let ks_matrix = generate_ks_matrix(&sk1, &sk2, &gadget, &mut sampler, &ctx);

        let msg1_coeffs: Vec<u64> = (0..params.ring_dim).map(|i| (i as u64) % 30).collect();
        let msg2_coeffs: Vec<u64> = (0..params.ring_dim)
            .map(|i| ((i + 10) as u64) % 30)
            .collect();

        let msg1 = Poly::from_coeffs_moduli(msg1_coeffs.clone(), params.moduli());
        let msg2 = Poly::from_coeffs_moduli(msg2_coeffs.clone(), params.moduli());

        let a1 = Poly::random_moduli(params.ring_dim, params.moduli());
        let e1 = sample_error_poly(params.ring_dim, params.moduli(), &mut sampler);
        let ct1 = RlweCiphertext::encrypt(&sk1, &msg1, delta, a1, &e1, &ctx);

        let a2 = Poly::random_moduli(params.ring_dim, params.moduli());
        let e2 = sample_error_poly(params.ring_dim, params.moduli(), &mut sampler);
        let ct2 = RlweCiphertext::encrypt(&sk1, &msg2, delta, a2, &e2, &ctx);

        let ct_sum = ct1.add(&ct2);

        let ct_switched = key_switch(&ct_sum, &ks_matrix, &ctx);

        let decrypted = ct_switched.decrypt(&sk2, delta, params.p, &ctx);

        for i in 0..params.ring_dim {
            let expected = (msg1_coeffs[i] + msg2_coeffs[i]) % params.p;
            assert_eq!(decrypted.coeff(i), expected, "Mismatch at coefficient {i}");
        }
    }
}
