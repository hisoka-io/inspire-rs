//! Core `Poly` arithmetic under 2-CRT, layer by layer, so a 2-CRT round-trip
//! failure can be localised below or above the primitives.

#![allow(
    clippy::needless_range_loop,
    reason = "lane-parallel KAT loops index two buffers in lockstep"
)]

use raven_inspire::math::{GaussianSampler, NttContext, Poly};
use raven_inspire::params::DEFAULT_Q_2CRT_30BIT;

/// Naive negacyclic multiply in u128, independent of `Poly` and the NTT.
fn naive_negacyclic_mul(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    let n = a.len();
    assert_eq!(b.len(), n);
    let q_u128 = q as u128;
    let mut result = vec![0u128; n];
    for i in 0..n {
        for j in 0..n {
            let idx = (i + j) % n;
            let sign_neg = (i + j) >= n;
            let prod = (a[i] as u128) * (b[j] as u128) % q_u128;
            if sign_neg {
                result[idx] = (result[idx] + q_u128 - prod) % q_u128;
            } else {
                result[idx] = (result[idx] + prod) % q_u128;
            }
        }
    }
    result.iter().map(|&c| c as u64).collect()
}

#[test]
fn two_crt_get_set_coeff_roundtrip_at_d256() {
    let moduli = DEFAULT_Q_2CRT_30BIT.to_vec();
    let q: u64 = moduli.iter().product();
    let mut poly = Poly::zero_moduli(256, &moduli);
    // near q/2, so both CRT limbs carry a non-trivial residue
    let target: u64 = q / 2 + 12345;
    poly.set_coeff(17, target);
    let recovered = poly.coeff(17);
    assert_eq!(
        recovered, target,
        "set_coeff / coeff must roundtrip under 2-CRT"
    );
}

#[test]
fn two_crt_scalar_mul_vs_naive() {
    let moduli = DEFAULT_Q_2CRT_30BIT.to_vec();
    let q: u64 = moduli.iter().product();

    let mut poly = Poly::zero_moduli(8, &moduli);
    for i in 0..8 {
        poly.set_coeff(i, (i as u64 * 1_000_000) % q);
    }
    let scalar: u64 = 1 << 20; // typical gadget_base
    let product = poly.scalar_mul(scalar);
    for i in 0..8 {
        let expected = ((i as u128 * 1_000_000 * scalar as u128) % q as u128) as u64;
        let got = product.coeff(i);
        assert_eq!(
            got, expected,
            "scalar_mul mismatch @ i={i}: {got} vs {expected}"
        );
    }
}

#[test]
fn two_crt_mul_ntt_vs_naive_at_d256() {
    let moduli = DEFAULT_Q_2CRT_30BIT.to_vec();
    let q: u64 = moduli.iter().product();
    let ctx = NttContext::with_moduli(256, &moduli);

    let a_vals: Vec<u64> = (0..256u64).map(|i| (i * 97 + 3) % q).collect();
    let b_vals: Vec<u64> = (0..256u64).map(|i| (i * 53 + 11) % q).collect();

    let mut a = Poly::zero_moduli(256, &moduli);
    let mut b = Poly::zero_moduli(256, &moduli);
    for i in 0..256 {
        a.set_coeff(i, a_vals[i]);
        b.set_coeff(i, b_vals[i]);
    }

    let product = a.mul_ntt(&b, &ctx);
    let expected = naive_negacyclic_mul(&a_vals, &b_vals, q);

    for i in 0..256 {
        let got = product.coeff(i);
        assert_eq!(
            got, expected[i],
            "mul_ntt mismatch @ i={i}: got={got}, expected={}",
            expected[i]
        );
    }
}

#[test]
fn two_crt_to_ntt_from_ntt_identity() {
    let moduli = DEFAULT_Q_2CRT_30BIT.to_vec();
    let q: u64 = moduli.iter().product();
    let ctx = NttContext::with_moduli(256, &moduli);

    let mut poly = Poly::zero_moduli(256, &moduli);
    for i in 0..256 {
        poly.set_coeff(i, (i as u64 * 12345 + 678) % q);
    }
    let original: Vec<u64> = (0..256).map(|i| poly.coeff(i)).collect();

    poly.to_ntt(&ctx);
    poly.from_ntt(&ctx);

    let recovered: Vec<u64> = (0..256).map(|i| poly.coeff(i)).collect();
    assert_eq!(original, recovered, "to_ntt -> from_ntt must be identity");
}

#[test]
fn two_crt_rlwe_encrypt_decrypt_roundtrip() {
    use raven_inspire::params::{InspireParams, SecurityLevel};
    use raven_inspire::rlwe::{RlweCiphertext, RlweSecretKey};

    let params = InspireParams {
        ring_dim: 256,
        q: 1_073_479_681 * 1_073_692_673,
        crt_moduli: vec![1_073_479_681, 1_073_692_673],
        p: 65537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    };
    params.validate().expect("2-CRT params must validate");
    let ctx = params.ntt_context();
    let delta = params.delta();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

    let sk = RlweSecretKey::generate(&params, &mut sampler);

    let mut msg = Poly::zero_moduli(256, params.moduli());
    for i in 0..256 {
        msg.set_coeff(i, (i as u64 * 101 + 3) % params.p);
    }

    let a_random = Poly::random_moduli(256, params.moduli());
    let error = Poly::sample_gaussian_moduli(256, params.moduli(), &mut sampler);
    let ct = RlweCiphertext::encrypt(&sk, &msg, delta, a_random, &error, &ctx);
    let recovered = ct.decrypt(&sk, delta, params.p, &ctx);

    for i in 0..256 {
        let got = recovered.coeff(i);
        let expected = msg.coeff(i);
        assert_eq!(
            got, expected,
            "RLWE roundtrip @ i={i}: got={got}, expected={expected}"
        );
    }
}

#[test]
fn two_crt_trivial_encrypt_decrypt_roundtrip() {
    // noiseless (0, delta*m), so only scalar_mul + decrypt rounding is in play
    use raven_inspire::params::{InspireParams, SecurityLevel};
    use raven_inspire::rlwe::{RlweCiphertext, RlweSecretKey};

    let params = InspireParams {
        ring_dim: 256,
        q: 1_073_479_681 * 1_073_692_673,
        crt_moduli: vec![1_073_479_681, 1_073_692_673],
        p: 65537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    };
    let ctx = params.ntt_context();
    let delta = params.delta();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

    let sk = RlweSecretKey::generate(&params, &mut sampler);

    let mut msg = Poly::zero_moduli(256, params.moduli());
    for i in 0..256 {
        msg.set_coeff(i, (i as u64 * 17 + 5) % params.p);
    }

    let ct = RlweCiphertext::trivial_encrypt(&msg, delta, &params);
    let recovered = ct.decrypt(&sk, delta, params.p, &ctx);

    for i in 0..256 {
        let got = recovered.coeff(i);
        let expected = msg.coeff(i);
        assert_eq!(got, expected, "trivial encrypt @ i={i}");
    }
}

#[test]
fn two_crt_external_product_identity() {
    use raven_inspire::params::{InspireParams, SecurityLevel};
    use raven_inspire::rgsw::{external_product, GadgetVector, RgswCiphertext};
    use raven_inspire::rlwe::{RlweCiphertext, RlweSecretKey};

    let params = InspireParams {
        ring_dim: 256,
        q: 1_073_479_681 * 1_073_692_673,
        crt_moduli: vec![1_073_479_681, 1_073_692_673],
        p: 65537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    };
    let ctx = params.ntt_context();
    let delta = params.delta();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

    let sk = RlweSecretKey::generate(&params, &mut sampler);
    let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, params.q);

    let mut msg = Poly::zero_moduli(256, params.moduli());
    for i in 0..256 {
        msg.set_coeff(i, (i as u64 * 3 + 7) % 100);
    }

    let a_random2 = Poly::random_moduli(256, params.moduli());
    let error2 = Poly::sample_gaussian_moduli(256, params.moduli(), &mut sampler);
    let rlwe = RlweCiphertext::encrypt(&sk, &msg, delta, a_random2, &error2, &ctx);

    // RGSW(1) must leave the RLWE plaintext unchanged up to noise
    let mut one = Poly::zero_moduli(256, params.moduli());
    one.set_coeff(0, 1);
    let rgsw = RgswCiphertext::encrypt(&sk, &one, &gadget, &mut sampler, &ctx);

    let product = external_product(&rlwe, &rgsw, &ctx);
    let recovered = product.decrypt(&sk, delta, params.p, &ctx);

    for i in 0..256 {
        let got = recovered.coeff(i);
        let expected = msg.coeff(i);
        assert_eq!(
            got, expected,
            "external_product with RGSW(1) should preserve message @ i={i}: got={got}, expected={expected}"
        );
    }
}

#[test]
fn two_crt_seeded_rgsw_expand_external_product_identity() {
    use raven_inspire::params::{InspireParams, SecurityLevel};
    use raven_inspire::rgsw::{external_product, GadgetVector, SeededRgswCiphertext};
    use raven_inspire::rlwe::{RlweCiphertext, RlweSecretKey};

    let params = InspireParams {
        ring_dim: 256,
        q: 1_073_479_681 * 1_073_692_673,
        crt_moduli: vec![1_073_479_681, 1_073_692_673],
        p: 65537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    };
    let ctx = params.ntt_context();
    let delta = params.delta();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

    let sk = RlweSecretKey::generate(&params, &mut sampler);
    let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, params.q);

    let mut msg = Poly::zero_moduli(256, params.moduli());
    for i in 0..256 {
        msg.set_coeff(i, (i as u64 * 3 + 7) % 100);
    }

    let a_random = Poly::random_moduli(256, params.moduli());
    let error = Poly::sample_gaussian_moduli(256, params.moduli(), &mut sampler);
    let rlwe = RlweCiphertext::encrypt(&sk, &msg, delta, a_random, &error, &ctx);

    // seeded encrypt -> expand, the path `query_seeded` takes
    let mut one = Poly::zero_moduli(256, params.moduli());
    one.set_coeff(0, 1);
    let seeded_rgsw = SeededRgswCiphertext::encrypt(&sk, &one, &gadget, &mut sampler, &ctx);
    let expanded_rgsw = seeded_rgsw.expand();

    let product = external_product(&rlwe, &expanded_rgsw, &ctx);
    let recovered = product.decrypt(&sk, delta, params.p, &ctx);

    for i in 0..256 {
        let got = recovered.coeff(i);
        let expected = msg.coeff(i);
        assert_eq!(
            got, expected,
            "seeded-RGSW expand -> external_product @ i={i}: got={got}, expected={expected}"
        );
    }
}

// Renamed from two_crt_inspiring_packing_offline_online_identity (2026-09-06
// mutation audit): with no reference plaintext there is no identity to assert,
// and a wrong-automorphism mutation in packing_offline survived this test while
// only respond_byte_identity_kat's golden caught it. GAP: no direct oracle pins
// packing_offline/packing_online output at 2-CRT moduli.
#[test]
fn two_crt_inspiring_packing_shape_and_nonconstant_smoke() {
    use raven_inspire::inspiring::{
        packing_offline, packing_online, ClientPackingKeys, OfflinePackingKeys, PackParams,
    };
    use raven_inspire::params::{InspireParams, SecurityLevel};
    use raven_inspire::rlwe::RlweSecretKey;

    let params = InspireParams {
        ring_dim: 256,
        q: 1_073_479_681 * 1_073_692_673,
        crt_moduli: vec![1_073_479_681, 1_073_692_673],
        p: 65537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    };
    params.validate().expect("2-CRT params must validate");

    let ctx = params.ntt_context();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let num_columns = 16usize;

    let pack_params =
        PackParams::try_new(&params, num_columns).expect("num_columns must be a legal width");
    let w_seed = [7u8; 32];
    let offline_keys = OfflinePackingKeys::generate(&pack_params, w_seed);

    let rlwe_sk = RlweSecretKey::generate(&params, &mut sampler);
    let client_keys = ClientPackingKeys::generate(&rlwe_sk, &pack_params, w_seed, &mut sampler);

    let a_ct_tilde: Vec<Poly> = (0..num_columns)
        .map(|_| Poly::random_moduli(256, params.moduli()))
        .collect();

    let mut b_poly = Poly::zero_moduli(256, params.moduli());
    for i in 0..256 {
        b_poly.set_coeff(i, (i as u64 * 97) % params.p);
    }

    let precomp = packing_offline(&pack_params, &offline_keys, &a_ct_tilde, &ctx);
    let packed = packing_online(&precomp, &client_keys.y_all, &b_poly, &ctx);

    // no reference plaintext exists for these synthetic inputs, so only shape
    // and range are asserted
    assert_eq!(packed.ring_dim(), 256);
    assert_eq!(packed.a.moduli(), params.moduli());
    assert_eq!(packed.b.moduli(), params.moduli());

    let recovered = packed.decrypt(&rlwe_sk, params.delta(), params.p, &ctx);
    let coeffs: Vec<u64> = (0..256).map(|i| recovered.coeff(i)).collect();
    let all_zero = coeffs.iter().all(|&c| c == 0);
    let all_max = coeffs.iter().all(|&c| c == params.p - 1);
    assert!(!all_zero, "packed decrypts to all-zero (pipeline broken)");
    assert!(!all_max, "packed decrypts to all-max");
}

#[test]
fn two_crt_gaussian_sample_then_scale_roundtrip() {
    // a per-limb-independent sampler would corrupt once scaled by delta
    let moduli = DEFAULT_Q_2CRT_30BIT.to_vec();
    let q: u64 = moduli.iter().product();

    let mut sampler = GaussianSampler::with_seed(6.4, 0);
    let sample = Poly::sample_gaussian_moduli(256, &moduli, &mut sampler);
    for i in 0..256 {
        let c = sample.coeff(i);
        // small, or near q in the negative representation
        let bound = 1000u64;
        let is_small = c < bound;
        let is_near_q = c > q - bound;
        assert!(
            is_small || is_near_q,
            "Gaussian sample coeff @ i={i} = {c} not in |small| < 1000; q = {q}"
        );
    }
}
