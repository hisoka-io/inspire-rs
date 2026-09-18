//! The pre-NTT'd RGSW external product must be byte-identical to the classical
//! one; that identity is what licenses its use in the respond hot path.

use raven_inspire::math::{GaussianSampler, Poly};
use raven_inspire::params::InspireParams;
use raven_inspire::rgsw::{
    external_product, external_product_with_ntt_rgsw, rgsw_rows_to_ntt, GadgetVector,
    RgswCiphertext,
};
use raven_inspire::rlwe::{RlweCiphertext, RlweSecretKey};

fn params() -> InspireParams {
    InspireParams::secure_128_d2048()
}

fn sample_db_poly(dim: usize, moduli: &[u64], seed: u64) -> Poly {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let coeffs: Vec<u64> = (0..dim)
        .map(|_| {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            z
        })
        .collect();
    Poly::from_coeffs_moduli(coeffs, moduli)
}

// Two example tests (a fixed RGSW(0) case and a 64-seed random loop) were
// converted into the differential property below (2026-09-06); the
// dropped-gadget-digit mutant that killed both kills the property.

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig {
        cases: 6,
        failure_persistence: None,
        .. proptest::prelude::ProptestConfig::default()
    })]

    /// The pre-NTT'd external product must be byte-identical to the classical
    /// one for arbitrary trivial-RLWE b bytes and every scalar arm; the RGSW(0) case
    /// runs in every drawn case alongside the drawn scalar.
    #[test]
    fn external_product_ntt_matches_classical_property(
        seed in proptest::prelude::any::<u64>(),
        scalar in 1u64..=5,
    ) {
        let p = params();
        let ctx = p.ntt_context();
        let gadget = GadgetVector::new(p.gadget_base, p.query_gadget_len, p.q);

        let mut sampler = GaussianSampler::with_seed(p.sigma, seed);
        let sk = RlweSecretKey::generate(&p, &mut sampler);

        let b_poly = sample_db_poly(p.ring_dim, p.moduli(), seed.wrapping_mul(17).wrapping_add(2));
        let rlwe = RlweCiphertext::from_parts(
            Poly::zero_moduli(p.ring_dim, p.moduli()),
            b_poly,
        );

        for m in [0u64, scalar] {
            let rgsw = RgswCiphertext::encrypt_scalar(&sk, m, &gadget, &mut sampler, &ctx);
            let classical = external_product(&rlwe, &rgsw, &ctx)
                .expect("trivial RLWE is accepted by classical external product");
            let rgsw_ntt = rgsw_rows_to_ntt(&rgsw, &ctx);
            let fast = external_product_with_ntt_rgsw(&rlwe, &rgsw_ntt, &gadget, &ctx)
                .expect("trivial RLWE is accepted by NTT external product");

            proptest::prop_assert_eq!(
                classical.a.coeffs(),
                fast.a.coeffs(),
                "RGSW({}): .a components diverged (seed {})",
                m,
                seed
            );
            proptest::prop_assert_eq!(
                classical.b.coeffs(),
                fast.b.coeffs(),
                "RGSW({}): .b components diverged (seed {})",
                m,
                seed
            );
        }
    }
}

#[test]
fn external_product_ntt_preserves_correctness_rgsw_scalar() {
    let p = params();
    let ctx = p.ntt_context();
    let mut sampler = GaussianSampler::with_seed(p.sigma, 0);
    let delta = p.delta();

    let sk = RlweSecretKey::generate(&p, &mut sampler);
    let gadget = GadgetVector::new(p.gadget_base, p.query_gadget_len, p.q);

    let msg_coeffs: Vec<u64> = (0..p.ring_dim).map(|i| (i as u64) % 10).collect();
    let msg = Poly::from_coeffs_moduli(msg_coeffs.clone(), p.moduli());
    let rlwe = RlweCiphertext::trivial_encrypt(&msg, delta, &p);

    let scalar = 3u64;
    let rgsw = RgswCiphertext::encrypt_scalar(&sk, scalar, &gadget, &mut sampler, &ctx);
    let rgsw_ntt = rgsw_rows_to_ntt(&rgsw, &ctx);

    let result = external_product_with_ntt_rgsw(&rlwe, &rgsw_ntt, &gadget, &ctx)
        .expect("trivial RLWE is accepted");
    let decrypted = result.decrypt(&sk, delta, p.p, &ctx);

    for (i, msg_coeff) in msg_coeffs.iter().enumerate().take(p.ring_dim) {
        let expected = (*msg_coeff * scalar) % p.p;
        assert_eq!(decrypted.coeff(i), expected, "Mismatch at coefficient {i}");
    }
}

#[test]
fn one_sided_external_products_refuse_a_nontrivial_rlwe_input() {
    let p = params();
    let ctx = p.ntt_context();
    let mut sampler = GaussianSampler::with_seed(p.sigma, 0x1206);
    let sk = RlweSecretKey::generate(&p, &mut sampler);
    let gadget = GadgetVector::new(p.gadget_base, p.query_gadget_len, p.q);
    let message = Poly::constant_moduli(1, p.ring_dim, p.moduli());
    let rgsw = RgswCiphertext::encrypt_scalar(&sk, 1, &gadget, &mut sampler, &ctx);
    let rgsw_ntt = rgsw_rows_to_ntt(&rgsw, &ctx);
    let rlwe =
        RlweCiphertext::from_parts(Poly::constant_moduli(1, p.ring_dim, p.moduli()), message);

    let classical = external_product(&rlwe, &rgsw, &ctx)
        .expect_err("one-sided classical product must reject nonzero a");
    let fast = external_product_with_ntt_rgsw(&rlwe, &rgsw_ntt, &gadget, &ctx)
        .expect_err("one-sided NTT product must reject nonzero a");
    for message in [classical.to_string(), fast.to_string()] {
        assert!(message.contains("nonzero a"), "{message}");
        assert!(message.contains("trivial_encrypt"), "{message}");
    }
}
