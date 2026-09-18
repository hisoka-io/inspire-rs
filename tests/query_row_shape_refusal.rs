#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "test fixture failures must be loud"
)]

use raven_inspire::math::{GaussianSampler, Poly};
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::pir::respond_seeded_inspiring_cached_with_session;
use raven_inspire::{
    extract_two_packing, query_seeded, setup, SeededClientQuery, ServerInspiringCache,
};

fn params() -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65_537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

fn assert_forged_row_refused(forged_b: Poly, expected_error: &str) {
    let params = params();
    let entry_size = 32usize;
    let target = 5u64;
    let db: Vec<u8> = (0..params.ring_dim * entry_size)
        .map(|index| (index % 251) as u8)
        .collect();
    let expected_row = &db[target as usize * entry_size..(target as usize + 1) * entry_size];
    let mut sampler = GaussianSampler::with_seed(params.sigma, 23);
    let (crs, encoded, secret_key) =
        setup(&params, &db, entry_size, &mut sampler).expect("valid test setup");
    let (state, query) = query_seeded(&crs, target, &encoded.config, &secret_key, &mut sampler)
        .expect("valid test query");
    let cache = ServerInspiringCache::new(&crs, &encoded).expect("valid server cache");

    let honest = respond_seeded_inspiring_cached_with_session(&crs, &encoded, &query, &cache, None)
        .expect("honest query must respond");
    let honest_row =
        extract_two_packing(&crs, &state, &honest, entry_size).expect("honest query must extract");
    assert_eq!(honest_row.as_slice(), expected_row);

    let mut forged = query;
    forged.rgsw_ciphertext.rows[0].b = forged_b;
    let wire = bincode::serialize(&forged).expect("serialize forged query");
    let decoded: SeededClientQuery =
        bincode::deserialize(&wire).expect("self-consistent forged polynomial must decode");
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        respond_seeded_inspiring_cached_with_session(&crs, &encoded, &decoded, &cache, None)
    }));
    match outcome {
        Ok(Err(error)) => assert!(
            error.to_string().contains(expected_error),
            "wrong refusal: {error}"
        ),
        Ok(Ok(_)) => panic!("forged RGSW row returned Ok"),
        Err(panic_payload) => std::panic::resume_unwind(panic_payload),
    }
}

#[test]
fn wire_rgsw_row_with_wrong_dimension_is_refused_before_expansion() {
    let params = params();
    let forged_b = Poly::from_coeffs_moduli(vec![2u64; 1024], params.moduli());
    assert_forged_row_refused(forged_b, "RGSW row 0 b dimension 1024");
}

#[test]
fn wire_rgsw_row_with_wrong_modulus_is_refused_before_expansion() {
    let params = params();
    let forged_b = Poly::from_coeffs_moduli(vec![2u64; params.ring_dim], &[12_289]);
    assert_forged_row_refused(forged_b, "RGSW row 0 b moduli");
}
