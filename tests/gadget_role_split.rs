#![allow(clippy::expect_used)]

use raven_inspire::math::GaussianSampler;
use raven_inspire::{query_seeded, setup, InspireParams, SecurityLevel};

fn params(query_gadget_len: usize, packing_gadget_len: usize) -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: raven_inspire::math::mod_q::DEFAULT_Q,
        crt_moduli: vec![raven_inspire::math::mod_q::DEFAULT_Q],
        p: 65_537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len,
        packing_gadget_len,
        security_level: SecurityLevel::Bits128,
    }
}

fn assert_roles(query_len: usize, packing_len: usize) {
    let params = params(query_len, packing_len);
    let database = vec![0u8; params.ring_dim * 32];
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0x5150);
    let (crs, encoded, secret_key) = setup(&params, &database, 32, &mut sampler).expect("setup");
    let (_, query) =
        query_seeded(&crs, 0, &encoded.config, &secret_key, &mut sampler).expect("query");

    assert_eq!(crs.rgsw_gadget.len, query_len);
    assert_eq!(query.rgsw_ciphertext.gadget.len, 1);
    assert_eq!(query.rgsw_ciphertext.rows.len(), 1);
    assert!(crs
        .galois_keys
        .iter()
        .all(|key| key.gadget_len() == packing_len));
    assert_eq!(
        crs.inspiring_pack_params
            .as_ref()
            .expect("packing params")
            .gadget
            .len,
        packing_len
    );
    assert_eq!(
        query
            .inspiring_packing_keys
            .as_ref()
            .expect("packing keys")
            .y_body
            .len(),
        packing_len
    );
}

#[test]
fn parameter_gadgets_remain_independent_of_the_one_row_fold_query() {
    assert_roles(4, 3);
    assert_roles(3, 4);
}
