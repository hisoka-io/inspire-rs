//! The TwoPacking extractor accepts only InspiRING responses.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::pir::{
    extract_two_packing, query_seeded, respond_seeded_inspiring, setup, PackingMode,
};

fn fixture() -> (
    raven_inspire::ServerCrs,
    raven_inspire::ClientState,
    raven_inspire::ServerResponse,
) {
    let params = InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65_537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    };
    let entry_size = 32;
    let database: Vec<u8> = (0..params.ring_dim * entry_size)
        .map(|i| ((i * 17 + 3) % 251) as u8)
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 37);
    let (crs, encoded_db, sk) = setup(&params, &database, entry_size, &mut sampler).expect("setup");
    let (state, query) =
        query_seeded(&crs, 42, &encoded_db.config, &sk, &mut sampler).expect("seeded query");
    let response = respond_seeded_inspiring(&crs, &encoded_db, &query).expect("respond");
    (crs, state, response)
}

#[test]
fn two_packing_refuses_untagged_and_tree_packed_responses() {
    let (crs, state, response) = fixture();

    for (mode, decoded, rims_byte) in [(None, "None", 0u8), (Some(PackingMode::Tree), "Tree", 2u8)]
    {
        let mut malformed = response.clone();
        malformed.packing_mode = mode;
        let err = extract_two_packing(&crs, &state, &malformed, 32)
            .expect_err("TwoPacking must refuse a tree-packed response");
        let message = err.to_string();
        assert!(message.contains("TwoPacking extractor"), "{message}");
        assert!(message.contains("response is tree-packed"), "{message}");
        assert!(message.contains(decoded), "{message}");
        assert!(
            message.contains(&format!("RIMS tag byte {rims_byte}")),
            "{message}"
        );
    }
}
