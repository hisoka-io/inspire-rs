//! The InspiRING query/response JSON codec, as an HTTP server would use it.
//!
//! Until 2026-09-06 this file was an axum round-trip gated on
//! `#![cfg(feature = "server")]` - a feature the Raven fork does not define -
//! so it compiled to an empty binary and ran zero tests, while its 400-path
//! assertion tested a check its own handler performed. What was worth keeping
//! is the serde property: a query and a response that cross a JSON wire must
//! still extract the right bytes. That is asserted here directly, with no HTTP
//! stack, and it is the only live guard against a JSON field drop or rename on
//! `ClientQuery` / `ServerResponse` (bincode stability cannot see those).

#![allow(
    clippy::unwrap_used,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::InspireParams;
use raven_inspire::pir::{
    extract_inspiring, query, respond_inspiring, setup, ClientQuery, PackingMode, ServerResponse,
};

fn test_params() -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: 1152921504606830593,
        crt_moduli: vec![1152921504606830593],
        p: 65536,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: raven_inspire::params::SecurityLevel::Bits128,
    }
}

#[test]
fn inspiring_query_response_json_round_trip() {
    let params = test_params();
    let d = params.ring_dim;

    let num_entries = d;
    let entry_size = 2; // 1 column per entry, values < 256

    let database: Vec<u8> = (0..num_entries)
        .flat_map(|i| {
            let low_byte = (i % 256) as u8;
            let high_byte = 0u8;
            vec![low_byte, high_byte]
        })
        .collect();

    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let (crs, encoded_db, rlwe_sk) =
        setup(&params, &database, entry_size, &mut sampler).expect("setup should succeed");

    let target_index = 42u64;
    let (state, mut client_query) = query(
        &crs,
        target_index,
        &encoded_db.config,
        &rlwe_sk,
        &mut sampler,
    )
    .expect("query should succeed");
    client_query.packing_mode = PackingMode::Inspiring;

    // The server must answer from the query AS DECODED, so a field the JSON
    // codec drops (rgsw rows, packing keys) breaks respond or the extraction.
    let query_json = serde_json::to_string(&client_query).expect("query to JSON");
    let wire_query: ClientQuery = serde_json::from_str(&query_json).expect("query from JSON");

    let response =
        respond_inspiring(&crs, &encoded_db, &wire_query).expect("respond should succeed");

    let response_json = serde_json::to_string(&response).expect("response to JSON");
    let wire_response: ServerResponse =
        serde_json::from_str(&response_json).expect("response from JSON");

    let extracted = extract_inspiring(&crs, &state, &wire_response, entry_size)
        .expect("extract should succeed");
    let expected_start = (target_index as usize) * entry_size;
    let expected = &database[expected_start..expected_start + entry_size];
    assert_eq!(
        extracted.as_slice(),
        expected,
        "entry bytes must survive the JSON wire in both directions"
    );
}
