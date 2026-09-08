//! Locks the serde contracts of the public API, including that secret-key
//! fields never reach the wire.

#![allow(
    clippy::unwrap_used,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]
#![allow(
    clippy::used_underscore_binding,
    clippy::no_effect_underscore_binding,
    reason = "underscore-prefixed bindings are deliberate compile-time type assertions"
)]

use raven_inspire::params::{InspireParams, InspireVariant, SecurityLevel, ShardConfig};
use raven_inspire::pir::{
    ClientQuery, ClientState, EncodedDatabase, InspireCrs, PackingMode, SeededClientQuery,
    ServerCrs, ServerResponse, ShardData,
};

fn test_params() -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: 1152921504606830593,
        crt_moduli: vec![1152921504606830593],
        p: 65536,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

fn assert_serialize<T: serde::Serialize>() {}
fn assert_deserialize<T: serde::de::DeserializeOwned>() {}
fn assert_clone<T: Clone>() {}
fn assert_debug<T: std::fmt::Debug>() {}

// These bounds fail `cargo check`, not the runtime run, so one body carries all
// eleven types; a dropped derive still names the offending type in the compiler
// error, and every violated bound reports at once.
#[test]
fn public_api_trait_bounds() {
    assert_serialize::<ClientQuery>();
    assert_deserialize::<ClientQuery>();
    assert_clone::<ClientQuery>();
    assert_debug::<ClientQuery>();

    assert_serialize::<SeededClientQuery>();
    assert_deserialize::<SeededClientQuery>();
    assert_clone::<SeededClientQuery>();
    assert_debug::<SeededClientQuery>();

    assert_serialize::<ClientState>();
    assert_deserialize::<ClientState>();
    assert_clone::<ClientState>();
    assert_debug::<ClientState>();

    assert_serialize::<ServerResponse>();
    assert_deserialize::<ServerResponse>();
    assert_clone::<ServerResponse>();
    assert_debug::<ServerResponse>();

    assert_serialize::<ServerCrs>();
    assert_deserialize::<ServerCrs>();
    assert_clone::<ServerCrs>();
    assert_debug::<ServerCrs>();

    assert_serialize::<InspireCrs>();
    assert_deserialize::<InspireCrs>();

    assert_serialize::<EncodedDatabase>();
    assert_deserialize::<EncodedDatabase>();
    assert_clone::<EncodedDatabase>();
    assert_debug::<EncodedDatabase>();

    assert_serialize::<ShardData>();
    assert_deserialize::<ShardData>();
    assert_clone::<ShardData>();
    assert_debug::<ShardData>();

    assert_serialize::<PackingMode>();
    assert_deserialize::<PackingMode>();
    assert_clone::<PackingMode>();
    assert_debug::<PackingMode>();
    fn assert_eq_copy_default<T: PartialEq + Eq + Copy + Default>() {}
    assert_eq_copy_default::<PackingMode>();

    assert_serialize::<InspireParams>();
    assert_deserialize::<InspireParams>();
    assert_clone::<InspireParams>();
    assert_debug::<InspireParams>();

    assert_serialize::<ShardConfig>();
    assert_deserialize::<ShardConfig>();
    assert_clone::<ShardConfig>();
    assert_debug::<ShardConfig>();
}

#[test]
fn serde_roundtrip_packing_mode_json() {
    for mode in [PackingMode::Inspiring, PackingMode::Tree] {
        let json = serde_json::to_string(&mode).unwrap();
        let back: PackingMode = serde_json::from_str(&json).unwrap();
        assert_eq!(mode, back);
    }
}

#[test]
fn serde_packing_mode_rename_all_snake_case() {
    let json = serde_json::to_string(&PackingMode::Inspiring).unwrap();
    assert_eq!(json, "\"inspiring\"");

    let json = serde_json::to_string(&PackingMode::Tree).unwrap();
    assert_eq!(json, "\"tree\"");
}

#[test]
fn serde_packing_mode_default_is_inspiring() {
    assert_eq!(PackingMode::default(), PackingMode::Inspiring);
}

#[test]
fn serde_roundtrip_inspire_params_json() {
    let params = test_params();
    let json = serde_json::to_string(&params).unwrap();
    let back: InspireParams = serde_json::from_str(&json).unwrap();
    assert_eq!(back, params);
}

#[test]
fn serde_roundtrip_shard_config_json() {
    let config = ShardConfig {
        shard_size_bytes: 8192,
        entry_size_bytes: 32,
        total_entries: 256,
    };
    let json = serde_json::to_string(&config).unwrap();
    let back: ShardConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back, config);
}

#[test]
fn client_state_secret_keys_not_serialized() {
    use raven_inspire::math::GaussianSampler;
    use raven_inspire::pir::setup;

    let params = test_params();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

    let entry_size = 32;
    let num_entries = params.ring_dim;
    let database: Vec<u8> = (0..(num_entries * entry_size))
        .map(|i| (i % 256) as u8)
        .collect();

    let (crs, encoded_db, rlwe_sk) = setup(&params, &database, entry_size, &mut sampler).unwrap();

    let (_state, _query) =
        raven_inspire::pir::query(&crs, 42, &encoded_db.config, &rlwe_sk, &mut sampler).unwrap();

    let json = serde_json::to_string(&_state).unwrap();
    let recovered: ClientState = serde_json::from_str(&json).unwrap();

    assert_eq!(
        recovered.secret_key.dim, 0,
        "LWE secret key must be zeroed after serde round-trip"
    );
    assert_eq!(
        recovered.rlwe_secret_key.ring_dim(),
        0,
        "RLWE secret key must be zeroed after serde round-trip"
    );

    assert_eq!(recovered.index, _state.index);
    assert_eq!(recovered.shard_id, _state.shard_id);
    assert_eq!(recovered.local_index, _state.local_index);
}

#[test]
fn client_state_secret_keys_absent_from_json() {
    use raven_inspire::math::GaussianSampler;
    use raven_inspire::pir::setup;

    let params = test_params();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

    let entry_size = 32;
    let num_entries = params.ring_dim;
    let database: Vec<u8> = (0..(num_entries * entry_size))
        .map(|i| (i % 256) as u8)
        .collect();

    let (crs, encoded_db, rlwe_sk) = setup(&params, &database, entry_size, &mut sampler).unwrap();

    let (state, _query) =
        raven_inspire::pir::query(&crs, 42, &encoded_db.config, &rlwe_sk, &mut sampler).unwrap();

    let json = serde_json::to_string(&state).unwrap();

    assert!(
        !json.contains("secret_key"),
        "JSON must not contain 'secret_key' field: {json}"
    );
    assert!(
        !json.contains("rlwe_secret_key"),
        "JSON must not contain 'rlwe_secret_key' field: {json}"
    );
}

#[test]
fn server_crs_skipped_fields_absent_from_json() {
    use raven_inspire::math::GaussianSampler;
    use raven_inspire::pir::setup;

    let params = test_params();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

    let entry_size = 32;
    let num_entries = params.ring_dim;
    let database: Vec<u8> = (0..(num_entries * entry_size))
        .map(|i| (i % 256) as u8)
        .collect();

    let (crs, _encoded_db, _rlwe_sk) = setup(&params, &database, entry_size, &mut sampler).unwrap();

    assert!(
        crs.inspiring_pack_params.is_some(),
        "pack_params should be set after setup"
    );
    assert!(
        crs.inspiring_packing_key.is_some(),
        "packing_key should be set after setup"
    );

    let json = serde_json::to_string(&crs).unwrap();

    assert!(
        !json.contains("inspiring_pack_params"),
        "JSON must not contain 'inspiring_pack_params'"
    );
    assert!(
        !json.contains("inspiring_packing_key"),
        "JSON must not contain 'inspiring_packing_key'"
    );
}

#[test]
fn server_crs_skipped_fields_none_after_roundtrip() {
    use raven_inspire::math::GaussianSampler;
    use raven_inspire::pir::setup;

    let params = test_params();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

    let entry_size = 32;
    let num_entries = params.ring_dim;
    let database: Vec<u8> = (0..(num_entries * entry_size))
        .map(|i| (i % 256) as u8)
        .collect();

    let (crs, _encoded_db, _rlwe_sk) = setup(&params, &database, entry_size, &mut sampler).unwrap();

    let json = serde_json::to_string(&crs).unwrap();
    let recovered: ServerCrs = serde_json::from_str(&json).unwrap();

    assert!(
        recovered.inspiring_pack_params.is_none(),
        "inspiring_pack_params must be None after serde round-trip"
    );
    assert!(
        recovered.inspiring_packing_key.is_none(),
        "inspiring_packing_key must be None after serde round-trip"
    );

    assert_eq!(recovered.params.ring_dim, crs.params.ring_dim);
    assert_eq!(recovered.inspiring_w_seed, crs.inspiring_w_seed);
    assert_eq!(recovered.inspiring_v_seed, crs.inspiring_v_seed);
    assert_eq!(recovered.inspiring_num_columns, crs.inspiring_num_columns);
}

// A bincode round-trip test asserting only ring_dim and column count lived
// here, next to a JSON sibling with the same two-field spot-check; both
// survived a mutation that zeroed every serialized Poly coefficient
// (2026-09-06 mutation audits). The bincode wire is now guarded by
// bincode_roundtrip_audit.rs's server_response_packing_mode_all_variants test,
// which round-trips all three packing_mode states and compares the full
// coefficient content; payload fidelity is additionally pinned by
// poly_wire_shape_refusal.rs and the respond_byte_identity_kat golden.

#[test]
fn client_query_packing_mode_defaults_to_inspiring() {
    let mut value = serde_json::json!({
        "shard_id": 0,
        "packing_mode": "tree",
        "rgsw_ciphertext": {
            "rows": [],
            "gadget": {"base": 1048576u64, "len": 3u64, "q": 1152921504606830593u64}
        }
    });
    value.as_object_mut().unwrap().remove("packing_mode");

    let query: ClientQuery = serde_json::from_value(value).unwrap();
    assert_eq!(query.packing_mode, PackingMode::Inspiring);
    assert!(query.inspiring_packing_keys.is_none());
}

#[test]
fn seeded_client_query_packing_mode_defaults_to_inspiring() {
    let mut value = serde_json::json!({
        "shard_id": 0,
        "packing_mode": "tree",
        "rgsw_ciphertext": {
            "rows": [],
            "gadget": {"base": 1048576u64, "len": 3u64, "q": 1152921504606830593u64}
        }
    });
    value.as_object_mut().unwrap().remove("packing_mode");

    let query: SeededClientQuery = serde_json::from_value(value).unwrap();
    assert_eq!(query.packing_mode, PackingMode::Inspiring);
    assert!(query.inspiring_packing_keys.is_none());
}

#[test]
fn reexport_extract_inspiring_accessible() {
    let _fn_ptr: fn(
        &ServerCrs,
        &ClientState,
        &ServerResponse,
        usize,
    ) -> raven_inspire::pir::Result<Vec<u8>> = raven_inspire::extract_inspiring;
    let _fn_ptr2: fn(
        &ServerCrs,
        &ClientState,
        &ServerResponse,
        usize,
    ) -> raven_inspire::pir::Result<Vec<u8>> = raven_inspire::pir::extract_inspiring;
}

#[test]
fn reexport_all_extract_variants_accessible() {
    type ExtractFn =
        fn(&ServerCrs, &ClientState, &ServerResponse, usize) -> raven_inspire::pir::Result<Vec<u8>>;
    type ExtractWithVariantFn = fn(
        &ServerCrs,
        &ClientState,
        &ServerResponse,
        usize,
        InspireVariant,
    ) -> raven_inspire::pir::Result<Vec<u8>>;

    let _: ExtractFn = raven_inspire::extract;
    let _: ExtractFn = raven_inspire::extract_inspiring;
    let _: ExtractFn = raven_inspire::extract_two_packing;
    let _: ExtractWithVariantFn = raven_inspire::extract_with_variant;
}

// Two enum-arity tautologies (assert_eq!(len, len) over local literals) lived
// here; both survived a variant-meaning swap plus an added variant
// (2026-09-06 mutation audit). NOTE the audit also showed the
// OnePacking/TwoPacking dispatch swap in extract_with_variant survives every
// live test - the only coverage is the two #[ignore]d e2e variant tests.

#[test]
fn api_setup_query_respond_extract_compiles() {
    use raven_inspire::math::GaussianSampler;
    use raven_inspire::pir::{extract, query, respond, setup};

    let params = test_params();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

    let entry_size = 32;
    let num_entries = params.ring_dim;
    let database: Vec<u8> = (0..(num_entries * entry_size))
        .map(|i| (i % 256) as u8)
        .collect();

    let (crs, encoded_db, rlwe_sk) = setup(&params, &database, entry_size, &mut sampler).unwrap();
    let (state, client_query) = query(&crs, 0, &encoded_db.config, &rlwe_sk, &mut sampler).unwrap();
    let response = respond(&crs, &encoded_db, &client_query).unwrap();
    let result = extract(&crs, &state, &response, entry_size).unwrap();

    assert_eq!(
        result,
        &database[..entry_size],
        "extract through the public API must return entry 0's bytes, not merely its length"
    );
}

// Three tests lived here and were deleted after mutation proofs (2026-09-06):
// - api_seeded_query_compiles asserted only its own input back (state.index==0)
//   and never inspected the expand() result; a corrupted expand() survived it
//   and was killed by e2e_pir::test_e2e_seeded_query.
// - client_query_json_roundtrip / seeded_client_query_json_roundtrip asserted
//   shard_id+packing_mode only; an RgswCiphertext that serialized empty rows
//   survived both. The live JSON-path guard is
//   http_inspiring::inspiring_query_response_json_round_trip, which byte-checks
//   the extracted entry after a JSON round trip of query and response.

// A per-mode trio of ServerResponse bincode round trips (packing_mode
// None/Tree/Inspiring) lived here, each paying its own full PIR setup. Their
// one real catch — `skip_serializing_if` on the tag EOFs bincode's positional
// decode — is carried by bincode_roundtrip_audit.rs's
// server_response_packing_mode_all_variants test, proven red under the same
// skip_serializing_if mutation that killed the None-mode trio member
// (2026-09-06 mutation audit; the Tree/Inspiring members survived it and
// asserted nothing the survivor does not).
