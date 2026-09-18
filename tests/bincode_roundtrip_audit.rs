//! Bincode roundtrips every public-API `Option<T>` / `Vec<T>` field at both
//! edge states. `skip_serializing_if` omits the field entirely, which bincode's
//! positional format decodes as EOF, so these roundtrips fail if one returns.

#![allow(
    clippy::panic,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::rlwe::RlweSecretKey;
use raven_inspire::{
    extract_inspiring, query, query_seeded, respond, respond_inspiring, respond_one_packing,
    respond_seeded_inspiring, setup, EncodedDatabase, PackingMode, ServerCrs, ServerResponse,
    ServerSessionHandle,
};

fn small_params() -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

/// Shared d=256 / 32-byte-entry cell; the sampler comes back positioned right
/// after setup, exactly as each test built it inline before the merge.
fn fixture_32b() -> (ServerCrs, EncodedDatabase, RlweSecretKey, GaussianSampler) {
    let params = small_params();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let entry_size = 32;
    let n = params.ring_dim;
    let db: Vec<u8> = (0..n * entry_size).map(|i| (i % 256) as u8).collect();
    let (crs, encoded_db, sk) = setup(&params, &db, entry_size, &mut sampler)
        .unwrap_or_else(|e| panic!("fixture setup failed: {e:?}"));
    (crs, encoded_db, sk, sampler)
}

fn bincode_roundtrip_stable<T>(value: &T, label: &str)
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let bytes = bincode::serialize(value)
        .unwrap_or_else(|e| panic!("{label}: bincode::serialize failed: {e:?}"));
    let recovered: T = bincode::deserialize(&bytes)
        .unwrap_or_else(|e| panic!("{label}: bincode::deserialize failed: {e:?}"));
    let bytes2 = bincode::serialize(&recovered)
        .unwrap_or_else(|e| panic!("{label}: re-serialize failed: {e:?}"));
    assert_eq!(
        bytes, bytes2,
        "{label}: bincode re-serialize must be byte-stable"
    );
}

#[test]
fn server_response_packing_mode_all_variants_bincode_stable() {
    let (crs, encoded_db, sk, mut sampler) = fixture_32b();
    let (_state, client_query) = query(&crs, 42, &encoded_db.config, &sk, &mut sampler).unwrap();
    let responses = [
        (None, respond(&crs, &encoded_db, &client_query).unwrap()),
        (
            Some(PackingMode::Tree),
            respond_one_packing(&crs, &encoded_db, &client_query).unwrap(),
        ),
        (
            Some(PackingMode::Inspiring),
            respond_inspiring(&crs, &encoded_db, &client_query).unwrap(),
        ),
    ];

    for (mode, base) in responses {
        bincode_roundtrip_stable(&base, &format!("ServerResponse.packing_mode = {mode:?}"));

        // Stability alone is blind to a lossy-but-idempotent codec (a
        // serializer that zeroed every coefficient re-serializes byte-stably),
        // so the decoded ciphertext content is compared against the in-memory
        // original too.
        let bytes = base
            .to_binary()
            .unwrap_or_else(|e| panic!("serialize: {e:?}"));
        let recovered =
            ServerResponse::from_binary(&bytes).unwrap_or_else(|e| panic!("deserialize: {e:?}"));
        assert_eq!(recovered.packing_mode, base.packing_mode, "mode = {mode:?}");
        assert_eq!(
            recovered.packed_coefficients, base.packed_coefficients,
            "retained coefficient count must survive the wire (mode = {mode:?})"
        );
        assert_eq!(
            recovered.ciphertext.a.coeffs(),
            base.ciphertext.a.coeffs(),
            "ciphertext.a coefficients must survive the wire (mode = {mode:?})"
        );
        if let Some(retained) = base.packed_coefficients {
            let retained = retained as usize;
            let dim = base.ciphertext.ring_dim();
            for limb in 0..base.ciphertext.b.crt_count() {
                let start = limb * dim;
                assert_eq!(
                    &recovered.ciphertext.b.coeffs()[start..start + retained],
                    &base.ciphertext.b.coeffs()[start..start + retained],
                    "ciphertext.b prefix must survive the wire (mode = {mode:?})"
                );
                assert!(
                    recovered.ciphertext.b.coeffs()[start + retained..start + dim]
                        .iter()
                        .all(|coefficient| *coefficient == 0),
                    "ciphertext.b tail must be reconstructed as zero (mode = {mode:?})"
                );
            }
        } else {
            assert_eq!(
                recovered.ciphertext.b.coeffs(),
                base.ciphertext.b.coeffs(),
                "full ciphertext.b must survive the wire (mode = {mode:?})"
            );
        }
        assert_eq!(
            recovered.column_ciphertexts.len(),
            base.column_ciphertexts.len(),
            "column count must survive the wire (mode = {mode:?})"
        );
        for (i, (rec, orig)) in recovered
            .column_ciphertexts
            .iter()
            .zip(base.column_ciphertexts.iter())
            .enumerate()
        {
            assert_eq!(
                rec.a.coeffs(),
                orig.a.coeffs(),
                "column {i} a-coefficients must survive the wire (mode = {mode:?})"
            );
            assert_eq!(
                rec.b.coeffs(),
                orig.b.coeffs(),
                "column {i} b-coefficients must survive the wire (mode = {mode:?})"
            );
        }
    }
}

#[test]
fn client_query_session_handle_both_states_bincode_stable() {
    let (crs, encoded_db, sk, mut sampler) = fixture_32b();
    let (_state, mut cq) = query(&crs, 17, &encoded_db.config, &sk, &mut sampler).unwrap();

    cq.session_handle = None;
    bincode_roundtrip_stable(&cq, "ClientQuery.session_handle = None");

    cq.session_handle = Some(ServerSessionHandle(42));
    bincode_roundtrip_stable(&cq, "ClientQuery.session_handle = Some(42)");
}

#[test]
fn seeded_client_query_session_handle_both_states_bincode_stable() {
    let (crs, encoded_db, sk, mut sampler) = fixture_32b();
    let (_state, mut sq) = query_seeded(&crs, 17, &encoded_db.config, &sk, &mut sampler).unwrap();

    sq.session_handle = None;
    bincode_roundtrip_stable(&sq, "SeededClientQuery.session_handle = None");

    sq.session_handle = Some(ServerSessionHandle(7));
    bincode_roundtrip_stable(&sq, "SeededClientQuery.session_handle = Some(7)");
}

#[test]
fn inspiring_packing_keys_z_body_empty_and_nonempty_bincode_stable() {
    // partial-InspiRING leaves `ClientPackingKeys.z_body` empty, which is the
    // state that needs the length prefix emitted; reached via query_seeded.
    let (crs, encoded_db, sk, mut sampler) = fixture_32b();
    let (_state, sq) = query_seeded(&crs, 13, &encoded_db.config, &sk, &mut sampler).unwrap();

    let keys = sq
        .inspiring_packing_keys
        .as_ref()
        .expect("session-015 partial-InspiRING path should produce inspiring_packing_keys");
    assert!(
        keys.z_body.is_empty(),
        "partial-InspiRING path must produce empty z_body (Bug 2 coverage)"
    );
    bincode_roundtrip_stable(keys, "ClientPackingKeys z_body empty");
}

#[test]
fn end_to_end_seeded_response_bincode_stable_across_packing_modes() {
    let params = small_params();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let entry_size = 2usize;
    let n = params.ring_dim;
    let db: Vec<u8> = (0..n as u64)
        .flat_map(|i| vec![(i % 256) as u8, 0u8])
        .collect();
    let (crs, encoded_db, sk) = setup(&params, &db, entry_size, &mut sampler).unwrap();

    for idx in &[11u64, 42, 100, 200] {
        let (state, mut sq) =
            query_seeded(&crs, *idx, &encoded_db.config, &sk, &mut sampler).unwrap();
        sq.packing_mode = PackingMode::Inspiring;
        let response: ServerResponse = respond_seeded_inspiring(&crs, &encoded_db, &sq).unwrap();
        bincode_roundtrip_stable(&response, &format!("ServerResponse @ idx={idx}"));
        let recovered = extract_inspiring(&crs, &state, &response, entry_size).unwrap();
        let expected = vec![((*idx as usize) % 256) as u8, 0u8];
        assert_eq!(recovered, expected, "roundtrip extract @ idx={idx}");
    }
}
