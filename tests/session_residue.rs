//! A serialized `ClientSession` residue must carry only CRS, secret key and
//! y_body, rehydrate without rebuilding `pack_params`, and still decode the
//! planted entry - the server re-derives `y_all` from y_body, so the client
//! never needs the multi-hundred-MiB automorph tables on the query path.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::pir::{extract_inspiring, respond_inspiring, ClientSession, ServerSessionStore};
use raven_inspire::setup;

fn d256_params() -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65_536,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

#[test]
fn residue_round_trip_rehydrated_session_decodes_without_pack_params() {
    let params = d256_params();
    let num_entries = 64usize;
    let entry_size = 32usize;
    let db: Vec<u8> = (0..num_entries * entry_size)
        .map(|i| (i % 251) as u8)
        .collect();

    let mut sampler = GaussianSampler::with_seed(params.sigma, 7);
    let (crs, encoded_db, rlwe_sk) = setup(&params, &db, entry_size, &mut sampler).expect("setup");

    let session = ClientSession::new(crs.clone(), rlwe_sk, &mut sampler).expect("session");
    assert!(
        session.has_inspiring_packing(),
        "d=256 setup must enable InspiRING packing for this test to exercise the residue path"
    );
    assert!(
        session.pack_params().is_some(),
        "a fresh session holds pack_params"
    );

    let residue = session.to_residue();
    let bytes = bincode::serialize(&residue).expect("serialize residue");
    // d=256 has no large tables; this only catches a table-shaped payload
    assert!(
        (bytes.len() as u64) < 2 * 1024 * 1024,
        "serialized residue is {} bytes; no automorph-table-shaped payload may persist",
        bytes.len()
    );

    let residue2 = bincode::deserialize(&bytes).expect("deserialize residue");
    let session2 = ClientSession::from_residue(residue2).expect("rehydrate consistent residue");
    assert!(
        session2.pack_params().is_none(),
        "a rehydrated session must NOT rebuild pack_params"
    );
    assert!(
        session2.has_inspiring_packing(),
        "a rehydrated session keeps its packing keys"
    );

    for target in [0u64, 1, 7, (num_entries as u64) - 1] {
        let (state, q) = session2
            .query(target, &encoded_db.config, &mut sampler)
            .expect("rehydrated query");
        let response = respond_inspiring(&crs, &encoded_db, &q).expect("respond");
        let decoded = extract_inspiring(&crs, &state, &response, entry_size).expect("extract");
        let lo = (target as usize) * entry_size;
        assert_eq!(
            decoded.as_slice(),
            &db[lo..lo + entry_size],
            "rehydrated-session decode mismatch at index {target}"
        );
    }
}

#[test]
fn residue_round_trip_preserves_tree_only_none_packing() {
    let params = d256_params();
    let num_entries = 64usize;
    let entry_size = 32usize;
    let db: Vec<u8> = (0..num_entries * entry_size)
        .map(|i| (i % 251) as u8)
        .collect();

    let mut sampler = GaussianSampler::with_seed(params.sigma, 11);
    let (mut crs, _encoded_db, rlwe_sk) =
        setup(&params, &db, entry_size, &mut sampler).expect("setup");
    crs.inspiring_num_columns = 0;

    let session = ClientSession::new(crs, rlwe_sk, &mut sampler).expect("session");
    assert!(
        !session.has_inspiring_packing(),
        "a Tree-only CRS must yield no packing keys"
    );

    let bytes = bincode::serialize(&session.to_residue()).expect("serialize residue");
    let session2 = ClientSession::from_residue(bincode::deserialize(&bytes).expect("deserialize"))
        .expect("rehydrate consistent residue");
    assert!(
        !session2.has_inspiring_packing(),
        "the None packing-keys shape must survive the residue round-trip"
    );
    assert!(session2.pack_params().is_none());
}

#[test]
fn residue_drops_handshake_handle_and_rehydrated_session_decodes_inline() {
    let params = d256_params();
    let num_entries = 64usize;
    let entry_size = 32usize;
    let db: Vec<u8> = (0..num_entries * entry_size)
        .map(|i| (i % 251) as u8)
        .collect();

    let mut sampler = GaussianSampler::with_seed(params.sigma, 13);
    let (crs, encoded_db, rlwe_sk) = setup(&params, &db, entry_size, &mut sampler).expect("setup");

    let mut session = ClientSession::new(crs.clone(), rlwe_sk, &mut sampler).expect("session");
    let store = ServerSessionStore::new();
    session.register_with(&store).expect("register");
    assert!(
        session.session_handle().is_some(),
        "registration sets a handshake handle"
    );

    let bytes = bincode::serialize(&session.to_residue()).expect("serialize residue");
    let session2 = ClientSession::from_residue(bincode::deserialize(&bytes).expect("deserialize"))
        .expect("rehydrate consistent residue");
    assert!(
        session2.session_handle().is_none(),
        "the residue must not carry the server-issued handshake handle"
    );

    let (state, q) = session2
        .query(7, &encoded_db.config, &mut sampler)
        .expect("rehydrated query");
    let response = respond_inspiring(&crs, &encoded_db, &q).expect("respond");
    let decoded = extract_inspiring(&crs, &state, &response, entry_size).expect("extract");
    assert_eq!(decoded.as_slice(), &db[7 * entry_size..8 * entry_size]);
}

// A CRS versioned-bytes test lived here; its round-trip half duplicated
// crs_ingest_width_validation.rs and its unique halves (the magic-mismatch
// guard and the decode-size cap) moved there with the rest of the CRS ingest
// surface (2026-09-06). This file is about the session residue.

#[test]
fn debug_redacts_secret_key_on_session_and_residue() {
    let params = d256_params();
    let entry_size = 32usize;
    let db = vec![0u8; params.ring_dim * entry_size];
    let mut sampler = GaussianSampler::with_seed(params.sigma, 19);
    let (crs, _encoded_db, rlwe_sk) = setup(&params, &db, entry_size, &mut sampler).expect("setup");

    // Poly's Debug prints coefficients; the secret key must never reach it
    let poly_dbg = format!("{:?}", rlwe_sk.poly);
    assert!(
        poly_dbg.contains("Poly") || poly_dbg.contains("coeffs"),
        "precondition: the raw polynomial Debug exposes coefficients"
    );

    let key_dbg = format!("{rlwe_sk:?}");
    let lwe_dbg = format!(
        "{:?}",
        raven_inspire::lwe::LweSecretKey::from_rlwe(&rlwe_sk)
    );
    assert!(
        !lwe_dbg.contains("coeffs"),
        "LweSecretKey Debug must not print coefficient material: {lwe_dbg}"
    );
    let session = ClientSession::new(crs, rlwe_sk, &mut sampler).expect("session");
    let session_dbg = format!("{session:?}");
    let residue_dbg = format!("{:?}", session.to_residue());

    for (label, dbg) in [
        ("RlweSecretKey", &key_dbg),
        ("ClientSession", &session_dbg),
        ("SessionResidue", &residue_dbg),
    ] {
        assert!(
            !dbg.contains("Poly") && !dbg.contains("coeffs"),
            "{label} Debug must not print secret-key polynomial material: {dbg}"
        );
        assert!(
            dbg.contains("ring_dim"),
            "{label} Debug should show the redacted shape"
        );
    }
}
