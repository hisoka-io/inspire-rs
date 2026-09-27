//! Hostile nested shapes against both ends of the wire. Self-inconsistent structures
//! (row count against gadget, a ring that differs inside one container, a forged NTT
//! flag, a non-canonical coefficient) must fail to decode. Self-consistent ones on the
//! wrong ring decode, and every responder and extractor must refuse them with an error:
//! never a panic, never `Ok`.

#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "test fixture failures must be loud"
)]

use std::panic::{catch_unwind, AssertUnwindSafe};

use raven_inspire::math::{GaussianSampler, Poly};
use raven_inspire::params::{InspireParams, InspireVariant, SecurityLevel};
use raven_inspire::pir::{
    extract, extract_inspiring, extract_two_packing, extract_with_variant, query, query_seeded,
    respond, respond_inspiring, respond_inspiring_cached, respond_inspiring_cached_with_session,
    respond_one_packing, respond_seeded, respond_seeded_inspiring, respond_seeded_inspiring_cached,
    respond_seeded_inspiring_cached_with_session, respond_seeded_packed,
    respond_seeded_with_variant, respond_sequential, respond_with_variant, ClientQuery,
    ClientSession, ClientState, EncodedDatabase, PackingMode, SeededClientQuery, ServerCrs,
    ServerInspiringCache, ServerResponse, ServerSessionStore, SessionResidue,
};
use raven_inspire::rlwe::{RlweCiphertext, RlweSecretKey};
use serde::de::DeserializeOwned;
use serde::Serialize;

const ENTRY_SIZE: usize = 32;
const TARGET: u64 = 5;
/// NTT-friendly at ring 256 (both `== 1 mod 512`), so a two-limb forgery reaches
/// arithmetic rather than failing on a table build.
const TWO_LIMBS: [u64; 2] = [268_369_921, 249_561_089];

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

struct Fixture {
    crs: ServerCrs,
    encoded: EncodedDatabase,
    cache: ServerInspiringCache,
    state: ClientState,
    seeded: SeededClientQuery,
    unseeded: ClientQuery,
    honest: ServerResponse,
    row: Vec<u8>,
    secret_key: RlweSecretKey,
}

fn fixture() -> Fixture {
    let params = params();
    let db: Vec<u8> = (0..params.ring_dim * ENTRY_SIZE)
        .map(|index| (index % 251) as u8)
        .collect();
    let row = db[TARGET as usize * ENTRY_SIZE..(TARGET as usize + 1) * ENTRY_SIZE].to_vec();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 41);
    let (crs, encoded, secret_key) =
        raven_inspire::setup(&params, &db, ENTRY_SIZE, &mut sampler).expect("setup");
    let (state, seeded) = query_seeded(&crs, TARGET, &encoded.config, &secret_key, &mut sampler)
        .expect("seeded query");
    let (_, unseeded) =
        query(&crs, TARGET, &encoded.config, &secret_key, &mut sampler).expect("query");
    let cache = ServerInspiringCache::new(&crs, &encoded).expect("cache");
    let honest =
        respond_seeded_inspiring_cached_with_session(&crs, &encoded, &seeded, &cache, None)
            .expect("honest response");
    Fixture {
        crs,
        encoded,
        cache,
        state,
        seeded,
        unseeded,
        honest,
        row,
        secret_key,
    }
}

fn bincode_round_trip<T: Serialize + DeserializeOwned>(value: &T) -> Result<T, String> {
    let bytes = bincode::serialize(value).expect("forgery serializes");
    bincode::deserialize(&bytes).map_err(|error| error.to_string())
}

fn assert_decode_refused<T: Serialize + DeserializeOwned>(what: &str, value: &T, needle: &str) {
    match bincode_round_trip(value) {
        Err(error) => assert!(error.contains(needle), "{what}: wrong refusal: {error}"),
        Ok(_) => panic!("{what}: decoded Ok; the wire boundary accepted a malformed shape"),
    }
}

fn assert_json_decode_refused<T: DeserializeOwned>(
    what: &str,
    json: serde_json::Value,
    needle: &str,
) {
    match serde_json::from_value::<T>(json) {
        Err(error) => assert!(
            error.to_string().contains(needle),
            "{what}: wrong refusal: {error}"
        ),
        Ok(_) => panic!("{what}: JSON decoded Ok; the wire boundary accepted a malformed shape"),
    }
}

fn assert_refused<T: std::fmt::Debug, E: std::fmt::Display>(
    what: &str,
    call: impl FnOnce() -> Result<T, E>,
    needle: &str,
) {
    match catch_unwind(AssertUnwindSafe(call)) {
        Ok(Err(error)) => assert!(
            error.to_string().contains(needle),
            "{what}: wrong refusal: {error}"
        ),
        Ok(Ok(value)) => panic!("{what}: returned Ok({value:?})"),
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_default();
            panic!("{what}: panicked instead of refusing: {message}")
        }
    }
}

fn ntt_flagged(mut poly: Poly) -> Poly {
    poly.force_ntt_domain();
    poly
}

fn poly_on(dim: usize, moduli: &[u64]) -> Poly {
    Poly::from_coeffs_moduli(vec![3; dim], moduli)
}

// ------------------------------------------------------------- decode boundary

#[test]
fn honest_query_and_response_bytes_are_unchanged_by_the_boundary() {
    let fx = fixture();
    let seeded = bincode::serialize(&fx.seeded).expect("seeded");
    let unseeded = bincode::serialize(&fx.unseeded).expect("unseeded");
    let response = bincode::serialize(&fx.honest).expect("response");
    let decoded_seeded: SeededClientQuery = bincode::deserialize(&seeded).expect("seeded decodes");
    let decoded_unseeded: ClientQuery = bincode::deserialize(&unseeded).expect("query decodes");
    let decoded_response: ServerResponse =
        bincode::deserialize(&response).expect("response decodes");
    assert_eq!(
        bincode::serialize(&decoded_seeded).expect("re-encode"),
        seeded
    );
    assert_eq!(
        bincode::serialize(&decoded_unseeded).expect("re-encode"),
        unseeded
    );
    assert_eq!(
        bincode::serialize(&decoded_response).expect("re-encode"),
        response
    );

    let json = serde_json::to_string(&fx.unseeded).expect("json");
    let from_json: ClientQuery = serde_json::from_str(&json).expect("json query decodes");
    assert_eq!(bincode::serialize(&from_json).expect("re-encode"), unseeded);

    let answer = respond_seeded_inspiring_cached_with_session(
        &fx.crs,
        &fx.encoded,
        &decoded_seeded,
        &fx.cache,
        None,
    )
    .expect("decoded honest query responds");
    assert_eq!(
        extract_two_packing(&fx.crs, &fx.state, &answer, ENTRY_SIZE).expect("extracts"),
        fx.row
    );
}

#[test]
fn rgsw_row_count_must_match_the_gadget_at_decode() {
    let fx = fixture();
    let mut forged = fx.seeded.clone();
    forged
        .rgsw_ciphertext
        .rows
        .push(forged.rgsw_ciphertext.rows[0].clone());
    assert_decode_refused("seeded row surplus", &forged, "2 rows for a 1-digit gadget");

    let mut forged = fx.unseeded.clone();
    forged.rgsw_ciphertext.rows.clear();
    assert_decode_refused(
        "unseeded rows empty",
        &forged,
        "0 rows for a 1-digit gadget",
    );

    let mut forged = fx.seeded.clone();
    forged.rgsw_ciphertext.rows.clear();
    forged.rgsw_ciphertext.gadget.len = 0;
    assert_decode_refused("zero-digit gadget", &forged, "needs base >= 2, len >= 1");
}

#[test]
fn rgsw_rows_must_share_one_ring_and_the_gadget_modulus_at_decode() {
    let fx = fixture();
    let moduli = fx.crs.params.moduli().to_vec();

    let mut forged = fx.seeded.clone();
    forged.rgsw_ciphertext.gadget.len = 2;
    let mut second = forged.rgsw_ciphertext.rows[0].clone();
    second.b = poly_on(128, &moduli);
    forged.rgsw_ciphertext.rows.push(second);
    assert_decode_refused(
        "seeded rows on two rings",
        &forged,
        "RGSW row 1 has ring_dim 128",
    );

    let mut forged = fx.seeded.clone();
    forged.rgsw_ciphertext.gadget.q = 12_289;
    assert_decode_refused("gadget modulus", &forged, "differs from gadget q 12289");

    let mut forged = fx.unseeded.clone();
    forged.rgsw_ciphertext.rows[0].b = poly_on(128, &moduli);
    assert_decode_refused(
        "unseeded row a/b rings",
        &forged,
        "RLWE ciphertext b has ring_dim 128",
    );

    let mut forged = fx.unseeded.clone();
    forged.rgsw_ciphertext.rows[0].b = poly_on(256, &TWO_LIMBS);
    assert_decode_refused(
        "unseeded row a/b limbs",
        &forged,
        "RLWE ciphertext b has ring_dim 256",
    );
}

#[test]
fn a_forged_ntt_flag_is_refused_at_decode() {
    let fx = fixture();

    let mut forged = fx.seeded.clone();
    let row = &mut forged.rgsw_ciphertext.rows[0];
    row.b = ntt_flagged(row.b.clone());
    assert_decode_refused(
        "seeded row flag",
        &forged,
        "coefficient-domain RGSW row 0 b required",
    );

    let mut forged = fx.unseeded.clone();
    let row = &mut forged.rgsw_ciphertext.rows[0];
    row.a = ntt_flagged(row.a.clone());
    row.b = ntt_flagged(row.b.clone());
    assert_decode_refused(
        "unseeded row flag",
        &forged,
        "coefficient-domain RGSW row 0 a required",
    );

    let mut forged = fx.seeded.clone();
    let keys = forged.inspiring_packing_keys.as_mut().expect("inline keys");
    keys.y_body[0] = ntt_flagged(keys.y_body[0].clone());
    assert_decode_refused(
        "first packing body flag",
        &forged,
        "coefficient-domain packing-key y_body[0] required",
    );

    let mut forged = fx.seeded.clone();
    let keys = forged.inspiring_packing_keys.as_mut().expect("inline keys");
    keys.y_body[2] = ntt_flagged(keys.y_body[2].clone());
    assert_decode_refused(
        "later packing body flag",
        &forged,
        "coefficient-domain packing-key y_body[2] required",
    );

    let mut empty = Poly::default();
    empty.force_ntt_domain();
    assert_decode_refused(
        "empty polynomial flag",
        &empty,
        "empty polynomial declares the NTT",
    );
}

#[test]
fn packing_key_bodies_must_share_one_ring_at_decode() {
    let fx = fixture();
    let moduli = fx.crs.params.moduli().to_vec();

    let mut forged = fx.seeded.clone();
    forged.inspiring_packing_keys.as_mut().expect("keys").y_body[1] = poly_on(128, &moduli);
    assert_decode_refused(
        "body ring dimension",
        &forged,
        "packing-key y_body[1] has ring_dim 128",
    );

    let mut forged = fx.seeded.clone();
    forged.inspiring_packing_keys.as_mut().expect("keys").y_body[2] = poly_on(256, &TWO_LIMBS);
    assert_decode_refused(
        "body limb count",
        &forged,
        "packing-key y_body[2] has ring_dim 256",
    );

    let mut forged = fx.seeded.clone();
    forged
        .inspiring_packing_keys
        .as_mut()
        .expect("keys")
        .z_body
        .push(poly_on(128, &moduli));
    assert_decode_refused(
        "conjugation body ring",
        &forged,
        "packing-key z_body[0] has ring_dim 128",
    );
}

#[test]
fn a_non_canonical_coefficient_is_refused_on_the_human_readable_codec() {
    let fx = fixture();
    let q = fx.crs.params.q;

    let mut json = serde_json::to_value(&fx.unseeded).expect("json");
    json["rgsw_ciphertext"]["rows"][0]["b"]["coeffs"][7] = serde_json::json!(q + 11);
    assert_json_decode_refused::<ClientQuery>("query coefficient", json, "not canonical");

    let mut json = serde_json::to_value(&fx.honest).expect("json");
    json["ciphertext"]["Packed"]["a"]["coeffs"][0] = serde_json::json!(q);
    assert_json_decode_refused::<ServerResponse>("response a coefficient", json, "not canonical");

    let mut json = serde_json::to_value(&fx.honest).expect("json");
    json["ciphertext"]["Packed"]["b_prefix"][0] = serde_json::json!(q + 1);
    assert_json_decode_refused::<ServerResponse>("response b prefix", json, "not canonical");
}

#[test]
fn unpacked_response_columns_must_share_the_response_ring_at_decode() {
    let fx = fixture();
    let moduli = fx.crs.params.moduli().to_vec();
    let honest_column = RlweCiphertext::from_parts(poly_on(256, &moduli), poly_on(256, &moduli));
    let unpacked = |columns: Vec<RlweCiphertext>| ServerResponse {
        ciphertext: honest_column.clone(),
        column_ciphertexts: columns,
        packing_mode: None,
        packed_coefficients: None,
    };

    let skewed = RlweCiphertext {
        a: poly_on(256, &moduli),
        b: poly_on(128, &moduli),
    };
    assert_decode_refused(
        "column a/b rings",
        &unpacked(vec![honest_column.clone(), skewed]),
        "RLWE ciphertext b has ring_dim 128",
    );

    let other_ring = RlweCiphertext::from_parts(poly_on(128, &moduli), poly_on(128, &moduli));
    assert_decode_refused(
        "column off the response ring",
        &unpacked(vec![honest_column.clone(), other_ring]),
        "column ciphertext[1] has ring_dim 128",
    );
}

// ------------------------------------------------------------ responder entry points

type Outcome = raven_inspire::pir::Result<ServerResponse>;
type Responder<'a> = (&'static str, Box<dyn Fn() -> Outcome + 'a>);

fn every_unseeded_responder(fx: &Fixture, query: &ClientQuery, needle: &str) {
    let calls: [Responder<'_>; 9] = [
        ("respond", Box::new(|| respond(&fx.crs, &fx.encoded, query))),
        (
            "respond_sequential",
            Box::new(|| respond_sequential(&fx.crs, &fx.encoded, query)),
        ),
        (
            "respond_one_packing",
            Box::new(|| respond_one_packing(&fx.crs, &fx.encoded, query)),
        ),
        (
            "respond_inspiring",
            Box::new(|| respond_inspiring(&fx.crs, &fx.encoded, query)),
        ),
        (
            "respond_inspiring_cached",
            Box::new(|| respond_inspiring_cached(&fx.crs, &fx.encoded, query, &fx.cache)),
        ),
        (
            "respond_inspiring_cached_with_session",
            Box::new(|| {
                respond_inspiring_cached_with_session(&fx.crs, &fx.encoded, query, &fx.cache, None)
            }),
        ),
        (
            "respond_with_variant(NoPacking)",
            Box::new(|| {
                respond_with_variant(&fx.crs, &fx.encoded, query, InspireVariant::NoPacking)
            }),
        ),
        (
            "respond_with_variant(OnePacking)",
            Box::new(|| {
                respond_with_variant(&fx.crs, &fx.encoded, query, InspireVariant::OnePacking)
            }),
        ),
        (
            "respond_with_variant(TwoPacking)",
            Box::new(|| {
                respond_with_variant(&fx.crs, &fx.encoded, query, InspireVariant::TwoPacking)
            }),
        ),
    ];
    for (name, call) in calls {
        let expected = if name.ends_with("(TwoPacking)") {
            "not supported on an unseeded"
        } else {
            needle
        };
        assert_refused(name, call, expected);
    }
}

fn every_seeded_responder(fx: &Fixture, query: &SeededClientQuery, needle: &str) {
    let calls: [Responder<'_>; 8] = [
        (
            "respond_seeded",
            Box::new(|| respond_seeded(&fx.crs, &fx.encoded, query)),
        ),
        (
            "respond_seeded_packed",
            Box::new(|| respond_seeded_packed(&fx.crs, &fx.encoded, query)),
        ),
        (
            "respond_seeded_inspiring",
            Box::new(|| respond_seeded_inspiring(&fx.crs, &fx.encoded, query)),
        ),
        (
            "respond_seeded_inspiring_cached",
            Box::new(|| respond_seeded_inspiring_cached(&fx.crs, &fx.encoded, query, &fx.cache)),
        ),
        (
            "respond_seeded_inspiring_cached_with_session",
            Box::new(|| {
                respond_seeded_inspiring_cached_with_session(
                    &fx.crs,
                    &fx.encoded,
                    query,
                    &fx.cache,
                    None,
                )
            }),
        ),
        (
            "respond_seeded_with_variant(NoPacking)",
            Box::new(|| {
                respond_seeded_with_variant(&fx.crs, &fx.encoded, query, InspireVariant::NoPacking)
            }),
        ),
        (
            "respond_seeded_with_variant(OnePacking)",
            Box::new(|| {
                respond_seeded_with_variant(&fx.crs, &fx.encoded, query, InspireVariant::OnePacking)
            }),
        ),
        (
            "respond_seeded_with_variant(TwoPacking)",
            Box::new(|| {
                respond_seeded_with_variant(&fx.crs, &fx.encoded, query, InspireVariant::TwoPacking)
            }),
        ),
    ];
    for (name, call) in calls {
        assert_refused(name, call, needle);
    }
}

/// Decodes, because every row agrees with itself; only the CRS can say it is wrong.
fn decoded<T: Serialize + DeserializeOwned>(value: &T) -> T {
    bincode_round_trip(value).expect("a self-consistent forgery decodes")
}

#[test]
fn every_responder_refuses_a_query_on_the_wrong_ring() {
    let fx = fixture();
    let moduli = fx.crs.params.moduli().to_vec();

    let mut seeded = fx.seeded.clone();
    seeded.rgsw_ciphertext.rows[0].b = poly_on(128, &moduli);
    every_seeded_responder(&fx, &decoded(&seeded), "RGSW row 0 b dimension 128");

    let mut unseeded = fx.unseeded.clone();
    unseeded.rgsw_ciphertext.rows[0] =
        RlweCiphertext::from_parts(poly_on(128, &moduli), poly_on(128, &moduli));
    every_unseeded_responder(&fx, &decoded(&unseeded), "RGSW row 0 a dimension 128");
}

#[test]
fn every_responder_refuses_a_query_on_the_wrong_limbs() {
    let fx = fixture();

    let mut seeded = fx.seeded.clone();
    seeded.rgsw_ciphertext.rows[0].b = poly_on(256, &TWO_LIMBS);
    seeded.rgsw_ciphertext.gadget.q = TWO_LIMBS[0] * TWO_LIMBS[1];
    every_seeded_responder(&fx, &decoded(&seeded), "RGSW gadget mismatch");

    let mut unseeded = fx.unseeded.clone();
    unseeded.rgsw_ciphertext.rows[0] =
        RlweCiphertext::from_parts(poly_on(256, &TWO_LIMBS), poly_on(256, &TWO_LIMBS));
    unseeded.rgsw_ciphertext.gadget.q = TWO_LIMBS[0] * TWO_LIMBS[1];
    every_unseeded_responder(&fx, &decoded(&unseeded), "RGSW gadget mismatch");
}

#[test]
fn every_responder_refuses_a_multi_row_query() {
    let fx = fixture();
    let mut seeded = fx.seeded.clone();
    seeded.rgsw_ciphertext.gadget.len = 2;
    seeded
        .rgsw_ciphertext
        .rows
        .push(seeded.rgsw_ciphertext.rows[0].clone());
    every_seeded_responder(&fx, &decoded(&seeded), "RGSW gadget mismatch");

    let mut unseeded = fx.unseeded.clone();
    unseeded.rgsw_ciphertext.gadget.len = 2;
    unseeded
        .rgsw_ciphertext
        .rows
        .push(unseeded.rgsw_ciphertext.rows[0].clone());
    every_unseeded_responder(&fx, &decoded(&unseeded), "RGSW gadget mismatch");
}

#[test]
fn every_inspiring_responder_refuses_packing_keys_on_the_wrong_ring() {
    let fx = fixture();
    let moduli = fx.crs.params.moduli().to_vec();
    for (bodies, needle) in [
        (poly_on(128, &moduli), "dimension 128"),
        (poly_on(256, &TWO_LIMBS), "y_body[0] moduli"),
    ] {
        let mut seeded = fx.seeded.clone();
        let keys = seeded.inspiring_packing_keys.as_mut().expect("keys");
        keys.y_body = vec![bodies.clone(); keys.y_body.len()];
        let seeded = decoded(&seeded);
        for (name, call) in [
            (
                "respond_seeded_inspiring",
                Box::new(|| respond_seeded_inspiring(&fx.crs, &fx.encoded, &seeded))
                    as Box<dyn Fn() -> Outcome + '_>,
            ),
            (
                "respond_seeded_inspiring_cached",
                Box::new(|| {
                    respond_seeded_inspiring_cached(&fx.crs, &fx.encoded, &seeded, &fx.cache)
                }),
            ),
            (
                "respond_seeded_inspiring_cached_with_session",
                Box::new(|| {
                    respond_seeded_inspiring_cached_with_session(
                        &fx.crs,
                        &fx.encoded,
                        &seeded,
                        &fx.cache,
                        None,
                    )
                }),
            ),
            (
                "respond_seeded_with_variant(TwoPacking)",
                Box::new(|| {
                    respond_seeded_with_variant(
                        &fx.crs,
                        &fx.encoded,
                        &seeded,
                        InspireVariant::TwoPacking,
                    )
                }),
            ),
        ] {
            assert_refused(name, call, needle);
        }

        let store = ServerSessionStore::new();
        let keys = seeded.inspiring_packing_keys.clone().expect("keys");
        let handle = store
            .register(keys)
            .expect("store accepts coefficient-domain keys");
        let mut by_handle = seeded.clone();
        by_handle.inspiring_packing_keys = None;
        by_handle.session_handle = Some(handle);
        assert_refused(
            "session-resolved keys",
            || {
                respond_seeded_inspiring_cached_with_session(
                    &fx.crs,
                    &fx.encoded,
                    &by_handle,
                    &fx.cache,
                    Some(&store),
                )
            },
            needle,
        );
    }
}

// ------------------------------------------------------------------ client extractors

fn packed_response(ciphertext: RlweCiphertext, mode: PackingMode) -> ServerResponse {
    ServerResponse {
        ciphertext,
        column_ciphertexts: vec![],
        packing_mode: Some(mode),
        packed_coefficients: Some(raven_inspire::num_columns(ENTRY_SIZE) as u32),
    }
}

fn every_packed_extractor(fx: &Fixture, response: &ServerResponse, needle: &str) {
    let inspiring = |call: &str, f: &dyn Fn() -> raven_inspire::pir::Result<Vec<u8>>| {
        assert_refused(call, f, needle);
    };
    inspiring("extract_two_packing", &|| {
        extract_two_packing(&fx.crs, &fx.state, response, ENTRY_SIZE)
    });
    inspiring("extract_inspiring", &|| {
        extract_inspiring(&fx.crs, &fx.state, response, ENTRY_SIZE)
    });
    inspiring("extract_with_variant(TwoPacking)", &|| {
        extract_with_variant(
            &fx.crs,
            &fx.state,
            response,
            ENTRY_SIZE,
            InspireVariant::TwoPacking,
        )
    });
    #[cfg(feature = "mod-switch-response")]
    inspiring("extract_inspiring_mod_switched", &|| {
        raven_inspire::pir::mod_switch::extract_inspiring_mod_switched(
            &fx.crs, &fx.state, response, ENTRY_SIZE,
        )
    });

    let mut tree = response.clone();
    tree.packing_mode = Some(PackingMode::Tree);
    assert_refused(
        "extract_with_variant(OnePacking)",
        || {
            extract_with_variant(
                &fx.crs,
                &fx.state,
                &tree,
                ENTRY_SIZE,
                InspireVariant::OnePacking,
            )
        },
        needle,
    );
}

#[test]
fn every_extractor_refuses_a_response_on_the_wrong_ring() {
    let fx = fixture();
    let moduli = fx.crs.params.moduli().to_vec();
    let forged = packed_response(
        RlweCiphertext::from_parts(poly_on(128, &moduli), poly_on(128, &moduli)),
        PackingMode::Inspiring,
    );
    every_packed_extractor(&fx, &decoded(&forged), "response.a has ring_dim 128");
}

#[test]
fn every_extractor_refuses_a_response_on_the_wrong_limbs() {
    let fx = fixture();
    let forged = packed_response(
        RlweCiphertext::from_parts(poly_on(256, &TWO_LIMBS), poly_on(256, &TWO_LIMBS)),
        PackingMode::Inspiring,
    );
    let decoded = decoded(&forged);
    for call in ["extract_two_packing", "extract_inspiring"] {
        let f = || {
            if call == "extract_inspiring" {
                extract_inspiring(&fx.crs, &fx.state, &decoded, ENTRY_SIZE)
            } else {
                extract_two_packing(&fx.crs, &fx.state, &decoded, ENTRY_SIZE)
            }
        };
        assert_refused(
            call,
            f,
            "response.a has ring_dim 256, moduli [268369921, 249561089]",
        );
    }
}

#[test]
fn the_unpacked_extractor_refuses_columns_on_the_wrong_ring() {
    let fx = fixture();
    let moduli = fx.crs.params.moduli().to_vec();
    let column = RlweCiphertext::from_parts(poly_on(128, &moduli), poly_on(128, &moduli));
    let forged = ServerResponse {
        ciphertext: column.clone(),
        column_ciphertexts: vec![column; raven_inspire::num_columns(ENTRY_SIZE)],
        packing_mode: None,
        packed_coefficients: None,
    };
    let decoded = decoded(&forged);
    assert_refused(
        "extract",
        || extract(&fx.crs, &fx.state, &decoded, ENTRY_SIZE),
        "column ciphertext[0].a has ring_dim 128",
    );
    assert_refused(
        "extract_with_variant(NoPacking)",
        || {
            extract_with_variant(
                &fx.crs,
                &fx.state,
                &decoded,
                ENTRY_SIZE,
                InspireVariant::NoPacking,
            )
        },
        "column ciphertext[0].a has ring_dim 128",
    );
}

/// `ClientState`'s keys are `#[serde(skip)]`; a decoded state that was never rehydrated
/// carries an empty key.
#[test]
fn every_extractor_refuses_a_state_whose_key_was_not_rehydrated() {
    let fx = fixture();
    let bytes = bincode::serialize(&fx.state).expect("state");
    let state: ClientState = bincode::deserialize(&bytes).expect("state decodes");
    let response = decoded(&fx.honest);
    let needle = "secret key has ring_dim 0";
    assert_refused(
        "extract_two_packing",
        || extract_two_packing(&fx.crs, &state, &response, ENTRY_SIZE),
        needle,
    );
    #[cfg(feature = "mod-switch-response")]
    {
        use raven_inspire::pir::mod_switch::{
            extract_inspiring_mod_switched, mod_switch_response_checked, MOD_SWITCH_TARGET_36BIT,
        };
        let production = InspireParams::secure_128_d2048();
        let db = vec![7u8; production.ring_dim * ENTRY_SIZE];
        let mut sampler = GaussianSampler::with_seed(production.sigma, 43);
        let (crs, encoded, key) =
            raven_inspire::setup(&production, &db, ENTRY_SIZE, &mut sampler).expect("setup");
        let (fresh, query) =
            query_seeded(&crs, 1, &encoded.config, &key, &mut sampler).expect("query");
        let switched = mod_switch_response_checked(
            &production,
            &respond_seeded_inspiring(&crs, &encoded, &query).expect("response"),
            MOD_SWITCH_TARGET_36BIT,
        )
        .expect("switch");
        let bytes = bincode::serialize(&fresh).expect("state");
        let stale: ClientState = bincode::deserialize(&bytes).expect("state decodes");
        assert_refused(
            "extract_inspiring_mod_switched",
            || extract_inspiring_mod_switched(&crs, &stale, &switched, ENTRY_SIZE),
            needle,
        );
        assert_eq!(
            extract_inspiring_mod_switched(&crs, &fresh, &switched, ENTRY_SIZE).expect("honest"),
            vec![7u8; ENTRY_SIZE]
        );
    }
}

// ------------------------------------------ CRS galois keys and session residues

fn one_packing_row(fx: &Fixture, crs: &ServerCrs) -> Vec<u8> {
    let response = respond_one_packing(crs, &fx.encoded, &fx.unseeded).expect("responds");
    extract_with_variant(
        crs,
        &fx.state,
        &response,
        ENTRY_SIZE,
        InspireVariant::OnePacking,
    )
    .expect("extracts")
}

fn versioned_round_trip(crs: &ServerCrs) -> raven_inspire::pir::Result<ServerCrs> {
    ServerCrs::from_versioned_bytes(&crs.to_versioned_bytes().expect("crs encodes"))
}

#[test]
fn an_honest_crs_keeps_its_bytes_and_serves_one_packing() {
    let fx = fixture();
    let bytes = fx.crs.to_versioned_bytes().expect("crs encodes");
    let decoded = ServerCrs::from_versioned_bytes(&bytes).expect("honest crs decodes");
    assert_eq!(decoded.to_versioned_bytes().expect("re-encode"), bytes);
    assert_eq!(one_packing_row(&fx, &decoded), fx.row);

    let mut published = fx.crs.clone();
    published.galois_keys.clear();
    versioned_round_trip(&published).expect("the published CRS strips galois keys and decodes");
}

#[test]
fn a_galois_key_short_of_its_gadget_is_refused() {
    let fx = fixture();
    let mut forged = fx.crs.clone();
    for key in &mut forged.galois_keys {
        key.rows.pop();
    }
    let short = forged.galois_keys[0].rows.len();
    let needle = format!("key-switching matrix carries {short} rows for a 3-digit gadget");
    assert_refused(
        "from_versioned_bytes",
        || versioned_round_trip(&forged),
        &needle,
    );
    assert_refused(
        "respond_one_packing",
        || respond_one_packing(&forged, &fx.encoded, &fx.unseeded),
        &format!("galois key 0 carries {short} rows"),
    );
}

#[test]
fn galois_keys_must_sit_on_the_crs_ring() {
    let fx = fixture();
    let moduli = fx.crs.params.moduli().to_vec();
    let mut mixed = fx.crs.clone();
    mixed.galois_keys[0].rows[1] =
        RlweCiphertext::from_parts(poly_on(128, &moduli), poly_on(128, &moduli));
    assert_refused(
        "from_versioned_bytes (mixed rows)",
        || versioned_round_trip(&mixed),
        "key-switching matrix row 1 has ring_dim 128",
    );

    let mut foreign = fx.crs.clone();
    for row in &mut foreign.galois_keys[0].rows {
        *row = RlweCiphertext::from_parts(poly_on(128, &moduli), poly_on(128, &moduli));
    }
    let needle = "galois key 0 is on ring_dim 128";
    assert_refused(
        "from_versioned_bytes (foreign ring)",
        || versioned_round_trip(&foreign),
        needle,
    );
    assert_refused(
        "respond_one_packing (foreign ring)",
        || respond_one_packing(&foreign, &fx.encoded, &fx.unseeded),
        needle,
    );
}

#[test]
fn a_crs_missing_tree_levels_is_refused() {
    let fx = fixture();
    let mut short = fx.crs.clone();
    short.galois_keys.pop();
    let needle = "7 galois keys, but ring_dim 256 packs through 8 tree levels";
    assert_refused(
        "from_versioned_bytes",
        || versioned_round_trip(&short),
        needle,
    );
    assert_refused(
        "respond_one_packing (short)",
        || respond_one_packing(&short, &fx.encoded, &fx.unseeded),
        needle,
    );

    let mut published = fx.crs.clone();
    published.galois_keys.clear();
    assert_refused(
        "respond_one_packing (published CRS)",
        || respond_one_packing(&published, &fx.encoded, &fx.unseeded),
        "0 galois keys",
    );
}

fn residue_json(fx: &Fixture) -> serde_json::Value {
    let mut sampler = GaussianSampler::with_seed(fx.crs.params.sigma, 47);
    let session =
        ClientSession::new(fx.crs.clone(), fx.secret_key.clone(), &mut sampler).expect("session");
    serde_json::to_value(session.to_residue()).expect("residue json")
}

fn rehydrate(json: serde_json::Value) -> raven_inspire::pir::Result<ClientSession> {
    let residue: SessionResidue = serde_json::from_value(json).expect("residue decodes");
    ClientSession::from_residue(residue)
}

#[test]
fn a_forged_session_residue_is_refused_not_panicked() {
    let fx = fixture();
    let honest = residue_json(&fx);
    let session = rehydrate(honest.clone()).expect("honest residue rehydrates");
    assert_eq!(
        serde_json::to_value(session.to_residue()).expect("json"),
        honest
    );

    let mut ntt_key = honest.clone();
    ntt_key["rlwe_sk"]["poly"]["is_ntt"] = serde_json::json!(true);
    assert_refused(
        "from_residue (NTT key)",
        || rehydrate(ntt_key),
        "secret key declares the NTT domain",
    );

    let mut width = honest.clone();
    width["crs"]["inspiring_num_columns"] = serde_json::json!(3);
    assert_refused(
        "from_residue (illegal width)",
        || rehydrate(width),
        "not a legal InspiRING packing width",
    );

    let mut keys = honest.clone();
    let foreign = serde_json::to_value(poly_on(128, fx.crs.params.moduli())).expect("json");
    for body in ["y_body", "z_body"] {
        for poly in keys["packing_keys"][body]
            .as_array_mut()
            .expect("body is an array")
        {
            *poly = foreign.clone();
        }
    }
    assert_refused(
        "from_residue (foreign packing keys)",
        || rehydrate(keys),
        "packing-key y_body[0] dimension 128",
    );
}
