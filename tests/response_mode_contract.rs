//! Response producers must tag their packing algorithm and reject malformed empty shards.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, InspireVariant, SecurityLevel};
use raven_inspire::pir::{
    extract_with_variant, query, query_seeded, respond, respond_inspiring,
    respond_inspiring_cached, respond_inspiring_cached_with_session, respond_one_packing,
    respond_seeded, respond_seeded_inspiring, respond_seeded_inspiring_cached,
    respond_seeded_inspiring_cached_with_session, respond_seeded_packed,
    respond_seeded_with_variant, respond_sequential, respond_with_variant, setup, ClientQuery,
    EncodedDatabase, PackingMode, SeededClientQuery, ServerCrs, ServerInspiringCache,
    ServerResponse,
};

#[derive(Clone)]
struct Fixture {
    crs: ServerCrs,
    encoded_db: EncodedDatabase,
    query: ClientQuery,
    seeded_query: SeededClientQuery,
    state: raven_inspire::ClientState,
    seeded_state: raven_inspire::ClientState,
    cache: ServerInspiringCache,
    database: Vec<u8>,
    entry_size: usize,
    target: usize,
}

fn fixture() -> Fixture {
    let params = InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65_537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    };
    let entry_size = 64;
    let database: Vec<u8> = (0..params.ring_dim * entry_size)
        .map(|i| ((i * 19 + 5) % 251) as u8)
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 41);
    let (crs, encoded_db, sk) = setup(&params, &database, entry_size, &mut sampler).expect("setup");
    let target = 42;
    let (state, query) =
        query(&crs, target as u64, &encoded_db.config, &sk, &mut sampler).expect("query");
    let (seeded_state, seeded_query) =
        query_seeded(&crs, target as u64, &encoded_db.config, &sk, &mut sampler)
            .expect("seeded query");
    let cache = ServerInspiringCache::new(&crs, &encoded_db).expect("cache");
    Fixture {
        crs,
        encoded_db,
        query,
        seeded_query,
        state,
        seeded_state,
        cache,
        database,
        entry_size,
        target,
    }
}

fn dense_two_shards(fixture: &Fixture) -> EncodedDatabase {
    let mut encoded_db = fixture.encoded_db.clone();
    let mut second = encoded_db.shards[0].clone();
    second.id = 1;
    encoded_db.shards.push(second);
    encoded_db
}

fn responder_outcomes(
    fixture: &Fixture,
    encoded_db: &EncodedDatabase,
    query: &ClientQuery,
    cache: &ServerInspiringCache,
) -> [(&'static str, raven_inspire::pir::Result<ServerResponse>); 6] {
    [
        ("respond", respond(&fixture.crs, encoded_db, query)),
        (
            "respond_one_packing",
            respond_one_packing(&fixture.crs, encoded_db, query),
        ),
        (
            "respond_inspiring",
            respond_inspiring(&fixture.crs, encoded_db, query),
        ),
        (
            "respond_inspiring_cached",
            respond_inspiring_cached(&fixture.crs, encoded_db, query, cache),
        ),
        (
            "respond_inspiring_cached_with_session",
            respond_inspiring_cached_with_session(&fixture.crs, encoded_db, query, cache, None),
        ),
        (
            "respond_sequential",
            respond_sequential(&fixture.crs, encoded_db, query),
        ),
    ]
}

#[test]
fn every_responder_serves_dense_first_and_last_shards_byte_identically() {
    let fixture = fixture();
    let encoded_db = dense_two_shards(&fixture);
    let cache = ServerInspiringCache::new(&fixture.crs, &encoded_db).expect("cache");
    let mut first_query = fixture.query.clone();
    first_query.shard_id = 0;
    let mut last_query = fixture.query.clone();
    last_query.shard_id = 1;

    let first = responder_outcomes(&fixture, &encoded_db, &first_query, &cache);
    let last = responder_outcomes(&fixture, &encoded_db, &last_query, &cache);
    for ((operation, first), (last_operation, last)) in first.into_iter().zip(last) {
        assert_eq!(operation, last_operation);
        let first_wire = first
            .expect("dense first shard")
            .to_binary()
            .expect("serialize first response");
        let last_wire = last
            .expect("dense last shard")
            .to_binary()
            .expect("serialize last response");
        assert_eq!(first_wire, last_wire, "{operation}");
    }
}

#[test]
fn every_responder_refuses_a_shard_vector_position_id_mismatch() {
    let fixture = fixture();
    let mut encoded_db = dense_two_shards(&fixture);
    encoded_db.shards.swap(0, 1);
    let cache = ServerInspiringCache::new(&fixture.crs, &encoded_db).expect("cache");
    let mut query = fixture.query.clone();
    query.shard_id = 1;

    for (operation, outcome) in responder_outcomes(&fixture, &encoded_db, &query, &cache) {
        let Err(error) = outcome else {
            panic!("{operation}: a non-dense shard vector must be refused");
        };
        let message = error.to_string();
        assert!(message.contains(operation), "{message}");
        assert!(message.contains("position 1"), "{message}");
        assert!(message.contains("shard id 0"), "{message}");
    }
}

#[test]
fn tree_responders_emit_a_tree_tag_that_the_existing_decoder_reads() {
    let f = fixture();
    let direct = respond_one_packing(&f.crs, &f.encoded_db, &f.query).expect("tree respond");
    let seeded =
        respond_seeded_packed(&f.crs, &f.encoded_db, &f.seeded_query).expect("seeded tree respond");

    for response in [direct, seeded] {
        assert_eq!(response.packing_mode, Some(PackingMode::Tree));
        let bytes = response.to_binary().expect("serialize");
        let decoded = ServerResponse::from_binary(&bytes).expect("existing decoder");
        assert_eq!(decoded.packing_mode, Some(PackingMode::Tree));
    }
}

#[test]
fn explicit_one_packing_dispatch_round_trips_an_unmodified_query() {
    let f = fixture();
    let expected = &f.database[f.target * f.entry_size..(f.target + 1) * f.entry_size];

    let response =
        respond_with_variant(&f.crs, &f.encoded_db, &f.query, InspireVariant::OnePacking)
            .expect("OnePacking respond");
    assert_eq!(response.packing_mode, Some(PackingMode::Tree));
    let decoded = extract_with_variant(
        &f.crs,
        &f.state,
        &response,
        f.entry_size,
        InspireVariant::OnePacking,
    )
    .expect("OnePacking extract");
    assert_eq!(decoded, expected);

    let response = respond_seeded_with_variant(
        &f.crs,
        &f.encoded_db,
        &f.seeded_query,
        InspireVariant::OnePacking,
    )
    .expect("seeded OnePacking respond");
    let decoded = extract_with_variant(
        &f.crs,
        &f.seeded_state,
        &response,
        f.entry_size,
        InspireVariant::OnePacking,
    )
    .expect("seeded OnePacking extract");
    assert_eq!(decoded, expected);
}

#[test]
fn every_extractor_rounds_an_odd_entry_width_up_to_two_columns() {
    let params = InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65_537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    };
    let entry_size = 3usize;
    let database: Vec<u8> = (0..params.ring_dim)
        .flat_map(|i| [u8::try_from(i % 251).unwrap(), 0, 1])
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 42);
    let (crs, encoded_db, sk) =
        setup(&params, &database, entry_size, &mut sampler).expect("odd-width setup");
    let target = 42usize;
    let (state, query) =
        query(&crs, target as u64, &encoded_db.config, &sk, &mut sampler).expect("odd-width query");
    let expected = &database[target * entry_size..(target + 1) * entry_size];

    for (variant, response) in [
        (
            InspireVariant::NoPacking,
            respond(&crs, &encoded_db, &query).expect("unpacked response"),
        ),
        (
            InspireVariant::OnePacking,
            respond_one_packing(&crs, &encoded_db, &query).expect("tree response"),
        ),
        (
            InspireVariant::TwoPacking,
            respond_inspiring(&crs, &encoded_db, &query).expect("InspiRING response"),
        ),
    ] {
        let decoded = extract_with_variant(&crs, &state, &response, entry_size, variant)
            .expect("odd-width extraction");
        assert_eq!(decoded, expected, "{variant:?}");
    }
}

fn with_shard_width(f: &Fixture, got: usize) -> EncodedDatabase {
    let mut encoded_db = f.encoded_db.clone();
    let polynomials = &mut encoded_db.shards[0].polynomials;
    if got <= polynomials.len() {
        polynomials.truncate(got);
    } else {
        let extra = polynomials[0].clone();
        polynomials.resize(got, extra);
    }
    encoded_db
}

fn assert_shard_width_error(message: &str, operation: &str, got: usize, expected: usize) {
    assert!(message.contains(operation), "{message}");
    assert!(message.contains("shard 0"), "{message}");
    assert!(message.contains(&format!("got {got}")), "{message}");
    assert!(
        message.contains(&format!("expected {expected}")),
        "{message}"
    );
}

fn assert_all_responders_refuse_forged_gadget(f: &Fixture, query: &ClientQuery) {
    let got = &query.rgsw_ciphertext.gadget;
    let expected = &f.crs.params;
    let got_fields = format!("got len={} base={} q={}", got.len, got.base, got.q);
    let expected_fields = format!(
        "expected len={} base={} q={}",
        1, expected.gadget_base, expected.q
    );
    for (operation, outcome) in [
        ("respond", respond(&f.crs, &f.encoded_db, query)),
        (
            "respond_one_packing",
            respond_one_packing(&f.crs, &f.encoded_db, query),
        ),
        (
            "respond_inspiring",
            respond_inspiring(&f.crs, &f.encoded_db, query),
        ),
        (
            "respond_inspiring_cached",
            respond_inspiring_cached(&f.crs, &f.encoded_db, query, &f.cache),
        ),
        (
            "respond_inspiring_cached_with_session",
            respond_inspiring_cached_with_session(&f.crs, &f.encoded_db, query, &f.cache, None),
        ),
        (
            "respond_sequential",
            respond_sequential(&f.crs, &f.encoded_db, query),
        ),
    ] {
        let error = outcome.expect_err("a forged RGSW gadget must be refused");
        let message = error.to_string();
        assert!(message.contains(operation), "{message}");
        assert!(message.contains("RGSW gadget mismatch"), "{message}");
        assert!(message.contains(&got_fields), "{message}");
        assert!(message.contains(&expected_fields), "{message}");
    }
}

#[test]
fn every_responder_refuses_each_forged_rgsw_gadget_field() {
    let f = fixture();

    let mut wrong_len = f.query.clone();
    wrong_len.rgsw_ciphertext.gadget.len -= 1;
    wrong_len
        .rgsw_ciphertext
        .rows
        .truncate(wrong_len.rgsw_ciphertext.gadget.len);

    let mut wrong_base = f.query.clone();
    wrong_base.rgsw_ciphertext.gadget.base += 1;

    let mut wrong_q = f.query.clone();
    wrong_q.rgsw_ciphertext.gadget.q -= 1;

    for forged_query in [wrong_len, wrong_base, wrong_q] {
        assert_all_responders_refuse_forged_gadget(&f, &forged_query);
    }
}

#[test]
fn every_responder_refuses_a_wrong_one_sided_rgsw_row_count() {
    let f = fixture();
    let mut query = f.query.clone();
    query
        .rgsw_ciphertext
        .rows
        .truncate(query.rgsw_ciphertext.gadget.len - 1);

    for (operation, outcome) in [
        ("respond", respond(&f.crs, &f.encoded_db, &query)),
        (
            "respond_one_packing",
            respond_one_packing(&f.crs, &f.encoded_db, &query),
        ),
        (
            "respond_inspiring",
            respond_inspiring(&f.crs, &f.encoded_db, &query),
        ),
        (
            "respond_inspiring_cached",
            respond_inspiring_cached(&f.crs, &f.encoded_db, &query, &f.cache),
        ),
        (
            "respond_inspiring_cached_with_session",
            respond_inspiring_cached_with_session(&f.crs, &f.encoded_db, &query, &f.cache, None),
        ),
        (
            "respond_sequential",
            respond_sequential(&f.crs, &f.encoded_db, &query),
        ),
    ] {
        let error = outcome.expect_err("wrong one-sided RGSW row count must be refused");
        let message = error.to_string();
        assert!(message.contains(operation), "{message}");
        assert!(message.contains("row-count mismatch"), "{message}");
        assert!(message.contains("got 0"), "{message}");
        assert!(message.contains("expected 1"), "{message}");
    }
}

#[test]
fn every_seeded_responder_refuses_legacy_three_row_shape_before_expansion() {
    let f = fixture();
    let mut query = f.seeded_query.clone();
    query.rgsw_ciphertext.gadget = f.crs.rgsw_gadget.clone();
    let row = query.rgsw_ciphertext.rows.first().expect("one row").clone();
    query
        .rgsw_ciphertext
        .rows
        .resize(query.rgsw_ciphertext.gadget.len, row);

    for (operation, outcome) in [
        (
            "respond_seeded",
            respond_seeded(&f.crs, &f.encoded_db, &query),
        ),
        (
            "respond_seeded_packed",
            respond_seeded_packed(&f.crs, &f.encoded_db, &query),
        ),
        (
            "respond_seeded_inspiring",
            respond_seeded_inspiring(&f.crs, &f.encoded_db, &query),
        ),
        (
            "respond_seeded_inspiring_cached",
            respond_seeded_inspiring_cached(&f.crs, &f.encoded_db, &query, &f.cache),
        ),
        (
            "respond_seeded_inspiring_cached_with_session",
            respond_seeded_inspiring_cached_with_session(
                &f.crs,
                &f.encoded_db,
                &query,
                &f.cache,
                None,
            ),
        ),
        (
            "respond_seeded_with_variant",
            respond_seeded_with_variant(&f.crs, &f.encoded_db, &query, InspireVariant::TwoPacking),
        ),
    ] {
        let error = outcome.expect_err("legacy seeded query must fail before expansion");
        let message = error.to_string();
        assert!(message.contains(operation), "{message}");
        assert!(message.contains("RGSW gadget mismatch"), "{message}");
        assert!(message.contains("got len=3"), "{message}");
        assert!(message.contains("expected len=1"), "{message}");
    }
}

#[test]
fn unpacked_responders_refuse_every_wrong_shard_width() {
    let f = fixture();
    let expected = f.crs.inspiring_num_columns;
    for got in [0, expected - 1, expected + 1] {
        let encoded_db = with_shard_width(&f, got);
        for (operation, outcome) in [
            ("respond", respond(&f.crs, &encoded_db, &f.query)),
            (
                "respond_sequential",
                respond_sequential(&f.crs, &encoded_db, &f.query),
            ),
            (
                "respond",
                respond_seeded(&f.crs, &encoded_db, &f.seeded_query),
            ),
        ] {
            let err = outcome.expect_err("wrong shard width must be refused");
            assert_shard_width_error(&err.to_string(), operation, got, expected);
        }
    }
}

#[test]
fn tree_responders_refuse_every_wrong_shard_width() {
    let f = fixture();
    let expected = f.crs.inspiring_num_columns;
    for got in [0, expected - 1, expected + 1] {
        let encoded_db = with_shard_width(&f, got);
        for outcome in [
            respond_one_packing(&f.crs, &encoded_db, &f.query),
            respond_seeded_packed(&f.crs, &encoded_db, &f.seeded_query),
        ] {
            let err = outcome.expect_err("wrong shard width must be refused");
            assert_shard_width_error(&err.to_string(), "respond_one_packing", got, expected);
        }
    }
}

#[test]
fn inspiring_responders_refuse_every_wrong_shard_width() {
    let f = fixture();
    let expected = f.crs.inspiring_num_columns;
    for got in [0, expected - 1, expected + 1] {
        let encoded_db = with_shard_width(&f, got);
        for (operation, outcome) in [
            (
                "respond_inspiring",
                respond_inspiring(&f.crs, &encoded_db, &f.query),
            ),
            (
                "respond_inspiring_cached",
                respond_inspiring_cached(&f.crs, &encoded_db, &f.query, &f.cache),
            ),
            (
                "respond_inspiring_cached_with_session",
                respond_inspiring_cached_with_session(
                    &f.crs,
                    &encoded_db,
                    &f.query,
                    &f.cache,
                    None,
                ),
            ),
        ] {
            let err = outcome.expect_err("wrong shard width must be refused");
            assert_shard_width_error(&err.to_string(), operation, got, expected);
        }
    }
}

#[test]
fn two_packing_server_refuses_a_tree_mode_query() {
    let f = fixture();
    let mut query = f.seeded_query;
    query.packing_mode = PackingMode::Tree;
    let err =
        respond_seeded_with_variant(&f.crs, &f.encoded_db, &query, InspireVariant::TwoPacking)
            .expect_err("TwoPacking server must refuse Tree mode");
    let message = err.to_string();
    assert!(message.contains("TwoPacking"), "{message}");
    assert!(message.contains("Tree"), "{message}");
}

#[cfg(feature = "mod-switch-response")]
#[test]
fn tree_responder_round_trips_through_the_existing_rims_byte_two_decoder() {
    use raven_inspire::pir::mod_switch::{decode_response_packed, encode_response_packed};

    let f = fixture();
    let response = respond_one_packing(&f.crs, &f.encoded_db, &f.query).expect("tree respond");
    let bytes = encode_response_packed(&response).expect("RIMS encode");
    assert_eq!(bytes[18], 2);
    let decoded = decode_response_packed(&bytes).expect("existing RIMS decoder");
    assert_eq!(decoded.packing_mode, Some(PackingMode::Tree));
}
