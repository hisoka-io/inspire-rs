//! Response producers must tag their packing algorithm and reject malformed empty shards.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, InspireVariant, SecurityLevel};
use raven_inspire::pir::{
    extract_with_variant, query, query_seeded, respond, respond_inspiring,
    respond_inspiring_cached, respond_inspiring_cached_with_session, respond_one_packing,
    respond_seeded, respond_seeded_packed, respond_seeded_with_variant, respond_sequential,
    respond_with_variant, setup, ClientQuery, EncodedDatabase, PackingMode, SeededClientQuery,
    ServerCrs, ServerInspiringCache, ServerResponse,
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
        gadget_len: 3,
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
