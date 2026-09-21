//! A shard holds one row per ring coefficient. The encoder refuses a config that says
//! otherwise, and every responder refuses a database whose config says otherwise, because
//! the rows such a database serves are not the rows the client addressed.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use raven_inspire::math::mod_q::DEFAULT_Q;
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::ShardConfig;
use raven_inspire::pir::{
    encode_database, extract_inspiring, query, respond, respond_inspiring,
    respond_inspiring_cached, respond_inspiring_cached_with_session, respond_one_packing,
    respond_sequential, setup, ClientQuery, Result,
};
use raven_inspire::rlwe::RlweSecretKey;
use raven_inspire::{
    ClientState, EncodedDatabase, InspireParams, PackingMode, SecurityLevel, ServerCrs,
    ServerInspiringCache, ServerResponse, ShardData,
};

const RING_DIM: usize = 256;
const ENTRY_SIZE: usize = 32;
const ROWS: usize = 512;
const FOREIGN_ROWS_PER_SHARD: usize = 128;
const TARGET: u64 = 300;

fn params() -> InspireParams {
    InspireParams {
        ring_dim: RING_DIM,
        q: DEFAULT_Q,
        crt_moduli: vec![DEFAULT_Q],
        p: 65_537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

fn row_tagged_database(rows: usize) -> Vec<u8> {
    (0..rows * ENTRY_SIZE)
        .map(|offset| ((offset / ENTRY_SIZE * 7 + offset % ENTRY_SIZE) % 251) as u8)
        .collect()
}

fn row(database: &[u8], index: u64) -> &[u8] {
    let start = usize::try_from(index).unwrap() * ENTRY_SIZE;
    &database[start..start + ENTRY_SIZE]
}

fn which_row(database: &[u8], bytes: &[u8]) -> String {
    (0..database.len() / ENTRY_SIZE)
        .find(|&index| row(database, index as u64) == bytes)
        .map_or_else(|| "no row".to_string(), |index| format!("row {index}"))
}

struct Honest {
    crs: ServerCrs,
    encoded_db: EncodedDatabase,
    sk: RlweSecretKey,
    database: Vec<u8>,
    sampler: GaussianSampler,
}

fn honest(rows: usize) -> Honest {
    let params = params();
    let database = row_tagged_database(rows);
    let mut sampler = GaussianSampler::with_seed(params.sigma, 11);
    let (crs, encoded_db, sk) = setup(&params, &database, ENTRY_SIZE, &mut sampler).expect("setup");
    Honest {
        crs,
        encoded_db,
        sk,
        database,
        sampler,
    }
}

fn config_with_rows_per_shard(rows_per_shard: usize) -> ShardConfig {
    ShardConfig {
        shard_size_bytes: (rows_per_shard * ENTRY_SIZE) as u64,
        entry_size_bytes: ENTRY_SIZE,
        total_entries: ROWS as u64,
    }
}

/// The query a client builds from the geometry it knows to be right.
fn addressed_query(honest: &mut Honest) -> (ClientState, ClientQuery) {
    let client_config = ShardConfig::for_ring_dim(RING_DIM, ENTRY_SIZE, ROWS as u64).unwrap();
    query(
        &honest.crs,
        TARGET,
        &client_config,
        &honest.sk,
        &mut honest.sampler,
    )
    .expect("query")
}

/// What an encoder sharding at `FOREIGN_ROWS_PER_SHARD` rows writes: each window on its
/// own, filling the first coefficients of a ring-sized shard.
fn foreign_shards(database: &[u8], params: &InspireParams) -> Vec<ShardData> {
    let window = ShardConfig::for_ring_dim(RING_DIM, ENTRY_SIZE, FOREIGN_ROWS_PER_SHARD as u64)
        .expect("window config");
    database
        .chunks(FOREIGN_ROWS_PER_SHARD * ENTRY_SIZE)
        .enumerate()
        .map(|(id, bytes)| {
            let mut shards = encode_database(bytes, ENTRY_SIZE, params, &window).expect("window");
            assert_eq!(shards.len(), 1, "one window is one partial shard");
            let mut shard = shards.pop().unwrap();
            shard.id = u32::try_from(id).unwrap();
            shard
        })
        .collect()
}

fn served_row(honest: &Honest, state: &ClientState, response: &ServerResponse) -> String {
    if response.packing_mode != Some(PackingMode::Inspiring) {
        return "an unpacked response".to_string();
    }
    extract_inspiring(&honest.crs, state, response, ENTRY_SIZE).map_or_else(
        |e| e.to_string(),
        |bytes| which_row(&honest.database, &bytes),
    )
}

#[test]
fn the_encoder_refuses_a_config_that_is_not_one_row_per_coefficient() {
    let mut honest = honest(ROWS);
    for rows_per_shard in [FOREIGN_ROWS_PER_SHARD, RING_DIM - 1, RING_DIM + 1] {
        let config = config_with_rows_per_shard(rows_per_shard);
        match encode_database(&honest.database, ENTRY_SIZE, &honest.crs.params, &config) {
            Err(refusal) => {
                let message = refusal.to_string();
                assert!(message.contains(&rows_per_shard.to_string()), "{message}");
                assert!(message.contains(&RING_DIM.to_string()), "{message}");
            }
            Ok(shards) => {
                let served = EncodedDatabase { shards, config };
                let (state, client_query) = addressed_query(&mut honest);
                let response =
                    respond_inspiring(&honest.crs, &served, &client_query).expect("respond");
                panic!(
                    "SILENT WRONG: encode_database accepted {rows_per_shard} rows per shard at \
                     ring_dim {RING_DIM}, and row {TARGET} came back as {}",
                    served_row(&honest, &state, &response)
                );
            }
        }
    }
}

#[test]
fn every_responder_refuses_a_database_sharded_at_another_row_count() {
    let mut honest = honest(ROWS);
    let cache = ServerInspiringCache::new(&honest.crs, &honest.encoded_db).expect("cache");
    let foreign = EncodedDatabase {
        shards: foreign_shards(&honest.database, &honest.crs.params),
        config: config_with_rows_per_shard(FOREIGN_ROWS_PER_SHARD),
    };
    let (state, client_query) = addressed_query(&mut honest);
    let crs = &honest.crs;
    let outcomes: [(&str, Result<ServerResponse>); 6] = [
        ("respond", respond(crs, &foreign, &client_query)),
        (
            "respond_one_packing",
            respond_one_packing(crs, &foreign, &client_query),
        ),
        (
            "respond_inspiring",
            respond_inspiring(crs, &foreign, &client_query),
        ),
        (
            "respond_inspiring_cached",
            respond_inspiring_cached(crs, &foreign, &client_query, &cache),
        ),
        (
            "respond_inspiring_cached_with_session",
            respond_inspiring_cached_with_session(crs, &foreign, &client_query, &cache, None),
        ),
        (
            "respond_sequential",
            respond_sequential(crs, &foreign, &client_query),
        ),
    ];
    for (operation, outcome) in outcomes {
        match outcome {
            Err(refusal) => {
                let message = refusal.to_string();
                assert!(message.contains(operation), "{message}");
                assert!(
                    message.contains(&FOREIGN_ROWS_PER_SHARD.to_string()),
                    "{message}"
                );
                assert!(message.contains(&RING_DIM.to_string()), "{message}");
            }
            Ok(response) => panic!(
                "SILENT WRONG: {operation} answered from a database that declares \
                 {FOREIGN_ROWS_PER_SHARD} rows per shard at ring_dim {RING_DIM}, and row \
                 {TARGET} came back as {}",
                served_row(&honest, &state, &response)
            ),
        }
    }
}

/// Fewer rows than a shard holds is a data size, not a geometry: the last shard is partial
/// and still addressed one row per coefficient.
#[test]
fn a_partial_last_shard_and_a_database_smaller_than_one_shard_still_serve() {
    for (rows, target) in [(RING_DIM + 44, RING_DIM as u64 + 43), (10, 7)] {
        let mut honest = honest(rows);
        assert_eq!(
            honest.encoded_db.config.entries_per_shard(),
            RING_DIM as u64
        );
        assert_eq!(honest.encoded_db.shards.len(), rows.div_ceil(RING_DIM));
        let (state, client_query) = query(
            &honest.crs,
            target,
            &honest.encoded_db.config,
            &honest.sk,
            &mut honest.sampler,
        )
        .expect("query");
        let response = respond_inspiring(&honest.crs, &honest.encoded_db, &client_query)
            .expect("a partial shard is served");
        let bytes = extract_inspiring(&honest.crs, &state, &response, ENTRY_SIZE).expect("extract");
        assert_eq!(
            bytes,
            row(&honest.database, target),
            "rows {rows}, target {target}"
        );
    }
}
