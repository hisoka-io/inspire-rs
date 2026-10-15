//! A shard that has folded the offline phase's first step over its columns serves the
//! bytes it served before, through every responder, and stops serving the moment its
//! columns are no longer the ones it folded.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test-target harness; an abort here is the failure report"
)]

mod packing_key_fixture;

use packing_key_fixture::{inline_query, modulus_sets, params_for, response_bytes, Cell};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use raven_inspire::inspiring::{PackParams, ShardPackingStore};
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, ShardConfig};
use raven_inspire::pir::{
    encode_database, extract_inspiring, resolve_packing_seat, respond_seeded_batch,
    respond_seeded_inspiring, respond_seeded_inspiring_cached,
    respond_seeded_inspiring_cached_with_session, respond_seeded_with_seat, setup_with_rng,
    ClientSession, EncodedDatabase, PackingKeySource, SeededBatchQuery, SeededQueryBatch,
    ServerCrs, ServerInspiringCache, ServerSessionStore, ShardData,
};

fn with_stores(cell: &Cell) -> EncodedDatabase {
    let mut encoded = cell.encoded.clone();
    let ctx = cell.params.ntt_context();
    for shard in &mut encoded.shards {
        shard
            .precompute_packing(cell.cache.pack_params(), &ctx)
            .expect("precompute");
    }
    encoded
}

fn batch_from(packing_key: &PackingKeySource, queries: &[SeededBatchQuery]) -> SeededQueryBatch {
    bincode::deserialize(&bincode::serialize(&(packing_key, queries)).unwrap()).unwrap()
}

/// One row request through the InspiRING responders: a registered seat and an inline
/// key always, the batch, resolved-seat, cached and uncached responders when `all`.
fn served_by_responders(
    cell: &Cell,
    encoded: &EncodedDatabase,
    query: &SeededBatchQuery,
    all: bool,
) -> Vec<(&'static str, Vec<u8>)> {
    let keys = cell.wire_keys();
    let ctx = cell.params.ntt_context();
    let store = ServerSessionStore::new();
    let handle = store
        .register_server_side(keys.clone(), cell.cache.pack_params(), &ctx)
        .unwrap();
    let inline = inline_query(query, &keys);
    let mut by_handle = inline.clone();
    by_handle.inspiring_packing_keys = None;
    by_handle.session_handle = Some(handle);
    let (crs, cache) = (&cell.crs, &cell.cache);

    let mut served = vec![
        (
            "registered seat",
            respond_seeded_inspiring_cached_with_session(
                crs,
                encoded,
                &by_handle,
                cache,
                Some(&store),
            )
            .unwrap(),
        ),
        (
            "inline key",
            respond_seeded_inspiring_cached_with_session(crs, encoded, &inline, cache, None)
                .unwrap(),
        ),
    ];
    if all {
        for (batch_name, seat_name, source) in [
            (
                "batch, inline key",
                "resolved seat, inline key",
                PackingKeySource::Inline(keys.clone()),
            ),
            (
                "batch, by handle",
                "resolved seat, by handle",
                PackingKeySource::Registered(handle),
            ),
        ] {
            let batch = batch_from(&source, std::slice::from_ref(query));
            let mut responses =
                respond_seeded_batch(crs, encoded, &batch, cache, Some(&store)).unwrap();
            served.push((batch_name, responses.pop().unwrap()));

            let seat = resolve_packing_seat(crs, cache, &source, Some(&store)).unwrap();
            served.push((
                seat_name,
                respond_seeded_with_seat(crs, encoded, query, cache, &seat).unwrap(),
            ));
        }
        served.push((
            "cached responder",
            respond_seeded_inspiring_cached(crs, encoded, &inline, cache).unwrap(),
        ));
        served.push((
            "uncached responder",
            respond_seeded_inspiring(crs, encoded, &inline).unwrap(),
        ));
    }
    served
        .iter()
        .map(|(name, response)| (*name, response_bytes(response)))
        .collect()
}

fn assert_same_bytes_with_and_without_stores(cell: &mut Cell, indices: &[u64], label: &str) {
    let precomputed = with_stores(cell);
    assert!(cell
        .encoded
        .shards
        .iter()
        .all(|s| s.packing_store().is_none()));
    assert!(precomputed
        .shards
        .iter()
        .all(|s| s.packing_store().is_some()));
    for (position, &index) in indices.iter().enumerate() {
        let query = cell.next_query(index);
        let per_query = served_by_responders(cell, &cell.encoded, &query, position == 0);
        let from_store = served_by_responders(cell, &precomputed, &query, position == 0);
        assert_eq!(per_query.len(), from_store.len());
        for ((name, expected), (_, served)) in per_query.iter().zip(&from_store) {
            assert_eq!(expected, &per_query[0].1, "{label} index {index}: {name}");
            assert_eq!(
                served, expected,
                "{label} index {index}: {name} moved its bytes under a packing store"
            );
        }
        let response = bincode::deserialize(&from_store[0].1).unwrap();
        cell.assert_decodes(index, &response);
    }
}

#[test]
fn production_cells_serve_the_same_bytes_from_a_store_at_gamma_128_256_512() {
    let params = InspireParams::secure_128_d2048();
    let d = params.ring_dim as u64;
    let mut rng = ChaCha20Rng::seed_from_u64(0x5354_4f00);
    for (entry_size, random_rows) in [(256usize, 6), (512, 4), (1024, 0)] {
        let gamma = raven_inspire::num_columns(entry_size);
        let mut cell = Cell::build(&params, entry_size, 0x5354_4f00 + entry_size as u64);
        let mut indices = vec![0, d - 1, d, 3 * d - 1];
        indices.extend((0..random_rows).map(|_| rng.gen_range(0..3 * d)));
        assert_same_bytes_with_and_without_stores(&mut cell, &indices, &format!("gamma {gamma}"));
    }
}

#[test]
fn constant_and_striped_tables_serve_the_same_bytes_from_a_store() {
    let params = InspireParams::secure_128_d2048();
    let d = params.ring_dim;
    let entry_size = 256;
    type Fill = fn(usize) -> u8;
    let fills: [(&str, Fill); 3] = [
        ("all zero", |_| 0),
        ("all ones", |_| 0xff),
        ("striped", |i| if i % 2 == 0 { 0xff } else { 0 }),
    ];
    for (label, fill) in fills {
        let mut cell = Cell::build(&params, entry_size, 0x5354_4f10);
        cell.db = (0..3 * d * entry_size).map(fill).collect();
        cell.encoded.shards =
            encode_database(&cell.db, entry_size, &params, &cell.encoded.config).unwrap();
        let rows = [0, d as u64 - 1, 2 * d as u64 + 5];
        assert_same_bytes_with_and_without_stores(&mut cell, &rows, label);
    }
}

#[test]
fn every_modulus_set_and_width_serves_the_same_bytes_from_a_store() {
    for (name, moduli) in modulus_sets() {
        let params = params_for(256, &moduli, 3);
        for entry_size in [2usize, 4, 32, 256] {
            let mut cell = Cell::build(&params, entry_size, 0x5354_4f20 + entry_size as u64);
            assert_same_bytes_with_and_without_stores(
                &mut cell,
                &[0, 255, 256, 700, 767],
                &format!("{name} entry {entry_size}"),
            );
        }
    }
}

/// A table whose last shard is part full, as a live table is between two fills.
struct LiveTable {
    params: InspireParams,
    entry_size: usize,
    db: Vec<u8>,
    crs: ServerCrs,
    encoded: EncodedDatabase,
    cache: ServerInspiringCache,
    session: ClientSession,
    sampler: GaussianSampler,
}

fn row_bytes(row: usize, entry_size: usize) -> Vec<u8> {
    (0..entry_size)
        .map(|byte| (row * 31 + byte * 7 + 1) as u8)
        .collect()
}

impl LiveTable {
    fn build(rows: usize) -> Self {
        let params = params_for(256, &modulus_sets()[0].1, 3);
        let entry_size = 64;
        let db: Vec<u8> = (0..rows)
            .flat_map(|row| row_bytes(row, entry_size))
            .collect();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0x4c49_5645);
        let mut rng = ChaCha20Rng::seed_from_u64(0x4c49_5646);
        let (crs, encoded, secret_key) =
            setup_with_rng(&params, &db, entry_size, &mut sampler, &mut rng).unwrap();
        let cache = ServerInspiringCache::new(&crs, &encoded).unwrap();
        let session = ClientSession::new(crs.clone(), secret_key, &mut sampler).unwrap();
        Self {
            params,
            entry_size,
            db,
            crs,
            encoded,
            cache,
            session,
            sampler,
        }
    }

    fn precompute(&mut self, shard: usize) {
        self.encoded.shards[shard]
            .precompute_packing(self.cache.pack_params(), &self.params.ntt_context())
            .unwrap();
    }

    /// Append rows the way a serving engine does: re-encode the last shard and put
    /// the new columns in the slot the old ones held.
    fn append_rows(&mut self, count: usize) {
        let rows = self.db.len() / self.entry_size;
        for row in rows..rows + count {
            self.db.extend(row_bytes(row, self.entry_size));
        }
        let d = self.params.ring_dim;
        let last = self.encoded.shards.len() - 1;
        let shard_bytes = &self.db[last * d * self.entry_size..];
        let one_shard = ShardConfig {
            total_entries: (shard_bytes.len() / self.entry_size) as u64,
            ..self.encoded.config.clone()
        };
        let mut rebuilt =
            encode_database(shard_bytes, self.entry_size, &self.params, &one_shard).unwrap();
        assert_eq!(rebuilt.len(), 1);
        self.encoded.shards[last].polynomials = rebuilt.pop().unwrap().polynomials;
        self.encoded.config.total_entries = (rows + count) as u64;
    }

    fn respond(&mut self, row: usize) -> Result<Vec<u8>, String> {
        let (state, query) = self
            .session
            .query_seeded(row as u64, &self.encoded.config, &mut self.sampler)
            .unwrap();
        let response = respond_seeded_inspiring_cached_with_session(
            &self.crs,
            &self.encoded,
            &query,
            &self.cache,
            None,
        )
        .map_err(|e| e.to_string())?;
        Ok(extract_inspiring(&self.crs, &state, &response, self.entry_size).unwrap())
    }
}

#[test]
fn a_shard_that_grew_refuses_the_store_folded_before_it_grew() {
    let d = 256;
    let mut table = LiveTable::build(2 * d + 100);
    assert_eq!(table.encoded.shards.len(), 3);
    for shard in 0..3 {
        table.precompute(shard);
    }
    assert_eq!(
        table.respond(2 * d + 99).unwrap(),
        row_bytes(2 * d + 99, 64)
    );

    table.append_rows(20);
    assert!(table.encoded.shards[2].packing_store().is_some());
    for row in [2 * d + 99, 2 * d + 100, 2 * d + 119, 2 * d] {
        let refusal = table
            .respond(row)
            .expect_err("the old store must not answer");
        assert!(
            refusal.contains("shard 2:")
                && refusal.contains("is not the column the store was built from")
                && refusal.contains("rebuild the store"),
            "row {row}: {refusal}"
        );
    }
    assert_eq!(table.respond(7).unwrap(), row_bytes(7, 64));
    assert_eq!(table.respond(d + 7).unwrap(), row_bytes(d + 7, 64));

    table.precompute(2);
    for row in [2 * d + 99, 2 * d + 100, 2 * d + 119] {
        assert_eq!(table.respond(row).unwrap(), row_bytes(row, 64));
    }

    table.append_rows(1);
    table.encoded.shards[2].drop_packing_store();
    assert_eq!(
        table.respond(2 * d + 120).unwrap(),
        row_bytes(2 * d + 120, 64)
    );
}

#[test]
fn any_edit_to_a_column_is_refused_until_it_is_undone_or_the_store_rebuilt() {
    let d = 256;
    let mut table = LiveTable::build(3 * d);
    table.precompute(1);
    let row = d + 40;
    assert_eq!(table.respond(row).unwrap(), row_bytes(row, 64));

    let modulus = table.params.q;
    type Edit = fn(&mut ShardData, u64);
    let edits: [(&str, Edit); 4] = [
        ("one word up by one", |shard, _| {
            shard.polynomials[31].coeffs_mut()[255] += 1;
        }),
        ("one word up by the modulus", |shard, modulus| {
            shard.polynomials[0].coeffs_mut()[0] += modulus;
        }),
        ("two columns swapped", |shard, _| {
            shard.polynomials.swap(4, 5);
        }),
        ("two rows swapped", |shard, _| {
            for column in &mut shard.polynomials {
                column.coeffs_mut().swap(10, 11);
            }
        }),
    ];
    for (label, edit) in edits {
        let untouched = table.encoded.shards[1].clone();
        edit(&mut table.encoded.shards[1], modulus);
        let refusal = table.respond(row).expect_err(label);
        assert!(refusal.contains("shard 1:"), "{label}: {refusal}");
        assert_eq!(
            table.respond(7).unwrap(),
            row_bytes(7, 64),
            "{label}: shard 0"
        );

        table.encoded.shards[1] = untouched;
        assert_eq!(table.respond(row).unwrap(), row_bytes(row, 64), "{label}");
    }
}

#[test]
fn a_store_follows_clones_and_never_reaches_the_encoded_bytes() {
    let d = 256;
    let mut table = LiveTable::build(3 * d);
    let plain_bytes = bincode::serialize(&table.encoded).unwrap();
    for shard in 0..3 {
        table.precompute(shard);
    }
    assert_eq!(bincode::serialize(&table.encoded).unwrap(), plain_bytes);

    let decoded: EncodedDatabase = bincode::deserialize(&plain_bytes).unwrap();
    assert!(decoded.shards.iter().all(|s| s.packing_store().is_none()));

    let cloned = table.encoded.shards[0].clone();
    assert_eq!(
        cloned.packing_store().map(ShardPackingStore::payload_bytes),
        Some(32 * 256 * 8)
    );
    let rebuilt = ShardData::new(0, cloned.polynomials.clone());
    assert!(rebuilt.packing_store().is_none());

    let narrower = PackParams::try_new(&table.params, 16).unwrap();
    let refusal = table.encoded.shards[0]
        .precompute_packing(&narrower, &table.params.ntt_context())
        .expect_err("a store of another width");
    assert!(refusal.to_string().contains("shard 0:"), "{refusal}");
    assert!(
        table.encoded.shards[0].packing_store().is_some(),
        "a refused rebuild leaves the working store in place"
    );
}

/// The store is the size of the shard's columns: gamma x 16 KiB at d = 2048.
#[test]
fn a_production_store_is_gamma_times_sixteen_kibibytes() {
    let params = InspireParams::secure_128_d2048();
    let ctx = params.ntt_context();
    for (entry_size, payload) in [
        (256usize, 2_097_152usize),
        (512, 4_194_304),
        (1024, 8_388_608),
    ] {
        let cell = Cell::build(&params, entry_size, 0x5354_4f30);
        let mut shard = cell.encoded.shards[0].clone();
        shard
            .precompute_packing(cell.cache.pack_params(), &ctx)
            .unwrap();
        let store = shard.packing_store().unwrap();
        let gamma = raven_inspire::num_columns(entry_size);
        assert_eq!(store.num_to_pack(), gamma);
        assert_eq!(store.payload_bytes(), payload);
        assert_eq!(payload, gamma * 16 * 1024);
        let overhead = store.resident_bytes() - payload;
        assert!(
            (32_768..32_768 + gamma * 128 + 256).contains(&overhead),
            "gamma {gamma}: {overhead} bytes around the sums"
        );
    }
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig {
        cases: 24,
        .. proptest::prelude::ProptestConfig::default()
    })]

    /// Any table, any row, any shipped modulus set: a store changes no byte.
    #[test]
    fn a_store_changes_no_response_byte(
        seed in proptest::prelude::any::<u64>(),
        modulus_set in 0usize..4,
        width in 0usize..5,
        row in 0u64..768,
    ) {
        let params = params_for(256, &modulus_sets()[modulus_set].1, 3);
        let entry_size = [2usize, 8, 32, 64, 256][width];
        let mut cell = Cell::build(&params, entry_size, seed);
        let precomputed = with_stores(&cell);
        let query = cell.next_query(row);
        let keys = cell.wire_keys();
        let respond = |encoded| {
            response_bytes(
                &respond_seeded_inspiring_cached(
                    &cell.crs,
                    encoded,
                    &inline_query(&query, &keys),
                    &cell.cache,
                )
                .unwrap(),
            )
        };
        proptest::prop_assert_eq!(respond(&precomputed), respond(&cell.encoded));
    }
}
