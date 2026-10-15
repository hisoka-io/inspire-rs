//! One packing key serves a whole request. A batch carries the key once, inline or as
//! a handle to a registered seat, and each of its responses is the bytes the same
//! query gets when it carries that key inline on its own.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test-target harness; an abort here is the failure report"
)]

mod packing_key_fixture;

use packing_key_fixture::{fnv1a, inline_query, modulus_sets, params_for, response_bytes, Cell};
use raven_inspire::inspiring::ClientPackingKeys;
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::InspireParams;
use raven_inspire::pir::{
    resolve_packing_seat, respond_seeded_batch, respond_seeded_inspiring_cached,
    respond_seeded_with_seat, PackingKeySource, SeededBatchQuery, SeededQueryBatch,
    ServerSessionHandle, ServerSessionStore,
};

const RECORDED: &str = include_str!("fixtures/packing_seat_response_kat.txt");

/// A batch as the wire carries it, from its two fields; clients only get one from a
/// session, so tests that need a hand-built batch go through the encoding.
fn batch_from(packing_key: &PackingKeySource, queries: &[SeededBatchQuery]) -> SeededQueryBatch {
    bincode::deserialize(&bincode::serialize(&(packing_key, queries)).unwrap()).unwrap()
}

fn per_query_inline(
    cell: &Cell,
    keys: &ClientPackingKeys,
    queries: &[SeededBatchQuery],
) -> Vec<Vec<u8>> {
    queries
        .iter()
        .map(|query| {
            response_bytes(
                &respond_seeded_inspiring_cached(
                    &cell.crs,
                    &cell.encoded,
                    &inline_query(query, keys),
                    &cell.cache,
                )
                .expect("inline respond"),
            )
        })
        .collect()
}

fn batch_bytes(
    cell: &Cell,
    batch: &SeededQueryBatch,
    store: Option<&ServerSessionStore>,
) -> Vec<Vec<u8>> {
    respond_seeded_batch(&cell.crs, &cell.encoded, batch, &cell.cache, store)
        .expect("batch respond")
        .iter()
        .map(response_bytes)
        .collect()
}

/// The production cell from the recorded fixture, answered as one batch: inline key
/// and registered seat give the per-query inline bytes, which are the recorded ones.
#[test]
fn a_production_batch_serves_the_recorded_per_query_bytes() {
    let params = InspireParams::secure_128_d2048();
    let d = params.ring_dim as u64;
    let (entry_size, seed) = (256usize, 0x5245_5300u64);
    let indices = [0, 1, d - 1, d, 2 * d + 17, 3 * d - 1];
    let mut cell = Cell::build(&params, entry_size, seed);
    let keys = cell.wire_keys();
    let queries: Vec<SeededBatchQuery> = indices.iter().map(|&i| cell.next_query(i)).collect();

    let inline = batch_from(&PackingKeySource::Inline(keys.clone()), &queries);
    let served = batch_bytes(&cell, &inline, None);
    assert_eq!(served, per_query_inline(&cell, &keys, &queries));

    let store = ServerSessionStore::new();
    let handle = store
        .register_server_side(keys, cell.cache.pack_params(), &params.ntt_context())
        .unwrap();
    let registered = batch_from(&PackingKeySource::Registered(handle), &queries);
    assert_eq!(batch_bytes(&cell, &registered, Some(&store)), served);

    for (index, bytes) in indices.iter().zip(&served) {
        let line = format!(
            "RESP d2048 entry={entry_size} seed={seed:#x} index={index} len={} fnv={:#018x}",
            bytes.len(),
            fnv1a(bytes)
        );
        assert!(
            RECORDED.lines().any(|recorded| recorded == line),
            "batch bytes moved; no recorded line equals:\n{line}"
        );
    }
}

/// Session-built batches of 1, 4 and 16 fresh queries at every modulus set, inline
/// and by handle, against the per-query inline answers. Every response also decodes.
#[test]
fn session_batches_match_per_query_inline_answers() {
    for (name, moduli) in modulus_sets() {
        let params = params_for(256, &moduli, 3);
        for entry_size in [4usize, 64, 256] {
            let cell = Cell::build(&params, entry_size, 0x4241_5400 + entry_size as u64);
            let mut sampler = GaussianSampler::new(params.sigma);
            for count in [1usize, 4, 16] {
                let indices: Vec<u64> = (0..count as u64).map(|i| (i * 53 + 7) % 768).collect();

                let (_, inline) = cell
                    .session
                    .query_batch(&indices, &cell.encoded.config, &mut sampler)
                    .unwrap();
                let PackingKeySource::Inline(keys) = inline.packing_key() else {
                    panic!("an unregistered session must inline its key");
                };
                assert_eq!(inline.queries().len(), count);
                let inline: SeededQueryBatch =
                    bincode::deserialize(&bincode::serialize(&inline).unwrap()).unwrap();
                let expected = per_query_inline(&cell, keys, inline.queries());
                assert_eq!(
                    batch_bytes(&cell, &inline, None),
                    expected,
                    "{name} entry {entry_size} n {count}: inline batch"
                );

                let store = ServerSessionStore::new();
                let mut registered_session = cell.session.clone();
                let handle = registered_session.register_with(&store).unwrap().unwrap();
                let (_, by_handle) = registered_session
                    .query_batch(&indices, &cell.encoded.config, &mut sampler)
                    .unwrap();
                assert!(
                    matches!(by_handle.packing_key(), PackingKeySource::Registered(h) if *h == handle),
                    "a registered session must send its handle and no key"
                );
                let expected = per_query_inline(&cell, keys, by_handle.queries());
                assert_eq!(
                    batch_bytes(&cell, &by_handle, Some(&store)),
                    expected,
                    "{name} entry {entry_size} n {count}: registered batch"
                );
            }
        }
    }
}

#[test]
fn a_production_session_batch_decodes_every_row() {
    let params = InspireParams::secure_128_d2048();
    let mut cell = Cell::build(&params, 256, 0x4445_4300);
    let indices = [5u64, 2048, 4097, 6143];
    let (states, batch) = cell
        .session
        .query_batch(&indices, &cell.encoded.config, &mut cell.sampler)
        .unwrap();
    let responses =
        respond_seeded_batch(&cell.crs, &cell.encoded, &batch, &cell.cache, None).unwrap();
    assert_eq!(responses.len(), indices.len());
    for ((index, state), response) in indices.iter().zip(&states).zip(&responses) {
        assert_eq!(state.index, *index);
        cell.assert_decodes(*index, response);
    }
}

/// A dispatcher resolves the seat once and answers each query on its own.
#[test]
fn a_resolved_seat_answers_each_query_like_the_batch() {
    let params = params_for(256, &modulus_sets()[0].1, 3);
    let mut cell = Cell::build(&params, 64, 0x5345_4100);
    let keys = cell.wire_keys();
    let queries: Vec<SeededBatchQuery> = [3u64, 300, 767]
        .iter()
        .map(|&i| cell.next_query(i))
        .collect();
    let source = PackingKeySource::Inline(keys.clone());
    let seat = resolve_packing_seat(&cell.crs, &cell.cache, &source, None).unwrap();
    let one_by_one: Vec<Vec<u8>> = queries
        .iter()
        .map(|query| {
            response_bytes(
                &respond_seeded_with_seat(&cell.crs, &cell.encoded, query, &cell.cache, &seat)
                    .unwrap(),
            )
        })
        .collect();
    assert_eq!(
        one_by_one,
        batch_bytes(&cell, &batch_from(&source, &queries), None)
    );
    assert_eq!(one_by_one, per_query_inline(&cell, &keys, &queries));
}

/// The comparison can fail: the same queries under another client's key are other bytes.
#[test]
fn a_batch_under_another_key_serves_different_bytes() {
    let params = params_for(256, &modulus_sets()[0].1, 3);
    let mut cell = Cell::build(&params, 64, 0x4f54_4800);
    let other = Cell::build(&params, 64, 0x4f54_4801);
    let queries: Vec<SeededBatchQuery> = [1u64, 2].iter().map(|&i| cell.next_query(i)).collect();
    let own = batch_bytes(
        &cell,
        &batch_from(&PackingKeySource::Inline(cell.wire_keys()), &queries),
        None,
    );
    let foreign = batch_bytes(
        &cell,
        &batch_from(&PackingKeySource::Inline(other.wire_keys()), &queries),
        None,
    );
    assert_eq!(own.len(), foreign.len());
    for (own, foreign) in own.iter().zip(&foreign) {
        assert_ne!(own, foreign);
    }
}

/// Request bytes at the production ring. The key is 46,252 B at every width and a
/// query is 15,477 B, so a batch is `46,264 + 15,477 n` inline and `20 + 15,477 n`
/// by handle.
#[test]
fn production_request_sizes_are_pinned() {
    let params = InspireParams::secure_128_d2048();
    for entry_size in [256usize, 512] {
        let mut cell = Cell::build(&params, entry_size, 0x5349_5a00);
        let key_bytes = bincode::serialize(cell.session.packing_keys().unwrap()).unwrap();
        assert_eq!(key_bytes.len(), 46_252);

        let mut sizes = Vec::new();
        for count in [1usize, 4, 16] {
            let indices: Vec<u64> = (0..count as u64).collect();
            let (_, batch) = cell
                .session
                .query_batch(&indices, &cell.encoded.config, &mut cell.sampler)
                .unwrap();
            sizes.push(bincode::serialize(&batch).unwrap().len());
        }
        assert_eq!(
            sizes,
            [61_741, 108_172, 293_896],
            "inline, entry size {entry_size}"
        );

        let store = ServerSessionStore::new();
        let mut session = cell.session.clone();
        session.register_with(&store).unwrap().expect("handle");
        let mut sizes = Vec::new();
        for count in [1usize, 4, 16] {
            let indices: Vec<u64> = (0..count as u64).collect();
            let (_, batch) = session
                .query_batch(&indices, &cell.encoded.config, &mut cell.sampler)
                .unwrap();
            sizes.push(bincode::serialize(&batch).unwrap().len());
        }
        assert_eq!(
            sizes,
            [15_497, 61_928, 247_652],
            "by handle, entry size {entry_size}"
        );
    }
}

/// Building the batch again resends the same key bytes.
#[test]
fn every_batch_of_a_session_carries_the_same_key_bytes() {
    let params = params_for(256, &modulus_sets()[0].1, 3);
    let mut cell = Cell::build(&params, 32, 0x5245_5400);
    let upload = bincode::serialize(cell.session.packing_keys().unwrap()).unwrap();
    let mut rows = std::collections::BTreeSet::new();
    for _ in 0..3 {
        let (_, batch) = cell
            .session
            .query_batch(&[11, 12], &cell.encoded.config, &mut cell.sampler)
            .unwrap();
        let PackingKeySource::Inline(keys) = batch.packing_key() else {
            panic!("an unregistered session must inline its key");
        };
        assert_eq!(bincode::serialize(keys).unwrap(), upload);
        rows.insert(bincode::serialize(batch.queries()).unwrap());
    }
    assert_eq!(
        rows.len(),
        3,
        "the row ciphertexts are fresh on every build"
    );
}

#[test]
fn a_batch_is_refused_whole() {
    let params = params_for(256, &modulus_sets()[0].1, 3);
    let mut cell = Cell::build(&params, 64, 0x5245_4600);
    let narrow = Cell::build(&params, 32, 0x5245_4601);
    let keys = cell.wire_keys();
    let queries: Vec<SeededBatchQuery> = [1u64, 2, 3].iter().map(|&i| cell.next_query(i)).collect();
    let refused = |batch: &SeededQueryBatch, store: Option<&ServerSessionStore>| {
        respond_seeded_batch(&cell.crs, &cell.encoded, batch, &cell.cache, store)
            .expect_err("the batch must be refused")
            .to_string()
    };

    let unknown = batch_from(
        &PackingKeySource::Registered(ServerSessionHandle(u64::MAX)),
        &queries,
    );
    let message = refused(&unknown, Some(&ServerSessionStore::new()));
    assert!(message.contains("no entry for session_handle"), "{message}");
    let message = refused(&unknown, None);
    assert!(
        message.contains("no ServerSessionStore was supplied"),
        "{message}"
    );

    let foreign_width = batch_from(&PackingKeySource::Inline(narrow.wire_keys()), &queries);
    let message = refused(&foreign_width, None);
    assert!(message.contains("width mismatch"), "{message}");

    let mut short = keys.clone();
    short.y_body.pop();
    let message = refused(
        &batch_from(&PackingKeySource::Inline(short), &queries),
        None,
    );
    assert!(message.contains("material mismatch"), "{message}");

    let mut off_table = queries.clone();
    off_table[2].shard_id = 99;
    let message = refused(
        &batch_from(&PackingKeySource::Inline(keys.clone()), &off_table),
        None,
    );
    assert!(message.contains("shard 99 not found"), "{message}");

    let mut two_rows = queries.clone();
    let extra = two_rows[1].rgsw_ciphertext.rows[0].clone();
    two_rows[1].rgsw_ciphertext.rows.push(extra);
    two_rows[1].rgsw_ciphertext.gadget.len = 2;
    let message = refused(
        &batch_from(&PackingKeySource::Inline(keys), &two_rows),
        None,
    );
    assert!(message.contains("RGSW gadget mismatch"), "{message}");
}

#[test]
fn a_batch_row_declaring_the_ntt_domain_does_not_decode() {
    let params = params_for(256, &modulus_sets()[0].1, 3);
    let mut cell = Cell::build(&params, 32, 0x4e54_5400);
    let mut query = cell.next_query(4);
    query.rgsw_ciphertext.rows[0].b.force_ntt_domain();
    let wire =
        bincode::serialize(&(PackingKeySource::Inline(cell.wire_keys()), vec![query])).unwrap();
    let message = bincode::deserialize::<SeededQueryBatch>(&wire)
        .expect_err("an NTT-flagged row must be refused at decode")
        .to_string();
    assert!(
        message.contains("coefficient-domain RGSW row 0 b required"),
        "{message}"
    );
}
