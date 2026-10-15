//! The approximate packing gadget is an option on the parameter set: the InspiRING
//! packing key carries no row for the lowest gadget digit and the server rounds that
//! digit away. A set without the option serializes to the bytes it always had; a set
//! with it has a layout of its own, decodes through every responder, and is refused
//! wherever the two sides disagree about it.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test-target harness; an abort here is the failure report"
)]

mod packing_key_fixture;

use bincode::Options;
use packing_key_fixture::{inline_query, modulus_sets, params_for, response_bytes, Cell};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use raven_inspire::cost::CostEstimator;
use raven_inspire::inspiring::{
    generate_rotations, packing_offline, packing_online, packing_online_gather, ClientPackingKeys,
    OfflinePackingKeys, PackParams,
};
use raven_inspire::math::{GaussianSampler, Poly};
use raven_inspire::params::{InspireParams, InspireVariant, SecurityLevel, DEFAULT_Q_2CRT_30BIT};
use raven_inspire::pir::{
    resolve_packing_seat, respond_seeded_batch, respond_seeded_inspiring,
    respond_seeded_inspiring_cached, respond_seeded_inspiring_cached_with_session,
    respond_seeded_with_seat, ClientSession, EncodedDatabase, PackingKeySeat, PackingKeySource,
    SeededBatchQuery, SeededQueryBatch, ServerCrs, ServerInspiringCache, ServerSessionStore,
};
use raven_inspire::rlwe::RlweSecretKey;
use serde::{Deserialize, Serialize};

/// `InspireParams` as it derived its serde before the option existed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename = "InspireParams")]
struct ParamsBeforeTheOption {
    ring_dim: usize,
    q: u64,
    crt_moduli: Vec<u64>,
    p: u64,
    sigma: f64,
    gadget_base: u64,
    query_gadget_len: usize,
    packing_gadget_len: usize,
    security_level: SecurityLevel,
}

impl ParamsBeforeTheOption {
    fn of(params: &InspireParams) -> Self {
        Self {
            ring_dim: params.ring_dim,
            q: params.q,
            crt_moduli: params.crt_moduli.clone(),
            p: params.p,
            sigma: params.sigma,
            gadget_base: params.gadget_base,
            query_gadget_len: params.query_gadget_len,
            packing_gadget_len: params.packing_gadget_len,
            security_level: params.security_level,
        }
    }

    fn as_exact_set(&self) -> InspireParams {
        InspireParams {
            ring_dim: self.ring_dim,
            q: self.q,
            crt_moduli: self.crt_moduli.clone(),
            p: self.p,
            sigma: self.sigma,
            gadget_base: self.gadget_base,
            query_gadget_len: self.query_gadget_len,
            packing_gadget_len: self.packing_gadget_len,
            approximate_packing_gadget: false,
            security_level: self.security_level,
        }
    }
}

fn approximate(params: &InspireParams) -> InspireParams {
    InspireParams {
        approximate_packing_gadget: true,
        ..params.clone()
    }
}

fn varint() -> impl Options + Copy {
    bincode::DefaultOptions::new()
}

/// Shipped presets, every shipped modulus set, and field values no preset uses.
fn exact_sets() -> Vec<InspireParams> {
    let mut sets = vec![
        InspireParams::secure_128_d2048(),
        InspireParams::secure_128_d4096(),
        InspireParams::for_scenario(1 << 20, 256, [64, 1024, 64], 1).expect("derived set"),
    ];
    for (_, moduli) in modulus_sets() {
        sets.push(params_for(256, &moduli, 3));
        sets.push(params_for(2048, &moduli, 4));
    }
    let mut rng = ChaCha20Rng::seed_from_u64(0x5345_5253);
    for limbs in 0..4usize {
        sets.push(InspireParams {
            ring_dim: rng.gen_range(0..usize::MAX),
            q: rng.gen(),
            crt_moduli: (0..limbs).map(|_| rng.gen()).collect(),
            p: rng.gen(),
            // Eighths survive a JSON round trip exactly.
            sigma: f64::from(rng.gen_range(1u32..1 << 20)) / 8.0,
            gadget_base: rng.gen(),
            query_gadget_len: rng.gen(),
            packing_gadget_len: rng.gen(),
            approximate_packing_gadget: false,
            security_level: if limbs % 2 == 0 {
                SecurityLevel::Bits128
            } else {
                SecurityLevel::Bits256
            },
        });
    }
    sets
}

/// The shipped preset, field by field in little-endian fixed width, written out by
/// hand rather than by any serializer.
const SHIPPED_PRESET_BYTES: &str =
    "000800000000000001c0ffffffffff0f010000000000000001c0ffffffffff0f\
    01000100000000009a9999999999194000001000000000000300000000000000030000000000000000000000";

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut hex, byte| {
        write!(hex, "{byte:02x}").expect("writing to a String");
        hex
    })
}

#[test]
fn exact_sets_serialize_as_they_always_did() {
    assert_eq!(
        hex(&bincode::serialize(&InspireParams::secure_128_d2048()).unwrap()),
        SHIPPED_PRESET_BYTES
    );
    for set in exact_sets() {
        let before = ParamsBeforeTheOption::of(&set);

        let fixed = bincode::serialize(&set).unwrap();
        assert_eq!(fixed, bincode::serialize(&before).unwrap(), "{set:?}");
        let decoded: InspireParams = bincode::deserialize(&fixed).unwrap();
        assert_eq!(hex(&bincode::serialize(&decoded).unwrap()), hex(&fixed));
        assert!(!decoded.approximate_packing_gadget);

        let compact = varint().serialize(&set).unwrap();
        assert_eq!(compact, varint().serialize(&before).unwrap(), "{set:?}");
        let decoded: InspireParams = varint().deserialize(&compact).unwrap();
        assert_eq!(varint().serialize(&decoded).unwrap(), compact);

        let json = serde_json::to_string(&set).unwrap();
        assert_eq!(json, serde_json::to_string(&before).unwrap());
        assert!(!json.contains("approximate"));
        let decoded: InspireParams = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, set);
    }
}

/// The comparison above can fail: one flipped field is other bytes in every format.
#[test]
fn a_set_that_differs_serializes_differently() {
    let set = InspireParams::secure_128_d2048();
    let before = ParamsBeforeTheOption::of(&set);
    let other = InspireParams {
        packing_gadget_len: 2,
        ..set.clone()
    };
    assert_ne!(
        bincode::serialize(&other).unwrap(),
        bincode::serialize(&before).unwrap()
    );
    assert_ne!(
        serde_json::to_string(&other).unwrap(),
        serde_json::to_string(&before).unwrap()
    );
    let on = approximate(&set);
    assert_ne!(
        bincode::serialize(&on).unwrap(),
        bincode::serialize(&before).unwrap()
    );
    assert_ne!(
        serde_json::to_string(&on).unwrap(),
        serde_json::to_string(&before).unwrap()
    );
    assert_ne!(on, set);
}

#[test]
fn an_approximate_set_round_trips_in_a_layout_of_its_own() {
    for exact in exact_sets() {
        let set = approximate(&exact);

        let fixed = bincode::serialize(&set).unwrap();
        let mut expected = vec![0xff; 8];
        expected.extend(bincode::serialize(&exact).unwrap());
        assert_eq!(fixed, expected);
        let decoded: InspireParams = bincode::deserialize(&fixed).unwrap();
        assert!(decoded.approximate_packing_gadget);
        assert_eq!(bincode::serialize(&decoded).unwrap(), fixed);

        let compact = varint().serialize(&set).unwrap();
        let decoded: InspireParams = varint().deserialize(&compact).unwrap();
        assert!(decoded.approximate_packing_gadget);
        assert_eq!(varint().serialize(&decoded).unwrap(), compact);
        let strict: InspireParams = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize(&fixed)
            .unwrap();
        assert!(strict.approximate_packing_gadget);

        let json = serde_json::to_string(&set).unwrap();
        assert!(
            json.ends_with(",\"approximate_packing_gadget\":true}"),
            "{json}"
        );
        let decoded: InspireParams = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, set);
        let explicit_off = json.replace(
            "\"approximate_packing_gadget\":true",
            "\"approximate_packing_gadget\":false",
        );
        let decoded: InspireParams = serde_json::from_str(&explicit_off).unwrap();
        assert_eq!(decoded, exact);
    }
}

#[test]
fn malformed_parameter_bytes_are_refused() {
    let exact = bincode::serialize(&InspireParams::secure_128_d2048()).unwrap();
    let marked = bincode::serialize(&approximate(&InspireParams::secure_128_d2048())).unwrap();
    for bytes in [&exact, &marked] {
        for cut in 0..bytes.len() {
            assert!(
                bincode::deserialize::<InspireParams>(&bytes[..cut]).is_err(),
                "a {cut}-byte prefix decoded"
            );
        }
    }
    // An exact set followed by eight more bytes is not an approximate set.
    let mut trailing = exact.clone();
    trailing.extend([0xff; 8]);
    let strict = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes();
    assert!(strict.deserialize::<InspireParams>(&trailing).is_err());
    assert!(strict.deserialize::<InspireParams>(&exact).is_ok());

    // A ring dimension equal to the marker would decode as another set, so it has no
    // exact encoding; marked, it round-trips and `validate` refuses it.
    let colliding = InspireParams {
        ring_dim: usize::MAX,
        ..InspireParams::secure_128_d2048()
    };
    if usize::BITS == 64 {
        let refusal = bincode::serialize(&colliding).unwrap_err().to_string();
        assert!(
            refusal.contains("layout marker") && !refusal.contains("  "),
            "{refusal}"
        );
        let marked = approximate(&colliding);
        let decoded: InspireParams =
            bincode::deserialize(&bincode::serialize(&marked).unwrap()).unwrap();
        assert_eq!(decoded, marked);
        assert!(decoded.validate().is_err());
    }
}

/// A reader built before the option existed must not take an approximate set for a
/// usable exact one: the two sides would then build different keys.
#[test]
fn a_reader_from_before_the_option_refuses_an_approximate_set() {
    let mut sets = vec![
        InspireParams::secure_128_d2048(),
        InspireParams::secure_128_d4096(),
    ];
    for (_, moduli) in modulus_sets() {
        sets.push(params_for(2048, &moduli, 4));
    }
    for exact in sets {
        assert!(exact.validate().is_ok(), "{exact:?}");
        let set = approximate(&exact);
        let alone = bincode::serialize(&set).unwrap();
        // Standing alone, and followed by megabytes of other fields as in a server CRS.
        let mut embedded = alone.clone();
        embedded.extend(std::iter::repeat_n(0x01, 4 << 20));
        for bytes in [&alone, &embedded] {
            match bincode::deserialize::<ParamsBeforeTheOption>(bytes) {
                Err(_) => {}
                Ok(read) => {
                    assert_eq!(read.ring_dim, usize::MAX);
                    assert!(read.as_exact_set().validate().is_err());
                }
            }
        }
        assert!(varint()
            .deserialize::<ParamsBeforeTheOption>(&varint().serialize(&set).unwrap())
            .map_or(true, |read| read.as_exact_set().validate().is_err()));
    }
}

fn small(moduli: &[u64]) -> InspireParams {
    approximate(&params_for(256, moduli, 3))
}

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

/// One row request through every InspiRING responder, with and without shard stores.
fn served_everywhere(cell: &Cell, query: &SeededBatchQuery) -> Vec<(String, Vec<u8>)> {
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

    let mut served = Vec::new();
    for (stores, encoded) in [
        ("per query", cell.encoded.clone()),
        ("stores", with_stores(cell)),
    ] {
        let encoded = &encoded;
        let mut push = |name: &str, response| {
            served.push((format!("{name}, {stores}"), response_bytes(&response)));
        };
        push(
            "registered seat",
            respond_seeded_inspiring_cached_with_session(
                crs,
                encoded,
                &by_handle,
                cache,
                Some(&store),
            )
            .unwrap(),
        );
        push(
            "inline key",
            respond_seeded_inspiring_cached_with_session(crs, encoded, &inline, cache, None)
                .unwrap(),
        );
        for (name, source) in [
            ("inline key", PackingKeySource::Inline(keys.clone())),
            ("by handle", PackingKeySource::Registered(handle)),
        ] {
            let batch = batch_from(&source, std::slice::from_ref(query));
            let mut responses =
                respond_seeded_batch(crs, encoded, &batch, cache, Some(&store)).unwrap();
            push(&format!("batch {name}"), responses.pop().unwrap());
            let seat = resolve_packing_seat(crs, cache, &source, Some(&store)).unwrap();
            push(
                &format!("resolved seat {name}"),
                respond_seeded_with_seat(crs, encoded, query, cache, &seat).unwrap(),
            );
        }
        push(
            "cached responder",
            respond_seeded_inspiring_cached(crs, encoded, &inline, cache).unwrap(),
        );
        push(
            "uncached responder",
            respond_seeded_inspiring(crs, encoded, &inline).unwrap(),
        );
    }
    served
}

fn assert_one_answer(cell: &mut Cell, indices: &[u64], label: &str) -> Vec<Vec<u8>> {
    indices
        .iter()
        .map(|&index| {
            let query = cell.next_query(index);
            let served = served_everywhere(cell, &query);
            assert_eq!(served.len(), 16);
            let (first_name, first) = &served[0];
            for (name, bytes) in &served[1..] {
                assert_eq!(
                    bytes, first,
                    "{label} index {index}: {name} vs {first_name}"
                );
            }
            first.clone()
        })
        .collect()
}

/// Under the approximate gadget every responder serves one answer, on every modulus
/// set, and that answer is not the exact gadget's.
#[test]
fn every_responder_serves_one_answer_under_the_approximate_gadget() {
    for (name, moduli) in modulus_sets() {
        for entry_size in [4usize, 32, 256] {
            let seed = 0x4150_5800 + entry_size as u64;
            let mut cell = Cell::build(&small(&moduli), entry_size, seed);
            assert_eq!(cell.cache.pack_params().key_rows(), 2);
            assert_eq!(cell.wire_keys().y_body.len(), 2);
            let label = format!("{name} entry {entry_size}");
            let approximate = assert_one_answer(&mut cell, &[0, 255, 256, 700], &label);

            let mut exact = Cell::build(&params_for(256, &moduli, 3), entry_size, seed);
            let exact = assert_one_answer(&mut exact, &[0, 255, 256, 700], &label);
            for (approximate, exact) in approximate.iter().zip(&exact) {
                assert_eq!(approximate.len(), exact.len(), "{label}: response size");
                assert_ne!(approximate, exact, "{label}");
            }
        }
    }
}

/// The production ring decodes every record width through the approximate gadget.
#[test]
fn production_cells_decode_under_the_approximate_gadget() {
    let params = approximate(&InspireParams::secure_128_d2048());
    let d = params.ring_dim as u64;
    for entry_size in [32usize, 256, 512, 1024] {
        let mut cell = Cell::build(&params, entry_size, 0x4150_5244);
        let stored = with_stores(&cell);
        for index in [0, d - 1, d, 2 * d + 17, 3 * d - 1] {
            let query = inline_query(&cell.next_query(index), &cell.wire_keys());
            let per_query =
                respond_seeded_inspiring_cached(&cell.crs, &cell.encoded, &query, &cell.cache)
                    .unwrap();
            let from_store =
                respond_seeded_inspiring_cached(&cell.crs, &stored, &query, &cell.cache).unwrap();
            assert_eq!(response_bytes(&per_query), response_bytes(&from_store));
            cell.assert_decodes(index, &per_query);
        }
    }

    let two_limb = approximate(&params_for(2048, &DEFAULT_Q_2CRT_30BIT, 3));
    let mut cell = Cell::build(&two_limb, 64, 0x4150_5245);
    for index in [0, d + 5, 3 * d - 1] {
        let query = inline_query(&cell.next_query(index), &cell.wire_keys());
        let response =
            respond_seeded_inspiring_cached(&cell.crs, &cell.encoded, &query, &cell.cache).unwrap();
        cell.assert_decodes(index, &response);
    }
}

/// What a client downloads after the served switch decodes as well.
#[cfg(feature = "mod-switch-response")]
#[test]
fn production_cells_decode_after_the_served_switch() {
    use raven_inspire::pir::mod_switch::{
        extract_inspiring_mod_switched, mod_switch_response_checked, MOD_SWITCH_TARGET_36BIT,
    };
    use raven_inspire::ServerResponse;

    let params = approximate(&InspireParams::secure_128_d2048());
    for entry_size in [256usize, 512] {
        let cell = Cell::build(&params, entry_size, 0x4150_5357);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0x4150_5358);
        for index in [3u64, 2047, 4096, 6143] {
            let (state, query) = cell
                .session
                .query_seeded(index, &cell.encoded.config, &mut sampler)
                .unwrap();
            let response =
                respond_seeded_inspiring_cached(&cell.crs, &cell.encoded, &query, &cell.cache)
                    .unwrap();
            let switched =
                mod_switch_response_checked(&params, &response, MOD_SWITCH_TARGET_36BIT).unwrap();
            let served = ServerResponse::from_binary(&switched.to_binary().unwrap()).unwrap();
            let decoded =
                extract_inspiring_mod_switched(&cell.crs, &state, &served, entry_size).unwrap();
            let start = index as usize * entry_size;
            assert_eq!(
                decoded,
                &cell.db[start..start + entry_size],
                "index {index}"
            );
        }
    }
}

/// The key is two polynomials: 30,843 B on the wire and 32,768 B in a seat, at every
/// record width.
#[test]
fn the_approximate_key_and_seat_sizes_are_pinned() {
    let exact = InspireParams::secure_128_d2048();
    let params = approximate(&exact);
    for entry_size in [32usize, 256, 512] {
        let mut cell = Cell::build(&params, entry_size, 0x5349_5a41);
        let keys = cell.session.packing_keys().unwrap().clone();
        assert_eq!(keys.y_body.len(), 2);
        assert_eq!(bincode::serialize(&keys).unwrap().len(), 30_843);

        let seat = PackingKeySeat::from_wire(
            cell.wire_keys(),
            cell.cache.pack_params(),
            &params.ntt_context(),
        )
        .unwrap();
        assert_eq!(seat.payload_bytes(), 32_768, "entry size {entry_size}");
        assert_eq!(
            seat.resident_bytes(),
            32_768 + std::mem::size_of::<PackingKeySeat>() + 2 * (std::mem::size_of::<Poly>() + 8)
        );

        let mut inline = Vec::new();
        for count in [1usize, 4, 16] {
            let indices: Vec<u64> = (0..count as u64).collect();
            let (_, batch) = cell
                .session
                .query_batch(&indices, &cell.encoded.config, &mut cell.sampler)
                .unwrap();
            inline.push(bincode::serialize(&batch).unwrap().len());
        }
        assert_eq!(inline, [46_332, 92_763, 278_487], "entry size {entry_size}");

        let store = ServerSessionStore::new();
        let mut session = cell.session.clone();
        let handle = session.register_with(&store).unwrap().expect("handle");
        assert_eq!(store.get(handle).unwrap().unwrap().payload_bytes(), 32_768);
        let mut by_handle = Vec::new();
        for count in [1usize, 4, 16] {
            let indices: Vec<u64> = (0..count as u64).collect();
            let (_, batch) = session
                .query_batch(&indices, &cell.encoded.config, &mut cell.sampler)
                .unwrap();
            by_handle.push(bincode::serialize(&batch).unwrap().len());
        }
        assert_eq!(by_handle, [15_497, 61_928, 247_652]);

        // The cost model follows the key: one polynomial of 60-bit coefficients less.
        let modeled = |set: &InspireParams| {
            CostEstimator::new(set, InspireVariant::TwoPacking, entry_size)
                .estimate()
                .communication
                .bytes
        };
        assert_eq!(modeled(&exact) - modeled(&params), 2048 * 60 / 8);
    }
}

/// A CRS and a session residue keep the option across their own serialization, so a
/// restart or a reload builds the same two-row key.
#[test]
fn the_option_survives_the_crs_and_the_session_residue() {
    let params = small(&modulus_sets()[0].1);
    let cell = Cell::build(&params, 32, 0x4352_5300);

    let wire = cell.crs.to_versioned_bytes().unwrap();
    let crs = ServerCrs::from_versioned_bytes(&wire).unwrap();
    assert!(crs.params.approximate_packing_gadget);
    assert_eq!(crs.params, cell.crs.params);
    let mut sampler = GaussianSampler::with_seed(params.sigma, 5);
    let session = ClientSession::with_fresh_secret(crs, &mut sampler).unwrap();
    assert_eq!(session.packing_keys().unwrap().y_body.len(), 2);

    let residue = serde_json::to_value(cell.session.to_residue()).unwrap();
    assert_eq!(
        residue["crs"]["params"]["approximate_packing_gadget"],
        serde_json::json!(true)
    );
    let reloaded = ClientSession::from_residue(serde_json::from_value(residue).unwrap()).unwrap();
    assert!(reloaded.crs().params.approximate_packing_gadget);
    assert_eq!(reloaded.packing_keys().unwrap().y_body.len(), 2);

    let binary = bincode::serialize(&cell.session.to_residue()).unwrap();
    let reloaded = ClientSession::from_residue(bincode::deserialize(&binary).unwrap()).unwrap();
    assert!(reloaded.crs().params.approximate_packing_gadget);

    // The exact set's CRS bytes carry no marker: its parameters open with ring_dim.
    let exact = Cell::build(&params_for(256, &modulus_sets()[0].1, 3), 32, 0x4352_5300);
    let exact_wire = exact.crs.to_versioned_bytes().unwrap();
    assert_eq!(exact_wire[16..24], 256u64.to_le_bytes());
    assert_eq!(wire[16..24], [0xff; 8]);
    assert_eq!(wire[24..32], 256u64.to_le_bytes());
}

/// The option is about the InspiRING key alone: the galois keys of the tree packing
/// keep every digit, and a tree-packed response still decodes.
#[test]
fn key_switching_matrices_keep_every_digit() {
    use raven_inspire::pir::{extract_with_variant, query, respond_with_variant};

    let params = approximate(&InspireParams::secure_128_d2048());
    let entry_size = 32;
    // Tree packing scales a column by d, so the columns stay below p / d.
    let database: Vec<u8> = (0..params.ring_dim * entry_size)
        .map(|i| if i % 2 == 0 { (i % 31) as u8 } else { 0 })
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0x4b53_4d00);
    let (crs, encoded, secret_key) =
        raven_inspire::setup(&params, &database, entry_size, &mut sampler).unwrap();
    assert!(crs
        .galois_keys
        .iter()
        .all(|key| key.gadget_len() == params.packing_gadget_len));
    let pack_params = crs.inspiring_pack_params.as_ref().unwrap();
    assert_eq!((pack_params.gadget.len, pack_params.key_rows()), (3, 2));

    for index in [0u64, 1234] {
        let (state, client_query) =
            query(&crs, index, &encoded.config, &secret_key, &mut sampler).unwrap();
        let response =
            respond_with_variant(&crs, &encoded, &client_query, InspireVariant::OnePacking)
                .unwrap();
        let decoded = extract_with_variant(
            &crs,
            &state,
            &response,
            entry_size,
            InspireVariant::OnePacking,
        )
        .unwrap();
        let start = index as usize * entry_size;
        assert_eq!(decoded, &database[start..start + entry_size]);
    }
}

fn refusal<T>(result: raven_inspire::pir::Result<T>) -> String {
    result.err().expect("refused").to_string()
}

/// A key built for one gadget is refused by a server running the other, by every
/// path that takes a key.
#[test]
fn a_key_for_the_other_gadget_is_refused() {
    let moduli = &modulus_sets()[0].1;
    let mut exact = Cell::build(&params_for(256, moduli, 3), 32, 0x4d49_5300);
    let mut approximate = Cell::build(&small(moduli), 32, 0x4d49_5300);
    assert_eq!(exact.crs.inspiring_w_seed, approximate.crs.inspiring_w_seed);

    let approximate_keys = approximate.wire_keys();
    assert_refused_everywhere(&mut exact, &approximate_keys);
    let exact_keys = exact.wire_keys();
    assert_refused_everywhere(&mut approximate, &exact_keys);
}

fn assert_refused_everywhere(server: &mut Cell, foreign: &ClientPackingKeys) {
    let ctx = server.params.ntt_context();
    let (crs, cache) = (&server.crs, &server.cache);
    let store = ServerSessionStore::new();
    let mismatch = |message: String| {
        assert!(message.contains("material mismatch"), "{message}");
    };
    mismatch(refusal(PackingKeySeat::from_wire(
        foreign.clone(),
        cache.pack_params(),
        &ctx,
    )));
    mismatch(refusal(store.register_server_side(
        foreign.clone(),
        cache.pack_params(),
        &ctx,
    )));
    assert!(store.is_empty());
    let source = PackingKeySource::Inline(foreign.clone());
    mismatch(refusal(resolve_packing_seat(crs, cache, &source, None)));

    let row = server.next_query(7);
    let (crs, cache, encoded) = (&server.crs, &server.cache, &server.encoded);
    let query = inline_query(&row, foreign);
    mismatch(refusal(respond_seeded_inspiring(crs, encoded, &query)));
    mismatch(refusal(respond_seeded_inspiring_cached(
        crs, encoded, &query, cache,
    )));
    mismatch(refusal(respond_seeded_inspiring_cached_with_session(
        crs, encoded, &query, cache, None,
    )));
    let batch = batch_from(&source, std::slice::from_ref(&row));
    mismatch(refusal(respond_seeded_batch(
        crs, encoded, &batch, cache, None,
    )));
}

/// Server state assembled for one gadget is refused under the other: a cache checked
/// against the wrong CRS, and offline keys expanded for the wrong row count.
#[test]
fn server_state_for_the_other_gadget_is_refused() {
    let moduli = &modulus_sets()[0].1;
    let exact = Cell::build(&params_for(256, moduli, 3), 32, 0x5356_5200);
    let mut approximate = Cell::build(&small(moduli), 32, 0x5356_5200);

    exact
        .cache
        .validate_for(&exact.crs, &exact.encoded)
        .unwrap();
    approximate
        .cache
        .validate_for(&approximate.crs, &approximate.encoded)
        .unwrap();
    for (cache, crs, encoded) in [
        (&exact.cache, &approximate.crs, &approximate.encoded),
        (&approximate.cache, &exact.crs, &exact.encoded),
    ] {
        let message = refusal(cache.validate_for(crs, encoded));
        assert!(message.contains("do not match CRS"), "{message}");
    }

    // Pack params of one gadget with the mask rows of the other pass no validation
    // on `from_parts`; the responder still refuses to fold under them.
    let row = approximate.next_query(9);
    let query = inline_query(&row, &approximate.wire_keys());
    let mixed = ServerInspiringCache::from_parts(
        approximate.cache.pack_params().clone(),
        exact.cache.offline_keys().clone(),
    );
    let message = refusal(respond_seeded_inspiring_cached(
        &approximate.crs,
        &approximate.encoded,
        &query,
        &mixed,
    ));
    assert!(message.contains("mask rows"), "{message}");
    let message = refusal(mixed.validate_for(&approximate.crs, &approximate.encoded));
    assert!(
        message.contains("offline packing keys do not match"),
        "{message}"
    );

    let exact_query = inline_query(&row, &exact.wire_keys());
    let mixed = ServerInspiringCache::from_parts(
        exact.cache.pack_params().clone(),
        approximate.cache.offline_keys().clone(),
    );
    let message = refusal(respond_seeded_inspiring_cached(
        &exact.crs,
        &exact.encoded,
        &exact_query,
        &mixed,
    ));
    assert!(message.contains("mask rows"), "{message}");
}

/// The table-reading kernel equals the rotation formula it stands for when the key
/// has two rows, and refuses a key of any other length.
#[test]
fn the_gather_kernel_equals_the_rotation_formula_under_the_approximate_gadget() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x4b45_524e);
    for (name, moduli) in modulus_sets() {
        let params = small(&moduli);
        let ctx = params.ntt_context();
        for gamma in [2usize, 16, 128] {
            let pack_params = PackParams::try_new(&params, gamma).unwrap();
            let offline_keys = OfflinePackingKeys::generate(&pack_params, [3; 32]);
            let mut sampler = GaussianSampler::with_seed(params.sigma, 17);
            let secret_key = RlweSecretKey::generate(&params, &mut sampler);
            let keys =
                ClientPackingKeys::generate(&secret_key, &params, gamma, [3; 32], &mut sampler)
                    .unwrap();
            let random = |rng: &mut ChaCha20Rng| {
                let coeffs = (0..256).map(|_| rng.gen_range(0..params.q)).collect();
                Poly::from_coeffs_moduli(coeffs, &moduli)
            };
            let a_polys: Vec<Poly> = (0..gamma).map(|_| random(&mut rng)).collect();
            let b_poly = random(&mut rng);
            let precomp = packing_offline(&pack_params, &offline_keys, &a_polys, &ctx);
            assert!(precomp.bold_t.iter().all(|digits| digits.len() == 2));

            let by_formula = packing_online(
                &precomp,
                &generate_rotations(&pack_params, &keys.y_body),
                &b_poly,
                &ctx,
            );
            let y_body_ntt: Vec<Poly> = keys.y_body.iter().map(|y| y.to_ntt_new(&ctx)).collect();
            let gathered =
                packing_online_gather(&precomp, &pack_params, &y_body_ntt, &b_poly, &ctx).unwrap();
            assert!(
                gathered.a == by_formula.a && gathered.b == by_formula.b,
                "{name} gamma {gamma}"
            );

            let mut three_rows = y_body_ntt.clone();
            three_rows.push(y_body_ntt[0].clone());
            for wrong in [&three_rows[..], &y_body_ntt[..1], &[]] {
                let refused = packing_online_gather(&precomp, &pack_params, wrong, &b_poly, &ctx)
                    .unwrap_err();
                assert!(refused.contains("key body must be 2"), "{refused}");
            }
        }
    }
}
