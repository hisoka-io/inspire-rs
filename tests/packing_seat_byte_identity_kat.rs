//! A packing-key seat keeps only `y_body` in NTT form, and every InspiRING responder
//! packs through it. These tests hold that to the bytes the rotation-cache
//! implementation served.
//!
//! `tests/fixtures/packing_seat_response_kat.txt` was recorded from that
//! implementation, where an inline key, a seat with cached rotations and a seat
//! without them were first asserted equal to each other. Every responder here must
//! reproduce each recorded response, and the kernel is compared with the plain
//! `generate_rotations` + `packing_online` formula on random and extreme residues.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test-target harness; an abort here is the failure report"
)]

mod packing_key_fixture;

use packing_key_fixture::{fnv1a, inline_query, modulus_sets, params_for, response_bytes, Cell};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use raven_inspire::inspiring::{
    generate_rotations, packing_offline, packing_online, packing_online_gather, ClientPackingKeys,
    PackParams, PrecompInsPIR,
};
use raven_inspire::math::{GaussianSampler, NttContext, Poly};
use raven_inspire::params::InspireParams;
use raven_inspire::pir::{
    respond_seeded_inspiring, respond_seeded_inspiring_cached,
    respond_seeded_inspiring_cached_with_session, PackingKeySeat, ServerResponse,
    ServerSessionStore,
};
use raven_inspire::rlwe::{RlweCiphertext, RlweSecretKey};

const RECORDED: &str = include_str!("fixtures/packing_seat_response_kat.txt");

/// Whether the replayed shards fold offline-packing step 1 over their columns first.
#[derive(Clone, Copy, Debug)]
enum Packing {
    PerQuery,
    FromShardStores,
}

/// Replays one recorded cell through every responder and returns the lines it serves,
/// in the fixture's format.
fn served_lines(
    tag: &str,
    params: &InspireParams,
    entry_size: usize,
    seed: u64,
    indices: &[u64],
    packing: Packing,
) -> Vec<String> {
    let mut cell = Cell::build(params, entry_size, seed);
    if matches!(packing, Packing::FromShardStores) {
        for shard in &mut cell.encoded.shards {
            shard
                .precompute_packing(cell.cache.pack_params(), &params.ntt_context())
                .expect("precompute");
        }
    }
    let stores = cell
        .encoded
        .shards
        .iter()
        .filter(|shard| shard.packing_store().is_some())
        .count();
    assert_eq!(
        stores,
        match packing {
            Packing::PerQuery => 0,
            Packing::FromShardStores => cell.encoded.shards.len(),
        }
    );
    let mut lines = Vec::new();

    let keys = cell.wire_keys();
    let key_bytes = bincode::serialize(&keys).unwrap();
    lines.push(format!(
        "SESSIONKEY {tag} entry={entry_size} seed={seed:#x} len={} fnv={:#018x}",
        key_bytes.len(),
        fnv1a(&key_bytes)
    ));

    let store = ServerSessionStore::new();
    let handle = store
        .register_server_side(
            keys.clone(),
            cell.cache.pack_params(),
            &params.ntt_context(),
        )
        .expect("register");

    for (position, &index) in indices.iter().enumerate() {
        let query = cell.next_query(index);
        let inline = inline_query(&query, &keys);
        let mut by_handle = inline.clone();
        by_handle.inspiring_packing_keys = None;
        by_handle.session_handle = Some(handle);

        let seat_response = respond_seeded_inspiring_cached_with_session(
            &cell.crs,
            &cell.encoded,
            &by_handle,
            &cell.cache,
            Some(&store),
        )
        .expect("seat respond");
        let served = response_bytes(&seat_response);
        let inline_served = response_bytes(
            &respond_seeded_inspiring_cached_with_session(
                &cell.crs,
                &cell.encoded,
                &inline,
                &cell.cache,
                None,
            )
            .expect("inline respond"),
        );
        assert_eq!(served, inline_served, "{tag} index {index}: seat vs inline");
        if position == 0 {
            let cached =
                respond_seeded_inspiring_cached(&cell.crs, &cell.encoded, &inline, &cell.cache)
                    .expect("cached respond");
            assert_eq!(served, response_bytes(&cached), "{tag}: cached responder");
            let uncached =
                respond_seeded_inspiring(&cell.crs, &cell.encoded, &inline).expect("respond");
            assert_eq!(
                served,
                response_bytes(&uncached),
                "{tag}: uncached responder"
            );
        }
        if params.ring_dim == 2048 {
            cell.assert_decodes(index, &seat_response);
        }
        lines.push(format!(
            "RESP {tag} entry={entry_size} seed={seed:#x} index={index} len={} fnv={:#018x}",
            served.len(),
            fnv1a(&served)
        ));
    }
    lines
}

fn assert_recorded(lines: &[String]) {
    assert!(!lines.is_empty());
    for line in lines {
        assert!(
            RECORDED.lines().any(|recorded| recorded == line),
            "served bytes moved; no recorded line equals:\n{line}"
        );
    }
}

fn recorded_count(tag: &str) -> usize {
    RECORDED
        .lines()
        .filter(|line| line.split(' ').nth(1) == Some(tag))
        .count()
}

#[test]
fn production_cells_serve_the_recorded_bytes() {
    let params = InspireParams::secure_128_d2048();
    let d = params.ring_dim as u64;
    for packing in [Packing::PerQuery, Packing::FromShardStores] {
        let mut served = 0;
        let cells: [(usize, &[u64]); 4] = [
            (32, &[0x5245_5300, 0x5245_5301]),
            (256, &[0x5245_5300, 0x5245_5301]),
            (512, &[0x5245_5300, 0x5245_5301]),
            (1024, &[0x5245_5300]),
        ];
        for (entry_size, seeds) in cells {
            for &seed in seeds {
                let lines = served_lines(
                    "d2048",
                    &params,
                    entry_size,
                    seed,
                    &[0, 1, d - 1, d, 2 * d + 17, 3 * d - 1],
                    packing,
                );
                assert_recorded(&lines);
                served += lines.len();
            }
        }
        assert_eq!(
            served,
            recorded_count("d2048"),
            "{packing:?}: every recorded line is replayed"
        );
    }
}

#[test]
fn every_modulus_set_serves_the_recorded_bytes() {
    for (name, moduli) in modulus_sets() {
        let params = params_for(256, &moduli, 3);
        let tag = format!("d256-{name}-l3");
        for packing in [Packing::PerQuery, Packing::FromShardStores] {
            let mut served = 0;
            for entry_size in [2usize, 4, 32, 64, 256] {
                let lines = served_lines(
                    &tag,
                    &params,
                    entry_size,
                    0x534d_4100 + entry_size as u64,
                    &[0, 255, 256, 511, 700],
                    packing,
                );
                assert_recorded(&lines);
                served += lines.len();
            }
            assert_eq!(
                served,
                recorded_count(&tag),
                "{tag} {packing:?}: every recorded line is replayed"
            );
        }
    }
}

/// The comparison above can fail: a response under another client's seat, or to
/// another row, is a different byte string.
#[test]
fn a_foreign_seat_or_row_serves_different_bytes() {
    let params = params_for(256, &modulus_sets()[0].1, 3);
    let mut cell = Cell::build(&params, 32, 0x4e45_4700);
    let other = Cell::build(&params, 32, 0x4e45_4701);
    let ctx = params.ntt_context();
    let store = ServerSessionStore::new();
    let own = store
        .register_server_side(cell.wire_keys(), cell.cache.pack_params(), &ctx)
        .unwrap();
    let foreign = store
        .register_server_side(other.wire_keys(), cell.cache.pack_params(), &ctx)
        .unwrap();

    let query = cell.next_query(7);
    let neighbour = cell.next_query(8);
    let respond = |query, handle| {
        let mut by_handle = inline_query(query, &cell.wire_keys());
        by_handle.inspiring_packing_keys = None;
        by_handle.session_handle = Some(handle);
        response_bytes(
            &respond_seeded_inspiring_cached_with_session(
                &cell.crs,
                &cell.encoded,
                &by_handle,
                &cell.cache,
                Some(&store),
            )
            .unwrap(),
        )
    };
    let served = respond(&query, own);
    assert_eq!(served, respond(&query, own), "respond is deterministic");
    assert_ne!(
        served,
        respond(&query, foreign),
        "another seat must change the bytes"
    );
    assert_ne!(
        served,
        respond(&neighbour, own),
        "another row must change the bytes"
    );
}

/// The responder written out from its public parts: fold the shard, run the offline
/// phase, rotate `y_body` in coefficient form and pack.
fn rotation_reference(
    cell: &Cell,
    shard_id: u32,
    rgsw: &raven_inspire::rgsw::RgswCiphertext,
    keys: &ClientPackingKeys,
) -> RlweCiphertext {
    let params = &cell.params;
    let ctx = params.ntt_context();
    let shard = &cell.encoded.shards[shard_id as usize];
    let columns: Vec<RlweCiphertext> = shard
        .polynomials
        .iter()
        .map(|column| rgsw.rows[0].poly_mul(column, &ctx))
        .collect();
    let a_ct_tilde: Vec<Poly> = columns.iter().map(|ct| ct.a.clone()).collect();
    let mut b_coeffs = vec![0u64; params.ring_dim];
    for (slot, ct) in columns.iter().enumerate() {
        b_coeffs[slot] = ct.sample_extract_coeff0().b;
    }
    let b_poly = Poly::from_coeffs_moduli(b_coeffs, params.moduli());
    let pack_params = cell.cache.pack_params();
    let precomp = packing_offline(pack_params, cell.cache.offline_keys(), &a_ct_tilde, &ctx);
    packing_online(
        &precomp,
        &generate_rotations(pack_params, &keys.y_body),
        &b_poly,
        &ctx,
    )
}

/// Fresh OS-seeded queries on every run, at every legal width of a small ring.
#[test]
fn seat_responses_match_the_rotation_reference_on_fresh_queries() {
    for (name, moduli) in modulus_sets() {
        let params = params_for(256, &moduli, 3);
        for entry_size in [2usize, 8, 64, 256] {
            let cell = Cell::build(&params, entry_size, 0x4652_4500 + entry_size as u64);
            let keys = cell.wire_keys();
            let store = ServerSessionStore::new();
            let mut session = cell.session.clone();
            session.register_with(&store).unwrap().expect("handle");
            let mut sampler = GaussianSampler::new(params.sigma);
            let mut rng = ChaCha20Rng::from_entropy();
            for _ in 0..6 {
                let index = rng.gen_range(0..3 * params.ring_dim as u64);
                let (_, query) = session
                    .query_seeded(index, &cell.encoded.config, &mut sampler)
                    .unwrap();
                let served = respond_seeded_inspiring_cached_with_session(
                    &cell.crs,
                    &cell.encoded,
                    &query,
                    &cell.cache,
                    Some(&store),
                )
                .unwrap();
                let reference = rotation_reference(
                    &cell,
                    query.shard_id,
                    &query.rgsw_ciphertext.expand(),
                    &keys,
                );
                assert_eq!(
                    served.ciphertext.a.coeffs(),
                    reference.a.coeffs(),
                    "{name} entry {entry_size} index {index}: a"
                );
                assert_eq!(
                    served.ciphertext.b.coeffs(),
                    reference.b.coeffs(),
                    "{name} entry {entry_size} index {index}: b"
                );
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Residues {
    Random,
    Zero,
    Max,
    OneHot,
}

/// A coefficient-form polynomial whose NTT form is exactly the requested residues,
/// so the kernel's multiply sees them.
fn poly_with_ntt_residues(
    pattern: Residues,
    n: usize,
    moduli: &[u64],
    ctx: &NttContext,
    rng: &mut ChaCha20Rng,
) -> Poly {
    let mut residues = Vec::with_capacity(n * moduli.len());
    for &modulus in moduli {
        let hot = rng.gen_range(0..n);
        for slot in 0..n {
            residues.push(match pattern {
                Residues::Random => rng.gen_range(0..modulus),
                Residues::Zero => 0,
                Residues::Max => modulus - 1,
                Residues::OneHot => {
                    if slot == hot {
                        modulus - 1
                    } else {
                        0
                    }
                }
            });
        }
    }
    let mut poly = Poly::from_crt_coeffs(residues, moduli);
    poly.force_ntt_domain();
    poly.from_ntt(ctx);
    poly
}

struct KernelCase {
    pack_params: PackParams,
    ctx: NttContext,
    precomp: PrecompInsPIR,
    y_body: Vec<Poly>,
    b_poly: Poly,
}

impl KernelCase {
    fn build(
        params: &InspireParams,
        gamma: usize,
        key: Residues,
        offline: Residues,
        rng: &mut ChaCha20Rng,
    ) -> Self {
        let n = params.ring_dim;
        let moduli = params.moduli();
        let ctx = params.ntt_context();
        let pack_params = PackParams::try_new(params, gamma).expect("legal width");
        let digits = params.packing_gadget_len;
        let y_body: Vec<Poly> = (0..digits)
            .map(|_| poly_with_ntt_residues(key, n, moduli, &ctx, rng))
            .collect();
        let bold_t: Vec<Vec<Poly>> = (0..gamma - 1)
            .map(|_| {
                (0..digits)
                    .map(|_| poly_with_ntt_residues(offline, n, moduli, &ctx, rng))
                    .collect()
            })
            .collect();
        let bold_t_ntt = bold_t
            .iter()
            .map(|row| row.iter().map(|t| t.to_ntt_new(&ctx)).collect())
            .collect();
        let precomp = PrecompInsPIR {
            a_hat: Poly::random_with_rng_moduli(n, moduli, rng),
            bold_t,
            bold_t_ntt,
            bold_t_bar: vec![],
            bold_t_hat: vec![],
            num_to_pack: gamma,
            ring_dim: n,
            q: params.q,
        };
        let b_poly = Poly::random_with_rng_moduli(n, moduli, rng);
        Self {
            pack_params,
            ctx,
            precomp,
            y_body,
            b_poly,
        }
    }

    fn y_body_ntt(&self) -> Vec<Poly> {
        self.y_body
            .iter()
            .map(|y| y.to_ntt_new(&self.ctx))
            .collect()
    }

    fn gather(&self) -> RlweCiphertext {
        packing_online_gather(
            &self.precomp,
            &self.pack_params,
            &self.y_body_ntt(),
            &self.b_poly,
            &self.ctx,
        )
        .expect("well-formed case")
    }

    fn reference(&self, rotations: &[Vec<Poly>]) -> RlweCiphertext {
        packing_online(&self.precomp, rotations, &self.b_poly, &self.ctx)
    }
}

fn same_ciphertext(left: &RlweCiphertext, right: &RlweCiphertext) -> bool {
    left.a.coeffs() == right.a.coeffs()
        && left.b.coeffs() == right.b.coeffs()
        && left.b.is_ntt() == right.b.is_ntt()
}

/// One, two, three and four key digits over one and two CRT limbs, on random, zero,
/// maximal and one-hot residues. Each case also carries its own negative control:
/// reading the rotations one generator step off must change the output.
#[test]
fn gather_kernel_matches_the_rotation_formula_on_random_and_extreme_residues() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x4b45_524e);
    let patterns = [
        Residues::Random,
        Residues::Zero,
        Residues::Max,
        Residues::OneHot,
    ];
    let mut compared = 0;
    let mut controls = 0;
    for (name, moduli) in modulus_sets() {
        for digits in [1usize, 2, 3, 4] {
            let params = params_for(64, &moduli, digits);
            for gamma in [1usize, 2, 8, 32] {
                for key in patterns {
                    for offline in patterns {
                        let case = KernelCase::build(&params, gamma, key, offline, &mut rng);
                        let rotations = generate_rotations(&case.pack_params, &case.y_body);
                        let gathered = case.gather();
                        assert!(
                            same_ciphertext(&gathered, &case.reference(&rotations)),
                            "{name} digits {digits} gamma {gamma} key {key:?} offline {offline:?}"
                        );
                        compared += 1;

                        if gamma > 2
                            && matches!(key, Residues::Random)
                            && matches!(offline, Residues::Random)
                        {
                            let mut shifted = rotations.clone();
                            shifted.rotate_left(1);
                            assert!(
                                !same_ciphertext(&gathered, &case.reference(&shifted)),
                                "{name} digits {digits} gamma {gamma}: a rotation read one \
                                 step off went unnoticed"
                            );
                            controls += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(compared, 4 * 4 * 4 * 16);
    assert_eq!(controls, 4 * 4 * 2);
}

#[test]
fn gather_kernel_refuses_shapes_that_would_drop_terms() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5245_4655);
    let params = params_for(64, &modulus_sets()[0].1, 3);
    let case = KernelCase::build(&params, 8, Residues::Random, Residues::Random, &mut rng);
    let y_body_ntt = case.y_body_ntt();
    let refused = |precomp: &PrecompInsPIR, pack: &PackParams, body: &[Poly], b: &Poly| {
        packing_online_gather(precomp, pack, body, b, &case.ctx)
            .expect_err("a malformed input must be refused, not packed")
    };

    let short_key = refused(
        &case.precomp,
        &case.pack_params,
        &y_body_ntt[..2],
        &case.b_poly,
    );
    assert!(short_key.contains("key body must be 3"), "{short_key}");

    let coefficient_key = refused(&case.precomp, &case.pack_params, &case.y_body, &case.b_poly);
    assert!(coefficient_key.contains("NTT-form"), "{coefficient_key}");

    let mut no_ntt_rows = case.precomp.clone();
    no_ntt_rows.bold_t_ntt.clear();
    let dropped = refused(&no_ntt_rows, &case.pack_params, &y_body_ntt, &case.b_poly);
    assert!(dropped.contains("0 NTT-form rotation rows"), "{dropped}");

    let mut short_row = case.precomp.clone();
    short_row.bold_t_ntt[3].pop();
    let row = refused(&short_row, &case.pack_params, &y_body_ntt, &case.b_poly);
    assert!(row.contains("offline rotation row 3"), "{row}");

    let other_width = PackParams::try_new(&params, 16).unwrap();
    let width = refused(&case.precomp, &other_width, &y_body_ntt, &case.b_poly);
    assert!(width.contains("pack params expect gamma 16"), "{width}");

    let mut short_table = case.pack_params.clone();
    let rotated = short_table.gen_pows[1];
    short_table.automorph_tables[(rotated - 1) / 2].pop();
    let table = refused(&case.precomp, &short_table, &y_body_ntt, &case.b_poly);
    assert!(table.contains("no full automorphism table"), "{table}");

    let other_ring = params_for(128, &modulus_sets()[0].1, 3).ntt_context();
    let context = packing_online_gather(
        &case.precomp,
        &case.pack_params,
        &y_body_ntt,
        &case.b_poly,
        &other_ring,
    )
    .expect_err("a context for another ring must be refused");
    assert!(
        context.contains("NTT context is for ring_dim 128"),
        "{context}"
    );

    let ntt_b = case.b_poly.to_ntt_new(&case.ctx);
    let b = refused(&case.precomp, &case.pack_params, &y_body_ntt, &ntt_b);
    assert!(b.contains("coefficient-form"), "{b}");
}

fn production_seat(entry_size: usize) -> (PackingKeySeat, PackParams, InspireParams) {
    let params = InspireParams::secure_128_d2048();
    let gamma = raven_inspire::num_columns(entry_size);
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0x5345_4154);
    let secret_key = RlweSecretKey::generate(&params, &mut sampler);
    let keys = ClientPackingKeys::generate(&secret_key, &params, gamma, [9; 32], &mut sampler)
        .expect("legal width");
    let pack_params = PackParams::try_new(&params, gamma).expect("pack params");
    let seat = PackingKeySeat::from_wire(keys, &pack_params, &params.ntt_context()).expect("seat");
    (seat, pack_params, params)
}

/// Three 2,048-coefficient polynomials at every record width: the seat does not grow
/// with gamma.
#[test]
fn a_production_seat_is_49152_bytes_at_every_width() {
    for entry_size in [32usize, 256, 512, 2048] {
        let (seat, pack_params, _) = production_seat(entry_size);
        assert_eq!(seat.num_to_pack(), pack_params.num_to_pack);
        assert_eq!(seat.payload_bytes(), 49_152, "entry size {entry_size}");
        assert_eq!(
            seat.resident_bytes(),
            49_152 + std::mem::size_of::<PackingKeySeat>() + 3 * (std::mem::size_of::<Poly>() + 8),
            "entry size {entry_size}"
        );
        assert!(seat.resident_bytes() < 49_152 + 512);
    }
}

#[test]
fn a_stored_seat_is_the_registered_one() {
    let (seat, _, _) = production_seat(512);
    let store = ServerSessionStore::new();
    let handle = store.register(seat).expect("register");
    let stored = store.get(handle).expect("lookup").expect("registered");
    assert_eq!(stored.payload_bytes(), 49_152);
    assert_eq!(stored.num_to_pack(), 256);
}

/// A handle registered against one table shape must not pack under another: the
/// generator, and so every rotation, differs.
#[test]
fn a_seat_from_another_width_is_refused_at_respond() {
    let params = params_for(256, &modulus_sets()[0].1, 3);
    let mut wide = Cell::build(&params, 64, 0x5749_4400);
    let narrow = Cell::build(&params, 32, 0x5749_4401);
    let store = ServerSessionStore::new();
    let narrow_handle = store
        .register_server_side(
            narrow.wire_keys(),
            narrow.cache.pack_params(),
            &params.ntt_context(),
        )
        .expect("register at the narrow width");

    let query = wide.next_query(5);
    let mut by_handle = inline_query(&query, &wide.wire_keys());
    by_handle.inspiring_packing_keys = None;
    by_handle.session_handle = Some(narrow_handle);
    let refusal: raven_inspire::pir::Result<ServerResponse> =
        respond_seeded_inspiring_cached_with_session(
            &wide.crs,
            &wide.encoded,
            &by_handle,
            &wide.cache,
            Some(&store),
        );
    let message = refusal
        .expect_err("a foreign-width seat must be refused")
        .to_string();
    assert!(message.contains("width mismatch"), "{message}");
    assert!(
        message.contains("16") && message.contains("32"),
        "{message}"
    );
}

/// A cache built for one table shape must not pack another table's shard.
#[test]
fn a_cache_from_another_table_shape_is_refused() {
    let params = params_for(256, &modulus_sets()[0].1, 3);
    let mut wide = Cell::build(&params, 64, 0x4341_4300);
    let narrow = Cell::build(&params, 32, 0x4341_4301);
    let query = wide.next_query(3);
    // Keys at the narrow width pass the seat check against the narrow cache, so only
    // the shard-width guard stands between this call and a wrong plaintext.
    let inline = inline_query(&query, &narrow.wire_keys());
    let message = respond_seeded_inspiring_cached(&wide.crs, &wide.encoded, &inline, &narrow.cache)
        .expect_err("a foreign cache must be refused")
        .to_string();
    assert!(message.contains("another table shape"), "{message}");
}
