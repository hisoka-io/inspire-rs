//! The 32-bit special-modulus path against the generic 64-bit reference: the same key,
//! query and response bytes from the same secret, errors, seeds and cells, at small rings
//! and at the served size, on every kernel this host runs. Then decode, wire pins and the
//! refusals.

#![allow(clippy::unwrap_used)]

mod simd_capability;
mod special_modulus_reference;

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use raven_inspire::math::GaussianSampler;
use raven_inspire::special_modulus::{
    Client, Kernel, PackingKey, Params, Query, Response, Server, ShardTable,
};
use special_modulus_reference as reference;

#[derive(Clone, Copy, Debug)]
enum Data {
    /// Every nibble -8: the largest fold.
    Worst,
    Random,
}

fn small(d: usize, gamma: usize) -> Params {
    Params {
        ring_dim: d,
        gamma,
        cell_bytes: gamma / 2,
        record_bytes: gamma / 2,
        ..Params::ring2048_cell256()
    }
}

/// Cells for `rows` rows; bytes past the record stay zero.
fn cells(params: &Params, rows: usize, data: Data, rng: &mut ChaCha20Rng) -> Vec<u8> {
    let mut out = vec![0u8; rows * params.cell_bytes];
    for cell in out.chunks_exact_mut(params.cell_bytes) {
        match data {
            Data::Worst => cell[..params.record_bytes].fill(0x88),
            Data::Random => rng.fill_bytes(&mut cell[..params.record_bytes]),
        }
    }
    out
}

fn signed_nibbles(cells: &[u8], params: &Params) -> Vec<i8> {
    let mut vals = vec![0i8; params.ring_dim * params.gamma];
    for (i, &b) in cells.iter().enumerate() {
        vals[2 * i] = ((b << 4) as i8) >> 4;
        vals[2 * i + 1] = (b as i8) >> 4;
    }
    vals
}

struct Case {
    params: Params,
    secret: Vec<i64>,
    key_error: Vec<i64>,
    cells: Vec<u8>,
    rows: usize,
    row: usize,
    seed: [u8; 32],
    query_error: Vec<i64>,
}

fn case(params: Params, data: Data, row: Option<usize>, seed: u64) -> Case {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let d = params.ring_dim;
    let mut noise = GaussianSampler::with_seed(params.sigma, seed ^ 0x5eed);
    let rows = if seed.is_multiple_of(3) {
        d
    } else {
        rng.gen_range(1..=d)
    };
    let cells = cells(&params, rows, data, &mut rng);
    let mut q_seed = [0u8; 32];
    rng.fill_bytes(&mut q_seed);
    Case {
        secret: noise.sample_vec(d),
        key_error: noise.sample_vec(d),
        query_error: noise.sample_vec(d),
        row: row.unwrap_or_else(|| rng.gen_range(0..d)),
        rows,
        cells,
        seed: q_seed,
        params,
    }
}

/// Key, query and response bytes on the 32-bit path.
fn fast(case: &Case, server: &Server, table: &ShardTable) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let p = &case.params;
    let client = Client::from_parts(p, &case.secret, &case.key_error, [0; 32]).unwrap();
    let key = client.packing_key();
    let query = client
        .query_from_parts(case.row, case.seed, &case.query_error)
        .unwrap();
    let seat = server.seat(key).unwrap();
    let response = server.respond(table, &seat, &query).unwrap();
    (
        key.to_bytes(p).unwrap(),
        query.to_bytes(p).unwrap(),
        response.to_bytes(p).unwrap(),
    )
}

fn pack(values: &[u64], bits: usize) -> Vec<u8> {
    let mut out = vec![0u8; (values.len() * bits).div_ceil(8)];
    for (i, &v) in values.iter().enumerate() {
        for b in 0..bits {
            if v >> b & 1 == 1 {
                let at = i * bits + b;
                out[at / 8] |= 1 << (at % 8);
            }
        }
    }
    out
}

fn bits(m: u64) -> usize {
    (64 - (m - 1).leading_zeros()) as usize
}

/// The same three messages from the generic reference, packed independently.
fn generic(case: &Case) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let p = &case.params;
    let (q, sp) = (p.q, p.special_prime);
    let ctx = reference::PackCtx::new(p.ring_dim, p.gamma, q, sp);
    let sk = reference::Secret::from_coeffs(&ctx, case.secret.clone());
    let key = reference::keygen(&ctx, &sk, p.key_mask_seed, &case.key_error);
    let body = reference::query(&ctx, &sk, case.row, case.seed, &case.query_error);
    let shard = reference::Shard {
        rows: case.rows,
        cols: p.gamma,
        vals: signed_nibbles(&case.cells, p),
    };
    let (mask, b) = reference::respond(&ctx, &shard, p.key_mask_seed, &key, case.seed, &body);
    let key_bytes = [pack(&key.q, bits(q)), pack(&key.p, bits(sp))].concat();
    let query_bytes = [case.seed.to_vec(), pack(&body, bits(q))].concat();
    let response_bytes = [
        pack(&mask, bits(q)),
        pack(&b[..p.body_coefficients()], bits(q)),
    ]
    .concat();
    (key_bytes, query_bytes, response_bytes)
}

fn kernels() -> Vec<Kernel> {
    [Kernel::Portable, Kernel::Avx512]
        .into_iter()
        .filter(|k| k.available())
        .collect()
}

fn check(case: &Case, tag: &str) {
    let want = generic(case);
    for kernel in kernels() {
        let server = Server::with_kernel(&case.params, kernel).unwrap();
        let table = server.table(&case.cells).unwrap();
        let got = fast(case, &server, &table);
        assert_eq!(got.0, want.0, "key bytes: {tag} {kernel:?}");
        assert_eq!(got.1, want.1, "query bytes: {tag} {kernel:?}");
        assert_eq!(got.2, want.2, "response bytes: {tag} {kernel:?}");
    }
}

#[test]
fn small_rings_match_the_generic_reference_byte_for_byte() {
    let mut seed = 1000;
    for (d, gamma) in [(256usize, 32usize), (512, 64), (256, 128), (64, 8)] {
        for data in [Data::Worst, Data::Random] {
            for row in [Some(0), Some(d - 1), None, None] {
                seed += 1;
                let case = case(small(d, gamma), data, row, seed);
                check(
                    &case,
                    &format!("d {d} gamma {gamma} {data:?} row {} seed {seed}", case.row),
                );
            }
        }
    }
}

#[test]
fn served_set_matches_the_generic_reference_byte_for_byte() {
    let params = Params::ring2048_cell256();
    for (i, (data, row)) in [
        (Data::Random, Some(0)),
        (Data::Worst, Some(2047)),
        (Data::Random, None),
    ]
    .into_iter()
    .enumerate()
    {
        let case = case(params.clone(), data, row, 77 + i as u64);
        check(&case, &format!("served {data:?} row {}", case.row));
    }
}

#[test]
fn the_reference_comparison_can_fail() {
    // One unit more key error in one coefficient changes the key bytes; one more unit of
    // query error changes the response.
    let case = case(small(256, 32), Data::Random, Some(5), 9);
    let base = generic(&case);
    let server = Server::with_kernel(&case.params, Kernel::Portable).unwrap();
    let table = server.table(&case.cells).unwrap();
    assert_eq!(fast(&case, &server, &table), base);
    let mut off_key = clone_case(&case);
    off_key.key_error[3] += 1;
    assert_ne!(fast(&off_key, &server, &table).0, base.0);
    let mut off_query = clone_case(&case);
    off_query.query_error[0] += 1;
    assert_ne!(fast(&off_query, &server, &table).2, base.2);
}

fn clone_case(c: &Case) -> Case {
    Case {
        params: c.params.clone(),
        secret: c.secret.clone(),
        key_error: c.key_error.clone(),
        cells: c.cells.clone(),
        rows: c.rows,
        row: c.row,
        seed: c.seed,
        query_error: c.query_error.clone(),
    }
}

#[test]
fn kernels_give_the_same_table_and_response() {
    let test = "kernels_give_the_same_table_and_response";
    if !simd_capability::require_cargo_feature(
        test,
        "simd-packing-offline",
        cfg!(feature = "simd-packing-offline"),
    ) {
        return;
    }
    if !Kernel::Avx512.available() {
        assert!(
            std::env::var(simd_capability::REQUIRE_ENV).as_deref() != Ok("1"),
            "{test}: host lacks AVX-512 F/BW/DQ/VL and {}=1 demands the kernel run",
            simd_capability::REQUIRE_ENV
        );
        eprintln!("LOUD SKIP: {test} did NOT run - host lacks AVX-512 F/BW/DQ/VL");
        return;
    }
    let params = Params::ring2048_cell256();
    let mut rng = ChaCha20Rng::seed_from_u64(4);
    let cells = cells(&params, 2048, Data::Random, &mut rng);
    let portable = Server::with_kernel(&params, Kernel::Portable).unwrap();
    let wide = Server::with_kernel(&params, Kernel::Avx512).unwrap();
    let (t1, t2) = (portable.table(&cells).unwrap(), wide.table(&cells).unwrap());
    let mut client = Client::from_seed(&params, [3; 32]).unwrap();
    let seat = portable.seat(client.packing_key()).unwrap();
    for row in [0, 1, 1000, 2047] {
        let query = client.query(row).unwrap();
        let a = portable.respond(&t1, &seat, &query).unwrap();
        let b = wide.respond(&t2, &seat, &query).unwrap();
        let c = wide.respond(&t1, &seat, &query).unwrap();
        assert_eq!(a, b, "row {row}");
        assert_eq!(
            a, c,
            "row {row}: a portable table served by the wide kernel"
        );
        assert_eq!(
            client.decode(&a).unwrap(),
            cells[row * 256..row * 256 + 246]
        );
    }
}

#[test]
fn served_records_decode_byte_exact() {
    let params = Params::ring2048_cell256();
    let server = Server::new(&params).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(11);
    // A short last shard: rows past the cells read as zero records.
    let rows = 1500;
    let cells = cells(&params, rows, Data::Random, &mut rng);
    let table = server.table(&cells).unwrap();
    assert_eq!(table.rows(), rows);
    let mut client = Client::new(&params).unwrap();
    let key =
        PackingKey::from_bytes(&params, &client.packing_key().to_bytes(&params).unwrap()).unwrap();
    let seat = server.seat(&key).unwrap();
    for row in [0, 1, 777, rows - 1, rows, 2047] {
        let query = Query::from_bytes(
            &params,
            &client.query(row).unwrap().to_bytes(&params).unwrap(),
        )
        .unwrap();
        let wire = server
            .respond(&table, &seat, &query)
            .unwrap()
            .to_bytes(&params)
            .unwrap();
        let record = client
            .decode(&Response::from_bytes(&params, &wire).unwrap())
            .unwrap();
        let want = if row < rows {
            cells[row * 256..row * 256 + 246].to_vec()
        } else {
            vec![0u8; 246]
        };
        assert_eq!(record, want, "row {row}");
    }
}

/// FNV-1a over the bytes: a pin that moves when any byte moves.
fn digest(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

#[test]
fn served_wire_bytes_are_pinned() {
    let params = Params::ring2048_cell256();
    let mut client = Client::from_seed(&params, [42; 32]).unwrap();
    let server = Server::with_kernel(&params, Kernel::Portable).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let cells = cells(&params, 2048, Data::Random, &mut rng);
    let table = server.table(&cells).unwrap();
    let seat = server.seat(client.packing_key()).unwrap();
    let key = client.packing_key().to_bytes(&params).unwrap();
    let query = client.query(1234).unwrap();
    let response = server.respond(&table, &seat, &query).unwrap();
    let query = query.to_bytes(&params).unwrap();
    let response = response.to_bytes(&params).unwrap();
    assert_eq!(
        (key.len(), query.len(), response.len()),
        (15_360, 7_456, 9_208)
    );
    assert_eq!(seat.bytes(), 16_384);
    assert_eq!(table.bytes(), (4 << 20) + (1 << 20));
    let got = [digest(&key), digest(&query), digest(&response)];
    if let Ok(path) = std::env::var("SPECIAL_MODULUS_PIN_OUT") {
        std::fs::write(path, format!("{got:#x?}\n")).unwrap();
    }
    assert_eq!(got, PINNED, "wire digests");
}

const PINNED: [u64; 3] = [
    0xd4b9_8b71_661b_c0aa,
    0xd889_f6a1_34b7_e313,
    0x7a56_7bd7_71a5_1590,
];

#[test]
fn one_client_keeps_one_key_and_fresh_queries() {
    let params = Params::ring2048_cell256();
    let mut client = Client::new(&params).unwrap();
    let first = client.packing_key().to_bytes(&params).unwrap();
    let q1 = client.query(5).unwrap().to_bytes(&params).unwrap();
    let q2 = client.query(5).unwrap().to_bytes(&params).unwrap();
    assert_eq!(client.packing_key().to_bytes(&params).unwrap(), first);
    assert_ne!(q1[..32], q2[..32], "mask seed reused");
    assert_ne!(q1[32..], q2[32..], "query body reused");
    let other = Client::new(&params).unwrap();
    assert_ne!(
        other.packing_key().to_bytes(&params).unwrap(),
        first,
        "secret reused"
    );
}

#[test]
fn a_foreign_key_reads_nothing_and_mismatches_are_refused() {
    let params = Params::ring2048_cell256();
    let server = Server::new(&params).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(8);
    let cells = cells(&params, 2048, Data::Random, &mut rng);
    let table = server.table(&cells).unwrap();
    let mut reader = Client::from_seed(&params, [1; 32]).unwrap();
    let stranger = Client::from_seed(&params, [2; 32]).unwrap();
    let query = reader.query(9).unwrap();
    let wrong = server
        .respond(
            &table,
            &server.seat(stranger.packing_key()).unwrap(),
            &query,
        )
        .unwrap();
    assert_ne!(
        reader.decode(&wrong).unwrap(),
        cells[9 * 256..9 * 256 + 246]
    );

    // Bytes past the record, a ragged shard and an oversized shard are refused.
    let mut tail = cells.clone();
    tail[250] = 1;
    assert!(server.table(&tail).is_err());
    assert!(server.table(&cells[..1000]).is_err());
    assert!(server
        .table(&[cells.clone(), vec![0; 256]].concat())
        .is_err());
    assert!(Query::from_bytes(&params, &[0u8; 7_455]).is_err());

    // A seat or table of another set is refused.
    let other = small(256, 32);
    let small_server = Server::new(&other).unwrap();
    let small_table = small_server.table(&[0u8; 16]).unwrap();
    let seat = server.seat(reader.packing_key()).unwrap();
    assert!(server.respond(&small_table, &seat, &query).is_err());
    assert!(reader.query(2048).is_err());
}

fn unpack(bytes: &[u8], bits: usize, count: usize) -> Vec<u64> {
    (0..count)
        .map(|i| {
            (0..bits).fold(0u64, |v, b| {
                let at = i * bits + b;
                v | u64::from(bytes[at / 8] >> (at % 8) & 1) << b
            })
        })
        .collect()
}

/// The value at `q * p` with residues `a` and `b`, in `[0, q * p)`.
fn compose(a: &[u64], b: &[u64], q: u64, p: u64) -> Vec<u64> {
    let q_inv = raven_inspire::math::mod_inverse(q % p, p);
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let t = u128::from((y + p - x % p) % p) * u128::from(q_inv) % u128::from(p);
            (u128::from(x) + u128::from(q) * t) as u64
        })
        .collect()
}

fn line<T: std::fmt::Display>(out: &mut String, name: &str, v: &[T]) {
    use std::fmt::Write as _;
    write!(out, "{name}").unwrap();
    for x in v {
        write!(out, " {x}").unwrap();
    }
    out.push('\n');
}

/// Dumps of the 32-bit path for the exact-integer reference to check off CI.
#[test]
#[ignore = "run with SPECIAL_MODULUS_DUMP_DIR set: writes dumps for the integer reference"]
fn dump_cases_for_the_integer_reference() {
    let dir = std::env::var("SPECIAL_MODULUS_DUMP_DIR")
        .expect("SPECIAL_MODULUS_DUMP_DIR names the directory the dumps go to");
    let cases = [
        ("d256_g32_random", small(256, 32), Data::Random, Some(0), 3),
        ("d512_g64_worst", small(512, 64), Data::Worst, Some(511), 6),
        ("d256_g128_random", small(256, 128), Data::Random, None, 9),
        ("d64_g8_random", small(64, 8), Data::Random, None, 12),
        (
            "d2048_served_random",
            Params::ring2048_cell256(),
            Data::Random,
            None,
            15,
        ),
    ];
    for (name, params, data, row, seed) in cases {
        let case = case(params, data, row, seed);
        let p = &case.params;
        let (q, sp, d) = (p.q, p.special_prime, p.ring_dim);
        let server = Server::with_kernel(p, Kernel::Portable).unwrap();
        let table = server.table(&case.cells).unwrap();
        let (key, query, response) = fast(&case, &server, &table);
        let kq = unpack(&key, bits(q), d);
        let kp = unpack(&key[(d * bits(q)).div_ceil(8)..], bits(sp), d);
        let ctx = reference::PackCtx::new(d, p.gamma, q, sp);
        let w = reference::expand_key_mask(&ctx, p.key_mask_seed);
        let mut out = String::new();
        line(
            &mut out,
            "meta",
            &[d as u64, p.gamma as u64, q, sp, 16, ctx.g as u64, 1],
        );
        line(&mut out, "s", &case.secret);
        line(&mut out, "kerr", &case.key_error);
        line(&mut out, "w", &compose(&w.q, &w.p, q, sp));
        line(&mut out, "y", &compose(&kq, &kp, q, sp));
        let vals = signed_nibbles(&case.cells, p);
        for j in 0..p.gamma {
            let col: Vec<u64> = (0..d)
                .map(|m| i64::from(vals[m * p.gamma + j]).rem_euclid(q as i64) as u64)
                .collect();
            line(&mut out, &format!("h{j}"), &col);
        }
        line(&mut out, "owner", &vec![0u64; p.gamma]);
        line(
            &mut out,
            "qa0",
            &reference::expand_query_mask(&ctx, case.seed),
        );
        line(&mut out, "qb0", &unpack(&query[32..], bits(q), d));
        line(&mut out, "qe0", &case.query_error);
        line(&mut out, "idx0", &[case.row as u64]);
        line(&mut out, "ahat", &unpack(&response, bits(q), d));
        let at = (d * bits(q)).div_ceil(8);
        line(
            &mut out,
            "bfin",
            &unpack(&response[at..], bits(q), p.body_coefficients()),
        );
        line(
            &mut out,
            "seeds",
            &[case.seed.to_vec(), p.key_mask_seed.to_vec()].concat(),
        );
        std::fs::write(format!("{dir}/{name}.txt"), out).unwrap();
    }
}
