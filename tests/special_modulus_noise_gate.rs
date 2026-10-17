//! Noise gate of the served special-modulus set, at full size, on fresh secrets.
//!
//! At least 1,000 responses over at least 40 secrets drawn from the platform entropy
//! source, every nibble at the worst value (-8). The client's view of each body
//! coefficient's error is pooled; its standard deviation sets the slack against the
//! decoder tail of the shipped failure level (2^10.762 standard deviations). The model
//! and an earlier full-size measurement of this key form both put that slack at 1.65-1.67
//! bits, and the measurement is exact enough (0.01 bit) that the gate holds it to a band.
//! A second pass decodes record-shaped cells and fixed sample records byte for byte.
//!
//! Counts can be raised for a longer run: SPECIAL_MODULUS_NOISE_SECRETS,
//! SPECIAL_MODULUS_NOISE_PER_SECRET.

#![allow(clippy::unwrap_used)]

use std::sync::Mutex;

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use raven_inspire::special_modulus::{Client, Params, Server, ShardTable};

const LOG2_TAIL: f64 = 10.762;
const SLACK_BAND: (f64, f64) = (1.57, 1.77);
/// The worst sample stays this many bits under the half step.
const WORST_MARGIN_BITS: f64 = 9.0;

/// Fixed 246-byte sample records in the served layout.
const SAMPLES: [&str; 4] = [
    concat!(
        "010100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000",
    ),
    concat!(
        "01000000002a0102030405060708090a0b0c0d0e0f100000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000",
        "000000000002000000000000000000000000000000000000000000000000000000000000000300000000000000000000",
        "000000000000000000000000000000000000000000040000000000000000000000000000000000000000000000000000",
        "000000000005",
    ),
    concat!(
        "0101fffffffffffefdfcfbfaf9f8f7f6f5f4f3f2f1f030644e72e131a029b85045b68181585d2833e84879b9709143e1",
        "f593f000000030644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffffff30644e72e131a029b850",
        "45b68181585d2833e84879b9709143e1f593effffffe30644e72e131a029b85045b68181585d2833e84879b9709143e1",
        "f593effffffd30644e72e131a029b85045b68181585d2833e84879b9709143e1f593effffffc30644e72e131a029b850",
        "45b68181585d2833e84879b9709143e1f593effffffb30644e72e131a029b85045b68181585d2833e84879b9709143e1",
        "f593effffffa",
    ),
    concat!(
        "01000000ffff808080808080808080808080808080800000000000000000000000000000000000000000000000000000",
        "000000000001000000000000000000000000000000000000000000000000000000000000000100000000000000000000",
        "000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000",
        "000000000001000000000000000000000000000000000000000000000000000000000000000100000000000000000000",
        "000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000",
        "000000000001",
    ),
];

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A 22-byte header, then seven 32-byte field words with the top three bits clear; the
/// first two words are zero on every other row.
fn record_shaped(rng: &mut ChaCha20Rng, leaf: u32) -> Vec<u8> {
    let mut r = vec![0u8; 246];
    r[0] = 1;
    r[1] = (leaf & 1) as u8;
    r[2..6].copy_from_slice(&leaf.to_be_bytes());
    rng.fill_bytes(&mut r[6..22]);
    for w in 0..7 {
        let at = 22 + 32 * w;
        if w < 2 && r[1] == 0 {
            continue;
        }
        rng.fill_bytes(&mut r[at..at + 32]);
        r[at] &= 0x1f;
    }
    r
}

fn cells_of(records: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * 256);
    for r in records {
        out.extend_from_slice(r);
        out.extend_from_slice(&[0u8; 10]);
    }
    out
}

#[derive(Default)]
struct Tally {
    responses: usize,
    secrets: usize,
    coefficients: u64,
    sum_sq: f64,
    worst: i64,
    wrong_records: usize,
}

fn run(
    server: &Server,
    table: &ShardTable,
    records: &[Vec<u8>],
    secrets: usize,
    per_secret: usize,
    tally: &Mutex<Tally>,
) {
    let params = server.params();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    let next = Mutex::new(0usize);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let k = {
                    let mut n = next.lock().unwrap();
                    if *n == secrets {
                        return;
                    }
                    *n += 1;
                    *n - 1
                };
                let mut client = Client::new(params).unwrap();
                let seat = server.seat(client.packing_key()).unwrap();
                let mut rng = ChaCha20Rng::seed_from_u64(k as u64);
                let mut local = Tally {
                    secrets: 1,
                    ..Tally::default()
                };
                for i in 0..per_secret {
                    // Both ring edges on every secret.
                    let row = match i {
                        0 => 0,
                        1 => records.len() - 1,
                        _ => rng.gen_range(0..records.len()),
                    };
                    let response = server
                        .respond(table, &seat, &client.query(row).unwrap())
                        .unwrap();
                    if client.decode(&response).unwrap() != records[row] {
                        local.wrong_records += 1;
                    }
                    for e in client.body_errors(&response, &records[row]).unwrap() {
                        local.sum_sq += (e * e) as f64;
                        local.worst = local.worst.max(e.abs());
                        local.coefficients += 1;
                    }
                    local.responses += 1;
                }
                let mut t = tally.lock().unwrap();
                t.responses += local.responses;
                t.secrets += local.secrets;
                t.coefficients += local.coefficients;
                t.sum_sq += local.sum_sq;
                t.worst = t.worst.max(local.worst);
                t.wrong_records += local.wrong_records;
            });
        }
    });
}

#[test]
fn served_set_noise_stays_in_its_band_on_fresh_secrets() {
    let params = Params::ring2048_cell256();
    let server = Server::new(&params).unwrap();
    let half_step = (params.q / 16 / 2) as f64;

    let worst: Vec<Vec<u8>> = vec![vec![0x88u8; 246]; 2048];
    let table = server.table(&cells_of(&worst)).unwrap();
    let secrets = env("SPECIAL_MODULUS_NOISE_SECRETS", 40);
    let per_secret = env("SPECIAL_MODULUS_NOISE_PER_SECRET", 25);
    let tally = Mutex::new(Tally::default());
    run(&server, &table, &worst, secrets, per_secret, &tally);
    let t = tally.into_inner().unwrap();
    let sd = (t.sum_sq / t.coefficients as f64).sqrt();
    let slack = half_step.log2() - LOG2_TAIL - sd.log2();
    let margin = half_step.log2() - (t.worst as f64).log2();
    eprintln!(
        "NOISE worst-value: {} responses, {} secrets, {} coefficients, sd {sd:.1}, slack {slack:.3} bits, worst |e| {} ({margin:.2} bits under the half step), wrong records {}",
        t.responses, t.secrets, t.coefficients, t.worst, t.wrong_records
    );
    assert!(t.responses >= 1000 && t.secrets >= 40, "too few samples");
    assert_eq!(t.wrong_records, 0);
    assert!(
        (SLACK_BAND.0..=SLACK_BAND.1).contains(&slack),
        "slack {slack:.3} outside {SLACK_BAND:?}"
    );
    assert!(
        margin >= WORST_MARGIN_BITS,
        "worst sample {margin:.2} bits under the half step"
    );

    let mut rng = ChaCha20Rng::seed_from_u64(246);
    let mut records: Vec<Vec<u8>> = SAMPLES.iter().map(|h| unhex(h)).collect();
    while records.len() < 2048 {
        let leaf = records.len() as u32;
        records.push(record_shaped(&mut rng, leaf));
    }
    let table = server.table(&cells_of(&records)).unwrap();
    let tally = Mutex::new(Tally::default());
    run(&server, &table, &records, 8, 16, &tally);
    let t = tally.into_inner().unwrap();
    let sd = (t.sum_sq / t.coefficients as f64).sqrt();
    eprintln!(
        "NOISE records: {} responses, {} secrets, sd {sd:.1}, slack {:.3} bits, wrong records {}",
        t.responses,
        t.secrets,
        half_step.log2() - LOG2_TAIL - sd.log2(),
        t.wrong_records
    );
    assert_eq!(t.wrong_records, 0);
}
