//! Client query generation at the d=2048 production cell: the seeded query a
//! session emits per read, and the scaled inverse monomial inside it. Emits
//! p50/p95 JSON so a change to the client arithmetic can be compared on one box.
//! Run with `cargo bench --bench query_latency_bench`; `cargo test` never runs it.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]

use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::InspireParams;
use raven_inspire::pir::{inverse_monomial, setup_with_rng};
use raven_inspire::ClientSession;

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn timed(iters: usize, mut f: impl FnMut(usize)) -> Vec<Duration> {
    let mut samples: Vec<Duration> = (0..iters)
        .map(|i| {
            let t = Instant::now();
            f(i);
            t.elapsed()
        })
        .collect();
    samples.sort_unstable();
    samples
}

fn main() {
    let params = InspireParams::secure_128_d2048();
    let d = params.ring_dim;
    let entry_size = 512usize;

    let db: Vec<u8> = (0..d * entry_size).map(|i| (i % 251) as u8).collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 40001);
    let mut rng = ChaCha20Rng::seed_from_u64(40002);
    let (crs, encoded_db, rlwe_sk) =
        setup_with_rng(&params, &db, entry_size, &mut sampler, &mut rng).expect("setup");
    let session = ClientSession::new(crs, rlwe_sk, &mut sampler).expect("session");
    let shard_config = encoded_db.config;

    for i in 0..5 {
        let _ = session
            .query_seeded(i as u64, &shard_config, &mut sampler)
            .expect("warmup");
    }

    const ITERS: usize = 200;
    let queries = timed(ITERS, |i| {
        let _ = session
            .query_seeded((i * 37 % d) as u64, &shard_config, &mut sampler)
            .expect("query");
    });

    let q = params.q;
    let delta = params.delta();
    let moduli = params.moduli().to_vec();
    let monomials = timed(ITERS, |i| {
        let poly = inverse_monomial(i * 37 % d, d, q, &moduli).scalar_mul(delta);
        std::hint::black_box(poly);
    });

    let us = |x: Duration| x.as_secs_f64() * 1e6;
    let json = serde_json::json!({
        "bench": "query_latency",
        "cell": { "d": d, "entry_size_bytes": entry_size },
        "iters": ITERS,
        "query_seeded_us": {
            "p50": us(percentile(&queries, 50.0)),
            "p95": us(percentile(&queries, 95.0)),
        },
        "scaled_inverse_monomial_us": {
            "p50": us(percentile(&monomials, 50.0)),
            "p95": us(percentile(&monomials, 95.0)),
        },
    });
    eprintln!("{}", serde_json::to_string(&json).expect("json"));
}
