//! `respond_inspiring` latency at the d=2048 production cell. Emits p50/p95/p99
//! JSON tagged with the `parallel` feature so the two builds can be compared;
//! `RAVEN_BENCH_FINDINGS_DIR` persists it.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]

use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::InspireParams;
use raven_inspire::pir::{
    query, respond_inspiring, respond_inspiring_cached, setup_with_rng, ServerInspiringCache,
};

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[test]
#[ignore = "real-hardware p50/p95/p99 baseline for respond_inspiring at the d=2048 production \
            cell; run under --release. Trigger: changing the respond hot path, or comparing the \
            parallel feature against the sequential build."]
fn respond_latency_baseline() {
    let params = InspireParams::secure_128_d2048();
    let d = params.ring_dim;
    let entry_size = 512usize;

    let db: Vec<u8> = (0..d * entry_size).map(|i| (i % 251) as u8).collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 30003);
    let mut rng = ChaCha20Rng::seed_from_u64(30004);
    let (crs, encoded_db, rlwe_sk) =
        setup_with_rng(&params, &db, entry_size, &mut sampler, &mut rng).expect("setup");
    let (_state, q) = query(&crs, 7, &encoded_db.config, &rlwe_sk, &mut sampler).expect("query");

    // cache built once, as the server does
    let cache = ServerInspiringCache::new(&crs, &encoded_db).expect("cache");
    for _ in 0..3 {
        let _ = respond_inspiring_cached(&crs, &encoded_db, &q, &cache).expect("warmup cached");
    }

    const ITERS: usize = 30;
    let mut cached: Vec<Duration> = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let _ = respond_inspiring_cached(&crs, &encoded_db, &q, &cache).expect("cached respond");
        cached.push(t.elapsed());
    }
    cached.sort_unstable();

    // uncached rebuilds PackParams per call; few iters because it is slow
    let mut noncached: Vec<Duration> = Vec::with_capacity(5);
    for _ in 0..5 {
        let t = Instant::now();
        let _ = respond_inspiring(&crs, &encoded_db, &q).expect("noncached respond");
        noncached.push(t.elapsed());
    }
    noncached.sort_unstable();

    let ms = |x: Duration| x.as_secs_f64() * 1000.0;
    let parallel = cfg!(feature = "parallel");
    let json = serde_json::json!({
        "bench": "respond_latency",
        "cell": { "d": d, "entry_size_bytes": entry_size },
        "parallel": parallel,
        "iters": ITERS,
        "cached_respond_ms": {
            "p50": ms(percentile(&cached, 50.0)),
            "p95": ms(percentile(&cached, 95.0)),
            "p99": ms(percentile(&cached, 99.0)),
        },
        "noncached_respond_p50_ms": ms(percentile(&noncached, 50.0)),
    });
    eprintln!("{}", serde_json::to_string(&json).expect("json"));

    if let Ok(dir) = std::env::var("RAVEN_BENCH_FINDINGS_DIR") {
        let tag = if parallel { "parallel" } else { "sequential" };
        let path = std::path::Path::new(&dir).join(format!("respond_latency_{tag}.json"));
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(&path, serde_json::to_string_pretty(&json).expect("json"));
    }
}
