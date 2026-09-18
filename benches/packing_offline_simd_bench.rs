//! Production-cell bench for the AVX-512-IFMA52 packing-offline dispatch.
//! Reports a 3-seed median, never a mean. Runs only under
//! `RAVEN_PACKING_OFFLINE_SIMD_BENCH=1`, since setup costs seconds per seed,
//! and skips where the dispatch would silently fall back to scalar.

#![allow(
    clippy::expect_used,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]
#![allow(
    clippy::used_underscore_binding,
    clippy::needless_pass_by_value,
    reason = "bench fixture plumbing"
)]

use std::time::Instant;

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::{
    respond_seeded_inspiring_cached_with_session, setup, ClientSession, PackingMode,
    ServerInspiringCache, ServerSessionStore,
};

const WARMUP_CALLS: usize = 2;
const TIMED_CALLS: usize = 5;
const SEEDS: [u64; 3] = [0x5EED_0001, 0x5EED_0002, 0x5EED_0003];

fn median_of(mut samples: Vec<u128>) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn drive_one_cell(label: &str, entries: usize, entry_bytes: usize, params: InspireParams) {
    eprintln!("=== {label}: entries={entries}, entry_bytes={entry_bytes} ===");
    eprintln!(
        "    ring_dim={}, gadget=(base=2^20, len={}), feature={}",
        params.ring_dim,
        params.packing_gadget_len,
        if cfg!(feature = "simd-packing-offline") {
            "simd-packing-offline"
        } else {
            "(scalar baseline)"
        }
    );

    let mut per_seed_medians: Vec<u128> = Vec::with_capacity(SEEDS.len());

    for &seed in &SEEDS {
        let mut sampler = GaussianSampler::with_seed(params.sigma, seed);

        let db: Vec<u8> = (0..entries * entry_bytes)
            .map(|i| ((i.wrapping_mul(0x9E37_79B9) >> 8) & 0xFF) as u8)
            .collect();

        let (crs, encoded_db, rlwe_sk) = setup(&params, &db, entry_bytes, &mut sampler)
            .expect("setup must succeed at production cell");

        let server_cache =
            ServerInspiringCache::new(&crs, &encoded_db).expect("server cache build must succeed");
        let session_store = ServerSessionStore::new();

        let mut session = ClientSession::new(crs.clone(), rlwe_sk, &mut sampler)
            .expect("client session must build");
        let _handle = session
            .register_with(&session_store)
            .expect("register must succeed")
            .expect("inspiring session must register a handle");

        // one query replayed, so query construction stays out of the measurement
        let target_idx = entries / 2 - 1;
        let (_state, mut query) = session
            .query_seeded(target_idx as u64, &encoded_db.config, &mut sampler)
            .expect("query_seeded must succeed");
        query.packing_mode = PackingMode::Inspiring;

        for _ in 0..WARMUP_CALLS {
            let _r = respond_seeded_inspiring_cached_with_session(
                &crs,
                &encoded_db,
                &query,
                &server_cache,
                Some(&session_store),
            )
            .expect("respond must succeed (warmup)");
            std::hint::black_box(&_r);
        }

        let mut per_call_ns: Vec<u128> = Vec::with_capacity(TIMED_CALLS);
        for _ in 0..TIMED_CALLS {
            let t0 = Instant::now();
            let r = respond_seeded_inspiring_cached_with_session(
                &crs,
                &encoded_db,
                &query,
                &server_cache,
                Some(&session_store),
            )
            .expect("respond must succeed (timed)");
            let elapsed = t0.elapsed().as_nanos();
            std::hint::black_box(&r);
            per_call_ns.push(elapsed);
        }

        let seed_median = median_of(per_call_ns.clone());
        per_seed_medians.push(seed_median);

        eprintln!(
            "  seed=0x{seed:08X}: per-call samples (ns) = {per_call_ns:?}, median = {seed_median} ns ({:.3} ms)",
            seed_median as f64 / 1_000_000.0
        );
    }

    let median = median_of(per_seed_medians.clone());
    eprintln!(
        "  3-seed medians (ns) = {per_seed_medians:?}, overall median = {median} ns ({:.3} ms)",
        median as f64 / 1_000_000.0
    );
    eprintln!();
}

/// Whether the bench body should run.
///
/// This used to be the whole story: unset the variable, the body early-returns, and libtest
/// reports `ok` - a passing test that measured nothing and could not fail. The tests now announce
/// the skip through the harness's own channel (`eprintln!` is invisible without `--nocapture`) and
/// the callers assert that they either measured something or were explicitly told not to.
fn should_run() -> bool {
    if std::env::var("RAVEN_PACKING_OFFLINE_SIMD_BENCH")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("SKIP: set RAVEN_PACKING_OFFLINE_SIMD_BENCH=1 to run");
        return false;
    }
    #[cfg(all(feature = "simd-packing-offline", target_arch = "x86_64"))]
    {
        if !is_x86_feature_detected!("avx512ifma") {
            eprintln!(
                "SKIP: simd-packing-offline feature is enabled but host lacks AVX-512-IFMA52"
            );
            return false;
        }
    }
    true
}

/// Production cell: 65536 entries x 512 B.
#[test]
#[ignore = "vacuous unless RAVEN_PACKING_OFFLINE_SIMD_BENCH=1: without it the body returns before \
            it measures anything and the test still reports PASS, so --ignored alone proves \
            nothing. Trigger: that variable plus --features simd-packing-offline --release on an \
            AVX-512-IFMA52 host, when changing the packing-offline kernel. ~12 s of setup per seed \
            x 3 seeds."]
fn bench_prod_cell_65536x512() {
    assert!(
        should_run(),
        "this bench measured nothing and must not report PASS. It is #[ignore]d, so reaching it \
         means someone asked for it explicitly; set RAVEN_PACKING_OFFLINE_SIMD_BENCH=1 (with \
         --features simd-packing-offline --release) or do not select it. A silent early return \
         here is a passing test that cannot fail."
    );

    let mut params = InspireParams::secure_128_d2048();
    params.security_level = SecurityLevel::Bits128;
    drive_one_cell("prod-cell-65536x512", 65_536, 512, params);
}

/// Production cell: 131072 entries x 32 B.
#[test]
#[ignore = "vacuous unless RAVEN_PACKING_OFFLINE_SIMD_BENCH=1: without it the body returns before \
            it measures anything and the test still reports PASS, so --ignored alone proves \
            nothing. Trigger: that variable plus --features simd-packing-offline --release on an \
            AVX-512-IFMA52 host, when changing the packing-offline kernel. ~12 s of setup per seed \
            x 3 seeds."]
fn bench_prod_cell_131072x32() {
    assert!(
        should_run(),
        "this bench measured nothing and must not report PASS. It is #[ignore]d, so reaching it \
         means someone asked for it explicitly; set RAVEN_PACKING_OFFLINE_SIMD_BENCH=1 (with \
         --features simd-packing-offline --release) or do not select it. A silent early return \
         here is a passing test that cannot fail."
    );

    let mut params = InspireParams::secure_128_d2048();
    params.security_level = SecurityLevel::Bits128;
    drive_one_cell("prod-cell-131072x32", 131_072, 32, params);
}
