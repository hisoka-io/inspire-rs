//! Wall time of a respond at the production d=2048 cell with and without a packing
//! store on the shard, and of building the store. Each round answers the same query
//! both ways, in alternating order, so the two see the same machine load. Under the
//! `parallel` feature every measurement runs in a 16-thread and in a 1-thread pool.
//!
//! `RAVEN_BENCH_ROUNDS` sets the sample count per cell (default 30).

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stderr,
    reason = "bench harness; an abort here is the failure report"
)]

#[path = "../tests/packing_key_fixture/mod.rs"]
mod packing_key_fixture;

use std::time::Instant;

use packing_key_fixture::{inline_query, Cell};
use raven_inspire::inspiring::ShardPackingStore;
use raven_inspire::params::InspireParams;
use raven_inspire::pir::{
    respond_seeded_inspiring_cached_with_session, EncodedDatabase, SeededClientQuery,
    ServerSessionStore,
};

fn rounds() -> usize {
    std::env::var("RAVEN_BENCH_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(30)
}

fn load() -> String {
    std::fs::read_to_string("/proc/loadavg").map_or_else(
        |_| "unavailable".to_owned(),
        |text| text.split(' ').take(3).collect::<Vec<_>>().join(" "),
    )
}

fn report(metric: &str, entry_size: usize, threads: usize, mut samples_ms: Vec<f64>) {
    let listed: Vec<String> = samples_ms.iter().map(|ms| format!("{ms:.4}")).collect();
    eprintln!(
        "SAMPLES {metric} entry={entry_size} threads={threads} {}",
        listed.join(" ")
    );
    samples_ms.sort_by(f64::total_cmp);
    let n = samples_ms.len();
    eprintln!(
        "SUMMARY {metric} entry={entry_size} threads={threads} n={n} min={:.4} median={:.4} \
         p90={:.4} max={:.4} ms load=[{}]",
        samples_ms[0],
        samples_ms[n / 2],
        samples_ms[(n * 9) / 10],
        samples_ms[n - 1],
        load()
    );
}

fn time_ms<T>(work: impl FnOnce() -> T) -> f64 {
    let start = Instant::now();
    let out = work();
    let elapsed = start.elapsed().as_secs_f64() * 1e3;
    std::hint::black_box(out);
    elapsed
}

#[cfg(feature = "parallel")]
mod pools {
    pub type Pool = rayon::ThreadPool;

    pub fn all() -> Vec<(usize, Pool)> {
        [16usize, 1]
            .into_iter()
            .map(|threads| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("thread pool");
                (threads, pool)
            })
            .collect()
    }

    pub fn run<T: Send>(pool: &Pool, work: impl FnOnce() -> T + Send) -> T {
        pool.install(work)
    }
}

/// Without rayon the library walks its loops on the calling thread.
#[cfg(not(feature = "parallel"))]
mod pools {
    pub struct Pool;

    pub fn all() -> Vec<(usize, Pool)> {
        vec![(1, Pool)]
    }

    pub fn run<T>(_pool: &Pool, work: impl FnOnce() -> T) -> T {
        work()
    }
}

#[test]
#[ignore = "cost: about two minutes of responds at d=2048; trigger: a change to the respond \
            path, the offline packing phase or the shard packing store"]
fn respond_with_and_without_a_shard_store() {
    let params = InspireParams::secure_128_d2048();
    let ctx = params.ntt_context();
    let d = params.ring_dim as u64;
    let pools = pools::all();
    for entry_size in [256usize, 512] {
        let mut cell = Cell::build(&params, entry_size, 0x5354_4200);
        let mut precomputed = cell.encoded.clone();
        for shard in &mut precomputed.shards {
            shard
                .precompute_packing(cell.cache.pack_params(), &ctx)
                .unwrap();
        }

        let keys = cell.wire_keys();
        let store = ServerSessionStore::new();
        let handle = store
            .register_server_side(keys.clone(), cell.cache.pack_params(), &ctx)
            .unwrap();
        let inline: Vec<SeededClientQuery> = [17, d + 17, 2 * d + 17, 3 * d - 1]
            .into_iter()
            .map(|index| inline_query(&cell.next_query(index), &keys))
            .collect();
        let by_handle: Vec<SeededClientQuery> = inline
            .iter()
            .map(|query| {
                let mut query = query.clone();
                query.inspiring_packing_keys = None;
                query.session_handle = Some(handle);
                query
            })
            .collect();

        let respond = |encoded: &EncodedDatabase, query: &SeededClientQuery| {
            respond_seeded_inspiring_cached_with_session(
                &cell.crs,
                encoded,
                query,
                &cell.cache,
                Some(&store),
            )
            .unwrap()
        };
        for query in inline.iter().chain(&by_handle) {
            assert_eq!(
                bincode::serialize(&respond(&precomputed, query)).unwrap(),
                bincode::serialize(&respond(&cell.encoded, query)).unwrap(),
                "a store changes no response byte"
            );
        }

        for (key_source, queries) in [("seat", &by_handle), ("inline", &inline)] {
            for (threads, pool) in &pools {
                let mut per_query = Vec::new();
                let mut from_store = Vec::new();
                eprintln!(
                    "LOAD {key_source} entry={entry_size} threads={threads} start [{}]",
                    load()
                );
                for round in 0..rounds() + 2 {
                    let query = &queries[round % queries.len()];
                    let slow = || pools::run(pool, || time_ms(|| respond(&cell.encoded, query)));
                    let fast = || pools::run(pool, || time_ms(|| respond(&precomputed, query)));
                    let (a, b) = if round % 2 == 0 {
                        let a = slow();
                        (a, fast())
                    } else {
                        let b = fast();
                        (slow(), b)
                    };
                    if round >= 2 {
                        per_query.push(a);
                        from_store.push(b);
                    }
                }
                eprintln!(
                    "LOAD {key_source} entry={entry_size} threads={threads} end [{}]",
                    load()
                );
                report(
                    &format!("respond_per_query_{key_source}"),
                    entry_size,
                    *threads,
                    per_query,
                );
                report(
                    &format!("respond_shard_store_{key_source}"),
                    entry_size,
                    *threads,
                    from_store,
                );
            }
        }
    }
}

#[test]
#[ignore = "cost: about twenty seconds of store builds at d=2048; trigger: a change to \
            ShardPackingStore::build or to step 1 of the offline packing phase"]
fn shard_store_build() {
    let params = InspireParams::secure_128_d2048();
    let ctx = params.ntt_context();
    let pools = pools::all();
    for entry_size in [256usize, 512, 1024] {
        let cell = Cell::build(&params, entry_size, 0x5354_4201);
        let pack_params = cell.cache.pack_params();
        for (threads, pool) in &pools {
            let mut build = Vec::new();
            for round in 0..rounds().min(12) {
                let columns = &cell.encoded.shards[round % 3].polynomials;
                build.push(pools::run(pool, || {
                    time_ms(|| ShardPackingStore::build(pack_params, columns, &ctx).unwrap())
                }));
            }
            report("shard_store_build", entry_size, *threads, build);
        }
    }
}
