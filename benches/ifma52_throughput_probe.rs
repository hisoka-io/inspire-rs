//! madd52lo/madd52hi throughput in isolation; IFMA per-cycle rates differ
//! sharply by microarchitecture, so the budget is measured, not assumed.
//! Runs only under `RAVEN_MICROBENCH=1`.

#![allow(
    clippy::incompatible_msrv,
    reason = "AVX-512 intrinsics stabilised in 1.89; probe is CPUID-gated at runtime"
)]

use std::arch::x86_64::*;
use std::time::Instant;

#[test]
#[ignore = "microbench; run with RAVEN_MICROBENCH=1 --release --ignored"]
fn ifma52_madd_throughput_probe() {
    if std::env::var("RAVEN_MICROBENCH").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set RAVEN_MICROBENCH=1 to run");
        return;
    }
    if !is_x86_feature_detected!("avx512ifma") {
        eprintln!("SKIP: host lacks AVX-512-IFMA");
        return;
    }

    unsafe { probe_madd52_throughput() };
}

#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn probe_madd52_throughput() {
    const ITERS: usize = 10_000_000;
    const UNROLL: usize = 16;

    // independent accumulators, else the dependency chain sets the rate
    let mask52 = _mm512_set1_epi64(((1u64 << 52) - 1) as i64);
    let a = _mm512_set1_epi64(0x1234_5678_9ABC_DEF0i64);
    let b = _mm512_set1_epi64(0x0FED_CBA9_8765_4321i64);
    let a_lo = _mm512_and_si512(a, mask52);
    let b_lo = _mm512_and_si512(b, mask52);

    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();
    let mut acc2 = _mm512_setzero_si512();
    let mut acc3 = _mm512_setzero_si512();
    let mut acc4 = _mm512_setzero_si512();
    let mut acc5 = _mm512_setzero_si512();
    let mut acc6 = _mm512_setzero_si512();
    let mut acc7 = _mm512_setzero_si512();

    for _ in 0..1000 {
        acc0 = _mm512_madd52lo_epu64(acc0, a_lo, b_lo);
    }

    let t0 = Instant::now();
    for _ in 0..(ITERS / UNROLL) {
        acc0 = _mm512_madd52lo_epu64(acc0, a_lo, b_lo);
        acc1 = _mm512_madd52lo_epu64(acc1, a_lo, b_lo);
        acc2 = _mm512_madd52lo_epu64(acc2, a_lo, b_lo);
        acc3 = _mm512_madd52lo_epu64(acc3, a_lo, b_lo);
        acc4 = _mm512_madd52lo_epu64(acc4, a_lo, b_lo);
        acc5 = _mm512_madd52lo_epu64(acc5, a_lo, b_lo);
        acc6 = _mm512_madd52lo_epu64(acc6, a_lo, b_lo);
        acc7 = _mm512_madd52lo_epu64(acc7, a_lo, b_lo);
        acc0 = _mm512_madd52hi_epu64(acc0, a_lo, b_lo);
        acc1 = _mm512_madd52hi_epu64(acc1, a_lo, b_lo);
        acc2 = _mm512_madd52hi_epu64(acc2, a_lo, b_lo);
        acc3 = _mm512_madd52hi_epu64(acc3, a_lo, b_lo);
        acc4 = _mm512_madd52hi_epu64(acc4, a_lo, b_lo);
        acc5 = _mm512_madd52hi_epu64(acc5, a_lo, b_lo);
        acc6 = _mm512_madd52hi_epu64(acc6, a_lo, b_lo);
        acc7 = _mm512_madd52hi_epu64(acc7, a_lo, b_lo);
    }
    let elapsed = t0.elapsed();

    // consume, else DCE removes the loop
    let accs = [acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7];
    std::hint::black_box(accs);

    let ns_per_op = elapsed.as_nanos() as f64 / ITERS as f64;
    let cycles_per_op = ns_per_op * 5.0; // 9800X3D boost ~5 GHz
    eprintln!("=== IFMA52 madd52 throughput probe ===");
    eprintln!("  iters: {ITERS}, unroll: {UNROLL}");
    eprintln!("  total: {} μs", elapsed.as_micros());
    eprintln!("  ns/madd52: {ns_per_op:.3}");
    eprintln!("  cycles/madd52 (at 5 GHz): {cycles_per_op:.3}");
    eprintln!("  Expected Intel: ~0.5 cycles (2 IFMA ports at 1/cycle)");
    eprintln!("  Expected Zen 4: ~4 cycles (1 port at 1/4-cycle)");
}

#[test]
#[ignore = "microbench; run with RAVEN_MICROBENCH=1 --release --ignored"]
fn scalar_mulx_throughput_probe() {
    if std::env::var("RAVEN_MICROBENCH").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set RAVEN_MICROBENCH=1 to run");
        return;
    }

    const ITERS: usize = 10_000_000;
    const UNROLL: usize = 16;

    let a: u64 = 0x1234_5678_9ABC_DEF0;
    let b: u64 = 0x0FED_CBA9_8765_4321;
    let mut acc0 = 1u128;
    let mut acc1 = 1u128;
    let mut acc2 = 1u128;
    let mut acc3 = 1u128;
    let mut acc4 = 1u128;
    let mut acc5 = 1u128;
    let mut acc6 = 1u128;
    let mut acc7 = 1u128;

    for _ in 0..1000 {
        acc0 = acc0.wrapping_add((a as u128) * (b as u128));
    }

    let t0 = Instant::now();
    for _ in 0..(ITERS / UNROLL) {
        acc0 = acc0.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc1 = acc1.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc2 = acc2.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc3 = acc3.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc4 = acc4.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc5 = acc5.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc6 = acc6.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc7 = acc7.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc0 = acc0.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc1 = acc1.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc2 = acc2.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc3 = acc3.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc4 = acc4.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc5 = acc5.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc6 = acc6.wrapping_add((a as u128).wrapping_mul(b as u128));
        acc7 = acc7.wrapping_add((a as u128).wrapping_mul(b as u128));
    }
    let elapsed = t0.elapsed();
    std::hint::black_box([acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7]);

    let ns_per_op = elapsed.as_nanos() as f64 / ITERS as f64;
    let cycles_per_op = ns_per_op * 5.0;
    eprintln!("=== Scalar mulx throughput probe ===");
    eprintln!("  iters: {ITERS}, unroll: {UNROLL}");
    eprintln!("  total: {} μs", elapsed.as_micros());
    eprintln!("  ns/mulx: {ns_per_op:.3}");
    eprintln!("  cycles/mulx (at 5 GHz): {cycles_per_op:.3}");
    eprintln!("  Expected x86-64: ~1 cycle (1 port at 1/cycle)");
}
