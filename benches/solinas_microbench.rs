//! Scalar and IFMA52 pointwise modular multiply, classical against Solinas, at
//! n=2048 and DEFAULT_Q. Runs only under `RAVEN_MICROBENCH=1`.

use raven_inspire::math::ifma52::avx512_ifma::pointwise_mont_mul_x8;
use raven_inspire::math::mod_q::DEFAULT_Q;
use raven_inspire::math::ntt::NttContext;
use raven_inspire::math::solinas_redc::{
    avx512_ifma::pointwise_solinas_mont_mul_x8, pointwise_solinas_mont_mul,
};

use std::time::Instant;

const N: usize = 2048;
const ITERS: usize = 50_000;

fn build_input(ctx: &NttContext, seed: u64) -> Vec<u64> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let q = ctx.modulus();
    (0..N)
        .map(|_| {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            z % q
        })
        .collect()
}

#[test]
#[ignore = "microbench; run with RAVEN_MICROBENCH=1 --release --ignored"]
fn microbench_solinas_pointwise_all_variants() {
    if std::env::var("RAVEN_MICROBENCH").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set RAVEN_MICROBENCH=1 to run");
        return;
    }

    let ctx = NttContext::with_default_q(N);
    let q = DEFAULT_Q;
    let q_inv_neg = ctx.q_inv_neg_for_test(0);
    let has_ifma = is_x86_feature_detected!("avx512ifma");

    let mut a = build_input(&ctx, 11);
    ctx.forward(&mut a);
    let mut b = build_input(&ctx, 17);
    ctx.forward(&mut b);

    let mut out_mont = vec![0u64; N];
    let mut out_sol_scalar = vec![0u64; N];
    let mut out_sol_ifma = vec![0u64; N];
    let mut out_ifma52 = vec![0u64; N];

    for _ in 0..100 {
        ctx.pointwise_mul(&a, &b, &mut out_mont);
        pointwise_solinas_mont_mul(&a, &b, &mut out_sol_scalar, q_inv_neg);
        std::hint::black_box(&out_mont);
        std::hint::black_box(&out_sol_scalar);
        if has_ifma {
            unsafe {
                pointwise_mont_mul_x8(&a, &b, &mut out_ifma52, q, q_inv_neg);
                pointwise_solinas_mont_mul_x8(&a, &b, &mut out_sol_ifma, q_inv_neg);
            }
            std::hint::black_box(&out_ifma52);
            std::hint::black_box(&out_sol_ifma);
        }
    }

    let t0 = Instant::now();
    for _ in 0..ITERS {
        ctx.pointwise_mul(&a, &b, &mut out_mont);
        std::hint::black_box(&out_mont);
    }
    let mont_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        pointwise_solinas_mont_mul(&a, &b, &mut out_sol_scalar, q_inv_neg);
        std::hint::black_box(&out_sol_scalar);
    }
    let sol_scalar_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    let ifma52_ns = if has_ifma {
        let t0 = Instant::now();
        for _ in 0..ITERS {
            unsafe {
                pointwise_mont_mul_x8(&a, &b, &mut out_ifma52, q, q_inv_neg);
            }
            std::hint::black_box(&out_ifma52);
        }
        t0.elapsed().as_nanos() as f64 / ITERS as f64
    } else {
        f64::NAN
    };

    let sol_ifma_ns = if has_ifma {
        let t0 = Instant::now();
        for _ in 0..ITERS {
            unsafe {
                pointwise_solinas_mont_mul_x8(&a, &b, &mut out_sol_ifma, q_inv_neg);
            }
            std::hint::black_box(&out_sol_ifma);
        }
        t0.elapsed().as_nanos() as f64 / ITERS as f64
    } else {
        f64::NAN
    };

    eprintln!("=== Pointwise mul, n={N}, {ITERS} iters, DEFAULT_Q ===");
    eprintln!(
        "  Scalar Montgomery (baseline): {mont_ns:>7.1} ns/call  ({:.3} μs)  1.000x",
        mont_ns / 1000.0
    );
    eprintln!(
        "  Scalar Solinas-Mont:          {sol_scalar_ns:>7.1} ns/call  ({:.3} μs)  {:.3}x",
        sol_scalar_ns / 1000.0,
        mont_ns / sol_scalar_ns
    );
    if has_ifma {
        eprintln!(
            "  IFMA52 split-52:              {ifma52_ns:>7.1} ns/call  ({:.3} μs)  {:.3}x",
            ifma52_ns / 1000.0,
            mont_ns / ifma52_ns
        );
        eprintln!(
            "  IFMA52 Solinas:               {sol_ifma_ns:>7.1} ns/call  ({:.3} μs)  {:.3}x",
            sol_ifma_ns / 1000.0,
            mont_ns / sol_ifma_ns
        );
        eprintln!();
        eprintln!(
            "  IFMA52 Solinas vs IFMA52 split-52: {:.3}x",
            ifma52_ns / sol_ifma_ns
        );
        eprintln!(
            "  Scalar Solinas vs scalar Mont:     {:.3}x",
            mont_ns / sol_scalar_ns
        );
    }

    assert_eq!(
        out_mont, out_sol_scalar,
        "scalar Solinas diverged from classical Montgomery"
    );
    if has_ifma {
        assert_eq!(
            out_mont, out_sol_ifma,
            "IFMA52 Solinas diverged from classical scalar Montgomery"
        );
    }
}

#[test]
#[ignore = "microbench; run with RAVEN_MICROBENCH=1 --release --ignored"]
fn microbench_solinas_forward_inverse_ntt() {
    if std::env::var("RAVEN_MICROBENCH").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set RAVEN_MICROBENCH=1 to run");
        return;
    }

    let ctx = NttContext::with_default_q(N);

    let iters = 5_000;

    for s in 0..16u64 {
        let mut v = build_input(&ctx, s);
        ctx.forward(&mut v);
    }
    for s in 0..16u64 {
        let mut v = build_input(&ctx, s);
        ctx.forward_solinas(&mut v);
    }

    let inputs: Vec<Vec<u64>> = (0..iters as u64).map(|s| build_input(&ctx, s)).collect();
    let t0 = Instant::now();
    for mut v in inputs.clone() {
        ctx.forward(&mut v);
        std::hint::black_box(&v);
    }
    let fwd_mont_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    let inputs: Vec<Vec<u64>> = (0..iters as u64).map(|s| build_input(&ctx, s)).collect();
    let t0 = Instant::now();
    for mut v in inputs {
        ctx.forward_solinas(&mut v);
        std::hint::black_box(&v);
    }
    let fwd_sol_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    let mont_inputs: Vec<Vec<u64>> = (0..iters as u64)
        .map(|s| {
            let mut v = build_input(&ctx, s);
            ctx.forward(&mut v);
            v
        })
        .collect();
    let t0 = Instant::now();
    for mut v in mont_inputs {
        ctx.inverse(&mut v);
        std::hint::black_box(&v);
    }
    let inv_mont_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    let sol_inputs: Vec<Vec<u64>> = (0..iters as u64)
        .map(|s| {
            let mut v = build_input(&ctx, s);
            ctx.forward_solinas(&mut v);
            v
        })
        .collect();
    let t0 = Instant::now();
    for mut v in sol_inputs {
        ctx.inverse_solinas(&mut v);
        std::hint::black_box(&v);
    }
    let inv_sol_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    eprintln!("=== NTT at n={N}, {iters} iters, DEFAULT_Q ===");
    eprintln!(
        "  Forward Montgomery:  {:>8.3} μs/call  1.000x",
        fwd_mont_ns / 1000.0
    );
    eprintln!(
        "  Forward Solinas:     {:>8.3} μs/call  {:.3}x",
        fwd_sol_ns / 1000.0,
        fwd_mont_ns / fwd_sol_ns
    );
    eprintln!(
        "  Inverse Montgomery:  {:>8.3} μs/call  1.000x",
        inv_mont_ns / 1000.0
    );
    eprintln!(
        "  Inverse Solinas:     {:>8.3} μs/call  {:.3}x",
        inv_sol_ns / 1000.0,
        inv_mont_ns / inv_sol_ns
    );

    let mut a = build_input(&ctx, 999);
    let original = a.clone();
    ctx.forward_solinas(&mut a);
    ctx.inverse_solinas(&mut a);
    assert_eq!(a, original, "Solinas forward+inverse roundtrip failed");
}
