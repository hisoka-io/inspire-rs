//! IFMA52 Montgomery must be byte-identical to the scalar reference before any
//! of it reaches the hot path. The SIMD case skips by name without
//! AVX-512-IFMA, and fails outright under `RAVEN_REQUIRE_AVX512=1`.

use raven_inspire::math::ifma52::mont_mul_split52;
use raven_inspire::math::mod_q::DEFAULT_Q;
use raven_inspire::math::ntt::NttContext;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

mod simd_capability;

/// `montgomery_mul_at` is private, so reach it through a `to_mont` round-trip.
fn ref_mont_mul(ctx: &NttContext, a: u64, b: u64) -> u64 {
    let a_mont = ctx.to_mont(a);
    let b_mont = ctx.to_mont(b);
    let ab_mont = ctx.pointwise_mul_single(a_mont, b_mont);
    ctx.from_mont(ab_mont)
}

fn naive_mul_mod(a: u64, b: u64, q: u64) -> u64 {
    (((a as u128) * (b as u128)) % (q as u128)) as u64
}

// The ifma52_product_lohi random and edge examples are retired into
// simd_differential_properties.rs's ifma52_product_lohi_matches_u128_property
// (2026-09-06; the hi-word misassembly mutant that killed both kills the
// property). ifma52_x8_matches_scalar_default_q is retired too: an x8 lane
// swap reddens the surviving _matches_naive below and the rescued full-vector
// pointwise differential (w4e evidence/w546-mc7b.txt). The two
// mont_mul_split52 examples below are KEPT: split52 is the classical oracle
// the Solinas property compares against, and an oracle keeps its own
// independent anchor.

#[test]
fn mont_mul_split52_matches_ref_default_q() {
    let q = DEFAULT_Q;
    let ctx = NttContext::with_default_q(2048);
    let q_inv_neg = ctx.q_inv_neg_for_test(0);

    let mut rng = StdRng::seed_from_u64(0xB0B_u64);
    for _ in 0..10_000 {
        let a = rng.gen_range(0..q);
        let b = rng.gen_range(0..q);

        let expected = ref_mont_mul(&ctx, a, b);

        // mont_mul_split52 computes a*b*R^-1, so feed Montgomery-form operands
        let a_mont = ctx.to_mont(a);
        let b_mont = ctx.to_mont(b);
        let candidate_mont_form = mont_mul_split52(a_mont, b_mont, q, q_inv_neg);
        let candidate = ctx.from_mont(candidate_mont_form);

        assert_eq!(
            candidate, expected,
            "mont_mul_split52 disagrees with reference: a={a} b={b} \
             candidate={candidate} expected={expected}"
        );
        assert_eq!(candidate, naive_mul_mod(a, b, q));
    }
}

#[test]
fn mont_mul_split52_edge_cases_default_q() {
    let q = DEFAULT_Q;
    let ctx = NttContext::with_default_q(2048);
    let q_inv_neg = ctx.q_inv_neg_for_test(0);

    let edges: Vec<u64> = vec![0, 1, (1u64 << 52) - 1, 1u64 << 52, q / 2, q - 2, q - 1];
    for &a in &edges {
        if a >= q {
            continue;
        }
        for &b in &edges {
            if b >= q {
                continue;
            }
            let a_mont = ctx.to_mont(a);
            let b_mont = ctx.to_mont(b);
            let candidate = ctx.from_mont(mont_mul_split52(a_mont, b_mont, q, q_inv_neg));
            let expected = naive_mul_mod(a, b, q);
            assert_eq!(
                candidate, expected,
                "edge a={a} b={b} got={candidate} expected={expected}"
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn ifma52_x8_matches_naive_default_q() {
    if !simd_capability::require_avx512ifma("ifma52_x8_matches_naive_default_q") {
        return;
    }

    use raven_inspire::math::ifma52::avx512_ifma::pointwise_mont_mul_x8;

    let q = DEFAULT_Q;
    let ctx = NttContext::with_default_q(2048);
    let q_inv_neg = ctx.q_inv_neg_for_test(0);

    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    let len = 8192usize;

    let a_std: Vec<u64> = (0..len).map(|_| rng.gen_range(0..q)).collect();
    let b_std: Vec<u64> = (0..len).map(|_| rng.gen_range(0..q)).collect();

    let a_mont: Vec<u64> = a_std.iter().map(|&a| ctx.to_mont(a)).collect();
    let b_mont: Vec<u64> = b_std.iter().map(|&b| ctx.to_mont(b)).collect();

    let mut simd_out = vec![0u64; len];
    unsafe {
        pointwise_mont_mul_x8(&a_mont, &b_mont, &mut simd_out, q, q_inv_neg);
    }

    for i in 0..len {
        let got = ctx.from_mont(simd_out[i]);
        let want = naive_mul_mod(a_std[i], b_std[i], q);
        assert_eq!(
            got, want,
            "lane {i}: a={} b={} got={} want={}",
            a_std[i], b_std[i], got, want
        );
    }
}
