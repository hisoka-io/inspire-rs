#![cfg(target_arch = "x86_64")]

//! The AVX-512-IFMA52 dispatch in `Poly::mul_acc_ntt_domain` must be
//! bit-identical to the scalar Solinas-Montgomery path at every coefficient,
//! over the gadget lengths the production sweep uses. Needs BOTH the host CPU
//! feature and `--features simd-packing-offline`; skips by name without either
//! and fails outright under `RAVEN_REQUIRE_AVX512=1`.

use raven_inspire::math::mod_q::DEFAULT_Q;
use raven_inspire::math::ntt::NttContext;
use raven_inspire::math::Poly;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

mod simd_capability;

/// Deterministic NTT-domain, Montgomery-form input, as `mul_acc_ntt_domain`
/// requires.
fn random_ntt_poly(seed: u64, n: usize, ctx: &NttContext) -> Poly {
    let mut rng = StdRng::seed_from_u64(seed);
    let q = DEFAULT_Q;
    let coeffs: Vec<u64> = (0..n).map(|_| rng.gen_range(0..q)).collect();
    let mut p = Poly::from_coeffs(coeffs, q);
    p.to_ntt(ctx);
    p
}

/// `gamma` is the gadget-decomposition fan-in, mirroring the backward
/// recursion's call shape.
fn run_kat_at(n: usize, gamma: usize, base_seed: u64) {
    let ctx = NttContext::with_default_q(n);

    let pairs: Vec<(Poly, Poly)> = (0..gamma)
        .map(|k| {
            let a = random_ntt_poly(base_seed + (k as u64) * 2, n, &ctx);
            let b = random_ntt_poly(base_seed + (k as u64) * 2 + 1, n, &ctx);
            (a, b)
        })
        .collect();

    let mut acc_ref = Poly::zero_default(n);
    acc_ref.to_ntt(&ctx);

    let mut acc_simd = acc_ref.clone();

    // mirrors the scalar branch of mul_acc_ntt_domain; raw slice because
    // coeff()/set_coeff() are forbidden in NTT domain
    let mut tmp_prod = vec![0u64; n];
    for (a, b) in &pairs {
        ctx.pointwise_mul(a.coeffs(), b.coeffs(), &mut tmp_prod);
        let acc_slice = acc_ref.coeffs_mut();
        for i in 0..n {
            let sum = acc_slice[i] + tmp_prod[i];
            acc_slice[i] = if sum >= DEFAULT_Q {
                sum - DEFAULT_Q
            } else {
                sum
            };
        }
    }

    for (a, b) in &pairs {
        acc_simd.mul_acc_ntt_domain(a, b, &ctx);
    }

    let ref_slice = acc_ref.coeffs();
    let simd_slice = acc_simd.coeffs();
    assert_eq!(
        ref_slice.len(),
        simd_slice.len(),
        "gamma={gamma} n={n} seed={base_seed}: coefficient slice lengths differ"
    );
    for i in 0..ref_slice.len() {
        assert_eq!(
            ref_slice[i], simd_slice[i],
            "gamma={gamma} n={n} seed={base_seed}: coeff[{i}] differs (ref={}, dispatched={})",
            ref_slice[i], simd_slice[i]
        );
    }
}

// Four per-gamma tests collapsed into one cell loop (2026-09-06): they
// differed only in (n, gamma, seed), and the accumulator-zeroing mutant that
// reddens any of them under --features simd-packing-offline reddens this loop
// at its first cell. Both preconditions are gates rather than notes: without
// the CPU feature the kernel cannot run, and without the cargo feature the
// dispatch IS the scalar reference, so the differential compares a loop with
// itself. RAVEN_REQUIRE_AVX512=1 turns either into a failure.
#[test]
fn dispatched_mul_acc_matches_scalar_over_production_cells() {
    const NAME: &str = "dispatched_mul_acc_matches_scalar_over_production_cells";
    if !simd_capability::require_avx512ifma(NAME) {
        return;
    }
    if !simd_capability::require_cargo_feature(
        NAME,
        "simd-packing-offline",
        cfg!(feature = "simd-packing-offline"),
    ) {
        return;
    }
    for (n, gamma, seed) in [
        (2048, 16, 0xC0DE_C0DE),
        (2048, 64, 0xBAD_CAFE),
        (2048, 256, 0xFEED_BEEF),
        (256, 16, 0x1234_5678),
    ] {
        run_kat_at(n, gamma, seed);
    }
}
