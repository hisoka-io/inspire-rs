//! Differential properties for the modular-arithmetic kernels: IFMA52 wide
//! product vs u128, Solinas-Montgomery vs classical REDC, the Shoup pointwise
//! path vs naive u128 over FULL vectors on a genuine 2-CRT context (the axis
//! no example test drove: shoup_montgomery_kat's `_2crt_` tests built
//! single-prime contexts and every pointwise example zeroed all but position
//! 0), and the fused mul_acc3 vs three serial mul_acc calls. Plus the rescued
//! microbench identities (scalar/IFMA Solinas pointwise vs Montgomery
//! pointwise, Solinas forward+inverse round trip), which previously lived only
//! behind RAVEN_MICROBENCH=1 and never ran in any CI lane. SIMD halves run
//! where the host has AVX-512-IFMA and loud-skip otherwise.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]

use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use raven_inspire::math::ifma52::{ifma52_product_lohi, mont_mul_split52};
use raven_inspire::math::mod_q::DEFAULT_Q;
use raven_inspire::math::ntt::NttContext;
use raven_inspire::math::solinas_redc::{pointwise_solinas_mont_mul, solinas_mont_mul_default_q};
use raven_inspire::math::Poly;
use raven_inspire::params::DEFAULT_Q_2CRT_30BIT;

mod simd_capability;

fn naive_mul_mod(a: u64, b: u64, q: u64) -> u64 {
    (((a as u128) * (b as u128)) % (q as u128)) as u64
}

/// Boundary operands; values >= the active q are filtered at use sites.
fn edges_for(q: u64) -> Vec<u64> {
    [
        0u64,
        1,
        2,
        1u64 << 14,
        1u64 << 32,
        (1u64 << 52) - 1,
        1u64 << 52,
        (1u64 << 52) + 1,
        1u64 << 59,
        q / 2,
        q - 2,
        q - 1,
    ]
    .into_iter()
    .filter(|&v| v < q)
    .collect()
}

const CASES: u32 = 8;
const DRAWS_PER_CASE: usize = 1024;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: CASES,
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    /// (a) ifma52_product_lohi(a, b) == (a as u128) * (b as u128), over random
    /// draws and the full pairwise edge grid every case.
    #[test]
    fn ifma52_product_lohi_matches_u128_property(seed in any::<u64>()) {
        let q = DEFAULT_Q;
        let mut rng = StdRng::seed_from_u64(seed);
        let edges = edges_for(q);

        let check = |a: u64, b: u64| -> Result<(), TestCaseError> {
            let (lo, hi) = ifma52_product_lohi(a, b);
            let combined: u128 = (lo as u128) | ((hi as u128) << 64);
            prop_assert_eq!(
                combined,
                (a as u128) * (b as u128),
                "lohi split diverged: a={} b={}",
                a,
                b
            );
            Ok(())
        };

        for _ in 0..DRAWS_PER_CASE {
            check(rng.gen_range(0..q), rng.gen_range(0..q))?;
        }
        for &a in &edges {
            for &b in &edges {
                check(a, b)?;
            }
        }
    }

    /// (b) solinas_mont_mul_default_q == mont_mul_split52 (and, stripped, the
    /// naive product), over random draws and the pairwise edge grid.
    #[test]
    fn solinas_scalar_matches_classical_montgomery_property(seed in any::<u64>()) {
        let q = DEFAULT_Q;
        let ctx = NttContext::with_default_q(64);
        let q_inv_neg = ctx.q_inv_neg_for_test(0);
        let mut rng = StdRng::seed_from_u64(seed);
        let edges = edges_for(q);

        let check = |a: u64, b: u64| -> Result<(), TestCaseError> {
            let a_mont = ctx.to_mont(a);
            let b_mont = ctx.to_mont(b);
            let solinas = solinas_mont_mul_default_q(a_mont, b_mont, q_inv_neg);
            let classical = mont_mul_split52(a_mont, b_mont, q, q_inv_neg);
            prop_assert_eq!(
                solinas,
                classical,
                "Solinas diverged from classical REDC: a={} b={}",
                a,
                b
            );
            prop_assert_eq!(
                ctx.from_mont(solinas),
                naive_mul_mod(a, b, q),
                "Solinas stripped diverged from naive: a={} b={}",
                a,
                b
            );
            Ok(())
        };

        for _ in 0..DRAWS_PER_CASE {
            check(rng.gen_range(0..q), rng.gen_range(0..q))?;
        }
        for &a in &edges {
            for &b in &edges {
                check(a, b)?;
            }
        }
    }

    /// (c) pointwise_mul_shoup == naive u128, ELEMENTWISE OVER FULL VECTORS,
    /// on every context shape including a genuine crt_count()==2 context with
    /// non-zero limb-1 coefficients; a per-limb modulus mix-up in
    /// pointwise_mul_shoup or shoup_precompute_vec has nowhere to hide. Each
    /// limb's head carries the pairwise edge grid, the tail is random. The
    /// same context then proves the Shoup forward+inverse round trip.
    #[test]
    fn pointwise_mul_shoup_matches_naive_over_full_vectors(
        n in prop::sample::select(&[64usize, 256, 2048]),
        seed in any::<u64>(),
    ) {
        // Every case walks all four context shapes: a drawn-shape axis let the
        // dropped-final-subtract mutant survive an 8-case run in which no
        // single-prime DEFAULT_Q shape happened to be drawn (measured
        // 2026-09-06; a 30-bit limb hits the subtract with p ~ 2^-34, so only
        // the 60-bit shape can catch it).
        for shape in 0..4usize {
            let ctx = match shape {
                0 => NttContext::with_default_q(n),
                1 => NttContext::with_moduli(n, &DEFAULT_Q_2CRT_30BIT),
                2 => NttContext::new(n, DEFAULT_Q_2CRT_30BIT[0]),
                _ => NttContext::new(n, DEFAULT_Q_2CRT_30BIT[1]),
            };
            let crt = ctx.crt_count();
            let total = n * crt;
            let mut rng = StdRng::seed_from_u64(seed ^ ((shape as u64) << 56));

            let mut a = vec![0u64; total];
            let mut b = vec![0u64; total];
            for idx in 0..crt {
                let q = ctx.moduli()[idx];
                let edges = edges_for(q);
                let e = edges.len();
                let start = idx * n;
                for i in 0..n {
                    if i < e * e {
                        a[start + i] = edges[i / e];
                        b[start + i] = edges[i % e];
                    } else {
                        a[start + i] = rng.gen_range(0..q);
                        b[start + i] = rng.gen_range(0..q);
                    }
                }
            }

            let b_shoup = ctx.shoup_precompute_vec(&b);
            let mut result = vec![0u64; total];
            ctx.pointwise_mul_shoup(&a, &b, &b_shoup, &mut result);

            for idx in 0..crt {
                let q = ctx.moduli()[idx];
                let start = idx * n;
                for i in 0..n {
                    prop_assert_eq!(
                        result[start + i],
                        naive_mul_mod(a[start + i], b[start + i], q),
                        "shoup pointwise diverged at shape {} limb {} position {} (q={}, a={}, b={})",
                        shape,
                        idx,
                        i,
                        q,
                        a[start + i],
                        b[start + i]
                    );
                }
            }

            let mut v = vec![0u64; total];
            for idx in 0..crt {
                let q = ctx.moduli()[idx];
                for i in 0..n {
                    v[idx * n + i] = rng.gen_range(0..q);
                }
            }
            let original = v.clone();
            ctx.forward_shoup(&mut v);
            ctx.inverse_shoup(&mut v);
            prop_assert_eq!(
                v,
                original,
                "Shoup forward+inverse round trip failed at shape {}",
                shape
            );
        }
    }

    /// (d) mul_acc3_ntt_domain == three chained mul_acc_ntt_domain calls,
    /// byte-identical, at single-prime DEFAULT_Q and at 2-CRT.
    #[test]
    fn mul_acc3_matches_serial_chain_property(
        two_crt in any::<bool>(),
        seed in any::<u64>(),
    ) {
        const D: usize = 256;
        let moduli: Vec<u64> = if two_crt {
            DEFAULT_Q_2CRT_30BIT.to_vec()
        } else {
            vec![DEFAULT_Q]
        };
        let ctx = NttContext::with_moduli(D, &moduli);
        let mut rng = StdRng::seed_from_u64(seed);

        let rand_ntt_poly = |rng: &mut StdRng| {
            let mut coeffs = Vec::with_capacity(D * moduli.len());
            for &q in &moduli {
                for _ in 0..D {
                    coeffs.push(rng.gen_range(0..q));
                }
            }
            let mut p = Poly::from_crt_coeffs(coeffs, &moduli);
            p.to_ntt(&ctx);
            p
        };

        for _ in 0..4 {
            let a0 = rand_ntt_poly(&mut rng);
            let b0 = rand_ntt_poly(&mut rng);
            let a1 = rand_ntt_poly(&mut rng);
            let b1 = rand_ntt_poly(&mut rng);
            let a2 = rand_ntt_poly(&mut rng);
            let b2 = rand_ntt_poly(&mut rng);
            let base = rand_ntt_poly(&mut rng);

            let mut serial = base.clone();
            serial.mul_acc_ntt_domain(&a0, &b0, &ctx);
            serial.mul_acc_ntt_domain(&a1, &b1, &ctx);
            serial.mul_acc_ntt_domain(&a2, &b2, &ctx);

            let mut fused = base.clone();
            fused.mul_acc3_ntt_domain(&a0, &b0, &a1, &b1, &a2, &b2, &ctx);

            prop_assert_eq!(fused.coeffs(), serial.coeffs(), "fused != serial chain");
            prop_assert_eq!(fused.is_ntt(), serial.is_ntt());
            prop_assert_eq!(fused.moduli(), serial.moduli());
        }
    }
}

/// Rescued from solinas_microbench (RAVEN_MICROBENCH-gated, so it never ran):
/// the scalar Solinas pointwise wrapper and both IFMA52 x8 pointwise kernels
/// must match the classical Montgomery pointwise product over full vectors in
/// the NTT domain. The scalar half is a genuine two-algorithm differential and
/// runs everywhere; the SIMD half runs where the host has AVX-512-IFMA.
#[test]
fn solinas_and_ifma52_pointwise_match_montgomery_full_vectors() {
    const N: usize = 2048;
    let ctx = NttContext::with_default_q(N);
    let q = DEFAULT_Q;
    let q_inv_neg = ctx.q_inv_neg_for_test(0);
    // Hoisted so the skip is named once, not once per seed.
    let simd_halves_run = simd_capability::require_avx512ifma(
        "solinas_and_ifma52_pointwise_match_montgomery_full_vectors (x8 halves; the \
         scalar half below runs everywhere)",
    );

    for seed in [11u64, 17, 23] {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut a: Vec<u64> = (0..N).map(|_| rng.gen_range(0..q)).collect();
        let mut b: Vec<u64> = (0..N).map(|_| rng.gen_range(0..q)).collect();
        ctx.forward(&mut a);
        ctx.forward(&mut b);

        let mut out_mont = vec![0u64; N];
        ctx.pointwise_mul(&a, &b, &mut out_mont);

        let mut out_sol_scalar = vec![0u64; N];
        pointwise_solinas_mont_mul(&a, &b, &mut out_sol_scalar, q_inv_neg);
        assert_eq!(
            out_mont, out_sol_scalar,
            "scalar Solinas pointwise diverged from classical Montgomery (seed {seed})"
        );

        #[cfg(target_arch = "x86_64")]
        if simd_halves_run {
            use raven_inspire::math::ifma52::avx512_ifma::pointwise_mont_mul_x8;
            use raven_inspire::math::solinas_redc::avx512_ifma::pointwise_solinas_mont_mul_x8;

            let mut out_sol_ifma = vec![0u64; N];
            let mut out_ifma52 = vec![0u64; N];
            unsafe {
                pointwise_solinas_mont_mul_x8(&a, &b, &mut out_sol_ifma, q_inv_neg);
                pointwise_mont_mul_x8(&a, &b, &mut out_ifma52, q, q_inv_neg);
            }
            assert_eq!(
                out_mont, out_sol_ifma,
                "IFMA52 Solinas pointwise diverged from classical Montgomery (seed {seed})"
            );
            assert_eq!(
                out_mont, out_ifma52,
                "IFMA52 split-52 pointwise diverged from classical Montgomery (seed {seed})"
            );
        }
    }
}

/// Rescued from solinas_microbench: forward_solinas + inverse_solinas is the
/// identity. Runs everywhere (the Solinas NTT path is scalar DEFAULT_Q-only).
#[test]
fn solinas_ntt_forward_inverse_round_trips() {
    for n in [256usize, 2048] {
        let ctx = NttContext::with_default_q(n);
        let q = DEFAULT_Q;
        let mut rng = StdRng::seed_from_u64(999 + n as u64);
        let original: Vec<u64> = (0..n).map(|_| rng.gen_range(0..q)).collect();
        let mut v = original.clone();
        ctx.forward_solinas(&mut v);
        ctx.inverse_solinas(&mut v);
        assert_eq!(
            v, original,
            "Solinas forward+inverse round trip failed at n={n}"
        );
    }
}
