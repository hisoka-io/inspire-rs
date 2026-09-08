//! `solinas_mont_mul_default_q` must be byte-identical to classical Montgomery
//! REDC at DEFAULT_Q, and its AVX-512-IFMA variant lane-identical to it.
//! Nothing Solinas-based enters the hot path until this is green.

use raven_inspire::math::ifma52::mont_mul_split52;
use raven_inspire::math::mod_q::DEFAULT_Q;
use raven_inspire::math::ntt::NttContext;
use raven_inspire::math::solinas_redc::solinas_mont_mul_default_q;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

mod simd_capability;

fn naive_mul_mod(a: u64, b: u64, q: u64) -> u64 {
    (((a as u128) * (b as u128)) % (q as u128)) as u64
}

// The random-draw and edge-grid scalar examples are retired into
// simd_differential_properties.rs's solinas_scalar_matches_classical_
// montgomery_property (2026-09-06; the dropped-final-subtract mutant that
// killed them kills the property). solinas_ifma52_x8_matches_scalar is
// retired too: an x8 lane swap reddens the surviving _matches_naive below and
// the rescued full-vector pointwise differential, so scalar-equality was
// implied coverage (w4e evidence/w546-mc7a.txt).

/// Solinas matches classical at m = 0 and across the random m distribution.
#[test]
fn solinas_m_boundary_identities() {
    let q = DEFAULT_Q;
    let ctx = NttContext::with_default_q(2048);
    let q_inv_neg = ctx.q_inv_neg_for_test(0);

    assert_eq!(solinas_mont_mul_default_q(0, 0, q_inv_neg), 0);

    let mut rng = StdRng::seed_from_u64(0x9001_BEEF);
    for _ in 0..1000 {
        let a_mont = rng.gen_range(0..q);
        let b_mont = rng.gen_range(0..q);
        let ab = (a_mont as u128).wrapping_mul(b_mont as u128);
        let m = (ab as u64).wrapping_mul(q_inv_neg);

        let solinas = solinas_mont_mul_default_q(a_mont, b_mont, q_inv_neg);
        let classical = mont_mul_split52(a_mont, b_mont, q, q_inv_neg);
        assert_eq!(solinas, classical, "m boundary check: m={m}");
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn solinas_ifma52_x8_matches_naive() {
    if !simd_capability::require_avx512ifma("solinas_ifma52_x8_matches_naive") {
        return;
    }

    use raven_inspire::math::solinas_redc::avx512_ifma::pointwise_solinas_mont_mul_x8;

    let q = DEFAULT_Q;
    let ctx = NttContext::with_default_q(2048);
    let q_inv_neg = ctx.q_inv_neg_for_test(0);
    let mut rng = StdRng::seed_from_u64(0xD2AD_B33F);

    let len = 8192usize;
    let a_std: Vec<u64> = (0..len).map(|_| rng.gen_range(0..q)).collect();
    let b_std: Vec<u64> = (0..len).map(|_| rng.gen_range(0..q)).collect();

    let a_mont: Vec<u64> = a_std.iter().map(|&a| ctx.to_mont(a)).collect();
    let b_mont: Vec<u64> = b_std.iter().map(|&b| ctx.to_mont(b)).collect();

    let mut simd_out = vec![0u64; len];
    unsafe {
        pointwise_solinas_mont_mul_x8(&a_mont, &b_mont, &mut simd_out, q_inv_neg);
    }

    for i in 0..len {
        let got = ctx.from_mont(simd_out[i]);
        let want = naive_mul_mod(a_std[i], b_std[i], q);
        assert_eq!(
            got, want,
            "SIMD Solinas lane {i}: a={} b={} got={} want={}",
            a_std[i], b_std[i], got, want
        );
    }
}
