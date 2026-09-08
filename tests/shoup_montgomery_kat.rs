//! The Shoup NTT path must be byte-identical to the Montgomery path at
//! DEFAULT_Q and at each 30-bit 2-CRT prime, cross-checked against u128
//! modular multiplication. Nothing Shoup-based enters the hot path until this
//! is green.

use raven_inspire::math::mod_q::DEFAULT_Q;
use raven_inspire::math::ntt::NttContext;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Six examples lived here (scalar-vs-naive at DEFAULT_Q, the two "_2crt_"
// tests - which despite their names built SINGLE-prime contexts, so
// crt_count()==1 and the per-limb indexing in pointwise_mul_shoup /
// shoup_precompute_vec was never exercised with two limbs - the two
// forward+inverse round trips, and the edge grid). All six are retired into
// simd_differential_properties.rs's pointwise_mul_shoup_matches_naive_over_
// full_vectors (2026-09-06), which walks all four context shapes every case,
// asserts the FULL vector on a genuine 2-CRT context, embeds the pairwise
// edge grid, and round-trips forward_shoup+inverse_shoup. Kill matrix
// (w4e evidence/w546-mc*.txt): the swapped-limb-moduli mutant left all six
// GREEN and kills the property; the dropped-final-subtract, unscaled-coeff-0
// and corrupted-twin mutants that killed them kill the property too.

#[test]
fn shoup_convolution_matches_montgomery_default_q() {
    let n = 256usize;
    let ctx = NttContext::with_default_q(n);
    let q = DEFAULT_Q;
    let mut rng = StdRng::seed_from_u64(0x1234_5678);

    for trial in 0..16 {
        let a: Vec<u64> = (0..n).map(|_| rng.gen_range(0..q)).collect();
        let b: Vec<u64> = (0..n).map(|_| rng.gen_range(0..q)).collect();

        let mut a_mont = a.clone();
        let mut b_mont = b.clone();
        ctx.forward(&mut a_mont);
        ctx.forward(&mut b_mont);
        let mut result_mont = vec![0u64; n];
        ctx.pointwise_mul(&a_mont, &b_mont, &mut result_mont);
        ctx.inverse(&mut result_mont);

        let mut a_shoup = a.clone();
        let mut b_shoup_coeffs = b.clone();
        ctx.forward_shoup(&mut a_shoup);
        ctx.forward_shoup(&mut b_shoup_coeffs);
        let b_shoup_twins = ctx.shoup_precompute_vec(&b_shoup_coeffs);
        let mut result_shoup = vec![0u64; n];
        ctx.pointwise_mul_shoup(&a_shoup, &b_shoup_coeffs, &b_shoup_twins, &mut result_shoup);
        ctx.inverse_shoup(&mut result_shoup);

        assert_eq!(
            result_mont, result_shoup,
            "Shoup convolution disagrees with Montgomery at trial={trial}"
        );
    }
}
