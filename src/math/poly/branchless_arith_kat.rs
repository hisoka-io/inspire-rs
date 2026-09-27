//! Old-vs-new differential for the branch- and divide-free ring arithmetic.
//!
//! Every `old_*` function is the spelling `Poly` shipped before, kept as the
//! oracle: a 128-bit remainder for the scalar product, `%` for residue
//! reduction, and a branch for the sum, difference and negation. Each
//! rewrite must return the same bytes at every modulus the crate ships, over
//! random inputs and the edges where a correction flips.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use super::{reduce_residues, scale_residues, Poly};
use crate::math::mod_q::DEFAULT_Q;
use crate::math::modular::{
    mul_mod_shoup, mul_mod_shoup_wide, shoup_precompute, SHOUP_NARROW_MAX_MODULUS,
};
use crate::params::{DEFAULT_CRT_MODULI, DEFAULT_Q_2CRT_30BIT};

const ADAPTIVE_CRT_MODULI: [u64; 2] = [67_043_329, 132_120_577];
/// The mod-switch targets the client decrypts under.
const SWITCH_TARGETS: [u64; 2] = [35_184_372_060_161, 68_718_428_161];

/// Every limb set a shipped `Poly` carries, plus the plaintext modulus.
const MODULI_SETS: [&[u64]; 8] = [
    &[DEFAULT_Q],
    &DEFAULT_CRT_MODULI,
    &DEFAULT_Q_2CRT_30BIT,
    &ADAPTIVE_CRT_MODULI,
    &[SWITCH_TARGETS[0]],
    &[SWITCH_TARGETS[1]],
    &[65_537],
    &[12_289],
];

/// Moduli outside the shipped sets that still reach the scalar product through
/// the public API, including both sides of the narrow Shoup bound.
const STRESS_MODULI: [u64; 8] = [
    2,
    3,
    (1 << 62) + 1,
    SHOUP_NARROW_MAX_MODULUS - 1,
    SHOUP_NARROW_MAX_MODULUS,
    SHOUP_NARROW_MAX_MODULUS + 1,
    u64::MAX - 58,
    u64::MAX,
];

const RANDOM_DRAWS: usize = 200_000;

fn old_scalar(c: u64, scalar: u64, modulus: u64) -> u64 {
    let scalar_mod = scalar % modulus;
    ((c as u128 * scalar_mod as u128) % modulus as u128) as u64
}

fn old_add(sum: u64, modulus: u64) -> u64 {
    if sum >= modulus {
        sum - modulus
    } else {
        sum
    }
}

/// Release semantics of the old `modulus - b + a`, which wraps on unreduced input.
fn old_sub(a: u64, b: u64, modulus: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        modulus.wrapping_sub(b).wrapping_add(a)
    }
}

fn old_neg(c: u64, modulus: u64) -> u64 {
    if c == 0 {
        0
    } else {
        modulus.wrapping_sub(c)
    }
}

fn all_moduli() -> Vec<u64> {
    let mut moduli: Vec<u64> = MODULI_SETS
        .iter()
        .flat_map(|set| set.iter().copied())
        .collect();
    moduli.extend(STRESS_MODULI);
    moduli
}

/// Reduced residues where a correction flips, then unreduced ones up to `2q - 1`
/// where the sum of two still fits a u64.
fn residue_edges(q: u64) -> Vec<u64> {
    let mut edges = vec![0, 1, 2, q / 2 - 1, q / 2, q / 2 + 1, q - 2, q - 1];
    if q < 1 << 61 {
        edges.extend([q, q + 1, 2 * q - 2, 2 * q - 1]);
    }
    edges
}

/// Multiplicands the scalar product must reduce correctly: any u64 at all.
fn multiplicand_edges(q: u64) -> Vec<u64> {
    let mut edges = residue_edges(q);
    edges.extend([
        q.wrapping_add(1),
        q.wrapping_mul(2),
        1 << 63,
        (1 << 63) - 1,
        u64::MAX - 1,
        u64::MAX,
    ]);
    edges
}

fn scalar_edges(q: u64) -> Vec<u64> {
    vec![
        0,
        1,
        2,
        1 << 20,
        1 << 40,
        q / 65_537,
        q / 2,
        q - 2,
        q - 1,
        q,
        q.wrapping_add(1),
        u64::MAX,
    ]
}

#[test]
fn shoup_product_matches_the_u128_remainder() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5C41_A401);
    for q in all_moduli() {
        for scalar in scalar_edges(q) {
            let b = scalar % q;
            let b_shoup = shoup_precompute(b, q);
            for a in multiplicand_edges(q) {
                let want = old_scalar(a, scalar, q);
                assert_eq!(
                    mul_mod_shoup_wide(a, b, b_shoup, q),
                    want,
                    "wide a={a} s={scalar} q={q}"
                );
                if q <= SHOUP_NARROW_MAX_MODULUS {
                    assert_eq!(
                        mul_mod_shoup(a, b, b_shoup, q),
                        want,
                        "a={a} s={scalar} q={q}"
                    );
                }
            }
        }
        for _ in 0..RANDOM_DRAWS / 10 {
            let (a, scalar): (u64, u64) = (rng.gen(), rng.gen());
            let b = scalar % q;
            let b_shoup = shoup_precompute(b, q);
            let want = old_scalar(a, scalar, q);
            assert_eq!(
                mul_mod_shoup_wide(a, b, b_shoup, q),
                want,
                "wide a={a} s={scalar} q={q}"
            );
            if q <= SHOUP_NARROW_MAX_MODULUS {
                assert_eq!(
                    mul_mod_shoup(a, b, b_shoup, q),
                    want,
                    "a={a} s={scalar} q={q}"
                );
            }
        }
    }
}

/// Both remainder arms must be reached, or a dropped correction would pass. The
/// estimate falls short with probability below `a / 2^64`, so the multiplicands
/// span the whole u64 range the scalar product accepts.
#[test]
fn shoup_draws_reach_both_correction_arms() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5C41_A402);
    for q in [
        DEFAULT_Q,
        DEFAULT_CRT_MODULI[0],
        SWITCH_TARGETS[1],
        u64::MAX,
    ] {
        let b = rng.gen_range(1..q);
        let b_shoup = shoup_precompute(b, q);
        let (mut exact, mut short) = (0usize, 0usize);
        for _ in 0..10_000 {
            let a: u64 = rng.gen();
            let estimate = ((u128::from(a) * u128::from(b_shoup)) >> 64) as u64;
            let rem = u128::from(a) * u128::from(b) - u128::from(estimate) * u128::from(q);
            if rem >= u128::from(q) {
                short += 1;
            } else {
                exact += 1;
            }
        }
        assert!(exact > 0 && short > 0, "q={q}: exact={exact} short={short}");
    }
}

#[test]
fn scale_residues_matches_the_old_scalar_product() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5C41_A403);
    for q in all_moduli() {
        let mut residues = multiplicand_edges(q);
        residues.extend((0..RANDOM_DRAWS / 10).map(|_| rng.gen::<u64>()));
        residues.extend((0..RANDOM_DRAWS / 10).map(|_| rng.gen_range(0..q)));
        let mut scalars = scalar_edges(q);
        scalars.push(rng.gen());
        for scalar in scalars {
            let want: Vec<u64> = residues.iter().map(|&c| old_scalar(c, scalar, q)).collect();
            let mut got = residues.clone();
            scale_residues(&mut got, scalar, q);
            assert_eq!(got, want, "scalar={scalar} q={q}");
        }
    }
}

#[test]
fn reduce_residues_matches_the_old_remainder() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5C41_A404);
    for q in all_moduli() {
        let mut residues = multiplicand_edges(q);
        residues.extend((0..RANDOM_DRAWS).map(|_| rng.gen::<u64>()));
        let want: Vec<u64> = residues.iter().map(|&c| c % q).collect();
        let mut got = residues;
        reduce_residues(&mut got, q);
        assert_eq!(got, want, "q={q}");
    }
}

/// A polynomial whose residues run through the edges, then random values, per limb.
fn edge_poly(moduli: &[u64], dim: usize, rng: &mut ChaCha20Rng, reduced: bool) -> Poly {
    let mut coeffs = Vec::with_capacity(dim * moduli.len());
    for &q in moduli {
        let edges = if reduced {
            residue_edges(q).into_iter().filter(|&e| e < q).collect()
        } else {
            residue_edges(q)
        };
        coeffs
            .extend((0..dim).map(|i| edges.get(i).copied().unwrap_or_else(|| rng.gen_range(0..q))));
    }
    Poly::from_crt_coeffs_reduced(coeffs, moduli)
}

fn limbwise(poly: &Poly, f: impl Fn(u64, u64) -> u64) -> Vec<u64> {
    let dim = poly.dimension();
    poly.coeffs()
        .iter()
        .enumerate()
        .map(|(i, &c)| f(c, poly.moduli()[i / dim]))
        .collect()
}

fn limbwise2(x: &Poly, y: &Poly, f: impl Fn(u64, u64, u64) -> u64) -> Vec<u64> {
    let dim = x.dimension();
    x.coeffs()
        .iter()
        .zip(y.coeffs())
        .enumerate()
        .map(|(i, (&a, &b))| f(a, b, x.moduli()[i / dim]))
        .collect()
}

#[test]
fn ring_sum_difference_and_negation_match_the_branching_forms() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5C41_A405);
    for moduli in MODULI_SETS {
        for dim in [16usize, 2048] {
            for reduced in [true, false] {
                let x = edge_poly(moduli, dim, &mut rng, reduced);
                let mut y = edge_poly(moduli, dim, &mut rng, reduced);
                // Pair every edge with every other at least once across limbs.
                y.coeffs_mut().rotate_left(3);

                assert_eq!(
                    (&x + &y).coeffs(),
                    limbwise2(&x, &y, |a, b, q| old_add(a + b, q))
                );
                assert_eq!((&x - &y).coeffs(), limbwise2(&x, &y, old_sub));
                assert_eq!((&y - &x).coeffs(), limbwise2(&y, &x, old_sub));
                assert_eq!((-&x).coeffs(), limbwise(&x, old_neg));
                assert_eq!((-&y).coeffs(), limbwise(&y, old_neg));

                let mut xn = x.clone();
                let mut yn = y.clone();
                xn.force_ntt_domain();
                yn.force_ntt_domain();
                let want_sum = limbwise2(&x, &y, |a, b, q| old_add(a + b, q));
                assert_eq!(xn.add_ntt_domain(&yn).coeffs(), want_sum);
                xn.add_assign_ntt_domain(&yn);
                assert_eq!(xn.coeffs(), want_sum);
            }
        }
    }
}

#[test]
fn poly_scalar_products_match_the_old_remainder() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5C41_A406);
    for moduli in MODULI_SETS {
        let x = edge_poly(moduli, 2048, &mut rng, false);
        let q: u64 = moduli.iter().product();
        for scalar in [
            0,
            1,
            1 << 20,
            1 << 40,
            q / 65_537,
            q - 1,
            u64::MAX,
            rng.gen(),
        ] {
            let want = limbwise(&x, |c, m| old_scalar(c, scalar, m));
            assert_eq!(
                x.scalar_mul(scalar).coeffs(),
                want,
                "scalar={scalar} {moduli:?}"
            );
            let mut y = x.clone();
            y.scalar_mul_assign(scalar);
            assert_eq!(y.coeffs(), want, "assign scalar={scalar} {moduli:?}");
        }
    }
}

#[test]
fn residue_constructors_match_the_old_remainder() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5C41_A407);
    for moduli in MODULI_SETS {
        let q: u64 = moduli.iter().product();
        let mut values = multiplicand_edges(q);
        values.extend(moduli.iter().flat_map(|&m| multiplicand_edges(m)));
        values.extend((values.len()..2048).map(|_| rng.gen::<u64>()));
        values.truncate(2048);

        let want: Vec<u64> = moduli
            .iter()
            .flat_map(|&m| values.iter().map(move |&c| c % m))
            .collect();
        assert_eq!(
            Poly::from_coeffs_moduli(values.clone(), moduli).coeffs(),
            want
        );

        let stacked: Vec<u64> = moduli.iter().flat_map(|_| values.iter().copied()).collect();
        assert_eq!(Poly::from_crt_coeffs(stacked, moduli).coeffs(), want);
    }
}

#[cfg(feature = "mod-switch-response")]
#[test]
fn switch_targets_are_the_shipped_ones() {
    assert_eq!(SWITCH_TARGETS, crate::pir::mod_switch::IMPLEMENTED_TARGETS);
}
