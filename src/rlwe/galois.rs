//! Galois automorphisms `tau_g(X) = X^g` for g in `(Z/2dZ)^*`.

use crate::math::{NttContext, Poly};
use subtle::{Choice, ConditionallySelectable, ConstantTimeGreater};

use super::types::RlweCiphertext;

/// `p(X^g) mod (X^d + 1)`; `g` MUST be odd and coprime to 2d.
///
/// X^d = -1, so X^i lands at `g*i mod 2d` and negates once that index passes d.
#[inline]
pub fn apply_automorphism(poly: &Poly, g: usize) -> Poly {
    let d = poly.dimension();
    let moduli = poly.moduli();
    if poly.is_ntt() {
        let ctx = NttContext::with_moduli(d, moduli);
        let mut coefficients = poly.clone();
        coefficients.from_ntt(&ctx);
        let mut transformed = apply_automorphism(&coefficients, g);
        transformed.to_ntt(&ctx);
        return transformed;
    }

    let two_d = 2 * d;
    let mut result_coeffs = vec![0u64; d * moduli.len()];

    for (limb, &modulus) in moduli.iter().enumerate() {
        let offset = limb * d;
        for (i, &coeff) in poly.coeffs_modulus(limb).iter().enumerate() {
            let new_idx = (g * i) % two_d;
            let (actual_idx, negate) = if new_idx < d {
                (new_idx, false)
            } else {
                (new_idx - d, true)
            };
            let destination = &mut result_coeffs[offset + actual_idx];
            let added = ct_mod_add(*destination, coeff, modulus);
            let subtracted = ct_mod_sub(*destination, coeff, modulus);
            *destination =
                u64::conditional_select(&added, &subtracted, Choice::from(u8::from(negate)));
        }
    }

    Poly::from_crt_coeffs_reduced(result_coeffs, moduli)
}

/// `(tau_g(a), tau_g(b))`. The result decrypts under `tau_g(s)`, so callers MUST
/// key-switch back to `s`.
pub fn automorphism_ciphertext(ct: &RlweCiphertext, g: usize) -> RlweCiphertext {
    RlweCiphertext {
        a: apply_automorphism(&ct.a, g),
        b: apply_automorphism(&ct.b, g),
    }
}

/// Generators of `(Z/2dZ)^* = Z_{d/2} x Z_2` for power-of-two d: `(3, 2d - 1)`.
pub fn galois_generators(d: usize) -> (usize, usize) {
    debug_assert!(d.is_power_of_two(), "d must be a power of 2");
    debug_assert!(d >= 4, "d must be at least 4");

    let g1 = 3;
    // The negation automorphism: X^{-1} = -X^{d-1}.
    let g2 = 2 * d - 1;

    (g1, g2)
}

/// Order of g in `(Z/2dZ)^*`.
pub fn automorphism_order(g: usize, d: usize) -> usize {
    let two_d = 2 * d;
    let mut val = g % two_d;
    let mut order = 1;

    while val != 1 {
        val = (val * g) % two_d;
        order += 1;

        assert!(order <= two_d, "g={g} is not in (Z/{two_d}Z)^*");
    }

    order
}

/// True when g is odd, below 2d, and coprime to 2d.
pub fn is_valid_galois_element(g: usize, d: usize) -> bool {
    g % 2 == 1 && g < 2 * d && gcd(g, 2 * d) == 1
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[inline]
fn ct_mod_add(a: u64, b: u64, modulus: u64) -> u64 {
    let (sum, overflow) = a.overflowing_add(b);
    let reduce = Choice::from(u8::from(overflow)) | sum.ct_gt(&modulus.wrapping_sub(1));
    u64::conditional_select(&sum, &sum.wrapping_sub(modulus), reduce)
}

#[inline]
fn ct_mod_sub(a: u64, b: u64, modulus: u64) -> u64 {
    let (difference, underflow) = a.overflowing_sub(b);
    u64::conditional_select(
        &difference,
        &difference.wrapping_add(modulus),
        Choice::from(u8::from(underflow)),
    )
}

/// `tau_g1 . tau_g2 = tau_{g1*g2 mod 2d}`.
pub fn compose_automorphisms(g1: usize, g2: usize, d: usize) -> usize {
    (g1 * g2) % (2 * d)
}

/// `tau_g^{-1} = tau_{g^{-1} mod 2d}`, or `None` when `g` is not coprime to 2d.
#[must_use]
pub fn try_inverse_automorphism(g: usize, d: usize) -> Option<usize> {
    mod_inverse(g, 2 * d)
}

/// [`try_inverse_automorphism`] for callers that already screened `g`.
///
/// # Panics
///
/// When `g` is not coprime to 2d.
#[allow(
    clippy::expect_used,
    reason = "documented abort with a typed sibling: try_inverse_automorphism"
)]
#[must_use]
pub fn inverse_automorphism(g: usize, d: usize) -> usize {
    try_inverse_automorphism(g, d).expect("g must be coprime to 2d")
}

fn mod_inverse(a: usize, m: usize) -> Option<usize> {
    let (g, x, _) = extended_gcd(a as i64, m as i64);
    if g == 1 {
        Some(((x % m as i64 + m as i64) % m as i64) as usize)
    } else {
        None
    }
}

fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if a == 0 {
        (b, 0, 1)
    } else {
        let (g, x, y) = extended_gcd(b % a, a);
        (g, y - (b / a) * x, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::InspireParams;
    use proptest::prelude::*;

    fn test_params() -> InspireParams {
        InspireParams::secure_128_d2048()
    }

    #[test]
    fn test_automorphism_identity() {
        let params = test_params();
        let d = params.ring_dim;

        let coeffs: Vec<u64> = (0..d).map(|i| i as u64).collect();
        let poly = Poly::from_coeffs_moduli(coeffs.clone(), params.moduli());

        let result = apply_automorphism(&poly, 1);

        for (i, expected) in coeffs.iter().enumerate().take(d) {
            assert_eq!(result.coeff(i), *expected, "Identity failed at {i}");
        }
    }

    #[test]
    fn test_automorphism_composition() {
        let params = test_params();
        let d = params.ring_dim;

        let (g1, g2) = galois_generators(d);

        let coeffs: Vec<u64> = (0..d).map(|i| ((i * 17 + 5) as u64) % params.p).collect();
        let poly = Poly::from_coeffs_moduli(coeffs, params.moduli());

        let step1 = apply_automorphism(&poly, g1);
        let composed = apply_automorphism(&step1, g2);

        let g_combined = compose_automorphisms(g1, g2, d);
        let direct = apply_automorphism(&poly, g_combined);

        for i in 0..d {
            assert_eq!(
                composed.coeff(i),
                direct.coeff(i),
                "Composition failed at coefficient {i}"
            );
        }
    }

    #[test]
    fn test_automorphism_inverse() {
        let params = test_params();
        let d = params.ring_dim;
        let (g1, _) = galois_generators(d);

        let coeffs: Vec<u64> = (0..d).map(|i| ((i * 13 + 7) as u64) % params.p).collect();
        let poly = Poly::from_coeffs_moduli(coeffs.clone(), params.moduli());

        let g_inv = inverse_automorphism(g1, d);
        let forward = apply_automorphism(&poly, g1);
        let back = apply_automorphism(&forward, g_inv);

        for (i, expected) in coeffs.iter().enumerate().take(d) {
            assert_eq!(back.coeff(i), *expected, "Inverse failed at {i}");
        }
    }

    #[test]
    fn ntt_input_preserves_domain_and_matches_the_coefficient_transform() {
        let modulus_sets = [
            vec![1_152_921_504_606_830_593],
            vec![268_369_921, 249_561_089],
        ];
        for moduli in modulus_sets {
            let dimension = 256;
            let q = moduli.iter().product::<u64>();
            let ctx = NttContext::with_moduli(dimension, &moduli);
            let coefficients: Vec<u64> = (0..dimension)
                .map(|index| ((index * 19 + 7) as u64) % q)
                .collect();
            let coefficient_poly = Poly::from_coeffs_moduli(coefficients, &moduli);
            let mut ntt_poly = coefficient_poly.clone();
            ntt_poly.to_ntt(&ctx);

            let mut expected = apply_automorphism(&coefficient_poly, 3);
            expected.to_ntt(&ctx);
            let actual = apply_automorphism(&ntt_poly, 3);

            assert!(actual.is_ntt());
            let mismatches: Vec<_> = actual
                .coeffs()
                .iter()
                .zip(expected.coeffs())
                .enumerate()
                .filter(|(_, (actual, expected))| actual != expected)
                .map(|(index, (actual, expected))| (index, *actual, *expected))
                .take(8)
                .collect();
            assert!(mismatches.is_empty(), "moduli={moduli:?}: {mismatches:?}");
        }
    }

    fn reference_automorphism(poly: &Poly, g: usize) -> Vec<u64> {
        let d = poly.dimension();
        let mut expected = vec![0u64; poly.coeffs().len()];
        for (limb, &modulus) in poly.moduli().iter().enumerate() {
            let offset = limb * d;
            for i in 0..d {
                let mapped = (g * i) % (2 * d);
                let coeff = poly.coeffs_modulus(limb)[i];
                if mapped < d {
                    expected[offset + mapped] = coeff;
                } else {
                    expected[offset + mapped - d] = if coeff == 0 { 0 } else { modulus - coeff };
                }
            }
        }
        expected
    }

    proptest! {
        #[test]
        fn constant_time_modular_steps_match_u128(
            left in any::<u64>(),
            right in any::<u64>(),
            modulus in 1u64..=u64::MAX,
        ) {
            let a = left % modulus;
            let b = right % modulus;
            let expected_add = ((u128::from(a) + u128::from(b)) % u128::from(modulus)) as u64;
            let expected_sub = ((u128::from(a) + u128::from(modulus) - u128::from(b))
                % u128::from(modulus)) as u64;
            prop_assert_eq!(ct_mod_add(a, b, modulus), expected_add);
            prop_assert_eq!(ct_mod_sub(a, b, modulus), expected_sub);
        }

        #[test]
        fn automorphism_matches_the_negacyclic_permutation(
            coefficients in proptest::collection::vec(any::<u64>(), 16),
            generator in prop::sample::select(vec![1usize, 3, 5, 15, 17, 31]),
            two_crt in any::<bool>(),
        ) {
            let moduli = if two_crt {
                vec![268_369_921, 249_561_089]
            } else {
                vec![1_152_921_504_606_830_593]
            };
            let q = moduli.iter().product::<u64>();
            let reduced: Vec<u64> = coefficients.into_iter().map(|value| value % q).collect();
            let poly = Poly::from_coeffs_moduli(reduced, &moduli);
            let expected = reference_automorphism(&poly, generator);
            let actual = apply_automorphism(&poly, generator);
            prop_assert_eq!(actual.coeffs(), expected.as_slice());
        }
    }

    #[test]
    fn test_galois_generators() {
        let d = 2048;
        let (g1, g2) = galois_generators(d);

        assert_eq!(g1, 3);
        assert_eq!(g2, 4095);

        let order_g1 = automorphism_order(g1, d);
        assert_eq!(order_g1, d / 2, "g1 should have order d/2");

        let order_g2 = automorphism_order(g2, d);
        assert_eq!(order_g2, 2, "g2 should have order 2");
    }

    #[test]
    fn test_galois_generators_d4096() {
        let d = 4096;
        let (g1, g2) = galois_generators(d);

        assert_eq!(g1, 3);
        assert_eq!(g2, 8191);

        let order_g1 = automorphism_order(g1, d);
        assert_eq!(order_g1, d / 2);

        let order_g2 = automorphism_order(g2, d);
        assert_eq!(order_g2, 2);
    }

    #[test]
    fn test_negation_automorphism() {
        let params = test_params();
        let d = params.ring_dim;
        let (_, g2) = galois_generators(d);

        let mut coeffs = vec![0u64; d];
        coeffs[1] = 1;
        let poly = Poly::from_coeffs_moduli(coeffs, params.moduli());

        let result = apply_automorphism(&poly, g2);

        assert_eq!(result.coeff(d - 1), params.q - 1);
        for i in 0..d {
            if i != d - 1 {
                assert_eq!(result.coeff(i), 0);
            }
        }
    }

    #[test]
    fn test_valid_galois_elements() {
        let d = 2048;

        assert!(is_valid_galois_element(1, d));
        assert!(is_valid_galois_element(3, d));
        assert!(is_valid_galois_element(5, d));
        assert!(is_valid_galois_element(4095, d)); // 2d - 1

        assert!(!is_valid_galois_element(2, d));
        assert!(!is_valid_galois_element(4, d));
    }

    #[test]
    fn test_automorphism_ciphertext() {
        let params = test_params();
        let d = params.ring_dim;
        let (g1, _) = galois_generators(d);

        let a_coeffs: Vec<u64> = (0..d).map(|i| (i as u64 * 3) % params.q).collect();
        let b_coeffs: Vec<u64> = (0..d).map(|i| (i as u64 * 7 + 1) % params.q).collect();

        let a = Poly::from_coeffs_moduli(a_coeffs, params.moduli());
        let b = Poly::from_coeffs_moduli(b_coeffs, params.moduli());
        let ct = RlweCiphertext::from_parts(a.clone(), b.clone());

        let ct_auto = automorphism_ciphertext(&ct, g1);

        let a_auto = apply_automorphism(&a, g1);
        let b_auto = apply_automorphism(&b, g1);

        for i in 0..d {
            assert_eq!(ct_auto.a.coeff(i), a_auto.coeff(i));
            assert_eq!(ct_auto.b.coeff(i), b_auto.coeff(i));
        }
    }

    #[test]
    fn test_automorphism_linearity() {
        let params = test_params();
        let d = params.ring_dim;
        let (g1, _) = galois_generators(d);

        let p1_coeffs: Vec<u64> = (0..d).map(|i| (i as u64 * 11) % params.p).collect();
        let p2_coeffs: Vec<u64> = (0..d).map(|i| (i as u64 * 13 + 3) % params.p).collect();

        let p1 = Poly::from_coeffs_moduli(p1_coeffs, params.moduli());
        let p2 = Poly::from_coeffs_moduli(p2_coeffs, params.moduli());

        let sum = &p1 + &p2;
        let auto_sum = apply_automorphism(&sum, g1);

        let auto_p1 = apply_automorphism(&p1, g1);
        let auto_p2 = apply_automorphism(&p2, g1);
        let sum_auto = &auto_p1 + &auto_p2;

        for i in 0..d {
            assert_eq!(
                auto_sum.coeff(i),
                sum_auto.coeff(i),
                "Linearity failed at {i}"
            );
        }
    }
}
