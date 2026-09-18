//! Aggregated intermediate ciphertexts -> RLWE via the two key-switching matrices
//! K_g and K_h, where CDKS needed log(d).

use crate::ks::KeySwitchingMatrix;
use crate::math::{NttContext, Poly};
use crate::params::InspireParams;
use crate::rgsw::{gadget_decompose, GadgetVector};
use crate::rlwe::{apply_automorphism, galois_generators, RlweCiphertext};

use super::collapse_one::collapse_one;
use super::types::{AggregatedCiphertext, IntermediateCiphertext};

/// Halve the dimension log2(d) times under the cyclic automorphism K_g, then finish
/// with the conjugation K_h.
pub fn collapse(
    aggregated: &AggregatedCiphertext,
    k_g: &KeySwitchingMatrix,
    k_h: &KeySwitchingMatrix,
    params: &InspireParams,
) -> RlweCiphertext {
    let d = params.ring_dim;
    let ctx = params.ntt_context();

    let (g, h) = galois_generators(d);

    let mut current = aggregated.to_intermediate();

    let num_iterations = (d as f64).log2() as usize;

    for iter in 0..num_iterations {
        let rho = compute_rotation_param(iter, d);
        current = collapse_iteration(&current, k_g, g, rho, &ctx, params);
    }

    final_collapse(&current, k_h, h, &ctx, params)
}

/// One K_g level: combine coefficient pairs and emit RLWE.
pub fn collapse_half(
    ct: &IntermediateCiphertext,
    k_g: &KeySwitchingMatrix,
    rho: usize,
    params: &InspireParams,
) -> RlweCiphertext {
    let d = params.ring_dim;
    let ctx = params.ntt_context();
    let (g, _) = galois_generators(d);

    let collapsed = collapse_iteration(ct, k_g, g, rho, &ctx, params);

    if collapsed.a_polys.is_empty() {
        RlweCiphertext::from_parts(Poly::zero_moduli(d, params.moduli()), collapsed.b_poly)
    } else {
        RlweCiphertext::from_parts(collapsed.a_polys[0].clone(), collapsed.b_poly)
    }
}

/// Collapse for gamma <= d/2 ciphertexts, which needs only K_g.
pub fn collapse_partial(
    gamma: usize,
    ct: &IntermediateCiphertext,
    k_g: &KeySwitchingMatrix,
    params: &InspireParams,
) -> RlweCiphertext {
    let d = params.ring_dim;
    let ctx = params.ntt_context();

    assert!(gamma <= d / 2, "gamma must be <= d/2 for partial collapse");

    let num_iterations = (gamma as f64).log2().ceil() as usize;
    let (g, _) = galois_generators(d);

    let mut current = ct.clone();

    for iter in 0..num_iterations {
        let rho = compute_rotation_param(iter, gamma);
        current = collapse_iteration(&current, k_g, g, rho, &ctx, params);
    }

    if current.a_polys.is_empty() {
        RlweCiphertext::from_parts(Poly::zero_moduli(d, params.moduli()), current.b_poly)
    } else {
        let mut final_a = Poly::zero_moduli(d, params.moduli());
        let mut final_b = current.b_poly.clone();
        for a_poly in &current.a_polys {
            let (ks_a, ks_b) = key_switch_absorb(a_poly, &final_b, k_g, &ctx, params);
            final_a = &final_a + &ks_a;
            final_b = ks_b;
        }
        RlweCiphertext::from_parts(final_a, final_b)
    }
}

/// One collapse level under the automorphism tau_g.
fn collapse_iteration(
    ct: &IntermediateCiphertext,
    k_g: &KeySwitchingMatrix,
    g: usize,
    rho: usize,
    ctx: &NttContext,
    params: &InspireParams,
) -> IntermediateCiphertext {
    let q = params.q;

    let ct_rotated = apply_automorphism_to_intermediate(ct, g);
    let ct_shifted = shift_intermediate(&ct_rotated, rho, q);
    let ct_combined = add_intermediates(ct, &ct_shifted);

    // Key-switching undoes the automorphism's effect on the secret key.
    key_switch_intermediate(&ct_combined, k_g, ctx, params)
}

/// Closing K_h step.
fn final_collapse(
    ct: &IntermediateCiphertext,
    k_h: &KeySwitchingMatrix,
    h: usize,
    ctx: &NttContext,
    params: &InspireParams,
) -> RlweCiphertext {
    let d = params.ring_dim;

    let ct_conj = apply_automorphism_to_intermediate(ct, h);
    let ct_combined = add_intermediates(ct, &ct_conj);
    let switched = key_switch_intermediate(&ct_combined, k_h, ctx, params);

    if switched.a_polys.is_empty() {
        RlweCiphertext::from_parts(Poly::zero_moduli(d, params.moduli()), switched.b_poly)
    } else {
        let mut final_a = switched.a_polys[0].clone();
        let mut final_b = switched.b_poly.clone();

        for a_poly in &switched.a_polys[1..] {
            let (ks_a, ks_b) = key_switch_absorb(a_poly, &final_b, k_h, ctx, params);
            final_a = &final_a + &ks_a;
            final_b = ks_b;
        }

        RlweCiphertext::from_parts(final_a, final_b)
    }
}

fn apply_automorphism_to_intermediate(
    ct: &IntermediateCiphertext,
    g: usize,
) -> IntermediateCiphertext {
    let a_polys: Vec<Poly> = ct
        .a_polys
        .iter()
        .map(|p| apply_automorphism(p, g))
        .collect();
    let b_poly = apply_automorphism(&ct.b_poly, g);

    IntermediateCiphertext::new(a_polys, b_poly)
}

fn shift_intermediate(ct: &IntermediateCiphertext, k: usize, q: u64) -> IntermediateCiphertext {
    let a_polys: Vec<Poly> = ct
        .a_polys
        .iter()
        .map(|p| mul_by_monomial(p, k, q))
        .collect();
    let b_poly = mul_by_monomial(&ct.b_poly, k, q);

    IntermediateCiphertext::new(a_polys, b_poly)
}

/// Multiply by X^k in the negacyclic ring.
#[inline]
fn mul_by_monomial(poly: &Poly, k: usize, q: u64) -> Poly {
    let d = poly.dimension();
    let k = k % (2 * d);

    if k == 0 {
        return poly.clone();
    }

    let mut result_coeffs = vec![0u64; d];

    for i in 0..d {
        let coeff = poly.coeff(i);
        // Ciphertext coefficients are public to the server.
        if coeff == 0 {
            continue;
        }

        let new_idx = i + k;
        if new_idx < d {
            result_coeffs[new_idx] = mod_add(result_coeffs[new_idx], coeff, q);
        } else if new_idx < 2 * d {
            let actual_idx = new_idx - d;
            let neg_coeff = mod_sub(0, coeff, q);
            result_coeffs[actual_idx] = mod_add(result_coeffs[actual_idx], neg_coeff, q);
        } else {
            let actual_idx = new_idx - 2 * d;
            result_coeffs[actual_idx] = mod_add(result_coeffs[actual_idx], coeff, q);
        }
    }

    Poly::from_coeffs_moduli(result_coeffs, poly.moduli())
}

fn add_intermediates(
    ct1: &IntermediateCiphertext,
    ct2: &IntermediateCiphertext,
) -> IntermediateCiphertext {
    assert_eq!(ct1.dimension(), ct2.dimension());

    let a_polys: Vec<Poly> = ct1
        .a_polys
        .iter()
        .zip(ct2.a_polys.iter())
        .map(|(p1, p2)| p1 + p2)
        .collect();
    let b_poly = &ct1.b_poly + &ct2.b_poly;

    IntermediateCiphertext::new(a_polys, b_poly)
}

fn key_switch_intermediate(
    ct: &IntermediateCiphertext,
    ks_matrix: &KeySwitchingMatrix,
    _ctx: &NttContext,
    params: &InspireParams,
) -> IntermediateCiphertext {
    if ct.a_polys.is_empty() {
        return ct.clone();
    }

    let (new_a, new_b) = collapse_one(&ct.a_polys, &ct.b_poly, ks_matrix, params);

    IntermediateCiphertext::new(new_a, new_b)
}

/// Absorb one a component; both K rows are load-bearing.
fn key_switch_absorb(
    a_component: &Poly,
    b: &Poly,
    ks_matrix: &KeySwitchingMatrix,
    ctx: &NttContext,
    params: &InspireParams,
) -> (Poly, Poly) {
    let d = params.ring_dim;
    let q = params.q;
    let gadget = GadgetVector::new(params.gadget_base, params.packing_gadget_len, q);
    let decomposed = gadget_decompose(a_component, &gadget);

    let mut result_a = Poly::zero_moduli(d, params.moduli());
    let mut result_b = b.clone();

    for (i, digit_poly) in decomposed.iter().enumerate() {
        if i < ks_matrix.len() {
            let ks_row = ks_matrix.get_row(i);

            let term_a = digit_poly.mul_ntt(&ks_row.a, ctx);
            result_a = &result_a + &term_a;

            let term_b = digit_poly.mul_ntt(&ks_row.b, ctx);
            result_b = &result_b + &term_b;
        }
    }

    (result_a, result_b)
}

/// Rotation for level i is d / 2^(i+1).
fn compute_rotation_param(iteration: usize, d: usize) -> usize {
    d >> (iteration + 1)
}

#[inline]
fn mod_add(a: u64, b: u64, q: u64) -> u64 {
    let sum = a as u128 + b as u128;
    (sum % q as u128) as u64
}

#[inline]
fn mod_sub(a: u64, b: u64, q: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        q - b + a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;

    fn test_params() -> InspireParams {
        InspireParams::secure_128_d2048()
    }

    fn random_poly<R: Rng>(rng: &mut R, d: usize, q: u64, moduli: &[u64]) -> Poly {
        let coeffs: Vec<u64> = (0..d).map(|_| rng.gen_range(0..q)).collect();
        Poly::from_coeffs_moduli(coeffs, moduli)
    }

    #[test]
    fn test_mul_by_monomial_identity() {
        let d = 256;
        let q = 1152921504606830593u64;
        let moduli = vec![q];
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(12345);
        let poly = random_poly(&mut rng, d, q, &moduli);

        let result = mul_by_monomial(&poly, 0, q);

        for i in 0..d {
            assert_eq!(result.coeff(i), poly.coeff(i));
        }
    }

    #[test]
    fn test_mul_by_monomial_single_shift() {
        let d = 256;
        let q = 1152921504606830593u64;
        let mut coeffs = vec![0u64; d];
        coeffs[0] = 1;
        let poly = Poly::from_coeffs_moduli(coeffs, &[q]);

        let result = mul_by_monomial(&poly, 1, q);

        assert_eq!(result.coeff(1), 1);
        assert_eq!(result.coeff(0), 0);
    }

    #[test]
    fn test_mul_by_monomial_wraparound() {
        let d = 256;
        let q = 1152921504606830593u64;
        let mut coeffs = vec![0u64; d];
        coeffs[d - 1] = 1;
        let poly = Poly::from_coeffs_moduli(coeffs, &[q]);

        // X^(d-1) * X = X^d = -1
        let result = mul_by_monomial(&poly, 1, q);

        assert_eq!(result.coeff(0), q - 1);
    }

    #[test]
    fn test_add_intermediates() {
        let params = test_params();
        let moduli = params.moduli();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(54321);

        let a1 = vec![random_poly(&mut rng, params.ring_dim, params.q, moduli)];
        let b1 = random_poly(&mut rng, params.ring_dim, params.q, moduli);
        let ct1 = IntermediateCiphertext::new(a1.clone(), b1.clone());

        let a2 = vec![random_poly(&mut rng, params.ring_dim, params.q, moduli)];
        let b2 = random_poly(&mut rng, params.ring_dim, params.q, moduli);
        let ct2 = IntermediateCiphertext::new(a2.clone(), b2.clone());

        let sum = add_intermediates(&ct1, &ct2);

        assert_eq!(sum.dimension(), 1);
        for i in 0..params.ring_dim {
            let expected_a = (a1[0].coeff(i) as u128 + a2[0].coeff(i) as u128) % params.q as u128;
            assert_eq!(sum.a_polys[0].coeff(i), expected_a as u64);

            let expected_b = (b1.coeff(i) as u128 + b2.coeff(i) as u128) % params.q as u128;
            assert_eq!(sum.b_poly.coeff(i), expected_b as u64);
        }
    }

    #[test]
    fn test_compute_rotation_param() {
        let d = 2048;

        assert_eq!(compute_rotation_param(0, d), 1024);
        assert_eq!(compute_rotation_param(1, d), 512);
        assert_eq!(compute_rotation_param(2, d), 256);
        assert_eq!(compute_rotation_param(10, d), 1);
    }

    #[test]
    fn test_apply_automorphism_to_intermediate() {
        let params = test_params();
        let (g, _) = galois_generators(params.ring_dim);
        let moduli = params.moduli();

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(99999);
        let a = vec![random_poly(&mut rng, params.ring_dim, params.q, moduli)];
        let b = random_poly(&mut rng, params.ring_dim, params.q, moduli);
        let ct = IntermediateCiphertext::new(a.clone(), b.clone());

        let rotated = apply_automorphism_to_intermediate(&ct, g);

        let expected_a = apply_automorphism(&a[0], g);
        let expected_b = apply_automorphism(&b, g);

        for i in 0..params.ring_dim {
            assert_eq!(rotated.a_polys[0].coeff(i), expected_a.coeff(i));
            assert_eq!(rotated.b_poly.coeff(i), expected_b.coeff(i));
        }
    }

    #[test]
    fn test_collapse_partial_dimensions() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(11111);
        let moduli = params.moduli();

        let gamma = 4;
        let a_polys: Vec<Poly> = (0..gamma)
            .map(|_| random_poly(&mut rng, params.ring_dim, params.q, moduli))
            .collect();
        let b_poly = random_poly(&mut rng, params.ring_dim, params.q, moduli);
        let ct = IntermediateCiphertext::new(a_polys, b_poly);

        let k_g =
            KeySwitchingMatrix::dummy(params.ring_dim, params.moduli(), params.packing_gadget_len);

        let result = collapse_partial(gamma, &ct, &k_g, &params);

        assert_eq!(result.ring_dim(), params.ring_dim);
    }
}
