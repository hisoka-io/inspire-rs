//! LWE -> intermediate representation for ring packing; the message rides the
//! constant term of the plaintext polynomial.

use crate::lwe::LweCiphertext;
use crate::math::{ModQ, Poly};
use crate::params::InspireParams;

use super::types::IntermediateCiphertext;

/// (a, b) in Z_q^d x Z_q -> (a_hat, b_tilde) in R_q^d x R_q.
pub fn transform(lwe: &LweCiphertext, params: &InspireParams) -> IntermediateCiphertext {
    let d = params.ring_dim;
    let q = params.q;
    let moduli = params.moduli();

    debug_assert_eq!(lwe.a.len(), d, "LWE dimension must match ring dimension");
    debug_assert_eq!(lwe.q, q, "LWE modulus must match params");

    // Constant polys here; aggregation applies the X^k multipliers that align slots.
    let a_polys: Vec<Poly> = lwe
        .a
        .iter()
        .map(|&a_j| Poly::constant_moduli(a_j, d, moduli))
        .collect();

    let b_poly = Poly::constant_moduli(lwe.b, d, moduli);

    IntermediateCiphertext::new(a_polys, b_poly)
}

/// Transform for gamma <= d/2 ciphertexts, which needs only the K_g matrix.
pub fn transform_partial(
    gamma: usize,
    lwe: &LweCiphertext,
    params: &InspireParams,
) -> IntermediateCiphertext {
    let d = params.ring_dim;
    let q = params.q;
    let moduli = params.moduli();

    debug_assert_eq!(lwe.a.len(), d, "LWE dimension must match ring dimension");
    debug_assert_eq!(lwe.q, q, "LWE modulus must match params");
    assert!(
        gamma > 0 && gamma <= d / 2,
        "gamma must be in the range 1..=ring_dim/2 for partial packing"
    );

    let group_size = d / gamma;

    let mut a_polys = Vec::with_capacity(gamma);

    for j in 0..gamma {
        let mut coeffs = vec![0u64; d];
        for (k, coeff) in coeffs.iter_mut().enumerate().take(group_size) {
            let idx = j * group_size + k;
            if idx < lwe.a.len() {
                *coeff = lwe.a[idx];
            }
        }
        a_polys.push(Poly::from_coeffs_moduli(coeffs, moduli));
    }

    let b_poly = Poly::constant_moduli(lwe.b, d, moduli);

    IntermediateCiphertext::new(a_polys, b_poly)
}

/// Embed an LWE ciphertext at slot `slot_index` via the multiplier X^slot_index.
pub fn transform_at_slot(
    lwe: &LweCiphertext,
    slot_index: usize,
    params: &InspireParams,
) -> IntermediateCiphertext {
    let d = params.ring_dim;
    let q = params.q;
    let moduli = params.moduli();

    debug_assert!(slot_index < d, "slot_index must be < ring_dim");
    debug_assert_eq!(lwe.a.len(), d, "LWE dimension must match ring dimension");

    let a_polys: Vec<Poly> = lwe
        .a
        .iter()
        .map(|&a_j| {
            let mut coeffs = vec![0u64; d];
            if slot_index < d {
                coeffs[slot_index] = a_j;
            } else {
                // Negacyclic: X^d = -1, so an odd number of wraps flips the sign.
                let actual_idx = slot_index % d;
                let sign = if (slot_index / d) % 2 == 1 {
                    ModQ::negate(a_j, q)
                } else {
                    a_j
                };
                coeffs[actual_idx] = sign;
            }
            Poly::from_coeffs_moduli(coeffs, moduli)
        })
        .collect();

    let mut b_coeffs = vec![0u64; d];
    b_coeffs[slot_index] = lwe.b;
    let b_poly = Poly::from_coeffs_moduli(b_coeffs, moduli);

    IntermediateCiphertext::new(a_polys, b_poly)
}

/// Sum intermediates that `transform_at_slot` already positioned.
pub fn aggregate(
    intermediates: &[IntermediateCiphertext],
    params: &InspireParams,
) -> super::types::AggregatedCiphertext {
    let d = params.ring_dim;
    let moduli = params.moduli();
    let n = intermediates.len();

    assert!(
        !intermediates.is_empty(),
        "Must have at least one ciphertext"
    );
    assert!(n <= d, "Cannot aggregate more than d ciphertexts");

    let num_a_polys = intermediates[0].dimension();

    let mut agg_a_polys: Vec<Poly> = (0..num_a_polys)
        .map(|_| Poly::zero_moduli(d, moduli))
        .collect();
    let mut agg_b_poly = Poly::zero_moduli(d, moduli);

    for ct in intermediates {
        assert_eq!(
            ct.dimension(),
            num_a_polys,
            "All intermediates must have same dimension"
        );

        for (j, a_poly) in ct.a_polys.iter().enumerate() {
            agg_a_polys[j] = &agg_a_polys[j] + a_poly;
        }

        agg_b_poly = &agg_b_poly + &ct.b_poly;
    }

    super::types::AggregatedCiphertext::new(agg_a_polys, agg_b_poly)
}

/// Multiply by X^k in R_q = Z_q[X]/(X^d + 1); X^d = -1.
#[allow(dead_code)]
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
            result_coeffs[new_idx] = ModQ::add(result_coeffs[new_idx], coeff, q);
        } else if new_idx < 2 * d {
            let actual_idx = new_idx - d;
            let neg_coeff = ModQ::negate(coeff, q);
            result_coeffs[actual_idx] = ModQ::add(result_coeffs[actual_idx], neg_coeff, q);
        } else {
            let actual_idx = new_idx - 2 * d;
            result_coeffs[actual_idx] = ModQ::add(result_coeffs[actual_idx], coeff, q);
        }
    }

    Poly::from_coeffs_moduli(result_coeffs, poly.moduli())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;

    fn test_params() -> InspireParams {
        InspireParams::secure_128_d2048()
    }

    fn random_lwe<R: Rng>(rng: &mut R, params: &InspireParams) -> LweCiphertext {
        let a: Vec<u64> = (0..params.ring_dim)
            .map(|_| rng.gen_range(0..params.q))
            .collect();
        let b = rng.gen_range(0..params.q);
        LweCiphertext { a, b, q: params.q }
    }

    #[test]
    fn test_transform_dimensions() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(12345);
        let lwe = random_lwe(&mut rng, &params);

        let intermediate = transform(&lwe, &params);

        assert_eq!(intermediate.dimension(), params.ring_dim);
        assert_eq!(intermediate.ring_dim(), params.ring_dim);
        assert_eq!(intermediate.modulus(), params.q);
    }

    #[test]
    fn test_transform_partial_dimensions() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(12345);
        let lwe = random_lwe(&mut rng, &params);
        let gamma = params.ring_dim / 4;

        let intermediate = transform_partial(gamma, &lwe, &params);

        assert_eq!(intermediate.dimension(), gamma);
        assert_eq!(intermediate.ring_dim(), params.ring_dim);
    }

    #[test]
    #[should_panic(expected = "gamma must be in the range 1..=ring_dim/2")]
    fn test_transform_partial_rejects_zero_gamma() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(12345);
        let lwe = random_lwe(&mut rng, &params);
        let _ = transform_partial(0, &lwe, &params);
    }

    #[test]
    #[should_panic(expected = "gamma must be in the range 1..=ring_dim/2")]
    fn test_transform_partial_rejects_gamma_above_half_dimension() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(12345);
        let lwe = random_lwe(&mut rng, &params);
        let _ = transform_partial(params.ring_dim / 2 + 1, &lwe, &params);
    }

    #[test]
    fn test_mul_by_monomial_identity() {
        let params = test_params();
        let coeffs: Vec<u64> = (0..params.ring_dim as u64).collect();
        let poly = Poly::from_coeffs(coeffs.clone(), params.q);

        let result = mul_by_monomial(&poly, 0, params.q);

        for i in 0..params.ring_dim {
            assert_eq!(result.coeff(i), poly.coeff(i));
        }
    }

    #[test]
    fn test_mul_by_monomial_shift() {
        let d = 256;
        let q = 1152921504606830593u64;
        let mut coeffs = vec![0u64; d];
        coeffs[0] = 1;
        let poly = Poly::from_coeffs(coeffs, q);

        let result = mul_by_monomial(&poly, 1, q);
        assert_eq!(result.coeff(0), 0);
        assert_eq!(result.coeff(1), 1);
        for i in 2..d {
            assert_eq!(result.coeff(i), 0);
        }
    }

    #[test]
    fn test_mul_by_monomial_wraparound() {
        let d = 256;
        let q = 1152921504606830593u64;
        let mut coeffs = vec![0u64; d];
        coeffs[d - 1] = 1;
        let poly = Poly::from_coeffs(coeffs, q);

        // X * X^(d-1) = X^d = -1
        let result = mul_by_monomial(&poly, 1, q);
        assert_eq!(result.coeff(0), q - 1);
        for i in 1..d {
            assert_eq!(result.coeff(i), 0);
        }
    }

    #[test]
    fn test_aggregate_single() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(12345);
        let lwe = random_lwe(&mut rng, &params);

        let intermediate = transform(&lwe, &params);
        let aggregated = aggregate(std::slice::from_ref(&intermediate), &params);

        assert_eq!(aggregated.dimension(), intermediate.dimension());
        for i in 0..aggregated.dimension() {
            for j in 0..params.ring_dim {
                assert_eq!(
                    aggregated.a_polys[i].coeff(j),
                    intermediate.a_polys[i].coeff(j)
                );
            }
        }
    }

    #[test]
    fn test_aggregate_multiple() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(54321);

        let n_cts = 4;
        let intermediates: Vec<IntermediateCiphertext> = (0..n_cts)
            .map(|_| {
                let lwe = random_lwe(&mut rng, &params);
                transform(&lwe, &params)
            })
            .collect();

        let aggregated = aggregate(&intermediates, &params);

        assert_eq!(aggregated.dimension(), params.ring_dim);
    }

    #[test]
    fn test_transform_at_slot() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(99999);
        let lwe = random_lwe(&mut rng, &params);
        let slot = 5;

        let intermediate = transform_at_slot(&lwe, slot, &params);

        assert_eq!(intermediate.b_poly.coeff(slot), lwe.b);
        for i in 0..params.ring_dim {
            if i != slot {
                assert_eq!(intermediate.b_poly.coeff(i), 0);
            }
        }
    }
}
