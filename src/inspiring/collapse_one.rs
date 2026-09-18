//! Dimension-reduction-by-one step underneath Collapse.

use crate::ks::KeySwitchingMatrix;
use crate::math::{NttContext, Poly};
use crate::params::InspireParams;
use crate::rgsw::gadget_decompose as rgsw_gadget_decompose;
use crate::rgsw::GadgetVector;

/// Absorb a_{k-1} into b via key-switching, returning a' of length k-1.
pub fn collapse_one(
    a: &[Poly],
    b: &Poly,
    ks_matrix: &KeySwitchingMatrix,
    params: &InspireParams,
) -> (Vec<Poly>, Poly) {
    let k = a.len();
    assert!(k >= 1, "Must have at least one polynomial to collapse");

    let d = params.ring_dim;
    let q = params.q;
    let ctx = params.ntt_context();
    let gadget = GadgetVector::new(params.gadget_base, params.packing_gadget_len, q);

    if k == 1 {
        let (result_a, new_b) = key_switch_component(&a[0], b, ks_matrix, &ctx, &gadget);
        if result_a.is_zero() {
            return (vec![], new_b);
        }
        return (vec![result_a], new_b);
    }

    let a_last = &a[k - 1];
    let decomposed = rgsw_gadget_decompose(a_last, &gadget);

    // (a', b') = (0, b) + sum_i decomposed_i * K[i]; both K rows are load-bearing.
    let mut result_a = Poly::zero_moduli(d, params.moduli());
    let mut result_b = b.clone();

    for (i, digit_poly) in decomposed.iter().enumerate() {
        if i < ks_matrix.len() {
            let ks_row = ks_matrix.get_row(i);

            let term_a = digit_poly.mul_ntt(&ks_row.a, &ctx);
            result_a = &result_a + &term_a;

            let term_b = digit_poly.mul_ntt(&ks_row.b, &ctx);
            result_b = &result_b + &term_b;
        }
    }

    let mut new_a: Vec<Poly> = a[..k - 1].to_vec();
    if !result_a.is_zero() {
        if new_a.is_empty() {
            new_a.push(result_a);
        } else {
            new_a[0] = &new_a[0] + &result_a;
        }
    }

    (new_a, result_b)
}

/// Fully absorb one a component; both K rows are load-bearing.
fn key_switch_component(
    a_component: &Poly,
    b: &Poly,
    ks_matrix: &KeySwitchingMatrix,
    ctx: &NttContext,
    gadget: &GadgetVector,
) -> (Poly, Poly) {
    let d = a_component.dimension();

    let decomposed = rgsw_gadget_decompose(a_component, gadget);

    let mut result_a = Poly::zero_moduli(d, a_component.moduli());
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

/// Split each coefficient into base-z digits; result i holds digit i of every
/// coefficient.
#[allow(dead_code)]
pub fn gadget_decompose(poly: &Poly, params: &InspireParams) -> Vec<Poly> {
    let gadget = GadgetVector::new(params.gadget_base, params.packing_gadget_len, params.q);
    rgsw_gadget_decompose(poly, &gadget)
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
    fn test_gadget_decompose() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(12345);
        let poly = random_poly(&mut rng, params.ring_dim, params.q, params.moduli());

        let decomposed = gadget_decompose(&poly, &params);
        assert_eq!(decomposed.len(), params.packing_gadget_len);

        for digit_poly in &decomposed {
            assert_eq!(digit_poly.dimension(), params.ring_dim);
        }
    }

    #[test]
    fn test_collapse_one_reduces_dimension() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(11111);
        let moduli = params.moduli();

        let k = 4;
        let a: Vec<Poly> = (0..k)
            .map(|_| random_poly(&mut rng, params.ring_dim, params.q, moduli))
            .collect();
        let b = random_poly(&mut rng, params.ring_dim, params.q, moduli);

        let ks_matrix =
            KeySwitchingMatrix::dummy(params.ring_dim, params.moduli(), params.packing_gadget_len);

        let (new_a, _new_b) = collapse_one(&a, &b, &ks_matrix, &params);

        assert_eq!(new_a.len(), k - 1);
    }
}
