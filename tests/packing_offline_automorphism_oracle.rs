//! Derived oracle for the `packing_offline` automorphism and backward recursion.
//!
//! The reference stays in coefficient form and derives every operation from
//! `X^n = -1` and `tau_g(X^i) = X^(gi)`. It does not read the production
//! monomial tables, automorphism tables, or any golden output.

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test oracle; indexed coefficient equations are the independent specification"
)]

use raven_inspire::inspiring::{packing_offline, OfflinePackingKeys, PackParams};
use raven_inspire::math::Poly;
use raven_inspire::params::{InspireParams, SecurityLevel, DEFAULT_Q_2CRT_30BIT};

const RING_DIM: usize = 16;
const PACKING_WIDTH: usize = 4;

struct ReferenceOffline {
    a_hat: Poly,
    bold_t: Vec<Vec<Poly>>,
}

fn add_mod(left: u64, right: u64, modulus: u64) -> u64 {
    ((u128::from(left) + u128::from(right)) % u128::from(modulus)) as u64
}

fn sub_mod(left: u64, right: u64, modulus: u64) -> u64 {
    ((u128::from(left) + u128::from(modulus) - u128::from(right)) % u128::from(modulus)) as u64
}

fn scalar_mul(poly: &Poly, scalar: u64) -> Poly {
    let modulus = poly.modulus();
    let coefficients = (0..poly.dimension())
        .map(|position| {
            ((u128::from(poly.coeff(position)) * u128::from(scalar)) % u128::from(modulus)) as u64
        })
        .collect();
    Poly::from_coeffs_moduli(coefficients, poly.moduli())
}

fn add_poly(left: &Poly, right: &Poly) -> Poly {
    let modulus = left.modulus();
    let coefficients = (0..left.dimension())
        .map(|position| add_mod(left.coeff(position), right.coeff(position), modulus))
        .collect();
    Poly::from_coeffs_moduli(coefficients, left.moduli())
}

fn mul_negacyclic(left: &Poly, right: &Poly) -> Poly {
    let ring_dim = left.dimension();
    let modulus = left.modulus();
    let mut coefficients = vec![0u64; ring_dim];
    for left_position in 0..ring_dim {
        for right_position in 0..ring_dim {
            let product = ((u128::from(left.coeff(left_position))
                * u128::from(right.coeff(right_position)))
                % u128::from(modulus)) as u64;
            let exponent = left_position + right_position;
            if exponent < ring_dim {
                coefficients[exponent] = add_mod(coefficients[exponent], product, modulus);
            } else {
                let wrapped = exponent - ring_dim;
                coefficients[wrapped] = sub_mod(coefficients[wrapped], product, modulus);
            }
        }
    }
    Poly::from_coeffs_moduli(coefficients, left.moduli())
}

fn mul_monomial(poly: &Poly, exponent: usize) -> Poly {
    let ring_dim = poly.dimension();
    let modulus = poly.modulus();
    let reduced_exponent = exponent % (2 * ring_dim);
    let mut coefficients = vec![0u64; ring_dim];
    for position in 0..ring_dim {
        let mapped = position + reduced_exponent;
        let wraps = mapped / ring_dim;
        let destination = mapped % ring_dim;
        if wraps.is_multiple_of(2) {
            coefficients[destination] =
                add_mod(coefficients[destination], poly.coeff(position), modulus);
        } else {
            coefficients[destination] =
                sub_mod(coefficients[destination], poly.coeff(position), modulus);
        }
    }
    Poly::from_coeffs_moduli(coefficients, poly.moduli())
}

fn automorphism(poly: &Poly, galois_element: usize) -> Poly {
    let ring_dim = poly.dimension();
    let modulus = poly.modulus();
    let mut coefficients = vec![0u64; ring_dim];
    for position in 0..ring_dim {
        let mapped = (galois_element * position) % (2 * ring_dim);
        if mapped < ring_dim {
            coefficients[mapped] = add_mod(coefficients[mapped], poly.coeff(position), modulus);
        } else {
            let destination = mapped - ring_dim;
            coefficients[destination] =
                sub_mod(coefficients[destination], poly.coeff(position), modulus);
        }
    }
    Poly::from_coeffs_moduli(coefficients, poly.moduli())
}

fn decompose(poly: &Poly, base: u64, digit_count: usize) -> Vec<Poly> {
    let ring_dim = poly.dimension();
    let mut digits = vec![vec![0u64; ring_dim]; digit_count];
    for position in 0..ring_dim {
        let mut coefficient = poly.coeff(position);
        for digit in &mut digits {
            digit[position] = coefficient % base;
            coefficient /= base;
        }
    }
    digits
        .into_iter()
        .map(|coefficients| Poly::from_coeffs_moduli(coefficients, poly.moduli()))
        .collect()
}

fn reference_offline(
    pack_params: &PackParams,
    packing_keys: &OfflinePackingKeys,
    inputs: &[Poly],
) -> ReferenceOffline {
    let ring_dim = pack_params.ring_dim;
    let width = pack_params.num_to_pack;
    let mut r_all = Vec::with_capacity(width);

    for rotation in 0..width {
        let inverse_power = pack_params.gen_pows[(ring_dim - rotation) % ring_dim];
        let mut combined = Poly::zero_moduli(ring_dim, &pack_params.moduli);
        for (column, input) in inputs.iter().enumerate() {
            let exponent = (column * inverse_power) % (2 * ring_dim);
            combined = add_poly(&combined, &mul_monomial(input, exponent));
        }
        let scaled = scalar_mul(&combined, pack_params.mod_inv_gamma);
        r_all.push(automorphism(&scaled, pack_params.gen_pows[rotation]));
    }

    let mut bold_t_reverse = Vec::with_capacity(width - 1);
    for rotation in (0..(width - 1)).rev() {
        let gadget_digits = decompose(
            &r_all[rotation + 1],
            pack_params.gadget.base,
            pack_params.gadget.len,
        );
        for (digit, gadget_poly) in gadget_digits.iter().enumerate() {
            let contribution = mul_negacyclic(&packing_keys.w_all[rotation][digit], gadget_poly);
            r_all[rotation] = add_poly(&r_all[rotation], &contribution);
        }
        bold_t_reverse.push(gadget_digits);
    }
    bold_t_reverse.reverse();

    ReferenceOffline {
        a_hat: r_all[0].clone(),
        bold_t: bold_t_reverse,
    }
}

fn params_with_moduli(moduli: Vec<u64>) -> InspireParams {
    InspireParams {
        ring_dim: RING_DIM,
        q: moduli.iter().product(),
        crt_moduli: moduli,
        p: 65_537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

fn asymmetric_inputs(params: &InspireParams) -> Vec<Poly> {
    (0..PACKING_WIDTH)
        .map(|column| {
            let coefficients = (0..RING_DIM)
                .map(|position| {
                    let column = column as u64 + 1;
                    let position = position as u64 + 1;
                    (column * 1_009 + position * position * 37 + column * position * 101) % params.q
                })
                .collect();
            Poly::from_coeffs_moduli(coefficients, params.moduli())
        })
        .collect()
}

fn assert_poly_eq(label: &str, actual: &Poly, expected: &Poly) {
    assert_eq!(
        actual.dimension(),
        expected.dimension(),
        "{label}: dimension"
    );
    assert_eq!(actual.moduli(), expected.moduli(), "{label}: moduli");
    assert_eq!(actual.is_ntt(), expected.is_ntt(), "{label}: domain");
    assert_eq!(actual.coeffs(), expected.coeffs(), "{label}: coefficients");
}

fn assert_oracle_case(label: &str, params: &InspireParams) {
    params.validate().expect("oracle parameters");
    let ctx = params.ntt_context();
    let pack_params =
        PackParams::try_new(params, PACKING_WIDTH).expect("legal partial packing width");
    let packing_keys = OfflinePackingKeys::generate(&pack_params, [0x5a; 32]);
    let inputs = asymmetric_inputs(params);
    let expected = reference_offline(&pack_params, &packing_keys, &inputs);
    let actual = packing_offline(&pack_params, &packing_keys, &inputs, &ctx);

    assert_poly_eq(&format!("{label} a_hat"), &actual.a_hat, &expected.a_hat);
    assert_eq!(actual.bold_t.len(), expected.bold_t.len(), "{label} rows");
    assert_eq!(
        actual.bold_t_ntt.len(),
        expected.bold_t.len(),
        "{label} NTT rows"
    );
    for (row, (actual_row, expected_row)) in actual.bold_t.iter().zip(&expected.bold_t).enumerate()
    {
        assert_eq!(actual_row.len(), expected_row.len(), "{label} row {row}");
        assert_eq!(actual.bold_t_ntt[row].len(), expected_row.len());
        for (digit, (actual_digit, expected_digit)) in
            actual_row.iter().zip(expected_row).enumerate()
        {
            assert_poly_eq(
                &format!("{label} bold_t[{row}][{digit}]"),
                actual_digit,
                expected_digit,
            );
            let mut expected_ntt = expected_digit.clone();
            expected_ntt.to_ntt(&ctx);
            assert_poly_eq(
                &format!("{label} bold_t_ntt[{row}][{digit}]"),
                &actual.bold_t_ntt[row][digit],
                &expected_ntt,
            );
        }
    }
    assert_eq!(actual.num_to_pack, PACKING_WIDTH);
    assert_eq!(actual.ring_dim, RING_DIM);
    assert_eq!(actual.q, params.q);
    assert!(actual.bold_t_bar.is_empty());
    assert!(actual.bold_t_hat.is_empty());
}

#[test]
fn packing_offline_matches_the_derived_automorphism_identity_single_prime() {
    assert_oracle_case(
        "single-prime",
        &params_with_moduli(vec![1_152_921_504_606_830_593]),
    );
}

#[test]
fn packing_offline_matches_the_derived_automorphism_identity_two_crt() {
    assert_oracle_case(
        "two-crt",
        &params_with_moduli(DEFAULT_Q_2CRT_30BIT.to_vec()),
    );
}
