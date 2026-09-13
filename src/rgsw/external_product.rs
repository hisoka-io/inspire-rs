//! External product for a trivial RLWE input and one-sided RGSW ciphertext.

use crate::math::mod_q::DEFAULT_Q;
use crate::math::{NttContext, Poly};
use crate::rlwe::RlweCiphertext;

use super::types::{GadgetVector, RgswCiphertext};

/// Rejection from a one-sided RGSW external product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalProductError {
    /// The left input carries a nonzero `a` component that needs the removed RGSW half.
    NonTrivialInput,
}

impl std::fmt::Display for ExternalProductError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonTrivialInput => f.write_str(
                "one-sided RGSW external product refuses a nonzero a component; construct the \
                 left input with RlweCiphertext::trivial_encrypt",
            ),
        }
    }
}

impl std::error::Error for ExternalProductError {}

/// Splits each coefficient into `ell` base-z digits, each in `[0, z)`.
pub fn gadget_decompose(poly: &Poly, gadget: &GadgetVector) -> Vec<Poly> {
    // Single-prime only: under 2-CRT `coeff`/`set_coeff` compose and re-split
    // across limbs, so direct slice access would decompose the wrong value.
    if gadget.base == (1u64 << 20)
        && gadget.len == 3
        && poly.moduli().len() == 1
        && poly.moduli()[0] == DEFAULT_Q
    {
        return gadget_decompose_default_q(poly);
    }

    let d = poly.dimension();
    let base = gadget.base;
    let ell = gadget.len;

    let mut result = Vec::with_capacity(ell);
    for _ in 0..ell {
        result.push(Poly::zero_moduli(d, poly.moduli()));
    }

    for j in 0..d {
        let mut val = poly.coeff(j);

        for result_poly in &mut result {
            let digit = val % base;
            result_poly.set_coeff(j, digit);
            val /= base;
        }
    }

    result
}

/// [`gadget_decompose`] specialised to `(base = 2^20, len = 3)` on single-prime DEFAULT_Q.
#[inline]
fn gadget_decompose_default_q(poly: &Poly) -> Vec<Poly> {
    const BASE_MASK: u64 = (1u64 << 20) - 1;

    let moduli = poly.moduli();
    debug_assert_eq!(
        moduli.len(),
        1,
        "gadget_decompose_default_q: caller must gate on single-prime moduli"
    );
    debug_assert_eq!(
        moduli[0], DEFAULT_Q,
        "gadget_decompose_default_q: caller must gate on DEFAULT_Q"
    );

    let d = poly.dimension();
    let src = poly.coeffs();

    let mut digit0 = vec![0u64; d];
    let mut digit1 = vec![0u64; d];
    let mut digit2 = vec![0u64; d];

    for j in 0..d {
        let val = src[j];
        digit0[j] = val & BASE_MASK;
        digit1[j] = (val >> 20) & BASE_MASK;
        digit2[j] = (val >> 40) & BASE_MASK;
    }

    // Digits are masked below 2^20 < DEFAULT_Q, so the reduce pass is redundant.
    vec![
        Poly::from_crt_coeffs_reduced(digit0, moduli),
        Poly::from_crt_coeffs_reduced(digit1, moduli),
        Poly::from_crt_coeffs_reduced(digit2, moduli),
    ]
}

/// Inverse of [`gadget_decompose`]: `sum_i p_i * z^i`.
pub fn gadget_reconstruct(decomposed: &[Poly], gadget: &GadgetVector) -> Poly {
    assert!(!decomposed.is_empty(), "Decomposition cannot be empty");
    assert_eq!(
        decomposed.len(),
        gadget.len,
        "Decomposition length must match gadget length"
    );

    let d = decomposed[0].dimension();
    let moduli = decomposed[0].moduli();
    for (idx, poly) in decomposed.iter().enumerate() {
        assert_eq!(
            poly.dimension(),
            d,
            "Decomposed poly[{idx}] has mismatched dimension"
        );
        assert_eq!(
            poly.moduli(),
            moduli,
            "Decomposed poly[{idx}] has mismatched moduli"
        );
    }
    let powers = gadget.powers();

    let mut result = Poly::zero_moduli(d, moduli);

    for (i, poly) in decomposed.iter().enumerate() {
        let scaled = poly.scalar_mul(powers[i]);
        result += scaled;
    }

    result
}

/// `(a_ntt, b_ntt)` per RGSW row, transformed once and reused across calls.
pub type RgswRowsNtt = Vec<(Poly, Poly)>;

/// Builds the [`RgswRowsNtt`] consumed by [`external_product_with_ntt_rgsw`].
pub fn rgsw_rows_to_ntt(rgsw: &RgswCiphertext, ctx: &NttContext) -> RgswRowsNtt {
    rgsw.rows
        .iter()
        .map(|row| {
            let mut a_ntt = row.a.clone();
            if !a_ntt.is_ntt() {
                a_ntt.to_ntt(ctx);
            }
            let mut b_ntt = row.b.clone();
            if !b_ntt.is_ntt() {
                b_ntt.to_ntt(ctx);
            }
            (a_ntt, b_ntt)
        })
        .collect()
}

/// [`external_product`] against pre-transformed rows, byte-identical to it.
///
/// Accumulating in NTT domain transforms each surviving row once and defers the
/// inverse transform to the final ciphertext pair.
pub fn external_product_with_ntt_rgsw(
    rlwe: &RlweCiphertext,
    rgsw_ntt: &[(Poly, Poly)],
    gadget: &GadgetVector,
    ctx: &NttContext,
) -> Result<RlweCiphertext, ExternalProductError> {
    if !rlwe.a.is_zero() {
        return Err(ExternalProductError::NonTrivialInput);
    }
    Ok(external_product_trivial_with_ntt_rgsw(
        &rlwe.b, rgsw_ntt, gadget, ctx,
    ))
}

pub(crate) fn external_product_trivial_with_ntt_rgsw(
    b: &Poly,
    rgsw_ntt: &[(Poly, Poly)],
    gadget: &GadgetVector,
    ctx: &NttContext,
) -> RlweCiphertext {
    let d = b.dimension();
    let moduli = b.moduli();
    let ell = gadget.len;
    assert_eq!(
        rgsw_ntt.len(),
        ell,
        "one-sided RGSW NTT rows must have gadget.len entries"
    );
    assert_eq!(
        ctx.moduli(),
        moduli,
        "NTT context moduli must match ciphertext moduli"
    );

    let b_decomp = gadget_decompose(b, gadget);

    let b_decomp_ntt: Vec<Poly> = b_decomp
        .into_iter()
        .map(|mut p| {
            p.to_ntt(ctx);
            p
        })
        .collect();

    let mut result_a = Poly::zero_moduli(d, moduli);
    let mut result_b = Poly::zero_moduli(d, moduli);
    result_a.to_ntt(ctx);
    result_b.to_ntt(ctx);

    for i in 0..ell {
        let (row_a_ntt, row_b_ntt) = &rgsw_ntt[i];
        result_a.mul_acc_ntt_domain(&b_decomp_ntt[i], row_a_ntt, ctx);
        result_b.mul_acc_ntt_domain(&b_decomp_ntt[i], row_b_ntt, ctx);
    }

    result_a.from_ntt(ctx);
    result_b.from_ntt(ctx);

    RlweCiphertext::from_parts(result_a, result_b)
}

/// `RLWE(0, m0) x RGSW(m1) -> RLWE(m0*m1)`: decompose `b` and sum its rows.
pub fn external_product(
    rlwe: &RlweCiphertext,
    rgsw: &RgswCiphertext,
    ctx: &NttContext,
) -> Result<RlweCiphertext, ExternalProductError> {
    if !rlwe.a.is_zero() {
        return Err(ExternalProductError::NonTrivialInput);
    }
    Ok(external_product_trivial(&rlwe.b, rgsw, ctx))
}

pub(crate) fn external_product_trivial(
    b: &Poly,
    rgsw: &RgswCiphertext,
    ctx: &NttContext,
) -> RlweCiphertext {
    let d = b.dimension();
    let moduli = b.moduli();
    let gadget = &rgsw.gadget;
    let ell = gadget.len;
    assert_eq!(
        ctx.moduli(),
        moduli,
        "NTT context moduli must match ciphertext moduli"
    );
    assert_eq!(
        rgsw.rows.len(),
        ell,
        "one-sided RGSW must have gadget.len rows"
    );
    for (idx, row) in rgsw.rows.iter().enumerate() {
        assert_eq!(
            row.ring_dim(),
            d,
            "RGSW row[{idx}] has mismatched ring dimension"
        );
        assert_eq!(
            row.a.moduli(),
            moduli,
            "RGSW row[{idx}] moduli mismatch in a component"
        );
        assert_eq!(
            row.b.moduli(),
            moduli,
            "RGSW row[{idx}] moduli mismatch in b component"
        );
    }

    let b_decomp = gadget_decompose(b, gadget);

    let mut result_a = Poly::zero_moduli(d, moduli);
    let mut result_b = Poly::zero_moduli(d, moduli);

    for (digit, row) in b_decomp.iter().zip(&rgsw.rows) {
        result_a += digit.mul_ntt(&row.a, ctx);
        result_b += digit.mul_ntt(&row.b, ctx);
    }

    RlweCiphertext::from_parts(result_a, result_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::GaussianSampler;
    use crate::params::InspireParams;
    use crate::rlwe::RlweSecretKey;

    fn test_params() -> InspireParams {
        InspireParams::secure_128_d2048()
    }

    fn make_ctx(params: &InspireParams) -> NttContext {
        params.ntt_context()
    }

    #[test]
    fn test_gadget_decompose_reconstruct_roundtrip() {
        let params = test_params();
        let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, params.q);

        let poly = Poly::random_moduli(params.ring_dim, params.moduli());

        let decomposed = gadget_decompose(&poly, &gadget);
        let reconstructed = gadget_reconstruct(&decomposed, &gadget);

        assert_eq!(poly, reconstructed);
    }

    #[test]
    fn test_gadget_decompose_small_digits() {
        let params = test_params();
        let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, params.q);

        let poly = Poly::random_moduli(params.ring_dim, params.moduli());
        let decomposed = gadget_decompose(&poly, &gadget);

        for digit_poly in &decomposed {
            for j in 0..params.ring_dim {
                let coeff = digit_poly.coeff(j);
                assert!(
                    coeff < params.gadget_base,
                    "Digit {} exceeds base {}",
                    coeff,
                    params.gadget_base
                );
            }
        }
    }

    #[test]
    fn test_gadget_decompose_zero() {
        let params = test_params();
        let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, params.q);

        let zero = Poly::zero_moduli(params.ring_dim, params.moduli());
        let decomposed = gadget_decompose(&zero, &gadget);

        for digit_poly in &decomposed {
            assert!(digit_poly.is_zero());
        }
    }

    #[test]
    fn test_external_product_by_zero() {
        let params = test_params();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
        let delta = params.delta();

        let sk = RlweSecretKey::generate(&params, &mut sampler);
        let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, params.q);

        let msg_coeffs: Vec<u64> = (0..params.ring_dim)
            .map(|i| (i as u64) % params.p)
            .collect();
        let msg = Poly::from_coeffs_moduli(msg_coeffs, params.moduli());
        let rlwe = RlweCiphertext::trivial_encrypt(&msg, delta, &params);

        let rgsw_zero =
            super::super::RgswCiphertext::encrypt_scalar(&sk, 0, &gadget, &mut sampler, &ctx);

        let result = external_product(&rlwe, &rgsw_zero, &ctx).expect("trivial RLWE input");
        let decrypted = result.decrypt(&sk, delta, params.p, &ctx);

        for i in 0..params.ring_dim {
            assert_eq!(decrypted.coeff(i), 0, "Expected 0 at coefficient {i}");
        }
    }

    #[test]
    fn test_external_product_by_one() {
        let params = test_params();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
        let delta = params.delta();

        let sk = RlweSecretKey::generate(&params, &mut sampler);
        let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, params.q);

        let msg_coeffs: Vec<u64> = (0..params.ring_dim)
            .map(|i| (i as u64) % params.p)
            .collect();
        let msg = Poly::from_coeffs_moduli(msg_coeffs.clone(), params.moduli());
        let rlwe = RlweCiphertext::trivial_encrypt(&msg, delta, &params);

        let rgsw_one =
            super::super::RgswCiphertext::encrypt_scalar(&sk, 1, &gadget, &mut sampler, &ctx);

        let result = external_product(&rlwe, &rgsw_one, &ctx).expect("trivial RLWE input");
        let decrypted = result.decrypt(&sk, delta, params.p, &ctx);

        for (i, expected) in msg_coeffs.iter().enumerate().take(params.ring_dim) {
            assert_eq!(decrypted.coeff(i), *expected, "Mismatch at coefficient {i}");
        }
    }

    #[test]
    fn test_external_product_scalar_multiplication() {
        let params = test_params();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
        let delta = params.delta();

        let sk = RlweSecretKey::generate(&params, &mut sampler);
        let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, params.q);

        let msg_coeffs: Vec<u64> = (0..params.ring_dim).map(|i| (i as u64) % 10).collect();
        let msg = Poly::from_coeffs_moduli(msg_coeffs.clone(), params.moduli());
        let rlwe = RlweCiphertext::trivial_encrypt(&msg, delta, &params);

        let scalar = 3u64;
        let rgsw_scalar =
            super::super::RgswCiphertext::encrypt_scalar(&sk, scalar, &gadget, &mut sampler, &ctx);

        let result = external_product(&rlwe, &rgsw_scalar, &ctx).expect("trivial RLWE input");
        let decrypted = result.decrypt(&sk, delta, params.p, &ctx);

        for (i, msg_coeff) in msg_coeffs.iter().enumerate().take(params.ring_dim) {
            let expected = (*msg_coeff * scalar) % params.p;
            assert_eq!(
                decrypted.coeff(i),
                expected,
                "Mismatch at coefficient {}: expected {}, got {}",
                i,
                expected,
                decrypted.coeff(i)
            );
        }
    }

    #[test]
    fn test_external_product_monomial() {
        let params = test_params();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
        let delta = params.delta();

        let sk = RlweSecretKey::generate(&params, &mut sampler);
        let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, params.q);

        let mut msg_coeffs = vec![0u64; params.ring_dim];
        msg_coeffs[0] = 5;
        let msg = Poly::from_coeffs_moduli(msg_coeffs, params.moduli());
        let rlwe = RlweCiphertext::trivial_encrypt(&msg, delta, &params);

        let mut monomial_coeffs = vec![0u64; params.ring_dim];
        monomial_coeffs[1] = 1;
        let monomial = Poly::from_coeffs_moduli(monomial_coeffs, params.moduli());
        let rgsw_mono =
            super::super::RgswCiphertext::encrypt(&sk, &monomial, &gadget, &mut sampler, &ctx);

        let result = external_product(&rlwe, &rgsw_mono, &ctx).expect("trivial RLWE input");
        let decrypted = result.decrypt(&sk, delta, params.p, &ctx);

        assert_eq!(decrypted.coeff(0), 0, "Constant term should be 0");
        assert_eq!(decrypted.coeff(1), 5, "X coefficient should be 5");
        for i in 2..params.ring_dim {
            assert_eq!(decrypted.coeff(i), 0, "Higher terms should be 0");
        }
    }
}
