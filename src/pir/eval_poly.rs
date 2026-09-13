//! Horner-method homomorphic polynomial evaluation, for schemes that need it. The
//! shipping PIR flow uses direct coefficient encoding with monomial rotation instead,
//! so nothing here is on a live path.

use crate::math::{NttContext, Poly};
use crate::params::InspireParams;
use crate::rgsw::{external_product, ExternalProductError, GadgetVector, RgswCiphertext};
use crate::rlwe::RlweCiphertext;

/// Evaluate plaintext h at an RGSW-encrypted point via Horner and external products.
#[allow(dead_code)]
pub fn eval_poly_homomorphic(
    poly_coeffs: &Poly,
    encrypted_point: &RgswCiphertext,
    params: &InspireParams,
) -> Result<RlweCiphertext, ExternalProductError> {
    let ctx = params.ntt_context();

    let degree = find_degree(poly_coeffs);

    if degree == 0 {
        let const_coeff = poly_coeffs.coeff(0);
        return Ok(encrypt_constant(const_coeff, params));
    }

    let delta = params.delta();
    let mut acc = encrypt_scaled_constant(poly_coeffs.coeff(degree), delta, params);

    for i in (0..degree).rev() {
        let product = external_product(&acc, encrypted_point, &ctx)?;

        let h_i = poly_coeffs.coeff(i);
        let h_i_scaled = encrypt_scaled_constant(h_i, delta, params);

        acc = product.add(&h_i_scaled);
    }

    Ok(acc)
}

/// `eval_poly_homomorphic` against a caller-owned NTT context.
#[allow(dead_code)]
pub fn eval_poly_homomorphic_with_ctx(
    poly_coeffs: &Poly,
    encrypted_point: &RgswCiphertext,
    params: &InspireParams,
    ctx: &NttContext,
) -> Result<RlweCiphertext, ExternalProductError> {
    let degree = find_degree(poly_coeffs);

    if degree == 0 {
        let const_coeff = poly_coeffs.coeff(0);
        return Ok(encrypt_constant(const_coeff, params));
    }

    let delta = params.delta();
    let mut acc = encrypt_scaled_constant(poly_coeffs.coeff(degree), delta, params);

    for i in (0..degree).rev() {
        let product = external_product(&acc, encrypted_point, ctx)?;

        let h_i = poly_coeffs.coeff(i);
        let h_i_scaled = encrypt_scaled_constant(h_i, delta, params);

        acc = product.add(&h_i_scaled);
    }

    Ok(acc)
}

/// Index of the highest non-zero coefficient.
fn find_degree(poly: &Poly) -> usize {
    for i in (0..poly.len()).rev() {
        if poly.coeff(i) != 0 {
            return i;
        }
    }
    0
}

/// Noiseless trivial encryption of a delta-scaled constant.
fn encrypt_constant(value: u64, params: &InspireParams) -> RlweCiphertext {
    let d = params.ring_dim;
    let q = params.q;
    let delta = params.delta();

    let a = Poly::zero_moduli(d, params.moduli());
    let mut b_coeffs = vec![0u64; d];
    b_coeffs[0] = ((value as u128 * delta as u128) % q as u128) as u64;
    let b = Poly::from_coeffs_moduli(b_coeffs, params.moduli());

    RlweCiphertext::from_parts(a, b)
}

/// Noiseless trivial encryption against a caller-supplied delta.
fn encrypt_scaled_constant(value: u64, delta: u64, params: &InspireParams) -> RlweCiphertext {
    let d = params.ring_dim;
    let q = params.q;

    let a = Poly::zero_moduli(d, params.moduli());
    let mut b_coeffs = vec![0u64; d];
    b_coeffs[0] = ((value as u128 * delta as u128) % q as u128) as u64;
    let b = Poly::from_coeffs_moduli(b_coeffs, params.moduli());

    RlweCiphertext::from_parts(a, b)
}

/// The t-th roots of unity [1, omega, ..., omega^(t-1)] in Z_q; t must divide q-1.
#[allow(dead_code)]
pub fn generate_scalar_eval_points(t: usize, q: u64) -> Vec<u64> {
    if t == 0 {
        return vec![];
    }

    let omega = find_primitive_root(t, q);

    let mut points = Vec::with_capacity(t);
    let mut current = 1u64;
    for _ in 0..t {
        points.push(current);
        current = ((current as u128 * omega as u128) % q as u128) as u64;
    }
    points
}

/// Lift a scalar evaluation point into a constant polynomial.
#[allow(dead_code)]
pub fn scalar_eval_point_to_poly(value: u64, d: usize, moduli: &[u64]) -> Poly {
    let mut coeffs = vec![0u64; d];
    coeffs[0] = value;
    Poly::from_coeffs_moduli(coeffs, moduli)
}

/// RGSW-encrypt a scalar evaluation point.
#[allow(dead_code)]
pub fn encrypt_scalar_eval_point(
    value: u64,
    sk: &crate::rlwe::RlweSecretKey,
    gadget: &GadgetVector,
    sampler: &mut crate::math::GaussianSampler,
    ctx: &NttContext,
) -> RgswCiphertext {
    let d = sk.ring_dim();
    let moduli = sk.poly.moduli();

    let point_poly = scalar_eval_point_to_poly(value, d, moduli);
    RgswCiphertext::encrypt(sk, &point_poly, gadget, sampler, ctx)
}

/// Primitive n-th root of unity mod q.
#[allow(
    clippy::panic,
    reason = "module is entirely dead_code; no shipping PIR flow reaches it"
)]
fn find_primitive_root(n: usize, q: u64) -> u64 {
    if n == 1 {
        return 1;
    }

    assert!(
        (q - 1).is_multiple_of(n as u64),
        "No {n}-th root of unity exists mod {q}"
    );

    let exp = (q - 1) / n as u64;

    for candidate in 2..1000.min(q) {
        let root = mod_pow(candidate, exp, q);
        if is_primitive_root(root, n, q) {
            return root;
        }
    }
    panic!("No primitive root found in first 1000 candidates (legacy eval_poly)");
}

fn is_primitive_root(g: u64, n: usize, q: u64) -> bool {
    if g == 0 || g == 1 {
        return false;
    }
    if mod_pow(g, n as u64, q) != 1 {
        return false;
    }
    if n > 1 && mod_pow(g, (n / 2) as u64, q) == 1 {
        return false;
    }
    true
}

fn mod_pow(base: u64, exp: u64, modulus: u64) -> u64 {
    let mut result = 1u128;
    let mut base = base as u128;
    let mut exp = exp;
    let m = modulus as u128;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % m;
        }
        base = (base * base) % m;
        exp >>= 1;
    }
    result as u64
}

/// Evaluation points as (index, negate) pairs for z_k = +/-X^index; needs t | 2d,
/// since X^(2d) = 1 in the negacyclic ring.
#[deprecated(note = "Use generate_scalar_eval_points for correct NTT domain alignment")]
#[allow(dead_code)]
pub fn generate_monomial_eval_points(t: usize, d: usize) -> Vec<(usize, bool)> {
    if t == 0 {
        return vec![];
    }

    assert!(
        (2 * d).is_multiple_of(t),
        "t must divide 2d for roots of unity to exist"
    );

    let step = (2 * d) / t;
    let mut points = Vec::with_capacity(t);

    for k in 0..t {
        let power = (k * step) % (2 * d);

        if power < d {
            points.push((power, false));
        } else {
            points.push((power - d, true));
        }
    }

    points
}

#[deprecated(note = "Use generate_scalar_eval_points for correct NTT domain alignment")]
#[allow(deprecated)]
#[allow(dead_code)]
pub fn generate_eval_points(t: usize, d: usize) -> Vec<(usize, bool)> {
    generate_monomial_eval_points(t, d)
}

/// Monomial +/-X^index as a polynomial.
#[deprecated(note = "Use scalar_eval_point_to_poly for correct NTT domain alignment")]
#[allow(dead_code)]
pub fn eval_point_to_poly(index: usize, negate: bool, d: usize, q: u64, moduli: &[u64]) -> Poly {
    let mut coeffs = vec![0u64; d];

    if negate {
        coeffs[index] = q - 1;
    } else {
        coeffs[index] = 1;
    }

    Poly::from_coeffs_moduli(coeffs, moduli)
}

/// RGSW-encrypt the monomial evaluation point +/-X^index.
#[deprecated(note = "Use encrypt_scalar_eval_point for correct NTT domain alignment")]
#[allow(deprecated)]
#[allow(dead_code)]
pub fn encrypt_eval_point(
    index: usize,
    negate: bool,
    sk: &crate::rlwe::RlweSecretKey,
    gadget: &crate::rgsw::GadgetVector,
    sampler: &mut crate::math::GaussianSampler,
    ctx: &NttContext,
) -> RgswCiphertext {
    let d = sk.ring_dim();
    let q = sk.modulus();
    let moduli = sk.poly.moduli();

    let point_poly = eval_point_to_poly(index, negate, d, q, moduli);
    RgswCiphertext::encrypt(sk, &point_poly, gadget, sampler, ctx)
}

/// Select one of `polynomials` under encrypted index bits, one tree level per bit.
#[allow(dead_code)]
pub fn homomorphic_select(
    polynomials: &[Poly],
    index_bits: &[RgswCiphertext],
    params: &InspireParams,
) -> Result<RlweCiphertext, ExternalProductError> {
    let delta = params.delta();
    let ctx = params.ntt_context();

    if polynomials.is_empty() {
        return Ok(RlweCiphertext::zero(params));
    }

    if polynomials.len() == 1 {
        return Ok(poly_to_rlwe(&polynomials[0], delta, params));
    }

    let mut current: Vec<RlweCiphertext> = polynomials
        .iter()
        .map(|p| poly_to_rlwe(p, delta, params))
        .collect();

    for bit_ct in index_bits {
        if current.len() == 1 {
            break;
        }

        let mut next = Vec::with_capacity(current.len() / 2 + current.len() % 2);

        for pair in current.chunks(2) {
            if pair.len() == 2 {
                let diff = pair[1].sub(&pair[0]);
                let selected = external_product(&diff, bit_ct, &ctx)?;
                let result = pair[0].add(&selected);
                next.push(result);
            } else {
                next.push(pair[0].clone());
            }
        }

        current = next;
    }

    Ok(current
        .into_iter()
        .next()
        .unwrap_or_else(|| RlweCiphertext::zero(params)))
}

/// Trivial (noiseless) RLWE encryption of a plaintext polynomial.
#[allow(dead_code)]
fn poly_to_rlwe(poly: &Poly, delta: u64, params: &InspireParams) -> RlweCiphertext {
    let d = params.ring_dim;
    let moduli = params.moduli();

    let a = Poly::zero_moduli(d, moduli);
    let b = poly.scalar_mul(delta);

    RlweCiphertext::from_parts(a, b)
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::math::GaussianSampler;
    use crate::rgsw::GadgetVector;
    use crate::rlwe::RlweSecretKey;

    fn test_params() -> InspireParams {
        InspireParams {
            ring_dim: 256,
            q: 1152921504606830593,
            crt_moduli: vec![1152921504606830593],
            p: 65536,
            sigma: 6.4,
            gadget_base: 1 << 20,
            gadget_len: 3,
            security_level: crate::params::SecurityLevel::Bits128,
        }
    }

    #[test]
    fn test_generate_scalar_eval_points_count() {
        let q = 1152921504606830593u64;

        for t in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
            let points = generate_scalar_eval_points(t, q);
            assert_eq!(points.len(), t, "Wrong number of eval points for t={t}");
        }
    }

    #[test]
    fn test_generate_scalar_eval_points_first_is_one() {
        let q = 1152921504606830593u64;
        let t = 16;

        let points = generate_scalar_eval_points(t, q);

        assert_eq!(points[0], 1, "First eval point should be omega^0 = 1");
    }

    #[test]
    fn test_generate_scalar_eval_points_are_roots() {
        let q = 1152921504606830593u64;
        let t = 16;

        let points = generate_scalar_eval_points(t, q);

        for (k, &omega_k) in points.iter().enumerate() {
            let omega_k_to_t = mod_pow(omega_k, t as u64, q);
            assert_eq!(omega_k_to_t, 1, "omega^{k} raised to t={t} should be 1");
        }
    }

    #[test]
    fn test_scalar_eval_point_to_poly() {
        let d = 256;
        let q = 1152921504606830593u64;
        let value = 42u64;

        let poly = scalar_eval_point_to_poly(value, d, &[q]);

        assert_eq!(poly.coeff(0), value);
        for i in 1..d {
            assert_eq!(poly.coeff(i), 0);
        }
    }

    #[test]
    fn test_generate_eval_points_count() {
        let d = 256;

        for t in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
            let points = generate_eval_points(t, d);
            assert_eq!(points.len(), t, "Wrong number of eval points for t={t}");
        }
    }

    #[test]
    fn test_generate_eval_points_first_is_one() {
        let d = 256;
        let t = 16;

        let points = generate_eval_points(t, d);

        assert_eq!(points[0], (0, false), "First eval point should be X^0 = 1");
    }

    #[test]
    fn test_eval_point_to_poly_positive() {
        let d = 256;
        let q = 1152921504606830593u64;

        let poly = eval_point_to_poly(5, false, d, q, &[q]);

        assert_eq!(poly.coeff(5), 1);
        for i in 0..d {
            if i != 5 {
                assert_eq!(poly.coeff(i), 0);
            }
        }
    }

    #[test]
    fn test_eval_point_to_poly_negative() {
        let d = 256;
        let q = 1152921504606830593u64;

        let poly = eval_point_to_poly(3, true, d, q, &[q]);

        assert_eq!(poly.coeff(3), q - 1);
        for i in 0..d {
            if i != 3 {
                assert_eq!(poly.coeff(i), 0);
            }
        }
    }

    #[test]
    fn test_find_degree() {
        let params = test_params();
        let d = params.ring_dim;

        let mut coeffs = vec![0u64; d];
        coeffs[0] = 1;
        coeffs[5] = 2;
        coeffs[10] = 3;
        let poly = Poly::from_coeffs_moduli(coeffs, params.moduli());

        assert_eq!(find_degree(&poly), 10);
    }

    #[test]
    fn test_find_degree_constant() {
        let params = test_params();
        let d = params.ring_dim;

        let mut coeffs = vec![0u64; d];
        coeffs[0] = 42;
        let poly = Poly::from_coeffs_moduli(coeffs, params.moduli());

        assert_eq!(find_degree(&poly), 0);
    }

    #[test]
    fn test_find_degree_zero() {
        let params = test_params();
        let d = params.ring_dim;
        let poly = Poly::zero_moduli(d, params.moduli());

        assert_eq!(find_degree(&poly), 0);
    }

    #[test]
    fn test_encrypt_constant_decrypts_correctly() {
        let params = test_params();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
        let ctx = params.ntt_context();

        let sk = RlweSecretKey::generate(&params, &mut sampler);
        let value = 42u64;
        let delta = params.delta();

        let ct = encrypt_constant(value, &params);
        let decrypted = ct.decrypt(&sk, delta, params.p, &ctx);

        assert_eq!(decrypted.coeff(0), value);
    }

    #[test]
    fn test_eval_poly_homomorphic_constant() {
        let params = test_params();
        let d = params.ring_dim;
        let q = params.q;
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
        let ctx = params.ntt_context();

        let sk = RlweSecretKey::generate(&params, &mut sampler);
        let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, q);

        let mut coeffs = vec![0u64; d];
        coeffs[0] = 100;
        let poly = Poly::from_coeffs_moduli(coeffs, params.moduli());

        let point_poly = eval_point_to_poly(0, false, d, q, params.moduli());
        let encrypted_point =
            RgswCiphertext::encrypt(&sk, &point_poly, &gadget, &mut sampler, &ctx);

        let result = eval_poly_homomorphic(&poly, &encrypted_point, &params)
            .expect("constant evaluation needs no external product");
        let decrypted = result.decrypt(&sk, params.delta(), params.p, &ctx);

        assert_eq!(decrypted.coeff(0), 100);
    }

    #[test]
    fn test_homomorphic_select_single() {
        let params = test_params();
        let d = params.ring_dim;

        let mut coeffs = vec![0u64; d];
        coeffs[0] = 42;
        let poly = Poly::from_coeffs_moduli(coeffs, params.moduli());

        let result =
            homomorphic_select(&[poly], &[], &params).expect("single input needs no select");

        assert_eq!(result.ring_dim(), d);
    }

    #[test]
    fn test_homomorphic_select_empty() {
        let params = test_params();
        let d = params.ring_dim;

        let result = homomorphic_select(&[], &[], &params).expect("empty input needs no select");

        assert_eq!(result.ring_dim(), d);
        assert!(result.a.is_zero());
        assert!(result.b.is_zero());
    }
}
