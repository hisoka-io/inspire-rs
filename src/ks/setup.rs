//! Key-switching matrix generation.

use crate::math::{GaussianSampler, NttContext, Poly};
use crate::rgsw::GadgetVector;
use crate::rlwe::{apply_automorphism, RlweCiphertext, RlweSecretKey};
use serde::{Deserialize, Serialize};

fn sample_error_poly(dim: usize, moduli: &[u64], sampler: &mut GaussianSampler) -> Poly {
    Poly::sample_gaussian_moduli(dim, moduli, sampler)
}

/// Rows `K[i] = RLWE_{s'}(s * z^i)` carrying s from key s to key s'.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeySwitchingMatrix {
    /// One row per gadget power.
    pub rows: Vec<RlweCiphertext>,
    /// Decomposition base and length.
    pub gadget: GadgetVector,
}

impl KeySwitchingMatrix {
    /// Pairs rows with the gadget they were built against.
    pub fn from_rows(rows: Vec<RlweCiphertext>, gadget: GadgetVector) -> Self {
        debug_assert_eq!(
            rows.len(),
            gadget.len,
            "KS matrix must have gadget.len rows"
        );
        Self { rows, gadget }
    }

    /// Ring dimension d.
    pub fn ring_dim(&self) -> usize {
        self.rows[0].ring_dim()
    }

    /// Modulus q.
    pub fn modulus(&self) -> u64 {
        self.rows[0].modulus()
    }

    /// Gadget length.
    pub fn gadget_len(&self) -> usize {
        self.gadget.len
    }

    /// Row count, equal to the gadget length.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// True when there are no rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Row `i`.
    pub fn get_row(&self, i: usize) -> &RlweCiphertext {
        &self.rows[i]
    }

    /// All-zero rows, for shape-only tests.
    pub fn dummy(ring_dim: usize, moduli: &[u64], gadget_len: usize) -> Self {
        let q = moduli.iter().product::<u64>();
        let gadget = GadgetVector::new(1 << 20, gadget_len, q);
        let rows: Vec<RlweCiphertext> = (0..gadget_len)
            .map(|_| {
                let a = Poly::zero_moduli(ring_dim, moduli);
                let b = Poly::zero_moduli(ring_dim, moduli);
                RlweCiphertext::from_parts(a, b)
            })
            .collect();
        Self { rows, gadget }
    }
}

/// Rows `K[i] = (a_i, -a_i*s' + e_i + s*z^i)` switching `from_key` to `to_key`.
pub fn generate_ks_matrix(
    from_key: &RlweSecretKey,
    to_key: &RlweSecretKey,
    gadget: &GadgetVector,
    sampler: &mut GaussianSampler,
    ctx: &NttContext,
) -> KeySwitchingMatrix {
    let d = from_key.ring_dim();
    let ell = gadget.len;
    let powers = gadget.powers();

    debug_assert_eq!(
        from_key.ring_dim(),
        to_key.ring_dim(),
        "Keys must have same ring dimension"
    );
    debug_assert_eq!(
        from_key.modulus(),
        to_key.modulus(),
        "Keys must have same modulus"
    );

    let moduli = from_key.poly.moduli();
    let mut rows = Vec::with_capacity(ell);
    assert!(
        powers.len() >= ell,
        "gadget powers must have at least {} entries, got {}",
        ell,
        powers.len()
    );

    for &power in &powers[..ell] {
        let a = Poly::random_moduli(d, moduli);

        let error = sample_error_poly(d, moduli, sampler);

        let a_times_s_prime = a.mul_ntt(&to_key.poly, ctx);
        let neg_a_s_prime = -a_times_s_prime;

        let s_scaled = from_key.poly.scalar_mul(power);

        let b = &(&neg_a_s_prime + &error) + &s_scaled;

        rows.push(RlweCiphertext::from_parts(a, b));
    }

    KeySwitchingMatrix {
        rows,
        gadget: gadget.clone(),
    }
}

/// [`generate_ks_matrix`] from an LWE key embedded as a polynomial to an RLWE key,
/// undoing `sample_extract_coeff0` during packing.
pub fn generate_packing_ks_matrix(
    lwe_sk: &crate::lwe::LweSecretKey,
    rlwe_sk: &RlweSecretKey,
    gadget: &GadgetVector,
    sampler: &mut GaussianSampler,
    ctx: &NttContext,
) -> KeySwitchingMatrix {
    let d = rlwe_sk.ring_dim();
    let q = rlwe_sk.modulus();

    debug_assert_eq!(
        lwe_sk.dim, d,
        "LWE key dimension must match RLWE ring dimension"
    );
    debug_assert_eq!(lwe_sk.q, q, "LWE key modulus must match RLWE modulus");

    let lwe_as_rlwe = RlweSecretKey::from_poly(Poly::from_coeffs_moduli(
        lwe_sk.coeffs.clone(),
        rlwe_sk.poly.moduli(),
    ));

    generate_ks_matrix(&lwe_as_rlwe, rlwe_sk, gadget, sampler, ctx)
}

/// [`generate_ks_matrix`] from `tau_g(s)` back to s; `g` MUST be odd and coprime to 2d.
pub fn generate_automorphism_ks_matrix(
    sk: &RlweSecretKey,
    automorphism: usize,
    gadget: &GadgetVector,
    sampler: &mut GaussianSampler,
    ctx: &NttContext,
) -> KeySwitchingMatrix {
    let auto_s = RlweSecretKey::from_poly(apply_automorphism(&sk.poly, automorphism));

    generate_ks_matrix(&auto_s, sk, gadget, sampler, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::InspireParams;

    fn test_params() -> InspireParams {
        InspireParams::secure_128_d2048()
    }

    fn make_ctx(params: &InspireParams) -> NttContext {
        params.ntt_context()
    }

    #[test]
    fn test_ks_matrix_generation() {
        let params = test_params();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk1 = RlweSecretKey::generate(&params, &mut sampler);
        let sk2 = RlweSecretKey::generate(&params, &mut sampler);

        let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, params.q);
        let ks_matrix = generate_ks_matrix(&sk1, &sk2, &gadget, &mut sampler, &ctx);

        assert_eq!(ks_matrix.rows.len(), params.gadget_len);
        assert_eq!(ks_matrix.ring_dim(), params.ring_dim);
        assert_eq!(ks_matrix.modulus(), params.q);
    }

    #[test]
    fn test_ks_matrix_decryption() {
        let params = test_params();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
        let delta = params.delta();

        let sk1 = RlweSecretKey::generate(&params, &mut sampler);
        let sk2 = RlweSecretKey::generate(&params, &mut sampler);

        let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, params.q);
        let ks_matrix = generate_ks_matrix(&sk1, &sk2, &gadget, &mut sampler, &ctx);

        let powers = gadget.powers();
        for (i, row) in ks_matrix.rows.iter().enumerate() {
            let a_s2 = row.a.mul_ntt(&sk2.poly, &ctx);
            let decrypted = &a_s2 + &row.b;

            let expected = sk1.poly.scalar_mul(powers[i]);

            for j in 0..params.ring_dim {
                let dec_val = decrypted.coeff(j);
                let exp_val = expected.coeff(j);

                let diff = dec_val.abs_diff(exp_val);
                let centered_diff = std::cmp::min(diff, params.q - diff);

                assert!(
                    centered_diff < delta / 10,
                    "Row {i} coefficient {j} has large error: {centered_diff}"
                );
            }
        }
    }

    #[test]
    fn test_automorphism_ks_matrix() {
        let params = test_params();
        let ctx = make_ctx(&params);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);

        let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, params.q);

        let auto_g = 3;
        let ks_matrix = generate_automorphism_ks_matrix(&sk, auto_g, &gadget, &mut sampler, &ctx);

        assert_eq!(ks_matrix.rows.len(), params.gadget_len);
        let transformed = apply_automorphism(&sk.poly, auto_g);
        for (row, power) in ks_matrix.rows.iter().zip(gadget.powers()) {
            let decrypted = &row.a.mul_ntt(&sk.poly, &ctx) + &row.b;
            let expected = transformed.scalar_mul(power);
            for coefficient in 0..params.ring_dim {
                let difference = decrypted
                    .coeff(coefficient)
                    .abs_diff(expected.coeff(coefficient));
                let centered = difference.min(params.q - difference);
                assert!(centered < params.delta() / 10);
            }
        }
    }
}
