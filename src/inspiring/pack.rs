//! Pack and PartialPack: transform, aggregate, collapse. Under the CRS model the `a`
//! vectors are fixed, so only `b` changes per query and the rest precomputes offline.

use crate::ks::KeySwitchingMatrix;
use crate::lwe::LweCiphertext;
use crate::math::Poly;
use crate::params::InspireParams;
use crate::rlwe::RlweCiphertext;

use super::collapse::{collapse, collapse_partial};
use super::transform::{aggregate, transform_at_slot};
use super::types::AggregatedCiphertext;

use serde::{Deserialize, Serialize};

/// Pack d LWE ciphertexts into one RLWE ciphertext carrying message i at coefficient i.
///
/// Noise bound is InsPIRe Theorem 2 (eprint 2025/1352), which holds only if K_g and
/// K_h are drawn from fresh error samples.
pub fn pack(
    lwe_ciphertexts: &[LweCiphertext],
    k_g: &KeySwitchingMatrix,
    k_h: &KeySwitchingMatrix,
    params: &InspireParams,
) -> RlweCiphertext {
    let d = params.ring_dim;
    assert_eq!(
        lwe_ciphertexts.len(),
        d,
        "Must provide exactly d ciphertexts for full packing"
    );

    let intermediates: Vec<_> = lwe_ciphertexts
        .iter()
        .enumerate()
        .map(|(i, lwe)| transform_at_slot(lwe, i, params))
        .collect();

    let aggregated = aggregate(&intermediates, params);

    collapse(&aggregated, k_g, k_h, params)
}

/// Pack gamma <= d/2 ciphertexts using only K_g.
///
/// Noise bound is InsPIRe Theorem 4 (eprint 2025/1352); skipping the conjugation
/// branch lowers it, but the outputs land on the EVEN coefficients 0, 2, ..., 2*gamma-2.
pub fn partial_pack(
    lwe_ciphertexts: &[LweCiphertext],
    k_g: &KeySwitchingMatrix,
    params: &InspireParams,
) -> RlweCiphertext {
    let gamma = lwe_ciphertexts.len();
    let d = params.ring_dim;

    assert!(
        gamma <= d / 2,
        "partial_pack requires gamma <= d/2 ciphertexts"
    );

    if gamma == 0 {
        return RlweCiphertext::from_parts(
            Poly::zero_moduli(d, params.moduli()),
            Poly::zero_moduli(d, params.moduli()),
        );
    }

    let intermediates: Vec<_> = lwe_ciphertexts
        .iter()
        .enumerate()
        .map(|(i, lwe)| transform_at_slot(lwe, i, params))
        .collect();

    let aggregated = aggregate(&intermediates, params);

    collapse_partial(gamma, &aggregated.to_intermediate(), k_g, params)
}

/// Transform + Aggregate over the CRS `a` vectors, which the online phase reuses.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackingPrecomputation {
    precomputed_a_aggregate: AggregatedCiphertext,
    num_ciphertexts: usize,
    ring_dim: usize,
    q: u64,
    moduli: Vec<u64>,
}

impl PackingPrecomputation {
    /// Ciphertext count this precomputation was built for.
    pub fn num_ciphertexts(&self) -> usize {
        self.num_ciphertexts
    }
}

/// Precompute everything over the CRS `a` vectors that does not depend on `b`.
pub fn precompute_packing(
    crs_a_vectors: &[Vec<u64>],
    _k_g: &KeySwitchingMatrix,
    _k_h: &KeySwitchingMatrix,
    params: &InspireParams,
) -> PackingPrecomputation {
    let d = params.ring_dim;
    let q = params.q;
    let moduli = params.moduli().to_vec();
    let n = crs_a_vectors.len();

    assert!(!crs_a_vectors.is_empty(), "Must have at least one a vector");
    assert_eq!(crs_a_vectors[0].len(), d, "a vectors must have dimension d");

    // b=0: only the a-component transformation matters here.
    let dummy_lwes: Vec<LweCiphertext> = crs_a_vectors
        .iter()
        .map(|a| LweCiphertext {
            a: a.clone(),
            b: 0,
            q,
        })
        .collect();

    let intermediates: Vec<_> = dummy_lwes
        .iter()
        .enumerate()
        .map(|(i, lwe)| transform_at_slot(lwe, i, params))
        .collect();

    let aggregated = aggregate(&intermediates, params);

    PackingPrecomputation {
        precomputed_a_aggregate: aggregated,
        num_ciphertexts: n,
        ring_dim: d,
        q,
        moduli,
    }
}

/// Online phase: pack from the `b` values alone against a `precompute_packing` result.
pub fn pack_online(
    lwe_b_values: &[u64],
    precomp: &PackingPrecomputation,
    k_g: &KeySwitchingMatrix,
    k_h: &KeySwitchingMatrix,
    params: &InspireParams,
) -> RlweCiphertext {
    let d = precomp.ring_dim;
    let moduli = &precomp.moduli;
    let n = lwe_b_values.len();

    assert_eq!(
        n, precomp.num_ciphertexts,
        "Number of b values must match precomputation"
    );

    let mut b_coeffs = vec![0u64; d];
    for (i, &b_val) in lwe_b_values.iter().enumerate() {
        if i < d {
            b_coeffs[i] = b_val;
        }
    }
    let b_poly = Poly::from_coeffs_moduli(b_coeffs, moduli);

    let full_aggregate = AggregatedCiphertext::new(
        precomp.precomputed_a_aggregate.a_polys.clone(),
        &precomp.precomputed_a_aggregate.b_poly + &b_poly,
    );

    collapse(&full_aggregate, k_g, k_h, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lwe::LweSecretKey;
    use crate::math::GaussianSampler;
    use rand::Rng;
    use rand::SeedableRng;

    fn test_params() -> InspireParams {
        InspireParams {
            ring_dim: 256,
            q: 1152921504606830593,
            crt_moduli: vec![1152921504606830593],
            p: 65536,
            sigma: 6.4,
            gadget_base: 1 << 20,
            query_gadget_len: 3,
            packing_gadget_len: 3,
            security_level: crate::params::SecurityLevel::Bits128,
        }
    }

    fn random_lwe<R: Rng>(rng: &mut R, params: &InspireParams) -> LweCiphertext {
        let a: Vec<u64> = (0..params.ring_dim)
            .map(|_| rng.gen_range(0..params.q))
            .collect();
        let b = rng.gen_range(0..params.q);
        LweCiphertext { a, b, q: params.q }
    }

    fn encrypt_lwe<R: Rng>(
        sk: &LweSecretKey,
        message: u64,
        rng: &mut R,
        params: &InspireParams,
    ) -> LweCiphertext {
        let a: Vec<u64> = (0..params.ring_dim)
            .map(|_| rng.gen_range(0..params.q))
            .collect();
        let error = (rng.next_u32() % 7) as i64 - 3;
        LweCiphertext::encrypt(sk, message, params.delta(), a, error)
    }

    #[test]
    fn test_pack_dimensions() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(12345);

        let lwe_cts: Vec<LweCiphertext> = (0..params.ring_dim)
            .map(|_| random_lwe(&mut rng, &params))
            .collect();

        let k_g =
            KeySwitchingMatrix::dummy(params.ring_dim, params.moduli(), params.packing_gadget_len);
        let k_h =
            KeySwitchingMatrix::dummy(params.ring_dim, params.moduli(), params.packing_gadget_len);

        let result = pack(&lwe_cts, &k_g, &k_h, &params);

        assert_eq!(result.ring_dim(), params.ring_dim);
        assert_eq!(result.modulus(), params.q);
    }

    #[test]
    fn test_partial_pack_dimensions() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(54321);

        let gamma = params.ring_dim / 4;
        let lwe_cts: Vec<LweCiphertext> =
            (0..gamma).map(|_| random_lwe(&mut rng, &params)).collect();

        let k_g =
            KeySwitchingMatrix::dummy(params.ring_dim, params.moduli(), params.packing_gadget_len);

        let result = partial_pack(&lwe_cts, &k_g, &params);

        assert_eq!(result.ring_dim(), params.ring_dim);
    }

    #[test]
    fn test_precompute_pack_online() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(98765);

        let n = 16;
        let crs_a_vectors: Vec<Vec<u64>> = (0..n)
            .map(|_| {
                (0..params.ring_dim)
                    .map(|_| rng.gen_range(0..params.q))
                    .collect()
            })
            .collect();

        let k_g =
            KeySwitchingMatrix::dummy(params.ring_dim, params.moduli(), params.packing_gadget_len);
        let k_h =
            KeySwitchingMatrix::dummy(params.ring_dim, params.moduli(), params.packing_gadget_len);

        let precomp = precompute_packing(&crs_a_vectors, &k_g, &k_h, &params);
        assert_eq!(precomp.num_ciphertexts(), n);

        let b_values: Vec<u64> = (0..n).map(|_| rng.gen_range(0..params.q)).collect();
        let result = pack_online(&b_values, &precomp, &k_g, &k_h, &params);

        assert_eq!(result.ring_dim(), params.ring_dim);
    }

    #[test]
    fn test_pack_with_real_encryption() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(11111);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let lwe_sk = LweSecretKey::generate(params.ring_dim, params.q, &mut sampler);

        let messages: Vec<u64> = (0..params.ring_dim)
            .map(|i| (i as u64 * 7) % params.p)
            .collect();

        let lwe_cts: Vec<LweCiphertext> = messages
            .iter()
            .map(|&m| encrypt_lwe(&lwe_sk, m, &mut rng, &params))
            .collect();

        for (ct, &expected) in lwe_cts.iter().zip(messages.iter()) {
            let decrypted = ct.decrypt(&lwe_sk, params.delta(), params.p);
            assert_eq!(decrypted, expected, "LWE decryption failed");
        }

        let k_g =
            KeySwitchingMatrix::dummy(params.ring_dim, params.moduli(), params.packing_gadget_len);
        let k_h =
            KeySwitchingMatrix::dummy(params.ring_dim, params.moduli(), params.packing_gadget_len);

        let packed = pack(&lwe_cts, &k_g, &k_h, &params);

        // Dummy key-switching matrices, so only the shape is checkable here.
        assert_eq!(packed.ring_dim(), params.ring_dim);
    }

    #[test]
    fn test_empty_partial_pack() {
        let params = test_params();
        let k_g =
            KeySwitchingMatrix::dummy(params.ring_dim, params.moduli(), params.packing_gadget_len);

        let result = partial_pack(&[], &k_g, &params);

        assert_eq!(result.ring_dim(), params.ring_dim);
        for i in 0..params.ring_dim {
            assert_eq!(result.a.coeff(i), 0);
            assert_eq!(result.b.coeff(i), 0);
        }
    }

    #[test]
    fn test_aggregate_properties() {
        let params = test_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(33333);

        let n = 4;
        let lwe_cts: Vec<LweCiphertext> = (0..n).map(|_| random_lwe(&mut rng, &params)).collect();

        let intermediates: Vec<_> = lwe_cts
            .iter()
            .enumerate()
            .map(|(i, lwe)| transform_at_slot(lwe, i, &params))
            .collect();

        let aggregated = aggregate(&intermediates, &params);

        for (i, ct) in lwe_cts.iter().enumerate() {
            assert_eq!(
                aggregated.b_poly.coeff(i),
                ct.b,
                "b coefficient mismatch at position {i}"
            );
        }
    }
}
