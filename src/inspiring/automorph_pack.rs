//! Galois-automorphism tree packing of many LWE ciphertexts into one RLWE.
//!
//! Shift-and-add cannot do this: a key-switched RLWE carries noise in every
//! coefficient, so summing shifted copies mixes it. Only the `a` polynomials are
//! tree-packed; the `b` values are re-added, scaled by d, at the end.

use crate::ks::KeySwitchingMatrix;
use crate::lwe::LweCiphertext;
use crate::math::{ModQ, NttContext, Poly};
use crate::params::InspireParams;
use crate::rgsw::gadget_decompose;
use crate::rlwe::{automorphism_ciphertext, RlweCiphertext};

/// Apply tau_t, then key-switch the result from tau_t(s) back to s.
pub fn homomorphic_automorph(
    ct: &RlweCiphertext,
    t: usize,
    ks_matrix: &KeySwitchingMatrix,
    ctx: &NttContext,
) -> RlweCiphertext {
    let d = ct.ring_dim();
    let moduli = ct.a.moduli();

    let ct_auto = automorphism_ciphertext(ct, t);

    let gadget = &ks_matrix.gadget;
    let a_decomp = gadget_decompose(&ct_auto.a, gadget);

    let mut result_a = Poly::zero_moduli(d, moduli);
    let mut result_b = ct_auto.b.clone();

    for (i, digit_poly) in a_decomp.iter().enumerate() {
        if i < ks_matrix.len() {
            let ks_row = ks_matrix.get_row(i);
            let term_a = digit_poly.mul_ntt(&ks_row.a, ctx);
            let term_b = digit_poly.mul_ntt(&ks_row.b, ctx);
            result_a = &result_a + &term_a;
            result_b = &result_b + &term_b;
        }
    }

    RlweCiphertext::from_parts(result_a, result_b)
}

/// Per-level tree-packing constants. X is a primitive 2d-th root of unity in
/// `Z_q[X]/(X^d + 1)`, so level L uses `y = X^(d / 2^(L+1))`.
pub struct YConstants {
    /// `y[L] = X^(d / 2^(L+1))`.
    pub y_polys: Vec<Poly>,
    /// Negated `y[L]`.
    pub neg_y_polys: Vec<Poly>,
}

impl YConstants {
    /// Build constants for all log2(d) levels.
    pub fn generate(d: usize, q: u64, moduli: &[u64]) -> Self {
        let log_d = (d as f64).log2() as usize;
        let mut y_polys = Vec::with_capacity(log_d);
        let mut neg_y_polys = Vec::with_capacity(log_d);

        for ell in 0..log_d {
            let step = d >> (ell + 1);

            let mut y_coeffs = vec![0u64; d];
            if step < d {
                y_coeffs[step] = 1;
            }
            let y_poly = Poly::from_coeffs_moduli(y_coeffs, moduli);

            let mut neg_y_coeffs = vec![0u64; d];
            if step < d {
                neg_y_coeffs[step] = q - 1;
            }
            let neg_y_poly = Poly::from_coeffs_moduli(neg_y_coeffs, moduli);

            y_polys.push(y_poly);
            neg_y_polys.push(neg_y_poly);
        }

        Self {
            y_polys,
            neg_y_polys,
        }
    }

    /// y at `level`.
    pub fn y(&self, level: usize) -> &Poly {
        &self.y_polys[level]
    }

    /// -y at `level`.
    pub fn neg_y(&self, level: usize) -> &Poly {
        &self.neg_y_polys[level]
    }
}

/// Pack the 2^ell ciphertexts reachable from `start_idx` into one, each landing at a
/// distinct coefficient. Combines the halves as
/// `(even + y*odd) + tau_t(even - y*odd)`.
pub fn pack_lwes_inner(
    ell: usize,
    start_idx: usize,
    rlwe_cts: &[RlweCiphertext],
    automorph_keys: &[KeySwitchingMatrix],
    y_constants: &YConstants,
    ctx: &NttContext,
    log_n: usize,
) -> RlweCiphertext {
    if ell == 0 {
        return rlwe_cts[start_idx].clone();
    }

    let step = 1 << (log_n - ell);
    let even = start_idx;
    let odd = start_idx + step;

    let ct_even = pack_lwes_inner(
        ell - 1,
        even,
        rlwe_cts,
        automorph_keys,
        y_constants,
        ctx,
        log_n,
    );
    let ct_odd = pack_lwes_inner(
        ell - 1,
        odd,
        rlwe_cts,
        automorph_keys,
        y_constants,
        ctx,
        log_n,
    );

    let y = y_constants.y(ell - 1);
    let neg_y = y_constants.neg_y(ell - 1);

    let y_times_odd = ct_odd.poly_mul(y, ctx);
    let ct_sum_0 = ct_even.add(&y_times_odd);

    let neg_y_times_odd = ct_odd.poly_mul(neg_y, ctx);
    let ct_sum_1 = ct_even.add(&neg_y_times_odd);

    let t = (1 << ell) + 1;

    // automorph_keys[i] holds t = d/2^i + 1, and this level needs t = 2^ell + 1.
    let log_d = automorph_keys.len();
    let ks_idx = log_d - ell;

    // Named abort beats the bare slice bounds check one line down.
    assert!(
        ks_idx < automorph_keys.len(),
        "ks_idx {ks_idx} out of bounds for {} automorph_keys",
        automorph_keys.len()
    );
    let ks_matrix = &automorph_keys[ks_idx];

    let ct_sum_1_auto = homomorphic_automorph(&ct_sum_1, t, ks_matrix, ctx);

    ct_sum_0.add(&ct_sum_1_auto)
}

/// Tree-pack up to d ciphertexts, each holding its message in coefficient 0, into one
/// carrying message k at coefficient k.
pub fn pack_rlwes_tree(
    rlwe_cts: &[RlweCiphertext],
    automorph_keys: &[KeySwitchingMatrix],
    params: &InspireParams,
) -> RlweCiphertext {
    let d = params.ring_dim;
    let q = params.q;
    let ctx = params.ntt_context();

    if rlwe_cts.is_empty() {
        return RlweCiphertext::from_parts(
            Poly::zero_moduli(d, params.moduli()),
            Poly::zero_moduli(d, params.moduli()),
        );
    }

    if rlwe_cts.len() == 1 {
        return rlwe_cts[0].clone();
    }

    let n = rlwe_cts.len();
    let log_n = (n as f64).log2().ceil() as usize;
    let padded_n = 1 << log_n;

    let mut padded_cts = rlwe_cts.to_vec();
    while padded_cts.len() < padded_n {
        padded_cts.push(RlweCiphertext::zero(params));
    }

    let y_constants = YConstants::generate(d, q, params.moduli());

    pack_lwes_inner(
        log_n,
        0,
        &padded_cts,
        automorph_keys,
        &y_constants,
        &ctx,
        log_n,
    )
}

/// Spread one ciphertext's coefficient-0 value across all coefficients by repeatedly
/// applying tau_t and adding; the result is scaled by d.
pub fn pack_single_lwe(
    ct: &RlweCiphertext,
    automorph_keys: &[KeySwitchingMatrix],
    params: &InspireParams,
) -> RlweCiphertext {
    let d = params.ring_dim;
    let ctx = params.ntt_context();
    let log_d = (d as f64).log2() as usize;

    assert!(
        automorph_keys.len() >= log_d,
        "pack_single_lwe requires at least {} automorphism keys, got {}",
        log_d,
        automorph_keys.len()
    );

    let mut cur = ct.clone();

    for (i, ks_matrix) in automorph_keys[..log_d].iter().enumerate() {
        let t = (d >> i) + 1;
        let tau_cur = homomorphic_automorph(&cur, t, ks_matrix, &ctx);
        cur = cur.add(&tau_cur);
    }

    cur
}

/// Undo `sample_extract_coeff0`: a_rlwe[0] = a_lwe[0], a_rlwe[j] = a_lwe[d-j], so
/// coeff_0(a_rlwe * s_rlwe) = <a_lwe, s_lwe>. No sign flip - the extraction's
/// negation already lives in the LWE vector.
fn invert_sample_extract(a_lwe: &[u64]) -> Vec<u64> {
    let d = a_lwe.len();
    let mut out = vec![0u64; d];

    out[0] = a_lwe[0];

    for j in 1..d {
        out[j] = a_lwe[d - j];
    }

    out
}

/// Build RLWEs with `a' = invert_sample_extract(lwe.a)` and `b' = 0`, returning the
/// b values separately for re-addition after packing.
pub fn prep_pack_lwes(
    lwe_cts: &[LweCiphertext],
    params: &InspireParams,
) -> (Vec<RlweCiphertext>, Vec<u64>) {
    let d = params.ring_dim;
    let moduli = params.moduli();

    let mut prepped_rlwes = Vec::with_capacity(lwe_cts.len());
    let mut b_values = Vec::with_capacity(lwe_cts.len());

    for lwe in lwe_cts {
        let a_coeffs = invert_sample_extract(&lwe.a);
        let a_poly = Poly::from_coeffs_moduli(a_coeffs, moduli);

        let b_poly = Poly::zero_moduli(d, moduli);

        prepped_rlwes.push(RlweCiphertext::from_parts(a_poly, b_poly));
        b_values.push(lwe.b);
    }

    (prepped_rlwes, b_values)
}

/// Tree-pack one LWE per column into a single RLWE holding column k at coefficient k,
/// scaled by d.
pub fn pack_lwes(
    lwe_cts: &[LweCiphertext],
    automorph_keys: &[KeySwitchingMatrix],
    params: &InspireParams,
) -> RlweCiphertext {
    let d = params.ring_dim;
    let q = params.q;
    let ctx = params.ntt_context();
    let moduli = params.moduli();

    if lwe_cts.is_empty() {
        return RlweCiphertext::from_parts(
            Poly::zero_moduli(d, moduli),
            Poly::zero_moduli(d, moduli),
        );
    }

    let (prepped_rlwes, b_values) = prep_pack_lwes(lwe_cts, params);

    // One element skips the tree, so it also skips the d-scaling.
    if prepped_rlwes.len() == 1 {
        let mut b_coeffs = vec![0u64; d];
        b_coeffs[0] = b_values[0];
        let b_poly = Poly::from_coeffs_moduli(b_coeffs, moduli);
        return RlweCiphertext::from_parts(prepped_rlwes[0].a.clone(), b_poly);
    }

    // Padding to a full d elements is what puts element k at coefficient k.
    let log_d = (d as f64).log2() as usize;

    let mut padded_cts = prepped_rlwes;
    let mut padded_b = b_values.clone();
    while padded_cts.len() < d {
        padded_cts.push(RlweCiphertext::zero(params));
        padded_b.push(0);
    }

    let y_constants = YConstants::generate(d, q, moduli);

    let mut packed = pack_lwes_inner(
        log_d,
        0,
        &padded_cts,
        automorph_keys,
        &y_constants,
        &ctx,
        log_d,
    );

    // The tree scales the packed a-side by d, so the b values must match it.
    let scale = d as u64;
    let mut b_coeffs = packed.b.coeffs().to_vec();
    for (z, &b_val) in padded_b.iter().enumerate() {
        if z < d {
            let scaled = ModQ::mul(b_val, scale, q);
            b_coeffs[z] = ModQ::add(b_coeffs[z], scaled, q);
        }
    }
    packed.b = Poly::from_coeffs_moduli(b_coeffs, moduli);

    packed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ks::generate_automorphism_ks_matrix;
    use crate::math::{GaussianSampler, ModQ};
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
            query_gadget_len: 3,
            packing_gadget_len: 3,
            security_level: crate::params::SecurityLevel::Bits128,
        }
    }

    fn sample_error_poly(dim: usize, q: u64, sampler: &mut GaussianSampler) -> Poly {
        let coeffs: Vec<u64> = (0..dim)
            .map(|_| ModQ::from_signed(sampler.sample(), q))
            .collect();
        Poly::from_coeffs(coeffs, q)
    }

    fn generate_automorph_keys(
        sk: &RlweSecretKey,
        params: &InspireParams,
        sampler: &mut GaussianSampler,
    ) -> Vec<KeySwitchingMatrix> {
        let d = params.ring_dim;
        let q = params.q;
        let ctx = params.ntt_context();
        let gadget = GadgetVector::new(params.gadget_base, params.packing_gadget_len, q);
        let log_d = (d as f64).log2() as usize;

        let mut keys = Vec::with_capacity(log_d);
        for i in 0..log_d {
            let t = (d >> i) + 1;
            let ks = generate_automorphism_ks_matrix(sk, t, &gadget, sampler, &ctx);
            keys.push(ks);
        }
        keys
    }

    #[test]
    fn test_y_constants_generation() {
        let d = 256;
        let q = 1152921504606830593u64;
        let y_consts = YConstants::generate(d, q, &[q]);

        assert_eq!(y_consts.y(0).coeff(128), 1);
        assert_eq!(y_consts.neg_y(0).coeff(128), q - 1);
        assert_eq!(y_consts.y(1).coeff(64), 1);
        assert_eq!(y_consts.y(7).coeff(1), 1);
    }

    #[test]
    fn test_homomorphic_automorph_identity() {
        let params = test_params();
        let d = params.ring_dim;
        let q = params.q;
        let delta = params.delta();
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);

        let message = 12345u64;
        let mut msg_coeffs = vec![0u64; d];
        msg_coeffs[0] = message;
        let msg_poly = Poly::from_coeffs(msg_coeffs, q);
        let a = Poly::random(d, q);
        let error = sample_error_poly(d, q, &mut sampler);
        let ct = RlweCiphertext::encrypt(&sk, &msg_poly, delta, a, &error, &ctx);

        let gadget = GadgetVector::new(params.gadget_base, params.packing_gadget_len, q);
        let ks_1 = generate_automorphism_ks_matrix(&sk, 1, &gadget, &mut sampler, &ctx);
        let ct_auto = homomorphic_automorph(&ct, 1, &ks_1, &ctx);

        let decrypted = ct_auto.decrypt(&sk, delta, params.p, &ctx);
        assert_eq!(
            decrypted.coeff(0),
            message,
            "Identity automorphism should preserve message"
        );
    }

    #[test]
    fn test_pack_single_runs() {
        let params = test_params();
        let d = params.ring_dim;
        let q = params.q;
        let delta = params.delta();
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);
        let automorph_keys = generate_automorph_keys(&sk, &params, &mut sampler);

        let message = 100u64;
        let mut msg_coeffs = vec![0u64; d];
        msg_coeffs[0] = message;
        let msg_poly = Poly::from_coeffs(msg_coeffs, q);
        let a = Poly::random(d, q);
        let error = sample_error_poly(d, q, &mut sampler);
        let ct = RlweCiphertext::encrypt(&sk, &msg_poly, delta, a, &error, &ctx);

        let orig_dec = ct.decrypt(&sk, delta, params.p, &ctx);
        assert_eq!(
            orig_dec.coeff(0),
            message,
            "Original message should decrypt correctly"
        );

        let packed = pack_single_lwe(&ct, &automorph_keys, &params);
        let decrypted = packed.decrypt(&sk, delta, params.p, &ctx);

        let expected_coeff0 = (message * (d as u64)) % params.p;
        assert_eq!(
            decrypted.coeff(0),
            expected_coeff0,
            "Coefficient 0 should be message * d mod p: got {}, expected {}",
            decrypted.coeff(0),
            expected_coeff0
        );
    }

    #[test]
    #[should_panic(expected = "pack_single_lwe requires at least")]
    fn test_pack_single_panics_when_automorph_keys_are_missing() {
        let params = test_params();
        let d = params.ring_dim;
        let q = params.q;
        let delta = params.delta();
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);
        let mut automorph_keys = generate_automorph_keys(&sk, &params, &mut sampler);
        automorph_keys.pop();

        let message = 7u64;
        let mut msg_coeffs = vec![0u64; d];
        msg_coeffs[0] = message;
        let msg_poly = Poly::from_coeffs(msg_coeffs, q);
        let a = Poly::random(d, q);
        let error = sample_error_poly(d, q, &mut sampler);
        let ct = RlweCiphertext::encrypt(&sk, &msg_poly, delta, a, &error, &ctx);

        let _ = pack_single_lwe(&ct, &automorph_keys, &params);
    }

    #[test]
    fn test_pack_two_rlwes() {
        let params = test_params();
        let d = params.ring_dim;
        let q = params.q;
        let delta = params.delta();
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);
        let automorph_keys = generate_automorph_keys(&sk, &params, &mut sampler);

        let messages = [100u64, 200u64];
        let cts: Vec<RlweCiphertext> = messages
            .iter()
            .map(|&msg| {
                let mut msg_coeffs = vec![0u64; d];
                msg_coeffs[0] = msg;
                let msg_poly = Poly::from_coeffs(msg_coeffs.clone(), q);
                let a = Poly::random(d, q);
                let error = sample_error_poly(d, q, &mut sampler);
                RlweCiphertext::encrypt(&sk, &msg_poly, delta, a, &error, &ctx)
            })
            .collect();

        let packed = pack_rlwes_tree(&cts, &automorph_keys, &params);
        let decrypted = packed.decrypt(&sk, delta, params.p, &ctx);

        println!("Decrypted coefficients (first 10):");
        for i in 0..10 {
            println!("  coeff[{}] = {}", i, decrypted.coeff(i));
        }
    }

    #[test]
    fn test_pack_lwes_single() {
        use crate::lwe::LweSecretKey;

        let params = test_params();
        let d = params.ring_dim;
        let q = params.q;
        let delta = params.delta();
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let rlwe_sk = RlweSecretKey::generate(&params, &mut sampler);
        let lwe_sk = LweSecretKey::from_rlwe(&rlwe_sk);
        let automorph_keys = generate_automorph_keys(&rlwe_sk, &params, &mut sampler);

        let message = 12345u64;
        let mut msg_coeffs = vec![0u64; d];
        msg_coeffs[0] = message;
        let msg_poly = Poly::from_coeffs(msg_coeffs, q);
        let a = Poly::random(d, q);
        let error = sample_error_poly(d, q, &mut sampler);
        let rlwe_ct = RlweCiphertext::encrypt(&rlwe_sk, &msg_poly, delta, a, &error, &ctx);

        let lwe_ct = rlwe_ct.sample_extract_coeff0();

        let lwe_dec = lwe_ct.decrypt(&lwe_sk, delta, params.p);
        assert_eq!(
            lwe_dec, message,
            "LWE decrypt failed: got {lwe_dec}, expected {message}"
        );

        let packed = pack_lwes(&[lwe_ct], &automorph_keys, &params);

        let packed_dec = packed.decrypt(&rlwe_sk, delta, params.p, &ctx);

        assert_eq!(
            packed_dec.coeff(0),
            message,
            "Packed single LWE decrypt failed: got {}, expected {}",
            packed_dec.coeff(0),
            message
        );
    }

    #[test]
    fn test_pack_lwes_two() {
        let params = test_params();
        let d = params.ring_dim;
        let q = params.q;
        let delta = params.delta();
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let rlwe_sk = RlweSecretKey::generate(&params, &mut sampler);
        let automorph_keys = generate_automorph_keys(&rlwe_sk, &params, &mut sampler);

        let messages = [100u64, 200u64];
        let lwe_cts: Vec<_> = messages
            .iter()
            .map(|&msg| {
                let mut msg_coeffs = vec![0u64; d];
                msg_coeffs[0] = msg;
                let msg_poly = Poly::from_coeffs(msg_coeffs, q);
                let a = Poly::random(d, q);
                let error = sample_error_poly(d, q, &mut sampler);
                let rlwe_ct = RlweCiphertext::encrypt(&rlwe_sk, &msg_poly, delta, a, &error, &ctx);
                rlwe_ct.sample_extract_coeff0()
            })
            .collect();

        let packed = pack_lwes(&lwe_cts, &automorph_keys, &params);
        let packed_dec = packed.decrypt(&rlwe_sk, delta, params.p, &ctx);

        println!("Pack 2 LWEs (first 4 coefficients):");
        for i in 0..4 {
            println!("  coeff[{}] = {}", i, packed_dec.coeff(i));
        }

        let p = params.p;
        assert_eq!(
            packed_dec.coeff(0),
            (100 * (d as u64)) % p,
            "coeff[0] should be 100*d mod p"
        );
        assert_eq!(
            packed_dec.coeff(1),
            (200 * (d as u64)) % p,
            "coeff[1] should be 200*d mod p"
        );
    }

    #[test]
    fn test_pack_lwes_trivial() {
        use crate::lwe::LweCiphertext;

        let params = test_params();
        let d = params.ring_dim;
        let q = params.q;
        let delta = params.delta();
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let rlwe_sk = RlweSecretKey::generate(&params, &mut sampler);
        let automorph_keys = generate_automorph_keys(&rlwe_sk, &params, &mut sampler);

        // a = 0 isolates b-value placement from noise.
        let messages = [100u64, 200u64, 300u64, 400u64];
        let lwe_cts: Vec<_> = messages
            .iter()
            .map(|&msg| {
                let a = vec![0u64; d];
                let b = ModQ::mul(delta, msg, q);
                LweCiphertext { a, b, q }
            })
            .collect();

        let packed = pack_lwes(&lwe_cts, &automorph_keys, &params);
        let packed_dec = packed.decrypt(&rlwe_sk, delta, params.p, &ctx);

        println!("Trivial packed LWE decryption (first 8 coefficients):");
        for i in 0..8 {
            println!("  coeff[{}] = {}", i, packed_dec.coeff(i));
        }

        let p = params.p;
        assert_eq!(
            packed_dec.coeff(0),
            (100 * (d as u64)) % p,
            "coeff[0] should be 100*d mod p"
        );
        assert_eq!(
            packed_dec.coeff(1),
            (200 * (d as u64)) % p,
            "coeff[1] should be 200*d mod p"
        );
        assert_eq!(
            packed_dec.coeff(2),
            (300 * (d as u64)) % p,
            "coeff[2] should be 300*d mod p"
        );
        assert_eq!(
            packed_dec.coeff(3),
            (400 * (d as u64)) % p,
            "coeff[3] should be 400*d mod p"
        );
    }

    #[test]
    fn test_pack_lwes_multiple() {
        use crate::lwe::LweSecretKey;

        let params = test_params();
        let d = params.ring_dim;
        let q = params.q;
        let delta = params.delta();
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let rlwe_sk = RlweSecretKey::generate(&params, &mut sampler);
        let lwe_sk = LweSecretKey::from_rlwe(&rlwe_sk);
        let automorph_keys = generate_automorph_keys(&rlwe_sk, &params, &mut sampler);

        let messages = [100u64, 200u64, 300u64, 400u64];
        let lwe_cts: Vec<_> = messages
            .iter()
            .map(|&msg| {
                let mut msg_coeffs = vec![0u64; d];
                msg_coeffs[0] = msg;
                let msg_poly = Poly::from_coeffs(msg_coeffs, q);
                let a = Poly::random(d, q);
                let error = sample_error_poly(d, q, &mut sampler);
                let rlwe_ct = RlweCiphertext::encrypt(&rlwe_sk, &msg_poly, delta, a, &error, &ctx);
                rlwe_ct.sample_extract_coeff0()
            })
            .collect();

        for (i, lwe) in lwe_cts.iter().enumerate() {
            let dec = lwe.decrypt(&lwe_sk, delta, params.p);
            assert_eq!(dec, messages[i], "LWE {i} decrypt failed");
        }

        let packed = pack_lwes(&lwe_cts, &automorph_keys, &params);
        let packed_dec = packed.decrypt(&rlwe_sk, delta, params.p, &ctx);

        println!("Packed LWE decryption (first 8 coefficients):");
        for i in 0..8 {
            println!("  coeff[{}] = {}", i, packed_dec.coeff(i));
        }

        let c0 = packed_dec.coeff(0);
        let c1 = packed_dec.coeff(1);
        assert_eq!(c0, 100 * (d as u64), "coeff[0] should be 100*d");
        assert_eq!(c1, 200 * (d as u64), "coeff[1] should be 200*d");
    }
}
