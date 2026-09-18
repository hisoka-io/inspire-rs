//! PIR.Extract: decrypt the server response and rebuild the entry bytes.

use super::error::{ExtractError, Result};

use crate::params::InspireVariant;

use super::encode_db::reconstruct_entry;
use super::query::ClientState;
use super::respond::ServerResponse;
use super::setup::InspireCrs;

/// Decrypt an unpacked response; the rotated target sits in the constant term.
pub fn extract(
    crs: &InspireCrs,
    state: &ClientState,
    response: &ServerResponse,
    entry_size: usize,
) -> Result<Vec<u8>> {
    let num_columns = validate_unpacked_response_columns("extract", response, entry_size)?;
    let p = crs.params.p;
    let delta = crs.params.delta();
    let ctx = crs.params.ntt_context();
    let mut column_values = Vec::with_capacity(num_columns);

    for col_ct in &response.column_ciphertexts {
        let decrypted = col_ct.decrypt(&state.rlwe_secret_key, delta, p, &ctx);
        let value = decrypted.coeff(0);
        column_values.push(value);
    }

    let entry = reconstruct_entry(&column_values, entry_size);

    Ok(entry)
}

/// Dispatch extraction on the variant the server responded with.
pub fn extract_with_variant(
    crs: &InspireCrs,
    state: &ClientState,
    response: &ServerResponse,
    entry_size: usize,
    variant: InspireVariant,
) -> Result<Vec<u8>> {
    match variant {
        InspireVariant::NoPacking => extract(crs, state, response, entry_size),
        InspireVariant::OnePacking => extract_packed(crs, state, response, entry_size),
        InspireVariant::TwoPacking => extract_two_packing(crs, state, response, entry_size),
    }
}

/// Extract a TwoPacking response, refusing any server-selected mode but InspiRING.
pub fn extract_two_packing(
    crs: &InspireCrs,
    state: &ClientState,
    response: &ServerResponse,
    entry_size: usize,
) -> Result<Vec<u8>> {
    use crate::pir::query::PackingMode;
    match response.packing_mode {
        Some(PackingMode::Inspiring) => extract_inspiring(crs, state, response, entry_size),
        Some(PackingMode::Tree) => Err(ExtractError::TwoPackingModeMismatch {
            mode: Some(PackingMode::Tree),
            rims_tag_byte: 2,
        }
        .into()),
        None => Err(ExtractError::TwoPackingModeMismatch {
            mode: None,
            rims_tag_byte: 0,
        }
        .into()),
    }
}

fn validate_unpacked_response_columns(
    operation: &'static str,
    response: &ServerResponse,
    entry_size: usize,
) -> Result<usize> {
    let expected = crate::num_columns(entry_size);
    let got = response.column_ciphertexts.len();
    if got != expected {
        return Err(ExtractError::ColumnCountMismatch {
            operation,
            got,
            expected,
            entry_size,
        }
        .into());
    }
    Ok(expected)
}

pub(crate) fn validate_packed_response_coefficients(
    operation: &'static str,
    response: &ServerResponse,
    required: usize,
) -> Result<()> {
    let Some(retained) = response.packed_coefficients else {
        return Ok(());
    };
    let retained = usize::try_from(retained).map_err(|_| ExtractError::PackedCoefficientCount {
        operation,
        got: usize::MAX,
        required,
    })?;
    if retained != required {
        return Err(ExtractError::PackedCoefficientCount {
            operation,
            got: retained,
            required,
        }
        .into());
    }
    Ok(())
}

/// Extract a tree-packed response: columns sit at coefficients 0.. scaled by d.
fn extract_packed(
    crs: &InspireCrs,
    state: &ClientState,
    response: &ServerResponse,
    entry_size: usize,
) -> Result<Vec<u8>> {
    let d = crs.ring_dim();
    let p = crs.params.p;
    let delta = crs.params.delta();
    let ctx = crs.params.ntt_context();

    let num_columns = crate::num_columns(entry_size);
    validate_packed_response_coefficients("extract_packed", response, num_columns)?;

    let decrypted = response
        .ciphertext
        .decrypt(&state.rlwe_secret_key, delta, p, &ctx);

    // d^{-1} exists only when gcd(d, p) == 1; substituting 1 would return d-scaled
    // garbage, so the misuse is surfaced instead.
    let d_inv =
        mod_inverse(d as u64, p).ok_or(ExtractError::DegreeNotInvertible { d: d as u64, p })?;

    let mut column_values = Vec::with_capacity(num_columns);
    for col in 0..num_columns {
        let scaled_value = decrypted.coeff(col);
        let value = (scaled_value as u128 * d_inv as u128 % p as u128) as u64;
        column_values.push(value);
    }

    let entry = reconstruct_entry(&column_values, entry_size);

    Ok(entry)
}

/// Extract an InspiRING-packed response; unlike tree packing it applies no d-scaling.
pub fn extract_inspiring(
    crs: &InspireCrs,
    state: &ClientState,
    response: &ServerResponse,
    entry_size: usize,
) -> Result<Vec<u8>> {
    let p = crs.params.p;
    let delta = crs.params.delta();
    let ctx = crs.params.ntt_context();

    let num_columns = crate::num_columns(entry_size);
    validate_packed_response_coefficients("extract_inspiring", response, num_columns)?;

    let decrypted = response
        .ciphertext
        .decrypt(&state.rlwe_secret_key, delta, p, &ctx);

    let mut column_values = Vec::with_capacity(num_columns);
    for col in 0..num_columns {
        let value = decrypted.coeff(col);
        column_values.push(value);
    }

    let entry = reconstruct_entry(&column_values, entry_size);

    Ok(entry)
}

fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    let (g, x, _) = extended_gcd(a as i64, m as i64);
    if g == 1 {
        Some(((x % m as i64 + m as i64) % m as i64) as u64)
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

/// Extract, rounding coefficients that sit within `tolerance` of a wrap boundary.
#[allow(dead_code)]
pub fn extract_with_tolerance(
    crs: &InspireCrs,
    state: &ClientState,
    response: &ServerResponse,
    entry_size: usize,
    tolerance: u64,
) -> Result<Vec<u8>> {
    let num_columns =
        validate_unpacked_response_columns("extract_with_tolerance", response, entry_size)?;
    let p = crs.params.p;
    let delta = crs.params.delta();
    let ctx = crs.params.ntt_context();
    let mut column_values = Vec::with_capacity(num_columns);

    let apply_tolerance = |mut value: u64| -> u64 {
        if value > p - tolerance && value < p {
            value = 0;
        } else if value > tolerance && value < 2 * tolerance {
            value %= p;
        }
        value
    };

    for col_ct in &response.column_ciphertexts {
        let decrypted = col_ct.decrypt(&state.rlwe_secret_key, delta, p, &ctx);
        let value = apply_tolerance(decrypted.coeff(0));
        column_values.push(value);
    }

    let entry = reconstruct_entry(&column_values, entry_size);

    Ok(entry)
}

/// Decrypt and return the constant term alone.
#[allow(dead_code)]
#[allow(
    clippy::unnecessary_wraps,
    reason = "uniform Result shape across the extract family; the sibling entry points are genuinely fallible"
)]
pub fn extract_single_coeff(
    crs: &InspireCrs,
    state: &ClientState,
    response: &ServerResponse,
) -> Result<u64> {
    let p = crs.params.p;
    let delta = crs.params.delta();
    let ctx = crs.params.ntt_context();

    let decrypted = response
        .ciphertext
        .decrypt(&state.rlwe_secret_key, delta, p, &ctx);

    Ok(decrypted.coeff(0))
}

/// Decrypt and return every coefficient.
#[allow(dead_code)]
#[allow(
    clippy::unnecessary_wraps,
    reason = "uniform Result shape across the extract family; the sibling entry points are genuinely fallible"
)]
pub fn extract_raw(
    crs: &InspireCrs,
    state: &ClientState,
    response: &ServerResponse,
) -> Result<Vec<u64>> {
    let p = crs.params.p;
    let delta = crs.params.delta();
    let ctx = crs.params.ntt_context();

    let decrypted = response
        .ciphertext
        .decrypt(&state.rlwe_secret_key, delta, p, &ctx);

    Ok(decrypted.coeffs().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::GaussianSampler;
    use crate::pir::query::query;
    use crate::pir::respond::respond;
    use crate::pir::setup::setup;

    fn test_params() -> crate::params::InspireParams {
        crate::params::InspireParams {
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

    #[test]
    fn test_extract_produces_correct_length() {
        let params = test_params();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let entry_size = 32;
        let num_entries = params.ring_dim;
        let database: Vec<u8> = (0..(num_entries * entry_size))
            .map(|i| (i % 256) as u8)
            .collect();

        let (crs, encoded_db, rlwe_sk) =
            setup(&params, &database, entry_size, &mut sampler).unwrap();

        let target_index = 42u64;
        let (state, client_query) = query(
            &crs,
            target_index,
            &encoded_db.config,
            &rlwe_sk,
            &mut sampler,
        )
        .unwrap();

        let response = respond(&crs, &encoded_db, &client_query).unwrap();

        let result = extract(&crs, &state, &response, entry_size);
        assert!(result.is_ok());

        let entry = result.unwrap();
        assert_eq!(entry.len(), entry_size);
    }

    #[test]
    fn test_extract_single_coeff() {
        let params = test_params();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let entry_size = 32;
        let num_entries = params.ring_dim;
        let database: Vec<u8> = (0..(num_entries * entry_size))
            .map(|i| (i % 256) as u8)
            .collect();

        let (crs, encoded_db, rlwe_sk) =
            setup(&params, &database, entry_size, &mut sampler).unwrap();

        let target_index = 10u64;
        let (state, client_query) = query(
            &crs,
            target_index,
            &encoded_db.config,
            &rlwe_sk,
            &mut sampler,
        )
        .unwrap();

        let response = respond(&crs, &encoded_db, &client_query).unwrap();

        let result = extract_single_coeff(&crs, &state, &response);
        assert!(result.is_ok());
    }

    #[test]
    fn test_extract_raw() {
        let params = test_params();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let entry_size = 32;
        let num_entries = params.ring_dim;
        let database: Vec<u8> = (0..(num_entries * entry_size))
            .map(|i| (i % 256) as u8)
            .collect();

        let (crs, encoded_db, rlwe_sk) =
            setup(&params, &database, entry_size, &mut sampler).unwrap();

        let target_index = 5u64;
        let (state, client_query) = query(
            &crs,
            target_index,
            &encoded_db.config,
            &rlwe_sk,
            &mut sampler,
        )
        .unwrap();

        let response = respond(&crs, &encoded_db, &client_query).unwrap();

        let result = extract_raw(&crs, &state, &response);
        assert!(result.is_ok());

        let raw = result.unwrap();
        assert_eq!(raw.len(), params.ring_dim);
    }

    #[test]
    fn test_reconstruct_entry() {
        let entry_size = 32;
        let column_values: Vec<u64> = (0..16).map(|i| i * 256 + i).collect();

        let entry = reconstruct_entry(&column_values, entry_size);

        assert_eq!(entry.len(), entry_size);

        for (i, &val) in column_values.iter().enumerate() {
            let low = entry[i * 2];
            let high = entry[i * 2 + 1];
            let reconstructed = low as u64 | ((high as u64) << 8);
            assert_eq!(reconstructed, val & 0xFFFF);
        }
    }

    #[test]
    fn extract_packed_returns_degree_not_invertible_when_gcd_d_p_nonunit() {
        use crate::pir::query::PackingMode;
        let params = test_params();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let entry_size = 32;
        let num_entries = params.ring_dim;
        let database: Vec<u8> = (0..(num_entries * entry_size))
            .map(|i| (i % 256) as u8)
            .collect();

        let (crs, encoded_db, rlwe_sk) =
            setup(&params, &database, entry_size, &mut sampler).unwrap();

        let target_index = 7u64;
        let (state, client_query) = query(
            &crs,
            target_index,
            &encoded_db.config,
            &rlwe_sk,
            &mut sampler,
        )
        .unwrap();

        let mut response = respond(&crs, &encoded_db, &client_query).unwrap();
        // Clearing `column_ciphertexts` is what forces `extract_packed` to run.
        response.packing_mode = Some(PackingMode::Tree);
        response.column_ciphertexts.clear();

        let err = extract_packed(&crs, &state, &response, entry_size).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("d^{-1} mod p does not exist"),
            "expected DegreeNotInvertible message, got: {msg}",
        );
        assert!(
            msg.contains("d=256"),
            "expected d=256 in message, got: {msg}"
        );
        assert!(
            msg.contains("p=65536"),
            "expected p=65536 in message, got: {msg}"
        );
    }

    #[test]
    fn extract_packed_succeeds_when_gcd_d_p_is_one() {
        let params = crate::params::InspireParams {
            ring_dim: 8,
            q: 1152921504606830593,
            crt_moduli: vec![1152921504606830593],
            p: 17,
            sigma: 6.4,
            gadget_base: 1 << 20,
            query_gadget_len: 3,
            packing_gadget_len: 3,
            security_level: crate::params::SecurityLevel::Bits128,
        };

        let inv = mod_inverse(params.ring_dim as u64, params.p).unwrap();
        assert_eq!(
            (params.ring_dim as u128 * inv as u128) % params.p as u128,
            1,
            "d * d_inv must equal 1 mod p when gcd(d, p) == 1"
        );
    }

    #[test]
    fn extract_error_degree_not_invertible_when_d_equals_p() {
        let p = 257u64;
        let d = 257u64;
        assert_eq!(mod_inverse(d, p), None);
        let err = ExtractError::DegreeNotInvertible { d, p };
        let msg = err.to_string();
        assert!(msg.contains("d=257"));
        assert!(msg.contains("p=257"));
    }

    #[test]
    fn test_extract_with_tolerance() {
        let params = test_params();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let entry_size = 32;
        let num_entries = params.ring_dim;
        let database: Vec<u8> = (0..(num_entries * entry_size))
            .map(|i| (i % 256) as u8)
            .collect();

        let (crs, encoded_db, rlwe_sk) =
            setup(&params, &database, entry_size, &mut sampler).unwrap();

        let target_index = 15u64;
        let (state, client_query) = query(
            &crs,
            target_index,
            &encoded_db.config,
            &rlwe_sk,
            &mut sampler,
        )
        .unwrap();

        let response = respond(&crs, &encoded_db, &client_query).unwrap();

        let tolerance = 10u64;
        let result = extract_with_tolerance(&crs, &state, &response, entry_size, tolerance);
        assert!(result.is_ok());

        let entry = result.unwrap();
        assert_eq!(entry.len(), entry_size);
    }

    #[test]
    fn extract_with_tolerance_refuses_every_wrong_column_count() {
        let params = test_params();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 13);
        let entry_size = 32;
        let database: Vec<u8> = (0..(params.ring_dim * entry_size))
            .map(|i| (i % 251) as u8)
            .collect();
        let (crs, encoded_db, rlwe_sk) =
            setup(&params, &database, entry_size, &mut sampler).unwrap();
        let (state, client_query) =
            query(&crs, 15, &encoded_db.config, &rlwe_sk, &mut sampler).unwrap();
        let response = respond(&crs, &encoded_db, &client_query).unwrap();
        let expected = crate::num_columns(entry_size);

        for got in [0, expected - 1, expected + 1] {
            let mut malformed = response.clone();
            if got <= expected {
                malformed.column_ciphertexts.truncate(got);
            } else {
                malformed
                    .column_ciphertexts
                    .push(response.column_ciphertexts[0].clone());
            }
            let err = extract_with_tolerance(&crs, &state, &malformed, entry_size, 10)
                .expect_err("wrong column count must be refused");
            let message = err.to_string();
            assert!(message.contains(&format!("got {got}")), "{message}");
            assert!(
                message.contains(&format!("expected {expected}")),
                "{message}"
            );
        }
    }
}
