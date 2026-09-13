#![allow(
    clippy::expect_used,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]

use proptest::prelude::*;
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::{
    extract, extract_inspiring, extract_with_variant, query_seeded, respond, respond_one_packing,
    respond_seeded_inspiring, setup, InspireVariant,
};
use serde::Serialize;

const ENTRY_SIZE: usize = 256;

#[derive(Serialize)]
struct ForgedResponseWire {
    ciphertext: ForgedResponseCiphertextWire,
    column_ciphertexts: Vec<raven_inspire::rlwe::RlweCiphertext>,
    packing_mode: Option<raven_inspire::PackingMode>,
}

#[derive(Serialize)]
enum ForgedResponseCiphertextWire {
    Full(raven_inspire::rlwe::RlweCiphertext),
}

fn params() -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65_537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

fn response_fixture() -> (
    raven_inspire::ServerCrs,
    raven_inspire::EncodedDatabase,
    raven_inspire::ClientState,
    raven_inspire::SeededClientQuery,
    raven_inspire::ServerResponse,
    Vec<u8>,
) {
    let params = params();
    let database: Vec<u8> = (0..params.ring_dim * ENTRY_SIZE)
        .map(|offset| u8::try_from((offset * 17 + 29) % 251).expect("mod 251"))
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0x5131);
    let (crs, encoded, secret_key) =
        setup(&params, &database, ENTRY_SIZE, &mut sampler).expect("setup");
    let (client_state, query) =
        query_seeded(&crs, 7, &encoded.config, &secret_key, &mut sampler).expect("query");
    let response = respond_seeded_inspiring(&crs, &encoded, &query).expect("respond");
    let expected = database
        .get(7 * ENTRY_SIZE..8 * ENTRY_SIZE)
        .expect("expected row")
        .to_vec();
    (crs, encoded, client_state, query, response, expected)
}

#[test]
fn packed_response_retains_exactly_gamma_b_coefficients() {
    let (crs, _, state, _, response, expected) = response_fixture();
    let gamma = raven_inspire::num_columns(ENTRY_SIZE);
    assert_eq!(response.packed_coefficients, Some(gamma as u32));

    let bytes = response.to_binary().expect("compact encode");
    let decoded = raven_inspire::ServerResponse::from_binary(&bytes).expect("compact decode");
    assert_eq!(decoded.packed_coefficients, Some(gamma as u32));
    assert_eq!(
        extract_inspiring(&crs, &state, &decoded, ENTRY_SIZE).expect("compact extract"),
        expected
    );

    let mut full = response.clone();
    full.packed_coefficients = Some(crs.ring_dim() as u32);
    let full_bytes = full.to_binary().expect("full control encode");
    let full_decoded =
        raven_inspire::ServerResponse::from_binary(&full_bytes).expect("full decode");
    let error = extract_inspiring(&crs, &state, &full_decoded, ENTRY_SIZE)
        .expect_err("surplus packed coefficients must be refused");
    assert!(error.to_string().contains("packed coefficient"), "{error}");
    assert!(
        bytes.len() < full_bytes.len(),
        "gamma prefix must reduce wire bytes"
    );
}

#[test]
fn gamma_minus_one_is_refused_and_last_retained_coefficient_is_required() {
    let (crs, _, state, _, response, expected) = response_fixture();
    let gamma = raven_inspire::num_columns(ENTRY_SIZE);

    let mut short = response.clone();
    short.packed_coefficients = Some((gamma - 1) as u32);
    let short_bytes = short.to_binary().expect("short encode");
    let short_decoded =
        raven_inspire::ServerResponse::from_binary(&short_bytes).expect("short decode");
    let error = extract_inspiring(&crs, &state, &short_decoded, ENTRY_SIZE)
        .expect_err("gamma-1 must be refused before extraction");
    assert!(error.to_string().contains("packed coefficient"), "{error}");

    let mut corrupted = response;
    let last = corrupted
        .ciphertext
        .b
        .coeffs_mut()
        .get_mut(gamma - 1)
        .expect("last retained coefficient");
    *last = (*last + crs.params.delta()) % crs.params.q;
    let corrupted_bytes = corrupted.to_binary().expect("corrupt encode");
    let corrupted =
        raven_inspire::ServerResponse::from_binary(&corrupted_bytes).expect("corrupt decode");
    let plaintext = extract_inspiring(&crs, &state, &corrupted, ENTRY_SIZE)
        .expect("corrupt response remains structurally valid");
    assert_ne!(
        plaintext, expected,
        "the last retained coefficient must affect output"
    );
    assert_ne!(
        plaintext.get(ENTRY_SIZE - 2..).expect("last output pair"),
        expected.get(ENTRY_SIZE - 2..).expect("expected last pair")
    );
}

#[test]
fn gamma_plus_one_is_refused() {
    let (crs, _, state, _, mut response, _) = response_fixture();
    let gamma = raven_inspire::num_columns(ENTRY_SIZE);
    response.packed_coefficients = Some((gamma + 1) as u32);
    let bytes = response.to_binary().expect("overlong encode");
    let response = raven_inspire::ServerResponse::from_binary(&bytes).expect("overlong decode");

    let error = extract_inspiring(&crs, &state, &response, ENTRY_SIZE)
        .expect_err("gamma+1 must be refused before extraction");
    assert!(error.to_string().contains("packed coefficient"), "{error}");
}

#[test]
fn packed_response_decoder_refuses_trailing_bytes() {
    let (_, _, _, _, response, _) = response_fixture();
    let mut bytes = response.to_binary().expect("compact encode");
    bytes.push(0xa5);

    let error = raven_inspire::ServerResponse::from_binary(&bytes)
        .expect_err("trailing bytes must not be accepted as part of a response");
    assert!(error.to_string().contains("bytes remaining"), "{error}");
}

#[test]
fn packed_mode_refuses_a_full_ciphertext_wire_variant() {
    let (_, _, _, _, mut response, _) = response_fixture();
    response.packed_coefficients = None;
    let serialize_error = response
        .to_binary()
        .expect_err("packed response must not bypass prefix serialization");
    assert!(
        serialize_error.to_string().contains("packed"),
        "{serialize_error}"
    );

    let forged = ForgedResponseWire {
        ciphertext: ForgedResponseCiphertextWire::Full(response.ciphertext),
        column_ciphertexts: vec![],
        packing_mode: Some(raven_inspire::PackingMode::Inspiring),
    };
    let bytes = bincode::serialize(&forged).expect("forge full packed response wire");
    let decode_error = raven_inspire::ServerResponse::from_binary(&bytes)
        .expect_err("packed response must not accept the full ciphertext wire variant");
    assert!(
        decode_error.to_string().contains("packed"),
        "{decode_error}"
    );
}

#[test]
fn tree_packing_uses_the_same_plaintext_prefix_boundary() {
    let (crs, encoded, state, query, response, expected) = response_fixture();
    let gamma = raven_inspire::num_columns(ENTRY_SIZE);
    let tree = respond_one_packing(&crs, &encoded, &query.expand()).expect("tree response");
    assert_eq!(tree.packed_coefficients, Some(gamma as u32));
    let decoded =
        raven_inspire::ServerResponse::from_binary(&tree.to_binary().expect("tree prefix encode"))
            .expect("tree prefix decode");
    assert_eq!(
        extract_with_variant(
            &crs,
            &state,
            &decoded,
            ENTRY_SIZE,
            InspireVariant::OnePacking
        )
        .expect("tree extract"),
        expected
    );
    assert_eq!(response.packed_coefficients, Some(gamma as u32));
}

#[test]
fn unpacked_response_keeps_full_column_ciphertexts() {
    let (crs, encoded, state, query, packed, expected) = response_fixture();
    let unpacked = respond(&crs, &encoded, &query.expand()).expect("unpacked response");
    assert_eq!(unpacked.packed_coefficients, None);
    let decoded = raven_inspire::ServerResponse::from_binary(
        &unpacked.to_binary().expect("unpacked full encode"),
    )
    .expect("unpacked full decode");
    assert_eq!(
        extract(&crs, &state, &decoded, ENTRY_SIZE).expect("unpacked extract"),
        expected
    );
    assert!(
        unpacked.to_binary().expect("unpacked bytes").len()
            > packed.to_binary().expect("packed bytes").len()
    );
}

#[test]
fn production_gamma_128_wire_bytes_are_exact() {
    let params = InspireParams::secure_128_d2048();
    let database: Vec<u8> = (0..64 * ENTRY_SIZE)
        .map(|offset| u8::try_from((offset * 19 + 11) % 251).expect("mod 251"))
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0x5134);
    let (crs, encoded, secret_key) =
        setup(&params, &database, ENTRY_SIZE, &mut sampler).expect("production setup");
    let (state, query) = query_seeded(&crs, 7, &encoded.config, &secret_key, &mut sampler)
        .expect("production query");
    let response = respond_seeded_inspiring(&crs, &encoded, &query).expect("production response");
    assert_eq!(response.packed_coefficients, Some(128));
    let compact = response.to_binary().expect("production compact encode");
    assert_eq!(compact.len(), 17_486);
    let decoded = raven_inspire::ServerResponse::from_binary(&compact).expect("production decode");
    assert_eq!(
        extract_inspiring(&crs, &state, &decoded, ENTRY_SIZE).expect("production extract"),
        database
            .get(7 * ENTRY_SIZE..8 * ENTRY_SIZE)
            .expect("production expected")
    );

    let mut full = response;
    full.packed_coefficients = Some(params.ring_dim as u32);
    let full_bytes = full.to_binary().expect("production full control");
    assert_eq!(full_bytes.len(), 32_846);
    let full = raven_inspire::ServerResponse::from_binary(&full_bytes).expect("full decode");
    let error = extract_inspiring(&crs, &state, &full, ENTRY_SIZE)
        .expect_err("surplus packed coefficients must be refused");
    assert!(error.to_string().contains("packed coefficient"), "{error}");
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn packed_wire_preserves_each_crt_limb_prefix(
        dimension_log in 3u32..=6,
        retained_seed in any::<u16>(),
        two_crt in any::<bool>(),
    ) {
        let dimension = 1usize << dimension_log;
        let retained = 1usize << (u32::from(retained_seed) % dimension_log);
        let moduli = if two_crt {
            vec![67_043_329, 132_120_577]
        } else {
            vec![1_152_921_504_606_830_593]
        };
        let coefficients = |offset: u64| {
            moduli
                .iter()
                .enumerate()
                .flat_map(|(limb, modulus)| {
                    (0..dimension).map(move |coefficient| {
                        (offset + limb as u64 * dimension as u64 + coefficient as u64) % modulus
                    })
                })
                .collect::<Vec<_>>()
        };
        let a_coefficients = coefficients(17);
        let b_coefficients = coefficients(97);
        let response = raven_inspire::ServerResponse {
            ciphertext: raven_inspire::rlwe::RlweCiphertext::from_parts(
                raven_inspire::math::Poly::from_crt_coeffs(a_coefficients.clone(), &moduli),
                raven_inspire::math::Poly::from_crt_coeffs(b_coefficients.clone(), &moduli),
            ),
            column_ciphertexts: vec![],
            packing_mode: Some(raven_inspire::PackingMode::Inspiring),
            packed_coefficients: Some(retained as u32),
        };

        let bytes = response.to_binary().expect("serialize packed response");
        let decoded = raven_inspire::ServerResponse::from_binary(&bytes)
            .expect("deserialize packed response");
        prop_assert_eq!(decoded.packed_coefficients, Some(retained as u32));
        prop_assert_eq!(decoded.ciphertext.a.coeffs(), a_coefficients.as_slice());
        for limb in 0..moduli.len() {
            let start = limb * dimension;
            prop_assert_eq!(
                &decoded.ciphertext.b.coeffs()[start..start + retained],
                &b_coefficients[start..start + retained],
            );
            prop_assert!(
                decoded.ciphertext.b.coeffs()[start + retained..start + dimension]
                    .iter()
                    .all(|coefficient| *coefficient == 0)
            );
        }

        let mut overlong = bytes;
        overlong.push(0xa5);
        prop_assert!(raven_inspire::ServerResponse::from_binary(&overlong).is_err());
    }
}
