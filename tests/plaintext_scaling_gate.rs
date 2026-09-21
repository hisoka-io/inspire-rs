//! `Delta = floor(q/p)` sets the rounding interval decryption tolerates. A fresh
//! encryption's error already reaches the sampler tailcut, so an interval no wider than that
//! decodes to a neighbouring plaintext with no error; `validate` refuses it.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::DEFAULT_Q_2CRT_30BIT;
use raven_inspire::pir::{extract_inspiring, query, respond_inspiring, setup};
use raven_inspire::{InspireParams, SecurityLevel};

const RING_DIM: usize = 256;
const ENTRY_SIZE: usize = 32;
const P: u64 = 65_537;
/// Prime, `== 1 mod 512`; `floor(q/p) = 78`, half-interval 39 = `ceil(6 * 6.4)`.
const Q_AT_THE_TAILCUT: u64 = 5_114_881;
/// Prime, `== 1 mod 512`; `floor(q/p) = 80`, half-interval 40.
const Q_ONE_PAST_THE_TAILCUT: u64 = 5_243_393;

fn params_with_q(q: u64, gadget_len: usize) -> InspireParams {
    InspireParams {
        ring_dim: RING_DIM,
        q,
        crt_moduli: vec![q],
        p: P,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: gadget_len,
        packing_gadget_len: gadget_len,
        security_level: SecurityLevel::Bits128,
    }
}

/// Rows that decode wrongly, over five queries, when the parameters are used anyway.
fn wrong_rows_when_served(params: &InspireParams) -> String {
    let database: Vec<u8> = (0..RING_DIM * ENTRY_SIZE)
        .map(|i| ((i * 17 + 3) % 251) as u8)
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 5);
    let (crs, encoded_db, sk) = setup(params, &database, ENTRY_SIZE, &mut sampler).expect("setup");
    let mut wrong = 0;
    for index in [0u64, 1, 42, 100, 255] {
        let (state, client_query) =
            query(&crs, index, &encoded_db.config, &sk, &mut sampler).expect("query");
        let response = respond_inspiring(&crs, &encoded_db, &client_query).expect("respond");
        let bytes = extract_inspiring(&crs, &state, &response, ENTRY_SIZE).expect("extract");
        let start = usize::try_from(index).unwrap() * ENTRY_SIZE;
        if bytes != database[start..start + ENTRY_SIZE] {
            wrong += 1;
        }
    }
    format!("{wrong} of 5 rows decoded wrongly, every one with Ok")
}

#[test]
fn q_equal_to_p_is_refused_instead_of_decoding_to_a_neighbour() {
    let params = params_with_q(P, 1);
    assert_eq!(params.delta(), 1);
    if let Err(refusal) = params.validate() {
        assert!(refusal.contains("tailcut"), "{refusal}");
        return;
    }
    panic!(
        "SILENT WRONG: validate accepted q == p, and {}",
        wrong_rows_when_served(&params)
    );
}

#[test]
fn the_rounding_half_interval_must_exceed_the_sampler_tailcut() {
    let at_the_tailcut = params_with_q(Q_AT_THE_TAILCUT, 2);
    assert_eq!(at_the_tailcut.delta() / 2, 39);
    let refusal = at_the_tailcut
        .validate()
        .expect_err("a half-interval equal to the tailcut admits a wrong rounding");
    assert!(refusal.contains("tailcut"), "{refusal}");

    let one_past = params_with_q(Q_ONE_PAST_THE_TAILCUT, 2);
    assert_eq!(one_past.delta() / 2, 40);
    one_past.validate().expect("one past the tailcut");

    // The bound follows sigma: at the floor width the tailcut is 20, so 39 clears it.
    let narrower = InspireParams {
        sigma: 3.19,
        ..params_with_q(Q_AT_THE_TAILCUT, 2)
    };
    narrower.validate().expect("tailcut 20 at sigma 3.19");
}

#[test]
fn every_shipped_parameter_set_keeps_its_headroom() {
    for params in [
        InspireParams::secure_128_d2048(),
        InspireParams::secure_128_d4096(),
        InspireParams::for_scenario(1 << 20, 256, [64, 1024, 64], 1).expect("derived"),
        InspireParams::for_scenario_with_crt(
            1 << 20,
            256,
            [64, 1024, 64],
            1,
            DEFAULT_Q_2CRT_30BIT.to_vec(),
        )
        .expect("30-bit pair"),
    ] {
        assert!(params.delta() / 2 > 1 << 8, "{:?}", params.crt_moduli);
        params.validate().expect("shipped");
    }
}
