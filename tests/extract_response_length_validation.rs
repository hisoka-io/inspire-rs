//! `extract` must reject a response whose column count disagrees with `entry_size`.
//!
//! `extract` derives `num_columns` from `entry_size` alone and never compares it to
//! `response.column_ciphertexts.len()`. A server that returns too few columns, or none,
//! therefore decodes to a well-formed-looking record on the client's final step, returned
//! as `Ok` with no diagnostic - the silent-wrong class, on the last hop.
//!
//! Every other test over `extract` feeds it a response `respond` just built, so nothing
//! in the suite exercises a wrong-length one. These pin the three shapes.

// The fixture helper is not a #[test] fn, so `allow-unwrap-in-tests` does not reach it.
#![allow(clippy::unwrap_used)]

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::pir::{extract, query, respond, setup, ClientState, InspireCrs, ServerResponse};

const ENTRY_SIZE: usize = 32;
/// `(ENTRY_SIZE * 8).div_ceil(16)`, the count `extract` derives internally.
const NUM_COLUMNS: usize = 16;

fn test_params() -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: 1152921504606830593,
        crt_moduli: vec![1152921504606830593],
        p: 65536,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

struct Fixture {
    crs: InspireCrs,
    state: ClientState,
    response: ServerResponse,
    truth: Vec<u8>,
}

/// A real setup/query/respond at index 13. The payload is non-zero and non-repeating in
/// both bytes of every column, so a zero tail and a repeated column are both visible.
fn fixture() -> Fixture {
    let params = test_params();
    let num_entries = 64usize;

    let mut database = vec![0u8; num_entries * ENTRY_SIZE];
    for i in 0..num_entries {
        for j in 0..ENTRY_SIZE {
            database[i * ENTRY_SIZE + j] = u8::try_from((i * 7 + j * 11) % 251 + 1).unwrap();
        }
    }

    let mut sampler = GaussianSampler::with_seed(params.sigma, 7);
    let (crs, encoded_db, rlwe_sk) = setup(&params, &database, ENTRY_SIZE, &mut sampler).unwrap();

    let target = 13usize;
    let (state, client_query) = query(
        &crs,
        target as u64,
        &encoded_db.config,
        &rlwe_sk,
        &mut sampler,
    )
    .unwrap();
    let response = respond(&crs, &encoded_db, &client_query).unwrap();

    Fixture {
        crs,
        state,
        response,
        truth: database[target * ENTRY_SIZE..(target + 1) * ENTRY_SIZE].to_vec(),
    }
}

/// Guards the fixture the pinned tests below tamper with: if this reddens, they are
/// measuring a broken setup rather than the missing length check.
#[test]
fn untampered_response_decodes_byte_exact_and_carries_one_ciphertext_per_column() {
    let f = fixture();

    assert_eq!(
        f.response.column_ciphertexts.len(),
        NUM_COLUMNS,
        "respond must emit exactly the column count extract derives from entry_size"
    );

    let decoded = extract(&f.crs, &f.state, &f.response, ENTRY_SIZE).unwrap();
    assert_eq!(
        decoded, f.truth,
        "untampered response must decode to the stored entry"
    );
}

#[test]
fn a_truncated_response_must_not_decode_to_a_zero_tail_record() {
    let f = fixture();

    for kept in [1usize, 4, NUM_COLUMNS - 1] {
        let mut truncated = f.response.clone();
        truncated.column_ciphertexts.truncate(kept);

        let err = extract(&f.crs, &f.state, &truncated, ENTRY_SIZE)
            .expect_err("a short response must be refused");
        let message = err.to_string();
        assert!(message.contains(&format!("got {kept}")), "{message}");
        assert!(
            message.contains(&format!("expected {NUM_COLUMNS}")),
            "{message}"
        );
    }
}

#[test]
fn an_empty_column_list_must_not_decode_to_a_repeated_column_record() {
    let f = fixture();

    let mut emptied = f.response.clone();
    emptied.column_ciphertexts.clear();

    let err = extract(&f.crs, &f.state, &emptied, ENTRY_SIZE)
        .expect_err("an empty column list must be refused");
    let message = err.to_string();
    assert!(message.contains("got 0"), "{message}");
    assert!(
        message.contains(&format!("expected {NUM_COLUMNS}")),
        "{message}"
    );
}

#[test]
fn an_over_length_response_must_not_be_silently_accepted() {
    let f = fixture();

    let mut padded = f.response.clone();
    let surplus = padded.column_ciphertexts[0].clone();
    padded.column_ciphertexts.push(surplus);

    let err = extract(&f.crs, &f.state, &padded, ENTRY_SIZE)
        .expect_err("an over-length column list must be refused");
    let message = err.to_string();
    assert!(message.contains("got 17"), "{message}");
    assert!(
        message.contains(&format!("expected {NUM_COLUMNS}")),
        "{message}"
    );
}
