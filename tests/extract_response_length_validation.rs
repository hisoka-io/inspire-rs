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
#[ignore = "pins a live defect: extract derives num_columns from entry_size and never checks \
            response.column_ciphertexts.len(), so a short response decodes to a zero-tail record \
            with Ok. RED until extract rejects a column count below num_columns. Trigger: \
            un-ignore the moment that length check lands in pir/extract.rs."]
fn a_truncated_response_must_not_decode_to_a_zero_tail_record() {
    let f = fixture();

    for kept in [1usize, 4, NUM_COLUMNS - 1] {
        let mut truncated = f.response.clone();
        truncated.column_ciphertexts.truncate(kept);

        let outcome = extract(&f.crs, &f.state, &truncated, ENTRY_SIZE);

        let Ok(decoded) = outcome else {
            continue;
        };

        let tail_is_zero = decoded[kept * 2..].iter().all(|&b| b == 0);
        let head_matches = decoded[..kept * 2] == f.truth[..kept * 2];
        panic!(
            "extract accepted {kept} of {NUM_COLUMNS} columns and returned Ok with a \
             {len}-byte record instead of rejecting the short response.\n  \
             returned: {decoded:?}\n  \
             expected: an error naming the column-count mismatch\n  \
             the record is indistinguishable from a real one by length alone \
             (head_matches_truth={head_matches}, tail_all_zero={tail_is_zero}); \
             the caller's entry is {truth:?}",
            len = decoded.len(),
            truth = f.truth,
        );
    }
}

#[test]
#[ignore = "pins a live defect: with column_ciphertexts empty, extract decrypts the summed \
            ciphertext once and repeats that one 16-bit value across every column, returning Ok. \
            The record is non-zero, so a zero-check does not catch it. RED until extract rejects \
            an empty column list on the unpacked path. Trigger: un-ignore the moment that length \
            check lands in pir/extract.rs."]
fn an_empty_column_list_must_not_decode_to_a_repeated_column_record() {
    let f = fixture();

    let mut emptied = f.response.clone();
    emptied.column_ciphertexts.clear();

    let Ok(decoded) = extract(&f.crs, &f.state, &emptied, ENTRY_SIZE) else {
        return;
    };

    let first_column = &decoded[..2];
    let every_column_identical = decoded.chunks_exact(2).all(|c| c == first_column);
    panic!(
        "extract accepted an empty column list and returned Ok with a {len}-byte record \
         instead of rejecting it.\n  \
         returned: {decoded:?}\n  \
         expected: an error naming the empty column list\n  \
         every 16-bit column holds the same value (all_columns_identical={every_column_identical}), \
         and the bytes are non-zero, so length and emptiness checks both pass; \
         the caller's entry is {truth:?}",
        len = decoded.len(),
        truth = f.truth,
    );
}

#[test]
#[ignore = "pins a live defect: extract's `.take(num_columns)` discards surplus columns without \
            comment, so an over-length response is accepted as valid. It decodes correctly today, \
            which is why nothing catches it - the missing check is on the count, not the bytes. \
            RED until extract rejects a column count above num_columns. Trigger: un-ignore the \
            moment that length check lands in pir/extract.rs."]
fn an_over_length_response_must_not_be_silently_accepted() {
    let f = fixture();

    let mut padded = f.response.clone();
    let surplus = padded.column_ciphertexts[0].clone();
    padded.column_ciphertexts.push(surplus);

    let Ok(decoded) = extract(&f.crs, &f.state, &padded, ENTRY_SIZE) else {
        return;
    };

    panic!(
        "extract accepted {got} columns where entry_size implies {NUM_COLUMNS} and returned Ok \
         with a {len}-byte record instead of rejecting the over-length response.\n  \
         returned: {decoded:?}\n  \
         expected: an error naming the column-count mismatch\n  \
         the surplus column was dropped by `.take(num_columns)`, so a server may append \
         arbitrary ciphertexts and still be believed",
        got = padded.column_ciphertexts.len(),
        len = decoded.len(),
    );
}
