//! `cost.rs`'s communication estimate, tied to the bytes that actually cross
//! the wire.
//!
//! `CostEstimator` is a coefficient-payload model: every polynomial is
//! `ring_dim * crt_limbs * 8` bytes and serde framing is out of scope. That is
//! a legitimate model, but nothing in the tree ever compared it to a real
//! serialized query, and the packing-key term was wrong by a factor of
//! `gamma / gadget_len` — `y_body` carries `gadget_len` polynomials, not
//! `gamma` (`src/pir/session.rs` refuses a query whose
//! `y_body.len() != pack_params.gadget.len`; `inspiring2.rs` builds it from
//! `generate_ksk_body(.., &pack_params.gadget, ..)`). At the shipped
//! `secure_128_d2048` cell that is 262,144 B claimed against 49,324 B on the
//! wire, and 4,194,304 B against the same 49,324 B at a 512-byte record.
//!
//! The framing constants below are DERIVED from the independently built closed
//! form in `crates/client/tests/query_generation_budget.rs`, not fitted to a
//! measurement here - a constant fitted to the thing it checks is a mirror
//! oracle. Its L2 anchor is 49,445 B for a `secure_128_d2048` query with a
//! session handle.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]

use raven_inspire::cost::CostEstimator;
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, InspireVariant, SecurityLevel};
use raven_inspire::pir::{query_seeded, respond_seeded_inspiring, setup, PackingMode};

/// Bytes bincode 1.3 legacy fixint spends on one `Poly` besides its
/// coefficients: `moduli` (8 length + 8 per limb), `q`, `dim`,
/// `crt_q0_inv_mod_q1`, `is_ntt`, and the `coeffs` length prefix.
/// `poly_bytes(d,k) = 8*d*k + (8*k + 41)` in `query_generation_budget.rs`.
const fn poly_framing(crt_limbs: usize) -> usize {
    8 * crt_limbs + 41
}

/// `SeededClientQuery` framing: `shard_id` (4), the seeded RGSW container
/// (`8` row-vector length + `24` trailer + `32` seed and one poly header per
/// each of its `ell` rows), `packing_mode` (4), the inlined
/// `ClientPackingKeys` (`Option` tag + `25` header + one poly header per each
/// of its `ell` rows) and the empty `session_handle` (1).
const fn seeded_query_framing(crt_limbs: usize, gadget_len: usize) -> usize {
    4 + (8 + 24 + gadget_len * (32 + poly_framing(crt_limbs)))
        + 4
        + (1 + 25 + gadget_len * poly_framing(crt_limbs))
        + 1
}

/// Packed response framing: enum tag, full-a header, b-prefix length, retained
/// count, empty column vector, and `Some(PackingMode)`.
const fn response_framing(crt_limbs: usize) -> usize {
    4 + poly_framing(crt_limbs) + 8 + 4 + 8 + 1 + 4
}

/// Full `a` plus the plaintext-bearing prefix of `b`.
const fn packed_response_payload(ring_dim: usize, gamma: usize, crt_limbs: usize) -> u64 {
    ((ring_dim + gamma) * crt_limbs * 8) as u64
}

/// `query_generation_budget.rs::query_bytes_with_inlined_keys(2048, 1, 3)`.
/// The inlined compatibility form is 49,445 B of seeded handle query plus
/// 49,324 B of packing keys, less the absent 8-byte handle.
const INLINED_QUERY_WIRE_BYTES: usize = 98_761;
/// The same closed form's response prediction at the shipped cell.
const SHIPPED_RESPONSE_WIRE_BYTES: usize = 16_590;

fn small_params() -> InspireParams {
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

/// A real seeded InspiRING query and its response at the fixture shape, plus
/// the estimator's opinion of the same cell.
struct Measured {
    query_wire: usize,
    response_wire: usize,
    estimated_communication: u64,
}

fn measure(params: &InspireParams, entry_size: usize) -> Measured {
    let num_entries = 64usize;
    let mut database = vec![0u8; num_entries * entry_size];
    for (i, byte) in database.iter_mut().enumerate() {
        *byte = ((i * 31 + 7) % 251) as u8;
    }

    let mut sampler = GaussianSampler::with_seed(params.sigma, 0xC057);
    let (crs, encoded_db, rlwe_sk) =
        setup(params, &database, entry_size, &mut sampler).expect("setup must succeed");

    let (_state, seeded_query) = query_seeded(&crs, 7, &encoded_db.config, &rlwe_sk, &mut sampler)
        .expect("seeded query must succeed");

    // The framing model above assumes the InspiRING shape: keys inlined, no
    // session handle. Assert the premises rather than trusting them.
    assert_eq!(
        seeded_query.packing_mode,
        PackingMode::Inspiring,
        "fixture must exercise the InspiRING (TwoPacking) wire shape"
    );
    assert!(
        seeded_query.inspiring_packing_keys.is_some(),
        "the measured compatibility query must inline the packing keys the model prices"
    );
    assert!(
        seeded_query.session_handle.is_none(),
        "a session handle would replace the inlined keys and change the model"
    );
    let keys = seeded_query.inspiring_packing_keys.as_ref().unwrap();
    assert_eq!(
        keys.y_body.len(),
        params.gadget_len,
        "y_body carries one polynomial per gadget digit - this is the row count \
         cost.rs must use, and the reason the gamma form was wrong"
    );
    assert!(
        keys.z_body.is_empty(),
        "half packing ships no z_body; the framing model prices an empty vector"
    );

    let response =
        respond_seeded_inspiring(&crs, &encoded_db, &seeded_query).expect("respond must succeed");
    assert!(
        response.column_ciphertexts.is_empty(),
        "the packed path returns one ciphertext; the framing model prices an empty vector"
    );
    assert!(
        response.packing_mode.is_some(),
        "the framing model prices Some(PackingMode)"
    );

    let estimator = CostEstimator::new(params, InspireVariant::TwoPacking, entry_size);

    Measured {
        query_wire: bincode::serialize(&seeded_query)
            .expect("query must serialize")
            .len(),
        response_wire: response.to_binary().expect("response must serialize").len(),
        estimated_communication: estimator.estimate().communication.bytes,
    }
}

/// Pinning the R1 response estimate is what lets the query half be isolated
/// from `communication.bytes` below.
#[test]
fn cost_model_response_bytes_match_the_wire() {
    let params = small_params();
    let m = measure(&params, 32);

    let modeled = packed_response_payload(
        params.ring_dim,
        raven_inspire::num_columns(32),
        params.crt_moduli.len(),
    ) as usize;
    assert_eq!(
        modeled + response_framing(params.crt_moduli.len()),
        m.response_wire,
        "modeled response payload {modeled} + framing must equal the {} bytes the \
         server actually sent",
        m.response_wire
    );
}

/// D2: the estimator's query term against a real serialized query.
#[test]
fn cost_model_query_bytes_match_the_wire() {
    let params = small_params();
    let k = params.crt_moduli.len();
    let m = measure(&params, 32);

    // Subtracting the response term (pinned to the wire by the test above) is
    // what isolates the query term; `communication.bytes` carries both. A wrap
    // here would read as a wildly wrong byte count instead of an underflow.
    let modeled_query = m
        .estimated_communication
        .checked_sub(packed_response_payload(
            params.ring_dim,
            raven_inspire::num_columns(32),
            k,
        ))
        .expect("communication.bytes must at least cover one packed RLWE response");
    let predicted_wire = modeled_query as usize + seeded_query_framing(k, params.gadget_len);

    assert_eq!(
        predicted_wire,
        m.query_wire,
        "cost.rs prices this query at {modeled_query} payload bytes (+{} framing) = \
         {predicted_wire} B, but the wire carries {} B. y_body has gadget_len ({}) \
         polynomials, not gamma.",
        seeded_query_framing(k, params.gadget_len),
        m.query_wire,
        params.gadget_len
    );
}

/// The same identity for the inlined form at the shipped parameter cell, against the closed form in
/// `crates/client/tests/query_generation_budget.rs` rather than a local
/// measurement. Analytic on both sides: no d=2048 crypto runs here.
#[test]
fn cost_model_total_matches_the_inlined_2048_cell() {
    let params = InspireParams::secure_128_d2048();
    let k = params.crt_moduli.len();
    assert_eq!(
        k, 1,
        "the client-side closed form is anchored at one CRT limb"
    );
    assert_eq!(params.gadget_len, 3, "and at gadget_len 3");
    assert_eq!(params.ring_dim, 2048, "and at ring_dim 2048");

    let estimated = CostEstimator::new(&params, InspireVariant::TwoPacking, 32)
        .estimate()
        .communication
        .bytes;

    let predicted =
        estimated as usize + seeded_query_framing(k, params.gadget_len) + response_framing(k);

    assert_eq!(
        predicted,
        INLINED_QUERY_WIRE_BYTES + SHIPPED_RESPONSE_WIRE_BYTES,
        "cost.rs totals {estimated} payload bytes for the inlined form at the shipped cell, which framed is \
         {predicted} B against the {} B the independently derived client model predicts. \
         The 49,445 B registered-query anchor and 49,324 B packing-key payload are exact; a disagreement is the \
         estimator's.",
        INLINED_QUERY_WIRE_BYTES + SHIPPED_RESPONSE_WIRE_BYTES
    );
}

/// Only the retained response prefix scales with record width.
#[test]
fn two_packing_communication_tracks_the_retained_response_prefix() {
    let params = InspireParams::secure_128_d2048();
    let at = |entry_size| {
        CostEstimator::new(&params, InspireVariant::TwoPacking, entry_size)
            .estimate()
            .communication
            .bytes
    };

    let baseline = at(32);
    let baseline_gamma = raven_inspire::num_columns(32) as u64;
    let limbs = params.crt_moduli.len() as u64;
    for entry_size in [2usize, 8, 64, 128, 512] {
        let gamma = raven_inspire::num_columns(entry_size) as u64;
        assert_eq!(
            at(entry_size),
            baseline - baseline_gamma * limbs * 8 + gamma * limbs * 8,
            "only the retained b prefix may move with record width {entry_size}"
        );
    }
}
