//! The extractor the clients call reads its modulus off the wire. Everything a server can
//! put there that this build cannot decrypt under is refused, in bounded time, and the two
//! things it can -- an unswitched response and an implemented target -- decode.

#![cfg(feature = "mod-switch-response")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

use std::sync::mpsc;
use std::time::Duration;

use raven_inspire::math::{GaussianSampler, Poly};
use raven_inspire::params::DEFAULT_Q_2CRT_30BIT;
use raven_inspire::pir::mod_switch::{
    extract_inspiring_mod_switched, mod_switch_response_checked, MOD_SWITCH_TARGET_36BIT,
    MOD_SWITCH_TARGET_45BIT,
};
use raven_inspire::rlwe::RlweCiphertext;
use raven_inspire::{
    query_seeded, respond_seeded_inspiring, setup, ClientState, InspireParams, PackingMode,
    SecurityLevel, ServerCrs, ServerResponse,
};

const ENTRY_SIZE: usize = 32;
const TARGET_INDEX: u64 = 42;
/// `162,739 * 422,267`: 36 bits, `== 1 mod 512`, at or above `p`, below `q`. It passes every
/// arithmetic precondition the switch has and no primitive root of it exists to be found.
const COMPOSITE_36BIT: u64 = 68_719_309_313;
/// Far longer than a refusal needs, far shorter than a `2..q` scan.
const REFUSAL_BOUND: Duration = Duration::from_secs(20);

fn params_with(crt_moduli: Vec<u64>) -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: crt_moduli.iter().product(),
        crt_moduli,
        p: 65_537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

struct Served {
    crs: ServerCrs,
    state: ClientState,
    response: ServerResponse,
    row: Vec<u8>,
}

fn serve(params: &InspireParams) -> Served {
    let database: Vec<u8> = (0..params.ring_dim * ENTRY_SIZE)
        .map(|i| ((i * 17 + 3) % 251) as u8)
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 37);
    let (crs, encoded_db, sk) = setup(params, &database, ENTRY_SIZE, &mut sampler).expect("setup");
    let (state, query) =
        query_seeded(&crs, TARGET_INDEX, &encoded_db.config, &sk, &mut sampler).expect("query");
    let response = respond_seeded_inspiring(&crs, &encoded_db, &query).expect("respond");
    let start = usize::try_from(TARGET_INDEX).unwrap() * ENTRY_SIZE;
    Served {
        crs,
        state,
        response,
        row: database[start..start + ENTRY_SIZE].to_vec(),
    }
}

fn single_limb() -> Served {
    serve(&params_with(vec![raven_inspire::math::mod_q::DEFAULT_Q]))
}

fn switched(served: &Served, target: u64) -> ServerResponse {
    mod_switch_response_checked(&served.crs.params, &served.response, target).expect("switch")
}

/// The same coefficients relabelled under another modulus, as a hostile server would send.
fn relabelled(response: &ServerResponse, build: impl Fn(Vec<u64>) -> Poly) -> ServerResponse {
    ServerResponse {
        ciphertext: RlweCiphertext::from_parts(
            build(response.ciphertext.a.coeffs().to_vec()),
            build(response.ciphertext.b.coeffs().to_vec()),
        ),
        column_ciphertexts: Vec::new(),
        packing_mode: response.packing_mode,
        packed_coefficients: response.packed_coefficients,
    }
}

/// Runs the extractor off-thread so a non-terminating one fails the test instead of the run.
fn extract_within_bound(
    served: Served,
    response: ServerResponse,
) -> Option<Result<Vec<u8>, String>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let outcome =
            extract_inspiring_mod_switched(&served.crs, &served.state, &response, ENTRY_SIZE)
                .map_err(|e| e.to_string());
        let _ = tx.send(outcome);
    });
    rx.recv_timeout(REFUSAL_BOUND).ok()
}

#[test]
fn an_implemented_target_and_an_unswitched_response_both_decode() {
    for target in [MOD_SWITCH_TARGET_36BIT, MOD_SWITCH_TARGET_45BIT] {
        let served = single_limb();
        let response = switched(&served, target);
        let got = extract_inspiring_mod_switched(&served.crs, &served.state, &response, ENTRY_SIZE)
            .expect("implemented target");
        assert_eq!(got, served.row, "target {target}");
    }
    let served = single_limb();
    let got =
        extract_inspiring_mod_switched(&served.crs, &served.state, &served.response, ENTRY_SIZE)
            .expect("unswitched");
    assert_eq!(got, served.row);
}

/// A two-limb parameter set that `validate()` admits. Its unswitched response carries both
/// limbs; sending it down the single-limb path panics on mismatched moduli.
#[test]
fn an_honest_unswitched_two_limb_response_decodes() {
    let served = serve(&params_with(DEFAULT_Q_2CRT_30BIT.to_vec()));
    assert_eq!(served.response.ciphertext.a.crt_count(), 2);
    let got =
        extract_inspiring_mod_switched(&served.crs, &served.state, &served.response, ENTRY_SIZE)
            .expect("two-limb unswitched");
    assert_eq!(got, served.row);
}

#[test]
fn a_composite_modulus_is_refused_in_bounded_time() {
    let served = single_limb();
    let base = switched(&served, MOD_SWITCH_TARGET_36BIT);
    let forged = relabelled(&base, |coeffs| {
        Poly::from_coeffs(
            coeffs.into_iter().map(|c| c % COMPOSITE_36BIT).collect(),
            COMPOSITE_36BIT,
        )
    });
    let outcome = extract_within_bound(served, forged)
        .expect("the extractor must return: a composite modulus has no primitive root to find");
    let err = outcome.expect_err("a composite modulus must not decode");
    assert!(err.contains("68719309313"), "{err}");
    assert!(err.contains("implemented mod-switch target"), "{err}");
}

#[test]
fn a_prime_target_this_build_has_no_constant_for_is_refused() {
    // `2^34 - 3 * 2^16 + 1`: prime, NTT-friendly, passes the noise gate. Not implemented.
    let unlisted = (1u64 << 34) - 3 * (1u64 << 16) + 1;
    let served = single_limb();
    let base = switched(&served, MOD_SWITCH_TARGET_36BIT);
    let forged = relabelled(&base, |coeffs| {
        Poly::from_coeffs(coeffs.into_iter().map(|c| c % unlisted).collect(), unlisted)
    });
    let err = extract_inspiring_mod_switched(&served.crs, &served.state, &forged, ENTRY_SIZE)
        .expect_err("an unlisted modulus must not decode")
        .to_string();
    assert!(err.contains("implemented mod-switch target"), "{err}");
}

#[test]
fn a_forged_two_limb_response_is_refused_not_panicked_on() {
    let served = single_limb();
    let base = switched(&served, MOD_SWITCH_TARGET_36BIT);
    let forged = relabelled(&base, |coeffs| {
        let limbs: Vec<u64> = DEFAULT_Q_2CRT_30BIT
            .iter()
            .flat_map(|&m| coeffs.iter().map(move |&c| c % m))
            .collect();
        Poly::from_crt_coeffs(limbs, &DEFAULT_Q_2CRT_30BIT)
    });
    let outcome = std::panic::catch_unwind(|| {
        extract_inspiring_mod_switched(&served.crs, &served.state, &forged, ENTRY_SIZE)
            .map_err(|e| e.to_string())
    })
    .expect("a forged limb count must be refused, not panic the client");
    let err = outcome.expect_err("two limbs the CRS does not have must not decode");
    assert!(err.contains("2 limb(s)"), "{err}");
}

#[test]
fn tree_packed_and_untagged_responses_are_refused() {
    for mode in [Some(PackingMode::Tree), None] {
        let served = single_limb();
        let mut response = switched(&served, MOD_SWITCH_TARGET_36BIT);
        response.packing_mode = mode;
        let err = extract_inspiring_mod_switched(&served.crs, &served.state, &response, ENTRY_SIZE)
            .expect_err("only InspiRING output decodes here")
            .to_string();
        assert!(err.contains("packing_mode=Inspiring"), "{mode:?}: {err}");
    }
}

#[test]
fn a_retained_prefix_one_short_or_one_long_is_refused() {
    let served = single_limb();
    let exact = switched(&served, MOD_SWITCH_TARGET_36BIT)
        .packed_coefficients
        .expect("packed response");
    for retained in [exact - 1, exact + 1] {
        let served = single_limb();
        let mut response = switched(&served, MOD_SWITCH_TARGET_36BIT);
        response.packed_coefficients = Some(retained);
        let err = extract_inspiring_mod_switched(&served.crs, &served.state, &response, ENTRY_SIZE)
            .expect_err("a partial or surplus prefix must not decode")
            .to_string();
        assert!(
            err.contains("packed coefficient prefix"),
            "{retained}: {err}"
        );
    }
}

#[test]
fn a_response_from_another_ring_is_refused() {
    let served = single_limb();
    let base = switched(&served, MOD_SWITCH_TARGET_36BIT);
    let forged = relabelled(&base, |coeffs| {
        Poly::from_coeffs(coeffs[..128].to_vec(), MOD_SWITCH_TARGET_36BIT)
    });
    let err = extract_inspiring_mod_switched(&served.crs, &served.state, &forged, ENTRY_SIZE)
        .expect_err("a 128-coefficient response cannot belong to a 256 ring")
        .to_string();
    assert!(err.contains("ring_dim"), "{err}");
}

/// The client re-runs the gate against ITS params: an implemented target that is too loud
/// for this plaintext modulus is refused rather than decrypted to the wrong row.
#[test]
fn the_noise_gate_runs_on_the_extracting_side() {
    let mut served = single_limb();
    let response = switched(&served, MOD_SWITCH_TARGET_36BIT);
    served.crs.params.p = 1 << 22;
    let err = extract_inspiring_mod_switched(&served.crs, &served.state, &response, ENTRY_SIZE)
        .expect_err("36 bits cannot carry a 22-bit plaintext modulus")
        .to_string();
    assert!(err.contains("noise-budget violation"), "{err}");
}

/// The producing side refuses the same switch, so a loud target never reaches a wire.
#[test]
fn the_checked_switch_refuses_a_target_the_gate_refuses() {
    let mut served = single_limb();
    served.crs.params.p = 1 << 22;
    let err = mod_switch_response_checked(
        &served.crs.params,
        &served.response,
        MOD_SWITCH_TARGET_36BIT,
    )
    .expect_err("the checked switch must run the gate")
    .to_string();
    assert!(err.contains("noise-budget violation"), "{err}");
}
