//! A `ServerCrs` carries `InspireParams` as plain wire data, and `ntt_context`
//! asserts rather than returns. Ingest must reject params the algebra cannot
//! run, or a hostile CRS traps the WASM client instead of erroring.

#![allow(
    clippy::expect_used,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel, ShardConfig};
use raven_inspire::pir::{encode_database, setup, ClientSession, ServerCrs};

fn params_at(ring_dim: usize) -> InspireParams {
    InspireParams {
        ring_dim,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65536,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

fn honest_crs(entry_size: usize) -> (ServerCrs, raven_inspire::rlwe::RlweSecretKey) {
    let params = params_at(256);
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let database = vec![7u8; params.ring_dim * entry_size];
    let (crs, _db, sk) =
        setup(&params, &database, entry_size, &mut sampler).expect("legal records must set up");
    (crs, sk)
}

/// The width stays legal, so only a params check can catch this. Decoding a
/// modulus outside `q = 1 (mod 2d)` and then building a session reaches the
/// NTT assertion.
#[test]
fn a_crt_modulus_outside_the_ntt_congruence_is_rejected_at_ingest() {
    let (mut crs, _sk) = honest_crs(32);
    crs.params.crt_moduli = vec![1_152_921_504_606_830_591];
    crs.params.q = 1_152_921_504_606_830_591;
    let bytes = crs.to_versioned_bytes().expect("serialize");

    let err = ServerCrs::from_versioned_bytes(&bytes)
        .expect_err("a modulus the NTT cannot use must not decode into a usable CRS");
    let message = err.to_string();
    assert!(
        message.contains("InspireParams"),
        "error must name the params as the offending field, got: {message}"
    );
}

#[test]
fn a_ring_dim_that_is_not_a_power_of_two_is_rejected_at_ingest() {
    let (mut crs, _sk) = honest_crs(32);
    crs.params.ring_dim = 24;
    let bytes = crs.to_versioned_bytes().expect("serialize");

    assert!(
        ServerCrs::from_versioned_bytes(&bytes).is_err(),
        "a non-power-of-two ring dimension must not decode into a usable CRS"
    );
}

/// The WASM client path. `ClientSession::new` must surface a hostile CRS as a
/// typed error; a trap there kills the session with no recoverable signal.
#[test]
fn a_hostile_crs_reaches_client_session_as_an_error_not_a_trap() {
    let (mut crs, sk) = honest_crs(32);
    crs.params.crt_moduli = vec![1_152_921_504_606_830_591];
    crs.params.q = 1_152_921_504_606_830_591;

    let mut sampler = GaussianSampler::with_seed(6.4, 1);
    assert!(
        ClientSession::new(crs, sk, &mut sampler).is_err(),
        "ClientSession::new must reject params the NTT cannot run"
    );
}

#[test]
fn setup_rejects_a_zero_entry_size_before_dividing_by_it() {
    let params = params_at(256);
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let database = vec![7u8; 1024];

    let err = setup(&params, &database, 0, &mut sampler)
        .expect_err("entry_size 0 divides the database by zero");
    assert!(
        err.to_string().contains("entry_size"),
        "error must name entry_size, got: {err}"
    );
}

#[test]
fn a_forged_shard_config_is_refused_before_database_encoding() {
    let params = params_at(256);
    let forged = ShardConfig {
        shard_size_bytes: params.ring_dim as u64,
        entry_size_bytes: 1,
        total_entries: u64::MAX,
    };
    let wire = bincode::serialize(&forged).expect("serialize forged ShardConfig");
    let decoded: ShardConfig = bincode::deserialize(&wire).expect("decode forged ShardConfig");

    let error = encode_database(&[], 1, &params, &decoded)
        .expect_err("a wire-decoded config with overflowing shard ids must be refused");

    assert!(
        error.to_string().contains("num_shards exceeds u32::MAX"),
        "the refusal must identify the forged shard count: {error}"
    );
}
