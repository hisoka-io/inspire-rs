//! An illegal `inspiring_num_columns` from a server-supplied CRS must surface
//! as a typed error: a trap on the WASM client path kills the session with no
//! recoverable signal.

#![allow(
    clippy::expect_used,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel, ShardConfig};
use raven_inspire::pir::{query, query_seeded, setup, ClientSession, ServerCrs};
use raven_inspire::rlwe::RlweSecretKey;

const ILLEGAL_WIDTH: usize = 24;

fn params_at(ring_dim: usize) -> InspireParams {
    InspireParams {
        ring_dim,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65536,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

fn honest_setup(
    entry_size: usize,
) -> (ServerCrs, RlweSecretKey, raven_inspire::params::ShardConfig) {
    let params = params_at(256);
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let database = vec![7u8; params.ring_dim * entry_size];
    let (crs, encoded_db, sk) =
        setup(&params, &database, entry_size, &mut sampler).expect("32-byte records must set up");
    (crs, sk, encoded_db.config)
}

#[test]
fn mutated_crs_width_is_rejected_at_deserialization() {
    let (mut crs, _sk, _config) = honest_setup(32);
    crs.inspiring_num_columns = ILLEGAL_WIDTH;
    let bytes = crs.to_versioned_bytes().expect("serialize");

    let err = ServerCrs::from_versioned_bytes(&bytes)
        .expect_err("an illegal InspiRING width must not decode into a usable CRS");
    let message = err.to_string();
    assert!(
        message.contains("inspiring_num_columns 24") && message.contains("legal widths"),
        "error must name the offending width and the legal set, got: {message}"
    );
}

#[test]
fn tree_packing_only_crs_still_deserializes() {
    let (mut crs, _sk, _config) = honest_setup(32);
    crs.inspiring_num_columns = 0;
    let bytes = crs.to_versioned_bytes().expect("serialize");

    let decoded = ServerCrs::from_versioned_bytes(&bytes)
        .expect("width 0 means tree packing only and must stay accepted");
    assert_eq!(decoded.inspiring_num_columns, 0);
}

#[test]
fn zero_width_crs_is_refused_by_both_free_query_builders() {
    let (mut crs, sk, config) = honest_setup(32);
    crs.inspiring_num_columns = 0;
    let mut sampler = GaussianSampler::with_seed(crs.params.sigma, 91);

    for message in [
        query(&crs, 3, &config, &sk, &mut sampler)
            .expect_err("unseeded query must refuse zero width")
            .to_string(),
        query_seeded(&crs, 3, &config, &sk, &mut sampler)
            .expect_err("seeded query must refuse zero width")
            .to_string(),
    ] {
        assert!(message.contains("zero InspiRING width"), "{message}");
        assert!(message.contains("tree-packed"), "{message}");
        assert!(message.contains("TwoPacking"), "{message}");
    }
}

#[test]
fn every_query_builder_refuses_mismatched_shard_geometry() {
    let (crs, sk, _) = honest_setup(32);
    for entries_per_shard in [crs.ring_dim() - 1, crs.ring_dim() + 1] {
        let config = ShardConfig {
            shard_size_bytes: (entries_per_shard * 32) as u64,
            entry_size_bytes: 32,
            total_entries: crs.ring_dim() as u64,
        };
        let mut session_sampler = GaussianSampler::with_seed(crs.params.sigma, 90);
        let session = ClientSession::new(crs.clone(), sk.clone(), &mut session_sampler)
            .expect("the CRS itself is valid");
        let mut sampler = GaussianSampler::with_seed(crs.params.sigma, 91);
        let messages = [
            query(&crs, 256, &config, &sk, &mut sampler)
                .err()
                .map(|error| error.to_string())
                .unwrap_or_default(),
            query_seeded(&crs, 256, &config, &sk, &mut sampler)
                .err()
                .map(|error| error.to_string())
                .unwrap_or_default(),
            session
                .query(256, &config, &mut sampler)
                .err()
                .map(|error| error.to_string())
                .unwrap_or_default(),
            session
                .query_seeded(256, &config, &mut sampler)
                .err()
                .map(|error| error.to_string())
                .unwrap_or_default(),
        ];
        for message in messages {
            assert!(message.contains("shard geometry"), "{message}");
            assert!(message.contains("ring_dim"), "{message}");
        }
    }
}

#[test]
fn honest_crs_round_trips_and_hostile_blobs_hit_the_magic_and_size_guards() {
    let (crs, _sk, _config) = honest_setup(32);
    let bytes = crs.to_versioned_bytes().expect("serialize");
    assert_eq!(
        &bytes[..16],
        ServerCrs::MAGIC.as_slice(),
        "the blob must carry the version magic prefix"
    );

    let decoded = ServerCrs::from_versioned_bytes(&bytes).expect("an honest CRS must decode");
    assert_eq!(decoded.inspiring_num_columns, 16);

    let raw = bincode::serialize(&crs).expect("raw serialize");
    let err = ServerCrs::from_versioned_bytes(&raw)
        .expect_err("an unversioned blob must fail the magic check");
    assert!(
        err.to_string().contains("magic mismatch"),
        "expected a magic-mismatch error, got: {err}"
    );
    assert!(ServerCrs::check_magic(&[0u8; 4]).is_err());

    // the length cap must reject before bincode allocates
    let mut oversize = ServerCrs::MAGIC.to_vec();
    oversize.resize(16 + ServerCrs::DECODE_LIMIT_BYTES + 1, 0);
    let err = ServerCrs::from_versioned_bytes(&oversize)
        .expect_err("a body over the decode cap must be rejected");
    assert!(
        err.to_string().contains("too large"),
        "expected the decode-cap error, got: {err}"
    );
}

#[test]
fn client_session_new_errors_on_a_mutated_crs_width() {
    let (mut crs, sk, _config) = honest_setup(32);
    crs.inspiring_num_columns = ILLEGAL_WIDTH;
    let mut sampler = GaussianSampler::with_seed(crs.params.sigma, 0);

    let err = ClientSession::new(crs, sk, &mut sampler)
        .expect_err("ClientSession::new must reject an illegal width instead of trapping");
    assert!(
        err.to_string().contains("InspiRING"),
        "error must name the failing subsystem, got: {err}"
    );
}

#[test]
fn query_errors_on_a_mutated_crs_width() {
    let (mut crs, sk, config) = honest_setup(32);
    crs.inspiring_num_columns = ILLEGAL_WIDTH;
    let mut sampler = GaussianSampler::with_seed(crs.params.sigma, 0);

    let err = query(&crs, 3, &config, &sk, &mut sampler)
        .expect_err("query must reject an illegal width instead of trapping");
    assert!(
        err.to_string().contains("InspiRING"),
        "error must name the failing subsystem, got: {err}"
    );
}

/// Width is derived as ceil(entry_size / 2), so the error names entry sizes.
#[test]
fn setup_rejects_an_entry_size_whose_derived_width_is_illegal() {
    let params = params_at(256);
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let entry_size = 6usize;
    let database = vec![7u8; params.ring_dim * entry_size];

    let err = setup(&params, &database, entry_size, &mut sampler)
        .expect_err("entry_size 6 derives width 3, which has no packing generator");
    let message = err.to_string();
    assert!(
        message.contains("entry_size 6") && message.contains("legal entry sizes"),
        "error must name the offending entry size and the legal set, got: {message}"
    );
}

#[test]
fn setup_accepts_every_entry_size_whose_derived_width_is_legal() {
    let params = params_at(256);
    for entry_size in [1usize, 2, 3, 4, 7, 8, 31, 32] {
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
        let database = vec![7u8; params.ring_dim * entry_size];
        assert!(
            setup(&params, &database, entry_size, &mut sampler).is_ok(),
            "entry_size {entry_size} derives a power-of-two width and must set up"
        );
    }
}

#[test]
fn setup_and_encoder_round_an_odd_entry_width_to_the_same_column_count() {
    let params = params_at(256);
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let entry_size = 3usize;
    let database = vec![7u8; params.ring_dim * entry_size];

    let (crs, encoded_db, _) =
        setup(&params, &database, entry_size, &mut sampler).expect("entry_size 3 is legal");

    assert_eq!(crs.inspiring_num_columns, 2);
    assert_eq!(encoded_db.shards[0].polynomials.len(), 2);
}
