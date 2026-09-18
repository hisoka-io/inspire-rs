#![allow(
    clippy::expect_used,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::pir::{query_seeded, respond_seeded_inspiring_cached, setup};
use raven_inspire::ServerInspiringCache;

fn params() -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65_537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

#[test]
fn fresh_setup_cache_is_moved_without_recomputation() {
    let params = params();
    let database = vec![0x5a; params.ring_dim * 32];
    let mut sampler = GaussianSampler::with_seed(params.sigma, 1);
    let (mut crs, encoded, secret_key) =
        setup(&params, &database, 32, &mut sampler).expect("setup");
    let expected_params = bincode::serialize(
        crs.inspiring_pack_params
            .as_ref()
            .expect("setup pack params"),
    )
    .expect("serialize params");
    let expected_keys = bincode::serialize(
        crs.inspiring_packing_key
            .as_ref()
            .expect("setup packing key"),
    )
    .expect("serialize keys");

    let moved = ServerInspiringCache::from_setup(&mut crs, &encoded).expect("move setup cache");
    assert!(crs.inspiring_pack_params.is_none());
    assert!(crs.inspiring_packing_key.is_none());
    assert_eq!(
        bincode::serialize(moved.pack_params()).expect("serialize moved params"),
        expected_params
    );
    assert_eq!(
        bincode::serialize(moved.offline_keys()).expect("serialize moved keys"),
        expected_keys
    );

    let rebuilt = ServerInspiringCache::new(&crs, &encoded).expect("independent rebuild");
    let mut query_sampler = GaussianSampler::with_seed(params.sigma, 2);
    let (_, query) =
        query_seeded(&crs, 7, &encoded.config, &secret_key, &mut query_sampler).expect("query");
    let moved_response =
        respond_seeded_inspiring_cached(&crs, &encoded, &query, &moved).expect("moved response");
    let rebuilt_response = respond_seeded_inspiring_cached(&crs, &encoded, &query, &rebuilt)
        .expect("rebuilt response");
    assert_eq!(
        bincode::serialize(&moved_response).expect("serialize moved response"),
        bincode::serialize(&rebuilt_response).expect("serialize rebuilt response")
    );
}

#[test]
fn setup_cache_refuses_wrong_database_width_before_taking() {
    let params = params();
    let database = vec![0x5a; params.ring_dim * 32];
    let mut sampler = GaussianSampler::with_seed(params.sigma, 3);
    let (mut crs, mut encoded, _) = setup(&params, &database, 32, &mut sampler).expect("setup");

    let removed = encoded.shards[0].polynomials.pop().expect("one column");
    let short = ServerInspiringCache::from_setup(&mut crs, &encoded)
        .expect_err("short database width must fail");
    assert!(short.to_string().contains("column count"), "{short}");
    assert!(crs.inspiring_pack_params.is_some());
    assert!(crs.inspiring_packing_key.is_some());

    encoded.shards[0].polynomials.push(removed.clone());
    encoded.shards[0].polynomials.push(removed);
    let surplus = ServerInspiringCache::from_setup(&mut crs, &encoded)
        .expect_err("surplus database width must fail");
    assert!(surplus.to_string().contains("column count"), "{surplus}");
    assert!(crs.inspiring_pack_params.is_some());
    assert!(crs.inspiring_packing_key.is_some());
}

#[test]
fn from_setup_reachable_source_contains_no_cache_rebuild() {
    let source = include_str!("../src/pir/respond.rs");
    let function_body = |name: &str| {
        let start = source.find(name).expect("function source must exist");
        let tail = &source[start..];
        let body_start = tail.find('{').expect("function body must start");
        let mut depth = 0usize;
        for (offset, byte) in tail.as_bytes()[body_start..].iter().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &tail[..=body_start + offset];
                    }
                }
                _ => {}
            }
        }
        panic!("function body must end");
    };
    let validation = function_body("fn validate_cache_parts");
    let rebuild = function_body("pub fn new");
    let from_setup = function_body("pub fn from_setup");
    let reachable = format!("{validation}\n{from_setup}");
    assert!(rebuild.contains("PackParams::try_new"), "{rebuild}");
    assert!(
        rebuild.contains("OfflinePackingKeys::generate"),
        "{rebuild}"
    );
    assert!(!reachable.contains("PackParams::try_new"), "{reachable}");
    assert!(
        !reachable.contains("OfflinePackingKeys::generate"),
        "{reachable}"
    );
    assert!(
        reachable.contains("inspiring_pack_params.take()"),
        "{reachable}"
    );
    assert!(
        reachable.contains("inspiring_packing_key.take()"),
        "{reachable}"
    );
}
