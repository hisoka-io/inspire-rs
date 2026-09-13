#![allow(
    clippy::expect_used,
    reason = "test-target assertions; an abort is the failure report"
)]

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::pir::{setup_with_rng, ClientSession, ServerSessionHandle};

fn params() -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65_536,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

#[test]
fn installing_a_server_handle_switches_both_query_forms_off_inline_keys() {
    let params = params();
    let database = vec![0u8; params.ring_dim * 32];
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0x5110);
    let mut crs_rng = ChaCha20Rng::seed_from_u64(0x5111);
    let (crs, encoded, secret_key) =
        setup_with_rng(&params, &database, 32, &mut sampler, &mut crs_rng)
            .expect("session fixture");
    let mut session = ClientSession::new(crs, secret_key, &mut sampler).expect("client session");
    let issued = ServerSessionHandle(0x5e55_10a);

    session
        .install_server_session_handle(issued)
        .expect("install server-issued handle");
    let (_, seeded) = session
        .query_seeded(7, &encoded.config, &mut sampler)
        .expect("seeded query");
    let (_, expanded) = session
        .query(8, &encoded.config, &mut sampler)
        .expect("expanded query");

    assert_eq!(session.session_handle(), Some(issued));
    assert_eq!(seeded.session_handle, Some(issued));
    assert!(seeded.inspiring_packing_keys.is_none());
    assert_eq!(expanded.session_handle, Some(issued));
    assert!(expanded.inspiring_packing_keys.is_none());
}

#[test]
fn installing_a_handle_without_packing_material_is_refused() {
    let params = params();
    let database = vec![0u8; params.ring_dim * 32];
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0x5120);
    let mut crs_rng = ChaCha20Rng::seed_from_u64(0x5121);
    let (mut crs, _encoded, secret_key) =
        setup_with_rng(&params, &database, 32, &mut sampler, &mut crs_rng)
            .expect("session fixture");
    crs.inspiring_num_columns = 0;
    let mut session = ClientSession::new(crs, secret_key, &mut sampler).expect("tree-only session");

    let error = session
        .install_server_session_handle(ServerSessionHandle(9))
        .expect_err("a handle cannot replace absent packing material");
    let message = error.to_string();
    assert!(message.contains("no InspiRING packing keys"), "{message}");
    assert_eq!(session.session_handle(), None);
}
