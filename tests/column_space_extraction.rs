#![cfg(feature = "mod-switch-response")]
#![allow(clippy::expect_used)]

use raven_inspire::math::GaussianSampler;
use raven_inspire::pir::mod_switch::{
    decode_response_packed, encode_response_packed, extract_inspiring_mod_switched,
    mod_switch_response_checked, MOD_SWITCH_TARGET_45BIT,
};
use raven_inspire::{
    extract_two_packing, query_seeded, respond_seeded_inspiring, setup, InspireParams,
    SecurityLevel,
};

fn sweep_params() -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: raven_inspire::math::mod_q::DEFAULT_Q,
        crt_moduli: vec![raven_inspire::math::mod_q::DEFAULT_Q],
        p: 65_537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

#[test]
fn every_16bit_column_round_trips_through_both_extractors() {
    const ENTRY_SIZE: usize = 256;
    const VALUES_PER_ENTRY: usize = ENTRY_SIZE / 2;
    const ENTRY_COUNT: usize = (u16::MAX as usize + 1) / VALUES_PER_ENTRY;

    let params = sweep_params();
    let mut database = Vec::with_capacity(ENTRY_COUNT * ENTRY_SIZE);
    for value in 0u16..=u16::MAX {
        database.extend_from_slice(&value.to_le_bytes());
    }
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0x1600);
    let (crs, encoded, secret_key) =
        setup(&params, &database, ENTRY_SIZE, &mut sampler).expect("setup");

    for target in 0..ENTRY_COUNT {
        let (state, query) = query_seeded(
            &crs,
            u64::try_from(target).expect("target fits u64"),
            &encoded.config,
            &secret_key,
            &mut sampler,
        )
        .expect("query");
        let response = respond_seeded_inspiring(&crs, &encoded, &query).expect("respond");
        let tight_wire = response.to_binary().expect("tight encode");
        let tight_response =
            raven_inspire::ServerResponse::from_binary(&tight_wire).expect("tight decode");
        let expected = database
            .get(target * ENTRY_SIZE..(target + 1) * ENTRY_SIZE)
            .expect("expected row");
        assert_eq!(
            extract_two_packing(&crs, &state, &tight_response, ENTRY_SIZE)
                .expect("ordinary extract"),
            expected,
            "ordinary target={target}"
        );

        let switched = mod_switch_response_checked(&params, &response, MOD_SWITCH_TARGET_45BIT)
            .expect("switch");
        let wire = encode_response_packed(&switched).expect("encode");
        let decoded = decode_response_packed(&wire).expect("decode");
        assert_eq!(
            extract_inspiring_mod_switched(&crs, &state, &decoded, ENTRY_SIZE)
                .expect("mod-switched extract"),
            expected,
            "mod-switched target={target}"
        );
    }
}
