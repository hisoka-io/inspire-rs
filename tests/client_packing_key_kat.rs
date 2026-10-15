//! The uploaded packing key, byte for byte, from fixed seeds: every shipped modulus set
//! at seven packing widths. `tests/fixtures/client_packing_key_kat.txt` was recorded
//! from the generator that also expanded the gamma-1 rotations and took a `PackParams`;
//! the key bytes and the sampler position after the call must not have moved.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-target harness; an abort here is the failure report"
)]

mod packing_key_fixture;

use packing_key_fixture::{fnv1a, modulus_sets, params_for, Cell};
use raven_inspire::inspiring::{ClientPackingKeys, PackParams, PackParamsError};
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::InspireParams;
use raven_inspire::pir::{setup, ClientSession};
use raven_inspire::rlwe::RlweSecretKey;

const RECORDED: &str = include_str!("fixtures/client_packing_key_kat.txt");

#[test]
fn key_bytes_and_sampler_position_match_the_recorded_generator() {
    let mut lines = Vec::new();
    for (name, moduli) in modulus_sets() {
        let params = params_for(2048, &moduli, 3);
        for gamma in [1usize, 2, 16, 128, 256, 512, 1024] {
            for seed in [0x4b45_5900u64, 0x4b45_5901] {
                let mut sampler = GaussianSampler::with_seed(params.sigma, seed);
                let secret_key = RlweSecretKey::generate(&params, &mut sampler);
                let keys = ClientPackingKeys::generate(
                    &secret_key,
                    &params,
                    gamma,
                    [gamma as u8; 32],
                    &mut sampler,
                )
                .expect("legal width");
                let bytes = bincode::serialize(&keys).unwrap();
                lines.push(format!(
                    "KEY set={name} gamma={gamma} seed={seed:#x} len={} fnv={:#018x} next={}",
                    bytes.len(),
                    fnv1a(&bytes),
                    sampler.sample()
                ));
            }
        }
    }
    let recorded: Vec<&str> = RECORDED.lines().collect();
    assert_eq!(lines.len(), recorded.len());
    for (line, recorded) in lines.iter().zip(recorded) {
        assert_eq!(
            line, recorded,
            "uploaded key bytes or sampler position moved"
        );
    }
}

/// The recorded lines are not all alike: another width, seed or modulus set is
/// another key.
#[test]
fn recorded_keys_differ_across_width_seed_and_modulus_set() {
    let digests: std::collections::BTreeSet<&str> = RECORDED
        .lines()
        .filter_map(|line| line.split(' ').find(|field| field.starts_with("fnv=")))
        .collect();
    assert_eq!(digests.len(), RECORDED.lines().count());
}

/// The key walks the generator the server's rotation tables are built on.
#[test]
fn the_key_generator_is_the_pack_params_generator() {
    let params = params_for(256, &modulus_sets()[0].1, 3);
    for gamma in PackParams::legal_widths(params.ring_dim) {
        let pack_params = PackParams::try_new(&params, gamma).expect("legal width");
        let mut sampler = GaussianSampler::with_seed(params.sigma, 5);
        let secret_key = RlweSecretKey::generate(&params, &mut sampler);
        let keys = ClientPackingKeys::generate(&secret_key, &params, gamma, [2; 32], &mut sampler)
            .expect("legal width");
        assert_eq!(keys.num_to_pack, pack_params.num_to_pack);
        assert_eq!(keys.y_body.len(), pack_params.gadget.len);
        assert!(!keys.full_key && keys.z_body.is_empty());

        // y_body[k] + s * w_mask[k] - tau_g(s) * g^k is the small error exactly when g is
        // the generator the tables use; under any other automorphism it is uniform.
        let ctx = params.ntt_context();
        let offline = raven_inspire::inspiring::OfflinePackingKeys::generate(&pack_params, [2; 32]);
        let tau_s =
            raven_inspire::rlwe::apply_automorphism(&secret_key.poly, pack_params.gen_pows[1]);
        for (k, y) in keys.y_body.iter().enumerate() {
            let s_w = secret_key.poly.mul_ntt(&offline.w_mask[k], &ctx);
            let gadget_power = pack_params.gadget.power(k);
            let residual = &(y + &s_w) + &tau_s.scalar_mul(gadget_power).negate();
            assert!(
                residual.linf_norm() < 256,
                "gamma {gamma} digit {k}: residual norm {} is not an error term",
                residual.linf_norm()
            );
        }
    }
}

#[test]
fn illegal_widths_are_refused_with_the_pack_params_error() {
    let params = InspireParams::secure_128_d2048();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 1);
    let secret_key = RlweSecretKey::generate(&params, &mut sampler);
    for width in [0usize, 3, 100, 2048, 4096] {
        let before = sampler.clone().sample();
        let refusal =
            ClientPackingKeys::generate(&secret_key, &params, width, [0; 32], &mut sampler)
                .expect_err("an illegal width must be refused");
        assert_eq!(
            refusal,
            PackParams::try_new(&params, width).expect_err("illegal for the server too")
        );
        assert!(matches!(refusal, PackParamsError::IllegalWidth { .. }));
        assert_eq!(
            sampler.clone().sample(),
            before,
            "a refusal must draw nothing"
        );
    }
}

/// A session builds no `PackParams`, so it must itself accept and refuse exactly the
/// CRS widths `PackParams` does.
#[test]
fn a_session_accepts_exactly_the_widths_pack_params_accepts() {
    for (name, moduli) in modulus_sets() {
        let params = params_for(256, &moduli, 3);
        params.validate().expect("shipped moduli validate");
        let database = vec![0u8; params.ring_dim * 32];
        let mut sampler = GaussianSampler::with_seed(params.sigma, 3);
        let (crs, _, secret_key) = setup(&params, &database, 32, &mut sampler).expect("setup");
        for width in 1..=params.ring_dim + 1 {
            let mut crs = crs.clone();
            crs.inspiring_num_columns = width;
            let session = ClientSession::new(crs, secret_key.clone(), &mut sampler);
            assert_eq!(
                session.is_ok(),
                PackParams::try_new(&params, width).is_ok(),
                "{name}: width {width}"
            );
        }
    }
}

/// One session, one key: every query, batch, upload and rehydration carries the bytes
/// built at construction.
#[test]
fn a_session_resends_the_key_it_built() {
    let params = params_for(256, &modulus_sets()[0].1, 3);
    let mut cell = Cell::build(&params, 32, 0x4f4e_4500);
    let upload = bincode::serialize(cell.session.packing_keys().expect("keys")).unwrap();

    for _ in 0..3 {
        let (_, query) = cell
            .session
            .query_seeded(9, &cell.encoded.config, &mut cell.sampler)
            .unwrap();
        let inline = query.inspiring_packing_keys.expect("inline keys");
        assert_eq!(bincode::serialize(&inline).unwrap(), upload);
    }

    let residue = bincode::serialize(&cell.session.to_residue()).unwrap();
    let rehydrated =
        ClientSession::from_residue(bincode::deserialize(&residue).unwrap()).expect("rehydrate");
    assert_eq!(
        bincode::serialize(rehydrated.packing_keys().expect("keys")).unwrap(),
        upload,
        "a rehydrated session must not rebuild its key"
    );

    let cloned = cell.session.clone();
    assert_eq!(
        bincode::serialize(cloned.packing_keys().expect("keys")).unwrap(),
        upload
    );
}

/// A new session under a new secret is the other legal retry.
#[test]
fn a_fresh_secret_gives_a_fresh_key() {
    let params = params_for(256, &modulus_sets()[0].1, 3);
    let cell = Cell::build(&params, 32, 0x4f4e_4501);
    let mut sampler = GaussianSampler::new(params.sigma);
    let first = ClientSession::with_fresh_secret(cell.crs.clone(), &mut sampler).expect("session");
    let second = ClientSession::with_fresh_secret(cell.crs.clone(), &mut sampler).expect("session");
    assert_ne!(
        first.rlwe_secret_key().poly.coeffs(),
        second.rlwe_secret_key().poly.coeffs()
    );
    assert_ne!(
        bincode::serialize(first.packing_keys().expect("keys")).unwrap(),
        bincode::serialize(second.packing_keys().expect("keys")).unwrap()
    );
}
