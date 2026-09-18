//! `respond` must emit identical bytes with and without the `parallel` feature.
//! A checked-in fixture is hashed to one golden under both builds, and the
//! order-preserving `par_iter().map().collect()` those kernels rely on is
//! pinned separately.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]
#![allow(
    clippy::doc_lazy_continuation,
    clippy::trivially_copy_pass_by_ref,
    reason = "KAT narration and fixture plumbing"
)]

use std::path::PathBuf;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use raven_inspire::ks::KeySwitchingMatrix;
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::pir::{
    query, respond_inspiring, setup_with_rng, ClientQuery, EncodedDatabase, ServerCrs,
};
use raven_inspire::rgsw::GadgetVector;
use serde::Deserialize;

#[derive(Deserialize)]
struct LegacyInspireParams {
    ring_dim: usize,
    q: u64,
    crt_moduli: Vec<u64>,
    p: u64,
    sigma: f64,
    gadget_base: u64,
    gadget_len: usize,
    security_level: SecurityLevel,
}

#[derive(Deserialize)]
struct LegacyServerCrs {
    params: LegacyInspireParams,
    galois_keys: Vec<KeySwitchingMatrix>,
    rgsw_gadget: GadgetVector,
    inspiring_w_seed: [u8; 32],
    inspiring_v_seed: [u8; 32],
    inspiring_num_columns: usize,
}

fn decode_fixture(bytes: &[u8]) -> (ServerCrs, EncodedDatabase, ClientQuery) {
    if let Ok(current) = bincode::deserialize(bytes) {
        return current;
    }
    let (legacy, encoded_db, query): (LegacyServerCrs, EncodedDatabase, ClientQuery) =
        bincode::deserialize(bytes).expect("deserialize legacy fixture");
    let params = InspireParams {
        ring_dim: legacy.params.ring_dim,
        q: legacy.params.q,
        crt_moduli: legacy.params.crt_moduli,
        p: legacy.params.p,
        sigma: legacy.params.sigma,
        gadget_base: legacy.params.gadget_base,
        query_gadget_len: legacy.params.gadget_len,
        packing_gadget_len: legacy.params.gadget_len,
        security_level: legacy.params.security_level,
    };
    (
        ServerCrs {
            params,
            galois_keys: legacy.galois_keys,
            rgsw_gadget: legacy.rgsw_gadget,
            inspiring_pack_params: None,
            inspiring_packing_key: None,
            inspiring_w_seed: legacy.inspiring_w_seed,
            inspiring_v_seed: legacy.inspiring_v_seed,
            inspiring_num_columns: legacy.inspiring_num_columns,
        },
        encoded_db,
        query,
    )
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn d256_params() -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65_536,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/respond_byte_identity_d256.bin")
}

// order-preservation is d-independent, so a d=256 fixture suffices
const GOLDEN_FNV1A: u64 = 0x796a_80ca_60d3_1a52;

#[test]
fn respond_byte_identical_par_vs_seq_on_fixed_input() {
    let path = fixture_path();
    let (crs, encoded_db, q): (ServerCrs, EncodedDatabase, ClientQuery) = if let Ok(bytes) =
        std::fs::read(&path)
    {
        decode_fixture(&bytes)
    } else {
        let params = d256_params();
        let entry_size = 32usize;
        let db: Vec<u8> = (0..params.ring_dim * entry_size)
            .map(|i| (i % 251) as u8)
            .collect();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 1);
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        let (crs, encoded_db, sk) =
            setup_with_rng(&params, &db, entry_size, &mut sampler, &mut rng).unwrap();
        let (_state, qy) = query(&crs, 3, &encoded_db.config, &sk, &mut sampler).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            bincode::serialize(&(&crs, &encoded_db, &qy)).unwrap(),
        )
        .unwrap();
        panic!("fixture generated at {path:?}; re-run this test to print the golden, then pin GOLDEN_FNV1A");
    };

    let response = respond_inspiring(&crs, &encoded_db, &q).expect("respond");
    let hash = fnv1a(&bincode::serialize(&response).unwrap());
    eprintln!("respond byte-identity (fixed input): fnv1a={hash:#018x}");
    assert_eq!(
        hash, GOLDEN_FNV1A,
        "respond bytes differ from the golden - a par/seq divergence (or the fixture changed)"
    );
}

// A test pinning rayon's documented guarantee that par_iter().map().collect()
// preserves order (rayon docs, ParallelIterator::collect) lived here; it ran
// over a test-local kernel and stayed green while respond's collect order was
// inverted (2026-09-06 mutation audit). The golden above is the real guard.
