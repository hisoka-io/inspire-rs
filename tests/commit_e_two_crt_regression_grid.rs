//! 2-CRT round-trip grid over ring dimension, shard layout, record size and
//! both CRT pairs. Cells above 2^15 entries are left to the bench harness:
//! multi-GiB databases exceed a CI runner's RAM and the algebra under test is
//! dimension-independent.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel, DEFAULT_Q_2CRT_30BIT};
use raven_inspire::{
    extract_inspiring, query_seeded, respond_seeded_inspiring, setup, PackingMode,
};

fn base_params(ring_dim: usize, crt_moduli: Vec<u64>) -> InspireParams {
    let q: u64 = crt_moduli.iter().product();
    InspireParams {
        ring_dim,
        q,
        crt_moduli,
        p: 65537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

fn smoke_cell(params: &InspireParams, entries: u64, record_bytes: usize, label: &str) {
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let total = (entries as usize) * record_bytes;
    let mut db = vec![0u8; total];
    for i in 0..entries as usize {
        for j in 0..record_bytes {
            db[i * record_bytes + j] = ((i + j) % 251) as u8;
        }
    }
    let (crs, encoded_db, sk) =
        setup(params, &db, record_bytes, &mut sampler).expect("setup must succeed");

    let n = entries.saturating_sub(1).max(1);
    let indices = [n / 4, n / 2, (3 * n) / 4];
    for &idx in &indices {
        let (state, mut seeded_query) =
            query_seeded(&crs, idx, &encoded_db.config, &sk, &mut sampler).unwrap();
        seeded_query.packing_mode = PackingMode::Inspiring;
        let response = respond_seeded_inspiring(&crs, &encoded_db, &seeded_query).unwrap();
        let response_wire = response.to_binary().expect("serialize response prefix");
        let response = raven_inspire::ServerResponse::from_binary(&response_wire)
            .expect("deserialize response prefix");
        let recovered = extract_inspiring(&crs, &state, &response, record_bytes).unwrap();
        let expected: Vec<u8> = (0..record_bytes)
            .map(|j| (((idx as usize) + j) % 251) as u8)
            .collect();
        assert_eq!(
            recovered, expected,
            "{label}: byte-match failed @ idx={idx}"
        );
    }
}

#[test]
fn commit_e_google_crt_d256_singleshard_variable_records() {
    let crt = vec![67_043_329u64, 132_120_577u64];
    for &rb in &[8usize, 32, 128, 256] {
        let params = base_params(256, crt.clone());
        smoke_cell(&params, 256, rb, &format!("google-crt d=256 rb={rb}"));
    }
}

#[test]
fn commit_e_google_crt_d512_multi_records() {
    let crt = vec![67_043_329u64, 132_120_577u64];
    for &rb in &[8usize, 32, 256] {
        let params = base_params(512, crt.clone());
        smoke_cell(&params, 512, rb, &format!("google-crt d=512 rb={rb}"));
    }
}

#[test]
fn commit_e_google_crt_d2048_one_shard() {
    let crt = vec![67_043_329u64, 132_120_577u64];
    for &rb in &[8usize, 32, 256] {
        let params = base_params(2048, crt.clone());
        smoke_cell(&params, 2048, rb, &format!("google-crt d=2048 rb={rb}"));
    }
}

#[test]
fn commit_e_google_crt_d2048_multi_shard_small_records() {
    let crt = vec![67_043_329u64, 132_120_577u64];
    let params = base_params(2048, crt);
    smoke_cell(
        &params,
        1 << 14,
        8,
        "google-crt d=2048 entries=2^14 rb=8 multi-shard",
    );
}

#[test]
fn commit_e_30bit_crt_d2048_multi_records() {
    let crt = DEFAULT_Q_2CRT_30BIT.to_vec();
    for &rb in &[8usize, 32, 256] {
        let params = base_params(2048, crt.clone());
        smoke_cell(&params, 2048, rb, &format!("30bit-crt d=2048 rb={rb}"));
    }
}

#[test]
fn commit_e_single_prime_default_q_no_regression() {
    let crt = vec![1_152_921_504_606_830_593u64];
    for &rb in &[8usize, 32, 256] {
        let params = base_params(2048, crt.clone());
        smoke_cell(&params, 2048, rb, &format!("default-q d=2048 rb={rb}"));
    }
}
