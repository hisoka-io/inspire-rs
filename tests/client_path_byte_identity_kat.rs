//! Byte pins on the client path: query generation, packing keys, encryption and
//! the LWE key, from fixed seeds. The digests were recorded before the scalar
//! product, the ring sum, difference and negation and the residue reduction
//! were made branch- and divide-free, so they prove those rewrites changed no
//! byte a client emits.
//!
//! Every modulus set the crate ships is covered: the single 60-bit prime and
//! the three two-prime sets. Query indices cover both ends of the ring.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test-target harness; an abort here is the failure report"
)]

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use raven_inspire::inspiring::{ClientPackingKeys, PackParams};
use raven_inspire::lwe::LweSecretKey;
use raven_inspire::math::{GaussianSampler, NttContext, Poly};
use raven_inspire::params::{
    InspireParams, SecurityLevel, DEFAULT_CRT_MODULI, DEFAULT_Q_2CRT_30BIT,
};
use raven_inspire::pir::inverse_monomial;
use raven_inspire::rgsw::{GadgetVector, RgswCiphertext, SeededRgswCiphertext};
use raven_inspire::rlwe::{RlweCiphertext, RlweSecretKey};

const D: usize = 2048;
const ADAPTIVE_CRT_MODULI: [u64; 2] = [67_043_329, 132_120_577];

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn digest_u64s(values: &[u64]) -> u64 {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    fnv1a(&bytes)
}

fn digest_serde<T: serde::Serialize>(value: &T) -> u64 {
    fnv1a(&bincode::serialize(value).expect("bincode"))
}

fn params_for(moduli: &[u64]) -> InspireParams {
    let q: u64 = moduli.iter().product();
    InspireParams {
        ring_dim: D,
        q,
        crt_moduli: moduli.to_vec(),
        p: 65537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

/// One digest per client-path output, in a fixed order, for one modulus set.
fn client_path_digests(moduli: &[u64]) -> Vec<(String, u64)> {
    let params = params_for(moduli);
    let q = params.q;
    let delta = params.delta();
    let ctx = NttContext::with_moduli(D, moduli);
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0x51C2_0001);
    let mut rng = ChaCha20Rng::seed_from_u64(0x51C2_0002);
    let mut out = Vec::new();

    let sk = RlweSecretKey::generate(&params, &mut sampler);
    out.push(("secret key".to_owned(), digest_u64s(sk.poly.coeffs())));
    let lwe = LweSecretKey::from_rlwe(&sk);
    out.push(("lwe key".to_owned(), digest_u64s(&lwe.coeffs)));

    let one_row = GadgetVector::new(params.gadget_base, 1, q);
    let three_rows = GadgetVector::new(params.gadget_base, 3, q);
    for k in [0usize, 1, 2, D / 2, D - 2, D - 1] {
        let scaled = inverse_monomial(k, D, q, moduli).scalar_mul(delta);
        out.push((
            format!("scaled monomial k={k}"),
            digest_u64s(scaled.coeffs()),
        ));
        let seeded = SeededRgswCiphertext::encrypt_with_rng(
            &sk,
            &scaled,
            &one_row,
            &mut sampler,
            &ctx,
            &mut rng,
        );
        out.push((format!("seeded query k={k}"), digest_serde(&seeded)));
        let full = RgswCiphertext::encrypt_with_rng(
            &sk,
            &scaled,
            &three_rows,
            &mut sampler,
            &ctx,
            &mut rng,
        );
        out.push((format!("three-row rgsw k={k}"), digest_serde(&full)));
    }

    let message: Vec<u64> = (0..D as u64).map(|i| (i * 7919) % params.p).collect();
    let message = Poly::from_coeffs_moduli(message, moduli);
    let a = Poly::from_seed_moduli(&[9u8; 32], D, moduli);
    let error = Poly::sample_gaussian_moduli(D, moduli, &mut sampler);
    let ct = RlweCiphertext::encrypt(&sk, &message, delta, a, &error, &ctx);
    out.push(("rlwe encrypt".to_owned(), digest_serde(&ct)));

    let pack_params = PackParams::try_new(&params, 16).expect("pack params");
    let keys = ClientPackingKeys::generate(&sk, &pack_params, [5u8; 32], &mut sampler);
    out.push(("packing keys".to_owned(), digest_serde(&keys)));
    let y_all: Vec<u64> = keys
        .y_all
        .iter()
        .flatten()
        .flat_map(|p| p.coeffs().iter().copied())
        .collect();
    out.push(("packing rotations".to_owned(), digest_u64s(&y_all)));
    out
}

fn assert_digests(moduli: &[u64], want: &[u64]) {
    let got = client_path_digests(moduli);
    let listing: Vec<String> = got
        .iter()
        .map(|(label, d)| format!("{label}: {d:#018x}"))
        .collect();
    let got_values: Vec<u64> = got.iter().map(|(_, d)| *d).collect();
    assert_eq!(
        got_values,
        want,
        "client-path bytes moved for moduli {moduli:?}:\n{}",
        listing.join("\n")
    );
}

#[test]
fn single_prime_client_path_is_byte_identical() {
    assert_digests(
        &[raven_inspire::math::DEFAULT_Q],
        &[
            0xa37a9f4a7223e493,
            0xe33e4f086f2fd774,
            0xdf1132eb5f97b681,
            0xb48df717a881a412,
            0x6164cb300e11e282,
            0xcef3fd397d7c9973,
            0x63b653e25ee25ab8,
            0x10457ce01009f049,
            0x507a7d722e047eb3,
            0x16bb366ccb92a8d5,
            0x6329bb5a5e78782e,
            0x64939ed1c42b6fb3,
            0x71de56b6acd6c6cb,
            0xbcf1c612e65fd08c,
            0xa6eb37ede6eeedb3,
            0xaa28e163b473cede,
            0x76ac31f4e878c8ff,
            0x2674291c42062173,
            0x6cdb498122f22a21,
            0x02f4e16304d8bbc1,
            0xd5efb66a14d986a7,
            0x567b7d3adf140362,
            0x2f632232bc9d6881,
        ],
    );
}

#[test]
fn upstream_two_prime_client_path_is_byte_identical() {
    assert_digests(
        &DEFAULT_CRT_MODULI,
        &[
            0x6e10016be83280ab,
            0x52151b96139eb60c,
            0xc7c803ad9e388adc,
            0xcc90316b7e6fd4b8,
            0xc52891df455c8336,
            0xd9c187ecee4c1979,
            0x07ccdd3359937079,
            0x4b976cb0b76f512f,
            0xf3eaae4973b69a39,
            0x1a626d60396b9c4b,
            0xbc19660f8fb98d3f,
            0xcccb1d92bd7b3eb9,
            0x6eae682d6871e9fe,
            0xe2d8f924b3b48c0a,
            0x26855039e463e839,
            0x23a6483fd1f631c0,
            0xfdcc7f897efca198,
            0xb1b46a407188d4f9,
            0x291ec67ba25daa0c,
            0xad27508c0cef4510,
            0x9b71f96b9e5e95b6,
            0x439d15dfb75ec46f,
            0x1626fc8213fd88e0,
        ],
    );
}

#[test]
fn thirty_bit_two_prime_client_path_is_byte_identical() {
    assert_digests(
        &DEFAULT_Q_2CRT_30BIT,
        &[
            0x796939c443164746,
            0x1240ff7382287b0c,
            0x532eafa3ef8fa1ed,
            0xf3edf2ae016ba25f,
            0xd26c5d5fd8d95ee9,
            0x2fa4849201905294,
            0x4c67708ebe120f08,
            0x2c1ed727730395f6,
            0xeb0a4dfcfac20234,
            0x0e2255d08d3467ae,
            0x7e9403c5f38d86d6,
            0xd900706f4e12fd74,
            0x963236becc766290,
            0xd5023073e2de90b1,
            0x7fc939463a8fb6b4,
            0x6fb68bb3cf3e4b10,
            0xe886db90a2ef1cc1,
            0xaff1abb839f0f1d4,
            0xb7af4866b85f52b4,
            0xea282a24924c211f,
            0xb0cfb6357aa13cfc,
            0xad5ad144494c103a,
            0x09a9086829907a4b,
        ],
    );
}

#[test]
fn adaptive_two_prime_client_path_is_byte_identical() {
    assert_digests(
        &ADAPTIVE_CRT_MODULI,
        &[
            0xd09c249bf21f15e3,
            0xceac0c48e3e55e06,
            0xed2472e47bd7cf09,
            0x7ce91dd24a6138cb,
            0x96ce8c7a2b3b0b64,
            0xfa76bc6106c975fa,
            0xd51a204c0ad42b2d,
            0x6d36ac58d2ced66a,
            0xe30fdcd935786fda,
            0x906973ec97f27ace,
            0xab38ca89e784fe1c,
            0xebf1405690e91a9a,
            0x2b37365ef1b3bb29,
            0x356af4a4d3f312de,
            0x8a6c1b4cbf79155a,
            0x7eb20f81e15cb012,
            0xc09b1b3c37d44788,
            0xcd01e25347a218ba,
            0xc89e3a1332832603,
            0x72b6eb94e08335d3,
            0x660755c1b632a959,
            0x029d977c70a07ab9,
            0x2572c2eb28804752,
        ],
    );
}
