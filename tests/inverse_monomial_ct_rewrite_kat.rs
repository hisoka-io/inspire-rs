//! Byte-level pin on `inverse_monomial`, so a branch-free rewrite of it can be
//! proved to return the same bytes.
//!
//! The function picks one coefficient by `k`, and `k` is the query index - the
//! secret the whole scheme exists to hide. Making it constant-time means
//! writing every coefficient and selecting through a barrier, which touches the
//! ciphertext the client publishes. This KAT is the precondition for that edit:
//! it fixes the output, backing residue by backing residue, over every index of
//! two rings and both CRT shapes.
//!
//! The expectation is derived, not observed. The ring is negacyclic
//! (X^d = -1), so X^(-k) = -X^(d-k) for k > 0 and 1 for k = 0: exactly one
//! nonzero coefficient, at d-k, carrying q-1, CRT-split when the modulus is a
//! product. The sweep digests are measured, and stand as a second witness that
//! nothing outside the derived positions moved.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test-target harness; an abort here is the failure report"
)]

use raven_inspire::math::{NttContext, Poly};
use raven_inspire::params::DEFAULT_Q_2CRT_30BIT;
use raven_inspire::pir::{encode_direct, inverse_monomial};

const Q_SINGLE: u64 = 1_152_921_504_606_830_593;

fn fnv1a_u64s(values: &[u64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The backing array `inverse_monomial(k, ..)` must produce: residues
/// concatenated by modulus, `dim * crt_count` long, zero except at `d-k`.
fn derived_backing(k: usize, d: usize, moduli: &[u64]) -> Vec<u64> {
    let q: u64 = moduli.iter().product();
    let (pos, value) = if k == 0 { (0, 1u64) } else { (d - k, q - 1) };
    let mut want = vec![0u64; d * moduli.len()];
    for (limb, &m) in moduli.iter().enumerate() {
        want[limb * d + pos] = value % m;
    }
    want
}

fn assert_pinned(k: usize, d: usize, moduli: &[u64]) {
    let q: u64 = moduli.iter().product();
    let poly = inverse_monomial(k, d, q, moduli);

    assert_eq!(poly.dimension(), d, "k={k}: dimension moved");
    assert_eq!(poly.moduli(), moduli, "k={k}: modulus set moved");
    assert!(
        !poly.is_ntt(),
        "k={k}: must be returned in coefficient domain"
    );
    assert_backing_matches(poly.coeffs(), k, d, moduli);
}

/// A whole-array `assert_eq!` prints two 2048-entry vectors and buries the one
/// residue that moved.
fn assert_backing_matches(got: &[u64], k: usize, d: usize, moduli: &[u64]) {
    let want = derived_backing(k, d, moduli);
    assert_eq!(got.len(), want.len(), "k={k}, d={d}: backing length moved");
    let differing: Vec<String> = got
        .iter()
        .zip(&want)
        .enumerate()
        .filter(|(_, (g, w))| g != w)
        .map(|(i, (g, w))| format!("[{}/{}] got {g}, want {w}", i / d, i % d))
        .take(8)
        .collect();
    assert!(
        differing.is_empty(),
        "k={k}, d={d}, moduli={moduli:?}: backing residues differ from X^(-k) \
         at [limb/coeff]:\n{}",
        differing.join("\n")
    );
}

/// `X^k * X^(-k) = 1`, which is the identity the byte pin above encodes. Kept
/// separate so the pin cannot drift into agreeing with a wrong implementation.
fn assert_inverts_x_to_the_k(k: usize, d: usize, moduli: &[u64]) {
    let q: u64 = moduli.iter().product();
    let ctx = NttContext::with_moduli(d, moduli);

    let mut mono = vec![0u64; d];
    mono[k] = 1;
    let product =
        Poly::from_coeffs_moduli(mono, moduli).mul_ntt(&inverse_monomial(k, d, q, moduli), &ctx);

    assert_eq!(
        product.coeff(0),
        1,
        "k={k}: X^k * X^(-k) has constant term != 1"
    );
    for i in 1..d {
        assert_eq!(
            product.coeff(i),
            0,
            "k={k}: X^k * X^(-k) is not the constant 1"
        );
    }
}

/// The retrieval this exists for: value at coefficient k lands at coefficient 0.
fn assert_rotates_value_k_to_zero(k: usize, d: usize, moduli: &[u64]) {
    let q: u64 = moduli.iter().product();
    let ctx = NttContext::with_moduli(d, moduli);
    let values: Vec<u64> = (0..d).map(|i| (i as u64 + 1) * 7).collect();
    let h = encode_direct(&values, d, q, moduli);

    let rotated = h.mul_ntt(&inverse_monomial(k, d, q, moduli), &ctx);
    assert_eq!(
        rotated.coeff(0),
        values[k],
        "k={k}: wrong value landed at coefficient 0"
    );
}

/// 0 and 1 straddle the `k == 0` special case; d-1 is the far end of the
/// `d - k` write; d/2 is an interior index with no boundary to hide behind.
fn boundary_indices(d: usize) -> [usize; 4] {
    [0, 1, d / 2, d - 1]
}

#[test]
fn pins_boundary_indices_single_modulus_d256() {
    for k in boundary_indices(256) {
        assert_pinned(k, 256, &[Q_SINGLE]);
        assert_inverts_x_to_the_k(k, 256, &[Q_SINGLE]);
        assert_rotates_value_k_to_zero(k, 256, &[Q_SINGLE]);
    }
}

#[test]
fn pins_boundary_indices_single_modulus_d2048() {
    for k in boundary_indices(2048) {
        assert_pinned(k, 2048, &[Q_SINGLE]);
        assert_inverts_x_to_the_k(k, 2048, &[Q_SINGLE]);
        assert_rotates_value_k_to_zero(k, 2048, &[Q_SINGLE]);
    }
}

/// Two CRT limbs split `q - 1` into two different residues, so a rewrite that
/// writes the composite value into limb 0 passes the single-modulus pin above
/// and fails here.
#[test]
fn pins_boundary_indices_two_crt_d256() {
    let moduli = DEFAULT_Q_2CRT_30BIT;
    for k in boundary_indices(256) {
        assert_pinned(k, 256, &moduli);
        assert_inverts_x_to_the_k(k, 256, &moduli);
        assert_rotates_value_k_to_zero(k, 256, &moduli);
    }
}

/// Four sampled indices cannot catch an off-by-one that only bites at some
/// other k. These digests cover every index of the ring.
fn sweep_digest(d: usize, moduli: &[u64]) -> u64 {
    let q: u64 = moduli.iter().product();
    let mut all = Vec::with_capacity(d * d * moduli.len());
    for k in 0..d {
        let poly = inverse_monomial(k, d, q, moduli);
        assert_backing_matches(poly.coeffs(), k, d, moduli);
        all.extend_from_slice(poly.coeffs());
    }
    fnv1a_u64s(&all)
}

const SWEEP_D256_SINGLE: u64 = 0x7e01_03e6_e9d1_6342;
const SWEEP_D256_TWO_CRT: u64 = 0x5a6b_dd63_4d42_8e52;
const SWEEP_D2048_SINGLE: u64 = 0x93b1_31cc_d5b1_6142;

#[test]
fn sweep_over_every_index_matches_the_pinned_digests() {
    assert_eq!(
        sweep_digest(256, &[Q_SINGLE]),
        SWEEP_D256_SINGLE,
        "d=256 single-modulus sweep moved"
    );
    assert_eq!(
        sweep_digest(256, &DEFAULT_Q_2CRT_30BIT),
        SWEEP_D256_TWO_CRT,
        "d=256 two-CRT sweep moved"
    );
    assert_eq!(
        sweep_digest(2048, &[Q_SINGLE]),
        SWEEP_D2048_SINGLE,
        "d=2048 single-modulus sweep moved"
    );
}
