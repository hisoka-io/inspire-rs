//! `Poly` decodes through a validating `TryFrom`, so a hostile or corrupt encoding
//! cannot produce one whose `dim` outruns its coefficients.
//!
//! `dim` is a bare fixint on the wire: a length cap on the encoded body does not bound
//! it. `apply_automorphism` allocates `vec![0u64; poly.dimension()]` before it reads a
//! single coefficient, and an allocation that large aborts the process - not a
//! catchable panic - taking every in-flight query with it.

#![allow(
    clippy::expect_used,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]

use raven_inspire::math::mod_q::DEFAULT_Q;
use raven_inspire::math::poly::Poly;

fn round_trip(p: &Poly) -> Result<Poly, Box<bincode::ErrorKind>> {
    let bytes = bincode::serialize(p).expect("serialize");
    bincode::deserialize::<Poly>(&bytes)
}

#[test]
fn a_legitimate_poly_round_trips() {
    let p = Poly::constant(42, 256, DEFAULT_Q);
    let back = round_trip(&p).expect("a well-formed Poly must decode");
    assert_eq!(back.dimension(), 256);
    assert_eq!(back.coeff(0), 42);
}

#[test]
fn the_default_poly_round_trips() {
    let p = Poly::default();
    let back = round_trip(&p).expect("Default must survive a round trip");
    assert_eq!(back.dimension(), 0);
}

/// The attack: overwrite `dim` in the encoded bytes and leave the coefficients alone.
#[test]
fn a_dim_larger_than_its_coefficients_is_refused() {
    let p = Poly::constant(7, 4, DEFAULT_Q);
    let mut bytes = bincode::serialize(&p).expect("serialize");

    // bincode lays the struct out in declaration order: coeffs (len prefix + data),
    // moduli (len prefix + data), q, then dim. Locate dim by its known value rather
    // than by a hardcoded offset, so a layout change fails the test instead of
    // silently patching the wrong field.
    let dim_le = 4usize.to_le_bytes();
    let at = bytes
        .windows(8)
        .rposition(|w| w == dim_le)
        .expect("dim must appear in the encoding");
    let hostile: usize = 1 << 44;
    bytes[at..at + 8].copy_from_slice(&hostile.to_le_bytes());

    let err = bincode::deserialize::<Poly>(&bytes)
        .expect_err("a dim of 2^44 against 4 coefficients must be refused");
    let msg = format!("{err}");
    assert!(
        msg.contains("tight coefficient"),
        "the refusal must name itself; got: {msg}"
    );
    assert!(
        msg.contains("payload") && msg.contains("expected"),
        "the refusal must name the length disagreement; got: {msg}"
    );
}

/// An empty modulus vector bypasses `init_moduli`'s `1 <= len <= 2` assertion, which
/// is documented as the sole entry point for the modulus vector.
#[test]
fn an_empty_modulus_vector_is_refused_unless_the_whole_value_is_default() {
    let p = Poly::constant(7, 2, DEFAULT_Q);
    let bytes = bincode::serialize(&p).expect("serialize");
    // moduli is the second length-prefixed vec; a 1 -> 0 flip on its length makes
    // the trailing modulus reparse as the next field. The window search must hard-
    // fail when the prefix moves: an if-let here silently skipped every assertion
    // once the fixture stopped producing a 1u64 window (2026-09-06 mutation audit).
    let mut hostile = bytes.clone();
    let one_le = 1u64.to_le_bytes();
    let at = hostile
        .windows(8)
        .rposition(|w| w == one_le)
        .expect("moduli len prefix must appear in the encoding");
    hostile[at..at + 8].copy_from_slice(&0u64.to_le_bytes());
    let err = bincode::deserialize::<Poly>(&hostile)
        .expect_err("a Poly whose modulus vector was emptied on the wire must not decode");
    let msg = format!("{err}");
    assert!(
        msg.contains("tight coefficient"),
        "the refusal must come from the validating decode, not an incidental \
         parse error; got: {msg}"
    );
}

/// The count guard bounds how MANY moduli arrive, never their values. A modulus of 0
/// makes `crt_modulus` return 0, so an encoded `q` of 0 matches it and every structural
/// check passes; the single-modulus path never reaches the coprimality guard. Downstream
/// reduction then divides by zero on wire-supplied input, inside a library.
#[test]
fn a_zero_modulus_is_refused() {
    let p = Poly::constant(7, 2, DEFAULT_Q);
    let mut bytes = bincode::serialize(&p).expect("serialize");

    // DEFAULT_Q appears twice: as moduli[0] and as q. Zero both so the CRT product still
    // equals the encoded q and only the VALUE guard can refuse it.
    let needle = DEFAULT_Q.to_le_bytes();
    let mut patched = 0usize;
    let mut at = 0;
    while at + 8 <= bytes.len() {
        if bytes[at..at + 8] == needle {
            bytes[at..at + 8].copy_from_slice(&0u64.to_le_bytes());
            patched += 1;
            at += 8;
        } else {
            at += 1;
        }
    }
    assert_eq!(patched, 2, "expected moduli[0] and q to carry DEFAULT_Q");

    let err = bincode::deserialize::<Poly>(&bytes)
        .expect_err("a zero modulus must be refused, not divided by");
    let msg = format!("{err}");
    assert!(
        msg.contains("tight coefficient"),
        "the refusal must name itself; got: {msg}"
    );
    assert!(
        msg.contains("modulus"),
        "the refusal must name the modulus value; got: {msg}"
    );
}
