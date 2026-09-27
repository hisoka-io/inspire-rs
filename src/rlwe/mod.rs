//! RLWE encryption over R_q = Z_q\[X\]/(X^d + 1): `b = -a*s + e + delta*m`.
//!
//! Galois automorphisms `tau_g(X) = X^g` rotate coefficients and drive the
//! LWE -> RLWE packing.

mod enc;
mod galois;
mod types;

pub use galois::{
    apply_automorphism, automorphism_ciphertext, automorphism_order, compose_automorphisms,
    galois_generators, inverse_automorphism, is_valid_galois_element, try_inverse_automorphism,
};
pub(crate) use types::require_same_ring;
pub use types::{RlweCiphertext, RlweSecretKey, SeededRlweCiphertext};
