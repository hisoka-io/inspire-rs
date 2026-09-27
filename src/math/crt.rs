//! CRT helpers. Ported from private-membership/research/InsPIRe,
//! commit 89f04516c4b8b48b8e65e50d25b37256e04096ad, Apache-2.0.
//!
//! The extended-Euclidean inverse here is variable-time in its inputs; every
//! in-tree caller passes public parameters only (moduli, ring_dim, Galois
//! elements). A secret-data call site must switch to Fermat exponentiation.

use super::modular::{ct_sub_if_ge, reduce_by_public_reciprocal};

/// `Some(x)` with `(a * x) % modulus == 1`, or `None` when `a` is not invertible.
pub fn try_mod_inverse(a: u64, modulus: u64) -> Option<u64> {
    let mut t: i128 = 0;
    let mut new_t: i128 = 1;
    let mut r: i128 = modulus as i128;
    let mut new_r: i128 = a as i128;

    while new_r != 0 {
        let quotient = r / new_r;
        let tmp_t = t - quotient * new_t;
        t = new_t;
        new_t = tmp_t;

        let tmp_r = r - quotient * new_r;
        r = new_r;
        new_r = tmp_r;
    }

    if r != 1 {
        return None;
    }

    if t < 0 {
        t += modulus as i128;
    }
    Some(t as u64)
}

/// `x` such that `(a * x) % modulus == 1`.
///
/// # Panics
///
/// If `a` is not invertible. Callers that cannot establish `gcd(a, modulus) == 1`
/// via `InspireParams::validate()` MUST use [`try_mod_inverse`].
#[allow(
    clippy::panic,
    reason = "documented abort with a typed sibling: try_mod_inverse"
)]
#[must_use]
pub fn mod_inverse(a: u64, modulus: u64) -> u64 {
    match try_mod_inverse(a, modulus) {
        Some(x) => x,
        None => panic!(
            "mod_inverse: value {a} is not invertible modulo {modulus} \
             (invariant violated; callers must check gcd or use try_mod_inverse)"
        ),
    }
}

/// `a0 + q0 * ((a1 - a0) * q0^{-1} mod q1)`, the residue pair recombined mod q0*q1.
pub fn crt_compose_2(a0: u64, a1: u64, q0: u64, q1: u64, q0_inv_mod_q1: u64) -> u64 {
    let a0_mod_q1 = a0 % q1;
    let diff = if a1 >= a0_mod_q1 {
        a1 - a0_mod_q1
    } else {
        (a1 + q1) - a0_mod_q1
    };
    let t = ((diff as u128 * q0_inv_mod_q1 as u128) % q1 as u128) as u64;
    a0 + q0 * t
}

/// [`crt_compose_2`] for secret residues: `a0 mod q1` by Barrett and the product with
/// `q0^{-1}` by Shoup, so the only divisions are over the public moduli.
pub(crate) struct CtCrtComposer {
    q0: u64,
    q1: u64,
    q: u64,
    q0_inv_mod_q1: u64,
    /// `floor(q0_inv_mod_q1 * 2^64 / q1)`; exact Shoup needs `q1 < 2^63`, which
    /// `q0 * q1 < 2^64` with `q0 >= 2` guarantees.
    q0_inv_shoup: u64,
    q1_reciprocal: u64,
    q_reciprocal: u64,
}

impl CtCrtComposer {
    pub(crate) fn new(q0: u64, q1: u64, q0_inv_mod_q1: u64) -> Self {
        let q0_inv_shoup = ((u128::from(q0_inv_mod_q1) << 64) / u128::from(q1)) as u64;
        let q = q0.wrapping_mul(q1);
        Self {
            q0,
            q1,
            q,
            q0_inv_mod_q1,
            q0_inv_shoup,
            q1_reciprocal: u64::MAX / q1,
            q_reciprocal: u64::MAX / q,
        }
    }

    pub(crate) fn compose(&self, a0: u64, a1: u64) -> u64 {
        let a0_mod_q1 = reduce_by_public_reciprocal(a0, self.q1, self.q1_reciprocal);
        let diff = ct_sub_if_ge(a1.wrapping_add(self.q1).wrapping_sub(a0_mod_q1), self.q1);
        let estimate = ((u128::from(diff) * u128::from(self.q0_inv_shoup)) >> 64) as u64;
        let lifted = ct_sub_if_ge(
            diff.wrapping_mul(self.q0_inv_mod_q1)
                .wrapping_sub(estimate.wrapping_mul(self.q1)),
            self.q1,
        );
        reduce_by_public_reciprocal(
            a0.wrapping_add(self.q0.wrapping_mul(lifted)),
            self.q,
            self.q_reciprocal,
        )
    }
}

/// Split a value into two CRT residues.
#[inline]
pub fn crt_decompose_2(value: u64, q0: u64, q1: u64) -> (u64, u64) {
    (value % q0, value % q1)
}

/// Compute the product of moduli (composite modulus).
pub fn crt_modulus(moduli: &[u64]) -> u64 {
    moduli.iter().copied().fold(1u64, u64::saturating_mul)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Old-vs-new over both residues' extremes and a wide random spread, at the two
    /// 2-CRT moduli pairs in tree: the upstream preset and the adaptive derivation.
    #[test]
    fn constant_time_composition_matches_crt_compose_2() {
        for (q0, q1) in [(268_369_921u64, 249_561_089u64), (67_043_329, 132_120_577)] {
            let inv = mod_inverse(q0, q1);
            let q = q0 * q1;
            let composer = CtCrtComposer::new(q0, q1, inv);
            let edges = |m: u64| [0, 1, 2, m / 2 - 1, m / 2, m / 2 + 1, m - 2, m - 1];
            for a0 in edges(q0) {
                for a1 in edges(q1) {
                    assert_eq!(
                        composer.compose(a0, a1),
                        crt_compose_2(a0, a1, q0, q1, inv) % q,
                        "a0={a0} a1={a1} q0={q0} q1={q1}"
                    );
                }
            }
            let mut walk = 0x2545_F491_4F6C_DD1Du64;
            for _ in 0..2_000_000 {
                walk ^= walk << 13;
                walk ^= walk >> 7;
                walk ^= walk << 17;
                let (a0, a1) = (walk % q0, (walk >> 32) % q1);
                assert_eq!(
                    composer.compose(a0, a1),
                    crt_compose_2(a0, a1, q0, q1, inv) % q,
                    "a0={a0} a1={a1} q0={q0} q1={q1}"
                );
            }
        }
    }
}
