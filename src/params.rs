//! Parameter sets for InsPIRe PIR
//!
//! This module defines the cryptographic parameters for the InsPIRe protocol,
//! including ring dimensions, moduli, and security levels. `validate` checks
//! structural and noise invariants only: it runs no lattice estimate, and
//! `security_level` is a declared target nothing verifies. The shipped
//! `secure_128_d2048` measures 121.5 bits, not 128 (see PRIVACY.md).
//!
//! # Overview
//!
//! The InsPIRe protocol requires careful parameter selection to balance:
//! - **Security**: Lattice-based hardness assumptions (LWE/RLWE)
//! - **Correctness**: Noise growth during homomorphic operations
//! - **Efficiency**: Communication and computation costs
//!
//! # Example
//!
//! ```
//! use raven_inspire::params::{InspireParams, SecurityLevel};
//!
//! // The shipped preset. Measures 121.5 bits, not 128; see PRIVACY.md.
//! let params = InspireParams::secure_128_d2048();
//! assert!(params.validate().is_ok());
//!
//! // Access scaling factor for encoding
//! let delta = params.delta();
//! ```

use crate::math::gaussian::tailcut_for;
use crate::math::primality::is_prime;
use crate::math::NttContext;
use serde::{Deserialize, Serialize};
use subtle::{ConditionallySelectable, ConstantTimeGreater};

/// Default CRT moduli from the InsPIRe reference implementation.
pub const DEFAULT_CRT_MODULI: [u64; 2] = [268_369_921, 249_561_089];

/// Advisory label for the caller's intended security target.
///
/// This label neither selects nor validates parameters.
///
/// # Variants
///
/// * `Bits128` - declares a 128-bit target
/// * `Bits256` - declares a 256-bit target
///
/// Both are declarations. No code reads this field and `validate` runs no
/// lattice estimate, so a variant here is not evidence of the level it names.
///
/// # Example
///
/// ```
/// use raven_inspire::params::SecurityLevel;
///
/// let level = SecurityLevel::Bits128;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Declared 128-bit target. Not validated; see the type-level note.
    Bits128,
    /// Declared 256-bit target. Not validated; see the type-level note.
    Bits256,
}

/// InsPIRe protocol variant controlling the packing strategy.
///
/// Different variants trade off communication size versus server computation time.
/// The InspiRING packing algorithm reduces response size by combining multiple
/// ciphertexts into fewer packed ciphertexts.
///
/// # Variants
///
/// * `NoPacking` - No packing, fastest server response
/// * `OnePacking` - Single-level packing, balanced tradeoff
/// * `TwoPacking` - Two-level packing, minimal communication
///
/// # Example
///
/// ```
/// use raven_inspire::params::InspireVariant;
///
/// let variant = InspireVariant::default(); // NoPacking
/// assert_eq!(variant, InspireVariant::NoPacking);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum InspireVariant {
    /// InsPIRe^0: No packing.
    ///
    /// Returns one RLWE ciphertext per database column. This is the fastest
    /// variant on the server side but has the largest response size.
    /// Best for latency-critical applications, small entries, or debugging.
    #[default]
    NoPacking,

    /// InsPIRe^1: Single-level InspiRING packing.
    ///
    /// Packs multiple ciphertexts using automorphism-based tree packing.
    /// Provides a balanced tradeoff between communication and computation.
    /// Best for general-purpose applications.
    #[allow(dead_code)]
    OnePacking,

    /// InsPIRe^2: Two-level InspiRING packing.
    ///
    /// Applies packing twice for minimal communication overhead.
    /// Slower server response but smallest response size.
    /// Best for bandwidth-constrained environments.
    #[allow(dead_code)]
    TwoPacking,
}

/// Core cryptographic parameters for the InsPIRe PIR protocol.
///
/// These parameters control the security, correctness, and efficiency of the protocol.
/// Parameters must satisfy certain constraints for the NTT and lattice-based security.
///
/// # Fields
///
/// * `ring_dim` - Ring dimension d, must be a power of two (typically 2048 or 4096)
/// * `q` - Ciphertext modulus, must be NTT-friendly: q ≡ 1 (mod 2d)
/// * `p` - Plaintext modulus for message encoding
/// * `sigma` - Standard deviation for discrete Gaussian error sampling
/// * `gadget_base` - Base for gadget decomposition (typically 2^20)
/// * `query_gadget_len` / `packing_gadget_len` - Gadget digits per role
/// * `security_level` - declared target; never read, never validated
///
/// # Example
///
/// ```
/// use raven_inspire::params::InspireParams;
///
/// // Use recommended parameters
/// let params = InspireParams::secure_128_d2048();
///
/// // Validate parameters
/// assert!(params.validate().is_ok());
///
/// // Get scaling factor for encoding
/// let delta = params.delta();
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InspireParams {
    /// Ring dimension d (power of two).
    ///
    /// Determines the polynomial ring R_q = Z_q\[X\]/(X^d + 1).
    /// Larger values provide more noise margin but increase computation.
    /// Typical values: 2048, 4096.
    pub ring_dim: usize,

    /// Ciphertext modulus q.
    ///
    /// Must be NTT-friendly: q ≡ 1 (mod 2d) to enable fast polynomial multiplication.
    /// Typical value: 2^60 - 2^14 + 1 = 1152921504606830593.
    pub q: u64,

    /// CRT moduli for ciphertext representation.
    ///
    /// When length is 1, the scheme uses a single modulus equal to `q`.
    /// When length is 2, coefficients are represented in CRT with these primes
    /// and `q` is their product.
    pub crt_moduli: Vec<u64>,

    /// Plaintext modulus p.
    ///
    /// Messages are encoded in Z_p before scaling by Δ = ⌊q/p⌋.
    /// For 32-byte entries, we use p = 65537 (Fermat prime F4).
    pub p: u64,

    /// Standard deviation for discrete Gaussian error sampling.
    ///
    /// Controls the noise added during encryption for security.
    /// Typical value: 6.4, the width the shipped presets use.
    pub sigma: f64,

    /// Gadget decomposition base z.
    ///
    /// Used for decomposing polynomials into small-norm components.
    /// Larger bases reduce decomposition length but increase noise.
    /// Typical value: 2^20.
    pub gadget_base: u64,

    /// Legacy query decomposition width retained in the serialized parameter identity.
    ///
    /// The one-row fold query does not allocate these digits.
    pub query_gadget_len: usize,

    /// Number of key-switching and InspiRING packing gadget digits.
    pub packing_gadget_len: usize,

    /// Advisory security target; serialized but not validated or dispatched on.
    pub security_level: SecurityLevel,
}

/// `extract_column_value` packs two bytes per column; `p` must exceed this.
const COLUMN_VALUE_CEILING: u64 = u16::MAX as u64;

/// Discrete-Gaussian width floor from the homomorphic-encryption security
/// standard. Below it the LWE error distribution stops hiding the secret:
/// `GaussianSampler` derives `tailcut = ceil(sigma * 6)`, so a small sigma
/// leaves every non-zero table weight underflowed to 0 and the sampler emits a
/// constant, yielding a zero key and a deterministic query.
pub const MIN_SIGMA: f64 = 3.19;

impl InspireParams {
    /// Creates the shipped d=2048 preset.
    ///
    /// The name predates the measurement: at `DEFAULT_Q` this parameter set is
    /// 121.5 bits, not 128 (malb/lattice-estimator @ 3e48ef4, binding attack
    /// `primal_bdd`). log2 q = 57 is the largest modulus clearing 128 bits.
    ///
    /// These are the recommended parameters for most applications, providing
    /// a good balance between security, performance, and noise margin.
    /// Suitable for databases up to ~1GB per shard.
    ///
    /// # Returns
    ///
    /// A new `InspireParams` instance with:
    /// - `ring_dim`: 2048
    /// - `q`: `DEFAULT_Q` = 2^60 - 2^14 + 1, a single NTT-friendly prime
    /// - `p`: 65537 (Fermat prime F4)
    /// - `sigma`: 6.4
    /// - `gadget_base`: 2^20
    /// - `query_gadget_len` / `packing_gadget_len`: 3
    ///
    /// # Example
    ///
    /// ```
    /// use raven_inspire::params::InspireParams;
    ///
    /// let params = InspireParams::secure_128_d2048();
    /// assert_eq!(params.ring_dim, 2048);
    /// assert!(params.validate().is_ok());
    /// ```
    pub fn secure_128_d2048() -> Self {
        // Raven-local patch: use the library-internal
        // DEFAULT_Q = 2^60 - 2^14 + 1 (`src/math/mod_q.rs`) as a
        // single-prime CRT modulus, matching `inspire-rs` internal
        // correctness tests and `inspire-exex` production
        // `default_params()`. The upstream 2-CRT form
        // `[268369921, 249561089]` gave q ≈ 2^55.89, ~4 bits below
        // DEFAULT_Q; at 256 B records (128 InspiRING columns) the
        // gap exhausted the noise budget and decryption scrambled
        // silently. No upstream test ever exercised the shipped
        // preset at a cell above 32 B records.
        let crt_moduli = vec![crate::math::mod_q::DEFAULT_Q];
        let q = crate::math::mod_q::DEFAULT_Q;
        let gadget_base: u64 = 1 << 20; // 2^20
        let gadget_len = ((q as f64).log2() / 20.0).ceil() as usize; // 3

        Self {
            ring_dim: 2048,
            q,
            crt_moduli,
            p: 65537, // Fermat prime F4, coprime with any power-of-2 ring dimension
            sigma: 6.4,
            gadget_base,
            query_gadget_len: gadget_len,
            packing_gadget_len: gadget_len,
            security_level: SecurityLevel::Bits128,
        }
    }

    /// Creates d=4096 parameters carrying a declared 128-bit target.
    ///
    /// The lattice-estimator run that measured `secure_128_d2048` at 121.5 bits
    /// did not cover this ring dimension, so its level is unmeasured.
    ///
    /// These parameters provide more noise margin than d=2048, suitable for
    /// applications requiring additional homomorphic operations or higher
    /// noise tolerance.
    ///
    /// # Returns
    ///
    /// A new `InspireParams` instance with:
    /// - `ring_dim`: 4096
    /// - `q`: `DEFAULT_Q` = 2^60 - 2^14 + 1, a single NTT-friendly prime
    /// - `p`: 65537 (Fermat prime F4)
    /// - `sigma`: 6.4
    /// - `gadget_base`: 2^20
    /// - `query_gadget_len` / `packing_gadget_len`: 3
    ///
    /// # Example
    ///
    /// ```
    /// use raven_inspire::params::InspireParams;
    ///
    /// let params = InspireParams::secure_128_d4096();
    /// assert_eq!(params.ring_dim, 4096);
    /// assert!(params.validate().is_ok());
    /// ```
    pub fn secure_128_d4096() -> Self {
        // Raven-local patch: DEFAULT_Q single-prime form matching
        // the d=2048 preset. See `secure_128_d2048` above for
        // full rationale.
        let crt_moduli = vec![crate::math::mod_q::DEFAULT_Q];
        let q = crate::math::mod_q::DEFAULT_Q;
        let gadget_base: u64 = 1 << 20;
        let gadget_len = ((q as f64).log2() / 20.0).ceil() as usize;

        Self {
            ring_dim: 4096,
            q,
            crt_moduli,
            p: 65537, // Fermat prime F4, coprime with any power-of-2 ring dimension
            sigma: 6.4,
            gadget_base,
            query_gadget_len: gadget_len,
            packing_gadget_len: gadget_len,
            security_level: SecurityLevel::Bits128,
        }
    }

    /// Computes the scaling factor Δ = ⌊q/p⌋.
    ///
    /// The scaling factor is used to encode plaintext messages into ciphertext space.
    /// A message m ∈ Z_p is encoded as Δ·m before encryption, and recovered by
    /// computing ⌊(decrypted_value + Δ/2) / Δ⌋ mod p.
    ///
    /// # Returns
    ///
    /// The scaling factor Δ = ⌊q/p⌋.
    ///
    /// # Example
    ///
    /// ```
    /// use raven_inspire::params::InspireParams;
    ///
    /// let params = InspireParams::secure_128_d2048();
    /// let delta = params.delta();
    /// // delta ≈ 2^44 for q ≈ 2^60 and p ≈ 2^16
    /// assert!(delta > (1 << 39));
    /// ```
    pub fn delta(&self) -> u64 {
        self.q / self.p
    }

    /// Returns the CRT moduli slice.
    pub fn moduli(&self) -> &[u64] {
        &self.crt_moduli
    }

    /// Number of CRT moduli.
    pub fn crt_count(&self) -> usize {
        self.crt_moduli.len()
    }

    /// Create an NTT context for these parameters.
    pub fn ntt_context(&self) -> NttContext {
        NttContext::with_moduli(self.ring_dim, self.moduli())
    }

    /// Validates that the parameters satisfy required constraints.
    ///
    /// Checks that:
    /// - `ring_dim` is a power of two
    /// - `q` is NTT-friendly: q ≡ 1 (mod 2d)
    /// - `q >= p` for valid scaling, and `floor(q/p)/2` wider than the sampler tailcut
    /// - both gadget lengths span every residue modulo `q`
    ///
    /// # Returns
    ///
    /// `Ok(())` if all constraints are satisfied.
    ///
    /// # Errors
    ///
    /// Returns an error string describing the constraint violation:
    /// - `"ring_dim must be a power of two"` if ring_dim is not a power of 2
    /// - `"q must be ≡ 1 (mod 2d) for NTT"` if q is not NTT-friendly
    /// - `"q must be >= p"` if q < p
    /// - a role-specific gadget-width error if either gadget is too narrow
    ///
    /// # Example
    ///
    /// ```
    /// use raven_inspire::params::InspireParams;
    ///
    /// let params = InspireParams::secure_128_d2048();
    /// assert!(params.validate().is_ok());
    ///
    /// // Invalid parameters would fail validation
    /// let invalid = InspireParams {
    ///     ring_dim: 1000, // Not a power of two
    ///     ..params
    /// };
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), &'static str> {
        // Ring dimension must be power of two
        if !self.ring_dim.is_power_of_two() {
            return Err("ring_dim must be a power of two");
        }

        // CRT modulus checks
        if self.crt_moduli.is_empty() {
            return Err("crt_moduli must be non-empty");
        }
        if self.crt_moduli.len() > 2 {
            return Err("crt_moduli length > 2 is not supported");
        }

        let two_n = 2 * self.ring_dim as u64;
        for &m in &self.crt_moduli {
            if m % two_n != 1 {
                return Err("CRT moduli must be congruent to 1 mod 2*ring_dim for the NTT");
            }
            if !is_prime(m) {
                return Err(
                    "CRT moduli must be prime: on a composite the NTT root search never ends, \
                     or accepts a root of the wrong order and decodes to wrong plaintext",
                );
            }
        }

        let crt_product: u64 = self.crt_moduli.iter().product();
        if self.q != crt_product {
            return Err("q must equal product of CRT moduli");
        }
        if self.crt_moduli.len() == 2 {
            let a = self.crt_moduli[0];
            let b = self.crt_moduli[1];
            if gcd_u64(a, b) != 1 {
                return Err("CRT moduli must be coprime");
            }
        }

        // p must be at most q to allow scaling (Δ = ⌊q/p⌋)
        if self.q < self.p {
            return Err("q must be >= p");
        }

        if !self.sigma.is_finite() || self.sigma < MIN_SIGMA {
            return Err(
                "sigma must be finite and >= 3.19: a smaller width collapses the LWE error to a \
                 constant, which zeroes the secret key and makes the query deterministic",
            );
        }

        if self.p <= COLUMN_VALUE_CEILING {
            return Err(
                "p must be > 65535: a column packs two bytes, so a smaller p silently \
                 reduces encoded values",
            );
        }

        // A fresh encryption's error reaches the sampler tailcut, so a rounding half-interval
        // no wider than it decodes a lone ciphertext to a neighbouring plaintext, with no
        // error. Necessary, not sufficient: a packed response carries far more noise.
        if self.delta() / 2 <= tailcut_for(self.sigma) as u64 {
            return Err(
                "floor(q/p)/2 must exceed the sampler tailcut ceil(6 sigma): at this q and p a \
                 single encryption's own error already decodes to a neighbouring plaintext",
            );
        }

        let covering_len =
            min_gadget_len(self.gadget_base, self.q).ok_or("gadget_base must be >= 2")?;
        if self.query_gadget_len < covering_len {
            return Err(
                "gadget_base^query_gadget_len must be >= q; a narrower query gadget drops \
                 high digits and returns wrong plaintext",
            );
        }
        if self.packing_gadget_len < covering_len {
            return Err(
                "gadget_base^packing_gadget_len must be >= q; a narrower packing gadget drops \
                 high digits and returns wrong plaintext",
            );
        }

        // Documentation-only note (no runtime rejection):
        //
        // `pir::extract::extract_packed` computes `mod_inverse(d, p)`.
        // Under p=65537 (Fermat F4, shipping config) + d=2048, gcd
        // is 1 and the inverse exists. Under legacy test params
        // (p=65536, Google's default) gcd is 256 and the inverse
        // does not exist, so `extract_packed` returns the typed
        // `ExtractError::DegreeNotInvertible` error rather than
        // silently producing garbage plaintext. (Pre-fork code used
        // `.unwrap_or(1)` here; that fallback was removed.)
        //
        // A strict rejection was removed from `validate()` because
        // it broke 15 pre-existing unit tests using the p=65536
        // fixture. The strict version is available for callers who
        // want to opt in:
        //   InspireParams::validate_strict_tree_packed()
        //
        // CORRECTION 2026-09-06: this block used to end by claiming the
        // shipping TwoPacking + InspiRING path does not reach
        // extract_packed's d_inv branch, so the invariant was "not load
        // bearing". That was measured false. `extract_two_packing`
        // dispatches on `response.packing_mode`, which is a byte decoded
        // from the SERVER's response, and routes both `Some(Tree)` and
        // `None` into extract_packed. Both shipping client entry points
        // go through it. At d=2048/p=65537 gcd is 1, so the
        // DegreeNotInvertible guard above never fires, and the packed
        // path returns wrong bytes with an Ok status. Reachability here
        // is server-controlled, not a local choice.
        Ok(())
    }

    /// Strict validation variant that additionally enforces
    /// `gcd(ring_dim, p) == 1` so tree-packed extract's `d_inv`
    /// computation always succeeds. Opt-in
    /// because legacy test params use p=65536 which fails this
    /// guard while still exercising non-tree-packed paths.
    pub fn validate_strict_tree_packed(&self) -> Result<(), &'static str> {
        self.validate()?;
        if gcd_u64(self.ring_dim as u64, self.p) != 1 {
            return Err(
                "gcd(ring_dim, p) must equal 1 for tree-packed extract to recover d^{-1} mod p",
            );
        }
        Ok(())
    }
}

/// Smallest `len` with `base^len >= q`. `None` when `base < 2` or no
/// `len <= 64` covers `q`.
fn min_gadget_len(base: u64, q: u64) -> Option<usize> {
    if base < 2 {
        return None;
    }
    let mut covered = base as u128;
    for len in 1..=64usize {
        if covered >= q as u128 {
            return Some(len);
        }
        covered *= base as u128;
    }
    None
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

impl Default for InspireParams {
    fn default() -> Self {
        Self::secure_128_d2048()
    }
}

// ===========================================================================
// Adaptive parameter derivation (Raven-local).
//
// Port of Google's `params_for_scenario_medium_payload` from
// `private-membership/research/InsPIRe/src/params.rs:109-167`.
// Brought into the fork so callers can derive paper-matching `InspireParams`
// per (N, record_size, gamma_triple) without leaving the fork's API surface.
//
// The derivation produces two 2-CRT ~27-bit primes
// (`[67043329, 132120577]`, product ~= 2^52.97, each <= 2^32) matching the
// paper's Table 1 InsPIRe shape. The <= 2^32 per prime is what unblocks NPIR
// NTT + YPIR matmul kernel ports: those kernels downcast moduli to u32
// before entering their AVX-512 lanes. Under a single 60-bit modulus, those
// kernels would produce garbage coefficients; under this derivation they
// are drop-in compatible.
//
// See the noise-budget proof documentation for the cryptographer-review gate.
// ===========================================================================

/// Inputs to the adaptive derivation. Same shape as Google's function.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveInputs {
    /// N - number of database items (2^20 for Raven's SLO cell).
    pub input_num_items: usize,
    /// Record size in bits (2048 for 256 B records).
    pub input_item_size_bits: usize,
    /// gamma triple per paper §7.1 (`[64, 1024, 64]` for 2^20 x 256 B).
    pub gammas: [usize; 3],
    /// Performance factor - power of 2 that skews (nu_1, nu_2) toward
    /// `nu_1 = more` for throughput. Defaults to 1 (no skew).
    pub performance_factor: usize,
}

/// Derived intermediate + final values. Pub so KATs can inspect every
/// field; callers should consume the produced `InspireParams` via
/// [`InspireParams::for_scenario`].
#[derive(Debug, Clone)]
pub struct AdaptiveDerivation {
    pub poly_len: usize,
    pub p: u64,
    pub nu_1: usize,
    pub nu_2: usize,
    pub q2_bits: usize,
    pub t_exp_left: usize,
    pub sigma_x: f64,
    pub z: u64,
    pub db_rows: usize,
    pub db_cols: usize,
    pub working_num_large_items: f64,
    pub num_tiles: f64,
    pub num_tiles_log2: usize,
    pub term_0_variance: f64,
    pub term_1_variance: f64,
    pub term_2_variance: f64,
    pub max_variance: f64,
    pub required_q_log2: f64,
    pub custom_moduli: Vec<u64>,
    pub custom_q_log2: f64,
}

/// `get_variance` from `params.rs:105-107` in Google's reference.
/// Returns variance in `log2` units.
///
/// The `(q~/q)^2` factor of Theorem 7 (eprint 2025/1352) does not belong here:
/// this is the pre-mod-switch variance, and `required_q_log2` sizes `q`, not
/// `q~`. Absent instead is the theorem's additive `d*sigma_chi^2/4` rounding
/// term. Responses are mod-switched, so that term is real; it is charged where
/// the switch happens, at its worst case, by
/// `pir::mod_switch::check_mod_switch_noise_budget`, not here.
///
/// Derivations at 2^20 x 256 B with paper gamma=[64, 1024, 64] clear the noise
/// budget with ~0.09-bit slack under this formula.
#[inline]
fn get_variance(
    dim: f64,
    p: f64,
    sigma_x: f64,
    ell_ks: f64,
    z: f64,
    poly_len: f64,
    gamma: f64,
) -> f64 {
    (dim * p.powi(2) * sigma_x.powi(2)
        + ell_ks * gamma * poly_len * z.powi(2) * sigma_x.powi(2) / 4.0)
        .log2()
        / 2.0
}

/// Port of `params_for_scenario_medium_payload`. Returns the full derivation
/// snapshot; callers typically want [`InspireParams::for_scenario`] which
/// wraps this + the InspireParams bridge + the `validate()` pass.
pub fn derive_medium_payload(inputs: &AdaptiveInputs) -> AdaptiveDerivation {
    let gamma_0 = inputs.gammas[0];
    let gamma_1 = inputs.gammas[1];
    let gamma_2 = inputs.gammas[2];

    let poly_len_log2: usize = 11;
    let poly_len: usize = 1 << poly_len_log2; // 2048

    let log_p: usize = 16;
    assert!(log_p <= 16, "log_p must be <= 16");
    // Raven-local deviation: Google uses p = 1 << 16 = 65536. inspire-rs
    // uses p = 65537 (Fermat prime F4) so that `mod_inverse(d, p)` holds for
    // d = 2048 under the tree-packed extract path (`extract_packed` at
    // extract.rs:135). 65536 is not coprime to 2048; Google's Params only
    // uses the InspiRING extract path, so they don't hit the invariant.
    // The noise formula's `p^2` term changes from 65536^2 to 65537^2 - a
    // delta of ~3.05e-5 bits in log space, well below the 0.01-bit
    // measurement floor. Variance bound is identical to the ported
    // derivation. See the noise-budget analysis for the proof.
    let p: u64 = 65537;

    let q2_bits: usize = 28;
    let t_exp_left: usize = 3;

    let working_num_large_items = (inputs.input_num_items as f64
        * inputs.input_item_size_bits as f64
        / (log_p * gamma_0) as f64)
        .ceil();

    let num_tiles = working_num_large_items / (poly_len * poly_len) as f64;
    let num_tiles_log2 = num_tiles.ceil().log2().ceil() as usize;

    let log_factor = inputs.performance_factor.trailing_zeros() as usize;
    let (nu_1, nu_2) = if num_tiles_log2.is_multiple_of(2) {
        (
            (num_tiles_log2 / 2).saturating_add(log_factor),
            (num_tiles_log2 / 2).saturating_sub(log_factor),
        )
    } else {
        (
            num_tiles_log2.div_ceil(2).saturating_add(log_factor),
            ((num_tiles_log2.saturating_sub(1)) / 2).saturating_sub(log_factor),
        )
    };

    let db_rows = nu_1;
    let db_cols = 1usize << (nu_2 + gamma_0.trailing_zeros() as usize);

    let size_over_t = 1usize << (nu_1 + poly_len_log2);
    let t_val = 1usize << (nu_2 + poly_len_log2);
    let sigma_x: f64 = 6.4;
    let z: u64 = 1 << 19;

    let term_0_variance = get_variance(
        size_over_t as f64,
        p as f64,
        sigma_x,
        t_exp_left as f64,
        z as f64,
        poly_len as f64,
        gamma_0 as f64,
    );
    let term_1_variance = get_variance(
        t_val as f64,
        p as f64,
        sigma_x,
        t_exp_left as f64,
        z as f64,
        poly_len as f64,
        gamma_1 as f64,
    );
    let term_2_variance = get_variance(
        t_val as f64,
        p as f64,
        sigma_x,
        t_exp_left as f64,
        z as f64,
        poly_len as f64,
        gamma_2 as f64,
    );

    let max_variance = term_0_variance.max(term_1_variance).max(term_2_variance);

    let required_q_log2 =
        (2.0 * 2.0 * p as f64).log2() + (2.0 * 41.0 * 2.0f64.ln()).sqrt().log2() + max_variance;

    // Google's CUSTOM_MOD for medium-payload scenarios. Both primes are
    // NTT-friendly: 67043329 - 1 = 67043328 = 4096 * 16368 and
    // 132120577 - 1 = 132120576 = 4096 * 32256, so each ~= 1 (mod 4096),
    // admitting length-2048 negacyclic NTT. Both fit in u32 which is what
    // unblocks NPIR + YPIR kernel ports.
    let custom_moduli: Vec<u64> = vec![67_043_329, 132_120_577];
    let custom_q: f64 = custom_moduli.iter().map(|&m| m as f64).product();
    let custom_q_log2 = custom_q.log2();

    AdaptiveDerivation {
        poly_len,
        p,
        nu_1,
        nu_2,
        q2_bits,
        t_exp_left,
        sigma_x,
        z,
        db_rows,
        db_cols,
        working_num_large_items,
        num_tiles,
        num_tiles_log2,
        term_0_variance,
        term_1_variance,
        term_2_variance,
        max_variance,
        required_q_log2,
        custom_moduli,
        custom_q_log2,
    }
}

impl InspireParams {
    /// Derive paper-aligned parameters for a specific cell shape.
    ///
    /// Ports Google's `params_for_scenario_medium_payload` natively inside
    /// the fork. Returns a validated `InspireParams` with 2-CRT ~27-bit
    /// moduli, p = 65537 (Fermat F4; Raven-local deviation from Google's
    /// p = 65536 documented in the noise-budget analysis), sigma = 6.4,
    /// gadget = (2^19, 3).
    ///
    /// ## Warning: noise-budget under-count for InspiRING packing
    ///
    /// Google's `get_variance` formula (reproduced in this module) covers
    /// Spiral-family LWE + gadget noise only. It does NOT model the
    /// additional noise that inspire-rs's InspiRING 2-matrix packing
    /// (`packing_online`) introduces. Empirical evidence shows that the
    /// derived q ~= 2^53 is
    /// insufficient for 2^20 x 256 B under TwoPacking + InspiRING, even
    /// though this function's noise-budget gate reports ~0.093 bits of
    /// slack. Decryption silently produces random-looking bytes when the
    /// InspiRING-specific noise term crosses the `delta = floor(q/p)`
    /// scaling boundary.
    ///
    /// **Measured 2026-09-20**, 1,000 responses per width on the shipped respond
    /// path (`benches/packing_noise_measurement.rs`), against the decode boundary
    /// `Delta/2 = q/(2p)` = 8,795,958,806,527: **11.225 bits** of margin at
    /// gamma 16 (32 B records) and **9.245 bits** at gamma 256 (512 B records).
    /// The bench asserts the worst sample, so an erosion fails there rather than
    /// scrambling a response in production. That is an empirical margin on the
    /// sampled path, **not** an analytic bound and not a tail bound - this
    /// function's own ~0.093-bit slack figure stays unreliable as a predictor
    /// because the term is still unmodelled. Root `SECURITY.md`, item G6.
    ///
    /// Use [`for_scenario_with_crt`](Self::for_scenario_with_crt) with a
    /// wider 2-CRT pair (typically 2 x 30-bit primes, q ~= 2^60) for the
    /// empirically-correctness-safe shape that matches `DEFAULT_Q`'s
    /// proven headroom while preserving u32-fit per limb (kernel ports
    /// stay unblocked). Keep `for_scenario` for scenarios where the
    /// tree-packed extract path is the only one in use (Google's own
    /// pipeline), or consult the noise-budget analysis for the principled
    /// analysis.
    ///
    /// # Arguments
    /// * `num_items` - number of database entries (2^20 for Raven's SLO cell)
    /// * `record_bytes` - size of each record in bytes (256 for SLO)
    /// * `gammas` - paper §7.1 `[gamma_0, gamma_1, gamma_2]` triple. For
    ///   256 B records, use `[64, 1024, 64]` (paper value); for 32 B use
    ///   `[16, 1024, 16]`.
    /// * `performance_factor` - power-of-two skew factor; 1 = balanced
    ///
    /// # Errors
    /// Returns `Err` if the derived parameters fail `validate()` (NTT
    /// friendliness, modulus coprimality, q >= p). Also returns `Err` if
    /// Google's noise-budget gate fails (`custom_q_log2 < required_q_log2`)
    /// rather than shipping parameters that violate even the Spiral-family
    /// bound.
    pub fn for_scenario(
        num_items: usize,
        record_bytes: usize,
        gammas: [usize; 3],
        performance_factor: usize,
    ) -> Result<Self, &'static str> {
        // Input validation on gammas BEFORE derivation.
        // Paper Algorithm 2 requires γ_0 ≤ ring_dim/2 (= 1024 for
        // the fixed poly_len = 2048) so the partial-packing bound
        // from Theorem 4 holds. Each γ MUST be positive (zero-γ
        // means no packing + div-by-zero in derivation).
        if gammas[0] == 0 || gammas[1] == 0 || gammas[2] == 0 {
            return Err("adaptive derivation: all γ values must be positive");
        }
        if !gammas[0].is_power_of_two() {
            return Err(
                "adaptive derivation: γ_0 must be a power of two (paper Algorithm 2 uses it as a shard factor)",
            );
        }
        if gammas[0] > 1024 {
            return Err(
                "adaptive derivation: γ_0 must be ≤ poly_len/2 (= 1024) per paper Algorithm 2",
            );
        }
        if performance_factor == 0 || !performance_factor.is_power_of_two() {
            return Err("adaptive derivation: performance_factor must be a positive power of two");
        }

        let inputs = AdaptiveInputs {
            input_num_items: num_items,
            input_item_size_bits: record_bytes.checked_mul(8).ok_or("record_bytes overflow")?,
            gammas,
            performance_factor,
        };
        let d = derive_medium_payload(&inputs);

        // Noise-budget gate (Google's formula). If the derived q cannot
        // support even the Spiral-family variance bound, we MUST NOT ship
        // these parameters. Note this gate does NOT cover the InspiRING
        // noise term - see method doc warning.
        if d.custom_q_log2 < d.required_q_log2 {
            return Err(
                "adaptive derivation: custom_q_log2 < required_q_log2 (noise budget violated)",
            );
        }

        // Post-derivation variance-bound check. Each
        // Σ_i variance term must fit under log2(q); otherwise the
        // decryption error exceeds the budget.
        let log2_q = d.custom_q_log2;
        if d.term_0_variance > log2_q || d.term_1_variance > log2_q || d.term_2_variance > log2_q {
            return Err(
                "adaptive derivation: individual variance term exceeds log2(q) (paper Theorems 4/7 violated)",
            );
        }

        let params = Self::from_derivation(&d);
        params.validate()?;
        Ok(params)
    }

    /// Sibling constructor that accepts an explicit CRT-moduli override.
    ///
    /// Use this when Google's derived `custom_moduli` (~27-bit primes
    /// giving q ~= 2^53) under-counts inspire-rs's InspiRING-specific
    /// noise growth and correctness smoke fails at the target cell. The
    /// recommended override pair for 2^20 x 256 B is
    /// [`DEFAULT_Q_2CRT_30BIT`] (two 30-bit NTT-friendly primes, product
    /// q ~= 2^60, matching `DEFAULT_Q`'s proven correctness headroom
    /// while keeping each limb <= 2^32 so the NPIR / YPIR u32-based
    /// AVX-512 kernel ports remain compatible).
    ///
    /// All other derivation fields (poly_len, p=65537, sigma=6.4,
    /// gadget_base=2^19) come from the Google derivation unchanged.
    /// Both gadget lengths are re-derived as `ceil(log_z(q))` from the override:
    /// the derivation's `t_exp_left = 3` covers only Google's ~2^53 q, and
    /// a gadget narrower than q makes every decoded byte wrong.
    ///
    /// # Errors
    /// Returns `Err` if the override fails `validate()` (NTT friendliness,
    /// 2-CRT coprimality, q >= p, gadget coverage), or if the override's
    /// product is LESS than Google's derived q (widening q can't make the
    /// noise budget worse by construction, but narrowing is not allowed via
    /// this path - that would be insecure relative to Google's own
    /// bound).
    pub fn for_scenario_with_crt(
        num_items: usize,
        record_bytes: usize,
        gammas: [usize; 3],
        performance_factor: usize,
        crt_moduli: Vec<u64>,
    ) -> Result<Self, &'static str> {
        let inputs = AdaptiveInputs {
            input_num_items: num_items,
            input_item_size_bits: record_bytes.checked_mul(8).ok_or("record_bytes overflow")?,
            gammas,
            performance_factor,
        };
        let d = derive_medium_payload(&inputs);

        // Override q must not be NARROWER than Google's derivation.
        // Widening is fine (more noise headroom under Google's bound);
        // narrowing would violate even the Spiral-family noise gate.
        let override_q: u128 = crt_moduli.iter().map(|&m| m as u128).product();
        let google_q: u128 = d.custom_moduli.iter().map(|&m| m as u128).product();
        if override_q < google_q {
            return Err(
                "for_scenario_with_crt: override CRT product is narrower than Google's derived \
                 q; widening only",
            );
        }
        if crt_moduli.len() > 2 {
            return Err("for_scenario_with_crt: only 1- or 2-limb CRT supported");
        }

        let q: u64 = crt_moduli
            .iter()
            .try_fold(1u64, |acc, &m| acc.checked_mul(m))
            .ok_or("for_scenario_with_crt: CRT product overflows u64")?;

        let gadget_len = min_gadget_len(d.z, q)
            .ok_or("for_scenario_with_crt: no gadget length <= 64 covers the override q")?
            .max(d.t_exp_left);

        let params = Self {
            ring_dim: d.poly_len,
            q,
            crt_moduli,
            p: d.p,
            sigma: d.sigma_x,
            gadget_base: d.z,
            query_gadget_len: gadget_len,
            packing_gadget_len: gadget_len,
            security_level: SecurityLevel::Bits128,
        };
        params.validate()?;
        Ok(params)
    }

    /// Unchecked conversion for audit-trail KATs comparing a derivation with
    /// its parameters. The sole in-tree caller, [`Self::for_scenario`], validates
    /// the result immediately. Other callers must call [`Self::validate`]
    /// before use; invalid derivations can also overflow the modulus product.
    pub fn from_derivation(d: &AdaptiveDerivation) -> Self {
        let q: u64 = d.custom_moduli.iter().product();
        Self {
            ring_dim: d.poly_len,
            q,
            crt_moduli: d.custom_moduli.clone(),
            p: d.p,
            sigma: d.sigma_x,
            gadget_base: d.z,
            query_gadget_len: d.t_exp_left,
            packing_gadget_len: d.t_exp_left,
            security_level: SecurityLevel::Bits128,
        }
    }
}

/// Default 2-CRT 30-bit NTT-friendly prime pair for the InspiRING
/// empirical-correctness-safe shape at ring dimension 2048.
///
/// Both primes verified by deterministic Miller-Rabin (witnesses
/// `{2,3,5,7,11,13,17,19,23,29,31,37}`, which is a prime-proving set
/// for all n < 3.3e14). They satisfy:
/// - `p[0] = 2^30 - 2^18 + 1 = 1073479681`. 4096 = 2^12 divides p-1.
/// - `p[1] = 1073692673 = 2^30 - 49151`. 4096 = 2^12 divides p-1.
/// - each is prime
/// - each `~= 1 (mod 4096)` so admits length-2048 negacyclic NTT
/// - each `<= 2^30` so products in the AVX-512 u32 kernels
///   `_mm512_mul_epu32` (u32 x u32 -> u64) fit without overflow
/// - gcd = 1 (distinct primes)
/// - product `q = 1152587268104077313, log2(q) ~ 59.9996` matches
///   `DEFAULT_Q`'s empirical correctness ceiling observed across
///   sessions 012-014 under
///   `TwoPacking + InspiRING + respond_seeded_inspiring`.
///
/// Historical note: the user-suggested pair
/// `[2^30 - 2^18 + 1, 2^30 - 2^14 + 1]` was initially adopted but
/// Miller-Rabin showed `2^30 - 2^14 + 1 = 1073725441` is composite
/// (factors verified during Phase E.5.1 search). `2^30 - 2^20 + 1
/// = 1072693249` is also composite. The chosen pair preserves the
/// intent (two 30-bit NTT-friendly primes close to 2^30) while using
/// only primality-verified values.
pub const DEFAULT_Q_2CRT_30BIT: [u64; 2] = [1_073_479_681, 1_073_692_673];

/// Database sharding configuration for large-scale PIR.
///
/// Sharding divides a large database into smaller chunks that can be processed
/// independently. This enables memory-mapped access for databases that exceed
/// available RAM (e.g., Ethereum's 73 GB state).
///
/// # Fields
///
/// * `shard_size_bytes` - Size of each shard in bytes
/// * `entry_size_bytes` - Size of each database entry in bytes
/// * `total_entries` - Total number of entries in the database
///
/// # Example
///
/// ```
/// use raven_inspire::params::ShardConfig;
///
/// // Shard geometry must match the ring dimension: the packing places one
/// // entry per ring coefficient, so a shard holds exactly `ring_dim` entries.
/// let config = ShardConfig::for_ring_dim(2048, 32, 2_417_514_276).unwrap();
/// assert_eq!(config.entries_per_shard(), 2048);
///
/// // Convert global index to shard coordinates
/// let (shard_id, local_idx) = config.try_index_to_shard(100_000_000).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardConfig {
    /// Size of each shard in bytes.
    ///
    /// Must equal `ring_dim * entry_size_bytes`: the InspiRING packing
    /// addresses one entry per ring coefficient.
    pub shard_size_bytes: u64,

    /// Size of each database entry in bytes.
    ///
    /// Common preset: 32 bytes (e.g. fixed-width state entries).
    pub entry_size_bytes: usize,

    /// Total number of entries in the database.
    ///
    /// Example: a corpus of about 2.4 billion entries.
    pub total_entries: u64,
}

/// Whether a shard holds exactly one row per ring coefficient.
///
/// ```
/// use raven_inspire::params::rows_per_shard_match_ring_dim;
///
/// assert!(rows_per_shard_match_ring_dim(2048, 2048));
/// assert!(!rows_per_shard_match_ring_dim(1024, 2048));
/// ```
#[must_use]
pub const fn rows_per_shard_match_ring_dim(entries_per_shard: u64, ring_dim: usize) -> bool {
    entries_per_shard == ring_dim as u64
}

fn div_rem_by_public_divisor(dividend: u64, divisor: u64) -> (u64, u64) {
    debug_assert_ne!(divisor, 0);
    let reciprocal = u64::MAX / divisor;
    let quotient = ((u128::from(dividend) * u128::from(reciprocal)) >> 64) as u64;
    let remainder = dividend.wrapping_sub(quotient.wrapping_mul(divisor));
    let correct = remainder.ct_gt(&divisor.wrapping_sub(1));
    let quotient_correction = u64::conditional_select(&0, &1, correct);
    (
        quotient.wrapping_add(quotient_correction),
        u64::conditional_select(&remainder, &remainder.wrapping_sub(divisor), correct),
    )
}

impl ShardConfig {
    /// Creates the shard configuration for a flat, fixed-width database at a
    /// given ring dimension.
    ///
    /// The geometry is `shard_size_bytes = ring_dim * entry_size_bytes`,
    /// which is what `encode_database` requires: the packing addresses one
    /// entry per ring coefficient, so a shard holding more than `ring_dim`
    /// entries cannot be queried.
    ///
    /// # Arguments
    ///
    /// * `ring_dim` - Ring dimension of the `InspireParams` this database is encoded under
    /// * `entry_size_bytes` - Size of each fixed-width entry in bytes
    /// * `total_entries` - Total number of entries in the database
    ///
    /// # Errors
    ///
    /// Returns `Err` for a zero ring dimension or entry size, when the shard
    /// size overflows `u64`, or when the resulting config fails [`validate`](Self::validate).
    ///
    /// # Example
    ///
    /// ```
    /// use raven_inspire::params::ShardConfig;
    ///
    /// let config = ShardConfig::for_ring_dim(2048, 32, 2_417_514_276).unwrap();
    /// assert_eq!(config.entry_size_bytes, 32);
    /// assert_eq!(config.entries_per_shard(), 2048);
    /// ```
    pub fn for_ring_dim(
        ring_dim: usize,
        entry_size_bytes: usize,
        total_entries: u64,
    ) -> Result<Self, &'static str> {
        if ring_dim == 0 {
            return Err("ShardConfig: ring_dim must be non-zero");
        }
        if entry_size_bytes == 0 {
            return Err("ShardConfig: entry_size_bytes must be non-zero");
        }
        let shard_size_bytes = (ring_dim as u64)
            .checked_mul(entry_size_bytes as u64)
            .ok_or("ShardConfig: ring_dim * entry_size_bytes overflows u64")?;
        let config = Self {
            shard_size_bytes,
            entry_size_bytes,
            total_entries,
        };
        config.validate()?;
        Ok(config)
    }

    /// Creates a shard configuration for a flat, fixed-width database at the
    /// default ring dimension ([`InspireParams::default`]).
    ///
    /// Use [`for_ring_dim`](Self::for_ring_dim) whenever the parameters are
    /// not the default ones. A geometry this constructor cannot satisfy
    /// yields a config that [`validate`](Self::validate) rejects rather than
    /// one that fails later inside `encode_database`.
    ///
    /// # Arguments
    ///
    /// * `entry_size_bytes` - Size of each fixed-width entry in bytes
    /// * `total_entries` - Total number of entries in the database
    ///
    /// # Example
    ///
    /// ```
    /// use raven_inspire::params::ShardConfig;
    ///
    /// // 32-byte entries, about 2.4 billion of them
    /// let config = ShardConfig::for_flat_db(32, 2_417_514_276);
    /// assert_eq!(config.entry_size_bytes, 32);
    /// assert!(config.validate().is_ok());
    /// ```
    pub fn for_flat_db(entry_size_bytes: usize, total_entries: u64) -> Self {
        let ring_dim = InspireParams::default().ring_dim;
        Self::for_ring_dim(ring_dim, entry_size_bytes, total_entries).unwrap_or(Self {
            shard_size_bytes: 0,
            entry_size_bytes,
            total_entries,
        })
    }

    /// Computes the number of entries that fit in each shard.
    ///
    /// # Returns
    ///
    /// The number of entries per shard: `shard_size_bytes / entry_size_bytes`,
    /// or `0` when `entry_size_bytes` is zero.
    ///
    /// # Example
    ///
    /// ```
    /// use raven_inspire::params::ShardConfig;
    ///
    /// let config = ShardConfig::for_ring_dim(2048, 32, 1_000_000).unwrap();
    /// assert_eq!(config.entries_per_shard(), 2048);
    /// ```
    pub fn entries_per_shard(&self) -> u64 {
        self.shard_size_bytes
            .checked_div(self.entry_size_bytes as u64)
            .unwrap_or(0)
    }

    /// Computes the total number of shards needed for the database.
    ///
    /// Uses ceiling division to ensure all entries are covered.
    ///
    /// # Returns
    ///
    /// The number of shards: `ceil(total_entries / entries_per_shard)`, or
    /// `0` when a shard holds no entries.
    ///
    /// # Example
    ///
    /// ```
    /// use raven_inspire::params::ShardConfig;
    ///
    /// // about 2.4 billion 32-byte entries at ring_dim 2048
    /// let config = ShardConfig::for_ring_dim(2048, 32, 2_417_514_276).unwrap();
    /// assert_eq!(config.num_shards(), 2_417_514_276u64.div_ceil(2048));
    /// ```
    pub fn num_shards(&self) -> u64 {
        let entries_per_shard = self.entries_per_shard();
        if entries_per_shard == 0 {
            return 0;
        }
        self.total_entries.div_ceil(entries_per_shard)
    }

    /// Converts a global index to shard coordinates.
    ///
    /// Maps a global database index to the (shard_id, local_index) pair
    /// needed to locate the entry within the sharded database.
    ///
    /// # Arguments
    ///
    /// * `global_idx` - The global index of the entry (0-indexed)
    ///
    /// # Returns
    ///
    /// A tuple `(shard_id, local_index)` where:
    /// - `shard_id` is the shard containing the entry
    /// - `local_index` is the entry's position within that shard
    ///
    /// # Example
    ///
    /// ```
    /// use raven_inspire::params::ShardConfig;
    ///
    /// let config = ShardConfig::for_ring_dim(2048, 32, 2_417_514_276).unwrap();
    /// let (shard_id, local_idx) = config.index_to_shard(100_000_000);
    ///
    /// // Verify roundtrip
    /// let recovered = config.shard_to_index(shard_id, local_idx);
    /// assert_eq!(recovered, 100_000_000);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when the config is degenerate (see
    /// [`try_index_to_shard`](Self::try_index_to_shard) for the same mapping
    /// with a typed error). A wrong-shard query returns a wrong record with
    /// no signal, so an invariant break must not be swallowed here.
    #[allow(
        clippy::panic,
        reason = "documented abort with a typed sibling: try_index_to_shard"
    )]
    #[must_use]
    pub fn index_to_shard(&self, global_idx: u64) -> (u32, u64) {
        match self.try_index_to_shard(global_idx) {
            Ok(coords) => coords,
            Err(e) => panic!("{e}"),
        }
    }

    /// Converts a global index to shard coordinates, reporting a degenerate
    /// config instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns `Err` when a shard holds no entries, or when the shard id
    /// exceeds `u32::MAX` (which would truncate and address the wrong shard).
    ///
    /// # Example
    ///
    /// ```
    /// use raven_inspire::params::ShardConfig;
    ///
    /// let config = ShardConfig::for_ring_dim(2048, 32, 1_000_000).unwrap();
    /// assert_eq!(config.try_index_to_shard(4096).unwrap(), (2, 0));
    /// ```
    pub fn try_index_to_shard(&self, global_idx: u64) -> Result<(u32, u64), &'static str> {
        let entries_per_shard = self.entries_per_shard();
        if entries_per_shard == 0 {
            return Err(
                "ShardConfig: entries_per_shard is zero; shard_size_bytes must be \
                 ring_dim * entry_size_bytes with a non-zero entry size",
            );
        }
        let (shard_id, local_index) = div_rem_by_public_divisor(global_idx, entries_per_shard);
        let shard_id: u32 = shard_id.try_into().map_err(|_| {
            "ShardConfig: shard_id exceeds u32::MAX; raise shard_size_bytes or lower total_entries"
        })?;
        Ok((shard_id, local_index))
    }

    /// Validate the ShardConfig against invariants that downstream
    /// arithmetic assumes: total_entries fits the shard layout,
    /// entries_per_shard is non-zero, and no shard_id overflows u32.
    ///
    /// Call at construction time (e.g. in an adapter before the
    /// first query) to fail fast rather than discover invariant
    /// violations mid-retrieval.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.entries_per_shard() == 0 {
            return Err("ShardConfig: entries_per_shard is zero");
        }
        if self.num_shards() > u32::MAX as u64 {
            return Err(
                "ShardConfig: num_shards exceeds u32::MAX; shard_id would overflow. \
                 Increase shard_size_bytes OR reduce total_entries.",
            );
        }
        Ok(())
    }

    /// Validate the shard geometry against the parameters the database is
    /// encoded under: a shard must hold exactly `ring_dim` entries, because
    /// the packing addresses one entry per ring coefficient.
    ///
    /// # Errors
    ///
    /// Returns `Err` when [`validate`](Self::validate) fails, or when
    /// `entries_per_shard != params.ring_dim`.
    ///
    /// # Example
    ///
    /// ```
    /// use raven_inspire::params::{InspireParams, ShardConfig};
    ///
    /// let params = InspireParams::secure_128_d2048();
    /// let config = ShardConfig::for_ring_dim(params.ring_dim, 32, 100_000).unwrap();
    /// assert!(config.validate_for_params(&params).is_ok());
    /// ```
    pub fn validate_for_params(&self, params: &InspireParams) -> Result<(), &'static str> {
        self.validate()?;
        if !rows_per_shard_match_ring_dim(self.entries_per_shard(), params.ring_dim) {
            return Err(
                "ShardConfig: entries_per_shard must equal params.ring_dim; set \
                 shard_size_bytes = ring_dim * entry_size_bytes (see ShardConfig::for_ring_dim)",
            );
        }
        Ok(())
    }

    /// Converts shard coordinates to a global index.
    ///
    /// Maps a (shard_id, local_index) pair back to the global database index.
    /// This is the inverse of [`index_to_shard`](Self::index_to_shard).
    ///
    /// # Arguments
    ///
    /// * `shard_id` - The shard identifier
    /// * `local_idx` - The entry's position within the shard
    ///
    /// # Returns
    ///
    /// The global index of the entry.
    ///
    /// # Example
    ///
    /// ```
    /// use raven_inspire::params::ShardConfig;
    ///
    /// let config = ShardConfig::for_ring_dim(2048, 32, 2_417_514_276).unwrap();
    ///
    /// // Entry 10 in shard 2
    /// let global_idx = config.shard_to_index(2, 10);
    /// assert_eq!(global_idx, 2 * config.entries_per_shard() + 10);
    /// ```
    pub fn shard_to_index(&self, shard_id: u32, local_idx: u64) -> u64 {
        shard_id as u64 * self.entries_per_shard() + local_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params_valid() {
        let params = InspireParams::default();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_delta_calculation() {
        let params = InspireParams::secure_128_d2048();
        let delta = params.delta();
        // delta = q / p (CRT q for secure_128_d2048)
        assert!(delta > 0);
        assert!(delta > (1 << 39)); // Should be large
    }

    #[test]
    fn test_shard_config() {
        // about 2.4 billion 32-byte entries at the default ring dimension
        let config = ShardConfig::for_flat_db(32, 2_417_514_276);

        let entries_per_shard = config.entries_per_shard();
        assert_eq!(entries_per_shard, 2048);

        assert_eq!(config.num_shards(), 2_417_514_276u64.div_ceil(2048));
    }

    #[test]
    fn test_index_conversion() {
        let config = ShardConfig::for_ring_dim(2048, 32, 2_417_514_276).unwrap();

        // Test roundtrip
        let global_idx = 100_000_000u64;
        let (shard_id, local_idx) = config.index_to_shard(global_idx);
        let recovered = config.shard_to_index(shard_id, local_idx);
        assert_eq!(global_idx, recovered);
    }
}
