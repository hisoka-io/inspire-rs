//! Polynomial arithmetic over R_q = Z_q\[X\]/(X^d + 1), coefficient or NTT domain.
//!
//! ```
//! use raven_inspire::math::{Poly, NttContext};
//! use raven_inspire::math::mod_q::DEFAULT_Q;
//!
//! let ctx = NttContext::with_default_q(256);
//! let a = Poly::random(256, DEFAULT_Q);
//! let b = Poly::random(256, DEFAULT_Q);
//! let product = a.mul_ntt(&b, &ctx);
//! ```

use super::crt::{crt_compose_2, crt_decompose_2, crt_modulus, mod_inverse};
use super::gaussian::{
    os_seeded_chacha, os_seeded_chacha_or_abort, EntropyUnavailable, GaussianSampler,
};
use super::mod_q::{ModQ, DEFAULT_Q};
use super::ntt::NttContext;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// CPUID verdict cached once; `is_x86_feature_detected!` per coefficient batch is measurable.
#[cfg(all(feature = "simd-packing-offline", target_arch = "x86_64"))]
fn has_avx512_ifma_cached() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512ifma")
    })
}

/// Polynomial in R_q = Z_q\[X\]/(X^d + 1), CRT-split across up to two moduli.
///
/// # Example
///
/// ```
/// use raven_inspire::math::Poly;
/// use raven_inspire::math::mod_q::DEFAULT_Q;
///
/// let poly = Poly::constant(42, 256, DEFAULT_Q);
/// assert_eq!(poly.coeff(0), 42);
/// assert_eq!(poly.dimension(), 256);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Poly {
    coeffs: Vec<u64>,
    moduli: Vec<u64>,
    q: u64,
    dim: usize,
    crt_q0_inv_mod_q1: u64,
    is_ntt: bool,
}

/// Wire form of [`Poly`].
///
/// A derived `Deserialize` on `Poly` would decouple `dim` from `coeffs.len()` and
/// bypass [`Poly::init_moduli`], the documented sole entry point for the modulus
/// vector. `dim` is a bare fixint, so a length cap on the encoded bytes does not
/// bound it: a small body can carry `dim = 2^44`, and `apply_automorphism` allocates
/// `vec![0u64; poly.dimension()]` before it touches `coeffs`, which aborts the
/// process rather than returning an error.
#[derive(serde::Serialize, serde::Deserialize)]
struct PolyWire {
    coeffs: Vec<u64>,
    moduli: Vec<u64>,
    q: u64,
    dim: usize,
    crt_q0_inv_mod_q1: u64,
    is_ntt: bool,
}

const MAX_TIGHT_COEFFICIENT_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug)]
pub(crate) struct TightBytes(Vec<u8>);

impl TightBytes {
    pub(crate) fn into_vec(self) -> Vec<u8> {
        self.0
    }
}

impl From<Vec<u8>> for TightBytes {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

impl serde::Serialize for TightBytes {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter())
    }
}

impl<'de> serde::Deserialize<'de> for TightBytes {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct TightBytesVisitor;

        impl<'de> serde::de::Visitor<'de> for TightBytesVisitor {
            type Value = TightBytes;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    formatter,
                    "at most {MAX_TIGHT_COEFFICIENT_BYTES} tight coefficient bytes"
                )
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                let declared = sequence.size_hint().unwrap_or(0);
                if declared > MAX_TIGHT_COEFFICIENT_BYTES {
                    return Err(serde::de::Error::custom(format!(
                        "tight coefficient payload declares {declared} bytes, cap is {MAX_TIGHT_COEFFICIENT_BYTES}"
                    )));
                }
                let mut bytes = Vec::with_capacity(declared);
                while let Some(byte) = sequence.next_element()? {
                    if bytes.len() == MAX_TIGHT_COEFFICIENT_BYTES {
                        return Err(serde::de::Error::custom(format!(
                            "tight coefficient payload exceeds {MAX_TIGHT_COEFFICIENT_BYTES}-byte cap"
                        )));
                    }
                    bytes.push(byte);
                }
                Ok(TightBytes(bytes))
            }
        }

        deserializer.deserialize_seq(TightBytesVisitor)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TightPolyWire {
    coeffs: TightBytes,
    moduli: Vec<u64>,
    q: u64,
    dim: usize,
    crt_q0_inv_mod_q1: u64,
    is_ntt: bool,
}

fn tight_coefficient_bits(moduli: &[u64]) -> Result<usize, String> {
    let largest = moduli
        .iter()
        .copied()
        .max()
        .ok_or_else(|| "tight coefficient codec requires at least one modulus".to_owned())?;
    if largest < 2 {
        return Err(format!(
            "tight coefficient codec requires every modulus >= 2, got {largest}"
        ));
    }
    Ok((u64::BITS - largest.saturating_sub(1).leading_zeros()) as usize)
}

pub(crate) fn pack_coefficients_tight(
    coeffs: &[u64],
    moduli: &[u64],
    dim: usize,
) -> Result<Vec<u8>, String> {
    if coeffs.is_empty() && moduli.is_empty() && dim == 0 {
        return Ok(Vec::new());
    }
    let bits = tight_coefficient_bits(moduli)?;
    let expected = dim
        .checked_mul(moduli.len())
        .ok_or_else(|| "tight coefficient count overflow".to_owned())?;
    if coeffs.len() != expected {
        return Err(format!(
            "tight coefficient shape needs {expected} values, got {}",
            coeffs.len()
        ));
    }
    let bit_len = coeffs
        .len()
        .checked_mul(bits)
        .ok_or_else(|| "tight coefficient bit length overflow".to_owned())?;
    let byte_len = bit_len
        .checked_add(7)
        .ok_or_else(|| "tight coefficient byte length overflow".to_owned())?
        / 8;
    let mut packed = vec![0u8; byte_len];
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    for (index, coefficient) in coeffs.iter().copied().enumerate() {
        let modulus = moduli[index / dim];
        if coefficient >= modulus {
            return Err(format!(
                "tight coefficient {coefficient} at index {index} is not canonical for modulus {modulus}"
            ));
        }
        let bit_offset = index * bits;
        let byte_offset = bit_offset / 8;
        let shift = bit_offset % 8;
        let value = coefficient & mask;
        let bytes_for_value = (bits + shift).div_ceil(8);
        let available = packed
            .len()
            .saturating_sub(byte_offset)
            .min(bytes_for_value);
        for byte_index in 0..available {
            let source_shift = byte_index * 8;
            let byte = if source_shift >= shift {
                value >> (source_shift - shift)
            } else {
                value << (shift - source_shift)
            };
            packed[byte_offset + byte_index] |= byte as u8;
        }
    }
    Ok(packed)
}

pub(crate) fn unpack_coefficients_tight(
    packed: &[u8],
    moduli: &[u64],
    dim: usize,
) -> Result<Vec<u64>, String> {
    if packed.is_empty() && moduli.is_empty() && dim == 0 {
        return Ok(Vec::new());
    }
    let bits = tight_coefficient_bits(moduli)?;
    let count = dim
        .checked_mul(moduli.len())
        .ok_or_else(|| "tight coefficient count overflow".to_owned())?;
    let bit_len = count
        .checked_mul(bits)
        .ok_or_else(|| "tight coefficient bit length overflow".to_owned())?;
    let expected_bytes = bit_len
        .checked_add(7)
        .ok_or_else(|| "tight coefficient byte length overflow".to_owned())?
        / 8;
    if packed.len() != expected_bytes {
        return Err(format!(
            "tight coefficient payload has {} bytes, expected {expected_bytes}",
            packed.len()
        ));
    }
    if bit_len % 8 != 0 {
        let used = bit_len % 8;
        let unused_mask = !((1u8 << used) - 1);
        if packed.last().copied().unwrap_or_default() & unused_mask != 0 {
            return Err("tight coefficient payload has nonzero unused high bits".to_owned());
        }
    }
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let mut coeffs = Vec::with_capacity(count);
    for index in 0..count {
        let bit_offset = index * bits;
        let byte_offset = bit_offset / 8;
        let shift = bit_offset % 8;
        let available = packed.len().saturating_sub(byte_offset).min(9);
        let mut window = 0u128;
        for byte_index in 0..available {
            window |= u128::from(packed[byte_offset + byte_index]) << (byte_index * 8);
        }
        let coefficient = ((window >> shift) as u64) & mask;
        let modulus = moduli[index / dim];
        if coefficient >= modulus {
            return Err(format!(
                "tight coefficient {coefficient} at index {index} is not canonical for modulus {modulus}"
            ));
        }
        coeffs.push(coefficient);
    }
    Ok(coeffs)
}

impl serde::Serialize for Poly {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            return PolyWire {
                coeffs: self.coeffs.clone(),
                moduli: self.moduli.clone(),
                q: self.q,
                dim: self.dim,
                crt_q0_inv_mod_q1: self.crt_q0_inv_mod_q1,
                is_ntt: self.is_ntt,
            }
            .serialize(serializer);
        }
        let coeffs = pack_coefficients_tight(&self.coeffs, &self.moduli, self.dim)
            .map_err(serde::ser::Error::custom)?;
        TightPolyWire {
            coeffs: coeffs.into(),
            moduli: self.moduli.clone(),
            q: self.q,
            dim: self.dim,
            crt_q0_inv_mod_q1: self.crt_q0_inv_mod_q1,
            is_ntt: self.is_ntt,
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Poly {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if deserializer.is_human_readable() {
            let wire = PolyWire::deserialize(deserializer)?;
            return Self::try_from(wire).map_err(serde::de::Error::custom);
        }
        let wire = TightPolyWire::deserialize(deserializer)?;
        let coeffs = unpack_coefficients_tight(&wire.coeffs.into_vec(), &wire.moduli, wire.dim)
            .map_err(serde::de::Error::custom)?;
        Self::try_from(PolyWire {
            coeffs,
            moduli: wire.moduli,
            q: wire.q,
            dim: wire.dim,
            crt_q0_inv_mod_q1: wire.crt_q0_inv_mod_q1,
            is_ntt: wire.is_ntt,
        })
        .map_err(serde::de::Error::custom)
    }
}

impl TryFrom<PolyWire> for Poly {
    type Error = String;

    fn try_from(w: PolyWire) -> Result<Self, Self::Error> {
        // `Poly::default()` is all-empty and must still round-trip.
        let is_default = w.coeffs.is_empty()
            && w.moduli.is_empty()
            && w.dim == 0
            && w.q == 0
            && w.crt_q0_inv_mod_q1 == 0;
        if is_default {
            return Ok(Self {
                coeffs: w.coeffs,
                moduli: w.moduli,
                q: w.q,
                dim: w.dim,
                crt_q0_inv_mod_q1: w.crt_q0_inv_mod_q1,
                is_ntt: w.is_ntt,
            });
        }

        let crt_count = w.moduli.len();
        if crt_count == 0 || crt_count > 2 {
            return Err(format!(
                "Poly decode refused: moduli count {crt_count} outside 1..=2, which \
                 init_moduli enforces for every constructed Poly"
            ));
        }
        let expected = w
            .dim
            .checked_mul(crt_count)
            .ok_or_else(|| format!("Poly decode refused: dim {} * {crt_count} overflows", w.dim))?;
        if w.coeffs.len() != expected {
            return Err(format!(
                "Poly decode refused: dim {} with {crt_count} moduli needs {expected} \
                 coefficients, got {}. A dim larger than the coefficients backing it \
                 makes dimension() a length no buffer satisfies",
                w.dim,
                w.coeffs.len()
            ));
        }
        // Count, not value, is all the guards above establish. `crt_modulus` folds with
        // `saturating_mul`, so moduli of 0 give q = 0 and an encoded q of 0 matches;
        // reduction mod 0 then divides by zero, and mod 1 collapses every coefficient to
        // a constant. Neither can arise from `init_moduli`, whose callers come from
        // validated params, so only the wire needs this.
        if let Some(bad) = w.moduli.iter().copied().find(|m| *m < 2) {
            return Err(format!(
                "Poly decode refused: modulus {bad} is below 2. Reduction mod 0 divides by \
                 zero and mod 1 maps every coefficient to 0, so no constructed Poly holds it"
            ));
        }
        // A product that overflows saturates at u64::MAX, which an encoded q of u64::MAX
        // would then match, so the equality check below cannot catch it.
        if w.moduli
            .iter()
            .copied()
            .try_fold(1u64, u64::checked_mul)
            .is_none()
        {
            return Err(format!(
                "Poly decode refused: the encoded moduli {:?} multiply past u64, so their \
                 CRT product saturates and no q can faithfully represent it",
                w.moduli
            ));
        }
        let q = crate::math::crt::crt_modulus(&w.moduli);
        if w.q != q {
            return Err(format!(
                "Poly decode refused: q {} is not the CRT product {q} of the encoded moduli",
                w.q
            ));
        }
        // `try_mod_inverse`, never `mod_inverse`: the latter is documented to panic on a
        // non-invertible value and explicitly forbids callers that cannot establish
        // gcd == 1 via `InspireParams::validate()`. A wire decoder never can, and
        // panicking here would put an abort on the exact input class this guard exists to
        // reject.
        let inv = if crt_count == 2 {
            match crate::math::crt::try_mod_inverse(w.moduli[0], w.moduli[1]) {
                Some(v) => v,
                None => {
                    return Err(format!(
                        "Poly decode refused: encoded moduli {} and {} are not coprime, so \
                         no CRT inverse exists",
                        w.moduli[0], w.moduli[1]
                    ))
                }
            }
        } else {
            0
        };
        if w.crt_q0_inv_mod_q1 != inv {
            return Err(format!(
                "Poly decode refused: crt_q0_inv_mod_q1 {} does not match the inverse {inv} \
                 derived from the encoded moduli",
                w.crt_q0_inv_mod_q1
            ));
        }
        Ok(Self {
            coeffs: w.coeffs,
            moduli: w.moduli,
            q: w.q,
            dim: w.dim,
            crt_q0_inv_mod_q1: w.crt_q0_inv_mod_q1,
            is_ntt: w.is_ntt,
        })
    }
}

impl Poly {
    /// Sole entry point for the modulus vector, so `1 <= moduli.len() <= 2` holds everywhere.
    fn init_moduli(moduli: &[u64]) -> (Vec<u64>, u64, u64) {
        assert!(!moduli.is_empty(), "moduli must be non-empty");
        assert!(
            moduli.len() <= 2,
            "CRT with more than 2 moduli not supported"
        );
        let moduli_vec = moduli.to_vec();
        let q = crt_modulus(&moduli_vec);
        let inv = if moduli_vec.len() == 2 {
            mod_inverse(moduli_vec[0], moduli_vec[1])
        } else {
            0
        };
        (moduli_vec, q, inv)
    }

    /// Zero polynomial.
    pub fn zero(dim: usize, q: u64) -> Self {
        Self::zero_moduli(dim, &[q])
    }

    /// Zero polynomial over CRT moduli.
    pub fn zero_moduli(dim: usize, moduli: &[u64]) -> Self {
        let (moduli_vec, q, inv) = Self::init_moduli(moduli);
        let crt_count = moduli_vec.len();
        Self {
            coeffs: vec![0; dim * crt_count],
            moduli: moduli_vec,
            q,
            dim,
            crt_q0_inv_mod_q1: inv,
            is_ntt: false,
        }
    }

    /// Zero polynomial over `DEFAULT_Q`.
    pub fn zero_default(dim: usize) -> Self {
        Self::zero(dim, DEFAULT_Q)
    }

    /// From standard-form coefficients.
    pub fn from_coeffs(coeffs: Vec<u64>, q: u64) -> Self {
        Self::from_coeffs_moduli(coeffs, &[q])
    }

    /// From coefficients modulo the composite modulus, split into CRT residues.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "owned Vec is the established shape across ~60 call sites"
    )]
    #[must_use]
    pub fn from_coeffs_moduli(coeffs: Vec<u64>, moduli: &[u64]) -> Self {
        let dim = coeffs.len();
        let (moduli_vec, q, inv) = Self::init_moduli(moduli);
        let crt_count = moduli_vec.len();
        let mut crt_coeffs = vec![0u64; dim * crt_count];

        if crt_count == 2 {
            let q0 = moduli_vec[0];
            let q1 = moduli_vec[1];
            for (i, &c) in coeffs.iter().enumerate() {
                let (c0, c1) = crt_decompose_2(c, q0, q1);
                crt_coeffs[i] = c0;
                crt_coeffs[i + dim] = c1;
            }
        } else {
            for (i, &c) in coeffs.iter().enumerate() {
                crt_coeffs[i] = c % moduli_vec[0];
            }
        }

        let mut p = Self {
            coeffs: crt_coeffs,
            moduli: moduli_vec,
            q,
            dim,
            crt_q0_inv_mod_q1: inv,
            is_ntt: false,
        };
        p.reduce();
        p
    }

    /// From residues concatenated by modulus, length `dim * crt_count`.
    pub fn from_crt_coeffs(coeffs: Vec<u64>, moduli: &[u64]) -> Self {
        let (moduli_vec, q, inv) = Self::init_moduli(moduli);
        let crt_count = moduli_vec.len();
        assert!(
            coeffs.len().is_multiple_of(crt_count),
            "CRT coeffs length must be a multiple of crt_count"
        );
        let dim = coeffs.len() / crt_count;
        let mut p = Self {
            coeffs,
            moduli: moduli_vec,
            q,
            dim,
            crt_q0_inv_mod_q1: inv,
            is_ntt: false,
        };
        p.reduce();
        p
    }

    /// [`Self::from_crt_coeffs`] without the trailing reduce pass.
    ///
    /// Caller MUST guarantee `coeffs[m * dim + j] < moduli[m]`; violating it
    /// leaves the `Poly` logically inconsistent (no memory unsafety, so not `unsafe`).
    pub fn from_crt_coeffs_reduced(coeffs: Vec<u64>, moduli: &[u64]) -> Self {
        let (moduli_vec, q, inv) = Self::init_moduli(moduli);
        let crt_count = moduli_vec.len();
        assert!(
            coeffs.len().is_multiple_of(crt_count),
            "CRT coeffs length must be a multiple of crt_count"
        );
        let dim = coeffs.len() / crt_count;
        Self {
            coeffs,
            moduli: moduli_vec,
            q,
            dim,
            crt_q0_inv_mod_q1: inv,
            is_ntt: false,
        }
    }

    /// [`Self::from_coeffs`] over `DEFAULT_Q`.
    pub fn from_coeffs_default(coeffs: Vec<u64>) -> Self {
        Self::from_coeffs(coeffs, DEFAULT_Q)
    }

    /// Constant polynomial.
    pub fn constant(value: u64, dim: usize, q: u64) -> Self {
        Self::constant_moduli(value, dim, &[q])
    }

    /// Constant polynomial over CRT moduli.
    pub fn constant_moduli(value: u64, dim: usize, moduli: &[u64]) -> Self {
        let mut coeffs = vec![0u64; dim];
        coeffs[0] = value;
        Self::from_coeffs_moduli(coeffs, moduli)
    }

    /// Discrete Gaussian error polynomial.
    pub fn sample_gaussian(dim: usize, q: u64, sampler: &mut GaussianSampler) -> Self {
        Self::sample_gaussian_moduli(dim, &[q], sampler)
    }

    /// Discrete Gaussian error polynomial over CRT moduli.
    pub fn sample_gaussian_moduli(
        dim: usize,
        moduli: &[u64],
        sampler: &mut GaussianSampler,
    ) -> Self {
        let (moduli_vec, q, inv) = Self::init_moduli(moduli);
        let crt_count = moduli_vec.len();
        let mut coeffs = vec![0u64; dim * crt_count];
        let samples = sampler.sample_vec(dim);
        for (m, &modulus) in moduli_vec.iter().enumerate() {
            let offset = m * dim;
            for i in 0..dim {
                coeffs[offset + i] = crate::math::ModQ::from_signed(samples[i], modulus);
            }
        }

        Self {
            coeffs,
            moduli: moduli_vec,
            q,
            dim,
            crt_q0_inv_mod_q1: inv,
            is_ntt: false,
        }
    }

    /// Uniform polynomial.
    pub fn random(dim: usize, q: u64) -> Self {
        Self::random_moduli(dim, &[q])
    }

    /// Uniform polynomial from an OS-seeded ChaCha20 stream; aborts on entropy failure.
    /// Use [`Self::try_random_moduli`] to carry that failure instead.
    pub fn random_moduli(dim: usize, moduli: &[u64]) -> Self {
        let mut rng = os_seeded_chacha_or_abort("Poly::random_moduli");
        Self::random_with_rng_moduli(dim, moduli, &mut rng)
    }

    /// [`Self::random_moduli`], surfacing entropy failure as an error.
    pub fn try_random_moduli(dim: usize, moduli: &[u64]) -> Result<Self, EntropyUnavailable> {
        let mut rng = os_seeded_chacha("Poly::try_random_moduli")?;
        Ok(Self::random_with_rng_moduli(dim, moduli, &mut rng))
    }

    /// Uniform polynomial from a caller-supplied RNG.
    pub fn random_with_rng<R: Rng>(dim: usize, q: u64, rng: &mut R) -> Self {
        Self::random_with_rng_moduli(dim, &[q], rng)
    }

    /// Uniform polynomial over CRT moduli from a caller-supplied RNG.
    pub fn random_with_rng_moduli<R: Rng>(dim: usize, moduli: &[u64], rng: &mut R) -> Self {
        let (moduli_vec, q, inv) = Self::init_moduli(moduli);
        let crt_count = moduli_vec.len();
        let mut coeffs = vec![0u64; dim * crt_count];
        for (m, &modulus) in moduli_vec.iter().enumerate() {
            for i in 0..dim {
                coeffs[m * dim + i] = rng.gen_range(0..modulus);
            }
        }
        Self {
            coeffs,
            moduli: moduli_vec,
            q,
            dim,
            crt_q0_inv_mod_q1: inv,
            is_ntt: false,
        }
    }

    /// Deterministic polynomial from a 32-byte seed via ChaCha20.
    pub fn from_seed(seed: &[u8; 32], dim: usize, q: u64) -> Self {
        Self::from_seed_moduli(seed, dim, &[q])
    }

    /// [`Self::from_seed`] over CRT moduli.
    pub fn from_seed_moduli(seed: &[u8; 32], dim: usize, moduli: &[u64]) -> Self {
        let mut rng = ChaCha20Rng::from_seed(*seed);
        Self::random_with_rng_moduli(dim, moduli, &mut rng)
    }

    /// [`Self::from_seed`] with the index XORed into the base seed.
    pub fn from_seed_indexed(seed: &[u8; 32], index: usize, dim: usize, q: u64) -> Self {
        Self::from_seed_indexed_moduli(seed, index, dim, &[q])
    }

    /// [`Self::from_seed_indexed`] over CRT moduli.
    pub fn from_seed_indexed_moduli(
        seed: &[u8; 32],
        index: usize,
        dim: usize,
        moduli: &[u64],
    ) -> Self {
        let mut derived_seed = *seed;
        let idx_bytes = (index as u64).to_le_bytes();
        for i in 0..8 {
            derived_seed[i] ^= idx_bytes[i];
        }
        Self::from_seed_moduli(&derived_seed, dim, moduli)
    }

    /// Ring dimension.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Alias of [`Self::dimension`].
    pub fn len(&self) -> usize {
        self.dim
    }

    /// True when the ring dimension is zero.
    pub fn is_empty(&self) -> bool {
        self.dim == 0
    }

    /// Composite modulus.
    pub fn modulus(&self) -> u64 {
        self.q
    }

    /// The CRT moduli.
    pub fn moduli(&self) -> &[u64] {
        &self.moduli
    }

    /// Number of CRT moduli.
    pub fn crt_count(&self) -> usize {
        self.moduli.len()
    }

    /// True when coefficients hold NTT evaluations.
    pub fn is_ntt(&self) -> bool {
        self.is_ntt
    }

    /// Retags as NTT domain without transforming; caller MUST already hold NTT values.
    #[inline]
    pub fn force_ntt_domain(&mut self) {
        self.is_ntt = true;
    }

    /// Retags as coefficient domain without transforming; caller MUST already hold coefficients.
    #[inline]
    pub fn force_coeff_domain(&mut self) {
        self.is_ntt = false;
    }

    /// Coefficient at `i`; meaningful only in coefficient domain.
    #[inline]
    #[must_use]
    pub fn coeff(&self, i: usize) -> u64 {
        assert!(!self.is_ntt, "Cannot access coefficients in NTT domain");
        if self.moduli.len() == 2 {
            let q0 = self.moduli[0];
            let q1 = self.moduli[1];
            let a0 = self.coeffs[i];
            let a1 = self.coeffs[i + self.dim];
            crt_compose_2(a0, a1, q0, q1, self.crt_q0_inv_mod_q1) % self.q
        } else {
            self.coeffs[i]
        }
    }

    /// Sets coefficient at `i`; meaningful only in coefficient domain.
    #[inline]
    pub fn set_coeff(&mut self, i: usize, value: u64) {
        assert!(!self.is_ntt, "Cannot set coefficients in NTT domain");
        if self.moduli.len() == 2 {
            let (c0, c1) = crt_decompose_2(value, self.moduli[0], self.moduli[1]);
            self.coeffs[i] = c0;
            self.coeffs[i + self.dim] = c1;
        } else {
            self.coeffs[i] = value % self.moduli[0];
        }
    }

    /// Backing values, residues concatenated by modulus.
    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
    }

    /// Mutable [`Self::coeffs`].
    pub fn coeffs_mut(&mut self) -> &mut [u64] {
        &mut self.coeffs
    }

    /// Residue stripe of CRT limb `idx`.
    pub fn coeffs_modulus(&self, modulus_idx: usize) -> &[u64] {
        let start = modulus_idx * self.dim;
        let end = start + self.dim;
        &self.coeffs[start..end]
    }

    /// Mutable [`Self::coeffs_at`].
    pub fn coeffs_modulus_mut(&mut self, modulus_idx: usize) -> &mut [u64] {
        let start = modulus_idx * self.dim;
        let end = start + self.dim;
        &mut self.coeffs[start..end]
    }

    /// Reduces every residue into `[0, moduli[m])`.
    fn reduce(&mut self) {
        for (m, &modulus) in self.moduli.iter().enumerate() {
            let start = m * self.dim;
            let end = start + self.dim;
            for c in &mut self.coeffs[start..end] {
                *c %= modulus;
            }
        }
    }

    /// Transforms into NTT domain in place.
    pub fn to_ntt(&mut self, ctx: &NttContext) {
        if !self.is_ntt {
            debug_assert_eq!(
                self.moduli,
                ctx.moduli(),
                "NTT context moduli must match polynomial moduli"
            );
            ctx.forward(&mut self.coeffs);
            self.is_ntt = true;
        }
    }

    /// Transforms back into coefficient domain in place.
    pub fn from_ntt(&mut self, ctx: &NttContext) {
        if self.is_ntt {
            debug_assert_eq!(
                self.moduli,
                ctx.moduli(),
                "NTT context moduli must match polynomial moduli"
            );
            ctx.inverse(&mut self.coeffs);
            self.is_ntt = false;
        }
    }

    /// [`Self::to_ntt`] on a clone.
    pub fn to_ntt_new(&self, ctx: &NttContext) -> Self {
        let mut result = self.clone();
        result.to_ntt(ctx);
        result
    }

    /// [`Self::from_ntt`] on a clone.
    pub fn from_ntt_new(&self, ctx: &NttContext) -> Self {
        let mut result = self.clone();
        result.from_ntt(ctx);
        result
    }

    /// Scalar product.
    pub fn scalar_mul(&self, scalar: u64) -> Self {
        let mut coeffs = self.coeffs.clone();
        for (m, &modulus) in self.moduli.iter().enumerate() {
            let scalar_mod = scalar % modulus;
            let start = m * self.dim;
            let end = start + self.dim;
            for c in &mut coeffs[start..end] {
                *c = ((*c as u128 * scalar_mod as u128) % modulus as u128) as u64;
            }
        }

        Self {
            coeffs,
            moduli: self.moduli.clone(),
            q: self.q,
            dim: self.dim,
            crt_q0_inv_mod_q1: self.crt_q0_inv_mod_q1,
            is_ntt: self.is_ntt,
        }
    }

    /// In-place scalar product.
    pub fn scalar_mul_assign(&mut self, scalar: u64) {
        for (m, &modulus) in self.moduli.iter().enumerate() {
            let scalar_mod = scalar % modulus;
            let start = m * self.dim;
            let end = start + self.dim;
            for c in &mut self.coeffs[start..end] {
                *c = ((*c as u128 * scalar_mod as u128) % modulus as u128) as u64;
            }
        }
    }

    /// [`Self::scalar_mul`] taking a [`ModQ`].
    pub fn scalar_mul_modq(&self, scalar: ModQ) -> Self {
        self.scalar_mul(scalar.value())
    }

    /// Negacyclic product via NTT.
    pub fn mul_ntt(&self, other: &Self, ctx: &NttContext) -> Self {
        assert_eq!(self.moduli, other.moduli, "Moduli must match");
        assert_eq!(
            self.coeffs.len(),
            other.coeffs.len(),
            "Dimensions must match"
        );

        let mut a = self.clone();
        let mut b = other.clone();

        a.to_ntt(ctx);
        b.to_ntt(ctx);

        let mut result = vec![0u64; self.coeffs.len()];
        ctx.pointwise_mul(&a.coeffs, &b.coeffs, &mut result);

        let mut poly = Self {
            coeffs: result,
            moduli: self.moduli.clone(),
            q: self.q,
            dim: self.dim,
            crt_q0_inv_mod_q1: self.crt_q0_inv_mod_q1,
            is_ntt: true,
        };
        poly.from_ntt(ctx);
        poly
    }

    /// Pointwise product; both operands must be in NTT domain.
    pub fn mul_ntt_domain(&self, other: &Self, ctx: &NttContext) -> Self {
        assert!(
            self.is_ntt && other.is_ntt,
            "Both polynomials must be in NTT domain"
        );
        assert_eq!(self.moduli, other.moduli, "Moduli must match");

        let mut result = vec![0u64; self.coeffs.len()];
        ctx.pointwise_mul(&self.coeffs, &other.coeffs, &mut result);

        Self {
            coeffs: result,
            moduli: self.moduli.clone(),
            q: self.q,
            dim: self.dim,
            crt_q0_inv_mod_q1: self.crt_q0_inv_mod_q1,
            is_ntt: true,
        }
    }

    /// Pointwise sum; both operands must be in NTT domain.
    pub fn add_ntt_domain(&self, other: &Self) -> Self {
        assert!(
            self.is_ntt && other.is_ntt,
            "Both polynomials must be in NTT domain"
        );
        assert_eq!(self.moduli, other.moduli, "Moduli must match");
        assert_eq!(
            self.coeffs.len(),
            other.coeffs.len(),
            "Dimensions must match"
        );

        let mut coeffs = self.coeffs.clone();
        for (m, &modulus) in self.moduli.iter().enumerate() {
            let start = m * self.dim;
            let end = start + self.dim;
            for (c, &o) in coeffs[start..end].iter_mut().zip(&other.coeffs[start..end]) {
                let sum = *c + o;
                *c = if sum >= modulus { sum - modulus } else { sum };
            }
        }

        Self {
            coeffs,
            moduli: self.moduli.clone(),
            q: self.q,
            dim: self.dim,
            crt_q0_inv_mod_q1: self.crt_q0_inv_mod_q1,
            is_ntt: true,
        }
    }

    /// In-place [`Self::add_ntt_domain`].
    pub fn add_assign_ntt_domain(&mut self, other: &Self) {
        assert!(
            self.is_ntt && other.is_ntt,
            "Both polynomials must be in NTT domain"
        );
        assert_eq!(self.moduli, other.moduli, "Moduli must match");

        for (m, &modulus) in self.moduli.iter().enumerate() {
            let start = m * self.dim;
            let end = start + self.dim;
            for i in start..end {
                let sum = self.coeffs[i] + other.coeffs[i];
                self.coeffs[i] = if sum >= modulus { sum - modulus } else { sum };
            }
        }
    }

    /// `self += a * b` pointwise; all three operands must be in NTT domain.
    pub fn mul_acc_ntt_domain(&mut self, a: &Self, b: &Self, ctx: &NttContext) {
        assert!(
            self.is_ntt && a.is_ntt && b.is_ntt,
            "All polynomials must be in NTT domain"
        );
        assert_eq!(self.moduli, a.moduli, "Moduli must match");
        assert_eq!(self.moduli, b.moduli, "Moduli must match");

        // 8-wide IFMA52 kernel, byte-identical to the scalar loop below.
        #[cfg(all(feature = "simd-packing-offline", target_arch = "x86_64"))]
        {
            if let Some(q_inv_neg) = ctx.solinas_q_inv_neg() {
                if self.moduli.len() == 1 && self.dim.is_multiple_of(8) && has_avx512_ifma_cached()
                {
                    // SAFETY: host has AVX-512F + IFMA; single DEFAULT_Q limb of
                    // length `dim` divisible by 8; borrowck rules out aliasing;
                    // NTT-domain precondition puts every input in `[0, DEFAULT_Q)`.
                    unsafe {
                        super::solinas_redc::avx512_ifma::solinas_mont_mul_acc_x8(
                            self.coeffs.as_mut_ptr(),
                            a.coeffs.as_ptr(),
                            b.coeffs.as_ptr(),
                            self.dim,
                            q_inv_neg,
                        );
                    }
                    return;
                }
            }
        }

        for (m, &modulus) in self.moduli.iter().enumerate() {
            let start = m * self.dim;
            let end = start + self.dim;
            for i in start..end {
                let prod = ctx.pointwise_mul_single_at(a.coeffs[i], b.coeffs[i], m);
                let sum = self.coeffs[i] + prod;
                self.coeffs[i] = if sum >= modulus { sum - modulus } else { sum };
            }
        }
    }

    /// `self += a0 * b0 + a1 * b1 + a2 * b2` pointwise, reduced once at the end.
    ///
    /// The single reduction is the point: with `4 * q < 2^64` the accumulator
    /// cannot overflow, so three reduction chains collapse into one.
    ///
    /// # Panics
    ///
    /// If any operand is outside NTT domain, moduli disagree, or `q >= 2^62`.
    pub fn mul_acc3_ntt_domain(
        &mut self,
        a0: &Self,
        b0: &Self,
        a1: &Self,
        b1: &Self,
        a2: &Self,
        b2: &Self,
        ctx: &NttContext,
    ) {
        assert!(
            self.is_ntt
                && a0.is_ntt
                && b0.is_ntt
                && a1.is_ntt
                && b1.is_ntt
                && a2.is_ntt
                && b2.is_ntt,
            "All polynomials must be in NTT domain"
        );
        assert_eq!(self.moduli, a0.moduli, "Moduli must match (a0)");
        assert_eq!(self.moduli, b0.moduli, "Moduli must match (b0)");
        assert_eq!(self.moduli, a1.moduli, "Moduli must match (a1)");
        assert_eq!(self.moduli, b1.moduli, "Moduli must match (b1)");
        assert_eq!(self.moduli, a2.moduli, "Moduli must match (a2)");
        assert_eq!(self.moduli, b2.moduli, "Moduli must match (b2)");

        for (m, &modulus) in self.moduli.iter().enumerate() {
            assert!(
                modulus < (1u64 << 62),
                "mul_acc3_ntt_domain requires modulus < 2^62 to avoid u64 overflow"
            );
            let start = m * self.dim;
            let end = start + self.dim;
            for i in start..end {
                let p0 = ctx.pointwise_mul_single_at(a0.coeffs[i], b0.coeffs[i], m);
                let p1 = ctx.pointwise_mul_single_at(a1.coeffs[i], b1.coeffs[i], m);
                let p2 = ctx.pointwise_mul_single_at(a2.coeffs[i], b2.coeffs[i], m);
                // Accumulator < 4q, so three conditional subtracts suffice.
                let mut acc = self.coeffs[i] + p0 + p1 + p2;
                if acc >= modulus {
                    acc -= modulus;
                }
                if acc >= modulus {
                    acc -= modulus;
                }
                if acc >= modulus {
                    acc -= modulus;
                }
                self.coeffs[i] = acc;
            }
        }
    }

    /// True when every coefficient is zero.
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|&c| c == 0)
    }

    /// L-infinity norm in centered representation, `min(c, q - c)` maximized.
    pub fn linf_norm(&self) -> u64 {
        assert!(!self.is_ntt, "Cannot compute norm in NTT domain");
        let mut max_val = 0u64;
        for i in 0..self.dim {
            let c = self.coeff(i);
            let centered = if c <= self.q / 2 { c } else { self.q - c };
            if centered > max_val {
                max_val = centered;
            }
        }
        max_val
    }

    /// Squared L2 norm in centered representation.
    pub fn l2_norm_squared(&self) -> u128 {
        assert!(!self.is_ntt, "Cannot compute norm in NTT domain");
        let mut sum = 0u128;
        for i in 0..self.dim {
            let c = self.coeff(i);
            let centered = if c <= self.q / 2 {
                c as i64
            } else {
                c as i64 - self.q as i64
            };
            sum += (centered as i128 * centered as i128) as u128;
        }
        sum
    }

    pub fn mul(&self, other: &Self) -> Self {
        let ctx = NttContext::with_moduli(self.dim, &self.moduli);
        self.mul_ntt(other, &ctx)
    }

    pub fn add(&self, other: &Self) -> Self {
        self + other
    }

    pub fn sub(&self, other: &Self) -> Self {
        self - other
    }

    pub fn negate(&self) -> Self {
        -self
    }
}

impl PartialEq for Poly {
    fn eq(&self, other: &Self) -> bool {
        self.q == other.q
            && self.moduli == other.moduli
            && self.dim == other.dim
            && self.is_ntt == other.is_ntt
            && self.coeffs == other.coeffs
    }
}

impl Eq for Poly {}

impl Add for Poly {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Add for &Poly {
    type Output = Poly;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.moduli, rhs.moduli, "Moduli must match");
        assert_eq!(self.is_ntt, rhs.is_ntt, "NTT domains must match");

        let mut coeffs = self.coeffs.clone();
        for (m, &modulus) in self.moduli.iter().enumerate() {
            let start = m * self.dim;
            let end = start + self.dim;
            for (c, &r) in coeffs[start..end].iter_mut().zip(&rhs.coeffs[start..end]) {
                let sum = *c + r;
                *c = if sum >= modulus { sum - modulus } else { sum };
            }
        }

        Poly {
            coeffs,
            moduli: self.moduli.clone(),
            q: self.q,
            dim: self.dim,
            crt_q0_inv_mod_q1: self.crt_q0_inv_mod_q1,
            is_ntt: self.is_ntt,
        }
    }
}

impl AddAssign for Poly {
    fn add_assign(&mut self, rhs: Self) {
        *self = &*self + &rhs;
    }
}

impl AddAssign<&Poly> for Poly {
    fn add_assign(&mut self, rhs: &Self) {
        *self = &*self + rhs;
    }
}

impl Sub for Poly {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Sub for &Poly {
    type Output = Poly;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.moduli, rhs.moduli, "Moduli must match");
        assert_eq!(self.is_ntt, rhs.is_ntt, "NTT domains must match");

        let mut coeffs = self.coeffs.clone();
        for (m, &modulus) in self.moduli.iter().enumerate() {
            let start = m * self.dim;
            let end = start + self.dim;
            for (c, &b) in coeffs[start..end].iter_mut().zip(&rhs.coeffs[start..end]) {
                let a = *c;
                *c = if a >= b { a - b } else { modulus - b + a };
            }
        }

        Poly {
            coeffs,
            moduli: self.moduli.clone(),
            q: self.q,
            dim: self.dim,
            crt_q0_inv_mod_q1: self.crt_q0_inv_mod_q1,
            is_ntt: self.is_ntt,
        }
    }
}

impl SubAssign for Poly {
    fn sub_assign(&mut self, rhs: Self) {
        *self = &*self - &rhs;
    }
}

impl SubAssign<&Poly> for Poly {
    fn sub_assign(&mut self, rhs: &Self) {
        *self = &*self - rhs;
    }
}

impl Neg for Poly {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl Neg for &Poly {
    type Output = Poly;

    fn neg(self) -> Self::Output {
        let mut coeffs = self.coeffs.clone();
        for (m, &modulus) in self.moduli.iter().enumerate() {
            let start = m * self.dim;
            let end = start + self.dim;
            for c in &mut coeffs[start..end] {
                *c = if *c == 0 { 0 } else { modulus - *c };
            }
        }

        Poly {
            coeffs,
            moduli: self.moduli.clone(),
            q: self.q,
            dim: self.dim,
            crt_q0_inv_mod_q1: self.crt_q0_inv_mod_q1,
            is_ntt: self.is_ntt,
        }
    }
}

impl Mul for Poly {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.is_ntt, rhs.is_ntt, "NTT domains must match");
        assert!(
            self.is_ntt,
            "Use mul_ntt for coefficient domain multiplication"
        );

        let ctx = NttContext::with_moduli(self.dim, &self.moduli);
        self.mul_ntt_domain(&rhs, &ctx)
    }
}

impl MulAssign for Poly {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn make_ctx(n: usize) -> NttContext {
        NttContext::with_default_q(n)
    }

    #[test]
    fn test_zero_polynomial() {
        let p = Poly::zero_default(256);
        assert!(p.is_zero());
        assert_eq!(p.dimension(), 256);
    }

    #[test]
    fn poly_random_with_rng_moduli_is_deterministic_under_seeded_rng() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let dim = 64usize;
        let moduli = &[DEFAULT_Q];

        let mut rng_a = ChaCha20Rng::seed_from_u64(0xCAFE_F00D_DEAD_BEEF);
        let p_a = Poly::random_with_rng_moduli(dim, moduli, &mut rng_a);

        let mut rng_b = ChaCha20Rng::seed_from_u64(0xCAFE_F00D_DEAD_BEEF);
        let p_b = Poly::random_with_rng_moduli(dim, moduli, &mut rng_b);

        assert_eq!(
            p_a.coeffs(),
            p_b.coeffs(),
            "same seed must produce identical coefficients"
        );

        let mut rng_diff = ChaCha20Rng::seed_from_u64(0x1234_5678_9ABC_DEF0);
        let p_diff = Poly::random_with_rng_moduli(dim, moduli, &mut rng_diff);
        assert_ne!(
            p_a.coeffs(),
            p_diff.coeffs(),
            "different seed must (with overwhelming probability) produce different coefficients"
        );
    }

    #[test]
    fn test_constant_polynomial() {
        let p = Poly::constant(42, 256, DEFAULT_Q);
        assert_eq!(p.coeff(0), 42);
        assert!(p.coeffs()[1..].iter().all(|&c| c == 0));
    }

    #[test]
    fn test_addition() {
        let a = Poly::from_coeffs_default(vec![1, 2, 3, 4]);
        let b = Poly::from_coeffs_default(vec![5, 6, 7, 8]);
        let c = &a + &b;

        assert_eq!(c.coeff(0), 6);
        assert_eq!(c.coeff(1), 8);
        assert_eq!(c.coeff(2), 10);
        assert_eq!(c.coeff(3), 12);
    }

    #[test]
    fn test_subtraction() {
        let q = DEFAULT_Q;
        let a = Poly::from_coeffs(vec![10, 20, 30, 40], q);
        let b = Poly::from_coeffs(vec![5, 6, 7, 8], q);
        let c = &a - &b;

        assert_eq!(c.coeff(0), 5);
        assert_eq!(c.coeff(1), 14);
        assert_eq!(c.coeff(2), 23);
        assert_eq!(c.coeff(3), 32);
    }

    #[test]
    fn test_subtraction_underflow() {
        let q = DEFAULT_Q;
        let a = Poly::from_coeffs(vec![5, 6, 7, 8], q);
        let b = Poly::from_coeffs(vec![10, 20, 30, 40], q);
        let c = &a - &b;

        assert_eq!(c.coeff(0), q - 5);
        assert_eq!(c.coeff(1), q - 14);
    }

    #[test]
    fn test_negation() {
        let q = DEFAULT_Q;
        let a = Poly::from_coeffs(vec![1, 2, 3, 0], q);
        let neg_a = -&a;

        assert_eq!(neg_a.coeff(0), q - 1);
        assert_eq!(neg_a.coeff(1), q - 2);
        assert_eq!(neg_a.coeff(2), q - 3);
        assert_eq!(neg_a.coeff(3), 0);

        let sum = &a + &neg_a;
        assert!(sum.is_zero());
    }

    #[test]
    fn test_scalar_multiplication() {
        let a = Poly::from_coeffs_default(vec![1, 2, 3, 4]);
        let b = a.scalar_mul(10);

        assert_eq!(b.coeff(0), 10);
        assert_eq!(b.coeff(1), 20);
        assert_eq!(b.coeff(2), 30);
        assert_eq!(b.coeff(3), 40);
    }

    #[test]
    fn test_ntt_roundtrip() {
        let ctx = make_ctx(256);
        let mut p = Poly::from_coeffs_default((0..256).collect());

        let original = p.clone();
        p.to_ntt(&ctx);
        assert!(p.is_ntt());
        p.from_ntt(&ctx);
        assert!(!p.is_ntt());

        assert_eq!(p, original);
    }

    #[test]
    fn test_poly_mul_ntt_identity() {
        let n = 256;
        let ctx = make_ctx(n);

        let a = Poly::from_coeffs_default((0..n as u64).collect());
        let one = Poly::constant(1, n, DEFAULT_Q);

        let result = a.mul_ntt(&one, &ctx);
        assert_eq!(result, a);
    }

    #[test]
    fn test_poly_mul_ntt_zero() {
        let n = 256;
        let ctx = make_ctx(n);

        let a = Poly::from_coeffs_default((0..n as u64).collect());
        let zero = Poly::zero_default(n);

        let result = a.mul_ntt(&zero, &ctx);
        assert!(result.is_zero());
    }

    #[test]
    fn test_poly_mul_ntt_simple() {
        let n = 256;
        let ctx = make_ctx(n);
        let q = DEFAULT_Q;

        let mut coeffs = vec![0u64; n];
        coeffs[0] = 1;
        coeffs[1] = 1;
        let a = Poly::from_coeffs(coeffs, q);

        let result = a.mul_ntt(&a, &ctx);

        assert_eq!(result.coeff(0), 1);
        assert_eq!(result.coeff(1), 2);
        assert_eq!(result.coeff(2), 1);
        assert!(result.coeffs()[3..].iter().all(|&c| c == 0));
    }

    #[test]
    fn test_poly_mul_ntt_negacyclic() {
        let n = 256;
        let ctx = make_ctx(n);
        let q = DEFAULT_Q;

        let mut a_coeffs = vec![0u64; n];
        a_coeffs[1] = 1; // x
        let a = Poly::from_coeffs(a_coeffs, q);

        let mut b_coeffs = vec![0u64; n];
        b_coeffs[n - 1] = 1; // x^(n-1)
        let b = Poly::from_coeffs(b_coeffs, q);

        let result = a.mul_ntt(&b, &ctx);

        assert_eq!(result.coeff(0), q - 1); // -1 mod q
        assert!(result.coeffs()[1..].iter().all(|&c| c == 0));
    }

    #[test]
    fn test_poly_mul_associativity() {
        let n = 256;
        let ctx = make_ctx(n);
        let q = DEFAULT_Q;

        let a = Poly::from_coeffs((0..n as u64).map(|i| i % 100).collect(), q);
        let b = Poly::from_coeffs((0..n as u64).map(|i| (i * 7) % 100).collect(), q);
        let c = Poly::from_coeffs((0..n as u64).map(|i| (i * 13) % 100).collect(), q);

        let ab = a.mul_ntt(&b, &ctx);
        let ab_c = ab.mul_ntt(&c, &ctx);

        let bc = b.mul_ntt(&c, &ctx);
        let a_bc = a.mul_ntt(&bc, &ctx);

        assert_eq!(ab_c, a_bc);
    }

    #[test]
    fn test_poly_mul_commutativity() {
        let n = 256;
        let ctx = make_ctx(n);
        let q = DEFAULT_Q;

        let a = Poly::from_coeffs((0..n as u64).map(|i| i % 100).collect(), q);
        let b = Poly::from_coeffs((0..n as u64).map(|i| (i * 7) % 100).collect(), q);

        let ab = a.mul_ntt(&b, &ctx);
        let ba = b.mul_ntt(&a, &ctx);

        assert_eq!(ab, ba);
    }

    #[test]
    fn test_poly_mul_distributivity() {
        let n = 256;
        let ctx = make_ctx(n);
        let q = DEFAULT_Q;

        let a = Poly::from_coeffs((0..n as u64).map(|i| i % 50).collect(), q);
        let b = Poly::from_coeffs((0..n as u64).map(|i| (i * 3) % 50).collect(), q);
        let c = Poly::from_coeffs((0..n as u64).map(|i| (i * 5) % 50).collect(), q);

        let b_plus_c = &b + &c;
        let left = a.mul_ntt(&b_plus_c, &ctx);

        let ab = a.mul_ntt(&b, &ctx);
        let ac = a.mul_ntt(&c, &ctx);
        let right = &ab + &ac;

        assert_eq!(left, right);
    }

    #[test]
    fn test_linf_norm() {
        let q = DEFAULT_Q;
        let mut coeffs = vec![0u64; 16];
        coeffs[0] = 100;
        coeffs[1] = q - 50; // represents -50
        let p = Poly::from_coeffs(coeffs, q);

        assert_eq!(p.linf_norm(), 100);
    }

    #[test]
    fn test_l2_norm() {
        let q = DEFAULT_Q;
        let mut coeffs = vec![0u64; 4];
        coeffs[0] = 3;
        coeffs[1] = 4;
        let p = Poly::from_coeffs(coeffs, q);

        assert_eq!(p.l2_norm_squared(), 25);
    }

    #[test]
    fn test_ntt_domain_multiplication() {
        let n = 256;
        let ctx = make_ctx(n);
        let q = DEFAULT_Q;

        let a = Poly::from_coeffs((0..n as u64).map(|i| i % 100).collect(), q);
        let b = Poly::from_coeffs((0..n as u64).map(|i| (i * 7) % 100).collect(), q);

        let result1 = a.mul_ntt(&b, &ctx);

        let a_ntt = a.to_ntt_new(&ctx);
        let b_ntt = b.to_ntt_new(&ctx);
        let mut result2 = a_ntt.mul_ntt_domain(&b_ntt, &ctx);
        result2.from_ntt(&ctx);

        assert_eq!(result1, result2);
    }

    #[test]
    fn tight_60bit_codec_round_trips_boundaries_and_exact_size() {
        let values = vec![
            0,
            1,
            (1u64 << 59) - 1,
            1u64 << 59,
            DEFAULT_Q - 2,
            DEFAULT_Q - 1,
            17,
            23,
        ];
        let packed = pack_coefficients_tight(&values, &[DEFAULT_Q], values.len()).unwrap();
        assert_eq!(packed.len(), 60);
        assert_eq!(
            unpack_coefficients_tight(&packed, &[DEFAULT_Q], values.len()).unwrap(),
            values
        );
    }

    #[test]
    fn tight_codec_refuses_shape_truncation_and_noncanonical_values() {
        let values = vec![0, DEFAULT_Q - 1, 7, 9];
        let mut packed = pack_coefficients_tight(&values, &[DEFAULT_Q], values.len()).unwrap();
        packed.pop();
        assert!(
            unpack_coefficients_tight(&packed, &[DEFAULT_Q], values.len())
                .unwrap_err()
                .contains("expected")
        );
        assert!(pack_coefficients_tight(&[DEFAULT_Q], &[DEFAULT_Q], 1)
            .unwrap_err()
            .contains("not canonical"));

        let encoded_noncanonical = pack_coefficients_tight(&[DEFAULT_Q], &[1u64 << 60], 1).unwrap();
        assert!(
            unpack_coefficients_tight(&encoded_noncanonical, &[DEFAULT_Q], 1)
                .unwrap_err()
                .contains("not canonical")
        );

        let mut nonzero_padding = pack_coefficients_tight(&[1], &[257], 1).unwrap();
        nonzero_padding[1] |= 0x80;
        assert!(unpack_coefficients_tight(&nonzero_padding, &[257], 1)
            .unwrap_err()
            .contains("unused high bits"));

        let mut overlong = pack_coefficients_tight(&[1], &[257], 1).unwrap();
        overlong.push(0);
        assert!(unpack_coefficients_tight(&overlong, &[257], 1)
            .unwrap_err()
            .contains("expected"));
    }

    #[test]
    fn tight_codec_round_trips_two_crt_limbs_without_mixing_them() {
        let moduli = [257, 769];
        let values = vec![0, 256, 1, 200, 0, 768, 500, 42];
        let packed = pack_coefficients_tight(&values, &moduli, 4).unwrap();
        assert_eq!(
            unpack_coefficients_tight(&packed, &moduli, 4).unwrap(),
            values
        );
    }

    #[test]
    fn tight_codec_covers_every_bit_offset_and_width_through_64() {
        let mut observed_offsets = [false; 8];
        for bits in 2usize..=64 {
            let modulus = if bits == 64 {
                u64::MAX
            } else {
                (1u64 << (bits - 1)) + 1
            };
            let coefficients: Vec<u64> = (0..17)
                .map(|index| {
                    u64::try_from((index as u128 * 0x9e37 + 1) % u128::from(modulus))
                        .expect("reduced coefficient fits u64")
                })
                .collect();
            for index in 0..coefficients.len() {
                observed_offsets[(index * bits) % 8] = true;
            }
            let packed = pack_coefficients_tight(&coefficients, &[modulus], coefficients.len())
                .expect("width fixture packs");
            assert_eq!(packed.len(), (coefficients.len() * bits).div_ceil(8));
            assert_eq!(
                unpack_coefficients_tight(&packed, &[modulus], coefficients.len())
                    .expect("width fixture unpacks"),
                coefficients,
                "bits={bits}"
            );
        }
        assert!(observed_offsets.into_iter().all(|observed| observed));
    }

    #[test]
    fn tight_codec_64bit_fallback_round_trips_boundaries() {
        let coefficients = vec![0, 1, 1u64 << 63, u64::MAX - 1];
        let packed = pack_coefficients_tight(&coefficients, &[u64::MAX], coefficients.len())
            .expect("64-bit coefficients pack");
        assert_eq!(packed.len(), 32);
        assert_eq!(
            unpack_coefficients_tight(&packed, &[u64::MAX], coefficients.len())
                .expect("64-bit coefficients unpack"),
            coefficients
        );
    }

    #[test]
    fn tight_byte_vector_refuses_an_oversized_declared_length_before_allocation() {
        let forged = u64::MAX.to_le_bytes();
        let error = bincode::deserialize::<TightBytes>(&forged)
            .expect_err("oversized tight byte vector must be refused");
        assert!(error.to_string().contains("cap is"), "{error}");
    }

    // Every boundary below is spelled as a literal rather than as
    // `MAX_TIGHT_COEFFICIENT_BYTES`: a test written in terms of the constant moves with it,
    // and the constant is the only thing between a forged length prefix and an allocation
    // the process cannot survive.
    #[test]
    fn tight_byte_cap_refuses_one_byte_over_and_still_decodes_exactly_at_the_cap() {
        let mut over_by_one = 16_777_217u64.to_le_bytes().to_vec();
        over_by_one.push(0);
        let error = bincode::deserialize::<TightBytes>(&over_by_one)
            .expect_err("a declared length one byte over the cap must be refused")
            .to_string();
        assert!(error.contains("declares 16777217 bytes"), "{error}");
        assert!(error.contains("cap is 16777216"), "{error}");

        // The refusal has to hold for the shape a snapshot actually decodes, not just for the
        // field type. `coeffs` is the first field of the tight wire form, so a forged prefix
        // is refused before any other field is read.
        let mut forged_poly = 16_777_217u64.to_le_bytes().to_vec();
        forged_poly.push(0);
        let error = bincode::deserialize::<Poly>(&forged_poly)
            .expect_err("a forged Poly coefficient length must be refused")
            .to_string();
        assert!(error.contains("cap is 16777216"), "{error}");

        // Inclusive: the declared-length guard is `>`, and the accumulation guard stops one
        // past the cap, so exactly 16 MiB is inside the contract and must still decode.
        let mut at_cap = 16_777_216u64.to_le_bytes().to_vec();
        at_cap.resize(at_cap.len() + 16_777_216, 0);
        assert_eq!(
            bincode::deserialize::<TightBytes>(&at_cap)
                .expect("a declared length exactly at the cap is inside the contract")
                .into_vec()
                .len(),
            16_777_216
        );

        assert_eq!(MAX_TIGHT_COEFFICIENT_BYTES, 16_777_216);
    }

    /// bincode reports the declared length as the seq size hint, so the declared-length guard
    /// always fires first and the accumulation guard is unreachable through it. Withholding
    /// the hint is the only way to reach the guard that bounds what a lying hint would admit.
    struct UnhintedBytes {
        remaining: usize,
    }

    impl<'de> serde::Deserializer<'de> for UnhintedBytes {
        type Error = serde::de::value::Error;

        fn deserialize_seq<V: serde::de::Visitor<'de>>(
            self,
            visitor: V,
        ) -> Result<V::Value, Self::Error> {
            visitor.visit_seq(self)
        }

        fn deserialize_any<V: serde::de::Visitor<'de>>(
            self,
            _visitor: V,
        ) -> Result<V::Value, Self::Error> {
            Err(serde::de::Error::custom(
                "UnhintedBytes yields sequences only",
            ))
        }

        serde::forward_to_deserialize_any! {
            bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes byte_buf option
            unit unit_struct newtype_struct tuple tuple_struct map struct enum identifier
            ignored_any
        }
    }

    impl<'de> serde::de::SeqAccess<'de> for UnhintedBytes {
        type Error = serde::de::value::Error;

        fn size_hint(&self) -> Option<usize> {
            None
        }

        fn next_element_seed<T: serde::de::DeserializeSeed<'de>>(
            &mut self,
            seed: T,
        ) -> Result<Option<T::Value>, Self::Error> {
            if self.remaining == 0 {
                return Ok(None);
            }
            self.remaining -= 1;
            seed.deserialize(serde::de::IntoDeserializer::into_deserializer(0u8))
                .map(Some)
        }
    }

    #[test]
    fn tight_byte_cap_bounds_a_stream_that_withholds_its_length() {
        let error = <TightBytes as serde::Deserialize>::deserialize(UnhintedBytes {
            remaining: 16_777_217,
        })
        .map(|accepted| accepted.into_vec().len())
        .expect_err("an unhinted stream past the cap must be refused")
        .to_string();
        assert!(error.contains("exceeds 16777216-byte cap"), "{error}");
    }

    proptest! {
        #[test]
        fn tight_codec_round_trips_arbitrary_default_q_coefficients(
            coefficients in proptest::collection::vec(0u64..DEFAULT_Q, 1..64)
        ) {
            let packed = pack_coefficients_tight(
                &coefficients,
                &[DEFAULT_Q],
                coefficients.len(),
            ).expect("canonical coefficients pack");
            prop_assert_eq!(
                unpack_coefficients_tight(&packed, &[DEFAULT_Q], coefficients.len())
                    .expect("packed coefficients unpack"),
                coefficients
            );
        }
    }
}
