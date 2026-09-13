//! `PirError` is feature-invariant so the public API does not shift with cargo features.

use std::fmt;

use super::query::PackingMode;

/// PIR operation error.
#[derive(Debug)]
pub struct PirError(pub String);

impl fmt::Display for PirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PirError {}

impl PirError {
    /// Wrap a message.
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl From<std::io::Error> for PirError {
    fn from(err: std::io::Error) -> Self {
        Self(err.to_string())
    }
}

impl From<bincode::Error> for PirError {
    fn from(err: bincode::Error) -> Self {
        Self(err.to_string())
    }
}

/// Failures raised by `PIR.Extract`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractError {
    /// `gcd(d, p) != 1`, so the tree-packed path cannot un-scale by `d^{-1} mod p`.
    DegreeNotInvertible {
        /// Ring dimension.
        d: u64,
        /// Plaintext modulus.
        p: u64,
    },
    /// An unpacked response carried a different number of ciphertexts than the record width needs.
    ColumnCountMismatch {
        /// Extractor that rejected the response.
        operation: &'static str,
        /// Ciphertexts received.
        got: usize,
        /// Ciphertexts required.
        expected: usize,
        /// Record width that determined `expected`.
        entry_size: usize,
    },
    /// A packed response carried a different prefix length than the record width requires.
    PackedCoefficientCount {
        /// Extractor that rejected the response.
        operation: &'static str,
        /// Leading coefficients received.
        got: usize,
        /// Leading coefficients required.
        required: usize,
    },
    /// The TwoPacking extractor received a tree-packed or untagged response.
    TwoPackingModeMismatch {
        /// Decoded semantic mode.
        mode: Option<PackingMode>,
        /// Equivalent tag in the optional RIMS codec.
        rims_tag_byte: u8,
    },
}

impl fmt::Display for ExtractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DegreeNotInvertible { d, p } => write!(
                f,
                "extract_packed: d^{{-1}} mod p does not exist (d={d}, p={p}, gcd != 1); \
                 use parameters with gcd(ring_dim, p) == 1"
            ),
            Self::ColumnCountMismatch {
                operation,
                got,
                expected,
                entry_size,
            } => write!(
                f,
                "{operation}: response column-count mismatch for entry_size {entry_size}: \
                 got {got}, expected {expected}; refusing a partial or surplus response"
            ),
            Self::PackedCoefficientCount {
                operation,
                got,
                required,
            } => write!(
                f,
                "{operation}: packed coefficient prefix has {got}, requires exactly {required}; \
                 refusing a partial or surplus response"
            ),
            Self::TwoPackingModeMismatch {
                mode,
                rims_tag_byte,
            } => write!(
                f,
                "TwoPacking extractor refuses decoded packing_mode={mode:?}: response is \
                 tree-packed (RIMS tag byte {rims_tag_byte}) while TwoPacking requires \
                 PackingMode::Inspiring"
            ),
        }
    }
}

impl std::error::Error for ExtractError {}

impl From<ExtractError> for PirError {
    fn from(err: ExtractError) -> Self {
        Self(err.to_string())
    }
}

/// Result carrying `PirError`.
pub type Result<T> = std::result::Result<T, PirError>;

macro_rules! pir_err {
    ($($arg:tt)*) => {
        $crate::pir::error::PirError(format!($($arg)*))
    };
}

pub(crate) use pir_err;
