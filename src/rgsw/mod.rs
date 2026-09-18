//! One-sided RGSW encryption: `ell` RLWE rows over the gadget powers.
//!
//! Retained for non-live polynomial evaluation and compatibility KATs.

mod external_product;
mod types;

pub use external_product::{
    external_product, external_product_with_ntt_rgsw, gadget_decompose, gadget_reconstruct,
    rgsw_rows_to_ntt, ExternalProductError, RgswRowsNtt,
};
pub use types::{GadgetVector, RgswCiphertext, SeededRgswCiphertext};
