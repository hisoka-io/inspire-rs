//! One-sided RGSW encryption: `ell` RLWE rows over the gadget powers.
//!
//! Used by polynomial evaluation and its KATs; not on the served query path.

mod external_product;
mod types;

pub use external_product::{
    external_product, external_product_with_ntt_rgsw, gadget_decompose, gadget_reconstruct,
    rgsw_rows_to_ntt, ExternalProductError, RgswRowsNtt,
};
pub(crate) use types::require_gadget_rows;
pub use types::{GadgetVector, RgswCiphertext, SeededRgswCiphertext};
