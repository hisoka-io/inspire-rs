//! One-sided RGSW encryption: `ell` RLWE rows over the gadget powers.
//!
//! The live external product multiplies a trivial RLWE input by an encrypted value.

mod external_product;
mod types;

pub(crate) use external_product::external_product_trivial_with_ntt_rgsw;
pub use external_product::{
    external_product, external_product_with_ntt_rgsw, gadget_decompose, gadget_reconstruct,
    rgsw_rows_to_ntt, ExternalProductError, RgswRowsNtt,
};
pub use types::{GadgetVector, RgswCiphertext, SeededRgswCiphertext};
