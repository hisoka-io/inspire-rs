//! InspiRING packs d LWE ciphertexts into one RLWE ciphertext with 2 key-switching
//! matrices instead of CDKS's log(d), in three stages: transform, aggregate, collapse.
//! Under the CRS model the `a` vectors are fixed, so only `b` changes per query and the
//! rest precomputes offline.

pub mod automorph_pack;
mod collapse;
mod collapse_one;
pub mod inspiring2;
mod pack;
mod simple_pack;
mod transform;
mod types;

pub use automorph_pack::{
    homomorphic_automorph, pack_lwes, pack_lwes_inner, pack_rlwes_tree, pack_single_lwe,
    prep_pack_lwes, YConstants,
};
pub use collapse::{collapse, collapse_half, collapse_partial};
pub use collapse_one::collapse_one;
pub use inspiring2::{
    full_packing_offline, generate_rotations, pack_inspiring, pack_inspiring_full,
    pack_inspiring_legacy, pack_inspiring_partial, packing_offline, packing_online,
    packing_online_gather, precompute_inspiring, ClientPackingKeys, GeneratorPowers,
    InspiringPrecomputation, OfflinePackingKeys, PackParams, PackParamsError, PackingKeyBody,
    PrecompInsPIR, RotatedKsMatrix,
};
pub use pack::{pack, pack_online, partial_pack, precompute_packing, PackingPrecomputation};
pub use simple_pack::{pack_lwe_to_rlwe, pack_rlwe_coeffs};
pub use transform::{aggregate, transform, transform_at_slot, transform_partial};
pub use types::{AggregatedCiphertext, IntermediateCiphertext};
