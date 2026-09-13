//! InsPIRe PIR, ported from eprint 2025/1352.

/// Number of 16-bit columns needed to encode an entry, with one column for an empty width.
///
/// ```
/// use raven_inspire::num_columns;
///
/// assert_eq!(num_columns(32), 16);
/// assert_eq!(num_columns(33), 17);
/// ```
#[must_use]
pub const fn num_columns(entry_size: usize) -> usize {
    let columns = entry_size.div_ceil(2);
    if columns == 0 {
        1
    } else {
        columns
    }
}

pub mod cost;
pub mod inspiring;
pub mod ks;
pub mod lwe;
pub mod math;
mod par_prelude;
pub mod params;
pub mod pir;
pub mod rgsw;
pub mod rlwe;

pub use pir::{
    encode_column, encode_database, encode_direct, extract, extract_inspiring, extract_two_packing,
    extract_with_variant, inverse_monomial, query, query_seeded, respond, respond_inspiring,
    respond_inspiring_cached, respond_inspiring_cached_with_session, respond_one_packing,
    respond_seeded, respond_seeded_inspiring, respond_seeded_inspiring_cached,
    respond_seeded_inspiring_cached_with_session, respond_seeded_packed,
    respond_seeded_with_variant, respond_with_variant, setup, setup_with_rng, ClientQuery,
    ClientSession, ClientState, EncodedDatabase, InspireCrs, PackingMode, SeededClientQuery,
    ServerCrs, ServerInspiringCache, ServerResponse, ServerSessionHandle, ServerSessionStore,
    SessionResidue, ShardData,
};

pub use params::{InspireParams, InspireVariant, SecurityLevel};
