//! InsPIRe PIR: setup -> query -> respond -> extract, over encrypted monomial rotation.

mod encode_db;
mod error;
mod eval_poly;
mod extract;
#[cfg(feature = "mod-switch-response")]
pub mod mod_switch;
mod query;
mod respond;
mod session;
mod setup;

pub use error::Result;

pub use encode_db::{
    encode_column, encode_database, encode_direct, inverse_monomial, reconstruct_entry,
};
pub use extract::{extract, extract_inspiring, extract_two_packing, extract_with_variant};
pub use query::{
    query, query_seeded, ClientQuery, ClientState, PackingMode, SeededClientQuery,
    ServerSessionHandle,
};
pub use respond::{
    respond, respond_inspiring, respond_inspiring_cached, respond_inspiring_cached_with_session,
    respond_one_packing, respond_seeded, respond_seeded_inspiring, respond_seeded_inspiring_cached,
    respond_seeded_inspiring_cached_with_session, respond_seeded_packed,
    respond_seeded_with_variant, respond_sequential, respond_with_variant, ServerInspiringCache,
    ServerResponse,
};
pub use session::{ClientSession, ServerSessionStore, SessionResidue};
pub use setup::{setup, setup_with_rng, EncodedDatabase, InspireCrs, ServerCrs, ShardData};
