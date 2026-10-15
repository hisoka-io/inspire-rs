//! Fixed-seed cells shared by the packing-key KATs: one table, one client session and
//! a reproducible query stream per (parameters, record size, seed).
//!
//! The recorded digests under `tests/fixtures/` came from this same sequence of
//! sampler and RNG draws, so the order of calls in [`Cell::build`] and
//! [`Cell::next_query`] is part of the fixture.

#![allow(
    dead_code,
    clippy::expect_used,
    reason = "each test binary includes the whole module and uses the parts it needs; an \
              abort in a fixture helper is the failure report"
)]

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use raven_inspire::inspiring::ClientPackingKeys;
use raven_inspire::lwe::LweSecretKey;
use raven_inspire::math::{GaussianSampler, DEFAULT_Q};
use raven_inspire::params::{
    InspireParams, SecurityLevel, DEFAULT_CRT_MODULI, DEFAULT_Q_2CRT_30BIT,
};
use raven_inspire::pir::{
    extract_inspiring, inverse_monomial, setup_with_rng, ClientSession, ClientState,
    EncodedDatabase, PackingMode, SeededBatchQuery, SeededClientQuery, ServerCrs,
    ServerInspiringCache, ServerResponse,
};
use raven_inspire::rgsw::{GadgetVector, SeededRgswCiphertext};
use raven_inspire::rlwe::RlweSecretKey;

pub const ADAPTIVE_CRT_MODULI: [u64; 2] = [67_043_329, 132_120_577];

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn params_for(ring_dim: usize, moduli: &[u64], gadget_len: usize) -> InspireParams {
    InspireParams {
        ring_dim,
        q: moduli.iter().product(),
        crt_moduli: moduli.to_vec(),
        p: 65537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: gadget_len,
        packing_gadget_len: gadget_len,
        security_level: SecurityLevel::Bits128,
    }
}

/// Every modulus set the crate ships: the 60-bit prime and the three two-prime sets.
pub fn modulus_sets() -> Vec<(&'static str, Vec<u64>)> {
    vec![
        ("single", vec![DEFAULT_Q]),
        ("upstream2", DEFAULT_CRT_MODULI.to_vec()),
        ("thirty2", DEFAULT_Q_2CRT_30BIT.to_vec()),
        ("adaptive2", ADAPTIVE_CRT_MODULI.to_vec()),
    ]
}

pub fn database(entries: usize, entry_size: usize, seed: u64) -> Vec<u8> {
    (0..entries * entry_size)
        .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9).wrapping_add(seed) >> 7) as u8)
        .collect()
}

pub fn response_bytes(response: &ServerResponse) -> Vec<u8> {
    bincode::serialize(response).expect("serialize response")
}

/// A three-shard table with its server cache and one client session.
pub struct Cell {
    pub params: InspireParams,
    pub entry_size: usize,
    pub db: Vec<u8>,
    pub crs: ServerCrs,
    pub encoded: EncodedDatabase,
    pub cache: ServerInspiringCache,
    pub secret_key: RlweSecretKey,
    pub session: ClientSession,
    pub sampler: GaussianSampler,
    pub rng: ChaCha20Rng,
}

impl Cell {
    pub fn build(params: &InspireParams, entry_size: usize, seed: u64) -> Self {
        let db = database(3 * params.ring_dim, entry_size, seed);
        let mut sampler = GaussianSampler::with_seed(params.sigma, seed);
        let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0x5eed);
        let (crs, encoded, secret_key) =
            setup_with_rng(params, &db, entry_size, &mut sampler, &mut rng).expect("setup");
        let cache = ServerInspiringCache::new(&crs, &encoded).expect("cache");
        let session =
            ClientSession::new(crs.clone(), secret_key.clone(), &mut sampler).expect("session");
        // The recorded stream spent one session query here; its sampler draws are part
        // of every later ciphertext.
        session
            .query_seeded(0, &encoded.config, &mut sampler)
            .expect("probe query");
        Self {
            params: params.clone(),
            entry_size,
            db,
            crs,
            encoded,
            cache,
            secret_key,
            session,
            sampler,
            rng,
        }
    }

    /// The session's uploaded key after a wire round trip, as a server receives it.
    pub fn wire_keys(&self) -> ClientPackingKeys {
        let keys = self.session.packing_keys().expect("InspiRING session");
        bincode::deserialize(&bincode::serialize(keys).expect("serialize keys"))
            .expect("deserialize keys")
    }

    /// The next row request of the reproducible stream, round-tripped through bincode.
    pub fn next_query(&mut self, index: u64) -> SeededBatchQuery {
        let params = &self.params;
        let (shard_id, local) = self.encoded.config.index_to_shard(index);
        let scaled = inverse_monomial(local as usize, params.ring_dim, params.q, params.moduli())
            .scalar_mul(params.delta());
        let one_row = GadgetVector::new(params.gadget_base, 1, params.q);
        let rgsw_ciphertext = SeededRgswCiphertext::encrypt_with_rng(
            &self.secret_key,
            &scaled,
            &one_row,
            &mut self.sampler,
            &params.ntt_context(),
            &mut self.rng,
        );
        let query = SeededBatchQuery {
            shard_id,
            rgsw_ciphertext,
        };
        bincode::deserialize(&bincode::serialize(&query).expect("serialize query"))
            .expect("deserialize query")
    }

    /// Decrypt `response` as the answer to `index` and compare it to the planted record.
    pub fn assert_decodes(&self, index: u64, response: &ServerResponse) {
        let (shard_id, local_index) = self.encoded.config.index_to_shard(index);
        let state = ClientState {
            secret_key: LweSecretKey::from_rlwe(&self.secret_key),
            rlwe_secret_key: self.secret_key.clone(),
            index,
            shard_id,
            local_index,
        };
        let decoded =
            extract_inspiring(&self.crs, &state, response, self.entry_size).expect("extract");
        let start = index as usize * self.entry_size;
        assert_eq!(
            decoded.as_slice(),
            &self.db[start..start + self.entry_size],
            "index {index} decoded to another record"
        );
    }
}

/// The same row request in the per-query wire form, carrying its own inline key.
pub fn inline_query(query: &SeededBatchQuery, keys: &ClientPackingKeys) -> SeededClientQuery {
    SeededClientQuery {
        shard_id: query.shard_id,
        rgsw_ciphertext: query.rgsw_ciphertext.clone(),
        packing_mode: PackingMode::Inspiring,
        inspiring_packing_keys: Some(keys.clone()),
        session_handle: None,
    }
}
