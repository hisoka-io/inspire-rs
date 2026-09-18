//! PIR Setup: Server preprocessing and CRS generation.
//!
//! Implements PIR.Setup(1^λ, D) → (crs, D', sk).
//!
//! The setup phase generates:
//!
//! - `ServerCrs`: Public parameters (galois keys, gadget, InspiRING packing seeds)
//! - `EncodedDatabase`: Database encoded as polynomials
//! - `RlweSecretKey`: Client secret key for encryption/decryption
//!
//! # Example
//!
//! ```ignore
//! use raven_inspire::pir::setup;
//! use raven_inspire::params::InspireParams;
//! use raven_inspire::math::GaussianSampler;
//!
//! let params = InspireParams::secure_128_d2048();
//! let database = vec![0u8; 1024 * 32]; // 1024 entries of 32 bytes
//! let mut sampler = GaussianSampler::from_os_entropy(params.sigma)?;
//!
//! let (crs, encoded_db, sk) = setup(&params, &database, 32, &mut sampler)?;
//! ```

use super::error::{pir_err, Result};
use serde::{Deserialize, Deserializer, Serialize};

use crate::inspiring::{PackParams, PackingKeyBody};
use crate::ks::{generate_automorphism_ks_matrix, KeySwitchingMatrix};
use crate::math::gaussian::os_seeded_chacha;
use crate::math::{GaussianSampler, Poly};
use crate::params::{InspireParams, ShardConfig};
use crate::rgsw::GadgetVector;
use crate::rlwe::RlweSecretKey;

use super::encode_db::encode_database;

fn deserialize_coefficient_galois_keys<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<KeySwitchingMatrix>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;

    let keys = Vec::<KeySwitchingMatrix>::deserialize(deserializer)?;
    for (key_index, key) in keys.iter().enumerate() {
        for (row_index, row) in key.rows.iter().enumerate() {
            for (component, poly) in [("a", &row.a), ("b", &row.b)] {
                if poly.is_ntt() {
                    return Err(D::Error::custom(format!(
                        "coefficient-domain CRS galois key[{key_index}] row[{row_index}] \
                         {component} required; wire key declares NTT domain"
                    )));
                }
            }
        }
    }
    Ok(keys)
}

/// Server Common Reference String containing public parameters.
///
/// Contains the public parameters needed for PIR query processing: galois keys,
/// gadget parameters, and the InspiRING packing seeds.
///
/// # Fields
///
/// * `params` - System parameters (ring dimension, modulus, etc.)
/// * `galois_keys` - Galois keys for the automorphism-packing (OnePacking) variant
/// * `rgsw_gadget` - RGSW gadget vector parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerCrs {
    /// System parameters
    pub params: InspireParams,
    /// Galois keys for the automorphism-packing (OnePacking) variant.
    #[serde(deserialize_with = "deserialize_coefficient_galois_keys")]
    pub galois_keys: Vec<KeySwitchingMatrix>,
    /// RGSW gadget vector parameters
    pub rgsw_gadget: GadgetVector,
    /// InspiRING packing parameters (canonical API)
    #[serde(skip)]
    pub inspiring_pack_params: Option<PackParams>,
    /// InspiRING packing key body (w_all rotations - server side)
    #[serde(skip)]
    pub inspiring_packing_key: Option<PackingKeyBody>,
    /// InspiRING w_seed: shared seed for generating w_mask
    /// Client uses this to generate y_body = τ_g(s)·G - s·w_mask + error
    pub inspiring_w_seed: [u8; 32],
    /// InspiRING v_seed: shared seed for conjugation mask (full packing only)
    pub inspiring_v_seed: [u8; 32],
    /// Number of columns per entry (γ for InspiRING packing)
    pub inspiring_num_columns: usize,
}

impl ServerCrs {
    /// Get the ring dimension
    pub fn ring_dim(&self) -> usize {
        self.params.ring_dim
    }

    /// Get the modulus
    pub fn modulus(&self) -> u64 {
        self.params.q
    }

    /// 16-byte version magic prefixing a serialized CRS, mirroring the storage
    /// snapshot magic. A CRS is a persistent cryptographic artifact that can be
    /// decoded without HTTP metadata, so its prefix identifies both the artifact
    /// kind and the layout that requires re-bootstrap when changed.
    pub const MAGIC: [u8; 16] = *b"RAVEN_CRS_v03\0\0\0";

    /// Reject a CRS body larger than this before decoding, so a malicious or stale
    /// (e.g. the pre-shrink ~35 MiB) blob cannot drive an unbounded bincode allocation.
    /// A length pre-check is the only effective bound: bincode 1.3 forces an Infinite
    /// limit on the slice path, so `with_limit` is a no-op there. ~6x the largest
    /// legitimate CRS (d=4096 is ~2.5 MiB). This type-specific cap is deliberately
    /// narrower than a client's general untrusted-input cap; it is a nested
    /// artifact boundary rather than a general transport envelope.
    pub const DECODE_LIMIT_BYTES: usize = 16 * 1024 * 1024;

    /// Serialize with the [`MAGIC`](Self::MAGIC) version prefix for wire / on-disk transport.
    pub fn to_versioned_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Self::MAGIC.to_vec();
        bincode::serialize_into(&mut out, self)
            .map_err(|e| pir_err!("ServerCrs serialize failed: {e}"))?;
        Ok(out)
    }

    /// Validate the [`MAGIC`](Self::MAGIC) prefix without decoding the body.
    pub fn check_magic(bytes: &[u8]) -> Result<()> {
        if bytes.get(..Self::MAGIC.len()) != Some(Self::MAGIC.as_slice()) {
            return Err(pir_err!(
                "ServerCrs version magic mismatch: expected a RAVEN_CRS_v03-prefixed blob; \
                 a CRS layout change requires the operator to re-bootstrap"
            ));
        }
        Ok(())
    }

    /// Reject a CRS whose params or packing width cannot drive the algebra.
    ///
    /// Width 0 offers tree packing only and is accepted. Params are checked
    /// here because they arrive over the wire and reach `ntt_context`, which
    /// asserts rather than returns.
    pub fn validate(&self) -> Result<()> {
        self.params
            .validate()
            .map_err(|e| pir_err!("ServerCrs carries invalid InspireParams: {e}"))?;
        if self.inspiring_num_columns == 0 {
            return Ok(());
        }
        PackParams::is_legal_width(self.params.ring_dim, self.inspiring_num_columns)
            .then_some(())
            .ok_or_else(|| {
                pir_err!(
                    "ServerCrs declares inspiring_num_columns {}, which is not a legal \
                     InspiRING packing width at ring_dim {}; legal widths are {:?}",
                    self.inspiring_num_columns,
                    self.params.ring_dim,
                    PackParams::legal_widths(self.params.ring_dim)
                )
            })
    }

    /// Deserialize a [`to_versioned_bytes`](Self::to_versioned_bytes) blob,
    /// validating the version prefix first so a layout mismatch is a typed error.
    ///
    /// This is the single trust boundary where server-supplied bytes become a
    /// `ServerCrs`, so [`validate`](Self::validate) runs here rather than at
    /// each use site: every consumer downstream holds the invariant by
    /// construction.
    pub fn from_versioned_bytes(bytes: &[u8]) -> Result<Self> {
        Self::check_magic(bytes)?;
        let body = bytes.get(Self::MAGIC.len()..).unwrap_or_default();
        if body.len() > Self::DECODE_LIMIT_BYTES {
            return Err(pir_err!(
                "ServerCrs blob too large: {} bytes exceeds the {} byte decode cap",
                body.len(),
                Self::DECODE_LIMIT_BYTES
            ));
        }
        let crs: Self = bincode::deserialize(body)
            .map_err(|e| pir_err!("ServerCrs decode failed after version check: {e}"))?;
        crs.validate()?;
        Ok(crs)
    }
}

/// Type alias for backward compatibility - InspireCrs is now ServerCrs
pub type InspireCrs = ServerCrs;

/// Encoded database ready for PIR queries
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodedDatabase {
    /// Database shards
    pub shards: Vec<ShardData>,
    /// Shard configuration
    pub config: ShardConfig,
}

/// Single database shard with encoded polynomials
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardData {
    /// Shard identifier
    pub id: u32,
    /// Database entries encoded as polynomial coefficients
    pub polynomials: Vec<Poly>,
}

/// PIR.Setup(1^λ, D) → (crs, D', sk)
///
/// Generates the Common Reference String and encodes the database.
///
/// # Arguments
/// * `params` - System parameters
/// * `database` - Raw database bytes (entries concatenated)
/// * `entry_size` - Size of each database entry in bytes
/// * `sampler` - Gaussian sampler for key generation
///
/// # Returns
/// * `ServerCrs` - Common reference string with public parameters
/// * `EncodedDatabase` - Database encoded as polynomials ready for PIR
/// * `RlweSecretKey` - Secret key for client queries (kept separate from public CRS)
pub fn setup(
    params: &InspireParams,
    database: &[u8],
    entry_size: usize,
    sampler: &mut GaussianSampler,
) -> Result<(ServerCrs, EncodedDatabase, RlweSecretKey)> {
    let mut rng = os_seeded_chacha("setup").map_err(|e| pir_err!("{}", e))?;
    setup_with_rng(params, database, entry_size, sampler, &mut rng)
}

/// `setup` with explicit caller-provided RNG.
///
/// Used by tests that need reproducible CRS bytes (seeded `ChaCha20Rng`)
/// and by WASM clients that pre-seed a deterministic RNG. Native
/// callers can keep using [`setup`], which delegates here with an
/// OS-seeded `ChaCha20Rng`.
pub fn setup_with_rng<R: rand::RngCore + rand::CryptoRng>(
    params: &InspireParams,
    database: &[u8],
    entry_size: usize,
    sampler: &mut GaussianSampler,
    rng: &mut R,
) -> Result<(ServerCrs, EncodedDatabase, RlweSecretKey)> {
    params.validate().map_err(|e| pir_err!("{}", e))?;
    if entry_size == 0 {
        return Err(pir_err!(
            "entry_size must be non-zero; it divides the database into records"
        ));
    }

    let d = params.ring_dim;
    let q = params.q;
    let ctx = params.ntt_context();

    let total_entries = database.len() / entry_size;
    let shard_config = ShardConfig::for_ring_dim(d, entry_size, total_entries as u64)
        .map_err(|cause| pir_err!("setup: {cause}"))?;

    let rlwe_sk = RlweSecretKey::generate(params, sampler);

    let packing_gadget = GadgetVector::new(params.gadget_base, params.packing_gadget_len, q);
    let query_gadget = GadgetVector::new(params.gadget_base, params.query_gadget_len, q);

    // Generate galois keys for tree packing automorphisms
    // For tree packing we need τ_t where t = d/2^i + 1 for i = 0..log_d
    // These allow combining ciphertexts in the packing tree
    let log_d = (d as f64).log2() as usize;
    let mut galois_keys = Vec::with_capacity(log_d);
    for i in 0..log_d {
        let t = (d >> i) + 1; // t = d/2^i + 1
        let ks_matrix =
            generate_automorphism_ks_matrix(&rlwe_sk, t, &packing_gadget, sampler, &ctx);
        galois_keys.push(ks_matrix);
    }

    // a column carries 16 bits of the entry, so the packing width is ceil(entry_size / 2)
    let bytes_per_column = 2usize;
    let num_columns = crate::num_columns(entry_size);

    if !PackParams::is_legal_width(d, num_columns) {
        let legal_entry_sizes: Vec<usize> = PackParams::legal_widths(d)
            .into_iter()
            .flat_map(|w| [2 * w - 1, 2 * w])
            .collect();
        return Err(pir_err!(
            "entry_size {entry_size} bytes yields InspiRING width \
             ceil({entry_size}/{bytes_per_column}) = {num_columns}, which is not a power of two \
             in [1, {}] at ring_dim {d}; legal entry sizes are {legal_entry_sizes:?}",
            d / 2
        ));
    }

    let inspiring_pack_params =
        PackParams::try_new(params, num_columns).map_err(|e| pir_err!("{e}"))?;

    // Generate random seeds for InspiRING (shared between client and server)
    let mut inspiring_w_seed = [0u8; 32];
    let mut inspiring_v_seed = [0u8; 32];
    rng.fill_bytes(&mut inspiring_w_seed);
    rng.fill_bytes(&mut inspiring_v_seed);

    // Generate offline packing keys from w_seed (server-side)
    let inspiring_packing_key = PackingKeyBody::generate(&inspiring_pack_params, inspiring_w_seed);

    let shard_data = encode_database(database, entry_size, params, &shard_config)?;

    let crs = ServerCrs {
        params: params.clone(),
        galois_keys,
        rgsw_gadget: query_gadget,
        inspiring_pack_params: Some(inspiring_pack_params),
        inspiring_packing_key: Some(inspiring_packing_key),
        inspiring_w_seed,
        inspiring_v_seed,
        inspiring_num_columns: num_columns,
    };

    let encoded_db = EncodedDatabase {
        shards: shard_data,
        config: shard_config,
    };

    Ok((crs, encoded_db, rlwe_sk))
}

/// Alias for setup() - kept for backward compatibility
///
/// Both `setup` and `setup_with_secret_key` now return the secret key separately.
#[deprecated(
    since = "0.2.0",
    note = "Use setup() directly - both now return the secret key"
)]
#[allow(dead_code)]
pub fn setup_with_secret_key(
    params: &InspireParams,
    database: &[u8],
    entry_size: usize,
    sampler: &mut GaussianSampler,
) -> Result<(ServerCrs, EncodedDatabase, RlweSecretKey)> {
    setup(params, database, entry_size, sampler)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> InspireParams {
        InspireParams {
            ring_dim: 256,
            q: 1152921504606830593,
            crt_moduli: vec![1152921504606830593],
            p: 65536,
            sigma: 6.4,
            gadget_base: 1 << 20,
            query_gadget_len: 3,
            packing_gadget_len: 3,
            security_level: crate::params::SecurityLevel::Bits128,
        }
    }

    #[test]
    fn test_setup_produces_valid_crs() {
        let params = test_params();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let entry_size = 32;
        let num_entries = params.ring_dim;
        let database: Vec<u8> = (0..(num_entries * entry_size))
            .map(|i| (i % 256) as u8)
            .collect();

        let result = setup(&params, &database, entry_size, &mut sampler);
        assert!(result.is_ok());

        let (crs, encoded_db, _sk) = result.unwrap();

        assert_eq!(crs.ring_dim(), params.ring_dim);
        assert_eq!(crs.modulus(), params.q);

        assert!(!encoded_db.shards.is_empty());
    }

    #[test]
    fn test_setup_with_secret_key() {
        let params = test_params();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let entry_size = 32;
        let num_entries = params.ring_dim;
        let database: Vec<u8> = (0..(num_entries * entry_size))
            .map(|i| (i % 256) as u8)
            .collect();

        let result = setup(&params, &database, entry_size, &mut sampler);
        assert!(result.is_ok());

        let (crs, encoded_db, sk) = result.unwrap();

        assert_eq!(sk.ring_dim(), params.ring_dim);
        assert!(!encoded_db.shards.is_empty());
        assert_eq!(crs.params.ring_dim, params.ring_dim);
    }

    /// H1: `setup_with_rng` must accept a caller-provided `CryptoRng`
    /// and produce a CRS whose RNG-derived bytes are reproducible across
    /// runs given the same seed. Locks down the explicit-RNG injection
    /// path used by WASM clients and deterministic test fixtures.
    #[test]
    fn setup_with_rng_is_deterministic_under_seeded_rng() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let params = test_params();
        let entry_size = 32;
        let num_entries = params.ring_dim;
        let database: Vec<u8> = (0..(num_entries * entry_size))
            .map(|i| (i % 256) as u8)
            .collect();

        let run_once = |seed: u64| -> ([u8; 32], [u8; 32]) {
            let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let (crs, _, _) =
                setup_with_rng(&params, &database, entry_size, &mut sampler, &mut rng).unwrap();
            (crs.inspiring_w_seed, crs.inspiring_v_seed)
        };

        let (w_a, v_a) = run_once(0xDEAD_BEEF_CAFE_F00D);
        let (w_b, v_b) = run_once(0xDEAD_BEEF_CAFE_F00D);
        assert_eq!(w_a, w_b, "same RNG seed must produce same w_seed");
        assert_eq!(v_a, v_b, "same RNG seed must produce same v_seed");

        let (w_diff, _) = run_once(0x1234_5678_9ABC_DEF0);
        assert_ne!(
            w_a, w_diff,
            "different RNG seed must produce different w_seed (sanity)"
        );
    }

    #[test]
    fn test_setup_empty_database() {
        let params = test_params();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let entry_size = 32;
        let database: Vec<u8> = vec![];

        let result = setup(&params, &database, entry_size, &mut sampler);
        assert!(result.is_ok());

        let (_, encoded_db, _sk) = result.unwrap();
        assert!(encoded_db.shards.is_empty() || encoded_db.shards[0].polynomials.is_empty());
    }
}
