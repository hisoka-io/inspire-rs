//! PIR.Query encrypts X^(-local_index); the server's multiply by h(X) then lands the
//! target value in coefficient 0.

use serde::{Deserialize, Serialize};

use crate::inspiring::{ClientPackingKeys, PackParams};
use crate::lwe::LweSecretKey;
use crate::math::GaussianSampler;
use crate::params::ShardConfig;
use crate::rgsw::{GadgetVector, RgswCiphertext, SeededRgswCiphertext};
use crate::rlwe::RlweSecretKey;

use super::encode_db::inverse_monomial;
use super::error::{pir_err, Result};
use super::setup::ServerCrs;

/// Packing algorithm selection for server responses.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PackingMode {
    /// InspiRING packing, which requires client packing keys.
    #[default]
    Inspiring,
    /// Tree packing: slower, log(d) matrices.
    Tree,
}

fn seeded_query_with_gadget(
    crs: &ServerCrs,
    global_index: u64,
    shard_config: &ShardConfig,
    rlwe_sk: &RlweSecretKey,
    sampler: &mut GaussianSampler,
) -> Result<(ClientState, SeededClientQuery)> {
    ensure_inspiring_query_width(crs)?;
    ensure_query_shard_geometry(crs, shard_config)?;
    let d = crs.ring_dim();
    let q = crs.modulus();
    let ctx = crs.params.ntt_context();

    let (shard_id, local_index) = shard_config.index_to_shard(global_index);

    let lwe_sk = LweSecretKey::from_rlwe(rlwe_sk);

    let inv_mono = inverse_monomial(local_index as usize, d, q, crs.params.moduli());
    let scaled_monomial = inv_mono.scalar_mul(crs.params.delta());
    let one_row = GadgetVector::new(crs.params.gadget_base, 1, q);
    let rgsw_ciphertext =
        SeededRgswCiphertext::encrypt(rlwe_sk, &scaled_monomial, &one_row, sampler, &ctx);

    let state = ClientState {
        secret_key: lwe_sk,
        rlwe_secret_key: rlwe_sk.clone(),
        index: global_index,
        shard_id,
        local_index,
    };

    let inspiring_packing_keys = Some(generate_packing_keys(crs, rlwe_sk, sampler)?);

    let query = SeededClientQuery {
        shard_id,
        rgsw_ciphertext,
        packing_mode: PackingMode::Inspiring,
        inspiring_packing_keys,
        session_handle: None,
    };

    Ok((state, query))
}

/// Secrets and query metadata needed to decrypt a response.
///
/// Both key fields are `#[serde(skip)]` so a serialized state can never carry key
/// material; a round-tripped state decodes them as zero and cannot decrypt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClientState {
    /// LWE secret key derived from the RLWE key.
    #[serde(skip, default)]
    pub secret_key: LweSecretKey,
    /// RLWE secret key for decrypting the packed response.
    #[serde(skip, default)]
    pub rlwe_secret_key: RlweSecretKey,
    /// Global index queried.
    pub index: u64,
    /// Shard holding the entry.
    pub shard_id: u32,
    /// Index within the shard.
    pub local_index: u64,
}

/// Reference to packing keys already uploaded via
/// [`crate::pir::ServerSessionStore::register`], so queries need not inline ~48 KiB
/// of keys. Opaque to clients, carries no secret, and MAY be logged. Servers that
/// persist state must prevent reuse across restart.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ServerSessionHandle(pub u64);

/// Query sent to the server.
///
/// Privacy caveat: `shard_id` travels in cleartext, so the anonymity set is one
/// shard rather than the whole database. See PRIVACY.md.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClientQuery {
    /// Target shard, unencrypted.
    pub shard_id: u32,
    /// One-row encrypted scaled inverse monomial.
    pub rgsw_ciphertext: RgswCiphertext,
    /// Packing algorithm the server should use.
    #[serde(default)]
    pub packing_mode: PackingMode,
    /// Inline InspiRING packing keys; MUST be `None` when `session_handle` is set.
    #[serde(default)]
    pub inspiring_packing_keys: Option<ClientPackingKeys>,
    /// Reference to pre-uploaded packing keys; `None` falls back to inlining them.
    #[serde(default)]
    pub session_handle: Option<ServerSessionHandle>,
}

/// `ClientQuery` carrying seeds in place of the `a` polynomials, roughly halving
/// query bytes; the server expands before processing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeededClientQuery {
    /// Target shard, unencrypted.
    pub shard_id: u32,
    /// Seeded one-row encrypted scaled inverse monomial.
    pub rgsw_ciphertext: SeededRgswCiphertext,
    /// Packing algorithm the server should use.
    #[serde(default)]
    pub packing_mode: PackingMode,
    /// Inline InspiRING packing keys; MUST be `None` when `session_handle` is set.
    #[serde(default)]
    pub inspiring_packing_keys: Option<ClientPackingKeys>,
    /// Reference to pre-uploaded packing keys; see [`ClientQuery::session_handle`].
    #[serde(default)]
    pub session_handle: Option<ServerSessionHandle>,
}

impl SeededClientQuery {
    /// Regenerate the `a` polynomials from their seeds.
    pub fn expand(&self) -> ClientQuery {
        ClientQuery {
            shard_id: self.shard_id,
            rgsw_ciphertext: self.rgsw_ciphertext.expand(),
            packing_mode: self.packing_mode,
            inspiring_packing_keys: self.inspiring_packing_keys.clone(),
            session_handle: self.session_handle,
        }
    }
}

fn generate_packing_keys(
    crs: &ServerCrs,
    rlwe_sk: &RlweSecretKey,
    sampler: &mut GaussianSampler,
) -> Result<ClientPackingKeys> {
    ensure_inspiring_query_width(crs)?;
    let pack_params = PackParams::try_new(&crs.params, crs.inspiring_num_columns)
        .map_err(|e| pir_err!("CRS carries an unusable InspiRING width: {e}"))?;
    Ok(ClientPackingKeys::generate(
        rlwe_sk,
        &pack_params,
        crs.inspiring_w_seed,
        sampler,
    ))
}

fn ensure_inspiring_query_width(crs: &ServerCrs) -> Result<()> {
    if crs.inspiring_num_columns == 0 {
        return Err(pir_err!(
            "PIR query refused a zero InspiRING width: falling back to a tree-packed \
             response would return wrong bytes under the TwoPacking extractor"
        ));
    }
    Ok(())
}

pub(super) fn ensure_query_shard_geometry(
    crs: &ServerCrs,
    shard_config: &ShardConfig,
) -> Result<()> {
    shard_config
        .validate_for_params(&crs.params)
        .map_err(|cause| {
            pir_err!(
                "PIR query shard geometry does not match ring_dim {}: {cause}",
                crs.ring_dim()
            )
        })
}

/// Build a query for `global_index` plus the client state needed to extract it.
pub fn query(
    crs: &ServerCrs,
    global_index: u64,
    shard_config: &ShardConfig,
    rlwe_sk: &RlweSecretKey,
    sampler: &mut GaussianSampler,
) -> Result<(ClientState, ClientQuery)> {
    ensure_inspiring_query_width(crs)?;
    ensure_query_shard_geometry(crs, shard_config)?;
    let d = crs.ring_dim();
    let q = crs.modulus();
    let ctx = crs.params.ntt_context();

    let (shard_id, local_index) = shard_config.index_to_shard(global_index);

    let lwe_sk = LweSecretKey::from_rlwe(rlwe_sk);

    let inv_mono = inverse_monomial(local_index as usize, d, q, crs.params.moduli());
    let scaled_monomial = inv_mono.scalar_mul(crs.params.delta());
    let one_row = GadgetVector::new(crs.params.gadget_base, 1, q);
    let rgsw_ciphertext =
        RgswCiphertext::encrypt(rlwe_sk, &scaled_monomial, &one_row, sampler, &ctx);

    let state = ClientState {
        secret_key: lwe_sk,
        rlwe_secret_key: rlwe_sk.clone(),
        index: global_index,
        shard_id,
        local_index,
    };

    let inspiring_packing_keys = Some(generate_packing_keys(crs, rlwe_sk, sampler)?);

    let query = ClientQuery {
        shard_id,
        rgsw_ciphertext,
        packing_mode: PackingMode::Inspiring,
        inspiring_packing_keys,
        session_handle: None,
    };

    Ok((state, query))
}

/// `query` in the compact seeded form; the server must `expand()` before processing.
pub fn query_seeded(
    crs: &ServerCrs,
    global_index: u64,
    shard_config: &ShardConfig,
    rlwe_sk: &RlweSecretKey,
    sampler: &mut GaussianSampler,
) -> Result<(ClientState, SeededClientQuery)> {
    seeded_query_with_gadget(crs, global_index, shard_config, rlwe_sk, sampler)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pir::setup::setup;

    fn test_params() -> crate::params::InspireParams {
        crate::params::InspireParams {
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
    fn test_query_generates_valid_output() {
        let params = test_params();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let entry_size = 32;
        let num_entries = params.ring_dim;
        let database: Vec<u8> = (0..(num_entries * entry_size))
            .map(|i| (i % 256) as u8)
            .collect();

        let (crs, encoded_db, rlwe_sk) =
            setup(&params, &database, entry_size, &mut sampler).unwrap();

        let target_index = 42u64;
        let (state, client_query) = query(
            &crs,
            target_index,
            &encoded_db.config,
            &rlwe_sk,
            &mut sampler,
        )
        .unwrap();

        assert_eq!(state.index, target_index);
        assert_eq!(state.shard_id, client_query.shard_id);
    }

    #[test]
    fn test_query_shard_assignment() {
        let params = test_params();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let entry_size = 32;
        let num_entries = params.ring_dim * 2;
        let database: Vec<u8> = (0..(num_entries * entry_size))
            .map(|i| (i % 256) as u8)
            .collect();

        let (crs, encoded_db, rlwe_sk) =
            setup(&params, &database, entry_size, &mut sampler).unwrap();

        let target_index = params.ring_dim as u64 + 10;
        let (state, client_query) = query(
            &crs,
            target_index,
            &encoded_db.config,
            &rlwe_sk,
            &mut sampler,
        )
        .unwrap();

        assert_eq!(state.shard_id, 1);
        assert_eq!(state.local_index, 10);
        assert_eq!(client_query.shard_id, 1);
    }

    #[test]
    fn test_query_size_comparison() {
        let params = crate::params::InspireParams {
            ring_dim: 2048,
            q: 1152921504606830593,
            crt_moduli: vec![1152921504606830593],
            p: 65536,
            sigma: 6.4,
            gadget_base: 1 << 20,
            query_gadget_len: 3,
            packing_gadget_len: 3,
            security_level: crate::params::SecurityLevel::Bits128,
        };
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let entry_size = 32;
        let num_entries = params.ring_dim;
        let database: Vec<u8> = (0..(num_entries * entry_size))
            .map(|i| (i % 256) as u8)
            .collect();

        let (crs, encoded_db, rlwe_sk) =
            setup(&params, &database, entry_size, &mut sampler).unwrap();

        let target_index = 42u64;

        let (_, full_query) = query(
            &crs,
            target_index,
            &encoded_db.config,
            &rlwe_sk,
            &mut sampler,
        )
        .unwrap();
        let (_, seeded_query) = query_seeded(
            &crs,
            target_index,
            &encoded_db.config,
            &rlwe_sk,
            &mut sampler,
        )
        .unwrap();

        let full_size = bincode::serialize(&full_query).unwrap().len();
        let seeded_size = bincode::serialize(&seeded_query).unwrap().len();

        println!(
            "\n=== Query Size Comparison (d={}, l_full={}) ===",
            params.ring_dim, params.query_gadget_len
        );
        println!(
            "Full query:     {:>8} bytes ({:.1} KB)",
            full_size,
            full_size as f64 / 1024.0
        );
        println!(
            "Seeded query:   {:>8} bytes ({:.1} KB)",
            seeded_size,
            seeded_size as f64 / 1024.0
        );
        println!("\nReductions:");
        println!(
            "  Seeded vs Full:   {:.1}%",
            100.0 * (1.0 - seeded_size as f64 / full_size as f64)
        );

        assert!(
            seeded_size < full_size,
            "Seeded should be smaller than full"
        );
    }
}
