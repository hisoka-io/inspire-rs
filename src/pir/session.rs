//! Packing keys depend only on (CRS, rlwe_sk, w_seed), never on the query index, so a
//! client session builds them once and resends the same bytes; the server keeps each
//! uploaded key as a [`PackingKeySeat`].

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::inspiring::{ClientPackingKeys, PackParams};
use crate::lwe::LweSecretKey;
use crate::math::{GaussianSampler, NttContext, Poly};
use crate::params::ShardConfig;
#[cfg(feature = "caller-secret-keys")]
use crate::rgsw::RgswCiphertext;
use crate::rgsw::{GadgetVector, SeededRgswCiphertext};
use crate::rlwe::RlweSecretKey;

use super::encode_db::inverse_monomial;
use super::error::{pir_err, Result};
#[cfg(feature = "caller-secret-keys")]
use super::query::ClientQuery;
use super::query::{
    ensure_query_shard_geometry, ClientState, PackingKeySource, PackingMode, SeededBatchQuery,
    SeededClientQuery, SeededQueryBatch, ServerSessionHandle,
};
use super::setup::ServerCrs;

/// Everything a client needs to emit many queries against one server-session CRS.
///
/// The session builds its packing keys once and every query, batch, upload and retry
/// carries those same bytes. [`register_with`](Self::register_with) or a remote
/// handshake swaps the inline keys for a handle.
#[derive(Clone)]
pub struct ClientSession {
    /// Owned CRS snapshot; rotating the CRS means building a new session.
    crs: ServerCrs,
    rlwe_sk: RlweSecretKey,
    lwe_sk: LweSecretKey,
    /// `None` on a Tree-packing-only server.
    packing_keys: Option<ClientPackingKeys>,
    /// `Some` switches queries to the compact handle-referencing wire form.
    session_handle: Option<ServerSessionHandle>,
}

impl std::fmt::Debug for ClientSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientSession")
            .field("ring_dim", &self.crs.ring_dim())
            .field("has_inspiring_packing", &self.packing_keys.is_some())
            .field("has_handle", &self.session_handle.is_some())
            .finish_non_exhaustive()
    }
}

impl ClientSession {
    /// [`Self::with_fresh_secret`] under a secret the caller holds.
    ///
    /// Test and KAT builds only: a second session on a copy of `rlwe_sk` publishes a
    /// second packing key under the same mask, which leaks the secret.
    #[cfg(feature = "caller-secret-keys")]
    pub fn new(
        crs: ServerCrs,
        rlwe_sk: RlweSecretKey,
        sampler: &mut GaussianSampler,
    ) -> Result<Self> {
        Self::build(crs, rlwe_sk, sampler)
    }

    fn build(
        crs: ServerCrs,
        rlwe_sk: RlweSecretKey,
        sampler: &mut GaussianSampler,
    ) -> Result<Self> {
        crs.validate()?;
        let lwe_sk = LweSecretKey::from_rlwe(&rlwe_sk);

        let packing_keys = if crs.inspiring_num_columns > 0 {
            Some(
                ClientPackingKeys::generate_once(
                    &rlwe_sk,
                    &crs.params,
                    crs.inspiring_num_columns,
                    crs.inspiring_w_seed,
                    sampler,
                )
                .map_err(|e| pir_err!("CRS carries an unusable InspiRING width: {e}"))?,
            )
        } else {
            None
        };

        Ok(Self {
            crs,
            rlwe_sk,
            lwe_sk,
            packing_keys,
            session_handle: None,
        })
    }

    /// Build a session under a secret key drawn here, and its one packing key.
    ///
    /// The secret and the key's error come from a fresh OS-seeded stream, never from a
    /// caller's sampler: a seeded sampler handed in twice would give one secret two keys.
    /// The secret never leaves the session except inside a [`SessionResidue`], which
    /// restores the stored key and never builds another, so one secret backs one key.
    /// A retry resends [`Self::packing_keys`]; a new key means a new session.
    ///
    /// # Errors
    ///
    /// Refuses when the platform entropy source fails, or the CRS is unusable.
    ///
    /// ```
    /// use raven_inspire::math::GaussianSampler;
    /// use raven_inspire::params::InspireParams;
    /// use raven_inspire::{setup, ClientSession};
    ///
    /// let params = InspireParams::secure_128_d2048();
    /// let database = vec![0u8; params.ring_dim * 32];
    /// let mut sampler = GaussianSampler::from_os_entropy(params.sigma)?;
    /// let (crs, _, _) = setup(&params, &database, 32, &mut sampler)?;
    /// let session = ClientSession::with_fresh_secret(crs)?;
    /// assert!(session.has_inspiring_packing());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_fresh_secret(crs: ServerCrs) -> Result<Self> {
        crs.validate()?;
        let mut sampler = GaussianSampler::from_os_entropy(crs.params.sigma)
            .map_err(|e| pir_err!("no session without a fresh secret: {e}"))?;
        let rlwe_sk = RlweSecretKey::generate(&crs.params, &mut sampler);
        Self::build(crs, rlwe_sk, &mut sampler)
    }

    /// [`Self::with_fresh_secret`] with the secret and key error drawn from `sampler`.
    ///
    /// Test and KAT builds only: a seeded sampler replays the secret, so two calls on
    /// one seed publish two keys under one secret and one mask.
    #[cfg(feature = "caller-secret-keys")]
    pub fn with_seeded_secret(crs: ServerCrs, sampler: &mut GaussianSampler) -> Result<Self> {
        crs.validate()?;
        let rlwe_sk = RlweSecretKey::generate(&crs.params, sampler);
        Self::build(crs, rlwe_sk, sampler)
    }

    /// Register this session's packing keys with an in-process store and switch to the
    /// compact wire format. `Ok(None)` means there were no keys to register.
    pub fn register_with(
        &mut self,
        store: &ServerSessionStore,
    ) -> Result<Option<ServerSessionHandle>> {
        let Some(keys) = &self.packing_keys else {
            return Ok(None);
        };
        let params = &self.crs.params;
        let seat = PackingKeySeat::build(
            keys.clone(),
            &PackingGeometry {
                ring_dim: params.ring_dim,
                moduli: params.moduli(),
                key_rows: params.packing_key_rows(),
                num_to_pack: self.crs.inspiring_num_columns,
            },
            &params.ntt_context(),
        )?;
        let handle = store.register(seat)?;
        self.session_handle = Some(handle);
        Ok(Some(handle))
    }

    /// The packing keys to upload when registering with a remote server; the same
    /// bytes on every call. `None` on a Tree-packing-only server.
    pub fn packing_keys(&self) -> Option<&ClientPackingKeys> {
        self.packing_keys.as_ref()
    }

    /// Current handle, or `None` if nothing was registered.
    pub fn session_handle(&self) -> Option<ServerSessionHandle> {
        self.session_handle
    }

    /// Install a handle returned by this session's remote server handshake.
    ///
    /// The bare handle carries no CRS or instance identity. The caller must install only a
    /// handle returned after uploading this session's packing keys to the server holding
    /// [`Self::crs`]; the server remains responsible for rejecting stale or foreign handles.
    ///
    /// # Errors
    ///
    /// Returns an error when this session has no InspiRING packing material to bind.
    ///
    /// ```
    /// use raven_inspire::math::GaussianSampler;
    /// use raven_inspire::params::InspireParams;
    /// use raven_inspire::{setup, ClientSession, ServerSessionHandle};
    ///
    /// let params = InspireParams::secure_128_d2048();
    /// let database = vec![0u8; params.ring_dim * 32];
    /// let mut sampler = GaussianSampler::from_os_entropy(params.sigma)?;
    /// let (crs, _, _) = setup(&params, &database, 32, &mut sampler)?;
    /// let mut session = ClientSession::with_fresh_secret(crs)?;
    /// let issued = ServerSessionHandle(42);
    /// session.install_server_session_handle(issued)?;
    /// assert_eq!(session.session_handle(), Some(issued));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn install_server_session_handle(&mut self, handle: ServerSessionHandle) -> Result<()> {
        if self.packing_keys.is_none() {
            return Err(pir_err!(
                "cannot install server session handle {}: this ClientSession has no InspiRING \
                 packing keys, so subsequent queries could not use the handle path",
                handle.0
            ));
        }
        self.session_handle = Some(handle);
        Ok(())
    }

    /// Borrow the captured CRS.
    pub fn crs(&self) -> &ServerCrs {
        &self.crs
    }

    /// Borrow the captured RLWE secret key, e.g. to extract a response.
    pub fn rlwe_secret_key(&self) -> &RlweSecretKey {
        &self.rlwe_sk
    }

    /// Emit a `ClientQuery`: scale and encrypt the inverse monomial, then attach cached keys.
    ///
    /// Test and KAT builds only: the query error comes from the caller's sampler, and a
    /// seeded one replays it, which hands the server the row.
    #[cfg(feature = "caller-secret-keys")]
    pub fn query(
        &self,
        global_index: u64,
        shard_config: &ShardConfig,
        sampler: &mut GaussianSampler,
    ) -> Result<(ClientState, ClientQuery)> {
        self.inspiring_query_material()?;
        ensure_query_shard_geometry(&self.crs, shard_config)?;
        let d = self.crs.ring_dim();
        let q = self.crs.modulus();
        let ctx = self.crs.params.ntt_context();

        let (shard_id, local_index) = shard_config.index_to_shard(global_index);

        let inv_mono = inverse_monomial(local_index as usize, d, q, self.crs.params.moduli());
        let scaled_monomial = inv_mono.scalar_mul(self.crs.params.delta());
        let one_row = GadgetVector::new(self.crs.params.gadget_base, 1, q);
        let rgsw_ciphertext =
            RgswCiphertext::encrypt(&self.rlwe_sk, &scaled_monomial, &one_row, sampler, &ctx);

        let state = ClientState {
            secret_key: self.lwe_sk.clone(),
            rlwe_secret_key: self.rlwe_sk.clone(),
            index: global_index,
            shard_id,
            local_index,
        };

        let (inspiring_packing_keys, session_handle) = match self.session_handle {
            Some(h) => (None, Some(h)),
            None => (self.packing_keys.clone(), None),
        };
        Ok((
            state,
            ClientQuery {
                shard_id,
                rgsw_ciphertext,
                packing_mode: PackingMode::Inspiring,
                inspiring_packing_keys,
                session_handle,
            },
        ))
    }

    /// [`query_seeded_fresh`](Self::query_seeded_fresh) with the query error drawn from
    /// `sampler`.
    ///
    /// Test and KAT builds only, for the reason [`query`](Self::query) gives.
    #[cfg(feature = "caller-secret-keys")]
    pub fn query_seeded(
        &self,
        global_index: u64,
        shard_config: &ShardConfig,
        sampler: &mut GaussianSampler,
    ) -> Result<(ClientState, SeededClientQuery)> {
        self.query_seeded_with(global_index, shard_config, sampler)
    }

    /// [`query_batch_fresh`](Self::query_batch_fresh) with the query errors drawn from
    /// `sampler`.
    ///
    /// Test and KAT builds only, for the reason [`query`](Self::query) gives.
    #[cfg(feature = "caller-secret-keys")]
    pub fn query_batch(
        &self,
        global_indices: &[u64],
        shard_config: &ShardConfig,
        sampler: &mut GaussianSampler,
    ) -> Result<(Vec<ClientState>, SeededQueryBatch)> {
        self.query_batch_with(global_indices, shard_config, sampler)
    }

    /// One query for `global_index` in the compact seeded form, its error drawn here from
    /// the platform entropy source.
    ///
    /// # Errors
    ///
    /// Refuses when the entropy source fails, or the index or shard geometry does not fit.
    pub fn query_seeded_fresh(
        &self,
        global_index: u64,
        shard_config: &ShardConfig,
    ) -> Result<(ClientState, SeededClientQuery)> {
        let mut sampler = self.fresh_noise()?;
        self.query_seeded_with(global_index, shard_config, &mut sampler)
    }

    /// One request for every index in `global_indices`, all under this session's one
    /// set of packing keys: by handle once registered, inline before that. Building
    /// the batch again resends the same key bytes; every query's error is drawn fresh
    /// here from the platform entropy source.
    ///
    /// ```
    /// use raven_inspire::math::GaussianSampler;
    /// use raven_inspire::params::InspireParams;
    /// use raven_inspire::{setup, ClientSession};
    ///
    /// let params = InspireParams::secure_128_d2048();
    /// let database = vec![0u8; params.ring_dim * 32];
    /// let mut sampler = GaussianSampler::from_os_entropy(params.sigma)?;
    /// let (crs, encoded, _) = setup(&params, &database, 32, &mut sampler)?;
    /// let session = ClientSession::with_fresh_secret(crs)?;
    /// let (states, batch) = session.query_batch_fresh(&[3, 5], &encoded.config)?;
    /// assert_eq!((states.len(), batch.queries().len()), (2, 2));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn query_batch_fresh(
        &self,
        global_indices: &[u64],
        shard_config: &ShardConfig,
    ) -> Result<(Vec<ClientState>, SeededQueryBatch)> {
        let mut sampler = self.fresh_noise()?;
        self.query_batch_with(global_indices, shard_config, &mut sampler)
    }

    fn fresh_noise(&self) -> Result<GaussianSampler> {
        GaussianSampler::from_os_entropy(self.crs.params.sigma)
            .map_err(|e| pir_err!("no query without fresh noise: {e}"))
    }

    fn query_seeded_with(
        &self,
        global_index: u64,
        shard_config: &ShardConfig,
        sampler: &mut GaussianSampler,
    ) -> Result<(ClientState, SeededClientQuery)> {
        let (state, row) = self.seeded_row(global_index, shard_config, sampler)?;
        let (inspiring_packing_keys, session_handle) = match self.session_handle {
            Some(h) => (None, Some(h)),
            None => (self.packing_keys.clone(), None),
        };
        Ok((
            state,
            SeededClientQuery {
                shard_id: row.shard_id,
                rgsw_ciphertext: row.rgsw_ciphertext,
                packing_mode: PackingMode::Inspiring,
                inspiring_packing_keys,
                session_handle,
            },
        ))
    }

    fn query_batch_with(
        &self,
        global_indices: &[u64],
        shard_config: &ShardConfig,
        sampler: &mut GaussianSampler,
    ) -> Result<(Vec<ClientState>, SeededQueryBatch)> {
        let keys = self.inspiring_query_material()?;
        let packing_key = match self.session_handle {
            Some(handle) => PackingKeySource::Registered(handle),
            None => PackingKeySource::Inline(keys.clone()),
        };
        let mut states = Vec::with_capacity(global_indices.len());
        let mut queries = Vec::with_capacity(global_indices.len());
        for &global_index in global_indices {
            let (state, row) = self.seeded_row(global_index, shard_config, sampler)?;
            states.push(state);
            queries.push(row);
        }
        Ok((states, SeededQueryBatch::new(packing_key, queries)))
    }

    fn seeded_row(
        &self,
        global_index: u64,
        shard_config: &ShardConfig,
        sampler: &mut GaussianSampler,
    ) -> Result<(ClientState, SeededBatchQuery)> {
        self.inspiring_query_material()?;
        ensure_query_shard_geometry(&self.crs, shard_config)?;
        let d = self.crs.ring_dim();
        let q = self.crs.modulus();
        let ctx = self.crs.params.ntt_context();

        let (shard_id, local_index) = shard_config.index_to_shard(global_index);

        let inv_mono = inverse_monomial(local_index as usize, d, q, self.crs.params.moduli());
        let scaled_monomial = inv_mono.scalar_mul(self.crs.params.delta());
        let one_row = GadgetVector::new(self.crs.params.gadget_base, 1, q);
        let rgsw_ciphertext =
            SeededRgswCiphertext::encrypt(&self.rlwe_sk, &scaled_monomial, &one_row, sampler, &ctx);

        let state = ClientState {
            secret_key: self.lwe_sk.clone(),
            rlwe_secret_key: self.rlwe_sk.clone(),
            index: global_index,
            shard_id,
            local_index,
        };
        Ok((
            state,
            SeededBatchQuery {
                shard_id,
                rgsw_ciphertext,
            },
        ))
    }

    /// Whether InspiRING packing machinery is cached.
    pub fn has_inspiring_packing(&self) -> bool {
        self.packing_keys.is_some()
    }

    fn inspiring_query_material(&self) -> Result<&ClientPackingKeys> {
        self.packing_keys.as_ref().ok_or_else(|| {
            pir_err!(
                "PIR query refused a zero InspiRING width: no packing material exists, and \
                 falling back to a tree-packed response would return wrong bytes under the \
                 TwoPacking extractor"
            )
        })
    }

    /// Capture the persistable residue: the CRS, the secret key and the packing keys,
    /// with no handle (it is only valid against the issuing store).
    ///
    /// The residue carries the client RLWE secret key in the clear and is not
    /// zeroized; a stolen blob plus observed traffic reveals which indices this
    /// client read.
    pub fn to_residue(&self) -> SessionResidue {
        SessionResidue {
            crs: self.crs.clone(),
            rlwe_sk: self.rlwe_sk.clone(),
            packing_keys: self.packing_keys.clone(),
        }
    }

    /// Rehydrate with the captured packing keys, never regenerating them. Mismatched
    /// CRS and secret-key shapes are rejected here rather than at a later query.
    ///
    /// # Errors
    /// The residue's CRS fails [`ServerCrs::validate`], its secret key is not a
    /// coefficient-domain polynomial on the CRS ring, or its packing keys are off that ring.
    pub fn from_residue(residue: SessionResidue) -> Result<Self> {
        residue
            .crs
            .validate()
            .map_err(|e| pir_err!("inconsistent SessionResidue: {e}"))?;
        let sk = &residue.rlwe_sk.poly;
        if residue.crs.ring_dim() != sk.dimension() || residue.crs.params.moduli() != sk.moduli() {
            return Err(pir_err!(
                "inconsistent SessionResidue: CRS (ring_dim {}, moduli {:?}) != secret key \
                 (ring_dim {}, moduli {:?})",
                residue.crs.ring_dim(),
                residue.crs.params.moduli(),
                sk.dimension(),
                sk.moduli()
            ));
        }
        // `LweSecretKey::from_rlwe` reads coefficients, which asserts on the NTT flag.
        if sk.is_ntt() {
            return Err(pir_err!(
                "inconsistent SessionResidue: the secret key declares the NTT domain; a \
                 captured key is always in the coefficient domain"
            ));
        }
        if let Some(keys) = residue.packing_keys.as_ref() {
            require_wire_packing_body_shape(
                keys,
                residue.crs.ring_dim(),
                residue.crs.params.moduli(),
            )
            .map_err(|e| pir_err!("inconsistent SessionResidue: {e}"))?;
        }
        let lwe_sk = LweSecretKey::from_rlwe(&residue.rlwe_sk);
        Ok(Self {
            crs: residue.crs,
            rlwe_sk: residue.rlwe_sk,
            lwe_sk,
            packing_keys: residue.packing_keys,
            session_handle: None,
        })
    }
}

/// Persistable form of a [`ClientSession`]. Carries the client RLWE secret key, so
/// treat it as secret material; see [`ClientSession::to_residue`].
#[derive(Clone, Serialize, Deserialize)]
pub struct SessionResidue {
    crs: ServerCrs,
    rlwe_sk: RlweSecretKey,
    packing_keys: Option<ClientPackingKeys>,
}

impl std::fmt::Debug for SessionResidue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionResidue")
            .field("ring_dim", &self.crs.ring_dim())
            .field("has_packing_keys", &self.packing_keys.is_some())
            .finish_non_exhaustive()
    }
}

/// A client's packing key as the server holds it: `y_body` in NTT form and nothing
/// derived from it. Packing reads each rotation through the automorphism tables the
/// server shares across clients, so a seat is one polynomial per key row at any gamma.
pub struct PackingKeySeat {
    y_body_ntt: Vec<Poly>,
    num_to_pack: usize,
}

impl std::fmt::Debug for PackingKeySeat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackingKeySeat")
            .field("digits", &self.y_body_ntt.len())
            .field("num_to_pack", &self.num_to_pack)
            .finish_non_exhaustive()
    }
}

/// The packing shape a key must match; `PackParams` and the CRS both describe it.
pub(crate) struct PackingGeometry<'a> {
    pub(crate) ring_dim: usize,
    pub(crate) moduli: &'a [u64],
    pub(crate) key_rows: usize,
    pub(crate) num_to_pack: usize,
}

impl<'a> From<&'a PackParams> for PackingGeometry<'a> {
    fn from(pack_params: &'a PackParams) -> Self {
        Self {
            ring_dim: pack_params.ring_dim,
            moduli: &pack_params.moduli,
            key_rows: pack_params.key_rows(),
            num_to_pack: pack_params.num_to_pack,
        }
    }
}

impl PackingKeySeat {
    /// Check uploaded keys against the server's packing shape and turn their body
    /// into the seat, in NTT form. The only way to build a seat, so every seat has
    /// passed these checks.
    ///
    /// # Errors
    ///
    /// The keys are not coefficient-form polynomials on the server ring, carry a
    /// different row count than the server's packing shape, or declare another gamma.
    ///
    /// ```
    /// use raven_inspire::inspiring::{ClientPackingKeys, PackParams};
    /// use raven_inspire::math::GaussianSampler;
    /// use raven_inspire::params::InspireParams;
    /// use raven_inspire::pir::PackingKeySeat;
    /// use raven_inspire::rlwe::RlweSecretKey;
    ///
    /// let params = InspireParams::secure_128_d2048();
    /// let mut sampler = GaussianSampler::with_seed(params.sigma, 1);
    /// let secret_key = RlweSecretKey::generate(&params, &mut sampler);
    /// let keys = ClientPackingKeys::generate(&secret_key, &params, 16, [7; 32], &mut sampler)?;
    /// let pack_params = PackParams::try_new(&params, 16)?;
    /// let seat = PackingKeySeat::from_wire(keys, &pack_params, &params.ntt_context())?;
    /// assert_eq!(seat.payload_bytes(), 3 * 2048 * 8);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_wire(
        keys: ClientPackingKeys,
        pack_params: &PackParams,
        ctx: &NttContext,
    ) -> Result<Self> {
        Self::build(keys, &PackingGeometry::from(pack_params), ctx)
    }

    pub(crate) fn build(
        keys: ClientPackingKeys,
        geometry: &PackingGeometry<'_>,
        ctx: &NttContext,
    ) -> Result<Self> {
        ensure_packing_width_matches(&keys, geometry)?;
        if ctx.dimension() != geometry.ring_dim || ctx.moduli() != geometry.moduli {
            return Err(pir_err!(
                "packing-key seat refused: NTT context is for ring_dim {} moduli {:?}, the \
                 packing shape is ring_dim {} moduli {:?}",
                ctx.dimension(),
                ctx.moduli(),
                geometry.ring_dim,
                geometry.moduli
            ));
        }
        let mut y_body_ntt = keys.y_body;
        for y in &mut y_body_ntt {
            y.to_ntt(ctx);
        }
        Ok(Self {
            y_body_ntt,
            num_to_pack: keys.num_to_pack,
        })
    }

    /// gamma the keys were built for.
    pub fn num_to_pack(&self) -> usize {
        self.num_to_pack
    }

    /// Bytes of key coefficients held: digits x CRT limbs x ring_dim x 8.
    pub fn payload_bytes(&self) -> usize {
        self.y_body_ntt
            .iter()
            .map(|y| std::mem::size_of_val(y.coeffs()))
            .sum()
    }

    /// [`Self::payload_bytes`] plus the containers around it, counted by length, for
    /// a server's per-session memory accounting.
    pub fn resident_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .y_body_ntt
                .iter()
                .map(|y| {
                    std::mem::size_of::<Poly>()
                        + std::mem::size_of_val(y.coeffs())
                        + std::mem::size_of_val(y.moduli())
                })
                .sum::<usize>()
    }

    pub(crate) fn y_body_ntt(&self) -> &[Poly] {
        &self.y_body_ntt
    }

    /// A seat packs only under the shape it was built for. Another gamma rotates the
    /// key under the wrong automorphism. Another ring or row count is refused by the
    /// online kernel too, but only once the offline phase has been paid for.
    pub(crate) fn require_shape(&self, pack_params: &PackParams) -> Result<()> {
        require_matching_width(self.num_to_pack, pack_params.num_to_pack)?;
        let on_ring = |y: &Poly| {
            y.dimension() == pack_params.ring_dim && y.moduli() == pack_params.moduli.as_slice()
        };
        if self.y_body_ntt.len() == pack_params.key_rows() && self.y_body_ntt.iter().all(on_ring) {
            return Ok(());
        }
        let seat_ring = self
            .y_body_ntt
            .iter()
            .find(|y| !on_ring(y))
            .or(self.y_body_ntt.first());
        Err(pir_err!(
            "packing-key seat is for another ring or gadget: it holds {} key rows on \
             ring_dim {:?} moduli {:?}, the server packs {} rows on ring_dim {} moduli {:?}. \
             The client must re-derive its packing keys against the current CRS.",
            self.y_body_ntt.len(),
            seat_ring.map(Poly::dimension),
            seat_ring.map(Poly::moduli),
            pack_params.key_rows(),
            pack_params.ring_dim,
            pack_params.moduli
        ))
    }
}

/// Server-side map from [`ServerSessionHandle`] to registered packing-key seats.
///
/// `RwLock` because every query reads and registration happens once per session.
/// The store removes explicit handles but has no automatic eviction policy.
/// Allocation is process-global, so replacing a flushed store cannot reissue a handle
/// during that process lifetime. Restart-safe allocation belongs to the embedding server.
///
/// The handle is an unauthenticated u64. Guessing another client's handle yields a
/// response encrypted under that client's keys, so it is a denial of useful response
/// rather than a confidentiality leak; deployments should still bind the handle to a
/// session token at the transport layer.
#[derive(Debug, Default)]
pub struct ServerSessionStore {
    /// `Arc` avoids cloning the seat per lookup.
    inner: RwLock<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    by_handle: HashMap<u64, Arc<PackingKeySeat>>,
}

#[derive(Debug)]
struct HandleAllocator {
    next: AtomicU64,
}

impl HandleAllocator {
    const fn new(next: u64) -> Self {
        Self {
            next: AtomicU64::new(next),
        }
    }

    fn allocate(&self) -> Result<ServerSessionHandle> {
        // The atomic reserves values across independent stores; it orders no session data.
        self.next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map(ServerSessionHandle)
            .map_err(|exhausted| {
                pir_err!(
                    "ServerSessionStore handle space exhausted at {exhausted}; refusing the \
                     session handshake because handle reuse could resolve stale clients to \
                     another client's packing keys"
                )
            })
    }
}

static SESSION_HANDLE_ALLOCATOR: HandleAllocator = HandleAllocator::new(0);

fn require_wire_packing_body_coefficient_domain(keys: &ClientPackingKeys) -> Result<()> {
    for (body_name, polynomials) in [
        ("y_body", keys.y_body.as_slice()),
        ("z_body", keys.z_body.as_slice()),
    ] {
        for (index, poly) in polynomials.iter().enumerate() {
            if poly.is_ntt() {
                return Err(pir_err!(
                    "coefficient-domain packing-key {body_name}[{index}] required; client \
                     material declares NTT domain. Rebuild the keys from the current CRS"
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn require_wire_packing_body_shape(
    keys: &ClientPackingKeys,
    ring_dim: usize,
    moduli: &[u64],
) -> Result<()> {
    require_wire_packing_body_coefficient_domain(keys)?;
    for (body_name, polynomials) in [
        ("y_body", keys.y_body.as_slice()),
        ("z_body", keys.z_body.as_slice()),
    ] {
        for (index, poly) in polynomials.iter().enumerate() {
            if poly.dimension() != ring_dim {
                return Err(pir_err!(
                    "packing-key {body_name}[{index}] dimension {}, expected {ring_dim}; \
                     rebuild the keys from the current CRS",
                    poly.dimension()
                ));
            }
            if poly.moduli() != moduli {
                return Err(pir_err!(
                    "packing-key {body_name}[{index}] moduli {:?}, expected {moduli:?}; \
                     rebuild the keys from the current CRS",
                    poly.moduli()
                ));
            }
        }
    }
    Ok(())
}

/// Refuse client packing keys that do not match the server's packing shape.
///
/// The automorphism generator is `(2n/gamma)+1`, so keys built at another gamma are
/// rotations under a different automorphism than the `bold_t` they multiply, and a key
/// whose rows are not the digits the server decomposes into pairs rows with the wrong
/// digits. Either would pack a wrong plaintext.
///
/// # Errors
/// [`crate::pir::Result`] error naming the mismatched key domain, material, or width.
pub(crate) fn ensure_packing_width_matches(
    keys: &ClientPackingKeys,
    geometry: &PackingGeometry<'_>,
) -> Result<()> {
    require_wire_packing_body_shape(keys, geometry.ring_dim, geometry.moduli)?;
    // The key MATERIAL, not just the declared width: gamma is a self-reported label,
    // `y_body` is what the packing sum reads.
    if keys.y_body.len() != geometry.key_rows {
        return Err(pir_err!(
            "packing-key material mismatch: client sent {} y_body polynomials, the server \
             decomposes into {} digits, and every digit needs its key polynomial. The \
             client must re-derive its packing keys against the current CRS.",
            keys.y_body.len(),
            geometry.key_rows
        ));
    }
    require_matching_width(keys.num_to_pack, geometry.num_to_pack)
}

fn require_matching_width(key_width: usize, server_width: usize) -> Result<()> {
    if key_width == server_width {
        return Ok(());
    }
    Err(pir_err!(
        "packing-key width mismatch: client keys declare gamma = {key_width}, server pack \
         params use gamma = {server_width}. The automorphism generator is (2n/gamma)+1, so \
         these keys rotate under a different automorphism than the server's precomputed \
         table and extraction would return wrong bytes with no error. The client must \
         re-derive its packing keys against the current CRS."
    ))
}

impl ServerSessionStore {
    /// Build an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `seat` under a freshly allocated, monotonically increasing handle.
    pub fn register(&self, seat: PackingKeySeat) -> Result<ServerSessionHandle> {
        let mut inner = self
            .inner
            .write()
            .map_err(|_| pir_err!("ServerSessionStore: lock poisoned on register"))?;
        let handle = SESSION_HANDLE_ALLOCATOR.allocate()?;
        inner.by_handle.insert(handle.0, Arc::new(seat));
        Ok(handle)
    }

    /// Check wire-delivered keys against the server's packing shape and register
    /// their seat.
    pub fn register_server_side(
        &self,
        keys: ClientPackingKeys,
        pack_params: &PackParams,
        ctx: &NttContext,
    ) -> Result<ServerSessionHandle> {
        self.register(PackingKeySeat::from_wire(keys, pack_params, ctx)?)
    }

    /// Seat stored under `handle`, if any.
    pub fn get(&self, handle: ServerSessionHandle) -> Result<Option<Arc<PackingKeySeat>>> {
        let inner = self
            .inner
            .read()
            .map_err(|_| pir_err!("ServerSessionStore: lock poisoned on get"))?;
        Ok(inner.by_handle.get(&handle.0).cloned())
    }

    /// Remove `handle`, returning whether it was registered.
    ///
    /// # Errors
    ///
    /// Returns an error if another thread poisoned the session-store lock.
    ///
    /// ```
    /// use raven_inspire::{ServerSessionHandle, ServerSessionStore};
    ///
    /// let store = ServerSessionStore::new();
    /// assert!(!store.remove(ServerSessionHandle(7))?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn remove(&self, handle: ServerSessionHandle) -> Result<bool> {
        let mut inner = self
            .inner
            .write()
            .map_err(|_| pir_err!("ServerSessionStore: lock poisoned on remove"))?;
        Ok(inner.by_handle.remove(&handle.0).is_some())
    }

    /// Registered session count.
    pub fn len(&self) -> usize {
        self.inner.read().map_or(0, |inner| inner.by_handle.len())
    }

    /// Whether the store holds no sessions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod from_residue_failfast_tests {
    use super::*;
    use crate::math::GaussianSampler;
    use crate::params::{InspireParams, SecurityLevel};
    use crate::setup;

    fn fresh_residue() -> SessionResidue {
        let params = InspireParams {
            ring_dim: 256,
            q: 1_152_921_504_606_830_593,
            crt_moduli: vec![1_152_921_504_606_830_593],
            p: 65_536,
            sigma: 6.4,
            gadget_base: 1 << 20,
            query_gadget_len: 3,
            packing_gadget_len: 3,
            approximate_packing_gadget: false,
            security_level: SecurityLevel::Bits128,
        };
        let db = vec![0u8; params.ring_dim * 32];
        let mut sampler = GaussianSampler::with_seed(params.sigma, 23);
        let (crs, _encoded_db, rlwe_sk) = setup(&params, &db, 32, &mut sampler).expect("setup");
        let session = ClientSession::new(crs, rlwe_sk, &mut sampler).expect("session");
        session.to_residue()
    }

    #[test]
    fn from_residue_rejects_ring_dim_mismatch() {
        let mut residue = fresh_residue();
        residue.crs.params.ring_dim += 1;
        let err =
            ClientSession::from_residue(residue).expect_err("ring_dim mismatch must be rejected");
        assert!(
            err.to_string().contains("inconsistent SessionResidue"),
            "got: {err}"
        );
    }

    #[test]
    fn from_residue_rejects_modulus_mismatch() {
        let mut residue = fresh_residue();
        residue.crs.params.q ^= 1;
        let err =
            ClientSession::from_residue(residue).expect_err("modulus mismatch must be rejected");
        assert!(
            err.to_string().contains("inconsistent SessionResidue"),
            "got: {err}"
        );
    }
}

#[cfg(test)]
mod session_store_internal_tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn handle_allocator_refuses_exhaustion_without_wrapping() {
        let allocator = HandleAllocator::new(u64::MAX - 1);
        assert_eq!(
            allocator.allocate().expect("last available handle"),
            ServerSessionHandle(u64::MAX - 1)
        );
        for _ in 0..2 {
            let error = allocator
                .allocate()
                .expect_err("exhausted allocator must stay exhausted");
            let message = error.to_string();
            assert!(message.contains("exhausted"), "{message}");
            assert!(message.contains("handle reuse"), "{message}");
        }
    }

    #[test]
    fn remove_reports_a_poisoned_store_lock() {
        let store = Arc::new(ServerSessionStore::new());
        let poisoner = Arc::clone(&store);
        let joined = std::thread::spawn(move || {
            let _guard = poisoner.inner.write().expect("unpoisoned premise");
            panic!("poison session store for error-path coverage");
        })
        .join();
        assert!(joined.is_err(), "poisoning thread must panic");

        let error = store
            .remove(ServerSessionHandle(0))
            .expect_err("remove must report lock poisoning");
        assert_eq!(
            error.to_string(),
            "ServerSessionStore: lock poisoned on remove"
        );
    }
}
