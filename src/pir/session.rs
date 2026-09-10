//! Packing params and client packing keys depend only on (CRS, rlwe_sk, w_seed), never
//! on the query index, so a session computes them once and clones them per query. The
//! automorph-table build they hide is O(d^3).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::inspiring::{ClientPackingKeys, PackParams};
use crate::lwe::LweSecretKey;
use crate::math::{GaussianSampler, NttContext};
use crate::params::ShardConfig;
use crate::rgsw::{RgswCiphertext, SeededRgswCiphertext};
use crate::rlwe::RlweSecretKey;

use super::encode_db::inverse_monomial;
use super::error::{pir_err, Result};
use super::query::{
    ensure_query_shard_geometry, ClientQuery, ClientState, PackingMode, SeededClientQuery,
    ServerSessionHandle,
};
use super::setup::ServerCrs;

/// Everything a client needs to emit many queries against one server-session CRS.
///
/// Optionally compresses the wire format: [`register_with`](Self::register_with)
/// uploads the ~48 KiB packing keys once and every later query carries a handle
/// instead. Without it the keys are inlined per query.
#[derive(Clone)]
pub struct ClientSession {
    /// Owned CRS snapshot; rotating the CRS means building a new session.
    crs: ServerCrs,
    rlwe_sk: RlweSecretKey,
    lwe_sk: LweSecretKey,
    /// Owns the O(d^3) automorph tables; `None` on a Tree-packing-only server.
    pack_params: Option<PackParams>,
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
    /// Build a session, running the one-time packing-key work.
    pub fn new(
        crs: ServerCrs,
        rlwe_sk: RlweSecretKey,
        sampler: &mut GaussianSampler,
    ) -> Result<Self> {
        crs.validate()?;
        let lwe_sk = rlwe_to_lwe_key(&rlwe_sk);

        let (pack_params, packing_keys) = if crs.inspiring_num_columns > 0 {
            let pp = PackParams::try_new(&crs.params, crs.inspiring_num_columns)
                .map_err(|e| pir_err!("CRS carries an unusable InspiRING width: {e}"))?;
            let keys = ClientPackingKeys::generate(&rlwe_sk, &pp, crs.inspiring_w_seed, sampler);
            (Some(pp), Some(keys))
        } else {
            (None, None)
        };

        Ok(Self {
            crs,
            rlwe_sk,
            lwe_sk,
            pack_params,
            packing_keys,
            session_handle: None,
        })
    }

    /// Upload the cached packing keys and switch to the compact wire format.
    /// `Ok(None)` means there were no keys to upload.
    pub fn register_with(
        &mut self,
        store: &ServerSessionStore,
    ) -> Result<Option<ServerSessionHandle>> {
        match &self.packing_keys {
            None => Ok(None),
            Some(keys) => {
                let handle = store.register(keys.clone())?;
                self.session_handle = Some(handle);
                Ok(Some(handle))
            }
        }
    }

    /// Current handle, or `None` if nothing was registered.
    pub fn session_handle(&self) -> Option<ServerSessionHandle> {
        self.session_handle
    }

    /// Register with the derived `y_all*` fields stripped, mimicking what survives a
    /// wire round-trip, so the server derives them once and later queries take the
    /// fully-NTT respond path. In-process callers want
    /// [`register_with`](Self::register_with) instead.
    pub fn register_with_server_derivation(
        &mut self,
        store: &ServerSessionStore,
    ) -> Result<Option<ServerSessionHandle>> {
        let (pp, ctx, wire_keys) = match (&self.pack_params, &self.packing_keys) {
            (Some(pp), Some(keys)) => {
                let ctx = self.crs.params.ntt_context();
                let wire_keys = ClientPackingKeys {
                    y_body: keys.y_body.clone(),
                    z_body: keys.z_body.clone(),
                    y_all: Vec::new(),
                    y_all_ntt: Vec::new(),
                    y_bar_all: Vec::new(),
                    y_bar_all_ntt: Vec::new(),
                    full_key: keys.full_key,
                    num_to_pack: keys.num_to_pack,
                };
                (pp, ctx, wire_keys)
            }
            _ => return Ok(None),
        };
        let handle = store.register_server_side(wire_keys, pp, &ctx)?;
        self.session_handle = Some(handle);
        Ok(Some(handle))
    }

    /// Borrow the captured CRS.
    pub fn crs(&self) -> &ServerCrs {
        &self.crs
    }

    /// Borrow the captured RLWE secret key, e.g. to extract a response.
    pub fn rlwe_secret_key(&self) -> &RlweSecretKey {
        &self.rlwe_sk
    }

    /// Emit a `ClientQuery`: inverse monomial, RGSW encrypt, clone cached keys.
    pub fn query(
        &self,
        global_index: u64,
        shard_config: &ShardConfig,
        sampler: &mut GaussianSampler,
    ) -> Result<(ClientState, ClientQuery)> {
        self.ensure_inspiring_query_material()?;
        ensure_query_shard_geometry(&self.crs, shard_config)?;
        let d = self.crs.ring_dim();
        let q = self.crs.modulus();
        let ctx = self.crs.params.ntt_context();

        let (shard_id, local_index) = shard_config.index_to_shard(global_index);

        let inv_mono = inverse_monomial(local_index as usize, d, q, self.crs.params.moduli());
        let rgsw_ciphertext = RgswCiphertext::encrypt(
            &self.rlwe_sk,
            &inv_mono,
            &self.crs.rgsw_gadget,
            sampler,
            &ctx,
        );

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

    /// [`query`](Self::query) in the compact seeded form.
    pub fn query_seeded(
        &self,
        global_index: u64,
        shard_config: &ShardConfig,
        sampler: &mut GaussianSampler,
    ) -> Result<(ClientState, SeededClientQuery)> {
        self.ensure_inspiring_query_material()?;
        ensure_query_shard_geometry(&self.crs, shard_config)?;
        let d = self.crs.ring_dim();
        let q = self.crs.modulus();
        let ctx = self.crs.params.ntt_context();

        let (shard_id, local_index) = shard_config.index_to_shard(global_index);

        let inv_mono = inverse_monomial(local_index as usize, d, q, self.crs.params.moduli());
        let rgsw_ciphertext = SeededRgswCiphertext::encrypt(
            &self.rlwe_sk,
            &inv_mono,
            &self.crs.rgsw_gadget,
            sampler,
            &ctx,
        );

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
            SeededClientQuery {
                shard_id,
                rgsw_ciphertext,
                packing_mode: PackingMode::Inspiring,
                inspiring_packing_keys,
                session_handle,
            },
        ))
    }

    /// Whether InspiRING packing machinery is cached.
    pub fn has_inspiring_packing(&self) -> bool {
        self.packing_keys.is_some()
    }

    fn ensure_inspiring_query_material(&self) -> Result<()> {
        if self.packing_keys.is_none() {
            return Err(pir_err!(
                "PIR query refused a zero InspiRING width: no packing material exists, and \
                 falling back to a tree-packed response would return wrong bytes under the \
                 TwoPacking extractor"
            ));
        }
        Ok(())
    }

    /// Borrow the cached `PackParams`.
    pub fn pack_params(&self) -> Option<&PackParams> {
        self.pack_params.as_ref()
    }

    /// Capture the persistable residue: no automorph tables, no derived rotations,
    /// no handle (it is only valid against the issuing store).
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

    /// Rehydrate without rebuilding the automorph tables, so the result serves
    /// queries but not
    /// [`register_with_server_derivation`](Self::register_with_server_derivation).
    /// Mismatched CRS and secret-key shapes are rejected here rather than at a
    /// later query.
    pub fn from_residue(residue: SessionResidue) -> Result<Self> {
        if residue.crs.ring_dim() != residue.rlwe_sk.ring_dim()
            || residue.crs.modulus() != residue.rlwe_sk.modulus()
        {
            return Err(pir_err!(
                "inconsistent SessionResidue: CRS (ring_dim {}, modulus {}) != secret key (ring_dim {}, modulus {})",
                residue.crs.ring_dim(),
                residue.crs.modulus(),
                residue.rlwe_sk.ring_dim(),
                residue.rlwe_sk.modulus()
            ));
        }
        let lwe_sk = rlwe_to_lwe_key(&residue.rlwe_sk);
        Ok(Self {
            crs: residue.crs,
            rlwe_sk: residue.rlwe_sk,
            lwe_sk,
            pack_params: None,
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

fn rlwe_to_lwe_key(rlwe_sk: &RlweSecretKey) -> LweSecretKey {
    let d = rlwe_sk.ring_dim();
    let mut coeffs = Vec::with_capacity(d);
    for i in 0..d {
        coeffs.push(rlwe_sk.poly.coeff(i));
    }
    let q = rlwe_sk.modulus();
    LweSecretKey::from_coeffs(coeffs, q)
}

/// Server-side map from [`ServerSessionHandle`] to registered packing keys.
///
/// `RwLock` because every query reads and registration happens once per session.
/// The store never evicts; wrap it if memory pressure matters.
///
/// The handle is an unauthenticated u64. Guessing another client's handle yields a
/// response encrypted under that client's keys, so it is a denial of useful response
/// rather than a confidentiality leak; deployments should still bind the handle to a
/// session token at the transport layer.
#[derive(Debug, Default)]
pub struct ServerSessionStore {
    /// `Arc` avoids cloning the ~48 KiB payload per lookup.
    inner: RwLock<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    by_handle: HashMap<u64, Arc<ClientPackingKeys>>,
    next_handle: u64,
}

/// Refuse client packing keys whose gamma differs from the server's.
///
/// The automorphism generator is `(2n/gamma)+1`, so keys built at another gamma are
/// rotations under a different automorphism than the `bold_t` they multiply.
/// `packing_online` guards its loop with `if i < len` and SKIPS the shortfall rather than
/// rejecting it, so a mismatch surfaces as a successful response carrying wrong plaintext.
///
/// Called from every site where client-supplied keys meet server pack params:
/// `register_server_side` here, plus the three inlined-key respond paths.
///
/// # Errors
/// [`crate::pir::Result`] error naming both widths.
pub(crate) fn ensure_packing_width_matches(
    keys: &ClientPackingKeys,
    pack_params: &PackParams,
) -> Result<()> {
    // The key MATERIAL, not just the declared width. `generate_rotations` sizes the
    // outer vector from the server's gamma, but each rotation is `y_body.len()` long,
    // and `packing_online`'s inner loop skips every digit `k >= y_all[i].len()`. A short
    // `y_body` under a correct `num_to_pack` therefore drops terms from the packing sum
    // and returns wrong plaintext at HTTP 200 - the same failure the gamma check exists
    // to stop, reached by a field the gamma check never reads.
    if keys.y_body.len() != pack_params.gadget.len {
        return Err(pir_err!(
            "packing-key material mismatch: client sent {} y_body polynomials, server \
             gadget has {} digits. `packing_online` skips every digit past the shorter \
             of the two, so the packing sum would silently lose terms and extraction \
             would return wrong bytes with no error. The client must re-derive its \
             packing keys against the current CRS.",
            keys.y_body.len(),
            pack_params.gadget.len
        ));
    }
    if keys.num_to_pack == pack_params.num_to_pack {
        return Ok(());
    }
    Err(pir_err!(
        "packing-key width mismatch: client keys declare gamma = {}, server pack params \
         use gamma = {}. The automorphism generator is (2n/gamma)+1, so these keys rotate \
         under a different automorphism than the server's precomputed table and \
         extraction would return wrong bytes with no error. The client must re-derive its \
         packing keys against the current CRS.",
        keys.num_to_pack,
        pack_params.num_to_pack
    ))
}

impl ServerSessionStore {
    /// Build an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `keys` under a freshly allocated, monotonically increasing handle.
    pub fn register(&self, keys: ClientPackingKeys) -> Result<ServerSessionHandle> {
        let mut inner = self
            .inner
            .write()
            .map_err(|_| pir_err!("ServerSessionStore: lock poisoned on register"))?;
        let h = inner.next_handle;
        inner.next_handle = inner.next_handle.checked_add(1).unwrap_or(0);
        inner.by_handle.insert(h, Arc::new(keys));
        Ok(ServerSessionHandle(h))
    }

    /// Register wire-delivered keys, deriving the `y_all*` rotations that
    /// `#[serde(skip)]` dropped so later responses take the fully-NTT path. The
    /// derivation is skipped when they are already populated.
    pub fn register_server_side(
        &self,
        mut keys: ClientPackingKeys,
        pack_params: &PackParams,
        ctx: &NttContext,
    ) -> Result<ServerSessionHandle> {
        // Geometry must agree before any derivative is built. Not the only place the
        // two meet: `respond_inspiring`, `respond_inspiring_cached` and
        // `respond_inspiring_cached_with_session` accept inlined keys and pair them with
        // server pack params too, so each calls the same guard.
        ensure_packing_width_matches(&keys, pack_params)?;
        keys.ensure_server_derivatives(pack_params, ctx);
        self.register(keys)
    }

    /// Packing keys stored under `handle`, if any.
    pub fn get(&self, handle: ServerSessionHandle) -> Result<Option<Arc<ClientPackingKeys>>> {
        let inner = self
            .inner
            .read()
            .map_err(|_| pir_err!("ServerSessionStore: lock poisoned on get"))?;
        Ok(inner.by_handle.get(&handle.0).cloned())
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
            gadget_len: 3,
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
