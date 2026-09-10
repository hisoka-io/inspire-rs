//! PIR.Respond: the external product RLWE(h) * RGSW(X^(-k)) rotates the target value
//! into coefficient 0.

use crate::par_prelude::*;
use serde::{Deserialize, Serialize};

use crate::inspiring::{packing_online, packing_online_fully_ntt};
use crate::math::Poly;
use crate::params::InspireVariant;
use crate::rgsw::{external_product_with_ntt_rgsw, rgsw_rows_to_ntt};
use crate::rlwe::RlweCiphertext;

use super::error::{pir_err, Result};
use super::query::{ClientQuery, PackingMode, SeededClientQuery};
use super::setup::{EncodedDatabase, ServerCrs, ShardData};

/// Server response: the packed ciphertext, or one ciphertext per column.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerResponse {
    /// Packed result, or the sum of the column ciphertexts when unpacked.
    pub ciphertext: RlweCiphertext,
    /// Populated only on the unpacked path.
    pub column_ciphertexts: Vec<RlweCiphertext>,
    /// Lets the extractor tell unscaled InspiRING output from d-scaled tree output.
    ///
    /// Never `skip_serializing_if`: bincode is positional, so an omitted field
    /// shifts every later read.
    #[serde(default)]
    pub packing_mode: Option<PackingMode>,
}

impl ServerResponse {
    /// Serialize to bincode.
    pub fn to_binary(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| pir_err!("bincode serialize failed: {}", e))
    }

    /// Deserialize from bincode.
    pub fn from_binary(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).map_err(|e| pir_err!("bincode deserialize failed: {}", e))
    }
}

fn require_shard_width(operation: &'static str, crs: &ServerCrs, shard: &ShardData) -> Result<()> {
    let got = shard.polynomials.len();
    let expected = crs.inspiring_num_columns;
    if expected == 0 || got != expected {
        return Err(pir_err!(
            "{operation}: shard {} column-count mismatch: got {got}, expected {expected} from \
             the CRS; refusing malformed encoded state before it returns wrong plaintext",
            shard.id,
        ));
    }
    Ok(())
}

/// Respond with one RLWE ciphertext per column, in parallel.
pub fn respond(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &ClientQuery,
) -> Result<ServerResponse> {
    let delta = crs.params.delta();
    let ctx = crs.params.ntt_context();

    let shard = encoded_db
        .shards
        .iter()
        .find(|s| s.id == query.shard_id)
        .ok_or_else(|| pir_err!("Shard {} not found", query.shard_id))?;
    require_shard_width("respond", crs, shard)?;

    // RGSW is constant across a shard's columns, so its forward NTTs amortize.
    let rgsw_ntt_rows = rgsw_rows_to_ntt(&query.rgsw_ciphertext, &ctx);
    let rgsw_gadget = &query.rgsw_ciphertext.gadget;
    let column_ciphertexts: Vec<RlweCiphertext> = shard
        .polynomials
        .par_iter()
        .map(|db_poly| {
            let rlwe_db = RlweCiphertext::trivial_encrypt(db_poly, delta, &crs.params);
            external_product_with_ntt_rgsw(&rlwe_db, &rgsw_ntt_rows, rgsw_gadget, &ctx)
        })
        .collect();

    let combined = if column_ciphertexts.len() == 1 {
        column_ciphertexts[0].clone()
    } else {
        column_ciphertexts
            .iter()
            .skip(1)
            .fold(column_ciphertexts[0].clone(), |acc, ct| acc.add(ct))
    };

    Ok(ServerResponse {
        ciphertext: combined,
        column_ciphertexts,
        packing_mode: None,
    })
}

/// Respond under an explicit variant. Tree packing runs only when asked for.
pub fn respond_with_variant(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &ClientQuery,
    variant: InspireVariant,
) -> Result<ServerResponse> {
    match variant {
        InspireVariant::NoPacking => respond(crs, encoded_db, query),
        InspireVariant::OnePacking => respond_one_packing(crs, encoded_db, query),
        // Refused rather than routed through OnePacking: the extractor would decode a
        // mismatched format and return wrong plaintext with no error.
        InspireVariant::TwoPacking => Err(pir_err!(
            "respond_with_variant(TwoPacking) is not supported on an unseeded \
             ClientQuery: TwoPacking requires the seeded pipeline \
             (query_seeded + respond_seeded_with_variant or respond_seeded_inspiring / \
             respond_seeded_packed). See docs/GOOGLE_ALIGNMENT.md."
        )),
    }
}

/// Respond to a seeded query under an explicit variant.
pub fn respond_seeded_with_variant(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &SeededClientQuery,
    variant: InspireVariant,
) -> Result<ServerResponse> {
    match variant {
        InspireVariant::NoPacking => respond_seeded(crs, encoded_db, query),
        InspireVariant::OnePacking => respond_seeded_packed(crs, encoded_db, query),
        InspireVariant::TwoPacking => match query.packing_mode {
            PackingMode::Inspiring => {
                if query.inspiring_packing_keys.is_none() {
                    return Err(pir_err!("TwoPacking requires InspiRING packing keys"));
                }
                respond_seeded_inspiring(crs, encoded_db, query)
            }
            PackingMode::Tree => Err(pir_err!(
                "respond_seeded_with_variant(TwoPacking) refuses PackingMode::Tree: \
                 TwoPacking requires an InspiRING response"
            )),
        },
    }
}

/// Respond with one tree-packed ciphertext holding column k at coefficient k.
///
/// Requires `crs.galois_keys`. Shift-and-add cannot replace the automorphism tree:
/// a key-switched RLWE carries noise in every coefficient, not just the target one.
pub fn respond_one_packing(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &ClientQuery,
) -> Result<ServerResponse> {
    use crate::inspiring::automorph_pack::pack_lwes;

    let _d = crs.ring_dim();
    let _q = crs.modulus();
    let delta = crs.params.delta();
    let ctx = crs.params.ntt_context();

    let shard = encoded_db
        .shards
        .iter()
        .find(|s| s.id == query.shard_id)
        .ok_or_else(|| pir_err!("Shard {} not found", query.shard_id))?;
    require_shard_width("respond_one_packing", crs, shard)?;

    let rgsw_ntt_rows = rgsw_rows_to_ntt(&query.rgsw_ciphertext, &ctx);
    let rgsw_gadget = &query.rgsw_ciphertext.gadget;
    let column_ciphertexts: Vec<RlweCiphertext> = shard
        .polynomials
        .par_iter()
        .map(|db_poly| {
            let rlwe_db = RlweCiphertext::trivial_encrypt(db_poly, delta, &crs.params);
            external_product_with_ntt_rgsw(&rlwe_db, &rgsw_ntt_rows, rgsw_gadget, &ctx)
        })
        .collect();

    let lwe_cts: Vec<_> = column_ciphertexts
        .iter()
        .map(crate::rlwe::RlweCiphertext::sample_extract_coeff0)
        .collect();

    // Places column k at coefficient k, scaled by d.
    let packed = pack_lwes(&lwe_cts, &crs.galois_keys, &crs.params);

    Ok(ServerResponse {
        ciphertext: packed,
        column_ciphertexts: vec![],
        packing_mode: Some(PackingMode::Tree),
    })
}

/// Respond with InspiRING 2-matrix packing, roughly 35x faster online than the
/// log(d)-matrix tree.
///
/// `packing_offline` runs per query, not at setup: its a-vectors are derived from
/// the RGSW query, so a CRS-static precomputation would be the wrong shape.
pub fn respond_inspiring(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &ClientQuery,
) -> Result<ServerResponse> {
    use crate::inspiring::{generate_rotations, packing_offline, OfflinePackingKeys, PackParams};

    let d = crs.ring_dim();
    let delta = crs.params.delta();
    let ctx = crs.params.ntt_context();

    let client_packing_keys = query
        .inspiring_packing_keys
        .as_ref()
        .ok_or_else(|| pir_err!("InspiRING client packing keys missing from query"))?;

    let shard = encoded_db
        .shards
        .iter()
        .find(|s| s.id == query.shard_id)
        .ok_or_else(|| pir_err!("Shard {} not found", query.shard_id))?;
    require_shard_width("respond_inspiring", crs, shard)?;

    let rgsw_ntt_rows = rgsw_rows_to_ntt(&query.rgsw_ciphertext, &ctx);
    let rgsw_gadget = &query.rgsw_ciphertext.gadget;
    let column_ciphertexts: Vec<RlweCiphertext> = shard
        .polynomials
        .par_iter()
        .map(|db_poly| {
            let rlwe_db = RlweCiphertext::trivial_encrypt(db_poly, delta, &crs.params);
            external_product_with_ntt_rgsw(&rlwe_db, &rgsw_ntt_rows, rgsw_gadget, &ctx)
        })
        .collect();

    let lwe_cts: Vec<_> = column_ciphertexts
        .iter()
        .map(crate::rlwe::RlweCiphertext::sample_extract_coeff0)
        .collect();

    let num_columns = lwe_cts.len();

    // InspiRING consumes the RLWE a-polynomials, not the LWE a-vectors: the latter are
    // negacyclic extractions with a different structure.
    let a_ct_tilde: Vec<Poly> = column_ciphertexts
        .iter()
        .map(|rlwe| rlwe.a.clone())
        .collect();

    let mut b_coeffs = vec![0u64; d];
    for (i, lwe) in lwe_cts.iter().enumerate() {
        if i < d {
            b_coeffs[i] = lwe.b;
        }
    }
    let b_poly = Poly::from_coeffs_moduli(b_coeffs, crs.params.moduli());

    let pack_params = PackParams::try_new(&crs.params, num_columns)
        .map_err(|e| pir_err!("shard column count is not a legal InspiRING width: {e}"))?;
    let offline_keys = OfflinePackingKeys::generate(&pack_params, crs.inspiring_w_seed);
    let precomp = packing_offline(&pack_params, &offline_keys, &a_ct_tilde, &ctx);

    // The wire format drops y_all, so re-derive it from y_body when absent.
    // Inlined keys bypass `register_server_side`, so the geometry guard has to run here
    // too or a width mismatch returns a successful response carrying wrong plaintext.
    crate::pir::session::ensure_packing_width_matches(client_packing_keys, &pack_params)?;
    let derived_y_all = if client_packing_keys.y_all.is_empty() {
        if client_packing_keys.y_body.is_empty() {
            return Err(pir_err!(
                "InspiRING packing keys invalid: y_all and y_body are both empty"
            ));
        }
        Some(generate_rotations(
            &pack_params,
            &client_packing_keys.y_body,
        ))
    } else {
        None
    };
    let y_all: &[Vec<Poly>] = derived_y_all
        .as_deref()
        .unwrap_or(&client_packing_keys.y_all);

    let packed = packing_online(&precomp, y_all, &b_poly, &ctx);

    Ok(ServerResponse {
        ciphertext: packed,
        column_ciphertexts: vec![],
        packing_mode: Some(PackingMode::Inspiring),
    })
}

/// Query-independent InspiRING pack params and offline keys, built once per CRS.
///
/// Both depend only on `(crs.params, num_columns, crs.inspiring_w_seed)`, and
/// `num_columns` is fixed by the shard shape chosen at setup.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerInspiringCache {
    pack_params: crate::inspiring::PackParams,
    offline_keys: crate::inspiring::OfflinePackingKeys,
}

fn validate_cache_parts(
    operation: &'static str,
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    pack_params: &crate::inspiring::PackParams,
    offline_keys: &crate::inspiring::OfflinePackingKeys,
) -> Result<()> {
    let expected_columns = crs.inspiring_num_columns;
    if expected_columns == 0 {
        return Err(pir_err!("{operation}: CRS has zero InspiRING columns"));
    }
    if encoded_db.shards.is_empty() {
        return Err(pir_err!("{operation}: encoded database has no shards"));
    }
    for (shard_index, shard) in encoded_db.shards.iter().enumerate() {
        let columns = shard.polynomials.len();
        if columns != expected_columns {
            return Err(pir_err!(
                "{operation}: shard {shard_index} column count {columns} != CRS width \
                 {expected_columns}"
            ));
        }
    }
    if pack_params.num_to_pack != expected_columns
        || pack_params.ring_dim != crs.params.ring_dim
        || pack_params.q != crs.params.q
        || pack_params.moduli != crs.params.moduli()
        || pack_params.gadget.base != crs.params.gadget_base
        || pack_params.gadget.len != crs.params.gadget_len
        || pack_params.gadget.q != crs.params.q
    {
        return Err(pir_err!(
            "{operation}: cached pack params do not match CRS params at width {expected_columns}"
        ));
    }
    if offline_keys.w_seed != crs.inspiring_w_seed
        || offline_keys.full_key
        || offline_keys.w_mask.len() != pack_params.gadget.len
        || offline_keys.w_all.len() != expected_columns - 1
        || offline_keys.w_all_ntt.len() != expected_columns - 1
    {
        return Err(pir_err!(
            "{operation}: offline packing keys do not match CRS seed or width {expected_columns}"
        ));
    }
    Ok(())
}

impl ServerInspiringCache {
    /// Build the cache, paying the one-time O(d^3) automorph-table search.
    pub fn new(crs: &ServerCrs, encoded_db: &EncodedDatabase) -> Result<Self> {
        let num_columns = encoded_db.shards.first().map_or(0, |s| s.polynomials.len());
        if num_columns == 0 {
            return Err(pir_err!(
                "ServerInspiringCache::new: encoded_db has no shard polynomials"
            ));
        }
        let pack_params = crate::inspiring::PackParams::try_new(&crs.params, num_columns)
            .map_err(|e| pir_err!("shard column count is not a legal InspiRING width: {e}"))?;
        let offline_keys =
            crate::inspiring::OfflinePackingKeys::generate(&pack_params, crs.inspiring_w_seed);
        Ok(Self {
            pack_params,
            offline_keys,
        })
    }

    /// Consume the cache parts produced by a fresh [`crate::pir::setup`] call.
    ///
    /// # Errors
    ///
    /// Returns an actionable error before taking either field when the database width,
    /// parameters, seed, or cached material does not match the CRS.
    pub fn from_setup(crs: &mut ServerCrs, encoded_db: &EncodedDatabase) -> Result<Self> {
        let pack_params = crs.inspiring_pack_params.as_ref().ok_or_else(|| {
            pir_err!("ServerInspiringCache::from_setup: fresh CRS is missing pack params")
        })?;
        let offline_keys = crs.inspiring_packing_key.as_ref().ok_or_else(|| {
            pir_err!("ServerInspiringCache::from_setup: fresh CRS is missing offline packing keys")
        })?;
        validate_cache_parts(
            "ServerInspiringCache::from_setup",
            crs,
            encoded_db,
            pack_params,
            offline_keys,
        )?;

        let pack_params = crs.inspiring_pack_params.take().ok_or_else(|| {
            pir_err!("ServerInspiringCache::from_setup: pack params disappeared before take")
        })?;
        let offline_keys = crs.inspiring_packing_key.take().ok_or_else(|| {
            pir_err!("ServerInspiringCache::from_setup: offline keys disappeared before take")
        })?;
        Ok(Self {
            pack_params,
            offline_keys,
        })
    }

    /// Rebuild from serialised parts, skipping the work [`ServerInspiringCache::new`]
    /// performs. Validates nothing: the caller must confirm the parts match the
    /// current CRS and encoded database.
    pub fn from_parts(
        pack_params: crate::inspiring::PackParams,
        offline_keys: crate::inspiring::OfflinePackingKeys,
    ) -> Self {
        Self {
            pack_params,
            offline_keys,
        }
    }

    /// Validate cached parts against the CRS, shard width, and public seed.
    ///
    /// # Errors
    ///
    /// Returns an actionable mismatch instead of allowing cached data to produce wrong responses.
    pub fn validate_for(&self, crs: &ServerCrs, encoded_db: &EncodedDatabase) -> Result<()> {
        validate_cache_parts(
            "ServerInspiringCache::validate_for",
            crs,
            encoded_db,
            &self.pack_params,
            &self.offline_keys,
        )
    }

    /// Borrow the cached pack params.
    pub fn pack_params(&self) -> &crate::inspiring::PackParams {
        &self.pack_params
    }

    /// Borrow the cached offline keys.
    pub fn offline_keys(&self) -> &crate::inspiring::OfflinePackingKeys {
        &self.offline_keys
    }
}

/// [`respond_inspiring`] against a pre-built cache. `packing_offline` still runs per
/// query because its `a_ct_tilde` input is query-derived.
pub fn respond_inspiring_cached(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &ClientQuery,
    cache: &ServerInspiringCache,
) -> Result<ServerResponse> {
    use crate::inspiring::{generate_rotations, packing_offline};

    let d = crs.ring_dim();
    let delta = crs.params.delta();
    let ctx = crs.params.ntt_context();

    let client_packing_keys = query
        .inspiring_packing_keys
        .as_ref()
        .ok_or_else(|| pir_err!("InspiRING client packing keys missing from query"))?;

    let shard = encoded_db
        .shards
        .iter()
        .find(|s| s.id == query.shard_id)
        .ok_or_else(|| pir_err!("Shard {} not found", query.shard_id))?;
    require_shard_width("respond_inspiring_cached", crs, shard)?;

    let rgsw_ntt_rows = rgsw_rows_to_ntt(&query.rgsw_ciphertext, &ctx);
    let rgsw_gadget = &query.rgsw_ciphertext.gadget;
    let column_ciphertexts: Vec<RlweCiphertext> = shard
        .polynomials
        .par_iter()
        .map(|db_poly| {
            let rlwe_db = RlweCiphertext::trivial_encrypt(db_poly, delta, &crs.params);
            external_product_with_ntt_rgsw(&rlwe_db, &rgsw_ntt_rows, rgsw_gadget, &ctx)
        })
        .collect();

    let lwe_cts: Vec<_> = column_ciphertexts
        .iter()
        .map(crate::rlwe::RlweCiphertext::sample_extract_coeff0)
        .collect();

    let a_ct_tilde: Vec<Poly> = column_ciphertexts
        .iter()
        .map(|rlwe| rlwe.a.clone())
        .collect();

    let mut b_coeffs = vec![0u64; d];
    for (i, lwe) in lwe_cts.iter().enumerate() {
        if i < d {
            b_coeffs[i] = lwe.b;
        }
    }
    let b_poly = Poly::from_coeffs_moduli(b_coeffs, crs.params.moduli());

    let precomp = packing_offline(&cache.pack_params, &cache.offline_keys, &a_ct_tilde, &ctx);

    // Inlined keys bypass `register_server_side`, so the geometry guard has to run here
    // too or a width mismatch returns a successful response carrying wrong plaintext.
    crate::pir::session::ensure_packing_width_matches(client_packing_keys, &cache.pack_params)?;
    let derived_y_all = if client_packing_keys.y_all.is_empty() {
        if client_packing_keys.y_body.is_empty() {
            return Err(pir_err!(
                "InspiRING packing keys invalid: y_all and y_body are both empty"
            ));
        }
        Some(generate_rotations(
            &cache.pack_params,
            &client_packing_keys.y_body,
        ))
    } else {
        None
    };
    let y_all: &[Vec<Poly>] = derived_y_all
        .as_deref()
        .unwrap_or(&client_packing_keys.y_all);

    let packed = packing_online(&precomp, y_all, &b_poly, &ctx);

    Ok(ServerResponse {
        ciphertext: packed,
        column_ciphertexts: vec![],
        packing_mode: Some(PackingMode::Inspiring),
    })
}

/// Seeded sibling of [`respond_inspiring_cached`].
pub fn respond_seeded_inspiring_cached(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &SeededClientQuery,
    cache: &ServerInspiringCache,
) -> Result<ServerResponse> {
    let expanded = query.expand();
    respond_inspiring_cached(crs, encoded_db, &expanded, cache)
}

/// [`respond_inspiring_cached`] resolving packing keys from a session store when the
/// query carries a handle, which lets it drop its ~48 KiB inline key payload.
pub fn respond_inspiring_cached_with_session(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &ClientQuery,
    cache: &ServerInspiringCache,
    session_store: Option<&super::session::ServerSessionStore>,
) -> Result<ServerResponse> {
    use crate::inspiring::{generate_rotations, packing_offline};

    let d = crs.ring_dim();
    let delta = crs.params.delta();
    let ctx = crs.params.ntt_context();

    let resolved_keys = match (&query.inspiring_packing_keys, query.session_handle) {
        (Some(inline), None) => PackingKeys::Inline(inline),
        (None, Some(handle)) => {
            let store = session_store.ok_or_else(|| {
                pir_err!(
                    "query references session_handle {:?} but no \
                     ServerSessionStore was supplied to respond_*_with_session",
                    handle
                )
            })?;
            let arc = store.get(handle)?.ok_or_else(|| {
                pir_err!(
                    "ServerSessionStore has no entry for session_handle {:?}",
                    handle
                )
            })?;
            PackingKeys::Owned(arc)
        }
        (Some(_), Some(h)) => {
            return Err(pir_err!(
                "query has both inlined inspiring_packing_keys AND \
                 session_handle {:?}; exactly one must be set",
                h
            ))
        }
        (None, None) => {
            return Err(pir_err!(
                "InspiRING client packing keys missing from query \
                 (neither inlined nor session_handle set)"
            ))
        }
    };
    let client_packing_keys = resolved_keys.as_ref();

    let shard = encoded_db
        .shards
        .iter()
        .find(|s| s.id == query.shard_id)
        .ok_or_else(|| pir_err!("Shard {} not found", query.shard_id))?;
    require_shard_width("respond_inspiring_cached_with_session", crs, shard)?;

    // Set RAVEN_PROFILE_RESPOND to any value for a per-region stderr breakdown.
    let profile = std::env::var_os("RAVEN_PROFILE_RESPOND").is_some();
    let t_extprod_start = if profile {
        Some(std::time::Instant::now())
    } else {
        None
    };

    // RGSW is constant across a shard's columns, so its forward NTTs amortize.
    let rgsw_ntt = rgsw_rows_to_ntt(&query.rgsw_ciphertext, &ctx);
    let gadget = &query.rgsw_ciphertext.gadget;

    let column_ciphertexts: Vec<RlweCiphertext> = shard
        .polynomials
        .par_iter()
        .map(|db_poly| {
            let rlwe_db = RlweCiphertext::trivial_encrypt(db_poly, delta, &crs.params);
            external_product_with_ntt_rgsw(&rlwe_db, &rgsw_ntt, gadget, &ctx)
        })
        .collect();

    let t_extprod_end = t_extprod_start.map(|s| s.elapsed());

    let t_extract_start = t_extprod_end.map(|_| std::time::Instant::now());

    let lwe_cts: Vec<_> = column_ciphertexts
        .iter()
        .map(crate::rlwe::RlweCiphertext::sample_extract_coeff0)
        .collect();

    let t_extract_end = t_extract_start.map(|s| s.elapsed());

    let num_columns = lwe_cts.len();

    let t_bpoly_start = t_extract_end.map(|_| std::time::Instant::now());

    let a_ct_tilde: Vec<Poly> = column_ciphertexts
        .iter()
        .map(|rlwe| rlwe.a.clone())
        .collect();

    let mut b_coeffs = vec![0u64; d];
    for (i, lwe) in lwe_cts.iter().enumerate() {
        if i < d {
            b_coeffs[i] = lwe.b;
        }
    }
    let b_poly = Poly::from_coeffs_moduli(b_coeffs, crs.params.moduli());

    let t_bpoly_end = t_bpoly_start.map(|s| s.elapsed());

    let t_packoff_start = t_bpoly_end.map(|_| std::time::Instant::now());

    let precomp = packing_offline(&cache.pack_params, &cache.offline_keys, &a_ct_tilde, &ctx);

    let t_packoff_end = t_packoff_start.map(|s| s.elapsed());

    let t_packonline_start = t_packoff_end.map(|_| std::time::Instant::now());

    // Inlined keys bypass `register_server_side`, so the geometry guard has to run here
    // too or a width mismatch returns a successful response carrying wrong plaintext.
    crate::pir::session::ensure_packing_width_matches(client_packing_keys, &cache.pack_params)?;
    let derived_y_all = if client_packing_keys.y_all.is_empty() {
        if client_packing_keys.y_body.is_empty() {
            return Err(pir_err!(
                "InspiRING packing keys invalid: y_all and y_body are both empty"
            ));
        }
        Some(generate_rotations(
            &cache.pack_params,
            &client_packing_keys.y_body,
        ))
    } else {
        None
    };
    let y_all: &[Vec<Poly>] = derived_y_all
        .as_deref()
        .unwrap_or(&client_packing_keys.y_all);

    // y_all_ntt is `#[serde(skip)]`, so it is populated only in-process; when it is
    // there the fully-NTT path avoids a per-call `to_ntt`. RAVEN_FORCE_PACKING_ONLINE
    // pins the fallback so the wire-format delta can be measured.
    let force_packing_online = std::env::var_os("RAVEN_FORCE_PACKING_ONLINE").is_some();
    let packed = if !force_packing_online
        && !client_packing_keys.y_all_ntt.is_empty()
        && derived_y_all.is_none()
    {
        packing_online_fully_ntt(&precomp, &client_packing_keys.y_all_ntt, &b_poly, &ctx)
    } else {
        packing_online(&precomp, y_all, &b_poly, &ctx)
    };

    let t_packonline_end = t_packonline_start.map(|s| s.elapsed());

    if profile {
        // Microseconds per field.
        if let (Some(e), Some(x), Some(b), Some(po), Some(pn)) = (
            t_extprod_end,
            t_extract_end,
            t_bpoly_end,
            t_packoff_end,
            t_packonline_end,
        ) {
            eprintln!(
                "RAVEN_PROFILE num_cols={} extprod_us={} extract_coeff0_us={} \
                 bpoly_us={} pack_offline_us={} pack_online_us={}",
                num_columns,
                e.as_micros(),
                x.as_micros(),
                b.as_micros(),
                po.as_micros(),
                pn.as_micros()
            );
        }
    }

    Ok(ServerResponse {
        ciphertext: packed,
        column_ciphertexts: vec![],
        packing_mode: Some(PackingMode::Inspiring),
    })
}

/// Holds inlined keys by reference or store-resolved keys by `Arc`, so neither path
/// clones the ~48 KiB payload.
enum PackingKeys<'a> {
    Inline(&'a crate::inspiring::ClientPackingKeys),
    Owned(std::sync::Arc<crate::inspiring::ClientPackingKeys>),
}

impl PackingKeys<'_> {
    fn as_ref(&self) -> &crate::inspiring::ClientPackingKeys {
        match self {
            PackingKeys::Inline(k) => k,
            PackingKeys::Owned(arc) => arc.as_ref(),
        }
    }
}

/// Seeded sibling of [`respond_inspiring_cached_with_session`].
pub fn respond_seeded_inspiring_cached_with_session(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &SeededClientQuery,
    cache: &ServerInspiringCache,
    session_store: Option<&super::session::ServerSessionStore>,
) -> Result<ServerResponse> {
    let expanded = query.expand();
    respond_inspiring_cached_with_session(crs, encoded_db, &expanded, cache, session_store)
}

/// Seeded sibling of [`respond_inspiring`].
pub fn respond_seeded_inspiring(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &SeededClientQuery,
) -> Result<ServerResponse> {
    let expanded = query.expand();
    respond_inspiring(crs, encoded_db, &expanded)
}

/// Seeded sibling of [`respond`].
pub fn respond_seeded(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &SeededClientQuery,
) -> Result<ServerResponse> {
    let expanded = query.expand();
    respond(crs, encoded_db, &expanded)
}

/// Seeded sibling of [`respond_one_packing`]; keeps responses packed without
/// modulus switching.
pub fn respond_seeded_packed(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &SeededClientQuery,
) -> Result<ServerResponse> {
    let expanded = query.expand();
    respond_one_packing(crs, encoded_db, &expanded)
}

/// [`respond`] with the columns walked sequentially, for parallel-vs-serial benchmarks.
pub fn respond_sequential(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &ClientQuery,
) -> Result<ServerResponse> {
    let _d = crs.ring_dim();
    let _q = crs.modulus();
    let delta = crs.params.delta();
    let ctx = crs.params.ntt_context();

    let shard = encoded_db
        .shards
        .iter()
        .find(|s| s.id == query.shard_id)
        .ok_or_else(|| pir_err!("Shard {} not found", query.shard_id))?;
    require_shard_width("respond_sequential", crs, shard)?;

    let rgsw_ntt_rows = rgsw_rows_to_ntt(&query.rgsw_ciphertext, &ctx);
    let rgsw_gadget = &query.rgsw_ciphertext.gadget;
    let mut column_ciphertexts = Vec::with_capacity(shard.polynomials.len());
    for db_poly in &shard.polynomials {
        let rlwe_db = RlweCiphertext::trivial_encrypt(db_poly, delta, &crs.params);
        let rotated = external_product_with_ntt_rgsw(&rlwe_db, &rgsw_ntt_rows, rgsw_gadget, &ctx);
        column_ciphertexts.push(rotated);
    }

    let combined = if column_ciphertexts.len() == 1 {
        column_ciphertexts[0].clone()
    } else {
        column_ciphertexts
            .iter()
            .skip(1)
            .fold(column_ciphertexts[0].clone(), |acc, ct| acc.add(ct))
    };

    Ok(ServerResponse {
        ciphertext: combined,
        column_ciphertexts,
        packing_mode: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{GaussianSampler, Poly};
    use crate::pir::query::query;
    use crate::pir::setup::setup;

    fn test_params() -> crate::params::InspireParams {
        crate::params::InspireParams {
            ring_dim: 256,
            q: 1152921504606830593,
            crt_moduli: vec![1152921504606830593],
            p: 65536,
            sigma: 6.4,
            gadget_base: 1 << 20,
            gadget_len: 3,
            security_level: crate::params::SecurityLevel::Bits128,
        }
    }

    #[test]
    fn test_respond_produces_valid_ciphertext() {
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
        let (_state, client_query) = query(
            &crs,
            target_index,
            &encoded_db.config,
            &rlwe_sk,
            &mut sampler,
        )
        .unwrap();

        let response = respond(&crs, &encoded_db, &client_query);
        assert!(response.is_ok());

        let response = response.unwrap();
        assert_eq!(response.ciphertext.ring_dim(), params.ring_dim);
    }

    #[test]
    fn test_respond_invalid_shard() {
        let params = test_params();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let entry_size = 32;
        let num_entries = params.ring_dim;
        let database: Vec<u8> = (0..(num_entries * entry_size))
            .map(|i| (i % 256) as u8)
            .collect();

        let (crs, encoded_db, rlwe_sk) =
            setup(&params, &database, entry_size, &mut sampler).unwrap();

        let target_index = 0u64;
        let (_, mut client_query) = query(
            &crs,
            target_index,
            &encoded_db.config,
            &rlwe_sk,
            &mut sampler,
        )
        .unwrap();

        client_query.shard_id = 999;

        let response = respond(&crs, &encoded_db, &client_query);
        assert!(response.is_err());
    }

    #[test]
    fn test_ciphertext_addition() {
        let params = test_params();
        let d = params.ring_dim;
        let moduli = params.moduli();

        let a1 = Poly::zero_moduli(d, moduli);
        let mut b1_coeffs = vec![0u64; d];
        b1_coeffs[0] = 100;
        let b1 = Poly::from_coeffs_moduli(b1_coeffs, moduli);
        let ct1 = RlweCiphertext::from_parts(a1, b1);

        let a2 = Poly::zero_moduli(d, moduli);
        let mut b2_coeffs = vec![0u64; d];
        b2_coeffs[0] = 200;
        let b2 = Poly::from_coeffs_moduli(b2_coeffs, moduli);
        let ct2 = RlweCiphertext::from_parts(a2, b2);

        let combined = ct1.add(&ct2);

        assert_eq!(combined.b.coeff(0), 300);
    }

    #[test]
    #[ignore = "tree-packed extract requires gcd(d, p) == 1, which this fixture (d=256, p=65536) \
                violates, so ExtractError::DegreeNotInvertible is the correct outcome and the test \
                can never pass as written. Trigger: refit the fixture to a (d, p) pair with gcd(d, \
                p) == 1, then un-ignore."]
    fn test_respond_one_packing_correctness() {
        use crate::params::InspireVariant;
        use crate::pir::extract_with_variant;

        let params = test_params();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let entry_size = 64;
        let num_entries = params.ring_dim;
        let database: Vec<u8> = (0..(num_entries * entry_size))
            .map(|i| (i % 256) as u8)
            .collect();

        let (crs, encoded_db, rlwe_sk) =
            setup(&params, &database, entry_size, &mut sampler).unwrap();

        for target_index in [0u64, 1, 42] {
            let (state, client_query) = query(
                &crs,
                target_index,
                &encoded_db.config,
                &rlwe_sk,
                &mut sampler,
            )
            .unwrap();

            let response_no_pack = respond(&crs, &encoded_db, &client_query).unwrap();
            let extracted_no_pack =
                crate::pir::extract(&crs, &state, &response_no_pack, entry_size).unwrap();

            let expected_start = (target_index as usize) * entry_size;
            let expected = &database[expected_start..expected_start + entry_size];

            assert_eq!(
                extracted_no_pack.as_slice(),
                expected,
                "NoPacking should work for index {target_index}"
            );

            let response_one_pack = respond_one_packing(&crs, &encoded_db, &client_query).unwrap();
            let extracted_one_pack = extract_with_variant(
                &crs,
                &state,
                &response_one_pack,
                entry_size,
                InspireVariant::OnePacking,
            )
            .unwrap();

            assert_eq!(
                extracted_one_pack.as_slice(),
                expected,
                "OnePacking should return the entry bytes for index {target_index}"
            );
        }
    }

    #[test]
    #[ignore = "tree-packed extract requires gcd(d, p) == 1, which this fixture (d=256, p=65536) \
                violates, so ExtractError::DegreeNotInvertible is the correct outcome and the test \
                can never pass as written. Trigger: refit the fixture to a (d, p) pair with gcd(d, \
                p) == 1, then un-ignore."]
    fn test_respond_one_packing_small_values() {
        use crate::params::InspireVariant;
        use crate::pir::extract_with_variant;

        let params = test_params();
        let d = params.ring_dim;
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        // d * column_value must stay under p, so the high byte is pinned to 0.
        let entry_size = 2;
        let num_entries = d;

        let database: Vec<u8> = (0..num_entries)
            .flat_map(|i| {
                let low_byte = (i % 256) as u8;
                let high_byte = 0u8;
                vec![low_byte, high_byte]
            })
            .collect();

        let (crs, encoded_db, rlwe_sk) =
            setup(&params, &database, entry_size, &mut sampler).unwrap();

        for target_index in [0u64, 1, 42, 100] {
            let (state, client_query) = query(
                &crs,
                target_index,
                &encoded_db.config,
                &rlwe_sk,
                &mut sampler,
            )
            .unwrap();

            let response_no_pack = respond(&crs, &encoded_db, &client_query).unwrap();
            let extracted_no_pack =
                crate::pir::extract(&crs, &state, &response_no_pack, entry_size).unwrap();

            let response_one_pack = respond_one_packing(&crs, &encoded_db, &client_query).unwrap();
            let extracted_one_pack = extract_with_variant(
                &crs,
                &state,
                &response_one_pack,
                entry_size,
                InspireVariant::OnePacking,
            )
            .unwrap();

            let expected_start = (target_index as usize) * entry_size;
            let expected = &database[expected_start..expected_start + entry_size];

            assert_eq!(
                extracted_no_pack.as_slice(),
                expected,
                "NoPacking should work for index {target_index}"
            );
            assert_eq!(
                extracted_one_pack.as_slice(),
                expected,
                "OnePacking should work with small values for index {target_index}"
            );
        }
    }

    #[test]
    fn test_inspire_sizes_production() {
        use crate::pir::query::query_seeded;

        let params = crate::params::InspireParams {
            ring_dim: 2048,
            q: 1152921504606830593,
            crt_moduli: vec![1152921504606830593],
            p: 65536,
            sigma: 6.4,
            gadget_base: 1 << 20,
            gadget_len: 3,
            security_level: crate::params::SecurityLevel::Bits128,
        };
        let d = params.ring_dim;
        let entry_size = 32;
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let num_entries = d;
        let database: Vec<u8> = (0..(num_entries * entry_size))
            .map(|i| (i % 256) as u8)
            .collect();

        let (crs, encoded_db, rlwe_sk) =
            setup(&params, &database, entry_size, &mut sampler).unwrap();

        let target_index = 42u64;

        let (_state, full_query) = query(
            &crs,
            target_index,
            &encoded_db.config,
            &rlwe_sk,
            &mut sampler,
        )
        .unwrap();
        let (_state, seeded_query) = query_seeded(
            &crs,
            target_index,
            &encoded_db.config,
            &rlwe_sk,
            &mut sampler,
        )
        .unwrap();

        let response_no_pack = respond(&crs, &encoded_db, &full_query).unwrap();
        let response_one_pack = respond_one_packing(&crs, &encoded_db, &full_query).unwrap();

        let query_full_bytes = bincode::serialize(&full_query).unwrap();
        let query_seeded_bytes = bincode::serialize(&seeded_query).unwrap();
        let resp_0_bytes = response_no_pack.to_binary().unwrap();
        let resp_1_bytes = response_one_pack.to_binary().unwrap();

        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  InsPIRe Size Comparison (d={d}, entry={entry_size}B, 16 columns)   ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  QUERY SIZES                                                 ║");
        println!("╟──────────────────────────────────────────────────────────────╢");
        println!(
            "║  Full query:      {:>8} bytes ({:>6.1} KB)                 ║",
            query_full_bytes.len(),
            query_full_bytes.len() as f64 / 1024.0
        );
        println!(
            "║  Seeded query:    {:>8} bytes ({:>6.1} KB)  [{:.0}% of full] ║",
            query_seeded_bytes.len(),
            query_seeded_bytes.len() as f64 / 1024.0,
            query_seeded_bytes.len() as f64 / query_full_bytes.len() as f64 * 100.0
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  RESPONSE SIZES                                              ║");
        println!("╟──────────────────────────────────────────────────────────────╢");
        println!(
            "║  NoPacking (^0):  {:>8} bytes ({:>6.1} KB)                 ║",
            resp_0_bytes.len(),
            resp_0_bytes.len() as f64 / 1024.0
        );
        println!(
            "║  OnePacking (^1): {:>8} bytes ({:>6.1} KB)  [{:.1}x smaller]  ║",
            resp_1_bytes.len(),
            resp_1_bytes.len() as f64 / 1024.0,
            resp_0_bytes.len() as f64 / resp_1_bytes.len() as f64
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  TOTAL ROUNDTRIP (Query + Response)                          ║");
        println!("╟──────────────────────────────────────────────────────────────╢");

        let total_0 = query_full_bytes.len() + resp_0_bytes.len();
        let total_1 = query_full_bytes.len() + resp_1_bytes.len();
        let total_2 = query_seeded_bytes.len() + resp_1_bytes.len();

        println!(
            "║  InsPIRe^0 (full+nopack):   {:>8} bytes ({:>6.1} KB)        ║",
            total_0,
            total_0 as f64 / 1024.0
        );
        println!(
            "║  InsPIRe^1 (full+packed):   {:>8} bytes ({:>6.1} KB)        ║",
            total_1,
            total_1 as f64 / 1024.0
        );
        println!(
            "║  InsPIRe^2 (seeded+packed): {:>8} bytes ({:>6.1} KB)        ║",
            total_2,
            total_2 as f64 / 1024.0
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  BANDWIDTH SAVINGS vs InsPIRe^0                              ║");
        println!("╟──────────────────────────────────────────────────────────────╢");
        println!(
            "║  InsPIRe^1: {:.1}x reduction                                   ║",
            total_0 as f64 / total_1 as f64
        );
        println!(
            "║  InsPIRe^2: {:.1}x reduction                                   ║",
            total_0 as f64 / total_2 as f64
        );
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
}
