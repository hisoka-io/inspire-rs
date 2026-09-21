//! PIR.Respond: multiply encrypted `Delta*X^(-k)` by public database polynomials.

use crate::par_prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::inspiring::{packing_online, packing_online_fully_ntt};
use crate::math::poly::{pack_coefficients_tight, unpack_coefficients_tight, TightBytes};
use crate::math::{NttContext, Poly};
use crate::params::InspireVariant;
use crate::rlwe::RlweCiphertext;

use super::error::{pir_err, Result};
use super::query::{ClientQuery, PackingMode, SeededClientQuery};
use super::setup::{EncodedDatabase, ServerCrs, ShardData};

/// Server response: the packed ciphertext, or one ciphertext per column.
#[derive(Clone, Debug)]
pub struct ServerResponse {
    /// Packed result, or the sum of the column ciphertexts when unpacked.
    pub ciphertext: RlweCiphertext,
    /// Populated only on the unpacked path.
    pub column_ciphertexts: Vec<RlweCiphertext>,
    /// Lets the extractor tell unscaled InspiRING output from d-scaled tree output.
    ///
    /// Never `skip_serializing_if`: bincode is positional, so an omitted field
    /// shifts every later read.
    pub packing_mode: Option<PackingMode>,
    /// Number of leading packed plaintext coefficients whose `b` terms cross the wire.
    /// The full `a` remains; eprint 2025/1352 section 3.3, p.12. `None` keeps the
    /// full ciphertext for unpacked responses.
    pub packed_coefficients: Option<u32>,
}

#[derive(Deserialize)]
struct ServerResponseWire {
    ciphertext: ResponseCiphertextWire,
    column_ciphertexts: Vec<RlweCiphertext>,
    packing_mode: Option<PackingMode>,
}

#[derive(Deserialize)]
enum ResponseCiphertextWire {
    Full(RlweCiphertext),
    Packed {
        a: Poly,
        b_prefix: Vec<u64>,
        retained: u32,
    },
}

#[derive(Deserialize)]
struct TightServerResponseWire {
    ciphertext: TightResponseCiphertextWire,
    column_ciphertexts: Vec<RlweCiphertext>,
    packing_mode: Option<PackingMode>,
}

#[derive(Deserialize)]
enum TightResponseCiphertextWire {
    Full(RlweCiphertext),
    Packed {
        a: Poly,
        b_prefix: TightBytes,
        retained: u32,
    },
}

#[derive(Serialize)]
struct ServerResponseWireRef<'a> {
    ciphertext: ResponseCiphertextWireRef<'a>,
    column_ciphertexts: &'a [RlweCiphertext],
    packing_mode: Option<PackingMode>,
}

#[derive(Serialize)]
enum ResponseCiphertextWireRef<'a> {
    Full(&'a RlweCiphertext),
    Packed {
        a: &'a Poly,
        b_prefix: Vec<u64>,
        retained: u32,
    },
}

#[derive(Serialize)]
struct TightServerResponseWireRef<'a> {
    ciphertext: TightResponseCiphertextWireRef<'a>,
    column_ciphertexts: &'a [RlweCiphertext],
    packing_mode: Option<PackingMode>,
}

#[derive(Serialize)]
enum TightResponseCiphertextWireRef<'a> {
    Full(&'a RlweCiphertext),
    Packed {
        a: &'a Poly,
        b_prefix: TightBytes,
        retained: u32,
    },
}

impl Serialize for ServerResponse {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::Error;

        if !serializer.is_human_readable() {
            let ciphertext = match self.packed_coefficients {
                None => {
                    if self.packing_mode.is_some() {
                        return Err(S::Error::custom(
                            "packed response requires an exact coefficient prefix",
                        ));
                    }
                    TightResponseCiphertextWireRef::Full(&self.ciphertext)
                }
                Some(retained) => {
                    if !self.column_ciphertexts.is_empty() || self.packing_mode.is_none() {
                        return Err(S::Error::custom(
                            "packed response prefix requires a packed mode and no column ciphertexts",
                        ));
                    }
                    let retained = usize::try_from(retained).map_err(|_| {
                        S::Error::custom("packed coefficient count does not fit usize")
                    })?;
                    let dim = self.ciphertext.ring_dim();
                    if retained == 0 || retained > dim {
                        return Err(S::Error::custom(format!(
                            "packed coefficient count {retained} outside 1..={dim}"
                        )));
                    }
                    let mut b_prefix = Vec::with_capacity(
                        retained
                            .checked_mul(self.ciphertext.b.crt_count())
                            .ok_or_else(|| S::Error::custom("packed coefficient count overflow"))?,
                    );
                    for limb in 0..self.ciphertext.b.crt_count() {
                        let start = limb.checked_mul(dim).ok_or_else(|| {
                            S::Error::custom("packed coefficient offset overflow")
                        })?;
                        let end = start
                            .checked_add(retained)
                            .ok_or_else(|| S::Error::custom("packed coefficient end overflow"))?;
                        b_prefix.extend_from_slice(
                            self.ciphertext.b.coeffs().get(start..end).ok_or_else(|| {
                                S::Error::custom("packed b polynomial shape mismatch")
                            })?,
                        );
                    }
                    let b_prefix =
                        pack_coefficients_tight(&b_prefix, self.ciphertext.b.moduli(), retained)
                            .map_err(S::Error::custom)?;
                    TightResponseCiphertextWireRef::Packed {
                        a: &self.ciphertext.a,
                        b_prefix: b_prefix.into(),
                        retained: u32::try_from(retained).map_err(|_| {
                            S::Error::custom("packed coefficient count exceeds u32")
                        })?,
                    }
                }
            };
            return TightServerResponseWireRef {
                ciphertext,
                column_ciphertexts: &self.column_ciphertexts,
                packing_mode: self.packing_mode,
            }
            .serialize(serializer);
        }

        let ciphertext = match self.packed_coefficients {
            None => {
                if self.packing_mode.is_some() {
                    return Err(S::Error::custom(
                        "packed response requires an exact coefficient prefix",
                    ));
                }
                ResponseCiphertextWireRef::Full(&self.ciphertext)
            }
            Some(retained) => {
                if !self.column_ciphertexts.is_empty() || self.packing_mode.is_none() {
                    return Err(S::Error::custom(
                        "packed response prefix requires a packed mode and no column ciphertexts",
                    ));
                }
                let retained = usize::try_from(retained)
                    .map_err(|_| S::Error::custom("packed coefficient count does not fit usize"))?;
                let dim = self.ciphertext.ring_dim();
                if retained == 0 || retained > dim {
                    return Err(S::Error::custom(format!(
                        "packed coefficient count {retained} outside 1..={dim}"
                    )));
                }
                let crt_count = self.ciphertext.b.crt_count();
                let prefix_len = retained
                    .checked_mul(crt_count)
                    .ok_or_else(|| S::Error::custom("packed coefficient count overflow"))?;
                let mut b_prefix = Vec::with_capacity(prefix_len);
                for limb in 0..crt_count {
                    let start = limb
                        .checked_mul(dim)
                        .ok_or_else(|| S::Error::custom("packed coefficient offset overflow"))?;
                    let end = start
                        .checked_add(retained)
                        .ok_or_else(|| S::Error::custom("packed coefficient end overflow"))?;
                    let limb_prefix =
                        self.ciphertext.b.coeffs().get(start..end).ok_or_else(|| {
                            S::Error::custom("packed b polynomial shape mismatch")
                        })?;
                    b_prefix.extend_from_slice(limb_prefix);
                }
                ResponseCiphertextWireRef::Packed {
                    a: &self.ciphertext.a,
                    b_prefix,
                    retained: u32::try_from(retained)
                        .map_err(|_| S::Error::custom("packed coefficient count exceeds u32"))?,
                }
            }
        };
        ServerResponseWireRef {
            ciphertext,
            column_ciphertexts: &self.column_ciphertexts,
            packing_mode: self.packing_mode,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ServerResponse {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        use serde::de::Error;

        let wire = if deserializer.is_human_readable() {
            ServerResponseWire::deserialize(deserializer)?
        } else {
            let tight = TightServerResponseWire::deserialize(deserializer)?;
            let ciphertext = match tight.ciphertext {
                TightResponseCiphertextWire::Full(ciphertext) => {
                    ResponseCiphertextWire::Full(ciphertext)
                }
                TightResponseCiphertextWire::Packed {
                    a,
                    b_prefix,
                    retained,
                } => {
                    let retained_usize = usize::try_from(retained).map_err(|_| {
                        D::Error::custom("packed coefficient count does not fit usize")
                    })?;
                    let b_prefix =
                        unpack_coefficients_tight(&b_prefix.into_vec(), a.moduli(), retained_usize)
                            .map_err(D::Error::custom)?;
                    ResponseCiphertextWire::Packed {
                        a,
                        b_prefix,
                        retained,
                    }
                }
            };
            ServerResponseWire {
                ciphertext,
                column_ciphertexts: tight.column_ciphertexts,
                packing_mode: tight.packing_mode,
            }
        };
        let (ciphertext, packed_coefficients) = match wire.ciphertext {
            ResponseCiphertextWire::Full(ciphertext) => {
                if wire.packing_mode.is_some() {
                    return Err(D::Error::custom(
                        "packed response requires an exact coefficient prefix",
                    ));
                }
                for (component, poly) in [("a", &ciphertext.a), ("b", &ciphertext.b)] {
                    if poly.is_ntt() {
                        return Err(D::Error::custom(format!(
                            "coefficient-domain full response {component} required; wire \
                             ciphertext declares NTT domain"
                        )));
                    }
                }
                for (column_index, column) in wire.column_ciphertexts.iter().enumerate() {
                    for (component, poly) in [("a", &column.a), ("b", &column.b)] {
                        if poly.is_ntt() {
                            return Err(D::Error::custom(format!(
                                "coefficient-domain column ciphertext[{column_index}] \
                                 {component} required; wire response declares NTT domain"
                            )));
                        }
                    }
                }
                (ciphertext, None)
            }
            ResponseCiphertextWire::Packed {
                a,
                b_prefix,
                retained,
            } => {
                if !wire.column_ciphertexts.is_empty() || wire.packing_mode.is_none() {
                    return Err(D::Error::custom(
                        "packed response prefix requires a packed mode and no column ciphertexts",
                    ));
                }
                if a.is_ntt() {
                    return Err(D::Error::custom(
                        "packed response prefix requires coefficient-domain ciphertext",
                    ));
                }
                let retained_usize = usize::try_from(retained)
                    .map_err(|_| D::Error::custom("packed coefficient count does not fit usize"))?;
                let dim = a.dimension();
                if retained_usize == 0 || retained_usize > dim {
                    return Err(D::Error::custom(format!(
                        "packed coefficient count {retained_usize} outside 1..={dim}"
                    )));
                }
                let crt_count = a.crt_count();
                let expected_prefix = retained_usize
                    .checked_mul(crt_count)
                    .ok_or_else(|| D::Error::custom("packed coefficient count overflow"))?;
                if b_prefix.len() != expected_prefix {
                    return Err(D::Error::custom(format!(
                        "packed b prefix has {} coefficients, expected {expected_prefix}",
                        b_prefix.len()
                    )));
                }
                let full_len = dim
                    .checked_mul(crt_count)
                    .ok_or_else(|| D::Error::custom("packed b polynomial size overflow"))?;
                let mut b_coeffs = vec![0u64; full_len];
                for limb in 0..crt_count {
                    let prefix_start = limb
                        .checked_mul(retained_usize)
                        .ok_or_else(|| D::Error::custom("packed b prefix offset overflow"))?;
                    let prefix_end = prefix_start
                        .checked_add(retained_usize)
                        .ok_or_else(|| D::Error::custom("packed b prefix end overflow"))?;
                    let output_start = limb
                        .checked_mul(dim)
                        .ok_or_else(|| D::Error::custom("packed b output offset overflow"))?;
                    let output_end = output_start
                        .checked_add(retained_usize)
                        .ok_or_else(|| D::Error::custom("packed b output end overflow"))?;
                    let prefix = b_prefix
                        .get(prefix_start..prefix_end)
                        .ok_or_else(|| D::Error::custom("packed b prefix shape mismatch"))?;
                    let output = b_coeffs
                        .get_mut(output_start..output_end)
                        .ok_or_else(|| D::Error::custom("packed b output shape mismatch"))?;
                    output.copy_from_slice(prefix);
                }
                let b = Poly::from_crt_coeffs_reduced(b_coeffs, a.moduli());
                (RlweCiphertext::from_parts(a, b), Some(retained))
            }
        };
        Ok(Self {
            ciphertext,
            column_ciphertexts: wire.column_ciphertexts,
            packing_mode: wire.packing_mode,
            packed_coefficients,
        })
    }
}

impl ServerResponse {
    /// Serialize to bincode.
    pub fn to_binary(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| pir_err!("bincode serialize failed: {}", e))
    }

    /// Deserialize from bincode.
    pub fn from_binary(bytes: &[u8]) -> Result<Self> {
        use bincode::Options;

        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize(bytes)
            .map_err(|e| pir_err!("bincode deserialize failed: {}", e))
    }
}

fn packed_coefficient_count(count: usize) -> Result<Option<u32>> {
    let count = u32::try_from(count)
        .map_err(|_| pir_err!("packed response coefficient count {count} exceeds u32"))?;
    Ok(Some(count))
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

/// The rows a query addresses are the rows the encoder placed only when a shard holds
/// exactly `ring_dim` of them; the same rule the client and the encoder enforce.
fn require_shard_geometry(
    operation: &'static str,
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
) -> Result<()> {
    encoded_db
        .config
        .validate_for_params(&crs.params)
        .map_err(|cause| {
            pir_err!(
                "{operation}: encoded database declares {} rows per shard at ring_dim {}; \
                 refusing to serve rows the client did not address: {cause}",
                encoded_db.config.entries_per_shard(),
                crs.params.ring_dim,
            )
        })
}

fn require_dense_shard<'a>(
    operation: &'static str,
    encoded_db: &'a EncodedDatabase,
    shard_id: u32,
) -> Result<&'a ShardData> {
    let position = usize::try_from(shard_id)
        .map_err(|_| pir_err!("{operation}: shard id {shard_id} does not fit a vector index"))?;
    let shard = encoded_db.shards.get(position).ok_or_else(|| {
        pir_err!(
            "{operation}: shard {shard_id} not found at dense position {position}; encoded database has {} shards",
            encoded_db.shards.len(),
        )
    })?;
    if shard.id != shard_id {
        return Err(pir_err!(
            "{operation}: shard vector position {position} holds shard id {}, expected {shard_id}; refusing non-dense encoded state",
            shard.id,
        ));
    }
    Ok(shard)
}

fn require_crs_rgsw_gadget(
    operation: &'static str,
    crs: &ServerCrs,
    query: &ClientQuery,
) -> Result<()> {
    require_rgsw_shape(
        operation,
        crs,
        &query.rgsw_ciphertext.gadget,
        query.rgsw_ciphertext.rows.len(),
    )?;
    for (row_index, row) in query.rgsw_ciphertext.rows.iter().enumerate() {
        for (component, poly) in [("a", &row.a), ("b", &row.b)] {
            require_rgsw_polynomial_shape(operation, crs, poly, row_index, component)?;
            if poly.is_ntt() {
                return Err(pir_err!(
                    "{operation}: coefficient-domain RGSW row {row_index} {component} required; \
                     client query declares NTT domain. Rebuild the query from the current CRS"
                ));
            }
        }
    }
    if let Some(keys) = query.inspiring_packing_keys.as_ref() {
        crate::pir::session::require_wire_packing_body_shape(
            keys,
            crs.params.ring_dim,
            crs.params.moduli(),
        )?;
    }
    Ok(())
}

fn require_seeded_crs_rgsw_gadget(
    operation: &'static str,
    crs: &ServerCrs,
    query: &SeededClientQuery,
) -> Result<()> {
    require_rgsw_shape(
        operation,
        crs,
        &query.rgsw_ciphertext.gadget,
        query.rgsw_ciphertext.rows.len(),
    )?;
    for (row_index, row) in query.rgsw_ciphertext.rows.iter().enumerate() {
        require_rgsw_polynomial_shape(operation, crs, &row.b, row_index, "b")?;
        if row.b.is_ntt() {
            return Err(pir_err!(
                "{operation}: coefficient-domain RGSW row {row_index} b required; \
                 client query declares NTT domain. Rebuild the query from the current CRS"
            ));
        }
    }
    if let Some(keys) = query.inspiring_packing_keys.as_ref() {
        crate::pir::session::require_wire_packing_body_shape(
            keys,
            crs.params.ring_dim,
            crs.params.moduli(),
        )?;
    }
    Ok(())
}

fn require_rgsw_polynomial_shape(
    operation: &'static str,
    crs: &ServerCrs,
    poly: &Poly,
    row_index: usize,
    component: &str,
) -> Result<()> {
    if poly.dimension() != crs.params.ring_dim {
        return Err(pir_err!(
            "{operation}: RGSW row {row_index} {component} dimension {}, expected {} from the CRS; \
             rebuild the query with matching parameters",
            poly.dimension(),
            crs.params.ring_dim
        ));
    }
    if poly.moduli() != crs.params.moduli() {
        return Err(pir_err!(
            "{operation}: RGSW row {row_index} {component} moduli {:?}, expected {:?} from the CRS; \
             rebuild the query with matching parameters",
            poly.moduli(),
            crs.params.moduli()
        ));
    }
    Ok(())
}

fn require_rgsw_shape(
    operation: &'static str,
    crs: &ServerCrs,
    got: &crate::rgsw::GadgetVector,
    got_rows: usize,
) -> Result<()> {
    let expected = &crs.params;
    if got.len != 1 || got.base != expected.gadget_base || got.q != expected.q {
        return Err(pir_err!(
            "{operation}: RGSW gadget mismatch: got len={} base={} q={}, expected len={} base={} \
             q={} from the CRS; refusing client-supplied decomposition parameters",
            got.len,
            got.base,
            got.q,
            1,
            expected.gadget_base,
            expected.q,
        ));
    }
    if got_rows != 1 {
        return Err(pir_err!(
            "{operation}: one-sided RGSW row-count mismatch: got {got_rows}, expected {} from \
             the CRS gadget length; refusing malformed query rows before external product",
            1
        ));
    }
    Ok(())
}

fn plaintext_multiply_columns(
    query: &ClientQuery,
    shard: &ShardData,
    ctx: &NttContext,
) -> Result<Vec<RlweCiphertext>> {
    let folding_ciphertext = query
        .rgsw_ciphertext
        .rows
        .first()
        .ok_or_else(|| pir_err!("plaintext-multiply query has no ciphertext row"))?;
    Ok(shard
        .polynomials
        .par_iter()
        .map(|db_poly| folding_ciphertext.poly_mul(db_poly, ctx))
        .collect())
}

/// Respond with one RLWE ciphertext per column, in parallel.
pub fn respond(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &ClientQuery,
) -> Result<ServerResponse> {
    require_crs_rgsw_gadget("respond", crs, query)?;
    let ctx = crs.params.ntt_context();

    require_shard_geometry("respond", crs, encoded_db)?;
    let shard = require_dense_shard("respond", encoded_db, query.shard_id)?;
    require_shard_width("respond", crs, shard)?;

    let column_ciphertexts = plaintext_multiply_columns(query, shard, &ctx)?;

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
        packed_coefficients: None,
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
    require_seeded_crs_rgsw_gadget("respond_seeded_with_variant", crs, query)?;
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

    require_crs_rgsw_gadget("respond_one_packing", crs, query)?;
    let _d = crs.ring_dim();
    let _q = crs.modulus();
    let ctx = crs.params.ntt_context();

    require_shard_geometry("respond_one_packing", crs, encoded_db)?;
    let shard = require_dense_shard("respond_one_packing", encoded_db, query.shard_id)?;
    require_shard_width("respond_one_packing", crs, shard)?;

    let column_ciphertexts = plaintext_multiply_columns(query, shard, &ctx)?;

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
        packed_coefficients: packed_coefficient_count(lwe_cts.len())?,
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

    require_crs_rgsw_gadget("respond_inspiring", crs, query)?;
    let d = crs.ring_dim();
    let ctx = crs.params.ntt_context();

    let client_packing_keys = query
        .inspiring_packing_keys
        .as_ref()
        .ok_or_else(|| pir_err!("InspiRING client packing keys missing from query"))?;

    require_shard_geometry("respond_inspiring", crs, encoded_db)?;
    let shard = require_dense_shard("respond_inspiring", encoded_db, query.shard_id)?;
    require_shard_width("respond_inspiring", crs, shard)?;

    let column_ciphertexts = plaintext_multiply_columns(query, shard, &ctx)?;

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
        packed_coefficients: packed_coefficient_count(num_columns)?,
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
        || pack_params.gadget.len != crs.params.packing_gadget_len
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

    require_crs_rgsw_gadget("respond_inspiring_cached", crs, query)?;
    let d = crs.ring_dim();
    let ctx = crs.params.ntt_context();

    let client_packing_keys = query
        .inspiring_packing_keys
        .as_ref()
        .ok_or_else(|| pir_err!("InspiRING client packing keys missing from query"))?;

    require_shard_geometry("respond_inspiring_cached", crs, encoded_db)?;
    let shard = require_dense_shard("respond_inspiring_cached", encoded_db, query.shard_id)?;
    require_shard_width("respond_inspiring_cached", crs, shard)?;

    let column_ciphertexts = plaintext_multiply_columns(query, shard, &ctx)?;

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
        packed_coefficients: packed_coefficient_count(cache.pack_params.num_to_pack)?,
    })
}

/// Seeded sibling of [`respond_inspiring_cached`].
pub fn respond_seeded_inspiring_cached(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &SeededClientQuery,
    cache: &ServerInspiringCache,
) -> Result<ServerResponse> {
    require_seeded_crs_rgsw_gadget("respond_seeded_inspiring_cached", crs, query)?;
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

    require_crs_rgsw_gadget("respond_inspiring_cached_with_session", crs, query)?;
    let d = crs.ring_dim();
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

    require_shard_geometry("respond_inspiring_cached_with_session", crs, encoded_db)?;
    let shard = require_dense_shard(
        "respond_inspiring_cached_with_session",
        encoded_db,
        query.shard_id,
    )?;
    require_shard_width("respond_inspiring_cached_with_session", crs, shard)?;

    // Set RAVEN_PROFILE_RESPOND to any value for a per-region stderr breakdown.
    let profile = std::env::var_os("RAVEN_PROFILE_RESPOND").is_some();
    let t_extprod_start = if profile {
        Some(std::time::Instant::now())
    } else {
        None
    };

    let column_ciphertexts = plaintext_multiply_columns(query, shard, &ctx)?;

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
        packed_coefficients: packed_coefficient_count(num_columns)?,
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
    require_seeded_crs_rgsw_gadget("respond_seeded_inspiring_cached_with_session", crs, query)?;
    let expanded = query.expand();
    respond_inspiring_cached_with_session(crs, encoded_db, &expanded, cache, session_store)
}

/// Seeded sibling of [`respond_inspiring`].
pub fn respond_seeded_inspiring(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &SeededClientQuery,
) -> Result<ServerResponse> {
    require_seeded_crs_rgsw_gadget("respond_seeded_inspiring", crs, query)?;
    let expanded = query.expand();
    respond_inspiring(crs, encoded_db, &expanded)
}

/// Seeded sibling of [`respond`].
pub fn respond_seeded(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &SeededClientQuery,
) -> Result<ServerResponse> {
    require_seeded_crs_rgsw_gadget("respond_seeded", crs, query)?;
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
    require_seeded_crs_rgsw_gadget("respond_seeded_packed", crs, query)?;
    let expanded = query.expand();
    respond_one_packing(crs, encoded_db, &expanded)
}

/// [`respond`] with the columns walked sequentially, for parallel-vs-serial benchmarks.
pub fn respond_sequential(
    crs: &ServerCrs,
    encoded_db: &EncodedDatabase,
    query: &ClientQuery,
) -> Result<ServerResponse> {
    require_crs_rgsw_gadget("respond_sequential", crs, query)?;
    let _d = crs.ring_dim();
    let _q = crs.modulus();
    let ctx = crs.params.ntt_context();

    require_shard_geometry("respond_sequential", crs, encoded_db)?;
    let shard = require_dense_shard("respond_sequential", encoded_db, query.shard_id)?;
    require_shard_width("respond_sequential", crs, shard)?;

    let mut column_ciphertexts = Vec::with_capacity(shard.polynomials.len());
    let folding_ciphertext = query
        .rgsw_ciphertext
        .rows
        .first()
        .ok_or_else(|| pir_err!("plaintext-multiply query has no ciphertext row"))?;
    for db_poly in &shard.polynomials {
        column_ciphertexts.push(folding_ciphertext.poly_mul(db_poly, &ctx));
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
        packed_coefficients: None,
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
            query_gadget_len: 3,
            packing_gadget_len: 3,
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
            query_gadget_len: 3,
            packing_gadget_len: 3,
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
