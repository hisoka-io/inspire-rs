//! Step 1 of the InspiRING offline phase, folded over one shard's columns ahead of
//! the queries.
//!
//! A responder hands [`packing_offline`] the vectors `a_j = a * h_j`: the query's `a`
//! times column j of the shard. Step 1 is linear in those vectors, so
//! `R[i] = tau_{g^i}(a) * S_i`, where `S_i` is step 1 applied to the columns
//! themselves. Holding the gamma `S_i` turns gamma^2 pointwise products per query
//! into gamma, and the b side shrinks to coefficient 0 of each `b * h_j`, one inner
//! product per column. The backward recursion is the one [`packing_offline`] runs.
//!
//! A store answers for the columns it was folded from and for no others. It keeps a
//! keyed tag per column, and a respond recomputes every tag from the columns it is
//! about to serve, in the pass that folds b, before any packing work.
//!
//! [`packing_offline`]: super::inspiring2::packing_offline

use rand::RngCore;

use super::inspiring2::{
    fold_rotations_backward, rotation_sums, PackParams, PackingKeyBody, PrecompInsPIR,
};
use crate::math::gaussian::os_seeded_chacha;
use crate::math::{NttContext, Poly};
use crate::par_prelude::*;

/// 2^61 - 1. Tags live in this field whatever the ring modulus, so a small CRT limb
/// does not weaken them.
const TAG_PRIME: u64 = (1 << 61) - 1;

/// A tag term is below 2^94, so a u128 sums this many without wrapping.
const MAX_TAGGED_WORDS: usize = 1 << 30;

/// The columns' step-1 sums `S_i`, with the tags that tie them to those columns.
///
/// Built by [`ShardPackingStore::build`] and read only through a check of the live
/// columns, so a store whose shard has since changed is refused, not served.
#[derive(Clone)]
pub struct ShardPackingStore {
    ring_dim: usize,
    moduli: Vec<u64>,
    num_to_pack: usize,
    generator: usize,
    /// `S_i` in NTT form, i in `0..gamma`.
    column_sums_ntt: Vec<Poly>,
    /// Two key words per column word, drawn once per store and never sent anywhere:
    /// someone who writes rows cannot aim a change at a key they do not see.
    tag_key: Vec<u64>,
    column_tags: Vec<u64>,
}

impl std::fmt::Debug for ShardPackingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardPackingStore")
            .field("ring_dim", &self.ring_dim)
            .field("moduli", &self.moduli)
            .field("num_to_pack", &self.num_to_pack)
            .field("payload_bytes", &self.payload_bytes())
            .finish_non_exhaustive()
    }
}

impl ShardPackingStore {
    /// Fold step 1 over `columns`, the shard's coefficient-form column polynomials.
    ///
    /// Costs gamma forward NTTs and gamma^2 pointwise products; the result holds
    /// gamma polynomials, as many words as the columns themselves.
    ///
    /// ```
    /// use raven_inspire::inspiring::{PackParams, ShardPackingStore};
    /// use raven_inspire::math::Poly;
    /// use raven_inspire::params::InspireParams;
    ///
    /// let params = InspireParams::secure_128_d2048();
    /// let pack_params = PackParams::try_new(&params, 16)?;
    /// let columns = vec![Poly::zero_moduli(params.ring_dim, params.moduli()); 16];
    /// let store = ShardPackingStore::build(&pack_params, &columns, &params.ntt_context())?;
    /// assert_eq!(store.payload_bytes(), 16 * 2048 * 8);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The columns are not gamma coefficient-form polynomials on the pack-params
    /// ring, the pack params lack a table step 1 reads, or the platform entropy
    /// source fails.
    pub fn build(
        pack_params: &PackParams,
        columns: &[Poly],
        ctx: &NttContext,
    ) -> Result<Self, String> {
        let n = pack_params.ring_dim;
        let gamma = pack_params.num_to_pack;
        let moduli = pack_params.moduli.as_slice();
        if ctx.dimension() != n || ctx.moduli() != moduli {
            return Err(format!(
                "packing store refused: NTT context is for ring_dim {} moduli {:?}, pack \
                 params are for ring_dim {n} moduli {moduli:?}",
                ctx.dimension(),
                ctx.moduli()
            ));
        }
        let words = n
            .checked_mul(moduli.len())
            .filter(|&words| n != 0 && words <= MAX_TAGGED_WORDS)
            .ok_or_else(|| {
                format!(
                    "packing store refused: ring_dim {n} over {} CRT limbs is outside the \
                     1..={MAX_TAGGED_WORDS} words a column tag covers",
                    moduli.len()
                )
            })?;
        // Step 1 indexes these without a bounds error of its own.
        if gamma == 0
            || pack_params.gen_pows.len() != n
            || pack_params.monomials_ntt.len() != n
            || pack_params.neg_monomials_ntt.len() != n
            || pack_params.rotation_tables(gamma).is_none()
        {
            return Err(format!(
                "packing store refused: pack params for gamma {gamma} at ring_dim {n} do \
                 not hold {n} generator powers, {n} monomials of each sign and a full \
                 automorphism table per rotation"
            ));
        }
        require_served_columns(columns, n, moduli, gamma)?;

        let mut rng = os_seeded_chacha("ShardPackingStore::build").map_err(|e| e.to_string())?;
        let tag_key: Vec<u64> = (0..2 * words)
            .map(|_| loop {
                let candidate = rng.next_u64() >> 3;
                if candidate < TAG_PRIME {
                    break candidate;
                }
            })
            .collect();
        let column_tags = columns
            .iter()
            .map(|column| column_tag(column.coeffs(), &tag_key))
            .collect();

        let columns_ntt: Vec<Poly> = columns
            .par_iter()
            .map(|column| column.to_ntt_new(ctx))
            .collect();
        let column_sums_ntt = rotation_sums(pack_params, &columns_ntt, ctx);

        Ok(Self {
            ring_dim: n,
            moduli: moduli.to_vec(),
            num_to_pack: gamma,
            generator: pack_params.generator,
            column_sums_ntt,
            tag_key,
            column_tags,
        })
    }

    /// gamma, the column count this store packs.
    pub fn num_to_pack(&self) -> usize {
        self.num_to_pack
    }

    /// Bytes of step-1 coefficients held: gamma x CRT limbs x ring_dim x 8.
    pub fn payload_bytes(&self) -> usize {
        self.column_sums_ntt
            .iter()
            .map(|sum| std::mem::size_of_val(sum.coeffs()))
            .sum()
    }

    /// [`Self::payload_bytes`] plus the tags, their key and the containers around
    /// them, counted by length, for a server's per-shard memory accounting.
    pub fn resident_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of_val(self.moduli.as_slice())
            + std::mem::size_of_val(self.tag_key.as_slice())
            + std::mem::size_of_val(self.column_tags.as_slice())
            + self
                .column_sums_ntt
                .iter()
                .map(|sum| {
                    std::mem::size_of::<Poly>()
                        + std::mem::size_of_val(sum.coeffs())
                        + std::mem::size_of_val(sum.moduli())
                })
                .sum::<usize>()
    }

    /// The only way to this store's sums: recompute every column's tag from `columns`
    /// and refuse on the first that differs, folding `b` over the columns in the
    /// same pass.
    ///
    /// # Errors
    ///
    /// The store was built under other pack params, `b` is off the ring, or a column
    /// is not the one the store was folded from.
    pub(crate) fn check_columns_and_fold_b<'a>(
        &'a self,
        pack_params: &'a PackParams,
        columns: &[Poly],
        b: &Poly,
    ) -> Result<CheckedColumns<'a>, String> {
        let n = self.ring_dim;
        let moduli = self.moduli.as_slice();
        if pack_params.ring_dim != n
            || pack_params.moduli != moduli
            || pack_params.num_to_pack != self.num_to_pack
            || pack_params.generator != self.generator
        {
            return Err(format!(
                "packing store refused: built for gamma {} generator {} at ring_dim {n} \
                 moduli {moduli:?}, asked to pack under gamma {} generator {} at ring_dim \
                 {} moduli {:?}",
                self.num_to_pack,
                self.generator,
                pack_params.num_to_pack,
                pack_params.generator,
                pack_params.ring_dim,
                pack_params.moduli
            ));
        }
        require_served_columns(columns, n, moduli, self.num_to_pack)?;
        if b.is_ntt() || b.dimension() != n || b.moduli() != moduli {
            return Err(format!(
                "packing store refused: the query's b must be a coefficient-form \
                 polynomial at ring_dim {n} moduli {moduli:?}"
            ));
        }

        let mut tags = vec![0u64; columns.len()];
        let mut b_coeffs = vec![0u64; n * moduli.len()];
        for (limb, &modulus) in moduli.iter().enumerate() {
            let stripe = limb * n..(limb + 1) * n;
            let b_reversed = reverse_for_coeff0(&b.coeffs()[stripe.clone()], modulus);
            let key = &self.tag_key[2 * stripe.start..2 * stripe.end];
            let folded: Vec<(u64, u64)> = columns
                .par_iter()
                .map(|column| {
                    fold_stripe(&column.coeffs()[stripe.clone()], key, &b_reversed, modulus)
                })
                .collect();
            for (j, (tag_share, coeff0)) in folded.into_iter().enumerate() {
                tags[j] = add_tags(tags[j], tag_share);
                b_coeffs[stripe.start + j] = coeff0;
            }
        }
        if let Some(column) = tags
            .iter()
            .zip(&self.column_tags)
            .position(|(served, built)| served != built)
        {
            return Err(format!(
                "packing store refused: column {column} is not the column the store was \
                 built from. The shard changed after the build; rebuild the store before \
                 serving it"
            ));
        }

        Ok(CheckedColumns {
            store: self,
            pack_params,
            b_poly: Poly::from_crt_coeffs_reduced(b_coeffs, moduli),
        })
    }
}

/// Proof that a store's tags matched the columns being served, holding the folded b:
/// coefficient j is coefficient 0 of `b * h_j`.
pub(crate) struct CheckedColumns<'a> {
    store: &'a ShardPackingStore,
    pack_params: &'a PackParams,
    b_poly: Poly,
}

impl CheckedColumns<'_> {
    /// What [`packing_offline`](super::inspiring2::packing_offline) returns for
    /// `a_j = a * h_j`, with step 1 read from the store.
    ///
    /// # Errors
    ///
    /// `a` or `ctx` is off the store's ring.
    pub(crate) fn packing_offline(
        &self,
        packing_key: &PackingKeyBody,
        a: &Poly,
        ctx: &NttContext,
    ) -> Result<PrecompInsPIR, String> {
        let store = self.store;
        let n = store.ring_dim;
        let moduli = store.moduli.as_slice();
        if ctx.dimension() != n || ctx.moduli() != moduli {
            return Err(format!(
                "packing store refused: NTT context is for ring_dim {} moduli {:?}, the \
                 store is for ring_dim {n} moduli {moduli:?}",
                ctx.dimension(),
                ctx.moduli()
            ));
        }
        if a.is_ntt() || a.dimension() != n || a.moduli() != moduli {
            return Err(format!(
                "packing store refused: the query's a must be a coefficient-form \
                 polynomial at ring_dim {n} moduli {moduli:?}"
            ));
        }
        let tables = self
            .pack_params
            .rotation_tables(store.num_to_pack)
            .ok_or_else(|| {
                format!(
                    "packing store refused: pack params hold no full automorphism table \
                     for one of the {} rotations at ring_dim {n}",
                    store.num_to_pack
                )
            })?;

        let a_ntt = a.to_ntt_new(ctx);
        let rotations: Vec<(&Poly, &[usize])> = store.column_sums_ntt.iter().zip(tables).collect();
        let r_all: Vec<Poly> = rotations
            .par_iter()
            .map(|&(sum, table)| {
                let mut rotated_product = Poly::zero_moduli(n, moduli);
                rotated_product.force_ntt_domain();
                for limb in 0..moduli.len() {
                    let stripe = limb * n..(limb + 1) * n;
                    let a_limb = &a_ntt.coeffs()[stripe.clone()];
                    let sum_limb = &sum.coeffs()[stripe.clone()];
                    let out = &mut rotated_product.coeffs_mut()[stripe];
                    for ((out_k, &source), &sum_k) in out.iter_mut().zip(table).zip(sum_limb) {
                        *out_k = ctx.pointwise_mul_single_at(a_limb[source], sum_k, limb);
                    }
                }
                rotated_product
            })
            .collect();

        Ok(fold_rotations_backward(
            self.pack_params,
            packing_key,
            r_all,
            ctx,
        ))
    }

    pub(crate) fn into_b_poly(self) -> Poly {
        self.b_poly
    }
}

fn require_served_columns(
    columns: &[Poly],
    ring_dim: usize,
    moduli: &[u64],
    num_to_pack: usize,
) -> Result<(), String> {
    if columns.len() != num_to_pack {
        return Err(format!(
            "packing store refused: the shard holds {} columns, the store packs \
             {num_to_pack}",
            columns.len()
        ));
    }
    match columns.iter().position(|column| {
        column.is_ntt()
            || column.dimension() != ring_dim
            || column.moduli() != moduli
            || column.coeffs().len() != ring_dim * moduli.len()
    }) {
        Some(column) => Err(format!(
            "packing store refused: column {column} is not a coefficient-form polynomial \
             at ring_dim {ring_dim} moduli {moduli:?}"
        )),
        None => Ok(()),
    }
}

/// `b` laid out so that `sum_k out[k] * h[k]` is coefficient 0 of `b * h` in
/// `Z_q[X]/(X^d + 1)`: `b_0 h_0 - sum_{k>0} b_{d-k} h_k`.
fn reverse_for_coeff0(b: &[u64], modulus: u64) -> Vec<u64> {
    let d = b.len();
    (0..d)
        .map(|k| {
            if k == 0 {
                b[0] % modulus
            } else {
                (modulus - b[d - k] % modulus) % modulus
            }
        })
        .collect()
}

/// The two half-words of `word` under their key pair, below 2^94.
#[inline]
fn tag_term(word: u64, key_pair: &[u64; 2]) -> u128 {
    u128::from(word & 0xffff_ffff) * u128::from(key_pair[0])
        + u128::from(word >> 32) * u128::from(key_pair[1])
}

fn add_tags(left: u64, right: u64) -> u64 {
    ((u128::from(left) + u128::from(right)) % u128::from(TAG_PRIME)) as u64
}

/// The tag of one column over all its limbs.
fn column_tag(words: &[u64], key: &[u64]) -> u64 {
    let sum: u128 = words
        .iter()
        .zip(key.as_chunks::<2>().0)
        .map(|(&word, key_pair)| tag_term(word, key_pair))
        .sum();
    (sum % u128::from(TAG_PRIME)) as u64
}

/// One pass over a column's limb stripe: its share of the column tag, and
/// `sum_k b_reversed[k] * column[k] mod modulus`.
fn fold_stripe(column: &[u64], key: &[u64], b_reversed: &[u64], modulus: u64) -> (u64, u64) {
    let mut tag = 0u128;
    // A product is below 2^128, so its halves are summed apart and joined at the end.
    let (mut low, mut high) = (0u128, 0u128);
    for ((&word, key_pair), &b_k) in column.iter().zip(key.as_chunks::<2>().0).zip(b_reversed) {
        tag += tag_term(word, key_pair);
        let product = u128::from(word) * u128::from(b_k);
        low += product & u128::from(u64::MAX);
        high += product >> 64;
    }
    let q = u128::from(modulus);
    let coeff0 = ((high % q) * ((1u128 << 64) % q) + low % q) % q;
    ((tag % u128::from(TAG_PRIME)) as u64, coeff0 as u64)
}

#[cfg(test)]
mod tests;
