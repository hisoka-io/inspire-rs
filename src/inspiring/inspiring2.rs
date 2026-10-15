//! InspiRING packing: gamma LWE ciphertexts into one RLWE with 2 key-switching
//! matrices where tree packing needs log(d). Indexing LWE samples by a multiplicative
//! generator of Z*_{2d} lets a backward recursion move nearly all work offline.
//!
//! Key split across the parties, all derived from the public `w_seed`:
//! - server offline holds `w_mask` and its rotations `w_all`, giving `a_hat` + `bold_t`
//! - the client uploads `y_body = tau_g(s)*G - s*w_mask + error`
//! - server online computes `y_all * bold_t + b_poly`, with `y_all[i] = tau_{g^i}(y_body)`
//!   read as an index permutation of `y_body` in NTT form
//!
//! Ported from eprint 2025/1352 and
//! <https://github.com/google/private-membership/tree/main/research/InsPIRe>.

use crate::lwe::LweCiphertext;
use crate::math::modular::csubq;
use crate::math::{GaussianSampler, NttContext, Poly};
use crate::params::InspireParams;
use crate::rgsw::{gadget_decompose, GadgetVector};
use crate::rlwe::{apply_automorphism, RlweCiphertext, RlweSecretKey};

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

/// Packing parameters. Monomials, the 1/gamma scalar and the automorphism tables are
/// all cached in NTT form so the hot paths are pointwise.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackParams {
    /// gamma: how many items pack into one ciphertext.
    pub num_to_pack: usize,
    /// Ring dimension.
    pub ring_dim: usize,
    /// Ciphertext modulus.
    pub q: u64,
    /// CRT moduli; length 1 in single-modulus mode.
    pub moduli: Vec<u64>,
    /// Generator of the multiplicative group.
    pub generator: usize,
    /// `gen_pows[i] = g^i mod 2n`.
    pub gen_pows: Vec<usize>,
    /// 1/gamma mod q.
    pub mod_inv_gamma: u64,
    /// 1/gamma as a constant polynomial in NTT form.
    pub mod_inv_poly_ntt: Poly,
    /// X^j.
    pub monomials: Vec<Poly>,
    /// -X^j.
    pub neg_monomials: Vec<Poly>,
    /// X^j in NTT form.
    pub monomials_ntt: Vec<Poly>,
    /// -X^j in NTT form.
    pub neg_monomials_ntt: Vec<Poly>,
    /// Gadget parameters.
    pub gadget: GadgetVector,
    /// tables[(t-1)/2] maps an NTT index to its source index under tau_t, odd t in [1, 2n).
    pub automorph_tables: Vec<Vec<usize>>,
    /// Low gadget digits with no key row: the recursion rounds them away. 0 is the
    /// exact gadget.
    pub dropped_digits: usize,
}

/// Rejection reason from [`PackParams::try_new`] / [`PackParams::try_new_full`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackParamsError {
    /// The packing width admits no generator of the required order.
    IllegalWidth {
        /// Requested gamma.
        num_to_pack: usize,
        /// Ring dimension the width was requested against.
        ring_dim: usize,
    },
    /// The ring dimension is not a power of two.
    RingDimNotPowerOfTwo {
        /// Offending ring dimension.
        ring_dim: usize,
    },
    /// gamma has no inverse mod q, so the packing cannot be rescaled by 1/gamma.
    WidthNotInvertible {
        /// Requested gamma.
        num_to_pack: usize,
        /// Ciphertext modulus.
        modulus: u64,
    },
    /// The NTT over the first CRT modulus does not evaluate at the n distinct roots of
    /// X^n + 1, so no automorphism permutation exists.
    AutomorphTablesUnderivable {
        /// Ring dimension of the rejected parameters.
        ring_dim: usize,
    },
    /// The parameter set asks for the approximate packing gadget where it is undefined.
    ApproximateGadgetUnsupported {
        /// What rules it out.
        reason: &'static str,
    },
}

impl std::fmt::Display for PackParamsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IllegalWidth {
                num_to_pack,
                ring_dim,
            } => {
                let legal = PackParams::legal_widths(*ring_dim);
                write!(
                    f,
                    "InspiRING packing width {num_to_pack} is illegal at ring_dim {ring_dim}: \
                     the width must be a power of two in [1, {}] so that the automorphism \
                     indices form a subgroup of Z*_{} of order {num_to_pack}. Legal widths: \
                     {legal:?}. A record encoded as 16-bit columns picks the width as \
                     ceil(entry_size_bytes / 2)",
                    ring_dim / 2,
                    2 * ring_dim
                )
            }
            Self::RingDimNotPowerOfTwo { ring_dim } => write!(
                f,
                "InspiRING packing requires a power-of-two ring dimension, got {ring_dim}"
            ),
            Self::WidthNotInvertible {
                num_to_pack,
                modulus,
            } => write!(
                f,
                "InspiRING packing width {num_to_pack} has no inverse mod q = {modulus}; \
                 the 1/gamma rescaling in the offline phase is undefined"
            ),
            Self::AutomorphTablesUnderivable { ring_dim } => write!(
                f,
                "InspiRING automorphism tables are undefined at ring_dim {ring_dim}: the NTT \
                 over the first CRT modulus does not evaluate at {ring_dim} distinct roots of \
                 X^n + 1"
            ),
            Self::ApproximateGadgetUnsupported { reason } => write!(
                f,
                "the approximate packing gadget is unavailable here: {reason}. Clear \
                 approximate_packing_gadget on the parameter set"
            ),
        }
    }
}

impl std::error::Error for PackParamsError {}

impl PackParams {
    /// Legal partial-packing widths at this ring dimension: the powers of two
    /// up to `ring_dim / 2`.
    pub fn legal_widths(ring_dim: usize) -> Vec<usize> {
        if !ring_dim.is_power_of_two() {
            return Vec::new();
        }
        (0..usize::BITS)
            .map(|k| 1usize << k)
            .take_while(|w| *w <= ring_dim / 2)
            .collect()
    }

    /// Whether `num_to_pack` is accepted by [`try_new`](Self::try_new).
    pub fn is_legal_width(ring_dim: usize, num_to_pack: usize) -> bool {
        ring_dim.is_power_of_two()
            && num_to_pack != 0
            && num_to_pack.is_power_of_two()
            && num_to_pack <= ring_dim / 2
    }

    /// Parameters for partial packing (gamma < n).
    ///
    /// The backward recursion projects onto the subring fixed by `<g>` only when
    /// `<g>` has order exactly gamma, which for `g = 2n/gamma + 1` means gamma must
    /// be a power of two in `[1, n/2]`; other widths give a non-group index set.
    pub fn try_new(params: &InspireParams, num_to_pack: usize) -> Result<Self, PackParamsError> {
        let n = params.ring_dim;
        if !n.is_power_of_two() {
            return Err(PackParamsError::RingDimNotPowerOfTwo { ring_dim: n });
        }
        if !Self::is_legal_width(n, num_to_pack) {
            return Err(PackParamsError::IllegalWidth {
                num_to_pack,
                ring_dim: n,
            });
        }
        Self::build(params, num_to_pack, Self::partial_generator(n, num_to_pack))
    }

    /// `2n/gamma + 1`, whose powers have order gamma in Z*_{2n}; the server's rotation
    /// tables and the client's key body must walk the same one.
    fn partial_generator(ring_dim: usize, num_to_pack: usize) -> usize {
        (2 * ring_dim / num_to_pack) + 1
    }

    /// Parameters for full packing (gamma = n).
    ///
    /// `ord(5) = n/2` in Z*_{2n} and `+/-<5> = Z*_{2n}`, so the `<5>` branch and its
    /// conjugate together cover all n slots. Only the `*_full` entry points accept
    /// the result.
    pub fn try_new_full(params: &InspireParams) -> Result<Self, PackParamsError> {
        let n = params.ring_dim;
        if !n.is_power_of_two() {
            return Err(PackParamsError::RingDimNotPowerOfTwo { ring_dim: n });
        }
        if params.approximate_packing_gadget {
            return Err(PackParamsError::ApproximateGadgetUnsupported {
                reason: "full packing (gamma = ring_dim) decomposes every digit",
            });
        }
        Self::build(params, n, 5)
    }

    fn build(
        params: &InspireParams,
        num_to_pack: usize,
        generator: usize,
    ) -> Result<Self, PackParamsError> {
        let n = params.ring_dim;
        let q = params.q;
        let moduli = params.moduli().to_vec();
        let two_n = 2 * n;
        let dropped_digits = params.packing_dropped_digits();
        if dropped_digits >= params.packing_gadget_len {
            return Err(PackParamsError::ApproximateGadgetUnsupported {
                reason: "dropping the lowest digit of a one-digit gadget leaves no key row",
            });
        }
        let ctx = params.ntt_context();

        let mut gen_pows = Vec::with_capacity(n);
        let mut val = 1usize;
        for _ in 0..n {
            gen_pows.push(val);
            val = (val * generator) % two_n;
        }

        let mod_inv_gamma =
            mod_inverse_u64(num_to_pack as u64, q).ok_or(PackParamsError::WidthNotInvertible {
                num_to_pack,
                modulus: q,
            })?;

        let mut mod_inv_poly_ntt = Poly::constant_moduli(mod_inv_gamma, n, &moduli);
        mod_inv_poly_ntt.to_ntt(&ctx);

        let mut monomials = Vec::with_capacity(n);
        let mut neg_monomials = Vec::with_capacity(n);
        let mut monomials_ntt = Vec::with_capacity(n);
        let mut neg_monomials_ntt = Vec::with_capacity(n);

        for j in 0..n {
            let mut coeffs = vec![0u64; n];
            coeffs[j] = 1;
            let mono = Poly::from_coeffs_moduli(coeffs.clone(), &moduli);
            let mut mono_ntt = mono.clone();
            mono_ntt.to_ntt(&ctx);
            monomials.push(mono);
            monomials_ntt.push(mono_ntt);

            coeffs[j] = q - 1;
            let neg_mono = Poly::from_coeffs_moduli(coeffs, &moduli);
            let mut neg_mono_ntt = neg_mono.clone();
            neg_mono_ntt.to_ntt(&ctx);
            neg_monomials.push(neg_mono);
            neg_monomials_ntt.push(neg_mono_ntt);
        }

        let gadget = GadgetVector::new(params.gadget_base, params.packing_gadget_len, q);

        let automorph_tables = derive_automorph_tables(&monomials_ntt, &moduli)
            .ok_or(PackParamsError::AutomorphTablesUnderivable { ring_dim: n })?;

        Ok(Self {
            num_to_pack,
            ring_dim: n,
            q,
            moduli,
            generator,
            gen_pows,
            mod_inv_gamma,
            mod_inv_poly_ntt,
            monomials,
            neg_monomials,
            monomials_ntt,
            neg_monomials_ntt,
            gadget,
            automorph_tables,
            dropped_digits,
        })
    }

    /// Polynomials in a packing key for these parameters, and digits per link of the
    /// offline recursion: the gadget digits above the dropped ones.
    pub fn key_rows(&self) -> usize {
        self.gadget.len.saturating_sub(self.dropped_digits)
    }

    /// Table index for tau_t, defined for odd t in [1, 2n).
    #[inline]
    pub fn table_index(&self, t: usize) -> usize {
        debug_assert!(t % 2 == 1, "Automorphism index must be odd");
        (t - 1) / 2
    }

    /// Automorphism table for tau_t.
    #[inline]
    pub fn get_automorph_table(&self, t: usize) -> &[usize] {
        &self.automorph_tables[self.table_index(t)]
    }

    /// The tables of `tau_{g^i}` for i in `0..count`, or `None` when one is missing or
    /// short. A short table would end a slot walk early and drop terms.
    pub(super) fn rotation_tables(&self, count: usize) -> Option<Vec<&[usize]>> {
        self.gen_pows
            .iter()
            .take(count)
            .map(|g_pow| {
                let table = self.automorph_tables.get(g_pow.checked_sub(1)? / 2)?;
                (table.len() == self.ring_dim).then_some(table.as_slice())
            })
            .collect::<Option<Vec<&[usize]>>>()
            .filter(|tables| tables.len() == count)
    }
}

/// Permutation tables satisfying `NTT(tau_t(poly))[i] = NTT(poly)[table[i]]` for every
/// odd t in [1, 2n), which turns an automorphism into an O(n) index shuffle.
///
/// Slot j of the forward NTT holds R * poly(w_j), with R the Montgomery factor and the
/// w_j the n distinct roots of X^n + 1 mod a prime, so tau_t moves slot j to the slot
/// holding w_j^t. Every root is an odd power e_j of w_0, hence that slot is the one
/// whose exponent is t * e_j mod 2n. Limb 0 fixes the permutation on its own. `None`
/// means limb 0 is not such an evaluation, where no permutation exists.
fn derive_automorph_tables(monomials_ntt: &[Poly], moduli: &[u64]) -> Option<Vec<Vec<usize>>> {
    let n = monomials_ntt.len();
    let two_n = 2 * n;
    let q0 = moduli.first().copied().filter(|&q| q > 1)?;

    // Limb 0 of NTT(X^k) at slot j for k in [0, 2n), via X^(k+n) = -X^k.
    let slot_value = |k: usize, j: usize| -> Option<u64> {
        let (base, negate) = if k < n { (k, false) } else { (k - n, true) };
        let value = *monomials_ntt.get(base)?.coeffs().get(j)? % q0;
        Some(if negate { (q0 - value) % q0 } else { value })
    };

    let mut powers_of_w0 = (1..two_n)
        .step_by(2)
        .map(|e| slot_value(e, 0).map(|value| (value, e)))
        .collect::<Option<Vec<_>>>()?;
    powers_of_w0.sort_unstable();
    if powers_of_w0.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return None;
    }

    // Odd exponent e is stored at e / 2.
    let mut exponents = Vec::with_capacity(n);
    let mut slot_of_exponent = vec![usize::MAX; n];
    for j in 0..n {
        let root = slot_value(1, j)?;
        let at = powers_of_w0
            .binary_search_by_key(&root, |&(value, _)| value)
            .ok()?;
        let e = powers_of_w0[at].1;
        if slot_of_exponent[e / 2] != usize::MAX {
            return None;
        }
        slot_of_exponent[e / 2] = j;
        exponents.push(e as u64);
    }

    let two_n = two_n as u64;
    Some(
        (1..two_n)
            .step_by(2)
            .map(|t| {
                exponents
                    .iter()
                    .map(|&e| slot_of_exponent[(((t * e) % two_n) / 2) as usize])
                    .collect()
            })
            .collect(),
    )
}

/// Apply tau_t as an O(n) index permutation in the NTT domain.
#[inline]
pub fn apply_automorphism_ntt(poly_ntt: &Poly, table: &[usize]) -> Poly {
    debug_assert!(poly_ntt.is_ntt(), "Polynomial must be in NTT domain");
    let n = poly_ntt.dimension();
    let moduli = poly_ntt.moduli();
    let crt_count = moduli.len();

    // Per-limb: the permutation depends on the ring dimension, not the modulus, so the
    // same table applies to every CRT limb and none may be dropped.
    let mut result_coeffs = vec![0u64; n * crt_count];
    let src = poly_ntt.coeffs();

    for m in 0..crt_count {
        let offset = m * n;
        for i in 0..n {
            result_coeffs[offset + i] = src[offset + table[i]];
        }
    }

    // `from_crt_coeffs`, not `from_coeffs_moduli`: these are already per-prime
    // residues, and it clears the NTT flag that the input carried.
    let mut result = Poly::from_crt_coeffs(result_coeffs, moduli);
    result.force_ntt_domain();
    result
}

/// [`apply_automorphism_ntt`] writing into a caller-owned buffer.
#[inline]
pub fn apply_automorphism_ntt_into(poly_ntt: &Poly, table: &[usize], out: &mut Poly) {
    debug_assert!(poly_ntt.is_ntt(), "Input must be in NTT domain");
    debug_assert!(out.is_ntt(), "Output must be in NTT domain");

    let n = poly_ntt.dimension();
    let crt_count = poly_ntt.moduli().len();
    let src = poly_ntt.coeffs();
    let dst = out.coeffs_mut();

    for m in 0..crt_count {
        let offset = m * n;
        for i in 0..n {
            dst[offset + i] = src[offset + table[i]];
        }
    }
}

/// Apply tau_t and its conjugate tau_{2n-t} in one pass.
#[inline]
pub fn apply_automorphism_ntt_double(
    poly_ntt: &Poly,
    table_pos: &[usize],
    table_neg: &[usize],
) -> (Poly, Poly) {
    debug_assert!(poly_ntt.is_ntt(), "Polynomial must be in NTT domain");
    let n = poly_ntt.dimension();
    let moduli = poly_ntt.moduli();
    let crt_count = moduli.len();

    let mut result_pos = vec![0u64; n * crt_count];
    let mut result_neg = vec![0u64; n * crt_count];
    let src = poly_ntt.coeffs();

    for m in 0..crt_count {
        let offset = m * n;
        for i in 0..n {
            result_pos[offset + i] = src[offset + table_pos[i]];
            result_neg[offset + i] = src[offset + table_neg[i]];
        }
    }

    let mut poly_pos = Poly::from_crt_coeffs(result_pos, moduli);
    let mut poly_neg = Poly::from_crt_coeffs(result_neg, moduli);
    poly_pos.force_ntt_domain();
    poly_neg.force_ntt_domain();

    (poly_pos, poly_neg)
}

/// Output of the offline phase.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrecompInsPIR {
    /// `R[0]` after the backward recursion, i.e. the packed ciphertext's `a`.
    pub a_hat: Poly,
    /// Gadget-decomposed R[i+1], shape (gamma-1) x gadget_len.
    pub bold_t: Vec<Vec<Poly>>,
    /// `bold_t` in NTT form; derived, so it is rebuilt rather than serialized.
    #[serde(skip)]
    pub bold_t_ntt: Vec<Vec<Poly>>,
    /// Conjugate branch, full packing only.
    pub bold_t_bar: Vec<Vec<Poly>>,
    /// Final gadget, full packing only.
    pub bold_t_hat: Vec<Poly>,
    /// gamma.
    pub num_to_pack: usize,
    /// Ring dimension.
    pub ring_dim: usize,
    /// Ciphertext modulus.
    pub q: u64,
}

impl PrecompInsPIR {
    /// Polynomial multiplications the online phase will perform.
    pub fn online_mult_count(&self) -> usize {
        if self.bold_t.is_empty() {
            0
        } else {
            self.bold_t.len() * self.bold_t[0].len()
        }
    }

    /// Rebuild `bold_t_ntt` if it was dropped by deserialization.
    pub fn ensure_ntt_cached(&mut self, ctx: &NttContext) {
        if self.bold_t_ntt.is_empty() && !self.bold_t.is_empty() {
            self.bold_t_ntt = self
                .bold_t
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|p| {
                            let mut pn = p.clone();
                            pn.to_ntt(ctx);
                            pn
                        })
                        .collect()
                })
                .collect();
        }
    }
}

/// Server-side `w_mask` and its rotations, all expanded from the public `w_seed`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OfflinePackingKeys {
    /// Public seed `w_mask` expands from.
    pub w_seed: [u8; 32],
    /// Public seed `v_mask` expands from, full packing only.
    pub v_seed: [u8; 32],
    /// One uniform polynomial per key row, expanded from `w_seed`: the row for digit `k`
    /// is polynomial `k` of the expansion.
    pub w_mask: Vec<Poly>,
    /// Conjugation mask, full packing only.
    pub v_mask: Vec<Poly>,
    /// `w_all[i] = tau_{g^i}(w_mask)`.
    pub w_all: Vec<Vec<Poly>>,
    /// `w_all` in NTT form.
    pub w_all_ntt: Vec<Vec<Poly>>,
    /// `w_bar_all[i] = tau_{2n-g^i}(w_mask)`, full packing only.
    pub w_bar_all: Vec<Vec<Poly>>,
    /// `w_bar_all` in NTT form.
    pub w_bar_all_ntt: Vec<Vec<Poly>>,
    /// Whether these keys are for full packing.
    pub full_key: bool,
}

impl OfflinePackingKeys {
    /// Expand keys for partial packing (gamma < n).
    pub fn generate(pack_params: &PackParams, w_seed: [u8; 32]) -> Self {
        let n = pack_params.ring_dim;
        let q = pack_params.q;
        let num_to_pack = pack_params.num_to_pack;
        let ctx = NttContext::with_moduli(n, &pack_params.moduli);

        // Generate w_mask from seed and convert to NTT form
        let w_mask = kept_mask_rows(
            w_seed,
            n,
            q,
            &pack_params.moduli,
            pack_params.gadget.len,
            pack_params.dropped_digits,
        );
        let w_mask_ntt: Vec<Poly> = w_mask
            .iter()
            .map(|p| {
                let mut pn = p.clone();
                pn.to_ntt(&ctx);
                pn
            })
            .collect();

        let mut w_all = Vec::with_capacity(num_to_pack - 1);
        let mut w_all_ntt = Vec::with_capacity(num_to_pack - 1);

        for i in 0..(num_to_pack - 1) {
            let g_pow_i = pack_params.gen_pows[i];
            let table = pack_params.get_automorph_table(g_pow_i);

            let rotated_ntt: Vec<Poly> = w_mask_ntt
                .iter()
                .map(|poly_ntt| apply_automorphism_ntt(poly_ntt, table))
                .collect();

            // Coefficient form is what gadget decomposition consumes.
            let rotated: Vec<Poly> = rotated_ntt
                .iter()
                .map(|p| {
                    let mut pc = p.clone();
                    pc.from_ntt(&ctx);
                    pc
                })
                .collect();

            w_all.push(rotated);
            w_all_ntt.push(rotated_ntt);
        }

        Self {
            w_seed,
            v_seed: [0u8; 32],
            w_mask,
            v_mask: vec![],
            w_all,
            w_all_ntt,
            w_bar_all: vec![],
            w_bar_all_ntt: vec![],
            full_key: false,
        }
    }

    /// Expand keys for full packing (gamma = n), both rotation branches at once.
    pub fn generate_full(pack_params: &PackParams, w_seed: [u8; 32], v_seed: [u8; 32]) -> Self {
        let n = pack_params.ring_dim;
        let q = pack_params.q;
        let num_to_pack_half = n / 2;
        let two_n = 2 * n;
        let gadget_len = pack_params.gadget.len;
        let ctx = NttContext::with_moduli(n, &pack_params.moduli);

        let w_mask = generate_mask_from_seed(w_seed, n, q, &pack_params.moduli, gadget_len);
        let v_mask = generate_mask_from_seed(v_seed, n, q, &pack_params.moduli, gadget_len);

        let w_mask_ntt: Vec<Poly> = w_mask
            .iter()
            .map(|p| {
                let mut pn = p.clone();
                pn.to_ntt(&ctx);
                pn
            })
            .collect();

        let mut w_all = Vec::with_capacity(num_to_pack_half - 1);
        let mut w_all_ntt = Vec::with_capacity(num_to_pack_half - 1);
        let mut w_bar_all = Vec::with_capacity(num_to_pack_half - 1);
        let mut w_bar_all_ntt = Vec::with_capacity(num_to_pack_half - 1);

        for i in 0..(num_to_pack_half - 1) {
            let g_pow_i = pack_params.gen_pows[i];
            let neg_g_pow_i = (two_n - g_pow_i) % two_n;

            let table_pos = pack_params.get_automorph_table(g_pow_i);
            let table_neg = pack_params.get_automorph_table(neg_g_pow_i);

            let mut rotated_ntt = Vec::with_capacity(gadget_len);
            let mut rotated_bar_ntt = Vec::with_capacity(gadget_len);

            for poly_ntt in &w_mask_ntt {
                let (pos, neg) = apply_automorphism_ntt_double(poly_ntt, table_pos, table_neg);
                rotated_ntt.push(pos);
                rotated_bar_ntt.push(neg);
            }

            let rotated: Vec<Poly> = rotated_ntt
                .iter()
                .map(|p| {
                    let mut pc = p.clone();
                    pc.from_ntt(&ctx);
                    pc
                })
                .collect();
            let rotated_bar: Vec<Poly> = rotated_bar_ntt
                .iter()
                .map(|p| {
                    let mut pc = p.clone();
                    pc.from_ntt(&ctx);
                    pc
                })
                .collect();

            w_all.push(rotated);
            w_all_ntt.push(rotated_ntt);
            w_bar_all.push(rotated_bar);
            w_bar_all_ntt.push(rotated_bar_ntt);
        }

        Self {
            w_seed,
            v_seed,
            w_mask,
            v_mask,
            w_all,
            w_all_ntt,
            w_bar_all,
            w_bar_all_ntt,
            full_key: true,
        }
    }
}

/// Client packing keys as they cross the wire. The server keeps them as a
/// [`crate::pir::PackingKeySeat`] and reads every rotation through the automorphism
/// tables, so neither side expands one.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "ClientPackingKeysWire")]
pub struct ClientPackingKeys {
    /// `y_body[r] = tau_g(s)*z^k - s*w_mask[r] + error`, one row per gadget digit `k`
    /// the key covers, lowest first.
    pub y_body: Vec<Poly>,
    /// Conjugation body, full packing only.
    ///
    /// Never `skip_serializing_if`: bincode is positional, so an omitted length
    /// prefix shifts every later read.
    #[serde(default)]
    pub z_body: Vec<Poly>,
    /// Whether these keys are for full packing.
    pub full_key: bool,
    /// gamma.
    pub num_to_pack: usize,
}

/// [`ClientPackingKeys`] before its bodies are checked.
#[derive(Deserialize)]
struct ClientPackingKeysWire {
    y_body: Vec<Poly>,
    #[serde(default)]
    z_body: Vec<Poly>,
    full_key: bool,
    num_to_pack: usize,
}

impl TryFrom<ClientPackingKeysWire> for ClientPackingKeys {
    type Error = String;

    /// Bodies are generated in the coefficient domain on one ring; the server rotates
    /// them, so a forged NTT flag or a stray ring would pack wrong values.
    fn try_from(wire: ClientPackingKeysWire) -> Result<Self, Self::Error> {
        let bodies = wire
            .y_body
            .iter()
            .enumerate()
            .map(|(index, poly)| ("y_body", index, poly))
            .chain(
                wire.z_body
                    .iter()
                    .enumerate()
                    .map(|(index, poly)| ("z_body", index, poly)),
            );
        let mut reference: Option<(String, &Poly)> = None;
        for (body, index, poly) in bodies {
            if poly.is_ntt() {
                return Err(format!(
                    "coefficient-domain packing-key {body}[{index}] required; client material \
                     declares NTT domain. Rebuild the keys from the current CRS"
                ));
            }
            let name = format!("packing-key {body}[{index}]");
            match &reference {
                None => reference = Some((name, poly)),
                Some((first_name, first)) => {
                    crate::rlwe::require_same_ring(&name, poly, first_name, first)?;
                }
            }
        }
        Ok(Self {
            y_body: wire.y_body,
            z_body: wire.z_body,
            full_key: wire.full_key,
            num_to_pack: wire.num_to_pack,
        })
    }
}

impl ClientPackingKeys {
    /// Derive packing keys from the secret key and the CRS `w_seed`.
    ///
    /// Reads the ring, the gadget and the generator only, so a client never builds
    /// [`PackParams`] and its automorphism tables.
    ///
    /// Call once per secret key: `w_mask` is fixed by the seed, so a second key under
    /// the same secret repeats the mask with fresh error. [`crate::pir::ClientSession`]
    /// builds its key once and resends those bytes.
    ///
    /// # Errors
    ///
    /// [`PackParamsError::IllegalWidth`] when `num_to_pack` is not a legal partial
    /// packing width at `params.ring_dim`, and
    /// [`PackParamsError::ApproximateGadgetUnsupported`] when the set drops its only
    /// packing digit.
    ///
    /// ```
    /// use raven_inspire::inspiring::ClientPackingKeys;
    /// use raven_inspire::math::GaussianSampler;
    /// use raven_inspire::params::InspireParams;
    /// use raven_inspire::rlwe::RlweSecretKey;
    ///
    /// let params = InspireParams::secure_128_d2048();
    /// let mut sampler = GaussianSampler::with_seed(params.sigma, 1);
    /// let secret_key = RlweSecretKey::generate(&params, &mut sampler);
    /// let keys = ClientPackingKeys::generate(&secret_key, &params, 16, [7; 32], &mut sampler)?;
    /// assert_eq!(keys.y_body.len(), params.packing_key_rows());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn generate(
        sk: &RlweSecretKey,
        params: &InspireParams,
        num_to_pack: usize,
        w_seed: [u8; 32],
        sampler: &mut GaussianSampler,
    ) -> Result<Self, PackParamsError> {
        let n = params.ring_dim;
        if !PackParams::is_legal_width(n, num_to_pack) {
            return Err(PackParamsError::IllegalWidth {
                num_to_pack,
                ring_dim: n,
            });
        }
        let first_digit = params.packing_dropped_digits();
        if first_digit >= params.packing_gadget_len {
            return Err(PackParamsError::ApproximateGadgetUnsupported {
                reason: "dropping the lowest digit of a one-digit gadget leaves no key row",
            });
        }
        let q = params.q;
        let moduli = params.moduli();
        let gadget = GadgetVector::new(params.gadget_base, params.packing_gadget_len, q);
        let ctx = NttContext::with_moduli(n, moduli);

        let w_mask = kept_mask_rows(w_seed, n, q, moduli, gadget.len, first_digit);
        let generator = PackParams::partial_generator(n, num_to_pack) % (2 * n);
        let y_body = generate_ksk_body(sk, generator, &gadget, first_digit, &w_mask, sampler, &ctx);

        Ok(Self {
            y_body,
            z_body: vec![],
            full_key: false,
            num_to_pack,
        })
    }

    /// [`Self::generate`] for full packing; `z_body` uses the conjugation index 2n-1.
    pub fn generate_full(
        sk: &RlweSecretKey,
        pack_params: &PackParams,
        w_seed: [u8; 32],
        v_seed: [u8; 32],
        sampler: &mut GaussianSampler,
    ) -> Self {
        let n = pack_params.ring_dim;
        let q = pack_params.q;
        let two_n = 2 * n;
        let gadget_len = pack_params.gadget.len;
        let ctx = NttContext::with_moduli(n, &pack_params.moduli);

        let w_mask = generate_mask_from_seed(w_seed, n, q, &pack_params.moduli, gadget_len);
        let v_mask = generate_mask_from_seed(v_seed, n, q, &pack_params.moduli, gadget_len);

        let gen_pow = pack_params.gen_pows[1];
        let gadget = &pack_params.gadget;
        let y_body = generate_ksk_body(sk, gen_pow, gadget, 0, &w_mask, sampler, &ctx);
        let z_body = generate_ksk_body(sk, two_n - 1, gadget, 0, &v_mask, sampler, &ctx);

        Self {
            y_body,
            z_body,
            full_key: true,
            num_to_pack: n,
        }
    }
}

/// The mask rows of a key that has no rows for its `dropped` lowest digits: row r is
/// the mask the full expansion gives digit `dropped + r`. A mask follows its digit, so
/// such a key is the full key less its lowest rows, and one secret under both gadgets
/// publishes each digit under one mask.
fn kept_mask_rows(
    seed: [u8; 32],
    n: usize,
    q: u64,
    moduli: &[u64],
    gadget_len: usize,
    dropped: usize,
) -> Vec<Poly> {
    generate_mask_from_seed(seed, n, q, moduli, gadget_len)
        .into_iter()
        .skip(dropped)
        .collect()
}

/// Deterministically expand a seed into gadget_len uniform polynomials.
fn generate_mask_from_seed(
    seed: [u8; 32],
    n: usize,
    q: u64,
    moduli: &[u64],
    gadget_len: usize,
) -> Vec<Poly> {
    let mut rng = ChaCha20Rng::from_seed(seed);
    let mut result = Vec::with_capacity(gadget_len);

    for _ in 0..gadget_len {
        let mut coeffs = vec![0u64; n];
        for coeff in &mut coeffs {
            *coeff = (rng.next_u64()) % q;
        }
        result.push(Poly::from_coeffs_moduli(coeffs, moduli));
    }

    result
}

/// Row r is `tau_g(s)*z^(first_digit + r) - s*w_mask[r] + error`: one row per mask
/// polynomial, so no row can go out without its mask.
fn generate_ksk_body(
    sk: &RlweSecretKey,
    gen_pow: usize,
    gadget: &GadgetVector,
    first_digit: usize,
    w_mask: &[Poly],
    sampler: &mut GaussianSampler,
    ctx: &NttContext,
) -> Vec<Poly> {
    let s = &sk.poly;
    let n = s.dimension();
    let moduli = s.moduli();

    let tau_s = apply_automorphism(s, gen_pow);

    let mut s_ntt = s.clone();
    s_ntt.to_ntt(ctx);

    let mut result = Vec::with_capacity(w_mask.len());

    for (row, w_row) in w_mask.iter().enumerate() {
        let gadget_power = gadget.power(first_digit + row);

        let tau_s_times_g = tau_s.scalar_mul(gadget_power);

        let mut w_row_ntt = w_row.clone();
        w_row_ntt.to_ntt(ctx);
        let mut s_times_mask = s_ntt.mul_ntt_domain(&w_row_ntt, ctx);
        s_times_mask.from_ntt(ctx);
        let neg_s_times_mask = s_times_mask.negate();

        let error = Poly::sample_gaussian_moduli(n, moduli, sampler);

        let result_k = &(&tau_s_times_g + &neg_s_times_mask) + &error;
        result.push(result_k);
    }

    result
}

/// Alias kept for callers naming the offline keys by their key-body role.
pub type PackingKeyBody = OfflinePackingKeys;

/// Offline phase: a_hat and bold_t from the query's a-vectors.
///
/// `R[i] = (1/gamma) * tau_{g^i}(sum_j X^{j*g^{n-i}} * a_j)`, then a backward
/// recursion for i from gamma-2 down to 0 with `T[i] = G^{-1}(R[i+1])` and
/// `R[i] += sum_k w_all[i][k] * T[i][k]`, yielding `a_hat = R[0]`.
pub fn packing_offline(
    pack_params: &PackParams,
    packing_key: &PackingKeyBody,
    a_ct_tilde: &[Poly],
    ctx: &NttContext,
) -> PrecompInsPIR {
    let a_ct_ntt: Vec<Poly> = a_ct_tilde
        .iter()
        .map(|a| {
            let mut a_ntt = a.clone();
            a_ntt.to_ntt(ctx);
            a_ntt
        })
        .collect();

    let r_all = rotation_sums(pack_params, &a_ct_ntt, ctx);
    fold_rotations_backward(pack_params, packing_key, r_all, ctx)
}

/// Step 1 of the offline phase, NTT form in and out:
/// `(1/gamma) * tau_{g^i}(sum_j X^{j*g^{n-i}} * inputs[j])` for i in `0..gamma`.
///
/// Linear in `inputs`, which is what lets a shard fold its own columns ahead of the
/// queries: with `inputs[j] = a * h_j` the result is `tau_{g^i}(a)` times the result
/// for `inputs[j] = h_j`.
pub(super) fn rotation_sums(
    pack_params: &PackParams,
    inputs_ntt: &[Poly],
    ctx: &NttContext,
) -> Vec<Poly> {
    let n = pack_params.ring_dim;
    let moduli = &pack_params.moduli;
    let gen_pows = &pack_params.gen_pows;

    // Each sum is independent, so this O(gamma^2) region parallelizes; computed per
    // query it dominates server time.
    use crate::par_prelude::*;
    (0..pack_params.num_to_pack)
        .into_par_iter()
        .map(|i| {
            let mut r_pow_i_ntt = Poly::zero_moduli(n, moduli);
            r_pow_i_ntt.to_ntt(ctx);

            for (j, input_ntt) in inputs_ntt.iter().enumerate() {
                // g^{n-i}: the index walks the generator in the inverse direction.
                let exp_index = (n - i) % n;
                let index = (j * gen_pows[exp_index]) % (2 * n);

                // index >= n has wrapped past X^n = -1.
                let monomial_ntt = if index < n {
                    &pack_params.monomials_ntt[index % n]
                } else {
                    &pack_params.neg_monomials_ntt[index % n]
                };

                r_pow_i_ntt.mul_acc_ntt_domain(monomial_ntt, input_ntt, ctx);
            }

            let r_pow_i_scaled = r_pow_i_ntt.mul_ntt_domain(&pack_params.mod_inv_poly_ntt, ctx);

            let g_pow_i = gen_pows[i];
            let table = pack_params.get_automorph_table(g_pow_i);
            apply_automorphism_ntt(&r_pow_i_scaled, table)
        })
        .collect()
}

/// The backward recursion of the offline phase over `r_all[i] = R[i]` in NTT form:
/// `T[i] = G^{-1}(R[i+1])`, `R[i] += sum_k w_all[i][k] * T[i][k]`, `a_hat = R[0]`.
/// Every way of producing `R` ends here, so their outputs cannot differ.
///
/// With dropped digits `G^{-1}` is [`gadget_decompose_rounded`]: `T[i]` then recomposes
/// to `R[i+1]` less a residual of at most half a dropped unit, which reaches the
/// packed ciphertext as `tau_g(s)` times that residual in place of the dropped rows'
/// key error.
pub(super) fn fold_rotations_backward(
    pack_params: &PackParams,
    packing_key: &PackingKeyBody,
    mut r_all: Vec<Poly>,
    ctx: &NttContext,
) -> PrecompInsPIR {
    let num_to_pack = pack_params.num_to_pack;
    // Its constants cost a transform per digit, which fewer links than that cannot repay.
    let last_digit = (num_to_pack > pack_params.gadget.len + 1)
        .then(|| LastDigitNtt::derive(pack_params, ctx))
        .flatten();

    let mut bold_t: Vec<Vec<Poly>> = Vec::with_capacity(num_to_pack - 1);
    let mut bold_t_ntt_rev: Vec<Vec<Poly>> = Vec::with_capacity(num_to_pack - 1);

    for i in (0..(num_to_pack - 1)).rev() {
        // Gadget decomposition needs coefficient domain.
        let mut r_i_plus_1_coeff = r_all[i + 1].clone();
        r_i_plus_1_coeff.from_ntt(ctx);

        let gadget_inv = if pack_params.dropped_digits == 0 {
            gadget_decompose(&r_i_plus_1_coeff, &pack_params.gadget)
        } else {
            gadget_decompose_rounded(
                &r_i_plus_1_coeff,
                &pack_params.gadget,
                pack_params.dropped_digits,
            )
        };

        // Serial on purpose: handing one digit to a second thread costs more than its
        // transform (1.5x slower measured), and a fused `mul_acc3_ntt_domain` measured
        // flat against three simple loops.
        let transformed = match &last_digit {
            Some(_) => gadget_inv.len().saturating_sub(1),
            None => gadget_inv.len(),
        };
        let mut gadget_inv_ntt: Vec<Poly> = gadget_inv
            .iter()
            .take(transformed)
            .map(|t_k| t_k.to_ntt_new(ctx))
            .collect();
        if let Some(last_digit) = &last_digit {
            gadget_inv_ntt.push(last_digit.ntt(&r_all[i + 1], &gadget_inv_ntt, ctx));
        }

        for (t_k_ntt, w_k_ntt) in gadget_inv_ntt.iter().zip(&packing_key.w_all_ntt[i]) {
            r_all[i].mul_acc_ntt_domain(t_k_ntt, w_k_ntt, ctx);
        }

        bold_t.push(gadget_inv);
        bold_t_ntt_rev.push(gadget_inv_ntt);
    }

    bold_t.reverse();
    let mut bold_t_ntt = bold_t_ntt_rev;
    bold_t_ntt.reverse();

    let mut a_hat = r_all[0].clone();
    a_hat.from_ntt(ctx);

    PrecompInsPIR {
        a_hat,
        bold_t,
        bold_t_ntt,
        bold_t_bar: vec![],
        bold_t_hat: vec![],
        num_to_pack,
        ring_dim: pack_params.ring_dim,
        q: pack_params.q,
    }
}

/// The NTT of a decomposition's last digit, read off the other digits' transforms.
///
/// When the gadget spans the modulus (`base^len >= q`) the digits recompose to the
/// coefficient: `sum_k base^k * T_k = R`. The NTT is linear, so
/// `NTT(T_last) = base^-(len-1) * (NTT(R) - sum_{k<len-1} base^k * NTT(T_k))`, and
/// `NTT(R)` is what the recursion already holds. Exact arithmetic in each limb's
/// field gives the transform's own words for `len` pointwise products.
struct LastDigitNtt {
    /// `base^-(len-1)`, constant in every slot.
    of_sum: Poly,
    /// `-base^(k-(len-1))` for each earlier digit k.
    of_digit: Vec<Poly>,
}

impl LastDigitNtt {
    /// `None` when the digits do not recompose (a gadget shorter than the modulus, or
    /// one that rounds its low digits away) or `base^(len-1)` has no inverse; every
    /// digit is then transformed.
    fn derive(pack_params: &PackParams, ctx: &NttContext) -> Option<Self> {
        if pack_params.dropped_digits != 0 {
            return None;
        }
        let gadget = &pack_params.gadget;
        let q = u128::from(pack_params.q);
        let base = u128::from(gadget.base);
        if base < 2 || q < 2 {
            return None;
        }
        let earlier_digits = gadget.len.checked_sub(1)?;

        let mut span = 1u128;
        let mut powers = Vec::with_capacity(gadget.len);
        let mut power = 1u128;
        for _ in 0..gadget.len {
            powers.push(power);
            power = power * (base % q) % q;
            span = span.saturating_mul(base);
        }
        if span < q {
            return None;
        }
        let top = u64::try_from(*powers.last()?).ok()?;
        let inverse = u128::from(mod_inverse_u64(top, pack_params.q)?);

        let constant = |value: u128| -> Option<Poly> {
            let mut poly = Poly::constant_moduli(
                u64::try_from(value).ok()?,
                pack_params.ring_dim,
                &pack_params.moduli,
            );
            poly.to_ntt(ctx);
            Some(poly)
        };
        Some(Self {
            of_sum: constant(inverse)?,
            of_digit: powers
                .iter()
                .take(earlier_digits)
                .map(|power| constant((q - inverse * power % q) % q))
                .collect::<Option<Vec<Poly>>>()?,
        })
    }

    /// `sum_ntt` is `NTT(R)`; `earlier_digits_ntt` are the transforms of digits
    /// `0..len-1`, in order.
    fn ntt(&self, sum_ntt: &Poly, earlier_digits_ntt: &[Poly], ctx: &NttContext) -> Poly {
        let mut last = Poly::zero_moduli(sum_ntt.dimension(), sum_ntt.moduli());
        last.force_ntt_domain();
        last.mul_acc_ntt_domain(sum_ntt, &self.of_sum, ctx);
        for (digit_ntt, constant) in earlier_digits_ntt.iter().zip(&self.of_digit) {
            last.mul_acc_ntt_domain(digit_ntt, constant, ctx);
        }
        last
    }
}

/// The base-z digits of `round(poly / z^dropped)`, lowest first: `len - dropped`
/// polynomials with `sum_r digit_r * z^(dropped + r) = poly - residual` over the
/// integers and `|residual| <= z^dropped / 2` per coefficient. The top digit keeps the
/// rounding carry, so it can equal z where every other digit stays below it.
fn gadget_decompose_rounded(poly: &Poly, gadget: &GadgetVector, dropped: usize) -> Vec<Poly> {
    // `coeff` composes the CRT limbs; one limb is already the coefficient.
    let composed: Vec<u64>;
    let coeffs = if poly.moduli().len() == 1 {
        poly.coeffs()
    } else {
        composed = (0..poly.dimension()).map(|j| poly.coeff(j)).collect();
        &composed
    };
    rounded_digit_rows(coeffs, gadget, dropped, true)
        .into_iter()
        .map(|row| Poly::from_coeffs_moduli(row, poly.moduli()))
        .collect()
}

/// [`gadget_decompose_rounded`] over plain coefficients. `shifts` lets a power-of-two
/// base split by shift and mask; the division form is the definition it must equal.
fn rounded_digit_rows(
    coeffs: &[u64],
    gadget: &GadgetVector,
    dropped: usize,
    shifts: bool,
) -> Vec<Vec<u64>> {
    let rows = gadget.len.saturating_sub(dropped);
    let mut digits = vec![vec![0u64; coeffs.len()]; rows];
    let Some(top) = rows.checked_sub(1) else {
        return digits;
    };
    let base_bits = gadget.base.trailing_zeros();
    let dropped_bits = u32::try_from(dropped)
        .ok()
        .and_then(|dropped| base_bits.checked_mul(dropped))
        .filter(|bits| shifts && gadget.base.is_power_of_two() && (1..64).contains(bits));

    if let Some(dropped_bits) = dropped_bits {
        let half = 1u128 << (dropped_bits - 1);
        let mask = gadget.base - 1;
        for (j, &coeff) in coeffs.iter().enumerate() {
            let mut rounded = ((u128::from(coeff) + half) >> dropped_bits) as u64;
            for row in &mut digits[..top] {
                row[j] = rounded & mask;
                rounded >>= base_bits;
            }
            digits[top][j] = rounded;
        }
    } else {
        let base = u128::from(gadget.base);
        let unit = base.saturating_pow(u32::try_from(dropped).unwrap_or(u32::MAX));
        for (j, &coeff) in coeffs.iter().enumerate() {
            let mut rounded = (u128::from(coeff) + unit / 2) / unit;
            for row in &mut digits[..top] {
                row[j] = (rounded % base) as u64;
                rounded /= base;
            }
            digits[top][j] = rounded as u64;
        }
    }
    digits
}

/// [`packing_offline`] for gamma = n, running the R and R-bar recursions together.
pub fn full_packing_offline(
    pack_params: &PackParams,
    packing_key: &PackingKeyBody,
    a_ct_tilde: &[Poly],
    ctx: &NttContext,
) -> PrecompInsPIR {
    let n = pack_params.ring_dim;
    let q = pack_params.q;
    let moduli = &pack_params.moduli;
    let num_to_pack_half = n / 2;
    let two_n = 2 * n;
    let gen_pows = &pack_params.gen_pows;
    let mod_inv = pack_params.mod_inv_gamma;

    assert_eq!(
        pack_params.num_to_pack, n,
        "Full packing requires gamma = n"
    );

    let mut r_all: Vec<Poly> = Vec::with_capacity(num_to_pack_half);
    let mut r_bar_all: Vec<Poly> = Vec::with_capacity(num_to_pack_half);

    for i in 0..num_to_pack_half {
        let mut r_pow_i = Poly::zero_moduli(n, moduli);
        let mut r_bar_pow_i = Poly::zero_moduli(n, moduli);

        for (j, a_j) in a_ct_tilde.iter().enumerate() {
            let exp_index = (n - i) % n;

            // Forward branch index j * g^{n-i}.
            let index = (j * gen_pows[exp_index]) % two_n;
            let monomial = if index < n {
                &pack_params.monomials[index % n]
            } else {
                &pack_params.neg_monomials[index % n]
            };
            let term = monomial.mul_ntt(a_j, ctx);
            r_pow_i = &r_pow_i + &term;

            // Conjugate branch negates the exponent.
            let neg_index = (two_n - (j * gen_pows[exp_index]) % two_n) % two_n;
            let neg_monomial = if neg_index < n {
                &pack_params.monomials[neg_index % n]
            } else {
                &pack_params.neg_monomials[neg_index % n]
            };
            let neg_term = neg_monomial.mul_ntt(a_j, ctx);
            r_bar_pow_i = &r_bar_pow_i + &neg_term;
        }

        r_pow_i = r_pow_i.scalar_mul(mod_inv);
        r_bar_pow_i = r_bar_pow_i.scalar_mul(mod_inv);

        let r_rotated = apply_automorphism(&r_pow_i, gen_pows[i]);
        let neg_g_pow_i = (two_n - gen_pows[i]) % two_n;
        let r_bar_rotated = apply_automorphism(&r_bar_pow_i, neg_g_pow_i);

        r_all.push(r_rotated);
        r_bar_all.push(r_bar_rotated);
    }

    let mut bold_t: Vec<Vec<Poly>> = Vec::with_capacity(num_to_pack_half - 1);
    let mut bold_t_bar: Vec<Vec<Poly>> = Vec::with_capacity(num_to_pack_half - 1);

    for i in (0..(num_to_pack_half - 1)).rev() {
        let gadget_inv = gadget_decompose(&r_all[i + 1], &pack_params.gadget);
        for (k, t_k) in gadget_inv.iter().enumerate() {
            if k < packing_key.w_all[i].len() {
                let term = t_k.mul_ntt(&packing_key.w_all[i][k], ctx);
                r_all[i] = &r_all[i] + &term;
            }
        }
        bold_t.push(gadget_inv);

        let gadget_inv_bar = gadget_decompose(&r_bar_all[i + 1], &pack_params.gadget);
        for (k, t_k) in gadget_inv_bar.iter().enumerate() {
            if k < packing_key.w_bar_all[i].len() {
                let term = t_k.mul_ntt(&packing_key.w_bar_all[i][k], ctx);
                r_bar_all[i] = &r_bar_all[i] + &term;
            }
        }
        bold_t_bar.push(gadget_inv_bar);
    }

    let bold_t_hat = gadget_decompose(&r_bar_all[0], &pack_params.gadget);

    for (k, t_k) in bold_t_hat.iter().enumerate() {
        if k < packing_key.v_mask.len() {
            let term = t_k.mul_ntt(&packing_key.v_mask[k], ctx);
            r_all[0] = &r_all[0] + &term;
        }
    }

    bold_t.reverse();
    bold_t_bar.reverse();

    let bold_t_ntt: Vec<Vec<Poly>> = bold_t
        .iter()
        .map(|row| {
            row.iter()
                .map(|p| {
                    let mut pn = p.clone();
                    pn.to_ntt(ctx);
                    pn
                })
                .collect()
        })
        .collect();

    PrecompInsPIR {
        a_hat: r_all[0].clone(),
        bold_t,
        bold_t_ntt,
        bold_t_bar,
        bold_t_hat,
        num_to_pack: n,
        ring_dim: n,
        q,
    }
}

/// Online phase: `y_all * bold_t + b_poly`, from coefficient-form inputs.
pub fn packing_online(
    precomp: &PrecompInsPIR,
    y_all: &[Vec<Poly>],
    b_poly: &Poly,
    ctx: &NttContext,
) -> RlweCiphertext {
    let n = precomp.ring_dim;
    let num_to_pack = precomp.num_to_pack;
    let moduli = precomp.a_hat.moduli();

    let mut sum_b_ntt = Poly::zero_moduli(n, moduli);
    sum_b_ntt.to_ntt(ctx);

    for i in 0..(num_to_pack - 1) {
        if i < y_all.len() && i < precomp.bold_t.len() {
            for (k, t_k) in precomp.bold_t[i].iter().enumerate() {
                if k < y_all[i].len() {
                    let mut y_ntt = y_all[i][k].clone();
                    y_ntt.to_ntt(ctx);
                    let mut t_ntt = t_k.clone();
                    t_ntt.to_ntt(ctx);

                    let term = y_ntt.mul_ntt_domain(&t_ntt, ctx);
                    sum_b_ntt = sum_b_ntt.add_ntt_domain(&term);
                }
            }
        }
    }

    sum_b_ntt.from_ntt(ctx);
    let final_b = b_poly + &sum_b_ntt;

    RlweCiphertext::from_parts(precomp.a_hat.clone(), final_b)
}

/// [`packing_online`] over `y_body` alone, in NTT form: rotation i of digit k is read
/// as `y_body_ntt[k][table_i[j]]`, because an automorphism permutes NTT slots. The
/// output is byte-identical to [`packing_online`] over `generate_rotations(y_body)`.
///
/// # Errors
///
/// The key body, `bold_t_ntt` or `b_poly` is not the shape `pack_params` describes.
/// [`packing_online`] skips what is missing, which packs a wrong plaintext.
pub fn packing_online_gather(
    precomp: &PrecompInsPIR,
    pack_params: &PackParams,
    y_body_ntt: &[Poly],
    b_poly: &Poly,
    ctx: &NttContext,
) -> Result<RlweCiphertext, String> {
    let n = pack_params.ring_dim;
    let moduli = pack_params.moduli.as_slice();
    let rotations = pack_params.num_to_pack.saturating_sub(1);
    let digits = y_body_ntt.len();

    if ctx.dimension() != n || ctx.moduli() != moduli {
        return Err(format!(
            "online packing refused: NTT context is for ring_dim {} moduli {:?}, pack \
             params are for ring_dim {n} moduli {moduli:?}",
            ctx.dimension(),
            ctx.moduli()
        ));
    }
    if precomp.num_to_pack != pack_params.num_to_pack
        || precomp.ring_dim != n
        || precomp.bold_t_ntt.len() != rotations
    {
        return Err(format!(
            "online packing refused: offline output has gamma {} at ring_dim {} with {} \
             NTT-form rotation rows, pack params expect gamma {} at ring_dim {n} with \
             {rotations}",
            precomp.num_to_pack,
            precomp.ring_dim,
            precomp.bold_t_ntt.len(),
            pack_params.num_to_pack
        ));
    }
    let on_ring = |poly: &Poly| poly.dimension() == n && poly.moduli() == moduli;
    if digits == 0
        || digits != pack_params.key_rows()
        || !y_body_ntt.iter().all(|y| y.is_ntt() && on_ring(y))
    {
        return Err(format!(
            "online packing refused: the key body must be {} NTT-form polynomials on the \
             pack-params ring, got {digits}",
            pack_params.key_rows()
        ));
    }
    if let Some(row) = precomp
        .bold_t_ntt
        .iter()
        .position(|t_i| t_i.len() != digits || !t_i.iter().all(|t| t.is_ntt() && on_ring(t)))
    {
        return Err(format!(
            "online packing refused: offline rotation row {row} is not {digits} NTT-form \
             digits on the pack-params ring"
        ));
    }
    if b_poly.is_ntt() || !on_ring(b_poly) || precomp.a_hat.is_ntt() || !on_ring(&precomp.a_hat) {
        return Err(
            "online packing refused: b and a_hat must be coefficient-form polynomials on \
             the pack-params ring"
                .to_owned(),
        );
    }
    let tables = pack_params.rotation_tables(rotations).ok_or_else(|| {
        format!(
            "online packing refused: pack params hold no full automorphism table for one \
             of the {rotations} rotations at ring_dim {n}"
        )
    })?;

    let mut sum_b_ntt = Poly::zero_moduli(n, moduli);
    sum_b_ntt.to_ntt(ctx);

    for (m, &modulus) in moduli.iter().enumerate() {
        let limb = m * n..(m + 1) * n;
        let acc = &mut sum_b_ntt.coeffs_mut()[limb.clone()];
        for (t_i, table) in precomp.bold_t_ntt.iter().zip(&tables) {
            for (y_k, t_k) in y_body_ntt.iter().zip(t_i) {
                // Sliced per limb first, so a table entry cannot reach another limb.
                let y = &y_k.coeffs()[limb.clone()];
                let t = &t_k.coeffs()[limb.clone()];
                for ((acc_j, &source), &t_j) in acc.iter_mut().zip(*table).zip(t) {
                    let product = ctx.pointwise_mul_single_at(y[source], t_j, m);
                    *acc_j = csubq(*acc_j + product, modulus);
                }
            }
        }
    }

    sum_b_ntt.from_ntt(ctx);
    let final_b = b_poly + &sum_b_ntt;

    Ok(RlweCiphertext::from_parts(precomp.a_hat.clone(), final_b))
}

/// `y_all[i] = tau_{g^i}(y_body)` for i in `0..gamma-1`.
pub fn generate_rotations(pack_params: &PackParams, y_body: &[Poly]) -> Vec<Vec<Poly>> {
    let num_to_pack = pack_params.num_to_pack;
    let gen_pows = &pack_params.gen_pows;
    if num_to_pack <= 1 {
        return Vec::new();
    }
    let required = num_to_pack - 1;
    assert!(
        gen_pows.len() >= required,
        "generator powers must have at least {} entries, got {}",
        required,
        gen_pows.len()
    );

    let mut y_all = Vec::with_capacity(required);
    for &g_pow_i in &gen_pows[..required] {
        let rotated: Vec<Poly> = y_body
            .iter()
            .map(|poly| apply_automorphism(poly, g_pow_i))
            .collect();
        y_all.push(rotated);
    }
    y_all
}

/// Offline and online phases in one call. Production callers split them so the
/// offline result can be cached.
pub fn pack_inspiring(
    lwe_ciphertexts: &[LweCiphertext],
    pack_params: &PackParams,
    packing_key: &PackingKeyBody,
    y_body: &[Poly],
    params: &InspireParams,
) -> RlweCiphertext {
    let d = params.ring_dim;
    let ctx = params.ntt_context();
    let moduli = params.moduli();

    let a_ct_tilde: Vec<Poly> = lwe_ciphertexts
        .iter()
        .map(|lwe| Poly::from_coeffs_moduli(lwe.a.clone(), params.moduli()))
        .collect();

    let precomp = packing_offline(pack_params, packing_key, &a_ct_tilde, &ctx);

    let y_all = generate_rotations(pack_params, y_body);

    let mut b_coeffs = vec![0u64; d];
    for (i, lwe) in lwe_ciphertexts.iter().enumerate() {
        if i < d {
            b_coeffs[i] = lwe.b;
        }
    }
    let b_poly = Poly::from_coeffs_moduli(b_coeffs, moduli);

    packing_online(&precomp, &y_all, &b_poly, &ctx)
}

/// Powers of the generator mod 2d, forward and inverse.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneratorPowers {
    /// g^i mod 2d.
    pub powers: Vec<usize>,
    /// g^{-i} mod 2d.
    pub inv_powers: Vec<usize>,
    /// Multiplicative group generator.
    pub generator: usize,
    /// Ring dimension.
    pub ring_dim: usize,
}

impl GeneratorPowers {
    /// Build the table; `d` must be a power of two, or the canonical generator is
    /// not invertible mod 2d.
    pub fn try_new(d: usize) -> Result<Self, PackParamsError> {
        if !d.is_power_of_two() {
            return Err(PackParamsError::RingDimNotPowerOfTwo { ring_dim: d });
        }
        let two_d = 2 * d;
        // Canonical generator of the odd part of (Z/2dZ)^*.
        let g = 5usize;

        let mut powers = Vec::with_capacity(d);
        let mut val = 1usize;
        for _ in 0..d {
            powers.push(val);
            val = (val * g) % two_d;
        }

        let g_inv = mod_inverse_usize(g, two_d)
            .ok_or(PackParamsError::RingDimNotPowerOfTwo { ring_dim: d })?;
        let mut inv_powers = Vec::with_capacity(d);
        val = 1;
        for _ in 0..d {
            inv_powers.push(val);
            val = (val * g_inv) % two_d;
        }

        Ok(Self {
            powers,
            inv_powers,
            generator: g,
            ring_dim: d,
        })
    }

    /// [`Self::try_new`] for callers whose signature cannot carry the failure; panics
    /// where that one errors.
    #[allow(clippy::panic, reason = "deprecated shim; in-tree callers use try_new")]
    #[deprecated(
        since = "0.1.0",
        note = "panics on a degenerate ring dimension; use `try_new`, which returns PackParamsError"
    )]
    #[must_use]
    pub fn new(d: usize) -> Self {
        match Self::try_new(d) {
            Ok(table) => table,
            Err(e) => panic!("{e}"),
        }
    }

    /// Number of powers, equal to the ring dimension.
    pub fn order(&self) -> usize {
        self.powers.len()
    }

    /// g^i mod 2d, wrapping cyclically.
    #[inline]
    pub fn pow(&self, i: usize) -> usize {
        self.powers[i % self.powers.len()]
    }

    /// g^{-i} mod 2d, wrapping cyclically.
    #[inline]
    pub fn inv_pow(&self, i: usize) -> usize {
        self.inv_powers[i % self.inv_powers.len()]
    }

    /// The generator.
    #[inline]
    pub fn generator(&self) -> usize {
        self.generator
    }
}

/// Precomputation shape for the `*_legacy` packing entry points.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InspiringPrecomputation {
    /// Generator powers.
    pub gen_pows: GeneratorPowers,
    /// Rotations of the base key-switching matrix.
    pub rotated_k_g: RotatedKsMatrix,
    /// R polynomials from the backward recursion.
    pub r_polys: Vec<Poly>,
    /// Gadget-decomposed T values.
    pub bold_t: Vec<Vec<Poly>>,
    /// gamma.
    pub num_to_pack: usize,
    /// Ring dimension.
    pub ring_dim: usize,
    /// Ciphertext modulus.
    pub q: u64,
}

/// Key-switching matrix rotated once per automorphism index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RotatedKsMatrix {
    /// Rows per automorphism index.
    pub rotations: Vec<Vec<RlweCiphertext>>,
    /// gamma.
    pub num_to_pack: usize,
    /// Gadget decomposition parameters.
    pub gadget: GadgetVector,
}

impl RotatedKsMatrix {
    /// Rotate `k_g` by each generator power.
    pub fn generate(
        k_g: &crate::ks::KeySwitchingMatrix,
        gen_pows: &GeneratorPowers,
        num_to_pack: usize,
    ) -> Self {
        let num_rotations = if num_to_pack > 0 { num_to_pack - 1 } else { 0 };
        let mut rotations = Vec::with_capacity(num_rotations);

        for i in 0..num_rotations {
            let g_pow_i = gen_pows.pow(i);
            let rotated_rows: Vec<RlweCiphertext> = k_g
                .rows
                .iter()
                .map(|row| {
                    let a_rot = apply_automorphism(&row.a, g_pow_i);
                    let b_rot = apply_automorphism(&row.b, g_pow_i);
                    RlweCiphertext::from_parts(a_rot, b_rot)
                })
                .collect();
            rotations.push(rotated_rows);
        }

        Self {
            rotations,
            num_to_pack,
            gadget: k_g.gadget.clone(),
        }
    }

    /// Rows at rotation index `i`.
    pub fn get_rotation(&self, i: usize) -> &[RlweCiphertext] {
        &self.rotations[i]
    }
}

/// Offline precomputation for the `*_legacy` packing entry points.
pub fn precompute_inspiring(
    crs_a_vectors: &[Vec<u64>],
    k_g: &crate::ks::KeySwitchingMatrix,
    params: &InspireParams,
) -> Result<InspiringPrecomputation, PackParamsError> {
    // K_g stands in for the packing key here and carries a row for every digit.
    if params.approximate_packing_gadget {
        return Err(PackParamsError::ApproximateGadgetUnsupported {
            reason: "the legacy precomputation packs under a full key-switching matrix",
        });
    }
    let d = params.ring_dim;
    let q = params.q;
    let num_to_pack = crs_a_vectors.len();
    let ctx = params.ntt_context();

    let pack_params = PackParams::try_new(params, num_to_pack)?;

    let gen_pows = GeneratorPowers::try_new(d)?;
    let rotated_k_g = RotatedKsMatrix::generate(k_g, &gen_pows, num_to_pack);

    let a_polys: Vec<Poly> = crs_a_vectors
        .iter()
        .map(|a| Poly::from_coeffs_moduli(a.clone(), params.moduli()))
        .collect();

    // Approximates w_all with the K_g rows; real callers use
    // `OfflinePackingKeys::generate`.
    let w_all: Vec<Vec<Poly>> = rotated_k_g
        .rotations
        .iter()
        .map(|rows| rows.iter().map(|r| r.b.clone()).collect())
        .collect();

    let w_all_ntt: Vec<Vec<Poly>> = w_all
        .iter()
        .map(|row| {
            row.iter()
                .map(|p| {
                    let mut pn = p.clone();
                    pn.to_ntt(&ctx);
                    pn
                })
                .collect()
        })
        .collect();

    let packing_key = OfflinePackingKeys {
        w_seed: [0u8; 32],
        v_seed: [0u8; 32],
        w_mask: vec![],
        v_mask: vec![],
        w_all,
        w_all_ntt,
        w_bar_all: vec![],
        w_bar_all_ntt: vec![],
        full_key: false,
    };

    let precomp = packing_offline(&pack_params, &packing_key, &a_polys, &ctx);

    Ok(InspiringPrecomputation {
        gen_pows,
        rotated_k_g,
        r_polys: vec![precomp.a_hat],
        bold_t: precomp.bold_t,
        num_to_pack,
        ring_dim: d,
        q,
    })
}

/// Pack against an [`InspiringPrecomputation`], using the rotated K_g rows in place
/// of real client rotations.
pub fn pack_inspiring_legacy(
    lwe_ciphertexts: &[LweCiphertext],
    precomp: &InspiringPrecomputation,
    _k_g: &crate::ks::KeySwitchingMatrix,
    params: &InspireParams,
) -> RlweCiphertext {
    let d = params.ring_dim;
    let q = params.q;
    let n = lwe_ciphertexts.len();
    let ctx = params.ntt_context();
    let moduli = params.moduli();

    assert_eq!(
        n, precomp.num_to_pack,
        "Number of LWEs must match precomputation"
    );

    let y_all: Vec<Vec<Poly>> = precomp
        .rotated_k_g
        .rotations
        .iter()
        .map(|rows| rows.iter().map(|r| r.b.clone()).collect())
        .collect();

    let mut b_coeffs = vec![0u64; d];
    for (i, lwe) in lwe_ciphertexts.iter().enumerate() {
        if i < d {
            b_coeffs[i] = lwe.b;
        }
    }
    let b_poly = Poly::from_coeffs_moduli(b_coeffs, moduli);

    let precomp_new = PrecompInsPIR {
        a_hat: precomp
            .r_polys
            .first()
            .cloned()
            .unwrap_or_else(|| Poly::zero_moduli(d, moduli)),
        bold_t: precomp.bold_t.clone(),
        bold_t_ntt: vec![],
        bold_t_bar: vec![],
        bold_t_hat: vec![],
        num_to_pack: n,
        ring_dim: d,
        q,
    };

    packing_online(&precomp_new, &y_all, &b_poly, &ctx)
}

/// Precompute and pack in one call; the ciphertext count must be a legal width.
pub fn pack_inspiring_partial(
    lwe_ciphertexts: &[LweCiphertext],
    k_g: &crate::ks::KeySwitchingMatrix,
    params: &InspireParams,
) -> Result<RlweCiphertext, PackParamsError> {
    let crs_a_vectors: Vec<Vec<u64>> = lwe_ciphertexts.iter().map(|lwe| lwe.a.clone()).collect();
    let precomp = precompute_inspiring(&crs_a_vectors, k_g, params)?;
    Ok(pack_inspiring_legacy(
        lwe_ciphertexts,
        &precomp,
        k_g,
        params,
    ))
}

/// Delegates to [`pack_inspiring_legacy`]; `k_h` is ignored, so no conjugate branch
/// is applied.
pub fn pack_inspiring_full(
    lwe_ciphertexts: &[LweCiphertext],
    precomp: &InspiringPrecomputation,
    k_g: &crate::ks::KeySwitchingMatrix,
    _k_h: &crate::ks::KeySwitchingMatrix,
    params: &InspireParams,
) -> RlweCiphertext {
    pack_inspiring_legacy(lwe_ciphertexts, precomp, k_g, params)
}

fn mod_inverse_u64(a: u64, m: u64) -> Option<u64> {
    let (g, x, _) = extended_gcd_i128(a as i128, m as i128);
    if g == 1 {
        Some(((x % m as i128 + m as i128) % m as i128) as u64)
    } else {
        None
    }
}

fn mod_inverse_usize(a: usize, m: usize) -> Option<usize> {
    let (g, x, _) = extended_gcd_i64(a as i64, m as i64);
    if g == 1 {
        Some(((x % m as i64 + m as i64) % m as i64) as usize)
    } else {
        None
    }
}

fn extended_gcd_i128(a: i128, b: i128) -> (i128, i128, i128) {
    if a == 0 {
        (b, 0, 1)
    } else {
        let (g, x, y) = extended_gcd_i128(b % a, a);
        (g, y - (b / a) * x, x)
    }
}

fn extended_gcd_i64(a: i64, b: i64) -> (i64, i64, i64) {
    if a == 0 {
        (b, 0, 1)
    } else {
        let (g, x, y) = extended_gcd_i64(b % a, a);
        (g, y - (b / a) * x, x)
    }
}

#[cfg(test)]
mod automorph_tables_oracle;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ks::generate_automorphism_ks_matrix;
    use crate::math::GaussianSampler;
    use crate::rlwe::RlweSecretKey;

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
            approximate_packing_gadget: false,
            security_level: crate::params::SecurityLevel::Bits128,
        }
    }

    fn legacy_error_poly(
        dim: usize,
        q: u64,
        moduli: &[u64],
        sampler: &mut GaussianSampler,
    ) -> Poly {
        let mut error_coeffs = vec![0u64; dim];
        for coeff in &mut error_coeffs {
            let sample = sampler.sample();
            *coeff = if sample >= 0 {
                sample as u64 % q
            } else {
                (q as i64 + (sample % q as i64)) as u64
            };
        }
        Poly::from_coeffs_moduli(error_coeffs, moduli)
    }

    #[test]
    fn shared_gaussian_poly_is_byte_identical_to_the_legacy_ksk_lift() {
        let modulus_sets = [
            vec![1_152_921_504_606_830_593],
            vec![268_369_921, 249_561_089],
        ];
        for moduli in modulus_sets {
            let q = moduli.iter().product();
            let mut legacy_sampler = GaussianSampler::with_seed(6.4, 0x6374_2d6b_736b);
            let mut shared_sampler = legacy_sampler.clone();
            let legacy = legacy_error_poly(256, q, &moduli, &mut legacy_sampler);
            let shared = Poly::sample_gaussian_moduli(256, &moduli, &mut shared_sampler);
            assert_eq!(shared.coeffs(), legacy.coeffs(), "moduli={moduli:?}");
            assert_eq!(shared_sampler.sample(), legacy_sampler.sample());
        }
    }

    /// The recursion as its formula reads, with a forward transform per digit.
    fn fold_rotations_transforming_every_digit(
        pack_params: &PackParams,
        packing_key: &PackingKeyBody,
        mut r_all: Vec<Poly>,
        ctx: &NttContext,
    ) -> (Poly, Vec<Vec<Poly>>, Vec<Vec<Poly>>) {
        let mut bold_t = Vec::new();
        let mut bold_t_ntt = Vec::new();
        for i in (0..pack_params.num_to_pack - 1).rev() {
            let digits = gadget_decompose(&r_all[i + 1].from_ntt_new(ctx), &pack_params.gadget);
            let digits_ntt: Vec<Poly> = digits.iter().map(|t| t.to_ntt_new(ctx)).collect();
            for (t_ntt, w_ntt) in digits_ntt.iter().zip(&packing_key.w_all_ntt[i]) {
                r_all[i] = r_all[i].add_ntt_domain(&t_ntt.mul_ntt_domain(w_ntt, ctx));
            }
            bold_t.push(digits);
            bold_t_ntt.push(digits_ntt);
        }
        bold_t.reverse();
        bold_t_ntt.reverse();
        (r_all[0].from_ntt_new(ctx), bold_t, bold_t_ntt)
    }

    /// Every shipped gadget spans its modulus, so every shipped set takes the derived
    /// digit; the one- and two-digit gadgets here do not, and must not.
    #[test]
    fn the_last_digit_is_derived_only_when_the_gadget_spans_the_modulus() {
        use rand::Rng;
        let modulus_sets = [
            vec![crate::math::DEFAULT_Q],
            crate::params::DEFAULT_CRT_MODULI.to_vec(),
            crate::params::DEFAULT_Q_2CRT_30BIT.to_vec(),
            vec![67_043_329, 132_120_577],
        ];
        let mut rng = ChaCha20Rng::seed_from_u64(0x4c44_4947);
        let (mut derived, mut transformed) = (0, 0);
        for moduli in &modulus_sets {
            for gadget_len in 1..=4usize {
                let params = InspireParams {
                    crt_moduli: moduli.clone(),
                    q: moduli.iter().product(),
                    query_gadget_len: gadget_len,
                    packing_gadget_len: gadget_len,
                    ..test_params()
                };
                let ctx = params.ntt_context();
                let spans = (1u128 << (20 * gadget_len)) >= u128::from(params.q);
                for gamma in [2usize, 16] {
                    let pack_params = PackParams::try_new(&params, gamma).unwrap();
                    let keys = OfflinePackingKeys::generate(&pack_params, [7; 32]);
                    assert_eq!(
                        LastDigitNtt::derive(&pack_params, &ctx).is_some(),
                        spans,
                        "moduli {moduli:?} gadget_len {gadget_len}"
                    );
                    for pattern in 0..4 {
                        let r_all: Vec<Poly> = (0..gamma)
                            .map(|_| {
                                let mut words = Vec::with_capacity(256 * moduli.len());
                                for &modulus in moduli {
                                    let hot = rng.gen_range(0..256);
                                    words.extend((0..256).map(|slot| match pattern {
                                        0 => rng.gen_range(0..modulus),
                                        1 => modulus - 1,
                                        2 => 0,
                                        _ => u64::from(slot == hot) * (modulus - 1),
                                    }));
                                }
                                let mut poly = Poly::from_crt_coeffs(words, moduli);
                                poly.force_ntt_domain();
                                poly
                            })
                            .collect();
                        let (a_hat, bold_t, bold_t_ntt) = fold_rotations_transforming_every_digit(
                            &pack_params,
                            &keys,
                            r_all.clone(),
                            &ctx,
                        );
                        let folded = fold_rotations_backward(&pack_params, &keys, r_all, &ctx);
                        assert!(
                            folded.a_hat == a_hat
                                && folded.bold_t == bold_t
                                && folded.bold_t_ntt == bold_t_ntt,
                            "moduli {moduli:?} gadget_len {gadget_len} gamma {gamma} pattern {pattern}"
                        );
                        if spans {
                            derived += 1;
                        } else {
                            transformed += 1;
                        }
                    }
                }
            }
        }
        assert_eq!((derived, transformed), (64, 64));
    }

    /// `round(coeff / z^dropped)` split into base-z digits by plain division, the top
    /// digit keeping whatever is left.
    fn rounded_digits_by_definition(coeff: u64, gadget: &GadgetVector, dropped: usize) -> Vec<u64> {
        let base = u128::from(gadget.base);
        let unit = base.pow(dropped as u32);
        let mut rounded = (u128::from(coeff) + unit / 2) / unit;
        let rows = gadget.len - dropped;
        (0..rows)
            .map(|row| {
                let digit = if row + 1 == rows {
                    rounded
                } else {
                    rounded % base
                };
                rounded /= base;
                digit as u64
            })
            .collect()
    }

    #[test]
    fn rounded_digits_recompose_within_half_a_dropped_unit() {
        use rand::Rng;
        let two_limb = |pair: [u64; 2]| pair[0] * pair[1];
        let gadgets = [
            (1u64 << 20, 3usize, vec![crate::math::DEFAULT_Q]),
            (1 << 20, 3, crate::params::DEFAULT_CRT_MODULI.to_vec()),
            (1 << 20, 3, crate::params::DEFAULT_Q_2CRT_30BIT.to_vec()),
            (1 << 19, 3, vec![67_043_329, 132_120_577]),
            (1 << 19, 4, crate::params::DEFAULT_Q_2CRT_30BIT.to_vec()),
            (1 << 30, 2, vec![crate::math::DEFAULT_Q]),
            (1000, 7, vec![crate::math::DEFAULT_Q]),
            (3, 38, vec![crate::math::DEFAULT_Q]),
        ];
        let mut rng = ChaCha20Rng::seed_from_u64(0x524f_554e);
        let mut carried = 0;
        for (base, len, moduli) in gadgets {
            let q = if moduli.len() == 2 {
                two_limb([moduli[0], moduli[1]])
            } else {
                moduli[0]
            };
            assert!(u128::from(base).pow(len as u32) >= u128::from(q));
            let gadget = GadgetVector::new(base, len, q);
            for dropped in 1..len.min(3) {
                let unit = u128::from(base).pow(dropped as u32);
                let half = (unit / 2) as u64;
                let mut coeffs: Vec<u64> = [
                    0,
                    1,
                    half.saturating_sub(1),
                    half,
                    half + 1,
                    unit as u64 - 1,
                    unit as u64,
                    q / 2,
                    q - half - 1,
                    q - half,
                    q - 2,
                    q - 1,
                ]
                .into_iter()
                .filter(|&coeff| coeff < q)
                .collect();
                coeffs.extend((0..500).map(|_| rng.gen_range(0..q)));
                coeffs.resize(coeffs.len().next_power_of_two(), q - 1);

                let rows = rounded_digit_rows(&coeffs, &gadget, dropped, true);
                assert_eq!(
                    rows,
                    rounded_digit_rows(&coeffs, &gadget, dropped, false),
                    "base {base} len {len} dropped {dropped}: shift form"
                );
                assert_eq!(rows.len(), len - dropped);
                let top_bound = u128::from(q) / u128::from(base).pow(len as u32 - 1) + 1;
                for (j, &coeff) in coeffs.iter().enumerate() {
                    let digits: Vec<u64> = rows.iter().map(|row| row[j]).collect();
                    assert_eq!(
                        digits,
                        rounded_digits_by_definition(coeff, &gadget, dropped)
                    );
                    let (top, lower) = digits.split_last().unwrap();
                    assert!(lower.iter().all(|&digit| digit < base));
                    assert!(u128::from(*top) <= top_bound);
                    carried += usize::from(*top == base);
                    let recomposed: u128 = digits
                        .iter()
                        .enumerate()
                        .map(|(row, &digit)| {
                            u128::from(digit) * u128::from(base).pow((dropped + row) as u32)
                        })
                        .sum();
                    let residual = i128::from(coeff) - recomposed as i128;
                    assert!(
                        residual.unsigned_abs() <= unit / 2,
                        "base {base} len {len} dropped {dropped} coeff {coeff}: {residual}"
                    );
                }

                // The polynomial form reads composed coefficients and writes each digit
                // back across the limbs.
                let poly = Poly::from_coeffs_moduli(coeffs.clone(), &moduli);
                let digit_polys = gadget_decompose_rounded(&poly, &gadget, dropped);
                for (row, digit_poly) in rows.iter().zip(&digit_polys) {
                    assert_eq!(digit_poly.moduli(), moduli.as_slice());
                    let read: Vec<u64> = (0..coeffs.len()).map(|j| digit_poly.coeff(j)).collect();
                    assert_eq!(&read, row, "base {base} len {len} dropped {dropped}");
                }
            }
        }
        // q - 1 rounds up to z^len at the shipped gadget: the carry is exercised.
        assert!(carried > 0);
    }

    /// A parameter set with the approximate gadget folds with the rounded digits, one
    /// mask row per kept digit, and never takes the derived last digit: its digits do
    /// not recompose to the coefficient.
    #[test]
    fn an_approximate_gadget_folds_its_kept_digits_and_derives_none() {
        use rand::Rng;
        let modulus_sets = [
            vec![crate::math::DEFAULT_Q],
            crate::params::DEFAULT_CRT_MODULI.to_vec(),
            crate::params::DEFAULT_Q_2CRT_30BIT.to_vec(),
            vec![67_043_329, 132_120_577],
        ];
        let mut rng = ChaCha20Rng::seed_from_u64(0x4150_5258);
        for moduli in &modulus_sets {
            for gadget_len in [3usize, 4] {
                let exact = InspireParams {
                    crt_moduli: moduli.clone(),
                    q: moduli.iter().product(),
                    query_gadget_len: gadget_len,
                    packing_gadget_len: gadget_len,
                    ..test_params()
                };
                let params = InspireParams {
                    approximate_packing_gadget: true,
                    ..exact.clone()
                };
                let ctx = params.ntt_context();
                for gamma in [2usize, 16] {
                    let pack_params = PackParams::try_new(&params, gamma).unwrap();
                    assert_eq!(
                        (pack_params.dropped_digits, pack_params.key_rows()),
                        (1, gadget_len - 1)
                    );
                    assert!(LastDigitNtt::derive(&pack_params, &ctx).is_none());
                    let keys = OfflinePackingKeys::generate(&pack_params, [7; 32]);
                    assert_eq!(keys.w_mask.len(), gadget_len - 1);
                    // Each kept digit keeps the mask the exact gadget gives it.
                    let exact_pack = PackParams::try_new(&exact, gamma).unwrap();
                    let exact_keys = OfflinePackingKeys::generate(&exact_pack, [7; 32]);
                    assert_eq!(keys.w_mask, exact_keys.w_mask[1..]);
                    assert_eq!(keys.w_all_ntt[0], exact_keys.w_all_ntt[0][1..]);

                    let r_all: Vec<Poly> = (0..gamma)
                        .map(|_| {
                            let mut words = Vec::with_capacity(256 * moduli.len());
                            for &modulus in moduli {
                                words.extend((0..256).map(|_| rng.gen_range(0..modulus)));
                            }
                            let mut poly = Poly::from_crt_coeffs(words, moduli);
                            poly.force_ntt_domain();
                            poly
                        })
                        .collect();

                    let mut expected = r_all.clone();
                    let mut expected_t = Vec::new();
                    for i in (0..gamma - 1).rev() {
                        let coeffs = expected[i + 1].from_ntt_new(&ctx);
                        let mut digits = vec![vec![0u64; 256]; gadget_len - 1];
                        for j in 0..256 {
                            let split = rounded_digits_by_definition(
                                coeffs.coeff(j),
                                &pack_params.gadget,
                                1,
                            );
                            for (row, digit) in digits.iter_mut().zip(split) {
                                row[j] = digit;
                            }
                        }
                        let digits: Vec<Poly> = digits
                            .into_iter()
                            .map(|row| Poly::from_coeffs_moduli(row, moduli))
                            .collect();
                        for (digit, w_ntt) in digits.iter().zip(&keys.w_all_ntt[i]) {
                            let term = digit.to_ntt_new(&ctx).mul_ntt_domain(w_ntt, &ctx);
                            expected[i] = expected[i].add_ntt_domain(&term);
                        }
                        expected_t.push(digits);
                    }
                    expected_t.reverse();

                    let folded = fold_rotations_backward(&pack_params, &keys, r_all, &ctx);
                    assert!(
                        folded.a_hat == expected[0].from_ntt_new(&ctx)
                            && folded.bold_t == expected_t
                            && folded
                                .bold_t_ntt
                                .iter()
                                .all(|row| row.len() == gadget_len - 1),
                        "moduli {moduli:?} gadget_len {gadget_len} gamma {gamma}"
                    );
                    let transformed: Vec<Vec<Poly>> = expected_t
                        .iter()
                        .map(|row| row.iter().map(|digit| digit.to_ntt_new(&ctx)).collect())
                        .collect();
                    assert_eq!(folded.bold_t_ntt, transformed);
                }
            }
        }
    }

    #[test]
    fn the_approximate_gadget_is_refused_where_it_is_undefined() {
        let one_digit = InspireParams {
            approximate_packing_gadget: true,
            packing_gadget_len: 1,
            ..test_params()
        };
        let refused = |result: Result<PackParams, PackParamsError>| {
            matches!(
                result,
                Err(PackParamsError::ApproximateGadgetUnsupported { .. })
            )
        };
        assert!(refused(PackParams::try_new(&one_digit, 16)));

        let approximate = InspireParams {
            approximate_packing_gadget: true,
            ..test_params()
        };
        assert!(PackParams::try_new(&approximate, 16).is_ok());
        assert!(refused(PackParams::try_new_full(&approximate)));
        assert!(PackParams::try_new_full(&test_params()).is_ok());

        let mut sampler = GaussianSampler::with_seed(approximate.sigma, 3);
        let sk = RlweSecretKey::generate(&one_digit, &mut sampler);
        assert!(matches!(
            ClientPackingKeys::generate(&sk, &one_digit, 16, [1; 32], &mut sampler),
            Err(PackParamsError::ApproximateGadgetUnsupported { .. })
        ));

        let ctx = approximate.ntt_context();
        let gadget = GadgetVector::new(
            approximate.gadget_base,
            approximate.packing_gadget_len,
            approximate.q,
        );
        let k_g = generate_automorphism_ks_matrix(&sk, 3, &gadget, &mut sampler, &ctx);
        let a_vectors: Vec<Vec<u64>> = (0..8)
            .map(|_| {
                Poly::random_moduli(256, approximate.moduli())
                    .coeffs()
                    .to_vec()
            })
            .collect();
        assert!(matches!(
            precompute_inspiring(&a_vectors, &k_g, &approximate),
            Err(PackParamsError::ApproximateGadgetUnsupported { .. })
        ));
        assert!(precompute_inspiring(&a_vectors, &k_g, &test_params()).is_ok());
    }

    /// An approximate key is an exact key less its lowest row: row r hides
    /// `tau_g(s) * z^(r + 1)` under the mask the exact key gives that digit, so the two
    /// keys of one secret differ by key error alone, digit for digit.
    #[test]
    fn an_approximate_key_is_the_exact_key_without_its_lowest_row() {
        let exact_params = test_params();
        let params = InspireParams {
            approximate_packing_gadget: true,
            ..exact_params.clone()
        };
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 11);
        let sk = RlweSecretKey::generate(&params, &mut sampler);
        let gamma = 16;
        let keys = ClientPackingKeys::generate(&sk, &params, gamma, [5; 32], &mut sampler).unwrap();
        let exact =
            ClientPackingKeys::generate(&sk, &exact_params, gamma, [5; 32], &mut sampler).unwrap();
        assert_eq!((keys.y_body.len(), exact.y_body.len()), (2, 3));

        let q = params.q;
        let worst = |poly: &Poly| {
            (0..256)
                .map(|j| poly.coeff(j).min(q - poly.coeff(j)))
                .max()
                .unwrap()
        };
        let w_mask = generate_mask_from_seed([5; 32], 256, q, params.moduli(), 3);
        let generator = PackParams::partial_generator(256, gamma);
        let tau_s = apply_automorphism(&sk.poly, generator);
        for (row, y) in keys.y_body.iter().enumerate() {
            let digit = row + 1;
            let power = (1u64 << 20).pow(digit as u32);
            let unmasked = y + &sk.poly.mul_ntt(&w_mask[digit], &ctx);
            let error = &unmasked + &tau_s.scalar_mul(power).negate();
            assert!(
                worst(&error) <= 39,
                "row {row}: {} is past the sampler tailcut",
                worst(&error)
            );
            // Same digit, same mask: two errors apart. Any other pairing leaves a
            // mask or a multiple of the secret uncancelled.
            let same_digit = y + &exact.y_body[digit].negate();
            assert!(
                worst(&same_digit) <= 78,
                "row {row}: {}",
                worst(&same_digit)
            );
            let same_row = y + &exact.y_body[row].negate();
            assert!(worst(&same_row) > q / 8, "row {row}");
        }
    }

    #[test]
    fn test_pack_params_generator() {
        let params = test_params();
        let n = params.ring_dim;

        let pack_8 = PackParams::try_new(&params, 8).expect("8 is a legal width at d=256");
        assert_eq!(pack_8.generator, (2 * n / 8) + 1);

        let pack_16 = PackParams::try_new(&params, 16).expect("16 is a legal width at d=256");
        assert_eq!(pack_16.generator, (2 * n / 16) + 1);

        let pack_n = PackParams::try_new_full(&params).unwrap();
        assert_eq!(pack_n.generator, 5);
        assert_eq!(pack_n.num_to_pack, n);
    }

    #[test]
    fn test_mod_inverse() {
        let q = 1152921504606830593u64;

        let inv_8 = mod_inverse_u64(8, q).unwrap();
        assert_eq!((8u128 * inv_8 as u128) % q as u128, 1);

        let inv_16 = mod_inverse_u64(16, q).unwrap();
        assert_eq!((16u128 * inv_16 as u128) % q as u128, 1);
    }

    #[test]
    fn test_generator_powers() {
        let params = test_params();
        let pack_params = PackParams::try_new(&params, 16).expect("16 is a legal width at d=256");
        let n = params.ring_dim;
        let two_n = 2 * n;

        assert_eq!(pack_params.gen_pows[0], 1);
        assert_eq!(pack_params.gen_pows[1], pack_params.generator);

        let g = pack_params.generator;
        for i in 0..10 {
            let expected = pow_mod(g, i, two_n);
            assert_eq!(pack_params.gen_pows[i], expected);
        }
    }

    fn pow_mod(base: usize, exp: usize, m: usize) -> usize {
        let mut result = 1usize;
        let mut b = base % m;
        let mut e = exp;
        while e > 0 {
            if e % 2 == 1 {
                result = (result * b) % m;
            }
            e /= 2;
            b = (b * b) % m;
        }
        result
    }

    #[test]
    fn test_legacy_api_compatibility() {
        let params = test_params();
        let d = params.ring_dim;
        let q = params.q;
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let sk = RlweSecretKey::generate(&params, &mut sampler);
        let gadget = GadgetVector::new(params.gadget_base, params.packing_gadget_len, q);

        let g = 3;
        let k_g = generate_automorphism_ks_matrix(&sk, g, &gadget, &mut sampler, &ctx);

        let num_to_pack = 8;
        let crs_a_vectors: Vec<Vec<u64>> = (0..num_to_pack)
            .map(|_| Poly::random_moduli(d, params.moduli()).coeffs().to_vec())
            .collect();

        let precomp = precompute_inspiring(&crs_a_vectors, &k_g, &params)
            .expect("8 is a legal width at d=256");

        assert_eq!(precomp.num_to_pack, num_to_pack);
        assert_eq!(precomp.bold_t.len(), num_to_pack - 1);
    }

    #[test]
    fn test_packing_offline_dimensions() {
        let params = test_params();
        let d = params.ring_dim;
        let ctx = params.ntt_context();
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

        let _sk = RlweSecretKey::generate(&params, &mut sampler);
        let num_to_pack = 8;

        let pack_params =
            PackParams::try_new(&params, num_to_pack).expect("8 is a legal width at d=256");

        let w_seed = [42u8; 32];
        let packing_key = OfflinePackingKeys::generate(&pack_params, w_seed);

        let a_polys: Vec<Poly> = (0..num_to_pack)
            .map(|_| Poly::random_moduli(d, params.moduli()))
            .collect();

        let precomp = packing_offline(&pack_params, &packing_key, &a_polys, &ctx);

        assert_eq!(precomp.num_to_pack, num_to_pack);
        assert_eq!(precomp.ring_dim, d);
        assert_eq!(precomp.bold_t.len(), num_to_pack - 1);
        assert_eq!(precomp.a_hat.dimension(), d);
    }
}
