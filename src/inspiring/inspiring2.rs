//! InspiRING packing: gamma LWE ciphertexts into one RLWE with 2 key-switching
//! matrices where tree packing needs log(d). Indexing LWE samples by a multiplicative
//! generator of Z*_{2d} lets a backward recursion move nearly all work offline.
//!
//! Key split across the parties, all derived from the public `w_seed`:
//! - server offline holds `w_mask` and its rotations `w_all`, giving `a_hat` + `bold_t`
//! - the client's query carries `y_body = tau_g(s)*G - s*w_mask + error` and rotations
//! - server online computes `y_all * bold_t + b_poly`
//!
//! Ported from eprint 2025/1352 and
//! <https://github.com/google/private-membership/tree/main/research/InsPIRe>.

use crate::lwe::LweCiphertext;
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
        Self::build(params, num_to_pack, (2 * n / num_to_pack) + 1)
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

        let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, q);

        let automorph_tables = generate_automorph_tables(n, &moduli, &ctx);

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
        })
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
}

/// Permutation tables satisfying `NTT(tau_t(poly))[i] = NTT(poly)[table[i]]` for every
/// odd t in [1, 2n), which turns an automorphism into an O(n) index shuffle.
///
/// Recovered by matching NTT coefficients of a random polynomial against its
/// automorphed image, retrying whenever a value is not unique.
fn generate_automorph_tables(n: usize, moduli: &[u64], ctx: &NttContext) -> Vec<Vec<usize>> {
    let two_n = 2 * n;
    let mut tables = Vec::with_capacity(n);

    for t in (1..two_n).step_by(2) {
        let mut table = vec![0usize; n];

        loop {
            let poly = Poly::random_moduli(n, moduli);
            let mut poly_ntt = poly.clone();
            poly_ntt.to_ntt(ctx);

            let poly_auto = apply_automorphism(&poly, t);
            let mut poly_auto_ntt = poly_auto.clone();
            poly_auto_ntt.to_ntt(ctx);

            let mut must_redo = false;

            for i in 0..n {
                let orig_val = poly_ntt.coeffs()[i];
                let mut found = 0usize;
                let mut count = 0usize;

                for j in 0..n {
                    if poly_auto_ntt.coeffs()[j] == orig_val {
                        count += 1;
                        found = j;
                    }
                }

                if count != 1 {
                    must_redo = true;
                    break;
                }

                table[found] = i;
            }

            if !must_redo {
                break;
            }
        }

        tables.push(table);
    }

    tables
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
    /// gadget_len random polynomials expanded from `w_seed`.
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
        let gadget_len = pack_params.gadget.len;
        let ctx = NttContext::with_moduli(n, &pack_params.moduli);

        // Generate w_mask from seed and convert to NTT form
        let w_mask = generate_mask_from_seed(w_seed, n, q, &pack_params.moduli, gadget_len);
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

/// Client-side packing keys shipped with a query. Only `y_body` and `z_body` cross the
/// wire; the rotations are derived on both sides.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClientPackingKeys {
    /// `y_body[k] = tau_g(s)*g^k - s*w_mask[k] + error`.
    pub y_body: Vec<Poly>,
    /// Conjugation body, full packing only.
    ///
    /// Never `skip_serializing_if`: bincode is positional, so an omitted length
    /// prefix shifts every later read.
    #[serde(default)]
    pub z_body: Vec<Poly>,
    /// `y_all[i] = tau_{g^i}(y_body)`, consumed by `packing_online`.
    #[serde(skip_serializing, skip_deserializing)]
    pub y_all: Vec<Vec<Poly>>,
    /// `y_all` in NTT form.
    #[serde(skip_serializing, skip_deserializing)]
    pub y_all_ntt: Vec<Vec<Poly>>,
    /// Conjugate rotations, full packing only.
    #[serde(skip_serializing, skip_deserializing)]
    pub y_bar_all: Vec<Vec<Poly>>,
    /// `y_bar_all` in NTT form.
    #[serde(skip_serializing, skip_deserializing)]
    pub y_bar_all_ntt: Vec<Vec<Poly>>,
    /// Whether these keys are for full packing.
    pub full_key: bool,
    /// gamma.
    pub num_to_pack: usize,
}

impl ClientPackingKeys {
    /// Derive packing keys from the secret key and the CRS `w_seed`.
    pub fn generate(
        sk: &RlweSecretKey,
        pack_params: &PackParams,
        w_seed: [u8; 32],
        sampler: &mut GaussianSampler,
    ) -> Self {
        let n = pack_params.ring_dim;
        let q = pack_params.q;
        let num_to_pack = pack_params.num_to_pack;
        let gadget_len = pack_params.gadget.len;
        let ctx = NttContext::with_moduli(n, &pack_params.moduli);

        let w_mask = generate_mask_from_seed(w_seed, n, q, &pack_params.moduli, gadget_len);

        let gen_pow = pack_params.gen_pows[1];
        let y_body = generate_ksk_body(sk, gen_pow, &pack_params.gadget, &w_mask, sampler, &ctx);

        let y_body_ntt: Vec<Poly> = y_body
            .iter()
            .map(|p| {
                let mut pn = p.clone();
                pn.to_ntt(&ctx);
                pn
            })
            .collect();

        let mut y_all = Vec::with_capacity(num_to_pack - 1);
        let mut y_all_ntt = Vec::with_capacity(num_to_pack - 1);
        for i in 0..(num_to_pack - 1) {
            let g_pow_i = pack_params.gen_pows[i];
            let table = pack_params.get_automorph_table(g_pow_i);

            let rotated_ntt: Vec<Poly> = y_body_ntt
                .iter()
                .map(|poly_ntt| apply_automorphism_ntt(poly_ntt, table))
                .collect();

            let rotated: Vec<Poly> = rotated_ntt
                .iter()
                .map(|p| {
                    let mut pc = p.clone();
                    pc.from_ntt(&ctx);
                    pc
                })
                .collect();

            y_all.push(rotated);
            y_all_ntt.push(rotated_ntt);
        }

        Self {
            y_body,
            z_body: vec![],
            y_all,
            y_all_ntt,
            y_bar_all: vec![],
            y_bar_all_ntt: vec![],
            full_key: false,
            num_to_pack,
        }
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
        let num_to_pack_half = n / 2;
        let two_n = 2 * n;
        let gadget_len = pack_params.gadget.len;
        let ctx = NttContext::with_moduli(n, &pack_params.moduli);

        let w_mask = generate_mask_from_seed(w_seed, n, q, &pack_params.moduli, gadget_len);
        let v_mask = generate_mask_from_seed(v_seed, n, q, &pack_params.moduli, gadget_len);

        let gen_pow = pack_params.gen_pows[1];
        let y_body = generate_ksk_body(sk, gen_pow, &pack_params.gadget, &w_mask, sampler, &ctx);
        let z_body = generate_ksk_body(sk, two_n - 1, &pack_params.gadget, &v_mask, sampler, &ctx);

        let y_body_ntt: Vec<Poly> = y_body
            .iter()
            .map(|p| {
                let mut pn = p.clone();
                pn.to_ntt(&ctx);
                pn
            })
            .collect();

        let mut y_all = Vec::with_capacity(num_to_pack_half - 1);
        let mut y_all_ntt = Vec::with_capacity(num_to_pack_half - 1);
        let mut y_bar_all = Vec::with_capacity(num_to_pack_half - 1);
        let mut y_bar_all_ntt = Vec::with_capacity(num_to_pack_half - 1);

        for i in 0..(num_to_pack_half - 1) {
            let g_pow_i = pack_params.gen_pows[i];
            let neg_g_pow_i = (two_n - g_pow_i) % two_n;

            let table_pos = pack_params.get_automorph_table(g_pow_i);
            let table_neg = pack_params.get_automorph_table(neg_g_pow_i);

            // Apply both automorphisms simultaneously in NTT domain
            let mut rotated_ntt = Vec::with_capacity(gadget_len);
            let mut rotated_bar_ntt = Vec::with_capacity(gadget_len);

            for poly_ntt in &y_body_ntt {
                let (pos, neg) = apply_automorphism_ntt_double(poly_ntt, table_pos, table_neg);
                rotated_ntt.push(pos);
                rotated_bar_ntt.push(neg);
            }

            // Convert to coefficient form
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

            y_all.push(rotated);
            y_all_ntt.push(rotated_ntt);
            y_bar_all.push(rotated_bar);
            y_bar_all_ntt.push(rotated_bar_ntt);
        }

        Self {
            y_body,
            z_body,
            y_all,
            y_all_ntt,
            y_bar_all,
            y_bar_all_ntt,
            full_key: true,
            num_to_pack: n,
        }
    }
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

/// y_body[k] = tau_g(s)*g^k - s*w_mask[k] + error.
fn generate_ksk_body(
    sk: &RlweSecretKey,
    gen_pow: usize,
    gadget: &GadgetVector,
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

    let mut result = Vec::with_capacity(gadget.len);

    for k in 0..gadget.len {
        let gadget_power = gadget.power(k);

        let tau_s_times_g = tau_s.scalar_mul(gadget_power);

        let s_times_mask = if k < w_mask.len() {
            let mut w_k_ntt = w_mask[k].clone();
            w_k_ntt.to_ntt(ctx);
            let mut prod = s_ntt.mul_ntt_domain(&w_k_ntt, ctx);
            prod.from_ntt(ctx);
            prod
        } else {
            Poly::zero_moduli(n, moduli)
        };
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
    let n = pack_params.ring_dim;
    let q = pack_params.q;
    let moduli = &pack_params.moduli;
    let num_to_pack = pack_params.num_to_pack;
    let gen_pows = &pack_params.gen_pows;

    let a_ct_ntt: Vec<Poly> = a_ct_tilde
        .iter()
        .map(|a| {
            let mut a_ntt = a.clone();
            a_ntt.to_ntt(ctx);
            a_ntt
        })
        .collect();

    // Each r_all[i] is independent - only the backward recursion below reads the
    // vector - so this O(gamma^2) region parallelizes, and it dominates server time.
    use crate::par_prelude::*;
    let r_all: Vec<Poly> = (0..num_to_pack)
        .into_par_iter()
        .map(|i| {
            let mut r_pow_i_ntt = Poly::zero_moduli(n, moduli);
            r_pow_i_ntt.to_ntt(ctx);

            for (j, a_j_ntt) in a_ct_ntt.iter().enumerate() {
                // g^{n-i}: the index walks the generator in the inverse direction.
                let exp_index = (n - i) % n;
                let index = (j * gen_pows[exp_index]) % (2 * n);

                // index >= n has wrapped past X^n = -1.
                let monomial_ntt = if index < n {
                    &pack_params.monomials_ntt[index % n]
                } else {
                    &pack_params.neg_monomials_ntt[index % n]
                };

                r_pow_i_ntt.mul_acc_ntt_domain(monomial_ntt, a_j_ntt, ctx);
            }

            let r_pow_i_scaled = r_pow_i_ntt.mul_ntt_domain(&pack_params.mod_inv_poly_ntt, ctx);

            let g_pow_i = gen_pows[i];
            let table = pack_params.get_automorph_table(g_pow_i);
            apply_automorphism_ntt(&r_pow_i_scaled, table)
        })
        .collect();
    let mut r_all = r_all;

    // The k-loop below already NTT-converts each gadget digit, so `bold_t_ntt` is
    // filled there rather than in a second pass.
    let mut bold_t: Vec<Vec<Poly>> = Vec::with_capacity(num_to_pack - 1);
    let mut bold_t_ntt_rev: Vec<Vec<Poly>> = Vec::with_capacity(num_to_pack - 1);

    for i in (0..(num_to_pack - 1)).rev() {
        // Gadget decomposition needs coefficient domain.
        let mut r_i_plus_1_coeff = r_all[i + 1].clone();
        r_i_plus_1_coeff.from_ntt(ctx);

        let gadget_inv = gadget_decompose(&r_i_plus_1_coeff, &pack_params.gadget);

        // Serial on purpose: at 3 gadget digits rayon measured break-even, and a fused
        // `mul_acc3_ntt_domain` measured flat against three simple loops.
        let mut gadget_inv_ntt: Vec<Poly> = Vec::with_capacity(gadget_inv.len());
        for (k, t_k) in gadget_inv.iter().enumerate() {
            let mut t_k_ntt = t_k.clone();
            t_k_ntt.to_ntt(ctx);
            if k < packing_key.w_all_ntt[i].len() {
                r_all[i].mul_acc_ntt_domain(&t_k_ntt, &packing_key.w_all_ntt[i][k], ctx);
            }
            gadget_inv_ntt.push(t_k_ntt);
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
        ring_dim: n,
        q,
    }
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

/// [`packing_online`] with the client rotations already in NTT form.
pub fn packing_online_ntt(
    precomp: &PrecompInsPIR,
    y_all_ntt: &[Vec<Poly>],
    b_poly: &Poly,
    ctx: &NttContext,
) -> RlweCiphertext {
    let n = precomp.ring_dim;
    let num_to_pack = precomp.num_to_pack;
    let moduli = precomp.a_hat.moduli();

    let mut sum_b_ntt = Poly::zero_moduli(n, moduli);
    sum_b_ntt.to_ntt(ctx);

    for i in 0..(num_to_pack - 1) {
        if i < y_all_ntt.len() && i < precomp.bold_t.len() {
            for (k, t_k) in precomp.bold_t[i].iter().enumerate() {
                if k < y_all_ntt[i].len() {
                    let mut t_ntt = t_k.clone();
                    t_ntt.to_ntt(ctx);

                    let term = y_all_ntt[i][k].mul_ntt_domain(&t_ntt, ctx);
                    sum_b_ntt = sum_b_ntt.add_ntt_domain(&term);
                }
            }
        }
    }

    sum_b_ntt.from_ntt(ctx);
    let final_b = b_poly + &sum_b_ntt;

    RlweCiphertext::from_parts(precomp.a_hat.clone(), final_b)
}

/// [`packing_online`] with both `bold_t_ntt` and `y_all_ntt` pre-converted, so the
/// loop is pure pointwise multiply-accumulate.
pub fn packing_online_fully_ntt(
    precomp: &PrecompInsPIR,
    y_all_ntt: &[Vec<Poly>],
    b_poly: &Poly,
    ctx: &NttContext,
) -> RlweCiphertext {
    let n = precomp.ring_dim;
    let num_to_pack = precomp.num_to_pack;
    let moduli = precomp.a_hat.moduli();

    let mut sum_b_ntt = Poly::zero_moduli(n, moduli);
    sum_b_ntt.to_ntt(ctx);

    for i in 0..(num_to_pack - 1) {
        if i < y_all_ntt.len() && i < precomp.bold_t_ntt.len() {
            for k in 0..precomp.bold_t_ntt[i].len() {
                if k < y_all_ntt[i].len() {
                    let term = y_all_ntt[i][k].mul_ntt_domain(&precomp.bold_t_ntt[i][k], ctx);
                    sum_b_ntt = sum_b_ntt.add_ntt_domain(&term);
                }
            }
        }
    }

    sum_b_ntt.from_ntt(ctx);
    let final_b = b_poly + &sum_b_ntt;

    RlweCiphertext::from_parts(precomp.a_hat.clone(), final_b)
}

impl ClientPackingKeys {
    /// Rebuild the `y_all*` rotations a wire round-trip dropped, once per session
    /// instead of once per respond call. No-op when both are already populated.
    pub fn ensure_server_derivatives(&mut self, pack_params: &PackParams, ctx: &NttContext) {
        if !self.y_all.is_empty() && !self.y_all_ntt.is_empty() {
            return;
        }
        if self.y_body.is_empty() {
            return;
        }
        let y_all_coeff = if self.y_all.is_empty() {
            generate_rotations(pack_params, &self.y_body)
        } else {
            self.y_all.clone()
        };
        let y_all_ntt: Vec<Vec<Poly>> = y_all_coeff
            .iter()
            .map(|inner| {
                inner
                    .iter()
                    .map(|poly| {
                        let mut pn = poly.clone();
                        pn.to_ntt(ctx);
                        pn
                    })
                    .collect()
            })
            .collect();
        if self.y_all.is_empty() {
            self.y_all = y_all_coeff;
        }
        if self.y_all_ntt.is_empty() {
            self.y_all_ntt = y_all_ntt;
        }
    }
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
            gadget_len: 3,
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
        let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, q);

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
