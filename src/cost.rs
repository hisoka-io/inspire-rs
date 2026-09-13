//! Cost Attribution Estimator
//!
//! Analytically computes per-component resource costs for each PIR phase
//! across all InsPIRe variants, without requiring actual cryptographic execution.
//!
//! # Variant mapping
//!
//! - **NoPacking** (InsPIRe^0): Full query, one RLWE per column, no packing
//! - **OnePacking** (InsPIRe^1): Full query, tree packing (log(d) automorphisms)
//! - **TwoPacking** (InsPIRe^2): Seeded query, InspiRING 2-matrix packing
//!
//! # Example
//!
//! ```
//! use raven_inspire::cost::CostEstimator;
//! use raven_inspire::params::{InspireParams, InspireVariant};
//!
//! let params = InspireParams::secure_128_d2048();
//! let entry_size = 32;
//! let estimator = CostEstimator::new(&params, InspireVariant::NoPacking, entry_size);
//! let breakdown = estimator.estimate();
//! println!("{}", breakdown);
//! ```

use crate::params::{InspireParams, InspireVariant};
use std::fmt;

/// Counts of individual cryptographic operations within a phase.
#[derive(Debug, Clone, Default)]
pub struct OpCounts {
    /// Number of forward/inverse NTT transforms (each on a degree-d polynomial).
    pub ntt_transforms: u64,
    /// Number of coefficient-wise polynomial multiplications in NTT domain.
    pub poly_multiplications: u64,
    /// Number of gadget decompositions (each decomposes one polynomial into ℓ digits).
    pub gadget_decompositions: u64,
    /// Number of external products (RLWE x RGSW -> RLWE).
    pub external_products: u64,
    /// Number of key-switch operations.
    pub key_switches: u64,
    /// Number of automorphism applications (Galois permutations).
    pub automorphisms: u64,
    /// Number of polynomial additions.
    pub poly_additions: u64,
    /// Bytes of ciphertext data generated or consumed.
    pub bytes: u64,
}

impl OpCounts {
    /// Merge another OpCounts into this one (additive).
    pub fn merge(&mut self, other: &OpCounts) {
        self.ntt_transforms += other.ntt_transforms;
        self.poly_multiplications += other.poly_multiplications;
        self.gadget_decompositions += other.gadget_decompositions;
        self.external_products += other.external_products;
        self.key_switches += other.key_switches;
        self.automorphisms += other.automorphisms;
        self.poly_additions += other.poly_additions;
        self.bytes += other.bytes;
    }

    /// True when all counters are zero.
    pub fn is_zero(&self) -> bool {
        self.ntt_transforms == 0
            && self.poly_multiplications == 0
            && self.gadget_decompositions == 0
            && self.external_products == 0
            && self.key_switches == 0
            && self.automorphisms == 0
            && self.poly_additions == 0
            && self.bytes == 0
    }
}

/// Per-phase cost breakdown for a complete PIR interaction.
#[derive(Debug, Clone)]
pub struct CostBreakdown {
    /// The parameters used.
    pub params: InspireParams,
    /// The variant estimated.
    pub variant: InspireVariant,
    /// Entry size in bytes.
    pub entry_size: usize,
    /// Number of 16-bit columns per entry.
    pub num_columns: usize,
    /// Ring dimension d.
    pub ring_dim: usize,
    /// Gadget length l.
    pub gadget_len: usize,

    /// Setup phase costs (key generation, database encoding).
    pub setup: OpCounts,
    /// Query generation costs (client side).
    pub query: OpCounts,
    /// Server respond core costs (external products / accumulation), excluding packing.
    pub respond: OpCounts,
    /// Client extraction costs.
    pub extract: OpCounts,
    /// Packing costs (respond sub-phase, broken out from `respond` for attribution).
    pub packing: OpCounts,
    /// Communication costs (query upload + response download).
    pub communication: OpCounts,
}

impl CostBreakdown {
    /// Total operation counts across all phases.
    pub fn total(&self) -> OpCounts {
        let mut t = OpCounts::default();
        t.merge(&self.setup);
        t.merge(&self.query);
        t.merge(&self.respond);
        t.merge(&self.extract);
        t.merge(&self.packing);
        t
    }

    /// Total communication in bytes (query + response).
    pub fn total_communication_bytes(&self) -> u64 {
        self.communication.bytes
    }
}

impl fmt::Display for CostBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Cost Breakdown: {:?} (d={}, entry={}B, {} columns, l={})",
            self.variant, self.ring_dim, self.entry_size, self.num_columns, self.gadget_len
        )?;
        writeln!(f, "{:-<72}", "")?;

        fn fmt_phase(f: &mut fmt::Formatter<'_>, name: &str, ops: &OpCounts) -> fmt::Result {
            writeln!(f, "  {name}:")?;
            if ops.ntt_transforms > 0 {
                writeln!(f, "    NTT transforms:         {:>8}", ops.ntt_transforms)?;
            }
            if ops.poly_multiplications > 0 {
                writeln!(
                    f,
                    "    Poly multiplications:   {:>8}",
                    ops.poly_multiplications
                )?;
            }
            if ops.gadget_decompositions > 0 {
                writeln!(
                    f,
                    "    Gadget decompositions:  {:>8}",
                    ops.gadget_decompositions
                )?;
            }
            if ops.external_products > 0 {
                writeln!(
                    f,
                    "    External products:      {:>8}",
                    ops.external_products
                )?;
            }
            if ops.key_switches > 0 {
                writeln!(f, "    Key-switches:           {:>8}", ops.key_switches)?;
            }
            if ops.automorphisms > 0 {
                writeln!(f, "    Automorphisms:          {:>8}", ops.automorphisms)?;
            }
            if ops.poly_additions > 0 {
                writeln!(f, "    Poly additions:         {:>8}", ops.poly_additions)?;
            }
            if ops.bytes > 0 {
                writeln!(
                    f,
                    "    Bytes:                  {:>8} ({:.1} KB)",
                    ops.bytes,
                    ops.bytes as f64 / 1024.0
                )?;
            }
            Ok(())
        }

        fmt_phase(f, "Setup", &self.setup)?;
        fmt_phase(f, "Query", &self.query)?;
        fmt_phase(f, "Respond", &self.respond)?;
        if !self.packing.is_zero() {
            fmt_phase(f, "Packing (respond sub-phase)", &self.packing)?;
        }
        fmt_phase(f, "Extract", &self.extract)?;
        fmt_phase(f, "Communication", &self.communication)?;

        let total = self.total();
        fmt_phase(f, "TOTAL (computation)", &total)?;

        Ok(())
    }
}

/// Analytical cost estimator for InsPIRe PIR.
///
/// Computes operation counts per phase without running any cryptographic operations.
///
/// Models the canonical paper variants:
/// - NoPacking (^0): Full query, per-column response
/// - OnePacking (^1): Full query, tree-packed response (log(d) key-switches)
/// - TwoPacking (^2): Seeded query, InspiRING-packed response (2 KS matrices)
pub struct CostEstimator {
    params: InspireParams,
    variant: InspireVariant,
    entry_size: usize,
    num_columns: usize,
}

impl CostEstimator {
    /// Create a new estimator for the given parameters, variant, and entry size.
    pub fn new(params: &InspireParams, variant: InspireVariant, entry_size: usize) -> Self {
        let num_columns = crate::num_columns(entry_size);
        Self {
            params: params.clone(),
            variant,
            entry_size,
            num_columns,
        }
    }

    /// Compute the full cost breakdown.
    pub fn estimate(&self) -> CostBreakdown {
        CostBreakdown {
            params: self.params.clone(),
            variant: self.variant,
            entry_size: self.entry_size,
            num_columns: self.num_columns,
            ring_dim: self.params.ring_dim,
            gadget_len: self.params.gadget_len,
            setup: self.estimate_setup(),
            query: self.estimate_query(),
            respond: self.estimate_respond(),
            extract: self.estimate_extract(),
            packing: self.estimate_packing(),
            communication: self.estimate_communication(),
        }
    }

    /// Number of CRT limbs used to represent each polynomial coefficient.
    fn crt_moduli_count(&self) -> u64 {
        self.params.crt_moduli.len().max(1) as u64
    }

    /// Size of one RLWE ciphertext in bytes: 2 polynomials x d coefficients x CRT limbs x 8 bytes.
    fn rlwe_bytes(&self) -> u64 {
        2u64.saturating_mul(self.params.ring_dim as u64)
            .saturating_mul(self.crt_moduli_count())
            .saturating_mul(8)
    }

    /// Size of one one-sided RGSW ciphertext in bytes: l RLWE ciphertexts.
    fn rgsw_bytes(&self) -> u64 {
        self.params.gadget_len as u64 * self.rlwe_bytes()
    }

    // ---- Setup phase ----

    fn estimate_setup(&self) -> OpCounts {
        let d = self.params.ring_dim as u64;
        let ell = self.params.gadget_len as u64;
        let log_d = if d == 0 { 0 } else { d.ilog2() as u64 };

        // Base key-switching matrices: each is l RLWE encryptions
        let ks_matrix_rows = ell;
        let num_ks_matrices = 2u64; // the scheme's two base key-switching matrices

        // Galois keys: log_d KS matrices for tree packing automorphisms
        let galois_key_rows = log_d.saturating_mul(ell);

        // Packing key-switching matrices: 2 more for the packing variant
        let packing_ks_rows = 2u64.saturating_mul(ell);

        // Setup keys are variant-specific. NoPacking does not need packing keys.
        // TwoPacking adds the packing matrices to the base pair (4*ell rows total).
        let total_encryptions = match self.variant {
            InspireVariant::NoPacking => 0,
            InspireVariant::OnePacking => galois_key_rows,
            InspireVariant::TwoPacking => num_ks_matrices
                .saturating_mul(ks_matrix_rows)
                .saturating_add(packing_ks_rows),
        };

        // Each encryption: 1 poly_mul (a*s via NTT), forward + inverse NTT
        let ntt_transforms = total_encryptions.saturating_mul(2);
        let poly_multiplications = total_encryptions;

        // Storage: each KS row is an RLWE ciphertext
        let ks_bytes = total_encryptions.saturating_mul(self.rlwe_bytes());

        OpCounts {
            ntt_transforms,
            poly_multiplications,
            poly_additions: total_encryptions,
            bytes: ks_bytes,
            ..Default::default()
        }
    }

    // ---- Query phase ----

    fn estimate_query(&self) -> OpCounts {
        let ell = self.params.gadget_len as u64;

        // One-sided RGSW ciphertext of X^(-k): l RLWE encryptions.
        let num_encryptions = ell;
        let ntt_transforms = num_encryptions * 2;
        let poly_multiplications = num_encryptions;
        let poly_additions = num_encryptions;

        // For TwoPacking (InspiRING), client generates y_body packing keys
        let (extra_ntt, extra_mul, extra_add) = match self.variant {
            InspireVariant::NoPacking | InspireVariant::OnePacking => (0, 0, 0),
            InspireVariant::TwoPacking => {
                // `ClientPackingKeys::generate` builds y_body as gadget_len rows
                // of tau_g(s)*g^k - s*w_mask[k] + error, then rotates the set
                // into gamma-1 automorphism copies - so the work scales with
                // gamma*gadget_len, not with gamma alone, and the rotations are
                // automorphisms rather than poly multiplications. Left at 2*gamma
                // as the coarse stand-in it has always been: nothing observes op
                // counts, whereas the BYTES are now pinned to the wire by
                // tests/cost_model_matches_the_wire.rs. Reading "gamma rows" off
                // this comment is what produced the 85x packing-key over-count
                // below.
                let gamma = self.num_columns as u64;
                (gamma * 2, gamma * 2, gamma * 2)
            }
        };

        OpCounts {
            ntt_transforms: ntt_transforms + extra_ntt,
            poly_multiplications: poly_multiplications + extra_mul,
            poly_additions: poly_additions + extra_add,
            ..Default::default()
        }
    }

    // ---- Respond phase ----

    fn estimate_respond(&self) -> OpCounts {
        let ell = self.params.gadget_len as u64;
        let gamma = self.num_columns as u64;

        // Per external product (RLWE x RGSW -> RLWE):
        //   1 gadget decomposition (the trivial input's b component)
        //   2l poly_muls (l digits x 2 row components)
        //   2l poly_adds (accumulate results)
        //   ~3 NTTs per poly_mul (fwd x2, inv)
        let gadget_decomp_per_ext = 1u64;
        let poly_mul_per_ext = 2 * ell;
        let ntt_per_ext = poly_mul_per_ext * 3;
        let poly_add_per_ext = 2 * ell;

        let respond_external_products = gamma;
        let respond_gadget_decompositions = gamma * gadget_decomp_per_ext;
        let respond_poly_muls = gamma * poly_mul_per_ext;
        let respond_ntts = gamma * ntt_per_ext;
        let respond_poly_adds = gamma * poly_add_per_ext;

        // NoPacking: combine all columns into one ciphertext
        // Each RLWE addition touches both a and b components: 2 poly_adds per RLWE add
        let combine_adds = match self.variant {
            InspireVariant::NoPacking => 2 * gamma.saturating_sub(1),
            _ => 0,
        };

        OpCounts {
            ntt_transforms: respond_ntts,
            poly_multiplications: respond_poly_muls,
            gadget_decompositions: respond_gadget_decompositions,
            external_products: respond_external_products,
            poly_additions: respond_poly_adds + combine_adds,
            ..Default::default()
        }
    }

    // ---- Packing phase (within respond, broken out for attribution) ----

    fn estimate_packing(&self) -> OpCounts {
        let d = self.params.ring_dim as u64;
        let ell = self.params.gadget_len as u64;
        let gamma = self.num_columns as u64;

        match self.variant {
            InspireVariant::NoPacking => OpCounts::default(),

            InspireVariant::OnePacking => {
                // Tree packing: pad to d elements, binary tree with log_d levels.
                //
                // Total internal nodes = d - 1.
                // Per internal node (counting individual polynomial operations):
                //   ct_odd.poly_mul(y)     → 2 mul_ntt (a*y, b*y)
                //   ct_odd.poly_mul(neg_y) → 2 mul_ntt
                //   homomorphic_automorph:
                //     1 automorphism (coefficient permutation)
                //     1 gadget_decompose (a component)
                //     l key-switch iterations, each: 2 mul_ntt + 2 poly_adds
                //   ct_even.add(y_times_odd)     → 2 poly_adds (a+a, b+b)
                //   ct_even.add(neg_y_times_odd) → 2 poly_adds
                //   ct_sum_0.add(ct_sum_1_auto)  → 2 poly_adds
                let total_nodes = d.saturating_sub(1);

                let tree_poly_muls = total_nodes.saturating_mul(4 + 2 * ell);
                let tree_ntts = tree_poly_muls.saturating_mul(3);
                let tree_poly_adds = total_nodes.saturating_mul(6 + 2 * ell);
                let tree_gadget_decomps = total_nodes;
                let tree_automorphisms = total_nodes;
                let tree_key_switches = total_nodes;

                // b-value finalization: d scalar mul + add
                let finalize_adds = d;

                OpCounts {
                    ntt_transforms: tree_ntts,
                    poly_multiplications: tree_poly_muls,
                    gadget_decompositions: tree_gadget_decomps,
                    key_switches: tree_key_switches,
                    automorphisms: tree_automorphisms,
                    poly_additions: tree_poly_adds + finalize_adds,
                    ..Default::default()
                }
            }

            InspireVariant::TwoPacking => {
                // InspiRING 2-matrix packing (packing_offline + packing_online).
                //
                // Offline phase (packing_offline):
                //   Step 1 - For each i in 0..γ:
                //     Inner product: γ × mul_acc_ntt_domain → γ² poly_muls + γ² poly_adds
                //     Scale by 1/γ: γ × mul_ntt_domain → γ poly_muls
                //     Apply automorphism τ_{g^i}: γ automorphisms (NTT-domain permutation)
                //   Step 2 - Backward recursion (γ-1 iterations):
                //     Each: 1 gadget_decompose + l × mul_acc_ntt_domain
                //     → (γ-1) gadget_decomps, (γ-1)*l poly_muls, (γ-1)*l poly_adds
                //   NTTs: γ forward (a_ct), (γ-1) inverse (for decomp),
                //          (γ-1)*l forward (t_k), 1 inverse (a_hat)
                //
                // Online phase (packing_online):
                //   (γ-1)*l iterations: mul_ntt_domain + add_ntt_domain
                //   → (γ-1)*l poly_muls, (γ-1)*l poly_adds
                //   Plus 1 poly_add (final b_poly addition)
                //   NTTs: 2*(γ-1)*l forward (y_ntt, t_ntt) + 1 inverse (sum_b)

                let gm1 = gamma.saturating_sub(1); // γ - 1

                // Offline
                let gamma_sq = gamma.saturating_mul(gamma);
                let offline_poly_muls = gamma_sq
                    .saturating_add(gamma)
                    .saturating_add(gm1.saturating_mul(ell));
                let offline_poly_adds = gamma_sq.saturating_add(gm1.saturating_mul(ell));
                let offline_automorphisms = gamma;
                let offline_gadget_decomps = gm1;
                let offline_ntts = gamma
                    .saturating_add(gm1)
                    .saturating_add(gm1.saturating_mul(ell))
                    .saturating_add(1);

                // Online
                let online_poly_muls = gm1.saturating_mul(ell);
                let online_poly_adds = gm1.saturating_mul(ell).saturating_add(1);
                let online_ntts = 2u64
                    .saturating_mul(gm1)
                    .saturating_mul(ell)
                    .saturating_add(1);

                OpCounts {
                    ntt_transforms: offline_ntts.saturating_add(online_ntts),
                    poly_multiplications: offline_poly_muls.saturating_add(online_poly_muls),
                    gadget_decompositions: offline_gadget_decomps,
                    automorphisms: offline_automorphisms,
                    poly_additions: offline_poly_adds.saturating_add(online_poly_adds),
                    ..Default::default()
                }
            }
        }
    }

    // ---- Extract phase ----

    fn estimate_extract(&self) -> OpCounts {
        let gamma = self.num_columns as u64;

        match self.variant {
            InspireVariant::NoPacking => {
                // Decrypt gamma RLWE ciphertexts
                // Each: 1 poly_mul (a*s), ~3 NTTs, 1 poly_add (b - a*s)
                OpCounts {
                    ntt_transforms: gamma * 3,
                    poly_multiplications: gamma,
                    poly_additions: gamma,
                    ..Default::default()
                }
            }
            InspireVariant::OnePacking | InspireVariant::TwoPacking => {
                // Decrypt 1 packed RLWE, read gamma coefficients
                OpCounts {
                    ntt_transforms: 3,
                    poly_multiplications: 1,
                    poly_additions: 1,
                    ..Default::default()
                }
            }
        }
    }

    // ---- Communication costs ----

    fn estimate_communication(&self) -> OpCounts {
        let gamma = self.num_columns as u64;

        let query_bytes_full = self.rgsw_bytes();
        let query_bytes_seeded = query_bytes_full / 2;

        // InspiRING packing keys: only `y_body` crosses the wire, and it holds
        // one polynomial per gadget digit - `ClientPackingKeys::generate` builds
        // it from `pack_params.gadget`, and `register_client_packing_keys`
        // refuses a query whose `y_body.len() != pack_params.gadget.len`. The
        // `y_all` rotations are derived on both sides and never serialized, so
        // pricing this by gamma overstated it by gamma/ell (85x at a 512-byte
        // record); tests/cost_model_matches_the_wire.rs holds it to the wire.
        // Only sent for TwoPacking (InspiRING mode).
        let packing_key_bytes = match self.variant {
            InspireVariant::NoPacking | InspireVariant::OnePacking => 0,
            InspireVariant::TwoPacking => (self.params.gadget_len as u64)
                .saturating_mul(self.params.ring_dim as u64)
                .saturating_mul(self.crt_moduli_count())
                .saturating_mul(8),
        };

        let response_bytes = match self.variant {
            InspireVariant::NoPacking => (1 + gamma) * self.rlwe_bytes(),
            InspireVariant::OnePacking | InspireVariant::TwoPacking => {
                (self.params.ring_dim as u64 + gamma)
                    .saturating_mul(self.crt_moduli_count())
                    .saturating_mul(8)
            }
        };

        let query_bytes = match self.variant {
            InspireVariant::NoPacking | InspireVariant::OnePacking => query_bytes_full,
            InspireVariant::TwoPacking => query_bytes_seeded + packing_key_bytes,
        };

        OpCounts {
            bytes: query_bytes + response_bytes,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_estimator_no_packing() {
        let params = InspireParams::secure_128_d2048();
        let estimator = CostEstimator::new(&params, InspireVariant::NoPacking, 32);
        let breakdown = estimator.estimate();

        assert_eq!(breakdown.num_columns, 16);
        assert_eq!(breakdown.ring_dim, 2048);
        assert_eq!(breakdown.gadget_len, 3);

        // 16 external products (one per column)
        assert_eq!(breakdown.respond.external_products, 16);
        // No packing
        assert_eq!(breakdown.packing.key_switches, 0);

        // Communication: full RGSW query + (1 + 16) RLWE response
        let crt_limbs = params.crt_moduli.len().max(1) as u64;
        let rlwe_size = 2u64 * 2048 * crt_limbs * 8;
        let rgsw_size = 3 * rlwe_size;
        let response_size = 17 * rlwe_size;
        assert_eq!(breakdown.communication.bytes, rgsw_size + response_size);
    }

    #[test]
    fn test_cost_estimator_one_packing() {
        let params = InspireParams::secure_128_d2048();
        let estimator = CostEstimator::new(&params, InspireVariant::OnePacking, 32);
        let breakdown = estimator.estimate();

        // Tree packing: d-1 = 2047 key-switches and automorphisms
        assert_eq!(breakdown.packing.key_switches, 2047);
        assert_eq!(breakdown.packing.automorphisms, 2047);
        assert_eq!(breakdown.packing.gadget_decompositions, 2047);

        // Per node: 4 + 2*l = 10 poly_muls, so 2047 * 10 = 20470
        assert_eq!(breakdown.packing.poly_multiplications, 2047 * 10);
        // Per node: 6 + 2*l = 12 poly_adds, so 2047*12 + 2048 finalize = 26612
        assert_eq!(breakdown.packing.poly_additions, 2047 * 12 + 2048);

        // Communication: full RGSW + full a and the plaintext-bearing b prefix.
        let crt_limbs = params.crt_moduli.len().max(1) as u64;
        let rlwe_size = 2u64 * 2048 * crt_limbs * 8;
        let rgsw_size = 3 * rlwe_size;
        let packed_response = (2048u64 + 16) * crt_limbs * 8;
        assert_eq!(breakdown.communication.bytes, rgsw_size + packed_response);
    }

    #[test]
    fn test_cost_estimator_two_packing() {
        let params = InspireParams::secure_128_d2048();
        let estimator = CostEstimator::new(&params, InspireVariant::TwoPacking, 32);
        let breakdown = estimator.estimate();

        // InspiRING: no key-switches in packing
        assert_eq!(breakdown.packing.key_switches, 0);
        // γ automorphisms in offline phase (one per i in 0..γ, NOT γ²)
        assert_eq!(breakdown.packing.automorphisms, 16);
        // (γ-1) gadget decompositions in backward recursion
        assert_eq!(breakdown.packing.gadget_decompositions, 15);

        // Offline: γ²+γ+(γ-1)*l = 256+16+45 = 317 poly_muls
        // Online: (γ-1)*l = 45 poly_muls
        assert_eq!(breakdown.packing.poly_multiplications, 317 + 45);
        // Offline: γ²+(γ-1)*l = 256+45 = 301 poly_adds
        // Online: (γ-1)*l+1 = 46 poly_adds
        assert_eq!(breakdown.packing.poly_additions, 301 + 46);

        // Communication: seeded RGSW + y_body packing keys + packed response prefix.
        // y_body is gadget_len polynomials, NOT gamma - this line read `16u64`
        // (gamma at a 32-byte record) and so restated the estimator's own
        // over-count instead of checking it. The wire is the oracle now:
        // tests/cost_model_matches_the_wire.rs.
        let crt_limbs = params.crt_moduli.len().max(1) as u64;
        let rlwe_size = 2u64 * 2048 * crt_limbs * 8;
        let rgsw_seeded = 3 * rlwe_size / 2;
        let packing_keys = params.gadget_len as u64 * 2048 * crt_limbs * 8;
        let packed_response = (2048u64 + 16) * crt_limbs * 8;
        assert_eq!(
            breakdown.communication.bytes,
            rgsw_seeded + packing_keys + packed_response
        );
    }

    #[test]
    fn test_setup_is_variant_specific() {
        let params = InspireParams::secure_128_d2048();

        let no_pack = CostEstimator::new(&params, InspireVariant::NoPacking, 32).estimate();
        let one_pack = CostEstimator::new(&params, InspireVariant::OnePacking, 32).estimate();
        let two_pack = CostEstimator::new(&params, InspireVariant::TwoPacking, 32).estimate();

        // NoPacking has no packing-specific setup encryptions.
        assert_eq!(no_pack.setup.poly_multiplications, 0);
        assert_eq!(no_pack.setup.ntt_transforms, 0);
        assert_eq!(no_pack.setup.poly_additions, 0);

        // OnePacking needs log2(d)*ell = 11*3 = 33 setup encryptions.
        assert_eq!(one_pack.setup.poly_multiplications, 33);
        assert_eq!(one_pack.setup.ntt_transforms, 66);
        assert_eq!(one_pack.setup.poly_additions, 33);

        // TwoPacking needs 4*ell = 12 setup encryptions.
        assert_eq!(two_pack.setup.poly_multiplications, 12);
        assert_eq!(two_pack.setup.ntt_transforms, 24);
        assert_eq!(two_pack.setup.poly_additions, 12);
    }

    #[test]
    fn test_network_bytes_only_counted_in_communication() {
        let params = InspireParams::secure_128_d2048();

        for variant in [
            InspireVariant::NoPacking,
            InspireVariant::OnePacking,
            InspireVariant::TwoPacking,
        ] {
            let breakdown = CostEstimator::new(&params, variant, 32).estimate();
            assert_eq!(breakdown.query.bytes, 0);
            assert_eq!(breakdown.respond.bytes, 0);
            assert_eq!(breakdown.extract.bytes, 0);
            assert_eq!(breakdown.packing.bytes, 0);
            assert!(
                breakdown.communication.bytes > 0,
                "communication bytes should be populated for {variant:?}"
            );
        }
    }

    #[test]
    fn test_respond_and_packing_are_disjoint_for_packed_variants() {
        let params = InspireParams::secure_128_d2048();

        let one_pack = CostEstimator::new(&params, InspireVariant::OnePacking, 32).estimate();
        assert_eq!(one_pack.respond.key_switches, 0);
        assert_eq!(one_pack.respond.automorphisms, 0);
        assert!(one_pack.packing.key_switches > 0);
        assert!(one_pack.packing.automorphisms > 0);

        let two_pack = CostEstimator::new(&params, InspireVariant::TwoPacking, 32).estimate();
        assert_eq!(two_pack.respond.key_switches, 0);
        assert_eq!(two_pack.respond.automorphisms, 0);
        assert!(two_pack.packing.automorphisms > 0);
    }

    #[test]
    fn test_communication_ordering() {
        let params = InspireParams::secure_128_d2048();

        let no_pack = CostEstimator::new(&params, InspireVariant::NoPacking, 32)
            .estimate()
            .total_communication_bytes();
        let one_pack = CostEstimator::new(&params, InspireVariant::OnePacking, 32)
            .estimate()
            .total_communication_bytes();
        let two_pack = CostEstimator::new(&params, InspireVariant::TwoPacking, 32)
            .estimate()
            .total_communication_bytes();

        // NoPacking has the largest communication (large multi-column response)
        assert!(
            no_pack > one_pack,
            "NoPacking ({no_pack}) should exceed OnePacking ({one_pack})"
        );
        assert!(
            no_pack > two_pack,
            "NoPacking ({no_pack}) should exceed TwoPacking ({two_pack})"
        );

        // TwoPacking is the smallest: its seeded query halves the RGSW and the
        // packing keys it adds back are gadget_len polynomials, not gamma of
        // them. The reverse ordering this test asserted until 2026-09-07 was an
        // artifact of that over-count, and it held at every entry width for the
        // same reason - so the corrected ordering is checked at both ends of the
        // width range rather than at one point.
        assert_eq!(
            two_pack, one_pack,
            "after L2, TwoPacking's seeded rows plus y_body carry the same coefficient payload \
             as OnePacking's one-sided unseeded rows"
        );
        for entry_size in [2usize, 512] {
            let wide = CostEstimator::new(&params, InspireVariant::TwoPacking, entry_size)
                .estimate()
                .total_communication_bytes();
            let wide_one = CostEstimator::new(&params, InspireVariant::OnePacking, entry_size)
                .estimate()
                .total_communication_bytes();
            assert_eq!(
                wide, wide_one,
                "OnePacking and TwoPacking must price the same retained response prefix \
                 at {entry_size} B"
            );
        }
    }

    #[test]
    fn test_d4096_params() {
        let params = InspireParams::secure_128_d4096();
        let estimator = CostEstimator::new(&params, InspireVariant::NoPacking, 32);
        let breakdown = estimator.estimate();

        assert_eq!(breakdown.ring_dim, 4096);
        assert_eq!(breakdown.respond.external_products, 16);
    }

    #[test]
    fn test_display() {
        let params = InspireParams::secure_128_d2048();
        let estimator = CostEstimator::new(&params, InspireVariant::NoPacking, 32);
        let breakdown = estimator.estimate();
        let output = format!("{breakdown}");
        assert!(output.contains("NoPacking"));
        assert!(output.contains("d=2048"));
    }
}
