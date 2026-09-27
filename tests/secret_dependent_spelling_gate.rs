//! Spelling gate over the sites that touch a secret, NOT a timing proof. Two
//! halves, both able to fail.
//!
//! The pins hold the branch-free form of the sites that already have one, so a
//! revert to a secret-indexed load, a secret-conditioned jump or a primitive
//! comparison has to be deliberate. `from_signed` is pinned here on purpose:
//! the codegen gate next door states it cannot see a sign fold reverted to
//! `if val >= 0`, and this closes that half.
//!
//! The ledger holds the sites that are still secret-dependent, recorded so a
//! new one cannot appear unnoticed. A ledger entry fails in both directions -
//! when the recorded shape changes, and when the site is fixed - because a fix
//! must be proved byte-identical (tests/inverse_monomial_ct_rewrite_kat.rs)
//! and then moved into the pins above. Failing on a fix is the cost of having
//! the open sites enumerated anywhere at all.
//!
//! Renaming a gated item reddens rather than escapes: the extractor panics on
//! a header it cannot find. What does escape is the same defect spelled a way
//! the patterns do not name, and offending code moved out of a gated item into
//! a helper the gate has never heard of. That is the limit of the form.

#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "test-target harness; an abort here is the failure report"
)]

const GAUSSIAN_SRC: &str = include_str!("../src/math/gaussian.rs");
const MODULAR_SRC: &str = include_str!("../src/math/modular.rs");
const POLY_SRC: &str = include_str!("../src/math/poly.rs");
const ENCODE_DB_SRC: &str = include_str!("../src/pir/encode_db.rs");
const INSPIRING_SRC: &str = include_str!("../src/inspiring/inspiring2.rs");
const GALOIS_SRC: &str = include_str!("../src/rlwe/galois.rs");
const KS_SETUP_SRC: &str = include_str!("../src/ks/setup.rs");
const LWE_ENC_SRC: &str = include_str!("../src/lwe/enc.rs");
const QUERY_SRC: &str = include_str!("../src/pir/query.rs");
const SESSION_SRC: &str = include_str!("../src/pir/session.rs");
const PARAMS_SRC: &str = include_str!("../src/params.rs");
const RLWE_ENC_SRC: &str = include_str!("../src/rlwe/enc.rs");
const CRT_SRC: &str = include_str!("../src/math/crt.rs");
const MOD_SWITCH_SRC: &str = include_str!("../src/pir/mod_switch.rs");
const NTT_SRC: &str = include_str!("../src/math/ntt.rs");
const SOLINAS_SRC: &str = include_str!("../src/math/solinas_redc.rs");

/// Body of the item introduced by `header`, brace-matched so an item nested in
/// an `impl` block ends at its own closing brace and not the block's.
fn item_source(src: &'static str, header: &str) -> &'static str {
    let start = src
        .find(header)
        .unwrap_or_else(|| panic!("`{header}` must be present; the gate would inspect nothing"));
    let rest = &src[start..];
    let open = rest
        .find('{')
        .unwrap_or_else(|| panic!("`{header}` has no body"));
    let mut depth = 0usize;
    for (offset, ch) in rest.char_indices().skip(open) {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &rest[..=offset];
                }
            }
            _ => {}
        }
    }
    panic!("`{header}` body is unbalanced; the gate would inspect nothing")
}

/// Unit tests exercise the secret paths by construction, so they would trip
/// every pattern below.
fn without_test_module(src: &'static str) -> &'static str {
    src.split("\n#[cfg(test)]").next().unwrap_or(src)
}

fn deny(body: &str, site: &str, forbidden: &[&str], why: &str) {
    for pattern in forbidden {
        assert!(
            !body.contains(pattern),
            "{site} contains `{pattern}`; {why}:\n{body}"
        );
    }
}

fn require(body: &str, site: &str, needed: &[&str], why: &str) {
    for pattern in needed {
        assert!(
            body.contains(pattern),
            "{site} no longer contains `{pattern}`; {why}:\n{body}"
        );
    }
}

// ---------------------------------------------------------------- pins

#[test]
fn acceptance_table_is_read_by_a_full_scan_not_by_the_drawn_value() {
    let body = item_source(GAUSSIAN_SRC, "fn accept_threshold_bits");
    deny(
        body,
        "accept_threshold_bits",
        &["self.accept_bits[", "self.accept_bits.get("],
        "the drawn value is secret, so the table must be scanned and selected",
    );
    require(
        body,
        "accept_threshold_bits",
        &[
            "self.accept_bits.iter().enumerate()",
            "u64::conditional_select",
            ".ct_eq(&x)",
        ],
        "every entry must be visited and selected through subtle's barrier",
    );
}

#[test]
fn sampler_accepts_through_a_constant_time_comparison() {
    let body = item_source(GAUSSIAN_SRC, "fn sample_rejection");
    deny(
        body,
        "sample_rejection",
        &["threshold > u.to_bits()", "u.to_bits() <"],
        "both operands are secret, so the comparison must go through ConstantTimeGreater",
    );
    require(
        body,
        "sample_rejection",
        &[".ct_gt(&u.to_bits())"],
        "acceptance must be tested through subtle's greater-than barrier",
    );
}

#[test]
fn centering_folds_the_sample_sign_without_a_source_branch() {
    let body = item_source(GAUSSIAN_SRC, "pub fn sample_centered");
    deny(
        body,
        "sample_centered",
        &["if s < 0", "if s.is_negative()", "if s >= 0"],
        "the sample is secret; fold the sign with u64::conditional_select instead",
    );
    require(
        body,
        "sample_centered",
        &["u64::conditional_select", "ct_is_negative(s)"],
        "the sign must be folded through subtle's barrier",
    );
}

#[test]
fn from_signed_folds_the_sign_without_a_source_branch() {
    let body = item_source(MODULAR_SRC, "pub fn from_signed");
    deny(
        body,
        "from_signed",
        &["if val >= 0", "if val < 0", "if val.is_negative()"],
        "val is a secret noise coefficient; the codegen gate cannot see this revert",
    );
    require(
        body,
        "from_signed",
        &["u64::conditional_select", "ct_is_negative(val)"],
        "the sign must be folded through subtle's barrier",
    );
}

#[test]
fn conditional_subtraction_is_selected_not_branched() {
    let body = item_source(MODULAR_SRC, "fn ct_sub_if_ge");
    deny(
        body,
        "ct_sub_if_ge",
        &["if v >=", "if v <", "if ge"],
        "v is secret; the subtraction must be selected, not jumped over",
    );
    require(
        body,
        "ct_sub_if_ge",
        &["u64::conditional_select", ".ct_gt(&"],
        "the subtraction must be selected through subtle's barrier",
    );
}

#[test]
fn reduction_divides_only_by_the_public_modulus() {
    let body = item_source(MODULAR_SRC, "fn reduce_by_public_modulus");
    deny(
        body,
        "reduce_by_public_modulus",
        &["a % q", "a / q"],
        "a is secret and a hardware divide is variable-latency on x86-64",
    );
    require(
        body,
        "reduce_by_public_modulus",
        &["u64::MAX / q"],
        "Barrett's only divide takes a constant over the public modulus",
    );

    let helper = item_source(MODULAR_SRC, "fn reduce_by_public_reciprocal");
    deny(
        helper,
        "reduce_by_public_reciprocal",
        &[" / ", " % ", "if "],
        "a is secret; only the caller's public reciprocal may come from a divide",
    );
    require(
        helper,
        "reduce_by_public_reciprocal",
        &["u128::from(a) * u128::from(recip)", "ct_sub_if_ge("],
        "the estimate must be a multiply and the correction a selection",
    );
}

#[test]
fn sample_extraction_key_negation_is_selected_not_branched() {
    let body = item_source(LWE_ENC_SRC, "pub fn from_rlwe");
    deny(
        body,
        "LweSecretKey::from_rlwe",
        &["if s_i == 0", "if s_i != 0"],
        "the RLWE coefficient is secret; modular negation must be selected",
    );
    require(
        body,
        "LweSecretKey::from_rlwe",
        &["u64::conditional_select", "s_i.ct_eq(&0)"],
        "zero must be selected through subtle's barrier",
    );
}

/// The query index is the secret the scheme exists to hide. It may be handed
/// to a callee; it may not steer this path.
#[test]
fn the_query_path_neither_branches_on_nor_indexes_by_the_local_index() {
    let sites = [
        ("src/pir/query.rs", QUERY_SRC),
        ("src/pir/session.rs", SESSION_SRC),
    ];
    for (path, src) in sites {
        let body = without_test_module(src);
        assert!(
            body.contains("inverse_monomial(local_index as usize"),
            "{path} no longer reaches inverse_monomial with the local index; the gate \
             is watching a path that moved"
        );
        deny(
            body,
            path,
            &[
                "if local_index",
                "match local_index",
                "while local_index",
                "[local_index",
                "local_index ==",
                "local_index !=",
                "local_index <",
                "local_index >",
            ],
            "the local index is the query secret and must not steer control flow \
             or address memory here",
        );
    }
}

#[test]
fn inverse_monomial_scans_every_crt_residue() {
    let body = item_source(ENCODE_DB_SRC, "pub fn inverse_monomial");
    deny(
        body,
        "inverse_monomial",
        &["if k", "coeffs[pos]", "Poly::from_coeffs_moduli"],
        "the query index must not select control flow, a write address, or a secret reduction",
    );
    require(
        body,
        "inverse_monomial",
        &[
            "moduli.iter().enumerate()",
            "coeffs.iter_mut().enumerate()",
            "u64::conditional_select",
            ".ct_eq(&",
            "Poly::from_crt_coeffs_reduced",
        ],
        "every CRT residue must be selected through a full scan",
    );

    let reducer = item_source(ENCODE_DB_SRC, "fn reduce_exponent_by_public_ring");
    deny(
        reducer,
        "reduce_exponent_by_public_ring",
        &["exponent / two_d", "exponent % two_d", "if remainder"],
        "the secret exponent must not reach a divider",
    );
    require(
        reducer,
        "reduce_exponent_by_public_ring",
        &[
            "u64::MAX / two_d",
            "u128::from(exponent) * u128::from(reciprocal)",
            ".ct_gt(&two_d.wrapping_sub(1))",
            "u64::conditional_select",
        ],
        "only the public reciprocal calculation may divide",
    );
}

#[test]
fn shard_mapping_divides_only_public_values() {
    let body = item_source(PARAMS_SRC, "pub fn try_index_to_shard");
    deny(
        body,
        "try_index_to_shard",
        &[
            "global_idx / entries_per_shard",
            "global_idx % entries_per_shard",
        ],
        "the secret global index must not reach a divider",
    );
    require(
        body,
        "try_index_to_shard",
        &["div_rem_by_public_divisor(global_idx, entries_per_shard)"],
        "the quotient and secret residue must come from the reciprocal mapping",
    );

    let helper = item_source(PARAMS_SRC, "fn div_rem_by_public_divisor");
    deny(
        helper,
        "div_rem_by_public_divisor",
        &["dividend / divisor", "dividend % divisor", "if remainder"],
        "the secret dividend must reach only multiplication and constant-time selection",
    );
    require(
        helper,
        "div_rem_by_public_divisor",
        &[
            "u64::MAX / divisor",
            "u128::from(dividend) * u128::from(reciprocal)",
            ".ct_gt(&divisor.wrapping_sub(1))",
            "u64::conditional_select",
        ],
        "only the public reciprocal calculation may divide",
    );
}

#[test]
fn ksk_error_uses_the_shared_constant_time_sampler() {
    let body = item_source(INSPIRING_SRC, "fn generate_ksk_body");
    deny(
        body,
        "generate_ksk_body",
        &["if sample", "sample %", "error_coeffs"],
        "Gaussian signs and magnitudes must not steer control flow or division",
    );
    require(
        body,
        "generate_ksk_body",
        &["Poly::sample_gaussian_moduli(n, moduli, sampler)"],
        "the established CRT-aware Gaussian lift must own the conversion",
    );

    let helper = item_source(POLY_SRC, "pub fn sample_gaussian_moduli");
    deny(
        helper,
        "sample_gaussian_moduli",
        &["if samples", "samples[i] %", "if sample"],
        "Gaussian samples must not steer branching or division",
    );
    require(
        helper,
        "sample_gaussian_moduli",
        &["ModQ::from_signed(samples[i], modulus)"],
        "each CRT limb must use the hardened signed lift",
    );
}

#[test]
fn secret_key_automorphism_visits_every_crt_residue() {
    let body = item_source(GALOIS_SRC, "pub fn apply_automorphism");
    deny(
        body,
        "apply_automorphism",
        &[
            "if coeff == 0",
            "continue;",
            "poly.coeff(i)",
            "Poly::from_coeffs_moduli",
        ],
        "a raw secret key reaches this function",
    );
    require(
        body,
        "apply_automorphism",
        &[
            "poly.coeffs_modulus(limb)",
            "u64::conditional_select",
            "Poly::from_crt_coeffs_reduced",
        ],
        "every secret residue must take the same arithmetic path",
    );

    for header in ["fn ct_mod_add", "fn ct_mod_sub"] {
        let helper = item_source(GALOIS_SRC, header);
        deny(
            helper,
            header,
            &["if ", "% modulus"],
            "secret residues must not steer modular arithmetic",
        );
        require(
            helper,
            header,
            &["overflowing_", "u64::conditional_select"],
            "modular correction must use fixed-path selection",
        );
    }
}

#[test]
fn automorphism_key_setup_reuses_the_hardened_transform() {
    let body = item_source(KS_SETUP_SRC, "pub fn generate_automorphism_ks_matrix");
    deny(
        body,
        "generate_automorphism_ks_matrix",
        &["auto_s_coeffs", "if coeff == 0"],
        "the secret-key automorphism must not be hand-copied",
    );
    require(
        body,
        "generate_automorphism_ks_matrix",
        &["apply_automorphism(&sk.poly, automorphism)"],
        "the shared constant-time automorphism is the single implementation",
    );
}

/// The noisy message `a*s + b` is the plaintext row plus noise; it may reach a
/// multiply, never a divider, a branch or an index.
#[test]
fn the_extract_key_switch_composes_the_secret_without_a_residue_divide() {
    let body = item_source(MOD_SWITCH_SRC, "fn mod_switch_secret_key");
    deny(
        body,
        "mod_switch_secret_key",
        &[".coeff(", "from_coeffs(", " % ", "if c "],
        "the key is secret; `coeff` composes two CRT limbs with a branch and a u128 remainder",
    );
    require(
        body,
        "mod_switch_secret_key",
        &[
            ".coeffs_composed_ct()",
            "reduce_by_public_modulus(",
            "Poly::from_crt_coeffs_reduced(",
        ],
        "composition, reduction and output must take the selected paths",
    );
}

#[test]
fn decryption_divides_only_public_values() {
    let body = item_source(RLWE_ENC_SRC, "pub fn decrypt");
    deny(
        body,
        "RlweCiphertext::decrypt",
        &[
            "/ delta",
            "% p",
            "as u128 /",
            ".coeff(",
            "&a_s + &self.b",
            "from_coeffs(",
        ],
        "the noisy message is secret and a u128 divide is a software call on wasm32",
    );
    require(
        body,
        "RlweCiphertext::decrypt",
        &[
            ".add_ct(&self.b)",
            ".coeffs_composed_ct()",
            "rounding.round(",
            "Poly::from_crt_coeffs_reduced(coeffs, &[p])",
        ],
        "the sum, the CRT composition, the rounding and the output must take the selected paths",
    );

    let setup = item_source(RLWE_ENC_SRC, "fn new(delta: u64, p: u64)");
    require(
        setup,
        "PlaintextRounding::new",
        &["u64::MAX / delta", "u64::MAX / p"],
        "the only divides take a constant over the public scale and modulus",
    );
    let rounding = item_source(RLWE_ENC_SRC, "fn round(&self, noisy: u64)");
    deny(
        rounding,
        "PlaintextRounding::round",
        &[" / ", " % ", "if ", "match "],
        "the noisy coefficient must reach only multiplies and selection",
    );
    require(
        rounding,
        "PlaintextRounding::round",
        &[
            ".ct_gt(&(self.delta - 1))",
            "u128::conditional_select",
            "reduce_by_public_reciprocal(",
        ],
        "the quotient correction must go through subtle's barrier",
    );

    let add = item_source(POLY_SRC, "pub(crate) fn add_ct");
    deny(
        add,
        "Poly::add_ct",
        &["if sum", ">= modulus {"],
        "a secret sum must not steer a branch",
    );
    require(
        add,
        "Poly::add_ct",
        &["ct_sub_if_ge("],
        "the reduction must be selected",
    );

    let composed = item_source(POLY_SRC, "pub(crate) fn coeffs_composed_ct");
    deny(
        composed,
        "Poly::coeffs_composed_ct",
        &["crt_compose_2(", ".coeff(", " % "],
        "the branching, dividing composition must not return here",
    );
    require(
        composed,
        "Poly::coeffs_composed_ct",
        &["composer.compose("],
        "composition must go through the constant-time composer",
    );

    let compose = item_source(CRT_SRC, "pub(crate) fn compose");
    deny(
        compose,
        "CtCrtComposer::compose",
        &[" / ", " % ", "if "],
        "the residues are secret",
    );
    require(
        compose,
        "CtCrtComposer::compose",
        &[
            "reduce_by_public_reciprocal(a0, self.q1, self.q1_reciprocal)",
            "ct_sub_if_ge(",
        ],
        "reduction and correction must be Barrett and selection",
    );
}

/// The NTT multiplies the secret key on the client and runs every respond on
/// the server. Its corrections are a written-out `cmov` on x86-64 and a borrow
/// mask elsewhere, not subtle selects, so this pin is what stops a `>= q`
/// branch coming back.
#[test]
fn ntt_corrections_are_branch_free() {
    let butterflies = [
        "fn forward_inplace_at",
        "fn inverse_inplace_at",
        "fn forward_inplace_shoup_at",
        "fn inverse_inplace_shoup_at",
        "fn forward_inplace_solinas_at",
        "fn inverse_inplace_solinas_at",
    ];
    for header in butterflies {
        let body = item_source(NTT_SRC, header);
        deny(
            body,
            header,
            &[
                "if ",
                "match ",
                ">= q",
                "< q",
                "u >= v",
                "u < v",
                "conditional_select",
            ],
            "butterfly operands are secret on the client; correct without a branch",
        );
        require(
            body,
            header,
            &["csubq(u + v, q)", "sub_mod_branchless(u, v, q)"],
            "the sum and the difference must both take the branch-free correction",
        );
    }

    let tails = [
        (NTT_SRC, "fn montgomery_mul_at"),
        (NTT_SRC, "fn to_montgomery("),
        (NTT_SRC, "fn shoup_mul_at"),
        (SOLINAS_SRC, "pub fn solinas_mont_mul_default_q"),
    ];
    for (src, header) in tails {
        let body = item_source(src, header);
        deny(
            body,
            header,
            &["if ", "match ", ">= q", ">= Q", "conditional_select"],
            "the reduction tail sees a secret product",
        );
        require(
            body,
            header,
            &["csubq("],
            "the final subtraction must be branch-free",
        );
    }

    // A plain mask is not enough on x86-64: LLVM rebuilds the select and its
    // cmov conversion turns it into a jump in the Montgomery and Shoup loops,
    // as it did before this form landed.
    let helper = item_source(MODULAR_SRC, "fn sub_mod_branchless");
    deny(
        helper,
        "sub_mod_branchless",
        &[
            "if ",
            "match ",
            "conditional_select",
            ">= ",
            " < b",
            "a < b",
        ],
        "the correction must come from the borrow, not a source comparison",
    );
    require(
        helper,
        "sub_mod_branchless",
        &[
            "#[cfg(target_arch = \"x86_64\")]",
            "\"cmp {a}, {b}\"",
            "\"cmovb {out}, {fixed}\"",
            "options(pure, nomem, nostack)",
            "#[cfg(not(target_arch = \"x86_64\"))]",
            "sub_mod_mask(a, b, q)",
        ],
        "x86-64 must move conditionally in asm and the rest must mask the borrow",
    );
    let mask = item_source(MODULAR_SRC, "fn sub_mod_mask");
    deny(
        mask,
        "sub_mod_mask",
        &["if ", "match ", "conditional_select", ">= ", "a < b"],
        "the mask must come from the borrow, not a comparison",
    );
    require(
        mask,
        "sub_mod_mask",
        &[
            "a.overflowing_sub(b)",
            "0u64.wrapping_sub(u64::from(borrow))",
            "q & mask",
        ],
        "the borrow must widen to a mask over q",
    );
    require(
        item_source(MODULAR_SRC, "fn csubq"),
        "csubq",
        &["sub_mod_branchless(x, q, q)"],
        "the one-sided correction must share the branch-free path",
    );
}

/// The public transforms and products carry a lift or strip loop and the
/// pointwise loops of their own, so the pin above does not reach them. Their
/// only permitted branch is the public modulus dispatch.
#[test]
fn ntt_entry_points_correct_only_through_the_pinned_helpers() {
    let entries = [
        (
            NTT_SRC,
            "pub fn forward(",
            &["to_montgomery_at(", "forward_inplace_at("][..],
        ),
        (
            NTT_SRC,
            "pub fn forward_inplace(",
            &["forward_inplace_at("][..],
        ),
        (
            NTT_SRC,
            "pub fn inverse(",
            &["montgomery_mul_at(*c, 1, idx)"][..],
        ),
        (
            NTT_SRC,
            "pub fn inverse_inplace(",
            &["inverse_inplace_at("][..],
        ),
        (
            NTT_SRC,
            "pub fn forward_shoup(",
            &["forward_inplace_shoup_at("][..],
        ),
        (
            NTT_SRC,
            "pub fn inverse_shoup(",
            &["inverse_inplace_shoup_at("][..],
        ),
        (
            NTT_SRC,
            "pub fn forward_solinas(",
            &["Self::to_montgomery(*c"][..],
        ),
        (
            NTT_SRC,
            "pub fn inverse_solinas(",
            &["montgomery_mul_at(*c, 1, 0)"][..],
        ),
        (
            NTT_SRC,
            "pub fn pointwise_mul(",
            &["solinas_mont_mul_default_q(", "montgomery_mul_at("][..],
        ),
        (
            NTT_SRC,
            "pub fn pointwise_mul_single(",
            &["montgomery_mul_at("][..],
        ),
        (
            NTT_SRC,
            "pub fn pointwise_mul_single_at(",
            &["solinas_mont_mul_default_q(", "montgomery_mul_at("][..],
        ),
        (
            NTT_SRC,
            "pub fn pointwise_mul_shoup(",
            &["Self::shoup_mul_at("][..],
        ),
        (
            NTT_SRC,
            "pub fn pointwise_mul_solinas(",
            &["solinas_mont_mul_default_q("][..],
        ),
        (NTT_SRC, "pub fn to_mont(", &["Self::to_montgomery("][..]),
        (
            NTT_SRC,
            "pub fn from_mont(",
            &["montgomery_mul_at(a, 1, 0)"][..],
        ),
        (
            NTT_SRC,
            "fn to_montgomery_at(",
            &["Self::to_montgomery("][..],
        ),
        (
            SOLINAS_SRC,
            "pub fn pointwise_solinas_mont_mul(",
            &["solinas_mont_mul_default_q("][..],
        ),
    ];
    for (src, header, helpers) in entries {
        let body = item_source(src, header);
        deny(
            body,
            header,
            &[
                "=>",
                ">= ",
                "< q",
                "> q",
                "- q",
                "- Q",
                "DEFAULT_Q {",
                "- self.moduli",
                "wrapping_sub",
                "overflowing_sub",
                "select",
                ".min(",
                "then_some(",
                " % ",
            ],
            "the coefficients are secret; a correction belongs in a pinned helper",
        );
        let branches = body.matches("if ").count();
        let dispatches = body.matches("if self.is_solinas_default_q() {").count();
        assert_eq!(
            branches, dispatches,
            "{header} branches on something other than the public modulus dispatch:\n{body}"
        );
        require(
            body,
            header,
            helpers,
            "the product must reach its correction through a pinned helper",
        );
    }
}

// --------------------------------------------------------- extractor

/// A gate whose extractor returns nothing passes every `deny` above. Each test
/// carries a positive assertion for that reason; this checks the extractor
/// directly.
#[test]
fn the_extractor_returns_whole_item_bodies() {
    let cases = [
        (GAUSSIAN_SRC, "fn accept_threshold_bits"),
        (GAUSSIAN_SRC, "fn sample_rejection"),
        (GAUSSIAN_SRC, "pub fn sample_centered"),
        (MODULAR_SRC, "pub fn from_signed"),
        (MODULAR_SRC, "fn ct_sub_if_ge"),
        (MODULAR_SRC, "fn reduce_by_public_modulus"),
        (MODULAR_SRC, "fn reduce_by_public_reciprocal"),
        (POLY_SRC, "pub fn sample_gaussian_moduli"),
        (ENCODE_DB_SRC, "fn reduce_exponent_by_public_ring"),
        (ENCODE_DB_SRC, "pub fn inverse_monomial"),
        (INSPIRING_SRC, "fn generate_ksk_body"),
        (GALOIS_SRC, "pub fn apply_automorphism"),
        (KS_SETUP_SRC, "pub fn generate_automorphism_ks_matrix"),
        (GALOIS_SRC, "fn ct_mod_add"),
        (GALOIS_SRC, "fn ct_mod_sub"),
        (PARAMS_SRC, "fn div_rem_by_public_divisor"),
        (PARAMS_SRC, "pub fn try_index_to_shard"),
        (RLWE_ENC_SRC, "pub fn decrypt"),
        (RLWE_ENC_SRC, "fn new(delta: u64, p: u64)"),
        (RLWE_ENC_SRC, "fn round(&self, noisy: u64)"),
        (POLY_SRC, "pub(crate) fn add_ct"),
        (POLY_SRC, "pub(crate) fn coeffs_composed_ct"),
        (CRT_SRC, "pub(crate) fn compose"),
        (NTT_SRC, "fn forward_inplace_at"),
        (NTT_SRC, "fn inverse_inplace_at"),
        (NTT_SRC, "fn forward_inplace_shoup_at"),
        (NTT_SRC, "fn inverse_inplace_shoup_at"),
        (NTT_SRC, "fn forward_inplace_solinas_at"),
        (NTT_SRC, "fn inverse_inplace_solinas_at"),
        (NTT_SRC, "fn montgomery_mul_at"),
        (NTT_SRC, "fn to_montgomery("),
        (NTT_SRC, "fn shoup_mul_at"),
        (SOLINAS_SRC, "pub fn solinas_mont_mul_default_q"),
        (MODULAR_SRC, "fn sub_mod_branchless"),
        (MODULAR_SRC, "fn csubq"),
        (MODULAR_SRC, "fn sub_mod_mask"),
        (NTT_SRC, "pub fn forward("),
        (NTT_SRC, "pub fn forward_inplace("),
        (NTT_SRC, "pub fn inverse("),
        (NTT_SRC, "pub fn inverse_inplace("),
        (NTT_SRC, "pub fn forward_shoup("),
        (NTT_SRC, "pub fn inverse_shoup("),
        (NTT_SRC, "pub fn forward_solinas("),
        (NTT_SRC, "pub fn inverse_solinas("),
        (NTT_SRC, "pub fn pointwise_mul("),
        (NTT_SRC, "pub fn pointwise_mul_single("),
        (NTT_SRC, "pub fn pointwise_mul_single_at("),
        (NTT_SRC, "pub fn pointwise_mul_shoup("),
        (NTT_SRC, "pub fn pointwise_mul_solinas("),
        (NTT_SRC, "pub fn to_mont("),
        (NTT_SRC, "pub fn from_mont("),
        (NTT_SRC, "fn to_montgomery_at("),
        (SOLINAS_SRC, "pub fn pointwise_solinas_mont_mul("),
    ];
    for (src, header) in cases {
        let body = item_source(src, header);
        assert!(
            body.starts_with(header),
            "{header}: body does not start at the header"
        );
        assert!(
            body.ends_with('}'),
            "{header}: body does not end at a closing brace"
        );
        assert!(
            (40..4000).contains(&body.len()),
            "{header}: extracted {} bytes, which is not one item body",
            body.len()
        );
    }
}
