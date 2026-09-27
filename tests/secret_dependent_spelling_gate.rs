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
