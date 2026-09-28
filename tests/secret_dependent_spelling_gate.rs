//! Source-text gate over the sites that touch a secret, NOT a timing proof. It
//! reads the source as text and fails on the spellings it lists; it never
//! looks at codegen.
//!
//! Each pin extracts one item body by its header and denies or requires
//! literal substrings in it, so a revert to a secret-indexed load, a
//! secret-conditioned jump, a divide or a primitive comparison has to be
//! deliberate. `from_signed` is pinned on purpose: the codegen gate next door
//! cannot see a sign fold reverted to `if val >= 0`.
//!
//! Renaming a gated item reddens rather than escapes: the extractor panics on
//! a header it cannot find. Every pinned body is also scanned for the
//! correction spellings a per-site list tends to miss, and a body pinned but
//! left out of that scan fails the gate, keyed by source and header so the
//! twins that share a header are counted apart. What still escapes is a
//! defect spelled a way none of that names, and offending code moved out of a
//! gated item into a helper the gate has never heard of.

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
const RGSW_TYPES_SRC: &str = include_str!("../src/rgsw/types.rs");
const EXTRACT_SRC: &str = include_str!("../src/pir/extract.rs");

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

const COMPARISONS: [&str; 6] = [" < ", " > ", " <= ", " >= ", " == ", " != "];

fn has_comparison(text: &str) -> bool {
    COMPARISONS.iter().any(|op| text.contains(op))
}

/// The parenthesised group whose `(` sits at byte `open`.
fn group_after(body: &str, open: usize) -> &str {
    let mut depth = 0usize;
    for (offset, ch) in body[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return &body[open..=open + offset];
                }
            }
            _ => {}
        }
    }
    &body[open..]
}

/// The parenthesised group whose `)` sits at byte `close`.
fn group_before(body: &str, close: usize) -> &str {
    let mut depth = 0usize;
    for (offset, ch) in body[..=close].char_indices().rev() {
        match ch {
            ')' => depth += 1,
            '(' => {
                depth -= 1;
                if depth == 0 {
                    return &body[offset..=close];
                }
            }
            _ => {}
        }
    }
    &body[..=close]
}

/// Comparisons turned into integers, `u64::from(x >= q)` or `(x >= q) as u64`:
/// the multiply-by-bool correction, which LLVM rebuilds into a select and then,
/// in a hot loop, into a jump.
fn bool_to_integer_spellings(body: &str) -> Vec<String> {
    const INTEGERS: [&str; 12] = [
        "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize",
    ];
    let mut hits = Vec::new();
    for ty in INTEGERS {
        let from = format!("{ty}::from(");
        for (at, _) in body.match_indices(&from) {
            let group = group_after(body, at + from.len() - 1);
            if has_comparison(group) {
                hits.push(format!("{ty}::from{group}"));
            }
        }
        let cast = format!(") as {ty}");
        for (at, _) in body.match_indices(&cast) {
            let group = group_before(body, at);
            if has_comparison(group) {
                hits.push(format!("{group} as {ty}"));
            }
        }
    }
    hits
}

/// The correction spellings a per-site deny list tends to miss.
fn deny_hidden_corrections(body: &str, site: &str) {
    deny(
        body,
        site,
        &[
            "checked_sub(",
            "checked_add(",
            ".unwrap_or(",
            ".unwrap_or_else(",
            ".unwrap_or_default(",
            "then_some(",
            ".then(",
            ".min(",
            ".max(",
            "} else {",
            "else if ",
        ],
        "a correction on secret data must not hide behind a fallback, a bool or an else arm",
    );
    let casts = bool_to_integer_spellings(body);
    assert!(
        casts.is_empty(),
        "{site} turns a comparison into an integer ({casts:?}); a correction on secret data \
         must come from a borrow or a pinned helper:\n{body}"
    );
}

/// For the small arithmetic helpers: no comparison at all, so a correction
/// cannot come back as a boolean held in a `let`.
fn deny_comparisons(body: &str, site: &str) {
    deny(
        body,
        site,
        &COMPARISONS,
        "operands here are secret; compare only through a borrow or subtle",
    );
}

/// The query bodies hold the secret index and leave all arithmetic on it to
/// pinned callees, so the divide, widening, correction, loop, index and
/// comparison tokens below, none of which they need today, are denied outright;
/// an alias of the index escapes a per-name pattern but not these. The one
/// `match` is on the public session handle.
fn deny_query_body(body: &str, site: &str) {
    deny(
        body,
        site,
        &[
            "/",
            "%",
            "div",
            "rem",
            "euclid",
            "128",
            "into(",
            "from(",
            "wrapping_",
            "overflowing_",
            "checked_",
            "saturating_",
            ".coeff(",
            "set_coeff(",
            "coeffs_mut(",
            "if ",
            "while ",
            "for ",
            "loop",
            "[",
            ".get(",
            "nth(",
            "skip(",
            "split_at(",
            "clamp(",
            "matches!",
            "select",
            "then",
            "min(",
            "max(",
            "cmp(",
            ".lt(",
            ".le(",
            ".gt(",
            ".ge(",
            ".eq(",
            ".ne(",
        ],
        "the query body holds the secret index and must leave its arithmetic to \
         the pinned callees",
    );
    deny_comparisons(body, site);
    assert_eq!(
        body.matches("match ").count(),
        body.matches("match self.session_handle {").count(),
        "{site} matches on something other than the public session handle:\n{body}"
    );
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
        &["if s_i == 0", "if s_i != 0", ".coeff(", " % "],
        "the RLWE coefficient is secret; composition and negation must not branch or divide",
    );
    require(
        body,
        "LweSecretKey::from_rlwe",
        &[
            ".coeffs_composed_ct()",
            "u64::conditional_select",
            "s_i.ct_eq(&0)",
        ],
        "limbs must compose without a divide and zero must be selected through subtle",
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
        assert!(
            body.contains("inv_mono.scalar_mul(") && !body.contains("% modulus"),
            "{path} must scale the secret monomial through the pinned `Poly::scalar_mul`"
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

    // Item-level, so arithmetic on an alias of the index cannot hide in a body
    // the whole-file patterns above do not name. The bodies need none of these
    // tokens today, so each is denied wholesale rather than by spelling.
    let bodies = [
        (QUERY_SRC, "fn seeded_query_with_gadget"),
        (QUERY_SRC, "pub fn query("),
        (SESSION_SRC, "pub fn query("),
        (SESSION_SRC, "pub fn query_seeded("),
    ];
    for (src, header) in bodies {
        let body = item_source(src, header);
        require(
            body,
            header,
            &[
                "index_to_shard(global_index)",
                "inverse_monomial(local_index as usize",
                "inv_mono.scalar_mul(",
            ],
            "the index must reach the monomial only through the pinned callees",
        );
        deny_query_body(body, header);
    }
    let wrapper = item_source(QUERY_SRC, "pub fn query_seeded(");
    deny_query_body(wrapper, "query_seeded");
    require(
        wrapper,
        "query_seeded",
        &["seeded_query_with_gadget(crs, global_index, shard_config, rlwe_sk, sampler)"],
        "the seeded query must stay a forward to the pinned body",
    );
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
        &["/ delta", "% p", "as u128 /", ".coeff(", "from_coeffs("],
        "the noisy message is secret and a u128 divide is a software call on wasm32",
    );
    require(
        body,
        "RlweCiphertext::decrypt",
        &[
            "&a_s + &self.b",
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
        (NTT_SRC, "fn montgomery_mul_at", "csubq("),
        (NTT_SRC, "fn to_montgomery(", "csubq("),
        (
            NTT_SRC,
            "fn shoup_mul_at",
            "mul_mod_shoup(a, b, b_shoup, q)",
        ),
        (MODULAR_SRC, "fn mul_mod_shoup(", "csubq("),
        (SOLINAS_SRC, "pub fn solinas_mont_mul_default_q", "csubq("),
    ];
    for (src, header, correction) in tails {
        let body = item_source(src, header);
        deny(
            body,
            header,
            &[
                "if ",
                "match ",
                ">= q",
                ">= Q",
                "conditional_select",
                " % ",
                " / ",
            ],
            "the reduction tail sees a secret product",
        );
        require(
            body,
            header,
            &[correction],
            "the final subtraction must be branch-free",
        );
    }

    // A plain mask is not enough on x86-64: LLVM rebuilds the select and its
    // cmov conversion turns it into a jump in the Montgomery and Shoup loops.
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

/// The ring arithmetic every client operation on a secret goes through: the
/// scalar product that scales the query monomial and the key, the sum,
/// difference and negation that assemble a ciphertext, and the residue
/// reduction behind every constructor.
#[test]
fn client_ring_arithmetic_is_branch_and_divide_free() {
    let elementwise = [
        ("impl Add for &Poly", "csubq(*c + r, modulus)"),
        ("impl Sub for &Poly", "sub_mod_branchless(*c, b, modulus)"),
        ("impl Neg for &Poly", "sub_mod_branchless(0, *c, modulus)"),
        ("pub fn add_ntt_domain", "csubq(*c + o, modulus)"),
        (
            "pub fn add_assign_ntt_domain",
            "csubq(self.coeffs[i] + other.coeffs[i], modulus)",
        ),
    ];
    for (header, correction) in elementwise {
        let body = item_source(POLY_SRC, header);
        deny(
            body,
            header,
            &["if ", "match ", " % ", " / ", "conditional_select"],
            "coefficients are secret on the client; correct without a branch or divide",
        );
        deny_comparisons(body, header);
        require(
            body,
            header,
            &[correction],
            "the correction must be the pinned helper",
        );
    }

    let scale = item_source(POLY_SRC, "fn scale_residues");
    let dispatch = "if modulus <= SHOUP_NARROW_MAX_MODULUS {";
    assert_eq!(
        (
            scale.matches("if ").count(),
            scale.matches(dispatch).count()
        ),
        (1, 1),
        "scale_residues may branch only on the public modulus width:\n{scale}"
    );
    assert_eq!(
        scale.matches(" % ").count(),
        1,
        "scale_residues may divide only the public scalar:\n{scale}"
    );
    deny(
        scale,
        "scale_residues",
        &["*c %", "c as u128", "as u128 %", "modulus as u128"],
        "the residue is secret and a 128-bit remainder is a variable-time software call",
    );
    require(
        scale,
        "scale_residues",
        &[
            "let scalar = scalar % modulus;",
            "shoup_precompute(scalar, modulus)",
            "mul_mod_shoup(*c, scalar, scalar_shoup, modulus)",
            "mul_mod_shoup_wide(*c, scalar, scalar_shoup, modulus)",
        ],
        "every residue must take a Shoup product",
    );

    for (src, header) in [
        (POLY_SRC, "pub fn scalar_mul("),
        (POLY_SRC, "pub fn scalar_mul_assign"),
    ] {
        let body = item_source(src, header);
        deny(body, header, &[" % ", "as u128"], "the residues are secret");
    }
    require(
        item_source(POLY_SRC, "pub fn scalar_mul("),
        "Poly::scalar_mul",
        &["scalar_mul_assign(scalar)"],
        "the copying form must share the pinned loop",
    );
    require(
        item_source(POLY_SRC, "pub fn scalar_mul_assign"),
        "Poly::scalar_mul_assign",
        &["scale_residues("],
        "every limb must go through the Shoup product",
    );

    let reduce = item_source(POLY_SRC, "fn reduce_residues");
    deny(
        reduce,
        "reduce_residues",
        &["%=", "*c %", "if "],
        "the residues are secret",
    );
    deny_comparisons(reduce, "reduce_residues");
    require(
        reduce,
        "reduce_residues",
        &[
            "u64::MAX / modulus",
            "u128::from(*c) * u128::from(reciprocal)",
            "csubq(c.wrapping_sub(quotient.wrapping_mul(modulus)), modulus)",
        ],
        "only the public reciprocal may come from a divide",
    );
    for header in ["fn reduce(&mut self)", "pub fn from_coeffs_moduli"] {
        let body = item_source(POLY_SRC, header);
        deny(
            body,
            header,
            &[" % ", "%=", "crt_decompose_2(", "if "],
            "the coefficients can be secret; reduce through the pinned Barrett loop",
        );
        require(
            body,
            header,
            &["reduce_residues("],
            "reduction must be Barrett",
        );
    }

    let wide = item_source(MODULAR_SRC, "fn mul_mod_shoup_wide");
    deny(
        wide,
        "mul_mod_shoup_wide",
        &["if ", "match ", " % ", " / "],
        "the multiplicand is secret",
    );
    deny_comparisons(wide, "mul_mod_shoup_wide");
    require(
        wide,
        "mul_mod_shoup_wide",
        &["u128::conditional_select(", ".ct_gt(&(q - 1))"],
        "the 128-bit remainder must be corrected through subtle",
    );
    deny_comparisons(
        item_source(MODULAR_SRC, "fn mul_mod_shoup("),
        "mul_mod_shoup",
    );
}

/// Encryption, key-switching setup and packing-key generation handle the secret
/// key and the secret monomial; they may reach it only through the pinned ring
/// arithmetic above, never with arithmetic of their own.
#[test]
fn client_encryption_reaches_only_the_pinned_arithmetic() {
    let (plain_rgsw, seeded_rgsw) = rgsw_impls();
    let sites = [
        (
            RLWE_ENC_SRC,
            "pub fn encrypt(",
            "message_poly.scalar_mul(delta)",
        ),
        (
            RLWE_ENC_SRC,
            "pub fn encrypt_with_crs(",
            "message_poly.scalar_mul(delta)",
        ),
        (
            plain_rgsw,
            "pub fn encrypt_with_rng<",
            "message.scalar_mul(power)",
        ),
        (
            seeded_rgsw,
            "pub fn encrypt_with_rng<",
            "message.scalar_mul(power)",
        ),
        (
            KS_SETUP_SRC,
            "pub fn generate_ks_matrix",
            "from_key.poly.scalar_mul(power)",
        ),
        (
            INSPIRING_SRC,
            "fn generate_ksk_body",
            "tau_s.scalar_mul(gadget_power)",
        ),
    ];
    for (src, header, scaling) in sites {
        let body = item_source(src, header);
        deny(
            body,
            header,
            &[
                " % ",
                ".coeff(",
                "set_coeff(",
                "coeffs_mut(",
                "as u128",
                "wrapping_",
            ],
            "secret-key arithmetic must go through the pinned Poly operations",
        );
        require(
            body,
            header,
            &[scaling],
            "the scaling must be the pinned scalar product",
        );
    }

    let packed = item_source(EXTRACT_SRC, "fn extract_packed");
    deny(
        packed,
        "extract_packed",
        &["% p", "as u128"],
        "the decrypted value is the record; a 128-bit remainder divides it",
    );
    require(
        packed,
        "extract_packed",
        &["mul_mod_shoup_wide(scaled_value, d_inv, d_inv_shoup, p)"],
        "the unscaling must be a Shoup product by the public inverse",
    );
}

/// The plain and seeded RGSW impls share the header `encrypt_with_rng<`.
fn rgsw_impls() -> (&'static str, &'static str) {
    RGSW_TYPES_SRC
        .split_once("impl SeededRgswCiphertext {")
        .expect("RGSW types must keep the seeded impl")
}

/// Every pinned body without a public branch, scanned as written.
fn dispatch_free_bodies() -> Vec<(&'static str, &'static str)> {
    let (plain_rgsw, seeded_rgsw) = rgsw_impls();
    vec![
        (MODULAR_SRC, "pub fn from_signed"),
        (MODULAR_SRC, "fn ct_sub_if_ge"),
        (MODULAR_SRC, "fn reduce_by_public_modulus"),
        (MODULAR_SRC, "fn reduce_by_public_reciprocal"),
        (MODULAR_SRC, "fn sub_mod_branchless"),
        (MODULAR_SRC, "fn sub_mod_mask"),
        (MODULAR_SRC, "fn csubq"),
        (MODULAR_SRC, "fn mul_mod_shoup("),
        (MODULAR_SRC, "fn mul_mod_shoup_wide"),
        (GAUSSIAN_SRC, "fn accept_threshold_bits"),
        (GAUSSIAN_SRC, "fn sample_rejection"),
        (GAUSSIAN_SRC, "pub fn sample_centered"),
        (POLY_SRC, "pub fn sample_gaussian_moduli"),
        (LWE_ENC_SRC, "pub fn from_rlwe"),
        (ENCODE_DB_SRC, "pub fn inverse_monomial"),
        (ENCODE_DB_SRC, "fn reduce_exponent_by_public_ring"),
        (PARAMS_SRC, "pub fn try_index_to_shard"),
        (PARAMS_SRC, "fn div_rem_by_public_divisor"),
        (GALOIS_SRC, "fn ct_mod_add"),
        (GALOIS_SRC, "fn ct_mod_sub"),
        (KS_SETUP_SRC, "pub fn generate_automorphism_ks_matrix"),
        (MOD_SWITCH_SRC, "fn mod_switch_secret_key"),
        (RLWE_ENC_SRC, "pub fn decrypt"),
        (RLWE_ENC_SRC, "fn new(delta: u64, p: u64)"),
        (RLWE_ENC_SRC, "fn round(&self, noisy: u64)"),
        (CRT_SRC, "pub(crate) fn compose"),
        (POLY_SRC, "fn reduce_residues"),
        (POLY_SRC, "fn reduce(&mut self)"),
        (POLY_SRC, "pub fn from_coeffs_moduli"),
        (POLY_SRC, "pub fn scalar_mul("),
        (POLY_SRC, "pub fn scalar_mul_assign"),
        (POLY_SRC, "impl Add for &Poly"),
        (POLY_SRC, "impl Sub for &Poly"),
        (POLY_SRC, "impl Neg for &Poly"),
        (POLY_SRC, "pub fn add_ntt_domain"),
        (POLY_SRC, "pub fn add_assign_ntt_domain"),
        (NTT_SRC, "fn forward_inplace_at"),
        (NTT_SRC, "fn inverse_inplace_at"),
        (NTT_SRC, "fn forward_inplace_shoup_at"),
        (NTT_SRC, "fn inverse_inplace_shoup_at"),
        (NTT_SRC, "fn forward_inplace_solinas_at"),
        (NTT_SRC, "fn inverse_inplace_solinas_at"),
        (NTT_SRC, "fn montgomery_mul_at"),
        (NTT_SRC, "fn to_montgomery("),
        (NTT_SRC, "fn to_montgomery_at("),
        (NTT_SRC, "fn shoup_mul_at"),
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
        (SOLINAS_SRC, "pub fn solinas_mont_mul_default_q"),
        (SOLINAS_SRC, "pub fn pointwise_solinas_mont_mul("),
        (RLWE_ENC_SRC, "pub fn encrypt("),
        (RLWE_ENC_SRC, "pub fn encrypt_with_crs("),
        (plain_rgsw, "pub fn encrypt_with_rng<"),
        (seeded_rgsw, "pub fn encrypt_with_rng<"),
        (KS_SETUP_SRC, "pub fn generate_ks_matrix"),
        (EXTRACT_SRC, "fn extract_packed"),
        (QUERY_SRC, "fn seeded_query_with_gadget"),
        (QUERY_SRC, "pub fn query("),
        (QUERY_SRC, "pub fn query_seeded("),
        (SESSION_SRC, "pub fn query("),
        (SESSION_SRC, "pub fn query_seeded("),
    ]
}

/// Each pinned body with one public branch, lifted out before the scan: the
/// modulus width, the limb count, the automorphism's destination (a function of
/// `g` and the position) and the gadget row count.
const PUBLIC_BRANCHES: [(&str, &str, &str); 4] = [
    (
        POLY_SRC,
        "fn scale_residues",
        "if modulus <= SHOUP_NARROW_MAX_MODULUS {",
    ),
    (
        POLY_SRC,
        "pub(crate) fn coeffs_composed_ct",
        "if let [q0, q1] = self.moduli[..] {",
    ),
    (GALOIS_SRC, "pub fn apply_automorphism", "if new_idx < d {"),
    (
        INSPIRING_SRC,
        "fn generate_ksk_body",
        "if k < w_mask.len() {",
    ),
];

/// The per-site lists above name the spellings each site once had. This runs
/// the spellings they missed over every pinned body.
#[test]
fn pinned_bodies_hide_no_correction_spellings() {
    for (src, header) in dispatch_free_bodies() {
        deny_hidden_corrections(item_source(src, header), header);
    }
    for (src, header, branch) in PUBLIC_BRANCHES {
        let body = item_source(src, header);
        assert_eq!(
            body.matches(branch).count(),
            1,
            "{header} lost its public branch"
        );
        let lifted = body.replacen(branch, "", 1).replacen("} else {", "", 1);
        deny_hidden_corrections(&lifted, header);
    }
}

/// The gated sources by the names this file uses for them. The RGSW file is
/// listed as its two impls, which is how every pin reads it.
fn named_sources() -> Vec<(&'static str, &'static str)> {
    let (plain_rgsw, seeded_rgsw) = rgsw_impls();
    vec![
        ("GAUSSIAN_SRC", GAUSSIAN_SRC),
        ("MODULAR_SRC", MODULAR_SRC),
        ("POLY_SRC", POLY_SRC),
        ("ENCODE_DB_SRC", ENCODE_DB_SRC),
        ("INSPIRING_SRC", INSPIRING_SRC),
        ("GALOIS_SRC", GALOIS_SRC),
        ("KS_SETUP_SRC", KS_SETUP_SRC),
        ("LWE_ENC_SRC", LWE_ENC_SRC),
        ("QUERY_SRC", QUERY_SRC),
        ("SESSION_SRC", SESSION_SRC),
        ("PARAMS_SRC", PARAMS_SRC),
        ("RLWE_ENC_SRC", RLWE_ENC_SRC),
        ("CRT_SRC", CRT_SRC),
        ("MOD_SWITCH_SRC", MOD_SWITCH_SRC),
        ("NTT_SRC", NTT_SRC),
        ("SOLINAS_SRC", SOLINAS_SRC),
        ("EXTRACT_SRC", EXTRACT_SRC),
        ("plain_rgsw", plain_rgsw),
        ("seeded_rgsw", seeded_rgsw),
        ("rgsw_impls().0", plain_rgsw),
        ("rgsw_impls().1", seeded_rgsw),
    ]
}

/// By content: a const `&str` has no guaranteed single address.
fn same_source(a: &str, b: &str) -> bool {
    a == b
}

/// The source a header literal at byte `at` is pinned against: the name
/// written just before it (`(NAME, "header"` or `item_source(NAME, "header"`),
/// else the one gated source that contains the header. A header two sources
/// share must name its source, so a twin is never counted as the other.
fn source_of_literal(gate: &str, at: usize, header: &str) -> &'static str {
    let before = gate[..at].trim_end();
    if let Some(before) = before.strip_suffix(',') {
        let before = before.trim_end();
        for (name, src) in named_sources() {
            let bounded = before.strip_suffix(name).is_some_and(|rest| {
                !rest
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
            });
            if bounded {
                return src;
            }
        }
    }
    let holders: Vec<&'static str> = named_sources()
        .into_iter()
        .filter(|(name, _)| !name.starts_with("rgsw_impls"))
        .map(|(_, src)| src)
        .filter(|src| src.contains(header))
        .collect();
    assert_eq!(
        holders.len(),
        1,
        "`{header}` is in {} gated sources; the pin must name its source",
        holders.len()
    );
    holders[0]
}

/// A body pinned above but left out of the scan would make the scan's claim
/// false with every test green, so this reads the headers this file names and
/// the source each is pinned against.
#[test]
fn every_pinned_body_is_scanned() {
    const GATE_SRC: &str = include_str!("secret_dependent_spelling_gate.rs");
    let scanned: Vec<(&str, &str)> = dispatch_free_bodies()
        .into_iter()
        .chain(
            PUBLIC_BRANCHES
                .iter()
                .map(|&(src, header, _)| (src, header)),
        )
        .collect();
    let is_scanned = |src: &str, header: &str| {
        scanned
            .iter()
            .any(|&(s, h)| same_source(s, src) && h == header)
    };
    let mut named = Vec::new();
    for (at, _) in GATE_SRC.match_indices('\u{22}') {
        let rest = &GATE_SRC[at + 1..];
        let Some(end) = rest.find('\u{22}') else {
            continue;
        };
        let literal = &rest[..end];
        let is_header = ["fn ", "pub fn ", "pub(crate) fn ", "impl "]
            .iter()
            .any(|prefix| literal.len() > prefix.len() && literal.starts_with(prefix));
        if is_header && !literal.contains('{') {
            named.push((source_of_literal(GATE_SRC, at, literal), literal));
        }
    }
    let (_, seeded_rgsw) = rgsw_impls();
    let found = |src: &str, header: &str| {
        named
            .iter()
            .any(|&(s, h)| same_source(s, src) && h == header)
    };
    assert!(
        found(POLY_SRC, "pub fn scalar_mul(")
            && found(SESSION_SRC, "pub fn query(")
            && found(seeded_rgsw, "pub fn encrypt_with_rng<"),
        "the header reader missed a pin; it would pass every omission"
    );
    for (src, header) in named {
        assert!(
            is_scanned(src, header),
            "`{header}` is pinned but not scanned for hidden corrections in {}",
            named_sources()
                .iter()
                .find(|&&(_, s)| same_source(s, src))
                .map_or("an unnamed source", |&(name, _)| name)
        );
    }
}

/// The scan must see each spelling it names, or it passes everything.
#[test]
fn the_hidden_correction_scan_sees_each_spelling() {
    let plants = [
        "let s = u + v; s.checked_sub(q).unwrap_or(s)",
        "a.wrapping_sub(b).wrapping_add(q * u64::from(a < b))",
        "x - q * ((x >= q) as u64)",
        "if x >= q { x - q } else { x }",
        "(x >= q).then_some(q).unwrap_or(0)",
        "x.min(x.wrapping_sub(q))",
    ];
    for plant in plants {
        let body = format!("fn planted(x: u64) -> u64 {{ {plant} }}");
        let caught = std::panic::catch_unwind(|| deny_hidden_corrections(&body, "planted"));
        assert!(caught.is_err(), "the scan missed `{plant}`");
    }
    let clean = "fn clean(a: u64, b: u64) -> u64 { let (d, borrow) = a.overflowing_sub(b); \
                 d.wrapping_add(q & 0u64.wrapping_sub(u64::from(borrow))) }";
    deny_hidden_corrections(clean, "clean");
    assert!(bool_to_integer_spellings("(u128::from(a) * u128::from(r)) >> 64) as u64").is_empty());
}

/// Arithmetic on an alias of the index, which the per-name patterns miss.
#[test]
fn the_query_body_scan_sees_each_spelling() {
    let plants = [
        "let i = local_index.rem_euclid(bound);",
        "let w: u128 = local_index.into();",
        "let w = widen(local_index.into());",
        "let mut i = local_index; if bound <= i { i -= bound; }",
        "let ge = local_index >= bound; let i = local_index - bound * u64::from(ge);",
        "let ge = local_index.ge(&bound); let i = local_index - bound * (ge as u64);",
        "let i = local_index; match i { 0 => a, _ => b }",
    ];
    for plant in plants {
        let body =
            format!("pub fn query(&self) {{ {plant} match self.session_handle {{ _ => () }} }}");
        let caught = std::panic::catch_unwind(|| deny_query_body(&body, "planted"));
        assert!(caught.is_err(), "the query body scan missed `{plant}`");
    }
    deny_query_body(
        "pub fn query(&self) { let x = f(local_index as usize); \
         match self.session_handle { Some(h) => (None, Some(h)), None => (x, None) } }",
        "clean",
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
        (MODULAR_SRC, "fn mul_mod_shoup("),
        (MODULAR_SRC, "fn mul_mod_shoup_wide"),
        (POLY_SRC, "fn scale_residues"),
        (POLY_SRC, "fn reduce_residues"),
        (POLY_SRC, "fn reduce(&mut self)"),
        (POLY_SRC, "pub fn from_coeffs_moduli"),
        (POLY_SRC, "pub fn scalar_mul("),
        (POLY_SRC, "pub fn scalar_mul_assign"),
        (POLY_SRC, "impl Add for &Poly"),
        (POLY_SRC, "impl Sub for &Poly"),
        (POLY_SRC, "impl Neg for &Poly"),
        (POLY_SRC, "pub fn add_ntt_domain"),
        (POLY_SRC, "pub fn add_assign_ntt_domain"),
        (RLWE_ENC_SRC, "pub fn encrypt("),
        (RLWE_ENC_SRC, "pub fn encrypt_with_crs("),
        (KS_SETUP_SRC, "pub fn generate_ks_matrix"),
        (INSPIRING_SRC, "fn generate_ksk_body"),
        (EXTRACT_SRC, "fn extract_packed"),
        (QUERY_SRC, "fn seeded_query_with_gadget"),
        (QUERY_SRC, "pub fn query("),
        (QUERY_SRC, "pub fn query_seeded("),
        (SESSION_SRC, "pub fn query("),
        (SESSION_SRC, "pub fn query_seeded("),
        (rgsw_impls().0, "pub fn encrypt_with_rng<"),
        (rgsw_impls().1, "pub fn encrypt_with_rng<"),
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
