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
const ENCODE_DB_SRC: &str = include_str!("../src/pir/encode_db.rs");
const QUERY_SRC: &str = include_str!("../src/pir/query.rs");
const SESSION_SRC: &str = include_str!("../src/pir/session.rs");
const PARAMS_SRC: &str = include_str!("../src/params.rs");

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

// ------------------------------------------------------------- ledger
//
// Sites that are still secret-dependent. Each assertion pins the recorded
// shape; a fix reddens it and must be moved into the pins above.

#[test]
fn ledger_inverse_monomial_still_branches_on_the_query_index() {
    let body = item_source(ENCODE_DB_SRC, "pub fn inverse_monomial");
    let recorded = ["if k == 0", "let pos = d - k;", "coeffs[pos]"];
    for pattern in recorded {
        assert!(
            body.contains(pattern),
            "inverse_monomial no longer matches its recorded shape (`{pattern}` is gone). \
             If it was made branch-free: prove byte-identity with \
             tests/inverse_monomial_ct_rewrite_kat.rs, then move this site into the pins \
             above. If it changed some other way, the channel may have moved with it:\n{body}"
        );
    }
    assert!(
        !ENCODE_DB_SRC.contains("use subtle"),
        "src/pir/encode_db.rs now imports subtle; if inverse_monomial was made \
         constant-time, move it into the pins above and delete this ledger entry"
    );
}

/// Lower reach than the monomial: `shard_id` is published with the query, so
/// only the residue is secret. The divide still takes it in the dividend.
#[test]
fn ledger_shard_mapping_still_divides_by_a_runtime_divisor() {
    let body = item_source(PARAMS_SRC, "pub fn try_index_to_shard");
    let recorded = [
        "global_idx / entries_per_shard",
        "global_idx % entries_per_shard",
    ];
    for pattern in recorded {
        assert!(
            body.contains(pattern),
            "try_index_to_shard no longer matches its recorded shape (`{pattern}` is gone). \
             If the divide was removed, move this site into the pins above:\n{body}"
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
        (ENCODE_DB_SRC, "pub fn inverse_monomial"),
        (PARAMS_SRC, "pub fn try_index_to_shard"),
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
