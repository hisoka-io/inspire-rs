//! Source-text gate: no default-build public function builds a packing key under a
//! secret the caller hands in.
//!
//! Two packing keys under one secret share the CRS mask and differ only in fresh
//! error, so publishing both leaks the secret. The default API builds keys only in
//! `ClientSession::with_fresh_secret`, which draws its own secret; every public
//! function that takes an `RlweSecretKey` and reaches a key builder must sit behind the
//! `caller-secret-keys` feature, which only test builds enable.
//!
//! `with_fresh_secret` takes no sampler: a caller's seeded sampler would replay the
//! secret, so a second call on the same seed would give that secret a second key.
//!
//! What escapes: a key builder reached through a helper not named in
//! [`KEY_BUILDERS`], and a secret handed in as anything but an `RlweSecretKey` or a
//! sampler the function draws it from.

#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "test-target harness; an abort here is the failure report"
)]

use std::path::Path;

const GATE: &str = "#[cfg(feature = \"caller-secret-keys\")]";

/// Calls that build a packing key, directly or through a crate-private helper.
const KEY_BUILDERS: &[&str] = &[
    "ClientPackingKeys::generate(",
    "generate_once(",
    "generate_ksk_body(",
    "generate_packing_keys(",
    "seeded_query_with_gadget(",
    "Self::build(",
];

fn body_from(rest: &str) -> &str {
    let Some(open) = rest.find('{') else {
        return rest;
    };
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
    rest
}

fn gated(before: &str) -> bool {
    before
        .lines()
        .rev()
        .map(str::trim)
        .take_while(|line| line.starts_with("#[") || line.starts_with("///"))
        .any(|line| line == GATE)
}

/// Public functions in `src` that take a caller secret, or a caller sampler they draw the
/// secret from, build a key, and are not gated.
fn ungated_key_builders(src: &str) -> Vec<String> {
    let src = src.split("\n#[cfg(test)]").next().unwrap_or(src);
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(at) = src[from..].find("pub fn ") {
        let start = from + at;
        from = start + "pub fn ".len();
        let item = body_from(&src[start..]);
        let header = item.split('{').next().unwrap_or(item);
        let secret_from_caller = header.contains("RlweSecretKey")
            || (header.contains("GaussianSampler") && item.contains("RlweSecretKey::generate("));
        if !secret_from_caller {
            continue;
        }
        if !KEY_BUILDERS.iter().any(|builder| item.contains(builder)) {
            continue;
        }
        let line_start = src[..start].rfind('\n').map_or(0, |n| n + 1);
        if !gated(&src[..line_start]) {
            let name = header["pub fn ".len()..]
                .split(['(', '<'])
                .next()
                .unwrap_or(header);
            found.push(name.to_owned());
        }
    }
    found
}

fn rust_sources(dir: &Path, out: &mut Vec<(String, String)>) {
    for entry in std::fs::read_dir(dir).expect("read src") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let text = std::fs::read_to_string(&path).expect("read source");
            out.push((path.display().to_string(), text));
        }
    }
}

#[test]
fn no_default_public_function_builds_a_key_under_a_caller_secret() {
    let mut sources = Vec::new();
    rust_sources(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut sources,
    );
    assert!(sources.len() > 20, "the walk found {} files", sources.len());
    let offenders: Vec<String> = sources
        .iter()
        .flat_map(|(path, text)| {
            ungated_key_builders(text)
                .into_iter()
                .map(move |name| format!("{path}: {name}"))
        })
        .collect();
    assert!(
        offenders.is_empty(),
        "public key builders that take a caller secret outside `{GATE}`:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn the_gate_sees_each_caller_secret_entry_point() {
    let session = include_str!("../src/pir/session.rs");
    let query = include_str!("../src/pir/query.rs");
    let inspiring = include_str!("../src/inspiring/inspiring2.rs");
    for (src, header) in [
        (session, "    pub fn new("),
        (session, "    pub fn with_seeded_secret("),
        (query, "pub fn query("),
        (query, "pub fn query_seeded("),
        (
            inspiring,
            "    pub fn generate(\n        sk: &RlweSecretKey",
        ),
    ] {
        assert!(src.contains(header), "`{header}` moved; re-point this pin");
        let ungated = src.replace(&format!("{GATE}\n{header}"), header);
        assert_ne!(ungated, src, "`{header}` is not directly under the gate");
        assert!(
            !ungated_key_builders(&ungated).is_empty(),
            "dropping the gate from `{header}` went unseen"
        );
    }
}

#[test]
fn the_fresh_secret_constructor_draws_its_own_secret() {
    let session = include_str!("../src/pir/session.rs");
    let start = session
        .find("pub fn with_fresh_secret(")
        .expect("with_fresh_secret present");
    let item = body_from(&session[start..]);
    let header = item.split('{').next().unwrap_or(item);
    assert!(!header.contains("RlweSecretKey"), "{header}");
    assert!(!header.contains("GaussianSampler"), "{header}");
    assert!(item.contains("GaussianSampler::from_os_entropy("), "{item}");
    assert!(item.contains("RlweSecretKey::generate("), "{item}");
}
