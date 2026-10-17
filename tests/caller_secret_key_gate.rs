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

/// Calls that encrypt a query under the session secret with the sampler they are handed.
const QUERY_ENCRYPTORS: &[&str] = &[
    "seeded_row(",
    "query_seeded_with(",
    "query_batch_with(",
    "RgswCiphertext::encrypt(",
    "seeded_query_with_gadget(",
];

/// Public functions in `src` that take a caller sampler, encrypt a query with it, and are
/// not gated. A seeded sampler replays the query error, and one query's error is enough to
/// tell the server which row it selects.
fn ungated_query_noise(src: &str) -> Vec<String> {
    let src = src.split("\n#[cfg(test)]").next().unwrap_or(src);
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(at) = src[from..].find("pub fn ") {
        let start = from + at;
        from = start + "pub fn ".len();
        let item = body_from(&src[start..]);
        let header = item.split('{').next().unwrap_or(item);
        if !header.contains("GaussianSampler") {
            continue;
        }
        if !QUERY_ENCRYPTORS.iter().any(|call| item.contains(call)) {
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

/// The modules that build client queries and that `pir` re-exports; `eval_poly` is private
/// to the crate and on no query path.
#[test]
fn no_default_public_query_takes_a_caller_sampler() {
    let offenders: Vec<String> = [
        ("session.rs", include_str!("../src/pir/session.rs")),
        ("query.rs", include_str!("../src/pir/query.rs")),
    ]
    .iter()
    .flat_map(|(path, text)| {
        ungated_query_noise(text)
            .into_iter()
            .map(move |name| format!("{path}: {name}"))
    })
    .collect();
    assert!(
        offenders.is_empty(),
        "public queries that draw their error from a caller sampler outside `{GATE}`:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn the_query_gate_sees_each_caller_sampler_entry_point() {
    let session = include_str!("../src/pir/session.rs");
    let query = include_str!("../src/pir/query.rs");
    for (src, header) in [
        (session, "    pub fn query(\n"),
        (session, "    pub fn query_seeded(\n"),
        (session, "    pub fn query_batch(\n"),
        (query, "pub fn query_seeded(\n"),
    ] {
        assert!(src.contains(header), "`{header}` moved; re-point this pin");
        let ungated = src.replace(&format!("{GATE}\n{header}"), header);
        assert_ne!(ungated, src, "`{header}` is not directly under the gate");
        assert!(
            !ungated_query_noise(&ungated).is_empty(),
            "dropping the gate from `{header}` went unseen"
        );
    }
}

#[test]
fn the_fresh_queries_draw_their_own_noise() {
    let session = include_str!("../src/pir/session.rs");
    for name in ["pub fn query_seeded_fresh(", "pub fn query_batch_fresh("] {
        let start = session.find(name).expect("fresh query present");
        let item = body_from(&session[start..]);
        let header = item.split('{').next().unwrap_or(item);
        assert!(!header.contains("GaussianSampler"), "{header}");
        assert!(item.contains("self.fresh_noise()"), "{item}");
    }
    assert!(session.contains("GaussianSampler::from_os_entropy(self.crs.params.sigma)"));
}

#[test]
fn the_special_modulus_client_takes_no_caller_secret_or_noise_by_default() {
    let client = include_str!("../src/special_modulus/client.rs");
    for header in [
        "    pub fn from_seed(",
        "    pub fn from_parts(",
        "    pub fn query_from_parts(",
        "    pub fn body_errors(",
    ] {
        let at = client
            .find(header)
            .unwrap_or_else(|| panic!("`{header}` moved; re-point this pin"));
        let line_start = client[..at].rfind('\n').map_or(0, |n| n + 1);
        assert!(
            gated(&client[..line_start]),
            "`{header}` is not under the gate"
        );
    }
    let item = |header: &str| {
        let start = client
            .find(header)
            .unwrap_or_else(|| panic!("`{header}` moved; re-point this pin"));
        body_from(&client[start..])
    };
    let new = item("    pub fn new(");
    for want in [
        "GaussianSampler::from_os_entropy(",
        "os_seeded_chacha(",
        "served_floor(params)?",
        "Self::drawn(params, noise, seeds)",
    ] {
        assert!(new.contains(want), "`{want}` missing from:\n{new}");
    }
    // The production secret and key error come only from the draw the seeded wire pins
    // cover; `new` composes nothing of its own.
    for deny in ["sample", "Self::build(", "key_error", "vec!["] {
        assert!(!new.contains(deny), "`{deny}` in:\n{new}");
    }
    assert!(item("    pub fn from_seed(").contains("Self::drawn(params, noise, seeds)"));
    let drawn = item("    fn drawn(");
    let body = drawn.split_once('{').map_or(drawn, |(_, b)| b);
    let lines: Vec<&str> = body
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(
        lines,
        [
            "let secret = noise.sample_vec(params.ring_dim);",
            "let key_error = noise.sample_vec(params.ring_dim);",
            "Self::build(params, &secret, &key_error, noise, seeds)",
            "}",
        ],
        "the shared draw changed; the seeded wire pins only cover it while it is this"
    );
}
