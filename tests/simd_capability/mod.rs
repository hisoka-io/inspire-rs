//! Capability gates for the SIMD differential tests, shared by every test
//! binary that has one.
//!
//! A differential test that `return`s early on a host without AVX-512-IFMA52
//! reports GREEN while comparing nothing, and the same is true of the packing
//! dispatch differential when `simd-packing-offline` is off: the dispatch then
//! IS the scalar loop, so both sides of the comparison are the same code.
//! Both skips are therefore named on stderr, and `RAVEN_REQUIRE_AVX512=1`
//! promotes either one to a hard failure — which is what a CI lane that claims
//! to exercise these kernels must set, so a runner that cannot run them says so
//! instead of passing.

#![allow(
    dead_code,
    reason = "each test binary includes the whole module and uses the gate it needs"
)]

/// Setting this to `1` turns a capability skip into a panic.
pub const REQUIRE_ENV: &str = "RAVEN_REQUIRE_AVX512";

fn skips_are_failures() -> bool {
    std::env::var(REQUIRE_ENV).is_ok_and(|v| v == "1")
}

/// Whether the host can execute the AVX-512-IFMA52 kernels.
pub fn have_avx512ifma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512ifma") && is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Gate on the host CPU. Returns `false` (after naming the skip) when the
/// kernels cannot run, and panics instead under `RAVEN_REQUIRE_AVX512=1`.
pub fn require_avx512ifma(test: &str) -> bool {
    if have_avx512ifma() {
        return true;
    }
    assert!(
        !skips_are_failures(),
        "{test}: host lacks AVX-512-IFMA52 (avx512ifma + avx512f) and {REQUIRE_ENV}=1 \
         demands the SIMD kernels actually run. Either run this lane on a runner with \
         those CPU flags or unset {REQUIRE_ENV}."
    );
    eprintln!(
        "LOUD SKIP: {test} did NOT run - host lacks AVX-512-IFMA52. This test proved \
         nothing on this host; set {REQUIRE_ENV}=1 to make that a failure."
    );
    false
}

/// Gate on a cargo feature the differential needs to be a differential at all.
/// Same contract as [`require_avx512ifma`].
pub fn require_cargo_feature(test: &str, feature: &str, enabled: bool) -> bool {
    if enabled {
        return true;
    }
    assert!(
        !skips_are_failures(),
        "{test}: built without --features {feature}, so the dispatch under test is the \
         scalar reference itself and the differential is vacuous; {REQUIRE_ENV}=1 demands \
         a real comparison. Add --features {feature} to this lane."
    );
    eprintln!(
        "LOUD SKIP: {test} did NOT run - built without --features {feature}, which makes \
         both sides of the differential the same code; set {REQUIRE_ENV}=1 to make that \
         a failure."
    );
    false
}
