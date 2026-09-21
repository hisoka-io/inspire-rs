//! Discrete Gaussian sampling over Z for lattice error terms.
//!
//! # Constant-time posture
//!
//! This module's own operations - [`GaussianSampler::sample`],
//! [`GaussianSampler::sample_centered`] and their vector forms - are
//! constant-time with respect to the value drawn. The acceptance probability is
//! tabulated once from the public sigma, read back through a branch-free
//! full-table scan, and compared as an IEEE-754 bit pattern, so no float op and
//! no memory index depends on the drawn value.
//!
//! The reject-resample trip count N stays variable, and that is sound: writing
//! B for the tailcut and p(x) for the acceptance probability,
//! `P(N=n, X=x) = R^(n-1) * p(x)/(2B+1)` with `R` the (constant) per-iteration
//! rejection probability. The joint factorises, so N is independent of X and
//! branching on acceptance reveals nothing about the sample.
//!
//! The property is local, not end-to-end: it covers the draw, not every
//! consumer of it. A caller that branches on, indexes by, or divides by a
//! sample re-opens the channel this module closes.

use crate::math::modular::ct_is_negative;
use rand::rngs::OsRng;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use subtle::{ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater};

/// Default Gaussian standard deviation
pub const DEFAULT_SIGMA: f64 = 3.2;

/// Samples beyond `ceil(6 sigma)` are rejected; `InspireParams::validate` sizes the
/// rounding interval against the same bound.
pub(crate) fn tailcut_for(sigma: f64) -> usize {
    (sigma * 6.0).ceil() as usize
}

/// The platform entropy source refused to produce a seed.
#[derive(Debug, Clone)]
pub struct EntropyUnavailable {
    /// Call site that needed entropy.
    pub what: &'static str,
    /// Underlying `getrandom` / `OsRng` failure text.
    pub detail: String,
}

impl std::fmt::Display for EntropyUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OS entropy unavailable for {}: {}",
            self.what, self.detail
        )
    }
}

impl std::error::Error for EntropyUnavailable {}

/// 32 fresh bytes from the platform entropy source.
pub fn os_seed(what: &'static str) -> Result<[u8; 32], EntropyUnavailable> {
    let mut seed = [0u8; 32];
    OsRng
        .try_fill_bytes(&mut seed)
        .map_err(|e| EntropyUnavailable {
            what,
            detail: e.to_string(),
        })?;
    Ok(seed)
}

/// A ChaCha20 stream keyed by [`os_seed`].
///
/// Library code MUST draw randomness through this rather than `rand::thread_rng()`:
/// it is reproducible when a caller supplies the seed instead, and it needs no
/// thread-local on `wasm32-unknown-unknown`.
pub fn os_seeded_chacha(what: &'static str) -> Result<ChaCha20Rng, EntropyUnavailable> {
    Ok(ChaCha20Rng::from_seed(os_seed(what)?))
}

/// [`os_seeded_chacha`] for entry points whose signature cannot carry the failure.
///
/// Aborts on entropy failure, exactly as the `rand::thread_rng()` call it replaces did.
/// Degrading to a weaker source instead would leak the query index to the server.
#[allow(
    clippy::panic,
    reason = "deliberate abort: a weaker RNG here leaks the query index; os_seeded_chacha is the typed sibling"
)]
pub(crate) fn os_seeded_chacha_or_abort(what: &'static str) -> ChaCha20Rng {
    match os_seeded_chacha(what) {
        Ok(rng) => rng,
        Err(e) => panic!("{e}"),
    }
}

/// Bit patterns of `exp(-x^2 / 2 sigma^2)` for `x` in `[-tailcut, tailcut]`, ascending.
///
/// A NaN probability is only reachable from a degenerate sigma; it maps to `+0.0`
/// so the acceptance test rejects, matching the pre-tabulation `u < NaN` outcome.
fn accept_threshold_table(sigma: f64, tailcut: usize) -> Vec<u64> {
    let sigma_sq_2 = 2.0 * sigma * sigma;
    let bound = tailcut as i64;
    (-bound..=bound)
        .map(|x| {
            let x_sq = (x * x) as f64;
            let prob = (-x_sq / sigma_sq_2).exp();
            if prob.is_nan() {
                0.0f64.to_bits()
            } else {
                prob.to_bits()
            }
        })
        .collect()
}

/// Discrete Gaussian sampler over Z using rejection sampling
#[derive(Clone)]
pub struct GaussianSampler {
    /// Standard deviation σ
    sigma: f64,
    /// Tailcut: reject samples beyond this many standard deviations
    tailcut: usize,
    /// Acceptance thresholds as f64 bit patterns, indexed by `x + tailcut`
    accept_bits: Vec<u64>,
    /// RNG for sampling
    rng: ChaCha20Rng,
}

impl GaussianSampler {
    /// [`Self::from_os_entropy`], aborting when the platform entropy source fails.
    ///
    /// The default constructor draws a fresh 32-byte OS seed, so two samplers
    /// built here never share a stream. Reproducible streams come from
    /// [`Self::with_seed`] or [`Self::from_seed`], which name their seed.
    pub fn new(sigma: f64) -> Self {
        Self::build(sigma, os_seeded_chacha_or_abort("GaussianSampler::new"))
    }

    /// Deterministic sampler over a 64-bit seed; for tests and known-answer vectors.
    pub fn with_seed(sigma: f64, seed: u64) -> Self {
        Self::build(sigma, ChaCha20Rng::seed_from_u64(seed))
    }

    /// Sampler over a caller-supplied full-width ChaCha seed.
    pub fn from_seed(sigma: f64, seed: [u8; 32]) -> Self {
        Self::build(sigma, ChaCha20Rng::from_seed(seed))
    }

    /// Sampler over a fresh 32-byte draw from the platform entropy source.
    pub fn from_os_entropy(sigma: f64) -> Result<Self, EntropyUnavailable> {
        Ok(Self::from_seed(sigma, os_seed("GaussianSampler")?))
    }

    fn build(sigma: f64, rng: ChaCha20Rng) -> Self {
        let tailcut = tailcut_for(sigma);
        Self {
            sigma,
            tailcut,
            accept_bits: accept_threshold_table(sigma, tailcut),
            rng,
        }
    }

    /// Get the standard deviation
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Sample a single value from the discrete Gaussian D_σ
    /// Returns a signed integer in centered representation
    pub fn sample(&mut self) -> i64 {
        self.sample_rejection()
    }

    /// Sample a single value as unsigned, centered around 0
    /// Positive values: 0, 1, 2, ...
    /// Negative values: q-1, q-2, ... (represented as large positive in Z_q)
    pub fn sample_centered(&mut self, q: u64) -> u64 {
        let s = self.sample();
        let wrapped = q.wrapping_add(s as u64);
        u64::conditional_select(&(s as u64), &wrapped, ct_is_negative(s))
    }

    /// Sample a vector of Gaussian values
    pub fn sample_vec(&mut self, len: usize) -> Vec<i64> {
        (0..len).map(|_| self.sample()).collect()
    }

    /// Sample a vector of Gaussian values as unsigned mod q
    pub fn sample_vec_centered(&mut self, len: usize, q: u64) -> Vec<u64> {
        (0..len).map(|_| self.sample_centered(q)).collect()
    }

    /// Acceptance threshold for `x`, read without indexing memory by `x`.
    fn accept_threshold_bits(&self, x: i64) -> u64 {
        let base = self.tailcut as i64;
        let mut acc = 0u64;
        for (i, &bits) in self.accept_bits.iter().enumerate() {
            acc |= u64::conditional_select(&0, &bits, (i as i64 - base).ct_eq(&x));
        }
        acc
    }

    /// Rejection sampling for discrete Gaussian.
    ///
    /// `threshold > u.to_bits()` is exactly `u < prob`: both operands are
    /// non-negative and non-NaN, and IEEE-754 orders non-negative floats the same
    /// way an unsigned integer orders their bit patterns. Keeping that equivalence
    /// is what makes the tabulated sampler emit the same stream as the float one.
    fn sample_rejection(&mut self) -> i64 {
        let bound = self.tailcut as i64;

        loop {
            let x = self.rng.gen_range(-bound..=bound);
            let threshold = self.accept_threshold_bits(x);
            let u: f64 = self.rng.gen_range(0.0..1.0);
            if bool::from(threshold.ct_gt(&u.to_bits())) {
                return x;
            }
        }
    }

    /// Replace the stream with one keyed by a caller-supplied full-width seed.
    ///
    /// Takes 32 bytes, not a `u64`: a `u64` reseed caps the stream at 64 bits of
    /// entropy however the sampler was originally built.
    pub fn reseed(&mut self, seed: [u8; 32]) {
        self.rng = ChaCha20Rng::from_seed(seed);
    }

    /// [`Self::reseed`] over a fresh draw from the platform entropy source.
    pub fn reseed_from_os_entropy(&mut self) -> Result<(), EntropyUnavailable> {
        self.rng = os_seeded_chacha("GaussianSampler::reseed_from_os_entropy")?;
        Ok(())
    }
}

#[allow(
    clippy::missing_fields_in_debug,
    reason = "the ChaCha20 state is secret and must never reach a log line"
)]
impl std::fmt::Debug for GaussianSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GaussianSampler")
            .field("sigma", &self.sigma)
            .field("tailcut", &self.tailcut)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::mod_q::DEFAULT_Q;
    use std::collections::HashMap;

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    #[test]
    fn ct_eq_matches_equality() {
        for a in [i64::MIN, -21, -1, 0, 1, 20, 21, i64::MAX] {
            for b in [i64::MIN, -21, -1, 0, 1, 20, 21, i64::MAX] {
                assert_eq!(bool::from(a.ct_eq(&b)), a == b, "a={a} b={b}");
            }
        }
    }

    #[test]
    fn ct_gt_matches_unsigned_comparison() {
        let corners = [
            0u64,
            1,
            2,
            u64::MAX,
            u64::MAX - 1,
            1 << 63,
            (1 << 63) - 1,
            (1 << 63) + 1,
            0x3ff0_0000_0000_0000,
            0x0010_0000_0000_0000,
        ];
        for &a in &corners {
            for &b in &corners {
                assert_eq!(bool::from(b.ct_gt(&a)), a < b, "a={a:#x} b={b:#x}");
            }
        }
        let mut rng = ChaCha20Rng::seed_from_u64(0xC7);
        for _ in 0..100_000 {
            let (a, b): (u64, u64) = (rng.gen(), rng.gen());
            assert_eq!(bool::from(b.ct_gt(&a)), a < b);
            let shared = a & 0xffff_ffff_0000_0000;
            let (a2, b2) = (shared | (a >> 32), shared | (b >> 32));
            assert_eq!(bool::from(b2.ct_gt(&a2)), a2 < b2);
        }
    }

    /// The tabulated threshold reproduces the float expression it replaced, for
    /// every reachable `x`, so the accept/reject decision cannot drift.
    #[test]
    fn tabulated_threshold_equals_runtime_float_expression() {
        for sigma in [1.0f64, 3.2, 6.4, 10.0] {
            let sampler = GaussianSampler::with_seed(sigma, 0);
            let sigma_sq_2 = 2.0 * sigma * sigma;
            let bound = sampler.tailcut as i64;
            for x in -bound..=bound {
                let x_sq = (x * x) as f64;
                let prob = (-x_sq / sigma_sq_2).exp();
                let threshold = sampler.accept_threshold_bits(x);
                assert_eq!(threshold, prob.to_bits(), "sigma={sigma} x={x}");
                for u in [0.0f64, 0.5, 0.999_999, prob * 0.5, prob] {
                    assert_eq!(
                        bool::from(threshold.ct_gt(&u.to_bits())),
                        u < prob,
                        "sigma={sigma} x={x} u={u}"
                    );
                }
            }
            assert_eq!(sampler.accept_threshold_bits(bound + 1), 0);
            assert_eq!(sampler.accept_threshold_bits(-bound - 1), 0);
        }
    }

    /// Frozen hashes of the pre-tabulation sampler. Any change to the draw order,
    /// the tailcut, or the acceptance decision moves these.
    #[test]
    fn seeded_stream_is_unchanged() {
        let golden: [(f64, u64, u64); 5] = [
            (3.2, 0, 0xd6a6_e62f_7739_6bd4),
            (3.2, 12345, 0x270f_db99_e427_4779),
            (6.4, 1, 0x220c_f0ea_2dc8_e0e5),
            (1.0, 42, 0x5a0a_b704_ab07_75a2),
            (10.0, 7, 0x503d_f016_7ffd_586d),
        ];
        for (sigma, seed, want) in golden {
            let mut sampler = GaussianSampler::with_seed(sigma, seed);
            let samples: Vec<i64> = (0..200_000).map(|_| sampler.sample()).collect();
            let bytes: Vec<u8> = samples.iter().flat_map(|x| x.to_le_bytes()).collect();
            assert_eq!(fnv1a(&bytes), want, "sigma={sigma} seed={seed}");
        }

        let mut sampler = GaussianSampler::from_seed(3.2, [7u8; 32]);
        let centered = sampler.sample_vec_centered(50_000, 1_152_921_504_606_830_593);
        let bytes: Vec<u8> = centered.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(fnv1a(&bytes), 0x56fd_cced_8a63_0609);
    }

    #[test]
    fn from_os_entropy_does_not_repeat() {
        let a: Vec<i64> = (0..32)
            .map(|_| {
                GaussianSampler::from_os_entropy(DEFAULT_SIGMA)
                    .expect("OS entropy")
                    .sample()
            })
            .collect();
        let b: Vec<i64> = (0..32)
            .map(|_| {
                GaussianSampler::from_os_entropy(DEFAULT_SIGMA)
                    .expect("OS entropy")
                    .sample()
            })
            .collect();
        assert_ne!(a, b, "from_os_entropy produced a repeating stream");
    }

    /// The default constructor must not be a fixed seed. A whole cohort of
    /// clients built from it would otherwise share one secret key, and an
    /// observer who runs the same constructor recovers every query index.
    #[test]
    fn default_constructor_streams_differ_across_instances() {
        let draw = || {
            let mut sampler = GaussianSampler::new(DEFAULT_SIGMA);
            sampler.sample_vec(64)
        };
        let runs: Vec<Vec<i64>> = (0..8).map(|_| draw()).collect();
        for (i, a) in runs.iter().enumerate() {
            for b in runs.iter().skip(i + 1) {
                assert_ne!(a, b, "GaussianSampler::new repeats its stream");
            }
        }

        let keys: Vec<Vec<u64>> = (0..8)
            .map(|_| GaussianSampler::new(DEFAULT_SIGMA).sample_vec_centered(64, DEFAULT_Q))
            .collect();
        for (i, a) in keys.iter().enumerate() {
            for b in keys.iter().skip(i + 1) {
                assert_ne!(a, b, "GaussianSampler::new mints a repeatable secret");
            }
        }
    }

    /// Explicit seeds stay byte-reproducible; that is what the KATs rest on.
    #[test]
    fn explicit_seed_constructors_are_reproducible() {
        let from_u64: Vec<i64> = {
            let mut a = GaussianSampler::with_seed(DEFAULT_SIGMA, 12345);
            let mut b = GaussianSampler::with_seed(DEFAULT_SIGMA, 12345);
            assert_eq!(a.sample_vec(512), b.sample_vec(512));
            GaussianSampler::with_seed(DEFAULT_SIGMA, 12345).sample_vec(8)
        };
        let mut c = GaussianSampler::from_seed(DEFAULT_SIGMA, [3u8; 32]);
        let mut d = GaussianSampler::from_seed(DEFAULT_SIGMA, [3u8; 32]);
        assert_eq!(c.sample_vec(512), d.sample_vec(512));
        assert_ne!(from_u64, c.sample_vec(8));
    }

    /// Library randomness must be an OS-seeded ChaCha20 stream, never an ambient
    /// thread-local: the thread-local generator is not reproducible under a
    /// caller-supplied seed and has no home on `wasm32-unknown-unknown`.
    #[test]
    fn no_ambient_thread_rng_in_crate_sources() {
        fn walk(dir: &std::path::Path, needle: &str, hits: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).expect("read_dir") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    walk(&path, needle, hits);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let text = std::fs::read_to_string(&path).expect("read source");
                    for (i, line) in text.lines().enumerate() {
                        if !line.trim_start().starts_with("//") && line.contains(needle) {
                            hits.push(format!("{}:{}", path.display(), i + 1));
                        }
                    }
                }
            }
        }
        // split so this assertion is not its own counterexample
        let needle = ["thread", "_rng()"].concat();
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut hits = Vec::new();
        walk(&src, &needle, &mut hits);
        assert!(hits.is_empty(), "ambient thread rng reachable at {hits:?}");
    }

    #[test]
    fn test_basic_sampling() {
        let mut sampler = GaussianSampler::new(DEFAULT_SIGMA);

        let tailcut_bound = (6.0 * DEFAULT_SIGMA).ceil() as i64;
        for _ in 0..1000 {
            let s = sampler.sample();
            assert!(
                s.abs() <= tailcut_bound,
                "Sample {s} exceeds tailcut bound {tailcut_bound}"
            );
        }
    }

    #[test]
    fn test_deterministic_seeding() {
        let mut sampler1 = GaussianSampler::with_seed(DEFAULT_SIGMA, 12345);
        let mut sampler2 = GaussianSampler::with_seed(DEFAULT_SIGMA, 12345);

        for _ in 0..100 {
            assert_eq!(sampler1.sample(), sampler2.sample());
        }
    }

    #[test]
    fn test_different_seeds() {
        let mut sampler1 = GaussianSampler::with_seed(DEFAULT_SIGMA, 12345);
        let mut sampler2 = GaussianSampler::with_seed(DEFAULT_SIGMA, 54321);

        let samples1: Vec<i64> = (0..100).map(|_| sampler1.sample()).collect();
        let samples2: Vec<i64> = (0..100).map(|_| sampler2.sample()).collect();

        assert_ne!(samples1, samples2);
    }

    #[test]
    fn test_centered_representation() {
        let q: u64 = 1152921504606830593;
        let mut sampler = GaussianSampler::new(DEFAULT_SIGMA);

        for _ in 0..1000 {
            let s = sampler.sample_centered(q);
            let centered = if s <= q / 2 {
                s as i64
            } else {
                s as i64 - q as i64
            };
            assert!(centered.abs() <= (6.0 * DEFAULT_SIGMA).ceil() as i64);
        }
    }

    #[test]
    fn test_sample_vec() {
        let mut sampler = GaussianSampler::new(DEFAULT_SIGMA);
        let vec = sampler.sample_vec(100);
        assert_eq!(vec.len(), 100);
    }

    #[test]
    fn test_distribution_symmetry() {
        let mut sampler = GaussianSampler::with_seed(DEFAULT_SIGMA, 42);
        let n = 100_000;

        let mut pos_count = 0;
        let mut neg_count = 0;
        let mut zero_count = 0;

        for _ in 0..n {
            match sampler.sample().cmp(&0) {
                std::cmp::Ordering::Greater => pos_count += 1,
                std::cmp::Ordering::Less => neg_count += 1,
                std::cmp::Ordering::Equal => zero_count += 1,
            }
        }

        let ratio = pos_count as f64 / neg_count as f64;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "Distribution not symmetric: pos={pos_count}, neg={neg_count}, ratio={ratio}"
        );

        assert!(zero_count > n / 50, "Zero count {zero_count} is too low");
    }

    #[test]
    fn test_distribution_mean() {
        let mut sampler = GaussianSampler::with_seed(DEFAULT_SIGMA, 42);
        let n = 100_000;

        let sum: i64 = (0..n).map(|_| sampler.sample()).sum();
        let mean = sum as f64 / n as f64;

        assert!(mean.abs() < 0.1, "Mean {mean} is too far from 0");
    }

    #[test]
    fn test_distribution_variance() {
        let mut sampler = GaussianSampler::with_seed(DEFAULT_SIGMA, 42);
        let n = 100_000;

        let samples: Vec<i64> = (0..n).map(|_| sampler.sample()).collect();
        let mean: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let variance: f64 = samples
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;

        let expected_variance = DEFAULT_SIGMA * DEFAULT_SIGMA;
        let relative_error = (variance - expected_variance).abs() / expected_variance;

        assert!(
            relative_error < 0.1,
            "Variance {} differs from expected {} by {:.1}%",
            variance,
            expected_variance,
            relative_error * 100.0
        );
    }

    #[test]
    fn test_distribution_shape() {
        let mut sampler = GaussianSampler::with_seed(DEFAULT_SIGMA, 42);
        let n = 100_000;

        let mut histogram: HashMap<i64, usize> = HashMap::new();
        for _ in 0..n {
            let s = sampler.sample();
            *histogram.entry(s).or_insert(0) += 1;
        }

        let count_0 = *histogram.get(&0).unwrap_or(&0);
        let count_5 = *histogram.get(&5).unwrap_or(&0) + *histogram.get(&-5).unwrap_or(&0);
        let count_10 = *histogram.get(&10).unwrap_or(&0) + *histogram.get(&-10).unwrap_or(&0);

        assert!(
            count_0 > count_5,
            "0 should be more frequent than ±5: {count_0} vs {count_5}"
        );
        assert!(
            count_5 > count_10,
            "±5 should be more frequent than ±10: {count_5} vs {count_10}"
        );
    }

    #[test]
    fn test_tailcut_bounds() {
        let sigma = 3.2;
        let mut sampler = GaussianSampler::new(sigma);
        let tailcut_bound = (6.0 * sigma).ceil() as i64;

        for _ in 0..100_000 {
            let s = sampler.sample();
            assert!(
                s.abs() <= tailcut_bound,
                "Sample {s} exceeds 6σ bound of {tailcut_bound}"
            );
        }
    }

    #[test]
    fn test_reseed() {
        let mut sampler = GaussianSampler::with_seed(DEFAULT_SIGMA, 0);

        sampler.reseed([9u8; 32]);
        let samples1: Vec<i64> = (0..10).map(|_| sampler.sample()).collect();

        sampler.reseed([9u8; 32]);
        let samples2: Vec<i64> = (0..10).map(|_| sampler.sample()).collect();

        assert_eq!(samples1, samples2);
    }

    /// A reseed must carry the full 256-bit key, not a 64-bit expansion: two
    /// seeds that agree on their low 64 bits must still diverge.
    #[test]
    fn reseed_consumes_the_whole_seed() {
        let mut low_only = [0u8; 32];
        low_only[..8].copy_from_slice(&7u64.to_le_bytes());
        let mut high_too = low_only;
        high_too[31] = 1;

        let mut a = GaussianSampler::with_seed(DEFAULT_SIGMA, 0);
        a.reseed(low_only);
        let mut b = GaussianSampler::with_seed(DEFAULT_SIGMA, 0);
        b.reseed(high_too);
        assert_ne!(a.sample_vec(64), b.sample_vec(64));
    }

    #[test]
    fn reseed_from_os_entropy_breaks_a_fixed_stream() {
        let mut a = GaussianSampler::with_seed(DEFAULT_SIGMA, 4);
        let mut b = GaussianSampler::with_seed(DEFAULT_SIGMA, 4);
        a.reseed_from_os_entropy().expect("OS entropy");
        b.reseed_from_os_entropy().expect("OS entropy");
        assert_ne!(a.sample_vec(64), b.sample_vec(64));
    }

    /// `sample_centered` maps the sign branch-free; it must still agree with the
    /// arithmetic it replaced on every reachable sample.
    #[test]
    fn sample_centered_matches_signed_lift() {
        let mut sampler = GaussianSampler::with_seed(DEFAULT_SIGMA, 31);
        let mut reference = GaussianSampler::with_seed(DEFAULT_SIGMA, 31);
        for q in [DEFAULT_Q, 12_289u64, u64::MAX] {
            for _ in 0..5_000 {
                let s = reference.sample();
                let want = if s >= 0 {
                    s as u64
                } else {
                    q.wrapping_add(s as u64)
                };
                assert_eq!(sampler.sample_centered(q), want, "q={q} s={s}");
            }
        }
    }

    #[test]
    fn test_different_sigma() {
        let small_sigma = 1.0;
        let large_sigma = 10.0;

        let mut small_sampler = GaussianSampler::with_seed(small_sigma, 42);
        let mut large_sampler = GaussianSampler::with_seed(large_sigma, 42);

        let n = 10_000;
        let small_variance: f64 = {
            let samples: Vec<i64> = (0..n).map(|_| small_sampler.sample()).collect();
            let mean: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
            samples
                .iter()
                .map(|&x| (x as f64 - mean).powi(2))
                .sum::<f64>()
                / n as f64
        };

        let large_variance: f64 = {
            let samples: Vec<i64> = (0..n).map(|_| large_sampler.sample()).collect();
            let mean: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
            samples
                .iter()
                .map(|&x| (x as f64 - mean).powi(2))
                .sum::<f64>()
                / n as f64
        };

        assert!(
            large_variance > small_variance * 10.0,
            "Large sigma should have much larger variance"
        );
    }
}
