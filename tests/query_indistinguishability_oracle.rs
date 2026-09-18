#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    reason = "research-test diagnostics and fixed experimental fixtures"
)]
//! Statistical regression oracle for same-shard shipping query transcripts.
//!
//! This is not an IND-CPA proof and makes no whole-query constant-time claim.
//! Fixed public constants below implement a pre-registered protocol.
//! It covers one fixed d=256 session, indices 0 and 255 in one shard, and seeded
//! and unseeded one-row fold bytes at the power demonstrated by a 25% one-byte leak.
//! It excludes the clear shard id, packing keys and handles, adaptive queries,
//! other parameters and indices, responses, OS entropy, arbitrary nonlinear
//! distinguishers, and every timing, cache, or power channel. A pass neither
//! establishes 128-bit advantage nor closes known Poly/NTT residuals.

use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::{setup_with_rng, ClientSession};

const TRAIN_PER_CLASS: usize = 256;
const HELD_OUT_PER_CLASS: usize = 256;
const PERMUTATIONS: usize = 8_191;
const SKETCH_BUCKETS: usize = 64;
const FIXED_POSITIONS: usize = 4;
const FEATURE_COUNT: usize = SKETCH_BUCKETS + FIXED_POSITIONS;

const KEY_SEED: [u8; 32] = [0x11; 32];
const SAMPLE_SEED: [u8; 32] = [0x22; 32];
const SCHEDULE_SEED: [u8; 32] = [0x33; 32];
const FEATURE_SEED: [u8; 32] = [0x44; 32];
const PERMUTATION_SEED: [u8; 32] = [0x55; 32];

#[derive(Clone)]
struct RawSplit {
    train_left: Vec<Vec<u8>>,
    train_right: Vec<Vec<u8>>,
    held_out_left: Vec<Vec<u8>>,
    held_out_right: Vec<Vec<u8>>,
}

#[derive(Debug)]
struct OracleVerdict {
    selected_feature: usize,
    train_ks_numerator: u64,
    held_out_ks_numerator: u64,
    p_numerator: u64,
    p_denominator: u64,
}

impl OracleVerdict {
    fn rejects_at_reciprocal(&self, reciprocal: u64) -> bool {
        self.p_numerator.saturating_mul(reciprocal) <= self.p_denominator
    }
}

#[derive(Clone, Copy, Debug)]
enum QueryForm {
    Unseeded,
    Seeded,
}

impl QueryForm {
    const fn name(self) -> &'static str {
        match self {
            Self::Unseeded => "unseeded fold row index 0 vs 255",
            Self::Seeded => "seeded fold row index 0 vs 255",
        }
    }
}

#[derive(Clone, Copy)]
enum Challenge {
    Null,
    EndpointIndices,
}

fn test_params() -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65_536,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

fn balanced_schedule() -> Vec<bool> {
    let per_class = TRAIN_PER_CLASS + HELD_OUT_PER_CLASS;
    let mut labels = vec![false; per_class];
    labels.extend(std::iter::repeat_n(true, per_class));
    labels.shuffle(&mut ChaCha20Rng::from_seed(SCHEDULE_SEED));
    labels
}

fn collect_raw(mut generate: impl FnMut(bool, &mut ChaCha20Rng) -> Vec<u8>) -> RawSplit {
    let mut split = RawSplit {
        train_left: Vec::with_capacity(TRAIN_PER_CLASS),
        train_right: Vec::with_capacity(TRAIN_PER_CLASS),
        held_out_left: Vec::with_capacity(HELD_OUT_PER_CLASS),
        held_out_right: Vec::with_capacity(HELD_OUT_PER_CLASS),
    };
    let mut entropy = ChaCha20Rng::from_seed(SAMPLE_SEED);

    for right in balanced_schedule() {
        let transcript = generate(right, &mut entropy);
        let (train, held_out) = if right {
            (&mut split.train_right, &mut split.held_out_right)
        } else {
            (&mut split.train_left, &mut split.held_out_left)
        };
        if train.len() < TRAIN_PER_CLASS {
            train.push(transcript);
        } else {
            held_out.push(transcript);
        }
    }

    assert_eq!(split.train_left.len(), TRAIN_PER_CLASS);
    assert_eq!(split.train_right.len(), TRAIN_PER_CLASS);
    assert_eq!(split.held_out_left.len(), HELD_OUT_PER_CLASS);
    assert_eq!(split.held_out_right.len(), HELD_OUT_PER_CLASS);
    split
}

fn synthetic_control(partial_leak: bool) -> RawSplit {
    let raw = collect_raw(|_, entropy| {
        let mut transcript = vec![0u8; 512];
        entropy.fill_bytes(&mut transcript);
        transcript
    });
    if partial_leak {
        with_partial_byte_leak(raw)
    } else {
        raw
    }
}

fn rgsw_samples(form: QueryForm, challenge: Challenge) -> RawSplit {
    let params = test_params();
    let database = vec![0u8; params.ring_dim * 32];
    let mut setup_sampler = GaussianSampler::from_seed(params.sigma, KEY_SEED);
    let mut setup_rng = ChaCha20Rng::from_seed(KEY_SEED);
    let (crs, encoded, secret_key) = setup_with_rng(
        &params,
        &database,
        32,
        &mut setup_sampler,
        &mut setup_rng,
    )
    .expect("deterministic setup");
    let mut session_sampler = GaussianSampler::from_seed(params.sigma, KEY_SEED);
    let session = ClientSession::new(crs, secret_key, &mut session_sampler).expect("session");

    collect_raw(|right, entropy| {
        let local_index = match challenge {
            Challenge::EndpointIndices if right => params.ring_dim - 1,
            Challenge::Null | Challenge::EndpointIndices => 0,
        };
        let mut gaussian_seed = [0u8; 32];
        entropy.fill_bytes(&mut gaussian_seed);
        let mut sampler = GaussianSampler::from_seed(params.sigma, gaussian_seed);

        match form {
            QueryForm::Unseeded => {
                let (_, query) = session
                    .query(local_index as u64, &encoded.config, &mut sampler)
                    .expect("unseeded shipping query");
                bincode::serialize(&query.rgsw_ciphertext)
                    .expect("unseeded fold-row serialization")
            }
            QueryForm::Seeded => {
                let (_, query) = session
                    .query_seeded(local_index as u64, &encoded.config, &mut sampler)
                    .expect("seeded shipping query");
                bincode::serialize(&query.rgsw_ciphertext)
                    .expect("seeded fold-row serialization")
            }
        }
    })
}

fn with_partial_byte_leak(mut split: RawSplit) -> RawSplit {
    for transcripts in [&mut split.train_right, &mut split.held_out_right] {
        for transcript in transcripts.iter_mut().step_by(4) {
            let first = transcript
                .first_mut()
                .expect("registered transcripts are non-empty");
            *first = 0xff;
        }
    }
    split
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn opaque_features(transcript: &[u8]) -> Vec<i64> {
    assert!(
        transcript.len() >= 2,
        "transcript must contain at least two bytes"
    );
    let mut features = vec![0i64; FEATURE_COUNT];
    let feature_word = u64::from_le_bytes([
        FEATURE_SEED[0],
        FEATURE_SEED[1],
        FEATURE_SEED[2],
        FEATURE_SEED[3],
        FEATURE_SEED[4],
        FEATURE_SEED[5],
        FEATURE_SEED[6],
        FEATURE_SEED[7],
    ]);
    for (offset, &byte) in transcript.iter().enumerate() {
        let hash = splitmix64(feature_word ^ offset as u64);
        let bucket = hash as usize % SKETCH_BUCKETS;
        let contribution = i64::from(byte);
        if hash >> 63 == 0 {
            features[bucket] += contribution;
        } else {
            features[bucket] -= contribution;
        }
    }
    let positions = [0, 1, transcript.len() / 2, transcript.len() - 1];
    for (feature, position) in positions.into_iter().enumerate() {
        features[SKETCH_BUCKETS + feature] = i64::from(transcript[position]);
    }
    features
}

fn feature_rows(transcripts: &[Vec<u8>], expected_len: usize) -> Vec<Vec<i64>> {
    transcripts
        .iter()
        .map(|transcript| {
            assert_eq!(
                transcript.len(),
                expected_len,
                "wire-length difference is an immediate transcript distinguisher"
            );
            opaque_features(transcript)
        })
        .collect()
}

fn values_and_labels(
    left: &[Vec<i64>],
    right: &[Vec<i64>],
    feature: usize,
) -> (Vec<i64>, Vec<bool>) {
    let mut values = Vec::with_capacity(left.len() + right.len());
    let mut labels = Vec::with_capacity(values.capacity());
    for row in left {
        values.push(row[feature]);
        labels.push(false);
    }
    for row in right {
        values.push(row[feature]);
        labels.push(true);
    }
    (values, labels)
}

fn sorted_order(values: &[i64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_unstable_by_key(|&index| values[index]);
    order
}

fn ks_numerator(values: &[i64], labels: &[bool], order: &[usize]) -> u64 {
    assert_eq!(values.len(), labels.len());
    let right_total = labels.iter().filter(|&&right| right).count() as u64;
    let left_total = labels.len() as u64 - right_total;
    assert!(left_total > 0 && right_total > 0);

    let mut left_seen = 0u64;
    let mut right_seen = 0u64;
    let mut maximum = 0u64;
    let mut cursor = 0usize;
    while cursor < order.len() {
        let tied_value = values[order[cursor]];
        while cursor < order.len() && values[order[cursor]] == tied_value {
            if labels[order[cursor]] {
                right_seen += 1;
            } else {
                left_seen += 1;
            }
            cursor += 1;
        }
        let distance = (left_seen * right_total).abs_diff(right_seen * left_total);
        maximum = maximum.max(distance);
    }
    maximum
}

fn permutation_verdict(
    values: &[i64],
    labels: &[bool],
    selected_feature: usize,
    train_ks: u64,
) -> OracleVerdict {
    let order = sorted_order(values);
    let observed = ks_numerator(values, labels, &order);
    let mut permuted_labels = labels.to_vec();
    let mut rng = ChaCha20Rng::from_seed(PERMUTATION_SEED);
    let mut exceedances = 0u64;
    for _ in 0..PERMUTATIONS {
        permuted_labels.shuffle(&mut rng);
        if ks_numerator(values, &permuted_labels, &order) >= observed {
            exceedances += 1;
        }
    }
    OracleVerdict {
        selected_feature,
        train_ks_numerator: train_ks,
        held_out_ks_numerator: observed,
        p_numerator: exceedances + 1,
        p_denominator: PERMUTATIONS as u64 + 1,
    }
}

fn evaluate(raw: &RawSplit) -> OracleVerdict {
    let expected_len = raw
        .train_left
        .first()
        .expect("registered training samples are non-empty")
        .len();
    let train_left = feature_rows(&raw.train_left, expected_len);
    let train_right = feature_rows(&raw.train_right, expected_len);
    let held_out_left = feature_rows(&raw.held_out_left, expected_len);
    let held_out_right = feature_rows(&raw.held_out_right, expected_len);

    let mut selected_feature = 0usize;
    let mut selected_train_ks = 0u64;
    for feature in 0..FEATURE_COUNT {
        let (values, labels) = values_and_labels(&train_left, &train_right, feature);
        let order = sorted_order(&values);
        let distance = ks_numerator(&values, &labels, &order);
        if distance > selected_train_ks {
            selected_feature = feature;
            selected_train_ks = distance;
        }
    }

    let (values, labels) = values_and_labels(&held_out_left, &held_out_right, selected_feature);
    permutation_verdict(&values, &labels, selected_feature, selected_train_ks)
}

fn report(name: &str, verdict: &OracleVerdict) {
    eprintln!(
        "{name}: feature={} train_ks_num={} held_out_ks_num={} p={}/{}",
        verdict.selected_feature,
        verdict.train_ks_numerator,
        verdict.held_out_ks_numerator,
        verdict.p_numerator,
        verdict.p_denominator,
    );
}

#[test]
fn instrument_does_not_reject_the_preregistered_null() {
    let null = synthetic_control(false);
    let verdict = evaluate(&null);
    report("synthetic null", &verdict);
    assert!(!verdict.rejects_at_reciprocal(20), "{verdict:?}");
}

#[test]
fn instrument_distinguishes_the_preregistered_partial_byte_leak() {
    let positive = synthetic_control(true);
    let verdict = evaluate(&positive);
    report("synthetic positive", &verdict);
    assert!(verdict.rejects_at_reciprocal(200), "{verdict:?}");
}

#[test]
#[ignore = "3072 d=256 RGSW encryptions plus 8191-permutation inference; run when query, RGSW encryption, serialization, or randomness changes and for privacy review"]
fn same_shard_query_transcript_research_oracle() {
    let null = rgsw_samples(QueryForm::Unseeded, Challenge::Null);
    let positive = with_partial_byte_leak(null.clone());
    let null_verdict = evaluate(&null);
    let positive_verdict = evaluate(&positive);
    report("actual-transcript null", &null_verdict);
    report("actual-transcript positive", &positive_verdict);
    assert!(!null_verdict.rejects_at_reciprocal(20), "{null_verdict:?}");
    assert!(
        positive_verdict.rejects_at_reciprocal(200),
        "{positive_verdict:?}"
    );

    for form in [QueryForm::Unseeded, QueryForm::Seeded] {
        let samples = rgsw_samples(form, Challenge::EndpointIndices);
        let verdict = evaluate(&samples);
        report(form.name(), &verdict);
        assert!(!verdict.rejects_at_reciprocal(200), "{form:?}: {verdict:?}");
    }
}
