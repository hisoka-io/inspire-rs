//! PIR round-trip property: for any legal cell, extract(respond(query(idx)))
//! returns exactly db[idx*entry_size..][..entry_size], and the client query's
//! shard_id is idx / entries_per_shard. Axes: entry size, entry count (with the
//! non-power-of-two 3), fill pattern (random / all-zeros / all-0xff), variant
//! (NoPacking / InspiRING) and CRT modulus shape (single-prime DEFAULT_Q, the
//! Google-derivation 27-bit pair, the 30-bit override pair) — the modulus axis
//! is what the retired single-modulus examples never drove. Every case also
//! walks both boundary indices 0 and num_entries-1. Scale cells (d=2048,
//! entries=2^14) stay in commit_e_two_crt_regression_grid.rs, whose enumerated
//! byte oracle over expensive cells is deliberately kept.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]

use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use raven_inspire::math::mod_q::DEFAULT_Q;
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel, DEFAULT_Q_2CRT_30BIT};
use raven_inspire::pir::{
    extract, extract_inspiring, query, respond, respond_inspiring, setup, PackingMode,
};

/// Same 27-bit pair the Google `for_scenario` derivation emits and the
/// commit-E regression grid pins; a test fixture, not a shipped parameter.
const GOOGLE_CRT: [u64; 2] = [67_043_329, 132_120_577];

fn params_for(crt_moduli: Vec<u64>) -> InspireParams {
    let q: u64 = crt_moduli.iter().product();
    InspireParams {
        ring_dim: 256,
        q,
        crt_moduli,
        p: 65537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

#[derive(Debug, Clone, Copy)]
enum Fill {
    Random,
    Zeros,
    Ff,
}

#[derive(Debug, Clone, Copy)]
enum Variant {
    NoPacking,
    Inspiring,
}

fn build_db(fill: Fill, num_entries: usize, entry_size: usize, seed: u64) -> Vec<u8> {
    let total = num_entries * entry_size;
    match fill {
        Fill::Zeros => vec![0u8; total],
        Fill::Ff => vec![0xFFu8; total],
        Fill::Random => {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut db = vec![0u8; total];
            rng.fill(&mut db[..]);
            db
        }
    }
}

/// One full cell check: setup once, then round-trip the drawn index plus both
/// boundary indices, asserting shard routing and exact bytes each time.
fn check_cell(
    crt_moduli: Vec<u64>,
    num_entries: usize,
    entry_size: usize,
    idx: usize,
    fill: Fill,
    variant: Variant,
    seed: u64,
) -> Result<(), TestCaseError> {
    let params = params_for(crt_moduli);
    let db = build_db(fill, num_entries, entry_size, seed);

    let mut sampler = GaussianSampler::with_seed(params.sigma, seed);
    let (crs, encoded_db, rlwe_sk) =
        setup(&params, &db, entry_size, &mut sampler).expect("setup must succeed on a legal cell");

    let entries_per_shard = usize::try_from(encoded_db.config.entries_per_shard()).unwrap();
    prop_assert!(entries_per_shard > 0, "degenerate shard geometry");

    let mut indices = vec![idx, 0, num_entries - 1];
    indices.dedup();

    for target in indices {
        let (state, client_query) = query(
            &crs,
            target as u64,
            &encoded_db.config,
            &rlwe_sk,
            &mut sampler,
        )
        .expect("query must succeed for an in-range index");

        prop_assert_eq!(
            client_query.shard_id as usize,
            target / entries_per_shard,
            "shard routing broke at idx {} (entries_per_shard {})",
            target,
            entries_per_shard
        );

        let recovered = match variant {
            Variant::NoPacking => {
                prop_assert_eq!(client_query.packing_mode, PackingMode::Inspiring);
                let response = respond(&crs, &encoded_db, &client_query)
                    .expect("NoPacking respond must succeed");
                extract(&crs, &state, &response, entry_size)
                    .expect("NoPacking extract must succeed")
            }
            Variant::Inspiring => {
                let response = respond_inspiring(&crs, &encoded_db, &client_query)
                    .expect("InspiRING respond must succeed");
                extract_inspiring(&crs, &state, &response, entry_size)
                    .expect("InspiRING extract must succeed")
            }
        };

        let expected = &db[target * entry_size..(target + 1) * entry_size];
        prop_assert_eq!(
            recovered.as_slice(),
            expected,
            "wrong bytes at idx {} ({:?}, entry_size {}, entries {})",
            target,
            variant,
            entry_size,
            num_entries
        );
    }
    Ok(())
}

fn modulus_axis() -> impl Strategy<Value = Vec<u64>> {
    prop_oneof![
        Just(vec![DEFAULT_Q]),
        Just(GOOGLE_CRT.to_vec()),
        Just(DEFAULT_Q_2CRT_30BIT.to_vec()),
    ]
}

fn fill_axis() -> impl Strategy<Value = Fill> {
    prop_oneof![
        3 => Just(Fill::Random),
        1 => Just(Fill::Zeros),
        1 => Just(Fill::Ff),
    ]
}

fn variant_axis() -> impl Strategy<Value = Variant> {
    prop_oneof![Just(Variant::NoPacking), Just(Variant::Inspiring)]
}

/// num_entries mixes the non-power-of-two 3, sub-shard sizes, multi-shard
/// sizes (a d=256 shard holds 256 entries, so >256 forces shard 1+), and the
/// exact shard-fill counts whose last index sits on the routing boundary.
fn num_entries_axis() -> impl Strategy<Value = usize> {
    prop_oneof![
        1 => Just(3usize),
        2 => 4usize..=256,
        2 => 257usize..=640,
        1 => Just(256usize),
        1 => Just(512usize),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 16,
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    #[test]
    fn pir_round_trip_recovers_db_idx(
        (crt, num_entries, entry_size, idx, fill, variant, seed) in (
            modulus_axis(),
            num_entries_axis(),
            prop::sample::select(&[2usize, 8, 16, 32, 64, 128, 256]),
            fill_axis(),
            variant_axis(),
            any::<u64>(),
        ).prop_flat_map(|(crt, ne, es, fill, variant, seed)| {
            (Just(crt), Just(ne), Just(es), 0..ne, Just(fill), Just(variant), Just(seed))
        })
    ) {
        check_cell(crt, num_entries, entry_size, idx, fill, variant, seed)?;
    }
}

/// Deterministic floor under the random sampler: the retired
/// ethereum_format.rs dimensions (num_entries=3 shard padding, all-zeros,
/// all-0xff) are guaranteed to run every invocation, on a 2-CRT context the
/// retired examples never used, through the same oracle as the property.
#[test]
fn pir_round_trip_pinned_corner_cells() {
    for (crt, fill, variant) in [
        (GOOGLE_CRT.to_vec(), Fill::Random, Variant::Inspiring),
        (
            DEFAULT_Q_2CRT_30BIT.to_vec(),
            Fill::Zeros,
            Variant::NoPacking,
        ),
        (vec![DEFAULT_Q], Fill::Ff, Variant::Inspiring),
    ] {
        check_cell(crt, 3, 32, 1, fill, variant, 7).expect("pinned corner cell must round-trip");
    }

    // Exact shard-boundary geometry: check_cell walks num_entries-1, the index
    // a boundary-conditioned routing bug corrupts first.
    check_cell(
        vec![DEFAULT_Q],
        256,
        32,
        100,
        Fill::Random,
        Variant::NoPacking,
        11,
    )
    .expect("full single shard must round-trip");
    check_cell(
        DEFAULT_Q_2CRT_30BIT.to_vec(),
        512,
        8,
        300,
        Fill::Random,
        Variant::Inspiring,
        13,
    )
    .expect("two full shards must round-trip");
}
