//! Preconditions that decide whether InsPIRe returns the record or noise:
//! legal InspiRING packing widths, gadget coverage of q, and shard geometry.

#![allow(
    clippy::maybe_infinite_iter,
    reason = "take_while over a doubling sequence terminates at usize overflow"
)]
#![allow(
    clippy::expect_used,
    reason = "test fixture and source-scan failures must abort at their boundary"
)]

use proptest::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use raven_inspire::inspiring::inspiring2::{PackParams, PackParamsError};
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel, ShardConfig, DEFAULT_Q_2CRT_30BIT};
use raven_inspire::pir::{extract_inspiring, query, respond_inspiring, setup, setup_with_rng};

fn params_at(ring_dim: usize) -> InspireParams {
    InspireParams {
        ring_dim,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65536,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

fn powers_of_two_up_to(bound: usize) -> Vec<usize> {
    (0..)
        .map(|k| 1usize << k)
        .take_while(|w| *w <= bound)
        .collect()
}

/// Multiplicative order of `g` in Z*_m, `None` when `gcd(g, m) != 1`.
fn order_mod(g: usize, m: usize) -> Option<usize> {
    let mut value = g % m;
    let mut k = 1usize;
    while value != 1 {
        value = (value * g) % m;
        k += 1;
        if k > m {
            return None;
        }
    }
    Some(k)
}

/// The offline phase sums tau_{g^i} over i in 0..gamma and rescales by 1/gamma.
/// That is the projection onto the subring fixed by <g> only when <g> has order
/// exactly gamma, which for g = 2n/gamma + 1 holds exactly on the powers of two
/// up to n/2.
#[test]
fn legal_packing_widths_are_the_powers_of_two_up_to_half_the_ring() {
    for ring_dim in [256usize, 2048, 4096] {
        let two_n = 2 * ring_dim;
        let algebraically_legal: Vec<usize> = (1..=two_n)
            .filter(|&gamma| {
                let g = (two_n / gamma) + 1;
                two_n % gamma == 0 && order_mod(g, two_n) == Some(gamma)
            })
            .collect();

        assert_eq!(
            algebraically_legal,
            powers_of_two_up_to(ring_dim / 2),
            "ring_dim {ring_dim}"
        );

        // rejections only: building at d=4096 runs an O(n^3) table search
        for gamma in (1..=two_n).filter(|g| !algebraically_legal.contains(g)) {
            assert!(
                PackParams::try_new(&params_at(ring_dim), gamma).is_err(),
                "ring_dim {ring_dim}: width {gamma} has no generator of order {gamma} but was accepted"
            );
        }
    }
}

#[test]
fn try_new_accepts_every_legal_width_and_names_the_generator() {
    let params = params_at(256);
    for gamma in powers_of_two_up_to(128) {
        let pack = PackParams::try_new(&params, gamma).expect("legal width must construct");
        assert_eq!(pack.generator, (512 / gamma) + 1);
        assert_eq!(order_mod(pack.generator, 512), Some(gamma));
        assert_eq!(pack.num_to_pack, gamma);
    }
    assert_eq!(
        PackParams::try_new(&params, 24).err(),
        Some(PackParamsError::IllegalWidth {
            num_to_pack: 24,
            ring_dim: 256
        })
    );
}

/// Full packing covers Z*_{2n} as +/-<5>, so its generator has order n/2 while
/// gamma is n. That width is illegal on the partial path used by `setup`.
#[test]
fn full_packing_width_is_reachable_only_through_try_new_full() {
    let params = params_at(256);
    assert!(PackParams::try_new(&params, 256).is_err());
    let full = PackParams::try_new_full(&params).expect("full packing must construct");
    assert_eq!(full.num_to_pack, 256);
    assert_eq!(full.generator, 5);
    assert_eq!(order_mod(5, 512), Some(128));
}

/// `setup` derives the packing width as ceil(entry_size / 2). Every entry size
/// must therefore either round-trip its record or fail loudly - never return
/// a different record.
#[test]
fn no_entry_size_silently_decodes_to_the_wrong_record() {
    let params = params_at(256);
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let mut silently_wrong = Vec::new();
    let mut round_tripped = Vec::new();
    for entry_size in 1..=40usize {
        let outcome = std::panic::catch_unwind(|| {
            let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
            let database: Vec<u8> = (0..params.ring_dim * entry_size)
                .map(|i| ((i * 37 + 11) % 251) as u8)
                .collect();
            let (crs, encoded_db, sk) =
                setup(&params, &database, entry_size, &mut sampler).expect("setup");
            let (state, client_query) =
                query(&crs, 7, &encoded_db.config, &sk, &mut sampler).expect("query");
            let response = respond_inspiring(&crs, &encoded_db, &client_query).expect("respond");
            let extracted =
                extract_inspiring(&crs, &state, &response, entry_size).expect("extract");
            extracted.as_slice() == &database[7 * entry_size..8 * entry_size]
        });
        match outcome {
            Ok(true) => round_tripped.push(entry_size),
            Ok(false) => silently_wrong.push(entry_size),
            Err(_) => {}
        }
    }
    std::panic::set_hook(default_hook);

    assert!(
        silently_wrong.is_empty(),
        "entry sizes returning a wrong record with no error: {silently_wrong:?}"
    );
    assert!(
        round_tripped.contains(&32),
        "32-byte records must still round-trip; round-tripped: {round_tripped:?}"
    );
}

/// `gadget_decompose` emits exactly `gadget_len` digits, so a gadget narrower
/// than q reconstructs `value mod gadget_base^gadget_len`.
#[test]
fn validate_rejects_a_gadget_narrower_than_q() {
    let q: u64 = DEFAULT_Q_2CRT_30BIT.iter().product();
    let narrow = InspireParams {
        ring_dim: 2048,
        q,
        crt_moduli: DEFAULT_Q_2CRT_30BIT.to_vec(),
        p: 65537,
        sigma: 6.4,
        gadget_base: 1 << 19,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    };
    assert!(
        (1u128 << 19).pow(3) < q as u128,
        "fixture must actually be narrower than q"
    );
    assert!(narrow.validate().is_err());

    let covering = InspireParams {
        gadget_len: 4,
        ..narrow
    };
    assert!(covering.validate().is_ok());
}

#[test]
fn for_scenario_with_crt_widens_the_gadget_to_cover_the_override() {
    let params = InspireParams::for_scenario_with_crt(
        1 << 20,
        256,
        [64, 1024, 64],
        1,
        DEFAULT_Q_2CRT_30BIT.to_vec(),
    )
    .expect("override must derive");

    assert_eq!(params.gadget_base, 1u64 << 19);
    assert_eq!(params.gadget_len, 4);
    assert!((params.gadget_base as u128).pow(params.gadget_len as u32) >= params.q as u128);
}

/// Causality for the gadget invariant: the recommended override decodes every
/// byte only once the gadget spans q.
#[test]
fn for_scenario_with_crt_round_trips_records() {
    let params = InspireParams::for_scenario_with_crt(
        1 << 20,
        256,
        [64, 1024, 64],
        1,
        DEFAULT_Q_2CRT_30BIT.to_vec(),
    )
    .expect("override must derive");

    let entry_size = 32usize;
    let entries = 64usize;
    let database: Vec<u8> = (0..entries * entry_size)
        .map(|i| ((i * 37 + 11) % 251) as u8)
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let (crs, encoded_db, sk) = setup(&params, &database, entry_size, &mut sampler).expect("setup");
    let (state, client_query) =
        query(&crs, 7, &encoded_db.config, &sk, &mut sampler).expect("query");
    let response = respond_inspiring(&crs, &encoded_db, &client_query).expect("respond");
    let extracted = extract_inspiring(&crs, &state, &response, entry_size).expect("extract");

    let expected = &database[7 * entry_size..8 * entry_size];
    let wrong = extracted
        .iter()
        .zip(expected)
        .filter(|(got, want)| got != want)
        .count();
    assert_eq!(wrong, 0, "wrong bytes {wrong}/{entry_size}");
}

#[test]
fn flat_db_shard_geometry_matches_the_ring_dimension() {
    let params = InspireParams::default();
    let config = ShardConfig::for_flat_db(32, 2_417_514_276);

    assert_eq!(config.entries_per_shard(), params.ring_dim as u64);
    assert_eq!(
        config.shard_size_bytes,
        params.ring_dim as u64 * 32,
        "a shard holds one entry per ring coefficient"
    );
    config
        .validate_for_params(&params)
        .expect("for_flat_db must produce an encodable geometry");
}

#[test]
fn for_ring_dim_rejects_geometry_the_encoder_cannot_serve() {
    let params = params_at(256);
    let config = ShardConfig::for_ring_dim(params.ring_dim, 32, 100_000).expect("valid geometry");
    assert!(config.validate_for_params(&params).is_ok());

    let oversized = ShardConfig {
        shard_size_bytes: 1 << 30,
        entry_size_bytes: 32,
        total_entries: 100_000,
    };
    assert!(oversized.validate_for_params(&params).is_err());

    assert!(ShardConfig::for_ring_dim(0, 32, 10).is_err());
    assert!(ShardConfig::for_ring_dim(2048, 0, 10).is_err());
    assert!(ShardConfig::for_ring_dim(usize::MAX, usize::MAX, 10).is_err());
}

#[test]
fn degenerate_shard_config_reports_instead_of_dividing_by_zero() {
    let zero_entry = ShardConfig::for_flat_db(0, 100);
    assert_eq!(zero_entry.entries_per_shard(), 0);
    assert_eq!(zero_entry.num_shards(), 0);
    assert!(zero_entry.validate().is_err());
    assert!(zero_entry.try_index_to_shard(0).is_err());

    let overflowing = ShardConfig {
        shard_size_bytes: 32,
        entry_size_bytes: 32,
        total_entries: u64::MAX,
    };
    assert!(overflowing.validate().is_err());
    assert!(overflowing.try_index_to_shard(u32::MAX as u64 + 1).is_err());
}

fn assert_shard_mapping_matches_native(global_idx: u64, entries_per_shard: u64) {
    let config = ShardConfig {
        shard_size_bytes: entries_per_shard,
        entry_size_bytes: 1,
        total_entries: u64::MAX,
    };
    let expected_shard = global_idx / entries_per_shard;
    let mapped = config.try_index_to_shard(global_idx);
    if expected_shard > u64::from(u32::MAX) {
        assert!(mapped.is_err());
    } else {
        assert_eq!(
            mapped,
            Ok((expected_shard as u32, global_idx % entries_per_shard))
        );
    }
}

#[test]
fn shard_mapping_matches_native_at_divisor_boundaries() {
    for entries_per_shard in [1, 2, 3, 7, 256, 2048, u32::MAX as u64, u64::MAX] {
        for global_idx in [
            0,
            entries_per_shard.saturating_sub(1),
            entries_per_shard,
            entries_per_shard.saturating_add(1),
            u64::MAX,
        ] {
            assert_shard_mapping_matches_native(global_idx, entries_per_shard);
        }
    }
}

#[test]
fn security_level_is_an_advisory_serialized_label() {
    let bits128 = InspireParams::secure_128_d2048();
    let mut bits256 = bits128.clone();
    bits256.security_level = SecurityLevel::Bits256;

    assert!(bits128.validate().is_ok());
    assert!(bits256.validate().is_ok());
    assert_ne!(
        bincode::serialize(&bits128).expect("serialize 128-bit declaration"),
        bincode::serialize(&bits256).expect("serialize 256-bit declaration")
    );
}

fn rust_sources_under(directory: &std::path::Path, sources: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(directory).expect("read source directory") {
        let entry = entry.expect("source entry");
        let source_path = entry.path();
        if source_path.is_dir() {
            rust_sources_under(&source_path, sources);
        } else if source_path
            .extension()
            .is_some_and(|extension| extension == "rs")
        {
            sources.push(source_path);
        }
    }
}

#[test]
fn security_level_has_no_production_read() {
    let mut sources = Vec::new();
    rust_sources_under(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut sources,
    );
    assert!(!sources.is_empty(), "source scan matched nothing");
    for source_path in sources {
        let source = std::fs::read_to_string(&source_path).expect("read Rust source");
        assert!(
            !source.contains(".security_level"),
            "SecurityLevel became behavior at {}",
            source_path.display()
        );
    }
}

#[test]
fn security_level_does_not_control_pir_correctness() {
    let run = |security_level| {
        let mut params = params_at(256);
        params.security_level = security_level;
        let database: Vec<u8> = (0..params.ring_dim * 32)
            .map(|index| u8::try_from(index % 251).expect("bounded byte"))
            .collect();
        let mut setup_sampler = GaussianSampler::with_seed(params.sigma, 41);
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let (crs, encoded, secret_key) =
            setup_with_rng(&params, &database, 32, &mut setup_sampler, &mut rng)
                .expect("advisory setup");
        let mut query_sampler = GaussianSampler::with_seed(params.sigma, 43);
        let (state, client_query) =
            query(&crs, 7, &encoded.config, &secret_key, &mut query_sampler)
                .expect("advisory query");
        let response = respond_inspiring(&crs, &encoded, &client_query).expect("advisory respond");
        extract_inspiring(&crs, &state, &response, 32).expect("advisory extract")
    };

    let bits128 = run(SecurityLevel::Bits128);
    let bits256 = run(SecurityLevel::Bits256);
    assert_eq!(bits128, bits256);
}

proptest! {
    #[test]
    fn shard_mapping_matches_native_division(
        global_idx in any::<u64>(),
        entries_per_shard in 1u64..=u64::MAX,
    ) {
        assert_shard_mapping_matches_native(global_idx, entries_per_shard);
    }
}
