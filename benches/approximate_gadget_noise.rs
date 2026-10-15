//! Served noise under the approximate packing gadget, against the exact gadget it
//! replaces, on the production ring at gamma 128 and 256.
//!
//! The approximate gadget trades the lowest digit's key-error term for a rounding
//! residual times the rotated secret. Whether that trade keeps the decode margin is an
//! empirical question, and a lost margin is silent: a coefficient past
//! `floor(q'/p) / 2` decodes to a neighbouring plaintext with no error. This file
//! serves the same queries under both gadgets and asserts the approximate one's worst
//! sample against the exact one's.
//!
//! Paired: both parameter sets encode one table under one CRS seed; every session
//! draws a fresh secret that both gadgets use, and every query's bytes go to both.
//! Both keys draw their rows' errors from one stream. A key row's error leaves the
//! same offset on every response of its session, whichever digit the row serves, so
//! the two gadgets share those offsets except for the one row the approximate key
//! does not have. Only that row and the server's decomposition differ.
//!
//! `cargo test` leaves bench targets alone and the test is ignored, so this file
//! runs only when named. The option is off in every preset; a preset or served table
//! that turns it on needs a CI lane running this bench first.

#![allow(clippy::expect_used, clippy::print_stderr)]

#[cfg(feature = "mod-switch-response")]
mod served {
    use std::time::Instant;

    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use raven_inspire::math::{GaussianSampler, NttContext, Poly};
    use raven_inspire::pir::mod_switch::{
        extract_inspiring_mod_switched, mod_switch_response_checked, MOD_SWITCH_TARGET_36BIT,
    };
    use raven_inspire::pir::{
        extract_inspiring, respond_seeded_inspiring_cached_with_session, setup_with_rng,
        ClientSession, EncodedDatabase, PackingMode, SeededClientQuery, ServerCrs,
        ServerInspiringCache, ServerResponse, ServerSessionStore,
    };
    use raven_inspire::rlwe::RlweSecretKey;
    use raven_inspire::InspireParams;

    /// Fresh secrets per (gamma, gadget) cell.
    const DEFAULT_SECRETS: usize = 50;
    /// Responses per secret; with the default secrets, 1,200 responses per cell.
    const DEFAULT_PER_SECRET: usize = 24;

    /// Fewest secrets at which the two spreads are compared.
    const MIN_SECRETS_FOR_SPREAD: usize = 40;

    /// How far below the exact gadget's worst-sample margin the approximate gadget's
    /// may land. The worst of ~1,200 responses is one draw from a tail, and a run is
    /// not a function of its seed (sessions draw OS entropy): over 16 paired cells the
    /// approximate margin less the exact one had mean +0.12 and standard deviation
    /// 0.18 bits, and the exact gadget's own margin ranged over 0.65 bits. Four
    /// deviations below the mean; the spread assertion is the sensitive one.
    const WORST_SAMPLE_ALLOWANCE_BITS: f64 = 0.6;

    fn env_or(name: &str, default: u64) -> u64 {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    /// One parameter set's server and the client's view of its CRS.
    struct Server {
        params: InspireParams,
        crs: ServerCrs,
        client_crs: ServerCrs,
        encoded: EncodedDatabase,
        cache: ServerInspiringCache,
        store: ServerSessionStore,
    }

    impl Server {
        fn build(params: &InspireParams, database: &[u8], entry_size: usize, seed: u64) -> Self {
            let mut sampler = GaussianSampler::with_seed(params.sigma, seed);
            let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0x5eed);
            let (mut crs, mut encoded, _) =
                setup_with_rng(params, database, entry_size, &mut sampler, &mut rng)
                    .expect("setup");
            let cache = ServerInspiringCache::from_setup(&mut crs, &encoded).expect("cache");
            let ctx = params.ntt_context();
            for shard in &mut encoded.shards {
                shard
                    .precompute_packing(cache.pack_params(), &ctx)
                    .expect("shard store");
            }
            let client_crs =
                ServerCrs::from_versioned_bytes(&crs.to_versioned_bytes().expect("crs bytes"))
                    .expect("wire crs");
            Self {
                params: params.clone(),
                crs,
                client_crs,
                encoded,
                cache,
                store: ServerSessionStore::new(),
            }
        }
    }

    /// Error statistics of one (gamma, gadget) cell, before and after the served switch.
    #[derive(Default)]
    struct Cell {
        responses: usize,
        coefficients: f64,
        pre_worst: u64,
        pre_sum: f64,
        pre_sum_squares: f64,
        post_worst: u64,
        post_sum: f64,
        post_sum_squares: f64,
        wrong_rows_pre: usize,
        wrong_rows_served: usize,
        worst_per_secret: Vec<u64>,
    }

    impl Cell {
        fn std(sum: f64, sum_squares: f64, count: f64) -> f64 {
            let mean = sum / count;
            (sum_squares / count - mean * mean).sqrt()
        }

        fn pre_std(&self) -> f64 {
            Self::std(self.pre_sum, self.pre_sum_squares, self.coefficients)
        }

        fn post_std(&self) -> f64 {
            Self::std(self.post_sum, self.post_sum_squares, self.coefficients)
        }
    }

    fn margin_bits(boundary: u64, worst: u64) -> f64 {
        (boundary as f64 / worst.max(1) as f64).log2()
    }

    fn centred(value: u64, expected: u64, modulus: u64) -> i64 {
        let distance = (value + modulus - expected) % modulus;
        if distance > modulus / 2 {
            -((modulus - distance) as i64)
        } else {
            distance as i64
        }
    }

    /// The secret is small: only its sign wrap moves to the served modulus.
    fn secret_at(secret: &Poly, q: u64, target: u64) -> Poly {
        let coeffs = (0..secret.dimension())
            .map(|i| {
                let c = secret.coeff(i);
                if c > q / 2 {
                    target - (q - c)
                } else {
                    c
                }
            })
            .collect();
        Poly::from_coeffs(coeffs, target)
    }

    fn measure(entry_size: usize, secrets: usize, per_secret: usize, seed: u64) -> [Cell; 2] {
        let exact = InspireParams::secure_128_d2048();
        let approximate = InspireParams {
            approximate_packing_gadget: true,
            ..exact.clone()
        };
        let gamma = raven_inspire::num_columns(entry_size);
        let (d, q, p) = (exact.ring_dim, exact.q, exact.p);
        let target = MOD_SWITCH_TARGET_36BIT;

        // Uniform bytes, and one row of the largest column value.
        let mut database = vec![0u8; d * entry_size];
        ChaCha20Rng::seed_from_u64(seed ^ 0xdb).fill_bytes(&mut database);
        database[..entry_size].fill(0xff);

        let servers = [
            Server::build(&exact, &database, entry_size, seed),
            Server::build(&approximate, &database, entry_size, seed),
        ];
        assert_eq!(
            servers[0].crs.inspiring_w_seed,
            servers[1].crs.inspiring_w_seed
        );
        assert_eq!(servers[0].cache.pack_params().key_rows(), 3);
        assert_eq!(servers[1].cache.pack_params().key_rows(), 2);

        let ctx = exact.ntt_context();
        let target_ctx = NttContext::with_moduli(d, &[target]);
        let (delta, served_delta) = (exact.delta(), target / p);
        let (pre_boundary, served_boundary) = (delta / 2, served_delta / 2);
        let mut query_sampler = GaussianSampler::with_seed(exact.sigma, seed ^ 0xa5a5_0000);
        let mut cells = [Cell::default(), Cell::default()];

        for secret_index in 0..secrets {
            let secret_seed = seed ^ 0x5151_0000 ^ ((secret_index as u64) << 20);
            let mut secret_sampler = GaussianSampler::with_seed(exact.sigma, secret_seed ^ 0x77);
            let secret_key = RlweSecretKey::generate(&exact, &mut secret_sampler);
            let served_secret = secret_at(&secret_key.poly, q, target);

            let mut handles = Vec::with_capacity(2);
            let mut sessions = Vec::with_capacity(2);
            for server in &servers {
                let mut key_sampler = GaussianSampler::with_seed(exact.sigma, secret_seed);
                let session = ClientSession::new(
                    server.client_crs.clone(),
                    secret_key.clone(),
                    &mut key_sampler,
                )
                .expect("session");
                let keys = session.packing_keys().expect("packing keys").clone();
                handles.push(
                    server
                        .store
                        .register_server_side(keys, server.cache.pack_params(), &ctx)
                        .expect("register"),
                );
                sessions.push(session);
            }

            let mut worst = [0u64; 2];
            for sample in 0..per_secret {
                // Row 0 holds the largest column value; every secret reads it once.
                let index = if sample == 0 {
                    0
                } else {
                    ((secret_index * per_secret + sample) * 7919) % d
                };
                let (state, query) = sessions[0]
                    .query_seeded(index as u64, &servers[0].encoded.config, &mut query_sampler)
                    .expect("query");
                let row = &database[index * entry_size..(index + 1) * entry_size];
                let messages: Vec<u64> = row
                    .chunks(2)
                    .map(|pair| u64::from(u16::from_le_bytes([pair[0], pair[1]])))
                    .collect();
                assert_eq!(messages.len(), gamma);

                for (variant, server) in servers.iter().enumerate() {
                    let by_handle = SeededClientQuery {
                        shard_id: query.shard_id,
                        rgsw_ciphertext: query.rgsw_ciphertext.clone(),
                        packing_mode: PackingMode::Inspiring,
                        inspiring_packing_keys: None,
                        session_handle: Some(handles[variant]),
                    };
                    let response = respond_seeded_inspiring_cached_with_session(
                        &server.crs,
                        &server.encoded,
                        &by_handle,
                        &server.cache,
                        Some(&server.store),
                    )
                    .expect("respond");
                    let cell = &mut cells[variant];
                    cell.responses += 1;
                    cell.coefficients += gamma as f64;

                    let noisy = &response.ciphertext.a.mul_ntt(&secret_key.poly, &ctx)
                        + &response.ciphertext.b;
                    for (c, &message) in messages.iter().enumerate() {
                        let expected =
                            (u128::from(message) * u128::from(delta) % u128::from(q)) as u64;
                        let error = centred(noisy.coeff(c), expected, q);
                        cell.pre_worst = cell.pre_worst.max(error.unsigned_abs());
                        cell.pre_sum += error as f64;
                        cell.pre_sum_squares += (error as f64) * (error as f64);
                    }
                    let decoded =
                        extract_inspiring(&server.client_crs, &state, &response, entry_size)
                            .expect("extract");
                    cell.wrong_rows_pre += usize::from(decoded != row);

                    // What the client downloads: the checked switch, then the serializer.
                    let switched = mod_switch_response_checked(&server.params, &response, target)
                        .expect("checked switch");
                    let served =
                        ServerResponse::from_binary(&switched.to_binary().expect("served encode"))
                            .expect("served decode");
                    let noisy = &served.ciphertext.a.mul_ntt(&served_secret, &target_ctx)
                        + &served.ciphertext.b;
                    for (c, &message) in messages.iter().enumerate() {
                        let expected = (u128::from(message) * u128::from(served_delta)
                            % u128::from(target)) as u64;
                        let error = centred(noisy.coeff(c), expected, target);
                        cell.post_worst = cell.post_worst.max(error.unsigned_abs());
                        worst[variant] = worst[variant].max(error.unsigned_abs());
                        cell.post_sum += error as f64;
                        cell.post_sum_squares += (error as f64) * (error as f64);
                    }
                    let decoded = extract_inspiring_mod_switched(
                        &server.client_crs,
                        &state,
                        &served,
                        entry_size,
                    )
                    .expect("served extract");
                    cell.wrong_rows_served += usize::from(decoded != row);
                }
            }
            for (variant, server) in servers.iter().enumerate() {
                cells[variant].worst_per_secret.push(worst[variant]);
                server.store.remove(handles[variant]).expect("release");
            }
        }

        for (name, cell) in ["exact", "approximate"].iter().zip(&cells) {
            let mut per_secret = cell.worst_per_secret.clone();
            per_secret.sort_unstable();
            eprintln!(
                "{{\"gamma\":{gamma},\"gadget\":\"{name}\",\"responses\":{},\"secrets\":{secrets},\
                 \"pre_worst\":{},\"pre_std\":{:.1},\"pre_std_log2\":{:.3},\
                 \"pre_boundary\":{pre_boundary},\"pre_margin_bits\":{:.3},\
                 \"served_modulus\":{target},\"served_worst\":{},\"served_std\":{:.2},\
                 \"served_boundary\":{served_boundary},\"served_margin_bits\":{:.3},\
                 \"per_secret_worst_min\":{},\"per_secret_worst_median\":{},\
                 \"wrong_rows_pre\":{},\"wrong_rows_served\":{}}}",
                cell.responses,
                cell.pre_worst,
                cell.pre_std(),
                cell.pre_std().log2(),
                margin_bits(pre_boundary, cell.pre_worst),
                cell.post_worst,
                cell.post_std(),
                margin_bits(served_boundary, cell.post_worst),
                per_secret[0],
                per_secret[per_secret.len() / 2],
                cell.wrong_rows_pre,
                cell.wrong_rows_served,
            );
        }
        cells
    }

    #[test]
    #[ignore = "2 x 1,200 production d=2048 responses at gamma 128 and at gamma 256, over 50 fresh secrets, each switched to the served modulus and decoded; run under --release with --features mod-switch-response,parallel when the packing gadget, the packing recursion, noise sampling or the parameters change"]
    fn approximate_gadget_keeps_the_exact_gadgets_margin() {
        let secrets = env_or("RAVEN_APPROXIMATE_NOISE_SECRETS", DEFAULT_SECRETS as u64) as usize;
        let per_secret = env_or(
            "RAVEN_APPROXIMATE_NOISE_PER_SECRET",
            DEFAULT_PER_SECRET as u64,
        ) as usize;
        let seed = env_or("RAVEN_APPROXIMATE_NOISE_SEED", 0x0a99_0000);
        assert!(secrets > 0 && per_secret > 0);
        let params = InspireParams::secure_128_d2048();
        let boundary = MOD_SWITCH_TARGET_36BIT / params.p / 2;

        for entry_size in [256usize, 512] {
            let gamma = raven_inspire::num_columns(entry_size);
            let started = Instant::now();
            let [exact, approximate] =
                measure(entry_size, secrets, per_secret, seed ^ entry_size as u64);
            let exact_margin = margin_bits(boundary, exact.post_worst);
            let approximate_margin = margin_bits(boundary, approximate.post_worst);
            eprintln!(
                "{{\"gamma\":{gamma},\"seed\":{seed},\"exact_margin_bits\":{exact_margin:.3},\
                 \"approximate_margin_bits\":{approximate_margin:.3},\"margin_gain_bits\":{:.3},\
                 \"served_std_ratio\":{:.4},\"pre_std_ratio\":{:.4},\"elapsed_seconds\":{:.1}}}",
                approximate_margin - exact_margin,
                approximate.post_std() / exact.post_std(),
                approximate.pre_std() / exact.pre_std(),
                started.elapsed().as_secs_f64(),
            );

            for (name, cell) in [("exact", &exact), ("approximate", &approximate)] {
                assert_eq!(
                    (cell.wrong_rows_pre, cell.wrong_rows_served),
                    (0, 0),
                    "gamma {gamma}, {name} gadget: rows decoded to the wrong bytes"
                );
                assert!(
                    cell.post_worst < boundary,
                    "gamma {gamma}, {name} gadget: served error {} reached the decode \
                     boundary {boundary}",
                    cell.post_worst
                );
            }
            // The trade removes one of three digit-times-key-error terms and adds a
            // quarter of one, so the spread must not grow. Most of a row's term is one
            // offset per session, so a spread over a handful of secrets is a handful of
            // draws and is only printed.
            assert!(
                secrets < MIN_SECRETS_FOR_SPREAD
                    || (approximate.pre_std() <= exact.pre_std()
                        && approximate.post_std() <= exact.post_std()),
                "gamma {gamma}: the approximate gadget is noisier than the exact one \
                 (pre-switch std {:.1} vs {:.1}, served {:.2} vs {:.2})",
                approximate.pre_std(),
                exact.pre_std(),
                approximate.post_std(),
                exact.post_std()
            );
            assert!(
                approximate_margin >= exact_margin - WORST_SAMPLE_ALLOWANCE_BITS,
                "gamma {gamma}: the approximate gadget's worst served sample leaves \
                 {approximate_margin:.3} bits against the exact gadget's {exact_margin:.3}; \
                 a set with this gadget would decode wrong where the exact one still holds"
            );
        }
    }
}
