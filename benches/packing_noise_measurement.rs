//! Empirical packed-response noise at the two production record widths, gated against the
//! decode boundary.
//!
//! `get_variance` models Spiral-family LWE and gadget noise and does **not** model the noise
//! InspiRING 2-matrix packing adds (root `SECURITY.md`, item G6). The failure mode is silent:
//! once the packed noise crosses `Delta/2` a coefficient decodes to the wrong plaintext and
//! nothing errors. This file is the empirical bound that disclosure needs — it measures
//! `||e_pack||_inf` on the shipped respond path at both production widths and now ASSERTS the
//! margin instead of printing a distribution nobody reads.

#![allow(clippy::expect_used, clippy::print_stderr)]

use std::time::Instant;

use raven_inspire::math::GaussianSampler;
use raven_inspire::{
    respond_seeded_inspiring_cached, setup, ClientSession, InspireParams, ServerInspiringCache,
};

const DEFAULT_SAMPLES: usize = 1_000;

fn percentile(sorted: &[u64], numerator: usize, denominator: usize) -> u64 {
    let index = (sorted.len() - 1) * numerator / denominator;
    sorted[index]
}

fn measure(entry_size: usize, samples: usize, seed: u64) -> Vec<u64> {
    let params = InspireParams::secure_128_d2048();
    let database: Vec<u8> = (0..params.ring_dim * entry_size)
        .map(|offset| u8::try_from((offset * 31 + 17) % 251).expect("reduced byte"))
        .collect();
    let mut setup_sampler = GaussianSampler::with_seed(params.sigma, seed);
    let (crs, encoded, secret_key) =
        setup(&params, &database, entry_size, &mut setup_sampler).expect("setup");
    let cache = ServerInspiringCache::new(&crs, &encoded).expect("server cache");
    let mut session_sampler = GaussianSampler::with_seed(params.sigma, seed ^ 0x5151_0000);
    let session =
        ClientSession::new(crs.clone(), secret_key, &mut session_sampler).expect("session");
    let mut query_sampler = GaussianSampler::with_seed(params.sigma, seed ^ 0xa5a5_0000);
    let ctx = params.ntt_context();
    let delta = params.delta();
    let q = params.q;
    let gamma = raven_inspire::num_columns(entry_size);
    let mut maxima = Vec::with_capacity(samples);

    for sample in 0..samples {
        let target = (sample * 7919) % params.ring_dim;
        let (state, query) = session
            .query_seeded(target as u64, &encoded.config, &mut query_sampler)
            .expect("query");
        let response =
            respond_seeded_inspiring_cached(&crs, &encoded, &query, &cache).expect("respond");
        let noisy_message = &response
            .ciphertext
            .a
            .mul_ntt(&state.rlwe_secret_key.poly, &ctx)
            + &response.ciphertext.b;
        let row = database
            .get(target * entry_size..(target + 1) * entry_size)
            .expect("target row");
        let mut maximum = 0u64;
        for coefficient in 0..gamma {
            let offset = coefficient * 2;
            let pair: [u8; 2] = row
                .get(offset..offset + 2)
                .expect("encoded column pair")
                .try_into()
                .expect("two-byte column");
            let message = u64::from(u16::from_le_bytes(pair));
            let expected = (u128::from(message) * u128::from(delta) % u128::from(q)) as u64;
            let distance = noisy_message.coeff(coefficient).abs_diff(expected);
            maximum = maximum.max(distance.min(q - distance));
        }
        maxima.push(maximum);
    }
    maxima
}

#[test]
#[ignore = "~1,000 production d=2048 InspiRING responses at gamma 16 and 256; run under --release when packing, noise sampling, parameters, or the mod-switch gate changes"]
fn packing_noise_distribution() {
    let samples = std::env::var("RAVEN_PACKING_NOISE_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SAMPLES);
    assert!(samples > 0, "RAVEN_PACKING_NOISE_SAMPLES must be positive");
    for (entry_size, seed) in [(32usize, 0x5010u64), (512usize, 0x5256u64)] {
        let started = Instant::now();
        let mut maxima = measure(entry_size, samples, seed);
        maxima.sort_unstable();
        let sum: u128 = maxima.iter().map(|value| u128::from(*value)).sum();
        let minimum = maxima.first().copied().expect("at least one sample");
        let maximum = maxima.last().copied().expect("at least one sample");

        // THE BOUND. A coefficient decodes correctly iff its noise magnitude stays under
        // `Delta/2 = q/(2p)`; at or past it the value rounds to a neighbouring plaintext and
        // is returned with no error. Assert the worst sample, not the mean — the mean says
        // nothing about the sample that scrambles a row.
        let params = InspireParams::secure_128_d2048();
        let boundary = params.delta() / 2;
        let margin_bits = (boundary as f64 / maximum.max(1) as f64).log2();
        assert!(
            maximum < boundary,
            "packed noise reached {maximum} against a decode boundary of {boundary} \
             (Delta/2 = q/2p) at gamma {}: a response at this width decodes to the WRONG \
             plaintext with no error raised. Widen q before shipping this cell.",
            raven_inspire::num_columns(entry_size)
        );
        eprintln!(
            "{{\"gamma\":{},\"samples\":{},\"min\":{},\"p05\":{},\"p25\":{},\"median\":{},\"p75\":{},\"p95\":{},\"max\":{},\"mean\":{:.3},\"elapsed_seconds\":{:.3}}}",
            raven_inspire::num_columns(entry_size),
            samples,
            minimum,
            percentile(&maxima, 5, 100),
            percentile(&maxima, 25, 100),
            percentile(&maxima, 50, 100),
            percentile(&maxima, 75, 100),
            percentile(&maxima, 95, 100),
            maximum,
            sum as f64 / samples as f64,
            started.elapsed().as_secs_f64(),
        );
        eprintln!(
            "{{\"gamma\":{},\"decode_boundary\":{},\"worst_sample\":{},\"margin_bits\":{:.3}}}",
            raven_inspire::num_columns(entry_size),
            boundary,
            maximum,
            margin_bits,
        );
    }
}
