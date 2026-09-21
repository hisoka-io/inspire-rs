//! The NTT root search is sound only on a prime. On a composite it either never ends or
//! accepts a root that is not one, so every public constructor of a context fails fast and
//! by name.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::mpsc;
use std::time::Duration;

use raven_inspire::math::mod_q::DEFAULT_Q;
use raven_inspire::math::{NttContext, Poly};
use raven_inspire::params::{DEFAULT_CRT_MODULI, DEFAULT_Q_2CRT_30BIT};
use raven_inspire::rlwe::apply_automorphism;
use raven_inspire::{InspireParams, SecurityLevel};

const RING_DIM: usize = 256;
/// `162,739 * 422,267`, `== 1 mod 512`: no factor a trial division by small primes finds.
const SEMIPRIME_36BIT: u64 = 68_719_309_313;
/// `7681 * 15361 * 23041`, a Carmichael number `== 1 mod 512`: `g^(q-1) = 1` for every unit,
/// so the root search accepts a candidate at once. It is a square root of 1 other than -1, and
/// the transform built on it round-trips while multiplying wrongly.
const CARMICHAEL: u64 = 2_718_557_844_481;
/// `3 * 5 * ...`, `== 1 mod 4096`: composite, and NTT-friendly at the production ring.
const COMPOSITE_AT_2048: u64 = 68_718_424_065;
/// Far longer than a refusal needs, far shorter than a `2..q` scan.
const REFUSAL_BOUND: Duration = Duration::from_secs(10);

fn params_with(ring_dim: usize, crt_moduli: Vec<u64>) -> InspireParams {
    InspireParams {
        ring_dim,
        q: crt_moduli.iter().product(),
        crt_moduli,
        p: 65_537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

/// Runs `build` off-thread so a scan that never ends fails this test instead of the run.
/// `Some(message)` is the panic it refused with; `None` means it built a context.
fn refusal_within_bound(
    entry_point: &'static str,
    build: impl FnOnce() + Send + 'static,
) -> Option<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(build));
        let _ = tx.send(outcome.err().map(|payload| {
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(ToString::to_string))
                .unwrap_or_default()
        }));
    });
    rx.recv_timeout(REFUSAL_BOUND)
        .unwrap_or_else(|_| panic!("{entry_point} did not return within {REFUSAL_BOUND:?}"))
}

#[test]
fn validate_refuses_a_composite_limb() {
    for (ring_dim, limbs) in [
        (RING_DIM, vec![SEMIPRIME_36BIT]),
        (2048, vec![COMPOSITE_AT_2048]),
        (RING_DIM, vec![CARMICHAEL]),
        (2048, vec![DEFAULT_Q_2CRT_30BIT[0], 1_073_725_441]),
    ] {
        let params = params_with(ring_dim, limbs.clone());
        for &limb in &limbs {
            assert_eq!(
                limb % (2 * ring_dim as u64),
                1,
                "{limb} must reach the prime check"
            );
        }
        let refusal = params
            .validate()
            .expect_err("a composite limb must not validate");
        assert!(refusal.contains("prime"), "{limbs:?}: {refusal}");
    }
}

#[test]
fn every_shipped_modulus_still_validates() {
    let shipped = [
        InspireParams::secure_128_d2048(),
        InspireParams::secure_128_d4096(),
        InspireParams::default(),
        params_with(2048, DEFAULT_CRT_MODULI.to_vec()),
        params_with(2048, DEFAULT_Q_2CRT_30BIT.to_vec()),
        InspireParams::for_scenario(1 << 20, 256, [64, 1024, 64], 1).expect("derived pair"),
        InspireParams::for_scenario_with_crt(
            1 << 20,
            256,
            [64, 1024, 64],
            1,
            DEFAULT_Q_2CRT_30BIT.to_vec(),
        )
        .expect("30-bit pair"),
    ];
    for params in shipped {
        params
            .validate()
            .unwrap_or_else(|refusal| panic!("{:?}: {refusal}", params.crt_moduli));
        assert_eq!(params.ntt_context().moduli(), params.moduli());
    }
}

#[test]
fn no_public_constructor_scans_a_composite() {
    type Build = Box<dyn FnOnce() + Send>;
    for composite in [SEMIPRIME_36BIT, CARMICHAEL] {
        let entry_points: [(&'static str, Build); 6] = [
            (
                "NttContext::new",
                Box::new(move || drop(NttContext::new(RING_DIM, composite))),
            ),
            (
                "NttContext::with_moduli",
                Box::new(move || drop(NttContext::with_moduli(RING_DIM, &[DEFAULT_Q, composite]))),
            ),
            (
                "InspireParams::ntt_context",
                Box::new(move || drop(params_with(RING_DIM, vec![composite]).ntt_context())),
            ),
            (
                "Poly::mul",
                Box::new(move || {
                    let poly = Poly::from_coeffs(vec![1; RING_DIM], composite);
                    drop(poly.mul(&poly));
                }),
            ),
            (
                "Poly * Poly",
                Box::new(move || {
                    let mut poly = Poly::from_coeffs(vec![1; RING_DIM], composite);
                    poly.force_ntt_domain();
                    drop(poly.clone() * poly);
                }),
            ),
            (
                "apply_automorphism",
                Box::new(move || {
                    let mut poly = Poly::from_coeffs(vec![1; RING_DIM], composite);
                    poly.force_ntt_domain();
                    drop(apply_automorphism(&poly, 3));
                }),
            ),
        ];
        for (entry_point, build) in entry_points {
            let refusal = refusal_within_bound(entry_point, build)
                .unwrap_or_else(|| panic!("{entry_point} built a context on {composite}"));
            assert!(refusal.contains("prime"), "{entry_point}: {refusal}");
            assert!(
                refusal.contains(&composite.to_string()),
                "{entry_point}: {refusal}"
            );
        }
    }
}

#[test]
fn a_prime_modulus_still_builds_at_every_supported_dimension() {
    for ring_dim in [2usize, 256, 2048, 4096, 8192] {
        let ctx = NttContext::with_default_q(ring_dim);
        let mut coeffs: Vec<u64> = (0..ring_dim as u64).collect();
        ctx.forward(&mut coeffs);
        ctx.inverse(&mut coeffs);
        assert_eq!(coeffs, (0..ring_dim as u64).collect::<Vec<_>>());
    }
}
