//! The Solinas NTT fast path is dispatched at runtime (its butterflies are
//! deliberately scalar - see ntt.rs's measurement note); the round-trip and
//! the convolution against a naive u128 negacyclic oracle gate it.

#![allow(
    clippy::needless_range_loop,
    reason = "lane-parallel KAT loops index two buffers in lockstep"
)]

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use raven_inspire::math::mod_q::DEFAULT_Q;
use raven_inspire::math::NttContext;

const ITERS: usize = 10;

fn random_coeffs(seed: u64, n: usize, q: u64) -> Vec<u64> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| rand::RngCore::next_u64(&mut rng) % q)
        .collect()
}

#[test]
fn solinas_ntt_roundtrip_byte_identity() {
    for &n in &[256usize, 512, 1024, 2048] {
        let ctx = NttContext::with_default_q(n);
        for iter in 0..ITERS {
            let original = random_coeffs((n as u64) * 100 + iter as u64, n, DEFAULT_Q);

            let mut coeffs = original.clone();
            ctx.forward(&mut coeffs);
            ctx.inverse(&mut coeffs);

            assert_eq!(
                coeffs, original,
                "NTT round-trip failed at n={n}, iter={iter}"
            );
        }
    }
}

/// NTT convolution must match naive negacyclic convolution; a lane-swap or
/// off-by-one in the SIMD butterfly diverges here.
#[test]
fn solinas_forward_convolution_matches_naive_mul() {
    let n = 256;
    let ctx = NttContext::with_default_q(n);
    let q = DEFAULT_Q;

    let mut rng = ChaCha20Rng::seed_from_u64(0xC001CAFE);
    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    for i in 0..32 {
        a[i] = rand::RngCore::next_u64(&mut rng) % 1000;
        b[i] = rand::RngCore::next_u64(&mut rng) % 1000;
    }

    let mut c_naive = vec![0u64; n];
    for i in 0..n {
        for j in 0..n {
            if a[i] == 0 || b[j] == 0 {
                continue;
            }
            let prod = (a[i] as u128 * b[j] as u128) % q as u128;
            let idx = i + j;
            if idx < n {
                c_naive[idx] = ((c_naive[idx] as u128 + prod) % q as u128) as u64;
            } else {
                let wrap = idx - n;
                // X^n = -1
                let q128 = q as u128;
                c_naive[wrap] = ((c_naive[wrap] as u128 + q128 - prod) % q128) as u64;
            }
        }
    }

    let mut a_ntt = a.clone();
    let mut b_ntt = b.clone();
    ctx.forward(&mut a_ntt);
    ctx.forward(&mut b_ntt);
    let mut product = vec![0u64; n];
    ctx.pointwise_mul(&a_ntt, &b_ntt, &mut product);
    ctx.inverse(&mut product);

    assert_eq!(
        product, c_naive,
        "NTT convolution does not match naive convolution"
    );
}

// A shipping-cell (n=2048) round-trip stress test lived here; it was a strict
// subset of solinas_ntt_roundtrip_byte_identity above (same function, n list
// already includes 2048, fewer seeds) and was retired 2026-09-06 - the
// unscaled-coeff-0 inverse mutant killed both.
