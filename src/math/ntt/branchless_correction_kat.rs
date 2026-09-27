//! Old-vs-new differential for the branch-free corrections in the NTT.
//!
//! Every `old_*` function below is the branching spelling the NTT shipped
//! before its corrections became branch-free, kept verbatim as the oracle.
//! Each rewritten function must return the same bytes at the shipped modulus
//! and at both two-CRT moduli sets, over random inputs and the edges where a
//! correction flips: 0, 1, q-2, q-1 for operands, and q-1, q, q+1, 2q-1 for
//! the unreduced values a correction sees.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use super::NttContext;
use crate::math::mod_q::DEFAULT_Q;
use crate::math::modular::{csubq, sub_mod_branchless, sub_mod_mask};
use crate::math::solinas_redc::{pointwise_solinas_mont_mul, solinas_mont_mul_default_q};
use crate::params::{DEFAULT_CRT_MODULI, DEFAULT_Q_2CRT_30BIT};

const MODULI_SETS: [&[u64]; 3] = [&[DEFAULT_Q], &DEFAULT_CRT_MODULI, &DEFAULT_Q_2CRT_30BIT];
const DIMS: [usize; 2] = [16, 2048];
const RANDOM_SCALARS: usize = 200_000;

fn old_mont_mul(a: u64, b: u64, q: u64, q_inv_neg: u64) -> u64 {
    let ab = (a as u128) * (b as u128);
    let m = ((ab as u64).wrapping_mul(q_inv_neg)) as u128;
    let t = ((ab + m * (q as u128)) >> 64) as u64;
    if t >= q {
        t - q
    } else {
        t
    }
}

/// The unreduced REDC value, to show the random draws reach both arms.
fn redc_t(a: u64, b: u64, q: u64, q_inv_neg: u64) -> u64 {
    let ab = (a as u128) * (b as u128);
    let m = ((ab as u64).wrapping_mul(q_inv_neg)) as u128;
    ((ab + m * (q as u128)) >> 64) as u64
}

fn old_solinas_mul(a: u64, b: u64, q_inv_neg: u64) -> u64 {
    const Q: u64 = DEFAULT_Q;
    let ab = (a as u128).wrapping_mul(b as u128);
    let m = (ab as u64).wrapping_mul(q_inv_neg);
    let mq = ((m as u128) << 60)
        .wrapping_sub((m as u128) << 14)
        .wrapping_add(m as u128);
    let t = (ab.wrapping_add(mq) >> 64) as u64;
    if t >= Q {
        t - Q
    } else {
        t
    }
}

fn old_shoup_mul(a: u64, b: u64, b_shoup: u64, q: u64) -> u64 {
    let q_est = ((a as u128) * (b_shoup as u128)) >> 64;
    let r = ((a as u128)
        .wrapping_mul(b as u128)
        .wrapping_sub(q_est.wrapping_mul(q as u128))) as u64;
    if r >= q {
        r - q
    } else {
        r
    }
}

fn old_add(u: u64, v: u64, q: u64) -> u64 {
    if u + v >= q {
        u + v - q
    } else {
        u + v
    }
}

fn old_sub(u: u64, v: u64, q: u64) -> u64 {
    if u >= v {
        u - v
    } else {
        q - v + u
    }
}

/// `mul(hi, w)` for the forward butterflies, `mul(diff, w)` for the inverse.
fn old_forward_ladder(coeffs: &mut [u64], psi: &[u64], q: u64, mul: impl Fn(u64, u64) -> u64) {
    let n = coeffs.len();
    let mut t = n;
    let mut m = 1;
    while m < n {
        t >>= 1;
        for i in 0..m {
            let j1 = 2 * i * t;
            let w = psi[m + i];
            for j in j1..j1 + t {
                let u = coeffs[j];
                let v = mul(coeffs[j + t], w);
                coeffs[j] = old_add(u, v, q);
                coeffs[j + t] = old_sub(u, v, q);
            }
        }
        m <<= 1;
    }
}

fn old_inverse_ladder(coeffs: &mut [u64], psi_inv: &[u64], q: u64, mul: impl Fn(u64, u64) -> u64) {
    let n = coeffs.len();
    let mut t = 1;
    let mut m = n;
    while m > 1 {
        m >>= 1;
        for i in 0..m {
            let j2 = i * 2 * t;
            let w = psi_inv[m + i];
            for j in j2..j2 + t {
                let u = coeffs[j];
                let v = coeffs[j + t];
                coeffs[j] = old_add(u, v, q);
                coeffs[j + t] = mul(old_sub(u, v, q), w);
            }
        }
        t <<= 1;
    }
}

/// The pre-rewrite `forward`, limb by limb.
fn old_forward(ctx: &NttContext, coeffs: &mut [u64]) {
    let n = ctx.n;
    for (idx, &q) in ctx.moduli.iter().enumerate() {
        let qi = ctx.q_inv_neg[idx];
        let limb = &mut coeffs[idx * n..(idx + 1) * n];
        for c in limb.iter_mut() {
            *c = old_mont_mul(*c, ctx.r_squared[idx], q, qi);
        }
        if ctx.is_solinas_default_q() {
            old_forward_ladder(limb, &ctx.psi_powers[idx], q, |a, w| {
                old_solinas_mul(a, w, qi)
            });
        } else {
            old_forward_ladder(limb, &ctx.psi_powers[idx], q, |a, w| {
                old_mont_mul(a, w, q, qi)
            });
        }
    }
}

/// The pre-rewrite `inverse_inplace`: output left in Montgomery form.
fn old_inverse_inplace(ctx: &NttContext, coeffs: &mut [u64]) {
    let n = ctx.n;
    for (idx, &q) in ctx.moduli.iter().enumerate() {
        let qi = ctx.q_inv_neg[idx];
        let limb = &mut coeffs[idx * n..(idx + 1) * n];
        let mul = |a: u64, w: u64| {
            if ctx.is_solinas_default_q() {
                old_solinas_mul(a, w, qi)
            } else {
                old_mont_mul(a, w, q, qi)
            }
        };
        old_inverse_ladder(limb, &ctx.psi_inv_powers[idx], q, mul);
        for c in limb.iter_mut() {
            *c = mul(*c, ctx.n_inv[idx]);
        }
    }
}

fn old_inverse(ctx: &NttContext, coeffs: &mut [u64]) {
    old_inverse_inplace(ctx, coeffs);
    let n = ctx.n;
    for (idx, &q) in ctx.moduli.iter().enumerate() {
        for c in &mut coeffs[idx * n..(idx + 1) * n] {
            *c = old_mont_mul(*c, 1, q, ctx.q_inv_neg[idx]);
        }
    }
}

fn old_forward_shoup(ctx: &NttContext, coeffs: &mut [u64]) {
    let n = ctx.n;
    for (idx, &q) in ctx.moduli.iter().enumerate() {
        let (psi, twin) = (&ctx.psi_powers_std[idx], &ctx.psi_powers_shoup[idx]);
        let limb = &mut coeffs[idx * n..(idx + 1) * n];
        // the twin is looked up by twiddle position, so the ladder is inlined
        let mut t = n;
        let mut m = 1;
        while m < n {
            t >>= 1;
            for i in 0..m {
                let j1 = 2 * i * t;
                for j in j1..j1 + t {
                    let u = limb[j];
                    let v = old_shoup_mul(limb[j + t], psi[m + i], twin[m + i], q);
                    limb[j] = old_add(u, v, q);
                    limb[j + t] = old_sub(u, v, q);
                }
            }
            m <<= 1;
        }
    }
}

fn old_inverse_shoup(ctx: &NttContext, coeffs: &mut [u64]) {
    let n = ctx.n;
    for (idx, &q) in ctx.moduli.iter().enumerate() {
        let (psi, twin) = (&ctx.psi_inv_powers_std[idx], &ctx.psi_inv_powers_shoup[idx]);
        let limb = &mut coeffs[idx * n..(idx + 1) * n];
        let mut t = 1;
        let mut m = n;
        while m > 1 {
            m >>= 1;
            for i in 0..m {
                let j2 = i * 2 * t;
                for j in j2..j2 + t {
                    let u = limb[j];
                    let v = limb[j + t];
                    limb[j] = old_add(u, v, q);
                    limb[j + t] = old_shoup_mul(old_sub(u, v, q), psi[m + i], twin[m + i], q);
                }
            }
            t <<= 1;
        }
        for c in limb.iter_mut() {
            *c = old_shoup_mul(*c, ctx.n_inv_std[idx], ctx.n_inv_shoup[idx], q);
        }
    }
}

/// Operand edges plus random draws, each below its limb's modulus.
fn vectors(ctx: &NttContext, rng: &mut ChaCha20Rng) -> Vec<Vec<u64>> {
    let n = ctx.n;
    let per_limb = |f: &dyn Fn(u64, usize) -> u64| -> Vec<u64> {
        ctx.moduli
            .iter()
            .flat_map(|&q| (0..n).map(move |i| (q, i)))
            .map(|(q, i)| f(q, i))
            .collect()
    };
    let mut out = vec![
        per_limb(&|_, _| 0),
        per_limb(&|_, _| 1),
        per_limb(&|q, _| q - 1),
        per_limb(&|q, _| q - 2),
        per_limb(&|q, i| if i % 2 == 0 { 0 } else { q - 1 }),
        per_limb(&|q, i| if i % 3 == 0 { q - 1 } else { 1 }),
    ];
    for _ in 0..8 {
        let v = ctx
            .moduli
            .iter()
            .flat_map(|&q| (0..n).map(move |_| q))
            .map(|q| rng.gen_range(0..q))
            .collect();
        out.push(v);
    }
    out
}

#[test]
fn correction_helpers_match_the_branching_spelling_at_every_edge() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6e74_7431);
    for &moduli in &MODULI_SETS {
        for &q in moduli {
            let unreduced = [0, 1, q - 2, q - 1, q, q + 1, 2 * q - 2, 2 * q - 1];
            for x in unreduced {
                assert_eq!(
                    csubq(x, q),
                    if x >= q { x - q } else { x },
                    "csubq({x}, {q})"
                );
                assert_eq!(sub_mod_mask(x, q, q), csubq(x, q), "mask csubq({x}, {q})");
            }
            let operands = [0, 1, 2, q - 2, q - 1];
            for u in operands {
                for v in operands {
                    assert_eq!(csubq(u + v, q), old_add(u, v, q), "add {u} {v} mod {q}");
                    assert_eq!(
                        sub_mod_branchless(u, v, q),
                        old_sub(u, v, q),
                        "sub {u} {v} mod {q}"
                    );
                    assert_eq!(sub_mod_mask(u, v, q), old_sub(u, v, q), "mask {u} {v} {q}");
                    assert_eq!(sub_mod_mask(u + v, q, q), old_add(u, v, q));
                }
            }
            for _ in 0..RANDOM_SCALARS {
                let (u, v) = (rng.gen_range(0..q), rng.gen_range(0..q));
                assert_eq!(csubq(u + v, q), old_add(u, v, q));
                assert_eq!(sub_mod_branchless(u, v, q), old_sub(u, v, q));
                assert_eq!(sub_mod_mask(u, v, q), old_sub(u, v, q));
                let x = rng.gen_range(0..2 * q);
                assert_eq!(csubq(x, q), if x >= q { x - q } else { x });
            }
        }
    }
}

#[test]
fn reduction_tails_match_the_branching_spelling() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6e74_7432);
    for &moduli in &MODULI_SETS {
        let ctx = NttContext::with_moduli(16, moduli);
        for (idx, &q) in moduli.iter().enumerate() {
            let qi = ctx.q_inv_neg[idx];
            let edges = [0, 1, 2, q - 2, q - 1, ctx.r_squared[idx]];
            let mut arms = [0usize; 2];
            let mut check = |a: u64, b: u64| {
                arms[usize::from(redc_t(a, b, q, qi) >= q)] += 1;
                let want = old_mont_mul(a, b, q, qi);
                assert_eq!(
                    ctx.montgomery_mul_at(a, b, idx),
                    want,
                    "redc {a}*{b} mod {q}"
                );
                assert_eq!(NttContext::to_montgomery(a, q, b, qi), want);
                if moduli.len() == 1 {
                    assert_eq!(ctx.pointwise_mul_single(a, b), want);
                }
                let twin = NttContext::shoup_precompute(b, q);
                assert_eq!(
                    NttContext::shoup_mul_at(a, b, twin, q),
                    old_shoup_mul(a, b, twin, q),
                    "shoup {a}*{b} mod {q}"
                );
                if q == DEFAULT_Q {
                    let want = old_solinas_mul(a, b, qi);
                    assert_eq!(solinas_mont_mul_default_q(a, b, qi), want);
                    assert_eq!(ctx.pointwise_mul_single_at(a, b, 0), want);
                }
            };
            for a in edges {
                for b in edges {
                    check(a, b);
                }
            }
            for _ in 0..RANDOM_SCALARS {
                check(rng.gen_range(0..q), rng.gen_range(0..q));
            }
            // Below 2^32, t < q + q^2/2^64 < q + 1 and t = q forces ab = 0 mod q,
            // so the REDC correction is dead there; the helper test covers its arm.
            assert!(arms[0] > 0, "REDC arms at {q}: {arms:?}");
            if q > 1 << 32 {
                assert!(arms[1] > 0, "REDC correction never taken at {q}: {arms:?}");
            }
        }
    }
}

#[test]
fn transforms_and_products_match_the_branching_spelling() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6e74_7433);
    for &moduli in &MODULI_SETS {
        for n in DIMS {
            let ctx = NttContext::with_moduli(n, moduli);
            let inputs = vectors(&ctx, &mut rng);
            for (k, input) in inputs.iter().enumerate() {
                let tag = format!("moduli {moduli:?} n {n} vector {k}");

                let mut new = input.clone();
                let mut old = input.clone();
                ctx.forward(&mut new);
                old_forward(&ctx, &mut old);
                assert_eq!(new, old, "forward, {tag}");

                // Montgomery-form input for the in-place pair and the product
                let mont = old.clone();
                let mut new = input.clone();
                ctx.forward_inplace(&mut new);
                let mut old = input.clone();
                old_forward_ladders_only(&ctx, &mut old);
                assert_eq!(new, old, "forward_inplace, {tag}");

                let mut new = mont.clone();
                let mut old = mont.clone();
                ctx.inverse(&mut new);
                old_inverse(&ctx, &mut old);
                assert_eq!(new, old, "inverse, {tag}");
                assert_eq!(&new, input, "round trip, {tag}");

                let mut new = input.clone();
                let mut old = input.clone();
                ctx.inverse_inplace(&mut new);
                old_inverse_inplace(&ctx, &mut old);
                assert_eq!(new, old, "inverse_inplace, {tag}");

                let other = &inputs[(k + 3) % inputs.len()];
                let mut new = vec![0u64; input.len()];
                ctx.pointwise_mul(input, other, &mut new);
                let old: Vec<u64> = (0..input.len())
                    .map(|i| {
                        let idx = i / n;
                        let (q, qi) = (ctx.moduli[idx], ctx.q_inv_neg[idx]);
                        if ctx.is_solinas_default_q() {
                            old_solinas_mul(input[i], other[i], qi)
                        } else {
                            old_mont_mul(input[i], other[i], q, qi)
                        }
                    })
                    .collect();
                assert_eq!(new, old, "pointwise_mul, {tag}");

                let mut new = input.clone();
                let mut old = input.clone();
                ctx.forward_shoup(&mut new);
                old_forward_shoup(&ctx, &mut old);
                assert_eq!(new, old, "forward_shoup, {tag}");
                let shoup_ntt = old;

                let mut new = shoup_ntt.clone();
                let mut old = shoup_ntt.clone();
                ctx.inverse_shoup(&mut new);
                old_inverse_shoup(&ctx, &mut old);
                assert_eq!(new, old, "inverse_shoup, {tag}");

                let twins = ctx.shoup_precompute_vec(other);
                let mut new = vec![0u64; input.len()];
                ctx.pointwise_mul_shoup(input, other, &twins, &mut new);
                let old: Vec<u64> = (0..input.len())
                    .map(|i| old_shoup_mul(input[i], other[i], twins[i], ctx.moduli[i / n]))
                    .collect();
                assert_eq!(new, old, "pointwise_mul_shoup, {tag}");

                if ctx.is_solinas_default_q() {
                    let qi = ctx.q_inv_neg[0];
                    let mut new = input.clone();
                    let mut old = input.clone();
                    ctx.forward_solinas(&mut new);
                    old_forward(&ctx, &mut old);
                    assert_eq!(new, old, "forward_solinas, {tag}");

                    let mut new = mont.clone();
                    let mut old = mont.clone();
                    ctx.inverse_solinas(&mut new);
                    old_inverse(&ctx, &mut old);
                    assert_eq!(new, old, "inverse_solinas, {tag}");

                    let mut new = vec![0u64; n];
                    let mut sliced = vec![0u64; n];
                    ctx.pointwise_mul_solinas(input, other, &mut new);
                    pointwise_solinas_mont_mul(input, other, &mut sliced, qi);
                    let old: Vec<u64> = (0..n)
                        .map(|i| old_solinas_mul(input[i], other[i], qi))
                        .collect();
                    assert_eq!(new, old, "pointwise_mul_solinas, {tag}");
                    assert_eq!(sliced, old, "pointwise_solinas_mont_mul, {tag}");
                }
            }
        }
    }
}

/// `forward_inplace` skips the Montgomery lift; the ladder alone.
fn old_forward_ladders_only(ctx: &NttContext, coeffs: &mut [u64]) {
    let n = ctx.n;
    for (idx, &q) in ctx.moduli.iter().enumerate() {
        let qi = ctx.q_inv_neg[idx];
        let limb = &mut coeffs[idx * n..(idx + 1) * n];
        if ctx.is_solinas_default_q() {
            old_forward_ladder(limb, &ctx.psi_powers[idx], q, |a, w| {
                old_solinas_mul(a, w, qi)
            });
        } else {
            old_forward_ladder(limb, &ctx.psi_powers[idx], q, |a, w| {
                old_mont_mul(a, w, q, qi)
            });
        }
    }
}
