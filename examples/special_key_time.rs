//! Timing of the special-modulus packing path at full size: server respond per query on one
//! thread and on a 16-thread pool, the pieces of a respond, the transform kernels that
//! actually run at 28-31-bit primes next to the 60-bit ones, and the client steps.
//!
//! Cells of a group are interleaved round-robin with a rotating start; the load average is
//! recorded per sample. Environment: KEYBENCH_OUT (required), KEYBENCH_ROUNDS, KEYBENCH_WARM,
//! KEYBENCH_GROUPS.
//!
//! Every respond cell reads a per-shard step-1 table that is built before the timers start;
//! its size per layout is written to the stats as `store_bytes` lines.

// The loop bodies must land inside their `#[target_feature]` callers to be compiled for the
// wide vector extensions; a plain `#[inline]` leaves that to the optimiser.
#![allow(clippy::inline_always)]
// A bench stops at the first step that fails.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use raven_inspire::inspiring::{
    packing_offline, packing_online_fully_ntt, ClientPackingKeys, OfflinePackingKeys, PackParams,
};
use raven_inspire::math::mod_q::DEFAULT_Q;
use raven_inspire::math::{GaussianSampler, NttContext, Poly};
use raven_inspire::rlwe::RlweSecretKey;
use raven_inspire::special_key::{
    body_values, decode_nibbles, expand_key_mask, expand_query_mask, keygen, mod_down,
    nibbles_to_bytes, phase, query, recursion, respond, response_bytes, step1_direct, step1_stored,
    PackCtx, Packed, Query, Rns, Secret, Segment, Shard, Step1, Store, SET_Q28, SET_Q29,
};
use raven_inspire::InspireParams;
use rayon::prelude::*;

const D: usize = 2048;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn load1() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .unwrap_or_default()
        .split_whitespace()
        .next()
        .unwrap_or("?")
        .to_owned()
}

type Cell<'a> = (&'a str, Box<dyn FnMut(usize) -> Sample + 'a>);

struct Sink {
    raw: String,
    stats: String,
    rounds: usize,
    warm: usize,
}

impl Sink {
    /// Runs every cell once per round in rotating order and records the samples.
    fn group(&mut self, group: &str, cells: &mut [Cell<'_>]) {
        let n = cells.len();
        let mut samples: Vec<Vec<Sample>> = vec![Vec::new(); n];
        let before = load1();
        for round in 0..self.warm + self.rounds {
            for k in 0..n {
                let c = (k + round) % n;
                let sample = (cells[c].1)(round);
                if round >= self.warm {
                    samples[c].push(sample);
                    writeln!(
                        self.raw,
                        "{group}\t{}\t{}\t{:.5}\t{:.5}\t{}",
                        cells[c].0,
                        round - self.warm,
                        sample.wall,
                        sample.cpu,
                        load1()
                    )
                    .unwrap();
                }
            }
        }
        let after = load1();
        for (c, v) in samples.iter().enumerate() {
            let n = v.len();
            let sorted = |f: fn(&Sample) -> f64| {
                let mut x: Vec<f64> = v.iter().map(f).collect();
                x.sort_by(f64::total_cmp);
                x
            };
            let (wall, cpu) = (sorted(|s| s.wall), sorted(|s| s.cpu));
            let pick = |x: &[f64], q: f64| x[((n - 1) as f64 * q).round() as usize];
            let num = |x: f64| {
                if x.is_finite() {
                    format!("{x:.4}")
                } else {
                    "null".to_owned()
                }
            };
            let line = format!(
                "{{\"group\":\"{group}\",\"cell\":\"{}\",\"n\":{n},\"min_ms\":{:.4},\"median_ms\":{:.4},\
                 \"mean_ms\":{:.4},\"p90_ms\":{:.4},\"max_ms\":{:.4},\"cpu_min_ms\":{},\"cpu_median_ms\":{},\
                 \"cpu_max_ms\":{},\"load_before\":{before},\"load_after\":{after}}}",
                cells[c].0,
                wall[0],
                pick(&wall, 0.5),
                wall.iter().sum::<f64>() / n as f64,
                pick(&wall, 0.9),
                wall[n - 1],
                num(cpu[0]),
                num(pick(&cpu, 0.5)),
                num(cpu[n - 1])
            );
            println!("{line}");
            self.stats.push_str(&line);
            self.stats.push('\n');
        }
    }
}

/// Wall time and on-CPU time of the calling thread, both in ms.
#[derive(Clone, Copy)]
struct Sample {
    wall: f64,
    cpu: f64,
}

/// Nanoseconds this thread has spent on a CPU, from the scheduler's own accounting.
fn thread_cpu_ns() -> f64 {
    std::fs::read_to_string("/proc/thread-self/schedstat")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()))
        .unwrap_or(f64::NAN)
}

fn timed<R>(f: impl FnOnce() -> R) -> Sample {
    let c0 = thread_cpu_ns();
    let t = Instant::now();
    black_box(f());
    let wall = t.elapsed().as_secs_f64() * 1e3;
    Sample {
        wall,
        cpu: (thread_cpu_ns() - c0) / 1e6,
    }
}

fn wall_only(s: Sample) -> Sample {
    Sample {
        wall: s.wall,
        cpu: f64::NAN,
    }
}

/// A 32-bit-word negacyclic transform over one prime below 2^31, written so the compiler can
/// vectorise it: what a kernel for these primes could look like.
mod k32 {
    pub struct Ntt32 {
        pub q: u32,
        pub qinv: u32,
        n: usize,
        psi: Vec<u32>,
        psi_inv: Vec<u32>,
        /// 1/n times R: an inverse transform scaled by it keeps the input's form.
        pub n_inv_keep: u32,
        /// 1/n: an inverse transform scaled by it strips one factor of R.
        pub n_inv_strip: u32,
        r2: u32,
    }

    #[inline(always)]
    pub fn mm(a: u32, b: u32, q: u32, qinv: u32) -> u32 {
        let t = u64::from(a) * u64::from(b);
        let m = (t as u32).wrapping_mul(qinv);
        let u = ((t + u64::from(m) * u64::from(q)) >> 32) as u32;
        u.min(u.wrapping_sub(q))
    }

    #[inline(always)]
    pub fn add(a: u32, b: u32, q: u32) -> u32 {
        let s = a + b;
        s.min(s.wrapping_sub(q))
    }

    #[inline(always)]
    pub fn sub(a: u32, b: u32, q: u32) -> u32 {
        let s = a.wrapping_sub(b);
        s.min(s.wrapping_add(q))
    }

    fn pow(mut b: u64, mut e: u64, q: u64) -> u64 {
        let mut r = 1u64;
        b %= q;
        while e > 0 {
            if e & 1 == 1 {
                r = r * b % q;
            }
            b = b * b % q;
            e >>= 1;
        }
        r
    }

    impl Ntt32 {
        /// Same root and slot order as the crate's transform over `q`.
        pub fn new(n: usize, q: u32) -> Self {
            let q64 = u64::from(q);
            let mut qinv = 1u32;
            for _ in 0..5 {
                qinv = qinv.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(qinv)));
            }
            let qinv = qinv.wrapping_neg();
            let r2 = ((1u128 << 64) % u128::from(q)) as u32;
            let psi = (2..q64)
                .map(|x| pow(x, (q64 - 1) / (2 * n as u64), q64))
                .find(|&c| pow(c, n as u64, q64) == q64 - 1)
                .unwrap();
            let psi_inv = pow(psi, q64 - 2, q64);
            let bits = n.trailing_zeros();
            let table = |root: u64| -> Vec<u32> {
                (0..n)
                    .map(|k| {
                        let e = (k as u32).reverse_bits() >> (32 - bits);
                        mm(pow(root, u64::from(e), q64) as u32, r2, q, qinv)
                    })
                    .collect()
            };
            let n_inv = pow(n as u64, q64 - 2, q64) as u32;
            Self {
                q,
                qinv,
                n,
                psi: table(psi),
                psi_inv: table(psi_inv),
                n_inv_keep: mm(n_inv, r2, q, qinv),
                n_inv_strip: n_inv,
                r2,
            }
        }

        pub fn to_mont(&self, a: u32) -> u32 {
            mm(a, self.r2, self.q, self.qinv)
        }

        #[inline(always)]
        pub fn forward_body(&self, a: &mut [u32]) {
            let (q, qinv) = (self.q, self.qinv);
            let mut t = self.n;
            let mut m = 1;
            while m < self.n {
                t >>= 1;
                for (i, chunk) in a.chunks_exact_mut(2 * t).enumerate() {
                    let w = self.psi[m + i];
                    let (lo, hi) = chunk.split_at_mut(t);
                    for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                        let u = *x;
                        let v = mm(*y, w, q, qinv);
                        *x = add(u, v, q);
                        *y = sub(u, v, q);
                    }
                }
                m <<= 1;
            }
        }

        #[inline(always)]
        pub fn inverse_body(&self, a: &mut [u32], scale: u32) {
            let (q, qinv) = (self.q, self.qinv);
            let mut t = 1;
            let mut m = self.n;
            while m > 1 {
                m >>= 1;
                for (i, chunk) in a.chunks_exact_mut(2 * t).enumerate() {
                    let w = self.psi_inv[m + i];
                    let (lo, hi) = chunk.split_at_mut(t);
                    for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                        let u = *x;
                        let v = *y;
                        *x = add(u, v, q);
                        *y = mm(sub(u, v, q), w, q, qinv);
                    }
                }
                t <<= 1;
            }
            for x in a.iter_mut() {
                *x = mm(*x, scale, q, qinv);
            }
        }

        #[inline(always)]
        fn mul_acc_body(&self, acc: &mut [u32], a: &[u32], b: &[u32]) {
            let (q, qinv) = (self.q, self.qinv);
            for ((s, &x), &y) in acc.iter_mut().zip(a).zip(b) {
                *s = add(*s, mm(x, y, q, qinv), q);
            }
        }

        /// Forward transform, portable code generation.
        pub fn forward(&self, a: &mut [u32]) {
            self.forward_body(a);
        }

        /// Inverse of [`Self::forward`], keeping the input's form.
        pub fn inverse(&self, a: &mut [u32]) {
            self.inverse_body(a, self.n_inv_keep);
        }

        /// `acc += a * b` pointwise.
        pub fn mul_acc(&self, acc: &mut [u32], a: &[u32], b: &[u32]) {
            self.mul_acc_body(acc, a, b);
        }

        /// The same loops compiled with the wide vector extensions enabled.
        ///
        /// # Safety
        ///
        /// The host must support AVX-512 F, DQ, VL and BW.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw")]
        pub unsafe fn forward_wide(&self, a: &mut [u32]) {
            self.forward_body(a);
        }

        /// # Safety
        ///
        /// As [`Self::forward_wide`].
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw")]
        pub unsafe fn inverse_wide(&self, a: &mut [u32]) {
            self.inverse_body(a, self.n_inv_keep);
        }

        /// # Safety
        ///
        /// As [`Self::forward_wide`].
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw")]
        pub unsafe fn mul_acc_wide(&self, acc: &mut [u32], a: &[u32], b: &[u32]) {
            self.mul_acc_body(acc, a, b);
        }
    }
}

/// The whole respond on 32-bit words: what the server would run with kernels sized for these
/// primes. Checked byte for byte against the crate-based path before it is timed.
mod fast32 {
    use super::k32::{add, mm, sub, Ntt32};
    use raven_inspire::math::mod_inverse;
    use raven_inspire::special_key::{
        body_values, expand_query_mask, PackCtx, Packed, Query, Rns, Segment, Store,
    };

    pub struct Fast<'a> {
        ctx: &'a PackCtx,
        kq: Ntt32,
        kp: Ntt32,
        half_q: u32,
        half_p: u32,
        p_minus_q: u32,
        neg_off: u32,
        q_recip: u64,
        pinv_m: u32,
        pinv_m2: u32,
    }

    /// Both limbs in the transform domain, each value times R.
    pub struct Key32 {
        q: Vec<u32>,
        p: Vec<u32>,
    }

    impl<'a> Fast<'a> {
        pub fn new(ctx: &'a PackCtx) -> Self {
            let (q, p) = (ctx.lq.q as u32, ctx.lp.q as u32);
            assert!(p > q);
            let kq = Ntt32::new(ctx.d, q);
            let kp = Ntt32::new(ctx.d, p);
            let pinv = mod_inverse(u64::from(p % q), u64::from(q)) as u32;
            let pinv_m = kq.to_mont(pinv);
            Self {
                ctx,
                half_q: q / 2,
                half_p: p / 2,
                p_minus_q: p - q,
                neg_off: (q - p % q) % q,
                q_recip: (1u64 << 32) / u64::from(q),
                pinv_m,
                pinv_m2: kq.to_mont(pinv_m),
                kq,
                kp,
            }
        }

        /// From the crate's transform-domain form.
        pub fn key(&self, x: &Rns) -> Key32 {
            let ctx = self.ctx;
            Key32 {
                q: x.q
                    .iter()
                    .map(|&v| self.kq.to_mont(ctx.lq.ctx.from_mont(v) as u32))
                    .collect(),
                p: x.p
                    .iter()
                    .map(|&v| self.kp.to_mont(ctx.lp.ctx.from_mont(v) as u32))
                    .collect(),
            }
        }

        /// The key mask with `1/p` folded into its `q` limb.
        pub fn mask(&self, w: &Rns) -> Key32 {
            let mut k = self.key(w);
            for v in &mut k.q {
                *v = mm(*v, self.pinv_m, self.kq.q, self.kq.qinv);
            }
            k
        }

        pub fn store(&self, store: &Store) -> Vec<Vec<u32>> {
            let ctx = self.ctx;
            store
                .segs
                .iter()
                .map(|t| {
                    t.iter()
                        .map(|&v| self.kq.to_mont(ctx.lq.ctx.from_mont(v) as u32))
                        .collect()
                })
                .collect()
        }

        #[inline(always)]
        fn respond_body(
            &self,
            segs: &[Segment<'_>],
            store: &[Vec<u32>],
            w: &Key32,
            key: &Key32,
            queries: &[&Query],
        ) -> Packed {
            let ctx = self.ctx;
            let (d, gamma) = (ctx.d, ctx.gamma);
            let (kq, kp) = (&self.kq, &self.kp);
            let (q, qi, p, pi) = (kq.q, kq.qinv, kp.q, kp.qinv);

            let masks: Vec<Vec<u32>> = queries
                .iter()
                .map(|qr| {
                    let mut a: Vec<u32> = expand_query_mask(ctx, qr.a_seed)
                        .iter()
                        .map(|&v| kq.to_mont(v as u32))
                        .collect();
                    kq.forward_body(&mut a);
                    a
                })
                .collect();
            let mut r: Vec<Vec<u32>> = (0..gamma)
                .map(|i| {
                    let tab = &ctx.tables[i];
                    let mut out = vec![0u32; d];
                    for (seg, table) in segs.iter().zip(store) {
                        let a = &masks[seg.query];
                        let h = &table[i * d..(i + 1) * d];
                        for k in 0..d {
                            out[k] = add(out[k], mm(a[tab[k] as usize], h[k], q, qi), q);
                        }
                    }
                    out
                })
                .collect();
            let bodies: Vec<&[u64]> = queries.iter().map(|qr| &qr.b[..]).collect();
            let b_poly = body_values(ctx, segs, &bodies);

            let mut coeff = vec![0u32; d];
            let mut tp = vec![0u32; d];
            let mut cp = vec![0u32; d];
            let mut rq = vec![0u32; d];
            let mut acc_q = vec![0u32; d];
            let mut acc_p = vec![0u32; d];
            for i in (0..gamma - 1).rev() {
                let tab = &ctx.tables[i];
                let (lo, hi) = r.split_at_mut(i + 1);
                let ri = &mut lo[i];
                let rn = &hi[0];

                coeff.copy_from_slice(rn);
                kq.inverse_body(&mut coeff, kq.n_inv_strip);
                for k in 0..d {
                    let c = coeff[k];
                    tp[k] = c + if c > self.half_q { self.p_minus_q } else { 0 };
                }
                kp.forward_body(&mut tp);
                for k in 0..d {
                    cp[k] = mm(tp[k], w.p[tab[k] as usize], p, pi);
                }
                kp.inverse_body(&mut cp, kp.n_inv_keep);
                for k in 0..d {
                    let c = cp[k];
                    let est = c - ((u64::from(c) * self.q_recip) >> 32) as u32 * q;
                    let base = est.min(est.wrapping_sub(q));
                    rq[k] = add(base, if c > self.half_p { self.neg_off } else { 0 }, q);
                }
                kq.forward_body(&mut rq);
                for k in 0..d {
                    let src = tab[k] as usize;
                    let dr = sub(
                        mm(rn[k], w.q[src], q, qi),
                        mm(rq[k], self.pinv_m2, q, qi),
                        q,
                    );
                    ri[k] = add(ri[k], dr, q);
                    acc_q[k] = add(acc_q[k], mm(rn[k], key.q[src], q, qi), q);
                    acc_p[k] = add(acc_p[k], mm(tp[k], key.p[src], p, pi), p);
                }
            }
            let mut a_hat = r.swap_remove(0);
            kq.inverse_body(&mut a_hat, kq.n_inv_strip);
            kq.inverse_body(&mut acc_q, kq.n_inv_strip);
            kp.inverse_body(&mut acc_p, kp.n_inv_keep);
            let b = (0..d)
                .map(|k| {
                    let c = acc_p[k];
                    let est = c - ((u64::from(c) * self.q_recip) >> 32) as u32 * q;
                    let base = est.min(est.wrapping_sub(q));
                    let rem = add(base, if c > self.half_p { self.neg_off } else { 0 }, q);
                    let v = mm(sub(acc_q[k], rem, q), self.pinv_m, q, qi);
                    let total = u64::from(v) + if k < gamma { b_poly[k] } else { 0 };
                    total % u64::from(q)
                })
                .collect();
            Packed {
                a: a_hat.iter().map(|&v| u64::from(v)).collect(),
                b,
            }
        }

        pub fn respond(
            &self,
            segs: &[Segment<'_>],
            store: &[Vec<u32>],
            w: &Key32,
            key: &Key32,
            queries: &[&Query],
        ) -> Packed {
            self.respond_body(segs, store, w, key, queries)
        }

        /// # Safety
        ///
        /// The host must support AVX-512 F, DQ, VL and BW.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw")]
        pub unsafe fn respond_wide(
            &self,
            segs: &[Segment<'_>],
            store: &[Vec<u32>],
            w: &Key32,
            key: &Key32,
            queries: &[&Query],
        ) -> Packed {
            self.respond_body(segs, store, w, key, queries)
        }
    }
}

fn wide_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The 32-bit transform agrees with the crate's on a negacyclic product.
fn check_k32(q: u64) {
    let ctx = NttContext::new(D, q);
    let k = k32::Ntt32::new(D, q as u32);
    let mut rng = ChaCha20Rng::seed_from_u64(q);
    let a: Vec<u64> = (0..D).map(|_| rng.gen_range(0..q)).collect();
    let b: Vec<u64> = (0..D).map(|_| rng.gen_range(0..q)).collect();
    let want = Poly::from_coeffs(a.clone(), q).mul_ntt(&Poly::from_coeffs(b.clone(), q), &ctx);
    for wide in [false, true] {
        if wide && !wide_ok() {
            continue;
        }
        let mut x: Vec<u32> = a.iter().map(|&v| k.to_mont(v as u32)).collect();
        let mut y: Vec<u32> = b.iter().map(|&v| k.to_mont(v as u32)).collect();
        let mut z = vec![0u32; D];
        if wide {
            // SAFETY: `wide_ok` checked the extensions.
            #[cfg(target_arch = "x86_64")]
            unsafe {
                k.forward_wide(&mut x);
                k.forward_wide(&mut y);
                k.mul_acc_wide(&mut z, &x, &y);
                k.inverse_wide(&mut z);
            }
        } else {
            k.forward(&mut x);
            k.forward(&mut y);
            k.mul_acc(&mut z, &x, &y);
            k.inverse(&mut z);
        }
        // z holds the product in Montgomery form with one factor of R left over.
        let got: Vec<u64> = z
            .iter()
            .map(|&v| {
                (u128::from(v) * u128::from(raven_inspire::math::mod_inverse((1u64 << 32) % q, q))
                    % u128::from(q)) as u64
            })
            .collect();
        assert_eq!(
            got,
            want.coeffs(),
            "32-bit transform disagrees at q = {q}, wide = {wide}"
        );
    }
}

struct Layout {
    name: &'static str,
    gamma: usize,
    shards: Vec<Shard>,
}

fn layouts(rng: &mut ChaCha20Rng) -> Vec<Layout> {
    let mut random = |rows: usize, cols: usize| Shard {
        rows,
        cols,
        vals: (0..rows * cols).map(|_| rng.gen_range(-8i8..=7)).collect(),
    };
    vec![
        Layout {
            name: "single",
            gamma: 512,
            shards: vec![random(D, 512)],
        },
        Layout {
            name: "pair",
            gamma: 1024,
            shards: vec![random(D, 512), random(D, 512)],
        },
        Layout {
            name: "dial",
            gamma: 1024,
            shards: vec![random(D / 2, 1024)],
        },
    ]
}

fn main() {
    let out_dir =
        std::env::var("KEYBENCH_OUT").expect("KEYBENCH_OUT must name the output directory");
    std::fs::create_dir_all(&out_dir).unwrap();
    let groups: Vec<String> = std::env::var("KEYBENCH_GROUPS")
        .map(|v| v.split(',').map(str::to_owned).collect())
        .unwrap_or_default();
    let want = |g: &str| groups.is_empty() || groups.iter().any(|x| g.starts_with(x.as_str()));
    let mut sink = Sink {
        raw: String::from("group\tcell\tround\twall_ms\tthread_cpu_ms\tload1\n"),
        stats: String::new(),
        rounds: env_usize("KEYBENCH_ROUNDS", 40),
        warm: env_usize("KEYBENCH_WARM", 3),
    };
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool16 = rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build()
        .unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(0x7132);
    println!(
        "{{\"env\":\"cpus {} wide_vectors {} ifma {} rounds {} warm {}\"}}",
        std::thread::available_parallelism().map_or(0, usize::from),
        wide_ok(),
        std::is_x86_feature_detected!("avx512ifma"),
        sink.rounds,
        sink.warm
    );

    // ---- transform and multiply-accumulate kernels, per call on 2,048 coefficients
    if want("kernels") {
        for q in [SET_Q29.0, SET_Q29.1, SET_Q28.0] {
            check_k32(q);
        }
        const REP: usize = 200;
        let per = |s: Sample| Sample {
            wall: s.wall / REP as f64,
            cpu: s.cpu / REP as f64,
        };
        let mk64 = |q: u64, rng: &mut ChaCha20Rng| -> Vec<u64> {
            (0..D).map(|_| rng.gen_range(0..q)).collect()
        };
        let c29 = NttContext::new(D, SET_Q29.0);
        let c31 = NttContext::new(D, SET_Q29.1);
        let c60 = NttContext::new(D, DEFAULT_Q);
        let k29 = k32::Ntt32::new(D, SET_Q29.0 as u32);
        let k31 = k32::Ntt32::new(D, SET_Q29.1 as u32);
        let v29 = mk64(SET_Q29.0, &mut rng);
        let v31 = mk64(SET_Q29.1, &mut rng);
        let v60 = mk64(DEFAULT_Q, &mut rng);
        let u29: Vec<u32> = v29.iter().map(|&x| x as u32).collect();
        let u31: Vec<u32> = v31.iter().map(|&x| x as u32).collect();
        let p60a = {
            let mut p = Poly::from_coeffs(v60.clone(), DEFAULT_Q);
            p.to_ntt(&c60);
            p
        };
        let p29a = {
            let mut p = Poly::from_coeffs(v29.clone(), SET_Q29.0);
            p.to_ntt(&c29);
            p
        };
        let two = [SET_Q29.0, SET_Q29.1];
        let c2 = NttContext::with_moduli(D, &two);
        let p2a = {
            let mut p = Poly::from_coeffs_moduli(v29.clone(), &two);
            p.to_ntt(&c2);
            p
        };
        let roundtrip64 = |ctx: &NttContext, v: &[u64]| {
            let mut x = v.to_vec();
            timed(|| {
                for _ in 0..REP / 2 {
                    ctx.forward(&mut x);
                    ctx.inverse(&mut x);
                }
            })
        };
        let roundtrip32 = |k: &k32::Ntt32, v: &[u32], wide: bool| {
            let mut x = v.to_vec();
            timed(|| {
                for _ in 0..REP / 2 {
                    if wide {
                        // SAFETY: the cell is only registered when `wide_ok` holds.
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            k.forward_wide(&mut x);
                            k.inverse_wide(&mut x);
                        }
                    } else {
                        k.forward(&mut x);
                        k.inverse(&mut x);
                    }
                }
            })
        };
        let mulacc_poly = |a: &Poly, ctx: &NttContext| {
            let mut acc = a.clone();
            timed(|| {
                for _ in 0..REP {
                    acc.mul_acc_ntt_domain(a, a, ctx);
                }
            })
        };
        let mulacc32 = |k: &k32::Ntt32, v: &[u32], wide: bool| {
            let mut acc = v.to_vec();
            timed(|| {
                for _ in 0..REP {
                    if wide {
                        // SAFETY: as above.
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            k.mul_acc_wide(&mut acc, v, v);
                        }
                    } else {
                        k.mul_acc(&mut acc, v, v);
                    }
                }
            })
        };
        let mut cells: Vec<Cell<'_>> = vec![
            (
                "ntt_60bit_crate",
                Box::new(|_| per(roundtrip64(&c60, &v60))),
            ),
            (
                "ntt_29bit_crate_generic",
                Box::new(|_| per(roundtrip64(&c29, &v29))),
            ),
            (
                "ntt_31bit_crate_generic",
                Box::new(|_| per(roundtrip64(&c31, &v31))),
            ),
            (
                "ntt_29bit_u32_portable",
                Box::new(|_| per(roundtrip32(&k29, &u29, false))),
            ),
            (
                "ntt_31bit_u32_portable",
                Box::new(|_| per(roundtrip32(&k31, &u31, false))),
            ),
            (
                "mulacc_60bit_crate",
                Box::new(|_| per(mulacc_poly(&p60a, &c60))),
            ),
            (
                "mulacc_29bit_crate_generic",
                Box::new(|_| per(mulacc_poly(&p29a, &c29))),
            ),
            (
                "mulacc_29x31_two_limb_crate_generic",
                Box::new(|_| per(mulacc_poly(&p2a, &c2))),
            ),
            (
                "mulacc_29bit_u32_portable",
                Box::new(|_| per(mulacc32(&k29, &u29, false))),
            ),
        ];
        if wide_ok() {
            cells.push((
                "ntt_29bit_u32_wide",
                Box::new(|_| per(roundtrip32(&k29, &u29, true))),
            ));
            cells.push((
                "ntt_31bit_u32_wide",
                Box::new(|_| per(roundtrip32(&k31, &u31, true))),
            ));
            cells.push((
                "mulacc_29bit_u32_wide",
                Box::new(|_| per(mulacc32(&k29, &u29, true))),
            ));
        }
        sink.group("kernels", &mut cells);
    }

    // ---- today's gadget packing at the same widths, for scale
    for gamma in [512usize, 1024] {
        let name = format!("today_gadget_g{gamma}");
        if !want(&name) {
            continue;
        }
        let params = InspireParams::secure_128_d2048();
        let pp = PackParams::try_new(&params, gamma).unwrap();
        let ctx = params.ntt_context();
        let moduli = params.moduli().to_vec();
        let mut sampler = GaussianSampler::with_seed(6.4, 5);
        let sk = RlweSecretKey::generate(&params, &mut sampler);
        let w_seed = [3u8; 32];
        let offline = OfflinePackingKeys::generate(&pp, w_seed);
        let client = ClientPackingKeys::generate(&sk, &pp, w_seed, &mut sampler);
        let a_ct: Vec<Poly> = (0..gamma)
            .map(|_| Poly::random_with_rng_moduli(D, &moduli, &mut rng))
            .collect();
        let b_poly = Poly::random_with_rng_moduli(D, &moduli, &mut rng);
        let run = || {
            let pre = packing_offline(&pp, &offline, &a_ct, &ctx);
            packing_online_fully_ntt(&pre, &client.y_all_ntt, &b_poly, &ctx)
        };
        let mut cells: Vec<Cell<'_>> = vec![
            ("pack_1thread", Box::new(|_| pool1.install(|| timed(run)))),
            (
                "pack_16threads",
                Box::new(|_| pool16.install(|| wall_only(timed(run)))),
            ),
        ];
        sink.group(&name, &mut cells);
    }

    // ---- respond per query
    let layouts = layouts(&mut rng);
    for (set_name, set, sigma_q) in [("q29", SET_Q29, 6.4), ("q28", SET_Q28, 3.2)] {
        for lay in &layouts {
            let name = format!("respond_{set_name}_{}", lay.name);
            let bname = format!("pieces_{set_name}_{}", lay.name);
            let tname = format!("throughput_{set_name}_{}", lay.name);
            if !(want(&name) || want(&bname) || want(&tname)) {
                continue;
            }
            let ctx = PackCtx::new(D, lay.gamma, set.0, set.1, 16);
            let segs: Vec<Segment<'_>> = lay
                .shards
                .iter()
                .enumerate()
                .map(|(k, shard)| Segment { shard, query: k })
                .collect();
            let store = Store::build(&ctx, &segs);
            let mut s_sampler = GaussianSampler::with_seed(6.4, 11);
            let mut k_sampler = GaussianSampler::with_seed(6.4, 12);
            let mut q_sampler = GaussianSampler::with_seed(sigma_q, 13);
            let sk = Secret::sample(&ctx, &mut s_sampler);
            let w_seed = [9u8; 32];
            let (key, _) = keygen(&ctx, &sk, w_seed, &mut k_sampler);
            let w_ntt = expand_key_mask(&ctx, w_seed).to_ntt(&ctx);
            let key_ntt = key.to_ntt(&ctx);
            let total = sink.warm + sink.rounds;
            // One fresh query set per round, built outside the timers.
            let rounds: Vec<Vec<(Query, usize)>> = (0..total)
                .map(|_| {
                    segs.iter()
                        .map(|seg| {
                            let index = rng.gen_range(0..seg.shard.rows);
                            let mut a_seed = [0u8; 32];
                            rng.fill_bytes(&mut a_seed);
                            (query(&ctx, &sk, index, a_seed, &mut q_sampler).0, index)
                        })
                        .collect()
                })
                .collect();
            let body = lay.gamma / 512 * 492;
            let serve = |round: usize, step1: &Step1<'_>| -> Vec<u8> {
                let qs: Vec<&Query> = rounds[round].iter().map(|(q, _)| q).collect();
                let packed = respond(&ctx, &segs, step1, &w_ntt, &key_ntt, &qs);
                response_bytes(&ctx, &packed, body)
            };
            // Every round must decode before any of it is timed.
            for (round, made) in rounds.iter().enumerate().take(3) {
                let qs: Vec<&Query> = made.iter().map(|(q, _)| q).collect();
                let a = respond(&ctx, &segs, &Step1::Stored(&store), &w_ntt, &key_ntt, &qs);
                let b = respond(&ctx, &segs, &Step1::Direct, &w_ntt, &key_ntt, &qs);
                assert_eq!(a, b, "stored and direct step 1 differ in round {round}");
                let nib = decode_nibbles(&ctx, &phase(&ctx, &sk, &a), lay.gamma);
                let mut off = 0;
                for (seg, (_, index)) in segs.iter().zip(made) {
                    let sh = seg.shard;
                    for j in 0..sh.cols {
                        assert_eq!(
                            i64::from(nib[off + j]),
                            i64::from(sh.vals[index * sh.cols + j]) + 8
                        );
                    }
                    off += sh.cols;
                }
            }

            let fast = fast32::Fast::new(&ctx);
            let (store32, w32, key32) = (fast.store(&store), fast.mask(&w_ntt), fast.key(&key_ntt));
            let join = |v: Vec<usize>| {
                let parts: Vec<String> = v.iter().map(usize::to_string).collect();
                parts.join(",")
            };
            let line = format!(
                "{{\"store_bytes\":\"{set_name}_{}\",\"segments\":{},\"data_bytes_per_segment\":{},\
                 \"table_bytes_u64\":[{}],\"table_bytes_u32\":[{}]}}",
                lay.name,
                segs.len(),
                lay.shards[0].vals.len() / 2,
                join(store.table_bytes()),
                join(store32.iter().map(|t| size_of_val(&t[..])).collect()),
            );
            println!("{line}");
            sink.stats.push_str(&line);
            sink.stats.push('\n');
            let serve32 = |round: usize, wide: bool| -> Vec<u8> {
                let qs: Vec<&Query> = rounds[round].iter().map(|(q, _)| q).collect();
                let packed = if wide {
                    // SAFETY: only called with `wide` when `wide_ok` holds.
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        fast.respond_wide(&segs, &store32, &w32, &key32, &qs)
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    unreachable!()
                } else {
                    fast.respond(&segs, &store32, &w32, &key32, &qs)
                };
                response_bytes(&ctx, &packed, body)
            };
            for round in 0..total.min(6) {
                let want_bytes = serve(round, &Step1::Stored(&store));
                assert!(
                    serve32(round, false) == want_bytes,
                    "32-bit respond differs"
                );
                assert!(
                    !wide_ok() || serve32(round, true) == want_bytes,
                    "wide respond differs"
                );
            }

            if want(&name) {
                let mut cells: Vec<Cell<'_>> = vec![
                    (
                        "stored_1thread_u32_portable",
                        Box::new(|r| pool1.install(|| timed(|| serve32(r, false)))),
                    ),
                    (
                        "stored_1thread",
                        Box::new(|r| pool1.install(|| timed(|| serve(r, &Step1::Stored(&store))))),
                    ),
                    (
                        "stored_16threads",
                        Box::new(|r| {
                            pool16.install(|| wall_only(timed(|| serve(r, &Step1::Stored(&store)))))
                        }),
                    ),
                    (
                        "direct_1thread",
                        Box::new(|r| pool1.install(|| timed(|| serve(r, &Step1::Direct)))),
                    ),
                    (
                        "direct_16threads",
                        Box::new(|r| {
                            pool16.install(|| wall_only(timed(|| serve(r, &Step1::Direct))))
                        }),
                    ),
                ];
                if wide_ok() {
                    cells.push((
                        "stored_1thread_u32_wide",
                        Box::new(|r| pool1.install(|| timed(|| serve32(r, true)))),
                    ));
                }
                sink.group(&name, &mut cells);
            }

            if want(&bname) {
                let masks = |round: usize| -> Vec<Vec<u64>> {
                    rounds[round]
                        .iter()
                        .map(|(q, _)| {
                            let mut a = expand_query_mask(&ctx, q.a_seed);
                            ctx.lq.ctx.forward(&mut a);
                            a
                        })
                        .collect()
                };
                let mut cells: Vec<Cell<'_>> = vec![
                    (
                        "mask_expand_and_transform",
                        Box::new(|r| timed(|| masks(r))),
                    ),
                    (
                        "step1_stored",
                        Box::new(|r| {
                            let m = masks(r);
                            pool1.install(|| timed(|| step1_stored(&ctx, &segs, &store, &m)))
                        }),
                    ),
                    (
                        "step1_direct",
                        Box::new(|r| {
                            let m = masks(r);
                            pool1.install(|| timed(|| step1_direct(&ctx, &segs, &m)))
                        }),
                    ),
                    (
                        "body_dot_products",
                        Box::new(|r| {
                            let bodies: Vec<&[u64]> =
                                rounds[r].iter().map(|(q, _)| &q.b[..]).collect();
                            timed(|| body_values(&ctx, &segs, &bodies))
                        }),
                    ),
                    (
                        "recursion_with_key",
                        Box::new(|r| {
                            let m = masks(r);
                            let mut rr = pool1.install(|| step1_stored(&ctx, &segs, &store, &m));
                            timed(|| recursion(&ctx, &mut rr, &w_ntt, &[&key_ntt], |_, _| {}))
                        }),
                    ),
                    (
                        "recursion_mask_only",
                        Box::new(|r| {
                            let m = masks(r);
                            let mut rr = pool1.install(|| step1_stored(&ctx, &segs, &store, &m));
                            timed(|| recursion(&ctx, &mut rr, &w_ntt, &[], |_, _| {}))
                        }),
                    ),
                    (
                        "divide_and_serialise",
                        Box::new(|r| {
                            let x = Rns {
                                q: rounds[r][0].0.b.clone(),
                                p: rounds[r][0].0.b.clone(),
                            };
                            timed(|| {
                                let b = mod_down(&ctx, &x);
                                response_bytes(&ctx, &Packed { a: x.q.clone(), b }, body)
                            })
                        }),
                    ),
                    (
                        "key_register_two_transforms",
                        Box::new(|_| timed(|| key.to_ntt(&ctx))),
                    ),
                ];
                sink.group(&bname, &mut cells);
            }

            if want(&tname) {
                // A cell is the wall time of the whole batch: every response in it is in
                // flight that long. All responses of a batch read the same shard and table.
                let batch = |pool: &rayon::ThreadPool, width: usize, r: usize| {
                    pool.install(|| {
                        let s = timed(|| {
                            (0..width).into_par_iter().for_each(|k| {
                                black_box(serve((r + k) % total, &Step1::Stored(&store)));
                            });
                        });
                        wall_only(s)
                    })
                };
                let batch32 = |pool: &rayon::ThreadPool, width: usize, r: usize| {
                    pool.install(|| {
                        let s = timed(|| {
                            (0..width).into_par_iter().for_each(|k| {
                                black_box(serve32((r + k) % total, true));
                            });
                        });
                        wall_only(s)
                    })
                };
                let pool8 = rayon::ThreadPoolBuilder::new()
                    .num_threads(8)
                    .build()
                    .unwrap();
                let mut cells: Vec<Cell<'_>> = vec![
                    (
                        "batch_wall_8_in_flight_8_threads",
                        Box::new(|r| batch(&pool8, 8, r)),
                    ),
                    (
                        "batch_wall_16_in_flight_16_threads",
                        Box::new(|r| batch(&pool16, 16, r)),
                    ),
                ];
                if wide_ok() {
                    cells.push((
                        "batch_wall_8_in_flight_8_threads_u32_wide",
                        Box::new(|r| batch32(&pool8, 8, r)),
                    ));
                    cells.push((
                        "batch_wall_16_in_flight_16_threads_u32_wide",
                        Box::new(|r| batch32(&pool16, 16, r)),
                    ));
                }
                sink.group(&tname, &mut cells);
            }
        }
    }

    // ---- client steps
    for (set_name, set, sigma_q) in [("q29", SET_Q29, 6.4), ("q28", SET_Q28, 3.2)] {
        let name = format!("client_{set_name}");
        if !want(&name) {
            continue;
        }
        let server = PackCtx::new(D, 1024, set.0, set.1, 16);
        let ctx = PackCtx::client(D, 1024, set.0, set.1, 16);
        let shards = [Shard::constant(D, 512, 3), Shard::constant(D, 512, -5)];
        let segs: Vec<Segment<'_>> = shards
            .iter()
            .enumerate()
            .map(|(k, shard)| Segment { shard, query: k })
            .collect();
        let store = Store::build(&server, &segs);
        let mut s_sampler = GaussianSampler::with_seed(6.4, 21);
        let mut k_sampler = GaussianSampler::with_seed(6.4, 22);
        let mut q_sampler = GaussianSampler::with_seed(sigma_q, 23);
        let sk = Secret::sample(&ctx, &mut s_sampler);
        let (key, _) = keygen(&ctx, &sk, [4u8; 32], &mut k_sampler);
        let qs: Vec<Query> = (0..2)
            .map(|k| query(&ctx, &sk, 100 + k, [k as u8; 32], &mut q_sampler).0)
            .collect();
        let refs: Vec<&Query> = qs.iter().collect();
        let packed = respond(
            &server,
            &segs,
            &Step1::Stored(&store),
            &expand_key_mask(&server, [4u8; 32]).to_ntt(&server),
            &key.to_ntt(&server),
            &refs,
        );
        let want_bytes = |v: i8| vec![((v + 8) as u8) | (((v + 8) as u8) << 4); 246];
        let decode =
            |n: usize| nibbles_to_bytes(&decode_nibbles(&ctx, &phase(&ctx, &sk, &packed), n));
        assert_eq!(decode(492), want_bytes(3));
        assert_eq!(decode(1004)[256..], want_bytes(-5)[..]);

        let mut seed = 0u64;
        let mut cells: Vec<Cell<'_>> = vec![
            (
                "context_two_primes",
                Box::new(|_| timed(|| PackCtx::client(D, 1024, set.0, set.1, 16))),
            ),
            (
                "fresh_secret",
                Box::new(|r| {
                    let mut s = GaussianSampler::with_seed(6.4, 1000 + r as u64);
                    timed(|| Secret::sample(&ctx, &mut s))
                }),
            ),
            (
                "key_build",
                Box::new(|r| {
                    let mut s = GaussianSampler::with_seed(6.4, 2000 + r as u64);
                    timed(|| keygen(&ctx, &sk, [r as u8; 32], &mut s))
                }),
            ),
            (
                "query_build",
                Box::new(|r| {
                    seed += 1;
                    let mut s = GaussianSampler::with_seed(sigma_q, 3000 + seed);
                    timed(|| query(&ctx, &sk, r % D, [r as u8; 32], &mut s))
                }),
            ),
            ("decode_one_record", Box::new(|_| timed(|| decode(492)))),
            ("decode_two_records", Box::new(|_| timed(|| decode(1004)))),
        ];
        sink.group(&name, &mut cells);
    }

    std::fs::write(format!("{out_dir}/raw.tsv"), &sink.raw).unwrap();
    std::fs::write(format!("{out_dir}/stats.jsonl"), &sink.stats).unwrap();
    println!("TIMING DONE");
}
