//! The one-row special-modulus packing on the crate's generic 64-bit transform, with step 1
//! computed column by column from the query (no per-shard precompute). This is the form
//! that was checked against an exact-integer reference before the 32-bit path existed;
//! the 32-bit path must give the same bytes.

#![allow(dead_code, reason = "each test binary uses the parts it needs")]

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use raven_inspire::math::{mod_inverse, NttContext};

fn csubq(x: u64, q: u64) -> u64 {
    if x >= q {
        x - q
    } else {
        x
    }
}

pub struct Limb {
    pub q: u64,
    pub ctx: NttContext,
    qinv: u64,
    half: u64,
}

impl Limb {
    fn new(d: usize, q: u64) -> Self {
        let ctx = NttContext::new(d, q);
        let qinv = ctx.q_inv_neg_for_test(0);
        Self {
            q,
            ctx,
            qinv,
            half: q / 2,
        }
    }

    /// Montgomery product in the transform's own form.
    pub fn mm(&self, a: u64, b: u64) -> u64 {
        let ab = u128::from(a) * u128::from(b);
        let m = u128::from((ab as u64).wrapping_mul(self.qinv));
        csubq(((ab + m * u128::from(self.q)) >> 64) as u64, self.q)
    }

    fn sub(&self, a: u64, b: u64) -> u64 {
        csubq(a + self.q - b, self.q)
    }

    pub fn residue(&self, x: i64) -> u64 {
        x.rem_euclid(self.q as i64) as u64
    }

    pub fn centred(&self, x: u64) -> i64 {
        if x > self.half {
            x as i64 - self.q as i64
        } else {
            x as i64
        }
    }
}

pub struct PackCtx {
    pub d: usize,
    pub gamma: usize,
    pub delta: u64,
    pub lq: Limb,
    pub lp: Limb,
    pub g: usize,
    gen_pows: Vec<usize>,
    tables: Vec<Vec<u32>>,
    inv_gamma_m: u64,
    p_inv_m: u64,
    p_mod_q: u64,
    neg_off: u64,
}

fn derive_tables(limb: &Limb, d: usize, exps: &[usize]) -> Vec<Vec<u32>> {
    let mut x = vec![0u64; d];
    x[1] = 1;
    limb.ctx.forward(&mut x);
    let slot_of: std::collections::HashMap<u64, u32> =
        x.iter().enumerate().map(|(k, &v)| (v, k as u32)).collect();
    assert_eq!(slot_of.len(), d);
    exps.iter()
        .map(|&t| {
            let mut m = vec![0u64; d];
            if t < d {
                m[t] = 1;
            } else {
                m[t - d] = limb.q - 1;
            }
            limb.ctx.forward(&mut m);
            m.iter().map(|v| slot_of[v]).collect()
        })
        .collect()
}

impl PackCtx {
    pub fn new(d: usize, gamma: usize, q: u64, p: u64) -> Self {
        let lq = Limb::new(d, q);
        let lp = Limb::new(d, p);
        let g = 2 * d / gamma + 1;
        let mut gen_pows = Vec::with_capacity(d);
        let mut v = 1usize;
        for _ in 0..d {
            gen_pows.push(v);
            v = (v * g) % (2 * d);
        }
        let tables = derive_tables(&lq, d, &gen_pows[..gamma]);
        assert_eq!(tables, derive_tables(&lp, d, &gen_pows[..gamma]));
        Self {
            d,
            gamma,
            delta: q / 16,
            g,
            inv_gamma_m: lq.ctx.to_mont(mod_inverse(gamma as u64, q)),
            p_inv_m: lq.ctx.to_mont(mod_inverse(p % q, q)),
            p_mod_q: p % q,
            neg_off: (q - p % q) % q,
            gen_pows,
            tables,
            lq,
            lp,
        }
    }

    fn p_to_q_centred(&self, r: u64) -> u64 {
        let base = r % self.lq.q;
        if r > self.lp.half {
            csubq(base + self.neg_off, self.lq.q)
        } else {
            base
        }
    }

    fn q_to_p_centred(&self, c: u64) -> u64 {
        if c > self.lq.half {
            c + (self.lp.q - self.lq.q)
        } else {
            c
        }
    }
}

pub fn expand_uniform(rng: &mut ChaCha20Rng, q: u64, out: &mut [u64]) {
    let zone = (u64::MAX / q) * q;
    for c in out {
        loop {
            let u = rng.next_u64();
            if u < zone {
                *c = u % q;
                break;
            }
        }
    }
}

/// A polynomial at `q * p`, one residue vector per limb.
pub struct Rns {
    pub q: Vec<u64>,
    pub p: Vec<u64>,
}

impl Rns {
    pub fn to_ntt(&self, ctx: &PackCtx) -> Self {
        let (mut q, mut p) = (self.q.clone(), self.p.clone());
        ctx.lq.ctx.forward(&mut q);
        ctx.lp.ctx.forward(&mut p);
        Self { q, p }
    }
}

pub fn expand_key_mask(ctx: &PackCtx, seed: [u8; 32]) -> Rns {
    let mut rng = ChaCha20Rng::from_seed(seed);
    let mut q = vec![0u64; ctx.d];
    let mut p = vec![0u64; ctx.d];
    expand_uniform(&mut rng, ctx.lq.q, &mut q);
    expand_uniform(&mut rng, ctx.lp.q, &mut p);
    Rns { q, p }
}

pub fn tau_signed(v: &[i64], t: usize) -> Vec<i64> {
    let d = v.len();
    let mut out = vec![0i64; d];
    for (i, &x) in v.iter().enumerate() {
        let k = (t * i) % (2 * d);
        if k < d {
            out[k] = x;
        } else {
            out[k - d] = -x;
        }
    }
    out
}

pub struct Secret {
    pub s: Vec<i64>,
    sq_ntt: Vec<u64>,
    sp_ntt: Vec<u64>,
}

impl Secret {
    pub fn from_coeffs(ctx: &PackCtx, s: Vec<i64>) -> Self {
        let mut sq: Vec<u64> = s.iter().map(|&x| ctx.lq.residue(x)).collect();
        let mut sp: Vec<u64> = s.iter().map(|&x| ctx.lp.residue(x)).collect();
        ctx.lq.ctx.forward(&mut sq);
        ctx.lp.ctx.forward(&mut sp);
        Self {
            s,
            sq_ntt: sq,
            sp_ntt: sp,
        }
    }
}

/// `y = p * tau_g(s) - s * w + e` at `q * p`, coefficient form.
pub fn keygen(ctx: &PackCtx, sk: &Secret, w_seed: [u8; 32], e: &[i64]) -> Rns {
    let w = expand_key_mask(ctx, w_seed).to_ntt(ctx);
    let ts = tau_signed(&sk.s, ctx.g);
    let mut sw_q: Vec<u64> = (0..ctx.d)
        .map(|k| ctx.lq.mm(sk.sq_ntt[k], w.q[k]))
        .collect();
    ctx.lq.ctx.inverse(&mut sw_q);
    let mut sw_p: Vec<u64> = (0..ctx.d)
        .map(|k| ctx.lp.mm(sk.sp_ntt[k], w.p[k]))
        .collect();
    ctx.lp.ctx.inverse(&mut sw_p);
    let q = ctx.lq.q;
    let yq = (0..ctx.d)
        .map(|k| {
            let lead = (u128::from(ctx.p_mod_q) * u128::from(ctx.lq.residue(ts[k])) % u128::from(q))
                as u64;
            csubq(ctx.lq.sub(lead, sw_q[k]) + ctx.lq.residue(e[k]), q)
        })
        .collect();
    let yp = (0..ctx.d)
        .map(|k| ctx.lp.sub(ctx.lp.residue(e[k]), sw_p[k]))
        .collect();
    Rns { q: yq, p: yp }
}

pub fn expand_query_mask(ctx: &PackCtx, seed: [u8; 32]) -> Vec<u64> {
    let mut rng = ChaCha20Rng::from_seed(seed);
    let mut a = vec![0u64; ctx.d];
    expand_uniform(&mut rng, ctx.lq.q, &mut a);
    a
}

/// `b = -a*s + e + delta * X^(-index)`.
pub fn query(ctx: &PackCtx, sk: &Secret, index: usize, a_seed: [u8; 32], e: &[i64]) -> Vec<u64> {
    let d = ctx.d;
    let q = ctx.lq.q;
    let mut a = expand_query_mask(ctx, a_seed);
    ctx.lq.ctx.forward(&mut a);
    let mut a_s: Vec<u64> = (0..d).map(|k| ctx.lq.mm(a[k], sk.sq_ntt[k])).collect();
    ctx.lq.ctx.inverse(&mut a_s);
    let mut b: Vec<u64> = (0..d)
        .map(|k| ctx.lq.sub(ctx.lq.residue(e[k]), a_s[k]))
        .collect();
    if index == 0 {
        b[0] = csubq(b[0] + ctx.delta, q);
    } else {
        b[d - index] = ctx.lq.sub(b[d - index], ctx.delta);
    }
    b
}

/// Column values, row-major: `vals[row * cols + col]`.
pub struct Shard {
    pub rows: usize,
    pub cols: usize,
    pub vals: Vec<i8>,
}

impl Shard {
    fn column(&self, j: usize, limb: &Limb, d: usize) -> Vec<u64> {
        let mut out = vec![0u64; d];
        for (m, slot) in out.iter_mut().enumerate().take(self.rows) {
            *slot = limb.residue(i64::from(self.vals[m * self.cols + j]));
        }
        out
    }
}

fn step1_row(ctx: &PackCtx, i: usize, cols: &[Vec<u64>]) -> Vec<u64> {
    let d = ctx.d;
    let q = ctx.lq.q;
    let e = ctx.gen_pows[(d - i) % d];
    let mut acc = vec![0u64; d];
    for (j, col) in cols.iter().enumerate() {
        let k = (j * e) % (2 * d);
        let (sh, flip) = if k < d { (k, false) } else { (k - d, true) };
        let (head, tail) = col.split_at(d - sh);
        if flip {
            for (a, &c) in acc[sh..].iter_mut().zip(head) {
                *a += q - c;
            }
            for (a, &c) in acc[..sh].iter_mut().zip(tail) {
                *a += c;
            }
        } else {
            for (a, &c) in acc[sh..].iter_mut().zip(head) {
                *a += c;
            }
            for (a, &c) in acc[..sh].iter_mut().zip(tail) {
                *a += q - c;
            }
        }
    }
    for a in &mut acc {
        *a %= q;
    }
    ctx.lq.ctx.forward(&mut acc);
    let tab = &ctx.tables[i];
    (0..d)
        .map(|k| ctx.lq.mm(acc[tab[k] as usize], ctx.inv_gamma_m))
        .collect()
}

/// Mask and body (coefficient form at `q`) of the response to one query.
pub fn respond(
    ctx: &PackCtx,
    shard: &Shard,
    w_seed: [u8; 32],
    key: &Rns,
    a_seed: [u8; 32],
    body: &[u64],
) -> (Vec<u64>, Vec<u64>) {
    let d = ctx.d;
    let (lq, lp) = (&ctx.lq, &ctx.lp);
    assert_eq!(shard.cols, ctx.gamma);
    let mut a = expand_query_mask(ctx, a_seed);
    lq.ctx.forward(&mut a);
    let prods: Vec<Vec<u64>> = (0..shard.cols)
        .map(|j| {
            let mut h = shard.column(j, lq, d);
            lq.ctx.forward(&mut h);
            for k in 0..d {
                h[k] = lq.mm(h[k], a[k]);
            }
            lq.ctx.inverse(&mut h);
            h
        })
        .collect();
    let mut r: Vec<Vec<u64>> = (0..ctx.gamma).map(|i| step1_row(ctx, i, &prods)).collect();

    let bc: Vec<i64> = body.iter().map(|&x| lq.centred(x)).collect();
    let mut b_poly = vec![0i64; shard.cols];
    for m in 0..shard.rows {
        let cm = if m == 0 { bc[0] } else { -bc[d - m] };
        for (j, acc) in b_poly.iter_mut().enumerate() {
            *acc += cm * i64::from(shard.vals[m * shard.cols + j]);
        }
    }

    let w = expand_key_mask(ctx, w_seed).to_ntt(ctx);
    let key = key.to_ntt(ctx);
    let mut coeff = vec![0u64; d];
    let mut tp = vec![0u64; d];
    let mut cp = vec![0u64; d];
    let mut rq = vec![0u64; d];
    let mut acc_q = vec![0u64; d];
    let mut acc_p = vec![0u64; d];
    for i in (0..ctx.gamma - 1).rev() {
        let tab = &ctx.tables[i];
        let (lo, hi) = r.split_at_mut(i + 1);
        let ri = &mut lo[i];
        let rn = &hi[0];
        coeff.copy_from_slice(rn);
        lq.ctx.inverse(&mut coeff);
        for k in 0..d {
            tp[k] = ctx.q_to_p_centred(coeff[k]);
        }
        lp.ctx.forward(&mut tp);
        for k in 0..d {
            cp[k] = lp.mm(tp[k], w.p[tab[k] as usize]);
        }
        lp.ctx.inverse(&mut cp);
        for k in 0..d {
            rq[k] = ctx.p_to_q_centred(cp[k]);
        }
        lq.ctx.forward(&mut rq);
        for k in 0..d {
            let cq = lq.mm(rn[k], w.q[tab[k] as usize]);
            let dr = lq.mm(lq.sub(cq, rq[k]), ctx.p_inv_m);
            ri[k] = csubq(ri[k] + dr, lq.q);
            let src = tab[k] as usize;
            acc_q[k] = (acc_q[k] + lq.mm(rn[k], key.q[src])) % lq.q;
            acc_p[k] = (acc_p[k] + lp.mm(tp[k], key.p[src])) % lp.q;
        }
    }
    let mut a_hat = r[0].clone();
    lq.ctx.inverse(&mut a_hat);
    lq.ctx.inverse(&mut acc_q);
    lp.ctx.inverse(&mut acc_p);
    let b: Vec<u64> = (0..d)
        .map(|k| {
            let r = ctx.p_to_q_centred(acc_p[k]);
            let down = lq.mm(lq.sub(acc_q[k], r), ctx.p_inv_m);
            if k < shard.cols {
                csubq(down + lq.residue(b_poly[k]), lq.q)
            } else {
                down
            }
        })
        .collect();
    (a_hat, b)
}
