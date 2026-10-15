//! PartialInspiRING with a one-row special-modulus packing key (measurement code).
//!
//! The query, the database multiply and the response live at a small modulus `q`. The
//! packing key is one polynomial at `q * p`; the server multiplies by it at `q * p` and
//! divides by `p` with rounding. Everything is kept per limb (residues mod `q` and mod `p`),
//! so no value is ever composed on the server.
//!
//! NTT-domain vectors are in the Montgomery form `NttContext` produces.

use std::collections::HashMap;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::math::mod_q::DEFAULT_Q;
use crate::math::modular::csubq;
use crate::math::primality::is_prime;
use crate::math::{mod_inverse, GaussianSampler, NttContext};

/// Ciphertext prime and special prime, key modulus near 2^60.
pub const SET_Q29: (u64, u64) = (536_813_569, 2_147_389_441);
/// Ciphertext prime and special prime, key modulus near 2^57.
pub const SET_Q28: (u64, u64) = (268_369_921, 536_813_569);

/// Whether `q` is a prime with a `2d`-th root of unity.
pub fn modulus_is_usable(q: u64, d: usize) -> bool {
    is_prime(q) && q % (2 * d as u64) == 1
}

/// One prime with its transform.
#[derive(Clone)]
pub struct Limb {
    /// The prime.
    pub q: u64,
    /// Transform over the prime.
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

    /// Montgomery product.
    #[inline]
    pub fn mm(&self, a: u64, b: u64) -> u64 {
        let ab = u128::from(a) * u128::from(b);
        let m = u128::from((ab as u64).wrapping_mul(self.qinv));
        csubq(((ab + m * u128::from(self.q)) >> 64) as u64, self.q)
    }

    #[inline]
    fn sub(&self, a: u64, b: u64) -> u64 {
        csubq(a + self.q - b, self.q)
    }

    /// Residue of a signed value.
    #[inline]
    pub fn from_signed(&self, x: i64) -> u64 {
        x.rem_euclid(self.q as i64) as u64
    }

    /// Centred representative in `[-(q-1)/2, (q-1)/2]`.
    #[inline]
    pub fn centred(&self, x: u64) -> i64 {
        if x > self.half {
            x as i64 - self.q as i64
        } else {
            x as i64
        }
    }

    fn fwd_signed(&self, v: &[i64]) -> Vec<u64> {
        let mut out: Vec<u64> = v.iter().map(|&x| self.from_signed(x)).collect();
        self.ctx.forward(&mut out);
        out
    }
}

/// Everything public that both sides derive from `(d, gamma, q, p)`.
pub struct PackCtx {
    /// Ring dimension.
    pub d: usize,
    /// Coefficients packed into one response.
    pub gamma: usize,
    /// Plaintext modulus.
    pub t: u64,
    /// floor(q / t).
    pub delta: u64,
    /// Ciphertext limb.
    pub lq: Limb,
    /// Special limb.
    pub lp: Limb,
    /// Generator of the order-gamma automorphism subgroup.
    pub g: usize,
    /// `g^k mod 2d` for `k < d`.
    pub gen_pows: Vec<usize>,
    /// `tables[i][k]`: source slot of slot `k` under `tau_{g^i}` in the transform domain.
    pub tables: Vec<Vec<u32>>,
    inv_gamma_m: u64,
    p_inv_m: u64,
    p_mod_q: u64,
    neg_off: u64,
}

fn derive_tables(limb: &Limb, d: usize, exps: &[usize]) -> Vec<Vec<u32>> {
    let mut x = vec![0u64; d];
    x[1] = 1;
    limb.ctx.forward(&mut x);
    let slot_of: HashMap<u64, u32> = x.iter().enumerate().map(|(k, &v)| (v, k as u32)).collect();
    assert_eq!(
        slot_of.len(),
        d,
        "transform is not an evaluation at d distinct roots"
    );
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
    /// # Panics
    ///
    /// If a modulus is unusable at `d`, `gamma` is not a power of two in `[2, d/2]`, or the
    /// two limbs disagree on an automorphism table.
    pub fn new(d: usize, gamma: usize, q: u64, p: u64, t: u64) -> Self {
        Self::build(d, gamma, q, p, t, true)
    }

    /// The client's view: no automorphism tables, which only the server reads.
    pub fn client(d: usize, gamma: usize, q: u64, p: u64, t: u64) -> Self {
        Self::build(d, gamma, q, p, t, false)
    }

    fn build(d: usize, gamma: usize, q: u64, p: u64, t: u64, with_tables: bool) -> Self {
        assert!(d.is_power_of_two() && gamma.is_power_of_two() && gamma >= 2 && gamma <= d / 2);
        assert!(modulus_is_usable(q, d) && modulus_is_usable(p, d) && q != p);
        assert!(u128::from(q) * u128::from(p) < 1u128 << 63);
        let lq = Limb::new(d, q);
        let lp = Limb::new(d, p);
        let g = 2 * d / gamma + 1;
        let mut gen_pows = Vec::with_capacity(d);
        let mut v = 1usize;
        for _ in 0..d {
            gen_pows.push(v);
            v = (v * g) % (2 * d);
        }
        let tables = if with_tables {
            let tables = derive_tables(&lq, d, &gen_pows[..gamma]);
            assert!(
                tables == derive_tables(&lp, d, &gen_pows[..gamma]),
                "limbs disagree on the automorphism slot permutation"
            );
            tables
        } else {
            Vec::new()
        };
        let inv_gamma_m = lq.ctx.to_mont(mod_inverse(gamma as u64, q));
        let p_inv_m = lq.ctx.to_mont(mod_inverse(p % q, q));
        Self {
            d,
            gamma,
            t,
            delta: q / t,
            g,
            gen_pows,
            tables,
            inv_gamma_m,
            p_inv_m,
            p_mod_q: p % q,
            neg_off: (q - p % q) % q,
            lq,
            lp,
        }
    }

    /// Residue mod `q` of the centred representative of a residue mod `p`.
    #[inline]
    fn p_to_q_centred(&self, r: u64) -> u64 {
        let base = r % self.lq.q;
        if r > self.lp.half {
            csubq(base + self.neg_off, self.lq.q)
        } else {
            base
        }
    }

    /// Residue mod `p` of the centred representative of a residue mod `q`.
    #[inline]
    fn q_to_p_centred(&self, c: u64) -> u64 {
        if self.lp.q > self.lq.q {
            if c > self.lq.half {
                c + (self.lp.q - self.lq.q)
            } else {
                c
            }
        } else {
            self.lp.from_signed(self.lq.centred(c))
        }
    }
}

/// Exact uniform residues: a 64-bit word is rejected when it falls in the short last
/// interval, so no residue is favoured.
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rns {
    /// Residues mod `q`.
    pub q: Vec<u64>,
    /// Residues mod `p`.
    pub p: Vec<u64>,
}

impl Rns {
    /// Both limbs to the transform domain.
    pub fn to_ntt(&self, ctx: &PackCtx) -> Self {
        let mut out = self.clone();
        ctx.lq.ctx.forward(&mut out.q);
        ctx.lp.ctx.forward(&mut out.p);
        out
    }

    /// Residues of a small signed polynomial.
    pub fn from_signed(ctx: &PackCtx, v: &[i64]) -> Self {
        Self {
            q: v.iter().map(|&x| ctx.lq.from_signed(x)).collect(),
            p: v.iter().map(|&x| ctx.lp.from_signed(x)).collect(),
        }
    }

    /// Centred value of each coefficient, composed over both limbs.
    pub fn centred(&self, ctx: &PackCtx) -> Vec<i128> {
        let (q, p) = (u128::from(ctx.lq.q), u128::from(ctx.lp.q));
        let q_inv_p = u128::from(mod_inverse(ctx.lq.q % ctx.lp.q, ctx.lp.q));
        let qp = q * p;
        self.q
            .iter()
            .zip(&self.p)
            .map(|(&a, &b)| {
                let diff = (u128::from(b) + p - u128::from(a) % p) % p;
                let x = u128::from(a) + q * (diff * q_inv_p % p);
                if x > qp / 2 {
                    x as i128 - qp as i128
                } else {
                    x as i128
                }
            })
            .collect()
    }
}

/// The public key mask, coefficient form, expanded limb by limb from one stream.
pub fn expand_key_mask(ctx: &PackCtx, seed: [u8; 32]) -> Rns {
    let mut rng = ChaCha20Rng::from_seed(seed);
    let mut q = vec![0u64; ctx.d];
    let mut p = vec![0u64; ctx.d];
    expand_uniform(&mut rng, ctx.lq.q, &mut q);
    expand_uniform(&mut rng, ctx.lp.q, &mut p);
    Rns { q, p }
}

/// `f(X^t)` for a small signed polynomial.
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

/// The client secret with its transforms.
pub struct Secret {
    /// Signed coefficients.
    pub s: Vec<i64>,
    sq_ntt: Vec<u64>,
    sp_ntt: Vec<u64>,
}

impl Secret {
    /// Fresh secret from the sampler.
    pub fn sample(ctx: &PackCtx, sampler: &mut GaussianSampler) -> Self {
        Self::from_coeffs(ctx, sampler.sample_vec(ctx.d))
    }

    /// Secret with given coefficients.
    pub fn from_coeffs(ctx: &PackCtx, s: Vec<i64>) -> Self {
        let sq_ntt = ctx.lq.fwd_signed(&s);
        let sp_ntt = ctx.lp.fwd_signed(&s);
        Self { s, sq_ntt, sp_ntt }
    }
}

/// `y = p * tau_g(s) - s * w + e` at `q * p`, coefficient form. Returns the key and its error.
pub fn keygen(
    ctx: &PackCtx,
    sk: &Secret,
    w_seed: [u8; 32],
    sampler: &mut GaussianSampler,
) -> (Rns, Vec<i64>) {
    let w = expand_key_mask(ctx, w_seed).to_ntt(ctx);
    let e = sampler.sample_vec(ctx.d);
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
            let lead = (u128::from(ctx.p_mod_q) * u128::from(ctx.lq.from_signed(ts[k]))
                % u128::from(q)) as u64;
            csubq(ctx.lq.sub(lead, sw_q[k]) + ctx.lq.from_signed(e[k]), q)
        })
        .collect();
    // The leading term vanishes mod p.
    let yp = (0..ctx.d)
        .map(|k| ctx.lp.sub(ctx.lp.from_signed(e[k]), sw_p[k]))
        .collect();
    (Rns { q: yq, p: yp }, e)
}

/// One query: the mask seed and the body at `q`.
#[derive(Clone)]
pub struct Query {
    /// Seed the mask expands from.
    pub a_seed: [u8; 32],
    /// Body coefficients.
    pub b: Vec<u64>,
}

/// The query mask in coefficient form.
pub fn expand_query_mask(ctx: &PackCtx, seed: [u8; 32]) -> Vec<u64> {
    let mut rng = ChaCha20Rng::from_seed(seed);
    let mut a = vec![0u64; ctx.d];
    expand_uniform(&mut rng, ctx.lq.q, &mut a);
    a
}

/// `b = -a*s + e + delta * X^(-index)`. Returns the query and its error.
pub fn query(
    ctx: &PackCtx,
    sk: &Secret,
    index: usize,
    a_seed: [u8; 32],
    sampler: &mut GaussianSampler,
) -> (Query, Vec<i64>) {
    let d = ctx.d;
    let q = ctx.lq.q;
    let mut a = expand_query_mask(ctx, a_seed);
    ctx.lq.ctx.forward(&mut a);
    let mut a_s: Vec<u64> = (0..d).map(|k| ctx.lq.mm(a[k], sk.sq_ntt[k])).collect();
    ctx.lq.ctx.inverse(&mut a_s);
    let e = sampler.sample_vec(d);
    let mut b: Vec<u64> = (0..d)
        .map(|k| ctx.lq.sub(ctx.lq.from_signed(e[k]), a_s[k]))
        .collect();
    if index == 0 {
        b[0] = csubq(b[0] + ctx.delta, q);
    } else {
        b[d - index] = ctx.lq.sub(b[d - index], ctx.delta);
    }
    (Query { a_seed, b }, e)
}

/// Centred column values, row-major: `vals[row * cols + col]`. Rows past `rows` are zero.
#[derive(Clone)]
pub struct Shard {
    /// Selectable positions.
    pub rows: usize,
    /// Columns one position fills.
    pub cols: usize,
    /// Values in `[-t/2, t/2)`.
    pub vals: Vec<i8>,
}

impl Shard {
    /// Every value equal.
    pub fn constant(rows: usize, cols: usize, v: i8) -> Self {
        Self {
            rows,
            cols,
            vals: vec![v; rows * cols],
        }
    }

    /// Rows of bytes, two centred nibbles per byte, low nibble first.
    pub fn from_cells(cells: &[Vec<u8>]) -> Self {
        let cols = 2 * cells[0].len();
        let mut vals = Vec::with_capacity(cells.len() * cols);
        for cell in cells {
            assert_eq!(2 * cell.len(), cols);
            for &byte in cell {
                vals.push((byte & 15) as i8 - 8);
                vals.push((byte >> 4) as i8 - 8);
            }
        }
        Self {
            rows: cells.len(),
            cols,
            vals,
        }
    }

    /// Column `j` as residues mod `q`.
    pub fn column(&self, j: usize, limb: &Limb, d: usize) -> Vec<u64> {
        let mut out = vec![0u64; d];
        for (m, slot) in out.iter_mut().enumerate().take(self.rows) {
            *slot = limb.from_signed(i64::from(self.vals[m * self.cols + j]));
        }
        out
    }
}

/// Bytes from decoded nibbles, low nibble first.
pub fn nibbles_to_bytes(nibbles: &[u8]) -> Vec<u8> {
    nibbles
        .as_chunks::<2>()
        .0
        .iter()
        .map(|n| n[0] | (n[1] << 4))
        .collect()
}

/// A run of packed columns answered by one query.
#[derive(Clone, Copy)]
pub struct Segment<'a> {
    /// Data the columns come from.
    pub shard: &'a Shard,
    /// Index of the query that selects from it.
    pub query: usize,
}

fn gamma_of(segs: &[Segment<'_>]) -> usize {
    segs.iter().map(|s| s.shard.cols).sum()
}

/// `coeff_0(c * h_j)` over the integers for every column of one shard, `c` signed.
pub fn coeff0_columns(shard: &Shard, c: &[i64]) -> Vec<i64> {
    let d = c.len();
    let mut acc = vec![0i64; shard.cols];
    for m in 0..shard.rows {
        // coeff_0(c*h) = c_0 h_0 - sum_{m>=1} c_{d-m} h_m
        let cm = if m == 0 { c[0] } else { -c[d - m] };
        let row = &shard.vals[m * shard.cols..(m + 1) * shard.cols];
        for (a, &v) in acc.iter_mut().zip(row) {
            *a += cm * i64::from(v);
        }
    }
    acc
}

/// `coeff_0(b * h_j) mod q` for every packed column.
pub fn body_values(ctx: &PackCtx, segs: &[Segment<'_>], bodies: &[&[u64]]) -> Vec<u64> {
    let mut out = Vec::with_capacity(ctx.gamma);
    for seg in segs {
        let b: Vec<i64> = bodies[seg.query]
            .iter()
            .map(|&x| ctx.lq.centred(x))
            .collect();
        out.extend(
            coeff0_columns(seg.shard, &b)
                .iter()
                .map(|&x| ctx.lq.from_signed(x)),
        );
    }
    out
}

/// `R_i = tau_{g^i}( gamma^-1 * sum_j X^(j * g^(d-i)) * col_j )` in the transform domain,
/// for columns given in coefficient form with their packed index.
pub fn step1_row(ctx: &PackCtx, i: usize, cols: &[(usize, &[u64])]) -> Vec<u64> {
    let d = ctx.d;
    let q = ctx.lq.q;
    let e = ctx.gen_pows[(d - i) % d];
    let mut acc = vec![0u64; d];
    for &(j, col) in cols {
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

/// Step 1 straight from the query: every column product `a * h_j`, then [`step1_row`].
pub fn step1_direct(ctx: &PackCtx, segs: &[Segment<'_>], masks_ntt: &[Vec<u64>]) -> Vec<Vec<u64>> {
    use crate::par_prelude::*;
    let d = ctx.d;
    let mut prods: Vec<Vec<u64>> = Vec::with_capacity(ctx.gamma);
    for seg in segs {
        let a = &masks_ntt[seg.query];
        for j in 0..seg.shard.cols {
            let mut h = seg.shard.column(j, &ctx.lq, d);
            ctx.lq.ctx.forward(&mut h);
            for k in 0..d {
                h[k] = ctx.lq.mm(h[k], a[k]);
            }
            ctx.lq.ctx.inverse(&mut h);
            prods.push(h);
        }
    }
    let cols: Vec<(usize, &[u64])> = prods.iter().enumerate().map(|(j, c)| (j, &c[..])).collect();
    (0..ctx.gamma)
        .into_par_iter()
        .map(|i| step1_row(ctx, i, &cols))
        .collect()
}

/// Per-segment precompute of step 1 with the query mask factored out:
/// `store[seg][i] = tau_{g^i}( gamma^-1 * sum_{j in seg} X^(j * g^(d-i)) * h_j )`.
pub struct Store {
    /// One flat `gamma * d` table per segment, transform domain.
    pub segs: Vec<Vec<u64>>,
}

impl Store {
    /// Built from the data alone; valid only for exactly these contents and this layout.
    pub fn build(ctx: &PackCtx, segs: &[Segment<'_>]) -> Self {
        use crate::par_prelude::*;
        assert_eq!(gamma_of(segs), ctx.gamma);
        let d = ctx.d;
        let mut off = 0;
        let mut out = Vec::with_capacity(segs.len());
        for seg in segs {
            let polys: Vec<Vec<u64>> = (0..seg.shard.cols)
                .map(|j| seg.shard.column(j, &ctx.lq, d))
                .collect();
            let cols: Vec<(usize, &[u64])> = polys
                .iter()
                .enumerate()
                .map(|(j, c)| (off + j, &c[..]))
                .collect();
            let rows: Vec<Vec<u64>> = (0..ctx.gamma)
                .into_par_iter()
                .map(|i| step1_row(ctx, i, &cols))
                .collect();
            out.push(rows.concat());
            off += seg.shard.cols;
        }
        Self { segs: out }
    }

    /// Bytes each segment's table holds: `gamma * d` words, whatever the segment's width.
    pub fn table_bytes(&self) -> Vec<usize> {
        self.segs.iter().map(|t| size_of_val(&t[..])).collect()
    }
}

/// Step 1 through a [`Store`]: `R_i = sum_seg tau_{g^i}(a_seg) * store[seg][i]`.
pub fn step1_stored(
    ctx: &PackCtx,
    segs: &[Segment<'_>],
    store: &Store,
    masks_ntt: &[Vec<u64>],
) -> Vec<Vec<u64>> {
    use crate::par_prelude::*;
    let d = ctx.d;
    let q = ctx.lq.q;
    (0..ctx.gamma)
        .into_par_iter()
        .map(|i| {
            let tab = &ctx.tables[i];
            let mut r = vec![0u64; d];
            for (seg, table) in segs.iter().zip(&store.segs) {
                let a = &masks_ntt[seg.query];
                let h = &table[i * d..(i + 1) * d];
                for k in 0..d {
                    r[k] = csubq(r[k] + ctx.lq.mm(a[tab[k] as usize], h[k]), q);
                }
            }
            r
        })
        .collect()
}

/// Backward recursion. For `i` from `gamma-2` down to `0`, with `T_i` the centred lift of
/// the finished `R_(i+1)`: `R_i += round(T_i * tau_{g^i}(w) / p)`. For each entry of `keys`
/// it also returns `sum_i T_i * tau_{g^i}(key)` at `q * p` in coefficient form.
///
/// `on_step(i, c)` sees the `p`-limb coefficients of `T_i * tau_{g^i}(w)`; their centred
/// values are the remainders of the division.
pub fn recursion(
    ctx: &PackCtx,
    r: &mut [Vec<u64>],
    w_ntt: &Rns,
    keys: &[&Rns],
    mut on_step: impl FnMut(usize, &[u64]),
) -> (Vec<u64>, Vec<Rns>) {
    let d = ctx.d;
    let (lq, lp) = (&ctx.lq, &ctx.lp);
    assert_eq!(r.len(), ctx.gamma);
    let mut coeff = vec![0u64; d];
    let mut tp = vec![0u64; d];
    let mut cp = vec![0u64; d];
    let mut rq = vec![0u64; d];
    // Each term is below its modulus (< 2^31) and there are fewer than 2^11 of them.
    let mut accs: Vec<Rns> = keys
        .iter()
        .map(|_| Rns {
            q: vec![0u64; d],
            p: vec![0u64; d],
        })
        .collect();

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
            cp[k] = lp.mm(tp[k], w_ntt.p[tab[k] as usize]);
        }
        lp.ctx.inverse(&mut cp);
        on_step(i, &cp);
        for k in 0..d {
            rq[k] = ctx.p_to_q_centred(cp[k]);
        }
        lq.ctx.forward(&mut rq);

        for k in 0..d {
            let cq = lq.mm(rn[k], w_ntt.q[tab[k] as usize]);
            let dr = lq.mm(lq.sub(cq, rq[k]), ctx.p_inv_m);
            ri[k] = csubq(ri[k] + dr, lq.q);
        }

        for (key, acc) in keys.iter().zip(&mut accs) {
            for k in 0..d {
                let src = tab[k] as usize;
                acc.q[k] += lq.mm(rn[k], key.q[src]);
                acc.p[k] += lp.mm(tp[k], key.p[src]);
            }
        }
    }

    let mut a_hat = r[0].clone();
    lq.ctx.inverse(&mut a_hat);
    for acc in &mut accs {
        for v in &mut acc.q {
            *v %= lq.q;
        }
        for v in &mut acc.p {
            *v %= lp.q;
        }
        lq.ctx.inverse(&mut acc.q);
        lp.ctx.inverse(&mut acc.p);
    }
    (a_hat, accs)
}

/// `round(x / p) mod q` for `x` at `q * p` in coefficient form.
pub fn mod_down(ctx: &PackCtx, x: &Rns) -> Vec<u64> {
    (0..ctx.d)
        .map(|k| {
            let r = ctx.p_to_q_centred(x.p[k]);
            ctx.lq.mm(ctx.lq.sub(x.q[k], r), ctx.p_inv_m)
        })
        .collect()
}

/// A packed response: the mask and the body, coefficient form at `q`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packed {
    /// Mask.
    pub a: Vec<u64>,
    /// Body; coefficients past `gamma` carry no data.
    pub b: Vec<u64>,
}

/// Which form of step 1 a response uses.
pub enum Step1<'a> {
    /// Column products per query.
    Direct,
    /// The per-data precompute.
    Stored(&'a Store),
}

/// The server's whole answer to the queries of one pack.
pub fn respond(
    ctx: &PackCtx,
    segs: &[Segment<'_>],
    step1: &Step1<'_>,
    w_ntt: &Rns,
    key_ntt: &Rns,
    queries: &[&Query],
) -> Packed {
    assert_eq!(gamma_of(segs), ctx.gamma);
    let masks_ntt: Vec<Vec<u64>> = queries
        .iter()
        .map(|qr| {
            let mut a = expand_query_mask(ctx, qr.a_seed);
            ctx.lq.ctx.forward(&mut a);
            a
        })
        .collect();
    let mut r = match step1 {
        Step1::Direct => step1_direct(ctx, segs, &masks_ntt),
        Step1::Stored(store) => step1_stored(ctx, segs, store, &masks_ntt),
    };
    let bodies: Vec<&[u64]> = queries.iter().map(|qr| &qr.b[..]).collect();
    let b_poly = body_values(ctx, segs, &bodies);
    let (a, accs) = recursion(ctx, &mut r, w_ntt, &[key_ntt], |_, _| {});
    let mut b = mod_down(ctx, &accs[0]);
    for (slot, &v) in b.iter_mut().zip(&b_poly) {
        *slot = csubq(*slot + v, ctx.lq.q);
    }
    Packed { a, b }
}

/// `a * s + b` at `q`.
pub fn phase(ctx: &PackCtx, sk: &Secret, packed: &Packed) -> Vec<u64> {
    let mut a = packed.a.clone();
    ctx.lq.ctx.forward(&mut a);
    for (x, &s) in a.iter_mut().zip(&sk.sq_ntt) {
        *x = ctx.lq.mm(*x, s);
    }
    ctx.lq.ctx.inverse(&mut a);
    for (x, &b) in a.iter_mut().zip(&packed.b) {
        *x = csubq(*x + b, ctx.lq.q);
    }
    a
}

/// The first `n` plaintext nibbles of a phase: `floor((x + delta/2) / delta) mod t`, recentred.
pub fn decode_nibbles(ctx: &PackCtx, phase: &[u64], n: usize) -> Vec<u8> {
    phase[..n]
        .iter()
        .map(|&x| ((((x + ctx.delta / 2) / ctx.delta) % ctx.t + ctx.t / 2) % ctx.t) as u8)
        .collect()
}

/// The wire bytes of a response: the full mask and the first `body` coefficients, each at
/// the bit width of `q`.
pub fn response_bytes(ctx: &PackCtx, packed: &Packed, body: usize) -> Vec<u8> {
    let q = [ctx.lq.q];
    let mut out =
        crate::math::poly::pack_coefficients_tight(&packed.a, &q, ctx.d).unwrap_or_default();
    out.extend(
        crate::math::poly::pack_coefficients_tight(&packed.b[..body], &q, body).unwrap_or_default(),
    );
    out
}

/// Variance of the sampler's tail-cut distribution.
pub fn sampler_variance(sigma: f64) -> f64 {
    let t = (sigma * 6.0).ceil() as i64;
    let (mut w, mut m2) = (0f64, 0f64);
    for x in -t..=t {
        let p = (-(x * x) as f64 / (2.0 * sigma * sigma)).exp();
        w += p;
        m2 += p * (x * x) as f64;
    }
    m2 / w
}

/// Exact integer side products for the error decomposition of a response.
pub struct NoiseProbe {
    big: NttContext,
}

/// `tau_{g^i}(s)` for every recursion step, transformed at the probe modulus.
pub struct SecretProbe {
    tau_s: Vec<Vec<u64>>,
}

/// One response with the exact error of every packed coefficient, split into its parts.
pub struct Measured {
    /// The response.
    pub packed: Packed,
    /// `phase - delta * value`, centred.
    pub err: Vec<i64>,
    /// Query error folded through the column: `coeff_0(e_q * h_j)`.
    pub fold: Vec<i64>,
    /// Key error before the division: `sum_i T_i * tau_{g^i}(e_k)`.
    pub key: Vec<i128>,
    /// Division remainders against the secret: `sum_i tau_{g^i}(s) * r_i`.
    pub rem: Vec<i128>,
    /// Coefficients where `err - fold != round((key - rem) / p)`.
    pub identity_misses: usize,
    /// Coefficients that decode to the wrong value.
    pub decode_misses: usize,
}

impl NoiseProbe {
    /// # Panics
    ///
    /// If the remainder sum could wrap the probe modulus.
    pub fn new(ctx: &PackCtx) -> Self {
        // |sum| <= gamma * d * tailcut * p/2 must stay below half the probe modulus.
        assert!(
            (ctx.gamma as u128) * (ctx.d as u128) * 64 * u128::from(ctx.lp.q)
                < u128::from(DEFAULT_Q)
        );
        Self {
            big: NttContext::new(ctx.d, DEFAULT_Q),
        }
    }

    /// Per-secret tables.
    pub fn secret(&self, ctx: &PackCtx, sk: &Secret) -> SecretProbe {
        let tau_s = (0..ctx.gamma - 1)
            .map(|i| {
                let mut v: Vec<u64> = tau_signed(&sk.s, ctx.gen_pows[i])
                    .iter()
                    .map(|&x| x.rem_euclid(DEFAULT_Q as i64) as u64)
                    .collect();
                self.big.forward(&mut v);
                v
            })
            .collect();
        SecretProbe { tau_s }
    }

    /// Runs one pack and returns the response with its exact error parts.
    ///
    /// `queries[k]` is the query, its error and the row it selects.
    pub fn measure(
        &self,
        ctx: &PackCtx,
        segs: &[Segment<'_>],
        step1: &Step1<'_>,
        sk: &Secret,
        sp: &SecretProbe,
        w_ntt: &Rns,
        key_ntt: &Rns,
        key_err: &[i64],
        queries: &[(&Query, &[i64], usize)],
    ) -> Measured {
        let d = ctx.d;
        let q = ctx.lq.q;
        let masks_ntt: Vec<Vec<u64>> = queries
            .iter()
            .map(|(qr, _, _)| {
                let mut a = expand_query_mask(ctx, qr.a_seed);
                ctx.lq.ctx.forward(&mut a);
                a
            })
            .collect();
        let mut r = match step1 {
            Step1::Direct => step1_direct(ctx, segs, &masks_ntt),
            Step1::Stored(store) => step1_stored(ctx, segs, store, &masks_ntt),
        };
        let bodies: Vec<&[u64]> = queries.iter().map(|(qr, _, _)| &qr.b[..]).collect();
        let b_poly = body_values(ctx, segs, &bodies);
        let err_ntt = Rns::from_signed(ctx, key_err).to_ntt(ctx);

        let mut rem_acc = vec![0u64; d];
        let mut scratch = vec![0u64; d];
        let mut prod = vec![0u64; d];
        let (a, accs) = recursion(ctx, &mut r, w_ntt, &[key_ntt, &err_ntt], |i, cp| {
            for k in 0..d {
                scratch[k] = ctx.lp.centred(cp[k]).rem_euclid(DEFAULT_Q as i64) as u64;
            }
            self.big.forward(&mut scratch);
            self.big.pointwise_mul(&scratch, &sp.tau_s[i], &mut prod);
            for k in 0..d {
                rem_acc[k] = csubq(rem_acc[k] + prod[k], DEFAULT_Q);
            }
        });
        self.big.inverse(&mut rem_acc);
        let rem: Vec<i128> = rem_acc[..ctx.gamma]
            .iter()
            .map(|&x| {
                if x > DEFAULT_Q / 2 {
                    i128::from(x) - i128::from(DEFAULT_Q)
                } else {
                    i128::from(x)
                }
            })
            .collect();
        let key: Vec<i128> = accs[1].centred(ctx)[..ctx.gamma].to_vec();

        let mut b = mod_down(ctx, &accs[0]);
        for (slot, &v) in b.iter_mut().zip(&b_poly) {
            *slot = csubq(*slot + v, q);
        }
        let packed = Packed { a, b };
        let ph = phase(ctx, sk, &packed);
        let decoded = decode_nibbles(ctx, &ph, ctx.gamma);

        let p = i128::from(ctx.lp.q);
        let mut err = Vec::with_capacity(ctx.gamma);
        let mut fold = Vec::with_capacity(ctx.gamma);
        let (mut identity_misses, mut decode_misses) = (0, 0);
        let mut off = 0;
        for seg in segs {
            let (_, e_q, index) = queries[seg.query];
            let sh = seg.shard;
            let folds = coeff0_columns(sh, e_q);
            for j in 0..sh.cols {
                let v = i64::from(sh.vals[index * sh.cols + j]);
                let want = (u128::from(ctx.delta) * u128::from(ctx.lq.from_signed(v))
                    % u128::from(q)) as u64;
                let e = ctx.lq.centred(ctx.lq.sub(ph[off + j], want));
                let pack = ctx.lq.centred(ctx.lq.from_signed(e - folds[j]));
                let expect = (key[off + j] - rem[off + j] + p / 2).div_euclid(p);
                if i128::from(pack) != expect {
                    identity_misses += 1;
                }
                if i64::from(decoded[off + j]) != v + (ctx.t / 2) as i64 {
                    decode_misses += 1;
                }
                err.push(e);
                fold.push(folds[j]);
            }
            off += sh.cols;
        }
        Measured {
            packed,
            err,
            fold,
            key,
            rem,
            identity_misses,
            decode_misses,
        }
    }
}
