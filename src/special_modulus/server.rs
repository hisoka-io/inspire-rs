//! The server: per-shard step-1 tables, seated keys, and the packing of one response.
//!
//! Step 1 of the packing is linear in the query mask `a`, so a shard's columns are folded
//! once into `S_i = tau_{g^i}(gamma^-1 * sum_j X^(j * g^(d-i)) * h_j)` and a response
//! needs only `R_i = tau_{g^i}(a) * S_i`. The backward recursion then switches each
//! `R_(i+1)` with the public key mask at `q * p` and divides by `p`; the client's key enters
//! only the body, accumulated over both limbs and divided by `p` once at the end.

use std::sync::Arc;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::par_prelude::*;
use crate::pir::error::{pir_err, Result};

use super::wire::{PackingKey, Query, Response};
use super::{expand_uniform, nibbles, Params, Ring};

/// Which compiled form of the 32-bit kernels runs. Both give the same bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// Plain loops for the baseline target.
    Portable,
    /// The same loops compiled for AVX-512; needs the `simd-packing-offline` feature and a
    /// host with AVX-512 F, BW, DQ and VL.
    Avx512,
}

impl Kernel {
    /// The fastest kernel this build and host can run.
    pub fn best() -> Self {
        if Self::Avx512.available() {
            Self::Avx512
        } else {
            Self::Portable
        }
    }

    /// Whether this build and host can run the kernel.
    pub fn available(self) -> bool {
        match self {
            Self::Portable => true,
            Self::Avx512 => avx512_available(),
        }
    }
}

#[cfg(all(feature = "simd-packing-offline", target_arch = "x86_64"))]
fn avx512_available() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512vl")
    })
}

#[cfg(not(all(feature = "simd-packing-offline", target_arch = "x86_64")))]
fn avx512_available() -> bool {
    false
}

/// A client's key as the server multiplies by it: both limbs in transform form.
#[derive(Clone)]
pub struct KeySeat {
    params: Params,
    q: Vec<u32>,
    p: Vec<u32>,
}

impl std::fmt::Debug for KeySeat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeySeat")
            .field("bytes", &self.bytes())
            .finish_non_exhaustive()
    }
}

impl KeySeat {
    /// Words held, in bytes.
    pub fn bytes(&self) -> usize {
        (self.q.len() + self.p.len()) * 4
    }
}

/// One shard's step-1 table and the cells it was folded from.
///
/// A response reads both from here, so it is always computed from the cells the table
/// was built from; a changed shard needs a new table.
#[derive(Clone)]
pub struct ShardTable {
    params: Params,
    /// `S_i` for `i < gamma`, transform form, Montgomery operands; `gamma * d` words.
    step1: Vec<u32>,
    /// Signed nibbles, row-major: row `m`, column `j` at `m * gamma + j`. Rows past the
    /// cells given are zero.
    values: Vec<i8>,
    rows: usize,
}

impl std::fmt::Debug for ShardTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardTable")
            .field("rows", &self.rows)
            .field("bytes", &self.bytes())
            .finish_non_exhaustive()
    }
}

impl ShardTable {
    /// Cells the table was built from.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Bytes held: the step-1 words and one byte per nibble.
    pub fn bytes(&self) -> usize {
        self.step1.len() * 4 + self.values.len()
    }
}

/// Answers queries for one set.
pub struct Server {
    ring: Arc<Ring>,
    kernel: Kernel,
}

impl std::fmt::Debug for Server {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Server")
            .field("ring_dim", &self.ring.params.ring_dim)
            .field("kernel", &self.kernel)
            .finish_non_exhaustive()
    }
}

impl Server {
    /// Public tables for `params`, on the fastest kernel available.
    pub fn new(params: &Params) -> Result<Self> {
        Self::with_kernel(params, Kernel::best())
    }

    /// Public tables for `params` on a named kernel.
    ///
    /// # Errors
    ///
    /// Refuses an invalid set or a kernel this build or host cannot run.
    pub fn with_kernel(params: &Params, kernel: Kernel) -> Result<Self> {
        if !kernel.available() {
            return Err(pir_err!(
                "kernel {kernel:?} is not available in this build or on this host"
            ));
        }
        Ok(Self {
            ring: Arc::new(Ring::new(params, true)?),
            kernel,
        })
    }

    /// The set served.
    pub fn params(&self) -> &Params {
        &self.ring.params
    }

    /// The kernel in use.
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// Seat an uploaded key.
    pub fn seat(&self, key: &PackingKey) -> Result<KeySeat> {
        let ring = &self.ring;
        let d = ring.params.ring_dim;
        if key.q.len() != d || key.p.len() != d {
            return Err(pir_err!("packing key needs {d} coefficients per limb"));
        }
        if key.q.iter().any(|&x| x >= ring.lq.p.q) || key.p.iter().any(|&x| x >= ring.lp.p.q) {
            return Err(pir_err!("packing key coefficient is not a residue"));
        }
        let (mut q, mut p) = (key.q.clone(), key.p.clone());
        ring.lq.forward(&mut q);
        ring.lp.forward(&mut p);
        for x in &mut q {
            *x = ring.lq.p.to_mont(*x);
        }
        for x in &mut p {
            *x = ring.lp.p.to_mont(*x);
        }
        Ok(KeySeat {
            params: ring.params.clone(),
            q,
            p,
        })
    }

    /// Fold one shard's cells, `cell_bytes` each, row 0 first; at most `ring_dim` cells.
    ///
    /// # Errors
    ///
    /// Refuses a ragged or oversized shard, and a cell with a nonzero byte past the
    /// record, which no response could return.
    pub fn table(&self, cells: &[u8]) -> Result<ShardTable> {
        let params = &self.ring.params;
        let (d, gamma, cell) = (params.ring_dim, params.gamma, params.cell_bytes);
        if !cells.len().is_multiple_of(cell) || cells.len() / cell > d {
            return Err(pir_err!(
                "{} bytes are not at most {d} cells of {cell} bytes",
                cells.len()
            ));
        }
        let rows = cells.len() / cell;
        let mut values = vec![0i8; d * gamma];
        for (m, bytes) in cells.chunks_exact(cell).enumerate() {
            if bytes[params.record_bytes..].iter().any(|&b| b != 0) {
                return Err(pir_err!(
                    "cell {m} has bytes past its {}-byte record",
                    params.record_bytes
                ));
            }
            nibbles(bytes, &mut values[m * gamma..(m + 1) * gamma]);
        }
        let mut columns = vec![0i8; gamma * d];
        for m in 0..rows {
            for j in 0..gamma {
                columns[j * d + m] = values[m * gamma + j];
            }
        }
        let ring = &self.ring;
        let kernel = self.kernel;
        let step1: Vec<Vec<u32>> = (0..gamma)
            .into_par_iter()
            .map(|i| step1_row(kernel, ring, i, &columns))
            .collect();
        Ok(ShardTable {
            params: params.clone(),
            step1: step1.concat(),
            values,
            rows,
        })
    }

    /// Answer one query from one shard under one seated key.
    pub fn respond(&self, table: &ShardTable, seat: &KeySeat, query: &Query) -> Result<Response> {
        let params = &self.ring.params;
        if table.params != *params || seat.params != *params {
            return Err(pir_err!(
                "table or key seat belongs to another parameter set"
            ));
        }
        if query.body.len() != params.ring_dim || query.body.iter().any(|&x| x >= self.ring.lq.p.q)
        {
            return Err(pir_err!("query does not fit the set"));
        }
        let mut mask = vec![0u32; params.ring_dim];
        let mut body = vec![0u32; params.body_coefficients()];
        match self.kernel {
            Kernel::Portable => {
                respond_portable(&self.ring, table, seat, query, &mut mask, &mut body);
            }
            #[cfg(all(feature = "simd-packing-offline", target_arch = "x86_64"))]
            // SAFETY: `with_kernel` admits this kernel only after CPUID reported AVX-512 F,
            // BW, DQ and VL, which is everything the function is compiled for.
            Kernel::Avx512 => unsafe {
                respond_avx512(&self.ring, table, seat, query, &mut mask, &mut body);
            },
            #[cfg(not(all(feature = "simd-packing-offline", target_arch = "x86_64")))]
            Kernel::Avx512 => return Err(pir_err!("kernel Avx512 is not built")),
        }
        Ok(Response { mask, body })
    }
}

fn step1_row(kernel: Kernel, ring: &Ring, i: usize, columns: &[i8]) -> Vec<u32> {
    let mut out = vec![0u32; ring.params.ring_dim];
    match kernel {
        Kernel::Portable => step1_row_portable(ring, i, columns, &mut out),
        #[cfg(all(feature = "simd-packing-offline", target_arch = "x86_64"))]
        // SAFETY: as in `Server::respond`; the kernel is admitted only on a capable host.
        Kernel::Avx512 => unsafe { step1_row_avx512(ring, i, columns, &mut out) },
        #[cfg(not(all(feature = "simd-packing-offline", target_arch = "x86_64")))]
        Kernel::Avx512 => step1_row_portable(ring, i, columns, &mut out),
    }
    out
}

#[inline(never)]
fn step1_row_portable(ring: &Ring, i: usize, columns: &[i8], out: &mut [u32]) {
    step1_row_impl(ring, i, columns, out);
}

#[cfg(all(feature = "simd-packing-offline", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
fn step1_row_avx512(ring: &Ring, i: usize, columns: &[i8], out: &mut [u32]) {
    step1_row_impl(ring, i, columns, out);
}

#[inline(never)]
fn respond_portable(
    ring: &Ring,
    table: &ShardTable,
    seat: &KeySeat,
    query: &Query,
    mask: &mut [u32],
    body: &mut [u32],
) {
    respond_impl(ring, table, seat, query, mask, body);
}

#[cfg(all(feature = "simd-packing-offline", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
fn respond_avx512(
    ring: &Ring,
    table: &ShardTable,
    seat: &KeySeat,
    query: &Query,
    mask: &mut [u32],
    body: &mut [u32],
) {
    respond_impl(ring, table, seat, query, mask, body);
}

/// `S_i` as Montgomery operands: `tau_{g^i}(gamma^-1 * sum_j X^(j * g^(d-i)) * h_j)`.
#[inline(always)]
fn step1_row_impl(ring: &Ring, i: usize, columns: &[i8], out: &mut [u32]) {
    let params = &ring.params;
    let (d, gamma) = (params.ring_dim, params.gamma);
    let two_d = 2 * d;
    // g^(d-i) mod 2d.
    let e = {
        let (mut acc, mut base, mut k) = (1usize, ring.g % two_d, (d - i) % d);
        while k > 0 {
            if k & 1 == 1 {
                acc = acc * base % two_d;
            }
            base = base * base % two_d;
            k >>= 1;
        }
        acc
    };
    // Each column adds at most 8 in magnitude per slot, so gamma <= 2^15 columns fit i32.
    let mut acc = vec![0i32; d];
    for j in 0..gamma {
        let col = &columns[j * d..(j + 1) * d];
        let k = (j * e) % two_d;
        let (sh, flip) = if k < d { (k, false) } else { (k - d, true) };
        let (head, tail) = col.split_at(d - sh);
        let (lo, hi) = acc.split_at_mut(sh);
        if flip {
            for (a, &c) in hi.iter_mut().zip(head) {
                *a -= i32::from(c);
            }
            for (a, &c) in lo.iter_mut().zip(tail) {
                *a += i32::from(c);
            }
        } else {
            for (a, &c) in hi.iter_mut().zip(head) {
                *a += i32::from(c);
            }
            for (a, &c) in lo.iter_mut().zip(tail) {
                *a -= i32::from(c);
            }
        }
    }
    let lq = &ring.lq;
    let mut poly: Vec<u32> = acc.iter().map(|&x| lq.p.small(i64::from(x))).collect();
    lq.forward(&mut poly);
    let inv_gamma =
        lq.p.to_mont(lq.p.to_mont(lq.p.pow(gamma as u64, params.q - 2)));
    // Masked indices into a slice of length d need no bounds check, so the gather vectorises.
    let (poly, m) = (&poly[..d], d - 1);
    for (o, &s) in out.iter_mut().zip(&ring.tables[i]) {
        *o = lq.p.mont(poly[s as usize & m], inv_gamma);
    }
}

#[inline(always)]
fn respond_impl(
    ring: &Ring,
    table: &ShardTable,
    seat: &KeySeat,
    query: &Query,
    mask: &mut [u32],
    body: &mut [u32],
) {
    let params = &ring.params;
    let (d, gamma) = (params.ring_dim, params.gamma);
    let (lq, lp) = (&ring.lq, &ring.lp);
    let (pq, pp) = (&lq.p, &lp.p);

    // The body: coefficient 0 of b * h_j for each column a response carries. Rows past the
    // table's cells are zero.
    let nb = body.len();
    let mut fold = vec![0i64; nb];
    for m in 0..table.rows {
        let c = if m == 0 {
            pq.centred(query.body[0])
        } else {
            -pq.centred(query.body[d - m])
        };
        let row = &table.values[m * gamma..m * gamma + nb];
        for (f, &v) in fold.iter_mut().zip(row) {
            *f += c * i64::from(v);
        }
    }

    let mut a = vec![0u32; d];
    expand_uniform(&mut ChaCha20Rng::from_seed(query.seed), params.q, &mut a);
    lq.forward(&mut a);

    // Masked indices into slices of length d need no bounds check, so the gathers vectorise.
    let m = d - 1;
    let a = &a[..d];
    let (w_q, w_p) = (&ring.w_q[..d], &ring.w_p[..d]);
    let (key_q, key_p) = (&seat.q[..d], &seat.p[..d]);
    let start = |i: usize, out: &mut [u32]| {
        let s = &table.step1[i * d..(i + 1) * d];
        for ((o, &t), &h) in out.iter_mut().zip(&ring.tables[i]).zip(s) {
            *o = pq.mont(a[t as usize & m], h);
        }
    };

    let mut cur = vec![0u32; d];
    let mut next = vec![0u32; d];
    let mut tp = vec![0u32; d];
    let mut cp = vec![0u32; d];
    let mut rq = vec![0u32; d];
    let mut acc_q = vec![0u32; d];
    let mut acc_p = vec![0u32; d];
    start(gamma - 1, &mut cur);
    for i in (0..gamma - 1).rev() {
        let tab = &ring.tables[i][..d];
        tp.copy_from_slice(&cur);
        lq.inverse(&mut tp);
        for x in &mut tp {
            *x = ring.q_to_p_centred(*x);
        }
        lp.forward(&mut tp);
        // Fixed 16-word blocks: the compiler vectorises a known-width body, gathers included.
        for ((c, x), t) in cp
            .as_chunks_mut::<16>()
            .0
            .iter_mut()
            .zip(tp.as_chunks::<16>().0)
            .zip(tab.as_chunks::<16>().0)
        {
            for j in 0..16 {
                c[j] = pp.mont(x[j], w_p[t[j] as usize & m]);
            }
        }
        lp.inverse(&mut cp);
        for (r, &c) in rq.iter_mut().zip(&cp) {
            *r = ring.p_to_q_centred(c);
        }
        lq.forward(&mut rq);
        // R_i = tau_{g^i}(a) * S_i + (R_(i+1) * tau_{g^i}(w) - r) / p, and the key terms.
        let s1 = &table.step1[i * d..(i + 1) * d];
        for (((((((n, c), r), t), aq), ap), x), h) in next
            .as_chunks_mut::<16>()
            .0
            .iter_mut()
            .zip(cur.as_chunks::<16>().0)
            .zip(rq.as_chunks::<16>().0)
            .zip(tab.as_chunks::<16>().0)
            .zip(acc_q.as_chunks_mut::<16>().0.iter_mut())
            .zip(acc_p.as_chunks_mut::<16>().0.iter_mut())
            .zip(tp.as_chunks::<16>().0)
            .zip(s1.as_chunks::<16>().0)
        {
            for j in 0..16 {
                let src = t[j] as usize & m;
                let cq = pq.mont(c[j], w_q[src]);
                n[j] = pq.add(pq.mont(a[src], h[j]), pq.mont(pq.sub(cq, r[j]), ring.p_inv));
                aq[j] = pq.add(aq[j], pq.mont(c[j], key_q[src]));
                ap[j] = pp.add(ap[j], pp.mont(x[j], key_p[src]));
            }
        }
        std::mem::swap(&mut cur, &mut next);
    }
    mask.copy_from_slice(&cur);
    lq.inverse(mask);

    lq.inverse(&mut acc_q);
    lp.inverse(&mut acc_p);
    for (k, out) in body.iter_mut().enumerate() {
        *out = pq.add(
            ring.mod_down(acc_q[k], acc_p[k]),
            pq.residue_public(fold[k]),
        );
    }
}
