//! Private reads packed with a one-row packing key at a two-prime modulus.
//!
//! A second parameter set beside the gadget-key scheme of [`crate::pir`]. The query, the
//! database multiply and the response live at a prime `q` below 2^29; the packing key is
//! one polynomial at `q * p` for a special prime `p` used only inside the key. The server
//! multiplies by the key at `q * p` and divides by `p` with rounding, per limb, so no value
//! is ever composed. Records are read as 4-bit columns (plaintext modulus 16) holding
//! signed nibbles in `[-8, 8)`, so a zero byte adds nothing to the folded query error.
//!
//! Roles:
//! - [`Client`] draws its secret inside, builds one [`PackingKey`], emits one [`Query`] per
//!   row read and decodes each [`Response`] to the record.
//! - [`Server`] holds the public tables. [`Server::table`] folds step 1 of the packing over
//!   one shard's cells into a [`ShardTable`] that keeps those cells, so a response is always
//!   computed from the cells its table was built from. [`Server::seat`] turns an uploaded
//!   key into a [`KeySeat`]. [`Server::respond`] answers one query.
//!
//! All arithmetic runs on 32-bit words; [`Kernel`] picks the portable loops or the same
//! loops compiled for AVX-512, and both give the same bytes.

// A kernel compiles for the vector unit only if it is inlined into its `#[target_feature]`
// caller; a plain `#[inline]` leaves that to the optimiser.
#![allow(clippy::inline_always)]

mod arith;
mod client;
mod server;
mod wire;

use crate::pir::error::{pir_err, Result};

use arith::Ntt32;

pub use client::Client;
pub use server::{Kernel, KeySeat, Server, ShardTable};
pub use wire::{PackingKey, Query, Response};

/// Plaintext modulus: one signed nibble per coefficient.
pub const PLAINTEXT_MODULUS: u64 = 16;

/// Bytes of the seed a query's mask expands from.
pub const SEED_BYTES: usize = 32;

/// The served set's ciphertext prime: the largest prime `1 mod 4096` below 2^29.
pub const Q: u64 = 536_813_569;

/// The served set's special prime: the largest prime `1 mod 4096` below 2^31.
pub const SPECIAL_PRIME: u64 = 2_147_389_441;

/// The served set's public key-mask seed. Fixed in code, so a server cannot hand
/// different clients different masks and tell their keys apart.
pub const KEY_MASK_SEED: [u8; SEED_BYTES] = *b"special-modulus packing key mask";

/// One parameter set.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Ring dimension; also the rows one query selects among.
    pub ring_dim: usize,
    /// Coefficients packed into one response: two per byte of a cell.
    pub gamma: usize,
    /// Ciphertext prime.
    pub q: u64,
    /// Prime that exists only inside the packing key.
    pub special_prime: u64,
    /// Bytes of one row as stored.
    pub cell_bytes: usize,
    /// Leading bytes of a cell a response returns; the rest of the cell must be zero.
    pub record_bytes: usize,
    /// Standard deviation of the secret, the key error and the query error.
    pub sigma: f64,
    /// Seed of the public key mask.
    pub key_mask_seed: [u8; SEED_BYTES],
}

impl Params {
    /// The served set: ring 2,048, 256-byte cells returning 246-byte records.
    #[must_use]
    pub const fn ring2048_cell256() -> Self {
        Self {
            ring_dim: 2048,
            gamma: 512,
            q: Q,
            special_prime: SPECIAL_PRIME,
            cell_bytes: 256,
            record_bytes: 246,
            sigma: 6.4,
            key_mask_seed: KEY_MASK_SEED,
        }
    }

    /// Refuses a set the arithmetic or the packing cannot run.
    pub fn validate(&self) -> Result<()> {
        let (d, gamma) = (self.ring_dim, self.gamma);
        if !d.is_power_of_two() || !(16..=1 << 16).contains(&d) {
            return Err(pir_err!(
                "ring dimension {d} is not a power of two in [16, 2^16]"
            ));
        }
        if !gamma.is_power_of_two() || gamma < 2 || gamma > d / 2 {
            return Err(pir_err!(
                "packed width {gamma} is not a power of two in [2, d/2]"
            ));
        }
        if self.cell_bytes * 2 != gamma {
            return Err(pir_err!(
                "a {}-byte cell fills {} nibbles, not the packed width {gamma}",
                self.cell_bytes,
                self.cell_bytes * 2
            ));
        }
        if self.record_bytes == 0 || self.record_bytes > self.cell_bytes {
            return Err(pir_err!(
                "record of {} bytes does not fit a {}-byte cell",
                self.record_bytes,
                self.cell_bytes
            ));
        }
        let (q, p) = (self.q, self.special_prime);
        for m in [q, p] {
            if m >= 1 << 31
                || !crate::math::primality::is_prime(m)
                || !(m - 1).is_multiple_of(2 * d as u64)
            {
                return Err(pir_err!(
                    "{m} is not a prime below 2^31 that is 1 mod {}",
                    2 * d
                ));
            }
        }
        // The division by p rounds; a special prime below the ciphertext prime would keep
        // more key noise than the response can hold.
        if p <= q {
            return Err(pir_err!(
                "special prime {p} is not above the ciphertext prime {q}"
            ));
        }
        if q / PLAINTEXT_MODULUS < 1 << 16 {
            return Err(pir_err!("ciphertext prime {q} leaves no room for noise"));
        }
        if !(self.sigma.is_finite() && (1.0..=64.0).contains(&self.sigma)) {
            return Err(pir_err!(
                "standard deviation {} is outside [1, 64]",
                self.sigma
            ));
        }
        Ok(())
    }

    fn bits(m: u64) -> usize {
        (u64::BITS - (m - 1).leading_zeros()) as usize
    }

    fn packed_bytes(count: usize, m: u64) -> usize {
        (count * Self::bits(m)).div_ceil(8)
    }

    /// Body coefficients a response carries: two per record byte.
    #[must_use]
    pub const fn body_coefficients(&self) -> usize {
        2 * self.record_bytes
    }

    /// Bytes of an uploaded packing key: each limb at its own prime's width.
    #[must_use]
    pub fn key_bytes(&self) -> usize {
        Self::packed_bytes(self.ring_dim, self.q)
            + Self::packed_bytes(self.ring_dim, self.special_prime)
    }

    /// Bytes of a query: the mask seed and the body.
    #[must_use]
    pub fn query_bytes(&self) -> usize {
        SEED_BYTES + Self::packed_bytes(self.ring_dim, self.q)
    }

    /// Bytes of a response: the whole mask and the record's body coefficients.
    #[must_use]
    pub fn response_bytes(&self) -> usize {
        Self::packed_bytes(self.ring_dim, self.q)
            + Self::packed_bytes(self.body_coefficients(), self.q)
    }

    /// Bytes a server keeps per seated key: both limbs in transform form, one word each.
    #[must_use]
    pub const fn seat_bytes(&self) -> usize {
        2 * self.ring_dim * 4
    }

    /// Bytes of a shard's step-1 table: `gamma` transform-form words per row slot.
    #[must_use]
    pub const fn table_bytes(&self) -> usize {
        self.gamma * self.ring_dim * 4
    }
}

/// Uniform residues from 64-bit words, a word in the short last interval rejected so no
/// residue is favoured.
fn expand_uniform(rng: &mut rand_chacha::ChaCha20Rng, q: u64, out: &mut [u32]) {
    use rand::RngCore;
    let zone = (u64::MAX / q) * q;
    for c in out {
        loop {
            let u = rng.next_u64();
            if u < zone {
                *c = (u % q) as u32;
                break;
            }
        }
    }
}

/// `f(X^t)` for a small signed polynomial.
fn tau_signed(v: &[i64], t: usize) -> Vec<i64> {
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

/// What both sides derive from a [`Params`].
#[derive(Debug)]
struct Ring {
    params: Params,
    lq: Ntt32,
    lp: Ntt32,
    /// Generator of the order-`gamma` automorphism subgroup.
    g: usize,
    delta: u32,
    p_mod_q: u32,
    /// `-p mod q`: the shift that takes a negative lift from limb `p` to limb `q`.
    neg_off: u32,
    /// Conditional subtractions that bring a residue mod `p` below `q`.
    p_over_q: u32,
    /// `p^-1 mod q` as a Montgomery operand.
    p_inv: u32,
    /// The key mask in transform form, Montgomery operands.
    w_q: Vec<u32>,
    w_p: Vec<u32>,
    /// `tables[i][k]`: slot of `f` that slot `k` of `tau_{g^i}(f)` reads. Server only.
    tables: Vec<Vec<u16>>,
}

impl Ring {
    fn new(params: &Params, with_tables: bool) -> Result<Self> {
        use rand::SeedableRng;
        params.validate()?;
        let d = params.ring_dim;
        let lq = Ntt32::new(d, params.q)?;
        let lp = Ntt32::new(d, params.special_prime)?;
        let g = 2 * d / params.gamma + 1;
        let mut gen_pows = Vec::with_capacity(params.gamma);
        let mut v = 1usize;
        for _ in 0..params.gamma {
            gen_pows.push(v);
            v = (v * g) % (2 * d);
        }
        let tables = if with_tables {
            let tables = lq.automorphism_tables(&gen_pows)?;
            if tables != lp.automorphism_tables(&gen_pows)? {
                return Err(pir_err!("the two limbs disagree on an automorphism"));
            }
            tables
        } else {
            Vec::new()
        };
        let mut rng = rand_chacha::ChaCha20Rng::from_seed(params.key_mask_seed);
        let mut w_q = vec![0u32; d];
        let mut w_p = vec![0u32; d];
        expand_uniform(&mut rng, params.q, &mut w_q);
        expand_uniform(&mut rng, params.special_prime, &mut w_p);
        lq.forward(&mut w_q);
        lp.forward(&mut w_p);
        for x in &mut w_q {
            *x = lq.p.to_mont(*x);
        }
        for x in &mut w_p {
            *x = lp.p.to_mont(*x);
        }
        let (q, p) = (params.q, params.special_prime);
        let p_mod_q = (p % q) as u32;
        Ok(Self {
            params: params.clone(),
            g,
            delta: (q / PLAINTEXT_MODULUS) as u32,
            p_mod_q,
            neg_off: ((q - p % q) % q) as u32,
            p_over_q: (p / q) as u32,
            p_inv: lq.p.to_mont(lq.p.pow(p % q, q - 2)),
            w_q,
            w_p,
            tables,
            lq,
            lp,
        })
    }

    /// Residue mod `q` of the centred value of a residue mod `p`.
    #[inline(always)]
    fn p_to_q_centred(&self, r: u32) -> u32 {
        let lq = &self.lq.p;
        let mut base = r;
        for _ in 0..self.p_over_q {
            base = lq.csub(base);
        }
        if r > self.lp.p.q / 2 {
            lq.add(base, self.neg_off)
        } else {
            base
        }
    }

    /// Residue mod `p` of the centred value of a residue mod `q` (`p > q`).
    #[inline(always)]
    fn q_to_p_centred(&self, c: u32) -> u32 {
        if c > self.lq.p.q / 2 {
            c + (self.lp.p.q - self.lq.p.q)
        } else {
            c
        }
    }

    /// `round(x / p) mod q` for `x` at `q * p` given by its two residues.
    #[inline(always)]
    fn mod_down(&self, xq: u32, xp: u32) -> u32 {
        let lq = &self.lq.p;
        lq.mont(lq.sub(xq, self.p_to_q_centred(xp)), self.p_inv)
    }
}

/// Signed nibbles of a cell, low nibble of each byte first.
fn nibbles(cell: &[u8], out: &mut [i8]) {
    for (pair, &byte) in out.as_chunks_mut::<2>().0.iter_mut().zip(cell) {
        pair[0] = ((byte << 4) as i8) >> 4;
        pair[1] = (byte as i8) >> 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::primality::is_prime;

    fn largest_prime_one_mod(step: u64, below: u64) -> u64 {
        let mut c = (below - 1) / step * step + 1;
        while !is_prime(c) {
            c -= step;
        }
        c
    }

    #[test]
    fn served_primes_are_derived_not_recalled() {
        assert_eq!(Q, largest_prime_one_mod(4096, 1 << 29));
        assert_eq!(SPECIAL_PRIME, largest_prime_one_mod(4096, 1 << 31));
        assert_eq!(Q % PLAINTEXT_MODULUS, 1, "decode offset");
        let log2 = (Q as f64).log2() + (SPECIAL_PRIME as f64).log2();
        assert!((log2 - 60.0).abs() < 1e-3, "key modulus near 2^60: {log2}");
    }

    #[test]
    fn served_set_is_valid_and_pins_its_sizes() {
        let p = Params::ring2048_cell256();
        p.validate().unwrap();
        assert!(
            (p.sigma - 6.4).abs() < f64::EPSILON,
            "query error standard deviation"
        );
        assert_eq!(p.key_bytes(), 7424 + 7936);
        assert_eq!(p.query_bytes(), 32 + 7424);
        assert_eq!(p.response_bytes(), 7424 + 1784);
        assert_eq!(p.seat_bytes(), 16_384);
        assert_eq!(p.table_bytes(), 4 << 20);
    }

    #[test]
    fn invalid_sets_are_refused() {
        let base = Params::ring2048_cell256();
        type Edit = Box<dyn Fn(&mut Params)>;
        let cases: [(&str, Edit); 7] = [
            ("ring", Box::new(|p| p.ring_dim = 3000)),
            ("width", Box::new(|p| p.gamma = 2048)),
            ("cell", Box::new(|p| p.cell_bytes = 255)),
            ("record", Box::new(|p| p.record_bytes = 257)),
            ("composite", Box::new(|p| p.q = 4097 * 4097)),
            (
                "order",
                Box::new(|p| std::mem::swap(&mut p.q, &mut p.special_prime)),
            ),
            ("sigma", Box::new(|p| p.sigma = f64::NAN)),
        ];
        for (name, edit) in cases {
            let mut p = base.clone();
            edit(&mut p);
            assert!(p.validate().is_err(), "{name}");
        }
    }

    /// The lifts and the rounded division by `p` against i128 arithmetic, at each lift
    /// boundary and at each step of the conditional subtractions, with random partners.
    #[test]
    fn mod_switch_helpers_match_exact_arithmetic_at_their_boundaries() {
        use rand::{Rng, SeedableRng};
        let ring = Ring::new(&Params::ring2048_cell256(), false).unwrap();
        let (q, p) = (i128::from(Q), i128::from(SPECIAL_PRIME));
        let centred = |x: i128, m: i128| if x > m / 2 { x - m } else { x };
        let mut rs: Vec<i128> = vec![0, 1, p / 2 - 1, p / 2, p / 2 + 1, p - 2, p - 1];
        for k in 1..=4 {
            rs.extend([k * q - 1, k * q, k * q + 1]);
        }
        let cs: Vec<i128> = vec![0, 1, q / 2 - 1, q / 2, q / 2 + 1, q - 2, q - 1];
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(306);
        let partners: Vec<i128> = (0..8).map(|_| rng.gen_range(0..q)).collect();
        for &r in &rs {
            let want = centred(r, p).rem_euclid(q);
            assert_eq!(
                i128::from(ring.p_to_q_centred(r as u32)),
                want,
                "p_to_q {r}"
            );
            for &xq in partners.iter().chain(&cs) {
                // round(x / p) mod q for x at q * p with residues (xq, r).
                let x = crt(xq, r, q, p);
                let want = (x - centred(r, p)) / p % q;
                let got = ring.mod_down(xq as u32, r as u32);
                assert_eq!(i128::from(got), want, "mod_down {xq} {r}");
                let exact = (x + (p - 1) / 2).div_euclid(p).rem_euclid(q);
                assert_eq!(want, exact, "rounding reference {xq} {r}");
            }
        }
        for &c in &cs {
            let want = centred(c, q).rem_euclid(p);
            assert_eq!(
                i128::from(ring.q_to_p_centred(c as u32)),
                want,
                "q_to_p {c}"
            );
        }
    }

    /// The value in `[0, q * p)` with residues `a` mod `q` and `b` mod `p`.
    fn crt(a: i128, b: i128, q: i128, p: i128) -> i128 {
        // Fermat: q^(p - 2) is the inverse of q mod the prime p.
        let (mut q_inv_p, mut base, mut e) = (1i128, q % p, p - 2);
        while e > 0 {
            if e & 1 == 1 {
                q_inv_p = q_inv_p * base % p;
            }
            base = base * base % p;
            e >>= 1;
        }
        let t = (b - a).rem_euclid(p) * q_inv_p % p;
        a + q * t
    }

    #[test]
    fn nibbles_are_signed_low_first() {
        let mut out = [0i8; 6];
        nibbles(&[0x00, 0x8f, 0x71], &mut out);
        assert_eq!(out, [0, 0, -1, -8, 1, 7]);
    }
}
