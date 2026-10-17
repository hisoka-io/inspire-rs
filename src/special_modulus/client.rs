//! The client: one secret, one packing key, fresh noise and a fresh mask seed per query.

use rand::RngCore;
use rand_chacha::ChaCha20Rng;

use crate::math::gaussian::os_seeded_chacha;
use crate::math::GaussianSampler;
use crate::pir::error::{pir_err, Result};

use subtle::{ConditionallySelectable, ConstantTimeEq};

use super::wire::{PackingKey, Query, Response};
use super::{
    expand_uniform, tau_signed, Params, Ring, KEY_MASK_SEED, PLAINTEXT_MODULUS, SEED_BYTES,
};

/// `floor((x + floor(delta/2)) / delta) mod 16` over a secret phase `x < 2^31`, dividing
/// only the public `delta` (once, here).
struct Rounding {
    delta: u64,
    half: u64,
    /// `floor(2^40 / delta)`: the estimate is at most one short for numerators below 2^32.
    reciprocal: u64,
}

impl Rounding {
    fn new(delta: u32) -> Self {
        Self {
            delta: u64::from(delta),
            half: u64::from(delta / 2),
            reciprocal: (1u64 << 40) / u64::from(delta),
        }
    }

    #[inline]
    fn nibble(&self, x: u32) -> u8 {
        let numerator = u64::from(x) + self.half;
        let estimate = (numerator * self.reciprocal) >> 40;
        let rest = numerator - estimate * self.delta;
        // One short exactly when `rest` reaches `delta`; the borrow of `rest - delta` says which.
        let quotient = estimate + 1 - (rest.wrapping_sub(self.delta) >> 63);
        (quotient & (PLAINTEXT_MODULUS - 1)) as u8
    }
}

/// A client under an entropy-drawn secret reads only under the served ring and primes,
/// its fixed key mask, and at least its error width. A mask the caller picks lets a server
/// that supplied it tell clients apart; a narrower error weakens the secret.
fn served_floor(params: &Params) -> Result<()> {
    let served = Params::ring2048_cell256();
    if params.ring_dim < served.ring_dim
        || params.q != served.q
        || params.special_prime != served.special_prime
    {
        return Err(pir_err!(
            "a client reads under the served ring and primes; other sets are for seeded tests"
        ));
    }
    if params.key_mask_seed != KEY_MASK_SEED {
        return Err(pir_err!(
            "the key mask is the set's own, not a supplied one"
        ));
    }
    if params.sigma.is_nan() || params.sigma < served.sigma {
        return Err(pir_err!(
            "standard deviation {} is below the served {}",
            params.sigma,
            served.sigma
        ));
    }
    Ok(())
}

/// Caller-supplied noise must be small enough for the branch-free lift to a residue.
#[cfg(feature = "caller-secret-keys")]
fn all_small(v: &[i64]) -> bool {
    v.iter().all(|x| x.unsigned_abs() < 1 << 20)
}

/// Reads rows under one secret.
///
/// The secret is drawn when the client is built and never leaves it. The key mask is a
/// public constant of the set, so a second key under the same secret would share that mask
/// and expose the secret: a client therefore builds exactly one [`PackingKey`] and resends
/// the same bytes on every upload. A new key means a new client.
pub struct Client {
    ring: Ring,
    /// Transform of the secret at `q`, Montgomery operands.
    s_q: Vec<u32>,
    key: PackingKey,
    noise: GaussianSampler,
    seeds: ChaCha20Rng,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("ring_dim", &self.ring.params.ring_dim)
            .finish_non_exhaustive()
    }
}

impl Client {
    /// A client under a secret drawn here from the platform entropy source.
    ///
    /// # Errors
    ///
    /// Refuses an invalid set; a set off the served ring and primes, with another key mask,
    /// or with a narrower error; or when the entropy source fails.
    pub fn new(params: &Params) -> Result<Self> {
        params.validate()?;
        served_floor(params)?;
        let noise = GaussianSampler::from_os_entropy(params.sigma)
            .map_err(|e| pir_err!("no client without a fresh secret: {e}"))?;
        let seeds = os_seeded_chacha("special-modulus query seeds")
            .map_err(|e| pir_err!("no client without fresh query seeds: {e}"))?;
        Self::drawn(params, noise, seeds)
    }

    /// Everything drawn from one caller seed: the secret, the key error, every query's
    /// error and every query's mask seed.
    ///
    /// Test and KAT builds only: a replayed seed is a replayed secret, and its second key
    /// or second query stream exposes it.
    #[cfg(feature = "caller-secret-keys")]
    pub fn from_seed(params: &Params, seed: [u8; SEED_BYTES]) -> Result<Self> {
        use rand::SeedableRng;
        let noise = GaussianSampler::from_seed(params.sigma, seed);
        let mut seeds = ChaCha20Rng::from_seed(seed);
        seeds.set_stream(1);
        Self::drawn(params, noise, seeds)
    }

    /// The secret, then the key error, from `noise`. The one draw for every sampler, so the
    /// pinned wire bytes of a seeded client cover the draw of an entropy-seeded one.
    fn drawn(params: &Params, mut noise: GaussianSampler, seeds: ChaCha20Rng) -> Result<Self> {
        let secret = noise.sample_vec(params.ring_dim);
        let key_error = noise.sample_vec(params.ring_dim);
        Self::build(params, &secret, &key_error, noise, seeds)
    }

    /// A client over a given secret and key error, its queries drawn from `seed`.
    ///
    /// Test and KAT builds only, for the same reason as [`Self::from_seed`].
    #[cfg(feature = "caller-secret-keys")]
    pub fn from_parts(
        params: &Params,
        secret: &[i64],
        key_error: &[i64],
        seed: [u8; SEED_BYTES],
    ) -> Result<Self> {
        use rand::SeedableRng;
        if !all_small(secret) || !all_small(key_error) {
            return Err(pir_err!("secret or key error is not a small value"));
        }
        let noise = GaussianSampler::from_seed(params.sigma, seed);
        let mut seeds = ChaCha20Rng::from_seed(seed);
        seeds.set_stream(1);
        Self::build(params, secret, key_error, noise, seeds)
    }

    fn build(
        params: &Params,
        secret: &[i64],
        key_error: &[i64],
        noise: GaussianSampler,
        seeds: ChaCha20Rng,
    ) -> Result<Self> {
        let ring = Ring::new(params, false)?;
        let d = params.ring_dim;
        if secret.len() != d || key_error.len() != d {
            return Err(pir_err!("secret and key error need {d} coefficients"));
        }
        let (lq, lp) = (&ring.lq, &ring.lp);
        let mut s_q: Vec<u32> = secret.iter().map(|&x| lq.p.small(x)).collect();
        let mut s_p: Vec<u32> = secret.iter().map(|&x| lp.p.small(x)).collect();
        lq.forward(&mut s_q);
        lp.forward(&mut s_p);

        let mut sw_q: Vec<u32> = s_q
            .iter()
            .zip(&ring.w_q)
            .map(|(&s, &w)| lq.p.mont(s, w))
            .collect();
        let mut sw_p: Vec<u32> = s_p
            .iter()
            .zip(&ring.w_p)
            .map(|(&s, &w)| lp.p.mont(s, w))
            .collect();
        lq.inverse(&mut sw_q);
        lp.inverse(&mut sw_p);

        let ts = tau_signed(secret, ring.g);
        let p_mod_q = lq.p.to_mont(ring.p_mod_q);
        let key_q = (0..d)
            .map(|k| {
                let lead = lq.p.mont(lq.p.small(ts[k]), p_mod_q);
                lq.p.add(lq.p.sub(lead, sw_q[k]), lq.p.small(key_error[k]))
            })
            .collect();
        // The leading term p * tau_g(s) vanishes mod p.
        let key_p = (0..d)
            .map(|k| lp.p.sub(lp.p.small(key_error[k]), sw_p[k]))
            .collect();

        for x in &mut s_q {
            *x = lq.p.to_mont(*x);
        }
        Ok(Self {
            key: PackingKey { q: key_q, p: key_p },
            s_q,
            ring,
            noise,
            seeds,
        })
    }

    /// The set this client reads under.
    pub fn params(&self) -> &Params {
        &self.ring.params
    }

    /// The one packing key of this client; the same bytes on every call.
    pub fn packing_key(&self) -> &PackingKey {
        &self.key
    }

    /// A query for `row` (below the ring dimension), under a fresh mask seed and fresh
    /// error.
    pub fn query(&mut self, row: usize) -> Result<Query> {
        let mut seed = [0u8; SEED_BYTES];
        self.seeds.fill_bytes(&mut seed);
        let error = self.noise.sample_vec(self.ring.params.ring_dim);
        self.query_with(row, seed, &error)
    }

    /// `b = -a * s + e + delta * X^-row`, `a` expanded from `seed`.
    ///
    /// Test and KAT builds only: the error must be fresh for every query.
    #[cfg(feature = "caller-secret-keys")]
    pub fn query_from_parts(
        &self,
        row: usize,
        seed: [u8; SEED_BYTES],
        error: &[i64],
    ) -> Result<Query> {
        if !all_small(error) {
            return Err(pir_err!("query error is not a small value"));
        }
        self.query_with(row, seed, error)
    }

    fn query_with(&self, row: usize, seed: [u8; SEED_BYTES], error: &[i64]) -> Result<Query> {
        use rand::SeedableRng;
        let ring = &self.ring;
        let d = ring.params.ring_dim;
        if row >= d {
            return Err(pir_err!("row {row} is past the {d} rows of a shard"));
        }
        if error.len() != d {
            return Err(pir_err!("query error needs {d} coefficients"));
        }
        let lq = &ring.lq;
        let mut a = vec![0u32; d];
        expand_uniform(&mut ChaCha20Rng::from_seed(seed), ring.params.q, &mut a);
        lq.forward(&mut a);
        for (x, &s) in a.iter_mut().zip(&self.s_q) {
            *x = lq.p.mont(*x, s);
        }
        lq.inverse(&mut a);
        // delta * X^-row: +delta at 0 for row 0, else -delta at d - row. The row is the
        // secret a query hides, so every position is visited and the message selected in.
        let target = ((d - row) & (d - 1)) as u32;
        let message =
            u32::conditional_select(&(lq.p.q - ring.delta), &ring.delta, (row as u32).ct_eq(&0));
        let body = error
            .iter()
            .zip(&a)
            .zip(0u32..)
            .map(|((&e, &x), k)| {
                let m = u32::conditional_select(&0, &message, k.ct_eq(&target));
                lq.p.add(lq.p.sub(lq.p.small(e), x), m)
            })
            .collect();
        Ok(Query { seed, body })
    }

    /// `a * s + b` on the body coefficients.
    fn phase(&self, response: &Response) -> Result<Vec<u32>> {
        let ring = &self.ring;
        let params = &ring.params;
        let d = params.ring_dim;
        if response.mask.len() != d || response.body.len() != params.body_coefficients() {
            return Err(pir_err!("response shape does not match the set"));
        }
        let lq = &ring.lq;
        let mut phase = response.mask.clone();
        lq.forward(&mut phase);
        for (x, &s) in phase.iter_mut().zip(&self.s_q) {
            *x = lq.p.mont(*x, s);
        }
        lq.inverse(&mut phase);
        phase.truncate(response.body.len());
        for (x, &b) in phase.iter_mut().zip(&response.body) {
            *x = lq.p.add(*x, b);
        }
        Ok(phase)
    }

    /// The record a response carries.
    pub fn decode(&self, response: &Response) -> Result<Vec<u8>> {
        let rounding = Rounding::new(self.ring.delta);
        let phase = self.phase(response)?;
        let nibble = |x: u32| rounding.nibble(x);
        Ok(phase
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| nibble(pair[0]) | (nibble(pair[1]) << 4))
            .collect())
    }

    /// The error of every body coefficient of `response` against the record it should
    /// carry: the phase minus `delta` times the signed nibble, centred.
    ///
    /// Test and measurement builds only: it reads the secret's view of the noise.
    #[cfg(feature = "caller-secret-keys")]
    pub fn body_errors(&self, response: &Response, record: &[u8]) -> Result<Vec<i64>> {
        let ring = &self.ring;
        if record.len() != ring.params.record_bytes {
            return Err(pir_err!("record needs {} bytes", ring.params.record_bytes));
        }
        let mut values = vec![0i8; 2 * record.len()];
        super::nibbles(record, &mut values);
        let lq = &ring.lq.p;
        Ok(self
            .phase(response)?
            .iter()
            .zip(&values)
            .map(|(&x, &v)| {
                let scaled = lq.small(i64::from(v) * i64::from(ring.delta));
                lq.centred(lq.sub(x, scaled))
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::{Client, Params, Rounding, KEY_MASK_SEED};
    use rand::{Rng, SeedableRng};

    /// The secret and the key error of a client, read back from its secret transform and
    /// its key: `e = key_p + s * w_p` in limb `p`, and `key_q + s * w_q - p * tau_g(s)` in
    /// limb `q`, which must agree.
    fn secret_and_key_error(client: &Client) -> (Vec<i64>, Vec<i64>) {
        let ring = &client.ring;
        let (lq, lp) = (&ring.lq, &ring.lp);
        let mut s_q: Vec<u32> = client.s_q.iter().map(|&x| lq.p.mont(x, 1)).collect();
        lq.inverse(&mut s_q);
        let secret: Vec<i64> = s_q.iter().map(|&x| lq.p.centred(x)).collect();

        let times_mask = |l: &crate::special_modulus::arith::Ntt32, w: &[u32]| {
            let mut v: Vec<u32> = secret.iter().map(|&x| l.p.small(x)).collect();
            l.forward(&mut v);
            for (x, &m) in v.iter_mut().zip(w) {
                *x = l.p.mont(*x, m);
            }
            l.inverse(&mut v);
            v
        };
        let sw_p = times_mask(lp, &ring.w_p);
        let sw_q = times_mask(lq, &ring.w_q);
        let e_p: Vec<i64> = client
            .key
            .p
            .iter()
            .zip(&sw_p)
            .map(|(&k, &x)| lp.p.centred(lp.p.add(k, x)))
            .collect();
        let ts = super::tau_signed(&secret, ring.g);
        let p_mod_q = lq.p.to_mont(ring.p_mod_q);
        let e_q: Vec<i64> = (0..secret.len())
            .map(|k| {
                let lead = lq.p.mont(lq.p.small(ts[k]), p_mod_q);
                lq.p.centred(lq.p.sub(lq.p.add(client.key.q[k], sw_q[k]), lead))
            })
            .collect();
        assert_eq!(e_p, e_q, "the two key limbs carry different errors");
        (secret, e_p)
    }

    fn std_dev(v: &[i64]) -> f64 {
        let n = v.len() as f64;
        let mean = v.iter().sum::<i64>() as f64 / n;
        (v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n).sqrt()
    }

    /// The production constructor's key carries a fresh error of the set's width, drawn
    /// apart from the secret. A noiseless key gives the secret away in one division, and
    /// the key error is too small a share of a response's noise for the decode margin to
    /// show its absence.
    #[test]
    fn an_entropy_seeded_key_carries_a_fresh_error_of_the_set_width() {
        let params = Params::ring2048_cell256();
        let mut errors = Vec::new();
        for _ in 0..4 {
            let client = Client::new(&params).unwrap();
            let (secret, error) = secret_and_key_error(&client);
            for (name, v) in [("secret", &secret), ("key error", &error)] {
                let sd = std_dev(v);
                // Over 2,048 draws the sample deviation sits within about 2% of sigma.
                assert!(
                    (sd / params.sigma - 1.0).abs() < 0.12,
                    "{name} deviation {sd} against {}",
                    params.sigma
                );
                assert!(v.iter().all(|x| x.unsigned_abs() < 128), "{name} not small");
            }
            assert_ne!(secret, error, "key error is the secret");
            assert!(!errors.contains(&error), "key error reused across clients");
            errors.push(error);
        }
    }

    /// The next prime `1 mod 4096` below the served special prime.
    fn next_special_prime() -> u64 {
        let mut c = super::super::SPECIAL_PRIME - 4096;
        while !crate::math::primality::is_prime(c) {
            c -= 4096;
        }
        c
    }

    #[test]
    fn an_entropy_seeded_client_refuses_a_set_below_the_served_one() {
        let served = Params::ring2048_cell256();
        type Edit = Box<dyn Fn(&mut Params)>;
        let cases: [(&str, Edit); 5] = [
            ("narrower error", Box::new(|p| p.sigma = 3.2)),
            ("supplied mask", Box::new(|p| p.key_mask_seed[0] ^= 1)),
            ("smaller ring", Box::new(|p| p.ring_dim = 1024)),
            (
                "other prime",
                Box::new(|p| p.special_prime = next_special_prime()),
            ),
            ("below the sampler range", Box::new(|p| p.sigma = 0.5)),
        ];
        for (name, edit) in cases {
            let mut p = served.clone();
            edit(&mut p);
            // The set itself runs; only the floor refuses it.
            if name != "below the sampler range" {
                p.validate().unwrap();
            }
            assert!(Client::new(&p).is_err(), "{name}");
        }
        let mut wider = served;
        wider.sigma = 8.0;
        assert_eq!(wider.key_mask_seed, KEY_MASK_SEED);
        Client::new(&wider).unwrap();
    }

    #[test]
    fn rounding_matches_the_division_at_every_boundary_and_at_random() {
        for q in [super::super::Q, 12_289 * 4096 + 1, 1_048_609] {
            let delta = (q / 16) as u32;
            let r = Rounding::new(delta);
            let exact =
                |x: u32| ((u64::from(x) + u64::from(delta / 2)) / u64::from(delta) % 16) as u8;
            let mut xs: Vec<u32> = vec![0, 1, q as u32 - 1, q as u32 - 2];
            for k in 0..=16u64 {
                for off in [-2i64, -1, 0, 1, 2] {
                    let at = (k * u64::from(delta)) as i64 - i64::from(delta / 2) + off;
                    if (0..q as i64).contains(&at) {
                        xs.push(at as u32);
                    }
                }
            }
            let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(q);
            xs.extend((0..200_000).map(|_| rng.gen_range(0..q as u32)));
            for x in xs {
                assert_eq!(r.nibble(x), exact(x), "q {q} x {x}");
            }
        }
    }
}
