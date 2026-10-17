//! Byte forms. Every coefficient is packed at its prime's bit width, little-endian bit
//! order, and a decoder refuses a wrong length, a value at or above its prime, and set
//! padding bits, so each message has exactly one encoding.

use crate::math::poly::{pack_coefficients_tight, unpack_coefficients_tight};
use crate::pir::error::{pir_err, Result};

use super::{Params, SEED_BYTES};

fn pack(values: &[u32], m: u64) -> Result<Vec<u8>> {
    let wide: Vec<u64> = values.iter().map(|&v| u64::from(v)).collect();
    pack_coefficients_tight(&wide, &[m], wide.len()).map_err(|e| pir_err!("{e}"))
}

fn unpack(bytes: &[u8], m: u64, count: usize) -> Result<Vec<u32>> {
    Ok(unpack_coefficients_tight(bytes, &[m], count)
        .map_err(|e| pir_err!("{e}"))?
        .into_iter()
        .map(|v| v as u32)
        .collect())
}

fn split<'a>(bytes: &'a [u8], at: usize, want: usize, what: &str) -> Result<(&'a [u8], &'a [u8])> {
    if bytes.len() != want {
        return Err(pir_err!(
            "{what} has {} bytes, expected {want}",
            bytes.len()
        ));
    }
    Ok(bytes.split_at(at))
}

/// A client's packing key: `y = p * tau_g(s) - s * w + e` at `q * p`, one residue
/// vector per prime, coefficient form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackingKey {
    pub(crate) q: Vec<u32>,
    pub(crate) p: Vec<u32>,
}

impl PackingKey {
    /// Limb `q`, then limb `p`.
    pub fn to_bytes(&self, params: &Params) -> Result<Vec<u8>> {
        let mut out = pack(&self.q, params.q)?;
        out.extend(pack(&self.p, params.special_prime)?);
        Ok(out)
    }

    /// Refuses anything [`Self::to_bytes`] would not emit for `params`.
    pub fn from_bytes(params: &Params, bytes: &[u8]) -> Result<Self> {
        let d = params.ring_dim;
        let at = Params::packed_bytes(d, params.q);
        let (q, p) = split(bytes, at, params.key_bytes(), "packing key")?;
        Ok(Self {
            q: unpack(q, params.q, d)?,
            p: unpack(p, params.special_prime, d)?,
        })
    }
}

/// One query: the seed its mask expands from, and its body at `q`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    pub(crate) seed: [u8; SEED_BYTES],
    pub(crate) body: Vec<u32>,
}

impl Query {
    /// Seed, then body.
    pub fn to_bytes(&self, params: &Params) -> Result<Vec<u8>> {
        let mut out = self.seed.to_vec();
        out.extend(pack(&self.body, params.q)?);
        Ok(out)
    }

    /// Refuses anything [`Self::to_bytes`] would not emit for `params`.
    pub fn from_bytes(params: &Params, bytes: &[u8]) -> Result<Self> {
        let (seed, body) = split(bytes, SEED_BYTES, params.query_bytes(), "query")?;
        let mut s = [0u8; SEED_BYTES];
        s.copy_from_slice(seed);
        Ok(Self {
            seed: s,
            body: unpack(body, params.q, params.ring_dim)?,
        })
    }
}

/// A packed response: the whole mask and the body coefficients that carry the record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Response {
    pub(crate) mask: Vec<u32>,
    pub(crate) body: Vec<u32>,
}

impl Response {
    /// Mask, then body.
    pub fn to_bytes(&self, params: &Params) -> Result<Vec<u8>> {
        let mut out = pack(&self.mask, params.q)?;
        out.extend(pack(&self.body, params.q)?);
        Ok(out)
    }

    /// Refuses anything [`Self::to_bytes`] would not emit for `params`.
    pub fn from_bytes(params: &Params, bytes: &[u8]) -> Result<Self> {
        let at = Params::packed_bytes(params.ring_dim, params.q);
        let (mask, body) = split(bytes, at, params.response_bytes(), "response")?;
        Ok(Self {
            mask: unpack(mask, params.q, params.ring_dim)?,
            body: unpack(body, params.q, params.body_coefficients())?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_message_round_trips_and_refuses_a_wrong_length_or_value() {
        let params = Params::ring2048_cell256();
        let d = params.ring_dim;
        let top_q = params.q as u32 - 1;
        let key = PackingKey {
            q: vec![top_q; d],
            p: vec![params.special_prime as u32 - 1; d],
        };
        let query = Query {
            seed: [7; SEED_BYTES],
            body: (0..d as u32).collect(),
        };
        let response = Response {
            mask: vec![top_q; d],
            body: vec![1; params.body_coefficients()],
        };
        let kb = key.to_bytes(&params).unwrap();
        let qb = query.to_bytes(&params).unwrap();
        let rb = response.to_bytes(&params).unwrap();
        assert_eq!(kb.len(), params.key_bytes());
        assert_eq!(qb.len(), params.query_bytes());
        assert_eq!(rb.len(), params.response_bytes());
        assert_eq!(PackingKey::from_bytes(&params, &kb).unwrap(), key);
        assert_eq!(Query::from_bytes(&params, &qb).unwrap(), query);
        assert_eq!(Response::from_bytes(&params, &rb).unwrap(), response);

        assert!(PackingKey::from_bytes(&params, &kb[1..]).is_err());
        assert!(Query::from_bytes(&params, &[qb.clone(), vec![0]].concat()).is_err());
        assert!(Response::from_bytes(&params, &rb[..rb.len() - 1]).is_err());

        // q itself in the first body coefficient of a query: not a residue.
        let mut bad = qb.clone();
        let q = params.q as u32;
        for (i, byte) in bad[SEED_BYTES..SEED_BYTES + 4].iter_mut().enumerate() {
            *byte = (q >> (8 * i)) as u8 | (*byte & if i == 3 { 0xe0 } else { 0 });
        }
        assert!(Query::from_bytes(&params, &bad).is_err());
        // A set padding bit in the last byte of a response body (492 * 29 bits ends mid-byte).
        let mut pad = rb.clone();
        *pad.last_mut().unwrap() |= 0x80;
        assert!(Response::from_bytes(&params, &pad).is_err());
    }
}
