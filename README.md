# raven-inspire

A Rust implementation of InsPIRe, single-server private information retrieval with
server-side preprocessing ([eprint 2025/1352](https://eprint.iacr.org/2025/1352)). A client
fetches one fixed-width record from a server-held database by an encrypted query, and the
server answers it without decrypting it.

It is a fork of inspire-rs by igor53627, originally published on GitHub as
`igor53627/inspire-rs`. It is a library only (no server, CLI or async runtime), so it also
builds for `wasm32-unknown-unknown`.

## Status

Alpha (`0.1.0-alpha.0`), not published to crates.io. The wire formats are versioned: the serialized CRS carries a version prefix,
and a CRS with an older layout is refused rather than decoded.

## Protocol

1. `setup` encodes the database as polynomials, one record per ring coefficient and
   `ring_dim` records per shard, and generates the CRS and the client secret key.
2. `query` / `query_seeded` encrypt the inverse monomial `X^(-k)` for local index `k`.
3. `respond*` multiplies each column polynomial by the encrypted monomial, which rotates
   record `k` to coefficient 0, then packs the result.
4. `extract*` decrypts and reassembles the record bytes.

Three variants (`InspireVariant`): `NoPacking` returns one ciphertext per 16-bit column,
`OnePacking` packs them with a log(d)-key automorphism tree, and `TwoPacking` uses a seeded
query and InspiRING packing with two key-switching matrices. `TwoPacking` is the one to use.

```rust
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::InspireParams;
use raven_inspire::{extract_inspiring, respond_seeded_inspiring, setup, ClientSession};

let params = InspireParams::secure_128_d2048();
let entry_size = 32;
let database = vec![0u8; params.ring_dim * entry_size];
let mut sampler = GaussianSampler::from_os_entropy(params.sigma)?;

let (crs, db, _) = setup(&params, &database, entry_size, &mut sampler)?;
let session = ClientSession::with_fresh_secret(crs.clone())?;
let (state, query) = session.query_seeded(42, &db.config, &mut sampler)?;
let response = respond_seeded_inspiring(&crs, &db, &query)?;
let record = extract_inspiring(&crs, &state, &response, entry_size)?;
```

`ClientSession::with_fresh_secret` draws the client's secret key from a fresh OS-seeded stream,
never from a sampler the caller passes in, and builds its one set of packing keys; every query,
retry and restored session resends those bytes. A second key under the same secret would leak it,
so no default-build function builds a key under a secret the caller passes in. It can register
the ~48 KiB of packing keys with a `ServerSessionStore` once and send a handle afterwards; the
server keeps a registered key as three polynomials (49,152 bytes at d=2048, at any record
width). `ClientSession::query_batch` sends any number of queries under that one key, inline or
by handle, and `respond_seeded_batch` answers them.

A server can fold the first step of the packing offline phase over a shard's columns ahead of
the queries with `ShardData::precompute_packing`. Responses keep their bytes and that step's
per-query work drops from gamma^2 pointwise products to gamma. The store is as large as the
shard's columns (gamma x 16 KiB at d=2048). Every respond checks the columns against the store
and refuses a shard edited after the fold until it is folded again.

## Parameters and sizes

`InspireParams::secure_128_d2048` uses ring dimension 2048, a single 60-bit prime
`q = 2^60 - 2^14 + 1`, `p = 65537` and `sigma = 6.4`. Sizes at d=2048 for a 512-byte
record are independent of database size:

| Message | Bytes |
|---|---:|
| First query, packing keys inline | 61,735 |
| Query carrying a session handle | 15,491 |
| Batch of n queries, packing keys inline | 46,264 + 15,477 n |
| Batch of n queries by session handle | 20 + 15,477 n |
| Response, mod-switched to a 36-bit modulus | 10,446 |
| Response, unswitched | 17,358 |

A parameter set with `approximate_packing_gadget` on packs under a key of two polynomials: the
server rounds the lowest gadget digit away and the key has no row for it. The key is then
30,843 bytes on the wire and 32,768 bytes registered, and a batch with the key inline is
30,855 + 15,477 n bytes. The presets leave the option off. `benches/approximate_gadget_noise.rs`
measures the served decode margin under both gadgets on the production ring.

## Special-modulus parameter set

`special_modulus` is a second parameter set for short records: ring dimension 2048, a 29-bit
ciphertext prime, plaintext modulus 16 (one signed nibble per coefficient) and a packing key of a
single polynomial at the ciphertext prime times a 31-bit special prime. The server switches each
packing step at the two-prime modulus and divides by the special prime, so the response stays at
the ciphertext prime with no modulus switch. A `Client` draws its secret inside and builds one key,
and reads only under the served ring, primes, key mask and error width (other sets are for the
seeded test constructors); a `Server` folds step 1 of the packing over each shard into a `ShardTable` that keeps the shard's
cells, so a response is always computed from the cells its table was built from. Arithmetic runs
on 32-bit words, with an AVX-512 build of the same loops under `simd-packing-offline`.

Sizes for `Params::ring2048_cell256` (256-byte cells returning a 246-byte record):

| Message or state | Bytes |
|---|---:|
| Packing key, once per client | 15,360 |
| Query | 7,456 |
| Response | 9,208 |
| Seated key on the server | 16,384 |
| Shard table on the server | 4,194,304 + 1,048,576 |

`tests/special_modulus_kat.rs` checks every byte against a generic 64-bit reference of the same
algorithm, and `tests/special_modulus_noise_gate.rs` measures the decode margin on 1,000
responses over 40 fresh secrets.

## Cargo features

| Feature | Effect |
|---|---|
| `parallel` | Runs the respond and packing loops on rayon. Output is byte-identical to the default sequential build. |
| `mod-switch-response` | Response modulus switching (`pir::mod_switch`) and the matching client extractor. |
| `simd-packing-offline` | AVX-512 kernels: IFMA52 for offline packing, and the AVX-512 build of the special-modulus loops. Selected at runtime by CPUID with a scalar fallback. |
| `caller-secret-keys` | Tests and known-answer fixtures only: `ClientSession::new`, `query`, `query_seeded` and `ClientPackingKeys::generate`, which build a packing key under a secret the caller supplies; `ClientSession::query`, `query_seeded` and `query_batch`, which draw query noise from a caller's sampler (clients use `query_seeded_fresh` and `query_batch_fresh`); and the special-modulus client's seeded constructors. |

## Security model

- **Parameters.** `secure_128_d2048` is ring dimension 2048 with a 60-bit modulus; the preset name is historical.
- **Honest-server answers.** A response is not bound to the queried index; an application checks the record it receives against a
  commitment it trusts.
- **Correctness margin.** The served 36-bit response keeps 8.88 bits of noise margin at 512-byte records and 10.30 bits at 32-byte
  records, measured by `benches/packing_noise_measurement.rs`.
- **Constant time.** Client-side NTT, ring arithmetic, query generation, key generation and decryption avoid secret-dependent branches,
  divisions and indexing; `tests/secret_dependent_spelling_gate.rs` enforces the spelling, and release builds for x86-64 and wasm32 were
  inspected at rustc 1.98. `SessionResidue` holds the client secret key: store it as a secret.

Report security issues privately through this repository's GitHub security advisories, not the public issue tracker.

## Differences from inspire-rs

- Secure presets use the single prime `2^60 - 2^14 + 1`. Upstream's two-prime modulus
  runs out of noise budget at 256-byte records and decrypts to wrong bytes.
- A fix for NTT-domain automorphisms, which previously dropped a CRT limb and corrupted
  two-prime decryption at d >= 256.
- Typed errors in place of panics for oversized shard geometry, a variant on the wrong
  query type, and malformed or mis-shaped wire input.
- A CRS shrunk from ~35 MiB to ~1.1 MiB, with a version prefix and a decode size cap. The
  binary codec is sized to the modulus width.
- Client sessions and a packing-key handshake, response modulus switching, branch-free
  client arithmetic, and `rayon` as an optional feature.
- Upstream's server, CLI, mmap and Ethereum database modules are removed.

## Building and testing

```bash
cargo build --release
cargo test --release --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

Run tests with `--release`, because the d=2048 tests are slow in debug builds. Tests
marked `#[ignore]` cover large cells (for example 2^20 records of 256 bytes) and run with
`-- --ignored`. The timing harnesses in `benches/` are run by hand, and each file states
how. Minimum supported Rust version: 1.89.

## License

Apache-2.0. This project is a fork of inspire-rs (`igor53627/inspire-rs`), which is
licensed MIT OR Apache-2.0.
