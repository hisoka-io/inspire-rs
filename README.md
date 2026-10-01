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
use raven_inspire::{extract_inspiring, query_seeded, respond_seeded_inspiring, setup};

let params = InspireParams::secure_128_d2048();
let entry_size = 32;
let database = vec![0u8; params.ring_dim * entry_size];
let mut sampler = GaussianSampler::from_os_entropy(params.sigma)?;

let (crs, db, sk) = setup(&params, &database, entry_size, &mut sampler)?;
let (state, query) = query_seeded(&crs, 42, &db.config, &sk, &mut sampler)?;
let response = respond_seeded_inspiring(&crs, &db, &query)?;
let record = extract_inspiring(&crs, &state, &response, entry_size)?;
```

`ClientSession` caches the packing material so repeated queries skip it. It can register
the ~48 KiB of packing keys with a `ServerSessionStore` once and send a handle afterwards.

## Parameters and sizes

`InspireParams::secure_128_d2048` uses ring dimension 2048, a single 60-bit prime
`q = 2^60 - 2^14 + 1`, `p = 65537` and `sigma = 6.4`. Sizes at d=2048 for a 512-byte
record are independent of database size:

| Message | Bytes |
|---|---:|
| First query, packing keys inline | 61,735 |
| Query carrying a session handle | 15,491 |
| Response, mod-switched to a 36-bit modulus | 10,446 |
| Response, unswitched | 17,358 |

## Cargo features

| Feature | Effect |
|---|---|
| `parallel` | Runs the respond and packing loops on rayon. Output is byte-identical to the default sequential build. |
| `mod-switch-response` | Response modulus switching (`pir::mod_switch`) and the matching client extractor. |
| `simd-packing-offline` | AVX-512 IFMA52 kernel for offline packing, selected at runtime by CPUID with a scalar fallback. |

## Security model

- **Parameters.** `secure_128_d2048` (ring dimension 2048, a 60-bit modulus) measures 121.5 bits with malb/lattice-estimator
  (binding attack `primal_bdd`); the preset keeps its historical name. The d=4096 preset is not measured.
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
