# Upstream attribution

This crate is a fork of inspire-rs (https://github.com/igor53627/inspire-rs),
licensed MIT OR Apache-2.0. It is redistributed here under Apache-2.0 per
Raven's framework-level license; no rights on the original upstream are waived.

## Local patches

| File | Change |
|---|---|
| `pir/encode_db.rs` | Oversized `ShardConfig` returns a typed error instead of panicking inside `inverse_monomial`. |
| `pir/respond.rs` | `respond_with_variant(TwoPacking)` errors to the seeded entry point instead of falling through to OnePacking. |
| `pir/extract.rs` | `extract_with_variant(TwoPacking)` routes InspiRING-shaped responses to `extract_inspiring`. |
| `params.rs` | `secure_128_d{2048,4096}` use the single-prime `DEFAULT_Q` (q = 2^60 - 2^14 + 1) instead of the under-provisioned 2-CRT moduli. |
| `inspiring/inspiring2.rs` | `apply_automorphism_ntt*` apply the permutation to each CRT limb; the pre-fork single-limb loop corrupted decryption at d >= 256 under 2-CRT TwoPacking + InspiRING. |
| `pir/setup.rs` | **CORRECTED 2026-09-12:** the first inventory counted five removed fields; the fork diff contains six because `inspiring_precomp` was removed with `crs_a_vectors`, `k_g`, `k_h`, `packing_k_h`, and `packing_k_g`. The InspiRING-active bit derives from `inspiring_num_columns > 0`, and live packing derives keys from `inspiring_w_seed`. Shrinks the client-shipped CRS ~34.9 -> ~1.1 MiB. Breaking change. The CRS is now wrapped with a `RAVEN_CRS_v01` magic prefix (`to_versioned_bytes`) and a `DECODE_LIMIT_BYTES` length cap; a blob under the prior untagged layout fails the magic check on load and requires re-bootstrap. |
| `Cargo.toml`, `par_prelude.rs`, `pir/respond.rs`, `inspiring/inspiring2.rs` | `rayon` is optional behind a non-default `parallel` feature; a sequential `par_prelude` shim is the default path, so the wasm client ships no rayon. The server opts in. Output is byte-identical with the feature on or off. |

## Local additions

| File | Purpose |
|---|---|
| `pir/session.rs` | `ClientSession` cache plus the packing-key pre-exchange handshake (keys uploaded once per session, referenced by handle). |
| `pir/session.rs` | `Clone` on `ClientSession`; `SessionResidue` (`Serialize`/`Deserialize`) with `to_residue`/`from_residue`. The residue persists CRS + secret key + packing-key body (~1.25 MiB), omitting the >160 MiB automorph tables; a rehydrated session leaves `pack_params` `None` and serves queries without rebuilding them. Carries the client secret key - see the `to_residue` security doc. |
| `params.rs` | `InspireParams::for_scenario` adaptive derivation, with `p = 65537` (Fermat F4). |
