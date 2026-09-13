# Google InsPIRe Alignment Notes

This document summarizes how inspire-rs aligns with the Google InsPIRe
reference implementation (research/InsPIRe) and highlights known
incompatibilities. It is intended as a practical guide for cross-checking
experiments and scoping alignment work.

## Parameter Mapping (High-Level)

> **CORRECTED 2026-09-12:** Google retains the two-modulus reference shape. The older inspire-rs
> column and defaults below also showed that pair; current Raven secure presets instead use one
> `DEFAULT_Q = 2^60 - 2^14 + 1` limb and `p = 65537`.

| Concept | Google InsPIRe (research/InsPIRe) | inspire-rs |
|--------|----------------------------------|-----------|
| Ring dimension | `d` / `poly_len` | `ring_dim` |
| Ciphertext modulus | CRT moduli list | single `DEFAULT_Q` limb by default; explicit two-CRT optional |
| Plaintext modulus | `p` | `p` |
| Gadget base | `B` | `gadget_base` |
| Gadget length | `l_gsw` | `gadget_len` |

## What we can align

- **Ring dimension**: both default to 2048.
- **Noise width (sigma)**: both default to ~6.4.
- **High-level protocol shape**: seeded queries plus InspiRING packing for InsPIRe^2.
- **Evaluation methodology**: compare sizes/latency using similar workloads.

## What is not directly comparable

- **Param selection pipeline**:
  - Google derives params via Spiral-specific routines with `q2_bits = 28`,
    `t_exp_left = 3`, and `p = 2^15` or `2^16`.
  - inspire-rs uses fixed “secure_128” parameters (ring_dim=2048, p=65537).

## Current inspire-rs defaults (reference)

- ring_dim: 2048
- crt_moduli: [`DEFAULT_Q`]
- q: `2^60 - 2^14 + 1`
- p: 65537
- sigma: 6.4
- gadget_base: 2^20
- gadget_len: 3

## Google InsPIRe defaults (reference)

From `research/InsPIRe/src/params.rs`:

- poly_len: 2048
- moduli: [268369921, 249561089]
- noise_width: 6.4
- q2_bits: 28
- t_exp_left: 3
- p: 2^15 or 2^16 (depends on scenario)

## Alignment gaps to track

1. **Evaluation parity**: match workloads + report comparable metrics.
2. **CRT vs single-modulus sizing**: size comparisons should note which mode was serialized.

## Practical Guidance

- Use `query_seeded()` + InspiRING packing (`respond_seeded_inspiring`) for InsPIRe^2 alignment.
