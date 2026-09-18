# InsPIRe Communication Cost Analysis

## Overview

This document analyzes the communication costs of InsPIRe PIR for Ethereum state queries.

**Important**: InsPIRe communication is **O(d)** where d is the ring dimension, not O(√N). The costs below are essentially independent of database size.

## Measured Communication (d=2048)

Current exact binary sizes at the 512-byte production record width:

| Component | Bytes | Notes |
|-----------|------:|-------|
| First query | 61,735 | One seeded fold row plus inline packing keys |
| Registered query | 15,491 | One seeded fold row plus session handle |
| Current tight response | 17,358 | Packed response with lossless 60-bit coefficients |
| Architecture served response | 17,518 | Current response plus the reserved 160-byte sibling addendum |
| Architecture registered round trip | **33,009** | Warm query plus served response |

The pre-W4 implementation used a three-row seeded query of 49,445 bytes and an
18,510-byte packed response. Older 96/192/544 KB figures described unoptimized
RGSW and no-packing shapes; they are historical and are not current wire sizes.

The live TwoPacking path combines the columns into one RLWE response and packs
each canonical coefficient at the modulus's exact 60-bit width. The smaller RIMS v2 codec is tested behind the default-off
`mod-switch-response` feature but is not wired into the adapter transport.

## CRS (Common Reference String) Overhead

The CRS is shared once and reused across queries.

### Conceptual Key Material (Algorithm)

| Approach | KS Matrices | Conceptual Storage |
|----------|-------------|-------------------|
| Tree Packing | log(d) = 11 | 11 × 96 KB = 1056 KB |
| InspiRING | 2 (seeds only) | 64 bytes |
| **Reduction** | **5.5x** | **16,000x** |

### Historical ServerCrs Size (pre-shrink)

> **SUPERSEDED 2026-09-12:** the table and note below describe the former CRS layout and are
> retained as history. `crs_a_vectors` and four other unread fields were removed; the current
> client-shipped CRS is ~1.1 MiB. Do not use the ~40-50 MB total as a current estimate.

| Component | Size (d=2048) | Purpose |
|-----------|---------------|---------|
| `crs_a_vectors` (d×d) | ~33 MB | Query verification / expansion |
| Galois keys (tree packing) | ~1 MB | Automorphisms τ_g |
| Key-switching matrices (k_g, k_h) | ~200 KB | InspiRING automorphisms |
| InspiRING precomputation | ~2-5 MB | Offline packing data |
| Metadata + seeds | <1 KB | Parameters, seeds |
| **Total ServerCrs** | **~40-50 MB** | Full CRS for d=2048 |

**Historical note**: The former implementation stored `crs_a_vectors` (d random a-vectors, d
coefficients each) in the CRS, which dominated storage. The 64-byte figure refers only to the
conceptual InspiRING packing-key seeds.

### Packing Approach in HTTP Server

The HTTP server defaults to **InspiRING** packing when clients include `ClientPackingKeys` (compact `y_body` form). Tree packing (`respond_one_packing` / `respond_mmap_one_packing`) is opt-in via `packing_mode=tree`. The client uses `extract_inspiring(...)` for InspiRING responses or `extract_with_variant(..., InspireVariant::OnePacking)` for tree packing.

InspiRING 2-matrix packing (`respond_inspiring`) is also implemented and can be enabled over the network by including `ClientPackingKeys` (compact `y_body` form) in the query. InspiRING is the default; if packing keys are missing, the server returns an error unless the client explicitly sets `packing_mode=tree`.

## Why PIR Sizes Are Constant

A common question: why do different database sizes produce identical query and response sizes?

**This is a fundamental privacy requirement.** If sizes varied with the target index or database, an observer could infer what's being queried just from traffic analysis.

### Query Size Formula

The query is one seeded RLWE row encrypting `delta * X^(-k)`:

```
Query coefficient payload = ceil(d * 60 / 8) bytes + 32-byte seed

Where:
  d = ring dimension (2048)
  60 = exact bit width of DEFAULT_Q coefficients

The production query with a session handle is 15,491 bytes. Before the session
handshake it also carries 46,252 bytes of packing keys, for 61,735 bytes.
```

**What affects query size:**
| Factor | Effect |
|--------|--------|
| Ring dimension (d) | Linear scaling |
| Coefficient size | Linear scaling |

**What does NOT affect query size:**
| Factor | Why Not |
|--------|---------|
| Database size | Index k only changes polynomial coefficients, not structure |
| Target index | Same seeded RLWE structure regardless of which entry |
| Number of shards | Shard ID is metadata, not ciphertext size |

### Sharding Tradeoffs (Implementation Note)

Our implementation keeps query size constant by sharding the database and sending `shard_id`
in the clear. This is practical for large databases but **weakens privacy granularity**:
an observer can learn which shard (range) is accessed. Sharding also adds storage/IO overhead
and preprocessing for many shard files, and small databases still pay the full query size
chosen by fixed parameters.

### Response Size Formula

The current tight response stores the full `a` polynomial and the
plaintext-bearing prefix of `b` at the exact modulus width:

```
Current bytes = ceil(d * 60 / 8) + ceil(gamma * 60 / 8) + 78
              = 15,360 + 1,920 + 78
              = 17,358 at d = 2048 and gamma = 256
```

The default-off RIMS v2 target switches to 45 bits and stores each coefficient
in six whole bytes:

```
RIMS v2 bytes = 23-byte header + (d + gamma) * ceil(45 / 8)

Where:
  d = ring dimension (2048)
  gamma = ceil(entry_size / 2), 256 for a 512-byte record
  45 = checked modulus-switch target bits

The feature-gated codec emits 23 + (2048 + 256) * 6 = 13,847 bytes. The
architecture's separate 160-byte sibling addendum raises the target served
response to 14,007 bytes; that addendum is not part of the RIMS codec. Until
the adapter wires this codec, the current tight response is 17,358 bytes.
```

**What affects response size:**
| Factor | Effect |
|--------|--------|
| Entry size | More columns → more ciphertexts |
| Ring dimension (d) | Linear scaling |
| Modulus-switch target | Linear in retained coefficient width |

**What does NOT affect response size:**
| Factor | Why Not |
|--------|---------|
| Database size | Same entry format regardless of DB size |
| Which entry retrieved | Ciphertext structure is identical |
| Number of entries | Server processes one shard, returns same format |

### Database Size Effect

| Database Size | Shards | Registered Query | 512 B Response | Server Time |
|---------------|--------|---------------------|---------------|-------------|
| 1K entries | 1 | 15,491 B | 17,358 B | ~1 ms |
| 64K entries | 32 | 15,491 B | 17,358 B | ~1.5 ms |
| 1M entries | 512 | 15,491 B | 17,358 B | ~3 ms |
| 100M entries | 50K | 15,491 B | 17,358 B | ~3 ms |

**The only thing that changes is server computation time** (selecting and processing the correct shard).

### Visual Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    WHY PIR SIZES ARE CONSTANT                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  QUERY (15,491 B after handshake)                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  RLWE(delta*X^(-k))                                     │   │
│  │  ├── one seeded row: b polynomial plus 32-byte a seed  │   │
│  │  └── Structure fixed by d, not by k or DB size         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  CURRENT TIGHT RESPONSE (17,358 B at 512-byte record width)     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Packed ServerResponse                                   │   │
│  │  ├── full a polynomial + 256-coefficient b prefix       │   │
│  │  ├── lossless 60-bit coefficients                       │   │
│  │  └── Structure fixed by (d, entry_size), not DB size   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  DATABASE SIZE only affects:                                    │
│  ├── Number of shards (more entries = more shards)             │
│  ├── Server computation (which shard to process)               │
│  └── NOT bandwidth (privacy would leak otherwise!)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Measured Performance

Benchmarked on AMD/Intel x64 server:

### Server Response Time

| Database Size | Shards | Respond Time |
|---------------|--------|--------------|
| 256K entries (8 MB) | 128 | 3.8 ms |
| 512K entries (16 MB) | 256 | 3.1 ms |
| 1M entries (32 MB) | 512 | 3.3 ms |

### End-to-End Latency

| Phase | Time |
|-------|------|
| Client: Query generation (seeded) | ~4 ms |
| Server: Expand + Respond | ~3-4 ms |
| Client: Extract result | ~5 ms |
| **Total round-trip** | **~12 ms** |

### Historical Request/Response Sizes (Superseded)

Measured on 2026-01-12 using `benches/query_size_latency.rs` with `INSPIRE_BENCH_SIZES_ONLY=1`.
These pre-one-row values are retained as benchmark history and are not current
wire or capacity-planning numbers.

| Variant | Request (bincode) | Response (bincode) | Request (JSON) | Response (JSON) | Notes |
|---------|-------------------|--------------------|----------------|-----------------|-------|
| InsPIRe^0 (NoPacking) | 384.7 KB | 1089.9 KB | 461.1 KB | 1305.9 KB | Full query + per-column response |
| InsPIRe^1 (OnePacking) | 480.9 KB | 64.1 KB | 576.4 KB | 76.8 KB | Full query + packed response (InspiRING) |
| InsPIRe^2 (TwoPacking) | 288.7 KB | 64.1 KB | 346.5 KB | 76.9 KB | Seeded query + packed response (InspiRING) |

## Modulus Switching Status

The default-off RIMS v2 codec applies a checked 45-bit modulus switch and
prefix truncation to packed responses. Its production-cell regression runs the
real respond, switch, encode, decode, and extract path and pins 13,847 codec bytes.
It is not wired into the adapter transport. The exported 33-bit target remains
rejected by the conservative noise gate.

## Why Generic Compression Won't Help

LWE/RLWE ciphertexts are cryptographically pseudorandom (indistinguishable from uniform random by design):

| Data Type | Entropy | Compression |
|-----------|---------|-------------|
| Random bytes | ~8 bits/byte | ~0% reduction |
| CRS (key material) | ~8 bits/byte | ~0-2% reduction |
| Query ciphertexts | ~8 bits/byte | ~0-2% reduction |

## Historical Ethereum Example (Superseded)

The scenario below came from the upstream application-specific prototype. Its
lookup model is retained as historical context, but its old JSON byte totals are
not current Raven measurements. Raven now exposes generic PIR and its adapters
define their own record shapes.

### Scenario: User opens wallet with 10 tokens, 3 NFTs, ETH balance

#### Data Requirements

| Asset Type | Count | Query Type | DB Lookups |
|------------|-------|------------|------------|
| ETH balance | 1 | Account | 1 (returns 96B) |
| ERC-20 tokens | 10 | Storage | 10 (each 32B) |
| NFTs (ERC-721) | 3 | Storage | 3 (each 32B) |
| **Total** | | | **14 queries** |

Actual payload needed: 96 + 13×32 = **512 bytes**

#### Former Communication Estimate

| Scenario | Upload | Download | Total |
|----------|--------|----------|-------|
| 14 queries (standard) | 6.4 MB | 18.1 MB | **24.5 MB** |
| 14 queries (seeded) | 3.2 MB | 18.1 MB | **21.3 MB** |

#### Historical Expectations

- These numbers predate the one-row query, response prefix truncation, binary
  HTTP wire, and session handshake.
- Do not use them for capacity planning.

## Optimization Strategies

### 1. Prefetch Common Data
- Cache CRS on app install (~100 KB)
- Background-fetch token balances during idle time

### 2. Batch by Access Pattern
- Queue all lookups before sending
- Group by shard when possible

### 3. Incremental Updates
- Subscribe to block updates
- Only re-query changed state (via state diffs)

### 4. Hybrid Privacy Tiers
- Use PIR for sensitive queries (balances, specific NFTs)
- Use public RPC for non-sensitive metadata (token names, decimals)

## Summary

| Current d=2048, 512-byte cell metric | Bytes |
|--------------------------------------|------:|
| First query with inline packing keys | 61,735 |
| Registered query | 15,491 |
| Current tight response | 17,358 |
| Architecture served response with reserved addendum | 17,518 |
| Architecture registered round trip | 33,009 |
| Default-off, unwired RIMS v2 codec target | 13,847 |
| RIMS served target with reserved addendum | 14,007 |

These sizes are constant with respect to the target index and database size.
The cleartext shard id limits query anonymity to one shard; communication-size
constancy does not enlarge that set.

## Interactive Visualization

For an interactive visualization of these costs with animated protocol flow, parameter sliders, and size breakdowns, see [protocol-visualization.html](protocol-visualization.html).
