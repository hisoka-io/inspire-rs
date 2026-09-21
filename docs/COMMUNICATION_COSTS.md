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
| Served response | 10,446 | Packed response mod-switched to 36-bit coefficients; an adapter's 2-byte schema prefix makes 10,448 on the wire |
| Served response with sibling addendum | 10,606 | Served response plus the 160-byte sibling addendum a batch slot carries |
| Registered round trip | **26,097** | Warm query plus served response plus addendum |

The pre-W4 implementation used a three-row seeded query of 49,445 bytes and an
18,510-byte packed response. Unswitched, at the 60-bit source modulus, the
response is 17,358 bytes (17,518 with the addendum, 33,009 per round trip). Older
96/192/544 KB figures described unoptimized RGSW and no-packing shapes; they are
historical and are not current wire sizes.

The served TwoPacking path combines the columns into one RLWE response, mod-switches
it to `MOD_SWITCH_TARGET_36BIT` and packs each canonical coefficient at the switched
modulus's exact 36-bit width. The switch sits behind the `mod-switch-response`
feature, default-off in this crate and enabled by the adapter and client workspaces.
The byte-aligned RIMS codec is not on the wire: it is 11,543 bytes at 36 bits, 1,097
more than the tight serializer.

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

The served response is mod-switched to a 36-bit modulus and stores the full `a`
polynomial and the plaintext-bearing prefix of `b` at the exact switched width:

```
Served bytes = ceil(d * 36 / 8) + ceil(gamma * 36 / 8) + 78
             = 9,216 + 1,152 + 78
             = 10,446 at d = 2048 and gamma = 256

Unswitched, at the 60-bit source modulus, the same layout is
               ceil(d * 60 / 8) + ceil(gamma * 60 / 8) + 78
             = 15,360 + 1,920 + 78
             = 17,358
```

The RIMS codec (`encode_response_packed`) is byte-aligned and stores each
coefficient in whole bytes:

```
RIMS bytes = 23-byte header + (d + gamma) * ceil(bits / 8)

Where:
  d = ring dimension (2048)
  gamma = ceil(entry_size / 2), 256 for a 512-byte record
  bits = modulus-switch target width

At the 45-bit target it emits 23 + (2048 + 256) * 6 = 13,847 bytes; at the served
36-bit target, 23 + (2048 + 256) * 5 = 11,543 bytes. It is not on the wire: the
tight serializer above is 1,097 bytes smaller at 36 bits. The separate 160-byte
sibling addendum is appended per batch slot and is part of neither codec; it
raises the served response to 10,606 bytes.
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
| 1K entries | 1 | 15,491 B | 10,446 B | ~1 ms |
| 64K entries | 32 | 15,491 B | 10,446 B | ~1.5 ms |
| 1M entries | 512 | 15,491 B | 10,446 B | ~3 ms |
| 100M entries | 50K | 15,491 B | 10,446 B | ~3 ms |

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
│  SERVED RESPONSE (10,446 B at 512-byte record width)            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Packed ServerResponse                                   │   │
│  │  ├── full a polynomial + 256-coefficient b prefix       │   │
│  │  ├── 36-bit mod-switched coefficients                   │   │
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

Every served response is mod-switched to `MOD_SWITCH_TARGET_36BIT =
68_718_428_161 = 2^36 - 2^20 + 1` before it is serialized, and the client
extracts through the mod-switched path. The feature `mod-switch-response` is
default-off in this crate and enabled by the adapter and client workspaces.

The target is a prime, `== 1 (mod 4096)` and `== 33 (mod 65537)`, chosen for the
small residue `q' mod p` rather than for size. Decryption divides by
`floor(q'/p)` while the switched coefficient carries `m * q'/p`, so `q' mod p`
is a deterministic per-coefficient offset, which the noise gate charges in full.
The gate's ratio alone is identical for two primes of one bit width: a seeded KAT
pins error 652 at this prime and 53,503 at the largest 36-bit NTT prime (residue
53,266).

`served_post_switch_noise_distribution` in `benches/packing_noise_measurement.rs`
(`--release --features mod-switch-response -- --ignored`) measures the decode
margin on the switched response after the response serializer, 1,000 samples per
width across 40 sessions. Worst error against a decode boundary of 524,272: 1,113
at 512-byte rows (gamma = 256), 8.88 bits of margin; 417 at 32-byte rows
(gamma = 16), 10.30 bits. A session's packing key puts a fixed offset on every
response of that session, so a margin taken inside one session reads a few tenths
of a bit high.

The modulus arrives off the wire. `extract_inspiring_mod_switched` refuses, before
deriving anything from it, any response modulus that is neither the CRS's own
limbs (unswitched) nor an implemented target (45-bit, 36-bit), so serving a
different rung is a client change as well as a server one.

The production-cell regression
`production_36bit_switched_response_round_trips_through_bincode` runs the real
respond, switch, serialize, deserialize, and extract path and pins 10,446
serialized bytes against 11,543 for the byte-aligned RIMS codec, which is why
RIMS is not on the wire. The 45-bit RIMS regression still pins 13,847 codec
bytes. The exported 33-bit target remains unwired; it fails the conservative
noise gate at 0.833x.

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
| Served response, mod-switched to 36 bits | 10,446 |
| Served response with the 160-byte sibling addendum | 10,606 |
| Registered round trip | 26,097 |
| RIMS codec at 36 bits, not on the wire | 11,543 |
| RIMS codec at 45 bits, not on the wire | 13,847 |

These sizes are constant with respect to the target index and database size.
The cleartext shard id limits query anonymity to one shard; communication-size
constancy does not enlarge that set.

## Interactive Visualization

For an interactive visualization of these costs with animated protocol flow, parameter sliders, and size breakdowns, see [protocol-visualization.html](protocol-visualization.html).
