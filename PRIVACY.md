# Privacy Properties of InsPIRe

This document describes the privacy guarantees, known limitations, and threat model
of the InsPIRe PIR library.

## Overview

InsPIRe implements a single-server Private Information Retrieval (PIR) protocol
based on ring-LWE encryption. The goal is to allow a client to retrieve an entry
from a server-hosted database without the server learning which entry was
requested.

**This library contains no telemetry, analytics, tracking, or outbound network
calls from library code.** The only network communication occurs in the
client/server binaries (`inspire-client`, `inspire-server`) over user-configured
HTTP endpoints.

## Privacy Guarantees

### Query Privacy (Computational)

The server cannot determine which *index within a shard* the client is querying,
assuming the hardness of ring-LWE with the chosen parameters (121.5 bits at the
shipped modulus, see Known Limitation 6). The query is an RLWE encryption of a
scaled inverse monomial; without the secret key, the server gains no information about
the local index.

This guarantee stops at the shard boundary. `shard_id` travels in cleartext and
the shard is capped at `ring_dim` entries, so the anonymity set is at most 2048
entries at the shipped preset. See Known Limitation 1.

### Constant-Size Communication

Query and response sizes are constant regardless of the queried index:

- **Registered query**: 15,491 bytes at d=2048
- **Served response**: 10,446 bytes at a 512-byte record width, mod-switched to a 36-bit modulus and serialized at 36 bits per coefficient (17,358 bytes unswitched)
- **Served response with sibling addendum**: 10,606 bytes after the separate 160-byte addendum a batch slot carries
- **RIMS codec**: 11,543 bytes at 36 bits, byte-aligned and not on the wire; the tight serializer is 1,097 bytes smaller

This prevents traffic-analysis attacks that could otherwise infer the queried
index from message sizes.

### Bounded Client-Specific Server State

The server stores bounded, expiring per-client packing-key sessions. Reusing a
session handle links that client's queries within the TTL window; the CRS and
encoded database remain generic and shared. Transport anonymity requires a
separate network layer such as Tor or a VPN.

### Packed Response a-Side Analysis

At d=256 and gamma=4, a bounded experiment reproduced the packed response's `a`
polynomial but failed to recover either a dense column or three independently
generated one-sparse targets. Failed bounded attacks are not privacy evidence.
The experiment covers few deterministic keys and queries and does not establish
a production d=2048 security bound.

## Known Limitations

### 1. Shard ID Sent in Cleartext

The `shard_id` field in `ClientQuery` is sent unencrypted (`src/pir/query.rs:128`).
This reveals which shard holds the target entry, so the anonymity set is one
shard, not the database.

**A shard is hard-capped at `ring_dim` entries.** `encode_db` rejects any
`ShardConfig` whose `entries_per_shard` exceeds `params.ring_dim`
(`src/pir/encode_db.rs:125`), because InspiRING packing places one entry per ring
coefficient. The shipped preset `InspireParams::secure_128_d2048` sets
`ring_dim = 2048` (`src/params.rs:213`, `src/params.rs:230`), so the largest
encodable shard, and therefore the largest achievable anonymity set, is **2048
entries**. The preset is dimensioned for deployments that size
`entries_per_shard` at that cap, so an operator running it as shipped is at the
bound rather than comfortably inside it.

**Impact**: An observer of a single query learns the target is one of at most
2048 entries in the named shard. Over a database of N entries that is a
1-in-`ceil(N / 2048)` partition, not a 1-in-shard-count partition of a large
shard. Repeated queries compound it: the shard-access pattern across a session
is fully visible and the protocol does nothing to hide it.

**`ShardConfig::for_flat_db` presets are not achievable anonymity sets.** That
constructor hardcodes `shard_size_bytes = 1 GB` (`src/params.rs:959`), which at
32-byte entries computes 33,554,432 entries per shard. That is 16384x the
`ring_dim` cap and `encode_db` rejects it with a typed error before any query is
served. Only configurations satisfying
`shard_size_bytes / entry_size_bytes <= ring_dim` encode at all.

**Mitigation**: None in-protocol. `InspireParams::secure_128_d4096`
(`src/params.rs:266`) raises the cap to 4096 entries at higher per-query cost.
Hiding shard identity would need multi-shard querying or oblivious shard
selection; neither is implemented.

### 2. Server Response Includes Processing Time

The `processing_time_ms` field in server responses exposes wall-clock query
processing time. In theory, processing time could vary based on memory access
patterns (e.g., cache behavior for specific shard/index combinations).

**Impact**: Low in practice — the dominant cost is polynomial multiplication,
which is data-independent. Memory-mapped mode (`--mmap`) may show more variance
due to page faults.

**Mitigation**: Deployments concerned about timing side-channels can strip this
field via a reverse proxy, or add artificial jitter.

### 3. No TLS Built In

The server binds a plain TCP listener with no TLS support. All communication
(queries, responses, CRS) travels unencrypted unless a TLS-terminating reverse
proxy is placed in front.

**Impact**: Without TLS, a network observer can read queries and responses. While
the PIR ciphertexts protect the queried index, the returned entry value is
also encrypted (RLWE) and cannot be decrypted without the client's secret key.

**Mitigation**: Deploy behind a TLS-terminating reverse proxy (nginx, Caddy, etc.)
for any non-local deployment.

### 4. Secret Key Stored as Plaintext JSON

The `setup` binary writes `secret_key.json` as a plain JSON file with default
filesystem permissions. No encryption-at-rest or restrictive file modes are
applied.

**Impact**: Any process or user with read access to the output directory can
recover the RLWE secret key and decrypt past responses.

**Mitigation**: Operators should restrict file permissions on `secret_key.json`
(e.g., `chmod 600`) and store it on encrypted storage.

### 5. Client Logs Query Indices to stdout

The client binary prints the queried index, shard ID, and result to stdout.
This is local-only but relevant in shared-machine or logging-forwarding
scenarios.

**Impact**: Anyone with access to the client's terminal output or log files
learns which indices were queried.

**Mitigation**: Redirect stdout in automated deployments. The library API
(`inspire::pir`) does not perform any logging — only the CLI binaries do.

### 6. Shipped Parameters Measure 121.5 Bits, Not 128

`InspireParams::secure_128_d2048` is named for a 128-bit target it does not
reach. Measured with malb/lattice-estimator @ 3e48ef4 under Sage, the binding
attack is `primal_bdd` and the level is **121.5 bits**.

The cause is the modulus. This crate ships `DEFAULT_Q = 2^60 - 2^14 + 1`
(`src/math/mod_q.rs`), chosen because the upstream 2-CRT form (q ~ 2^55.89)
exhausted the noise budget at 256 B records and decryption scrambled silently.
Security falls as q grows at fixed `ring_dim` and `sigma`: log2 q = 57 is the
largest modulus clearing 128 bits (128.8), and log2 q = 58 already measures
126.3. The preset name predates that measurement.

`security_level: SecurityLevel::Bits128` does not contradict this at runtime
because nothing reads it: `InspireParams::validate` checks structural, NTT and
sigma invariants and runs no lattice estimate. The field is a declared target,
not a verified property.

**Impact**: query privacy rests on 121.5-bit ring-LWE hardness, not 128. That is
above every near-term practical attack and below the level the preset name and
the pre-correction documentation advertised. Deployments with a hard 128-bit
floor must not use this preset as shipped.

**Mitigation**: none applied. Reducing q to log2 q = 57 would clear 128 bits and
reopen the noise failure this modulus was raised to fix, so the two constraints
are in direct tension and the tradeoff is unresolved.

## Threat Model

### In Scope

- **Honest-but-curious server**: The server follows the protocol but attempts to
  learn which entry the client queries. InsPIRe provides computational query
  privacy against this adversary.
- **Network eavesdropper** (with TLS): Observes encrypted traffic. Constant-size
  messages prevent traffic analysis.
- **Network eavesdropper** (without TLS): Can read ciphertexts but cannot
  determine the queried index without the secret key. The response value is
  also encrypted and cannot be read without that secret key.

### Out of Scope

- **Malicious server**: A server that deviates from the protocol (e.g., returns
  crafted responses to fingerprint clients) is not covered.
- **Side-channel attacks**: Physical or microarchitectural side channels on the
  server hardware are not addressed.
- **Client compromise**: If the client machine is compromised, the attacker has
  access to the secret key and all query/response data.
- **Structured-database inference by a client**: Bounded sparse and dense recovery
  attempts failed. Failure is not a security bound; stronger and production-scale
  recovery remain uncharacterized.

## Data Handling Summary

| Data | Where | Encrypted | Notes |
|------|-------|-----------|-------|
| Local index within shard | Client → Server | Yes (RLWE) | Computationally hidden |
| Shard ID | Client → Server | No | Anonymity set is one shard, capped at `ring_dim` (2048 at the shipped preset) |
| Response entry | Server → Client | Yes (RLWE) | Decrypted client-side |
| Processing time | Server → Client | No | Potential timing side-channel |
| Secret key | Client filesystem | No | Plaintext JSON |
| CRS | Server → Client | No | Public parameters, no secret data |
