//! `y_all` and `y_all_ntt` are `#[serde(skip)]`, so a deserialized server sees
//! only `y_body` and must re-derive them to reach the fully-NTT dispatch. The
//! derivation must be byte-identical to what the client computed in-process.

use raven_inspire::inspiring::{ClientPackingKeys, PackParams};
use raven_inspire::math::{GaussianSampler, NttContext};
use raven_inspire::params::InspireParams;
use raven_inspire::rlwe::RlweSecretKey;

fn test_params() -> InspireParams {
    // same automorph and key-switch ladder as the production cell, smaller ring
    InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: raven_inspire::params::SecurityLevel::Bits128,
    }
}

// A hand-constructed-wire twin of the bincode round-trip test below lived
// here; it reached the same ensure_server_derivatives state by clearing fields
// manually. Its stronger per-poly assertions (y_all coefficients and the
// is_ntt flags) moved into the round-trip test (2026-09-06), which models the
// wire with a real bincode round trip; the NTT-transform-skip mutant that
// killed both kills the merged survivor.

// A "noop when populated" test lived here; the early-return guard it named is
// a pure perf shortcut with no observable behaviour (the recompute is discarded
// by independently guarded write-backs), so deleting the guard left the test
// green (2026-09-06 mutation audit).

#[test]
fn bincode_roundtrip_then_server_derive_matches_original_y_all_ntt() {
    let params = test_params();
    let num_to_pack = 16;
    let pack_params =
        PackParams::try_new(&params, num_to_pack).expect("num_to_pack must be a legal width");
    let ctx = NttContext::with_moduli(params.ring_dim, params.moduli());

    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let rlwe_sk = RlweSecretKey::generate(&params, &mut sampler);
    let original = ClientPackingKeys::generate(&rlwe_sk, &pack_params, [42u8; 32], &mut sampler);

    let bytes = bincode::serialize(&original).expect("bincode serialize");
    let mut wire: ClientPackingKeys = bincode::deserialize(&bytes).expect("bincode deserialize");
    assert!(wire.y_all.is_empty(), "serde(skip) drops y_all on the wire");
    assert!(
        wire.y_all_ntt.is_empty(),
        "serde(skip) drops y_all_ntt on the wire"
    );

    wire.ensure_server_derivatives(&pack_params, &ctx);

    assert_eq!(wire.y_all.len(), original.y_all.len());
    assert_eq!(wire.y_all_ntt.len(), original.y_all_ntt.len());
    for (i, (got, want)) in wire.y_all.iter().zip(original.y_all.iter()).enumerate() {
        assert_eq!(got.len(), want.len(), "y_all[{i}] inner length mismatch");
        for (k, (pg, pw)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(pg.coeffs(), pw.coeffs(), "y_all[{i}][{k}] coeffs mismatch");
            assert_eq!(
                pg.is_ntt(),
                pw.is_ntt(),
                "y_all[{i}][{k}] is_ntt flag mismatch"
            );
        }
    }
    for (i, (got, want)) in wire
        .y_all_ntt
        .iter()
        .zip(original.y_all_ntt.iter())
        .enumerate()
    {
        assert_eq!(
            got.len(),
            want.len(),
            "y_all_ntt[{i}] inner length mismatch"
        );
        for (k, (pg, pw)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                pg.coeffs(),
                pw.coeffs(),
                "post-roundtrip y_all_ntt[{i}][{k}] mismatch"
            );
            assert!(pg.is_ntt(), "y_all_ntt[{i}][{k}] must be in NTT form");
            assert!(
                pw.is_ntt(),
                "expected y_all_ntt[{i}][{k}] must be in NTT form"
            );
        }
    }
}
