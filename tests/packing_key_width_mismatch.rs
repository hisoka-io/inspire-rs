//! Client packing keys built at one gamma must not register against a server using
//! another. The automorphism generator is `(2n/gamma)+1`, so mismatched keys are
//! rotations under a different automorphism than the `bold_t` they multiply - and
//! `packing_online` guards its loop with `if i < len`, skipping the shortfall instead
//! of rejecting it. The result is HTTP 200 carrying wrong plaintext.

#![allow(
    clippy::expect_used,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]

use raven_inspire::inspiring::inspiring2::{ClientPackingKeys, PackParams};
use raven_inspire::math::{GaussianSampler, NttContext};
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::pir::ServerSessionStore;
use raven_inspire::rlwe::RlweSecretKey;

const RING_DIM: usize = 256;

fn params() -> InspireParams {
    InspireParams {
        ring_dim: RING_DIM,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65536,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

fn keys_at(width: usize) -> (ClientPackingKeys, PackParams) {
    let p = params();
    let pack = PackParams::try_new(&p, width).expect("legal width");
    let mut sampler = GaussianSampler::with_seed(p.sigma, 7);
    let sk = RlweSecretKey::generate(&p, &mut sampler);
    let keys = ClientPackingKeys::generate(&sk, &pack, [3u8; 32], &mut sampler);
    (keys, pack)
}

// Two single-cell tests lived here ((8,8) accept and (4,8) refuse); both are
// cells of the 5x5 table below, which since 2026-09-06 also asserts the
// refusal message on every mismatched pair (the reworded-message mutant that
// killed the deleted (4,8) test kills the table now).

/// Every legal width pairing must be screened, not just the one above.
#[test]
fn every_mismatched_legal_width_pair_is_refused() {
    for client_width in [2usize, 4, 8, 16, 32] {
        for server_width in [2usize, 4, 8, 16, 32] {
            let (client_keys, _) = keys_at(client_width);
            let (_unused, server_pack) = keys_at(server_width);
            let ctx = NttContext::new(RING_DIM, server_pack.q);
            let store = ServerSessionStore::new();
            let outcome = store.register_server_side(client_keys, &server_pack, &ctx);
            assert_eq!(
                outcome.is_ok(),
                client_width == server_width,
                "client gamma {client_width} against server gamma {server_width}: \
                 registration must succeed only when the widths agree"
            );
            if let Err(err) = outcome {
                let msg = format!("{err}");
                assert!(
                    msg.contains("width mismatch"),
                    "the refusal must name the mismatch; got: {msg}"
                );
                assert!(
                    msg.contains(&client_width.to_string())
                        && msg.contains(&server_width.to_string()),
                    "the refusal must name both widths; got: {msg}"
                );
            }
        }
    }
}

/// The inlined-key path is a SECOND place the two widths meet. `respond_inspiring` takes
/// the keys straight off the query and never goes through `register_server_side`, so a
/// guard placed only there would leave the wrong-bytes-as-success failure fully reachable.
#[test]
fn respond_refuses_inlined_keys_from_a_different_width() {
    use raven_inspire::pir::{query, setup};

    let p = params();
    let mut sampler = GaussianSampler::with_seed(p.sigma, 11);
    let entry_size = 32usize;
    let database = vec![7u8; RING_DIM * entry_size];
    let (crs, encoded_db, sk) = setup(&p, &database, entry_size, &mut sampler).expect("setup");

    let (_state, mut query) = query(&crs, 0, &encoded_db.config, &sk, &mut sampler).expect("query");
    let keys = query
        .inspiring_packing_keys
        .as_mut()
        .expect("the InspiRING path inlines packing keys");
    let honest_width = keys.num_to_pack;
    // Any other legal width; the automorphism generator differs, so the rotation table
    // the server precomputed no longer matches these keys.
    keys.num_to_pack = if honest_width == 8 { 4 } else { 8 };

    let err = raven_inspire::pir::respond_inspiring(&crs, &encoded_db, &query)
        .expect_err("inlined keys at a foreign width must be refused, not packed");
    let msg = format!("{err}");
    assert!(
        msg.contains("width mismatch"),
        "the refusal must name the mismatch; got: {msg}"
    );
}

/// The gamma scalar is a self-reported label; `y_body` is the material the packing sum
/// actually indexes. A client can declare the server's gamma and still send short key
/// material: `generate_rotations` sizes the outer vector from the SERVER's gamma, so the
/// outer bound looks right, while each rotation is only `y_body.len()` long and
/// `packing_online` skips every gadget digit past it. The sum silently loses terms.
#[test]
fn keys_with_truncated_material_are_refused_even_at_the_declared_width() {
    let (mut keys, pack) = keys_at(8);
    assert_eq!(keys.num_to_pack, pack.num_to_pack, "gamma agrees");
    assert_eq!(
        keys.y_body.len(),
        pack.gadget.len,
        "a well-formed key carries one y_body poly per gadget digit"
    );

    keys.y_body
        .pop()
        .expect("drop one gadget digit of key material");

    let ctx = NttContext::new(RING_DIM, pack.q);
    let store = ServerSessionStore::new();
    let err = store
        .register_server_side(keys, &pack, &ctx)
        .expect_err("short key material must be refused even when gamma matches");
    let msg = format!("{err}");
    assert!(
        msg.contains("material mismatch"),
        "the refusal must name the material, not the width; got: {msg}"
    );
}
