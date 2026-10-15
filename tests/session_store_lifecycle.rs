#![allow(
    clippy::expect_used,
    reason = "test-target assertions; an abort is the failure report"
)]

use raven_inspire::inspiring::{ClientPackingKeys, PackParams};
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::pir::PackingKeySeat;
use raven_inspire::rlwe::RlweSecretKey;
use raven_inspire::ServerSessionStore;

fn seat() -> PackingKeySeat {
    let params = InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65_537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    };
    let mut sampler = GaussianSampler::with_seed(params.sigma, 3);
    let secret_key = RlweSecretKey::generate(&params, &mut sampler);
    let keys = ClientPackingKeys::generate(&secret_key, &params, 16, [1; 32], &mut sampler)
        .expect("16 is a legal width");
    let pack_params = PackParams::try_new(&params, 16).expect("pack params");
    PackingKeySeat::from_wire(keys, &pack_params, &params.ntt_context()).expect("seat")
}

#[test]
fn a_flushed_handle_is_not_reissued_to_a_new_store() {
    let retired_store = ServerSessionStore::new();
    let stale = retired_store.register(seat()).expect("first handle");
    drop(retired_store);

    let replacement_store = ServerSessionStore::new();
    let live = replacement_store
        .register(seat())
        .expect("replacement handle");

    assert_ne!(
        stale, live,
        "rebuilding the store must not restart handle allocation"
    );
    assert!(
        replacement_store
            .get(stale)
            .expect("stale lookup")
            .is_none(),
        "a stale handle must not resolve to replacement keys"
    );
    assert!(
        replacement_store.get(live).expect("live lookup").is_some(),
        "the replacement handle must resolve"
    );
}

#[test]
fn independent_store_allocations_are_strictly_monotonic() {
    let mut handles = Vec::new();
    for _ in 0..8 {
        let store = ServerSessionStore::new();
        handles.push(store.register(seat()).expect("fresh handle").0);
    }

    assert!(
        handles.windows(2).all(|pair| pair[1] > pair[0]),
        "process-local handles must increase across store replacement: {handles:?}"
    );
}

#[test]
fn removal_updates_occupancy_and_is_idempotent() {
    let store = ServerSessionStore::new();
    let removed_handle = store.register(seat()).expect("removed handle");
    let retained_handle = store.register(seat()).expect("retained handle");
    assert_eq!(store.len(), 2);

    assert!(store.remove(removed_handle).expect("first removal"));
    assert_eq!(store.len(), 1, "removal must release the stored key set");
    assert!(
        store.get(removed_handle).expect("removed lookup").is_none(),
        "removed handles must be stale"
    );
    assert!(
        store
            .get(retained_handle)
            .expect("retained lookup")
            .is_some(),
        "removing one handle must not disturb another"
    );

    assert!(!store.remove(removed_handle).expect("repeated removal"));
    assert_eq!(store.len(), 1, "repeated removal must not change occupancy");
}
