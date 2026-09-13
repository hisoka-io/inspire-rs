#![allow(
    clippy::expect_used,
    reason = "test-target assertions; an abort is the failure report"
)]

use raven_inspire::inspiring::ClientPackingKeys;
use raven_inspire::ServerSessionStore;

fn empty_keys() -> ClientPackingKeys {
    ClientPackingKeys {
        y_body: Vec::new(),
        z_body: Vec::new(),
        y_all: Vec::new(),
        y_all_ntt: Vec::new(),
        y_bar_all: Vec::new(),
        y_bar_all_ntt: Vec::new(),
        full_key: false,
        num_to_pack: 1,
    }
}

#[test]
fn a_flushed_handle_is_not_reissued_to_a_new_store() {
    let retired_store = ServerSessionStore::new();
    let stale = retired_store.register(empty_keys()).expect("first handle");
    drop(retired_store);

    let replacement_store = ServerSessionStore::new();
    let live = replacement_store
        .register(empty_keys())
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
        handles.push(store.register(empty_keys()).expect("fresh handle").0);
    }

    assert!(
        handles.windows(2).all(|pair| pair[1] > pair[0]),
        "process-local handles must increase across store replacement: {handles:?}"
    );
}

#[test]
fn removal_updates_occupancy_and_is_idempotent() {
    let store = ServerSessionStore::new();
    let removed_handle = store.register(empty_keys()).expect("removed handle");
    let retained_handle = store.register(empty_keys()).expect("retained handle");
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
