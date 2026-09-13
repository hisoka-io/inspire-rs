use proptest::prelude::*;
use raven_inspire::params::rows_per_shard_match_ring_dim;

#[test]
fn rows_per_shard_match_only_the_same_ring_dimension() {
    assert!(rows_per_shard_match_ring_dim(2048, 2048));
    assert!(!rows_per_shard_match_ring_dim(2047, 2048));
    assert!(!rows_per_shard_match_ring_dim(2049, 2048));
}

proptest! {
    #[test]
    fn rows_per_shard_predicate_matches_integer_equality(
        entries_per_shard in any::<u32>(),
        ring_dim in any::<u32>(),
    ) {
        prop_assert_eq!(
            rows_per_shard_match_ring_dim(u64::from(entries_per_shard), ring_dim as usize),
            entries_per_shard == ring_dim,
        );
    }
}
