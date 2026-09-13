#![allow(clippy::expect_used)]

use proptest::prelude::*;
use raven_inspire::num_columns;

#[test]
fn num_columns_pins_zero_odd_even_and_production_widths() {
    for (entry_size, expected) in [(0, 1), (1, 1), (2, 1), (3, 2), (32, 16), (512, 256)] {
        assert_eq!(num_columns(entry_size), expected, "entry_size {entry_size}");
    }
}

proptest! {
    #[test]
    fn num_columns_matches_an_independent_integer_oracle(entry_size in any::<usize>()) {
        let expected = (entry_size / 2 + entry_size % 2).max(1);
        prop_assert_eq!(num_columns(entry_size), expected);
    }
}
