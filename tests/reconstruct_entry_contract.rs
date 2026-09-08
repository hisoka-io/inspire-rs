//! `reconstruct_entry`'s zero-fill on short column input is a DOCUMENTED
//! contract, not a bug: the function silently zero-pads every byte past the
//! provided columns, which is fine at this layer and lethal one layer up -
//! which is exactly WHY extract's response-length refusal exists (see
//! tests/extract_response_length_validation.rs). This pin stops a future
//! reader from "fixing" the zero-fill in encode_db and stops the extract
//! refusal from being deleted as redundant.

use proptest::prelude::*;
use raven_inspire::pir::reconstruct_entry;

/// The 16-bit little-endian column split reconstruct_entry inverts.
fn split_columns(entry: &[u8]) -> Vec<u64> {
    entry
        .chunks(2)
        .map(|pair| {
            let low = pair[0] as u64;
            let high = pair.get(1).copied().unwrap_or(0) as u64;
            low | (high << 8)
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    /// (a) With enough columns (ceil(entry_size/2), extras ignored), the
    /// reconstruction inverts the split exactly.
    #[test]
    fn full_column_input_round_trips_the_split(
        entry in prop::collection::vec(any::<u8>(), 1..=64),
        extra in prop::collection::vec(0u64..=u16::MAX as u64, 0..4),
    ) {
        let entry_size = entry.len();
        let mut columns = split_columns(&entry);
        columns.extend(extra);
        prop_assert_eq!(
            reconstruct_entry(&columns, entry_size),
            entry,
            "full column input must round-trip the 16-bit LE split"
        );
    }

    /// (b) With FEWER columns than the entry needs, every byte from
    /// 2*columns.len() onward is zero, the provided head is honoured, and no
    /// error or panic is raised - the zero-fill contract.
    #[test]
    fn short_column_input_zero_fills_the_tail_silently(
        entry in prop::collection::vec(any::<u8>(), 4..=64),
        cut in 0usize..=1,
    ) {
        let entry_size = entry.len();
        let full = split_columns(&entry);
        // strictly shorter: drop at least one column
        let keep = full.len().saturating_sub(1 + cut);
        let short = &full[..keep];

        let out = reconstruct_entry(short, entry_size);
        prop_assert_eq!(out.len(), entry_size, "output length is entry_size regardless");

        let head_bytes = 2 * keep;
        for (i, b) in out.iter().enumerate() {
            if i < head_bytes.min(entry_size) {
                prop_assert_eq!(*b, entry[i], "provided columns must be honoured at byte {}", i);
            } else {
                prop_assert_eq!(
                    *b,
                    0,
                    "byte {} past the provided columns must be ZERO-filled (documented contract)",
                    i
                );
            }
        }
    }
}
