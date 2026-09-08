//! `GeneratorPowers::try_new` totality over ring dimensions: every `d` that is
//! not a power of two (zero included) must surface as a typed
//! `PackParamsError` naming `d` - never an abort - and every power of two must
//! build a table satisfying the algebraic inverse law. The old file argued
//! about 5-invertibility mod 2d, but the implementation rejects every
//! non-power-of-two through the same guard with the same variant, so the nine
//! hand-picked rejects collapsed into this exhaustive sweep (2026-09-06).

use raven_inspire::inspiring::{GeneratorPowers, PackParamsError};

/// Full enumeration beats sampling here: try_new is O(d) and the whole
/// 0..=4096 range costs milliseconds, so totality is proven, not sampled.
#[test]
fn try_new_totality_over_ring_dims_0_to_4096() {
    for d in 0usize..=4096 {
        match GeneratorPowers::try_new(d) {
            Err(PackParamsError::RingDimNotPowerOfTwo { ring_dim }) => {
                assert!(
                    d == 0 || !d.is_power_of_two(),
                    "legal power-of-two d={d} was rejected"
                );
                assert_eq!(ring_dim, d, "the error must name the offending dimension");
            }
            Err(other) => panic!("d={d}: unexpected error variant {other:?}"),
            Ok(table) => {
                assert!(
                    d.is_power_of_two(),
                    "non-power-of-two d={d} built a generator table silently"
                );
                assert_eq!(table.order(), d);
                assert_eq!(table.pow(0), 1);
                // below d=4, 2d <= 5 wraps the generator itself
                if d >= 4 {
                    assert_eq!(table.pow(1), 5);
                }
                // g^i * g^{-i} == 1 mod 2d for every i - a wrong modular
                // inverse is a silent-wrong-permutation defect, not a crash.
                let two_d = 2 * d;
                for i in 0..d {
                    assert_eq!(
                        (table.pow(i) * table.inv_pow(i)) % two_d,
                        1,
                        "inverse law failed at d={d}, i={i}"
                    );
                }
            }
        }
    }
}
