//! A `p` at or below the two-byte column ceiling wraps encoded values silently.

use raven_inspire::params::InspireParams;

/// Both sides of the ceiling: with every other preset field legal, validate()
/// must reject exactly p <= 65535, walked densely around the boundary and
/// sparsely across the range (65536 clears the ceiling; coprimality with the
/// ring stays opt-in, see the last test).
#[test]
fn validate_p_boundary_is_exact_on_both_sides() {
    let pinned = [2u64, 257, 4097, 32771, 65534, 65535, 65536, 65537, 65538];
    let swept = (1..=64u64).map(|i| i * 4096 + 1);
    for p in pinned.into_iter().chain(swept) {
        let mut params = InspireParams::secure_128_d2048();
        params.p = p;
        let should_reject = p <= 65535;
        let outcome = params.validate();
        assert_eq!(
            outcome.is_err(),
            should_reject,
            "p={p}: validate() disagreed with the column-ceiling predicate"
        );
        if let Err(err) = outcome {
            assert!(err.contains("p must be"), "error must name p, got: {err}");
        }
    }
}

#[test]
fn validate_accepts_the_shipping_plaintext_modulus() {
    let params = InspireParams::secure_128_d2048();
    assert_eq!(params.p, 65537, "preset must stay above the ceiling");
    params.validate().expect("shipping preset must validate");

    let d4096 = InspireParams::secure_128_d4096();
    d4096.validate().expect("d4096 preset must validate");
}

/// Coprimality stays opt-in: existing fixtures use `p = 65536`, which clears
/// the ceiling but shares a factor with the ring.
#[test]
fn the_ceiling_leaves_the_opt_in_coprimality_rule_where_it_was() {
    let mut params = InspireParams::secure_128_d2048();
    params.p = 65536;
    params
        .validate()
        .expect("65536 is above the ceiling; coprimality is opt-in");
    params
        .validate_strict_tree_packed()
        .expect_err("the strict variant still refuses a p sharing a factor with the ring");
}
