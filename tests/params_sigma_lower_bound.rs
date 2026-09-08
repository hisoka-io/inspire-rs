//! `sigma` below the security floor collapses the LWE error to a constant, which
//! zeroes the secret key and makes the query a deterministic function of the index.

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, MIN_SIGMA};

/// The reason the bound exists: a sub-floor width emits a constant, so the
/// "noise" hiding the query index is not noise at all.
#[test]
fn a_sub_floor_sigma_makes_the_sampler_constant() {
    let mut degenerate = GaussianSampler::with_seed(0.001, 7);
    let drawn = degenerate.sample_vec(512);
    assert!(
        drawn.iter().all(|&x| x == 0),
        "sigma = 0.001 must collapse to a constant; that collapse is what the floor prevents"
    );

    let mut shipping = GaussianSampler::with_seed(6.4, 7);
    let spread = shipping.sample_vec(512);
    assert!(
        spread.iter().any(|&x| x != 0),
        "sigma = 6.4 must produce a non-constant error distribution"
    );
}

/// Both sides of the floor: with every other preset field legal,
/// validate() must reject exactly (!finite || sigma < MIN_SIGMA), and the
/// boundary is walked at adjacent representable values of MIN_SIGMA itself.
#[test]
fn validate_sigma_boundary_is_exact_on_both_sides() {
    let below_floor = f64::from_bits(MIN_SIGMA.to_bits() - 1);
    let pinned = [
        0.0f64,
        0.001,
        1.0,
        3.0,
        -6.4,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        below_floor,
        MIN_SIGMA,
        6.4,
        20.0,
    ];
    let swept = (0..=200).map(|i| i as f64 * 0.05);
    for sigma in pinned.into_iter().chain(swept) {
        let mut params = InspireParams::secure_128_d2048();
        params.sigma = sigma;
        let should_reject = !sigma.is_finite() || sigma < MIN_SIGMA;
        let outcome = params.validate();
        assert_eq!(
            outcome.is_err(),
            should_reject,
            "sigma={sigma:?}: validate() disagreed with the floor predicate"
        );
        if let Err(err) = outcome {
            assert!(
                err.contains("sigma"),
                "the error must name sigma, got: {err}"
            );
        }
    }
}

#[test]
fn validate_accepts_the_shipping_sigma() {
    let params = InspireParams::secure_128_d2048();
    assert_eq!(
        params.sigma.to_bits(),
        6.4f64.to_bits(),
        "preset must stay above the floor"
    );
    assert!(params.sigma >= MIN_SIGMA);
    params.validate().expect("shipping preset must validate");

    InspireParams::secure_128_d4096()
        .validate()
        .expect("d4096 preset must validate");
}
