//! Wall time of the whole `PackParams::try_new` build at the production d=2048 cell,
//! median of 5. The O(n^3) automorphism table search used to dominate it; the 2n
//! monomial NTTs now take most of it.

use std::time::Instant;

use raven_inspire::inspiring::PackParams;
use raven_inspire::params::InspireParams;

const RUNS: usize = 5;
const PRODUCTION_WIDTH: usize = 256;

#[test]
#[ignore = "cost: 5 production d=2048 PackParams builds, under 1 s after the table \
            derivation and about 16 s before; trigger: a change to PackParams::build or to the \
            automorphism table derivation"]
fn pack_params_build_median_at_production_cell() {
    let params = InspireParams::secure_128_d2048();
    let mut samples_ms = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let start = Instant::now();
        let pack = PackParams::try_new(&params, PRODUCTION_WIDTH)
            .expect("the production width must construct");
        samples_ms.push(start.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&pack);
    }
    samples_ms.sort_by(f64::total_cmp);
    eprintln!(
        "PackParams::try_new(secure_128_d2048, {PRODUCTION_WIDTH}): median {:.1} ms over \
         {RUNS} runs, samples {samples_ms:.1?}",
        samples_ms[RUNS / 2]
    );
}
