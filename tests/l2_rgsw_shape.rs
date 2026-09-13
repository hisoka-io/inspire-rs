#![allow(
    clippy::expect_used,
    reason = "test-target assertions; an abort is the failure report"
)]

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use raven_inspire::math::{GaussianSampler, Poly};
use raven_inspire::params::InspireParams;
use raven_inspire::rgsw::{GadgetVector, SeededRgswCiphertext};
use raven_inspire::rlwe::RlweSecretKey;

#[test]
fn seeded_rgsw_uses_one_row_per_gadget_digit_and_exact_wire_bytes() {
    let params = InspireParams::secure_128_d2048();
    let ctx = params.ntt_context();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0x1204);
    let secret_key = RlweSecretKey::generate(&params, &mut sampler);
    let message = Poly::constant_moduli(1, params.ring_dim, params.moduli());
    let gadget = GadgetVector::new(params.gadget_base, params.gadget_len, params.q);
    let mut row_rng = ChaCha20Rng::seed_from_u64(0x1205);

    let seeded = SeededRgswCiphertext::encrypt_with_rng(
        &secret_key,
        &message,
        &gadget,
        &mut sampler,
        &ctx,
        &mut row_rng,
    );

    assert_eq!(seeded.rows.len(), params.gadget_len);
    assert_eq!(seeded.expand().rows.len(), params.gadget_len);
    assert_eq!(
        bincode::serialize(&seeded)
            .expect("serialize seeded RGSW")
            .len(),
        49_427
    );
}
