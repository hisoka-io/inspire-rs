//! RLWE coefficient-zero sample extraction must decrypt under every ClientState LWE key.

use raven_inspire::math::{GaussianSampler, Poly};
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::rlwe::{RlweCiphertext, RlweSecretKey};
use raven_inspire::{query, setup, ClientSession, ClientState, EncodedDatabase, ServerCrs};

type Fixture = (
    InspireParams,
    ServerCrs,
    EncodedDatabase,
    RlweSecretKey,
    GaussianSampler,
);

fn fixture(seed: u64) -> Result<Fixture, Box<dyn std::error::Error>> {
    let params = InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65_536,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    };
    let database = vec![0u8; params.ring_dim * 32];
    let mut sampler = GaussianSampler::with_seed(params.sigma, seed);
    let (crs, encoded_db, rlwe_sk) = setup(&params, &database, 32, &mut sampler)?;
    Ok((params, crs, encoded_db, rlwe_sk, sampler))
}

fn assert_decrypts_coeff0(state: &ClientState, rlwe_sk: &RlweSecretKey, params: &InspireParams) {
    let message = 12_345u64;
    let mut message_coeffs = vec![0u64; params.ring_dim];
    message_coeffs[0] = message;
    let message_poly = Poly::from_coeffs(message_coeffs, params.q);
    let a = Poly::from_coeffs(
        (0..params.ring_dim)
            .map(|index| params.q / 3 + (index as u64 + 1) * 7_919)
            .collect(),
        params.q,
    );
    let error = Poly::zero(params.ring_dim, params.q);
    let ciphertext = RlweCiphertext::encrypt(
        rlwe_sk,
        &message_poly,
        params.delta(),
        a,
        &error,
        &params.ntt_context(),
    );
    let extracted = ciphertext.sample_extract_coeff0();

    assert!(
        extracted.decrypt(&state.secret_key, params.delta(), params.p) == message,
        "ClientState LWE key must decrypt coefficient-0 sample extraction"
    );
}

#[test]
fn direct_query_client_state_decrypts_sample_extracted_coeff0(
) -> Result<(), Box<dyn std::error::Error>> {
    let (params, crs, encoded_db, rlwe_sk, mut sampler) = fixture(0)?;
    let (state, _query) = query(&crs, 0, &encoded_db.config, &rlwe_sk, &mut sampler)?;

    assert_decrypts_coeff0(&state, &rlwe_sk, &params);
    Ok(())
}

#[test]
fn session_query_client_state_decrypts_sample_extracted_coeff0(
) -> Result<(), Box<dyn std::error::Error>> {
    let (params, crs, encoded_db, rlwe_sk, mut sampler) = fixture(17)?;
    let session = ClientSession::new(crs, rlwe_sk.clone(), &mut sampler)?;
    let (state, _query) = session.query(0, &encoded_db.config, &mut sampler)?;

    assert_decrypts_coeff0(&state, &rlwe_sk, &params);
    Ok(())
}
