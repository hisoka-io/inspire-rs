//! Seeded-query round trips over (ring_dim, record_bytes) cells under both the
//! single-prime and the 2-CRT modulus. Formerly ~30 hand-enumerated cells
//! whose per-cell PASS/FAIL report was unreachable (the assert inside
//! `run_cell` fired on the first failure) and whose d=2048 legs duplicated
//! commit_e_two_crt_regression_grid.rs cell-for-cell (same fixture, indices
//! and assertion). Converted 2026-09-06: the d<=1024 cells - including the
//! single-prime control the grid lacks below d=2048 - are drawn by the
//! property below, both modulus shapes every case; the d=2048 legs live in the
//! commit-E grid, which stays.

use proptest::prelude::*;
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, SecurityLevel, DEFAULT_Q_2CRT_30BIT};
use raven_inspire::{
    extract_inspiring, query_seeded, respond_seeded_inspiring, setup, PackingMode,
};

fn params_for(ring_dim: usize, crt_moduli: Vec<u64>) -> InspireParams {
    let q: u64 = crt_moduli.iter().product();
    InspireParams {
        ring_dim,
        q,
        crt_moduli,
        p: 65537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

fn round_trip_cell(
    params: &InspireParams,
    entries: u64,
    record_bytes: usize,
) -> Result<(), String> {
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let total = (entries as usize)
        .checked_mul(record_bytes)
        .ok_or_else(|| "entries * record_bytes overflow".to_string())?;
    let mut db = vec![0u8; total];
    for i in 0..entries as usize {
        for j in 0..record_bytes {
            db[i * record_bytes + j] = ((i + j) % 251) as u8;
        }
    }
    let (crs, encoded_db, sk) = setup(params, &db, record_bytes, &mut sampler)
        .map_err(|e| format!("setup failed: {e:?}"))?;

    let n = entries.saturating_sub(1).max(1);
    let indices: [u64; 3] = [n / 4, n / 2, (3 * n) / 4];
    for &idx in &indices {
        let (state, mut seeded_query) =
            query_seeded(&crs, idx, &encoded_db.config, &sk, &mut sampler)
                .map_err(|e| format!("query_seeded @ idx={idx}: {e:?}"))?;
        seeded_query.packing_mode = PackingMode::Inspiring;
        let response = respond_seeded_inspiring(&crs, &encoded_db, &seeded_query)
            .map_err(|e| format!("respond_seeded_inspiring @ idx={idx}: {e:?}"))?;
        let recovered = extract_inspiring(&crs, &state, &response, record_bytes)
            .map_err(|e| format!("extract_inspiring @ idx={idx}: {e:?}"))?;
        let expected: Vec<u8> = (0..record_bytes)
            .map(|j| (((idx as usize) + j) % 251) as u8)
            .collect();
        if recovered != expected {
            return Err(format!(
                "byte-mismatch @ idx={idx}: recovered[..4] = {:?}, expected[..4] = {:?}",
                &recovered[..4.min(recovered.len())],
                &expected[..4.min(expected.len())]
            ));
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 6,
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    /// Every drawn (ring_dim, record_bytes) cell round-trips under BOTH the
    /// single-prime DEFAULT_Q and the 2-CRT 30-bit modulus (both shapes per
    /// case, so a shape-specific defect cannot escape a run). record_bytes
    /// includes 128, carried from the deleted d=2048 large-ring test.
    #[test]
    fn seeded_round_trip_over_ring_and_record_cells(
        ring_dim in prop::sample::select(&[256usize, 512, 1024]),
        record_bytes in prop::sample::select(&[8usize, 32, 128, 256]),
    ) {
        let entries = ring_dim as u64;
        for crt in [
            vec![1_152_921_504_606_830_593u64],
            DEFAULT_Q_2CRT_30BIT.to_vec(),
        ] {
            let label = if crt.len() == 1 { "1-CRT" } else { "2-CRT" };
            let params = params_for(ring_dim, crt);
            if let Err(e) = round_trip_cell(&params, entries, record_bytes) {
                return Err(TestCaseError::fail(format!(
                    "{label} d={ring_dim} entries={entries} record={record_bytes}B: {e}"
                )));
            }
        }
    }
}
