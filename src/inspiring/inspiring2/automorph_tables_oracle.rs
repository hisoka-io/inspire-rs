//! The automorphism tables `PackParams` derives, checked against the probe search they
//! replaced. The search assumes nothing about NTT ordering or scaling: it reads the
//! permutation off a random polynomial and its automorphed image, so it is the
//! independent specification here.

use super::PackParams;
use crate::math::{NttContext, Poly};
use crate::params::{InspireParams, DEFAULT_CRT_MODULI, DEFAULT_Q_2CRT_30BIT};
use crate::rlwe::apply_automorphism;

/// The O(n^3) search `PackParams` shipped before the derivation, byte for byte.
fn search_automorph_tables(n: usize, moduli: &[u64], ctx: &NttContext) -> Vec<Vec<usize>> {
    let two_n = 2 * n;
    let mut tables = Vec::with_capacity(n);

    for t in (1..two_n).step_by(2) {
        let mut table = vec![0usize; n];

        loop {
            let poly = Poly::random_moduli(n, moduli);
            let mut poly_ntt = poly.clone();
            poly_ntt.to_ntt(ctx);

            let poly_auto = apply_automorphism(&poly, t);
            let mut poly_auto_ntt = poly_auto.clone();
            poly_auto_ntt.to_ntt(ctx);

            let mut must_redo = false;

            for i in 0..n {
                let orig_val = poly_ntt.coeffs()[i];
                let mut found = 0usize;
                let mut count = 0usize;

                for j in 0..n {
                    if poly_auto_ntt.coeffs()[j] == orig_val {
                        count += 1;
                        found = j;
                    }
                }

                if count != 1 {
                    must_redo = true;
                    break;
                }

                table[found] = i;
            }

            if !must_redo {
                break;
            }
        }

        tables.push(table);
    }

    tables
}

/// Every moduli shape `NttContext` takes: the Solinas single prime, a Montgomery
/// single prime, and both shipped 2-CRT pairs.
fn moduli_sets() -> Vec<(&'static str, Vec<u64>)> {
    vec![
        ("DEFAULT_Q", vec![crate::math::mod_q::DEFAULT_Q]),
        ("DEFAULT_CRT_MODULI[0]", vec![DEFAULT_CRT_MODULI[0]]),
        ("DEFAULT_CRT_MODULI", DEFAULT_CRT_MODULI.to_vec()),
        ("DEFAULT_Q_2CRT_30BIT", DEFAULT_Q_2CRT_30BIT.to_vec()),
    ]
}

fn assert_derivation_matches_search(ring_dim: usize, label: &str, moduli: &[u64]) {
    let params = InspireParams {
        ring_dim,
        q: moduli.iter().product(),
        crt_moduli: moduli.to_vec(),
        ..InspireParams::secure_128_d2048()
    };
    let derived = PackParams::try_new_full(&params)
        .expect("a valid ring must build")
        .automorph_tables;
    let searched = search_automorph_tables(ring_dim, moduli, &params.ntt_context());

    assert_eq!(derived.len(), ring_dim, "{label} d={ring_dim}: table count");
    for (row, (got, want)) in derived.iter().zip(&searched).enumerate() {
        let t = 2 * row + 1;
        if let Some(slot) = (0..ring_dim).find(|&slot| got.get(slot) != want.get(slot)) {
            panic!(
                "{label} d={ring_dim}: tau_{t} slot {slot} derived {:?}, search found {:?}",
                got.get(slot),
                want.get(slot)
            );
        }
        assert_eq!(
            got.len(),
            want.len(),
            "{label} d={ring_dim}: tau_{t} length"
        );
    }
}

#[test]
fn derived_tables_equal_the_search_for_every_odd_t_up_to_d256() {
    for (label, moduli) in moduli_sets() {
        for log_dim in 1..=8 {
            assert_derivation_matches_search(1 << log_dim, label, &moduli);
        }
    }
}

#[test]
#[ignore = "cost: about 15-25 s under --release, the O(n^3) search at d=2048 \
            over four moduli shapes; trigger: a change to NTT ordering or to the table \
            derivation"]
fn derived_tables_equal_the_search_for_every_odd_t_at_d2048() {
    for (label, moduli) in moduli_sets() {
        assert_derivation_matches_search(2048, label, &moduli);
    }
}
