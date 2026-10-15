//! The store against the per-query formula it replaces, on parts no responder
//! exposes: the whole offline output, the folded b, and the tags.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use super::*;
use crate::inspiring::inspiring2::{packing_offline, OfflinePackingKeys};
use crate::math::primality::is_prime;
use crate::math::DEFAULT_Q;
use crate::params::{InspireParams, SecurityLevel, DEFAULT_CRT_MODULI, DEFAULT_Q_2CRT_30BIT};
use crate::rlwe::RlweCiphertext;

const ADAPTIVE_CRT_MODULI: [u64; 2] = [67_043_329, 132_120_577];

fn modulus_sets() -> Vec<Vec<u64>> {
    vec![
        vec![DEFAULT_Q],
        DEFAULT_CRT_MODULI.to_vec(),
        DEFAULT_Q_2CRT_30BIT.to_vec(),
        ADAPTIVE_CRT_MODULI.to_vec(),
    ]
}

struct Case {
    pack_params: PackParams,
    keys: OfflinePackingKeys,
    ctx: NttContext,
    moduli: Vec<u64>,
    n: usize,
}

impl Case {
    fn new(ring_dim: usize, moduli: &[u64], gamma: usize) -> Self {
        let params = InspireParams {
            ring_dim,
            q: moduli.iter().product(),
            crt_moduli: moduli.to_vec(),
            p: 65537,
            sigma: 6.4,
            gadget_base: 1 << 20,
            query_gadget_len: 3,
            packing_gadget_len: 3,
            security_level: SecurityLevel::Bits128,
        };
        let pack_params = PackParams::try_new(&params, gamma).unwrap();
        let keys = OfflinePackingKeys::generate(&pack_params, [0x5a; 32]);
        Self {
            pack_params,
            keys,
            ctx: params.ntt_context(),
            moduli: moduli.to_vec(),
            n: ring_dim,
        }
    }

    fn random_poly(&self, rng: &mut ChaCha20Rng) -> Poly {
        Poly::random_with_rng_moduli(self.n, &self.moduli, rng)
    }

    fn random_columns(&self, rng: &mut ChaCha20Rng) -> Vec<Poly> {
        (0..self.pack_params.num_to_pack)
            .map(|_| self.random_poly(rng))
            .collect()
    }

    fn store(&self, columns: &[Poly]) -> ShardPackingStore {
        ShardPackingStore::build(&self.pack_params, columns, &self.ctx).unwrap()
    }

    /// What a responder computes with no store: one ciphertext-by-column product
    /// per column, coefficient 0 of each b, then the whole offline phase.
    fn per_query(&self, columns: &[Poly], a: &Poly, b: &Poly) -> (PrecompInsPIR, Poly) {
        let row = RlweCiphertext::from_parts(a.clone(), b.clone());
        let products: Vec<RlweCiphertext> = columns
            .iter()
            .map(|column| row.poly_mul(column, &self.ctx))
            .collect();
        let a_ct_tilde: Vec<Poly> = products.iter().map(|ct| ct.a.clone()).collect();
        let mut b_coeffs = vec![0u64; self.n];
        for (slot, product) in products.iter().enumerate() {
            b_coeffs[slot] = product.sample_extract_coeff0().b;
        }
        (
            packing_offline(&self.pack_params, &self.keys, &a_ct_tilde, &self.ctx),
            Poly::from_coeffs_moduli(b_coeffs, &self.moduli),
        )
    }

    fn through_store(
        &self,
        store: &ShardPackingStore,
        columns: &[Poly],
        a: &Poly,
        b: &Poly,
    ) -> Result<(PrecompInsPIR, Poly), String> {
        let checked = store.check_columns_and_fold_b(&self.pack_params, columns, b)?;
        let precomp = checked.packing_offline(&self.keys, a, &self.ctx)?;
        Ok((precomp, checked.into_b_poly()))
    }
}

fn same_offline_output(left: &(PrecompInsPIR, Poly), right: &(PrecompInsPIR, Poly)) -> bool {
    let (l, r) = (&left.0, &right.0);
    l.a_hat == r.a_hat
        && l.bold_t == r.bold_t
        && l.bold_t_ntt == r.bold_t_ntt
        && l.bold_t_bar == r.bold_t_bar
        && l.bold_t_hat == r.bold_t_hat
        && (l.num_to_pack, l.ring_dim, l.q) == (r.num_to_pack, r.ring_dim, r.q)
        && left.1 == right.1
}

#[test]
fn the_tag_field_is_a_prime_field() {
    assert_eq!(TAG_PRIME, 2_305_843_009_213_693_951);
    assert!(is_prime(TAG_PRIME));
}

#[test]
fn the_store_gives_the_per_query_offline_output_on_every_modulus_set() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5348_4100);
    let mut compared = 0;
    for moduli in modulus_sets() {
        for gamma in [1usize, 2, 16, 128] {
            let case = Case::new(256, &moduli, gamma);
            let columns = case.random_columns(&mut rng);
            let store = case.store(&columns);
            for _ in 0..3 {
                let (a, b) = (case.random_poly(&mut rng), case.random_poly(&mut rng));
                let expected = case.per_query(&columns, &a, &b);
                let served = case.through_store(&store, &columns, &a, &b).unwrap();
                assert!(
                    same_offline_output(&served, &expected),
                    "moduli {moduli:?} gamma {gamma}"
                );
                assert!(
                    !served.1.is_ntt() && served.0.bold_t_ntt.len() == gamma - 1,
                    "the output is the shape the online phase checks"
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 48);
}

#[derive(Clone, Copy, Debug)]
enum Words {
    Zero,
    Max,
    OneHot,
    Uniform,
}

fn patterned(pattern: Words, n: usize, moduli: &[u64], rng: &mut ChaCha20Rng) -> Poly {
    let mut words = Vec::with_capacity(n * moduli.len());
    for &modulus in moduli {
        let hot = rng.gen_range(0..n);
        for slot in 0..n {
            words.push(match pattern {
                Words::Zero => 0,
                Words::Max => modulus - 1,
                Words::OneHot => u64::from(slot == hot) * (modulus - 1),
                Words::Uniform => rng.gen_range(0..modulus),
            });
        }
    }
    Poly::from_crt_coeffs(words, moduli)
}

#[test]
fn the_store_gives_the_per_query_offline_output_on_extreme_words() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5348_4101);
    let patterns = [Words::Zero, Words::Max, Words::OneHot, Words::Uniform];
    let mut compared = 0;
    for moduli in modulus_sets() {
        let case = Case::new(256, &moduli, 8);
        for column_words in patterns {
            let columns: Vec<Poly> = (0..8)
                .map(|_| patterned(column_words, 256, &moduli, &mut rng))
                .collect();
            let store = case.store(&columns);
            for query_words in patterns {
                let a = patterned(query_words, 256, &moduli, &mut rng);
                let b = patterned(query_words, 256, &moduli, &mut rng);
                assert!(
                    same_offline_output(
                        &case.through_store(&store, &columns, &a, &b).unwrap(),
                        &case.per_query(&columns, &a, &b)
                    ),
                    "moduli {moduli:?} columns {column_words:?} query {query_words:?}"
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 64);
}

/// Words at or above the modulus never arrive over the wire, which refuses them, but
/// a caller can write them in memory. Both paths read them modulo the limb.
#[test]
fn words_above_the_modulus_are_read_the_same_way_as_the_per_query_path() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5348_4102);
    let mut compared = 0;
    for moduli in modulus_sets() {
        let case = Case::new(256, &moduli, 16);
        let lift = |poly: &Poly, rng: &mut ChaCha20Rng| {
            let mut lifted = poly.clone();
            let n = poly.dimension();
            for (position, word) in lifted.coeffs_mut().iter_mut().enumerate() {
                let modulus = moduli[position / n];
                match rng.gen_range(0..7) {
                    0 => *word += modulus,
                    1 => *word += modulus * ((u64::MAX - *word) / modulus),
                    2 => *word = u64::MAX,
                    _ => {}
                }
            }
            lifted
        };
        for _ in 0..4 {
            let columns = case.random_columns(&mut rng);
            let (a, b) = (case.random_poly(&mut rng), case.random_poly(&mut rng));

            let (lifted_a, lifted_b) = (lift(&a, &mut rng), lift(&b, &mut rng));
            assert_ne!((&lifted_a, &lifted_b), (&a, &b));
            let store = case.store(&columns);
            let served = case
                .through_store(&store, &columns, &lifted_a, &lifted_b)
                .unwrap();
            assert!(
                same_offline_output(&served, &case.per_query(&columns, &lifted_a, &lifted_b)),
                "moduli {moduli:?}: lifted query"
            );

            let lifted_columns: Vec<Poly> = columns.iter().map(|c| lift(c, &mut rng)).collect();
            let lifted_store = case.store(&lifted_columns);
            let served = case
                .through_store(&lifted_store, &lifted_columns, &lifted_a, &lifted_b)
                .unwrap();
            assert!(
                same_offline_output(
                    &served,
                    &case.per_query(&lifted_columns, &lifted_a, &lifted_b)
                ),
                "moduli {moduli:?}: lifted columns"
            );
            compared += 2;
        }
    }
    assert_eq!(compared, 32);
}

/// The comparisons above can fail: one word of one stored sum off by one moves the
/// offline output.
#[test]
fn one_wrong_word_in_a_stored_sum_changes_the_offline_output() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5348_4103);
    for moduli in modulus_sets() {
        let case = Case::new(256, &moduli, 16);
        let columns = case.random_columns(&mut rng);
        let (a, b) = (case.random_poly(&mut rng), case.random_poly(&mut rng));
        let expected = case.per_query(&columns, &a, &b);
        let store = case.store(&columns);
        assert!(same_offline_output(
            &case.through_store(&store, &columns, &a, &b).unwrap(),
            &expected
        ));
        let words = 256 * moduli.len();
        for (sum, word) in [(0usize, 0usize), (8, 5), (15, words - 1)] {
            let mut wrong = store.clone();
            let slot = &mut wrong.column_sums_ntt[sum].coeffs_mut()[word];
            *slot = (*slot + 1) % moduli[word / 256];
            assert!(
                !same_offline_output(
                    &case.through_store(&wrong, &columns, &a, &b).unwrap(),
                    &expected
                ),
                "moduli {moduli:?}: sum {sum} word {word} went unnoticed"
            );
        }
    }
}

/// What the tag check stands in front of: sums folded from other columns pack
/// another ciphertext, and nothing after the check would notice.
#[test]
fn sums_from_other_columns_pack_a_different_output_once_the_tags_agree() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5348_4104);
    let case = Case::new(256, &[DEFAULT_Q], 16);
    let built_from = case.random_columns(&mut rng);
    let mut served = built_from.clone();
    served[3].coeffs_mut()[200] ^= 1;
    let (a, b) = (case.random_poly(&mut rng), case.random_poly(&mut rng));

    let stale = case.store(&built_from);
    let refusal = case
        .through_store(&stale, &served, &a, &b)
        .expect_err("a stale store must be refused");
    assert!(
        refusal.contains("column 3 is not the column the store was built from"),
        "{refusal}"
    );

    let mut tags_forged = stale.clone();
    tags_forged.column_tags = served
        .iter()
        .map(|column| column_tag(column.coeffs(), &tags_forged.tag_key))
        .collect();
    let packed = case.through_store(&tags_forged, &served, &a, &b).unwrap();
    let expected = case.per_query(&served, &a, &b);
    assert_eq!(packed.1, expected.1, "b is folded from the served columns");
    assert_ne!(packed.0.a_hat, expected.0.a_hat);
}

#[test]
fn a_tag_moves_with_any_change_to_its_column() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5348_4105);
    for moduli in modulus_sets() {
        let case = Case::new(256, &moduli, 4);
        let columns = case.random_columns(&mut rng);
        let store = case.store(&columns);
        let tag = |words: &[u64]| column_tag(words, &store.tag_key);
        let words = columns[0].coeffs().to_vec();
        assert_eq!(tag(&words), store.column_tags[0]);
        assert_eq!(store.tag_key.len(), 2 * words.len());
        assert!(store.tag_key.iter().all(|&key| key < TAG_PRIME));

        let last = words.len() - 1;
        for position in [0, 1, last / 2, last] {
            let deltas = [
                1,
                1 << 32,
                moduli[position / 256],
                TAG_PRIME,
                TAG_PRIME << 1,
                u64::MAX,
            ];
            for delta in deltas {
                let mut changed = words.clone();
                changed[position] = changed[position].wrapping_add(delta);
                assert_ne!(
                    tag(&changed),
                    tag(&words),
                    "moduli {moduli:?}: word {position} + {delta} kept the tag"
                );
            }
        }
        let mut swapped = words.clone();
        swapped.swap(3, 200);
        assert_ne!(tag(&swapped), tag(&words));
        assert_ne!(
            store.column_tags[0], store.column_tags[1],
            "tags follow the column, not its position"
        );
    }
}

#[test]
fn a_stripe_folds_to_coefficient_zero_of_the_negacyclic_product() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5348_4106);
    let key = vec![1u64; 2 * 24];
    for modulus in [3u64, 67_043_329, DEFAULT_Q, (1 << 63) - 25, u64::MAX] {
        for round in 0..40 {
            let draw = |rng: &mut ChaCha20Rng| -> Vec<u64> {
                (0..24)
                    .map(|_| match round % 4 {
                        0 => u64::MAX,
                        1 => rng.gen_range(0..modulus),
                        _ => rng.gen(),
                    })
                    .collect()
            };
            let (b, h) = (draw(&mut rng), draw(&mut rng));

            // Schoolbook: X^d = -1, so the terms with i + j = d land on coefficient 0, negated.
            let q = u128::from(modulus);
            let mut coeff0 = (u128::from(b[0]) % q) * (u128::from(h[0]) % q) % q;
            for i in 1..24 {
                let term = (u128::from(b[i]) % q) * (u128::from(h[24 - i]) % q) % q;
                coeff0 = (coeff0 + q - term) % q;
            }

            let (_, folded) = fold_stripe(&h, &key, &reverse_for_coeff0(&b, modulus), modulus);
            assert_eq!(
                u128::from(folded),
                coeff0,
                "modulus {modulus} round {round}"
            );
        }
    }
}

#[test]
fn a_store_is_refused_for_anything_but_its_own_shape() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5348_4107);
    let case = Case::new(256, &[DEFAULT_Q], 16);
    let columns = case.random_columns(&mut rng);
    let store = case.store(&columns);
    let (a, b) = (case.random_poly(&mut rng), case.random_poly(&mut rng));
    let refused = |outcome: Result<(PrecompInsPIR, Poly), String>, needle: &str| {
        let refusal = outcome.expect_err("refused");
        assert!(refusal.contains(needle), "{refusal}");
    };

    refused(
        case.through_store(&store, &columns[..15], &a, &b),
        "the shard holds 15 columns, the store packs 16",
    );
    let mut ntt_column = columns.clone();
    ntt_column[9].to_ntt(&case.ctx);
    refused(
        case.through_store(&store, &ntt_column, &a, &b),
        "column 9 is not a coefficient-form polynomial",
    );
    refused(
        case.through_store(&store, &columns, &a, &b.to_ntt_new(&case.ctx)),
        "the query's b must be a coefficient-form polynomial",
    );
    refused(
        case.through_store(&store, &columns, &a.to_ntt_new(&case.ctx), &b),
        "the query's a must be a coefficient-form polynomial",
    );

    let narrower = Case::new(256, &[DEFAULT_Q], 8);
    refused(
        narrower.through_store(&store, &columns, &a, &b),
        "built for gamma 16",
    );
    let other_ring = Case::new(256, &DEFAULT_CRT_MODULI, 16);
    refused(
        other_ring.through_store(&store, &columns, &a, &b),
        "built for gamma 16",
    );

    let build = |pack_params: &PackParams, columns: &[Poly], ctx: &NttContext| {
        ShardPackingStore::build(pack_params, columns, ctx).expect_err("refused")
    };
    assert!(build(&case.pack_params, &columns[..15], &case.ctx).contains("holds 15 columns"));
    assert!(build(&case.pack_params, &ntt_column, &case.ctx).contains("column 9"));
    assert!(build(&case.pack_params, &columns, &other_ring.ctx).contains("NTT context"));
    let mut short_tables = case.pack_params.clone();
    short_tables.automorph_tables[0].pop();
    assert!(build(&short_tables, &columns, &case.ctx).contains("automorphism table"));
}

#[test]
fn a_store_holds_one_polynomial_per_column() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5348_4108);
    for (moduli, gamma) in [(vec![DEFAULT_Q], 32usize), (DEFAULT_CRT_MODULI.to_vec(), 8)] {
        let case = Case::new(256, &moduli, gamma);
        let store = case.store(&case.random_columns(&mut rng));
        let words = 256 * moduli.len();
        assert_eq!(store.num_to_pack(), gamma);
        assert_eq!(store.payload_bytes(), gamma * words * 8);
        let tags_and_key = (2 * words + gamma) * 8;
        let containers = std::mem::size_of::<ShardPackingStore>()
            + gamma * std::mem::size_of::<Poly>()
            + (gamma + 1) * moduli.len() * 8;
        assert_eq!(
            store.resident_bytes(),
            store.payload_bytes() + tags_and_key + containers
        );
        assert!(!format!("{store:?}").contains("tag_key"));
    }
}

/// The store produces `R` and hands it on: a second copy of the recursion could
/// drift from the one the per-query path runs.
#[test]
fn the_backward_recursion_has_one_implementation() {
    let (store_source, _tests) = include_str!("../shard_store.rs")
        .split_once("#[cfg(test)]")
        .expect("test module");
    assert!(!store_source.contains("gadget_decompose"));
    assert_eq!(store_source.matches("fold_rotations_backward(").count(), 1);
    assert_eq!(store_source.matches("rotation_sums(").count(), 1);

    let (offline_source, _tests) = include_str!("../inspiring2.rs")
        .split_once("#[cfg(test)]")
        .expect("test module");
    assert_eq!(
        offline_source
            .matches("fn fold_rotations_backward(")
            .count(),
        1
    );
    assert_eq!(
        offline_source.matches("fold_rotations_backward(").count(),
        2
    );
    assert_eq!(offline_source.matches("fn rotation_sums(").count(), 1);
    assert_eq!(offline_source.matches("rotation_sums(").count(), 2);
}
