//! Wall time of the packing-key path at the production d=2048 cell: the client's key
//! build, seat registration, a respond under a registered and an inline key, and the
//! online packing kernel alone. Prints every sample so runs can be paired.
//!
//! `RAVEN_BENCH_ROUNDS` sets the sample count per cell (default 30).

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stderr,
    reason = "bench harness; an abort here is the failure report"
)]

#[path = "../tests/packing_key_fixture/mod.rs"]
mod packing_key_fixture;

use std::time::Instant;

use packing_key_fixture::{inline_query, Cell};
use raven_inspire::inspiring::{
    generate_rotations, packing_offline, packing_online_gather, ClientPackingKeys, PackParams,
    PrecompInsPIR,
};
use raven_inspire::math::{GaussianSampler, NttContext, Poly};
use raven_inspire::params::InspireParams;
use raven_inspire::pir::{
    respond_seeded_inspiring_cached_with_session, ClientSession, PackingKeySeat, ServerCrs,
    ServerSessionStore,
};
use raven_inspire::rlwe::{RlweCiphertext, RlweSecretKey};

fn rounds() -> usize {
    std::env::var("RAVEN_BENCH_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(30)
}

fn load() -> String {
    std::fs::read_to_string("/proc/loadavg").map_or_else(
        |_| "unavailable".to_owned(),
        |text| text.split(' ').take(3).collect::<Vec<_>>().join(" "),
    )
}

fn report(metric: &str, entry_size: usize, mut samples_ms: Vec<f64>) {
    let listed: Vec<String> = samples_ms.iter().map(|ms| format!("{ms:.4}")).collect();
    eprintln!("SAMPLES {metric} entry={entry_size} {}", listed.join(" "));
    samples_ms.sort_by(f64::total_cmp);
    let n = samples_ms.len();
    eprintln!(
        "SUMMARY {metric} entry={entry_size} n={n} min={:.4} median={:.4} p90={:.4} max={:.4} ms load=[{}]",
        samples_ms[0],
        samples_ms[n / 2],
        samples_ms[(n * 9) / 10],
        samples_ms[n - 1],
        load()
    );
}

fn time_ms<T>(work: impl FnOnce() -> T) -> f64 {
    let start = Instant::now();
    let out = work();
    let elapsed = start.elapsed().as_secs_f64() * 1e3;
    std::hint::black_box(out);
    elapsed
}

#[test]
#[ignore = "cost: about 10 s of key builds at d=2048; trigger: a change to \
            ClientPackingKeys::generate, ClientSession::new or PackingKeySeat::from_wire"]
fn client_key_build_and_seat_registration() {
    let params = InspireParams::secure_128_d2048();
    let ctx = params.ntt_context();
    for entry_size in [256usize, 512, 1024] {
        let gamma = raven_inspire::num_columns(entry_size);
        let cell = Cell::build(&params, entry_size, 0x4245_4e00);
        let mut sampler = GaussianSampler::with_seed(params.sigma, 0x4245_4e01);
        let secret_key = RlweSecretKey::generate(&params, &mut sampler);

        // What a client downloads: the wire form drops the server-only CRS parts.
        let client_crs: ServerCrs =
            bincode::deserialize(&bincode::serialize(&cell.crs).unwrap()).unwrap();

        let mut key_build = Vec::new();
        let mut session_build = Vec::new();
        let mut seat_build = Vec::new();
        for _ in 0..rounds() {
            key_build.push(time_ms(|| {
                ClientPackingKeys::generate(&secret_key, &params, gamma, [1; 32], &mut sampler)
                    .unwrap()
            }));
            let (crs, key) = (client_crs.clone(), secret_key.clone());
            session_build.push(time_ms(|| {
                ClientSession::new(crs, key, &mut sampler).unwrap()
            }));
            let keys = cell.wire_keys();
            seat_build.push(time_ms(|| {
                PackingKeySeat::from_wire(keys, cell.cache.pack_params(), &ctx).unwrap()
            }));
        }
        report("client_key_build", entry_size, key_build);
        report("client_session_build", entry_size, session_build);
        report("seat_registration", entry_size, seat_build);

        let tables: Vec<f64> = (0..5)
            .map(|_| time_ms(|| PackParams::try_new(&params, gamma).unwrap()))
            .collect();
        report("pack_params_build", entry_size, tables);
    }
}

#[test]
#[ignore = "cost: about a minute of responds at d=2048; trigger: a change to the respond \
            path, the seat or the online packing kernel"]
fn respond_under_a_seat_and_the_online_kernel() {
    let params = InspireParams::secure_128_d2048();
    let ctx = params.ntt_context();
    let index = params.ring_dim as u64 + 17;
    for entry_size in [256usize, 512] {
        let mut cell = Cell::build(&params, entry_size, 0x4245_4e00);
        let keys = cell.wire_keys();
        let store = ServerSessionStore::new();
        let handle = store
            .register_server_side(keys.clone(), cell.cache.pack_params(), &ctx)
            .unwrap();
        let query = cell.next_query(index);
        let inline = inline_query(&query, &keys);
        let mut by_handle = inline.clone();
        by_handle.inspiring_packing_keys = None;
        by_handle.session_handle = Some(handle);

        let respond = |query, store| {
            respond_seeded_inspiring_cached_with_session(
                &cell.crs,
                &cell.encoded,
                query,
                &cell.cache,
                store,
            )
            .unwrap()
        };
        let expected = bincode::serialize(&respond(&by_handle, Some(&store))).unwrap();

        let mut seat = Vec::new();
        let mut inline_key = Vec::new();
        for round in 0..rounds() + 3 {
            let (a, b) = if round % 2 == 0 {
                let a = time_ms(|| respond(&by_handle, Some(&store)));
                (a, time_ms(|| respond(&inline, None)))
            } else {
                let b = time_ms(|| respond(&inline, None));
                (time_ms(|| respond(&by_handle, Some(&store))), b)
            };
            if round >= 3 {
                seat.push(a);
                inline_key.push(b);
            }
        }
        assert_eq!(
            bincode::serialize(&respond(&inline, None)).unwrap(),
            expected,
            "both key sources serve the same bytes"
        );
        report("respond_registered_seat", entry_size, seat);
        report("respond_inline_key", entry_size, inline_key);

        // The kernel alone, on the offline output of the same query.
        let rgsw = query.rgsw_ciphertext.expand();
        let shard = &cell.encoded.shards[query.shard_id as usize];
        let columns: Vec<RlweCiphertext> = shard
            .polynomials
            .iter()
            .map(|column| rgsw.rows[0].poly_mul(column, &ctx))
            .collect();
        let a_ct_tilde: Vec<Poly> = columns.iter().map(|ct| ct.a.clone()).collect();
        let b_poly = Poly::zero_moduli(params.ring_dim, params.moduli());
        let pack_params = cell.cache.pack_params();
        let precomp = packing_offline(pack_params, cell.cache.offline_keys(), &a_ct_tilde, &ctx);
        let y_body_ntt: Vec<Poly> = keys.y_body.iter().map(|y| y.to_ntt_new(&ctx)).collect();
        let y_all_ntt: Vec<Vec<Poly>> = generate_rotations(pack_params, &keys.y_body)
            .iter()
            .map(|rotation| rotation.iter().map(|y| y.to_ntt_new(&ctx)).collect())
            .collect();
        let gathered =
            packing_online_gather(&precomp, pack_params, &y_body_ntt, &b_poly, &ctx).unwrap();
        let materialized = pack_over_materialized_rotations(&precomp, &y_all_ntt, &b_poly, &ctx);
        assert_eq!(gathered.b.coeffs(), materialized.b.coeffs());

        let mut kernel = Vec::new();
        let mut reference = Vec::new();
        for round in 0..rounds() + 3 {
            let gather = || {
                time_ms(|| {
                    packing_online_gather(&precomp, pack_params, &y_body_ntt, &b_poly, &ctx)
                        .unwrap()
                })
            };
            let cached = || {
                time_ms(|| pack_over_materialized_rotations(&precomp, &y_all_ntt, &b_poly, &ctx))
            };
            let (a, b) = if round % 2 == 0 {
                let a = gather();
                (a, cached())
            } else {
                let b = cached();
                (gather(), b)
            };
            if round >= 3 {
                kernel.push(a);
                reference.push(b);
            }
        }
        report("online_kernel", entry_size, kernel);
        report(
            "online_kernel_materialized_rotations",
            entry_size,
            reference,
        );
    }
}

/// Every rotation of the key materialized in NTT form, one pointwise product and sum
/// per (rotation, digit): the timing reference for the gather.
fn pack_over_materialized_rotations(
    precomp: &PrecompInsPIR,
    y_all_ntt: &[Vec<Poly>],
    b_poly: &Poly,
    ctx: &NttContext,
) -> RlweCiphertext {
    let mut sum_b_ntt = Poly::zero_moduli(precomp.ring_dim, b_poly.moduli());
    sum_b_ntt.to_ntt(ctx);
    for (y_i, t_i) in y_all_ntt.iter().zip(&precomp.bold_t_ntt) {
        for (y, t) in y_i.iter().zip(t_i) {
            sum_b_ntt = sum_b_ntt.add_ntt_domain(&y.mul_ntt_domain(t, ctx));
        }
    }
    sum_b_ntt.from_ntt(ctx);
    RlweCiphertext::from_parts(precomp.a_hat.clone(), b_poly + &sum_b_ntt)
}
