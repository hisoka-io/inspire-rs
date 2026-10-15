//! Full-size noise bench of the special-modulus packing path.
//!
//! Every response is decoded and its error is split exactly into the folded query error, the
//! key error before the division, and the division remainders. Worst-value shards give the
//! noise rows; shards of 256-byte record cells give the byte-exact decode rows.
//!
//! The first response of every secret, and every response of the first secret, is also
//! answered by the serving function and must equal the instrumented response.
//!
//! Environment: KEYBENCH_OUT (directory, required), KEYBENCH_ROWS (comma list of substrings),
//! KEYBENCH_SECRETS, KEYBENCH_PER_SECRET, KEYBENCH_REC_SECRETS, KEYBENCH_REC_PER,
//! KEYBENCH_THREADS, KEYBENCH_SEED, KEYBENCH_VECTORS (json of real record cells).

// A bench stops at the first step that fails.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use raven_inspire::math::GaussianSampler;
use raven_inspire::special_key::{
    decode_nibbles, expand_key_mask, keygen, nibbles_to_bytes, phase, query, respond,
    sampler_variance, NoiseProbe, PackCtx, Query, Secret, Segment, Shard, Step1, Store, SET_Q28,
    SET_Q29,
};

const D: usize = 2048;
const T: u64 = 16;
const RECORD_BYTES: usize = 246;
const CELL_BYTES: usize = 256;
/// Decoder tail of the policy "today's failure level", in standard deviations (log2).
const LOG2_R_SAME: f64 = 10.762;
const SIGMA_S: f64 = 6.4;
const SIGMA_K: f64 = 6.4;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Shape {
    Single,
    Pair,
    Dial,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Worst,
    Records,
}

#[derive(Clone, Copy)]
struct Row {
    name: &'static str,
    set: (u64, u64),
    sigma_q: f64,
    shape: Shape,
}

const ROWS: &[Row] = &[
    Row {
        name: "q29-single",
        set: SET_Q29,
        sigma_q: 6.4,
        shape: Shape::Single,
    },
    Row {
        name: "q29-pair",
        set: SET_Q29,
        sigma_q: 6.4,
        shape: Shape::Pair,
    },
    Row {
        name: "q29-dial",
        set: SET_Q29,
        sigma_q: 6.4,
        shape: Shape::Dial,
    },
    Row {
        name: "q28-single",
        set: SET_Q28,
        sigma_q: 3.2,
        shape: Shape::Single,
    },
    Row {
        name: "q28-pair",
        set: SET_Q28,
        sigma_q: 3.2,
        shape: Shape::Pair,
    },
    Row {
        name: "q28-dial",
        set: SET_Q28,
        sigma_q: 3.2,
        shape: Shape::Dial,
    },
    Row {
        name: "q28e64-single",
        set: SET_Q28,
        sigma_q: 6.4,
        shape: Shape::Single,
    },
    Row {
        name: "q28e64-pair",
        set: SET_Q28,
        sigma_q: 6.4,
        shape: Shape::Pair,
    },
    Row {
        name: "q28e64-dial",
        set: SET_Q28,
        sigma_q: 6.4,
        shape: Shape::Dial,
    },
];

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .unwrap_or_default()
        .split_whitespace()
        .take(3)
        .collect::<Vec<_>>()
        .join(" ")
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// The committed record vectors, then cells in the same layout with random field bytes.
fn record_cells(rng: &mut ChaCha20Rng, n: usize, real: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut cells: Vec<Vec<u8>> = real.to_vec();
    while cells.len() < n {
        let leaf = cells.len() as u32;
        let mut cell = vec![0u8; CELL_BYTES];
        cell[0] = 1;
        cell[1] = (leaf & 1) as u8;
        cell[2..6].copy_from_slice(&leaf.to_be_bytes());
        rng.fill_bytes(&mut cell[6..22]);
        // Seven 32-byte field words; the two after the prefix are zero on a self row.
        for w in 0..7 {
            let at = 22 + 32 * w;
            if w < 2 && cell[1] == 0 {
                continue;
            }
            rng.fill_bytes(&mut cell[at..at + 32]);
            cell[at] &= 0x1f;
        }
        cells.push(cell);
    }
    cells.truncate(n);
    cells
}

struct Data {
    shards: Vec<Shard>,
    /// Record cells behind each shard, `per_row` consecutive cells per selectable row.
    cells: Vec<Vec<Vec<u8>>>,
    per_row: usize,
}

fn build_data(row: &Row, mode: Mode, rng: &mut ChaCha20Rng, real: &[Vec<u8>]) -> Data {
    let nibbles = 2 * CELL_BYTES;
    match (mode, row.shape) {
        (Mode::Worst, Shape::Single) => Data {
            shards: vec![Shard::constant(D, nibbles, -8)],
            cells: vec![],
            per_row: 1,
        },
        (Mode::Worst, Shape::Pair) => Data {
            shards: vec![
                Shard::constant(D, nibbles, -8),
                Shard::constant(D, nibbles, -8),
            ],
            cells: vec![],
            per_row: 1,
        },
        (Mode::Worst, Shape::Dial) => Data {
            shards: vec![Shard::constant(D / 2, 2 * nibbles, -8)],
            cells: vec![],
            per_row: 2,
        },
        (Mode::Records, Shape::Single) => {
            let cells = record_cells(rng, D, real);
            Data {
                shards: vec![Shard::from_cells(&cells)],
                cells: vec![cells],
                per_row: 1,
            }
        }
        (Mode::Records, Shape::Pair) => {
            let x = record_cells(rng, D, real);
            let y = record_cells(rng, D, &[]);
            Data {
                shards: vec![Shard::from_cells(&x), Shard::from_cells(&y)],
                cells: vec![x, y],
                per_row: 1,
            }
        }
        (Mode::Records, Shape::Dial) => {
            let cells = record_cells(rng, D, real);
            let joined: Vec<Vec<u8>> = cells.chunks(2).map(<[Vec<u8>]>::concat).collect();
            Data {
                shards: vec![Shard::from_cells(&joined)],
                cells: vec![cells],
                per_row: 2,
            }
        }
    }
}

#[derive(Default, Clone)]
struct Stat {
    secret: usize,
    resp: usize,
    n: usize,
    sum_err2: f64,
    sum_err4: f64,
    max_err: i64,
    sum_fold2: f64,
    sum_pack2: f64,
    max_pack: i64,
    sum_key2: f64,
    sum_round2: f64,
    sum_fold_pack: f64,
    /// One fold value per segment (the first column): independent across responses.
    fold_first: Vec<i64>,
    identity_misses: usize,
    decode_misses: usize,
    records_checked: usize,
    records_wrong: usize,
    /// Responses also answered by the serving function, and how many of those differed.
    served_checked: usize,
    served_wrong: usize,
    /// Coefficients with |err| above 3, 4, 5 expected standard deviations (worst values only).
    err_tail: [usize; 3],
    /// The same for the packing part alone.
    pack_tail: [usize; 3],
    ms: f64,
}

struct Job<'a> {
    row: &'a Row,
    mode: Mode,
    ctx: &'a PackCtx,
    probe: &'a NoiseProbe,
    data: &'a Data,
    store: &'a Store,
    seed: u64,
    per_secret: usize,
    sd_total: f64,
    sd_pack: f64,
}

fn run_secret(job: &Job<'_>, secret: usize) -> Vec<Stat> {
    let ctx = job.ctx;
    let tag = job
        .seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((secret as u64) << 20 | (job.mode as u64) << 16);
    let mut rng = ChaCha20Rng::seed_from_u64(tag);
    let mut s_sampler = GaussianSampler::with_seed(SIGMA_S, tag ^ 0xA1);
    let mut k_sampler = GaussianSampler::with_seed(SIGMA_K, tag ^ 0xA2);
    let mut q_sampler = GaussianSampler::with_seed(job.row.sigma_q, tag ^ 0xA3);

    let sk = Secret::sample(ctx, &mut s_sampler);
    let mut w_seed = [0u8; 32];
    rng.fill_bytes(&mut w_seed);
    let (key, key_err) = keygen(ctx, &sk, w_seed, &mut k_sampler);
    let w_ntt = expand_key_mask(ctx, w_seed).to_ntt(ctx);
    let key_ntt = key.to_ntt(ctx);
    let sp = job.probe.secret(ctx, &sk);
    let segs: Vec<Segment<'_>> = job
        .data
        .shards
        .iter()
        .enumerate()
        .map(|(k, shard)| Segment { shard, query: k })
        .collect();
    let p = ctx.lp.q as f64;

    (0..job.per_secret)
        .map(|resp| {
            let t0 = Instant::now();
            let made: Vec<(Query, Vec<i64>, usize)> = segs
                .iter()
                .map(|seg| {
                    let rows = seg.shard.rows;
                    // The first responses of the first secret walk the low rows (the real
                    // vectors in record mode) and both ring edges.
                    let index = match (secret, resp) {
                        (0, r) if r < 4 => r % rows,
                        (0, 4) => rows - 1,
                        _ => rng.gen_range(0..rows),
                    };
                    let mut a_seed = [0u8; 32];
                    rng.fill_bytes(&mut a_seed);
                    let (qr, e) = query(ctx, &sk, index, a_seed, &mut q_sampler);
                    (qr, e, index)
                })
                .collect();
            let refs: Vec<(&Query, &[i64], usize)> =
                made.iter().map(|(q, e, i)| (q, &e[..], *i)).collect();
            let m = job.probe.measure(
                ctx,
                &segs,
                &Step1::Stored(job.store),
                &sk,
                &sp,
                &w_ntt,
                &key_ntt,
                &key_err,
                &refs,
            );

            let mut st = Stat {
                secret,
                resp,
                n: m.err.len(),
                ..Stat::default()
            };
            for j in 0..m.err.len() {
                let (e, f) = (m.err[j], m.fold[j]);
                let pack = e - f;
                let key_part = m.key[j] as f64 / p;
                let round_part = pack as f64 - key_part;
                let ef = e as f64;
                st.sum_err2 += ef * ef;
                st.sum_err4 += ef * ef * ef * ef;
                st.max_err = st.max_err.max(e.abs());
                st.sum_fold2 += (f as f64) * (f as f64);
                st.sum_pack2 += (pack as f64) * (pack as f64);
                st.max_pack = st.max_pack.max(pack.abs());
                st.sum_key2 += key_part * key_part;
                st.sum_round2 += round_part * round_part;
                st.sum_fold_pack += f as f64 * pack as f64;
                for (k, mult) in [3.0, 4.0, 5.0].into_iter().enumerate() {
                    if ef.abs() > mult * job.sd_total {
                        st.err_tail[k] += 1;
                    }
                    if (pack as f64).abs() > mult * job.sd_pack {
                        st.pack_tail[k] += 1;
                    }
                }
            }
            let mut off = 0;
            for seg in &segs {
                st.fold_first.push(m.fold[off]);
                off += seg.shard.cols;
            }
            st.identity_misses = m.identity_misses;
            st.decode_misses = m.decode_misses;

            if secret == 0 || resp == 0 {
                let plain: Vec<&Query> = made.iter().map(|(q, _, _)| q).collect();
                let served = respond(
                    ctx,
                    &segs,
                    &Step1::Stored(job.store),
                    &w_ntt,
                    &key_ntt,
                    &plain,
                );
                st.served_checked = 1;
                st.served_wrong = usize::from(served != m.packed);
            }

            if job.mode == Mode::Records {
                let nibbles = decode_nibbles(ctx, &phase(ctx, &sk, &m.packed), ctx.gamma);
                let mut off = 0;
                for (k, seg) in segs.iter().enumerate() {
                    let index = made[k].2;
                    for r in 0..job.data.per_row {
                        let cell = &job.data.cells[k][index * job.data.per_row + r];
                        let at = off + r * 2 * CELL_BYTES;
                        // What the wire carries: the first 492 coefficients of each record.
                        let got = nibbles_to_bytes(&nibbles[at..at + 2 * RECORD_BYTES]);
                        st.records_checked += 1;
                        if got != cell[..RECORD_BYTES] {
                            st.records_wrong += 1;
                        }
                    }
                    off += seg.shard.cols;
                }
            }
            st.ms = t0.elapsed().as_secs_f64() * 1e3;
            st
        })
        .collect()
}

fn log2(x: f64) -> f64 {
    x.log2()
}

fn main() {
    let out_dir =
        std::env::var("KEYBENCH_OUT").expect("KEYBENCH_OUT must name the output directory");
    std::fs::create_dir_all(&out_dir).unwrap();
    let filter: Vec<String> = std::env::var("KEYBENCH_ROWS")
        .map(|v| v.split(',').map(str::to_owned).collect())
        .unwrap_or_default();
    let secrets = env_usize("KEYBENCH_SECRETS", 50);
    let per_secret = env_usize("KEYBENCH_PER_SECRET", 20);
    let rec_secrets = env_usize("KEYBENCH_REC_SECRETS", 20);
    let rec_per = env_usize("KEYBENCH_REC_PER", 10);
    let threads = env_usize("KEYBENCH_THREADS", 8);
    let master = env_usize("KEYBENCH_SEED", 0x4851_2026_1010) as u64;
    let real: Vec<Vec<u8>> = std::env::var("KEYBENCH_VECTORS")
        .ok()
        .map(|path| {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            v["vectors"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| unhex(x["cell"].as_str().unwrap()))
                .collect()
        })
        .unwrap_or_default();
    for cell in &real {
        assert_eq!(cell.len(), CELL_BYTES);
        assert!(cell[RECORD_BYTES..].iter().all(|&b| b == 0));
    }

    let (var_s, var_k) = (sampler_variance(SIGMA_S), sampler_variance(SIGMA_K));
    let mut tsv = String::from(
        "row\tmode\tsecret\tresp\tn\tsum_err2\tsum_err4\tmax_err\tsum_fold2\tsum_pack2\tmax_pack\t\
         sum_key2\tsum_round2\tsum_fold_pack\tfold_first\tidentity_misses\tdecode_misses\t\
         records_checked\trecords_wrong\terr_gt3\terr_gt4\terr_gt5\tpack_gt3\tpack_gt4\tpack_gt5\tms\t\
         served_checked\tserved_wrong\n",
    );
    let mut summary = String::new();
    let mut failures: Vec<String> = Vec::new();
    println!(
        "real record vectors: {}; secrets x responses: worst {secrets} x {per_secret}, records \
         {rec_secrets} x {rec_per}; threads {threads}; seed {master:#x}",
        real.len()
    );

    for (ri, row) in ROWS.iter().enumerate() {
        if !filter.is_empty() && !filter.iter().any(|f| row.name.contains(f.as_str())) {
            continue;
        }
        let gamma = match row.shape {
            Shape::Single => 2 * CELL_BYTES,
            _ => 4 * CELL_BYTES,
        };
        let ctx = PackCtx::new(D, gamma, row.set.0, row.set.1, T);
        let probe = NoiseProbe::new(&ctx);
        let (q, p) = (ctx.lq.q as f64, ctx.lp.q as f64);
        let half = (ctx.delta / 2) as f64;
        let var_q = sampler_variance(row.sigma_q);

        for mode in [Mode::Worst, Mode::Records] {
            let (n_secrets, n_per) = match mode {
                Mode::Worst => (secrets, per_secret),
                Mode::Records => (rec_secrets, rec_per),
            };
            if n_secrets == 0 || n_per == 0 {
                continue;
            }
            let load_start = loadavg();
            let t_row = Instant::now();
            let mut data_rng =
                ChaCha20Rng::seed_from_u64(master ^ ((ri as u64) << 8) ^ (mode as u64) ^ 0xDA7A);
            let data = build_data(row, mode, &mut data_rng, &real);
            let segs: Vec<Segment<'_>> = data
                .shards
                .iter()
                .enumerate()
                .map(|(k, shard)| Segment { shard, query: k })
                .collect();
            let store = Store::build(&ctx, &segs);
            let rows_per_query = data.shards[0].rows as f64;
            let fold_model = match mode {
                Mode::Worst => (var_q * rows_per_query * 64.0).sqrt(),
                Mode::Records => f64::NAN,
            };
            let switches = (gamma - 1) as f64;
            let key_model = (switches * D as f64 * var_k * (q * q - 1.0) / 12.0).sqrt() / p;
            let round_model = (switches * D as f64 * var_s / 12.0 + 1.0 / 12.0).sqrt();
            let pack_model = (key_model.powi(2) + round_model.powi(2)).sqrt();
            let total_model = (fold_model.powi(2) + pack_model.powi(2)).sqrt();
            let job = Job {
                row,
                mode,
                ctx: &ctx,
                probe: &probe,
                data: &data,
                store: &store,
                seed: master ^ ((ri as u64) << 32),
                per_secret: n_per,
                sd_total: total_model,
                sd_pack: pack_model,
            };
            let next = AtomicUsize::new(0);
            let results: Mutex<Vec<Stat>> = Mutex::new(Vec::new());
            std::thread::scope(|scope| {
                for _ in 0..threads.min(n_secrets) {
                    scope.spawn(|| loop {
                        let secret = next.fetch_add(1, Ordering::Relaxed);
                        if secret >= n_secrets {
                            break;
                        }
                        let stats = run_secret(&job, secret);
                        results.lock().unwrap().extend(stats);
                    });
                }
            });
            let mut stats = results.into_inner().unwrap();
            stats.sort_by_key(|s| (s.secret, s.resp));

            let mode_name = format!("{mode:?}").to_lowercase();
            for s in &stats {
                writeln!(
                    tsv,
                    "{}\t{mode_name}\t{}\t{}\t{}\t{:.6e}\t{:.6e}\t{}\t{:.6e}\t{:.6e}\t{}\t{:.6e}\t{:.6e}\t{:.6e}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{}\t{}",
                    row.name,
                    s.secret,
                    s.resp,
                    s.n,
                    s.sum_err2,
                    s.sum_err4,
                    s.max_err,
                    s.sum_fold2,
                    s.sum_pack2,
                    s.max_pack,
                    s.sum_key2,
                    s.sum_round2,
                    s.sum_fold_pack,
                    s.fold_first.iter().map(i64::to_string).collect::<Vec<_>>().join(","),
                    s.identity_misses,
                    s.decode_misses,
                    s.records_checked,
                    s.records_wrong,
                    s.err_tail[0],
                    s.err_tail[1],
                    s.err_tail[2],
                    s.pack_tail[0],
                    s.pack_tail[1],
                    s.pack_tail[2],
                    s.ms,
                    s.served_checked,
                    s.served_wrong
                )
                .unwrap();
            }

            let responses = stats.len();
            let n: f64 = stats.iter().map(|s| s.n as f64).sum();
            let sum = |f: fn(&Stat) -> f64| stats.iter().map(f).sum::<f64>();
            let rms = |f: fn(&Stat) -> f64| (sum(f) / n).sqrt();
            let err_rms = rms(|s| s.sum_err2);
            let fold_rms = rms(|s| s.sum_fold2);
            let pack_rms = rms(|s| s.sum_pack2);
            let key_rms = rms(|s| s.sum_key2);
            let round_rms = rms(|s| s.sum_round2);
            let kurt = sum(|s| s.sum_err4) / n / err_rms.powi(4);
            let corr = sum(|s| s.sum_fold_pack) / n / (fold_rms * pack_rms);
            let max_err = stats.iter().map(|s| s.max_err).max().unwrap_or(0) as f64;
            let max_pack = stats.iter().map(|s| s.max_pack).max().unwrap_or(0) as f64;
            let identity: usize = stats.iter().map(|s| s.identity_misses).sum();
            let decode: usize = stats.iter().map(|s| s.decode_misses).sum();
            let rec_checked: usize = stats.iter().map(|s| s.records_checked).sum();
            let rec_wrong: usize = stats.iter().map(|s| s.records_wrong).sum();
            let served: usize = stats.iter().map(|s| s.served_checked).sum();
            let served_wrong: usize = stats.iter().map(|s| s.served_wrong).sum();
            let tail = |f: fn(&Stat) -> [usize; 3]| -> [f64; 3] {
                let mut t = [0usize; 3];
                for s in &stats {
                    let v = f(s);
                    for k in 0..3 {
                        t[k] += v[k];
                    }
                }
                [t[0] as f64 / n, t[1] as f64 / n, t[2] as f64 / n]
            };
            let err_tail = tail(|s| s.err_tail);
            let pack_tail = tail(|s| s.pack_tail);
            // Folds of distinct responses are independent; within a worst-value response
            // every column of a segment shares one.
            let firsts: Vec<f64> = stats
                .iter()
                .flat_map(|s| s.fold_first.iter().map(|&f| f as f64))
                .collect();
            let fold_first_rms =
                (firsts.iter().map(|f| f * f).sum::<f64>() / firsts.len() as f64).sqrt();
            // Standard error of the measured total rms, from the spread of per-response mean
            // squares.
            let per: Vec<f64> = stats.iter().map(|s| s.sum_err2 / s.n as f64).collect();
            let mean = per.iter().sum::<f64>() / per.len() as f64;
            let var = per.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (per.len() as f64 - 1.0).max(1.0);
            let se_bits = 0.5 * (var / per.len() as f64).sqrt() / mean / std::f64::consts::LN_2;

            let total_exact_fold = (fold_model.powi(2) + pack_rms.powi(2)).sqrt();
            let slack = |sd: f64| log2(half) - LOG2_R_SAME - log2(sd);

            let name = format!("{}/{mode_name}", row.name);
            let line = format!(
                "{{\"row\":\"{}\",\"mode\":\"{mode_name}\",\"q\":{},\"p\":{},\"sigma_q\":{},\"gamma_tot\":{gamma},\
                 \"secrets\":{n_secrets},\"responses\":{responses},\"coefficients\":{n},\
                 \"decode_misses\":{decode},\"identity_misses\":{identity},\
                 \"records_checked\":{rec_checked},\"records_wrong\":{rec_wrong},\
                 \"served_checked\":{served},\"served_wrong\":{served_wrong},\
                 \"log2_fold_rms\":{:.4},\"log2_fold_first_rms\":{:.4},\"log2_fold_model\":{:.4},\
                 \"log2_key_rms\":{:.4},\"log2_key_model\":{:.4},\
                 \"log2_round_rms\":{:.4},\"log2_round_model\":{:.4},\
                 \"log2_pack_rms\":{:.4},\"log2_pack_model\":{:.4},\
                 \"log2_total_rms\":{:.4},\"log2_total_model\":{:.4},\"total_rms_se_bits\":{:.4},\
                 \"log2_worst_sample\":{:.4},\"worst_sample_sigmas\":{:.3},\"worst_margin_bits\":{:.3},\
                 \"log2_worst_pack\":{:.4},\"kurtosis\":{:.4},\"fold_pack_corr\":{:.5},\
                 \"log2_half_step\":{:.4},\"err_tail_3_4_5\":[{:.3e},{:.3e},{:.3e}],\"pack_tail_3_4_5\":[{:.3e},{:.3e},{:.3e}],\
                 \"slack_measured\":{:.4},\"slack_exact_fold\":{:.4},\"slack_model\":{:.4},\
                 \"wall_s\":{:.1},\"load_start\":\"{load_start}\",\"load_end\":\"{}\"}}",
                row.name,
                ctx.lq.q,
                ctx.lp.q,
                row.sigma_q,
                log2(fold_rms),
                log2(fold_first_rms),
                log2(fold_model),
                log2(key_rms),
                log2(key_model),
                log2(round_rms),
                log2(round_model),
                log2(pack_rms),
                log2(pack_model),
                log2(err_rms),
                log2(total_model),
                se_bits,
                log2(max_err),
                max_err / err_rms,
                log2(half) - log2(max_err),
                log2(max_pack),
                kurt,
                corr,
                log2(half),
                err_tail[0],
                err_tail[1],
                err_tail[2],
                pack_tail[0],
                pack_tail[1],
                pack_tail[2],
                slack(err_rms),
                slack(total_exact_fold),
                slack(total_model),
                t_row.elapsed().as_secs_f64(),
                loadavg()
            );
            println!("{line}");
            summary.push_str(&line);
            summary.push('\n');

            if decode != 0 {
                failures.push(format!("{name}: {decode} coefficients decoded wrong"));
            }
            if identity != 0 {
                failures.push(format!(
                    "{name}: {identity} coefficients miss the error identity"
                ));
            }
            if rec_wrong != 0 {
                failures.push(format!("{name}: {rec_wrong} records not byte-exact"));
            }
            if served_wrong != 0 || served == 0 {
                failures.push(format!(
                    "{name}: {served_wrong} of {served} served responses differ from the probe"
                ));
            }
            if mode == Mode::Records && rec_checked == 0 {
                failures.push(format!("{name}: no record was checked"));
            }
            // The two packing terms are asserted one by one, then the fold.
            if (log2(key_rms) - log2(key_model)).abs() > 0.05 {
                failures.push(format!("{name}: key term off its expectation"));
            }
            if (log2(round_rms) - log2(round_model)).abs() > 0.05 {
                failures.push(format!("{name}: division term off its expectation"));
            }
            if mode == Mode::Worst && (log2(fold_first_rms) - log2(fold_model)).abs() > 0.15 {
                failures.push(format!("{name}: fold term off its expectation"));
            }
        }
    }

    std::fs::write(format!("{out_dir}/responses.tsv"), tsv).unwrap();
    std::fs::write(format!("{out_dir}/summary.jsonl"), summary).unwrap();
    if failures.is_empty() {
        println!("NOISE BENCH PASS");
    } else {
        for f in &failures {
            println!("FAIL {f}");
        }
        std::process::exit(1);
    }
}
