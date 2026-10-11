//! Small-ring checks of the special-modulus packing path: exact decode, the exact error
//! identity, the stored step 1 against the direct one, and dumps for an independent
//! exact-integer reference.

use std::fmt::Write as _;

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use raven_inspire::howl_g::{
    expand_key_mask, expand_uniform, keygen, modulus_is_usable, query, respond, GCtx, NoiseProbe,
    Packed, Query, Rns, Secret, Segment, Shard, Step1, Store, G121, G128,
};
use raven_inspire::math::GaussianSampler;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Shape {
    Single,
    Pair,
    Dial,
}

#[derive(Clone, Copy)]
enum Vals {
    Worst,
    Random,
}

struct Case {
    ctx: GCtx,
    shards: Vec<Shard>,
    sk: Secret,
    key: Rns,
    key_err: Vec<i64>,
    w: Rns,
    queries: Vec<(Query, Vec<i64>, usize)>,
}

fn shard(rng: &mut ChaCha20Rng, rows: usize, cols: usize, vals: Vals) -> Shard {
    match vals {
        Vals::Worst => Shard::constant(rows, cols, -8),
        Vals::Random => Shard {
            rows,
            cols,
            vals: (0..rows * cols).map(|_| rng.gen_range(-8i8..=7)).collect(),
        },
    }
}

fn build(
    d: usize,
    gamma: usize,
    (q, p): (u64, u64),
    sigma_q: f64,
    shape: Shape,
    vals: Vals,
    seed: u64,
) -> Case {
    let ctx = GCtx::new(d, gamma, q, p, 16);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let shards = match shape {
        Shape::Single => vec![shard(&mut rng, d, gamma, vals)],
        Shape::Pair => vec![
            shard(&mut rng, d, gamma / 2, vals),
            shard(&mut rng, d, gamma / 2, vals),
        ],
        Shape::Dial => vec![shard(&mut rng, d / 2, gamma, vals)],
    };
    let mut s_sampler = GaussianSampler::with_seed(6.4, seed ^ 0x51);
    let mut k_sampler = GaussianSampler::with_seed(6.4, seed ^ 0x52);
    let mut q_sampler = GaussianSampler::with_seed(sigma_q, seed ^ 0x53);
    let sk = Secret::sample(&ctx, &mut s_sampler);
    let mut w_seed = [0u8; 32];
    rng.fill_bytes(&mut w_seed);
    let (key, key_err) = keygen(&ctx, &sk, w_seed, &mut k_sampler);
    let w = expand_key_mask(&ctx, w_seed);
    let queries = shards
        .iter()
        .enumerate()
        .map(|(k, sh)| {
            // Force both ring edges into the first two cases of every shape.
            let index = match (seed % 4, k) {
                (0, _) => 0,
                (1, _) => sh.rows - 1,
                _ => rng.gen_range(0..sh.rows),
            };
            let mut a_seed = [0u8; 32];
            rng.fill_bytes(&mut a_seed);
            let (qr, e) = query(&ctx, &sk, index, a_seed, &mut q_sampler);
            (qr, e, index)
        })
        .collect();
    Case {
        ctx,
        shards,
        sk,
        key,
        key_err,
        w,
        queries,
    }
}

impl Case {
    fn segs(&self) -> Vec<Segment<'_>> {
        self.shards
            .iter()
            .enumerate()
            .map(|(k, shard)| Segment { shard, query: k })
            .collect()
    }

    fn query_refs(&self) -> Vec<(&Query, &[i64], usize)> {
        self.queries
            .iter()
            .map(|(q, e, i)| (q, &e[..], *i))
            .collect()
    }
}

fn grid() -> Vec<(usize, usize, (u64, u64), f64, Shape)> {
    let mut out = Vec::new();
    for &(d, gamma) in &[(256usize, 32usize), (512, 64), (256, 128)] {
        for &(set, sigma_q) in &[(G128, 3.2), (G121, 6.4)] {
            for &shape in &[Shape::Single, Shape::Pair, Shape::Dial] {
                out.push((d, gamma, set, sigma_q, shape));
            }
        }
    }
    out
}

#[test]
fn design_primes_are_usable_and_sized() {
    for &(q, p) in &[G121, G128] {
        assert!(modulus_is_usable(q, 2048) && modulus_is_usable(p, 2048));
        assert_eq!(q % 4096, 1);
        assert_eq!(p % 4096, 1);
        assert_eq!(q % 16, 1, "decode residue");
        assert!(
            p > 2 * q - q / 1000,
            "special prime is at least about twice q"
        );
    }
    assert_eq!(64 - (G121.0 - 1).leading_zeros(), 29);
    assert_eq!(64 - (G121.1 - 1).leading_zeros(), 31);
    assert_eq!(64 - (G128.0 - 1).leading_zeros(), 28);
    assert_eq!(64 - (G128.1 - 1).leading_zeros(), 29);
    let bits = |(q, p): (u64, u64)| (q as f64).log2() + (p as f64).log2();
    assert!((bits(G121) - 59.9998).abs() < 1e-3);
    assert!((bits(G128) - 56.9995).abs() < 1e-3);
    // A composite and a prime without the root are both refused.
    assert!(!modulus_is_usable(4097, 2048), "17 * 241");
    assert!(
        !modulus_is_usable(268_435_459, 2048),
        "prime, but 3 mod 4096"
    );
}

#[test]
fn every_shape_decodes_and_meets_the_exact_error_identity() {
    let mut seed = 100;
    for (d, gamma, set, sigma_q, shape) in grid() {
        for vals in [Vals::Worst, Vals::Random] {
            for _ in 0..4 {
                seed += 1;
                let case = build(d, gamma, set, sigma_q, shape, vals, seed);
                let ctx = &case.ctx;
                let segs = case.segs();
                let probe = NoiseProbe::new(ctx);
                let sp = probe.secret(ctx, &case.sk);
                let w_ntt = case.w.to_ntt(ctx);
                let key_ntt = case.key.to_ntt(ctx);
                let store = Store::build(ctx, &segs);
                let qs = case.query_refs();

                let direct = probe.measure(
                    ctx,
                    &segs,
                    &Step1::Direct,
                    &case.sk,
                    &sp,
                    &w_ntt,
                    &key_ntt,
                    &case.key_err,
                    &qs,
                );
                let stored = probe.measure(
                    ctx,
                    &segs,
                    &Step1::Stored(&store),
                    &case.sk,
                    &sp,
                    &w_ntt,
                    &key_ntt,
                    &case.key_err,
                    &qs,
                );
                let tag = format!("d{d} g{gamma} q{} {shape:?} seed {seed}", set.0);
                assert_eq!(direct.packed, stored.packed, "stored step 1 differs: {tag}");
                assert_eq!(direct.decode_misses, 0, "decode: {tag}");
                assert_eq!(direct.identity_misses, 0, "error identity: {tag}");
                assert_eq!(direct.err.len(), gamma);

                let plain: Vec<&Query> = case.queries.iter().map(|(q, _, _)| q).collect();
                let served = respond(ctx, &segs, &Step1::Stored(&store), &w_ntt, &key_ntt, &plain);
                assert_eq!(
                    served, direct.packed,
                    "respond differs from the probe: {tag}"
                );
            }
        }
    }
}

#[test]
fn a_wrong_key_error_breaks_the_identity_and_a_wrong_key_breaks_decode() {
    let case = build(512, 64, G128, 3.2, Shape::Single, Vals::Random, 7);
    let ctx = &case.ctx;
    let segs = case.segs();
    let probe = NoiseProbe::new(ctx);
    let sp = probe.secret(ctx, &case.sk);
    let w_ntt = case.w.to_ntt(ctx);
    let qs = case.query_refs();
    let run = |key: &Rns, key_err: &[i64]| {
        probe.measure(
            ctx,
            &segs,
            &Step1::Direct,
            &case.sk,
            &sp,
            &w_ntt,
            &key.to_ntt(ctx),
            key_err,
            &qs,
        )
    };
    let honest = run(&case.key, &case.key_err);
    assert_eq!((honest.identity_misses, honest.decode_misses), (0, 0));

    // The claimed key error is off by one in one coefficient: decode holds, the identity must not.
    let mut claimed = case.key_err.clone();
    claimed[3] += 1;
    let off_by_one = run(&case.key, &claimed);
    assert_eq!(off_by_one.decode_misses, 0);
    assert!(
        off_by_one.identity_misses > 0,
        "the identity check cannot fail"
    );

    // A key for another secret: nothing decodes.
    let other = build(512, 64, G128, 3.2, Shape::Single, Vals::Random, 8);
    let foreign = run(&other.key, &other.key_err);
    assert!(foreign.decode_misses > 32, "decode check cannot fail");
}

#[test]
fn a_store_built_for_other_contents_gives_a_different_and_wrong_answer() {
    let case = build(512, 64, G121, 6.4, Shape::Single, Vals::Random, 21);
    let ctx = &case.ctx;
    let segs = case.segs();
    let probe = NoiseProbe::new(ctx);
    let sp = probe.secret(ctx, &case.sk);
    let w_ntt = case.w.to_ntt(ctx);
    let key_ntt = case.key.to_ntt(ctx);
    let qs = case.query_refs();
    let store = Store::build(ctx, &segs);

    // One value of one row changes after the store was built.
    let mut changed = case.shards[0].clone();
    let at = 5 * changed.cols + 9;
    changed.vals[at] = if changed.vals[at] == 3 { 4 } else { 3 };
    let new_segs = [Segment {
        shard: &changed,
        query: 0,
    }];
    let stale = probe.measure(
        ctx,
        &new_segs,
        &Step1::Stored(&store),
        &case.sk,
        &sp,
        &w_ntt,
        &key_ntt,
        &case.key_err,
        &qs,
    );
    let fresh = probe.measure(
        ctx,
        &new_segs,
        &Step1::Direct,
        &case.sk,
        &sp,
        &w_ntt,
        &key_ntt,
        &case.key_err,
        &qs,
    );
    assert_eq!(fresh.decode_misses, 0);
    assert_ne!(
        stale.packed, fresh.packed,
        "byte-identity check cannot fail"
    );
    assert!(stale.decode_misses > 0 || stale.identity_misses > 0);
}

#[test]
fn mask_expander_is_exact_and_pinned() {
    // A modulus just above 2^63 rejects almost half of all words; the survivors must be the
    // accepted words in order, reduced.
    let q = (1u64 << 63) + 29;
    let mut out = vec![0u64; 64];
    expand_uniform(&mut ChaCha20Rng::seed_from_u64(9), q, &mut out);
    let mut rng = ChaCha20Rng::seed_from_u64(9);
    let mut want = Vec::new();
    let mut rejected = 0;
    while want.len() < 64 {
        let u = rng.next_u64();
        if u < q {
            want.push(u);
        } else {
            rejected += 1;
        }
    }
    assert_eq!(out, want);
    assert!(rejected > 16, "the rejection branch was not exercised");

    // The key mask is a fixed function of the seed: limb q first, then limb p, one stream.
    let ctx = GCtx::new(2048, 512, G121.0, G121.1, 16);
    let w = expand_key_mask(&ctx, [7u8; 32]);
    assert!(w.q.iter().all(|&x| x < G121.0) && w.p.iter().all(|&x| x < G121.1));
    let digest =
        w.q.iter()
            .chain(&w.p)
            .fold(0xcbf2_9ce4_8422_2325u64, |h, &x| {
                (h ^ x).wrapping_mul(0x0100_0000_01b3)
            });
    let first = [w.q[0], w.q[1], w.p[0], w.p[1]];
    if let Ok(path) = std::env::var("HOWL_G_PIN_OUT") {
        std::fs::write(path, format!("{digest:#x} {first:?}\n")).unwrap();
    }
    assert_eq!(digest, PINNED_MASK_DIGEST, "first values {first:?}");
}

const PINNED_MASK_DIGEST: u64 = 3_556_523_134_167_814_496;

fn composed(ctx: &GCtx, x: &Rns) -> Vec<u64> {
    let qp = i128::from(ctx.lq.q) * i128::from(ctx.lp.q);
    x.centred(ctx)
        .iter()
        .map(|&v| v.rem_euclid(qp) as u64)
        .collect()
}

fn line<T: std::fmt::Display>(out: &mut String, name: &str, v: &[T]) {
    write!(out, "{name}").unwrap();
    for x in v {
        write!(out, " {x}").unwrap();
    }
    out.push('\n');
}

fn dump(name: &str, case: &Case, packed: &Packed, dir: &str) {
    let ctx = &case.ctx;
    let mut out = String::new();
    line(
        &mut out,
        "meta",
        &[
            ctx.d as u64,
            ctx.gamma as u64,
            ctx.lq.q,
            ctx.lp.q,
            ctx.t,
            ctx.g as u64,
            case.queries.len() as u64,
        ],
    );
    line(&mut out, "s", &case.sk.s);
    line(&mut out, "kerr", &case.key_err);
    line(&mut out, "w", &composed(ctx, &case.w));
    line(&mut out, "y", &composed(ctx, &case.key));
    let mut owner = Vec::new();
    let mut j = 0;
    for (k, sh) in case.shards.iter().enumerate() {
        for c in 0..sh.cols {
            line(&mut out, &format!("h{j}"), &sh.column(c, &ctx.lq, ctx.d));
            owner.push(k as u64);
            j += 1;
        }
    }
    line(&mut out, "owner", &owner);
    for (k, (qr, e, index)) in case.queries.iter().enumerate() {
        line(
            &mut out,
            &format!("qa{k}"),
            &raven_inspire::howl_g::expand_query_mask(ctx, qr.a_seed),
        );
        line(&mut out, &format!("qb{k}"), &qr.b);
        line(&mut out, &format!("qe{k}"), e);
        line(&mut out, &format!("idx{k}"), &[*index as u64]);
    }
    line(&mut out, "ahat", &packed.a);
    line(&mut out, "bfin", &packed.b);
    std::fs::write(format!("{dir}/{name}.txt"), out).unwrap();
}

#[test]
fn dump_cases_for_the_integer_reference() {
    let Ok(dir) = std::env::var("HOWL_G_DUMP_DIR") else {
        return;
    };
    let cases = [
        (
            "g128_d256_g32_single",
            256,
            32,
            G128,
            3.2,
            Shape::Single,
            Vals::Random,
            41,
        ),
        (
            "g121_d512_g64_single",
            512,
            64,
            G121,
            6.4,
            Shape::Single,
            Vals::Random,
            42,
        ),
        (
            "g128_d256_g64_pair",
            256,
            64,
            G128,
            3.2,
            Shape::Pair,
            Vals::Random,
            43,
        ),
        (
            "g121_d512_g64_dial",
            512,
            64,
            G121,
            6.4,
            Shape::Dial,
            Vals::Random,
            44,
        ),
        (
            "g121_d256_g128_pair_worst",
            256,
            128,
            G121,
            6.4,
            Shape::Pair,
            Vals::Worst,
            45,
        ),
        (
            "g128_d256_g32_single_edge0",
            256,
            32,
            G128,
            3.2,
            Shape::Single,
            Vals::Worst,
            48,
        ),
        (
            "g128_d256_g32_single_edgelast",
            256,
            32,
            G128,
            3.2,
            Shape::Single,
            Vals::Random,
            49,
        ),
    ];
    for (name, d, gamma, set, sigma_q, shape, vals, seed) in cases {
        let case = build(d, gamma, set, sigma_q, shape, vals, seed);
        let ctx = &case.ctx;
        let segs = case.segs();
        let store = Store::build(ctx, &segs);
        let plain: Vec<&Query> = case.queries.iter().map(|(q, _, _)| q).collect();
        let packed = respond(
            ctx,
            &segs,
            &Step1::Stored(&store),
            &case.w.to_ntt(ctx),
            &case.key.to_ntt(ctx),
            &plain,
        );
        dump(name, &case, &packed, &dir);
    }
}
