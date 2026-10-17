//! Timing of the served special-modulus set at full size: one response on one thread per
//! kernel, sixteen responses in flight, the per-shard table build, and the client steps.
//!
//! Cells are interleaved round-robin with a rotating start and the 1-minute load average is
//! recorded with every sample, so a busy box shows in the numbers rather than hiding in
//! them. Run with `cargo bench --bench special_modulus_bench`; add `--features
//! simd-packing-offline` for the AVX-512 kernel. SPECIAL_MODULUS_BENCH_ROUNDS sets the
//! rounds (default 36).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]

use std::hint::black_box;
use std::time::Instant;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use raven_inspire::special_modulus::{Client, Kernel, Params, Server};

fn load1() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()))
        .unwrap_or(f64::NAN)
}

struct Cell<'a> {
    name: String,
    run: Box<dyn FnMut(usize) + 'a>,
    ms: Vec<f64>,
    load: Vec<f64>,
}

fn report(cells: &[Cell<'_>]) {
    for c in cells {
        let mut v = c.ms.clone();
        v.sort_by(f64::total_cmp);
        let at = |p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
        let (lo, hi) = c
            .load
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        println!(
            "{:<34} n {:>3}  min {:>8.3}  median {:>8.3}  p90 {:>8.3}  max {:>8.3} ms  load {:.1}-{:.1}",
            c.name,
            v.len(),
            v[0],
            at(0.5),
            at(0.9),
            v[v.len() - 1],
            lo,
            hi
        );
    }
}

fn interleave(cells: &mut [Cell<'_>], rounds: usize) {
    for c in cells.iter_mut() {
        (c.run)(usize::MAX);
    }
    for r in 0..rounds {
        let n = cells.len();
        for k in 0..n {
            let c = &mut cells[(r + k) % n];
            let load = load1();
            let t = Instant::now();
            (c.run)(r);
            c.ms.push(t.elapsed().as_secs_f64() * 1e3);
            c.load.push(load);
        }
    }
}

fn main() {
    let rounds = std::env::var("SPECIAL_MODULUS_BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(36);
    let params = Params::ring2048_cell256();
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let mut cells = vec![0u8; 2048 * 256];
    for cell in cells.as_chunks_mut::<256>().0 {
        rng.fill_bytes(&mut cell[..246]);
    }
    let kernels: Vec<Kernel> = [Kernel::Portable, Kernel::Avx512]
        .into_iter()
        .filter(|k| k.available())
        .collect();
    let servers: Vec<Server> = kernels
        .iter()
        .map(|&k| Server::with_kernel(&params, k).unwrap())
        .collect();
    let tables: Vec<_> = servers.iter().map(|s| s.table(&cells).unwrap()).collect();
    let mut client = Client::new(&params).unwrap();
    let seat = servers[0].seat(client.packing_key()).unwrap();
    let queries: Vec<_> = (0..64)
        .map(|i| client.query(i * 31 % 2048).unwrap())
        .collect();
    let in_flight = 16;
    println!(
        "special-modulus served set: ring {}, gamma {}, table {} B per shard, seat {} B; {} rounds; \
         {} threads available",
        params.ring_dim,
        params.gamma,
        tables[0].bytes(),
        seat.bytes(),
        rounds,
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get)
    );

    let response = servers[0].respond(&tables[0], &seat, &queries[0]).unwrap();
    let client = std::cell::RefCell::new(client);
    let mut timed: Vec<Cell<'_>> = Vec::new();
    for (server, table) in servers.iter().zip(&tables) {
        let (seat, queries) = (&seat, &queries);
        timed.push(Cell {
            name: format!("respond, 1 thread, {:?}", server.kernel()),
            run: Box::new(move |r| {
                let q = &queries[r % queries.len()];
                black_box(server.respond(table, seat, q).unwrap());
            }),
            ms: Vec::new(),
            load: Vec::new(),
        });
        timed.push(Cell {
            name: format!(
                "respond, {in_flight} in flight, {:?} (wall)",
                server.kernel()
            ),
            run: Box::new(move |r| {
                std::thread::scope(|s| {
                    for k in 0..in_flight {
                        let q = &queries[(r + k) % queries.len()];
                        s.spawn(move || black_box(server.respond(table, seat, q).unwrap()));
                    }
                });
            }),
            ms: Vec::new(),
            load: Vec::new(),
        });
        let client = &client;
        timed.push(Cell {
            name: format!("server: seat a key, {:?}", server.kernel()),
            run: Box::new(move |_| {
                black_box(server.seat(client.borrow().packing_key()).unwrap());
            }),
            ms: Vec::new(),
            load: Vec::new(),
        });
    }
    let (client, params_ref, response) = (&client, &params, &response);
    timed.push(Cell {
        name: "client: new (secret + key)".into(),
        run: Box::new(move |_| {
            black_box(Client::new(params_ref).unwrap());
        }),
        ms: Vec::new(),
        load: Vec::new(),
    });
    timed.push(Cell {
        name: "client: query".into(),
        run: Box::new(move |r| {
            black_box(client.borrow_mut().query(r % 2048).unwrap());
        }),
        ms: Vec::new(),
        load: Vec::new(),
    });
    timed.push(Cell {
        name: "client: decode".into(),
        run: Box::new(move |_| {
            black_box(client.borrow().decode(response).unwrap());
        }),
        ms: Vec::new(),
        load: Vec::new(),
    });
    interleave(&mut timed, rounds);
    report(&timed);

    // The table build is long; fewer rounds.
    let mut builds: Vec<Cell<'_>> = servers
        .iter()
        .map(|server| {
            let cells = &cells;
            Cell {
                name: format!("server: table build, {:?}", server.kernel()),
                run: Box::new(move |_| {
                    black_box(server.table(cells).unwrap());
                }),
                ms: Vec::new(),
                load: Vec::new(),
            }
        })
        .collect();
    interleave(&mut builds, rounds.min(8));
    report(&builds);
}
