//! Byte budget for the d=2048 session: what the client transfers, and what the
//! server keeps per client and per table. Holds the CRS under 2 MiB and decodes a
//! query through a round-tripped CRS, which is what actually catches a `ServerCrs`
//! wire-format change.

#![allow(
    clippy::expect_used,
    reason = "test-target fixture helpers; an abort here is the failure report"
)]

use raven_inspire::inspiring::{ClientPackingKeys, PackParams};
use raven_inspire::math::GaussianSampler;
use raven_inspire::params::InspireParams;
use raven_inspire::pir::{extract_inspiring, query, respond_inspiring, PackingKeySeat, ServerCrs};
use raven_inspire::setup;
use serde::Serialize;

fn ssize<T: Serialize>(v: &T) -> u64 {
    bincode::serialized_size(v).expect("serialized_size")
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn profile(entry_size: usize) {
    let params = InspireParams::secure_128_d2048();
    let d = params.ring_dim;
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);

    let db = vec![0u8; d * entry_size];
    let (crs, encoded_db, rlwe_sk) = setup(&params, &db, entry_size, &mut sampler).expect("setup");

    let galois = ssize(&crs.galois_keys);
    let crs_total = ssize(&crs);
    let rlwe = ssize(&rlwe_sk);

    let pp = PackParams::try_new(&crs.params, crs.inspiring_num_columns)
        .expect("CRS width must be legal");
    let pp_resident = ssize(&pp);
    let keys = ClientPackingKeys::generate(
        &rlwe_sk,
        &crs.params,
        crs.inspiring_num_columns,
        crs.inspiring_w_seed,
        &mut sampler,
    )
    .expect("CRS width must be legal");
    let y_body = ssize(&keys.y_body);
    let z_body = ssize(&keys.z_body);
    let seat = PackingKeySeat::from_wire(keys, &pp, &crs.params.ntt_context()).expect("seat");

    let transferred = crs_total + y_body + z_body + rlwe;

    eprintln!("================================================================");
    eprintln!(
        "d={d}  entry_size={entry_size} B  num_columns(gamma)={}  num_to_pack={}",
        crs.inspiring_num_columns, pp.num_to_pack
    );
    eprintln!("---- transferred / persisted (bincode-serialized) ----");
    eprintln!("  galois_keys          {:>10.2} MiB", mib(galois));
    eprintln!("  ServerCrs total      {:>10.2} MiB", mib(crs_total));
    eprintln!("  y_body (+z_body)     {:>10.2} MiB", mib(y_body + z_body));
    eprintln!("  rlwe_sk              {:>10.2} MiB", mib(rlwe));
    eprintln!("  >>> TRANSFERRED      {:>10.2} MiB", mib(transferred));
    eprintln!("---- server resident ----");
    eprintln!(
        "  pack_params tables   {:>10.2} MiB  (per table)",
        mib(pp_resident)
    );
    eprintln!(
        "  packing-key seat     {:>10} B    (per client)",
        seat.resident_bytes()
    );

    // a CRS past 2 MiB means a public matrix re-entered the wire format
    assert!(
        crs_total < 2 * 1024 * 1024,
        "ServerCrs is {:.2} MiB; expected < 2 MiB. A >2 MiB CRS means a large field \
         (e.g. a reintroduced public a-matrix) crept back into the client wire format.",
        mib(crs_total)
    );

    // extract through the deserialized CRS: a field dropped on the wire fails here
    let crs2: ServerCrs = bincode::deserialize(&bincode::serialize(&crs).expect("serialize crs"))
        .expect("deserialize crs");
    let (state, q) = query(&crs2, 0, &encoded_db.config, &rlwe_sk, &mut sampler).expect("query");
    let response = respond_inspiring(&crs2, &encoded_db, &q).expect("respond");
    let decoded = extract_inspiring(&crs2, &state, &response, entry_size).expect("extract");
    assert_eq!(
        decoded.as_slice(),
        &db[0..entry_size],
        "production-cell decode through a deserialized CRS must match the planted entry"
    );
}

#[test]
fn session_byte_profile() {
    for es in [32usize, 256, 512] {
        profile(es);
    }
}
