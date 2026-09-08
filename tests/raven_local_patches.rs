//! Locks each Raven-local deviation from upstream (see `UPSTREAM.md`) so a
//! later upstream pull cannot silently revert it.

use raven_inspire::math::GaussianSampler;
use raven_inspire::params::{InspireParams, InspireVariant, SecurityLevel, ShardConfig};

use raven_inspire::{
    encode_database, extract_inspiring, extract_with_variant, query_seeded,
    respond_seeded_inspiring, respond_seeded_inspiring_cached_with_session, respond_with_variant,
    setup, ClientSession, PackingMode, ServerInspiringCache, ServerResponse, ServerSessionStore,
};

fn small_params() -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

/// `entries_per_shard > ring_dim` is a typed error; upstream `debug_assert!`
/// vanishes in release and panics deep in `inverse_monomial` instead.
#[test]
fn commit_a_encode_database_returns_error_on_oversized_shard() {
    let params = small_params();
    let shard_config = ShardConfig {
        shard_size_bytes: 1 << 30,
        entry_size_bytes: 32,
        total_entries: 1 << 20,
    };
    // non-empty, else the is_empty early return masks the check
    let db = vec![0u8; 32 * 4];
    let err = encode_database(&db, 32, &params, &shard_config)
        .expect_err("oversized ShardConfig must return a typed error");
    let msg = err.to_string();
    assert!(
        msg.contains("entries_per_shard"),
        "error must explain the invariant; got: {msg}"
    );
    assert!(
        msg.contains("ring_dim"),
        "error must cite ring_dim; got: {msg}"
    );
}

/// `respond_with_variant(TwoPacking)` errors toward the seeded pipeline rather
/// than routing through OnePacking, which decodes to wrong plaintext.
#[test]
fn commit_b_respond_with_variant_twopacking_returns_error() {
    let params = small_params();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let num_entries = params.ring_dim;
    let entry_size = 2usize;
    let db: Vec<u8> = (0..num_entries)
        .flat_map(|i| vec![(i % 256) as u8, 0u8])
        .collect();
    let (crs, encoded_db, rlwe_sk) = setup(&params, &db, entry_size, &mut sampler).unwrap();

    let target_index = 42u64;
    let (_state, client_query) = raven_inspire::query(
        &crs,
        target_index,
        &encoded_db.config,
        &rlwe_sk,
        &mut sampler,
    )
    .unwrap();

    let err = respond_with_variant(&crs, &encoded_db, &client_query, InspireVariant::TwoPacking)
        .expect_err("TwoPacking on unseeded query must return a typed error");
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("seeded") || msg.to_lowercase().contains("twopacking"),
        "error must direct to the seeded path; got: {msg}"
    );
}

/// `extract_with_variant(TwoPacking)` routes to `extract_inspiring` on an
/// InspiRING-shaped response, matching what upstream's client did directly.
#[test]
fn commit_c_extract_with_variant_handles_inspiring_response() {
    let params = small_params();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let num_entries = params.ring_dim;
    let entry_size = 2usize;
    let db: Vec<u8> = (0..num_entries)
        .flat_map(|i| vec![(i % 256) as u8, 0u8])
        .collect();
    let (crs, encoded_db, rlwe_sk) = setup(&params, &db, entry_size, &mut sampler).unwrap();

    let target_index = 17u64;
    let (state, mut seeded_query) = query_seeded(
        &crs,
        target_index,
        &encoded_db.config,
        &rlwe_sk,
        &mut sampler,
    )
    .unwrap();
    seeded_query.packing_mode = PackingMode::Inspiring;
    let response: ServerResponse =
        respond_seeded_inspiring(&crs, &encoded_db, &seeded_query).unwrap();

    let via_inspiring = extract_inspiring(&crs, &state, &response, entry_size).unwrap();
    let via_wrapper = extract_with_variant(
        &crs,
        &state,
        &response,
        entry_size,
        InspireVariant::TwoPacking,
    )
    .unwrap();

    assert_eq!(via_inspiring, via_wrapper, "commit C must route TwoPacking extraction to extract_inspiring for InspiRING-shaped responses");
    let target_byte = (target_index as usize) * entry_size;
    assert_eq!(&via_wrapper[..], &db[target_byte..target_byte + entry_size]);
}

/// The `secure_128_d*` presets carry single-prime DEFAULT_Q, not the
/// under-provisioned 2-CRT pair whose product lands near 2^56.
///
/// `#[ignore]`d: 2^20 x 256 B allocates a 256 MiB database and runs ~10 s.
#[test]
#[ignore = "2^20 x 256 B cell: allocates a 256 MiB database and runs ~10 s under --release. \
            Trigger: changing the secure_128_d* presets, DEFAULT_Q, or the CRT form."]
fn commit_d_preset_default_q_passes_smoke_at_2_20_x_256b() {
    let params = InspireParams::secure_128_d2048();
    assert_eq!(
        params.q, 1_152_921_504_606_830_593,
        "secure_128_d2048 must ship DEFAULT_Q after commit D"
    );
    assert_eq!(
        params.crt_moduli,
        vec![1_152_921_504_606_830_593],
        "secure_128_d2048 must use single-prime CRT form after commit D"
    );

    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let entries: u64 = 1 << 20;
    let entry_size: usize = 256;
    let total = (entries as usize) * entry_size;
    let mut db = vec![0u8; total];
    for i in 0..entries as usize {
        for j in 0..entry_size {
            db[i * entry_size + j] = ((i + j) % 251) as u8;
        }
    }
    let (crs, encoded_db, sk) =
        setup(&params, &db, entry_size, &mut sampler).expect("setup must succeed");

    for &idx in &[entries / 4 - 1, entries / 2 - 1, 3 * entries / 4 - 1] {
        let (state, mut seeded_query) =
            query_seeded(&crs, idx, &encoded_db.config, &sk, &mut sampler).unwrap();
        seeded_query.packing_mode = PackingMode::Inspiring;
        let response = respond_seeded_inspiring(&crs, &encoded_db, &seeded_query).unwrap();
        let recovered = extract_inspiring(&crs, &state, &response, entry_size).unwrap();
        let expected: Vec<u8> = (0..entry_size)
            .map(|j| (((idx as usize) + j) % 251) as u8)
            .collect();
        assert_eq!(
            &recovered[..],
            &expected[..],
            "commit D: smoke byte-match must pass at index {idx}"
        );
    }
}

/// Registering packing keys once must shrink the wire query and still decode
/// byte-equal to the inlined-keys path.
#[test]
fn phase_b_handshake_roundtrip_byte_equal_to_inlined_keys() {
    let params = small_params();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 0);
    let num_entries = params.ring_dim;
    let entry_size = 2usize;
    let db: Vec<u8> = (0..num_entries)
        .flat_map(|i| vec![(i % 256) as u8, 0u8])
        .collect();
    let (crs, encoded_db, rlwe_sk) = setup(&params, &db, entry_size, &mut sampler).unwrap();

    let server_cache = ServerInspiringCache::new(&crs, &encoded_db).unwrap();
    let session_store = ServerSessionStore::new();

    let session_inline = ClientSession::new(crs.clone(), rlwe_sk.clone(), &mut sampler).unwrap();
    assert!(session_inline.session_handle().is_none());

    let (state_inline, mut q_inline) = session_inline
        .query_seeded(42, &encoded_db.config, &mut sampler)
        .unwrap();
    q_inline.packing_mode = PackingMode::Inspiring;

    assert!(
        q_inline.inspiring_packing_keys.is_some(),
        "pre-handshake session must inline the keys"
    );
    assert!(
        q_inline.session_handle.is_none(),
        "pre-handshake session must NOT reference a handle"
    );

    let response_inline = respond_seeded_inspiring(&crs, &encoded_db, &q_inline).unwrap();
    let decoded_inline =
        extract_inspiring(&crs, &state_inline, &response_inline, entry_size).unwrap();

    let mut session_hs = ClientSession::new(crs.clone(), rlwe_sk, &mut sampler).unwrap();
    let handle = session_hs
        .register_with(&session_store)
        .unwrap()
        .expect("InspiRING session must register a handle");
    assert_eq!(session_hs.session_handle(), Some(handle));
    assert_eq!(session_store.len(), 1);

    let (state_hs, mut q_hs) = session_hs
        .query_seeded(42, &encoded_db.config, &mut sampler)
        .unwrap();
    q_hs.packing_mode = PackingMode::Inspiring;

    assert!(
        q_hs.inspiring_packing_keys.is_none(),
        "handshake session must drop inlined keys"
    );
    assert_eq!(
        q_hs.session_handle,
        Some(handle),
        "handshake session must carry the handle"
    );

    let bytes_inline = bincode::serialize(&q_inline).unwrap().len();
    let bytes_hs = bincode::serialize(&q_hs).unwrap().len();
    assert!(
        bytes_hs < bytes_inline,
        "handshake query must be smaller than inlined: hs={bytes_hs} B, inline={bytes_inline} B"
    );

    let response_hs = respond_seeded_inspiring_cached_with_session(
        &crs,
        &encoded_db,
        &q_hs,
        &server_cache,
        Some(&session_store),
    )
    .unwrap();
    let decoded_hs = extract_inspiring(&crs, &state_hs, &response_hs, entry_size).unwrap();

    let target_byte = 42 * entry_size;
    let expected = &db[target_byte..target_byte + entry_size];
    assert_eq!(decoded_inline.as_slice(), expected);
    assert_eq!(decoded_hs.as_slice(), expected);
    assert_eq!(
        decoded_inline, decoded_hs,
        "handshake and inlined paths must produce the same plaintext"
    );
}
