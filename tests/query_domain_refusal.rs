#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "test fixture failures must be loud"
)]

use raven_inspire::inspiring::{ClientPackingKeys, OfflinePackingKeys, PackParams};
use raven_inspire::math::{GaussianSampler, Poly};
use raven_inspire::params::{InspireParams, SecurityLevel};
use raven_inspire::pir::{
    respond_inspiring_cached_with_session, respond_seeded_inspiring_cached_with_session,
};
use raven_inspire::{
    extract, extract_two_packing, query_seeded, respond, setup, ClientQuery, SeededClientQuery,
    ServerCrs, ServerInspiringCache, ServerResponse, ServerSessionStore,
};

fn params() -> InspireParams {
    InspireParams {
        ring_dim: 256,
        q: 1_152_921_504_606_830_593,
        crt_moduli: vec![1_152_921_504_606_830_593],
        p: 65_537,
        sigma: 6.4,
        gadget_base: 1 << 20,
        query_gadget_len: 3,
        packing_gadget_len: 3,
        security_level: SecurityLevel::Bits128,
    }
}

fn forge_ntt_flag(poly: &Poly) -> Poly {
    let mut bytes = bincode::serialize(poly).expect("serialize coefficient-domain polynomial");
    let flag = bytes
        .last_mut()
        .expect("serialized polynomial has a domain flag");
    assert_eq!(
        *flag, 0,
        "fixture polynomial must start in coefficient domain"
    );
    *flag = 1;
    bincode::deserialize(&bytes).expect("generic Poly codec also carries trusted NTT caches")
}

#[test]
fn wire_seeded_rgsw_b_ntt_flag_is_refused_before_expansion() {
    let params = params();
    let entry_size = 32usize;
    let target = 5u64;
    let db: Vec<u8> = (0..params.ring_dim * entry_size)
        .map(|index| (index % 251) as u8)
        .collect();
    let expected_row = &db[target as usize * entry_size..(target as usize + 1) * entry_size];
    let mut sampler = GaussianSampler::with_seed(params.sigma, 7);
    let (crs, encoded, secret_key) =
        setup(&params, &db, entry_size, &mut sampler).expect("valid test setup");
    let (state, query) = query_seeded(&crs, target, &encoded.config, &secret_key, &mut sampler)
        .expect("valid test query");
    let cache = ServerInspiringCache::new(&crs, &encoded).expect("valid server cache");

    let honest = respond_seeded_inspiring_cached_with_session(&crs, &encoded, &query, &cache, None)
        .expect("honest query must respond");
    let honest_row =
        extract_two_packing(&crs, &state, &honest, entry_size).expect("honest query must extract");
    assert_eq!(honest_row.as_slice(), expected_row);

    let honest_wire = bincode::serialize(&query).expect("serialize honest query");
    let mut forged = query.clone();
    forged.rgsw_ciphertext.rows[0].b = forge_ntt_flag(&forged.rgsw_ciphertext.rows[0].b);
    let forged_wire = bincode::serialize(&forged).expect("serialize forged query");
    assert_eq!(forged_wire.len(), honest_wire.len());
    assert_eq!(
        forged_wire
            .iter()
            .zip(&honest_wire)
            .filter(|(left, right)| left != right)
            .count(),
        1,
        "only one domain flag byte may differ"
    );
    let decoded: SeededClientQuery =
        bincode::deserialize(&forged_wire).expect("decode forged query for boundary check");

    let error = match respond_seeded_inspiring_cached_with_session(
        &crs, &encoded, &decoded, &cache, None,
    ) {
        Ok(response) => {
            let row = extract_two_packing(&crs, &state, &response, entry_size)
                .expect("accepted forged response must extract for the negative control");
            panic!(
                "forged RGSW domain flag returned Ok; plaintext matched independent row: {}",
                row.as_slice() == expected_row
            );
        }
        Err(error) => error,
    };
    assert!(
        error.to_string().starts_with(
            "respond_seeded_inspiring_cached_with_session: coefficient-domain RGSW row 0 b"
        ),
        "wrong refusal: {error}"
    );
}

#[test]
fn wire_unseeded_rgsw_a_ntt_flag_is_refused_before_respond() {
    let params = params();
    let entry_size = 32usize;
    let db: Vec<u8> = (0..params.ring_dim * entry_size)
        .map(|index| (index % 251) as u8)
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 9);
    let (crs, encoded, secret_key) =
        setup(&params, &db, entry_size, &mut sampler).expect("valid test setup");
    let (_, seeded) = query_seeded(&crs, 5, &encoded.config, &secret_key, &mut sampler)
        .expect("valid test query");
    let honest = seeded.expand();
    let honest_wire = bincode::serialize(&honest).expect("serialize honest unseeded query");
    let mut forged = honest;
    forged.rgsw_ciphertext.rows[0].a = forge_ntt_flag(&forged.rgsw_ciphertext.rows[0].a);
    let forged_wire = bincode::serialize(&forged).expect("serialize forged unseeded query");
    assert_eq!(forged_wire.len(), honest_wire.len());
    assert_eq!(
        forged_wire
            .iter()
            .zip(&honest_wire)
            .filter(|(left, right)| left != right)
            .count(),
        1
    );
    let decoded: ClientQuery =
        bincode::deserialize(&forged_wire).expect("decode forged unseeded query");
    let Err(error) = respond(&crs, &encoded, &decoded) else {
        panic!("forged unseeded RGSW a domain flag returned Ok");
    };
    assert!(
        error
            .to_string()
            .contains("coefficient-domain RGSW row 0 a"),
        "wrong refusal: {error}"
    );
}

#[test]
fn wire_packing_key_ntt_flag_is_refused_before_cached_respond() {
    let params = params();
    let entry_size = 32usize;
    let db: Vec<u8> = (0..params.ring_dim * entry_size)
        .map(|index| (index % 251) as u8)
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 11);
    let (crs, encoded, secret_key) =
        setup(&params, &db, entry_size, &mut sampler).expect("valid test setup");
    let (_, mut query) = query_seeded(&crs, 5, &encoded.config, &secret_key, &mut sampler)
        .expect("valid test query");
    let cache = ServerInspiringCache::new(&crs, &encoded).expect("valid server cache");
    let honest_wire = bincode::serialize(&query).expect("serialize honest query");
    let keys = query
        .inspiring_packing_keys
        .as_mut()
        .expect("query carries inline packing keys");
    keys.y_body[0] = forge_ntt_flag(&keys.y_body[0]);
    let forged_wire = bincode::serialize(&query).expect("serialize forged query");
    assert_eq!(forged_wire.len(), honest_wire.len());
    assert_eq!(
        forged_wire
            .iter()
            .zip(&honest_wire)
            .filter(|(left, right)| left != right)
            .count(),
        1
    );
    let decoded: SeededClientQuery =
        bincode::deserialize(&forged_wire).expect("decode forged query for boundary check");

    let Err(error) =
        respond_inspiring_cached_with_session(&crs, &encoded, &decoded.expand(), &cache, None)
    else {
        panic!("forged packing-key domain flag returned Ok");
    };
    assert!(
        error
            .to_string()
            .contains("coefficient-domain packing-key y_body[0]"),
        "wrong refusal: {error}"
    );
}

#[test]
fn wire_packing_key_ntt_flag_is_refused_before_session_registration() {
    let params = params();
    let entry_size = 32usize;
    let db: Vec<u8> = (0..params.ring_dim * entry_size)
        .map(|index| (index % 251) as u8)
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 13);
    let (crs, encoded, secret_key) =
        setup(&params, &db, entry_size, &mut sampler).expect("valid test setup");
    let (_, query) = query_seeded(&crs, 5, &encoded.config, &secret_key, &mut sampler)
        .expect("valid test query");
    let cache = ServerInspiringCache::new(&crs, &encoded).expect("valid server cache");
    let mut keys = query
        .inspiring_packing_keys
        .expect("query carries inline packing keys");
    keys.y_body[0] = forge_ntt_flag(&keys.y_body[0]);
    let wire = bincode::serialize(&keys).expect("serialize forged session keys");
    let decoded: ClientPackingKeys =
        bincode::deserialize(&wire).expect("decode forged session keys");

    let store = ServerSessionStore::new();
    let Err(error) =
        store.register_server_side(decoded, cache.pack_params(), &params.ntt_context())
    else {
        panic!("forged session packing key was registered");
    };
    assert!(
        error
            .to_string()
            .contains("coefficient-domain packing-key y_body[0]"),
        "wrong refusal: {error}"
    );
}

#[test]
fn direct_store_refuses_ntt_flag_in_wire_z_body() {
    let params = params();
    let entry_size = 32usize;
    let db: Vec<u8> = (0..params.ring_dim * entry_size)
        .map(|index| (index % 251) as u8)
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 15);
    let (crs, encoded, secret_key) =
        setup(&params, &db, entry_size, &mut sampler).expect("valid test setup");
    let (_, query) = query_seeded(&crs, 5, &encoded.config, &secret_key, &mut sampler)
        .expect("valid test query");
    let mut keys = query
        .inspiring_packing_keys
        .expect("query carries inline packing keys");
    keys.z_body.push(forge_ntt_flag(&keys.y_body[0]));
    let wire = bincode::serialize(&keys).expect("serialize forged session keys");
    let decoded: ClientPackingKeys =
        bincode::deserialize(&wire).expect("decode forged session keys");

    let store = ServerSessionStore::new();
    let Err(error) = store.register(decoded) else {
        panic!("forged z_body polynomial was registered");
    };
    assert!(
        error
            .to_string()
            .contains("coefficient-domain packing-key z_body[0]"),
        "wrong refusal: {error}"
    );
}

#[test]
fn inline_packing_key_with_wrong_dimension_is_refused_before_respond() {
    let params = params();
    let entry_size = 32usize;
    let db: Vec<u8> = (0..params.ring_dim * entry_size)
        .map(|index| (index % 251) as u8)
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 16);
    let (crs, encoded, secret_key) =
        setup(&params, &db, entry_size, &mut sampler).expect("valid test setup");
    let (_, mut query) = query_seeded(&crs, 5, &encoded.config, &secret_key, &mut sampler)
        .expect("valid test query");
    let cache = ServerInspiringCache::new(&crs, &encoded).expect("valid server cache");
    let keys = query
        .inspiring_packing_keys
        .as_mut()
        .expect("query carries inline packing keys");
    keys.y_body[0] = Poly::from_coeffs_moduli(vec![2u64; 1024], params.moduli());
    let wire = bincode::serialize(&query).expect("serialize forged query");
    let decoded: SeededClientQuery =
        bincode::deserialize(&wire).expect("self-consistent forged polynomial must decode");

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        respond_seeded_inspiring_cached_with_session(&crs, &encoded, &decoded, &cache, None)
    }));
    let Ok(Err(error)) = outcome else {
        panic!("wrong-dimension inline packing key must return a typed error");
    };
    assert!(
        error
            .to_string()
            .contains("packing-key y_body[0] dimension 1024"),
        "wrong refusal: {error}"
    );
}

#[test]
fn registered_packing_key_with_wrong_modulus_is_refused_before_derivation() {
    let params = params();
    let entry_size = 32usize;
    let db: Vec<u8> = (0..params.ring_dim * entry_size)
        .map(|index| (index % 251) as u8)
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 18);
    let (crs, encoded, secret_key) =
        setup(&params, &db, entry_size, &mut sampler).expect("valid test setup");
    let (_, query) = query_seeded(&crs, 5, &encoded.config, &secret_key, &mut sampler)
        .expect("valid test query");
    let cache = ServerInspiringCache::new(&crs, &encoded).expect("valid server cache");
    let mut keys = query
        .inspiring_packing_keys
        .expect("query carries inline packing keys");
    keys.y_body[0] = Poly::from_coeffs_moduli(vec![2u64; params.ring_dim], &[12_289]);
    let wire = bincode::serialize(&keys).expect("serialize forged packing keys");
    let decoded: ClientPackingKeys =
        bincode::deserialize(&wire).expect("self-consistent forged polynomial must decode");

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ServerSessionStore::new().register_server_side(
            decoded,
            cache.pack_params(),
            &params.ntt_context(),
        )
    }));
    let Ok(Err(error)) = outcome else {
        panic!("wrong-modulus session packing key must return a typed error");
    };
    assert!(
        error.to_string().contains("packing-key y_body[0] moduli"),
        "wrong refusal: {error}"
    );
}

#[test]
fn registered_z_body_with_wrong_dimension_is_refused_before_derivation() {
    let params = params();
    let entry_size = 32usize;
    let db: Vec<u8> = (0..params.ring_dim * entry_size)
        .map(|index| (index % 251) as u8)
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 20);
    let (crs, encoded, secret_key) =
        setup(&params, &db, entry_size, &mut sampler).expect("valid test setup");
    let (_, query) = query_seeded(&crs, 5, &encoded.config, &secret_key, &mut sampler)
        .expect("valid test query");
    let cache = ServerInspiringCache::new(&crs, &encoded).expect("valid server cache");
    let mut keys = query
        .inspiring_packing_keys
        .expect("query carries inline packing keys");
    keys.z_body
        .push(Poly::from_coeffs_moduli(vec![2u64; 1024], params.moduli()));
    let wire = bincode::serialize(&keys).expect("serialize forged packing keys");
    let decoded: ClientPackingKeys =
        bincode::deserialize(&wire).expect("self-consistent forged polynomial must decode");

    let Err(error) = ServerSessionStore::new().register_server_side(
        decoded,
        cache.pack_params(),
        &params.ntt_context(),
    ) else {
        panic!("wrong-dimension z_body packing key was registered");
    };
    assert!(
        error
            .to_string()
            .contains("packing-key z_body[0] dimension 1024"),
        "wrong refusal: {error}"
    );
}

#[test]
fn trusted_ntt_offline_keys_still_round_trip() {
    let params = params();
    let pack_params = PackParams::try_new(&params, 16).expect("legal 32-byte row width");
    let keys = OfflinePackingKeys::generate(&pack_params, [7u8; 32]);
    assert!(
        keys.w_all_ntt.iter().flatten().any(Poly::is_ntt),
        "fixture must contain real NTT-domain polynomials"
    );
    let wire = bincode::serialize(&keys).expect("serialize trusted offline keys");
    let decoded: OfflinePackingKeys =
        bincode::deserialize(&wire).expect("decode trusted offline keys");
    assert_eq!(
        bincode::serialize(&decoded).expect("re-serialize trusted offline keys"),
        wire,
        "the query guard must not rewrite generic NTT-capable Poly storage"
    );
}

#[derive(Clone, Copy)]
enum FullResponsePoly {
    AggregateA,
    AggregateB,
    ColumnA,
    ColumnB,
}

fn assert_full_response_domain_refusal(location: FullResponsePoly, error_field: &str) {
    let params = params();
    let entry_size = 32usize;
    let target = 5u64;
    let db: Vec<u8> = (0..params.ring_dim * entry_size)
        .map(|index| (index % 251) as u8)
        .collect();
    let expected_row = &db[target as usize * entry_size..(target as usize + 1) * entry_size];
    let mut sampler = GaussianSampler::with_seed(params.sigma, 17);
    let (crs, encoded, secret_key) =
        setup(&params, &db, entry_size, &mut sampler).expect("valid test setup");
    let (state, query) = query_seeded(&crs, target, &encoded.config, &secret_key, &mut sampler)
        .expect("valid test query");
    let honest = respond(&crs, &encoded, &query.expand()).expect("honest query must respond");
    let honest_wire = honest.to_binary().expect("serialize honest full response");
    let decoded = ServerResponse::from_binary(&honest_wire).expect("decode honest full response");
    let honest_row =
        extract(&crs, &state, &decoded, entry_size).expect("honest response must extract");
    assert_eq!(honest_row.as_slice(), expected_row);

    let mut forged = honest;
    let poly = match location {
        FullResponsePoly::AggregateA => &mut forged.ciphertext.a,
        FullResponsePoly::AggregateB => &mut forged.ciphertext.b,
        FullResponsePoly::ColumnA => &mut forged.column_ciphertexts[0].a,
        FullResponsePoly::ColumnB => &mut forged.column_ciphertexts[0].b,
    };
    *poly = forge_ntt_flag(poly);
    let forged_wire = forged.to_binary().expect("serialize forged full response");
    assert_eq!(forged_wire.len(), honest_wire.len());
    assert_eq!(
        forged_wire
            .iter()
            .zip(&honest_wire)
            .filter(|(left, right)| left != right)
            .count(),
        1,
        "only one response domain flag byte may differ"
    );

    let error = match ServerResponse::from_binary(&forged_wire) {
        Ok(decoded) => {
            if matches!(location, FullResponsePoly::ColumnA) {
                let row = extract(&crs, &state, &decoded, entry_size)
                    .expect("accepted forged column response must extract for the control");
                panic!(
                    "forged full response decoded and extracted Ok; row matched independent data: {}",
                    row.as_slice() == expected_row
                );
            }
            panic!("forged full response {error_field} decoded without refusal");
        }
        Err(error) => error,
    };
    assert!(
        error.to_string().contains(error_field),
        "wrong refusal: {error}"
    );
}

#[test]
fn full_response_column_a_ntt_flag_is_refused() {
    assert_full_response_domain_refusal(FullResponsePoly::ColumnA, "column ciphertext[0] a");
}

#[test]
fn full_response_column_b_ntt_flag_is_refused() {
    assert_full_response_domain_refusal(FullResponsePoly::ColumnB, "column ciphertext[0] b");
}

#[test]
fn full_response_aggregate_a_ntt_flag_is_refused() {
    assert_full_response_domain_refusal(FullResponsePoly::AggregateA, "full response a");
}

#[test]
fn full_response_aggregate_b_ntt_flag_is_refused() {
    assert_full_response_domain_refusal(FullResponsePoly::AggregateB, "full response b");
}

#[test]
fn full_crs_galois_key_ntt_flag_is_refused_at_decode() {
    let params = params();
    let db: Vec<u8> = (0..params.ring_dim * 32)
        .map(|index| (index % 251) as u8)
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 19);
    let (crs, _, _) = setup(&params, &db, 32, &mut sampler).expect("valid test setup");
    let honest_wire = crs.to_versioned_bytes().expect("serialize honest full CRS");
    ServerCrs::from_versioned_bytes(&honest_wire).expect("decode honest full CRS");

    let mut forged = crs;
    forged.galois_keys[0].rows[0].a = forge_ntt_flag(&forged.galois_keys[0].rows[0].a);
    let forged_wire = forged
        .to_versioned_bytes()
        .expect("serialize forged full CRS");
    assert_eq!(forged_wire.len(), honest_wire.len());
    assert_eq!(
        forged_wire
            .iter()
            .zip(&honest_wire)
            .filter(|(left, right)| left != right)
            .count(),
        1
    );
    let Err(error) = ServerCrs::from_versioned_bytes(&forged_wire) else {
        panic!("forged full CRS galois-key domain flag decoded");
    };
    assert!(
        error
            .to_string()
            .contains("coefficient-domain CRS galois key[0] row[0] a"),
        "wrong refusal: {error}"
    );
}

#[test]
fn tree_only_crs_galois_key_ntt_flag_is_refused_before_width_shortcut() {
    let params = params();
    let db: Vec<u8> = (0..params.ring_dim * 32)
        .map(|index| (index % 251) as u8)
        .collect();
    let mut sampler = GaussianSampler::with_seed(params.sigma, 21);
    let (mut crs, _, _) = setup(&params, &db, 32, &mut sampler).expect("valid test setup");
    crs.inspiring_num_columns = 0;
    crs.galois_keys[0].rows[0].b = forge_ntt_flag(&crs.galois_keys[0].rows[0].b);
    let wire = crs.to_versioned_bytes().expect("serialize tree-only CRS");
    let Err(error) = ServerCrs::from_versioned_bytes(&wire) else {
        panic!("tree-only CRS bypassed the galois-key domain guard");
    };
    assert!(
        error
            .to_string()
            .contains("coefficient-domain CRS galois key[0] row[0] b"),
        "wrong refusal: {error}"
    );
}
