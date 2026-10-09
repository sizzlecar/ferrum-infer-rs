use super::*;

// Exact legacy field ordering, independent of the new Arc/cache representation.
#[derive(Serialize)]
struct LegacyPrepared<'a> {
    contract: &'a CompositeNumericalArithmetic,
    projections: &'a [PreparedProjection],
}

#[derive(Serialize)]
struct LegacyWave<'a> {
    prepared_numerics: LegacyPrepared<'a>,
    facts: &'a UpstreamProjectionWaveFacts,
    route: &'a PreparedUpstreamProjectionRoute,
}

fn values() -> Vec<ResolvedValueBinding> {
    vec![
        binding(2, &[Some("quantization.gguf.iq4-xs")], 256, 32, false),
        binding(3, &[None], 256, 32, false),
        binding(4, &[None], 256, 32, false),
        binding(5, &[None], 256, 32, false),
    ]
}

fn legacy(prepared: &PreparedProjectionNumerics) -> LegacyPrepared<'_> {
    LegacyPrepared {
        contract: prepared.contract(),
        projections: prepared.projections(),
    }
}

fn facts(rows: u32) -> UpstreamProjectionWaveFacts {
    UpstreamProjectionWaveFacts {
        role: ProjectionRole::CausalQuery,
        component_id: id("component.2.0"),
        local_rows: rows,
        layout: UpstreamProjectionLayout::Columns,
        input_stride: 256,
        output_stride: 32,
        input_byte_offset: 0,
        output_byte_offset: 0,
        weight_byte_offset: 0,
        input_available_bytes: u64::from(rows) * 256 * 2,
        output_available_bytes: u64::from(rows) * 32 * 2,
        weight_available_bytes: 32 * 136,
        retained_zero_padded_weight_rows: 32,
    }
}

fn native(mmq: bool) -> UpstreamNativePlanFacts {
    UpstreamNativePlanFacts {
        implementation_fingerprint: "native.fixture.sharing".into(),
        device_architecture: 1200,
        multiprocessors: 128,
        geometry: if mmq {
            UpstreamNativeGeometry::Mmq {
                padded_inputs: 512,
                row_tile: 32,
                column_tile: 128,
                threads: 256,
                shared_bytes: 32768,
                packed_guard_blocks: 32,
                blocks: 128,
                fixup: true,
            }
        } else {
            UpstreamNativeGeometry::Mmvq {
                padded_inputs: 512,
                padded_outputs: 32,
                columns: 8,
                channels: 1,
                warps: 4,
                rows_per_block: 2,
            }
        },
        maximum_dynamic_shared_bytes: 65536,
    }
}

#[test]
fn shared_prepared_and_wave_keep_legacy_bytes_and_fingerprints() {
    let contracts = [
        Q8ActAttentionProfile::Causal.arithmetic(),
        UpstreamMarkerV2Profile::Causal.arithmetic(),
        UpstreamMarkerV2Profile::CausalPrefill.arithmetic(),
        UpstreamMarkerV2Profile::CausalG32MmqPrefill.arithmetic(),
    ];
    for contract in contracts {
        let prepared = PreparedProjectionNumerics::prepare(&contract, &values()).unwrap();
        let bytes = serde_json::to_vec(&legacy(&prepared)).unwrap();
        let expected = format!("{:x}", Sha256::digest(&bytes));
        let restored: PreparedProjectionNumerics = serde_json::from_slice(&bytes).unwrap();
        assert!(restored.data.static_contract_validation.get().is_none());
        assert!(restored.data.fingerprint.get().is_none());
        for p in [&prepared, &restored, &restored.clone()] {
            assert_eq!(serde_json::to_vec(p).unwrap(), bytes);
            p.validate_static_contract().unwrap();
            p.validate_bindings(&contract, &values()).unwrap();
            assert_eq!(p.fingerprint(), expected);
            assert_eq!(serde_json::to_vec(p).unwrap(), bytes);
            assert_eq!(p, &prepared);
        }
    }
    for (profile, rows, native) in [
        (UpstreamMarkerV2Profile::Causal, 8, Some(native(false))),
        (
            UpstreamMarkerV2Profile::CausalPrefill,
            33,
            Some(native(true)),
        ),
        (UpstreamMarkerV2Profile::CausalG32MmqPrefill, 8, None),
        (UpstreamMarkerV2Profile::CausalPrefill, 2049, None),
    ] {
        let prepared =
            PreparedProjectionNumerics::prepare(&profile.arithmetic(), &values()).unwrap();
        let facts = facts(rows);
        let wave =
            PreparedUpstreamProjectionWave::prepare(&prepared, &facts, native.as_ref()).unwrap();
        let bytes = serde_json::to_vec(&LegacyWave {
            prepared_numerics: legacy(&prepared),
            facts: &facts,
            route: wave.route(),
        })
        .unwrap();
        assert_eq!(serde_json::to_vec(&wave).unwrap(), bytes);
        assert_eq!(
            wave.fingerprint().unwrap(),
            format!("{:x}", Sha256::digest(&bytes))
        );
        let restored: PreparedUpstreamProjectionWave = serde_json::from_slice(&bytes).unwrap();
        restored
            .validate_reconstructed(&profile.arithmetic(), &values(), &facts, native.as_ref())
            .unwrap();
        assert_eq!(restored, wave);
        assert_eq!(restored.fingerprint().unwrap(), wave.fingerprint().unwrap());
    }
}

#[test]
fn deserialized_invalid_contract_never_inherits_a_valid_cache() {
    let contract = UpstreamMarkerV2Profile::CausalPrefill.arithmetic();
    let prepared = PreparedProjectionNumerics::prepare(&contract, &values()).unwrap();
    prepared.validate_static_contract().unwrap();
    let _ = prepared.fingerprint();
    let invalid = tamper(&prepared, |data| data.contract.schema_version = u32::MAX);
    assert!(invalid.data.static_contract_validation.get().is_none());
    // The first consumer is the real wave preparation entrypoint.
    assert!(PreparedUpstreamProjectionWave::prepare(&invalid, &facts(2049), None).is_err());
    assert!(matches!(
        invalid.data.static_contract_validation.get(),
        Some(Err(_))
    ));
    let first = invalid.validate_static_contract().unwrap_err();
    assert_eq!(
        invalid.clone().validate_static_contract().unwrap_err(),
        first
    );
    assert!(PreparedUpstreamProjectionWave::prepare(&invalid, &facts(2049), None).is_err());
    assert!(invalid.validate_bindings(&contract, &values()).is_err());
    assert!(prepared.validate_static_contract().is_ok());
    for field in ["static_contract_validation", "fingerprint", "data"] {
        let mut wire = serde_json::to_value(&prepared).unwrap();
        wire[field] = serde_json::json!(true);
        assert!(serde_json::from_value::<PreparedProjectionNumerics>(wire).is_err());
    }
}

#[test]
fn static_validation_cache_does_not_authorize_forged_physical_bindings() {
    let contract = UpstreamMarkerV2Profile::CausalPrefill.arithmetic();
    let prepared = PreparedProjectionNumerics::prepare(&contract, &values()).unwrap();
    for mutate in [
        (|d: &mut PreparedProjectionNumericsData| {
            d.projections[0].leaves[0].component_id = id("forged.component");
        }) as fn(&mut PreparedProjectionNumericsData),
        |d| d.projections[0].leaves[0].output_offset = 1,
        |d| {
            d.projections[0].leaves[0].route = PreparedProjectionRoute::StrictBase {
                reason: StrictProjectionReason::FormatNotDeclared,
            }
        },
    ] {
        let changed = tamper(&prepared, mutate);
        // Its declaration is still valid; that says nothing about its leaves.
        changed.validate_static_contract().unwrap();
        assert!(changed.validate_bindings(&contract, &values()).is_err());
    }
    let mut different = values();
    different[0] = binding(2, &[Some("quantization.gguf.q5-k")], 256, 32, false);
    prepared.validate_static_contract().unwrap();
    assert!(prepared.validate_bindings(&contract, &different).is_err());
}

#[test]
fn shared_metadata_survives_owner_drop_and_concurrent_validation() {
    let contract = UpstreamMarkerV2Profile::CausalPrefill.arithmetic();
    let prepared = PreparedProjectionNumerics::prepare(&contract, &values()).unwrap();
    let bytes = serde_json::to_vec(&prepared).unwrap();
    let restored: PreparedProjectionNumerics = serde_json::from_slice(&bytes).unwrap();
    let a = restored.clone();
    let b = restored.clone();
    drop(restored);
    drop(prepared);
    std::thread::scope(|scope| {
        for p in [&a, &b] {
            let bytes = &bytes;
            let contract = &contract;
            scope.spawn(move || {
                p.validate_static_contract().unwrap();
                p.validate_bindings(&contract, &values()).unwrap();
                assert_eq!(p.fingerprint(), format!("{:x}", Sha256::digest(&bytes)));
                assert_eq!(&serde_json::to_vec(p).unwrap(), bytes);
            });
        }
    });
    assert_eq!(a, b);
}

#[test]
fn warm_static_cache_preserves_live_wave_and_native_validation() {
    let contract = UpstreamMarkerV2Profile::CausalPrefill.arithmetic();
    let prepared = PreparedProjectionNumerics::prepare(&contract, &values()).unwrap();
    let base = facts(33);
    let native = native(true);
    let wave = PreparedUpstreamProjectionWave::prepare(&prepared, &base, Some(&native)).unwrap();
    assert!(prepared.data.static_contract_validation.get().is_some());
    for mutate in [
        (|f: &mut UpstreamProjectionWaveFacts| f.input_available_bytes = 1)
            as fn(&mut UpstreamProjectionWaveFacts),
        |f| f.output_available_bytes = 1,
        |f| f.weight_available_bytes = 1,
        |f| f.input_byte_offset = 1,
        |f| f.input_stride = 255,
        |f| f.component_id = id("other.component"),
        |f| f.local_rows = 0,
    ] {
        let mut bad = base.clone();
        mutate(&mut bad);
        assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &bad, Some(&native)).is_err());
    }
    assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &base, None).is_err());
    let mut invalid_native = native.clone();
    if let UpstreamNativeGeometry::Mmq { row_tile, .. } = &mut invalid_native.geometry {
        *row_tile = 16;
    }
    assert!(
        PreparedUpstreamProjectionWave::prepare(&prepared, &base, Some(&invalid_native)).is_err()
    );
    let mut changed_native = native;
    changed_native.implementation_fingerprint = "different.native".into();
    assert!(wave
        .validate_reconstructed(&contract, &values(), &base, Some(&changed_native))
        .is_err());
    let odd_weight = UpstreamProjectionWaveFacts {
        weight_byte_offset: 1,
        ..base
    };
    assert!(matches!(
        PreparedUpstreamProjectionWave::prepare(&prepared, &odd_weight, None)
            .unwrap()
            .route(),
        PreparedUpstreamProjectionRoute::StrictBase {
            reason: UpstreamStrictReason::WeightAlignmentNotSupported
        }
    ));
}

#[test]
fn shared_equality_preserves_independent_wire_tamper_and_owner_lifetime() {
    let prepared = PreparedProjectionNumerics::prepare(
        &UpstreamMarkerV2Profile::CausalPrefill.arithmetic(),
        &values(),
    )
    .unwrap();
    let shared = prepared.clone();
    assert!(Arc::ptr_eq(&prepared.data, &shared.data));
    assert_eq!(prepared, shared);
    let bytes = serde_json::to_vec(&prepared).unwrap();
    let independent: PreparedProjectionNumerics = serde_json::from_slice(&bytes).unwrap();
    assert!(!Arc::ptr_eq(&prepared.data, &independent.data));
    assert_eq!(prepared, independent);
    let mut altered: PreparedProjectionNumerics = serde_json::from_slice(&bytes).unwrap();
    Arc::get_mut(&mut altered.data).unwrap().projections[0].leaves[0].output_offset += 1;
    assert_ne!(prepared, altered);
    let weak = Arc::downgrade(&prepared.data);
    drop(prepared);
    assert!(weak.upgrade().is_some());
    assert_eq!(serde_json::to_vec(&shared).unwrap(), bytes);
    drop(shared);
    assert!(weak.upgrade().is_none());
    assert_eq!(serde_json::to_vec(&independent).unwrap(), bytes);
}

#[test]
#[ignore = "paired immutable-metadata CPU diagnostic, no speed threshold"]
fn paired_shared_prepared_equality() {
    use std::{hint::black_box, time::Instant};
    let prepared = PreparedProjectionNumerics::prepare(
        &UpstreamMarkerV2Profile::CausalPrefill.arithmetic(),
        &values(),
    )
    .unwrap();
    let shared = prepared.clone();
    for round in 0..6 {
        for fast in if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let start = Instant::now();
            for _ in 0..16384 {
                let left = black_box(&prepared);
                let right = black_box(&shared);
                let equal = if fast {
                    left == right
                } else {
                    left.data.contract == right.data.contract
                        && left.data.projections == right.data.projections
                };
                black_box(equal);
            }
            if round >= 2 {
                println!(
                    "{}",
                    serde_json::json!({"kind":"prepared_equality_pair","pair":round-2,"pointer_fastpath":fast,"iterations":16384,"elapsed_ns":start.elapsed().as_nanos()})
                );
            }
        }
    }
}
