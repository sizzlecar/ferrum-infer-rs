use super::*;

#[test]
fn imported_payload_counts_owned_metadata_and_capacity_not_source_encoding_size() {
    let (_, _, _, collector) = collected_source();
    let mut catalog = collector
        .activate_same_process_memory(
            StructuredServiceClockV6 {
                monotonic_ns: 50_000,
                wall_unix_ns: 1,
            },
            &CostProfileLoadLimits::default(),
        )
        .unwrap();
    let mut child = catalog.children.pop().unwrap();
    assert_eq!(
        serde_json::to_value(&child.provenance.producer).unwrap(),
        header().producer
    );
    let parameters = child.parameters_signature();
    let original = child.retained_payload_bytes().unwrap();
    // These source receipts describe a stream that is not retained in memory.
    child.provenance.source_bytes = u64::MAX;
    child.provenance.file_bytes = u64::MAX;
    assert_eq!(child.retained_payload_bytes(), Some(original));

    let old_capacity = child.scope.coverage.pending_counts.capacity();
    child.scope.coverage.pending_counts.reserve_exact(1024);
    let extra_capacity = (child.scope.coverage.pending_counts.capacity() - old_capacity)
        * std::mem::size_of::<u32>();
    let mut path = PathBuf::from("preserved-origin");
    path.reserve(1024);
    let extra_path = path.capacity();
    child.provenance.source_path = Some(path);
    assert_eq!(
        child.retained_payload_bytes(),
        Some(original + extra_capacity + extra_path)
    );

    let old_json = child.provenance.producer.get().len();
    let mut producer: serde_json::Value =
        serde_json::from_str(child.provenance.producer.get()).unwrap();
    producer["additional_audit_context"] =
        serde_json::json!({"entries": ["x".repeat(2048), "y".repeat(1024)]});
    child.provenance.producer = serde_json::value::to_raw_value(&producer).unwrap();
    let extra_json = child.provenance.producer.get().len() - old_json;
    assert_eq!(
        serde_json::to_value(&child.provenance.producer).unwrap(),
        producer
    );
    assert_eq!(
        child.retained_payload_bytes(),
        Some(original + extra_capacity + extra_path + extra_json)
    );
    assert_eq!(child.parameters_signature(), parameters);
}

#[test]
fn memory_activation_preserves_source_parameters_and_original_sample_age() {
    let (bytes, input, _, collector) = collected_source();
    let expected_parameters = collector.models().next().unwrap().1.parameters_signature();
    let now = StructuredServiceClockV6 {
        monotonic_ns: 50_000,
        wall_unix_ns: 1,
    };
    let catalog = collector
        .activate_same_process_memory(now, &CostProfileLoadLimits::default())
        .unwrap();
    assert_eq!(catalog.source_bytes, bytes.len() as u64);
    assert_eq!(
        catalog.source_sha256,
        <[u8; 32]>::from(Sha256::digest(bytes))
    );
    let child = &catalog.children[0];
    assert_eq!(child.parameters_signature(), expected_parameters);
    assert_eq!(child.source_path(), None);
    let p = child.provenance();
    assert_eq!(p.storage, ferrum_types::SloCostProfileStorage::Memory);
    assert_eq!(
        p.clock_basis,
        ferrum_types::SloCostProfileClockBasis::SameProcessMonotonic
    );
    assert_eq!(p.loaded_from, None);
    assert_eq!(p.oldest_imported_age_ns, 46_900);
    assert_eq!(p.newest_imported_age_ns, 900);
    assert_eq!(p.clock.source_monotonic_anchor_ns, 50_000);
    assert_eq!(p.clock.model_anchor_ns, 50_000);
    let wire = serde_json::to_value(p).unwrap();
    assert_eq!(wire["storage"], "memory");
    assert!(wire.get("loaded_from").is_none());
    assert!(wire.get("source_path").is_none());
    let query = StructuredQueryV2::exact(input);
    let prediction = child
        .predict_query_local(&old::fingerprint(), &query, 50_000)
        .unwrap();
    assert_eq!(prediction.valid_until_ns, 1_000_003_100);
    assert!(child
        .predict_query_local(&old::fingerprint(), &query, prediction.valid_until_ns + 1)
        .is_err());
}

#[test]
fn memory_activation_requires_sealed_population_freshness_and_bounded_metadata() {
    for poison in [false, true] {
        let mut collector =
            StructuredServiceCollectorV6::new(header(), CostProfileLoadLimits::default()).unwrap();
        if poison {
            assert!(collector
                .push(&StructuredServiceRecordV6::Completed {
                    wave: wave(1, StructuredPhaseV2::Fit),
                })
                .is_err());
        }
        assert!(collector
            .activate_same_process_memory(
                StructuredServiceClockV6 {
                    monotonic_ns: 50_000,
                    wall_unix_ns: 1
                },
                &CostProfileLoadLimits::default(),
            )
            .is_err());
    }
    for now in [49_110, 1_000_003_101] {
        let (_, _, _, collector) = collected_source();
        assert!(matches!(
            collector.activate_same_process_memory(
                StructuredServiceClockV6 {
                    monotonic_ns: now,
                    wall_unix_ns: u64::MAX
                },
                &CostProfileLoadLimits::default(),
            ),
            Err(CostProfileError::Clock(_))
        ));
    }
    let (bytes, _, _, collector) = collected_source();
    let mut limits = CostProfileLoadLimits::default();
    limits.max_file_bytes = std::num::NonZeroUsize::new(bytes.len()).unwrap();
    assert!(matches!(
        collector.activate_same_process_memory(
            StructuredServiceClockV6 {
                monotonic_ns: 50_000,
                wall_unix_ns: 1
            },
            &limits,
        ),
        Err(CostProfileError::Limit(_))
    ));
}

#[test]
fn live_catalog_keeps_original_clock_and_matches_independent_file_replay() {
    let (bytes, input, close_wall, collector) = collected_source();
    let f = Files::new(&bytes);
    let limits = CostProfileLoadLimits::default();
    let now = StructuredServiceClockV6 {
        monotonic_ns: 50_000,
        // The live path must not infer sample age from a wall-clock offset.
        wall_unix_ns: 1,
    };
    let live = collector
        .publish_same_process(
            &f.source,
            bytes.len() as u64,
            Sha256::digest(&bytes).into(),
            &f.profile,
            now,
            &limits,
        )
        .unwrap();
    let child = &live.children[0];
    let p = child.provenance();
    assert_eq!(
        p.clock_basis,
        ferrum_types::SloCostProfileClockBasis::SameProcessMonotonic
    );
    assert_eq!(p.conservative_clock_error_ns, 0);
    assert_eq!(p.clock.source_monotonic_anchor_ns, now.monotonic_ns);
    assert_eq!(p.clock.model_anchor_ns, now.monotonic_ns);
    assert_eq!(p.oldest_imported_age_ns, now.monotonic_ns - 3_100);
    assert_eq!(p.newest_imported_age_ns, now.monotonic_ns - 49_100);
    assert_eq!(p.loaded_unix_ns, 1);
    assert_eq!(child.model_now_ns(50_777).unwrap(), 50_777);
    let query = StructuredQueryV2::exact(input);
    let prediction = child
        .predict_query_local(&old::fingerprint(), &query, now.monotonic_ns)
        .unwrap();
    assert_eq!(prediction.valid_until_ns, 1_000_003_100);
    assert!(child
        .predict_query_local(&old::fingerprint(), &query, prediction.valid_until_ns + 1)
        .is_err());

    // Independent replay stays available for explicitly declared imports.
    let external = f.dir.join("external.profile.json");
    export_structured_profile_v13(
        &f.source,
        Sha256::digest(&bytes).into(),
        &external,
        0,
        &limits,
    )
    .unwrap();
    let replayed = load_structured_profile_v13(
        &external,
        &old::fingerprint(),
        &limits,
        ProfileLoadClock {
            monotonic_now_ns: 100,
            wall_unix_ns: Some(close_wall + 889),
            wall_max_error_ns: Some(0),
        },
    )
    .unwrap();
    assert_eq!(
        child.parameters_signature(),
        replayed.children[0].parameters_signature()
    );
    assert_eq!(p.phases, replayed.children[0].provenance().phases);
    let replay_prediction = replayed.children[0]
        .predict_query_local(&old::fingerprint(), &query, 100)
        .unwrap();
    assert_eq!(format!("{prediction:?}"), format!("{replay_prediction:?}"));

    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&f.profile).unwrap()).unwrap();
    assert!(metadata.get("source_clock_max_error_ns").is_none());
    assert!(matches!(
        load_structured_profile_v13(
            &f.profile,
            &old::fingerprint(),
            &limits,
            ProfileLoadClock {
                monotonic_now_ns: 100,
                wall_unix_ns: Some(close_wall),
                wall_max_error_ns: Some(0),
            }
        ),
        Err(CostProfileError::Clock(_))
    ));
}

#[test]
fn live_catalog_rejects_changed_writer_receipt_without_publishing() {
    for change_length in [false, true] {
        let (bytes, _, _, collector) = collected_source();
        let f = Files::new(&bytes);
        let digest = if change_length {
            Sha256::digest(&bytes).into()
        } else {
            [19; 32]
        };
        assert!(collector
            .publish_same_process(
                &f.source,
                bytes.len() as u64 + u64::from(change_length),
                digest,
                &f.profile,
                StructuredServiceClockV6 {
                    wall_unix_ns: 0,
                    monotonic_ns: 50_000
                },
                &CostProfileLoadLimits::default(),
            )
            .is_err());
        assert!(!f.profile.exists());
    }
}

#[test]
fn live_catalog_rejects_expired_or_backwards_clock_without_renewing_ttl() {
    for now in [49_110, 1_000_003_101] {
        let (bytes, _, _, collector) = collected_source();
        let f = Files::new(&bytes);
        assert!(matches!(
            collector.publish_same_process(
                &f.source,
                bytes.len() as u64,
                Sha256::digest(&bytes).into(),
                &f.profile,
                StructuredServiceClockV6 {
                    wall_unix_ns: u64::MAX,
                    monotonic_ns: now
                },
                &CostProfileLoadLimits::default(),
            ),
            Err(CostProfileError::Clock(_))
        ));
        assert!(!f.profile.exists());
    }
}

#[test]
fn live_catalog_requires_complete_unpoisoned_population() {
    for poison in [false, true] {
        let h = header();
        let bytes = replay::record_bytes(&h).unwrap();
        let f = Files::new(&bytes);
        let mut collector =
            StructuredServiceCollectorV6::new(h, CostProfileLoadLimits::default()).unwrap();
        if poison {
            assert!(collector
                .push(&StructuredServiceRecordV6::Completed {
                    wave: wave(1, StructuredPhaseV2::Fit),
                })
                .is_err());
        }
        assert!(collector
            .publish_same_process(
                &f.source,
                bytes.len() as u64,
                Sha256::digest(&bytes).into(),
                &f.profile,
                StructuredServiceClockV6 {
                    wall_unix_ns: 0,
                    monotonic_ns: 50_000
                },
                &CostProfileLoadLimits::default(),
            )
            .is_err());
        assert!(!f.profile.exists());
    }
}
