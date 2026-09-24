use super::*;

fn stages(capacity: usize) -> Arc<HostStageEvidenceV1> {
    Arc::new(HostStageEvidenceV1 {
        presubmit_prediction: None,
        statistical_evidence: None,
        schema_version: 1,
        call_id: 7,
        fingerprint: Some(fingerprint()),
        actual_shape: Some(observation(120).actual_shape),
        prepare_started_at_ns: Some(100),
        executor_returned_at_ns: Some(110),
        rows: Vec::with_capacity(capacity),
        finalized_at_ns: Some(150),
        full_wall_ns: Some(50),
        // Missing row evidence must remain diagnostics, never a cost sample.
        completeness: HostStageCompleteness::MissingEvidence,
    })
}

fn only(stages: Arc<HostStageEvidenceV1>) -> CostEvidenceEntry {
    CostEvidenceEntry::StagesOnly {
        stages,
        legacy_rejection: CostCallRejection::Composite,
    }
}

#[test]
fn auxiliary_fifo_charges_spare_capacity_and_keeps_sample_denominator_separate() {
    let sink = BoundedCostSampleSink::new(CostSampleSinkLimits {
        max_samples: 1,
        max_shape_rows: 4,
    })
    .unwrap();
    // The row Vec is empty but owns five slots, plus one actual-shape slot.
    assert_eq!(
        sink.offer_evidence_numbered(only(stages(5))),
        Err(CostSampleDrop::Capacity)
    );
    assert_eq!(sink.offer_evidence_numbered(only(stages(0))), Ok(1));
    assert_eq!(sink.offer(observation(120)), Err(CostSampleDrop::Capacity));
    assert_eq!(
        sink.with_locked_queue(|| sink.offer_evidence_numbered(only(stages(0)))),
        Err(CostSampleDrop::Contended)
    );
    let (ordinal, entry) = sink.pop_numbered().unwrap();
    assert_eq!(ordinal, 1);
    assert!(matches!(entry, CostEvidenceEntry::StagesOnly { .. }));
    // Released auxiliary storage and entry slot can be reused by a real sample.
    assert_eq!(sink.offer_numbered(observation(120)), Ok(2));
    assert!(matches!(
        sink.pop_numbered().unwrap().1,
        CostEvidenceEntry::Training { .. }
    ));
    let stats = sink.stats();
    assert_eq!((stats.offered, stats.published, stats.drained), (2, 1, 1));
    assert_eq!(
        (
            stats.entries_offered,
            stats.entries_published,
            stats.entries_drained
        ),
        (5, 2, 2)
    );
    assert_eq!(
        (
            stats.host_stages_offered,
            stats.host_stages_published,
            stats.host_stages_drained
        ),
        (3, 1, 1)
    );
    assert_eq!(
        (
            stats.entries_dropped_capacity,
            stats.entries_dropped_contention
        ),
        (2, 1)
    );
    assert!(stats.has_lost_samples());
}

#[tokio::test]
async fn stages_only_advances_checkpoint_without_training_or_refreshing_age() {
    let mut config = SloCostObservationConfig::default();
    config.model.min_samples = NonZeroUsize::MIN;
    config.model.max_sample_age_ns = NonZeroU64::new(1000).unwrap();
    let runtime = runtime_with_fixed_clock(&config, false);
    runtime.sink.offer(observation(100)).unwrap();
    runtime.consume_samples();
    let old = runtime.snapshot().unwrap();
    runtime
        .sink
        .offer_evidence_numbered(only(stages(0)))
        .unwrap();
    let waiter = runtime.request_checkpoint().unwrap();
    runtime.consume_samples();
    let cut = waiter.wait().await.unwrap();
    assert_eq!(cut.accepted_ordinal, 2);
    assert_eq!(runtime.trained_samples(), 1);
    assert!(Arc::ptr_eq(&old, &runtime.snapshot().unwrap()));
    assert!(matches!(
        old.predict(
            &fingerprint(),
            &observation(0).actual_shape,
            model::CostBoundary::PreparationToCommit,
            1101
        ),
        model::CostPrediction::Unknown(model::CostUnknownReason::StaleSamples)
    ));
    assert_eq!(runtime.sink.stats().drained, 1);
    assert_eq!(runtime.sink.stats().entries_drained, 2);
    runtime.shutdown().await.unwrap();
}

#[test]
fn raw_and_cut_join_auxiliary_entries_without_reassigning_cost_source_records() {
    let f = Fixture::new();
    let mut export = f.exporter().unwrap();
    export
        .record_entry(
            1,
            Some((
                &observation(120),
                TrainingDisposition::Recorded,
                PreUpdatePrediction::NoPublishedModel,
            )),
            Some(stages(0)),
            None,
        )
        .unwrap();
    export
        .record_entry(2, None, Some(stages(0)), Some(CostCallRejection::Composite))
        .unwrap();
    export
        .record_numbered(
            3,
            &observation(130),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    let receipt = export
        .write_cut(
            CostProfileCutPaths {
                profile: f.dir.join("cut.json"),
                source: f.dir.join("cut.jsonl"),
            },
            3,
            ExportClockReading {
                wall_unix_ns: 1300,
                monotonic_ns: 400,
            },
            stats(),
            &training_stats(),
        )
        .unwrap();
    let cut: Vec<serde_json::Value> = fs::read_to_string(&receipt.source)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(cut[0]["schema_version"], 2);
    assert_eq!(cut[1]["accepted_ordinal"], 1);
    assert_eq!(cut[2]["kind"], "host_stages_v1");
    assert_eq!(cut[2]["accepted_ordinal"], 1);
    assert_eq!(cut[2]["source_record"], 0);
    assert_eq!(cut[3]["accepted_ordinal"], 2);
    assert!(cut[3]["source_record"].is_null());
    assert_eq!(cut[3]["legacy_rejection"], "composite");
    assert_eq!(cut[4]["sample"]["source_record"], 1);
    assert_eq!(cut[4]["accepted_ordinal"], 3);
    assert_eq!(receipt.raw_retained_observations, 2);
    let profile: profile_v2::CostProfileFileV2 =
        serde_json::from_slice(&fs::read(receipt.profile).unwrap()).unwrap();
    assert_eq!(profile.samples.len(), 2);
    assert_eq!(
        profile
            .samples
            .iter()
            .map(|row| (row.source_record, row.measured_unix_ns))
            .collect::<Vec<_>>(),
        vec![(0, 1020), (1, 1030)]
    );
    let receipt = f.finish(export).unwrap();
    assert_eq!(receipt.counts.received_entries, 3);
    assert_eq!(receipt.counts.raw_retained_entries, 3);
    assert_eq!(receipt.counts.received_host_stages, 2);
    assert_eq!(receipt.counts.raw_retained_host_stages, 2);
    assert_eq!(receipt.counts.received_observations, 2);
    assert_eq!(f.records()[0]["schema_version"], 4);
}

#[test]
fn combined_entry_export_limits_drop_both_records_without_partial_join() {
    for limit in 0..3 {
        let mut f = Fixture::new();
        if limit == 0 {
            f.options.max_samples = NonZeroUsize::MIN;
        }
        if limit == 1 {
            f.options.max_total_shape_rows = NonZeroUsize::MIN;
        }
        let mut export = f.exporter().unwrap();
        if limit == 0 {
            export
                .record_numbered(
                    1,
                    &observation(120),
                    TrainingDisposition::Recorded,
                    PreUpdatePrediction::NoPublishedModel,
                )
                .unwrap();
        }
        if limit == 2 {
            export.max_body_bytes = 0;
        }
        let ordinal = if limit == 0 { 2 } else { 1 };
        export
            .record_entry(
                ordinal,
                Some((
                    &observation(130),
                    TrainingDisposition::Recorded,
                    PreUpdatePrediction::NoPublishedModel,
                )),
                Some(stages(0)),
                None,
            )
            .unwrap();
        assert_eq!(export.counts.raw_retained_host_stages, 0);
        assert_eq!(
            export.counts.raw_retained_observations,
            u64::from(limit == 0)
        );
        let counters = &export.counts;
        let drops = match limit {
            0 => (
                counters.dropped_sample_limit,
                counters.host_stages_dropped_sample_limit,
            ),
            1 => (
                counters.dropped_shape_row_limit,
                counters.host_stages_dropped_shape_row_limit,
            ),
            _ => (
                counters.dropped_file_byte_limit,
                counters.host_stages_dropped_file_byte_limit,
            ),
        };
        assert_eq!(drops, (1, 1));
    }
}
