use super::*;
use std::num::NonZeroU32;

fn mode() -> model::CostFeatureModel {
    model::CostFeatureModel::EmpiricalHostContentV1 {
        host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
    }
}
pub(super) fn stages(start: u64, wall: u64, terminal: bool) -> Arc<HostStageEvidenceV1> {
    let mut shape = numeric_observation(start + wall + 1).actual_shape;
    shape.host_content_features = Some(HostContentCostFeaturesV1 {
        schema_version: 1,
        output_policy_signature: [if terminal { 22 } else { 21 }; 32],
    });
    if terminal {
        shape.numeric_features.as_mut().unwrap().rows[0].maximum_output_tokens = 4;
    }
    Arc::new(HostStageEvidenceV1 {
        statistical_evidence: None,
        schema_version: 1,
        call_id: start + 1,
        fingerprint: Some(fingerprint()),
        actual_shape: Some(shape),
        prepare_started_at_ns: Some(start),
        executor_returned_at_ns: Some(start + 5),
        rows: vec![HostRowStageV1 {
            request_id: ferrum_types::RequestId::new(),
            owner_incarnation: 1,
            work_generation: 4,
            input_index: 0,
            actual_work: HostStageWork::Decode { kv_tokens: 128 },
            host_processing_ordinal: Some(0),
            host_started_at_ns: Some(start + 10),
            token_committed_at_ns: Some(start + 20),
            output_published_at_ns: Some(start + 30),
            completion_started_at_ns: terminal.then_some(start + 31),
            settled_at_ns: Some(start + wall),
            terminal: terminal.then_some(HostTerminalStageV1 {
                finish_reason: ferrum_types::FinishReason::Length,
                generated_tokens: 4,
                through_output_ordinal: 4,
                output_failed: false,
                physical_failed: false,
                scheduler_failed: false,
                terminal_handoff_succeeded: true,
                pending_restore_removed: false,
                admission_cancellation_work:
                    ferrum_interfaces::model_executor::ExecutorCompletionWork::NoAdditionalWork,
                cache_completion_work:
                    ferrum_interfaces::model_executor::ExecutorCompletionWork::NoAdditionalWork,
                other_physical_resources: false,
                request_slot_closed: true,
                owner_matched: true,
            }),
            completeness: HostStageCompleteness::CompleteSingleWave,
        }],
        finalized_at_ns: Some(start + wall + 1),
        full_wall_ns: Some(wall),
        completeness: HostStageCompleteness::CompleteSingleWave,
    })
}
pub(super) fn only(stages: Arc<HostStageEvidenceV1>) -> CostEvidenceEntry {
    CostEvidenceEntry::StagesOnly {
        stages,
        legacy_rejection: CostCallRejection::Composite,
    }
}
pub(super) fn evaluation() -> HostContentEvaluation {
    HostContentEvaluation {
        rejection: None,
        training: TrainingDisposition::Recorded,
        pre_update_prediction: Some(PreUpdatePrediction::NoPublishedModel),
    }
}
pub(super) fn actual(stages: &HostStageEvidenceV1) -> model::WaveCostObservation {
    model::WaveCostObservation {
        fingerprint: fingerprint(),
        actual_shape: stages.actual_shape.clone().unwrap(),
        boundary: model::CostBoundary::PreparationToHostSettledV1,
        outcome: model::WaveObservationOutcome::Completed,
        timing: model::WaveTiming {
            wall_total_ns: stages.full_wall_ns.unwrap(),
            device_elapsed_ns: None,
            stages: Default::default(),
        },
        observed_at_ns: stages.finalized_at_ns.unwrap(),
    }
}

#[tokio::test]
async fn auxiliary_training_is_opt_in_uses_pre_batch_snapshot_and_counts_terminal_errors() {
    let mut config = SloCostObservationConfig::default();
    config.model.feature_model = mode();
    config.model.min_samples = NonZeroUsize::new(2).unwrap();
    config.model.drift_margin_ns = 0;
    config.model.residual_quantile = 1.0;
    config.model.max_sample_age_ns = NonZeroU64::new(1000).unwrap();
    let runtime = runtime_with_fixed_clock(&config, false);
    for start in [100, 201] {
        runtime
            .sink
            .offer_evidence_numbered(only(stages(start, 50, true)))
            .unwrap();
    }
    runtime.consume_samples();
    let first = runtime.audit_snapshot();
    assert_eq!(first.training.host_content.offered_entries, 2);
    assert_eq!(first.training.host_content.outcomes.recorded, 2);
    assert_eq!(
        first
            .training
            .host_content
            .pre_update_prediction
            .counts
            .no_published_model,
        2
    );
    assert_eq!(first.training.outcomes.recorded, 0);
    assert_eq!(first.sink.drained, 0);
    assert_eq!(runtime.trained_samples(), 2);
    let old = runtime.snapshot().unwrap();
    let slow = stages(302, 200, true);
    runtime
        .sink
        .offer_evidence_numbered(only(Arc::clone(&slow)))
        .unwrap();
    runtime.consume_samples();
    let audit = runtime.audit_snapshot();
    assert_eq!(
        audit
            .training
            .host_content
            .pre_update_prediction
            .counts
            .known,
        1
    );
    assert_eq!(
        audit
            .training
            .host_content
            .pre_update_prediction
            .counts
            .underestimates,
        1
    );
    assert_eq!(
        audit
            .training
            .host_content
            .pre_update_prediction
            .counts
            .max_underestimate_ns,
        150
    );
    let shape = slow.actual_shape.as_ref().unwrap();
    let model::CostPrediction::Known(before) = old.predict(
        &fingerprint(),
        shape,
        model::CostBoundary::PreparationToHostSettledV1,
        503,
    ) else {
        panic!("old model must remain queryable")
    };
    assert_eq!(before.planning_ns, 50);
    assert!(matches!(
        old.predict(
            &fingerprint(),
            shape,
            model::CostBoundary::PreparationToHostSettledV1,
            1152
        ),
        model::CostPrediction::Unknown(model::CostUnknownReason::StaleSamples)
    ));
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn capacity_missing_domain_and_failed_stages_stay_in_host_offered_denominator() {
    let mut config = SloCostObservationConfig::default();
    config.model.feature_model = mode();
    config.model.min_samples = NonZeroUsize::MIN;
    config.model.max_buckets = NonZeroUsize::MIN;
    let runtime = runtime_with_fixed_clock(&config, false);
    runtime
        .sink
        .offer_evidence_numbered(only(stages(100, 50, false)))
        .unwrap();
    // Different terminal class really needs another bucket; capacity is not softened.
    runtime
        .sink
        .offer_evidence_numbered(only(stages(200, 50, true)))
        .unwrap();
    let mut absent = stages(300, 50, false);
    Arc::make_mut(&mut absent)
        .actual_shape
        .as_mut()
        .unwrap()
        .host_content_features = None;
    runtime.sink.offer_evidence_numbered(only(absent)).unwrap();
    let mut failed = stages(400, 50, false);
    Arc::make_mut(&mut failed).completeness = HostStageCompleteness::Failed;
    runtime.sink.offer_evidence_numbered(only(failed)).unwrap();
    runtime.sink.offer(observation(500)).unwrap();
    runtime.consume_samples();
    let audit = runtime.audit_snapshot();
    assert_eq!(audit.training.host_content.offered_entries, 5);
    assert_eq!(audit.training.host_content.eligible, 2);
    assert_eq!(audit.training.host_content.outcomes.recorded, 1);
    assert_eq!(
        audit.training.host_content.outcomes.errors
            [audit::TrainingErrorReason::CapacityExceeded as usize]
            .count,
        1
    );
    assert_eq!(
        audit
            .training
            .host_content
            .rejected
            .iter()
            .map(|entry| entry.count)
            .sum::<u64>(),
        3
    );
    assert_eq!(audit.training.outcomes.recorded, 0);
    assert_eq!(runtime.trained_samples(), 1);
    runtime.shutdown().await.unwrap();
}

#[test]
fn terminal_only_cut_and_final_profile_roundtrip_keep_source_join_and_original_age() {
    let mut f = Fixture::new();
    f.settings.feature_model = mode();
    f.settings.drift_margin_ns = 0;
    let mut export = f.exporter().unwrap();
    let stages = stages(100, 50, true);
    let sample = actual(&stages);
    export
        .record_entry_with_host(
            1,
            None,
            Some(stages),
            Some(CostCallRejection::Composite),
            Some((evaluation(), Some(&sample))),
        )
        .unwrap();
    let cut = export
        .write_cut(
            CostProfileCutPaths {
                profile: f.dir.join("host-cut.json"),
                source: f.dir.join("host-cut.jsonl"),
            },
            1,
            ExportClockReading {
                wall_unix_ns: 1300,
                monotonic_ns: 400,
            },
            stats(),
            &training_stats(),
        )
        .unwrap();
    assert_eq!(cut.retained_samples, 1);
    assert_eq!(cut.raw_retained_observations, 0);
    let lines: Vec<serde_json::Value> = fs::read_to_string(&cut.source)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines[0]["schema_version"], 3);
    assert_eq!(lines[1]["kind"], "host_stages_v1");
    assert!(lines[1]["source_record"].is_null());
    assert_eq!(lines[2]["host_source_record"], 0);
    assert_eq!(lines[2]["accepted_ordinal"], 1);
    let bytes = fs::read(&cut.profile).unwrap();
    assert!(serde_json::from_slice::<profile_v2::CostProfileFileV2>(&bytes).is_err());
    let file: profile_v3::CostProfileFileV3 = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(file.samples[0].measured_unix_ns, 1051);
    assert_eq!(file.samples[0].timing.wall_total_ns, 50);
    assert_eq!(file.samples[0].accepted_ordinal, 1);
    assert_eq!(file.source.observation_artifact_sha256, cut.source_digest);
    let loaded = profile::load_cost_profile_bytes(
        &bytes,
        &fingerprint(),
        &f.settings,
        &Default::default(),
        profile::ProfileLoadClock {
            wall_unix_ns: Some(1400),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 0,
        },
    )
    .unwrap();
    let model::CostPrediction::Known(prediction) = loaded.snapshot.predict(
        &fingerprint(),
        &sample.actual_shape,
        model::CostBoundary::PreparationToHostSettledV1,
        0,
    ) else {
        panic!("real importer must expose the host boundary")
    };
    assert_eq!(prediction.valid_for_ns, 651);
    assert_eq!(prediction.planning_ns, 50);
    assert!(matches!(
        loaded.snapshot.predict(
            &fingerprint(),
            &sample.actual_shape,
            model::CostBoundary::PreparationToHostSettledV1,
            652
        ),
        model::CostPrediction::Unknown(model::CostUnknownReason::StaleSamples)
    ));
    let final_receipt = f.finish(export).unwrap();
    assert_eq!(final_receipt.counts.retained_samples, 0);
    assert_eq!(final_receipt.counts.retained_host_content_samples, 1);
    assert_eq!(f.records()[0]["schema_version"], 5);
}

#[test]
fn auxiliary_sample_copy_is_charged_to_retention_and_not_partially_exported() {
    let mut f = Fixture::new();
    f.settings.feature_model = mode();
    // One stage row + its work/numeric vectors consumes three rows; the
    // additional retained profile work/numeric vectors require two more.
    f.options.max_total_shape_rows = NonZeroUsize::new(4).unwrap();
    let stages = stages(100, 50, true);
    let sample = actual(&stages);
    let mut export = f.exporter().unwrap();
    export
        .record_entry_with_host(
            1,
            None,
            Some(stages),
            Some(CostCallRejection::Composite),
            Some((evaluation(), Some(&sample))),
        )
        .unwrap();
    assert_eq!(export.counts.received_host_content_evaluations, 1);
    assert_eq!(export.counts.raw_retained_entries, 0);
    assert_eq!(export.counts.retained_host_content_samples, 0);
    assert_eq!(export.counts.host_stages_dropped_shape_row_limit, 1);
}
