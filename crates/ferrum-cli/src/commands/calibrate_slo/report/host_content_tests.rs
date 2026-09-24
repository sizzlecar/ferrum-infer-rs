use super::*;
use ferrum_engine::continuous_engine::{HostRowStageV1, HostStageWork, HostTerminalStageV1};
use ferrum_interfaces::{
    execution_cost::{
        CanonicalWaveCostFeatures, CostRowNumericFeatures, HostContentCostFeaturesV1,
        HostRowMultisetCostFeaturesV2, HostRowRoleV2, HostRowStaticCostFeaturesV2,
    },
    model_executor::ExecutorCompletionWork,
};
use ferrum_scheduler::implementations::continuous::cost_model::*;
use ferrum_types::{FinishReason, RequestId};
use std::num::{NonZeroU32, NonZeroUsize};

fn terminal() -> HostStageEvidenceV1 {
    let shape = WaveExecutionShape {
        kind: WaveKind::Decode,
        path: WaveExecutionPath::PlanRuntime,
        provider_signature: [3; 32],
        output_policy_signature: [4; 32],
        graph_state: WaveGraphState::Warm,
        order: BatchOrderSemantics::Ordered,
        decode_kv_tokens: vec![20],
        prefill_chunks: vec![],
        recurrent_state_bytes: 0,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
        host_content_features: Some(HostContentCostFeaturesV1 {
            schema_version: 1,
            output_policy_signature: [5; 32],
        }),
        row_multiset_features: None,
        numeric_features: Some(CanonicalWaveCostFeatures {
            schema_version: 1,
            output_policy_signature: [4; 32],
            rows: vec![CostRowNumericFeatures {
                generated_tokens_before: 3,
                maximum_output_tokens: 4,
                sampling_history_tokens: 3,
                repetition_tokens: 0,
                decoded_prefix_tokens: 4,
                decoded_text_bytes_bound: 40,
                decode_scratch_bytes_bound: 16,
            }],
        }),
    };
    HostStageEvidenceV1 {
        presubmit_prediction: None,
        statistical_evidence: None,
        schema_version: 1,
        call_id: 1,
        fingerprint: Some(ExecutionFingerprint {
            model_weights: [1; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        }),
        actual_shape: Some(shape),
        prepare_started_at_ns: Some(100),
        executor_returned_at_ns: Some(110),
        rows: vec![HostRowStageV1 {
            request_id: RequestId::new(),
            owner_incarnation: 1,
            work_generation: 4,
            input_index: 0,
            actual_work: HostStageWork::Decode { kv_tokens: 20 },
            host_processing_ordinal: Some(0),
            host_started_at_ns: Some(111),
            token_committed_at_ns: Some(120),
            output_published_at_ns: Some(125),
            completion_started_at_ns: Some(126),
            settled_at_ns: Some(200),
            terminal: Some(HostTerminalStageV1 {
                finish_reason: FinishReason::Length,
                generated_tokens: 4,
                through_output_ordinal: 4,
                output_failed: false,
                physical_failed: false,
                scheduler_failed: false,
                terminal_handoff_succeeded: true,
                pending_restore_removed: false,
                admission_cancellation_work: ExecutorCompletionWork::NoAdditionalWork,
                cache_completion_work: ExecutorCompletionWork::NoAdditionalWork,
                other_physical_resources: false,
                request_slot_closed: true,
                owner_matched: true,
            }),
            completeness: HostStageCompleteness::CompleteSingleWave,
        }],
        finalized_at_ns: Some(201),
        full_wall_ns: Some(100),
        completeness: HostStageCompleteness::CompleteSingleWave,
    }
}

#[test]
fn v2_terminal_predictions_require_real_v2_rows_without_filling_legacy_evidence() {
    let mut stages = terminal();
    let shape = stages.actual_shape.as_mut().unwrap();
    shape.host_content_features = None;
    shape.row_multiset_features = Some(HostRowMultisetCostFeaturesV2 {
        schema_version: 2,
        wave_policy_signature: [8; 32],
        rows: vec![HostRowStaticCostFeaturesV2 {
            role: HostRowRoleV2::Decode,
            categorical_signature: [9; 32],
        }],
    });
    let fingerprint = stages.fingerprint.clone().unwrap();
    let shape = stages.actual_shape.clone().unwrap();
    let mut trainer = CostModelTrainer::new(
        fingerprint.clone(),
        CostModelSettings {
            feature_model: CostFeatureModel::EmpiricalRowMultisetV2 {
                host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
            },
            min_samples: NonZeroUsize::MIN,
            drift_margin_ns: 0,
            ..Default::default()
        },
    )
    .unwrap();
    trainer
        .observe(WaveCostObservation {
            fingerprint: fingerprint.clone(),
            actual_shape: shape,
            boundary: CostBoundary::PreparationToHostSettledV1,
            outcome: WaveObservationOutcome::Completed,
            timing: WaveTiming {
                wall_total_ns: 100,
                device_elapsed_ns: None,
                stages: Default::default(),
            },
            observed_at_ns: 201,
        })
        .unwrap();
    let model = trainer.publish(201).unwrap();
    let mut totals = Summary::default();
    let result = host_content_prediction(Some(&stages), "test_snapshot", &mut totals, |query| {
        assert!(
            query.host_content_features.is_none(),
            "CLI invented old host identity"
        );
        Ok(Some(model.predict(
            &fingerprint,
            query,
            model.planning_boundary(),
            201,
        )))
    })
    .unwrap();
    assert_eq!(result["kind"], "known");
    assert_eq!(result["observed_row_multiset_schema"], 2);
    // An old complete terminal has legitimate V1 evidence, but the V2 model
    // must return Unknown rather than deriving absent row classes from its SHA.
    let old = terminal();
    let result = host_content_prediction(Some(&old), "test_snapshot", &mut totals, |query| {
        Ok(Some(model.predict(
            &fingerprint,
            query,
            model.planning_boundary(),
            201,
        )))
    })
    .unwrap();
    assert_eq!(result["kind"], "unknown");
    assert!(result["observed_row_multiset_schema"].is_null());
    assert_eq!(
        (
            totals.host_content_validation_offered,
            totals.host_content_validation_known,
            totals.host_content_validation_unknown
        ),
        (2, 1, 1)
    );
}

#[test]
fn terminal_validation_uses_real_model_boundary_and_separate_complete_wall_denominator() {
    let mut stages = terminal();
    let fingerprint = stages.fingerprint.clone().unwrap();
    let shape = stages.actual_shape.clone().unwrap();
    let mut trainer = CostModelTrainer::new(
        fingerprint.clone(),
        CostModelSettings {
            feature_model: CostFeatureModel::EmpiricalHostContentV1 {
                host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
            },
            min_samples: NonZeroUsize::MIN,
            drift_margin_ns: 0,
            ..Default::default()
        },
    )
    .unwrap();
    trainer
        .observe(WaveCostObservation {
            fingerprint: fingerprint.clone(),
            actual_shape: shape.clone(),
            boundary: CostBoundary::PreparationToHostSettledV1,
            outcome: WaveObservationOutcome::Completed,
            timing: WaveTiming {
                wall_total_ns: 100,
                device_elapsed_ns: None,
                stages: Default::default(),
            },
            observed_at_ns: 201,
        })
        .unwrap();
    let model = trainer.publish(201).unwrap();
    // Held-out terminal includes the completion tail; token commit remains 20ns.
    stages.rows[0].settled_at_ns = Some(350);
    stages.finalized_at_ns = Some(351);
    stages.full_wall_ns = Some(250);
    let mut totals = Summary::default();
    let result = host_content_prediction(Some(&stages), "live_frozen", &mut totals, |query| {
        assert_eq!(query, &shape);
        Ok(Some(model.predict(
            &fingerprint,
            query,
            model.planning_boundary(),
            351,
        )))
    })
    .unwrap();
    assert_eq!(result["kind"], "known");
    assert_eq!(result["actual_wall_ns"], 250);
    assert_eq!(result["underestimate_ns"], 150);
    assert_eq!(totals.host_content_validation_offered, 1);
    assert_eq!(totals.host_content_validation_underestimates, 1);
    assert_eq!(totals.validation_known, 0);
    assert_eq!(totals.validation_unknown, 0);

    stages.completeness = HostStageCompleteness::Failed;
    host_content_prediction(Some(&stages), "live_frozen", &mut totals, |_| {
        panic!("failed terminal cannot query a successful cost")
    })
    .unwrap();
    host_content_prediction(None, "live_frozen", &mut totals, |_| {
        panic!("absent auxiliary evidence cannot query a successful cost")
    })
    .unwrap();
    assert_eq!(totals.host_content_validation_offered, 3);
    assert_eq!(totals.host_content_validation_known, 1);
    assert_eq!(totals.host_content_validation_unknown, 2);
}
