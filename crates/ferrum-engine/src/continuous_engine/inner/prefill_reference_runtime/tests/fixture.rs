use super::*;
use ferrum_interfaces::execution_cost::*;
use ferrum_scheduler::implementations::continuous::{
    cost_profile::{v2::*, *},
    prefill_reference::*,
};
use std::num::NonZeroUsize;

pub(super) fn n64(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}
fn n32(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).unwrap()
}
pub(super) fn fingerprint() -> ExecutionFingerprint {
    ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
pub(super) fn identity() -> ExecutorCostIdentityAvailability {
    ExecutorCostIdentityAvailability::Known(Arc::new(ExecutorCostIdentity {
        schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }))
}
fn host(generated: u64) -> HostCostFeaturesV1 {
    HostCostFeaturesV1 {
        policy: HostCostPolicyV2 {
            empirical_content_domain: None,
            categorical_signature: [5; 32],
            decoder_text_bytes_per_token: 4,
            decoder_scratch_bytes_per_token: 8,
            raw_token_bytes_bound: 4,
        },
        state: HostCostStateV1 {
            generated_tokens_before: generated,
            maximum_output_tokens: 8,
            sampling_history_tokens: generated,
            sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
            pending_decoded_utf8: false,
            completion_state_signature: satisfied_completion_cost_signature(),
        },
    }
}
fn shape(work: ActualRowWork) -> ProfileWaveShapeV2 {
    let (kind, decode, prefill, host, output) = match work {
        ActualRowWork::Decode { kv_tokens } => (
            ProfileWaveKind::Decode,
            vec![kv_tokens],
            vec![],
            host(1),
            CostRowOutput::Decode {
                requires_full_logits: false,
                repetition_tokens: 0,
                repetition_penalty_bits: 1_f32.to_bits(),
            },
        ),
        ActualRowWork::Prefill {
            offset,
            count,
            total_prompt_tokens,
        } => (
            ProfileWaveKind::Prefill,
            vec![],
            vec![ProfilePrefillShape {
                offset,
                count: n32(count),
                total_prompt_tokens: n32(total_prompt_tokens),
            }],
            host(0),
            CostRowOutput::Prefill {
                final_logits: offset + count == total_prompt_tokens,
            },
        ),
        _ => unreachable!(),
    };
    ProfileWaveShapeV2 {
        exact: ProfileWaveShape {
            kind,
            path: ProfileExecutionPath::PlanRuntime,
            provider_signature: [6; 32],
            output_policy_signature: [7; 32],
            graph_state: ProfileGraphState::Disabled,
            order: ProfileBatchOrder::Ordered,
            decode_kv_tokens: decode,
            prefill_chunks: prefill,
            recurrent_state_bytes: 0,
            restore_bytes: 0,
            maintenance_bytes: 0,
            maintenance_units: 0,
        },
        numeric_features: Some(CanonicalWaveCostFeatures {
            schema_version: COST_NUMERIC_FEATURE_SCHEMA_V1,
            output_policy_signature: [8; 32],
            rows: vec![project_host_cost_features(host, work, output).unwrap()],
        }),
    }
}
fn sample(
    record: u64,
    shape: ProfileWaveShapeV2,
    wall: u64,
    commit: ReferenceCommitReceipt,
) -> ReferenceObservedSample {
    ReferenceObservedSample {
        record: ReferenceRecordId {
            source_sha256: [9; 32],
            ordinal: record,
        },
        observation: ProfileSampleV2 {
            source_record: record,
            measured_unix_ns: 100 + record,
            shape,
            boundary: ProfileCostBoundary::PreparationToCommit,
            outcome: ProfileObservationOutcome::Completed {},
            timing: ProfileWaveTiming {
                wall_total_ns: wall,
                device_elapsed_ns: None,
                stages: Default::default(),
            },
        },
        commit,
    }
}
pub(super) fn artifact() -> ReferenceCalibrationV1 {
    artifact_with_lengths(2, &[4])
}
fn artifact_with_lengths(granule: u32, lengths: &[u32]) -> ReferenceCalibrationV1 {
    let decode = shape(ActualRowWork::Decode { kv_tokens: 4 });
    let protocol = ReferenceProtocolV1 {
        granule_tokens: n32(granule),
        repetitions: NonZeroUsize::new(1).unwrap(),
        estimator: ReferenceEstimator::UpperMedianWallV1,
        input_preprocessing_sha256: [10; 32],
        measurement_conditions_sha256: [11; 32],
        prefill_host: host(0),
        decode_host: host(1),
        decode_shape: decode.clone(),
    };
    let mut builder = ReferenceCalibrationBuilder::new(
        n64(42),
        (&fingerprint()).into(),
        protocol,
        n64(1000),
        Default::default(),
    )
    .unwrap();
    builder
        .set_decode_samples(vec![sample(
            1,
            decode,
            7,
            ReferenceCommitReceipt {
                owner_incarnation: n64(1),
                work_generation: n64(2),
                origin: ReferenceStateOrigin::PreparedDecode,
                previous_record: None,
                prefix_before: 4,
                prefix_after: 5,
                generated_before: 1,
                generated_after: 2,
            },
        )])
        .unwrap();
    let mut record = 2;
    for (curve_index, &length) in lengths.iter().enumerate() {
        let partition = (0..length)
            .step_by(granule as usize)
            .map(|offset| {
                shape(ActualRowWork::Prefill {
                    offset,
                    count: granule.min(length - offset),
                    total_prompt_tokens: length,
                })
            })
            .collect::<Vec<_>>();
        let mut previous_record = None;
        let samples = partition
            .iter()
            .enumerate()
            .map(|(index, shape)| {
                let row = &shape.exact.prefill_chunks[0];
                let value = sample(
                    record,
                    shape.clone(),
                    5 + index as u64 * 4,
                    ReferenceCommitReceipt {
                        owner_incarnation: n64(2 + curve_index as u64),
                        work_generation: n64(index as u64 + 1),
                        origin: if index == 0 {
                            ReferenceStateOrigin::Fresh
                        } else {
                            ReferenceStateOrigin::CommittedContinuation
                        },
                        previous_record,
                        prefix_before: row.offset,
                        prefix_after: row.offset + row.count.get(),
                        generated_before: 0,
                        generated_after: u32::from(row.offset + row.count.get() == length),
                    },
                );
                previous_record = Some(value.record);
                record += 1;
                value
            })
            .collect();
        builder
            .add_curve(ReferenceCurveInput {
                total_prompt_tokens: n32(length),
                partition,
                trials: vec![ReferencePrefillTrial {
                    trial_index: 0,
                    input_tokens_sha256: [12; 32],
                    samples,
                }],
            })
            .unwrap();
    }
    builder.finish().unwrap()
}

pub(super) struct ArtifactFile(pub SloPrefillReferenceConfig);
impl ArtifactFile {
    pub fn new() -> Self {
        Self::from_artifact(artifact())
    }
    fn from_artifact(artifact: ReferenceCalibrationV1) -> Self {
        let path =
            std::env::temp_dir().join(format!("ferrum-reference-{}.json", uuid::Uuid::new_v4()));
        std::fs::write(&path, serde_json::to_vec(&artifact).unwrap()).unwrap();
        Self(SloPrefillReferenceConfig {
            artifact_path: path,
            expected_protocol_sha256: artifact.protocol.sha256().unwrap(),
            limits: Default::default(),
        })
    }
    pub fn runtime(&self) -> Arc<EnginePrefillReferenceRuntime> {
        EnginePrefillReferenceRuntime::load(Some(&self.0), identity)
            .unwrap()
            .unwrap()
    }
}
pub(super) fn shared_runtime() -> Arc<EnginePrefillReferenceRuntime> {
    ArtifactFile::from_artifact(artifact_with_lengths(1, &[1, 2, 4])).runtime()
}
pub(super) fn piecewise_runtime() -> Arc<EnginePrefillReferenceRuntime> {
    let artifact = artifact_with_lengths(1, &[1, 4]);
    let spec =
        ferrum_scheduler::implementations::continuous::prefill_reference::PiecewiseReferenceSpec {
            minimum_prompt_tokens: n32(1),
            maximum_prompt_tokens: n32(4),
            body_endpoints: vec![n32(1), n32(2), n32(3)],
        };
    let digest = spec.protocol_sha256(&artifact.protocol).unwrap();
    let mut builder = ReferenceCalibrationBuilder::new(
        artifact.reference_revision,
        artifact.fingerprint,
        artifact.protocol,
        artifact.generated_unix_ns,
        Default::default(),
    )
    .unwrap()
    .with_piecewise(spec)
    .unwrap();
    builder.set_decode_samples(artifact.decode_samples).unwrap();
    for curve in artifact.curves {
        builder.add_curve(curve).unwrap();
    }
    let path = std::env::temp_dir().join(format!("ferrum-piecewise-{}.json", uuid::Uuid::new_v4()));
    std::fs::write(&path, builder.finish_bytes().unwrap()).unwrap();
    let file = ArtifactFile(SloPrefillReferenceConfig {
        artifact_path: path,
        expected_protocol_sha256: digest,
        limits: Default::default(),
    });
    file.runtime()
}
impl Drop for ArtifactFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0.artifact_path);
    }
}
