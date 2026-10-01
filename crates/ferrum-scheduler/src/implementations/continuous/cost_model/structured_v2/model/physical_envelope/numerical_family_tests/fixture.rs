//! Typed canonical producers; synthetic walls below are not hardware evidence.
use super::*;
use std::num::{NonZeroU32, NonZeroU64};

pub(super) fn domain() -> CostWorkloadDomainV1 {
    CostWorkloadDomainV1::new_vnext(
        &ExecutorCostIdentity {
            schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
            model_weights: [1; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        },
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(8).unwrap(),
            maximum_context_tokens: NonZeroU32::new(1024).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(8).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(4096).unwrap(),
            repetition_slot_capacity: 512,
            fixed_state_bytes_per_row: 32,
        },
    )
    .unwrap()
}

pub(super) struct Wave {
    pub rows: u32,
    pub algorithm: &'static str,
    pub extra_algorithms: Vec<&'static str>,
    pub graph: ActualWaveGraphState,
    pub order: ActualWaveRowOrder,
    pub retries: u32,
    pub product: CostProductOutput,
    pub readback: CoreReadbackRoute,
    pub policy: HostCostPolicyV2,
    pub last_row_policy: Option<HostCostPolicyV2>,
    pub generated: u64,
    pub state_bytes_per_row: u64,
    pub mask: Vec<u32>,
    pub pending: Vec<u32>,
    pub length: Vec<u32>,
}

impl Default for Wave {
    fn default() -> Self {
        Self {
            rows: 2,
            algorithm: "fixture.family.decode",
            extra_algorithms: Vec::new(),
            graph: ActualWaveGraphState::Disabled,
            order: ActualWaveRowOrder::Ordered,
            retries: 0,
            product: CostProductOutput::GreedyToken,
            readback: CoreReadbackRoute::SubmissionStaged,
            policy: HostCostPolicyV2 {
                empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                categorical_signature: [4; 32],
                decoder_text_bytes_per_token: 4,
                decoder_scratch_bytes_per_token: 8,
                raw_token_bytes_bound: 4,
            },
            last_row_policy: None,
            generated: 3,
            state_bytes_per_row: 32,
            mask: Vec::new(),
            pending: Vec::new(),
            length: Vec::new(),
        }
    }
}

impl Wave {
    pub fn build(&self) -> CanonicalStatisticalWave {
        self.build_with_kv_tokens(64)
    }

    pub fn build_with_kv_tokens(&self, kv_tokens: u32) -> CanonicalStatisticalWave {
        let mut selected =
            SelectedCommandCostBuilderV1::new_with_algorithm_work(u64::from(self.rows));
        for algorithm in std::iter::once(&self.algorithm).chain(&self.extra_algorithms) {
            selected
                .kernel(
                    SelectedAlgorithmClassV1::new(algorithm, 1, [1; 32], [2; 32]).unwrap(),
                    KernelNumericWorkV1 {
                        logical_units: u64::from(self.rows) * 8,
                        padded_units: u64::from(self.rows) * 8,
                        inner_units_per_logical_unit: 2,
                        grid: [self.rows, 1, 1],
                        scratch_bytes: u64::from(self.rows) * 32,
                        staged_weight_bytes: 0,
                    },
                )
                .unwrap();
        }
        let selected = selected.finish().unwrap();
        let mut builder =
            CanonicalWaveCostBuilder::new_with_structured_statistics(self.retries, self.product);
        builder
            .physical_command(CostPhysicalCommand {
                native_op_id: "fixture.family",
                command_index: 0,
                node_index: Some(0),
                command_phase: DeviceCommandPhase::Compute,
                provider: Some(CostProviderIdentity {
                    provider_id: "numerical-fixture",
                    implementation_fingerprint: "v1",
                    operation_fingerprint: "v1",
                }),
                path: CostCommandPath::Eager,
                participant_start: 0,
                participant_count: self.rows,
                token_count: u64::from(self.rows),
                batching_form: "packed",
                compute_dispatch_count: 1 + self.extra_algorithms.len() as u64,
                transfer_command_count: 0,
                reusable_graph_node_count: None,
                statistical_evidence: Some(&selected),
            })
            .unwrap();
        builder.core_readback_route(self.readback).unwrap();
        for position in 0..self.rows {
            let policy = if position + 1 == self.rows {
                self.last_row_policy.unwrap_or(self.policy)
            } else {
                self.policy
            };
            builder
                .row(CanonicalCostRow {
                    work: ActualRowWork::Decode { kv_tokens },
                    host_policy_signature: [3; 32],
                    mask_upload_required: self.mask.contains(&position),
                    output: CostRowOutput::Decode {
                        requires_full_logits: self.product == CostProductOutput::FullLogits,
                        repetition_tokens: 0,
                        repetition_penalty_bits: 1f32.to_bits(),
                    },
                    host_features: Some(HostCostFeaturesV1 {
                        policy,
                        state: HostCostStateV1 {
                            generated_tokens_before: self.generated,
                            maximum_output_tokens: self.generated
                                + if self.length.contains(&position) {
                                    1
                                } else {
                                    18
                                },
                            sampling_history_tokens: self.generated,
                            sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                            pending_decoded_utf8: self.pending.contains(&position),
                            completion_state_signature: satisfied_completion_cost_signature(),
                        },
                    }),
                })
                .unwrap();
        }
        builder
            .finish_with_captured_structure(
                ActualWaveKind::Decode,
                ActualWavePath::PlanRuntime,
                self.graph,
                self.order,
                u64::from(self.rows) * self.state_bytes_per_row,
            )
            .unwrap()
    }
}

pub(super) fn checked(wave: &CanonicalStatisticalWave) -> Result<NumericalFamilyInputV1> {
    let selected = wave.statistical.as_ref().unwrap();
    NumericalFamilyInputV1::from_actual(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
        &domain(),
    )
}

pub(super) fn population(
    phase: StructuredPhaseV2,
    with_mask: bool,
) -> (NumericalFamilyV1, Vec<StructuredNumericObservationV2>) {
    let mut family = None;
    let samples = (0..24)
        .map(|i| {
            let rows = [2, 4, 8][i % 3];
            let wave = Wave {
                rows,
                mask: if with_mask && i % 6 == 0 {
                    vec![0]
                } else {
                    Vec::new()
                },
                ..Default::default()
            }
            .build();
            let input = checked(&wave).unwrap();
            if let Some(expected) = &family {
                assert_eq!(input.family(), expected);
            } else {
                family = Some(input.family().clone());
            }
            let ordinal = phase.index() as u64 * 24 + i as u64 + 1;
            StructuredObservationV2 {
                source: [10; 32],
                protocol: [11; 32],
                ordinal,
                membership: StructuredMemberBindingV2 {
                    rule_signature: [12; 32],
                    offered_ordinal: ordinal,
                    member_ordinal: i as u64 + 1,
                    phase,
                },
                call_id: ordinal,
                fingerprint: ExecutionFingerprint {
                    model_weights: [1; 32],
                    numerical_policy: [2; 32],
                    device_runtime: [3; 32],
                    execution_config: [4; 32],
                },
                input: input.original_input().clone(),
                boundary: CostBoundary::PreparationToHostSettledV1,
                outcome: WaveObservationOutcome::Completed,
                observed_at_ns: ordinal * 10,
                wall_ns: 1000 + u64::from(rows) * 10 + u64::from(with_mask && i % 6 == 0) * 7,
            }
        })
        .collect();
    (family.unwrap(), samples)
}

pub(super) fn fit(samples: &[StructuredNumericObservationV2]) -> NonNegativeFit {
    let axes: Vec<_> = samples
        .iter()
        .map(|s| envelope::axes(&s.input).unwrap())
        .collect();
    let rows: Vec<_> = samples
        .iter()
        .zip(&axes)
        .map(|(s, axes)| FitSample {
            axes,
            wall_ns: s.wall_ns,
        })
        .collect();
    NonNegativeFit::fit(
        &rows,
        &StructuredSettingsV2::default(),
        EnvelopeSettings::default(),
    )
    .unwrap()
}
