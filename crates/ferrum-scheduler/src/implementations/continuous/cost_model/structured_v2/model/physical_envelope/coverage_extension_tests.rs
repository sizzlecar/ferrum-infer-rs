//! Structural extension uses real canonical inputs and the actual query gate.
//! Synthetic timings test numerical contracts, not hardware performance.
use super::*;
use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};
use ferrum_types::FinishReason;
#[path = "numerical_family_tests/fixture.rs"]
mod fixture;

#[derive(Clone, Copy)]
struct Corner {
    pending: bool,
    length: bool,
    repetition: u64,
    // 0=continuation, 1=EOS, 2=Stop. Length uses its mandatory actual cause.
    cause: u8,
    mask: bool,
}

fn corners() -> Vec<Corner> {
    let mut values = Vec::new();
    for pending in [false, true] {
        for length in [false, true] {
            for repetition in [0, 1] {
                for cause in 0..if length { 1 } else { 3 } {
                    values.push(Corner {
                        pending,
                        length,
                        repetition,
                        cause,
                        mask: true,
                    });
                }
            }
        }
    }
    values
}

fn prepared(c: Corner, rows: u32) -> StructuredInputV2 {
    let mut command = SelectedCommandCostBuilderV1::new_with_algorithm_work(u64::from(rows));
    command
        .kernel(
            SelectedAlgorithmClassV1::new("coverage.extension", 1, [1; 32], [2; 32]).unwrap(),
            KernelNumericWorkV1 {
                logical_units: u64::from(rows),
                padded_units: u64::from(rows),
                inner_units_per_logical_unit: 2,
                grid: [1, 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    let selected = command.finish().unwrap();
    let mut builder =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    builder
        .physical_command(CostPhysicalCommand {
            native_op_id: "coverage.extension",
            command_index: 0,
            node_index: Some(0),
            command_phase: DeviceCommandPhase::Compute,
            provider: Some(CostProviderIdentity {
                provider_id: "fixture",
                implementation_fingerprint: "v1",
                operation_fingerprint: "v1",
            }),
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: rows,
            token_count: u64::from(rows),
            batching_form: "packed",
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            reusable_graph_node_count: None,
            statistical_evidence: Some(&selected),
        })
        .unwrap();
    builder
        .core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    for _ in 0..rows {
        builder
            .row(CanonicalCostRow {
                work: ActualRowWork::Decode { kv_tokens: 64 },
                output: CostRowOutput::Decode {
                    requires_full_logits: true,
                    repetition_tokens: c.repetition,
                    repetition_penalty_bits: 1.1f32.to_bits(),
                },
                host_policy_signature: [3; 32],
                mask_upload_required: c.mask,
                host_features: Some(HostCostFeaturesV1 {
                    policy: HostCostPolicyV2 {
                        empirical_content_domain: Some(HostContentDomainV1::PlainTextInstalledV2(
                            PlainTextPolicyCapabilityV2 {
                                sampling: PlainTextSamplingRouteV2::FullLogits,
                                model_eos: true,
                                user_stop: true,
                            },
                        )),
                        categorical_signature: [4; 32],
                        decoder_text_bytes_per_token: 8,
                        decoder_scratch_bytes_per_token: 16,
                        raw_token_bytes_bound: 8,
                    },
                    state: HostCostStateV1 {
                        generated_tokens_before: 2,
                        maximum_output_tokens: if c.length { 3 } else { 20 },
                        sampling_history_tokens: 2,
                        sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                        pending_decoded_utf8: c.pending,
                        completion_state_signature: satisfied_completion_cost_signature(),
                    },
                }),
            })
            .unwrap();
    }
    let wave = builder
        .finish_with_captured_structure(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            u64::from(rows) * 32,
        )
        .unwrap();
    let selected = wave.statistical.as_ref().unwrap();
    StructuredInputV2::from_actual_with_domain(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
        &fixture::domain(),
    )
    .unwrap()
}

fn samples(
    phase: StructuredPhaseV2,
    inputs: &[Corner],
    rows: u32,
) -> Vec<StructuredNumericObservationV2> {
    let seed = fixture::population(phase, false).1.remove(0);
    inputs
        .iter()
        .cycle()
        .take(inputs.len() * 2)
        .enumerate()
        .map(|(index, &c)| {
            let mut sample = seed.clone();
            let ordinal = phase.index() as u64 * 128 + index as u64 + 1;
            sample.ordinal = ordinal;
            sample.call_id = ordinal;
            sample.membership.offered_ordinal = ordinal;
            sample.membership.member_ordinal = index as u64 + 1;
            sample.observed_at_ns = ordinal * 10;
            sample.wall_ns = 1000;
            let causes: Vec<_> = (0..rows)
                .filter_map(|position| {
                    let cause = if c.length {
                        Some(FinishReason::Length)
                    } else {
                        match c.cause {
                            1 => Some(FinishReason::EOS),
                            2 => Some(FinishReason::Stop),
                            _ => None,
                        }
                    };
                    cause.map(|cause| (position, cause))
                })
                .collect();
            sample.input = prepared(c, rows)
                .with_settled_terminal_causes(&causes)
                .unwrap();
            sample
        })
        .collect()
}

fn query(c: Corner, rows: u32) -> StructuredQueryV2 {
    StructuredQueryV2::exact(prepared(c, rows))
}

#[test]
fn physical_coverage_extension_preserves_each_actual_authorization_branch_in_all_phases() {
    let all = corners();
    for phase in [
        StructuredPhaseV2::Fit,
        StructuredPhaseV2::Residual,
        StructuredPhaseV2::Qualification,
    ] {
        let old = ChallengeCoverage::observe(&samples(phase, &all, 2)).unwrap();
        // Each loss is selected from original input/outcome facts, not by
        // editing coverage flags. The omitted query previously passed.
        for branch in 0..8 {
            let keep = |c: &Corner| match branch {
                0 => c.pending,
                1 => !c.pending,
                2 => c.length,
                3 => !c.length,
                4 => c.repetition != 0,
                5 => c.repetition == 0,
                6 => c.length || c.cause == 0,
                7 => c.length || c.cause != 0,
                _ => unreachable!(),
            };
            let subset: Vec<_> = all.iter().copied().filter(keep).collect();
            let replacement = ChallengeCoverage::observe(&samples(phase, &subset, 2)).unwrap();
            let omitted = all.iter().copied().find(|c| !keep(c)).unwrap();
            let q = query(omitted, 2);
            let upper = envelope::query_upper(&q, &fixture::domain()).unwrap();
            old.authorize(&q, &upper).unwrap();
            assert!(
                matches!(
                    replacement.authorize(&q, &upper),
                    Err(StructuredUnknown::QualificationCoverage)
                ),
                "phase={phase:?} branch={branch}"
            );
            assert!(
                !replacement.contains_coverage(&old),
                "phase={phase:?} branch={branch}"
            );
        }
    }
}

fn coverage_digest(coverage: &ChallengeCoverage) -> [u8; 32] {
    let mut digest = Sha256::new();
    coverage.bind(&mut digest);
    digest.finalize().into()
}

#[test]
fn physical_coverage_extension_ignores_zero_and_cause_facts_unused_by_authorize() {
    let all = corners();
    let eos_only: Vec<_> = all.iter().copied().filter(|c| c.cause != 2).collect();
    let both_masks: Vec<_> = all
        .iter()
        .flat_map(|&c| [c, Corner { mask: false, ..c }])
        .collect();
    for phase in [
        StructuredPhaseV2::Fit,
        StructuredPhaseV2::Residual,
        StructuredPhaseV2::Qualification,
    ] {
        let base = ChallengeCoverage::observe(&samples(phase, &all, 2)).unwrap();
        for (previous, next) in [(&both_masks, &all), (&all, &eos_only)] {
            let old = ChallengeCoverage::observe(&samples(phase, previous, 2)).unwrap();
            let replacement = ChallengeCoverage::observe(&samples(phase, next, 2)).unwrap();
            assert_ne!(coverage_digest(&replacement), coverage_digest(&old));
            assert!(replacement.contains_coverage(&old));
            for &c in previous {
                let q = query(c, 2);
                let upper = envelope::query_upper(&q, &fixture::domain()).unwrap();
                base.authorize(&q, &upper).unwrap();
                old.authorize(&q, &upper).unwrap();
                replacement.authorize(&q, &upper).unwrap();
            }
        }
    }
}

fn physical(rows: u32) -> PhysicalEnvelope {
    let corners = corners();
    let settings = StructuredSettingsV2 {
        max_wave_ns: 1500,
        static_margin_ns: 0,
        ..Default::default()
    };
    let contract = NonNegativeEnvelopeContractV1 {
        algorithm_universe: None,
        planning_estimator: NonNegativePlanningEstimatorV1::CoefficientEnvelopeV1,
        workload_domain: fixture::domain(),
        settings: Default::default(),
        challenge: WorkAxisAndBranchChallengesV1::WorkAxesAndPlainTextBranchesV1,
        template_policy: StructuredCostTemplatePolicyV1::OrderedV1,
        population_policy: StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
    };
    let mut physical = PhysicalEnvelope::fit(
        contract,
        &samples(StructuredPhaseV2::Fit, &corners, rows),
        &settings,
        None,
    )
    .unwrap();
    for phase in [
        StructuredPhaseV2::Residual,
        StructuredPhaseV2::Qualification,
    ] {
        physical
            .freeze_phase(phase, &samples(phase, &corners, rows))
            .unwrap();
    }
    physical
}

#[test]
fn physical_coverage_extension_smaller_positive_axes_do_not_promise_a_prediction() {
    let old = physical(2);
    let new = physical(1);
    let q = query(
        Corner {
            pending: true,
            length: true,
            repetition: 1,
            cause: 0,
            mask: true,
        },
        2,
    );
    assert!(new
        .certificate()
        .column_maxima
        .iter()
        .zip(&old.certificate().column_maxima)
        .any(|(n, o)| n < o));
    assert!(new.contains_input_coverage(&old));
    old.bounds(&q, 3).unwrap();
    // Identified positive directions permit magnitude extrapolation. Its
    // larger certified cost still obeys the unchanged numerical wave cap.
    assert!(matches!(
        new.bounds(&q, 3),
        Err(StructuredUnknown::Capacity)
    ));
}
