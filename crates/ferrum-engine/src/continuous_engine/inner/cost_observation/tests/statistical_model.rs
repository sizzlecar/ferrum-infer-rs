use super::*;
use ferrum_interfaces::{model_executor::ExecutorCompletionWork, vnext::DeviceCommandPhase};
use ferrum_scheduler::implementations::continuous::cost_model::statistical::model::ModelUnknown;
use ferrum_types::{FinishReason, InferenceRequest, TokenId};

// Reuses the engine's real recorder, FIFO acceptance and private settlement
// receipt. It is CPU producer-contract evidence, not a device performance run.
fn recorded() -> (u64, CostEvidenceEntry) {
    recorded_with(&EngineCostIds::default(), &sink(4, 32))
}
fn recorded_with(
    ids: &EngineCostIds,
    queue: &Arc<BoundedCostSampleSink>,
) -> (u64, CostEvidenceEntry) {
    let (ordinal, entry, _) = recorded_with_capture(ids, queue);
    (ordinal, entry)
}
fn recorded_with_capture(
    ids: &EngineCostIds,
    queue: &Arc<BoundedCostSampleSink>,
) -> (u64, CostEvidenceEntry, Arc<CostCalibrationCapture>) {
    recorded_case(ids, queue, true, 0, 0)
}
fn recorded_case(
    ids: &EngineCostIds,
    queue: &Arc<BoundedCostSampleSink>,
    terminal: bool,
    epoch: u64,
    host_delay_ns: u64,
) -> (u64, CostEvidenceEntry, Arc<CostCalibrationCapture>) {
    let work = ActualRowWork::Decode { kv_tokens: 7 };
    let mut command = SelectedCommandCostBuilderV1::new(1);
    command
        .kernel(
            SelectedAlgorithmClassV1::new("fixture.selected", 1, [1; 32], [2; 32]).unwrap(),
            KernelNumericWorkV1 {
                logical_units: 32,
                padded_units: 32,
                inner_units_per_logical_unit: 32,
                grid: [1, 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    let selected = command.finish().unwrap();
    let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::FullLogits);
    builder
        .physical_command(CostPhysicalCommand {
            native_op_id: "fixture.selected",
            command_index: 0,
            node_index: None,
            command_phase: DeviceCommandPhase::Compute,
            provider: None,
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: 1,
            token_count: 1,
            batching_form: "one",
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            reusable_graph_node_count: None,
            statistical_evidence: Some(&selected),
        })
        .unwrap();
    builder
        .core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    builder
        .row(CanonicalCostRow {
            work,
            host_policy_signature: [6; 32],
            mask_upload_required: false,
            output: CostRowOutput::Decode {
                requires_full_logits: true,
                repetition_tokens: 0,
                repetition_penalty_bits: 1f32.to_bits(),
            },
            host_features: Some(HostCostFeaturesV1 {
                policy: HostCostPolicyV2 {
                    empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                    categorical_signature: [9; 32],
                    decoder_text_bytes_per_token: 4,
                    decoder_scratch_bytes_per_token: 8,
                    raw_token_bytes_bound: 4,
                },
                state: HostCostStateV1 {
                    generated_tokens_before: 2,
                    maximum_output_tokens: if terminal { 3 } else { 4 },
                    sampling_history_tokens: 2,
                    sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                    pending_decoded_utf8: false,
                    completion_state_signature: satisfied_completion_cost_signature(),
                },
            }),
        })
        .unwrap();
    let canonical = builder
        .finish_with_statistics(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            64,
        )
        .unwrap();
    let mut actual = shape(&[work]);
    actual.provider_signature = canonical.exact.provider_signature;
    actual.output_policy_signature = canonical.exact.output_policy_signature;
    actual.numeric_features = canonical.exact.numeric_features;
    actual.host_content_features = canonical.exact.host_content_features;
    actual.row_multiset_features = canonical.exact.row_multiset_features;
    actual.statistical_evidence = Some(canonical.statistical.unwrap());
    let clock = Arc::new(VirtualClock(AtomicU64::new(epoch + 2)));
    let mut call = EngineCostCall::begin(
        ids,
        clock.clone(),
        queue.clone(),
        EngineCostCallSpec {
            identity: identity(),
            participants: actual
                .rows
                .iter()
                .map(|row| CostObservationParticipant {
                    request_id: row.request_id.clone(),
                    owner_incarnation: row.owner_incarnation,
                    work_generation: row.work_generation,
                    input_index: row.input_index,
                    output_policy_signature: Some([6; 32]),
                    host_features: None,
                })
                .collect(),
            prepare_started_at_ns: Some(epoch + 1),
            boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
            recorder_limits: CostRecorderLimits {
                max_waves: 4,
                max_rows_per_wave: 8,
                max_retained_rows: 32,
            },
        },
    )
    .unwrap();
    let capture = Arc::new(CostCalibrationCapture::default());
    call.attach_calibration_capture(Arc::clone(&capture));
    let mut context = call.context().unwrap();
    context.physical_wave(Ok(actual.clone()), Some(epoch + 3));
    clock.set(epoch + 6);
    context.terminal(ActualWaveOutcome::Completed, None);
    context.finish_call(ObservedCallOutcome::Completed);
    drop(context);
    let row = &actual.rows[0];
    clock.set(epoch + 10);
    call.begin_host_row(&row.request_id);
    clock.set(epoch + 11);
    let evidence = committed(row, epoch + 11);
    call.note_host_token_commit(&evidence);
    if !terminal {
        clock.set(epoch + 20 + host_delay_ns);
        assert!(call.host_publication(&evidence, false, true).is_none());
        call.record_host_result(evidence);
        clock.set(epoch + 21 + host_delay_ns);
        assert_eq!(call.finish(), CostCallDisposition::Published);
        let (ordinal, entry) = queue.pop_numbered().unwrap();
        return (ordinal, entry, capture);
    }
    let mut pending = call.host_publication(&evidence, true, true).unwrap();
    clock.set(epoch + 15);
    pending.terminal_handed_off();
    let mut request = InferenceRequest::new("input", "model");
    request.id = row.request_id.clone();
    let mut owner = crate::continuous_engine::SequenceState::new(request, vec![TokenId::new(1)]);
    owner.cost_frontier = Some(CostFrontier {
        owner_incarnation: NonZeroU64::new(row.owner_incarnation).unwrap(),
        work_generation: NonZeroU64::new(row.work_generation + 1).unwrap(),
    });
    clock.set(epoch + 20 + host_delay_ns);
    call.record_settled(pending.settle(
        owner,
        HostTerminalStageV1 {
            finish_reason: FinishReason::Length,
            generated_tokens: 3,
            through_output_ordinal: 3,
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
        },
    ));
    call.reject(CostCallRejection::Composite);
    clock.set(epoch + 21 + host_delay_ns);
    assert_eq!(
        call.finish(),
        CostCallDisposition::Rejected(CostCallRejection::Composite)
    );
    let (ordinal, entry) = queue.pop_numbered().unwrap();
    (ordinal, entry, capture)
}
#[test]
fn whole_wave_converter_consumes_same_fifo_terminal_receipt() {
    let (ordinal, entry) = recorded();
    let result = super::super::trainer::whole_wave_observation(&entry, ordinal, [7; 32]).unwrap();
    assert_eq!(result.accepted_ordinal, ordinal);
    assert_eq!(result.wall_ns, 19);
    assert_eq!(result.observed_at_ns, 21);
    assert_eq!(
        result.boundary,
        model::CostBoundary::PreparationToHostSettledV1
    );
    assert_eq!(
        result.exact.rows,
        vec![ActualRowWork::Decode { kv_tokens: 7 }]
    );
    result.selected.validate_exact(&result.exact).unwrap();
    let CostEvidenceEntry::StagesOnly { stages, .. } = &entry else {
        panic!("terminal must remain legacy-rejected")
    };
    assert_eq!(result.call_id, stages.call_id);
}
#[test]
fn converter_rejects_missing_unbound_failed_or_unaccepted_evidence() {
    let (ordinal, entry) = recorded();
    let CostEvidenceEntry::StagesOnly {
        stages,
        legacy_rejection,
    } = entry
    else {
        panic!()
    };
    for case in 0..4 {
        let mut altered = (*stages).clone();
        let mut legacy = legacy_rejection;
        match case {
            0 => altered.statistical_evidence = None,
            1 => altered.actual_shape.as_mut().unwrap().provider_signature = [91; 32],
            2 => altered.completeness = HostStageCompleteness::Failed,
            _ => legacy = CostCallRejection::ExecutorFailed,
        }
        let altered = CostEvidenceEntry::StagesOnly {
            stages: Arc::new(altered),
            legacy_rejection: legacy,
        };
        assert!(super::super::trainer::whole_wave_observation(&altered, ordinal, [7; 32]).is_err());
    }
    let entry = CostEvidenceEntry::StagesOnly {
        stages,
        legacy_rejection,
    };
    assert!(matches!(
        super::super::trainer::whole_wave_observation(&entry, 0, [7; 32]),
        Err(ModelUnknown::WrongSource)
    ));
}

#[path = "statistical_model/runtime.rs"]
mod runtime;

#[path = "statistical_model/capture.rs"]
pub(in crate::continuous_engine::inner) mod capture;
