//! Actual TestRuntime submit/fence/retirement contracts, not device numerics.
use super::*;
use crate::operation::dispatch_contract::DisabledSubmissionWaveDispatchTimingSink;
use ferrum_types::InvocationPreparationStrategy;

struct Preparation;
impl InvocationPreparationSink for Preparation {
    fn record_preparation(&self, _: InvocationPreparationStats) {}
}

type ObservedRow = (std::ops::Range<u64>, std::ops::Range<u64>, u64);
type ObservedNode = (u32, Vec<ObservedRow>);
#[derive(Default)]
struct FactsSink(Mutex<Vec<Vec<ObservedNode>>>);

impl ExecutionEventSink for FactsSink {
    fn is_enabled(&self, _: ExecutionEventKind) -> bool {
        false
    }
    fn execution_facts_capture(&self) -> ExecutionFactsCapture {
        ExecutionFactsCapture::On
    }
    fn record(&self, _: EventEmissionPermit) -> Result<(), ExecutionEventSinkError> {
        Ok(())
    }
    fn record_execution_facts(
        &self,
        facts: TerminalExecutionFacts<'_>,
    ) -> Result<(), ExecutionEventSinkError> {
        assert!(facts.attribution().is_none());
        assert_eq!(
            facts.device_coverage(),
            DeviceExecutionFactsCoverage::Unavailable(
                DeviceExecutionFactsUnavailableReason::NotCaptured,
            )
        );
        self.0.lock().unwrap().push(
            facts
                .work()
                .nodes()
                .iter()
                .map(|node| {
                    (
                        node.node_index(),
                        node.participants()
                            .iter()
                            .map(|row| {
                                (
                                    row.immediate_token_range(),
                                    row.source_token_range(),
                                    row.full_input_tokens(),
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
        );
        Ok(())
    }
}

fn submitted_work(
    label: &str,
    capture: ExecutionFactsCapture,
    attribution_panic: bool,
    failed: bool,
) -> (
    OperationCompletionReceipt,
    SubmissionObservation,
    InvocationPreparationStats,
) {
    submitted_work_with_device(
        label,
        capture,
        attribution_panic,
        failed,
        DeviceCase::default(),
    )
}

#[derive(Default)]
struct DeviceCase {
    templates: Option<Arc<[ProjectionInvocationTemplate]>>,
    fault: DeviceFactsFault,
    attribution: bool,
    expect_contract: bool,
}

fn submitted_work_with_device(
    label: &str,
    capture: ExecutionFactsCapture,
    attribution_panic: bool,
    failed: bool,
    device: DeviceCase,
) -> (
    OperationCompletionReceipt,
    SubmissionObservation,
    InvocationPreparationStats,
) {
    let fixture = if device.attribution {
        fixture_with_provider_behavior(false, ProviderBehavior::ScratchOverwrite)
    } else {
        fixture()
    };
    // Lexical request order intentionally differs from the actual admitted order.
    let resources = ["z", "a", "m"].map(|name| {
        logical_resources(
            &fixture.plan_resources,
            &format!("run.work-facts.{label}.{name}"),
            &format!("request.work-facts.{label}.{name}"),
        )
    });
    let sessions = resources
        .iter()
        .map(|r| r.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let spans = [1..2, 0..3, 2..4]
        .map(|range| TokenSpanWork::from_token_ids(&[1, 2, 3, 4], range).unwrap());
    let step = step_for(&batch, &lane, spans.into());
    let step_weak = Arc::downgrade(&step);
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = batch
        .sessions()
        .iter()
        .map(|s| TrustedActiveSequenceBinding::from_session(s).unwrap())
        .collect::<Vec<_>>();
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, &lane).unwrap();
    let identity = OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
        &topology,
        active.iter(),
        &wave,
        &lane,
        InvocationPreparationStrategy::IdentityProjection,
    )
    .unwrap();
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let reaper = CompletionReaper::new();
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.submission_attribution_panics = attribution_panic;
        trace.device_facts_templates = device.templates;
        trace.device_facts_fault = device.fault;
        trace.fence_behavior = if failed {
            FenceBehavior::FailedButQuiescent
        } else {
            FenceBehavior::Succeeded
        };
    }
    let dispatched = OperationDispatch::encode_and_submit_wave_with_inputs_and_preparation(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        if device.attribution {
            DeviceTimingMode::Kernel
        } else {
            DeviceTimingMode::Off
        },
        capture,
        &[],
        SubmissionExecutionPolicy::adaptive(),
        InvocationPreparationStrategy::IdentityProjection,
        &DisabledSubmissionWaveDispatchTimingSink,
        &Preparation,
        wave,
        &lane,
        &reaper,
    );
    let (handle, observation) = if attribution_panic || device.expect_contract {
        match dispatched {
            Err(SubmissionWaveDispatchError::PostSubmitContract {
                completion,
                observation,
                ..
            }) => (completion, observation),
            _ => panic!("post-submit attribution panic must retain the original handle and facts"),
        }
    } else {
        dispatched.unwrap().into_parts()
    };
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 1);
    assert_eq!(
        fixture.runtime_trace.lock().unwrap().device_facts_calls,
        usize::from(capture == ExecutionFactsCapture::On)
    );
    assert!(fixture
        .provider_trace
        .lock()
        .unwrap()
        .facts_capture_at_encode
        .iter()
        .all(|observed| *observed == capture));
    assert_eq!(
        handle.receipt().has_materialized_participant_receipts(),
        false
    );
    assert_eq!(identity.preparation_snapshot().parts_materialized, 0);
    let snapshot = identity.preparation_snapshot();
    let completion = match handle.wait().unwrap() {
        CompletionObservation::Terminal(receipt) => receipt,
        _ => panic!("actual submitted test fence must reach terminal"),
    };
    assert_eq!(
        observation.work().is_some(),
        capture == ExecutionFactsCapture::On
    );
    if let Some(work) = observation.work() {
        assert_eq!(work.submission(), completion.submission());
        assert_eq!(work.nodes().len(), identity.node_count());
        for (admitted_node, node) in work.nodes().iter().enumerate() {
            assert_eq!(node.node_index() as usize, admitted_node);
            for (row, authority) in node
                .participants()
                .iter()
                .zip(step.work_shape().participants())
            {
                assert_eq!(row.participant(), *authority);
            }
        }
    }
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(lane.in_flight_count(), 0);
    drop(handle);
    if failed {
        step.try_abort().unwrap();
    } else {
        step.try_retire_normal().unwrap();
    }
    assert!(
        step_weak.upgrade().is_none(),
        "work facts must not retain the Step owner"
    );
    drop(providers);
    drop(active);
    drop(batch);
    for session in &sessions {
        if failed {
            session.try_abort().unwrap();
        } else {
            session.try_complete().unwrap();
        }
    }
    drop(sessions);
    drop(resources);
    drop(reaper);
    drop(lane);
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
    (completion, observation, snapshot)
}

pub(super) fn projection_templates(
    looped: bool,
    participants: u32,
) -> Arc<[ProjectionInvocationTemplate]> {
    let count = if looped { participants } else { 1 };
    (0..count)
        .map(|ordinal| ProjectionInvocationTemplate {
            ordinal,
            role: DeviceProjectionRole::OutputHead,
            component: id("weight.test.observation"),
            dimensions: ProjectionDimensions {
                m: if looped { 1 } else { participants },
                k: 32,
                n: 64,
            },
            layout: ProjectionLayout {
                input_stride: 32,
                output_stride: 64,
                output_offset: 0,
                input_type: ElementType::F32,
                output_type: ElementType::F32,
            },
            arithmetic: ProjectionArithmetic::Direct {
                implementation: DeviceNativeOperationId::new("test_projection").unwrap(),
                row_tile: 1,
                launch: ProjectionLaunchGeometry {
                    grid: [1, 1, 1],
                    block: [32, 1, 1],
                    shared_bytes: 0,
                },
            },
            input_transforms: Box::new([]),
            rows: if looped {
                vec![ParticipantRowSelector {
                    participant_index: ordinal,
                    source: ParticipantTokenSelector::LastSourceToken,
                    physical_rows: 0..1,
                }]
            } else {
                (0..participants)
                    .map(|index| ParticipantRowSelector {
                        participant_index: index,
                        source: ParticipantTokenSelector::LastSourceToken,
                        physical_rows: index..index + 1,
                    })
                    .collect()
            }
            .into_boxed_slice(),
        })
        .collect::<Vec<_>>()
        .into()
}

#[test]
fn execution_device_facts_bind_real_unequal_windows_without_off_materialization() {
    let templates = projection_templates(false, 3);
    let (_, off, off_stats) = submitted_work_with_device(
        "device-off",
        ExecutionFactsCapture::Off,
        false,
        false,
        DeviceCase {
            templates: Some(Arc::clone(&templates)),
            ..Default::default()
        },
    );
    assert!(off.device_facts().is_none());
    let (completion, observed, stats) = submitted_work_with_device(
        "device-on",
        ExecutionFactsCapture::On,
        false,
        false,
        DeviceCase {
            templates: Some(Arc::clone(&templates)),
            ..Default::default()
        },
    );
    assert_eq!(stats, off_stats);
    let bound = observed.device_facts().unwrap();
    assert_eq!(bound.submission(), completion.submission());
    let facts =
        TerminalExecutionFacts::new(&completion, observed.work().unwrap(), None, Some(bound))
            .unwrap();
    assert_eq!(
        facts.device_coverage(),
        DeviceExecutionFactsCoverage::ComputeCommands {
            described_compute_commands: observed.work().unwrap().nodes().len() as u32,
            unavailable_compute_commands: 0,
        }
    );
    for command in bound.device().compute_commands() {
        let ProjectionCommandCoverage::Covered(actual) = &command.coverage else {
            panic!("fixture projection must be covered");
        };
        assert!(
            Arc::ptr_eq(actual, &templates),
            "binding must not deep-clone cached templates"
        );
        assert_eq!(
            actual[0].dimensions.m, 3,
            "selected last-token matrix rows differ from six packed input tokens"
        );
    }
    for projection in bound.projection_rows() {
        assert_eq!(
            projection
                .rows
                .iter()
                .map(|row| (
                    row.participant_index,
                    row.source_token_range.clone(),
                    row.packed_token_range.clone(),
                    row.physical_rows.clone()
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, 1..2, 0..1, 0..1),
                (1, 2..3, 3..4, 1..2),
                (2, 3..4, 5..6, 2..3)
            ]
        );
    }
    let (foreign, foreign_observed, _) =
        submitted_work("foreign-device", ExecutionFactsCapture::On, false, false);
    assert!(TerminalExecutionFacts::new(
        &foreign,
        foreign_observed.work().unwrap(),
        None,
        Some(bound)
    )
    .is_err());
}

#[test]
fn execution_device_facts_bind_participant_loop_to_actual_source_rows() {
    let (_, observed, _) = submitted_work_with_device(
        "device-loop",
        ExecutionFactsCapture::On,
        false,
        false,
        DeviceCase {
            templates: Some(projection_templates(true, 3)),
            ..Default::default()
        },
    );
    let bound = observed.device_facts().unwrap();
    for node in observed.work().unwrap().nodes() {
        let projections = bound
            .projection_rows()
            .iter()
            .filter(|row| row.node_index == node.node_index())
            .collect::<Vec<_>>();
        assert_eq!(projections.len(), 3);
        for (participant, projection) in projections.into_iter().enumerate() {
            assert_eq!(projection.rows.len(), 1);
            let row = &projection.rows[0];
            assert_eq!(row.participant_index, participant as u32);
            assert_eq!(row.physical_rows, 0..1);
            let source = node.participants()[participant].source_token_range();
            let packed = node.participants()[participant].immediate_token_range();
            assert_eq!(row.source_token_range, source.end - 1..source.end);
            assert_eq!(row.packed_token_range, packed.end - 1..packed.end);
        }
    }
}

#[test]
fn execution_device_facts_post_submit_failure_retains_original_completion_and_other_attachment() {
    for fault in [
        DeviceFactsFault::GetterError,
        DeviceFactsFault::GetterPanic,
        DeviceFactsFault::MissingCompute,
        DeviceFactsFault::WrongNode,
        DeviceFactsFault::WrongReplayOrdinal,
    ] {
        let (completion, observation, _) = submitted_work_with_device(
            "device-invalid",
            ExecutionFactsCapture::On,
            false,
            false,
            DeviceCase {
                templates: Some(projection_templates(false, 3)),
                fault,
                attribution: true,
                expect_contract: true,
            },
        );
        assert_eq!(
            completion.disposition(),
            &OperationCompletionDisposition::Succeeded
        );
        assert!(observation.work().is_some());
        assert!(
            observation.attribution().is_some(),
            "a facts failure must retain valid original attribution"
        );
        assert!(observation.device_facts().is_none());
    }
    let (completion, observation, _) = submitted_work_with_device(
        "attr-invalid-device-valid",
        ExecutionFactsCapture::On,
        true,
        false,
        DeviceCase {
            templates: Some(projection_templates(false, 3)),
            ..Default::default()
        },
    );
    assert!(observation.attribution().is_none());
    assert!(observation.device_facts().is_some());
    assert_eq!(
        completion.disposition(),
        &OperationCompletionDisposition::Succeeded
    );
}

#[test]
fn execution_device_facts_adaptive_replay_rejects_malformed_post_submit_ordinals() {
    for fault in [
        DeviceFactsFault::AdaptiveReplayFirstOrdinal,
        DeviceFactsFault::AdaptiveReplaySkippedOrdinal,
    ] {
        let (completion, observation, _) = submitted_work_with_device(
            "adaptive-bad-ordinal",
            ExecutionFactsCapture::On,
            false,
            false,
            DeviceCase {
                templates: Some(projection_templates(false, 3)),
                fault,
                attribution: true,
                expect_contract: true,
            },
        );
        // The real submitted fence still succeeds. Only the deliberately false
        // replay coordinates are rejected; the original work/attribution remain.
        assert_eq!(
            completion.disposition(),
            &OperationCompletionDisposition::Succeeded
        );
        assert!(observation.work().is_some());
        assert!(observation.attribution().is_some());
        assert!(observation.device_facts().is_none());
    }
}

fn packed_templates(geometry: PackedProjectionGeometry) -> Arc<[ProjectionInvocationTemplate]> {
    let mut templates = projection_templates(false, 3).as_ref().to_vec();
    templates[0].arithmetic = ProjectionArithmetic::PackedQuantized {
        implementation: DeviceNativeOperationId::new("test_packed_projection").unwrap(),
        geometry,
    };
    templates.into()
}

#[test]
fn execution_device_facts_packed_workspace_rejects_inconsistent_post_submit_bytes() {
    // These are neutral structural observations on TestRuntime, not Q6 ABI
    // geometry. Native planning retains its own exact layout validation.
    let base = PackedProjectionGeometry {
        padded_k: 32,
        padded_n: 64,
        row_tile: 1,
        column_tile: 1,
        blocks: 1,
        threads: 32,
        shared_bytes: 0,
        fixup: false,
        guard_blocks: 0,
        pack_abi: 1,
        packed_bytes: 64,
        fixup_bytes: 0,
        workspace_bytes: 64,
    };
    for geometry in [
        base,
        PackedProjectionGeometry {
            fixup: true,
            fixup_bytes: 32,
            workspace_bytes: 128,
            ..base
        },
    ] {
        let (completion, observation, _) = submitted_work_with_device(
            "packed-valid-workspace",
            ExecutionFactsCapture::On,
            false,
            false,
            DeviceCase {
                templates: Some(packed_templates(geometry)),
                ..Default::default()
            },
        );
        assert_eq!(
            completion.disposition(),
            &OperationCompletionDisposition::Succeeded
        );
        assert!(observation.device_facts().is_some());
    }
    for geometry in [
        PackedProjectionGeometry {
            packed_bytes: 0,
            ..base
        },
        PackedProjectionGeometry {
            fixup: true,
            ..base
        },
        PackedProjectionGeometry {
            fixup_bytes: 16,
            workspace_bytes: 96,
            ..base
        },
        PackedProjectionGeometry {
            fixup: true,
            fixup_bytes: 32,
            workspace_bytes: 95,
            ..base
        },
        PackedProjectionGeometry {
            packed_bytes: u64::MAX,
            fixup: true,
            fixup_bytes: 32,
            workspace_bytes: u64::MAX,
            ..base
        },
    ] {
        let (completion, observation, _) = submitted_work_with_device(
            "packed-invalid-workspace",
            ExecutionFactsCapture::On,
            false,
            false,
            DeviceCase {
                templates: Some(packed_templates(geometry)),
                attribution: true,
                expect_contract: true,
                ..Default::default()
            },
        );
        assert_eq!(
            completion.disposition(),
            &OperationCompletionDisposition::Succeeded
        );
        assert!(observation.work().is_some());
        assert!(observation.attribution().is_some());
        assert!(observation.device_facts().is_none());
    }
}

#[test]
fn execution_device_facts_invalid_row_extent_retains_actual_failed_terminal() {
    let mut templates = projection_templates(false, 3).as_ref().to_vec();
    templates[0].dimensions.m = 4;
    let (completion, observation, _) = submitted_work_with_device(
        "invalid-m",
        ExecutionFactsCapture::On,
        false,
        true,
        DeviceCase {
            templates: Some(templates.into()),
            attribution: true,
            expect_contract: true,
            ..Default::default()
        },
    );
    assert!(matches!(
        completion.disposition(),
        OperationCompletionDisposition::FailedButQuiescent(_)
    ));
    assert!(observation.attribution().is_some());
    assert!(observation.device_facts().is_none());
}

#[test]
fn execution_work_facts_preserve_real_unequal_windows_and_off_materialization() {
    let (off_completion, off, off_preparation) =
        submitted_work("off", ExecutionFactsCapture::Off, false, false);
    assert!(off.work().is_none());
    assert!(off.attribution().is_none());
    let (completion, observation, preparation) =
        submitted_work("on", ExecutionFactsCapture::On, false, false);
    assert_eq!(
        preparation, off_preparation,
        "observation must not materialize more identity parts"
    );
    let work = observation.work().unwrap();
    assert!(!completion
        .submission()
        .has_materialized_participant_receipts());
    assert!(TerminalExecutionFacts::new(&off_completion, work, None, None).is_err());
    let sink = FactsSink::default();
    sink.record_execution_facts(
        TerminalExecutionFacts::new(&completion, work, None, None).unwrap(),
    )
    .unwrap();
    let records = sink.0.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].len(),
        completion.submission().batch_identity().node_count()
    );
    for (node_index, rows) in &records[0] {
        assert_eq!(
            rows,
            &vec![(0..1, 1..2, 4), (1..4, 0..3, 4), (4..6, 2..4, 4)]
        );
        assert!((*node_index as usize) < records[0].len());
    }
    assert!(!completion
        .submission()
        .has_materialized_participant_receipts());
}

#[test]
fn execution_work_facts_survive_real_post_submit_contract_failure() {
    let (completion, observation, _) =
        submitted_work("post-submit", ExecutionFactsCapture::On, true, false);
    let facts =
        TerminalExecutionFacts::new(&completion, observation.work().unwrap(), None, None).unwrap();
    assert_eq!(
        facts.completion().disposition(),
        &OperationCompletionDisposition::Succeeded
    );
    assert_eq!(
        facts.work().nodes()[0].participants()[1].immediate_token_range(),
        1..4
    );
}

#[test]
fn execution_work_facts_keep_actual_failed_terminal_disposition() {
    let (completion, observation, _) =
        submitted_work("failed", ExecutionFactsCapture::On, false, true);
    let facts =
        TerminalExecutionFacts::new(&completion, observation.work().unwrap(), None, None).unwrap();
    assert!(matches!(
        facts.completion().disposition(),
        OperationCompletionDisposition::FailedButQuiescent(_)
    ));
    assert_eq!(
        facts.work().nodes()[0].participants()[2].source_token_range(),
        2..4
    );
}
