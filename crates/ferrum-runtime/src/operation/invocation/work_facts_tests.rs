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
    let fixture = fixture();
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
        DeviceTimingMode::Off,
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
    let (handle, observation) = if attribution_panic {
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
    assert!(TerminalExecutionFacts::new(&off_completion, work, None).is_err());
    let sink = FactsSink::default();
    sink.record_execution_facts(TerminalExecutionFacts::new(&completion, work, None).unwrap())
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
        TerminalExecutionFacts::new(&completion, observation.work().unwrap(), None).unwrap();
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
        TerminalExecutionFacts::new(&completion, observation.work().unwrap(), None).unwrap();
    assert!(matches!(
        facts.completion().disposition(),
        OperationCompletionDisposition::FailedButQuiescent(_)
    ));
    assert_eq!(
        facts.work().nodes()[0].participants()[2].source_token_range(),
        2..4
    );
}
