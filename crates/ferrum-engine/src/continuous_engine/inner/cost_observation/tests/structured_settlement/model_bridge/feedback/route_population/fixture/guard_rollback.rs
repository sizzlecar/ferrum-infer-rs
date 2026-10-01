//! Real native guard rejection, original Step rollback, observation and FIFO.
use super::*;
use ferrum_interfaces::execution_cost::{
    CallNoSubmissionReasonV1, CoreReadbackRoute, ExpectedWaveInput, ExpectedWaveWork,
    ExpectedWorkSelection, GuardedNotSubmittedReason, HostSubmissionRejection,
};
use ferrum_interfaces::model_executor::PrefillChunk;
use ferrum_interfaces::vnext::*;
use std::num::NonZeroU64;

#[derive(Clone, Copy)]
pub(in crate::continuous_engine::inner::cost_observation) enum GuardRollbackFault {
    None,
    UnboundReceipt,
    ChangedParticipant,
    RepeatedPreparation,
    Lost,
    Unknown,
    ObservedWave,
}
struct Reject;
impl PreparedWaveSubmissionGuard for Reject {
    fn check(
        &self,
        actual: &DeviceSubmissionAttribution,
        _: CoreReadbackRoute,
    ) -> std::result::Result<(), GuardedNotSubmittedReason> {
        assert!(!actual.commands().is_empty());
        Err(GuardedNotSubmittedReason::HostRejected(
            HostSubmissionRejection::WitnessExpired,
        ))
    }
}

pub(in crate::continuous_engine::inner::cost_observation) fn record_guard_rollback(
    runtime: &EngineCostRuntime,
    clock: &Arc<VirtualClock>,
    w: Wave,
    fault: GuardRollbackFault,
) {
    let fixture =
        core::fixture_with_provider_behavior(false, core::ProviderBehavior::ScratchOverwrite);
    let (fixture, sequences, sessions, batch, step) = setup_cohort(fixture, &w.actual.rows);
    let lane = Arc::clone(step.execution_lane());
    // Capture the original checked work while all real participants are Ready.
    step.try_rollback_unsubmitted().unwrap();
    let refs = sessions.iter().map(AsRef::as_ref).collect::<Vec<_>>();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let view = loop {
        match fixture.plan_resources.resource_planning_view_on_lane(
            &refs,
            &lane,
            ResourcePlanningLimits::default(),
            &mut || true,
        ) {
            ResourcePlanningAvailability::Known(view) => break view,
            // This pure capture uses try_lock, including the device budget
            // shared by parallel CPU fixtures. Retry only that typed temporary
            // read failure; Busy/stale/unsupported inputs remain test failures.
            ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::ReadUnavailable(_))
                if std::time::Instant::now() < deadline =>
            {
                std::thread::yield_now();
            }
            other => panic!("original ready resource capture: {other:?}"),
        }
    };
    let expected = ExpectedWaveWork::new(
        &view,
        ActualWaveKind::Prefill,
        w.actual
            .rows
            .iter()
            .enumerate()
            .map(|(i, row)| ExpectedWorkSelection {
                participant_index: i,
                request_id: row.request_id.clone(),
                owner_incarnation: NonZeroU64::new(row.owner_incarnation).unwrap(),
                work_generation: NonZeroU64::new(row.work_generation).unwrap(),
                input: ExpectedWaveInput::Prefill {
                    chunk: PrefillChunk::new(0, 1, 1).unwrap(),
                },
                work: ActualRowWork::Prefill {
                    offset: 0,
                    count: 1,
                    total_prompt_tokens: 1,
                },
                decode_policy: None,
            })
            .collect(),
    )
    .unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(
                w.actual
                    .rows
                    .iter()
                    .map(|_| core::one_token_span())
                    .collect(),
            )
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let step = match batch.try_begin_step(request, &lane).unwrap() {
        StepResourceAdmissionDecision::Admitted(step) => step,
        _ => panic!("original rolled-back Step must remain reusable"),
    };
    let wave = wave_core::prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let active = sessions
        .iter()
        .map(|s| TrustedActiveSequenceBinding::from_session(s).unwrap())
        .collect::<Vec<_>>();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    fixture.runtime_trace.lock().unwrap().cost_graph_state = Some(
        DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::Unconfigured, 0, 0, 0)
            .unwrap(),
    );
    let route = OperationDispatch::observe_non_reusable_cost_route_for_wave(&lane, &wave);
    assert_eq!(
        route.class(),
        ferrum_interfaces::execution_cost::PreparedCostRouteClassV1::GraphDisabled
    );
    let start = clock.now_ns().unwrap() + 100;
    clock.set(start);
    let ticket = runtime
        .reserve_live_ticket(Some(start))
        .expect("original live offer");
    let mut participants = w
        .actual
        .rows
        .iter()
        .map(|row| CostObservationParticipant {
            request_id: row.request_id.clone(),
            owner_incarnation: row.owner_incarnation,
            work_generation: row.work_generation,
            input_index: row.input_index,
            output_policy_signature: Some([6; 32]),
            host_features: Some(w.host),
        })
        .collect::<Vec<_>>();
    if matches!(fault, GuardRollbackFault::ChangedParticipant) {
        participants[0].work_generation += 1;
    }
    let mut call = EngineCostCall::begin(
        &runtime.ids,
        clock.clone(),
        runtime.sink.clone(),
        EngineCostCallSpec {
            identity: super::identity(),
            participants,
            prepare_started_at_ns: Some(start),
            boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
            recorder_limits: CostRecorderLimits {
                max_waves: 1,
                max_rows_per_wave: 8,
                max_retained_rows: 128,
            },
        },
    )
    .unwrap()
    .with_structured_capture(true)
    .with_live_ticket(Some(ticket));
    clock.set(start + 1);
    let mut context = call.context().unwrap();
    context.prepared_route(route.clone(), Ok(Vec::new()));
    if matches!(fault, GuardRollbackFault::RepeatedPreparation) {
        context.prepared_route(route, Ok(Vec::new()));
    }
    let reaper = CompletionReaper::new();
    let pending = match OperationDispatch::encode_and_submit_guarded_wave(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        &[],
        &Reject,
        wave,
        &lane,
        &reaper,
    ) {
        GuardedWaveSubmissionOutcome::NotSubmitted(pending) => pending,
        _ => panic!("real core guard must reject before physical submission"),
    };
    assert!(fixture.provider_trace.lock().unwrap().encode_calls > 0);
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.submit_calls, 0);
        assert_eq!(trace.guarded_submit_calls, 1);
    }
    let receipt = if matches!(fault, GuardRollbackFault::UnboundReceipt) {
        pending.reconcile_step(step)
    } else {
        pending.reconcile_step_with_observation(step, &expected)
    }
    .unwrap_or_else(|(e, _)| panic!("original rejected Step rollback: {e}"));
    if matches!(fault, GuardRollbackFault::Unknown) {
        context.mark_unknown(ActualWaveEvidenceUnknown::ProviderPath);
    }
    if matches!(fault, GuardRollbackFault::ObservedWave) {
        context.physical_wave(Ok(w.actual.clone()), Some(start + 2));
    }
    clock.set(start + 3);
    context.finish_guard_rollback(&receipt);
    drop(context);
    if matches!(fault, GuardRollbackFault::Lost) {
        call.recorder.note_lost(1);
    }
    if matches!(fault, GuardRollbackFault::None) {
        let proof = call
            .recorder
            .no_submission()
            .expect("same original successful rollback");
        assert!(matches!(
            proof.reason(),
            CallNoSubmissionReasonV1::GuardRollback { .. }
        ));
        assert_eq!(
            proof.reason().protocol(),
            "ferrum.inference-guard-rollback.v1"
        );
    } else {
        assert!(call.recorder.no_submission().is_none());
    }
    clock.set(start + 5);
    assert_eq!(call.finish(), CostCallDisposition::Queued);
    drop(active);
    drop(identity);
    drop(providers);
    drop(lane);
    drop(batch);
    for session in &sessions {
        // Neither rolled-back Step retired a physical frame. Completed would
        // invent that progress; teardown aborts only the genuinely idle session
        // after the original observation/rollback has already been checked.
        session.try_abort_if_quiescent().unwrap();
    }
    drop(sessions);
    drop(sequences);
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
    clock.set(start + 20);
    runtime.consume_samples();
}
