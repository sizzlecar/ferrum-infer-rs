//! Bounded diagnostics exercise real CPU submission/retirement contracts.
//! They do not establish native backend timing or performance equivalence.
use super::{captures_participant_wave, VNextExecutionJournal, VNextWaveInstrumentation};
use ferrum_interfaces::model_executor::ExecutorRequestOrigin;
use std::num::NonZeroU32;

#[path = "../../../../ferrum-interfaces/tests/vnext_device_operation_contract/mod.rs"]
mod vnext_device_operation_contract;
#[path = "../../../../ferrum-interfaces/tests/vnext_device_operation_wave_contract/mod.rs"]
mod vnext_device_operation_wave_contract;
use vnext_device_operation_contract::*;
use vnext_device_operation_wave_contract::*;

struct CaptureSink {
    policy: ExecutionEventCapturePolicy,
    mode: DeviceTimingMode,
    kinds: Mutex<Vec<ExecutionEventKind>>,
    summaries: Mutex<Vec<ExecutionFrameCaptureSummary>>,
    fail_summary: AtomicBool,
    summary_attempts: std::sync::atomic::AtomicUsize,
}

impl CaptureSink {
    fn bounded(limit: u32, mode: DeviceTimingMode) -> Arc<Self> {
        Arc::new(Self {
            policy: ExecutionEventCapturePolicy::FirstFramesPerRequest(
                NonZeroU32::new(limit).unwrap(),
            ),
            mode,
            kinds: Mutex::new(Vec::new()),
            summaries: Mutex::new(Vec::new()),
            fail_summary: AtomicBool::new(false),
            summary_attempts: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}

impl ExecutionEventSink for CaptureSink {
    fn enablement(&self) -> ExecutionEventSinkEnablement {
        ExecutionEventSinkEnablement::All
    }

    fn is_enabled(&self, _: ExecutionEventKind) -> bool {
        true
    }

    fn device_timing_mode(&self) -> DeviceTimingMode {
        self.mode
    }

    fn capture_policy(&self) -> ExecutionEventCapturePolicy {
        self.policy
    }

    fn record(&self, permit: EventEmissionPermit) -> Result<(), ExecutionEventSinkError> {
        self.kinds.lock().unwrap().push(permit.event().kind());
        Ok(())
    }

    fn record_frame_capture_summary(
        &self,
        summary: &ExecutionFrameCaptureSummary,
    ) -> Result<(), ExecutionEventSinkError> {
        self.summary_attempts.fetch_add(1, Ordering::Relaxed);
        if self.fail_summary.load(Ordering::Relaxed) {
            return Err(ExecutionEventSinkError::new("capture sink failure"));
        }
        self.summaries.lock().unwrap().push(summary.clone());
        Ok(())
    }
}

struct JournalHarness {
    fixture: Fixture,
    sequence: Arc<AdmittedSequenceResources<TestRuntime>>,
    session: Arc<SequenceSession<TestRuntime>>,
    batch: ExecutionBatchParticipants<TestRuntime>,
    step: Option<Arc<StepResourceLease<TestRuntime>>>,
    lane: Arc<ExecutionLane<TestRuntime>>,
    reaper: Arc<CompletionReaper<TestRuntime>>,
    journal: Option<VNextExecutionJournal>,
    sink: Arc<CaptureSink>,
}

impl JournalHarness {
    fn new(limit: u32, mode: DeviceTimingMode) -> Self {
        let (fixture, sequence, session, batch, step) = setup();
        let sink = CaptureSink::bounded(limit, mode);
        let active = Arc::new(TrustedActiveSequenceBinding::from_session(&session).unwrap());
        let journal = VNextExecutionJournal::open(
            sink.clone(),
            &fixture.plan,
            active,
            ExecutorRequestOrigin::Product,
        )
        .unwrap();
        Self {
            fixture,
            sequence,
            session,
            batch,
            lane: Arc::clone(step.execution_lane()),
            step: Some(step),
            reaper: CompletionReaper::new(),
            journal: Some(journal),
            sink,
        }
    }

    fn submit(&mut self) -> CompletionHandle<TestRuntime> {
        let step = self.step.get_or_insert_with(|| {
            begin_single_participant_step_on_lane_with_bucket(
                &self.batch,
                &self.lane,
                self.fixture.reusable_execution_bucket.as_ref(),
            )
        });
        let wave = prepare_wave(&self.fixture.plan_resources, &self.fixture.plan, step);
        let active = wave_active_bindings(&wave, &self.session);
        let providers = self
            .fixture
            .registry
            .bind_plan(&self.fixture.resolved)
            .unwrap();
        let identity = OperationDispatch::bind_submission_wave_identity(
            &self.fixture.resolved,
            active.iter(),
            &wave,
            &self.lane,
        )
        .unwrap();
        let journal = self.journal.as_mut().unwrap();
        let instrumentation =
            VNextWaveInstrumentation::new(self.sink.mode, journal.captures_device_wave());
        let completion = OperationDispatch::encode_and_submit_wave_with_inputs_and_policy(
            providers.providers(),
            &self.fixture.resolved,
            &identity,
            active.iter(),
            instrumentation.timing_mode,
            &[],
            instrumentation.execution_policy,
            wave,
            &self.lane,
            &self.reaper,
        )
        .unwrap();
        journal.submitted(completion.receipt()).unwrap();
        completion
    }

    fn complete_wave(&mut self, completion: CompletionHandle<TestRuntime>) {
        let CompletionObservation::Terminal(receipt) = completion.wait().unwrap() else {
            panic!("CPU fixture must return a terminal completion");
        };
        self.journal.as_mut().unwrap().completed(&receipt).unwrap();
        self.step.take().unwrap().try_retire_normal().unwrap();
    }

    fn complete_request(&mut self) -> Result<(), ExecutionEventSinkError> {
        let terminal = self.session.try_complete().unwrap();
        self.journal
            .as_mut()
            .unwrap()
            .complete_sequence(&terminal, 1, 2)
    }

    fn close(self) {
        let Self {
            fixture,
            sequence,
            session,
            batch,
            step,
            lane,
            reaper,
            journal,
            ..
        } = self;
        drop(journal);
        if let Some(step) = step {
            step.try_retire_normal().unwrap();
        }
        let _ = session.request_cancel();
        let _ = session.try_abort();
        assert_eq!(reaper.retained_count(), 0);
        assert_eq!(lane.in_flight_count(), 0);
        drop(batch);
        drop(session);
        drop(sequence);
        drop(lane);
        drop(reaper);
        drop(fixture.registry);
        drop(fixture.impostor_registry);
        drop(fixture.runtime);
        assert!(matches!(
            PlanRuntimeResources::close(fixture.plan_resources),
            Ok(PlanRuntimeCloseOutcome::Closed(_))
        ));
    }
}

#[test]
fn bounded_frame_capture_preserves_configured_execution_controls() {
    for configured in [
        DeviceTimingMode::Off,
        DeviceTimingMode::Completion,
        DeviceTimingMode::Replay,
        DeviceTimingMode::Kernel,
        DeviceTimingMode::Verification,
    ] {
        let captured = VNextWaveInstrumentation::new(configured, true);
        let suppressed = VNextWaveInstrumentation::new(configured, false);
        assert_eq!(captured.execution_policy, suppressed.execution_policy);
        assert_eq!(
            captured.direct_reusable_execution_allowed,
            suppressed.direct_reusable_execution_allowed
        );
        assert_eq!(captured.timing_mode, configured);
        assert_eq!(suppressed.timing_mode, DeviceTimingMode::Off);
        if matches!(
            configured,
            DeviceTimingMode::Kernel | DeviceTimingMode::Verification
        ) {
            assert!(!suppressed.direct_reusable_execution_allowed);
        }
    }
    let verification = VNextWaveInstrumentation::new(DeviceTimingMode::Verification, false);
    assert_eq!(
        verification.execution_policy,
        SubmissionExecutionPolicy::determinism_eager(0)
    );
}

#[test]
fn bounded_frame_capture_keeps_real_verification_submissions_eager_after_cap() {
    let mut harness = JournalHarness::new(1, DeviceTimingMode::Verification);
    for _ in 0..3 {
        let completion = harness.submit();
        harness.complete_wave(completion);
    }
    let journal = harness.journal.as_ref().unwrap();
    assert_eq!(journal.completed_frames, 3);
    assert_eq!(journal.emitter.cursor().completed_frames(), 1);
    assert!(!journal.captures_device_wave());
    assert!(harness
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .submitted_compute_path_requirements
        .iter()
        .all(|path| *path == DeviceComputePathRequirement::EagerOnly));
    harness.complete_request().unwrap();
    let sink = Arc::clone(&harness.sink);
    harness.close();
    let summaries = sink.summaries.lock().unwrap();
    assert_eq!(summaries.len(), 1);
    let summary = &summaries[0];
    assert_eq!(summary.completed_frames, 3);
    assert_eq!(summary.captured_completed_frames, 1);
    assert!(summary.request_succeeded && summary.journal_terminal_observed);
    assert!(!summary.pending_submission);
    assert_eq!(
        sink.kinds.lock().unwrap().last(),
        Some(&ExecutionEventKind::RequestCompleted)
    );
}

#[test]
fn bounded_frame_capture_drop_reports_pending_work_without_retiring_it() {
    let mut harness = JournalHarness::new(1, DeviceTimingMode::Off);
    let completion = harness.submit();
    drop(harness.journal.take());
    {
        let summaries = harness.sink.summaries.lock().unwrap();
        assert_eq!(summaries.len(), 1);
        let summary = &summaries[0];
        assert_eq!(summary.completed_frames, 0);
        assert_eq!(summary.captured_completed_frames, 0);
        assert!(!summary.request_succeeded && !summary.journal_terminal_observed);
        assert!(summary.pending_submission);
    }
    assert_eq!(harness.lane.in_flight_count(), 1);
    assert!(Arc::clone(harness.step.as_ref().unwrap())
        .try_retire_normal()
        .is_err());
    assert!(matches!(
        completion.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    harness.close();
}

#[test]
fn bounded_frame_capture_cancelled_prefix_is_partial_and_sink_failure_is_not_retried() {
    let mut harness = JournalHarness::new(2, DeviceTimingMode::Off);
    let completion = harness.submit();
    harness.complete_wave(completion);
    harness.session.request_cancel().unwrap();
    harness.session.try_abort().unwrap();
    let sink = Arc::clone(&harness.sink);
    harness.close();
    let summaries = sink.summaries.lock().unwrap();
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].completed_frames, 1);
    assert!(!summaries[0].request_succeeded);
    assert!(!summaries[0].journal_terminal_observed);
    drop(summaries);

    let mut harness = JournalHarness::new(1, DeviceTimingMode::Off);
    let completion = harness.submit();
    harness.complete_wave(completion);
    harness.sink.fail_summary.store(true, Ordering::Relaxed);
    assert!(harness.complete_request().is_err());
    let sink = Arc::clone(&harness.sink);
    harness.close();
    assert_eq!(sink.summary_attempts.load(Ordering::Relaxed), 1);
    assert!(sink.summaries.lock().unwrap().is_empty());
}

#[test]
fn bounded_frame_summary_rejects_identity_counter_and_terminal_forgery() {
    let sink = CaptureSink::bounded(2, DeviceTimingMode::Off);
    let run: RunId = id("run.capture");
    let request: RequestIdentity = id("request.capture");
    let mut emitter =
        ExecutionEventEmitter::from_shared(sink.clone(), run.clone(), request.clone());
    let valid = ExecutionFrameCaptureSummary {
        run_id: run,
        request_id: request,
        limit: NonZeroU32::new(2).unwrap(),
        completed_frames: 0,
        captured_completed_frames: 0,
        request_succeeded: false,
        journal_terminal_observed: false,
        pending_submission: false,
    };
    for invalid in [
        ExecutionFrameCaptureSummary {
            run_id: id("run.other"),
            ..valid.clone()
        },
        ExecutionFrameCaptureSummary {
            request_id: id("request.other"),
            ..valid.clone()
        },
        ExecutionFrameCaptureSummary {
            limit: NonZeroU32::new(1).unwrap(),
            ..valid.clone()
        },
        ExecutionFrameCaptureSummary {
            completed_frames: 1,
            ..valid.clone()
        },
        ExecutionFrameCaptureSummary {
            captured_completed_frames: 1,
            ..valid.clone()
        },
        ExecutionFrameCaptureSummary {
            request_succeeded: true,
            ..valid.clone()
        },
        ExecutionFrameCaptureSummary {
            journal_terminal_observed: true,
            ..valid.clone()
        },
    ] {
        assert!(emitter.record_frame_capture_summary(&invalid).is_err());
    }
    assert!(sink.summaries.lock().unwrap().is_empty());
    emitter.record_frame_capture_summary(&valid).unwrap();
    assert!(emitter.record_frame_capture_summary(&valid).is_err());
    assert_eq!(sink.summaries.lock().unwrap().len(), 1);
}

#[test]
fn bounded_frame_capture_shared_wave_keeps_new_participant_and_physical_scope() {
    let mut harness = JournalHarness::new(1, DeviceTimingMode::Off);
    let completion = harness.submit();
    harness.complete_wave(completion);
    let old = parking_lot::Mutex::new(harness.journal.take().unwrap());
    assert!(!captures_participant_wave([Some(&old)].into_iter()));
    let fresh_resources = logical_resources(
        &harness.fixture.plan_resources,
        "run.capture.new",
        "request.capture.new",
    );
    let fresh_session = fresh_resources.open_session().unwrap();
    let fresh_active =
        Arc::new(TrustedActiveSequenceBinding::from_session(&fresh_session).unwrap());
    let fresh = parking_lot::Mutex::new(
        VNextExecutionJournal::open(
            harness.sink.clone(),
            &harness.fixture.plan,
            fresh_active.clone(),
            ExecutorRequestOrigin::Product,
        )
        .unwrap(),
    );
    assert!(captures_participant_wave(
        [Some(&old), Some(&fresh)].into_iter()
    ));
    let batch = ExecutionBatchParticipants::new(vec![
        Arc::clone(&harness.session),
        Arc::clone(&fresh_session),
    ])
    .unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![one_token_span(), one_token_span()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let step = (0..4)
        .find_map(|_| {
            match batch
                .try_begin_step(request.clone(), &harness.lane)
                .unwrap()
            {
                StepResourceAdmissionDecision::Admitted(step) => Some(step),
                StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                    deferred.maintain().unwrap();
                    None
                }
                _ => panic!("shared capture fixture cannot admit its step"),
            }
        })
        .unwrap();
    let wave = prepare_wave(
        &harness.fixture.plan_resources,
        &harness.fixture.plan,
        &step,
    );
    let active = [
        TrustedActiveSequenceBinding::from_session(&harness.session).unwrap(),
        (*fresh_active).clone(),
    ];
    let providers = harness
        .fixture
        .registry
        .bind_plan(&harness.fixture.resolved)
        .unwrap();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &harness.fixture.resolved,
        active.iter(),
        &wave,
        &harness.lane,
    )
    .unwrap();
    let completion = OperationDispatch::encode_and_submit_wave_with_inputs_and_policy(
        providers.providers(),
        &harness.fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &[],
        SubmissionExecutionPolicy::adaptive(),
        wave,
        &harness.lane,
        &harness.reaper,
    )
    .unwrap();
    old.lock().submitted(completion.receipt()).unwrap();
    fresh.lock().submitted(completion.receipt()).unwrap();
    // Both logical requests retain the original physical wave's identity.
    let requests = completion
        .receipt()
        .participants()
        .iter()
        .map(|participant| participant.identity().parts().request_id.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        requests,
        BTreeSet::from([
            active[0].request_id().clone(),
            active[1].request_id().clone(),
        ])
    );
    let CompletionObservation::Terminal(receipt) = completion.wait().unwrap() else {
        panic!("shared capture fixture must complete");
    };
    old.lock().completed(&receipt).unwrap();
    fresh.lock().completed(&receipt).unwrap();
    step.try_retire_normal().unwrap();
    assert!(!captures_participant_wave(
        [Some(&old), Some(&fresh)].into_iter()
    ));
    assert_eq!(old.lock().completed_frames, 2);
    assert_eq!(old.lock().emitter.cursor().completed_frames(), 1);
    assert_eq!(fresh.lock().completed_frames, 1);
    assert_eq!(fresh.lock().emitter.cursor().completed_frames(), 1);
    fresh
        .lock()
        .complete_sequence(&fresh_session.try_complete().unwrap(), 1, 1)
        .unwrap();
    harness.journal = Some(old.into_inner());
    harness.complete_request().unwrap();
    let sink = Arc::clone(&harness.sink);
    drop(fresh);
    drop(providers);
    drop(active);
    drop(fresh_active);
    drop(batch);
    drop(fresh_session);
    drop(fresh_resources);
    harness.close();
    let summaries = sink.summaries.lock().unwrap();
    assert_eq!(summaries.len(), 2);
    assert!(summaries.iter().all(|summary| summary.request_succeeded));
    assert_ne!(summaries[0].request_id, summaries[1].request_id);
    assert_eq!(
        summaries
            .iter()
            .map(|summary| summary.completed_frames)
            .sum::<u64>(),
        3
    );
}
