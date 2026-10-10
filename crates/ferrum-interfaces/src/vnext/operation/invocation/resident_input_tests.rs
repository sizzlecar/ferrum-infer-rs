//! Real admission, dispatch, fence completion and retirement exercise the
//! lane-owned input proof. TestRuntime counts actual encode_upload calls; it
//! does not execute CUDA arithmetic (covered by selection's device fixture).
use super::vnext_device_operation_wave_contract::{determinism_restore, prepare_determinism_wave};
use super::*;

pub(super) struct InputFixture {
    pub(super) fixture: Fixture,
    sequences: Vec<Arc<AdmittedSequenceResources<TestRuntime>>>,
    sessions: Vec<Arc<SequenceSession<TestRuntime>>>,
    batch: ExecutionBatchParticipants<TestRuntime>,
    pub(super) lane: Arc<ExecutionLane<TestRuntime>>,
    pub(super) reaper: Arc<CompletionReaper<TestRuntime>>,
    step: Option<Arc<StepResourceLease<TestRuntime>>>,
    pub(super) width: usize,
    pub(super) last_slot: Option<LaneStableArenaSlotIdentity>,
    last_frames: Vec<ExecutionFrameId>,
}

impl InputFixture {
    pub(super) fn new(width: usize, stable_slot: bool) -> Self {
        Self::with_determinism_output_retention(width, stable_slot, false)
    }

    fn with_determinism_output_retention(
        width: usize,
        stable_slot: bool,
        retain_determinism_outputs: bool,
    ) -> Self {
        let fixture = if stable_slot {
            fixture_with_fixed_inputs_and_bucket(
                ReusableExecutionBucketSpec::new(
                    ReusableExecutionClassId::new("execution.resident-input").unwrap(),
                    ReusableExecutionCapacity::new(width as u32, width as u64, 1).unwrap(),
                )
                .unwrap(),
                retain_determinism_outputs,
            )
        } else {
            fixture()
        };
        let sequences = (0..width)
            .map(|p| {
                logical_resources(
                    &fixture.plan_resources,
                    &format!("run.resident-input.{p}"),
                    &format!("request.resident-input.{p}"),
                )
            })
            .collect::<Vec<_>>();
        let sessions = sequences
            .iter()
            .map(|sequence| sequence.open_session().unwrap())
            .collect::<Vec<_>>();
        let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
        let lane = fixture.plan_resources.create_execution_lane().unwrap();
        Self {
            fixture,
            sequences,
            sessions,
            batch,
            lane,
            reaper: CompletionReaper::new(),
            step: None,
            width,
            last_slot: None,
            last_frames: Vec::new(),
        }
    }

    fn prepare(&mut self) -> PreparedStepSubmissionWave<TestRuntime> {
        self.prepare_for(false)
    }

    fn prepare_for(&mut self, determinism: bool) -> PreparedStepSubmissionWave<TestRuntime> {
        assert!(self.step.is_none());
        let request = StepResourceAdmissionRequest::new(
            self.batch
                .bind_work_shape(vec![one_token_span(); self.width])
                .unwrap(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let request = match &self.fixture.reusable_execution_bucket {
            Some(bucket) => request.with_reusable_execution_bucket(bucket.bucket_id().clone()),
            None => request,
        };
        for attempt in 0..=3 {
            match self
                .batch
                .try_begin_step(request.clone(), &self.lane)
                .unwrap()
            {
                StepResourceAdmissionDecision::Admitted(step) => {
                    self.last_slot = step.claimed_backing().lane_stable_slot_identity();
                    self.last_frames = step
                        .participant_frames()
                        .map(|frame| frame.frame_id())
                        .collect();
                    let wave = if determinism {
                        prepare_determinism_wave(
                            &self.fixture.plan_resources,
                            &self.fixture.plan,
                            &step,
                        )
                    } else {
                        prepare_wave(&self.fixture.plan_resources, &self.fixture.plan, &step)
                    };
                    self.step = Some(step);
                    return wave;
                }
                StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                    deferred.maintain().unwrap();
                }
                _ => panic!("real input fixture admission cannot progress"),
            }
        }
        unreachable!("bounded admission either returns or reports its failure")
    }

    pub(super) fn dispatch(
        &mut self,
        inputs: &[SubmissionWaveInputUpload],
    ) -> Result<CompletionHandle<TestRuntime>, SubmissionWaveDispatchError<TestRuntime>> {
        let wave = self.prepare();
        self.dispatch_prepared(wave, inputs)
    }

    fn dispatch_prepared(
        &self,
        wave: PreparedStepSubmissionWave<TestRuntime>,
        inputs: &[SubmissionWaveInputUpload],
    ) -> Result<CompletionHandle<TestRuntime>, SubmissionWaveDispatchError<TestRuntime>> {
        let active = self
            .batch
            .sessions()
            .iter()
            .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
            .collect::<Vec<_>>();
        let identity = OperationDispatch::bind_submission_wave_identity(
            &self.fixture.resolved,
            active.iter(),
            &wave,
            &self.lane,
        )
        .unwrap();
        let providers = self
            .fixture
            .registry
            .bind_plan(&self.fixture.resolved)
            .unwrap();
        OperationDispatch::encode_and_submit_wave_with_inputs(
            providers.providers(),
            &self.fixture.resolved,
            &identity,
            active.iter(),
            DeviceTimingMode::Off,
            inputs,
            wave,
            &self.lane,
            &self.reaper,
        )
    }

    pub(super) fn finish(&mut self, handle: CompletionHandle<TestRuntime>) {
        let publication = handle.take_input_residency_publication();
        let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
            panic!("fixture fence did not reach terminal state")
        };
        assert!(matches!(
            receipt.disposition(),
            OperationCompletionDisposition::Succeeded
        ));
        if let Some(publication) = publication {
            publication
                .publish(&receipt, self.step.as_ref().unwrap())
                .unwrap();
        }
        drop(handle);
        self.retire();
    }

    fn retire(&mut self) {
        self.step.take().unwrap().try_retire_normal().unwrap();
    }

    pub(super) fn run(&mut self, inputs: &[SubmissionWaveInputUpload]) -> Vec<Vec<u8>> {
        let before = self
            .fixture
            .runtime_trace
            .lock()
            .unwrap()
            .uploaded_payloads
            .len();
        let handle = self.dispatch(inputs).unwrap();
        self.finish(handle);
        self.fixture.runtime_trace.lock().unwrap().uploaded_payloads[before..].to_vec()
    }

    pub(super) fn close(self) {
        assert!(self.step.is_none());
        drop(self.batch);
        for session in &self.sessions {
            session.try_complete().unwrap();
        }
        drop(self.sessions);
        drop(self.sequences);
        drop(self.reaper);
        drop(self.lane);
        drop(self.fixture.registry);
        drop(self.fixture.impostor_registry);
        drop(self.fixture.runtime);
        assert!(matches!(
            PlanRuntimeResources::close(self.fixture.plan_resources),
            Ok(PlanRuntimeCloseOutcome::Closed(_))
        ));
    }
}

fn input(p: usize, offset: u64, values: &[f32], cache: bool) -> SubmissionWaveInputUpload {
    let upload = SubmissionWaveInputUpload::new(
        id("node.main"),
        p as u32,
        0,
        offset,
        HostTransferLayout::new(ElementType::F32, values.len() as u64).unwrap(),
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
    .unwrap();
    if cache {
        upload.request_stable_residency()
    } else {
        upload
    }
}

pub(super) fn neutral_inputs(width: usize, cache: bool) -> Vec<SubmissionWaveInputUpload> {
    // Two nonadjacent ranges per physical participant exercise both inputs
    // without letting upload coalescing hide the number of emitted commands.
    (0..width)
        .map(|p| input(p, 0, &[0.0, 0.0], cache))
        .chain((0..width).map(|p| input(p, 12, &[1.0], cache)))
        .collect()
}

#[test]
fn resident_inputs_first_write_then_fresh_frame_hits_without_reencoding_uploads() {
    let mut fixture = InputFixture::new(2, true);
    let inputs = neutral_inputs(2, true);
    assert_eq!(fixture.run(&inputs).len(), 4);
    let slot = fixture.last_slot.clone().expect("real reusable Step slot");
    let first_frames = fixture.last_frames.clone();
    let provider_before = fixture.fixture.provider_trace.lock().unwrap().encode_calls;
    let submits_before = fixture.fixture.runtime_trace.lock().unwrap().submit_calls;
    assert!(fixture.run(&inputs).is_empty());
    assert_eq!(fixture.last_slot.as_ref(), Some(&slot));
    assert!(fixture
        .last_frames
        .iter()
        .zip(&first_frames)
        .all(|(now, before)| now != before));
    assert!(fixture.fixture.provider_trace.lock().unwrap().encode_calls > provider_before);
    assert_eq!(
        fixture.fixture.runtime_trace.lock().unwrap().submit_calls,
        submits_before + 1
    );
    assert_eq!(fixture.reaper.retained_count(), 0);
    assert_eq!(fixture.lane.in_flight_count(), 0);
    fixture.close();
}

#[test]
fn resident_inputs_active_overwrite_and_physical_participant_reorder_rewrite_neutral() {
    let mut fixture = InputFixture::new(2, true);
    let neutral = neutral_inputs(2, true);
    assert_eq!(fixture.run(&neutral).len(), 4);
    let mixed = [
        input(0, 0, &[0.0, 1.0], false),
        input(1, 0, &[0.0, 0.0], true),
        input(0, 12, &[2.0], false),
        input(1, 12, &[1.0], true),
    ];
    assert_eq!(fixture.run(&mixed).len(), 2);
    let reordered = [
        input(0, 0, &[0.0, 0.0], true),
        input(1, 0, &[0.0, 1.0], false),
        input(0, 12, &[1.0], true),
        input(1, 12, &[2.0], false),
    ];
    assert_eq!(fixture.run(&reordered).len(), 4);
    assert_eq!(fixture.run(&neutral).len(), 2);
    assert!(fixture.run(&neutral).is_empty());
    fixture.close();
}

#[test]
fn resident_inputs_without_stable_step_slot_keep_rewriting() {
    let mut fixture = InputFixture::new(1, false);
    let inputs = neutral_inputs(1, true);
    assert_eq!(fixture.run(&inputs).len(), 2);
    assert!(fixture.last_slot.is_none());
    assert_eq!(fixture.run(&inputs).len(), 2);
    fixture.close();
}

#[test]
fn resident_inputs_changed_window_and_complete_lane_slot_do_not_inherit_proofs() {
    let mut fixture = InputFixture::new(1, true);
    let original = [input(0, 0, &[0.0], true)];
    assert_eq!(fixture.run(&original).len(), 1);
    assert!(fixture.run(&original).is_empty());
    let shifted = [input(0, 4, &[0.0], true)];
    assert_eq!(fixture.run(&shifted).len(), 1);
    assert!(fixture.run(&shifted).is_empty());
    let prior_slot = fixture.last_slot.clone().unwrap();
    fixture.lane = fixture
        .fixture
        .plan_resources
        .create_execution_lane()
        .unwrap();
    assert_eq!(fixture.run(&shifted).len(), 1);
    assert_ne!(fixture.last_slot.as_ref(), Some(&prior_slot));
    assert!(fixture.run(&shifted).is_empty());
    fixture.lane.clear_input_residency().unwrap();
    assert_eq!(
        fixture.run(&shifted).len(),
        1,
        "startup/reset clears successful contents"
    );
    assert!(fixture.run(&shifted).is_empty());
    fixture.close();
}

#[test]
fn resident_inputs_changed_participant_layout_uses_current_physical_windows() {
    let mut fixture = InputFixture::new(2, true);
    let two = neutral_inputs(2, true);
    assert_eq!(fixture.run(&two).len(), 4);
    let previous = fixture.last_slot.clone().unwrap();
    fixture.batch =
        ExecutionBatchParticipants::new(vec![Arc::clone(&fixture.sessions[0])]).unwrap();
    fixture.width = 1;
    let one = neutral_inputs(1, true);
    let copies = fixture.run(&one);
    if fixture.last_slot.as_ref() != Some(&previous) {
        assert_eq!(
            copies.len(),
            2,
            "a different actual Step layout must rewrite"
        );
    }
    assert!(fixture.run(&one).is_empty());
    fixture.close();
}

#[test]
fn resident_inputs_success_without_publication_and_stale_step_ticket_do_not_seal() {
    let mut fixture = InputFixture::new(1, true);
    let inputs = neutral_inputs(1, true);
    let handle = fixture.dispatch(&inputs).unwrap();
    let ticket = handle
        .take_input_residency_publication()
        .expect("cache request has a ticket");
    assert!(handle.take_input_residency_publication().is_none());
    let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
        panic!("fixture fence did not complete")
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    drop(handle);
    fixture.retire();
    let wave = fixture.prepare();
    assert!(
        !ticket
            .publish(&receipt, fixture.step.as_ref().unwrap())
            .unwrap(),
        "old completion cannot publish against the next frame even when the slot is reused"
    );
    let before = fixture
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .uploaded_payloads
        .len();
    let handle = fixture.dispatch_prepared(wave, &inputs).unwrap();
    // A terminal success is necessary but insufficient: losing the take-once
    // ticket must leave no claim that bytes are resident.
    drop(handle.take_input_residency_publication());
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    drop(handle);
    fixture.retire();
    assert_eq!(
        fixture
            .fixture
            .runtime_trace
            .lock()
            .unwrap()
            .uploaded_payloads
            .len()
            - before,
        2
    );
    assert_eq!(fixture.run(&inputs).len(), 2);
    assert!(fixture.run(&inputs).is_empty());
    fixture.close();
}

#[test]
fn resident_inputs_hit_still_checks_fresh_runtime_descriptor_and_plan_input_type() {
    let mut fixture = InputFixture::new(1, true);
    let inputs = neutral_inputs(1, true);
    assert_eq!(fixture.run(&inputs).len(), 2);
    let submitted = fixture.fixture.runtime_trace.lock().unwrap().submit_calls;
    fixture
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .tamper_buffer_descriptor = true;
    let result = fixture.dispatch(&inputs);
    fixture
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .tamper_buffer_descriptor = false;
    assert!(matches!(
        result,
        Err(SubmissionWaveDispatchError::Contract(_))
    ));
    drop(result);
    assert_eq!(
        fixture.fixture.runtime_trace.lock().unwrap().submit_calls,
        submitted
    );
    fixture.retire();
    assert_eq!(
        fixture.run(&inputs).len(),
        2,
        "descriptor failure must invalidate even before the input transaction begins"
    );
    assert!(fixture.run(&inputs).is_empty());
    let submitted = fixture.fixture.runtime_trace.lock().unwrap().submit_calls;

    let wrong_type = SubmissionWaveInputUpload::new(
        id("node.main"),
        0,
        0,
        0,
        HostTransferLayout::new(ElementType::U32, 2).unwrap(),
        vec![0; 8],
    )
    .unwrap()
    .request_stable_residency();
    let result = fixture.dispatch(&[wrong_type]);
    assert!(matches!(
        result,
        Err(SubmissionWaveDispatchError::Contract(_))
    ));
    drop(result);
    assert_eq!(
        fixture.fixture.runtime_trace.lock().unwrap().submit_calls,
        submitted
    );
    fixture.retire();
    assert_eq!(
        fixture.run(&inputs).len(),
        2,
        "failed attempts must not leave a residency proof"
    );
    let submitted = fixture.fixture.runtime_trace.lock().unwrap().submit_calls;
    let outside = input(0, 16, &[1.0], true);
    let result = fixture.dispatch(&[outside]);
    assert!(matches!(
        result,
        Err(SubmissionWaveDispatchError::Contract(_))
    ));
    drop(result);
    assert_eq!(
        fixture.fixture.runtime_trace.lock().unwrap().submit_calls,
        submitted
    );
    fixture.retire();
    fixture.close();
}

#[test]
fn resident_inputs_partial_encoding_and_reservation_drop_do_not_publish() {
    let mut fixture = InputFixture::new(1, true);
    let inputs = neutral_inputs(1, true);
    // The first run is a legal upload run; the next run names an absent node.
    // Failure drops the actual CompletionReservation after partial encoding.
    let invalid = SubmissionWaveInputUpload::new(
        id("node.absent"),
        0,
        0,
        0,
        HostTransferLayout::new(ElementType::F32, 1).unwrap(),
        1.0_f32.to_le_bytes().to_vec(),
    )
    .unwrap()
    .request_stable_residency();
    let result = fixture.dispatch(&[inputs[0].clone(), invalid]);
    assert!(matches!(
        result,
        Err(SubmissionWaveDispatchError::Contract(_))
    ));
    drop(result);
    assert_eq!(
        fixture
            .fixture
            .runtime_trace
            .lock()
            .unwrap()
            .uploaded_payloads
            .len(),
        1
    );
    assert_eq!(
        fixture.fixture.runtime_trace.lock().unwrap().submit_calls,
        0
    );
    assert_eq!(fixture.reaper.retained_count(), 0);
    fixture.retire();
    assert_eq!(fixture.run(&inputs).len(), 2);
    assert!(fixture.run(&inputs).is_empty());
    fixture.close();
}

#[test]
fn resident_inputs_definitely_not_submitted_retry_revalidates_and_rewrites() {
    let mut fixture = InputFixture::new(1, true);
    let inputs = neutral_inputs(1, true);
    assert_eq!(fixture.run(&inputs).len(), 2);
    fixture
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .submit_behavior = SubmitBehavior::DefinitelyNotSubmitted;
    let active = [input(0, 0, &[0.0, 1.0], false), input(0, 12, &[2.0], false)];
    let retry = match fixture.dispatch(&active) {
        Err(SubmissionWaveDispatchError::DefinitelyNotSubmitted { retry, .. }) => retry,
        other => panic!("expected typed definitely-not-submitted retry: {other:?}"),
    };
    fixture
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .submit_behavior = SubmitBehavior::Success;
    let before = fixture
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .uploaded_payloads
        .len();
    let retry_wave = retry.retry().unwrap();
    let handle = fixture.dispatch_prepared(retry_wave, &inputs).unwrap();
    fixture.finish(handle);
    assert_eq!(
        fixture
            .fixture
            .runtime_trace
            .lock()
            .unwrap()
            .uploaded_payloads
            .len()
            - before,
        2
    );
    assert!(fixture.run(&inputs).is_empty());
    fixture.close();
}

#[test]
fn resident_inputs_dropped_observer_and_failed_terminal_cannot_seal_pending_writes() {
    let mut fixture = InputFixture::new(1, true);
    let inputs = neutral_inputs(1, true);
    fixture.fixture.runtime_trace.lock().unwrap().fence_behavior = FenceBehavior::Pending;
    let handle = fixture.dispatch(&inputs).unwrap();
    assert!(matches!(
        handle.poll().unwrap(),
        CompletionObservation::Pending
    ));
    let slot = handle.slot_id();
    drop(handle);
    // A dropped observer must neither release the actual pending Step nor
    // publish success; the reaper retains ownership until an actual terminal.
    assert_eq!(fixture.reaper.retained_count(), 1);
    assert_eq!(fixture.lane.in_flight_count(), 1);
    fixture.fixture.runtime_trace.lock().unwrap().fence_behavior =
        FenceBehavior::FailedButQuiescent;
    let CompletionObservation::Terminal(receipt) =
        fixture.reaper.wait_slot_for_recovery(slot).unwrap()
    else {
        panic!("failed fence did not settle")
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::FailedButQuiescent(_)
    ));
    fixture.retire();
    fixture.fixture.runtime_trace.lock().unwrap().fence_behavior = FenceBehavior::Succeeded;
    assert_eq!(fixture.run(&inputs).len(), 2);
    assert!(fixture.run(&inputs).is_empty());
    fixture.close();
}

#[test]
fn resident_inputs_determinism_restore_invalidates_external_input_on_same_slot() {
    // Restore/readback needs the planner's exact terminal witnesses; ordinary
    // input tests and the paired benchmark retain their original layout.
    let mut fixture = InputFixture::with_determinism_output_retention(1, true, true);
    let neutral = neutral_inputs(1, true);
    assert_eq!(fixture.run(&neutral).len(), 2);
    assert!(fixture.run(&neutral).is_empty());
    let slot = fixture.last_slot.clone().unwrap();
    let wave = fixture.prepare_for(true);
    assert_eq!(fixture.last_slot.as_ref(), Some(&slot));
    let active = fixture
        .batch
        .sessions()
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.fixture.resolved,
        active.iter(),
        &wave,
        &fixture.lane,
    )
    .unwrap();
    let providers = fixture
        .fixture
        .registry
        .bind_plan(&fixture.fixture.resolved)
        .unwrap();
    let restore = determinism_restore(
        &fixture.fixture,
        providers.providers(),
        &identity,
        &active,
        &wave,
        0x3f,
    );
    assert!(restore
        .initializations()
        .iter()
        .any(|initialization| matches!(
            initialization.kind(),
            ExecutionDeterminismInitializationKind::ExternalInput { .. }
        )));
    let before = fixture
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .uploaded_payloads
        .len();
    let handle = OperationDispatch::encode_and_submit_determinism_eager_wave(
        providers.providers(),
        &fixture.fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &restore,
        0xa5,
        wave,
        &fixture.lane,
        &fixture.reaper,
    )
    .unwrap();
    let CompletionReadbackCollectionObservation::Terminal(receipt) =
        handle.wait_with_determinism_readback().unwrap()
    else {
        panic!("determinism restore did not finish")
    };
    assert!(matches!(
        receipt.completion().disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    assert!(fixture
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .uploaded_payloads[before..]
        .iter()
        .any(|bytes| bytes.as_slice() == [0x3f; 16]));
    drop(receipt);
    drop(handle);
    drop(providers);
    drop(active);
    fixture.retire();
    assert_eq!(
        fixture.run(&neutral).len(),
        2,
        "successful restore overwrote the old neutral bytes and cannot inherit their proof"
    );
    assert_eq!(fixture.last_slot.as_ref(), Some(&slot));
    assert!(fixture.run(&neutral).is_empty());
    fixture.close();
}
