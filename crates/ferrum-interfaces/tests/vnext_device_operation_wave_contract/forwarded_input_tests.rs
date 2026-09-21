use super::*;

#[path = "forwarded_input_tests/cold_child.rs"]
mod cold_child;

fn token_source(row: u32) -> CompletionReadbackRequest {
    CompletionReadbackRequest::new(
        id("node.tail"),
        row,
        id("resource.output"),
        0,
        HostTransferLayout::new(ElementType::U32, 1).unwrap(),
    )
    .unwrap()
}

fn token_sources(rows: usize) -> CompletionReadbackBatchRequest {
    CompletionReadbackBatchRequest::new((0..rows).map(|row| token_source(row as u32)).collect())
        .unwrap()
}

fn upload(row: u32, value: u32) -> SubmissionWaveInputUpload {
    SubmissionWaveInputUpload::new(
        id("node.main"),
        row,
        0,
        0,
        HostTransferLayout::new(ElementType::U32, 1).unwrap(),
        value.to_le_bytes().to_vec(),
    )
    .unwrap()
}

struct TokenFixture {
    fixture: Fixture,
    resources: Vec<Arc<AdmittedSequenceResources<TestRuntime>>>,
    sessions: Vec<Arc<SequenceSession<TestRuntime>>>,
    batch: ExecutionBatchParticipants<TestRuntime>,
    lane: Arc<ExecutionLane<TestRuntime>>,
    reaper: Arc<CompletionReaper<TestRuntime>>,
    parent: Arc<StepResourceLease<TestRuntime>>,
    parent_handle: CompletionHandle<TestRuntime>,
    predecessor: Arc<SubmittedWavePredecessor<TestRuntime>>,
    parent_fence: u64,
}

impl TokenFixture {
    fn new(tokens: &[u32]) -> Self {
        let fixture = fixture_tokens();
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_readback_enabled = true;
        let lane = fixture.plan_resources.create_execution_lane().unwrap();
        lane.configure_submission_readback_staging(tokens.len() as u64 * 8)
            .unwrap();
        let reaper = CompletionReaper::new();
        let resources = tokens
            .iter()
            .enumerate()
            .map(|(row, _)| {
                logical_resources(
                    &fixture.plan_resources,
                    &format!("run.forward.{row}"),
                    &format!("request.forward.{row}"),
                )
            })
            .collect::<Vec<_>>();
        let sessions = resources
            .iter()
            .map(|resources| resources.open_session().unwrap())
            .collect::<Vec<_>>();
        let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
        let work = tokens
            .iter()
            .map(|token| TokenSpanWork::from_token_ids_with_fit(&[*token], 0..1, 4).unwrap())
            .collect();
        let request = StepResourceAdmissionRequest::new(
            batch.bind_work_shape(work).unwrap(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let mut parent = None;
        for _ in 0..4 {
            match batch.try_begin_step(request.clone(), &lane).unwrap() {
                StepResourceAdmissionDecision::Admitted(step) => {
                    parent = Some(step);
                    break;
                }
                StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                    deferred.maintain().unwrap();
                }
                _ => panic!("token parent admission did not converge"),
            }
        }
        let parent = parent.unwrap();
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &parent)
            .with_submission_readbacks(token_sources(tokens.len()))
            .unwrap();
        let inputs = tokens
            .iter()
            .enumerate()
            .map(|(row, token)| upload(row as u32, *token))
            .collect::<Vec<_>>();
        let parent_handle = submit(&fixture, &sessions, &lane, &reaper, wave, &inputs).unwrap();
        let parent_fence = fixture.runtime_trace.lock().unwrap().next_fence;
        let predecessor = Arc::new(parent_handle.take_submitted_predecessor().unwrap());
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .fence_behaviors
            .insert(parent_fence, FenceBehavior::Pending);
        Self {
            fixture,
            resources,
            sessions,
            batch,
            lane,
            reaper,
            parent,
            parent_handle,
            predecessor,
            parent_fence,
        }
    }

    fn child(&self) -> Arc<StepResourceLease<TestRuntime>> {
        let work = self
            .predecessor
            .bind_next_token_work(token_sources(self.sessions.len()))
            .unwrap();
        let request = StepResourceAdmissionRequest::new(
            work,
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        for _ in 0..4 {
            match self
                .batch
                .try_begin_successor_step(
                    request.clone(),
                    &self.lane,
                    Arc::clone(&self.predecessor),
                )
                .unwrap()
            {
                StepResourceAdmissionDecision::Admitted(step) => return step,
                StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                    deferred.maintain().unwrap();
                }
                _ => panic!("token child admission did not converge"),
            }
        }
        panic!("token child backing did not converge")
    }

    fn forwards(&self) -> Vec<SubmissionWaveInputForward<TestRuntime>> {
        (0..self.sessions.len())
            .map(|row| {
                self.predecessor
                    .forward_token(token_source(row as u32), id("node.main"), 0, 0)
                    .unwrap()
            })
            .collect()
    }

    fn finish_parent(&self, expected: &[u32]) {
        self.fixture
            .runtime_trace
            .lock()
            .unwrap()
            .fence_behaviors
            .insert(self.parent_fence, FenceBehavior::Succeeded);
        let observed = self
            .parent_handle
            .wait_with_readbacks(token_sources(self.sessions.len()))
            .unwrap();
        assert_tokens(observed, expected);
    }
}

fn submit(
    fixture: &Fixture,
    sessions: &[Arc<SequenceSession<TestRuntime>>],
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    wave: PreparedStepSubmissionWave<TestRuntime>,
    uploads: &[SubmissionWaveInputUpload],
) -> Result<CompletionHandle<TestRuntime>, SubmissionWaveDispatchError<TestRuntime>> {
    let active = sessions
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        lane,
    )
    .unwrap();
    OperationDispatch::encode_and_submit_wave_with_inputs(
        &providers,
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        uploads,
        wave,
        lane,
        reaper,
    )
}

fn assert_tokens(observation: CompletionReadbackBatchObservation, expected: &[u32]) {
    let CompletionReadbackBatchObservation::Terminal(receipt) = observation else {
        panic!("token readback is not terminal")
    };
    assert_eq!(receipt.dispositions().len(), expected.len());
    for (disposition, expected) in receipt.dispositions().iter().zip(expected) {
        let CompletionReadbackDisposition::Succeeded(output) = disposition else {
            panic!("token readback failed: {disposition:?}")
        };
        assert_eq!(output.bytes(), expected.to_le_bytes());
    }
}

#[test]
fn forwarded_device_tokens_execute_nonzero_rows_and_commit_in_parent_order() {
    let setup = TokenFixture::new(&[53, 127]);
    let child = setup.child();
    let wave = prepare_wave(&setup.fixture.plan_resources, &setup.fixture.plan, &child)
        .with_forwarded_inputs(setup.forwards())
        .unwrap();
    let child_handle = submit(
        &setup.fixture,
        &setup.sessions,
        &setup.lane,
        &setup.reaper,
        wave,
        &[],
    )
    .unwrap();
    // The mock executed both device waves, but the host has not committed the
    // parent. A successful child fence must not publish an early success seal.
    assert!(matches!(
        child_handle.poll().unwrap(),
        CompletionObservation::Pending
    ));
    {
        let trace = setup.fixture.runtime_trace.lock().unwrap();
        let phases = trace.submitted_command_phases.last().unwrap();
        assert_eq!(
            phases
                .iter()
                .filter(|phase| **phase == DeviceCommandPhase::DynamicBinding)
                .count(),
            2
        );
        assert_eq!(trace.readback_calls, 0);
    }
    setup.finish_parent(&[54, 128]);
    drop(setup.parent_handle);
    setup.parent.try_retire_normal().unwrap();
    assert_tokens(
        child_handle.wait_with_readbacks(token_sources(2)).unwrap(),
        &[55, 129],
    );
    drop(child_handle);
    child.try_retire_normal().unwrap();
    drop(setup.predecessor);
    drop(setup.batch);
    for session in &setup.sessions {
        session.try_complete().unwrap();
    }
    drop(setup.sessions);
    drop(setup.resources);
    drop(setup.lane);
    drop(setup.reaper);
    drop(setup.fixture.registry);
    drop(setup.fixture.impostor_registry);
    drop(setup.fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(setup.fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}

#[test]
fn forwarded_device_token_rejects_missing_duplicate_and_host_sources_before_submit() {
    for case in ["missing", "duplicate", "host"] {
        let setup = TokenFixture::new(&[53]);
        let child = setup.child();
        let mut wave = prepare_wave(&setup.fixture.plan_resources, &setup.fixture.plan, &child);
        if case != "missing" {
            let mut forwards = setup.forwards();
            if case == "duplicate" {
                forwards.extend(setup.forwards());
            }
            wave = wave.with_forwarded_inputs(forwards).unwrap();
        }
        let inputs = if case == "host" {
            vec![upload(0, 999)]
        } else {
            vec![]
        };
        let before = setup.fixture.runtime_trace.lock().unwrap().submit_calls;
        let error = submit(
            &setup.fixture,
            &setup.sessions,
            &setup.lane,
            &setup.reaper,
            wave,
            &inputs,
        )
        .unwrap_err();
        assert!(matches!(error, SubmissionWaveDispatchError::Contract(_)));
        assert_eq!(
            setup.fixture.runtime_trace.lock().unwrap().submit_calls,
            before
        );
        setup.finish_parent(&[54]);
        drop(setup.parent_handle);
        setup.parent.try_retire_normal().unwrap();
        // A rejected child was never submitted; abort its reserved frame.
        // This terminates the sequence rather than pretending it committed.
        child.try_abort().unwrap();
    }
}

#[test]
fn forwarded_token_source_rejects_input_wrong_type_and_out_of_extent() {
    let setup = TokenFixture::new(&[53]);
    for (node, resource, element_type, offset) in [
        ("node.main", "resource.input", ElementType::U32, 0),
        ("node.tail", "resource.output", ElementType::F32, 0),
        ("node.tail", "resource.output", ElementType::U32, 4),
    ] {
        let request = CompletionReadbackRequest::new(
            id(node),
            0,
            id(resource),
            offset,
            HostTransferLayout::new(element_type, 1).unwrap(),
        )
        .unwrap();
        assert!(setup
            .predecessor
            .forward_token(request, id("node.main"), 0, 0)
            .is_err());
    }
    setup.finish_parent(&[54]);
    drop(setup.parent_handle);
    setup.parent.try_retire_normal().unwrap();
}

#[test]
fn forwarded_child_drains_without_success_after_parent_failure_or_cancellation() {
    for cancel in [false, true] {
        let setup = TokenFixture::new(&[53, 127]);
        let child = setup.child();
        let wave = prepare_wave(&setup.fixture.plan_resources, &setup.fixture.plan, &child)
            .with_forwarded_inputs(setup.forwards())
            .unwrap();
        let child_handle = submit(
            &setup.fixture,
            &setup.sessions,
            &setup.lane,
            &setup.reaper,
            wave,
            &[],
        )
        .unwrap();
        let child_fence = setup.fixture.runtime_trace.lock().unwrap().next_fence;
        assert!(matches!(
            child_handle.poll().unwrap(),
            CompletionObservation::Pending
        ));
        if cancel {
            for session in &setup.sessions {
                session.request_cancel().unwrap();
            }
        }
        setup
            .fixture
            .runtime_trace
            .lock()
            .unwrap()
            .fence_behaviors
            .insert(
                setup.parent_fence,
                if cancel {
                    FenceBehavior::Succeeded
                } else {
                    FenceBehavior::FailedButQuiescent
                },
            );
        assert!(matches!(
            setup.parent_handle.wait().unwrap(),
            CompletionObservation::Terminal(_)
        ));
        drop(setup.parent_handle);
        setup.parent.try_abort().unwrap();
        let reads_before = setup.fixture.runtime_trace.lock().unwrap().readback_calls;
        let observed = child_handle.wait_with_readbacks(token_sources(2)).unwrap();
        let CompletionReadbackBatchObservation::Terminal(receipt) = observed else {
            panic!("failed parent child must drain to a terminal disposition");
        };
        assert!(matches!(
            receipt.completion().disposition(),
            OperationCompletionDisposition::ContractFailedButQuiescent(_)
        ));
        assert_eq!(receipt.dispositions().len(), 2);
        for (row, disposition) in receipt.dispositions().iter().enumerate() {
            let CompletionReadbackDisposition::NotAttempted(request) = disposition else {
                panic!("failed dependency must skip its exact readback: {disposition:?}");
            };
            assert_eq!(request, &token_source(row as u32));
        }
        {
            let trace = setup.fixture.runtime_trace.lock().unwrap();
            assert!(
                trace.waited_fences.contains(&child_fence),
                "child's own device work must be observed quiescent"
            );
            assert_eq!(
                trace.readback_calls, reads_before,
                "invalid child tokens may not be read or published"
            );
        }
        drop(child_handle);
        let failure = child
            .try_retire_normal()
            .expect_err("failed dependency may not issue a successful frontier");
        failure.into_step().try_abort().unwrap();
        drop(setup.predecessor);
        drop(setup.batch);
        for session in &setup.sessions {
            session.try_abort().unwrap();
        }
    }
}

#[test]
fn cancelling_one_forwarded_row_discards_only_that_rows_frontier() {
    let setup = TokenFixture::new(&[53, 127]);
    let child = setup.child();
    let wave = prepare_wave(&setup.fixture.plan_resources, &setup.fixture.plan, &child)
        .with_forwarded_inputs(setup.forwards())
        .unwrap();
    let child_handle = submit(
        &setup.fixture,
        &setup.sessions,
        &setup.lane,
        &setup.reaper,
        wave,
        &[],
    )
    .unwrap();
    let cancelled = setup.sessions[0].sequence_authority();
    setup.sessions[0].request_cancel().unwrap();
    assert!(matches!(
        child_handle.poll().unwrap(),
        CompletionObservation::Pending
    ));
    setup.finish_parent(&[54, 128]);
    drop(setup.parent_handle);
    let parent_retirement = setup.parent.try_retire_normal().unwrap();
    // Readback is the core device result. Publishing output to a cancelled
    // request remains the product runner's responsibility; only its peer gets
    // a committed logical frontier here.
    assert_tokens(
        child_handle.wait_with_readbacks(token_sources(2)).unwrap(),
        &[55, 129],
    );
    drop(child_handle);
    let child_retirement = child.try_retire_normal().unwrap();
    for receipt in [parent_retirement, child_retirement] {
        assert_eq!(receipt.participants().len(), 2);
        for participant in receipt.participants() {
            assert_eq!(
                participant.disposition(),
                if participant.assignment().sequence_authority() == cancelled {
                    StepParticipantRetirementDisposition::DiscardedCancelled
                } else {
                    StepParticipantRetirementDisposition::Committed
                }
            );
        }
    }
    drop(setup.predecessor);
    drop(setup.batch);
    setup.sessions[0].try_abort().unwrap();
    setup.sessions[1].try_complete().unwrap();
}
