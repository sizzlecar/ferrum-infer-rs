use super::*;

struct SubmittedCohort {
    handle: CompletionHandle<TestRuntime>,
    fence: u64,
    step: Arc<StepResourceLease<TestRuntime>>,
    batch: ExecutionBatchParticipants<TestRuntime>,
    sessions: Vec<Arc<SequenceSession<TestRuntime>>>,
    _resources: Vec<Arc<AdmittedSequenceResources<TestRuntime>>>,
}

impl SubmittedCohort {
    fn retire(self) {
        drop(self.handle);
        self.step.try_retire_normal().unwrap();
        drop(self.batch);
        for session in &self.sessions {
            session.try_complete().unwrap();
        }
    }
}

fn requests(participants: u32, offset: u64) -> CompletionReadbackBatchRequest {
    CompletionReadbackBatchRequest::new(
        (0..participants)
            .map(|participant| {
                CompletionReadbackRequest::new(
                    id("node.tail"),
                    participant,
                    id("resource.output"),
                    offset,
                    HostTransferLayout::new(ElementType::F32, 2).unwrap(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()
}

fn submit_cohort(
    fixture: &Fixture,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    name: &str,
    participants: usize,
    readbacks: Option<CompletionReadbackBatchRequest>,
) -> SubmittedCohort {
    submit_cohort_with_spans(
        fixture,
        lane,
        reaper,
        name,
        vec![one_token_span(); participants],
        None,
        readbacks,
    )
}

fn submit_cohort_with_spans(
    fixture: &Fixture,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    name: &str,
    spans: Vec<TokenSpanWork>,
    input_tokens: Option<Vec<u32>>,
    readbacks: Option<CompletionReadbackBatchRequest>,
) -> SubmittedCohort {
    let participants = spans.len();
    let resources = (0..participants)
        .map(|participant| {
            logical_resources(
                &fixture.plan_resources,
                &format!("run.staged-readback.{name}.{participant}"),
                &format!("request.staged-readback.{name}.{participant}"),
            )
        })
        .collect::<Vec<_>>();
    let sessions = resources
        .iter()
        .map(|resources| resources.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(spans).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let mut admitted = None;
    for _ in 0..=3 {
        match batch.try_begin_step(request.clone(), lane).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => {
                admitted = Some(step);
                break;
            }
            StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                deferred.maintain().unwrap();
            }
            _ => panic!("readback cohort could not acquire a step"),
        }
    }
    let step = admitted.expect("test step backing growth must converge");
    let mut wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    if let Some(request) = readbacks {
        wave = wave.with_submission_readbacks(request).unwrap();
    }
    let active = batch
        .sessions()
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
    let uploads = input_tokens
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(row, token)| {
            SubmissionWaveInputUpload::new(
                id("node.main"),
                row as u32,
                0,
                0,
                HostTransferLayout::new(ElementType::U32, 1).unwrap(),
                token.to_le_bytes().to_vec(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let handle = OperationDispatch::encode_and_submit_wave_with_inputs(
        &providers,
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &uploads,
        wave,
        lane,
        reaper,
    )
    .unwrap();
    let fence = fixture.runtime_trace.lock().unwrap().next_fence;
    SubmittedCohort {
        handle,
        fence,
        step,
        batch,
        sessions,
        _resources: resources,
    }
}

#[test]
fn staged_parent_readback_does_not_wait_for_pending_same_lane_child() {
    let fixture = fixture();
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .submission_readback_enabled = true;
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    lane.configure_submission_readback_staging(16).unwrap();
    let reaper = CompletionReaper::new();
    let readbacks = requests(2, 4);
    let parent = submit_cohort(
        &fixture,
        &lane,
        &reaper,
        "parent",
        2,
        Some(readbacks.clone()),
    );
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .fence_behaviors
        .insert(parent.fence, FenceBehavior::Pending);
    assert!(matches!(
        parent.handle.poll().unwrap(),
        CompletionObservation::Pending
    ));
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_readback_reads,
        0
    );
    // Same number of participants and valid ranges, but not the exact request
    // whose snapshot was attached to this parent before its submission.
    assert!(parent.handle.wait_with_readbacks(requests(2, 0)).is_err());
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_readback_reads,
        0
    );

    let child = submit_cohort(&fixture, &lane, &reaper, "child", 1, None);
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace
            .fence_behaviors
            .insert(parent.fence, FenceBehavior::Succeeded);
        trace
            .fence_behaviors
            .insert(child.fence, FenceBehavior::Pending);
        trace.waited_fences.clear();
    }
    let receipt = match parent.handle.wait_with_readbacks(readbacks).unwrap() {
        CompletionReadbackBatchObservation::Terminal(receipt) => receipt,
        other => panic!("parent readback did not terminate: {other:?}"),
    };
    let ranges = {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.waited_fences, [parent.fence]);
        assert_eq!(trace.submission_readback_reads, 2);
        assert_eq!(
            trace.readback_calls, 0,
            "staged output must not use generic readback"
        );
        assert_eq!(trace.synchronize_calls, 0);
        trace.submission_readback_ranges.clone()
    };
    assert_eq!(ranges.len(), 2);
    assert_eq!(
        ranges[1].0.source_offset_bytes() - ranges[0].0.source_offset_bytes(),
        16
    );
    for (participant, (disposition, (range, layout))) in
        receipt.dispositions().iter().zip(ranges).enumerate()
    {
        let CompletionReadbackDisposition::Succeeded(output) = disposition else {
            panic!("parent output was not successful: {disposition:?}");
        };
        assert_eq!(output.request().participant_index(), participant as u32);
        assert_eq!(layout.element_type(), ElementType::F32);
        assert_eq!(range.length_bytes(), 8);
        let expected = (range.source_offset_bytes()..range.source_offset_bytes() + 8)
            .map(|offset| offset as u8)
            .collect::<Vec<_>>();
        assert_eq!(output.bytes(), expected);
    }
    assert!(matches!(
        child.handle.poll().unwrap(),
        CompletionObservation::Pending
    ));
    assert_eq!(lane.in_flight_count(), 1);
    assert_eq!(reaper.retained_count(), 1);
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .fence_behaviors
        .insert(child.fence, FenceBehavior::Succeeded);
    assert!(matches!(
        child.handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    parent.retire();
    child.retire();
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_readback_live,
        0
    );
    lane.configure_submission_readback_staging(16).unwrap();
}

#[test]
fn staging_budget_shortfall_drops_partial_snapshots_and_falls_back() {
    let fixture = fixture();
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .submission_readback_enabled = true;
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    // One eight-byte row fits; two cannot. A partially prepared batch must
    // release the first reservation before using ordinary terminal readback.
    lane.configure_submission_readback_staging(12).unwrap();
    let reaper = CompletionReaper::new();
    let readbacks = requests(2, 4);
    let cohort = submit_cohort(
        &fixture,
        &lane,
        &reaper,
        "budget",
        2,
        Some(readbacks.clone()),
    );
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_readback_live,
        0
    );
    let receipt = match cohort.handle.wait_with_readbacks(readbacks).unwrap() {
        CompletionReadbackBatchObservation::Terminal(receipt) => receipt,
        other => panic!("fallback readback did not terminate: {other:?}"),
    };
    assert!(receipt
        .dispositions()
        .iter()
        .all(|result| matches!(result, CompletionReadbackDisposition::Succeeded(_))));
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.submission_readback_reads, 0);
        assert_eq!(trace.readback_calls, 2);
    }
    cohort.retire();
    lane.configure_submission_readback_staging(16).unwrap();
    // A later complete reservation proves the fallback did not leak capacity.
    let followup_requests = requests(2, 4);
    let followup = submit_cohort(
        &fixture,
        &lane,
        &reaper,
        "followup",
        2,
        Some(followup_requests.clone()),
    );
    assert!(matches!(
        followup
            .handle
            .wait_with_readbacks(followup_requests)
            .unwrap(),
        CompletionReadbackBatchObservation::Terminal(_)
    ));
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_readback_reads,
        2
    );
    followup.retire();
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_readback_live,
        0
    );
    lane.configure_submission_readback_staging(12).unwrap();
}

#[test]
fn staged_readback_classification_panic_returns_failure_and_releases_terminal_ownership() {
    let fixture = fixture();
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.submission_readback_enabled = true;
        trace.submission_readback_fails = true;
        trace.describe_error_panics = true;
    }
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    lane.configure_submission_readback_staging(8).unwrap();
    let reaper = CompletionReaper::new();
    let readbacks = requests(1, 4);
    let cohort = submit_cohort(
        &fixture,
        &lane,
        &reaper,
        "classification-panic",
        1,
        Some(readbacks.clone()),
    );
    let step = Arc::downgrade(&cohort.step);
    let source = Arc::downgrade(&cohort._resources[0]);

    // The production terminal path must contain the classification panic.
    // Catching it here would hide skipped lane accounting and slot removal.
    let receipt = match cohort
        .handle
        .wait_with_readbacks(readbacks.clone())
        .unwrap()
    {
        CompletionReadbackBatchObservation::Terminal(receipt) => receipt,
        other => panic!("classification failure did not return a terminal receipt: {other:?}"),
    };
    assert!(matches!(
        receipt.dispositions(),
        [CompletionReadbackDisposition::ContractFailedButQuiescent { request, .. }]
            if request == &readbacks.requests()[0]
    ));
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(lane.in_flight_count(), 0);
    assert!(!lane.is_reusable());
    assert!(
        cohort.handle.poll().is_err(),
        "the terminal slot was removed"
    );
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.submission_readback_reads, 1);
        assert_eq!(trace.submission_readback_live, 1);
        assert_eq!(trace.readback_calls, 0);
        assert_eq!(trace.synchronize_calls, 0);
    }
    cohort.retire();
    assert!(step.upgrade().is_none());
    assert!(source.upgrade().is_none());
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_readback_live,
        0
    );
}

#[test]
fn submitted_parent_retirement_preserves_admitted_successor_and_physical_claims() {
    let fixture = fixture_tokens();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    let parent = submit_cohort_with_spans(
        &fixture,
        &lane,
        &reaper,
        "successor-admission",
        [53, 71]
            .into_iter()
            .map(|token| TokenSpanWork::from_token_ids_with_fit(&[token], 0..1, 8).unwrap())
            .collect(),
        Some(vec![53, 71]),
        None,
    );
    let predecessor = Arc::new(parent.handle.take_submitted_predecessor().unwrap());
    assert!(
        parent.handle.take_submitted_predecessor().is_err(),
        "authority is captured once"
    );
    assert!(
        predecessor.bind_next_token_work(requests(2, 0)).is_err(),
        "F32 bytes are not a token source"
    );
    let sources = CompletionReadbackBatchRequest::new(
        (0..2)
            .map(|row| {
                CompletionReadbackRequest::new(
                    id("node.tail"),
                    row,
                    id("resource.output"),
                    0,
                    HostTransferLayout::new(ElementType::U32, 1).unwrap(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let work = predecessor.bind_next_token_work(sources).unwrap();
    let request = StepResourceAdmissionRequest::new(
        work,
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let child = loop {
        match parent
            .batch
            .try_begin_successor_step(request.clone(), &lane, Arc::clone(&predecessor))
            .unwrap()
        {
            StepResourceAdmissionDecision::Admitted(step) => break step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                deferred.maintain().unwrap();
            }
            _ => panic!("successor could not acquire its separate backing"),
        }
    };
    loop {
        match parent.batch.try_begin_successor_step(
            request.clone(),
            &lane,
            Arc::clone(&predecessor),
        ) {
            Err(_) => break,
            Ok(StepResourceAdmissionDecision::BackingDeferred(deferred)) => {
                deferred.maintain().unwrap();
            }
            _ => panic!("the same submitted parent cannot admit a third frame"),
        }
    }
    assert!(
        parent.batch.try_begin_step(request.clone(), &lane).is_err(),
        "ordinary admission remains serial"
    );
    let node_requests = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| {
            InvocationResourceAdmissionRequest::for_all_step_participants(
                node.id().clone(),
                child
                    .bind_all_invocation_work_shape(
                        child
                            .work_shape()
                            .participant_work()
                            .iter()
                            .map(|work| work.token_span().clone())
                            .collect(),
                    )
                    .unwrap(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    assert!(child
        .try_admit_invocation(node_requests[0].clone())
        .is_err());
    assert!(child
        .try_prepare_submission_wave(vec![node_requests[0].clone()])
        .is_err());
    assert!(child
        .try_prepare_determinism_submission_wave(node_requests)
        .is_err());
    let parent_slices = parent.step.claimed_backing().backing_slices();
    let child_slices = child.claimed_backing().backing_slices();
    assert!(!parent_slices.is_empty());
    assert!(!child_slices.is_empty());
    for parent_slice in parent_slices {
        for child_slice in child_slices {
            for a in parent_slice.evidence().segments() {
                for b in child_slice.evidence().segments() {
                    assert!(
                        a.chunk() != b.chunk()
                            || a.offset_bytes() + a.length_bytes() <= b.offset_bytes()
                            || b.offset_bytes() + b.length_bytes() <= a.offset_bytes(),
                        "live predecessor and successor may not overlap physical storage"
                    );
                }
            }
        }
    }
    assert!(matches!(
        parent.handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    let SubmittedCohort {
        handle,
        step,
        batch,
        sessions,
        _resources,
        ..
    } = parent;
    drop(handle);
    let weak_parent_step = Arc::downgrade(&step);
    step.try_retire_normal().unwrap();
    assert!(
        weak_parent_step.upgrade().is_none(),
        "child authority retains claims, not its parent Step"
    );
    for session in &sessions {
        assert!(
            session.try_complete().is_err(),
            "parent retirement must leave the successor live"
        );
    }
    child.try_rollback_unsubmitted().unwrap();
    assert!(
        batch.try_begin_step(request, &lane).is_err(),
        "an otherwise idle session must reject device-dependent work through serial admission"
    );
    drop(predecessor);
    drop(batch);
    for session in &sessions {
        session.try_complete().unwrap();
    }
}
