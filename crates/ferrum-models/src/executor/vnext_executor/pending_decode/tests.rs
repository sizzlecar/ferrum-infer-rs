use super::vnext_device_operation_contract::*;
use super::vnext_device_operation_wave_contract::*;
use super::*;

struct PairFixture {
    fixture: Fixture,
    sessions: Vec<Arc<SequenceSession<TestRuntime>>>,
    _resources: Vec<Arc<AdmittedSequenceResources<TestRuntime>>>,
    _batch: ExecutionBatchParticipants<TestRuntime>,
    lane: Arc<ExecutionLane<TestRuntime>>,
    reaper: Arc<CompletionReaper<TestRuntime>>,
    parent: SubmittedPairWave<TestRuntime>,
    child: SubmittedPairWave<TestRuntime>,
    parent_fence: u64,
    child_fence: u64,
}

fn requests() -> CompletionReadbackBatchRequest {
    CompletionReadbackBatchRequest::new(
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
    .unwrap()
}

fn submit(
    fixture: &Fixture,
    sessions: &[Arc<SequenceSession<TestRuntime>>],
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    wave: PreparedStepSubmissionWave<TestRuntime>,
    uploads: &[SubmissionWaveInputUpload],
) -> CompletionHandle<TestRuntime> {
    let active = sessions
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        lane,
    )
    .unwrap();
    OperationDispatch::encode_and_submit_wave_with_inputs(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        uploads,
        wave,
        lane,
        reaper,
    )
    .unwrap()
}

impl PairFixture {
    fn new() -> Self {
        let fixture = fixture_tokens();
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_readback_enabled = true;
        let lane = fixture.plan_resources.create_execution_lane().unwrap();
        lane.configure_submission_readback_staging(16).unwrap();
        let reaper = CompletionReaper::new();
        let resources = (0..2)
            .map(|row| {
                logical_resources(
                    &fixture.plan_resources,
                    &format!("run.pair.{row}"),
                    &format!("request.pair.{row}"),
                )
            })
            .collect::<Vec<_>>();
        let sessions = resources
            .iter()
            .map(|resource| resource.open_session().unwrap())
            .collect::<Vec<_>>();
        let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
        let tokens = [53_u32, 127];
        let work = batch
            .bind_work_shape(
                tokens
                    .iter()
                    .map(|token| {
                        TokenSpanWork::from_token_ids_with_fit(&[*token], 0..1, 4).unwrap()
                    })
                    .collect(),
            )
            .unwrap();
        let request = StepResourceAdmissionRequest::new(
            work,
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let parent = loop {
            match batch.try_begin_step(request.clone(), &lane).unwrap() {
                StepResourceAdmissionDecision::Admitted(step) => break step,
                StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                    deferred.maintain().unwrap();
                }
                _ => panic!("parent backing did not admit"),
            }
        };
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &parent)
            .with_submission_readbacks(requests())
            .unwrap();
        let uploads = tokens
            .iter()
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
        let parent_handle = submit(&fixture, &sessions, &lane, &reaper, wave, &uploads);
        let parent_fence = fixture.runtime_trace.lock().unwrap().next_fence;
        let predecessor = Arc::new(parent_handle.take_submitted_predecessor().unwrap());
        let work = predecessor.bind_next_token_work(requests()).unwrap();
        let request = StepResourceAdmissionRequest::new(
            work,
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let child = loop {
            match batch
                .try_begin_successor_step(request.clone(), &lane, Arc::clone(&predecessor))
                .unwrap()
            {
                StepResourceAdmissionDecision::Admitted(step) => break step,
                StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                    deferred.maintain().unwrap();
                }
                _ => panic!("child backing did not admit"),
            }
        };
        let forwards = requests()
            .requests()
            .iter()
            .cloned()
            .map(|source| {
                predecessor
                    .forward_token(source, id("node.main"), 0, 0)
                    .unwrap()
            })
            .collect();
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &child)
            .with_forwarded_inputs(forwards)
            .unwrap();
        let child_handle = submit(&fixture, &sessions, &lane, &reaper, wave, &[]);
        let child_fence = fixture.runtime_trace.lock().unwrap().next_fence;
        Self {
            fixture,
            sessions,
            _resources: resources,
            _batch: batch,
            lane,
            reaper,
            parent: SubmittedPairWave::new(parent_handle, requests(), parent),
            child: SubmittedPairWave::new(child_handle, requests(), child),
            parent_fence,
            child_fence,
        }
    }
}

fn assert_row(row: &CompletionReadbackDisposition, expected: u32) {
    let CompletionReadbackDisposition::Succeeded(output) = row else {
        panic!("expected successful row: {row:?}")
    };
    assert_eq!(output.bytes(), expected.to_le_bytes());
}

#[tokio::test]
async fn parent_ack_precedes_child_observation_and_rows_are_consumed_independently() {
    let setup = PairFixture::new();
    let parent_weak = Arc::downgrade(&setup.parent.step);
    let child_weak = Arc::downgrade(&setup.child.step);
    let worker = VNextCompletionWorker::new().unwrap();
    let (parent, cohort) = submit_pair(
        worker.reserve().await.unwrap(),
        setup.parent,
        setup.child,
        Arc::clone(&setup.reaper),
    );
    let (receipt, guard) = parent.wait().await.unwrap();
    assert_row(&receipt.dispositions()[0], 54);
    assert_row(&receipt.dispositions()[1], 128);
    assert_eq!(guard.step().participant_count(), 2);
    assert!(!cohort.is_ready());
    assert_eq!(
        setup.fixture.runtime_trace.lock().unwrap().waited_fences,
        [setup.parent_fence]
    );
    guard.retire_normal().unwrap();
    cohort.wait().await.unwrap();
    assert!(parent_weak.upgrade().is_none());
    assert!(child_weak.upgrade().is_none());
    assert_eq!(cohort.row_count(), 2);
    assert_eq!(cohort.rows_remaining(), 2);
    assert_row(cohort.peek_row(0).unwrap().disposition(), 55);
    assert_row(cohort.peek_row(0).unwrap().disposition(), 55);
    // A capacity-deferred caller can peek repeatedly without losing the row.
    assert_eq!(cohort.rows_remaining(), 2);
    cohort.discard_row(0).unwrap();
    assert!(cohort.take_row(0).is_err());
    assert_row(cohort.take_row(1).unwrap().disposition(), 129);
    assert!(cohort.take_row(1).is_err());
    assert_eq!(cohort.rows_remaining(), 0);
    assert_eq!(setup.reaper.retained_count(), 0);
    assert_eq!(setup.lane.in_flight_count(), 0);
    for session in &setup.sessions {
        session.try_complete().unwrap();
    }
}

#[tokio::test]
async fn abandoned_parent_receiver_or_guard_still_drains_the_child() {
    for receive_parent in [false, true] {
        let setup = PairFixture::new();
        let parent_weak = Arc::downgrade(&setup.parent.step);
        let child_weak = Arc::downgrade(&setup.child.step);
        let worker = VNextCompletionWorker::new().unwrap();
        let (parent, cohort) = submit_pair(
            worker.reserve().await.unwrap(),
            setup.parent,
            setup.child,
            Arc::clone(&setup.reaper),
        );
        if receive_parent {
            let (_, guard) = parent.wait().await.unwrap();
            drop(guard);
        } else {
            drop(parent);
        }
        assert!(cohort.wait().await.is_err());
        assert!(parent_weak.upgrade().is_none());
        assert!(child_weak.upgrade().is_none());
        assert_eq!(setup.reaper.retained_count(), 0);
        assert_eq!(setup.lane.in_flight_count(), 0);
        let trace = setup.fixture.runtime_trace.lock().unwrap();
        assert!(trace.waited_fences.contains(&setup.child_fence));
        assert_eq!(
            trace.readback_calls, 0,
            "failed child output must not be read"
        );
        drop(trace);
        for session in &setup.sessions {
            session.try_abort().unwrap();
        }
    }
}

#[tokio::test]
async fn indeterminate_parent_is_drained_before_child_cleanup() {
    let setup = PairFixture::new();
    setup
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .fence_behaviors
        .insert(setup.parent_fence, FenceBehavior::Indeterminate);
    let worker = VNextCompletionWorker::new().unwrap();
    let (parent, cohort) = submit_pair(
        worker.reserve().await.unwrap(),
        setup.parent,
        setup.child,
        Arc::clone(&setup.reaper),
    );
    assert!(parent.wait().await.is_err());
    assert!(cohort.wait().await.is_err());
    let trace = setup.fixture.runtime_trace.lock().unwrap();
    assert!(
        trace.synchronize_calls > 0,
        "nonterminal parent requires a real lane drain"
    );
    assert!(trace.waited_fences.contains(&setup.child_fence));
    assert_eq!(trace.readback_calls, 0);
    assert_eq!(setup.reaper.retained_count(), 0);
    assert!(setup.lane.is_fail_closed());
}

#[tokio::test]
async fn abandoning_child_result_does_not_cancel_accepted_cleanup() {
    let setup = PairFixture::new();
    let child_weak = Arc::downgrade(&setup.child.step);
    let worker = VNextCompletionWorker::new().unwrap();
    let (parent, cohort) = submit_pair(
        worker.reserve().await.unwrap(),
        setup.parent,
        setup.child,
        Arc::clone(&setup.reaper),
    );
    let cohort_weak = Arc::downgrade(&cohort);
    let (_, guard) = parent.wait().await.unwrap();
    drop(cohort);
    guard.retire_normal().unwrap();
    // The same bounded worker's next job is a completion barrier, without
    // polling timings or spawning a second cleanup worker.
    worker
        .reserve()
        .await
        .unwrap()
        .submit(VNextCompletionTaskKind::WaveReadback, || ())
        .wait()
        .await
        .unwrap();
    assert!(cohort_weak.upgrade().is_none());
    assert!(child_weak.upgrade().is_none());
    assert_eq!(setup.reaper.retained_count(), 0);
    assert_eq!(setup.lane.in_flight_count(), 0);
    for session in &setup.sessions {
        session.try_complete().unwrap();
    }
}

#[tokio::test]
async fn preobserved_parent_reuses_its_receipt_without_observing_the_removed_slot() {
    let setup = PairFixture::new();
    let worker = VNextCompletionWorker::new().unwrap();
    let parent = observe_parent(
        worker.reserve().await.unwrap(),
        setup.parent,
        Arc::clone(&setup.reaper),
    )
    .wait()
    .await
    .unwrap()
    .unwrap();
    assert_row(&parent.receipt().dispositions()[0], 54);
    assert_eq!(
        setup.reaper.retained_count(),
        1,
        "only the child slot remains"
    );
    let before = setup
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .waited_fences
        .clone();
    let (parent, cohort) = submit_pair_with_observed_parent(
        worker.reserve().await.unwrap(),
        parent,
        setup.child,
        Arc::clone(&setup.reaper),
        |_| {},
    );
    let (receipt, guard) = parent.wait().await.unwrap();
    assert_row(&receipt.dispositions()[1], 128);
    assert_eq!(
        setup.fixture.runtime_trace.lock().unwrap().waited_fences,
        before
    );
    assert!(!cohort.is_ready());
    guard.retire_normal().unwrap();
    cohort.wait().await.unwrap();
    assert_row(cohort.take_row(0).unwrap().disposition(), 55);
    assert_row(cohort.take_row(1).unwrap().disposition(), 129);
    assert_eq!(setup.reaper.retained_count(), 0);
    assert_eq!(setup.lane.in_flight_count(), 0);
    for session in &setup.sessions {
        session.try_complete().unwrap();
    }
}

#[tokio::test]
async fn abandoned_preobserved_parent_aborts_without_repeating_fence_observation() {
    let setup = PairFixture::new();
    let parent_weak = Arc::downgrade(&setup.parent.step);
    let worker = VNextCompletionWorker::new().unwrap();
    let parent = observe_parent(
        worker.reserve().await.unwrap(),
        setup.parent,
        Arc::clone(&setup.reaper),
    )
    .wait()
    .await
    .unwrap()
    .unwrap();
    let before = setup
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .waited_fences
        .clone();
    drop(parent);
    assert!(parent_weak.upgrade().is_none());
    assert_eq!(
        setup.fixture.runtime_trace.lock().unwrap().waited_fences,
        before
    );
    let mut child = OwnedWave::new(setup.child, Arc::clone(&setup.reaper));
    worker
        .reserve()
        .await
        .unwrap()
        .submit(VNextCompletionTaskKind::PostSubmitDrain, move || {
            let receipt = child.observe().unwrap();
            assert!(matches!(
                receipt.completion().disposition(),
                OperationCompletionDisposition::ContractFailedButQuiescent(_)
            ));
            assert!(receipt
                .dispositions()
                .iter()
                .all(|row| matches!(row, CompletionReadbackDisposition::NotAttempted(_))));
            drop(child);
        })
        .wait()
        .await
        .unwrap();
    assert_eq!(setup.reaper.retained_count(), 0);
    assert_eq!(setup.lane.in_flight_count(), 0);
    for session in &setup.sessions {
        session.try_abort().unwrap();
    }
}

#[tokio::test]
async fn discarding_row_during_accepted_read_preserves_physical_output_and_peer() {
    let setup = PairFixture::new();
    let worker = VNextCompletionWorker::new().unwrap();
    let (entered, observed) = tokio::sync::oneshot::channel();
    let (release, wait_release) = sync_channel::<()>(1);
    let (parent, cohort) = submit_pair_with_observer(
        worker.reserve().await.unwrap(),
        setup.parent,
        setup.child,
        Arc::clone(&setup.reaper),
        move |result| {
            assert!(result.is_ok());
            let _ = entered.send(());
            // Dropping the sender also releases this gate on test failure.
            let _ = wait_release.recv();
        },
    );
    let (_, guard) = parent.wait().await.unwrap();
    let sequence = product_sequence(&setup.sessions[0], &cohort, 0);
    let pending = sequence.pending_decode.lock().clone().unwrap();
    guard.retire_normal().unwrap();
    observed.await.unwrap();
    let read = pending.cohort.read_submitted_row(pending.row);
    tokio::pin!(read);
    std::future::poll_fn(|cx| {
        assert!(std::future::Future::poll(read.as_mut(), cx).is_pending());
        std::task::Poll::Ready(())
    })
    .await;

    // This is the exact synchronous cancellation action used by release_cache,
    // after the accepted caller cloned its pending row and began awaiting it.
    sequence.abort();
    assert!(sequence.pending_decode.lock().is_none());
    drop(release);
    let output = read
        .await
        .expect("accepted read must retain its real submitted output");
    assert_row(output.disposition(), 55);
    assert!(
        cohort.peek_row(0).is_err(),
        "discarded row has no consumption right"
    );
    assert!(cohort.take_row(0).is_err());
    assert_row(cohort.take_row(1).unwrap().disposition(), 129);
    setup.sessions[1].try_complete().unwrap();
    assert_eq!(setup.reaper.retained_count(), 0);
    assert_eq!(setup.lane.in_flight_count(), 0);
}

fn product_sequence(
    session: &Arc<SequenceSession<TestRuntime>>,
    cohort: &Arc<PendingDecodeCohort>,
    row: usize,
) -> Arc<VNextSequence<TestRuntime>> {
    let request =
        VNextRequestRoot::bind_initial(RequestId::new(), session.resources().request_id(), session)
            .unwrap();
    Arc::new(VNextSequence {
        prefix_capture_interests: parking_lot::Mutex::new(Vec::new()),
        cache_id: format!("pending-product-row-{row}"),
        request,
        session: Arc::clone(session),
        active_binding: Arc::new(TrustedActiveSequenceBinding::from_session(session).unwrap()),
        request_origin: ExecutorRequestOrigin::Product,
        tokens: parking_lot::Mutex::new(vec![if row == 0 { 53 } else { 127 }]),
        pending_decode: parking_lot::Mutex::new(Some(PendingDecodeRow {
            cohort: Arc::clone(cohort),
            row,
            expected_cache_tokens: 1,
            expected_input_token: if row == 0 { 54 } else { 128 },
        })),
        maximum_tokens: 4,
        active: AtomicBool::new(true),
        operation: AsyncMutex::new(()),
        events: None,
        product_prompt_tokens: 1,
        replayed_output_tokens: 0,
        prefill_tokens_processed: AtomicUsize::new(1),
    })
}

fn product_completion(sequence: &VNextSequence<TestRuntime>) -> ExecutorSequenceCompletion {
    ExecutorSequenceCompletion::new(
        sequence.request_id().clone(),
        sequence.cache_id.clone(),
        1,
        1,
    )
    .unwrap()
}

#[tokio::test]
async fn product_completion_drains_pending_child_and_discards_only_its_row() {
    use std::future::Future;
    use std::task::{Context, Waker};

    let setup = PairFixture::new();
    let worker = VNextCompletionWorker::new().unwrap();
    let (parent, cohort) = submit_pair(
        worker.reserve().await.unwrap(),
        setup.parent,
        setup.child,
        Arc::clone(&setup.reaper),
    );
    let (_, parent) = parent.wait().await.unwrap();
    let sequence = product_sequence(&setup.sessions[0], &cohort, 0);
    let completion = product_completion(&sequence);
    let mut completing = Box::pin(async {
        let mut guard = PendingSequenceCompletion {
            sequence: &sequence,
            operation: Some(sequence.operation.lock().await),
            completed: false,
        };
        sequence.complete(&completion).await?;
        guard.completed = true;
        Result::<()>::Ok(())
    });
    // Holding the real parent retirement acknowledgement is an event gate,
    // proving completion awaits accepted child cleanup without a timing test.
    assert!(completing
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
        .is_pending());
    assert!(sequence.operation.try_lock().is_err());
    assert!(sequence.pending_decode.lock().is_some());
    assert!(setup.sessions[0]
        .request_cancel()
        .unwrap()
        .successor_frame()
        .is_some());
    assert_eq!(cohort.rows_remaining(), 2);
    parent.retire_normal().unwrap();
    completing.await.unwrap();
    assert!(!sequence.active.load(Ordering::Acquire));
    assert!(sequence.pending_decode.lock().is_none());
    assert!(sequence.operation.try_lock().is_ok());
    assert!(setup.sessions[0].request_cancel().is_err());
    assert!(cohort.take_row(0).is_err());
    assert_row(cohort.take_row(1).unwrap().disposition(), 129);
    setup.sessions[1].try_complete().unwrap();
    assert_eq!(setup.reaper.retained_count(), 0);
    assert_eq!(setup.lane.in_flight_count(), 0);
}

#[tokio::test]
async fn dropped_product_completion_cancels_only_its_pending_row_and_worker_keeps_draining() {
    use std::future::Future;
    use std::task::{Context, Waker};

    let setup = PairFixture::new();
    let worker = VNextCompletionWorker::new().unwrap();
    let (parent, cohort) = submit_pair(
        worker.reserve().await.unwrap(),
        setup.parent,
        setup.child,
        Arc::clone(&setup.reaper),
    );
    let (_, parent) = parent.wait().await.unwrap();
    let sequence = product_sequence(&setup.sessions[0], &cohort, 0);
    let completion = product_completion(&sequence);
    let mut completing = Box::pin(async {
        let mut guard = PendingSequenceCompletion {
            sequence: &sequence,
            operation: Some(sequence.operation.lock().await),
            completed: false,
        };
        sequence.complete(&completion).await?;
        guard.completed = true;
        Result::<()>::Ok(())
    });
    assert!(completing
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
        .is_pending());
    drop(completing);
    assert!(!sequence.active.load(Ordering::Acquire));
    assert!(sequence.operation.try_lock().is_ok());
    assert!(sequence.pending_decode.lock().is_none());
    assert_eq!(cohort.rows_remaining(), 1);
    assert!(!cohort.is_ready());
    parent.retire_normal().unwrap();
    cohort.wait().await.unwrap();
    assert!(cohort.take_row(0).is_err());
    assert_row(cohort.take_row(1).unwrap().disposition(), 129);
    drop(sequence);
    assert!(setup.sessions[0].request_cancel().is_err());
    setup.sessions[1].try_complete().unwrap();
    assert_eq!(setup.reaper.retained_count(), 0);
    assert_eq!(setup.lane.in_flight_count(), 0);
}

#[tokio::test]
async fn releasing_or_dropping_product_pending_row_does_not_poison_its_peer() {
    for explicit_abort in [false, true] {
        let setup = PairFixture::new();
        let worker = VNextCompletionWorker::new().unwrap();
        let (parent, cohort) = submit_pair(
            worker.reserve().await.unwrap(),
            setup.parent,
            setup.child,
            Arc::clone(&setup.reaper),
        );
        let (_, parent) = parent.wait().await.unwrap();
        let sequence = product_sequence(&setup.sessions[0], &cohort, 0);
        if explicit_abort {
            sequence.abort();
            assert!(!sequence.active.load(Ordering::Acquire));
            assert!(sequence.pending_decode.lock().is_none());
        }
        drop(sequence);
        assert_eq!(cohort.rows_remaining(), 1);
        parent.retire_normal().unwrap();
        cohort.wait().await.unwrap();
        assert!(cohort.take_row(0).is_err());
        assert_row(cohort.take_row(1).unwrap().disposition(), 129);
        // The fixture independently retains the native sessions. Physical
        // cleanup is complete, so this retained cancelled session can abort.
        setup.sessions[0].try_abort().unwrap();
        setup.sessions[1].try_complete().unwrap();
        assert_eq!(setup.reaper.retained_count(), 0);
        assert_eq!(setup.lane.in_flight_count(), 0);
    }
}
