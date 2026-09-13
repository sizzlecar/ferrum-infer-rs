//! Real Engine iteration/admission/restore publication with a CPU-only executor.
//! Native numerical copy correctness remains a backend/runtime test obligation.
use super::*;
use ferrum_interfaces::model_executor::{
    PlanRuntimePrefixRestoreInput, PlanRuntimePrefixRestoreOutput, PrefixCaptureBoundary,
    PrefixCaptureLease, PrefixCaptureRequest, PrefixCaptureStatus,
};

#[derive(Debug, Clone, PartialEq, Eq)]
enum Event {
    Admitted(RequestId),
    Computed(RequestId, usize, usize),
    Captured(RequestId, usize),
    Acknowledged(RequestId, usize),
    Maintained(RequestId),
}

#[derive(Debug, Default)]
pub(super) struct CaptureState {
    leases: Mutex<Vec<Weak<TestCapture>>>,
    events: Mutex<Vec<Event>>,
    skip_capture: AtomicBool,
    skip_restore: AtomicBool,
    admission_pressure: AtomicBool,
    initial_materialization: AtomicBool,
    pub(super) wait_for_release: AtomicBool,
}

#[derive(Debug)]
struct TestCapture {
    owner: Arc<CaptureState>,
    source: RequestId,
    prefix: Vec<TokenId>,
    status: std::sync::atomic::AtomicU8,
    release_epoch: Arc<AtomicU64>,
    release_signal: tokio::sync::watch::Sender<u64>,
}

impl Drop for TestCapture {
    fn drop(&mut self) {
        if self.status.load(Ordering::Acquire) == 1 {
            let epoch = self.release_epoch.fetch_add(1, Ordering::AcqRel) + 1;
            self.release_signal.send_replace(epoch);
        }
    }
}
impl PrefixCaptureLease for TestCapture {
    fn boundary(&self) -> usize {
        self.prefix.len()
    }
    fn status(&self) -> PrefixCaptureStatus {
        match self.status.load(Ordering::Acquire) {
            0 => PrefixCaptureStatus::Pending,
            1 => PrefixCaptureStatus::Ready,
            _ => PrefixCaptureStatus::Unavailable,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CaptureState {
    pub(super) fn has_ready_capture(&self) -> bool {
        self.leases
            .lock()
            .iter()
            .filter_map(Weak::upgrade)
            .any(|lease| lease.status() == PrefixCaptureStatus::Ready)
    }
    pub(super) fn defer_admission_once(&self) -> bool {
        self.initial_materialization.swap(false, Ordering::AcqRel)
            || (self.has_ready_capture() && self.admission_pressure.swap(false, Ordering::AcqRel))
    }
    pub(super) fn maintained(&self, id: &RequestId) {
        self.events.lock().push(Event::Maintained(id.clone()));
    }
    pub(super) fn boundary(
        &self,
        input: PrefixCaptureBoundary<'_>,
    ) -> Option<ferrum_interfaces::model_executor::PrefixCapturePlan> {
        let boundary = input
            .follower_prompt_tokens
            .iter()
            .copied()
            .chain([input.source_prompt_tokens])
            .map(|length| length.saturating_sub(1))
            .fold(input.common_prefix_tokens, usize::min);
        (boundary > input.processed_tokens && !input.follower_prompt_tokens.is_empty()).then_some(
            ferrum_interfaces::model_executor::PrefixCapturePlan {
                boundary,
                span: ferrum_interfaces::vnext::CheckpointTokenSpanConstraint::any_positive(),
            },
        )
    }
    pub(super) fn arm(
        self: &Arc<Self>,
        input: PrefixCaptureRequest<'_>,
        release_epoch: Arc<AtomicU64>,
        release_signal: tokio::sync::watch::Sender<u64>,
    ) -> Option<Arc<dyn PrefixCaptureLease>> {
        let lease = Arc::new(TestCapture {
            owner: Arc::clone(self),
            source: input.source_request_id.clone(),
            prefix: input.source_tokens[..input.boundary].to_vec(),
            status: std::sync::atomic::AtomicU8::new(0),
            release_epoch,
            release_signal,
        });
        self.leases.lock().push(Arc::downgrade(&lease));
        Some(lease)
    }
    pub(super) fn admitted(&self, id: &RequestId) {
        self.events.lock().push(Event::Admitted(id.clone()));
    }
    pub(super) fn retired(&self, id: &RequestId, chunk: PrefillChunk) {
        self.events.lock().push(Event::Computed(
            id.clone(),
            chunk.tokens_processed(),
            chunk.end(),
        ));
        for lease in self.leases.lock().iter().filter_map(Weak::upgrade) {
            if &lease.source == id && chunk.end() == lease.boundary() {
                if self.skip_capture.load(Ordering::Relaxed) {
                    lease.status.store(2, Ordering::Release);
                } else {
                    lease.status.store(1, Ordering::Release);
                    self.events
                        .lock()
                        .push(Event::Captured(id.clone(), chunk.end()));
                }
            }
        }
    }
    pub(super) fn restore(
        self: &Arc<Self>,
        input: PlanRuntimePrefixRestoreInput<'_>,
    ) -> Result<Option<PlanRuntimePrefixRestoreOutput>> {
        let Some(lease) = input
            .checkpoint
            .and_then(|lease| lease.as_any().downcast_ref::<TestCapture>())
        else {
            return Ok(None);
        };
        assert!(Arc::ptr_eq(&lease.owner, self));
        assert_eq!(lease.status(), PrefixCaptureStatus::Ready);
        assert!(input.input_tokens.starts_with(&lease.prefix));
        if self.skip_restore.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let boundary = lease.boundary();
        let target = input.request_id.clone();
        let owner = Arc::clone(self);
        let cache = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
            target.clone(),
            1,
            boundary,
        ));
        PlanRuntimePrefixRestoreOutput::new(
            target.clone(),
            boundary,
            input.input_tokens.len(),
            cache,
            move || {
                owner
                    .events
                    .lock()
                    .push(Event::Acknowledged(target, boundary));
                Ok(())
            },
        )
        .map(Some)
    }
}

fn engine() -> (ContinuousBatchEngine, Arc<CaptureState>) {
    let mut config = EngineConfig::default();
    config.scheduler.max_running_requests = 3;
    config.scheduler.prefill_step_chunk = Some(3);
    config.scheduler.prefill_first_until_active = Some(3);
    config.scheduler.prefix_rendezvous_max_wait_ms = std::num::NonZeroU64::new(10_000);
    let state = Arc::new(CaptureState::default());
    let mut executor = PlanRuntimeChunkedPrefillTestExecutor::new(false);
    executor.rendezvous = Some(Arc::clone(&state));
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        scheduler,
        Arc::new(ferrum_testkit::MockTokenizer::new(128)),
        Arc::new(ferrum_testkit::MockSampler),
        Arc::new(executor),
        Arc::new(MockTensorFactory),
    )
    .unwrap();
    (engine, state)
}

async fn submit(
    engine: &ContinuousBatchEngine,
    tokens: &[u32],
) -> (
    RequestId,
    tokio::sync::oneshot::Receiver<Result<InferenceResponse>>,
) {
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;
    request.metadata.insert(
        PROMPT_TOKENS_METADATA_KEY.to_owned(),
        serde_json::json!(tokens.len()),
    );
    let id = request.id.clone();
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request.clone(),
        tokens.iter().copied().map(TokenId::new).collect(),
        Some(Arc::clone(&engine.inner.tokenizer)),
        Some(128),
    );
    let (sender, receiver) = tokio::sync::oneshot::channel();
    sequence.response_sender = Some(sender);
    engine.inner.sequences.write().insert(id.clone(), sequence);
    engine.inner.scheduler.submit(request).await.unwrap();
    (id, receiver)
}

async fn finish(engine: &ContinuousBatchEngine) {
    for _ in 0..32 {
        if engine.inner.sequences.read().is_empty() {
            return;
        }
        engine.inner.run_iteration().await.unwrap();
    }
    panic!("bounded prefix fixture did not finish");
}

#[tokio::test]
async fn prefix_rendezvous_handoff_admits_after_capture_and_executes_only_unique_suffixes() {
    let (engine, state) = engine();
    let (source, source_rx) = submit(&engine, &[11, 12, 13, 14, 21]).await;
    let (a, a_rx) = submit(&engine, &[11, 12, 13, 14, 22, 23]).await;
    let (b, b_rx) = submit(&engine, &[11, 12, 13, 14, 24]).await;
    finish(&engine).await;
    for receiver in [source_rx, a_rx, b_rx] {
        assert!(receiver.await.unwrap().is_ok());
    }
    let events = state.events.lock();
    let captured = events
        .iter()
        .position(|event| event == &Event::Captured(source.clone(), 4))
        .unwrap();
    for id in [&a, &b] {
        let admitted = events
            .iter()
            .position(|event| event == &Event::Admitted(id.clone()))
            .unwrap();
        let acknowledged = events
            .iter()
            .position(|event| event == &Event::Acknowledged(id.clone(), 4))
            .unwrap();
        assert!(captured < admitted && admitted < acknowledged);
        let (computed, start) = events
            .iter()
            .enumerate()
            .find_map(|(index, event)| match event {
                Event::Computed(request, start, _) if request == id => Some((index, *start)),
                _ => None,
            })
            .unwrap();
        assert!(acknowledged < computed);
        assert_eq!(start, 4);
    }
    assert!(
        events.contains(&Event::Computed(source, 3, 4)),
        "scheduler must stop at a non-chunk-aligned common boundary"
    );
    drop(events);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_rendezvous_late_arrivals_share_a_boundary_after_the_retired_first_chunk() {
    let (engine, state) = engine();
    let (source, source_rx) = submit(&engine, &[11, 12, 13, 14, 15, 16, 21]).await;
    for _ in 0..4 {
        engine.inner.run_iteration().await.unwrap();
        if state
            .events
            .lock()
            .contains(&Event::Computed(source.clone(), 0, 3))
        {
            break;
        }
    }
    assert!(state
        .events
        .lock()
        .contains(&Event::Computed(source.clone(), 0, 3)));
    assert!(state.leases.lock().is_empty());
    let (target, target_rx) = submit(&engine, &[11, 12, 13, 14, 15, 16, 22]).await;
    finish(&engine).await;
    assert!(source_rx.await.unwrap().is_ok());
    assert!(target_rx.await.unwrap().is_ok());
    let events = state.events.lock();
    assert!(events.contains(&Event::Captured(source, 6)));
    assert!(events.contains(&Event::Acknowledged(target.clone(), 6)));
    assert!(!events.iter().any(
        |event| matches!(event, Event::Computed(id, start, _) if id == &target && *start < 6)
    ));
    drop(events);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_rendezvous_capture_pressure_or_prewrite_restore_skip_runs_cold_without_rearming() {
    for skip_capture in [true, false] {
        let (engine, state) = engine();
        state.skip_capture.store(skip_capture, Ordering::Relaxed);
        state.skip_restore.store(!skip_capture, Ordering::Relaxed);
        let (_source, source_rx) = submit(&engine, &[11, 12, 13, 14, 21]).await;
        let (target, target_rx) = submit(&engine, &[11, 12, 13, 14, 22]).await;
        finish(&engine).await;
        assert!(source_rx.await.unwrap().is_ok());
        assert!(target_rx.await.unwrap().is_ok());
        let events = state.events.lock();
        assert!(events
            .iter()
            .any(|event| matches!(event, Event::Computed(id, 0, _) if id == &target)));
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::Acknowledged(_, _))));
        assert_eq!(state.leases.lock().len(), 1);
        drop(events);
        engine.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn prefix_rendezvous_cancelled_leader_releases_follower_and_unrelated_request_runs() {
    let (engine, state) = engine();
    let (source, source_rx) = submit(&engine, &[11, 12, 13, 14, 21]).await;
    let (target, target_rx) = submit(&engine, &[11, 12, 13, 14, 22]).await;
    let (other, other_rx) = submit(&engine, &[61, 62, 63, 64, 65]).await;
    engine.inner.run_iteration().await.unwrap();
    assert!(!state
        .events
        .lock()
        .contains(&Event::Admitted(target.clone())));
    assert!(state.events.lock().contains(&Event::Admitted(other)));
    drop(source_rx);
    finish(&engine).await;
    assert!(target_rx.await.unwrap().is_ok());
    assert!(other_rx.await.unwrap().is_ok());
    let events = state.events.lock();
    assert!(!events.contains(&Event::Captured(source, 4)));
    assert!(events
        .iter()
        .any(|event| matches!(event, Event::Computed(id, 0, _) if id == &target)));
    drop(events);
    engine.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn prefix_rendezvous_expiry_releases_optional_wait_once_without_failing_request() {
    let (engine, state) = engine();
    let (_source, source_rx) = submit(&engine, &[11, 12, 13, 14, 21]).await;
    let (target, target_rx) = submit(&engine, &[11, 12, 13, 14, 22]).await;
    engine.inner.run_iteration().await.unwrap();
    assert!(!state
        .events
        .lock()
        .contains(&Event::Admitted(target.clone())));
    let wait = engine
        .inner
        .config
        .scheduler
        .prefix_rendezvous_max_wait_ms
        .unwrap();
    // Exercise the same timer future used by Idle/CapacityBlocked, without a
    // real wait or a spawned rendezvous task. Virtual Tokio time wakes it;
    // native lease expiry still uses the real monotonic clock below.
    let mut wake = tokio_test::task::spawn(engine.inner.wait_for_prefix_deadline());
    tokio_test::assert_pending!(wake.poll());
    tokio::time::advance(Duration::from_millis(wait.get()) + Duration::from_secs(1)).await;
    tokio_test::assert_ready!(wake.poll());
    drop(wake);
    engine
        .inner
        .refresh_prefix_rendezvous_at(Instant::now() + Duration::from_millis(wait.get()))
        .unwrap();
    assert_eq!(engine.inner.scheduler.prefix_held_waiting_count(), 0);
    assert!(state
        .leases
        .lock()
        .iter()
        .all(|lease| lease.upgrade().is_none()));
    finish(&engine).await;
    assert!(source_rx.await.unwrap().is_ok());
    assert!(target_rx.await.unwrap().is_ok());
    assert_eq!(state.leases.lock().len(), 1);
    assert!(state
        .events
        .lock()
        .iter()
        .any(|event| matches!(event, Event::Computed(id, 0, _) if id == &target)));
    engine.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn prefix_rendezvous_shutdown_drops_wait_and_pending_checkpoint_interest() {
    let (engine, state) = engine();
    let (_source, source_rx) = submit(&engine, &[11, 12, 13, 14, 21]).await;
    let (_target, target_rx) = submit(&engine, &[11, 12, 13, 14, 22]).await;
    engine.inner.run_iteration().await.unwrap();
    assert_eq!(engine.inner.scheduler.prefix_held_waiting_count(), 1);
    let mut wake = tokio_test::task::spawn(engine.inner.wait_for_prefix_deadline());
    tokio_test::assert_pending!(wake.poll());
    // select! cancels this borrowed future when another branch wins. It owns
    // neither a checkpoint pin nor an independently spawned timer task.
    drop(wake);
    engine.shutdown().await.unwrap();
    assert_eq!(engine.inner.scheduler.prefix_held_waiting_count(), 0);
    assert!(state
        .leases
        .lock()
        .iter()
        .all(|lease| lease.upgrade().is_none()));
    let mut absent = tokio_test::task::spawn(engine.inner.wait_for_prefix_deadline());
    tokio_test::assert_pending!(absent.poll());
    tokio::time::advance(Duration::from_secs(60)).await;
    tokio_test::assert_pending!(absent.poll());
    drop(absent);
    drop((source_rx, target_rx));
}

#[tokio::test]
async fn prefix_rendezvous_preserves_materialization_but_releases_pins_on_real_pressure() {
    for pressure in [false, true] {
        let (engine, state) = engine();
        state.admission_pressure.store(true, Ordering::Relaxed);
        state.initial_materialization.store(true, Ordering::Relaxed);
        state.wait_for_release.store(pressure, Ordering::Relaxed);
        let (source, source_rx) = submit(&engine, &[11, 12, 13, 14, 21]).await;
        let (target, target_rx) = submit(&engine, &[11, 12, 13, 14, 22]).await;
        finish(&engine).await;
        assert!(source_rx.await.unwrap().is_ok());
        assert!(target_rx.await.unwrap().is_ok());
        let events = state.events.lock();
        assert!(events.contains(&Event::Maintained(source)));
        assert!(events.contains(&Event::Maintained(target.clone())));
        let first_offset = events
            .iter()
            .find_map(|event| match event {
                Event::Computed(id, start, _) if id == &target => Some(*start),
                _ => None,
            })
            .unwrap();
        assert_eq!(first_offset, if pressure { 0 } else { 4 });
        assert_eq!(events.contains(&Event::Acknowledged(target, 4)), !pressure);
        drop(events);
        assert!(state
            .leases
            .lock()
            .iter()
            .all(|lease| lease.upgrade().is_none()));
        engine.shutdown().await.unwrap();
    }
}
