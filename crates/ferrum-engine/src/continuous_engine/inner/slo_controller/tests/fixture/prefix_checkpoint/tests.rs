//! Real CPU checkpoint evidence through the executor boundary and original FIFO.
//! Portable receipt and ordinary cold-loop gates. Qualified hold is separate.
mod frontier;
mod host_lengths;
mod lifecycle;
mod ordinary;
mod qualified;
use super::*;
use ferrum_scheduler::implementations::continuous::{
    cost_model as model,
    slo_planner::{
        PlanningCostClockAnchor, PlanningPrefixCostModel, PrefixMaintenanceStage,
        PrefixRendezvousOffer, RequestWorkKey,
    },
};
use std::num::NonZeroU32;
use vnext::{NativeCheckpointTransferCostDomain as Domain, NativeCheckpointTransferKind as Kind};

struct AllowHost;
impl NonblockingHostSubmissionGuard for AllowHost {
    fn check(&self) -> std::result::Result<(), HostSubmissionRejection> {
        Ok(())
    }
}

#[derive(Default)]
struct Guard {
    reject: AtomicBool,
    calls: AtomicUsize,
    expected: Option<Domain>,
}
impl vnext::CheckpointTransferSubmissionGuard for Guard {
    fn check(
        &self,
        actual: &vnext::PreparedCheckpointTransfer<'_>,
    ) -> std::result::Result<(), GuardedNotSubmittedReason> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        if self.reject.load(Ordering::Acquire) {
            return Err(GuardedNotSubmittedReason::HostRejected(
                HostSubmissionRejection::Cancelled,
            ));
        }
        if self
            .expected
            .as_ref()
            .is_some_and(|expected| expected != actual.cost_domain())
        {
            return Err(GuardedNotSubmittedReason::ActualRouteMismatch);
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Observed {
    identity: vnext::NativeCheckpointTransferIdentity,
    source: Option<vnext::NativeCheckpointTransferIdentity>,
    domain: Domain,
    host_work: Option<vnext::NativeCheckpointTransferHostWork>,
    wall: Duration,
}
struct Tap {
    records: Mutex<Vec<Observed>>,
    on_restore: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    forward: Option<Arc<dyn vnext::NativeCheckpointObservationSink>>,
}
impl Tap {
    fn new(forward: Option<Arc<dyn vnext::NativeCheckpointObservationSink>>) -> Arc<Self> {
        Arc::new(Self {
            records: Mutex::new(Vec::with_capacity(128)),
            on_restore: Mutex::new(None),
            forward,
        })
    }
}
impl vnext::NativeCheckpointObservationSink for Tap {
    fn try_record(&self, observation: vnext::NativeCheckpointTransferObservation) -> bool {
        if observation.cost_domain().kind() == Kind::Restore {
            if let Some(check) = self.on_restore.lock().as_ref() {
                check();
            }
        }
        let Some(mut records) = self.records.try_lock() else {
            return false;
        };
        if records.len() == records.capacity() {
            return false;
        }
        records.push(Observed {
            identity: observation.identity().clone(),
            source: observation.source_capture_identity().cloned(),
            domain: observation.cost_domain().clone(),
            host_work: observation.host_work().copied(),
            wall: observation.wall_elapsed(),
        });
        drop(records);
        self.forward
            .as_ref()
            .map_or(true, |sink| sink.try_record(observation))
    }
}
fn install(executor: &ControlledExecutor, tap: &Arc<Tap>) {
    let erased: Arc<dyn vnext::NativeCheckpointObservationSink> = tap.clone();
    assert!(executor
        .install_checkpoint_observation_sink(Arc::downgrade(&erased))
        .unwrap());
}
fn known<T>(result: ExecutionCostRouteAvailability<T>) -> T {
    match result {
        ExecutionCostRouteAvailability::Known(value) => value,
        ExecutionCostRouteAvailability::Unknown(reason) => {
            panic!("real CPU checkpoint evidence: {reason:?}")
        }
    }
}
fn view(
    executor: &ControlledExecutor,
    rows: &[(&RequestId, Option<&str>)],
) -> ExecutionCostRouteView {
    let requests = rows
        .iter()
        .map(|(id, cache)| ExecutorResourcePlanningRequest {
            request_id: id,
            cache_id: *cache,
        })
        .collect::<Vec<_>>();
    known(executor.execution_cost_route_view(
        &requests,
        ResourcePlanningLimits::default(),
        &mut || true,
    ))
}
fn input() -> PlanRuntimePrefillInput {
    PlanRuntimePrefillInput::new(
        RequestId::new(),
        Arc::<[ferrum_types::TokenId]>::from(vec![ferrum_types::TokenId::new(5); 4]),
        8,
        PrefillChunk::new(0, 2, 4).unwrap(),
    )
    .unwrap()
}
fn admit(executor: &ControlledExecutor, input: &PlanRuntimePrefillInput) {
    assert!(matches!(
        executor
            .try_admit_prefill(ExecutorPrefillAdmission::for_diagnostic(
                &input.request_id,
                &input.input_tokens,
                input.maximum_sequence_tokens
            ))
            .unwrap(),
        ExecutorPrefillAdmissionDecision::Admitted(_)
    ));
}
fn partial(
    executor: &ControlledExecutor,
    input: &PlanRuntimePrefillInput,
) -> PlanRuntimePrefillCompletion {
    admit(executor, input);
    view(executor, &[(&input.request_id, None)]);
    let mut output = executor
        .native_structured_output(None, std::slice::from_ref(input), &[], &AllowHost)
        .expect("native opt-in")
        .unwrap();
    executor
        .prefill_output_from_logits(input, output.remove(0))
        .unwrap()
}
fn request(input: &PlanRuntimePrefillInput) -> PrefixCaptureRequest<'_> {
    PrefixCaptureRequest {
        source_request_id: &input.request_id,
        source_tokens: &input.input_tokens,
        maximum_sequence_tokens: input.maximum_sequence_tokens,
        boundary: input.chunk.end(),
        expires_at: Instant::now() + Duration::from_secs(30),
    }
}
fn capture(
    executor: &ControlledExecutor,
    input: &PlanRuntimePrefillInput,
    guard: &Guard,
) -> Arc<dyn PrefixCaptureLease> {
    let lease = executor
        .retain_prefix_capture_interest(request(input))
        .unwrap()
        .unwrap();
    for _ in 0..4 {
        if executor.prefix_capture(request(input), guard).unwrap() {
            assert_eq!(lease.status(), PrefixCaptureStatus::Ready);
            return lease;
        }
    }
    panic!("native capture did not become ready after original capacity maintenance");
}
fn restored(
    executor: &ControlledExecutor,
    target: &PlanRuntimePrefillInput,
    lease: &dyn PrefixCaptureLease,
    guard: &Guard,
) -> PlanRuntimePrefixRestoreOutput {
    admit(executor, target);
    view(executor, &[(&target.request_id, None)]);
    for _ in 0..4 {
        match executor
            .prefix_restore(
                PlanRuntimePrefixRestoreInput {
                    request_id: &target.request_id,
                    input_tokens: &target.input_tokens,
                    maximum_sequence_tokens: target.maximum_sequence_tokens,
                    checkpoint: Some(lease),
                    retry: None,
                },
                guard,
            )
            .unwrap()
        {
            PlanRuntimePrefixRestoreOutcome::Restored(output) => return output,
            PlanRuntimePrefixRestoreOutcome::Unavailable => {}
            other => panic!("CPU restore deferred unexpectedly: {other:?}"),
        }
    }
    panic!("native restore did not become ready");
}
fn copies(executor: &ControlledExecutor) -> Vec<Vec<u8>> {
    let registry = executor
        .evidence
        .fixture
        .as_ref()
        .unwrap()
        .runtime_trace
        .lock()
        .unwrap()
        .memory
        .as_ref()
        .unwrap()
        .clone();
    let result = registry.lock().unwrap().copied_payloads().to_vec();
    result
}
fn fingerprint(executor: &ControlledExecutor) -> model::ExecutionFingerprint {
    let ExecutorCostIdentityAvailability::Known(identity) = executor.execution_cost_identity()
    else {
        panic!("fixture executor identity")
    };
    model::ExecutionFingerprint {
        model_weights: identity.model_weights,
        numerical_policy: identity.numerical_policy,
        device_runtime: identity.device_runtime,
        execution_config: identity.execution_config,
    }
}
fn offer(source: &RequestId, target: &RequestId) -> PrefixRendezvousOffer {
    let key = |id: &RequestId| RequestWorkKey {
        request_id: id.clone(),
        incarnation: 1,
        work_generation: Default::default(),
    };
    PrefixRendezvousOffer {
        identity: [7; 32],
        based_on_generation: 1,
        producer: key(source),
        target: key(target),
        boundary_tokens: NonZeroU32::new(2).unwrap(),
        expires_at_ns: 10_000_000_000,
    }
}
fn predict(
    runtime: &EngineCostRuntime,
    executor: &ControlledExecutor,
    receipt: &Observed,
    offer: &PrefixRendezvousOffer,
) -> bool {
    let Some(snapshot) = runtime.prefix_cost_snapshot() else {
        return false;
    };
    let domain = &receipt.domain;
    snapshot
        .anchored(PlanningCostClockAnchor::exact(100, 100), offer)
        .predict(
            &fingerprint(executor),
            offer,
            match domain.kind() {
                Kind::Capture => PrefixMaintenanceStage::Capture,
                Kind::Restore => PrefixMaintenanceStage::Restore,
            },
            &prefix_cost_shape(domain, receipt.host_work.as_ref()).unwrap(),
            100,
        )
        .is_some()
}

#[tokio::test]
async fn native_prefix_cpu_bytes_guard_and_publication_ack() {
    let (_, executor) = startup_checkpoint_components(3).await;
    let tap = Tap::new(None);
    install(&executor, &tap);
    let source = input();
    let source_output = partial(&executor, &source);
    let target = input();
    admit(&executor, &target);
    view(&executor, &[(&target.request_id, None)]);
    let guard = Guard {
        reject: AtomicBool::new(true),
        ..Default::default()
    };
    let before = executor.native_structured_counts().0;
    for _ in 0..4 {
        assert!(!executor.prefix_capture(request(&source), &guard).unwrap());
        if guard.calls.load(Ordering::Acquire) > 0 {
            break;
        }
    }
    assert!(
        guard.calls.load(Ordering::Acquire) > 0,
        "reject must reach actual final native guard"
    );
    assert_eq!(executor.native_structured_counts().0, before);
    assert!(copies(&executor).is_empty());
    assert!(tap.records.lock().is_empty());

    guard.reject.store(false, Ordering::Release);
    let lease = capture(&executor, &source, &guard);
    assert_eq!(copies(&executor), vec![2u32.to_le_bytes().to_vec()]);
    let captured = tap.records.lock()[0].clone();
    assert_eq!(captured.domain.kind(), Kind::Capture);
    assert!(captured.wall > Duration::ZERO);
    assert_eq!(captured.domain.geometry().copy_bytes(), 4);
    assert_eq!(captured.domain.geometry().initialization_bytes(), 0);

    // The fresh numeric replay uses the same plan, source, real retention and
    // target allocation evidence; this metadata grants no native submission.
    let cache = source_output.output().kv_cache().cache_id();
    let fresh = view(
        &executor,
        &[
            (&source.request_id, Some(&cache)),
            (&target.request_id, None),
        ],
    );
    let bound = known(executor.bind_execution_retained_checkpoint(
        &fresh,
        &fresh.initial_state(),
        lease.as_ref(),
        0,
        &mut || true,
    ));
    let projected = known(executor.project_execution_checkpoint(
        &fresh,
        &bound.state,
        vnext::FutureCheckpointCostQuery::Restore {
            checkpoint: &bound.checkpoint,
            target: 1,
            prompt_tokens: 4,
        },
        &mut || true,
    ));
    let restore_guard = Guard {
        expected: Some(projected.cost_domain.clone()),
        ..Default::default()
    };
    let output = restored(&executor, &target, lease.as_ref(), &restore_guard);
    assert_eq!(
        tap.records.lock().len(),
        1,
        "native outbox alone cannot publish a Restore sample"
    );
    assert_eq!(copies(&executor), vec![2u32.to_le_bytes().to_vec(); 2]);
    let observed_history = executor.native_structured_history.clone();
    let restored_request = target.request_id.clone();
    let observed_ready = Arc::new(AtomicBool::new(false));
    let observed_ready_at_ack = Arc::clone(&observed_ready);
    *tap.on_restore.lock() = Some(Box::new(move || {
        let history = observed_history
            .try_lock()
            .expect("full ack holds no history lock");
        assert_eq!(
            history.get(&restored_request).map(Vec::as_slice),
            Some(&[5, 5][..]),
            "a Restore receipt must already expose its actual committed history"
        );
        observed_ready_at_ack.store(true, Ordering::Release);
    }));
    let authority = output.acknowledge().unwrap();
    assert!(observed_ready.load(Ordering::Acquire));
    assert_eq!(authority.committed_tokens(), 2);
    let records = tap.records.lock();
    assert_eq!(records.len(), 2);
    assert!(records[1]
        .source
        .as_ref()
        .unwrap()
        .same_transfer(&captured.identity));
    assert_eq!(records[1].domain, projected.cost_domain);
    assert_eq!(records[1].host_work, Some(projected.host_work));
    assert_eq!(projected.host_work.prefix_tokens(), 2);
    assert_eq!(projected.host_work.full_input_tokens(), 4);
    assert!(prefix_cost_shape(&projected.cost_domain, None).is_err());
    drop(records);

    let target_cache = authority.kv_cache().cache_id();
    let after = view(
        &executor,
        &[
            (&source.request_id, Some(&cache)),
            (&target.request_id, Some(&target_cache)),
        ],
    );
    assert!(known(executor.execution_checkpoint_restore_completed(
        &after,
        lease.as_ref(),
        1,
        &mut || true
    )));
    let suffix = PlanRuntimePrefillInput::new(
        target.request_id.clone(),
        target.input_tokens.clone(),
        8,
        PrefillChunk::new(2, 2, 4).unwrap(),
    )
    .unwrap();
    let mut logits = executor
        .native_structured_output(None, &[suffix.clone()], &[], &AllowHost)
        .unwrap()
        .unwrap();
    let completed = executor
        .prefill_output_from_logits(&suffix, logits.remove(0))
        .unwrap();
    assert_eq!(completed.output().kv_cache().num_tokens(), 4);

    let cancelled = input();
    admit(&executor, &cancelled);
    view(&executor, &[(&cancelled.request_id, None)]);
    let pending = executor
        .retain_prefix_capture_interest(request(&cancelled))
        .unwrap()
        .unwrap();
    assert!(executor.cancel_prefill_admission(&cancelled.request_id));
    assert_eq!(pending.status(), PrefixCaptureStatus::Unavailable);
    let before = executor.native_structured_counts();
    assert!(!executor
        .prefix_capture(request(&cancelled), &guard)
        .unwrap());
    assert_eq!(executor.native_structured_counts(), before);
    assert_eq!(tap.records.lock().len(), 2);
}

#[tokio::test]
async fn native_prefix_cpu_receipts_train_original_fifo_from_cold_unknown() {
    let mut config = ferrum_types::SloCostObservationConfig::default();
    // A declared two-entry FIFO permits each real Capture/Restore pair. The
    // final phase exercises actual overflow without fabricating a receipt.
    config.max_queued_samples = NonZeroUsize::new(2).unwrap();
    config.max_samples_per_update = config.max_queued_samples;
    let samples = config.model.min_samples.get();
    assert!(samples * 2 + 3 <= 64);
    let (_, executor) = startup_checkpoint_components(samples * 2 + 3).await;
    let runtime = EngineCostRuntime::build(
        executor.execution_cost_identity(),
        Arc::new(Clock(AtomicU64::new(100))),
        &config,
        false,
    )
    .unwrap();
    assert!(runtime.snapshot().is_none());
    assert!(runtime.prefix_cost_snapshot().is_none());
    let tap = Tap::new(Some(runtime.prefix_cost_sink().unwrap()));
    install(&executor, &tap);
    let mut retained = Vec::new();
    let guard = Guard::default();
    let mut prior = None;
    for index in 0..samples {
        let source = input();
        let source_output = partial(&executor, &source);
        let target = input();
        let lease = capture(&executor, &source, &guard);
        let offer = offer(&source.request_id, &target.request_id);
        let capture_domain = tap.records.lock().last().unwrap().clone();
        if index == 0 {
            assert!(!predict(&runtime, &executor, &capture_domain, &offer));
        }
        let restored = restored(&executor, &target, lease.as_ref(), &guard);
        let before_ack = tap.records.lock().len();
        assert_eq!(before_ack, index * 2 + 1);
        let authority = restored.acknowledge().unwrap();
        let restore_domain = tap.records.lock().last().unwrap().clone();
        runtime.drain_calibration_fixture();
        if let Some(old) = prior.take() {
            let old: Arc<PrefixCostSnapshot> = old;
            assert!(
                !old.current(),
                "actual new receipts revoke the old maintenance revision"
            );
        }
        assert_eq!(
            predict(&runtime, &executor, &capture_domain, &offer),
            index + 1 >= samples
        );
        assert_eq!(
            predict(&runtime, &executor, &restore_domain, &offer),
            index + 1 >= samples
        );
        assert!(
            runtime.snapshot().is_none(),
            "maintenance receipts cannot qualify inference"
        );
        prior = runtime.prefix_cost_snapshot();
        retained.push((source_output, authority, lease));
    }
    assert_eq!(tap.records.lock().len(), samples * 2);
    assert_eq!(
        copies(&executor),
        vec![2u32.to_le_bytes().to_vec(); samples * 2]
    );
    let stats = runtime.sink.stats();
    assert_eq!(stats.entries_dropped_contention, 0);
    assert_eq!(stats.entries_dropped_capacity, 0);
    assert_eq!(
        stats.offered, 0,
        "native maintenance must not enter inference population"
    );
    let published = prior.unwrap();
    assert!(published.current());
    let queued_source = input();
    let queued_output = partial(&executor, &queued_source);
    let queued_lease = capture(&executor, &queued_source, &guard);
    let queued_target = input();
    let queued_restore = restored(&executor, &queued_target, queued_lease.as_ref(), &guard)
        .acknowledge()
        .unwrap();
    let lost_source = input();
    let lost_output = partial(&executor, &lost_source);
    let lost_lease = capture(&executor, &lost_source, &guard);
    assert_eq!(
        lost_lease.status(),
        PrefixCaptureStatus::Ready,
        "observation loss cannot undo actual native/model publication"
    );
    assert!(!published.current());
    assert!(runtime.prefix_cost_snapshot().is_none());
    let after_loss = runtime.sink.stats();
    assert_eq!(
        after_loss.entries_dropped_capacity, stats.entries_dropped_capacity,
        "maintenance overflow cannot revoke the inference population"
    );
    drop((
        queued_output,
        queued_lease,
        queued_restore,
        lost_output,
        lost_lease,
    ));
    drop(retained);
}
