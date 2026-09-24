//! Real controller captures against the shared fixed calibration fixture.
use super::*;
use crate::continuous_engine::inner::{
    cost_observation::*, prefill_reference_runtime::ReferenceBindingUnknown,
};
use ferrum_interfaces::engine::InferenceEngine;
use ferrum_scheduler::implementations::continuous::prefill_reference::ReferenceUnknown;
use std::sync::atomic::AtomicU64;

async fn request_at(
    engine: &ContinuousBatchEngine,
    tokens: usize,
    ingress: Instant,
) -> (RequestId, CreditedOutputSession) {
    let mut request = ferrum_types::InferenceRequest::new(
        vec!["test"; tokens].join(" "),
        engine.inner.config.model.model_id.clone(),
    );
    request.stream = true;
    request.sampling_params.max_tokens = 8;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    let id = request.id.clone();
    let session = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::from_ingress(ingress),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    ready(engine, &id).await;
    (id, session)
}

async fn admit(engine: &ContinuousBatchEngine, maximum: usize) {
    let _iteration = engine.inner.iteration_lock.lock().await;
    assert!(engine
        .inner
        .prepare_slo_admission_turn(maximum)
        .await
        .unwrap());
}
// Negative reference tests stop before resource capture. Keep their explicit
// error assertion; successful captures below use the shared typed retry helper.
fn capture(engine: &ContinuousBatchEngine) -> ControllerResult<ControllerSnapshot> {
    engine.inner.capture_slo_controller_snapshot(
        &ferrum_interfaces::BatchHint::simple(8),
        ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap(),
    )
}
fn progress<'a>(snapshot: &'a ControllerSnapshot, id: &RequestId) -> &'a PrefillProgressView {
    let row = snapshot
        .snapshot
        .requests
        .iter()
        .find(|row| row.key.request_id == *id)
        .unwrap();
    let RequestPhaseView::Prefill(progress) = &row.phase else {
        panic!("expected prefill")
    };
    progress
}

#[tokio::test]
async fn snapshot_uses_bound_exact_curves_and_fixed_tau_for_one_two_four_tokens() {
    let (engine, _, executor) = fixture_with_width(3).await;
    let mut sessions = Vec::new();
    let mut requests = Vec::new();
    for length in [1, 2, 4] {
        let (id, session) = request_at(&engine, length, slo_clock_now()).await;
        requests.push((id, length));
        sessions.push(session);
    }
    admit(&engine, 3).await;
    let snapshot = prefill::captured(&engine, &executor).await;
    let reference = engine
        .inner
        .prefill_reference_runtime
        .as_ref()
        .unwrap()
        .calibration();
    assert_eq!(
        snapshot.snapshot.scope.reference_decode_token_ns,
        reference.tau_ref_ns()
    );
    assert_eq!(reference.tau_ref_ns().get(), 7);
    assert_eq!(
        snapshot.snapshot.scope.reference_work_version,
        reference.identity().revision.get()
    );
    for (id, length) in requests {
        let progress = progress(&snapshot, &id);
        assert_eq!(progress.total_prompt_tokens.get() as usize, length);
        assert_eq!(progress.logical_high_water, 0);
        assert_eq!(
            progress
                .reference
                .points
                .iter()
                .map(|point| point.prompt_tokens)
                .collect::<Vec<_>>(),
            (0..=length as u32).collect::<Vec<_>>()
        );
        assert!(Arc::ptr_eq(
            &progress.reference,
            &reference
                .curve(NonZeroU32::new(length as u32).unwrap())
                .unwrap()
        ));
        let sequence = engine.inner.sequences.read();
        let binding = sequence[&id]
            .prefill_reference
            .as_ref()
            .unwrap()
            .known()
            .unwrap();
        assert_eq!(binding.identity(), reference.identity());
        assert_eq!(
            snapshot
                .origin
                .instant_at_ns(progress.admitted_at_ns)
                .unwrap(),
            snapshot
                .origin
                .instant_at_ns(
                    binding
                        .project_progress(&snapshot.origin, 0, length as u32, &[])
                        .unwrap()
                        .admitted_at_ns
                )
                .unwrap()
        );
    }
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    drop(snapshot);
    drop(sessions);
    engine.shutdown().await.unwrap();
}

struct SampleClock(AtomicU64);
impl CostObservationClock for SampleClock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.0.load(Ordering::Relaxed))
    }
}
fn publish_another_online_sample(runtime: &EngineCostRuntime) {
    let id = RequestId::new();
    let clock = Arc::new(SampleClock(AtomicU64::new(2)));
    let mut call = EngineCostCall::begin(
        &runtime.ids,
        clock.clone(),
        runtime.sink.clone(),
        EngineCostCallSpec {
            identity: runtime.identity.clone(),
            participants: vec![CostObservationParticipant {
                request_id: id.clone(),
                owner_incarnation: 1,
                work_generation: 1,
                input_index: 0,
                output_policy_signature: Some([6; 32]),
                host_features: None,
            }],
            prepare_started_at_ns: Some(1),
            boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
            recorder_limits: runtime.recorder_limits,
        },
    )
    .unwrap();
    {
        let mut context = call.context().unwrap();
        context.physical_wave(
            Ok(ActualWaveShape {
                statistical_evidence: None,
                kind: ActualWaveKind::Decode,
                path: ActualWavePath::PlanRuntime,
                graph: ActualWaveGraphState::Disabled,
                row_order: ActualWaveRowOrder::Ordered,
                provider_signature: [5; 32],
                output_policy_signature: [6; 32],
                numeric_features: None,
                host_content_features: None,
                row_multiset_features: None,
                rows: vec![ActualWaveRow {
                    request_id: id.clone(),
                    owner_incarnation: 1,
                    work_generation: 1,
                    input_index: 0,
                    work: ActualRowWork::Decode { kv_tokens: 1 },
                }],
                recurrent_state_bytes: 0,
                restore_bytes: 0,
                maintenance_bytes: 0,
                maintenance_units: 0,
            }),
            Some(3),
        );
        clock.0.store(6, Ordering::Relaxed);
        context.terminal(ActualWaveOutcome::Completed, None);
        context.finish_call(ObservedCallOutcome::Completed);
    }
    call.record_host_result(HostCommitEvidence {
        request_id: id,
        owner_incarnation: 1,
        work_generation: 1,
        input_index: 0,
        outcome: HostCommitOutcome::Committed(HostCommittedWork::Decode {
            kv_tokens_before: 1,
            kv_tokens_after: 2,
            generated_tokens_before: 1,
            generated_tokens_after: 2,
        }),
        committed_at_ns: Some(9),
    });
    clock.0.store(10, Ordering::Relaxed);
    assert_eq!(call.finish(), CostCallDisposition::Published);
    runtime.consume_samples();
}

#[tokio::test]
async fn online_publish_and_actual_partial_commit_preserve_the_reference_epoch() {
    let (engine, scheduler, executor) = fixture().await;
    let (id, session) = request_at(&engine, 4, slo_clock_now()).await;
    admit(&engine, 1).await;
    let before = prefill::captured(&engine, &executor).await;
    let original_curve = progress(&before, &id).reference.clone();
    let original_admission = before
        .origin
        .instant_at_ns(progress(&before, &id).admitted_at_ns)
        .unwrap();
    let runtime = engine.inner.cost_runtime.as_ref().unwrap();
    publish_another_online_sample(runtime);
    let updated = prefill::captured(&engine, &executor).await;
    assert!(updated.snapshot.cost_model_version > before.snapshot.cost_model_version);
    assert_eq!(
        updated.snapshot.scope.reference_work_version,
        before.snapshot.scope.reference_work_version
    );
    assert_eq!(
        updated.snapshot.scope.reference_decode_token_ns,
        before.snapshot.scope.reference_decode_token_ns
    );
    assert!(Arc::ptr_eq(
        &progress(&updated, &id).reference,
        &original_curve
    ));
    drop(before);
    drop(updated);
    let mut hint = ferrum_interfaces::BatchHint::simple(1);
    hint.max_tokens = 2;
    let batch = scheduler.next_batch(hint).await.unwrap();
    engine.inner.process_batch(&batch).await.unwrap();
    let partial = prefill::captured(&engine, &executor).await;
    let work = progress(&partial, &id);
    assert_eq!((work.offset, work.logical_high_water), (2, 2));
    assert_eq!(work.reference_work_at_admission_ns, 0);
    assert!(Arc::ptr_eq(&work.reference, &original_curve));
    assert_eq!(
        partial.origin.instant_at_ns(work.admitted_at_ns).unwrap(),
        original_admission
    );
    let sequences = engine.inner.sequences.read();
    assert!(sequences[&id].generated_tokens.is_empty());
    assert_eq!(
        sequences[&id]
            .prefill_reference
            .as_ref()
            .unwrap()
            .known()
            .unwrap()
            .binding()
            .logical_high_water(),
        work.logical_high_water
    );
    drop(sequences);
    drop(partial);
    cleanup(engine, session).await;
}

// The same physical release and public scheduler transition used by capacity
// deferral. No frontier, admission timestamp or reference state is fabricated.
async fn release_for_recompute(engine: &ContinuousBatchEngine, id: &RequestId) {
    let resources = engine
        .inner
        .sequences
        .write()
        .get_mut(id)
        .unwrap()
        .take_physical_resources_for_recompute();
    engine
        .inner
        .release_sequence_physical_resources(id, resources)
        .await;
}

async fn run_prefill_chunk(
    engine: &ContinuousBatchEngine,
    scheduler: &ContinuousBatchScheduler,
    id: &RequestId,
    offset: usize,
    count: usize,
) {
    let _iteration = engine.inner.iteration_lock.lock().await;
    let mut hint = ferrum_interfaces::BatchHint::simple(1);
    hint.max_tokens = count;
    let batch = scheduler.next_batch(hint).await.unwrap();
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(&batch.requests[0].request.id, id);
    assert_eq!(batch.requests[0].tokens_processed, offset);
    assert_eq!(batch.requests[0].tokens_to_process, Some(count));
    engine.inner.process_batch(&batch).await.unwrap();
}

#[tokio::test]
async fn partial_recompute_preserves_reference_credit_through_real_readmission_and_replay() {
    let (engine, scheduler, executor) = fixture().await;
    let (id, session) = request_at(&engine, 4, slo_clock_now()).await;
    admit(&engine, 1).await;
    run_prefill_chunk(&engine, &scheduler, &id, 0, 2).await;
    let before = prefill::captured(&engine, &executor).await;
    let original = progress(&before, &id);
    assert_eq!((original.offset, original.logical_high_water), (2, 2));
    let curve = original.reference.clone();
    let admitted = before
        .origin
        .instant_at_ns(original.admitted_at_ns)
        .unwrap();
    let milestones = original
        .milestones
        .iter()
        .map(|point| {
            (
                before.origin.instant_at_ns(point.at_ns).unwrap(),
                point.required_reference_work_ns,
            )
        })
        .collect::<Vec<_>>();
    let scope = before.snapshot.scope.clone();
    drop(before);

    {
        let _iteration = engine.inner.iteration_lock.lock().await;
        release_for_recompute(&engine, &id).await;
        assert!(scheduler.defer_prefill_to_waiting(&id));
    }
    let waiting = prefill::captured(&engine, &executor).await;
    let row = &waiting.queue.requests()[0];
    assert_eq!(row.queue, PlanningQueueKind::Waiting);
    assert_eq!(row.recompute_target_tokens, Some(2));
    assert_eq!(row.prompt_tokens, Some(4));
    assert_eq!((row.computed_tokens, row.prefill_offset), (0, 0));
    assert!(!row.readiness.ready());
    assert_eq!(
        (
            progress(&waiting, &id).offset,
            progress(&waiting, &id).logical_high_water
        ),
        (0, 2)
    );
    assert_eq!(progress(&waiting, &id).total_prompt_tokens.get(), 4);
    drop(waiting);

    // Real partial-prefill admission clears the old physical target of two;
    // it does not turn it into a two-token prompt or bind a new reference.
    admit(&engine, 1).await;
    let readmitted = prefill::captured(&engine, &executor).await;
    let row = &readmitted.queue.requests()[0];
    assert_eq!(row.queue, PlanningQueueKind::Prefill);
    assert_eq!(row.recompute_target_tokens, None);
    assert_eq!(row.prompt_tokens, Some(4));
    assert!(row.readiness.ready());
    drop(readmitted);

    for (offset, count) in [(0, 1), (1, 1)] {
        run_prefill_chunk(&engine, &scheduler, &id, offset, count).await;
        let replayed = prefill::captured(&engine, &executor).await;
        let work = progress(&replayed, &id);
        assert_eq!(work.offset as usize, offset + count);
        assert_eq!(
            work.logical_high_water, 2,
            "replay cannot earn duplicate credit"
        );
        assert_eq!(work.reference_work_at_admission_ns, 0);
        assert!(Arc::ptr_eq(&work.reference, &curve));
        assert_eq!(
            replayed.origin.instant_at_ns(work.admitted_at_ns).unwrap(),
            admitted
        );
        assert_eq!(
            work.milestones
                .iter()
                .map(|point| (
                    replayed.origin.instant_at_ns(point.at_ns).unwrap(),
                    point.required_reference_work_ns,
                ))
                .collect::<Vec<_>>(),
            milestones
        );
        assert_eq!(
            replayed.snapshot.scope.reference_decode_token_ns,
            scope.reference_decode_token_ns
        );
        assert_eq!(
            replayed.snapshot.scope.reference_work_version,
            scope.reference_work_version
        );
        assert!(engine.inner.sequences.read()[&id]
            .generated_tokens
            .is_empty());
    }

    run_prefill_chunk(&engine, &scheduler, &id, 2, 2).await;
    let sequences = engine.inner.sequences.read();
    let sequence = &sequences[&id];
    assert_eq!(sequence.prefill_tokens_processed, 4);
    assert_eq!(sequence.generated_tokens.len(), 1);
    let binding = sequence
        .prefill_reference
        .as_ref()
        .unwrap()
        .known()
        .unwrap();
    assert_eq!(binding.binding().logical_high_water(), 4);
    assert!(binding.binding().first_token_committed());
    assert!(Arc::ptr_eq(binding.binding().reference(), &curve));
    drop(sequences);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn decode_recompute_with_expanded_context_does_not_reuse_original_prompt_curve() {
    let (engine, scheduler, _) = fixture().await;
    let (id, session) = request_at(&engine, 4, slo_clock_now()).await;
    admit(&engine, 1).await;
    run_prefill_chunk(&engine, &scheduler, &id, 0, 4).await;
    let (curve, admission) = {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert_eq!(sequence.generated_tokens.len(), 1);
        assert_eq!(sequence.prefill_context_len(), 5);
        let binding = sequence
            .prefill_reference
            .as_ref()
            .unwrap()
            .known()
            .unwrap();
        (
            binding.binding().reference().clone(),
            binding.binding().admitted_at_ns(),
        )
    };
    {
        let _iteration = engine.inner.iteration_lock.lock().await;
        release_for_recompute(&engine, &id).await;
        assert!(scheduler.defer_decode_to_waiting_for_capacity(&id, 1));
    }
    // The actual engine context now includes the previously committed token.
    // Neither the waiting target nor readmission may stretch the original N4
    // curve to cover that fifth token.
    for readmitted in [false, true] {
        if readmitted {
            admit(&engine, 1).await;
        }
        let error = match capture(&engine) {
            Ok(_) => panic!("expanded recompute context acquired a false reference"),
            Err(error) => error,
        };
        assert_eq!(error.reason, "reference_frontier_mismatch");
        assert_eq!(error.obligations, 1);
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert_eq!(sequence.prefill_tokens_processed, 0);
        assert_eq!(sequence.prefill_context_len(), 5);
        let binding = sequence
            .prefill_reference
            .as_ref()
            .unwrap()
            .known()
            .unwrap();
        assert_eq!(binding.total_prompt_tokens().get(), 4);
        assert_eq!(binding.binding().logical_high_water(), 4);
        assert_eq!(binding.binding().admitted_at_ns(), admission);
        assert!(Arc::ptr_eq(binding.binding().reference(), &curve));
    }
    cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn removing_earlier_ingress_changes_origin_without_resetting_survivor_milestones() {
    let (engine, _, executor) = fixture_with_width(2).await;
    let now = slo_clock_now();
    let (earlier, first_session) = request_at(&engine, 2, now - Duration::from_secs(2)).await;
    let (later, second_session) = request_at(&engine, 4, now - Duration::from_secs(1)).await;
    admit(&engine, 2).await;
    let before = prefill::captured(&engine, &executor).await;
    let old = progress(&before, &later);
    let admitted = before.origin.instant_at_ns(old.admitted_at_ns).unwrap();
    let milestones = old
        .milestones
        .iter()
        .map(|point| {
            (
                before.origin.instant_at_ns(point.at_ns).unwrap(),
                point.required_reference_work_ns,
            )
        })
        .collect::<Vec<_>>();
    let curve = old.reference.clone();
    let old_admitted_ns = old.admitted_at_ns;
    drop(before);
    engine
        .inner
        .complete_request_with_error(&earlier, FerrumError::cancelled("remove earlier ingress"))
        .await
        .unwrap();
    drop(first_session);
    let after = prefill::captured(&engine, &executor).await;
    assert_eq!(after.snapshot.requests.len(), 1);
    let new = progress(&after, &later);
    assert_eq!(old_admitted_ns - new.admitted_at_ns, 1_000_000_000);
    assert_eq!(
        after.origin.instant_at_ns(new.admitted_at_ns).unwrap(),
        admitted
    );
    assert_eq!(
        new.milestones
            .iter()
            .map(|point| (
                after.origin.instant_at_ns(point.at_ns).unwrap(),
                point.required_reference_work_ns
            ))
            .collect::<Vec<_>>(),
        milestones
    );
    assert!(Arc::ptr_eq(&new.reference, &curve));
    drop(after);
    cleanup(engine, second_session).await;
}

#[tokio::test]
async fn uncalibrated_length_is_missing_reference_even_with_live_cost_and_resource_evidence() {
    let (engine, _, executor) = fixture().await;
    let (id, session) = request_at(&engine, 3, slo_clock_now()).await;
    admit(&engine, 1).await;
    assert!(engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_some());
    assert!(matches!(
        engine.inner.sequences.read()[&id]
            .prefill_reference
            .as_ref()
            .unwrap()
            .known(),
        Err(ReferenceBindingUnknown::Reference(
            ReferenceUnknown::LengthNotCalibrated
        ))
    ));
    let unavailable = match capture(&engine) {
        Ok(_) => panic!("uncalibrated length became known"),
        Err(error) => error,
    };
    assert_eq!(unavailable.reason, "missing_reference_work");
    assert_eq!(unavailable.obligations, 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn typed_retry_waits_one_millisecond_without_changing_work_or_capacity() {
    let (mut engine, _, executor) = fixture().await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
    inner.config.scheduler.slo.planner.retry_backoff_ms = NonZeroU64::new(1).unwrap();
    let (id, session) = request_at(&engine, 4, slo_clock_now()).await;
    admit(&engine, 1).await;
    let before = prefill::captured(&engine, &executor).await;
    let credits = pool(&engine).snapshot();
    let owner = engine.inner.sequences.read()[&id]
        .stream_projection_identity
        .clone();
    let epochs = executor.execution_capacity_epochs().unwrap();
    engine
        .inner
        .arm_controller_retry(retry::ControllerRetryReason::SnapshotBusy);
    let mut waiting = Box::pin(engine.inner.wait_for_slo_controller_retry());
    assert!(waiting.as_mut().now_or_never().is_none());
    tokio::time::advance(Duration::from_micros(500)).await;
    assert!(waiting.as_mut().now_or_never().is_none());
    engine
        .inner
        .arm_controller_retry(retry::ControllerRetryReason::ComputeBudget);
    tokio::time::advance(Duration::from_micros(500)).await;
    assert!(waiting.as_mut().now_or_never().is_some());
    drop(waiting);
    assert!(engine.inner.slo_controller.lock().retry.is_none());
    let after = prefill::captured(&engine, &executor).await;
    assert_eq!(before.queue.requests(), after.queue.requests());
    assert!(Arc::ptr_eq(
        &owner,
        &engine.inner.sequences.read()[&id].stream_projection_identity
    ));
    assert_eq!(executor.execution_capacity_epochs().unwrap(), epochs);
    assert_eq!(pool(&engine).snapshot().data_used, credits.data_used);
    assert_eq!(
        pool(&engine).snapshot().terminal_held,
        credits.terminal_held
    );
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    drop(before);
    drop(after);
    cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn observe_retry_never_arms_an_extra_controller_iteration() {
    let (engine, _, executor) = fixture().await;
    let (id, session) = request_at(&engine, 4, slo_clock_now()).await;
    let owner = engine.inner.sequences.read()[&id]
        .stream_projection_identity
        .clone();
    engine
        .inner
        .arm_controller_retry(retry::ControllerRetryReason::ChangedEvidence);
    assert!(engine.inner.slo_controller.lock().retry.is_none());
    let mut waiting = Box::pin(engine.inner.wait_for_slo_controller_retry());
    assert!(waiting.as_mut().now_or_never().is_none());
    tokio::time::advance(Duration::from_millis(10)).await;
    assert!(waiting.as_mut().now_or_never().is_none());
    drop(waiting);
    assert!(Arc::ptr_eq(
        &owner,
        &engine.inner.sequences.read()[&id].stream_projection_identity
    ));
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn piecewise_loaded_reference_binds_unmeasured_input_in_real_snapshot() {
    let (mut engine, _, executor) = fixture_with_width(1).await;
    let runtime=crate::continuous_engine::inner::prefill_reference_runtime::test_piecewise_calibration_runtime();
    assert_eq!(
        runtime
            .calibration()
            .supported_lengths()
            .collect::<Vec<_>>(),
        vec![1, 4]
    );
    let identity = runtime.calibration().identity();
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .prefill_reference_runtime = Some(runtime);
    let ingress = slo_clock_now();
    let (id, session) = request_at(&engine, 3, ingress).await;
    admit(&engine, 1).await;
    let snapshot = prefill::captured(&engine, &executor).await;
    let work = progress(&snapshot, &id);
    assert_eq!(work.total_prompt_tokens.get(), 3);
    assert!(work.reference.work_at(2).unwrap() > work.reference.work_at(1).unwrap());
    assert_eq!(snapshot.snapshot.scope.reference_decode_token_ns.get(), 7);
    assert!(!snapshot
        .snapshot
        .capabilities
        .prefill_chunk_sizes
        .is_empty());
    let sequences = engine.inner.sequences.read();
    let binding = sequences[&id]
        .prefill_reference
        .as_ref()
        .unwrap()
        .known()
        .unwrap();
    assert_eq!(binding.identity(), identity);
    assert_eq!(
        snapshot
            .origin
            .instant_at_ns(snapshot.snapshot.requests[0].timing.ingress_at_ns)
            .unwrap(),
        ingress
    );
    drop(sequences);
    drop(session);
    engine.shutdown().await.unwrap();
}
