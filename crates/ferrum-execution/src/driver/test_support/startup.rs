//! Startup uses the same token path for real CPU lifecycle and CUDA reuse.

use super::*;

#[derive(Default)]
struct StartupEvents(Mutex<Vec<ExecutionEvent>>, DeviceTimingMode);

impl ExecutionEventSink for StartupEvents {
    fn device_timing_mode(&self) -> DeviceTimingMode {
        self.1
    }

    fn enablement(&self) -> ExecutionEventSinkEnablement {
        ExecutionEventSinkEnablement::All
    }

    fn is_enabled(&self, _: ExecutionEventKind) -> bool {
        true
    }

    fn record(
        &self,
        permit: EventEmissionPermit,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        self.0.lock().push(permit.into_event());
        Ok(())
    }
}

impl ProductExecutionEventSink for StartupEvents {}

async fn settle<R: DeviceRuntime>(executor: &ProgramExecutor<R>) {
    // Each job completes before the worker's recovery sweep. A second FIFO
    // marker proves the prior post-task sweep has had its actual turn.
    for _ in 0..2 {
        executor
            .completion_worker
            .execute(VNextCompletionTaskKind::PostSubmitDrain, || ())
            .await
            .unwrap();
    }
    assert_eq!(executor.sequences.lock().total_len(), 0);
    assert!(executor.sequences.lock().token_index.is_empty());
    assert_eq!(executor.reaper.retained_count(), 0);
    assert_eq!(executor.reaper.quarantined_count(), 0);
    assert_eq!(executor.lane.in_flight_count(), 0);
    for pool in executor
        .plan_resources
        .dynamic_pool_status()
        .unwrap()
        .pools()
    {
        assert_eq!(pool.live_occupancy().transient().total().claim_count(), 0);
        assert_eq!(
            pool.live_occupancy().transient().total().physical_bytes(),
            0
        );
    }
}

pub async fn native_startup_cpu_drop_after_prerequisite_closes_exact_incarnation<
    R: DeviceRuntime,
>(
    owned_executor: ProgramExecutor<R>,
) -> Arc<R> {
    let executor = &owned_executor;
    let (ready, ready_rx) = tokio::sync::oneshot::channel();
    let mut startup = Box::pin(async {
        let mut resources = VNextStartupSequenceGuard::new(executor);
        let tokens = [TokenId::new(0), TokenId::new(1), TokenId::new(2)];
        let index = executor
            .reserve_startup_sequence(&mut resources, &tokens, 16)
            .unwrap()
            .unwrap();
        let identity = resources.sequences[index].frontier.sequence().clone();
        let owner = Arc::downgrade(&resources.sequences[index].sequence);
        assert_eq!(
            resources.sequences[index].sequence.request_origin,
            ExecutorRequestOrigin::Startup
        );
        assert!(executor
            .execute_startup_prefill_chunk(
                &mut resources,
                index,
                PrefillChunk::new(0, 1, 3).unwrap(),
                "test prerequisite"
            )
            .await
            .unwrap());
        assert_eq!(resources.sequences[index].frontier.consumed_tokens(), 1);
        assert_eq!(
            resources.sequences[index]
                .sequence
                .prefill_tokens_processed
                .load(Ordering::Acquire),
            1
        );
        assert_eq!(
            executor.sequences.lock().active.len(),
            0,
            "intermediate prefill remains in its original slot"
        );
        ready.send((owner, identity)).unwrap();
        std::future::pending::<()>().await;
        resources.complete().unwrap();
    });
    let (owner, identity) = tokio::select! {
        result = ready_rx => result.unwrap(),
        () = &mut startup => panic!("startup unexpectedly completed before cancellation"),
    };
    assert!(owner.upgrade().is_some());
    drop(startup);
    settle(executor).await;
    assert!(owner.upgrade().is_none());
    assert!(
        identity.observed_end().is_some(),
        "only real session terminalization marks Ended"
    );
    assert_eq!(
        executor.metrics.prefill_operations.load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        executor
            .metrics
            .prefill_frontier_narrowings
            .load(Ordering::Relaxed),
        0
    );
    assert_executor_closed(owned_executor)
}

pub async fn native_startup_cpu_exact_chunks_decode_and_startup_completion<R: DeviceRuntime>(
    owned_executor: ProgramExecutor<R>,
) -> Arc<R> {
    let executor = &owned_executor;
    let sink = Arc::new(StartupEvents::default());
    executor.attach_execution_event_sink(sink.clone());

    // This is the production helper's actual prerequisite + final chunk path.
    executor
        .execute_startup_prefill_request(PrefillChunk::new(1, 2, 3).unwrap(), "test exact prefill")
        .await
        .unwrap();
    assert_eq!(
        executor.metrics.prefill_operations.load(Ordering::Relaxed),
        2
    );
    assert_eq!(
        executor
            .metrics
            .prefill_frontier_narrowings
            .load(Ordering::Relaxed),
        0
    );

    let mut resources = VNextStartupSequenceGuard::new(executor);
    assert!(executor
        .admit_startup_sequence(&mut resources, Arc::from([TokenId::new(0)]), 16)
        .await
        .unwrap());
    let identity = resources.sequences[0].frontier.sequence().clone();
    executor
        .execute_startup_decode_pass(&mut resources, TokenId::new(3), 1, "test startup decode")
        .await
        .unwrap();
    assert_eq!(resources.sequences[0].frontier.consumed_tokens(), 2);
    assert_eq!(
        resources.sequences[0].sequence.tokens.lock().as_slice(),
        &[0, 3]
    );
    resources.complete().unwrap();
    assert!(identity.observed_end().is_some());
    settle(executor).await;

    let events = sink.0.lock();
    assert!(!events.is_empty());
    for event in events.iter() {
        assert_eq!(
            ExecutorRequestOrigin::from_namespaced_request_identity(
                event.identity().parts().request_id.as_str()
            ),
            Some(ExecutorRequestOrigin::Startup),
            "startup must never emit product accounting"
        );
    }
    let completed: Vec<_> = events
        .iter()
        .filter(|event| event.kind() == ExecutionEventKind::RequestCompleted)
        .collect();
    assert_eq!(completed.len(), 2);
    assert!(matches!(
        completed[0].detail(),
        ExecutionEventDetail::Counters {
            input: 3,
            output: 0
        }
    ));
    assert!(matches!(
        completed[1].detail(),
        ExecutionEventDetail::Counters {
            input: 1,
            output: 0
        }
    ));
    drop(events);
    executor.reset_request_metrics_after_startup().unwrap();
    assert_eq!(
        executor.metrics.prefill_operations.load(Ordering::Relaxed),
        0
    );
    assert_eq!(
        executor.metrics.decode_operations.load(Ordering::Relaxed),
        0
    );
    assert_executor_closed(owned_executor)
}

/// Derive the actual minimum used by the upper layer's second CPU fixture.
/// This is diagnostic sizing, never an admission or submission authority.
pub fn startup_minimum_capacity<R: DeviceRuntime>(executor: &ProgramExecutor<R>) -> u64 {
    let memory = executor.resolved_plan.execution_plan().payload().memory();
    assert_eq!(memory.minimum_request_bytes(), 0);
    assert_eq!(memory.minimum_sequence_bytes(), 0);
    memory
        .static_bytes()
        .checked_add(memory.minimum_runnable_request_bytes())
        .unwrap()
}

pub async fn native_startup_cpu_unsubmitted_prefill_cannot_enter_capture<R: DeviceRuntime>(
    owned_executor: ProgramExecutor<R>,
    usable: u64,
) -> Arc<R> {
    let executor = &owned_executor;
    // Retain a complete real unsubmitted wave, including Invocation backing.
    // A bare Step leaves that working-set budget available to another Step.
    let mut competitor = VNextStartupSequenceGuard::new(executor);
    let tokens = [TokenId::new(0)];
    let competitor_index = executor
        .reserve_startup_sequence(&mut competitor, &tokens, 16)
        .unwrap()
        .unwrap();
    let competing_sequence = Arc::clone(&competitor.sequences[competitor_index].sequence);
    let span = TokenSpanWork::from_token_ids(&[0], 0..1).unwrap();
    executor
        .extend_sequence(
            &competing_sequence,
            ResourceWorkShape::single(span.clone()).unwrap(),
        )
        .unwrap();
    let batch =
        ExecutionBatchParticipants::new(vec![Arc::clone(&competing_sequence.session)]).unwrap();
    let VNextExecutionCapacityDecision::Ready(step) = executor
        .begin_step_for_spans_with_capacity(
            &batch,
            std::slice::from_ref(&competing_sequence),
            std::slice::from_ref(&span),
            VNextExecutionWaveKind::Prefill,
        )
        .unwrap()
    else {
        panic!("competing Step must acquire actual capacity");
    };
    let wave = executor
        .prepare_wave(&step, &competing_sequence, &span)
        .unwrap();
    let occupied = executor.plan_resources.dynamic_pool_status().unwrap();
    assert_eq!(
        occupied.budget_claimed_bytes(),
        usable,
        "actual complete working-set occupancy: {occupied:?}"
    );
    let mut resources = VNextStartupSequenceGuard::new(executor);
    let error = executor
        .admit_startup_sequence(&mut resources, Arc::from([TokenId::new(1)]), 16)
        .await
        .unwrap_err();
    assert!(
        matches!(error, FerrumError::ResourceExhausted { .. }),
        "actual startup error: {error:?}"
    );
    assert!(
        error
            .to_string()
            .contains("decode-sequence prefill was not submitted"),
        "actual startup error: {error:?}"
    );
    assert!(
        executor.metrics.step_deferrals.load(Ordering::Relaxed) > 0
            || executor.metrics.backing_deferrals.load(Ordering::Relaxed) > 0,
        "must hit actual temporary logical/backing capacity"
    );
    assert_eq!(
        resources.sequences.len(),
        1,
        "the guard retains the actual admitted owner"
    );
    let identity = resources.sequences[0].frontier.sequence().clone();
    let owner = Arc::downgrade(&resources.sequences[0].sequence);
    assert_eq!(resources.sequences[0].frontier.consumed_tokens(), 0);
    assert_eq!(
        resources.sequences[0]
            .sequence
            .prefill_tokens_processed
            .load(Ordering::Acquire),
        0
    );
    assert_eq!(executor.metrics.submitted_waves.load(Ordering::Relaxed), 0);
    assert_eq!(
        executor.metrics.decode_operations.load(Ordering::Relaxed),
        0
    );
    assert_eq!(
        executor
            .metrics
            .prefill_frontier_narrowings
            .load(Ordering::Relaxed),
        0
    );
    assert!(executor.startup_reusable_programs.lock().is_empty());
    drop(resources);
    drop(wave);
    step.try_abort().unwrap();
    drop((batch, competing_sequence));
    drop(competitor);
    settle(executor).await;
    assert!(owner.upgrade().is_none());
    assert!(identity.observed_end().is_some());
    executor
        .execute_startup_prefill_request(
            PrefillChunk::new(0, 1, 1).unwrap(),
            "released competing capacity",
        )
        .await
        .unwrap();
    settle(executor).await;
    assert_executor_closed(owned_executor)
}

pub async fn native_startup_rejects_segments_without_program_binding<R: DeviceRuntime>(
    owned_executor: ProgramExecutor<R>,
) -> Arc<R> {
    let executor = &owned_executor;
    let error = executor.prepare_startup().await.unwrap_err();
    assert!(matches!(error, FerrumError::Internal { .. }), "{error:?}");
    assert!(
        error.to_string().contains("registered no typed programs"),
        "{error:?}"
    );
    // These operators can capture segments, but provide no typed program
    // binding. Captured fragments alone must never qualify a startup catalog.
    assert!(
        executor
            .lane
            .reusable_executable_preparation()
            .unwrap()
            .resident_executables()
            > 0
    );
    assert!(executor.startup_reusable_programs.lock().is_empty());
    assert!(executor.reusable_execution_catalog.read().is_none());
    settle(executor).await;
    assert_executor_closed(owned_executor)
}

pub async fn assert_startup_capture_replay<R: DeviceRuntime>(executor: &ProgramExecutor<R>) {
    // Reuse observations are collected only with a profile attached. Replay
    // timing preserves direct reusable execution, unlike kernel verification.
    executor.attach_execution_event_sink(Arc::new(StartupEvents(
        Mutex::new(Vec::new()),
        DeviceTimingMode::Replay,
    )));
    executor.prepare_startup().await.unwrap();
    let inventory = {
        let state = executor.startup_preparation.lock();
        let VNextStartupPreparationState::Ready { report } = &*state else {
            panic!("actual reusable startup did not reach Ready: {state:?}");
        };
        assert!(
            report.enabled && report.supported,
            "unsupported is not a CUDA pass"
        );
        assert!(
            !report.eager_fallback_required,
            "the tiny language program must actually capture"
        );
        assert!(!report.requested_descriptors.is_empty());
        assert_eq!(report.prepared_descriptors, report.requested_descriptors);
        assert_eq!(
            report.prepared_decode_widths,
            report.requested_decode_widths
        );
        assert_eq!(
            report.prepared_prefill_chunks,
            report.requested_prefill_chunks
        );
        assert!(!report.requested_prefill_chunks.is_empty());
        assert!(report.capture_waves > 0 && report.replay_inventory_check_waves > 0);
        assert_eq!(
            report.capture_case_receipts.len(),
            report.requested_descriptors.len()
        );
        for receipt in &report.capture_case_receipts {
            assert!(report.requested_descriptors.contains(&receipt.descriptor));
            assert_eq!(receipt.observed_program_fingerprints.len(), 1);
            assert_eq!(
                receipt.resident_program_fingerprints,
                receipt.observed_program_fingerprints
            );
        }
        assert!(report.prepared_programs > 0);
        assert_eq!(report.prepared_programs, report.catalog_programs.len());
        let inventory = report.device_preparation;
        assert_eq!(
            inventory.state(),
            DeviceReusableExecutionPreparationState::Ready
        );
        assert!(inventory.resident_executables() > 0);
        assert_eq!(
            inventory.captured_executables(),
            inventory.resident_executables()
        );
        assert_eq!(
            inventory.uploaded_executables(),
            inventory.resident_executables()
        );
        assert_eq!(inventory.rejected_executables(), 0);
        assert_eq!(inventory.capacity_deferred_executables(), 0);
        inventory
    };
    settle(executor).await;
    assert_eq!(
        executor.metrics.submitted_waves.load(Ordering::Relaxed),
        0,
        "startup resets product counters"
    );

    // A fresh incarnation uses the sealed physical program. Actual replay
    // observations, not just an unchanged catalog, establish execution reuse.
    let maximum_tokens = executor
        .reusable_execution_startup_plan
        .as_ref()
        .unwrap()
        .maximum_decode_sequence_tokens;
    let mut resources = VNextStartupSequenceGuard::new(executor);
    assert!(executor
        .admit_startup_sequence(&mut resources, Arc::from([TokenId::new(0)]), maximum_tokens)
        .await
        .unwrap());
    let replay = &executor.metrics.wave_timing.reusable_execution;
    let prefill_segments = replay.replayed_segments.load(Ordering::Relaxed);
    let prefill_commands = replay.replayed_commands.load(Ordering::Relaxed);
    assert!(prefill_segments > 0 && prefill_commands > 0);
    executor
        .execute_startup_decode_pass(&mut resources, TokenId::new(1), 1, "test sealed replay")
        .await
        .unwrap();
    assert_eq!(resources.sequences[0].frontier.consumed_tokens(), 2);
    resources.complete().unwrap();
    assert!(replay.replayed_segments.load(Ordering::Relaxed) > prefill_segments);
    assert!(replay.replayed_commands.load(Ordering::Relaxed) > prefill_commands);
    let after = executor.lane.reusable_executable_preparation().unwrap();
    assert!(reusable_executable_inventory_matches(inventory, after));
    settle(executor).await;
}
