//! Logical controller integration. Selected costs are explicit fixtures;
//! production missing reference/host evidence is still Unknown. Real native
//! encoding and guarded rollback are covered separately by Metal product tests.
use super::*;

pub(in crate::continuous_engine::inner::slo_controller) async fn request(
    engine: &ContinuousBatchEngine,
    tokens: usize,
    maximum: usize,
) -> (RequestId, CreditedOutputSession) {
    let mut request = ferrum_types::InferenceRequest::new(
        vec!["test"; tokens].join(" "),
        engine.inner.config.model.model_id.clone(),
    );
    request.stream = true;
    request.sampling_params.max_tokens = maximum;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    let id = request.id.clone();
    let session = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    ready(engine, &id).await;
    assert_eq!(
        engine.inner.sequences.read()[&id].input_tokens.len(),
        tokens
    );
    (id, session)
}

pub(in crate::continuous_engine::inner::slo_controller) async fn admit(
    engine: &ContinuousBatchEngine,
    maximum: usize,
) {
    let _iteration = engine.inner.iteration_lock.lock().await;
    assert!(engine
        .inner
        .prepare_slo_admission_turn(maximum)
        .await
        .unwrap());
}

/// A retry may reread resource evidence, but cannot execute or replace an
/// owner. Capture only semantic frontiers; output readiness may change while
/// its real actor runs independently.
pub(in crate::continuous_engine::inner::slo_controller) struct UnsubmittedState {
    physical: usize,
    entries: usize,
    frontiers: std::collections::HashMap<
        RequestId,
        (usize, Option<(u64, u64)>, usize, bool, usize, usize),
    >,
}

impl UnsubmittedState {
    pub(in crate::continuous_engine::inner::slo_controller) fn capture(
        engine: &ContinuousBatchEngine,
        executor: &ControlledExecutor,
    ) -> Self {
        Self {
            physical: executor.physical.load(Ordering::Acquire),
            entries: executor.entries.load(Ordering::Acquire),
            frontiers: engine
                .inner
                .sequences
                .read()
                .iter()
                .map(|(id, sequence)| {
                    (
                        id.clone(),
                        (
                            Arc::as_ptr(&sequence.stream_projection_identity) as usize,
                            sequence.cost_frontier.map(|frontier| {
                                (
                                    frontier.owner_incarnation.get(),
                                    frontier.work_generation.get(),
                                )
                            }),
                            sequence.prefill_tokens_processed,
                            sequence.prefill_complete,
                            sequence.generated_tokens.len(),
                            sequence
                                .model_kv
                                .as_ref()
                                .map_or(0, |kv| kv.handle().num_tokens()),
                        ),
                    )
                })
                .collect(),
        }
    }

    pub(in crate::continuous_engine::inner::slo_controller) fn assert_unchanged(
        &self,
        engine: &ContinuousBatchEngine,
        executor: &ControlledExecutor,
    ) {
        let current = Self::capture(engine, executor);
        assert_eq!(
            current.physical, self.physical,
            "capture retry submitted physical work"
        );
        assert_eq!(
            current.entries, self.entries,
            "capture retry entered the executor"
        );
        assert_eq!(
            current.frontiers, self.frontiers,
            "capture retry changed an owner or work frontier"
        );
    }
}

pub(in crate::continuous_engine::inner::slo_controller) fn transient_route_read(
    executor: &ControlledExecutor,
) -> bool {
    use ferrum_interfaces::vnext::ExecutionCostRouteUnknown;
    matches!(
        *executor.cost_route_unknown.lock(),
        Some(ExecutionCostRouteUnknown::Resource(
            ResourcePlanningUnknown::ReadUnavailable(_)
                | ResourcePlanningUnknown::BusyOrUnavailable
        ))
    )
}

pub(in crate::continuous_engine::inner::slo_controller) async fn captured(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
) -> ControllerSnapshot {
    captured_with_hint(engine, executor, &ferrum_interfaces::BatchHint::simple(8)).await
}

pub(in crate::continuous_engine::inner::slo_controller) async fn captured_with_hint(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
    hint: &ferrum_interfaces::BatchHint,
) -> ControllerSnapshot {
    let before = UnsubmittedState::capture(engine, executor);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    loop {
        *executor.cost_route_unknown.lock() = None;
        let result = engine.inner.capture_slo_controller_snapshot(
            hint,
            ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap(),
        );
        before.assert_unchanged(engine, executor);
        match result {
            Ok(snapshot) => return snapshot,
            Err(error) => {
                assert!(
                    matches!(error.reason, "resource_unknown" | "shape_unavailable")
                        && transient_route_read(executor),
                    "non-transient fixture capture failure: {error:?}; route={:?}",
                    executor.cost_route_unknown.lock()
                );
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "real route read remained contended: {error:?}; route={:?}",
                    executor.cost_route_unknown.lock()
                );
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
    }
}

/// Exercise the actual controller retry timer. An Idle result is not itself
/// permission to spin: a transient typed read or publication race must have
/// armed the production retry, and malformed/permanent evidence fails here.
pub(in crate::continuous_engine::inner::slo_controller) async fn selected_after_retry(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
    mut prepare: impl FnMut() -> Result<SloIterationPlan>,
) -> owner::PreparedControllerWave {
    let before = UnsubmittedState::capture(engine, executor);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    loop {
        *executor.cost_route_unknown.lock() = None;
        *executor.resource_planning_unknown.lock() = None;
        executor
            .resource_revalidation_changed
            .store(false, Ordering::Release);
        let previous = engine.inner.slo_controller.lock().observations;
        let plan = prepare().unwrap();
        before.assert_unchanged(engine, executor);
        match plan {
            SloIterationPlan::Selected(prepared) => return prepared,
            SloIterationPlan::Idle => {
                let (retry, observation) = {
                    let state = engine.inner.slo_controller.lock();
                    (
                        state.retry.is_some(),
                        (state.observations != previous)
                            .then_some(state.last_observation)
                            .flatten(),
                    )
                };
                let route = *executor.cost_route_unknown.lock();
                let resource = *executor.resource_planning_unknown.lock();
                let changed = executor
                    .resource_revalidation_changed
                    .load(Ordering::Acquire);
                assert!(retry, "Idle without an independent retry: {observation:?}; route={route:?}; resource={resource:?}");
                if route.is_some() || resource.is_some() {
                    assert!(transient_route_read(executor),
                        "permanent resource failure must not be retried: {observation:?}; route={route:?}; resource={resource:?}");
                } else if !changed {
                    // An earlier branch in this same preparation may have
                    // recorded missing-reference/Unknown before real fallback
                    // publication detected changed evidence. The actual
                    // comparison and armed retry authorize that retry, not an
                    // unrelated last-observation string.
                    if let Some(observation) = observation {
                        assert!(
                            matches!(
                                observation.reason,
                                "capacity_snapshot_busy"
                                    | "sequence_snapshot_busy"
                                    | "controller_busy"
                                    | "scheduler_snapshot_unavailable"
                                    | "publication_release_busy"
                                    | "compute_budget_exhausted"
                            ),
                            "non-transient completion selection: {observation:?}"
                        );
                    }
                }
                tokio::time::timeout_at(deadline, engine.inner.wait_for_slo_controller_retry())
                    .await
                    .expect("production controller retry failed to wake before the test deadline");
                assert!(tokio::time::Instant::now() < deadline,
                    "completion remained contended until the test deadline: {observation:?}; route={route:?}");
            }
            SloIterationPlan::Legacy => {
                panic!("completion retry unexpectedly selected the legacy driver")
            }
        }
    }
}

fn expected_row(
    captured: &ControllerSnapshot,
    id: &RequestId,
    action: PlanningWorkAction,
) -> ExpectedWaveParticipant {
    let index = captured
        .fences
        .iter()
        .position(|row| row.key.request_id == *id)
        .unwrap();
    let fence = &captured.fences[index];
    ExpectedWaveParticipant {
        participant_index: index,
        request_id: id.clone(),
        input: match action {
            PlanningWorkAction::Decode => ExpectedWaveInput::Decode {
                cache_id: fence.cache_id.clone().unwrap(),
            },
            PlanningWorkAction::Prefill { offset, count } => ExpectedWaveInput::Prefill {
                chunk: PrefillChunk::new(offset, count.get(), fence.prefill_total).unwrap(),
            },
        },
        host: CostObservationParticipant {
            request_id: id.clone(),
            owner_incarnation: fence.incarnation,
            work_generation: fence.generation,
            input_index: 0,
            output_policy_signature: Some(host_history_cost_signature(
                captured.snapshot.requests[index].output_policy_signature,
                fence.generated as u64,
            )),
            host_features: fence.host_features,
        },
    }
}

async fn install(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
    scheduler: &ContinuousBatchScheduler,
    choices: &[(RequestId, PlanningWorkAction)],
) -> owner::PreparedControllerWave {
    install_with_admission(engine, executor, scheduler, choices, None).await
}

pub(in crate::continuous_engine::inner::slo_controller) async fn install_with_admission(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
    scheduler: &ContinuousBatchScheduler,
    choices: &[(RequestId, PlanningWorkAction)],
    admission: Option<super::super::admission::PendingTimeWitness>,
) -> owner::PreparedControllerWave {
    let captured = captured(engine, executor).await;
    let mut participants = Vec::new();
    let mut selected = Vec::new();
    let mut rows = Vec::new();
    for (id, action) in choices {
        let mut row = expected_row(&captured, id, *action);
        row.host.input_index = participants.len() as u32;
        let fence = &captured.fences[row.participant_index];
        selected.push(PlanningWorkSelection {
            key: fence.key.clone(),
            action: *action,
        });
        rows.push(match row.input {
            ExpectedWaveInput::Prefill { chunk } => ActualRowWork::Prefill {
                offset: chunk.tokens_processed() as u32,
                count: chunk.tokens_to_process() as u32,
                total_prompt_tokens: chunk.total_prompt_tokens() as u32,
            },
            ExpectedWaveInput::Decode { .. } => ActualRowWork::Decode {
                kv_tokens: fence.context as u32,
            },
        });
        participants.push(row);
    }
    let has_prefill = rows
        .iter()
        .any(|row| matches!(row, ActualRowWork::Prefill { .. }));
    let has_decode = rows
        .iter()
        .any(|row| matches!(row, ActualRowWork::Decode { .. }));
    let mut shape = canonical(1);
    shape.kind = match (has_prefill, has_decode) {
        (true, true) => ActualWaveKind::Mixed,
        (true, false) => ActualWaveKind::Prefill,
        (false, true) => ActualWaveKind::Decode,
        _ => unreachable!(),
    };
    shape.rows = rows;
    let expected =
        ExpectedExecutionCostWave::new(captured.route.clone(), shape, participants).unwrap();
    let mut availability = Vec::new();
    let epochs = engine
        .inner
        .model_executor
        .write_execution_capacity_snapshot(&mut availability)
        .unwrap()
        .unwrap();
    let PlanningSelectionOutcome::Published { batch, receipt } = scheduler
        .try_select_planned_wave(
            &captured.queue,
            &selected,
            &ferrum_interfaces::BatchHint::simple(8),
            AdmissionWakeSnapshot::new(
                AdmissionWakeEpochs::new(
                    epochs.coordinator_id,
                    epochs.release_epoch,
                    epochs.capacity_epoch,
                    0,
                ),
                &availability,
            ),
        )
        .unwrap()
    else {
        panic!("exact fixture selection unavailable");
    };
    assert!(engine.inner.reserve_batch_output(&batch).unwrap());
    captured.budget.finish_planning();
    engine
        .inner
        .install_controller_wave(
            owner::ControllerWork {
                batch,
                expected: ExpectedExecutionWave::from_cost_witness(expected, |id| {
                    captured
                        .fences
                        .iter()
                        .find(|fence| fence.key.request_id == *id)
                        .map(|fence| &fence.logits_policy)
                })
                .unwrap(),
                timing: owner::ControllerTimingCommitment::Witness {
                    predicted_wall_ns: 1,
                    admission,
                    valid_until: slo_clock_now() + Duration::from_secs(30),
                    model_version: captured.snapshot.cost_model_version,
                },
                proof: captured.into_safety(),
            },
            receipt,
        )
        .unwrap()
}

fn prefill(offset: usize, count: usize) -> PlanningWorkAction {
    PlanningWorkAction::Prefill {
        offset,
        count: NonZeroUsize::new(count).unwrap(),
    }
}

#[tokio::test]
async fn fresh_partial_final_decode_uses_exact_guarded_waves_and_token_only_grants() {
    let (engine, scheduler, executor) = fixture().await;
    let (id, mut session) = request(&engine, 4, 2).await;
    admit(&engine, 1).await;
    let snapshot = captured(&engine, &executor).await;
    assert!(!matches!(
        engine.inner.propose_slo_controller(&snapshot),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::MissingReferenceWork,
            ..
        }
    ));
    let RequestPhaseView::Prefill(progress) = &snapshot.snapshot.requests[0].phase else {
        panic!("prefill reference");
    };
    assert!(!progress.reference.points.is_empty());
    assert_eq!(snapshot.snapshot.scope.reference_decode_token_ns.get(), 7);
    assert!(snapshot.fences[0].resource_cache_id().is_none());
    let partial = install(
        &engine,
        &executor,
        &scheduler,
        &[(id.clone(), prefill(0, 2))],
    )
    .await;
    assert!(engine.inner.sequences.read()[&id]
        .credited_output
        .as_ref()
        .unwrap()
        .grant
        .is_none());
    assert!(matches!(
        engine
            .inner
            .execute_slo_controller_wave(partial)
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert_eq!(sequence.prefill_tokens_processed, 2);
        assert!(!sequence.prefill_complete);
        assert!(sequence.generated_tokens.is_empty());
        assert!(sequence.credited_output.as_ref().unwrap().grant.is_none());
    }
    let snapshot = captured(&engine, &executor).await;
    assert_eq!(snapshot.queue.requests()[0].computed_tokens, 2);
    assert_eq!(snapshot.queue.requests()[0].committed_output_tokens, 0);
    assert!(snapshot.fences[0].cache_id.is_some());
    assert!(snapshot.fences[0].resource_cache_id().is_none());
    let final_wave = install(
        &engine,
        &executor,
        &scheduler,
        &[(id.clone(), prefill(2, 2))],
    )
    .await;
    assert!(engine.inner.sequences.read()[&id]
        .credited_output
        .as_ref()
        .unwrap()
        .grant
        .is_some());
    engine
        .inner
        .execute_slo_controller_wave(final_wave)
        .await
        .unwrap();
    drop(bounded(session.frames.next()).await.unwrap());
    ready(&engine, &id).await;
    let snapshot = captured(&engine, &executor).await;
    assert_eq!(snapshot.queue.requests()[0].committed_output_tokens, 1);
    assert_eq!(snapshot.queue.requests()[0].computed_tokens, 4);
    assert!(snapshot.fences[0].resource_cache_id().is_some());
    let decode = install(
        &engine,
        &executor,
        &scheduler,
        &[(id.clone(), PlanningWorkAction::Decode)],
    )
    .await;
    engine
        .inner
        .execute_slo_controller_wave(decode)
        .await
        .unwrap();
    assert!(!engine.inner.sequences.read().contains_key(&id));
    assert_eq!(executor.entries.load(Ordering::Acquire), 3);
    assert_eq!(executor.physical.load(Ordering::Acquire), 3);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn abandoned_partial_publication_releases_exact_work_without_token_credit_or_dispatch() {
    let (engine, scheduler, executor) = fixture().await;
    let (id, session) = request(&engine, 4, 2).await;
    admit(&engine, 1).await;
    drop(
        install(
            &engine,
            &executor,
            &scheduler,
            &[(id.clone(), prefill(0, 2))],
        )
        .await,
    );
    engine.inner.drain_slo_execution().await.unwrap();
    let snapshot = captured(&engine, &executor).await;
    assert_eq!(snapshot.queue.requests()[0].scheduled_tokens, 0);
    assert_eq!(snapshot.queue.requests()[0].computed_tokens, 0);
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    let wave = install(
        &engine,
        &executor,
        &scheduler,
        &[(id.clone(), prefill(0, 2))],
    )
    .await;
    engine
        .inner
        .execute_slo_controller_wave(wave)
        .await
        .unwrap();
    assert_eq!(
        engine.inner.sequences.read()[&id].prefill_tokens_processed,
        2
    );
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn stale_prefill_completion_cannot_write_reused_incarnation_or_sample() {
    let (engine, _, executor) = fixture().await;
    let (id, session) = request(&engine, 2, 2).await;
    admit(&engine, 1).await;
    let snapshot = captured(&engine, &executor).await;
    let expected = expected_row(&snapshot, &id, prefill(0, 2));
    let ExpectedWaveInput::Prefill { chunk } = expected.input else {
        unreachable!()
    };
    {
        let mut sequences = engine.inner.sequences.write();
        let sequence = sequences.get_mut(&id).unwrap();
        sequence.cost_frontier = Some(
            engine
                .inner
                .cost_runtime
                .as_ref()
                .unwrap()
                .ids
                .new_frontier()
                .unwrap(),
        );
    }
    let cache = Arc::new(MockKvCacheHandle::new(id.clone(), 1, 2));
    let completion = PlanRuntimePrefillCompletion::exact(
        PlanRuntimePrefillOutput::final_logits(id.clone(), 2, vec![0.0; 64], cache).unwrap(),
        chunk,
    );
    assert!(engine
        .inner
        .commit_plan_runtime_prefill_completion_fenced(
            &id,
            2,
            chunk,
            completion,
            &mut None,
            Some(ControllerCommitFence::Cost(&expected)),
        )
        .await
        .is_err());
    assert!(engine.inner.sequences.read()[&id]
        .generated_tokens
        .is_empty());
    assert_eq!(
        engine.inner.sequences.read()[&id].prefill_tokens_processed,
        0
    );
    assert!(engine.inner.sequences.read()[&id].model_kv.is_none());
    cleanup(engine, session).await;
}

#[tokio::test]
async fn guarded_prefill_rejects_a_valid_adaptively_narrowed_receipt_without_host_commit() {
    let (engine, _, executor) = fixture().await;
    let (id, session) = request(&engine, 4, 2).await;
    admit(&engine, 1).await;
    let snapshot = captured(&engine, &executor).await;
    let expected = expected_row(&snapshot, &id, prefill(0, 4));
    let ExpectedWaveInput::Prefill { chunk: planned } = expected.input else {
        unreachable!()
    };
    let completed = PrefillChunk::new(0, 2, 4).unwrap();
    let cache = Arc::new(MockKvCacheHandle::new(id.clone(), 1, 2));
    let completion = PlanRuntimePrefillCompletion::new(
        PlanRuntimePrefillOutput::intermediate(id.clone(), 2, cache),
        planned,
        completed,
        1,
    )
    .unwrap();
    // The ordinary capacity-aware contract accepts this narrower work. A
    // selected wave has already committed to one exact shape and cannot.
    completion.validate_for(&id, planned, 64).unwrap();
    assert!(engine
        .inner
        .commit_plan_runtime_prefill_completion_fenced(
            &id,
            4,
            planned,
            completion,
            &mut None,
            Some(ControllerCommitFence::Cost(&expected)),
        )
        .await
        .is_err());
    let current = captured(&engine, &executor).await;
    assert_eq!(current.queue.requests()[0].computed_tokens, 0);
    assert_eq!(current.queue.requests()[0].committed_output_tokens, 0);
    assert!(engine.inner.sequences.read()[&id].model_kv.is_none());
    assert!(engine.inner.sequences.read()[&id]
        .generated_tokens
        .is_empty());
    cleanup(engine, session).await;
}

#[tokio::test]
async fn mixed_partial_final_and_decode_publish_once_with_phase_correct_grants() {
    let (engine, scheduler, executor) = fixture_with_width(3).await;
    let (decode_id, mut decode_session) = request(&engine, 1, 4).await;
    let seed = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    engine.inner.process_batch(&seed).await.unwrap();
    drop(bounded(decode_session.frames.next()).await.unwrap());
    ready(&engine, &decode_id).await;
    let (partial_id, partial_session) = request(&engine, 4, 4).await;
    let (final_id, mut final_session) = request(&engine, 2, 4).await;
    admit(&engine, 2).await;
    let wave = install(
        &engine,
        &executor,
        &scheduler,
        &[
            (partial_id.clone(), prefill(0, 2)),
            (decode_id.clone(), PlanningWorkAction::Decode),
            (final_id.clone(), prefill(0, 2)),
        ],
    )
    .await;
    {
        let sequences = engine.inner.sequences.read();
        assert!(sequences[&partial_id]
            .credited_output
            .as_ref()
            .unwrap()
            .grant
            .is_none());
        assert!(sequences[&final_id]
            .credited_output
            .as_ref()
            .unwrap()
            .grant
            .is_some());
        assert!(sequences[&decode_id]
            .credited_output
            .as_ref()
            .unwrap()
            .grant
            .is_some());
    }
    engine
        .inner
        .execute_slo_controller_wave(wave)
        .await
        .unwrap();
    assert_eq!(executor.entries.load(Ordering::Acquire), 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(
        engine.inner.sequences.read()[&partial_id].prefill_tokens_processed,
        2
    );
    assert_eq!(
        engine.inner.sequences.read()[&partial_id]
            .generated_tokens
            .len(),
        0
    );
    assert_eq!(
        engine.inner.sequences.read()[&final_id]
            .generated_tokens
            .len(),
        1
    );
    assert_eq!(
        engine.inner.sequences.read()[&decode_id]
            .generated_tokens
            .len(),
        2
    );
    drop(bounded(final_session.frames.next()).await.unwrap());
    drop(bounded(decode_session.frames.next()).await.unwrap());
    drop((partial_session, final_session));
    cleanup(engine, decode_session).await;
}

#[tokio::test]
async fn mixed_later_narrowed_receipt_rejects_every_host_commit_and_releases_the_whole_wave() {
    let (engine, scheduler, executor) = fixture_with_width(3).await;
    let (decode_id, mut decode_session) = request(&engine, 1, 4).await;
    let seed = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    engine.inner.process_batch(&seed).await.unwrap();
    drop(bounded(decode_session.frames.next()).await.unwrap());
    ready(&engine, &decode_id).await;
    let (first_id, mut first_session) = request(&engine, 2, 4).await;
    let (last_id, mut last_session) = request(&engine, 4, 4).await;
    admit(&engine, 2).await;
    let expected_frontiers = [
        (decode_id.clone(), 1, 1),
        (first_id.clone(), 0, 0),
        (last_id.clone(), 0, 0),
    ];
    // Discard precedes terminal request cleanup. Check real host and scheduler
    // frontiers there, while all three original owners still exist; their later
    // removal must not conceal a partial commit.
    let inner = Arc::downgrade(&engine.inner);
    let queue = Arc::clone(&scheduler);
    *executor.before_prefill_discard.lock() = Some(Box::new(move || {
        let inner = inner.upgrade().unwrap();
        {
            let sequences = inner.sequences.read();
            for (id, context, generated) in &expected_frontiers {
                let sequence = &sequences[id];
                assert_eq!(sequence.prefill_tokens_processed, *context);
                assert_eq!(sequence.generated_tokens.len(), *generated);
                assert_eq!(
                    sequence
                        .model_kv
                        .as_ref()
                        .map_or(0, |kv| kv.handle().num_tokens()),
                    *context
                );
            }
        }
        let mut availability = Vec::new();
        let epochs = inner
            .model_executor
            .write_execution_capacity_snapshot(&mut availability)
            .unwrap()
            .unwrap();
        let captured = queue
            .planning_state(
                NonZeroUsize::new(3).unwrap(),
                AdmissionWakeSnapshot::new(
                    AdmissionWakeEpochs::new(
                        epochs.coordinator_id,
                        epochs.release_epoch,
                        epochs.capacity_epoch,
                        0,
                    ),
                    &availability,
                ),
            )
            .unwrap();
        for (id, context, generated) in &expected_frontiers {
            let row = captured
                .requests()
                .iter()
                .find(|row| row.key.request_id == *id)
                .unwrap();
            assert_eq!(row.computed_tokens, *context);
            assert_eq!(row.committed_output_tokens, *generated);
        }
    }));
    let before_prefill = engine.inner.total_prefill_tokens.load(Ordering::Acquire);
    let before_decode = engine.inner.total_decode_tokens.load(Ordering::Acquire);
    executor
        .narrow_last_mixed_prefill
        .store(true, Ordering::Release);
    let wave = install(
        &engine,
        &executor,
        &scheduler,
        &[
            (decode_id.clone(), PlanningWorkAction::Decode),
            (first_id.clone(), prefill(0, 2)),
            (last_id.clone(), prefill(0, 4)),
        ],
    )
    .await;
    assert!(engine
        .inner
        .execute_slo_controller_wave(wave)
        .await
        .is_err());
    assert_eq!(
        engine.inner.total_prefill_tokens.load(Ordering::Acquire),
        before_prefill
    );
    assert_eq!(
        engine.inner.total_decode_tokens.load(Ordering::Acquire),
        before_decode
    );
    assert_eq!(
        *executor.discarded_prefills.lock(),
        vec![format!("mock_{first_id}"), format!("mock_{last_id}")]
    );
    assert!(executor
        .produced_caches
        .lock()
        .iter()
        .all(|cache| cache.upgrade().is_none()));
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(scheduler.active_count(), 0);
    assert_eq!(scheduler.waiting_count(), 0);
    assert!(engine.inner.slo_controller.lock().pending_release.is_none());
    assert!(engine.inner.drain_slo_execution().await.unwrap().is_none());
    assert_eq!(executor.entries.load(Ordering::Acquire), 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    for session in [&mut decode_session, &mut first_session, &mut last_session] {
        bounded(async {
            while let Some(frame) = session.frames.next().await {
                assert!(
                    frame.metadata().terminal,
                    "an invalid batch published token output"
                );
            }
        })
        .await;
        let completion = bounded(&mut session.completion).await.unwrap();
        assert!(matches!(completion.payload(), OutputCompletion::Failed(_)));
    }
    drop((first_session, last_session));
    cleanup(engine, decode_session).await;
}
