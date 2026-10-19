//! Engine publication tests. Native copy/numerical correctness is exercised
//! by the runtime tests; this fixture detects scheduling and ownership errors.
use super::*;
use ferrum_execution::model_executor::{
    PlanRuntimePrefixRestoreInput, PlanRuntimePrefixRestoreOutcome, PlanRuntimePrefixRestoreOutput,
};

#[test]
fn prefix_restore_decisions_reach_run_and_serve_journals_with_capacity_evidence() {
    use ferrum_execution::model_executor::{
        PrefixRestoreDecision, PrefixRestoreObservation, PrefixRestoreSource,
    };
    use ferrum_runtime::{
        CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition,
    };
    use std::num::NonZeroU64;
    let capacity = test_execution_capacity_deferral(
        ExecutorAdmissionEpochs::new(NonZeroU64::new(19).unwrap(), 3, 5),
        CapacityWaitCondition::from_observation(
            19,
            vec![CapacityAvailabilityEpoch::new(
                CapacityAvailabilitySource::ActiveSequenceSlots,
                7,
            )
            .unwrap()],
        )
        .unwrap(),
        ExecutorExecutionCapacityStage::SequenceExtension,
    );
    for entrypoint in [ProfileEntrypoint::Run, ProfileEntrypoint::Serve] {
        let profile_path = resource_trace_temp_path("prefix-restore-decisions-profile");
        let scheduler_path = resource_trace_temp_path("prefix-restore-decisions-scheduler");
        let profile = create_scheduler_trace_sink(Some(&profile_path)).unwrap();
        let scheduler = create_scheduler_trace_sink(Some(&scheduler_path)).unwrap();
        let sink = VNextProfileExecutionEventSink::with_journals(
            vec![profile.clone(), scheduler.clone()],
            entrypoint,
            &EngineConfig::default(),
        );
        let request_id = RequestId::new();
        for decision in [
            PrefixRestoreDecision::NoReusableEntry,
            PrefixRestoreDecision::SequenceCapacityNotReady {
                candidate_prefix_tokens: 128,
                capacity: &capacity,
            },
            PrefixRestoreDecision::Restored {
                candidate_prefix_tokens: 128,
            },
        ] {
            sink.record_prefix_restore_decision(&PrefixRestoreObservation {
                request_id: &request_id,
                source: PrefixRestoreSource::Index,
                decision,
            })
            .unwrap();
        }
        profile.flush().unwrap();
        scheduler.flush().unwrap();
        let events = read_engine_profile_events(&profile_path);
        assert_eq!(events, read_engine_profile_events(&scheduler_path));
        assert_eq!(
            events
                .iter()
                .map(|e| e.shape["outcome"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "no_reusable_entry",
                "sequence_capacity_not_ready",
                "restored"
            ]
        );
        assert!(events[0].shape.get("candidate_prefix_tokens").is_none());
        assert_eq!(events[1].shape["candidate_prefix_tokens"], 128);
        assert_eq!(
            events[1].attributes["restore_decision"]["capacity"],
            serde_json::to_value(&capacity).unwrap()
        );
        assert_eq!(
            events[1].attributes["restore_decision"]["capacity"]["stage"],
            "sequence_extension"
        );
        for event in &events {
            assert_eq!(event.request_id, request_id.to_string());
            assert_eq!(event.entrypoint, entrypoint);
            assert_eq!(event.phase, "vnext.prefix_restore_decision");
            assert_eq!(event.shape["source"], "index");
            event.validate().unwrap();
        }
        drop(sink);
        profile.close().unwrap();
        scheduler.close().unwrap();
        std::fs::remove_file(profile_path).unwrap();
        std::fs::remove_file(scheduler_path).unwrap();
    }
}

#[derive(Clone, Copy)]
enum RestoreBehavior {
    Hit,
    Miss,
    ForeignRequest,
    FailedAcknowledgement,
    AcknowledgedWithObserverError,
}

pub(super) struct RestoreState {
    behavior: RestoreBehavior,
    tokens: usize,
    acknowledgements: AtomicU64,
    cancellations: AtomicU64,
}

struct PublicationGuard {
    state: Arc<RestoreState>,
    acknowledged: bool,
}

impl Drop for PublicationGuard {
    fn drop(&mut self) {
        if !self.acknowledged {
            self.state.cancellations.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl RestoreState {
    pub(super) fn restore(
        self: &Arc<Self>,
        input: PlanRuntimePrefixRestoreInput<'_>,
        tokens: Arc<ControlledTokenState>,
    ) -> Result<PlanRuntimePrefixRestoreOutcome> {
        if matches!(self.behavior, RestoreBehavior::Miss) {
            return Ok(PlanRuntimePrefixRestoreOutcome::Unavailable);
        }
        let request_id = if matches!(self.behavior, RestoreBehavior::ForeignRequest) {
            RequestId::new()
        } else {
            input.request_id.clone()
        };
        let mut guard = PublicationGuard {
            state: Arc::clone(self),
            acknowledged: false,
        };
        let sequence = tokens
            .frontier_for_request(input.request_id)
            .expect("restore target must be admitted")
            .sequence()
            .clone();
        PlanRuntimePrefixRestoreOutput::new_tokens(
            request_id,
            self.tokens,
            input.input_tokens.len(),
            sequence.clone(),
            move || {
                if matches!(guard.state.behavior, RestoreBehavior::FailedAcknowledgement) {
                    return Err(FerrumError::cancelled("restored target was cancelled"));
                }
                let frontier = tokens.restore_frontier(&sequence, guard.state.tokens)?;
                guard.acknowledged = true;
                guard.state.acknowledgements.fetch_add(1, Ordering::Relaxed);
                let mut acknowledgement =
                    ferrum_execution::model_executor::PrefixRestoreAcknowledgement::new(frontier);
                if matches!(
                    guard.state.behavior,
                    RestoreBehavior::AcknowledgedWithObserverError
                ) {
                    acknowledgement.error =
                        Some(FerrumError::backend("post-acknowledgement observer error"));
                }
                Ok(acknowledgement)
            },
        )
        .map(PlanRuntimePrefixRestoreOutcome::Restored)
    }
}

fn engine(
    behavior: RestoreBehavior,
    tokens: usize,
    trace_path: Option<PathBuf>,
) -> (
    Arc<ContinuousBatchEngine>,
    Arc<PlanRuntimeChunkedPrefillObservation>,
    Arc<RestoreState>,
) {
    engine_observed(behavior, tokens, trace_path, None)
}

fn engine_observed(
    behavior: RestoreBehavior,
    tokens: usize,
    trace_path: Option<PathBuf>,
    observer: Option<Arc<dyn ExecutionAttemptObserver>>,
) -> (
    Arc<ContinuousBatchEngine>,
    Arc<PlanRuntimeChunkedPrefillObservation>,
    Arc<RestoreState>,
) {
    let mut config = EngineConfig::default();
    config.runtime.scheduler_trace_jsonl = trace_path;
    config.scheduler.max_running_requests = 1;
    config.scheduler.prefill_step_chunk = Some(2);
    let state = Arc::new(RestoreState {
        behavior,
        tokens,
        acknowledgements: AtomicU64::new(0),
        cancellations: AtomicU64::new(0),
    });
    let mut executor = PlanRuntimeChunkedPrefillTestExecutor::new(false);
    executor.prefix_restore = Some(Arc::clone(&state));
    let observation = executor.observer();
    let executor = ExecutorOwner::new(Box::new(executor));
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let engine = ContinuousBatchEngine::new_observed(
        config,
        scheduler,
        Arc::new(ferrum_testkit::MockTokenizer::new(128)),
        Arc::new(ferrum_testkit::MockSampler),
        executor,
        None,
        observer,
        None,
    )
    .unwrap();
    (Arc::new(engine), observation, state)
}

fn request() -> InferenceRequest {
    let mut request = policy_request();
    request.prompt = "one two three four".to_owned();
    request.sampling_params.max_tokens = 1;
    request
}

#[tokio::test]
async fn prefix_restore_time_planning_reprojects_current_frontier_with_original_snapshot() {
    use super::cost_feedback_tests::{submit, Source};
    use crate::continuous_engine::attempt_observation_tests::AttemptRecorder;
    use ferrum_execution::execution_attempt::{AttemptPrefixRestore, ExecutionAttemptEligibility};
    use ferrum_scheduler::cost_prediction::CostWork;

    let mut config = EngineConfig::default();
    config.runtime.profile_detail = ObservabilityProfileDetail::Off;
    config.scheduler.max_running_requests = 1;
    config.scheduler.slo = Some("ttft:1000,tpot:100,itl:100".parse().unwrap());
    let state = Arc::new(RestoreState {
        behavior: RestoreBehavior::Hit,
        tokens: 2,
        acknowledgements: AtomicU64::new(0),
        cancellations: AtomicU64::new(0),
    });
    let mut implementation = PlanRuntimeChunkedPrefillTestExecutor::new(false);
    implementation.prefix_restore = Some(state.clone());
    implementation.prompt_tail_reserve = Some(2);
    let observation = implementation.observer();
    let recorder = Arc::new(AttemptRecorder::default());
    let source = Source::new(true);
    source.replace_after_load.store(true, Ordering::Release);
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let engine = ContinuousBatchEngine::new_observed(
        config.clone(),
        Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
        tokenizer.clone(),
        Arc::new(ferrum_testkit::MockSampler),
        ExecutorOwner::new(Box::new(implementation)),
        None,
        Some(recorder.clone()),
        Some(source.clone()),
    )
    .unwrap();
    let id = submit(&engine, &tokenizer, "one two three four").await;
    let inner = engine.active_inner().unwrap();
    inner.run_iteration().await.unwrap();
    assert_eq!(source.load_count(), 1);
    assert_eq!(
        observation.shape_policy_reads.load(Ordering::Relaxed),
        1,
        "execution must borrow the policy already used before admission awaits"
    );
    assert_eq!(state.acknowledgements.load(Ordering::Relaxed), 1);
    assert_eq!(state.cancellations.load(Ordering::Relaxed), 0);
    let first_queries = source.take_work_queries();
    assert!(!first_queries.is_empty());
    assert!(first_queries.iter().all(|(revision, _)| *revision == 1));
    assert!(first_queries.iter().any(|(_, work)| work.len() == 1
        && work[0].requested.start() == 0
        && work[0].requested.end() == 3));
    let first = recorder.take();
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].eligibility, ExecutionAttemptEligibility::Completed);
    assert_eq!(first[0].participants.len(), 1);
    let actual = &first[0].participants[0];
    assert_eq!(actual.request_id, id);
    assert_eq!(actual.requested.as_range(), 2..3);
    assert_eq!(
        actual.prefix_restore,
        AttemptPrefixRestore::Restored { tokens: 2 }
    );
    let priced = CostWork {
        kind: actual.kind,
        requested: actual.requested,
        context_tokens: actual.context_tokens,
        prefix_restore: actual.prefix_restore,
        internally_reused_tokens: actual.internally_reused_tokens,
    };
    assert!(
        first_queries.iter().any(|(_, work)| work == &[priced]),
        "the restored suffix must be reprojected and priced in the same snapshot"
    );
    let handle = inner.sequences.get(&id).unwrap();
    let incarnation = {
        let guard = handle.lock();
        let sequence = guard.as_ref().unwrap();
        let frontier = sequence.execution_frontier.as_ref().unwrap();
        assert_eq!(frontier.consumed_tokens(), 3);
        assert!(sequence.generated_tokens.is_empty());
        let (restored, tokens) = sequence.confirmed_prefix_restore.as_ref().unwrap();
        assert_eq!(restored, frontier.sequence());
        assert_eq!(*tokens, 2);
        frontier.sequence().clone()
    };
    inner.run_iteration().await.unwrap();
    assert_eq!(source.load_count(), 2);
    assert_eq!(observation.shape_policy_reads.load(Ordering::Relaxed), 2);
    let second_queries = source.take_work_queries();
    assert!(!second_queries.is_empty());
    assert!(second_queries.iter().all(|(revision, _)| *revision == 2));
    assert_eq!(
        *observation.attempted_chunks.lock().unwrap(),
        [
            PrefillChunk::new(2, 1, 5).unwrap(),
            PrefillChunk::new(3, 2, 5).unwrap(),
        ]
    );
    {
        let guard = handle.lock();
        let sequence = guard.as_ref().unwrap();
        let frontier = sequence.execution_frontier.as_ref().unwrap();
        assert_eq!(frontier.sequence(), &incarnation);
        assert_eq!(frontier.consumed_tokens(), 5);
        assert_eq!(sequence.generated_tokens.len(), 1);
    }
    drop((handle, inner));
    engine.shutdown().await.unwrap();
    engine.wait_for_shutdown_completion().await.unwrap();
    assert!(incarnation.observed_end().is_some());
    assert!(observation.retained.lock().unwrap().is_empty());
    assert!(observation.tokens.0.upgrade().is_none());
}

#[tokio::test]
async fn prefix_restore_executes_only_suffix_and_samples_after_real_prefill() {
    let (engine, observation, state) = engine(RestoreBehavior::Hit, 3, None);
    let response = tokio::time::timeout(Duration::from_secs(2), engine.infer(request()))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(response.response().tokens.len(), 1);
    assert_eq!(state.acknowledgements.load(Ordering::Relaxed), 1);
    assert_eq!(state.cancellations.load(Ordering::Relaxed), 0);
    assert_eq!(
        *observation.attempted_chunks.lock().unwrap(),
        [PrefillChunk::new(3, 2, 5).unwrap()]
    );
    assert_eq!(
        engine
            .active_inner()
            .expect("engine execution is open")
            .total_prefill_tokens
            .load(Ordering::Relaxed),
        2
    );
    assert_eq!(
        engine
            .active_inner()
            .expect("engine execution is open")
            .prefix_cache_hits
            .load(Ordering::Relaxed),
        1
    );
    assert!(observation.retained.lock().unwrap().is_empty());
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_restore_miss_preserves_all_cold_prefill_work() {
    let (engine, observation, state) = engine(RestoreBehavior::Miss, 3, None);
    tokio::time::timeout(Duration::from_secs(2), engine.infer(request()))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        *observation.attempted_chunks.lock().unwrap(),
        [
            PrefillChunk::new(0, 2, 5).unwrap(),
            PrefillChunk::new(2, 2, 5).unwrap(),
            PrefillChunk::new(4, 1, 5).unwrap(),
        ]
    );
    assert_eq!(
        engine
            .active_inner()
            .expect("engine execution is open")
            .total_prefill_tokens
            .load(Ordering::Relaxed),
        5
    );
    assert_eq!(
        engine
            .active_inner()
            .expect("engine execution is open")
            .prefix_cache_hits
            .load(Ordering::Relaxed),
        0
    );
    assert_eq!(state.acknowledgements.load(Ordering::Relaxed), 0);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_restore_rejects_foreign_full_or_cancelled_state_before_execution() {
    for (behavior, tokens) in [
        (RestoreBehavior::ForeignRequest, 3),
        (RestoreBehavior::Hit, 0),
        (RestoreBehavior::Hit, 5),
        (RestoreBehavior::FailedAcknowledgement, 3),
    ] {
        let (engine, observation, state) = engine(behavior, tokens, None);
        assert!(
            tokio::time::timeout(Duration::from_secs(2), engine.infer(request()))
                .await
                .expect("invalid restoration must finish cleanup")
                .is_err()
        );
        assert!(observation.attempted_chunks.lock().unwrap().is_empty());
        assert!(observation.retained.lock().unwrap().is_empty());
        assert_eq!(state.acknowledgements.load(Ordering::Relaxed), 0);
        assert_eq!(state.cancellations.load(Ordering::Relaxed), 1);
        assert_eq!(
            engine
                .active_inner()
                .expect("engine execution is open")
                .total_prefill_tokens
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            engine
                .active_inner()
                .expect("engine execution is open")
                .prefix_cache_hits
                .load(Ordering::Relaxed),
            0
        );
        assert!(engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .is_empty());
        engine.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn prefix_restore_profile_accounts_only_for_committed_prefill_work() {
    for behavior in [
        RestoreBehavior::Hit,
        RestoreBehavior::Miss,
        RestoreBehavior::FailedAcknowledgement,
    ] {
        let path = resource_trace_temp_path("prefix-work-accounting");
        let (engine, executor, _) = engine(behavior, 3, Some(path.clone()));
        let result = tokio::time::timeout(Duration::from_secs(2), engine.infer(request()))
            .await
            .unwrap();
        let failed = matches!(behavior, RestoreBehavior::FailedAcknowledgement);
        assert_eq!(result.is_err(), failed);
        flush_engine_profile_events(&engine);
        let events = read_engine_profile_events(&path);
        let restored = events
            .iter()
            .filter(|event| event.phase == "vnext.prefix_restore")
            .map(|event| event.shape["restored_tokens"].as_u64().unwrap())
            .sum::<u64>();
        let committed = events
            .iter()
            .filter(|event| event.phase == "vnext.prefill_chunk_completed")
            .map(|event| {
                (
                    event.shape["start_token"].as_u64().unwrap(),
                    event.shape["end_token"].as_u64().unwrap(),
                    event.shape["computed_tokens"].as_u64().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let executed = executor
            .attempted_chunks
            .lock()
            .unwrap()
            .iter()
            .map(|chunk| {
                (
                    chunk.tokens_processed() as u64,
                    chunk.end() as u64,
                    chunk.tokens_to_process() as u64,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(committed, executed);
        if failed {
            assert_eq!(restored, 0);
            assert!(committed.is_empty());
        } else {
            assert_eq!(
                restored + committed.iter().map(|(_, _, n)| n).sum::<u64>(),
                5
            );
            assert_eq!(committed.first().unwrap().0, restored);
        }
        engine.shutdown().await.unwrap();
        std::fs::remove_file(path).unwrap();
    }
}

#[tokio::test]
async fn attempt_prefix_facts_require_acknowledgement_and_preserve_suffix_ranges() {
    use crate::continuous_engine::attempt_observation_tests::AttemptRecorder;
    use ferrum_execution::execution_attempt::*;
    for behavior in [
        RestoreBehavior::Hit,
        RestoreBehavior::Miss,
        RestoreBehavior::FailedAcknowledgement,
        RestoreBehavior::AcknowledgedWithObserverError,
    ] {
        let recorder = Arc::new(AttemptRecorder::default());
        let (engine, execution, restore) =
            engine_observed(behavior, 3, None, Some(recorder.clone()));
        let result = engine.infer(request()).await;
        let samples = recorder.take();
        let failed = matches!(
            behavior,
            RestoreBehavior::FailedAcknowledgement | RestoreBehavior::AcknowledgedWithObserverError
        );
        assert_eq!(result.is_err(), failed);
        if failed {
            assert!(samples.is_empty(), "a restore is not an execute invocation");
            assert!(execution.attempted_chunks.lock().unwrap().is_empty());
            assert_eq!(
                restore.acknowledgements.load(Ordering::Relaxed),
                u64::from(matches!(
                    behavior,
                    RestoreBehavior::AcknowledgedWithObserverError
                ))
            );
        } else {
            let mut start = if matches!(behavior, RestoreBehavior::Hit) {
                3
            } else {
                0
            };
            for sample in &samples {
                assert_eq!(sample.eligibility, ExecutionAttemptEligibility::Completed);
                assert_eq!(sample.participants.len(), 1);
                let member = &sample.participants[0];
                assert_eq!(
                    member.prefix_restore,
                    if matches!(behavior, RestoreBehavior::Hit) {
                        AttemptPrefixRestore::Restored { tokens: 3 }
                    } else {
                        AttemptPrefixRestore::NoneObserved
                    }
                );
                assert_eq!(member.internally_reused_tokens, None);
                let AttemptParticipantOutcome::Completed {
                    work,
                    frontier_committed: true,
                } = member.outcome
                else {
                    panic!("suffix was not actually committed");
                };
                assert_eq!(work.range().unwrap().start(), start);
                start = work.range().unwrap().end();
            }
            assert_eq!(start, 5);
        }
        engine.shutdown().await.unwrap();
    }
}
