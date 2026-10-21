//! Actual Metal checkpoint capture/acknowledgement -> Engine attempt facts ->
//! reference learning. No restore decision or training summary is fabricated.

use super::ContinuousBatchEngine;
use ferrum_execution::{
    execution_attempt::*,
    executor_lifecycle::{ExecutorOwner, ExecutorShutdownPhase},
    ModelExecutor,
};
use ferrum_interfaces::{
    engine::{InferenceEngine, LlmInferenceEngine},
    tokenizer::{TokenizerInfo, TokenizerType},
    Tokenizer,
};
use ferrum_models::test_support::stateful_program;
use ferrum_scheduler::{cost_prediction::*, ContinuousBatchScheduler};
use ferrum_slo::{ReferenceEstimator, ReferenceEstimatorConfig};
use ferrum_types::{
    FerrumError, InferenceRequest, InferenceResponse, ObservabilityProfileDetail, Result,
    SchedulerSloConfig, SpecialTokens, TokenId,
};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::sync::{watch, Notify};

/// Matches the real fixture's declared BOS/EOS/PAD and complete vocabulary.
/// This text mapping supplies inputs; all model math and checkpoint copies are
/// performed by the original admitted ProgramExecutor and Metal providers.
struct FixtureTokenizer {
    special: SpecialTokens,
    vocabulary: Vec<String>,
}

impl FixtureTokenizer {
    fn new() -> Self {
        Self {
            special: SpecialTokens {
                bos_token: Some(TokenId::new(1)),
                eos_token: Some(TokenId::new(2)),
                pad_token: Some(TokenId::new(0)),
                ..Default::default()
            },
            vocabulary: (0..stateful_program::VOCAB)
                .map(|id| format!("t{id}"))
                .collect(),
        }
    }
}

impl Tokenizer for FixtureTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        let mut tokens = Vec::new();
        if add_special {
            tokens.push(self.special.bos_token.unwrap());
        }
        for word in text.split_whitespace() {
            tokens.push(
                self.token_id(word)
                    .ok_or_else(|| FerrumError::request_validation("unknown fixture token"))?,
            );
        }
        Ok(tokens)
    }

    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        let mut text = String::new();
        for &token in tokens {
            if !skip_special || !self.is_special_token(token) {
                text.push_str(self.token_text(token).ok_or_else(|| {
                    FerrumError::request_validation("fixture token is out of vocabulary")
                })?);
            }
        }
        Ok(text)
    }

    fn decode_incremental(&self, _previous: &[TokenId], next: TokenId) -> Result<String> {
        self.decode(&[next], true)
    }

    fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }

    fn token_id(&self, text: &str) -> Option<TokenId> {
        self.vocabulary
            .iter()
            .position(|value| value == text)
            .map(|id| TokenId::new(id as u32))
    }

    fn token_text(&self, token: TokenId) -> Option<&str> {
        self.vocabulary
            .get(token.get() as usize)
            .map(String::as_str)
    }

    fn info(&self) -> TokenizerInfo {
        TokenizerInfo {
            tokenizer_type: TokenizerType::Custom,
            vocab_size: self.vocab_size(),
            special_tokens: self.special.clone(),
            supports_incremental: true,
            supports_chat_template: false,
            max_token_length: Some(3),
            model_name: Some("causal-prefix-feedback-fixture".into()),
        }
    }
}

struct AttemptWitness {
    observer: Arc<dyn ExecutionAttemptObserver>,
    samples: Mutex<Vec<ExecutionAttemptSummary>>,
    published: AtomicUsize,
    recorded: Notify,
}

impl AttemptWitness {
    async fn take(&self, expected: usize) -> Vec<ExecutionAttemptSummary> {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let recorded = self.recorded.notified();
                // The callback publishes this count only after releasing the
                // sample lock. Waiting must not contend with its try_lock.
                if self.published.load(Ordering::Acquire) >= expected {
                    let mut samples = self.samples.lock().unwrap();
                    assert_eq!(samples.len(), expected, "unexpected extra actual attempt");
                    self.published.store(0, Ordering::Release);
                    return std::mem::take(&mut *samples);
                }
                recorded.await;
            }
        })
        .await
        .expect("actual attempt reconciliation did not finish")
    }
}

impl ExecutionAttemptObserver for AttemptWitness {
    fn limits(&self) -> ExecutionAttemptLimits {
        self.observer.limits()
    }

    fn try_record(&self, summary: &ExecutionAttemptSummary) -> ExecutionAttemptDelivery {
        let delivery = self.observer.try_record(summary);
        assert_eq!(delivery, ExecutionAttemptDelivery::Recorded);
        let mut samples = self
            .samples
            .try_lock()
            .expect("witness access does not overlap execution");
        assert!(samples.len() < 8, "bounded test witness exhausted");
        samples.push(summary.clone());
        drop(samples);
        self.published.fetch_add(1, Ordering::Release);
        self.recorded.notify_one();
        delivery
    }

    fn record_drop(&self, reason: ExecutionAttemptDrop) {
        self.observer.record_drop(reason);
    }
}

struct PredictionWitness {
    original: Arc<dyn CostSnapshot>,
    queries: Arc<Mutex<Vec<(Vec<CostWork>, CostPrediction)>>>,
}

impl CostSnapshot for PredictionWitness {
    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        self.original.query_limits()
    }
    fn revision(&self) -> u64 {
        self.original.revision()
    }

    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction {
        let prediction = self.original.predict(query, now);
        let mut queries = self
            .queries
            .try_lock()
            .expect("query witness is read between requests");
        assert!(queries.len() < 128, "bounded query witness exhausted");
        queries.push((query.work.to_vec(), prediction));
        prediction
    }
}

struct SourceWitness {
    original: Arc<dyn CostSnapshotSource>,
    queries: Arc<Mutex<Vec<(Vec<CostWork>, CostPrediction)>>>,
}

impl CostSnapshotSource for SourceWitness {
    fn load(&self) -> Arc<dyn CostSnapshot> {
        Arc::new(PredictionWitness {
            original: self.original.load(),
            queries: Arc::clone(&self.queries),
        })
    }

    fn subscribe(&self) -> watch::Receiver<u64> {
        self.original.subscribe()
    }
}

fn request() -> InferenceRequest {
    let mut request = InferenceRequest::new("t3 t10 t17 t24", "causal-prefix-feedback-fixture");
    request.sampling_params.max_tokens = 1;
    request.sampling_params.temperature = 0.0;
    request
}

fn actual_work(sample: &ExecutionAttemptSummary) -> CostWork {
    assert_eq!(sample.eligibility, ExecutionAttemptEligibility::Completed);
    assert_eq!(sample.participants.len(), 1);
    let participant = &sample.participants[0];
    assert!(!participant.product_failed);
    assert!(matches!(
        participant.outcome,
        AttemptParticipantOutcome::Completed {
            frontier_committed: true,
            ..
        }
    ));
    CostWork {
        kind: participant.kind,
        requested: participant.requested,
        context_tokens: participant.context_tokens,
        prefix_restore: participant.prefix_restore,
        internally_reused_tokens: participant.internally_reused_tokens,
    }
}

fn predict(snapshot: &dyn CostSnapshot, work: CostWork) -> CostPrediction {
    snapshot.predict(
        &CostQuery {
            work: &[work],
            work_is_exact: true,
            execution: ferrum_scheduler::cost_prediction::CostExecution::SingleInvocation,
        },
        Instant::now(),
    )
}

async fn learned(
    source: &dyn CostSnapshotSource,
    work: CostWork,
    samples: usize,
) -> Arc<dyn CostSnapshot> {
    let mut changes = source.subscribe();
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let snapshot = source.load();
            if matches!(predict(snapshot.as_ref(), work), CostPrediction::Known { samples: actual, .. } if actual == samples) {
                return snapshot;
            }
            changes.changed().await.expect("original reference worker remains live");
        }
    }).await.expect("actual restored domain was not learned")
}

fn same_response(actual: &InferenceResponse, cold: &InferenceResponse) {
    assert_eq!(actual.tokens, cold.tokens);
    assert_eq!(actual.text, cold.text);
    assert_eq!(actual.finish_reason, cold.finish_reason);
    assert_eq!(actual.usage.prompt_tokens, cold.usage.prompt_tokens);
    assert_eq!(actual.usage.completion_tokens, cold.usage.completion_tokens);
    assert_eq!(actual.usage.total_tokens, cold.usage.total_tokens);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires actual Metal language providers and native checkpoint copies"]
async fn native_prefix_feedback_metal_learns_acknowledged_restore_and_retires_original_owner() {
    let stateful_program::Fixture {
        directory,
        executor,
        runtime,
        mut engine_config,
    } = stateful_program::metal();
    assert!(executor.supports_plan_runtime_prefix_restore());
    let initial = executor.cache_metrics_snapshot().unwrap();
    assert_eq!(initial["enabled"], true);
    assert_eq!(initial["entries"], 0);
    let owner = ExecutorOwner::new(Box::new(executor));
    let reference_config = ReferenceEstimatorConfig {
        min_samples: 1,
        snapshot_ttl_ms: 60_000,
        ..Default::default()
    };
    let estimator = ReferenceEstimator::spawn(reference_config).unwrap();
    let source = estimator.source();
    let attempts = Arc::new(AttemptWitness {
        observer: estimator.observer(),
        samples: Mutex::new(Vec::new()),
        published: AtomicUsize::new(0),
        recorded: Notify::new(),
    });
    let queries = Arc::new(Mutex::new(Vec::new()));
    let observed_source = Arc::new(SourceWitness {
        original: source.clone(),
        queries: queries.clone(),
    });
    // A declared large budget preserves the original legal prefix/suffix split.
    // This test proves distinct learning domains and real consumption, not an SLO.
    engine_config.runtime.profile_detail = ObservabilityProfileDetail::Off;
    engine_config.scheduler.slo = Some(SchedulerSloConfig {
        ttft_ms: 60_000.0,
        tpot_ms: 60_000.0,
        itl_ms: 60_000.0,
        query: Default::default(),
        planning: Default::default(),
        joint: Default::default(),
        strategy: Default::default(),
        unknown_stopping: Default::default(),
        unknown_cost: Default::default(),
    });
    let tokenizer = Arc::new(FixtureTokenizer::new());
    assert_eq!(tokenizer.encode(&request().prompt, true).unwrap().len(), 5);
    let engine = ContinuousBatchEngine::new_observed(
        engine_config.clone(),
        Arc::new(ContinuousBatchScheduler::new(
            engine_config.scheduler.clone(),
        )),
        tokenizer,
        Arc::new(crate::registry::GreedySampler),
        owner.scope().unwrap(),
        None,
        Some(attempts.clone()),
        Some(observed_source),
    )
    .unwrap();
    // Use the same real readiness boundary as EngineBuilder. In particular,
    // direct construction alone must not be treated as prepared execution.
    engine.prepare_startup().await.unwrap();

    let cold = tokio::time::timeout(Duration::from_secs(30), engine.infer(request()))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(cold.response().usage.prompt_tokens, 5);
    assert_eq!(cold.response().usage.completion_tokens, 1);
    let cold_samples = attempts.take(2).await;
    let prefix = actual_work(&cold_samples[0]);
    let cold_suffix = actual_work(&cold_samples[1]);
    assert_eq!((prefix.requested.start(), prefix.requested.end()), (0, 4));
    assert_eq!(
        (cold_suffix.requested.start(), cold_suffix.requested.end()),
        (4, 5)
    );
    assert_eq!(cold_suffix.kind, AttemptWorkKind::Prefill);
    assert_eq!(cold_suffix.context_tokens, 5);
    assert_eq!(
        cold_suffix.prefix_restore,
        AttemptPrefixRestore::NoneObserved
    );
    assert_eq!(cold_suffix.internally_reused_tokens, None);
    let captured = engine.cache_metrics_snapshot().unwrap();
    assert!(captured["entries"].as_u64().unwrap() > 0);
    assert!(captured["bytes"].as_u64().unwrap() > 0);
    assert_eq!(captured["hits"], 0);
    assert_eq!(captured["saved_prefill_tokens"], 0);

    let cold_snapshot = learned(source.as_ref(), cold_suffix, 1).await;
    // This is a prediction query, not a submitted observation or a restore fact.
    let prospective_restored = CostWork {
        prefix_restore: AttemptPrefixRestore::Restored { tokens: 4 },
        ..cold_suffix
    };
    assert_eq!(
        predict(cold_snapshot.as_ref(), prospective_restored),
        CostPrediction::Unknown(CostUnknownReason::NoCoverage)
    );

    let restored = tokio::time::timeout(Duration::from_secs(30), engine.infer(request()))
        .await
        .unwrap()
        .unwrap();
    same_response(restored.response(), cold.response());
    let restored_samples = attempts.take(1).await;
    let restored_work = actual_work(&restored_samples[0]);
    assert_eq!(restored_work, prospective_restored);
    assert_ne!(
        restored_samples[0].participants[0].request_id,
        cold_samples[0].participants[0].request_id
    );
    let restored_cache = engine.cache_metrics_snapshot().unwrap();
    assert_eq!(restored_cache["hits"], 1);
    assert_eq!(restored_cache["saved_prefill_tokens"], 4);
    let first_restored_snapshot = learned(source.as_ref(), restored_work, 1).await;
    assert!(matches!(
        predict(first_restored_snapshot.as_ref(), cold_suffix),
        CostPrediction::Known { samples: 1, .. }
    ));
    queries.lock().unwrap().clear();

    let repeated = tokio::time::timeout(Duration::from_secs(30), engine.infer(request()))
        .await
        .unwrap()
        .unwrap();
    same_response(repeated.response(), cold.response());
    let repeated_samples = attempts.take(1).await;
    assert_eq!(actual_work(&repeated_samples[0]), restored_work);
    assert!(
        queries.lock().unwrap().iter().any(|(work, prediction)| {
            work.as_slice() == &[restored_work]
                && matches!(prediction, CostPrediction::Known { samples: 1, .. })
        }),
        "the actual scheduler must consume the learned restored domain"
    );
    let repeated_cache = engine.cache_metrics_snapshot().unwrap();
    assert_eq!(repeated_cache["hits"], 2);
    assert_eq!(repeated_cache["saved_prefill_tokens"], 8);
    let both = learned(source.as_ref(), restored_work, 2).await;
    assert!(matches!(
        predict(both.as_ref(), cold_suffix),
        CostPrediction::Known { samples: 1, .. }
    ));

    engine.shutdown().await.unwrap();
    engine.wait_for_shutdown_completion().await.unwrap();
    drop((engine, cold, restored, repeated));
    let exit = estimator.shutdown().await.unwrap();
    assert_eq!(exit.stats.queued_records, 0);
    assert_eq!(exit.stats.queued_bytes, 0);
    assert_eq!(
        exit.stats.invalid_attempts + exit.stats.excluded_attempts + exit.stats.worker_failures,
        0
    );
    assert_eq!(
        exit.stats.dropped_full
            + exit.stats.dropped_bytes
            + exit.stats.dropped_oversize
            + exit.stats.dropped_closed,
        0
    );
    let retired = tokio::time::timeout(Duration::from_secs(10), owner.begin_shutdown().wait())
        .await
        .unwrap();
    assert_eq!(retired.phase(), ExecutorShutdownPhase::Retired);
    retired.result().unwrap();
    assert!(
        runtime.upgrade().is_none(),
        "original plan, completion worker, and checkpoint owners retired"
    );
    drop(directory);
}
