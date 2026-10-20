//! Actual tiny CPU math and the original worker's whole-model close custody.

use super::*;
use crate::common::llm::{
    ModelExecutionRecovery, ModelKvPreparation, ModelKvReservation, ModelRetirementOutcome,
    ModelTokenBatchOutcome, ModelTokenCloseOutcome, ModelTokenInput,
};
use crate::models::LlamaFamilyModel;
use crate::test_support::{tiny_llama_model, tiny_model_info, TinyLlamaConfig};
use ferrum_execution::executor_lifecycle::{
    ExecutorOwner, ExecutorShutdownObserver, ExecutorShutdownPhase,
};
use ferrum_execution::model_executor::{
    ExecutorPrefillAdmission, ExecutorSamplingOutput, LogitsReturnPolicy, PrefillChunk,
};
use ferrum_execution::token_execution::*;
use ferrum_kernels::backend::cpu::CpuBackend;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{mpsc, Weak};
use std::time::Duration;
use tokio::sync::oneshot;

const WAIT: Duration = Duration::from_secs(10);

#[test]
fn model_shutdown_default_does_not_manufacture_retirement() {
    struct NoRetirement(LlmRuntimeConfig);
    impl DecoderOnlyLLM for NoRetirement {
        fn config(&self) -> &LlmRuntimeConfig {
            &self.0
        }
        fn prefill(&mut self, _: &str, _: &[u32]) -> Vec<f32> {
            panic!("unused numerical entry")
        }
        fn decode(&mut self, _: &str, _: u32, _: u32) -> Vec<f32> {
            panic!("unused numerical entry")
        }
        fn release(&mut self, _: &str) {}
        fn reset(&mut self) {}
    }
    let mut reference = tiny_llama_model(&TinyLlamaConfig::default());
    let mut model = NoRetirement(reference.config().clone());
    assert!(matches!(
        model.retire_model(),
        ModelRetirementOutcome::Retained {
            error: FerrumError::Unsupported { .. },
        }
    ));
    reference.close().unwrap();
}

async fn retired(
    mut observer: ExecutorShutdownObserver,
) -> ferrum_execution::executor_lifecycle::ExecutorShutdownReport {
    tokio::time::timeout(WAIT, async move {
        loop {
            let report = observer.snapshot();
            if report.phase() == ExecutorShutdownPhase::Retired {
                return report;
            }
            observer.changed().await;
        }
    })
    .await
    .expect("original worker must actually join")
}

async fn actual_prefill(
    executor: &dyn ModelExecutor,
    reference: &mut LlamaFamilyModel<CpuBackend>,
) -> ExecutionFrontier {
    let id = RequestId::new();
    let prompt = [TokenId(1), TokenId(2), TokenId(3)];
    let TokenAdmissionOutcome::Admitted(frontier) = executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(&id, &prompt, 8),
            adapter: None,
        })
        .await
        .unwrap()
    else {
        panic!("real tiny admission must succeed")
    };
    let policy = LogitsReturnPolicy::FullLogits;
    let work = [ExecutionWork::new(
        &frontier,
        TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        &policy,
    )
    .unwrap()];
    let ExecutionBatchOutcome::Completed(batch) = executor.execute_tokens(&work).await.unwrap()
    else {
        panic!("real tiny prefill must complete")
    };
    let mut rows = batch.into_completions();
    assert_eq!(rows.len(), 1);
    let row = rows.remove(0);
    let expected = reference.prefill_internal("reference", &[1, 2, 3]);
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(actual)) = row.product()
    else {
        panic!("must retain the entire real vocabulary")
    };
    assert_eq!(actual.len(), expected.len());
    for (&actual, expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite() && expected.is_finite());
        assert_eq!(actual, expected);
    }
    row.next_frontier().clone()
}

#[tokio::test]
async fn model_shutdown_cpu_cold_owner_releases_original_slot_without_admission() {
    let cfg = TinyLlamaConfig::default();
    let executor = LlmExecutor::new(Box::new(tiny_llama_model(&cfg)), tiny_model_info(&cfg));
    assert!(executor.token_worker.get().is_none());
    let slot = Arc::clone(&executor.model);
    let owner = ExecutorOwner::new(Box::new(executor));
    let report = retired(owner.begin_shutdown()).await;
    report.result().unwrap();
    assert!(
        slot.lock().is_none(),
        "checked retirement must release the actual model Box"
    );
    assert!(owner.scope().is_err());
}

#[tokio::test]
async fn model_shutdown_cpu_shared_scope_keeps_peer_numerical_execution_live() {
    let cfg = TinyLlamaConfig::default();
    let executor = LlmExecutor::new(Box::new(tiny_llama_model(&cfg)), tiny_model_info(&cfg));
    let slot = Arc::clone(&executor.model);
    let owner = ExecutorOwner::new(Box::new(executor));
    let first = owner.scope().unwrap();
    let second = owner.scope().unwrap();
    drop(first);
    let observer = owner.begin_shutdown();
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Draining);
    let mut reference = tiny_llama_model(&cfg);
    let frontier = actual_prefill(second.as_ref(), &mut reference).await;
    // The remaining use scope still owns actual execution even after the
    // final owner seals registration; another Engine cannot retire it early.
    assert!(slot.lock().is_some());
    assert!(frontier.sequence().observed_end().is_none());
    let close = second.cancel_tokens(frontier.sequence()).unwrap();
    assert!(matches!(
        close.disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    drop(second);
    retired(observer).await.result().unwrap();
    assert!(slot.lock().is_none());
    reference.close().unwrap();
}

struct Gate {
    entered: oneshot::Sender<()>,
    release: mpsc::Receiver<()>,
}

struct Release(Option<mpsc::Sender<()>>);
impl Release {
    fn now(&mut self) {
        if let Some(release) = self.0.take() {
            let _ = release.send(());
        }
    }
}
impl Drop for Release {
    fn drop(&mut self) {
        self.now();
    }
}

#[derive(Default)]
struct RetirementControl {
    gate: Mutex<Option<Gate>>,
    blocked: AtomicBool,
    retired_error: AtomicBool,
    attempts: AtomicUsize,
    actual_retirements: AtomicUsize,
    forwards: AtomicUsize,
}

struct UnblockRetirement(Arc<RetirementControl>);
impl Drop for UnblockRetirement {
    fn drop(&mut self) {
        self.0.blocked.store(false, Ordering::SeqCst);
    }
}

struct ObservedRetirement {
    model: LlamaFamilyModel<CpuBackend>,
    control: Arc<RetirementControl>,
    _lifetime: Arc<()>,
}

impl DecoderOnlyLLM for ObservedRetirement {
    fn config(&self) -> &LlmRuntimeConfig {
        self.model.config()
    }
    // This real CPU fixture exercises the worker protocol, not CUDA qualification.
    fn token_execution_mode(&self) -> ModelTokenExecutionMode {
        ModelTokenExecutionMode::ManagedWorker
    }
    fn kv_capacity(&self) -> usize {
        self.model.kv_capacity()
    }
    fn prepare_token_kv(&mut self, request: ModelKvPreparation<'_>) -> Result<ModelKvReservation> {
        self.model.prepare_token_kv(request)
    }
    fn execute_token_batch(&mut self, work: &[ModelTokenInput<'_>]) -> ModelTokenBatchOutcome {
        self.control.forwards.fetch_add(1, Ordering::SeqCst);
        self.model.execute_token_batch(work)
    }
    fn recover_token_execution(&mut self) -> ModelExecutionRecovery {
        self.model.recover_token_execution()
    }
    fn close_token_execution(&mut self, id: &str) -> ModelTokenCloseOutcome {
        self.model.close_token_execution(id)
    }
    fn retire_model(&mut self) -> ModelRetirementOutcome {
        self.control.attempts.fetch_add(1, Ordering::SeqCst);
        if let Some(gate) = self.control.gate.lock().take() {
            let _ = gate.entered.send(());
            gate.release
                .recv_timeout(WAIT)
                .expect("test must release original retirement");
        }
        if self.control.blocked.load(Ordering::SeqCst) {
            return ModelRetirementOutcome::Retained {
                error: FerrumError::backend("first checked retirement failure"),
            };
        }
        let outcome = self.model.retire_model();
        if matches!(outcome, ModelRetirementOutcome::Retired { .. }) {
            self.control
                .actual_retirements
                .fetch_add(1, Ordering::SeqCst);
            if self.control.retired_error.load(Ordering::SeqCst) {
                return ModelRetirementOutcome::Retired {
                    error: Some(FerrumError::backend(
                        "terminal operation error after actual retirement",
                    )),
                };
            }
        }
        outcome
    }
    fn prefill(&mut self, _: &str, _: &[u32]) -> Vec<f32> {
        panic!("typed token path required")
    }
    fn decode(&mut self, _: &str, _: u32, _: u32) -> Vec<f32> {
        panic!("typed token path required")
    }
    fn release(&mut self, id: &str) {
        self.model.release(id);
    }
    fn reset(&mut self) {
        self.model.reset();
    }
}

fn managed_owner(control: Arc<RetirementControl>) -> (ExecutorOwner, ModelSlot, Weak<()>) {
    let cfg = TinyLlamaConfig::default();
    let lifetime = Arc::new(());
    let weak = Arc::downgrade(&lifetime);
    let executor = LlmExecutor::new(
        Box::new(ObservedRetirement {
            model: tiny_llama_model(&cfg),
            control,
            _lifetime: lifetime,
        }),
        tiny_model_info(&cfg),
    );
    let slot = Arc::clone(&executor.model);
    (ExecutorOwner::new(Box::new(executor)), slot, weak)
}

#[tokio::test]
async fn model_shutdown_cpu_cancelled_waiter_keeps_original_retirement_and_join() {
    let control = Arc::new(RetirementControl::default());
    let (entered, started) = oneshot::channel();
    let (release, receiver) = mpsc::channel();
    let mut release = Release(Some(release));
    *control.gate.lock() = Some(Gate {
        entered,
        release: receiver,
    });
    let (owner, slot, lifetime) = managed_owner(Arc::clone(&control));
    let scope = owner.scope().unwrap();
    let mut reference = tiny_llama_model(&TinyLlamaConfig::default());
    let frontier = actual_prefill(scope.as_ref(), &mut reference).await;
    drop(scope);
    let observer = owner.begin_shutdown();
    tokio::time::timeout(WAIT, started).await.unwrap().unwrap();
    assert!(frontier.sequence().observed_end().is_some());
    assert!(lifetime.upgrade().is_some());
    let mut wait = Box::pin(observer.wait());
    assert!(matches!(
        futures::poll!(wait.as_mut()),
        std::task::Poll::Pending
    ));
    drop(wait);
    release.now();
    retired(owner.begin_shutdown()).await.result().unwrap();
    assert!(slot.lock().is_none());
    assert!(lifetime.upgrade().is_none());
    assert_eq!(control.actual_retirements.load(Ordering::SeqCst), 1);
    assert_eq!(control.forwards.load(Ordering::SeqCst), 1);
    reference.close().unwrap();
}

#[tokio::test]
async fn model_shutdown_cpu_retained_failure_retries_same_owner_and_preserves_first_error() {
    let control = Arc::new(RetirementControl::default());
    control.blocked.store(true, Ordering::SeqCst);
    let _unblock = UnblockRetirement(Arc::clone(&control));
    let (owner, slot, lifetime) = managed_owner(Arc::clone(&control));
    let observer = owner.begin_shutdown();
    let report = tokio::time::timeout(WAIT, observer.wait()).await.unwrap();
    assert_eq!(report.phase(), ExecutorShutdownPhase::Retained);
    assert!(report
        .result()
        .unwrap_err()
        .to_string()
        .contains("first checked retirement failure"));
    assert!(slot.lock().is_some());
    assert!(lifetime.upgrade().is_some());
    control.blocked.store(false, Ordering::SeqCst);
    let report = retired(observer).await;
    assert!(report
        .result()
        .unwrap_err()
        .to_string()
        .contains("first checked retirement failure"));
    assert!(slot.lock().is_none());
    assert!(lifetime.upgrade().is_none());
    assert!(control.attempts.load(Ordering::SeqCst) >= 2);
    assert_eq!(control.actual_retirements.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn model_shutdown_cpu_retired_error_does_not_retry_or_keep_weights() {
    let control = Arc::new(RetirementControl::default());
    control.retired_error.store(true, Ordering::SeqCst);
    let (owner, slot, lifetime) = managed_owner(Arc::clone(&control));
    let report = retired(owner.begin_shutdown()).await;
    assert!(report
        .result()
        .unwrap_err()
        .to_string()
        .contains("terminal operation error after actual retirement"));
    assert!(slot.lock().is_none());
    assert!(lifetime.upgrade().is_none());
    assert_eq!(control.attempts.load(Ordering::SeqCst), 1);
    assert_eq!(control.actual_retirements.load(Ordering::SeqCst), 1);
}
