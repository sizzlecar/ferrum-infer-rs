//! Real tiny-model tests of the managed worker. CPU cases qualify queue and
//! delivery ownership only; selected Metal/CUDA cases run the actual backend.

use super::*;
use crate::common::llm::{
    ModelExecutionRecovery, ModelTokenBatchOutcome, ModelTokenCloseOutcome, ModelTokenInput,
};
use crate::common::LlmRuntimeConfig;
use crate::models::llama_family::LlamaFamilyConfig;
use crate::models::LlamaFamilyModel;
use crate::test_support::{tiny_model_info, SyntheticLlamaLoader, TinyLlamaConfig};
use ferrum_execution::model_executor::{ExecutorPrefillAdmission, PrefillChunk};
use ferrum_kernels::backend::{cpu::CpuBackend, MoeLlmBackend};
use std::future::Future;
use std::sync::{atomic::AtomicBool, mpsc};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::oneshot;

mod capacity;
#[cfg(any(
    feature = "cuda",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
mod gpu_capacity;

struct Gate {
    entered: oneshot::Sender<()>,
    release: mpsc::Receiver<()>,
}

#[derive(Default)]
struct Control {
    gate: Mutex<Option<Gate>>,
    close_once: AtomicBool,
    batches: AtomicU64,
    omit_middle_result: AtomicBool,
}

impl Control {
    fn block_next(&self) -> (oneshot::Receiver<()>, mpsc::Sender<()>) {
        let (entered, started) = oneshot::channel();
        let (release, receiver) = mpsc::channel();
        assert!(self
            .gate
            .lock()
            .replace(Gate {
                entered,
                release: receiver
            })
            .is_none());
        (started, release)
    }
}

struct ManagedModel<B: MoeLlmBackend> {
    model: Arc<Mutex<LlamaFamilyModel<B>>>,
    control: Arc<Control>,
    config: LlmRuntimeConfig,
}

impl<B: MoeLlmBackend> DecoderOnlyLLM for ManagedModel<B> {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.config
    }
    // This test-only wrapper exercises qualification before changing the
    // production device capability. It delegates every numerical operation.
    fn supports_direct_token_execution(&self) -> bool {
        true
    }
    fn token_execution_requires_worker(&self) -> bool {
        true
    }
    fn kv_capacity(&self) -> usize {
        self.model.lock().kv_capacity()
    }
    fn prepare_token_kv(
        &mut self,
        id: &str,
        capacity: usize,
        prompt: usize,
    ) -> Result<ModelKvReservation> {
        self.model.lock().prepare_token_kv(id, capacity, prompt)
    }
    fn observe_token_allocator_reclaim(
        &self,
        blocked: &str,
        domain: &AllocatorCapacityDomain,
        candidates: &[&str],
    ) -> Result<Option<TokenAllocatorReclaim>> {
        self.model
            .lock()
            .observe_token_allocator_reclaim(blocked, domain, candidates)
    }
    fn execute_token_batch(&mut self, inputs: &[ModelTokenInput<'_>]) -> ModelTokenBatchOutcome {
        if let Some(gate) = self.control.gate.lock().take() {
            let _ = gate.entered.send(());
            gate.release
                .recv_timeout(Duration::from_secs(10))
                .expect("test must release actual worker");
        }
        self.control.batches.fetch_add(1, Ordering::Relaxed);
        let mut result = self.model.lock().execute_token_batch(inputs);
        if self
            .control
            .omit_middle_result
            .swap(false, Ordering::AcqRel)
        {
            // All numerical work has actually completed. Remove the middle
            // result to test that positional rows are no longer attributable.
            {
                let model = self.model.lock();
                for input in inputs {
                    let end = match input.work {
                        ModelTokenWork::Prefill { tokens, position } => position + tokens.len(),
                        ModelTokenWork::Decode { position, .. } => position as usize + 1,
                    };
                    assert!(model.kv_caches[input.cache_id]
                        .iter()
                        .all(|cache| cache.len == end));
                }
            }
            match &mut result {
                ModelTokenBatchOutcome::Serial(rows) => {
                    assert_eq!(rows.len(), 3);
                    drop(rows.remove(1));
                }
                ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::Terminal(Ok(rows))) => {
                    assert_eq!(rows.len(), 3);
                    drop(rows.remove(1));
                }
                _ => panic!("malformed-row injection requires actual successful model work"),
            }
        }
        result
    }
    fn recover_token_execution(&mut self) -> ModelExecutionRecovery {
        self.model.lock().recover_token_execution()
    }
    fn close_token_execution(&mut self, id: &str) -> ModelTokenCloseOutcome {
        if self.control.close_once.swap(false, Ordering::AcqRel) {
            if let Some(gate) = self.control.gate.lock().take() {
                let _ = gate.entered.send(());
                gate.release
                    .recv_timeout(Duration::from_secs(10))
                    .expect("test must release actual close observer");
            }
            // Conservative protocol fault after real model preparation. This
            // is not an emulated GPU completion or a hardware failure claim.
            return ModelTokenCloseOutcome::Held {
                error: FerrumError::backend("first close observation failed"),
            };
        }
        self.model.lock().close_token_execution(id)
    }
    fn prefill(&mut self, _: &str, _: &[u32]) -> Vec<f32> {
        panic!("no legacy Tensor adapter")
    }
    fn decode(&mut self, _: &str, _: u32, _: u32) -> Vec<f32> {
        panic!("no legacy Tensor adapter")
    }
    fn release(&mut self, _: &str) {
        panic!("checked closure required")
    }
    fn reset(&mut self) {
        panic!("closing one request must not reset peers")
    }
}

fn real_model<B: MoeLlmBackend>(paged: bool) -> LlamaFamilyModel<B> {
    let config = LlamaFamilyConfig {
        hidden_size: 256,
        intermediate_size: 512,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 128,
        num_layers: 1,
        vocab_size: 128,
        max_seq_len: 32,
        ..Default::default()
    };
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut model = LlamaFamilyModel::new(config, &loader).unwrap();
    model.runtime_env.metal_paged_kv = Some(paged);
    model.runtime_env.prefix_cache = false;
    model.runtime_env.cuda_graph = false;
    model.runtime_env.kv_capacity = Some(32);
    model.runtime_env.kv_max_blocks = Some(16);
    model.runtime_env.paged_max_seqs = 8;
    model
}

fn fixture<B: MoeLlmBackend>(
    paged: bool,
) -> (LlmExecutor, Arc<Mutex<LlamaFamilyModel<B>>>, Arc<Control>) {
    let model = real_model::<B>(paged);
    let config = model.config().clone();
    let model = Arc::new(Mutex::new(model));
    let control = Arc::new(Control::default());
    let info = tiny_model_info(&TinyLlamaConfig {
        hidden_size: 256,
        intermediate_size: 512,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 128,
        num_layers: 1,
        vocab_size: 128,
        max_seq_len: 32,
    });
    (
        LlmExecutor::new(
            Box::new(ManagedModel {
                model: Arc::clone(&model),
                control: Arc::clone(&control),
                config,
            }),
            info,
        ),
        model,
        control,
    )
}

async fn admit(
    executor: &LlmExecutor,
    request: &RequestId,
    prompt: &[TokenId],
) -> ExecutionFrontier {
    match executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(request, prompt, 32),
            adapter: None,
        })
        .await
        .unwrap()
    {
        TokenAdmissionOutcome::Admitted(frontier) => frontier,
        other => panic!("unexpected admission: {other:?}"),
    }
}

fn poll_pending<F: Future>(future: std::pin::Pin<&mut F>) {
    let waker = futures::task::noop_waker_ref();
    assert!(matches!(
        future.poll(&mut Context::from_waker(waker)),
        Poll::Pending
    ));
}

async fn terminal(close: ExecutionCloseOutcome) -> ExecutionTerminalOutcome {
    let result = match close.disposition {
        ExecutionCloseDisposition::Ended(ended) => ExecutionTerminalOutcome::ended(ended),
        ExecutionCloseDisposition::Held(held) => {
            tokio::time::timeout(Duration::from_secs(5), held.wait())
                .await
                .expect("retained worker must reach terminal")
                .unwrap()
        }
    };
    match close.error {
        Some(error) => result.with_error(error),
        None => result,
    }
}

fn completed(outcome: ExecutionBatchOutcome) -> Vec<ExecutionCompletion> {
    match outcome {
        ExecutionBatchOutcome::Completed(batch) => batch.into_completions(),
        other => panic!("expected complete actual model work: {other:?}"),
    }
}

fn full_logits(completion: &ExecutionCompletion) -> &[f32] {
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
        completion.product()
    else {
        panic!("full vocabulary output is required");
    };
    logits
}

fn compare_logits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), 128);
    assert_eq!(actual.len(), expected.len());
    let magnitude = expected
        .iter()
        .fold(0.0_f32, |max, value| max.max(value.abs()));
    let mut maximum_error = 0.0_f32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite() && expected.is_finite());
        maximum_error = maximum_error.max((actual - expected).abs());
    }
    // FP16 batched and serial reductions may round differently. Compare the
    // complete output, with the same 2% relative envelope used by the actual
    // shared-pool numerical fixture, not just the selected token.
    assert!(
        maximum_error <= 0.02 * magnitude.max(1e-6),
        "complete-logits error={maximum_error}, magnitude={magnitude}"
    );
}

#[tokio::test]
async fn managed_worker_cpu_repeated_close_preserves_first_error_and_exact_end() {
    let (executor, model, control) = fixture::<CpuBackend>(false);
    let frontier = admit(&executor, &RequestId::new(), &[TokenId(1), TokenId(2)]).await;
    let (started, release) = control.block_next();
    control.close_once.store(true, Ordering::Release);
    let first = executor.cancel_tokens(frontier.sequence()).unwrap();
    started.await.unwrap();
    let second = executor.cancel_tokens(frontier.sequence()).unwrap();
    release.send(()).unwrap();
    let first = terminal(first).await;
    let second = terminal(second).await;
    assert_eq!(first.ended.sequence(), frontier.sequence());
    assert_eq!(second.ended, first.ended);
    for result in [first, second] {
        assert!(result
            .error
            .unwrap()
            .to_string()
            .contains("first close observation failed"));
    }
    assert!(model.lock().kv_caches.is_empty());
    assert!(executor.token_sequences.lock().live.is_empty());
    let again = executor.cancel_tokens(frontier.sequence()).unwrap();
    assert!(matches!(
        again.disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
}

#[tokio::test]
async fn managed_worker_cpu_waiting_and_queued_cancellation_keep_original_owners() {
    let (executor, model, control) = fixture::<CpuBackend>(false);
    let prompt = [TokenId(1), TokenId(2)];
    let first = admit(&executor, &RequestId::new(), &prompt).await;
    let queued = admit(&executor, &RequestId::new(), &prompt).await;
    let policy = LogitsReturnPolicy::FullLogits;
    let first_input = [ExecutionWork::new(
        &first,
        TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
        &policy,
    )
    .unwrap()];
    let queued_input = [ExecutionWork::new(
        &queued,
        TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
        &policy,
    )
    .unwrap()];
    let (started, release) = control.block_next();
    let mut first_work = Box::pin(executor.execute_tokens(&first_input));
    poll_pending(first_work.as_mut());
    started.await.unwrap();
    let mut queued_work = Box::pin(executor.execute_tokens(&queued_input));
    poll_pending(queued_work.as_mut());
    let waiting_id = RequestId::new();
    let mut waiting = Box::pin(executor.admit_tokens(ExecutionAdmission {
        context: ExecutorPrefillAdmission::for_startup(&waiting_id, &prompt, 32),
        adapter: None,
    }));
    poll_pending(waiting.as_mut());
    let waiting_sequence = executor
        .token_sequences
        .lock()
        .live
        .iter()
        .find(|(_, entry)| entry.request_id == waiting_id)
        .unwrap()
        .0
        .clone();
    let queued_close = executor
        .token_device_worker()
        .unwrap()
        .observe_terminal(&queued)
        .unwrap();
    assert!(waiting_sequence.observed_end().is_none());
    assert!(queued.sequence().observed_end().is_none());
    drop(waiting);
    drop(queued_work);
    // Before enqueue, the caller still owns its claim. A queued claim instead
    // belongs to the actual command until the worker sees its closed receiver.
    {
        let sequences = executor.token_sequences.lock();
        assert!(sequences.live[&waiting_sequence].closing);
    }
    let waiting_close = executor.cancel_tokens(&waiting_sequence).unwrap();
    release.send(()).unwrap();
    let rows = completed(first_work.await.unwrap());
    assert_eq!(rows.len(), 1);
    assert!(terminal(waiting_close).await.error.is_none());
    assert!(terminal(queued_close).await.error.is_none());
    assert_eq!(control.batches.load(Ordering::Relaxed), 1);
    assert_eq!(model.lock().kv_caches.len(), 1);
    assert!(terminal(executor.cancel_tokens(first.sequence()).unwrap())
        .await
        .error
        .is_none());
    assert!(model.lock().kv_caches.is_empty());
}

#[tokio::test]
async fn managed_worker_cpu_sent_unpolled_result_and_executor_drop_still_close() {
    let (executor, model, control) = fixture::<CpuBackend>(false);
    let prompt = [TokenId(1), TokenId(2)];
    let frontier = admit(&executor, &RequestId::new(), &prompt).await;
    let policy = LogitsReturnPolicy::FullLogits;
    let inputs = [ExecutionWork::new(
        &frontier,
        TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
        &policy,
    )
    .unwrap()];
    let (started, release) = control.block_next();
    let mut work = Box::pin(executor.execute_tokens(&inputs));
    poll_pending(work.as_mut());
    started.await.unwrap();
    release.send(()).unwrap();
    // FIFO admission is a real later command: when it returns, the previous
    // result sender has completed, while `work` has not been polled again.
    let sentinel = admit(&executor, &RequestId::new(), &prompt).await;
    assert_eq!(control.batches.load(Ordering::Relaxed), 1);
    let original_close = executor
        .token_device_worker()
        .unwrap()
        .observe_terminal(&frontier)
        .unwrap();
    drop(work);
    {
        let sequences = executor.token_sequences.lock();
        assert!(
            sequences
                .live
                .get(frontier.sequence())
                .is_some_and(|entry| entry.closing)
                || frontier.sequence().observed_end().is_some()
        );
    }
    let sentinel_close = executor.cancel_tokens(sentinel.sequence()).unwrap();
    drop(executor);
    assert!(terminal(original_close).await.error.is_none());
    assert!(terminal(sentinel_close).await.error.is_none());
    assert!(model.lock().kv_caches.is_empty());
    assert_eq!(
        frontier.sequence().observed_end().unwrap().sequence(),
        frontier.sequence()
    );
}

async fn stale_admission_delivery(terminal_before_accept: bool) {
    let (executor, model, control) = fixture::<CpuBackend>(false);
    let prompt = [TokenId(1), TokenId(2)];
    let policy = LogitsReturnPolicy::FullLogits;
    let blocker = admit(&executor, &RequestId::new(), &prompt).await;
    let blocker_input = [ExecutionWork::new(
        &blocker,
        TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
        &policy,
    )
    .unwrap()];
    let (started, release) = control.block_next();
    let mut blocked_work = Box::pin(executor.execute_tokens(&blocker_input));
    poll_pending(blocked_work.as_mut());
    started.await.unwrap();
    let request = RequestId::new();
    let mut pending = Box::pin(executor.admit_tokens(ExecutionAdmission {
        context: ExecutorPrefillAdmission::for_startup(&request, &prompt, 32),
        adapter: None,
    }));
    poll_pending(pending.as_mut());
    release.send(()).unwrap();
    assert_eq!(completed(blocked_work.await.unwrap()).len(), 1);
    // A later FIFO command proves the original admission response was sent.
    // Its receiver has deliberately not been polled again.
    let sentinel = admit(&executor, &RequestId::new(), &prompt).await;
    let (original, original_cache) = {
        let sequences = executor.token_sequences.lock();
        let entry = sequences
            .live
            .values()
            .find(|entry| entry.request_id == request)
            .unwrap();
        (entry.frontier.clone(), entry.cache_id.clone())
    };
    assert!(model.lock().kv_caches.contains_key(&original_cache));
    assert!(original.sequence().observed_end().is_none());

    control.close_once.store(true, Ordering::Release);
    let close_gate = (!terminal_before_accept).then(|| control.block_next());
    let close = executor.cancel_tokens(original.sequence()).unwrap();
    if let Some((started, release)) = close_gate {
        started.await.unwrap();
        // Acceptance must reject the closing incarnation and keep waiting
        // for its real terminal fact, not return the queued Admitted value.
        poll_pending(pending.as_mut());
        assert!(original.sequence().observed_end().is_none());
        assert!(model.lock().kv_caches.contains_key(&original_cache));
        release.send(()).unwrap();
    }
    let closed = terminal(close).await;
    assert_eq!(closed.ended.sequence(), original.sequence());
    assert!(closed
        .error
        .unwrap()
        .to_string()
        .contains("first close observation failed"));
    assert!(!executor
        .token_sequences
        .lock()
        .live
        .contains_key(original.sequence()));
    assert!(!model.lock().kv_caches.contains_key(&original_cache));

    // When terminal preceded acceptance, a replacement can already own the
    // same public request ID. The old delivery may close only its old identity.
    let replacement = if terminal_before_accept {
        Some(admit(&executor, &request, &prompt).await)
    } else {
        None
    };
    let error = pending
        .await
        .expect_err("an ended admission must never be published");
    assert!(error.to_string().contains("first close observation failed"));
    let replacement = match replacement {
        Some(frontier) => frontier,
        None => admit(&executor, &request, &prompt).await,
    };
    assert_ne!(replacement.sequence(), original.sequence());
    assert!(replacement.sequence().observed_end().is_none());
    {
        let sequences = executor.token_sequences.lock();
        let entry = &sequences.live[replacement.sequence()];
        assert!(!entry.closing);
        assert_eq!(entry.frontier, replacement);
    }
    let replacement_input = [ExecutionWork::new(
        &replacement,
        TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
        &policy,
    )
    .unwrap()];
    let rows = completed(executor.execute_tokens(&replacement_input).await.unwrap());
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].next_frontier().sequence(), replacement.sequence());
    assert_eq!(rows[0].next_frontier().consumed_tokens(), 2);
    let mut reference = real_model::<CpuBackend>(false);
    compare_logits(
        full_logits(&rows[0]),
        &reference.prefill_internal("reference", &[1, 2]),
    );
    reference.release("reference");
    assert_eq!(control.batches.load(Ordering::Relaxed), 2);
    for frontier in [blocker, sentinel, replacement] {
        let closed = terminal(executor.cancel_tokens(frontier.sequence()).unwrap()).await;
        assert_eq!(closed.ended.sequence(), frontier.sequence());
        assert!(closed.error.is_none());
    }
    assert!(executor.token_sequences.lock().live.is_empty());
    assert!(model.lock().kv_caches.is_empty());
}

#[tokio::test]
async fn managed_worker_cpu_admission_accept_waits_for_closing_owner() {
    stale_admission_delivery(false).await;
}

#[tokio::test]
async fn managed_worker_cpu_admission_accept_rejects_ended_original_not_replacement() {
    stale_admission_delivery(true).await;
}

#[test]
fn managed_worker_cpu_admission_registration_cannot_follow_shutdown_scan() {
    let (executor, model, _) = fixture::<CpuBackend>(false);
    // This state uses the actual executor's model and original registry. No
    // preparation or independent resource owner is invented for the race.
    let state = Arc::new(TokenWorkerState::new(&executor).unwrap());
    let (entered, reached) = mpsc::channel();
    let (resume, resumed) = mpsc::channel();
    *state.admission_before_registry_lock.lock() = Some((entered, resumed));
    let registering = Arc::clone(&state);
    let registration = std::thread::spawn(move || {
        registering.register_device_admission(
            ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_startup(
                    &RequestId::new(),
                    &[TokenId(1), TokenId(2)],
                    32,
                ),
                adapter: None,
            },
            "shutdown-race".to_owned(),
        )
    });
    reached.recv_timeout(Duration::from_secs(5)).unwrap();
    // Registration has passed the original pre-lock check. Complete the real
    // shutdown scan before allowing it to acquire the same registry lock.
    state.request_shutdown();
    assert!(state.shutdown_complete());
    resume.send(()).unwrap();
    let error = registration
        .join()
        .unwrap()
        .expect_err("a closed worker must not acquire a late unpublished incarnation");
    assert!(error.to_string().contains("model worker is closing"));
    assert!(executor.token_sequences.lock().live.is_empty());
    assert!(model.lock().kv_caches.is_empty());
    assert!(state.shutdown_complete());
}

#[tokio::test]
async fn managed_worker_cpu_missing_middle_row_preserves_raw_logits_without_misattribution() {
    let (executor, model, control) = fixture::<CpuBackend>(false);
    let mut reference = real_model::<CpuBackend>(false);
    let prompts = [
        [TokenId(1), TokenId(2)],
        [TokenId(3), TokenId(4)],
        [TokenId(5), TokenId(6)],
    ];
    let mut frontiers = Vec::new();
    for prompt in &prompts {
        frontiers.push(admit(&executor, &RequestId::new(), prompt).await);
    }
    let policy = LogitsReturnPolicy::FullLogits;
    let inputs: Vec<_> = frontiers
        .iter()
        .map(|frontier| {
            ExecutionWork::new(
                frontier,
                TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
                &policy,
            )
            .unwrap()
        })
        .collect();
    control.omit_middle_result.store(true, Ordering::Release);
    let ExecutionBatchOutcome::ContractViolation(violation) =
        executor.execute_tokens(&inputs).await.unwrap()
    else {
        panic!("missing identity-free middle row must be rejected as a whole attribution");
    };
    assert!(
        violation.completed.is_empty(),
        "C's logits must not be committed as B"
    );
    assert_eq!(violation.unassigned_products.len(), 2);
    assert_eq!(violation.remaining.len(), 3);
    assert_eq!(control.batches.load(Ordering::Relaxed), 1);
    for (product, tokens) in violation.unassigned_products.iter().zip([[1, 2], [5, 6]]) {
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(actual)) = product else {
            panic!("actual full logits must survive the malformed result");
        };
        let expected = reference.prefill_internal("oracle", &tokens);
        compare_logits(actual, &expected);
        reference.release("oracle");
    }
    for (participant, frontier) in violation.remaining.into_iter().zip(&frontiers) {
        assert_eq!(participant.sequence(), frontier.sequence());
        let (_, _, progress, continuation, _) = participant.into_parts();
        assert!(matches!(progress, ExecutionProgress::Unknown { .. }));
        let ContinuationDisposition::Held(held) = continuation else {
            panic!("original owner must supply checked terminal observation");
        };
        let observed = tokio::time::timeout(Duration::from_secs(5), held.wait())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(observed.ended.sequence(), frontier.sequence());
        assert!(observed.error.is_some());
    }
    assert!(model.lock().kv_caches.is_empty());
    assert!(executor.token_sequences.lock().live.is_empty());
}

async fn numerical_and_peer_cancellation<B: MoeLlmBackend>(paged: bool, require_batched: bool) {
    // Construct independent reference before the candidate; model construction
    // may reset backend graph state. Graphs are disabled in this fixture.
    let mut reference = real_model::<B>(paged);
    let (executor, model, control) = fixture::<B>(paged);
    if require_batched {
        assert!(model.lock().supports_batched_decode);
        assert!(model.lock().supports_varlen_qkv);
    }
    let prompt = [TokenId(1), TokenId(2), TokenId(3)];
    let a = admit(&executor, &RequestId::new(), &prompt).await;
    let b = admit(&executor, &RequestId::new(), &prompt).await;
    let policy = LogitsReturnPolicy::FullLogits;
    // Partial prefill advances exact frontiers without a sampling product.
    let partial = [
        ExecutionWork::new(
            &a,
            TokenWork::Prefill(PrefillChunk::new(0, 2, 3).unwrap()),
            &policy,
        )
        .unwrap(),
        ExecutionWork::new(
            &b,
            TokenWork::Prefill(PrefillChunk::new(0, 2, 3).unwrap()),
            &policy,
        )
        .unwrap(),
    ];
    let rows = completed(executor.execute_tokens(&partial).await.unwrap());
    assert!(rows
        .iter()
        .all(|row| matches!(row.product(), ExecutionProduct::Intermediate)));
    if require_batched {
        let actual = model.lock();
        assert!(actual.scratch.unified_capacity >= 4);
        assert!(actual.scratch.unified_norm_out.is_some());
    }
    let a = rows[0].next_frontier().clone();
    let b = rows[1].next_frontier().clone();
    let final_inputs = [
        ExecutionWork::new(
            &a,
            TokenWork::Prefill(PrefillChunk::new(2, 1, 3).unwrap()),
            &policy,
        )
        .unwrap(),
        ExecutionWork::new(
            &b,
            TokenWork::Prefill(PrefillChunk::new(2, 1, 3).unwrap()),
            &policy,
        )
        .unwrap(),
    ];
    let rows = completed(executor.execute_tokens(&final_inputs).await.unwrap());
    let expected = reference.prefill_internal("reference", &[1, 2, 3]);
    for row in &rows {
        compare_logits(full_logits(row), &expected);
    }
    let a = rows[0].next_frontier().clone();
    let b = rows[1].next_frontier().clone();
    let decode = [
        ExecutionWork::new(
            &a,
            TokenWork::Decode {
                token: TokenId(4),
                position: 3,
            },
            &policy,
        )
        .unwrap(),
        ExecutionWork::new(
            &b,
            TokenWork::Decode {
                token: TokenId(4),
                position: 3,
            },
            &policy,
        )
        .unwrap(),
    ];
    let (started, release) = control.block_next();
    let mut work = Box::pin(executor.execute_tokens(&decode));
    poll_pending(work.as_mut());
    started.await.unwrap();
    let a_close = executor.cancel_tokens(a.sequence()).unwrap();
    assert!(a.sequence().observed_end().is_none());
    release.send(()).unwrap();
    let ExecutionBatchOutcome::Interrupted(outcome) = work.await.unwrap() else {
        panic!("cancelled participant must keep its actual progress separately");
    };
    let (completed, remaining, _) = outcome.into_parts();
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].input_index(), 1);
    compare_logits(
        full_logits(&completed[0]),
        &reference.decode_internal("reference", 4, 3),
    );
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].sequence(), a.sequence());
    assert!(matches!(
        remaining[0].progress(),
        ExecutionProgress::Terminal {
            disposition: TerminalModelDisposition::Successful,
            ..
        }
    ));
    let (_, _, _, continuation, _) = remaining.into_iter().next().unwrap().into_parts();
    if let ContinuationDisposition::Held(held) = continuation {
        let terminal = held.wait().await.unwrap();
        assert_eq!(terminal.ended.sequence(), a.sequence());
        assert!(
            terminal.error.is_none(),
            "successful work followed by normal cancellation is not a device error"
        );
    }
    let a_terminal = terminal(a_close).await;
    assert_eq!(a_terminal.ended.sequence(), a.sequence());
    assert!(a_terminal.error.is_none());
    let b = completed[0].next_frontier().clone();
    assert_eq!(b.consumed_tokens(), 4);
    assert!(b.sequence().observed_end().is_none());
    let next = [ExecutionWork::new(
        &b,
        TokenWork::Decode {
            token: TokenId(5),
            position: 4,
        },
        &policy,
    )
    .unwrap()];
    let rows = self::completed(executor.execute_tokens(&next).await.unwrap());
    compare_logits(
        full_logits(&rows[0]),
        &reference.decode_internal("reference", 5, 4),
    );
    {
        let cache_id = executor.token_sequences.lock().live[b.sequence()]
            .cache_id
            .clone();
        let actual = model.lock();
        assert!(actual.kv_caches[&cache_id]
            .iter()
            .all(|cache| cache.len == 5));
        if require_batched {
            let stats = actual.cache_metrics_snapshot().unwrap();
            assert!(stats["decode_batch"]["max_items"].as_u64().unwrap() >= 2);
        }
    }
    assert!(terminal(executor.cancel_tokens(b.sequence()).unwrap())
        .await
        .error
        .is_none());
    assert!(model.lock().kv_caches.is_empty());
    assert!(executor.token_sequences.lock().live.is_empty());
    reference.release("reference");
}

async fn singleton_sampling_policies<B: MoeLlmBackend>(paged: bool) {
    let mut reference = real_model::<B>(paged);
    let (executor, model, _) = fixture::<B>(paged);
    assert!(model.lock().supports_batched_decode);
    let prompt = [TokenId(1), TokenId(2), TokenId(3)];
    let full_policy = LogitsReturnPolicy::FullLogits;
    let greedy_policy = LogitsReturnPolicy::GreedyArgmax {
        token_mask: None,
        repetition_penalty: None,
    };
    let mut frontiers = Vec::new();
    let expected_prefill = reference.prefill_internal("reference", &[1, 2, 3]);
    for _ in 0..2 {
        let frontier = admit(&executor, &RequestId::new(), &prompt).await;
        let input = [ExecutionWork::new(
            &frontier,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
            &full_policy,
        )
        .unwrap()];
        let rows = completed(executor.execute_tokens(&input).await.unwrap());
        assert_eq!(rows.len(), 1);
        compare_logits(full_logits(&rows[0]), &expected_prefill);
        frontiers.push(rows[0].next_frontier().clone());
    }
    for (position, token) in [(3, 4), (4, 5)] {
        let expected = reference.decode_internal("reference", token, position);
        let before = model.lock().cache_metrics_snapshot().unwrap();
        let calls = before["decode_batch"]["calls"].as_u64().unwrap();
        let full_calls = before["decode_batch"]["force_full_logits_calls"]
            .as_u64()
            .unwrap();
        let full_input = [ExecutionWork::new(
            &frontiers[0],
            TokenWork::Decode {
                token: TokenId(token),
                position: position as usize,
            },
            &full_policy,
        )
        .unwrap()];
        let full = completed(executor.execute_tokens(&full_input).await.unwrap());
        assert_eq!(full.len(), 1);
        compare_logits(full_logits(&full[0]), &expected);
        frontiers[0] = full[0].next_frontier().clone();
        let after_full = model.lock().cache_metrics_snapshot().unwrap();
        assert_eq!(
            after_full["decode_batch"]["calls"].as_u64().unwrap(),
            calls,
            "singleton FullLogits must retain the scalar decode route"
        );

        let greedy_input = [ExecutionWork::new(
            &frontiers[1],
            TokenWork::Decode {
                token: TokenId(token),
                position: position as usize,
            },
            &greedy_policy,
        )
        .unwrap()];
        let greedy = completed(executor.execute_tokens(&greedy_input).await.unwrap());
        assert_eq!(greedy.len(), 1);
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::GreedyToken(selected)) =
            greedy[0].product()
        else {
            panic!("singleton GreedyArgmax must preserve the typed selected-token route");
        };
        // Independent full-vocabulary scalar decode is the numerical oracle;
        // neither the returned token nor a model route flag defines it.
        let expected_token =
            expected.iter().enumerate().fold(
                0,
                |best, (index, value)| {
                    if *value > expected[best] {
                        index
                    } else {
                        best
                    }
                },
            );
        assert_eq!(selected.get() as usize, expected_token);
        frontiers[1] = greedy[0].next_frontier().clone();
        let after_greedy = model.lock().cache_metrics_snapshot().unwrap();
        assert_eq!(
            after_greedy["decode_batch"]["calls"].as_u64().unwrap(),
            calls + 1
        );
        assert_eq!(
            after_greedy["decode_batch"]["last_items"].as_u64().unwrap(),
            1
        );
        assert_eq!(
            after_greedy["decode_batch"]["force_full_logits_calls"]
                .as_u64()
                .unwrap(),
            full_calls
        );
        let sequences = executor.token_sequences.lock();
        let actual = model.lock();
        for frontier in &frontiers {
            assert_eq!(frontier.consumed_tokens(), position as usize + 1);
            let cache_id = &sequences.live[frontier.sequence()].cache_id;
            assert!(actual.kv_caches[cache_id]
                .iter()
                .all(|cache| cache.len == position as usize + 1));
        }
    }
    for frontier in frontiers {
        let closed = terminal(executor.cancel_tokens(frontier.sequence()).unwrap()).await;
        assert_eq!(closed.ended.sequence(), frontier.sequence());
        assert!(closed.error.is_none());
    }
    assert!(executor.token_sequences.lock().live.is_empty());
    assert!(model.lock().kv_caches.is_empty());
    reference.release("reference");
}

#[tokio::test]
async fn managed_worker_cpu_singleton_sampling_policy_routes() {
    singleton_sampling_policies::<CpuBackend>(false).await;
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[tokio::test]
#[ignore = "requires an actual Metal device"]
async fn managed_worker_real_metal_singleton_sampling_policy_routes() {
    singleton_sampling_policies::<ferrum_kernels::backend::metal::MetalBackend>(true).await;
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires an actual CUDA device"]
async fn managed_worker_real_cuda_singleton_sampling_policy_routes() {
    singleton_sampling_policies::<ferrum_kernels::backend::cuda::CudaBackend>(true).await;
}

#[tokio::test]
async fn managed_worker_cpu_real_logits_and_single_peer_cancellation() {
    numerical_and_peer_cancellation::<CpuBackend>(false, false).await;
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[tokio::test]
#[ignore = "requires an actual Metal device"]
async fn managed_worker_real_metal_logits_and_single_peer_cancellation() {
    numerical_and_peer_cancellation::<ferrum_kernels::backend::metal::MetalBackend>(true, false)
        .await;
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires an actual CUDA device"]
async fn managed_worker_real_cuda_logits_and_single_peer_cancellation() {
    numerical_and_peer_cancellation::<ferrum_kernels::backend::cuda::CudaBackend>(true, true).await;
}
