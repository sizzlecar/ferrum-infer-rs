//! The production pipeline itself is placed in LlmExecutor. Decorators only
//! observe/fault the actual mathematical leaf; they grant no lifecycle mode.
use super::*;
use crate::models::llama_family::LlamaFamilyLayerStageConfig;
use crate::models::llama_family_pipeline::{
    LlamaFamilyPipelineModel, LlamaPipelineMode, LlamaPipelinePlacement, LlamaPipelineStageBatchOps,
};
use ferrum_execution::executor_lifecycle::{ExecutorOwner, ExecutorShutdownPhase};
use ferrum_interfaces::kv_dtype::KvFp16;
use ferrum_kernels::backend::Backend;
use ferrum_kernels::{Linear, LinearMetadata, LoadedLinear};
use ferrum_quantization::{QuantConfig, WeightLoader};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc,
};
use tokio::sync::oneshot;

const PIPELINE_ERROR: &str = "actual pipeline head encoded then failed";
const WAIT: Duration = Duration::from_secs(20);
#[derive(Default)]
struct Control {
    forwards: AtomicUsize,
    drops: AtomicUsize,
    fail_once: AtomicBool,
    gate: Mutex<Option<(oneshot::Sender<()>, mpsc::Receiver<()>)>>,
}
#[cfg(any(
    feature = "cuda",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
struct Release(Option<mpsc::Sender<()>>);
#[cfg(any(
    feature = "cuda",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
impl Release {
    fn now(&mut self) {
        if let Some(tx) = self.0.take() {
            let _ = tx.send(());
        }
    }
}
#[cfg(any(
    feature = "cuda",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
impl Drop for Release {
    fn drop(&mut self) {
        self.now();
    }
}
#[cfg(any(
    feature = "cuda",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
impl Control {
    fn gate(&self) -> (oneshot::Receiver<()>, Release) {
        let (entered, wait) = oneshot::channel();
        let (release, receive) = mpsc::channel();
        assert!(self.gate.lock().replace((entered, receive)).is_none());
        (wait, Release(Some(release)))
    }
}
struct ObservedLinear<B: Backend> {
    inner: Box<dyn Linear<B>>,
    control: Arc<Control>,
}
impl<B: Backend> Linear<B> for ObservedLinear<B> {
    fn in_features(&self) -> usize {
        self.inner.in_features()
    }
    fn out_features(&self) -> usize {
        self.inner.out_features()
    }
    fn metadata(&self) -> LinearMetadata {
        self.inner.metadata()
    }
    fn forward(
        &self,
        ctx: &mut B::Context,
        input: &B::Buffer,
        output: &mut B::Buffer,
        rows: usize,
    ) {
        self.inner.forward(ctx, input, output, rows);
        self.control.forwards.fetch_add(1, Ordering::SeqCst);
        if let Some((entered, resume)) = self.control.gate.lock().take() {
            entered.send(()).expect("observe actual pipeline encode");
            resume
                .recv_timeout(WAIT)
                .expect("release actual pipeline encode");
        }
        if self.control.fail_once.swap(false, Ordering::SeqCst) {
            std::panic::panic_any(FerrumError::backend(PIPELINE_ERROR));
        }
    }
}
impl<B: Backend> Drop for ObservedLinear<B> {
    fn drop(&mut self) {
        self.control.drops.fetch_add(1, Ordering::SeqCst);
    }
}
struct Loader {
    inner: SyntheticLlamaLoader,
    owners: [Arc<Control>; 3],
}
impl<B: Backend> WeightLoader<B> for Loader {
    fn load_tensor(&self, ctx: &mut B::Context, name: &str) -> Result<B::Buffer> {
        <SyntheticLlamaLoader as WeightLoader<B>>::load_tensor(&self.inner, ctx, name)
    }
    fn load_linear(&self, ctx: &mut B::Context, name: &str) -> Result<LoadedLinear<B>> {
        let inner = <SyntheticLlamaLoader as WeightLoader<B>>::load_linear(&self.inner, ctx, name)?;
        let slot = match name {
            "model.layers.0.mlp.down_proj" => Some(0),
            "model.layers.1.mlp.down_proj" => Some(1),
            "lm_head" => Some(2),
            _ => None,
        };
        Ok(match slot {
            Some(index) => LoadedLinear::new(Box::new(ObservedLinear {
                inner: inner.into_eager_box(),
                control: Arc::clone(&self.owners[index]),
            })),
            None => inner,
        })
    }
    fn has_tensor(&self, name: &str) -> bool {
        <SyntheticLlamaLoader as WeightLoader<B>>::has_tensor(&self.inner, name)
    }
    fn quant_config(&self) -> Option<&QuantConfig> {
        <SyntheticLlamaLoader as WeightLoader<B>>::quant_config(&self.inner)
    }
}
fn pipeline_config() -> LlamaFamilyConfig {
    LlamaFamilyConfig {
        num_layers: 2,
        max_seq_len: 32,
        ..config()
    }
}
fn build<B, K>(
    mode: LlamaPipelineMode,
    devices: Option<[usize; 2]>,
) -> (LlmExecutor, [Arc<Control>; 3])
where
    B: MoeLlmBackend,
    K: KvLayer<B>,
    LlamaFamilyModel<B, K>: DecoderOnlyLLM + LlamaPipelineStageBatchOps<B> + Send,
{
    build_with_greedy::<B, K>(mode, devices, false)
}
fn build_with_greedy<B, K>(
    mode: LlamaPipelineMode,
    devices: Option<[usize; 2]>,
    greedy: bool,
) -> (LlmExecutor, [Arc<Control>; 3])
where
    B: MoeLlmBackend,
    K: KvLayer<B>,
    LlamaFamilyModel<B, K>: DecoderOnlyLLM + LlamaPipelineStageBatchOps<B> + Send,
{
    let cfg = pipeline_config();
    let owners = std::array::from_fn(|_| Arc::new(Control::default()));
    let loader = Loader {
        inner: SyntheticLlamaLoader::new(cfg.clone()),
        owners: owners.clone(),
    };
    let mut stages = Vec::new();
    for index in 0..2 {
        let mut execution = LlamaExecutionConfig {
            kv_capacity: Some(32),
            paged_max_seqs: 17,
            options: LlamaExecutionOptions {
                metal_paged_kv: Some(B::supports_paged_kv()),
                kv_max_blocks: Some(64),
                greedy_argmax: greedy,
                ..Default::default()
            },
            ..Default::default()
        };
        if let Some(devices) = devices {
            execution.backend_context.device_ordinal = devices[index];
        }
        stages.push(
            LlamaFamilyModel::<B, K>::new_layer_stage_with_execution_config(
                cfg.clone(),
                &loader,
                LlamaFamilyLayerStageConfig::pipeline_stage(
                    index..index + 1,
                    index == 0,
                    index == 1,
                ),
                execution,
            )
            .unwrap(),
        );
    }
    let placement = devices
        .map_or_else(
            || LlamaPipelinePlacement::unplaced(2),
            |devices| {
                LlamaPipelinePlacement::from_backend_device_ordinals(
                    devices.into_iter().map(Some).collect(),
                )
            },
        )
        .with_pipeline_mode(mode);
    let model = LlamaFamilyPipelineModel::new_with_placement(stages, placement).unwrap();
    assert_eq!(
        model.token_execution_mode(),
        if B::host_execution_is_synchronous() {
            ModelTokenExecutionMode::InlineSynchronous
        } else {
            ModelTokenExecutionMode::ManagedWorker
        }
    );
    let mut info = tiny_model_info(&TinyLlamaConfig {
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_heads: cfg.num_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        num_layers: cfg.num_layers,
        vocab_size: cfg.vocab_size,
        max_seq_len: cfg.max_seq_len,
    });
    info.device = if let Some(devices) = devices {
        Device::CUDA(devices[0])
    } else if B::host_execution_is_synchronous() {
        Device::CPU
    } else {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            Device::Metal
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            panic!("device pipeline fixture requires explicit CUDA ordinals")
        }
    };
    (LlmExecutor::new(Box::new(model), info), owners)
}
fn compare_pipeline_logits<B: Backend>(completion: &ExecutionCompletion, expected: &[f32]) {
    compare_logits(completion, expected);
    if B::host_execution_is_synchronous() {
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(actual)) =
            completion.product()
        else {
            unreachable!()
        };
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "CPU pipeline retains its strict full-vector oracle"
            );
        }
    }
}
fn reference() -> LlamaFamilyModel<CpuBackend> {
    let cfg = pipeline_config();
    LlamaFamilyModel::new_with_execution_config(
        cfg.clone(),
        &SyntheticLlamaLoader::new(cfg),
        LlamaExecutionConfig {
            kv_capacity: Some(32),
            options: LlamaExecutionOptions {
                metal_paged_kv: Some(false),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap()
}
async fn close(executor: LlmExecutor, owners: &[Arc<Control>; 3], expected_error: bool) {
    let slot = Arc::clone(&executor.model);
    let owner = ExecutorOwner::new(Box::new(executor));
    let mut observer = owner.begin_shutdown();
    let report = tokio::time::timeout(WAIT, async {
        loop {
            let report = observer.snapshot();
            if report.phase() == ExecutorShutdownPhase::Retired {
                break report;
            }
            observer.changed().await;
        }
    })
    .await
    .expect("original pipeline worker joins after actual stage retirement");
    assert_eq!(report.result().is_err(), expected_error);
    assert!(slot.lock().is_none());
    assert!(owners
        .iter()
        .all(|control| control.drops.load(Ordering::SeqCst) == 1));
    assert!(owner.scope().is_err());
}
async fn numerical<B, K>(mode: LlamaPipelineMode, count: usize, devices: Option<[usize; 2]>)
where
    B: MoeLlmBackend,
    K: KvLayer<B>,
    LlamaFamilyModel<B, K>: DecoderOnlyLLM + LlamaPipelineStageBatchOps<B> + Send,
{
    let (executor, owners) = build::<B, K>(mode, devices);
    let mut oracle = reference();
    let mut frontiers = Vec::new();
    for index in 0..count {
        let frontier = admit(&executor, &RequestId::new()).await;
        let partial = execute(
            &executor,
            &frontier,
            TokenWork::Prefill(PrefillChunk::new(0, 2, 3).unwrap()),
        )
        .await;
        assert!(matches!(partial.product(), ExecutionProduct::Intermediate));
        oracle.prefill_internal(&format!("r{index}"), &[1, 2]);
        let completed = execute(
            &executor,
            partial.next_frontier(),
            TokenWork::Prefill(PrefillChunk::new(2, 1, 3).unwrap()),
        )
        .await;
        compare_pipeline_logits::<B>(
            &completed,
            &oracle.prefill_internal(&format!("r{index}"), &[3]),
        );
        frontiers.push(completed.next_frontier().clone());
    }
    for position in 3..6 {
        let policies: Vec<_> = (0..count)
            .map(|index| {
                if position == 5 && index != 0 {
                    LogitsReturnPolicy::GreedyArgmax {
                        token_mask: Some(
                            ferrum_execution::model_executor::TokenSelectionMask::new(
                                vec![1; VOCABULARY],
                            ),
                        ),
                        repetition_penalty: None,
                    }
                } else {
                    LogitsReturnPolicy::FullLogits
                }
            })
            .collect();
        let inputs: Vec<_> = frontiers
            .iter()
            .zip(&policies)
            .map(|(f, policy)| {
                ExecutionWork::new(
                    f,
                    TokenWork::Decode {
                        token: TokenId(position as u32 + 1),
                        position,
                    },
                    policy,
                )
                .unwrap()
            })
            .collect();
        let ExecutionBatchOutcome::Completed(batch) =
            executor.execute_tokens(&inputs).await.unwrap()
        else {
            panic!("actual pipeline batch must complete")
        };
        let rows = batch.into_completions();
        assert_eq!(rows.len(), count);
        for (index, completion) in rows.iter().enumerate() {
            assert_eq!(
                completion.next_frontier().sequence(),
                frontiers[index].sequence()
            );
            compare_pipeline_logits::<B>(
                completion,
                &oracle.decode_internal(&format!("r{index}"), position as u32 + 1, position as u32),
            );
        }
        frontiers = rows.iter().map(|row| row.next_frontier().clone()).collect();
    }
    if !B::host_execution_is_synchronous() {
        let metrics = executor
            .lock_model()
            .unwrap()
            .cache_metrics_snapshot()
            .unwrap();
        let stats = &metrics["pipeline_decode"];
        assert_eq!(stats["last_batch"], count);
        if mode == LlamaPipelineMode::Overlapped && count >= 16 {
            assert_eq!(stats["in_flight_stage_count_last"], 2);
        }
    }
    for (index, frontier) in frontiers.iter().enumerate() {
        terminal(
            executor
                .finish_tokens(
                    frontier.sequence(),
                    ExecutionEndReason::Completed {
                        input_tokens: 3,
                        output_tokens: 4,
                    },
                )
                .await
                .unwrap(),
        )
        .await;
        oracle.release(&format!("r{index}"));
    }
    oracle.close().unwrap();
    close(executor, &owners, false).await;
}

#[tokio::test]
async fn qualification_pipeline_cpu_original_owner_and_full_logits() {
    numerical::<CpuBackend, KvFp16>(LlamaPipelineMode::Batch, 2, None).await;
}
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[tokio::test]
#[ignore = "requires real Metal pipeline contexts, token execution and original-owner retirement"]
async fn qualification_pipeline_metal_batch_and_overlap_full_logits() {
    use ferrum_kernels::backend::metal::MetalBackend;
    numerical::<MetalBackend, KvFp16>(LlamaPipelineMode::Batch, 2, None).await;
    numerical::<MetalBackend, KvFp16>(LlamaPipelineMode::Overlapped, 16, None).await;
    greedy_head::<MetalBackend>(None).await;
}
#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires real CUDA stage contexts, token execution and original-owner retirement"]
async fn qualification_pipeline_cuda_batch_and_overlap_full_logits() {
    use ferrum_kernels::backend::cuda::CudaBackend;
    numerical::<CudaBackend, KvFp16>(LlamaPipelineMode::Batch, 2, Some([0, 0])).await;
    numerical::<CudaBackend, KvFp16>(LlamaPipelineMode::Overlapped, 16, Some([0, 0])).await;
    greedy_head::<CudaBackend>(Some([0, 0])).await;
}
#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires two real CUDA devices for explicit stage placement"]
async fn qualification_pipeline_cuda_two_devices_original_stage_owners() {
    use ferrum_kernels::backend::cuda::CudaBackend;
    numerical::<CudaBackend, KvFp16>(LlamaPipelineMode::Overlapped, 16, Some([0, 1])).await;
}

#[cfg(any(
    feature = "cuda",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
async fn failure_and_cancellation<B, K>(devices: Option<[usize; 2]>)
where
    B: MoeLlmBackend,
    K: KvLayer<B>,
    LlamaFamilyModel<B, K>: DecoderOnlyLLM + LlamaPipelineStageBatchOps<B> + Send,
{
    let (executor, owners) = build::<B, K>(LlamaPipelineMode::Overlapped, devices);
    let failed = admit(&executor, &RequestId::new()).await;
    let peer = admit(&executor, &RequestId::new()).await;
    owners[2].fail_once.store(true, Ordering::SeqCst);
    let policy = LogitsReturnPolicy::FullLogits;
    let input = [ExecutionWork::new(
        &failed,
        TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        &policy,
    )
    .unwrap()];
    let ExecutionBatchOutcome::Interrupted(interrupted) =
        executor.execute_tokens(&input).await.unwrap()
    else {
        panic!("encoded stage failure must not complete")
    };
    let (completed, mut remaining, causes) = interrupted.into_parts();
    assert!(completed.is_empty());
    assert_eq!(remaining.len(), 1);
    assert!(causes.iter().any(|cause| matches!(cause, ExecutionInterruptionCause::Error(error) if error.to_string().contains(PIPELINE_ERROR))));
    let participant = remaining.remove(0);
    assert_eq!(participant.sequence(), failed.sequence());
    assert!(matches!(
        participant.progress(),
        ExecutionProgress::Terminal {
            disposition: TerminalModelDisposition::Failed,
            ..
        }
    ));
    let (_, _, _, continuation, _) = participant.into_parts();
    let ended = match continuation {
        ContinuationDisposition::Ended(ended) => ended,
        ContinuationDisposition::Held(held) => {
            let result = tokio::time::timeout(WAIT, held.wait())
                .await
                .unwrap()
                .unwrap();
            assert!(result.error.is_some());
            result.ended
        }
        _ => panic!("partly executed pipeline must not return replay authority"),
    };
    assert_eq!(ended.sequence(), failed.sequence());
    assert!(owners
        .iter()
        .all(|control| control.forwards.load(Ordering::SeqCst) == 1));
    let mut oracle = reference();
    let prefill = execute(
        &executor,
        &peer,
        TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
    )
    .await;
    compare_logits(&prefill, &oracle.prefill_internal("peer", &[1, 2, 3]));
    let decoded = execute(
        &executor,
        prefill.next_frontier(),
        TokenWork::Decode {
            token: TokenId(4),
            position: 3,
        },
    )
    .await;
    compare_logits(&decoded, &oracle.decode_internal("peer", 4, 3));
    assert!(
        owners
            .iter()
            .all(|control| control.forwards.load(Ordering::SeqCst) == 3),
        "failed token computation must not replay"
    );
    terminal(
        executor
            .finish_tokens(
                peer.sequence(),
                ExecutionEndReason::Completed {
                    input_tokens: 3,
                    output_tokens: 2,
                },
            )
            .await
            .unwrap(),
    )
    .await;
    oracle.release("peer");
    oracle.close().unwrap();
    // The original request/cohort error was already delivered and retired.
    // Whole-executor close reports its own recovery/retirement failures.
    close(executor, &owners, false).await;

    let (executor, owners) = build::<B, K>(LlamaPipelineMode::Overlapped, devices);
    let cancelled = admit(&executor, &RequestId::new()).await;
    let peer = admit(&executor, &RequestId::new()).await;
    let (entered, mut release) = owners[2].gate();
    let input = [ExecutionWork::new(
        &cancelled,
        TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        &policy,
    )
    .unwrap()];
    let mut pending = Box::pin(executor.execute_tokens(&input));
    assert!(matches!(
        futures::poll!(pending.as_mut()),
        std::task::Poll::Pending
    ));
    tokio::time::timeout(WAIT, entered).await.unwrap().unwrap();
    let closing = executor.cancel_tokens(cancelled.sequence()).unwrap();
    assert!(matches!(
        &closing.disposition,
        ExecutionCloseDisposition::Held(_)
    ));
    drop(pending);
    assert!(cancelled.sequence().observed_end().is_none());
    assert!(owners
        .iter()
        .all(|control| control.drops.load(Ordering::SeqCst) == 0));
    assert!(
        executor.model.try_lock().is_none(),
        "original worker keeps the actual stages"
    );
    release.now();
    assert_eq!(terminal(closing).await.sequence(), cancelled.sequence());
    let mut oracle = reference();
    let prefill = execute(
        &executor,
        &peer,
        TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
    )
    .await;
    compare_logits(&prefill, &oracle.prefill_internal("peer", &[1, 2, 3]));
    assert!(owners
        .iter()
        .all(|control| control.forwards.load(Ordering::SeqCst) == 2));
    terminal(
        executor
            .finish_tokens(
                peer.sequence(),
                ExecutionEndReason::Completed {
                    input_tokens: 3,
                    output_tokens: 1,
                },
            )
            .await
            .unwrap(),
    )
    .await;
    oracle.release("peer");
    oracle.close().unwrap();
    close(executor, &owners, false).await;
}
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[tokio::test]
#[ignore = "requires actual Metal stage encoding, cancellation and original-owner retirement"]
async fn qualification_pipeline_metal_later_stage_failure_and_cancellation() {
    failure_and_cancellation::<ferrum_kernels::backend::metal::MetalBackend, KvFp16>(None).await;
    overlapped_failure::<ferrum_kernels::backend::metal::MetalBackend>(None).await;
}
#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires actual CUDA stage encoding, cancellation and original-owner retirement"]
async fn qualification_pipeline_cuda_later_stage_failure_and_cancellation() {
    failure_and_cancellation::<ferrum_kernels::backend::cuda::CudaBackend, KvFp16>(Some([0, 0]))
        .await;
    overlapped_failure::<ferrum_kernels::backend::cuda::CudaBackend>(Some([0, 0])).await;
}
#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires actual CUDA INT8-KV pipeline and independent full-vocabulary CPU oracle"]
async fn qualification_pipeline_cuda_int8_original_stage_owners() {
    use ferrum_interfaces::kv_dtype::KvInt8;
    use ferrum_kernels::backend::cuda::CudaBackend;
    numerical::<CudaBackend, KvInt8>(LlamaPipelineMode::Batch, 2, Some([0, 0])).await;
    failure_and_cancellation::<CudaBackend, KvInt8>(Some([0, 0])).await;
}

#[cfg(any(
    feature = "cuda",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
async fn greedy_head<B: MoeLlmBackend>(devices: Option<[usize; 2]>)
where
    LlamaFamilyModel<B, KvFp16>: DecoderOnlyLLM + LlamaPipelineStageBatchOps<B> + Send,
{
    let (greedy, greedy_owners) =
        build_with_greedy::<B, KvFp16>(LlamaPipelineMode::Batch, devices, true);
    let (full, full_owners) =
        build_with_greedy::<B, KvFp16>(LlamaPipelineMode::Batch, devices, true);
    let mut greedy_frontiers = Vec::new();
    let mut full_frontiers = Vec::new();
    for _ in 0..2 {
        for (executor, frontiers) in [
            (&greedy, &mut greedy_frontiers),
            (&full, &mut full_frontiers),
        ] {
            let initial = admit(executor, &RequestId::new()).await;
            let prefill = execute(
                executor,
                &initial,
                TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
            )
            .await;
            frontiers.push(prefill.next_frontier().clone());
        }
    }
    let greedy_policy = LogitsReturnPolicy::GreedyArgmax {
        token_mask: None,
        repetition_penalty: None,
    };
    let full_policy = LogitsReturnPolicy::FullLogits;
    let greedy_inputs: Vec<_> = greedy_frontiers
        .iter()
        .map(|f| {
            ExecutionWork::new(
                f,
                TokenWork::Decode {
                    token: TokenId(4),
                    position: 3,
                },
                &greedy_policy,
            )
            .unwrap()
        })
        .collect();
    let full_inputs: Vec<_> = full_frontiers
        .iter()
        .map(|f| {
            ExecutionWork::new(
                f,
                TokenWork::Decode {
                    token: TokenId(4),
                    position: 3,
                },
                &full_policy,
            )
            .unwrap()
        })
        .collect();
    let ExecutionBatchOutcome::Completed(selected) =
        greedy.execute_tokens(&greedy_inputs).await.unwrap()
    else {
        panic!("native greedy pipeline failed")
    };
    let ExecutionBatchOutcome::Completed(reference) =
        full.execute_tokens(&full_inputs).await.unwrap()
    else {
        panic!("same-shape full pipeline failed")
    };
    let selected = selected.into_completions();
    let reference = reference.into_completions();
    assert_eq!(selected.len(), 2);
    assert_eq!(reference.len(), 2);
    for (selected, reference) in selected.iter().zip(&reference) {
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::GreedyToken(token)) =
            selected.product()
        else {
            panic!("the actual head must return the typed greedy product")
        };
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
            reference.product()
        else {
            panic!("reference must read the complete same-shape vocabulary")
        };
        assert_eq!(logits.len(), VOCABULARY);
        assert!(logits.iter().all(|v| v.is_finite()));
        let argmax =
            logits.iter().enumerate().fold(
                0,
                |best, (index, value)| if *value > logits[best] { index } else { best },
            );
        assert_eq!(token.get() as usize, argmax);
    }
    for (executor, rows) in [(&greedy, &selected), (&full, &reference)] {
        for row in rows {
            terminal(
                executor
                    .finish_tokens(
                        row.next_frontier().sequence(),
                        ExecutionEndReason::Completed {
                            input_tokens: 3,
                            output_tokens: 2,
                        },
                    )
                    .await
                    .unwrap(),
            )
            .await;
        }
    }
    close(greedy, &greedy_owners, false).await;
    close(full, &full_owners, false).await;
}

#[cfg(any(
    feature = "cuda",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
async fn overlapped_failure<B: MoeLlmBackend>(devices: Option<[usize; 2]>)
where
    LlamaFamilyModel<B, KvFp16>: DecoderOnlyLLM + LlamaPipelineStageBatchOps<B> + Send,
{
    let (executor, owners) = build::<B, KvFp16>(LlamaPipelineMode::Overlapped, devices);
    let mut frontiers = Vec::new();
    for _ in 0..16 {
        let initial = admit(&executor, &RequestId::new()).await;
        let prefill = execute(
            &executor,
            &initial,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        )
        .await;
        frontiers.push(prefill.next_frontier().clone());
    }
    let peer = admit(&executor, &RequestId::new()).await;
    let before = owners
        .each_ref()
        .map(|control| control.forwards.load(Ordering::SeqCst));
    owners[2].fail_once.store(true, Ordering::SeqCst);
    let policy = LogitsReturnPolicy::FullLogits;
    let inputs: Vec<_> = frontiers
        .iter()
        .map(|f| {
            ExecutionWork::new(
                f,
                TokenWork::Decode {
                    token: TokenId(4),
                    position: 3,
                },
                &policy,
            )
            .unwrap()
        })
        .collect();
    let result = tokio::time::timeout(WAIT, executor.execute_tokens(&inputs))
        .await
        .expect("original stage0 thread must join after stage1 fails")
        .unwrap();
    let ExecutionBatchOutcome::Interrupted(interrupted) = result else {
        panic!("partly executed physical pipeline cannot complete")
    };
    let (completed, remaining, causes) = interrupted.into_parts();
    assert!(completed.is_empty());
    assert_eq!(remaining.len(), 16);
    assert!(causes.iter().any(|cause| matches!(cause,ExecutionInterruptionCause::Error(error) if error.to_string().contains(PIPELINE_ERROR))));
    for (participant, frontier) in remaining.into_iter().zip(&frontiers) {
        assert_eq!(participant.sequence(), frontier.sequence());
        assert!(matches!(
            participant.progress(),
            ExecutionProgress::Terminal {
                disposition: TerminalModelDisposition::Failed,
                ..
            }
        ));
        let (_, _, _, continuation, _) = participant.into_parts();
        let ended = match continuation {
            ContinuationDisposition::Ended(ended) => ended,
            ContinuationDisposition::Held(held) => {
                tokio::time::timeout(WAIT, held.wait())
                    .await
                    .unwrap()
                    .unwrap()
                    .ended
            }
            _ => panic!("physical cohort may not be replayed after a later-stage error"),
        };
        assert_eq!(ended.sequence(), frontier.sequence());
    }
    let after = owners
        .each_ref()
        .map(|control| control.forwards.load(Ordering::SeqCst));
    // Stage0 may have already finished the second original microbatch. Stage1
    // and head encoded exactly one. Neither thread is left running on return.
    assert!((1..=2).contains(&(after[0] - before[0])));
    assert_eq!(after[1] - before[1], 1);
    assert_eq!(after[2] - before[2], 1);
    let mut oracle = reference();
    let completed = execute(
        &executor,
        &peer,
        TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
    )
    .await;
    compare_logits(&completed, &oracle.prefill_internal("peer", &[1, 2, 3]));
    assert_eq!(
        owners
            .each_ref()
            .map(|control| control.forwards.load(Ordering::SeqCst)),
        after.map(|count| count + 1)
    );
    terminal(
        executor
            .finish_tokens(
                peer.sequence(),
                ExecutionEndReason::Completed {
                    input_tokens: 3,
                    output_tokens: 1,
                },
            )
            .await
            .unwrap(),
    )
    .await;
    oracle.release("peer");
    oracle.close().unwrap();
    // The original request/cohort error was already delivered and retired.
    // Whole-executor close reports its own recovery/retirement failures.
    close(executor, &owners, false).await;
}
