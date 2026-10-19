//! Production qualification: the executor receives the real model directly.
//! No wrapper changes its execution mode or manufactures device completion.

use super::*;
use crate::common::llm::ModelTokenExecutionMode;
use crate::models::llama_family::LlamaFamilyConfig;
use crate::models::LlamaFamilyModel;
use crate::test_support::{tiny_model_info, SyntheticLlamaLoader, TinyLlamaConfig};
use ferrum_execution::model_executor::{ExecutorPrefillAdmission, PrefillChunk};
use ferrum_kernels::backend::{cpu::CpuBackend, KvLayer, MoeLlmBackend};
use ferrum_types::{Device, LlamaExecutionConfig, LlamaExecutionOptions};
use std::time::Duration;

const VOCABULARY: usize = 128;

fn config() -> LlamaFamilyConfig {
    LlamaFamilyConfig {
        hidden_size: 256,
        intermediate_size: 512,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 128,
        num_layers: 1,
        vocab_size: VOCABULARY,
        max_seq_len: 32,
        ..Default::default()
    }
}

fn real_model<B: MoeLlmBackend>(paged: bool) -> LlamaFamilyModel<B> {
    let config = config();
    let loader = SyntheticLlamaLoader::new(config.clone());
    LlamaFamilyModel::new_with_execution_config(
        config,
        &loader,
        LlamaExecutionConfig {
            kv_capacity: Some(32),
            paged_max_seqs: 4,
            options: LlamaExecutionOptions {
                metal_paged_kv: Some(paged),
                kv_max_blocks: Some(4),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap()
}

fn fixture<B: MoeLlmBackend>(
    paged: bool,
    mode: ModelTokenExecutionMode,
    device: Device,
) -> LlmExecutor {
    let model = real_model::<B>(paged);
    assert_eq!(model.token_execution_mode(), mode);
    executor_from_model(model, device)
}

fn executor_from_model<B, K>(model: LlamaFamilyModel<B, K>, device: Device) -> LlmExecutor
where
    B: MoeLlmBackend,
    K: KvLayer<B>,
    LlamaFamilyModel<B, K>: DecoderOnlyLLM,
{
    let config = config();
    let mut info = tiny_model_info(&TinyLlamaConfig {
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        num_layers: model.config().num_layers,
        vocab_size: config.vocab_size,
        max_seq_len: config.max_seq_len,
    });
    info.device = device;
    LlmExecutor::new(Box::new(model), info)
}

async fn admit(executor: &LlmExecutor, request: &RequestId) -> ExecutionFrontier {
    let prompt = [TokenId(1), TokenId(2), TokenId(3)];
    match executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(request, &prompt, 6),
            adapter: None,
        })
        .await
        .unwrap()
    {
        TokenAdmissionOutcome::Admitted(frontier) => frontier,
        other => panic!("real model admission failed: {other:?}"),
    }
}

async fn execute(
    executor: &LlmExecutor,
    frontier: &ExecutionFrontier,
    work: TokenWork,
) -> ExecutionCompletion {
    let policy = LogitsReturnPolicy::FullLogits;
    let input = [ExecutionWork::new(frontier, work, &policy).unwrap()];
    let outcome = executor.execute_tokens(&input).await.unwrap();
    let ExecutionBatchOutcome::Completed(batch) = outcome else {
        panic!("real model did not finish its actual work: {outcome:?}");
    };
    let mut completions = batch.into_completions();
    assert_eq!(completions.len(), 1);
    completions.remove(0)
}

fn compare_logits(completion: &ExecutionCompletion, expected: &[f32]) {
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(actual)) =
        completion.product()
    else {
        panic!("qualification requires the complete vocabulary output");
    };
    assert_eq!(actual.len(), VOCABULARY);
    assert_eq!(expected.len(), VOCABULARY);
    // Same full-vector envelope as the real managed numerical tests. The
    // independent CPU model supplies the oracle, never a returned GPU token.
    crate::test_support::numerical::assert_full_logits_qualification(actual, expected);
}

async fn terminal(close: ExecutionCloseOutcome) -> ExecutionEnded {
    assert!(close.error.is_none(), "unexpected immediate close error");
    match close.disposition {
        ExecutionCloseDisposition::Ended(ended) => ended,
        ExecutionCloseDisposition::Held(held) => {
            let terminal = tokio::time::timeout(Duration::from_secs(10), held.wait())
                .await
                .expect("the original worker must finish actual closure")
                .unwrap();
            assert!(terminal.error.is_none(), "unexpected deferred close error");
            terminal.ended
        }
    }
}

fn assert_all_slots_free(executor: &LlmExecutor, paged: bool) {
    assert!(executor.token_sequences.lock().live.is_empty());
    let model = executor.lock_model().unwrap();
    match model.kv_slot_capacity_snapshot() {
        Some(snapshot) => {
            assert!(paged);
            assert_eq!(snapshot.total_blocks, 4);
            assert_eq!(snapshot.free_blocks, snapshot.total_blocks);
        }
        None => assert!(!paged),
    }
}

async fn qualified_completion<B: MoeLlmBackend>(
    paged: bool,
    mode: ModelTokenExecutionMode,
    device: Device,
) {
    let executor = fixture::<B>(paged, mode, device);
    drop(qualified_completion_with_executor(executor, paged, mode).await);
}

async fn qualified_completion_with_executor(
    executor: LlmExecutor,
    paged: bool,
    mode: ModelTokenExecutionMode,
) -> LlmExecutor {
    let mut reference = real_model::<CpuBackend>(false);
    let request = RequestId::new();
    let initial = admit(&executor, &request).await;
    assert_eq!(initial.consumed_tokens(), 0);
    assert_eq!(executor.token_execution_mode, mode);
    assert_eq!(
        executor.token_worker.get().is_some(),
        mode == ModelTokenExecutionMode::ManagedWorker
    );

    let partial = execute(
        &executor,
        &initial,
        TokenWork::Prefill(PrefillChunk::new(0, 2, 3).unwrap()),
    )
    .await;
    assert!(matches!(partial.product(), ExecutionProduct::Intermediate));
    assert_eq!(partial.next_frontier().consumed_tokens(), 2);
    reference.prefill_internal("reference", &[1, 2]);
    let prefill = execute(
        &executor,
        partial.next_frontier(),
        TokenWork::Prefill(PrefillChunk::new(2, 1, 3).unwrap()),
    )
    .await;
    compare_logits(&prefill, &reference.prefill_internal("reference", &[3]));

    // A real product validation failure must not close the original owner or
    // consume its frontier. This is not a simulated hardware failure.
    let rejected = executor
        .finish_tokens(
            initial.sequence(),
            ExecutionEndReason::Completed {
                input_tokens: 4,
                output_tokens: 1,
            },
        )
        .await;
    assert!(matches!(
        rejected,
        Err(FerrumError::RequestValidation { .. })
    ));
    assert!(initial.sequence().observed_end().is_none());
    assert_eq!(
        executor.token_sequences.lock().live[initial.sequence()].frontier,
        *prefill.next_frontier()
    );

    let mut frontier = prefill.next_frontier().clone();
    for (position, token) in [(3, 4), (4, 5)] {
        let decoded = execute(
            &executor,
            &frontier,
            TokenWork::Decode {
                token: TokenId(token),
                position,
            },
        )
        .await;
        compare_logits(
            &decoded,
            &reference.decode_internal("reference", token, position as u32),
        );
        assert_eq!(decoded.next_frontier().consumed_tokens(), position + 1);
        frontier = decoded.next_frontier().clone();
    }
    let ended = terminal(
        executor
            .finish_tokens(
                frontier.sequence(),
                ExecutionEndReason::Completed {
                    input_tokens: 3,
                    output_tokens: 3,
                },
            )
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(ended.sequence(), initial.sequence());
    assert_all_slots_free(&executor, paged);
    match mode {
        ModelTokenExecutionMode::InlineSynchronous => {
            // Inline closure removes the live entry. Its published terminal
            // fact survives, but a second close still requires a live entry.
            assert!(matches!(
                executor.cancel_tokens(initial.sequence()),
                Err(FerrumError::RequestValidation { .. })
            ));
        }
        ModelTokenExecutionMode::ManagedWorker => {
            // The worker additionally coalesces repeated closure observers,
            // including requests made after its live entry was removed.
            let repeated = terminal(executor.cancel_tokens(initial.sequence()).unwrap()).await;
            assert_eq!(repeated, ended);
        }
        ModelTokenExecutionMode::Unsupported => panic!("unqualified completion fixture"),
    }
    assert_eq!(initial.sequence().observed_end(), Some(ended));
    assert_all_slots_free(&executor, paged);

    // The same request id gets a fresh incarnation only after the actual old
    // one ended. Reusing the model must neither replay nor inherit its KV.
    let next = admit(&executor, &request).await;
    assert_ne!(next.sequence(), initial.sequence());
    let next_prefill = execute(
        &executor,
        &next,
        TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
    )
    .await;
    compare_logits(
        &next_prefill,
        &reference.prefill_internal("fresh", &[1, 2, 3]),
    );
    terminal(executor.cancel_tokens(next.sequence()).unwrap()).await;
    assert_all_slots_free(&executor, paged);
    reference.release("reference");
    reference.release("fresh");
    reference.close().unwrap();
    executor
}

#[tokio::test]
async fn qualification_unwrapped_cpu_completes_and_rejects_invalid_close_without_losing_progress() {
    qualified_completion::<CpuBackend>(
        false,
        ModelTokenExecutionMode::InlineSynchronous,
        Device::CPU,
    )
    .await;
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[tokio::test]
#[ignore = "requires actual Metal execution and the production model qualification"]
async fn qualification_unwrapped_metal_completes_and_rejects_invalid_close_without_losing_progress()
{
    for paged in [false, true] {
        qualified_completion::<ferrum_kernels::backend::metal::MetalBackend>(
            paged,
            ModelTokenExecutionMode::ManagedWorker,
            Device::Metal,
        )
        .await;
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[tokio::test]
#[ignore = "requires actual Metal worker ownership and independent CPU logits"]
async fn qualification_unwrapped_metal_cancelled_delivery_closes_original_and_keeps_peer_usable() {
    use std::future::Future;
    use std::task::{Context, Poll};

    let executor = fixture::<ferrum_kernels::backend::metal::MetalBackend>(
        true,
        ModelTokenExecutionMode::ManagedWorker,
        Device::Metal,
    );
    let mut reference = real_model::<CpuBackend>(false);
    let cancelled = admit(&executor, &RequestId::new()).await;
    let peer = admit(&executor, &RequestId::new()).await;
    let expected = reference.prefill_internal("peer", &[1, 2, 3]);
    let mut frontiers = Vec::new();
    for frontier in [&cancelled, &peer] {
        let prefill = execute(
            &executor,
            frontier,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        )
        .await;
        compare_logits(&prefill, &expected);
        frontiers.push(prefill.next_frontier().clone());
    }
    let policy = LogitsReturnPolicy::FullLogits;
    let work = [ExecutionWork::new(
        &frontiers[0],
        TokenWork::Decode {
            token: TokenId(4),
            position: 3,
        },
        &policy,
    )
    .unwrap()];
    let passive = executor
        .token_device_worker()
        .unwrap()
        .observe_terminal(&frontiers[0])
        .unwrap();
    let closing = {
        // Hold the real model mutex: the actual worker cannot perform work or
        // release KV until it is returned. No model wrapper overrides mode or
        // supplies a terminal observation. No guard crosses an await.
        let model = executor.lock_model().unwrap();
        let before = model.kv_slot_capacity_snapshot().unwrap();
        assert_eq!(before.free_blocks, 2);
        let mut pending = Box::pin(executor.execute_tokens(&work));
        let waker = futures::task::noop_waker_ref();
        assert!(matches!(
            pending.as_mut().poll(&mut Context::from_waker(waker)),
            Poll::Pending
        ));
        let closing = executor.cancel_tokens(cancelled.sequence()).unwrap();
        assert!(matches!(
            &closing.disposition,
            ExecutionCloseDisposition::Held(_)
        ));
        assert!(cancelled.sequence().observed_end().is_none());
        drop(pending);
        assert_eq!(model.kv_slot_capacity_snapshot().unwrap().free_blocks, 2);
        closing
    };
    let ended = terminal(closing).await;
    assert_eq!(ended.sequence(), cancelled.sequence());
    assert_eq!(terminal(passive).await, ended);
    {
        let sequences = executor.token_sequences.lock();
        assert_eq!(sequences.live.len(), 1);
        assert_eq!(sequences.live[peer.sequence()].frontier, frontiers[1]);
    }
    assert_eq!(
        executor
            .model
            .lock()
            .as_ref()
            .expect("qualified model is still live")
            .kv_slot_capacity_snapshot()
            .unwrap()
            .free_blocks,
        3
    );
    let decoded = execute(
        &executor,
        &frontiers[1],
        TokenWork::Decode {
            token: TokenId(4),
            position: 3,
        },
    )
    .await;
    compare_logits(&decoded, &reference.decode_internal("peer", 4, 3));
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
    assert_all_slots_free(&executor, true);
    reference.release("peer");
    reference.close().unwrap();
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod encoded_failure {
    use super::*;
    use ferrum_kernels::backend::{metal::MetalBackend, Backend};
    use ferrum_kernels::{Linear, LinearMetadata, LoadedLinear};
    use ferrum_quantization::{QuantConfig, WeightLoader};
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    struct HeadFailure {
        armed: AtomicBool,
        actual_forwards: AtomicUsize,
    }

    struct FaultingHead {
        inner: Box<dyn Linear<MetalBackend>>,
        failure: Arc<HeadFailure>,
    }

    impl Linear<MetalBackend> for FaultingHead {
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
            ctx: &mut <MetalBackend as Backend>::Context,
            input: &<MetalBackend as Backend>::Buffer,
            out: &mut <MetalBackend as Backend>::Buffer,
            m: usize,
        ) {
            self.inner.forward(ctx, input, out, m);
            self.failure.actual_forwards.fetch_add(1, Ordering::SeqCst);
            if self.failure.armed.swap(false, Ordering::SeqCst) {
                // This is after the real head encoded into the original
                // context, before its normal readback/completion boundary.
                std::panic::panic_any(FerrumError::backend("qualification encoded head failure"));
            }
        }
    }

    struct FaultingLoader {
        inner: SyntheticLlamaLoader,
        failure: Arc<HeadFailure>,
    }

    impl WeightLoader<MetalBackend> for FaultingLoader {
        fn load_tensor(
            &self,
            ctx: &mut <MetalBackend as Backend>::Context,
            name: &str,
        ) -> Result<<MetalBackend as Backend>::Buffer> {
            <SyntheticLlamaLoader as WeightLoader<MetalBackend>>::load_tensor(
                &self.inner,
                ctx,
                name,
            )
        }
        fn load_linear(
            &self,
            ctx: &mut <MetalBackend as Backend>::Context,
            name: &str,
        ) -> Result<LoadedLinear<MetalBackend>> {
            let inner = <SyntheticLlamaLoader as WeightLoader<MetalBackend>>::load_linear(
                &self.inner,
                ctx,
                name,
            )?;
            if name == "lm_head" {
                // Graph replay would skip this per-execution host fault.
                // Decorating the mathematical projection drops its graph qualification.
                Ok(LoadedLinear::new(Box::new(FaultingHead {
                    inner: inner.into_eager_box(),
                    failure: Arc::clone(&self.failure),
                })))
            } else {
                Ok(inner)
            }
        }
        fn has_tensor(&self, name: &str) -> bool {
            <SyntheticLlamaLoader as WeightLoader<MetalBackend>>::has_tensor(&self.inner, name)
        }
        fn quant_config(&self) -> Option<&QuantConfig> {
            <SyntheticLlamaLoader as WeightLoader<MetalBackend>>::quant_config(&self.inner)
        }
    }

    #[tokio::test]
    #[ignore = "requires actual Metal head encoding and production model recovery"]
    async fn qualification_unwrapped_metal_encoded_failure_retires_once_and_peer_matches_cpu() {
        let failure = Arc::new(HeadFailure {
            armed: AtomicBool::new(false),
            actual_forwards: AtomicUsize::new(0),
        });
        let loader = FaultingLoader {
            inner: SyntheticLlamaLoader::new(config()),
            failure: Arc::clone(&failure),
        };
        let model = LlamaFamilyModel::<MetalBackend>::new_with_execution_config(
            config(),
            &loader,
            LlamaExecutionConfig {
                kv_capacity: Some(32),
                paged_max_seqs: 4,
                options: LlamaExecutionOptions {
                    metal_paged_kv: Some(true),
                    kv_max_blocks: Some(4),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            model.token_execution_mode(),
            ModelTokenExecutionMode::ManagedWorker
        );
        // Only the weight's mathematical operation has a fault injection;
        // the executor receives the unwrapped production Llama model.
        let executor = executor_from_model(model, Device::Metal);
        let failed = admit(&executor, &RequestId::new()).await;
        let peer = admit(&executor, &RequestId::new()).await;
        assert_eq!(failure.actual_forwards.load(Ordering::SeqCst), 0);
        failure.armed.store(true, Ordering::SeqCst);
        let policy = LogitsReturnPolicy::FullLogits;
        let work = [ExecutionWork::new(
            &failed,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
            &policy,
        )
        .unwrap()];
        let outcome = executor.execute_tokens(&work).await.unwrap();
        let ExecutionBatchOutcome::Interrupted(interrupted) = outcome else {
            panic!("encoded host failure must preserve the original failed participant");
        };
        let (completed, mut remaining, causes) = interrupted.into_parts();
        assert!(completed.is_empty());
        assert_eq!(remaining.len(), 1);
        assert_eq!(causes.len(), 1);
        assert!(
            matches!(&causes[0], ExecutionInterruptionCause::Error(error)
            if error.to_string() == FerrumError::backend("qualification encoded head failure").to_string())
        );
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
                let terminal = tokio::time::timeout(Duration::from_secs(10), held.wait())
                    .await
                    .expect("actual encoded frame must reach terminal")
                    .unwrap();
                assert_eq!(
                    terminal.error.unwrap().to_string(),
                    FerrumError::backend("qualification encoded head failure").to_string()
                );
                terminal.ended
            }
            _ => panic!("failed encoded work cannot be replayed"),
        };
        assert_eq!(ended.sequence(), failed.sequence());
        assert_eq!(failure.actual_forwards.load(Ordering::SeqCst), 1);
        assert_eq!(
            executor
                .model
                .lock()
                .as_ref()
                .expect("qualified model is still live")
                .kv_slot_capacity_snapshot()
                .unwrap()
                .free_blocks,
            3
        );
        assert_eq!(
            executor.token_sequences.lock().live[peer.sequence()].frontier,
            peer
        );

        let mut reference = real_model::<CpuBackend>(false);
        let prefill = execute(
            &executor,
            &peer,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        )
        .await;
        compare_logits(&prefill, &reference.prefill_internal("peer", &[1, 2, 3]));
        let decode = execute(
            &executor,
            prefill.next_frontier(),
            TokenWork::Decode {
                token: TokenId(4),
                position: 3,
            },
        )
        .await;
        compare_logits(&decode, &reference.decode_internal("peer", 4, 3));
        assert_eq!(
            failure.actual_forwards.load(Ordering::SeqCst),
            3,
            "recovery must observe the failed submission without replaying its head"
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
        assert_all_slots_free(&executor, true);
        reference.release("peer");
        reference.close().unwrap();
    }
}

#[tokio::test]
async fn qualification_default_unsupported_never_admits_or_starts_worker() {
    struct Unqualified(crate::common::LlmRuntimeConfig);
    impl DecoderOnlyLLM for Unqualified {
        fn config(&self) -> &crate::common::LlmRuntimeConfig {
            &self.0
        }
        fn prefill(&mut self, _: &str, _: &[u32]) -> Vec<f32> {
            panic!("unqualified execution")
        }
        fn decode(&mut self, _: &str, _: u32, _: u32) -> Vec<f32> {
            panic!("unqualified execution")
        }
        fn release(&mut self, _: &str) {
            panic!("nothing was admitted")
        }
        fn reset(&mut self) {
            panic!("nothing was admitted")
        }
    }
    let model = real_model::<CpuBackend>(false);
    let unqualified = Unqualified(model.config().clone());
    assert_eq!(
        unqualified.token_execution_mode(),
        ModelTokenExecutionMode::Unsupported
    );
    let executor = LlmExecutor::new(
        Box::new(unqualified),
        tiny_model_info(&TinyLlamaConfig::default()),
    );
    let request = RequestId::new();
    let prompt = [TokenId(1), TokenId(2), TokenId(3)];
    assert!(matches!(
        executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_startup(&request, &prompt, 6),
                adapter: None,
            })
            .await,
        Err(FerrumError::Unsupported { .. })
    ));
    assert!(executor.token_worker.get().is_none());
    assert!(executor.token_sequences.lock().live.is_empty());
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[tokio::test]
#[ignore = "requires actual Metal construction and production qualification checks"]
async fn qualification_unwrapped_metal_backbone_and_partial_stage_remain_unsupported() {
    use crate::models::llama_family::LlamaFamilyLayerStageConfig;
    use ferrum_kernels::backend::metal::MetalBackend;

    let mut config = config();
    config.num_layers = 2;
    let loader = SyntheticLlamaLoader::new(config.clone());
    let backbone = LlamaFamilyModel::<MetalBackend>::new_backbone_only_with_execution_config(
        config.clone(),
        &loader,
        LlamaExecutionConfig::default(),
    )
    .unwrap();
    let partial = LlamaFamilyModel::<MetalBackend>::new_layer_stage_with_execution_config(
        config,
        &loader,
        LlamaFamilyLayerStageConfig {
            source_layers: 0..1,
            load_embedding: true,
            load_lm_head: true,
        },
        LlamaExecutionConfig::default(),
    )
    .unwrap();
    assert!(partial.embed.is_some() && partial.lm_head.is_some());
    assert_eq!(partial.source_layer_range(), 0..1);
    for model in [backbone, partial] {
        assert_eq!(
            model.token_execution_mode(),
            ModelTokenExecutionMode::Unsupported
        );
        let executor = executor_from_model(model, Device::Metal);
        let request = RequestId::new();
        let prompt = [TokenId(1), TokenId(2), TokenId(3)];
        assert!(matches!(
            executor
                .admit_tokens(ExecutionAdmission {
                    context: ExecutorPrefillAdmission::for_startup(&request, &prompt, 6),
                    adapter: None,
                })
                .await,
            Err(FerrumError::Unsupported { .. })
        ));
        assert!(executor.token_worker.get().is_none());
        assert!(executor.token_sequences.lock().live.is_empty());
        assert!(executor
            .lock_model()
            .unwrap()
            .kv_slot_capacity_snapshot()
            .is_none());
    }
}

#[cfg(feature = "cuda")]
mod cuda;

mod pipeline;
