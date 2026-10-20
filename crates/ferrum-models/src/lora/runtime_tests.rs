use crate::lora::{load_runtime_lora_adapter, RuntimeLoraWorkspace};
use crate::{ActiveLoraAdapter, LoraAdapterConfig};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::{Backend, BackendCompletion};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::collections::HashMap;
use tempfile::TempDir;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn write_adapter(
    dir: &std::path::Path,
    config: &LoraAdapterConfig,
    tensors: Vec<(&str, Vec<usize>, Vec<f32>)>,
) {
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_vec_pretty(config).unwrap(),
    )
    .unwrap();
    let encoded_bytes: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, values)| f32_bytes(values))
        .collect();
    let encoded: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .zip(&encoded_bytes)
        .map(|((name, shape, _), bytes)| {
            (
                (*name).to_string(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    serialize_to_file(
        encoded,
        &None::<HashMap<String, String>>,
        &dir.join("adapter_model.safetensors"),
    )
    .unwrap();
}

#[test]
fn runtime_lora_loader_materializes_projection_metadata() {
    let tmp = TempDir::new().unwrap();
    let config = LoraAdapterConfig {
        r: 2,
        lora_alpha: 6,
        target_modules: vec!["qkv_proj".to_string()],
        base_model_name_or_path: "qwen3:0.6b".to_string(),
    };
    write_adapter(
        tmp.path(),
        &config,
        vec![
            (
                "base_model.model.layers.7.self_attn.qkv_proj.lora_A.weight",
                vec![2, 3],
                vec![0.1; 6],
            ),
            (
                "base_model.model.layers.7.self_attn.qkv_proj.lora_B.weight",
                vec![4, 2],
                vec![0.2; 8],
            ),
        ],
    );

    let mut ctx = CpuBackend::new_context(&ferrum_types::BackendContextConfig::default()).unwrap();
    let adapter = load_runtime_lora_adapter::<CpuBackend>(
        &mut ctx,
        &ActiveLoraAdapter {
            name: "sql".to_string(),
            path: tmp.path().to_path_buf(),
        },
    )
    .unwrap();

    assert_eq!(adapter.name, "sql");
    assert_eq!(adapter.config.r, 2);
    assert_eq!(adapter.linears.len(), 1);
    let linear = &adapter.linears[0];
    assert_eq!(linear.layer_index, Some(7));
    assert_eq!(linear.target_module, "qkv_proj");
    assert_eq!(linear.in_features, 3);
    assert_eq!(linear.out_features, 4);
    assert_eq!(linear.rank, 2);
    assert_eq!(linear.scaling, 3.0);
}

fn lora_workspace_growth_matches_complete_outputs<B: Backend>() {
    let tmp = TempDir::new().unwrap();
    let config = LoraAdapterConfig {
        r: 2,
        lora_alpha: 6,
        target_modules: vec!["qkv_proj".to_string()],
        base_model_name_or_path: "qwen3:0.6b".to_string(),
    };
    write_adapter(
        tmp.path(),
        &config,
        vec![
            (
                "base_model.model.layers.7.self_attn.qkv_proj.lora_A.weight",
                vec![2, 3],
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ),
            (
                "base_model.model.layers.7.self_attn.qkv_proj.lora_B.weight",
                vec![4, 2],
                vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            ),
        ],
    );
    let mut ctx = B::new_context(&ferrum_types::BackendContextConfig::default()).unwrap();
    let mut workspace = RuntimeLoraWorkspace::<B>::default();
    let mut input = None;
    let mut out = None;
    B::begin_execution(&mut ctx).unwrap();
    let adapter = load_runtime_lora_adapter::<B>(
        &mut ctx,
        &ActiveLoraAdapter {
            name: "sql".to_string(),
            path: tmp.path().to_path_buf(),
        },
    )
    .unwrap();
    // Every device owner precedes this guard, so an allocation/encoding error
    // retires this original context before the payload can unwind.
    let mut retirement = crate::common::construction::ConstructionContext::<B>::new(&mut ctx);

    let inputs = [1.0, 2.0, 3.0, -2.0, 0.5, 7.0];
    let initial = [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0];
    // A selects (x0, x1), B produces (x0, x1, x0+x1, 0), and the
    // declared alpha/rank is 6/2 = 3. Every result is exactly representable
    // in F16/F32; the third input coordinate must not influence any output.
    let expected = [13.0, 26.0, 39.0, 40.0, -5.0, 3.5, -1.5, 4.0];
    for rows in [1, 2] {
        if rows > 1 {
            B::begin_execution(retirement.context).unwrap();
        }
        workspace
            .prepare(retirement.context, &adapter, rows)
            .unwrap();
        input = Some(B::from_slice_in_context(
            retirement.context,
            &inputs[..rows * 3],
        ));
        out = Some(B::from_slice_in_context(
            retirement.context,
            &initial[..rows * 4],
        ));
        let applied = adapter
            .apply_projection(
                retirement.context,
                &mut workspace,
                7,
                "qkv_proj",
                input.as_ref().unwrap(),
                out.as_mut().unwrap(),
                rows,
            )
            .unwrap();
        // Observe once after both GEMMs and the scaled addition. Keep all
        // inputs, outputs and this context's workspace alive through this boundary.
        assert!(matches!(
            B::observe_completion(retirement.context),
            BackendCompletion::Succeeded
        ));
        let actual = B::to_vec_in_context(retirement.context, out.as_ref().unwrap(), rows * 4);
        assert_eq!(applied, 1);
        assert_eq!(actual, expected[..rows * 4]);
    }
    B::retire_context(retirement.context).unwrap();
    retirement.transfer();
    drop(retirement);
    drop(out);
    drop(input);
    drop(workspace);
    drop(adapter);
    drop(ctx);
}

#[test]
fn runtime_lora_apply_projection_adds_scaled_delta() {
    lora_workspace_growth_matches_complete_outputs::<CpuBackend>();
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires actual Metal execution and LoRA workspace growth"]
fn metal_lora_workspace_growth_matches_complete_outputs() {
    lora_workspace_growth_matches_complete_outputs::<ferrum_kernels::backend::metal::MetalBackend>(
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires actual CUDA execution and LoRA workspace growth"]
fn cuda_lora_workspace_growth_matches_complete_outputs() {
    lora_workspace_growth_matches_complete_outputs::<ferrum_kernels::backend::cuda::CudaBackend>();
}

mod execution {
    use super::*;
    use crate::test_support::{tiny_llama_executor, TinyLlamaConfig};
    use ferrum_execution::model_executor::{
        ExecutorPrefillAdmission, ExecutorSamplingOutput, LogitsReturnPolicy, PrefillChunk,
    };
    use ferrum_execution::token_execution::*;
    use ferrum_execution::ModelExecutor;
    use ferrum_interfaces::adapter::LoraAdapterSelection;
    use ferrum_types::{FerrumError, RequestId, Result, TokenId};

    pub(super) fn selection(path: &std::path::Path) -> LoraAdapterSelection<'_> {
        LoraAdapterSelection {
            name: "delta",
            path: path.to_str().unwrap(),
        }
    }

    pub(super) async fn try_admit(
        executor: &dyn ModelExecutor,
        request: &RequestId,
        ids: &[u32],
        adapter: Option<LoraAdapterSelection<'_>>,
    ) -> Result<TokenAdmissionOutcome> {
        let tokens: Vec<_> = ids.iter().copied().map(TokenId::new).collect();
        executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_product_request(
                    request,
                    &tokens,
                    16,
                    tokens.len(),
                    0,
                )?,
                adapter,
            })
            .await
    }

    pub(super) async fn admit(
        executor: &dyn ModelExecutor,
        request: &RequestId,
        ids: &[u32],
        adapter: Option<LoraAdapterSelection<'_>>,
    ) -> ExecutionFrontier {
        let outcome = try_admit(executor, request, ids, adapter).await.unwrap();
        let TokenAdmissionOutcome::Admitted(frontier) = outcome else {
            panic!("tiny LoRA request did not admit: {outcome:?}");
        };
        assert_eq!(frontier.consumed_tokens(), 0);
        frontier
    }

    pub(super) async fn execute(
        executor: &dyn ModelExecutor,
        frontier: &ExecutionFrontier,
        work: TokenWork,
    ) -> ExecutionCompletion {
        let policy = LogitsReturnPolicy::FullLogits;
        let input = ExecutionWork::new(frontier, work, &policy).unwrap();
        let outcome = executor
            .execute_tokens(std::slice::from_ref(&input))
            .await
            .unwrap();
        let ExecutionBatchOutcome::Completed(batch) = outcome else {
            panic!("tiny LoRA work did not complete: {outcome:?}");
        };
        let mut completed = batch.into_completions();
        assert_eq!(completed.len(), 1);
        let completion = completed.pop().unwrap();
        completion
            .validate_for(0, &input, executor.info().vocab_size)
            .unwrap();
        completion
    }

    pub(super) async fn prefill(
        executor: &dyn ModelExecutor,
        frontier: &ExecutionFrontier,
        count: usize,
        total: usize,
    ) -> ExecutionCompletion {
        execute(
            executor,
            frontier,
            TokenWork::Prefill(
                PrefillChunk::new(frontier.consumed_tokens(), count, total).unwrap(),
            ),
        )
        .await
    }

    pub(super) async fn decode(
        executor: &dyn ModelExecutor,
        frontier: &ExecutionFrontier,
        token: u32,
    ) -> ExecutionCompletion {
        execute(
            executor,
            frontier,
            TokenWork::Decode {
                token: TokenId::new(token),
                position: frontier.consumed_tokens(),
            },
        )
        .await
    }

    pub(super) fn logits(completion: &ExecutionCompletion) -> &[f32] {
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
            completion.product()
        else {
            panic!("expected the complete model vocabulary: {completion:?}");
        };
        assert!(logits.iter().all(|value| value.is_finite()));
        logits
    }

    pub(super) async fn finish(
        executor: &dyn ModelExecutor,
        frontier: &ExecutionFrontier,
        input_tokens: usize,
        output_tokens: usize,
    ) {
        let closed = executor
            .finish_tokens(
                frontier.sequence(),
                ExecutionEndReason::Completed {
                    input_tokens,
                    output_tokens,
                },
            )
            .await
            .unwrap();
        assert!(closed.error.is_none(), "{:?}", closed.error);
        let ended = match closed.disposition {
            ExecutionCloseDisposition::Ended(ended) => ended,
            ExecutionCloseDisposition::Held(held) => {
                let terminal = held.wait().await.unwrap();
                assert!(terminal.error.is_none(), "{:?}", terminal.error);
                terminal.ended
            }
        };
        assert_eq!(ended.sequence(), frontier.sequence());
    }

    pub(super) fn model_adapter(cfg: &TinyLlamaConfig) -> TempDir {
        let directory = TempDir::new().unwrap();
        let mut a = vec![0.0; cfg.hidden_size];
        a[0] = 1.0;
        let b = (0..cfg.hidden_size)
            .map(|index| (index as f32 + 1.0) * 0.25)
            .collect();
        write_adapter(
            directory.path(),
            &LoraAdapterConfig {
                r: 1,
                lora_alpha: 2,
                target_modules: vec!["o_proj".into()],
                base_model_name_or_path: "tiny-llama-test".into(),
            },
            vec![
                (
                    "base_model.model.layers.0.self_attn.o_proj.lora_A.weight",
                    vec![1, cfg.hidden_size],
                    a,
                ),
                (
                    "base_model.model.layers.0.self_attn.o_proj.lora_B.weight",
                    vec![cfg.hidden_size, 1],
                    b,
                ),
            ],
        );
        directory
    }

    #[tokio::test]
    async fn lora_real_executor_changes_logits_without_contaminating_base() {
        let cfg = TinyLlamaConfig::default();
        let executor = tiny_llama_executor(&cfg);
        assert_eq!(
            executor.adapter_execution_support(),
            ferrum_interfaces::adapter::AdapterExecutionSupport::PerSequenceLora
        );
        let directory = model_adapter(&cfg);
        let mut outputs = Vec::new();
        for adapter in [None, Some(selection(directory.path())), None] {
            let start = admit(executor.as_ref(), &RequestId::new(), &[0, 1, 2], adapter).await;
            let first = prefill(executor.as_ref(), &start, 3, 3).await;
            let next = decode(executor.as_ref(), first.next_frontier(), 3).await;
            outputs.push((logits(&first).to_vec(), logits(&next).to_vec()));
            finish(executor.as_ref(), next.next_frontier(), 3, 2).await;
            assert_eq!(
                executor.lora_metrics_snapshot().unwrap()["active_cache_bindings"],
                0
            );
        }
        assert_eq!(
            outputs[0], outputs[2],
            "adapter must not alter a later base request"
        );
        for (base, adapted) in [
            (&outputs[0].0, &outputs[1].0),
            (&outputs[0].1, &outputs[1].1),
        ] {
            assert!(base.iter().chain(adapted).all(|value| value.is_finite()));
            assert!(
                base.iter().zip(adapted).any(|(a, b)| (a - b).abs() > 1e-6),
                "the actual model logits must include the adapter delta"
            );
        }
    }

    #[tokio::test]
    async fn lora_invalid_admission_cannot_advance_or_rebind_existing_base() {
        let cfg = TinyLlamaConfig::default();
        let executor = tiny_llama_executor(&cfg);
        let reference = tiny_llama_executor(&cfg);
        let request = RequestId::new();
        let start = admit(executor.as_ref(), &request, &[0, 1], None).await;
        let reference_start = admit(reference.as_ref(), &RequestId::new(), &[0, 1], None).await;
        let actual = prefill(executor.as_ref(), &start, 2, 2).await;
        let expected = prefill(reference.as_ref(), &reference_start, 2, 2).await;
        assert_eq!(logits(&actual), logits(&expected));
        let directory = model_adapter(&cfg);
        let rejected_request = RequestId::new();

        // Selection is immutable admission state, not per-work metadata. Invalid
        // admissions cannot publish a base fallback or change a live incarnation.
        for invalid in [
            LoraAdapterSelection {
                name: "",
                path: directory.path().to_str().unwrap(),
            },
            LoraAdapterSelection {
                name: "delta",
                path: "",
            },
        ] {
            assert!(matches!(
                try_admit(executor.as_ref(), &rejected_request, &[0, 1], Some(invalid)).await,
                Err(FerrumError::RequestValidation { .. })
            ));
        }
        assert!(matches!(
            try_admit(
                executor.as_ref(),
                &request,
                &[0, 1],
                Some(selection(directory.path())),
            )
            .await,
            Err(FerrumError::RequestValidation { .. })
        ));
        let metrics = executor.lora_metrics_snapshot().unwrap();
        assert_eq!(metrics["active_cache_bindings"], 0);
        assert_eq!(metrics["adapter_count"], 0);
        let actual = decode(executor.as_ref(), actual.next_frontier(), 2).await;
        let expected = decode(reference.as_ref(), expected.next_frontier(), 2).await;
        assert_eq!(logits(&actual), logits(&expected));
        finish(executor.as_ref(), actual.next_frontier(), 2, 2).await;
        finish(reference.as_ref(), expected.next_frontier(), 2, 2).await;

        // The rejected request never owned a live incarnation; it can now
        // explicitly admit a base request through the same public protocol.
        let fresh = admit(executor.as_ref(), &rejected_request, &[0, 1], None).await;
        let first = prefill(executor.as_ref(), &fresh, 2, 2).await;
        assert!(!logits(&first).is_empty());
        finish(executor.as_ref(), first.next_frontier(), 2, 1).await;
    }
}

mod preparation;
