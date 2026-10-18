use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::Backend;
use ferrum_models::{load_runtime_lora_adapter, ActiveLoraAdapter, LoraAdapterConfig};
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
    let encoded: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, shape, values)| {
            let bytes = f32_bytes(values);
            let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
            (
                (*name).to_string(),
                TensorView::new(Dtype::F32, shape.clone(), leaked).unwrap(),
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

    let adapter = load_runtime_lora_adapter::<CpuBackend>(&ActiveLoraAdapter {
        name: "sql".to_string(),
        path: tmp.path().to_path_buf(),
    })
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

#[test]
fn runtime_lora_apply_projection_adds_scaled_delta() {
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
    let adapter = load_runtime_lora_adapter::<CpuBackend>(&ActiveLoraAdapter {
        name: "sql".to_string(),
        path: tmp.path().to_path_buf(),
    })
    .unwrap();

    let mut ctx = CpuBackend::new_context();
    let input = CpuBackend::from_slice(&[1.0, 2.0, 3.0]);
    let mut out = CpuBackend::from_slice(&[10.0, 20.0, 30.0, 40.0]);
    adapter
        .apply_projection(&mut ctx, 7, "qkv_proj", &input, &mut out, 1)
        .unwrap();

    assert_eq!(CpuBackend::to_vec(&out, 4), vec![13.0, 26.0, 39.0, 40.0]);
}

#[cfg(feature = "test-support")]
mod execution {
    use super::*;
    use ferrum_interfaces::model_executor::{
        DecodeInput, PrefillInput, UnifiedBatch, UnifiedBatchItem,
    };
    use ferrum_models::executor::common;
    use ferrum_models::test_support::{tiny_llama_executor, TinyLlamaConfig};
    use ferrum_types::FerrumError;

    pub(super) fn tokens(ids: &[u32]) -> ferrum_interfaces::TensorRef {
        common::wrap_tensor(common::tokens_to_tensor(ids, &candle_core::Device::Cpu).unwrap())
    }

    pub(super) fn metadata(path: &std::path::Path) -> HashMap<String, serde_json::Value> {
        HashMap::from([
            ("ferrum_lora_adapter".into(), serde_json::json!("delta")),
            ("ferrum_lora_path".into(), serde_json::json!(path)),
        ])
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
        let adapter = metadata(directory.path());
        let mut outputs = Vec::new();
        for selection in [HashMap::new(), adapter.clone(), HashMap::new()] {
            let prefill = executor
                .prefill(&PrefillInput::new(tokens(&[0, 1, 2])).with_metadata(selection.clone()))
                .await
                .unwrap();
            let decode = executor
                .decode(&DecodeInput::new(tokens(&[3]), prefill.kv_cache).with_metadata(selection))
                .await
                .unwrap();
            outputs.push((
                prefill.logits.to_vec_f32().unwrap(),
                decode.logits.to_vec_f32().unwrap(),
            ));
            executor.release_cache(&decode.kv_cache.cache_id());
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
    async fn lora_invalid_second_unified_member_cannot_advance_first_cache() {
        let cfg = TinyLlamaConfig::default();
        let executor = tiny_llama_executor(&cfg);
        let reference = tiny_llama_executor(&cfg);
        let prompt = PrefillInput::new(tokens(&[0, 1]));
        let actual = executor.prefill(&prompt).await.unwrap();
        let expected = reference.prefill(&prompt).await.unwrap();
        let directory = model_adapter(&cfg);
        let adapter = metadata(directory.path());
        let malformed = HashMap::from([
            ("ferrum_lora_adapter".into(), serde_json::json!(17)),
            ("ferrum_lora_path".into(), serde_json::json!(false)),
        ]);
        let valid_prefill = PrefillInput::new(tokens(&[0, 1])).with_metadata(adapter.clone());
        let invalid_prefill = PrefillInput::new(tokens(&[0, 1])).with_metadata(malformed.clone());
        let result = executor
            .batch_prefill(&[valid_prefill, invalid_prefill])
            .await;
        assert!(matches!(result, Err(FerrumError::RequestValidation { .. })));
        let valid_decode =
            DecodeInput::new(tokens(&[2]), actual.kv_cache.clone()).with_metadata(adapter.clone());
        let invalid_decode = DecodeInput::new(tokens(&[2]), actual.kv_cache.clone())
            .with_metadata(malformed.clone());
        let result = executor
            .batch_decode(&[valid_decode.clone(), invalid_decode.clone()])
            .await;
        assert!(matches!(result, Err(FerrumError::RequestValidation { .. })));
        let result = executor
            .forward_verify(&[valid_decode.clone(), invalid_decode])
            .await;
        assert!(matches!(result, Err(FerrumError::RequestValidation { .. })));
        // A valid but different later selection must not be ignored either.
        let result = executor
            .forward_verify(&[
                valid_decode,
                DecodeInput::new(tokens(&[3]), actual.kv_cache.clone()),
            ])
            .await;
        assert!(matches!(result, Err(FerrumError::RequestValidation { .. })));
        let mut second = UnifiedBatchItem {
            seq_id: "invalid-lora-second".into(),
            q_tokens: vec![2, 3],
            kv_cache: std::sync::Arc::new(common::GenericKvCacheHandle::new(
                cfg.num_layers,
                cfg.num_heads,
                cfg.head_dim,
                candle_core::Device::Cpu,
                0,
                "invalid-lora-second".into(),
            )),
            recurrent_state: None,
            pos_offset: 0,
            is_final_chunk: true,
            metadata: malformed,
            logits_policy: Default::default(),
        };
        let first = UnifiedBatchItem {
            seq_id: actual.kv_cache.cache_id(),
            q_tokens: vec![2, 3],
            kv_cache: actual.kv_cache.clone(),
            recurrent_state: None,
            pos_offset: 2,
            is_final_chunk: true,
            metadata: adapter,
            logits_policy: Default::default(),
        };
        let result = executor
            .unified_decode(&UnifiedBatch {
                items: vec![first, second.clone()],
            })
            .await;
        assert!(
            matches!(result, Err(FerrumError::RequestValidation { .. })),
            "{result:?}"
        );
        let metrics = executor.lora_metrics_snapshot().unwrap();
        assert_eq!(metrics["active_cache_bindings"].as_u64().unwrap(), 0);
        assert_eq!(metrics["adapter_count"].as_u64().unwrap(), 0);
        let actual = executor
            .decode(&DecodeInput::new(tokens(&[2]), actual.kv_cache))
            .await
            .unwrap();
        let expected = reference
            .decode(&DecodeInput::new(tokens(&[2]), expected.kv_cache))
            .await
            .unwrap();
        assert_eq!(
            actual.logits.to_vec_f32().unwrap(),
            expected.logits.to_vec_f32().unwrap()
        );
        executor.release_cache(&actual.kv_cache.cache_id());
        reference.release_cache(&expected.kv_cache.cache_id());
        second.metadata.clear();
        executor
            .unified_decode(&UnifiedBatch {
                items: vec![second],
            })
            .await
            .unwrap();
        executor.release_cache("invalid-lora-second");
    }
}

#[cfg(feature = "test-support")]
#[path = "lora_runtime_loader/preparation.rs"]
mod preparation;
