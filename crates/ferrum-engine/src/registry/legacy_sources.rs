//! Legacy model construction consumes the same frozen semantic source that
//! selected its architecture. Physical weight metadata cannot select it again.

use std::path::Path;
use std::sync::Arc;

use ferrum_execution::ModelExecutor;
use ferrum_models::vnext::ProductionModelSourceBundle;
use ferrum_models::{ConfigManager, LlmExecutor, ModelDefinition};
use ferrum_types::{ModelInfo, Result};

pub(super) async fn load_definition(
    weight_path: &Path,
    sources: Option<&ProductionModelSourceBundle>,
) -> Result<ModelDefinition> {
    let mut manager = ConfigManager::new();
    match sources {
        Some(sources) => manager.load_from_bytes(sources.config_json()),
        None => manager.load_from_path(weight_path).await,
    }
}

pub(super) fn bind_executor(
    model: Box<dyn ferrum_models::common::DecoderOnlyLLM>,
    info: ModelInfo,
    sources: Option<Arc<ProductionModelSourceBundle>>,
) -> Arc<dyn ModelExecutor + Send + Sync> {
    Arc::new(match sources {
        Some(sources) => LlmExecutor::new_with_model_sources(model, info, sources),
        None => LlmExecutor::new(model, info),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{ComponentConfig, ComponentFactory, LlmExecutorFactory};
    use ferrum_execution::model_executor::{
        ExecutorPrefillAdmission, ExecutorSamplingOutput, LogitsReturnPolicy, PrefillChunk,
    };
    use ferrum_execution::token_execution::{
        ExecutionAdmission, ExecutionBatchOutcome, ExecutionCloseDisposition, ExecutionProduct,
        ExecutionWork, TokenAdmissionOutcome, TokenWork,
    };
    use ferrum_models::vnext::ProductionWeightArtifact;
    use ferrum_runtime::{ModelSourceKind, OriginalModelSource, OriginalModelSources};
    use ferrum_types::{Device, EngineConfig, RequestId, TokenId};
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use serde_json::json;

    fn semantic_config() -> serde_json::Value {
        json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
            "num_attention_heads": 1, "num_key_value_heads": 1,
            "vocab_size": 4, "max_position_embeddings": 32,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0
        })
    }

    fn checkpoint(root: &Path) {
        let mut shapes = vec![
            ("model.embed_tokens.weight".to_owned(), vec![4, 8]),
            ("model.norm.weight".to_owned(), vec![8]),
            ("lm_head.weight".to_owned(), vec![4, 8]),
        ];
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            shapes.push((format!("model.layers.0.{norm}.weight"), vec![8]));
        }
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            shapes.push((
                format!("model.layers.0.self_attn.{projection}.weight"),
                vec![8, 8],
            ));
        }
        for projection in ["gate_proj", "up_proj"] {
            shapes.push((
                format!("model.layers.0.mlp.{projection}.weight"),
                vec![16, 8],
            ));
        }
        shapes.push(("model.layers.0.mlp.down_proj.weight".into(), vec![8, 16]));
        let payloads: Vec<_> = shapes
            .iter()
            .map(|(_, shape)| {
                (0..shape.iter().product::<usize>())
                    .flat_map(|_| 0.125_f32.to_le_bytes())
                    .collect::<Vec<_>>()
            })
            .collect();
        let views = shapes.iter().zip(&payloads).map(|((name, shape), bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        });
        serialize_to_file(views, &None, &root.join("model.safetensors")).unwrap();
    }

    #[tokio::test]
    async fn legacy_factory_consumes_frozen_semantics_and_reports_actual_sources() {
        let root = tempfile::tempdir().unwrap();
        let semantic = root.path().join("semantic");
        let weights = root.path().join("weights");
        std::fs::create_dir_all(&semantic).unwrap();
        std::fs::create_dir_all(&weights).unwrap();
        std::fs::write(
            semantic.join("config.json"),
            serde_json::to_vec(&semantic_config()).unwrap(),
        )
        .unwrap();
        std::fs::write(semantic.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
        // The physical config describes storage; architecture and dimensions
        // come from the semantic role. Actual tensors match that definition.
        let physical = json!({"torch_dtype": "float32"});
        std::fs::write(
            weights.join("config.json"),
            serde_json::to_vec(&physical).unwrap(),
        )
        .unwrap();
        checkpoint(&weights);
        let original = |path: &Path| OriginalModelSource {
            kind: ModelSourceKind::LocalDirectory,
            location: path.display().to_string(),
            requested_revision: None,
        };
        let sources = Arc::new(
            ProductionModelSourceBundle::open(
                &semantic,
                &semantic,
                ProductionWeightArtifact::safetensors_directory(&weights),
                OriginalModelSources {
                    semantic: original(&semantic),
                    tokenizer: original(&semantic),
                    weights: original(&weights),
                },
            )
            .unwrap(),
        );
        // A subsequent file replacement cannot reroute the frozen model.
        std::fs::write(
            semantic.join("config.json"),
            br#"{"architectures":["Qwen3MoeForCausalLM"]}"#,
        )
        .unwrap();
        let mut engine = EngineConfig::default();
        engine.backend.device = Device::CPU;
        let component =
            ComponentConfig::from_engine_config_and_sources(&engine, Some(Arc::clone(&sources)));
        let executor = LlmExecutorFactory.create(&component).await.unwrap();
        assert_eq!(executor.info().hidden_size, 8);
        assert_eq!(executor.info().num_layers, 1);
        assert_eq!(
            executor.bound_resolved_sources(),
            Some(sources.resolved_sources())
        );
        let request = RequestId::new();
        let tokens = [TokenId::new(0), TokenId::new(1)];
        let TokenAdmissionOutcome::Admitted(frontier) = executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_diagnostic(&request, &tokens, 4),
                adapter: None,
            })
            .await
            .unwrap()
        else {
            panic!("tiny frozen model admission must succeed")
        };
        let policy = LogitsReturnPolicy::FullLogits;
        let work = ExecutionWork::new(
            &frontier,
            TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
            &policy,
        )
        .unwrap();
        let ExecutionBatchOutcome::Completed(output) =
            executor.execute_tokens(&[work]).await.unwrap()
        else {
            panic!("tiny frozen model prefill must complete")
        };
        let completions = output.into_completions();
        assert_eq!(completions.len(), 1);
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
            completions[0].product()
        else {
            panic!("final prefill must return its real logits")
        };
        assert_eq!(logits.len(), 4);
        assert!(logits.iter().all(|value| value.is_finite()));
        assert_eq!(completions[0].next_frontier().consumed_tokens(), 2);
        let closed = executor.cancel_tokens(frontier.sequence()).unwrap();
        assert!(closed.error.is_none());
        let ExecutionCloseDisposition::Ended(ended) = closed.disposition else {
            panic!("tiny CPU model must release its exact incarnation")
        };
        assert_eq!(ended.sequence(), frontier.sequence());
    }

    #[tokio::test]
    async fn legacy_definition_without_sources_keeps_explicit_path_loading() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&semantic_config()).unwrap(),
        )
        .unwrap();
        let definition = load_definition(root.path(), None).await.unwrap();
        assert_eq!(definition.hidden_size, 8);
        assert_eq!(definition.architecture, ferrum_models::Architecture::Llama);
    }

    #[test]
    fn legacy_factory_keeps_instance_capacity_after_another_config_is_applied() {
        use ferrum_interfaces::kv_dtype::KvFp16;
        use ferrum_kernels::backend::cpu::CpuBackend;
        use ferrum_models::models::LlamaFamilyConfig;
        use ferrum_types::RuntimeConfigSnapshot;

        let root = tempfile::tempdir().unwrap();
        checkpoint(root.path());
        let definition = ConfigManager::new()
            .load_from_bytes(&serde_json::to_vec(&semantic_config()).unwrap())
            .unwrap();
        let config = LlamaFamilyConfig::llama_from_def(&definition);
        let mut first = EngineConfig::default();
        first
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([
                ("FERRUM_KV_CAPACITY", "8"),
                ("FERRUM_METAL_PAGED_KV", "0"),
            ]))
            .unwrap();
        let first_input = first.llama_execution_config();
        let mut second = EngineConfig::default();
        second
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([
                ("FERRUM_KV_CAPACITY", "16"),
                ("FERRUM_METAL_PAGED_KV", "0"),
            ]))
            .unwrap();
        let build = |execution| {
            crate::registry::build_llm::<CpuBackend, KvFp16>(
                config.clone(),
                root.path().to_str().unwrap(),
                None,
                None,
                execution,
                ferrum_quantization::NativeSafetensorsLoadOptions::default(),
            )
            .unwrap()
        };
        let mut a = build(first_input);
        let mut b = build(second.llama_execution_config());
        assert_eq!(a.kv_capacity(), 8);
        assert_eq!(b.kv_capacity(), 16);
        assert_eq!(a.prefill("a", &[0, 1]), b.prefill("b", &[0, 1]));
        assert_eq!(a.decode("a", 2, 2), b.decode("b", 2, 2));
        let expected = a.decode("a", 3, 3);
        a.release("a");
        assert_eq!(b.decode("b", 3, 3), expected);
        b.release("b");
    }
}
