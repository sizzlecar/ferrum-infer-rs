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
    use ferrum_execution::model_executor::PrefillInput;
    use ferrum_models::vnext::ProductionWeightArtifact;
    use ferrum_runtime::{ModelSourceKind, OriginalModelSource, OriginalModelSources};
    use ferrum_types::{Device, EngineConfig};
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
        let tokens =
            ferrum_models::executor::common::tokens_to_tensor(&[0, 1], &candle_core::Device::Cpu)
                .unwrap();
        let output = executor
            .prefill(&PrefillInput::new(
                ferrum_models::executor::common::wrap_tensor(tokens),
            ))
            .await
            .unwrap();
        assert_eq!(output.logits.shape().last().copied(), Some(4));
        assert!(output
            .logits
            .to_vec_f32()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
        let cache = output
            .kv_cache
            .as_any()
            .downcast_ref::<ferrum_models::executor::common::GenericKvCacheHandle>()
            .unwrap();
        executor.release_cache(cache.request_cache_id());
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
}
