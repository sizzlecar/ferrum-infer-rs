//! Qwen3.5 / Qwen3.6 typed model weights.
//!
//! This is the W3 materialization boundary: it turns the resolved semantic
//! weight plan into backend-native buffers and linears, but intentionally does
//! not implement product forward execution yet.

use std::ops::Range;

use ferrum_interfaces::RecurrentStateSpec;
use ferrum_kernels::backend::Backend;
use ferrum_quantization::Linear;
use ferrum_types::{DataType, Device, FerrumError, RequestId, Result};

use crate::{
    common::LlmRuntimeConfig,
    definition::ModelDefinition,
    qwen35_config::{Qwen35LayerType, Qwen35MlpKind, Qwen35TextConfig},
    qwen35_weights::{Qwen35ResolvedWeightPlan, Qwen35WeightPlanLoader},
};

pub struct Qwen35ModelWeights<B: Backend> {
    pub config: Qwen35TextConfig,
    pub runtime_cfg: LlmRuntimeConfig,
    pub embed_tokens: B::Buffer,
    pub final_norm: B::Buffer,
    pub lm_head: Box<dyn Linear<B>>,
    pub layers: Vec<Qwen35LayerWeights<B>>,
}

pub struct Qwen35LayerWeights<B: Backend> {
    pub layer_index: usize,
    pub input_layernorm: B::Buffer,
    pub post_attention_layernorm: B::Buffer,
    pub attention: Qwen35AttentionWeights<B>,
    pub mlp: Qwen35MlpWeights<B>,
}

pub enum Qwen35AttentionWeights<B: Backend> {
    Linear(Qwen35LinearAttentionWeights<B>),
    Full(Qwen35FullAttentionWeights<B>),
}

pub struct Qwen35LinearAttentionWeights<B: Backend> {
    pub qkv_proj: Box<dyn Linear<B>>,
    pub z_proj: Box<dyn Linear<B>>,
    pub b_proj: Box<dyn Linear<B>>,
    pub a_proj: Box<dyn Linear<B>>,
    pub conv1d_weight: B::Buffer,
    pub a_log: B::Buffer,
    pub dt_bias: B::Buffer,
    pub norm_weight: B::Buffer,
    pub out_proj: Box<dyn Linear<B>>,
}

pub struct Qwen35FullAttentionWeights<B: Backend> {
    pub q_proj: Box<dyn Linear<B>>,
    pub k_proj: Box<dyn Linear<B>>,
    pub v_proj: Box<dyn Linear<B>>,
    pub o_proj: Box<dyn Linear<B>>,
    pub q_norm_weight: B::Buffer,
    pub k_norm_weight: B::Buffer,
}

pub enum Qwen35MlpWeights<B: Backend> {
    Dense(Qwen35DenseMlpWeights<B>),
    SparseMoeSharedExpert(Qwen35SparseMoeSharedExpertWeights<B>),
}

pub struct Qwen35DenseMlpWeights<B: Backend> {
    pub gate_proj: Box<dyn Linear<B>>,
    pub up_proj: Box<dyn Linear<B>>,
    pub down_proj: Box<dyn Linear<B>>,
}

pub struct Qwen35SparseMoeSharedExpertWeights<B: Backend> {
    pub router: Box<dyn Linear<B>>,
    pub shared_expert_gate: B::Buffer,
    pub shared_expert_gate_proj: Box<dyn Linear<B>>,
    pub shared_expert_up_proj: Box<dyn Linear<B>>,
    pub shared_expert_down_proj: Box<dyn Linear<B>>,
    pub fused_gate_up_proj: B::Buffer,
    pub fused_down_proj: B::Buffer,
}

pub struct Qwen35RecurrentStateCache<B: Backend> {
    pub request_id: RequestId,
    pub dtype: DataType,
    pub device: Device,
    pub max_batch_slots: usize,
    pub tensors: Vec<Qwen35RecurrentStateTensor<B>>,
}

pub struct Qwen35RecurrentStateTensor<B: Backend> {
    pub layer_index: usize,
    pub name: String,
    pub shape: Vec<usize>,
    pub elements_per_slot: usize,
    pub buffer: B::Buffer,
}

pub fn qwen35_runtime_config(
    config: &Qwen35TextConfig,
    vocab_size: usize,
    max_seq_len: usize,
) -> LlmRuntimeConfig {
    LlmRuntimeConfig {
        hidden_size: config.hidden_size,
        num_layers: config.num_hidden_layers,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        vocab_size,
        max_seq_len,
    }
}

pub fn qwen35_runtime_config_from_definition(def: &ModelDefinition) -> Result<LlmRuntimeConfig> {
    let config =
        Qwen35TextConfig::from_model_definition(def).map_err(ferrum_types::FerrumError::model)?;
    Ok(qwen35_runtime_config(
        &config,
        def.vocab_size,
        def.max_position_embeddings,
    ))
}

impl<B: Backend> Qwen35RecurrentStateCache<B> {
    pub fn from_spec(spec: &RecurrentStateSpec) -> Result<Self> {
        if spec.max_batch_slots == 0 {
            return Err(FerrumError::model(
                "Qwen3.5 recurrent state requires at least one batch slot",
            ));
        }
        let mut tensors = Vec::with_capacity(spec.tensors.len());
        for tensor in &spec.tensors {
            if tensor.shape.is_empty() {
                return Err(FerrumError::model(format!(
                    "Qwen3.5 recurrent tensor layer={} name={} has empty shape",
                    tensor.layer_index, tensor.name
                )));
            }
            let elements_per_slot = tensor.num_elements();
            let total_elements = elements_per_slot.saturating_mul(spec.max_batch_slots);
            let zeros = vec![0.0f32; total_elements];
            tensors.push(Qwen35RecurrentStateTensor {
                layer_index: tensor.layer_index,
                name: tensor.name.clone(),
                shape: tensor.shape.clone(),
                elements_per_slot,
                buffer: B::from_slice(&zeros),
            });
        }
        Ok(Self {
            request_id: spec.request_id.clone(),
            dtype: spec.dtype,
            device: spec.device.clone(),
            max_batch_slots: spec.max_batch_slots,
            tensors,
        })
    }

    pub fn total_elements(&self) -> usize {
        self.tensors
            .iter()
            .map(|tensor| tensor.elements_per_slot * self.max_batch_slots)
            .sum()
    }

    pub fn estimated_memory_bytes(&self) -> usize {
        self.total_elements() * self.dtype.size_bytes()
    }

    pub fn tensor(&self, layer_index: usize, name: &str) -> Option<&Qwen35RecurrentStateTensor<B>> {
        self.tensors
            .iter()
            .find(|tensor| tensor.layer_index == layer_index && tensor.name == name)
    }

    pub fn tensor_mut(
        &mut self,
        layer_index: usize,
        name: &str,
    ) -> Option<&mut Qwen35RecurrentStateTensor<B>> {
        self.tensors
            .iter_mut()
            .find(|tensor| tensor.layer_index == layer_index && tensor.name == name)
    }
}

impl<B: Backend> Qwen35RecurrentStateTensor<B> {
    pub fn slot_range(&self, slot: usize, max_batch_slots: usize) -> Result<Range<usize>> {
        if slot >= max_batch_slots {
            return Err(FerrumError::model(format!(
                "Qwen3.5 recurrent state slot {slot} exceeds max_batch_slots {max_batch_slots}"
            )));
        }
        let start = slot * self.elements_per_slot;
        Ok(start..start + self.elements_per_slot)
    }
}

impl<B: Backend> Qwen35ModelWeights<B> {
    pub fn load(
        config: Qwen35TextConfig,
        runtime_cfg: LlmRuntimeConfig,
        plan: &Qwen35ResolvedWeightPlan,
        loader: &dyn ferrum_quantization::WeightLoader<B>,
    ) -> Result<Self> {
        let planned = Qwen35WeightPlanLoader::<B>::new(plan, loader);
        let embed_tokens = planned.load_global_tensor("embed_tokens")?;
        let final_norm = planned.load_global_tensor("final_norm")?;
        let lm_head = if planned.has_global_tensor("lm_head") {
            planned.load_global_linear("lm_head")?
        } else {
            planned.load_global_linear("embed_tokens")?
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_plan in config
            .layer_plan()
            .map_err(ferrum_types::FerrumError::model)?
        {
            layers.push(Qwen35LayerWeights {
                layer_index: layer_plan.layer_index,
                input_layernorm: planned
                    .load_layer_tensor(layer_plan.layer_index, "input_layernorm")?,
                post_attention_layernorm: planned
                    .load_layer_tensor(layer_plan.layer_index, "post_attention_layernorm")?,
                attention: match layer_plan.attention {
                    Qwen35LayerType::LinearAttention => {
                        Qwen35AttentionWeights::Linear(Qwen35LinearAttentionWeights {
                            qkv_proj: planned
                                .load_layer_linear(layer_plan.layer_index, "linear_attn_qkv")?,
                            z_proj: planned
                                .load_layer_linear(layer_plan.layer_index, "linear_attn_z")?,
                            b_proj: planned
                                .load_layer_linear(layer_plan.layer_index, "linear_attn_b")?,
                            a_proj: planned
                                .load_layer_linear(layer_plan.layer_index, "linear_attn_a")?,
                            conv1d_weight: planned
                                .load_layer_tensor(layer_plan.layer_index, "linear_attn_conv")?,
                            a_log: planned
                                .load_layer_tensor(layer_plan.layer_index, "linear_attn_a_log")?,
                            dt_bias: planned
                                .load_layer_tensor(layer_plan.layer_index, "linear_attn_dt_bias")?,
                            norm_weight: planned
                                .load_layer_tensor(layer_plan.layer_index, "linear_attn_norm")?,
                            out_proj: planned
                                .load_layer_linear(layer_plan.layer_index, "linear_attn_out")?,
                        })
                    }
                    Qwen35LayerType::FullAttention => {
                        Qwen35AttentionWeights::Full(Qwen35FullAttentionWeights {
                            q_proj: planned
                                .load_layer_linear(layer_plan.layer_index, "self_attn_q")?,
                            k_proj: planned
                                .load_layer_linear(layer_plan.layer_index, "self_attn_k")?,
                            v_proj: planned
                                .load_layer_linear(layer_plan.layer_index, "self_attn_v")?,
                            o_proj: planned
                                .load_layer_linear(layer_plan.layer_index, "self_attn_o")?,
                            q_norm_weight: planned
                                .load_layer_tensor(layer_plan.layer_index, "self_attn_q_norm")?,
                            k_norm_weight: planned
                                .load_layer_tensor(layer_plan.layer_index, "self_attn_k_norm")?,
                        })
                    }
                },
                mlp: match layer_plan.mlp {
                    Qwen35MlpKind::Dense => Qwen35MlpWeights::Dense(Qwen35DenseMlpWeights {
                        gate_proj: planned.load_layer_linear(layer_plan.layer_index, "mlp_gate")?,
                        up_proj: planned.load_layer_linear(layer_plan.layer_index, "mlp_up")?,
                        down_proj: planned.load_layer_linear(layer_plan.layer_index, "mlp_down")?,
                    }),
                    Qwen35MlpKind::SparseMoeSharedExpert => {
                        Qwen35MlpWeights::SparseMoeSharedExpert(
                            Qwen35SparseMoeSharedExpertWeights {
                                router: planned
                                    .load_layer_linear(layer_plan.layer_index, "moe_router")?,
                                shared_expert_gate: planned.load_layer_tensor(
                                    layer_plan.layer_index,
                                    "moe_shared_expert_gate",
                                )?,
                                shared_expert_gate_proj: planned.load_layer_linear(
                                    layer_plan.layer_index,
                                    "moe_shared_expert_gate_proj",
                                )?,
                                shared_expert_up_proj: planned.load_layer_linear(
                                    layer_plan.layer_index,
                                    "moe_shared_expert_up_proj",
                                )?,
                                shared_expert_down_proj: planned.load_layer_linear(
                                    layer_plan.layer_index,
                                    "moe_shared_expert_down_proj",
                                )?,
                                fused_gate_up_proj: planned.load_layer_tensor(
                                    layer_plan.layer_index,
                                    "moe_fused_gate_up_proj",
                                )?,
                                fused_down_proj: planned.load_layer_tensor(
                                    layer_plan.layer_index,
                                    "moe_fused_down_proj",
                                )?,
                            },
                        )
                    }
                },
            });
        }

        Ok(Self {
            config,
            runtime_cfg,
            embed_tokens,
            final_norm,
            lm_head,
            layers,
        })
    }
}

impl<B: Backend> Qwen35AttentionWeights<B> {
    pub fn kind(&self) -> Qwen35LayerType {
        match self {
            Self::Linear(_) => Qwen35LayerType::LinearAttention,
            Self::Full(_) => Qwen35LayerType::FullAttention,
        }
    }
}

impl<B: Backend> Qwen35MlpWeights<B> {
    pub fn kind(&self) -> Qwen35MlpKind {
        match self {
            Self::Dense(_) => Qwen35MlpKind::Dense,
            Self::SparseMoeSharedExpert(_) => Qwen35MlpKind::SparseMoeSharedExpert,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use ferrum_kernels::backend::cpu::CpuBackend;
    use ferrum_quantization::{DenseLinear, QuantConfig, WeightLoader};
    use ferrum_types::{FerrumError, Result as FerrumResult};

    use super::*;
    use crate::qwen35_weights::Qwen35WeightInventory;
    use crate::{definition::ConfigManager, registry::Architecture};

    fn dense_config() -> Qwen35TextConfig {
        Qwen35TextConfig::from_hf_config_str(
            r#"{
              "model_type": "qwen3_5",
              "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "head_dim": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 32,
                "tie_word_embeddings": true
              }
            }"#,
        )
        .unwrap()
    }

    fn moe_config() -> Qwen35TextConfig {
        Qwen35TextConfig::from_hf_config_str(
            r#"{
              "model_type": "qwen3_5_moe",
              "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "head_dim": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
                "shared_expert_intermediate_size": 8,
                "tie_word_embeddings": false
              }
            }"#,
        )
        .unwrap()
    }

    struct RecordingLoader {
        tensors: HashMap<String, Vec<f32>>,
        linears: Mutex<Vec<String>>,
    }

    impl RecordingLoader {
        fn from_required_manifest(config: &Qwen35TextConfig) -> Self {
            let manifest = config.weight_manifest("model").unwrap();
            let tensors = manifest
                .global_tensors
                .iter()
                .chain(
                    manifest
                        .layers
                        .iter()
                        .flat_map(|layer| layer.tensors.iter()),
                )
                .filter(|tensor| tensor.required)
                .map(|tensor| (tensor.name.clone(), vec![1.0]))
                .collect();
            Self {
                tensors,
                linears: Mutex::new(Vec::new()),
            }
        }

        fn plan(&self, config: &Qwen35TextConfig) -> Qwen35ResolvedWeightPlan {
            Qwen35WeightInventory::from_names(self.tensors.keys().cloned())
                .detect_prefix_and_resolve(config)
                .unwrap()
        }

        fn linears(&self) -> Vec<String> {
            self.linears.lock().unwrap().clone()
        }
    }

    impl WeightLoader<CpuBackend> for RecordingLoader {
        fn load_tensor(&self, name: &str) -> FerrumResult<Vec<f32>> {
            self.tensors
                .get(name)
                .cloned()
                .ok_or_else(|| FerrumError::model(format!("missing tensor {name}")))
        }

        fn load_linear(
            &self,
            name: &str,
        ) -> FerrumResult<Box<dyn ferrum_quantization::Linear<CpuBackend>>> {
            let tensor_name = format!("{name}.weight");
            if !self.tensors.contains_key(&tensor_name) {
                return Err(FerrumError::model(format!(
                    "missing linear weight {tensor_name}"
                )));
            }
            self.linears.lock().unwrap().push(name.to_string());
            Ok(Box::new(DenseLinear::<CpuBackend>::from_rows(
                &[0.0, 0.0],
                1,
                2,
            )))
        }

        fn has_tensor(&self, name: &str) -> bool {
            self.tensors.contains_key(name)
        }

        fn quant_config(&self) -> Option<&QuantConfig> {
            None
        }
    }

    fn runtime_config(config: &Qwen35TextConfig) -> LlmRuntimeConfig {
        qwen35_runtime_config(config, 128, 64)
    }

    #[test]
    fn materializes_dense_qwen35_weights_from_plan() {
        let config = dense_config();
        let loader = RecordingLoader::from_required_manifest(&config);
        let plan = loader.plan(&config);

        let model = Qwen35ModelWeights::<CpuBackend>::load(
            config.clone(),
            runtime_config(&config),
            &plan,
            &loader,
        )
        .unwrap();

        assert_eq!(model.config.num_hidden_layers, 4);
        assert_eq!(model.runtime_cfg.hidden_size, 16);
        assert_eq!(model.runtime_cfg.vocab_size, 128);
        assert_eq!(model.layers.len(), 4);
        assert_eq!(
            model.layers[0].attention.kind(),
            Qwen35LayerType::LinearAttention
        );
        assert_eq!(model.layers[0].mlp.kind(), Qwen35MlpKind::Dense);
        assert_eq!(
            model.layers[3].attention.kind(),
            Qwen35LayerType::FullAttention
        );
        assert_eq!(model.layers[3].mlp.kind(), Qwen35MlpKind::Dense);
        assert!(loader.linears().contains(&"model.embed_tokens".to_string()));
        assert!(!loader.linears().contains(&"model.lm_head".to_string()));
        assert!(loader
            .linears()
            .contains(&"model.layers.0.linear_attn.in_proj_qkv".to_string()));
        assert!(loader
            .linears()
            .contains(&"model.layers.3.self_attn.q_proj".to_string()));
    }

    #[test]
    fn materializes_moe_qwen35_weights_from_plan() {
        let config = moe_config();
        let loader = RecordingLoader::from_required_manifest(&config);
        let plan = loader.plan(&config);

        let model = Qwen35ModelWeights::<CpuBackend>::load(
            config.clone(),
            runtime_config(&config),
            &plan,
            &loader,
        )
        .unwrap();

        assert_eq!(model.layers.len(), 4);
        assert_eq!(model.runtime_cfg.num_kv_heads, 1);
        assert_eq!(
            model.layers[0].mlp.kind(),
            Qwen35MlpKind::SparseMoeSharedExpert
        );
        assert_eq!(
            model.layers[3].mlp.kind(),
            Qwen35MlpKind::SparseMoeSharedExpert
        );
        assert!(loader.linears().contains(&"model.lm_head".to_string()));
        assert!(loader
            .linears()
            .contains(&"model.layers.0.mlp.gate".to_string()));
        assert!(loader
            .linears()
            .contains(&"model.layers.3.mlp.shared_expert.down_proj".to_string()));
        assert!(matches!(
            &model.layers[0].mlp,
            Qwen35MlpWeights::SparseMoeSharedExpert(weights)
                if weights.fused_gate_up_proj == vec![1.0]
        ));
    }

    #[test]
    fn derives_runtime_config_from_model_definition() {
        let raw = serde_json::json!({
          "model_type": "qwen3_5",
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "vocab_size": 32000,
          "max_position_embeddings": 4096,
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "head_dim": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 32,
            "tie_word_embeddings": true
          }
        });
        let mut manager = ConfigManager::new();
        let def = manager.parse_config_for_tests(&raw).unwrap();
        let runtime = qwen35_runtime_config_from_definition(&def).unwrap();

        assert_eq!(def.architecture, Architecture::Qwen35);
        assert_eq!(runtime.hidden_size, 16);
        assert_eq!(runtime.num_layers, 4);
        assert_eq!(runtime.num_kv_heads, 1);
        assert_eq!(runtime.head_dim, 4);
        assert_eq!(runtime.vocab_size, 32000);
        assert_eq!(runtime.max_seq_len, 4096);
    }

    #[test]
    fn allocates_recurrent_state_cache_from_spec() {
        let config = dense_config();
        let request_id = RequestId::new();
        let spec = config
            .to_recurrent_state_spec(request_id.clone(), DataType::BF16, Device::CPU, 2)
            .unwrap();

        let cache = Qwen35RecurrentStateCache::<CpuBackend>::from_spec(&spec).unwrap();
        let first = cache
            .tensor(0, crate::qwen35_config::QWEN35_DELTA_STATE_NAME)
            .unwrap();
        let second_slot = first.slot_range(1, cache.max_batch_slots).unwrap();

        assert_eq!(cache.request_id, request_id);
        assert_eq!(cache.dtype, DataType::BF16);
        assert_eq!(cache.device, Device::CPU);
        assert_eq!(cache.max_batch_slots, 2);
        assert_eq!(cache.tensors.len(), 3);
        assert_eq!(first.shape, vec![2, 4, 4]);
        assert_eq!(first.elements_per_slot, 32);
        assert_eq!(second_slot, 32..64);
        assert_eq!(cache.total_elements(), 3 * 2 * 32);
        assert_eq!(cache.estimated_memory_bytes(), 3 * 2 * 32 * 2);
    }

    #[test]
    fn rejects_recurrent_state_slot_overflow() {
        let config = dense_config();
        let spec = config
            .to_recurrent_state_spec(RequestId::new(), DataType::FP16, Device::CPU, 1)
            .unwrap();
        let cache = Qwen35RecurrentStateCache::<CpuBackend>::from_spec(&spec).unwrap();
        let first = cache
            .tensor(0, crate::qwen35_config::QWEN35_DELTA_STATE_NAME)
            .unwrap();
        let err = first
            .slot_range(1, cache.max_batch_slots)
            .expect_err("slot 1 should exceed one-slot recurrent state cache");

        assert!(err.to_string().contains("slot 1"), "{err}");
    }
}
