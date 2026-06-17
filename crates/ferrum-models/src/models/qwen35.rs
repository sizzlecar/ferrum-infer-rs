//! Qwen3.5 / Qwen3.6 typed model weights.
//!
//! This is the W3 materialization boundary: it turns the resolved semantic
//! weight plan into backend-native buffers and linears, but intentionally does
//! not implement product forward execution yet.

use ferrum_kernels::backend::Backend;
use ferrum_quantization::Linear;
use ferrum_types::Result;

use crate::{
    qwen35_config::{Qwen35LayerType, Qwen35MlpKind, Qwen35TextConfig},
    qwen35_weights::{Qwen35ResolvedWeightPlan, Qwen35WeightPlanLoader},
};

pub struct Qwen35ModelWeights<B: Backend> {
    pub config: Qwen35TextConfig,
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

impl<B: Backend> Qwen35ModelWeights<B> {
    pub fn load(
        config: Qwen35TextConfig,
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

    #[test]
    fn materializes_dense_qwen35_weights_from_plan() {
        let config = dense_config();
        let loader = RecordingLoader::from_required_manifest(&config);
        let plan = loader.plan(&config);

        let model = Qwen35ModelWeights::<CpuBackend>::load(config.clone(), &plan, &loader).unwrap();

        assert_eq!(model.config.num_hidden_layers, 4);
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

        let model = Qwen35ModelWeights::<CpuBackend>::load(config.clone(), &plan, &loader).unwrap();

        assert_eq!(model.layers.len(), 4);
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
}
