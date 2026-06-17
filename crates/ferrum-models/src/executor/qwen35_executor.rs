//! Qwen3.5 / Qwen3.6 W3 executor boundary.
//!
//! This skeleton is intentionally not registered as a product executor yet. It
//! exposes the recurrent-state allocation contract that the real W3 executor
//! will use. Product registration remains disabled until the full W3 run/serve
//! path is wired and gated.

use async_trait::async_trait;
use candle_core::{Device as CandleDevice, Tensor};
use ferrum_interfaces::{
    model_executor::{
        DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorMemoryUsage, ExecutorState,
        ExecutorStatus, MemoryRequirements, PrefillInput, PrefillOutput,
    },
    ModelExecutor, RecurrentStateSpec,
};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_quantization::{NativeSafetensorsLoader, WeightLoader};
use ferrum_types::{DataType, Device, FerrumError, ModelInfo, RequestId, Result, TokenId};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::{
    definition::ModelDefinition,
    executor::common::{self, GenericKvCacheHandle},
    models::qwen35::{
        qwen35_dense_reference_model_forward_cpu, Qwen35DenseFullAttentionLayerShape,
        Qwen35DenseLinearAttentionLayerShape, Qwen35DenseReferenceFullLayer,
        Qwen35DenseReferenceLayer, Qwen35DenseReferenceLinearLayer, Qwen35DenseReferenceModel,
        Qwen35DenseReferenceModelOutput, Qwen35FullAttentionShape, Qwen35LinearAttentionShape,
    },
    qwen35_config::{Qwen35LayerType, Qwen35TextConfig},
    qwen35_weights::{
        Qwen35ResolvedWeightPlan, Qwen35ResolvedWeightSpec, Qwen35WeightInventory,
        Qwen35WeightValidation,
    },
};

const UNSUPPORTED_EXECUTION_MESSAGE: &str = "Qwen3.5/Qwen3.6 W3 executor exposes recurrent-state \
spec only; prefill/decode are not wired for product execution yet";

static QWEN35_REFERENCE_CACHE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct Qwen35DenseReferenceRuntime {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub eps: f32,
    pub embed_tokens: Vec<f32>,
    pub final_norm_weight: Vec<f32>,
    pub lm_head_weight: Vec<f32>,
    pub layers: Vec<Qwen35DenseReferenceRuntimeLayer>,
}

#[derive(Debug, Clone)]
pub enum Qwen35DenseReferenceRuntimeLayer {
    Linear(Qwen35DenseReferenceRuntimeLinearLayer),
    Full(Qwen35DenseReferenceRuntimeFullLayer),
}

#[derive(Debug, Clone)]
pub struct Qwen35DenseReferenceRuntimeLinearLayer {
    pub intermediate_size: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
    pub input_norm_weight: Vec<f32>,
    pub qkv_weight: Vec<f32>,
    pub z_weight: Vec<f32>,
    pub b_weight: Vec<f32>,
    pub a_weight: Vec<f32>,
    pub conv1d_weight: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm_weight: Vec<f32>,
    pub out_proj_weight: Vec<f32>,
    pub post_attention_norm_weight: Vec<f32>,
    pub gate_proj_weight: Vec<f32>,
    pub up_proj_weight: Vec<f32>,
    pub down_proj_weight: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Qwen35DenseReferenceRuntimeFullLayer {
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub position_offset: usize,
    pub rope_theta: f32,
    pub input_norm_weight: Vec<f32>,
    pub q_weight: Vec<f32>,
    pub k_weight: Vec<f32>,
    pub v_weight: Vec<f32>,
    pub q_norm_weight: Vec<f32>,
    pub k_norm_weight: Vec<f32>,
    pub o_weight: Vec<f32>,
    pub post_attention_norm_weight: Vec<f32>,
    pub gate_proj_weight: Vec<f32>,
    pub up_proj_weight: Vec<f32>,
    pub down_proj_weight: Vec<f32>,
}

impl Qwen35DenseReferenceRuntime {
    pub fn from_cpu_weight_plan(
        config: &Qwen35TextConfig,
        vocab_size: usize,
        eps: f32,
        rope_theta: f32,
        plan: &Qwen35ResolvedWeightPlan,
        loader: &dyn WeightLoader<CpuBackend>,
    ) -> Result<Self> {
        if config.is_moe() {
            return Err(FerrumError::unsupported(
                "Qwen3.5 dense reference runtime cannot materialize sparse MoE weights",
            ));
        }
        let intermediate_size = config.dense_intermediate_size.ok_or_else(|| {
            FerrumError::model("Qwen3.5 dense reference runtime missing intermediate_size")
        })?;
        let embed_tokens = load_global_cpu_tensor(plan, loader, "embed_tokens")?;
        let final_norm_weight = load_global_cpu_tensor(plan, loader, "final_norm")?;
        let lm_head_weight = match optional_global_cpu_tensor(plan, loader, "lm_head")? {
            Some(weight) => weight,
            None => embed_tokens.clone(),
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (layer_index, kind) in config.layer_types.iter().copied().enumerate() {
            let input_norm_weight =
                load_layer_cpu_tensor(plan, loader, layer_index, "input_layernorm")?;
            let post_attention_norm_weight =
                load_layer_cpu_tensor(plan, loader, layer_index, "post_attention_layernorm")?;
            let gate_proj_weight = load_layer_cpu_tensor(plan, loader, layer_index, "mlp_gate")?;
            let up_proj_weight = load_layer_cpu_tensor(plan, loader, layer_index, "mlp_up")?;
            let down_proj_weight = load_layer_cpu_tensor(plan, loader, layer_index, "mlp_down")?;

            let layer = match kind {
                Qwen35LayerType::LinearAttention => Qwen35DenseReferenceRuntimeLayer::Linear(
                    Qwen35DenseReferenceRuntimeLinearLayer {
                        intermediate_size,
                        key_heads: config.linear_attention.num_key_heads,
                        value_heads: config.linear_attention.num_value_heads,
                        key_dim: config.linear_attention.key_head_dim,
                        value_dim: config.linear_attention.value_head_dim,
                        conv_kernel: config.linear_attention.conv_kernel_dim,
                        input_norm_weight,
                        qkv_weight: load_layer_cpu_tensor(
                            plan,
                            loader,
                            layer_index,
                            "linear_attn_qkv",
                        )?,
                        z_weight: load_layer_cpu_tensor(
                            plan,
                            loader,
                            layer_index,
                            "linear_attn_z",
                        )?,
                        b_weight: load_layer_cpu_tensor(
                            plan,
                            loader,
                            layer_index,
                            "linear_attn_b",
                        )?,
                        a_weight: load_layer_cpu_tensor(
                            plan,
                            loader,
                            layer_index,
                            "linear_attn_a",
                        )?,
                        conv1d_weight: load_layer_cpu_tensor(
                            plan,
                            loader,
                            layer_index,
                            "linear_attn_conv",
                        )?,
                        a_log: load_layer_cpu_tensor(
                            plan,
                            loader,
                            layer_index,
                            "linear_attn_a_log",
                        )?,
                        dt_bias: load_layer_cpu_tensor(
                            plan,
                            loader,
                            layer_index,
                            "linear_attn_dt_bias",
                        )?,
                        norm_weight: load_layer_cpu_tensor(
                            plan,
                            loader,
                            layer_index,
                            "linear_attn_norm",
                        )?,
                        out_proj_weight: load_layer_cpu_tensor(
                            plan,
                            loader,
                            layer_index,
                            "linear_attn_out",
                        )?,
                        post_attention_norm_weight,
                        gate_proj_weight,
                        up_proj_weight,
                        down_proj_weight,
                    },
                ),
                Qwen35LayerType::FullAttention => {
                    Qwen35DenseReferenceRuntimeLayer::Full(Qwen35DenseReferenceRuntimeFullLayer {
                        intermediate_size,
                        num_heads: config.num_attention_heads,
                        num_kv_heads: config.num_key_value_heads,
                        head_dim: config.head_dim,
                        position_offset: 0,
                        rope_theta,
                        input_norm_weight,
                        q_weight: load_layer_cpu_tensor(plan, loader, layer_index, "self_attn_q")?,
                        k_weight: load_layer_cpu_tensor(plan, loader, layer_index, "self_attn_k")?,
                        v_weight: load_layer_cpu_tensor(plan, loader, layer_index, "self_attn_v")?,
                        q_norm_weight: load_layer_cpu_tensor(
                            plan,
                            loader,
                            layer_index,
                            "self_attn_q_norm",
                        )?,
                        k_norm_weight: load_layer_cpu_tensor(
                            plan,
                            loader,
                            layer_index,
                            "self_attn_k_norm",
                        )?,
                        o_weight: load_layer_cpu_tensor(plan, loader, layer_index, "self_attn_o")?,
                        post_attention_norm_weight,
                        gate_proj_weight,
                        up_proj_weight,
                        down_proj_weight,
                    })
                }
            };
            layers.push(layer);
        }
        let runtime = Self {
            vocab_size,
            hidden_size: config.hidden_size,
            eps,
            embed_tokens,
            final_norm_weight,
            lm_head_weight,
            layers,
        };
        runtime.validate_for_config(config, vocab_size)?;
        Ok(runtime)
    }

    pub fn validate_for_config(&self, config: &Qwen35TextConfig, vocab_size: usize) -> Result<()> {
        if config.is_moe() {
            return Err(FerrumError::unsupported(
                "Qwen3.5 dense reference runtime does not support sparse MoE layers",
            ));
        }
        if self.vocab_size != vocab_size {
            return Err(FerrumError::model(format!(
                "Qwen3.5 dense reference vocab_size {} does not match model vocab_size {vocab_size}",
                self.vocab_size
            )));
        }
        if self.hidden_size != config.hidden_size {
            return Err(FerrumError::model(format!(
                "Qwen3.5 dense reference hidden_size {} does not match config hidden_size {}",
                self.hidden_size, config.hidden_size
            )));
        }
        if self.layers.len() != config.num_hidden_layers {
            return Err(FerrumError::model(format!(
                "Qwen3.5 dense reference layer count {} does not match config num_hidden_layers {}",
                self.layers.len(),
                config.num_hidden_layers
            )));
        }
        for (idx, layer) in self.layers.iter().enumerate() {
            let expected = config.layer_types[idx];
            let actual = layer.attention_kind();
            if actual != expected {
                return Err(FerrumError::model(format!(
                    "Qwen3.5 dense reference layer {idx} kind {actual:?} does not match config {expected:?}",
                )));
            }
        }
        Ok(())
    }

    pub fn forward(&self, input_ids: &[usize]) -> Result<Qwen35DenseReferenceModelOutput> {
        let tokens = input_ids.len();
        let layers = self
            .layers
            .iter()
            .map(|layer| layer.as_reference_layer(tokens, self.hidden_size))
            .collect::<Vec<_>>();
        qwen35_dense_reference_model_forward_cpu(
            Qwen35DenseReferenceModel {
                vocab_size: self.vocab_size,
                hidden_size: self.hidden_size,
                eps: self.eps,
                embed_tokens: &self.embed_tokens,
                final_norm_weight: &self.final_norm_weight,
                lm_head_weight: &self.lm_head_weight,
                layers: &layers,
            },
            input_ids,
        )
    }
}

fn load_global_cpu_tensor(
    plan: &Qwen35ResolvedWeightPlan,
    loader: &dyn WeightLoader<CpuBackend>,
    role: &str,
) -> Result<Vec<f32>> {
    let spec = required_global_spec(plan, role)?;
    loader.load_tensor(&spec.name)
}

fn optional_global_cpu_tensor(
    plan: &Qwen35ResolvedWeightPlan,
    loader: &dyn WeightLoader<CpuBackend>,
    role: &str,
) -> Result<Option<Vec<f32>>> {
    let Some(spec) = plan.global_tensor(role) else {
        return Ok(None);
    };
    if !spec.present {
        return Ok(None);
    }
    loader.load_tensor(&spec.name).map(Some)
}

fn load_layer_cpu_tensor(
    plan: &Qwen35ResolvedWeightPlan,
    loader: &dyn WeightLoader<CpuBackend>,
    layer_index: usize,
    role: &str,
) -> Result<Vec<f32>> {
    let spec = plan.layer_tensor(layer_index, role).ok_or_else(|| {
        FerrumError::model(format!(
            "Qwen3.5 resolved weight plan has no present layer tensor role {role:?} at layer {layer_index}"
        ))
    })?;
    loader.load_tensor(&spec.name)
}

fn required_global_spec<'a>(
    plan: &'a Qwen35ResolvedWeightPlan,
    role: &str,
) -> Result<&'a Qwen35ResolvedWeightSpec> {
    let spec = plan.global_tensor(role).ok_or_else(|| {
        FerrumError::model(format!(
            "Qwen3.5 resolved weight plan has no global tensor role {role:?}"
        ))
    })?;
    if !spec.present {
        return Err(FerrumError::model(format!(
            "Qwen3.5 global tensor role {role:?} is absent: {}",
            spec.name
        )));
    }
    Ok(spec)
}

impl Qwen35DenseReferenceRuntimeLayer {
    fn attention_kind(&self) -> Qwen35LayerType {
        match self {
            Self::Linear(_) => Qwen35LayerType::LinearAttention,
            Self::Full(_) => Qwen35LayerType::FullAttention,
        }
    }

    fn as_reference_layer(
        &self,
        tokens: usize,
        hidden_size: usize,
    ) -> Qwen35DenseReferenceLayer<'_> {
        match self {
            Self::Linear(layer) => {
                Qwen35DenseReferenceLayer::Linear(Qwen35DenseReferenceLinearLayer {
                    shape: Qwen35DenseLinearAttentionLayerShape {
                        tokens,
                        hidden_size,
                        intermediate_size: layer.intermediate_size,
                        attention: Qwen35LinearAttentionShape {
                            tokens,
                            key_heads: layer.key_heads,
                            value_heads: layer.value_heads,
                            key_dim: layer.key_dim,
                            value_dim: layer.value_dim,
                            conv_kernel: layer.conv_kernel,
                        },
                    },
                    input_norm_weight: &layer.input_norm_weight,
                    qkv_weight: &layer.qkv_weight,
                    z_weight: &layer.z_weight,
                    b_weight: &layer.b_weight,
                    a_weight: &layer.a_weight,
                    conv1d_weight: &layer.conv1d_weight,
                    a_log: &layer.a_log,
                    dt_bias: &layer.dt_bias,
                    norm_weight: &layer.norm_weight,
                    out_proj_weight: &layer.out_proj_weight,
                    post_attention_norm_weight: &layer.post_attention_norm_weight,
                    gate_proj_weight: &layer.gate_proj_weight,
                    up_proj_weight: &layer.up_proj_weight,
                    down_proj_weight: &layer.down_proj_weight,
                })
            }
            Self::Full(layer) => Qwen35DenseReferenceLayer::Full(Qwen35DenseReferenceFullLayer {
                shape: Qwen35DenseFullAttentionLayerShape {
                    tokens,
                    hidden_size,
                    intermediate_size: layer.intermediate_size,
                    attention: Qwen35FullAttentionShape {
                        tokens,
                        num_heads: layer.num_heads,
                        num_kv_heads: layer.num_kv_heads,
                        head_dim: layer.head_dim,
                        position_offset: layer.position_offset,
                        rope_theta: layer.rope_theta,
                    },
                },
                input_norm_weight: &layer.input_norm_weight,
                q_weight: &layer.q_weight,
                k_weight: &layer.k_weight,
                v_weight: &layer.v_weight,
                q_norm_weight: &layer.q_norm_weight,
                k_norm_weight: &layer.k_norm_weight,
                o_weight: &layer.o_weight,
                post_attention_norm_weight: &layer.post_attention_norm_weight,
                gate_proj_weight: &layer.gate_proj_weight,
                up_proj_weight: &layer.up_proj_weight,
                down_proj_weight: &layer.down_proj_weight,
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen35W3Executor {
    info: ModelInfo,
    config: Qwen35TextConfig,
    dtype: DataType,
    device: Device,
    weight_validation: Option<Qwen35WeightValidation>,
    weight_plan: Option<Qwen35ResolvedWeightPlan>,
    dense_reference_runtime: Option<Arc<Qwen35DenseReferenceRuntime>>,
}

impl Qwen35W3Executor {
    pub fn from_definition(
        model_id: impl Into<String>,
        def: &ModelDefinition,
        dtype: DataType,
        device: Device,
    ) -> Result<Self> {
        let config = Qwen35TextConfig::from_model_definition(def)
            .map_err(|err| FerrumError::model(format!("invalid Qwen3.5/Qwen3.6 config: {err}")))?;
        let mut info = def.to_model_info(model_id);
        info.dtype = dtype;
        info.device = device.clone();
        Ok(Self {
            info,
            config,
            dtype,
            device,
            weight_validation: None,
            weight_plan: None,
            dense_reference_runtime: None,
        })
    }

    pub fn from_definition_with_weight_preflight(
        model_id: impl Into<String>,
        def: &ModelDefinition,
        model_dir: &Path,
        dtype: DataType,
        device: Device,
    ) -> Result<Self> {
        let mut executor = Self::from_definition(model_id, def, dtype, device)?;
        let inventory = Qwen35WeightInventory::from_safetensors_dir(model_dir)
            .map_err(|err| FerrumError::model(format!("Qwen3.5 weight inventory failed: {err}")))?;
        let weight_plan = inventory
            .detect_prefix_and_resolve(&executor.config)
            .map_err(|err| FerrumError::model(format!("Qwen3.5 weight preflight failed: {err}")))?;
        let validation = weight_plan.validation();
        executor.weight_validation = Some(validation);
        executor.weight_plan = Some(weight_plan);
        Ok(executor)
    }

    pub fn from_definition_with_dense_reference_cpu_safetensors(
        model_id: impl Into<String>,
        def: &ModelDefinition,
        model_dir: &Path,
        dtype: DataType,
        device: Device,
    ) -> Result<Self> {
        if dtype != DataType::FP32 || device != Device::CPU {
            return Err(FerrumError::unsupported(
                "Qwen3.5 dense reference safetensors executor requires FP32 CPU",
            ));
        }
        let mut executor = Self::from_definition(model_id, def, dtype, device)?;
        let inventory = Qwen35WeightInventory::from_safetensors_dir(model_dir)
            .map_err(|err| FerrumError::model(format!("Qwen3.5 weight inventory failed: {err}")))?;
        let weight_plan = inventory
            .detect_prefix_and_resolve(&executor.config)
            .map_err(|err| FerrumError::model(format!("Qwen3.5 weight preflight failed: {err}")))?;
        let validation = weight_plan.validation();
        let loader = NativeSafetensorsLoader::<CpuBackend>::open(model_dir)?;
        let runtime = Qwen35DenseReferenceRuntime::from_cpu_weight_plan(
            &executor.config,
            def.vocab_size,
            def.norm_eps as f32,
            def.rope_theta.unwrap_or(10_000.0) as f32,
            &weight_plan,
            &loader,
        )?;
        executor.weight_validation = Some(validation);
        executor.weight_plan = Some(weight_plan);
        executor.with_dense_reference_runtime(runtime)
    }

    pub fn qwen35_config(&self) -> &Qwen35TextConfig {
        &self.config
    }

    pub fn weight_validation(&self) -> Option<&Qwen35WeightValidation> {
        self.weight_validation.as_ref()
    }

    pub fn weight_plan(&self) -> Option<&Qwen35ResolvedWeightPlan> {
        self.weight_plan.as_ref()
    }

    pub fn with_dense_reference_runtime(
        mut self,
        runtime: Qwen35DenseReferenceRuntime,
    ) -> Result<Self> {
        runtime.validate_for_config(&self.config, self.info.vocab_size)?;
        self.dense_reference_runtime = Some(Arc::new(runtime));
        Ok(self)
    }

    fn unsupported_execution() -> FerrumError {
        FerrumError::unsupported(UNSUPPORTED_EXECUTION_MESSAGE)
    }

    fn next_reference_cache_id() -> String {
        let id = QWEN35_REFERENCE_CACHE_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("qwen35-reference-prefill-{id}")
    }
}

#[async_trait]
impl ModelExecutor for Qwen35W3Executor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        _input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        self.config
            .to_recurrent_state_spec(request_id.clone(), self.dtype, self.device.clone(), 1)
            .map(Some)
            .map_err(|err| FerrumError::model(format!("invalid Qwen3.5 recurrent spec: {err}")))
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        let runtime = self
            .dense_reference_runtime
            .as_ref()
            .ok_or_else(Self::unsupported_execution)?;
        let tokens = common::tensor_to_tokens(&input.input_ids)?;
        let input_ids = tokens
            .iter()
            .map(|token| *token as usize)
            .collect::<Vec<_>>();
        let output = runtime.forward(&input_ids)?;
        let vocab_size = runtime.vocab_size;
        let last_token_start = (input_ids.len() - 1) * vocab_size;
        let last_logits = output.logits[last_token_start..last_token_start + vocab_size].to_vec();
        let logits_tensor = Tensor::new(last_logits.as_slice(), &CandleDevice::Cpu)
            .map_err(|err| FerrumError::model(format!("Qwen3.5 logits tensor failed: {err}")))?
            .reshape((1, 1, vocab_size))
            .map_err(|err| FerrumError::model(format!("Qwen3.5 logits reshape failed: {err}")))?;
        let kv_cache = Arc::new(GenericKvCacheHandle::new(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
            self.config.head_dim,
            CandleDevice::Cpu,
            input_ids.len(),
            Self::next_reference_cache_id(),
        ));
        Ok(PrefillOutput::new(
            common::wrap_tensor(logits_tensor),
            kv_cache,
        ))
    }

    async fn decode(&self, _input: &DecodeInput) -> Result<DecodeOutput> {
        Err(Self::unsupported_execution())
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        let recurrent_state_bytes = self
            .config
            .recurrent_state_elements_per_slot()
            .unwrap_or(0)
            .saturating_mul(self.dtype.size_bytes());
        ExecutorCapabilities {
            max_batch_size: 1,
            max_sequence_length: self.info.max_sequence_length,
            attention_mechanisms: Vec::new(),
            supports_dynamic_batching: false,
            supports_continuous_batching: false,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![self.dtype],
            supported_devices: vec![self.device.clone()],
            memory_requirements: MemoryRequirements {
                parameter_memory: self.info.num_parameters.saturating_mul(2),
                activation_memory_per_token: self.info.hidden_size,
                kv_cache_memory_per_token: 0,
                overhead_memory: recurrent_state_bytes as u64,
            },
        }
    }

    fn status(&self) -> ExecutorStatus {
        let ready = self.dense_reference_runtime.is_some();
        ExecutorStatus {
            state: if ready {
                ExecutorState::Ready
            } else {
                ExecutorState::Error
            },
            is_ready: ready,
            current_batch_size: 0,
            prefill_operations: 0,
            decode_operations: 0,
            avg_prefill_time_ms: 0.0,
            avg_decode_time_ms: 0.0,
            memory_usage: ExecutorMemoryUsage {
                allocated_bytes: 0,
                used_bytes: 0,
                peak_bytes: 0,
                utilization_percent: 0.0,
            },
            last_operation: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{definition::ModelDefinition, registry::Architecture};
    use ferrum_interfaces::{
        model_executor::{DecodeInput, ExecutorState, PrefillInput},
        ModelExecutor,
    };
    use ferrum_quantization::QuantConfig;
    use ferrum_testkit::MockTensor;
    use std::collections::HashMap;
    use std::path::Path;

    fn toy_dense_definition() -> ModelDefinition {
        let mut def = ModelDefinition::default();
        def.architecture = Architecture::Qwen35;
        def.hidden_size = 2;
        def.intermediate_size = 2;
        def.vocab_size = 3;
        def.num_hidden_layers = 2;
        def.num_attention_heads = 1;
        def.num_key_value_heads = Some(1);
        def.max_position_embeddings = 16;
        def.rope_theta = Some(10_000.0);
        def.norm_eps = 1e-6;
        def.extra_params = serde_json::json!({
            "model_type": "qwen3_5_text",
            "hidden_size": 2,
            "intermediate_size": 2,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 1,
            "linear_value_head_dim": 1,
            "linear_conv_kernel_dim": 1,
            "head_dim": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "tie_word_embeddings": false
        });
        def
    }

    fn toy_dense_runtime() -> Qwen35DenseReferenceRuntime {
        Qwen35DenseReferenceRuntime {
            vocab_size: 3,
            hidden_size: 2,
            eps: 1e-6,
            embed_tokens: vec![
                1.0, 0.0, //
                0.0, 1.0, //
                1.0, 1.0,
            ],
            final_norm_weight: vec![0.0, 0.0],
            lm_head_weight: vec![
                1.0, 0.0, //
                0.0, 1.0, //
                1.0, 1.0,
            ],
            layers: vec![
                Qwen35DenseReferenceRuntimeLayer::Linear(Qwen35DenseReferenceRuntimeLinearLayer {
                    intermediate_size: 2,
                    key_heads: 1,
                    value_heads: 1,
                    key_dim: 1,
                    value_dim: 1,
                    conv_kernel: 1,
                    input_norm_weight: vec![0.0, 0.0],
                    qkv_weight: vec![
                        1.0, 0.0, //
                        0.0, 1.0, //
                        1.0, 1.0,
                    ],
                    z_weight: vec![1.0, -1.0],
                    b_weight: vec![0.5, 0.25],
                    a_weight: vec![-0.25, 0.75],
                    conv1d_weight: vec![1.0, 1.0, 1.0],
                    a_log: vec![0.0],
                    dt_bias: vec![0.0],
                    norm_weight: vec![1.0],
                    out_proj_weight: vec![1.0, -0.5],
                    post_attention_norm_weight: vec![0.0, 0.0],
                    gate_proj_weight: vec![
                        0.2, 0.1, //
                        -0.1, 0.3,
                    ],
                    up_proj_weight: vec![
                        0.4, -0.2, //
                        0.3, 0.5,
                    ],
                    down_proj_weight: vec![
                        1.0, 0.0, //
                        0.0, 1.0,
                    ],
                }),
                Qwen35DenseReferenceRuntimeLayer::Full(Qwen35DenseReferenceRuntimeFullLayer {
                    intermediate_size: 2,
                    num_heads: 1,
                    num_kv_heads: 1,
                    head_dim: 2,
                    position_offset: 0,
                    rope_theta: 10_000.0,
                    input_norm_weight: vec![0.0, 0.0],
                    q_weight: vec![
                        1.0, 0.0, //
                        0.0, 1.0,
                    ],
                    k_weight: vec![
                        0.5, 0.0, //
                        0.0, 0.5,
                    ],
                    v_weight: vec![
                        1.0, 1.0, //
                        -0.5, 0.5,
                    ],
                    q_norm_weight: vec![1.0, 1.0],
                    k_norm_weight: vec![1.0, 1.0],
                    o_weight: vec![
                        1.0, 0.0, //
                        0.0, 1.0,
                    ],
                    post_attention_norm_weight: vec![0.0, 0.0],
                    gate_proj_weight: vec![
                        -0.2, 0.2, //
                        0.1, 0.3,
                    ],
                    up_proj_weight: vec![
                        0.25, 0.5, //
                        -0.3, 0.4,
                    ],
                    down_proj_weight: vec![
                        0.5, 0.25, //
                        -0.2, 0.75,
                    ],
                }),
            ],
        }
    }

    fn toy_dense_weight_map() -> HashMap<String, Vec<f32>> {
        let runtime = toy_dense_runtime();
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            runtime.embed_tokens.clone(),
        );
        tensors.insert(
            "model.norm.weight".to_string(),
            runtime.final_norm_weight.clone(),
        );
        tensors.insert(
            "model.lm_head.weight".to_string(),
            runtime.lm_head_weight.clone(),
        );
        let Qwen35DenseReferenceRuntimeLayer::Linear(linear) = &runtime.layers[0] else {
            panic!("toy layer 0 must be linear attention");
        };
        tensors.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            linear.input_norm_weight.clone(),
        );
        tensors.insert(
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            linear.post_attention_norm_weight.clone(),
        );
        tensors.insert(
            "model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
            linear.qkv_weight.clone(),
        );
        tensors.insert(
            "model.layers.0.linear_attn.in_proj_z.weight".to_string(),
            linear.z_weight.clone(),
        );
        tensors.insert(
            "model.layers.0.linear_attn.in_proj_b.weight".to_string(),
            linear.b_weight.clone(),
        );
        tensors.insert(
            "model.layers.0.linear_attn.in_proj_a.weight".to_string(),
            linear.a_weight.clone(),
        );
        tensors.insert(
            "model.layers.0.linear_attn.conv1d.weight".to_string(),
            linear.conv1d_weight.clone(),
        );
        tensors.insert(
            "model.layers.0.linear_attn.A_log".to_string(),
            linear.a_log.clone(),
        );
        tensors.insert(
            "model.layers.0.linear_attn.dt_bias".to_string(),
            linear.dt_bias.clone(),
        );
        tensors.insert(
            "model.layers.0.linear_attn.norm.weight".to_string(),
            linear.norm_weight.clone(),
        );
        tensors.insert(
            "model.layers.0.linear_attn.out_proj.weight".to_string(),
            linear.out_proj_weight.clone(),
        );
        tensors.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            linear.gate_proj_weight.clone(),
        );
        tensors.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            linear.up_proj_weight.clone(),
        );
        tensors.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            linear.down_proj_weight.clone(),
        );

        let Qwen35DenseReferenceRuntimeLayer::Full(full) = &runtime.layers[1] else {
            panic!("toy layer 1 must be full attention");
        };
        tensors.insert(
            "model.layers.1.input_layernorm.weight".to_string(),
            full.input_norm_weight.clone(),
        );
        tensors.insert(
            "model.layers.1.post_attention_layernorm.weight".to_string(),
            full.post_attention_norm_weight.clone(),
        );
        tensors.insert(
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            full.q_weight.clone(),
        );
        tensors.insert(
            "model.layers.1.self_attn.k_proj.weight".to_string(),
            full.k_weight.clone(),
        );
        tensors.insert(
            "model.layers.1.self_attn.v_proj.weight".to_string(),
            full.v_weight.clone(),
        );
        tensors.insert(
            "model.layers.1.self_attn.o_proj.weight".to_string(),
            full.o_weight.clone(),
        );
        tensors.insert(
            "model.layers.1.self_attn.q_norm.weight".to_string(),
            full.q_norm_weight.clone(),
        );
        tensors.insert(
            "model.layers.1.self_attn.k_norm.weight".to_string(),
            full.k_norm_weight.clone(),
        );
        tensors.insert(
            "model.layers.1.mlp.gate_proj.weight".to_string(),
            full.gate_proj_weight.clone(),
        );
        tensors.insert(
            "model.layers.1.mlp.up_proj.weight".to_string(),
            full.up_proj_weight.clone(),
        );
        tensors.insert(
            "model.layers.1.mlp.down_proj.weight".to_string(),
            full.down_proj_weight.clone(),
        );
        tensors
    }

    struct MapWeightLoader {
        tensors: HashMap<String, Vec<f32>>,
    }

    impl WeightLoader<CpuBackend> for MapWeightLoader {
        fn load_tensor(&self, name: &str) -> Result<Vec<f32>> {
            self.tensors
                .get(name)
                .cloned()
                .ok_or_else(|| FerrumError::model(format!("missing tensor {name}")))
        }

        fn load_linear(
            &self,
            name: &str,
        ) -> Result<Box<dyn ferrum_quantization::Linear<CpuBackend>>> {
            Err(FerrumError::model(format!(
                "unexpected linear load in dense reference materializer: {name}"
            )))
        }

        fn has_tensor(&self, name: &str) -> bool {
            self.tensors.contains_key(name)
        }

        fn quant_config(&self) -> Option<&QuantConfig> {
            None
        }
    }

    fn write_toy_safetensors(dir: &Path, tensors: HashMap<String, Vec<f32>>) {
        use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

        let views = tensors
            .into_iter()
            .map(|(name, values)| {
                let bytes = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let bytes: &'static [u8] = Box::leak(bytes);
                (
                    name,
                    TensorView::new(Dtype::F32, vec![values.len()], bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(
            views,
            &None::<HashMap<String, String>>,
            &dir.join("model.safetensors"),
        )
        .unwrap();
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn qwen35_w3_executor_without_reference_runtime_keeps_prefill_unsupported() {
        let def = toy_dense_definition();
        let executor = Qwen35W3Executor::from_definition(
            "qwen35-reference",
            &def,
            DataType::FP32,
            Device::CPU,
        )
        .unwrap();
        let input = PrefillInput::new(MockTensor::from_u32(&[0, 1], &[2]).into_ref());

        let err = tokio_test::block_on(executor.prefill(&input))
            .expect_err("default W3 executor must not pretend product execution is wired");

        assert!(err.to_string().contains("not wired"), "{err}");
        let status = executor.status();
        assert_eq!(status.state, ExecutorState::Error);
        assert!(!status.is_ready);
    }

    #[test]
    fn qwen35_w3_reference_prefill_returns_last_token_logits_and_kv_handle() {
        let def = toy_dense_definition();
        let runtime = toy_dense_runtime();
        let expected = runtime.forward(&[0, 1]).unwrap();
        let executor = Qwen35W3Executor::from_definition(
            "qwen35-reference",
            &def,
            DataType::FP32,
            Device::CPU,
        )
        .unwrap()
        .with_dense_reference_runtime(runtime)
        .unwrap();
        let input = PrefillInput::new(MockTensor::from_u32(&[0, 1], &[2]).into_ref());

        let output = tokio_test::block_on(executor.prefill(&input)).unwrap();

        assert_eq!(output.logits.shape(), &[1, 1, 3]);
        assert_close(&output.logits.to_vec_f32().unwrap(), &expected.logits[3..6]);
        assert_eq!(output.kv_cache.block_table().sequence_length, 2);
        assert_eq!(output.kv_cache.num_layers(), 2);
        assert_eq!(output.kv_cache.num_heads(), 1);
        assert_eq!(output.kv_cache.head_dim(), 2);
        assert!(output.recurrent_state.is_none());
        let status = executor.status();
        assert_eq!(status.state, ExecutorState::Ready);
        assert!(status.is_ready);
    }

    #[test]
    fn qwen35_dense_reference_runtime_materializes_from_cpu_weight_plan() {
        let def = toy_dense_definition();
        let config = Qwen35TextConfig::from_model_definition(&def).unwrap();
        let loader = MapWeightLoader {
            tensors: toy_dense_weight_map(),
        };
        let plan = Qwen35WeightInventory::from_names(loader.tensors.keys().cloned())
            .detect_prefix_and_resolve(&config)
            .unwrap();

        let runtime = Qwen35DenseReferenceRuntime::from_cpu_weight_plan(
            &config,
            def.vocab_size,
            def.norm_eps as f32,
            def.rope_theta.unwrap() as f32,
            &plan,
            &loader,
        )
        .unwrap();
        let expected = toy_dense_runtime().forward(&[0, 1]).unwrap();
        let actual = runtime.forward(&[0, 1]).unwrap();

        assert_close(&actual.logits, &expected.logits);
        assert_close(
            &actual.linear_recurrent_states[0],
            &expected.linear_recurrent_states[0],
        );
    }

    #[test]
    fn qwen35_w3_reference_executor_prefills_from_dense_safetensors() {
        let def = toy_dense_definition();
        let tmp = tempfile::TempDir::new().unwrap();
        write_toy_safetensors(tmp.path(), toy_dense_weight_map());

        let executor = Qwen35W3Executor::from_definition_with_dense_reference_cpu_safetensors(
            "qwen35-reference",
            &def,
            tmp.path(),
            DataType::FP32,
            Device::CPU,
        )
        .unwrap();
        let expected = toy_dense_runtime().forward(&[0, 1]).unwrap();
        let input = PrefillInput::new(MockTensor::from_u32(&[0, 1], &[2]).into_ref());

        let output = tokio_test::block_on(executor.prefill(&input)).unwrap();

        assert_close(&output.logits.to_vec_f32().unwrap(), &expected.logits[3..6]);
        assert_eq!(output.kv_cache.block_table().sequence_length, 2);
        assert!(executor.weight_validation().unwrap().is_pass());
        assert_eq!(executor.weight_plan().unwrap().prefix, "model");
    }

    #[test]
    fn qwen35_w3_reference_prefill_keeps_decode_unsupported_until_state_semantics_exist() {
        let def = toy_dense_definition();
        let executor = Qwen35W3Executor::from_definition(
            "qwen35-reference",
            &def,
            DataType::FP32,
            Device::CPU,
        )
        .unwrap()
        .with_dense_reference_runtime(toy_dense_runtime())
        .unwrap();
        let kv_cache = Arc::new(GenericKvCacheHandle::new(
            2,
            1,
            2,
            CandleDevice::Cpu,
            2,
            "qwen35-reference-test".to_string(),
        ));
        let input = DecodeInput::new(MockTensor::from_u32(&[1], &[1]).into_ref(), kv_cache);

        let err = tokio_test::block_on(executor.decode(&input))
            .expect_err("decode should remain unsupported before recurrent/KV state semantics");

        assert!(err.to_string().contains("not wired"), "{err}");
    }
}
