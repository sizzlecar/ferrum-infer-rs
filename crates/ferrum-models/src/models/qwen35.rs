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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35DeltaRuleShape {
    pub tokens: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35LinearAttentionShape {
    pub tokens: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35LinearAttentionReference {
    pub mixed_qkv_conv: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub g: Vec<f32>,
    pub beta: Vec<f32>,
    pub delta_core: Vec<f32>,
    pub delta_norm: Vec<f32>,
    pub final_state: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35DenseLinearAttentionLayerShape {
    pub tokens: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub attention: Qwen35LinearAttentionShape,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35DenseLinearAttentionLayerReference {
    pub input_norm: Vec<f32>,
    pub mixed_qkv_raw: Vec<f32>,
    pub z_raw: Vec<f32>,
    pub b_raw: Vec<f32>,
    pub a_raw: Vec<f32>,
    pub attention: Qwen35LinearAttentionReference,
    pub delta_output: Vec<f32>,
    pub residual_after_mixer: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub mlp_output: Vec<f32>,
    pub layer_output: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen35FullAttentionShape {
    pub tokens: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub position_offset: usize,
    pub rope_theta: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35FullAttentionReference {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub context: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen35DenseFullAttentionLayerShape {
    pub tokens: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub attention: Qwen35FullAttentionShape,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35DenseFullAttentionLayerReference {
    pub input_norm: Vec<f32>,
    pub query_raw: Vec<f32>,
    pub key_raw: Vec<f32>,
    pub value_raw: Vec<f32>,
    pub attention: Qwen35FullAttentionReference,
    pub attn_output: Vec<f32>,
    pub residual_after_attention: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub mlp_output: Vec<f32>,
    pub layer_output: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35SparseMoeShape {
    pub tokens: usize,
    pub hidden_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub expert_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35SparseMoeReference {
    pub router_logits: Vec<f32>,
    pub router_topk_indices: Vec<u32>,
    pub router_topk_weights: Vec<f32>,
    pub routed_expert_output: Vec<f32>,
    pub shared_expert_gate: Vec<f32>,
    pub shared_expert_output: Vec<f32>,
    pub moe_output: Vec<f32>,
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

#[allow(clippy::too_many_arguments)]
pub fn qwen35_dense_full_attention_layer_cpu(
    layer_input: &[f32],
    input_norm_weight: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    v_weight: &[f32],
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    o_weight: &[f32],
    post_attention_norm_weight: &[f32],
    gate_proj_weight: &[f32],
    up_proj_weight: &[f32],
    down_proj_weight: &[f32],
    shape: Qwen35DenseFullAttentionLayerShape,
    eps: f32,
) -> Result<Qwen35DenseFullAttentionLayerReference> {
    shape.validate()?;
    let attention_shape = shape.attention;
    let hidden_len = shape.tokens * shape.hidden_size;
    let q_total = attention_shape.q_total();
    let kv_total = attention_shape.kv_total();
    validate_len("dense full layer input", layer_input.len(), hidden_len)?;
    validate_len(
        "dense full layer input_norm_weight",
        input_norm_weight.len(),
        shape.hidden_size,
    )?;
    validate_len(
        "dense full layer q_weight",
        q_weight.len(),
        q_total * shape.hidden_size,
    )?;
    validate_len(
        "dense full layer k_weight",
        k_weight.len(),
        kv_total * shape.hidden_size,
    )?;
    validate_len(
        "dense full layer v_weight",
        v_weight.len(),
        kv_total * shape.hidden_size,
    )?;
    validate_len(
        "dense full layer q_norm_weight",
        q_norm_weight.len(),
        attention_shape.head_dim,
    )?;
    validate_len(
        "dense full layer k_norm_weight",
        k_norm_weight.len(),
        attention_shape.head_dim,
    )?;
    validate_len(
        "dense full layer o_weight",
        o_weight.len(),
        shape.hidden_size * q_total,
    )?;
    validate_len(
        "dense full layer post_attention_norm_weight",
        post_attention_norm_weight.len(),
        shape.hidden_size,
    )?;
    validate_len(
        "dense full layer gate_proj_weight",
        gate_proj_weight.len(),
        shape.intermediate_size * shape.hidden_size,
    )?;
    validate_len(
        "dense full layer up_proj_weight",
        up_proj_weight.len(),
        shape.intermediate_size * shape.hidden_size,
    )?;
    validate_len(
        "dense full layer down_proj_weight",
        down_proj_weight.len(),
        shape.hidden_size * shape.intermediate_size,
    )?;

    let input_norm = qwen35_rms_norm_plus_one_cpu(
        layer_input,
        input_norm_weight,
        shape.tokens,
        shape.hidden_size,
        eps,
    )?;
    let query_raw = qwen35_linear_cpu(
        &input_norm,
        q_weight,
        shape.tokens,
        shape.hidden_size,
        q_total,
    )?;
    let key_raw = qwen35_linear_cpu(
        &input_norm,
        k_weight,
        shape.tokens,
        shape.hidden_size,
        kv_total,
    )?;
    let value_raw = qwen35_linear_cpu(
        &input_norm,
        v_weight,
        shape.tokens,
        shape.hidden_size,
        kv_total,
    )?;
    let attention = qwen35_full_attention_core_cpu(
        &query_raw,
        &key_raw,
        &value_raw,
        q_norm_weight,
        k_norm_weight,
        attention_shape,
        eps,
    )?;
    let attn_output = qwen35_linear_cpu(
        &attention.context,
        o_weight,
        shape.tokens,
        q_total,
        shape.hidden_size,
    )?;
    let residual_after_attention =
        add_same_len(layer_input, &attn_output, "residual_after_attention")?;
    let post_attention_norm = qwen35_rms_norm_plus_one_cpu(
        &residual_after_attention,
        post_attention_norm_weight,
        shape.tokens,
        shape.hidden_size,
        eps,
    )?;
    let mlp_output = qwen35_dense_mlp_cpu(
        &post_attention_norm,
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        shape.tokens,
        shape.hidden_size,
        shape.intermediate_size,
    )?;
    let layer_output = add_same_len(&residual_after_attention, &mlp_output, "layer_output")?;

    Ok(Qwen35DenseFullAttentionLayerReference {
        input_norm,
        query_raw,
        key_raw,
        value_raw,
        attention,
        attn_output,
        residual_after_attention,
        post_attention_norm,
        mlp_output,
        layer_output,
    })
}

pub fn qwen35_full_attention_core_cpu(
    query_raw: &[f32],
    key_raw: &[f32],
    value_raw: &[f32],
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    shape: Qwen35FullAttentionShape,
    eps: f32,
) -> Result<Qwen35FullAttentionReference> {
    shape.validate()?;
    validate_len("full attention query_raw", query_raw.len(), shape.q_len())?;
    validate_len("full attention key_raw", key_raw.len(), shape.kv_len())?;
    validate_len("full attention value_raw", value_raw.len(), shape.kv_len())?;
    validate_len(
        "full attention q_norm_weight",
        q_norm_weight.len(),
        shape.head_dim,
    )?;
    validate_len(
        "full attention k_norm_weight",
        k_norm_weight.len(),
        shape.head_dim,
    )?;

    let mut query = qwen35_rms_norm_cpu(
        query_raw,
        q_norm_weight,
        shape.tokens * shape.num_heads,
        shape.head_dim,
        eps,
    )?;
    let mut key = qwen35_rms_norm_cpu(
        key_raw,
        k_norm_weight,
        shape.tokens * shape.num_kv_heads,
        shape.head_dim,
        eps,
    )?;
    qwen35_apply_rope_cpu(
        &mut query,
        shape.tokens,
        shape.num_heads,
        shape.head_dim,
        shape.position_offset,
        shape.rope_theta,
    )?;
    qwen35_apply_rope_cpu(
        &mut key,
        shape.tokens,
        shape.num_kv_heads,
        shape.head_dim,
        shape.position_offset,
        shape.rope_theta,
    )?;

    let repeat = shape.num_heads / shape.num_kv_heads;
    let scale = (shape.head_dim as f32).sqrt().recip();
    let mut context = vec![0.0; shape.q_len()];
    let mut scores = vec![0.0; shape.tokens];
    for token in 0..shape.tokens {
        for query_head in 0..shape.num_heads {
            let kv_head = query_head / repeat;
            let q_base = full_q_idx(shape, token, query_head, 0);
            let mut peak = f32::NEG_INFINITY;
            for key_token in 0..=token {
                let k_base = full_kv_idx(shape, key_token, kv_head, 0);
                let mut dot = 0.0;
                for dim in 0..shape.head_dim {
                    dot += query[q_base + dim] * key[k_base + dim];
                }
                let score = dot * scale;
                scores[key_token] = score;
                peak = peak.max(score);
            }
            let mut denom = 0.0;
            for key_token in 0..=token {
                let exp_score = (scores[key_token] - peak).exp();
                scores[key_token] = exp_score;
                denom += exp_score;
            }
            for dim in 0..shape.head_dim {
                let mut acc = 0.0;
                for key_token in 0..=token {
                    let prob = scores[key_token] / denom;
                    acc += prob * value_raw[full_kv_idx(shape, key_token, kv_head, dim)];
                }
                context[full_q_idx(shape, token, query_head, dim)] = acc;
            }
        }
    }

    Ok(Qwen35FullAttentionReference {
        query,
        key,
        value: value_raw.to_vec(),
        context,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_sparse_moe_shared_expert_cpu(
    x: &[f32],
    router_weight: &[f32],
    fused_gate_up_proj: &[f32],
    fused_down_proj: &[f32],
    shared_expert_gate_weight: &[f32],
    shared_expert_gate_proj_weight: &[f32],
    shared_expert_up_proj_weight: &[f32],
    shared_expert_down_proj_weight: &[f32],
    shape: Qwen35SparseMoeShape,
) -> Result<Qwen35SparseMoeReference> {
    shape.validate()?;
    validate_len(
        "sparse MoE input",
        x.len(),
        shape.tokens * shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE router_weight",
        router_weight.len(),
        shape.num_experts * shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE fused_gate_up_proj",
        fused_gate_up_proj.len(),
        shape.num_experts * 2 * shape.expert_intermediate_size * shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE fused_down_proj",
        fused_down_proj.len(),
        shape.num_experts * shape.hidden_size * shape.expert_intermediate_size,
    )?;
    validate_len(
        "sparse MoE shared_expert_gate_weight",
        shared_expert_gate_weight.len(),
        shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE shared_expert_gate_proj_weight",
        shared_expert_gate_proj_weight.len(),
        shape.shared_expert_intermediate_size * shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE shared_expert_up_proj_weight",
        shared_expert_up_proj_weight.len(),
        shape.shared_expert_intermediate_size * shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE shared_expert_down_proj_weight",
        shared_expert_down_proj_weight.len(),
        shape.hidden_size * shape.shared_expert_intermediate_size,
    )?;

    let router_logits = qwen35_linear_cpu(
        x,
        router_weight,
        shape.tokens,
        shape.hidden_size,
        shape.num_experts,
    )?;
    let routed = crate::moe::router::route(
        &router_logits,
        shape.tokens,
        shape.num_experts,
        shape.top_k,
        shape.norm_topk_prob,
    );
    let mut routed_expert_output = vec![0.0; shape.tokens * shape.hidden_size];
    for token_idx in 0..shape.tokens {
        let token = &x[token_idx * shape.hidden_size..(token_idx + 1) * shape.hidden_size];
        for slot in 0..shape.top_k {
            let pair_idx = token_idx * shape.top_k + slot;
            let expert_id = routed.expert_ids[pair_idx] as usize;
            let weight = routed.expert_weights[pair_idx];
            let expert_output = qwen35_fused_expert_mlp_cpu(
                token,
                fused_gate_up_proj,
                fused_down_proj,
                shape,
                expert_id,
            )?;
            for hidden_idx in 0..shape.hidden_size {
                routed_expert_output[token_idx * shape.hidden_size + hidden_idx] +=
                    weight * expert_output[hidden_idx];
            }
        }
    }

    let shared_expert_gate_raw = qwen35_linear_cpu(
        x,
        shared_expert_gate_weight,
        shape.tokens,
        shape.hidden_size,
        1,
    )?;
    let shared_expert_gate = shared_expert_gate_raw
        .into_iter()
        .map(sigmoid)
        .collect::<Vec<_>>();
    let shared_dense = qwen35_dense_mlp_cpu(
        x,
        shared_expert_gate_proj_weight,
        shared_expert_up_proj_weight,
        shared_expert_down_proj_weight,
        shape.tokens,
        shape.hidden_size,
        shape.shared_expert_intermediate_size,
    )?;
    let mut shared_expert_output = vec![0.0; shape.tokens * shape.hidden_size];
    for token_idx in 0..shape.tokens {
        let gate = shared_expert_gate[token_idx];
        for hidden_idx in 0..shape.hidden_size {
            let idx = token_idx * shape.hidden_size + hidden_idx;
            shared_expert_output[idx] = gate * shared_dense[idx];
        }
    }
    let moe_output = add_same_len(&routed_expert_output, &shared_expert_output, "moe_output")?;

    Ok(Qwen35SparseMoeReference {
        router_logits,
        router_topk_indices: routed.expert_ids,
        router_topk_weights: routed.expert_weights,
        routed_expert_output,
        shared_expert_gate,
        shared_expert_output,
        moe_output,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_dense_linear_attention_layer_cpu(
    layer_input: &[f32],
    input_norm_weight: &[f32],
    qkv_weight: &[f32],
    z_weight: &[f32],
    b_weight: &[f32],
    a_weight: &[f32],
    conv1d_weight: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    norm_weight: &[f32],
    out_proj_weight: &[f32],
    post_attention_norm_weight: &[f32],
    gate_proj_weight: &[f32],
    up_proj_weight: &[f32],
    down_proj_weight: &[f32],
    initial_state: &[f32],
    shape: Qwen35DenseLinearAttentionLayerShape,
    eps: f32,
) -> Result<Qwen35DenseLinearAttentionLayerReference> {
    shape.validate()?;
    let attention_shape = shape.attention;
    let hidden_len = shape.tokens * shape.hidden_size;
    let value_total = attention_shape.value_total();
    validate_len("dense layer input", layer_input.len(), hidden_len)?;
    validate_len(
        "dense layer input_norm_weight",
        input_norm_weight.len(),
        shape.hidden_size,
    )?;
    validate_len(
        "dense layer qkv_weight",
        qkv_weight.len(),
        attention_shape.conv_channels() * shape.hidden_size,
    )?;
    validate_len(
        "dense layer z_weight",
        z_weight.len(),
        value_total * shape.hidden_size,
    )?;
    validate_len(
        "dense layer b_weight",
        b_weight.len(),
        attention_shape.value_heads * shape.hidden_size,
    )?;
    validate_len(
        "dense layer a_weight",
        a_weight.len(),
        attention_shape.value_heads * shape.hidden_size,
    )?;
    validate_len(
        "dense layer out_proj_weight",
        out_proj_weight.len(),
        shape.hidden_size * value_total,
    )?;
    validate_len(
        "dense layer post_attention_norm_weight",
        post_attention_norm_weight.len(),
        shape.hidden_size,
    )?;
    validate_len(
        "dense layer gate_proj_weight",
        gate_proj_weight.len(),
        shape.intermediate_size * shape.hidden_size,
    )?;
    validate_len(
        "dense layer up_proj_weight",
        up_proj_weight.len(),
        shape.intermediate_size * shape.hidden_size,
    )?;
    validate_len(
        "dense layer down_proj_weight",
        down_proj_weight.len(),
        shape.hidden_size * shape.intermediate_size,
    )?;

    let input_norm = qwen35_rms_norm_plus_one_cpu(
        layer_input,
        input_norm_weight,
        shape.tokens,
        shape.hidden_size,
        eps,
    )?;
    let mixed_qkv_raw = qwen35_linear_cpu(
        &input_norm,
        qkv_weight,
        shape.tokens,
        shape.hidden_size,
        attention_shape.conv_channels(),
    )?;
    let z_raw = qwen35_linear_cpu(
        &input_norm,
        z_weight,
        shape.tokens,
        shape.hidden_size,
        value_total,
    )?;
    let b_raw = qwen35_linear_cpu(
        &input_norm,
        b_weight,
        shape.tokens,
        shape.hidden_size,
        attention_shape.value_heads,
    )?;
    let a_raw = qwen35_linear_cpu(
        &input_norm,
        a_weight,
        shape.tokens,
        shape.hidden_size,
        attention_shape.value_heads,
    )?;
    let attention = qwen35_linear_attention_core_cpu(
        &mixed_qkv_raw,
        &z_raw,
        &a_raw,
        &b_raw,
        conv1d_weight,
        a_log,
        dt_bias,
        norm_weight,
        initial_state,
        attention_shape,
        eps,
    )?;
    let delta_output = qwen35_linear_cpu(
        &attention.delta_norm,
        out_proj_weight,
        shape.tokens,
        value_total,
        shape.hidden_size,
    )?;
    let residual_after_mixer = add_same_len(layer_input, &delta_output, "residual_after_mixer")?;
    let post_attention_norm = qwen35_rms_norm_plus_one_cpu(
        &residual_after_mixer,
        post_attention_norm_weight,
        shape.tokens,
        shape.hidden_size,
        eps,
    )?;
    let gate = qwen35_linear_cpu(
        &post_attention_norm,
        gate_proj_weight,
        shape.tokens,
        shape.hidden_size,
        shape.intermediate_size,
    )?;
    let up = qwen35_linear_cpu(
        &post_attention_norm,
        up_proj_weight,
        shape.tokens,
        shape.hidden_size,
        shape.intermediate_size,
    )?;
    let fused = gate
        .iter()
        .zip(&up)
        .map(|(gate, up)| silu(*gate) * up)
        .collect::<Vec<_>>();
    let mlp_output = qwen35_linear_cpu(
        &fused,
        down_proj_weight,
        shape.tokens,
        shape.intermediate_size,
        shape.hidden_size,
    )?;
    let layer_output = add_same_len(&residual_after_mixer, &mlp_output, "layer_output")?;

    Ok(Qwen35DenseLinearAttentionLayerReference {
        input_norm,
        mixed_qkv_raw,
        z_raw,
        b_raw,
        a_raw,
        attention,
        delta_output,
        residual_after_mixer,
        post_attention_norm,
        mlp_output,
        layer_output,
    })
}

pub fn qwen35_linear_cpu(
    x: &[f32],
    weight: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    if rows == 0 || in_dim == 0 || out_dim == 0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 linear shape must be positive, got rows={rows} in_dim={in_dim} out_dim={out_dim}"
        )));
    }
    validate_len("linear input", x.len(), rows * in_dim)?;
    validate_len("linear weight", weight.len(), out_dim * in_dim)?;

    let mut out = vec![0.0; rows * out_dim];
    for row in 0..rows {
        for out_col in 0..out_dim {
            let mut acc = 0.0;
            for in_col in 0..in_dim {
                acc += x[row * in_dim + in_col] * weight[out_col * in_dim + in_col];
            }
            out[row * out_dim + out_col] = acc;
        }
    }
    Ok(out)
}

pub fn qwen35_dense_mlp_cpu(
    x: &[f32],
    gate_proj_weight: &[f32],
    up_proj_weight: &[f32],
    down_proj_weight: &[f32],
    tokens: usize,
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<Vec<f32>> {
    if tokens == 0 || hidden_size == 0 || intermediate_size == 0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 dense MLP shape must be positive, got tokens={tokens} hidden_size={hidden_size} intermediate_size={intermediate_size}"
        )));
    }
    validate_len("dense MLP input", x.len(), tokens * hidden_size)?;
    validate_len(
        "dense MLP gate_proj_weight",
        gate_proj_weight.len(),
        intermediate_size * hidden_size,
    )?;
    validate_len(
        "dense MLP up_proj_weight",
        up_proj_weight.len(),
        intermediate_size * hidden_size,
    )?;
    validate_len(
        "dense MLP down_proj_weight",
        down_proj_weight.len(),
        hidden_size * intermediate_size,
    )?;

    let gate = qwen35_linear_cpu(x, gate_proj_weight, tokens, hidden_size, intermediate_size)?;
    let up = qwen35_linear_cpu(x, up_proj_weight, tokens, hidden_size, intermediate_size)?;
    let fused = gate
        .iter()
        .zip(&up)
        .map(|(gate, up)| silu(*gate) * up)
        .collect::<Vec<_>>();
    qwen35_linear_cpu(
        &fused,
        down_proj_weight,
        tokens,
        intermediate_size,
        hidden_size,
    )
}

fn qwen35_fused_expert_mlp_cpu(
    token: &[f32],
    fused_gate_up_proj: &[f32],
    fused_down_proj: &[f32],
    shape: Qwen35SparseMoeShape,
    expert_id: usize,
) -> Result<Vec<f32>> {
    if expert_id >= shape.num_experts {
        return Err(FerrumError::model(format!(
            "Qwen3.5 sparse MoE expert_id {expert_id} exceeds num_experts {}",
            shape.num_experts
        )));
    }
    validate_len("sparse MoE expert input", token.len(), shape.hidden_size)?;
    let gate_up_per_expert = 2 * shape.expert_intermediate_size * shape.hidden_size;
    let down_per_expert = shape.hidden_size * shape.expert_intermediate_size;
    let gate_up_start = expert_id * gate_up_per_expert;
    let gate_start = gate_up_start;
    let up_start = gate_start + shape.expert_intermediate_size * shape.hidden_size;
    let down_start = expert_id * down_per_expert;
    let gate = qwen35_linear_cpu(
        token,
        &fused_gate_up_proj[gate_start..up_start],
        1,
        shape.hidden_size,
        shape.expert_intermediate_size,
    )?;
    let up = qwen35_linear_cpu(
        token,
        &fused_gate_up_proj[up_start..gate_up_start + gate_up_per_expert],
        1,
        shape.hidden_size,
        shape.expert_intermediate_size,
    )?;
    let fused = gate
        .iter()
        .zip(&up)
        .map(|(gate, up)| silu(*gate) * up)
        .collect::<Vec<_>>();
    qwen35_linear_cpu(
        &fused,
        &fused_down_proj[down_start..down_start + down_per_expert],
        1,
        shape.expert_intermediate_size,
        shape.hidden_size,
    )
}

pub fn qwen35_rms_norm_cpu(
    x: &[f32],
    weight: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if rows == 0 || dim == 0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 RMSNorm shape must be positive, got rows={rows} dim={dim}"
        )));
    }
    validate_len("RMSNorm input", x.len(), rows * dim)?;
    validate_len("RMSNorm weight", weight.len(), dim)?;

    let mut out = vec![0.0; rows * dim];
    for row in 0..rows {
        let base = row * dim;
        let mean = x[base..base + dim]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / dim as f32;
        let inv = (mean + eps).sqrt().recip();
        for i in 0..dim {
            out[base + i] = x[base + i] * inv * weight[i];
        }
    }
    Ok(out)
}

pub fn qwen35_rms_norm_plus_one_cpu(
    x: &[f32],
    weight: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if rows == 0 || dim == 0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 RMSNorm shape must be positive, got rows={rows} dim={dim}"
        )));
    }
    validate_len("RMSNorm input", x.len(), rows * dim)?;
    validate_len("RMSNorm weight", weight.len(), dim)?;

    let mut out = vec![0.0; rows * dim];
    for row in 0..rows {
        let base = row * dim;
        let mean = x[base..base + dim]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / dim as f32;
        let inv = (mean + eps).sqrt().recip();
        for i in 0..dim {
            out[base + i] = x[base + i] * inv * (1.0 + weight[i]);
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_linear_attention_core_cpu(
    mixed_qkv_raw: &[f32],
    z_raw: &[f32],
    a_raw: &[f32],
    b_raw: &[f32],
    conv1d_weight: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    norm_weight: &[f32],
    initial_state: &[f32],
    shape: Qwen35LinearAttentionShape,
    eps: f32,
) -> Result<Qwen35LinearAttentionReference> {
    shape.validate()?;
    validate_len("mixed_qkv_raw", mixed_qkv_raw.len(), shape.mixed_qkv_len())?;
    validate_len("z_raw", z_raw.len(), shape.value_len())?;
    let gating_len = shape.gating_len();
    validate_len("a_raw", a_raw.len(), gating_len)?;
    validate_len("b_raw", b_raw.len(), gating_len)?;
    validate_len(
        "conv1d_weight",
        conv1d_weight.len(),
        shape.conv_channels() * shape.conv_kernel,
    )?;
    validate_len("a_log", a_log.len(), shape.value_heads)?;
    validate_len("dt_bias", dt_bias.len(), shape.value_heads)?;
    validate_len("norm_weight", norm_weight.len(), shape.value_dim)?;
    validate_len("initial_state", initial_state.len(), shape.state_len())?;

    let mixed_qkv_conv = qwen35_depthwise_causal_conv_silu_cpu(
        mixed_qkv_raw,
        conv1d_weight,
        shape.tokens,
        shape.conv_channels(),
        shape.conv_kernel,
    )?;
    let (query, key, value) = qwen35_split_linear_attention_qkv_cpu(&mixed_qkv_conv, shape)?;
    let (g, beta) = qwen35_gdn_gating_cpu(
        a_log,
        a_raw,
        b_raw,
        dt_bias,
        shape.tokens,
        shape.value_heads,
    )?;
    let (delta_core, final_state) = qwen35_recurrent_gated_delta_rule_cpu(
        &query,
        &key,
        &value,
        &g,
        &beta,
        initial_state,
        shape.delta_shape(),
        true,
        None,
    )?;
    let delta_norm = qwen35_gated_rms_norm_cpu(
        &delta_core,
        z_raw,
        norm_weight,
        shape.tokens,
        shape.value_heads,
        shape.value_dim,
        eps,
    )?;

    Ok(Qwen35LinearAttentionReference {
        mixed_qkv_conv,
        query,
        key,
        value,
        g,
        beta,
        delta_core,
        delta_norm,
        final_state,
    })
}

pub fn qwen35_depthwise_causal_conv_silu_cpu(
    x: &[f32],
    weight: &[f32],
    tokens: usize,
    channels: usize,
    kernel: usize,
) -> Result<Vec<f32>> {
    if tokens == 0 || channels == 0 || kernel == 0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 depthwise conv shape must be positive, got tokens={tokens} channels={channels} kernel={kernel}"
        )));
    }
    validate_len("depthwise conv input", x.len(), tokens * channels)?;
    validate_len("depthwise conv weight", weight.len(), channels * kernel)?;

    let pad = kernel - 1;
    let mut out = vec![0.0; tokens * channels];
    for token in 0..tokens {
        for channel in 0..channels {
            let mut acc = 0.0;
            for kernel_idx in 0..kernel {
                let padded_idx = token + kernel_idx;
                if padded_idx >= pad {
                    let src_token = padded_idx - pad;
                    if src_token < tokens {
                        acc += x[src_token * channels + channel]
                            * weight[(channel * kernel) + kernel_idx];
                    }
                }
            }
            out[token * channels + channel] = silu(acc);
        }
    }
    Ok(out)
}

pub fn qwen35_split_linear_attention_qkv_cpu(
    mixed_qkv: &[f32],
    shape: Qwen35LinearAttentionShape,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    shape.validate()?;
    validate_len("mixed_qkv", mixed_qkv.len(), shape.mixed_qkv_len())?;
    let qk_total = shape.qk_total();
    let value_total = shape.value_total();
    let width = shape.conv_channels();
    Ok((
        split_features(mixed_qkv, shape.tokens, width, 0, qk_total),
        split_features(mixed_qkv, shape.tokens, width, qk_total, qk_total),
        split_features(mixed_qkv, shape.tokens, width, qk_total * 2, value_total),
    ))
}

pub fn qwen35_gated_rms_norm_cpu(
    core: &[f32],
    z: &[f32],
    weight: &[f32],
    tokens: usize,
    heads: usize,
    dim: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if tokens == 0 || heads == 0 || dim == 0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 gated RMSNorm shape must be positive, got tokens={tokens} heads={heads} dim={dim}"
        )));
    }
    let len = tokens * heads * dim;
    validate_len("gated RMSNorm core", core.len(), len)?;
    validate_len("gated RMSNorm z", z.len(), len)?;
    validate_len("gated RMSNorm weight", weight.len(), dim)?;

    let rows = tokens * heads;
    let mut out = vec![0.0; len];
    for row in 0..rows {
        let base = row * dim;
        let mean = core[base..base + dim]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / dim as f32;
        let inv = (mean + eps).sqrt().recip();
        for i in 0..dim {
            out[base + i] = core[base + i] * inv * weight[i] * silu(z[base + i]);
        }
    }
    Ok(out)
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

pub fn qwen35_recurrent_gated_delta_rule_cpu(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    g: &[f32],
    beta: &[f32],
    initial_state: &[f32],
    shape: Qwen35DeltaRuleShape,
    use_qk_l2norm: bool,
    scale: Option<f32>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    validate_delta_rule_shapes(query, key, value, g, beta, initial_state, shape)?;
    if shape.value_heads % shape.key_heads != 0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 DeltaNet value_heads {} must be divisible by key_heads {}",
            shape.value_heads, shape.key_heads
        )));
    }
    let repeat_factor = shape.value_heads / shape.key_heads;
    let scale = scale.unwrap_or_else(|| (shape.key_dim as f32).sqrt().recip());
    let mut state = initial_state.to_vec();
    let mut out = vec![0.0; shape.tokens * shape.value_heads * shape.value_dim];

    for token in 0..shape.tokens {
        for value_head in 0..shape.value_heads {
            let key_head = value_head / repeat_factor;
            let mut q = vec![0.0; shape.key_dim];
            let mut k = vec![0.0; shape.key_dim];
            for d in 0..shape.key_dim {
                q[d] = query[delta_qk_idx(shape, token, key_head, d)];
                k[d] = key[delta_qk_idx(shape, token, key_head, d)];
            }
            if use_qk_l2norm {
                l2_normalize(&mut q);
                l2_normalize(&mut k);
            }
            for value in &mut q {
                *value *= scale;
            }

            let decay = g[token * shape.value_heads + value_head].exp();
            let beta_t = beta[token * shape.value_heads + value_head];
            for vd in 0..shape.value_dim {
                let mut kv_mem = 0.0;
                for kd in 0..shape.key_dim {
                    let state_idx = delta_state_idx(shape, value_head, vd, kd);
                    state[state_idx] *= decay;
                    kv_mem += state[state_idx] * k[kd];
                }
                let v_t = value[delta_value_idx(shape, token, value_head, vd)];
                let delta = (v_t - kv_mem) * beta_t;
                for kd in 0..shape.key_dim {
                    let state_idx = delta_state_idx(shape, value_head, vd, kd);
                    state[state_idx] += delta * k[kd];
                }
                let mut acc = 0.0;
                for kd in 0..shape.key_dim {
                    acc += state[delta_state_idx(shape, value_head, vd, kd)] * q[kd];
                }
                out[delta_value_idx(shape, token, value_head, vd)] = acc;
            }
        }
    }

    Ok((out, state))
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_gated_delta_attention_cpu(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    initial_state: &[f32],
    shape: Qwen35DeltaRuleShape,
    use_qk_l2norm: bool,
    scale: Option<f32>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (g, beta) = qwen35_gdn_gating_cpu(a_log, a, b, dt_bias, shape.tokens, shape.value_heads)?;
    qwen35_recurrent_gated_delta_rule_cpu(
        query,
        key,
        value,
        &g,
        &beta,
        initial_state,
        shape,
        use_qk_l2norm,
        scale,
    )
}

pub fn qwen35_gdn_gating_cpu(
    a_log: &[f32],
    a: &[f32],
    b: &[f32],
    dt_bias: &[f32],
    tokens: usize,
    value_heads: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let gating_len = tokens
        .checked_mul(value_heads)
        .ok_or_else(|| FerrumError::model("Qwen3.5 GDN gating shape overflow".to_string()))?;
    for (label, actual, expected) in [
        ("A_log", a_log.len(), value_heads),
        ("dt_bias", dt_bias.len(), value_heads),
        ("a", a.len(), gating_len),
        ("b", b.len(), gating_len),
    ] {
        if actual != expected {
            return Err(FerrumError::model(format!(
                "Qwen3.5 GDN gating {label} length {actual} != expected {expected}"
            )));
        }
    }
    let mut g = vec![0.0; gating_len];
    let mut beta = vec![0.0; gating_len];
    for token in 0..tokens {
        for head in 0..value_heads {
            let idx = token * value_heads + head;
            g[idx] = -a_log[head].exp() * softplus(a[idx] + dt_bias[head]);
            beta[idx] = sigmoid(b[idx]);
        }
    }
    Ok((g, beta))
}

impl Qwen35LinearAttentionShape {
    fn validate(self) -> Result<()> {
        if self.tokens == 0
            || self.key_heads == 0
            || self.value_heads == 0
            || self.key_dim == 0
            || self.value_dim == 0
            || self.conv_kernel == 0
        {
            return Err(FerrumError::model(format!(
                "Qwen3.5 linear-attention shape must be positive, got {self:?}"
            )));
        }
        if self.value_heads % self.key_heads != 0 {
            return Err(FerrumError::model(format!(
                "Qwen3.5 linear-attention value_heads {} must be divisible by key_heads {}",
                self.value_heads, self.key_heads
            )));
        }
        Ok(())
    }

    fn qk_total(self) -> usize {
        self.key_heads * self.key_dim
    }

    fn value_total(self) -> usize {
        self.value_heads * self.value_dim
    }

    fn conv_channels(self) -> usize {
        self.qk_total() * 2 + self.value_total()
    }

    fn mixed_qkv_len(self) -> usize {
        self.tokens * self.conv_channels()
    }

    fn value_len(self) -> usize {
        self.tokens * self.value_total()
    }

    fn gating_len(self) -> usize {
        self.tokens * self.value_heads
    }

    fn state_len(self) -> usize {
        self.value_heads * self.value_dim * self.key_dim
    }

    fn delta_shape(self) -> Qwen35DeltaRuleShape {
        Qwen35DeltaRuleShape {
            tokens: self.tokens,
            key_heads: self.key_heads,
            value_heads: self.value_heads,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
        }
    }
}

impl Qwen35DenseLinearAttentionLayerShape {
    fn validate(self) -> Result<()> {
        if self.tokens == 0 || self.hidden_size == 0 || self.intermediate_size == 0 {
            return Err(FerrumError::model(format!(
                "Qwen3.5 dense linear-attention layer shape must be positive, got {self:?}"
            )));
        }
        self.attention.validate()?;
        if self.attention.tokens != self.tokens {
            return Err(FerrumError::model(format!(
                "Qwen3.5 dense layer tokens {} must match attention tokens {}",
                self.tokens, self.attention.tokens
            )));
        }
        Ok(())
    }
}

impl Qwen35FullAttentionShape {
    fn validate(self) -> Result<()> {
        if self.tokens == 0
            || self.num_heads == 0
            || self.num_kv_heads == 0
            || self.head_dim == 0
            || self.rope_theta <= 0.0
        {
            return Err(FerrumError::model(format!(
                "Qwen3.5 full-attention shape must be positive, got {self:?}"
            )));
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err(FerrumError::model(format!(
                "Qwen3.5 full-attention num_heads {} must be divisible by num_kv_heads {}",
                self.num_heads, self.num_kv_heads
            )));
        }
        if self.head_dim % 2 != 0 {
            return Err(FerrumError::model(format!(
                "Qwen3.5 full-attention head_dim {} must be even for RoPE",
                self.head_dim
            )));
        }
        Ok(())
    }

    fn q_total(self) -> usize {
        self.num_heads * self.head_dim
    }

    fn kv_total(self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    fn q_len(self) -> usize {
        self.tokens * self.q_total()
    }

    fn kv_len(self) -> usize {
        self.tokens * self.kv_total()
    }
}

impl Qwen35DenseFullAttentionLayerShape {
    fn validate(self) -> Result<()> {
        if self.tokens == 0 || self.hidden_size == 0 || self.intermediate_size == 0 {
            return Err(FerrumError::model(format!(
                "Qwen3.5 dense full-attention layer shape must be positive, got {self:?}"
            )));
        }
        self.attention.validate()?;
        if self.attention.tokens != self.tokens {
            return Err(FerrumError::model(format!(
                "Qwen3.5 dense full layer tokens {} must match attention tokens {}",
                self.tokens, self.attention.tokens
            )));
        }
        if self.attention.q_total() != self.hidden_size {
            return Err(FerrumError::model(format!(
                "Qwen3.5 dense full layer hidden_size {} must match attention q_total {}",
                self.hidden_size,
                self.attention.q_total()
            )));
        }
        Ok(())
    }
}

impl Qwen35SparseMoeShape {
    fn validate(self) -> Result<()> {
        if self.tokens == 0
            || self.hidden_size == 0
            || self.num_experts == 0
            || self.top_k == 0
            || self.expert_intermediate_size == 0
            || self.shared_expert_intermediate_size == 0
        {
            return Err(FerrumError::model(format!(
                "Qwen3.5 sparse MoE shape must be positive, got {self:?}"
            )));
        }
        if self.top_k > self.num_experts {
            return Err(FerrumError::model(format!(
                "Qwen3.5 sparse MoE top_k {} exceeds num_experts {}",
                self.top_k, self.num_experts
            )));
        }
        Ok(())
    }
}

fn validate_delta_rule_shapes(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    g: &[f32],
    beta: &[f32],
    initial_state: &[f32],
    shape: Qwen35DeltaRuleShape,
) -> Result<()> {
    if shape.tokens == 0
        || shape.key_heads == 0
        || shape.value_heads == 0
        || shape.key_dim == 0
        || shape.value_dim == 0
    {
        return Err(FerrumError::model(format!(
            "Qwen3.5 DeltaNet shape must be positive, got {shape:?}"
        )));
    }
    let qk_len = shape.tokens * shape.key_heads * shape.key_dim;
    let value_len = shape.tokens * shape.value_heads * shape.value_dim;
    let gating_len = shape.tokens * shape.value_heads;
    let state_len = shape.value_heads * shape.value_dim * shape.key_dim;
    for (label, actual, expected) in [
        ("query", query.len(), qk_len),
        ("key", key.len(), qk_len),
        ("value", value.len(), value_len),
        ("g", g.len(), gating_len),
        ("beta", beta.len(), gating_len),
        ("initial_state", initial_state.len(), state_len),
    ] {
        if actual != expected {
            return Err(FerrumError::model(format!(
                "Qwen3.5 DeltaNet {label} length {actual} != expected {expected} for {shape:?}"
            )));
        }
    }
    Ok(())
}

fn add_same_len(lhs: &[f32], rhs: &[f32], label: &str) -> Result<Vec<f32>> {
    if lhs.len() != rhs.len() {
        return Err(FerrumError::model(format!(
            "Qwen3.5 {label} add length mismatch: lhs {} != rhs {}",
            lhs.len(),
            rhs.len()
        )));
    }
    Ok(lhs
        .iter()
        .zip(rhs)
        .map(|(left, right)| left + right)
        .collect())
}

fn qwen35_apply_rope_cpu(
    x: &mut [f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    position_offset: usize,
    rope_theta: f32,
) -> Result<()> {
    if tokens == 0 || heads == 0 || head_dim == 0 || rope_theta <= 0.0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 RoPE shape must be positive, got tokens={tokens} heads={heads} head_dim={head_dim} rope_theta={rope_theta}"
        )));
    }
    if head_dim % 2 != 0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 RoPE head_dim {head_dim} must be even"
        )));
    }
    validate_len("RoPE input", x.len(), tokens * heads * head_dim)?;

    let half = head_dim / 2;
    for token in 0..tokens {
        let position = (position_offset + token) as f32;
        for pair in 0..half {
            let inv_freq = rope_theta.powf(-(2.0 * pair as f32) / head_dim as f32);
            let angle = position * inv_freq;
            let (sin, cos) = angle.sin_cos();
            for head in 0..heads {
                let base = ((token * heads + head) * head_dim) + pair;
                let x0 = x[base];
                let x1 = x[base + half];
                x[base] = x0 * cos - x1 * sin;
                x[base + half] = x0 * sin + x1 * cos;
            }
        }
    }
    Ok(())
}

fn validate_len(label: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(FerrumError::model(format!(
            "Qwen3.5 {label} length {actual} != expected {expected}"
        )));
    }
    Ok(())
}

fn l2_normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>();
    let inv = (norm + 1e-6).sqrt().recip();
    for value in values {
        *value *= inv;
    }
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn delta_qk_idx(shape: Qwen35DeltaRuleShape, token: usize, head: usize, dim: usize) -> usize {
    ((token * shape.key_heads + head) * shape.key_dim) + dim
}

fn delta_value_idx(shape: Qwen35DeltaRuleShape, token: usize, head: usize, dim: usize) -> usize {
    ((token * shape.value_heads + head) * shape.value_dim) + dim
}

fn delta_state_idx(
    shape: Qwen35DeltaRuleShape,
    head: usize,
    value_dim: usize,
    key_dim: usize,
) -> usize {
    ((head * shape.value_dim + value_dim) * shape.key_dim) + key_dim
}

fn full_q_idx(shape: Qwen35FullAttentionShape, token: usize, head: usize, dim: usize) -> usize {
    ((token * shape.num_heads + head) * shape.head_dim) + dim
}

fn full_kv_idx(shape: Qwen35FullAttentionShape, token: usize, head: usize, dim: usize) -> usize {
    ((token * shape.num_kv_heads + head) * shape.head_dim) + dim
}

fn split_features(x: &[f32], rows: usize, width: usize, start: usize, len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * len);
    for row in 0..rows {
        out.extend_from_slice(&x[row * width + start..row * width + start + len]);
    }
    out
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

    fn assert_close_slice(got: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(
            got.len(),
            expected.len(),
            "length mismatch: got {} expected {}",
            got.len(),
            expected.len()
        );
        for (idx, (got, expected)) in got.iter().zip(expected).enumerate() {
            assert!(
                (got - expected).abs() <= tolerance,
                "idx {idx}: {got} != {expected}"
            );
        }
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

    #[test]
    fn linear_cpu_uses_row_major_out_in_weights() {
        let out = qwen35_linear_cpu(
            &[1.0, 2.0, 3.0, 4.0],
            &[
                1.0, 0.0, //
                0.0, 1.0, //
                1.0, 1.0,
            ],
            2,
            2,
            3,
        )
        .unwrap();

        assert_eq!(out, vec![1.0, 2.0, 3.0, 3.0, 4.0, 7.0]);
    }

    #[test]
    fn rms_norm_plus_one_matches_qwen35_semantics() {
        let out = qwen35_rms_norm_plus_one_cpu(&[3.0, 4.0], &[0.0, 1.0], 1, 2, 0.0).unwrap();
        let inv = ((3.0f32 * 3.0 + 4.0 * 4.0) / 2.0).sqrt().recip();

        assert_close_slice(&out, &[3.0 * inv, 4.0 * inv * 2.0], 1e-6);
    }

    #[test]
    fn rms_norm_matches_qk_norm_semantics() {
        let out = qwen35_rms_norm_cpu(&[3.0, 4.0], &[2.0, 3.0], 1, 2, 0.0).unwrap();
        let inv = ((3.0f32 * 3.0 + 4.0 * 4.0) / 2.0).sqrt().recip();

        assert_close_slice(&out, &[3.0 * inv * 2.0, 4.0 * inv * 3.0], 1e-6);
    }

    #[test]
    fn rope_uses_non_interleaved_half_rotation() {
        let mut values = vec![1.0, 0.0];
        qwen35_apply_rope_cpu(&mut values, 1, 1, 2, 1, 10_000.0).unwrap();
        let (sin, cos) = 1.0f32.sin_cos();

        assert_close_slice(&values, &[cos, sin], 1e-6);
    }

    #[test]
    fn full_attention_core_applies_causal_softmax() {
        let shape = Qwen35FullAttentionShape {
            tokens: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            position_offset: 0,
            rope_theta: 10_000.0,
        };
        let reference = qwen35_full_attention_core_cpu(
            &[0.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0],
            &[2.0, 4.0, 6.0, 8.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            shape,
            1e-6,
        )
        .unwrap();

        assert_close_slice(&reference.query, &[0.0, 0.0, 0.0, 0.0], 1e-6);
        assert_close_slice(&reference.key, &[0.0, 0.0, 0.0, 0.0], 1e-6);
        assert_close_slice(&reference.value, &[2.0, 4.0, 6.0, 8.0], 1e-6);
        assert_close_slice(&reference.context, &[2.0, 4.0, 4.0, 6.0], 1e-6);
    }

    #[test]
    fn full_attention_core_rejects_invalid_gqa_shape() {
        let shape = Qwen35FullAttentionShape {
            tokens: 1,
            num_heads: 3,
            num_kv_heads: 2,
            head_dim: 2,
            position_offset: 0,
            rope_theta: 10_000.0,
        };
        let err = qwen35_full_attention_core_cpu(
            &[0.0; 6], &[0.0; 4], &[0.0; 4], &[1.0; 2], &[1.0; 2], shape, 1e-6,
        )
        .expect_err("num_heads must divide num_kv_heads");

        assert!(err.to_string().contains("must be divisible"), "{err}");
    }

    #[test]
    fn depthwise_causal_conv_uses_left_context_only() {
        let out =
            qwen35_depthwise_causal_conv_silu_cpu(&[1.0, 2.0, 3.0], &[10.0, 1.0], 3, 1, 2).unwrap();
        let expected_raw = [1.0, 12.0, 23.0];
        let expected = expected_raw.map(silu);

        assert_close_slice(&out, &expected, 1e-6);
    }

    #[test]
    fn gated_rms_norm_matches_reference_formula() {
        let out = qwen35_gated_rms_norm_cpu(
            &[3.0, 4.0, 1.0, 2.0],
            &[0.0, 1.0, -1.0, 2.0],
            &[2.0, 3.0],
            2,
            1,
            2,
            0.0,
        )
        .unwrap();
        let first_inv = ((3.0f32 * 3.0 + 4.0 * 4.0) / 2.0).sqrt().recip();
        let second_inv = ((1.0f32 * 1.0 + 2.0 * 2.0) / 2.0).sqrt().recip();
        let expected = vec![
            3.0 * first_inv * 2.0 * silu(0.0),
            4.0 * first_inv * 3.0 * silu(1.0),
            1.0 * second_inv * 2.0 * silu(-1.0),
            2.0 * second_inv * 3.0 * silu(2.0),
        ];

        assert_close_slice(&out, &expected, 1e-6);
    }

    #[test]
    fn dense_linear_attention_layer_composes_attention_residual_and_dense_mlp() {
        let attention_shape = Qwen35LinearAttentionShape {
            tokens: 2,
            key_heads: 1,
            value_heads: 1,
            key_dim: 1,
            value_dim: 1,
            conv_kernel: 1,
        };
        let shape = Qwen35DenseLinearAttentionLayerShape {
            tokens: 2,
            hidden_size: 2,
            intermediate_size: 2,
            attention: attention_shape,
        };
        let layer_input = vec![1.0, 2.0, 3.0, 4.0];
        let input_norm_weight = vec![0.0, 0.0];
        let qkv_weight = vec![
            1.0, 0.0, //
            0.0, 1.0, //
            1.0, 1.0,
        ];
        let z_weight = vec![1.0, -1.0];
        let b_weight = vec![0.5, 0.25];
        let a_weight = vec![-0.25, 0.75];
        let conv1d_weight = vec![1.0, 1.0, 1.0];
        let a_log = vec![0.0];
        let dt_bias = vec![0.0];
        let norm_weight = vec![1.0];
        let out_proj_weight = vec![1.0, -0.5];
        let post_attention_norm_weight = vec![0.0, 0.0];
        let gate_proj_weight = vec![
            0.2, 0.1, //
            -0.1, 0.3,
        ];
        let up_proj_weight = vec![
            0.4, -0.2, //
            0.3, 0.5,
        ];
        let down_proj_weight = vec![
            1.0, 0.0, //
            0.0, 1.0,
        ];
        let initial_state = vec![0.0; attention_shape.state_len()];

        let input_norm = qwen35_rms_norm_plus_one_cpu(
            &layer_input,
            &input_norm_weight,
            shape.tokens,
            shape.hidden_size,
            1e-6,
        )
        .unwrap();
        let mixed_qkv_raw = qwen35_linear_cpu(
            &input_norm,
            &qkv_weight,
            shape.tokens,
            shape.hidden_size,
            attention_shape.conv_channels(),
        )
        .unwrap();
        let z_raw = qwen35_linear_cpu(
            &input_norm,
            &z_weight,
            shape.tokens,
            shape.hidden_size,
            attention_shape.value_total(),
        )
        .unwrap();
        let b_raw = qwen35_linear_cpu(
            &input_norm,
            &b_weight,
            shape.tokens,
            shape.hidden_size,
            attention_shape.value_heads,
        )
        .unwrap();
        let a_raw = qwen35_linear_cpu(
            &input_norm,
            &a_weight,
            shape.tokens,
            shape.hidden_size,
            attention_shape.value_heads,
        )
        .unwrap();
        let attention = qwen35_linear_attention_core_cpu(
            &mixed_qkv_raw,
            &z_raw,
            &a_raw,
            &b_raw,
            &conv1d_weight,
            &a_log,
            &dt_bias,
            &norm_weight,
            &initial_state,
            attention_shape,
            1e-6,
        )
        .unwrap();
        let delta_output = qwen35_linear_cpu(
            &attention.delta_norm,
            &out_proj_weight,
            shape.tokens,
            attention_shape.value_total(),
            shape.hidden_size,
        )
        .unwrap();
        let residual_after_mixer =
            add_same_len(&layer_input, &delta_output, "residual_after_mixer").unwrap();
        let post_attention_norm = qwen35_rms_norm_plus_one_cpu(
            &residual_after_mixer,
            &post_attention_norm_weight,
            shape.tokens,
            shape.hidden_size,
            1e-6,
        )
        .unwrap();
        let gate = qwen35_linear_cpu(
            &post_attention_norm,
            &gate_proj_weight,
            shape.tokens,
            shape.hidden_size,
            shape.intermediate_size,
        )
        .unwrap();
        let up = qwen35_linear_cpu(
            &post_attention_norm,
            &up_proj_weight,
            shape.tokens,
            shape.hidden_size,
            shape.intermediate_size,
        )
        .unwrap();
        let fused = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| silu(*gate) * up)
            .collect::<Vec<_>>();
        let mlp_output = qwen35_linear_cpu(
            &fused,
            &down_proj_weight,
            shape.tokens,
            shape.intermediate_size,
            shape.hidden_size,
        )
        .unwrap();
        let layer_output =
            add_same_len(&residual_after_mixer, &mlp_output, "layer_output").unwrap();

        let reference = qwen35_dense_linear_attention_layer_cpu(
            &layer_input,
            &input_norm_weight,
            &qkv_weight,
            &z_weight,
            &b_weight,
            &a_weight,
            &conv1d_weight,
            &a_log,
            &dt_bias,
            &norm_weight,
            &out_proj_weight,
            &post_attention_norm_weight,
            &gate_proj_weight,
            &up_proj_weight,
            &down_proj_weight,
            &initial_state,
            shape,
            1e-6,
        )
        .unwrap();

        assert_close_slice(&reference.input_norm, &input_norm, 1e-6);
        assert_close_slice(&reference.mixed_qkv_raw, &mixed_qkv_raw, 1e-6);
        assert_close_slice(&reference.z_raw, &z_raw, 1e-6);
        assert_close_slice(&reference.b_raw, &b_raw, 1e-6);
        assert_close_slice(&reference.a_raw, &a_raw, 1e-6);
        assert_eq!(reference.attention, attention);
        assert_close_slice(&reference.delta_output, &delta_output, 1e-6);
        assert_close_slice(&reference.residual_after_mixer, &residual_after_mixer, 1e-6);
        assert_close_slice(&reference.post_attention_norm, &post_attention_norm, 1e-6);
        assert_close_slice(&reference.mlp_output, &mlp_output, 1e-6);
        assert_close_slice(&reference.layer_output, &layer_output, 1e-6);
    }

    #[test]
    fn dense_linear_attention_layer_rejects_token_shape_mismatch() {
        let shape = Qwen35DenseLinearAttentionLayerShape {
            tokens: 2,
            hidden_size: 2,
            intermediate_size: 2,
            attention: Qwen35LinearAttentionShape {
                tokens: 1,
                key_heads: 1,
                value_heads: 1,
                key_dim: 1,
                value_dim: 1,
                conv_kernel: 1,
            },
        };
        let err = qwen35_dense_linear_attention_layer_cpu(
            &[0.0; 4],
            &[0.0; 2],
            &[0.0; 6],
            &[0.0; 2],
            &[0.0; 2],
            &[0.0; 2],
            &[1.0; 3],
            &[0.0],
            &[0.0],
            &[1.0],
            &[0.0; 2],
            &[0.0; 2],
            &[0.0; 4],
            &[0.0; 4],
            &[0.0; 4],
            &[0.0],
            shape,
            1e-6,
        )
        .expect_err("layer tokens should reject mismatched attention tokens");

        assert!(
            err.to_string().contains("must match attention tokens"),
            "{err}"
        );
    }

    #[test]
    fn dense_full_attention_layer_composes_attention_residual_and_dense_mlp() {
        let attention_shape = Qwen35FullAttentionShape {
            tokens: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            position_offset: 0,
            rope_theta: 10_000.0,
        };
        let shape = Qwen35DenseFullAttentionLayerShape {
            tokens: 2,
            hidden_size: 2,
            intermediate_size: 2,
            attention: attention_shape,
        };
        let layer_input = vec![1.0, 2.0, 3.0, 4.0];
        let input_norm_weight = vec![0.0, 0.0];
        let q_weight = vec![
            1.0, 0.0, //
            0.0, 1.0,
        ];
        let k_weight = vec![
            0.5, 0.0, //
            0.0, 0.5,
        ];
        let v_weight = vec![
            1.0, 1.0, //
            -0.5, 0.5,
        ];
        let q_norm_weight = vec![1.0, 1.0];
        let k_norm_weight = vec![1.0, 1.0];
        let o_weight = vec![
            1.0, 0.0, //
            0.0, 1.0,
        ];
        let post_attention_norm_weight = vec![0.0, 0.0];
        let gate_proj_weight = vec![
            0.2, 0.1, //
            -0.1, 0.3,
        ];
        let up_proj_weight = vec![
            0.4, -0.2, //
            0.3, 0.5,
        ];
        let down_proj_weight = vec![
            1.0, 0.0, //
            0.0, 1.0,
        ];

        let input_norm = qwen35_rms_norm_plus_one_cpu(
            &layer_input,
            &input_norm_weight,
            shape.tokens,
            shape.hidden_size,
            1e-6,
        )
        .unwrap();
        let query_raw = qwen35_linear_cpu(
            &input_norm,
            &q_weight,
            shape.tokens,
            shape.hidden_size,
            attention_shape.q_total(),
        )
        .unwrap();
        let key_raw = qwen35_linear_cpu(
            &input_norm,
            &k_weight,
            shape.tokens,
            shape.hidden_size,
            attention_shape.kv_total(),
        )
        .unwrap();
        let value_raw = qwen35_linear_cpu(
            &input_norm,
            &v_weight,
            shape.tokens,
            shape.hidden_size,
            attention_shape.kv_total(),
        )
        .unwrap();
        let attention = qwen35_full_attention_core_cpu(
            &query_raw,
            &key_raw,
            &value_raw,
            &q_norm_weight,
            &k_norm_weight,
            attention_shape,
            1e-6,
        )
        .unwrap();
        let attn_output = qwen35_linear_cpu(
            &attention.context,
            &o_weight,
            shape.tokens,
            attention_shape.q_total(),
            shape.hidden_size,
        )
        .unwrap();
        let residual_after_attention =
            add_same_len(&layer_input, &attn_output, "residual_after_attention").unwrap();
        let post_attention_norm = qwen35_rms_norm_plus_one_cpu(
            &residual_after_attention,
            &post_attention_norm_weight,
            shape.tokens,
            shape.hidden_size,
            1e-6,
        )
        .unwrap();
        let mlp_output = qwen35_dense_mlp_cpu(
            &post_attention_norm,
            &gate_proj_weight,
            &up_proj_weight,
            &down_proj_weight,
            shape.tokens,
            shape.hidden_size,
            shape.intermediate_size,
        )
        .unwrap();
        let layer_output =
            add_same_len(&residual_after_attention, &mlp_output, "layer_output").unwrap();

        let reference = qwen35_dense_full_attention_layer_cpu(
            &layer_input,
            &input_norm_weight,
            &q_weight,
            &k_weight,
            &v_weight,
            &q_norm_weight,
            &k_norm_weight,
            &o_weight,
            &post_attention_norm_weight,
            &gate_proj_weight,
            &up_proj_weight,
            &down_proj_weight,
            shape,
            1e-6,
        )
        .unwrap();

        assert_close_slice(&reference.input_norm, &input_norm, 1e-6);
        assert_close_slice(&reference.query_raw, &query_raw, 1e-6);
        assert_close_slice(&reference.key_raw, &key_raw, 1e-6);
        assert_close_slice(&reference.value_raw, &value_raw, 1e-6);
        assert_close_slice(&reference.attention.query, &attention.query, 1e-6);
        assert_close_slice(&reference.attention.key, &attention.key, 1e-6);
        assert_close_slice(&reference.attention.value, &attention.value, 1e-6);
        assert_close_slice(&reference.attention.context, &attention.context, 1e-6);
        assert_close_slice(&reference.attn_output, &attn_output, 1e-6);
        assert_close_slice(
            &reference.residual_after_attention,
            &residual_after_attention,
            1e-6,
        );
        assert_close_slice(&reference.post_attention_norm, &post_attention_norm, 1e-6);
        assert_close_slice(&reference.mlp_output, &mlp_output, 1e-6);
        assert_close_slice(&reference.layer_output, &layer_output, 1e-6);
    }

    #[test]
    fn dense_full_attention_layer_rejects_hidden_size_mismatch() {
        let shape = Qwen35DenseFullAttentionLayerShape {
            tokens: 1,
            hidden_size: 3,
            intermediate_size: 2,
            attention: Qwen35FullAttentionShape {
                tokens: 1,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
                position_offset: 0,
                rope_theta: 10_000.0,
            },
        };
        let err = qwen35_dense_full_attention_layer_cpu(
            &[0.0; 3], &[0.0; 3], &[0.0; 6], &[0.0; 6], &[0.0; 6], &[1.0; 2], &[1.0; 2], &[0.0; 6],
            &[0.0; 3], &[0.0; 6], &[0.0; 6], &[0.0; 6], shape, 1e-6,
        )
        .expect_err("hidden size must match attention q_total");

        assert!(err.to_string().contains("hidden_size 3"), "{err}");
    }

    #[test]
    fn sparse_moe_shared_expert_composes_router_fused_experts_and_shared_gate() {
        let shape = Qwen35SparseMoeShape {
            tokens: 2,
            hidden_size: 2,
            num_experts: 3,
            top_k: 2,
            expert_intermediate_size: 2,
            shared_expert_intermediate_size: 2,
            norm_topk_prob: true,
        };
        let x = vec![
            1.0, 0.0, //
            0.0, 1.0,
        ];
        let router_weight = vec![
            2.0, 0.0, //
            0.0, 2.0, //
            1.0, 1.0,
        ];
        let mut fused_gate_up_proj = Vec::new();
        let mut fused_down_proj = Vec::new();
        for expert in 0..shape.num_experts {
            let scale = expert as f32 + 1.0;
            fused_gate_up_proj.extend_from_slice(&[
                scale, 0.0, //
                0.0, scale,
            ]);
            fused_gate_up_proj.extend_from_slice(&[
                1.0, 0.5, //
                -0.25, 0.75,
            ]);
            fused_down_proj.extend_from_slice(&[
                0.5 * scale,
                0.0, //
                0.0,
                -0.25 * scale,
            ]);
        }
        let shared_expert_gate_weight = vec![0.0, 0.0];
        let shared_expert_gate_proj_weight = vec![
            0.2, -0.1, //
            0.3, 0.4,
        ];
        let shared_expert_up_proj_weight = vec![
            0.5, 0.1, //
            -0.2, 0.6,
        ];
        let shared_expert_down_proj_weight = vec![
            1.0, 0.0, //
            0.0, 1.0,
        ];

        let router_logits = qwen35_linear_cpu(
            &x,
            &router_weight,
            shape.tokens,
            shape.hidden_size,
            shape.num_experts,
        )
        .unwrap();
        let routed = crate::moe::router::route(
            &router_logits,
            shape.tokens,
            shape.num_experts,
            shape.top_k,
            shape.norm_topk_prob,
        );
        assert_eq!(routed.expert_ids, vec![0, 2, 1, 2]);

        let mut routed_expert_output = vec![0.0; shape.tokens * shape.hidden_size];
        for token_idx in 0..shape.tokens {
            let token = &x[token_idx * shape.hidden_size..(token_idx + 1) * shape.hidden_size];
            for slot in 0..shape.top_k {
                let pair_idx = token_idx * shape.top_k + slot;
                let expert_id = routed.expert_ids[pair_idx] as usize;
                let expert_output = qwen35_fused_expert_mlp_cpu(
                    token,
                    &fused_gate_up_proj,
                    &fused_down_proj,
                    shape,
                    expert_id,
                )
                .unwrap();
                for hidden_idx in 0..shape.hidden_size {
                    routed_expert_output[token_idx * shape.hidden_size + hidden_idx] +=
                        routed.expert_weights[pair_idx] * expert_output[hidden_idx];
                }
            }
        }
        let shared_dense = qwen35_dense_mlp_cpu(
            &x,
            &shared_expert_gate_proj_weight,
            &shared_expert_up_proj_weight,
            &shared_expert_down_proj_weight,
            shape.tokens,
            shape.hidden_size,
            shape.shared_expert_intermediate_size,
        )
        .unwrap();
        let shared_expert_output = shared_dense
            .iter()
            .map(|value| 0.5 * value)
            .collect::<Vec<_>>();
        let moe_output =
            add_same_len(&routed_expert_output, &shared_expert_output, "moe_output").unwrap();

        let reference = qwen35_sparse_moe_shared_expert_cpu(
            &x,
            &router_weight,
            &fused_gate_up_proj,
            &fused_down_proj,
            &shared_expert_gate_weight,
            &shared_expert_gate_proj_weight,
            &shared_expert_up_proj_weight,
            &shared_expert_down_proj_weight,
            shape,
        )
        .unwrap();

        assert_close_slice(&reference.router_logits, &router_logits, 1e-6);
        assert_eq!(reference.router_topk_indices, routed.expert_ids);
        assert_close_slice(&reference.router_topk_weights, &routed.expert_weights, 1e-6);
        assert_close_slice(&reference.routed_expert_output, &routed_expert_output, 1e-6);
        assert_close_slice(&reference.shared_expert_gate, &[0.5, 0.5], 1e-6);
        assert_close_slice(&reference.shared_expert_output, &shared_expert_output, 1e-6);
        assert_close_slice(&reference.moe_output, &moe_output, 1e-6);
    }

    #[test]
    fn sparse_moe_rejects_invalid_fused_expert_layout() {
        let shape = Qwen35SparseMoeShape {
            tokens: 1,
            hidden_size: 2,
            num_experts: 2,
            top_k: 1,
            expert_intermediate_size: 2,
            shared_expert_intermediate_size: 2,
            norm_topk_prob: true,
        };
        let err = qwen35_sparse_moe_shared_expert_cpu(
            &[1.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[0.0; 7],
            &[0.0; 8],
            &[0.0, 0.0],
            &[0.0; 4],
            &[0.0; 4],
            &[0.0; 4],
            shape,
        )
        .expect_err("fused gate/up tensor should be too short");

        assert!(
            err.to_string().contains("fused_gate_up_proj length"),
            "{err}"
        );
    }

    #[test]
    fn linear_attention_core_composes_conv_gdn_delta_and_norm() {
        let shape = Qwen35LinearAttentionShape {
            tokens: 2,
            key_heads: 1,
            value_heads: 1,
            key_dim: 2,
            value_dim: 2,
            conv_kernel: 1,
        };
        let mixed_qkv_raw = vec![
            1.0, 0.0, 1.0, 0.0, 2.0, 4.0, //
            0.0, 1.0, 0.0, 1.0, 3.0, 5.0,
        ];
        let conv1d_weight = vec![1.0; shape.conv_channels()];
        let z_raw = vec![0.0, 1.0, 2.0, 3.0];
        let a_raw = vec![0.0, 0.5];
        let b_raw = vec![0.0, 1.0];
        let a_log = vec![0.0];
        let dt_bias = vec![0.0];
        let norm_weight = vec![1.0, 2.0];
        let initial_state = vec![0.0; shape.state_len()];

        let mixed_qkv_conv = qwen35_depthwise_causal_conv_silu_cpu(
            &mixed_qkv_raw,
            &conv1d_weight,
            shape.tokens,
            shape.conv_channels(),
            shape.conv_kernel,
        )
        .unwrap();
        let (query, key, value) =
            qwen35_split_linear_attention_qkv_cpu(&mixed_qkv_conv, shape).unwrap();
        let (g, beta) = qwen35_gdn_gating_cpu(
            &a_log,
            &a_raw,
            &b_raw,
            &dt_bias,
            shape.tokens,
            shape.value_heads,
        )
        .unwrap();
        let (delta_core, final_state) = qwen35_recurrent_gated_delta_rule_cpu(
            &query,
            &key,
            &value,
            &g,
            &beta,
            &initial_state,
            shape.delta_shape(),
            true,
            None,
        )
        .unwrap();
        let delta_norm = qwen35_gated_rms_norm_cpu(
            &delta_core,
            &z_raw,
            &norm_weight,
            shape.tokens,
            shape.value_heads,
            shape.value_dim,
            1e-6,
        )
        .unwrap();

        let reference = qwen35_linear_attention_core_cpu(
            &mixed_qkv_raw,
            &z_raw,
            &a_raw,
            &b_raw,
            &conv1d_weight,
            &a_log,
            &dt_bias,
            &norm_weight,
            &initial_state,
            shape,
            1e-6,
        )
        .unwrap();

        assert_close_slice(&reference.mixed_qkv_conv, &mixed_qkv_conv, 1e-6);
        assert_close_slice(&reference.query, &query, 1e-6);
        assert_close_slice(&reference.key, &key, 1e-6);
        assert_close_slice(&reference.value, &value, 1e-6);
        assert_close_slice(&reference.g, &g, 1e-6);
        assert_close_slice(&reference.beta, &beta, 1e-6);
        assert_close_slice(&reference.delta_core, &delta_core, 1e-6);
        assert_close_slice(&reference.delta_norm, &delta_norm, 1e-6);
        assert_close_slice(&reference.final_state, &final_state, 1e-6);
    }

    #[test]
    fn linear_attention_core_rejects_shape_mismatch() {
        let shape = Qwen35LinearAttentionShape {
            tokens: 1,
            key_heads: 1,
            value_heads: 1,
            key_dim: 1,
            value_dim: 1,
            conv_kernel: 1,
        };
        let err = qwen35_linear_attention_core_cpu(
            &[0.0, 0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0, 1.0, 1.0],
            &[0.0],
            &[0.0],
            &[1.0],
            &[0.0],
            shape,
            1e-6,
        )
        .expect_err("mixed_qkv_raw should be too short");

        assert!(err.to_string().contains("mixed_qkv_raw length"), "{err}");
    }

    #[test]
    fn recurrent_delta_rule_single_token_updates_state() {
        let shape = Qwen35DeltaRuleShape {
            tokens: 1,
            key_heads: 1,
            value_heads: 1,
            key_dim: 2,
            value_dim: 2,
        };
        let query = vec![1.0, 0.0];
        let key = vec![1.0, 0.0];
        let value = vec![2.0, 4.0];
        let g = vec![0.0];
        let beta = vec![0.5];
        let initial_state = vec![0.0; 4];

        let (out, state) = qwen35_recurrent_gated_delta_rule_cpu(
            &query,
            &key,
            &value,
            &g,
            &beta,
            &initial_state,
            shape,
            false,
            Some(1.0),
        )
        .unwrap();

        assert_eq!(state, vec![1.0, 0.0, 2.0, 0.0]);
        assert_eq!(out, vec![1.0, 2.0]);
    }

    #[test]
    fn recurrent_delta_rule_repeats_qk_heads_over_value_heads() {
        let shape = Qwen35DeltaRuleShape {
            tokens: 1,
            key_heads: 1,
            value_heads: 2,
            key_dim: 1,
            value_dim: 1,
        };
        let query = vec![2.0];
        let key = vec![3.0];
        let value = vec![5.0, 7.0];
        let g = vec![0.0, 0.0];
        let beta = vec![1.0, 1.0];
        let initial_state = vec![0.0, 0.0];

        let (out, state) = qwen35_recurrent_gated_delta_rule_cpu(
            &query,
            &key,
            &value,
            &g,
            &beta,
            &initial_state,
            shape,
            false,
            Some(1.0),
        )
        .unwrap();

        assert_eq!(state, vec![15.0, 21.0]);
        assert_eq!(out, vec![30.0, 42.0]);
    }

    #[test]
    fn recurrent_delta_rule_rejects_shape_mismatch() {
        let shape = Qwen35DeltaRuleShape {
            tokens: 1,
            key_heads: 2,
            value_heads: 3,
            key_dim: 1,
            value_dim: 1,
        };
        let err = qwen35_recurrent_gated_delta_rule_cpu(
            &[1.0, 2.0],
            &[1.0, 2.0],
            &[1.0, 2.0, 3.0],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
            shape,
            false,
            Some(1.0),
        )
        .expect_err("value heads must divide key heads before update");

        assert!(err.to_string().contains("value_heads 3"), "{err}");
    }

    #[test]
    fn gdn_gating_matches_reference_formula() {
        let (g, beta) = qwen35_gdn_gating_cpu(
            &[0.0, (2.0f32).ln()],
            &[0.0, 1.0, -1.0, 2.0],
            &[0.0, 1.0, -1.0, 2.0],
            &[0.0, 0.5],
            2,
            2,
        )
        .unwrap();
        let expected_g = [
            -softplus(0.0),
            -2.0 * softplus(1.5),
            -softplus(-1.0),
            -2.0 * softplus(2.5),
        ];
        let expected_beta = [sigmoid(0.0), sigmoid(1.0), sigmoid(-1.0), sigmoid(2.0)];

        for (got, expected) in g.iter().zip(expected_g) {
            assert!((got - expected).abs() < 1e-6, "{got} != {expected}");
        }
        for (got, expected) in beta.iter().zip(expected_beta) {
            assert!((got - expected).abs() < 1e-6, "{got} != {expected}");
        }
    }

    #[test]
    fn gdn_gating_rejects_shape_mismatch() {
        let err = qwen35_gdn_gating_cpu(&[0.0], &[0.0, 1.0], &[0.0], &[0.0], 2, 1)
            .expect_err("b length should fail");

        assert!(err.to_string().contains("b length"), "{err}");
    }

    #[test]
    fn gated_delta_attention_matches_gating_plus_recurrent_reference() {
        let shape = Qwen35DeltaRuleShape {
            tokens: 2,
            key_heads: 1,
            value_heads: 1,
            key_dim: 2,
            value_dim: 1,
        };
        let query = vec![1.0, 0.0, 0.0, 1.0];
        let key = vec![1.0, 0.0, 0.0, 1.0];
        let value = vec![2.0, 4.0];
        let a = vec![0.0, 0.5];
        let b = vec![0.0, 1.0];
        let a_log = vec![0.0];
        let dt_bias = vec![0.0];
        let initial_state = vec![0.0; 2];
        let (g, beta) = qwen35_gdn_gating_cpu(&a_log, &a, &b, &dt_bias, 2, 1).unwrap();
        let (expected_out, expected_state) = qwen35_recurrent_gated_delta_rule_cpu(
            &query,
            &key,
            &value,
            &g,
            &beta,
            &initial_state,
            shape,
            false,
            Some(1.0),
        )
        .unwrap();

        let (out, state) = qwen35_gated_delta_attention_cpu(
            &query,
            &key,
            &value,
            &a,
            &b,
            &a_log,
            &dt_bias,
            &initial_state,
            shape,
            false,
            Some(1.0),
        )
        .unwrap();

        assert_eq!(out, expected_out);
        assert_eq!(state, expected_state);
    }
}
