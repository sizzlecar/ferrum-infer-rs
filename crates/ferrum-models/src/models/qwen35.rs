//! Qwen3.5 / Qwen3.6 typed model weights.
//!
//! This is the W3 materialization boundary: it turns the resolved semantic
//! weight plan into backend-native buffers, linears, and product forward
//! execution for the Qwen3.5/Qwen3.6 gated-DeltaNet family.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ferrum_interfaces::{
    kv_dtype::KvFp16,
    model_executor::{GreedyRepetitionPenalty, LogitsReturnPolicy, TokenSelectionMask},
    RecurrentStateHandle, RecurrentStateHandleStats, RecurrentStateManager,
    RecurrentStateManagerStats, RecurrentStateSpec,
};
use ferrum_kernels::backend::{AttnConfig, Backend, BackendPagedKv, Dtype, KvCache, MoeLlmBackend};
use ferrum_quantization::{Linear, NativeSafetensorsLoader, StackedExpertLinear, WeightLoader};
use ferrum_types::{DataType, Device, FerrumError, RequestId, Result};
use parking_lot::{Mutex, MutexGuard};

use crate::{
    common::{paged_pool::BlockAllocator, DecoderOnlyLLM, LlmRuntimeConfig},
    definition::ModelDefinition,
    moe::{moe_forward_bucketed, ExpertStack, MoeForwardBucketedParams, MoeRouteScratch},
    qwen35_config::{Qwen35LayerType, Qwen35MlpKind, Qwen35TextConfig},
    qwen35_weights::{
        Qwen35ResolvedWeightPlan, Qwen35WeightInventory, Qwen35WeightPlanLoader,
        Qwen35WeightValidation,
    },
};

pub struct Qwen35ModelWeights<B: MoeLlmBackend> {
    pub config: Qwen35TextConfig,
    pub runtime_cfg: LlmRuntimeConfig,
    pub embed_tokens: B::Buffer,
    pub final_norm: B::Buffer,
    pub lm_head: Box<dyn Linear<B>>,
    pub layers: Vec<Qwen35LayerWeights<B>>,
}

pub struct Qwen35BackendModel<B: MoeLlmBackend> {
    pub weights: Qwen35ModelWeights<B>,
    pub weight_plan: Qwen35ResolvedWeightPlan,
    pub weight_validation: Qwen35WeightValidation,
    pub rope: Qwen35BackendRopeCache<B>,
    pub sequences: HashMap<String, Qwen35SequenceState<B>>,
    pub kv_capacity: usize,
    pub use_paged_kv: bool,
    pub use_vllm_paged_attn: bool,
    pub paged_pools: Option<Vec<(B::Buffer, B::Buffer)>>,
    pub paged_block_alloc: Option<std::sync::Mutex<BlockAllocator>>,
    pub paged_max_seqs: usize,
    pub paged_max_blocks_per_seq: usize,
    paged_scratch: Qwen35PagedScratch<B>,
    decode_scratch: Option<Qwen35DecodeScratch<B>>,
    linear_state_pools: Option<Qwen35LinearStatePools<B>>,
    linear_free_slots: Vec<usize>,
    linear_slot_indices: Option<B::Buffer>,
    linear_slot_indices_capacity: usize,
    pub greedy_argmax: bool,
    argmax_token_mask: Option<B::Buffer>,
    argmax_token_mask_fingerprint: Option<u64>,
    argmax_token_mask_len: usize,
    argmax_repetition_offsets: Option<B::Buffer>,
    argmax_repetition_token_ids: Option<B::Buffer>,
    argmax_repetition_penalties: Option<B::Buffer>,
    argmax_repetition_offsets_capacity: usize,
    argmax_repetition_token_ids_capacity: usize,
    argmax_repetition_penalties_capacity: usize,
}

pub struct Qwen35LayerWeights<B: MoeLlmBackend> {
    pub layer_index: usize,
    pub input_layernorm: B::Buffer,
    pub post_attention_layernorm: B::Buffer,
    pub attention: Qwen35AttentionWeights<B>,
    pub mlp: Qwen35MlpWeights<B>,
}

pub enum Qwen35AttentionWeights<B: MoeLlmBackend> {
    Linear(Qwen35LinearAttentionWeights<B>),
    Full(Qwen35FullAttentionWeights<B>),
}

pub struct Qwen35LinearAttentionWeights<B: MoeLlmBackend> {
    pub qkv_proj: Box<dyn Linear<B>>,
    pub z_proj: Box<dyn Linear<B>>,
    pub b_proj: Box<dyn Linear<B>>,
    pub a_proj: Box<dyn Linear<B>>,
    pub qkvz_proj: Option<Box<dyn Linear<B>>>,
    pub ba_proj: Option<Box<dyn Linear<B>>>,
    pub conv1d_weight: B::Buffer,
    pub a_log: B::Buffer,
    pub dt_bias: B::Buffer,
    pub norm_weight: B::Buffer,
    pub out_proj: Box<dyn Linear<B>>,
}

pub struct Qwen35FullAttentionWeights<B: MoeLlmBackend> {
    pub q_proj: Box<dyn Linear<B>>,
    pub k_proj: Box<dyn Linear<B>>,
    pub v_proj: Box<dyn Linear<B>>,
    pub o_proj: Box<dyn Linear<B>>,
    pub q_norm_weight: B::Buffer,
    pub k_norm_weight: B::Buffer,
}

enum Qwen35LinearDecodeProjectionBuffers<T> {
    Separate {
        mixed_qkv_raw: T,
        z_raw: T,
        b_raw: T,
        a_raw: T,
    },
    Packed {
        mixed_qkvz_raw: T,
        ba_raw: T,
        z_raw: T,
    },
}

pub enum Qwen35MlpWeights<B: MoeLlmBackend> {
    Dense(Qwen35DenseMlpWeights<B>),
    SparseMoeSharedExpert(Qwen35SparseMoeSharedExpertWeights<B>),
}

pub struct Qwen35DenseMlpWeights<B: MoeLlmBackend> {
    pub gate_up_proj: Box<dyn Linear<B>>,
    pub down_proj: Box<dyn Linear<B>>,
}

pub struct Qwen35SparseMoeSharedExpertWeights<B: MoeLlmBackend> {
    pub router: Box<dyn Linear<B>>,
    pub experts: ExpertStack<B>,
    pub shared_expert_gate: B::Buffer,
    pub shared_expert_gate_proj: Box<dyn Linear<B>>,
    pub shared_expert_up_proj: Box<dyn Linear<B>>,
    pub shared_expert_down_proj: Box<dyn Linear<B>>,
    pub fused_gate_up_proj: B::Buffer,
    pub fused_down_proj: B::Buffer,
}

pub struct Qwen35SequenceState<B: MoeLlmBackend> {
    pub tokens: Vec<u32>,
    pub layers: Vec<Qwen35LayerRuntimeState<B>>,
    pub linear_slot: Option<usize>,
}

pub enum Qwen35LayerRuntimeState<B: MoeLlmBackend> {
    Linear {
        conv_state: B::Buffer,
        delta_state: B::Buffer,
    },
    Full {
        kv: KvCache<B, KvFp16>,
    },
}

pub struct Qwen35LinearStatePools<B: MoeLlmBackend> {
    pub conv_states: Vec<Option<B::Buffer>>,
    pub delta_states: Vec<Option<B::Buffer>>,
    pub max_slots: usize,
    pub conv_state_len: usize,
    pub delta_state_len: usize,
}

pub struct Qwen35RecurrentStateCache<B: Backend> {
    pub request_id: RequestId,
    pub num_layers: usize,
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

pub struct Qwen35RecurrentStateHandle<B: Backend> {
    cache: Arc<Mutex<Qwen35RecurrentStateCache<B>>>,
    cache_id: String,
    valid: Arc<AtomicBool>,
    created_at: Instant,
}

#[derive(Debug, Clone)]
pub struct Qwen35RecurrentStateManagerConfig {
    pub total_memory_bytes: usize,
    pub total_batch_slots: usize,
}

pub struct Qwen35RecurrentStateManager<B: Backend> {
    config: Qwen35RecurrentStateManagerConfig,
    handles: Mutex<HashMap<RequestId, Arc<Qwen35RecurrentStateHandle<B>>>>,
    allocation_count: AtomicU64,
    allocation_failures: AtomicU64,
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
    pub final_conv_state: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub g: Vec<f32>,
    pub beta: Vec<f32>,
    pub delta_core: Vec<f32>,
    pub delta_norm: Vec<f32>,
    pub final_state: Vec<f32>,
}

pub struct Qwen35BackendDeltaRuleOutput<B: Backend> {
    pub output: B::Buffer,
    pub final_state: B::Buffer,
}

pub struct Qwen35BackendVarlenDeltaRuleOutput<B: Backend> {
    pub output: B::Buffer,
    pub final_states: B::Buffer,
}

pub struct Qwen35BackendLinearAttentionPrefillOutput<B: Backend> {
    pub query: B::Buffer,
    pub key: B::Buffer,
    pub value: B::Buffer,
    pub g: B::Buffer,
    pub beta: B::Buffer,
    pub delta_core: B::Buffer,
    pub delta_norm: B::Buffer,
    pub final_state: B::Buffer,
}

pub struct Qwen35BackendLinearAttentionVarlenPrefillOutput<B: Backend> {
    pub query: B::Buffer,
    pub key: B::Buffer,
    pub value: B::Buffer,
    pub g: B::Buffer,
    pub beta: B::Buffer,
    pub final_conv_states: B::Buffer,
    pub delta_core: B::Buffer,
    pub delta_norm: B::Buffer,
    pub final_states: B::Buffer,
}

pub struct Qwen35BackendLinearAttentionDecodeOutput<B: Backend> {
    pub query: B::Buffer,
    pub key: B::Buffer,
    pub value: B::Buffer,
    pub g: B::Buffer,
    pub beta: B::Buffer,
    pub next_conv_state: B::Buffer,
    pub delta_core: B::Buffer,
    pub delta_norm: B::Buffer,
    pub final_state: B::Buffer,
}

pub struct Qwen35BackendRopeCache<B: Backend> {
    pub cos: B::Buffer,
    pub sin: B::Buffer,
}

pub struct Qwen35PagedScratch<B: MoeLlmBackend> {
    cu_seqlens_q: Option<B::Buffer>,
    pos_offsets: Option<B::Buffer>,
    block_tables: Option<B::Buffer>,
    context_lens: Option<B::Buffer>,
    q: Option<B::Buffer>,
    out: Option<B::Buffer>,
    max_seqs: usize,
    max_tokens: usize,
    max_blocks_per_seq: usize,
}

pub struct Qwen35DecodeScratch<B: MoeLlmBackend> {
    max_tokens: usize,
    hidden_size: usize,
    num_experts: usize,
    top_k: usize,
    expert_intermediate_size: usize,
    shared_expert_intermediate_size: usize,
    post_attention_norm: B::Buffer,
    mlp_output: B::Buffer,
    router_logits: B::Buffer,
    routed_output: B::Buffer,
    x_packed: B::Buffer,
    gate_up_packed: B::Buffer,
    silu_packed: B::Buffer,
    down_packed: B::Buffer,
    shared_gate: B::Buffer,
    shared_gate_up: B::Buffer,
    shared_gate_proj: B::Buffer,
    shared_up_proj: B::Buffer,
    shared_fused: B::Buffer,
    shared_output: B::Buffer,
    route_selected_ids: B::Buffer,
    route_pair_weights: B::Buffer,
    route_pairs_by_token: B::Buffer,
    route_packed_token_idx: B::Buffer,
    route_expert_offsets: B::Buffer,
    route_sorted_tokens: B::Buffer,
    route_block_ids: B::Buffer,
    route_total_post_pad: B::Buffer,
    route_scratch: MoeRouteScratch,
}

pub struct Qwen35BackendFullAttentionOutput<B: Backend> {
    pub query_head_major: B::Buffer,
    pub key_head_major: B::Buffer,
    pub value_head_major: B::Buffer,
    pub context_head_major: B::Buffer,
    pub context: B::Buffer,
}

pub struct Qwen35BackendDenseMlpOutput<B: Backend> {
    pub gate_up: B::Buffer,
    pub fused: B::Buffer,
    pub output: B::Buffer,
}

pub struct Qwen35BackendSparseMoeOutput<B: MoeLlmBackend> {
    pub router_logits: B::Buffer,
    pub routed_output: B::Buffer,
    pub shared_gate: B::Buffer,
    pub shared_gate_up: B::Buffer,
    pub shared_fused: B::Buffer,
    pub shared_output: B::Buffer,
    pub output: B::Buffer,
}

pub struct Qwen35BackendDenseFullAttentionLayerOutput<B: Backend> {
    pub input_norm: B::Buffer,
    pub query_raw: B::Buffer,
    pub key_raw: B::Buffer,
    pub value_raw: B::Buffer,
    pub attention: Qwen35BackendFullAttentionOutput<B>,
    pub attn_output: B::Buffer,
    pub residual_after_attention: B::Buffer,
    pub post_attention_norm: B::Buffer,
    pub mlp: Qwen35BackendDenseMlpOutput<B>,
    pub layer_output: B::Buffer,
}

pub struct Qwen35BackendDenseLinearAttentionLayerOutput<B: Backend> {
    pub input_norm: B::Buffer,
    pub mixed_qkv_raw: B::Buffer,
    pub z_raw: B::Buffer,
    pub b_raw: B::Buffer,
    pub a_raw: B::Buffer,
    pub attention: Qwen35BackendLinearAttentionPrefillOutput<B>,
    pub final_conv_state: B::Buffer,
    pub delta_output: B::Buffer,
    pub residual_after_mixer: B::Buffer,
    pub post_attention_norm: B::Buffer,
    pub mlp: Qwen35BackendDenseMlpOutput<B>,
    pub layer_output: B::Buffer,
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
    pub rope_dim: usize,
    pub position_offset: usize,
    pub rope_theta: f32,
    pub rope_interleaved: bool,
    pub attn_output_gate: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35FullAttentionReference {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub attention_gate: Option<Vec<f32>>,
    pub context_ungated: Vec<f32>,
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
pub struct Qwen35SparseMoeLinearAttentionLayerShape {
    pub tokens: usize,
    pub hidden_size: usize,
    pub attention: Qwen35LinearAttentionShape,
    pub moe: Qwen35SparseMoeShape,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35SparseMoeLinearAttentionLayerReference {
    pub input_norm: Vec<f32>,
    pub mixed_qkv_raw: Vec<f32>,
    pub z_raw: Vec<f32>,
    pub b_raw: Vec<f32>,
    pub a_raw: Vec<f32>,
    pub attention: Qwen35LinearAttentionReference,
    pub delta_output: Vec<f32>,
    pub residual_after_mixer: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub moe: Qwen35SparseMoeReference,
    pub layer_output: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen35SparseMoeFullAttentionLayerShape {
    pub tokens: usize,
    pub hidden_size: usize,
    pub attention: Qwen35FullAttentionShape,
    pub moe: Qwen35SparseMoeShape,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35SparseMoeFullAttentionLayerReference {
    pub input_norm: Vec<f32>,
    pub query_raw: Vec<f32>,
    pub key_raw: Vec<f32>,
    pub value_raw: Vec<f32>,
    pub attention: Qwen35FullAttentionReference,
    pub attn_output: Vec<f32>,
    pub residual_after_attention: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub moe: Qwen35SparseMoeReference,
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

#[derive(Debug, Clone, Copy)]
pub struct Qwen35DenseReferenceModel<'a> {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub eps: f32,
    pub embed_tokens: &'a [f32],
    pub final_norm_weight: &'a [f32],
    pub lm_head_weight: &'a [f32],
    pub layers: &'a [Qwen35DenseReferenceLayer<'a>],
}

#[derive(Debug, Clone, Copy)]
pub enum Qwen35DenseReferenceLayer<'a> {
    Linear(Qwen35DenseReferenceLinearLayer<'a>),
    Full(Qwen35DenseReferenceFullLayer<'a>),
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen35DenseReferenceLinearLayer<'a> {
    pub shape: Qwen35DenseLinearAttentionLayerShape,
    pub input_norm_weight: &'a [f32],
    pub qkv_weight: &'a [f32],
    pub z_weight: &'a [f32],
    pub b_weight: &'a [f32],
    pub a_weight: &'a [f32],
    pub conv1d_weight: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm_weight: &'a [f32],
    pub out_proj_weight: &'a [f32],
    pub post_attention_norm_weight: &'a [f32],
    pub gate_proj_weight: &'a [f32],
    pub up_proj_weight: &'a [f32],
    pub down_proj_weight: &'a [f32],
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen35DenseReferenceFullLayer<'a> {
    pub shape: Qwen35DenseFullAttentionLayerShape,
    pub input_norm_weight: &'a [f32],
    pub q_weight: &'a [f32],
    pub k_weight: &'a [f32],
    pub v_weight: &'a [f32],
    pub q_norm_weight: &'a [f32],
    pub k_norm_weight: &'a [f32],
    pub o_weight: &'a [f32],
    pub post_attention_norm_weight: &'a [f32],
    pub gate_proj_weight: &'a [f32],
    pub up_proj_weight: &'a [f32],
    pub down_proj_weight: &'a [f32],
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen35SparseMoeReferenceModel<'a> {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub eps: f32,
    pub embed_tokens: &'a [f32],
    pub final_norm_weight: &'a [f32],
    pub lm_head_weight: &'a [f32],
    pub layers: &'a [Qwen35SparseMoeReferenceLayer<'a>],
}

#[derive(Debug, Clone, Copy)]
pub enum Qwen35SparseMoeReferenceLayer<'a> {
    Linear(Qwen35SparseMoeReferenceLinearLayer<'a>),
    Full(Qwen35SparseMoeReferenceFullLayer<'a>),
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen35SparseMoeReferenceLinearLayer<'a> {
    pub shape: Qwen35SparseMoeLinearAttentionLayerShape,
    pub input_norm_weight: &'a [f32],
    pub qkv_weight: &'a [f32],
    pub z_weight: &'a [f32],
    pub b_weight: &'a [f32],
    pub a_weight: &'a [f32],
    pub conv1d_weight: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm_weight: &'a [f32],
    pub out_proj_weight: &'a [f32],
    pub post_attention_norm_weight: &'a [f32],
    pub router_weight: &'a [f32],
    pub fused_gate_up_proj: &'a [f32],
    pub fused_down_proj: &'a [f32],
    pub shared_expert_gate_weight: &'a [f32],
    pub shared_expert_gate_proj_weight: &'a [f32],
    pub shared_expert_up_proj_weight: &'a [f32],
    pub shared_expert_down_proj_weight: &'a [f32],
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen35SparseMoeReferenceFullLayer<'a> {
    pub shape: Qwen35SparseMoeFullAttentionLayerShape,
    pub input_norm_weight: &'a [f32],
    pub q_weight: &'a [f32],
    pub k_weight: &'a [f32],
    pub v_weight: &'a [f32],
    pub q_norm_weight: &'a [f32],
    pub k_norm_weight: &'a [f32],
    pub o_weight: &'a [f32],
    pub post_attention_norm_weight: &'a [f32],
    pub router_weight: &'a [f32],
    pub fused_gate_up_proj: &'a [f32],
    pub fused_down_proj: &'a [f32],
    pub shared_expert_gate_weight: &'a [f32],
    pub shared_expert_gate_proj_weight: &'a [f32],
    pub shared_expert_up_proj_weight: &'a [f32],
    pub shared_expert_down_proj_weight: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35DenseReferenceModelOutput {
    pub hidden: Vec<f32>,
    pub final_hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub layer_hidden_states: Vec<Vec<f32>>,
    pub linear_conv_states: Vec<Vec<f32>>,
    pub linear_recurrent_states: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35SparseMoeReferenceModelOutput {
    pub hidden: Vec<f32>,
    pub final_hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub layer_hidden_states: Vec<Vec<f32>>,
    pub linear_conv_states: Vec<Vec<f32>>,
    pub linear_recurrent_states: Vec<Vec<f32>>,
    pub sparse_moe_outputs: Vec<Qwen35SparseMoeReference>,
}

#[derive(Debug, Clone)]
enum Qwen35DecodeLogitsReturn<'a> {
    Full,
    LegacyDefault,
    GreedyArgmax {
        token_mask: Option<&'a TokenSelectionMask>,
        repetition_penalties: Vec<Option<&'a GreedyRepetitionPenalty>>,
    },
}

#[derive(Debug, Clone, Copy)]
enum Qwen35ArgmaxMode<'a> {
    Raw,
    Masked(&'a TokenSelectionMask),
    SparseRepetition {
        token_mask: Option<&'a TokenSelectionMask>,
        repetition_penalties: &'a [Option<&'a GreedyRepetitionPenalty>],
    },
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

const QWEN35_DEFAULT_KV_CAPACITY: usize = 512;
const QWEN35_PAGED_BLOCK_SIZE: usize = 16;

impl<B: MoeLlmBackend> Qwen35PagedScratch<B> {
    fn new() -> Self {
        Self {
            cu_seqlens_q: None,
            pos_offsets: None,
            block_tables: None,
            context_lens: None,
            q: None,
            out: None,
            max_seqs: 0,
            max_tokens: 0,
            max_blocks_per_seq: 0,
        }
    }

    fn ensure(
        &mut self,
        total_tokens: usize,
        num_seqs: usize,
        max_blocks_per_seq: usize,
        q_total: usize,
    ) {
        let grow_tables = self.cu_seqlens_q.is_none()
            || self.max_seqs < num_seqs
            || self.max_blocks_per_seq < max_blocks_per_seq;
        let grow_tokens = self.q.is_none() || self.max_tokens < total_tokens;
        if grow_tables {
            self.cu_seqlens_q = Some(B::alloc_typed(Dtype::U32, num_seqs + 1));
            self.pos_offsets = Some(B::alloc_typed(Dtype::U32, num_seqs));
            self.context_lens = Some(B::alloc_typed(Dtype::U32, num_seqs));
            self.block_tables = Some(B::alloc_typed(Dtype::U32, num_seqs * max_blocks_per_seq));
            self.max_seqs = num_seqs;
            self.max_blocks_per_seq = max_blocks_per_seq;
        }
        if grow_tokens {
            self.q = Some(B::alloc(total_tokens * q_total));
            self.out = Some(B::alloc(total_tokens * q_total));
            self.max_tokens = total_tokens;
        }
    }
}

impl<B: MoeLlmBackend> Qwen35DecodeScratch<B> {
    fn alloc(config: &Qwen35TextConfig, max_tokens: usize) -> Self {
        let max_tokens = max_tokens.max(1);
        let hidden_size = config.hidden_size.max(1);
        let (num_experts, top_k, expert_intermediate_size, shared_expert_intermediate_size) =
            if let Some(moe) = config.moe.as_ref() {
                (
                    moe.num_experts.max(1),
                    moe.num_experts_per_tok.max(1),
                    moe.moe_intermediate_size.max(1),
                    moe.shared_expert_intermediate_size.max(1),
                )
            } else {
                (1, 1, config.dense_intermediate_size.unwrap_or(1).max(1), 1)
            };
        let total_pairs = max_tokens * top_k;
        Self {
            max_tokens,
            hidden_size,
            num_experts,
            top_k,
            expert_intermediate_size,
            shared_expert_intermediate_size,
            post_attention_norm: B::alloc(max_tokens * hidden_size),
            mlp_output: B::alloc(max_tokens * hidden_size),
            router_logits: B::alloc(max_tokens * num_experts),
            routed_output: B::alloc(max_tokens * hidden_size),
            x_packed: B::alloc(total_pairs * hidden_size),
            gate_up_packed: B::alloc(total_pairs * 2 * expert_intermediate_size),
            silu_packed: B::alloc(total_pairs * expert_intermediate_size),
            down_packed: B::alloc(total_pairs * hidden_size),
            shared_gate: B::alloc(max_tokens),
            shared_gate_up: B::alloc(max_tokens * 2 * shared_expert_intermediate_size),
            shared_gate_proj: B::alloc(max_tokens * shared_expert_intermediate_size),
            shared_up_proj: B::alloc(max_tokens * shared_expert_intermediate_size),
            shared_fused: B::alloc(max_tokens * shared_expert_intermediate_size),
            shared_output: B::alloc(max_tokens * hidden_size),
            route_selected_ids: B::alloc_typed(Dtype::I32, total_pairs),
            route_pair_weights: B::alloc_typed(Dtype::F32, total_pairs),
            route_pairs_by_token: B::alloc_typed(Dtype::I32, total_pairs),
            route_packed_token_idx: B::alloc_typed(Dtype::I32, total_pairs),
            route_expert_offsets: B::alloc_typed(Dtype::I32, num_experts + 1),
            route_sorted_tokens: B::alloc_typed(
                Dtype::I32,
                total_pairs + num_experts * crate::moe::dispatch::MOE_BLOCK_SIZE_MAX,
            ),
            route_block_ids: B::alloc_typed(Dtype::I32, total_pairs / 16 + num_experts + 1),
            route_total_post_pad: B::alloc_typed(Dtype::I32, 1),
            route_scratch: MoeRouteScratch::new(),
        }
    }

    fn covers(&self, config: &Qwen35TextConfig, tokens: usize) -> bool {
        let (num_experts, top_k, expert_intermediate_size, shared_expert_intermediate_size) =
            if let Some(moe) = config.moe.as_ref() {
                (
                    moe.num_experts.max(1),
                    moe.num_experts_per_tok.max(1),
                    moe.moe_intermediate_size.max(1),
                    moe.shared_expert_intermediate_size.max(1),
                )
            } else {
                (1, 1, config.dense_intermediate_size.unwrap_or(1).max(1), 1)
            };
        self.max_tokens >= tokens.max(1)
            && self.hidden_size == config.hidden_size.max(1)
            && self.num_experts == num_experts
            && self.top_k == top_k
            && self.expert_intermediate_size == expert_intermediate_size
            && self.shared_expert_intermediate_size == shared_expert_intermediate_size
    }
}

fn qwen35_effective_max_seq_len_from_snapshot(
    snapshot: &ferrum_types::RuntimeConfigSnapshot,
    model_max_seq_len: usize,
) -> usize {
    let requested = qwen35_runtime_positive_usize(snapshot, "FERRUM_KV_CAPACITY")
        .or_else(|| qwen35_runtime_positive_usize(snapshot, "FERRUM_MAX_MODEL_LEN"));
    requested
        .map(|cap| cap.min(model_max_seq_len))
        .unwrap_or_else(|| model_max_seq_len.min(QWEN35_DEFAULT_KV_CAPACITY))
        .max(1)
}

fn qwen35_effective_max_seq_len(model_max_seq_len: usize) -> usize {
    qwen35_effective_max_seq_len_from_snapshot(
        &ferrum_types::active_runtime_snapshot(),
        model_max_seq_len,
    )
}

fn qwen35_runtime_positive_usize(
    snapshot: &ferrum_types::RuntimeConfigSnapshot,
    key: &str,
) -> Option<usize> {
    snapshot
        .entries
        .iter()
        .find(|entry| entry.key == key)
        .and_then(|entry| entry.effective_value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn qwen35_runtime_bool(snapshot: &ferrum_types::RuntimeConfigSnapshot, key: &str) -> Option<bool> {
    snapshot
        .entries
        .iter()
        .find(|entry| entry.key == key)
        .and_then(|entry| match entry.effective_value.as_str() {
            "1" | "true" | "TRUE" | "True" => Some(true),
            "0" | "false" | "FALSE" | "False" => Some(false),
            _ => None,
        })
}

fn qwen35_paged_max_seqs(snapshot: &ferrum_types::RuntimeConfigSnapshot) -> usize {
    qwen35_runtime_positive_usize(snapshot, "FERRUM_PAGED_MAX_SEQS").unwrap_or(32)
}

fn qwen35_paged_total_blocks(
    snapshot: &ferrum_types::RuntimeConfigSnapshot,
    max_blocks_per_seq: usize,
    max_seqs: usize,
) -> usize {
    qwen35_runtime_positive_usize(snapshot, "FERRUM_KV_MAX_BLOCKS")
        .unwrap_or_else(|| max_seqs * max_blocks_per_seq)
        .max(1)
}

fn qwen35_first_full_kv<B: MoeLlmBackend>(
    state: &Qwen35SequenceState<B>,
) -> Option<&KvCache<B, KvFp16>> {
    state
        .layers
        .iter()
        .find_map(|layer_state| match layer_state {
            Qwen35LayerRuntimeState::Full { kv } => Some(kv),
            Qwen35LayerRuntimeState::Linear { .. } => None,
        })
}

fn qwen35_decode_logits_return_from_policies<'a>(
    policies: &'a [LogitsReturnPolicy],
    batch_len: usize,
) -> Qwen35DecodeLogitsReturn<'a> {
    if policies.len() != batch_len {
        return Qwen35DecodeLogitsReturn::Full;
    }

    let mut selected_mask: Option<&'a TokenSelectionMask> = None;
    let mut saw_unmasked = false;
    let mut repetition_penalties = Vec::with_capacity(batch_len);
    for policy in policies {
        match policy {
            LogitsReturnPolicy::FullLogits => return Qwen35DecodeLogitsReturn::Full,
            LogitsReturnPolicy::GreedyArgmax {
                token_mask,
                repetition_penalty,
            } => {
                repetition_penalties.push(repetition_penalty.as_ref());
                match token_mask.as_ref() {
                    Some(mask) => {
                        if saw_unmasked {
                            return Qwen35DecodeLogitsReturn::Full;
                        }
                        if let Some(selected) = selected_mask {
                            if selected.fingerprint != mask.fingerprint
                                || selected.len() != mask.len()
                            {
                                return Qwen35DecodeLogitsReturn::Full;
                            }
                        } else {
                            selected_mask = Some(mask);
                        }
                    }
                    None => {
                        if selected_mask.is_some() {
                            return Qwen35DecodeLogitsReturn::Full;
                        }
                        saw_unmasked = true;
                    }
                }
            }
        }
    }

    Qwen35DecodeLogitsReturn::GreedyArgmax {
        token_mask: selected_mask,
        repetition_penalties,
    }
}

fn qwen35_has_repetition_penalties(penalties: &[Option<&GreedyRepetitionPenalty>]) -> bool {
    penalties
        .iter()
        .flatten()
        .any(|penalty| !penalty.is_empty())
}

pub fn qwen35_runtime_config_from_definition(def: &ModelDefinition) -> Result<LlmRuntimeConfig> {
    let config =
        Qwen35TextConfig::from_model_definition(def).map_err(ferrum_types::FerrumError::model)?;
    Ok(qwen35_runtime_config(
        &config,
        def.vocab_size,
        qwen35_effective_max_seq_len(def.max_position_embeddings),
    ))
}

impl<B: MoeLlmBackend + BackendPagedKv> Qwen35BackendModel<B> {
    pub fn from_weight_plan(
        config: Qwen35TextConfig,
        runtime_cfg: LlmRuntimeConfig,
        weight_plan: Qwen35ResolvedWeightPlan,
        loader: &dyn WeightLoader<B>,
    ) -> Result<Self> {
        let weight_validation = weight_plan.validation();
        if !weight_validation.is_pass() {
            return Err(FerrumError::model(format!(
                "Qwen3.5 backend model weight plan is incomplete for prefix {}: {} missing \
                 required tensors",
                weight_validation.prefix,
                weight_validation.missing_required.len()
            )));
        }
        let rope = qwen35_build_rope_cache_backend::<B>(
            runtime_cfg.max_seq_len,
            config.full_attention_rope_dim(),
            config.rope_parameters.rope_theta as f32,
        )?;
        let kv_capacity = runtime_cfg.max_seq_len;
        let weights = Qwen35ModelWeights::<B>::load(config, runtime_cfg, &weight_plan, loader)?;
        let snapshot = ferrum_types::active_runtime_snapshot();
        let greedy_argmax = qwen35_runtime_bool(&snapshot, "FERRUM_GREEDY_ARGMAX").unwrap_or(false);
        let use_paged_kv = B::supports_paged_kv()
            && B::supports_varlen_qkv()
            && B::supports_qwen35_paged_qkv()
            && qwen35_runtime_bool(&snapshot, "FERRUM_METAL_PAGED_KV").unwrap_or(true);
        let use_vllm_paged_attn = use_paged_kv
            && qwen35_runtime_bool(&snapshot, "FERRUM_USE_VLLM_PAGED_ATTN").unwrap_or(false)
            && B::supports_vllm_paged_attn()
            && B::supports_qwen35_paged_qkv_vllm();
        let paged_max_seqs = qwen35_paged_max_seqs(&snapshot);
        let paged_max_blocks_per_seq = kv_capacity.div_ceil(QWEN35_PAGED_BLOCK_SIZE);
        Ok(Self {
            weights,
            weight_plan,
            weight_validation,
            rope,
            sequences: HashMap::new(),
            kv_capacity,
            use_paged_kv,
            use_vllm_paged_attn,
            paged_pools: None,
            paged_block_alloc: None,
            paged_max_seqs,
            paged_max_blocks_per_seq,
            paged_scratch: Qwen35PagedScratch::new(),
            decode_scratch: None,
            linear_state_pools: None,
            linear_free_slots: Vec::new(),
            linear_slot_indices: None,
            linear_slot_indices_capacity: 0,
            greedy_argmax,
            argmax_token_mask: None,
            argmax_token_mask_fingerprint: None,
            argmax_token_mask_len: 0,
            argmax_repetition_offsets: None,
            argmax_repetition_token_ids: None,
            argmax_repetition_penalties: None,
            argmax_repetition_offsets_capacity: 0,
            argmax_repetition_token_ids_capacity: 0,
            argmax_repetition_penalties_capacity: 0,
        })
    }

    pub fn from_definition_with_loader(
        def: &ModelDefinition,
        weight_plan: Qwen35ResolvedWeightPlan,
        loader: &dyn WeightLoader<B>,
    ) -> Result<Self> {
        let config = Qwen35TextConfig::from_model_definition(def)
            .map_err(|err| FerrumError::model(format!("invalid Qwen3.5/Qwen3.6 config: {err}")))?;
        let runtime_cfg = qwen35_runtime_config(
            &config,
            def.vocab_size,
            qwen35_effective_max_seq_len(def.max_position_embeddings),
        );
        Self::from_weight_plan(config, runtime_cfg, weight_plan, loader)
    }

    pub fn qwen35_config(&self) -> &Qwen35TextConfig {
        &self.weights.config
    }

    pub fn runtime_config(&self) -> &LlmRuntimeConfig {
        &self.weights.runtime_cfg
    }

    fn ensure_paged_pools(&mut self) {
        if !self.use_paged_kv || self.paged_pools.is_some() {
            return;
        }
        let snapshot = ferrum_types::active_runtime_snapshot();
        let total_blocks = qwen35_paged_total_blocks(
            &snapshot,
            self.paged_max_blocks_per_seq,
            self.paged_max_seqs,
        );
        let pool_len = total_blocks
            * self.weights.config.num_key_value_heads
            * QWEN35_PAGED_BLOCK_SIZE
            * self.weights.config.head_dim;
        let mut pools = Vec::with_capacity(self.weights.config.num_hidden_layers);
        for _ in 0..self.weights.config.num_hidden_layers {
            pools.push((B::alloc(pool_len), B::alloc(pool_len)));
        }
        self.paged_pools = Some(pools);
        self.paged_block_alloc = Some(std::sync::Mutex::new(BlockAllocator::new(
            total_blocks as u32,
        )));
    }

    fn ensure_decode_scratch(&mut self, tokens: usize) {
        let needed_tokens = tokens.max(1);
        let keep_existing = self
            .decode_scratch
            .as_ref()
            .is_some_and(|scratch| scratch.covers(&self.weights.config, needed_tokens));
        if keep_existing {
            return;
        }
        let capacity = needed_tokens.next_power_of_two();
        self.decode_scratch = Some(Qwen35DecodeScratch::alloc(&self.weights.config, capacity));
    }

    fn has_linear_attention_layers(&self) -> Result<bool> {
        Ok(self
            .weights
            .config
            .layer_plan()
            .map_err(FerrumError::model)?
            .iter()
            .any(|layer| matches!(layer.attention, Qwen35LayerType::LinearAttention)))
    }

    fn linear_attention_state_shape(&self) -> Qwen35LinearAttentionShape {
        Qwen35LinearAttentionShape {
            tokens: 1,
            key_heads: self.weights.config.linear_attention.num_key_heads,
            value_heads: self.weights.config.linear_attention.num_value_heads,
            key_dim: self.weights.config.linear_attention.key_head_dim,
            value_dim: self.weights.config.linear_attention.value_head_dim,
            conv_kernel: self.weights.config.linear_attention.conv_kernel_dim,
        }
    }

    fn ensure_linear_state_pools(&mut self) -> Result<()> {
        if !B::supports_qwen35_indexed_recurrent_state()
            || self.linear_state_pools.is_some()
            || !self.has_linear_attention_layers()?
        {
            return Ok(());
        }
        let shape = self.linear_attention_state_shape();
        let conv_state_len = shape.conv_channels() * shape.conv_kernel.saturating_sub(1);
        let delta_state_len = shape.state_len();
        let max_slots = self.paged_max_seqs.max(1);
        let layer_plan = self
            .weights
            .config
            .layer_plan()
            .map_err(FerrumError::model)?;
        let mut conv_states = (0..self.weights.config.num_hidden_layers)
            .map(|_| None)
            .collect::<Vec<_>>();
        let mut delta_states = (0..self.weights.config.num_hidden_layers)
            .map(|_| None)
            .collect::<Vec<_>>();
        for layer in layer_plan {
            if matches!(layer.attention, Qwen35LayerType::LinearAttention) {
                conv_states[layer.layer_index] =
                    Some(B::alloc_typed(Dtype::F32, max_slots * conv_state_len));
                delta_states[layer.layer_index] =
                    Some(B::alloc_typed(Dtype::F32, max_slots * delta_state_len));
            }
        }
        self.linear_state_pools = Some(Qwen35LinearStatePools {
            conv_states,
            delta_states,
            max_slots,
            conv_state_len,
            delta_state_len,
        });
        self.linear_free_slots = (0..max_slots).rev().collect();
        Ok(())
    }

    fn allocate_linear_slot(&mut self) -> Result<Option<usize>> {
        if !B::supports_qwen35_indexed_recurrent_state() || !self.has_linear_attention_layers()? {
            return Ok(None);
        }
        self.ensure_linear_state_pools()?;
        let slot = self.linear_free_slots.pop().ok_or_else(|| {
            FerrumError::model(format!(
                "Qwen3.5 linear state slot pool exhausted: max_slots={}",
                self.linear_state_pools
                    .as_ref()
                    .map(|pools| pools.max_slots)
                    .unwrap_or(0)
            ))
        })?;
        Ok(Some(slot))
    }

    fn free_linear_slot(&mut self, state: &mut Qwen35SequenceState<B>) {
        let Some(slot) = state.linear_slot.take() else {
            return;
        };
        if let Some(pools) = self.linear_state_pools.as_ref() {
            if slot < pools.max_slots && !self.linear_free_slots.contains(&slot) {
                self.linear_free_slots.push(slot);
            }
        }
    }

    fn sync_sequence_linear_state_from_slot(
        &mut self,
        ctx: &mut B::Context,
        state: &mut Qwen35SequenceState<B>,
    ) -> Result<()> {
        let (Some(slot), Some(pools)) = (state.linear_slot, self.linear_state_pools.as_mut())
        else {
            return Ok(());
        };
        for (layer_index, layer_state) in state.layers.iter_mut().enumerate() {
            let Qwen35LayerRuntimeState::Linear {
                conv_state,
                delta_state,
            } = layer_state
            else {
                continue;
            };
            let conv_slots = pools.conv_states[layer_index].as_ref().ok_or_else(|| {
                FerrumError::model(format!(
                    "Qwen3.5 missing linear conv state slot pool for layer {layer_index}"
                ))
            })?;
            let delta_slots = pools.delta_states[layer_index].as_ref().ok_or_else(|| {
                FerrumError::model(format!(
                    "Qwen3.5 missing linear delta state slot pool for layer {layer_index}"
                ))
            })?;
            B::copy_slice(
                ctx,
                conv_slots,
                slot * pools.conv_state_len,
                conv_state,
                0,
                pools.conv_state_len,
            );
            B::copy_slice(
                ctx,
                delta_slots,
                slot * pools.delta_state_len,
                delta_state,
                0,
                pools.delta_state_len,
            );
        }
        Ok(())
    }

    fn sync_sequence_linear_state_to_slot(
        &mut self,
        ctx: &mut B::Context,
        state: &mut Qwen35SequenceState<B>,
    ) -> Result<()> {
        let (Some(slot), Some(pools)) = (state.linear_slot, self.linear_state_pools.as_mut())
        else {
            return Ok(());
        };
        for (layer_index, layer_state) in state.layers.iter_mut().enumerate() {
            let Qwen35LayerRuntimeState::Linear {
                conv_state,
                delta_state,
            } = layer_state
            else {
                continue;
            };
            let conv_slots = pools.conv_states[layer_index].as_mut().ok_or_else(|| {
                FerrumError::model(format!(
                    "Qwen3.5 missing linear conv state slot pool for layer {layer_index}"
                ))
            })?;
            let delta_slots = pools.delta_states[layer_index].as_mut().ok_or_else(|| {
                FerrumError::model(format!(
                    "Qwen3.5 missing linear delta state slot pool for layer {layer_index}"
                ))
            })?;
            B::copy_slice(
                ctx,
                conv_state,
                0,
                conv_slots,
                slot * pools.conv_state_len,
                pools.conv_state_len,
            );
            B::copy_slice(
                ctx,
                delta_state,
                0,
                delta_slots,
                slot * pools.delta_state_len,
                pools.delta_state_len,
            );
        }
        Ok(())
    }

    fn upload_linear_slot_indices(
        &mut self,
        ctx: &mut B::Context,
        states: &[(String, Qwen35SequenceState<B>)],
    ) -> Result<()> {
        if !B::supports_qwen35_indexed_recurrent_state() || self.linear_state_pools.is_none() {
            return Ok(());
        }
        let mut slots = Vec::with_capacity(states.len());
        for (cache_id, state) in states {
            let slot = state.linear_slot.ok_or_else(|| {
                FerrumError::model(format!(
                    "Qwen3.5 missing linear state slot for cache_id={cache_id:?}"
                ))
            })?;
            slots.push(u32::try_from(slot).map_err(|_| {
                FerrumError::model(format!("Qwen3.5 linear state slot {slot} exceeds u32"))
            })?);
        }
        if self.linear_slot_indices_capacity < slots.len() {
            self.linear_slot_indices = Some(B::alloc_typed(Dtype::U32, slots.len().max(1)));
            self.linear_slot_indices_capacity = slots.len();
        }
        let slot_indices = self
            .linear_slot_indices
            .as_mut()
            .expect("Qwen3.5 linear slot index buffer allocation failed");
        B::write_typed::<u32>(ctx, slot_indices, &slots);
        Ok(())
    }

    fn ensure_paged_kv_capacity_for_state(
        &mut self,
        ctx: &mut B::Context,
        cache_id: &str,
        state: &mut Qwen35SequenceState<B>,
        target_len: usize,
    ) -> Result<()> {
        if !self.use_paged_kv {
            return Ok(());
        }
        let Some((block_size, max_blocks_per_seq, mut block_indices)) = qwen35_first_full_kv(state)
            .map(|kv| {
                (
                    kv.block_size,
                    kv.capacity / kv.block_size.max(1),
                    kv.paged_block_indices.clone(),
                )
            })
        else {
            return Ok(());
        };
        if block_size == 0 {
            return Ok(());
        }
        let needed_blocks = target_len.div_ceil(block_size);
        if needed_blocks > max_blocks_per_seq {
            return Err(FerrumError::model(format!(
                "Qwen3.5 paged KV: target_len={target_len} needs {needed_blocks} blocks, \
                 exceeds per-seq table capacity {max_blocks_per_seq} for cache_id={cache_id:?}"
            )));
        }
        if block_indices.len() < needed_blocks {
            let extra = needed_blocks - block_indices.len();
            let new_blocks = {
                let alloc = self.paged_block_alloc.as_ref().ok_or_else(|| {
                    FerrumError::model(
                        "Qwen3.5 paged KV grow missing allocator while paged pools are set",
                    )
                })?;
                let mut alloc = alloc.lock().unwrap_or_else(|p| p.into_inner());
                alloc.allocate_n(extra)?
            };
            block_indices.extend(new_blocks);
        }
        let mut padded = block_indices.clone();
        padded.resize(max_blocks_per_seq, 0);
        for layer_state in &mut state.layers {
            if let Qwen35LayerRuntimeState::Full { kv } = layer_state {
                kv.paged_block_indices = block_indices.clone();
                if let Some(block_table) = kv.block_table.as_mut() {
                    B::write_typed::<u32>(ctx, block_table, &padded);
                }
            }
        }
        Ok(())
    }

    fn release_sequence_state_blocks(&mut self, state: &mut Qwen35SequenceState<B>) {
        self.free_linear_slot(state);
        if !self.use_paged_kv {
            return;
        }
        let Some(blocks) = qwen35_first_full_kv(state)
            .map(|kv| kv.paged_block_indices.clone())
            .filter(|blocks| !blocks.is_empty())
        else {
            return;
        };
        if let Some(alloc) = self.paged_block_alloc.as_ref() {
            let mut alloc = alloc.lock().unwrap_or_else(|p| p.into_inner());
            alloc.free(&blocks);
        }
        for layer_state in &mut state.layers {
            if let Qwen35LayerRuntimeState::Full { kv } = layer_state {
                kv.paged_block_indices.clear();
                kv.len = 0;
            }
        }
    }

    fn allocate_sequence_state(&mut self) -> Result<Qwen35SequenceState<B>> {
        if self.use_paged_kv {
            self.ensure_paged_pools();
        }
        let layer_plan = self
            .weights
            .config
            .layer_plan()
            .map_err(FerrumError::model)?;
        let linear_slot = self.allocate_linear_slot()?;
        let mut layers = Vec::with_capacity(self.weights.config.num_hidden_layers);
        for layer_plan in layer_plan {
            match layer_plan.attention {
                Qwen35LayerType::LinearAttention => {
                    let shape = Qwen35LinearAttentionShape {
                        tokens: 1,
                        key_heads: self.weights.config.linear_attention.num_key_heads,
                        value_heads: self.weights.config.linear_attention.num_value_heads,
                        key_dim: self.weights.config.linear_attention.key_head_dim,
                        value_dim: self.weights.config.linear_attention.value_head_dim,
                        conv_kernel: self.weights.config.linear_attention.conv_kernel_dim,
                    };
                    layers.push(Qwen35LayerRuntimeState::Linear {
                        conv_state: B::alloc_typed(
                            Dtype::F32,
                            shape.conv_channels() * shape.conv_kernel.saturating_sub(1),
                        ),
                        delta_state: B::alloc_typed(Dtype::F32, shape.state_len()),
                    });
                }
                Qwen35LayerType::FullAttention => {
                    let kv = if self.use_paged_kv {
                        let mut context_lens = B::alloc_typed(Dtype::U32, 1);
                        let mut block_table =
                            B::alloc_typed(Dtype::U32, self.paged_max_blocks_per_seq);
                        let mut ctx = B::new_context();
                        let padded = vec![0u32; self.paged_max_blocks_per_seq];
                        B::write_typed::<u32>(&mut ctx, &mut context_lens, &[0u32]);
                        B::write_typed::<u32>(&mut ctx, &mut block_table, &padded);
                        B::sync(&mut ctx);
                        KvCache {
                            k: B::alloc(1),
                            v: B::alloc(1),
                            len: 0,
                            capacity: self.paged_max_blocks_per_seq * QWEN35_PAGED_BLOCK_SIZE,
                            num_kv_heads: self.weights.config.num_key_value_heads,
                            head_dim: self.weights.config.head_dim,
                            block_size: QWEN35_PAGED_BLOCK_SIZE,
                            block_table: Some(block_table),
                            context_lens: Some(context_lens),
                            paged_block_indices: Vec::new(),
                            _kv_dtype: std::marker::PhantomData,
                        }
                    } else {
                        KvCache {
                            k: B::alloc(
                                self.weights.config.num_key_value_heads
                                    * self.kv_capacity
                                    * self.weights.config.head_dim,
                            ),
                            v: B::alloc(
                                self.weights.config.num_key_value_heads
                                    * self.kv_capacity
                                    * self.weights.config.head_dim,
                            ),
                            len: 0,
                            capacity: self.kv_capacity,
                            num_kv_heads: self.weights.config.num_key_value_heads,
                            head_dim: self.weights.config.head_dim,
                            block_size: 0,
                            block_table: None,
                            context_lens: None,
                            paged_block_indices: Vec::new(),
                            _kv_dtype: std::marker::PhantomData,
                        }
                    };
                    layers.push(Qwen35LayerRuntimeState::Full { kv });
                }
            }
        }
        Ok(Qwen35SequenceState {
            tokens: Vec::new(),
            layers,
            linear_slot,
        })
    }

    fn ensure_sequence_state(&mut self, cache_id: &str) -> Result<()> {
        if self.sequences.contains_key(cache_id) {
            return Ok(());
        }
        let state = self.allocate_sequence_state()?;
        self.sequences.insert(cache_id.to_string(), state);
        Ok(())
    }

    fn forward_stateful(&mut self, cache_id: &str, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(FerrumError::model(
                "Qwen3.5 stateful forward requires at least one token",
            ));
        }
        self.ensure_sequence_state(cache_id)?;
        if tokens.len() > 1
            && self
                .sequences
                .get(cache_id)
                .is_some_and(|state| !state.tokens.is_empty())
        {
            let mut last = Vec::new();
            for token in tokens {
                last = self.forward_stateful_chunk(cache_id, &[*token])?;
            }
            return Ok(last);
        }
        self.forward_stateful_chunk(cache_id, tokens)
    }

    fn forward_stateful_chunk(&mut self, cache_id: &str, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut state = self.sequences.remove(cache_id).ok_or_else(|| {
            FerrumError::model(format!("Qwen3.5 missing sequence state for {cache_id:?}"))
        })?;
        let result = self.forward_stateful_chunk_taken(cache_id, tokens, &mut state);
        if result.is_ok() {
            state.tokens.extend_from_slice(tokens);
        }
        self.sequences.insert(cache_id.to_string(), state);
        result
    }

    fn forward_stateful_chunk_taken(
        &mut self,
        cache_id: &str,
        tokens: &[u32],
        state: &mut Qwen35SequenceState<B>,
    ) -> Result<Vec<f32>> {
        if state.layers.len() != self.weights.layers.len() {
            return Err(FerrumError::model(format!(
                "Qwen3.5 sequence state layer count {} != model layers {}",
                state.layers.len(),
                self.weights.layers.len()
            )));
        }
        let tokens_len = tokens.len();
        if state.tokens.len() + tokens_len > self.kv_capacity {
            return Err(FerrumError::model(format!(
                "Qwen3.5 sequence length {} exceeds KV capacity {}",
                state.tokens.len() + tokens_len,
                self.kv_capacity
            )));
        }

        let mut ctx = B::new_context();
        let prior_len = state.tokens.len();
        let profile_enabled = qwen35_prefill_profile_enabled();
        let mut prefill_profile = Qwen35PrefillProfile::default();
        if prior_len > 0 {
            self.sync_sequence_linear_state_from_slot(&mut ctx, state)?;
        }
        self.ensure_paged_kv_capacity_for_state(
            &mut ctx,
            cache_id,
            state,
            state.tokens.len() + tokens_len,
        )?;
        let weights = &self.weights;
        let rope = &self.rope;
        let mut hidden = B::alloc(tokens_len * weights.config.hidden_size);
        let t_embedding = qwen35_prefill_profile_stage_start::<B>(&mut ctx, profile_enabled);
        B::embedding_lookup(
            &mut ctx,
            &weights.embed_tokens,
            tokens,
            &mut hidden,
            weights.config.hidden_size,
        );
        prefill_profile.embedding_us +=
            qwen35_prefill_profile_stage_finish::<B>(&mut ctx, t_embedding, "qwen35_embedding");
        qwen35_trace_buffer_stats::<B>(
            &mut ctx,
            "embedding",
            &hidden,
            tokens_len * weights.config.hidden_size,
        );
        let eps = 1e-6f32;
        let hidden_len = tokens_len * weights.config.hidden_size;
        let mut residual_f32 = if B::supports_device_f32_residual_shadow() {
            let mut shadow = B::alloc_typed(Dtype::F32, hidden_len);
            B::activation_to_f32_shadow(&mut ctx, &hidden, &mut shadow, hidden_len);
            Some(shadow)
        } else {
            None
        };
        let mut branch_f32 = residual_f32
            .as_ref()
            .map(|_| B::alloc_typed(Dtype::F32, hidden_len));
        for (layer_index, (layer, layer_state)) in weights
            .layers
            .iter()
            .zip(state.layers.iter_mut())
            .enumerate()
        {
            let layer_kind = match layer_state {
                Qwen35LayerRuntimeState::Linear { .. } => "linear",
                Qwen35LayerRuntimeState::Full { .. } => "full",
            };
            let t_layer = qwen35_prefill_profile_stage_start::<B>(&mut ctx, profile_enabled);
            hidden = match layer_state {
                Qwen35LayerRuntimeState::Linear {
                    conv_state,
                    delta_state,
                } => qwen35_linear_attention_stateful_layer_backend::<B>(
                    &mut ctx,
                    &hidden,
                    conv_state,
                    delta_state,
                    layer,
                    &weights.config,
                    tokens_len,
                    eps,
                    residual_f32.as_mut(),
                    branch_f32.as_mut(),
                )?,
                Qwen35LayerRuntimeState::Full { kv } => {
                    let paged = if self.use_paged_kv {
                        self.paged_pools
                            .as_mut()
                            .map(|pools| (&mut pools[layer_index], &mut self.paged_scratch))
                    } else {
                        None
                    };
                    qwen35_full_attention_stateful_layer_backend::<B>(
                        &mut ctx,
                        &hidden,
                        kv,
                        paged,
                        layer,
                        rope,
                        &weights.config,
                        self.use_vllm_paged_attn,
                        tokens_len,
                        eps,
                        residual_f32.as_mut(),
                        branch_f32.as_mut(),
                    )?
                }
            };
            let layer_elapsed =
                qwen35_prefill_profile_stage_finish::<B>(&mut ctx, t_layer, "qwen35_prefill_layer");
            prefill_profile.record_layer(layer_index, layer_kind, layer_elapsed);
            if std::env::var_os("FERRUM_QWEN35_BUFFER_TRACE").is_some() {
                let label = format!("layer{layer_index}.output");
                qwen35_trace_buffer_stats::<B>(
                    &mut ctx,
                    &label,
                    &hidden,
                    tokens_len * weights.config.hidden_size,
                );
            }
        }

        let mut final_hidden = B::alloc(tokens_len * weights.config.hidden_size);
        let t_final_norm = qwen35_prefill_profile_stage_start::<B>(&mut ctx, profile_enabled);
        if let Some(residual_f32) = residual_f32.as_ref() {
            B::rms_norm_f32_to_activation(
                &mut ctx,
                residual_f32,
                &weights.final_norm,
                eps,
                &mut final_hidden,
                tokens_len,
                weights.config.hidden_size,
            );
        } else {
            B::rms_norm(
                &mut ctx,
                &hidden,
                &weights.final_norm,
                eps,
                &mut final_hidden,
                tokens_len,
                weights.config.hidden_size,
            );
        }
        prefill_profile.final_norm_us +=
            qwen35_prefill_profile_stage_finish::<B>(&mut ctx, t_final_norm, "qwen35_final_norm");
        qwen35_trace_buffer_stats::<B>(
            &mut ctx,
            "final_norm",
            &final_hidden,
            tokens_len * weights.config.hidden_size,
        );
        let vocab_size = weights.runtime_cfg.vocab_size;
        let layer_count = weights.layers.len();
        let t_final_gather =
            qwen35_prefill_profile_stage_start::<B>(&mut ctx, profile_enabled && tokens_len > 1);
        let last_hidden = if tokens_len == 1 {
            final_hidden
        } else {
            let mut last_hidden = B::alloc(weights.config.hidden_size);
            B::copy_slice(
                &mut ctx,
                &final_hidden,
                (tokens_len - 1) * weights.config.hidden_size,
                &mut last_hidden,
                0,
                weights.config.hidden_size,
            );
            last_hidden
        };
        prefill_profile.final_gather_us += qwen35_prefill_profile_stage_finish::<B>(
            &mut ctx,
            t_final_gather,
            "qwen35_final_token_gather",
        );
        let mut last_logits = B::alloc(vocab_size);
        let t_lm_head = qwen35_prefill_profile_stage_start::<B>(&mut ctx, profile_enabled);
        weights
            .lm_head
            .forward(&mut ctx, &last_hidden, &mut last_logits, 1);
        prefill_profile.lm_head_us +=
            qwen35_prefill_profile_stage_finish::<B>(&mut ctx, t_lm_head, "qwen35_lm_head");
        let t_readback = if profile_enabled {
            Some(Instant::now())
        } else {
            None
        };
        B::sync_before_host_readback(&mut ctx);
        B::sync(&mut ctx);
        let out = B::to_vec(&last_logits, vocab_size);
        self.sync_sequence_linear_state_to_slot(&mut ctx, state)?;
        B::sync(&mut ctx);
        if let Some(start) = t_readback {
            prefill_profile.readback_us += start.elapsed().as_micros() as u64;
        }
        if profile_enabled {
            prefill_profile.log(cache_id, prior_len, tokens_len, layer_count);
        }
        if std::env::var_os("FERRUM_QWEN35_LOGITS_TRACE").is_some() {
            let mut finite = 0usize;
            let mut top: Vec<(usize, f32)> = Vec::new();
            for (token_id, &logit) in out.iter().enumerate() {
                if !logit.is_finite() {
                    continue;
                }
                finite += 1;
                let insert_at = top
                    .iter()
                    .position(|&(_, value)| logit > value)
                    .unwrap_or(top.len());
                if insert_at < 8 {
                    top.insert(insert_at, (token_id, logit));
                    top.truncate(8);
                }
            }
            eprintln!(
                "[qwen35-logits-trace] cache_id={cache_id:?} tokens_len={} vocab={} finite={} \
                 nonfinite={} top={top:?}",
                tokens_len,
                vocab_size,
                finite,
                out.len().saturating_sub(finite),
            );
        }
        Ok(out)
    }

    fn forward_stateful_prefill_batch(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
    ) -> Result<Vec<Option<Vec<f32>>>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        if !self.use_paged_kv {
            return Err(FerrumError::unsupported(
                "Qwen3.5 unified_forward fresh prefill requires paged KV",
            ));
        }
        for (_, tokens, pos_offset, is_final_chunk) in items {
            if tokens.is_empty() {
                return Err(FerrumError::unsupported(
                    "Qwen3.5 unified_forward does not accept empty prefill chunks",
                ));
            }
            if *pos_offset != 0 || !*is_final_chunk {
                return Err(FerrumError::unsupported(
                    "Qwen3.5 unified_forward currently supports fresh final prefill batches only",
                ));
            }
        }
        let mut seen = HashSet::with_capacity(items.len());
        for (cache_id, _, _, _) in items {
            if !seen.insert(cache_id.as_str()) {
                return Err(FerrumError::unsupported(
                    "Qwen3.5 unified_forward requires unique cache ids",
                ));
            }
        }
        for (cache_id, _, _, _) in items {
            self.ensure_sequence_state(cache_id)?;
            let state = self.sequences.get(cache_id).ok_or_else(|| {
                FerrumError::model(format!("Qwen3.5 missing sequence state for {cache_id:?}"))
            })?;
            if !state.tokens.is_empty() {
                return Err(FerrumError::unsupported(
                    "Qwen3.5 unified_forward fresh prefill received a non-empty sequence state",
                ));
            }
            if qwen35_first_full_kv(state).is_some_and(|kv| kv.len != 0) {
                return Err(FerrumError::unsupported(
                    "Qwen3.5 unified_forward fresh prefill received non-empty full-attention KV",
                ));
            }
            if state.layers.len() != self.weights.layers.len() {
                return Err(FerrumError::model(format!(
                    "Qwen3.5 sequence state layer count {} != model layers {}",
                    state.layers.len(),
                    self.weights.layers.len()
                )));
            }
        }
        self.ensure_paged_pools();

        let mut states = Vec::with_capacity(items.len());
        for (cache_id, _, _, _) in items {
            let state = self.sequences.remove(cache_id).ok_or_else(|| {
                FerrumError::model(format!("Qwen3.5 missing sequence state for {cache_id:?}"))
            })?;
            states.push((cache_id.clone(), state));
        }
        let result = self.forward_stateful_prefill_batch_taken(items, &mut states);
        let success = result.is_ok();
        for ((cache_id, mut state), (_, tokens, _, _)) in states.into_iter().zip(items.iter()) {
            if success {
                state.tokens.extend_from_slice(tokens);
            }
            self.sequences.insert(cache_id, state);
        }
        result.map(|rows| rows.into_iter().map(Some).collect())
    }

    fn forward_stateful_prefill_batch_taken(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
        states: &mut [(String, Qwen35SequenceState<B>)],
    ) -> Result<Vec<Vec<f32>>> {
        let batch_len = items.len();
        if batch_len == 0 {
            return Ok(Vec::new());
        }
        let mut q_lens = Vec::with_capacity(batch_len);
        let mut cu = Vec::with_capacity(batch_len + 1);
        let mut all_tokens = Vec::new();
        cu.push(0u32);
        for (_, tokens, _, _) in items {
            if tokens.len() > self.kv_capacity {
                return Err(FerrumError::model(format!(
                    "Qwen3.5 prefill length {} exceeds KV capacity {}",
                    tokens.len(),
                    self.kv_capacity
                )));
            }
            all_tokens.extend_from_slice(tokens);
            q_lens.push(tokens.len());
            let next = u32::try_from(all_tokens.len()).map_err(|_| {
                FerrumError::model("Qwen3.5 unified prefill total token count exceeds u32")
            })?;
            cu.push(next);
        }
        let total_tokens = all_tokens.len();
        let mut ctx = B::new_context();
        for (row, ((cache_id, _, _, _), (_, state))) in
            items.iter().zip(states.iter_mut()).enumerate()
        {
            self.ensure_paged_kv_capacity_for_state(&mut ctx, cache_id, state, q_lens[row])?;
        }
        let mut cu_buf = B::alloc_typed(Dtype::U32, cu.len());
        B::write_typed::<u32>(&mut ctx, &mut cu_buf, &cu);

        let weights = &self.weights;
        let rope = &self.rope;
        let hidden_len = total_tokens * weights.config.hidden_size;
        let mut hidden = B::alloc(hidden_len);
        B::embedding_lookup(
            &mut ctx,
            &weights.embed_tokens,
            &all_tokens,
            &mut hidden,
            weights.config.hidden_size,
        );
        let eps = 1e-6f32;
        let mut residual_f32 = if B::supports_device_f32_residual_shadow() {
            let mut shadow = B::alloc_typed(Dtype::F32, hidden_len);
            B::activation_to_f32_shadow(&mut ctx, &hidden, &mut shadow, hidden_len);
            Some(shadow)
        } else {
            None
        };
        let mut branch_f32 = residual_f32
            .as_ref()
            .map(|_| B::alloc_typed(Dtype::F32, hidden_len));

        for (layer_index, layer) in weights.layers.iter().enumerate() {
            hidden = match &layer.attention {
                Qwen35AttentionWeights::Linear(_) => {
                    qwen35_linear_attention_prefill_batch_layer_backend::<B>(
                        &mut ctx,
                        &hidden,
                        states,
                        self.linear_state_pools.as_mut(),
                        layer_index,
                        layer,
                        &weights.config,
                        &cu_buf,
                        &q_lens,
                        total_tokens,
                        eps,
                        residual_f32.as_mut(),
                        branch_f32.as_mut(),
                    )?
                }
                Qwen35AttentionWeights::Full(_) => {
                    let pools = self.paged_pools.as_mut().ok_or_else(|| {
                        FerrumError::model(
                            "Qwen3.5 unified prefill missing paged full-attention pools",
                        )
                    })?;
                    qwen35_full_attention_prefill_batch_layer_backend::<B>(
                        &mut ctx,
                        &hidden,
                        states,
                        &mut pools[layer_index],
                        &mut self.paged_scratch,
                        layer_index,
                        layer,
                        rope,
                        &weights.config,
                        self.use_vllm_paged_attn,
                        &cu_buf,
                        &cu,
                        &q_lens,
                        total_tokens,
                        eps,
                        residual_f32.as_mut(),
                        branch_f32.as_mut(),
                    )?
                }
            };
        }

        let mut final_hidden = B::alloc(hidden_len);
        if let Some(residual_f32) = residual_f32.as_ref() {
            B::rms_norm_f32_to_activation(
                &mut ctx,
                residual_f32,
                &weights.final_norm,
                eps,
                &mut final_hidden,
                total_tokens,
                weights.config.hidden_size,
            );
        } else {
            B::rms_norm(
                &mut ctx,
                &hidden,
                &weights.final_norm,
                eps,
                &mut final_hidden,
                total_tokens,
                weights.config.hidden_size,
            );
        }
        let mut sampled_hidden = B::alloc(batch_len * weights.config.hidden_size);
        for row in 0..batch_len {
            let final_token = cu[row + 1] as usize - 1;
            B::copy_slice(
                &mut ctx,
                &final_hidden,
                final_token * weights.config.hidden_size,
                &mut sampled_hidden,
                row * weights.config.hidden_size,
                weights.config.hidden_size,
            );
        }
        let vocab = weights.runtime_cfg.vocab_size;
        let mut logits = B::alloc(batch_len * vocab);
        weights
            .lm_head
            .forward(&mut ctx, &sampled_hidden, &mut logits, batch_len);
        B::sync_before_host_readback(&mut ctx);
        B::sync(&mut ctx);
        let flat = B::to_vec(&logits, batch_len * vocab);
        Ok(flat.chunks_exact(vocab).map(|row| row.to_vec()).collect())
    }

    fn forward_stateful_decode_batch(
        &mut self,
        batch: &[(String, u32, u32)],
    ) -> Result<Vec<Vec<f32>>> {
        self.forward_stateful_decode_batch_with_logits_return(
            batch,
            Qwen35DecodeLogitsReturn::LegacyDefault,
        )
    }

    fn forward_stateful_decode_batch_with_logits_return(
        &mut self,
        batch: &[(String, u32, u32)],
        logits_return: Qwen35DecodeLogitsReturn<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        let mut seen = HashSet::with_capacity(batch.len());
        for (cache_id, _, _) in batch {
            if !seen.insert(cache_id.as_str()) {
                return Err(FerrumError::unsupported(
                    "Qwen3.5 decode_batch requires unique cache ids",
                ));
            }
        }
        for (cache_id, _, _) in batch {
            self.ensure_sequence_state(cache_id)?;
        }
        for (cache_id, _, pos) in batch {
            let state = self.sequences.get(cache_id).ok_or_else(|| {
                FerrumError::model(format!("Qwen3.5 missing sequence state for {cache_id:?}"))
            })?;
            if state.layers.len() != self.weights.layers.len() {
                return Err(FerrumError::model(format!(
                    "Qwen3.5 sequence state layer count {} != model layers {}",
                    state.layers.len(),
                    self.weights.layers.len()
                )));
            }
            if state.tokens.len() != *pos as usize {
                return Err(FerrumError::model(format!(
                    "Qwen3.5 decode_batch position mismatch for {cache_id:?}: state len {} != pos {}",
                    state.tokens.len(),
                    pos
                )));
            }
            if state.tokens.len() + 1 > self.kv_capacity {
                return Err(FerrumError::model(format!(
                    "Qwen3.5 sequence length {} exceeds KV capacity {}",
                    state.tokens.len() + 1,
                    self.kv_capacity
                )));
            }
        }

        let mut states = Vec::with_capacity(batch.len());
        for (cache_id, _, _) in batch {
            let state = self.sequences.remove(cache_id).ok_or_else(|| {
                FerrumError::model(format!("Qwen3.5 missing sequence state for {cache_id:?}"))
            })?;
            states.push((cache_id.clone(), state));
        }
        let tokens: Vec<u32> = batch.iter().map(|(_, token, _)| *token).collect();
        let result = self.forward_stateful_decode_batch_taken(&tokens, &mut states, logits_return);
        let success = result.is_ok();
        for ((cache_id, mut state), token) in states.into_iter().zip(tokens.iter().copied()) {
            if success {
                state.tokens.push(token);
            }
            self.sequences.insert(cache_id, state);
        }
        result
    }

    fn forward_stateful_decode_batch_taken(
        &mut self,
        tokens: &[u32],
        states: &mut [(String, Qwen35SequenceState<B>)],
        logits_return: Qwen35DecodeLogitsReturn<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        let batch_len = tokens.len();
        if batch_len == 0 {
            return Ok(Vec::new());
        }
        self.ensure_decode_scratch(batch_len);
        let mut ctx = B::new_context();
        let decode_profile_enabled = qwen35_decode_profile_enabled();
        let mut decode_profile = Qwen35DecodeProfile::default();
        for (cache_id, state) in states.iter_mut() {
            self.ensure_paged_kv_capacity_for_state(
                &mut ctx,
                cache_id,
                state,
                state.tokens.len() + 1,
            )?;
        }
        self.upload_linear_slot_indices(&mut ctx, states)?;
        let weights = &self.weights;
        let rope = &self.rope;
        let total_layers = weights.layers.len();
        let hidden_len = batch_len * weights.config.hidden_size;
        let mut hidden = B::alloc(hidden_len);
        let t_embedding = qwen35_decode_profile_stage_start::<B>(&mut ctx, decode_profile_enabled);
        B::embedding_lookup(
            &mut ctx,
            &weights.embed_tokens,
            tokens,
            &mut hidden,
            weights.config.hidden_size,
        );
        decode_profile.embedding_us += qwen35_decode_profile_stage_finish::<B>(
            &mut ctx,
            t_embedding,
            "qwen35_decode_embedding",
        );
        let eps = 1e-6f32;
        let mut residual_f32 = if B::supports_device_f32_residual_shadow() {
            let mut shadow = B::alloc_typed(Dtype::F32, hidden_len);
            B::activation_to_f32_shadow(&mut ctx, &hidden, &mut shadow, hidden_len);
            Some(shadow)
        } else {
            None
        };
        let mut branch_f32 = residual_f32
            .as_ref()
            .map(|_| B::alloc_typed(Dtype::F32, hidden_len));
        for (layer_index, layer) in weights.layers.iter().enumerate() {
            let layer_kind = match &layer.attention {
                Qwen35AttentionWeights::Linear(_) => "linear",
                Qwen35AttentionWeights::Full(_) => "full",
            };
            let t_layer = qwen35_decode_profile_stage_start::<B>(&mut ctx, decode_profile_enabled);
            let next_hidden = match &layer.attention {
                Qwen35AttentionWeights::Linear(_) => {
                    qwen35_linear_attention_decode_batch_layer_backend::<B>(
                        &mut ctx,
                        &hidden,
                        states,
                        self.linear_state_pools.as_mut(),
                        self.linear_slot_indices.as_ref(),
                        layer_index,
                        layer,
                        &weights.config,
                        eps,
                        residual_f32.as_mut(),
                        branch_f32.as_mut(),
                        self.decode_scratch.as_mut(),
                    )?
                }
                Qwen35AttentionWeights::Full(_) => {
                    let paged = if self.use_paged_kv {
                        self.paged_pools
                            .as_mut()
                            .map(|pools| (&mut pools[layer_index], &mut self.paged_scratch))
                    } else {
                        None
                    };
                    qwen35_full_attention_decode_batch_layer_backend::<B>(
                        &mut ctx,
                        &hidden,
                        states,
                        paged,
                        layer_index,
                        layer,
                        rope,
                        &weights.config,
                        self.use_vllm_paged_attn,
                        eps,
                        residual_f32.as_mut(),
                        branch_f32.as_mut(),
                        self.decode_scratch.as_mut(),
                    )?
                }
            };
            if let Some(next_hidden) = next_hidden {
                hidden = next_hidden;
            }
            let layer_elapsed =
                qwen35_decode_profile_stage_finish::<B>(&mut ctx, t_layer, "qwen35_decode_layer");
            decode_profile.record_layer(layer_index, layer_kind, layer_elapsed);
        }

        let mut final_hidden = B::alloc(hidden_len);
        let t_final_norm = qwen35_decode_profile_stage_start::<B>(&mut ctx, decode_profile_enabled);
        if let Some(residual_f32) = residual_f32.as_ref() {
            B::rms_norm_f32_to_activation(
                &mut ctx,
                residual_f32,
                &weights.final_norm,
                eps,
                &mut final_hidden,
                batch_len,
                weights.config.hidden_size,
            );
        } else {
            B::rms_norm(
                &mut ctx,
                &hidden,
                &weights.final_norm,
                eps,
                &mut final_hidden,
                batch_len,
                weights.config.hidden_size,
            );
        }
        decode_profile.final_norm_us += qwen35_decode_profile_stage_finish::<B>(
            &mut ctx,
            t_final_norm,
            "qwen35_decode_final_norm",
        );
        let vocab = weights.runtime_cfg.vocab_size;
        let mut logits = B::alloc(batch_len * vocab);
        let t_lm_head = qwen35_decode_profile_stage_start::<B>(&mut ctx, decode_profile_enabled);
        weights
            .lm_head
            .forward(&mut ctx, &final_hidden, &mut logits, batch_len);
        decode_profile.lm_head_us +=
            qwen35_decode_profile_stage_finish::<B>(&mut ctx, t_lm_head, "qwen35_decode_lm_head");
        let argmax_mode = match &logits_return {
            Qwen35DecodeLogitsReturn::Full => None,
            Qwen35DecodeLogitsReturn::LegacyDefault if self.greedy_argmax => {
                Some(Qwen35ArgmaxMode::Raw)
            }
            Qwen35DecodeLogitsReturn::LegacyDefault => None,
            Qwen35DecodeLogitsReturn::GreedyArgmax {
                token_mask,
                repetition_penalties,
            } => {
                if qwen35_has_repetition_penalties(repetition_penalties) {
                    Some(Qwen35ArgmaxMode::SparseRepetition {
                        token_mask: *token_mask,
                        repetition_penalties,
                    })
                } else {
                    token_mask
                        .map(Qwen35ArgmaxMode::Masked)
                        .or(Some(Qwen35ArgmaxMode::Raw))
                }
            }
        };
        if let Some(argmax_mode) = argmax_mode {
            let t_argmax = qwen35_decode_profile_stage_start::<B>(&mut ctx, decode_profile_enabled);
            let tokens = match argmax_mode {
                Qwen35ArgmaxMode::Raw => B::argmax_rows_f16(&mut ctx, &logits, batch_len, vocab),
                Qwen35ArgmaxMode::Masked(mask) => {
                    self.ensure_argmax_token_mask(&mut ctx, mask);
                    let mask_len = self.argmax_token_mask_len;
                    let device_mask = self
                        .argmax_token_mask
                        .as_ref()
                        .expect("Qwen3.5 argmax token mask upload failed");
                    B::argmax_rows_f16_masked(
                        &mut ctx,
                        &logits,
                        device_mask,
                        mask_len,
                        batch_len,
                        vocab,
                    )
                }
                Qwen35ArgmaxMode::SparseRepetition {
                    token_mask,
                    repetition_penalties,
                } => {
                    if let Some(mask) = token_mask {
                        self.ensure_argmax_token_mask(&mut ctx, mask);
                    }
                    let total_token_ids = self.upload_argmax_repetition_penalty(
                        &mut ctx,
                        repetition_penalties,
                        batch_len,
                    );
                    let valid_token_mask = token_mask.map(|_| {
                        (
                            self.argmax_token_mask
                                .as_ref()
                                .expect("Qwen3.5 argmax token mask upload failed"),
                            self.argmax_token_mask_len,
                        )
                    });
                    let row_offsets = self
                        .argmax_repetition_offsets
                        .as_ref()
                        .expect("Qwen3.5 repetition row offsets upload failed");
                    let token_ids = self
                        .argmax_repetition_token_ids
                        .as_ref()
                        .expect("Qwen3.5 repetition token ids upload failed");
                    let penalties = self
                        .argmax_repetition_penalties
                        .as_ref()
                        .expect("Qwen3.5 repetition penalties upload failed");
                    B::argmax_rows_f16_sparse_repetition_penalty(
                        &mut ctx,
                        &mut logits,
                        valid_token_mask,
                        row_offsets,
                        token_ids,
                        penalties,
                        total_token_ids,
                        batch_len,
                        vocab,
                    )
                }
            };
            decode_profile.argmax_us +=
                qwen35_decode_profile_stage_finish::<B>(&mut ctx, t_argmax, "qwen35_decode_argmax");
            if let Ok(tokens) = tokens {
                if tokens.iter().all(|&token| token != u32::MAX) {
                    let out = tokens.into_iter().map(|token| vec![token as f32]).collect();
                    if decode_profile_enabled {
                        decode_profile.log(batch_len, total_layers, "argmax");
                    }
                    return Ok(out);
                }
            }
        }
        let t_readback = if decode_profile_enabled {
            Some(Instant::now())
        } else {
            None
        };
        B::sync_before_host_readback(&mut ctx);
        B::sync(&mut ctx);
        let flat = B::to_vec(&logits, batch_len * vocab);
        if let Some(start) = t_readback {
            decode_profile.readback_us += start.elapsed().as_micros() as u64;
            decode_profile.log(batch_len, total_layers, "full_logits");
        }
        Ok(flat.chunks_exact(vocab).map(|row| row.to_vec()).collect())
    }

    fn ensure_argmax_token_mask(
        &mut self,
        ctx: &mut B::Context,
        mask: &TokenSelectionMask,
    ) -> &B::Buffer {
        let needs_upload = self.argmax_token_mask_fingerprint != Some(mask.fingerprint)
            || self.argmax_token_mask_len != mask.len();
        if needs_upload {
            let mut device_mask = B::alloc_typed(Dtype::I8, mask.len().max(1));
            B::write_typed::<i8>(ctx, &mut device_mask, &mask.valid_token_mask);
            self.argmax_token_mask = Some(device_mask);
            self.argmax_token_mask_fingerprint = Some(mask.fingerprint);
            self.argmax_token_mask_len = mask.len();
        }
        self.argmax_token_mask
            .as_ref()
            .expect("Qwen3.5 argmax token mask upload failed")
    }

    fn upload_argmax_repetition_penalty(
        &mut self,
        ctx: &mut B::Context,
        penalties: &[Option<&GreedyRepetitionPenalty>],
        batch_len: usize,
    ) -> usize {
        let mut offsets = Vec::with_capacity(batch_len + 1);
        let mut token_ids = Vec::new();
        let mut row_penalties = Vec::with_capacity(batch_len);
        offsets.push(0u32);
        for row in 0..batch_len {
            let penalty = penalties.get(row).and_then(|value| *value);
            row_penalties.push(penalty.map(|value| value.penalty).unwrap_or(1.0));
            let before = token_ids.len();
            if let Some(penalty) = penalty {
                let mut seen = HashSet::new();
                for &token_id in penalty.token_ids.iter() {
                    if seen.insert(token_id) {
                        token_ids.push(token_id);
                    }
                }
            }
            debug_assert!(token_ids.len() >= before);
            offsets.push(token_ids.len() as u32);
        }

        let offsets_len = offsets.len().max(1);
        if self.argmax_repetition_offsets_capacity < offsets_len {
            self.argmax_repetition_offsets = Some(B::alloc_typed(Dtype::U32, offsets_len));
            self.argmax_repetition_offsets_capacity = offsets_len;
        }
        let token_ids_len = token_ids.len().max(1);
        if self.argmax_repetition_token_ids_capacity < token_ids_len {
            self.argmax_repetition_token_ids = Some(B::alloc_typed(Dtype::U32, token_ids_len));
            self.argmax_repetition_token_ids_capacity = token_ids_len;
        }
        let penalties_len = row_penalties.len().max(1);
        if self.argmax_repetition_penalties_capacity < penalties_len {
            self.argmax_repetition_penalties = Some(B::alloc_typed(Dtype::F32, penalties_len));
            self.argmax_repetition_penalties_capacity = penalties_len;
        }

        B::write_typed::<u32>(
            ctx,
            self.argmax_repetition_offsets
                .as_mut()
                .expect("Qwen3.5 repetition row offsets allocation failed"),
            &offsets,
        );
        if token_ids.is_empty() {
            B::write_typed::<u32>(
                ctx,
                self.argmax_repetition_token_ids
                    .as_mut()
                    .expect("Qwen3.5 repetition token ids allocation failed"),
                &[0],
            );
        } else {
            B::write_typed::<u32>(
                ctx,
                self.argmax_repetition_token_ids
                    .as_mut()
                    .expect("Qwen3.5 repetition token ids allocation failed"),
                &token_ids,
            );
        }
        B::write_typed::<f32>(
            ctx,
            self.argmax_repetition_penalties
                .as_mut()
                .expect("Qwen3.5 repetition penalties allocation failed"),
            &row_penalties,
        );

        token_ids.len()
    }
}

#[derive(Default)]
struct Qwen35PrefillProfile {
    embedding_us: u64,
    linear_layer_us: u64,
    full_layer_us: u64,
    linear_layers: usize,
    full_layers: usize,
    final_norm_us: u64,
    final_gather_us: u64,
    lm_head_us: u64,
    readback_us: u64,
    slow_layers: Vec<(usize, &'static str, u64)>,
}

static QWEN35_PREFILL_PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
struct Qwen35DecodeProfile {
    embedding_us: u64,
    linear_layer_us: u64,
    full_layer_us: u64,
    linear_layers: usize,
    full_layers: usize,
    final_norm_us: u64,
    lm_head_us: u64,
    argmax_us: u64,
    readback_us: u64,
    slow_layers: Vec<(usize, &'static str, u64)>,
}

#[derive(Default)]
struct Qwen35LinearDecodeDetailProfile {
    input_norm_us: u64,
    qkv_proj_us: u64,
    z_proj_us: u64,
    b_proj_us: u64,
    a_proj_us: u64,
    qkvz_proj_us: u64,
    ba_proj_us: u64,
    indexed_prepare_us: u64,
    indexed_recurrent_us: u64,
    fallback_state_gather_us: u64,
    fallback_prepare_us: u64,
    fallback_recurrent_us: u64,
    fallback_state_scatter_us: u64,
    gated_norm_us: u64,
    f32_to_activation_us: u64,
    out_proj_us: u64,
    residual_update_us: u64,
    mlp_us: u64,
}

#[derive(Default)]
struct Qwen35SparseMoeDetailProfile {
    router_us: u64,
    routed_experts_us: u64,
    shared_gate_us: u64,
    shared_gate_proj_us: u64,
    shared_up_proj_us: u64,
    shared_pack_us: u64,
    shared_fused_us: u64,
    shared_down_us: u64,
    shared_apply_gate_us: u64,
    merge_us: u64,
    total_us: u64,
}

static QWEN35_DECODE_PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);
static QWEN35_LINEAR_DECODE_DETAIL_PROFILE_EVENTS: AtomicU64 = AtomicU64::new(0);
static QWEN35_SPARSE_MOE_DETAIL_PROFILE_EVENTS: AtomicU64 = AtomicU64::new(0);

fn qwen35_env_flag_enabled(name: &str) -> bool {
    let Some(value) = std::env::var_os(name) else {
        return false;
    };
    let value = value.to_string_lossy();
    let value = value.trim();
    !matches!(
        value.to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "off" | "no"
    )
}

fn qwen35_prefill_profile_enabled() -> bool {
    qwen35_env_flag_enabled("FERRUM_QWEN35_PREFILL_PROFILE")
}

fn qwen35_decode_profile_enabled() -> bool {
    qwen35_env_flag_enabled("FERRUM_QWEN35_DECODE_PROFILE")
}

fn qwen35_layer_detail_profile_enabled() -> bool {
    qwen35_env_flag_enabled("FERRUM_QWEN35_LAYER_DETAIL_PROFILE")
}

fn qwen35_prefill_profile_stage_start<B: Backend>(
    ctx: &mut B::Context,
    enabled: bool,
) -> Option<B::Timer> {
    ferrum_kernels::backend::timer::start_probe_timer_if::<B>(enabled, ctx)
}

fn qwen35_prefill_profile_stage_finish<B: Backend>(
    ctx: &mut B::Context,
    timer: Option<B::Timer>,
    name: &str,
) -> u64 {
    ferrum_kernels::backend::timer::finish_probe_timer_traced::<B>(
        timer,
        ctx,
        name,
        "qwen35_prefill",
        0,
    )
    .unwrap_or(0)
}

fn qwen35_decode_profile_stage_start<B: Backend>(
    ctx: &mut B::Context,
    enabled: bool,
) -> Option<B::Timer> {
    ferrum_kernels::backend::timer::start_probe_timer_if::<B>(enabled, ctx)
}

fn qwen35_decode_profile_stage_finish<B: Backend>(
    ctx: &mut B::Context,
    timer: Option<B::Timer>,
    name: &str,
) -> u64 {
    ferrum_kernels::backend::timer::finish_probe_timer_traced::<B>(
        timer,
        ctx,
        name,
        "qwen35_decode",
        0,
    )
    .unwrap_or(0)
}

fn qwen35_detail_profile_stage_start<B: Backend>(
    ctx: &mut B::Context,
    enabled: bool,
) -> Option<B::Timer> {
    ferrum_kernels::backend::timer::start_probe_timer_if::<B>(enabled, ctx)
}

fn qwen35_detail_profile_stage_finish<B: Backend>(
    ctx: &mut B::Context,
    timer: Option<B::Timer>,
    name: &str,
) -> u64 {
    ferrum_kernels::backend::timer::finish_probe_timer_traced::<B>(
        timer,
        ctx,
        name,
        "qwen35_detail",
        0,
    )
    .unwrap_or(0)
}

impl Qwen35LinearDecodeDetailProfile {
    fn log(&self, layer_index: usize, batch: usize, used_indexed: bool) {
        let event = QWEN35_LINEAR_DECODE_DETAIL_PROFILE_EVENTS.fetch_add(1, Ordering::Relaxed) + 1;
        let projection_us = self.qkv_proj_us
            + self.z_proj_us
            + self.b_proj_us
            + self.a_proj_us
            + self.qkvz_proj_us
            + self.ba_proj_us;
        let indexed_core_us = self.indexed_prepare_us + self.indexed_recurrent_us;
        let fallback_core_us = self.fallback_prepare_us
            + self.fallback_recurrent_us
            + self.fallback_state_gather_us
            + self.fallback_state_scatter_us;
        let attention_post_us = self.gated_norm_us + self.f32_to_activation_us + self.out_proj_us;
        let accounted_us = self.input_norm_us
            + projection_us
            + indexed_core_us
            + fallback_core_us
            + attention_post_us
            + self.residual_update_us
            + self.mlp_us;

        if event <= 96 || event % 512 == 0 {
            eprintln!(
                "[qwen35-linear-decode-detail] event#{} layer={} batch={} indexed={} \
                 accounted={}us input_norm={}us projections={}us qkv={}us z={}us b={}us a={}us \
                 qkvz={}us ba={}us \
                 indexed_core={}us indexed_prepare={}us indexed_recurrent={}us \
                 fallback_core={}us gated_norm={}us f32_to_activation={}us out_proj={}us \
                 residual_update={}us mlp={}us",
                event,
                layer_index,
                batch,
                used_indexed,
                accounted_us,
                self.input_norm_us,
                projection_us,
                self.qkv_proj_us,
                self.z_proj_us,
                self.b_proj_us,
                self.a_proj_us,
                self.qkvz_proj_us,
                self.ba_proj_us,
                indexed_core_us,
                self.indexed_prepare_us,
                self.indexed_recurrent_us,
                fallback_core_us,
                self.gated_norm_us,
                self.f32_to_activation_us,
                self.out_proj_us,
                self.residual_update_us,
                self.mlp_us,
            );
        }

        let profile = ferrum_bench_core::global_profile();
        if profile.is_enabled() {
            let _ = profile.push_event(
                "qwen35_linear_decode_detail",
                ferrum_bench_core::profile_fields_from_json(serde_json::json!({
                    "event": event,
                    "layer": layer_index,
                    "batch": batch,
                    "used_indexed": used_indexed,
                })),
                ferrum_bench_core::profile_fields_from_json(serde_json::json!({
                    "accounted": accounted_us,
                    "input_norm": self.input_norm_us,
                    "projection": projection_us,
                    "qkv_proj": self.qkv_proj_us,
                    "z_proj": self.z_proj_us,
                    "b_proj": self.b_proj_us,
                    "a_proj": self.a_proj_us,
                    "qkvz_proj": self.qkvz_proj_us,
                    "ba_proj": self.ba_proj_us,
                    "indexed_core": indexed_core_us,
                    "indexed_prepare": self.indexed_prepare_us,
                    "indexed_recurrent": self.indexed_recurrent_us,
                    "fallback_core": fallback_core_us,
                    "fallback_state_gather": self.fallback_state_gather_us,
                    "fallback_prepare": self.fallback_prepare_us,
                    "fallback_recurrent": self.fallback_recurrent_us,
                    "fallback_state_scatter": self.fallback_state_scatter_us,
                    "attention_post": attention_post_us,
                    "gated_norm": self.gated_norm_us,
                    "f32_to_activation": self.f32_to_activation_us,
                    "out_proj": self.out_proj_us,
                    "residual_update": self.residual_update_us,
                    "mlp": self.mlp_us,
                })),
                false,
            );
        }
    }
}

impl Qwen35SparseMoeDetailProfile {
    fn log(&self, layer_index: usize, tokens: usize, top_k: usize, num_experts: usize) {
        let event = QWEN35_SPARSE_MOE_DETAIL_PROFILE_EVENTS.fetch_add(1, Ordering::Relaxed) + 1;
        let shared_projection_us = self.shared_gate_proj_us + self.shared_up_proj_us;
        let shared_path_us = self.shared_gate_us
            + shared_projection_us
            + self.shared_pack_us
            + self.shared_fused_us
            + self.shared_down_us
            + self.shared_apply_gate_us;
        let accounted_us = self.router_us + self.routed_experts_us + shared_path_us + self.merge_us;

        if event <= 96 || event % 512 == 0 {
            eprintln!(
                "[qwen35-sparse-moe-detail] event#{} layer={} tokens={} top_k={} experts={} \
                 total={}us accounted={}us router={}us routed_experts={}us shared_path={}us \
                 shared_gate={}us shared_projection={}us shared_gate_proj={}us \
                 shared_up_proj={}us shared_pack={}us shared_fused={}us shared_down={}us \
                 shared_apply_gate={}us merge={}us",
                event,
                layer_index,
                tokens,
                top_k,
                num_experts,
                self.total_us,
                accounted_us,
                self.router_us,
                self.routed_experts_us,
                shared_path_us,
                self.shared_gate_us,
                shared_projection_us,
                self.shared_gate_proj_us,
                self.shared_up_proj_us,
                self.shared_pack_us,
                self.shared_fused_us,
                self.shared_down_us,
                self.shared_apply_gate_us,
                self.merge_us,
            );
        }

        let profile = ferrum_bench_core::global_profile();
        if profile.is_enabled() {
            let _ = profile.push_event(
                "qwen35_sparse_moe_detail",
                ferrum_bench_core::profile_fields_from_json(serde_json::json!({
                    "event": event,
                    "layer": layer_index,
                    "tokens": tokens,
                    "top_k": top_k,
                    "num_experts": num_experts,
                })),
                ferrum_bench_core::profile_fields_from_json(serde_json::json!({
                    "total": self.total_us,
                    "accounted": accounted_us,
                    "router": self.router_us,
                    "routed_experts": self.routed_experts_us,
                    "shared_path": shared_path_us,
                    "shared_gate": self.shared_gate_us,
                    "shared_projection": shared_projection_us,
                    "shared_gate_proj": self.shared_gate_proj_us,
                    "shared_up_proj": self.shared_up_proj_us,
                    "shared_pack": self.shared_pack_us,
                    "shared_fused": self.shared_fused_us,
                    "shared_down": self.shared_down_us,
                    "shared_apply_gate": self.shared_apply_gate_us,
                    "merge": self.merge_us,
                })),
                false,
            );
        }
    }
}

impl Qwen35DecodeProfile {
    fn record_layer(&mut self, layer_index: usize, kind: &'static str, elapsed_us: u64) {
        match kind {
            "linear" => {
                self.linear_layers += 1;
                self.linear_layer_us += elapsed_us;
            }
            "full" => {
                self.full_layers += 1;
                self.full_layer_us += elapsed_us;
            }
            _ => {}
        }

        let insert_at = self
            .slow_layers
            .iter()
            .position(|&(_, _, value)| elapsed_us > value)
            .unwrap_or(self.slow_layers.len());
        if insert_at < 8 {
            self.slow_layers
                .insert(insert_at, (layer_index, kind, elapsed_us));
            self.slow_layers.truncate(8);
        }
    }

    fn log(&self, batch: usize, total_layers: usize, logits_return: &'static str) {
        let call = QWEN35_DECODE_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        if call > 16 && call % 32 != 0 {
            return;
        }

        let layer_sum = self.linear_layer_us + self.full_layer_us;
        let final_sum = self.final_norm_us + self.lm_head_us + self.argmax_us + self.readback_us;
        eprintln!(
            "[qwen35-decode-prof] call#{} batch={} logits_return={} layers={} \
             linear_layers={} full_layers={} embedding={}us layer_sum={}us \
             linear_layer_sum={}us full_layer_sum={}us final_sum={}us final_norm={}us \
             lm_head={}us argmax={}us readback={}us slow_layers={:?}",
            call,
            batch,
            logits_return,
            total_layers,
            self.linear_layers,
            self.full_layers,
            self.embedding_us,
            layer_sum,
            self.linear_layer_us,
            self.full_layer_us,
            final_sum,
            self.final_norm_us,
            self.lm_head_us,
            self.argmax_us,
            self.readback_us,
            self.slow_layers,
        );

        let profile = ferrum_bench_core::global_profile();
        if profile.is_enabled() {
            let slow_layers = self
                .slow_layers
                .iter()
                .map(|(layer, kind, us)| {
                    serde_json::json!({
                        "layer": layer,
                        "kind": kind,
                        "us": us,
                    })
                })
                .collect::<Vec<_>>();
            let _ = profile.push_event(
                "qwen35_decode_prof",
                ferrum_bench_core::profile_fields_from_json(serde_json::json!({
                    "call": call,
                    "batch": batch,
                    "layers": total_layers,
                    "linear_layers": self.linear_layers,
                    "full_layers": self.full_layers,
                    "logits_return": logits_return,
                })),
                ferrum_bench_core::profile_fields_from_json(serde_json::json!({
                    "embedding": self.embedding_us,
                    "layer_sum": layer_sum,
                    "linear_layer_sum": self.linear_layer_us,
                    "full_layer_sum": self.full_layer_us,
                    "final_sum": final_sum,
                    "final_norm": self.final_norm_us,
                    "lm_head": self.lm_head_us,
                    "argmax": self.argmax_us,
                    "readback": self.readback_us,
                    "slow_layers": slow_layers,
                })),
                false,
            );
        }
    }
}

impl Qwen35PrefillProfile {
    fn record_layer(&mut self, layer_index: usize, kind: &'static str, elapsed_us: u64) {
        match kind {
            "linear" => {
                self.linear_layers += 1;
                self.linear_layer_us += elapsed_us;
            }
            "full" => {
                self.full_layers += 1;
                self.full_layer_us += elapsed_us;
            }
            _ => {}
        }

        let insert_at = self
            .slow_layers
            .iter()
            .position(|&(_, _, value)| elapsed_us > value)
            .unwrap_or(self.slow_layers.len());
        if insert_at < 8 {
            self.slow_layers
                .insert(insert_at, (layer_index, kind, elapsed_us));
            self.slow_layers.truncate(8);
        }
    }

    fn log(&self, cache_id: &str, prior_len: usize, tokens_len: usize, total_layers: usize) {
        let call = QWEN35_PREFILL_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        if call > 16 && call % 64 != 0 {
            return;
        }

        let layer_sum = self.linear_layer_us + self.full_layer_us;
        let final_sum =
            self.final_norm_us + self.final_gather_us + self.lm_head_us + self.readback_us;
        eprintln!(
            "[qwen35-prefill-prof] call#{} cache_id={cache_id:?} prior_len={} tokens={} \
             layers={} linear_layers={} full_layers={} embedding={}us layer_sum={}us \
             linear_layer_sum={}us full_layer_sum={}us final_sum={}us final_norm={}us \
             final_gather={}us lm_head={}us readback={}us slow_layers={:?}",
            call,
            prior_len,
            tokens_len,
            total_layers,
            self.linear_layers,
            self.full_layers,
            self.embedding_us,
            layer_sum,
            self.linear_layer_us,
            self.full_layer_us,
            final_sum,
            self.final_norm_us,
            self.final_gather_us,
            self.lm_head_us,
            self.readback_us,
            self.slow_layers,
        );

        let profile = ferrum_bench_core::global_profile();
        if profile.is_enabled() {
            let slow_layers = self
                .slow_layers
                .iter()
                .map(|(layer, kind, us)| {
                    serde_json::json!({
                        "layer": layer,
                        "kind": kind,
                        "us": us,
                    })
                })
                .collect::<Vec<_>>();
            let _ = profile.push_event(
                "qwen35_prefill_prof",
                ferrum_bench_core::profile_fields_from_json(serde_json::json!({
                    "call": call,
                    "prior_len": prior_len,
                    "tokens": tokens_len,
                    "layers": total_layers,
                    "linear_layers": self.linear_layers,
                    "full_layers": self.full_layers,
                })),
                ferrum_bench_core::profile_fields_from_json(serde_json::json!({
                    "embedding": self.embedding_us,
                    "layer_sum": layer_sum,
                    "linear_layer_sum": self.linear_layer_us,
                    "full_layer_sum": self.full_layer_us,
                    "final_sum": final_sum,
                    "final_norm": self.final_norm_us,
                    "final_gather": self.final_gather_us,
                    "lm_head": self.lm_head_us,
                    "readback": self.readback_us,
                    "slow_layers": slow_layers,
                })),
                false,
            );
        }
    }
}

fn qwen35_trace_buffer_stats<B: Backend>(
    ctx: &mut B::Context,
    label: &str,
    buffer: &B::Buffer,
    len: usize,
) {
    if std::env::var_os("FERRUM_QWEN35_BUFFER_TRACE").is_none() {
        return;
    }
    B::sync_before_host_readback(ctx);
    B::sync(ctx);
    let values = B::to_vec(buffer, len);
    let mut finite = 0usize;
    let mut nan = 0usize;
    let mut infinite = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut first_nonfinite = None;
    for (idx, value) in values.iter().copied().enumerate() {
        if value.is_finite() {
            finite += 1;
            min = min.min(value);
            max = max.max(value);
        } else {
            if value.is_nan() {
                nan += 1;
            } else {
                infinite += 1;
            }
            first_nonfinite.get_or_insert((idx, value));
        }
    }
    if finite == 0 {
        min = f32::NAN;
        max = f32::NAN;
    }
    eprintln!(
        "[qwen35-buffer-trace] label={label} len={len} finite={finite} nan={nan} inf={infinite} \
         min={min:?} max={max:?} first_nonfinite={first_nonfinite:?}"
    );
}

fn qwen35_trace_layer_buffer_stats<B: Backend>(
    ctx: &mut B::Context,
    layer_index: usize,
    name: &str,
    buffer: &B::Buffer,
    len: usize,
) {
    let Some(filter) = std::env::var_os("FERRUM_QWEN35_LAYER_TRACE") else {
        return;
    };
    let filter = filter.to_string_lossy();
    let matches = filter
        .split(',')
        .map(str::trim)
        .any(|value| value == "all" || value.parse::<usize>().ok() == Some(layer_index));
    if !matches {
        return;
    }
    let label = format!("layer{layer_index}.{name}");
    qwen35_trace_buffer_stats::<B>(ctx, &label, buffer, len);
}

fn qwen35_trace_layer_moe_route(
    layer_index: usize,
    tokens: usize,
    num_experts: usize,
    top_k: usize,
    expert_ids: &[u32],
    expert_weights: &[f32],
) {
    let Some(filter) = std::env::var_os("FERRUM_QWEN35_LAYER_TRACE") else {
        return;
    };
    let filter = filter.to_string_lossy();
    let matches = filter
        .split(',')
        .map(str::trim)
        .any(|value| value == "all" || value.parse::<usize>().ok() == Some(layer_index));
    if !matches {
        return;
    }

    let rows = tokens.min(4);
    let mut first_rows = Vec::with_capacity(rows);
    for token in 0..rows {
        let mut row = Vec::with_capacity(top_k);
        for slot in 0..top_k {
            let pair = token * top_k + slot;
            if pair < expert_ids.len() && pair < expert_weights.len() {
                row.push((expert_ids[pair], expert_weights[pair]));
            }
        }
        first_rows.push(row);
    }

    let mut counts = vec![0usize; num_experts];
    for expert_id in expert_ids {
        let expert = *expert_id as usize;
        if expert < counts.len() {
            counts[expert] += 1;
        }
    }
    let mut top_counts = counts
        .into_iter()
        .enumerate()
        .filter(|(_, count)| *count > 0)
        .collect::<Vec<_>>();
    top_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    top_counts.truncate(16);

    eprintln!(
        "[qwen35-moe-route-trace] layer={layer_index} tokens={tokens} top_k={top_k} \
         first_rows={first_rows:?} top_expert_counts={top_counts:?}"
    );
}

impl<B: MoeLlmBackend + BackendPagedKv> DecoderOnlyLLM for Qwen35BackendModel<B> {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.weights.runtime_cfg
    }

    fn kv_capacity(&self) -> usize {
        self.kv_capacity
    }

    fn prefill(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        self.forward_stateful(cache_id, tokens)
            .unwrap_or_else(|err| panic!("Qwen3.5 prefill failed: {err}"))
    }

    fn decode(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        if let Some(state) = self.sequences.get(cache_id) {
            debug_assert_eq!(
                state.tokens.len(),
                pos as usize,
                "Qwen3.5 decode position mismatch"
            );
        }
        self.forward_stateful(cache_id, &[token])
            .unwrap_or_else(|err| panic!("Qwen3.5 decode failed: {err}"))
    }

    fn decode_batch(&mut self, batch: &[(String, u32, u32)]) -> Vec<Vec<f32>> {
        self.decode_batch_with_full_logits(batch, false)
    }

    fn decode_batch_with_full_logits(
        &mut self,
        batch: &[(String, u32, u32)],
        force_full_logits: bool,
    ) -> Vec<Vec<f32>> {
        if batch.len() <= 1 && force_full_logits {
            return batch
                .iter()
                .map(|(cache_id, token, pos)| self.decode(cache_id, *token, *pos))
                .collect();
        }
        self.forward_stateful_decode_batch_with_logits_return(
            batch,
            if force_full_logits {
                Qwen35DecodeLogitsReturn::Full
            } else {
                Qwen35DecodeLogitsReturn::LegacyDefault
            },
        )
        .unwrap_or_else(|err| panic!("Qwen3.5 decode_batch failed: {err}"))
    }

    fn decode_batch_with_logits_policy(
        &mut self,
        batch: &[(String, u32, u32)],
        policies: &[LogitsReturnPolicy],
    ) -> Vec<Vec<f32>> {
        let logits_return = qwen35_decode_logits_return_from_policies(policies, batch.len());
        if batch.len() <= 1 && matches!(logits_return, Qwen35DecodeLogitsReturn::Full) {
            return self.decode_batch_with_full_logits(batch, true);
        }
        self.forward_stateful_decode_batch_with_logits_return(batch, logits_return)
            .unwrap_or_else(|err| panic!("Qwen3.5 decode_batch failed: {err}"))
    }

    fn unified_forward(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
    ) -> std::result::Result<Vec<Option<Vec<f32>>>, FerrumError> {
        self.forward_stateful_prefill_batch(items)
    }

    fn release(&mut self, cache_id: &str) {
        if let Some(mut state) = self.sequences.remove(cache_id) {
            self.release_sequence_state_blocks(&mut state);
        }
    }

    fn reset(&mut self) {
        let mut states = std::mem::take(&mut self.sequences);
        for state in states.values_mut() {
            self.release_sequence_state_blocks(state);
        }
    }
}

impl<B: MoeLlmBackend + BackendPagedKv> Qwen35BackendModel<B> {
    pub fn from_definition_with_native_safetensors(
        def: &ModelDefinition,
        model_dir: &Path,
    ) -> Result<Self> {
        let config = Qwen35TextConfig::from_model_definition(def)
            .map_err(|err| FerrumError::model(format!("invalid Qwen3.5/Qwen3.6 config: {err}")))?;
        let inventory = Qwen35WeightInventory::from_safetensors_dir(model_dir)
            .map_err(|err| FerrumError::model(format!("Qwen3.5 weight inventory failed: {err}")))?;
        let weight_plan = inventory
            .detect_prefix_and_resolve(&config)
            .map_err(|err| FerrumError::model(format!("Qwen3.5 weight preflight failed: {err}")))?;
        let runtime_cfg = qwen35_runtime_config(
            &config,
            def.vocab_size,
            qwen35_effective_max_seq_len(def.max_position_embeddings),
        );
        let loader = NativeSafetensorsLoader::<B>::open(model_dir)?;
        Self::from_weight_plan(config, runtime_cfg, weight_plan, &loader)
    }
}

pub fn qwen35_dense_reference_model_forward_cpu(
    model: Qwen35DenseReferenceModel<'_>,
    input_ids: &[usize],
) -> Result<Qwen35DenseReferenceModelOutput> {
    model.validate(input_ids.len())?;
    if input_ids.is_empty() {
        return Err(FerrumError::model(
            "Qwen3.5 dense reference forward requires at least one input token",
        ));
    }
    let tokens = input_ids.len();
    let mut hidden = gather_embedding_rows(
        model.embed_tokens,
        input_ids,
        model.vocab_size,
        model.hidden_size,
    )?;
    let mut layer_hidden_states = Vec::with_capacity(model.layers.len());
    let mut linear_conv_states = Vec::new();
    let mut linear_recurrent_states = Vec::new();

    for layer in model.layers {
        match layer {
            Qwen35DenseReferenceLayer::Linear(layer) => {
                if layer.shape.tokens != tokens || layer.shape.hidden_size != model.hidden_size {
                    return Err(FerrumError::model(format!(
                        "Qwen3.5 dense reference linear layer shape {:?} does not match tokens={tokens} hidden={}",
                        layer.shape, model.hidden_size
                    )));
                }
                let state_len = layer.shape.attention.state_len();
                let initial_state = vec![0.0; state_len];
                let out = qwen35_dense_linear_attention_layer_cpu(
                    &hidden,
                    layer.input_norm_weight,
                    layer.qkv_weight,
                    layer.z_weight,
                    layer.b_weight,
                    layer.a_weight,
                    layer.conv1d_weight,
                    layer.a_log,
                    layer.dt_bias,
                    layer.norm_weight,
                    layer.out_proj_weight,
                    layer.post_attention_norm_weight,
                    layer.gate_proj_weight,
                    layer.up_proj_weight,
                    layer.down_proj_weight,
                    &initial_state,
                    layer.shape,
                    model.eps,
                )?;
                linear_conv_states.push(out.attention.final_conv_state.clone());
                linear_recurrent_states.push(out.attention.final_state);
                hidden = out.layer_output;
            }
            Qwen35DenseReferenceLayer::Full(layer) => {
                if layer.shape.tokens != tokens || layer.shape.hidden_size != model.hidden_size {
                    return Err(FerrumError::model(format!(
                        "Qwen3.5 dense reference full layer shape {:?} does not match tokens={tokens} hidden={}",
                        layer.shape, model.hidden_size
                    )));
                }
                let out = qwen35_dense_full_attention_layer_cpu(
                    &hidden,
                    layer.input_norm_weight,
                    layer.q_weight,
                    layer.k_weight,
                    layer.v_weight,
                    layer.q_norm_weight,
                    layer.k_norm_weight,
                    layer.o_weight,
                    layer.post_attention_norm_weight,
                    layer.gate_proj_weight,
                    layer.up_proj_weight,
                    layer.down_proj_weight,
                    layer.shape,
                    model.eps,
                )?;
                hidden = out.layer_output;
            }
        }
        layer_hidden_states.push(hidden.clone());
    }

    let final_hidden = qwen35_rms_norm_plus_one_cpu(
        &hidden,
        model.final_norm_weight,
        tokens,
        model.hidden_size,
        model.eps,
    )?;
    let logits = qwen35_linear_cpu(
        &final_hidden,
        model.lm_head_weight,
        tokens,
        model.hidden_size,
        model.vocab_size,
    )?;

    Ok(Qwen35DenseReferenceModelOutput {
        hidden,
        final_hidden,
        logits,
        layer_hidden_states,
        linear_conv_states,
        linear_recurrent_states,
    })
}

pub fn qwen35_sparse_moe_reference_model_forward_cpu(
    model: Qwen35SparseMoeReferenceModel<'_>,
    input_ids: &[usize],
) -> Result<Qwen35SparseMoeReferenceModelOutput> {
    model.validate(input_ids.len())?;
    if input_ids.is_empty() {
        return Err(FerrumError::model(
            "Qwen3.5 sparse-MoE reference forward requires at least one input token",
        ));
    }
    let tokens = input_ids.len();
    let mut hidden = gather_embedding_rows(
        model.embed_tokens,
        input_ids,
        model.vocab_size,
        model.hidden_size,
    )?;
    let mut layer_hidden_states = Vec::with_capacity(model.layers.len());
    let mut linear_conv_states = Vec::new();
    let mut linear_recurrent_states = Vec::new();
    let mut sparse_moe_outputs = Vec::with_capacity(model.layers.len());

    for layer in model.layers {
        match layer {
            Qwen35SparseMoeReferenceLayer::Linear(layer) => {
                if layer.shape.tokens != tokens || layer.shape.hidden_size != model.hidden_size {
                    return Err(FerrumError::model(format!(
                        "Qwen3.5 sparse-MoE reference linear layer shape {:?} does not match tokens={tokens} hidden={}",
                        layer.shape, model.hidden_size
                    )));
                }
                let state_len = layer.shape.attention.state_len();
                let initial_state = vec![0.0; state_len];
                let out = qwen35_sparse_moe_linear_attention_layer_cpu(
                    &hidden,
                    layer.input_norm_weight,
                    layer.qkv_weight,
                    layer.z_weight,
                    layer.b_weight,
                    layer.a_weight,
                    layer.conv1d_weight,
                    layer.a_log,
                    layer.dt_bias,
                    layer.norm_weight,
                    layer.out_proj_weight,
                    layer.post_attention_norm_weight,
                    layer.router_weight,
                    layer.fused_gate_up_proj,
                    layer.fused_down_proj,
                    layer.shared_expert_gate_weight,
                    layer.shared_expert_gate_proj_weight,
                    layer.shared_expert_up_proj_weight,
                    layer.shared_expert_down_proj_weight,
                    &initial_state,
                    layer.shape,
                    model.eps,
                )?;
                linear_conv_states.push(out.attention.final_conv_state.clone());
                linear_recurrent_states.push(out.attention.final_state);
                sparse_moe_outputs.push(out.moe);
                hidden = out.layer_output;
            }
            Qwen35SparseMoeReferenceLayer::Full(layer) => {
                if layer.shape.tokens != tokens || layer.shape.hidden_size != model.hidden_size {
                    return Err(FerrumError::model(format!(
                        "Qwen3.5 sparse-MoE reference full layer shape {:?} does not match tokens={tokens} hidden={}",
                        layer.shape, model.hidden_size
                    )));
                }
                let out = qwen35_sparse_moe_full_attention_layer_cpu(
                    &hidden,
                    layer.input_norm_weight,
                    layer.q_weight,
                    layer.k_weight,
                    layer.v_weight,
                    layer.q_norm_weight,
                    layer.k_norm_weight,
                    layer.o_weight,
                    layer.post_attention_norm_weight,
                    layer.router_weight,
                    layer.fused_gate_up_proj,
                    layer.fused_down_proj,
                    layer.shared_expert_gate_weight,
                    layer.shared_expert_gate_proj_weight,
                    layer.shared_expert_up_proj_weight,
                    layer.shared_expert_down_proj_weight,
                    layer.shape,
                    model.eps,
                )?;
                sparse_moe_outputs.push(out.moe);
                hidden = out.layer_output;
            }
        }
        layer_hidden_states.push(hidden.clone());
    }

    let final_hidden = qwen35_rms_norm_plus_one_cpu(
        &hidden,
        model.final_norm_weight,
        tokens,
        model.hidden_size,
        model.eps,
    )?;
    let logits = qwen35_linear_cpu(
        &final_hidden,
        model.lm_head_weight,
        tokens,
        model.hidden_size,
        model.vocab_size,
    )?;

    Ok(Qwen35SparseMoeReferenceModelOutput {
        hidden,
        final_hidden,
        logits,
        layer_hidden_states,
        linear_conv_states,
        linear_recurrent_states,
        sparse_moe_outputs,
    })
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
    let q_proj_total = attention_shape.q_proj_total();
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
        q_proj_total * shape.hidden_size,
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
        q_proj_total,
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

pub fn qwen35_dense_full_attention_layer_backend<B: MoeLlmBackend>(
    ctx: &mut B::Context,
    layer_input: &B::Buffer,
    layer: &Qwen35LayerWeights<B>,
    rope: &Qwen35BackendRopeCache<B>,
    shape: Qwen35DenseFullAttentionLayerShape,
    eps: f32,
) -> Result<Qwen35BackendDenseFullAttentionLayerOutput<B>> {
    shape.validate()?;
    let attention_shape = shape.attention;
    let hidden_len = shape.tokens * shape.hidden_size;
    let q_total = attention_shape.q_total();
    let q_proj_total = attention_shape.q_proj_total();
    let kv_total = attention_shape.kv_total();
    let attention = match &layer.attention {
        Qwen35AttentionWeights::Full(attention) => attention,
        Qwen35AttentionWeights::Linear(_) => {
            return Err(FerrumError::model(format!(
                "Qwen3.5 backend dense full layer expected full attention at layer {}",
                layer.layer_index
            )));
        }
    };
    let mlp = match &layer.mlp {
        Qwen35MlpWeights::Dense(mlp) => mlp,
        Qwen35MlpWeights::SparseMoeSharedExpert(_) => {
            return Err(FerrumError::model(format!(
                "Qwen3.5 backend dense full layer expected dense MLP at layer {}",
                layer.layer_index
            )));
        }
    };

    let mut input_norm = B::alloc(hidden_len);
    B::rms_norm(
        ctx,
        layer_input,
        &layer.input_layernorm,
        eps,
        &mut input_norm,
        shape.tokens,
        shape.hidden_size,
    );
    let mut query_raw = B::alloc(shape.tokens * q_proj_total);
    let mut key_raw = B::alloc(shape.tokens * kv_total);
    let mut value_raw = B::alloc(shape.tokens * kv_total);
    attention
        .q_proj
        .forward(ctx, &input_norm, &mut query_raw, shape.tokens);
    attention
        .k_proj
        .forward(ctx, &input_norm, &mut key_raw, shape.tokens);
    attention
        .v_proj
        .forward(ctx, &input_norm, &mut value_raw, shape.tokens);

    let attention_out = qwen35_full_attention_core_backend::<B>(
        ctx,
        &query_raw,
        &key_raw,
        &value_raw,
        &attention.q_norm_weight,
        &attention.k_norm_weight,
        &rope.cos,
        &rope.sin,
        attention_shape,
        eps,
    )?;
    let mut attn_output = B::alloc(hidden_len);
    attention
        .o_proj
        .forward(ctx, &attention_out.context, &mut attn_output, shape.tokens);

    let mut residual_after_attention = B::alloc(hidden_len);
    B::copy_slice(
        ctx,
        layer_input,
        0,
        &mut residual_after_attention,
        0,
        hidden_len,
    );
    B::add_inplace(ctx, &mut residual_after_attention, &attn_output, hidden_len);

    let mut post_attention_norm = B::alloc(hidden_len);
    B::rms_norm(
        ctx,
        &residual_after_attention,
        &layer.post_attention_layernorm,
        eps,
        &mut post_attention_norm,
        shape.tokens,
        shape.hidden_size,
    );
    let mlp_out = qwen35_dense_mlp_backend::<B>(
        ctx,
        &post_attention_norm,
        &*mlp.gate_up_proj,
        &*mlp.down_proj,
        shape.tokens,
        shape.hidden_size,
        shape.intermediate_size,
    )?;
    let mut layer_output = B::alloc(hidden_len);
    B::copy_slice(
        ctx,
        &residual_after_attention,
        0,
        &mut layer_output,
        0,
        hidden_len,
    );
    B::add_inplace(ctx, &mut layer_output, &mlp_out.output, hidden_len);

    Ok(Qwen35BackendDenseFullAttentionLayerOutput {
        input_norm,
        query_raw,
        key_raw,
        value_raw,
        attention: attention_out,
        attn_output,
        residual_after_attention,
        post_attention_norm,
        mlp: mlp_out,
        layer_output,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_sparse_moe_linear_attention_layer_cpu(
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
    router_weight: &[f32],
    fused_gate_up_proj: &[f32],
    fused_down_proj: &[f32],
    shared_expert_gate_weight: &[f32],
    shared_expert_gate_proj_weight: &[f32],
    shared_expert_up_proj_weight: &[f32],
    shared_expert_down_proj_weight: &[f32],
    initial_state: &[f32],
    shape: Qwen35SparseMoeLinearAttentionLayerShape,
    eps: f32,
) -> Result<Qwen35SparseMoeLinearAttentionLayerReference> {
    shape.validate()?;
    let attention_shape = shape.attention;
    let hidden_len = shape.tokens * shape.hidden_size;
    let value_total = attention_shape.value_total();
    validate_len(
        "sparse MoE linear layer input",
        layer_input.len(),
        hidden_len,
    )?;
    validate_len(
        "sparse MoE linear layer input_norm_weight",
        input_norm_weight.len(),
        shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE linear layer qkv_weight",
        qkv_weight.len(),
        attention_shape.conv_channels() * shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE linear layer z_weight",
        z_weight.len(),
        value_total * shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE linear layer b_weight",
        b_weight.len(),
        attention_shape.value_heads * shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE linear layer a_weight",
        a_weight.len(),
        attention_shape.value_heads * shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE linear layer out_proj_weight",
        out_proj_weight.len(),
        shape.hidden_size * value_total,
    )?;
    validate_len(
        "sparse MoE linear layer post_attention_norm_weight",
        post_attention_norm_weight.len(),
        shape.hidden_size,
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
    let moe = qwen35_sparse_moe_shared_expert_cpu(
        &post_attention_norm,
        router_weight,
        fused_gate_up_proj,
        fused_down_proj,
        shared_expert_gate_weight,
        shared_expert_gate_proj_weight,
        shared_expert_up_proj_weight,
        shared_expert_down_proj_weight,
        shape.moe,
    )?;
    let layer_output = add_same_len(&residual_after_mixer, &moe.moe_output, "layer_output")?;

    Ok(Qwen35SparseMoeLinearAttentionLayerReference {
        input_norm,
        mixed_qkv_raw,
        z_raw,
        b_raw,
        a_raw,
        attention,
        delta_output,
        residual_after_mixer,
        post_attention_norm,
        moe,
        layer_output,
    })
}

pub fn qwen35_dense_linear_attention_layer_backend<B: MoeLlmBackend>(
    ctx: &mut B::Context,
    layer_input: &B::Buffer,
    initial_state: &B::Buffer,
    layer: &Qwen35LayerWeights<B>,
    shape: Qwen35DenseLinearAttentionLayerShape,
    eps: f32,
) -> Result<Qwen35BackendDenseLinearAttentionLayerOutput<B>> {
    shape.validate()?;
    let attention_shape = shape.attention;
    let hidden_len = shape.tokens * shape.hidden_size;
    let value_total = attention_shape.value_total();
    let attention = match &layer.attention {
        Qwen35AttentionWeights::Linear(attention) => attention,
        Qwen35AttentionWeights::Full(_) => {
            return Err(FerrumError::model(format!(
                "Qwen3.5 backend dense linear layer expected linear attention at layer {}",
                layer.layer_index
            )));
        }
    };
    let mlp = match &layer.mlp {
        Qwen35MlpWeights::Dense(mlp) => mlp,
        Qwen35MlpWeights::SparseMoeSharedExpert(_) => {
            return Err(FerrumError::model(format!(
                "Qwen3.5 backend dense linear layer expected dense MLP at layer {}",
                layer.layer_index
            )));
        }
    };

    let mut input_norm = B::alloc(hidden_len);
    B::rms_norm(
        ctx,
        layer_input,
        &layer.input_layernorm,
        eps,
        &mut input_norm,
        shape.tokens,
        shape.hidden_size,
    );
    let mut mixed_qkv_raw = B::alloc(attention_shape.mixed_qkv_len());
    let mut z_raw = B::alloc(attention_shape.value_len());
    let mut b_raw = B::alloc(attention_shape.gating_len());
    let mut a_raw = B::alloc(attention_shape.gating_len());
    attention
        .qkv_proj
        .forward(ctx, &input_norm, &mut mixed_qkv_raw, shape.tokens);
    attention
        .z_proj
        .forward(ctx, &input_norm, &mut z_raw, shape.tokens);
    attention
        .b_proj
        .forward(ctx, &input_norm, &mut b_raw, shape.tokens);
    attention
        .a_proj
        .forward(ctx, &input_norm, &mut a_raw, shape.tokens);

    let final_conv_state =
        qwen35_final_conv_state_backend::<B>(ctx, &mixed_qkv_raw, attention_shape)?;
    let attention_out = qwen35_linear_attention_prefill_core_backend::<B>(
        ctx,
        &mixed_qkv_raw,
        &z_raw,
        &a_raw,
        &b_raw,
        &attention.conv1d_weight,
        &attention.a_log,
        &attention.dt_bias,
        &attention.norm_weight,
        initial_state,
        attention_shape,
        eps,
    )?;
    let mut delta_activation = B::alloc(attention_shape.value_len());
    B::f32_to_activation(
        ctx,
        &attention_out.delta_norm,
        &mut delta_activation,
        attention_shape.value_len(),
    );
    let mut delta_output = B::alloc(hidden_len);
    attention
        .out_proj
        .forward(ctx, &delta_activation, &mut delta_output, shape.tokens);

    let mut residual_after_mixer = B::alloc(hidden_len);
    B::copy_slice(
        ctx,
        layer_input,
        0,
        &mut residual_after_mixer,
        0,
        hidden_len,
    );
    B::add_inplace(ctx, &mut residual_after_mixer, &delta_output, hidden_len);

    let mut post_attention_norm = B::alloc(hidden_len);
    B::rms_norm(
        ctx,
        &residual_after_mixer,
        &layer.post_attention_layernorm,
        eps,
        &mut post_attention_norm,
        shape.tokens,
        shape.hidden_size,
    );
    let mlp_out = qwen35_dense_mlp_backend::<B>(
        ctx,
        &post_attention_norm,
        &*mlp.gate_up_proj,
        &*mlp.down_proj,
        shape.tokens,
        shape.hidden_size,
        shape.intermediate_size,
    )?;
    let mut layer_output = B::alloc(hidden_len);
    B::copy_slice(
        ctx,
        &residual_after_mixer,
        0,
        &mut layer_output,
        0,
        hidden_len,
    );
    B::add_inplace(ctx, &mut layer_output, &mlp_out.output, hidden_len);

    Ok(Qwen35BackendDenseLinearAttentionLayerOutput {
        input_norm,
        mixed_qkv_raw,
        z_raw,
        b_raw,
        a_raw,
        attention: attention_out,
        final_conv_state,
        delta_output,
        residual_after_mixer,
        post_attention_norm,
        mlp: mlp_out,
        layer_output,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_sparse_moe_full_attention_layer_cpu(
    layer_input: &[f32],
    input_norm_weight: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    v_weight: &[f32],
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    o_weight: &[f32],
    post_attention_norm_weight: &[f32],
    router_weight: &[f32],
    fused_gate_up_proj: &[f32],
    fused_down_proj: &[f32],
    shared_expert_gate_weight: &[f32],
    shared_expert_gate_proj_weight: &[f32],
    shared_expert_up_proj_weight: &[f32],
    shared_expert_down_proj_weight: &[f32],
    shape: Qwen35SparseMoeFullAttentionLayerShape,
    eps: f32,
) -> Result<Qwen35SparseMoeFullAttentionLayerReference> {
    shape.validate()?;
    let attention_shape = shape.attention;
    let hidden_len = shape.tokens * shape.hidden_size;
    let q_total = attention_shape.q_total();
    let q_proj_total = attention_shape.q_proj_total();
    let kv_total = attention_shape.kv_total();
    validate_len("sparse MoE full layer input", layer_input.len(), hidden_len)?;
    validate_len(
        "sparse MoE full layer input_norm_weight",
        input_norm_weight.len(),
        shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE full layer q_weight",
        q_weight.len(),
        q_proj_total * shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE full layer k_weight",
        k_weight.len(),
        kv_total * shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE full layer v_weight",
        v_weight.len(),
        kv_total * shape.hidden_size,
    )?;
    validate_len(
        "sparse MoE full layer q_norm_weight",
        q_norm_weight.len(),
        attention_shape.head_dim,
    )?;
    validate_len(
        "sparse MoE full layer k_norm_weight",
        k_norm_weight.len(),
        attention_shape.head_dim,
    )?;
    validate_len(
        "sparse MoE full layer o_weight",
        o_weight.len(),
        shape.hidden_size * q_total,
    )?;
    validate_len(
        "sparse MoE full layer post_attention_norm_weight",
        post_attention_norm_weight.len(),
        shape.hidden_size,
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
        q_proj_total,
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
    let moe = qwen35_sparse_moe_shared_expert_cpu(
        &post_attention_norm,
        router_weight,
        fused_gate_up_proj,
        fused_down_proj,
        shared_expert_gate_weight,
        shared_expert_gate_proj_weight,
        shared_expert_up_proj_weight,
        shared_expert_down_proj_weight,
        shape.moe,
    )?;
    let layer_output = add_same_len(&residual_after_attention, &moe.moe_output, "layer_output")?;

    Ok(Qwen35SparseMoeFullAttentionLayerReference {
        input_norm,
        query_raw,
        key_raw,
        value_raw,
        attention,
        attn_output,
        residual_after_attention,
        post_attention_norm,
        moe,
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
    qwen35_full_attention_core_cpu_impl(
        query_raw,
        key_raw,
        value_raw,
        q_norm_weight,
        k_norm_weight,
        shape,
        eps,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn qwen35_full_attention_core_cpu_impl(
    query_raw: &[f32],
    key_raw: &[f32],
    value_raw: &[f32],
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    shape: Qwen35FullAttentionShape,
    eps: f32,
    norm_weight_folded: bool,
) -> Result<Qwen35FullAttentionReference> {
    shape.validate()?;
    validate_len(
        "full attention query_raw",
        query_raw.len(),
        shape.q_proj_len(),
    )?;
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

    let (query_source, attention_gate) = split_qwen35_full_attention_query_gate(
        query_raw,
        shape.tokens,
        shape.num_heads,
        shape.head_dim,
        shape.q_proj_total(),
        shape.attn_output_gate,
    )?;

    let mut query = if norm_weight_folded {
        qwen35_rms_norm_cpu(
            &query_source,
            q_norm_weight,
            shape.tokens * shape.num_heads,
            shape.head_dim,
            eps,
        )?
    } else {
        qwen35_rms_norm_plus_one_cpu(
            &query_source,
            q_norm_weight,
            shape.tokens * shape.num_heads,
            shape.head_dim,
            eps,
        )?
    };
    let mut key = if norm_weight_folded {
        qwen35_rms_norm_cpu(
            key_raw,
            k_norm_weight,
            shape.tokens * shape.num_kv_heads,
            shape.head_dim,
            eps,
        )?
    } else {
        qwen35_rms_norm_plus_one_cpu(
            key_raw,
            k_norm_weight,
            shape.tokens * shape.num_kv_heads,
            shape.head_dim,
            eps,
        )?
    };
    qwen35_apply_rope_cpu(
        &mut query,
        shape.tokens,
        shape.num_heads,
        shape.head_dim,
        shape.rope_dim,
        shape.position_offset,
        shape.rope_theta,
        shape.rope_interleaved,
    )?;
    qwen35_apply_rope_cpu(
        &mut key,
        shape.tokens,
        shape.num_kv_heads,
        shape.head_dim,
        shape.rope_dim,
        shape.position_offset,
        shape.rope_theta,
        shape.rope_interleaved,
    )?;

    let repeat = shape.num_heads / shape.num_kv_heads;
    let scale = (shape.head_dim as f32).sqrt().recip();
    let mut context_ungated = vec![0.0; shape.q_len()];
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
                context_ungated[full_q_idx(shape, token, query_head, dim)] = acc;
            }
        }
    }
    let mut context = context_ungated.clone();
    if let Some(gate) = &attention_gate {
        for (value, gate) in context.iter_mut().zip(gate) {
            *value *= sigmoid(*gate);
        }
    }

    Ok(Qwen35FullAttentionReference {
        query,
        key,
        value: value_raw.to_vec(),
        attention_gate,
        context_ungated,
        context,
    })
}

pub fn qwen35_build_rope_cache_backend<B: Backend>(
    max_seq_len: usize,
    rope_dim: usize,
    rope_theta: f32,
) -> Result<Qwen35BackendRopeCache<B>> {
    if max_seq_len == 0 || rope_dim == 0 || rope_theta <= 0.0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 RoPE cache shape must be positive, got max_seq_len={max_seq_len} \
             rope_dim={rope_dim} rope_theta={rope_theta}"
        )));
    }
    if rope_dim % 2 != 0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 RoPE cache rope_dim {rope_dim} must be even"
        )));
    }

    let half = rope_dim / 2;
    let mut cos = vec![0.0; max_seq_len * half];
    let mut sin = vec![0.0; max_seq_len * half];
    for position in 0..max_seq_len {
        for pair in 0..half {
            let inv_freq = rope_theta.powf(-(2.0 * pair as f32) / rope_dim as f32);
            let angle = position as f32 * inv_freq;
            let (sin_value, cos_value) = angle.sin_cos();
            cos[position * half + pair] = cos_value;
            sin[position * half + pair] = sin_value;
        }
    }

    Ok(Qwen35BackendRopeCache {
        cos: B::from_slice(&cos),
        sin: B::from_slice(&sin),
    })
}

/// Backend-native Qwen3.5 full-attention core.
///
/// `q_norm_weight` and `k_norm_weight` are expected to be already folded with
/// the Gemma/Qwen `1 + weight` convention, matching `Qwen35ModelWeights::load`.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_full_attention_core_backend<B: Backend>(
    ctx: &mut B::Context,
    query_raw: &B::Buffer,
    key_raw: &B::Buffer,
    value_raw: &B::Buffer,
    q_norm_weight: &B::Buffer,
    k_norm_weight: &B::Buffer,
    rope_cos: &B::Buffer,
    rope_sin: &B::Buffer,
    shape: Qwen35FullAttentionShape,
    eps: f32,
) -> Result<Qwen35BackendFullAttentionOutput<B>> {
    shape.validate()?;
    let q_len = shape.q_len();
    let kv_len = shape.kv_len();
    let mut query_head_major = B::alloc(q_len);
    let mut key_head_major = B::alloc(kv_len);
    let mut value_head_major = B::alloc(kv_len);
    let mut context_head_major = B::alloc(q_len);
    let mut context = B::alloc(q_len);

    B::qk_norm_rope_partial(
        ctx,
        query_raw,
        q_norm_weight,
        rope_cos,
        rope_sin,
        &mut query_head_major,
        shape.tokens,
        shape.num_heads,
        shape.head_dim,
        shape.rope_dim,
        shape.q_proj_total(),
        0,
        if shape.attn_output_gate {
            2 * shape.head_dim
        } else {
            shape.head_dim
        },
        shape.position_offset,
        eps,
        if shape.rope_interleaved { 3 } else { 1 },
    )?;
    B::qk_norm_rope_partial(
        ctx,
        key_raw,
        k_norm_weight,
        rope_cos,
        rope_sin,
        &mut key_head_major,
        shape.tokens,
        shape.num_kv_heads,
        shape.head_dim,
        shape.rope_dim,
        shape.kv_total(),
        0,
        shape.head_dim,
        shape.position_offset,
        eps,
        if shape.rope_interleaved { 3 } else { 1 },
    )?;
    B::qk_norm_rope_partial(
        ctx,
        value_raw,
        q_norm_weight,
        rope_cos,
        rope_sin,
        &mut value_head_major,
        shape.tokens,
        shape.num_kv_heads,
        shape.head_dim,
        shape.head_dim,
        shape.kv_total(),
        0,
        shape.head_dim,
        0,
        eps,
        0,
    )?;

    let attn_cfg = AttnConfig {
        num_heads: shape.num_heads,
        num_kv_heads: shape.num_kv_heads,
        head_dim: shape.head_dim,
        causal: true,
        scale: (shape.head_dim as f32).sqrt().recip(),
        kv_seq_stride: 0,
        sliding_window: 0,
    };
    B::flash_attention(
        ctx,
        &query_head_major,
        &key_head_major,
        &value_head_major,
        &mut context_head_major,
        1,
        shape.tokens,
        shape.tokens,
        0,
        &attn_cfg,
    );
    B::transpose_head_to_token(
        ctx,
        &context_head_major,
        &mut context,
        shape.tokens,
        shape.num_heads,
        shape.head_dim,
    );
    if shape.attn_output_gate {
        B::qwen35_apply_attention_gate(
            ctx,
            &mut context,
            query_raw,
            shape.tokens,
            shape.q_total(),
            shape.q_proj_total(),
            shape.head_dim,
        )?;
    }

    Ok(Qwen35BackendFullAttentionOutput {
        query_head_major,
        key_head_major,
        value_head_major,
        context_head_major,
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

pub fn qwen35_dense_mlp_backend<B: Backend>(
    ctx: &mut B::Context,
    x: &B::Buffer,
    gate_up_proj: &dyn Linear<B>,
    down_proj: &dyn Linear<B>,
    tokens: usize,
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<Qwen35BackendDenseMlpOutput<B>> {
    if tokens == 0 || hidden_size == 0 || intermediate_size == 0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 backend dense MLP shape must be positive, got tokens={tokens} \
             hidden_size={hidden_size} intermediate_size={intermediate_size}"
        )));
    }
    if gate_up_proj.in_features() != hidden_size
        || gate_up_proj.out_features() != 2 * intermediate_size
    {
        return Err(FerrumError::model(format!(
            "Qwen3.5 backend dense MLP gate_up projection shape {}x{} does not match \
             hidden={hidden_size} intermediate={intermediate_size}",
            gate_up_proj.out_features(),
            gate_up_proj.in_features()
        )));
    }
    if down_proj.in_features() != intermediate_size || down_proj.out_features() != hidden_size {
        return Err(FerrumError::model(format!(
            "Qwen3.5 backend dense MLP down projection shape {}x{} does not match hidden={} \
             intermediate={}",
            down_proj.out_features(),
            down_proj.in_features(),
            hidden_size,
            intermediate_size
        )));
    }

    let mut gate_up = B::alloc(tokens * 2 * intermediate_size);
    let mut fused = B::alloc(tokens * intermediate_size);
    let mut output = B::alloc(tokens * hidden_size);
    gate_up_proj.forward(ctx, x, &mut gate_up, tokens);
    B::fused_silu_mul_split(ctx, &gate_up, &mut fused, tokens, intermediate_size);
    down_proj.forward(ctx, &fused, &mut output, tokens);
    Ok(Qwen35BackendDenseMlpOutput {
        gate_up,
        fused,
        output,
    })
}

pub fn qwen35_sparse_moe_shared_expert_backend<B: MoeLlmBackend>(
    ctx: &mut B::Context,
    x: &B::Buffer,
    moe: &Qwen35SparseMoeSharedExpertWeights<B>,
    shape: Qwen35SparseMoeShape,
    layer_index: usize,
) -> Result<Qwen35BackendSparseMoeOutput<B>> {
    shape.validate()?;
    if moe.router.in_features() != shape.hidden_size
        || moe.router.out_features() != shape.num_experts
    {
        return Err(FerrumError::model(format!(
            "Qwen3.5 sparse MoE router shape {}x{} does not match hidden={} experts={}",
            moe.router.out_features(),
            moe.router.in_features(),
            shape.hidden_size,
            shape.num_experts
        )));
    }
    if moe.shared_expert_gate_proj.in_features() != shape.hidden_size
        || moe.shared_expert_gate_proj.out_features() != shape.shared_expert_intermediate_size
        || moe.shared_expert_up_proj.in_features() != shape.hidden_size
        || moe.shared_expert_up_proj.out_features() != shape.shared_expert_intermediate_size
        || moe.shared_expert_down_proj.in_features() != shape.shared_expert_intermediate_size
        || moe.shared_expert_down_proj.out_features() != shape.hidden_size
    {
        return Err(FerrumError::model(
            "Qwen3.5 sparse MoE shared expert projection shapes do not match config",
        ));
    }

    let detail_enabled = qwen35_layer_detail_profile_enabled();
    let total_timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    let mut detail = Qwen35SparseMoeDetailProfile::default();

    let mut router_logits = B::alloc(shape.tokens * shape.num_experts);
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    moe.router.forward(ctx, x, &mut router_logits, shape.tokens);
    detail.router_us += qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_router");
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.router_logits",
        &router_logits,
        shape.tokens * shape.num_experts,
    );

    let mut routed_output = B::alloc(shape.tokens * shape.hidden_size);
    B::zero_buffer(ctx, &mut routed_output, shape.tokens * shape.hidden_size)?;
    let total_pairs = shape.tokens * shape.top_k;
    let mut x_packed = B::alloc(total_pairs * shape.hidden_size);
    let mut gate_up_packed = B::alloc(total_pairs * 2 * shape.expert_intermediate_size);
    let mut silu_packed = B::alloc(total_pairs * shape.expert_intermediate_size);
    let mut down_packed = B::alloc(total_pairs * shape.hidden_size);
    let mut route_scratch = MoeRouteScratch::new();
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    moe_forward_bucketed(MoeForwardBucketedParams {
        ctx,
        x,
        router_logits: &router_logits,
        out: &mut routed_output,
        batch: shape.tokens,
        hidden_size: shape.hidden_size,
        expert_intermediate: shape.expert_intermediate_size,
        num_experts: shape.num_experts,
        top_k: shape.top_k,
        norm_topk_prob: shape.norm_topk_prob,
        experts: &moe.experts,
        x_packed: &mut x_packed,
        gate_up_packed: &mut gate_up_packed,
        silu_packed: &mut silu_packed,
        down_packed: &mut down_packed,
        route_scratch: &mut route_scratch,
        device_route: None,
    })?;
    detail.routed_experts_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_routed_experts");
    qwen35_trace_layer_moe_route(
        layer_index,
        shape.tokens,
        shape.num_experts,
        shape.top_k,
        &route_scratch.output.expert_ids,
        &route_scratch.output.expert_weights,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.x_packed",
        &x_packed,
        total_pairs * shape.hidden_size,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.gate_up_packed",
        &gate_up_packed,
        total_pairs * 2 * shape.expert_intermediate_size,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.silu_packed",
        &silu_packed,
        total_pairs * shape.expert_intermediate_size,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.down_packed",
        &down_packed,
        total_pairs * shape.hidden_size,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.routed_output",
        &routed_output,
        shape.tokens * shape.hidden_size,
    );

    let mut shared_gate = B::alloc(shape.tokens);
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    B::gemm(
        ctx,
        x,
        &moe.shared_expert_gate,
        &mut shared_gate,
        shape.tokens,
        1,
        shape.hidden_size,
    );
    detail.shared_gate_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_shared_gate");
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.shared_gate",
        &shared_gate,
        shape.tokens,
    );
    let shared_inter = shape.shared_expert_intermediate_size;
    let mut shared_gate_up = B::alloc(shape.tokens * 2 * shared_inter);
    let mut shared_gate_proj = B::alloc(shape.tokens * shared_inter);
    let mut shared_up_proj = B::alloc(shape.tokens * shared_inter);
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    moe.shared_expert_gate_proj
        .forward(ctx, x, &mut shared_gate_proj, shape.tokens);
    detail.shared_gate_proj_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_shared_gate_proj");
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    moe.shared_expert_up_proj
        .forward(ctx, x, &mut shared_up_proj, shape.tokens);
    detail.shared_up_proj_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_shared_up_proj");
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.shared_gate_proj",
        &shared_gate_proj,
        shape.tokens * shared_inter,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.shared_up_proj",
        &shared_up_proj,
        shape.tokens * shared_inter,
    );
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    B::qwen35_interleave_gate_up(
        ctx,
        &shared_gate_proj,
        &shared_up_proj,
        &mut shared_gate_up,
        shape.tokens,
        shared_inter,
    )?;
    detail.shared_pack_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_shared_pack");
    let mut shared_fused = B::alloc(shape.tokens * shared_inter);
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    B::fused_silu_mul_split(
        ctx,
        &shared_gate_up,
        &mut shared_fused,
        shape.tokens,
        shared_inter,
    );
    detail.shared_fused_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_shared_fused");
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.shared_fused",
        &shared_fused,
        shape.tokens * shared_inter,
    );
    let mut shared_output = B::alloc(shape.tokens * shape.hidden_size);
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    moe.shared_expert_down_proj
        .forward(ctx, &shared_fused, &mut shared_output, shape.tokens);
    detail.shared_down_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_shared_down");
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.shared_down_output",
        &shared_output,
        shape.tokens * shape.hidden_size,
    );
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    B::qwen35_apply_token_gate(
        ctx,
        &mut shared_output,
        &shared_gate,
        shape.tokens,
        shape.hidden_size,
    )?;
    detail.shared_apply_gate_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_shared_apply_gate");
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.shared_output",
        &shared_output,
        shape.tokens * shape.hidden_size,
    );

    let mut output = B::alloc(shape.tokens * shape.hidden_size);
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    B::copy_slice(
        ctx,
        &routed_output,
        0,
        &mut output,
        0,
        shape.tokens * shape.hidden_size,
    );
    B::add_inplace(
        ctx,
        &mut output,
        &shared_output,
        shape.tokens * shape.hidden_size,
    );
    detail.merge_us += qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_merge");
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.output",
        &output,
        shape.tokens * shape.hidden_size,
    );

    detail.total_us = qwen35_detail_profile_stage_finish::<B>(ctx, total_timer, "qwen35_moe_total");
    if detail_enabled {
        detail.log(layer_index, shape.tokens, shape.top_k, shape.num_experts);
    }

    Ok(Qwen35BackendSparseMoeOutput {
        router_logits,
        routed_output,
        shared_gate,
        shared_gate_up,
        shared_fused,
        shared_output,
        output,
    })
}

fn qwen35_sparse_moe_shared_expert_decode_scratch<B: MoeLlmBackend>(
    ctx: &mut B::Context,
    moe: &Qwen35SparseMoeSharedExpertWeights<B>,
    shape: Qwen35SparseMoeShape,
    layer_index: usize,
    scratch: &mut Qwen35DecodeScratch<B>,
) -> Result<()> {
    shape.validate()?;
    if shape.tokens > scratch.max_tokens
        || shape.hidden_size != scratch.hidden_size
        || shape.num_experts != scratch.num_experts
        || shape.top_k != scratch.top_k
        || shape.expert_intermediate_size != scratch.expert_intermediate_size
        || shape.shared_expert_intermediate_size != scratch.shared_expert_intermediate_size
    {
        return Err(FerrumError::model(format!(
            "Qwen3.5 decode scratch shape mismatch: scratch tokens={} hidden={} experts={} top_k={} expert_inter={} shared_inter={} vs request tokens={} hidden={} experts={} top_k={} expert_inter={} shared_inter={}",
            scratch.max_tokens,
            scratch.hidden_size,
            scratch.num_experts,
            scratch.top_k,
            scratch.expert_intermediate_size,
            scratch.shared_expert_intermediate_size,
            shape.tokens,
            shape.hidden_size,
            shape.num_experts,
            shape.top_k,
            shape.expert_intermediate_size,
            shape.shared_expert_intermediate_size,
        )));
    }
    if moe.router.in_features() != shape.hidden_size
        || moe.router.out_features() != shape.num_experts
    {
        return Err(FerrumError::model(format!(
            "Qwen3.5 sparse MoE router shape {}x{} does not match hidden={} experts={}",
            moe.router.out_features(),
            moe.router.in_features(),
            shape.hidden_size,
            shape.num_experts
        )));
    }
    if moe.shared_expert_gate_proj.in_features() != shape.hidden_size
        || moe.shared_expert_gate_proj.out_features() != shape.shared_expert_intermediate_size
        || moe.shared_expert_up_proj.in_features() != shape.hidden_size
        || moe.shared_expert_up_proj.out_features() != shape.shared_expert_intermediate_size
        || moe.shared_expert_down_proj.in_features() != shape.shared_expert_intermediate_size
        || moe.shared_expert_down_proj.out_features() != shape.hidden_size
    {
        return Err(FerrumError::model(
            "Qwen3.5 sparse MoE shared expert projection shapes do not match config",
        ));
    }

    let x = &scratch.post_attention_norm;
    let detail_enabled = qwen35_layer_detail_profile_enabled();
    let total_timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    let mut detail = Qwen35SparseMoeDetailProfile::default();

    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    moe.router
        .forward(ctx, x, &mut scratch.router_logits, shape.tokens);
    detail.router_us += qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_router");
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.router_logits",
        &scratch.router_logits,
        shape.tokens * shape.num_experts,
    );

    B::zero_buffer(
        ctx,
        &mut scratch.routed_output,
        shape.tokens * shape.hidden_size,
    )?;
    let total_pairs = shape.tokens * shape.top_k;
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    moe_forward_bucketed(MoeForwardBucketedParams {
        ctx,
        x,
        router_logits: &scratch.router_logits,
        out: &mut scratch.routed_output,
        batch: shape.tokens,
        hidden_size: shape.hidden_size,
        expert_intermediate: shape.expert_intermediate_size,
        num_experts: shape.num_experts,
        top_k: shape.top_k,
        norm_topk_prob: shape.norm_topk_prob,
        experts: &moe.experts,
        x_packed: &mut scratch.x_packed,
        gate_up_packed: &mut scratch.gate_up_packed,
        silu_packed: &mut scratch.silu_packed,
        down_packed: &mut scratch.down_packed,
        route_scratch: &mut scratch.route_scratch,
        device_route: Some(crate::moe::dispatch::DeviceRouteScratch {
            selected_ids: &mut scratch.route_selected_ids,
            pair_weights: &mut scratch.route_pair_weights,
            pairs_by_token: &mut scratch.route_pairs_by_token,
            packed_token_idx: &mut scratch.route_packed_token_idx,
            expert_offsets: &mut scratch.route_expert_offsets,
            sorted_tokens: &mut scratch.route_sorted_tokens,
            block_ids: &mut scratch.route_block_ids,
            total_post_pad: &mut scratch.route_total_post_pad,
        }),
    })?;
    detail.routed_experts_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_routed_experts");
    qwen35_trace_layer_moe_route(
        layer_index,
        shape.tokens,
        shape.num_experts,
        shape.top_k,
        &scratch.route_scratch.output.expert_ids,
        &scratch.route_scratch.output.expert_weights,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.x_packed",
        &scratch.x_packed,
        total_pairs * shape.hidden_size,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.gate_up_packed",
        &scratch.gate_up_packed,
        total_pairs * 2 * shape.expert_intermediate_size,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.silu_packed",
        &scratch.silu_packed,
        total_pairs * shape.expert_intermediate_size,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.down_packed",
        &scratch.down_packed,
        total_pairs * shape.hidden_size,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.routed_output",
        &scratch.routed_output,
        shape.tokens * shape.hidden_size,
    );

    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    B::gemm(
        ctx,
        x,
        &moe.shared_expert_gate,
        &mut scratch.shared_gate,
        shape.tokens,
        1,
        shape.hidden_size,
    );
    detail.shared_gate_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_shared_gate");
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.shared_gate",
        &scratch.shared_gate,
        shape.tokens,
    );

    let shared_inter = shape.shared_expert_intermediate_size;
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    moe.shared_expert_gate_proj
        .forward(ctx, x, &mut scratch.shared_gate_proj, shape.tokens);
    detail.shared_gate_proj_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_shared_gate_proj");
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    moe.shared_expert_up_proj
        .forward(ctx, x, &mut scratch.shared_up_proj, shape.tokens);
    detail.shared_up_proj_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_shared_up_proj");
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.shared_gate_proj",
        &scratch.shared_gate_proj,
        shape.tokens * shared_inter,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.shared_up_proj",
        &scratch.shared_up_proj,
        shape.tokens * shared_inter,
    );

    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    B::qwen35_interleave_gate_up(
        ctx,
        &scratch.shared_gate_proj,
        &scratch.shared_up_proj,
        &mut scratch.shared_gate_up,
        shape.tokens,
        shared_inter,
    )?;
    detail.shared_pack_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_shared_pack");
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    B::fused_silu_mul_split(
        ctx,
        &scratch.shared_gate_up,
        &mut scratch.shared_fused,
        shape.tokens,
        shared_inter,
    );
    detail.shared_fused_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_shared_fused");
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.shared_fused",
        &scratch.shared_fused,
        shape.tokens * shared_inter,
    );

    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    moe.shared_expert_down_proj.forward(
        ctx,
        &scratch.shared_fused,
        &mut scratch.shared_output,
        shape.tokens,
    );
    detail.shared_down_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_shared_down");
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.shared_down_output",
        &scratch.shared_output,
        shape.tokens * shape.hidden_size,
    );
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    B::qwen35_apply_token_gate(
        ctx,
        &mut scratch.shared_output,
        &scratch.shared_gate,
        shape.tokens,
        shape.hidden_size,
    )?;
    detail.shared_apply_gate_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_shared_apply_gate");
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.shared_output",
        &scratch.shared_output,
        shape.tokens * shape.hidden_size,
    );

    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    B::copy_slice(
        ctx,
        &scratch.routed_output,
        0,
        &mut scratch.mlp_output,
        0,
        shape.tokens * shape.hidden_size,
    );
    B::add_inplace(
        ctx,
        &mut scratch.mlp_output,
        &scratch.shared_output,
        shape.tokens * shape.hidden_size,
    );
    detail.merge_us += qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_moe_merge");
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer_index,
        "moe.output",
        &scratch.mlp_output,
        shape.tokens * shape.hidden_size,
    );

    detail.total_us = qwen35_detail_profile_stage_finish::<B>(ctx, total_timer, "qwen35_moe_total");
    if detail_enabled {
        detail.log(layer_index, shape.tokens, shape.top_k, shape.num_experts);
    }

    Ok(())
}

fn qwen35_mlp_backend<B: MoeLlmBackend>(
    ctx: &mut B::Context,
    x: &B::Buffer,
    mlp: &Qwen35MlpWeights<B>,
    config: &Qwen35TextConfig,
    tokens: usize,
    layer_index: usize,
) -> Result<B::Buffer> {
    match mlp {
        Qwen35MlpWeights::Dense(mlp) => {
            let intermediate = config.dense_intermediate_size.ok_or_else(|| {
                FerrumError::model("Qwen3.5 dense MLP config missing intermediate_size")
            })?;
            Ok(qwen35_dense_mlp_backend::<B>(
                ctx,
                x,
                &*mlp.gate_up_proj,
                &*mlp.down_proj,
                tokens,
                config.hidden_size,
                intermediate,
            )?
            .output)
        }
        Qwen35MlpWeights::SparseMoeSharedExpert(moe) => {
            let moe_config = config
                .moe
                .as_ref()
                .ok_or_else(|| FerrumError::model("Qwen3.5 sparse MoE config missing"))?;
            Ok(qwen35_sparse_moe_shared_expert_backend::<B>(
                ctx,
                x,
                moe,
                Qwen35SparseMoeShape {
                    tokens,
                    hidden_size: config.hidden_size,
                    num_experts: moe_config.num_experts,
                    top_k: moe_config.num_experts_per_tok,
                    expert_intermediate_size: moe_config.moe_intermediate_size,
                    shared_expert_intermediate_size: moe_config.shared_expert_intermediate_size,
                    norm_topk_prob: moe_config.norm_topk_prob,
                },
                layer_index,
            )?
            .output)
        }
    }
}

fn qwen35_finish_layer_with_mlp<B: MoeLlmBackend>(
    ctx: &mut B::Context,
    residual_after_attention: &B::Buffer,
    layer: &Qwen35LayerWeights<B>,
    config: &Qwen35TextConfig,
    tokens: usize,
    eps: f32,
) -> Result<B::Buffer> {
    let hidden_len = tokens * config.hidden_size;
    let mut post_attention_norm = B::alloc(hidden_len);
    B::rms_norm(
        ctx,
        residual_after_attention,
        &layer.post_attention_layernorm,
        eps,
        &mut post_attention_norm,
        tokens,
        config.hidden_size,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "post_attention_norm",
        &post_attention_norm,
        hidden_len,
    );
    let mlp_out = qwen35_mlp_backend::<B>(
        ctx,
        &post_attention_norm,
        &layer.mlp,
        config,
        tokens,
        layer.layer_index,
    )?;
    qwen35_trace_layer_buffer_stats::<B>(ctx, layer.layer_index, "mlp_out", &mlp_out, hidden_len);
    let mut layer_output = B::alloc(hidden_len);
    B::copy_slice(
        ctx,
        residual_after_attention,
        0,
        &mut layer_output,
        0,
        hidden_len,
    );
    B::add_inplace(ctx, &mut layer_output, &mlp_out, hidden_len);
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "layer_output_after_mlp",
        &layer_output,
        hidden_len,
    );
    Ok(layer_output)
}

fn qwen35_finish_layer_with_mlp_f32_residual<B: MoeLlmBackend>(
    ctx: &mut B::Context,
    residual_f32: &mut B::Buffer,
    branch_f32: &mut B::Buffer,
    layer: &Qwen35LayerWeights<B>,
    config: &Qwen35TextConfig,
    tokens: usize,
    eps: f32,
) -> Result<B::Buffer> {
    let hidden_len = tokens * config.hidden_size;
    let mut post_attention_norm = B::alloc(hidden_len);
    B::rms_norm_f32_to_activation(
        ctx,
        residual_f32,
        &layer.post_attention_layernorm,
        eps,
        &mut post_attention_norm,
        tokens,
        config.hidden_size,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "post_attention_norm",
        &post_attention_norm,
        hidden_len,
    );
    let mlp_out = qwen35_mlp_backend::<B>(
        ctx,
        &post_attention_norm,
        &layer.mlp,
        config,
        tokens,
        layer.layer_index,
    )?;
    qwen35_trace_layer_buffer_stats::<B>(ctx, layer.layer_index, "mlp_out", &mlp_out, hidden_len);
    B::activation_to_f32_shadow(ctx, &mlp_out, branch_f32, hidden_len);
    B::add_inplace(ctx, residual_f32, branch_f32, hidden_len);
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "layer_output_after_mlp",
        residual_f32,
        hidden_len,
    );
    let mut layer_output = B::alloc(hidden_len);
    B::f32_to_activation(ctx, residual_f32, &mut layer_output, hidden_len);
    Ok(layer_output)
}

fn qwen35_sparse_moe_shape_from_config(
    config: &Qwen35TextConfig,
    tokens: usize,
) -> Result<Qwen35SparseMoeShape> {
    let moe_config = config
        .moe
        .as_ref()
        .ok_or_else(|| FerrumError::model("Qwen3.5 sparse MoE config missing"))?;
    Ok(Qwen35SparseMoeShape {
        tokens,
        hidden_size: config.hidden_size,
        num_experts: moe_config.num_experts,
        top_k: moe_config.num_experts_per_tok,
        expert_intermediate_size: moe_config.moe_intermediate_size,
        shared_expert_intermediate_size: moe_config.shared_expert_intermediate_size,
        norm_topk_prob: moe_config.norm_topk_prob,
    })
}

fn qwen35_finish_layer_with_mlp_decode_scratch<B: MoeLlmBackend>(
    ctx: &mut B::Context,
    residual_after_attention: &B::Buffer,
    layer: &Qwen35LayerWeights<B>,
    config: &Qwen35TextConfig,
    tokens: usize,
    eps: f32,
    scratch: &mut Qwen35DecodeScratch<B>,
) -> Result<B::Buffer> {
    let hidden_len = tokens * config.hidden_size;
    B::rms_norm(
        ctx,
        residual_after_attention,
        &layer.post_attention_layernorm,
        eps,
        &mut scratch.post_attention_norm,
        tokens,
        config.hidden_size,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "post_attention_norm",
        &scratch.post_attention_norm,
        hidden_len,
    );
    match &layer.mlp {
        Qwen35MlpWeights::SparseMoeSharedExpert(moe) => {
            qwen35_sparse_moe_shared_expert_decode_scratch::<B>(
                ctx,
                moe,
                qwen35_sparse_moe_shape_from_config(config, tokens)?,
                layer.layer_index,
                scratch,
            )?;
            qwen35_trace_layer_buffer_stats::<B>(
                ctx,
                layer.layer_index,
                "mlp_out",
                &scratch.mlp_output,
                hidden_len,
            );
            let mut layer_output = B::alloc(hidden_len);
            B::copy_slice(
                ctx,
                residual_after_attention,
                0,
                &mut layer_output,
                0,
                hidden_len,
            );
            B::add_inplace(ctx, &mut layer_output, &scratch.mlp_output, hidden_len);
            qwen35_trace_layer_buffer_stats::<B>(
                ctx,
                layer.layer_index,
                "layer_output_after_mlp",
                &layer_output,
                hidden_len,
            );
            Ok(layer_output)
        }
        Qwen35MlpWeights::Dense(_) => {
            let mlp_out = qwen35_mlp_backend::<B>(
                ctx,
                &scratch.post_attention_norm,
                &layer.mlp,
                config,
                tokens,
                layer.layer_index,
            )?;
            qwen35_trace_layer_buffer_stats::<B>(
                ctx,
                layer.layer_index,
                "mlp_out",
                &mlp_out,
                hidden_len,
            );
            let mut layer_output = B::alloc(hidden_len);
            B::copy_slice(
                ctx,
                residual_after_attention,
                0,
                &mut layer_output,
                0,
                hidden_len,
            );
            B::add_inplace(ctx, &mut layer_output, &mlp_out, hidden_len);
            qwen35_trace_layer_buffer_stats::<B>(
                ctx,
                layer.layer_index,
                "layer_output_after_mlp",
                &layer_output,
                hidden_len,
            );
            Ok(layer_output)
        }
    }
}

fn qwen35_finish_layer_with_mlp_f32_residual_decode_scratch<B: MoeLlmBackend>(
    ctx: &mut B::Context,
    residual_f32: &mut B::Buffer,
    branch_f32: &mut B::Buffer,
    layer: &Qwen35LayerWeights<B>,
    config: &Qwen35TextConfig,
    tokens: usize,
    eps: f32,
    scratch: &mut Qwen35DecodeScratch<B>,
) -> Result<()> {
    let hidden_len = tokens * config.hidden_size;
    B::rms_norm_f32_to_activation(
        ctx,
        residual_f32,
        &layer.post_attention_layernorm,
        eps,
        &mut scratch.post_attention_norm,
        tokens,
        config.hidden_size,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "post_attention_norm",
        &scratch.post_attention_norm,
        hidden_len,
    );
    match &layer.mlp {
        Qwen35MlpWeights::SparseMoeSharedExpert(moe) => {
            qwen35_sparse_moe_shared_expert_decode_scratch::<B>(
                ctx,
                moe,
                qwen35_sparse_moe_shape_from_config(config, tokens)?,
                layer.layer_index,
                scratch,
            )?;
            qwen35_trace_layer_buffer_stats::<B>(
                ctx,
                layer.layer_index,
                "mlp_out",
                &scratch.mlp_output,
                hidden_len,
            );
            B::activation_to_f32_shadow(ctx, &scratch.mlp_output, branch_f32, hidden_len);
        }
        Qwen35MlpWeights::Dense(_) => {
            let mlp_out = qwen35_mlp_backend::<B>(
                ctx,
                &scratch.post_attention_norm,
                &layer.mlp,
                config,
                tokens,
                layer.layer_index,
            )?;
            qwen35_trace_layer_buffer_stats::<B>(
                ctx,
                layer.layer_index,
                "mlp_out",
                &mlp_out,
                hidden_len,
            );
            B::activation_to_f32_shadow(ctx, &mlp_out, branch_f32, hidden_len);
        }
    }
    B::add_inplace(ctx, residual_f32, branch_f32, hidden_len);
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "layer_output_after_mlp",
        residual_f32,
        hidden_len,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn qwen35_linear_attention_decode_batch_layer_backend<B: MoeLlmBackend>(
    ctx: &mut B::Context,
    layer_input: &B::Buffer,
    states: &mut [(String, Qwen35SequenceState<B>)],
    mut linear_state_pools: Option<&mut Qwen35LinearStatePools<B>>,
    linear_slot_indices: Option<&B::Buffer>,
    layer_index: usize,
    layer: &Qwen35LayerWeights<B>,
    config: &Qwen35TextConfig,
    eps: f32,
    mut residual_f32: Option<&mut B::Buffer>,
    mut branch_f32: Option<&mut B::Buffer>,
    mut decode_scratch: Option<&mut Qwen35DecodeScratch<B>>,
) -> Result<Option<B::Buffer>> {
    let batch_len = states.len();
    let attention_shape = Qwen35LinearAttentionShape {
        tokens: 1,
        key_heads: config.linear_attention.num_key_heads,
        value_heads: config.linear_attention.num_value_heads,
        key_dim: config.linear_attention.key_head_dim,
        value_dim: config.linear_attention.value_head_dim,
        conv_kernel: config.linear_attention.conv_kernel_dim,
    };
    let hidden_len = batch_len * config.hidden_size;
    let qkv_width = attention_shape.conv_channels();
    let value_total = attention_shape.value_total();
    let gating_width = attention_shape.value_heads;
    let attention = match &layer.attention {
        Qwen35AttentionWeights::Linear(attention) => attention,
        Qwen35AttentionWeights::Full(_) => {
            return Err(FerrumError::model(format!(
                "Qwen3.5 batch decode expected linear attention at layer {}",
                layer.layer_index
            )));
        }
    };

    let detail_enabled = qwen35_layer_detail_profile_enabled();
    let mut detail = Qwen35LinearDecodeDetailProfile::default();

    let mut input_norm = B::alloc(hidden_len);
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    if let Some(residual_f32) = residual_f32.as_ref() {
        B::rms_norm_f32_to_activation(
            ctx,
            residual_f32,
            &layer.input_layernorm,
            eps,
            &mut input_norm,
            batch_len,
            config.hidden_size,
        );
    } else {
        B::rms_norm(
            ctx,
            layer_input,
            &layer.input_layernorm,
            eps,
            &mut input_norm,
            batch_len,
            config.hidden_size,
        );
    }
    detail.input_norm_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_linear_decode_input_norm");

    let can_use_packed_decode_prepare = B::supports_qwen35_packed_gdn_decode_prepare()
        && B::supports_qwen35_indexed_recurrent_state()
        && linear_state_pools.is_some()
        && linear_slot_indices.is_some()
        && attention.qkvz_proj.is_some()
        && attention.ba_proj.is_some();
    let mut projection_buffers = if can_use_packed_decode_prepare {
        let qkvz_width = qkv_width + value_total;
        let mut mixed_qkvz_raw = B::alloc(batch_len * qkvz_width);
        let mut ba_raw = B::alloc(batch_len * 2 * gating_width);
        let z_raw = B::alloc_typed(Dtype::F32, batch_len * value_total);
        let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
        attention
            .qkvz_proj
            .as_ref()
            .expect("checked above")
            .forward(ctx, &input_norm, &mut mixed_qkvz_raw, batch_len);
        detail.qkvz_proj_us +=
            qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_linear_decode_qkvz_proj");
        let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
        attention.ba_proj.as_ref().expect("checked above").forward(
            ctx,
            &input_norm,
            &mut ba_raw,
            batch_len,
        );
        detail.ba_proj_us +=
            qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_linear_decode_ba_proj");
        Qwen35LinearDecodeProjectionBuffers::Packed {
            mixed_qkvz_raw,
            ba_raw,
            z_raw,
        }
    } else {
        let mut mixed_qkv_raw = B::alloc(batch_len * qkv_width);
        let mut z_raw = B::alloc(batch_len * value_total);
        let mut b_raw = B::alloc(batch_len * gating_width);
        let mut a_raw = B::alloc(batch_len * gating_width);
        let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
        attention
            .qkv_proj
            .forward(ctx, &input_norm, &mut mixed_qkv_raw, batch_len);
        detail.qkv_proj_us +=
            qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_linear_decode_qkv_proj");
        let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
        attention
            .z_proj
            .forward(ctx, &input_norm, &mut z_raw, batch_len);
        detail.z_proj_us +=
            qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_linear_decode_z_proj");
        let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
        attention
            .b_proj
            .forward(ctx, &input_norm, &mut b_raw, batch_len);
        detail.b_proj_us +=
            qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_linear_decode_b_proj");
        let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
        attention
            .a_proj
            .forward(ctx, &input_norm, &mut a_raw, batch_len);
        detail.a_proj_us +=
            qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_linear_decode_a_proj");
        Qwen35LinearDecodeProjectionBuffers::Separate {
            mixed_qkv_raw,
            z_raw,
            b_raw,
            a_raw,
        }
    };

    let mut query = B::alloc_typed(Dtype::F32, batch_len * attention_shape.qk_total());
    let mut key = B::alloc_typed(Dtype::F32, batch_len * attention_shape.qk_total());
    let mut value = B::alloc_typed(Dtype::F32, batch_len * value_total);
    let mut g = B::alloc_typed(Dtype::F32, batch_len * gating_width);
    let mut beta = B::alloc_typed(Dtype::F32, batch_len * gating_width);
    let mut delta_core = B::alloc_typed(Dtype::F32, batch_len * value_total);

    let used_indexed = if B::supports_qwen35_indexed_recurrent_state() {
        if let (Some(pools), Some(slot_indices)) =
            (linear_state_pools.as_mut(), linear_slot_indices)
        {
            let pools = &mut **pools;
            let conv_slots = pools.conv_states[layer_index].as_mut().ok_or_else(|| {
                FerrumError::model(format!(
                    "Qwen3.5 missing linear conv state slot pool for decode layer {layer_index}"
                ))
            })?;
            let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
            match &mut projection_buffers {
                Qwen35LinearDecodeProjectionBuffers::Packed {
                    mixed_qkvz_raw,
                    ba_raw,
                    z_raw,
                } => B::linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f32(
                    ctx,
                    mixed_qkvz_raw,
                    ba_raw,
                    &attention.conv1d_weight,
                    conv_slots,
                    slot_indices,
                    &attention.a_log,
                    &attention.dt_bias,
                    &mut query,
                    &mut key,
                    &mut value,
                    z_raw,
                    &mut g,
                    &mut beta,
                    batch_len,
                    pools.max_slots,
                    attention_shape.key_heads,
                    attention_shape.value_heads,
                    attention_shape.key_dim,
                    attention_shape.value_dim,
                    attention_shape.conv_kernel,
                    true,
                )?,
                Qwen35LinearDecodeProjectionBuffers::Separate {
                    mixed_qkv_raw,
                    b_raw,
                    a_raw,
                    ..
                } => B::linear_attention_decode_prepare_batch_indexed_f32(
                    ctx,
                    mixed_qkv_raw,
                    &attention.conv1d_weight,
                    conv_slots,
                    slot_indices,
                    a_raw,
                    b_raw,
                    &attention.a_log,
                    &attention.dt_bias,
                    &mut query,
                    &mut key,
                    &mut value,
                    &mut g,
                    &mut beta,
                    batch_len,
                    pools.max_slots,
                    attention_shape.key_heads,
                    attention_shape.value_heads,
                    attention_shape.key_dim,
                    attention_shape.value_dim,
                    attention_shape.conv_kernel,
                    true,
                )?,
            }
            detail.indexed_prepare_us += qwen35_detail_profile_stage_finish::<B>(
                ctx,
                timer,
                "qwen35_linear_decode_indexed_prepare",
            );

            let delta_slots = pools.delta_states[layer_index].as_mut().ok_or_else(|| {
                FerrumError::model(format!(
                    "Qwen3.5 missing linear delta state slot pool for decode layer {layer_index}"
                ))
            })?;
            let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
            B::recurrent_gated_delta_rule_batch_indexed_f32(
                ctx,
                &query,
                &key,
                &value,
                &g,
                &beta,
                delta_slots,
                slot_indices,
                &mut delta_core,
                batch_len,
                pools.max_slots,
                attention_shape.key_heads,
                attention_shape.value_heads,
                attention_shape.key_dim,
                attention_shape.value_dim,
                false,
                (attention_shape.key_dim as f32).sqrt().recip(),
            )?;
            detail.indexed_recurrent_us += qwen35_detail_profile_stage_finish::<B>(
                ctx,
                timer,
                "qwen35_linear_decode_indexed_recurrent",
            );
            true
        } else {
            false
        }
    } else {
        false
    };

    if !used_indexed {
        let conv_state_len = qkv_width * attention_shape.conv_kernel.saturating_sub(1);
        let delta_state_len = attention_shape.state_len();
        let mut conv_states = B::alloc_typed(Dtype::F32, batch_len * conv_state_len);
        let mut next_conv_states = B::alloc_typed(Dtype::F32, batch_len * conv_state_len);
        let mut initial_delta_states = B::alloc_typed(Dtype::F32, batch_len * delta_state_len);
        let mut final_delta_states = B::alloc_typed(Dtype::F32, batch_len * delta_state_len);
        let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
        for (row, (cache_id, state)) in states.iter_mut().enumerate() {
            let Qwen35LayerRuntimeState::Linear {
                conv_state,
                delta_state,
            } = state.layers.get_mut(layer_index).ok_or_else(|| {
                FerrumError::model(format!(
                    "Qwen3.5 missing layer {layer_index} state for {cache_id:?}"
                ))
            })?
            else {
                return Err(FerrumError::model(format!(
                    "Qwen3.5 batch decode expected linear layer state at layer {layer_index} for {cache_id:?}"
                )));
            };
            B::copy_slice(
                ctx,
                conv_state,
                0,
                &mut conv_states,
                row * conv_state_len,
                conv_state_len,
            );
            B::copy_slice(
                ctx,
                delta_state,
                0,
                &mut initial_delta_states,
                row * delta_state_len,
                delta_state_len,
            );
        }
        detail.fallback_state_gather_us += qwen35_detail_profile_stage_finish::<B>(
            ctx,
            timer,
            "qwen35_linear_decode_fallback_state_gather",
        );

        let Qwen35LinearDecodeProjectionBuffers::Separate {
            mixed_qkv_raw,
            b_raw,
            a_raw,
            ..
        } = &projection_buffers
        else {
            return Err(FerrumError::model(
                "Qwen3.5 packed linear decode projections require indexed recurrent state",
            ));
        };
        let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
        B::linear_attention_decode_prepare_batch_f32(
            ctx,
            mixed_qkv_raw,
            &attention.conv1d_weight,
            &conv_states,
            a_raw,
            b_raw,
            &attention.a_log,
            &attention.dt_bias,
            &mut query,
            &mut key,
            &mut value,
            &mut g,
            &mut beta,
            &mut next_conv_states,
            batch_len,
            attention_shape.key_heads,
            attention_shape.value_heads,
            attention_shape.key_dim,
            attention_shape.value_dim,
            attention_shape.conv_kernel,
            true,
        )?;
        detail.fallback_prepare_us += qwen35_detail_profile_stage_finish::<B>(
            ctx,
            timer,
            "qwen35_linear_decode_fallback_prepare",
        );

        let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
        B::recurrent_gated_delta_rule_batch_f32(
            ctx,
            &query,
            &key,
            &value,
            &g,
            &beta,
            &initial_delta_states,
            &mut delta_core,
            &mut final_delta_states,
            batch_len,
            attention_shape.key_heads,
            attention_shape.value_heads,
            attention_shape.key_dim,
            attention_shape.value_dim,
            false,
            (attention_shape.key_dim as f32).sqrt().recip(),
        )?;
        detail.fallback_recurrent_us += qwen35_detail_profile_stage_finish::<B>(
            ctx,
            timer,
            "qwen35_linear_decode_fallback_recurrent",
        );

        let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
        for (row, (cache_id, state)) in states.iter_mut().enumerate() {
            let Qwen35LayerRuntimeState::Linear {
                conv_state,
                delta_state,
            } = state.layers.get_mut(layer_index).ok_or_else(|| {
                FerrumError::model(format!(
                    "Qwen3.5 missing layer {layer_index} state for {cache_id:?}"
                ))
            })?
            else {
                return Err(FerrumError::model(format!(
                    "Qwen3.5 batch decode expected linear layer state at layer {layer_index} for {cache_id:?}"
                )));
            };
            B::copy_slice(
                ctx,
                &next_conv_states,
                row * conv_state_len,
                conv_state,
                0,
                conv_state_len,
            );
            B::copy_slice(
                ctx,
                &final_delta_states,
                row * delta_state_len,
                delta_state,
                0,
                delta_state_len,
            );
        }
        detail.fallback_state_scatter_us += qwen35_detail_profile_stage_finish::<B>(
            ctx,
            timer,
            "qwen35_linear_decode_fallback_state_scatter",
        );
    }

    let mut delta_norm = B::alloc_typed(Dtype::F32, batch_len * value_total);
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    let z_raw = match &projection_buffers {
        Qwen35LinearDecodeProjectionBuffers::Separate { z_raw, .. }
        | Qwen35LinearDecodeProjectionBuffers::Packed { z_raw, .. } => z_raw,
    };
    B::gated_rms_norm_f32(
        ctx,
        &delta_core,
        z_raw,
        &attention.norm_weight,
        &mut delta_norm,
        batch_len,
        attention_shape.value_heads,
        attention_shape.value_dim,
        eps,
    )?;
    detail.gated_norm_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_linear_decode_gated_norm");

    let mut delta_activation = B::alloc(batch_len * value_total);
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    B::f32_to_activation(
        ctx,
        &delta_norm,
        &mut delta_activation,
        batch_len * value_total,
    );
    detail.f32_to_activation_us += qwen35_detail_profile_stage_finish::<B>(
        ctx,
        timer,
        "qwen35_linear_decode_f32_to_activation",
    );
    let mut delta_output = B::alloc(hidden_len);
    let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
    attention
        .out_proj
        .forward(ctx, &delta_activation, &mut delta_output, batch_len);
    detail.out_proj_us +=
        qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_linear_decode_out_proj");

    let layer_output = if let (Some(residual_f32), Some(branch_f32)) =
        (residual_f32.as_deref_mut(), branch_f32.as_deref_mut())
    {
        let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
        B::activation_to_f32_shadow(ctx, &delta_output, branch_f32, hidden_len);
        B::add_inplace(ctx, residual_f32, branch_f32, hidden_len);
        detail.residual_update_us += qwen35_detail_profile_stage_finish::<B>(
            ctx,
            timer,
            "qwen35_linear_decode_residual_update",
        );
        let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
        let output = if let Some(scratch) = decode_scratch.as_deref_mut() {
            qwen35_finish_layer_with_mlp_f32_residual_decode_scratch::<B>(
                ctx,
                residual_f32,
                branch_f32,
                layer,
                config,
                batch_len,
                eps,
                scratch,
            )?;
            None
        } else {
            Some(qwen35_finish_layer_with_mlp_f32_residual::<B>(
                ctx,
                residual_f32,
                branch_f32,
                layer,
                config,
                batch_len,
                eps,
            )?)
        };
        detail.mlp_us +=
            qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_linear_decode_mlp");
        output
    } else {
        let mut residual_after_mixer = B::alloc(hidden_len);
        let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
        B::copy_slice(
            ctx,
            layer_input,
            0,
            &mut residual_after_mixer,
            0,
            hidden_len,
        );
        B::add_inplace(ctx, &mut residual_after_mixer, &delta_output, hidden_len);
        detail.residual_update_us += qwen35_detail_profile_stage_finish::<B>(
            ctx,
            timer,
            "qwen35_linear_decode_residual_update",
        );
        let timer = qwen35_detail_profile_stage_start::<B>(ctx, detail_enabled);
        let output = if let Some(scratch) = decode_scratch.as_deref_mut() {
            Some(qwen35_finish_layer_with_mlp_decode_scratch::<B>(
                ctx,
                &residual_after_mixer,
                layer,
                config,
                batch_len,
                eps,
                scratch,
            )?)
        } else {
            Some(qwen35_finish_layer_with_mlp::<B>(
                ctx,
                &residual_after_mixer,
                layer,
                config,
                batch_len,
                eps,
            )?)
        };
        detail.mlp_us +=
            qwen35_detail_profile_stage_finish::<B>(ctx, timer, "qwen35_linear_decode_mlp");
        output
    };

    if detail_enabled {
        detail.log(layer_index, batch_len, used_indexed);
    }
    Ok(layer_output)
}

#[allow(clippy::too_many_arguments)]
fn qwen35_linear_attention_prefill_batch_layer_backend<B: MoeLlmBackend>(
    ctx: &mut B::Context,
    layer_input: &B::Buffer,
    states: &mut [(String, Qwen35SequenceState<B>)],
    linear_state_pools: Option<&mut Qwen35LinearStatePools<B>>,
    layer_index: usize,
    layer: &Qwen35LayerWeights<B>,
    config: &Qwen35TextConfig,
    cu_seqlens: &B::Buffer,
    q_lens: &[usize],
    total_tokens: usize,
    eps: f32,
    mut residual_f32: Option<&mut B::Buffer>,
    mut branch_f32: Option<&mut B::Buffer>,
) -> Result<B::Buffer> {
    let batch_len = states.len();
    if q_lens.len() != batch_len {
        return Err(FerrumError::model(format!(
            "Qwen3.5 prefill batch q_lens {} != states {}",
            q_lens.len(),
            batch_len
        )));
    }
    let attention_shape = Qwen35LinearAttentionShape {
        tokens: total_tokens,
        key_heads: config.linear_attention.num_key_heads,
        value_heads: config.linear_attention.num_value_heads,
        key_dim: config.linear_attention.key_head_dim,
        value_dim: config.linear_attention.value_head_dim,
        conv_kernel: config.linear_attention.conv_kernel_dim,
    };
    let hidden_len = total_tokens * config.hidden_size;
    let qkv_width = attention_shape.conv_channels();
    let value_total = attention_shape.value_total();
    let gating_width = attention_shape.value_heads;
    let attention = match &layer.attention {
        Qwen35AttentionWeights::Linear(attention) => attention,
        Qwen35AttentionWeights::Full(_) => {
            return Err(FerrumError::model(format!(
                "Qwen3.5 batch prefill expected linear attention at layer {}",
                layer.layer_index
            )));
        }
    };

    let mut input_norm = B::alloc(hidden_len);
    if let Some(residual_f32) = residual_f32.as_ref() {
        B::rms_norm_f32_to_activation(
            ctx,
            residual_f32,
            &layer.input_layernorm,
            eps,
            &mut input_norm,
            total_tokens,
            config.hidden_size,
        );
    } else {
        B::rms_norm(
            ctx,
            layer_input,
            &layer.input_layernorm,
            eps,
            &mut input_norm,
            total_tokens,
            config.hidden_size,
        );
    }

    let mut mixed_qkv_raw = B::alloc(total_tokens * qkv_width);
    let mut z_raw = B::alloc(total_tokens * value_total);
    let mut b_raw = B::alloc(total_tokens * gating_width);
    let mut a_raw = B::alloc(total_tokens * gating_width);
    attention
        .qkv_proj
        .forward(ctx, &input_norm, &mut mixed_qkv_raw, total_tokens);
    attention
        .z_proj
        .forward(ctx, &input_norm, &mut z_raw, total_tokens);
    attention
        .b_proj
        .forward(ctx, &input_norm, &mut b_raw, total_tokens);
    attention
        .a_proj
        .forward(ctx, &input_norm, &mut a_raw, total_tokens);

    let conv_state_len = qkv_width * attention_shape.conv_kernel.saturating_sub(1);
    let delta_state_len = attention_shape.state_len();
    let mut initial_conv_states = B::alloc_typed(Dtype::F32, batch_len * conv_state_len);
    let mut initial_delta_states = B::alloc_typed(Dtype::F32, batch_len * delta_state_len);
    for (row, (cache_id, state)) in states.iter_mut().enumerate() {
        let Qwen35LayerRuntimeState::Linear {
            conv_state,
            delta_state,
        } = state.layers.get_mut(layer_index).ok_or_else(|| {
            FerrumError::model(format!(
                "Qwen3.5 missing layer {layer_index} state for {cache_id:?}"
            ))
        })?
        else {
            return Err(FerrumError::model(format!(
                "Qwen3.5 batch prefill expected linear layer state at layer {layer_index} for {cache_id:?}"
            )));
        };
        B::copy_slice(
            ctx,
            conv_state,
            0,
            &mut initial_conv_states,
            row * conv_state_len,
            conv_state_len,
        );
        B::copy_slice(
            ctx,
            delta_state,
            0,
            &mut initial_delta_states,
            row * delta_state_len,
            delta_state_len,
        );
    }

    let attention_out = qwen35_linear_attention_prefill_varlen_core_backend::<B>(
        ctx,
        &mixed_qkv_raw,
        &z_raw,
        &a_raw,
        &b_raw,
        &attention.conv1d_weight,
        &initial_conv_states,
        &attention.a_log,
        &attention.dt_bias,
        &attention.norm_weight,
        &initial_delta_states,
        cu_seqlens,
        batch_len,
        attention_shape,
        eps,
    )?;

    for (row, (cache_id, state)) in states.iter_mut().enumerate() {
        let Qwen35LayerRuntimeState::Linear {
            conv_state,
            delta_state,
        } = state.layers.get_mut(layer_index).ok_or_else(|| {
            FerrumError::model(format!(
                "Qwen3.5 missing layer {layer_index} state for {cache_id:?}"
            ))
        })?
        else {
            return Err(FerrumError::model(format!(
                "Qwen3.5 batch prefill expected linear layer state at layer {layer_index} for {cache_id:?}"
            )));
        };
        B::copy_slice(
            ctx,
            &attention_out.final_conv_states,
            row * conv_state_len,
            conv_state,
            0,
            conv_state_len,
        );
        B::copy_slice(
            ctx,
            &attention_out.final_states,
            row * delta_state_len,
            delta_state,
            0,
            delta_state_len,
        );
    }

    if let Some(pools) = linear_state_pools {
        let conv_slots = pools.conv_states[layer_index].as_mut().ok_or_else(|| {
            FerrumError::model(format!(
                "Qwen3.5 missing linear conv state slot pool for prefill layer {layer_index}"
            ))
        })?;
        let delta_slots = pools.delta_states[layer_index].as_mut().ok_or_else(|| {
            FerrumError::model(format!(
                "Qwen3.5 missing linear delta state slot pool for prefill layer {layer_index}"
            ))
        })?;
        for (row, (cache_id, state)) in states.iter().enumerate() {
            let slot = state.linear_slot.ok_or_else(|| {
                FerrumError::model(format!(
                    "Qwen3.5 missing linear state slot for prefill cache_id={cache_id:?}"
                ))
            })?;
            B::copy_slice(
                ctx,
                &attention_out.final_conv_states,
                row * conv_state_len,
                conv_slots,
                slot * pools.conv_state_len,
                conv_state_len,
            );
            B::copy_slice(
                ctx,
                &attention_out.final_states,
                row * delta_state_len,
                delta_slots,
                slot * pools.delta_state_len,
                delta_state_len,
            );
        }
    }

    let mut delta_activation = B::alloc(total_tokens * value_total);
    B::f32_to_activation(
        ctx,
        &attention_out.delta_norm,
        &mut delta_activation,
        total_tokens * value_total,
    );
    let mut delta_output = B::alloc(hidden_len);
    attention
        .out_proj
        .forward(ctx, &delta_activation, &mut delta_output, total_tokens);

    if let (Some(residual_f32), Some(branch_f32)) =
        (residual_f32.as_deref_mut(), branch_f32.as_deref_mut())
    {
        B::activation_to_f32_shadow(ctx, &delta_output, branch_f32, hidden_len);
        B::add_inplace(ctx, residual_f32, branch_f32, hidden_len);
        qwen35_finish_layer_with_mlp_f32_residual::<B>(
            ctx,
            residual_f32,
            branch_f32,
            layer,
            config,
            total_tokens,
            eps,
        )
    } else {
        let mut residual_after_mixer = B::alloc(hidden_len);
        B::copy_slice(
            ctx,
            layer_input,
            0,
            &mut residual_after_mixer,
            0,
            hidden_len,
        );
        B::add_inplace(ctx, &mut residual_after_mixer, &delta_output, hidden_len);
        qwen35_finish_layer_with_mlp::<B>(
            ctx,
            &residual_after_mixer,
            layer,
            config,
            total_tokens,
            eps,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn qwen35_full_attention_prefill_batch_layer_backend<B: MoeLlmBackend + BackendPagedKv>(
    ctx: &mut B::Context,
    layer_input: &B::Buffer,
    states: &mut [(String, Qwen35SequenceState<B>)],
    pool: &mut (B::Buffer, B::Buffer),
    scratch: &mut Qwen35PagedScratch<B>,
    layer_index: usize,
    layer: &Qwen35LayerWeights<B>,
    rope: &Qwen35BackendRopeCache<B>,
    config: &Qwen35TextConfig,
    use_vllm_paged_attn: bool,
    cu_seqlens: &B::Buffer,
    cu_host: &[u32],
    q_lens: &[usize],
    total_tokens: usize,
    eps: f32,
    mut residual_f32: Option<&mut B::Buffer>,
    mut branch_f32: Option<&mut B::Buffer>,
) -> Result<B::Buffer> {
    let batch_len = states.len();
    if batch_len == 0 {
        return Err(FerrumError::model(
            "Qwen3.5 batch prefill full attention received empty states",
        ));
    }
    if q_lens.len() != batch_len {
        return Err(FerrumError::model(format!(
            "Qwen3.5 full-attention prefill q_lens {} != states {}",
            q_lens.len(),
            batch_len
        )));
    }
    if cu_host.len() != batch_len + 1
        || cu_host.first().copied() != Some(0)
        || cu_host.last().copied() != Some(total_tokens as u32)
    {
        return Err(FerrumError::model(
            "Qwen3.5 full-attention prefill received invalid cu_seqlens",
        ));
    }
    if q_lens.iter().any(|len| *len == 0) {
        return Err(FerrumError::model(
            "Qwen3.5 full-attention prefill does not accept empty sequences",
        ));
    }

    let attention = match &layer.attention {
        Qwen35AttentionWeights::Full(attention) => attention,
        Qwen35AttentionWeights::Linear(_) => {
            return Err(FerrumError::model(format!(
                "Qwen3.5 batch prefill expected full attention at layer {}",
                layer.layer_index
            )));
        }
    };
    let attention_shape = Qwen35FullAttentionShape {
        tokens: total_tokens,
        num_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        rope_dim: config.full_attention_rope_dim(),
        position_offset: 0,
        rope_theta: config.rope_parameters.rope_theta as f32,
        rope_interleaved: config.full_attention_text_rope_interleaved(),
        attn_output_gate: config.attn_output_gate,
    };
    attention_shape.validate()?;
    let hidden_len = total_tokens * config.hidden_size;
    let q_total = attention_shape.q_total();
    let q_proj_total = attention_shape.q_proj_total();
    let kv_total = attention_shape.kv_total();

    let mut input_norm = B::alloc(hidden_len);
    if let Some(residual_f32) = residual_f32.as_ref() {
        B::rms_norm_f32_to_activation(
            ctx,
            residual_f32,
            &layer.input_layernorm,
            eps,
            &mut input_norm,
            total_tokens,
            config.hidden_size,
        );
    } else {
        B::rms_norm(
            ctx,
            layer_input,
            &layer.input_layernorm,
            eps,
            &mut input_norm,
            total_tokens,
            config.hidden_size,
        );
    }

    let mut query_raw = B::alloc(total_tokens * q_proj_total);
    let mut key_raw = B::alloc(total_tokens * kv_total);
    let mut value_raw = B::alloc(total_tokens * kv_total);
    attention
        .q_proj
        .forward(ctx, &input_norm, &mut query_raw, total_tokens);
    attention
        .k_proj
        .forward(ctx, &input_norm, &mut key_raw, total_tokens);
    attention
        .v_proj
        .forward(ctx, &input_norm, &mut value_raw, total_tokens);

    let mut pos_offsets = Vec::with_capacity(batch_len);
    let mut context_lens = Vec::with_capacity(batch_len);
    let mut stacked = Vec::new();
    let mut block_size = None;
    let mut max_blocks_per_seq = None;
    for (row, (cache_id, state)) in states.iter().enumerate() {
        let Qwen35LayerRuntimeState::Full { kv } =
            state.layers.get(layer_index).ok_or_else(|| {
                FerrumError::model(format!(
                    "Qwen3.5 missing layer {layer_index} state for {cache_id:?}"
                ))
            })?
        else {
            return Err(FerrumError::model(format!(
                "Qwen3.5 batch prefill expected full layer state at layer {layer_index} for {cache_id:?}"
            )));
        };
        if kv.block_size == 0 {
            return Err(FerrumError::model(
                "Qwen3.5 batch prefill received non-paged KV metadata",
            ));
        }
        let this_max_blocks = kv.capacity / kv.block_size;
        if this_max_blocks == 0 {
            return Err(FerrumError::model(
                "Qwen3.5 batch prefill received empty paged block table capacity",
            ));
        }
        if block_size.is_some_and(|expected| expected != kv.block_size) {
            return Err(FerrumError::model(
                "Qwen3.5 batch prefill received mixed paged KV block sizes",
            ));
        }
        if max_blocks_per_seq.is_some_and(|expected| expected != this_max_blocks) {
            return Err(FerrumError::model(
                "Qwen3.5 batch prefill received mixed paged KV table widths",
            ));
        }
        block_size = Some(kv.block_size);
        max_blocks_per_seq = Some(this_max_blocks);
        if kv.paged_block_indices.len() > this_max_blocks {
            return Err(FerrumError::model(format!(
                "Qwen3.5 batch prefill block table overflow at layer {layer_index} for {cache_id:?}"
            )));
        }
        let context_len = kv.len + q_lens[row];
        if context_len > kv.capacity {
            return Err(FerrumError::model(format!(
                "Qwen3.5 batch prefill KV overflow at layer {layer_index} for {cache_id:?}: \
                 target={context_len} capacity={}",
                kv.capacity
            )));
        }
        pos_offsets
            .push(u32::try_from(kv.len).map_err(|_| {
                FerrumError::model("Qwen3.5 batch prefill KV position exceeds u32")
            })?);
        context_lens.push(
            u32::try_from(context_len)
                .map_err(|_| FerrumError::model("Qwen3.5 batch prefill KV length exceeds u32"))?,
        );
        let mut blocks = kv.paged_block_indices.clone();
        blocks.resize(this_max_blocks, 0);
        stacked.extend(blocks);
    }
    let block_size = block_size
        .ok_or_else(|| FerrumError::model("Qwen3.5 batch prefill missing paged KV block size"))?;
    let max_blocks_per_seq = max_blocks_per_seq
        .ok_or_else(|| FerrumError::model("Qwen3.5 batch prefill missing paged KV table width"))?;

    scratch.ensure(total_tokens, batch_len, max_blocks_per_seq, q_total);
    let pos_buf = scratch
        .pos_offsets
        .as_mut()
        .expect("qwen35 paged pos missing");
    let bt_buf = scratch
        .block_tables
        .as_mut()
        .expect("qwen35 paged block_tables missing");
    let lens_buf = scratch
        .context_lens
        .as_mut()
        .expect("qwen35 paged context_lens missing");
    B::write_typed::<u32>(ctx, pos_buf, &pos_offsets);
    B::write_typed::<u32>(ctx, bt_buf, &stacked);
    B::write_typed::<u32>(ctx, lens_buf, &context_lens);
    let q_buf = scratch.q.as_mut().expect("qwen35 paged q missing");
    let out_buf = scratch.out.as_mut().expect("qwen35 paged out missing");

    if use_vllm_paged_attn {
        B::qwen35_split_qkv_norm_rope_into_paged_cache_varlen_vllm(
            ctx,
            &query_raw,
            &key_raw,
            &value_raw,
            &attention.q_norm_weight,
            &attention.k_norm_weight,
            &rope.cos,
            &rope.sin,
            q_buf,
            &mut pool.0,
            &mut pool.1,
            cu_seqlens,
            pos_buf,
            bt_buf,
            batch_len,
            total_tokens,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            attention_shape.rope_dim,
            q_proj_total,
            if attention_shape.attn_output_gate {
                2 * config.head_dim
            } else {
                config.head_dim
            },
            kv_total,
            eps,
            if attention_shape.rope_interleaved {
                3
            } else {
                1
            },
            block_size,
            max_blocks_per_seq,
        )?;
    } else {
        B::qwen35_split_qkv_norm_rope_into_paged_cache_varlen(
            ctx,
            &query_raw,
            &key_raw,
            &value_raw,
            &attention.q_norm_weight,
            &attention.k_norm_weight,
            &rope.cos,
            &rope.sin,
            q_buf,
            &mut pool.0,
            &mut pool.1,
            cu_seqlens,
            pos_buf,
            bt_buf,
            batch_len,
            total_tokens,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            attention_shape.rope_dim,
            q_proj_total,
            if attention_shape.attn_output_gate {
                2 * config.head_dim
            } else {
                config.head_dim
            },
            kv_total,
            eps,
            if attention_shape.rope_interleaved {
                3
            } else {
                1
            },
            block_size,
            max_blocks_per_seq,
        )?;
    }

    for (row, (cache_id, state)) in states.iter_mut().enumerate() {
        let Qwen35LayerRuntimeState::Full { kv } =
            state.layers.get_mut(layer_index).ok_or_else(|| {
                FerrumError::model(format!(
                    "Qwen3.5 missing layer {layer_index} state for {cache_id:?}"
                ))
            })?
        else {
            return Err(FerrumError::model(format!(
                "Qwen3.5 batch prefill expected full layer state at layer {layer_index} for {cache_id:?}"
            )));
        };
        kv.len += q_lens[row];
        if kv.len != context_lens[row] as usize {
            return Err(FerrumError::model(format!(
                "Qwen3.5 batch prefill context length mismatch at layer {layer_index} for {cache_id:?}"
            )));
        }
        if let Some(cl) = kv.context_lens.as_mut() {
            B::write_typed::<u32>(ctx, cl, &[context_lens[row]]);
        }
    }

    let max_kv_len = context_lens.iter().copied().max().unwrap_or(1) as usize;
    if use_vllm_paged_attn {
        B::paged_varlen_attention_vllm(
            ctx,
            q_buf,
            &pool.0,
            &pool.1,
            out_buf,
            cu_seqlens,
            pos_buf,
            bt_buf,
            batch_len,
            total_tokens,
            max_kv_len,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            block_size,
            max_blocks_per_seq,
        )?;
    } else {
        B::paged_varlen_attention(
            ctx,
            q_buf,
            &pool.0,
            &pool.1,
            out_buf,
            cu_seqlens,
            pos_buf,
            bt_buf,
            batch_len,
            total_tokens,
            max_kv_len,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            0,
            block_size,
            max_blocks_per_seq,
        )?;
    }

    let mut context = B::alloc(total_tokens * q_total);
    B::copy_slice(ctx, out_buf, 0, &mut context, 0, total_tokens * q_total);
    if attention_shape.attn_output_gate {
        B::qwen35_apply_attention_gate(
            ctx,
            &mut context,
            &query_raw,
            total_tokens,
            q_total,
            q_proj_total,
            config.head_dim,
        )?;
    }
    let mut attn_output = B::alloc(hidden_len);
    attention
        .o_proj
        .forward(ctx, &context, &mut attn_output, total_tokens);

    if let (Some(residual_f32), Some(branch_f32)) =
        (residual_f32.as_deref_mut(), branch_f32.as_deref_mut())
    {
        B::activation_to_f32_shadow(ctx, &attn_output, branch_f32, hidden_len);
        B::add_inplace(ctx, residual_f32, branch_f32, hidden_len);
        qwen35_finish_layer_with_mlp_f32_residual::<B>(
            ctx,
            residual_f32,
            branch_f32,
            layer,
            config,
            total_tokens,
            eps,
        )
    } else {
        let mut residual_after_attention = B::alloc(hidden_len);
        B::copy_slice(
            ctx,
            layer_input,
            0,
            &mut residual_after_attention,
            0,
            hidden_len,
        );
        B::add_inplace(ctx, &mut residual_after_attention, &attn_output, hidden_len);
        qwen35_finish_layer_with_mlp::<B>(
            ctx,
            &residual_after_attention,
            layer,
            config,
            total_tokens,
            eps,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn qwen35_full_attention_decode_batch_layer_backend<B: MoeLlmBackend + BackendPagedKv>(
    ctx: &mut B::Context,
    layer_input: &B::Buffer,
    states: &mut [(String, Qwen35SequenceState<B>)],
    paged: Option<(&mut (B::Buffer, B::Buffer), &mut Qwen35PagedScratch<B>)>,
    layer_index: usize,
    layer: &Qwen35LayerWeights<B>,
    rope: &Qwen35BackendRopeCache<B>,
    config: &Qwen35TextConfig,
    use_vllm_paged_attn: bool,
    eps: f32,
    mut residual_f32: Option<&mut B::Buffer>,
    mut branch_f32: Option<&mut B::Buffer>,
    mut decode_scratch: Option<&mut Qwen35DecodeScratch<B>>,
) -> Result<Option<B::Buffer>> {
    let batch_len = states.len();
    let hidden_len = batch_len * config.hidden_size;
    let attention = match &layer.attention {
        Qwen35AttentionWeights::Full(attention) => attention,
        Qwen35AttentionWeights::Linear(_) => {
            return Err(FerrumError::model(format!(
                "Qwen3.5 batch decode expected full attention at layer {}",
                layer.layer_index
            )));
        }
    };
    let q_total = config.num_attention_heads * config.head_dim;
    let q_proj_total = if config.attn_output_gate {
        2 * q_total
    } else {
        q_total
    };
    let kv_total = config.num_key_value_heads * config.head_dim;

    let mut input_norm = B::alloc(hidden_len);
    if let Some(residual_f32) = residual_f32.as_ref() {
        B::rms_norm_f32_to_activation(
            ctx,
            residual_f32,
            &layer.input_layernorm,
            eps,
            &mut input_norm,
            batch_len,
            config.hidden_size,
        );
    } else {
        B::rms_norm(
            ctx,
            layer_input,
            &layer.input_layernorm,
            eps,
            &mut input_norm,
            batch_len,
            config.hidden_size,
        );
    }

    let mut query_raw = B::alloc(batch_len * q_proj_total);
    let mut key_raw = B::alloc(batch_len * kv_total);
    let mut value_raw = B::alloc(batch_len * kv_total);
    attention
        .q_proj
        .forward(ctx, &input_norm, &mut query_raw, batch_len);
    attention
        .k_proj
        .forward(ctx, &input_norm, &mut key_raw, batch_len);
    attention
        .v_proj
        .forward(ctx, &input_norm, &mut value_raw, batch_len);

    if let Some((pool, scratch)) = paged {
        let mut pos_offsets = Vec::with_capacity(batch_len);
        let mut context_lens = Vec::with_capacity(batch_len);
        let mut stacked = Vec::new();
        let mut block_size = 0usize;
        let mut max_blocks_per_seq = 0usize;
        for (_, state) in states.iter() {
            let Qwen35LayerRuntimeState::Full { kv } =
                state.layers.get(layer_index).ok_or_else(|| {
                    FerrumError::model(format!(
                        "Qwen3.5 missing layer {layer_index} state during paged decode"
                    ))
                })?
            else {
                return Err(FerrumError::model(format!(
                    "Qwen3.5 paged decode expected full layer state at layer {layer_index}"
                )));
            };
            if kv.block_size == 0 {
                return Err(FerrumError::model(
                    "Qwen3.5 paged decode received non-paged KV metadata",
                ));
            }
            if block_size == 0 {
                block_size = kv.block_size;
                max_blocks_per_seq = kv.capacity / kv.block_size;
            }
            pos_offsets.push(kv.len as u32);
            context_lens.push((kv.len + 1) as u32);
            let mut blocks = kv.paged_block_indices.clone();
            blocks.resize(max_blocks_per_seq, 0);
            stacked.extend(blocks);
        }
        let cu: Vec<u32> = (0..=batch_len as u32).collect();
        scratch.ensure(batch_len, batch_len, max_blocks_per_seq, q_total);
        let cu_buf = scratch
            .cu_seqlens_q
            .as_mut()
            .expect("qwen35 paged cu missing");
        let pos_buf = scratch
            .pos_offsets
            .as_mut()
            .expect("qwen35 paged pos missing");
        let bt_buf = scratch
            .block_tables
            .as_mut()
            .expect("qwen35 paged block_tables missing");
        let lens_buf = scratch
            .context_lens
            .as_mut()
            .expect("qwen35 paged context_lens missing");
        B::write_typed::<u32>(ctx, cu_buf, &cu);
        B::write_typed::<u32>(ctx, pos_buf, &pos_offsets);
        B::write_typed::<u32>(ctx, bt_buf, &stacked);
        B::write_typed::<u32>(ctx, lens_buf, &context_lens);
        let q_buf = scratch.q.as_mut().expect("qwen35 paged q missing");
        let out_buf = scratch.out.as_mut().expect("qwen35 paged out missing");
        let attention_shape = Qwen35FullAttentionShape {
            tokens: 1,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rope_dim: config.full_attention_rope_dim(),
            position_offset: 0,
            rope_theta: config.rope_parameters.rope_theta as f32,
            rope_interleaved: config.full_attention_text_rope_interleaved(),
            attn_output_gate: config.attn_output_gate,
        };
        attention_shape.validate()?;
        if use_vllm_paged_attn {
            B::qwen35_split_qkv_norm_rope_into_paged_cache_varlen_vllm(
                ctx,
                &query_raw,
                &key_raw,
                &value_raw,
                &attention.q_norm_weight,
                &attention.k_norm_weight,
                &rope.cos,
                &rope.sin,
                q_buf,
                &mut pool.0,
                &mut pool.1,
                cu_buf,
                pos_buf,
                bt_buf,
                batch_len,
                batch_len,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                attention_shape.rope_dim,
                q_proj_total,
                if attention_shape.attn_output_gate {
                    2 * config.head_dim
                } else {
                    config.head_dim
                },
                kv_total,
                eps,
                if attention_shape.rope_interleaved {
                    3
                } else {
                    1
                },
                block_size,
                max_blocks_per_seq,
            )?;
        } else {
            B::qwen35_split_qkv_norm_rope_into_paged_cache_varlen(
                ctx,
                &query_raw,
                &key_raw,
                &value_raw,
                &attention.q_norm_weight,
                &attention.k_norm_weight,
                &rope.cos,
                &rope.sin,
                q_buf,
                &mut pool.0,
                &mut pool.1,
                cu_buf,
                pos_buf,
                bt_buf,
                batch_len,
                batch_len,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                attention_shape.rope_dim,
                q_proj_total,
                if attention_shape.attn_output_gate {
                    2 * config.head_dim
                } else {
                    config.head_dim
                },
                kv_total,
                eps,
                if attention_shape.rope_interleaved {
                    3
                } else {
                    1
                },
                block_size,
                max_blocks_per_seq,
            )?;
        }
        for (_, state) in states.iter_mut() {
            let Qwen35LayerRuntimeState::Full { kv } =
                state.layers.get_mut(layer_index).ok_or_else(|| {
                    FerrumError::model(format!(
                        "Qwen3.5 missing layer {layer_index} state during paged decode len update"
                    ))
                })?
            else {
                return Err(FerrumError::model(format!(
                    "Qwen3.5 paged decode expected full layer state at layer {layer_index}"
                )));
            };
            kv.len += 1;
            if let Some(cl) = kv.context_lens.as_mut() {
                B::write_typed::<u32>(ctx, cl, &[kv.len as u32]);
            }
        }
        let max_kv_len = context_lens.iter().copied().max().unwrap_or(1) as usize;
        if use_vllm_paged_attn {
            B::paged_decode_attention_v2(
                ctx,
                q_buf,
                &pool.0,
                &pool.1,
                out_buf,
                bt_buf,
                lens_buf,
                batch_len,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                block_size,
                max_blocks_per_seq,
                max_kv_len,
            )?;
        } else {
            B::paged_batched_decode_attention(
                ctx,
                q_buf,
                &pool.0,
                &pool.1,
                out_buf,
                bt_buf,
                lens_buf,
                batch_len,
                max_kv_len,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                block_size,
                max_blocks_per_seq,
            )?;
        }
        let mut context = B::alloc(batch_len * q_total);
        B::copy_slice(ctx, out_buf, 0, &mut context, 0, batch_len * q_total);
        if attention_shape.attn_output_gate {
            B::qwen35_apply_attention_gate(
                ctx,
                &mut context,
                &query_raw,
                batch_len,
                q_total,
                q_proj_total,
                config.head_dim,
            )?;
        }
        let mut attn_output = B::alloc(hidden_len);
        attention
            .o_proj
            .forward(ctx, &context, &mut attn_output, batch_len);
        if let (Some(residual_f32), Some(branch_f32)) =
            (residual_f32.as_deref_mut(), branch_f32.as_deref_mut())
        {
            B::activation_to_f32_shadow(ctx, &attn_output, branch_f32, hidden_len);
            B::add_inplace(ctx, residual_f32, branch_f32, hidden_len);
            return if let Some(scratch) = decode_scratch.as_deref_mut() {
                qwen35_finish_layer_with_mlp_f32_residual_decode_scratch::<B>(
                    ctx,
                    residual_f32,
                    branch_f32,
                    layer,
                    config,
                    batch_len,
                    eps,
                    scratch,
                )?;
                Ok(None)
            } else {
                Ok(Some(qwen35_finish_layer_with_mlp_f32_residual::<B>(
                    ctx,
                    residual_f32,
                    branch_f32,
                    layer,
                    config,
                    batch_len,
                    eps,
                )?))
            };
        }
        let mut residual_after_attention = B::alloc(hidden_len);
        B::copy_slice(
            ctx,
            layer_input,
            0,
            &mut residual_after_attention,
            0,
            hidden_len,
        );
        B::add_inplace(ctx, &mut residual_after_attention, &attn_output, hidden_len);
        return if let Some(scratch) = decode_scratch.as_deref_mut() {
            Ok(Some(qwen35_finish_layer_with_mlp_decode_scratch::<B>(
                ctx,
                &residual_after_attention,
                layer,
                config,
                batch_len,
                eps,
                scratch,
            )?))
        } else {
            Ok(Some(qwen35_finish_layer_with_mlp::<B>(
                ctx,
                &residual_after_attention,
                layer,
                config,
                batch_len,
                eps,
            )?))
        };
    }

    let mut context = B::alloc(batch_len * q_total);
    for (row, (cache_id, state)) in states.iter_mut().enumerate() {
        let Qwen35LayerRuntimeState::Full { kv } =
            state.layers.get_mut(layer_index).ok_or_else(|| {
                FerrumError::model(format!(
                    "Qwen3.5 missing layer {layer_index} state for {cache_id:?}"
                ))
            })?
        else {
            return Err(FerrumError::model(format!(
                "Qwen3.5 batch decode expected full layer state at layer {layer_index} for {cache_id:?}"
            )));
        };
        let position_offset = kv.len;
        if position_offset + 1 > kv.capacity {
            return Err(FerrumError::model(format!(
                "Qwen3.5 full-attention KV overflow at layer {} for {cache_id:?}: target={} capacity={}",
                layer.layer_index,
                position_offset + 1,
                kv.capacity
            )));
        }
        let attention_shape = Qwen35FullAttentionShape {
            tokens: 1,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rope_dim: config.full_attention_rope_dim(),
            position_offset,
            rope_theta: config.rope_parameters.rope_theta as f32,
            rope_interleaved: config.full_attention_text_rope_interleaved(),
            attn_output_gate: config.attn_output_gate,
        };
        attention_shape.validate()?;

        let mut query_head_major = B::alloc(q_total);
        let mut key_head_major = B::alloc(kv_total);
        let mut value_head_major = B::alloc(kv_total);
        B::qk_norm_rope_partial(
            ctx,
            &query_raw,
            &attention.q_norm_weight,
            &rope.cos,
            &rope.sin,
            &mut query_head_major,
            1,
            config.num_attention_heads,
            config.head_dim,
            attention_shape.rope_dim,
            batch_len * q_proj_total,
            row * q_proj_total,
            if attention_shape.attn_output_gate {
                2 * config.head_dim
            } else {
                config.head_dim
            },
            position_offset,
            eps,
            if attention_shape.rope_interleaved {
                3
            } else {
                1
            },
        )?;
        B::qk_norm_rope_partial(
            ctx,
            &key_raw,
            &attention.k_norm_weight,
            &rope.cos,
            &rope.sin,
            &mut key_head_major,
            1,
            config.num_key_value_heads,
            config.head_dim,
            attention_shape.rope_dim,
            batch_len * kv_total,
            row * kv_total,
            config.head_dim,
            position_offset,
            eps,
            if attention_shape.rope_interleaved {
                3
            } else {
                1
            },
        )?;
        B::qk_norm_rope_partial(
            ctx,
            &value_raw,
            &attention.q_norm_weight,
            &rope.cos,
            &rope.sin,
            &mut value_head_major,
            1,
            config.num_key_value_heads,
            config.head_dim,
            config.head_dim,
            batch_len * kv_total,
            row * kv_total,
            config.head_dim,
            0,
            eps,
            0,
        )?;
        B::kv_cache_append_head_major(
            ctx,
            &mut kv.k,
            &mut kv.v,
            position_offset,
            kv.capacity,
            &key_head_major,
            &value_head_major,
            1,
            config.num_key_value_heads,
            config.head_dim,
        );
        kv.len += 1;

        let mut context_head_major = B::alloc(q_total);
        let attn_cfg = AttnConfig {
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            causal: true,
            scale: (config.head_dim as f32).sqrt().recip(),
            kv_seq_stride: kv.capacity,
            sliding_window: 0,
        };
        B::flash_attention(
            ctx,
            &query_head_major,
            &kv.k,
            &kv.v,
            &mut context_head_major,
            1,
            1,
            kv.len,
            position_offset,
            &attn_cfg,
        );
        let mut row_context = B::alloc(q_total);
        B::transpose_head_to_token(
            ctx,
            &context_head_major,
            &mut row_context,
            1,
            config.num_attention_heads,
            config.head_dim,
        );
        if attention_shape.attn_output_gate {
            let mut row_query_raw = B::alloc(q_proj_total);
            B::copy_slice(
                ctx,
                &query_raw,
                row * q_proj_total,
                &mut row_query_raw,
                0,
                q_proj_total,
            );
            B::qwen35_apply_attention_gate(
                ctx,
                &mut row_context,
                &row_query_raw,
                1,
                q_total,
                q_proj_total,
                config.head_dim,
            )?;
        }
        B::copy_slice(ctx, &row_context, 0, &mut context, row * q_total, q_total);
    }

    let mut attn_output = B::alloc(hidden_len);
    attention
        .o_proj
        .forward(ctx, &context, &mut attn_output, batch_len);

    if let (Some(residual_f32), Some(branch_f32)) =
        (residual_f32.as_deref_mut(), branch_f32.as_deref_mut())
    {
        B::activation_to_f32_shadow(ctx, &attn_output, branch_f32, hidden_len);
        B::add_inplace(ctx, residual_f32, branch_f32, hidden_len);
        if let Some(scratch) = decode_scratch.as_deref_mut() {
            qwen35_finish_layer_with_mlp_f32_residual_decode_scratch::<B>(
                ctx,
                residual_f32,
                branch_f32,
                layer,
                config,
                batch_len,
                eps,
                scratch,
            )?;
            Ok(None)
        } else {
            Ok(Some(qwen35_finish_layer_with_mlp_f32_residual::<B>(
                ctx,
                residual_f32,
                branch_f32,
                layer,
                config,
                batch_len,
                eps,
            )?))
        }
    } else {
        let mut residual_after_attention = B::alloc(hidden_len);
        B::copy_slice(
            ctx,
            layer_input,
            0,
            &mut residual_after_attention,
            0,
            hidden_len,
        );
        B::add_inplace(ctx, &mut residual_after_attention, &attn_output, hidden_len);
        if let Some(scratch) = decode_scratch.as_deref_mut() {
            Ok(Some(qwen35_finish_layer_with_mlp_decode_scratch::<B>(
                ctx,
                &residual_after_attention,
                layer,
                config,
                batch_len,
                eps,
                scratch,
            )?))
        } else {
            Ok(Some(qwen35_finish_layer_with_mlp::<B>(
                ctx,
                &residual_after_attention,
                layer,
                config,
                batch_len,
                eps,
            )?))
        }
    }
}

fn qwen35_linear_attention_stateful_layer_backend<B: MoeLlmBackend>(
    ctx: &mut B::Context,
    layer_input: &B::Buffer,
    conv_state: &mut B::Buffer,
    delta_state: &mut B::Buffer,
    layer: &Qwen35LayerWeights<B>,
    config: &Qwen35TextConfig,
    tokens: usize,
    eps: f32,
    mut residual_f32: Option<&mut B::Buffer>,
    mut branch_f32: Option<&mut B::Buffer>,
) -> Result<B::Buffer> {
    let attention_shape = Qwen35LinearAttentionShape {
        tokens,
        key_heads: config.linear_attention.num_key_heads,
        value_heads: config.linear_attention.num_value_heads,
        key_dim: config.linear_attention.key_head_dim,
        value_dim: config.linear_attention.value_head_dim,
        conv_kernel: config.linear_attention.conv_kernel_dim,
    };
    let hidden_len = tokens * config.hidden_size;
    let attention = match &layer.attention {
        Qwen35AttentionWeights::Linear(attention) => attention,
        Qwen35AttentionWeights::Full(_) => {
            return Err(FerrumError::model(format!(
                "Qwen3.5 stateful linear layer expected linear attention at layer {}",
                layer.layer_index
            )));
        }
    };

    let mut input_norm = B::alloc(hidden_len);
    if let Some(residual_f32) = residual_f32.as_ref() {
        B::rms_norm_f32_to_activation(
            ctx,
            residual_f32,
            &layer.input_layernorm,
            eps,
            &mut input_norm,
            tokens,
            config.hidden_size,
        );
    } else {
        B::rms_norm(
            ctx,
            layer_input,
            &layer.input_layernorm,
            eps,
            &mut input_norm,
            tokens,
            config.hidden_size,
        );
    }
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "linear.input_norm",
        &input_norm,
        hidden_len,
    );
    let mut mixed_qkv_raw = B::alloc(attention_shape.mixed_qkv_len());
    let mut z_raw = B::alloc(attention_shape.value_len());
    let mut b_raw = B::alloc(attention_shape.gating_len());
    let mut a_raw = B::alloc(attention_shape.gating_len());
    attention
        .qkv_proj
        .forward(ctx, &input_norm, &mut mixed_qkv_raw, tokens);
    attention
        .z_proj
        .forward(ctx, &input_norm, &mut z_raw, tokens);
    attention
        .b_proj
        .forward(ctx, &input_norm, &mut b_raw, tokens);
    attention
        .a_proj
        .forward(ctx, &input_norm, &mut a_raw, tokens);
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "linear.mixed_qkv_raw",
        &mixed_qkv_raw,
        attention_shape.mixed_qkv_len(),
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "linear.z_raw",
        &z_raw,
        attention_shape.value_len(),
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "linear.b_raw",
        &b_raw,
        attention_shape.gating_len(),
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "linear.a_raw",
        &a_raw,
        attention_shape.gating_len(),
    );

    let delta_norm = if tokens == 1 {
        let attention_out = qwen35_linear_attention_decode_core_backend::<B>(
            ctx,
            &mixed_qkv_raw,
            &z_raw,
            &a_raw,
            &b_raw,
            &attention.conv1d_weight,
            conv_state,
            &attention.a_log,
            &attention.dt_bias,
            &attention.norm_weight,
            delta_state,
            attention_shape,
            eps,
        )?;
        *conv_state = attention_out.next_conv_state;
        *delta_state = attention_out.final_state;
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.query",
            &attention_out.query,
            attention_shape.qk_total(),
        );
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.key",
            &attention_out.key,
            attention_shape.qk_total(),
        );
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.value",
            &attention_out.value,
            attention_shape.value_total(),
        );
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.g",
            &attention_out.g,
            attention_shape.value_heads,
        );
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.beta",
            &attention_out.beta,
            attention_shape.value_heads,
        );
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.delta_core",
            &attention_out.delta_core,
            attention_shape.value_len(),
        );
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.delta_norm",
            &attention_out.delta_norm,
            attention_shape.value_len(),
        );
        attention_out.delta_norm
    } else {
        let final_conv_state =
            qwen35_final_conv_state_backend::<B>(ctx, &mixed_qkv_raw, attention_shape)?;
        let attention_out = qwen35_linear_attention_prefill_core_backend::<B>(
            ctx,
            &mixed_qkv_raw,
            &z_raw,
            &a_raw,
            &b_raw,
            &attention.conv1d_weight,
            &attention.a_log,
            &attention.dt_bias,
            &attention.norm_weight,
            delta_state,
            attention_shape,
            eps,
        )?;
        *conv_state = final_conv_state;
        *delta_state = attention_out.final_state;
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.final_conv_state",
            conv_state,
            attention_shape.conv_channels() * attention_shape.conv_kernel.saturating_sub(1),
        );
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.query",
            &attention_out.query,
            tokens * attention_shape.qk_total(),
        );
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.key",
            &attention_out.key,
            tokens * attention_shape.qk_total(),
        );
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.value",
            &attention_out.value,
            attention_shape.value_len(),
        );
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.g",
            &attention_out.g,
            attention_shape.gating_len(),
        );
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.beta",
            &attention_out.beta,
            attention_shape.gating_len(),
        );
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.delta_core",
            &attention_out.delta_core,
            attention_shape.value_len(),
        );
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.delta_norm",
            &attention_out.delta_norm,
            attention_shape.value_len(),
        );
        attention_out.delta_norm
    };

    let mut delta_activation = B::alloc(attention_shape.value_len());
    B::f32_to_activation(
        ctx,
        &delta_norm,
        &mut delta_activation,
        attention_shape.value_len(),
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "linear.delta_activation",
        &delta_activation,
        attention_shape.value_len(),
    );
    let mut delta_output = B::alloc(hidden_len);
    attention
        .out_proj
        .forward(ctx, &delta_activation, &mut delta_output, tokens);
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "linear.delta_output",
        &delta_output,
        hidden_len,
    );

    if let (Some(residual_f32), Some(branch_f32)) =
        (residual_f32.as_deref_mut(), branch_f32.as_deref_mut())
    {
        B::activation_to_f32_shadow(ctx, &delta_output, branch_f32, hidden_len);
        B::add_inplace(ctx, residual_f32, branch_f32, hidden_len);
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.residual_after_mixer",
            residual_f32,
            hidden_len,
        );
        qwen35_finish_layer_with_mlp_f32_residual::<B>(
            ctx,
            residual_f32,
            branch_f32,
            layer,
            config,
            tokens,
            eps,
        )
    } else {
        let mut residual_after_mixer = B::alloc(hidden_len);
        B::copy_slice(
            ctx,
            layer_input,
            0,
            &mut residual_after_mixer,
            0,
            hidden_len,
        );
        B::add_inplace(ctx, &mut residual_after_mixer, &delta_output, hidden_len);
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "linear.residual_after_mixer",
            &residual_after_mixer,
            hidden_len,
        );
        qwen35_finish_layer_with_mlp::<B>(ctx, &residual_after_mixer, layer, config, tokens, eps)
    }
}

fn qwen35_full_attention_stateful_layer_backend<B: MoeLlmBackend + BackendPagedKv>(
    ctx: &mut B::Context,
    layer_input: &B::Buffer,
    kv: &mut KvCache<B, KvFp16>,
    paged: Option<(&mut (B::Buffer, B::Buffer), &mut Qwen35PagedScratch<B>)>,
    layer: &Qwen35LayerWeights<B>,
    rope: &Qwen35BackendRopeCache<B>,
    config: &Qwen35TextConfig,
    use_vllm_paged_attn: bool,
    tokens: usize,
    eps: f32,
    mut residual_f32: Option<&mut B::Buffer>,
    mut branch_f32: Option<&mut B::Buffer>,
) -> Result<B::Buffer> {
    let attention = match &layer.attention {
        Qwen35AttentionWeights::Full(attention) => attention,
        Qwen35AttentionWeights::Linear(_) => {
            return Err(FerrumError::model(format!(
                "Qwen3.5 stateful full layer expected full attention at layer {}",
                layer.layer_index
            )));
        }
    };
    let position_offset = kv.len;
    if position_offset + tokens > kv.capacity {
        return Err(FerrumError::model(format!(
            "Qwen3.5 full-attention KV overflow at layer {}: target={} capacity={}",
            layer.layer_index,
            position_offset + tokens,
            kv.capacity
        )));
    }
    let attention_shape = Qwen35FullAttentionShape {
        tokens,
        num_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        rope_dim: config.full_attention_rope_dim(),
        position_offset,
        rope_theta: config.rope_parameters.rope_theta as f32,
        rope_interleaved: config.full_attention_text_rope_interleaved(),
        attn_output_gate: config.attn_output_gate,
    };
    attention_shape.validate()?;
    let hidden_len = tokens * config.hidden_size;
    let q_total = attention_shape.q_total();
    let q_proj_total = attention_shape.q_proj_total();
    let kv_total = attention_shape.kv_total();

    let mut input_norm = B::alloc(hidden_len);
    if let Some(residual_f32) = residual_f32.as_ref() {
        B::rms_norm_f32_to_activation(
            ctx,
            residual_f32,
            &layer.input_layernorm,
            eps,
            &mut input_norm,
            tokens,
            config.hidden_size,
        );
    } else {
        B::rms_norm(
            ctx,
            layer_input,
            &layer.input_layernorm,
            eps,
            &mut input_norm,
            tokens,
            config.hidden_size,
        );
    }
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "full.input_norm",
        &input_norm,
        hidden_len,
    );
    let mut query_raw = B::alloc(tokens * q_proj_total);
    let mut key_raw = B::alloc(tokens * kv_total);
    let mut value_raw = B::alloc(tokens * kv_total);
    attention
        .q_proj
        .forward(ctx, &input_norm, &mut query_raw, tokens);
    attention
        .k_proj
        .forward(ctx, &input_norm, &mut key_raw, tokens);
    attention
        .v_proj
        .forward(ctx, &input_norm, &mut value_raw, tokens);
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "full.query_raw",
        &query_raw,
        tokens * q_proj_total,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "full.key_raw",
        &key_raw,
        tokens * kv_total,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "full.value_raw",
        &value_raw,
        tokens * kv_total,
    );

    if let Some((pool, scratch)) = paged {
        let block_size = kv.block_size;
        let max_blocks_per_seq = kv.capacity / block_size.max(1);
        if block_size == 0 || max_blocks_per_seq == 0 {
            return Err(FerrumError::model(
                "Qwen3.5 paged full attention received non-paged KV metadata",
            ));
        }
        let mut padded = kv.paged_block_indices.clone();
        padded.resize(max_blocks_per_seq, 0);
        let cu = vec![0u32, tokens as u32];
        let pos_offsets = vec![position_offset as u32];
        let context_lens = vec![(position_offset + tokens) as u32];
        scratch.ensure(tokens, 1, max_blocks_per_seq, q_total);
        let cu_buf = scratch
            .cu_seqlens_q
            .as_mut()
            .expect("qwen35 paged cu missing");
        let pos_buf = scratch
            .pos_offsets
            .as_mut()
            .expect("qwen35 paged pos missing");
        let bt_buf = scratch
            .block_tables
            .as_mut()
            .expect("qwen35 paged block_tables missing");
        let lens_buf = scratch
            .context_lens
            .as_mut()
            .expect("qwen35 paged context_lens missing");
        B::write_typed::<u32>(ctx, cu_buf, &cu);
        B::write_typed::<u32>(ctx, pos_buf, &pos_offsets);
        B::write_typed::<u32>(ctx, bt_buf, &padded);
        B::write_typed::<u32>(ctx, lens_buf, &context_lens);
        if let Some(cl) = kv.context_lens.as_mut() {
            B::write_typed::<u32>(ctx, cl, &context_lens);
        }
        let q_buf = scratch.q.as_mut().expect("qwen35 paged q missing");
        let out_buf = scratch.out.as_mut().expect("qwen35 paged out missing");
        if use_vllm_paged_attn {
            B::qwen35_split_qkv_norm_rope_into_paged_cache_varlen_vllm(
                ctx,
                &query_raw,
                &key_raw,
                &value_raw,
                &attention.q_norm_weight,
                &attention.k_norm_weight,
                &rope.cos,
                &rope.sin,
                q_buf,
                &mut pool.0,
                &mut pool.1,
                cu_buf,
                pos_buf,
                bt_buf,
                1,
                tokens,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                attention_shape.rope_dim,
                q_proj_total,
                if attention_shape.attn_output_gate {
                    2 * config.head_dim
                } else {
                    config.head_dim
                },
                kv_total,
                eps,
                if attention_shape.rope_interleaved {
                    3
                } else {
                    1
                },
                block_size,
                max_blocks_per_seq,
            )?;
        } else {
            B::qwen35_split_qkv_norm_rope_into_paged_cache_varlen(
                ctx,
                &query_raw,
                &key_raw,
                &value_raw,
                &attention.q_norm_weight,
                &attention.k_norm_weight,
                &rope.cos,
                &rope.sin,
                q_buf,
                &mut pool.0,
                &mut pool.1,
                cu_buf,
                pos_buf,
                bt_buf,
                1,
                tokens,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                attention_shape.rope_dim,
                q_proj_total,
                if attention_shape.attn_output_gate {
                    2 * config.head_dim
                } else {
                    config.head_dim
                },
                kv_total,
                eps,
                if attention_shape.rope_interleaved {
                    3
                } else {
                    1
                },
                block_size,
                max_blocks_per_seq,
            )?;
        }
        kv.len += tokens;
        if use_vllm_paged_attn {
            B::paged_varlen_attention_vllm(
                ctx,
                q_buf,
                &pool.0,
                &pool.1,
                out_buf,
                cu_buf,
                pos_buf,
                bt_buf,
                1,
                tokens,
                kv.len,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                block_size,
                max_blocks_per_seq,
            )?;
        } else {
            B::paged_varlen_attention(
                ctx,
                q_buf,
                &pool.0,
                &pool.1,
                out_buf,
                cu_buf,
                pos_buf,
                bt_buf,
                1,
                tokens,
                kv.len,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                0,
                block_size,
                max_blocks_per_seq,
            )?;
        }
        let mut context = B::alloc(tokens * q_total);
        B::copy_slice(ctx, out_buf, 0, &mut context, 0, tokens * q_total);
        if attention_shape.attn_output_gate {
            B::qwen35_apply_attention_gate(
                ctx,
                &mut context,
                &query_raw,
                tokens,
                q_total,
                q_proj_total,
                config.head_dim,
            )?;
        }
        let mut attn_output = B::alloc(hidden_len);
        attention
            .o_proj
            .forward(ctx, &context, &mut attn_output, tokens);
        if let (Some(residual_f32), Some(branch_f32)) =
            (residual_f32.as_deref_mut(), branch_f32.as_deref_mut())
        {
            B::activation_to_f32_shadow(ctx, &attn_output, branch_f32, hidden_len);
            B::add_inplace(ctx, residual_f32, branch_f32, hidden_len);
            return qwen35_finish_layer_with_mlp_f32_residual::<B>(
                ctx,
                residual_f32,
                branch_f32,
                layer,
                config,
                tokens,
                eps,
            );
        }
        let mut residual_after_attention = B::alloc(hidden_len);
        B::copy_slice(
            ctx,
            layer_input,
            0,
            &mut residual_after_attention,
            0,
            hidden_len,
        );
        B::add_inplace(ctx, &mut residual_after_attention, &attn_output, hidden_len);
        return qwen35_finish_layer_with_mlp::<B>(
            ctx,
            &residual_after_attention,
            layer,
            config,
            tokens,
            eps,
        );
    }

    let mut query_head_major = B::alloc(tokens * q_total);
    let mut key_head_major = B::alloc(tokens * kv_total);
    let mut value_head_major = B::alloc(tokens * kv_total);
    B::qk_norm_rope_partial(
        ctx,
        &query_raw,
        &attention.q_norm_weight,
        &rope.cos,
        &rope.sin,
        &mut query_head_major,
        tokens,
        config.num_attention_heads,
        config.head_dim,
        attention_shape.rope_dim,
        q_proj_total,
        0,
        if attention_shape.attn_output_gate {
            2 * config.head_dim
        } else {
            config.head_dim
        },
        position_offset,
        eps,
        if attention_shape.rope_interleaved {
            3
        } else {
            1
        },
    )?;
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "full.query_head_major",
        &query_head_major,
        tokens * q_total,
    );
    B::qk_norm_rope_partial(
        ctx,
        &key_raw,
        &attention.k_norm_weight,
        &rope.cos,
        &rope.sin,
        &mut key_head_major,
        tokens,
        config.num_key_value_heads,
        config.head_dim,
        attention_shape.rope_dim,
        kv_total,
        0,
        config.head_dim,
        position_offset,
        eps,
        if attention_shape.rope_interleaved {
            3
        } else {
            1
        },
    )?;
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "full.key_head_major",
        &key_head_major,
        tokens * kv_total,
    );
    B::qk_norm_rope_partial(
        ctx,
        &value_raw,
        &attention.q_norm_weight,
        &rope.cos,
        &rope.sin,
        &mut value_head_major,
        tokens,
        config.num_key_value_heads,
        config.head_dim,
        config.head_dim,
        kv_total,
        0,
        config.head_dim,
        0,
        eps,
        0,
    )?;
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "full.value_head_major",
        &value_head_major,
        tokens * kv_total,
    );
    B::kv_cache_append_head_major(
        ctx,
        &mut kv.k,
        &mut kv.v,
        position_offset,
        kv.capacity,
        &key_head_major,
        &value_head_major,
        tokens,
        config.num_key_value_heads,
        config.head_dim,
    );
    kv.len += tokens;

    let mut context_head_major = B::alloc(tokens * q_total);
    let attn_cfg = AttnConfig {
        num_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        causal: true,
        scale: (config.head_dim as f32).sqrt().recip(),
        kv_seq_stride: kv.capacity,
        sliding_window: 0,
    };
    B::flash_attention(
        ctx,
        &query_head_major,
        &kv.k,
        &kv.v,
        &mut context_head_major,
        1,
        tokens,
        kv.len,
        position_offset,
        &attn_cfg,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "full.context_head_major",
        &context_head_major,
        tokens * q_total,
    );
    let mut context = B::alloc(tokens * q_total);
    B::transpose_head_to_token(
        ctx,
        &context_head_major,
        &mut context,
        tokens,
        config.num_attention_heads,
        config.head_dim,
    );
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "full.context",
        &context,
        tokens * q_total,
    );
    if attention_shape.attn_output_gate {
        B::qwen35_apply_attention_gate(
            ctx,
            &mut context,
            &query_raw,
            tokens,
            q_total,
            q_proj_total,
            config.head_dim,
        )?;
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "full.context_after_gate",
            &context,
            tokens * q_total,
        );
    }

    let mut attn_output = B::alloc(hidden_len);
    attention
        .o_proj
        .forward(ctx, &context, &mut attn_output, tokens);
    qwen35_trace_layer_buffer_stats::<B>(
        ctx,
        layer.layer_index,
        "full.attn_output",
        &attn_output,
        hidden_len,
    );
    if let (Some(residual_f32), Some(branch_f32)) =
        (residual_f32.as_deref_mut(), branch_f32.as_deref_mut())
    {
        B::activation_to_f32_shadow(ctx, &attn_output, branch_f32, hidden_len);
        B::add_inplace(ctx, residual_f32, branch_f32, hidden_len);
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "full.residual_after_attention",
            residual_f32,
            hidden_len,
        );
        qwen35_finish_layer_with_mlp_f32_residual::<B>(
            ctx,
            residual_f32,
            branch_f32,
            layer,
            config,
            tokens,
            eps,
        )
    } else {
        let mut residual_after_attention = B::alloc(hidden_len);
        B::copy_slice(
            ctx,
            layer_input,
            0,
            &mut residual_after_attention,
            0,
            hidden_len,
        );
        B::add_inplace(ctx, &mut residual_after_attention, &attn_output, hidden_len);
        qwen35_trace_layer_buffer_stats::<B>(
            ctx,
            layer.layer_index,
            "full.residual_after_attention",
            &residual_after_attention,
            hidden_len,
        );
        qwen35_finish_layer_with_mlp::<B>(
            ctx,
            &residual_after_attention,
            layer,
            config,
            tokens,
            eps,
        )
    }
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

    let final_conv_state = qwen35_final_conv_state_cpu(mixed_qkv_raw, shape)?;
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
        final_conv_state,
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

pub fn qwen35_final_conv_state_cpu(
    mixed_qkv_raw: &[f32],
    shape: Qwen35LinearAttentionShape,
) -> Result<Vec<f32>> {
    shape.validate()?;
    validate_len("mixed_qkv_raw", mixed_qkv_raw.len(), shape.mixed_qkv_len())?;
    let channels = shape.conv_channels();
    let state_len = shape.conv_kernel.saturating_sub(1);
    let mut state = vec![0.0; channels * state_len];
    if state_len == 0 {
        return Ok(state);
    }

    let copied_tokens = state_len.min(shape.tokens);
    let source_token_start = shape.tokens - copied_tokens;
    let state_token_start = state_len - copied_tokens;
    for channel in 0..channels {
        for state_token in 0..copied_tokens {
            let source_token = source_token_start + state_token;
            state[channel * state_len + state_token_start + state_token] =
                mixed_qkv_raw[source_token * channels + channel];
        }
    }
    Ok(state)
}

pub fn qwen35_final_conv_state_backend<B: Backend>(
    ctx: &mut B::Context,
    mixed_qkv_raw: &B::Buffer,
    shape: Qwen35LinearAttentionShape,
) -> Result<B::Buffer> {
    shape.validate()?;
    let channels = shape.conv_channels();
    let state_tokens = shape.conv_kernel.saturating_sub(1);
    let state_len = channels * state_tokens;
    let mut state = B::from_slice_typed::<f32>(&vec![0.0; state_len]);
    if state_tokens == 0 {
        return Ok(state);
    }

    let mut mixed_qkv_f32 = B::alloc_typed(Dtype::F32, shape.mixed_qkv_len());
    B::activation_to_f32_shadow(
        ctx,
        mixed_qkv_raw,
        &mut mixed_qkv_f32,
        shape.mixed_qkv_len(),
    );

    let copied_tokens = state_tokens.min(shape.tokens);
    let source_token_start = shape.tokens - copied_tokens;
    let state_token_start = state_tokens - copied_tokens;
    for channel in 0..channels {
        for state_token in 0..copied_tokens {
            let source_token = source_token_start + state_token;
            let src_offset = source_token * channels + channel;
            let dst_offset = channel * state_tokens + state_token_start + state_token;
            B::copy_slice(ctx, &mixed_qkv_f32, src_offset, &mut state, dst_offset, 1);
        }
    }

    Ok(state)
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
        let storage_dtype = qwen35_recurrent_backend_dtype(spec.dtype)?;
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
            tensors.push(Qwen35RecurrentStateTensor {
                layer_index: tensor.layer_index,
                name: tensor.name.clone(),
                shape: tensor.shape.clone(),
                elements_per_slot,
                buffer: B::alloc_typed(storage_dtype, total_elements),
            });
        }
        Ok(Self {
            request_id: spec.request_id.clone(),
            num_layers: spec.num_layers,
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

impl<B: Backend> Qwen35RecurrentStateHandle<B> {
    pub fn from_spec(spec: &RecurrentStateSpec) -> Result<Self> {
        Self::from_cache(Qwen35RecurrentStateCache::<B>::from_spec(spec)?)
    }

    pub fn from_cache(cache: Qwen35RecurrentStateCache<B>) -> Result<Self> {
        let request_id = cache.request_id.clone();
        Ok(Self {
            cache: Arc::new(Mutex::new(cache)),
            cache_id: format!("qwen35-recurrent-state-{request_id}"),
            valid: Arc::new(AtomicBool::new(true)),
            created_at: Instant::now(),
        })
    }

    pub fn cache(&self) -> MutexGuard<'_, Qwen35RecurrentStateCache<B>> {
        self.cache.lock()
    }

    fn invalidate(&self) {
        self.valid.store(false, Ordering::Relaxed);
    }
}

impl Default for Qwen35RecurrentStateManagerConfig {
    fn default() -> Self {
        Self {
            total_memory_bytes: usize::MAX,
            total_batch_slots: usize::MAX,
        }
    }
}

impl<B: Backend> Qwen35RecurrentStateManager<B> {
    pub fn new(config: Qwen35RecurrentStateManagerConfig) -> Self {
        Self {
            config,
            handles: Mutex::new(HashMap::new()),
            allocation_count: AtomicU64::new(0),
            allocation_failures: AtomicU64::new(0),
        }
    }

    fn used_memory_bytes_locked(
        handles: &HashMap<RequestId, Arc<Qwen35RecurrentStateHandle<B>>>,
    ) -> usize {
        handles
            .values()
            .map(|handle| handle.cache.lock().estimated_memory_bytes())
            .sum()
    }

    fn used_batch_slots_locked(
        handles: &HashMap<RequestId, Arc<Qwen35RecurrentStateHandle<B>>>,
    ) -> usize {
        handles
            .values()
            .map(|handle| handle.cache.lock().max_batch_slots)
            .sum()
    }

    fn active_state_tensors_locked(
        handles: &HashMap<RequestId, Arc<Qwen35RecurrentStateHandle<B>>>,
    ) -> usize {
        handles
            .values()
            .map(|handle| handle.cache.lock().tensors.len())
            .sum()
    }
}

impl<B: Backend> fmt::Debug for Qwen35RecurrentStateHandle<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cache = self.cache.lock();
        f.debug_struct("Qwen35RecurrentStateHandle")
            .field("request_id", &cache.request_id)
            .field("dtype", &cache.dtype)
            .field("device", &cache.device)
            .field("max_batch_slots", &cache.max_batch_slots)
            .field("state_tensors", &cache.tensors.len())
            .field("cache_id", &self.cache_id)
            .field("valid", &self.valid.load(Ordering::Relaxed))
            .finish()
    }
}

impl<B: Backend> fmt::Debug for Qwen35RecurrentStateManager<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Qwen35RecurrentStateManager")
            .field("config", &self.config)
            .field("active_states", &self.handles.lock().len())
            .finish()
    }
}

impl<B: Backend> RecurrentStateHandle for Qwen35RecurrentStateHandle<B> {
    fn request_id(&self) -> RequestId {
        self.cache.lock().request_id.clone()
    }

    fn device(&self) -> Device {
        self.cache.lock().device.clone()
    }

    fn num_layers(&self) -> usize {
        self.cache.lock().num_layers
    }

    fn state_bytes(&self) -> usize {
        self.cache.lock().estimated_memory_bytes()
    }

    fn clone_handle(&self) -> Result<Arc<dyn RecurrentStateHandle>> {
        Ok(Arc::new(Self {
            cache: Arc::clone(&self.cache),
            cache_id: self.cache_id.clone(),
            valid: Arc::clone(&self.valid),
            created_at: self.created_at,
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn stats(&self) -> RecurrentStateHandleStats {
        let cache = self.cache.lock();
        RecurrentStateHandleStats {
            memory_bytes: cache.estimated_memory_bytes(),
            state_tensors: cache.tensors.len(),
            batch_slots: cache.max_batch_slots,
            last_access: self.created_at,
        }
    }

    fn is_valid(&self) -> bool {
        self.valid.load(Ordering::Relaxed)
    }

    fn cache_id(&self) -> String {
        self.cache_id.clone()
    }
}

#[async_trait::async_trait]
impl<B: Backend> RecurrentStateManager for Qwen35RecurrentStateManager<B> {
    async fn allocate(&self, spec: &RecurrentStateSpec) -> Result<Arc<dyn RecurrentStateHandle>> {
        let mut handles = self.handles.lock();
        if handles.contains_key(&spec.request_id) {
            self.allocation_failures.fetch_add(1, Ordering::Relaxed);
            return Err(FerrumError::already_exists(format!(
                "Qwen3.5 recurrent state already allocated for {}",
                spec.request_id
            )));
        }

        let requested_bytes = spec.estimated_memory_bytes();
        let projected_memory =
            Self::used_memory_bytes_locked(&handles).saturating_add(requested_bytes);
        let projected_slots =
            Self::used_batch_slots_locked(&handles).saturating_add(spec.max_batch_slots);
        if projected_memory > self.config.total_memory_bytes
            || projected_slots > self.config.total_batch_slots
        {
            self.allocation_failures.fetch_add(1, Ordering::Relaxed);
            return Err(FerrumError::resource_exhausted(
                "insufficient Qwen3.5 recurrent-state capacity",
            ));
        }

        let handle = match Qwen35RecurrentStateHandle::<B>::from_spec(spec) {
            Ok(handle) => Arc::new(handle),
            Err(err) => {
                self.allocation_failures.fetch_add(1, Ordering::Relaxed);
                return Err(err);
            }
        };
        handles.insert(spec.request_id.clone(), Arc::clone(&handle));
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        Ok(handle)
    }

    async fn deallocate(&self, request_id: RequestId) -> Result<()> {
        if let Some(handle) = self.handles.lock().remove(&request_id) {
            handle.invalidate();
        }
        Ok(())
    }

    fn can_allocate(&self, spec: &RecurrentStateSpec) -> bool {
        let handles = self.handles.lock();
        if handles.contains_key(&spec.request_id) {
            return false;
        }
        Self::used_memory_bytes_locked(&handles).saturating_add(spec.estimated_memory_bytes())
            <= self.config.total_memory_bytes
            && Self::used_batch_slots_locked(&handles).saturating_add(spec.max_batch_slots)
                <= self.config.total_batch_slots
    }

    fn get_handle(&self, request_id: RequestId) -> Option<Arc<dyn RecurrentStateHandle>> {
        self.handles
            .lock()
            .get(&request_id)
            .map(|handle| handle.clone() as Arc<dyn RecurrentStateHandle>)
    }

    fn list_handles(&self) -> Vec<(RequestId, Arc<dyn RecurrentStateHandle>)> {
        self.handles
            .lock()
            .iter()
            .map(|(request_id, handle)| {
                (
                    request_id.clone(),
                    handle.clone() as Arc<dyn RecurrentStateHandle>,
                )
            })
            .collect()
    }

    fn stats(&self) -> RecurrentStateManagerStats {
        let handles = self.handles.lock();
        RecurrentStateManagerStats {
            total_memory_bytes: self.config.total_memory_bytes,
            used_memory_bytes: Self::used_memory_bytes_locked(&handles),
            active_states: handles.len(),
            active_state_tensors: Self::active_state_tensors_locked(&handles),
            total_batch_slots: self.config.total_batch_slots,
            used_batch_slots: Self::used_batch_slots_locked(&handles),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            allocation_failures: self.allocation_failures.load(Ordering::Relaxed),
            eviction_count: 0,
        }
    }

    async fn reset(&self) -> Result<()> {
        let mut handles = self.handles.lock();
        for handle in handles.values() {
            handle.invalidate();
        }
        handles.clear();
        Ok(())
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

fn qwen35_recurrent_backend_dtype(dtype: DataType) -> Result<Dtype> {
    match dtype {
        DataType::FP32 => Ok(Dtype::F32),
        DataType::FP16 | DataType::BF16 => Ok(Dtype::F16),
        other => Err(FerrumError::unsupported(format!(
            "Qwen3.5 recurrent state dtype {other:?} is not supported"
        ))),
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

#[allow(clippy::too_many_arguments)]
pub fn qwen35_recurrent_gated_delta_rule_backend<B: Backend>(
    ctx: &mut B::Context,
    query: &B::Buffer,
    key: &B::Buffer,
    value: &B::Buffer,
    g: &B::Buffer,
    beta: &B::Buffer,
    initial_state: &B::Buffer,
    shape: Qwen35DeltaRuleShape,
    use_qk_l2norm: bool,
    scale: Option<f32>,
) -> Result<Qwen35BackendDeltaRuleOutput<B>> {
    validate_delta_rule_shape_values(shape)?;
    let scale = scale.unwrap_or_else(|| (shape.key_dim as f32).sqrt().recip());
    let mut output = B::alloc_typed(
        Dtype::F32,
        shape.tokens * shape.value_heads * shape.value_dim,
    );
    let mut final_state = B::alloc_typed(
        Dtype::F32,
        shape.value_heads * shape.value_dim * shape.key_dim,
    );
    B::recurrent_gated_delta_rule_f32(
        ctx,
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        &mut output,
        &mut final_state,
        shape.tokens,
        shape.key_heads,
        shape.value_heads,
        shape.key_dim,
        shape.value_dim,
        use_qk_l2norm,
        scale,
    )?;
    Ok(Qwen35BackendDeltaRuleOutput {
        output,
        final_state,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_recurrent_gated_delta_rule_varlen_backend<B: Backend>(
    ctx: &mut B::Context,
    query: &B::Buffer,
    key: &B::Buffer,
    value: &B::Buffer,
    g: &B::Buffer,
    beta: &B::Buffer,
    initial_states: &B::Buffer,
    cu_seqlens: &B::Buffer,
    batch: usize,
    shape: Qwen35DeltaRuleShape,
    use_qk_l2norm: bool,
    scale: Option<f32>,
) -> Result<Qwen35BackendVarlenDeltaRuleOutput<B>> {
    validate_delta_rule_shape_values(shape)?;
    if batch == 0 {
        return Err(FerrumError::model(
            "Qwen3.5 DeltaNet varlen batch must be positive",
        ));
    }
    let scale = scale.unwrap_or_else(|| (shape.key_dim as f32).sqrt().recip());
    let mut output = B::alloc_typed(
        Dtype::F32,
        shape.tokens * shape.value_heads * shape.value_dim,
    );
    let mut final_states = B::alloc_typed(
        Dtype::F32,
        batch * shape.value_heads * shape.value_dim * shape.key_dim,
    );
    B::recurrent_gated_delta_rule_varlen_f32(
        ctx,
        query,
        key,
        value,
        g,
        beta,
        initial_states,
        cu_seqlens,
        &mut output,
        &mut final_states,
        batch,
        shape.tokens,
        shape.key_heads,
        shape.value_heads,
        shape.key_dim,
        shape.value_dim,
        use_qk_l2norm,
        scale,
    )?;
    Ok(Qwen35BackendVarlenDeltaRuleOutput {
        output,
        final_states,
    })
}

/// Backend-native Qwen3.5 linear-attention core for a prefill segment with
/// zero external conv-state dependency.
///
/// This mirrors vLLM's prefill ordering: post-conv preparation writes Q/K in
/// L2-normalized form, so the recurrent DeltaNet call disables in-kernel
/// Q/K normalization. Stateful decode needs a separate conv-state update path.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_linear_attention_prefill_core_backend<B: Backend>(
    ctx: &mut B::Context,
    mixed_qkv_raw: &B::Buffer,
    z_raw: &B::Buffer,
    a_raw: &B::Buffer,
    b_raw: &B::Buffer,
    conv1d_weight: &B::Buffer,
    a_log: &B::Buffer,
    dt_bias: &B::Buffer,
    norm_weight: &B::Buffer,
    initial_state: &B::Buffer,
    shape: Qwen35LinearAttentionShape,
    eps: f32,
) -> Result<Qwen35BackendLinearAttentionPrefillOutput<B>> {
    shape.validate()?;
    let qk_total = shape.qk_total();
    let value_total = shape.value_total();
    let value_len = shape.value_len();
    let gating_len = shape.gating_len();

    let mut query = B::alloc_typed(Dtype::F32, shape.tokens * qk_total);
    let mut key = B::alloc_typed(Dtype::F32, shape.tokens * qk_total);
    let mut value = B::alloc_typed(Dtype::F32, value_len);
    let mut g = B::alloc_typed(Dtype::F32, gating_len);
    let mut beta = B::alloc_typed(Dtype::F32, gating_len);
    B::linear_attention_prepare_f32(
        ctx,
        mixed_qkv_raw,
        conv1d_weight,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        &mut query,
        &mut key,
        &mut value,
        &mut g,
        &mut beta,
        shape.tokens,
        shape.key_heads,
        shape.value_heads,
        shape.key_dim,
        shape.value_dim,
        shape.conv_kernel,
        true,
    )?;

    let delta = qwen35_recurrent_gated_delta_rule_backend::<B>(
        ctx,
        &query,
        &key,
        &value,
        &g,
        &beta,
        initial_state,
        shape.delta_shape(),
        false,
        Some((shape.key_dim as f32).sqrt().recip()),
    )?;
    let mut delta_norm = B::alloc_typed(Dtype::F32, value_len);
    B::gated_rms_norm_f32(
        ctx,
        &delta.output,
        z_raw,
        norm_weight,
        &mut delta_norm,
        shape.tokens,
        shape.value_heads,
        shape.value_dim,
        eps,
    )?;

    Ok(Qwen35BackendLinearAttentionPrefillOutput {
        query,
        key,
        value,
        g,
        beta,
        delta_core: delta.output,
        delta_norm,
        final_state: delta.final_state,
    })
}

/// Backend-native Qwen3.5 varlen linear-attention prefill core.
///
/// This is the product-batchable counterpart of
/// [`qwen35_linear_attention_prefill_core_backend`]. The flat token axis is
/// partitioned by `cu_seqlens`; each sequence reads its own causal-conv and
/// DeltaNet initial states and writes one final state for each.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_linear_attention_prefill_varlen_core_backend<B: Backend>(
    ctx: &mut B::Context,
    mixed_qkv_raw: &B::Buffer,
    z_raw: &B::Buffer,
    a_raw: &B::Buffer,
    b_raw: &B::Buffer,
    conv1d_weight: &B::Buffer,
    initial_conv_states: &B::Buffer,
    a_log: &B::Buffer,
    dt_bias: &B::Buffer,
    norm_weight: &B::Buffer,
    initial_states: &B::Buffer,
    cu_seqlens: &B::Buffer,
    batch: usize,
    shape: Qwen35LinearAttentionShape,
    eps: f32,
) -> Result<Qwen35BackendLinearAttentionVarlenPrefillOutput<B>> {
    shape.validate()?;
    if batch == 0 {
        return Err(FerrumError::model(
            "Qwen3.5 varlen linear-attention batch must be positive",
        ));
    }
    let qk_total = shape.qk_total();
    let value_len = shape.value_len();
    let gating_len = shape.gating_len();
    let conv_state_len = shape.conv_channels() * shape.conv_kernel.saturating_sub(1);

    let mut query = B::alloc_typed(Dtype::F32, shape.tokens * qk_total);
    let mut key = B::alloc_typed(Dtype::F32, shape.tokens * qk_total);
    let mut value = B::alloc_typed(Dtype::F32, value_len);
    let mut g = B::alloc_typed(Dtype::F32, gating_len);
    let mut beta = B::alloc_typed(Dtype::F32, gating_len);
    let mut final_conv_states = B::alloc_typed(Dtype::F32, batch * conv_state_len);
    B::linear_attention_prepare_varlen_f32(
        ctx,
        mixed_qkv_raw,
        conv1d_weight,
        initial_conv_states,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        cu_seqlens,
        &mut query,
        &mut key,
        &mut value,
        &mut g,
        &mut beta,
        &mut final_conv_states,
        batch,
        shape.tokens,
        shape.key_heads,
        shape.value_heads,
        shape.key_dim,
        shape.value_dim,
        shape.conv_kernel,
        true,
    )?;

    let delta = qwen35_recurrent_gated_delta_rule_varlen_backend::<B>(
        ctx,
        &query,
        &key,
        &value,
        &g,
        &beta,
        initial_states,
        cu_seqlens,
        batch,
        shape.delta_shape(),
        false,
        Some((shape.key_dim as f32).sqrt().recip()),
    )?;
    let mut delta_norm = B::alloc_typed(Dtype::F32, value_len);
    B::gated_rms_norm_f32(
        ctx,
        &delta.output,
        z_raw,
        norm_weight,
        &mut delta_norm,
        shape.tokens,
        shape.value_heads,
        shape.value_dim,
        eps,
    )?;

    Ok(Qwen35BackendLinearAttentionVarlenPrefillOutput {
        query,
        key,
        value,
        g,
        beta,
        final_conv_states,
        delta_core: delta.output,
        delta_norm,
        final_states: delta.final_states,
    })
}

/// Backend-native Qwen3.5 linear-attention core for one decode token.
///
/// This mirrors vLLM's decode ordering: update the dim-first causal-conv state,
/// emit L2-normalized Q/K for the current token, then update the temporal
/// DeltaNet state and apply gated RMSNorm.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_linear_attention_decode_core_backend<B: Backend>(
    ctx: &mut B::Context,
    mixed_qkv_raw: &B::Buffer,
    z_raw: &B::Buffer,
    a_raw: &B::Buffer,
    b_raw: &B::Buffer,
    conv1d_weight: &B::Buffer,
    conv_state: &B::Buffer,
    a_log: &B::Buffer,
    dt_bias: &B::Buffer,
    norm_weight: &B::Buffer,
    initial_delta_state: &B::Buffer,
    shape: Qwen35LinearAttentionShape,
    eps: f32,
) -> Result<Qwen35BackendLinearAttentionDecodeOutput<B>> {
    shape.validate()?;
    if shape.tokens != 1 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 decode linear-attention core expects one token, got {}",
            shape.tokens
        )));
    }

    let qk_total = shape.qk_total();
    let value_total = shape.value_total();
    let value_len = shape.value_len();
    let gating_len = shape.gating_len();
    let conv_state_len = shape.conv_channels() * shape.conv_kernel.saturating_sub(1);

    let mut query = B::alloc_typed(Dtype::F32, qk_total);
    let mut key = B::alloc_typed(Dtype::F32, qk_total);
    let mut value = B::alloc_typed(Dtype::F32, value_total);
    let mut g = B::alloc_typed(Dtype::F32, gating_len);
    let mut beta = B::alloc_typed(Dtype::F32, gating_len);
    let mut next_conv_state = B::alloc_typed(Dtype::F32, conv_state_len);
    B::linear_attention_decode_prepare_f32(
        ctx,
        mixed_qkv_raw,
        conv1d_weight,
        conv_state,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        &mut query,
        &mut key,
        &mut value,
        &mut g,
        &mut beta,
        &mut next_conv_state,
        shape.key_heads,
        shape.value_heads,
        shape.key_dim,
        shape.value_dim,
        shape.conv_kernel,
        true,
    )?;

    let delta = qwen35_recurrent_gated_delta_rule_backend::<B>(
        ctx,
        &query,
        &key,
        &value,
        &g,
        &beta,
        initial_delta_state,
        shape.delta_shape(),
        false,
        Some((shape.key_dim as f32).sqrt().recip()),
    )?;
    let mut delta_norm = B::alloc_typed(Dtype::F32, value_len);
    B::gated_rms_norm_f32(
        ctx,
        &delta.output,
        z_raw,
        norm_weight,
        &mut delta_norm,
        shape.tokens,
        shape.value_heads,
        shape.value_dim,
        eps,
    )?;

    Ok(Qwen35BackendLinearAttentionDecodeOutput {
        query,
        key,
        value,
        g,
        beta,
        next_conv_state,
        delta_core: delta.output,
        delta_norm,
        final_state: delta.final_state,
    })
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
            || self.rope_dim == 0
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
        if self.rope_dim > self.head_dim || self.rope_dim % 2 != 0 {
            return Err(FerrumError::model(format!(
                "Qwen3.5 full-attention rope_dim {} must be even and <= head_dim {}",
                self.rope_dim, self.head_dim
            )));
        }
        Ok(())
    }

    fn q_total(self) -> usize {
        self.num_heads * self.head_dim
    }

    fn q_proj_total(self) -> usize {
        let base = self.q_total();
        if self.attn_output_gate {
            base * 2
        } else {
            base
        }
    }

    fn kv_total(self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    fn q_len(self) -> usize {
        self.tokens * self.q_total()
    }

    fn q_proj_len(self) -> usize {
        self.tokens * self.q_proj_total()
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
        Ok(())
    }
}

impl Qwen35SparseMoeLinearAttentionLayerShape {
    fn validate(self) -> Result<()> {
        if self.tokens == 0 || self.hidden_size == 0 {
            return Err(FerrumError::model(format!(
                "Qwen3.5 sparse-MoE linear-attention layer shape must be positive, got {self:?}"
            )));
        }
        self.attention.validate()?;
        self.moe.validate()?;
        if self.attention.tokens != self.tokens {
            return Err(FerrumError::model(format!(
                "Qwen3.5 sparse-MoE linear layer tokens {} must match attention tokens {}",
                self.tokens, self.attention.tokens
            )));
        }
        if self.moe.tokens != self.tokens || self.moe.hidden_size != self.hidden_size {
            return Err(FerrumError::model(format!(
                "Qwen3.5 sparse-MoE linear layer shape tokens={} hidden={} must match MoE {:?}",
                self.tokens, self.hidden_size, self.moe
            )));
        }
        Ok(())
    }
}

impl Qwen35SparseMoeFullAttentionLayerShape {
    fn validate(self) -> Result<()> {
        if self.tokens == 0 || self.hidden_size == 0 {
            return Err(FerrumError::model(format!(
                "Qwen3.5 sparse-MoE full-attention layer shape must be positive, got {self:?}"
            )));
        }
        self.attention.validate()?;
        self.moe.validate()?;
        if self.attention.tokens != self.tokens {
            return Err(FerrumError::model(format!(
                "Qwen3.5 sparse-MoE full layer tokens {} must match attention tokens {}",
                self.tokens, self.attention.tokens
            )));
        }
        if self.moe.tokens != self.tokens || self.moe.hidden_size != self.hidden_size {
            return Err(FerrumError::model(format!(
                "Qwen3.5 sparse-MoE full layer shape tokens={} hidden={} must match MoE {:?}",
                self.tokens, self.hidden_size, self.moe
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

impl Qwen35DenseReferenceModel<'_> {
    fn validate(self, tokens: usize) -> Result<()> {
        if self.vocab_size == 0 || self.hidden_size == 0 || self.layers.is_empty() {
            return Err(FerrumError::model(format!(
                "Qwen3.5 dense reference model shape must be positive, got vocab={} hidden={} layers={}",
                self.vocab_size,
                self.hidden_size,
                self.layers.len()
            )));
        }
        validate_len(
            "dense reference embed_tokens",
            self.embed_tokens.len(),
            self.vocab_size * self.hidden_size,
        )?;
        validate_len(
            "dense reference final_norm_weight",
            self.final_norm_weight.len(),
            self.hidden_size,
        )?;
        validate_len(
            "dense reference lm_head_weight",
            self.lm_head_weight.len(),
            self.vocab_size * self.hidden_size,
        )?;
        for (idx, layer) in self.layers.iter().enumerate() {
            match layer {
                Qwen35DenseReferenceLayer::Linear(layer) => {
                    if layer.shape.tokens != tokens || layer.shape.hidden_size != self.hidden_size {
                        return Err(FerrumError::model(format!(
                            "Qwen3.5 dense reference layer {idx} linear shape {:?} does not match tokens={tokens} hidden={}",
                            layer.shape, self.hidden_size
                        )));
                    }
                    layer.shape.validate()?;
                }
                Qwen35DenseReferenceLayer::Full(layer) => {
                    if layer.shape.tokens != tokens || layer.shape.hidden_size != self.hidden_size {
                        return Err(FerrumError::model(format!(
                            "Qwen3.5 dense reference layer {idx} full shape {:?} does not match tokens={tokens} hidden={}",
                            layer.shape, self.hidden_size
                        )));
                    }
                    layer.shape.validate()?;
                }
            }
        }
        Ok(())
    }
}

impl Qwen35SparseMoeReferenceModel<'_> {
    fn validate(self, tokens: usize) -> Result<()> {
        if self.vocab_size == 0 || self.hidden_size == 0 || self.layers.is_empty() {
            return Err(FerrumError::model(format!(
                "Qwen3.5 sparse-MoE reference model shape must be positive, got vocab={} hidden={} layers={}",
                self.vocab_size,
                self.hidden_size,
                self.layers.len()
            )));
        }
        validate_len(
            "sparse-MoE reference embed_tokens",
            self.embed_tokens.len(),
            self.vocab_size * self.hidden_size,
        )?;
        validate_len(
            "sparse-MoE reference final_norm_weight",
            self.final_norm_weight.len(),
            self.hidden_size,
        )?;
        validate_len(
            "sparse-MoE reference lm_head_weight",
            self.lm_head_weight.len(),
            self.vocab_size * self.hidden_size,
        )?;
        for (idx, layer) in self.layers.iter().enumerate() {
            match layer {
                Qwen35SparseMoeReferenceLayer::Linear(layer) => {
                    if layer.shape.tokens != tokens || layer.shape.hidden_size != self.hidden_size {
                        return Err(FerrumError::model(format!(
                            "Qwen3.5 sparse-MoE reference layer {idx} linear shape {:?} does not match tokens={tokens} hidden={}",
                            layer.shape, self.hidden_size
                        )));
                    }
                    layer.shape.validate()?;
                }
                Qwen35SparseMoeReferenceLayer::Full(layer) => {
                    if layer.shape.tokens != tokens || layer.shape.hidden_size != self.hidden_size {
                        return Err(FerrumError::model(format!(
                            "Qwen3.5 sparse-MoE reference layer {idx} full shape {:?} does not match tokens={tokens} hidden={}",
                            layer.shape, self.hidden_size
                        )));
                    }
                    layer.shape.validate()?;
                }
            }
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
    validate_delta_rule_shape_values(shape)?;
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

fn validate_delta_rule_shape_values(shape: Qwen35DeltaRuleShape) -> Result<()> {
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
    if shape.value_heads % shape.key_heads != 0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 DeltaNet value_heads {} must be divisible by key_heads {}",
            shape.value_heads, shape.key_heads
        )));
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

fn gather_embedding_rows(
    embed_tokens: &[f32],
    input_ids: &[usize],
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<f32>> {
    validate_len(
        "embedding table",
        embed_tokens.len(),
        vocab_size * hidden_size,
    )?;
    let mut out = Vec::with_capacity(input_ids.len() * hidden_size);
    for (position, token_id) in input_ids.iter().copied().enumerate() {
        if token_id >= vocab_size {
            return Err(FerrumError::model(format!(
                "Qwen3.5 input token id {token_id} at position {position} exceeds vocab_size {vocab_size}"
            )));
        }
        let start = token_id * hidden_size;
        out.extend_from_slice(&embed_tokens[start..start + hidden_size]);
    }
    Ok(out)
}

fn qwen35_apply_rope_cpu(
    x: &mut [f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    position_offset: usize,
    rope_theta: f32,
    interleaved: bool,
) -> Result<()> {
    if tokens == 0 || heads == 0 || head_dim == 0 || rope_dim == 0 || rope_theta <= 0.0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 RoPE shape must be positive, got tokens={tokens} heads={heads} head_dim={head_dim} rope_dim={rope_dim} rope_theta={rope_theta}"
        )));
    }
    if rope_dim > head_dim || rope_dim % 2 != 0 {
        return Err(FerrumError::model(format!(
            "Qwen3.5 RoPE rope_dim {rope_dim} must be even and <= head_dim {head_dim}"
        )));
    }
    validate_len("RoPE input", x.len(), tokens * heads * head_dim)?;

    let half = rope_dim / 2;
    for token in 0..tokens {
        let position = (position_offset + token) as f32;
        for pair in 0..half {
            let inv_freq = rope_theta.powf(-(2.0 * pair as f32) / rope_dim as f32);
            let angle = position * inv_freq;
            let (sin, cos) = angle.sin_cos();
            for head in 0..heads {
                let head_base = (token * heads + head) * head_dim;
                if interleaved {
                    let base = head_base + 2 * pair;
                    let x0 = x[base];
                    let x1 = x[base + 1];
                    x[base] = x0 * cos - x1 * sin;
                    x[base + 1] = x1 * cos + x0 * sin;
                } else {
                    let base = head_base + pair;
                    let x0 = x[base];
                    let x1 = x[base + half];
                    x[base] = x0 * cos - x1 * sin;
                    x[base + half] = x0 * sin + x1 * cos;
                }
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

fn split_qwen35_full_attention_query_gate(
    query_raw: &[f32],
    tokens: usize,
    num_heads: usize,
    head_dim: usize,
    q_proj_total: usize,
    attn_output_gate: bool,
) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    let q_total = num_heads * head_dim;
    if !attn_output_gate {
        return Ok((
            split_features(query_raw, tokens, q_proj_total, 0, q_total),
            None,
        ));
    }
    if q_proj_total < num_heads * 2 * head_dim {
        return Err(FerrumError::model(format!(
            "Qwen3.5 gated q_proj width {q_proj_total} is too small for num_heads={num_heads} head_dim={head_dim}"
        )));
    }
    validate_len(
        "full attention query_raw",
        query_raw.len(),
        tokens * q_proj_total,
    )?;

    let mut query = vec![0.0; tokens * q_total];
    let mut gate = vec![0.0; tokens * q_total];
    for token in 0..tokens {
        let row_base = token * q_proj_total;
        let out_base = token * q_total;
        for head in 0..num_heads {
            let src_head = row_base + head * 2 * head_dim;
            let dst_head = out_base + head * head_dim;
            query[dst_head..dst_head + head_dim]
                .copy_from_slice(&query_raw[src_head..src_head + head_dim]);
            gate[dst_head..dst_head + head_dim]
                .copy_from_slice(&query_raw[src_head + head_dim..src_head + 2 * head_dim]);
        }
    }
    Ok((query, Some(gate)))
}

fn token_major_to_head_major_f32(
    values: &[f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; tokens * heads * head_dim];
    for token in 0..tokens {
        for head in 0..heads {
            let src = (token * heads + head) * head_dim;
            let dst = (head * tokens + token) * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&values[src..src + head_dim]);
        }
    }
    out
}

fn qwen35_load_sparse_moe_experts<B: MoeLlmBackend>(
    planned: &Qwen35WeightPlanLoader<'_, B>,
    layer_index: usize,
    config: &Qwen35TextConfig,
) -> Result<(ExpertStack<B>, B::Buffer, B::Buffer)> {
    let moe_config = config
        .moe
        .as_ref()
        .ok_or_else(|| FerrumError::model("Qwen3.5 sparse MoE config missing"))?;
    let num_experts = moe_config.num_experts;
    let hidden = config.hidden_size;
    let expert_intermediate = moe_config.moe_intermediate_size;

    if planned.has_layer_tensor(layer_index, "moe_per_expert_gate_proj_qweight") {
        let (gate_up_marlin, gate_up_n_per_expert, _) = planned.load_layer_stacked_gptq_experts(
            layer_index,
            num_experts,
            &["gate_proj", "up_proj"],
        )?;
        let (down_marlin, down_n_per_expert, _) =
            planned.load_layer_stacked_gptq_experts(layer_index, num_experts, &["down_proj"])?;

        let mut gate_up: Vec<Box<dyn Linear<B>>> = Vec::with_capacity(num_experts);
        let mut down: Vec<Box<dyn Linear<B>>> = Vec::with_capacity(num_experts);
        for expert in 0..num_experts {
            gate_up.push(Box::new(StackedExpertLinear::<B>::new(
                gate_up_marlin.clone(),
                expert * gate_up_n_per_expert,
                gate_up_n_per_expert,
            )?));
            down.push(Box::new(StackedExpertLinear::<B>::new(
                down_marlin.clone(),
                expert * down_n_per_expert,
                down_n_per_expert,
            )?));
        }
        let experts = ExpertStack {
            gate_up,
            down,
            gate_stacked: None,
            up_stacked: None,
            down_stacked: None,
            gate_up_marlin_stack: Some(gate_up_marlin),
            down_marlin_stack: Some(down_marlin),
        };
        return Ok((experts, B::from_slice(&[0.0]), B::from_slice(&[0.0])));
    }

    if planned.has_layer_tensor(layer_index, "moe_fused_gate_up_proj")
        && planned.has_layer_tensor(layer_index, "moe_fused_down_proj")
    {
        let fused_gate_up_proj =
            planned.load_layer_tensor(layer_index, "moe_fused_gate_up_proj")?;
        let fused_down_proj = planned.load_layer_tensor(layer_index, "moe_fused_down_proj")?;
        let fused_gate_up_host = B::to_vec(
            &fused_gate_up_proj,
            num_experts * 2 * expert_intermediate * hidden,
        );
        let fused_down_host =
            B::to_vec(&fused_down_proj, num_experts * hidden * expert_intermediate);
        let per_gate = expert_intermediate * hidden;
        let per_fused = 2 * per_gate;
        let mut gate_stack = Vec::with_capacity(num_experts * per_gate);
        let mut up_stack = Vec::with_capacity(num_experts * per_gate);
        for expert in 0..num_experts {
            let base = expert * per_fused;
            gate_stack.extend_from_slice(&fused_gate_up_host[base..base + per_gate]);
            up_stack.extend_from_slice(&fused_gate_up_host[base + per_gate..base + per_fused]);
        }
        let experts = ExpertStack::from_dense_stacks(
            &gate_stack,
            &up_stack,
            &fused_down_host,
            num_experts,
            hidden,
            expert_intermediate,
        )?;
        return Ok((experts, fused_gate_up_proj, fused_down_proj));
    }

    Err(FerrumError::model(format!(
        "Qwen3.5 sparse MoE layer {layer_index} has neither official per-expert GPTQ qweight \
         tensors nor fused dense expert tensors"
    )))
}

impl<B: MoeLlmBackend> Qwen35ModelWeights<B> {
    pub fn load(
        config: Qwen35TextConfig,
        runtime_cfg: LlmRuntimeConfig,
        plan: &Qwen35ResolvedWeightPlan,
        loader: &dyn ferrum_quantization::WeightLoader<B>,
    ) -> Result<Self> {
        let planned = Qwen35WeightPlanLoader::<B>::new(plan, loader);
        let hidden_size = config.hidden_size;
        let head_dim = config.head_dim;
        let embed_tokens = planned.load_global_tensor("embed_tokens")?;
        let final_norm = qwen35_fold_gemma_norm_weight::<B>(
            planned.load_global_tensor("final_norm")?,
            hidden_size,
        );
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
                input_layernorm: qwen35_fold_gemma_norm_weight::<B>(
                    planned.load_layer_tensor(layer_plan.layer_index, "input_layernorm")?,
                    hidden_size,
                ),
                post_attention_layernorm: qwen35_fold_gemma_norm_weight::<B>(
                    planned
                        .load_layer_tensor(layer_plan.layer_index, "post_attention_layernorm")?,
                    hidden_size,
                ),
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
                            qkvz_proj: if B::supports_qwen35_packed_gdn_decode_prepare() {
                                Some(
                                    planned
                                        .load_layer_linear_attention_qkvz(layer_plan.layer_index)?,
                                )
                            } else {
                                None
                            },
                            ba_proj: if B::supports_qwen35_packed_gdn_decode_prepare() {
                                Some(
                                    planned
                                        .load_layer_linear_attention_ba(layer_plan.layer_index)?,
                                )
                            } else {
                                None
                            },
                            conv1d_weight: planned
                                .load_layer_tensor(layer_plan.layer_index, "linear_attn_conv")?,
                            a_log: qwen35_promote_to_f32_buffer::<B>(
                                planned.load_layer_tensor(
                                    layer_plan.layer_index,
                                    "linear_attn_a_log",
                                )?,
                                config.linear_attention.num_value_heads,
                            ),
                            dt_bias: qwen35_promote_to_f32_buffer::<B>(
                                planned.load_layer_tensor(
                                    layer_plan.layer_index,
                                    "linear_attn_dt_bias",
                                )?,
                                config.linear_attention.num_value_heads,
                            ),
                            norm_weight: qwen35_promote_to_f32_buffer::<B>(
                                planned.load_layer_tensor(
                                    layer_plan.layer_index,
                                    "linear_attn_norm",
                                )?,
                                config.linear_attention.value_head_dim,
                            ),
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
                            q_norm_weight: qwen35_fold_gemma_norm_weight::<B>(
                                planned.load_layer_tensor(
                                    layer_plan.layer_index,
                                    "self_attn_q_norm",
                                )?,
                                head_dim,
                            ),
                            k_norm_weight: qwen35_fold_gemma_norm_weight::<B>(
                                planned.load_layer_tensor(
                                    layer_plan.layer_index,
                                    "self_attn_k_norm",
                                )?,
                                head_dim,
                            ),
                        })
                    }
                },
                mlp: match layer_plan.mlp {
                    Qwen35MlpKind::Dense => Qwen35MlpWeights::Dense(Qwen35DenseMlpWeights {
                        gate_up_proj: planned
                            .load_layer_dense_gate_up_linear(layer_plan.layer_index)?,
                        down_proj: planned.load_layer_linear(layer_plan.layer_index, "mlp_down")?,
                    }),
                    Qwen35MlpKind::SparseMoeSharedExpert => {
                        let (experts, fused_gate_up_proj, fused_down_proj) =
                            qwen35_load_sparse_moe_experts::<B>(
                                &planned,
                                layer_plan.layer_index,
                                &config,
                            )?;
                        Qwen35MlpWeights::SparseMoeSharedExpert(
                            Qwen35SparseMoeSharedExpertWeights {
                                router: planned
                                    .load_layer_linear(layer_plan.layer_index, "moe_router")?,
                                experts,
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
                                fused_gate_up_proj,
                                fused_down_proj,
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

fn qwen35_fold_gemma_norm_weight<B: Backend>(buf: B::Buffer, len: usize) -> B::Buffer {
    let mut values = B::to_vec(&buf, len);
    for value in &mut values {
        *value += 1.0;
    }
    B::from_slice(&values)
}

fn qwen35_promote_to_f32_buffer<B: Backend>(buf: B::Buffer, len: usize) -> B::Buffer {
    let values = B::to_vec(&buf, len);
    B::from_slice_typed::<f32>(&values)
}

impl<B: MoeLlmBackend> Qwen35AttentionWeights<B> {
    pub fn kind(&self) -> Qwen35LayerType {
        match self {
            Self::Linear(_) => Qwen35LayerType::LinearAttention,
            Self::Full(_) => Qwen35LayerType::FullAttention,
        }
    }
}

impl<B: MoeLlmBackend> Qwen35MlpWeights<B> {
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

    use ferrum_kernels::backend::{cpu::CpuBackend, Backend};
    use ferrum_quantization::{DenseLinear, QuantConfig, WeightLoader};
    use ferrum_types::{
        FerrumError, Result as FerrumResult, RuntimeConfigEntry, RuntimeConfigSnapshot,
        RuntimeConfigSource,
    };

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

    #[test]
    fn qwen35_text_rope_uses_neox_when_mrope_section_is_present() {
        let cfg = Qwen35TextConfig::from_hf_config_str(
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
                "tie_word_embeddings": false,
                "rope_parameters": {
                  "rope_theta": 10000000,
                  "partial_rotary_factor": 1.0,
                  "mrope_interleaved": true,
                  "mrope_section": [1, 1]
                }
              }
            }"#,
        )
        .unwrap();

        assert_eq!(cfg.rope_parameters.mrope_section, Some(vec![1, 1]));
        assert!(!cfg.full_attention_text_rope_interleaved());
    }

    struct RecordingLoader {
        tensors: HashMap<String, Vec<f32>>,
        linears: Mutex<Vec<String>>,
    }

    impl RecordingLoader {
        fn from_required_manifest(config: &Qwen35TextConfig) -> Self {
            let manifest = config.weight_manifest("model").unwrap();
            let mut tensors: HashMap<String, Vec<f32>> = manifest
                .global_tensors
                .iter()
                .chain(
                    manifest
                        .layers
                        .iter()
                        .flat_map(|layer| layer.tensors.iter()),
                )
                .filter(|tensor| tensor.required)
                .map(|tensor| {
                    let len = match tensor.role.as_str() {
                        "final_norm" | "input_layernorm" | "post_attention_layernorm" => {
                            config.hidden_size
                        }
                        "linear_attn_a_log" | "linear_attn_dt_bias" => {
                            config.linear_attention.num_value_heads
                        }
                        "linear_attn_norm" => config.linear_attention.value_head_dim,
                        "self_attn_q_norm" | "self_attn_k_norm" => config.head_dim,
                        _ => 1,
                    };
                    (tensor.name.clone(), vec![1.0; len])
                })
                .collect();
            if let Some(moe) = &config.moe {
                let gate_up_len =
                    moe.num_experts * 2 * moe.moe_intermediate_size * config.hidden_size;
                let down_len = moe.num_experts * config.hidden_size * moe.moe_intermediate_size;
                for layer in config.sparse_moe_layers() {
                    tensors.insert(
                        format!("model.layers.{layer}.mlp.experts.gate_up_proj"),
                        vec![1.0; gate_up_len],
                    );
                    tensors.insert(
                        format!("model.layers.{layer}.mlp.experts.down_proj"),
                        vec![1.0; down_len],
                    );
                }
            }
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
            if let Some(prefix) = name.strip_suffix("gate_up_proj") {
                let gate = format!("{prefix}gate_proj.weight");
                let up = format!("{prefix}up_proj.weight");
                if self.tensors.contains_key(&gate) && self.tensors.contains_key(&up) {
                    self.linears.lock().unwrap().push(name.to_string());
                    return Ok(Box::new(DenseLinear::<CpuBackend>::from_rows(
                        &[0.0, 0.0, 0.0, 0.0],
                        2,
                        2,
                    )));
                }
            }
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

    fn runtime_snapshot(vars: &[(&str, &str)]) -> RuntimeConfigSnapshot {
        RuntimeConfigSnapshot::from_entries(
            vars.iter().map(|(key, value)| {
                RuntimeConfigEntry::new(*key, *value, RuntimeConfigSource::Cli)
            }),
        )
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

    fn token_major_to_head_major(
        values: &[f32],
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0; values.len()];
        for token in 0..tokens {
            for head in 0..heads {
                let src = (token * heads + head) * head_dim;
                let dst = (head * tokens + token) * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&values[src..src + head_dim]);
            }
        }
        out
    }

    struct SmallMoeWeights {
        router_weight: Vec<f32>,
        fused_gate_up_proj: Vec<f32>,
        fused_down_proj: Vec<f32>,
        shared_expert_gate_weight: Vec<f32>,
        shared_expert_gate_proj_weight: Vec<f32>,
        shared_expert_up_proj_weight: Vec<f32>,
        shared_expert_down_proj_weight: Vec<f32>,
    }

    fn single_expert_moe_shape(tokens: usize, hidden_size: usize) -> Qwen35SparseMoeShape {
        Qwen35SparseMoeShape {
            tokens,
            hidden_size,
            num_experts: 1,
            top_k: 1,
            expert_intermediate_size: 2,
            shared_expert_intermediate_size: 2,
            norm_topk_prob: false,
        }
    }

    fn small_moe_weights() -> SmallMoeWeights {
        SmallMoeWeights {
            router_weight: vec![0.1, -0.2],
            fused_gate_up_proj: vec![
                0.2, 0.1, //
                -0.1, 0.3, //
                0.4, -0.2, //
                0.3, 0.5,
            ],
            fused_down_proj: vec![
                0.5, 0.25, //
                -0.2, 0.75,
            ],
            shared_expert_gate_weight: vec![0.25, -0.1],
            shared_expert_gate_proj_weight: vec![
                -0.2, 0.2, //
                0.1, 0.3,
            ],
            shared_expert_up_proj_weight: vec![
                0.25, 0.5, //
                -0.3, 0.4,
            ],
            shared_expert_down_proj_weight: vec![
                0.5, 0.25, //
                -0.2, 0.75,
            ],
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
        assert_eq!(model.final_norm, vec![2.0; config.hidden_size]);
        assert_eq!(model.layers.len(), 4);
        assert_eq!(
            model.layers[0].input_layernorm,
            vec![2.0; config.hidden_size]
        );
        assert_eq!(
            model.layers[0].post_attention_layernorm,
            vec![2.0; config.hidden_size]
        );
        assert_eq!(
            model.layers[0].attention.kind(),
            Qwen35LayerType::LinearAttention
        );
        assert_eq!(model.layers[0].mlp.kind(), Qwen35MlpKind::Dense);
        assert_eq!(
            model.layers[3].attention.kind(),
            Qwen35LayerType::FullAttention
        );
        assert!(
            matches!(&model.layers[3].attention, Qwen35AttentionWeights::Full(weights)
                if weights.q_norm_weight == vec![2.0; config.head_dim]
                    && weights.k_norm_weight == vec![2.0; config.head_dim])
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
                if weights.fused_gate_up_proj.len() == 8 * 2 * 8 * 16
                    && weights.fused_gate_up_proj.iter().all(|value| *value == 1.0)
        ));
    }

    #[test]
    fn qwen35_backend_model_materializes_from_definition_and_weight_plan() {
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
        let config = Qwen35TextConfig::from_model_definition(&def).unwrap();
        let loader = RecordingLoader::from_required_manifest(&config);
        let plan = loader.plan(&config);

        let model =
            Qwen35BackendModel::<CpuBackend>::from_definition_with_loader(&def, plan, &loader)
                .unwrap();

        assert_eq!(model.qwen35_config().num_hidden_layers, 4);
        assert_eq!(model.runtime_config().hidden_size, 16);
        assert_eq!(model.runtime_config().vocab_size, 32000);
        assert!(model.weight_validation.is_pass());
        assert_eq!(model.weight_plan.prefix, "model");
        assert!(loader
            .linears()
            .contains(&"model.layers.0.linear_attn.in_proj_qkv".to_string()));
    }

    #[test]
    fn qwen35_unified_forward_requires_paged_kv_for_fresh_batch_prefill() {
        let config = dense_config();
        let loader = RecordingLoader::from_required_manifest(&config);
        let plan = loader.plan(&config);
        let mut model = Qwen35BackendModel::<CpuBackend>::from_weight_plan(
            config.clone(),
            runtime_config(&config),
            plan,
            &loader,
        )
        .unwrap();

        let err = <Qwen35BackendModel<CpuBackend> as DecoderOnlyLLM>::unified_forward(
            &mut model,
            &[
                ("req-a".to_string(), vec![1, 2, 3], 0, true),
                ("req-b".to_string(), vec![4, 5], 0, true),
            ],
        )
        .expect_err("CPU/non-paged Qwen3.5 unified prefill must fall back through Unsupported");

        assert!(
            err.to_string()
                .contains("Qwen3.5 unified_forward fresh prefill requires paged KV"),
            "{err}"
        );
    }

    #[test]
    fn qwen35_backend_model_rejects_incomplete_weight_plan() {
        let config = dense_config();
        let loader = RecordingLoader::from_required_manifest(&config);
        let mut plan = loader.plan(&config);
        plan.layers[0].tensors[0].present = false;

        let err = match Qwen35BackendModel::<CpuBackend>::from_weight_plan(
            config.clone(),
            runtime_config(&config),
            plan,
            &loader,
        ) {
            Ok(_) => panic!("incomplete backend weight plan must fail"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("weight plan is incomplete for prefix model"),
            "{err}"
        );
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
        assert_eq!(runtime.max_seq_len, QWEN35_DEFAULT_KV_CAPACITY);
    }

    #[test]
    fn qwen35_effective_max_seq_len_honors_max_model_len_snapshot() {
        let snapshot = runtime_snapshot(&[("FERRUM_MAX_MODEL_LEN", "4096")]);

        assert_eq!(
            qwen35_effective_max_seq_len_from_snapshot(&snapshot, 262_144),
            4096
        );
    }

    #[test]
    fn qwen35_effective_max_seq_len_prefers_kv_capacity() {
        let snapshot = runtime_snapshot(&[
            ("FERRUM_KV_CAPACITY", "2048"),
            ("FERRUM_MAX_MODEL_LEN", "4096"),
        ]);

        assert_eq!(
            qwen35_effective_max_seq_len_from_snapshot(&snapshot, 262_144),
            2048
        );
    }

    #[test]
    fn qwen35_effective_max_seq_len_clamps_to_model_context() {
        let snapshot = runtime_snapshot(&[("FERRUM_MAX_MODEL_LEN", "8192")]);

        assert_eq!(
            qwen35_effective_max_seq_len_from_snapshot(&snapshot, 1024),
            1024
        );
    }

    #[test]
    fn qwen35_effective_max_seq_len_uses_safe_default_for_invalid_or_absent_snapshot() {
        let invalid = runtime_snapshot(&[("FERRUM_KV_CAPACITY", "bad")]);

        assert_eq!(
            qwen35_effective_max_seq_len_from_snapshot(&invalid, 262_144),
            QWEN35_DEFAULT_KV_CAPACITY
        );
        assert_eq!(
            qwen35_effective_max_seq_len_from_snapshot(&RuntimeConfigSnapshot::default(), 256),
            256
        );
    }

    #[test]
    fn qwen35_runtime_bool_reads_typed_snapshot() {
        let enabled = runtime_snapshot(&[("FERRUM_GREEDY_ARGMAX", "1")]);
        let disabled = runtime_snapshot(&[("FERRUM_GREEDY_ARGMAX", "0")]);
        let invalid = runtime_snapshot(&[("FERRUM_GREEDY_ARGMAX", "maybe")]);

        assert_eq!(
            qwen35_runtime_bool(&enabled, "FERRUM_GREEDY_ARGMAX"),
            Some(true)
        );
        assert_eq!(
            qwen35_runtime_bool(&disabled, "FERRUM_GREEDY_ARGMAX"),
            Some(false)
        );
        assert_eq!(qwen35_runtime_bool(&invalid, "FERRUM_GREEDY_ARGMAX"), None);
    }

    #[test]
    fn qwen35_decode_logits_policy_uses_greedy_only_for_consistent_masks() {
        let mask_a = ferrum_interfaces::model_executor::TokenSelectionMask::new(vec![1, 0, 1]);
        let mask_a_same = ferrum_interfaces::model_executor::TokenSelectionMask::new(vec![1, 0, 1]);
        let mask_b = ferrum_interfaces::model_executor::TokenSelectionMask::new(vec![1, 1, 0]);

        let consistent = [
            LogitsReturnPolicy::GreedyArgmax {
                token_mask: Some(mask_a.clone()),
                repetition_penalty: None,
            },
            LogitsReturnPolicy::GreedyArgmax {
                token_mask: Some(mask_a_same),
                repetition_penalty: None,
            },
        ];
        assert!(matches!(
            qwen35_decode_logits_return_from_policies(&consistent, 2),
            Qwen35DecodeLogitsReturn::GreedyArgmax {
                token_mask: Some(_),
                ..
            }
        ));

        let mixed_mask = [
            LogitsReturnPolicy::GreedyArgmax {
                token_mask: Some(mask_a.clone()),
                repetition_penalty: None,
            },
            LogitsReturnPolicy::GreedyArgmax {
                token_mask: None,
                repetition_penalty: None,
            },
        ];
        assert!(matches!(
            qwen35_decode_logits_return_from_policies(&mixed_mask, 2),
            Qwen35DecodeLogitsReturn::Full
        ));

        let different_masks = [
            LogitsReturnPolicy::GreedyArgmax {
                token_mask: Some(mask_a),
                repetition_penalty: None,
            },
            LogitsReturnPolicy::GreedyArgmax {
                token_mask: Some(mask_b),
                repetition_penalty: None,
            },
        ];
        assert!(matches!(
            qwen35_decode_logits_return_from_policies(&different_masks, 2),
            Qwen35DecodeLogitsReturn::Full
        ));

        let penalty = GreedyRepetitionPenalty::new(1.1, vec![10, 11]);
        let repeated = [
            LogitsReturnPolicy::GreedyArgmax {
                token_mask: None,
                repetition_penalty: Some(penalty),
            },
            LogitsReturnPolicy::GreedyArgmax {
                token_mask: None,
                repetition_penalty: None,
            },
        ];
        let Qwen35DecodeLogitsReturn::GreedyArgmax {
            repetition_penalties,
            ..
        } = qwen35_decode_logits_return_from_policies(&repeated, 2)
        else {
            panic!("Qwen3.5 should keep per-row repetition penalty on greedy argmax path");
        };
        assert!(repetition_penalties[0].is_some());
        assert!(repetition_penalties[1].is_none());
    }

    #[test]
    fn allocates_recurrent_state_cache_from_spec() {
        let config = dense_config();
        let request_id = RequestId::new();
        let spec = config
            .to_recurrent_state_spec(request_id.clone(), DataType::BF16, Device::CPU, 2)
            .unwrap();

        let cache = Qwen35RecurrentStateCache::<CpuBackend>::from_spec(&spec).unwrap();
        let first_conv = cache
            .tensor(0, crate::qwen35_config::QWEN35_CONV_STATE_NAME)
            .unwrap();
        let first_delta = cache
            .tensor(0, crate::qwen35_config::QWEN35_DELTA_STATE_NAME)
            .unwrap();
        let second_conv_slot = first_conv.slot_range(1, cache.max_batch_slots).unwrap();
        let second_delta_slot = first_delta.slot_range(1, cache.max_batch_slots).unwrap();

        assert_eq!(cache.request_id, request_id);
        assert_eq!(cache.num_layers, 4);
        assert_eq!(cache.dtype, DataType::BF16);
        assert_eq!(cache.device, Device::CPU);
        assert_eq!(cache.max_batch_slots, 2);
        assert_eq!(cache.tensors.len(), 6);
        assert_eq!(first_conv.shape, vec![24, 3]);
        assert_eq!(first_conv.elements_per_slot, 72);
        assert_eq!(first_delta.shape, vec![2, 4, 4]);
        assert_eq!(first_delta.elements_per_slot, 32);
        assert_eq!(second_conv_slot, 72..144);
        assert_eq!(second_delta_slot, 32..64);
        assert_eq!(cache.total_elements(), 3 * 2 * (72 + 32));
        assert_eq!(cache.estimated_memory_bytes(), 3 * 2 * (72 + 32) * 2);
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
    fn rejects_unsupported_recurrent_state_dtype() {
        let config = dense_config();
        let spec = config
            .to_recurrent_state_spec(RequestId::new(), DataType::FP8, Device::CPU, 1)
            .unwrap();

        let err = match Qwen35RecurrentStateCache::<CpuBackend>::from_spec(&spec) {
            Ok(_) => panic!("FP8 recurrent state storage is not implemented"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("dtype FP8"), "{err}");
    }

    #[test]
    fn recurrent_state_handle_downcasts_and_clones_shared_cache() {
        let config = dense_config();
        let request_id = RequestId::new();
        let spec = config
            .to_recurrent_state_spec(request_id.clone(), DataType::FP16, Device::CPU, 1)
            .unwrap();
        let handle = Qwen35RecurrentStateHandle::<CpuBackend>::from_spec(&spec).unwrap();

        let cloned = handle.clone_handle().unwrap();
        let typed = cloned
            .as_any()
            .downcast_ref::<Qwen35RecurrentStateHandle<CpuBackend>>()
            .expect("clone_handle should preserve the Qwen35 typed handle");
        let stats = typed.stats();

        assert_eq!(typed.request_id(), request_id);
        assert_eq!(typed.num_layers(), 4);
        assert_eq!(typed.device(), Device::CPU);
        assert_eq!(typed.cache_id(), handle.cache_id());
        assert_eq!(stats.state_tensors, 6);
        assert_eq!(stats.batch_slots, 1);
        assert_eq!(stats.memory_bytes, 3 * (72 + 32) * 2);
        let cache = typed.cache();
        assert_eq!(cache.tensors.len(), 6);
        assert!(typed.is_valid());
    }

    #[test]
    fn recurrent_state_manager_allocates_rejects_capacity_and_invalidates() {
        let config = dense_config();
        let manager =
            Qwen35RecurrentStateManager::<CpuBackend>::new(Qwen35RecurrentStateManagerConfig {
                total_memory_bytes: 3 * (72 + 32) * 2,
                total_batch_slots: 1,
            });
        let request_id = RequestId::new();
        let spec = config
            .to_recurrent_state_spec(request_id.clone(), DataType::FP16, Device::CPU, 1)
            .unwrap();
        let second_spec = config
            .to_recurrent_state_spec(RequestId::new(), DataType::FP16, Device::CPU, 1)
            .unwrap();

        assert!(manager.can_allocate(&spec));
        let handle = tokio_test::block_on(manager.allocate(&spec)).unwrap();
        let typed = handle
            .as_any()
            .downcast_ref::<Qwen35RecurrentStateHandle<CpuBackend>>()
            .expect("manager should allocate a Qwen35 typed recurrent-state handle");

        assert_eq!(typed.cache().tensors.len(), 6);
        assert_eq!(manager.stats().active_states, 1);
        assert_eq!(manager.stats().active_state_tensors, 6);
        assert_eq!(manager.stats().used_memory_bytes, 3 * (72 + 32) * 2);
        assert!(!manager.can_allocate(&spec));
        assert!(!manager.can_allocate(&second_spec));

        let duplicate = tokio_test::block_on(manager.allocate(&spec)).unwrap_err();
        assert!(
            duplicate.to_string().contains("already allocated"),
            "{duplicate}"
        );
        let overcommit = tokio_test::block_on(manager.allocate(&second_spec)).unwrap_err();
        assert!(
            overcommit
                .to_string()
                .contains("insufficient Qwen3.5 recurrent-state capacity"),
            "{overcommit}"
        );
        assert_eq!(manager.stats().allocation_failures, 2);

        tokio_test::block_on(manager.deallocate(request_id.clone())).unwrap();
        assert!(!handle.is_valid());
        assert!(manager.get_handle(request_id).is_none());
        assert_eq!(manager.stats().active_states, 0);

        let replacement = tokio_test::block_on(manager.allocate(&second_spec)).unwrap();
        assert!(replacement.is_valid());
        tokio_test::block_on(manager.reset()).unwrap();
        assert!(!replacement.is_valid());
        assert_eq!(manager.stats().active_states, 0);
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
    fn full_attention_qk_norm_uses_vllm_gemma_plus_one_semantics() {
        let shape = Qwen35FullAttentionShape {
            tokens: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            rope_dim: 2,
            position_offset: 0,
            rope_theta: 10_000.0,
            rope_interleaved: false,
            attn_output_gate: false,
        };
        let reference = qwen35_full_attention_core_cpu(
            &[3.0, 4.0],
            &[5.0, 12.0],
            &[7.0, 11.0],
            &[0.0, 1.0],
            &[0.5, 1.5],
            shape,
            0.0,
        )
        .unwrap();
        let q_inv = ((3.0f32 * 3.0 + 4.0 * 4.0) / 2.0).sqrt().recip();
        let k_inv = ((5.0f32 * 5.0 + 12.0 * 12.0) / 2.0).sqrt().recip();

        assert_close_slice(&reference.query, &[3.0 * q_inv, 4.0 * q_inv * 2.0], 1e-6);
        assert_close_slice(
            &reference.key,
            &[5.0 * k_inv * 1.5, 12.0 * k_inv * 2.5],
            1e-6,
        );
        assert_close_slice(&reference.value, &[7.0, 11.0], 1e-6);
    }

    #[test]
    fn rope_uses_non_interleaved_half_rotation() {
        let mut values = vec![1.0, 0.0];
        qwen35_apply_rope_cpu(&mut values, 1, 1, 2, 2, 1, 10_000.0, false).unwrap();
        let (sin, cos) = 1.0f32.sin_cos();

        assert_close_slice(&values, &[cos, sin], 1e-6);
    }

    #[test]
    fn rope_uses_partial_interleaved_rotation() {
        let mut values = vec![1.0, 0.0, 3.0, 4.0];
        qwen35_apply_rope_cpu(&mut values, 1, 1, 4, 2, 1, 10_000.0, true).unwrap();
        let (sin, cos) = 1.0f32.sin_cos();

        assert_close_slice(&values, &[cos, sin, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn full_attention_core_applies_causal_softmax() {
        let shape = Qwen35FullAttentionShape {
            tokens: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            rope_dim: 2,
            position_offset: 0,
            rope_theta: 10_000.0,
            rope_interleaved: false,
            attn_output_gate: false,
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
    fn full_attention_core_applies_qwen35_output_gate() {
        let gates = vec![0.0, 1.0, -1.0, 2.0];
        let query_raw = vec![0.0, 0.0, gates[0], gates[1], 0.0, 0.0, gates[2], gates[3]];
        let shape = Qwen35FullAttentionShape {
            tokens: 1,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            rope_dim: 2,
            position_offset: 0,
            rope_theta: 10_000.0,
            rope_interleaved: true,
            attn_output_gate: true,
        };
        let reference = qwen35_full_attention_core_cpu(
            &query_raw,
            &[0.0; 2],
            &[2.0, 4.0],
            &[0.0; 2],
            &[0.0; 2],
            shape,
            1e-6,
        )
        .unwrap();
        let expected = vec![
            2.0 * sigmoid(gates[0]),
            4.0 * sigmoid(gates[1]),
            2.0 * sigmoid(gates[2]),
            4.0 * sigmoid(gates[3]),
        ];

        assert_close_slice(&reference.query, &[0.0, 0.0, 0.0, 0.0], 1e-6);
        assert_close_slice(&reference.context_ungated, &[2.0, 4.0, 2.0, 4.0], 1e-6);
        assert_close_slice(&reference.context, &expected, 1e-6);
        assert_eq!(reference.attention_gate.as_deref(), Some(gates.as_slice()));
    }

    #[test]
    fn full_attention_backend_core_matches_reference() {
        let raw_shape = Qwen35FullAttentionShape {
            tokens: 3,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            rope_dim: 2,
            position_offset: 1,
            rope_theta: 10_000.0,
            rope_interleaved: false,
            attn_output_gate: false,
        };
        let q_raw: Vec<f32> = (0..raw_shape.q_len())
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.2)
            .collect();
        let k_raw: Vec<f32> = (0..raw_shape.kv_len())
            .map(|i| ((i as f32 % 5.0) - 2.0) * 0.15)
            .collect();
        let v_raw: Vec<f32> = (0..raw_shape.kv_len())
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.1)
            .collect();
        let q_norm_raw = vec![0.0, 0.5];
        let k_norm_raw = vec![0.25, -0.5];
        let q_norm_folded = q_norm_raw
            .iter()
            .map(|value| 1.0 + value)
            .collect::<Vec<_>>();
        let k_norm_folded = k_norm_raw
            .iter()
            .map(|value| 1.0 + value)
            .collect::<Vec<_>>();
        let eps = 1e-6;
        let reference = qwen35_full_attention_core_cpu(
            &q_raw,
            &k_raw,
            &v_raw,
            &q_norm_raw,
            &k_norm_raw,
            raw_shape,
            eps,
        )
        .unwrap();

        let rope = qwen35_build_rope_cache_backend::<CpuBackend>(
            raw_shape.position_offset + raw_shape.tokens,
            raw_shape.rope_dim,
            raw_shape.rope_theta,
        )
        .unwrap();
        let mut ctx = CpuBackend::new_context();
        let output = qwen35_full_attention_core_backend::<CpuBackend>(
            &mut ctx,
            &CpuBackend::from_slice_typed(&q_raw),
            &CpuBackend::from_slice_typed(&k_raw),
            &CpuBackend::from_slice_typed(&v_raw),
            &CpuBackend::from_slice_typed(&q_norm_folded),
            &CpuBackend::from_slice_typed(&k_norm_folded),
            &rope.cos,
            &rope.sin,
            raw_shape,
            eps,
        )
        .unwrap();

        assert_close_slice(
            &CpuBackend::to_vec(&output.query_head_major, raw_shape.q_len()),
            &token_major_to_head_major(
                &reference.query,
                raw_shape.tokens,
                raw_shape.num_heads,
                raw_shape.head_dim,
            ),
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.key_head_major, raw_shape.kv_len()),
            &token_major_to_head_major(
                &reference.key,
                raw_shape.tokens,
                raw_shape.num_kv_heads,
                raw_shape.head_dim,
            ),
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.value_head_major, raw_shape.kv_len()),
            &token_major_to_head_major(
                &reference.value,
                raw_shape.tokens,
                raw_shape.num_kv_heads,
                raw_shape.head_dim,
            ),
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.context_head_major, raw_shape.q_len()),
            &token_major_to_head_major(
                &reference.context,
                raw_shape.tokens,
                raw_shape.num_heads,
                raw_shape.head_dim,
            ),
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.context, raw_shape.q_len()),
            &reference.context,
            1e-6,
        );
    }

    #[test]
    fn full_attention_core_rejects_invalid_gqa_shape() {
        let shape = Qwen35FullAttentionShape {
            tokens: 1,
            num_heads: 3,
            num_kv_heads: 2,
            head_dim: 2,
            rope_dim: 2,
            position_offset: 0,
            rope_theta: 10_000.0,
            rope_interleaved: false,
            attn_output_gate: false,
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
    fn dense_mlp_backend_matches_reference_with_fused_gate_up() {
        let tokens = 2;
        let hidden_size = 2;
        let intermediate_size = 2;
        let x = vec![
            1.0, 2.0, //
            -1.0, 0.5,
        ];
        let gate_weight = vec![
            0.2, 0.1, //
            -0.1, 0.3,
        ];
        let up_weight = vec![
            0.4, -0.2, //
            0.3, 0.5,
        ];
        let down_weight = vec![
            1.0, -0.25, //
            0.5, 0.75,
        ];
        let mut gate_up_weight = gate_weight.clone();
        gate_up_weight.extend_from_slice(&up_weight);
        let expected = qwen35_dense_mlp_cpu(
            &x,
            &gate_weight,
            &up_weight,
            &down_weight,
            tokens,
            hidden_size,
            intermediate_size,
        )
        .unwrap();
        let gate =
            qwen35_linear_cpu(&x, &gate_weight, tokens, hidden_size, intermediate_size).unwrap();
        let up = qwen35_linear_cpu(&x, &up_weight, tokens, hidden_size, intermediate_size).unwrap();
        let mut expected_gate_up = Vec::with_capacity(tokens * 2 * intermediate_size);
        for token in 0..tokens {
            expected_gate_up.extend_from_slice(
                &gate[token * intermediate_size..(token + 1) * intermediate_size],
            );
            expected_gate_up
                .extend_from_slice(&up[token * intermediate_size..(token + 1) * intermediate_size]);
        }
        let expected_fused = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| silu(*gate) * up)
            .collect::<Vec<_>>();
        let gate_up_proj = DenseLinear::<CpuBackend>::from_rows(
            &gate_up_weight,
            2 * intermediate_size,
            hidden_size,
        );
        let down_proj =
            DenseLinear::<CpuBackend>::from_rows(&down_weight, hidden_size, intermediate_size);

        let mut ctx = CpuBackend::new_context();
        let output = qwen35_dense_mlp_backend::<CpuBackend>(
            &mut ctx,
            &CpuBackend::from_slice(&x),
            &gate_up_proj,
            &down_proj,
            tokens,
            hidden_size,
            intermediate_size,
        )
        .unwrap();

        assert_close_slice(
            &CpuBackend::to_vec(&output.gate_up, expected_gate_up.len()),
            &expected_gate_up,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.fused, expected_fused.len()),
            &expected_fused,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.output, expected.len()),
            &expected,
            1e-6,
        );
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

        let input_norm_folded = input_norm_weight
            .iter()
            .map(|value| 1.0 + value)
            .collect::<Vec<_>>();
        let post_norm_folded = post_attention_norm_weight
            .iter()
            .map(|value| 1.0 + value)
            .collect::<Vec<_>>();
        let mut gate_up_weight = gate_proj_weight.clone();
        gate_up_weight.extend_from_slice(&up_proj_weight);
        let layer = Qwen35LayerWeights {
            layer_index: 0,
            input_layernorm: CpuBackend::from_slice(&input_norm_folded),
            post_attention_layernorm: CpuBackend::from_slice(&post_norm_folded),
            attention: Qwen35AttentionWeights::Linear(Qwen35LinearAttentionWeights {
                qkv_proj: Box::new(DenseLinear::<CpuBackend>::from_rows(
                    &qkv_weight,
                    attention_shape.conv_channels(),
                    shape.hidden_size,
                )),
                z_proj: Box::new(DenseLinear::<CpuBackend>::from_rows(
                    &z_weight,
                    attention_shape.value_total(),
                    shape.hidden_size,
                )),
                b_proj: Box::new(DenseLinear::<CpuBackend>::from_rows(
                    &b_weight,
                    attention_shape.value_heads,
                    shape.hidden_size,
                )),
                a_proj: Box::new(DenseLinear::<CpuBackend>::from_rows(
                    &a_weight,
                    attention_shape.value_heads,
                    shape.hidden_size,
                )),
                qkvz_proj: None,
                ba_proj: None,
                conv1d_weight: CpuBackend::from_slice(&conv1d_weight),
                a_log: CpuBackend::from_slice(&a_log),
                dt_bias: CpuBackend::from_slice(&dt_bias),
                norm_weight: CpuBackend::from_slice(&norm_weight),
                out_proj: Box::new(DenseLinear::<CpuBackend>::from_rows(
                    &out_proj_weight,
                    shape.hidden_size,
                    attention_shape.value_total(),
                )),
            }),
            mlp: Qwen35MlpWeights::Dense(Qwen35DenseMlpWeights {
                gate_up_proj: Box::new(DenseLinear::<CpuBackend>::from_rows(
                    &gate_up_weight,
                    2 * shape.intermediate_size,
                    shape.hidden_size,
                )),
                down_proj: Box::new(DenseLinear::<CpuBackend>::from_rows(
                    &down_proj_weight,
                    shape.hidden_size,
                    shape.intermediate_size,
                )),
            }),
        };
        let mut ctx = CpuBackend::new_context();
        let backend = qwen35_dense_linear_attention_layer_backend::<CpuBackend>(
            &mut ctx,
            &CpuBackend::from_slice(&layer_input),
            &CpuBackend::from_slice(&initial_state),
            &layer,
            shape,
            1e-6,
        )
        .unwrap();

        assert_close_slice(
            &CpuBackend::to_vec(&backend.input_norm, input_norm.len()),
            &input_norm,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.mixed_qkv_raw, mixed_qkv_raw.len()),
            &mixed_qkv_raw,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.z_raw, z_raw.len()),
            &z_raw,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.b_raw, b_raw.len()),
            &b_raw,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.a_raw, a_raw.len()),
            &a_raw,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.attention.value, attention.value.len()),
            &attention.value,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.attention.g, attention.g.len()),
            &attention.g,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.attention.beta, attention.beta.len()),
            &attention.beta,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.attention.delta_core, attention.delta_core.len()),
            &attention.delta_core,
            1e-5,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.attention.delta_norm, attention.delta_norm.len()),
            &attention.delta_norm,
            1e-5,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.attention.final_state, attention.final_state.len()),
            &attention.final_state,
            1e-5,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.final_conv_state, attention.final_conv_state.len()),
            &attention.final_conv_state,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.delta_output, delta_output.len()),
            &delta_output,
            1e-5,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.residual_after_mixer, residual_after_mixer.len()),
            &residual_after_mixer,
            1e-5,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.post_attention_norm, post_attention_norm.len()),
            &post_attention_norm,
            1e-5,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.mlp.output, mlp_output.len()),
            &mlp_output,
            1e-5,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.layer_output, layer_output.len()),
            &layer_output,
            1e-5,
        );
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
            rope_dim: 2,
            position_offset: 0,
            rope_theta: 10_000.0,
            rope_interleaved: false,
            attn_output_gate: false,
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

        let input_norm_folded = input_norm_weight
            .iter()
            .map(|value| 1.0 + value)
            .collect::<Vec<_>>();
        let q_norm_folded = q_norm_weight
            .iter()
            .map(|value| 1.0 + value)
            .collect::<Vec<_>>();
        let k_norm_folded = k_norm_weight
            .iter()
            .map(|value| 1.0 + value)
            .collect::<Vec<_>>();
        let post_norm_folded = post_attention_norm_weight
            .iter()
            .map(|value| 1.0 + value)
            .collect::<Vec<_>>();
        let mut gate_up_weight = gate_proj_weight.clone();
        gate_up_weight.extend_from_slice(&up_proj_weight);
        let layer = Qwen35LayerWeights {
            layer_index: 0,
            input_layernorm: CpuBackend::from_slice(&input_norm_folded),
            post_attention_layernorm: CpuBackend::from_slice(&post_norm_folded),
            attention: Qwen35AttentionWeights::Full(Qwen35FullAttentionWeights {
                q_proj: Box::new(DenseLinear::<CpuBackend>::from_rows(
                    &q_weight,
                    attention_shape.q_total(),
                    shape.hidden_size,
                )),
                k_proj: Box::new(DenseLinear::<CpuBackend>::from_rows(
                    &k_weight,
                    attention_shape.kv_total(),
                    shape.hidden_size,
                )),
                v_proj: Box::new(DenseLinear::<CpuBackend>::from_rows(
                    &v_weight,
                    attention_shape.kv_total(),
                    shape.hidden_size,
                )),
                o_proj: Box::new(DenseLinear::<CpuBackend>::from_rows(
                    &o_weight,
                    shape.hidden_size,
                    attention_shape.q_total(),
                )),
                q_norm_weight: CpuBackend::from_slice(&q_norm_folded),
                k_norm_weight: CpuBackend::from_slice(&k_norm_folded),
            }),
            mlp: Qwen35MlpWeights::Dense(Qwen35DenseMlpWeights {
                gate_up_proj: Box::new(DenseLinear::<CpuBackend>::from_rows(
                    &gate_up_weight,
                    2 * shape.intermediate_size,
                    shape.hidden_size,
                )),
                down_proj: Box::new(DenseLinear::<CpuBackend>::from_rows(
                    &down_proj_weight,
                    shape.hidden_size,
                    shape.intermediate_size,
                )),
            }),
        };
        let rope = qwen35_build_rope_cache_backend::<CpuBackend>(
            attention_shape.position_offset + attention_shape.tokens,
            attention_shape.rope_dim,
            attention_shape.rope_theta,
        )
        .unwrap();
        let mut ctx = CpuBackend::new_context();
        let backend = qwen35_dense_full_attention_layer_backend::<CpuBackend>(
            &mut ctx,
            &CpuBackend::from_slice(&layer_input),
            &layer,
            &rope,
            shape,
            1e-6,
        )
        .unwrap();

        assert_close_slice(
            &CpuBackend::to_vec(&backend.input_norm, input_norm.len()),
            &input_norm,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.query_raw, query_raw.len()),
            &query_raw,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.key_raw, key_raw.len()),
            &key_raw,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.value_raw, value_raw.len()),
            &value_raw,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.attention.context, attention.context.len()),
            &attention.context,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.attn_output, attn_output.len()),
            &attn_output,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(
                &backend.residual_after_attention,
                residual_after_attention.len(),
            ),
            &residual_after_attention,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.post_attention_norm, post_attention_norm.len()),
            &post_attention_norm,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.mlp.output, mlp_output.len()),
            &mlp_output,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&backend.layer_output, layer_output.len()),
            &layer_output,
            1e-6,
        );
    }

    #[test]
    fn sparse_moe_linear_attention_layer_composes_attention_residual_and_moe() {
        let attention_shape = Qwen35LinearAttentionShape {
            tokens: 2,
            key_heads: 1,
            value_heads: 1,
            key_dim: 1,
            value_dim: 1,
            conv_kernel: 1,
        };
        let moe_shape = single_expert_moe_shape(2, 2);
        let shape = Qwen35SparseMoeLinearAttentionLayerShape {
            tokens: 2,
            hidden_size: 2,
            attention: attention_shape,
            moe: moe_shape,
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
        let initial_state = vec![0.0; attention_shape.state_len()];
        let moe_weights = small_moe_weights();

        let dense_attention_only = qwen35_dense_linear_attention_layer_cpu(
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
            &[0.0; 4],
            &[0.0; 4],
            &[0.0; 4],
            &initial_state,
            Qwen35DenseLinearAttentionLayerShape {
                tokens: 2,
                hidden_size: 2,
                intermediate_size: 2,
                attention: attention_shape,
            },
            1e-6,
        )
        .unwrap();
        let moe = qwen35_sparse_moe_shared_expert_cpu(
            &dense_attention_only.post_attention_norm,
            &moe_weights.router_weight,
            &moe_weights.fused_gate_up_proj,
            &moe_weights.fused_down_proj,
            &moe_weights.shared_expert_gate_weight,
            &moe_weights.shared_expert_gate_proj_weight,
            &moe_weights.shared_expert_up_proj_weight,
            &moe_weights.shared_expert_down_proj_weight,
            moe_shape,
        )
        .unwrap();
        let expected_layer_output = add_same_len(
            &dense_attention_only.residual_after_mixer,
            &moe.moe_output,
            "layer_output",
        )
        .unwrap();

        let reference = qwen35_sparse_moe_linear_attention_layer_cpu(
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
            &moe_weights.router_weight,
            &moe_weights.fused_gate_up_proj,
            &moe_weights.fused_down_proj,
            &moe_weights.shared_expert_gate_weight,
            &moe_weights.shared_expert_gate_proj_weight,
            &moe_weights.shared_expert_up_proj_weight,
            &moe_weights.shared_expert_down_proj_weight,
            &initial_state,
            shape,
            1e-6,
        )
        .unwrap();

        assert_close_slice(
            &reference.residual_after_mixer,
            &dense_attention_only.residual_after_mixer,
            1e-6,
        );
        assert_close_slice(
            &reference.post_attention_norm,
            &dense_attention_only.post_attention_norm,
            1e-6,
        );
        assert_eq!(reference.moe, moe);
        assert_close_slice(&reference.layer_output, &expected_layer_output, 1e-6);
    }

    #[test]
    fn sparse_moe_full_attention_layer_composes_attention_residual_and_moe() {
        let attention_shape = Qwen35FullAttentionShape {
            tokens: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            rope_dim: 2,
            position_offset: 0,
            rope_theta: 10_000.0,
            rope_interleaved: false,
            attn_output_gate: false,
        };
        let moe_shape = single_expert_moe_shape(2, 2);
        let shape = Qwen35SparseMoeFullAttentionLayerShape {
            tokens: 2,
            hidden_size: 2,
            attention: attention_shape,
            moe: moe_shape,
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
        let moe_weights = small_moe_weights();

        let dense_attention_only = qwen35_dense_full_attention_layer_cpu(
            &layer_input,
            &input_norm_weight,
            &q_weight,
            &k_weight,
            &v_weight,
            &q_norm_weight,
            &k_norm_weight,
            &o_weight,
            &post_attention_norm_weight,
            &[0.0; 4],
            &[0.0; 4],
            &[0.0; 4],
            Qwen35DenseFullAttentionLayerShape {
                tokens: 2,
                hidden_size: 2,
                intermediate_size: 2,
                attention: attention_shape,
            },
            1e-6,
        )
        .unwrap();
        let moe = qwen35_sparse_moe_shared_expert_cpu(
            &dense_attention_only.post_attention_norm,
            &moe_weights.router_weight,
            &moe_weights.fused_gate_up_proj,
            &moe_weights.fused_down_proj,
            &moe_weights.shared_expert_gate_weight,
            &moe_weights.shared_expert_gate_proj_weight,
            &moe_weights.shared_expert_up_proj_weight,
            &moe_weights.shared_expert_down_proj_weight,
            moe_shape,
        )
        .unwrap();
        let expected_layer_output = add_same_len(
            &dense_attention_only.residual_after_attention,
            &moe.moe_output,
            "layer_output",
        )
        .unwrap();

        let reference = qwen35_sparse_moe_full_attention_layer_cpu(
            &layer_input,
            &input_norm_weight,
            &q_weight,
            &k_weight,
            &v_weight,
            &q_norm_weight,
            &k_norm_weight,
            &o_weight,
            &post_attention_norm_weight,
            &moe_weights.router_weight,
            &moe_weights.fused_gate_up_proj,
            &moe_weights.fused_down_proj,
            &moe_weights.shared_expert_gate_weight,
            &moe_weights.shared_expert_gate_proj_weight,
            &moe_weights.shared_expert_up_proj_weight,
            &moe_weights.shared_expert_down_proj_weight,
            shape,
            1e-6,
        )
        .unwrap();

        assert_close_slice(
            &reference.residual_after_attention,
            &dense_attention_only.residual_after_attention,
            1e-6,
        );
        assert_close_slice(
            &reference.post_attention_norm,
            &dense_attention_only.post_attention_norm,
            1e-6,
        );
        assert_eq!(reference.moe, moe);
        assert_close_slice(&reference.layer_output, &expected_layer_output, 1e-6);
    }

    #[test]
    fn dense_full_attention_layer_accepts_qwen35_gate_shape_with_hidden_not_q_total() {
        let shape = Qwen35DenseFullAttentionLayerShape {
            tokens: 1,
            hidden_size: 2,
            intermediate_size: 1,
            attention: Qwen35FullAttentionShape {
                tokens: 1,
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 4,
                rope_dim: 2,
                position_offset: 0,
                rope_theta: 10_000.0,
                rope_interleaved: true,
                attn_output_gate: true,
            },
        };
        let reference = qwen35_dense_full_attention_layer_cpu(
            &[0.0; 2], &[0.0; 2], &[0.0; 32], &[0.0; 8], &[0.0; 8], &[0.0; 4], &[0.0; 4],
            &[0.0; 16], &[0.0; 2], &[0.0; 2], &[0.0; 2], &[0.0; 2], shape, 1e-6,
        )
        .expect("Qwen3.5 full attention permits hidden_size != q_total with gated q_proj");

        assert_eq!(shape.hidden_size, 2);
        assert_eq!(shape.attention.q_total(), 8);
        assert_eq!(shape.attention.q_proj_total(), 16);
        assert_eq!(reference.query_raw.len(), shape.attention.q_proj_len());
        assert_eq!(reference.attention.context.len(), shape.attention.q_len());
        assert_eq!(
            reference.attention.attention_gate.as_ref().unwrap().len(),
            8
        );
        assert_eq!(reference.attn_output.len(), shape.hidden_size);
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
    fn dense_reference_model_forward_composes_layers_norm_and_lm_head() {
        let input_ids = vec![0, 1];
        let embed_tokens = vec![
            1.0, 0.0, //
            0.0, 1.0, //
            1.0, 1.0,
        ];
        let linear_shape = Qwen35DenseLinearAttentionLayerShape {
            tokens: 2,
            hidden_size: 2,
            intermediate_size: 2,
            attention: Qwen35LinearAttentionShape {
                tokens: 2,
                key_heads: 1,
                value_heads: 1,
                key_dim: 1,
                value_dim: 1,
                conv_kernel: 1,
            },
        };
        let linear_input_norm = vec![0.0, 0.0];
        let linear_qkv = vec![
            1.0, 0.0, //
            0.0, 1.0, //
            1.0, 1.0,
        ];
        let linear_z = vec![1.0, -1.0];
        let linear_b = vec![0.5, 0.25];
        let linear_a = vec![-0.25, 0.75];
        let linear_conv = vec![1.0, 1.0, 1.0];
        let linear_a_log = vec![0.0];
        let linear_dt_bias = vec![0.0];
        let linear_norm = vec![1.0];
        let linear_out = vec![1.0, -0.5];
        let linear_post_norm = vec![0.0, 0.0];
        let linear_gate = vec![
            0.2, 0.1, //
            -0.1, 0.3,
        ];
        let linear_up = vec![
            0.4, -0.2, //
            0.3, 0.5,
        ];
        let linear_down = vec![
            1.0, 0.0, //
            0.0, 1.0,
        ];

        let full_shape = Qwen35DenseFullAttentionLayerShape {
            tokens: 2,
            hidden_size: 2,
            intermediate_size: 2,
            attention: Qwen35FullAttentionShape {
                tokens: 2,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
                rope_dim: 2,
                position_offset: 0,
                rope_theta: 10_000.0,
                rope_interleaved: false,
                attn_output_gate: false,
            },
        };
        let full_input_norm = vec![0.0, 0.0];
        let full_q = vec![
            1.0, 0.0, //
            0.0, 1.0,
        ];
        let full_k = vec![
            0.5, 0.0, //
            0.0, 0.5,
        ];
        let full_v = vec![
            1.0, 1.0, //
            -0.5, 0.5,
        ];
        let full_q_norm = vec![1.0, 1.0];
        let full_k_norm = vec![1.0, 1.0];
        let full_o = vec![
            1.0, 0.0, //
            0.0, 1.0,
        ];
        let full_post_norm = vec![0.0, 0.0];
        let full_gate = vec![
            -0.2, 0.2, //
            0.1, 0.3,
        ];
        let full_up = vec![
            0.25, 0.5, //
            -0.3, 0.4,
        ];
        let full_down = vec![
            0.5, 0.25, //
            -0.2, 0.75,
        ];
        let final_norm = vec![0.0, 0.0];
        let lm_head = vec![
            1.0, 0.0, //
            0.0, 1.0, //
            1.0, 1.0,
        ];

        let hidden0 = gather_embedding_rows(&embed_tokens, &input_ids, 3, 2).unwrap();
        let linear_ref = qwen35_dense_linear_attention_layer_cpu(
            &hidden0,
            &linear_input_norm,
            &linear_qkv,
            &linear_z,
            &linear_b,
            &linear_a,
            &linear_conv,
            &linear_a_log,
            &linear_dt_bias,
            &linear_norm,
            &linear_out,
            &linear_post_norm,
            &linear_gate,
            &linear_up,
            &linear_down,
            &[0.0; 1],
            linear_shape,
            1e-6,
        )
        .unwrap();
        let full_ref = qwen35_dense_full_attention_layer_cpu(
            &linear_ref.layer_output,
            &full_input_norm,
            &full_q,
            &full_k,
            &full_v,
            &full_q_norm,
            &full_k_norm,
            &full_o,
            &full_post_norm,
            &full_gate,
            &full_up,
            &full_down,
            full_shape,
            1e-6,
        )
        .unwrap();
        let expected_final =
            qwen35_rms_norm_plus_one_cpu(&full_ref.layer_output, &final_norm, 2, 2, 1e-6).unwrap();
        let expected_logits = qwen35_linear_cpu(&expected_final, &lm_head, 2, 2, 3).unwrap();
        let layers = vec![
            Qwen35DenseReferenceLayer::Linear(Qwen35DenseReferenceLinearLayer {
                shape: linear_shape,
                input_norm_weight: &linear_input_norm,
                qkv_weight: &linear_qkv,
                z_weight: &linear_z,
                b_weight: &linear_b,
                a_weight: &linear_a,
                conv1d_weight: &linear_conv,
                a_log: &linear_a_log,
                dt_bias: &linear_dt_bias,
                norm_weight: &linear_norm,
                out_proj_weight: &linear_out,
                post_attention_norm_weight: &linear_post_norm,
                gate_proj_weight: &linear_gate,
                up_proj_weight: &linear_up,
                down_proj_weight: &linear_down,
            }),
            Qwen35DenseReferenceLayer::Full(Qwen35DenseReferenceFullLayer {
                shape: full_shape,
                input_norm_weight: &full_input_norm,
                q_weight: &full_q,
                k_weight: &full_k,
                v_weight: &full_v,
                q_norm_weight: &full_q_norm,
                k_norm_weight: &full_k_norm,
                o_weight: &full_o,
                post_attention_norm_weight: &full_post_norm,
                gate_proj_weight: &full_gate,
                up_proj_weight: &full_up,
                down_proj_weight: &full_down,
            }),
        ];
        let output = qwen35_dense_reference_model_forward_cpu(
            Qwen35DenseReferenceModel {
                vocab_size: 3,
                hidden_size: 2,
                eps: 1e-6,
                embed_tokens: &embed_tokens,
                final_norm_weight: &final_norm,
                lm_head_weight: &lm_head,
                layers: &layers,
            },
            &input_ids,
        )
        .unwrap();

        assert_eq!(output.layer_hidden_states.len(), 2);
        assert_eq!(output.linear_conv_states.len(), 1);
        assert_eq!(output.linear_recurrent_states.len(), 1);
        assert_close_slice(
            &output.layer_hidden_states[0],
            &linear_ref.layer_output,
            1e-6,
        );
        assert_close_slice(&output.hidden, &full_ref.layer_output, 1e-6);
        assert_close_slice(&output.final_hidden, &expected_final, 1e-6);
        assert_close_slice(&output.logits, &expected_logits, 1e-6);
        assert_close_slice(
            &output.linear_conv_states[0],
            &linear_ref.attention.final_conv_state,
            1e-6,
        );
        assert_close_slice(
            &output.linear_recurrent_states[0],
            &linear_ref.attention.final_state,
            1e-6,
        );
    }

    #[test]
    fn sparse_moe_reference_model_forward_composes_layers_norm_and_lm_head() {
        let input_ids = vec![0, 1];
        let embed_tokens = vec![
            1.0, 0.0, //
            0.0, 1.0, //
            1.0, 1.0,
        ];
        let linear_attention = Qwen35LinearAttentionShape {
            tokens: 2,
            key_heads: 1,
            value_heads: 1,
            key_dim: 1,
            value_dim: 1,
            conv_kernel: 1,
        };
        let moe_shape = single_expert_moe_shape(2, 2);
        let linear_shape = Qwen35SparseMoeLinearAttentionLayerShape {
            tokens: 2,
            hidden_size: 2,
            attention: linear_attention,
            moe: moe_shape,
        };
        let linear_input_norm = vec![0.0, 0.0];
        let linear_qkv = vec![
            1.0, 0.0, //
            0.0, 1.0, //
            1.0, 1.0,
        ];
        let linear_z = vec![1.0, -1.0];
        let linear_b = vec![0.5, 0.25];
        let linear_a = vec![-0.25, 0.75];
        let linear_conv = vec![1.0, 1.0, 1.0];
        let linear_a_log = vec![0.0];
        let linear_dt_bias = vec![0.0];
        let linear_norm = vec![1.0];
        let linear_out = vec![1.0, -0.5];
        let linear_post_norm = vec![0.0, 0.0];

        let full_attention = Qwen35FullAttentionShape {
            tokens: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            rope_dim: 2,
            position_offset: 0,
            rope_theta: 10_000.0,
            rope_interleaved: false,
            attn_output_gate: false,
        };
        let full_shape = Qwen35SparseMoeFullAttentionLayerShape {
            tokens: 2,
            hidden_size: 2,
            attention: full_attention,
            moe: moe_shape,
        };
        let full_input_norm = vec![0.0, 0.0];
        let full_q = vec![
            1.0, 0.0, //
            0.0, 1.0,
        ];
        let full_k = vec![
            0.5, 0.0, //
            0.0, 0.5,
        ];
        let full_v = vec![
            1.0, 1.0, //
            -0.5, 0.5,
        ];
        let full_q_norm = vec![1.0, 1.0];
        let full_k_norm = vec![1.0, 1.0];
        let full_o = vec![
            1.0, 0.0, //
            0.0, 1.0,
        ];
        let full_post_norm = vec![0.0, 0.0];
        let final_norm = vec![0.0, 0.0];
        let lm_head = vec![
            1.0, 0.0, //
            0.0, 1.0, //
            1.0, 1.0,
        ];
        let moe_weights = small_moe_weights();

        let hidden0 = gather_embedding_rows(&embed_tokens, &input_ids, 3, 2).unwrap();
        let linear_ref = qwen35_sparse_moe_linear_attention_layer_cpu(
            &hidden0,
            &linear_input_norm,
            &linear_qkv,
            &linear_z,
            &linear_b,
            &linear_a,
            &linear_conv,
            &linear_a_log,
            &linear_dt_bias,
            &linear_norm,
            &linear_out,
            &linear_post_norm,
            &moe_weights.router_weight,
            &moe_weights.fused_gate_up_proj,
            &moe_weights.fused_down_proj,
            &moe_weights.shared_expert_gate_weight,
            &moe_weights.shared_expert_gate_proj_weight,
            &moe_weights.shared_expert_up_proj_weight,
            &moe_weights.shared_expert_down_proj_weight,
            &[0.0; 1],
            linear_shape,
            1e-6,
        )
        .unwrap();
        let full_ref = qwen35_sparse_moe_full_attention_layer_cpu(
            &linear_ref.layer_output,
            &full_input_norm,
            &full_q,
            &full_k,
            &full_v,
            &full_q_norm,
            &full_k_norm,
            &full_o,
            &full_post_norm,
            &moe_weights.router_weight,
            &moe_weights.fused_gate_up_proj,
            &moe_weights.fused_down_proj,
            &moe_weights.shared_expert_gate_weight,
            &moe_weights.shared_expert_gate_proj_weight,
            &moe_weights.shared_expert_up_proj_weight,
            &moe_weights.shared_expert_down_proj_weight,
            full_shape,
            1e-6,
        )
        .unwrap();
        let expected_final =
            qwen35_rms_norm_plus_one_cpu(&full_ref.layer_output, &final_norm, 2, 2, 1e-6).unwrap();
        let expected_logits = qwen35_linear_cpu(&expected_final, &lm_head, 2, 2, 3).unwrap();
        let layers = vec![
            Qwen35SparseMoeReferenceLayer::Linear(Qwen35SparseMoeReferenceLinearLayer {
                shape: linear_shape,
                input_norm_weight: &linear_input_norm,
                qkv_weight: &linear_qkv,
                z_weight: &linear_z,
                b_weight: &linear_b,
                a_weight: &linear_a,
                conv1d_weight: &linear_conv,
                a_log: &linear_a_log,
                dt_bias: &linear_dt_bias,
                norm_weight: &linear_norm,
                out_proj_weight: &linear_out,
                post_attention_norm_weight: &linear_post_norm,
                router_weight: &moe_weights.router_weight,
                fused_gate_up_proj: &moe_weights.fused_gate_up_proj,
                fused_down_proj: &moe_weights.fused_down_proj,
                shared_expert_gate_weight: &moe_weights.shared_expert_gate_weight,
                shared_expert_gate_proj_weight: &moe_weights.shared_expert_gate_proj_weight,
                shared_expert_up_proj_weight: &moe_weights.shared_expert_up_proj_weight,
                shared_expert_down_proj_weight: &moe_weights.shared_expert_down_proj_weight,
            }),
            Qwen35SparseMoeReferenceLayer::Full(Qwen35SparseMoeReferenceFullLayer {
                shape: full_shape,
                input_norm_weight: &full_input_norm,
                q_weight: &full_q,
                k_weight: &full_k,
                v_weight: &full_v,
                q_norm_weight: &full_q_norm,
                k_norm_weight: &full_k_norm,
                o_weight: &full_o,
                post_attention_norm_weight: &full_post_norm,
                router_weight: &moe_weights.router_weight,
                fused_gate_up_proj: &moe_weights.fused_gate_up_proj,
                fused_down_proj: &moe_weights.fused_down_proj,
                shared_expert_gate_weight: &moe_weights.shared_expert_gate_weight,
                shared_expert_gate_proj_weight: &moe_weights.shared_expert_gate_proj_weight,
                shared_expert_up_proj_weight: &moe_weights.shared_expert_up_proj_weight,
                shared_expert_down_proj_weight: &moe_weights.shared_expert_down_proj_weight,
            }),
        ];
        let output = qwen35_sparse_moe_reference_model_forward_cpu(
            Qwen35SparseMoeReferenceModel {
                vocab_size: 3,
                hidden_size: 2,
                eps: 1e-6,
                embed_tokens: &embed_tokens,
                final_norm_weight: &final_norm,
                lm_head_weight: &lm_head,
                layers: &layers,
            },
            &input_ids,
        )
        .unwrap();

        assert_eq!(output.layer_hidden_states.len(), 2);
        assert_eq!(output.linear_conv_states.len(), 1);
        assert_eq!(output.linear_recurrent_states.len(), 1);
        assert_eq!(output.sparse_moe_outputs.len(), 2);
        assert_close_slice(
            &output.layer_hidden_states[0],
            &linear_ref.layer_output,
            1e-6,
        );
        assert_close_slice(&output.hidden, &full_ref.layer_output, 1e-6);
        assert_close_slice(&output.final_hidden, &expected_final, 1e-6);
        assert_close_slice(&output.logits, &expected_logits, 1e-6);
        assert_close_slice(
            &output.linear_conv_states[0],
            &linear_ref.attention.final_conv_state,
            1e-6,
        );
        assert_close_slice(
            &output.linear_recurrent_states[0],
            &linear_ref.attention.final_state,
            1e-6,
        );
        assert_eq!(output.sparse_moe_outputs[0], linear_ref.moe);
        assert_eq!(output.sparse_moe_outputs[1], full_ref.moe);
    }

    #[test]
    fn dense_reference_model_forward_rejects_oov_token() {
        let embed_tokens = vec![1.0, 0.0, 0.0, 1.0];
        let layers = vec![Qwen35DenseReferenceLayer::Full(
            Qwen35DenseReferenceFullLayer {
                shape: Qwen35DenseFullAttentionLayerShape {
                    tokens: 1,
                    hidden_size: 2,
                    intermediate_size: 1,
                    attention: Qwen35FullAttentionShape {
                        tokens: 1,
                        num_heads: 1,
                        num_kv_heads: 1,
                        head_dim: 2,
                        rope_dim: 2,
                        position_offset: 0,
                        rope_theta: 10_000.0,
                        rope_interleaved: false,
                        attn_output_gate: false,
                    },
                },
                input_norm_weight: &[0.0, 0.0],
                q_weight: &[1.0, 0.0, 0.0, 1.0],
                k_weight: &[1.0, 0.0, 0.0, 1.0],
                v_weight: &[1.0, 0.0, 0.0, 1.0],
                q_norm_weight: &[1.0, 1.0],
                k_norm_weight: &[1.0, 1.0],
                o_weight: &[1.0, 0.0, 0.0, 1.0],
                post_attention_norm_weight: &[0.0, 0.0],
                gate_proj_weight: &[1.0, 1.0],
                up_proj_weight: &[1.0, 1.0],
                down_proj_weight: &[1.0, 1.0],
            },
        )];
        let err = qwen35_dense_reference_model_forward_cpu(
            Qwen35DenseReferenceModel {
                vocab_size: 2,
                hidden_size: 2,
                eps: 1e-6,
                embed_tokens: &embed_tokens,
                final_norm_weight: &[0.0, 0.0],
                lm_head_weight: &[1.0, 0.0, 0.0, 1.0],
                layers: &layers,
            },
            &[2],
        )
        .expect_err("token id 2 should exceed vocab size 2");

        assert!(err.to_string().contains("token id 2"), "{err}");
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
        assert!(reference.final_conv_state.is_empty());
        assert_close_slice(&reference.final_state, &final_state, 1e-6);
    }

    #[test]
    fn final_conv_state_uses_dim_first_left_padded_layout() {
        let shape = Qwen35LinearAttentionShape {
            tokens: 2,
            key_heads: 1,
            value_heads: 1,
            key_dim: 2,
            value_dim: 2,
            conv_kernel: 4,
        };
        let mixed_qkv_raw = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];

        let state = qwen35_final_conv_state_cpu(&mixed_qkv_raw, shape).unwrap();

        assert_eq!(
            state,
            vec![
                0.0, 1.0, 7.0, //
                0.0, 2.0, 8.0, //
                0.0, 3.0, 9.0, //
                0.0, 4.0, 10.0, //
                0.0, 5.0, 11.0, //
                0.0, 6.0, 12.0,
            ]
        );
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
    fn linear_attention_prefill_backend_matches_reference_core() {
        let shape = Qwen35LinearAttentionShape {
            tokens: 3,
            key_heads: 1,
            value_heads: 2,
            key_dim: 2,
            value_dim: 2,
            conv_kernel: 2,
        };
        let mixed_qkv_raw: Vec<f32> = (0..shape.mixed_qkv_len())
            .map(|i| ((i as f32 % 9.0) - 4.0) * 0.2)
            .collect();
        let conv1d_weight: Vec<f32> = (0..shape.conv_channels() * shape.conv_kernel)
            .map(|i| if i % 2 == 0 { -0.25 } else { 0.75 })
            .collect();
        let z_raw: Vec<f32> = (0..shape.value_len())
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.15)
            .collect();
        let a_raw: Vec<f32> = (0..shape.gating_len())
            .map(|i| ((i as f32 % 5.0) - 2.0) * 0.25)
            .collect();
        let b_raw: Vec<f32> = (0..shape.gating_len())
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.1)
            .collect();
        let a_log = vec![0.5f32.ln(), 1.25f32.ln()];
        let dt_bias = vec![-0.1, 0.2];
        let norm_weight = vec![0.75, 1.5];
        let initial_state: Vec<f32> = (0..shape.state_len())
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.05)
            .collect();
        let eps = 1e-6;

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
            eps,
        )
        .unwrap();

        let mut ctx = CpuBackend::new_context();
        let output = qwen35_linear_attention_prefill_core_backend::<CpuBackend>(
            &mut ctx,
            &CpuBackend::from_slice_typed(&mixed_qkv_raw),
            &CpuBackend::from_slice_typed(&z_raw),
            &CpuBackend::from_slice_typed(&a_raw),
            &CpuBackend::from_slice_typed(&b_raw),
            &CpuBackend::from_slice_typed(&conv1d_weight),
            &CpuBackend::from_slice_typed(&a_log),
            &CpuBackend::from_slice_typed(&dt_bias),
            &CpuBackend::from_slice_typed(&norm_weight),
            &CpuBackend::from_slice_typed(&initial_state),
            shape,
            eps,
        )
        .unwrap();

        assert_close_slice(
            &CpuBackend::to_vec(&output.value, shape.value_len()),
            &reference.value,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.g, shape.gating_len()),
            &reference.g,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.beta, shape.gating_len()),
            &reference.beta,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.delta_core, shape.value_len()),
            &reference.delta_core,
            1e-5,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.delta_norm, shape.value_len()),
            &reference.delta_norm,
            1e-5,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.final_state, shape.state_len()),
            &reference.final_state,
            1e-5,
        );
    }

    #[test]
    fn linear_attention_decode_backend_matches_full_reference_last_token() {
        let full_shape = Qwen35LinearAttentionShape {
            tokens: 3,
            key_heads: 1,
            value_heads: 2,
            key_dim: 2,
            value_dim: 2,
            conv_kernel: 3,
        };
        let prefix_shape = Qwen35LinearAttentionShape {
            tokens: 2,
            ..full_shape
        };
        let decode_shape = Qwen35LinearAttentionShape {
            tokens: 1,
            ..full_shape
        };
        let mixed_qkv_raw: Vec<f32> = (0..full_shape.mixed_qkv_len())
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.125)
            .collect();
        let conv1d_weight: Vec<f32> = (0..full_shape.conv_channels() * full_shape.conv_kernel)
            .map(|i| match i % 3 {
                0 => -0.25,
                1 => 0.5,
                _ => 0.75,
            })
            .collect();
        let z_raw: Vec<f32> = (0..full_shape.value_len())
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.15)
            .collect();
        let a_raw: Vec<f32> = (0..full_shape.gating_len())
            .map(|i| ((i as f32 % 5.0) - 2.0) * 0.25)
            .collect();
        let b_raw: Vec<f32> = (0..full_shape.gating_len())
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
            .collect();
        let a_log = vec![0.5f32.ln(), 1.25f32.ln()];
        let dt_bias = vec![-0.1, 0.2];
        let norm_weight = vec![0.75, 1.5];
        let initial_state = vec![0.0; full_shape.state_len()];
        let eps = 1e-6;

        let prefix_mixed_len = prefix_shape.mixed_qkv_len();
        let prefix_value_len = prefix_shape.value_len();
        let prefix_gating_len = prefix_shape.gating_len();
        let prefix = qwen35_linear_attention_core_cpu(
            &mixed_qkv_raw[..prefix_mixed_len],
            &z_raw[..prefix_value_len],
            &a_raw[..prefix_gating_len],
            &b_raw[..prefix_gating_len],
            &conv1d_weight,
            &a_log,
            &dt_bias,
            &norm_weight,
            &initial_state,
            prefix_shape,
            eps,
        )
        .unwrap();
        let full = qwen35_linear_attention_core_cpu(
            &mixed_qkv_raw,
            &z_raw,
            &a_raw,
            &b_raw,
            &conv1d_weight,
            &a_log,
            &dt_bias,
            &norm_weight,
            &initial_state,
            full_shape,
            eps,
        )
        .unwrap();

        let decode_mixed_start = prefix_mixed_len;
        let decode_value_start = prefix_value_len;
        let decode_gating_start = prefix_gating_len;
        let mut ctx = CpuBackend::new_context();
        let output = qwen35_linear_attention_decode_core_backend::<CpuBackend>(
            &mut ctx,
            &CpuBackend::from_slice_typed(&mixed_qkv_raw[decode_mixed_start..]),
            &CpuBackend::from_slice_typed(&z_raw[decode_value_start..]),
            &CpuBackend::from_slice_typed(&a_raw[decode_gating_start..]),
            &CpuBackend::from_slice_typed(&b_raw[decode_gating_start..]),
            &CpuBackend::from_slice_typed(&conv1d_weight),
            &CpuBackend::from_slice_typed(&prefix.final_conv_state),
            &CpuBackend::from_slice_typed(&a_log),
            &CpuBackend::from_slice_typed(&dt_bias),
            &CpuBackend::from_slice_typed(&norm_weight),
            &CpuBackend::from_slice_typed(&prefix.final_state),
            decode_shape,
            eps,
        )
        .unwrap();

        let last_token = prefix_shape.tokens;
        let last_qk_start = last_token * full_shape.qk_total();
        let mut expected_query = full.query[last_qk_start..].to_vec();
        let mut expected_key = full.key[last_qk_start..].to_vec();
        for head in 0..full_shape.key_heads {
            let base = head * full_shape.key_dim;
            let mut q_sum = 0.0;
            let mut k_sum = 0.0;
            for d in 0..full_shape.key_dim {
                q_sum += expected_query[base + d] * expected_query[base + d];
                k_sum += expected_key[base + d] * expected_key[base + d];
            }
            let q_inv = (q_sum + 1e-6).sqrt().recip();
            let k_inv = (k_sum + 1e-6).sqrt().recip();
            for d in 0..full_shape.key_dim {
                expected_query[base + d] *= q_inv;
                expected_key[base + d] *= k_inv;
            }
        }
        assert_close_slice(
            &CpuBackend::to_vec(&output.query, decode_shape.qk_total()),
            &expected_query,
            1e-5,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.key, decode_shape.qk_total()),
            &expected_key,
            1e-5,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.value, decode_shape.value_len()),
            &full.value[decode_value_start..],
            1e-5,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.g, decode_shape.gating_len()),
            &full.g[decode_gating_start..],
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.beta, decode_shape.gating_len()),
            &full.beta[decode_gating_start..],
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.next_conv_state, full.final_conv_state.len()),
            &full.final_conv_state,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.delta_core, decode_shape.value_len()),
            &full.delta_core[decode_value_start..],
            1e-5,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.delta_norm, decode_shape.value_len()),
            &full.delta_norm[decode_value_start..],
            1e-5,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.final_state, decode_shape.state_len()),
            &full.final_state,
            1e-5,
        );
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
    fn recurrent_delta_rule_backend_matches_reference() {
        let shape = Qwen35DeltaRuleShape {
            tokens: 2,
            key_heads: 1,
            value_heads: 2,
            key_dim: 2,
            value_dim: 1,
        };
        let query = vec![1.0, 0.5, -0.25, 0.75];
        let key = vec![0.5, -0.5, 1.0, 0.25];
        let value = vec![2.0, -1.0, 0.5, 3.0];
        let g = vec![0.0, -0.25, -0.5, 0.0];
        let beta = vec![0.5, 0.75, 0.25, 1.0];
        let initial_state = vec![0.1, -0.2, 0.0, 0.3];
        let scale = (shape.key_dim as f32).sqrt().recip();
        let (expected_out, expected_state) = qwen35_recurrent_gated_delta_rule_cpu(
            &query,
            &key,
            &value,
            &g,
            &beta,
            &initial_state,
            shape,
            true,
            Some(scale),
        )
        .unwrap();

        let mut ctx = CpuBackend::new_context();
        let output = qwen35_recurrent_gated_delta_rule_backend::<CpuBackend>(
            &mut ctx,
            &CpuBackend::from_slice_typed(&query),
            &CpuBackend::from_slice_typed(&key),
            &CpuBackend::from_slice_typed(&value),
            &CpuBackend::from_slice_typed(&g),
            &CpuBackend::from_slice_typed(&beta),
            &CpuBackend::from_slice_typed(&initial_state),
            shape,
            true,
            Some(scale),
        )
        .unwrap();

        assert_close_slice(
            &CpuBackend::to_vec(&output.output, expected_out.len()),
            &expected_out,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.final_state, expected_state.len()),
            &expected_state,
            1e-6,
        );
    }

    #[test]
    fn recurrent_delta_rule_varlen_backend_matches_per_sequence_reference() {
        let shape = Qwen35DeltaRuleShape {
            tokens: 5,
            key_heads: 1,
            value_heads: 2,
            key_dim: 2,
            value_dim: 1,
        };
        let cu_seqlens = vec![0u32, 2, 5];
        let batch = cu_seqlens.len() - 1;
        let state_len = shape.value_heads * shape.value_dim * shape.key_dim;
        let qk_per_token = shape.key_heads * shape.key_dim;
        let value_per_token = shape.value_heads * shape.value_dim;
        let gating_per_token = shape.value_heads;
        let query: Vec<f32> = (0..shape.tokens * qk_per_token)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.125)
            .collect();
        let key: Vec<f32> = (0..shape.tokens * qk_per_token)
            .map(|i| ((i as f32 % 5.0) - 2.0) * 0.2)
            .collect();
        let value: Vec<f32> = (0..shape.tokens * value_per_token)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.15)
            .collect();
        let g: Vec<f32> = (0..shape.tokens * gating_per_token)
            .map(|i| -0.05 * (i as f32 + 1.0))
            .collect();
        let beta: Vec<f32> = (0..shape.tokens * gating_per_token)
            .map(|i| 0.25 + 0.05 * (i as f32 % 4.0))
            .collect();
        let initial_states: Vec<f32> = (0..batch * state_len)
            .map(|i| ((i as f32 % 9.0) - 4.0) * 0.03)
            .collect();
        let scale = (shape.key_dim as f32).sqrt().recip();

        let mut expected_out = Vec::with_capacity(shape.tokens * value_per_token);
        let mut expected_states = Vec::with_capacity(batch * state_len);
        for seq in 0..batch {
            let start = cu_seqlens[seq] as usize;
            let end = cu_seqlens[seq + 1] as usize;
            let seq_shape = Qwen35DeltaRuleShape {
                tokens: end - start,
                ..shape
            };
            let (seq_out, seq_state) = qwen35_recurrent_gated_delta_rule_cpu(
                &query[start * qk_per_token..end * qk_per_token],
                &key[start * qk_per_token..end * qk_per_token],
                &value[start * value_per_token..end * value_per_token],
                &g[start * gating_per_token..end * gating_per_token],
                &beta[start * gating_per_token..end * gating_per_token],
                &initial_states[seq * state_len..(seq + 1) * state_len],
                seq_shape,
                true,
                Some(scale),
            )
            .unwrap();
            expected_out.extend(seq_out);
            expected_states.extend(seq_state);
        }

        let mut ctx = CpuBackend::new_context();
        let output = qwen35_recurrent_gated_delta_rule_varlen_backend::<CpuBackend>(
            &mut ctx,
            &CpuBackend::from_slice_typed(&query),
            &CpuBackend::from_slice_typed(&key),
            &CpuBackend::from_slice_typed(&value),
            &CpuBackend::from_slice_typed(&g),
            &CpuBackend::from_slice_typed(&beta),
            &CpuBackend::from_slice_typed(&initial_states),
            &CpuBackend::from_slice_typed(&cu_seqlens),
            batch,
            shape,
            true,
            Some(scale),
        )
        .unwrap();

        assert_close_slice(
            &CpuBackend::to_vec(&output.output, expected_out.len()),
            &expected_out,
            1e-6,
        );
        assert_close_slice(
            &CpuBackend::to_vec(&output.final_states, expected_states.len()),
            &expected_states,
            1e-6,
        );
    }

    fn stateful_linear_attention_conv_reference(
        mixed_qkv_raw: &[f32],
        conv_weight: &[f32],
        initial_conv_state: &[f32],
        shape: Qwen35LinearAttentionShape,
    ) -> (Vec<f32>, Vec<f32>) {
        let channels = shape.conv_channels();
        let state_len = shape.conv_kernel.saturating_sub(1);
        let mut conv = vec![0.0; shape.tokens * channels];
        let mut final_state = vec![0.0; channels * state_len];
        for token in 0..shape.tokens {
            for channel in 0..channels {
                let state_base = channel * state_len;
                let mut acc = 0.0;
                for kernel_idx in 0..shape.conv_kernel {
                    let source = token as isize + kernel_idx as isize - state_len as isize;
                    let x = if source >= 0 {
                        mixed_qkv_raw[source as usize * channels + channel]
                    } else {
                        initial_conv_state[state_base + (state_len as isize + source) as usize]
                    };
                    acc += x * conv_weight[channel * shape.conv_kernel + kernel_idx];
                }
                conv[token * channels + channel] = silu(acc);
            }
        }
        for channel in 0..channels {
            let state_base = channel * state_len;
            for pos in 0..state_len {
                let source = shape.tokens as isize + pos as isize - state_len as isize;
                final_state[state_base + pos] = if source >= 0 {
                    mixed_qkv_raw[source as usize * channels + channel]
                } else {
                    initial_conv_state[state_base + (state_len as isize + source) as usize]
                };
            }
        }
        (conv, final_state)
    }

    fn l2_normalize_qk_for_test(query: &mut [f32], key: &mut [f32], shape: Qwen35DeltaRuleShape) {
        for row in 0..shape.tokens * shape.key_heads {
            let base = row * shape.key_dim;
            let mut q_sum = 0.0;
            let mut k_sum = 0.0;
            for d in 0..shape.key_dim {
                q_sum += query[base + d] * query[base + d];
                k_sum += key[base + d] * key[base + d];
            }
            let q_inv = (q_sum + 1e-6).sqrt().recip();
            let k_inv = (k_sum + 1e-6).sqrt().recip();
            for d in 0..shape.key_dim {
                query[base + d] *= q_inv;
                key[base + d] *= k_inv;
            }
        }
    }

    fn assert_linear_attention_prefill_varlen_backend_matches_per_sequence_stateful_reference<
        B: Backend,
    >() {
        let shape = Qwen35LinearAttentionShape {
            tokens: 5,
            key_heads: 1,
            value_heads: 2,
            key_dim: 2,
            value_dim: 2,
            conv_kernel: 3,
        };
        let cu_seqlens = vec![0u32, 2, 5];
        let batch = cu_seqlens.len() - 1;
        let conv_state_len = shape.conv_channels() * shape.conv_kernel.saturating_sub(1);
        let delta_state_len = shape.state_len();
        let mixed_qkv_raw: Vec<f32> = (0..shape.mixed_qkv_len())
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.075)
            .collect();
        let z_raw: Vec<f32> = (0..shape.value_len())
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.09)
            .collect();
        let a_raw: Vec<f32> = (0..shape.gating_len())
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.11)
            .collect();
        let b_raw: Vec<f32> = (0..shape.gating_len())
            .map(|i| ((i as f32 % 5.0) - 2.0) * 0.17)
            .collect();
        let conv1d_weight: Vec<f32> = (0..shape.conv_channels() * shape.conv_kernel)
            .map(|i| match i % 4 {
                0 => -0.25,
                1 => 0.5,
                2 => 0.75,
                _ => -0.125,
            })
            .collect();
        let initial_conv_states: Vec<f32> = (0..batch * conv_state_len)
            .map(|i| ((i as f32 % 17.0) - 8.0) * 0.03)
            .collect();
        let a_log = vec![0.5f32.ln(), 1.25f32.ln()];
        let dt_bias = vec![-0.1, 0.2];
        let norm_weight = vec![0.75, 1.5];
        let initial_states: Vec<f32> = (0..batch * delta_state_len)
            .map(|i| ((i as f32 % 19.0) - 9.0) * 0.02)
            .collect();
        let eps = 1e-6;
        let scale = (shape.key_dim as f32).sqrt().recip();

        let mut expected_query = Vec::with_capacity(shape.tokens * shape.qk_total());
        let mut expected_key = Vec::with_capacity(shape.tokens * shape.qk_total());
        let mut expected_value = Vec::with_capacity(shape.value_len());
        let mut expected_g = Vec::with_capacity(shape.gating_len());
        let mut expected_beta = Vec::with_capacity(shape.gating_len());
        let mut expected_final_conv_states = Vec::with_capacity(batch * conv_state_len);
        let mut expected_delta_core = Vec::with_capacity(shape.value_len());
        let mut expected_delta_norm = Vec::with_capacity(shape.value_len());
        let mut expected_final_states = Vec::with_capacity(batch * delta_state_len);
        for seq in 0..batch {
            let start = cu_seqlens[seq] as usize;
            let end = cu_seqlens[seq + 1] as usize;
            let seq_shape = Qwen35LinearAttentionShape {
                tokens: end - start,
                ..shape
            };
            let mixed_start = start * shape.conv_channels();
            let mixed_end = end * shape.conv_channels();
            let value_start = start * shape.value_total();
            let value_end = end * shape.value_total();
            let gating_start = start * shape.value_heads;
            let gating_end = end * shape.value_heads;
            let (conv, final_conv_state) = stateful_linear_attention_conv_reference(
                &mixed_qkv_raw[mixed_start..mixed_end],
                &conv1d_weight,
                &initial_conv_states[seq * conv_state_len..(seq + 1) * conv_state_len],
                seq_shape,
            );
            let (mut query, mut key, value) =
                qwen35_split_linear_attention_qkv_cpu(&conv, seq_shape).unwrap();
            l2_normalize_qk_for_test(&mut query, &mut key, seq_shape.delta_shape());
            let (g, beta) = qwen35_gdn_gating_cpu(
                &a_log,
                &a_raw[gating_start..gating_end],
                &b_raw[gating_start..gating_end],
                &dt_bias,
                seq_shape.tokens,
                seq_shape.value_heads,
            )
            .unwrap();
            let (delta_core, final_state) = qwen35_recurrent_gated_delta_rule_cpu(
                &query,
                &key,
                &value,
                &g,
                &beta,
                &initial_states[seq * delta_state_len..(seq + 1) * delta_state_len],
                seq_shape.delta_shape(),
                false,
                Some(scale),
            )
            .unwrap();
            let delta_norm = qwen35_gated_rms_norm_cpu(
                &delta_core,
                &z_raw[value_start..value_end],
                &norm_weight,
                seq_shape.tokens,
                seq_shape.value_heads,
                seq_shape.value_dim,
                eps,
            )
            .unwrap();

            expected_query.extend(query);
            expected_key.extend(key);
            expected_value.extend(value);
            expected_g.extend(g);
            expected_beta.extend(beta);
            expected_final_conv_states.extend(final_conv_state);
            expected_delta_core.extend(delta_core);
            expected_delta_norm.extend(delta_norm);
            expected_final_states.extend(final_state);
        }

        let mut ctx = B::new_context();
        let output = qwen35_linear_attention_prefill_varlen_core_backend::<B>(
            &mut ctx,
            &B::from_slice_typed(&mixed_qkv_raw),
            &B::from_slice_typed(&z_raw),
            &B::from_slice_typed(&a_raw),
            &B::from_slice_typed(&b_raw),
            &B::from_slice_typed(&conv1d_weight),
            &B::from_slice_typed(&initial_conv_states),
            &B::from_slice_typed(&a_log),
            &B::from_slice_typed(&dt_bias),
            &B::from_slice_typed(&norm_weight),
            &B::from_slice_typed(&initial_states),
            &B::from_slice_typed(&cu_seqlens),
            batch,
            shape,
            eps,
        )
        .unwrap();

        assert_close_slice(
            &B::to_vec(&output.query, expected_query.len()),
            &expected_query,
            1e-6,
        );
        assert_close_slice(
            &B::to_vec(&output.key, expected_key.len()),
            &expected_key,
            1e-6,
        );
        assert_close_slice(
            &B::to_vec(&output.value, expected_value.len()),
            &expected_value,
            1e-6,
        );
        assert_close_slice(&B::to_vec(&output.g, expected_g.len()), &expected_g, 1e-6);
        assert_close_slice(
            &B::to_vec(&output.beta, expected_beta.len()),
            &expected_beta,
            1e-6,
        );
        assert_close_slice(
            &B::to_vec(&output.final_conv_states, expected_final_conv_states.len()),
            &expected_final_conv_states,
            1e-6,
        );
        assert_close_slice(
            &B::to_vec(&output.delta_core, expected_delta_core.len()),
            &expected_delta_core,
            1e-5,
        );
        assert_close_slice(
            &B::to_vec(&output.delta_norm, expected_delta_norm.len()),
            &expected_delta_norm,
            1e-5,
        );
        assert_close_slice(
            &B::to_vec(&output.final_states, expected_final_states.len()),
            &expected_final_states,
            1e-5,
        );
    }

    #[test]
    fn linear_attention_prefill_varlen_backend_matches_per_sequence_stateful_reference() {
        assert_linear_attention_prefill_varlen_backend_matches_per_sequence_stateful_reference::<
            CpuBackend,
        >();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn linear_attention_prefill_varlen_cuda_backend_matches_per_sequence_stateful_reference() {
        use ferrum_kernels::backend::cuda::CudaBackend;

        assert_linear_attention_prefill_varlen_backend_matches_per_sequence_stateful_reference::<
            CudaBackend,
        >();
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
