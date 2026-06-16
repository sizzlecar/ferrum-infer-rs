//! Llama-family decoder model as explicit code.
//!
//! Covers all "standard Llama-style GQA + SwiGLU + RoPE" decoders:
//!   - Llama / Llama-2 / Llama-3  (no QK-norm)
//!   - Qwen2 / Qwen2.5            (no QK-norm, structurally Llama)
//!   - Qwen3                      (QK-norm per head, larger rope_theta)
//!   - Mistral                    (sliding-window attention — not yet
//!                                 supported on the forward path; will
//!                                 require an `AttnConfig.sliding_window`
//!                                 field + shader support in Phase D)
//!
//! Variant differences are controlled by `LlamaFamilyConfig::has_qk_norm`
//! and RoPE theta. Weight loading accepts both fused (`qkv_proj`,
//! `gate_up_proj`) and split (`q_proj`+`k_proj`+`v_proj`,
//! `gate_proj`+`up_proj`) projection layouts — the loader (e.g.
//! `CandleVarBuilderLoader`) fuses split weights on load so model code
//! sees a uniform `qkv_proj` / `gate_up_proj` Linear.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::AtomicU64;

use ferrum_interfaces::{
    kv_dtype::{KvDtypeKind, KvFp16, KvInt8},
    model_executor::LogitsReturnPolicy,
};
use ferrum_kernels::backend::{
    Backend, BackendGraph, BackendInt8KvOps, BackendMoeFused, BackendPagedKv, BackendQuantGguf,
    BackendQuantMarlin, KvCache, KvLayer, LlmBackend, MoeLlmBackend, QuantLlmBackend,
    MAX_LAYERS_FOR_GRAPH,
};

/// Graph cache key for the single-item decode path (`decode_internal`).
/// Distinct from any `m_padded`-based key used by the batched path.
pub(crate) const SINGLE_ITEM_GRAPH_KEY: u64 = 0;

/// Diag counters for the batched graph dispatcher (replay vs eager).
pub(crate) static BATCHED_GRAPH_REPLAY_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static BATCHED_GRAPH_EAGER_COUNT: AtomicU64 = AtomicU64::new(0);

pub(crate) static ATTN_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static ATTN_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static QKR_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static QKR_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MATMUL_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MATMUL_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static NORM_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static NORM_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static OTHER_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static OTHER_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static TAIL_NORM_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static TAIL_NORM_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static TAIL_GATE_UP_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static TAIL_GATE_UP_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static TAIL_DOWN_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static TAIL_DOWN_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static TAIL_ACT_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static TAIL_ACT_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static TAIL_RESID_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static TAIL_RESID_CALLS: AtomicU64 = AtomicU64::new(0);
use ferrum_quantization::{Linear, WeightLoader};
use ferrum_types::{Activation, Result};

use crate::common::paged_pool::block_hash_chain;
use crate::common::{DecoderOnlyLLM, LlmRuntimeConfig};
use crate::lora::{load_runtime_lora_adapter, ActiveLoraAdapter, RuntimeLoraAdapter};

const DEFAULT_KV_CAPACITY: usize = 512;

fn alloc_model_buffer<B: QuantLlmBackend + BackendMoeFused>(
    label: &'static str,
    len: usize,
) -> B::Buffer {
    let _ = label;
    #[cfg(feature = "cuda")]
    let _alloc_label = ferrum_kernels::backend::cuda::push_alloc_label(label);
    B::alloc(len)
}

fn alloc_model_typed_buffer<B: QuantLlmBackend + BackendMoeFused>(
    label: &'static str,
    dtype: ferrum_kernels::backend::Dtype,
    len: usize,
) -> B::Buffer {
    let _ = label;
    #[cfg(feature = "cuda")]
    let _alloc_label = ferrum_kernels::backend::cuda::push_alloc_label(label);
    B::alloc_typed(dtype, len)
}

pub(crate) fn elapsed_micros_u64_floor1(t0: std::time::Instant) -> u64 {
    t0.elapsed().as_micros().min(u64::MAX as u128).max(1) as u64
}

#[derive(Debug, Clone, Copy)]
struct DumpStats {
    n: usize,
    finite: usize,
    nan: usize,
    posinf: usize,
    neginf: usize,
    maxabs: f32,
    maxabs_index: Option<usize>,
}

fn stats_for(values: &[f32]) -> DumpStats {
    let mut stats = DumpStats {
        n: values.len(),
        finite: 0,
        nan: 0,
        posinf: 0,
        neginf: 0,
        maxabs: 0.0,
        maxabs_index: None,
    };
    for (i, &x) in values.iter().enumerate() {
        if x.is_finite() {
            stats.finite += 1;
            let abs = x.abs();
            if abs > stats.maxabs || stats.maxabs_index.is_none() {
                stats.maxabs = abs;
                stats.maxabs_index = Some(i);
            }
        } else if x.is_nan() {
            stats.nan += 1;
        } else if x.is_sign_positive() {
            stats.posinf += 1;
        } else {
            stats.neginf += 1;
        }
    }
    stats
}

fn stat_coord(stats: DumpStats, row_width: Option<usize>) -> Option<(usize, usize)> {
    let width = row_width?;
    if width == 0 {
        return None;
    }
    let index = stats.maxabs_index?;
    Some((index / width, index % width))
}

fn append_dump_summary(dir: &str, label: &str, stats: DumpStats, row_width: Option<usize>) {
    use std::io::Write;

    let path = format!("{dir}/summary.jsonl");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let label = label.replace('\\', "\\\\").replace('"', "\\\"");
        let maxabs_index = stats
            .maxabs_index
            .map(|i| i.to_string())
            .unwrap_or_else(|| "null".to_string());
        if let Some((row, col)) = stat_coord(stats, row_width) {
            let _ = writeln!(
                f,
                "{{\"label\":\"{}\",\"n\":{},\"finite\":{},\"nan\":{},\"posinf\":{},\"neginf\":{},\"maxabs\":{},\"maxabs_index\":{},\"row_width\":{},\"maxabs_row\":{},\"maxabs_col\":{}}}",
                label,
                stats.n,
                stats.finite,
                stats.nan,
                stats.posinf,
                stats.neginf,
                stats.maxabs,
                maxabs_index,
                row_width.unwrap_or(0),
                row,
                col
            );
        } else {
            let _ = writeln!(
                f,
                "{{\"label\":\"{}\",\"n\":{},\"finite\":{},\"nan\":{},\"posinf\":{},\"neginf\":{},\"maxabs\":{},\"maxabs_index\":{}}}",
                label,
                stats.n,
                stats.finite,
                stats.nan,
                stats.posinf,
                stats.neginf,
                stats.maxabs,
                maxabs_index
            );
        }
    }
}

fn write_f32_dump(dir: &str, label: &str, values: &[f32]) {
    write_f32_dump_with_width(dir, label, values, None);
}

fn write_f32_dump_with_width(dir: &str, label: &str, values: &[f32], row_width: Option<usize>) {
    let _ = std::fs::create_dir_all(dir);
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let _ = std::fs::write(format!("{dir}/{label}.bin"), bytes);
    append_dump_summary(dir, label, stats_for(values), row_width);
}

fn nan_trace_raw_matches(raw: &str, local_layer: usize, source_layer: usize) -> bool {
    LlamaNanTraceConfig::from_raw(raw).matches_layer(local_layer, source_layer)
}

fn parse_nan_trace_layer_selector(part: &str) -> Option<usize> {
    let part = part.trim();
    if part.is_empty() {
        return None;
    }
    if let Ok(layer) = part.parse::<usize>() {
        return Some(layer);
    }
    for prefix in ["layer", "source", "l"] {
        if part.len() > prefix.len() && part[..prefix.len()].eq_ignore_ascii_case(prefix) {
            return part[prefix.len()..].parse::<usize>().ok();
        }
    }
    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LlamaNanTraceConfig {
    Disabled,
    All,
    Layers(Vec<usize>),
}

impl Default for LlamaNanTraceConfig {
    fn default() -> Self {
        Self::Disabled
    }
}

impl LlamaNanTraceConfig {
    fn from_raw(raw: &str) -> Self {
        let raw = raw.trim();
        if raw.is_empty() || raw == "1" || raw.eq_ignore_ascii_case("all") {
            return Self::All;
        }
        if raw == "0" || raw.eq_ignore_ascii_case("false") {
            return Self::Disabled;
        }
        let layers: Vec<_> = raw
            .split(',')
            .filter_map(parse_nan_trace_layer_selector)
            .collect();
        if layers.is_empty() {
            Self::Disabled
        } else {
            Self::Layers(layers)
        }
    }

    fn matches_layer(&self, local_layer: usize, source_layer: usize) -> bool {
        match self {
            Self::Disabled => false,
            Self::All => true,
            Self::Layers(layers) => layers
                .iter()
                .any(|&layer| layer == local_layer || layer == source_layer),
        }
    }
}

fn nan_trace_dump<B: Backend>(
    ctx: &mut B::Context,
    source_layer: usize,
    op: &str,
    buf: &B::Buffer,
    len: usize,
    row_width: Option<usize>,
    op_dump_dir: Option<&str>,
) {
    B::sync(ctx);
    let v = B::to_vec(buf, len);
    let stats = stats_for(&v);
    let maxabs_index = stats
        .maxabs_index
        .map(|i| i.to_string())
        .unwrap_or_else(|| "none".to_string());
    if let Some((row, col)) = stat_coord(stats, row_width) {
        eprintln!(
            "[nan-trace] layer={} op={} n={} finite={} nan={} posinf={} neginf={} maxabs={:.3} maxidx={} row_width={} row={} col={}",
            source_layer,
            op,
            stats.n,
            stats.finite,
            stats.nan,
            stats.posinf,
            stats.neginf,
            stats.maxabs,
            maxabs_index,
            row_width.unwrap_or(0),
            row,
            col
        );
    } else {
        eprintln!(
            "[nan-trace] layer={} op={} n={} finite={} nan={} posinf={} neginf={} maxabs={:.3} maxidx={}",
            source_layer,
            op,
            stats.n,
            stats.finite,
            stats.nan,
            stats.posinf,
            stats.neginf,
            stats.maxabs,
            maxabs_index
        );
    }
    if let Some(dir) = op_dump_dir {
        let safe_op = op.replace('/', "_");
        write_f32_dump_with_width(
            dir,
            &format!("layer_{source_layer:02}_{safe_op}"),
            &v,
            row_width,
        );
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct LlamaStageHiddenBridgeTiming {
    pub(crate) bridge_us: u64,
    pub(crate) host_copy_us: u64,
    pub(crate) device_copy_us: u64,
}

impl LlamaStageHiddenBridgeTiming {
    pub(crate) fn add(self, other: Self) -> Self {
        Self {
            bridge_us: self.bridge_us.saturating_add(other.bridge_us),
            host_copy_us: self.host_copy_us.saturating_add(other.host_copy_us),
            device_copy_us: self.device_copy_us.saturating_add(other.device_copy_us),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LlamaFamilyRuntimeEnv {
    pub(crate) kv_capacity: Option<usize>,
    pub(crate) metal_paged_kv: Option<bool>,
    pub(crate) paged_max_seqs: usize,
    pub(crate) decode_op_profile: bool,
    pub(crate) prefill_op_profile: bool,
    pub(crate) prefix_cache: bool,
    pub(crate) cuda_graph: bool,
    pub(crate) decode_layer_profile: bool,
    pub(crate) nan_trace: LlamaNanTraceConfig,
    /// `FERRUM_OP_DUMP=<dir>`: when nan tracing is enabled, write each traced
    /// op buffer as f32 little-endian `.bin` plus summary JSONL. Debug-only.
    pub(crate) op_dump_dir: Option<String>,
    /// `FERRUM_LAYER_DUMP=<dir>`: write per-layer prefill residuals (f32
    /// little-endian .bin) for numerical comparison against HF
    /// transformers (W2 new-family L1 gate). Debug-only — syncs per layer.
    pub(crate) layer_dump_dir: Option<String>,
}

impl LlamaFamilyRuntimeEnv {
    /// Resolve from the process-wide snapshot installed at the composition root
    /// (was a direct `std::env::vars()` read). The model holds this in a
    /// `runtime_env` field, resolved once at construction.
    fn from_env() -> Self {
        Self::from_runtime_config_snapshot(&ferrum_types::active_runtime_snapshot())
    }

    fn from_runtime_config_snapshot(snapshot: &ferrum_types::RuntimeConfigSnapshot) -> Self {
        Self::from_env_vars(
            snapshot
                .entries
                .iter()
                .map(|e| (e.key.as_str(), e.effective_value.as_str())),
        )
    }

    fn from_env_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut config = Self {
            kv_capacity: None,
            metal_paged_kv: None,
            paged_max_seqs: 32,
            decode_op_profile: false,
            prefill_op_profile: false,
            prefix_cache: false,
            cuda_graph: false,
            decode_layer_profile: false,
            nan_trace: LlamaNanTraceConfig::Disabled,
            op_dump_dir: None,
            layer_dump_dir: None,
        };
        for (name, value) in vars {
            let value = value.as_ref();
            match name.as_ref() {
                "FERRUM_KV_CAPACITY" => config.kv_capacity = value.parse::<usize>().ok(),
                "FERRUM_METAL_PAGED_KV" => config.metal_paged_kv = Some(value != "0"),
                "FERRUM_PAGED_MAX_SEQS" => {
                    if let Ok(max_seqs) = value.parse::<usize>() {
                        config.paged_max_seqs = max_seqs;
                    }
                }
                "FERRUM_DECODE_OP_PROFILE" => config.decode_op_profile = true,
                "FERRUM_PREFILL_OP_PROFILE" => config.prefill_op_profile = true,
                "FERRUM_PREFIX_CACHE" => config.prefix_cache = value == "1",
                "FERRUM_CUDA_GRAPH" => config.cuda_graph = true,
                "FERRUM_DECODE_LAYER_PROFILE" => config.decode_layer_profile = true,
                "FERRUM_NAN_TRACE" => config.nan_trace = LlamaNanTraceConfig::from_raw(value),
                "FERRUM_OP_DUMP" => config.op_dump_dir = Some(value.to_string()),
                "FERRUM_LAYER_DUMP" => config.layer_dump_dir = Some(value.to_string()),
                _ => {}
            }
        }
        config
    }

    pub(crate) fn kv_capacity_for_model(&self, model_max: usize) -> usize {
        self.kv_capacity
            .map(|cap| cap.min(model_max))
            .unwrap_or_else(|| model_max.min(DEFAULT_KV_CAPACITY))
    }

    fn paged_kv_enabled<B: BackendPagedKv>(&self) -> bool {
        self.metal_paged_kv
            .unwrap_or_else(|| B::supports_paged_kv())
    }

    pub(crate) fn graph_capture_allowed(&self) -> bool {
        !self.decode_op_profile
            && !self.prefill_op_profile
            && !self.decode_layer_profile
            && self.nan_trace == LlamaNanTraceConfig::Disabled
            && self.op_dump_dir.is_none()
            && self.layer_dump_dir.is_none()
    }
}

fn paged_kv_allowed_for_layer_schedule(
    paged_enabled: bool,
    backend_supports_varlen_qkv: bool,
    sliding_window_pattern: usize,
) -> bool {
    paged_enabled && (sliding_window_pattern == 0 || backend_supports_varlen_qkv)
}

/// Whether decode-op profiling is enabled, resolved from the runtime snapshot
/// installed at the composition root. Used by pipeline paths that operate on a
/// `LlamaFamilyPipelineModel` and so don't hold a `LlamaFamilyModel`'s
/// `runtime_env` field. Called once per decode-batch, not per op.
pub(crate) fn llama_family_decode_op_profile_enabled() -> bool {
    LlamaFamilyRuntimeEnv::from_env().decode_op_profile
}

#[derive(Clone, Debug, PartialEq)]
pub enum RopeScalingConfig {
    /// Meta Llama 3.1/3.2/3.3 long-context RoPE scaling.
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: f64,
    },
    /// Plain positional interpolation (`rope_type: "linear"`): every
    /// frequency divided by `factor`. Gemma 3 4B+ applies this (factor 8)
    /// to the global-attention layers only.
    Linear { factor: f64 },
}

impl RopeScalingConfig {
    pub fn llama3_default() -> Self {
        Self::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192.0,
        }
    }
}

/// Full Qwen3 architecture config (everything the model code needs, not just
/// the engine-facing subset in `LlmRuntimeConfig`).
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaFamilyConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScalingConfig>,
    /// GGUF LLaMA stores Q/K in the llama.cpp interleaved RoPE layout.
    /// HF safetensors Qwen/LLaMA definitions keep the default half-split
    /// GPT-NeoX layout used by existing Ferrum kernels.
    pub rope_interleaved: bool,
    /// Whether the checkpoint has `q_norm` / `k_norm` per layer. All known
    /// Qwen3 checkpoints do; some derivatives may strip them.
    pub has_qk_norm: bool,
    /// Sliding-window attention size. `0` disables (full causal).
    /// Mistral v0.1 sets 4096; Mistral v0.2+ removed the limitation (0).
    pub sliding_window: usize,
    /// MLP activation. All pre-Gemma families use SiLU (SwiGLU); Gemma 3
    /// uses `gelu_pytorch_tanh` (GeGLU).
    pub activation: Activation,
    /// Layer-pattern period for local/global attention scheduling.
    /// `0` = uniform (every layer uses `sliding_window` as-is). Gemma 3
    /// sets 6: every 6th layer (`(idx+1) % 6 == 0`) is global (full
    /// attention, main rope table); the rest are local (sliding window,
    /// local rope table).
    pub sliding_window_pattern: usize,
    /// RoPE θ for local-attention layers when the family keeps two rope
    /// tables (Gemma 3: 10_000 local vs 1_000_000 global). `None` = one
    /// table for all layers. The local table is never rope-scaled.
    pub rope_local_theta: Option<f64>,
    /// Softmax scale numerator override: attention scale becomes
    /// `1/sqrt(query_pre_attn_scalar)` instead of `1/sqrt(head_dim)`.
    /// Folded into the loaded `q_norm` weights (requires `has_qk_norm`),
    /// so kernels stay untouched. Gemma 3 27B: 168 (head_dim is 128).
    pub query_pre_attn_scalar: Option<f64>,
    /// Gemma RMSNorm convention `x̂·(1+w)`: all norm weights get +1 folded
    /// in at load time (kernels keep computing `x̂·w`).
    pub rmsnorm_unit_offset: bool,
    /// Gemma sandwich norms: `post_attention_layernorm` is applied to the
    /// attention output BEFORE the residual add (not pre-MLP), and a
    /// `post_feedforward_layernorm` wraps the MLP output. The pre-MLP
    /// norm is `pre_feedforward_layernorm`.
    pub sandwich_norms: bool,
    /// Multiply embedding output by this constant (Gemma: √hidden_size,
    /// rounded through bf16 like HF does).
    pub embed_scale: Option<f32>,
}

impl Default for LlamaFamilyConfig {
    /// Zeroed shape with legacy family semantics (SiLU, single rope table,
    /// uniform window, plain RMSNorm). Exists so config literals can fill
    /// only the fields they care about via struct-update syntax.
    fn default() -> Self {
        Self {
            hidden_size: 0,
            intermediate_size: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            num_layers: 0,
            vocab_size: 0,
            max_seq_len: 0,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_scaling: None,
            rope_interleaved: false,
            has_qk_norm: false,
            sliding_window: 0,
            activation: Activation::SiLU,
            sliding_window_pattern: 0,
            rope_local_theta: None,
            query_pre_attn_scalar: None,
            rmsnorm_unit_offset: false,
            sandwich_norms: false,
            embed_scale: None,
        }
    }
}

impl LlamaFamilyConfig {
    pub fn to_runtime(&self) -> LlmRuntimeConfig {
        LlmRuntimeConfig {
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_seq_len,
        }
    }

    /// Build config from a `ModelDefinition`, shared field extraction.
    /// Variant-specific constructors below set `has_qk_norm` and fall back
    /// to different `rope_theta` defaults when the checkpoint doesn't set one.
    fn from_def_base(def: &crate::definition::ModelDefinition) -> LlamaFamilyConfigBase {
        let num_kv_heads = def.num_key_value_heads.unwrap_or(def.num_attention_heads);
        let head_dim = def
            .extra_params
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(def.hidden_size / def.num_attention_heads);
        // Mistral / Gemma: "sliding_window" may be null (v0.2+) or a positive
        // integer (v0.1). Non-null value passes through; missing/null → 0.
        let sliding_window = def
            .extra_params
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        LlamaFamilyConfigBase {
            hidden_size: def.hidden_size,
            intermediate_size: def.intermediate_size,
            num_heads: def.num_attention_heads,
            num_kv_heads,
            head_dim,
            num_layers: def.num_hidden_layers,
            vocab_size: def.vocab_size,
            max_seq_len: effective_max_seq_len(def),
            rms_norm_eps: def.norm_eps as f32,
            rope_theta_opt: def.rope_theta,
            rope_scaling: rope_scaling_from_model_def(def),
            sliding_window,
        }
    }

    fn from_base(b: LlamaFamilyConfigBase, rope_default: f64, has_qk_norm: bool) -> Self {
        Self {
            hidden_size: b.hidden_size,
            intermediate_size: b.intermediate_size,
            num_heads: b.num_heads,
            num_kv_heads: b.num_kv_heads,
            head_dim: b.head_dim,
            num_layers: b.num_layers,
            vocab_size: b.vocab_size,
            max_seq_len: b.max_seq_len,
            rms_norm_eps: b.rms_norm_eps,
            rope_theta: b.rope_theta_opt.unwrap_or(rope_default),
            rope_scaling: b.rope_scaling,
            rope_interleaved: false,
            has_qk_norm,
            sliding_window: b.sliding_window,
            activation: Activation::SiLU,
            sliding_window_pattern: 0,
            rope_local_theta: None,
            query_pre_attn_scalar: None,
            rmsnorm_unit_offset: false,
            sandwich_norms: false,
            embed_scale: None,
        }
    }

    /// Qwen3: QK-norm on, rope_theta default 1e6.
    pub fn qwen3_from_def(def: &crate::definition::ModelDefinition) -> Self {
        Self::from_base(Self::from_def_base(def), 1_000_000.0, true)
    }

    /// Llama / Llama-2 / Llama-3: no QK-norm; rope_theta varies by version
    /// (10k for Llama-2, 500k for Llama-3.1+) — use the checkpoint value or
    /// fall back to the most common modern value.
    pub fn llama_from_def(def: &crate::definition::ModelDefinition) -> Self {
        Self::from_base(Self::from_def_base(def), 500_000.0, false)
    }

    /// Qwen2 / Qwen2.5: structurally Llama; no QK-norm; rope_theta default 1e6.
    pub fn qwen2_from_def(def: &crate::definition::ModelDefinition) -> Self {
        Self::from_base(Self::from_def_base(def), 1_000_000.0, false)
    }

    /// Mistral: no QK-norm; `rope_theta` commonly 10_000 (v0.1 / v0.2).
    /// Picks up `sliding_window` from the checkpoint's config.json
    /// (Mistral v0.1: 4096; Mistral v0.2+: 0 / null).
    pub fn mistral_from_def(def: &crate::definition::ModelDefinition) -> Self {
        Self::from_base(Self::from_def_base(def), 10_000.0, false)
    }

    /// Gemma 3 text family (1B/4B/12B/27B): QK-norm on (Gemma-style),
    /// global rope θ default 1e6 + optional linear scaling; local layers
    /// get their own unscaled table (`rope_local_base_freq`, default 1e4);
    /// 5:1 local/global scheduling (`sliding_window_pattern`, default 6);
    /// GeGLU; sandwich norms; RMSNorm `x̂·(1+w)`; embeddings ×√hidden
    /// (bf16-rounded like HF); attention scale 1/√query_pre_attn_scalar.
    pub fn gemma3_from_def(def: &crate::definition::ModelDefinition) -> Self {
        let mut cfg = Self::from_base(Self::from_def_base(def), 1_000_000.0, true);
        cfg.activation = def.activation;
        let extra = |key: &str| def.extra_params.get(key).and_then(|v| v.as_f64());
        cfg.rope_local_theta = Some(extra("rope_local_base_freq").unwrap_or(10_000.0));
        cfg.sliding_window_pattern = extra("sliding_window_pattern").unwrap_or(6.0) as usize;
        cfg.query_pre_attn_scalar = extra("query_pre_attn_scalar");
        cfg.rmsnorm_unit_offset = true;
        cfg.sandwich_norms = true;
        // HF computes `normalizer = hidden_size**0.5` then casts it to the
        // embedding dtype (bf16) before multiplying — replicate the
        // rounding for L1 byte-equality.
        cfg.embed_scale = Some(bf16_round((cfg.hidden_size as f64).sqrt() as f32));
        cfg
    }
}

/// Round an f32 through bf16 (round-to-nearest-even), back to f32.
pub(crate) fn bf16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

struct LlamaFamilyConfigBase {
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    max_seq_len: usize,
    rms_norm_eps: f32,
    rope_theta_opt: Option<f64>,
    rope_scaling: Option<RopeScalingConfig>,
    sliding_window: usize,
}

/// Models whose long context depends on a rope_scaling variant ferrum does
/// not implement (e.g. YaRN on DeepSeek-R1-0528-Qwen3-8B) must not claim the
/// scaled window: positions past the original training window would rotate
/// with unscaled frequencies and produce garbage. Clamp `max_seq_len` to the
/// declared original window and warn.
fn effective_max_seq_len(def: &crate::definition::ModelDefinition) -> usize {
    let configured = def.max_position_embeddings;
    let Some(obj) = def
        .extra_params
        .get("rope_scaling")
        .and_then(|v| v.as_object())
    else {
        return configured;
    };
    let rope_type = obj
        .get("rope_type")
        .or_else(|| obj.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if rope_type == "llama3" || rope_type == "linear" {
        // Implemented — the scaled window is real.
        return configured;
    }
    let original = obj
        .get("original_max_position_embeddings")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    match original {
        Some(original) if original < configured => {
            tracing::warn!(
                "rope_scaling type '{rope_type}' is not implemented; clamping \
                 context window from {configured} to the original training \
                 window {original}"
            );
            original
        }
        _ => configured,
    }
}

fn rope_scaling_from_model_def(
    def: &crate::definition::ModelDefinition,
) -> Option<RopeScalingConfig> {
    let value = def.extra_params.get("rope_scaling")?;
    let obj = value.as_object()?;
    let rope_type = obj
        .get("rope_type")
        .or_else(|| obj.get("type"))
        .and_then(|v| v.as_str())?;
    if rope_type == "linear" {
        let factor = json_f64(obj.get("factor"))?;
        if factor <= 0.0 {
            return None;
        }
        return Some(RopeScalingConfig::Linear { factor });
    }
    if rope_type != "llama3" {
        return None;
    }
    let factor = json_f64(obj.get("factor"))?;
    let low_freq_factor = json_f64(obj.get("low_freq_factor"))?;
    let high_freq_factor = json_f64(obj.get("high_freq_factor"))?;
    let original_max_position_embeddings = json_f64(obj.get("original_max_position_embeddings"))
        .or_else(|| {
            def.extra_params
                .get("original_max_position_embeddings")
                .and_then(|v| json_f64(Some(v)))
        })
        .unwrap_or(8192.0);
    if factor <= 0.0
        || low_freq_factor <= 0.0
        || high_freq_factor <= low_freq_factor
        || original_max_position_embeddings <= 0.0
    {
        return None;
    }
    Some(RopeScalingConfig::Llama3 {
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max_position_embeddings,
    })
}

fn json_f64(value: Option<&serde_json::Value>) -> Option<f64> {
    match value? {
        serde_json::Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

/// Per-layer weights. `Box<dyn Linear<B>>` means each projection can be
/// Dense / GPTQ / AWQ / GGUF without the surrounding code caring.
pub struct LlamaFamilyLayer<B: QuantLlmBackend + BackendMoeFused> {
    pub input_ln_w: B::Buffer,
    pub qkv_proj: Box<dyn Linear<B>>,
    /// QK-norm weight per head: `[head_dim]`. Optional for non-Qwen3 derivatives.
    pub q_norm_w: Option<B::Buffer>,
    pub k_norm_w: Option<B::Buffer>,
    pub o_proj: Box<dyn Linear<B>>,
    /// Pre-MLP norm. Llama naming: `post_attention_layernorm`; Gemma 3
    /// sandwich naming: `pre_feedforward_layernorm`.
    pub post_ln_w: B::Buffer,
    /// Gemma 3 sandwich: norm applied to the attention output BEFORE the
    /// residual add (`post_attention_layernorm` in Gemma naming).
    pub post_attn_ln_w: Option<B::Buffer>,
    /// Gemma 3 sandwich: norm applied to the MLP output before the
    /// residual add (`post_feedforward_layernorm`).
    pub post_ffn_ln_w: Option<B::Buffer>,
    pub gate_up_proj: Box<dyn Linear<B>>,
    pub down_proj: Box<dyn Linear<B>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaFamilyLayerStageConfig {
    pub source_layers: Range<usize>,
    pub load_embedding: bool,
    pub load_lm_head: bool,
}

impl LlamaFamilyLayerStageConfig {
    pub fn full(num_layers: usize) -> Self {
        Self {
            source_layers: 0..num_layers,
            load_embedding: true,
            load_lm_head: true,
        }
    }

    pub fn backbone(num_layers: usize) -> Self {
        Self {
            source_layers: 0..num_layers,
            load_embedding: false,
            load_lm_head: false,
        }
    }

    pub fn pipeline_stage(
        source_layers: Range<usize>,
        is_first_stage: bool,
        is_last_stage: bool,
    ) -> Self {
        Self {
            source_layers,
            load_embedding: is_first_stage,
            load_lm_head: is_last_stage,
        }
    }
}

fn load_llama_family_layers<B: MoeLlmBackend>(
    cfg: &LlamaFamilyConfig,
    loader: &dyn WeightLoader<B>,
    source_layers: Range<usize>,
) -> Result<Vec<LlamaFamilyLayer<B>>> {
    if source_layers.start > source_layers.end || source_layers.end > cfg.num_layers {
        return Err(ferrum_types::FerrumError::model(format!(
            "llama layer range {}..{} is outside model layer count {}",
            source_layers.start, source_layers.end, cfg.num_layers
        )));
    }

    let mut layers = Vec::with_capacity(source_layers.end.saturating_sub(source_layers.start));
    // Gemma 3 attention scale is 1/√query_pre_attn_scalar while the kernels
    // hardcode 1/√head_dim; the ratio is folded into q_norm (norm output is
    // scale-equivariant, RoPE is a rotation, so the fold commutes).
    let q_norm_scale = cfg
        .query_pre_attn_scalar
        .map(|s| ((cfg.head_dim as f64 / s).sqrt()) as f32)
        .unwrap_or(1.0);
    for li in source_layers {
        let prefix = format!("model.layers.{li}");
        let norm = |name: &str, len: usize, scale: f32| -> Result<B::Buffer> {
            let t = loader.load_tensor(&format!("{prefix}.{name}.weight"))?;
            Ok(fold_norm_weight::<B>(
                t,
                len,
                cfg.rmsnorm_unit_offset,
                scale,
            ))
        };
        let h = cfg.hidden_size;
        let input_ln_w = norm("input_layernorm", h, 1.0)?;
        let qkv_proj = loader.load_linear(&format!("{prefix}.self_attn.qkv_proj"))?;
        let o_proj = loader.load_linear(&format!("{prefix}.self_attn.o_proj"))?;
        let (post_ln_w, post_attn_ln_w, post_ffn_ln_w) = if cfg.sandwich_norms {
            (
                norm("pre_feedforward_layernorm", h, 1.0)?,
                Some(norm("post_attention_layernorm", h, 1.0)?),
                Some(norm("post_feedforward_layernorm", h, 1.0)?),
            )
        } else {
            (norm("post_attention_layernorm", h, 1.0)?, None, None)
        };
        let gate_up_proj = loader.load_linear(&format!("{prefix}.mlp.gate_up_proj"))?;
        let down_proj = loader.load_linear(&format!("{prefix}.mlp.down_proj"))?;

        let (q_norm_w, k_norm_w) = if cfg.has_qk_norm {
            let q = norm("self_attn.q_norm", cfg.head_dim, q_norm_scale).ok();
            let k = norm("self_attn.k_norm", cfg.head_dim, 1.0).ok();
            (q, k)
        } else {
            (None, None)
        };

        layers.push(LlamaFamilyLayer {
            input_ln_w,
            qkv_proj,
            q_norm_w,
            k_norm_w,
            o_proj,
            post_ln_w,
            post_attn_ln_w,
            post_ffn_ln_w,
            gate_up_proj,
            down_proj,
        });
    }
    Ok(layers)
}

/// Apply Gemma-family load-time weight folds to a norm tensor:
/// `w' = (w + 1)·scale` when `unit_offset` (Gemma RMSNorm is `x̂·(1+w)`),
/// plus an optional extra scale (query_pre_attn_scalar fold on q_norm).
fn fold_norm_weight<B: MoeLlmBackend>(
    buf: B::Buffer,
    len: usize,
    unit_offset: bool,
    scale: f32,
) -> B::Buffer {
    if !unit_offset && scale == 1.0 {
        return buf;
    }
    let mut v = B::to_vec(&buf, len);
    let off = if unit_offset { 1.0 } else { 0.0 };
    for x in v.iter_mut() {
        *x = (*x + off) * scale;
    }
    B::from_slice(&v)
}

fn load_llama_family_lm_head<B: MoeLlmBackend>(
    cfg: &LlamaFamilyConfig,
    loader: &dyn WeightLoader<B>,
) -> Result<Box<dyn Linear<B>>> {
    // LM head: either dedicated `lm_head.weight` or tied to embedding.
    // Many models (Qwen3-4B, Llama-3.2-1B, some Qwen2.5) use TIED
    // embeddings — lm_head shares weights with model.embed_tokens. When
    // no dedicated lm_head tensor exists, re-load the embed tensor as a
    // DenseLinear. This duplicates the buffer (memory cost = vocab*h*2
    // bytes, e.g. ~770MB for Qwen3-4B) but keeps the Linear trait's
    // owned-weights invariant. Sharing via Arc is a future optimisation.
    let lm_head = if loader.has_tensor("lm_head.weight") {
        loader.load_linear("lm_head")?
    } else {
        tracing::info!(
            "LlamaFamilyModel: tied embeddings — loading model.embed_tokens.weight as lm_head"
        );
        let as_linear = loader.load_linear("model.embed_tokens")?;
        // Sanity check: shape must be [vocab, hidden].
        if as_linear.out_features() != cfg.vocab_size || as_linear.in_features() != cfg.hidden_size
        {
            return Err(ferrum_types::FerrumError::model(format!(
                "tied embed shape mismatch: got [{}, {}], expected [{}, {}]",
                as_linear.out_features(),
                as_linear.in_features(),
                cfg.vocab_size,
                cfg.hidden_size
            )));
        }
        as_linear
    };
    Ok(lm_head)
}

fn supports_sandwich_legacy_batched_decode(
    cfg: &LlamaFamilyConfig,
    backend_supports_device_f32_residual_shadow: bool,
) -> bool {
    cfg.sandwich_norms
        && cfg.sliding_window_pattern != 0
        && backend_supports_device_f32_residual_shadow
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LlamaLayerAttentionSchedule {
    pub is_global_layer: bool,
    pub sliding_window: usize,
}

pub(crate) fn llama_qk_mode(cfg: &LlamaFamilyConfig) -> i32 {
    if cfg.has_qk_norm {
        1
    } else if cfg.rope_interleaved {
        3
    } else {
        2
    }
}

pub(crate) fn llama_layer_attention_schedule(
    cfg: &LlamaFamilyConfig,
    source_layer_index: usize,
) -> LlamaLayerAttentionSchedule {
    let pattern = cfg.sliding_window_pattern;
    let is_global_layer = pattern == 0 || (source_layer_index + 1) % pattern == 0;
    let sliding_window = if pattern == 0 {
        cfg.sliding_window
    } else if is_global_layer {
        0
    } else {
        cfg.sliding_window
    };
    LlamaLayerAttentionSchedule {
        is_global_layer,
        sliding_window,
    }
}

fn unified_varlen_qkv_unsupported_reason(
    cfg: &LlamaFamilyConfig,
    backend_supports_varlen_qkv: bool,
    backend_supports_device_f32_residual_shadow: bool,
) -> Option<&'static str> {
    if !backend_supports_varlen_qkv {
        return Some(
            "LlamaFamilyModel::unified_forward: backend lacks varlen \
             QKV kernels. Engine will fall back to per-item dispatch.",
        );
    }
    if cfg.sandwich_norms {
        if cfg.sliding_window_pattern == 0 {
            return Some(
                "LlamaFamilyModel::unified_forward: sandwich-norm family requires \
                 an explicit local/global sliding-window layer pattern. Engine will \
                 fall back to per-item dispatch.",
            );
        }
        if !backend_supports_device_f32_residual_shadow {
            return Some(
                "LlamaFamilyModel::unified_forward: sandwich-norm family requires \
                 backend device-side F32 residual shadow support for unified \
                 GeGLU and post-attn/post-ffn sandwich norms. Engine will fall \
                 back to per-item dispatch.",
            );
        }
    }
    None
}

#[derive(Debug, Clone, Default)]
pub(crate) struct LlamaDecodeBatchStats {
    calls: u64,
    total_items: u64,
    max_items: u64,
    last_items: u64,
    force_full_logits_calls: u64,
    singleton_fast_path_calls: u64,
    unsupported_fallback_calls: u64,
    lora_fallback_calls: u64,
    m1_calls: u64,
    m2_calls: u64,
    m3_4_calls: u64,
    m5_8_calls: u64,
    m9_16_calls: u64,
    m17_32_calls: u64,
    m_gt_32_calls: u64,
}

impl LlamaDecodeBatchStats {
    pub(crate) fn record_call(&mut self, items: usize, force_full_logits: bool) {
        let items = items as u64;
        self.calls += 1;
        self.total_items += items;
        self.max_items = self.max_items.max(items);
        self.last_items = items;
        if force_full_logits {
            self.force_full_logits_calls += 1;
        }
        match items {
            0 => {}
            1 => self.m1_calls += 1,
            2 => self.m2_calls += 1,
            3..=4 => self.m3_4_calls += 1,
            5..=8 => self.m5_8_calls += 1,
            9..=16 => self.m9_16_calls += 1,
            17..=32 => self.m17_32_calls += 1,
            _ => self.m_gt_32_calls += 1,
        }
    }

    pub(crate) fn record_singleton_fast_path(&mut self) {
        self.singleton_fast_path_calls += 1;
    }

    pub(crate) fn record_unsupported_fallback(&mut self) {
        self.unsupported_fallback_calls += 1;
    }

    pub(crate) fn record_lora_fallback(&mut self) {
        self.lora_fallback_calls += 1;
    }

    fn average_items_per_call(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.total_items as f64 / self.calls as f64
        }
    }

    fn snapshot_json(&self) -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "calls": self.calls,
            "total_items": self.total_items,
            "avg_items_per_call": self.average_items_per_call(),
            "max_items": self.max_items,
            "last_items": self.last_items,
            "force_full_logits_calls": self.force_full_logits_calls,
            "singleton_fast_path_calls": self.singleton_fast_path_calls,
            "unsupported_fallback_calls": self.unsupported_fallback_calls,
            "lora_fallback_calls": self.lora_fallback_calls,
            "bucket_calls": {
                "m1": self.m1_calls,
                "m2": self.m2_calls,
                "m3_4": self.m3_4_calls,
                "m5_8": self.m5_8_calls,
                "m9_16": self.m9_16_calls,
                "m17_32": self.m17_32_calls,
                "m_gt_32": self.m_gt_32_calls,
            },
        })
    }
}

/// Precomputed RoPE cos/sin tables (shape `[max_seq, head_dim / 2]` each).
pub struct RopeCache<B: QuantLlmBackend + BackendMoeFused> {
    pub cos: B::Buffer,
    pub sin: B::Buffer,
}

/// Reusable per-layer scratch buffers sized for `max_tokens` tokens of a
/// single forward pass (prefill or decode step).
///
/// Sized lazily on first use so tiny decode steps don't pay for prefill-sized
/// buffers. Grows monotonically when a larger prefill arrives.
pub struct LlamaFamilyScratch<B: QuantLlmBackend + BackendMoeFused> {
    /// Residual stream — wrapped in Option so decode_internal can
    /// `.take()` it without needing an alloc placeholder.
    ///
    /// Why this matters for graph capture: the old pattern was
    /// `mem::replace(&mut scratch.residual, B::alloc(1))` which creates a
    /// 1-element buffer at every decode step. When graph capture is on,
    /// that alloc-during-capture + drop-after-capture pair surfaces as
    /// cuMemFreeAsync(INVALID_VALUE) because the free tries to release a
    /// pointer the captured graph may still reference. Option::take leaves
    /// None and moves the real buffer into a local — no spurious alloc.
    pub residual: Option<B::Buffer>,
    pub norm_out: B::Buffer,
    pub qkv_out: B::Buffer,
    /// `[tokens, h]` temp for sandwich-norm families (Gemma 3): holds the
    /// normed attention/MLP output before its residual add. `None` for
    /// legacy families.
    pub sandwich_tmp: Option<B::Buffer>,
    /// Device-side F32 residual shadow for sandwich-norm families on CUDA.
    /// Keeps Gemma3's residual stream finite without per-layer D2H syncs.
    pub residual_f32_shadow: Option<B::Buffer>,
    /// Device-side F32 branch scratch for post-attn/post-ffn norms and final
    /// last-token normalization staging.
    pub sandwich_branch_f32: Option<B::Buffer>,
    // ── Per-item scratch for batched decode path ──────────────────────
    // decode_batch_internal runs tokens=M batched ops for the GEMM-heavy
    // half (norm, qkv_proj, split_qkv, o_proj, post_norm, gate_up, silu,
    // down, residual_add) but must loop per-item for rope + KV append +
    // attention (each item has its own KV cache at a different kv_len).
    // These single-item buffers hold item i's slice during that loop.
    /// Item-scope q_buf slice, sized `q_dim`.
    pub q_single: B::Buffer,
    pub k_single: B::Buffer,
    pub v_single: B::Buffer,
    pub q_head_major_single: B::Buffer,
    pub k_head_major_single: B::Buffer,
    pub v_head_major_single: B::Buffer,
    pub attn_head_major_single: B::Buffer,
    pub attn_flat_single: B::Buffer,
    /// Batched logits output. Kept to one row for prefill-sized scratch and
    /// grown only by the few paths that actually materialize multi-row logits.
    /// Prefill/single-decode use the regular `logits`.
    pub batch_logits: B::Buffer,
    pub batch_logits_capacity: usize,
    pub argmax_token_mask: Option<B::Buffer>,
    pub argmax_token_mask_fingerprint: Option<u64>,
    pub argmax_token_mask_len: usize,
    /// Token-major Q/K/V right after `split_qkv`. Stride: heads * hd per row.
    pub q_buf: B::Buffer,
    pub k_buf: B::Buffer,
    pub v_buf: B::Buffer,
    /// Head-major Q produced by `qk_norm_rope` — fed into `flash_attention`.
    pub q_head_major: B::Buffer,
    /// Head-major K/V staging — produced by `qk_norm_rope`, consumed by
    /// `kv_cache_append_head_major` (no reuse after append).
    pub k_head_major: B::Buffer,
    pub v_head_major: B::Buffer,
    /// Head-major attention output from `flash_attention`.
    pub attn_head_major_out: B::Buffer,
    /// Token-major attention output after `transpose_head_to_token`.
    pub attn_flat: B::Buffer,
    pub o_proj_out: B::Buffer,
    pub gate_up_out: B::Buffer,
    pub silu_out: B::Buffer,
    pub mlp_out: B::Buffer,
    /// Paged batched dispatch scratch (Phase 4b). Sized for
    /// `FERRUM_PAGED_MAX_SEQS × q_dim` so multi-seq decode can fan
    /// in M items' Q into a single buffer for one batched
    /// `paged_decode_attention(num_seqs=M)` call. `None` when paged
    /// mode is off.
    pub paged_batch_q: Option<B::Buffer>,
    pub paged_batch_o: Option<B::Buffer>,
    /// Stacked per-seq block tables for batched paged dispatch.
    /// Layout: `[max_M, max_blocks_per_seq]` u32. Written
    /// host-side per decode_batch step.
    pub paged_batch_block_tables: Option<B::Buffer>,
    /// Stacked per-seq context lengths for batched paged dispatch
    /// (`[max_M]` u32).
    pub paged_batch_context_lens: Option<B::Buffer>,
    /// `max_blocks_per_seq` value baked into the stacked block_tables
    /// stride. Set when `paged_batch_block_tables` is allocated.
    pub paged_max_blocks_per_seq: usize,
    /// Engine-side max concurrent sequences (= `FERRUM_PAGED_MAX_SEQS`).
    /// Pinned at the first `enable_paged_batch` so the unified-forward
    /// scratch sizes (`unified_cu_seqlens_q`, `unified_pos_offsets`,
    /// `unified_block_tables`, `unified_packed_*`) are big enough for
    /// any subsequent batch up to that bound.
    pub paged_max_seqs: usize,
    /// Per-item RoPE positions for the batched-decode path (`[max_M]`
    /// i32 / u32). Written host-side once per batched-decode step from
    /// each request's `pos` field, read by the batched
    /// `qk_norm_rope_batched_per_item` CUDA kernel.
    pub batch_positions: B::Buffer,
    /// Per-item input token ids for the batched-decode path (`[max_M]`
    /// u32). Written once per call before forward; read by the
    /// graph-capture-friendly `embedding_lookup_batched_dyn` variant.
    pub batch_tokens: B::Buffer,
    /// Per-item KV-cache length BEFORE this step's kv_append
    /// (`[max_M]` u32). Used by `kv_cache_append_batched_per_cache_dyn`
    /// to write at the right slot for graph replay.
    pub batch_kv_lens_pre: B::Buffer,
    /// Per-item KV-cache length AFTER this step's kv_append
    /// (`[max_M]` u32 = pre + 1). Used by
    /// `flash_attention_batched_per_cache_dyn` for the attention
    /// reduce window length.
    pub batch_kv_lens_post: B::Buffer,
    /// Output buffers for the batched per-item qk_norm_rope kernel.
    /// Same shape as q_buf / k_buf / v_buf — separate so the kernel
    /// API can take `&input` and `&mut output` without aliasing.
    pub q_normed_batched: B::Buffer,
    pub k_normed_batched: B::Buffer,
    pub v_normed_batched: B::Buffer,

    // ── Unified mixed-batch scratch (chunked-prefill path; Step 5b) ─────
    // Buffers sized for `M_total = sum(items[i].q_tokens.len())`. Grown
    // on demand by `ensure_unified_scratch(M_total_max)`. Used only by
    // the new `unified_forward_internal`; legacy `forward_layer_batched_decode`
    // continues to use the per-item-stride scratch above.
    pub unified_capacity: usize, // current allocated M_total slots
    pub unified_residual: Option<B::Buffer>,
    pub unified_norm_out: Option<B::Buffer>,
    pub unified_qkv_out: Option<B::Buffer>,
    pub unified_packed_q: Option<B::Buffer>,
    pub unified_attn_out: Option<B::Buffer>,
    pub unified_o_proj_out: Option<B::Buffer>,
    pub unified_sandwich_tmp: Option<B::Buffer>,
    pub unified_residual_f32_shadow: Option<B::Buffer>,
    pub unified_sandwich_branch_f32: Option<B::Buffer>,
    pub unified_gate_up_out: Option<B::Buffer>,
    pub unified_silu_out: Option<B::Buffer>,
    pub unified_mlp_out: Option<B::Buffer>,
    /// Per-item index buffers (i32-stored-as-f16): cu_seqlens_q is
    /// length `max_seqs+1`, pos_offsets is `max_seqs`, block_tables is
    /// `max_seqs * max_blocks_per_seq`. Sized once at first use to
    /// match `paged_batch_*` capacity since they share `max_seqs`.
    pub unified_cu_seqlens_q: Option<B::Buffer>,
    pub unified_pos_offsets: Option<B::Buffer>,
    pub unified_block_tables: Option<B::Buffer>,
    /// Packed last-token hidden states for is_final_chunk items
    /// (`[num_sampled, h]`). Used as input to lm_head.
    pub unified_packed_normed: Option<B::Buffer>,
    /// Packed logits output (`[num_sampled, vocab]`).
    pub unified_packed_logits: Option<B::Buffer>,
    /// Last token's hidden state (`[h]`). For prefill this is populated via
    /// `copy_slice(residual, (seq_len-1)*h, ..)`; for decode `residual` already
    /// holds only 1 row so `last_hidden` is unused on that path.
    pub last_hidden: B::Buffer,
    /// Final-norm output for the last token (`[h]`).
    pub last_normed: B::Buffer,
    /// lm_head logits (`[vocab]`).
    pub logits: B::Buffer,
    /// The max tokens-per-step this scratch has been sized for.
    pub max_tokens: usize,
}

impl<B: QuantLlmBackend + BackendMoeFused> LlamaFamilyScratch<B> {
    fn alloc(cfg: &LlamaFamilyConfig, max_tokens: usize) -> Self {
        let h = cfg.hidden_size;
        let im = cfg.intermediate_size;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let t = max_tokens;
        Self {
            residual: Some(alloc_model_buffer::<B>("llama.scratch.residual", t * h)),
            norm_out: alloc_model_buffer::<B>("llama.scratch.norm_out", t * h),
            qkv_out: alloc_model_buffer::<B>("llama.scratch.qkv_out", t * qkv_dim),
            sandwich_tmp: cfg
                .sandwich_norms
                .then(|| alloc_model_buffer::<B>("llama.scratch.sandwich_tmp", t * h)),
            residual_f32_shadow: (cfg.sandwich_norms && B::supports_device_f32_residual_shadow())
                .then(|| {
                    alloc_model_typed_buffer::<B>(
                        "llama.scratch.residual_f32_shadow",
                        ferrum_kernels::backend::Dtype::F32,
                        t * h,
                    )
                }),
            sandwich_branch_f32: (cfg.sandwich_norms && B::supports_device_f32_residual_shadow())
                .then(|| {
                    alloc_model_typed_buffer::<B>(
                        "llama.scratch.sandwich_branch_f32",
                        ferrum_kernels::backend::Dtype::F32,
                        t * h,
                    )
                }),
            q_buf: alloc_model_buffer::<B>("llama.scratch.q_buf", t * q_dim),
            k_buf: alloc_model_buffer::<B>("llama.scratch.k_buf", t * kv_dim),
            v_buf: alloc_model_buffer::<B>("llama.scratch.v_buf", t * kv_dim),
            q_head_major: alloc_model_buffer::<B>(
                "llama.scratch.q_head_major",
                cfg.num_heads * t * cfg.head_dim,
            ),
            k_head_major: alloc_model_buffer::<B>(
                "llama.scratch.k_head_major",
                cfg.num_kv_heads * t * cfg.head_dim,
            ),
            v_head_major: alloc_model_buffer::<B>(
                "llama.scratch.v_head_major",
                cfg.num_kv_heads * t * cfg.head_dim,
            ),
            attn_head_major_out: alloc_model_buffer::<B>(
                "llama.scratch.attn_head_major_out",
                cfg.num_heads * t * cfg.head_dim,
            ),
            attn_flat: alloc_model_buffer::<B>("llama.scratch.attn_flat", t * q_dim),
            o_proj_out: alloc_model_buffer::<B>("llama.scratch.o_proj_out", t * h),
            gate_up_out: alloc_model_buffer::<B>("llama.scratch.gate_up_out", t * 2 * im),
            silu_out: alloc_model_buffer::<B>("llama.scratch.silu_out", t * im),
            mlp_out: alloc_model_buffer::<B>("llama.scratch.mlp_out", t * h),
            last_hidden: alloc_model_buffer::<B>("llama.scratch.last_hidden", h),
            last_normed: alloc_model_buffer::<B>("llama.scratch.last_normed", h),
            logits: alloc_model_buffer::<B>("llama.scratch.logits", cfg.vocab_size),
            q_single: alloc_model_buffer::<B>("llama.scratch.q_single", q_dim),
            k_single: alloc_model_buffer::<B>("llama.scratch.k_single", kv_dim),
            v_single: alloc_model_buffer::<B>("llama.scratch.v_single", kv_dim),
            q_head_major_single: alloc_model_buffer::<B>(
                "llama.scratch.q_head_major_single",
                q_dim,
            ),
            k_head_major_single: alloc_model_buffer::<B>(
                "llama.scratch.k_head_major_single",
                kv_dim,
            ),
            v_head_major_single: alloc_model_buffer::<B>(
                "llama.scratch.v_head_major_single",
                kv_dim,
            ),
            attn_head_major_single: alloc_model_buffer::<B>(
                "llama.scratch.attn_head_major_single",
                q_dim,
            ),
            attn_flat_single: alloc_model_buffer::<B>("llama.scratch.attn_flat_single", q_dim),
            batch_logits: alloc_model_buffer::<B>("llama.scratch.batch_logits", cfg.vocab_size),
            batch_logits_capacity: 1,
            argmax_token_mask: None,
            argmax_token_mask_fingerprint: None,
            argmax_token_mask_len: 0,
            // Paged batched dispatch scratch. None until `enable_paged_batch`
            // is called from `ensure_kv` once the model knows max_seqs +
            // max_blocks_per_seq. This avoids paying the alloc cost when
            // paged mode is off.
            paged_batch_q: None,
            paged_batch_o: None,
            paged_batch_block_tables: None,
            paged_batch_context_lens: None,
            paged_max_blocks_per_seq: 0,
            paged_max_seqs: 0,
            batch_positions: alloc_model_typed_buffer::<B>(
                "llama.scratch.batch_positions",
                ferrum_kernels::backend::Dtype::U32,
                t.max(1),
            ),
            batch_tokens: alloc_model_typed_buffer::<B>(
                "llama.scratch.batch_tokens",
                ferrum_kernels::backend::Dtype::U32,
                t.max(1),
            ),
            batch_kv_lens_pre: alloc_model_typed_buffer::<B>(
                "llama.scratch.batch_kv_lens_pre",
                ferrum_kernels::backend::Dtype::U32,
                t.max(1),
            ),
            batch_kv_lens_post: alloc_model_typed_buffer::<B>(
                "llama.scratch.batch_kv_lens_post",
                ferrum_kernels::backend::Dtype::U32,
                t.max(1),
            ),
            q_normed_batched: alloc_model_buffer::<B>("llama.scratch.q_normed_batched", t * q_dim),
            k_normed_batched: alloc_model_buffer::<B>("llama.scratch.k_normed_batched", t * kv_dim),
            v_normed_batched: alloc_model_buffer::<B>("llama.scratch.v_normed_batched", t * kv_dim),
            unified_capacity: 0,
            unified_residual: None,
            unified_norm_out: None,
            unified_qkv_out: None,
            unified_packed_q: None,
            unified_attn_out: None,
            unified_o_proj_out: None,
            unified_sandwich_tmp: None,
            unified_residual_f32_shadow: None,
            unified_sandwich_branch_f32: None,
            unified_gate_up_out: None,
            unified_silu_out: None,
            unified_mlp_out: None,
            unified_cu_seqlens_q: None,
            unified_pos_offsets: None,
            unified_block_tables: None,
            unified_packed_normed: None,
            unified_packed_logits: None,
            max_tokens: t,
        }
    }

    pub(crate) fn ensure_batch_logits(&mut self, cfg: &LlamaFamilyConfig, rows: usize) {
        let rows = rows.max(1);
        if rows <= self.batch_logits_capacity {
            return;
        }
        self.batch_logits =
            alloc_model_buffer::<B>("llama.scratch.batch_logits.grow", rows * cfg.vocab_size);
        self.batch_logits_capacity = rows;
    }

    pub(crate) fn ensure_argmax_token_mask(
        &mut self,
        ctx: &mut B::Context,
        mask: &ferrum_interfaces::model_executor::TokenSelectionMask,
    ) -> &B::Buffer {
        let needs_upload = self.argmax_token_mask_fingerprint != Some(mask.fingerprint)
            || self.argmax_token_mask_len != mask.len();
        if needs_upload {
            let mut device_mask = alloc_model_typed_buffer::<B>(
                "llama.scratch.argmax_token_mask",
                ferrum_kernels::backend::Dtype::I8,
                mask.len().max(1),
            );
            B::write_typed::<i8>(ctx, &mut device_mask, &mask.valid_token_mask);
            self.argmax_token_mask = Some(device_mask);
            self.argmax_token_mask_fingerprint = Some(mask.fingerprint);
            self.argmax_token_mask_len = mask.len();
        }
        self.argmax_token_mask
            .as_ref()
            .expect("argmax token mask upload failed")
    }

    /// Grow unified-path scratch buffers to accommodate `m_total` query
    /// tokens. Called lazily from `unified_forward_internal` so single-
    /// path workloads (no chunked prefill) don't pay the alloc cost.
    /// Returns true when buffers were rebuilt and any captured graph that
    /// references the previous pointers must be invalidated by the caller.
    pub(crate) fn ensure_unified_scratch(
        &mut self,
        cfg: &LlamaFamilyConfig,
        m_total: usize,
        max_seqs: usize,
        max_blocks_per_seq: usize,
    ) -> bool {
        if m_total <= self.unified_capacity
            && self.unified_residual.is_some()
            && self.unified_cu_seqlens_q.is_some()
            && (!cfg.sandwich_norms
                || (self.unified_sandwich_tmp.is_some()
                    && (!B::supports_device_f32_residual_shadow()
                        || (self.unified_residual_f32_shadow.is_some()
                            && self.unified_sandwich_branch_f32.is_some()))))
        {
            return false;
        }
        let cap = m_total.max(self.unified_capacity).max(1);
        let h = cfg.hidden_size;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let im = cfg.intermediate_size;
        let v = cfg.vocab_size;
        self.unified_residual = None;
        self.unified_norm_out = None;
        self.unified_qkv_out = None;
        self.unified_packed_q = None;
        self.unified_attn_out = None;
        self.unified_o_proj_out = None;
        self.unified_sandwich_tmp = None;
        self.unified_residual_f32_shadow = None;
        self.unified_sandwich_branch_f32 = None;
        self.unified_gate_up_out = None;
        self.unified_silu_out = None;
        self.unified_mlp_out = None;

        self.unified_residual = Some(alloc_model_buffer::<B>("llama.unified.residual", cap * h));
        self.unified_norm_out = Some(alloc_model_buffer::<B>("llama.unified.norm_out", cap * h));
        self.unified_qkv_out = Some(alloc_model_buffer::<B>(
            "llama.unified.qkv_out",
            cap * qkv_dim,
        ));
        self.unified_packed_q = Some(alloc_model_buffer::<B>(
            "llama.unified.packed_q",
            cap * q_dim,
        ));
        self.unified_attn_out = Some(alloc_model_buffer::<B>(
            "llama.unified.attn_out",
            cap * q_dim,
        ));
        self.unified_o_proj_out =
            Some(alloc_model_buffer::<B>("llama.unified.o_proj_out", cap * h));
        if cfg.sandwich_norms {
            self.unified_sandwich_tmp = Some(alloc_model_buffer::<B>(
                "llama.unified.sandwich_tmp",
                cap * h,
            ));
            if B::supports_device_f32_residual_shadow() {
                self.unified_residual_f32_shadow = Some(alloc_model_typed_buffer::<B>(
                    "llama.unified.residual_f32_shadow",
                    ferrum_kernels::backend::Dtype::F32,
                    cap * h,
                ));
                self.unified_sandwich_branch_f32 = Some(alloc_model_typed_buffer::<B>(
                    "llama.unified.sandwich_branch_f32",
                    ferrum_kernels::backend::Dtype::F32,
                    cap * h,
                ));
            }
        }
        self.unified_gate_up_out = Some(alloc_model_buffer::<B>(
            "llama.unified.gate_up_out",
            cap * 2 * im,
        ));
        self.unified_silu_out = Some(alloc_model_buffer::<B>("llama.unified.silu_out", cap * im));
        self.unified_mlp_out = Some(alloc_model_buffer::<B>("llama.unified.mlp_out", cap * h));
        if self.unified_cu_seqlens_q.is_none() {
            self.unified_cu_seqlens_q = Some(alloc_model_typed_buffer::<B>(
                "llama.unified.cu_seqlens_q",
                ferrum_kernels::backend::Dtype::U32,
                max_seqs + 1,
            ));
            self.unified_pos_offsets = Some(alloc_model_typed_buffer::<B>(
                "llama.unified.pos_offsets",
                ferrum_kernels::backend::Dtype::U32,
                max_seqs,
            ));
            self.unified_block_tables = Some(alloc_model_typed_buffer::<B>(
                "llama.unified.block_tables",
                ferrum_kernels::backend::Dtype::U32,
                max_seqs * max_blocks_per_seq,
            ));
            self.unified_packed_normed = Some(alloc_model_buffer::<B>(
                "llama.unified.packed_normed",
                max_seqs * h,
            ));
            self.unified_packed_logits = Some(alloc_model_buffer::<B>(
                "llama.unified.packed_logits",
                max_seqs * v,
            ));
        }
        self.unified_capacity = cap;
        true
    }

    /// Allocate scratch for batched paged dispatch (Phase 4b). Called
    /// lazily from `ensure_kv` once paged mode is enabled and we know
    /// the pool dimensions. Idempotent.
    fn enable_paged_batch(
        &mut self,
        cfg: &LlamaFamilyConfig,
        max_seqs: usize,
        max_blocks_per_seq: usize,
    ) {
        if self.paged_batch_q.is_some() {
            return;
        }
        let q_dim = cfg.num_heads * cfg.head_dim;
        self.paged_batch_q = Some(alloc_model_buffer::<B>(
            "llama.paged_batch.q",
            max_seqs * q_dim,
        ));
        self.paged_batch_o = Some(alloc_model_buffer::<B>(
            "llama.paged_batch.o",
            max_seqs * q_dim,
        ));
        self.paged_batch_block_tables = Some(alloc_model_typed_buffer::<B>(
            "llama.paged_batch.block_tables",
            ferrum_kernels::backend::Dtype::U32,
            max_seqs * max_blocks_per_seq,
        ));
        self.paged_batch_context_lens = Some(alloc_model_typed_buffer::<B>(
            "llama.paged_batch.context_lens",
            ferrum_kernels::backend::Dtype::U32,
            max_seqs,
        ));
        self.paged_max_blocks_per_seq = max_blocks_per_seq;
        self.paged_max_seqs = max_seqs;
    }
}

/// Qwen3 model — decoder-only LLM, one per (backend, weights) combination.
///
/// Holds all parameters, scratch space, RoPE cache, and per-sequence KV caches.
///
/// `B: BackendGraph + BackendQuantMarlin + BackendQuantGguf` because the decode hot path uses CUDA Graph capture/replay
/// when the backend supports it; non-graph backends (Metal/CPU) inherit no-op
/// defaults, so this bound is satisfied by every concrete `Backend`.
///
/// `K: KvDtypeKind = KvFp16` (Dim 5): selects the KV cache element type.
/// `K = KvFp16` constructs only `LayerKvCache::Fp16(...)` variants;
/// `K = KvInt8` constructs only `LayerKvCache::Int8(...)` variants. The
/// `B: BackendKvDtype<KvInt8>` bound is structurally required by the
/// enum and is satisfied by all backends via stub impls (Cpu/Metal) or
/// the real impl (CUDA).
pub struct LlamaFamilyModel<B: MoeLlmBackend, K: KvLayer<B> = KvFp16> {
    pub cfg: LlamaFamilyConfig,
    pub runtime_cfg: LlmRuntimeConfig,

    /// Backend capabilities resolved once at construction (the GOAL's allowed
    /// "构造期 capability 决策一次"). Decode/prefill hot paths branch on these
    /// fields instead of calling `B::supports_*()` inline, so a capability gate
    /// can't silently rot in the forward code.
    pub(crate) supports_varlen_qkv: bool,
    pub(crate) supports_batched_decode: bool,
    /// Runtime knobs resolved once at construction from the snapshot installed
    /// at the composition root (was a process-wide OnceLock reading env).
    /// Forward/decode read these fields instead of a global accessor.
    pub(crate) runtime_env: LlamaFamilyRuntimeEnv,
    pub(crate) batched_cfg: super::llama_family_forward_batched::LlamaBatchedRuntimeConfig,

    /// Token embedding table. `None` for backbone-only models (e.g. the
    /// Qwen3-TTS Talker, which embeds inputs externally and feeds via
    /// `prefill_from_embeds`).
    pub embed: Option<B::Buffer>,
    /// Source checkpoint layer range loaded into `layers`. Full LLM models
    /// use `0..cfg.num_layers`; layer-split stages load a contiguous subset.
    pub layer_source_start: usize,
    pub layer_source_end: usize,
    pub layers: Vec<LlamaFamilyLayer<B>>,
    pub final_norm_w: B::Buffer,
    /// LM output head. `None` for backbone-only models.
    pub lm_head: Option<Box<dyn Linear<B>>>,

    pub rope: RopeCache<B>,
    /// Second rope table for local-attention layers (Gemma 3 dual-θ
    /// scheduling). `None` for single-table families.
    pub rope_local: Option<RopeCache<B>>,
    pub scratch: LlamaFamilyScratch<B>,

    /// Per-sequence KV caches, one `Vec<KvCache<B>>` of length `num_layers`.
    ///
    /// Two layouts overlay this same map:
    /// - **Contiguous mode** (default): each cache holds its own
    ///   `[num_kv_heads, capacity, head_dim]` k/v buffers.
    /// - **Paged mode** (`FERRUM_METAL_PAGED_KV=1`): k/v are unused
    ///   placeholders; the real K/V live in [`Self::paged_pools`] and
    ///   the cache's `block_table` + `context_lens` index into them.
    pub kv_caches: HashMap<String, Vec<K::Layer>>,
    /// Free pool of pre-allocated KV cache slots. Released caches return
    /// here instead of being dropped, so their device pointers stay valid
    /// across requests — critical for graph capture (pointers baked into
    /// the captured graph would otherwise dangle).
    kv_free_pool: Vec<Vec<K::Layer>>,

    // ── Paged-KV multi-seq state (Phase 4) ─────────────────────────────
    //
    // Only populated when `FERRUM_METAL_PAGED_KV=1`. When set, every
    // `kv_caches` entry becomes a "view" into the shared pool: its
    // `k` / `v` buffers are placeholders; reads / writes go through
    // `paged_pools[layer].k` / `.v` indexed via the cache's
    // `block_table`. Multiple cache_ids share the same pool, with
    // disjoint physical block sets owned by `paged_block_alloc`.
    //
    /// Shared K/V pools, one per layer. Sized at model load time for the
    /// configured `MAX_RUNNING_REQUESTS × max_blocks_per_seq` blocks.
    pub paged_pools: Option<Vec<(B::Buffer, B::Buffer)>>,
    /// Block allocator hands out physical block indices from the pool.
    /// `Mutex` because the engine can call `ensure_kv` / `release_kv`
    /// from multiple threads in concurrent serving.
    pub paged_block_alloc: Option<std::sync::Mutex<crate::common::paged_pool::BlockAllocator>>,
    /// Paged-batch dispatch dimensions `(max_seqs, max_blocks_per_seq)`,
    /// pinned at the first `ensure_kv` when paged-KV is on. Stored on
    /// the model (not on scratch) so `ensure_scratch`'s realloc can
    /// re-call `enable_paged_batch` after wiping scratch's
    /// `paged_batch_block_tables` / `paged_batch_q` etc.
    pub paged_dims: Option<(usize, usize)>,

    // ── Graph capture state (CUDA only; harmless no-op on other backends) ──
    /// Count of eager decode steps run so far. After `GRAPH_WARMUP`, the
    /// next step captures the decode flow as a graph.
    pub(crate) graph_warmup: usize,
    /// True if capture was attempted but failed (e.g. backend doesn't
    /// support graph capture). Stops further attempts, falls back to eager.
    pub(crate) graph_capture_failed: bool,
    /// Same warmup counter for the batched-decode path.
    pub(crate) batched_graph_warmup: usize,
    /// True if batched graph capture failed.
    pub(crate) batched_graph_failed: bool,
    /// Set of `m_padded` values (as u64 graph keys) for which a batched
    /// graph has been captured. Multi-slot via cuda.rs's HashMap-keyed
    /// graph cache — different batch shapes don't thrash a single slot.
    pub(crate) batched_graph_keys_seen: std::collections::HashSet<u64>,
    /// Cache IDs for which device-pointer scratch is currently populated.
    /// Populate only re-runs when the batch composition changes (new
    /// requests joined / requests finished). Hot-path optimization:
    /// avoids 3 sync cuMemcpyHtoD_v2's per decode token (~5% TPOT).
    pub(crate) batched_pointers_for: Option<Vec<String>>,
    /// CUDA-graph state for the unified_forward path. Mirrors the
    /// `batched_graph_*` triple but keyed on `(m_total, num_seqs)`
    /// so different concurrency levels each get their own cached
    /// graph instead of thrashing a single slot.
    pub(crate) unified_graph_warmup: usize,
    pub(crate) unified_graph_failed: bool,
    pub(crate) unified_graph_keys_seen: std::collections::HashSet<u64>,

    // ── Real paged-KV prefix cache counters ─────────────────────────────
    prefix_cache_hits: u64,
    prefix_cache_misses: u64,
    prefix_cache_saved_prefill_tokens: u64,
    pub(crate) decode_batch_stats: LlamaDecodeBatchStats,

    // ── Startup LoRA runtime state ──────────────────────────────────────
    lora_adapters: HashMap<String, RuntimeLoraAdapter<B>>,
    lora_cache_adapters: HashMap<String, String>,
    lora_projection_applications: u64,
}

impl<B: MoeLlmBackend, K: KvLayer<B>> LlamaFamilyModel<B, K> {
    /// Build a Qwen3 model from weights provided by the loader.
    ///
    /// The loader decides per-projection whether to instantiate DenseLinear,
    /// GptqLinear, etc. — this code doesn't care.
    pub fn new(cfg: LlamaFamilyConfig, loader: &dyn WeightLoader<B>) -> Result<Self> {
        let num_layers = cfg.num_layers;
        Self::new_with_stage_config(cfg, loader, LlamaFamilyLayerStageConfig::full(num_layers))
    }

    /// Build a backbone-only Qwen3 transformer stack (no embed, no lm_head).
    ///
    /// Intended for composing the transformer inside a larger model where
    /// embedding and output-head logic differs from the standard LLM path —
    /// e.g. Qwen3-TTS Talker uses dual text/codec embeddings with a projection
    /// MLP, and a codec_head output. The caller drives forward via
    /// `prefill_from_embeds` / `decode_from_embed`.
    ///
    /// Loader must provide: per-layer weights under `model.layers.{i}.*` and
    /// the final `model.norm.weight`. `model.embed_tokens` and `lm_head`
    /// are NOT read.
    pub fn new_backbone_only(cfg: LlamaFamilyConfig, loader: &dyn WeightLoader<B>) -> Result<Self> {
        let num_layers = cfg.num_layers;
        Self::new_with_stage_config(
            cfg,
            loader,
            LlamaFamilyLayerStageConfig::backbone(num_layers),
        )
    }

    /// Build a layer-split pipeline stage. The stage always loads its local
    /// transformer layers and final norm; embedding and lm_head are loaded only
    /// for the first and last stages respectively.
    pub fn new_layer_stage(
        cfg: LlamaFamilyConfig,
        loader: &dyn WeightLoader<B>,
        stage: LlamaFamilyLayerStageConfig,
    ) -> Result<Self> {
        Self::new_with_stage_config(cfg, loader, stage)
    }

    fn new_with_stage_config(
        cfg: LlamaFamilyConfig,
        loader: &dyn WeightLoader<B>,
        stage: LlamaFamilyLayerStageConfig,
    ) -> Result<Self> {
        if stage.source_layers.is_empty() {
            return Err(ferrum_types::FerrumError::model(
                "llama layer stage must include at least one source layer",
            ));
        }
        if stage.source_layers.end > cfg.num_layers {
            return Err(ferrum_types::FerrumError::model(format!(
                "llama layer stage range {}..{} is outside model layer count {}",
                stage.source_layers.start, stage.source_layers.end, cfg.num_layers
            )));
        }

        // Invalidate any graph from a previously-loaded model. The captured
        // graph references the old model's scratch buffers; a fresh model
        // gets fresh scratch, so reusing the graph would read/write freed
        // pointers. Matters for test suites where multiple models coexist.
        {
            let mut ctx = B::new_context();
            B::reset_all_graphs(&mut ctx);
        }
        let rope = build_rope_cache::<B>(&cfg);
        // Local-attention rope table (Gemma 3): own θ, never rope-scaled.
        let rope_local = cfg.rope_local_theta.map(|theta| {
            let mut local_cfg = cfg.clone();
            local_cfg.rope_theta = theta;
            local_cfg.rope_scaling = None;
            build_rope_cache::<B>(&local_cfg)
        });
        let scratch = LlamaFamilyScratch::alloc(&cfg, 1);
        let embed = if stage.load_embedding {
            Some(loader.load_tensor("model.embed_tokens.weight")?)
        } else {
            None
        };
        let layers = load_llama_family_layers(&cfg, loader, stage.source_layers.clone())?;

        let lm_head = if stage.load_lm_head {
            Some(load_llama_family_lm_head(&cfg, loader)?)
        } else {
            None
        };
        let final_norm_w = fold_norm_weight::<B>(
            loader.load_tensor("model.norm.weight")?,
            cfg.hidden_size,
            cfg.rmsnorm_unit_offset,
            1.0,
        );

        let layer_source_start = stage.source_layers.start;
        let layer_source_end = stage.source_layers.end;
        let runtime_cfg = cfg.to_runtime();
        // Construction-time capability resolution (GOAL-allowed): read once here
        // so the forward hot paths read self.* instead of B::supports_*().
        // Sandwich-norm families (Gemma 3) require the Gemma layer semantics
        // (GeGLU, post-attn/post-ffn norms, dual rope, per-layer windows)
        // and a device-side F32 residual shadow before a batched/unified
        // fast path is enabled.
        let supports_sandwich_batched_decode =
            supports_sandwich_legacy_batched_decode(&cfg, B::supports_device_f32_residual_shadow());
        let supports_varlen_qkv = unified_varlen_qkv_unsupported_reason(
            &cfg,
            B::supports_varlen_qkv(),
            B::supports_device_f32_residual_shadow(),
        )
        .is_none();
        let supports_batched_decode = B::supports_llama_family_batched_decode()
            && (!cfg.sandwich_norms || supports_sandwich_batched_decode);
        if cfg.sandwich_norms {
            tracing::info!(
                "Gemma3 family: legacy batched_decode={} varlen_unified={}",
                supports_batched_decode,
                supports_varlen_qkv
            );
        }
        let runtime_env = LlamaFamilyRuntimeEnv::from_env();
        let batched_cfg =
            super::llama_family_forward_batched::LlamaBatchedRuntimeConfig::from_env();
        Ok(Self {
            cfg,
            runtime_cfg,
            supports_varlen_qkv,
            supports_batched_decode,
            runtime_env,
            batched_cfg,
            embed,
            layer_source_start,
            layer_source_end,
            layers,
            final_norm_w,
            lm_head,
            rope,
            rope_local,
            scratch,
            kv_caches: HashMap::new(),
            kv_free_pool: Vec::new(),
            paged_pools: None,
            paged_block_alloc: None,
            paged_dims: None,
            graph_warmup: 0,
            graph_capture_failed: false,
            batched_graph_warmup: 0,
            batched_graph_failed: false,
            batched_graph_keys_seen: std::collections::HashSet::new(),
            batched_pointers_for: None,
            unified_graph_warmup: 0,
            unified_graph_failed: false,
            unified_graph_keys_seen: std::collections::HashSet::new(),
            prefix_cache_hits: 0,
            prefix_cache_misses: 0,
            prefix_cache_saved_prefill_tokens: 0,
            decode_batch_stats: LlamaDecodeBatchStats::default(),
            lora_adapters: HashMap::new(),
            lora_cache_adapters: HashMap::new(),
            lora_projection_applications: 0,
        })
    }

    pub fn source_layer_range(&self) -> Range<usize> {
        self.layer_source_start..self.layer_source_end
    }

    pub fn local_layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn cache_len(&self, cache_id: &str) -> usize {
        self.kv_caches
            .get(cache_id)
            .and_then(|layers| layers.first())
            .map(K::len)
            .unwrap_or(0)
    }

    fn local_layer_indices(&self) -> Range<usize> {
        0..self.local_layer_count()
    }

    pub(crate) fn source_layer_index(&self, local_layer_index: usize) -> usize {
        self.layer_source_start + local_layer_index
    }

    fn local_layer_index_for_source(&self, source_layer_index: usize) -> Option<usize> {
        if source_layer_index < self.layer_source_start
            || source_layer_index >= self.layer_source_end
        {
            return None;
        }
        Some(source_layer_index - self.layer_source_start)
    }

    fn resize_scratch(&mut self, tokens: usize) {
        let tokens = tokens.max(1);
        if self.scratch.max_tokens == tokens {
            return;
        }
        // Any captured decode graph holds pointers to the old scratch
        // buffers; those are about to be freed. Invalidate ALL captured
        // graphs (both single-item and per-m_padded batched) — every
        // captured kernel-arg pointer into scratch is stale.
        {
            let mut ctx = B::new_context();
            B::reset_all_graphs(&mut ctx);
        }
        let minimal = LlamaFamilyScratch::alloc(&self.cfg, 1);
        let old = std::mem::replace(&mut self.scratch, minimal);
        drop(old);
        self.scratch = LlamaFamilyScratch::alloc(&self.cfg, tokens);
        self.graph_warmup = 0;
        self.graph_capture_failed = false;
        self.batched_graph_keys_seen.clear();
        self.batched_graph_warmup = 0;
        self.batched_graph_failed = false;
        self.unified_graph_keys_seen.clear();
        self.unified_graph_warmup = 0;
        self.unified_graph_failed = false;
        // Realloc wiped paged_batch_*. Re-enable using the dims
        // pinned at first ensure_kv. Without this, the next
        // `forward_layer_batched_decode` panics on
        // `paged_batch_block_tables missing`.
        if let Some((max_seqs, max_blocks_per_seq)) = self.paged_dims {
            self.scratch
                .enable_paged_batch(&self.cfg, max_seqs, max_blocks_per_seq);
        }
    }

    /// Grow scratch buffers if `tokens` exceeds the current sizing.
    pub(crate) fn ensure_scratch(&mut self, tokens: usize) {
        if self.scratch.max_tokens < tokens {
            self.resize_scratch(tokens);
        }
    }

    /// Ensure per-layer KV caches exist for `cache_id`, pre-allocated to
    /// `max_seq_len` slots per head. Enables the in-place
    /// `kv_cache_append_head_major` path — no realloc per layer.
    pub(crate) fn ensure_kv(&mut self, cache_id: &str) {
        self.ensure_kv_with_capacity_hint(cache_id, None);
    }

    pub(crate) fn ensure_kv_with_capacity_hint(
        &mut self,
        cache_id: &str,
        capacity_hint: Option<usize>,
    ) {
        if self.kv_caches.contains_key(cache_id) {
            return;
        }
        let nkv = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim;
        // KV capacity defaults to a chat-friendly 4096 to keep the working
        // set sane on a 32 GB Mac (a 48-layer / 4-kv-head / 128-head_dim
        // model spends ~786 MB on 4096 KV slots, vs ~6 GB on the model's
        // declared 32K which would push the 17 GB Qwen3-30B-A3B model into
        // swap). `FERRUM_KV_CAPACITY=N` overrides; clamp to the model's
        // declared max so we never lie to the model about its window.
        let model_max = self.cfg.max_seq_len;
        // 512 in 0.7.2 — matches the value used in
        // docs/bench/macos-2026-05-02 to get the published numbers.
        // pre-0.7.2 default of 4096 was safe only because paged-KV was
        // opt-in (pool wasn't allocated). With paged-KV now on by
        // default + MAX_SEQS=32, the pool occupies physical memory:
        // ~3 GB on Qwen3-30B-A3B Q4_K_M leaves 18 GB weights + 3 GB pool
        // = 21 GB, fits comfortably on a 32 GB Mac. Long-context users
        // can `FERRUM_KV_CAPACITY=4096` and accept lower max_seqs.
        let runtime_env = &self.runtime_env;
        let max = runtime_env.kv_capacity_for_model(model_max);
        let contig_initial_capacity = capacity_hint
            .map(|hint| hint.max(1).min(max.max(1)))
            .unwrap_or(max.max(1));

        // Paged-KV mode: `FERRUM_METAL_PAGED_KV=1` switches every cache
        // for this model into block-table-indirect layout. Kernels from
        // PR #68 (decode read) + PR #69 (decode write) handle the
        // indirect addressing; the LlamaFamily decode path below picks
        // them up automatically by checking `cache.block_size > 0`.
        //
        // Pool sizing: round capacity up to a multiple of block_size,
        // identity-assign logical→physical block. Memory footprint is
        // the same as contiguous (within block_size rounding); the
        // benefit only shows up under multi-seq sharing in Phase 4+.
        // Default ON when the backend supports paged-KV (Metal). Users
        // can force off with `FERRUM_METAL_PAGED_KV=0`. The flag was
        // opt-in pre-0.7.2; flipping the default so default `ferrum
        // serve` matches the bench-quality numbers without requiring
        // env-var knowledge.
        // Windowed Gemma3 can use paged KV only on backends whose varlen QKV
        // path takes the per-layer sliding-window schedule.
        let paged = paged_kv_allowed_for_layer_schedule(
            runtime_env.paged_kv_enabled::<B>(),
            B::supports_varlen_qkv(),
            self.cfg.sliding_window_pattern,
        );
        const PAGED_BLOCK_SIZE: usize = 16;

        // Phase 4 shared-pool sizing. The pool sees ALL concurrent
        // sequences; per-cache_id state just owns indices into it.
        // Default 32: covers c=16 burst with 2× headroom for the
        // fresh-cache-id-per-request pattern that bench/server harnesses
        // use. Pool memory is `max_seqs × max_blocks_per_seq` total
        // blocks — we lowered DEFAULT_KV_CAPACITY to 2048 so this 2× max_seqs
        // bump keeps the pool footprint identical to the pre-0.7.2 default.
        let max_seqs = runtime_env.paged_max_seqs;
        let max_blocks_per_seq = max.div_ceil(PAGED_BLOCK_SIZE);
        let total_pool_blocks = max_seqs * max_blocks_per_seq;

        // Lazy-allocate the shared paged pools on the FIRST paged
        // ensure_kv call. Pools are big — for Llama-8B (8 kv_heads,
        // head_dim=128) at 16 seqs × 256 blocks × 16 slots = 65536 KV
        // slots: 65536 * 8 * 128 * 4 = 256 MB per layer × 32 layers
        // = 8 GB total. Sized this large only because `max_seqs=16`
        // is the default; lower it via env to shrink.
        if paged && self.paged_pools.is_none() {
            let mut pools = Vec::with_capacity(self.local_layer_count());
            for _ in self.local_layer_indices() {
                let pool_floats = total_pool_blocks * nkv * PAGED_BLOCK_SIZE * hd;
                pools.push((B::alloc(pool_floats), B::alloc(pool_floats)));
            }
            self.paged_pools = Some(pools);
            self.paged_block_alloc = Some(std::sync::Mutex::new(
                crate::common::paged_pool::BlockAllocator::new(total_pool_blocks as u32),
            ));
        }
        // Phase 4b: ensure batched-dispatch scratch is allocated whenever
        // paged is on. Idempotent — re-init is a no-op if already
        // sized. Has to live outside the `paged_pools.is_none()` branch
        // because `ensure_scratch` may have replaced the scratch struct
        // since the pools were first allocated.
        if paged {
            self.scratch
                .enable_paged_batch(&self.cfg, max_seqs, max_blocks_per_seq);
            // Pin dims on the model so `ensure_scratch`'s realloc can
            // re-init paged_batch_* after wiping scratch.
            self.paged_dims = Some((max_seqs, max_blocks_per_seq));
        }

        // Try pool first — reused buffers have stable device pointers,
        // so a captured decode graph can be replayed for this request too.
        // K::NAME selects which `LayerKvCache` variant to construct:
        // K-aware allocation: K::alloc_paged / K::alloc_contig pick the
        // right cache layout (FP16 → KvCache, INT8 → KvCacheQuant) per the
        // model's K marker. INT8 supports paged mode only — KvInt8::alloc_contig
        // panics, surfacing the misconfiguration here at first ensure_kv.
        let local_layer_count = self.local_layer_count();
        let allocate_caches = || {
            (0..local_layer_count)
                .map(|_| {
                    if paged {
                        K::alloc_paged(max_blocks_per_seq, PAGED_BLOCK_SIZE, nkv, hd)
                    } else {
                        K::alloc_contig(contig_initial_capacity, nkv, hd)
                    }
                })
                .collect()
        };
        let mut caches = match self.kv_free_pool.pop() {
            Some(mut cached)
                if !paged
                    && capacity_hint.is_some()
                    && cached
                        .first()
                        .is_some_and(|cache| K::capacity(cache) != contig_initial_capacity) =>
            {
                for cache in cached.iter_mut() {
                    K::set_len(cache, 0);
                }
                drop(cached);
                allocate_caches()
            }
            Some(mut cached) => {
                if !paged {
                    for cache in cached.iter_mut() {
                        K::set_len(cache, 0);
                    }
                }
                cached
            }
            None => allocate_caches(),
        };

        // Allocate physical blocks for THIS cache_id from the shared
        // pool. We allocate all `max_blocks_per_seq` upfront for
        // simplicity (matches contig's "pre-alloc to capacity"
        // semantics); a smarter Phase 4b can grow on demand to save
        // pool occupancy.
        if paged {
            let alloc_arc = self
                .paged_block_alloc
                .as_ref()
                .expect("paged_block_alloc must be initialised when paged=true");
            // Recover from a previously-poisoned mutex instead of panicking
            // (poison just means a prior holder panicked; the BlockAllocator
            // state is still intact since allocate_n is fail-safe).
            let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
            let block_indices = match alloc.allocate_n(max_blocks_per_seq) {
                Ok(idx) => idx,
                Err(e) => {
                    // Pool exhaustion is a back-pressure signal, not a crash.
                    // Drop the lock, return the cache to the free pool, and
                    // bail before inserting it into kv_caches. The downstream
                    // call will then fail with a clean per-request error
                    // ("ensure_kv must be called before ...") instead of
                    // dragging every other in-flight request down with it.
                    drop(alloc);
                    self.kv_free_pool.push(caches);
                    eprintln!(
                        "[ferrum] paged KV pool exhausted on ensure_kv for \
                         cache_id={cache_id:?}: {e}. Increase \
                         FERRUM_PAGED_MAX_SEQS (currently {max_seqs}) or \
                         throttle concurrent requests.",
                    );
                    return;
                }
            };
            // Write the block table to each layer's cache. All layers
            // share the same logical→physical mapping for this seq.
            // Also stash the host-side index list so release_kv can
            // return them to the allocator without a device readback.
            let mut padded = block_indices.clone();
            padded.resize(max_blocks_per_seq, 0);
            let mut ctx_tmp = B::new_context();
            for c in caches.iter_mut() {
                if let Some(bt) = K::block_table_mut(c) {
                    B::write_typed::<u32>(&mut ctx_tmp, bt, &padded);
                }
                *K::paged_block_indices_mut(c) = block_indices.clone();
            }
            B::sync(&mut ctx_tmp);
        }

        // Reset logical length; buffers stay. No need to zero the memory —
        // the kv_cache_append writes new K/V in place, and attention only
        // reads up to `cache_len`.
        for c in caches.iter_mut() {
            K::set_len(c, 0);
            if let Some(cl) = K::context_lens_mut(c) {
                let mut ctx_tmp = B::new_context();
                B::write_typed::<u32>(&mut ctx_tmp, cl, &[0u32]);
                B::sync(&mut ctx_tmp);
            }
        }
        self.kv_caches.insert(cache_id.to_string(), caches);
    }

    fn record_prefix_cache_probe(&mut self, saved_tokens: usize) {
        if saved_tokens > 0 {
            self.prefix_cache_hits += 1;
            self.prefix_cache_saved_prefill_tokens += saved_tokens as u64;
        } else {
            self.prefix_cache_misses += 1;
        }
    }

    fn try_acquire_prefix_cache(&mut self, cache_id: &str, tokens: &[u32]) -> usize {
        let Some(alloc_arc) = self.paged_block_alloc.as_ref() else {
            return 0;
        };
        let caches = match self.kv_caches.get(cache_id) {
            Some(caches) => caches,
            None => return 0,
        };
        let block_size = caches.first().map(K::block_size).unwrap_or(0);
        if block_size == 0 {
            return 0;
        }

        let token_ids: Vec<ferrum_types::TokenId> = tokens
            .iter()
            .map(|&token| ferrum_types::TokenId::new(token))
            .collect();
        let hashes = block_hash_chain(&token_ids, block_size);
        if hashes.is_empty() {
            return 0;
        }

        let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
        let mut matched = Vec::with_capacity(hashes.len());
        for hash in hashes {
            match alloc.try_acquire_by_hash(hash) {
                Some(block) => matched.push(block),
                None => break,
            }
        }
        if matched.is_empty() {
            return 0;
        }
        let n_matched = matched.len();

        let displaced = caches
            .first()
            .map(|cache| K::paged_block_indices(cache)[..n_matched].to_vec())
            .unwrap_or_default();
        alloc.free(&displaced);
        drop(alloc);

        let caches_mut = self.kv_caches.get_mut(cache_id).expect("cache present");
        let max_blocks = caches_mut
            .first()
            .map(|cache| K::paged_block_indices(cache).len())
            .unwrap_or(0);
        let new_len = n_matched * block_size;
        let mut ctx = B::new_context();
        for cache in caches_mut.iter_mut() {
            {
                let indices = K::paged_block_indices_mut(cache);
                for (idx, &block) in matched.iter().enumerate() {
                    indices[idx] = block;
                }
            }
            K::set_len(cache, new_len);
            let padded = {
                let mut padded = K::paged_block_indices(cache).to_vec();
                padded.resize(max_blocks, 0);
                padded
            };
            if let Some(block_table) = K::block_table_mut(cache) {
                B::write_typed::<u32>(&mut ctx, block_table, &padded);
            }
            if let Some(context_lens) = K::context_lens_mut(cache) {
                B::write_typed::<u32>(&mut ctx, context_lens, &[new_len as u32]);
            }
        }
        B::sync(&mut ctx);

        new_len
    }

    fn register_prefix_cache(
        &mut self,
        cache_id: &str,
        all_tokens: &[u32],
        prior_cached_tokens: usize,
    ) {
        let Some(alloc_arc) = self.paged_block_alloc.as_ref() else {
            return;
        };
        let caches = match self.kv_caches.get(cache_id) {
            Some(caches) => caches,
            None => return,
        };
        let cache0 = match caches.first() {
            Some(cache) => cache,
            None => return,
        };
        let block_size = K::block_size(cache0);
        if block_size == 0 {
            return;
        }

        let token_ids: Vec<ferrum_types::TokenId> = all_tokens
            .iter()
            .map(|&token| ferrum_types::TokenId::new(token))
            .collect();
        let hashes = block_hash_chain(&token_ids, block_size);
        if hashes.is_empty() {
            return;
        }

        let start_block = prior_cached_tokens / block_size;
        let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
        for i in start_block..hashes.len().min(K::paged_block_indices(cache0).len()) {
            let block_end_token = (i + 1) * block_size;
            if block_end_token > K::len(cache0) {
                break;
            }
            alloc.register_block_hash(K::paged_block_indices(cache0)[i], hashes[i]);
        }
    }

    fn prefix_cache_snapshot_json(&self) -> serde_json::Value {
        let (entries, block_size) = self
            .paged_block_alloc
            .as_ref()
            .and_then(|alloc| {
                let alloc = alloc.lock().ok()?;
                let block_size = self
                    .kv_caches
                    .values()
                    .find_map(|layers| layers.first().map(K::block_size))
                    .unwrap_or(16);
                Some((alloc.hash_table_size() as u64, block_size))
            })
            .unwrap_or((0, 16));
        let bytes_per_entry = (block_size
            * self.local_layer_count()
            * self.cfg.num_kv_heads
            * self.cfg.head_dim
            * K::BYTES_PER_ELEM
            * 2) as u64;
        serde_json::json!({
            "position": "real-kv-reuse",
            "source": "llama-family-paged-block-prefix-cache",
            "enabled": self.runtime_env.prefix_cache,
            "hits": self.prefix_cache_hits,
            "misses": self.prefix_cache_misses,
            "evictions": 0u64,
            "saved_prefill_tokens": self.prefix_cache_saved_prefill_tokens,
            "entries": entries,
            "bytes": entries.saturating_mul(bytes_per_entry),
            "block_size": block_size,
            "kv_dtype": K::NAME,
            "decode_batch": self.decode_batch_stats.snapshot_json(),
        })
    }

    fn lora_projection_shape(
        &self,
        layer_index: usize,
        target_module: &str,
    ) -> Option<(usize, usize)> {
        let local_layer_index = self.local_layer_index_for_source(layer_index)?;
        let layer = self.layers.get(local_layer_index)?;
        match target_module {
            "qkv_proj" => Some((layer.qkv_proj.in_features(), layer.qkv_proj.out_features())),
            "o_proj" => Some((layer.o_proj.in_features(), layer.o_proj.out_features())),
            "gate_up_proj" => Some((
                layer.gate_up_proj.in_features(),
                layer.gate_up_proj.out_features(),
            )),
            "down_proj" => Some((
                layer.down_proj.in_features(),
                layer.down_proj.out_features(),
            )),
            _ => None,
        }
    }

    fn validate_lora_adapter(&self, adapter: &RuntimeLoraAdapter<B>) -> Result<()> {
        if adapter.linears.is_empty() {
            return Err(ferrum_types::FerrumError::config(format!(
                "LoRA adapter {} has no runtime tensors",
                adapter.name
            )));
        }
        for linear in &adapter.linears {
            let layer_index = linear.layer_index.ok_or_else(|| {
                ferrum_types::FerrumError::config(format!(
                    "LoRA tensor for target {} must include model.layers.<N> in its tensor name",
                    linear.target_module
                ))
            })?;
            if layer_index < self.cfg.num_layers
                && !self.source_layer_range().contains(&layer_index)
            {
                continue;
            }
            let Some((expected_in, expected_out)) =
                self.lora_projection_shape(layer_index, &linear.target_module)
            else {
                return Err(ferrum_types::FerrumError::unsupported(format!(
                    "LoRA target {} is not supported by Llama-family runtime; supported targets: qkv_proj, o_proj, gate_up_proj, down_proj",
                    linear.target_module
                )));
            };
            if linear.in_features != expected_in || linear.out_features != expected_out {
                return Err(ferrum_types::FerrumError::config(format!(
                    "LoRA tensor shape mismatch for layer {} target {}: got out={} in={}, expected out={} in={}",
                    layer_index,
                    linear.target_module,
                    linear.out_features,
                    linear.in_features,
                    expected_out,
                    expected_in
                )));
            }
        }
        Ok(())
    }

    fn ensure_lora_adapter_loaded(&mut self, adapter: ActiveLoraAdapter) -> Result<()> {
        if self.lora_adapters.contains_key(&adapter.name) {
            return Ok(());
        }
        let runtime = load_runtime_lora_adapter::<B>(&adapter)?;
        self.validate_lora_adapter(&runtime)?;
        self.lora_adapters.insert(adapter.name.clone(), runtime);
        Ok(())
    }

    pub(crate) fn active_lora_adapter_for_cache(
        &self,
        cache_id: &str,
    ) -> Option<&RuntimeLoraAdapter<B>> {
        let adapter_name = self.lora_cache_adapters.get(cache_id)?;
        self.lora_adapters.get(adapter_name)
    }

    fn active_lora_adapter_ptr_for_cache(
        &self,
        cache_id: &str,
    ) -> Option<*const RuntimeLoraAdapter<B>> {
        self.active_lora_adapter_for_cache(cache_id)
            .map(|adapter| adapter as *const RuntimeLoraAdapter<B>)
    }

    fn lora_metrics_snapshot_json(&self) -> serde_json::Value {
        serde_json::json!({
            "enabled": !self.lora_adapters.is_empty(),
            "adapter_count": self.lora_adapters.len() as u64,
            "active_cache_bindings": self.lora_cache_adapters.len() as u64,
            "projection_applications": self.lora_projection_applications,
            "position": "real-inference",
            "source": "llama-family-runtime-lora",
        })
    }

    pub(crate) fn use_host_residual_shadow(&self) -> bool {
        self.cfg.sandwich_norms
            && B::activation_elem_size_bytes() < std::mem::size_of::<f32>()
            && !B::supports_device_f32_residual_shadow()
    }

    pub(crate) fn use_device_residual_shadow(&self) -> bool {
        self.cfg.sandwich_norms
            && B::activation_elem_size_bytes() < std::mem::size_of::<f32>()
            && B::supports_device_f32_residual_shadow()
    }

    fn rms_norm_host(
        input: &[f32],
        weight: &[f32],
        tokens: usize,
        dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        assert_eq!(input.len(), tokens * dim);
        assert_eq!(weight.len(), dim);
        let mut out = vec![0.0f32; input.len()];
        for row in 0..tokens {
            let off = row * dim;
            let row_in = &input[off..off + dim];
            let mut variance = 0.0f32;
            for &x in row_in {
                variance += x * x;
            }
            let inv_rms = (variance / dim as f32 + eps).sqrt().recip();
            for i in 0..dim {
                out[off + i] = row_in[i] * inv_rms * weight[i];
            }
        }
        out
    }

    pub(crate) fn rms_norm_host_to_activation(
        ctx: &mut B::Context,
        input: &[f32],
        weight: &B::Buffer,
        eps: f32,
        out: &mut B::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        let weight_h = B::to_vec(weight, dim);
        let normed = Self::rms_norm_host(input, &weight_h, tokens, dim, eps);
        B::write_f32_to_activation(ctx, out, &normed);
    }

    fn rms_norm_device_to_host(
        input: &B::Buffer,
        weight: &B::Buffer,
        eps: f32,
        tokens: usize,
        dim: usize,
    ) -> Vec<f32> {
        let input_h = B::to_vec(input, tokens * dim);
        let weight_h = B::to_vec(weight, dim);
        Self::rms_norm_host(&input_h, &weight_h, tokens, dim, eps)
    }

    fn add_host_residual(residual: &mut [f32], branch: &[f32]) {
        assert_eq!(residual.len(), branch.len());
        for (r, &x) in residual.iter_mut().zip(branch) {
            *r += x;
        }
    }

    /// Run one transformer layer. Mutates `residual` in place.
    ///
    /// `pos_offset` is the absolute position of token 0 in this batch
    /// (decode: `pos`; prefill: 0). `tokens` is the batch size.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_layer(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        cache_id: &str,
        residual: &mut B::Buffer,
        pos_offset: usize,
        tokens: usize,
    ) {
        self.forward_layer_with_host_residual(
            ctx, li, cache_id, residual, pos_offset, tokens, None,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_layer_with_host_residual(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        cache_id: &str,
        residual: &mut B::Buffer,
        pos_offset: usize,
        tokens: usize,
        mut host_residual: Option<&mut [f32]>,
    ) {
        self.forward_layer_with_residual_shadow(
            ctx,
            li,
            cache_id,
            residual,
            pos_offset,
            tokens,
            host_residual.as_deref_mut(),
            None,
            None,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_layer_with_residual_shadow(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        cache_id: &str,
        residual: &mut B::Buffer,
        pos_offset: usize,
        tokens: usize,
        mut host_residual: Option<&mut [f32]>,
        mut device_residual: Option<&mut B::Buffer>,
        mut device_branch: Option<&mut B::Buffer>,
    ) {
        let source_li = self.source_layer_index(li);
        let layer = &self.layers[li];
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let nh = cfg.num_heads;
        let nkv = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let im = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;

        // FERRUM_NAN_TRACE=all or a comma-separated layer list: after each
        // op of matching layers, sync and report non-finite counts.
        // Debug-only; intentionally not part of the product path.
        let nan_trace = self.runtime_env.nan_trace.matches_layer(li, source_li);
        let op_dump_dir = self.runtime_env.op_dump_dir.as_deref();
        macro_rules! nt {
            ($name:expr, $buf:expr, $len:expr, $row_width:expr) => {
                if nan_trace {
                    nan_trace_dump::<B>(
                        ctx,
                        source_li,
                        $name,
                        $buf,
                        $len,
                        Some($row_width),
                        op_dump_dir,
                    );
                }
            };
        }
        nt!("residual-in", residual, tokens * h, h);
        let op_profile = self.runtime_env.decode_op_profile
            || (self.runtime_env.prefill_op_profile && tokens > 1);

        // 1. Input RMSNorm
        let _t0 = if op_profile {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        if let Some(host) = host_residual.as_deref() {
            Self::rms_norm_host_to_activation(
                ctx,
                host,
                &layer.input_ln_w,
                eps,
                &mut self.scratch.norm_out,
                tokens,
                h,
            );
        } else if let Some(device) = device_residual.as_ref() {
            B::rms_norm_f32_to_activation(
                ctx,
                &**device,
                &layer.input_ln_w,
                eps,
                &mut self.scratch.norm_out,
                tokens,
                h,
            );
        } else {
            B::rms_norm(
                ctx,
                residual,
                &layer.input_ln_w,
                eps,
                &mut self.scratch.norm_out,
                tokens,
                h,
            );
        }
        nt!("input_norm", &self.scratch.norm_out, tokens * h, h);
        if let Some(t0) = _t0 {
            B::sync(ctx);
            NORM_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            NORM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 2. Fused QKV projection (Linear dispatches to Dense/GPTQ/AWQ/GGUF)
        let _t0 = if op_profile {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        {
            #[cfg(feature = "cuda")]
            let _alloc_label =
                ferrum_kernels::backend::cuda::push_alloc_label("llama.forward_layer.qkv_proj");
            layer.qkv_proj.forward(
                ctx,
                &self.scratch.norm_out,
                &mut self.scratch.qkv_out,
                tokens,
            );
        }
        nt!(
            "qkv_proj",
            &self.scratch.qkv_out,
            tokens * (q_dim + 2 * kv_dim),
            q_dim + 2 * kv_dim
        );
        if let Some(adapter) = self.active_lora_adapter_ptr_for_cache(cache_id) {
            // SAFETY: `lora_adapters` is not mutated during a forward pass;
            // the pointer is used only for this immediate read-only call.
            let applied = unsafe { &*adapter }
                .apply_projection(
                    ctx,
                    source_li,
                    "qkv_proj",
                    &self.scratch.norm_out,
                    &mut self.scratch.qkv_out,
                    tokens,
                )
                .expect("validated LoRA qkv_proj");
            self.lora_projection_applications += applied as u64;
        }
        if let Some(t0) = _t0 {
            B::sync(ctx);
            MATMUL_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            MATMUL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 3-5. Fused split-QKV + QK-norm + RoPE + cache-write.
        //
        // Single Metal dispatch replaces the (split_qkv → 3× qk_norm_rope
        // → kv_cache_append_head_major) five-launch chain on the decode
        // hot path. Reads qkv_out once, writes Q to head-major scratch
        // and K/V straight into the pre-allocated KV cache slot at
        // `cache_len + tok`. Saves 4 dispatches per layer when the
        // backend implements the fused kernel; CPU and other backends
        // keep using the unfused chain via the Unsupported fallbacks.
        //
        // qk_mode: 1 = norm + half-split RoPE (Qwen3/Qwen HF);
        //          2 = half-split RoPE only;
        //          3 = interleaved RoPE only (GGUF LLaMA / llama.cpp layout).
        // V always passes apply_norm=0.
        let qk_mode: i32 = llama_qk_mode(cfg);
        let dummy = &layer.input_ln_w;
        let q_norm_w = layer.q_norm_w.as_ref().unwrap_or(dummy);
        let k_norm_w = layer.k_norm_w.as_ref().unwrap_or(dummy);

        // Per-layer attention scheduling (Gemma 3 `sliding_window_pattern`):
        // every Nth layer is global (full window, main rope table); the rest
        // are local (sliding window, local rope table). Pattern 0 = uniform
        // legacy behavior. Field-precise rope refs keep the later
        // `&mut self.scratch` / `kv_caches` borrows legal.
        let schedule = llama_layer_attention_schedule(cfg, source_li);
        let layer_window = schedule.sliding_window;
        let (rope_cos, rope_sin) = match (&self.rope_local, schedule.is_global_layer) {
            (Some(local), false) => (&local.cos, &local.sin),
            _ => (&self.rope.cos, &self.rope.sin),
        };

        // Grab the per-layer KV cache up front so the deepest fusion can
        // write K/V straight into it.
        //
        // Paged mode: also need this layer's shared pool buffers
        // (self.paged_pools[li]). The pool is a separate field from
        // kv_caches, so we take a raw pointer to its (k, v) here while
        // we still hold &mut self, then deref via unsafe inside the
        // paged dispatch below. Safety: paged_pools is allocated once
        // and never resized; we don't touch self.paged_pools while the
        // pointer is in use.
        let paged_pool_ptr: Option<(*mut B::Buffer, *mut B::Buffer)> =
            if let Some(pools) = self.paged_pools.as_mut() {
                let pool = &mut pools[li];
                Some((&mut pool.0 as *mut _, &mut pool.1 as *mut _))
            } else {
                None
            };
        let caches = self
            .kv_caches
            .get_mut(cache_id)
            .expect("ensure_kv must be called before forward_layer");
        // Read shared metadata (variant-agnostic) once. The K-aware
        // attn section below re-borrows the right enum variant.
        let cache_len_before = K::len(&caches[li]);
        let cache_block_size_before = K::block_size(&caches[li]);
        let required_kv_len = cache_len_before + tokens;
        let logical_capacity = if cache_block_size_before > 0 {
            K::capacity(&caches[li])
        } else {
            self.runtime_env.kv_capacity_for_model(self.cfg.max_seq_len)
        };

        // Defense in depth: refuse to write past the KV buffer. The
        // graceful path is the caller pre-checking via `kv_capacity()`
        // and either compacting or refusing the request; this panic only
        // fires when that contract is broken (and silent overflow would
        // otherwise corrupt the cache + adjacent allocations).
        if required_kv_len > logical_capacity {
            panic!(
                "KV cache overflow on source layer {source_li} (local layer {li}): would write tokens [{cache_len_before}..{}) but capacity is {logical_capacity} (cache_id={cache_id:?}). Increase FERRUM_KV_CAPACITY or call /clear in the REPL.",
                required_kv_len
            );
        }
        if cache_block_size_before == 0 {
            K::ensure_contig_capacity(ctx, &mut caches[li], required_kv_len, nkv, hd)
                .expect("K::ensure_contig_capacity");
        }
        let cache_capacity = K::capacity(&caches[li]);
        let cache_block_size = K::block_size(&caches[li]);

        // Paged path: K::paged_write fuses split_qkv_norm_rope + cache append
        // (FP16: into_paged_cache; INT8: split_qkv_norm_rope + int8_kv_append_paged).
        // K::paged_decode_attention reads from layer-local INT8 buffers or the
        // shared FP16 pool depending on K. Then K-agnostic post-attn tail.
        if cache_block_size > 0 {
            let (pool_k_ptr, pool_v_ptr) =
                paged_pool_ptr.expect("paged_pools must be allocated when block_size > 0");
            // SAFETY: paged_pools is allocated once and never resized; the
            // raw pointers don't outlive this method scope.
            let pool_k = unsafe { &mut *pool_k_ptr };
            let pool_v = unsafe { &mut *pool_v_ptr };

            K::paged_write(
                ctx,
                &mut caches[li],
                &self.scratch.qkv_out,
                q_norm_w,
                k_norm_w,
                rope_cos,
                rope_sin,
                &mut self.scratch.q_head_major,
                &mut self.scratch.k_head_major,
                &mut self.scratch.v_head_major,
                pool_k,
                pool_v,
                tokens,
                nh,
                nkv,
                hd,
                pos_offset,
                eps,
                qk_mode,
            )
            .expect("K::paged_write");

            let new_len = cache_len_before + tokens;
            K::set_len(&mut caches[li], new_len);

            let pool_k_imm = unsafe { &*pool_k_ptr };
            let pool_v_imm = unsafe { &*pool_v_ptr };
            K::paged_decode_attention(
                ctx,
                &mut caches[li],
                &self.scratch.q_head_major,
                pool_k_imm,
                pool_v_imm,
                &mut self.scratch.attn_head_major_out,
                nh,
                nkv,
                hd,
                new_len,
                tokens,
            )
            .expect("K::paged_decode_attention");

            return self.forward_layer_post_attn_with_host_residual(
                ctx,
                li,
                cache_id,
                residual,
                tokens,
                host_residual,
                device_residual,
                device_branch,
            );
        }

        // Non-paged (contig) path. INT8 path doesn't reach here:
        // KvInt8::alloc_contig panics in ensure_kv.
        let _qkr_t0 = if op_profile {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        K::contig_write(
            ctx,
            &mut caches[li],
            &self.scratch.qkv_out,
            q_norm_w,
            k_norm_w,
            rope_cos,
            rope_sin,
            &mut self.scratch.q_head_major,
            &mut self.scratch.k_head_major,
            &mut self.scratch.v_head_major,
            &mut self.scratch.q_buf,
            &mut self.scratch.k_buf,
            &mut self.scratch.v_buf,
            tokens,
            nh,
            nkv,
            hd,
            pos_offset,
            eps,
            qk_mode,
        )
        .expect("K::contig_write");
        nt!(
            "contig_write.q",
            &self.scratch.q_head_major,
            nh * tokens * hd,
            hd
        );
        if let Some(t0) = _qkr_t0 {
            B::sync(ctx);
            QKR_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            QKR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let new_len = cache_len_before + tokens;
        K::set_len(&mut caches[li], new_len);
        let kv_stride = cache_capacity;

        let _attn_t0 = if op_profile {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        let attn_cfg = ferrum_kernels::backend::AttnConfig {
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            causal: true,
            scale: 1.0 / (hd as f32).sqrt(),
            kv_seq_stride: kv_stride,
            sliding_window: layer_window,
        };
        K::contig_decode_attention(
            ctx,
            &caches[li],
            &self.scratch.q_head_major,
            &mut self.scratch.attn_head_major_out,
            attn_cfg,
            tokens,
            pos_offset,
        )
        .expect("K::contig_decode_attention");
        nt!(
            "attn",
            &self.scratch.attn_head_major_out,
            nh * tokens * hd,
            hd
        );
        let _ = q_dim;
        let _ = kv_dim;
        let _ = dummy;
        if let Some(t0) = _attn_t0 {
            B::sync(ctx);
            ATTN_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            ATTN_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        self.forward_layer_post_attn_with_host_residual(
            ctx,
            li,
            cache_id,
            residual,
            tokens,
            host_residual,
            device_residual,
            device_branch,
        );
    }

    /// Post-attention tail of `forward_layer`: untranspose Q (if needed),
    /// O-proj, fused residual+post_norm, gate_up_proj, SwiGLU, down_proj,
    /// final residual add. K-agnostic — reads `self.scratch.attn_head_major_out`
    /// which both the FP16 and INT8 attn paths populate.
    pub(crate) fn forward_layer_post_attn(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        cache_id: &str,
        residual: &mut B::Buffer,
        tokens: usize,
    ) {
        self.forward_layer_post_attn_with_host_residual(
            ctx, li, cache_id, residual, tokens, None, None, None,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_layer_post_attn_with_host_residual(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        cache_id: &str,
        residual: &mut B::Buffer,
        tokens: usize,
        mut host_residual: Option<&mut [f32]>,
        mut device_residual: Option<&mut B::Buffer>,
        mut device_branch: Option<&mut B::Buffer>,
    ) {
        let source_li = self.source_layer_index(li);
        let layer = &self.layers[li];
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let nh = cfg.num_heads;
        let hd = cfg.head_dim;
        let im = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;

        let nan_trace = self.runtime_env.nan_trace.matches_layer(li, source_li);
        let op_dump_dir = self.runtime_env.op_dump_dir.as_deref();
        macro_rules! nt {
            ($name:expr, $buf:expr, $len:expr, $row_width:expr) => {
                if nan_trace {
                    nan_trace_dump::<B>(
                        ctx,
                        source_li,
                        $name,
                        $buf,
                        $len,
                        Some($row_width),
                        op_dump_dir,
                    );
                }
            };
        }

        // 7. Untranspose head-major → token-major for O-proj input.
        let attn_token_major = if tokens == 1 {
            &self.scratch.attn_head_major_out
        } else {
            B::transpose_head_to_token(
                ctx,
                &self.scratch.attn_head_major_out,
                &mut self.scratch.attn_flat,
                tokens,
                nh,
                hd,
            );
            &self.scratch.attn_flat
        };

        // 8. O projection.
        {
            #[cfg(feature = "cuda")]
            let _alloc_label =
                ferrum_kernels::backend::cuda::push_alloc_label("llama.forward_layer.o_proj");
            layer
                .o_proj
                .forward(ctx, attn_token_major, &mut self.scratch.o_proj_out, tokens);
        }
        if let Some(adapter) = self.active_lora_adapter_ptr_for_cache(cache_id) {
            // SAFETY: see qkv_proj LoRA application above.
            let applied = unsafe { &*adapter }
                .apply_projection(
                    ctx,
                    source_li,
                    "o_proj",
                    attn_token_major,
                    &mut self.scratch.o_proj_out,
                    tokens,
                )
                .expect("validated LoRA o_proj");
            self.lora_projection_applications += applied as u64;
        }
        nt!("o_proj", &self.scratch.o_proj_out, tokens * h, h);

        self.forward_layer_post_o_proj_with_residual_shadow(
            ctx,
            li,
            Some(cache_id),
            residual,
            tokens,
            host_residual,
            device_residual,
            device_branch,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_layer_post_o_proj_with_residual_shadow(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        cache_id: Option<&str>,
        residual: &mut B::Buffer,
        tokens: usize,
        mut host_residual: Option<&mut [f32]>,
        mut device_residual: Option<&mut B::Buffer>,
        mut device_branch: Option<&mut B::Buffer>,
    ) {
        let source_li = self.source_layer_index(li);
        let layer = &self.layers[li];
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let im = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;

        let nan_trace = self.runtime_env.nan_trace.matches_layer(li, source_li);
        let op_dump_dir = self.runtime_env.op_dump_dir.as_deref();
        macro_rules! nt {
            ($name:expr, $buf:expr, $len:expr, $row_width:expr) => {
                if nan_trace {
                    nan_trace_dump::<B>(
                        ctx,
                        source_li,
                        $name,
                        $buf,
                        $len,
                        Some($row_width),
                        op_dump_dir,
                    );
                }
            };
        }
        let tail_profile = self.runtime_env.decode_op_profile
            || (self.runtime_env.prefill_op_profile && tokens > 1)
            || self.batched_cfg.decode_op_profile_enabled();
        macro_rules! time_tail {
            ($bucket_us:expr, $bucket_n:expr, $body:block) => {{
                if tail_profile {
                    B::sync(ctx);
                    let t0 = std::time::Instant::now();
                    let out = { $body };
                    B::sync(ctx);
                    $bucket_us.fetch_add(
                        t0.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    $bucket_n.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    out
                } else {
                    $body
                }
            }};
        }

        // 9. Residual + pre-MLP norm.
        //
        // Sandwich families (Gemma 3): norm the attention output FIRST
        // (post_attention_layernorm in Gemma naming), add to residual,
        // then norm the residual with pre_feedforward_layernorm.
        // Legacy families: single fused residual-add + post-attn norm.
        time_tail!(TAIL_NORM_TIME_US, TAIL_NORM_CALLS, {
            if let Some(post_attn_w) = &layer.post_attn_ln_w {
                if let Some(host) = host_residual.as_deref_mut() {
                    let branch = Self::rms_norm_device_to_host(
                        &self.scratch.o_proj_out,
                        post_attn_w,
                        eps,
                        tokens,
                        h,
                    );
                    Self::add_host_residual(host, &branch);
                    Self::rms_norm_host_to_activation(
                        ctx,
                        host,
                        &layer.post_ln_w,
                        eps,
                        &mut self.scratch.norm_out,
                        tokens,
                        h,
                    );
                } else if let Some(device) = device_residual.as_mut() {
                    let device = &mut **device;
                    let branch = device_branch
                        .as_mut()
                        .expect("device F32 sandwich branch scratch missing");
                    let branch = &mut **branch;
                    if nan_trace {
                        B::rms_norm_activation_to_f32(
                            ctx,
                            &self.scratch.o_proj_out,
                            post_attn_w,
                            eps,
                            branch,
                            tokens,
                            h,
                        );
                        nt!("post_attn_norm", &*branch, tokens * h, h);
                        B::add_inplace(ctx, device, branch, tokens * h);
                    } else {
                        B::rms_norm_activation_add_to_f32(
                            ctx,
                            &self.scratch.o_proj_out,
                            post_attn_w,
                            eps,
                            device,
                            branch,
                            tokens,
                            h,
                        );
                    }
                    nt!("resid_attn", &*device, tokens * h, h);
                    B::rms_norm_f32_to_activation(
                        ctx,
                        device,
                        &layer.post_ln_w,
                        eps,
                        &mut self.scratch.norm_out,
                        tokens,
                        h,
                    );
                } else {
                    let tmp = self
                        .scratch
                        .sandwich_tmp
                        .as_mut()
                        .expect("sandwich_tmp allocated for sandwich-norm config");
                    B::rms_norm(
                        ctx,
                        &self.scratch.o_proj_out,
                        post_attn_w,
                        eps,
                        tmp,
                        tokens,
                        h,
                    );
                    nt!("post_attn_norm", &*tmp, tokens * h, h);
                    B::add_inplace(ctx, residual, tmp, tokens * h);
                    nt!("resid_attn", &*residual, tokens * h, h);
                    B::rms_norm(
                        ctx,
                        residual,
                        &layer.post_ln_w,
                        eps,
                        &mut self.scratch.norm_out,
                        tokens,
                        h,
                    );
                }
            } else {
                B::fused_add_rms_norm(
                    ctx,
                    residual,
                    &self.scratch.o_proj_out,
                    &layer.post_ln_w,
                    eps,
                    &mut self.scratch.norm_out,
                    tokens,
                    h,
                );
            }
        });

        // 10. Fused gate+up projection.
        time_tail!(TAIL_GATE_UP_TIME_US, TAIL_GATE_UP_CALLS, {
            #[cfg(feature = "cuda")]
            let _alloc_label =
                ferrum_kernels::backend::cuda::push_alloc_label("llama.forward_layer.gate_up_proj");
            layer.gate_up_proj.forward(
                ctx,
                &self.scratch.norm_out,
                &mut self.scratch.gate_up_out,
                tokens,
            );
            if let Some(adapter) =
                cache_id.and_then(|cache_id| self.active_lora_adapter_ptr_for_cache(cache_id))
            {
                // SAFETY: see qkv_proj LoRA application above.
                let applied = unsafe { &*adapter }
                    .apply_projection(
                        ctx,
                        source_li,
                        "gate_up_proj",
                        &self.scratch.norm_out,
                        &mut self.scratch.gate_up_out,
                        tokens,
                    )
                    .expect("validated LoRA gate_up_proj");
                self.lora_projection_applications += applied as u64;
            }
        });

        nt!("pre_mlp_norm", &self.scratch.norm_out, tokens * h, h);
        nt!(
            "gate_up",
            &self.scratch.gate_up_out,
            tokens * 2 * im,
            2 * im
        );
        // 11. Gated activation: SwiGLU (silu) or GeGLU (gelu_tanh, Gemma).
        time_tail!(TAIL_ACT_TIME_US, TAIL_ACT_CALLS, {
            match cfg.activation {
                Activation::GeluTanh => B::fused_gelu_tanh_mul_split(
                    ctx,
                    &self.scratch.gate_up_out,
                    &mut self.scratch.silu_out,
                    tokens,
                    im,
                ),
                _ => B::fused_silu_mul_split(
                    ctx,
                    &self.scratch.gate_up_out,
                    &mut self.scratch.silu_out,
                    tokens,
                    im,
                ),
            }
        });

        nt!("act_mul", &self.scratch.silu_out, tokens * im, im);
        // 12. Down projection.
        time_tail!(TAIL_DOWN_TIME_US, TAIL_DOWN_CALLS, {
            #[cfg(feature = "cuda")]
            let _alloc_label =
                ferrum_kernels::backend::cuda::push_alloc_label("llama.forward_layer.down_proj");
            layer.down_proj.forward(
                ctx,
                &self.scratch.silu_out,
                &mut self.scratch.mlp_out,
                tokens,
            );
            if let Some(adapter) =
                cache_id.and_then(|cache_id| self.active_lora_adapter_ptr_for_cache(cache_id))
            {
                // SAFETY: see qkv_proj LoRA application above.
                let applied = unsafe { &*adapter }
                    .apply_projection(
                        ctx,
                        source_li,
                        "down_proj",
                        &self.scratch.silu_out,
                        &mut self.scratch.mlp_out,
                        tokens,
                    )
                    .expect("validated LoRA down_proj");
                self.lora_projection_applications += applied as u64;
            }
        });
        nt!("down_proj", &self.scratch.mlp_out, tokens * h, h);

        // 13. Final residual add. Sandwich families norm the MLP output
        // (post_feedforward_layernorm) before adding it to the residual.
        time_tail!(TAIL_RESID_TIME_US, TAIL_RESID_CALLS, {
            if let Some(post_ffn_w) = &layer.post_ffn_ln_w {
                if let Some(host) = host_residual.as_deref_mut() {
                    let branch = Self::rms_norm_device_to_host(
                        &self.scratch.mlp_out,
                        post_ffn_w,
                        eps,
                        tokens,
                        h,
                    );
                    Self::add_host_residual(host, &branch);
                } else if let Some(device) = device_residual.as_mut() {
                    let device = &mut **device;
                    let branch = device_branch
                        .as_mut()
                        .expect("device F32 sandwich branch scratch missing");
                    let branch = &mut **branch;
                    if nan_trace {
                        B::rms_norm_activation_to_f32(
                            ctx,
                            &self.scratch.mlp_out,
                            post_ffn_w,
                            eps,
                            branch,
                            tokens,
                            h,
                        );
                        nt!("post_ffn_norm", &*branch, tokens * h, h);
                        B::add_inplace(ctx, device, branch, tokens * h);
                    } else {
                        B::rms_norm_activation_add_to_f32(
                            ctx,
                            &self.scratch.mlp_out,
                            post_ffn_w,
                            eps,
                            device,
                            branch,
                            tokens,
                            h,
                        );
                    }
                } else {
                    let tmp = self
                        .scratch
                        .sandwich_tmp
                        .as_mut()
                        .expect("sandwich_tmp allocated for sandwich-norm config");
                    B::rms_norm(ctx, &self.scratch.mlp_out, post_ffn_w, eps, tmp, tokens, h);
                    nt!("post_ffn_norm", &*tmp, tokens * h, h);
                    B::add_inplace(ctx, residual, tmp, tokens * h);
                }
            } else {
                B::add_inplace(ctx, residual, &self.scratch.mlp_out, tokens * h);
            }
        });
        if let Some(device) = device_residual.as_ref() {
            nt!("resid_ffn", &**device, tokens * h, h);
        } else if host_residual.is_none() {
            nt!("resid_ffn", &*residual, tokens * h, h);
        }
    }

    /// Multi-position decode-verify: run one forward pass over `tokens`
    /// starting at the cache's current end position, write their K/V
    /// into the KV cache, and return logits for ALL `tokens.len()`
    /// positions as a flat `Vec<f32>` of length `seq_len * vocab_size`.
    ///
    /// Used by speculative decoding: target receives
    /// `[last_token, draft_0, ..., draft_{N-1}]` (N+1 inputs) and produces
    /// N+1 logit rows in a single forward instead of N+1 sequential
    /// decode() calls. Positions are implicit — the model looks up
    /// `pos_offset = cache.len` the same way prefill_internal does, so
    /// chunked prefill semantics carry over for free.
    pub fn forward_verify(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        let seq_len = tokens.len();
        assert!(seq_len > 0, "forward_verify called with empty tokens");
        self.ensure_scratch(seq_len);
        self.scratch.ensure_batch_logits(&self.cfg, seq_len);
        self.ensure_kv(cache_id);

        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;

        let pos_offset = self
            .kv_caches
            .get(cache_id)
            .and_then(|layers| layers.first())
            .map(|c| K::len(c))
            .unwrap_or(0);

        let mut ctx = B::new_context();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");

        let embed = self
            .embed
            .as_ref()
            .expect("forward_verify called on backbone-only model (no embed)");
        B::embedding_lookup(&mut ctx, embed, tokens, &mut residual, h);
        if let Some(scale) = self.cfg.embed_scale {
            B::scale_inplace(&mut ctx, &mut residual, scale, seq_len * h);
        }
        let use_host_residual_shadow = self.use_host_residual_shadow();
        let use_device_residual_shadow = self.use_device_residual_shadow();
        let mut host_residual = if use_host_residual_shadow {
            B::sync(&mut ctx);
            Some(B::to_vec(&residual, seq_len * h))
        } else {
            None
        };
        let mut device_residual_shadow = if use_device_residual_shadow {
            let mut shadow = self
                .scratch
                .residual_f32_shadow
                .take()
                .expect("device F32 residual shadow scratch missing");
            B::activation_to_f32_shadow(&mut ctx, &residual, &mut shadow, seq_len * h);
            Some(shadow)
        } else {
            None
        };
        let mut device_branch_shadow = if use_device_residual_shadow {
            Some(
                self.scratch
                    .sandwich_branch_f32
                    .take()
                    .expect("device F32 sandwich branch scratch missing"),
            )
        } else {
            None
        };

        for li in self.local_layer_indices() {
            self.forward_layer_with_residual_shadow(
                &mut ctx,
                li,
                cache_id,
                &mut residual,
                pos_offset,
                seq_len,
                host_residual.as_deref_mut(),
                device_residual_shadow.as_mut(),
                device_branch_shadow.as_mut(),
            );
        }

        // RMSNorm on ALL seq_len positions (prefill_internal only norms
        // the last one; verify needs the full grid).
        if let Some(host) = host_residual.as_deref() {
            Self::rms_norm_host_to_activation(
                &mut ctx,
                host,
                &self.final_norm_w,
                self.cfg.rms_norm_eps,
                &mut self.scratch.norm_out,
                seq_len,
                h,
            );
        } else if let Some(device) = device_residual_shadow.as_ref() {
            B::rms_norm_f32_to_activation(
                &mut ctx,
                device,
                &self.final_norm_w,
                self.cfg.rms_norm_eps,
                &mut self.scratch.norm_out,
                seq_len,
                h,
            );
        } else {
            B::rms_norm(
                &mut ctx,
                &residual,
                &self.final_norm_w,
                self.cfg.rms_norm_eps,
                &mut self.scratch.norm_out,
                seq_len,
                h,
            );
        }

        // LM head applied to all positions → `seq_len * vocab` logits.
        // Reuses the existing `batch_logits` scratch (sized max_tokens *
        // vocab) so no extra allocation.
        let lm_head = self
            .lm_head
            .as_ref()
            .expect("forward_verify called on backbone-only model (no lm_head)");
        {
            #[cfg(feature = "cuda")]
            let _alloc_label =
                ferrum_kernels::backend::cuda::push_alloc_label("llama.forward_verify.lm_head");
            lm_head.forward(
                &mut ctx,
                &self.scratch.norm_out,
                &mut self.scratch.batch_logits,
                seq_len,
            );
        }

        B::sync(&mut ctx);
        if use_device_residual_shadow {
            self.scratch.residual_f32_shadow = device_residual_shadow;
            self.scratch.sandwich_branch_f32 = device_branch_shadow;
        }
        self.scratch.residual = Some(residual);
        B::to_vec(&self.scratch.batch_logits, seq_len * vocab)
    }

    /// Prefill: process `tokens` prompt tokens in a single batch, return
    /// `[vocab_size]` logits for the last position.
    ///
    /// Supports incremental prefill: if the KV cache for `cache_id` already
    /// contains earlier tokens, the new chunk's positions are computed as
    /// `[kv_len, kv_len + tokens.len())` so RoPE and causal masking stay
    /// aligned. Used by the engine's chunked-prefill path.
    pub fn prefill_internal(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        assert!(!tokens.is_empty(), "prefill called with empty token list");
        self.ensure_kv(cache_id);

        let cache_len_before = self
            .kv_caches
            .get(cache_id)
            .and_then(|layers| layers.first())
            .map(K::len)
            .unwrap_or(0);
        let mut cached_prefix_tokens = if self.runtime_env.prefix_cache && cache_len_before == 0 {
            self.try_acquire_prefix_cache(cache_id, tokens)
        } else {
            0
        };
        if cached_prefix_tokens >= tokens.len() {
            let block_size = self
                .kv_caches
                .get(cache_id)
                .and_then(|layers| layers.first())
                .map(K::block_size)
                .unwrap_or(16);
            cached_prefix_tokens = cached_prefix_tokens
                .saturating_sub(block_size)
                .min(tokens.len() - 1);
        }
        if self.runtime_env.prefix_cache && cache_len_before == 0 {
            self.record_prefix_cache_probe(cached_prefix_tokens);
        }

        if cached_prefix_tokens > 0 {
            let caches_mut = self.kv_caches.get_mut(cache_id).expect("cache present");
            let mut ctx_tmp = B::new_context();
            for cache in caches_mut.iter_mut() {
                if K::len(cache) != cached_prefix_tokens {
                    K::set_len(cache, cached_prefix_tokens);
                    if let Some(context_lens) = K::context_lens_mut(cache) {
                        B::write_typed::<u32>(
                            &mut ctx_tmp,
                            context_lens,
                            &[cached_prefix_tokens as u32],
                        );
                    }
                }
            }
            B::sync(&mut ctx_tmp);
        }

        let suffix_tokens = &tokens[cached_prefix_tokens..];
        let seq_len = suffix_tokens.len();
        assert!(
            seq_len > 0,
            "prefix cache must leave at least one suffix token"
        );
        self.ensure_scratch(seq_len);

        // Starting position for this chunk — 0 for a fresh prefill, kv_len
        // for the second+ chunk of a split prefill.
        let pos_offset = self
            .kv_caches
            .get(cache_id)
            .and_then(|layers| layers.first())
            .map(|c| K::len(c))
            .unwrap_or(0);

        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let mut ctx = B::new_context();

        // Move `residual` out of `scratch` to work around the borrow checker:
        // `forward_layer` re-borrows `&mut self` to reach `self.layers` /
        // `self.kv_caches`, which would conflict with an outstanding
        // `&mut self.scratch.residual`. Use Option::take to move it out
        // (no placeholder alloc → no transient cuMemFreeAsync that could
        // corrupt stream pool state after graph ops on Blackwell).
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        let embed = self
            .embed
            .as_ref()
            .expect("prefill_internal called on backbone-only model (no embed)");
        B::embedding_lookup(&mut ctx, embed, suffix_tokens, &mut residual, h);
        if let Some(scale) = self.cfg.embed_scale {
            B::scale_inplace(&mut ctx, &mut residual, scale, suffix_tokens.len() * h);
        }
        let use_host_residual_shadow = self.use_host_residual_shadow();
        let use_device_residual_shadow = self.use_device_residual_shadow();
        let mut host_residual = if use_host_residual_shadow {
            B::sync(&mut ctx);
            Some(B::to_vec(&residual, seq_len * h))
        } else {
            None
        };
        let mut device_residual_shadow = if use_device_residual_shadow {
            let mut shadow = self
                .scratch
                .residual_f32_shadow
                .take()
                .expect("device F32 residual shadow scratch missing");
            B::activation_to_f32_shadow(&mut ctx, &residual, &mut shadow, seq_len * h);
            Some(shadow)
        } else {
            None
        };
        let mut device_branch_shadow = if use_device_residual_shadow {
            Some(
                self.scratch
                    .sandwich_branch_f32
                    .take()
                    .expect("device F32 sandwich branch scratch missing"),
            )
        } else {
            None
        };

        let prefill_profile = self.runtime_env.prefill_op_profile;
        let prefill_t0 = if prefill_profile {
            B::sync(&mut ctx);
            ATTN_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            ATTN_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            QKR_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            QKR_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            MATMUL_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            MATMUL_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            NORM_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            NORM_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            OTHER_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            OTHER_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            TAIL_NORM_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            TAIL_NORM_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            TAIL_GATE_UP_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            TAIL_GATE_UP_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            TAIL_ACT_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            TAIL_ACT_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            TAIL_DOWN_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            TAIL_DOWN_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            TAIL_RESID_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            TAIL_RESID_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            Some(std::time::Instant::now())
        } else {
            None
        };

        // W2 L1 gate: per-layer residual dumps for HF comparison.
        let layer_dump = self.runtime_env.layer_dump_dir.clone();
        let dump = |ctx: &mut B::Context, buf: &B::Buffer, name: &str| {
            if let Some(dir) = &layer_dump {
                B::sync(ctx);
                let v = B::to_vec(buf, seq_len * h);
                write_f32_dump(dir, name, &v);
            }
        };
        if let Some(dir) = &layer_dump {
            if let Some(host) = host_residual.as_deref() {
                write_f32_dump(dir, "embed", host);
            } else if let Some(device) = device_residual_shadow.as_ref() {
                dump(&mut ctx, device, "embed");
            } else {
                dump(&mut ctx, &residual, "embed");
            }
        }

        for li in self.local_layer_indices() {
            self.forward_layer_with_residual_shadow(
                &mut ctx,
                li,
                cache_id,
                &mut residual,
                pos_offset,
                seq_len,
                host_residual.as_deref_mut(),
                device_residual_shadow.as_mut(),
                device_branch_shadow.as_mut(),
            );
            if layer_dump.is_some() {
                if let Some(host) = host_residual.as_deref() {
                    if let Some(dir) = &layer_dump {
                        write_f32_dump(dir, &format!("layer_{li:02}"), host);
                    }
                } else if let Some(device) = device_residual_shadow.as_ref() {
                    dump(&mut ctx, device, &format!("layer_{li:02}"));
                } else {
                    dump(&mut ctx, &residual, &format!("layer_{li:02}"));
                }
            }
        }

        if let Some(t0) = prefill_t0 {
            B::sync(&mut ctx);
            let total_us = t0.elapsed().as_micros() as u64;
            let attn_us = ATTN_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let attn_n = ATTN_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let qkr_us = QKR_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let qkr_n = QKR_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let mm_us = MATMUL_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let mm_n = MATMUL_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let norm_us = NORM_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let norm_n = NORM_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let other_us = OTHER_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let other_n = OTHER_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let tail_norm_us = TAIL_NORM_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let tail_norm_n = TAIL_NORM_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let tail_gate_up_us =
                TAIL_GATE_UP_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let tail_gate_up_n = TAIL_GATE_UP_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let tail_act_us = TAIL_ACT_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let tail_act_n = TAIL_ACT_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let tail_down_us = TAIL_DOWN_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let tail_down_n = TAIL_DOWN_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let tail_resid_us = TAIL_RESID_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let tail_resid_n = TAIL_RESID_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "[prefill-profile] tokens={} layers total={} ms",
                seq_len,
                total_us / 1000
            );
            let bucket = |label: &str, n: u64, us: u64| {
                if n > 0 {
                    eprintln!(
                        "[prefill-profile] {label}: {} calls {} ms (avg {} us)",
                        n,
                        us / 1000,
                        us / n
                    );
                }
            };
            bucket("flash_attn", attn_n, attn_us);
            bucket("qk_norm_rope", qkr_n, qkr_us);
            bucket("matmuls", mm_n, mm_us);
            bucket("norms", norm_n, norm_us);
            bucket("other", other_n, other_us);
            bucket("tail_norm", tail_norm_n, tail_norm_us);
            bucket("tail_gate_up", tail_gate_up_n, tail_gate_up_us);
            bucket("tail_act", tail_act_n, tail_act_us);
            bucket("tail_down", tail_down_n, tail_down_us);
            bucket(
                "tail_mlp",
                tail_gate_up_n.max(tail_act_n).max(tail_down_n),
                tail_gate_up_us + tail_act_us + tail_down_us,
            );
            bucket("tail_resid", tail_resid_n, tail_resid_us);
        }

        if let Some(host) = host_residual.as_deref() {
            let last = &host[(seq_len - 1) * h..seq_len * h];
            Self::rms_norm_host_to_activation(
                &mut ctx,
                last,
                &self.final_norm_w,
                self.cfg.rms_norm_eps,
                &mut self.scratch.last_normed,
                1,
                h,
            );
        } else if let Some(device) = device_residual_shadow.as_mut() {
            let branch = device_branch_shadow
                .as_mut()
                .expect("device F32 sandwich branch scratch missing");
            B::copy_slice(&mut ctx, device, (seq_len - 1) * h, branch, 0, h);
            B::rms_norm_f32_to_activation(
                &mut ctx,
                branch,
                &self.final_norm_w,
                self.cfg.rms_norm_eps,
                &mut self.scratch.last_normed,
                1,
                h,
            );
        } else {
            // Take the last token's hidden state: residual[(seq_len-1)*h .. seq_len*h]
            B::copy_slice(
                &mut ctx,
                &residual,
                (seq_len - 1) * h,
                &mut self.scratch.last_hidden,
                0,
                h,
            );

            // Final RMSNorm on the last hidden.
            B::rms_norm(
                &mut ctx,
                &self.scratch.last_hidden,
                &self.final_norm_w,
                self.cfg.rms_norm_eps,
                &mut self.scratch.last_normed,
                1,
                h,
            );
        }

        // LM head (m=1 — triggers GEMV on MetalBackend).
        let lm_head = self
            .lm_head
            .as_ref()
            .expect("prefill_internal called on backbone-only model (no lm_head)");
        {
            #[cfg(feature = "cuda")]
            let _alloc_label =
                ferrum_kernels::backend::cuda::push_alloc_label("llama.prefill.lm_head");
            lm_head.forward(
                &mut ctx,
                &self.scratch.last_normed,
                &mut self.scratch.logits,
                1,
            );
        }

        // Sync ctx before to_vec: on Metal, `to_vec` just reads the shared
        // buffer's CPU pointer without flushing the command buffer, so the
        // GPU must complete all pending work first or we read stale/random
        // data. CUDA's to_vec does an internal stream.synchronize, making
        // the call redundant there (~50µs/step cost), but correctness on
        // Metal requires the explicit flush here.
        B::sync(&mut ctx);

        if let Some(dir) = &layer_dump {
            let v = B::to_vec(&self.scratch.logits, vocab);
            write_f32_dump(dir, "logits", &v);
        }

        // Restore residual into scratch for reuse on the next call.
        if use_device_residual_shadow {
            self.scratch.residual_f32_shadow = device_residual_shadow;
            self.scratch.sandwich_branch_f32 = device_branch_shadow;
        }
        self.scratch.residual = Some(residual);
        if self.runtime_env.prefix_cache && cache_len_before == 0 {
            self.register_prefix_cache(cache_id, tokens, cached_prefix_tokens);
        }

        B::to_vec(&self.scratch.logits, vocab)
    }

    /// Decode: process 1 token at position `pos`, return `[vocab_size]` logits.
    pub fn decode_internal(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        self.ensure_scratch(1);
        self.ensure_kv(cache_id);

        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;

        // Context creation is cheap (CUDA reuses the process-global stream).
        // The captured graph lives in a process-global slot, not on ctx.
        let mut ctx = B::new_context();

        // Graph capture is opt-in via FERRUM_CUDA_GRAPH=1. Replay is currently
        // single-request-only on Blackwell + CUDA 12.8 (see
        // docs/phase-e-cuda-status.md). In pure eager mode, we skip the
        // per-step device-state memcpy_htod trio entirely.
        const GRAPH_WARMUP: usize = 3;
        let use_host_residual_shadow = self.use_host_residual_shadow();
        let use_device_residual_shadow = self.use_device_residual_shadow();
        let graph_enabled = self.runtime_env.cuda_graph
            && self.runtime_env.graph_capture_allowed()
            && !use_host_residual_shadow
            && !use_device_residual_shadow;

        if graph_enabled {
            // Refresh device-side dynamic state (token/pos/kv_len) before
            // replay — captured graph reads these from device buffers.
            B::set_decode_state(&mut ctx, token, pos);

            // Fast path: graph replay (if available). Single-item path
            // uses key=SINGLE_ITEM_GRAPH_KEY (0) — separate from the
            // batched path's m_padded keys.
            match B::replay_graph(&mut ctx, SINGLE_ITEM_GRAPH_KEY) {
                Ok(true) => {
                    B::sync(&mut ctx);
                    return B::to_vec(&self.scratch.logits, vocab);
                }
                Ok(false) => { /* no graph yet, fall through to eager */ }
                Err(_) => { /* backend error or unsupported, eager */ }
            }
        }

        let should_capture =
            graph_enabled && !self.graph_capture_failed && self.graph_warmup >= GRAPH_WARMUP;

        if should_capture {
            B::set_dev_state_mode(&mut ctx, true);
            if B::begin_graph_capture(&mut ctx).is_err() {
                self.graph_capture_failed = true;
                B::set_dev_state_mode(&mut ctx, false);
            }
        }

        // Eager forward (records into graph if capture is active).
        // mem::replace needs a placeholder. B::alloc(0) was our choice but
        // cuMemAllocFromPoolAsync(stream, 0) can return CUDA_ERROR_INVALID_VALUE
        // on Blackwell after graph replay corrupts the pool state. Size-1 is
        // always valid and costs 2 bytes of transient VRAM per decode step.
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        let embed = self
            .embed
            .as_ref()
            .expect("decode_internal called on backbone-only model (no embed)");
        B::embedding_lookup(&mut ctx, embed, &[token], &mut residual, h);
        if let Some(scale) = self.cfg.embed_scale {
            B::scale_inplace(&mut ctx, &mut residual, scale, h);
        }
        let mut host_residual = if use_host_residual_shadow {
            B::sync(&mut ctx);
            Some(B::to_vec(&residual, h))
        } else {
            None
        };
        let mut device_residual_shadow = if use_device_residual_shadow {
            let mut shadow = self
                .scratch
                .residual_f32_shadow
                .take()
                .expect("device F32 residual shadow scratch missing");
            B::activation_to_f32_shadow(&mut ctx, &residual, &mut shadow, h);
            Some(shadow)
        } else {
            None
        };
        let mut device_branch_shadow = if use_device_residual_shadow {
            Some(
                self.scratch
                    .sandwich_branch_f32
                    .take()
                    .expect("device F32 sandwich branch scratch missing"),
            )
        } else {
            None
        };

        // Per-layer wall-time profile (env-gated, off by default — adds
        // a B::sync between layers which serializes the pipeline). Helps
        // localise non-matmul bottlenecks during perf work.
        let layer_profile = self.runtime_env.decode_layer_profile;
        let mut layer_times = if layer_profile {
            Some(Vec::with_capacity(self.local_layer_count()))
        } else {
            None
        };

        for li in self.local_layer_indices() {
            if layer_profile {
                let t0 = std::time::Instant::now();
                self.forward_layer_with_residual_shadow(
                    &mut ctx,
                    li,
                    cache_id,
                    &mut residual,
                    pos as usize,
                    1,
                    host_residual.as_deref_mut(),
                    device_residual_shadow.as_mut(),
                    device_branch_shadow.as_mut(),
                );
                B::sync(&mut ctx);
                let elapsed_us = t0.elapsed().as_micros() as u64;
                if let Some(v) = layer_times.as_mut() {
                    v.push(elapsed_us);
                }
            } else {
                self.forward_layer_with_residual_shadow(
                    &mut ctx,
                    li,
                    cache_id,
                    &mut residual,
                    pos as usize,
                    1,
                    host_residual.as_deref_mut(),
                    device_residual_shadow.as_mut(),
                    device_branch_shadow.as_mut(),
                );
            }
        }
        if let Some(times) = layer_times.take() {
            let sum: u64 = times.iter().sum();
            let avg = sum / times.len() as u64;
            let mn = *times.iter().min().unwrap_or(&0);
            let mx = *times.iter().max().unwrap_or(&0);
            eprintln!(
                "[layer-profile] {} layers total={} ms avg={} us min={} us max={} us",
                times.len(),
                sum / 1000,
                avg,
                mn,
                mx
            );
            for (i, t) in times.iter().enumerate() {
                eprint!("L{i}={}ms ", t / 1000);
                if (i + 1) % 6 == 0 {
                    eprintln!();
                }
            }
            eprintln!();
            let attn_us = ATTN_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let attn_n = ATTN_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let qkr_us = QKR_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let qkr_n = QKR_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let mm_us = MATMUL_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let mm_n = MATMUL_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let norm_us = NORM_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let norm_n = NORM_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let other_us = OTHER_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let other_n = OTHER_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "[op-profile] flash_attn: {} calls {} ms (avg {} us)",
                attn_n,
                attn_us / 1000,
                if attn_n > 0 { attn_us / attn_n } else { 0 }
            );
            eprintln!(
                "[op-profile] qk_norm_rope: {} calls {} ms (avg {} us)",
                qkr_n,
                qkr_us / 1000,
                if qkr_n > 0 { qkr_us / qkr_n } else { 0 }
            );
            eprintln!(
                "[op-profile] matmuls (Linear::forward): {} calls {} ms (avg {} us)",
                mm_n,
                mm_us / 1000,
                if mm_n > 0 { mm_us / mm_n } else { 0 }
            );
            eprintln!(
                "[op-profile] norms (rms+fused_add_rms): {} calls {} ms (avg {} us)",
                norm_n,
                norm_us / 1000,
                if norm_n > 0 { norm_us / norm_n } else { 0 }
            );
            eprintln!(
                "[op-profile] other (split_qkv, kv_append, transpose, silu, add): {} calls {} ms (avg {} us)",
                other_n, other_us / 1000, if other_n > 0 { other_us / other_n } else { 0 }
            );
        }

        if let Some(host) = host_residual.as_deref() {
            Self::rms_norm_host_to_activation(
                &mut ctx,
                host,
                &self.final_norm_w,
                self.cfg.rms_norm_eps,
                &mut self.scratch.last_normed,
                1,
                h,
            );
        } else if let Some(device) = device_residual_shadow.as_ref() {
            B::rms_norm_f32_to_activation(
                &mut ctx,
                device,
                &self.final_norm_w,
                self.cfg.rms_norm_eps,
                &mut self.scratch.last_normed,
                1,
                h,
            );
        } else {
            B::rms_norm(
                &mut ctx,
                &residual,
                &self.final_norm_w,
                self.cfg.rms_norm_eps,
                &mut self.scratch.last_normed,
                1,
                h,
            );
        }

        let lm_head = self
            .lm_head
            .as_ref()
            .expect("decode_internal called on backbone-only model (no lm_head)");
        {
            #[cfg(feature = "cuda")]
            let _alloc_label =
                ferrum_kernels::backend::cuda::push_alloc_label("llama.decode.lm_head");
            lm_head.forward(
                &mut ctx,
                &self.scratch.last_normed,
                &mut self.scratch.logits,
                1,
            );
        }

        if should_capture && B::graph_capture_in_flight(&ctx) {
            if B::end_graph_capture(&mut ctx, SINGLE_ITEM_GRAPH_KEY).is_err() {
                self.graph_capture_failed = true;
            } else {
                // Stream capture mode RECORDS ops into the graph without
                // executing them. scratch.logits still holds the previous
                // step's value. Replay the just-captured graph once to
                // actually execute and produce this step's logits. Without
                // this, the capture step's to_vec returns stale logits,
                // yielding a 1-token offset in the generated sequence.
                if B::replay_graph(&mut ctx, SINGLE_ITEM_GRAPH_KEY).is_err() {
                    self.graph_capture_failed = true;
                }
            }
            B::set_dev_state_mode(&mut ctx, false);
        } else {
            self.graph_warmup += 1;
        }

        // Sync ctx before to_vec: on Metal, `to_vec` just reads the shared
        // buffer's CPU pointer without flushing the command buffer, so the
        // GPU must complete all pending work first or we read stale/random
        // data. CUDA's to_vec does an internal stream.synchronize, making
        // the call redundant there (~50µs/step cost), but correctness on
        // Metal requires the explicit flush here.
        B::sync(&mut ctx);
        if use_device_residual_shadow {
            self.scratch.residual_f32_shadow = device_residual_shadow;
            self.scratch.sandwich_branch_f32 = device_branch_shadow;
        }
        self.scratch.residual = Some(residual);

        B::to_vec(&self.scratch.logits, vocab)
    }

    fn stage_tokens_to_hidden(
        &mut self,
        cache_id: &str,
        tokens: &[u32],
        pos_offset: usize,
    ) -> Vec<f32> {
        let seq_len = tokens.len();
        assert!(seq_len > 0, "stage token forward called with zero length");
        self.ensure_scratch(seq_len);
        self.ensure_kv(cache_id);

        let h = self.cfg.hidden_size;
        let mut ctx = B::new_context();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        let embed = self
            .embed
            .as_ref()
            .expect("stage token forward called on stage without embedding");
        B::embedding_lookup(&mut ctx, embed, tokens, &mut residual, h);
        if let Some(scale) = self.cfg.embed_scale {
            B::scale_inplace(&mut ctx, &mut residual, scale, tokens.len() * h);
        }
        let use_host_residual_shadow = self.use_host_residual_shadow();
        let use_device_residual_shadow = self.use_device_residual_shadow();
        let mut host_residual = if use_host_residual_shadow {
            B::sync(&mut ctx);
            Some(B::to_vec(&residual, seq_len * h))
        } else {
            None
        };
        let mut device_residual_shadow = if use_device_residual_shadow {
            let mut shadow = self
                .scratch
                .residual_f32_shadow
                .take()
                .expect("device F32 residual shadow scratch missing");
            B::activation_to_f32_shadow(&mut ctx, &residual, &mut shadow, seq_len * h);
            Some(shadow)
        } else {
            None
        };
        let mut device_branch_shadow = if use_device_residual_shadow {
            Some(
                self.scratch
                    .sandwich_branch_f32
                    .take()
                    .expect("device F32 sandwich branch scratch missing"),
            )
        } else {
            None
        };

        for li in self.local_layer_indices() {
            self.forward_layer_with_residual_shadow(
                &mut ctx,
                li,
                cache_id,
                &mut residual,
                pos_offset,
                seq_len,
                host_residual.as_deref_mut(),
                device_residual_shadow.as_mut(),
                device_branch_shadow.as_mut(),
            );
        }

        B::sync(&mut ctx);
        let out = if let Some(host) = host_residual.as_deref() {
            host.to_vec()
        } else if let Some(device) = device_residual_shadow.as_ref() {
            B::to_vec(device, seq_len * h)
        } else {
            B::to_vec(&residual, seq_len * h)
        };
        if use_device_residual_shadow {
            self.scratch.residual_f32_shadow = device_residual_shadow;
            self.scratch.sandwich_branch_f32 = device_branch_shadow;
        }
        self.scratch.residual = Some(residual);
        out
    }

    /// Run the first layer-split stage over prompt tokens and return all
    /// output hidden states for the next stage.
    pub fn prefill_stage_tokens_to_hidden(
        &mut self,
        cache_id: &str,
        tokens: &[u32],
        pos_offset: usize,
    ) -> Vec<f32> {
        self.stage_tokens_to_hidden(cache_id, tokens, pos_offset)
    }

    /// Decode-side first-stage token forward. Returns one hidden row for the
    /// next stage.
    pub fn decode_stage_token_to_hidden(
        &mut self,
        cache_id: &str,
        token: u32,
        pos: u32,
    ) -> Vec<f32> {
        self.stage_tokens_to_hidden(cache_id, &[token], pos as usize)
    }

    fn stage_hidden_from_host(
        &mut self,
        cache_id: &str,
        hidden: &[f32],
        seq_len: usize,
        pos_offset: usize,
    ) -> Vec<f32> {
        self.stage_hidden_from_host_with_timing(cache_id, hidden, seq_len, pos_offset)
            .0
    }

    fn stage_hidden_from_host_with_timing(
        &mut self,
        cache_id: &str,
        hidden: &[f32],
        seq_len: usize,
        pos_offset: usize,
    ) -> (Vec<f32>, LlamaStageHiddenBridgeTiming) {
        let h = self.cfg.hidden_size;
        assert_eq!(
            hidden.len(),
            seq_len * h,
            "hidden length {} != seq_len * hidden_size {}",
            hidden.len(),
            seq_len * h
        );
        assert!(seq_len > 0, "stage hidden forward called with zero length");

        self.ensure_scratch(seq_len);
        self.ensure_kv(cache_id);

        let mut ctx = B::new_context();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");

        let bridge_t0 = std::time::Instant::now();
        let host_copy_t0 = std::time::Instant::now();
        let hidden_buf = B::from_slice(hidden);
        let host_copy_us = elapsed_micros_u64_floor1(host_copy_t0);
        let device_copy_t0 = std::time::Instant::now();
        B::copy_slice(&mut ctx, &hidden_buf, 0, &mut residual, 0, seq_len * h);
        let device_copy_us = elapsed_micros_u64_floor1(device_copy_t0);
        let bridge_timing = LlamaStageHiddenBridgeTiming {
            bridge_us: elapsed_micros_u64_floor1(bridge_t0),
            host_copy_us,
            device_copy_us,
        };

        for li in self.local_layer_indices() {
            self.forward_layer(&mut ctx, li, cache_id, &mut residual, pos_offset, seq_len);
        }

        B::sync(&mut ctx);
        let out = B::to_vec(&residual, seq_len * h);
        self.scratch.residual = Some(residual);
        (out, bridge_timing)
    }

    /// Run this layer-split stage over prefill hidden states supplied by the
    /// previous stage. Returns all output hidden states in row-major
    /// `[seq_len, hidden_size]` form for the next stage.
    pub fn prefill_stage_hidden_from_host(
        &mut self,
        cache_id: &str,
        hidden: &[f32],
        seq_len: usize,
        pos_offset: usize,
    ) -> Vec<f32> {
        self.stage_hidden_from_host(cache_id, hidden, seq_len, pos_offset)
    }

    /// Decode-side companion for one hidden state row supplied by the previous
    /// stage. Returns this stage's output hidden row.
    pub fn decode_stage_hidden_from_host(
        &mut self,
        cache_id: &str,
        hidden: &[f32],
        pos: u32,
    ) -> Vec<f32> {
        self.stage_hidden_from_host(cache_id, hidden, 1, pos as usize)
    }

    pub(crate) fn decode_stage_hidden_from_host_with_timing(
        &mut self,
        cache_id: &str,
        hidden: &[f32],
        pos: u32,
    ) -> (Vec<f32>, LlamaStageHiddenBridgeTiming) {
        self.stage_hidden_from_host_with_timing(cache_id, hidden, 1, pos as usize)
    }

    /// Apply final norm + lm_head to one hidden row. Used by the last pipeline
    /// stage after local transformer layers have produced the final hidden
    /// state for sampling.
    pub fn logits_from_hidden(&mut self, hidden: &[f32]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        assert_eq!(
            hidden.len(),
            h,
            "hidden length {} != hidden_size {}",
            hidden.len(),
            h
        );
        self.ensure_scratch(1);

        let mut ctx = B::new_context();
        let hidden_buf = B::from_slice(hidden);
        B::rms_norm(
            &mut ctx,
            &hidden_buf,
            &self.final_norm_w,
            self.cfg.rms_norm_eps,
            &mut self.scratch.last_normed,
            1,
            h,
        );
        let lm_head = self
            .lm_head
            .as_ref()
            .expect("logits_from_hidden called on stage without lm_head");
        {
            #[cfg(feature = "cuda")]
            let _alloc_label =
                ferrum_kernels::backend::cuda::push_alloc_label("llama.logits_from_hidden.lm_head");
            lm_head.forward(
                &mut ctx,
                &self.scratch.last_normed,
                &mut self.scratch.logits,
                1,
            );
        }
        B::sync(&mut ctx);
        B::to_vec(&self.scratch.logits, vocab)
    }

    /// Prefill with pre-computed embeddings instead of token IDs.
    ///
    /// Used by models that embed inputs outside the LLM (e.g. Qwen3-TTS
    /// mixes text-embedding + codec-embedding before feeding the LM).
    /// Skips `final_norm` + `lm_head`; returns the last position's pre-norm
    /// hidden state. Caller applies its own output head.
    ///
    /// `embeds` is row-major `[seq_len * hidden_size]`, f32.
    pub fn prefill_from_embeds(
        &mut self,
        cache_id: &str,
        embeds: &[f32],
        seq_len: usize,
    ) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        assert_eq!(
            embeds.len(),
            seq_len * h,
            "embeds length {} != seq_len * hidden_size {}",
            embeds.len(),
            seq_len * h
        );
        assert!(seq_len > 0, "prefill_from_embeds called with zero length");

        self.ensure_scratch(seq_len);
        self.ensure_kv(cache_id);

        let mut ctx = B::new_context();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");

        // Upload embeds → residual[0 .. seq_len*h].
        let embed_buf = B::from_slice(embeds);
        B::copy_slice(&mut ctx, &embed_buf, 0, &mut residual, 0, seq_len * h);

        for li in self.local_layer_indices() {
            self.forward_layer(&mut ctx, li, cache_id, &mut residual, 0, seq_len);
        }

        B::copy_slice(
            &mut ctx,
            &residual,
            (seq_len - 1) * h,
            &mut self.scratch.last_hidden,
            0,
            h,
        );
        B::sync(&mut ctx);
        self.scratch.residual = Some(residual);
        B::to_vec(&self.scratch.last_hidden, h)
    }

    /// Decode with a single pre-computed embedding (shape `[hidden]`).
    /// Returns the pre-norm hidden state for the position `pos`. Caller
    /// applies final norm + its own output head.
    pub fn decode_from_embed(&mut self, cache_id: &str, embed: &[f32], pos: u32) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        assert_eq!(
            embed.len(),
            h,
            "embed length {} != hidden_size {}",
            embed.len(),
            h
        );

        self.ensure_scratch(1);
        self.ensure_kv(cache_id);

        let mut ctx = B::new_context();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");

        let embed_buf = B::from_slice(embed);
        B::copy_slice(&mut ctx, &embed_buf, 0, &mut residual, 0, h);

        for li in self.local_layer_indices() {
            self.forward_layer(&mut ctx, li, cache_id, &mut residual, pos as usize, 1);
        }

        B::copy_slice(&mut ctx, &residual, 0, &mut self.scratch.last_hidden, 0, h);
        B::sync(&mut ctx);
        self.scratch.residual = Some(residual);
        B::to_vec(&self.scratch.last_hidden, h)
    }

    /// Variant of `prefill_from_embeds` that applies `final_norm` to every
    /// position and returns the whole `[seq_len * hidden_size]` vector.
    /// Accepts `pos_offset` so callers can continue an existing sequence
    /// (e.g. Qwen3-TTS voice-clone: one prefill for the role prefix, a
    /// follow-up prefill for the reference-audio ICL block, then
    /// autoregressive decoding — all against the same KV cache).
    ///
    /// Used by TTS where `forward_step` in the candle-based wrapper is
    /// expected to return **post-norm all-positions** hidden state so
    /// `codec_head` can be applied on candle side.
    pub fn prefill_all_post_norm(
        &mut self,
        cache_id: &str,
        embeds: &[f32],
        seq_len: usize,
        pos_offset: usize,
    ) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        assert_eq!(
            embeds.len(),
            seq_len * h,
            "embeds length {} != seq_len * hidden_size {}",
            embeds.len(),
            seq_len * h
        );
        assert!(seq_len > 0);

        self.ensure_scratch(seq_len);
        self.ensure_kv(cache_id);

        let mut ctx = B::new_context();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");

        let embed_buf = B::from_slice(embeds);
        B::copy_slice(&mut ctx, &embed_buf, 0, &mut residual, 0, seq_len * h);

        for li in self.local_layer_indices() {
            self.forward_layer(&mut ctx, li, cache_id, &mut residual, pos_offset, seq_len);
        }

        // Apply final_norm over all seq_len positions → scratch.norm_out.
        B::rms_norm(
            &mut ctx,
            &residual,
            &self.final_norm_w,
            self.cfg.rms_norm_eps,
            &mut self.scratch.norm_out,
            seq_len,
            h,
        );
        B::sync(&mut ctx);
        self.scratch.residual = Some(residual);
        B::to_vec(&self.scratch.norm_out, seq_len * h)
    }

    /// Decode-side companion to `prefill_all_post_norm`. Runs a single-token
    /// decode step at `pos`, applies `final_norm`, and returns the post-norm
    /// hidden state `[hidden_size]`.
    pub fn decode_post_norm_from_embed(
        &mut self,
        cache_id: &str,
        embed: &[f32],
        pos: u32,
    ) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        assert_eq!(embed.len(), h);

        self.ensure_scratch(1);
        self.ensure_kv(cache_id);

        let mut ctx = B::new_context();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");

        let embed_buf = B::from_slice(embed);
        B::copy_slice(&mut ctx, &embed_buf, 0, &mut residual, 0, h);

        for li in self.local_layer_indices() {
            self.forward_layer(&mut ctx, li, cache_id, &mut residual, pos as usize, 1);
        }

        B::rms_norm(
            &mut ctx,
            &residual,
            &self.final_norm_w,
            self.cfg.rms_norm_eps,
            &mut self.scratch.last_normed,
            1,
            h,
        );
        B::sync(&mut ctx);
        self.scratch.residual = Some(residual);
        B::to_vec(&self.scratch.last_normed, h)
    }
}

// FP16 DecoderOnlyLLM impl — full path with batched + unified-forward overrides.
impl<B: MoeLlmBackend> DecoderOnlyLLM for LlamaFamilyModel<B, KvFp16> {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.runtime_cfg
    }

    fn cache_metrics_snapshot(&self) -> Option<serde_json::Value> {
        Some(self.prefix_cache_snapshot_json())
    }

    fn lora_metrics_snapshot(&self) -> Option<serde_json::Value> {
        Some(self.lora_metrics_snapshot_json())
    }

    fn set_lora_adapter_for_cache(
        &mut self,
        cache_id: &str,
        adapter: Option<ActiveLoraAdapter>,
    ) -> std::result::Result<(), ferrum_types::FerrumError> {
        if let Some(adapter) = adapter {
            self.ensure_lora_adapter_loaded(adapter.clone())?;
            self.lora_cache_adapters
                .insert(cache_id.to_string(), adapter.name);
        } else {
            self.lora_cache_adapters.remove(cache_id);
        }
        Ok(())
    }

    fn prepare(&mut self, cache_id: &str, max_tokens: usize) {
        self.ensure_scratch(max_tokens);
        self.ensure_kv(cache_id);

        const WARMUP_CACHE: &str = "__ferrum_warmup__";
        let _ = self.prefill_internal(WARMUP_CACHE, &[0u32]);
        if let Some(mut caches) = self.kv_caches.remove(WARMUP_CACHE) {
            if let Some(alloc_arc) = self.paged_block_alloc.as_ref() {
                let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(c0) = caches.first() {
                    if !c0.paged_block_indices.is_empty() {
                        alloc.free(&c0.paged_block_indices);
                    }
                }
                for c in caches.iter_mut() {
                    c.paged_block_indices.clear();
                }
            }
            self.kv_free_pool.push(caches);
        }
    }

    fn prepare_kv_capacity(&mut self, cache_id: &str, capacity_hint: usize) {
        self.ensure_kv_with_capacity_hint(cache_id, Some(capacity_hint));
    }

    fn kv_capacity(&self) -> usize {
        self.runtime_env.kv_capacity_for_model(self.cfg.max_seq_len)
    }

    fn prefill(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        self.prefill_internal(cache_id, tokens)
    }

    fn decode(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        self.decode_internal(cache_id, token, pos)
    }

    fn decode_batch(&mut self, batch: &[(String, u32, u32)]) -> Vec<Vec<f32>> {
        self.decode_batch_with_full_logits(batch, false)
    }

    fn decode_batch_with_full_logits(
        &mut self,
        batch: &[(String, u32, u32)],
        force_full_logits: bool,
    ) -> Vec<Vec<f32>> {
        self.decode_batch_internal_with_full_logits(batch, force_full_logits)
    }

    fn decode_batch_with_logits_policy(
        &mut self,
        batch: &[(String, u32, u32)],
        policies: &[LogitsReturnPolicy],
    ) -> Vec<Vec<f32>> {
        self.decode_batch_internal_with_logits_policy(batch, policies)
    }

    fn unified_forward(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
    ) -> std::result::Result<Vec<Option<Vec<f32>>>, ferrum_types::FerrumError> {
        self.unified_forward_with_logits_policy(items, &[])
    }

    fn unified_forward_with_logits_policy(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
        logits_policies: &[LogitsReturnPolicy],
    ) -> std::result::Result<Vec<Option<Vec<f32>>>, ferrum_types::FerrumError> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        if self.runtime_env.prefix_cache
            && items
                .iter()
                .any(|(_, tokens, pos_offset, _)| *pos_offset == 0 && tokens.len() > 1)
        {
            return Err(ferrum_types::FerrumError::unsupported(
                "LlamaFamilyModel::unified_forward: fresh prefill with prefix cache enabled \
                 routes through prefill_internal so real paged-block KV reuse can probe and \
                 register block hashes",
            ));
        }
        if !self.supports_varlen_qkv {
            let reason = unified_varlen_qkv_unsupported_reason(
                &self.cfg,
                B::supports_varlen_qkv(),
                B::supports_device_f32_residual_shadow(),
            )
            .unwrap_or(
                "LlamaFamilyModel::unified_forward: varlen QKV support disabled. Engine will \
                     fall back to per-item dispatch.",
            );
            return Err(ferrum_types::FerrumError::unsupported(reason));
        }
        if items
            .iter()
            .any(|(cache_id, _, _, _)| self.active_lora_adapter_for_cache(cache_id).is_some())
        {
            return Err(ferrum_types::FerrumError::unsupported(
                "LlamaFamilyModel::unified_forward: active LoRA adapter routes through \
                 per-item dispatch until unified LoRA supports row-selective adapters.",
            ));
        }
        self.ensure_kv(&items[0].0);
        if self.paged_pools.is_none() {
            return Err(ferrum_types::FerrumError::unsupported(
                "LlamaFamilyModel::unified_forward: paged KV required; \
                 enable via FERRUM_METAL_PAGED_KV=1 (cross-backend env). \
                 Engine will fall back to per-item dispatch.",
            ));
        }
        for (cid, _, _, _) in items {
            self.ensure_kv(cid);
            if !self.kv_caches.contains_key(cid) {
                return Err(ferrum_types::FerrumError::resource_exhausted(format!(
                    "paged KV pool exhausted for cache_id={cid:?}; back off"
                )));
            }
        }
        let logits_policies = (logits_policies.len() == items.len()).then_some(logits_policies);
        Ok(self.unified_forward_internal(items, logits_policies))
    }

    fn forward_verify(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        LlamaFamilyModel::<B, KvFp16>::forward_verify(self, cache_id, tokens)
    }

    fn truncate_kv(&mut self, cache_id: &str, new_len: usize) {
        if let Some(caches) = self.kv_caches.get_mut(cache_id) {
            for c in caches.iter_mut() {
                if new_len < c.len {
                    c.len = new_len;
                }
            }
        }
        let mut ctx = B::new_context();
        B::reset_graph(&mut ctx, SINGLE_ITEM_GRAPH_KEY);
        self.graph_warmup = 0;
        self.graph_capture_failed = false;
    }

    fn release(&mut self, cache_id: &str) {
        let mut ctx = B::new_context();
        B::sync(&mut ctx);
        self.graph_warmup = 0;
        self.graph_capture_failed = false;
        self.lora_cache_adapters.remove(cache_id);
        if let Some(mut caches) = self.kv_caches.remove(cache_id) {
            if let Some(alloc_arc) = self.paged_block_alloc.as_ref() {
                let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(c0) = caches.first() {
                    if !c0.paged_block_indices.is_empty() {
                        alloc.free(&c0.paged_block_indices);
                    }
                }
                for c in caches.iter_mut() {
                    c.paged_block_indices.clear();
                }
            }
            self.kv_free_pool.push(caches);
        }
    }

    fn reset(&mut self) {
        let mut ctx = B::new_context();
        B::sync(&mut ctx);
        B::reset_all_graphs(&mut ctx);
        B::sync(&mut ctx);
        self.graph_warmup = 0;
        self.graph_capture_failed = false;
        self.batched_graph_keys_seen.clear();
        self.batched_graph_warmup = 0;
        self.batched_graph_failed = false;
        self.kv_caches.clear();
        self.kv_free_pool.clear();
        self.lora_cache_adapters.clear();
    }
}

// INT8 DecoderOnlyLLM impl — minimal: no batched / unified-forward overrides
// (default trait impl falls back to per-item decode). PR D will add INT8
// batched paths once the kernels stabilize.
impl<B: MoeLlmBackend + BackendInt8KvOps> DecoderOnlyLLM for LlamaFamilyModel<B, KvInt8> {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.runtime_cfg
    }

    fn cache_metrics_snapshot(&self) -> Option<serde_json::Value> {
        Some(self.prefix_cache_snapshot_json())
    }

    fn lora_metrics_snapshot(&self) -> Option<serde_json::Value> {
        Some(self.lora_metrics_snapshot_json())
    }

    fn set_lora_adapter_for_cache(
        &mut self,
        cache_id: &str,
        adapter: Option<ActiveLoraAdapter>,
    ) -> std::result::Result<(), ferrum_types::FerrumError> {
        if let Some(adapter) = adapter {
            self.ensure_lora_adapter_loaded(adapter.clone())?;
            self.lora_cache_adapters
                .insert(cache_id.to_string(), adapter.name);
        } else {
            self.lora_cache_adapters.remove(cache_id);
        }
        Ok(())
    }

    fn prepare(&mut self, cache_id: &str, max_tokens: usize) {
        self.ensure_scratch(max_tokens);
        self.ensure_kv(cache_id);

        const WARMUP_CACHE: &str = "__ferrum_warmup__";
        let _ = self.prefill_internal(WARMUP_CACHE, &[0u32]);
        if let Some(mut caches) = self.kv_caches.remove(WARMUP_CACHE) {
            if let Some(alloc_arc) = self.paged_block_alloc.as_ref() {
                let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(c0) = caches.first() {
                    if !c0.paged_block_indices.is_empty() {
                        alloc.free(&c0.paged_block_indices);
                    }
                }
                for c in caches.iter_mut() {
                    c.paged_block_indices.clear();
                }
            }
            self.kv_free_pool.push(caches);
        }
    }

    fn prepare_kv_capacity(&mut self, cache_id: &str, capacity_hint: usize) {
        self.ensure_kv_with_capacity_hint(cache_id, Some(capacity_hint));
    }

    fn kv_capacity(&self) -> usize {
        self.runtime_env.kv_capacity_for_model(self.cfg.max_seq_len)
    }

    fn prefill(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        self.prefill_internal(cache_id, tokens)
    }

    fn decode(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        self.decode_internal(cache_id, token, pos)
    }

    // decode_batch + unified_forward use trait defaults (per-item fallback).

    fn forward_verify(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        LlamaFamilyModel::<B, KvInt8>::forward_verify(self, cache_id, tokens)
    }

    fn truncate_kv(&mut self, cache_id: &str, new_len: usize) {
        if let Some(caches) = self.kv_caches.get_mut(cache_id) {
            for c in caches.iter_mut() {
                if new_len < c.len {
                    c.len = new_len;
                }
            }
        }
        let mut ctx = B::new_context();
        B::reset_graph(&mut ctx, SINGLE_ITEM_GRAPH_KEY);
        self.graph_warmup = 0;
        self.graph_capture_failed = false;
    }

    fn release(&mut self, cache_id: &str) {
        let mut ctx = B::new_context();
        B::sync(&mut ctx);
        self.graph_warmup = 0;
        self.graph_capture_failed = false;
        self.lora_cache_adapters.remove(cache_id);
        if let Some(mut caches) = self.kv_caches.remove(cache_id) {
            if let Some(alloc_arc) = self.paged_block_alloc.as_ref() {
                let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(c0) = caches.first() {
                    if !c0.paged_block_indices.is_empty() {
                        alloc.free(&c0.paged_block_indices);
                    }
                }
                for c in caches.iter_mut() {
                    c.paged_block_indices.clear();
                }
            }
            self.kv_free_pool.push(caches);
        }
    }

    fn reset(&mut self) {
        let mut ctx = B::new_context();
        B::sync(&mut ctx);
        B::reset_all_graphs(&mut ctx);
        B::sync(&mut ctx);
        self.graph_warmup = 0;
        self.graph_capture_failed = false;
        self.kv_caches.clear();
        self.kv_free_pool.clear();
        self.lora_cache_adapters.clear();
    }
}

fn build_rope_cache<B: QuantLlmBackend + BackendMoeFused>(cfg: &LlamaFamilyConfig) -> RopeCache<B> {
    let hd = cfg.head_dim;
    let half = hd / 2;
    let max = cfg.max_seq_len;
    let mut cos = vec![0.0f32; max * half];
    let mut sin = vec![0.0f32; max * half];
    for pos in 0..max {
        for i in 0..half {
            let freq = rope_freq(cfg, i);
            let angle = pos as f64 * freq;
            cos[pos * half + i] = angle.cos() as f32;
            sin[pos * half + i] = angle.sin() as f32;
        }
    }
    RopeCache {
        cos: B::from_slice(&cos),
        sin: B::from_slice(&sin),
    }
}

fn rope_freq(cfg: &LlamaFamilyConfig, pair_idx: usize) -> f64 {
    let base_freq = 1.0f64
        / cfg
            .rope_theta
            .powf((2 * pair_idx) as f64 / cfg.head_dim as f64);
    match &cfg.rope_scaling {
        Some(RopeScalingConfig::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
        }) => scale_llama3_rope_freq(
            base_freq,
            *factor,
            *low_freq_factor,
            *high_freq_factor,
            *original_max_position_embeddings,
        ),
        Some(RopeScalingConfig::Linear { factor }) => base_freq / factor,
        None => base_freq,
    }
}

fn scale_llama3_rope_freq(
    freq: f64,
    factor: f64,
    low_freq_factor: f64,
    high_freq_factor: f64,
    original_max_position_embeddings: f64,
) -> f64 {
    let wavelen = 2.0 * std::f64::consts::PI / freq;
    let low_freq_wavelen = original_max_position_embeddings / low_freq_factor;
    let high_freq_wavelen = original_max_position_embeddings / high_freq_factor;
    if wavelen < high_freq_wavelen {
        freq
    } else if wavelen > low_freq_wavelen {
        freq / factor
    } else {
        let smooth = (original_max_position_embeddings / wavelen - low_freq_factor)
            / (high_freq_factor - low_freq_factor);
        (1.0 - smooth) * freq / factor + smooth * freq
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use ferrum_kernels::backend::cpu::CpuBackend;
    use ferrum_quantization::{DenseLinear, QuantConfig, WeightLoader};
    use ferrum_types::{Activation, Result};

    use super::{
        bf16_round, llama_layer_attention_schedule, llama_qk_mode, load_llama_family_layers,
        nan_trace_raw_matches, paged_kv_allowed_for_layer_schedule, rope_freq, stats_for,
        supports_sandwich_legacy_batched_decode, unified_varlen_qkv_unsupported_reason,
        LlamaDecodeBatchStats, LlamaFamilyConfig, LlamaFamilyLayerStageConfig, LlamaFamilyModel,
        LlamaFamilyRuntimeEnv, LlamaFamilyScratch, LlamaNanTraceConfig, RopeScalingConfig,
        DEFAULT_KV_CAPACITY,
    };

    #[test]
    fn decode_batch_stats_snapshot_records_shape_and_fallbacks() {
        let mut stats = LlamaDecodeBatchStats::default();
        stats.record_call(1, false);
        stats.record_singleton_fast_path();
        stats.record_call(4, true);
        stats.record_unsupported_fallback();
        stats.record_call(16, false);
        stats.record_lora_fallback();

        let snapshot = stats.snapshot_json();
        assert_eq!(snapshot["schema_version"], 1);
        assert_eq!(snapshot["calls"], 3);
        assert_eq!(snapshot["total_items"], 21);
        assert_eq!(snapshot["max_items"], 16);
        assert_eq!(snapshot["last_items"], 16);
        assert_eq!(snapshot["force_full_logits_calls"], 1);
        assert_eq!(snapshot["singleton_fast_path_calls"], 1);
        assert_eq!(snapshot["unsupported_fallback_calls"], 1);
        assert_eq!(snapshot["lora_fallback_calls"], 1);
        assert_eq!(snapshot["bucket_calls"]["m1"], 1);
        assert_eq!(snapshot["bucket_calls"]["m3_4"], 1);
        assert_eq!(snapshot["bucket_calls"]["m9_16"], 1);
    }

    #[test]
    fn unsupported_rope_scaling_clamps_max_seq_len() {
        // DeepSeek-R1-0528-Qwen3-8B: YaRN factor 4 over a 32k training
        // window. Ferrum has no YaRN, so the claimed 131072 window must
        // clamp to the original 32768.
        let mut def = crate::definition::ModelDefinition::default();
        def.max_position_embeddings = 131072;
        def.extra_params = serde_json::json!({
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768
            }
        });
        let cfg = LlamaFamilyConfig::qwen3_from_def(&def);
        assert_eq!(cfg.max_seq_len, 32768);
        assert!(cfg.rope_scaling.is_none());
    }

    #[test]
    fn llama3_rope_scaling_keeps_configured_window() {
        let mut def = crate::definition::ModelDefinition::default();
        def.max_position_embeddings = 131072;
        def.extra_params = serde_json::json!({
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192
            }
        });
        let cfg = LlamaFamilyConfig::llama_from_def(&def);
        assert_eq!(cfg.max_seq_len, 131072);
        assert!(matches!(
            cfg.rope_scaling,
            Some(RopeScalingConfig::Llama3 { .. })
        ));
    }

    #[derive(Default)]
    struct RecordingLoader {
        tensors: Mutex<Vec<String>>,
        linears: Mutex<Vec<String>>,
    }

    impl WeightLoader<CpuBackend> for RecordingLoader {
        fn load_tensor(&self, name: &str) -> Result<Vec<f32>> {
            self.tensors.lock().unwrap().push(name.to_string());
            Ok(vec![1.0])
        }

        fn load_linear(
            &self,
            name: &str,
        ) -> Result<Box<dyn ferrum_quantization::Linear<CpuBackend>>> {
            self.linears.lock().unwrap().push(name.to_string());
            Ok(Box::new(DenseLinear::<CpuBackend>::from_rows(&[1.0], 1, 1)))
        }

        fn has_tensor(&self, _name: &str) -> bool {
            false
        }

        fn quant_config(&self) -> Option<&QuantConfig> {
            None
        }
    }

    fn test_llama_config(num_layers: usize, has_qk_norm: bool) -> LlamaFamilyConfig {
        LlamaFamilyConfig {
            hidden_size: 1,
            intermediate_size: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            num_layers,
            vocab_size: 1,
            max_seq_len: 1,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling: None,
            rope_interleaved: false,
            has_qk_norm,
            sliding_window: 0,
            ..Default::default()
        }
    }

    #[test]
    fn sandwich_legacy_batched_decode_requires_device_shadow_and_layer_schedule() {
        let mut cfg = test_llama_config(1, true);
        cfg.sandwich_norms = true;
        cfg.sliding_window = 512;
        cfg.sliding_window_pattern = 6;

        assert!(supports_sandwich_legacy_batched_decode(&cfg, true));
        assert!(!supports_sandwich_legacy_batched_decode(&cfg, false));

        cfg.sliding_window_pattern = 0;
        assert!(!supports_sandwich_legacy_batched_decode(&cfg, true));

        cfg.sandwich_norms = false;
        cfg.sliding_window_pattern = 6;
        assert!(!supports_sandwich_legacy_batched_decode(&cfg, true));
    }

    #[test]
    fn llama_attention_semantics_cover_qk_mode_and_layer_windows() {
        let mut cfg = test_llama_config(12, false);
        assert_eq!(llama_qk_mode(&cfg), 2);

        cfg.rope_interleaved = true;
        assert_eq!(llama_qk_mode(&cfg), 3);

        cfg.has_qk_norm = true;
        assert_eq!(llama_qk_mode(&cfg), 1);

        cfg.sliding_window = 512;
        cfg.sliding_window_pattern = 0;
        let uniform = llama_layer_attention_schedule(&cfg, 0);
        assert!(uniform.is_global_layer);
        assert_eq!(uniform.sliding_window, 512);

        cfg.sliding_window_pattern = 6;
        let local = llama_layer_attention_schedule(&cfg, 4);
        assert!(!local.is_global_layer);
        assert_eq!(local.sliding_window, 512);

        let global = llama_layer_attention_schedule(&cfg, 5);
        assert!(global.is_global_layer);
        assert_eq!(global.sliding_window, 0);
    }

    #[test]
    fn unified_varlen_qkv_requires_gemma_sandwich_prerequisites() {
        let mut cfg = test_llama_config(1, true);
        cfg.sandwich_norms = true;
        cfg.sliding_window = 512;
        cfg.sliding_window_pattern = 6;
        cfg.activation = Activation::GeluTanh;

        let reason = unified_varlen_qkv_unsupported_reason(&cfg, true, false)
            .expect("sandwich configs without device F32 shadow must be rejected");

        assert!(reason.contains("sandwich-norm family"));
        assert!(reason.contains("device-side F32 residual shadow"));
        assert!(reason.contains("post-attn/post-ffn sandwich norms"));

        cfg.sandwich_norms = false;
        assert!(unified_varlen_qkv_unsupported_reason(&cfg, true, false).is_none());
        assert!(unified_varlen_qkv_unsupported_reason(&cfg, false, false)
            .expect("backend without varlen support should be rejected")
            .contains("backend lacks varlen"));

        cfg.sandwich_norms = true;
        cfg.sliding_window_pattern = 0;
        assert!(unified_varlen_qkv_unsupported_reason(&cfg, true, true)
            .expect("sandwich configs without layer schedule must be rejected")
            .contains("local/global sliding-window layer pattern"));

        cfg.sliding_window_pattern = 6;
        assert!(unified_varlen_qkv_unsupported_reason(&cfg, true, true).is_none());
    }

    #[test]
    fn paged_kv_layer_schedule_allows_windowed_models_with_varlen_backend() {
        assert!(paged_kv_allowed_for_layer_schedule(true, false, 0));
        assert!(paged_kv_allowed_for_layer_schedule(true, true, 0));
        assert!(!paged_kv_allowed_for_layer_schedule(true, false, 6));
        assert!(paged_kv_allowed_for_layer_schedule(true, true, 6));
        assert!(!paged_kv_allowed_for_layer_schedule(false, true, 6));
    }

    #[test]
    fn llama_family_layer_loader_uses_source_layer_range() {
        let cfg = test_llama_config(5, true);
        let loader = RecordingLoader::default();

        let layers = load_llama_family_layers(&cfg, &loader, 2..4).unwrap();

        assert_eq!(layers.len(), 2);
        assert!(layers.iter().all(|layer| layer.q_norm_w.is_some()));
        assert!(layers.iter().all(|layer| layer.k_norm_w.is_some()));
        assert_eq!(
            loader.tensors.into_inner().unwrap(),
            vec![
                "model.layers.2.input_layernorm.weight",
                "model.layers.2.post_attention_layernorm.weight",
                "model.layers.2.self_attn.q_norm.weight",
                "model.layers.2.self_attn.k_norm.weight",
                "model.layers.3.input_layernorm.weight",
                "model.layers.3.post_attention_layernorm.weight",
                "model.layers.3.self_attn.q_norm.weight",
                "model.layers.3.self_attn.k_norm.weight",
            ]
        );
        assert_eq!(
            loader.linears.into_inner().unwrap(),
            vec![
                "model.layers.2.self_attn.qkv_proj",
                "model.layers.2.self_attn.o_proj",
                "model.layers.2.mlp.gate_up_proj",
                "model.layers.2.mlp.down_proj",
                "model.layers.3.self_attn.qkv_proj",
                "model.layers.3.self_attn.o_proj",
                "model.layers.3.mlp.gate_up_proj",
                "model.layers.3.mlp.down_proj",
            ]
        );
    }

    #[test]
    fn llama_family_layer_loader_rejects_out_of_bounds_range() {
        let cfg = test_llama_config(2, false);
        let loader = RecordingLoader::default();

        let err = match load_llama_family_layers(&cfg, &loader, 1..3) {
            Ok(_) => panic!("expected out-of-bounds layer range to fail"),
            Err(err) => err,
        };

        assert!(
            err.to_string().contains("outside model layer count"),
            "{err}"
        );
    }

    #[test]
    fn llama_family_full_model_records_source_layer_range() {
        let cfg = test_llama_config(3, false);
        let loader = RecordingLoader::default();
        let mut model = LlamaFamilyModel::<CpuBackend>::new(cfg, &loader).unwrap();

        assert_eq!(model.source_layer_range(), 0..3);
        assert_eq!(model.local_layer_count(), 3);
        assert_eq!(model.source_layer_index(2), 2);

        model.ensure_kv("test-cache");
        assert_eq!(
            model.kv_caches["test-cache"].len(),
            model.local_layer_count()
        );
    }

    #[test]
    fn contiguous_kv_capacity_hint_sizes_physical_cache() {
        let mut cfg = test_llama_config(2, false);
        cfg.max_seq_len = 64;
        cfg.sliding_window_pattern = 6;
        let loader = RecordingLoader::default();
        let mut model = LlamaFamilyModel::<CpuBackend>::new(cfg, &loader).unwrap();

        model.ensure_kv_with_capacity_hint("short", Some(17));
        assert_eq!(model.kv_caches["short"][0].capacity, 17);
        assert_eq!(model.kv_caches["short"][1].capacity, 17);

        crate::common::DecoderOnlyLLM::release(&mut model, "short");
        model.ensure_kv_with_capacity_hint("longer", Some(19));
        assert_eq!(
            model.kv_caches["longer"][0].capacity, 19,
            "free-pool reuse must not keep a larger or smaller hinted cache"
        );
    }

    #[test]
    fn llama_family_layer_stage_loads_only_requested_weights() {
        let cfg = test_llama_config(5, false);
        let loader = RecordingLoader::default();

        let model = LlamaFamilyModel::<CpuBackend>::new_layer_stage(
            cfg,
            &loader,
            LlamaFamilyLayerStageConfig::pipeline_stage(2..5, false, true),
        )
        .unwrap();

        assert_eq!(model.source_layer_range(), 2..5);
        assert_eq!(model.local_layer_count(), 3);
        assert!(model.embed.is_none());
        assert!(model.lm_head.is_some());
        assert_eq!(
            loader.tensors.into_inner().unwrap(),
            vec![
                "model.layers.2.input_layernorm.weight",
                "model.layers.2.post_attention_layernorm.weight",
                "model.layers.3.input_layernorm.weight",
                "model.layers.3.post_attention_layernorm.weight",
                "model.layers.4.input_layernorm.weight",
                "model.layers.4.post_attention_layernorm.weight",
                "model.norm.weight",
            ]
        );
        assert_eq!(
            loader.linears.into_inner().unwrap(),
            vec![
                "model.layers.2.self_attn.qkv_proj",
                "model.layers.2.self_attn.o_proj",
                "model.layers.2.mlp.gate_up_proj",
                "model.layers.2.mlp.down_proj",
                "model.layers.3.self_attn.qkv_proj",
                "model.layers.3.self_attn.o_proj",
                "model.layers.3.mlp.gate_up_proj",
                "model.layers.3.mlp.down_proj",
                "model.layers.4.self_attn.qkv_proj",
                "model.layers.4.self_attn.o_proj",
                "model.layers.4.mlp.gate_up_proj",
                "model.layers.4.mlp.down_proj",
                "model.embed_tokens",
            ]
        );
    }

    #[test]
    fn llama_family_layer_stage_runs_hidden_forward_bridge() {
        let cfg = test_llama_config(1, false);
        let loader = RecordingLoader::default();
        let mut model = LlamaFamilyModel::<CpuBackend>::new_layer_stage(
            cfg,
            &loader,
            LlamaFamilyLayerStageConfig::pipeline_stage(0..1, false, false),
        )
        .unwrap();

        let hidden = model.prefill_stage_hidden_from_host("stage-cache", &[1.0], 1, 0);

        assert_eq!(hidden.len(), 1);
        assert!(hidden[0].is_finite());
        assert_eq!(
            model.kv_caches["stage-cache"].len(),
            model.local_layer_count()
        );
    }

    #[test]
    fn llama_family_last_stage_projects_hidden_to_logits() {
        let cfg = test_llama_config(1, false);
        let loader = RecordingLoader::default();
        let mut model = LlamaFamilyModel::<CpuBackend>::new_layer_stage(
            cfg,
            &loader,
            LlamaFamilyLayerStageConfig::pipeline_stage(0..1, false, true),
        )
        .unwrap();

        let logits = model.logits_from_hidden(&[2.0]);

        assert_eq!(logits.len(), 1);
        assert!(logits[0].is_finite());
    }

    #[test]
    fn llama_family_runtime_env_parses_startup_knobs() {
        let env = LlamaFamilyRuntimeEnv::from_env_vars([
            ("FERRUM_KV_CAPACITY", "4096"),
            ("FERRUM_METAL_PAGED_KV", "0"),
            ("FERRUM_PAGED_MAX_SEQS", "64"),
            ("FERRUM_DECODE_OP_PROFILE", "0"),
            ("FERRUM_PREFILL_OP_PROFILE", ""),
            ("FERRUM_CUDA_GRAPH", ""),
            ("FERRUM_DECODE_LAYER_PROFILE", "false"),
            ("FERRUM_NAN_TRACE", "layer3, source9"),
            ("FERRUM_OP_DUMP", "/tmp/ferrum-op-dump"),
        ]);

        assert_eq!(env.kv_capacity, Some(4096));
        assert_eq!(env.metal_paged_kv, Some(false));
        assert_eq!(env.paged_max_seqs, 64);
        assert!(env.decode_op_profile);
        assert!(env.prefill_op_profile);
        assert!(env.cuda_graph);
        assert!(env.decode_layer_profile);
        assert!(env.nan_trace.matches_layer(3, 9));
        assert_eq!(env.op_dump_dir.as_deref(), Some("/tmp/ferrum-op-dump"));
        assert!(!env.graph_capture_allowed());
        assert_eq!(env.kv_capacity_for_model(2048), 2048);
    }

    #[test]
    fn llama_family_runtime_env_uses_defaults_for_invalid_values() {
        let env = LlamaFamilyRuntimeEnv::from_env_vars([
            ("FERRUM_KV_CAPACITY", "bad"),
            ("FERRUM_PAGED_MAX_SEQS", "bad"),
            ("FERRUM_METAL_PAGED_KV", "1"),
        ]);

        assert_eq!(env.kv_capacity, None);
        assert_eq!(env.metal_paged_kv, Some(true));
        assert_eq!(env.paged_max_seqs, 32);
        assert_eq!(
            env.kv_capacity_for_model(DEFAULT_KV_CAPACITY * 2),
            DEFAULT_KV_CAPACITY
        );
        assert_eq!(env.nan_trace, LlamaNanTraceConfig::Disabled);
        assert_eq!(env.op_dump_dir, None);
        assert!(env.graph_capture_allowed());
    }

    #[test]
    fn dump_stats_counts_nonfinite_values() {
        let stats = stats_for(&[1.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -2.0]);

        assert_eq!(stats.n, 5);
        assert_eq!(stats.finite, 2);
        assert_eq!(stats.nan, 1);
        assert_eq!(stats.posinf, 1);
        assert_eq!(stats.neginf, 1);
        assert_eq!(stats.maxabs, 2.0);
        assert_eq!(stats.maxabs_index, Some(4));
    }

    #[test]
    fn prefill_scratch_keeps_batch_logits_lazy() {
        let cfg = test_llama_config(1, false);
        let mut scratch = LlamaFamilyScratch::<CpuBackend>::alloc(&cfg, 128);

        assert_eq!(scratch.max_tokens, 128);
        assert_eq!(scratch.batch_logits_capacity, 1);
        assert_eq!(scratch.batch_logits.len(), cfg.vocab_size);

        scratch.ensure_batch_logits(&cfg, 128);
        assert_eq!(scratch.batch_logits_capacity, 128);
        assert_eq!(scratch.batch_logits.len(), 128 * cfg.vocab_size);
    }

    #[test]
    fn nan_trace_selector_accepts_all_or_layer_lists() {
        assert!(nan_trace_raw_matches("all", 3, 9));
        assert!(nan_trace_raw_matches("1", 3, 9));
        assert!(nan_trace_raw_matches("3", 3, 9));
        assert!(nan_trace_raw_matches("9", 3, 9));
        assert!(nan_trace_raw_matches("2, 9, 12", 3, 9));
        assert!(!nan_trace_raw_matches("0", 3, 9));
        assert!(!nan_trace_raw_matches("false", 3, 9));
        assert!(!nan_trace_raw_matches("2,4,6", 3, 9));
        assert!(nan_trace_raw_matches("layer0", 0, 0));
        assert!(nan_trace_raw_matches("l0", 0, 0));
        assert!(nan_trace_raw_matches("source0", 1, 0));
        assert!(nan_trace_raw_matches("layer0, layer3", 3, 9));
    }

    #[test]
    fn bf16_round_matches_known_values() {
        // sqrt(1152) = 33.941...; bf16 mantissa keeps 8 bits.
        let r = bf16_round((1152f64).sqrt() as f32);
        let bits = r.to_bits();
        assert_eq!(bits & 0xFFFF, 0, "low mantissa bits must be cleared");
        assert!((r - 33.941125).abs() < 0.25, "rounded value far off: {r}");
        // Exact powers of two are fixed points.
        assert_eq!(bf16_round(32.0), 32.0);
    }

    #[test]
    fn gemma3_from_def_parses_real_1b_config() {
        let config = serde_json::json!({
            "architectures": ["Gemma3ForCausalLM"],
            "model_type": "gemma3_text",
            "hidden_size": 1152,
            "intermediate_size": 6912,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "num_hidden_layers": 26,
            "head_dim": 256,
            "vocab_size": 262144,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "sliding_window": 512,
            "sliding_window_pattern": 6,
            "query_pre_attn_scalar": 256,
            "hidden_activation": "gelu_pytorch_tanh"
        });
        let mut mgr = crate::definition::ConfigManager::new();
        let def = mgr.parse_config_for_tests(&config).unwrap();
        assert_eq!(def.architecture, crate::registry::Architecture::Gemma3);
        assert_eq!(def.activation, Activation::GeluTanh);
        let cfg = LlamaFamilyConfig::gemma3_from_def(&def);
        assert_eq!(cfg.hidden_size, 1152);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.sliding_window, 512);
        assert_eq!(cfg.sliding_window_pattern, 6);
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert_eq!(cfg.rope_local_theta, Some(10_000.0));
        assert_eq!(cfg.query_pre_attn_scalar, Some(256.0));
        assert!(cfg.rmsnorm_unit_offset);
        assert!(cfg.sandwich_norms);
        assert!(cfg.has_qk_norm);
        let scale = cfg.embed_scale.unwrap();
        assert!((scale - 33.941125f32).abs() < 0.25);
        // 5:1 pattern: layers 5, 11, 17, 23 are global.
        for li in 0..26usize {
            let global = (li + 1) % 6 == 0;
            assert_eq!(global, matches!(li, 5 | 11 | 17 | 23), "layer {li}");
        }
    }

    #[test]
    fn gemma3_nested_text_config_flattens() {
        let config = serde_json::json!({
            "architectures": ["Gemma3ForConditionalGeneration"],
            "model_type": "gemma3",
            "eos_token_id": [1, 106],
            "text_config": {
                "hidden_size": 5376,
                "intermediate_size": 21504,
                "num_attention_heads": 32,
                "num_key_value_heads": 16,
                "num_hidden_layers": 62,
                "head_dim": 128,
                "vocab_size": 262208,
                "max_position_embeddings": 131072,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "rope_local_base_freq": 10000.0,
                "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
                "sliding_window": 1024,
                "sliding_window_pattern": 6,
                "query_pre_attn_scalar": 168,
                "hidden_activation": "gelu_pytorch_tanh"
            },
            "vision_config": {"hidden_size": 1152}
        });
        let mut mgr = crate::definition::ConfigManager::new();
        let def = mgr.parse_config_for_tests(&config).unwrap();
        let cfg = LlamaFamilyConfig::gemma3_from_def(&def);
        assert_eq!(cfg.hidden_size, 5376);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.num_layers, 62);
        assert_eq!(cfg.sliding_window, 1024);
        assert_eq!(cfg.query_pre_attn_scalar, Some(168.0));
        assert_eq!(
            cfg.rope_scaling,
            Some(RopeScalingConfig::Linear { factor: 8.0 })
        );
        assert_eq!(cfg.max_seq_len, 131072);
        // q_norm fold factor sqrt(128/168) ~= 0.8729
        let fold = (cfg.head_dim as f64 / cfg.query_pre_attn_scalar.unwrap()).sqrt();
        assert!((fold - 0.872871).abs() < 1e-5);
    }

    #[test]
    fn linear_rope_scaling_divides_frequency() {
        let mut cfg = LlamaFamilyConfig::default();
        cfg.head_dim = 4;
        cfg.rope_theta = 10_000.0;
        cfg.rope_scaling = Some(RopeScalingConfig::Linear { factor: 8.0 });
        let scaled = rope_freq(&cfg, 0);
        cfg.rope_scaling = None;
        let unscaled = rope_freq(&cfg, 0);
        assert!((scaled - unscaled / 8.0).abs() < 1e-12);
    }

    #[test]
    fn cpu_gelu_tanh_mul_split_matches_reference() {
        use ferrum_kernels::backend::cpu::CpuBackend;
        use ferrum_kernels::backend::Backend;
        let mut ctx = <CpuBackend as Backend>::new_context();
        let im = 3usize;
        // one token: gate = [-1, 0.5, 2], up = [1, 2, 3]
        let gate_up = vec![-1.0f32, 0.5, 2.0, 1.0, 2.0, 3.0];
        let gu = <CpuBackend as Backend>::from_slice(&gate_up);
        let mut out = <CpuBackend as Backend>::alloc(im);
        <CpuBackend as Backend>::fused_gelu_tanh_mul_split(&mut ctx, &gu, &mut out, 1, im);
        let v = <CpuBackend as Backend>::to_vec(&out, im);
        let gelu = |x: f32| 0.5 * x * (1.0 + (0.79788456f32 * (x + 0.044715 * x * x * x)).tanh());
        for i in 0..im {
            let expect = gelu(gate_up[i]) * gate_up[im + i];
            assert!((v[i] - expect).abs() < 1e-6, "i={i}: {} vs {expect}", v[i]);
        }
    }

    #[test]
    fn cpu_scale_inplace_scales() {
        use ferrum_kernels::backend::cpu::CpuBackend;
        use ferrum_kernels::backend::Backend;
        let mut ctx = <CpuBackend as Backend>::new_context();
        let mut buf = <CpuBackend as Backend>::from_slice(&[1.0f32, -2.0, 0.5]);
        <CpuBackend as Backend>::scale_inplace(&mut ctx, &mut buf, 33.9375, 3);
        let v = <CpuBackend as Backend>::to_vec(&buf, 3);
        assert_eq!(v, vec![33.9375, -67.875, 16.96875]);
    }
}
