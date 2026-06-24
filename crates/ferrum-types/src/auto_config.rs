//! Startup auto-configuration and selector decision trace types.
//!
//! This is the typed control-plane surface for gradually replacing M3 shell
//! env bundles with validated model/hardware/workload driven selections.

use crate::{
    parse_bool_env_value, parse_usize_env_value, RuntimeConfigEffect, RuntimeConfigEntry,
    RuntimeConfigSnapshot, RuntimeConfigSource,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

pub const M3_QWEN3_30B_A3B_INT4_PRESET: &str = "m3_qwen3_30b_a3b_int4";
pub const QWEN25_72B_GPTQ_INT4_2X4090_LAYER_SPLIT_PRESET: &str =
    "qwen25_72b_gptq_int4_2x4090_layer_split";
const DEFAULT_KV_BLOCK_SIZE_TOKENS: usize = 16;
const DEFAULT_KV_BLOCKS: usize = 2048;
const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub architecture: String,
    pub quantization: Option<String>,
    pub moe: Option<MoeCapabilities>,
    pub max_context_len: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub head_dim: Option<usize>,
    pub kv_heads: Option<usize>,
    pub estimated_weight_bytes: Option<u64>,
    pub recurrent_state_bytes_per_sequence: Option<u64>,
    pub supported_dtypes: Vec<String>,
    pub graph_safe_moe: bool,
}

impl ModelCapabilities {
    pub fn unknown() -> Self {
        Self {
            architecture: "unknown".to_string(),
            quantization: None,
            moe: None,
            max_context_len: None,
            num_hidden_layers: None,
            head_dim: None,
            kv_heads: None,
            estimated_weight_bytes: None,
            recurrent_state_bytes_per_sequence: None,
            supported_dtypes: vec!["fp16".to_string(), "fp32".to_string()],
            graph_safe_moe: false,
        }
    }

    pub fn qwen3_30b_a3b_gptq_int4() -> Self {
        Self {
            architecture: "qwen3_moe".to_string(),
            quantization: Some("gptq_int4".to_string()),
            moe: Some(MoeCapabilities {
                num_experts: 128,
                experts_per_token: 8,
                moe_intermediate_size: Some(768),
            }),
            max_context_len: Some(40960),
            num_hidden_layers: Some(48),
            head_dim: Some(128),
            kv_heads: Some(4),
            // Conservative GPTQ int4 weight footprint including quant scales
            // and loader/runtime overhead. This keeps the RTX 4090 M3 preset
            // at the historical 2048 KV blocks while still allowing smaller
            // GPUs to be downgraded before startup allocation.
            estimated_weight_bytes: Some(18 * GIB),
            recurrent_state_bytes_per_sequence: None,
            supported_dtypes: vec!["fp16".to_string()],
            graph_safe_moe: false,
        }
    }

    pub fn qwen25_72b_gptq_int4() -> Self {
        Self {
            architecture: "qwen2".to_string(),
            quantization: Some("gptq_int4".to_string()),
            moe: None,
            max_context_len: Some(32_768),
            num_hidden_layers: Some(80),
            head_dim: Some(128),
            kv_heads: Some(8),
            estimated_weight_bytes: Some(39 * GIB),
            recurrent_state_bytes_per_sequence: None,
            supported_dtypes: vec!["fp16".to_string()],
            graph_safe_moe: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeCapabilities {
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub moe_intermediate_size: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    pub backend: String,
    pub cuda_runtime: Option<String>,
    pub compute_capability: Option<String>,
    pub vram_bytes: Option<u64>,
    pub sm_count: Option<u32>,
    pub supported_dtypes: Vec<String>,
    pub supported_kv_dtypes: Vec<String>,
    pub graph_support: bool,
    pub compiled_features: CompiledKernelFeatures,
}

impl HardwareCapabilities {
    pub fn unknown() -> Self {
        Self {
            backend: "unknown".to_string(),
            cuda_runtime: None,
            compute_capability: None,
            vram_bytes: None,
            sm_count: None,
            supported_dtypes: vec!["fp16".to_string(), "fp32".to_string()],
            supported_kv_dtypes: vec!["fp16".to_string()],
            graph_support: false,
            compiled_features: CompiledKernelFeatures::default(),
        }
    }

    pub fn rtx4090_cuda(features: CompiledKernelFeatures) -> Self {
        Self {
            backend: "cuda".to_string(),
            cuda_runtime: None,
            compute_capability: Some("8.9".to_string()),
            vram_bytes: Some(24 * 1024 * 1024 * 1024),
            sm_count: Some(128),
            supported_dtypes: vec!["fp16".to_string(), "fp32".to_string()],
            supported_kv_dtypes: vec!["fp16".to_string(), "int8".to_string()],
            graph_support: true,
            compiled_features: features,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledKernelFeatures {
    pub cuda: bool,
    pub vllm_paged_attn: bool,
    pub vllm_moe_marlin: bool,
    pub cuda_graph: bool,
    pub greedy_argmax: bool,
    pub fa2_source: bool,
    pub fa2_direct_ffi: bool,
}

impl Default for CompiledKernelFeatures {
    fn default() -> Self {
        Self {
            cuda: false,
            vllm_paged_attn: false,
            vllm_moe_marlin: false,
            cuda_graph: false,
            greedy_argmax: false,
            fa2_source: false,
            fa2_direct_ffi: false,
        }
    }
}

impl CompiledKernelFeatures {
    pub fn m3_fast_path_without_fa2() -> Self {
        Self {
            cuda: true,
            vllm_paged_attn: true,
            vllm_moe_marlin: true,
            cuda_graph: true,
            greedy_argmax: true,
            fa2_source: false,
            fa2_direct_ffi: false,
        }
    }

    pub fn m3_fast_path_with_source_fa2() -> Self {
        Self {
            fa2_source: true,
            ..Self::m3_fast_path_without_fa2()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadProfile {
    pub preset: Option<String>,
    pub serving_mode: String,
    pub target_concurrency: usize,
    pub prompt_length_class: String,
    pub output_length_class: String,
    pub priority: WorkloadPriority,
}

impl WorkloadProfile {
    pub fn serving_default() -> Self {
        Self {
            preset: None,
            serving_mode: "openai_chat".to_string(),
            target_concurrency: 1,
            prompt_length_class: "unknown".to_string(),
            output_length_class: "unknown".to_string(),
            priority: WorkloadPriority::Balanced,
        }
    }

    pub fn serving_default_for_hardware(hardware: &HardwareCapabilities) -> Self {
        let mut profile = Self::serving_default();
        if hardware.backend.eq_ignore_ascii_case("cuda")
            || hardware.backend.eq_ignore_ascii_case("metal")
        {
            profile.target_concurrency = hardware
                .vram_bytes
                .map(vram_default_max_sequences)
                .unwrap_or(4)
                .max(1);
        }
        profile
    }

    pub fn m3_qwen3_30b_a3b_int4() -> Self {
        Self {
            preset: Some(M3_QWEN3_30B_A3B_INT4_PRESET.to_string()),
            serving_mode: "bench_serve".to_string(),
            target_concurrency: 32,
            prompt_length_class: "random_256".to_string(),
            output_length_class: "random_128".to_string(),
            priority: WorkloadPriority::Throughput,
        }
    }

    pub fn qwen25_72b_gptq_int4_2x4090_layer_split() -> Self {
        Self {
            preset: Some(QWEN25_72B_GPTQ_INT4_2X4090_LAYER_SPLIT_PRESET.to_string()),
            serving_mode: "bench_serve".to_string(),
            target_concurrency: 16,
            prompt_length_class: "random_256".to_string(),
            output_length_class: "random_128".to_string(),
            priority: WorkloadPriority::Throughput,
        }
    }

    fn is_m3_preset(&self) -> bool {
        self.is_preset(M3_QWEN3_30B_A3B_INT4_PRESET)
    }

    fn is_preset(&self, preset: &str) -> bool {
        self.preset.as_deref() == Some(preset)
    }
}

impl Default for WorkloadProfile {
    fn default() -> Self {
        Self::serving_default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadPriority {
    Latency,
    Throughput,
    Balanced,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedFerrumConfig {
    pub schema_version: u32,
    pub preset: Option<String>,
    pub runtime_config: RuntimeConfigSnapshot,
    pub model_capabilities: ModelCapabilities,
    pub hardware_capabilities: HardwareCapabilities,
    pub workload_profile: WorkloadProfile,
    pub decisions: Vec<AutoConfigDecision>,
}

impl ResolvedFerrumConfig {
    pub fn effective_config_document(&self) -> serde_json::Value {
        let backend = self.hardware_capabilities.backend.clone();
        let requested_gpu_devices = self
            .runtime_csv_usize("FERRUM_REQUESTED_GPU_DEVICES")
            .or_else(|| default_gpu_devices_for_backend(&backend));
        let selected_gpu_devices = self
            .runtime_csv_usize("FERRUM_SELECTED_GPU_DEVICES")
            .or_else(|| requested_gpu_devices.clone())
            .or_else(|| default_gpu_devices_for_backend(&backend));
        let cuda_device_count = self
            .runtime_usize("FERRUM_CUDA_DEVICE_COUNT")
            .or_else(|| {
                backend.eq_ignore_ascii_case("cuda").then(|| {
                    selected_gpu_devices
                        .as_ref()
                        .map(|devices| devices.len())
                        .unwrap_or(1)
                })
            })
            .unwrap_or(0);
        let selected_distributed_strategy = self
            .runtime_entry_value("FERRUM_SELECTED_DISTRIBUTED_STRATEGY")
            .unwrap_or_else(|| {
                if selected_gpu_devices
                    .as_ref()
                    .map(|devices| devices.len() > 1)
                    .unwrap_or(false)
                {
                    "layer_split".to_string()
                } else if backend.eq_ignore_ascii_case("cuda") {
                    "single_gpu".to_string()
                } else {
                    "none".to_string()
                }
            });
        let selected_layer_split_plan =
            self.runtime_entry_value("FERRUM_SELECTED_LAYER_SPLIT_PLAN");
        let selected_layer_split_stages =
            self.runtime_json_value("FERRUM_SELECTED_LAYER_SPLIT_STAGES");
        let selected_layer_split_stage_count = selected_layer_split_stages
            .as_ref()
            .and_then(|value| value.as_array().map(|stages| stages.len()))
            .or_else(|| {
                selected_layer_split_plan
                    .as_ref()
                    .and_then(|_| selected_gpu_devices.as_ref().map(Vec::len))
            });
        let requested_pipeline_mode = self.runtime_entry_value("FERRUM_LAYER_SPLIT_PIPELINE_MODE");
        let selected_pipeline_mode = if selected_layer_split_plan.is_some() {
            requested_pipeline_mode.unwrap_or_else(|| {
                if selected_layer_split_stage_count == Some(2) {
                    "overlapped".to_string()
                } else {
                    "batch".to_string()
                }
            })
        } else {
            "sequential".to_string()
        };
        let selected_max_sequences = self.selected_usize("max_sequences");
        let selected_microbatch_size = if selected_layer_split_plan.is_some() {
            selected_max_sequences.map(|max_sequences| {
                if selected_pipeline_mode == "overlapped" {
                    max_sequences.div_ceil(2).max(1)
                } else {
                    max_sequences
                }
            })
        } else {
            Some(1)
        };
        let selected_stage_bridge = selected_layer_split_plan.as_ref().map(|_| "host");
        let selected_max_model_len = self.selected_usize("max_model_len");
        let selected_kv_capacity = self.runtime_usize("FERRUM_KV_CAPACITY");
        let selected_max_batched_tokens = self.selected_usize("max_batched_tokens");
        let selected_recurrent_state_max_slots = self.selected_recurrent_state_max_slots();
        let selected_admission_limit =
            effective_admission_limit(selected_max_sequences, selected_recurrent_state_max_slots);
        serde_json::json!({
            "schema_version": 1,
            "preset": self.preset,
            "env_hash": self.runtime_env_hash(),
            "backend": backend.clone(),
            "requested_gpu_devices": requested_gpu_devices.clone(),
            "selected_gpu_devices": selected_gpu_devices.clone(),
            "cuda_device_count": cuda_device_count,
            "selected_distributed_strategy": selected_distributed_strategy.clone(),
            "selected_layer_split_plan": selected_layer_split_plan.clone(),
            "selected_layer_split_stages": selected_layer_split_stages,
            "selected_pipeline_mode": selected_pipeline_mode,
            "selected_microbatch_size": selected_microbatch_size,
            "selected_stage_bridge": selected_stage_bridge,
            "selected_weight_placement": if selected_layer_split_plan.is_some() { "layer_split" } else { "single_device" },
            "selected_kv_layout": if backend.eq_ignore_ascii_case("cpu") { "contiguous" } else { "paged" },
            "selected_attention_impl": self.selected_string("attention_decode_backend"),
            "selected_graph_mode": self.selected_graph_mode(),
            "selected_max_sequences": selected_max_sequences,
            "selected_max_model_len": selected_max_model_len,
            "selected_kv_capacity": selected_kv_capacity,
            "selected_max_batched_tokens": selected_max_batched_tokens,
            "selected_recurrent_state_max_slots": selected_recurrent_state_max_slots,
            "selected_admission_limit": selected_admission_limit,
            "entries": self.runtime_config.entries,
            "model_capabilities": self.model_capabilities,
            "hardware_capabilities": self.hardware_capabilities,
            "workload_profile": self.workload_profile,
            "admission": self.admission_summary_document(),
            "decisions": self.decisions,
        })
    }

    pub fn admission_summary_document(&self) -> serde_json::Value {
        let max_sequences = self.selected_usize("max_sequences");
        let recurrent_state_max_slots = self.selected_recurrent_state_max_slots();
        let effective_max_concurrent =
            effective_admission_limit(max_sequences, recurrent_state_max_slots);
        let kv_blocks = self.selected_usize("kv_block_count");
        let max_batched_tokens = self.selected_usize("max_batched_tokens");
        let max_model_len = self.selected_usize("max_model_len");
        let kv_capacity_tokens =
            kv_blocks.map(|blocks| blocks.saturating_mul(DEFAULT_KV_BLOCK_SIZE_TOKENS));
        let kv_bytes_per_token = kv_cache_bytes_per_token_for_model(&self.model_capabilities);
        let recurrent_budget =
            recurrent_state_budget_for(&self.model_capabilities, &self.hardware_capabilities);
        let scheduler_policy = self
            .selected_string("scheduler_admission_policy")
            .unwrap_or_else(|| "unknown".to_string());
        serde_json::json!({
            "schema_version": 1,
            "backend": self.hardware_capabilities.backend,
            "model_architecture": self.model_capabilities.architecture,
            "scheduler_policy": scheduler_policy,
            "effective_max_concurrent": effective_max_concurrent,
            "queue_depth": 0u64,
            "active_prefill": 0u64,
            "active_decode": 0u64,
            "current_batch_size": 0u64,
            "rejected_requests_total": 0u64,
            "failed_requests_total": 0u64,
            "completed_requests_total": 0u64,
            "max_sequences": max_sequences,
            "recurrent_state_max_slots": recurrent_state_max_slots,
            "kv_block_count": kv_blocks,
            "kv_block_size_tokens": DEFAULT_KV_BLOCK_SIZE_TOKENS,
            "kv_capacity_tokens": kv_capacity_tokens,
            "max_model_length": max_model_len,
            "max_batched_tokens": max_batched_tokens,
            "memory_estimate": {
                "vram_bytes": self.hardware_capabilities.vram_bytes,
                "estimated_weight_bytes": self.model_capabilities.estimated_weight_bytes,
                "kv_bytes_per_token": kv_bytes_per_token,
                "recurrent_state_bytes_per_sequence": self.model_capabilities.recurrent_state_bytes_per_sequence,
                "recurrent_state_budget_bytes": recurrent_budget.map(|budget| budget.remaining_bytes),
                "recurrent_state_budget_raw_slots": recurrent_budget.map(|budget| budget.raw_slots),
                "recurrent_state_budget_max_slots": recurrent_budget.map(|budget| budget.floored_slots),
                "recurrent_state_capacity_bytes": match (recurrent_state_max_slots, self.model_capabilities.recurrent_state_bytes_per_sequence) {
                    (Some(slots), Some(bytes_per_sequence)) => {
                        (slots as u64).checked_mul(bytes_per_sequence)
                    }
                    _ => None,
                },
                "kv_capacity_bytes": match (kv_capacity_tokens, kv_bytes_per_token) {
                    (Some(tokens), Some(bytes_per_token)) => {
                        (tokens as u64).checked_mul(bytes_per_token)
                    }
                    _ => None,
                },
            },
        })
    }

    pub fn decision_trace_jsonl(&self) -> Result<String, serde_json::Error> {
        let mut out = String::new();
        for decision in &self.decisions {
            out.push_str(&serde_json::to_string(decision)?);
            out.push('\n');
        }
        Ok(out)
    }

    pub fn runtime_env_hash(&self) -> String {
        use sha2::{Digest, Sha256};

        let bytes = serde_json::to_vec(&self.runtime_config.entries).unwrap_or_default();
        let digest = Sha256::digest(bytes);
        format!("sha256:{digest:x}")
    }

    fn selected_usize(&self, selection: &str) -> Option<usize> {
        self.selected_string(selection)?.parse().ok()
    }

    fn selected_string(&self, selection: &str) -> Option<String> {
        self.decisions
            .iter()
            .find(|decision| decision.selection == selection)
            .map(|decision| decision.selected.clone())
    }

    fn selected_recurrent_state_max_slots(&self) -> Option<usize> {
        self.selected_usize("recurrent_state_max_slots")
            .or_else(|| self.runtime_usize("FERRUM_RECURRENT_STATE_MAX_SLOTS"))
            .or_else(|| self.runtime_usize("FERRUM_QWEN35_LINEAR_STATE_MAX_SLOTS"))
    }

    fn selected_graph_mode(&self) -> Option<String> {
        let decode_graph = self.selected_string("decode_graph_policy");
        if decode_graph
            .as_deref()
            .is_some_and(|mode| mode != "graph_disabled")
        {
            return decode_graph;
        }
        self.selected_string("moe_graph_policy")
    }

    fn runtime_entry_value(&self, key: &str) -> Option<String> {
        self.runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| entry.effective_value.clone())
    }

    fn runtime_usize(&self, key: &str) -> Option<usize> {
        self.runtime_entry_value(key)?.parse().ok()
    }

    fn runtime_csv_usize(&self, key: &str) -> Option<Vec<usize>> {
        let raw = self.runtime_entry_value(key)?;
        let mut out = Vec::new();
        for part in raw.split(',') {
            let value = part.trim();
            if value.is_empty() {
                return None;
            }
            out.push(value.parse().ok()?);
        }
        Some(out)
    }

    fn runtime_json_value(&self, key: &str) -> Option<serde_json::Value> {
        serde_json::from_str(&self.runtime_entry_value(key)?).ok()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoConfigDecision {
    pub schema_version: u32,
    pub selection: String,
    pub selected: String,
    pub source: AutoConfigSource,
    pub source_key: Option<String>,
    pub candidates: Vec<String>,
    pub rejected: Vec<RejectedCandidate>,
    pub affects: Vec<RuntimeConfigEffect>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedCandidate {
    pub value: String,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoConfigSource {
    Default,
    Cli,
    ConfigFile,
    Env,
    ScriptCase,
    ModelMetadata,
    HardwareCapability,
    MemoryProfile,
    WorkloadPreset,
    CompiledFeature,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AutoConfigError {
    #[error("{key}: invalid override: {reason}")]
    InvalidOverride { key: String, reason: String },
    #[error("{selection}: unsupported combination: {reason}")]
    UnsupportedCombination { selection: String, reason: String },
}

pub struct FerrumConfigBuilder {
    runtime_config: RuntimeConfigSnapshot,
    model: ModelCapabilities,
    hardware: HardwareCapabilities,
    workload: WorkloadProfile,
}

impl FerrumConfigBuilder {
    pub fn new(runtime_config: RuntimeConfigSnapshot) -> Self {
        Self {
            runtime_config,
            model: ModelCapabilities::unknown(),
            hardware: HardwareCapabilities::unknown(),
            workload: WorkloadProfile::default(),
        }
    }

    pub fn m3_qwen3_30b_a3b_int4(runtime_config: RuntimeConfigSnapshot) -> Self {
        Self::new(runtime_config)
            .with_model_capabilities(ModelCapabilities::qwen3_30b_a3b_gptq_int4())
            .with_hardware_capabilities(HardwareCapabilities::rtx4090_cuda(
                CompiledKernelFeatures::m3_fast_path_without_fa2(),
            ))
            .with_workload_profile(WorkloadProfile::m3_qwen3_30b_a3b_int4())
    }

    pub fn with_model_capabilities(mut self, model: ModelCapabilities) -> Self {
        self.model = model;
        self
    }

    pub fn with_hardware_capabilities(mut self, hardware: HardwareCapabilities) -> Self {
        self.hardware = hardware;
        self
    }

    pub fn with_workload_profile(mut self, workload: WorkloadProfile) -> Self {
        self.workload = workload;
        self
    }

    pub fn resolve(self) -> Result<ResolvedFerrumConfig, AutoConfigError> {
        let mut decisions = Vec::new();
        let cuda_backend = self.is_cuda_backend();
        // Any CUDA GPTQ/INT4 MoE model gets the vLLM-Marlin fast MoE path when
        // the kernel is compiled — not only the m3 bench preset. `ferrum run`
        // resolves with the serving-default workload (not the m3 preset), so
        // without this it silently fell back to the slow host-route MoE
        // (~9.7 vs ~59 tok/s on a 4090 for Qwen3-30B-A3B). Capability-gated,
        // never model-name-gated.
        let cuda_gptq_moe = cuda_backend
            && self.model.moe.is_some()
            && self.model.quantization.as_deref().is_some_and(|q| {
                let q = q.to_ascii_lowercase();
                q.contains("gptq") || q.contains("int4")
            });
        let cuda_qwen_moe = cuda_backend
            && self.model.moe.is_some()
            && qwen_moe_architecture_uses_vllm_paged_attn(&self.model.architecture);
        let use_vllm_paged_attn = self.bool_value(
            "FERRUM_USE_VLLM_PAGED_ATTN",
            (self.workload.is_m3_preset() || cuda_qwen_moe)
                && cuda_backend
                && self.hardware.compiled_features.vllm_paged_attn,
            AutoConfigSource::WorkloadPreset,
        )?;
        let fa_layout =
            self.bool_value("FERRUM_FA_LAYOUT_VARLEN", false, AutoConfigSource::Default)?;
        let fa2_source = self.bool_value("FERRUM_FA2_SOURCE", false, AutoConfigSource::Default)?;
        let shim_present = self.raw("FERRUM_FA2_DIRECT_FFI_SHIM").is_some();
        let fa2_direct_ffi = self.bool_value(
            "FERRUM_FA2_DIRECT_FFI",
            shim_present,
            if shim_present {
                AutoConfigSource::Env
            } else {
                AutoConfigSource::Default
            },
        )?;
        let vllm_v1_short = self.bool_value(
            "FERRUM_VLLM_PAGED_ATTN_V1_SHORT",
            use_vllm_paged_attn.value && self.model.head_dim.unwrap_or(128) <= 128,
            AutoConfigSource::Default,
        )?;
        let vllm_moe = self.bool_value(
            "FERRUM_VLLM_MOE",
            (cuda_gptq_moe || (self.workload.is_m3_preset() && cuda_backend))
                && self.hardware.compiled_features.vllm_moe_marlin,
            AutoConfigSource::WorkloadPreset,
        )?;
        let device_route = self.bool_value(
            "FERRUM_MOE_DEVICE_ROUTE",
            vllm_moe.value,
            AutoConfigSource::WorkloadPreset,
        )?;
        let pair_ids = self.bool_value(
            "FERRUM_VLLM_MOE_PAIR_IDS",
            vllm_moe.value,
            AutoConfigSource::WorkloadPreset,
        )?;
        let graph = self.bool_value("FERRUM_MOE_GRAPH", false, AutoConfigSource::WorkloadPreset)?;
        let batched_graph =
            self.bool_value("FERRUM_BATCHED_GRAPH", false, AutoConfigSource::Default)?;
        let unified_graph =
            self.bool_value("FERRUM_UNIFIED_GRAPH", false, AutoConfigSource::Default)?;
        let unified_graph_layers_only = self.bool_value(
            "FERRUM_UNIFIED_GRAPH_LAYERS_ONLY",
            false,
            AutoConfigSource::Default,
        )?;
        let unified_graph_lm_head_eager = self.bool_value(
            "FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER",
            false,
            AutoConfigSource::Default,
        )?;
        let greedy = self.bool_value(
            "FERRUM_GREEDY_ARGMAX",
            (cuda_backend || self.hardware.backend.eq_ignore_ascii_case("metal"))
                && self.hardware.compiled_features.greedy_argmax,
            AutoConfigSource::HardwareCapability,
        )?;
        let prefix_cache = self.bool_value(
            "FERRUM_PREFIX_CACHE",
            false,
            if self.workload.is_m3_preset() {
                AutoConfigSource::WorkloadPreset
            } else {
                AutoConfigSource::Default
            },
        )?;
        let default_max_sequences = self.default_max_sequences();
        let max_sequences = self.usize_value(
            "FERRUM_PAGED_MAX_SEQS",
            default_max_sequences.value,
            default_max_sequences.source,
        )?;
        let default_recurrent_state_max_slots =
            self.default_recurrent_state_max_slots(&max_sequences);
        let recurrent_state_max_slots = if default_recurrent_state_max_slots.is_some()
            || self.entry("FERRUM_RECURRENT_STATE_MAX_SLOTS").is_some()
            || self.entry("FERRUM_QWEN35_LINEAR_STATE_MAX_SLOTS").is_some()
        {
            let default = default_recurrent_state_max_slots
                .as_ref()
                .unwrap_or(&max_sequences);
            Some(self.usize_value_with_legacy_alias(
                "FERRUM_RECURRENT_STATE_MAX_SLOTS",
                "FERRUM_QWEN35_LINEAR_STATE_MAX_SLOTS",
                default.value,
                default.source,
            )?)
        } else {
            None
        };
        let default_kv_blocks = self.default_kv_blocks(&max_sequences);
        let kv_blocks = self.usize_value(
            "FERRUM_KV_MAX_BLOCKS",
            default_kv_blocks.value,
            default_kv_blocks.source,
        )?;
        let default_max_batched_tokens =
            self.default_max_batched_tokens(&max_sequences, &kv_blocks);
        let max_batched_tokens = self.usize_value(
            "FERRUM_MAX_BATCHED_TOKENS",
            default_max_batched_tokens.value,
            default_max_batched_tokens.source,
        )?;
        let max_model_len = self.optional_usize_value("FERRUM_MAX_MODEL_LEN")?;
        let default_prefill_first_until_active =
            self.default_prefill_first_until_active(&max_sequences);
        let default_prefill_step_chunk =
            self.default_prefill_step_chunk(&max_sequences, &max_batched_tokens);
        self.validate_attention(
            use_vllm_paged_attn.value,
            fa_layout.value,
            fa2_source.value,
            fa2_direct_ffi.value,
            shim_present,
            vllm_v1_short.value,
        )?;
        self.validate_moe(
            vllm_moe.value,
            device_route.value,
            pair_ids.value,
            graph.value,
        )?;
        self.validate_batched_graph(batched_graph.value)?;
        self.validate_unified_graph(
            unified_graph.value,
            unified_graph_layers_only.value,
            unified_graph_lm_head_eager.value,
        )?;
        self.validate_memory(
            kv_blocks.value,
            max_sequences.value,
            recurrent_state_max_slots.as_ref().map(|slots| slots.value),
            max_batched_tokens.value,
            max_model_len.as_ref().map(|value| value.value),
        )?;
        self.validate_dtypes()?;
        self.validate_layer_split_pipeline_mode()?;
        self.validate_sampling(greedy.value)?;

        decisions.push(self.attention_prefill_decision(
            use_vllm_paged_attn.clone(),
            fa_layout,
            fa2_source,
            fa2_direct_ffi,
        ));
        decisions.push(
            self.attention_decode_decision(use_vllm_paged_attn.clone(), vllm_v1_short.clone()),
        );
        // Materialize the auto-resolved fast-path MoE knobs into the effective
        // config BEFORE moe_decision consumes them, so they reach the model
        // (which reads FERRUM_*, not the decisions). Only auto-derived values —
        // user/env entries are already present. Without this, `ferrum run`'s
        // non-preset path resolved FERRUM_VLLM_MOE as a decision only and the
        // model never saw it (~9.7 vs ~59 tok/s on a 4090 for Qwen3-30B-A3B).
        let mut runtime_config = self.runtime_config.clone();
        for (key, resolved) in [
            ("FERRUM_USE_VLLM_PAGED_ATTN", &use_vllm_paged_attn),
            ("FERRUM_VLLM_PAGED_ATTN_V1_SHORT", &vllm_v1_short),
            ("FERRUM_VLLM_MOE", &vllm_moe),
            ("FERRUM_MOE_DEVICE_ROUTE", &device_route),
            ("FERRUM_VLLM_MOE_PAIR_IDS", &pair_ids),
            ("FERRUM_BATCHED_GRAPH", &batched_graph),
            ("FERRUM_UNIFIED_GRAPH", &unified_graph),
            (
                "FERRUM_UNIFIED_GRAPH_LAYERS_ONLY",
                &unified_graph_layers_only,
            ),
            (
                "FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER",
                &unified_graph_lm_head_eager,
            ),
            ("FERRUM_GREEDY_ARGMAX", &greedy),
        ] {
            if resolved.source != AutoConfigSource::Env {
                runtime_config.upsert(
                    key,
                    if resolved.value { "1" } else { "0" },
                    RuntimeConfigSource::MemoryProfile,
                );
            }
        }
        if let Some(until) = default_prefill_first_until_active.as_ref() {
            if self
                .entry("FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE")
                .is_none()
                && self.entry("FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE").is_none()
            {
                runtime_config.upsert(
                    "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE",
                    until.value.to_string(),
                    RuntimeConfigSource::Default,
                );
            }
        }
        if let Some(chunk) = default_prefill_step_chunk.as_ref() {
            if self.entry("FERRUM_SCHED_PREFILL_STEP_CHUNK").is_none()
                && self.entry("FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE").is_none()
            {
                runtime_config.upsert(
                    "FERRUM_SCHED_PREFILL_STEP_CHUNK",
                    chunk.value.to_string(),
                    RuntimeConfigSource::Default,
                );
            }
        }
        if let Some(slots) = recurrent_state_max_slots.as_ref() {
            if self.entry("FERRUM_RECURRENT_STATE_MAX_SLOTS").is_none()
                && self.entry("FERRUM_QWEN35_LINEAR_STATE_MAX_SLOTS").is_none()
            {
                runtime_config.upsert(
                    "FERRUM_RECURRENT_STATE_MAX_SLOTS",
                    slots.value.to_string(),
                    RuntimeConfigSource::MemoryProfile,
                );
            }
        }
        decisions.push(self.moe_decision(vllm_moe, device_route, pair_ids));
        decisions.push(self.graph_decision(graph));
        decisions.push(self.decode_graph_decision(
            batched_graph,
            unified_graph,
            unified_graph_layers_only,
            unified_graph_lm_head_eager,
        ));
        decisions.push(self.scalar_decision(
            "kv_block_count",
            kv_blocks,
            RuntimeConfigEffect::Memory,
        ));
        decisions.push(self.scalar_decision(
            "max_sequences",
            max_sequences,
            RuntimeConfigEffect::Memory,
        ));
        if let Some(slots) = recurrent_state_max_slots {
            decisions.push(self.scalar_decision(
                "recurrent_state_max_slots",
                slots,
                RuntimeConfigEffect::Memory,
            ));
        }
        decisions.push(self.scalar_decision(
            "max_batched_tokens",
            max_batched_tokens,
            RuntimeConfigEffect::Performance,
        ));
        if let Some(max_model_len) = max_model_len {
            decisions.push(self.scalar_decision(
                "max_model_len",
                max_model_len,
                RuntimeConfigEffect::Memory,
            ));
        }
        decisions.push(self.prefix_cache_decision(prefix_cache));
        decisions.push(self.scheduler_decision(
            default_prefill_first_until_active,
            default_prefill_step_chunk,
        )?);
        decisions.push(self.sampling_decision(greedy));

        Ok(ResolvedFerrumConfig {
            schema_version: 1,
            preset: self.workload.preset.clone(),
            runtime_config,
            model_capabilities: self.model.clone(),
            hardware_capabilities: self.hardware.clone(),
            workload_profile: self.workload.clone(),
            decisions,
        })
    }

    fn entries(&self) -> BTreeMap<&str, &str> {
        self.runtime_config
            .entries
            .iter()
            .map(|entry| (entry.key.as_str(), entry.effective_value.as_str()))
            .collect()
    }

    fn raw(&self, key: &str) -> Option<&str> {
        self.entry(key).map(|entry| entry.effective_value.as_str())
    }

    fn entry(&self, key: &str) -> Option<&RuntimeConfigEntry> {
        self.runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == key)
    }

    fn source_for_key(&self, key: &str, default_source: AutoConfigSource) -> AutoConfigSource {
        self.entry(key)
            .map(|entry| auto_config_source_from_runtime(entry.source))
            .unwrap_or(default_source)
    }

    fn is_cuda_backend(&self) -> bool {
        self.hardware.backend.eq_ignore_ascii_case("cuda")
    }

    fn is_accelerator_backend(&self) -> bool {
        self.is_cuda_backend() || self.hardware.backend.eq_ignore_ascii_case("metal")
    }

    fn cuda_compute_capability_at_least(&self, major: u32, minor: u32) -> Option<bool> {
        let (actual_major, actual_minor) =
            parse_compute_capability(self.hardware.compute_capability.as_deref()?)?;
        Some((actual_major, actual_minor) >= (major, minor))
    }

    fn default_max_sequences(&self) -> ResolvedValue<usize> {
        let target = self.workload.target_concurrency.max(1);
        let mut selected = target;
        if self.workload.is_m3_preset() {
            if let Some(sm_count) = self.hardware.sm_count {
                // The M3 throughput preset assumes a large GPU. On smaller
                // known GPUs, avoid auto-selecting a c32-sized admission
                // window before memory profiling has a chance to refine KV.
                selected = selected.min((sm_count as usize / 4).max(1));
            }
            if let Some(vram_bytes) = self.hardware.vram_bytes {
                selected = selected.min(vram_default_max_sequences(vram_bytes));
            }
        }
        ResolvedValue {
            value: selected.max(1),
            source: if selected < target {
                AutoConfigSource::HardwareCapability
            } else {
                AutoConfigSource::WorkloadPreset
            },
            source_key: None,
        }
    }

    fn default_max_batched_tokens(
        &self,
        max_sequences: &ResolvedValue<usize>,
        kv_blocks: &ResolvedValue<usize>,
    ) -> ResolvedValue<usize> {
        let kv_token_capacity = kv_blocks
            .value
            .saturating_mul(DEFAULT_KV_BLOCK_SIZE_TOKENS)
            .max(max_sequences.value.max(1));
        let target = if self
            .workload
            .is_preset(QWEN25_72B_GPTQ_INT4_2X4090_LAYER_SPLIT_PRESET)
        {
            1536
        } else {
            max_sequences.value.max(1).saturating_mul(64)
        };
        let value = target
            .min(kv_token_capacity)
            .max(max_sequences.value.max(1));
        ResolvedValue {
            value,
            source: if max_sequences.source == AutoConfigSource::HardwareCapability
                || kv_blocks.source == AutoConfigSource::HardwareCapability
            {
                AutoConfigSource::HardwareCapability
            } else {
                AutoConfigSource::WorkloadPreset
            },
            source_key: None,
        }
    }

    fn default_prefill_first_until_active(
        &self,
        max_sequences: &ResolvedValue<usize>,
    ) -> Option<ResolvedValue<usize>> {
        if max_sequences.value <= 1 || !self.is_accelerator_backend() {
            return None;
        }
        Some(ResolvedValue {
            value: max_sequences.value,
            source: AutoConfigSource::Default,
            source_key: None,
        })
    }

    fn default_prefill_step_chunk(
        &self,
        max_sequences: &ResolvedValue<usize>,
        max_batched_tokens: &ResolvedValue<usize>,
    ) -> Option<ResolvedValue<usize>> {
        if max_sequences.value <= 1 || !self.is_accelerator_backend() {
            return None;
        }
        Some(ResolvedValue {
            value: max_batched_tokens
                .value
                .div_ceil(max_sequences.value.max(1))
                .max(1),
            source: AutoConfigSource::Default,
            source_key: None,
        })
    }

    fn default_kv_blocks(&self, max_sequences: &ResolvedValue<usize>) -> ResolvedValue<usize> {
        let min_blocks = ceil_div(max_sequences.value.max(1), DEFAULT_KV_BLOCK_SIZE_TOKENS);
        if self
            .workload
            .is_preset(QWEN25_72B_GPTQ_INT4_2X4090_LAYER_SPLIT_PRESET)
        {
            return ResolvedValue {
                value: 1024.max(min_blocks),
                source: AutoConfigSource::WorkloadPreset,
                source_key: None,
            };
        }
        let target = DEFAULT_KV_BLOCKS.max(min_blocks);
        let selected = match (
            self.hardware.vram_bytes,
            self.model.estimated_weight_bytes,
            self.kv_cache_bytes_per_token(),
        ) {
            (Some(vram_bytes), Some(weight_bytes), Some(kv_bytes_per_token))
                if kv_bytes_per_token > 0 =>
            {
                let headroom = (vram_bytes / 10).max(2 * GIB);
                let available = vram_bytes.saturating_sub(weight_bytes.saturating_add(headroom));
                let kv_token_budget = (available / kv_bytes_per_token) as usize;
                let block_budget = kv_token_budget / DEFAULT_KV_BLOCK_SIZE_TOKENS;
                target.min(block_budget.max(min_blocks))
            }
            _ => target,
        };
        ResolvedValue {
            value: selected.max(1),
            source: if selected < target {
                AutoConfigSource::HardwareCapability
            } else {
                AutoConfigSource::WorkloadPreset
            },
            source_key: None,
        }
    }

    fn default_recurrent_state_max_slots(
        &self,
        max_sequences: &ResolvedValue<usize>,
    ) -> Option<ResolvedValue<usize>> {
        let limit = self.recurrent_state_budget_max_slots()?;
        let selected = max_sequences.value.min(limit.max(1));
        Some(ResolvedValue {
            value: selected.max(1),
            source: if selected < max_sequences.value {
                AutoConfigSource::MemoryProfile
            } else {
                max_sequences.source
            },
            source_key: None,
        })
    }

    fn recurrent_state_budget_max_slots(&self) -> Option<usize> {
        self.recurrent_state_budget()
            .map(|budget| budget.floored_slots)
    }

    fn recurrent_state_budget(&self) -> Option<RecurrentStateBudget> {
        recurrent_state_budget_for(&self.model, &self.hardware)
    }

    fn kv_cache_bytes_per_token(&self) -> Option<u64> {
        kv_cache_bytes_per_token_for_model(&self.model)
    }

    fn bool_value(
        &self,
        key: &str,
        default: bool,
        default_source: AutoConfigSource,
    ) -> Result<ResolvedValue<bool>, AutoConfigError> {
        match self.entry(key) {
            Some(entry) => Ok(ResolvedValue {
                value: parse_bool_env_value(&entry.effective_value).map_err(|reason| {
                    AutoConfigError::InvalidOverride {
                        key: key.to_string(),
                        reason,
                    }
                })?,
                source: auto_config_source_from_runtime(entry.source),
                source_key: Some(key.to_string()),
            }),
            None => Ok(ResolvedValue {
                value: default,
                source: default_source,
                source_key: None,
            }),
        }
    }

    fn usize_value(
        &self,
        key: &str,
        default: usize,
        default_source: AutoConfigSource,
    ) -> Result<ResolvedValue<usize>, AutoConfigError> {
        match self.entry(key) {
            Some(entry) => Ok(ResolvedValue {
                value: parse_usize_env_value(&entry.effective_value).map_err(|reason| {
                    AutoConfigError::InvalidOverride {
                        key: key.to_string(),
                        reason,
                    }
                })?,
                source: auto_config_source_from_runtime(entry.source),
                source_key: Some(key.to_string()),
            }),
            None => Ok(ResolvedValue {
                value: default,
                source: default_source,
                source_key: None,
            }),
        }
    }

    fn usize_value_with_legacy_alias(
        &self,
        primary_key: &str,
        legacy_key: &str,
        default: usize,
        default_source: AutoConfigSource,
    ) -> Result<ResolvedValue<usize>, AutoConfigError> {
        if self.entry(primary_key).is_some() {
            return self.usize_value(primary_key, default, default_source);
        }
        if self.entry(legacy_key).is_some() {
            return self.usize_value(legacy_key, default, default_source);
        }
        Ok(ResolvedValue {
            value: default,
            source: default_source,
            source_key: None,
        })
    }

    fn optional_usize_value(
        &self,
        key: &str,
    ) -> Result<Option<ResolvedValue<usize>>, AutoConfigError> {
        match self.entry(key) {
            Some(entry) => Ok(Some(ResolvedValue {
                value: parse_usize_env_value(&entry.effective_value).map_err(|reason| {
                    AutoConfigError::InvalidOverride {
                        key: key.to_string(),
                        reason,
                    }
                })?,
                source: auto_config_source_from_runtime(entry.source),
                source_key: Some(key.to_string()),
            })),
            None => Ok(None),
        }
    }

    fn validate_attention(
        &self,
        use_vllm_paged_attn: bool,
        fa_layout: bool,
        fa2_source: bool,
        fa2_direct_ffi: bool,
        shim_present: bool,
        vllm_v1_short: bool,
    ) -> Result<(), AutoConfigError> {
        if use_vllm_paged_attn && !self.hardware.compiled_features.vllm_paged_attn {
            return self.invalid(
                "FERRUM_USE_VLLM_PAGED_ATTN",
                "vLLM paged attention is not compiled",
            );
        }
        if use_vllm_paged_attn && !self.is_cuda_backend() {
            return self.invalid(
                "FERRUM_USE_VLLM_PAGED_ATTN",
                "vLLM paged attention requires CUDA backend",
            );
        }
        if fa_layout && !use_vllm_paged_attn {
            return self.invalid(
                "FERRUM_FA_LAYOUT_VARLEN",
                "FA layout requires vLLM paged attention layout",
            );
        }
        if fa2_source && !self.hardware.compiled_features.fa2_source {
            return self.invalid(
                "FERRUM_FA2_SOURCE",
                "source-linked FA2 support is not compiled",
            );
        }
        if fa2_source && !self.is_cuda_backend() {
            return self.invalid(
                "FERRUM_FA2_SOURCE",
                "source-linked FA2 requires CUDA backend",
            );
        }
        if fa2_source && !use_vllm_paged_attn {
            return self.invalid(
                "FERRUM_FA2_SOURCE",
                "source-linked FA2 requires vLLM paged attention layout",
            );
        }
        if fa2_source && self.cuda_compute_capability_at_least(8, 0) == Some(false) {
            return self.invalid(
                "FERRUM_FA2_SOURCE",
                "source-linked FA2 requires CUDA compute capability >= 8.0",
            );
        }
        if fa2_direct_ffi && !self.hardware.compiled_features.fa2_direct_ffi {
            return self.invalid(
                "FERRUM_FA2_DIRECT_FFI",
                "direct FA2 FFI shim support is not compiled",
            );
        }
        if fa2_direct_ffi && !self.is_cuda_backend() {
            return self.invalid(
                "FERRUM_FA2_DIRECT_FFI",
                "direct FA2 FFI shim requires CUDA backend",
            );
        }
        if fa2_direct_ffi && self.cuda_compute_capability_at_least(8, 0) == Some(false) {
            return self.invalid(
                "FERRUM_FA2_DIRECT_FFI",
                "direct FA2 FFI shim requires CUDA compute capability >= 8.0",
            );
        }
        if fa2_direct_ffi && !shim_present {
            return self.invalid(
                "FERRUM_FA2_DIRECT_FFI",
                "requires FERRUM_FA2_DIRECT_FFI_SHIM",
            );
        }
        if fa2_source && fa2_direct_ffi {
            return self.unsupported(
                "attention_prefill_mixed_backend",
                "FA2 source and direct FFI shim cannot both own the prefill path",
            );
        }
        if vllm_v1_short && !use_vllm_paged_attn {
            return self.invalid(
                "FERRUM_VLLM_PAGED_ATTN_V1_SHORT",
                "short-context v1 requires vLLM paged attention",
            );
        }
        Ok(())
    }

    fn validate_moe(
        &self,
        vllm_moe: bool,
        device_route: bool,
        pair_ids: bool,
        graph: bool,
    ) -> Result<(), AutoConfigError> {
        if vllm_moe && !self.hardware.compiled_features.vllm_moe_marlin {
            return self.invalid("FERRUM_VLLM_MOE", "vLLM Marlin MoE is not compiled");
        }
        if vllm_moe && !self.is_cuda_backend() {
            return self.invalid("FERRUM_VLLM_MOE", "vLLM Marlin MoE requires CUDA backend");
        }
        if device_route && !vllm_moe {
            return self.invalid(
                "FERRUM_MOE_DEVICE_ROUTE",
                "device route currently requires vLLM MoE",
            );
        }
        if pair_ids && !vllm_moe {
            return self.invalid(
                "FERRUM_VLLM_MOE_PAIR_IDS",
                "pair-id routing requires vLLM MoE",
            );
        }
        let graph_relevant = self.model.moe.is_some() || self.workload.is_m3_preset();
        if graph && graph_relevant && !self.hardware.graph_support {
            return self.invalid(
                "FERRUM_MOE_GRAPH",
                "hardware/backend does not support CUDA graph replay",
            );
        }
        if graph && graph_relevant && !self.hardware.compiled_features.cuda_graph {
            return self.invalid("FERRUM_MOE_GRAPH", "CUDA graph support is not compiled");
        }
        if graph && graph_relevant && !vllm_moe {
            return self.invalid(
                "FERRUM_MOE_GRAPH",
                "graph decode requires the graph-clean vLLM MoE path",
            );
        }
        if graph && graph_relevant && self.model.moe.is_some() && !self.model.graph_safe_moe {
            return self.unsupported(
                "moe_graph_policy",
                "model MoE path is not marked graph-safe",
            );
        }
        Ok(())
    }

    fn validate_batched_graph(&self, graph: bool) -> Result<(), AutoConfigError> {
        if !graph {
            return Ok(());
        }
        if self.model.moe.is_some() {
            return self.invalid(
                "FERRUM_BATCHED_GRAPH",
                "legacy batched decode graph does not apply to MoE models",
            );
        }
        if !self.is_cuda_backend() {
            return self.invalid(
                "FERRUM_BATCHED_GRAPH",
                "legacy batched decode graph requires CUDA backend",
            );
        }
        if !self.hardware.graph_support {
            return self.invalid(
                "FERRUM_BATCHED_GRAPH",
                "hardware/backend does not support CUDA graph replay",
            );
        }
        if !self.hardware.compiled_features.cuda_graph {
            return self.invalid("FERRUM_BATCHED_GRAPH", "CUDA graph support is not compiled");
        }
        Ok(())
    }

    fn validate_unified_graph(
        &self,
        graph: bool,
        layers_only: bool,
        lm_head_eager: bool,
    ) -> Result<(), AutoConfigError> {
        if layers_only && !graph {
            return self.invalid(
                "FERRUM_UNIFIED_GRAPH_LAYERS_ONLY",
                "layers-only unified graph capture requires FERRUM_UNIFIED_GRAPH=1",
            );
        }
        if lm_head_eager && !graph {
            return self.invalid(
                "FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER",
                "lm-head-eager unified graph capture requires FERRUM_UNIFIED_GRAPH=1",
            );
        }
        if layers_only && lm_head_eager {
            return self.invalid(
                "FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER",
                "lm-head-eager unified graph capture conflicts with layers-only capture",
            );
        }
        if !graph {
            return Ok(());
        }
        if self.model.moe.is_some() {
            return self.invalid(
                "FERRUM_UNIFIED_GRAPH",
                "unified decode graph does not apply to MoE models",
            );
        }
        if self.model.architecture.eq_ignore_ascii_case("gemma3") && !layers_only && !lm_head_eager
        {
            return self.invalid(
                "FERRUM_UNIFIED_GRAPH",
                "full unified decode graph is disabled for Gemma3 sandwich-norm models",
            );
        }
        if !self.is_cuda_backend() {
            return self.invalid(
                "FERRUM_UNIFIED_GRAPH",
                "unified decode graph requires CUDA backend",
            );
        }
        if !self.hardware.graph_support {
            return self.invalid(
                "FERRUM_UNIFIED_GRAPH",
                "hardware/backend does not support CUDA graph replay",
            );
        }
        if !self.hardware.compiled_features.cuda_graph {
            return self.invalid("FERRUM_UNIFIED_GRAPH", "CUDA graph support is not compiled");
        }
        Ok(())
    }

    fn validate_sampling(&self, greedy: bool) -> Result<(), AutoConfigError> {
        if greedy && !self.hardware.compiled_features.greedy_argmax {
            return self.invalid("FERRUM_GREEDY_ARGMAX", "GPU argmax is not compiled");
        }
        if greedy
            && !(self.is_cuda_backend() || self.hardware.backend.eq_ignore_ascii_case("metal"))
        {
            return self.invalid(
                "FERRUM_GREEDY_ARGMAX",
                "greedy argmax requires CUDA or Metal backend",
            );
        }
        Ok(())
    }

    fn validate_memory(
        &self,
        kv_blocks: usize,
        max_sequences: usize,
        recurrent_state_max_slots: Option<usize>,
        max_batched_tokens: usize,
        requested_max_model_len: Option<usize>,
    ) -> Result<(), AutoConfigError> {
        if kv_blocks == 0 {
            return self.invalid("FERRUM_KV_MAX_BLOCKS", "must be greater than zero");
        }
        if max_sequences == 0 {
            return self.invalid("FERRUM_PAGED_MAX_SEQS", "must be greater than zero");
        }
        if recurrent_state_max_slots == Some(0) {
            return self.invalid(
                "FERRUM_RECURRENT_STATE_MAX_SLOTS",
                "must be greater than zero",
            );
        }
        if let Some(limit) = self.recurrent_state_budget_max_slots() {
            let recurrent_slots = recurrent_state_max_slots.unwrap_or(max_sequences);
            if recurrent_slots > limit {
                let key = if self.entry("FERRUM_RECURRENT_STATE_MAX_SLOTS").is_some() {
                    "FERRUM_RECURRENT_STATE_MAX_SLOTS"
                } else if self.entry("FERRUM_QWEN35_LINEAR_STATE_MAX_SLOTS").is_some() {
                    "FERRUM_QWEN35_LINEAR_STATE_MAX_SLOTS"
                } else {
                    "FERRUM_PAGED_MAX_SEQS"
                };
                return Err(AutoConfigError::InvalidOverride {
                    key: key.to_string(),
                    reason: format!(
                        "recurrent-state slot pool exceeds the model/hardware memory budget: slots={recurrent_slots}, budget={limit}; use FERRUM_RECURRENT_STATE_MAX_SLOTS={limit}, lower --max-num-seqs, or a larger-memory GPU."
                    ),
                });
            }
        }
        if max_batched_tokens < max_sequences {
            return self.invalid(
                "FERRUM_MAX_BATCHED_TOKENS",
                "must be at least FERRUM_PAGED_MAX_SEQS",
            );
        }
        let kv_token_capacity = kv_blocks.saturating_mul(DEFAULT_KV_BLOCK_SIZE_TOKENS);
        if max_batched_tokens > kv_token_capacity {
            return self.invalid(
                "FERRUM_MAX_BATCHED_TOKENS",
                "exceeds KV cache token capacity",
            );
        }
        if let Some(max_model_len) = requested_max_model_len {
            if max_model_len == 0 {
                return self.invalid("FERRUM_MAX_MODEL_LEN", "must be greater than zero");
            }
            if let Some(model_max) = self.model.max_context_len {
                if max_model_len > model_max {
                    return self.invalid(
                        "FERRUM_MAX_MODEL_LEN",
                        "exceeds model metadata max context length",
                    );
                }
            }
            if max_model_len > kv_token_capacity {
                return self.invalid(
                    "FERRUM_KV_MAX_BLOCKS",
                    "KV cache token capacity is smaller than FERRUM_MAX_MODEL_LEN",
                );
            }
        }
        Ok(())
    }

    fn validate_dtypes(&self) -> Result<(), AutoConfigError> {
        if let Some(dtype) = self.raw("FERRUM_DTYPE") {
            let dtype = dtype.to_ascii_lowercase();
            if !self.hardware.supported_dtypes.iter().any(|d| d == &dtype) {
                return self.invalid("FERRUM_DTYPE", "dtype is not supported by hardware profile");
            }
        }
        if let Some(dtype) = self.raw("FERRUM_KV_DTYPE") {
            let dtype = dtype.to_ascii_lowercase();
            if !self
                .hardware
                .supported_kv_dtypes
                .iter()
                .any(|d| d == &dtype)
            {
                return self.invalid(
                    "FERRUM_KV_DTYPE",
                    "KV dtype is not supported by hardware profile",
                );
            }
        }
        Ok(())
    }

    fn validate_layer_split_pipeline_mode(&self) -> Result<(), AutoConfigError> {
        let Some(mode) = self.raw("FERRUM_LAYER_SPLIT_PIPELINE_MODE") else {
            return Ok(());
        };
        match mode.trim().to_ascii_lowercase().as_str() {
            "batch" | "overlapped" => Ok(()),
            _ => self.invalid(
                "FERRUM_LAYER_SPLIT_PIPELINE_MODE",
                "must be batch or overlapped",
            ),
        }
    }

    fn attention_prefill_decision(
        &self,
        use_vllm_paged_attn: ResolvedValue<bool>,
        fa_layout: ResolvedValue<bool>,
        fa2_source: ResolvedValue<bool>,
        fa2_direct_ffi: ResolvedValue<bool>,
    ) -> AutoConfigDecision {
        let (selected, source, source_key) = if fa2_source.value {
            ("fa2_source", fa2_source.source, fa2_source.source_key)
        } else if fa2_direct_ffi.value {
            (
                "fa2_direct_ffi",
                fa2_direct_ffi.source,
                fa2_direct_ffi.source_key,
            )
        } else if fa_layout.value {
            ("fa_layout_varlen", fa_layout.source, fa_layout.source_key)
        } else if use_vllm_paged_attn.value {
            (
                "vllm_paged_varlen",
                use_vllm_paged_attn.source,
                use_vllm_paged_attn.source_key,
            )
        } else {
            ("legacy_paged_varlen", AutoConfigSource::Default, None)
        };
        self.decision(
            "attention_prefill_mixed_backend",
            selected,
            source,
            source_key,
            [
                "fa2_source",
                "fa2_direct_ffi",
                "fa_layout_varlen",
                "vllm_paged_varlen",
                "legacy_paged_varlen",
            ],
            self.rejected_except(
                selected,
                [
                    ("fa2_source", "source-linked FA2 path not selected"),
                    ("fa2_direct_ffi", "diagnostic direct FFI shim not selected"),
                    ("fa_layout_varlen", "FA-compatible layout not selected"),
                    ("vllm_paged_varlen", "vLLM paged varlen bridge not selected"),
                    (
                        "legacy_paged_varlen",
                        "a higher-priority attention path was selected",
                    ),
                ],
            ),
            vec![
                RuntimeConfigEffect::Performance,
                RuntimeConfigEffect::Memory,
            ],
        )
    }

    fn attention_decode_decision(
        &self,
        use_vllm_paged_attn: ResolvedValue<bool>,
        vllm_v1_short: ResolvedValue<bool>,
    ) -> AutoConfigDecision {
        let (selected, source, source_key) = if use_vllm_paged_attn.value {
            if vllm_v1_short.value {
                (
                    "vllm_paged_attn_v1_short",
                    vllm_v1_short.source,
                    vllm_v1_short.source_key,
                )
            } else {
                (
                    "vllm_paged_attn_v2",
                    vllm_v1_short.source,
                    vllm_v1_short.source_key,
                )
            }
        } else {
            ("legacy_paged_decode", use_vllm_paged_attn.source, None)
        };
        self.decision(
            "attention_decode_backend",
            selected,
            source,
            source_key,
            [
                "vllm_paged_attn_v1_short",
                "vllm_paged_attn_v2",
                "legacy_paged_decode",
            ],
            self.rejected_except(
                selected,
                [
                    (
                        "vllm_paged_attn_v1_short",
                        "short-context v1 decode not selected",
                    ),
                    ("vllm_paged_attn_v2", "v2 decode not selected"),
                    ("legacy_paged_decode", "legacy decode not selected"),
                ],
            ),
            vec![RuntimeConfigEffect::Performance],
        )
    }

    fn moe_decision(
        &self,
        vllm_moe: ResolvedValue<bool>,
        device_route: ResolvedValue<bool>,
        pair_ids: ResolvedValue<bool>,
    ) -> AutoConfigDecision {
        let selected = if vllm_moe.value && device_route.value && pair_ids.value {
            "vllm_marlin_moe_device_route_pair_ids"
        } else if vllm_moe.value && device_route.value {
            "vllm_marlin_moe_device_route"
        } else if vllm_moe.value {
            "vllm_marlin_moe"
        } else {
            "legacy_moe"
        };
        self.decision(
            "moe_implementation",
            selected,
            vllm_moe.source,
            vllm_moe.source_key,
            [
                "vllm_marlin_moe_device_route_pair_ids",
                "vllm_marlin_moe_device_route",
                "vllm_marlin_moe",
                "legacy_moe",
            ],
            self.rejected_except(
                selected,
                [
                    (
                        "vllm_marlin_moe_device_route_pair_ids",
                        "pair-id device route not selected",
                    ),
                    (
                        "vllm_marlin_moe_device_route",
                        "device-route MoE not selected",
                    ),
                    ("vllm_marlin_moe", "vLLM Marlin MoE not selected"),
                    ("legacy_moe", "legacy MoE not selected"),
                ],
            ),
            vec![RuntimeConfigEffect::Performance],
        )
    }

    fn graph_decision(&self, graph: ResolvedValue<bool>) -> AutoConfigDecision {
        let selected = if graph.value {
            "graph_clean_decode"
        } else {
            "graph_disabled"
        };
        self.decision(
            "moe_graph_policy",
            selected,
            graph.source,
            graph.source_key,
            ["graph_clean_decode", "graph_disabled"],
            self.rejected_except(
                selected,
                [
                    ("graph_clean_decode", "graph decode not selected"),
                    ("graph_disabled", "graph decode selected"),
                ],
            ),
            vec![
                RuntimeConfigEffect::Performance,
                RuntimeConfigEffect::Correctness,
            ],
        )
    }

    fn decode_graph_decision(
        &self,
        batched_graph: ResolvedValue<bool>,
        unified_graph: ResolvedValue<bool>,
        unified_graph_layers_only: ResolvedValue<bool>,
        unified_graph_lm_head_eager: ResolvedValue<bool>,
    ) -> AutoConfigDecision {
        let selected = if unified_graph.value && unified_graph_layers_only.value {
            "unified_decode_graph_layers_only"
        } else if unified_graph.value && unified_graph_lm_head_eager.value {
            "unified_decode_graph_lm_head_eager"
        } else if unified_graph.value {
            "unified_decode_graph"
        } else if batched_graph.value {
            "legacy_batched_decode_graph"
        } else {
            "graph_disabled"
        };
        let source_value = if unified_graph_layers_only.value {
            unified_graph_layers_only
        } else if unified_graph_lm_head_eager.value {
            unified_graph_lm_head_eager
        } else if unified_graph.value
            || (!batched_graph.value && unified_graph.source != AutoConfigSource::Default)
        {
            unified_graph
        } else {
            batched_graph
        };
        self.decision(
            "decode_graph_policy",
            selected,
            source_value.source,
            source_value.source_key,
            [
                "unified_decode_graph_layers_only",
                "unified_decode_graph_lm_head_eager",
                "unified_decode_graph",
                "legacy_batched_decode_graph",
                "graph_disabled",
            ],
            self.rejected_except(
                selected,
                [
                    (
                        "unified_decode_graph_layers_only",
                        "layers-only unified decode graph not selected",
                    ),
                    (
                        "unified_decode_graph_lm_head_eager",
                        "lm-head-eager unified decode graph not selected",
                    ),
                    ("unified_decode_graph", "unified decode graph not selected"),
                    (
                        "legacy_batched_decode_graph",
                        "legacy batched decode graph not selected",
                    ),
                    ("graph_disabled", "decode graph selected"),
                ],
            ),
            vec![
                RuntimeConfigEffect::Performance,
                RuntimeConfigEffect::Correctness,
            ],
        )
    }

    fn scalar_decision(
        &self,
        selection: &str,
        value: ResolvedValue<usize>,
        effect: RuntimeConfigEffect,
    ) -> AutoConfigDecision {
        self.decision(
            selection,
            &value.value.to_string(),
            value.source,
            value.source_key,
            [value.value.to_string()],
            Vec::new(),
            vec![effect],
        )
    }

    fn scheduler_decision(
        &self,
        default_prefill_first_until_active: Option<ResolvedValue<usize>>,
        default_prefill_step_chunk: Option<ResolvedValue<usize>>,
    ) -> Result<AutoConfigDecision, AutoConfigError> {
        let entries = self.entries();
        let prompt_scheduler =
            || -> Result<(String, AutoConfigSource, Option<String>), AutoConfigError> {
                let prompt_token_estimate = self.bool_value(
                    "FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE",
                    true,
                    AutoConfigSource::Default,
                )?;
                let selected = if prompt_token_estimate.value {
                    "prompt_token_estimate"
                } else {
                    "continuous_default"
                };
                Ok((
                    selected.to_string(),
                    prompt_token_estimate.source,
                    prompt_token_estimate.source_key,
                ))
            };
        let explicit_prefill_first = entries.get("FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE");
        let explicit_prefill_first_present = explicit_prefill_first.is_some();
        let implicit_prefill_first = if explicit_prefill_first.is_none()
            && !entries.contains_key("FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE")
        {
            default_prefill_first_until_active
                .as_ref()
                .map(|until| until.value.to_string())
        } else {
            None
        };
        let prefill_first = explicit_prefill_first
            .copied()
            .or(implicit_prefill_first.as_deref());
        let active_decode_chunk = entries.get("FERRUM_ACTIVE_DECODE_PREFILL_CHUNK");
        if let Some(chunk) = active_decode_chunk {
            let chunk_value = parse_usize_env_value(chunk).map_err(|reason| {
                AutoConfigError::InvalidOverride {
                    key: "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK".to_string(),
                    reason,
                }
            })?;
            self.unsupported_if(
                chunk_value == 0,
                "scheduler_admission_policy",
                "active decode prefill chunk must be greater than zero",
            )?;
        }
        let explicit_prefill_step_chunk = entries.get("FERRUM_SCHED_PREFILL_STEP_CHUNK");
        let explicit_prefill_step_chunk_present = explicit_prefill_step_chunk.is_some();
        let implicit_prefill_step_chunk = if explicit_prefill_step_chunk.is_none()
            && !entries.contains_key("FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE")
        {
            default_prefill_step_chunk
                .as_ref()
                .map(|chunk| chunk.value.to_string())
        } else {
            None
        };
        let prefill_step_chunk = explicit_prefill_step_chunk
            .copied()
            .or(implicit_prefill_step_chunk.as_deref());
        if let Some(chunk) = prefill_step_chunk {
            let chunk_value = parse_usize_env_value(chunk).map_err(|reason| {
                AutoConfigError::InvalidOverride {
                    key: "FERRUM_SCHED_PREFILL_STEP_CHUNK".to_string(),
                    reason,
                }
            })?;
            self.unsupported_if(
                chunk_value == 0,
                "scheduler_admission_policy",
                "scheduler prefill step chunk must be greater than zero",
            )?;
        }
        let (mut selected, mut source, mut source_key) = if let (Some(until), Some(chunk)) =
            (prefill_first, active_decode_chunk)
        {
            parse_usize_env_value(until).map_err(|reason| AutoConfigError::InvalidOverride {
                key: "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE".to_string(),
                reason,
            })?;
            parse_usize_env_value(chunk).map_err(|reason| AutoConfigError::InvalidOverride {
                key: "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK".to_string(),
                reason,
            })?;
            let key = "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK";
            (
                format!("prefill_first_until_active:{until}+active_decode_prefill_chunk:{chunk}"),
                self.source_for_key(key, AutoConfigSource::Default),
                Some(key.to_string()),
            )
        } else if let Some(chunk) = active_decode_chunk {
            parse_usize_env_value(chunk).map_err(|reason| AutoConfigError::InvalidOverride {
                key: "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK".to_string(),
                reason,
            })?;
            let key = "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK";
            (
                format!("active_decode_prefill_chunk:{chunk}"),
                self.source_for_key(key, AutoConfigSource::Default),
                Some(key.to_string()),
            )
        } else if let Some(until) = prefill_first {
            parse_usize_env_value(until).map_err(|reason| AutoConfigError::InvalidOverride {
                key: "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE".to_string(),
                reason,
            })?;
            let key = "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE";
            let (source, source_key) = if explicit_prefill_first_present {
                (
                    self.source_for_key(key, AutoConfigSource::Default),
                    Some(key.to_string()),
                )
            } else if let Some(default) = default_prefill_first_until_active.as_ref() {
                (default.source, default.source_key.clone())
            } else {
                (AutoConfigSource::Default, None)
            };
            (
                format!("prefill_first_until_active:{until}"),
                source,
                source_key,
            )
        } else if !entries.contains_key("FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE") {
            if let Some(until) = default_prefill_first_until_active.as_ref() {
                (
                    format!("prefill_first_until_active:{}", until.value),
                    until.source,
                    until.source_key.clone(),
                )
            } else {
                prompt_scheduler()?
            }
        } else {
            prompt_scheduler()?
        };
        if let Some(chunk) = prefill_step_chunk {
            selected.push_str(&format!("+prefill_step_chunk:{chunk}"));
            if explicit_prefill_step_chunk_present && source_key.is_none() {
                let key = "FERRUM_SCHED_PREFILL_STEP_CHUNK";
                source = self.source_for_key(key, AutoConfigSource::Default);
                source_key = Some(key.to_string());
            }
        }
        Ok(self.decision(
            "scheduler_admission_policy",
            &selected,
            source,
            source_key,
            [
                "continuous_default",
                "prompt_token_estimate",
                "prefill_first_until_active",
                "prefill_first_until_active+active_decode_prefill_chunk",
                "active_decode_prefill_chunk",
                "prefill_step_chunk",
            ],
            Vec::new(),
            vec![RuntimeConfigEffect::Performance],
        ))
    }

    fn prefix_cache_decision(&self, prefix_cache: ResolvedValue<bool>) -> AutoConfigDecision {
        let selected = if prefix_cache.value {
            "prefix_cache_enabled"
        } else {
            "prefix_cache_disabled"
        };
        self.decision(
            "prefix_cache_policy",
            selected,
            prefix_cache.source,
            prefix_cache.source_key,
            ["prefix_cache_enabled", "prefix_cache_disabled"],
            self.rejected_except(
                selected,
                [
                    ("prefix_cache_enabled", "prefix cache not selected"),
                    ("prefix_cache_disabled", "prefix cache enabled"),
                ],
            ),
            vec![
                RuntimeConfigEffect::Correctness,
                RuntimeConfigEffect::Performance,
                RuntimeConfigEffect::Memory,
            ],
        )
    }

    fn sampling_decision(&self, greedy: ResolvedValue<bool>) -> AutoConfigDecision {
        let selected = if greedy.value {
            "gpu_greedy_argmax"
        } else {
            "logits_readback"
        };
        self.decision(
            "sampling_readback_path",
            selected,
            greedy.source,
            greedy.source_key,
            ["gpu_greedy_argmax", "logits_readback"],
            self.rejected_except(
                selected,
                [
                    ("gpu_greedy_argmax", "GPU argmax not selected"),
                    ("logits_readback", "logits readback not selected"),
                ],
            ),
            vec![
                RuntimeConfigEffect::Performance,
                RuntimeConfigEffect::Correctness,
            ],
        )
    }

    fn decision<I, C>(
        &self,
        selection: &str,
        selected: &str,
        source: AutoConfigSource,
        source_key: Option<String>,
        candidates: I,
        rejected: Vec<RejectedCandidate>,
        affects: Vec<RuntimeConfigEffect>,
    ) -> AutoConfigDecision
    where
        I: IntoIterator<Item = C>,
        C: Into<String>,
    {
        AutoConfigDecision {
            schema_version: 1,
            selection: selection.to_string(),
            selected: selected.to_string(),
            source,
            source_key,
            candidates: candidates.into_iter().map(Into::into).collect(),
            rejected,
            affects,
        }
    }

    fn rejected_except<I>(&self, selected: &str, candidates: I) -> Vec<RejectedCandidate>
    where
        I: IntoIterator<Item = (&'static str, &'static str)>,
    {
        candidates
            .into_iter()
            .filter(|(value, _)| *value != selected)
            .map(|(value, reason)| RejectedCandidate {
                value: value.to_string(),
                reason: reason.to_string(),
            })
            .collect()
    }

    fn invalid<T>(&self, key: &str, reason: &str) -> Result<T, AutoConfigError> {
        Err(AutoConfigError::InvalidOverride {
            key: key.to_string(),
            reason: reason.to_string(),
        })
    }

    fn unsupported<T>(&self, selection: &str, reason: &str) -> Result<T, AutoConfigError> {
        Err(AutoConfigError::UnsupportedCombination {
            selection: selection.to_string(),
            reason: reason.to_string(),
        })
    }

    fn unsupported_if(
        &self,
        condition: bool,
        selection: &str,
        reason: &str,
    ) -> Result<(), AutoConfigError> {
        if condition {
            self.unsupported(selection, reason)
        } else {
            Ok(())
        }
    }
}

fn kv_cache_bytes_per_token_for_model(model: &ModelCapabilities) -> Option<u64> {
    let layers = model.num_hidden_layers? as u64;
    let kv_heads = model.kv_heads? as u64;
    let head_dim = model.head_dim? as u64;
    layers
        .checked_mul(2)?
        .checked_mul(kv_heads)?
        .checked_mul(head_dim)?
        .checked_mul(2)
}

fn qwen_moe_architecture_uses_vllm_paged_attn(architecture: &str) -> bool {
    architecture.eq_ignore_ascii_case("qwen3_moe")
        || architecture.eq_ignore_ascii_case("qwen3_5_moe")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedValue<T> {
    value: T,
    source: AutoConfigSource,
    source_key: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecurrentStateBudget {
    remaining_bytes: u64,
    bytes_per_sequence: u64,
    raw_slots: usize,
    floored_slots: usize,
}

fn recurrent_state_budget_for(
    model: &ModelCapabilities,
    hardware: &HardwareCapabilities,
) -> Option<RecurrentStateBudget> {
    if !(hardware.backend.eq_ignore_ascii_case("cuda")
        || hardware.backend.eq_ignore_ascii_case("metal"))
    {
        return None;
    }
    let bytes_per_sequence = model.recurrent_state_bytes_per_sequence?.max(1);
    let vram_bytes = hardware.vram_bytes?;
    let weight_bytes = model.estimated_weight_bytes?;
    let remaining = vram_bytes.saturating_sub(weight_bytes);
    let raw_slots = (remaining / bytes_per_sequence) as usize;
    Some(RecurrentStateBudget {
        remaining_bytes: remaining,
        bytes_per_sequence,
        raw_slots,
        floored_slots: floor_power_of_two(raw_slots.max(1)),
    })
}

fn parse_compute_capability(value: &str) -> Option<(u32, u32)> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let (major, minor) = value.split_once('.').unwrap_or((value, "0"));
    Some((major.trim().parse().ok()?, minor.trim().parse().ok()?))
}

fn vram_default_max_sequences(vram_bytes: u64) -> usize {
    match vram_bytes {
        bytes if bytes >= 20 * GIB => 32,
        bytes if bytes >= 12 * GIB => 16,
        bytes if bytes >= 8 * GIB => 8,
        _ => 4,
    }
}

fn default_gpu_devices_for_backend(backend: &str) -> Option<Vec<usize>> {
    backend.eq_ignore_ascii_case("cuda").then(|| vec![0])
}

fn ceil_div(value: usize, divisor: usize) -> usize {
    value.div_ceil(divisor)
}

fn effective_admission_limit(
    max_sequences: Option<usize>,
    recurrent_state_max_slots: Option<usize>,
) -> Option<usize> {
    match (max_sequences, recurrent_state_max_slots) {
        (Some(max_sequences), Some(recurrent_slots)) => Some(max_sequences.min(recurrent_slots)),
        (Some(max_sequences), None) => Some(max_sequences),
        (None, Some(recurrent_slots)) => Some(recurrent_slots),
        (None, None) => None,
    }
}

fn floor_power_of_two(value: usize) -> usize {
    if value <= 1 {
        return 1;
    }
    1usize << (usize::BITS - 1 - value.leading_zeros())
}

fn auto_config_source_from_runtime(source: RuntimeConfigSource) -> AutoConfigSource {
    match source {
        RuntimeConfigSource::Default => AutoConfigSource::Default,
        RuntimeConfigSource::ConfigFile => AutoConfigSource::ConfigFile,
        RuntimeConfigSource::Cli => AutoConfigSource::Cli,
        RuntimeConfigSource::Env => AutoConfigSource::Env,
        RuntimeConfigSource::ScriptCase => AutoConfigSource::ScriptCase,
        RuntimeConfigSource::MemoryProfile => AutoConfigSource::MemoryProfile,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(vars: &[(&str, &str)]) -> RuntimeConfigSnapshot {
        RuntimeConfigSnapshot::from_env_vars(vars.iter().copied())
    }

    fn snapshot_with_sources(vars: &[(&str, &str, RuntimeConfigSource)]) -> RuntimeConfigSnapshot {
        let mut entries: Vec<_> = vars
            .iter()
            .map(|(key, effective_value, source)| RuntimeConfigEntry {
                key: (*key).to_string(),
                effective_value: (*effective_value).to_string(),
                source: *source,
                affects: vec![RuntimeConfigEffect::Performance],
            })
            .collect();
        entries.sort_by(|a, b| a.key.cmp(&b.key));
        RuntimeConfigSnapshot { entries }
    }

    fn m3(vars: &[(&str, &str)], features: CompiledKernelFeatures) -> FerrumConfigBuilder {
        FerrumConfigBuilder::new(snapshot(vars))
            .with_model_capabilities(ModelCapabilities::qwen3_30b_a3b_gptq_int4())
            .with_hardware_capabilities(HardwareCapabilities::rtx4090_cuda(features))
            .with_workload_profile(WorkloadProfile::m3_qwen3_30b_a3b_int4())
    }

    fn m3_with_hardware(
        vars: &[(&str, &str)],
        hardware: HardwareCapabilities,
    ) -> FerrumConfigBuilder {
        FerrumConfigBuilder::new(snapshot(vars))
            .with_model_capabilities(ModelCapabilities::qwen3_30b_a3b_gptq_int4())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(WorkloadProfile::m3_qwen3_30b_a3b_int4())
    }

    fn qwen35_moe_gptq_int4_model() -> ModelCapabilities {
        let mut model = ModelCapabilities::qwen3_30b_a3b_gptq_int4();
        model.architecture = "qwen3_5_moe".to_string();
        model.head_dim = Some(256);
        model.num_hidden_layers = Some(40);
        model.kv_heads = Some(8);
        model.estimated_weight_bytes = Some(24_419_939_760);
        model.recurrent_state_bytes_per_sequence = Some(65_863_680);
        model
    }

    fn synthetic_tight_recurrent_state_model() -> ModelCapabilities {
        ModelCapabilities {
            architecture: "synthetic_recurrent_state".to_string(),
            quantization: None,
            moe: None,
            max_context_len: Some(262_144),
            num_hidden_layers: Some(40),
            head_dim: Some(256),
            kv_heads: Some(8),
            estimated_weight_bytes: Some(24_419_939_760),
            recurrent_state_bytes_per_sequence: Some(65_863_680),
            supported_dtypes: vec!["fp16".to_string()],
            graph_safe_moe: false,
        }
    }

    fn qwen25_layer_split_runtime_entries(source: RuntimeConfigSource) -> RuntimeConfigSnapshot {
        snapshot_with_sources(&[
            ("FERRUM_REQUESTED_GPU_DEVICES", "0,1", source),
            ("FERRUM_SELECTED_GPU_DEVICES", "0,1", source),
            ("FERRUM_CUDA_DEVICE_COUNT", "2", source),
            (
                "FERRUM_SELECTED_DISTRIBUTED_STRATEGY",
                "layer_split",
                source,
            ),
            (
                "FERRUM_SELECTED_LAYER_SPLIT_PLAN",
                "stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79",
                source,
            ),
            ("FERRUM_LAYER_SPLIT_PIPELINE_MODE", "batch", source),
            ("FERRUM_MAX_MODEL_LEN", "4096", source),
            ("FERRUM_KV_MAX_BLOCKS", "1024", source),
            ("FERRUM_KV_CAPACITY", "1024", source),
            ("FERRUM_PAGED_MAX_SEQS", "16", source),
            ("FERRUM_MAX_BATCHED_TOKENS", "1536", source),
            ("FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE", "16", source),
        ])
    }

    fn gemma3_gptq_model() -> ModelCapabilities {
        ModelCapabilities {
            architecture: "gemma3".to_string(),
            quantization: Some("gptq_int4".to_string()),
            moe: None,
            max_context_len: Some(131_072),
            num_hidden_layers: Some(62),
            head_dim: Some(256),
            kv_heads: Some(16),
            estimated_weight_bytes: Some(15 * GIB),
            recurrent_state_bytes_per_sequence: None,
            supported_dtypes: vec!["fp16".to_string()],
            graph_safe_moe: false,
        }
    }

    fn expect_invalid_key(vars: &[(&str, &str)], key: &str) {
        expect_invalid_key_with_features(
            vars,
            key,
            CompiledKernelFeatures::m3_fast_path_without_fa2(),
        );
    }

    fn expect_invalid_key_with_features(
        vars: &[(&str, &str)],
        key: &str,
        features: CompiledKernelFeatures,
    ) {
        expect_invalid_key_with_hardware(vars, key, HardwareCapabilities::rtx4090_cuda(features));
    }

    fn expect_invalid_key_with_hardware(
        vars: &[(&str, &str)],
        key: &str,
        hardware: HardwareCapabilities,
    ) {
        let err = m3_with_hardware(vars, hardware)
            .resolve()
            .expect_err("override should fail");
        match err {
            AutoConfigError::InvalidOverride { key: actual, .. } => assert_eq!(actual, key),
            other => panic!("expected invalid override for {key}, got {other:?}"),
        }
    }

    fn cpu_hardware_with_features(features: CompiledKernelFeatures) -> HardwareCapabilities {
        HardwareCapabilities {
            backend: "cpu".to_string(),
            supported_dtypes: vec!["fp32".to_string()],
            supported_kv_dtypes: vec!["fp16".to_string()],
            compiled_features: features,
            ..HardwareCapabilities::unknown()
        }
    }

    #[test]
    fn m3_preset_selects_current_safe_fast_path_without_fa2() {
        let resolved = m3(&[], CompiledKernelFeatures::m3_fast_path_without_fa2())
            .resolve()
            .unwrap();
        let decisions: BTreeMap<_, _> = resolved
            .decisions
            .iter()
            .map(|decision| (decision.selection.as_str(), decision.selected.as_str()))
            .collect();
        assert_eq!(
            decisions["attention_prefill_mixed_backend"],
            "vllm_paged_varlen"
        );
        assert_eq!(
            decisions["attention_decode_backend"],
            "vllm_paged_attn_v1_short"
        );
        assert_eq!(
            decisions["moe_implementation"],
            "vllm_marlin_moe_device_route_pair_ids"
        );
        assert_eq!(decisions["moe_graph_policy"], "graph_disabled");
        assert_eq!(decisions["decode_graph_policy"], "graph_disabled");
        assert_eq!(decisions["prefix_cache_policy"], "prefix_cache_disabled");
        assert_eq!(decisions["sampling_readback_path"], "gpu_greedy_argmax");
        assert_eq!(
            resolved.preset.as_deref(),
            Some(M3_QWEN3_30B_A3B_INT4_PRESET)
        );
    }

    #[test]
    fn cuda_gptq_moe_enables_vllm_marlin_without_m3_preset() {
        // `ferrum run` resolves with the serving-default workload, NOT the m3
        // bench preset, so the old `is_m3_preset()`-gated FERRUM_VLLM_MOE never
        // fired and the 30B fell back to the slow host-route MoE (~9.7 vs ~59
        // tok/s on a 4090). A CUDA GPTQ MoE must get the vLLM-Marlin fast path
        // on capability alone.
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot(&[]))
            .with_model_capabilities(ModelCapabilities::qwen3_30b_a3b_gptq_int4())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .unwrap();
        let decisions: BTreeMap<_, _> = resolved
            .decisions
            .iter()
            .map(|decision| (decision.selection.as_str(), decision.selected.as_str()))
            .collect();
        assert_ne!(
            resolved.preset.as_deref(),
            Some(M3_QWEN3_30B_A3B_INT4_PRESET),
            "serving-default workload must not be the m3 preset"
        );
        assert_eq!(
            decisions["moe_implementation"], "vllm_marlin_moe_device_route_pair_ids",
            "CUDA GPTQ MoE should get the fast vLLM-Marlin path without the m3 preset"
        );
        // The decision is not enough — the model reads FERRUM_VLLM_MOE from the
        // effective config, not the decisions. The resolved knob must be a
        // runtime_config entry so `ferrum run`'s materialize/apply propagates it.
        let entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|e| e.key == "FERRUM_VLLM_MOE");
        assert_eq!(
            entry.map(|e| e.effective_value.as_str()),
            Some("1"),
            "resolved FERRUM_VLLM_MOE must be materialized into the effective config"
        );
    }

    #[test]
    fn cuda_qwen3_moe_enables_vllm_paged_attn_without_m3_preset() {
        // `ferrum run` and ordinary `serve` use the serving-default workload,
        // not the m3 preset. Qwen3-MoE on CUDA with the VPA kernel compiled
        // must still select and materialize the paged-attention runtime knob,
        // otherwise the effective config/decision trace says "legacy" while
        // the model runtime can take the VPA path through its own defaults.
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot(&[]))
            .with_model_capabilities(ModelCapabilities::qwen3_30b_a3b_gptq_int4())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .unwrap();
        let decisions: BTreeMap<_, _> = resolved
            .decisions
            .iter()
            .map(|decision| (decision.selection.as_str(), decision.selected.as_str()))
            .collect();
        assert_eq!(
            decisions["attention_decode_backend"], "vllm_paged_attn_v1_short",
            "CUDA Qwen3-MoE should get VPA decode without the m3 preset"
        );
        let entry = |key: &str| {
            resolved
                .runtime_config
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing runtime config entry {key}"))
        };
        assert_eq!(entry("FERRUM_USE_VLLM_PAGED_ATTN").effective_value, "1");
        assert_eq!(
            entry("FERRUM_VLLM_PAGED_ATTN_V1_SHORT").effective_value,
            "1"
        );
    }

    #[test]
    fn cuda_qwen35_moe_enables_vllm_paged_attn_v2_without_m3_preset() {
        // Qwen3.5-MoE shares the Qwen MoE CUDA fast-path requirements, but its
        // full-attention head_dim is 256. The H256 path uses the v2 paged
        // attention launcher, so the default decision trace must not report
        // the H128-oriented v1-short path.
        let mut hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        hardware.vram_bytes = Some(48 * GIB);
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let model = qwen35_moe_gptq_int4_model();
        let resolved = FerrumConfigBuilder::new(snapshot(&[]))
            .with_model_capabilities(model)
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .unwrap();
        let decisions: BTreeMap<_, _> = resolved
            .decisions
            .iter()
            .map(|decision| (decision.selection.as_str(), decision.selected.as_str()))
            .collect();
        assert_eq!(
            decisions["attention_prefill_mixed_backend"],
            "vllm_paged_varlen"
        );
        assert_eq!(decisions["attention_decode_backend"], "vllm_paged_attn_v2");
        let entry = |key: &str| {
            resolved
                .runtime_config
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing runtime config entry {key}"))
        };
        assert_eq!(entry("FERRUM_USE_VLLM_PAGED_ATTN").effective_value, "1");
        assert_eq!(
            entry("FERRUM_VLLM_PAGED_ATTN_V1_SHORT").effective_value,
            "0"
        );
    }

    #[test]
    fn recurrent_state_budget_caps_default_slots_without_model_vram_special_case() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot(&[("FERRUM_PAGED_MAX_SEQS", "32")]))
            .with_model_capabilities(synthetic_tight_recurrent_state_model())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .unwrap();
        let decision = |selection: &str| {
            resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == selection)
                .unwrap_or_else(|| panic!("missing decision {selection}"))
        };

        assert_eq!(decision("max_sequences").selected, "32");
        assert_eq!(decision("recurrent_state_max_slots").selected, "16");
        assert_eq!(
            decision("recurrent_state_max_slots").source,
            AutoConfigSource::MemoryProfile
        );
        let entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_RECURRENT_STATE_MAX_SLOTS")
            .expect("memory-profile recurrent slot cap should reach effective runtime config");
        assert_eq!(entry.effective_value, "16");
        assert_eq!(entry.source, RuntimeConfigSource::MemoryProfile);
        let doc = resolved.effective_config_document();
        assert_eq!(doc["selected_max_sequences"], serde_json::json!(32));
        assert_eq!(
            doc["selected_recurrent_state_max_slots"],
            serde_json::json!(16)
        );
        assert_eq!(doc["selected_admission_limit"], serde_json::json!(16));
        assert_eq!(
            doc["admission"]["effective_max_concurrent"],
            serde_json::json!(16)
        );
        assert_eq!(
            doc["admission"]["recurrent_state_max_slots"],
            serde_json::json!(16)
        );
        assert_eq!(
            doc["admission"]["memory_estimate"]["recurrent_state_bytes_per_sequence"],
            serde_json::json!(65_863_680u64)
        );
        assert_eq!(
            doc["admission"]["memory_estimate"]["recurrent_state_budget_bytes"],
            serde_json::json!(24u64 * GIB - 24_419_939_760u64)
        );
        assert_eq!(
            doc["admission"]["memory_estimate"]["recurrent_state_budget_raw_slots"],
            serde_json::json!(20)
        );
        assert_eq!(
            doc["admission"]["memory_estimate"]["recurrent_state_budget_max_slots"],
            serde_json::json!(16)
        );
        assert_eq!(
            doc["admission"]["memory_estimate"]["recurrent_state_capacity_bytes"],
            serde_json::json!(16u64 * 65_863_680u64)
        );
    }

    #[test]
    fn qwen35_fast_recurrent_state_budget_caps_default_slots_without_vram_special_case() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot(&[("FERRUM_PAGED_MAX_SEQS", "32")]))
            .with_model_capabilities(qwen35_moe_gptq_int4_model())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .unwrap();
        let decision = |selection: &str| {
            resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == selection)
                .unwrap_or_else(|| panic!("missing decision {selection}"))
        };

        assert_eq!(decision("max_sequences").selected, "32");
        assert_eq!(decision("recurrent_state_max_slots").selected, "16");
        assert_eq!(
            decision("recurrent_state_max_slots").source,
            AutoConfigSource::MemoryProfile
        );
        let doc = resolved.effective_config_document();
        assert_eq!(doc["selected_max_sequences"], serde_json::json!(32));
        assert_eq!(
            doc["selected_recurrent_state_max_slots"],
            serde_json::json!(16)
        );
        assert_eq!(doc["selected_admission_limit"], serde_json::json!(16));
        assert_eq!(
            doc["admission"]["memory_estimate"]["recurrent_state_bytes_per_sequence"],
            serde_json::json!(65_863_680u64)
        );
        assert_eq!(
            doc["admission"]["memory_estimate"]["recurrent_state_budget_raw_slots"],
            serde_json::json!(20)
        );
        assert_eq!(
            doc["admission"]["memory_estimate"]["recurrent_state_budget_max_slots"],
            serde_json::json!(16)
        );
        assert_eq!(
            doc["admission"]["memory_estimate"]["recurrent_state_capacity_bytes"],
            serde_json::json!(16u64 * 65_863_680u64)
        );
    }

    #[test]
    fn recurrent_state_budget_rejects_explicit_slot_pool_above_budget() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let err = FerrumConfigBuilder::new(snapshot(&[
            ("FERRUM_PAGED_MAX_SEQS", "32"),
            ("FERRUM_RECURRENT_STATE_MAX_SLOTS", "32"),
        ]))
        .with_model_capabilities(synthetic_tight_recurrent_state_model())
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .unwrap_err();

        match err {
            AutoConfigError::InvalidOverride { key, reason } => {
                assert_eq!(key, "FERRUM_RECURRENT_STATE_MAX_SLOTS");
                assert!(
                    reason.contains("recurrent-state slot pool exceeds"),
                    "{reason}"
                );
                assert!(
                    reason.contains("FERRUM_RECURRENT_STATE_MAX_SLOTS=16"),
                    "{reason}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn effective_admission_limit_is_capped_by_recurrent_state_slots_when_present() {
        assert_eq!(effective_admission_limit(Some(32), Some(16)), Some(16));
        assert_eq!(effective_admission_limit(Some(16), Some(32)), Some(16));
        assert_eq!(effective_admission_limit(Some(32), None), Some(32));
        assert_eq!(effective_admission_limit(None, Some(16)), Some(16));
        assert_eq!(effective_admission_limit(None, None), None);
    }

    #[test]
    fn explicit_recurrent_state_slot_cap_can_keep_scheduler_width_above_state_pool() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot(&[
            ("FERRUM_PAGED_MAX_SEQS", "32"),
            ("FERRUM_RECURRENT_STATE_MAX_SLOTS", "16"),
        ]))
        .with_model_capabilities(synthetic_tight_recurrent_state_model())
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .unwrap();
        let decision = |selection: &str| {
            resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == selection)
                .unwrap_or_else(|| panic!("missing decision {selection}"))
        };
        assert_eq!(decision("max_sequences").selected, "32");
        assert_eq!(decision("recurrent_state_max_slots").selected, "16");
        let entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_RECURRENT_STATE_MAX_SLOTS")
            .expect("recurrent-state slot cap should reach effective runtime config");
        assert_eq!(entry.effective_value, "16");
    }

    #[test]
    fn recurrent_state_budget_accepts_legacy_qwen35_slot_alias() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot(&[
            ("FERRUM_PAGED_MAX_SEQS", "32"),
            ("FERRUM_QWEN35_LINEAR_STATE_MAX_SLOTS", "16"),
        ]))
        .with_model_capabilities(synthetic_tight_recurrent_state_model())
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .unwrap();
        let decision = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "recurrent_state_max_slots")
            .expect("missing recurrent_state_max_slots decision");
        assert_eq!(decision.selected, "16");
        assert_eq!(
            decision.source_key.as_deref(),
            Some("FERRUM_QWEN35_LINEAR_STATE_MAX_SLOTS")
        );
        assert!(resolved
            .runtime_config
            .entries
            .iter()
            .any(|entry| entry.key == "FERRUM_QWEN35_LINEAR_STATE_MAX_SLOTS"
                && entry.effective_value == "16"));
    }

    #[test]
    fn recurrent_state_budget_rejects_zero_slot_pool() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let err = FerrumConfigBuilder::new(snapshot(&[
            ("FERRUM_PAGED_MAX_SEQS", "16"),
            ("FERRUM_RECURRENT_STATE_MAX_SLOTS", "0"),
        ]))
        .with_model_capabilities(synthetic_tight_recurrent_state_model())
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .unwrap_err();

        assert!(matches!(
            err,
            AutoConfigError::InvalidOverride { key, .. }
                if key == "FERRUM_RECURRENT_STATE_MAX_SLOTS"
        ));
    }

    #[test]
    fn recurrent_state_budget_allows_c32_when_memory_budget_fits() {
        let mut hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        hardware.vram_bytes = Some(48 * GIB);
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot(&[("FERRUM_PAGED_MAX_SEQS", "32")]))
            .with_model_capabilities(synthetic_tight_recurrent_state_model())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .unwrap();
        let max_sequences = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "max_sequences")
            .unwrap();
        assert_eq!(max_sequences.selected, "32");
    }

    #[test]
    fn cuda_qwen3_moe_vllm_paged_attn_env_opt_out_is_materialized() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot(&[("FERRUM_USE_VLLM_PAGED_ATTN", "0")]))
            .with_model_capabilities(ModelCapabilities::qwen3_30b_a3b_gptq_int4())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .unwrap();
        let decisions: BTreeMap<_, _> = resolved
            .decisions
            .iter()
            .map(|decision| (decision.selection.as_str(), decision.selected.as_str()))
            .collect();
        assert_eq!(decisions["attention_decode_backend"], "legacy_paged_decode");
        let entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_USE_VLLM_PAGED_ATTN")
            .expect("env opt-out should stay in effective config");
        assert_eq!(entry.effective_value, "0");
        assert_eq!(entry.source, RuntimeConfigSource::Env);
    }

    #[test]
    fn qwen25_72b_layer_split_preset_selects_batch_tuned_defaults() {
        let resolved = FerrumConfigBuilder::new(qwen25_layer_split_runtime_entries(
            RuntimeConfigSource::Default,
        ))
        .with_model_capabilities(ModelCapabilities::qwen25_72b_gptq_int4())
        .with_hardware_capabilities(HardwareCapabilities::rtx4090_cuda(
            CompiledKernelFeatures::m3_fast_path_without_fa2(),
        ))
        .with_workload_profile(WorkloadProfile::qwen25_72b_gptq_int4_2x4090_layer_split())
        .resolve()
        .unwrap();
        let decision = |selection: &str| {
            resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == selection)
                .unwrap_or_else(|| panic!("missing decision {selection}"))
        };

        assert_eq!(
            resolved.preset.as_deref(),
            Some(QWEN25_72B_GPTQ_INT4_2X4090_LAYER_SPLIT_PRESET)
        );
        assert_eq!(decision("kv_block_count").selected, "1024");
        assert_eq!(decision("max_sequences").selected, "16");
        assert_eq!(decision("max_batched_tokens").selected, "1536");
        assert_eq!(decision("max_model_len").selected, "4096");
        assert_eq!(
            decision("scheduler_admission_policy").selected,
            "prefill_first_until_active:16+prefill_step_chunk:96"
        );
        assert_eq!(
            decision("scheduler_admission_policy").source,
            AutoConfigSource::Default
        );

        let doc = resolved.effective_config_document();
        assert_eq!(doc["selected_pipeline_mode"], "batch");
        assert_eq!(doc["selected_microbatch_size"], 16);
        assert_eq!(doc["selected_kv_capacity"], 1024);
    }

    #[test]
    fn source_fa2_selects_source_linked_attention_when_compiled() {
        let resolved = m3(
            &[("FERRUM_FA2_SOURCE", "1")],
            CompiledKernelFeatures::m3_fast_path_with_source_fa2(),
        )
        .resolve()
        .unwrap();
        let decisions: BTreeMap<_, _> = resolved
            .decisions
            .iter()
            .map(|decision| (decision.selection.as_str(), decision.selected.as_str()))
            .collect();

        assert_eq!(decisions["attention_prefill_mixed_backend"], "fa2_source");
    }

    #[test]
    fn source_fa2_is_rejected_when_not_compiled() {
        expect_invalid_key(&[("FERRUM_FA2_SOURCE", "1")], "FERRUM_FA2_SOURCE");
    }

    #[test]
    fn hardware_capabilities_keep_m3_preset_on_compatible_backend_paths() {
        let resolved = m3_with_hardware(
            &[],
            cpu_hardware_with_features(CompiledKernelFeatures::m3_fast_path_with_source_fa2()),
        )
        .resolve()
        .unwrap();
        let decisions: BTreeMap<_, _> = resolved
            .decisions
            .iter()
            .map(|decision| (decision.selection.as_str(), decision.selected.as_str()))
            .collect();

        assert_eq!(
            decisions["attention_prefill_mixed_backend"],
            "legacy_paged_varlen"
        );
        assert_eq!(decisions["attention_decode_backend"], "legacy_paged_decode");
        assert_eq!(decisions["moe_implementation"], "legacy_moe");
        assert_eq!(decisions["moe_graph_policy"], "graph_disabled");
        assert_eq!(decisions["sampling_readback_path"], "logits_readback");
    }

    #[test]
    fn effective_config_document_records_cuda_gpu_device_selection() {
        let resolved = FerrumConfigBuilder::new(snapshot_with_sources(&[
            (
                "FERRUM_REQUESTED_GPU_DEVICES",
                "0,1",
                RuntimeConfigSource::Cli,
            ),
            (
                "FERRUM_SELECTED_GPU_DEVICES",
                "0,1",
                RuntimeConfigSource::Cli,
            ),
            ("FERRUM_CUDA_DEVICE_COUNT", "2", RuntimeConfigSource::Cli),
            (
                "FERRUM_SELECTED_DISTRIBUTED_STRATEGY",
                "layer_split",
                RuntimeConfigSource::Cli,
            ),
            (
                "FERRUM_SELECTED_LAYER_SPLIT_PLAN",
                "stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79",
                RuntimeConfigSource::Cli,
            ),
            (
                "FERRUM_SELECTED_LAYER_SPLIT_STAGES",
                r#"[{"stage":0,"device":0,"layer_start":0,"layer_end":39},{"stage":1,"device":1,"layer_start":40,"layer_end":79}]"#,
                RuntimeConfigSource::Cli,
            ),
            ("FERRUM_KV_CAPACITY", "512", RuntimeConfigSource::Cli),
        ]))
        .with_hardware_capabilities(HardwareCapabilities::rtx4090_cuda(
            CompiledKernelFeatures::m3_fast_path_without_fa2(),
        ))
        .resolve()
        .unwrap();

        let doc = resolved.effective_config_document();
        assert_eq!(doc["backend"], "cuda");
        assert_eq!(doc["requested_gpu_devices"], serde_json::json!([0, 1]));
        assert_eq!(doc["selected_gpu_devices"], serde_json::json!([0, 1]));
        assert_eq!(doc["cuda_device_count"], 2);
        assert_eq!(doc["selected_distributed_strategy"], "layer_split");
        assert_eq!(
            doc["selected_layer_split_plan"],
            "stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79"
        );
        assert_eq!(
            doc["selected_layer_split_stages"],
            serde_json::json!([
                {"stage": 0, "device": 0, "layer_start": 0, "layer_end": 39},
                {"stage": 1, "device": 1, "layer_start": 40, "layer_end": 79}
            ])
        );
        assert_eq!(doc["selected_weight_placement"], "layer_split");
        assert_eq!(doc["selected_pipeline_mode"], "overlapped");
        assert_eq!(doc["selected_stage_bridge"], "host");
        assert_eq!(
            doc["selected_microbatch_size"],
            serde_json::json!(doc["selected_max_sequences"].as_u64().unwrap().div_ceil(2))
        );
        assert_eq!(
            doc["selected_admission_limit"],
            doc["selected_max_sequences"]
        );
        assert_eq!(doc["selected_kv_capacity"], 512);
    }

    #[test]
    fn effective_config_document_honors_explicit_layer_split_batch_mode() {
        let resolved = FerrumConfigBuilder::new(snapshot_with_sources(&[
            (
                "FERRUM_REQUESTED_GPU_DEVICES",
                "0,1",
                RuntimeConfigSource::Cli,
            ),
            (
                "FERRUM_SELECTED_GPU_DEVICES",
                "0,1",
                RuntimeConfigSource::Cli,
            ),
            (
                "FERRUM_SELECTED_DISTRIBUTED_STRATEGY",
                "layer_split",
                RuntimeConfigSource::Cli,
            ),
            (
                "FERRUM_SELECTED_LAYER_SPLIT_PLAN",
                "stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79",
                RuntimeConfigSource::Cli,
            ),
            (
                "FERRUM_LAYER_SPLIT_PIPELINE_MODE",
                "batch",
                RuntimeConfigSource::Cli,
            ),
            ("FERRUM_PAGED_MAX_SEQS", "16", RuntimeConfigSource::Cli),
        ]))
        .with_hardware_capabilities(HardwareCapabilities::rtx4090_cuda(
            CompiledKernelFeatures::m3_fast_path_without_fa2(),
        ))
        .resolve()
        .unwrap();

        let doc = resolved.effective_config_document();
        assert_eq!(doc["selected_pipeline_mode"], "batch");
        assert_eq!(doc["selected_microbatch_size"], 16);
    }

    #[test]
    fn invalid_layer_split_pipeline_mode_is_rejected() {
        expect_invalid_key_with_hardware(
            &[("FERRUM_LAYER_SPLIT_PIPELINE_MODE", "serial")],
            "FERRUM_LAYER_SPLIT_PIPELINE_MODE",
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2()),
        );
    }

    #[test]
    fn hardware_incompatible_attention_and_sampling_overrides_are_rejected() {
        let cpu =
            cpu_hardware_with_features(CompiledKernelFeatures::m3_fast_path_with_source_fa2());
        expect_invalid_key_with_hardware(
            &[("FERRUM_USE_VLLM_PAGED_ATTN", "1")],
            "FERRUM_USE_VLLM_PAGED_ATTN",
            cpu.clone(),
        );
        expect_invalid_key_with_hardware(
            &[("FERRUM_VLLM_MOE", "1")],
            "FERRUM_VLLM_MOE",
            cpu.clone(),
        );
        expect_invalid_key_with_hardware(
            &[("FERRUM_GREEDY_ARGMAX", "1")],
            "FERRUM_GREEDY_ARGMAX",
            cpu.clone(),
        );
        expect_invalid_key_with_hardware(&[("FERRUM_FA2_SOURCE", "1")], "FERRUM_FA2_SOURCE", cpu);

        let mut old_cuda = HardwareCapabilities::rtx4090_cuda(
            CompiledKernelFeatures::m3_fast_path_with_source_fa2(),
        );
        old_cuda.compute_capability = Some("7.5".to_string());
        expect_invalid_key_with_hardware(
            &[("FERRUM_FA2_SOURCE", "1")],
            "FERRUM_FA2_SOURCE",
            old_cuda,
        );
    }

    #[test]
    fn hardware_capacity_sizes_default_sequence_budget_without_overriding_user_values() {
        let mut small_gpu =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        small_gpu.sm_count = Some(16);
        small_gpu.vram_bytes = Some(24 * 1024 * 1024 * 1024);

        let resolved = m3_with_hardware(&[], small_gpu.clone()).resolve().unwrap();
        let decision = |selection: &str| {
            resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == selection)
                .unwrap()
        };
        let max_sequences = decision("max_sequences");
        assert_eq!(max_sequences.selected, "4");
        assert_eq!(max_sequences.source, AutoConfigSource::HardwareCapability);
        let max_batched_tokens = decision("max_batched_tokens");
        assert_eq!(max_batched_tokens.selected, "256");
        assert_eq!(
            max_batched_tokens.source,
            AutoConfigSource::HardwareCapability
        );

        let resolved = m3_with_hardware(&[("FERRUM_PAGED_MAX_SEQS", "16")], small_gpu)
            .resolve()
            .unwrap();
        let max_sequences = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "max_sequences")
            .unwrap();
        assert_eq!(max_sequences.selected, "16");
        assert_eq!(max_sequences.source, AutoConfigSource::Env);
        assert_eq!(
            max_sequences.source_key.as_deref(),
            Some("FERRUM_PAGED_MAX_SEQS")
        );
    }

    #[test]
    fn vram_capacity_caps_m3_default_sequence_budget() {
        let mut low_vram_gpu =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        low_vram_gpu.sm_count = Some(128);
        low_vram_gpu.vram_bytes = Some(7 * 1024 * 1024 * 1024);

        let resolved = m3_with_hardware(&[], low_vram_gpu).resolve().unwrap();
        let max_sequences = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "max_sequences")
            .unwrap();
        assert_eq!(max_sequences.selected, "4");
        assert_eq!(max_sequences.source, AutoConfigSource::HardwareCapability);
    }

    #[test]
    fn memory_budget_keeps_rtx4090_m3_kv_blocks_but_caps_constrained_vram() {
        let resolved = m3(&[], CompiledKernelFeatures::m3_fast_path_without_fa2())
            .resolve()
            .unwrap();
        let decision = |selection: &str| {
            resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == selection)
                .unwrap()
        };
        assert_eq!(decision("kv_block_count").selected, "2048");
        assert_eq!(
            decision("kv_block_count").source,
            AutoConfigSource::WorkloadPreset
        );

        let mut constrained =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        constrained.vram_bytes = Some(20 * 1024 * 1024 * 1024);
        let resolved = m3_with_hardware(&[], constrained).resolve().unwrap();
        let decision = |selection: &str| {
            resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == selection)
                .unwrap()
        };
        assert_eq!(decision("kv_block_count").selected, "2");
        assert_eq!(
            decision("kv_block_count").source,
            AutoConfigSource::HardwareCapability
        );
        assert_eq!(decision("max_batched_tokens").selected, "32");
        assert_eq!(
            decision("max_batched_tokens").source,
            AutoConfigSource::HardwareCapability
        );
    }

    #[test]
    fn compute_capability_parser_accepts_major_minor_and_major_only() {
        assert_eq!(parse_compute_capability("8.9"), Some((8, 9)));
        assert_eq!(parse_compute_capability("9"), Some((9, 0)));
        assert_eq!(parse_compute_capability("N/A"), None);
    }

    #[test]
    fn vram_capacity_tiers_are_monotonic() {
        assert_eq!(vram_default_max_sequences(24 * 1024 * 1024 * 1024), 32);
        assert_eq!(vram_default_max_sequences(16 * 1024 * 1024 * 1024), 16);
        assert_eq!(vram_default_max_sequences(8 * 1024 * 1024 * 1024), 8);
        assert_eq!(vram_default_max_sequences(6 * 1024 * 1024 * 1024), 4);
    }

    #[test]
    fn accelerator_serving_default_uses_hardware_concurrency_budget() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        assert_eq!(workload.target_concurrency, 32);

        let resolved = FerrumConfigBuilder::new(snapshot(&[]))
            .with_model_capabilities(ModelCapabilities::unknown())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .unwrap();
        let max_sequences = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "max_sequences")
            .unwrap();
        assert_eq!(max_sequences.selected, "32");
        let scheduler = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "scheduler_admission_policy")
            .unwrap();
        assert_eq!(
            scheduler.selected,
            "prefill_first_until_active:32+prefill_step_chunk:64"
        );
        assert_eq!(scheduler.source, AutoConfigSource::Default);
        let scheduler_entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE")
            .unwrap_or_else(|| panic!("missing FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE entry"));
        assert_eq!(scheduler_entry.effective_value, "32");
        assert_eq!(scheduler_entry.source, RuntimeConfigSource::Default);
        let step_entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_SCHED_PREFILL_STEP_CHUNK")
            .unwrap_or_else(|| panic!("missing FERRUM_SCHED_PREFILL_STEP_CHUNK entry"));
        assert_eq!(step_entry.effective_value, "64");
        assert_eq!(step_entry.source, RuntimeConfigSource::Default);
    }

    #[test]
    fn accelerator_serving_default_enables_greedy_argmax_when_compiled() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot(&[]))
            .with_model_capabilities(ModelCapabilities::unknown())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .unwrap();
        let sampling = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "sampling_readback_path")
            .unwrap();
        assert_eq!(sampling.selected, "gpu_greedy_argmax");
        assert_eq!(sampling.source, AutoConfigSource::HardwareCapability);
        let greedy_entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_GREEDY_ARGMAX")
            .unwrap_or_else(|| panic!("missing FERRUM_GREEDY_ARGMAX entry"));
        assert_eq!(greedy_entry.effective_value, "1");
    }

    #[test]
    fn explicit_greedy_argmax_disable_keeps_logits_readback() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot_with_sources(&[(
            "FERRUM_GREEDY_ARGMAX",
            "0",
            RuntimeConfigSource::Cli,
        )]))
        .with_model_capabilities(ModelCapabilities::unknown())
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .unwrap();
        let sampling = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "sampling_readback_path")
            .unwrap();
        assert_eq!(sampling.selected, "logits_readback");
        assert_eq!(sampling.source, AutoConfigSource::Cli);
        let greedy_entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_GREEDY_ARGMAX")
            .unwrap_or_else(|| panic!("missing FERRUM_GREEDY_ARGMAX entry"));
        assert_eq!(greedy_entry.effective_value, "0");
    }

    #[test]
    fn cpu_serving_default_keeps_single_sequence_budget() {
        let hardware = HardwareCapabilities {
            backend: "cpu".to_string(),
            supported_dtypes: vec!["fp32".to_string()],
            ..HardwareCapabilities::unknown()
        };
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        assert_eq!(workload.target_concurrency, 1);
        let resolved = FerrumConfigBuilder::new(snapshot(&[]))
            .with_model_capabilities(ModelCapabilities::unknown())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .unwrap();
        let scheduler = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "scheduler_admission_policy")
            .unwrap();
        assert_eq!(scheduler.selected, "prompt_token_estimate");
        assert!(resolved
            .runtime_config
            .entries
            .iter()
            .all(|entry| entry.key != "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE"));
        assert!(resolved
            .runtime_config
            .entries
            .iter()
            .all(|entry| entry.key != "FERRUM_SCHED_PREFILL_STEP_CHUNK"));
    }

    #[test]
    fn validates_invalid_override_matrix() {
        expect_invalid_key(
            &[("FERRUM_USE_VLLM_PAGED_ATTN", "maybe")],
            "FERRUM_USE_VLLM_PAGED_ATTN",
        );
        expect_invalid_key(&[("FERRUM_PREFIX_CACHE", "maybe")], "FERRUM_PREFIX_CACHE");
        expect_invalid_key(
            &[
                ("FERRUM_FA_LAYOUT_VARLEN", "1"),
                ("FERRUM_USE_VLLM_PAGED_ATTN", "0"),
            ],
            "FERRUM_FA_LAYOUT_VARLEN",
        );
        expect_invalid_key(&[("FERRUM_FA2_DIRECT_FFI", "1")], "FERRUM_FA2_DIRECT_FFI");
        expect_invalid_key_with_features(
            &[("FERRUM_VLLM_MOE", "1")],
            "FERRUM_VLLM_MOE",
            CompiledKernelFeatures::default(),
        );
        expect_invalid_key(
            &[("FERRUM_MOE_DEVICE_ROUTE", "1"), ("FERRUM_VLLM_MOE", "0")],
            "FERRUM_MOE_DEVICE_ROUTE",
        );
        expect_invalid_key(
            &[("FERRUM_VLLM_MOE_PAIR_IDS", "1"), ("FERRUM_VLLM_MOE", "0")],
            "FERRUM_VLLM_MOE_PAIR_IDS",
        );
        expect_invalid_key(
            &[("FERRUM_MOE_GRAPH", "1"), ("FERRUM_VLLM_MOE", "0")],
            "FERRUM_MOE_GRAPH",
        );
        expect_invalid_key(&[("FERRUM_KV_MAX_BLOCKS", "0")], "FERRUM_KV_MAX_BLOCKS");
        expect_invalid_key(&[("FERRUM_PAGED_MAX_SEQS", "0")], "FERRUM_PAGED_MAX_SEQS");
        expect_invalid_key(
            &[
                ("FERRUM_PAGED_MAX_SEQS", "32"),
                ("FERRUM_MAX_BATCHED_TOKENS", "16"),
            ],
            "FERRUM_MAX_BATCHED_TOKENS",
        );
        expect_invalid_key(
            &[
                ("FERRUM_KV_MAX_BLOCKS", "16"),
                ("FERRUM_MAX_BATCHED_TOKENS", "512"),
            ],
            "FERRUM_MAX_BATCHED_TOKENS",
        );
        expect_invalid_key(&[("FERRUM_MAX_MODEL_LEN", "0")], "FERRUM_MAX_MODEL_LEN");
        expect_invalid_key(&[("FERRUM_MAX_MODEL_LEN", "50000")], "FERRUM_MAX_MODEL_LEN");
        expect_invalid_key(
            &[
                ("FERRUM_KV_MAX_BLOCKS", "16"),
                ("FERRUM_MAX_MODEL_LEN", "1024"),
            ],
            "FERRUM_KV_MAX_BLOCKS",
        );
        expect_invalid_key(&[("FERRUM_DTYPE", "bf16")], "FERRUM_DTYPE");
        expect_invalid_key(&[("FERRUM_KV_DTYPE", "fp8")], "FERRUM_KV_DTYPE");
        expect_invalid_key(
            &[
                ("FERRUM_VLLM_PAGED_ATTN_V1_SHORT", "1"),
                ("FERRUM_USE_VLLM_PAGED_ATTN", "0"),
            ],
            "FERRUM_VLLM_PAGED_ATTN_V1_SHORT",
        );
    }

    #[test]
    fn requested_max_model_len_is_optional_and_reflected_when_valid() {
        let default_resolved = m3(&[], CompiledKernelFeatures::m3_fast_path_without_fa2())
            .resolve()
            .unwrap();
        assert!(!default_resolved
            .decisions
            .iter()
            .any(|decision| decision.selection == "max_model_len"));

        let resolved = m3(
            &[
                ("FERRUM_KV_MAX_BLOCKS", "64"),
                ("FERRUM_MAX_MODEL_LEN", "1024"),
            ],
            CompiledKernelFeatures::m3_fast_path_without_fa2(),
        )
        .resolve()
        .unwrap();
        let max_model_len = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "max_model_len")
            .unwrap();
        assert_eq!(max_model_len.selected, "1024");
        assert_eq!(
            max_model_len.source_key.as_deref(),
            Some("FERRUM_MAX_MODEL_LEN")
        );
    }

    #[test]
    fn graph_enabled_with_graph_unsafe_moe_is_rejected() {
        let mut model = ModelCapabilities::qwen3_30b_a3b_gptq_int4();
        model.graph_safe_moe = false;
        let err = FerrumConfigBuilder::new(snapshot(&[("FERRUM_MOE_GRAPH", "1")]))
            .with_model_capabilities(model)
            .with_hardware_capabilities(HardwareCapabilities::rtx4090_cuda(
                CompiledKernelFeatures::m3_fast_path_without_fa2(),
            ))
            .with_workload_profile(WorkloadProfile::m3_qwen3_30b_a3b_int4())
            .resolve()
            .expect_err("graph unsafe MoE must fail");
        assert!(matches!(
            err,
            AutoConfigError::UnsupportedCombination {
                selection,
                ..
            } if selection == "moe_graph_policy"
        ));
    }

    #[test]
    fn batched_graph_override_materializes_decode_graph_policy() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot_with_sources(&[(
            "FERRUM_BATCHED_GRAPH",
            "1",
            RuntimeConfigSource::Cli,
        )]))
        .with_model_capabilities(ModelCapabilities::unknown())
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .unwrap();
        let decisions: BTreeMap<_, _> = resolved
            .decisions
            .iter()
            .map(|decision| (decision.selection.as_str(), decision.selected.as_str()))
            .collect();
        assert_eq!(
            decisions["decode_graph_policy"],
            "legacy_batched_decode_graph"
        );
        let entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_BATCHED_GRAPH")
            .expect("batched graph entry");
        assert_eq!(entry.effective_value, "1");
        assert_eq!(
            resolved.effective_config_document()["selected_graph_mode"],
            "legacy_batched_decode_graph"
        );
    }

    #[test]
    fn unified_graph_override_materializes_decode_graph_policy() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot_with_sources(&[(
            "FERRUM_UNIFIED_GRAPH",
            "1",
            RuntimeConfigSource::Cli,
        )]))
        .with_model_capabilities(ModelCapabilities::unknown())
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .unwrap();
        let decisions: BTreeMap<_, _> = resolved
            .decisions
            .iter()
            .map(|decision| (decision.selection.as_str(), decision.selected.as_str()))
            .collect();
        assert_eq!(decisions["decode_graph_policy"], "unified_decode_graph");
        let entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_UNIFIED_GRAPH")
            .expect("unified graph entry");
        assert_eq!(entry.effective_value, "1");
        assert_eq!(
            resolved.effective_config_document()["selected_graph_mode"],
            "unified_decode_graph"
        );
    }

    #[test]
    fn unified_graph_layers_only_override_materializes_decode_graph_policy() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot_with_sources(&[
            ("FERRUM_UNIFIED_GRAPH", "1", RuntimeConfigSource::Cli),
            (
                "FERRUM_UNIFIED_GRAPH_LAYERS_ONLY",
                "1",
                RuntimeConfigSource::Cli,
            ),
        ]))
        .with_model_capabilities(gemma3_gptq_model())
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .unwrap();
        let decisions: BTreeMap<_, _> = resolved
            .decisions
            .iter()
            .map(|decision| (decision.selection.as_str(), decision.selected.as_str()))
            .collect();
        assert_eq!(
            decisions["decode_graph_policy"],
            "unified_decode_graph_layers_only"
        );
        let entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_UNIFIED_GRAPH_LAYERS_ONLY")
            .expect("unified graph layers-only entry");
        assert_eq!(entry.effective_value, "1");
        assert_eq!(
            resolved.effective_config_document()["selected_graph_mode"],
            "unified_decode_graph_layers_only"
        );
    }

    #[test]
    fn unified_graph_lm_head_eager_override_materializes_decode_graph_policy() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot_with_sources(&[
            ("FERRUM_UNIFIED_GRAPH", "1", RuntimeConfigSource::Cli),
            (
                "FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER",
                "1",
                RuntimeConfigSource::Cli,
            ),
        ]))
        .with_model_capabilities(gemma3_gptq_model())
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .unwrap();
        let decisions: BTreeMap<_, _> = resolved
            .decisions
            .iter()
            .map(|decision| (decision.selection.as_str(), decision.selected.as_str()))
            .collect();
        assert_eq!(
            decisions["decode_graph_policy"],
            "unified_decode_graph_lm_head_eager"
        );
        let entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER")
            .expect("unified graph lm-head-eager entry");
        assert_eq!(entry.effective_value, "1");
        assert_eq!(
            resolved.effective_config_document()["selected_graph_mode"],
            "unified_decode_graph_lm_head_eager"
        );
    }

    #[test]
    fn unified_graph_layers_only_requires_unified_graph() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let err = FerrumConfigBuilder::new(snapshot(&[("FERRUM_UNIFIED_GRAPH_LAYERS_ONLY", "1")]))
            .with_model_capabilities(ModelCapabilities::unknown())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .expect_err("layers-only graph scope should require unified graph");
        assert!(matches!(
            err,
            AutoConfigError::InvalidOverride { key, .. }
                if key == "FERRUM_UNIFIED_GRAPH_LAYERS_ONLY"
        ));
    }

    #[test]
    fn unified_graph_lm_head_eager_requires_unified_graph() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let err =
            FerrumConfigBuilder::new(snapshot(&[("FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER", "1")]))
                .with_model_capabilities(ModelCapabilities::unknown())
                .with_hardware_capabilities(hardware)
                .with_workload_profile(workload)
                .resolve()
                .expect_err("lm-head-eager graph scope should require unified graph");
        assert!(matches!(
            err,
            AutoConfigError::InvalidOverride { key, .. }
                if key == "FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER"
        ));
    }

    #[test]
    fn unified_graph_lm_head_eager_conflicts_with_layers_only() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let err = FerrumConfigBuilder::new(snapshot(&[
            ("FERRUM_UNIFIED_GRAPH", "1"),
            ("FERRUM_UNIFIED_GRAPH_LAYERS_ONLY", "1"),
            ("FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER", "1"),
        ]))
        .with_model_capabilities(ModelCapabilities::unknown())
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .expect_err("lm-head-eager graph scope should conflict with layers-only graph scope");
        assert!(matches!(
            err,
            AutoConfigError::InvalidOverride { key, .. }
                if key == "FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER"
        ));
    }

    #[test]
    fn batched_graph_requires_cuda_graph_support() {
        let mut features = CompiledKernelFeatures::m3_fast_path_without_fa2();
        features.cuda_graph = false;
        let hardware = HardwareCapabilities::rtx4090_cuda(features);
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let err = FerrumConfigBuilder::new(snapshot(&[("FERRUM_BATCHED_GRAPH", "1")]))
            .with_model_capabilities(ModelCapabilities::unknown())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .expect_err("batched graph should require compiled graph support");
        assert!(matches!(
            err,
            AutoConfigError::InvalidOverride { key, .. } if key == "FERRUM_BATCHED_GRAPH"
        ));
    }

    #[test]
    fn unified_graph_requires_cuda_graph_support() {
        let mut features = CompiledKernelFeatures::m3_fast_path_without_fa2();
        features.cuda_graph = false;
        let hardware = HardwareCapabilities::rtx4090_cuda(features);
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let err = FerrumConfigBuilder::new(snapshot(&[("FERRUM_UNIFIED_GRAPH", "1")]))
            .with_model_capabilities(ModelCapabilities::unknown())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .expect_err("unified graph should require compiled graph support");
        assert!(matches!(
            err,
            AutoConfigError::InvalidOverride { key, .. } if key == "FERRUM_UNIFIED_GRAPH"
        ));
    }

    #[test]
    fn unified_graph_rejects_gemma3_sandwich_models() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let err = FerrumConfigBuilder::new(snapshot(&[("FERRUM_UNIFIED_GRAPH", "1")]))
            .with_model_capabilities(gemma3_gptq_model())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .expect_err("Gemma3 unified graph should stay disabled until graph instantiate fits");
        assert!(matches!(
            err,
            AutoConfigError::InvalidOverride { key, .. } if key == "FERRUM_UNIFIED_GRAPH"
        ));
    }

    #[test]
    fn batched_graph_rejects_non_cuda_backend() {
        let hardware =
            cpu_hardware_with_features(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let err = FerrumConfigBuilder::new(snapshot(&[("FERRUM_BATCHED_GRAPH", "1")]))
            .with_model_capabilities(ModelCapabilities::unknown())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .expect_err("batched graph should require CUDA");
        assert!(matches!(
            err,
            AutoConfigError::InvalidOverride { key, .. } if key == "FERRUM_BATCHED_GRAPH"
        ));
    }

    #[test]
    fn scheduler_active_chunk_combines_with_accelerator_prefill_first_default() {
        let resolved = m3(
            &[("FERRUM_ACTIVE_DECODE_PREFILL_CHUNK", "64")],
            CompiledKernelFeatures::m3_fast_path_without_fa2(),
        )
        .resolve()
        .unwrap();
        let scheduler = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "scheduler_admission_policy")
            .unwrap();
        assert_eq!(
            scheduler.selected,
            "prefill_first_until_active:32+active_decode_prefill_chunk:64+prefill_step_chunk:64"
        );
        assert_eq!(
            scheduler.source_key.as_deref(),
            Some("FERRUM_ACTIVE_DECODE_PREFILL_CHUNK")
        );
        let prefill_entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE")
            .expect("accelerator default prefill-first should still be materialized");
        assert_eq!(prefill_entry.effective_value, "32");
        assert_eq!(prefill_entry.source, RuntimeConfigSource::Default);
        let step_entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_SCHED_PREFILL_STEP_CHUNK")
            .expect("accelerator default prefill-step chunk should be materialized");
        assert_eq!(step_entry.effective_value, "64");
        assert_eq!(step_entry.source, RuntimeConfigSource::Default);
    }

    #[test]
    fn scheduler_prefill_step_chunk_override_is_reflected_in_decision_trace() {
        let resolved = m3(
            &[("FERRUM_SCHED_PREFILL_STEP_CHUNK", "128")],
            CompiledKernelFeatures::m3_fast_path_without_fa2(),
        )
        .resolve()
        .unwrap();
        let scheduler = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "scheduler_admission_policy")
            .unwrap();
        assert_eq!(
            scheduler.selected,
            "prefill_first_until_active:32+prefill_step_chunk:128"
        );
        assert_eq!(scheduler.source, AutoConfigSource::Env);
        assert_eq!(
            scheduler.source_key.as_deref(),
            Some("FERRUM_SCHED_PREFILL_STEP_CHUNK")
        );
        let step_entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_SCHED_PREFILL_STEP_CHUNK")
            .expect("explicit prefill-step chunk should be preserved");
        assert_eq!(step_entry.effective_value, "128");
        assert_eq!(step_entry.source, RuntimeConfigSource::Env);
    }

    #[test]
    fn cuda_gemma3_gptq_uses_generic_scheduler_default() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot(&[]))
            .with_model_capabilities(gemma3_gptq_model())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .unwrap();
        let scheduler = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "scheduler_admission_policy")
            .unwrap();
        assert_eq!(
            scheduler.selected,
            "prefill_first_until_active:32+prefill_step_chunk:64"
        );
        assert_eq!(scheduler.source, AutoConfigSource::Default);
        assert!(resolved
            .runtime_config
            .entries
            .iter()
            .all(|entry| entry.key != "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK"));
        let entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE")
            .expect("generic scheduler default should be materialized");
        assert_eq!(entry.effective_value, "32");
        assert_eq!(entry.source, RuntimeConfigSource::Default);
    }

    #[test]
    fn explicit_scheduler_prompt_estimate_is_reflected_for_gemma3() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved =
            FerrumConfigBuilder::new(snapshot(&[("FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE", "1")]))
                .with_model_capabilities(gemma3_gptq_model())
                .with_hardware_capabilities(hardware)
                .with_workload_profile(workload)
                .resolve()
                .unwrap();
        let scheduler = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "scheduler_admission_policy")
            .unwrap();
        assert_eq!(scheduler.selected, "prompt_token_estimate");
        assert_eq!(
            scheduler.source_key.as_deref(),
            Some("FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE")
        );
        assert!(resolved
            .runtime_config
            .entries
            .iter()
            .all(|entry| entry.key != "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK"));
    }

    #[test]
    fn scheduler_prefill_first_is_default_accelerator_policy() {
        let resolved = m3(&[], CompiledKernelFeatures::m3_fast_path_without_fa2())
            .resolve()
            .unwrap();
        let scheduler = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "scheduler_admission_policy")
            .unwrap();
        assert_eq!(
            scheduler.selected,
            "prefill_first_until_active:32+prefill_step_chunk:64"
        );
        assert_eq!(scheduler.source, AutoConfigSource::Default);
        assert_eq!(scheduler.source_key, None);
        let entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE")
            .expect("generic scheduler default should be materialized");
        assert_eq!(entry.effective_value, "32");
        assert_eq!(entry.source, RuntimeConfigSource::Default);
        let step_entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_SCHED_PREFILL_STEP_CHUNK")
            .expect("generic scheduler chunk default should be materialized");
        assert_eq!(step_entry.effective_value, "64");
        assert_eq!(step_entry.source, RuntimeConfigSource::Default);
    }

    #[test]
    fn scheduler_prompt_token_estimate_can_be_disabled_in_decision_trace() {
        let resolved = m3(
            &[("FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE", "0")],
            CompiledKernelFeatures::m3_fast_path_without_fa2(),
        )
        .resolve()
        .unwrap();
        let scheduler = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "scheduler_admission_policy")
            .unwrap();
        assert_eq!(scheduler.selected, "continuous_default");
        assert_eq!(scheduler.source, AutoConfigSource::Env);
        assert_eq!(
            scheduler.source_key.as_deref(),
            Some("FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE")
        );
    }

    #[test]
    fn prefix_cache_override_is_reflected_in_decision_trace() {
        let resolved = m3(
            &[("FERRUM_PREFIX_CACHE", "1")],
            CompiledKernelFeatures::m3_fast_path_without_fa2(),
        )
        .resolve()
        .unwrap();
        let prefix_cache = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "prefix_cache_policy")
            .unwrap();
        assert_eq!(prefix_cache.selected, "prefix_cache_enabled");
        assert_eq!(
            prefix_cache.source_key.as_deref(),
            Some("FERRUM_PREFIX_CACHE")
        );
    }

    #[test]
    fn non_env_runtime_sources_are_preserved_in_decision_trace() {
        let runtime_config = snapshot_with_sources(&[
            (
                "FERRUM_FA_LAYOUT_VARLEN",
                "1",
                RuntimeConfigSource::ConfigFile,
            ),
            ("FERRUM_PAGED_MAX_SEQS", "48", RuntimeConfigSource::Cli),
            (
                "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE",
                "32",
                RuntimeConfigSource::ScriptCase,
            ),
        ]);
        let resolved = FerrumConfigBuilder::new(runtime_config)
            .with_model_capabilities(ModelCapabilities::qwen3_30b_a3b_gptq_int4())
            .with_hardware_capabilities(HardwareCapabilities::rtx4090_cuda(
                CompiledKernelFeatures::m3_fast_path_without_fa2(),
            ))
            .with_workload_profile(WorkloadProfile::m3_qwen3_30b_a3b_int4())
            .resolve()
            .unwrap();

        let decision = |selection: &str| {
            resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == selection)
                .unwrap()
        };
        let attention = decision("attention_prefill_mixed_backend");
        assert_eq!(attention.selected, "fa_layout_varlen");
        assert_eq!(attention.source, AutoConfigSource::ConfigFile);
        assert_eq!(
            attention.source_key.as_deref(),
            Some("FERRUM_FA_LAYOUT_VARLEN")
        );

        let max_sequences = decision("max_sequences");
        assert_eq!(max_sequences.selected, "48");
        assert_eq!(max_sequences.source, AutoConfigSource::Cli);
        assert_eq!(
            max_sequences.source_key.as_deref(),
            Some("FERRUM_PAGED_MAX_SEQS")
        );

        let scheduler = decision("scheduler_admission_policy");
        assert_eq!(
            scheduler.selected,
            "prefill_first_until_active:32+prefill_step_chunk:64"
        );
        assert_eq!(scheduler.source, AutoConfigSource::ScriptCase);
        assert_eq!(
            scheduler.source_key.as_deref(),
            Some("FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE")
        );
    }

    #[test]
    fn renders_effective_config_and_decision_trace_artifacts() {
        let resolved = m3(&[], CompiledKernelFeatures::m3_fast_path_without_fa2())
            .resolve()
            .unwrap();
        let effective = resolved.effective_config_document();
        assert_eq!(effective["schema_version"], 1);
        assert!(effective["env_hash"]
            .as_str()
            .unwrap()
            .starts_with("sha256:"));
        assert!(effective["entries"].is_array());
        assert_eq!(effective["model_capabilities"]["architecture"], "qwen3_moe");
        assert_eq!(effective["hardware_capabilities"]["backend"], "cuda");
        assert_eq!(
            effective["workload_profile"]["preset"],
            M3_QWEN3_30B_A3B_INT4_PRESET
        );
        assert_eq!(
            effective["decisions"].as_array().unwrap().len(),
            resolved.decisions.len()
        );
        let trace = resolved.decision_trace_jsonl().unwrap();
        assert_eq!(trace.lines().count(), resolved.decisions.len());
        assert!(trace.contains("\"attention_prefill_mixed_backend\""));
    }

    #[test]
    fn auto_config_artifacts_match_locked_schema_shape() {
        let resolved = FerrumConfigBuilder::m3_qwen3_30b_a3b_int4(snapshot_with_sources(&[
            (
                "FERRUM_FA_LAYOUT_VARLEN",
                "1",
                RuntimeConfigSource::ScriptCase,
            ),
            ("FERRUM_PAGED_MAX_SEQS", "32", RuntimeConfigSource::Cli),
        ]))
        .resolve()
        .unwrap();

        let effective = resolved.effective_config_document();
        assert_eq!(effective["schema_version"], 1);
        assert!(effective["env_hash"]
            .as_str()
            .unwrap()
            .starts_with("sha256:"));

        let entries = effective["entries"].as_array().unwrap();
        let keys: Vec<_> = entries
            .iter()
            .map(|entry| entry["key"].as_str().unwrap())
            .collect();
        let mut sorted_keys = keys.clone();
        sorted_keys.sort_unstable();
        assert_eq!(keys, sorted_keys);
        for entry in entries {
            assert!(entry["key"].as_str().unwrap().starts_with("FERRUM_"));
            assert!(entry["effective_value"].is_string());
            assert!(matches!(
                entry["source"].as_str().unwrap(),
                "default" | "config_file" | "cli" | "env" | "script_case" | "memory_profile"
            ));
            assert!(!entry["affects"].as_array().unwrap().is_empty());
        }
        assert_eq!(
            effective["model_capabilities"]["quantization"].as_str(),
            Some("gptq_int4")
        );
        assert_eq!(
            effective["model_capabilities"]["moe"]["experts_per_token"].as_u64(),
            Some(8)
        );
        assert_eq!(
            effective["hardware_capabilities"]["compute_capability"].as_str(),
            Some("8.9")
        );
        assert_eq!(
            effective["hardware_capabilities"]["compiled_features"]["vllm_moe_marlin"].as_bool(),
            Some(true)
        );
        assert_eq!(
            effective["workload_profile"]["target_concurrency"].as_u64(),
            Some(32)
        );
        assert_eq!(
            effective["workload_profile"]["priority"].as_str(),
            Some("throughput")
        );
        let admission = &effective["admission"];
        for field in [
            "effective_max_concurrent",
            "queue_depth",
            "active_prefill",
            "active_decode",
            "current_batch_size",
            "rejected_requests_total",
            "failed_requests_total",
            "completed_requests_total",
        ] {
            assert!(admission[field].is_number(), "admission.{field} missing");
        }

        let trace = resolved.decision_trace_jsonl().unwrap();
        let trace_decisions: Vec<AutoConfigDecision> = trace
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(trace_decisions, resolved.decisions);
        assert_eq!(
            serde_json::from_value::<Vec<AutoConfigDecision>>(effective["decisions"].clone())
                .unwrap(),
            trace_decisions
        );

        for decision in &trace_decisions {
            assert_eq!(decision.schema_version, 1);
            assert!(!decision.selection.trim().is_empty());
            assert!(!decision.selected.trim().is_empty());
            assert!(!decision.candidates.is_empty());
            assert!(!decision.affects.is_empty());
            if let Some(source_key) = &decision.source_key {
                assert!(source_key.starts_with("FERRUM_"));
            }
            for rejected in &decision.rejected {
                assert!(!rejected.value.trim().is_empty());
                assert!(!rejected.reason.trim().is_empty());
            }
        }
    }
}
