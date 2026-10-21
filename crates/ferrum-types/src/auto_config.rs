//! Startup auto-configuration and selector decision trace types.
//!
//! This is the typed control-plane surface for gradually replacing M3 shell
//! env bundles with validated model/hardware/workload driven selections.

use crate::{
    is_sha256_digest, parse_bool_env_value, parse_usize_env_value, AttentionExecutionPolicy,
    CompiledNativeOperatorIdentity, ExecutionPlanningModel, RuntimeConfigEffect,
    RuntimeConfigEntry, RuntimeConfigSnapshot, RuntimeConfigSource,
    CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

mod execution_defaults;
#[cfg(test)]
mod legacy_capacity_tests;

pub const M3_QWEN3_30B_A3B_INT4_PRESET: &str = "m3_qwen3_30b_a3b_int4";
pub const QWEN25_72B_GPTQ_INT4_2X4090_LAYER_SPLIT_PRESET: &str =
    "qwen25_72b_gptq_int4_2x4090_layer_split";
const DEFAULT_KV_BLOCK_SIZE_TOKENS: usize = 16;
const DEFAULT_KV_BLOCKS: usize = 2048;
const GIB: u64 = 1024 * 1024 * 1024;
const FA2_NATIVE_MANIFEST_KEY: &str = "FERRUM_FA2_NATIVE_MANIFEST";
const FA2_NATIVE_ARTIFACT_KEY: &str = "FERRUM_FA2_NATIVE_ARTIFACT";
const FA2_NATIVE_SOURCE_SHA256_KEY: &str = "FERRUM_FA2_NATIVE_SOURCE_SHA256";
const FA2_NATIVE_INPUTS_SHA256_KEY: &str = "FERRUM_FA2_NATIVE_INPUTS_SHA256";

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
pub struct CompiledNativeOperatorArtifact {
    pub manifest_path: String,
    pub artifact_path: String,
    pub source_package_sha256: String,
    pub inputs_sha256: String,
    pub binary_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledKernelFeatures {
    pub cuda: bool,
    pub vllm_paged_attn: bool,
    #[serde(default)]
    pub cuda_indexed_causal_attention: bool,
    #[serde(default)]
    pub cuda_causal_gqa2: bool,
    #[serde(default)]
    pub cuda_q4_shared_weight_decode: bool,
    #[serde(default)]
    pub cuda_q5_shared_weight_decode: bool,
    pub vllm_moe_marlin: bool,
    pub cuda_graph: bool,
    pub greedy_argmax: bool,
    pub fa2_source: bool,
    pub fa2_direct_ffi: bool,
    pub fa2_native_operator_artifact: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fa2_native_operator_artifact_metadata: Option<CompiledNativeOperatorArtifact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub native_operator_artifacts: Vec<CompiledNativeOperatorIdentity>,
}

impl Default for CompiledKernelFeatures {
    fn default() -> Self {
        Self {
            cuda: false,
            vllm_paged_attn: false,
            cuda_indexed_causal_attention: false,
            cuda_causal_gqa2: false,
            cuda_q4_shared_weight_decode: false,
            cuda_q5_shared_weight_decode: false,
            vllm_moe_marlin: false,
            cuda_graph: false,
            greedy_argmax: false,
            fa2_source: false,
            fa2_direct_ffi: false,
            fa2_native_operator_artifact: false,
            fa2_native_operator_artifact_metadata: None,
            native_operator_artifacts: Vec::new(),
        }
    }
}

impl CompiledKernelFeatures {
    pub fn m3_fast_path_without_fa2() -> Self {
        Self {
            cuda: true,
            vllm_paged_attn: true,
            cuda_indexed_causal_attention: false,
            cuda_causal_gqa2: false,
            cuda_q4_shared_weight_decode: false,
            cuda_q5_shared_weight_decode: false,
            vllm_moe_marlin: true,
            cuda_graph: true,
            greedy_argmax: true,
            fa2_source: false,
            fa2_direct_ffi: false,
            fa2_native_operator_artifact: false,
            fa2_native_operator_artifact_metadata: None,
            native_operator_artifacts: Vec::new(),
        }
    }

    pub fn m3_fast_path_with_source_fa2() -> Self {
        Self {
            fa2_source: true,
            ..Self::m3_fast_path_without_fa2()
        }
    }

    pub fn m3_fast_path_with_native_fa2_artifact() -> Self {
        Self {
            fa2_native_operator_artifact: true,
            fa2_native_operator_artifact_metadata: Some(CompiledNativeOperatorArtifact {
                manifest_path: "/tmp/native/fa2/native_operator_manifest.json".to_string(),
                artifact_path: "/tmp/native/fa2/libferrum_native_fa2.a".to_string(),
                source_package_sha256:
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
                inputs_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                    .to_string(),
                binary_sha256: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
                    .to_string(),
            }),
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

/// Cold selection of the KV allocation layout used by the retained model.
/// This is supplied by the model/backend selector, never inferred from the
/// planning-model tag. Unknown does not authorize a block-to-token limit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SelectedKvCacheLayout {
    #[default]
    Unknown,
    Contiguous,
    Paged {
        block_size_tokens: usize,
    },
}

impl SelectedKvCacheLayout {
    fn block_size_tokens(self) -> Option<usize> {
        match self {
            Self::Paged { block_size_tokens } => Some(block_size_tokens),
            Self::Unknown | Self::Contiguous => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedFerrumConfig {
    pub schema_version: u32,
    pub preset: Option<String>,
    // Retain the historical observation field for stored configs/reports.
    #[serde(default, rename = "execution_resource_authority")]
    pub execution_planning_model: ExecutionPlanningModel,
    #[serde(default)]
    pub selected_kv_cache_layout: SelectedKvCacheLayout,
    /// Actual retained allocator storage, independently of the compatibility dtype option.
    #[serde(default)]
    pub selected_kv_cache_storage: Option<crate::KvCacheStorage>,
    #[serde(default)]
    pub requested_attention_policy: Option<AttentionExecutionPolicy>,
    #[serde(default)]
    pub compiled_attention_policy: Option<AttentionExecutionPolicy>,
    pub runtime_config: RuntimeConfigSnapshot,
    pub model_capabilities: ModelCapabilities,
    pub hardware_capabilities: HardwareCapabilities,
    pub workload_profile: WorkloadProfile,
    pub decisions: Vec<AutoConfigDecision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub startup_memory_plan: Option<crate::StartupMemoryPlan>,
}

impl ResolvedFerrumConfig {
    /// Publish the same selected limits that the compiled executor and engine
    /// use. Preserve explicit provenance; only inferred values become memory
    /// profile decisions.
    pub fn apply_startup_memory_plan(&mut self, plan: &crate::StartupMemoryPlan) {
        for (key, selection, value) in [
            (
                "FERRUM_MAX_MODEL_LEN",
                "max_model_len",
                plan.selected.context_tokens as u64,
            ),
            (
                "FERRUM_PAGED_MAX_SEQS",
                "max_sequences",
                plan.selected.max_sequences as u64,
            ),
            (
                "FERRUM_MAX_BATCHED_TOKENS",
                "max_batched_tokens",
                plan.selected.max_batch_tokens as u64,
            ),
            (
                "FERRUM_RUNTIME_MEMORY_BUDGET_BYTES",
                "runtime_memory_budget_bytes",
                plan.request.usable_capacity_bytes,
            ),
        ] {
            let source = self
                .runtime_config
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .map(|entry| entry.source)
                .filter(|source| {
                    matches!(
                        source,
                        RuntimeConfigSource::Cli
                            | RuntimeConfigSource::Env
                            | RuntimeConfigSource::ConfigFile
                            | RuntimeConfigSource::ScriptCase
                    )
                })
                .unwrap_or(RuntimeConfigSource::MemoryProfile);
            self.runtime_config.upsert(key, value.to_string(), source);
            self.decisions
                .retain(|decision| decision.selection != selection);
            self.decisions.push(AutoConfigDecision {
                schema_version: 1,
                selection: selection.to_owned(),
                selected: value.to_string(),
                source: auto_config_source_from_runtime(source),
                source_key: Some(key.to_owned()),
                candidates: vec![value.to_string()],
                rejected: Vec::new(),
                affects: vec![
                    RuntimeConfigEffect::Memory,
                    RuntimeConfigEffect::Performance,
                ],
            });
        }
        self.startup_memory_plan = Some(plan.clone());
    }

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
        // This is the declared protocol ceiling, not a model-state capacity estimate.
        let selected_admission_limit = selected_max_sequences;
        let selected_attention_policy = self.selected_string("attention_execution_policy");
        let selected_attention_impl = self.selected_string("attention_decode_backend");
        serde_json::json!({
            "schema_version": 1,
            "preset": self.preset,
            "env_hash": self.runtime_env_hash(),
            "backend": backend.clone(),
            "execution_resource_authority": self.execution_planning_model,
            "startup_memory_plan": self.startup_memory_plan,
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
            "selected_kv_layout": match self.selected_kv_cache_layout {
                SelectedKvCacheLayout::Unknown => "unknown",
                SelectedKvCacheLayout::Contiguous => "contiguous",
                SelectedKvCacheLayout::Paged { .. } => "paged",
            },
            "selected_kv_cache_storage": self.selected_kv_cache_storage,
            "selected_attention_impl": selected_attention_impl,
            "selected_attention_policy": selected_attention_policy,
            "attention_execution": self.attention_execution_document(),
            "selected_graph_mode": self.selected_graph_mode(),
            "selected_max_sequences": selected_max_sequences,
            "selected_max_model_len": selected_max_model_len,
            "selected_kv_capacity": selected_kv_capacity,
            "selected_kv_capacity_source": "configured_token_limit",
            "selected_max_batched_tokens": selected_max_batched_tokens,
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
        let effective_max_concurrent = max_sequences;
        let kv_blocks = self.selected_usize("kv_block_count");
        let max_batched_tokens = self.selected_usize("max_batched_tokens");
        let max_model_len = self.selected_usize("max_model_len");
        let block_size = self.selected_kv_cache_layout.block_size_tokens();
        // Preserve the explicitly labelled geometry estimate for old reports;
        // applicability is true only when the actual selector supplied Paged.
        let kv_capacity_tokens = kv_blocks.and_then(|blocks| {
            blocks.checked_mul(block_size.unwrap_or(DEFAULT_KV_BLOCK_SIZE_TOKENS))
        });
        let selected_storage = (self.execution_planning_model
            == ExecutionPlanningModel::ExecutorManaged)
            .then_some(self.selected_kv_cache_storage)
            .flatten();
        let kv_bytes_per_token = selected_storage.and_then(|storage| {
            storage.bytes_per_token(
                self.model_capabilities.num_hidden_layers?,
                self.model_capabilities.kv_heads?,
                self.model_capabilities.head_dim?,
            )
        });
        let scheduler_policy = self
            .selected_string("scheduler_admission_policy")
            .unwrap_or_else(|| "unknown".to_string());
        let memory_estimate = serde_json::json!({
            "source": if selected_storage.is_some() { "selected_storage_geometry" } else { "unqualified_storage_geometry" },
            "applies_to_selected_state_layout": kv_bytes_per_token.is_some() && block_size.is_some(),
            "is_resident_usage": false,
            "kv_storage": selected_storage,
            "excludes": ["allocator_overhead", "metadata", "workspace", "weights"],
            "selected_state_evidence_source": if self.execution_planning_model == ExecutionPlanningModel::CompiledPlan { "executor.kv_storage.logical_sequence_state" } else { "legacy_runtime" },
            "vram_bytes": self.hardware_capabilities.vram_bytes,
            "estimated_weight_bytes": self.model_capabilities.estimated_weight_bytes,
            "estimated_weight_bytes_basis": "model_metadata_or_source_file_size_not_resident_storage",
            "kv_bytes_per_token": kv_bytes_per_token,
            "recurrent_state_bytes_per_sequence": self.model_capabilities.recurrent_state_bytes_per_sequence,
            "kv_capacity_bytes": match (block_size, kv_capacity_tokens, kv_bytes_per_token) {
                (Some(_), Some(tokens), Some(bytes_per_token)) => {
                    (tokens as u64).checked_mul(bytes_per_token)
                }
                _ => None,
            },
        });
        serde_json::json!({
            "schema_version": 1,
            "backend": self.hardware_capabilities.backend,
            "model_architecture": self.model_capabilities.architecture,
            "resource_authority": self.execution_planning_model,
            "source": "startup_preflight",
            "scheduler_policy": scheduler_policy,
            "effective_max_concurrent": effective_max_concurrent,
            "maximum_active_sequences": max_sequences,
            "maximum_scheduled_tokens": max_batched_tokens,
            "queue_depth": 0u64,
            "active_prefill": 0u64,
            "active_decode": 0u64,
            "current_batch_size": 0u64,
            "rejected_requests_total": 0u64,
            "failed_requests_total": 0u64,
            "completed_requests_total": 0u64,
            "max_sequences": max_sequences,
            "kv_capacity_source": "legacy_block_preflight_estimate",
            "selected_kv_cache_layout": self.selected_kv_cache_layout,
            "kv_capacity_applies_to_selected_state_layout": block_size.is_some(),
            "kv_capacity_is_resident_usage": false,
            "kv_block_count": kv_blocks,
            "kv_block_size_tokens": block_size.unwrap_or(DEFAULT_KV_BLOCK_SIZE_TOKENS),
            "kv_capacity_tokens": kv_capacity_tokens,
            "max_model_length": max_model_len,
            "max_batched_tokens": max_batched_tokens,
            "memory_estimate": memory_estimate,
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

    fn selected_graph_mode(&self) -> Option<String> {
        self.selected_string("decode_graph_policy")
    }

    fn attention_execution_document(&self) -> serde_json::Value {
        let compiled_policy = self
            .compiled_attention_policy
            .map(AttentionExecutionPolicy::as_runtime_value);
        let requested_policy = self
            .requested_attention_policy
            .map(AttentionExecutionPolicy::as_runtime_value);
        if self.execution_planning_model != ExecutionPlanningModel::CompiledPlan {
            return serde_json::json!({
                "schema_version": 1,
                "resource_authority": self.execution_planning_model,
                "requested_policy": null,
                "compiled_policy": null,
                "selection_scope": "legacy_runtime",
                "observed_variant_source": "legacy_backend_profile",
                "legacy_attention_keys_apply": true,
            });
        }
        let native_adaptive =
            compiled_policy == Some(AttentionExecutionPolicy::NativeAdaptive.as_runtime_value());
        serde_json::json!({
            "schema_version": 1,
            "resource_authority": self.execution_planning_model,
            "requested_policy": requested_policy,
            "compiled_policy": compiled_policy,
            "selection_scope": "per_invocation",
            "observed_variant_source": "vnext.device_native_work",
            "legacy_attention_keys_apply": false,
            "decode": if native_adaptive && self.hardware_capabilities.backend.eq_ignore_ascii_case("cuda") {
                serde_json::json!({
                    "selector": "sequence_frontier",
                    "short_variant": "vllm_paged_attention_v1_addressed",
                    "long_variant": "vllm_paged_attention_v2_addressed",
                    "short_max_sequence_tokens": CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS,
                })
            } else {
                serde_json::json!({
                    "selector": "compiled_provider",
                })
            },
            "prefill": {
                "selector": "admitted_shape",
                "variants_are_observed_at_runtime": true,
            },
        })
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

#[derive(Clone)]
pub struct FerrumConfigBuilder {
    runtime_config: RuntimeConfigSnapshot,
    model: ModelCapabilities,
    hardware: HardwareCapabilities,
    workload: WorkloadProfile,
    execution_planning_model: ExecutionPlanningModel,
    selected_kv_cache_layout: SelectedKvCacheLayout,
    selected_kv_cache_storage: Option<crate::KvCacheStorage>,
}

impl FerrumConfigBuilder {
    pub fn new(runtime_config: RuntimeConfigSnapshot) -> Self {
        Self {
            runtime_config,
            model: ModelCapabilities::unknown(),
            hardware: HardwareCapabilities::unknown(),
            workload: WorkloadProfile::default(),
            execution_planning_model: ExecutionPlanningModel::ExecutorManaged,
            selected_kv_cache_layout: SelectedKvCacheLayout::Unknown,
            selected_kv_cache_storage: None,
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

    pub fn with_execution_planning_model(mut self, authority: ExecutionPlanningModel) -> Self {
        self.execution_planning_model = authority;
        self
    }

    /// The caller must use the same cold selector as its actual model allocator.
    /// Compiled-runtime layouts are admitted by their plan and remain Unknown here.
    pub fn with_selected_kv_cache_layout(mut self, layout: SelectedKvCacheLayout) -> Self {
        self.selected_kv_cache_layout = layout;
        self
    }

    pub fn with_selected_kv_cache_storage(
        mut self,
        storage: Option<crate::KvCacheStorage>,
    ) -> Self {
        self.selected_kv_cache_storage = storage;
        self
    }

    pub fn resolve(self) -> Result<ResolvedFerrumConfig, AutoConfigError> {
        if let Err((key, reason)) = self.runtime_config.validate_retired_options() {
            return self.invalid(key, reason);
        }
        if self.selected_kv_cache_layout.block_size_tokens() == Some(0) {
            return self.invalid(
                "selected_kv_cache_layout",
                "paged block size must be positive",
            );
        }
        if self.execution_planning_model == ExecutionPlanningModel::CompiledPlan
            && (self.selected_kv_cache_layout != SelectedKvCacheLayout::Unknown
                || self.selected_kv_cache_storage.is_some())
        {
            return self.invalid(
                "selected_kv_cache_layout",
                "compiled runtime capacity and storage belong to its actual plan",
            );
        }
        let mut decisions = Vec::new();
        let cuda_backend = self.is_cuda_backend();
        let plan_runtime = self.execution_planning_model == ExecutionPlanningModel::CompiledPlan;
        if plan_runtime {
            for key in [
                "FERRUM_USE_VLLM_PAGED_ATTN",
                "FERRUM_VLLM_PAGED_ATTN_V1_SHORT",
            ] {
                if self.entry(key).is_some() {
                    return Err(AutoConfigError::InvalidOverride {
                        key: key.to_owned(),
                        reason: format!(
                            "legacy attention control does not apply to CompiledPlan; use FERRUM_ATTENTION_POLICY=auto|portable|native-adaptive"
                        ),
                    });
                }
            }
        }
        let plan_attention = plan_runtime
            .then(|| self.resolve_plan_attention_policy())
            .transpose()?;
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
        let fa2_native_manifest = self.optional_string_value(FA2_NATIVE_MANIFEST_KEY)?;
        let fa2_native_artifact = self.optional_string_value(FA2_NATIVE_ARTIFACT_KEY)?;
        let fa2_native_source_sha256 = self.optional_string_value(FA2_NATIVE_SOURCE_SHA256_KEY)?;
        let fa2_native_inputs_sha256 = self.optional_string_value(FA2_NATIVE_INPUTS_SHA256_KEY)?;
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
        let pair_ids = self.bool_value(
            "FERRUM_VLLM_MOE_PAIR_IDS",
            vllm_moe.value,
            AutoConfigSource::WorkloadPreset,
        )?;
        let batched_graph =
            self.bool_value("FERRUM_BATCHED_GRAPH", false, AutoConfigSource::Default)?;
        let reusable_execution = self.bool_value(
            "FERRUM_REUSABLE_EXECUTION",
            cuda_backend
                && self.hardware.graph_support
                && self.hardware.compiled_features.cuda_graph,
            AutoConfigSource::HardwareCapability,
        )?;
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
        let mut max_sequences = self.usize_value(
            "FERRUM_PAGED_MAX_SEQS",
            default_max_sequences.value,
            default_max_sequences.source,
        )?;
        if plan_runtime
            && !matches!(
                max_sequences.source,
                AutoConfigSource::Cli
                    | AutoConfigSource::Env
                    | AutoConfigSource::ConfigFile
                    | AutoConfigSource::ScriptCase
            )
        {
            if let Some(batch) = self.optional_usize_value("FERRUM_MAX_BATCHED_TOKENS")? {
                // A configured batch bounds automatic decode width. Explicit
                // concurrency still reaches validation unchanged.
                max_sequences.value = max_sequences.value.min(batch.value.max(1));
            }
        }
        let max_model_len = self
            .optional_usize_value("FERRUM_MAX_MODEL_LEN")?
            .or_else(|| {
                plan_runtime
                    .then_some(self.model.max_context_len)
                    .flatten()
                    .map(|value| ResolvedValue {
                        value,
                        source: AutoConfigSource::ModelMetadata,
                        source_key: None,
                    })
            });
        let legacy_kv_capacity =
            if !plan_runtime && self.selected_kv_cache_layout != SelectedKvCacheLayout::Unknown {
                Some(self.resolve_legacy_kv_capacity(max_model_len.as_ref())?)
            } else {
                None
            };
        let default_kv_blocks =
            self.default_kv_blocks(&max_sequences, legacy_kv_capacity.as_ref())?;
        let kv_blocks = self.usize_value(
            "FERRUM_KV_MAX_BLOCKS",
            default_kv_blocks.value,
            default_kv_blocks.source,
        )?;
        let default_max_batched_tokens =
            self.default_max_batched_tokens(&max_sequences, &kv_blocks);
        let mut max_batched_tokens = self.usize_value(
            "FERRUM_MAX_BATCHED_TOKENS",
            default_max_batched_tokens.value,
            default_max_batched_tokens.source,
        )?;
        let explicit_batch = self
            .entry("FERRUM_MAX_BATCHED_TOKENS")
            .is_some_and(|entry| {
                matches!(
                    entry.source,
                    RuntimeConfigSource::Cli
                        | RuntimeConfigSource::Env
                        | RuntimeConfigSource::ConfigFile
                        | RuntimeConfigSource::ScriptCase
                )
            });
        if !plan_runtime && !explicit_batch {
            if let Some(block_size) = self.selected_kv_cache_layout.block_size_tokens() {
                let capacity = kv_blocks.value.checked_mul(block_size).ok_or_else(|| {
                    AutoConfigError::InvalidOverride {
                        key: "FERRUM_KV_MAX_BLOCKS".into(),
                        reason: "physical KV token capacity overflows".into(),
                    }
                })?;
                // Autosizing may have supplied a generic batch before the actual
                // logical/physical pool was selected. Its inferred value must fit
                // that pool, while explicit requests retain strict validation.
                max_batched_tokens.value = max_batched_tokens
                    .value
                    .min(capacity)
                    .max(max_sequences.value.max(1));
            }
        }
        let default_prefill_first_until_active =
            self.default_prefill_first_until_active(&max_sequences);
        let default_active_decode_prefill_chunk =
            self.default_active_decode_prefill_chunk(&max_sequences);
        if !plan_runtime {
            self.validate_attention(
                use_vllm_paged_attn.value,
                fa_layout.value,
                fa2_source.value,
                fa2_direct_ffi.value,
                shim_present,
                vllm_v1_short.value,
            )?;
            self.validate_fa2_native_artifact(
                fa2_native_manifest.as_ref(),
                fa2_native_artifact.as_ref(),
                fa2_native_source_sha256.as_ref(),
                fa2_native_inputs_sha256.as_ref(),
                fa2_source.value,
                fa2_direct_ffi.value,
            )?;
        }
        self.validate_moe(vllm_moe.value, pair_ids.value)?;
        self.validate_batched_graph(batched_graph.value)?;
        self.validate_unified_graph(
            unified_graph.value,
            unified_graph_layers_only.value,
            unified_graph_lm_head_eager.value,
        )?;
        self.validate_memory(
            kv_blocks.value,
            max_sequences.value,
            max_batched_tokens.value,
            max_model_len.as_ref().map(|value| value.value),
        )?;
        self.validate_dtypes()?;
        self.validate_layer_split_pipeline_mode()?;
        self.validate_sampling(greedy.value)?;

        if let Some((_, compiled_attention)) = plan_attention.as_ref() {
            decisions.push(self.plan_attention_execution_decision(compiled_attention));
            decisions.push(self.plan_attention_phase_decision(
                "attention_prefill_mixed_backend",
                compiled_attention,
            ));
            decisions.push(
                self.plan_attention_phase_decision("attention_decode_backend", compiled_attention),
            );
        } else {
            decisions.push(self.attention_prefill_decision(
                use_vllm_paged_attn.clone(),
                fa_layout,
                fa2_source,
                fa2_direct_ffi,
            ));
            decisions.push(self.fa2_native_artifact_decision(
                fa2_native_manifest.as_ref(),
                fa2_native_artifact.as_ref(),
            ));
            decisions.push(self.fa2_native_runtime_selection_decision(
                fa2_native_manifest.as_ref(),
                fa2_native_artifact.as_ref(),
            ));
            decisions.push(
                self.attention_decode_decision(use_vllm_paged_attn.clone(), vllm_v1_short.clone()),
            );
        }
        // Materialize the auto-resolved fast-path MoE knobs into the effective
        // config BEFORE moe_decision consumes them, so they reach the model
        // (which reads FERRUM_*, not the decisions). Only auto-derived values —
        // user/env entries are already present. Without this, `ferrum run`'s
        // non-preset path resolved FERRUM_VLLM_MOE as a decision only and the
        // model never saw it (~9.7 vs ~59 tok/s on a 4090 for Qwen3-30B-A3B).
        let execution = self.resolve_execution_defaults(
            reusable_execution.value,
            plan_attention.as_ref().map(|(_, compiled)| compiled.value),
        )?;
        let mut runtime_config = execution.runtime_config.clone();
        // The retained allocator consumes the snapshot, not the decision trace.
        // Publish its actual physical pool and logical table limits together;
        // otherwise its private defaults can allocate a different pool.
        if let Some(capacity) = legacy_kv_capacity.as_ref() {
            for (key, resolved) in [
                ("FERRUM_KV_CAPACITY", capacity),
                ("FERRUM_KV_MAX_BLOCKS", &kv_blocks),
            ] {
                if key == "FERRUM_KV_MAX_BLOCKS"
                    && self.selected_kv_cache_layout.block_size_tokens().is_none()
                {
                    continue;
                }
                let source = self.entry(key).map(|entry| entry.source).unwrap_or(
                    if resolved.source == AutoConfigSource::HardwareCapability {
                        RuntimeConfigSource::MemoryProfile
                    } else {
                        RuntimeConfigSource::Default
                    },
                );
                runtime_config.upsert(key, resolved.value.to_string(), source);
            }
        }
        // Native resource selection is one typed decision. A diagnostic-only
        // decision must never leave the engine running a different default.
        if plan_runtime || legacy_kv_capacity.is_some() {
            for (key, resolved) in [
                ("FERRUM_PAGED_MAX_SEQS", Some(&max_sequences)),
                ("FERRUM_MAX_BATCHED_TOKENS", Some(&max_batched_tokens)),
                ("FERRUM_MAX_MODEL_LEN", max_model_len.as_ref()),
            ] {
                if let Some(resolved) = resolved {
                    let value = resolved.value.to_string();
                    if self
                        .entry(key)
                        .is_none_or(|entry| entry.effective_value != value)
                    {
                        runtime_config.upsert(
                            key,
                            value,
                            if let Some(entry) = self.entry(key) {
                                entry.source
                            } else if resolved.source == AutoConfigSource::HardwareCapability {
                                RuntimeConfigSource::MemoryProfile
                            } else {
                                RuntimeConfigSource::Default
                            },
                        );
                    }
                }
            }
        }
        let legacy_attention_values = [
            ("FERRUM_USE_VLLM_PAGED_ATTN", &use_vllm_paged_attn),
            ("FERRUM_VLLM_PAGED_ATTN_V1_SHORT", &vllm_v1_short),
        ];
        for (key, resolved) in legacy_attention_values
            .into_iter()
            .filter(|_| !plan_runtime)
            .chain([
                ("FERRUM_VLLM_MOE", &vllm_moe),
                ("FERRUM_VLLM_MOE_PAIR_IDS", &pair_ids),
                ("FERRUM_BATCHED_GRAPH", &batched_graph),
                ("FERRUM_REUSABLE_EXECUTION", &reusable_execution),
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
            ])
        {
            if self.entry(key).is_none() {
                runtime_config.upsert(
                    key,
                    if resolved.value { "1" } else { "0" },
                    RuntimeConfigSource::MemoryProfile,
                );
            }
        }
        if self.entry("FERRUM_ATTENTION_POLICY").is_none() {
            if let Some((_, compiled_attention)) = plan_attention.as_ref() {
                runtime_config.upsert(
                    "FERRUM_ATTENTION_POLICY",
                    compiled_attention.value.as_runtime_value(),
                    RuntimeConfigSource::Default,
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
        if let Some(chunk) = default_active_decode_prefill_chunk.as_ref() {
            if self.entry("FERRUM_ACTIVE_DECODE_PREFILL_CHUNK").is_none() {
                runtime_config.upsert(
                    "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK",
                    chunk.value.to_string(),
                    RuntimeConfigSource::Default,
                );
            }
        }
        decisions.push(self.moe_decision(vllm_moe));
        decisions.push(self.reusable_execution_decision(reusable_execution));
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
            default_active_decode_prefill_chunk,
        )?);
        // Validate and describe the same resolved values that reach the engine.
        // In particular, dependent defaults must not re-read absent raw entries.
        let mut execution_decisions = vec![
            execution.prefill_decode_execution_decision()?,
            execution.invocation_preparation_strategy_decision()?,
            execution.sampling_readback_mode_decision()?,
            execution.program_binding_upload_strategy_decision()?,
            execution.completion_readback_policy_decision()?,
            execution.segment_binding_owner_view_mode_decision()?,
            execution.causal_decode_attention_grouping_decision(
                plan_attention.as_ref().map(|(_, compiled)| compiled.value),
            )?,
            execution.causal_decode_preparation_mode_decision(
                plan_attention.as_ref().map(|(_, compiled)| compiled.value),
            )?,
            execution.causal_decode_kv_reuse_decision()?,
            execution.sequence_backing_growth_decision()?,
            execution.q4_mmvq_weight_decode_decision()?,
            execution.q5_mmvq_weight_decode_decision()?,
            execution.q8_projection_weight_preparation_decision()?,
        ];
        for decision in &mut execution_decisions {
            if decision
                .source_key
                .as_deref()
                .is_some_and(|key| self.entry(key).is_none())
            {
                decision.source_key = None;
            }
        }
        decisions.extend(execution_decisions);
        decisions.push(self.sampling_decision(greedy));

        Ok(ResolvedFerrumConfig {
            schema_version: 1,
            preset: self.workload.preset.clone(),
            execution_planning_model: self.execution_planning_model,
            selected_kv_cache_layout: self.selected_kv_cache_layout,
            selected_kv_cache_storage: self.selected_kv_cache_storage,
            requested_attention_policy: plan_attention
                .as_ref()
                .map(|(requested, _)| requested.value),
            compiled_attention_policy: plan_attention.as_ref().map(|(_, compiled)| compiled.value),
            runtime_config,
            model_capabilities: self.model.clone(),
            hardware_capabilities: self.hardware.clone(),
            workload_profile: self.workload.clone(),
            decisions,
            startup_memory_plan: None,
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
        let kv_token_capacity = self
            .selected_kv_cache_layout
            .block_size_tokens()
            .map(|block_size| kv_blocks.value.saturating_mul(block_size));
        let target = if self
            .workload
            .is_preset(QWEN25_72B_GPTQ_INT4_2X4090_LAYER_SPLIT_PRESET)
        {
            1536
        } else {
            max_sequences.value.max(1).saturating_mul(64)
        };
        // Only the selected paged allocator turns this block count into token
        // capacity. Contiguous and unqualified layouts retain the workload target.
        let value = kv_token_capacity
            .map_or(target, |capacity| target.min(capacity))
            .max(max_sequences.value.max(1));
        ResolvedValue {
            value,
            source: if max_sequences.source == AutoConfigSource::HardwareCapability
                || (kv_token_capacity.is_some()
                    && kv_blocks.source == AutoConfigSource::HardwareCapability)
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

    fn default_active_decode_prefill_chunk(
        &self,
        max_sequences: &ResolvedValue<usize>,
    ) -> Option<ResolvedValue<usize>> {
        if self.execution_planning_model != ExecutionPlanningModel::CompiledPlan
            || max_sequences.value <= 1
        {
            return None;
        }
        let value = if self.hardware.backend.eq_ignore_ascii_case("metal") {
            128
        } else if self.is_cuda_backend() && self.hardware.compiled_features.cuda {
            256
        } else {
            return None;
        };
        // Bound mixed-prefill work even with only one active decoder. Cold
        // prefills keep their elastic token budget; explicit limits still win.
        Some(ResolvedValue {
            value,
            source: AutoConfigSource::Default,
            source_key: None,
        })
    }

    fn default_kv_blocks(
        &self,
        max_sequences: &ResolvedValue<usize>,
        legacy_kv_capacity: Option<&ResolvedValue<usize>>,
    ) -> Result<ResolvedValue<usize>, AutoConfigError> {
        let block_size = self
            .selected_kv_cache_layout
            .block_size_tokens()
            .unwrap_or(DEFAULT_KV_BLOCK_SIZE_TOKENS);
        let single_sequence_blocks = legacy_kv_capacity
            .map(|capacity| capacity.value.div_ceil(block_size))
            .unwrap_or(1);
        let min_blocks =
            ceil_div(max_sequences.value.max(1), block_size).max(single_sequence_blocks);
        let useful_blocks = match (
            legacy_kv_capacity,
            self.selected_kv_cache_layout.block_size_tokens(),
        ) {
            (Some(capacity), Some(block_size)) => Some(
                capacity
                    .value
                    .div_ceil(block_size)
                    .checked_mul(max_sequences.value)
                    .ok_or_else(|| AutoConfigError::InvalidOverride {
                        key: "FERRUM_KV_CAPACITY".into(),
                        reason: "combined sequence capacity overflows the physical block count"
                            .into(),
                    })?,
            ),
            _ => None,
        };
        let preset_blocks = if self
            .workload
            .is_preset(QWEN25_72B_GPTQ_INT4_2X4090_LAYER_SPLIT_PRESET)
        {
            1024
        } else {
            DEFAULT_KV_BLOCKS
        };
        let target = useful_blocks.map_or(preset_blocks.max(min_blocks), |limit| {
            preset_blocks.max(min_blocks).min(limit)
        });
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
                let block_budget = kv_token_budget / block_size;
                if block_budget < min_blocks && self.entry("FERRUM_KV_MAX_BLOCKS").is_none() {
                    return self.invalid("FERRUM_MAX_MODEL_LEN", &format!("estimated available KV capacity ({block_budget} blocks) cannot fit the selected context and concurrency minimum ({min_blocks} blocks); no physical pool was allocated"));
                }
                target.min(block_budget.max(min_blocks))
            }
            _ => target,
        };
        Ok(ResolvedValue {
            value: selected.max(1),
            source: if selected < target {
                AutoConfigSource::HardwareCapability
            } else {
                AutoConfigSource::WorkloadPreset
            },
            source_key: None,
        })
    }

    fn resolve_legacy_kv_capacity(
        &self,
        max_model_len: Option<&ResolvedValue<usize>>,
    ) -> Result<ResolvedValue<usize>, AutoConfigError> {
        if max_model_len.is_some_and(|ceiling| ceiling.value == 0) {
            return self.invalid("FERRUM_MAX_MODEL_LEN", "must be greater than zero");
        }
        let configured = self.optional_usize_value("FERRUM_KV_CAPACITY")?;
        let explicit = self.entry("FERRUM_KV_CAPACITY").is_some_and(|entry| {
            matches!(
                entry.source,
                RuntimeConfigSource::Cli
                    | RuntimeConfigSource::Env
                    | RuntimeConfigSource::ConfigFile
                    | RuntimeConfigSource::ScriptCase
            )
        });
        let selected = if explicit {
            configured.expect("explicit KV capacity was parsed")
        } else if let Some(max_model_len) = max_model_len {
            ResolvedValue {
                value: max_model_len.value,
                source: AutoConfigSource::Default,
                source_key: None,
            }
        } else {
            configured.unwrap_or(ResolvedValue {
                // Preserve the retained allocator's bounded default, but make it
                // visible and pass it explicitly instead of hiding it in execution.
                value: self.model.max_context_len.unwrap_or(512).min(512),
                source: AutoConfigSource::Default,
                source_key: None,
            })
        };
        if selected.value == 0 {
            return self.invalid("FERRUM_KV_CAPACITY", "must be greater than zero");
        }
        if max_model_len.is_some_and(|ceiling| selected.value < ceiling.value) {
            return self.invalid(
                "FERRUM_KV_CAPACITY",
                "per-sequence KV capacity is smaller than FERRUM_MAX_MODEL_LEN",
            );
        }
        if self
            .model
            .max_context_len
            .is_some_and(|ceiling| selected.value > ceiling)
        {
            if explicit {
                return self.invalid(
                    "FERRUM_KV_CAPACITY",
                    "exceeds model metadata max context length",
                );
            }
            return Ok(ResolvedValue {
                value: self.model.max_context_len.expect("checked model ceiling"),
                ..selected
            });
        }
        Ok(selected)
    }

    fn kv_cache_bytes_per_token(&self) -> Option<u64> {
        self.selected_kv_cache_storage?.bytes_per_token(
            self.model.num_hidden_layers?,
            self.model.kv_heads?,
            self.model.head_dim?,
        )
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

    fn optional_string_value(
        &self,
        key: &str,
    ) -> Result<Option<ResolvedValue<String>>, AutoConfigError> {
        match self.entry(key) {
            Some(entry) if entry.effective_value.trim().is_empty() => {
                Err(AutoConfigError::InvalidOverride {
                    key: key.to_string(),
                    reason: "must be non-empty".to_string(),
                })
            }
            Some(entry) => Ok(Some(ResolvedValue {
                value: entry.effective_value.clone(),
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
        if fa2_source {
            return self.invalid(
                "FERRUM_FA2_SOURCE",
                "source-linked FA2 path has been removed; use a native operator artifact",
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

    fn validate_fa2_native_artifact(
        &self,
        manifest: Option<&ResolvedValue<String>>,
        artifact: Option<&ResolvedValue<String>>,
        source_sha256: Option<&ResolvedValue<String>>,
        inputs_sha256: Option<&ResolvedValue<String>>,
        fa2_source: bool,
        fa2_direct_ffi: bool,
    ) -> Result<(), AutoConfigError> {
        let configured = manifest.is_some() || artifact.is_some();
        if manifest.is_some() != artifact.is_some() {
            let key = if manifest.is_none() {
                FA2_NATIVE_MANIFEST_KEY
            } else {
                FA2_NATIVE_ARTIFACT_KEY
            };
            return self.invalid(
                key,
                "FA2 native operator manifest and artifact must be configured together",
            );
        }
        for (key, value) in [
            (FA2_NATIVE_SOURCE_SHA256_KEY, source_sha256),
            (FA2_NATIVE_INPUTS_SHA256_KEY, inputs_sha256),
        ] {
            if let Some(value) = value {
                if !is_sha256_digest(&value.value) {
                    return self.invalid(key, "must be a lowercase hex sha256 digest");
                }
                if !configured {
                    return self.invalid(
                        key,
                        "sha256 pins require FA2 native operator manifest and artifact",
                    );
                }
            }
        }
        if configured && !self.is_cuda_backend() {
            return self.invalid(
                FA2_NATIVE_MANIFEST_KEY,
                "FA2 native operator artifacts require CUDA backend",
            );
        }
        if configured && !self.hardware.compiled_features.fa2_native_operator_artifact {
            return self.invalid(
                FA2_NATIVE_MANIFEST_KEY,
                "FA2 native operator artifact config requires a binary built with a validated native operator artifact",
            );
        }
        if configured {
            let metadata = self
                .hardware
                .compiled_features
                .fa2_native_operator_artifact_metadata
                .as_ref()
                .ok_or_else(|| AutoConfigError::InvalidOverride {
                    key: FA2_NATIVE_MANIFEST_KEY.to_string(),
                    reason: "FA2 native operator artifact capability is missing build metadata"
                        .to_string(),
                })?;
            let manifest = manifest.expect("configured manifest is present");
            let artifact = artifact.expect("configured artifact is present");
            if manifest.value != metadata.manifest_path {
                return self.invalid(
                    FA2_NATIVE_MANIFEST_KEY,
                    "FA2 native operator manifest does not match the artifact linked into this binary",
                );
            }
            if artifact.value != metadata.artifact_path {
                return self.invalid(
                    FA2_NATIVE_ARTIFACT_KEY,
                    "FA2 native operator artifact does not match the artifact linked into this binary",
                );
            }
            if let Some(source_sha256) = source_sha256 {
                if source_sha256.value != metadata.source_package_sha256 {
                    return self.invalid(
                        FA2_NATIVE_SOURCE_SHA256_KEY,
                        "FA2 native operator source_package sha256 does not match the artifact linked into this binary",
                    );
                }
            }
            if let Some(inputs_sha256) = inputs_sha256 {
                if inputs_sha256.value != metadata.inputs_sha256 {
                    return self.invalid(
                        FA2_NATIVE_INPUTS_SHA256_KEY,
                        "FA2 native operator inputs sha256 does not match the artifact linked into this binary",
                    );
                }
            }
        }
        if configured && fa2_source {
            return self.unsupported(
                "attention_prefill_mixed_backend",
                "FA2 native operator artifact cannot be combined with removed source-linked FA2",
            );
        }
        if configured && fa2_direct_ffi {
            return self.unsupported(
                "attention_prefill_mixed_backend",
                "FA2 native operator artifact and diagnostic direct FFI shim cannot both own the prefill path",
            );
        }
        Ok(())
    }

    fn validate_moe(&self, vllm_moe: bool, pair_ids: bool) -> Result<(), AutoConfigError> {
        if vllm_moe && !self.hardware.compiled_features.vllm_moe_marlin {
            return self.invalid("FERRUM_VLLM_MOE", "vLLM Marlin MoE is not compiled");
        }
        if vllm_moe && !self.is_cuda_backend() {
            return self.invalid("FERRUM_VLLM_MOE", "vLLM Marlin MoE requires CUDA backend");
        }
        if pair_ids && !vllm_moe {
            return self.invalid(
                "FERRUM_VLLM_MOE_PAIR_IDS",
                "pair-id routing requires vLLM MoE",
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
        max_batched_tokens: usize,
        requested_max_model_len: Option<usize>,
    ) -> Result<(), AutoConfigError> {
        if kv_blocks == 0 {
            return self.invalid("FERRUM_KV_MAX_BLOCKS", "must be greater than zero");
        }
        if self.selected_kv_cache_layout.block_size_tokens().is_some()
            && u32::try_from(kv_blocks).is_err()
        {
            return self.invalid(
                "FERRUM_KV_MAX_BLOCKS",
                "exceeds the physical allocator's u32 block index domain",
            );
        }
        if max_sequences == 0 {
            return self.invalid("FERRUM_PAGED_MAX_SEQS", "must be greater than zero");
        }
        if max_batched_tokens < max_sequences {
            return self.invalid(
                "FERRUM_MAX_BATCHED_TOKENS",
                "must be at least FERRUM_PAGED_MAX_SEQS",
            );
        }
        let kv_token_capacity = self
            .selected_kv_cache_layout
            .block_size_tokens()
            .map(|block_size| {
                kv_blocks
                    .checked_mul(block_size)
                    .ok_or_else(|| AutoConfigError::InvalidOverride {
                        key: "FERRUM_KV_MAX_BLOCKS".into(),
                        reason: "physical KV token capacity overflows".into(),
                    })
            })
            .transpose()?;
        if let (Some(tokens), Some(bytes_per_token)) =
            (kv_token_capacity, self.kv_cache_bytes_per_token())
        {
            if u64::try_from(tokens)
                .ok()
                .and_then(|tokens| tokens.checked_mul(bytes_per_token))
                .is_none()
            {
                return self.invalid(
                    "FERRUM_KV_MAX_BLOCKS",
                    "physical KV storage byte count overflows",
                );
            }
        }
        if kv_token_capacity.is_some_and(|capacity| max_batched_tokens > capacity) {
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
            if kv_token_capacity.is_some_and(|capacity| max_model_len > capacity) {
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

    fn resolve_plan_attention_policy(
        &self,
    ) -> Result<
        (
            ResolvedValue<AttentionExecutionPolicy>,
            ResolvedValue<AttentionExecutionPolicy>,
        ),
        AutoConfigError,
    > {
        let requested = match self.entry("FERRUM_ATTENTION_POLICY") {
            Some(entry) => ResolvedValue {
                value: AttentionExecutionPolicy::parse_runtime_value(&entry.effective_value)
                    .map_err(|reason| AutoConfigError::InvalidOverride {
                        key: "FERRUM_ATTENTION_POLICY".to_owned(),
                        reason,
                    })?,
                source: auto_config_source_from_runtime(entry.source),
                source_key: Some("FERRUM_ATTENTION_POLICY".to_owned()),
            },
            None => ResolvedValue {
                value: AttentionExecutionPolicy::Auto,
                source: AutoConfigSource::Default,
                source_key: None,
            },
        };
        let kv_dtype = self
            .raw("FERRUM_KV_DTYPE")
            .map(|raw| {
                crate::KvCacheDtype::parse(raw).ok_or_else(|| AutoConfigError::InvalidOverride {
                    key: "FERRUM_KV_DTYPE".to_owned(),
                    reason: format!("unknown KV dtype {raw}"),
                })
            })
            .transpose()?
            .unwrap_or_default();
        let kv_storage = crate::KvStorageFormat::try_from(kv_dtype).map_err(|reason| {
            AutoConfigError::InvalidOverride {
                key: "FERRUM_KV_DTYPE".to_owned(),
                reason,
            }
        })?;
        // The native CUDA ABI consumes F16 pages. Resolve Auto within the
        // requested typed storage before materializing a default policy into
        // the runtime snapshot, or a later stage mistakes that default for an
        // explicit incompatible NativeAdaptive request.
        if kv_storage != crate::KvStorageFormat::F16
            && requested.value == AttentionExecutionPolicy::NativeAdaptive
        {
            return Err(AutoConfigError::UnsupportedCombination {
                selection: "attention_execution_policy".to_owned(),
                reason: "native-adaptive attention does not implement the requested INT8 KV storage; use auto or portable".to_owned(),
            });
        }
        let native_adaptive_supported = self.is_cuda_backend()
            && self.hardware.compiled_features.vllm_paged_attn
            && kv_storage == crate::KvStorageFormat::F16;
        let compiled = requested
            .value
            .resolve(native_adaptive_supported)
            .map_err(|reason| AutoConfigError::UnsupportedCombination {
                selection: "attention_execution_policy".to_owned(),
                reason,
            })?;
        Ok((
            requested.clone(),
            ResolvedValue {
                value: compiled,
                source: if requested.value == AttentionExecutionPolicy::Auto {
                    AutoConfigSource::CompiledFeature
                } else {
                    requested.source
                },
                source_key: requested.source_key.clone(),
            },
        ))
    }

    fn plan_attention_execution_decision(
        &self,
        compiled: &ResolvedValue<AttentionExecutionPolicy>,
    ) -> AutoConfigDecision {
        let selected = compiled.value.as_runtime_value();
        self.decision(
            "attention_execution_policy",
            selected,
            compiled.source,
            compiled.source_key.clone(),
            ["native-adaptive", "portable"],
            self.rejected_except(
                selected,
                [
                    (
                        "native-adaptive",
                        "native adaptive provider is not selected or compiled",
                    ),
                    ("portable", "portable provider is not selected"),
                ],
            ),
            vec![
                RuntimeConfigEffect::Correctness,
                RuntimeConfigEffect::Performance,
            ],
        )
    }

    fn plan_attention_phase_decision(
        &self,
        selection: &str,
        compiled: &ResolvedValue<AttentionExecutionPolicy>,
    ) -> AutoConfigDecision {
        let selected = match (
            self.hardware.backend.to_ascii_lowercase().as_str(),
            compiled.value,
        ) {
            ("cuda", AttentionExecutionPolicy::NativeAdaptive) => "cuda_native_adaptive",
            ("metal", AttentionExecutionPolicy::NativeAdaptive) => "metal_native_adaptive",
            (_, AttentionExecutionPolicy::Portable) => "portable",
            (_, AttentionExecutionPolicy::Auto) => "unresolved",
            _ => "portable",
        };
        self.decision(
            selection,
            selected,
            compiled.source,
            compiled.source_key.clone(),
            ["cuda_native_adaptive", "metal_native_adaptive", "portable"],
            self.rejected_except(
                selected,
                [
                    (
                        "cuda_native_adaptive",
                        "CUDA native adaptive provider is not selected",
                    ),
                    (
                        "metal_native_adaptive",
                        "Metal native adaptive provider is not selected",
                    ),
                    ("portable", "portable provider is not selected"),
                ],
            ),
            vec![
                RuntimeConfigEffect::Correctness,
                RuntimeConfigEffect::Performance,
            ],
        )
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

    fn fa2_native_artifact_decision(
        &self,
        manifest: Option<&ResolvedValue<String>>,
        artifact: Option<&ResolvedValue<String>>,
    ) -> AutoConfigDecision {
        let configured = manifest.is_some() && artifact.is_some();
        let selected = if configured {
            "configured"
        } else {
            "not_configured"
        };
        let (source, source_key) = manifest
            .map(|manifest| (manifest.source, manifest.source_key.clone()))
            .unwrap_or((AutoConfigSource::Default, None));
        self.decision(
            "fa2_native_operator_artifact",
            selected,
            source,
            source_key,
            ["configured", "not_configured"],
            self.rejected_except(
                selected,
                [
                    (
                        "configured",
                        "no typed FA2 native operator manifest/artifact was configured",
                    ),
                    (
                        "not_configured",
                        "typed FA2 native operator manifest/artifact is present",
                    ),
                ],
            ),
            vec![
                RuntimeConfigEffect::Correctness,
                RuntimeConfigEffect::Performance,
            ],
        )
    }

    fn fa2_native_runtime_selection_decision(
        &self,
        manifest: Option<&ResolvedValue<String>>,
        artifact: Option<&ResolvedValue<String>>,
    ) -> AutoConfigDecision {
        let configured = manifest.is_some() && artifact.is_some();
        let selected = if configured {
            "not_selected"
        } else {
            "not_configured"
        };
        let (source, source_key) = manifest
            .map(|manifest| (manifest.source, manifest.source_key.clone()))
            .unwrap_or((AutoConfigSource::Default, None));
        self.decision(
            "fa2_native_operator_runtime_selection",
            selected,
            source,
            source_key,
            ["selected", "not_selected", "not_configured"],
            self.rejected_except(
                selected,
                [
                    (
                        "selected",
                        "FA2 native runtime dispatch requires artifact-selected actual model smoke and retention evidence",
                    ),
                    (
                        "not_selected",
                        "typed FA2 native operator manifest/artifact is present",
                    ),
                    (
                        "not_configured",
                        "no typed FA2 native operator manifest/artifact was configured",
                    ),
                ],
            ),
            vec![
                RuntimeConfigEffect::Correctness,
                RuntimeConfigEffect::Performance,
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

    fn moe_decision(&self, vllm_moe: ResolvedValue<bool>) -> AutoConfigDecision {
        // Startup selects the shared implementation preference. Actual device
        // routing additionally requires executor-owned route scratch; it must
        // not be claimed by an environment option with no execution consumer.
        let selected = if vllm_moe.value {
            "vllm_marlin_enabled"
        } else {
            "vllm_marlin_disabled"
        };
        self.decision(
            "moe_dispatch_preference",
            selected,
            vllm_moe.source,
            vllm_moe.source_key,
            ["vllm_marlin_enabled", "vllm_marlin_disabled"],
            self.rejected_except(
                selected,
                [
                    ("vllm_marlin_enabled", "vLLM Marlin MoE is not enabled"),
                    ("vllm_marlin_disabled", "vLLM Marlin MoE is enabled"),
                ],
            ),
            vec![RuntimeConfigEffect::Performance],
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

    fn reusable_execution_decision(
        &self,
        reusable_execution: ResolvedValue<bool>,
    ) -> AutoConfigDecision {
        let selected = if reusable_execution.value {
            "enabled_when_runtime_capable"
        } else {
            "disabled"
        };
        self.decision(
            "reusable_execution_policy",
            selected,
            reusable_execution.source,
            reusable_execution.source_key,
            ["enabled_when_runtime_capable", "disabled"],
            self.rejected_except(
                selected,
                [
                    (
                        "enabled_when_runtime_capable",
                        "reusable device programs are disabled",
                    ),
                    ("disabled", "reusable device programs are enabled"),
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

    fn invocation_preparation_strategy_decision(
        &self,
    ) -> Result<AutoConfigDecision, AutoConfigError> {
        let key = "FERRUM_INVOCATION_PREPARATION_STRATEGY";
        let (strategy, source, source_key) = match self.entry(key) {
            Some(entry) => (
                crate::InvocationPreparationStrategy::parse_runtime_value(&entry.effective_value)
                    .map_err(|reason| AutoConfigError::InvalidOverride {
                    key: key.to_owned(),
                    reason,
                })?,
                auto_config_source_from_runtime(entry.source),
                Some(key.to_owned()),
            ),
            None => (
                crate::InvocationPreparationStrategy::default(),
                AutoConfigSource::Default,
                None,
            ),
        };
        if strategy.uses_decode_segment_program() {
            let owner_mode = self
                .entry(crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY)
                .map(crate::SegmentBindingOwnerViewMode::parse_config_entry)
                .transpose()
                .map_err(|reason| AutoConfigError::InvalidOverride {
                    key: crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY.to_owned(),
                    reason,
                })?
                .unwrap_or_default();
            let upload = self
                .entry("FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY")
                .map(|entry| {
                    crate::ProgramBindingUploadStrategy::parse_runtime_value(&entry.effective_value)
                })
                .transpose()
                .map_err(|reason| AutoConfigError::InvalidOverride {
                    key: "FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY".to_owned(),
                    reason,
                })?
                .unwrap_or_default();
            if !self.is_cuda_backend()
                || self.execution_planning_model != ExecutionPlanningModel::CompiledPlan
                || owner_mode != crate::SegmentBindingOwnerViewMode::Indexed
                || upload != crate::ProgramBindingUploadStrategy::CompactScatter
            {
                return Err(AutoConfigError::InvalidOverride {
                    key: key.to_owned(),
                    reason: format!(
                        "{} requires a CUDA plan runtime, indexed owner views and compact-scatter uploads",
                        strategy.as_runtime_value()
                    ),
                });
            }
        }
        Ok(self.decision(
            "invocation_preparation_strategy",
            strategy.as_runtime_value(),
            source,
            source_key,
            [
                "full",
                "identity-projection",
                "decode-segment",
                "decode-segment-program",
                "resident-decode-program",
            ],
            Vec::new(),
            vec![RuntimeConfigEffect::Performance],
        ))
    }

    fn completion_readback_policy_decision(&self) -> Result<AutoConfigDecision, AutoConfigError> {
        let key = crate::COMPLETION_READBACK_POLICY_CONFIG_KEY;
        let (policy, source, source_key) = match self.entry(key) {
            Some(entry) => (
                crate::CompletionReadbackPolicy::parse_config_entry(entry).map_err(|reason| {
                    AutoConfigError::InvalidOverride {
                        key: key.to_owned(),
                        reason,
                    }
                })?,
                auto_config_source_from_runtime(entry.source),
                Some(key.to_owned()),
            ),
            None => (
                crate::CompletionReadbackPolicy::default(),
                AutoConfigSource::Default,
                None,
            ),
        };
        Ok(self.decision(
            "completion_readback_policy",
            policy.as_runtime_value(),
            source,
            source_key,
            ["synchronous", "batched-u32"],
            Vec::new(),
            vec![RuntimeConfigEffect::Performance],
        ))
    }

    fn sampling_readback_mode_decision(&self) -> Result<AutoConfigDecision, AutoConfigError> {
        let key = "FERRUM_SAMPLING_READBACK_MODE";
        let (mode, source, source_key) = match self.entry(key) {
            Some(entry) => (
                crate::SamplingReadbackMode::parse_runtime_value(&entry.effective_value).map_err(
                    |reason| AutoConfigError::InvalidOverride {
                        key: key.to_owned(),
                        reason,
                    },
                )?,
                auto_config_source_from_runtime(entry.source),
                Some(key.to_owned()),
            ),
            None => (
                crate::SamplingReadbackMode::default(),
                AutoConfigSource::Default,
                None,
            ),
        };
        if mode == crate::SamplingReadbackMode::Participant
            && self.execution_planning_model != ExecutionPlanningModel::CompiledPlan
        {
            return Err(AutoConfigError::InvalidOverride {
                key: key.to_owned(),
                reason: "participant sampling readback requires a plan runtime".to_owned(),
            });
        }
        Ok(self.decision(
            "sampling_readback_mode",
            mode.as_runtime_value(),
            source,
            source_key,
            ["uniform", "participant"],
            Vec::new(),
            vec![
                RuntimeConfigEffect::Correctness,
                RuntimeConfigEffect::Performance,
                RuntimeConfigEffect::Memory,
            ],
        ))
    }

    fn program_binding_upload_strategy_decision(
        &self,
    ) -> Result<AutoConfigDecision, AutoConfigError> {
        let key = "FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY";
        let (strategy, source, source_key) = match self.entry(key) {
            Some(entry) => (
                crate::ProgramBindingUploadStrategy::parse_runtime_value(&entry.effective_value)
                    .map_err(|reason| AutoConfigError::InvalidOverride {
                        key: key.to_owned(),
                        reason,
                    })?,
                auto_config_source_from_runtime(entry.source),
                Some(key.to_owned()),
            ),
            None => (
                crate::ProgramBindingUploadStrategy::default(),
                AutoConfigSource::Default,
                None,
            ),
        };
        if strategy != crate::ProgramBindingUploadStrategy::Sparse
            && (!self.is_cuda_backend()
                || self.execution_planning_model != ExecutionPlanningModel::CompiledPlan)
        {
            return Err(AutoConfigError::InvalidOverride {
                key: key.to_owned(),
                reason: format!(
                    "{} requires a CUDA plan runtime",
                    strategy.as_runtime_value()
                ),
            });
        }
        Ok(self.decision(
            "program_binding_upload_strategy",
            strategy.as_runtime_value(),
            source,
            source_key,
            ["sparse", "uniform-live-prefix", "compact-scatter"],
            Vec::new(),
            vec![RuntimeConfigEffect::Performance],
        ))
    }

    fn segment_binding_owner_view_mode_decision(
        &self,
    ) -> Result<AutoConfigDecision, AutoConfigError> {
        let key = crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY;
        let (mode, source, source_key) = match self.entry(key) {
            Some(entry) => (
                crate::SegmentBindingOwnerViewMode::parse_config_entry(entry).map_err(
                    |reason| AutoConfigError::InvalidOverride {
                        key: key.to_owned(),
                        reason,
                    },
                )?,
                auto_config_source_from_runtime(entry.source),
                Some(key.to_owned()),
            ),
            None => (
                crate::SegmentBindingOwnerViewMode::default(),
                AutoConfigSource::Default,
                None,
            ),
        };
        if mode == crate::SegmentBindingOwnerViewMode::Indexed {
            let preparation = self
                .entry("FERRUM_INVOCATION_PREPARATION_STRATEGY")
                .map(|entry| {
                    crate::InvocationPreparationStrategy::parse_runtime_value(
                        &entry.effective_value,
                    )
                })
                .transpose()
                .map_err(|reason| AutoConfigError::InvalidOverride {
                    key: "FERRUM_INVOCATION_PREPARATION_STRATEGY".to_owned(),
                    reason,
                })?
                .unwrap_or_default();
            if !self.is_cuda_backend()
                || self.execution_planning_model != ExecutionPlanningModel::CompiledPlan
                || !preparation.uses_decode_segment()
            {
                return Err(AutoConfigError::InvalidOverride {
                    key: key.to_owned(),
                    reason: "indexed segment binding owner views require a CUDA plan runtime with decode-segment preparation".to_owned(),
                });
            }
        }
        Ok(self.decision(
            "segment_binding_owner_view_mode",
            mode.as_runtime_value(),
            source,
            source_key,
            ["legacy", "indexed"],
            Vec::new(),
            vec![RuntimeConfigEffect::Performance],
        ))
    }

    fn causal_decode_preparation_mode_decision(
        &self,
        compiled_attention: Option<AttentionExecutionPolicy>,
    ) -> Result<AutoConfigDecision, AutoConfigError> {
        let key = crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY;
        let (mode, source, source_key) = match self.entry(key) {
            Some(entry) => (
                crate::CausalDecodePreparationMode::parse_config_entry(entry).map_err(
                    |reason| AutoConfigError::InvalidOverride {
                        key: key.to_owned(),
                        reason,
                    },
                )?,
                auto_config_source_from_runtime(entry.source),
                Some(key.to_owned()),
            ),
            None => (
                crate::CausalDecodePreparationMode::default(),
                AutoConfigSource::Default,
                None,
            ),
        };
        if mode == crate::CausalDecodePreparationMode::Packed
            && (!self.is_cuda_backend()
                || self.execution_planning_model != ExecutionPlanningModel::CompiledPlan
                || !self.hardware.compiled_features.vllm_paged_attn
                || compiled_attention != Some(AttentionExecutionPolicy::NativeAdaptive))
        {
            return Err(AutoConfigError::InvalidOverride {
                key: key.to_owned(),
                reason: "packed causal decode preparation requires a CUDA plan runtime with compiled native-adaptive paged attention and F16 KV storage".to_owned(),
            });
        }
        Ok(self.decision(
            "causal_decode_preparation_mode",
            mode.as_runtime_value(),
            source,
            source_key,
            ["per-participant", "packed"],
            Vec::new(),
            vec![RuntimeConfigEffect::Performance],
        ))
    }

    fn causal_decode_attention_grouping_decision(
        &self,
        compiled_attention: Option<AttentionExecutionPolicy>,
    ) -> Result<AutoConfigDecision, AutoConfigError> {
        let key = crate::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY;
        let (grouping, source, source_key) = match self.entry(key) {
            Some(entry) => (
                crate::CausalDecodeAttentionGrouping::parse_config_entry(entry).map_err(
                    |reason| AutoConfigError::InvalidOverride {
                        key: key.to_owned(),
                        reason,
                    },
                )?,
                auto_config_source_from_runtime(entry.source),
                Some(key.to_owned()),
            ),
            None => (
                crate::CausalDecodeAttentionGrouping::default(),
                AutoConfigSource::Default,
                None,
            ),
        };
        if matches!(
            grouping,
            crate::CausalDecodeAttentionGrouping::Indexed
                | crate::CausalDecodeAttentionGrouping::FixedCapacity
                | crate::CausalDecodeAttentionGrouping::FixedExecutionCapacity
        ) {
            let preparation = self
                .entry(crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY)
                .map(crate::CausalDecodePreparationMode::parse_config_entry)
                .transpose()
                .map_err(|reason| AutoConfigError::InvalidOverride {
                    key: crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY.to_owned(),
                    reason,
                })?
                .unwrap_or_default();
            if !self.is_cuda_backend()
                || self.execution_planning_model != ExecutionPlanningModel::CompiledPlan
                || !self.hardware.compiled_features.vllm_paged_attn
                || !self
                    .hardware
                    .compiled_features
                    .cuda_indexed_causal_attention
                || compiled_attention != Some(AttentionExecutionPolicy::NativeAdaptive)
                || preparation != crate::CausalDecodePreparationMode::Packed
            {
                return Err(AutoConfigError::InvalidOverride {
                    key: key.to_owned(),
                    reason: format!("{} causal decode attention grouping requires the cuda-indexed-causal-attention build feature, a CUDA plan runtime with native-adaptive paged attention, F16 KV storage, and packed causal decode preparation", grouping.as_runtime_value()),
                });
            }
        }
        Ok(self.decision(
            "causal_decode_attention_grouping",
            grouping.as_runtime_value(),
            source,
            source_key,
            [
                "contiguous",
                "indexed",
                "fixed-capacity",
                "fixed-execution-capacity",
            ],
            Vec::new(),
            vec![RuntimeConfigEffect::Performance],
        ))
    }

    fn sequence_backing_growth_decision(&self) -> Result<AutoConfigDecision, AutoConfigError> {
        let key = crate::SEQUENCE_BACKING_GROWTH_CONFIG_KEY;
        let invalid = |reason| AutoConfigError::InvalidOverride {
            key: key.to_owned(),
            reason,
        };
        let (mode, source, source_key) = match self.entry(key) {
            Some(entry) => (
                crate::SequenceBackingGrowth::parse_config_entry(entry).map_err(invalid)?,
                auto_config_source_from_runtime(entry.source),
                Some(key.to_owned()),
            ),
            None => (
                crate::SequenceBackingGrowth::Exact,
                AutoConfigSource::Default,
                None,
            ),
        };
        if mode != crate::SequenceBackingGrowth::Exact {
            let preparation = self
                .entry("FERRUM_INVOCATION_PREPARATION_STRATEGY")
                .map(|entry| {
                    crate::InvocationPreparationStrategy::parse_runtime_value(
                        &entry.effective_value,
                    )
                })
                .transpose()
                .map_err(invalid)?
                .unwrap_or_default();
            if !self.is_cuda_backend()
                || self.execution_planning_model != ExecutionPlanningModel::CompiledPlan
                || preparation != crate::InvocationPreparationStrategy::ResidentDecodeProgram
            {
                return Err(invalid(format!(
                    "{} requires a CUDA plan runtime and resident-decode-program preparation",
                    mode.as_runtime_value()
                )));
            }
        }
        Ok(self.decision(
            "sequence_backing_growth",
            mode.as_runtime_value(),
            source,
            source_key,
            ["exact", "paged-group4", "paged-group4-grow-only"],
            Vec::new(),
            vec![
                RuntimeConfigEffect::Performance,
                RuntimeConfigEffect::Memory,
            ],
        ))
    }

    fn causal_decode_kv_reuse_decision(&self) -> Result<AutoConfigDecision, AutoConfigError> {
        let key = crate::CAUSAL_DECODE_KV_REUSE_CONFIG_KEY;
        let invalid = |reason| AutoConfigError::InvalidOverride {
            key: key.to_owned(),
            reason,
        };
        let (mode, source, source_key) = match self.entry(key) {
            Some(entry) => (
                crate::CausalDecodeKvReuse::parse_config_entry(entry).map_err(invalid)?,
                auto_config_source_from_runtime(entry.source),
                Some(key.to_owned()),
            ),
            None => (
                crate::CausalDecodeKvReuse::None,
                AutoConfigSource::Default,
                None,
            ),
        };
        if mode != crate::CausalDecodeKvReuse::None {
            let grouping = self
                .entry(crate::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY)
                .map(crate::CausalDecodeAttentionGrouping::parse_config_entry)
                .transpose()
                .map_err(invalid)?
                .unwrap_or_default();
            // The grouping decision above checks packed preparation, native
            // attention, F16 KV and the CUDA plan-runtime authority.
            if !self.hardware.compiled_features.cuda_causal_gqa2
                || grouping != crate::CausalDecodeAttentionGrouping::FixedExecutionCapacity
            {
                return Err(invalid("paired-query-heads KV reuse requires the cuda-causal-gqa2 build feature and fixed-execution-capacity causal decode grouping".into()));
            }
        }
        Ok(self.decision(
            "causal_decode_kv_reuse",
            mode.as_runtime_value(),
            source,
            source_key,
            [
                "none",
                "paired-query-heads",
                "paired-query-heads-unified-v2",
            ],
            Vec::new(),
            vec![RuntimeConfigEffect::Performance],
        ))
    }

    fn q4_mmvq_weight_decode_decision(&self) -> Result<AutoConfigDecision, AutoConfigError> {
        let key = crate::Q4_MMVQ_WEIGHT_DECODE_CONFIG_KEY;
        let (mode, source, source_key) = match self.entry(key) {
            Some(entry) => (
                crate::Q4MmvqWeightDecode::parse_config_entry(entry).map_err(|reason| {
                    AutoConfigError::InvalidOverride {
                        key: key.to_owned(),
                        reason,
                    }
                })?,
                auto_config_source_from_runtime(entry.source),
                Some(key.to_owned()),
            ),
            None => (
                crate::Q4MmvqWeightDecode::default(),
                AutoConfigSource::Default,
                None,
            ),
        };
        if matches!(
            mode,
            crate::Q4MmvqWeightDecode::SharedM8 | crate::Q4MmvqWeightDecode::SharedM4M8
        ) && (!self.is_cuda_backend()
            || self.execution_planning_model != ExecutionPlanningModel::CompiledPlan
            || !self.hardware.compiled_features.cuda_q4_shared_weight_decode)
        {
            return Err(AutoConfigError::InvalidOverride {
                key: key.to_owned(),
                reason: "shared Q4 weight decode requires the cuda-q4-shared-weight-decode build feature, a CUDA plan runtime, and a selected Q4_K MarkerV2 MMVQ numerical route for the requested rows".to_owned(),
            });
        }
        Ok(self.decision(
            "q4_mmvq_weight_decode",
            mode.as_runtime_value(),
            source,
            source_key,
            ["auto", "baseline", "shared-m8", "shared-m4-m8"],
            Vec::new(),
            vec![RuntimeConfigEffect::Performance],
        ))
    }
    fn q5_mmvq_weight_decode_decision(&self) -> Result<AutoConfigDecision, AutoConfigError> {
        let key = crate::Q5_MMVQ_WEIGHT_DECODE_CONFIG_KEY;
        let (mode, source, source_key) = match self.entry(key) {
            Some(entry) => (
                crate::Q5MmvqWeightDecode::parse_config_entry(entry).map_err(|reason| {
                    AutoConfigError::InvalidOverride {
                        key: key.to_owned(),
                        reason,
                    }
                })?,
                auto_config_source_from_runtime(entry.source),
                Some(key.to_owned()),
            ),
            None => (
                crate::Q5MmvqWeightDecode::default(),
                AutoConfigSource::Default,
                None,
            ),
        };
        if matches!(
            mode,
            crate::Q5MmvqWeightDecode::SharedM8 | crate::Q5MmvqWeightDecode::SharedM4M8
        ) && (!self.is_cuda_backend()
            || self.execution_planning_model != ExecutionPlanningModel::CompiledPlan
            || !self.hardware.compiled_features.cuda_q5_shared_weight_decode)
        {
            return Err(AutoConfigError::InvalidOverride {
                key: key.to_owned(),
                reason: "shared Q5 weight decode requires the cuda-q5-shared-weight-decode build feature, a CUDA plan runtime, and a selected Q5_K MarkerV2 MMVQ numerical route for the requested rows".to_owned(),
            });
        }
        Ok(self.decision(
            "q5_mmvq_weight_decode",
            mode.as_runtime_value(),
            source,
            source_key,
            ["auto", "baseline", "shared-m8", "shared-m4-m8"],
            Vec::new(),
            vec![RuntimeConfigEffect::Performance],
        ))
    }

    fn q8_projection_weight_preparation_decision(
        &self,
    ) -> Result<AutoConfigDecision, AutoConfigError> {
        let key = crate::Q8_PROJECTION_WEIGHT_PREPARATION_CONFIG_KEY;
        let (mode, source, source_key) = match self.entry(key) {
            Some(entry) => (
                crate::Q8ProjectionWeightPreparation::parse_config_entry(entry).map_err(
                    |reason| AutoConfigError::InvalidOverride {
                        key: key.to_owned(),
                        reason,
                    },
                )?,
                auto_config_source_from_runtime(entry.source),
                Some(key.to_owned()),
            ),
            None => (
                crate::Q8ProjectionWeightPreparation::default(),
                AutoConfigSource::Default,
                None,
            ),
        };
        if mode == crate::Q8ProjectionWeightPreparation::PredecodedF32
            && (!self.is_cuda_backend()
                || !self.hardware.compiled_features.cuda
                || self.execution_planning_model != ExecutionPlanningModel::CompiledPlan)
        {
            return Err(AutoConfigError::InvalidOverride {
                key: key.to_owned(),
                reason: "predecoded-f32 Q8 projection weights require a CUDA plan runtime and a supported strict Q8 projection numerical route".to_owned(),
            });
        }
        Ok(self.decision(
            "q8_projection_weight_preparation",
            mode.as_runtime_value(),
            source,
            source_key,
            ["auto", "compressed", "predecoded-f32"],
            Vec::new(),
            vec![RuntimeConfigEffect::Performance],
        ))
    }

    fn prefill_decode_execution_decision(&self) -> Result<AutoConfigDecision, AutoConfigError> {
        let key = "FERRUM_PREFILL_DECODE_EXECUTION";
        let (policy, source, source_key) = match self.entry(key) {
            Some(entry) => (
                crate::PrefillDecodeExecution::parse_runtime_value(&entry.effective_value)
                    .map_err(|reason| AutoConfigError::InvalidOverride {
                        key: key.to_owned(),
                        reason,
                    })?,
                auto_config_source_from_runtime(entry.source),
                Some(key.to_owned()),
            ),
            None => (
                crate::PrefillDecodeExecution::default(),
                AutoConfigSource::Default,
                None,
            ),
        };
        Ok(self.decision(
            "prefill_decode_execution",
            policy.as_runtime_value(),
            source,
            source_key,
            ["split", "mixed"],
            Vec::new(),
            vec![RuntimeConfigEffect::Performance],
        ))
    }

    fn scheduler_decision(
        &self,
        default_prefill_first_until_active: Option<ResolvedValue<usize>>,
        default_active_decode_prefill_chunk: Option<ResolvedValue<usize>>,
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
        let explicit_active_decode_chunk = entries.get("FERRUM_ACTIVE_DECODE_PREFILL_CHUNK");
        let implicit_active_decode_chunk = default_active_decode_prefill_chunk
            .as_ref()
            .map(|chunk| chunk.value.to_string());
        let active_decode_chunk = explicit_active_decode_chunk
            .copied()
            .or(implicit_active_decode_chunk.as_deref());
        let active_decode_chunk_source = if explicit_active_decode_chunk.is_some() {
            let key = "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK";
            (
                self.source_for_key(key, AutoConfigSource::Default),
                Some(key.to_owned()),
            )
        } else {
            (AutoConfigSource::Default, None)
        };
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
        let prefill_step_chunk = explicit_prefill_step_chunk.copied();
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
            let (source, source_key) =
                if explicit_active_decode_chunk.is_none() && explicit_prefill_first_present {
                    let key = "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE";
                    (
                        self.source_for_key(key, AutoConfigSource::Default),
                        Some(key.to_owned()),
                    )
                } else {
                    active_decode_chunk_source.clone()
                };
            (
                format!("prefill_first_until_active:{until}+active_decode_prefill_chunk:{chunk}"),
                source,
                source_key,
            )
        } else if let Some(chunk) = active_decode_chunk {
            parse_usize_env_value(chunk).map_err(|reason| AutoConfigError::InvalidOverride {
                key: "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK".to_string(),
                reason,
            })?;
            (
                format!("active_decode_prefill_chunk:{chunk}"),
                active_decode_chunk_source.0,
                active_decode_chunk_source.1,
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
        } else if default_prefill_first_until_active.is_some()
            && !entries.contains_key("FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE")
        {
            selected.push_str("+prefill_token_budget:elastic");
        }
        let budget_key = "FERRUM_ACTIVE_DECODE_PREFILL_TOKEN_BUDGET";
        if let Some(value) = entries.get(budget_key) {
            let budget = parse_usize_env_value(value).map_err(|reason| {
                AutoConfigError::InvalidOverride {
                    key: budget_key.to_owned(),
                    reason,
                }
            })?;
            if budget == 0 {
                selected.push_str("+active_decode_prefill_token_budget:disabled");
            } else {
                selected.push_str(&format!("+active_decode_prefill_token_budget:{budget}"));
            }
            if source_key.is_none() {
                source = self.source_for_key(budget_key, AutoConfigSource::Default);
                source_key = Some(budget_key.to_owned());
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
                "active_decode_prefill_token_budget",
                "prefill_step_chunk",
                "prefill_token_budget:elastic",
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
            .with_selected_kv_cache_storage(Some(crate::KvCacheStorage::F16))
            .with_selected_kv_cache_layout(SelectedKvCacheLayout::Paged {
                block_size_tokens: 16,
            })
    }

    fn m3_with_hardware(
        vars: &[(&str, &str)],
        hardware: HardwareCapabilities,
    ) -> FerrumConfigBuilder {
        FerrumConfigBuilder::new(snapshot(vars))
            .with_model_capabilities(ModelCapabilities::qwen3_30b_a3b_gptq_int4())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(WorkloadProfile::m3_qwen3_30b_a3b_int4())
            .with_selected_kv_cache_storage(Some(crate::KvCacheStorage::F16))
            .with_selected_kv_cache_layout(SelectedKvCacheLayout::Paged {
                block_size_tokens: 16,
            })
    }

    fn qwen35_moe_gptq_int4_model() -> ModelCapabilities {
        let mut model = ModelCapabilities::qwen3_30b_a3b_gptq_int4();
        model.architecture = "qwen3_5_moe".to_string();
        model.head_dim = Some(256);
        model.num_hidden_layers = Some(40);
        model.kv_heads = Some(8);
        model.estimated_weight_bytes = Some(24_419_939_760);
        model.recurrent_state_bytes_per_sequence = Some(32_931_840);
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
    fn retired_native_overrides_are_rejected_for_every_executor() {
        for key in [
            "FERRUM_MARLIN_TILE",
            "FERRUM_VLLM_MOE_THREAD_K",
            "FERRUM_VLLM_MOE_THREAD_N",
            "FERRUM_VLLM_MOE_BLOCKS_PER_SM",
        ] {
            for value in ["", "0", "-1", "128"] {
                for authority in [
                    ExecutionPlanningModel::ExecutorManaged,
                    ExecutionPlanningModel::CompiledPlan,
                ] {
                    let error = FerrumConfigBuilder::new(snapshot(&[(key, value)]))
                        .with_execution_planning_model(authority)
                        .resolve()
                        .expect_err("native overrides must not affect executor selection");
                    assert!(matches!(error,
                        AutoConfigError::InvalidOverride { key: actual, reason }
                        if actual == key && reason.contains("retired")
                    ));
                }
            }
        }
    }

    #[test]
    fn retired_execution_controls_are_rejected_for_every_executor() {
        for key in [
            "FERRUM_WHOLE_PROMPT_PREFIX_CACHE",
            "FERRUM_SPEC_DRAFT",
            "FERRUM_SPEC_N",
        ] {
            for value in ["", "false", "0", "1"] {
                for authority in [
                    ExecutionPlanningModel::ExecutorManaged,
                    ExecutionPlanningModel::CompiledPlan,
                ] {
                    let error = FerrumConfigBuilder::new(snapshot(&[(key, value)]))
                        .with_execution_planning_model(authority)
                        .resolve()
                        .expect_err("retired controls must not select ordinary decoding");
                    assert!(matches!(error,
                        AutoConfigError::InvalidOverride { key: actual, reason }
                        if actual == key && reason.contains("retired")
                    ));
                }
            }
        }
    }

    #[test]
    fn retired_legacy_moe_controls_are_rejected_for_every_executor() {
        for key in [
            "FERRUM_MOE_GRAPH",
            "FERRUM_MOE_DEVICE_ROUTE",
            "FERRUM_MOE_HOST_TOPK",
            "FERRUM_MOE_FUSED_GATE_UP_SILU",
            "FERRUM_MOE_DIRECT_DISPATCH",
            "FERRUM_MOE_BATCHED",
            "FERRUM_MOE_BATCHED_DECODE",
            "FERRUM_MOE_BATCH_THRESHOLD",
            "FERRUM_MOE_BUCKETED",
            "FERRUM_MOE_PREFILL_THRESHOLD",
            "FERRUM_QWEN_UNIFIED_PREFILL",
            "FERRUM_QWEN_UNIFIED_TRACE",
            "FERRUM_UNIFIED_GREEDY_ARGMAX",
            "FERRUM_UNIFIED_LAYER_PROF",
            "FERRUM_UNIFIED_LAYER_PROF_EVERY",
            "FERRUM_UNIFIED_LAYER_PROF_MAX_M",
            "FERRUM_UNIFIED_LAYER_PROF_MIN_SEQS",
            "FERRUM_VLLM_DECODE_VARLEN",
            "FERRUM_VLLM_VARLEN_TILED_Q4",
        ] {
            for value in ["0", "1"] {
                for authority in [
                    ExecutionPlanningModel::ExecutorManaged,
                    ExecutionPlanningModel::CompiledPlan,
                ] {
                    let error = FerrumConfigBuilder::new(snapshot(&[(key, value)]))
                        .with_execution_planning_model(authority)
                        .resolve()
                        .expect_err("removed legacy controls cannot silently apply to typed MoE");
                    assert!(matches!(error,
                        AutoConfigError::InvalidOverride { key: actual, reason }
                        if actual == key && reason.contains("retired")
                    ));
                }
            }
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
        assert_eq!(decisions["moe_dispatch_preference"], "vllm_marlin_enabled");
        assert!(!decisions.contains_key("moe_graph_policy"));
        assert!(resolved.runtime_config.entries.iter().all(|entry| {
            entry.key != "FERRUM_MOE_DEVICE_ROUTE" && entry.key != "FERRUM_MOE_HOST_TOPK"
        }));
        assert!(resolved.runtime_config.entries.iter().any(|entry| {
            entry.key == "FERRUM_VLLM_MOE_PAIR_IDS" && entry.effective_value == "1"
        }));
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
            decisions["moe_dispatch_preference"], "vllm_marlin_enabled",
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
    fn recurrent_state_estimate_does_not_create_a_slot_pool_or_admission_cap() {
        for model in [
            synthetic_tight_recurrent_state_model(),
            qwen35_moe_gptq_int4_model(),
        ] {
            for authority in [
                ExecutionPlanningModel::ExecutorManaged,
                ExecutionPlanningModel::CompiledPlan,
            ] {
                let hardware = HardwareCapabilities::rtx4090_cuda(
                    CompiledKernelFeatures::m3_fast_path_without_fa2(),
                );
                let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
                let expected_estimate = model.recurrent_state_bytes_per_sequence;
                let resolved =
                    FerrumConfigBuilder::new(snapshot(&[("FERRUM_PAGED_MAX_SEQS", "32")]))
                        .with_model_capabilities(model.clone())
                        .with_hardware_capabilities(hardware)
                        .with_workload_profile(workload)
                        .with_execution_planning_model(authority)
                        .resolve()
                        .unwrap();
                assert!(resolved
                    .decisions
                    .iter()
                    .all(|decision| decision.selection != "recurrent_state_max_slots"));
                assert!(resolved
                    .runtime_config
                    .entries
                    .iter()
                    .all(|entry| entry.key != "FERRUM_RECURRENT_STATE_MAX_SLOTS"));
                let doc = resolved.effective_config_document();
                assert_eq!(doc["selected_max_sequences"], 32);
                assert_eq!(doc["selected_admission_limit"], 32);
                assert_eq!(doc["admission"]["effective_max_concurrent"], 32);
                assert!(doc.get("selected_recurrent_state_max_slots").is_none());
                for key in [
                    "recurrent_state_max_slots",
                    "legacy_recurrent_state_max_slots_estimate",
                    "legacy_recurrent_state_limit_applies",
                ] {
                    assert!(doc["admission"].get(key).is_none(), "{key}");
                }
                let memory = &doc["admission"]["memory_estimate"];
                assert_eq!(
                    memory["recurrent_state_bytes_per_sequence"],
                    serde_json::json!(expected_estimate)
                );
                assert_eq!(memory["is_resident_usage"], false);
                for key in [
                    "recurrent_state_budget_bytes",
                    "recurrent_state_budget_raw_slots",
                    "recurrent_state_budget_max_slots",
                    "recurrent_state_capacity_bytes",
                ] {
                    assert!(memory.get(key).is_none(), "{key}");
                }
            }
        }
    }

    #[test]
    fn auto_config_rejects_retired_recurrent_state_slots_for_each_authority() {
        for authority in [
            ExecutionPlanningModel::ExecutorManaged,
            ExecutionPlanningModel::CompiledPlan,
        ] {
            for value in ["", "0", "16", "32"] {
                let hardware = HardwareCapabilities::rtx4090_cuda(
                    CompiledKernelFeatures::m3_fast_path_without_fa2(),
                );
                let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
                let error = FerrumConfigBuilder::new(snapshot(&[
                    ("FERRUM_PAGED_MAX_SEQS", "32"),
                    ("FERRUM_RECURRENT_STATE_MAX_SLOTS", value),
                ]))
                .with_model_capabilities(synthetic_tight_recurrent_state_model())
                .with_hardware_capabilities(hardware)
                .with_workload_profile(workload)
                .with_execution_planning_model(authority)
                .resolve()
                .unwrap_err();
                assert!(matches!(error,
                    AutoConfigError::InvalidOverride { key, reason }
                    if key == "FERRUM_RECURRENT_STATE_MAX_SLOTS" && reason.contains("retired")
                ));
            }
        }
    }

    #[test]
    fn plan_runtime_uses_dynamic_admission_and_typed_attention_authority() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot(&[("FERRUM_PAGED_MAX_SEQS", "32")]))
            .with_model_capabilities(synthetic_tight_recurrent_state_model())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .with_execution_planning_model(ExecutionPlanningModel::CompiledPlan)
            .resolve()
            .unwrap();

        assert_eq!(
            resolved.execution_planning_model,
            ExecutionPlanningModel::CompiledPlan
        );
        assert_eq!(
            resolved.requested_attention_policy,
            Some(AttentionExecutionPolicy::Auto)
        );
        assert_eq!(
            resolved.compiled_attention_policy,
            Some(AttentionExecutionPolicy::NativeAdaptive)
        );
        assert!(resolved
            .decisions
            .iter()
            .all(|decision| decision.selection != "recurrent_state_max_slots"));
        assert!(resolved
            .runtime_config
            .entries
            .iter()
            .all(|entry| !matches!(
                entry.key.as_str(),
                "FERRUM_RECURRENT_STATE_MAX_SLOTS"
                    | "FERRUM_USE_VLLM_PAGED_ATTN"
                    | "FERRUM_VLLM_PAGED_ATTN_V1_SHORT"
            )));

        let document = resolved.effective_config_document();
        assert_eq!(document["selected_max_sequences"], serde_json::json!(32));
        assert_eq!(document["selected_admission_limit"], serde_json::json!(32));
        assert_eq!(
            document["selected_attention_policy"],
            serde_json::json!("native-adaptive")
        );
        assert_eq!(
            document["attention_execution"]["requested_policy"],
            serde_json::json!("auto")
        );
        assert_eq!(
            document["attention_execution"]["compiled_policy"],
            serde_json::json!("native-adaptive")
        );
        assert_eq!(
            document["attention_execution"]["legacy_attention_keys_apply"],
            serde_json::json!(false)
        );
        assert_eq!(
            document["admission"]["resource_authority"],
            serde_json::json!("plan_runtime")
        );
        assert_eq!(
            document["admission"]["effective_max_concurrent"],
            serde_json::json!(32)
        );
        assert!(document["admission"]
            .get("legacy_recurrent_state_limit_applies")
            .is_none());
    }

    #[test]
    fn plan_runtime_preflight_labels_legacy_estimates_without_breaking_capacity_consumers() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let model = synthetic_tight_recurrent_state_model();
        for authority in [
            ExecutionPlanningModel::CompiledPlan,
            ExecutionPlanningModel::ExecutorManaged,
        ] {
            let document = FerrumConfigBuilder::new(snapshot(&[
                ("FERRUM_KV_DTYPE", "int8"),
                ("FERRUM_KV_CAPACITY", "512"),
            ]))
            .with_model_capabilities(model.clone())
            .with_hardware_capabilities(hardware.clone())
            .with_execution_planning_model(authority)
            .resolve()
            .unwrap()
            .effective_config_document();
            assert_eq!(document["selected_kv_capacity"], 512);
            assert_eq!(
                document["selected_kv_capacity_source"],
                "configured_token_limit"
            );
            let admission = &document["admission"];
            for field in [
                "kv_block_count",
                "kv_block_size_tokens",
                "kv_capacity_tokens",
            ] {
                assert!(
                    admission[field].as_u64().is_some(),
                    "missing compatibility field {field}"
                );
            }
            assert_eq!(
                admission["kv_capacity_source"],
                "legacy_block_preflight_estimate"
            );
            assert_eq!(admission["kv_capacity_is_resident_usage"], false);
            assert_eq!(
                admission["kv_capacity_applies_to_selected_state_layout"],
                false
            );
            let estimate = &admission["memory_estimate"];
            assert!(estimate["kv_bytes_per_token"].is_null());
            assert_eq!(estimate["source"], "unqualified_storage_geometry");
            assert_eq!(estimate["applies_to_selected_state_layout"], false);
            assert_eq!(estimate["is_resident_usage"], false);
            if authority == ExecutionPlanningModel::CompiledPlan {
                assert_eq!(
                    estimate["selected_state_evidence_source"],
                    "executor.kv_storage.logical_sequence_state"
                );
            }
        }
    }

    #[test]
    fn plan_runtime_int8_implicit_and_explicit_auto_resolve_to_compatible_attention() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let resolve = |dtype, attention: Option<&str>| {
            let mut entries = vec![("FERRUM_KV_DTYPE", dtype)];
            if let Some(attention) = attention {
                entries.push(("FERRUM_ATTENTION_POLICY", attention));
            }
            FerrumConfigBuilder::new(snapshot(&entries))
                .with_hardware_capabilities(hardware.clone())
                .with_execution_planning_model(ExecutionPlanningModel::CompiledPlan)
                .resolve()
        };
        for requested in [None, Some("auto"), Some("portable")] {
            let resolved = resolve("int8", requested).unwrap();
            assert_eq!(
                resolved.compiled_attention_policy,
                Some(AttentionExecutionPolicy::Portable)
            );
            let mut engine = crate::EngineConfig::default();
            engine
                .apply_runtime_config_snapshot(&resolved.runtime_config)
                .unwrap();
            assert_eq!(
                engine
                    .runtime
                    .attention_execution_policy
                    .resolve(false)
                    .unwrap(),
                AttentionExecutionPolicy::Portable
            );
            assert_eq!(
                resolved.requested_attention_policy,
                Some(if requested == Some("portable") {
                    AttentionExecutionPolicy::Portable
                } else {
                    AttentionExecutionPolicy::Auto
                })
            );
        }
        assert!(
            matches!(resolve("int8", Some("native-adaptive")), Err(AutoConfigError::UnsupportedCombination {
            selection, reason,
        }) if selection == "attention_execution_policy" && reason.contains("INT8 KV"))
        );
        assert_eq!(
            resolve("fp16", None).unwrap().compiled_attention_policy,
            Some(AttentionExecutionPolicy::NativeAdaptive)
        );
    }

    #[test]
    fn selected_kv_layout_applies_block_limits_only_to_actual_paged_geometry() {
        let mut model = ModelCapabilities::unknown();
        model.max_context_len = Some(32);
        let resolve = |layout, batch, window| {
            FerrumConfigBuilder::new(snapshot(&[
                ("FERRUM_KV_MAX_BLOCKS", "1"),
                ("FERRUM_PAGED_MAX_SEQS", "1"),
                ("FERRUM_MAX_BATCHED_TOKENS", batch),
                ("FERRUM_MAX_MODEL_LEN", window),
            ]))
            .with_model_capabilities(model.clone())
            .with_selected_kv_cache_layout(layout)
            .resolve()
        };
        for layout in [
            SelectedKvCacheLayout::Contiguous,
            SelectedKvCacheLayout::Unknown,
        ] {
            let resolved = resolve(layout, "32", "32").unwrap();
            assert_eq!(resolved.selected_usize("max_model_len"), Some(32));
            assert_eq!(resolved.selected_usize("max_batched_tokens"), Some(32));
            assert_eq!(resolved.selected_kv_cache_layout, layout);
            assert_eq!(
                resolved.admission_summary_document()
                    ["kv_capacity_applies_to_selected_state_layout"],
                false
            );
        }
        let paged = SelectedKvCacheLayout::Paged {
            block_size_tokens: 16,
        };
        assert!(
            matches!(resolve(paged, "32", "16"), Err(AutoConfigError::InvalidOverride { key, .. }) if key == "FERRUM_MAX_BATCHED_TOKENS")
        );
        assert!(
            matches!(resolve(paged, "16", "32"), Err(AutoConfigError::InvalidOverride { key, .. }) if key == "FERRUM_KV_MAX_BLOCKS")
        );
        let actual = resolve(paged, "16", "16").unwrap();
        assert_eq!(
            actual.admission_summary_document()["kv_capacity_tokens"],
            16
        );
        assert_eq!(
            actual.admission_summary_document()["kv_capacity_applies_to_selected_state_layout"],
            true
        );
        // The selected allocator's geometry is the authority, not a hardcoded 16.
        let wider = resolve(
            SelectedKvCacheLayout::Paged {
                block_size_tokens: 32,
            },
            "32",
            "32",
        )
        .unwrap();
        assert_eq!(wider.admission_summary_document()["kv_capacity_tokens"], 32);
        assert!(resolve(
            SelectedKvCacheLayout::Paged {
                block_size_tokens: 0
            },
            "1",
            "1"
        )
        .is_err());
    }

    #[test]
    fn selected_kv_layout_metadata_uses_same_selection_instead_of_backend_label() {
        let mut hardware = HardwareCapabilities::unknown();
        hardware.backend = "metal".into();
        for (layout, label) in [
            (SelectedKvCacheLayout::Unknown, "unknown"),
            (SelectedKvCacheLayout::Contiguous, "contiguous"),
            (
                SelectedKvCacheLayout::Paged {
                    block_size_tokens: 16,
                },
                "paged",
            ),
        ] {
            let resolved = FerrumConfigBuilder::new(snapshot(&[]))
                .with_hardware_capabilities(hardware.clone())
                .with_selected_kv_cache_layout(layout)
                .resolve()
                .unwrap();
            let document = resolved.effective_config_document();
            assert_eq!(document["selected_kv_layout"], label);
            assert_eq!(
                document["admission"]["selected_kv_cache_layout"],
                serde_json::to_value(layout).unwrap()
            );
        }
    }

    #[test]
    fn selected_kv_layout_default_workload_target_does_not_cap_contiguous_work() {
        for (layout, expected) in [
            (SelectedKvCacheLayout::Contiguous, 64),
            (SelectedKvCacheLayout::Unknown, 64),
            (
                SelectedKvCacheLayout::Paged {
                    block_size_tokens: 16,
                },
                16,
            ),
        ] {
            let resolved = FerrumConfigBuilder::new(snapshot(&[
                ("FERRUM_KV_MAX_BLOCKS", "1"),
                ("FERRUM_PAGED_MAX_SEQS", "1"),
            ]))
            .with_selected_kv_cache_layout(layout)
            .resolve()
            .unwrap();
            assert_eq!(
                resolved.selected_usize("max_batched_tokens"),
                Some(expected)
            );
        }
    }

    #[test]
    fn plan_runtime_context_and_batch_limits_do_not_inherit_legacy_kv_block_estimates() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let mut model = ModelCapabilities::unknown();
        model.max_context_len = Some(131_072);
        // A tiny legacy block count is not vNext capacity authority. This is
        // preflight acceptance only: native admission still charges both INT8
        // payload and scales, physical alignment and operation workspace.
        for max_batched in ["16", "32"] {
            let entries = snapshot(&[
                ("FERRUM_KV_DTYPE", "int8"),
                ("FERRUM_KV_MAX_BLOCKS", "1"),
                ("FERRUM_PAGED_MAX_SEQS", "1"),
                ("FERRUM_MAX_BATCHED_TOKENS", max_batched),
                ("FERRUM_MAX_MODEL_LEN", "65536"),
            ]);
            let builder = FerrumConfigBuilder::new(entries.clone())
                .with_hardware_capabilities(hardware.clone())
                .with_model_capabilities(model.clone())
                .with_selected_kv_cache_layout(SelectedKvCacheLayout::Paged {
                    block_size_tokens: 16,
                });
            assert!(
                builder.resolve().is_err(),
                "selected paged capacity must remain enforced"
            );
            let resolved = FerrumConfigBuilder::new(entries)
                .with_hardware_capabilities(hardware.clone())
                .with_model_capabilities(model.clone())
                .with_execution_planning_model(ExecutionPlanningModel::CompiledPlan)
                .resolve()
                .unwrap();
            let mut engine = crate::EngineConfig::default();
            engine
                .apply_runtime_config_snapshot(&resolved.runtime_config)
                .unwrap();
            assert_eq!(engine.runtime.max_model_len, Some(65_536));
            assert_eq!(
                engine.batching.max_num_batched_tokens,
                max_batched.parse::<usize>().unwrap()
            );
        }
        let invalid = FerrumConfigBuilder::new(snapshot(&[("FERRUM_MAX_MODEL_LEN", "131073")]))
            .with_hardware_capabilities(hardware)
            .with_model_capabilities(model)
            .with_execution_planning_model(ExecutionPlanningModel::CompiledPlan)
            .resolve()
            .unwrap_err();
        assert!(
            matches!(invalid, AutoConfigError::InvalidOverride { key, .. } if key == "FERRUM_MAX_MODEL_LEN")
        );
    }

    #[test]
    fn plan_runtime_rejects_legacy_attention_overrides() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        for key in [
            "FERRUM_USE_VLLM_PAGED_ATTN",
            "FERRUM_VLLM_PAGED_ATTN_V1_SHORT",
        ] {
            let error = FerrumConfigBuilder::new(snapshot(&[(key, "0")]))
                .with_model_capabilities(synthetic_tight_recurrent_state_model())
                .with_hardware_capabilities(hardware.clone())
                .with_workload_profile(workload.clone())
                .with_execution_planning_model(ExecutionPlanningModel::CompiledPlan)
                .resolve()
                .unwrap_err();
            assert!(matches!(
                error,
                AutoConfigError::InvalidOverride {
                    key: rejected_key,
                    reason,
                } if rejected_key == key && reason.contains("FERRUM_ATTENTION_POLICY")
            ));
        }
    }

    #[test]
    fn metal_plan_runtime_preflight_matches_portable_runtime_capability() {
        let mut hardware = HardwareCapabilities::unknown();
        hardware.backend = "metal".to_owned();
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot(&[]))
            .with_model_capabilities(synthetic_tight_recurrent_state_model())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .with_execution_planning_model(ExecutionPlanningModel::CompiledPlan)
            .resolve()
            .unwrap();

        assert_eq!(
            resolved.compiled_attention_policy,
            Some(AttentionExecutionPolicy::Portable)
        );
        assert_eq!(
            resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == "attention_decode_backend")
                .map(|decision| decision.selected.as_str()),
            Some("portable")
        );
    }

    #[test]
    fn removed_qwen35_slot_alias_does_not_create_a_recurrent_slot_cap() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot(&[
            ("FERRUM_PAGED_MAX_SEQS", "32"),
            ("FERRUM_QWEN35_LINEAR_STATE_MAX_SLOTS", "7"),
        ]))
        .with_model_capabilities(synthetic_tight_recurrent_state_model())
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .unwrap();
        assert_eq!(
            resolved.effective_config_document()["selected_admission_limit"],
            32
        );
        assert!(resolved
            .decisions
            .iter()
            .all(|decision| decision.selection != "recurrent_state_max_slots"));
        assert!(resolved
            .runtime_config
            .entries
            .iter()
            .all(|entry| entry.key != "FERRUM_RECURRENT_STATE_MAX_SLOTS"));
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
            "prefill_first_until_active:16+prefill_token_budget:elastic"
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
    fn source_fa2_is_rejected_even_when_legacy_feature_is_compiled() {
        let err = m3(
            &[("FERRUM_FA2_SOURCE", "1")],
            CompiledKernelFeatures::m3_fast_path_with_source_fa2(),
        )
        .resolve()
        .unwrap_err();

        match err {
            AutoConfigError::InvalidOverride { key, reason } => {
                assert_eq!(key, "FERRUM_FA2_SOURCE");
                assert!(reason.contains("native operator artifact"));
            }
            AutoConfigError::UnsupportedCombination { .. } => {
                panic!("expected invalid FERRUM_FA2_SOURCE override")
            }
        }
    }

    #[test]
    fn source_fa2_is_rejected_when_not_compiled() {
        expect_invalid_key(&[("FERRUM_FA2_SOURCE", "1")], "FERRUM_FA2_SOURCE");
    }

    #[test]
    fn fa2_native_artifact_typed_config_is_recorded_without_selecting_runtime_path() {
        let resolved = m3(
            &[
                (
                    FA2_NATIVE_MANIFEST_KEY,
                    "/tmp/native/fa2/native_operator_manifest.json",
                ),
                (
                    FA2_NATIVE_ARTIFACT_KEY,
                    "/tmp/native/fa2/libferrum_native_fa2.a",
                ),
                (
                    FA2_NATIVE_SOURCE_SHA256_KEY,
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                ),
                (
                    FA2_NATIVE_INPUTS_SHA256_KEY,
                    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                ),
            ],
            CompiledKernelFeatures::m3_fast_path_with_native_fa2_artifact(),
        )
        .resolve()
        .unwrap();

        let decisions: BTreeMap<_, _> = resolved
            .decisions
            .iter()
            .map(|decision| (decision.selection.as_str(), decision.selected.as_str()))
            .collect();
        assert_eq!(decisions["fa2_native_operator_artifact"], "configured");
        assert_eq!(
            decisions["fa2_native_operator_runtime_selection"],
            "not_selected"
        );
        assert_ne!(decisions["attention_prefill_mixed_backend"], "fa2_native");
        assert!(resolved
            .runtime_config
            .entries
            .iter()
            .any(|entry| entry.key == FA2_NATIVE_MANIFEST_KEY
                && entry.source == RuntimeConfigSource::Env));
    }

    #[test]
    fn fa2_native_runtime_selection_is_explicit_when_not_configured() {
        let resolved = m3(&[], CompiledKernelFeatures::m3_fast_path_without_fa2())
            .resolve()
            .unwrap();
        let decision = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "fa2_native_operator_runtime_selection")
            .unwrap();
        assert_eq!(decision.selected, "not_configured");
        assert!(decision.rejected.iter().any(|candidate| {
            candidate.value == "selected"
                && candidate
                    .reason
                    .contains("artifact-selected actual model smoke")
        }));
    }

    #[test]
    fn fa2_native_artifact_requires_manifest_artifact_pair_and_valid_pins() {
        expect_invalid_key(
            &[(
                FA2_NATIVE_MANIFEST_KEY,
                "/tmp/native/fa2/native_operator_manifest.json",
            )],
            FA2_NATIVE_ARTIFACT_KEY,
        );
        expect_invalid_key(
            &[(
                FA2_NATIVE_ARTIFACT_KEY,
                "/tmp/native/fa2/libferrum_native_fa2.a",
            )],
            FA2_NATIVE_MANIFEST_KEY,
        );
        expect_invalid_key(
            &[
                (
                    FA2_NATIVE_MANIFEST_KEY,
                    "/tmp/native/fa2/native_operator_manifest.json",
                ),
                (
                    FA2_NATIVE_ARTIFACT_KEY,
                    "/tmp/native/fa2/libferrum_native_fa2.a",
                ),
                (FA2_NATIVE_SOURCE_SHA256_KEY, "not-a-sha"),
            ],
            FA2_NATIVE_SOURCE_SHA256_KEY,
        );
        expect_invalid_key(
            &[(
                FA2_NATIVE_INPUTS_SHA256_KEY,
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            )],
            FA2_NATIVE_INPUTS_SHA256_KEY,
        );
    }

    #[test]
    fn fa2_native_artifact_requires_linked_binary_capability() {
        expect_invalid_key(
            &[
                (
                    FA2_NATIVE_MANIFEST_KEY,
                    "/tmp/native/fa2/native_operator_manifest.json",
                ),
                (
                    FA2_NATIVE_ARTIFACT_KEY,
                    "/tmp/native/fa2/libferrum_native_fa2.a",
                ),
            ],
            FA2_NATIVE_MANIFEST_KEY,
        );
    }

    #[test]
    fn fa2_native_artifact_requires_linked_binary_metadata() {
        let mut features = CompiledKernelFeatures::m3_fast_path_without_fa2();
        features.fa2_native_operator_artifact = true;

        let err = m3(
            &[
                (
                    FA2_NATIVE_MANIFEST_KEY,
                    "/tmp/native/fa2/native_operator_manifest.json",
                ),
                (
                    FA2_NATIVE_ARTIFACT_KEY,
                    "/tmp/native/fa2/libferrum_native_fa2.a",
                ),
            ],
            features,
        )
        .resolve()
        .unwrap_err();
        match err {
            AutoConfigError::InvalidOverride { key, reason } => {
                assert_eq!(key, FA2_NATIVE_MANIFEST_KEY);
                assert!(reason.contains("missing build metadata"));
            }
            other => panic!("expected invalid override, got {other:?}"),
        }
    }

    #[test]
    fn fa2_native_artifact_must_match_linked_binary_metadata() {
        let features = CompiledKernelFeatures::m3_fast_path_with_native_fa2_artifact();
        expect_invalid_key_with_features(
            &[
                (
                    FA2_NATIVE_MANIFEST_KEY,
                    "/tmp/native/fa2/other_manifest.json",
                ),
                (
                    FA2_NATIVE_ARTIFACT_KEY,
                    "/tmp/native/fa2/libferrum_native_fa2.a",
                ),
            ],
            FA2_NATIVE_MANIFEST_KEY,
            features.clone(),
        );
        expect_invalid_key_with_features(
            &[
                (
                    FA2_NATIVE_MANIFEST_KEY,
                    "/tmp/native/fa2/native_operator_manifest.json",
                ),
                (
                    FA2_NATIVE_ARTIFACT_KEY,
                    "/tmp/native/fa2/libferrum_native_other.a",
                ),
            ],
            FA2_NATIVE_ARTIFACT_KEY,
            features.clone(),
        );
        expect_invalid_key_with_features(
            &[
                (
                    FA2_NATIVE_MANIFEST_KEY,
                    "/tmp/native/fa2/native_operator_manifest.json",
                ),
                (
                    FA2_NATIVE_ARTIFACT_KEY,
                    "/tmp/native/fa2/libferrum_native_fa2.a",
                ),
                (
                    FA2_NATIVE_SOURCE_SHA256_KEY,
                    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                ),
            ],
            FA2_NATIVE_SOURCE_SHA256_KEY,
            features.clone(),
        );
        expect_invalid_key_with_features(
            &[
                (
                    FA2_NATIVE_MANIFEST_KEY,
                    "/tmp/native/fa2/native_operator_manifest.json",
                ),
                (
                    FA2_NATIVE_ARTIFACT_KEY,
                    "/tmp/native/fa2/libferrum_native_fa2.a",
                ),
                (
                    FA2_NATIVE_INPUTS_SHA256_KEY,
                    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                ),
            ],
            FA2_NATIVE_INPUTS_SHA256_KEY,
            features,
        );
    }

    #[test]
    fn fa2_native_artifact_requires_cuda_backend() {
        let hardware =
            cpu_hardware_with_features(CompiledKernelFeatures::m3_fast_path_without_fa2());
        expect_invalid_key_with_hardware(
            &[
                (
                    FA2_NATIVE_MANIFEST_KEY,
                    "/tmp/native/fa2/native_operator_manifest.json",
                ),
                (
                    FA2_NATIVE_ARTIFACT_KEY,
                    "/tmp/native/fa2/libferrum_native_fa2.a",
                ),
            ],
            FA2_NATIVE_MANIFEST_KEY,
            hardware,
        );
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
        assert_eq!(decisions["moe_dispatch_preference"], "vllm_marlin_disabled");
        assert!(!decisions.contains_key("moe_graph_policy"));
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

        let builder = m3_with_hardware(&[], low_vram_gpu);
        let max_sequences = builder.default_max_sequences();
        assert_eq!(max_sequences.value, 4);
        assert_eq!(max_sequences.source, AutoConfigSource::HardwareCapability);
        // The width recommendation cannot make a model larger than the known
        // memory estimate fit. Startup must reject its impossible minimum pool.
        assert!(
            matches!(builder.resolve(), Err(AutoConfigError::InvalidOverride { key, reason }) if key == "FERRUM_MAX_MODEL_LEN" && reason.contains("cannot fit"))
        );
    }

    #[test]
    fn memory_budget_caps_pool_to_usable_context_and_constrained_vram() {
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
        // 32 live sequences with the selected 512-token default need 1024
        // blocks; the materialized pool must not grow to a diagnostic-only target.
        assert_eq!(decision("kv_block_count").selected, "1024");
        assert_eq!(
            decision("kv_block_count").source,
            AutoConfigSource::WorkloadPreset
        );

        let mut constrained =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        constrained.vram_bytes = Some(20 * 1024 * 1024 * 1024);
        let error = m3_with_hardware(&[], constrained).resolve().unwrap_err();
        assert!(
            matches!(error, AutoConfigError::InvalidOverride { key, reason }
            if key == "FERRUM_MAX_MODEL_LEN" && reason.contains("cannot fit"))
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
            "prefill_first_until_active:32+prefill_token_budget:elastic"
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
        assert!(resolved
            .runtime_config
            .entries
            .iter()
            .all(|entry| entry.key != "FERRUM_SCHED_PREFILL_STEP_CHUNK"));
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
    fn reusable_execution_defaults_on_without_enabling_legacy_graphs() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(RuntimeConfigSnapshot::default())
            .with_model_capabilities(ModelCapabilities::unknown())
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .resolve()
            .unwrap();
        let entry = |key: &str| {
            resolved
                .runtime_config
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key} entry"))
        };
        let decisions: BTreeMap<_, _> = resolved
            .decisions
            .iter()
            .map(|decision| (decision.selection.as_str(), decision.selected.as_str()))
            .collect();

        assert_eq!(entry("FERRUM_BATCHED_GRAPH").effective_value, "0");
        assert_eq!(entry("FERRUM_REUSABLE_EXECUTION").effective_value, "1");
        assert_eq!(decisions["decode_graph_policy"], "graph_disabled");
        assert_eq!(
            decisions["reusable_execution_policy"],
            "enabled_when_runtime_capable"
        );
    }

    #[test]
    fn reusable_execution_explicit_disable_is_preserved() {
        let hardware =
            HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2());
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
        let resolved = FerrumConfigBuilder::new(snapshot_with_sources(&[(
            "FERRUM_REUSABLE_EXECUTION",
            "0",
            RuntimeConfigSource::Cli,
        )]))
        .with_model_capabilities(ModelCapabilities::unknown())
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .unwrap();
        let entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_REUSABLE_EXECUTION")
            .expect("reusable execution entry");
        let decision = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "reusable_execution_policy")
            .expect("reusable execution decision");

        assert_eq!(entry.effective_value, "0");
        assert_eq!(entry.source, RuntimeConfigSource::Cli);
        assert_eq!(decision.selected, "disabled");
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
            "prefill_first_until_active:32+active_decode_prefill_chunk:64+prefill_token_budget:elastic"
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
        assert!(resolved
            .runtime_config
            .entries
            .iter()
            .all(|entry| entry.key != "FERRUM_SCHED_PREFILL_STEP_CHUNK"));
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

    fn scheduler_resolution(
        backend: &str,
        authority: ExecutionPlanningModel,
        sequences: usize,
        runtime_config: RuntimeConfigSnapshot,
    ) -> ResolvedFerrumConfig {
        let mut hardware = HardwareCapabilities::unknown();
        hardware.backend = backend.to_owned();
        let mut workload = WorkloadProfile::serving_default();
        workload.target_concurrency = sequences;
        FerrumConfigBuilder::new(runtime_config)
            .with_hardware_capabilities(hardware)
            .with_workload_profile(workload)
            .with_execution_planning_model(authority)
            .resolve()
            .unwrap()
    }

    #[test]
    fn prefill_decode_execution_records_split_default_and_explicit_sources() {
        let key = "FERRUM_PREFILL_DECODE_EXECUTION";
        for (value, runtime_source, source) in [
            (
                None,
                RuntimeConfigSource::Default,
                AutoConfigSource::Default,
            ),
            (
                Some("mixed"),
                RuntimeConfigSource::ConfigFile,
                AutoConfigSource::ConfigFile,
            ),
            (
                Some("mixed"),
                RuntimeConfigSource::Env,
                AutoConfigSource::Env,
            ),
            (
                Some("split"),
                RuntimeConfigSource::Cli,
                AutoConfigSource::Cli,
            ),
        ] {
            let snapshot = value
                .map(|value| snapshot_with_sources(&[(key, value, runtime_source)]))
                .unwrap_or_default();
            let resolved =
                scheduler_resolution("metal", ExecutionPlanningModel::CompiledPlan, 4, snapshot);
            let expected = value.unwrap_or("split");
            let entry = resolved
                .runtime_config
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap();
            assert_eq!(entry.effective_value, expected);
            assert_eq!(entry.source, runtime_source);
            let decision = resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == "prefill_decode_execution")
                .unwrap();
            assert_eq!(decision.selected, expected);
            assert_eq!(decision.source, source);
            assert_eq!(decision.source_key.as_deref(), value.map(|_| key));
            let mut engine = crate::EngineConfig::default();
            engine
                .apply_runtime_config_snapshot(&resolved.runtime_config)
                .unwrap();
            assert_eq!(
                engine.batching.prefill_decode_execution.as_runtime_value(),
                expected
            );
        }
        let error = FerrumConfigBuilder::new(snapshot(&[(key, "automatic")]))
            .resolve()
            .unwrap_err();
        assert!(
            matches!(error, AutoConfigError::InvalidOverride { key: actual, .. } if actual == key)
        );
    }

    #[test]
    fn sampling_readback_records_default_sources_and_keeps_sampling_path_independent() {
        let key = "FERRUM_SAMPLING_READBACK_MODE";
        let baseline = scheduler_resolution(
            "cpu",
            ExecutionPlanningModel::CompiledPlan,
            4,
            RuntimeConfigSnapshot::default(),
        );
        let baseline_path = baseline
            .decisions
            .iter()
            .find(|d| d.selection == "sampling_readback_path")
            .unwrap();
        for (value, runtime_source, source) in [
            (
                None,
                RuntimeConfigSource::Default,
                AutoConfigSource::Default,
            ),
            (
                Some("participant"),
                RuntimeConfigSource::ConfigFile,
                AutoConfigSource::ConfigFile,
            ),
            (
                Some("participant"),
                RuntimeConfigSource::Env,
                AutoConfigSource::Env,
            ),
            (
                Some("uniform"),
                RuntimeConfigSource::Cli,
                AutoConfigSource::Cli,
            ),
        ] {
            let snapshot = value
                .map(|v| snapshot_with_sources(&[(key, v, runtime_source)]))
                .unwrap_or_default();
            let resolved =
                scheduler_resolution("cpu", ExecutionPlanningModel::CompiledPlan, 4, snapshot);
            let expected = value.unwrap_or("uniform");
            let entry = resolved
                .runtime_config
                .entries
                .iter()
                .find(|e| e.key == key)
                .unwrap();
            assert_eq!(entry.effective_value, expected);
            assert_eq!(entry.source, runtime_source);
            let decision = resolved
                .decisions
                .iter()
                .find(|d| d.selection == "sampling_readback_mode")
                .unwrap();
            assert_eq!(decision.selected, expected);
            assert_eq!(decision.source, source);
            assert_eq!(decision.source_key.as_deref(), value.map(|_| key));
            let path = resolved
                .decisions
                .iter()
                .find(|d| d.selection == "sampling_readback_path")
                .unwrap();
            assert_eq!(path.selected, baseline_path.selected);
            let mut engine = crate::EngineConfig::default();
            engine
                .apply_runtime_config_snapshot(&resolved.runtime_config)
                .unwrap();
            assert_eq!(
                engine.runtime.sampling_readback_mode.as_runtime_value(),
                expected
            );
        }
        for (value, authority) in [
            ("automatic", ExecutionPlanningModel::CompiledPlan),
            ("participant", ExecutionPlanningModel::ExecutorManaged),
        ] {
            let error = FerrumConfigBuilder::new(snapshot(&[(key, value)]))
                .with_execution_planning_model(authority)
                .resolve()
                .unwrap_err();
            assert!(
                matches!(error, AutoConfigError::InvalidOverride { key: actual, .. } if actual == key)
            );
        }
    }

    #[test]
    fn invocation_preparation_records_full_default_and_explicit_sources() {
        let key = "FERRUM_INVOCATION_PREPARATION_STRATEGY";
        for (value, runtime_source, source) in [
            (
                Some("decode-segment"),
                RuntimeConfigSource::Cli,
                AutoConfigSource::Cli,
            ),
            (
                Some("decode-segment"),
                RuntimeConfigSource::ConfigFile,
                AutoConfigSource::ConfigFile,
            ),
            (
                None,
                RuntimeConfigSource::Default,
                AutoConfigSource::Default,
            ),
            (
                Some("identity-projection"),
                RuntimeConfigSource::ConfigFile,
                AutoConfigSource::ConfigFile,
            ),
            (
                Some("identity-projection"),
                RuntimeConfigSource::Env,
                AutoConfigSource::Env,
            ),
            (
                Some("full"),
                RuntimeConfigSource::Cli,
                AutoConfigSource::Cli,
            ),
        ] {
            let snapshot = value
                .map(|value| snapshot_with_sources(&[(key, value, runtime_source)]))
                .unwrap_or_default();
            let resolved =
                scheduler_resolution("metal", ExecutionPlanningModel::CompiledPlan, 4, snapshot);
            let expected = value.unwrap_or("full");
            let entry = resolved
                .runtime_config
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap();
            assert_eq!(entry.effective_value, expected);
            assert_eq!(entry.source, runtime_source);
            let decision = resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == "invocation_preparation_strategy")
                .unwrap();
            assert_eq!(decision.selected, expected);
            assert_eq!(decision.source, source);
            assert_eq!(decision.source_key.as_deref(), value.map(|_| key));
            let mut engine = crate::EngineConfig::default();
            engine
                .apply_runtime_config_snapshot(&resolved.runtime_config)
                .unwrap();
            assert_eq!(
                engine
                    .runtime
                    .invocation_preparation_strategy
                    .as_runtime_value(),
                expected
            );
        }
        let error = FerrumConfigBuilder::new(snapshot(&[(key, "automatic")]))
            .resolve()
            .unwrap_err();
        assert!(
            matches!(error, AutoConfigError::InvalidOverride { key: actual, .. } if actual == key)
        );
    }

    #[test]
    fn segment_owner_mode_reports_default_and_explicit_sources() {
        let key = crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY;
        for (mode, runtime_source, source) in [
            (
                None,
                RuntimeConfigSource::Default,
                AutoConfigSource::Default,
            ),
            (
                Some("indexed"),
                RuntimeConfigSource::ConfigFile,
                AutoConfigSource::ConfigFile,
            ),
            (
                Some("legacy"),
                RuntimeConfigSource::Cli,
                AutoConfigSource::Cli,
            ),
            (
                Some("indexed"),
                RuntimeConfigSource::Cli,
                AutoConfigSource::Cli,
            ),
        ] {
            for upload in ["sparse", "compact-scatter"] {
                let mut entries = vec![
                    RuntimeConfigEntry::new(
                        "FERRUM_INVOCATION_PREPARATION_STRATEGY",
                        "decode-segment",
                        RuntimeConfigSource::Cli,
                    ),
                    RuntimeConfigEntry::new(
                        "FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY",
                        upload,
                        RuntimeConfigSource::Cli,
                    ),
                ];
                if let Some(mode) = mode {
                    entries.push(RuntimeConfigEntry::new(key, mode, runtime_source));
                }
                let resolved = scheduler_resolution(
                    "cuda",
                    ExecutionPlanningModel::CompiledPlan,
                    4,
                    RuntimeConfigSnapshot::from_entries(entries),
                );
                let expected = mode.unwrap_or("legacy");
                let entry = resolved
                    .runtime_config
                    .entries
                    .iter()
                    .find(|entry| entry.key == key)
                    .unwrap();
                assert_eq!(entry.effective_value, expected);
                assert_eq!(entry.source, runtime_source);
                assert_eq!(entry.affects, vec![RuntimeConfigEffect::Performance]);
                let decision = resolved
                    .decisions
                    .iter()
                    .find(|decision| decision.selection == "segment_binding_owner_view_mode")
                    .unwrap();
                assert_eq!(decision.selected, expected);
                assert_eq!(decision.source, source);
                assert_eq!(decision.source_key.as_deref(), mode.map(|_| key));
                let mut engine = crate::EngineConfig::default();
                engine
                    .apply_runtime_config_snapshot(&resolved.runtime_config)
                    .unwrap();
                assert_eq!(
                    engine
                        .runtime
                        .segment_binding_owner_view_mode
                        .as_runtime_value(),
                    expected
                );
                assert_eq!(
                    engine
                        .runtime
                        .program_binding_upload_strategy
                        .as_runtime_value(),
                    upload
                );
            }
        }
    }

    #[test]
    fn compiled_segments_require_cuda_plan_and_checked_compact_transport() {
        let key = "FERRUM_INVOCATION_PREPARATION_STRATEGY";
        for strategy in [
            crate::InvocationPreparationStrategy::DecodeSegmentProgram,
            crate::InvocationPreparationStrategy::ResidentDecodeProgram,
        ] {
            for (backend, authority, owner, upload, supported) in [
                (
                    "cuda",
                    ExecutionPlanningModel::CompiledPlan,
                    "indexed",
                    "compact-scatter",
                    true,
                ),
                (
                    "metal",
                    ExecutionPlanningModel::CompiledPlan,
                    "indexed",
                    "compact-scatter",
                    false,
                ),
                (
                    "cuda",
                    ExecutionPlanningModel::ExecutorManaged,
                    "indexed",
                    "compact-scatter",
                    false,
                ),
                (
                    "cuda",
                    ExecutionPlanningModel::CompiledPlan,
                    "legacy",
                    "compact-scatter",
                    false,
                ),
                (
                    "cuda",
                    ExecutionPlanningModel::CompiledPlan,
                    "indexed",
                    "sparse",
                    false,
                ),
            ] {
                let entries = snapshot_with_sources(&[
                    (key, strategy.as_runtime_value(), RuntimeConfigSource::Cli),
                    (
                        crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY,
                        owner,
                        RuntimeConfigSource::Cli,
                    ),
                    (
                        "FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY",
                        upload,
                        RuntimeConfigSource::Cli,
                    ),
                ]);
                let mut hardware = HardwareCapabilities::unknown();
                hardware.backend = backend.to_owned();
                hardware.compiled_features.cuda = backend == "cuda";
                let result = FerrumConfigBuilder::new(entries)
                    .with_hardware_capabilities(hardware)
                    .with_execution_planning_model(authority)
                    .resolve();
                if supported {
                    let resolved = result.unwrap();
                    let mut engine = crate::EngineConfig::default();
                    engine
                        .apply_runtime_config_snapshot(&resolved.runtime_config)
                        .unwrap();
                    assert_eq!(engine.runtime.invocation_preparation_strategy, strategy);
                } else {
                    assert!(
                        matches!(result, Err(AutoConfigError::InvalidOverride { key: actual, .. }) if actual == key)
                    );
                }
            }
        }
    }

    #[test]
    fn causal_decode_attention_grouping_resolves_defaults_and_explicit_sources() {
        let key = crate::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY;
        for (grouping, source) in [
            (None, RuntimeConfigSource::Default),
            (Some("contiguous"), RuntimeConfigSource::Cli),
            (Some("indexed"), RuntimeConfigSource::ConfigFile),
            (Some("indexed"), RuntimeConfigSource::Cli),
            (Some("fixed-capacity"), RuntimeConfigSource::ConfigFile),
            (Some("fixed-capacity"), RuntimeConfigSource::Cli),
            (
                Some("fixed-execution-capacity"),
                RuntimeConfigSource::ConfigFile,
            ),
            (Some("fixed-execution-capacity"), RuntimeConfigSource::Cli),
        ] {
            for (preparation, upload, owner) in [
                ("full", "sparse", "legacy"),
                ("decode-segment", "compact-scatter", "indexed"),
            ] {
                let mut entries = vec![
                    RuntimeConfigEntry::new(
                        crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY,
                        "packed",
                        RuntimeConfigSource::ConfigFile,
                    ),
                    RuntimeConfigEntry::new(
                        "FERRUM_INVOCATION_PREPARATION_STRATEGY",
                        preparation,
                        RuntimeConfigSource::Cli,
                    ),
                    RuntimeConfigEntry::new(
                        "FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY",
                        upload,
                        RuntimeConfigSource::Cli,
                    ),
                    RuntimeConfigEntry::new(
                        crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY,
                        owner,
                        RuntimeConfigSource::Cli,
                    ),
                ];
                if let Some(grouping) = grouping {
                    entries.push(RuntimeConfigEntry::new(key, grouping, source));
                }
                let mut hardware = HardwareCapabilities::unknown();
                hardware.backend = "cuda".to_owned();
                hardware.compiled_features.cuda = true;
                hardware.compiled_features.vllm_paged_attn = true;
                hardware.compiled_features.cuda_indexed_causal_attention = true;
                let resolved =
                    FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_entries(entries))
                        .with_hardware_capabilities(hardware)
                        .with_execution_planning_model(ExecutionPlanningModel::CompiledPlan)
                        .resolve()
                        .unwrap();
                let expected = grouping.unwrap_or("fixed-execution-capacity");
                let entry = resolved
                    .runtime_config
                    .entries
                    .iter()
                    .find(|e| e.key == key)
                    .unwrap();
                assert_eq!(entry.effective_value, expected);
                assert_eq!(entry.source, source);
                let decision = resolved
                    .decisions
                    .iter()
                    .find(|d| d.selection == "causal_decode_attention_grouping")
                    .unwrap();
                assert_eq!(decision.selected, expected);
                assert_eq!(decision.source, auto_config_source_from_runtime(source));
                assert_eq!(decision.source_key.as_deref(), grouping.map(|_| key));
                let mut engine = crate::EngineConfig::default();
                engine
                    .apply_runtime_config_snapshot(&resolved.runtime_config)
                    .unwrap();
                assert_eq!(
                    engine
                        .runtime
                        .causal_decode_attention_grouping
                        .as_runtime_value(),
                    expected
                );
                assert_eq!(
                    engine.runtime.causal_decode_preparation_mode,
                    crate::CausalDecodePreparationMode::Packed
                );
                assert_eq!(
                    engine
                        .runtime
                        .invocation_preparation_strategy
                        .as_runtime_value(),
                    preparation
                );
                assert_eq!(
                    engine
                        .runtime
                        .program_binding_upload_strategy
                        .as_runtime_value(),
                    upload
                );
                assert_eq!(
                    engine
                        .runtime
                        .segment_binding_owner_view_mode
                        .as_runtime_value(),
                    owner
                );
            }
        }
        // The default does not impose CUDA or Packed prerequisites on existing configurations.
        let resolved = FerrumConfigBuilder::new(RuntimeConfigSnapshot::default())
            .resolve()
            .unwrap();
        let mut engine = crate::EngineConfig::default();
        engine
            .apply_runtime_config_snapshot(&resolved.runtime_config)
            .unwrap();
        assert_eq!(
            engine.runtime.causal_decode_attention_grouping,
            crate::CausalDecodeAttentionGrouping::Contiguous
        );
        assert_eq!(
            engine.runtime.causal_decode_preparation_mode,
            crate::CausalDecodePreparationMode::PerParticipant
        );
    }

    #[test]
    fn causal_decode_attention_grouping_rejects_env_and_unsupported_execution() {
        let key = crate::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY;
        for grouping in ["indexed", "fixed-capacity", "fixed-execution-capacity"] {
            for (backend, authority, compiled, policy, kv, preparation, source) in [
                (
                    "cuda",
                    ExecutionPlanningModel::CompiledPlan,
                    true,
                    "auto",
                    "fp16",
                    Some("packed"),
                    RuntimeConfigSource::Env,
                ),
                (
                    "cpu",
                    ExecutionPlanningModel::CompiledPlan,
                    true,
                    "auto",
                    "fp16",
                    Some("packed"),
                    RuntimeConfigSource::Cli,
                ),
                (
                    "metal",
                    ExecutionPlanningModel::CompiledPlan,
                    true,
                    "auto",
                    "fp16",
                    Some("packed"),
                    RuntimeConfigSource::Cli,
                ),
                (
                    "cuda",
                    ExecutionPlanningModel::ExecutorManaged,
                    true,
                    "auto",
                    "fp16",
                    Some("packed"),
                    RuntimeConfigSource::Cli,
                ),
                (
                    "cuda",
                    ExecutionPlanningModel::CompiledPlan,
                    false,
                    "auto",
                    "fp16",
                    Some("packed"),
                    RuntimeConfigSource::Cli,
                ),
                (
                    "cuda",
                    ExecutionPlanningModel::CompiledPlan,
                    true,
                    "portable",
                    "fp16",
                    Some("packed"),
                    RuntimeConfigSource::Cli,
                ),
                (
                    "cuda",
                    ExecutionPlanningModel::CompiledPlan,
                    true,
                    "auto",
                    "int8",
                    Some("packed"),
                    RuntimeConfigSource::Cli,
                ),
                (
                    "cuda",
                    ExecutionPlanningModel::CompiledPlan,
                    true,
                    "auto",
                    "fp16",
                    Some("per-participant"),
                    RuntimeConfigSource::Cli,
                ),
            ] {
                let mut entries = vec![
                    RuntimeConfigEntry::new(key, grouping, source),
                    RuntimeConfigEntry::new(
                        "FERRUM_ATTENTION_POLICY",
                        policy,
                        RuntimeConfigSource::Cli,
                    ),
                    RuntimeConfigEntry::new("FERRUM_KV_DTYPE", kv, RuntimeConfigSource::Cli),
                ];
                if let Some(preparation) = preparation {
                    entries.push(RuntimeConfigEntry::new(
                        crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY,
                        preparation,
                        RuntimeConfigSource::Cli,
                    ));
                }
                let mut hardware = HardwareCapabilities::unknown();
                hardware.backend = backend.to_owned();
                hardware.compiled_features.cuda = backend == "cuda";
                hardware.compiled_features.vllm_paged_attn = compiled;
                hardware.compiled_features.cuda_indexed_causal_attention = true;
                hardware.supported_kv_dtypes = vec!["fp16".to_owned(), "int8".to_owned()];
                let error = FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_entries(entries))
                    .with_hardware_capabilities(hardware)
                    .with_execution_planning_model(authority)
                    .resolve()
                    .unwrap_err();
                assert!(
                    matches!(&error, AutoConfigError::InvalidOverride { key: actual, .. } if actual == key),
                    "{error:?}"
                );
            }
        }
    }

    #[test]
    fn causal_decode_kv_reuse_requires_compiled_fixed_native_execution() {
        let key = crate::CAUSAL_DECODE_KV_REUSE_CONFIG_KEY;
        for (mode, compiled, grouping, supported) in [
            ("none", false, "contiguous", true),
            (
                "paired-query-heads",
                false,
                "fixed-execution-capacity",
                false,
            ),
            ("paired-query-heads", true, "indexed", false),
            ("paired-query-heads", true, "fixed-execution-capacity", true),
            (
                "paired-query-heads-unified-v2",
                false,
                "fixed-execution-capacity",
                false,
            ),
            ("paired-query-heads-unified-v2", true, "indexed", false),
            (
                "paired-query-heads-unified-v2",
                true,
                "fixed-execution-capacity",
                true,
            ),
        ] {
            let entries = snapshot_with_sources(&[
                (key, mode, RuntimeConfigSource::Cli),
                (
                    crate::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY,
                    grouping,
                    RuntimeConfigSource::Cli,
                ),
                (
                    crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY,
                    "packed",
                    RuntimeConfigSource::Cli,
                ),
            ]);
            let mut hardware = HardwareCapabilities::unknown();
            hardware.backend = "cuda".to_owned();
            hardware.compiled_features.cuda = true;
            hardware.compiled_features.vllm_paged_attn = true;
            hardware.compiled_features.cuda_indexed_causal_attention = true;
            hardware.compiled_features.cuda_causal_gqa2 = compiled;
            let result = FerrumConfigBuilder::new(entries)
                .with_hardware_capabilities(hardware)
                .with_execution_planning_model(ExecutionPlanningModel::CompiledPlan)
                .resolve();
            assert_eq!(result.is_ok(), supported, "{mode}/{grouping}: {result:?}");
            if let Ok(resolved) = result {
                let mut engine = crate::EngineConfig::default();
                engine
                    .apply_runtime_config_snapshot(&resolved.runtime_config)
                    .unwrap();
                assert_eq!(
                    engine.runtime.causal_decode_kv_reuse.as_runtime_value(),
                    mode
                );
            }
        }
        let mut old = serde_json::to_value(CompiledKernelFeatures::default()).unwrap();
        old.as_object_mut().unwrap().remove("cuda_causal_gqa2");
        assert!(
            !serde_json::from_value::<CompiledKernelFeatures>(old)
                .unwrap()
                .cuda_causal_gqa2
        );
    }

    #[test]
    fn causal_decode_attention_grouping_requires_its_additive_compiled_capability() {
        let key = crate::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY;
        let old_features = CompiledKernelFeatures::m3_fast_path_without_fa2();
        assert!(old_features.vllm_paged_attn);
        assert!(!old_features.cuda_indexed_causal_attention);
        let mut old_serialized = serde_json::to_value(&old_features).unwrap();
        old_serialized
            .as_object_mut()
            .unwrap()
            .remove("cuda_indexed_causal_attention");
        let decoded: CompiledKernelFeatures = serde_json::from_value(old_serialized).unwrap();
        assert!(!decoded.cuda_indexed_causal_attention);

        for compiled in [false, true] {
            for grouping in [
                "contiguous",
                "indexed",
                "fixed-capacity",
                "fixed-execution-capacity",
            ] {
                let entries = snapshot_with_sources(&[
                    (key, grouping, RuntimeConfigSource::Cli),
                    (
                        crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY,
                        "packed",
                        RuntimeConfigSource::Cli,
                    ),
                ]);
                let mut hardware = HardwareCapabilities::unknown();
                hardware.backend = "cuda".to_owned();
                hardware.compiled_features.cuda = true;
                hardware.compiled_features.vllm_paged_attn = true;
                hardware.compiled_features.cuda_indexed_causal_attention = compiled;
                let result = FerrumConfigBuilder::new(entries)
                    .with_hardware_capabilities(hardware)
                    .with_execution_planning_model(ExecutionPlanningModel::CompiledPlan)
                    .resolve();
                if grouping != "contiguous" && !compiled {
                    assert!(
                        matches!(result, Err(AutoConfigError::InvalidOverride { key: actual, .. }) if actual == key)
                    );
                } else {
                    let resolved = result.unwrap();
                    assert_eq!(
                        resolved
                            .decisions
                            .iter()
                            .find(|d| d.selection == "causal_decode_attention_grouping")
                            .unwrap()
                            .selected,
                        grouping
                    );
                }
            }
        }
    }

    #[test]
    fn causal_decode_preparation_reports_sources_and_preserves_independent_policies() {
        let key = crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY;
        for (mode, source) in [
            (None, RuntimeConfigSource::Default),
            (Some("packed"), RuntimeConfigSource::ConfigFile),
            (Some("per-participant"), RuntimeConfigSource::Cli),
            (Some("packed"), RuntimeConfigSource::Cli),
        ] {
            for preparation in ["full", "decode-segment"] {
                for upload in ["sparse", "compact-scatter"] {
                    let mut entries = vec![
                        RuntimeConfigEntry::new(
                            "FERRUM_INVOCATION_PREPARATION_STRATEGY",
                            preparation,
                            RuntimeConfigSource::Cli,
                        ),
                        RuntimeConfigEntry::new(
                            "FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY",
                            upload,
                            RuntimeConfigSource::Cli,
                        ),
                    ];
                    if let Some(mode) = mode {
                        entries.push(RuntimeConfigEntry::new(key, mode, source));
                    }
                    let mut hardware = HardwareCapabilities::unknown();
                    hardware.backend = "cuda".to_owned();
                    hardware.compiled_features.cuda = true;
                    hardware.compiled_features.vllm_paged_attn = true;
                    let resolved =
                        FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_entries(entries))
                            .with_hardware_capabilities(hardware)
                            .with_execution_planning_model(ExecutionPlanningModel::CompiledPlan)
                            .resolve()
                            .unwrap();
                    let expected = mode.unwrap_or("packed");
                    let entry = resolved
                        .runtime_config
                        .entries
                        .iter()
                        .find(|entry| entry.key == key)
                        .unwrap();
                    assert_eq!(entry.effective_value, expected);
                    assert_eq!(entry.source, source);
                    let decision = resolved
                        .decisions
                        .iter()
                        .find(|decision| decision.selection == "causal_decode_preparation_mode")
                        .unwrap();
                    assert_eq!(decision.selected, expected);
                    assert_eq!(decision.source, auto_config_source_from_runtime(source));
                    assert_eq!(decision.source_key.as_deref(), mode.map(|_| key));
                    let mut engine = crate::EngineConfig::default();
                    engine
                        .apply_runtime_config_snapshot(&resolved.runtime_config)
                        .unwrap();
                    assert_eq!(
                        engine
                            .runtime
                            .causal_decode_preparation_mode
                            .as_runtime_value(),
                        expected
                    );
                    assert_eq!(
                        engine
                            .runtime
                            .program_binding_upload_strategy
                            .as_runtime_value(),
                        upload
                    );
                    assert_eq!(
                        engine
                            .runtime
                            .invocation_preparation_strategy
                            .as_runtime_value(),
                        preparation
                    );
                    assert_eq!(
                        resolved.compiled_attention_policy,
                        Some(AttentionExecutionPolicy::NativeAdaptive)
                    );
                }
            }
        }
    }

    #[test]
    fn causal_decode_preparation_rejects_env_and_unavailable_native_execution() {
        let key = crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY;
        for (backend, authority, compiled, policy, kv, source) in [
            (
                "cuda",
                ExecutionPlanningModel::CompiledPlan,
                true,
                "auto",
                "fp16",
                RuntimeConfigSource::Env,
            ),
            (
                "cpu",
                ExecutionPlanningModel::CompiledPlan,
                true,
                "auto",
                "fp16",
                RuntimeConfigSource::Cli,
            ),
            (
                "metal",
                ExecutionPlanningModel::CompiledPlan,
                true,
                "auto",
                "fp16",
                RuntimeConfigSource::Cli,
            ),
            (
                "cuda",
                ExecutionPlanningModel::ExecutorManaged,
                true,
                "auto",
                "fp16",
                RuntimeConfigSource::Cli,
            ),
            (
                "cuda",
                ExecutionPlanningModel::CompiledPlan,
                false,
                "auto",
                "fp16",
                RuntimeConfigSource::Cli,
            ),
            (
                "cuda",
                ExecutionPlanningModel::CompiledPlan,
                true,
                "portable",
                "fp16",
                RuntimeConfigSource::Cli,
            ),
            (
                "cuda",
                ExecutionPlanningModel::CompiledPlan,
                true,
                "auto",
                "int8",
                RuntimeConfigSource::Cli,
            ),
        ] {
            let entries = snapshot_with_sources(&[
                (key, "packed", source),
                ("FERRUM_ATTENTION_POLICY", policy, RuntimeConfigSource::Cli),
                ("FERRUM_KV_DTYPE", kv, RuntimeConfigSource::Cli),
            ]);
            let mut hardware = HardwareCapabilities::unknown();
            hardware.backend = backend.to_owned();
            hardware.compiled_features.cuda = backend == "cuda";
            hardware.compiled_features.vllm_paged_attn = compiled;
            // INT8 is a valid storage choice for this fixture; the packed
            // preparation mode must reject its resolved portable attention.
            hardware.supported_kv_dtypes = vec!["fp16".to_owned(), "int8".to_owned()];
            let error = FerrumConfigBuilder::new(entries)
                .with_hardware_capabilities(hardware)
                .with_execution_planning_model(authority)
                .resolve()
                .unwrap_err();
            assert!(
                matches!(&error, AutoConfigError::InvalidOverride { key: actual, .. } if actual == key),
                "backend={backend}, authority={authority:?}, compiled={compiled}, policy={policy}, kv={kv}, source={source:?}: {error:?}"
            );
        }
    }

    #[test]
    fn segment_owner_mode_rejects_environment_and_unsupported_execution() {
        let key = crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY;
        for (backend, authority, preparation, source) in [
            (
                "cuda",
                ExecutionPlanningModel::CompiledPlan,
                "decode-segment",
                RuntimeConfigSource::Env,
            ),
            (
                "cpu",
                ExecutionPlanningModel::CompiledPlan,
                "decode-segment",
                RuntimeConfigSource::Cli,
            ),
            (
                "metal",
                ExecutionPlanningModel::CompiledPlan,
                "decode-segment",
                RuntimeConfigSource::Cli,
            ),
            (
                "cuda",
                ExecutionPlanningModel::ExecutorManaged,
                "decode-segment",
                RuntimeConfigSource::Cli,
            ),
            (
                "cuda",
                ExecutionPlanningModel::CompiledPlan,
                "full",
                RuntimeConfigSource::Cli,
            ),
            (
                "cuda",
                ExecutionPlanningModel::CompiledPlan,
                "identity-projection",
                RuntimeConfigSource::Cli,
            ),
        ] {
            let snapshot = snapshot_with_sources(&[
                (key, "indexed", source),
                (
                    "FERRUM_INVOCATION_PREPARATION_STRATEGY",
                    preparation,
                    RuntimeConfigSource::Cli,
                ),
            ]);
            let mut hardware = HardwareCapabilities::unknown();
            hardware.backend = backend.to_owned();
            let error = FerrumConfigBuilder::new(snapshot)
                .with_hardware_capabilities(hardware)
                .with_execution_planning_model(authority)
                .resolve()
                .unwrap_err();
            assert!(
                matches!(error, AutoConfigError::InvalidOverride { key: actual, .. } if actual == key)
            );
        }
        for backend in ["cpu", "metal", "cuda"] {
            scheduler_resolution(
                backend,
                ExecutionPlanningModel::ExecutorManaged,
                4,
                RuntimeConfigSnapshot::default(),
            );
        }
    }

    #[test]
    fn completion_readback_policy_resolves_explicit_config_and_preserves_default() {
        let key = crate::COMPLETION_READBACK_POLICY_CONFIG_KEY;
        for backend in ["cpu", "metal", "cuda"] {
            for (value, source) in [
                (None, RuntimeConfigSource::Default),
                (Some("batched-u32"), RuntimeConfigSource::ConfigFile),
                (Some("synchronous"), RuntimeConfigSource::Cli),
            ] {
                let snapshot = value
                    .map(|value| snapshot_with_sources(&[(key, value, source)]))
                    .unwrap_or_default();
                let resolved = scheduler_resolution(
                    backend,
                    ExecutionPlanningModel::CompiledPlan,
                    4,
                    snapshot,
                );
                let expected = value.unwrap_or("synchronous");
                let entry = resolved
                    .runtime_config
                    .entries
                    .iter()
                    .find(|entry| entry.key == key)
                    .unwrap();
                assert_eq!(entry.effective_value, expected);
                assert_eq!(entry.source, source);
                let decision = resolved
                    .decisions
                    .iter()
                    .find(|decision| decision.selection == "completion_readback_policy")
                    .unwrap();
                assert_eq!(decision.selected, expected);
                let mut engine = crate::EngineConfig::default();
                engine
                    .apply_runtime_config_snapshot(&resolved.runtime_config)
                    .unwrap();
                assert_eq!(
                    engine.runtime.completion_readback_policy.as_runtime_value(),
                    expected
                );
            }
        }
        for (value, source) in [
            ("batched-u32", RuntimeConfigSource::Env),
            ("automatic", RuntimeConfigSource::Cli),
        ] {
            let error = FerrumConfigBuilder::new(snapshot_with_sources(&[(key, value, source)]))
                .resolve()
                .unwrap_err();
            assert!(
                matches!(error, AutoConfigError::InvalidOverride { key: actual, .. } if actual == key)
            );
        }
    }

    #[test]
    fn program_binding_upload_records_sparse_default_and_explicit_sources() {
        let key = "FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY";
        for (value, runtime_source, source) in [
            (
                None,
                RuntimeConfigSource::Default,
                AutoConfigSource::Default,
            ),
            (
                Some("uniform-live-prefix"),
                RuntimeConfigSource::ConfigFile,
                AutoConfigSource::ConfigFile,
            ),
            (
                Some("uniform-live-prefix"),
                RuntimeConfigSource::Env,
                AutoConfigSource::Env,
            ),
            (
                Some("compact-scatter"),
                RuntimeConfigSource::Cli,
                AutoConfigSource::Cli,
            ),
            (
                Some("sparse"),
                RuntimeConfigSource::Cli,
                AutoConfigSource::Cli,
            ),
        ] {
            let snapshot = value
                .map(|value| snapshot_with_sources(&[(key, value, runtime_source)]))
                .unwrap_or_default();
            let resolved =
                scheduler_resolution("cuda", ExecutionPlanningModel::CompiledPlan, 4, snapshot);
            let expected = value.unwrap_or("sparse");
            let entry = resolved
                .runtime_config
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap();
            assert_eq!(entry.effective_value, expected);
            assert_eq!(entry.source, runtime_source);
            let decision = resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == "program_binding_upload_strategy")
                .unwrap();
            assert_eq!(decision.selected, expected);
            assert!(decision.candidates.contains(&decision.selected));
            assert_eq!(decision.source, source);
            assert_eq!(decision.source_key.as_deref(), value.map(|_| key));
            let mut engine = crate::EngineConfig::default();
            engine
                .apply_runtime_config_snapshot(&resolved.runtime_config)
                .unwrap();
            assert_eq!(
                engine
                    .runtime
                    .program_binding_upload_strategy
                    .as_runtime_value(),
                expected
            );
        }
        let error = FerrumConfigBuilder::new(snapshot(&[(key, "automatic")]))
            .resolve()
            .unwrap_err();
        assert!(
            matches!(error, AutoConfigError::InvalidOverride { key: actual, .. } if actual == key)
        );
        for (backend, authority) in [
            ("cpu", ExecutionPlanningModel::CompiledPlan),
            ("metal", ExecutionPlanningModel::CompiledPlan),
            ("cuda", ExecutionPlanningModel::ExecutorManaged),
        ] {
            let mut hardware = HardwareCapabilities::unknown();
            hardware.backend = backend.to_owned();
            for strategy in ["uniform-live-prefix", "compact-scatter"] {
                let error = FerrumConfigBuilder::new(snapshot(&[(key, strategy)]))
                    .with_hardware_capabilities(hardware.clone())
                    .with_execution_planning_model(authority)
                    .resolve()
                    .unwrap_err();
                assert!(
                    matches!(error, AutoConfigError::InvalidOverride { key: actual, .. } if actual == key)
                );
            }
            scheduler_resolution(backend, authority, 4, snapshot(&[(key, "sparse")]));
        }
    }

    #[test]
    fn active_decode_prefill_token_budget_preserves_override_sources_and_disable() {
        let key = "FERRUM_ACTIVE_DECODE_PREFILL_TOKEN_BUDGET";
        for backend in ["metal", "cuda"] {
            for (value, runtime_source, source) in [
                (
                    "96",
                    RuntimeConfigSource::ConfigFile,
                    AutoConfigSource::ConfigFile,
                ),
                ("192", RuntimeConfigSource::Env, AutoConfigSource::Env),
                ("0", RuntimeConfigSource::Cli, AutoConfigSource::Cli),
            ] {
                let resolved = scheduler_resolution(
                    backend,
                    ExecutionPlanningModel::CompiledPlan,
                    4,
                    snapshot_with_sources(&[(key, value, runtime_source)]),
                );
                let mut engine = crate::EngineConfig::default();
                engine
                    .apply_runtime_config_snapshot(&resolved.runtime_config)
                    .unwrap();
                let expected = value.parse::<usize>().unwrap();
                assert_eq!(
                    engine.scheduler.active_decode_prefill_token_budget,
                    (expected > 0).then_some(expected)
                );
                let entry = resolved
                    .runtime_config
                    .entries
                    .iter()
                    .find(|entry| entry.key == key)
                    .unwrap();
                assert_eq!(entry.effective_value, value);
                assert_eq!(entry.source, runtime_source);
                let policy = resolved
                    .decisions
                    .iter()
                    .find(|decision| decision.selection == "scheduler_admission_policy")
                    .unwrap();
                let suffix = if expected == 0 { "disabled" } else { value };
                assert!(policy
                    .selected
                    .split('+')
                    .any(|part| part == format!("active_decode_prefill_token_budget:{suffix}")));
                assert_eq!(policy.source, source);
                assert_eq!(policy.source_key.as_deref(), Some(key));
            }
        }
    }

    #[test]
    fn active_decode_prefill_token_budget_has_no_implicit_default() {
        for backend in ["cpu", "metal", "cuda"] {
            let resolved = scheduler_resolution(
                backend,
                ExecutionPlanningModel::CompiledPlan,
                4,
                snapshot(&[]),
            );
            let mut engine = crate::EngineConfig::default();
            engine
                .apply_runtime_config_snapshot(&resolved.runtime_config)
                .unwrap();
            assert!(engine
                .scheduler
                .active_decode_prefill_token_budget
                .is_none());
            assert!(resolved
                .runtime_config
                .entries
                .iter()
                .all(|entry| entry.key != "FERRUM_ACTIVE_DECODE_PREFILL_TOKEN_BUDGET"));
            let policy = resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == "scheduler_admission_policy")
                .unwrap();
            assert!(!policy
                .selected
                .contains("active_decode_prefill_token_budget"));
        }
    }

    #[test]
    fn active_decode_prefill_token_budget_rejects_invalid_overrides() {
        let key = "FERRUM_ACTIVE_DECODE_PREFILL_TOKEN_BUDGET";
        for value in ["many", "-1", "1.5"] {
            let error = FerrumConfigBuilder::new(snapshot(&[(key, value)]))
                .resolve()
                .unwrap_err();
            assert!(
                matches!(error, AutoConfigError::InvalidOverride { key: actual, .. } if actual == key)
            );
        }
    }

    #[test]
    fn metal_plan_active_prefill_default_reaches_engine_and_decision_trace() {
        let resolved = scheduler_resolution(
            "metal",
            ExecutionPlanningModel::CompiledPlan,
            4,
            snapshot(&[]),
        );
        let mut engine = crate::EngineConfig::default();
        engine
            .apply_runtime_config_snapshot(&resolved.runtime_config)
            .unwrap();
        assert_eq!(engine.scheduler.active_decode_prefill_chunk, Some(128));
        assert_eq!(engine.scheduler.prefill_step_chunk, None);
        let entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK")
            .unwrap();
        assert_eq!(entry.source, RuntimeConfigSource::Default);
        assert_eq!(
            entry.effective_value.parse::<usize>().unwrap(),
            engine.scheduler.active_decode_prefill_chunk.unwrap()
        );
        let policy = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "scheduler_admission_policy")
            .unwrap();
        assert_eq!(policy.selected,
            "prefill_first_until_active:4+active_decode_prefill_chunk:128+prefill_token_budget:elastic");
        assert_eq!(policy.source, AutoConfigSource::Default);
        assert_eq!(policy.source_key, None);
    }

    #[test]
    fn active_prefill_default_does_not_change_other_execution_paths() {
        for (backend, authority, sequences) in [
            ("metal", ExecutionPlanningModel::CompiledPlan, 1),
            ("metal", ExecutionPlanningModel::ExecutorManaged, 4),
            ("cpu", ExecutionPlanningModel::CompiledPlan, 4),
            ("cuda", ExecutionPlanningModel::CompiledPlan, 4),
        ] {
            let resolved = scheduler_resolution(backend, authority, sequences, snapshot(&[]));
            let mut engine = crate::EngineConfig::default();
            engine
                .apply_runtime_config_snapshot(&resolved.runtime_config)
                .unwrap();
            assert_eq!(engine.scheduler.active_decode_prefill_chunk, None);
            assert!(resolved
                .runtime_config
                .entries
                .iter()
                .all(|entry| entry.key != "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK"));
            assert!(resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == "scheduler_admission_policy")
                .unwrap()
                .selected
                .split('+')
                .all(|part| !part.starts_with("active_decode_prefill_chunk:")));
        }
    }

    #[test]
    fn cuda_plan_active_prefill_default_and_overrides_reach_engine_and_trace() {
        let key = "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK";
        let resolve = |authority, sequences, compiled_cuda, entry: Option<RuntimeConfigEntry>| {
            let mut hardware = HardwareCapabilities::unknown();
            hardware.backend = "cuda".into();
            hardware.compiled_features.cuda = compiled_cuda;
            let mut workload = WorkloadProfile::serving_default();
            workload.target_concurrency = sequences;
            FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_entries(entry))
                .with_hardware_capabilities(hardware)
                .with_workload_profile(workload)
                .with_execution_planning_model(authority)
                .resolve()
                .unwrap()
        };
        for (authority, sequences, compiled_cuda, expected) in [
            (ExecutionPlanningModel::CompiledPlan, 4, true, Some(256)),
            (ExecutionPlanningModel::CompiledPlan, 1, true, None),
            (ExecutionPlanningModel::ExecutorManaged, 4, true, None),
            (ExecutionPlanningModel::CompiledPlan, 4, false, None),
        ] {
            let resolved = resolve(authority, sequences, compiled_cuda, None);
            let mut engine = crate::EngineConfig::default();
            engine
                .apply_runtime_config_snapshot(&resolved.runtime_config)
                .unwrap();
            assert_eq!(engine.scheduler.active_decode_prefill_chunk, expected);
            assert_eq!(engine.scheduler.active_decode_prefill_token_budget, None);
            assert_eq!(engine.scheduler.prefill_step_chunk, None);
            let entry = resolved
                .runtime_config
                .entries
                .iter()
                .find(|entry| entry.key == key);
            let policy = resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == "scheduler_admission_policy")
                .unwrap();
            if let Some(expected) = expected {
                let entry = entry.unwrap();
                assert_eq!(entry.effective_value, expected.to_string());
                assert_eq!(entry.source, RuntimeConfigSource::Default);
                assert!(policy
                    .selected
                    .split('+')
                    .any(|part| part == format!("active_decode_prefill_chunk:{expected}")));
                assert_eq!(policy.source, AutoConfigSource::Default);
                assert_eq!(policy.source_key, None);
            } else {
                assert!(entry.is_none());
                assert!(!policy.selected.contains("active_decode_prefill_chunk:"));
            }
        }
        for (value, source) in [
            ("24", RuntimeConfigSource::ConfigFile),
            ("96", RuntimeConfigSource::Cli),
            ("512", RuntimeConfigSource::Env),
            ("32", RuntimeConfigSource::Default),
        ] {
            let resolved = resolve(
                ExecutionPlanningModel::CompiledPlan,
                4,
                true,
                Some(RuntimeConfigEntry::new(key, value, source)),
            );
            let mut engine = crate::EngineConfig::default();
            engine
                .apply_runtime_config_snapshot(&resolved.runtime_config)
                .unwrap();
            assert_eq!(
                engine.scheduler.active_decode_prefill_chunk,
                Some(value.parse().unwrap())
            );
            let entry = resolved
                .runtime_config
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap();
            assert_eq!(entry.effective_value, value);
            assert_eq!(entry.source, source);
            let policy = resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == "scheduler_admission_policy")
                .unwrap();
            assert_eq!(policy.source, auto_config_source_from_runtime(source));
            assert_eq!(policy.source_key.as_deref(), Some(key));
            assert!(policy
                .selected
                .split('+')
                .any(|part| part == format!("active_decode_prefill_chunk:{value}")));
        }
    }

    #[test]
    fn explicit_active_prefill_limits_keep_their_value_and_source() {
        for (value, runtime_source, source) in [
            (
                "24",
                RuntimeConfigSource::ConfigFile,
                AutoConfigSource::ConfigFile,
            ),
            ("96", RuntimeConfigSource::Cli, AutoConfigSource::Cli),
            ("256", RuntimeConfigSource::Env, AutoConfigSource::Env),
        ] {
            let resolved = scheduler_resolution(
                "metal",
                ExecutionPlanningModel::CompiledPlan,
                4,
                snapshot_with_sources(&[(
                    "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK",
                    value,
                    runtime_source,
                )]),
            );
            let mut engine = crate::EngineConfig::default();
            engine
                .apply_runtime_config_snapshot(&resolved.runtime_config)
                .unwrap();
            assert_eq!(
                engine.scheduler.active_decode_prefill_chunk,
                Some(value.parse().unwrap())
            );
            let entry = resolved
                .runtime_config
                .entries
                .iter()
                .find(|entry| entry.key == "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK")
                .unwrap();
            assert_eq!(entry.effective_value, value);
            assert_eq!(entry.source, runtime_source);
            let policy = resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == "scheduler_admission_policy")
                .unwrap();
            assert!(policy
                .selected
                .split('+')
                .any(|part| part == format!("active_decode_prefill_chunk:{value}")));
            assert_eq!(policy.source, source);
            assert_eq!(
                policy.source_key.as_deref(),
                Some("FERRUM_ACTIVE_DECODE_PREFILL_CHUNK")
            );
        }
    }

    #[test]
    fn default_active_prefill_limit_preserves_explicit_scheduler_decision_source() {
        for (key, value) in [
            ("FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE", "3"),
            ("FERRUM_SCHED_PREFILL_STEP_CHUNK", "24"),
        ] {
            let resolved = scheduler_resolution(
                "metal",
                ExecutionPlanningModel::CompiledPlan,
                4,
                snapshot_with_sources(&[(key, value, RuntimeConfigSource::Cli)]),
            );
            let policy = resolved
                .decisions
                .iter()
                .find(|decision| decision.selection == "scheduler_admission_policy")
                .unwrap();
            assert_eq!(policy.source, AutoConfigSource::Cli);
            assert_eq!(policy.source_key.as_deref(), Some(key));
            let mut engine = crate::EngineConfig::default();
            engine
                .apply_runtime_config_snapshot(&resolved.runtime_config)
                .unwrap();
            assert_eq!(engine.scheduler.active_decode_prefill_chunk, Some(128));
        }
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
            "prefill_first_until_active:32+prefill_token_budget:elastic"
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
            "prefill_first_until_active:32+prefill_token_budget:elastic"
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
        assert!(resolved
            .runtime_config
            .entries
            .iter()
            .all(|entry| entry.key != "FERRUM_SCHED_PREFILL_STEP_CHUNK"));
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
            "prefill_first_until_active:32+prefill_token_budget:elastic"
        );
        assert_eq!(scheduler.source, AutoConfigSource::ScriptCase);
        assert_eq!(
            scheduler.source_key.as_deref(),
            Some("FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE")
        );
    }

    #[test]
    fn execution_planning_model_preserves_resolved_config_wire_and_runtime_hash() {
        for (planning_model, wire_label) in [
            (ExecutionPlanningModel::ExecutorManaged, "legacy_engine"),
            (ExecutionPlanningModel::CompiledPlan, "plan_runtime"),
        ] {
            let resolved = FerrumConfigBuilder::new(RuntimeConfigSnapshot::default())
                .with_execution_planning_model(planning_model)
                .resolve()
                .unwrap();
            let historical_wire = serde_json::to_value(&resolved).unwrap();
            assert_eq!(historical_wire["execution_resource_authority"], wire_label);
            assert!(historical_wire.get("execution_planning_model").is_none());
            let reread: ResolvedFerrumConfig =
                serde_json::from_value(historical_wire.clone()).unwrap();
            assert_eq!(reread.execution_planning_model, planning_model);
            assert_eq!(reread.runtime_config, resolved.runtime_config);
            assert_eq!(reread.runtime_env_hash(), resolved.runtime_env_hash());
            assert_eq!(serde_json::to_value(&reread).unwrap(), historical_wire);
            let effective = reread.effective_config_document();
            assert_eq!(effective["schema_version"], 1);
            assert_eq!(effective["execution_resource_authority"], wire_label);
            assert_eq!(effective["admission"]["resource_authority"], wire_label);
            assert_eq!(
                effective["attention_execution"]["resource_authority"],
                wire_label
            );
            assert!(effective.get("execution_planning_model").is_none());
        }
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
    #[test]
    fn q4_mmvq_weight_decode_resolves_sources_and_rejects_unsupported_execution() {
        let key = crate::Q4_MMVQ_WEIGHT_DECODE_CONFIG_KEY;
        for backend in ["cpu", "metal", "cuda"] {
            for authority in [
                ExecutionPlanningModel::ExecutorManaged,
                ExecutionPlanningModel::CompiledPlan,
            ] {
                for compiled in [false, true] {
                    for mode in ["baseline", "shared-m8", "shared-m4-m8"] {
                        let mut hardware = HardwareCapabilities::unknown();
                        hardware.backend = backend.into();
                        hardware.compiled_features.cuda = backend == "cuda";
                        hardware.compiled_features.cuda_q4_shared_weight_decode = compiled;
                        let builder =
                            FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_entries([
                                RuntimeConfigEntry::new(key, mode, RuntimeConfigSource::Cli),
                            ]))
                            .with_hardware_capabilities(hardware)
                            .with_execution_planning_model(authority);
                        let result = builder.resolve();
                        let supported = mode == "baseline"
                            || (backend == "cuda"
                                && authority == ExecutionPlanningModel::CompiledPlan
                                && compiled);
                        if supported {
                            let resolved = result.unwrap();
                            let mut engine = crate::EngineConfig::default();
                            engine
                                .apply_runtime_config_snapshot(&resolved.runtime_config)
                                .unwrap();
                            assert_eq!(
                                engine.runtime.q4_mmvq_weight_decode.as_runtime_value(),
                                mode
                            );
                            let decision = resolved
                                .decisions
                                .iter()
                                .find(|d| d.selection == "q4_mmvq_weight_decode")
                                .unwrap();
                            assert_eq!(decision.source, AutoConfigSource::Cli);
                        } else {
                            assert!(
                                matches!(result, Err(AutoConfigError::InvalidOverride { key: actual, .. }) if actual == key)
                            );
                        }
                    }
                }
            }
        }
        for source in [
            RuntimeConfigSource::Default,
            RuntimeConfigSource::ConfigFile,
            RuntimeConfigSource::Cli,
        ] {
            let mut hardware = HardwareCapabilities::unknown();
            hardware.backend = "cuda".into();
            hardware.compiled_features.cuda = true;
            hardware.compiled_features.cuda_q4_shared_weight_decode = true;
            let resolved = FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_entries([
                RuntimeConfigEntry::new(key, "shared-m8", source),
            ]))
            .with_hardware_capabilities(hardware)
            .with_execution_planning_model(ExecutionPlanningModel::CompiledPlan)
            .resolve()
            .unwrap();
            let entry = resolved
                .runtime_config
                .entries
                .iter()
                .find(|e| e.key == key)
                .unwrap();
            assert_eq!(entry.source, source);
        }
    }
    #[test]
    fn q5_mmvq_weight_decode_resolves_sources_and_rejects_unsupported_execution() {
        let key = crate::Q5_MMVQ_WEIGHT_DECODE_CONFIG_KEY;
        for backend in ["cpu", "metal", "cuda"] {
            for authority in [
                ExecutionPlanningModel::ExecutorManaged,
                ExecutionPlanningModel::CompiledPlan,
            ] {
                for compiled in [false, true] {
                    for mode in ["baseline", "shared-m8", "shared-m4-m8"] {
                        let mut hardware = HardwareCapabilities::unknown();
                        hardware.backend = backend.into();
                        hardware.compiled_features.cuda = backend == "cuda";
                        hardware.compiled_features.cuda_q5_shared_weight_decode = compiled;
                        let builder =
                            FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_entries([
                                RuntimeConfigEntry::new(key, mode, RuntimeConfigSource::Cli),
                            ]))
                            .with_hardware_capabilities(hardware)
                            .with_execution_planning_model(authority);
                        let result = builder.resolve();
                        let supported = mode == "baseline"
                            || (backend == "cuda"
                                && authority == ExecutionPlanningModel::CompiledPlan
                                && compiled);
                        if supported {
                            let resolved = result.unwrap();
                            let mut engine = crate::EngineConfig::default();
                            engine
                                .apply_runtime_config_snapshot(&resolved.runtime_config)
                                .unwrap();
                            assert_eq!(
                                engine.runtime.q5_mmvq_weight_decode.as_runtime_value(),
                                mode
                            );
                            let decision = resolved
                                .decisions
                                .iter()
                                .find(|d| d.selection == "q5_mmvq_weight_decode")
                                .unwrap();
                            assert_eq!(decision.source, AutoConfigSource::Cli);
                        } else {
                            assert!(
                                matches!(result, Err(AutoConfigError::InvalidOverride { key: actual, .. }) if actual == key)
                            );
                        }
                    }
                }
            }
        }
        for source in [
            RuntimeConfigSource::Default,
            RuntimeConfigSource::ConfigFile,
            RuntimeConfigSource::Cli,
        ] {
            let mut hardware = HardwareCapabilities::unknown();
            hardware.backend = "cuda".into();
            hardware.compiled_features.cuda = true;
            hardware.compiled_features.cuda_q5_shared_weight_decode = true;
            let resolved = FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_entries([
                RuntimeConfigEntry::new(key, "shared-m8", source),
            ]))
            .with_hardware_capabilities(hardware)
            .with_execution_planning_model(ExecutionPlanningModel::CompiledPlan)
            .resolve()
            .unwrap();
            let entry = resolved
                .runtime_config
                .entries
                .iter()
                .find(|e| e.key == key)
                .unwrap();
            assert_eq!(entry.source, source);
        }
    }

    #[test]
    fn q8_projection_weight_preparation_requires_cuda_plan_runtime() {
        let key = crate::Q8_PROJECTION_WEIGHT_PREPARATION_CONFIG_KEY;
        for backend in ["cpu", "cuda"] {
            for authority in [
                ExecutionPlanningModel::ExecutorManaged,
                ExecutionPlanningModel::CompiledPlan,
            ] {
                for compiled in [false, true] {
                    for mode in ["auto", "compressed", "predecoded-f32"] {
                        let mut hardware = HardwareCapabilities::unknown();
                        hardware.backend = backend.into();
                        hardware.compiled_features.cuda = compiled;
                        let result =
                            FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_entries([
                                RuntimeConfigEntry::new(key, mode, RuntimeConfigSource::Cli),
                            ]))
                            .with_hardware_capabilities(hardware)
                            .with_execution_planning_model(authority)
                            .resolve();
                        let supported = mode != "predecoded-f32"
                            || (backend == "cuda"
                                && compiled
                                && authority == ExecutionPlanningModel::CompiledPlan);
                        if supported {
                            let resolved = result.unwrap();
                            let mut engine = crate::EngineConfig::default();
                            engine
                                .apply_runtime_config_snapshot(&resolved.runtime_config)
                                .unwrap();
                            assert_eq!(
                                engine
                                    .runtime
                                    .q8_projection_weight_preparation
                                    .as_runtime_value(),
                                mode
                            );
                        } else {
                            assert!(
                                matches!(result, Err(AutoConfigError::InvalidOverride { key: actual, .. }) if actual == key)
                            );
                        }
                    }
                }
            }
        }
        assert!(
            FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_entries([
                RuntimeConfigEntry::new(key, "predecoded-f32", RuntimeConfigSource::Env),
            ]))
            .resolve()
            .is_err()
        );
    }

    #[test]
    fn q4_mmvq_weight_decode_keeps_legacy_default_and_rejects_environment() {
        let key = crate::Q4_MMVQ_WEIGHT_DECODE_CONFIG_KEY;
        let mut old = serde_json::to_value(CompiledKernelFeatures::default()).unwrap();
        old.as_object_mut()
            .unwrap()
            .remove("cuda_q4_shared_weight_decode");
        let decoded: CompiledKernelFeatures = serde_json::from_value(old).unwrap();
        assert!(!decoded.cuda_q4_shared_weight_decode);
        let resolved = FerrumConfigBuilder::new(RuntimeConfigSnapshot::default())
            .resolve()
            .unwrap();
        assert_eq!(
            resolved
                .runtime_config
                .entries
                .iter()
                .find(|e| e.key == key)
                .unwrap()
                .effective_value,
            "baseline"
        );
        for value in ["auto", "baseline", "shared-m8", "shared-m4-m8", "invalid"] {
            let result = FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_entries([
                RuntimeConfigEntry::new(key, value, RuntimeConfigSource::Env),
            ]))
            .resolve();
            assert!(
                matches!(result, Err(AutoConfigError::InvalidOverride { key: actual, .. }) if actual == key)
            );
            let mut engine = crate::EngineConfig::default();
            assert!(engine
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_entries([
                    RuntimeConfigEntry::new(key, value, RuntimeConfigSource::Env),
                ]))
                .is_err());
            assert_eq!(
                engine.runtime.q4_mmvq_weight_decode,
                crate::Q4MmvqWeightDecode::Baseline
            );
        }
    }
    #[test]
    fn q5_mmvq_weight_decode_keeps_legacy_default_and_rejects_environment() {
        let key = crate::Q5_MMVQ_WEIGHT_DECODE_CONFIG_KEY;
        let mut old = serde_json::to_value(CompiledKernelFeatures::default()).unwrap();
        old.as_object_mut()
            .unwrap()
            .remove("cuda_q5_shared_weight_decode");
        let decoded: CompiledKernelFeatures = serde_json::from_value(old).unwrap();
        assert!(!decoded.cuda_q5_shared_weight_decode);
        let resolved = FerrumConfigBuilder::new(RuntimeConfigSnapshot::default())
            .resolve()
            .unwrap();
        assert_eq!(
            resolved
                .runtime_config
                .entries
                .iter()
                .find(|e| e.key == key)
                .unwrap()
                .effective_value,
            "baseline"
        );
        for value in ["auto", "baseline", "shared-m8", "shared-m4-m8", "invalid"] {
            let result = FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_entries([
                RuntimeConfigEntry::new(key, value, RuntimeConfigSource::Env),
            ]))
            .resolve();
            assert!(
                matches!(result, Err(AutoConfigError::InvalidOverride { key: actual, .. }) if actual == key)
            );
            let mut engine = crate::EngineConfig::default();
            assert!(engine
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_entries([
                    RuntimeConfigEntry::new(key, value, RuntimeConfigSource::Env),
                ]))
                .is_err());
            assert_eq!(
                engine.runtime.q5_mmvq_weight_decode,
                crate::Q5MmvqWeightDecode::Baseline
            );
        }
    }

    #[test]
    fn sequence_backing_growth_requires_explicit_cuda_resident_execution() {
        let key = crate::SEQUENCE_BACKING_GROWTH_CONFIG_KEY;
        use ExecutionPlanningModel::{CompiledPlan, ExecutorManaged};
        for (backend, authority, preparation, mode, supported) in [
            ("cpu", CompiledPlan, "full", "exact", true),
            (
                "cuda",
                CompiledPlan,
                "resident-decode-program",
                "paged-group4",
                true,
            ),
            ("cuda", CompiledPlan, "full", "paged-group4", false),
            (
                "cuda",
                CompiledPlan,
                "resident-decode-program",
                "paged-group4-grow-only",
                true,
            ),
            (
                "cuda",
                CompiledPlan,
                "full",
                "paged-group4-grow-only",
                false,
            ),
            (
                "cpu",
                CompiledPlan,
                "resident-decode-program",
                "paged-group4-grow-only",
                false,
            ),
            (
                "metal",
                CompiledPlan,
                "resident-decode-program",
                "paged-group4-grow-only",
                false,
            ),
            (
                "cuda",
                ExecutorManaged,
                "resident-decode-program",
                "paged-group4-grow-only",
                false,
            ),
        ] {
            let mut values = vec![
                (key, mode, RuntimeConfigSource::Cli),
                (
                    "FERRUM_INVOCATION_PREPARATION_STRATEGY",
                    preparation,
                    RuntimeConfigSource::Cli,
                ),
            ];
            if backend == "cuda" {
                values.extend([
                    (
                        crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY,
                        "indexed",
                        RuntimeConfigSource::Cli,
                    ),
                    (
                        "FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY",
                        "compact-scatter",
                        RuntimeConfigSource::Cli,
                    ),
                ]);
            }
            let entries = snapshot_with_sources(&values);
            let mut hardware = HardwareCapabilities::unknown();
            hardware.backend = backend.to_owned();
            hardware.compiled_features.cuda = backend == "cuda";
            let builder = FerrumConfigBuilder::new(entries)
                .with_hardware_capabilities(hardware)
                .with_execution_planning_model(authority);
            assert_eq!(
                builder.sequence_backing_growth_decision().is_ok(),
                supported
            );
            let result = builder.resolve();
            assert_eq!(result.is_ok(), supported, "{result:?}");
            if let Ok(resolved) = result {
                let entry = resolved
                    .runtime_config
                    .entries
                    .iter()
                    .find(|e| e.key == key)
                    .unwrap();
                assert_eq!(entry.effective_value, mode);
                assert_eq!(entry.source, RuntimeConfigSource::Cli);
                let decision = resolved
                    .decisions
                    .iter()
                    .find(|d| d.selection == "sequence_backing_growth")
                    .unwrap();
                assert!(decision.affects.contains(&RuntimeConfigEffect::Memory));
                assert!(decision
                    .candidates
                    .iter()
                    .any(|candidate| candidate == mode));
            }
        }
    }
}
