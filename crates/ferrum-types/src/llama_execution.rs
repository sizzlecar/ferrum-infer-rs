//! Instance-owned construction options for the retained Llama-family executor.
//!
//! Runtime snapshots are parsed at assembly. Numerical execution only reads
//! these typed values; constructing a library model never consults process state.

use serde::{Deserialize, Serialize};

/// Diagnostic layer selection, resolved before loading model weights.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlamaNanTraceConfig {
    #[default]
    Disabled,
    All,
    Layers(Vec<usize>),
}

impl LlamaNanTraceConfig {
    pub fn from_runtime_value(raw: &str) -> Self {
        let raw = raw.trim();
        if raw.is_empty() || raw == "1" || raw.eq_ignore_ascii_case("all") {
            return Self::All;
        }
        if raw == "0" || raw.eq_ignore_ascii_case("false") {
            return Self::Disabled;
        }
        let layers: Vec<_> = raw
            .split(',')
            .filter_map(|part| {
                let part = part.trim();
                part.parse::<usize>().ok().or_else(|| {
                    ["layer", "source", "l"].into_iter().find_map(|prefix| {
                        part.get(..prefix.len())
                            .filter(|head| head.eq_ignore_ascii_case(prefix))
                            .and_then(|_| part.get(prefix.len()..)?.parse().ok())
                    })
                })
            })
            .collect();
        if layers.is_empty() {
            Self::Disabled
        } else {
            Self::Layers(layers)
        }
    }

    pub fn matches_layer(&self, local_layer: usize, source_layer: usize) -> bool {
        match self {
            Self::Disabled => false,
            Self::All => true,
            Self::Layers(layers) => layers
                .iter()
                .any(|&layer| layer == local_layer || layer == source_layer),
        }
    }
}

/// Options specific to the retained model, in addition to the engine's
/// existing capacity, prefix and batched-graph controls.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlamaExecutionOptions {
    /// Explicit model-pool override. None preserves the existing model-sized
    /// pool default; the Engine KV manager's default is a separate capacity.
    pub kv_max_blocks: Option<usize>,
    pub metal_paged_kv: Option<bool>,
    pub cuda_graph: bool,
    pub unified_graph: bool,
    pub unified_graph_layers_only: bool,
    pub unified_graph_lm_head_eager: bool,
    pub greedy_argmax: bool,
    /// Legacy cold-assembly alias for backend_context.cuda.split_k_attn.
    /// A missing canonical value adopts this alias; explicit conflicts fail.
    pub split_k_attn: Option<bool>,
    pub nan_trace: LlamaNanTraceConfig,
    pub op_dump_dir: Option<String>,
    pub layer_dump_dir: Option<String>,
}

impl LlamaExecutionOptions {
    pub(crate) fn apply_runtime_config_snapshot(
        &mut self,
        snapshot: &crate::RuntimeConfigSnapshot,
    ) {
        for entry in &snapshot.entries {
            let value = entry.effective_value.as_str();
            match entry.key.as_str() {
                "FERRUM_METAL_PAGED_KV" => self.metal_paged_kv = Some(value != "0"),
                "FERRUM_CUDA_GRAPH" => self.cuda_graph = true,
                "FERRUM_UNIFIED_GRAPH" => self.unified_graph = value == "1",
                "FERRUM_UNIFIED_GRAPH_LAYERS_ONLY" => self.unified_graph_layers_only = value == "1",
                "FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER" => {
                    self.unified_graph_lm_head_eager = value == "1"
                }
                "FERRUM_GREEDY_ARGMAX" => self.greedy_argmax = value == "1",
                "FERRUM_SPLIT_K_ATTN" => {
                    self.split_k_attn = match value {
                        "1" => Some(true),
                        "0" => Some(false),
                        _ => None,
                    }
                }
                "FERRUM_NAN_TRACE" => {
                    self.nan_trace = LlamaNanTraceConfig::from_runtime_value(value)
                }
                "FERRUM_OP_DUMP" => self.op_dump_dir = Some(value.to_owned()),
                "FERRUM_LAYER_DUMP" => self.layer_dump_dir = Some(value.to_owned()),
                _ => {}
            }
        }
    }
}

/// Complete model construction input. A value is captured before the first
/// weight load and passed unchanged to every layer stage of that instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlamaExecutionConfig {
    pub backend_context: crate::BackendContextConfig,
    pub kv_capacity: Option<usize>,
    pub paged_max_seqs: usize,
    pub prefix_cache: bool,
    pub batched_graph: bool,
    pub options: LlamaExecutionOptions,
}

impl Default for LlamaExecutionConfig {
    fn default() -> Self {
        Self {
            backend_context: crate::BackendContextConfig::default(),
            kv_capacity: None,
            paged_max_seqs: 32,
            prefix_cache: false,
            batched_graph: false,
            options: LlamaExecutionOptions::default(),
        }
    }
}

impl LlamaExecutionConfig {
    /// Resolve legacy aliases once, before constructing the original backend.
    pub fn normalized(mut self) -> Result<Self, String> {
        self.validate()?;
        if self.backend_context.cuda.split_k_attn.is_none() {
            self.backend_context.cuda.split_k_attn = self.options.split_k_attn;
        }
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), String> {
        if let (Some(alias), Some(canonical)) = (
            self.options.split_k_attn,
            self.backend_context.cuda.split_k_attn,
        ) {
            if alias != canonical {
                return Err(
                    "split-K alias conflicts with backend_context.cuda.split_k_attn".into(),
                );
            }
        }
        self.backend_context
            .validate()
            .map_err(|error| error.to_string())?;
        if self.kv_capacity == Some(0)
            || self.options.kv_max_blocks == Some(0)
            || self.paged_max_seqs == 0
        {
            return Err("Llama execution capacity and page counts must be positive".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineConfig, RuntimeConfigSnapshot};

    #[test]
    fn split_k_alias_normalizes_without_overriding_canonical_policy() {
        let mut direct = LlamaExecutionConfig::default();
        direct.options.split_k_attn = Some(false);
        let normalized = direct.clone().normalized().unwrap();
        assert_eq!(normalized.backend_context.cuda.split_k_attn, Some(false));
        assert_eq!(direct.backend_context.cuda.split_k_attn, None);
        direct.backend_context.cuda.split_k_attn = Some(true);
        assert!(direct.validate().is_err());
        assert!(direct.normalized().is_err());
        let mut engine = EngineConfig::default();
        engine.runtime.backend_context.cuda.split_k_attn = Some(true);
        let canonical = engine.llama_execution_config().normalized().unwrap();
        assert_eq!(canonical.options.split_k_attn, None);
        assert_eq!(canonical.backend_context.cuda.split_k_attn, Some(true));
        engine.runtime.llama.split_k_attn = Some(false);
        assert!(engine.llama_execution_config().validate().is_err());
        engine.runtime.llama.split_k_attn = Some(true);
        assert!(engine.llama_execution_config().normalized().is_ok());
    }

    #[test]
    fn llama_instance_defaults_preserve_model_pool_and_graph_policy() {
        let engine = EngineConfig::default();
        let execution = engine.llama_execution_config();
        assert_eq!(execution, LlamaExecutionConfig::default());
        assert_eq!(execution.options.kv_max_blocks, None);
        assert!(!engine.runtime.gptq_gidx_trace);
        assert_ne!(engine.kv_cache.max_blocks, 0);
    }

    #[test]
    fn llama_instance_config_snapshots_do_not_mutate_another_assembly() {
        let mut first = EngineConfig::default();
        first
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([
                ("FERRUM_KV_CAPACITY", "128"),
                ("FERRUM_KV_MAX_BLOCKS", "19"),
                ("FERRUM_PAGED_MAX_SEQS", "3"),
                ("FERRUM_PREFIX_CACHE", "1"),
                ("FERRUM_BATCHED_GRAPH", "1"),
                ("FERRUM_UNIFIED_GRAPH", "1"),
                ("FERRUM_NAN_TRACE", "source2,layer5"),
                ("FERRUM_GPTQ_GIDX_TRACE", "1"),
            ]))
            .unwrap();
        let frozen = first.llama_execution_config();
        let mut second = EngineConfig::default();
        second
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([
                ("FERRUM_KV_CAPACITY", "64"),
                ("FERRUM_KV_MAX_BLOCKS", "7"),
                ("FERRUM_PAGED_MAX_SEQS", "2"),
                ("FERRUM_PREFIX_CACHE", "0"),
                ("FERRUM_BATCHED_GRAPH", "0"),
                ("FERRUM_UNIFIED_GRAPH", "0"),
                ("FERRUM_NAN_TRACE", "0"),
                ("FERRUM_GPTQ_GIDX_TRACE", "0"),
            ]))
            .unwrap();
        assert_eq!(first.llama_execution_config(), frozen);
        assert_eq!(frozen.kv_capacity, Some(128));
        assert_eq!(frozen.options.kv_max_blocks, Some(19));
        assert!(frozen.prefix_cache && frozen.batched_graph && frozen.options.unified_graph);
        assert_eq!(
            frozen.options.nan_trace,
            LlamaNanTraceConfig::Layers(vec![2, 5])
        );
        assert!(first.runtime.gptq_gidx_trace);
        assert!(!second.runtime.gptq_gidx_trace);
        first
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::default())
            .unwrap();
        assert_eq!(first.llama_execution_config(), frozen);
        let roundtrip: EngineConfig =
            serde_json::from_value(serde_json::to_value(&first).unwrap()).unwrap();
        assert_eq!(roundtrip.llama_execution_config(), frozen);
    }

    #[test]
    fn llama_instance_explicit_typed_controls_do_not_require_a_snapshot() {
        let mut engine = EngineConfig::default();
        engine.runtime.kv_capacity = Some(31);
        engine.runtime.llama.kv_max_blocks = Some(5);
        engine.runtime.llama.greedy_argmax = true;
        engine.scheduler.max_running_requests = 2;
        let execution = engine.llama_execution_config();
        assert_eq!(execution.kv_capacity, Some(31));
        assert_eq!(execution.paged_max_seqs, 2);
        assert_eq!(execution.options.kv_max_blocks, Some(5));
        assert!(execution.options.greedy_argmax);
        execution.validate().unwrap();
    }
}
