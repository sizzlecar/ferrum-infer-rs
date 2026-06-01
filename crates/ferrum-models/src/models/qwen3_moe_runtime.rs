//! Runtime environment snapshot for Qwen3-MoE model paths.

use std::collections::HashMap;

use ferrum_types::RuntimeConfigSnapshot;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Qwen3MoeRuntimeEnv {
    pub(crate) decode_op_profile: bool,
    pub(crate) fa2_direct_ffi: bool,
    pub(crate) fa2_source: bool,
    pub(crate) fa_layout_varlen: bool,
    pub(crate) greedy_argmax: bool,
    pub(crate) initial_scratch_tokens: usize,
    pub(crate) moe_batched_decode_enabled: bool,
    pub(crate) moe_batched_enabled: bool,
    pub(crate) moe_batch_threshold: usize,
    pub(crate) moe_bucketed: bool,
    pub(crate) moe_graph_requested: bool,
    pub(crate) moe_graph_vllm_clean: bool,
    pub(crate) moe_prefill_threshold: usize,
    pub(crate) moe_profile: bool,
    pub(crate) paged_max_seqs: usize,
    pub(crate) prefix_cache: bool,
    pub(crate) qwen_unified_prefill: bool,
    pub(crate) qwen_unified_trace: bool,
    pub(crate) rbd_prof: bool,
    pub(crate) unified_greedy_argmax: bool,
    pub(crate) unified_layer_prof: bool,
    pub(crate) unified_layer_prof_every: u64,
    pub(crate) unified_layer_prof_max_m: Option<usize>,
    pub(crate) unified_layer_prof_min_seqs: Option<usize>,
    pub(crate) use_vllm_paged_attn: bool,
    pub(crate) vllm_decode_varlen: bool,
    pub(crate) vllm_varlen_tiled_q4: bool,
    kv_capacity: Option<usize>,
    metal_paged_kv: Option<bool>,
}

impl Qwen3MoeRuntimeEnv {
    pub(crate) fn from_env() -> Self {
        Self::from_env_vars(std::env::vars())
    }

    pub(crate) fn from_runtime_config_snapshot(snapshot: &RuntimeConfigSnapshot) -> Self {
        let mut entries: Vec<(String, String)> = snapshot
            .entries
            .iter()
            .map(|entry| (entry.key.clone(), entry.effective_value.clone()))
            .collect();
        entries.extend(std::env::vars().filter(|(key, _)| {
            key.starts_with("FERRUM_") || key == "MTL_CAPTURE_ENABLED"
        }));
        Self::from_env_vars(entries)
    }

    fn from_env_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let vars: HashMap<String, String> = vars
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        let fa2_source = trueish(vars.get("FERRUM_FA2_SOURCE"));
        let fa2_direct_ffi = match vars.get("FERRUM_FA2_DIRECT_FFI").map(String::as_str) {
            Some("0" | "false" | "FALSE" | "off" | "OFF") => false,
            Some("1" | "true" | "TRUE" | "on" | "ON") => true,
            Some(_) => true,
            None => vars.contains_key("FERRUM_FA2_DIRECT_FFI_SHIM"),
        };
        let greedy_argmax = vars.get("FERRUM_GREEDY_ARGMAX").is_some_and(|v| v == "1");

        Self {
            decode_op_profile: vars.contains_key("FERRUM_DECODE_OP_PROFILE"),
            fa2_direct_ffi,
            fa2_source,
            fa_layout_varlen: vars
                .get("FERRUM_FA_LAYOUT_VARLEN")
                .is_some_and(|v| v == "1"),
            greedy_argmax,
            initial_scratch_tokens: positive_usize(&vars, "FERRUM_MAX_BATCHED_TOKENS")
                .unwrap_or(2048),
            moe_batched_decode_enabled: vars
                .get("FERRUM_MOE_BATCHED_DECODE")
                .is_none_or(|v| v != "0"),
            moe_batched_enabled: vars.get("FERRUM_MOE_BATCHED").is_none_or(|v| v != "0"),
            moe_batch_threshold: positive_usize(&vars, "FERRUM_MOE_BATCH_THRESHOLD").unwrap_or(4),
            moe_bucketed: vars.get("FERRUM_MOE_BUCKETED").is_none_or(|v| v != "0"),
            moe_graph_requested: vars.get("FERRUM_MOE_GRAPH").is_some_and(|v| v == "1"),
            moe_graph_vllm_clean: vars.get("FERRUM_VLLM_MOE").is_some_and(|v| v == "1")
                && !vars.get("FERRUM_MOE_HOST_ROUTE").is_some_and(|v| v == "1"),
            moe_prefill_threshold: positive_usize(&vars, "FERRUM_MOE_PREFILL_THRESHOLD")
                .unwrap_or(32),
            moe_profile: vars.contains_key("FERRUM_MOE_PROFILE"),
            paged_max_seqs: positive_usize(&vars, "FERRUM_PAGED_MAX_SEQS").unwrap_or(32),
            prefix_cache: vars.get("FERRUM_PREFIX_CACHE").is_some_and(|v| v == "1"),
            qwen_unified_prefill: vars
                .get("FERRUM_QWEN_UNIFIED_PREFILL")
                .is_none_or(|v| v != "0"),
            qwen_unified_trace: vars
                .get("FERRUM_QWEN_UNIFIED_TRACE")
                .is_some_and(|v| v == "1"),
            rbd_prof: vars.contains_key("FERRUM_RBD_PROF"),
            unified_greedy_argmax: greedy_argmax
                && vars
                    .get("FERRUM_UNIFIED_GREEDY_ARGMAX")
                    .is_none_or(|v| v != "0"),
            unified_layer_prof: vars
                .get("FERRUM_UNIFIED_LAYER_PROF")
                .is_some_and(|v| v != "0"),
            unified_layer_prof_every: vars
                .get("FERRUM_UNIFIED_LAYER_PROF_EVERY")
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(16),
            unified_layer_prof_max_m: positive_usize(&vars, "FERRUM_UNIFIED_LAYER_PROF_MAX_M"),
            unified_layer_prof_min_seqs: positive_usize(
                &vars,
                "FERRUM_UNIFIED_LAYER_PROF_MIN_SEQS",
            ),
            use_vllm_paged_attn: vars
                .get("FERRUM_USE_VLLM_PAGED_ATTN")
                .is_none_or(|v| v != "0"),
            vllm_decode_varlen: vars
                .get("FERRUM_VLLM_DECODE_VARLEN")
                .is_some_and(|v| v == "1"),
            vllm_varlen_tiled_q4: vars
                .get("FERRUM_VLLM_VARLEN_TILED_Q4")
                .is_some_and(|v| v == "1"),
            kv_capacity: positive_usize(&vars, "FERRUM_KV_CAPACITY"),
            metal_paged_kv: vars.get("FERRUM_METAL_PAGED_KV").map(|v| v != "0"),
        }
    }

    pub(crate) fn kv_capacity(&self, model_max: usize) -> usize {
        const DEFAULT_KV_CAPACITY: usize = 512;
        self.kv_capacity
            .map(|cap| cap.min(model_max))
            .unwrap_or_else(|| model_max.min(DEFAULT_KV_CAPACITY))
    }

    pub(crate) fn metal_paged_kv_enabled(&self, backend_default: bool) -> bool {
        self.metal_paged_kv.unwrap_or(backend_default)
    }

    pub(crate) fn unified_layer_prof_selected(&self, m_total: usize, num_seqs: usize) -> bool {
        if !self.unified_layer_prof {
            return false;
        }
        if let Some(max_m) = self.unified_layer_prof_max_m {
            if m_total > max_m {
                return false;
            }
        }
        if let Some(min_seqs) = self.unified_layer_prof_min_seqs {
            if num_seqs < min_seqs {
                return false;
            }
        }
        true
    }
}

fn positive_usize(vars: &HashMap<String, String>, name: &str) -> Option<usize> {
    vars.get(name)
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
}

fn trueish(value: Option<&String>) -> bool {
    matches!(
        value.map(String::as_str),
        Some("1" | "true" | "TRUE" | "on" | "ON")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_moe_runtime_env_parses_m3_startup_knobs() {
        let env = Qwen3MoeRuntimeEnv::from_env_vars([
            ("FERRUM_DECODE_OP_PROFILE", "1"),
            ("FERRUM_FA2_DIRECT_FFI_SHIM", "/tmp/shim.so"),
            ("FERRUM_FA_LAYOUT_VARLEN", "1"),
            ("FERRUM_GREEDY_ARGMAX", "1"),
            ("FERRUM_MAX_BATCHED_TOKENS", "4096"),
            ("FERRUM_METAL_PAGED_KV", "0"),
            ("FERRUM_MOE_BATCHED", "0"),
            ("FERRUM_MOE_BATCHED_DECODE", "0"),
            ("FERRUM_MOE_BATCH_THRESHOLD", "7"),
            ("FERRUM_MOE_BUCKETED", "0"),
            ("FERRUM_MOE_GRAPH", "1"),
            ("FERRUM_MOE_PREFILL_THRESHOLD", "33"),
            ("FERRUM_MOE_PROFILE", "1"),
            ("FERRUM_PAGED_MAX_SEQS", "64"),
            ("FERRUM_PREFIX_CACHE", "1"),
            ("FERRUM_QWEN_UNIFIED_PREFILL", "0"),
            ("FERRUM_QWEN_UNIFIED_TRACE", "1"),
            ("FERRUM_RBD_PROF", "1"),
            ("FERRUM_UNIFIED_GREEDY_ARGMAX", "0"),
            ("FERRUM_UNIFIED_LAYER_PROF", "1"),
            ("FERRUM_UNIFIED_LAYER_PROF_EVERY", "9"),
            ("FERRUM_UNIFIED_LAYER_PROF_MAX_M", "128"),
            ("FERRUM_UNIFIED_LAYER_PROF_MIN_SEQS", "4"),
            ("FERRUM_USE_VLLM_PAGED_ATTN", "0"),
            ("FERRUM_VLLM_DECODE_VARLEN", "1"),
            ("FERRUM_VLLM_MOE", "1"),
            ("FERRUM_VLLM_VARLEN_TILED_Q4", "1"),
        ]);

        assert!(env.decode_op_profile);
        assert!(env.fa2_direct_ffi);
        assert!(!env.fa2_source);
        assert!(env.fa_layout_varlen);
        assert!(env.greedy_argmax);
        assert_eq!(env.initial_scratch_tokens, 4096);
        assert!(!env.metal_paged_kv_enabled(true));
        assert!(!env.moe_batched_enabled);
        assert!(!env.moe_batched_decode_enabled);
        assert_eq!(env.moe_batch_threshold, 7);
        assert!(!env.moe_bucketed);
        assert!(env.moe_graph_requested);
        assert!(env.moe_graph_vllm_clean);
        assert_eq!(env.moe_prefill_threshold, 33);
        assert!(env.moe_profile);
        assert_eq!(env.paged_max_seqs, 64);
        assert!(env.prefix_cache);
        assert!(!env.qwen_unified_prefill);
        assert!(env.qwen_unified_trace);
        assert!(env.rbd_prof);
        assert!(!env.unified_greedy_argmax);
        assert!(env.unified_layer_prof_selected(128, 4));
        assert!(!env.unified_layer_prof_selected(129, 4));
        assert!(!env.unified_layer_prof_selected(128, 3));
        assert_eq!(env.unified_layer_prof_every, 9);
        assert!(!env.use_vllm_paged_attn);
        assert!(env.vllm_decode_varlen);
        assert!(env.vllm_varlen_tiled_q4);
    }

    #[test]
    fn qwen3_moe_runtime_env_uses_defaults_and_bounds() {
        let env = Qwen3MoeRuntimeEnv::from_env_vars([
            ("FERRUM_KV_CAPACITY", "4096"),
            ("FERRUM_MOE_HOST_ROUTE", "1"),
            ("FERRUM_MOE_GRAPH", "1"),
            ("FERRUM_VLLM_MOE", "1"),
        ]);

        assert_eq!(env.initial_scratch_tokens, 2048);
        assert_eq!(env.kv_capacity(1024), 1024);
        assert_eq!(env.kv_capacity(8192), 4096);
        assert!(env.metal_paged_kv_enabled(true));
        assert!(!env.metal_paged_kv_enabled(false));
        assert!(env.moe_batched_enabled);
        assert!(env.moe_batched_decode_enabled);
        assert_eq!(env.moe_batch_threshold, 4);
        assert_eq!(env.moe_prefill_threshold, 32);
        assert_eq!(env.paged_max_seqs, 32);
        assert!(!env.moe_graph_vllm_clean);
        assert!(env.qwen_unified_prefill);
        assert!(env.use_vllm_paged_attn);
        assert!(!env.unified_layer_prof_selected(1, 1));
    }

    #[test]
    fn qwen3_moe_runtime_env_keeps_fa2_source_distinct_from_direct_ffi() {
        let env = Qwen3MoeRuntimeEnv::from_env_vars([("FERRUM_FA2_SOURCE", "1")]);

        assert!(env.fa2_source);
        assert!(!env.fa2_direct_ffi);
    }

    #[test]
    fn qwen3_moe_runtime_env_can_use_typed_snapshot_without_process_env() {
        let snapshot = RuntimeConfigSnapshot::from_entries([
            ferrum_types::RuntimeConfigEntry::new(
                "FERRUM_FA_LAYOUT_VARLEN",
                "1",
                ferrum_types::RuntimeConfigSource::Default,
            ),
            ferrum_types::RuntimeConfigEntry::new(
                "FERRUM_FA2_SOURCE",
                "1",
                ferrum_types::RuntimeConfigSource::Default,
            ),
            ferrum_types::RuntimeConfigEntry::new(
                "FERRUM_MAX_BATCHED_TOKENS",
                "3072",
                ferrum_types::RuntimeConfigSource::MemoryProfile,
            ),
            ferrum_types::RuntimeConfigEntry::new(
                "FERRUM_MOE_GRAPH",
                "1",
                ferrum_types::RuntimeConfigSource::Default,
            ),
            ferrum_types::RuntimeConfigEntry::new(
                "FERRUM_VLLM_MOE",
                "1",
                ferrum_types::RuntimeConfigSource::Default,
            ),
        ]);

        let env = Qwen3MoeRuntimeEnv::from_runtime_config_snapshot(&snapshot);

        assert!(env.fa_layout_varlen);
        assert!(env.fa2_source);
        assert!(!env.fa2_direct_ffi);
        assert_eq!(env.initial_scratch_tokens, 3072);
        assert!(env.moe_graph_requested);
        assert!(env.moe_graph_vllm_clean);
    }
}
