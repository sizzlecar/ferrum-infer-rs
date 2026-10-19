//! CLI runtime environment compatibility bridge.
//!
//! Product control-plane values should be represented as typed
//! `RuntimeConfigEntry` values first. This module contains the small remaining
//! bridge that materializes those entries into process env only for backend
//! paths that have not yet been converted to typed config.

use ferrum_types::{RuntimeConfigEntry, RuntimeConfigSnapshot, RuntimeConfigSource};

fn model_construction_is_typed_only(key: &str) -> bool {
    matches!(
        key,
        "FERRUM_KV_MAX_BLOCKS"
            | "FERRUM_METAL_PAGED_KV"
            | "FERRUM_PAGED_MAX_SEQS"
            | "FERRUM_CUDA_GRAPH"
            | "FERRUM_BATCHED_GRAPH"
            | "FERRUM_UNIFIED_GRAPH"
            | "FERRUM_UNIFIED_GRAPH_LAYERS_ONLY"
            | "FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER"
            | "FERRUM_GREEDY_ARGMAX"
            | "FERRUM_NAN_TRACE"
            | "FERRUM_OP_DUMP"
            | "FERRUM_LAYER_DUMP"
            | "FERRUM_GPTQ_GIDX_TRACE"
    )
    // KV_CAPACITY and SPLIT_K_ATTN still have retained CUDA backend consumers;
    // PREFIX_CACHE also has HTTP metadata compatibility. Their separate
    // environment bridges retire with those consumers, not by dropping input.
}

/// Reject retired controls before source resolution or device preparation.
pub fn capture_supported_runtime_environment() -> ferrum_types::Result<RuntimeConfigSnapshot> {
    let snapshot = RuntimeConfigSnapshot::capture_current();
    snapshot
        .validate_retired_options()
        .map_err(|(key, reason)| ferrum_types::FerrumError::config(format!("{key}: {reason}")))?;
    Ok(snapshot)
}

pub fn push_cli_runtime_entry(
    entries: &mut Vec<RuntimeConfigEntry>,
    key: &str,
    value: Option<&str>,
) {
    if let Some(value) = value.filter(|value| !value.trim().is_empty()) {
        entries.push(RuntimeConfigEntry::new(
            key,
            value.to_string(),
            RuntimeConfigSource::Cli,
        ));
    }
}

pub fn push_cli_runtime_usize(
    entries: &mut Vec<RuntimeConfigEntry>,
    key: &str,
    value: Option<usize>,
) {
    if let Some(value) = value {
        entries.push(RuntimeConfigEntry::new(
            key,
            value.to_string(),
            RuntimeConfigSource::Cli,
        ));
    }
}

pub fn materialize_runtime_env_defaults(entries: &[RuntimeConfigEntry]) -> Vec<String> {
    let mut materialized = Vec::new();
    for entry in entries {
        if model_construction_is_typed_only(&entry.key)
            || entry.key == ferrum_types::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY
            || entry.key == ferrum_types::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY
            || entry.key == ferrum_types::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY
            || entry.key == ferrum_types::CAUSAL_DECODE_KV_REUSE_CONFIG_KEY
            || entry.key == ferrum_types::SEQUENCE_BACKING_GROWTH_CONFIG_KEY
            || entry.key == ferrum_types::Q4_MMVQ_WEIGHT_DECODE_CONFIG_KEY
            || entry.key == ferrum_types::Q5_MMVQ_WEIGHT_DECODE_CONFIG_KEY
            || entry.key == ferrum_types::Q8_PROJECTION_WEIGHT_PREPARATION_CONFIG_KEY
        {
            continue;
        }
        if std::env::var_os(&entry.key).is_none() {
            std::env::set_var(&entry.key, &entry.effective_value);
            materialized.push(entry.key.clone());
        }
    }
    materialized
}

pub fn materialize_runtime_env_effective(snapshot: &RuntimeConfigSnapshot) -> Vec<String> {
    let mut materialized = Vec::new();
    for entry in &snapshot.entries {
        if model_construction_is_typed_only(&entry.key)
            || entry.key == ferrum_types::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY
            || entry.key == ferrum_types::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY
            || entry.key == ferrum_types::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY
            || entry.key == ferrum_types::CAUSAL_DECODE_KV_REUSE_CONFIG_KEY
            || entry.key == ferrum_types::SEQUENCE_BACKING_GROWTH_CONFIG_KEY
            || entry.key == ferrum_types::Q4_MMVQ_WEIGHT_DECODE_CONFIG_KEY
            || entry.key == ferrum_types::Q5_MMVQ_WEIGHT_DECODE_CONFIG_KEY
            || entry.key == ferrum_types::Q8_PROJECTION_WEIGHT_PREPARATION_CONFIG_KEY
        {
            continue;
        }
        if entry.source != RuntimeConfigSource::Env {
            std::env::set_var(&entry.key, &entry.effective_value);
            materialized.push(entry.key.clone());
        }
    }
    materialized
}

pub fn runtime_snapshot_value<'a>(
    snapshot: &'a RuntimeConfigSnapshot,
    key: &str,
) -> Option<&'a str> {
    snapshot
        .entries
        .iter()
        .find(|entry| entry.key == key)
        .map(|entry| entry.effective_value.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(entries: &[(&str, &str, RuntimeConfigSource)]) -> RuntimeConfigSnapshot {
        RuntimeConfigSnapshot::from_entries(
            entries
                .iter()
                .map(|(key, value, source)| RuntimeConfigEntry::new(*key, *value, *source)),
        )
    }

    #[test]
    fn model_instance_options_are_never_materialized_as_environment() {
        for (key, value) in [
            ("FERRUM_KV_MAX_BLOCKS", "17"),
            ("FERRUM_METAL_PAGED_KV", "0"),
            ("FERRUM_PAGED_MAX_SEQS", "3"),
            ("FERRUM_CUDA_GRAPH", "1"),
            ("FERRUM_BATCHED_GRAPH", "1"),
            ("FERRUM_UNIFIED_GRAPH", "1"),
            ("FERRUM_UNIFIED_GRAPH_LAYERS_ONLY", "1"),
            ("FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER", "1"),
            ("FERRUM_GREEDY_ARGMAX", "1"),
            ("FERRUM_NAN_TRACE", "layer3"),
            ("FERRUM_OP_DUMP", "op-diagnostic"),
            ("FERRUM_LAYER_DUMP", "layer-diagnostic"),
            ("FERRUM_GPTQ_GIDX_TRACE", "1"),
        ] {
            let original = std::env::var_os(key);
            for source in [
                RuntimeConfigSource::Default,
                RuntimeConfigSource::ConfigFile,
                RuntimeConfigSource::Cli,
                RuntimeConfigSource::Env,
            ] {
                let current = snapshot(&[(key, value, source)]);
                assert!(materialize_runtime_env_defaults(&current.entries).is_empty());
                assert!(materialize_runtime_env_effective(&current).is_empty());
                assert_eq!(std::env::var_os(key), original);
                let mut engine = ferrum_types::EngineConfig::default();
                engine.apply_runtime_config_snapshot(&current).unwrap();
                assert_eq!(runtime_snapshot_value(&current, key), Some(value));
            }
        }
    }

    #[test]
    fn segment_binding_owner_view_provenance_is_never_materialized_as_environment() {
        let key = ferrum_types::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY;
        let original = std::env::var_os(key);
        for source in [
            RuntimeConfigSource::Default,
            RuntimeConfigSource::ConfigFile,
            RuntimeConfigSource::Cli,
        ] {
            let current = snapshot(&[(key, "indexed", source)]);
            assert!(materialize_runtime_env_defaults(&current.entries).is_empty());
            assert!(materialize_runtime_env_effective(&current).is_empty());
            assert_eq!(std::env::var_os(key), original);
            assert_eq!(runtime_snapshot_value(&current, key), Some("indexed"));
            assert_eq!(current.entries[0].source, source);
        }
    }

    #[test]
    fn causal_decode_preparation_provenance_is_never_materialized_as_environment() {
        let key = ferrum_types::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY;
        let original = std::env::var_os(key);
        for source in [
            RuntimeConfigSource::Default,
            RuntimeConfigSource::ConfigFile,
            RuntimeConfigSource::Cli,
        ] {
            let current = snapshot(&[(key, "packed", source)]);
            assert!(materialize_runtime_env_defaults(&current.entries).is_empty());
            assert!(materialize_runtime_env_effective(&current).is_empty());
            assert_eq!(std::env::var_os(key), original);
            assert_eq!(runtime_snapshot_value(&current, key), Some("packed"));
            assert_eq!(current.entries[0].source, source);
        }
    }

    #[test]
    fn causal_decode_attention_grouping_provenance_is_never_materialized_as_environment() {
        let key = ferrum_types::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY;
        let original = std::env::var_os(key);
        for mode in ["indexed", "fixed-capacity", "fixed-execution-capacity"] {
            for source in [
                RuntimeConfigSource::Default,
                RuntimeConfigSource::ConfigFile,
                RuntimeConfigSource::Cli,
            ] {
                let current = snapshot(&[(key, mode, source)]);
                assert!(materialize_runtime_env_defaults(&current.entries).is_empty());
                assert!(materialize_runtime_env_effective(&current).is_empty());
                assert_eq!(std::env::var_os(key), original);
                assert_eq!(runtime_snapshot_value(&current, key), Some(mode));
                assert_eq!(current.entries[0].source, source);
            }
        }
    }

    #[test]
    fn q4_mmvq_weight_decode_provenance_is_never_materialized_as_environment() {
        let key = ferrum_types::Q4_MMVQ_WEIGHT_DECODE_CONFIG_KEY;
        let original = std::env::var_os(key);
        for source in [
            RuntimeConfigSource::Default,
            RuntimeConfigSource::ConfigFile,
            RuntimeConfigSource::Cli,
        ] {
            let current = snapshot(&[(key, "shared-m8", source)]);
            assert!(materialize_runtime_env_defaults(&current.entries).is_empty());
            assert!(materialize_runtime_env_effective(&current).is_empty());
            assert_eq!(std::env::var_os(key), original);
            assert_eq!(runtime_snapshot_value(&current, key), Some("shared-m8"));
            assert_eq!(current.entries[0].source, source);
        }
    }
    #[test]
    fn q5_mmvq_weight_decode_provenance_is_never_materialized_as_environment() {
        let key = ferrum_types::Q5_MMVQ_WEIGHT_DECODE_CONFIG_KEY;
        let original = std::env::var_os(key);
        for source in [
            RuntimeConfigSource::Default,
            RuntimeConfigSource::ConfigFile,
            RuntimeConfigSource::Cli,
        ] {
            let current = snapshot(&[(key, "shared-m8", source)]);
            assert!(materialize_runtime_env_defaults(&current.entries).is_empty());
            assert!(materialize_runtime_env_effective(&current).is_empty());
            assert_eq!(std::env::var_os(key), original);
            assert_eq!(runtime_snapshot_value(&current, key), Some("shared-m8"));
            assert_eq!(current.entries[0].source, source);
        }
    }
    #[test]
    fn q8_projection_weight_preparation_provenance_is_never_materialized_as_environment() {
        let key = ferrum_types::Q8_PROJECTION_WEIGHT_PREPARATION_CONFIG_KEY;
        let original = std::env::var_os(key);
        for source in [
            RuntimeConfigSource::Default,
            RuntimeConfigSource::ConfigFile,
            RuntimeConfigSource::Cli,
        ] {
            let current = snapshot(&[(key, "predecoded-f32", source)]);
            assert!(materialize_runtime_env_defaults(&current.entries).is_empty());
            assert!(materialize_runtime_env_effective(&current).is_empty());
            assert_eq!(std::env::var_os(key), original);
            assert_eq!(
                runtime_snapshot_value(&current, key),
                Some("predecoded-f32")
            );
            assert_eq!(current.entries[0].source, source);
        }
        let from_environment = snapshot(&[(key, "predecoded-f32", RuntimeConfigSource::Env)]);
        assert!(ferrum_types::EngineConfig::default()
            .apply_runtime_config_snapshot(&from_environment)
            .is_err());
    }

    #[test]
    fn causal_decode_kv_reuse_provenance_is_never_materialized_as_environment() {
        let key = ferrum_types::CAUSAL_DECODE_KV_REUSE_CONFIG_KEY;
        let original = std::env::var_os(key);
        for source in [
            RuntimeConfigSource::Default,
            RuntimeConfigSource::ConfigFile,
            RuntimeConfigSource::Cli,
        ] {
            let current = snapshot(&[(key, "paired-query-heads", source)]);
            assert!(materialize_runtime_env_defaults(&current.entries).is_empty());
            assert!(materialize_runtime_env_effective(&current).is_empty());
            assert_eq!(std::env::var_os(key), original);
            assert_eq!(
                runtime_snapshot_value(&current, key),
                Some("paired-query-heads")
            );
            assert_eq!(current.entries[0].source, source);
        }
        let environment = snapshot(&[(key, "paired-query-heads", RuntimeConfigSource::Env)]);
        assert!(ferrum_types::EngineConfig::default()
            .apply_runtime_config_snapshot(&environment)
            .is_err());
    }

    #[test]
    fn sequence_backing_growth_provenance_is_never_environment() {
        let key = ferrum_types::SEQUENCE_BACKING_GROWTH_CONFIG_KEY;
        let original = std::env::var_os(key);
        for value in ["paged-group4", "paged-group4-grow-only"] {
            for source in [
                RuntimeConfigSource::Default,
                RuntimeConfigSource::ConfigFile,
                RuntimeConfigSource::Cli,
            ] {
                let current = snapshot(&[(key, value, source)]);
                assert!(materialize_runtime_env_defaults(&current.entries).is_empty());
                assert!(materialize_runtime_env_effective(&current).is_empty());
                assert_eq!(std::env::var_os(key), original);
            }
            assert!(ferrum_types::EngineConfig::default()
                .apply_runtime_config_snapshot(&snapshot(&[(key, value, RuntimeConfigSource::Env)]))
                .is_err());
        }
    }
}
