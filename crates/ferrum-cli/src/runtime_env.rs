//! CLI runtime environment compatibility bridge.
//!
//! Product control-plane values should be represented as typed
//! `RuntimeConfigEntry` values first. This module contains the small remaining
//! bridge that materializes those entries into process env only for backend
//! paths that have not yet been converted to typed config.

use ferrum_types::{RuntimeConfigEntry, RuntimeConfigSnapshot, RuntimeConfigSource};

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
        if entry.key == ferrum_types::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY
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
        if entry.key == ferrum_types::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY
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
