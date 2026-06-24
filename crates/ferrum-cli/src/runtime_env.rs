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
        if entry.source != RuntimeConfigSource::Env {
            std::env::set_var(&entry.key, &entry.effective_value);
            materialized.push(entry.key.clone());
        }
    }
    materialized
}

/// Default-OFF MoE CUDA graph policy for Qwen3-MoE startup profiles.
///
/// The returned entries are only values absent from `current`. This preserves
/// explicit env/config overrides and lets callers keep source attribution when
/// they later merge a full startup snapshot.
pub fn moe_graph_default_entries(
    current: &RuntimeConfigSnapshot,
    source: RuntimeConfigSource,
) -> Vec<RuntimeConfigEntry> {
    let mut entries = Vec::new();
    let moe_graph_enabled = match runtime_snapshot_value(current, "FERRUM_MOE_GRAPH") {
        Some(value) => value == "1",
        None => {
            entries.push(RuntimeConfigEntry::new("FERRUM_MOE_GRAPH", "0", source));
            false
        }
    };

    if !moe_graph_enabled {
        return entries;
    }

    #[cfg(feature = "vllm-moe-marlin")]
    {
        if runtime_snapshot_value(current, "FERRUM_VLLM_MOE").is_none() {
            entries.push(RuntimeConfigEntry::new("FERRUM_VLLM_MOE", "1", source));
        }
    }

    entries
}

pub fn warn_if_moe_graph_needs_unbuilt_vllm_moe(_snapshot: &RuntimeConfigSnapshot) {
    #[cfg(not(feature = "vllm-moe-marlin"))]
    if runtime_snapshot_value(_snapshot, "FERRUM_MOE_GRAPH") == Some("1")
        && runtime_snapshot_value(_snapshot, "FERRUM_VLLM_MOE") != Some("1")
    {
        eprintln!(
            "[auto-size] MOE_GRAPH=1 requested, but vllm-moe-marlin is not built; graph capture requires FERRUM_VLLM_MOE=1"
        );
    }
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
    fn moe_graph_defaults_add_missing_graph_default() {
        let entries = moe_graph_default_entries(
            &RuntimeConfigSnapshot::default(),
            RuntimeConfigSource::Default,
        );
        let resolved = RuntimeConfigSnapshot::from_entries(entries);
        let graph = runtime_snapshot_value(&resolved, "FERRUM_MOE_GRAPH");

        assert_eq!(graph, Some("0"));
        assert_eq!(resolved.entries.len(), 1);
    }

    #[test]
    fn moe_graph_defaults_respect_forced_off_graph() {
        let current = snapshot(&[("FERRUM_MOE_GRAPH", "0", RuntimeConfigSource::Env)]);
        let entries = moe_graph_default_entries(&current, RuntimeConfigSource::Default);

        assert!(entries.is_empty());
    }

    #[test]
    fn moe_graph_defaults_complete_graph_enabled_snapshot() {
        let current = snapshot(&[("FERRUM_MOE_GRAPH", "1", RuntimeConfigSource::ConfigFile)]);
        let entries = moe_graph_default_entries(&current, RuntimeConfigSource::Default);
        let resolved = RuntimeConfigSnapshot::from_entries(entries);

        assert_eq!(runtime_snapshot_value(&resolved, "FERRUM_MOE_GRAPH"), None);
        #[cfg(feature = "vllm-moe-marlin")]
        assert_eq!(
            runtime_snapshot_value(&resolved, "FERRUM_VLLM_MOE"),
            Some("1")
        );
        #[cfg(not(feature = "vllm-moe-marlin"))]
        assert!(resolved.entries.is_empty());
    }
}
