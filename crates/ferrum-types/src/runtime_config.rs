//! Runtime configuration snapshot and small env parsing helpers.
//!
//! This is intentionally a narrow data surface first: it makes effective
//! `FERRUM_*` overrides visible in health and bench artifacts while the
//! hot-path env reads are migrated to typed config structs.

use serde::{Deserialize, Serialize};
use std::sync::RwLock;
use std::{collections::BTreeMap, path::PathBuf};

/// Process-wide runtime snapshot, installed once at the composition root.
///
/// This is the single env-bridge seam the test-architecture goal asks for:
/// the CLI (`serve`/`run`/`bench`) captures `FERRUM_*` via
/// [`RuntimeConfigSnapshot::capture_current`] and installs it here when it
/// applies the snapshot to the engine config; model code downstream reads
/// [`active_runtime_snapshot`] instead of `std::env`, so no model/engine
/// module freezes its own env config. Re-installable (RwLock, not OnceLock)
/// so per-construction test paths can vary it after `std::env::set_var`.
static ACTIVE_SNAPSHOT: RwLock<Option<RuntimeConfigSnapshot>> = RwLock::new(None);

/// Install the process-wide runtime snapshot resolved at the composition root.
pub fn install_runtime_snapshot(snapshot: RuntimeConfigSnapshot) {
    *ACTIVE_SNAPSHOT
        .write()
        .expect("runtime snapshot lock poisoned") = Some(snapshot);
}

/// The installed runtime snapshot, or an empty snapshot when none was
/// installed (unit tests that do not exercise runtime knobs see defaults).
pub fn active_runtime_snapshot() -> RuntimeConfigSnapshot {
    ACTIVE_SNAPSHOT
        .read()
        .expect("runtime snapshot lock poisoned")
        .clone()
        .unwrap_or_default()
}

/// Stable snapshot of non-default runtime configuration visible to the process.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeConfigSnapshot {
    /// Sorted by key for stable JSON and machine-readable diffs.
    pub entries: Vec<RuntimeConfigEntry>,
}

impl RuntimeConfigSnapshot {
    /// Capture all currently set `FERRUM_*` env overrides.
    pub fn capture_current() -> Self {
        Self::from_env_vars(std::env::vars())
    }

    /// Build a snapshot from a supplied environment map or iterator.
    pub fn from_env_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let mut sorted = BTreeMap::new();
        for (key, value) in vars {
            let key = key.into();
            if key.starts_with("FERRUM_") {
                sorted.insert(key, value.into());
            }
        }

        Self {
            entries: sorted
                .into_iter()
                .map(|(key, effective_value)| RuntimeConfigEntry {
                    affects: infer_effects(&key),
                    key,
                    effective_value,
                    source: RuntimeConfigSource::Env,
                })
                .collect(),
        }
    }

    /// Build a stable snapshot from explicit entries. Later entries for the
    /// same key replace earlier entries.
    pub fn from_entries<I>(entries: I) -> Self
    where
        I: IntoIterator<Item = RuntimeConfigEntry>,
    {
        let mut sorted = BTreeMap::new();
        for entry in entries {
            sorted.insert(entry.key.clone(), entry);
        }
        Self {
            entries: sorted.into_values().collect(),
        }
    }

    /// Insert or replace one effective value, preserving stable key order.
    pub fn upsert(
        &mut self,
        key: impl Into<String>,
        effective_value: impl Into<String>,
        source: RuntimeConfigSource,
    ) {
        self.upsert_entry(RuntimeConfigEntry::new(key, effective_value, source));
    }

    /// Insert or replace one explicit entry, preserving stable key order.
    pub fn upsert_entry(&mut self, entry: RuntimeConfigEntry) {
        let mut entries = std::mem::take(&mut self.entries);
        entries.retain(|existing| existing.key != entry.key);
        entries.push(entry);
        *self = Self::from_entries(entries);
    }

    /// Return a snapshot with one additional effective value.
    pub fn with_entry(
        mut self,
        key: impl Into<String>,
        effective_value: impl Into<String>,
        source: RuntimeConfigSource,
    ) -> Self {
        self.upsert(key, effective_value, source);
        self
    }
}

/// One effective config value in a runtime snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeConfigEntry {
    pub key: String,
    pub effective_value: String,
    pub source: RuntimeConfigSource,
    pub affects: Vec<RuntimeConfigEffect>,
}

impl RuntimeConfigEntry {
    pub fn new(
        key: impl Into<String>,
        effective_value: impl Into<String>,
        source: RuntimeConfigSource,
    ) -> Self {
        let key = key.into();
        Self {
            affects: infer_effects(&key),
            key,
            effective_value: effective_value.into(),
            source,
        }
    }
}

/// Source of an effective config value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeConfigSource {
    Default,
    ConfigFile,
    Cli,
    Env,
    ScriptCase,
    MemoryProfile,
}

/// Impact classes used by config snapshots and artifact diffs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeConfigEffect {
    Correctness,
    Performance,
    Memory,
    Diagnostics,
}

/// Tri-state env override used by paths that distinguish unset from forced off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvTriState {
    Default,
    ForcedOff,
    ForcedOn,
}

pub fn parse_bool_env_value(raw: &str) -> Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(format!("invalid boolean env value: {other:?}")),
    }
}

pub fn parse_usize_env_value(raw: &str) -> Result<usize, String> {
    raw.trim()
        .parse::<usize>()
        .map_err(|_| format!("invalid integer env value: {raw:?}"))
}

pub fn parse_path_env_value(raw: &str) -> Result<PathBuf, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("path env value must not be empty".to_string());
    }
    Ok(PathBuf::from(trimmed))
}

pub fn parse_tri_state_env_value(raw: Option<&str>) -> Result<EnvTriState, String> {
    let Some(raw) = raw else {
        return Ok(EnvTriState::Default);
    };
    if raw.trim().is_empty() {
        return Ok(EnvTriState::Default);
    }
    Ok(if parse_bool_env_value(raw)? {
        EnvTriState::ForcedOn
    } else {
        EnvTriState::ForcedOff
    })
}

fn infer_effects(key: &str) -> Vec<RuntimeConfigEffect> {
    let mut effects = Vec::new();

    if key.contains("DIAG")
        || key.contains("PROF")
        || key.contains("TRACE")
        || key.contains("DUMP")
        || key.contains("LOG_CONFIG")
        || key.contains("CAPTURE")
        || key.contains("DEBUG")
    {
        effects.push(RuntimeConfigEffect::Diagnostics);
    }

    if key.contains("KV")
        || key.contains("BATCHED_TOKENS")
        || key.contains("PAGED_MAX_SEQS")
        || key.contains("MODEL_LEN")
        || key.contains("STATE_MAX_SLOTS")
        || key.contains("MEMORY")
    {
        effects.push(RuntimeConfigEffect::Memory);
    }

    if key.contains("PREFIX_CACHE")
        || key.contains("MODEL_PATH")
        || key.contains("MODEL_LEN")
        || key.contains("RUNTIME_MEMORY_BUDGET")
        || key.contains("NATIVE")
        || key.contains("ARTIFACT")
        || key.contains("SPEC_")
        || key.contains("REF_")
        || key.contains("DTYPE")
    {
        effects.push(RuntimeConfigEffect::Correctness);
    }

    if effects.is_empty()
        || key.contains("MOE")
        || key.contains("VLLM")
        || key.contains("MARLIN")
        || key.contains("PAGED")
        || key.contains("GRAPH")
        || key.contains("SCHED")
        || key.contains("BATCH")
        || key.contains("ATTN")
        || key.contains("FLASH")
        || key.contains("CUDA")
        || key.contains("TRITON")
        || key.contains("GREEDY")
        || key.contains("FA")
    {
        effects.push(RuntimeConfigEffect::Performance);
    }

    effects.sort();
    effects.dedup();
    effects
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_boolean_values() {
        assert_eq!(parse_bool_env_value("1").unwrap(), true);
        assert_eq!(parse_bool_env_value("off").unwrap(), false);
        assert!(parse_bool_env_value("maybe").is_err());
    }

    #[test]
    fn parses_integer_values() {
        assert_eq!(parse_usize_env_value("4096").unwrap(), 4096);
        assert!(parse_usize_env_value("-1").is_err());
        assert!(parse_usize_env_value("many").is_err());
    }

    #[test]
    fn parses_path_values() {
        assert_eq!(
            parse_path_env_value("/tmp/model").unwrap(),
            PathBuf::from("/tmp/model")
        );
        assert!(parse_path_env_value("   ").is_err());
    }

    #[test]
    fn parses_tri_state_values() {
        assert_eq!(
            parse_tri_state_env_value(None).unwrap(),
            EnvTriState::Default
        );
        assert_eq!(
            parse_tri_state_env_value(Some("0")).unwrap(),
            EnvTriState::ForcedOff
        );
        assert_eq!(
            parse_tri_state_env_value(Some("on")).unwrap(),
            EnvTriState::ForcedOn
        );
        assert!(parse_tri_state_env_value(Some("auto")).is_err());
    }

    #[test]
    fn snapshot_is_sorted_and_classified() {
        let snapshot = RuntimeConfigSnapshot::from_env_vars([
            ("OTHER_ENV", "ignored"),
            ("FERRUM_FA2_NATIVE_ARTIFACT", "/tmp/libferrum_native_fa2.a"),
            ("FERRUM_PREFIX_CACHE", "1"),
            ("FERRUM_MOE_GRAPH", "1"),
        ]);
        let keys: Vec<_> = snapshot
            .entries
            .iter()
            .map(|entry| entry.key.as_str())
            .collect();
        assert_eq!(
            keys,
            vec![
                "FERRUM_FA2_NATIVE_ARTIFACT",
                "FERRUM_MOE_GRAPH",
                "FERRUM_PREFIX_CACHE"
            ]
        );
        assert_eq!(snapshot.entries[0].source, RuntimeConfigSource::Env);
        assert!(snapshot.entries[0]
            .affects
            .contains(&RuntimeConfigEffect::Correctness));
        assert!(snapshot.entries[0]
            .affects
            .contains(&RuntimeConfigEffect::Performance));
        assert!(snapshot.entries[1]
            .affects
            .contains(&RuntimeConfigEffect::Performance));
        assert!(snapshot.entries[2]
            .affects
            .contains(&RuntimeConfigEffect::Correctness));
    }

    #[test]
    fn upsert_preserves_non_env_source_and_stable_order() {
        let mut snapshot = RuntimeConfigSnapshot::from_env_vars([
            ("FERRUM_KV_DTYPE", "fp16"),
            ("FERRUM_MOE_GRAPH", "1"),
        ]);
        snapshot.upsert("FERRUM_KV_DTYPE", "int8", RuntimeConfigSource::Cli);
        snapshot.upsert(
            "FERRUM_PROFILE_JSONL",
            "/tmp/profile.jsonl",
            RuntimeConfigSource::Cli,
        );

        let keys: Vec<_> = snapshot
            .entries
            .iter()
            .map(|entry| entry.key.as_str())
            .collect();
        assert_eq!(
            keys,
            [
                "FERRUM_KV_DTYPE",
                "FERRUM_MOE_GRAPH",
                "FERRUM_PROFILE_JSONL"
            ]
        );
        let kv = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_KV_DTYPE")
            .unwrap();
        assert_eq!(kv.effective_value, "int8");
        assert_eq!(kv.source, RuntimeConfigSource::Cli);
        assert!(kv.affects.contains(&RuntimeConfigEffect::Correctness));

        let profile = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_PROFILE_JSONL")
            .unwrap();
        assert_eq!(profile.source, RuntimeConfigSource::Cli);
        assert!(profile.affects.contains(&RuntimeConfigEffect::Diagnostics));
    }
}
