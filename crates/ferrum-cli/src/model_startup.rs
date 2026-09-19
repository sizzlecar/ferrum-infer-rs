//! Curated model startup defaults shared by `run` and `serve`.

use crate::source_resolver::recipes::{self, StartupProfile};
use ferrum_types::{Device, RuntimeConfigEntry, RuntimeConfigSnapshot, RuntimeConfigSource};

/// Resolve before legacy autosizing so its environment bridge cannot turn an
/// automatically selected value into an apparent user override.
#[derive(Default)]
pub(crate) struct ModelStartupDefaults {
    entries: Vec<RuntimeConfigEntry>,
}

impl ModelStartupDefaults {
    pub(crate) fn resolve(model: &str, device: &Device, requested: &RuntimeConfigSnapshot) -> Self {
        let Some(recipe) = recipes::find(model) else {
            return Self::default();
        };
        let applies = match (recipe.startup_profile, device) {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            (StartupProfile::Bonsai2Metal, Device::Metal) => true,
            _ => false,
        };
        if applies {
            Self::from_profile(recipe.startup_profile, requested)
        } else {
            Self::default()
        }
    }

    fn from_profile(profile: StartupProfile, requested: &RuntimeConfigSnapshot) -> Self {
        let defaults: &[(&str, &str)] = match profile {
            StartupProfile::Bonsai2Metal => &[
                ("FERRUM_MAX_MODEL_LEN", "8192"),
                ("FERRUM_PAGED_MAX_SEQS", "1"),
                ("FERRUM_MAX_BATCHED_TOKENS", "128"),
                ("FERRUM_RUNTIME_MEMORY_BUDGET_BYTES", "10737418240"),
            ],
        };
        Self {
            entries: defaults
                .iter()
                .map(|(key, value)| {
                    requested
                        .entries
                        .iter()
                        .find(|entry| {
                            entry.key == *key
                                && !matches!(
                                    entry.source,
                                    RuntimeConfigSource::Default
                                        | RuntimeConfigSource::MemoryProfile
                                )
                        })
                        .cloned()
                        .unwrap_or_else(|| {
                            RuntimeConfigEntry::new(*key, *value, RuntimeConfigSource::Default)
                        })
                })
                .collect(),
        }
    }

    /// Apply only the recipe's resource settings. Numerical policy, KV dtype,
    /// thinking, and source metadata retain their normal product semantics.
    pub(crate) fn apply(&self, effective: &mut RuntimeConfigSnapshot) {
        for entry in &self.entries {
            effective.upsert_entry(entry.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonsai_defaults_replace_autosizing_without_changing_model_behavior() {
        let requested = RuntimeConfigSnapshot::from_entries([
            RuntimeConfigEntry::new("FERRUM_PAGED_MAX_SEQS", "1", RuntimeConfigSource::Default),
            RuntimeConfigEntry::new("FERRUM_KV_DTYPE", "int8", RuntimeConfigSource::Cli),
        ]);
        let defaults = ModelStartupDefaults::from_profile(StartupProfile::Bonsai2Metal, &requested);
        let mut effective = requested;
        effective.upsert(
            "FERRUM_MAX_BATCHED_TOKENS",
            "2048",
            RuntimeConfigSource::Env,
        );
        defaults.apply(&mut effective);
        let mut engine = ferrum_types::EngineConfig::default();
        engine.apply_runtime_config_snapshot(&effective).unwrap();

        assert_eq!(engine.runtime.max_model_len, Some(8192));
        assert_eq!(engine.runtime.kv_capacity, None);
        assert_eq!(engine.scheduler.max_running_requests, 1);
        assert_eq!(engine.batching.max_num_batched_tokens, 128);
        assert_eq!(
            engine.memory.usable_capacity_bytes,
            Some(10 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            engine.numerical_execution,
            ferrum_types::NumericalExecutionPolicy::Auto
        );
        assert_eq!(
            crate::runtime_env::runtime_snapshot_value(&effective, "FERRUM_KV_DTYPE"),
            Some("int8")
        );
    }

    #[test]
    fn bonsai_defaults_preserve_explicit_values_and_their_sources() {
        for source in [
            RuntimeConfigSource::ConfigFile,
            RuntimeConfigSource::Env,
            RuntimeConfigSource::Cli,
        ] {
            let requested = RuntimeConfigSnapshot::from_entries([
                RuntimeConfigEntry::new("FERRUM_MAX_MODEL_LEN", "4096", source),
                RuntimeConfigEntry::new("FERRUM_PAGED_MAX_SEQS", "2", source),
                RuntimeConfigEntry::new("FERRUM_MAX_BATCHED_TOKENS", "256", source),
                RuntimeConfigEntry::new(
                    "FERRUM_RUNTIME_MEMORY_BUDGET_BYTES",
                    "12884901888",
                    source,
                ),
            ]);
            let defaults =
                ModelStartupDefaults::from_profile(StartupProfile::Bonsai2Metal, &requested);
            let mut effective = RuntimeConfigSnapshot::default();
            defaults.apply(&mut effective);
            assert_eq!(effective, requested);
        }
    }

    #[test]
    fn bonsai_metal_defaults_do_not_change_other_backends() {
        let requested = RuntimeConfigSnapshot::default();
        for device in [Device::CPU, Device::CUDA(0)] {
            let mut effective = requested.clone();
            ModelStartupDefaults::resolve("bonsai2:27b", &device, &requested).apply(&mut effective);
            assert_eq!(effective, requested);
        }
    }
}
