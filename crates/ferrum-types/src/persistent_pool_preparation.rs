//! Explicit startup residency preparation, independent of scheduling policy.

use serde::{Deserialize, Serialize};

pub const PERSISTENT_POOL_PREPARATION_CONFIG_KEY: &str = "FERRUM_PERSISTENT_POOL_PREPARATION";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PersistentPoolPreparation {
    #[default]
    DemandDriven,
    /// Allocate the configured full persistent envelope before readiness.
    /// This is initial residency, not a hard floor; original maintenance may reclaim it.
    ConfiguredEnvelope,
}

impl PersistentPoolPreparation {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::DemandDriven => "demand-driven",
            Self::ConfiguredEnvelope => "configured-envelope",
        }
    }

    pub fn parse_runtime_value(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "demand-driven" => Ok(Self::DemandDriven),
            "configured-envelope" => Ok(Self::ConfiguredEnvelope),
            _ => Err(format!(
                "expected demand-driven or configured-envelope; got {value:?}"
            )),
        }
    }

    pub fn parse_config_entry(entry: &crate::RuntimeConfigEntry) -> Result<Self, String> {
        if entry.source == crate::RuntimeConfigSource::Env {
            return Err(
                "select persistent pool preparation through CLI or config, not environment".into(),
            );
        }
        Self::parse_runtime_value(&entry.effective_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn persistent_pool_preparation_defaults_and_explicit_provenance() {
        let mut engine = crate::EngineConfig::default();
        assert_eq!(
            engine.memory.persistent_pool_preparation,
            PersistentPoolPreparation::DemandDriven
        );
        let mut old = serde_json::to_value(&engine.memory).unwrap();
        old.as_object_mut()
            .unwrap()
            .remove("persistent_pool_preparation");
        assert_eq!(
            serde_json::from_value::<crate::MemoryConfig>(old)
                .unwrap()
                .persistent_pool_preparation,
            PersistentPoolPreparation::DemandDriven
        );
        for source in [
            crate::RuntimeConfigSource::Cli,
            crate::RuntimeConfigSource::ConfigFile,
        ] {
            let mut snapshot = crate::RuntimeConfigSnapshot::default();
            snapshot.upsert(
                PERSISTENT_POOL_PREPARATION_CONFIG_KEY,
                "configured-envelope",
                source,
            );
            engine.apply_runtime_config_snapshot(&snapshot).unwrap();
            assert_eq!(
                engine.memory.persistent_pool_preparation,
                PersistentPoolPreparation::ConfiguredEnvelope
            );
        }
        let entry = crate::RuntimeConfigEntry::new(
            PERSISTENT_POOL_PREPARATION_CONFIG_KEY,
            "configured-envelope",
            crate::RuntimeConfigSource::Env,
        );
        assert!(PersistentPoolPreparation::parse_config_entry(&entry).is_err());
    }
}
