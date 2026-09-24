//! Shared file loading and provenance for run/serve SLO policy.

use ferrum_types::{
    ExecutionResourceAuthority, FerrumError, Result, RuntimeConfigEntry, RuntimeConfigSnapshot,
    RuntimeConfigSource, SloConfig, SloMode, SLO_CONFIG_DIGEST_RUNTIME_KEY,
    SLO_CONFIG_PATH_RUNTIME_KEY, SLO_CONFIG_RUNTIME_KEY,
};
use sha2::{Digest, Sha256};
use std::path::Path;

pub(super) const HELP: &str = "Load a TOML or JSON SLO policy for this model. Defaults to Off; Observe measures the existing policy, while Enforce requires supported bounded execution and a validated cost profile. Internal token-commit budgets and optional client-visible targets are separate. Config: runtime.slo_config. No environment override.";

#[derive(Debug)]
pub(super) struct LoadedSloConfig {
    pub config: SloConfig,
    entries: Vec<RuntimeConfigEntry>,
}

impl LoadedSloConfig {
    pub fn apply_to_snapshot(&self, snapshot: &mut RuntimeConfigSnapshot) {
        for entry in &self.entries {
            snapshot.upsert_entry(entry.clone());
        }
    }

    pub fn runtime_entries(&self) -> &[RuntimeConfigEntry] {
        &self.entries
    }

    pub fn validate_real_execution(&self, synthetic: bool) -> Result<()> {
        if synthetic && self.config.mode != SloMode::Off {
            return Err(FerrumError::unsupported(
                "enabled SLO policy requires real language-model execution, not synthetic observability",
            ));
        }
        Ok(())
    }

    pub fn validate_authority(&self, authority: ExecutionResourceAuthority) -> Result<()> {
        if self.config.mode == SloMode::Enforce
            && authority != ExecutionResourceAuthority::PlanRuntime
        {
            return Err(FerrumError::unsupported(
                "SLO Enforce requires a native plan runtime with bounded execution; legacy execution supports Observe only",
            ));
        }
        Ok(())
    }
}

pub(super) async fn load(
    cli_path: Option<&Path>,
    configured_path: Option<&Path>,
) -> Result<Option<LoadedSloConfig>> {
    let Some((path, source)) = cli_path
        .map(|path| (path, RuntimeConfigSource::Cli))
        .or_else(|| configured_path.map(|path| (path, RuntimeConfigSource::ConfigFile)))
    else {
        return Ok(None);
    };
    if path.as_os_str().is_empty() {
        return Err(FerrumError::config("SLO config path must not be empty"));
    }
    let absolute = tokio::fs::canonicalize(path).await.map_err(|error| {
        FerrumError::config(format!("resolve SLO config {}: {error}", path.display()))
    })?;
    let bytes = tokio::fs::read(&absolute).await.map_err(|error| {
        FerrumError::config(format!("read SLO config {}: {error}", absolute.display()))
    })?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| FerrumError::config(format!("SLO config must be UTF-8: {error}")))?;
    let mut config: SloConfig = match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) if extension.eq_ignore_ascii_case("json") => serde_json::from_str(text)
            .map_err(|error| {
                FerrumError::config(format!("parse SLO JSON {}: {error}", path.display()))
            })?,
        Some(extension) if extension.eq_ignore_ascii_case("toml") => {
            toml::from_str(text).map_err(|error| {
                FerrumError::config(format!("parse SLO TOML {}: {error}", path.display()))
            })?
        }
        _ => {
            return Err(FerrumError::config(
                "SLO config file must use a .toml or .json extension",
            ))
        }
    };
    config.validate().map_err(FerrumError::config)?;
    if let Some(profile) = config.cost_profile.as_mut() {
        if profile.is_relative() {
            // The policy file owns this reference; resolving it once also makes
            // the effective configuration independent of later working dirs.
            *profile = absolute.parent().unwrap_or(Path::new("/")).join(&*profile);
        }
    }
    if let Some(export) = config.cost_observation.profile_export.as_mut() {
        for path in [&mut export.path, &mut export.observations_path] {
            if path.is_relative() {
                *path = absolute.parent().unwrap_or(Path::new("/")).join(&*path);
            }
        }
    }
    if let Some(reference) = config.prefill_reference.as_mut() {
        if reference.artifact_path.is_relative() {
            reference.artifact_path = absolute
                .parent()
                .unwrap_or(Path::new("/"))
                .join(&reference.artifact_path);
        }
    }
    let entries = vec![
        RuntimeConfigEntry::new(
            SLO_CONFIG_RUNTIME_KEY,
            serde_json::to_string(&config).map_err(|error| {
                FerrumError::config(format!("serialize effective SLO config: {error}"))
            })?,
            source,
        ),
        RuntimeConfigEntry::new(
            SLO_CONFIG_PATH_RUNTIME_KEY,
            absolute.to_string_lossy(),
            source,
        ),
        RuntimeConfigEntry::new(
            SLO_CONFIG_DIGEST_RUNTIME_KEY,
            format!("sha256:{:x}", Sha256::digest(&bytes)),
            source,
        ),
    ];
    Ok(Some(LoadedSloConfig { config, entries }))
}

#[cfg(test)]
mod tests;
