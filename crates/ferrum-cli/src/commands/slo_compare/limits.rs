//! Small, strict control-plane input; artifact budgets remain in bench-core.

use super::*;
use serde::{Deserialize, Serialize};
use std::io::Read;

const MAX_CONFIG_BYTES: u64 = 16 * 1024;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LimitsConfig {
    pub(super) schema_version: u32,
    pub(super) limits: ArtifactLoadLimits,
}

pub(super) struct LoadedLimits {
    pub(super) evidence: ArtifactLoadLimits,
    pub(super) config_path: Option<PathBuf>,
}

pub(super) fn load(path: Option<&Path>) -> Result<LoadedLimits> {
    let Some(path) = path else {
        return Ok(LoadedLimits {
            evidence: ArtifactLoadLimits::default(),
            config_path: None,
        });
    };
    let error =
        |message: String| FerrumError::config(format!("SLO comparison limits-config: {message}"));
    let path = path
        .canonicalize()
        .map_err(|err| error(format!("resolve file: {err}")))?;
    // Inspect before open to avoid blocking on a directory/device/FIFO input.
    if !fs::metadata(&path)
        .map_err(|err| error(err.to_string()))?
        .is_file()
    {
        return Err(error("configuration must be a regular JSON file".into()));
    }
    let file = File::open(&path).map_err(|err| error(err.to_string()))?;
    let metadata = file.metadata().map_err(|err| error(err.to_string()))?;
    if !metadata.is_file() || metadata.len() > MAX_CONFIG_BYTES {
        return Err(error(
            "configuration must be a regular file of at most 16 KiB".into(),
        ));
    }
    let mut bytes = Vec::new();
    file.take(MAX_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|err| error(err.to_string()))?;
    if bytes.len() as u64 > MAX_CONFIG_BYTES || bytes.len() as u64 != metadata.len() {
        return Err(error(
            "configuration changed size while reading or exceeds 16 KiB".into(),
        ));
    }
    let config: LimitsConfig =
        serde_json::from_slice(&bytes).map_err(|err| error(format!("invalid JSON: {err}")))?;
    if config.schema_version != 1 {
        return Err(error("schema_version must be 1".into()));
    }
    config
        .limits
        .validate()
        .map_err(|err| error(err.to_string()))?;
    Ok(LoadedLimits {
        evidence: config.limits,
        config_path: Some(path),
    })
}
