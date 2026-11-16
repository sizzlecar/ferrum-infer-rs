use anyhow::{ensure, Result};
use ferrum_models::fast_h3::audio_vae::{FastH3AudioLatentLayout, FastH3AudioLatentUnits};
use serde::{Deserialize, Serialize};
use std::{
    io::Read,
    path::{Path, PathBuf},
};
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Input {
    File { path: PathBuf },
    DiagnosticDyadic,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    pub vae_config: PathBuf,
    pub vae_file: PathBuf,
    pub input: Input,
    pub input_layout: FastH3AudioLatentLayout,
    pub latent_units: FastH3AudioLatentUnits,
    pub requests: u64,
    pub latent_samples: u64,
    pub output_dir: PathBuf,
    pub device: usize,
    pub host_bytes: usize,
    pub device_bytes: u64,
    pub bootstrap_bytes: u64,
    pub stream_bytes: u64,
    pub cudnn_library: PathBuf,
    pub cudnn_dependencies: Vec<PathBuf>,
    pub cudnn_opaque_bytes: u64,
    pub cudnn_workspace_bytes: u64,
    pub maximum_metadata_bytes: usize,
    pub maximum_source_bytes: u64,
    pub maximum_stored_bytes: u64,
    pub maximum_plan_nodes: usize,
    pub maximum_activation_bytes: u64,
}
impl Configuration {
    pub fn load(path: &Path) -> Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(1_048_577)
            .read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 1_048_576, "configuration exceeds 1MiB");
        let c: Self = serde_json::from_slice(&bytes)?;
        ensure!(
            c.requests > 0 && c.latent_samples > 0,
            "positive requests and latent samples required"
        );
        ensure!(
            [c.host_bytes, c.maximum_metadata_bytes, c.maximum_plan_nodes]
                .into_iter()
                .all(|n| n > 0 && n < isize::MAX as usize),
            "invalid host or metadata capacity"
        );
        ensure!(
            [
                c.device_bytes,
                c.bootstrap_bytes,
                c.stream_bytes,
                c.cudnn_opaque_bytes,
                c.maximum_source_bytes,
                c.maximum_stored_bytes,
                c.maximum_activation_bytes
            ]
            .into_iter()
            .all(|n| n > 0 && n < isize::MAX as u64),
            "invalid finite capacity"
        );
        Ok(c)
    }
}
