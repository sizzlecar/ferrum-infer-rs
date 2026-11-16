use anyhow::{ensure, Context, Result};
pub use ferrum_models::fast_h3::video_vae::FastH3VideoLatentUnits as LatentUnits;
use serde::{Deserialize, Serialize};
use std::{
    io::Read,
    path::{Path, PathBuf},
};
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LatentInput {
    File {
        path: PathBuf,
    },
    /// Allocates explicit all-zero diagnostic latents on the same Host ledger.
    /// This proves decoder execution, not prompt-conditioned output quality.
    DiagnosticZeros,
    /// Deterministic dyadic cycle ±1/64 .. ±31/64, with no zero samples.
    /// This is a decoder diagnostic, not a sampled diffusion trajectory.
    DiagnosticDyadic,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    pub vae_config: PathBuf,
    pub packed_decoder: PathBuf,
    pub input: LatentInput,
    pub latent_shape: [usize; 5],
    #[serde(default)]
    pub latent_layout: ferrum_models::fast_h3::video_vae::FastH3VideoLatentLayout,
    pub latent_units: LatentUnits,
    #[serde(default)]
    pub spatial: ferrum_models::fast_h3::video_vae::FastH3VideoSpatialPolicy,
    pub output_dir: PathBuf,
    pub frame_rate: u32,
    pub device: usize,
    pub host_bytes: usize,
    pub device_bytes: u64,
    pub bootstrap_bytes: u64,
    pub stream_bytes: u64,
    pub nvfp4_opaque_bytes: u64,
    pub cudnn_library: PathBuf,
    pub cudnn_dependencies: Vec<PathBuf>,
    pub cudnn_opaque_bytes: u64,
    pub cudnn_workspace_bytes: u64,
    pub sdpa_cache_bytes: u64,
    pub maximum_tensors: usize,
    pub maximum_source_bytes: usize,
    pub maximum_metadata_bytes: usize,
    pub maximum_output_frames: usize,
}
impl Configuration {
    pub fn load(path: &Path) -> Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(1_048_577)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() <= 1_048_576,
            "deployment configuration exceeds 1 MiB"
        );
        let c: Self = serde_json::from_slice(&bytes)?;
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.latent_shape.contains(&0) && self.frame_rate > 0,
            "invalid latent shape or frame rate"
        );
        ensure!(
            [
                self.host_bytes,
                self.maximum_tensors,
                self.maximum_source_bytes,
                self.maximum_metadata_bytes,
                self.maximum_output_frames
            ]
            .into_iter()
            .all(|v| v > 0 && v < isize::MAX as usize),
            "invalid Host/source capacity"
        );
        ensure!(
            [
                self.device_bytes,
                self.bootstrap_bytes,
                self.stream_bytes,
                self.nvfp4_opaque_bytes,
                self.cudnn_opaque_bytes,
                self.sdpa_cache_bytes
            ]
            .into_iter()
            .all(|v| v > 0 && v < isize::MAX as u64)
                && self.cudnn_workspace_bytes < isize::MAX as u64,
            "invalid device/native capacity"
        );
        ensure!(
            self.cudnn_library.is_absolute()
                && self.cudnn_dependencies.iter().all(|p| p.is_absolute()),
            "cuDNN paths must be absolute"
        );
        ensure!(
            self.input_bytes()? <= self.host_bytes,
            "input latent exceeds Host budget"
        );
        Ok(())
    }
    pub fn input_bytes(&self) -> Result<usize> {
        self.latent_shape
            .into_iter()
            .try_fold(4usize, |n, v| n.checked_mul(v))
            .filter(|v| *v < isize::MAX as usize)
            .context("latent byte count overflow")
    }
}
