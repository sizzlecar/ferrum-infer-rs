use anyhow::{ensure, Context, Result};
use ferrum_models::fast_h3::video_vae::FastH3ImageSpatialPolicy;
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
    pub vae_dir: PathBuf,
    pub index_file: String,
    pub input: Input,
    pub input_shape: [u64; 4],
    #[serde(default)]
    pub spatial: FastH3ImageSpatialPolicy,
    pub output_file: PathBuf,
    /// Optional normalized F32 NC1HW posterior, using the declared fresh-seed
    /// CPU scalar profile and the original F16 rounding boundary.
    #[serde(default)]
    pub posterior_file: Option<PathBuf>,
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
    pub maximum_stored_bytes: u64,
    pub maximum_plan_nodes: usize,
}
impl Configuration {
    pub fn load(path: &Path) -> Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(1_048_577)
            .read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 1_048_576, "configuration exceeds 1 MiB");
        let c: Self = serde_json::from_slice(&bytes)?;
        ensure!(
            !c.input_shape.contains(&0) && c.input_shape[1] == 3,
            "input requires positive F32 NCHW RGB"
        );
        ensure!(
            [c.host_bytes, c.maximum_metadata_bytes, c.maximum_plan_nodes]
                .into_iter()
                .all(|x| x > 0 && x < isize::MAX as usize)
                && [
                    c.device_bytes,
                    c.bootstrap_bytes,
                    c.stream_bytes,
                    c.cudnn_opaque_bytes,
                    c.maximum_stored_bytes
                ]
                .into_iter()
                .all(|x| x > 0 && x < isize::MAX as u64),
            "invalid capacity"
        );
        ensure!(
            c.input_bytes()? <= c.host_bytes,
            "image input exceeds Host capacity"
        );
        Ok(c)
    }
    pub fn input_bytes(&self) -> Result<usize> {
        let bytes = self
            .input_shape
            .into_iter()
            .try_fold(4u64, |n, v| n.checked_mul(v))
            .context("input extent overflow")?;
        usize::try_from(bytes).context("input exceeds address space")
    }
}
