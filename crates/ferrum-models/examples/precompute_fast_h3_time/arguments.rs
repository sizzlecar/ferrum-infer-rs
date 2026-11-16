use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Device/Host numbers are admission limits, not measured peak usage.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Arguments {
    pub model_dir: PathBuf,
    pub output_dir: PathBuf,
    pub first_block: usize,
    pub block_count: usize,
    pub iterations: usize,
    pub device: u32,
    pub device_bytes: u64,
    pub host_bytes: usize,
    pub bootstrap_bytes: u64,
    pub stream_bytes: u64,
    pub staging_bytes: u64,
    pub max_metadata_bytes: usize,
    pub max_source_bytes: usize,
    pub max_tensors: usize,
    pub max_pdd_steps: usize,
}
impl Default for Arguments {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            output_dir: PathBuf::new(),
            first_block: 0,
            block_count: 1,
            iterations: 2,
            device: 0,
            device_bytes: 2 << 30,
            host_bytes: 256 << 20,
            bootstrap_bytes: 256 << 20,
            stream_bytes: 16 << 20,
            staging_bytes: 16 << 20,
            max_metadata_bytes: 1 << 20,
            max_source_bytes: 64 << 30,
            max_tensors: 4096,
            max_pdd_steps: 128,
        }
    }
}
impl Arguments {
    pub fn read(path: &Path) -> Result<Self> {
        let args: Self = serde_json::from_slice(&super::package::bounded_read(path, 65536)?)?;
        args.validate()?;
        Ok(args)
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.model_dir.as_os_str().is_empty() && !self.output_dir.as_os_str().is_empty(),
            "model_dir and output_dir are required"
        );
        ensure!(
            self.block_count > 0 && self.first_block.checked_add(self.block_count).is_some(),
            "invalid block range"
        );
        ensure!(self.iterations > 0, "iterations must be positive");
        for (name, count) in [
            ("device_bytes", self.device_bytes),
            ("host_bytes", self.host_bytes as u64),
            ("bootstrap_bytes", self.bootstrap_bytes),
            ("stream_bytes", self.stream_bytes),
            ("staging_bytes", self.staging_bytes),
            ("max_metadata_bytes", self.max_metadata_bytes as u64),
            ("max_source_bytes", self.max_source_bytes as u64),
            ("max_tensors", self.max_tensors as u64),
            ("max_pdd_steps", self.max_pdd_steps as u64),
        ] {
            ensure!(count > 0 && count < isize::MAX as u64, "invalid {name}");
        }
        ensure!(
            self.staging_bytes <= self.host_bytes as u64,
            "staging exceeds Host budget"
        );
        Ok(())
    }
}

#[test]
fn explicit_paths_and_real_capacity_are_required() {
    let mut args = Arguments::default();
    assert!(args.validate().is_err());
    args.model_dir = "/model".into();
    args.output_dir = "/output".into();
    assert!(args.validate().is_ok());
    args.staging_bytes = args.host_bytes as u64 + 1;
    assert!(args.validate().is_err());
    assert!(serde_json::from_str::<Arguments>(r#"{"allow_precision_fallback":true}"#).is_err());
}
