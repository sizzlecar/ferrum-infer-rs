//! Offline preparation uses the same declared materializer as run and serve.
use super::image_product::ImageProductConfig;
use anyhow::{ensure, Result};
use clap::{Args, ValueEnum};
use ferrum_engine::image_generation::ImageStage;
use ferrum_runtime::HostMemoryAccount;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum PreparedImageStage {
    TextEncoder,
    Denoiser,
}
impl From<PreparedImageStage> for ImageStage {
    fn from(stage: PreparedImageStage) -> Self {
        match stage {
            PreparedImageStage::TextEncoder => Self::TextEncoder,
            PreparedImageStage::Denoiser => Self::Denoiser,
        }
    }
}

#[derive(Debug, Args)]
pub struct PrepareImageWeightsCommand {
    /// Image JSON declaring source files, FP8 precision, and Host memory limits.
    #[arg(long, value_name = "JSON")]
    pub image_config: PathBuf,
    /// Prepare one stage using its declared numerical conversion recipe.
    #[arg(long, value_enum)]
    pub stage: PreparedImageStage,
    /// New directory for the prepared shards and hash-bound manifest.
    #[arg(long, value_name = "DIRECTORY")]
    pub output: PathBuf,
}

pub fn execute(command: PrepareImageWeightsCommand) -> Result<()> {
    ensure!(
        !command.output.try_exists()?,
        "prepared output directory already exists"
    );
    let config = ImageProductConfig::load(&command.image_config)?;
    let stage = command.stage.into();
    ensure!(
        !config
            .prepared_weights
            .entries()
            .any(|(selected, _)| selected == stage),
        "export requires the original source; remove prepared_weights for the selected stage"
    );
    let profile = config.backend.prepared_fp8_profile(stage)?;
    let host = HostMemoryAccount::new(config.backend.host_limits()?);
    let assets = config.prepare_assets()?;
    let receipt = assets.export_prepared_fp8(stage, &profile, &host, &command.output)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
