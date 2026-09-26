//! Fixed-history, full-vocabulary model quality capture. Not a timing run.
use crate::config::CliConfig;
use clap::{Args, ValueEnum};
use ferrum_models::{VNextTeacherExecutionSpec, VNextTeacherMode, VNextTeacherOwner};
use ferrum_types::{FerrumError, Result};
use serde::Deserialize;
use std::path::PathBuf;

mod artifacts;
#[cfg(any(
    feature = "cuda",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
mod startup;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum TeacherModeArg {
    Serial,
    Batched,
}

impl TeacherModeArg {
    fn mode(self) -> VNextTeacherMode {
        match self {
            Self::Serial => VNextTeacherMode::Serial,
            Self::Batched => VNextTeacherMode::Batched,
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Batched => "batched",
        }
    }
}

#[derive(Args)]
#[command(
    after_help = "Consumes exact prompt and teacher token IDs. Both arms use real prefill/state and unmodified FullLogits before sampling. Serial uses one physical decode wave per owner; batched requires all owners in every decode wave. This diagnostic deliberately reads full vocabulary logits and provides no serving-performance evidence."
)]
pub struct VNextTeacherCommand {
    /// Registered model source directory or GGUF artifact.
    pub model: String,
    #[command(flatten)]
    pub product_sources: crate::source_resolver::ProductSourceArgs,
    /// Production accelerator used for the capture. CPU capture is unsupported.
    #[arg(long, default_value = "auto")]
    pub backend: String,
    /// Version 1 JSON: owners with owner_id, prompt_token_ids and teacher_token_ids.
    #[arg(long)]
    pub history_file: PathBuf,
    #[arg(long, value_enum)]
    pub mode: TeacherModeArg,
    #[arg(long)]
    pub output_dir: PathBuf,
    /// Exact declared numerical profile; auto selection is not accepted.
    #[arg(long)]
    pub numerical_profile: ferrum_types::NumericalProfileId,
    /// Model context ceiling. Each request admits its complete prompt plus teacher output budget.
    #[arg(long)]
    pub max_model_len: std::num::NonZeroUsize,
    #[arg(long, default_value = "256")]
    pub prefill_chunk_tokens: std::num::NonZeroUsize,
    /// Fixed production runtime budget. Both modes retain all declared owners.
    #[arg(long)]
    pub runtime_memory_budget_bytes: std::num::NonZeroUsize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoryFile {
    schema_version: u32,
    owners: Vec<VNextTeacherOwner>,
}

pub async fn execute(cmd: VNextTeacherCommand, config: CliConfig) -> Result<()> {
    let bytes = std::fs::read(&cmd.history_file)
        .map_err(|error| FerrumError::io(format!("read teacher history: {error}")))?;
    let history: HistoryFile = serde_json::from_slice(&bytes)
        .map_err(|error| FerrumError::config(format!("parse teacher history: {error}")))?;
    if history.schema_version != 1 || cmd.numerical_profile.as_str() == "auto" {
        return Err(FerrumError::config(
            "teacher capture requires history schema 1 and an exact numerical profile",
        ));
    }
    let spec = VNextTeacherExecutionSpec {
        owners: history.owners,
        mode: cmd.mode.mode(),
        maximum_sequence_tokens: cmd.max_model_len.get(),
        prefill_chunk_tokens: cmd.prefill_chunk_tokens.get(),
    };
    spec.validate(usize::MAX, cmd.max_model_len.get())?;
    let mut artifacts = artifacts::Artifacts::create(&cmd, &spec)?;
    #[cfg(any(
        feature = "cuda",
        all(feature = "metal", any(target_os = "macos", target_os = "ios"))
    ))]
    let result = startup::collect(&cmd, &config, &spec, &mut artifacts).await;
    #[cfg(not(any(
        feature = "cuda",
        all(feature = "metal", any(target_os = "macos", target_os = "ios"))
    )))]
    let result = {
        let _ = config;
        Err(FerrumError::unsupported(
            "real-history teacher capture requires a compiled Metal or CUDA backend",
        ))
    };
    artifacts.finish(result.as_ref().err())?;
    result?;
    println!(
        "Real-history full-vocabulary capture saved to {}",
        cmd.output_dir.display()
    );
    Ok(())
}
