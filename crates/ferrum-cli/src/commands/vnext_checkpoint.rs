use std::path::PathBuf;

use clap::Args;
use ferrum_types::{FerrumError, Result, VNextCheckpointCaptureConfig};

#[derive(Args, Clone, Debug, Default)]
pub struct VNextCheckpointArgs {
    /// Empty directory for typed vNext activation evidence.
    #[arg(long = "vnext-checkpoint-dir", value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    /// Semantic ProgramValueId retained and captured after prefill. Repeat for
    /// multiple layer or logits checkpoints.
    #[arg(
        long = "vnext-checkpoint-value",
        value_name = "VALUE_ID",
        action = clap::ArgAction::Append
    )]
    pub value_ids: Vec<String>,

    /// Maximum number of real prefill waves to capture. Startup warmup is
    /// excluded. Defaults to one when capture is configured.
    #[arg(long = "vnext-checkpoint-prefill-waves", value_name = "N")]
    pub maximum_prefill_waves: Option<usize>,
}

impl VNextCheckpointArgs {
    pub fn to_config(&self) -> Result<Option<VNextCheckpointCaptureConfig>> {
        let configured = self.output_dir.is_some()
            || !self.value_ids.is_empty()
            || self.maximum_prefill_waves.is_some();
        if !configured {
            return Ok(None);
        }
        let output_dir = self.output_dir.clone().ok_or_else(|| {
            FerrumError::config(
                "--vnext-checkpoint-dir is required when checkpoint capture is configured",
            )
        })?;
        if self.value_ids.is_empty() {
            return Err(FerrumError::config(
                "at least one --vnext-checkpoint-value is required",
            ));
        }
        Ok(Some(VNextCheckpointCaptureConfig {
            output_dir,
            value_ids: self.value_ids.clone(),
            maximum_prefill_waves: self.maximum_prefill_waves.unwrap_or(1),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_flags_preserve_the_product_default() {
        assert!(VNextCheckpointArgs::default()
            .to_config()
            .unwrap()
            .is_none());
    }

    #[test]
    fn capture_requires_a_directory_and_semantic_value() {
        let missing_directory = VNextCheckpointArgs {
            value_ids: vec!["value.output.logits".to_owned()],
            ..VNextCheckpointArgs::default()
        };
        assert!(missing_directory.to_config().is_err());

        let missing_value = VNextCheckpointArgs {
            output_dir: Some(PathBuf::from("capture")),
            ..VNextCheckpointArgs::default()
        };
        assert!(missing_value.to_config().is_err());
    }

    #[test]
    fn capture_defaults_to_one_real_prefill_wave() {
        let config = VNextCheckpointArgs {
            output_dir: Some(PathBuf::from("capture")),
            value_ids: vec!["value.output.logits".to_owned()],
            maximum_prefill_waves: None,
        }
        .to_config()
        .unwrap()
        .unwrap();
        assert_eq!(config.maximum_prefill_waves, 1);
    }
}
