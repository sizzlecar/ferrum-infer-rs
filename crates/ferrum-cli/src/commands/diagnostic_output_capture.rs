use std::{fs::File, io::Read, path::PathBuf};

use clap::Args;
use ferrum_types::{FerrumError, Result, RetainedOutputCaptureConfig};

const MAXIMUM_SPEC_BYTES: u64 = 8 * 1024;

#[derive(Args, Clone, Debug, Default)]
pub struct DiagnosticOutputCaptureArgs {
    /// Capture one retained output row matching an exact consumed token history.
    /// Diagnostic only: the extra readback is excluded from performance claims.
    #[arg(long = "diagnostic-output-capture-spec", value_name = "JSON")]
    pub spec: Option<PathBuf>,
}

impl DiagnosticOutputCaptureArgs {
    pub fn to_config(&self) -> Result<Option<RetainedOutputCaptureConfig>> {
        let Some(path) = &self.spec else {
            return Ok(None);
        };
        let mut bytes = Vec::new();
        File::open(path)
            .and_then(|file| file.take(MAXIMUM_SPEC_BYTES + 1).read_to_end(&mut bytes))
            .map_err(|error| {
                FerrumError::config(format!(
                    "read output capture spec {}: {error}",
                    path.display()
                ))
            })?;
        if bytes.len() as u64 > MAXIMUM_SPEC_BYTES {
            return Err(FerrumError::config("output capture spec exceeds 8KiB"));
        }
        let config: RetainedOutputCaptureConfig =
            serde_json::from_slice(&bytes).map_err(|error| {
                FerrumError::config(format!("invalid output capture spec: {error}"))
            })?;
        config.validate().map_err(FerrumError::config)?;
        Ok(Some(config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Args, Command, FromArgMatches};

    #[test]
    fn output_capture_spec_is_shared_by_run_and_serve() {
        let run = crate::commands::run::RunCommand::augment_args(Command::new("run"))
            .try_get_matches_from([
                "run",
                "model.gguf",
                "--diagnostic-output-capture-spec",
                "row.json",
            ])
            .unwrap();
        let serve = crate::commands::serve::ServeCommand::augment_args(Command::new("serve"))
            .try_get_matches_from([
                "serve",
                "--model",
                "model.gguf",
                "--diagnostic-output-capture-spec",
                "row.json",
            ])
            .unwrap();
        assert_eq!(
            crate::commands::run::RunCommand::from_arg_matches(&run)
                .unwrap()
                .diagnostic_output_capture
                .spec,
            Some(PathBuf::from("row.json"))
        );
        assert_eq!(
            crate::commands::serve::ServeCommand::from_arg_matches(&serve)
                .unwrap()
                .diagnostic_output_capture
                .spec,
            Some(PathBuf::from("row.json"))
        );
        assert!(DiagnosticOutputCaptureArgs::default()
            .to_config()
            .unwrap()
            .is_none());
    }

    #[test]
    fn output_capture_spec_rejects_unbounded_or_non_decode_requests() {
        let mut config = RetainedOutputCaptureConfig {
            output_dir: "capture".into(),
            prompt_token_count: 18,
            prompt_token_sha256: "a".repeat(64),
            source_span: 56..57,
            consumed_history_sha256: "b".repeat(64),
            maximum_readback_bytes: 1_048_576,
        };
        assert!(config.validate().is_ok());
        config.maximum_readback_bytes += 1;
        assert!(config.validate().is_err());
        config.maximum_readback_bytes = 1;
        config.source_span = 56..58;
        assert!(config.validate().is_err());
        config.source_span = 17..18;
        assert!(config.validate().is_err());
    }
}
