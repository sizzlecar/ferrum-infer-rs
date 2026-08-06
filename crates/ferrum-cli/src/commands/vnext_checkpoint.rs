use std::{
    fs,
    path::{Path, PathBuf},
};

use clap::Args;
use ferrum_types::{
    FerrumError, Result, TokenId, VNextCheckpointCaptureConfig, VNextTeacherForcingConfig,
};
use serde::Deserialize;

const MAX_TEACHER_TOKEN_FILE_BYTES: usize = 64 * 1024;

#[derive(Args, Clone, Debug, Default)]
pub struct VNextCheckpointArgs {
    /// Empty directory for typed vNext activation evidence.
    #[arg(long = "vnext-checkpoint-dir", value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    /// Semantic ProgramValueId retained and captured after a selected execution
    /// wave. Repeat for multiple layer or logits checkpoints.
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

    /// Maximum number of real decode waves to capture. Startup warmup is
    /// excluded. Defaults to zero when capture is configured.
    #[arg(long = "vnext-checkpoint-decode-waves", value_name = "N")]
    pub maximum_decode_waves: Option<usize>,

    /// Capture the existing product logits/token readback without retaining an
    /// activation or changing the compiled memory plan.
    #[arg(long = "vnext-checkpoint-product-output")]
    pub capture_product_output: bool,

    /// JSON file containing a bounded canonical token history for a same-history
    /// numerical diagnostic. Supported only by one-shot `ferrum run`.
    #[arg(long = "vnext-checkpoint-teacher-token-file", value_name = "JSON")]
    pub teacher_token_file: Option<PathBuf>,
}

impl VNextCheckpointArgs {
    pub fn to_config(&self) -> Result<Option<VNextCheckpointCaptureConfig>> {
        let configured = self.output_dir.is_some()
            || !self.value_ids.is_empty()
            || self.maximum_prefill_waves.is_some()
            || self.maximum_decode_waves.is_some()
            || self.capture_product_output
            || self.teacher_token_file.is_some();
        if !configured {
            return Ok(None);
        }
        let teacher_forcing = self
            .teacher_token_file
            .as_deref()
            .map(load_teacher_forcing)
            .transpose()?;
        let output_dir = self.output_dir.clone().ok_or_else(|| {
            FerrumError::config(
                "--vnext-checkpoint-dir is required when checkpoint capture is configured",
            )
        })?;
        if self.value_ids.is_empty() && !self.capture_product_output {
            return Err(FerrumError::config(
                "at least one --vnext-checkpoint-value or --vnext-checkpoint-product-output is required",
            ));
        }
        if let Some(teacher_forcing) = &teacher_forcing {
            if !self.capture_product_output {
                return Err(FerrumError::config(
                    "--vnext-checkpoint-teacher-token-file requires --vnext-checkpoint-product-output",
                ));
            }
            if self.maximum_prefill_waves.is_some_and(|waves| waves != 1) {
                return Err(FerrumError::config(
                    "teacher-forced checkpoint capture requires exactly one final prefill wave",
                ));
            }
            let expected_decode_waves = teacher_forcing.token_count().saturating_sub(1);
            if self
                .maximum_decode_waves
                .is_some_and(|waves| waves != expected_decode_waves)
            {
                return Err(FerrumError::config(format!(
                    "teacher-forced checkpoint capture requires {expected_decode_waves} decode waves for {} tokens",
                    teacher_forcing.token_count()
                )));
            }
        }
        Ok(Some(VNextCheckpointCaptureConfig {
            output_dir,
            value_ids: self.value_ids.clone(),
            maximum_prefill_waves: self.maximum_prefill_waves.unwrap_or(1),
            maximum_decode_waves: self.maximum_decode_waves.unwrap_or_else(|| {
                teacher_forcing
                    .as_ref()
                    .map_or(0, |teacher| teacher.token_count().saturating_sub(1))
            }),
            capture_product_output: self.capture_product_output,
            teacher_forcing,
        }))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TeacherTokenFile {
    schema_version: u32,
    encoding: String,
    token_ids: Vec<u32>,
}

fn load_teacher_forcing(path: &Path) -> Result<VNextTeacherForcingConfig> {
    let bytes = fs::read(path).map_err(|error| {
        FerrumError::config(format!(
            "cannot read vNext teacher-token file {}: {error}",
            path.display()
        ))
    })?;
    if bytes.len() > MAX_TEACHER_TOKEN_FILE_BYTES {
        return Err(FerrumError::config(format!(
            "vNext teacher-token file exceeds {MAX_TEACHER_TOKEN_FILE_BYTES} bytes"
        )));
    }
    parse_teacher_forcing(&bytes)
}

fn parse_teacher_forcing(bytes: &[u8]) -> Result<VNextTeacherForcingConfig> {
    let parsed: TeacherTokenFile = serde_json::from_slice(bytes).map_err(|error| {
        FerrumError::config(format!("invalid vNext teacher-token JSON: {error}"))
    })?;
    if parsed.schema_version != 1 {
        return Err(FerrumError::config(
            "vNext teacher-token schema_version must be 1",
        ));
    }
    if parsed.encoding != "u32-le" {
        return Err(FerrumError::config(
            "vNext teacher-token encoding must be u32-le",
        ));
    }
    VNextTeacherForcingConfig::new(parsed.token_ids.into_iter().map(TokenId::new).collect())
        .map_err(FerrumError::config)
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
            maximum_decode_waves: None,
            capture_product_output: false,
            teacher_token_file: None,
        }
        .to_config()
        .unwrap()
        .unwrap();
        assert_eq!(config.maximum_prefill_waves, 1);
        assert_eq!(config.maximum_decode_waves, 0);
    }

    #[test]
    fn decode_capture_is_an_explicit_shared_product_option() {
        let config = VNextCheckpointArgs {
            output_dir: Some(PathBuf::from("capture")),
            value_ids: vec!["value.output.greedy_token".to_owned()],
            maximum_prefill_waves: Some(1),
            maximum_decode_waves: Some(64),
            capture_product_output: false,
            teacher_token_file: None,
        }
        .to_config()
        .unwrap()
        .unwrap();

        assert_eq!(config.maximum_prefill_waves, 1);
        assert_eq!(config.maximum_decode_waves, 64);
    }

    #[test]
    fn product_output_capture_does_not_require_retained_values() {
        let config = VNextCheckpointArgs {
            output_dir: Some(PathBuf::from("capture")),
            maximum_prefill_waves: Some(1),
            maximum_decode_waves: Some(64),
            capture_product_output: true,
            ..VNextCheckpointArgs::default()
        }
        .to_config()
        .unwrap()
        .unwrap();

        assert!(config.value_ids.is_empty());
        assert!(config.capture_product_output);
    }

    #[test]
    fn teacher_token_json_is_typed_and_bounded() {
        let parsed = parse_teacher_forcing(
            br#"{"schema_version":1,"encoding":"u32-le","token_ids":[11690,369]}"#,
        )
        .unwrap();
        assert_eq!(
            parsed
                .token_ids()
                .iter()
                .map(|token| token.get())
                .collect::<Vec<_>>(),
            [11690, 369]
        );

        assert!(parse_teacher_forcing(
            br#"{"schema_version":2,"encoding":"u32-le","token_ids":[1]}"#
        )
        .is_err());
        assert!(parse_teacher_forcing(
            br#"{"schema_version":1,"encoding":"json","token_ids":[1]}"#
        )
        .is_err());
        assert!(parse_teacher_forcing(
            br#"{"schema_version":1,"encoding":"u32-le","token_ids":[],"extra":true}"#
        )
        .is_err());

        let excessive = serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "encoding": "u32-le",
            "token_ids": vec![0_u32; ferrum_types::MAX_VNEXT_TEACHER_FORCED_TOKENS + 1],
        }))
        .unwrap();
        assert!(parse_teacher_forcing(&excessive).is_err());
    }

    #[test]
    fn teacher_token_file_derives_exact_wave_contract() {
        let root = std::env::temp_dir().join(format!(
            "ferrum-vnext-teacher-token-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let token_file = root.join("tokens.json");
        std::fs::write(
            &token_file,
            br#"{"schema_version":1,"encoding":"u32-le","token_ids":[7,11,13]}"#,
        )
        .unwrap();
        let args = VNextCheckpointArgs {
            output_dir: Some(root.join("capture")),
            capture_product_output: true,
            teacher_token_file: Some(token_file.clone()),
            ..VNextCheckpointArgs::default()
        };
        let config = args.to_config().unwrap().unwrap();
        assert_eq!(config.maximum_prefill_waves, 1);
        assert_eq!(config.maximum_decode_waves, 2);
        assert_eq!(config.teacher_forcing.unwrap().token_count(), 3);

        let wrong_decode_count = VNextCheckpointArgs {
            maximum_decode_waves: Some(3),
            ..args.clone()
        };
        assert!(wrong_decode_count.to_config().is_err());
        let missing_product_output = VNextCheckpointArgs {
            capture_product_output: false,
            value_ids: vec!["value.output.logits".to_owned()],
            ..args
        };
        assert!(missing_product_output.to_config().is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
