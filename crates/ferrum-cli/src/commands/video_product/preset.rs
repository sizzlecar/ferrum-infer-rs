//! A named configuration selects the same expanded example distributed for tuning.
use anyhow::{ensure, Result};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
enum Preset {
    #[serde(rename = "rtx5090_ref2va_v1")]
    Rtx5090Ref2vaV1,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PresetConfig {
    preset: Preset,
    pub model_directory: PathBuf,
    pub output_directory: PathBuf,
}

impl PresetConfig {
    /// Choose the representation without materializing a JSON object. The
    /// final typed deserializer sees the original bytes, including duplicate
    /// fields and invalid nested configuration values.
    pub(super) fn from_document(data: &[u8]) -> Result<Option<Self>> {
        #[derive(Deserialize)]
        struct Format {
            preset: Option<Preset>,
        }
        if serde_json::from_slice::<Format>(data)?.preset.is_none() {
            return Ok(None);
        }
        let short: Self = serde_json::from_slice(data)?;
        ensure!(
            !short.model_directory.as_os_str().is_empty(),
            "model_directory must not be empty"
        );
        ensure!(
            short.output_directory.is_absolute(),
            "output_directory must be an absolute path"
        );
        Ok(Some(short))
    }

    pub(super) fn template(&self) -> &'static [u8] {
        match self.preset {
            Preset::Rtx5090Ref2vaV1 => {
                include_bytes!("../../../examples/config/fast-h3-video-cuda.json")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compact_config_selects_an_expanded_document_and_preserves_installation_paths() {
        let output = tempfile::tempdir().unwrap();
        let json = json!({
            "preset": "rtx5090_ref2va_v1",
            "model_directory": "models/Ref2VA",
            "output_directory": output.path(),
        });
        let short = PresetConfig::from_document(&serde_json::to_vec(&json).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(short.model_directory, PathBuf::from("models/Ref2VA"));
        assert_eq!(short.output_directory, output.path());
        assert!(PresetConfig::from_document(short.template())
            .unwrap()
            .is_none());
    }

    #[test]
    fn compact_config_rejects_ambiguity_and_misspelled_or_invalid_fields() {
        for invalid in [
            r#"{"preset":"unknown","model_directory":"a","output_directory":"/b"}"#,
            r#"{"preset":"rtx5090_ref2va_v1","preset":"rtx5090_ref2va_v1","model_directory":"a","output_directory":"/b"}"#,
            r#"{"preset":"rtx5090_ref2va_v1","model_directory":"a","model_directory":"b","output_directory":"/b"}"#,
            r#"{"preset":"rtx5090_ref2va_v1","model_directory":"a","output_directory":"/b","cuda":{}}"#,
            r#"{"preset":"rtx5090_ref2va_v1","model_diretory":"a","output_directory":"/b"}"#,
            r#"{"preset":"rtx5090_ref2va_v1","model_directory":"","output_directory":"/b"}"#,
            r#"{"preset":"rtx5090_ref2va_v1","model_directory":"a","output_directory":"relative"}"#,
        ] {
            assert!(
                PresetConfig::from_document(invalid.as_bytes()).is_err(),
                "{invalid}"
            );
        }
    }
}
