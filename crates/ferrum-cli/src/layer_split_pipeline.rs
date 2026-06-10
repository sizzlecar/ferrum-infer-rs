use ferrum_types::{FerrumError, Result, RuntimeConfigEntry, RuntimeConfigSnapshot};
use serde_json::Value;
use std::collections::HashMap;

pub const LAYER_SPLIT_PIPELINE_MODE_KEY: &str = "FERRUM_LAYER_SPLIT_PIPELINE_MODE";
pub const LAYER_SPLIT_PIPELINE_MODE_BACKEND_OPTION: &str = "layer_split_pipeline_mode";

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum LayerSplitPipelineModeArg {
    Batch,
    Overlapped,
}

impl LayerSplitPipelineModeArg {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::Overlapped => "overlapped",
        }
    }
}

pub fn push_cli_runtime_entry(
    entries: &mut Vec<RuntimeConfigEntry>,
    mode: Option<LayerSplitPipelineModeArg>,
) {
    if let Some(mode) = mode {
        entries.push(RuntimeConfigEntry::new(
            LAYER_SPLIT_PIPELINE_MODE_KEY,
            mode.as_str(),
            ferrum_types::RuntimeConfigSource::Cli,
        ));
    }
}

pub fn normalize_layer_split_pipeline_mode(value: &str) -> Result<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "batch" => Ok("batch"),
        "overlapped" => Ok("overlapped"),
        other => Err(FerrumError::config(format!(
            "{LAYER_SPLIT_PIPELINE_MODE_KEY} must be batch or overlapped, got {other:?}"
        ))),
    }
}

pub fn runtime_layer_split_pipeline_mode(
    snapshot: &RuntimeConfigSnapshot,
) -> Result<Option<&'static str>> {
    snapshot
        .entries
        .iter()
        .find(|entry| entry.key == LAYER_SPLIT_PIPELINE_MODE_KEY)
        .map(|entry| normalize_layer_split_pipeline_mode(&entry.effective_value))
        .transpose()
}

pub fn insert_backend_option_from_runtime(
    snapshot: &RuntimeConfigSnapshot,
    backend_options: &mut HashMap<String, Value>,
) -> Result<()> {
    if let Some(mode) = runtime_layer_split_pipeline_mode(snapshot)? {
        backend_options.insert(
            LAYER_SPLIT_PIPELINE_MODE_BACKEND_OPTION.to_string(),
            Value::String(mode.to_string()),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_types::{RuntimeConfigSnapshot, RuntimeConfigSource};

    #[test]
    fn cli_mode_entry_records_product_pipeline_mode() {
        let mut entries = Vec::new();
        push_cli_runtime_entry(&mut entries, Some(LayerSplitPipelineModeArg::Batch));

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, LAYER_SPLIT_PIPELINE_MODE_KEY);
        assert_eq!(entries[0].effective_value, "batch");
        assert_eq!(entries[0].source, RuntimeConfigSource::Cli);
    }

    #[test]
    fn invalid_runtime_mode_is_rejected() {
        let snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
            LAYER_SPLIT_PIPELINE_MODE_KEY,
            "serial",
            RuntimeConfigSource::ConfigFile,
        )]);

        let err = runtime_layer_split_pipeline_mode(&snapshot)
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be batch or overlapped"));
    }
}
