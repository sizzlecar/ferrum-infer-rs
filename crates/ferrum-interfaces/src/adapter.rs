//! Adapter selection at the transitional request metadata boundary.
//!
//! Support comes from the implementation that executes the adapter, independently
//! of model names or the owner of execution resources.

use std::collections::HashMap;

use ferrum_types::{FerrumError, Result};
use serde_json::Value;

/// Adapter execution provided by the selected implementation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AdapterExecutionSupport {
    #[default]
    Unsupported,
    /// The selected LoRA is applied to, and isolated within, each sequence.
    PerSequenceLora,
}

impl AdapterExecutionSupport {
    /// Reject adapter registration when inference would use base weights.
    pub fn require_lora(self) -> Result<()> {
        match self {
            Self::PerSequenceLora => Ok(()),
            Self::Unsupported => Err(FerrumError::unsupported(
                "LoRA inference is not supported by the selected execution implementation",
            )),
        }
    }

    /// Validate before admission or execution; a batch must validate all members
    /// before allowing any member to advance.
    pub fn validate_metadata<'a>(
        self,
        metadata: &'a HashMap<String, Value>,
    ) -> Result<Option<LoraAdapterSelection<'a>>> {
        let selection = parse_lora_selection(metadata)?;
        if selection.is_some() {
            self.require_lora()?;
        }
        Ok(selection)
    }
}

/// Borrowed, validated selection; parsing a base request allocates nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoraAdapterSelection<'a> {
    pub name: &'a str,
    pub path: &'a str,
}

/// Parse the existing wire metadata without interpreting malformed values as a
/// request for the base model. The public model id is optional, but if present
/// must accompany a complete adapter selection.
pub fn parse_lora_selection(
    metadata: &HashMap<String, Value>,
) -> Result<Option<LoraAdapterSelection<'_>>> {
    const NAME: &str = "ferrum_lora_adapter";
    const PATH: &str = "ferrum_lora_path";
    const MODEL_ID: &str = "ferrum_lora_model_id";

    if !metadata.contains_key(NAME)
        && !metadata.contains_key(PATH)
        && !metadata.contains_key(MODEL_ID)
    {
        return Ok(None);
    }
    let required_string = |key| {
        metadata
            .get(key)
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                FerrumError::invalid_request(format!(
                    "LoRA metadata requires a non-empty string for {key}"
                ))
            })
    };
    let name = required_string(NAME)?;
    let path = required_string(PATH)?;
    if metadata.contains_key(MODEL_ID) {
        required_string(MODEL_ID)?;
    }
    Ok(Some(LoraAdapterSelection { name, path }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn metadata(value: Value) -> HashMap<String, Value> {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn base_requests_do_not_require_adapter_support() {
        let metadata = metadata(json!({"unrelated": true}));
        assert_eq!(
            AdapterExecutionSupport::Unsupported
                .validate_metadata(&metadata)
                .unwrap(),
            None
        );
    }

    #[test]
    fn valid_selection_requires_actual_execution_support() {
        let mut metadata = metadata(json!({
            "ferrum_lora_adapter": "tenant-a",
            "ferrum_lora_path": "/adapters/tenant-a"
        }));
        for model_id in [None, Some(json!("base:tenant-a"))] {
            if let Some(model_id) = model_id {
                metadata.insert("ferrum_lora_model_id".into(), model_id);
            }
            assert!(AdapterExecutionSupport::Unsupported
                .validate_metadata(&metadata)
                .is_err());
            assert_eq!(
                AdapterExecutionSupport::PerSequenceLora
                    .validate_metadata(&metadata)
                    .unwrap(),
                Some(LoraAdapterSelection {
                    name: "tenant-a",
                    path: "/adapters/tenant-a",
                })
            );
        }
    }

    #[test]
    fn malformed_selection_never_becomes_a_base_request() {
        for value in [
            json!({"ferrum_lora_adapter": "a"}),
            json!({"ferrum_lora_path": "/a"}),
            json!({"ferrum_lora_model_id": "base:a"}),
            json!({"ferrum_lora_adapter": 7, "ferrum_lora_path": false}),
            json!({"ferrum_lora_adapter": null, "ferrum_lora_path": null}),
            json!({"ferrum_lora_adapter": [], "ferrum_lora_path": {}}),
            json!({"ferrum_lora_adapter": " ", "ferrum_lora_path": "/a"}),
            json!({"ferrum_lora_adapter": "a", "ferrum_lora_path": ""}),
            json!({"ferrum_lora_adapter": "a", "ferrum_lora_path": "/a", "ferrum_lora_model_id": 3}),
            json!({"ferrum_lora_adapter": "a", "ferrum_lora_path": "/a", "ferrum_lora_model_id": " "}),
        ] {
            let metadata = metadata(value);
            for support in [
                AdapterExecutionSupport::Unsupported,
                AdapterExecutionSupport::PerSequenceLora,
            ] {
                assert!(
                    support.validate_metadata(&metadata).is_err(),
                    "{metadata:?}"
                );
            }
        }
    }
}
