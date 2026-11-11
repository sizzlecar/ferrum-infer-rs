//! Accepted wire declarations for the existing fixed axial-rotary profile.
//! This validates architecture metadata only; it does not select a new
//! activation dtype or qualify Comfy's F32 visual execution path.
use super::invalid;
use ferrum_runtime::VNextError;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AxialRotaryDeclaration {
    rope_type: AxialRotaryKind,
    rope_theta: f64,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum AxialRotaryKind {
    Axial,
}

pub(super) fn check_vision_declaration(v: &Value) -> Result<(), VNextError> {
    if !matches!(
        v.get("model_type").and_then(Value::as_str),
        Some("qwen3_vl" | "qwen3_vl_vision")
    ) || v.get("dtype").and_then(Value::as_str) != Some("bfloat16")
        || v.get("hidden_act").and_then(Value::as_str) != Some("gelu_pytorch_tanh")
    {
        return Err(invalid(
            "vision_config",
            "requires declared BF16 Qwen3VL vision with tanh-GELU blocks",
        ));
    }
    check_axial_declaration(v)
}

pub(super) fn check_qwen35_vision_declaration(v: &Value) -> Result<(), VNextError> {
    if v.get("model_type").and_then(Value::as_str) != Some("qwen3_5_vision")
        || !matches!(
            v.get("dtype").and_then(Value::as_str),
            Some("float32" | "bfloat16")
        )
        || v.get("hidden_act").and_then(Value::as_str) != Some("gelu_pytorch_tanh")
        || v.get("rope_parameters").is_none()
    {
        return Err(invalid(
            "vision_config",
            "requires Qwen3.5 vision with declared tanh-GELU and axial rotary parameters",
        ));
    }
    check_axial_declaration(v)
}

fn check_axial_declaration(v: &Value) -> Result<(), VNextError> {
    if v.get("rope_theta").is_some() || v.get("layer_norm_eps").is_some() {
        return Err(invalid(
            "vision_config",
            "top-level rotary or normalization overrides require a separate declared profile",
        ));
    }
    if let Some(raw) = v.get("rope_parameters") {
        let declaration = serde_json::from_value::<AxialRotaryDeclaration>(raw.clone())
            .map_err(|e| invalid("rope_parameters", e.to_string()))?;
        let AxialRotaryKind::Axial = declaration.rope_type;
        if declaration.rope_theta != 10000.0 {
            return Err(invalid(
                "rope_parameters",
                "this profile requires axial rotary with theta 10000",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
