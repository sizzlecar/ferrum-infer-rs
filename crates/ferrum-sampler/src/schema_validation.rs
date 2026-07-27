//! Semantic validation for the JSON Schema subset used by guided decoding.
//!
//! Generation constraints and final validation have different jobs: the
//! regex translator limits token choices, while this module validates parsed
//! JSON without depending on property order or whitespace serialization.

use serde_json::{Map, Value};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum JsonSchemaValidationError {
    #[error("invalid JSON Schema at {path}: {message}")]
    InvalidSchema { path: String, message: String },
    #[error("JSON value at {path} does not satisfy schema: {message}")]
    Mismatch { path: String, message: String },
}

pub fn validate_json_schema_value(
    schema: &Value,
    value: &Value,
) -> Result<(), JsonSchemaValidationError> {
    validate_at(schema, value, "$".to_string())
}

fn validate_at(
    schema: &Value,
    value: &Value,
    path: String,
) -> Result<(), JsonSchemaValidationError> {
    let schema = schema
        .as_object()
        .ok_or_else(|| invalid_schema(&path, "schema must be an object"))?;

    if let Some(expected) = schema.get("const") {
        if value != expected {
            return Err(mismatch(&path, "value differs from const"));
        }
    }
    if let Some(allowed) = schema.get("enum") {
        let allowed = allowed
            .as_array()
            .ok_or_else(|| invalid_schema(&path, "enum must be an array"))?;
        if allowed.is_empty() {
            return Err(invalid_schema(&path, "enum must not be empty"));
        }
        if !allowed.contains(value) {
            return Err(mismatch(&path, "value is not in enum"));
        }
    }

    let Some(schema_type) = schema.get("type") else {
        if schema.contains_key("const") || schema.contains_key("enum") {
            return Ok(());
        }
        return Err(invalid_schema(&path, "missing type, const, or enum"));
    };
    let schema_type = schema_type
        .as_str()
        .ok_or_else(|| invalid_schema(&path, "type must be a string"))?;

    match schema_type {
        "object" => validate_object(schema, value, &path),
        "array" => validate_array(schema, value, &path),
        "string" => validate_string(schema, value, &path),
        "integer" if is_json_integer(value) => Ok(()),
        "integer" => Err(mismatch(&path, "expected integer")),
        "number" if value.is_number() => Ok(()),
        "number" => Err(mismatch(&path, "expected number")),
        "boolean" if value.is_boolean() => Ok(()),
        "boolean" => Err(mismatch(&path, "expected boolean")),
        "null" if value.is_null() => Ok(()),
        "null" => Err(mismatch(&path, "expected null")),
        other => Err(invalid_schema(&path, format!("unsupported type '{other}'"))),
    }
}

fn validate_object(
    schema: &Map<String, Value>,
    value: &Value,
    path: &str,
) -> Result<(), JsonSchemaValidationError> {
    let object = value
        .as_object()
        .ok_or_else(|| mismatch(path, "expected object"))?;
    let properties = schema
        .get("properties")
        .map(|value| {
            value
                .as_object()
                .ok_or_else(|| invalid_schema(path, "properties must be an object"))
        })
        .transpose()?;

    if let Some(required) = schema.get("required") {
        let required = required
            .as_array()
            .ok_or_else(|| invalid_schema(path, "required must be an array"))?;
        for key in required {
            let key = key
                .as_str()
                .ok_or_else(|| invalid_schema(path, "required entries must be strings"))?;
            if !object.contains_key(key) {
                return Err(mismatch(path, format!("missing required property '{key}'")));
            }
        }
    }

    for (key, member) in object {
        let member_path = format!("{path}.{key}");
        if let Some(member_schema) = properties.and_then(|properties| properties.get(key)) {
            validate_at(member_schema, member, member_path)?;
            continue;
        }
        match schema.get("additionalProperties") {
            Some(Value::Bool(false)) => {
                return Err(mismatch(path, format!("unexpected property '{key}'")));
            }
            Some(Value::Object(member_schema)) => {
                validate_at(&Value::Object(member_schema.clone()), member, member_path)?;
            }
            Some(Value::Bool(true)) | None => {}
            Some(_) => {
                return Err(invalid_schema(
                    path,
                    "additionalProperties must be a boolean or schema object",
                ));
            }
        }
    }
    Ok(())
}

fn validate_array(
    schema: &Map<String, Value>,
    value: &Value,
    path: &str,
) -> Result<(), JsonSchemaValidationError> {
    let array = value
        .as_array()
        .ok_or_else(|| mismatch(path, "expected array"))?;
    let items = schema
        .get("items")
        .ok_or_else(|| invalid_schema(path, "array schema missing items"))?;
    for (index, member) in array.iter().enumerate() {
        validate_at(items, member, format!("{path}[{index}]"))?;
    }
    Ok(())
}

fn validate_string(
    schema: &Map<String, Value>,
    value: &Value,
    path: &str,
) -> Result<(), JsonSchemaValidationError> {
    let text = value
        .as_str()
        .ok_or_else(|| mismatch(path, "expected string"))?;
    let char_count = text.chars().count() as u64;
    if let Some(min) = schema.get("minLength") {
        let min = min
            .as_u64()
            .ok_or_else(|| invalid_schema(path, "minLength must be a non-negative integer"))?;
        if char_count < min {
            return Err(mismatch(path, format!("string length is below {min}")));
        }
    }
    if let Some(max) = schema.get("maxLength") {
        let max = max
            .as_u64()
            .ok_or_else(|| invalid_schema(path, "maxLength must be a non-negative integer"))?;
        if char_count > max {
            return Err(mismatch(path, format!("string length exceeds {max}")));
        }
    }
    Ok(())
}

fn is_json_integer(value: &Value) -> bool {
    let Some(number) = value.as_number() else {
        return false;
    };
    number.as_i64().is_some()
        || number.as_u64().is_some()
        || number.as_f64().is_some_and(|value| value.fract() == 0.0)
}

fn invalid_schema(path: &str, message: impl Into<String>) -> JsonSchemaValidationError {
    JsonSchemaValidationError::InvalidSchema {
        path: path.to_string(),
        message: message.into(),
    }
}

fn mismatch(path: &str, message: impl Into<String>) -> JsonSchemaValidationError {
    JsonSchemaValidationError::Mismatch {
        path: path.to_string(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validates_required_const_property_without_serialization_constraints() {
        let schema = json!({
            "type": "object",
            "properties": {"value": {"type": "string", "const": "marker-21"}},
            "required": ["value"],
            "additionalProperties": false
        });
        validate_json_schema_value(&schema, &json!({"value": "marker-21"})).unwrap();
        let error = validate_json_schema_value(&schema, &json!({"value": "wrong"})).unwrap_err();
        assert!(error.to_string().contains("differs from const"));
    }

    #[test]
    fn validates_optional_properties_and_rejects_unknown_properties() {
        let schema = json!({
            "type": "object",
            "properties": {
                "required": {"type": "string"},
                "optional": {"type": "integer"}
            },
            "required": ["required"],
            "additionalProperties": false
        });
        validate_json_schema_value(&schema, &json!({"optional": 2, "required": "ok"})).unwrap();
        let error =
            validate_json_schema_value(&schema, &json!({"required": "ok", "unknown": true}))
                .unwrap_err();
        assert!(error.to_string().contains("unexpected property 'unknown'"));
    }

    #[test]
    fn object_without_properties_accepts_any_object() {
        validate_json_schema_value(&json!({"type": "object"}), &json!({"city": "Paris"})).unwrap();
    }
}
