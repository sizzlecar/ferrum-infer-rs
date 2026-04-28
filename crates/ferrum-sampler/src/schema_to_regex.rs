//! Minimal JSON Schema → regex translator for structured output.
//!
//! Accepts a subset of JSON Schema sufficient for most structured-output
//! use cases — the classic "give me a typed object back" prompt flow
//! that OpenAI's `response_format = json_schema` enables. Coverage:
//!
//!   - `{ "type": "string" }` → `"[^"\\]*"` (no escape handling yet —
//!     models very rarely emit backslashes in short structured answers).
//!   - `{ "type": "integer" }` → `-?\d+`
//!   - `{ "type": "number" }`  → `-?\d+(\.\d+)?([eE][+-]?\d+)?`
//!   - `{ "type": "boolean" }` → `true|false`
//!   - `{ "type": "null" }`    → `null`
//!   - `{ "enum": [...] }`     → alternation of JSON-quoted values
//!   - `{ "type": "array", "items": T }` → `\[\s*(T(\s*,\s*T)*)?\s*\]`
//!   - `{ "type": "object", "properties": {...}, "required": [...] }`
//!     → required fields in declared order: `\{ "k1": T1, "k2": T2, ... \}`
//!     Optional fields are skipped to keep the DFA small.
//!
//! Not supported (falls back to JsonObject-style unconstrained JSON):
//!   - `$ref`, `oneOf`, `anyOf`, `allOf`, `not`
//!   - nested-depth limits, min/max constraints, string patterns
//!   - additionalProperties, propertyNames
//!
//! When the translator can't handle a schema, the caller should skip
//! guided decoding rather than silently produce wrong output.

use ferrum_types::{FerrumError, Result};
use serde_json::Value;

/// Compile a JSON Schema (serialised as a JSON string) into a regex
/// pattern suitable for feeding into `RegexGuidedProcessor`.
///
/// The returned pattern:
///   * has a leading `\s*` so BPE tokenisers that prepend a space
///     to the first generated token (Qwen3, Llama-3) still transition
///     cleanly — without it the DFA would die at the very first step.
///   * is anchored to end internally by the processor (which wraps
///     with `^(?:...)\z`).
pub fn schema_to_regex(schema_json: &str) -> Result<String> {
    let schema: Value = serde_json::from_str(schema_json).map_err(|e| {
        FerrumError::invalid_request(format!("response_format.schema is not valid JSON: {e}"))
    })?;
    let inner = translate(&schema)?;
    Ok(format!(r"\s*{inner}\s*"))
}

fn translate(node: &Value) -> Result<String> {
    // `enum` takes precedence over `type`.
    if let Some(en) = node.get("enum").and_then(|v| v.as_array()) {
        return enum_pattern(en);
    }

    let ty = node.get("type").and_then(|v| v.as_str());
    match ty {
        Some("string") => Ok(r#""[^"]*""#.to_string()),
        Some("integer") => Ok(r"-?\d+".to_string()),
        Some("number") => Ok(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?".to_string()),
        Some("boolean") => Ok("(?:true|false)".to_string()),
        Some("null") => Ok("null".to_string()),
        Some("array") => array_pattern(node),
        Some("object") => object_pattern(node),
        Some(other) => Err(FerrumError::invalid_request(format!(
            "unsupported JSON Schema type '{other}' in response_format"
        ))),
        None => Err(FerrumError::invalid_request(
            "JSON Schema node missing 'type' field and no 'enum' present",
        )),
    }
}

fn enum_pattern(values: &[Value]) -> Result<String> {
    if values.is_empty() {
        return Err(FerrumError::invalid_request(
            "response_format enum must have at least one value",
        ));
    }
    let mut alts: Vec<String> = Vec::with_capacity(values.len());
    for v in values {
        // Serialise each allowed value exactly as it would appear in JSON
        // output, then escape regex metachars.
        let literal = serde_json::to_string(v).map_err(|e| {
            FerrumError::invalid_request(format!("enum value not JSON-serialisable: {e}"))
        })?;
        alts.push(regex_escape(&literal));
    }
    Ok(format!("(?:{})", alts.join("|")))
}

fn array_pattern(node: &Value) -> Result<String> {
    let items_schema = node
        .get("items")
        .ok_or_else(|| FerrumError::invalid_request("array schema missing 'items'"))?;
    let item_pat = translate(items_schema)?;
    // `\s*` between tokens so minor whitespace variation doesn't break the match.
    Ok(format!(r"\[\s*(?:{item_pat}(?:\s*,\s*{item_pat})*)?\s*\]"))
}

fn object_pattern(node: &Value) -> Result<String> {
    let props = node
        .get("properties")
        .and_then(|v| v.as_object())
        .ok_or_else(|| FerrumError::invalid_request("object schema missing 'properties'"))?;
    let required: Vec<&str> = node
        .get("required")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    // Emit required fields in the order listed. Optional fields dropped
    // for pattern compactness — still produces a superset of valid outputs
    // but model will usually follow the constrained form.
    let keys: Vec<&str> = if required.is_empty() {
        // No `required` → walk properties in insertion order. `properties`
        // is a `serde_json::Map` which is ordered in our dep config, so
        // this is deterministic.
        props.keys().map(String::as_str).collect()
    } else {
        required
    };

    if keys.is_empty() {
        return Ok(r"\{\s*\}".to_string());
    }

    let mut fields: Vec<String> = Vec::with_capacity(keys.len());
    for key in keys {
        let sub = props.get(key).ok_or_else(|| {
            FerrumError::invalid_request(format!(
                "required property '{key}' missing from 'properties'"
            ))
        })?;
        let sub_pat = translate(sub)?;
        let key_literal = regex_escape(&format!("\"{key}\""));
        fields.push(format!(r"\s*{key_literal}\s*:\s*{sub_pat}"));
    }

    Ok(format!(r"\{{{}\s*\}}", fields.join(r"\s*,")))
}

fn regex_escape(s: &str) -> String {
    // regex-syntax::escape would be nicer but pulling that crate in for
    // a few metachars is overkill; explicit table matches its behaviour
    // for our literal inputs.
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        match ch {
            '\\' | '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$'
            | '/' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    // Use the std `regex` crate in tests — it ships with easier full-match
    // semantics than raw regex_automata DFAs and isn't in the prod hot path.
    use regex_lite::Regex;

    fn compile(pat: &str) -> Regex {
        Regex::new(&format!("^(?:{pat})$")).expect("regex compiles")
    }

    #[test]
    fn string_schema_matches_quoted() {
        let re = compile(&schema_to_regex(r#"{"type":"string"}"#).unwrap());
        assert!(re.is_match("\"hello\""));
        assert!(re.is_match("\"\""));
        assert!(!re.is_match("hello"));
        assert!(!re.is_match("\"un\"closed"));
    }

    #[test]
    fn integer_schema() {
        let re = compile(&schema_to_regex(r#"{"type":"integer"}"#).unwrap());
        assert!(re.is_match("0"));
        assert!(re.is_match("42"));
        assert!(re.is_match("-7"));
        assert!(!re.is_match("1.5"));
        assert!(!re.is_match("abc"));
    }

    #[test]
    fn number_schema_accepts_decimal_and_exp() {
        let re = compile(&schema_to_regex(r#"{"type":"number"}"#).unwrap());
        assert!(re.is_match("3"));
        assert!(re.is_match("3.14"));
        assert!(re.is_match("-0.001"));
        assert!(re.is_match("1e5"));
        assert!(re.is_match("1.2e-3"));
    }

    #[test]
    fn boolean_schema() {
        let re = compile(&schema_to_regex(r#"{"type":"boolean"}"#).unwrap());
        assert!(re.is_match("true"));
        assert!(re.is_match("false"));
        assert!(!re.is_match("True"));
    }

    #[test]
    fn enum_schema() {
        let re = compile(&schema_to_regex(r#"{"enum":["red","green","blue"]}"#).unwrap());
        assert!(re.is_match("\"red\""));
        assert!(re.is_match("\"blue\""));
        assert!(!re.is_match("\"yellow\""));
    }

    #[test]
    fn array_of_integers() {
        let re =
            compile(&schema_to_regex(r#"{"type":"array","items":{"type":"integer"}}"#).unwrap());
        assert!(re.is_match("[]"));
        assert!(re.is_match("[1]"));
        assert!(re.is_match("[1, 2, 3]"));
        assert!(re.is_match("[-1, 0, 2]"));
        assert!(!re.is_match("[1.5]"));
        assert!(!re.is_match("[1, \"two\"]"));
    }

    #[test]
    fn object_with_required_fields() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }"#;
        let re = compile(&schema_to_regex(schema).unwrap());
        assert!(re.is_match(r#"{"name": "Alice", "age": 30}"#));
        assert!(re.is_match(r#"{ "name":"Bob" , "age":7 }"#));
        assert!(!re.is_match(r#"{"name": "Alice"}"#));
        assert!(!re.is_match(r#"{"age": 30, "name": "Alice"}"#));
    }

    #[test]
    fn nested_object_and_array() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "count": {"type": "integer"}
            },
            "required": ["tags", "count"]
        }"#;
        let re = compile(&schema_to_regex(schema).unwrap());
        assert!(re.is_match(r#"{"tags": ["a", "b"], "count": 2}"#));
        assert!(re.is_match(r#"{"tags": [], "count": 0}"#));
        assert!(!re.is_match(r#"{"tags": ["a"], "count": "two"}"#));
    }

    #[test]
    fn unsupported_type_errors_clearly() {
        let err = schema_to_regex(r#"{"type":"mystery"}"#).unwrap_err();
        assert!(
            err.to_string().contains("unsupported JSON Schema type"),
            "got: {err}"
        );
    }
}
