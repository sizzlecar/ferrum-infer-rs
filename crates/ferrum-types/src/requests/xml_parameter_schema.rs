//! Conservative JSON Schema probing for Qwen-style XML tool parameters.

use std::collections::{HashMap, HashSet};

const MAX_SCHEMA_PROBE_DEPTH: usize = 64;
const MAX_SCHEMA_PROBE_NODES: usize = 4096;

type JsonTypeMask = u8;

const JSON_TYPE_NONE: JsonTypeMask = 0;
const JSON_TYPE_NULL: JsonTypeMask = 1 << 0;
const JSON_TYPE_BOOLEAN: JsonTypeMask = 1 << 1;
const JSON_TYPE_NUMBER: JsonTypeMask = 1 << 2;
const JSON_TYPE_STRING: JsonTypeMask = 1 << 3;
const JSON_TYPE_ARRAY: JsonTypeMask = 1 << 4;
const JSON_TYPE_OBJECT: JsonTypeMask = 1 << 5;
const JSON_TYPE_ALL: JsonTypeMask = JSON_TYPE_NULL
    | JSON_TYPE_BOOLEAN
    | JSON_TYPE_NUMBER
    | JSON_TYPE_STRING
    | JSON_TYPE_ARRAY
    | JSON_TYPE_OBJECT;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SchemaRefBehavior {
    /// Drafts 4 through 7 ignore every keyword adjacent to `$ref`.
    Exclusive,
    /// Draft 2019-09 and 2020-12 apply `$ref` and its siblings together.
    Adjacent,
    /// An unknown dialect cannot be resolved safely by this small probe.
    Disabled,
}

/// A bounded, conservative proof of the JSON types a tool parameter may take.
///
/// The probe only converts XML text when the declared schema proves that no
/// string can be valid. Its masks are upper bounds: an unsupported keyword,
/// reference, cycle, or exhausted budget returns every type and therefore
/// preserves the original text. The server's full JSON Schema validator remains
/// authoritative after parsing.
pub(super) struct XmlParameterSchemaProbe<'a> {
    root: &'a serde_json::Value,
    remaining_nodes: usize,
    ref_behavior: SchemaRefBehavior,
    supports_const: bool,
    active_value_refs: HashSet<String>,
    value_ref_memo: HashMap<String, JsonTypeMask>,
    active_parameter_refs: HashSet<(String, String)>,
    parameter_ref_memo: HashMap<(String, String), JsonTypeMask>,
}

impl<'a> XmlParameterSchemaProbe<'a> {
    pub(super) fn new(root: &'a serde_json::Value) -> Self {
        let (ref_behavior, supports_const) = schema_dialect_behavior(root);
        Self {
            root,
            remaining_nodes: MAX_SCHEMA_PROBE_NODES,
            ref_behavior,
            supports_const,
            active_value_refs: HashSet::new(),
            value_ref_memo: HashMap::new(),
            active_parameter_refs: HashSet::new(),
            parameter_ref_memo: HashMap::new(),
        }
    }

    pub(super) fn decode(&mut self, name: &str, value: &str) -> serde_json::Value {
        if self.ref_behavior == SchemaRefBehavior::Disabled {
            return serde_json::Value::String(value.to_string());
        }
        let possible_types = self.parameter_types(self.root, name, 0);
        if possible_types == JSON_TYPE_NONE || possible_types & JSON_TYPE_STRING != 0 {
            return serde_json::Value::String(value.to_string());
        }

        // Keep malformed native JSON as text. The authoritative schema
        // validator at the API boundary then returns a controlled error rather
        // than letting an auto tool call fall back to raw XML assistant content.
        serde_json::from_str(value).unwrap_or_else(|_| serde_json::Value::String(value.to_string()))
    }

    fn visit(&mut self, depth: usize) -> bool {
        if depth >= MAX_SCHEMA_PROBE_DEPTH || self.remaining_nodes == 0 {
            return false;
        }
        self.remaining_nodes -= 1;
        true
    }

    /// Upper-bound the types of `name` when the root instance is an object
    /// containing that member.
    fn parameter_types(
        &mut self,
        schema: &'a serde_json::Value,
        name: &str,
        depth: usize,
    ) -> JsonTypeMask {
        if !self.visit(depth) {
            return JSON_TYPE_ALL;
        }
        match schema {
            serde_json::Value::Bool(false) => return JSON_TYPE_NONE,
            serde_json::Value::Bool(true) => return JSON_TYPE_ALL,
            _ => {}
        }
        let Some(object) = schema.as_object() else {
            return JSON_TYPE_ALL;
        };
        if has_nested_schema_boundary(schema, self.root) {
            return JSON_TYPE_ALL;
        }

        let mut possible = JSON_TYPE_ALL;
        if object.contains_key("$ref") {
            let Some(reference) = object.get("$ref").and_then(serde_json::Value::as_str) else {
                return JSON_TYPE_ALL;
            };
            let Some(referenced) = self.parameter_ref_types(reference, name, depth + 1) else {
                return JSON_TYPE_ALL;
            };
            if self.ref_behavior == SchemaRefBehavior::Exclusive {
                return referenced;
            }
            possible &= referenced;
        }

        if let Some(schema_types) = object.get("type").and_then(json_schema_type_mask) {
            if schema_types & JSON_TYPE_OBJECT == 0 {
                return JSON_TYPE_NONE;
            }
        }

        possible &= self.direct_parameter_types(object, name, depth + 1);
        possible &= self.combined_parameter_types(object, name, depth + 1);
        possible
    }

    fn direct_parameter_types(
        &mut self,
        schema: &'a serde_json::Map<String, serde_json::Value>,
        name: &str,
        depth: usize,
    ) -> JsonTypeMask {
        match schema.get("properties") {
            Some(serde_json::Value::Object(properties)) => {
                if let Some(property) = properties.get(name) {
                    return self.value_types(property, depth);
                }
            }
            Some(_) => return JSON_TYPE_ALL,
            None => {}
        }

        match schema.get("patternProperties") {
            Some(serde_json::Value::Object(patterns)) if patterns.is_empty() => {}
            Some(serde_json::Value::Object(_)) | Some(_) => return JSON_TYPE_ALL,
            None => {}
        }

        match schema.get("additionalProperties") {
            Some(value @ (serde_json::Value::Object(_) | serde_json::Value::Bool(_))) => {
                self.value_types(value, depth)
            }
            Some(_) | None => JSON_TYPE_ALL,
        }
    }

    fn combined_parameter_types(
        &mut self,
        schema: &'a serde_json::Map<String, serde_json::Value>,
        name: &str,
        depth: usize,
    ) -> JsonTypeMask {
        let mut possible = JSON_TYPE_ALL;
        if let Some(branches) = nonempty_schema_array(schema.get("allOf")) {
            for branch in branches {
                possible &= self.parameter_types(branch, name, depth);
            }
        }
        for keyword in ["anyOf", "oneOf"] {
            if let Some(branches) = nonempty_schema_array(schema.get(keyword)) {
                let mut union = JSON_TYPE_NONE;
                for branch in branches {
                    union |= self.parameter_types(branch, name, depth);
                }
                possible &= union;
            }
        }
        possible
    }

    fn parameter_ref_types(
        &mut self,
        reference: &str,
        name: &str,
        depth: usize,
    ) -> Option<JsonTypeMask> {
        if self.ref_behavior == SchemaRefBehavior::Disabled {
            return None;
        }
        let key = (reference.to_string(), name.to_string());
        if let Some(possible) = self.parameter_ref_memo.get(&key) {
            return Some(*possible);
        }
        if !self.active_parameter_refs.insert(key.clone()) {
            return Some(JSON_TYPE_ALL);
        }
        let root = self.root;
        let possible = resolve_local_schema_ref(root, reference)
            .map(|resolved| self.parameter_types(resolved, name, depth));
        self.active_parameter_refs.remove(&key);
        if let Some(possible) = possible {
            self.parameter_ref_memo.insert(key, possible);
        }
        possible
    }

    fn value_types(&mut self, schema: &'a serde_json::Value, depth: usize) -> JsonTypeMask {
        if !self.visit(depth) {
            return JSON_TYPE_ALL;
        }
        match schema {
            serde_json::Value::Bool(false) => return JSON_TYPE_NONE,
            serde_json::Value::Bool(true) => return JSON_TYPE_ALL,
            _ => {}
        }
        let Some(object) = schema.as_object() else {
            return JSON_TYPE_ALL;
        };
        if has_nested_schema_boundary(schema, self.root) {
            return JSON_TYPE_ALL;
        }

        let mut possible = JSON_TYPE_ALL;
        if object.contains_key("$ref") {
            let Some(reference) = object.get("$ref").and_then(serde_json::Value::as_str) else {
                return JSON_TYPE_ALL;
            };
            let Some(referenced) = self.value_ref_types(reference, depth + 1) else {
                return JSON_TYPE_ALL;
            };
            if self.ref_behavior == SchemaRefBehavior::Exclusive {
                return referenced;
            }
            possible &= referenced;
        }

        if let Some(schema_types) = object.get("type").and_then(json_schema_type_mask) {
            possible &= schema_types;
        }
        if self.supports_const {
            if let Some(value) = object.get("const") {
                possible &= json_value_type_mask(value);
            }
        }
        if let Some(values) = nonempty_schema_array(object.get("enum")) {
            possible &= values.iter().fold(JSON_TYPE_NONE, |mask, value| {
                mask | json_value_type_mask(value)
            });
        }
        if let Some(branches) = nonempty_schema_array(object.get("allOf")) {
            for branch in branches {
                possible &= self.value_types(branch, depth + 1);
            }
        }
        for keyword in ["anyOf", "oneOf"] {
            if let Some(branches) = nonempty_schema_array(object.get(keyword)) {
                let mut union = JSON_TYPE_NONE;
                for branch in branches {
                    union |= self.value_types(branch, depth + 1);
                }
                possible &= union;
            }
        }
        possible
    }

    fn value_ref_types(&mut self, reference: &str, depth: usize) -> Option<JsonTypeMask> {
        if self.ref_behavior == SchemaRefBehavior::Disabled {
            return None;
        }
        if let Some(possible) = self.value_ref_memo.get(reference) {
            return Some(*possible);
        }
        if !self.active_value_refs.insert(reference.to_string()) {
            return Some(JSON_TYPE_ALL);
        }
        let root = self.root;
        let possible = resolve_local_schema_ref(root, reference)
            .map(|resolved| self.value_types(resolved, depth));
        self.active_value_refs.remove(reference);
        if let Some(possible) = possible {
            self.value_ref_memo.insert(reference.to_string(), possible);
        }
        possible
    }
}

fn schema_dialect_behavior(root: &serde_json::Value) -> (SchemaRefBehavior, bool) {
    let Some(schema) = root.get("$schema") else {
        // jsonschema 0.48 defaults schemas without an explicit dialect to
        // Draft 2020-12, where adjacent `$ref` keywords apply.
        return (SchemaRefBehavior::Adjacent, true);
    };
    let Some(schema) = schema.as_str() else {
        return (SchemaRefBehavior::Disabled, false);
    };
    match schema.trim_end_matches('#') {
        "https://json-schema.org/draft-04/schema" | "http://json-schema.org/draft-04/schema" => {
            (SchemaRefBehavior::Exclusive, false)
        }
        "https://json-schema.org/draft-06/schema"
        | "http://json-schema.org/draft-06/schema"
        | "https://json-schema.org/draft-07/schema"
        | "http://json-schema.org/draft-07/schema" => (SchemaRefBehavior::Exclusive, true),
        "https://json-schema.org/draft/2019-09/schema"
        | "http://json-schema.org/draft/2019-09/schema"
        | "https://json-schema.org/draft/2020-12/schema"
        | "http://json-schema.org/draft/2020-12/schema" => (SchemaRefBehavior::Adjacent, true),
        _ => (SchemaRefBehavior::Disabled, false),
    }
}

fn has_nested_schema_boundary(schema: &serde_json::Value, root: &serde_json::Value) -> bool {
    !std::ptr::eq(schema, root)
        && schema.as_object().is_some_and(|object| {
            object.get("$id").is_some_and(serde_json::Value::is_string)
                || object.get("id").is_some_and(serde_json::Value::is_string)
                || object
                    .get("$schema")
                    .is_some_and(serde_json::Value::is_string)
        })
}

fn json_schema_type_mask(value: &serde_json::Value) -> Option<JsonTypeMask> {
    fn named_type(kind: &str) -> Option<JsonTypeMask> {
        match kind {
            "null" => Some(JSON_TYPE_NULL),
            "boolean" => Some(JSON_TYPE_BOOLEAN),
            "number" | "integer" => Some(JSON_TYPE_NUMBER),
            "string" => Some(JSON_TYPE_STRING),
            "array" => Some(JSON_TYPE_ARRAY),
            "object" => Some(JSON_TYPE_OBJECT),
            _ => None,
        }
    }

    match value {
        serde_json::Value::String(kind) => named_type(kind),
        serde_json::Value::Array(kinds) if !kinds.is_empty() => {
            kinds.iter().try_fold(JSON_TYPE_NONE, |mask, kind| {
                named_type(kind.as_str()?).map(|kind| mask | kind)
            })
        }
        _ => None,
    }
}

fn json_value_type_mask(value: &serde_json::Value) -> JsonTypeMask {
    match value {
        serde_json::Value::Null => JSON_TYPE_NULL,
        serde_json::Value::Bool(_) => JSON_TYPE_BOOLEAN,
        serde_json::Value::Number(_) => JSON_TYPE_NUMBER,
        serde_json::Value::String(_) => JSON_TYPE_STRING,
        serde_json::Value::Array(_) => JSON_TYPE_ARRAY,
        serde_json::Value::Object(_) => JSON_TYPE_OBJECT,
    }
}

fn nonempty_schema_array(value: Option<&serde_json::Value>) -> Option<&[serde_json::Value]> {
    value
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
        .filter(|values| !values.is_empty())
}

fn resolve_local_schema_ref<'a>(
    root_schema: &'a serde_json::Value,
    reference: &str,
) -> Option<&'a serde_json::Value> {
    let pointer = reference.strip_prefix('#')?;
    if pointer.is_empty() {
        return Some(root_schema);
    }
    // URI fragments are percent-decoded before JSON Pointer evaluation. This
    // small resolver deliberately does not implement URI decoding, so reject
    // encoded fragments instead of looking up the wrong literal `%xx` key.
    if pointer.contains('%') {
        return None;
    }
    let pointer = pointer.strip_prefix('/')?;
    let mut current = root_schema;
    for encoded_token in pointer.split('/') {
        let token = decode_json_pointer_token(encoded_token)?;
        current = match current {
            serde_json::Value::Object(object) => object.get(&token)?,
            serde_json::Value::Array(array) => {
                if token.len() > 1 && token.starts_with('0') {
                    return None;
                }
                array.get(token.parse::<usize>().ok()?)?
            }
            _ => return None,
        };

        // A nested identifier or dialect declaration starts a schema boundary.
        // Resolving a fragment from the tool root through that boundary could
        // give `#` the wrong base or keywords the wrong meaning, so preserve
        // the parameter text instead.
        if has_nested_schema_boundary(current, root_schema) {
            return None;
        }
    }
    Some(current)
}

fn decode_json_pointer_token(token: &str) -> Option<String> {
    let mut decoded = String::with_capacity(token.len());
    let mut chars = token.chars();
    while let Some(character) = chars.next() {
        if character != '~' {
            decoded.push(character);
            continue;
        }
        match chars.next()? {
            '0' => decoded.push('~'),
            '1' => decoded.push('/'),
            _ => return None,
        }
    }
    Some(decoded)
}
