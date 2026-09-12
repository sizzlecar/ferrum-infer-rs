//! Exact reconstruction of the explicitly selected Orchestral text-parts wire.
//! This validates the whole content value; equivalent flattened text is not
//! evidence that the declared parts, metadata and source bytes were replayed.

use serde_json::{json, Value};

pub(super) fn content(result: &Value, is_error: bool) -> Value {
    let mut metadata = json!({"is_error": is_error, "result": result});
    let mut texts = Vec::new();
    match result {
        Value::String(text) => {
            metadata.as_object_mut().unwrap().remove("result");
            texts.push(("Result text".to_owned(), text.as_str()));
        }
        Value::Object(fields) => {
            let mut fields = fields.iter().collect::<Vec<_>>();
            fields.sort_by(|(a, _), (b, _)| a.cmp(b));
            for (key, value) in fields {
                if let Some(text) = value.as_str() {
                    metadata["result"].as_object_mut().unwrap().remove(key);
                    texts.push((format!("Text field {}", json!(key)), text));
                }
            }
        }
        _ => {}
    }
    let mut parts = vec![json!({
        "type": "text",
        "text": format!("Tool result metadata:\n{}\n", sorted_objects(metadata))
    })];
    for (title, text) in texts {
        let longest_ticks = text.split(|c| c != '`').map(str::len).max().unwrap_or(0);
        let fence = "`".repeat(3.max(longest_ticks + 1));
        let has_final_newline = text.ends_with('\n');
        let newline_label = if has_final_newline { "yes" } else { "no" };
        let separator = if has_final_newline { "" } else { "\n" };
        parts.push(json!({
            "type": "text",
            "text": format!("{title} (final newline: {newline_label})\n{fence}text\n{text}{separator}{fence}\n")
        }));
    }
    Value::Array(parts)
}

fn sorted_objects(value: Value) -> Value {
    match value {
        Value::Object(fields) => {
            let fields = fields
                .into_iter()
                .collect::<std::collections::BTreeMap<_, _>>();
            Value::Object(
                fields
                    .into_iter()
                    .map(|(key, value)| (key, sorted_objects(value)))
                    .collect(),
            )
        }
        Value::Array(values) => Value::Array(values.into_iter().map(sorted_objects).collect()),
        value => value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_and_non_text_metadata_are_preserved_without_escaped_duplicates() {
        let source = "    let path = \"C:\\work\";\r\n\t// ```\r\n";
        let parts = content(
            &json!({"content":source,"line":7,"nested":{"text":"retained"}}),
            false,
        );
        assert_eq!(
            parts[0],
            json!({"type":"text","text":"Tool result metadata:\n{\"is_error\":false,\"result\":{\"line\":7,\"nested\":{\"text\":\"retained\"}}}\n"})
        );
        assert_eq!(
            parts[1],
            json!({"type":"text","text":format!("Text field \"content\" (final newline: yes)\n````text\n{source}````\n")})
        );
        assert_eq!(parts.as_array().unwrap().len(), 2);
    }

    #[test]
    fn root_strings_empty_strings_and_missing_final_newlines_are_distinct() {
        assert_eq!(
            content(&json!("  x"), true),
            json!([
                {"type":"text","text":"Tool result metadata:\n{\"is_error\":true}\n"},
                {"type":"text","text":"Result text (final newline: no)\n```text\n  x\n```\n"}
            ])
        );
        assert_eq!(
            content(&json!(""), false)[1]["text"],
            "Result text (final newline: no)\n```text\n\n```\n"
        );
        assert_eq!(
            content(&json!(null), false),
            json!([
                {"type":"text","text":"Tool result metadata:\n{\"is_error\":false,\"result\":null}\n"}
            ])
        );
    }
}
