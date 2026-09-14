//! Exact reconstruction of the explicitly selected Orchestral text-parts wire.
//! This validates the whole content value; equivalent flattened text is not
//! evidence that the declared parts, metadata and source bytes were replayed.

use serde_json::{json, Value};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Revision {
    V1,
    V2,
    V3,
}

pub(super) fn content(result: &Value, is_error: bool) -> Value {
    content_with_revision(result, is_error, Revision::V1)
}

pub(super) fn content_v2(result: &Value, is_error: bool) -> Value {
    content_with_revision(result, is_error, Revision::V2)
}

pub(super) fn content_v3(result: &Value, is_error: bool) -> Value {
    content_with_revision(result, is_error, Revision::V3)
}

fn content_with_revision(result: &Value, is_error: bool, revision: Revision) -> Value {
    let extract = |text: &str| revision == Revision::V1 || text.contains(['\n', '\r']);
    let mut metadata = json!({"is_error": is_error, "result": result});
    if revision == Revision::V3 && !is_error {
        metadata.as_object_mut().unwrap().remove("is_error");
    }
    let mut texts = Vec::new();
    match result {
        Value::String(text) if extract(text) => {
            metadata.as_object_mut().unwrap().remove("result");
            texts.push(("Result text".to_owned(), text.as_str()));
        }
        Value::Object(fields) => {
            let mut fields = fields.iter().collect::<Vec<_>>();
            fields.sort_by(|(a, _), (b, _)| a.cmp(b));
            for (key, value) in fields {
                if let Some(text) = value.as_str().filter(|text| extract(text)) {
                    metadata["result"].as_object_mut().unwrap().remove(key);
                    let title = if revision == Revision::V3 {
                        json!(key).to_string()
                    } else {
                        format!("Text field {}", json!(key))
                    };
                    texts.push((title, text));
                }
            }
        }
        _ => {}
    }
    let heading = if revision == Revision::V3 {
        ""
    } else {
        "Tool result metadata:\n"
    };
    let mut parts = vec![json!({
        "type": "text",
        "text": format!("{heading}{}\n", sorted_objects(metadata))
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
    fn v3_compact_wire_preserves_typed_metadata_source_and_error_state() {
        let result = json!({
            "path":"src/λ.rs", "empty":"", "exit_code":0, "optional":null,
            "nested":{"z":false,"a":"kept\ninside JSON"},
            "stdout":"α\r\n````\nend", "stderr":"warning\r"
        });
        let expected = json!([
            {"type":"text","text":"{\"result\":{\"empty\":\"\",\"exit_code\":0,\"nested\":{\"a\":\"kept\\ninside JSON\",\"z\":false},\"optional\":null,\"path\":\"src/λ.rs\"}}\n"},
            {"type":"text","text":"\"stderr\" (final newline: no)\n```text\nwarning\r\n```\n"},
            {"type":"text","text":"\"stdout\" (final newline: no)\n`````text\nα\r\n````\nend\n`````\n"}
        ]);
        assert_eq!(content_v3(&result, false), expected);
        let mut error = expected;
        error[0]["text"] = json!("{\"is_error\":true,\"result\":{\"empty\":\"\",\"exit_code\":0,\"nested\":{\"a\":\"kept\\ninside JSON\",\"z\":false},\"optional\":null,\"path\":\"src/λ.rs\"}}\n");
        assert_eq!(content_v3(&result, true), error);
        assert_eq!(
            content_v3(&json!("line\r\n"), false),
            json!([
                {"type":"text","text":"{}\n"},
                {"type":"text","text":"Result text (final newline: yes)\n```text\nline\r\n```\n"}
            ])
        );
        assert_eq!(
            content_v3(&json!({"\"key\n":"body\n"}), true)[1]["text"],
            "\"\\\"key\\n\" (final newline: yes)\n```text\nbody\n```\n"
        );
        for result in [
            json!(""),
            json!("single line"),
            json!(null),
            json!(["nested\n"]),
        ] {
            for is_error in [false, true] {
                let mut metadata = json!({"result":result});
                if is_error {
                    metadata["is_error"] = json!(true);
                }
                assert_eq!(
                    content_v3(&result, is_error),
                    json!([{"type":"text","text":format!("{}\n", sorted_objects(metadata))}])
                );
            }
        }
    }

    #[test]
    fn v2_keeps_scalars_in_exact_metadata_and_only_extracts_lf_or_cr_text() {
        let result = json!({
            "empty":"", "path":"src/λ.rs", "count":2, "alive":false,
            "optional":null, "nested":{"text":"kept\ninside JSON"},
            "stdout":"α\n````\nend", "stderr":"warning\r"
        });
        assert_eq!(
            content_v2(&result, false),
            json!([
                {"type":"text","text":"Tool result metadata:\n{\"is_error\":false,\"result\":{\"alive\":false,\"count\":2,\"empty\":\"\",\"nested\":{\"text\":\"kept\\ninside JSON\"},\"optional\":null,\"path\":\"src/λ.rs\"}}\n"},
                {"type":"text","text":"Text field \"stderr\" (final newline: no)\n```text\nwarning\r\n```\n"},
                {"type":"text","text":"Text field \"stdout\" (final newline: no)\n`````text\nα\n````\nend\n`````\n"}
            ])
        );
        assert_ne!(content(&result, false), content_v2(&result, false));
        for text in ["", "one line", "λ"] {
            assert_eq!(
                content_v2(&json!(text), true),
                json!([
                    {"type":"text","text":format!("Tool result metadata:\n{}\n", json!({"is_error":true,"result":text}))}
                ])
            );
            assert_ne!(content(&json!(text), true), content_v2(&json!(text), true));
        }
        for text in ["LF\n", "CR\r", "CRLF\r\n", "no\nfinal newline"] {
            assert_eq!(
                content_v2(&json!(text), false),
                content(&json!(text), false)
            );
        }
    }

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
