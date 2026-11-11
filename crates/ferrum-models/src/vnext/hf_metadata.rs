//! Shared Hugging Face tokenizer metadata parser for production model packages.

use std::collections::{BTreeMap, BTreeSet};

use ferrum_runtime::{
    CausalLanguageMetadata, SpecialTokenCollision, SpecialTokenCollisionPolicy,
    SpecialTokenMetadata, SpecialTokenRole, TemplateMetadata,
};
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

pub(super) fn parse_hf_model_semantic_metadata(
    model_config: &Value,
    tokenizer_config_bytes: &[u8],
    chat_template_jinja: Option<&[u8]>,
    chat_template_json: Option<&[u8]>,
) -> Result<CausalLanguageMetadata, String> {
    let tokenizer_config: Value = serde_json::from_slice(tokenizer_config_bytes)
        .map_err(|error| format!("parse tokenizer tokenizer_config.json: {error}"))?;
    parse_hf_model_semantic_metadata_from_config(
        model_config,
        tokenizer_config_bytes,
        &tokenizer_config,
        chat_template_jinja,
        chat_template_json,
    )
}

/// Resolve named special tokens from both frozen HF tokenizer artifacts.
/// Newer snapshots keep their IDs only in tokenizer.json's added_tokens.
pub(super) fn parse_hf_model_semantic_metadata_with_tokenizer(
    model_config: &Value,
    tokenizer_config_bytes: &[u8],
    tokenizer_json_bytes: &[u8],
    chat_template_jinja: Option<&[u8]>,
    chat_template_json: Option<&[u8]>,
) -> Result<CausalLanguageMetadata, String> {
    let mut tokenizer_config: Value = serde_json::from_slice(tokenizer_config_bytes)
        .map_err(|error| format!("parse tokenizer tokenizer_config.json: {error}"))?;
    merge_added_tokens(&mut tokenizer_config, tokenizer_json_bytes)?;
    parse_hf_model_semantic_metadata_from_config(
        model_config,
        tokenizer_config_bytes,
        &tokenizer_config,
        chat_template_jinja,
        chat_template_json,
    )
}

#[derive(Deserialize)]
struct TokenizerAddedTokens {
    #[serde(default)]
    added_tokens: Vec<TokenizerAddedToken>,
}

#[derive(Deserialize)]
struct TokenizerAddedToken {
    id: u32,
    content: String,
}

fn merge_added_tokens(tokenizer_config: &mut Value, tokenizer_json: &[u8]) -> Result<(), String> {
    // Ignore the model vocabulary while deserializing: only the small added-token
    // table is needed, and the immutable bundle already owns the original bytes.
    let added: TokenizerAddedTokens = serde_json::from_slice(tokenizer_json)
        .map_err(|error| format!("parse tokenizer.json added_tokens: {error}"))?;
    let config = tokenizer_config
        .as_object_mut()
        .ok_or("tokenizer_config.json root must be an object")?;
    let mut by_id = BTreeMap::<u32, String>::new();
    let mut by_content = BTreeMap::<String, u32>::new();
    let mut insert = |id: u32, content: &str| -> Result<(), String> {
        if let Some(previous) = by_id.get(&id) {
            if previous != content {
                return Err(format!(
                    "conflicting added token id {id}: {previous:?} and {content:?}"
                ));
            }
        }
        if let Some(previous) = by_content.get(content) {
            if *previous != id {
                return Err(format!(
                    "conflicting added token {content:?}: ids {previous} and {id}"
                ));
            }
        }
        by_id.insert(id, content.to_owned());
        by_content.insert(content.to_owned(), id);
        Ok(())
    };
    if let Some(decoder) = config
        .get("added_tokens_decoder")
        .filter(|value| !value.is_null())
    {
        let decoder = decoder
            .as_object()
            .ok_or("tokenizer_config.json added_tokens_decoder must be an object")?;
        for (id, token) in decoder {
            let id = id
                .parse::<u32>()
                .map_err(|error| format!("invalid added_tokens_decoder id {id:?}: {error}"))?;
            let content = token
                .get("content")
                .and_then(Value::as_str)
                .ok_or("added_tokens_decoder token must contain string content")?;
            insert(id, content)?;
        }
    }
    for token in added.added_tokens {
        insert(token.id, &token.content)?;
    }
    // This is a transient semantic lookup, never a replacement source artifact.
    // Template hashing below still uses the exact original selected file bytes.
    config.insert(
        "added_tokens_decoder".to_owned(),
        Value::Object(
            by_id
                .into_iter()
                .map(|(id, content)| (id.to_string(), serde_json::json!({"content": content})))
                .collect(),
        ),
    );
    Ok(())
}

fn parse_hf_model_semantic_metadata_from_config(
    model_config: &Value,
    tokenizer_config_bytes: &[u8],
    tokenizer_config: &Value,
    chat_template_jinja: Option<&[u8]>,
    chat_template_json: Option<&[u8]>,
) -> Result<CausalLanguageMetadata, String> {
    // Modern HF snapshots save a standalone template. Select it from the
    // immutable tokenizer source, with the same precedence as the product
    // renderer, and bind the digest to the file actually selected.
    let (template, source_file, source_bytes) = if let Some(bytes) = chat_template_jinja {
        let template = std::str::from_utf8(bytes)
            .map_err(|error| format!("chat_template.jinja is not UTF-8: {error}"))?;
        (template.to_owned(), "chat_template.jinja", bytes)
    } else if let Some(bytes) = chat_template_json {
        let value: Value = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse chat_template.json: {error}"))?;
        let template = value
            .as_str()
            .or_else(|| template_value(&value))
            .ok_or("chat_template.json missing chat_template")?;
        (template.to_owned(), "chat_template.json", bytes)
    } else {
        let template = template_value(tokenizer_config)
            .ok_or("tokenizer_config.json missing non-empty chat_template and no standalone template supplied")?;
        (
            template.to_owned(),
            "tokenizer_config.json",
            tokenizer_config_bytes,
        )
    };
    if template.trim().is_empty() {
        return Err(format!("{source_file} chat template must be non-empty"));
    }
    let special_tokens = parse_special_tokens(model_config, tokenizer_config)?;
    Ok(CausalLanguageMetadata {
        template: TemplateMetadata {
            template,
            source_file: source_file.to_owned(),
            sha256: format!("{:x}", Sha256::digest(source_bytes)),
        },
        special_tokens,
    })
}

fn template_value(value: &Value) -> Option<&str> {
    match value.get("chat_template")? {
        Value::String(template) => Some(template),
        Value::Array(items) => items
            .iter()
            .find(|item| item.get("name").and_then(Value::as_str) == Some("default"))
            .or_else(|| items.first())
            .and_then(|item| item.get("template").and_then(Value::as_str)),
        Value::Object(object) => object.get("template").and_then(Value::as_str),
        _ => None,
    }
}

pub(super) fn is_hf_template_source(source: &str) -> bool {
    matches!(
        source,
        "tokenizer_config.json" | "chat_template.jinja" | "chat_template.json"
    )
}

pub(super) fn parse_hf_model_semantic_metadata_with_external_template(
    model_config: &Value,
    tokenizer_config_bytes: &[u8],
    chat_template_bytes: &[u8],
    generation_config_bytes: &[u8],
) -> Result<CausalLanguageMetadata, String> {
    let tokenizer_config: Value = serde_json::from_slice(tokenizer_config_bytes)
        .map_err(|error| format!("parse tokenizer tokenizer_config.json: {error}"))?;
    let generation_config: Value = serde_json::from_slice(generation_config_bytes)
        .map_err(|error| format!("parse generation_config.json: {error}"))?;
    let generation_config = generation_config
        .as_object()
        .ok_or_else(|| "generation_config.json root must be an object".to_owned())?;
    let template = std::str::from_utf8(chat_template_bytes)
        .map_err(|error| format!("chat_template.jinja is not UTF-8: {error}"))?;
    if template.is_empty() {
        return Err("chat_template.jinja must be non-empty".to_owned());
    }
    let special_tokens =
        parse_special_tokens_with_generation(model_config, &tokenizer_config, generation_config)?;
    Ok(CausalLanguageMetadata {
        template: TemplateMetadata {
            template: template.to_owned(),
            source_file: "chat_template.jinja".to_owned(),
            sha256: format!("{:x}", Sha256::digest(chat_template_bytes)),
        },
        special_tokens,
    })
}

fn parse_special_tokens(
    model_config: &Value,
    tokenizer_config: &Value,
) -> Result<SpecialTokenMetadata, String> {
    let bos_token_id = token_id(model_config, tokenizer_config, "bos_token")?;
    let pad_token_id = token_id(model_config, tokenizer_config, "pad_token")?;
    let eos_value = tokenizer_config
        .get("eos_token")
        .or_else(|| tokenizer_config.get("eos_token_id"))
        .or_else(|| model_config.get("eos_token_id"))
        .or_else(|| {
            model_config
                .get("text_config")
                .and_then(|value| value.get("eos_token_id"))
        })
        .ok_or_else(|| "model/tokenizer metadata missing eos_token".to_owned())?;
    let eos_values = eos_value
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_else(|| std::slice::from_ref(eos_value));
    let eos_token_ids = eos_values
        .iter()
        .map(|value| resolve_token_id(value, tokenizer_config))
        .collect::<Result<BTreeSet<_>, _>>()?;
    if eos_token_ids.is_empty() {
        return Err("resolved EOS token set is empty".to_owned());
    }
    let collision_policy = collision_policy(bos_token_id, &eos_token_ids, pad_token_id)?;
    Ok(SpecialTokenMetadata {
        bos_token_id,
        eos_token_ids,
        pad_token_id,
        collision_policy,
    })
}

fn parse_special_tokens_with_generation(
    model_config: &Value,
    tokenizer_config: &Value,
    generation_config: &Map<String, Value>,
) -> Result<SpecialTokenMetadata, String> {
    let bos_token_id =
        match generation_token_id(generation_config, "bos_token_id", tokenizer_config)? {
            Some(token_id) => Some(token_id),
            None => token_id(model_config, tokenizer_config, "bos_token")?,
        };
    let pad_token_id =
        match generation_token_id(generation_config, "pad_token_id", tokenizer_config)? {
            Some(token_id) => Some(token_id),
            None => token_id(model_config, tokenizer_config, "pad_token")?,
        };
    let eos_value = generation_config
        .get("eos_token_id")
        .ok_or_else(|| "generation_config.json missing eos_token_id".to_owned())?;
    let eos_values = eos_value
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_else(|| std::slice::from_ref(eos_value));
    let eos_token_ids = eos_values
        .iter()
        .map(|value| resolve_token_id(value, tokenizer_config))
        .collect::<Result<BTreeSet<_>, _>>()?;
    if eos_token_ids.is_empty() {
        return Err("generation_config.json resolved EOS token set is empty".to_owned());
    }
    let collision_policy = collision_policy(bos_token_id, &eos_token_ids, pad_token_id)?;
    Ok(SpecialTokenMetadata {
        bos_token_id,
        eos_token_ids,
        pad_token_id,
        collision_policy,
    })
}

fn generation_token_id(
    generation_config: &Map<String, Value>,
    field: &str,
    tokenizer_config: &Value,
) -> Result<Option<u32>, String> {
    generation_config
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| resolve_token_id(value, tokenizer_config))
        .transpose()
}

fn token_id(
    model_config: &Value,
    tokenizer_config: &Value,
    name: &str,
) -> Result<Option<u32>, String> {
    let id_name = format!("{name}_id");
    tokenizer_config
        .get(name)
        .or_else(|| tokenizer_config.get(&id_name))
        .or_else(|| model_config.get(&id_name))
        .or_else(|| {
            model_config
                .get("text_config")
                .and_then(|value| value.get(&id_name))
        })
        .filter(|value| !value.is_null())
        .map(|value| resolve_token_id(value, tokenizer_config))
        .transpose()
}

fn resolve_token_id(value: &Value, tokenizer_config: &Value) -> Result<u32, String> {
    if let Some(id) = value.as_u64() {
        return u32::try_from(id).map_err(|_| format!("token id {id} exceeds u32"));
    }
    let content = value
        .as_str()
        .or_else(|| value.get("content").and_then(Value::as_str))
        .ok_or_else(|| format!("unsupported token metadata {value}"))?;
    tokenizer_config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .and_then(|tokens| {
            tokens.iter().find_map(|(id, metadata)| {
                (metadata.get("content").and_then(Value::as_str) == Some(content)).then_some(id)
            })
        })
        .ok_or_else(|| format!("token {content:?} has no added_tokens_decoder id"))?
        .parse::<u32>()
        .map_err(|error| format!("invalid token id for {content:?}: {error}"))
}

fn collision_policy(
    bos_token_id: Option<u32>,
    eos_token_ids: &BTreeSet<u32>,
    pad_token_id: Option<u32>,
) -> Result<SpecialTokenCollisionPolicy, String> {
    let mut allowed = BTreeSet::new();
    if let Some(bos) = bos_token_id {
        if eos_token_ids.contains(&bos) {
            allowed.insert(
                SpecialTokenCollision::new(SpecialTokenRole::Bos, SpecialTokenRole::Eos)
                    .map_err(|error| error.to_string())?,
            );
        }
        if pad_token_id == Some(bos) {
            allowed.insert(
                SpecialTokenCollision::new(SpecialTokenRole::Bos, SpecialTokenRole::Pad)
                    .map_err(|error| error.to_string())?,
            );
        }
    }
    if pad_token_id.is_some_and(|pad| eos_token_ids.contains(&pad)) {
        allowed.insert(
            SpecialTokenCollision::new(SpecialTokenRole::Eos, SpecialTokenRole::Pad)
                .map_err(|error| error.to_string())?,
        );
    }
    Ok(SpecialTokenCollisionPolicy::new(allowed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tokenizer_added_tokens_resolve_modern_named_roles_without_rewriting_source() {
        // Same schema as current HF processor packages; the IDs are arbitrary
        // sparse fixture IDs so resolution must use the supplied artifact.
        let config = br#"{
            "bos_token":null,"eos_token":"<|im_end|>",
            "pad_token":"<|endoftext|>","chat_template":"{{ messages }}"
        }"#;
        let tokenizer = br#"{
            "added_tokens":[
                {"id":137,"content":"<|endoftext|>","special":true},
                {"id":901,"content":"<|im_end|>","special":true}
            ],
            "model":{"type":"BPE","vocab":{"ordinary":0},"merges":[]}
        }"#;
        let metadata = parse_hf_model_semantic_metadata_with_tokenizer(
            &json!({"bos_token_id":1,"eos_token_id":2}),
            config,
            tokenizer,
            None,
            None,
        )
        .unwrap();
        assert_eq!(metadata.special_tokens.bos_token_id, None);
        assert_eq!(metadata.special_tokens.pad_token_id, Some(137));
        assert_eq!(metadata.special_tokens.eos_token_ids, BTreeSet::from([901]));
        assert_eq!(metadata.template.source_file, "tokenizer_config.json");
        assert_eq!(
            metadata.template.sha256,
            format!("{:x}", Sha256::digest(config))
        );

        let standalone = b"{{ messages[0].content }}\n";
        let selected = parse_hf_model_semantic_metadata_with_tokenizer(
            &json!({}),
            config,
            tokenizer,
            Some(standalone),
            None,
        )
        .unwrap();
        assert_eq!(selected.special_tokens, metadata.special_tokens);
        assert_eq!(selected.template.source_file, "chat_template.jinja");
        assert_eq!(
            selected.template.sha256,
            format!("{:x}", Sha256::digest(standalone))
        );
    }

    #[test]
    fn tokenizer_added_tokens_cross_check_legacy_decoder_and_numeric_roles() {
        let config = br#"{
            "eos_token":"end","pad_token_id":17,"chat_template":"template",
            "added_tokens_decoder":{"4":{"content":"end","special":true}}
        }"#;
        let metadata = parse_hf_model_semantic_metadata_with_tokenizer(
            &json!({}),
            config,
            br#"{"added_tokens":[{"id":4,"content":"end"}]}"#,
            None,
            None,
        )
        .unwrap();
        assert_eq!(metadata.special_tokens.eos_token_ids, BTreeSet::from([4]));
        // Numeric roles may refer to base vocabulary IDs, not only added tokens.
        assert_eq!(metadata.special_tokens.pad_token_id, Some(17));
        for tokenizer in [
            br#"{"added_tokens":[{"id":5,"content":"end"}]}"#.as_slice(),
            br#"{"added_tokens":[{"id":4,"content":"other"}]}"#,
            br#"{"added_tokens":[{"id":7,"content":"x"},{"id":8,"content":"x"}]}"#,
            br#"{"added_tokens":[{"id":7,"content":"x"},{"id":7,"content":"y"}]}"#,
        ] {
            let error = parse_hf_model_semantic_metadata_with_tokenizer(
                &json!({}),
                config,
                tokenizer,
                None,
                None,
            )
            .unwrap_err();
            assert!(error.contains("conflicting added token"), "{error}");
        }
    }

    #[test]
    fn tokenizer_added_tokens_reject_invalid_or_missing_id_sources() {
        let config = br#"{"eos_token":"end","chat_template":"template"}"#;
        for tokenizer in [
            br#"{"added_tokens":[{"id":4294967296,"content":"end"}]}"#.as_slice(),
            br#"{"added_tokens":[{"id":-1,"content":"end"}]}"#,
            br#"{"added_tokens":[{"id":2.5,"content":"end"}]}"#,
            br#"{"added_tokens":[{"content":"end"}]}"#,
            br#"{"added_tokens":[{"id":2,"content":null}]}"#,
            br#"{"added_tokens":{}}"#,
            br#"{}"#,
            br#"{"added_tokens":[{"id":2,"content":"different"}]}"#,
        ] {
            assert!(parse_hf_model_semantic_metadata_with_tokenizer(
                &json!({}),
                config,
                tokenizer,
                None,
                None,
            )
            .is_err());
        }
    }

    #[test]
    fn parses_numeric_and_named_special_tokens_into_one_typed_record() {
        let model = json!({"text_config": {"eos_token_id": [2, 3]}});
        let tokenizer = br#"{
            "chat_template": "{{ messages }}",
            "bos_token": {"content": "<bos>"},
            "pad_token_id": 0,
            "added_tokens_decoder": {"1": {"content": "<bos>"}}
        }"#;
        let metadata = parse_hf_model_semantic_metadata(&model, tokenizer, None, None).unwrap();
        assert_eq!(metadata.special_tokens.bos_token_id, Some(1));
        assert_eq!(
            metadata.special_tokens.eos_token_ids,
            BTreeSet::from([2, 3])
        );
        assert_eq!(metadata.special_tokens.pad_token_id, Some(0));
        assert_eq!(metadata.template.sha256.len(), 64);
    }

    #[test]
    fn accepts_tokenizer_eos_id_and_fails_closed_without_template() {
        let model = json!({});
        let tokenizer = br#"{
            "chat_template": "{{ messages }}",
            "eos_token_id": 2
        }"#;
        assert_eq!(
            parse_hf_model_semantic_metadata(&model, tokenizer, None, None)
                .unwrap()
                .special_tokens
                .eos_token_ids,
            BTreeSet::from([2])
        );
        assert!(
            parse_hf_model_semantic_metadata(&model, br#"{"eos_token_id":2}"#, None, None).is_err()
        );
    }

    #[test]
    fn standalone_template_retains_token_semantics_and_selected_file_digest() {
        let model = json!({"eos_token_id": 2});
        for embedded in [Value::Null, json!("legacy template")] {
            let tokenizer = serde_json::to_vec(&json!({
                "chat_template": embedded,
                "eos_token": "<end>",
                "pad_token_id": 0,
                "added_tokens_decoder": {"3":{"content":"<end>"}}
            }))
            .unwrap();
            let jinja = b"{{ messages[0].content }}<end>\n";
            let metadata = parse_hf_model_semantic_metadata(
                &model,
                &tokenizer,
                Some(jinja),
                Some(br#"{"chat_template":"older standalone"}"#),
            )
            .unwrap();
            assert_eq!(metadata.template.template.as_bytes(), jinja);
            assert_eq!(metadata.template.source_file, "chat_template.jinja");
            assert_eq!(
                metadata.template.sha256,
                format!("{:x}", Sha256::digest(jinja))
            );
            assert_eq!(metadata.special_tokens.eos_token_ids, BTreeSet::from([3]));
            assert_eq!(metadata.special_tokens.pad_token_id, Some(0));
        }
    }

    #[test]
    fn standalone_json_and_legacy_templates_share_default_selection() {
        let tokenizer = br#"{"eos_token_id":2,"chat_template":"embedded"}"#;
        for value in [
            json!("selected"),
            json!({"chat_template":"selected"}),
            json!({"chat_template":[{"name":"tool_use","template":"tools"},{"name":"default","template":"selected"}]}),
        ] {
            let bytes = serde_json::to_vec(&value).unwrap();
            let metadata =
                parse_hf_model_semantic_metadata(&json!({}), tokenizer, None, Some(&bytes))
                    .unwrap();
            assert_eq!(metadata.template.template, "selected");
            assert_eq!(metadata.template.source_file, "chat_template.json");
            assert_eq!(
                metadata.template.sha256,
                format!("{:x}", Sha256::digest(&bytes))
            );
        }
        let embedded = br#"{"eos_token_id":2,"chat_template":[{"name":"tool_use","template":"tools"},{"name":"default","template":"selected"}]}"#;
        let metadata = parse_hf_model_semantic_metadata(&json!({}), embedded, None, None).unwrap();
        assert_eq!(metadata.template.template, "selected");
        assert_eq!(metadata.template.source_file, "tokenizer_config.json");
    }

    #[test]
    fn invalid_declared_sidecar_never_silently_uses_another_template() {
        let tokenizer = br#"{"eos_token_id":2,"chat_template":"valid embedded"}"#;
        for bytes in [b"".as_slice(), b" \n", &[0xff]] {
            assert!(
                parse_hf_model_semantic_metadata(&json!({}), tokenizer, Some(bytes), None).is_err()
            );
        }
        for bytes in [b"{".as_slice(), br#"{"chat_template":null}"#, br#""  ""#] {
            assert!(
                parse_hf_model_semantic_metadata(&json!({}), tokenizer, None, Some(bytes)).is_err()
            );
        }
    }

    #[test]
    fn external_harmony_template_uses_generation_terminal_set() {
        let model = json!({"eos_token_id": 200002, "pad_token_id": 199999});
        let tokenizer = br#"{
            "chat_template": null,
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>"
        }"#;
        let generation = br#"{
            "bos_token_id": 199998,
            "eos_token_id": [200002, 199999, 200012],
            "pad_token_id": 199999
        }"#;
        let template = b"{{ messages }}<|start|>assistant<|channel|>final<|message|>";
        let metadata = parse_hf_model_semantic_metadata_with_external_template(
            &model, tokenizer, template, generation,
        )
        .unwrap();

        assert_eq!(metadata.template.source_file, "chat_template.jinja");
        assert_eq!(metadata.template.template.as_bytes(), template);
        assert_eq!(
            metadata.template.sha256,
            format!("{:x}", Sha256::digest(template))
        );
        assert_eq!(metadata.special_tokens.bos_token_id, Some(199998));
        assert_eq!(metadata.special_tokens.pad_token_id, Some(199999));
        assert_eq!(
            metadata.special_tokens.eos_token_ids,
            BTreeSet::from([199999, 200002, 200012])
        );
        assert!(metadata
            .special_tokens
            .collision_policy
            .allows(SpecialTokenRole::Eos, SpecialTokenRole::Pad));
    }

    #[test]
    fn external_template_metadata_fails_closed_without_generation_eos() {
        let model = json!({"eos_token_id": 200002});
        let error = parse_hf_model_semantic_metadata_with_external_template(
            &model,
            br#"{}"#,
            b"{{ messages }}",
            br#"{"bos_token_id":199998}"#,
        )
        .unwrap_err();
        assert!(error.contains("generation_config.json missing eos_token_id"));
    }
}
