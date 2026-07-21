//! Shared Hugging Face tokenizer metadata parser for production model packages.

use std::collections::BTreeSet;

use ferrum_interfaces::vnext::{
    ModelSemanticMetadata, SpecialTokenCollision, SpecialTokenCollisionPolicy,
    SpecialTokenMetadata, SpecialTokenRole, TemplateMetadata,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(super) fn parse_hf_model_semantic_metadata(
    model_config: &Value,
    tokenizer_config_bytes: &[u8],
) -> Result<ModelSemanticMetadata, String> {
    let tokenizer_config: Value = serde_json::from_slice(tokenizer_config_bytes)
        .map_err(|error| format!("parse tokenizer tokenizer_config.json: {error}"))?;
    let template = tokenizer_config
        .get("chat_template")
        .and_then(Value::as_str)
        .filter(|template| !template.is_empty())
        .ok_or_else(|| "tokenizer_config.json missing non-empty string chat_template".to_owned())?;
    let special_tokens = parse_special_tokens(model_config, &tokenizer_config)?;
    Ok(ModelSemanticMetadata {
        template: TemplateMetadata {
            template: template.to_owned(),
            source_file: "tokenizer_config.json".to_owned(),
            sha256: format!("{:x}", Sha256::digest(tokenizer_config_bytes)),
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
    fn parses_numeric_and_named_special_tokens_into_one_typed_record() {
        let model = json!({"text_config": {"eos_token_id": [2, 3]}});
        let tokenizer = br#"{
            "chat_template": "{{ messages }}",
            "bos_token": {"content": "<bos>"},
            "pad_token_id": 0,
            "added_tokens_decoder": {"1": {"content": "<bos>"}}
        }"#;
        let metadata = parse_hf_model_semantic_metadata(&model, tokenizer).unwrap();
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
            parse_hf_model_semantic_metadata(&model, tokenizer)
                .unwrap()
                .special_tokens
                .eos_token_ids,
            BTreeSet::from([2])
        );
        assert!(parse_hf_model_semantic_metadata(&model, br#"{"eos_token_id":2}"#).is_err());
    }
}
