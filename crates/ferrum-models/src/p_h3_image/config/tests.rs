use super::*;
use serde_json::{json, Value};

// Relevant wire fields from PrunaAI/P-H3-Image revision
// 7f98acef503cf2bcfba821a78cb1cf1a9928b2df; no weights are needed.
fn transformer() -> Value {
    json!({
        "_class_name":"MiniMaxH3Transformer3DModel",
        "_diffusers_version":"0.42.0.dev0",
        "attention_head_dim":128,"audio_in_channels":32,"ffn_dim":14336,
        "final_norm_eps":1e-5,"freq_dim":256,"hidden_size":5376,
        "in_channels":24,"norm_eps":1e-5,"num_attention_heads":56,
        "num_layers":50,"num_refiner_layers":2,"patch_size":[1,2,2],
        "qk_norm_eps":1e-5,"rope_freq_dim":16,"rope_theta":10000.0,
        "text_dim":5120,"time_embed_dim":2688,"time_embed_hidden_dim":5376
    })
}

fn connector() -> Value {
    json!({"_class_name":"PH3ImageTextConnector",
        "hidden_state_indices":[13,19,25,31,36],
        "text_hidden_size":2560,"out_channels":5120})
}

fn parse(value: &Value) -> Result<PH3ImageTransformerConfig> {
    PH3ImageTransformerConfig::from_hf_json(&serde_json::to_vec(value)?)
}

#[test]
fn released_transformer_keeps_residual_and_attention_widths_distinct() {
    let config = parse(&transformer()).unwrap();
    assert_eq!(config.hidden_size(), 5376);
    assert_eq!(config.attention_inner_dim(), 7168);
    assert_eq!(config.layers(), 50);
    assert_eq!(config.refiner_layers(), 2);
    assert_eq!(config.patch_input_dim(), 96);
    assert_eq!(config.audio_in_channels(), 32);
    assert_eq!(config.text_dim(), 5120);
    assert_eq!(config.rope_freq_dim() * 6, 96);
}

#[test]
fn dimensions_norms_rope_and_overflow_are_checked() {
    for (field, value) in [
        ("_class_name", json!("QwenImage21Transformer2DModel")),
        ("hidden_size", json!(0)),
        ("attention_head_dim", json!(95)),
        ("freq_dim", json!(255)),
        ("num_layers", json!(0)),
        ("in_channels", json!(-24)),
        ("patch_size", json!([2, 2, 2])),
        ("patch_size", json!([1, 0, 2])),
        ("norm_eps", json!(0)),
        ("qk_norm_eps", json!(-1e-5)),
        ("rope_theta", json!(0)),
        ("ffn_dim", json!(u64::MAX)),
        ("num_layers", json!(u64::MAX)),
        ("new_attention_rule", json!(true)),
    ] {
        let mut value_under_test = transformer();
        value_under_test[field] = value;
        assert!(parse(&value_under_test).is_err(), "{field}");
    }
}

#[test]
fn no_model_name_or_producer_version_gate_is_used() {
    let mut value = transformer();
    value["_diffusers_version"] = json!("future-producer");
    assert!(parse(&value).is_ok());
    value.as_object_mut().unwrap().remove("_diffusers_version");
    assert!(parse(&value).is_ok());
    // A smaller architecture with the same declared operations remains valid.
    value["hidden_size"] = json!(8);
    value["num_attention_heads"] = json!(1);
    value["attention_head_dim"] = json!(8);
    value["rope_freq_dim"] = json!(1);
    value["num_layers"] = json!(1);
    value["num_refiner_layers"] = json!(0);
    assert!(parse(&value).is_ok());
}

#[test]
fn original_object_bytes_reject_missing_duplicate_or_positional_fields() {
    let mut value = transformer();
    value.as_object_mut().unwrap().remove("patch_size");
    assert!(parse(&value).is_err());
    let original = serde_json::to_string(&transformer()).unwrap();
    let duplicate = format!("{{\"hidden_size\":8,{}", &original[1..]);
    assert!(PH3ImageTransformerConfig::from_hf_json(duplicate.as_bytes()).is_err());
    assert!(PH3ImageTransformerConfig::from_hf_json(b"[]").is_err());
}

#[test]
fn connector_taps_and_both_component_widths_are_checked() {
    let transformer = parse(&transformer()).unwrap();
    let config =
        PH3ImageTextConnectorConfig::from_hf_json(&serde_json::to_vec(&connector()).unwrap())
            .unwrap();
    assert_eq!(config.input_dim(), 12800);
    assert_eq!(config.hidden_state_indices(), &[13, 19, 25, 31, 36]);
    config.validate_components(36, 2560, &transformer).unwrap();
    assert!(config.validate_components(35, 2560, &transformer).is_err());
    assert!(config.validate_components(36, 4096, &transformer).is_err());
    let mut wrong_output = connector();
    wrong_output["out_channels"] = json!(4096);
    let wrong_output =
        PH3ImageTextConnectorConfig::from_hf_json(&serde_json::to_vec(&wrong_output).unwrap())
            .unwrap();
    assert!(wrong_output
        .validate_components(36, 2560, &transformer)
        .is_err());
}

#[test]
fn connector_rejects_unsupported_class_empty_taps_and_overflow() {
    for (field, value) in [
        ("_class_name", json!("Identity")),
        ("hidden_state_indices", json!([])),
        ("hidden_state_indices", json!([-1])),
        ("text_hidden_size", json!(0)),
        ("out_channels", json!(u64::MAX)),
        ("new_normalization_rule", json!(true)),
    ] {
        let mut config = connector();
        config[field] = value;
        assert!(
            PH3ImageTextConnectorConfig::from_hf_json(&serde_json::to_vec(&config).unwrap())
                .is_err(),
            "{field}"
        );
    }
}
