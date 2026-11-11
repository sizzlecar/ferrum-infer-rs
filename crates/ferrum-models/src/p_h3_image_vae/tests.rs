use super::*;
use ferrum_runtime::ElementType;
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// Decoder-only architecture fixture from the declared HF schema. No encoder,
/// image payload, paths or irrelevant model card metadata enter this mapping.
struct DecoderFixture {
    layers: u64,
    heads: u64,
    head_dim: u64,
    latent_channels: u64,
}
impl DecoderFixture {
    fn released() -> Self {
        Self {
            layers: 36,
            heads: 32,
            head_dim: 64,
            latent_channels: 24,
        }
    }
    fn tiny() -> Self {
        Self {
            layers: 2,
            heads: 2,
            head_dim: 16,
            latent_channels: 4,
        }
    }
    fn wire(&self) -> Value {
        json!({
            "_class_name": PH3_IMAGE_VAE_CLASS,
            "latent_channels": self.latent_channels,
            "out_channels": 3,
            "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
            "decoder_num_layers": self.layers,
            "decoder_num_attention_heads": self.heads,
            "decoder_attention_head_dim": self.head_dim,
            "decoder_num_register_tokens": 4,
            "decoder_ffn_mult": 4,
            "decoder_rope_dim_ratio": 0.75,
            "decoder_rope_theta": 100.0,
            "decoder_norm_eps": 0.00001
        })
    }
    fn config(&self) -> PH3ImageVaeDecoderConfig {
        PH3ImageVaeDecoderConfig::from_hf_json(&serde_json::to_vec(&self.wire()).unwrap()).unwrap()
    }
}

#[test]
fn declared_architecture_keeps_rotary_prefix_patch_and_register_suffix() {
    let config = DecoderFixture::released().config();
    assert_eq!(config.hidden_size().unwrap(), 2048);
    assert_eq!(config.intermediate_size().unwrap(), 8192);
    assert_eq!(config.latent_channels(), 24);
    assert_eq!(config.patch_size(), 16);
    assert_eq!(config.patch_output_channels().unwrap(), 768);
    assert_eq!(config.rotary_dim(), 48);
    assert_eq!(config.rope_theta().numerator(), 100);
    assert_eq!(config.rope_theta().denominator(), 1);
    assert_eq!(config.sequence_length(16, 16).unwrap(), 261);
    assert_eq!(config.norm_epsilon().numerator(), 1);
    assert_eq!(config.norm_epsilon().denominator(), 100000);
    let wire = serde_json::to_value(&config).unwrap();
    assert_eq!(
        serde_json::from_value::<PH3ImageVaeDecoderConfig>(wire).unwrap(),
        config
    );
}

#[test]
fn real_comfy_header_shapes_preserve_fused_qkv_ffn_and_quantization_components() {
    // Typed entries transcribed from the official INT8 file's safetensors
    // header, revision 44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543.
    let config = DecoderFixture::released().config();
    let weights = config.expected_comfy_weights().unwrap();
    let find = |name: &str| weights.iter().find(|w| w.external_name() == name).unwrap();
    for (name, shape) in [
        (
            "decoder.transformer_blocks.0.attn.to_qkv.weight",
            [6144, 2048],
        ),
        (
            "decoder.transformer_blocks.0.attn.to_out.weight",
            [2048, 2048],
        ),
        ("decoder.transformer_blocks.35.ff.w1.weight", [16384, 2048]),
        ("decoder.transformer_blocks.35.ff.w2.weight", [2048, 8192]),
    ] {
        let weight = find(name);
        assert_eq!(weight.dimensions(), shape);
        assert_eq!(weight.encoding(), PH3ImageVaeWeightEncoding::ConvRotInt8);
        assert_eq!(weight.source_element_type(), ElementType::I8);
        assert_eq!(weight.value_bytes().unwrap(), shape[0] * shape[1]);
        assert_eq!(
            weight.scale_external_name().unwrap(),
            format!("{name}_scale")
        );
        let scale = weight.scale_tensor().unwrap().unwrap();
        assert_eq!(scale.dimensions(), [shape[0], 1]);
        assert_eq!(scale.element_type(), ElementType::F32);
        assert_eq!(
            weight.quantization_external_name().unwrap(),
            format!("{}.comfy_quant", name.strip_suffix(".weight").unwrap())
        );
    }
    for (name, shape) in [
        ("post_quant_conv.weight", vec![24, 24, 1, 1, 1]),
        ("decoder.x_embedder.weight", vec![2048, 24]),
        ("decoder.proj_out.weight", vec![768, 2048]),
        ("decoder.register_tokens", vec![1, 4, 2048]),
        ("decoder.transformer_blocks.0.attn.to_qkv.bias", vec![6144]),
        ("decoder.transformer_blocks.0.ff.w1.bias", vec![16384]),
    ] {
        let weight = find(name);
        assert_eq!(weight.dimensions(), shape);
        assert_eq!(weight.encoding(), PH3ImageVaeWeightEncoding::Bf16);
        assert_eq!(weight.source_element_type(), ElementType::Bf16);
        assert!(weight.scale_external_name().is_none());
        assert!(weight.quantization_external_name().is_none());
    }
    let names = weights
        .iter()
        .map(|w| w.external_name())
        .collect::<BTreeSet<_>>();
    assert_eq!(names.len(), weights.len());
    assert!(names
        .iter()
        .all(|name| name.starts_with("decoder.") || name.starts_with("post_quant_conv.")));
}

#[test]
fn tiny_architecture_derives_mapping_instead_of_selecting_released_model_dimensions() {
    let config = DecoderFixture::tiny().config();
    let weights = config.expected_comfy_weights().unwrap();
    assert_eq!(config.hidden_size().unwrap(), 32);
    let qkv = weights
        .iter()
        .find(|w| w.external_name() == "decoder.transformer_blocks.1.attn.to_qkv.weight")
        .unwrap();
    assert_eq!(qkv.dimensions(), [96, 32]);
    assert!(!weights.iter().any(|w| w
        .external_name()
        .starts_with("decoder.transformer_blocks.2.")));
}

#[test]
fn invalid_architecture_wire_and_capacity_extents_fail_before_inventory_allocation() {
    for (field, value) in [
        ("_class_name", json!("AutoencoderKLQwenImage21")),
        ("latent_channels", json!(0)),
        ("decoder_num_layers", json!(0)),
        ("decoder_num_attention_heads", json!(u64::MAX)),
        ("decoder_num_register_tokens", json!(0)),
        ("decoder_num_register_tokens", json!(u64::MAX)),
        ("decoder_ffn_mult", json!(u64::MAX)),
        ("decoder_rope_dim_ratio", json!(0.7)),
        ("decoder_rope_dim_ratio", json!(1.25)),
        ("decoder_rope_theta", json!(0)),
        ("decoder_norm_eps", json!(0)),
        ("decoder_norm_eps", json!(-0.1)),
        ("decoder_norm_eps", json!(1.1)),
        ("spatial_downsample_factors", json!([])),
        ("spatial_downsample_factors", json!([2, 0])),
        ("spatial_downsample_factors", json!([u64::MAX, 2])),
    ] {
        let mut wire = DecoderFixture::tiny().wire();
        wire[field] = value;
        assert!(
            PH3ImageVaeDecoderConfig::from_hf_json(&serde_json::to_vec(&wire).unwrap()).is_err(),
            "accepted {field}: {wire}"
        );
    }
    let mut canonical = serde_json::to_value(DecoderFixture::tiny().config()).unwrap();
    canonical["rotary_dim"] = json!(7);
    assert!(serde_json::from_value::<PH3ImageVaeDecoderConfig>(canonical).is_err());
    let wire = serde_json::to_string(&DecoderFixture::tiny().wire()).unwrap();
    let duplicate = format!("{{\"decoder_num_layers\":1,{}", &wire[1..]);
    assert!(PH3ImageVaeDecoderConfig::from_hf_json(duplicate.as_bytes()).is_err());
    let config = DecoderFixture::tiny().config();
    assert!(config.sequence_length(0, 1).is_err());
    assert!(config.sequence_length(u64::MAX, 2).is_err());
}
