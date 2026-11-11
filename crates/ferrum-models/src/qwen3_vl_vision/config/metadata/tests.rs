use super::super::Qwen3VLVisionArchitecture;
use super::*;
use crate::qwen3_vl_vision::{Qwen3VLVisionConfig, Qwen3VLVisionFamily, Qwen3VLVisionGrid};
use ferrum_runtime::{ElementType, ModelFamilyProvider, WeightEncoding};
use serde_json::json;

// Architecture values from the released Qwen3-VL-4B metadata. No model weights
// or external fixture path is needed to verify the wire and physical shapes.
fn released() -> Value {
    json!({
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "vision_config": {
            "model_type": "qwen3_vl_vision",
            "dtype": "bfloat16",
            "hidden_act": "gelu_pytorch_tanh",
            "depth": 24,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2560,
            "deepstack_visual_indexes": [5, 11, 17],
            "in_channels": 3,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "rope_parameters": {"rope_type": "axial", "rope_theta": 10000.0}
        }
    })
}

#[test]
fn released_vision_wire_preserves_real_4b_shapes_and_bf16_profile() {
    let raw = released();
    let a = Qwen3VLVisionArchitecture::from_hf_value(&raw).unwrap();
    assert_eq!(
        (
            a.depth(),
            a.hidden_size(),
            a.intermediate_size(),
            a.heads(),
            a.output_size()
        ),
        (24, 1024, 4096, 16, 2560)
    );
    assert_eq!(a.deepstack_layers(), [5, 11, 17]);
    let config =
        Qwen3VLVisionConfig::new(a, vec![Qwen3VLVisionGrid::new(1, 2, 2).unwrap()]).unwrap();
    let family = Qwen3VLVisionFamily::new().unwrap();
    let weights = family.weights(&config).unwrap();
    let schema = weights.stored_schema().unwrap();
    assert_eq!(schema.components.len(), 314); // Original learned table is an input adapter source.
    let shape = |name| {
        &schema
            .components
            .iter()
            .find(|c| c.external_names == [name])
            .unwrap()
            .dimensions
    };
    assert_eq!(
        shape("model.visual.patch_embed.proj.weight"),
        &[1024, 3, 2, 16, 16]
    );
    assert_eq!(
        shape("model.visual.blocks.23.attn.qkv.weight"),
        &[3072, 1024]
    );
    assert_eq!(
        shape("model.visual.deepstack_merger_list.2.linear_fc2.weight"),
        &[2560, 4096]
    );
    assert!(schema.components.iter().all(|c| c.encoding
        == WeightEncoding::Dense {
            element_type: ElementType::Bf16
        }));
}

#[test]
fn legacy_implicit_and_current_explicit_axial_declarations_normalize_identically() {
    let current = released();
    let expected = Qwen3VLVisionArchitecture::from_hf_value(&current).unwrap();
    let mut legacy = current.clone();
    legacy["vision_config"]["model_type"] = json!("qwen3_vl");
    legacy["vision_config"]
        .as_object_mut()
        .unwrap()
        .remove("rope_parameters");
    assert_eq!(
        Qwen3VLVisionArchitecture::from_hf_value(&legacy).unwrap(),
        expected
    );
    legacy["vision_config"]["rope_parameters"] = json!({"rope_type": "axial", "rope_theta": 10000});
    assert_eq!(
        Qwen3VLVisionArchitecture::from_hf_value(&legacy).unwrap(),
        expected
    );
}

#[test]
fn unsupported_rotary_rules_and_conflicting_overrides_fail_closed() {
    for parameters in [
        Value::Null,
        json!({}),
        json!({"rope_type": "default", "rope_theta": 10000}),
        json!({"rope_type": "axial", "rope_theta": 10001}),
        json!({"rope_type": "axial", "rope_theta": -10000}),
        json!({"rope_type": "axial", "rope_theta": "10000"}),
        json!({"rope_type": "axial"}),
        json!({"rope_type": "axial", "rope_theta": 10000, "factor": 1}),
    ] {
        let mut raw = released();
        raw["vision_config"]["rope_parameters"] = parameters;
        assert!(Qwen3VLVisionArchitecture::from_hf_value(&raw).is_err());
    }
    for (key, value) in [
        ("rope_theta", json!(10000)),
        ("layer_norm_eps", json!(1e-6)),
        ("model_type", json!("qwen2_vl")),
        ("dtype", json!("float32")),
    ] {
        let mut raw = released();
        raw["vision_config"][key] = value;
        assert!(Qwen3VLVisionArchitecture::from_hf_value(&raw).is_err());
    }
}

fn released_qwen35_upsampler() -> Value {
    let mut raw = released();
    raw["architectures"] = json!(["Qwen3_5ForConditionalGeneration"]);
    raw["model_type"] = json!("qwen3_5");
    let vision = &mut raw["vision_config"];
    vision["model_type"] = json!("qwen3_5_vision");
    vision["dtype"] = json!("float32");
    vision["depth"] = json!(27);
    vision["hidden_size"] = json!(1152);
    vision["intermediate_size"] = json!(4304);
    vision["out_hidden_size"] = json!(4096);
    vision["deepstack_visual_indexes"] = json!([]);
    raw
}

#[test]
fn qwen35_upsampler_vision_declares_main_only_without_changing_stored_bf16_precision() {
    let raw = released_qwen35_upsampler();
    assert!(Qwen3VLVisionArchitecture::from_hf_value(&raw).is_err());
    let a = Qwen3VLVisionArchitecture::from_qwen35_hf_value(&raw).unwrap();
    assert!(a.deepstack_layers().is_empty());
    assert_eq!(
        (
            a.depth(),
            a.hidden_size(),
            a.intermediate_size(),
            a.output_size()
        ),
        (27, 1152, 4304, 4096)
    );
    let c = Qwen3VLVisionConfig::new(a, vec![Qwen3VLVisionGrid::new(1, 2, 2).unwrap()]).unwrap();
    let f = Qwen3VLVisionFamily::new().unwrap();
    let weights = f.weights(&c).unwrap();
    let schema = weights.stored_schema().unwrap();
    assert!(schema.components.iter().all(|w| w
        .external_names
        .iter()
        .all(|name| !name.contains("deepstack"))));
    let main = schema
        .components
        .iter()
        .find(|w| w.external_names == ["model.visual.merger.linear_fc2.weight"])
        .unwrap();
    assert_eq!(main.dimensions, [4096, 4608]);
    assert!(schema.components.iter().all(|w| w.encoding
        == WeightEncoding::Dense {
            element_type: ElementType::Bf16
        }));
    let profiles = f.numerical_profiles(&c).unwrap();
    let profile = profiles
        .resolve(
            &ferrum_runtime::NumericalProfileId::new(Qwen3VLVisionFamily::REFERENCE_PROFILE_ID)
                .unwrap(),
        )
        .unwrap();
    let graph = f.semantic_program(&c, profile).unwrap();
    assert_eq!(
        graph.outputs(),
        &[ferrum_runtime::ProgramValueId::new(Qwen3VLVisionFamily::OUTPUT_ID).unwrap()]
    );
    assert_eq!(
        serde_json::from_value::<Qwen3VLVisionConfig>(serde_json::to_value(&c).unwrap()).unwrap(),
        c
    );
}

#[test]
fn vision_metadata_cannot_exchange_topologies_or_silently_override_rotary_semantics() {
    for stages in [
        json!([0]),
        json!([0, 1]),
        json!([0, 1, 2]),
        json!([0, 1, 2, 3]),
        Value::Null,
    ] {
        let mut raw = released_qwen35_upsampler();
        raw["vision_config"]["deepstack_visual_indexes"] = stages;
        assert!(Qwen3VLVisionArchitecture::from_qwen35_hf_value(&raw).is_err());
    }
    let mut raw = released();
    raw["vision_config"]["deepstack_visual_indexes"] = json!([]);
    assert!(Qwen3VLVisionArchitecture::from_hf_value(&raw).is_err());
    assert!(Qwen3VLVisionArchitecture::from_qwen35_hf_value(&released()).is_err());
    for (key, value) in [
        ("dtype", json!("float16")),
        ("hidden_act", json!("gelu")),
        (
            "rope_parameters",
            json!({"rope_type": "axial", "rope_theta": 10001}),
        ),
        ("rope_theta", json!(10000)),
        ("layer_norm_eps", json!(1e-6)),
    ] {
        let mut raw = released_qwen35_upsampler();
        raw["vision_config"][key] = value;
        assert!(Qwen3VLVisionArchitecture::from_qwen35_hf_value(&raw).is_err());
    }
}
