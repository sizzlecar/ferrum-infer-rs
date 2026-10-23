use super::*;
use serde_json::json;

fn hf(real: bool) -> Value {
    let (depth, h, m, heads, positions, out, indexes) = if real {
        (27, 1152, 4304, 16, 2304, 4096, [8, 16, 24])
    } else {
        (3, 8, 12, 2, 16, 10, [0, 1, 2])
    };
    json!({"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl","vision_config":{"model_type":"qwen3_vl","dtype":"bfloat16","hidden_act":"gelu_pytorch_tanh","depth":depth,"hidden_size":h,"intermediate_size":m,"num_heads":heads,"num_position_embeddings":positions,"out_hidden_size":out,"deepstack_visual_indexes":indexes,"in_channels":3,"patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2}})
}
fn config(real: bool) -> Qwen3VLVisionConfig {
    Qwen3VLVisionConfig::new(
        Qwen3VLVisionArchitecture::from_hf_value(&hf(real)).unwrap(),
        vec![
            Qwen3VLVisionGrid::new(1, 2, 4).unwrap(),
            Qwen3VLVisionGrid::new(1, 4, 4).unwrap(),
        ],
    )
    .unwrap()
}
#[test]
fn actual_schema_preserves_patch_rank_and_merger_normalization_shapes() {
    let c = config(true);
    let family = Qwen3VLVisionFamily::new().unwrap();
    let stored = family.weights(&c).unwrap();
    let schema = stored.stored_schema().unwrap();
    let shape = |name: &str| {
        schema
            .components
            .iter()
            .find(|w| w.external_names == [name])
            .unwrap()
            .dimensions
            .clone()
    };
    assert_eq!(
        shape("model.visual.patch_embed.proj.weight"),
        [1152, 3, 2, 16, 16]
    );
    assert_eq!(
        shape("model.visual.blocks.26.attn.qkv.weight"),
        [3456, 1152]
    );
    assert_eq!(
        shape("model.visual.blocks.26.mlp.linear_fc1.weight"),
        [4304, 1152]
    );
    assert_eq!(shape("model.visual.merger.norm.weight"), [1152]);
    for stage in 0..3 {
        assert_eq!(
            shape(&format!(
                "model.visual.deepstack_merger_list.{stage}.norm.weight"
            )),
            [4608]
        );
    }
    // The separately consumed learned table is a real BF16 [2304,1152] payload.
    assert_eq!(
        schema
            .components
            .iter()
            .map(|w| w.physical_bytes().unwrap())
            .sum::<u64>()
            + 2304 * 1152 * 2,
        1_152_776_672
    );
    assert!(schema.components.iter().all(|w| w.encoding
        == WeightEncoding::Dense {
            element_type: ElementType::Bf16
        }));
    assert!(!schema.components.iter().any(|w| w
        .external_names
        .iter()
        .any(|n| n == Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME)));
}
#[test]
fn graph_preserves_split_rotary_image_isolation_and_merger_order() {
    let c = config(false);
    let family = Qwen3VLVisionFamily::new().unwrap();
    let profiles = family.numerical_profiles(&c).unwrap();
    let p = family
        .semantic_program(&c, &profiles.profiles()[0])
        .unwrap();
    let nodes = &p.blocks()[0].nodes;
    let find = |suffix: &str| {
        nodes
            .iter()
            .find(|n| n.id.as_str() == format!("node.qwen3_vl.vision.{suffix}"))
            .unwrap()
    };
    let qkv = find("layer0.qkv");
    for (i, name) in ["q", "k", "v"].into_iter().enumerate() {
        let n = find(&format!("layer0.{name}"));
        assert_eq!(n.inputs, [qkv.outputs[0].clone()]);
        assert_eq!(
            n.attributes[&AttributeId::new("column_start").unwrap()],
            SemanticValue::Unsigned(i as u64 * 8)
        );
    }
    let attn = find("layer0.attention");
    assert_eq!(attn.operation_id.as_str(), program::ATTENTION);
    assert_eq!(
        attn.attributes[&AttributeId::new("segment_lengths").unwrap()],
        SemanticValue::Integers(vec![8, 16])
    );
    assert_eq!(
        attn.inputs,
        [
            find("layer0.q_rotary").outputs[0].clone(),
            find("layer0.k_rotary").outputs[0].clone(),
            find("layer0.v").outputs[0].clone()
        ]
    );
    assert_eq!(
        find("merger.shuffle").inputs,
        [find("merger.norm").outputs[0].clone()]
    );
    for i in 0..3 {
        assert_eq!(
            find(&format!("deepstack_merger{i}.norm")).inputs[0],
            find(&format!("deepstack_merger{i}.shuffle")).outputs[0]
        );
        assert_eq!(
            find(&format!("deepstack_merger{i}.shuffle")).inputs[0],
            find(&format!("layer{i}.output")).outputs[0]
        );
        assert_eq!(
            find(&format!("deepstack_merger{i}.gelu"))
                .operation_id
                .as_str(),
            program::GELU
        );
    }
    assert_eq!(
        find("layer0.gelu").operation_id.as_str(),
        GELU_TANH_BF16_OPERATION_ID
    );
    let specs = c.compile_options().unwrap();
    for output in p.outputs() {
        assert_eq!(specs.tensor_specs()[output].dimensions, [6, 10]);
    }
    assert_eq!(
        p.input_tensor_specs()[&ProgramValueId::new(Qwen3VLVisionFamily::COSINE_ID).unwrap()]
            .element_type,
        ElementType::F32
    );
}
#[test]
fn grids_are_identity_and_invalid_shapes_fail_before_preparation() {
    let c = config(false);
    let reg = TypedFamilyRegistration::new(Qwen3VLVisionFamily::new().unwrap());
    let profile = NumericalProfileId::new(Qwen3VLVisionFamily::REFERENCE_PROFILE_ID).unwrap();
    let a = reg
        .prepare_with_profile(&serde_json::to_value(&c).unwrap(), &profile)
        .unwrap();
    let reversed = Qwen3VLVisionConfig::new(
        c.architecture().clone(),
        c.grids().iter().rev().cloned().collect(),
    )
    .unwrap();
    let b = reg
        .prepare_with_profile(&serde_json::to_value(&reversed).unwrap(), &profile)
        .unwrap();
    assert_ne!(a.fingerprint().unwrap(), b.fingerprint().unwrap());
    assert_eq!(a.weights(), b.weights());
    let mut wire = serde_json::to_value(&c).unwrap();
    wire["grids"][0]["temporal"] = json!(2);
    assert!(serde_json::from_value::<Qwen3VLVisionConfig>(wire).is_err());
    assert!(Qwen3VLVisionConfig::new(
        c.architecture().clone(),
        vec![Qwen3VLVisionGrid::new(1, 3, 4).unwrap()]
    )
    .is_err());
    assert!(Qwen3VLVisionGrid::new(1, u64::MAX, 2).is_err());
    for (field, bad) in [
        ("num_heads", json!(3)),
        ("deepstack_visual_indexes", json!([0, 0, 2])),
        ("num_position_embeddings", json!(15)),
        ("hidden_act", json!("gelu")),
    ] {
        let mut raw = hf(false);
        raw["vision_config"][field] = bad;
        assert!(Qwen3VLVisionArchitecture::from_hf_value(&raw).is_err());
    }
}

#[test]
fn governed_position_tables_use_merge_block_order_and_release_all_bytes() {
    let architecture = Qwen3VLVisionArchitecture::from_hf_value(&hf(false)).unwrap();
    let c = Qwen3VLVisionConfig::new(architecture, vec![Qwen3VLVisionGrid::new(1, 4, 4).unwrap()])
        .unwrap();
    let account = |n| {
        HostMemoryAccount::new(
            HostMemoryLimits::new(
                n,
                HostMemoryClassLimits {
                    weight_backing_bytes: 0,
                    input_preprocess_bytes: n,
                    transfer_staging_bytes: 0,
                    output_bytes: 0,
                },
            )
            .unwrap(),
        )
    };
    let table = (0..16)
        .flat_map(|p| {
            std::iter::repeat_n(half::bf16::from_f32(p as f32).to_bits().to_le_bytes(), 8).flatten()
        })
        .collect::<Vec<_>>();
    // 16 rows: BF16 positions[8] + two FP32 rotary tables[4].
    let bytes = 16 * (8 * 2 + 2 * 4 * 4);
    let host = account(bytes);
    let output = c.position_inputs_bytes(&host, &table).unwrap();
    let expected = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
    for (row, value) in expected.into_iter().enumerate() {
        let offset = row * 16;
        assert_eq!(
            &output[0].bytes().as_ref()[offset..offset + 2],
            &half::bf16::from_f32(value as f32).to_bits().to_le_bytes()
        );
    }
    let f32_at = |index: usize| {
        f32::from_le_bytes(
            output[1].bytes().as_ref()[index * 4..index * 4 + 4]
                .try_into()
                .unwrap(),
        )
    };
    assert_eq!(f32_at(0), 1.0);
    assert_eq!(f32_at(1), 1.0);
    assert_eq!(f32_at(4), 1.0);
    assert_eq!(f32_at(5), libm::cosf(1.0));
    assert_eq!(f32_at(8), libm::cosf(1.0));
    assert_eq!(f32_at(9), 1.0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), bytes);
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let short = account(bytes - 1);
    assert!(c.position_inputs_bytes(&short, &table).is_err());
    assert_eq!(short.usage().unwrap().committed_bytes(), 0);
    let mut invalid = table;
    invalid[0..2].copy_from_slice(&0x7fc1_u16.to_le_bytes());
    assert!(c.position_inputs_bytes(&host, &invalid).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
