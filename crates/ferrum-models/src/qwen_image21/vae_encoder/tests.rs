use super::*;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_quantization::SafetensorsArchive;
use half::bf16;
use serde_json::json;
#[path = "../../../examples/support/tensor_component.rs"]
mod shared;

fn id<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> T {
    T::try_from(s.into()).unwrap()
}
fn raw() -> Value {
    json!({"_class_name":"AutoencoderKLQwenImage21","z_dim":64,"base_dim":96,
        "decoder_base_dim":144,"dim_mult":[1,2,4,8,8],"num_res_blocks":2,
        "temperal_downsample":[false,true,true,true],"in_channels":4,
        "is_residual":true,"patch_size":null,"dropout":0.0,"attn_scales":[]})
}
fn real() -> QwenImage21VaeEncoderConfig {
    QwenImage21VaeEncoderConfig::from_hf_json(&serde_json::to_vec(&raw()).unwrap(), 1, 64, 96)
        .unwrap()
}
#[test]
fn encoder_matches_real_single_frame_schema_and_posterior_mean_boundary() {
    let c = real();
    assert_eq!(c.encoder_channels().unwrap(), [96, 96, 192, 384, 768, 768]);
    assert_eq!(c.input_tensor().unwrap().dimensions(), [1, 4, 64, 96]);
    assert_eq!(c.output_tensor().unwrap().dimensions(), [1, 64, 4, 6]);
    let weights = c.expected_weights().unwrap();
    assert_eq!(weights.len(), 98);
    assert_eq!(weights.iter().map(|w| w.bytes()).sum::<u64>(), 155_833_408);
    let by_name = weights
        .iter()
        .map(|w| (w.external_name(), w.dimensions()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(by_name["encoder.conv_in.weight"], [96, 4, 3, 3]);
    assert_eq!(
        by_name["encoder.mid_block.attentions.0.norm.gamma"],
        [768, 1, 1]
    );
    assert_eq!(by_name["encoder.conv_out.weight"], [128, 768, 3, 3]);
    assert_eq!(by_name["quant_conv.weight"], [128, 128, 1, 1]);
    assert!(weights
        .iter()
        .all(|w| !w.external_name().contains("time_conv")
            && !w.external_name().starts_with("decoder.")
            && !w.external_name().starts_with("post_quant_conv.")));
    let family = QwenImage21VaeEncoderFamily::new().unwrap();
    let profiles = family.numerical_profiles(&c).unwrap();
    let program = family
        .semantic_program(&c, &profiles.profiles()[0])
        .unwrap();
    assert_eq!(program.weights().len(), 98);
    assert_eq!(
        program.outputs(),
        [id(QwenImage21VaeEncoderFamily::OUTPUT_ID)]
    );
    assert_eq!(
        program.input_tensor_specs(),
        &c.input_tensor_specs().unwrap()
    );
    let schema = family.weights(&c).unwrap();
    assert!(schema
        .stored_schema()
        .unwrap()
        .tensors
        .iter()
        .all(|w| w.required));
}
#[test]
fn encoder_preserves_padding_original_shortcuts_and_last_outer_residual() {
    let graph = graph::build(&real()).unwrap();
    let node = |name: &str| {
        graph
            .nodes
            .iter()
            .find(|n| n.id.as_str() == format!("node.qwen_image21.vae.encoder.{name}"))
            .unwrap()
    };
    for i in 0..5 {
        let shortcut = node(&format!("encoder.down_blocks.{i}.first_frame_shortcut"));
        let first = node(&format!("encoder.down_blocks.{i}.resnets.0.norm1"));
        assert_eq!(shortcut.inputs[0], first.inputs[0]);
        assert_eq!(
            shortcut.attributes[&id::<AttributeId>("frame_factor")],
            SemanticValue::Unsigned(if (1..4).contains(&i) { 2 } else { 1 })
        );
        assert_eq!(
            shortcut.attributes[&id::<AttributeId>("scale_h")],
            SemanticValue::Unsigned(if i < 4 { 2 } else { 1 })
        );
        assert_eq!(
            node(&format!("encoder.down_blocks.{i}.residual")).inputs[1],
            shortcut.outputs[0]
        );
        if i < 4 {
            let down = node(&format!("encoder.down_blocks.{i}.downsampler.resample.1"));
            assert_eq!(
                down.attributes[&id::<AttributeId>("padding")],
                SemanticValue::Integers(vec![0, 1, 0, 1])
            );
            assert_eq!(
                down.attributes[&id::<AttributeId>("stride")],
                SemanticValue::Integers(vec![2, 2])
            );
            assert_eq!(
                down.attributes[&id::<AttributeId>("padding_mode")],
                SemanticValue::Text("explicit_zero".into())
            );
        }
    }
    let mean = node("posterior_mean");
    assert_eq!(
        mean.operation_id.as_str(),
        SPATIAL_UPSAMPLE_GATHER_BF16_OPERATION_ID
    );
    assert_eq!(mean.inputs, node("quant_conv").outputs);
    assert_eq!(
        mean.attributes[&id::<AttributeId>("channel_offsets")],
        SemanticValue::Integers(vec![0])
    );
}
#[test]
fn encoder_checked_config_rejects_wrong_pixel_geometry_and_unsupported_architecture() {
    let c = real();
    let mut wire = serde_json::to_value(&c).unwrap();
    assert_eq!(
        serde_json::from_value::<QwenImage21VaeEncoderConfig>(wire.clone()).unwrap(),
        c
    );
    wire["frames"] = json!(2);
    assert!(serde_json::from_value::<QwenImage21VaeEncoderConfig>(wire).is_err());
    for (key, value) in [
        ("in_channels", json!(3)),
        ("patch_size", json!(2)),
        ("is_residual", json!(false)),
        ("temperal_downsample", json!([false])),
        ("base_dim", json!(0)),
    ] {
        let mut changed = raw();
        changed[key] = value;
        assert!(QwenImage21VaeEncoderConfig::from_hf_json(
            &serde_json::to_vec(&changed).unwrap(),
            1,
            64,
            96
        )
        .is_err());
    }
    assert!(QwenImage21VaeEncoderConfig::from_hf_json(
        &serde_json::to_vec(&raw()).unwrap(),
        1,
        63,
        96
    )
    .is_err());
    assert!(QwenImage21VaeEncoderConfig::new(1, 1, vec![1, 3], 1, vec![false], 1, 4, 4).is_err());
}
#[test]
fn complete_tiny_encoder_executes_real_stored_graph_and_retires_owners() {
    let c = QwenImage21VaeEncoderConfig::new(1, 1, vec![1, 2, 2], 1, vec![false, true], 2, 4, 4)
        .unwrap();
    let family = TypedFamilyRegistration::new(QwenImage21VaeEncoderFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &NumericalProfileId::new(QwenImage21VaeEncoderFamily::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let mut payloads = BTreeMap::new();
    for w in c.expected_weights().unwrap() {
        let name = w.external_name();
        let shape = w.dimensions();
        let mut data = vec![0_u16; (w.bytes() / 2) as usize];
        if name.ends_with(".gamma") {
            data.fill(0x3f80);
        }
        if name == "encoder.conv_in.weight" {
            data[4] = 0x3f80;
        }
        if name.ends_with("conv_shortcut.weight") {
            for o in 0..shape[0] as usize {
                data[o * shape[1] as usize + o % shape[1] as usize] = 0x3f80;
            }
        }
        // All residual main branches and downsampler convolutions are zero;
        // the genuine average shortcuts carry the nonzero, sample-specific data.
        if name == "encoder.conv_out.weight" {
            data[9 + 4] = 0x3f80;
        }
        if name == "quant_conv.weight" {
            data[0] = 0x3f80;
        }
        payloads.insert(
            name.to_string(),
            (
                shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                data.into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
        );
    }
    let directory = tempfile::tempdir().unwrap();
    let views = payloads
        .iter()
        .map(|(name, (shape, data))| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, shape.clone(), data)
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    safetensors::tensor::serialize_to_file(
        views,
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    let capacity = 64 << 20;
    let composition = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.vae.encoder.tiny"),
        capacity,
    )
    .unwrap();
    let mut input = vec![0u8; 2 * 4 * 4 * 4 * 2];
    for n in 0..2 {
        for p in 0..16 {
            let x = (p + 1) as f32 * if n == 0 { 1.0 } else { -1.0 };
            let offset = (n * 64 + p) * 2;
            input[offset..offset + 2].copy_from_slice(&bf16::from_f32(x).to_bits().to_le_bytes());
        }
    }
    let (_, cleanup) = shared::execute_stored_component_with_options(
        &family,
        &archive,
        shared::StoredComponentRuntime {
            runtime: composition.runtime().clone(),
            registry: composition.registry(),
            catalog: composition.catalog(),
            materializers: composition.weight_materializers(),
            capacity_bytes: capacity,
            staging_bytes: 4096,
            provenance: json!({"scope":"single-frame encoder; real stored tiny graph"}),
        },
        vec![shared::ComponentInput {
            value_id: id(QwenImage21VaeEncoderFamily::INPUT_ID),
            tensor: c.input_tensor().unwrap(),
            bytes: &input,
        }],
        c.compile_options().unwrap(),
        |outputs, _| {
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].tensor().dimensions(), [2, 1, 1, 1]);
            let values = outputs[0]
                .bytes()
                .bytes()
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect::<Vec<_>>();
            assert!(values[0].is_finite() && values[0] > 0.0);
            assert!(values[1].is_finite() && values[1] < 0.0);
            Ok(json!({"nonzero_samples_preserved":true}))
        },
    )
    .unwrap();
    assert_eq!(cleanup["managed_host_committed_bytes"], 0);
    assert_eq!(cleanup["managed_host_allocated_bytes"], 0);
    assert_eq!(cleanup["released_static_resources"], 1);
}
