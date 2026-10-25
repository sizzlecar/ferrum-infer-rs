use super::*;
use serde_json::json;
fn hf(layers: u64, h: u64, m: u64, q: u64, kv: u64, d: u64) -> Value {
    json!({"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl","dtype":"bfloat16",
        "text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16","attention_bias":false,"attention_dropout":0.0,
        "hidden_act":"silu","num_hidden_layers":layers,"hidden_size":h,"intermediate_size":m,"num_attention_heads":q,"num_key_value_heads":kv,"head_dim":d,
        "vocab_size":151936,"max_position_embeddings":262144,"rms_norm_eps":0.000001,"rope_theta":5000000,
        "rope_scaling":{"rope_type":"default","mrope_interleaved":true,"mrope_section":[d/2,0,0]}}})
}
fn layout(left: bool) -> Qwen3VLTextLayout {
    Qwen3VLTextLayout::new(
        3,
        vec![
            vec![
                Qwen3VLTextKeyRun::new(1, !left),
                Qwen3VLTextKeyRun::new(2, left),
            ],
            vec![Qwen3VLTextKeyRun::new(3, true)],
        ],
    )
    .unwrap()
}
fn tiny() -> Qwen3VLTextConfig {
    Qwen3VLTextConfig::new(
        Qwen3VLTextArchitecture::from_hf_value(&hf(2, 8, 12, 4, 2, 2)).unwrap(),
        layout(true),
    )
    .unwrap()
}
#[test]
fn explicit_fp8_text_profile_keeps_separate_linear_scales_and_bf16_boundaries() {
    let config = tiny();
    let family = Qwen3VLTextEncoderFamily::new().unwrap();
    let registration = TypedFamilyRegistration::new(family.clone());
    let prepare = |name| {
        registration
            .prepare_with_profile(
                &serde_json::to_value(&config).unwrap(),
                &NumericalProfileId::new(name).unwrap(),
            )
            .unwrap()
    };
    let bf16 = prepare(Qwen3VLTextEncoderFamily::CUDA_PROFILE_ID);
    let fp8 = prepare(Qwen3VLTextEncoderFamily::CUDA_FP8_PROFILE_ID);
    let groups = Qwen3VLTextEncoderFamily::fp8_weight_groups(&config).unwrap();
    assert_eq!(groups.len(), 14);
    assert!(groups.iter().all(|g| g.len() == 1));
    assert_eq!(bf16.weights(), fp8.weights());
    assert_eq!(
        bf16.numerical_profile().boundaries,
        fp8.numerical_profile().boundaries
    );
    assert_ne!(bf16.fingerprint().unwrap(), fp8.fingerprint().unwrap());
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            4096,
            HostMemoryClassLimits {
                weight_backing_bytes: 4096,
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 0,
            },
        )
        .unwrap(),
    );
    let materializer =
        ferrum_quantization::Bf16WholeTensorFp8Materializer::new(host, groups).unwrap();
    assert!(materializer.selection(&bf16).is_err());
    assert!(materializer.selection(&fp8).is_ok());
    for (a, b) in bf16
        .program()
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .zip(fp8.program().blocks().iter().flat_map(|b| &b.nodes))
    {
        let mut restored = b.clone();
        if b.operation_id.as_str() == DENSE_LINEAR_FP8_BF16_OPERATION_ID {
            assert_eq!(a.operation_id.as_str(), DENSE_LINEAR_BF16_OPERATION_ID);
            restored.operation_id = a.operation_id.clone();
        }
        assert_eq!(&restored, a);
    }
}
#[test]
fn real_metadata_schema_has_exactly_required_text_layer_weights_without_unexecuted_components() {
    let mut raw = hf(36, 4096, 12288, 32, 8, 128);
    raw["text_config"]["rope_scaling"]["mrope_section"] = json!([24, 20, 20]);
    let c = Qwen3VLTextConfig::new(
        Qwen3VLTextArchitecture::from_hf_value(&raw).unwrap(),
        layout(true),
    )
    .unwrap();
    let family = Qwen3VLTextEncoderFamily::new().unwrap();
    let stored = family.weights(&c).unwrap();
    let s = stored.stored_schema().unwrap();
    assert_eq!(s.tensors.len(), 36 * 11);
    assert_eq!(s.components.len(), 36 * 11);
    assert!(s.components.iter().all(|w| w.required));
    assert!(s.tensors.iter().all(|w| w.required));
    assert_eq!(
        s.components
            .iter()
            .map(|w| w.physical_bytes().unwrap())
            .sum::<u64>(),
        13_892_143_104
    );
    for layer in [0, 35] {
        for (name, shape) in [
            ("self_attn.q_proj.weight", vec![4096, 4096]),
            ("self_attn.k_proj.weight", vec![1024, 4096]),
            ("self_attn.v_proj.weight", vec![1024, 4096]),
            ("self_attn.o_proj.weight", vec![4096, 4096]),
            ("self_attn.q_norm.weight", vec![128]),
            ("self_attn.k_norm.weight", vec![128]),
            ("input_layernorm.weight", vec![4096]),
            ("post_attention_layernorm.weight", vec![4096]),
            ("mlp.gate_proj.weight", vec![12288, 4096]),
            ("mlp.up_proj.weight", vec![12288, 4096]),
            ("mlp.down_proj.weight", vec![4096, 12288]),
        ] {
            let external = weights::external_name(layer, name);
            let w = s
                .components
                .iter()
                .find(|w| w.external_names == [external.clone()])
                .unwrap();
            assert_eq!(w.dimensions, shape);
            assert_eq!(
                w.encoding,
                WeightEncoding::Dense {
                    element_type: ElementType::Bf16
                }
            );
        }
    }
    assert!(s.components.iter().all(|w| w
        .external_names
        .iter()
        .all(|n| n.starts_with("model.language_model.layers."))));
}
#[test]
fn two_layer_program_has_seventeen_fixed_nodes_per_layer_and_only_pre_final_norm_output() {
    let c = tiny();
    let family = Qwen3VLTextEncoderFamily::new().unwrap();
    let profiles = family.numerical_profiles(&c).unwrap();
    for profile in profiles.profiles() {
        let p = family.semantic_program(&c, profile).unwrap();
        assert_eq!(p.inputs().len(), 3);
        assert_eq!(p.weights().len(), 22);
        assert_eq!(
            p.outputs(),
            [ProgramValueId::new(Qwen3VLTextEncoderFamily::OUTPUT_ID).unwrap()]
        );
        assert_eq!(p.input_tensor_specs(), &c.input_tensor_specs().unwrap());
        let dense = if matches!(
            profile.id.as_str(),
            Qwen3VLTextEncoderFamily::CUDA_FP8_PROFILE_ID
                | Qwen3VLTextEncoderFamily::METAL_FP8_PROFILE_ID
        ) {
            DENSE_LINEAR_FP8_BF16_OPERATION_ID
        } else {
            DENSE_LINEAR_BF16_OPERATION_ID
        };
        let expected = [
            HEAD_RMS_NORM_BF16_OPERATION_ID,
            dense,
            dense,
            dense,
            HEAD_RMS_NORM_BF16_OPERATION_ID,
            HEAD_RMS_NORM_BF16_OPERATION_ID,
            HALF_SPLIT_ROPE_BF16_OPERATION_ID,
            HALF_SPLIT_ROPE_BF16_OPERATION_ID,
            STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID,
            dense,
            PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
            HEAD_RMS_NORM_BF16_OPERATION_ID,
            dense,
            dense,
            SILU_MUL_BF16_OPERATION_ID,
            dense,
            PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
        ];
        for (i, b) in p.blocks().iter().enumerate() {
            assert_eq!(
                b.nodes
                    .iter()
                    .map(|n| n.operation_id.as_str())
                    .collect::<Vec<_>>(),
                expected
            );
            assert!(b.nodes.iter().all(|n| n.work == ProgramNodeWorkSpec::Fixed));
            assert_eq!(b.nodes[8].attributes, c.attention_attributes().unwrap());
            assert_eq!(
                b.nodes[8].inputs,
                [
                    b.nodes[6].outputs[0].clone(),
                    b.nodes[7].outputs[0].clone(),
                    b.nodes[3].outputs[0].clone()
                ]
            );
            assert_eq!(b.nodes[10].inputs[0], b.nodes[0].inputs[0]);
            assert_eq!(b.nodes[16].inputs[0], b.nodes[10].outputs[0]);
            if i == 1 {
                assert_eq!(b.nodes[0].inputs[0], p.blocks()[0].nodes[16].outputs[0]);
            }
        }
        assert_eq!(profile.boundaries.len(), 34);
        assert_eq!(profile.operations.len(), 6);
        assert!(profile.states.is_empty() && profile.kv_storage.is_empty());
    }
    assert_eq!(
        family.semantic_metadata(&c).unwrap(),
        ModelSemanticMetadata::TensorComponent
    );
}
#[test]
fn explicit_padding_runs_and_frequency_shapes_survive_canonical_preparation() {
    let c = tiny();
    let registration = TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new().unwrap());
    let profile = NumericalProfileId::new(Qwen3VLTextEncoderFamily::REFERENCE_PROFILE_ID).unwrap();
    let prepared = registration
        .prepare_with_profile(&serde_json::to_value(&c).unwrap(), &profile)
        .unwrap();
    let other = Qwen3VLTextConfig::new(c.architecture().clone(), layout(false)).unwrap();
    let other_prepared = registration
        .prepare_with_profile(&serde_json::to_value(&other).unwrap(), &profile)
        .unwrap();
    assert_ne!(prepared.program(), other_prepared.program());
    let specs = c.input_tensor_specs().unwrap();
    assert_eq!(
        specs[&ProgramValueId::new(Qwen3VLTextEncoderFamily::COSINE_ID).unwrap()].dimensions,
        [6, 2]
    );
    assert_eq!(
        specs[&ProgramValueId::new(Qwen3VLTextEncoderFamily::EMBEDDINGS_ID).unwrap()].dimensions,
        [6, 8]
    );
    let mut canonical = serde_json::to_value(&c).unwrap();
    canonical["layout"]["key_runs"][0][0]["length"] = json!(2);
    assert!(serde_json::from_value::<Qwen3VLTextConfig>(canonical).is_err());
    let mut canonical = serde_json::to_value(&c).unwrap();
    canonical["final_norm"] = json!(true);
    assert!(registration
        .prepare_with_profile(&canonical, &profile)
        .is_err());
}
#[test]
fn unsupported_text_semantics_and_invalid_head_ratio_fail_closed() {
    for (name, value) in [
        ("model_type", json!("qwen3")),
        ("dtype", json!("float16")),
        ("attention_bias", json!(true)),
        ("hidden_act", json!("gelu")),
        ("num_key_value_heads", json!(3)),
        ("head_dim", json!(3)),
        ("attention_dropout", json!(0.1)),
        ("layer_types", json!(["full_attention"])),
    ] {
        let mut raw = hf(2, 8, 12, 4, 2, 2);
        raw["text_config"][name] = value;
        assert!(Qwen3VLTextArchitecture::from_hf_value(&raw).is_err());
    }
    assert!(Qwen3VLTextLayout::new(
        3,
        vec![vec![
            Qwen3VLTextKeyRun::new(1, true),
            Qwen3VLTextKeyRun::new(2, true)
        ]]
    )
    .is_err());
    assert!(Qwen3VLTextLayout::new(3, vec![vec![Qwen3VLTextKeyRun::new(2, true)]]).is_err());
}

#[test]
fn deepstack_updates_follow_each_complete_layer_and_are_typed_inputs() {
    let base = Qwen3VLTextConfig::new(
        Qwen3VLTextArchitecture::from_hf_value(&hf(4, 8, 12, 4, 2, 2)).unwrap(),
        layout(true),
    )
    .unwrap();
    let runs = Qwen3VLTextDeepStack::new(vec![
        Qwen3VLTextVisualRun::new(1, 2).unwrap(),
        Qwen3VLTextVisualRun::new(4, 1).unwrap(),
    ])
    .unwrap();
    let edit = base.clone().with_deepstack(runs).unwrap();
    let family = Qwen3VLTextEncoderFamily::new().unwrap();
    assert_eq!(
        family.weights(&base).unwrap(),
        family.weights(&edit).unwrap()
    );
    let profiles = family.numerical_profiles(&edit).unwrap();
    let p = family
        .semantic_program(&edit, &profiles.profiles()[0])
        .unwrap();
    for layer in 0..3 {
        let block = &p.blocks()[layer];
        let inject = block.nodes.last().unwrap();
        assert_eq!(inject.operation_id.as_str(), edit::SCATTER_ADD_OPERATION_ID);
        assert_eq!(
            inject.inputs[0],
            block.nodes[block.nodes.len() - 2].outputs[0]
        );
        assert_eq!(
            inject.inputs[1],
            ProgramValueId::new(Qwen3VLTextEncoderFamily::DEEPSTACK_IDS[layer]).unwrap()
        );
        assert_eq!(p.blocks()[layer + 1].nodes[0].inputs[0], inject.outputs[0]);
        assert_eq!(p.input_tensor_specs()[&inject.inputs[1]].dimensions, [3, 8]);
    }
    assert!(!serde_json::to_value(&base)
        .unwrap()
        .as_object()
        .unwrap()
        .contains_key("deepstack"));
    let wire = serde_json::to_value(&edit).unwrap();
    assert_eq!(
        serde_json::from_value::<Qwen3VLTextConfig>(wire).unwrap(),
        edit
    );
    let reg = TypedFamilyRegistration::new(family);
    let id = NumericalProfileId::new(Qwen3VLTextEncoderFamily::REFERENCE_PROFILE_ID).unwrap();
    assert_ne!(
        reg.prepare_with_profile(&serde_json::to_value(&base).unwrap(), &id)
            .unwrap()
            .fingerprint()
            .unwrap(),
        reg.prepare_with_profile(&serde_json::to_value(&edit).unwrap(), &id)
            .unwrap()
            .fingerprint()
            .unwrap()
    );
}

#[test]
fn deepstack_rejects_overlap_padding_cross_sample_and_unchecked_wire() {
    let base = Qwen3VLTextConfig::new(
        Qwen3VLTextArchitecture::from_hf_value(&hf(3, 8, 12, 4, 2, 2)).unwrap(),
        layout(true),
    )
    .unwrap();
    assert!(Qwen3VLTextDeepStack::new(vec![
        Qwen3VLTextVisualRun::new(1, 2).unwrap(),
        Qwen3VLTextVisualRun::new(2, 1).unwrap()
    ])
    .is_err());
    for (start, len) in [(0, 1), (2, 2), (6, 1)] {
        let runs = Qwen3VLTextDeepStack::new(vec![Qwen3VLTextVisualRun::new(start, len).unwrap()])
            .unwrap();
        assert!(base.clone().with_deepstack(runs).is_err());
    }
    let mut wire = serde_json::to_value(&base).unwrap();
    wire["deepstack"] = json!({"runs":[{"start":1,"length":0}]});
    assert!(serde_json::from_value::<Qwen3VLTextConfig>(wire).is_err());
    assert!(Qwen3VLTextVisualRun::new(u64::MAX, 2).is_err());
}

#[test]
fn edit_mrope_uses_image_axes_and_resumes_text_after_maximum_axis() {
    let mut raw = hf(3, 8, 12, 2, 1, 8);
    raw["text_config"]["rope_theta"] = json!(256);
    raw["text_config"]["rope_scaling"]["mrope_section"] = json!([2, 1, 1]);
    let architecture = Qwen3VLTextArchitecture::from_hf_value(&raw).unwrap();
    let layout = Qwen3VLTextLayout::new(
        7,
        vec![vec![
            Qwen3VLTextKeyRun::new(1, false),
            Qwen3VLTextKeyRun::new(6, true),
        ]],
    )
    .unwrap();
    let spans = [Qwen3VLTextImageSpan::new(2, 4, 4, 2).unwrap()];
    let c = Qwen3VLTextConfig::new(architecture, layout)
        .unwrap()
        .with_image_spans(&spans)
        .unwrap();
    let bytes = 7 * 8 * 2 * 2;
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                weight_backing_bytes: 0,
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: 0,
                output_bytes: 0,
            },
        )
        .unwrap(),
    );
    let output = c.managed_edit_mrope(&host, &spans).unwrap();
    // Padding and one text row precede the 2x2 image. Pair lanes select T,H,W,T.
    let expected = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 2, 1],
        [1, 2, 1, 1],
        [1, 2, 2, 1],
        [3, 3, 3, 3],
    ];
    for (row, phases) in expected.into_iter().enumerate() {
        for (column, p) in phases.into_iter().enumerate() {
            // 256^(pair/4) is exactly 1, 4, 16, 64. The oracle uses
            // analytical inverse frequencies rather than the production powf.
            let phase = p as f32 * [1.0, 0.25, 0.0625, 0.015625][column];
            let c = half::bf16::from_f32(libm::cosf(phase))
                .to_bits()
                .to_le_bytes();
            let s = half::bf16::from_f32(libm::sinf(phase))
                .to_bits()
                .to_le_bytes();
            for col in [column, column + 4] {
                let offset = (row * 8 + col) * 2;
                assert_eq!(&output[0].bytes().as_ref()[offset..offset + 2], &c);
                assert_eq!(&output[1].bytes().as_ref()[offset..offset + 2], &s);
            }
        }
    }
    assert_eq!(host.usage().unwrap().allocated_bytes(), bytes);
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(c
        .managed_edit_mrope(&host, &[Qwen3VLTextImageSpan::new(1, 4, 4, 2).unwrap()])
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
