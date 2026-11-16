use super::super::{Qwen3VLTextKeyRun, Qwen3VLTextLayout};
use super::*;
mod bindings;
use ferrum_quantization::SafetensorsArchive;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};

fn metadata() -> Value {
    let mut raw = serde_json::json!({"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl",
        "text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16","attention_bias":false,"attention_dropout":0.0,
        "hidden_act":"silu","num_hidden_layers":4,"hidden_size":128,"intermediate_size":256,"num_attention_heads":2,
        "num_key_value_heads":1,"head_dim":128,"vocab_size":512,"max_position_embeddings":1024,"rms_norm_eps":0.000001,
        "rope_theta":5000000,"rope_scaling":{"rope_type":"default","mrope_interleaved":true,"mrope_section":[24,20,20]}}});
    raw["num_hidden_layers_override"] = 2.into();
    raw["quantization_config"] = serde_json::json!({"quant_method":"nvfp4","activation_scheme":"dynamic",
        "fmt":"e2m1","group_size":16,"scale_fmt":"e4m3","scale_layout":"128x4",
        "modules_to_not_convert":["model.visual","lm_head"]});
    raw
}
type Tensor = (Dtype, Vec<usize>, Vec<u8>);
fn tensors() -> BTreeMap<String, Tensor> {
    let mut out = BTreeMap::new();
    // Independent physical inventory for two retained layers, with Q width
    // deliberately different from hidden width and KV width.
    for layer in 0..2 {
        let prefix = format!("model.language_model.layers.{layer}");
        for name in [
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ] {
            out.insert(
                format!("{prefix}.{name}.weight"),
                (Dtype::F32, vec![128], 1.0f32.to_le_bytes().repeat(128)),
            );
        }
        for (name, n, k) in [
            ("self_attn.q_proj", 256, 128),
            ("self_attn.k_proj", 128, 128),
            ("self_attn.v_proj", 128, 128),
            ("self_attn.o_proj", 128, 256),
            ("mlp.gate_proj", 256, 128),
            ("mlp.up_proj", 256, 128),
            ("mlp.down_proj", 128, 256),
        ] {
            out.insert(
                format!("{prefix}.{name}.weight_packed"),
                (Dtype::U8, vec![n, k / 2], vec![0x62; n * k / 2]),
            );
            out.insert(
                format!("{prefix}.{name}.weight_scale"),
                (Dtype::U8, vec![n, k / 16], vec![0x38; n * k / 16]),
            );
            out.insert(
                format!("{prefix}.{name}.weight_global_scale"),
                (Dtype::F32, vec![1], 3.0f32.to_le_bytes().to_vec()),
            );
        }
    }
    out
}
fn open(
    raw: &Value,
    tensors: BTreeMap<String, Tensor>,
    layers: u64,
    limits: Qwen3VLTextNvfp4Limits,
) -> (
    tempfile::TempDir,
    anyhow::Result<Qwen3VLTextNvfp4Checkpoint>,
) {
    let dir = tempfile::tempdir().unwrap();
    let views = tensors.iter().map(|(name, (dtype, shape, bytes))| {
        (name, TensorView::new(*dtype, shape.clone(), bytes).unwrap())
    });
    serialize_to_file(views, &None, &dir.path().join("model.safetensors")).unwrap();
    let archive = SafetensorsArchive::open(dir.path()).unwrap();
    let layout = Qwen3VLTextLayout::new(3, vec![vec![Qwen3VLTextKeyRun::new(3, true)]]).unwrap();
    let result = Qwen3VLTextNvfp4Checkpoint::new(raw, archive, layout, layers, limits);
    (dir, result)
}
fn limits() -> Qwen3VLTextNvfp4Limits {
    Qwen3VLTextNvfp4Limits {
        maximum_projections: 14,
        maximum_source_bytes: 1 << 20,
    }
}

#[test]
fn fused_causal_profile_preserves_sources_and_rejects_padding() {
    let (_dir, c) = open(&metadata(), tensors(), 2, limits());
    let old = c.unwrap();
    let schema = old.config().schema().unwrap();
    let c = old
        .with_attention(Qwen3VLTextNvfp4Attention::VendorFusedCausal)
        .unwrap();
    assert_eq!(c.config().schema().unwrap(), schema);
    let family = Qwen3VLTextNvfp4Family::new().unwrap();
    let profiles = family.numerical_profiles(c.config()).unwrap();
    let profile = profiles
        .resolve(&NumericalProfileId::new(Qwen3VLTextNvfp4Family::CUDA_FUSED_PROFILE_ID).unwrap())
        .unwrap();
    let graph = family.semantic_program(c.config(), profile).unwrap();
    assert_eq!(
        graph
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .filter(|n| n.operation_id.as_str() == STRUCTURED_GQA_ATTENTION_FUSED_BF16_OPERATION_ID)
            .count(),
        2
    );
    assert!(!graph
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .any(|n| n.operation_id.as_str() == STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID));
    let mut raw = serde_json::to_value(c.config()).unwrap();
    raw["text"]["layout"]["key_runs"][0] =
        serde_json::json!([{"length":1,"valid":false},{"length":2,"valid":true}]);
    let bad: Qwen3VLTextNvfp4Config = serde_json::from_value(raw).unwrap();
    assert!(bad.validate().is_err());
}
fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}

#[test]
fn visual_deepstack_preserves_original_nvfp4_sources_and_three_post_layer_updates() {
    use super::super::{Qwen3VLTextDeepStack, Qwen3VLTextVisualRun};
    let mut raw = metadata();
    raw["num_hidden_layers_override"] = 3.into();
    let mut source = tensors();
    let third: Vec<_> = source
        .iter()
        .filter(|(name, _)| name.starts_with("model.language_model.layers.1."))
        .map(|(name, tensor)| (name.replacen("layers.1.", "layers.2.", 1), tensor.clone()))
        .collect();
    source.extend(third);
    let mut capacity = limits();
    capacity.maximum_projections = 21;
    let (_dir, checkpoint) = open(&raw, source, 3, capacity);
    let checkpoint = checkpoint.unwrap();
    let original_schema = checkpoint.config().schema().unwrap();
    let visual = Qwen3VLTextDeepStack::new(vec![Qwen3VLTextVisualRun::new(1, 1).unwrap()]).unwrap();
    let checkpoint = checkpoint.with_deepstack(visual).unwrap();
    assert_eq!(checkpoint.config().schema().unwrap(), original_schema);
    let c = checkpoint.config();
    let family = Qwen3VLTextNvfp4Family::new().unwrap();
    let graph = program::build(c, family.family_id()).unwrap();
    let dense = super::super::program::build(c.text(), family.family_id()).unwrap();
    let options = c.compile_options().unwrap();
    for name in DenseFamily::DEEPSTACK_IDS {
        let port = ProgramValueId::new(name).unwrap();
        assert_eq!(options.tensor_specs()[&port].dimensions, [1, 128]);
        assert!(graph.inputs().contains(&port));
        let expected = dense
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .find(|n| n.inputs.contains(&port))
            .unwrap();
        assert_eq!(
            expected.operation_id.as_str(),
            ROW_SCATTER_ADD_BF16_OPERATION_ID
        );
        let actual = graph
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .find(|n| n.id == expected.id)
            .unwrap();
        assert_eq!(actual, expected);
        row_scatter_add_bf16_contract()
            .unwrap()
            .descriptor()
            .validate_attributes(&actual.attributes)
            .unwrap();
    }
    let account = host(1 << 20);
    let managed = checkpoint.managed_source(&account).unwrap();
    for component in &original_schema.components {
        let payload = managed.component(component).unwrap();
        assert_eq!(payload.external_names(), component.external_names);
    }
    drop(managed);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    let (_dir, too_short) = open(&metadata(), tensors(), 2, limits());
    let visual = Qwen3VLTextDeepStack::new(vec![Qwen3VLTextVisualRun::new(1, 1).unwrap()]).unwrap();
    assert!(too_short.unwrap().with_deepstack(visual).is_err());
}

#[test]
fn encoder_prefix_reuses_bf16_graph_and_preserves_physical_nvfp4_sources() {
    let (_dir, checkpoint) = open(&metadata(), tensors(), 1, limits());
    let checkpoint = checkpoint.unwrap();
    let c = checkpoint.config();
    assert_eq!(
        (
            c.declared_layers(),
            c.exported_layers(),
            c.executed_layers(),
            c.projection_count()
        ),
        (4, 2, 1, 7)
    );
    let family = Qwen3VLTextNvfp4Family::new().unwrap();
    let prepared = TypedFamilyRegistration::new(family)
        .prepare_with_profile(
            &serde_json::to_value(c).unwrap(),
            &NumericalProfileId::new(Qwen3VLTextNvfp4Family::CUDA_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let graph = prepared.program();
    assert_eq!(
        graph.outputs(),
        [ProgramValueId::new(DenseFamily::OUTPUT_ID).unwrap()]
    );
    assert_eq!(graph.blocks().len(), 1);
    let nodes = &graph.blocks()[0].nodes;
    assert_eq!(
        nodes
            .iter()
            .filter(|n| n.operation_id.as_str() == NVFP4_QUANTIZE_BF16_OPERATION_ID)
            .count(),
        4
    );
    assert_eq!(
        nodes
            .iter()
            .filter(|n| n.operation_id.as_str() == NVFP4_MATMUL_BF16_OPERATION_ID)
            .count(),
        7
    );
    assert!(!nodes
        .iter()
        .any(|n| n.operation_id.as_str() == DENSE_LINEAR_BF16_OPERATION_ID));
    // Every unaffected node is exactly the existing text program node. In
    // particular this keeps the actual causal/padding mask and pre-norm output.
    let base = super::super::program::build(c.text(), graph.family_id()).unwrap();
    for original in &base.blocks()[0].nodes {
        if original.operation_id.as_str() != DENSE_LINEAR_BF16_OPERATION_ID {
            assert_eq!(
                nodes.iter().find(|n| n.id == original.id).unwrap(),
                original
            );
        }
    }
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("stored")
    };
    assert_eq!(schema.components.len(), 4 + 7 * 3 + 1);
    for t in &schema.tensors {
        let binding = ResolvedWeightBinding::from_schema(schema, &t.id).unwrap();
        if let Some(view) = binding.physical_component_view() {
            assert!(matches!(
                view.role(),
                WeightComponentRole::PackedValues | WeightComponentRole::Scales
            ));
            assert!(t.logical_elements().is_err());
        }
    }
}

#[test]
fn encoder_scalars_and_norms_use_host_account_with_retained_payload_ownership() {
    let (_dir, checkpoint) = open(&metadata(), tensors(), 1, limits());
    let checkpoint = checkpoint.unwrap();
    let account = host(4096);
    let source = checkpoint.managed_source(&account).unwrap();
    assert_eq!(account.usage().unwrap().allocated_bytes(), 32);
    let schema = checkpoint.config().schema().unwrap();
    let alpha = schema
        .components
        .iter()
        .find(|c| c.id.as_str().starts_with("derived."))
        .unwrap();
    let payload = source.component(alpha).unwrap();
    assert_eq!(payload.bytes(), &0x3eaa_aaab_u32.to_le_bytes());
    let retained = payload.retained_host_memory().unwrap().clone();
    drop(payload);
    let packed = schema
        .components
        .iter()
        .find(|c| c.role == WeightComponentRole::PackedValues)
        .unwrap();
    assert_eq!(source.component(packed).unwrap().bytes()[0], 0x62);
    let norm = schema
        .components
        .iter()
        .find(|c| c.external_names[0].ends_with("input_layernorm.weight"))
        .unwrap();
    let norm_payload = source.component(norm).unwrap();
    assert_eq!(norm_payload.element_type(), ElementType::Bf16);
    assert_eq!(norm_payload.bytes(), &0x3f80_u16.to_le_bytes().repeat(128));
    assert_eq!(account.usage().unwrap().allocated_bytes(), 32 + 256);
    drop(norm_payload);
    let mut forged = packed.clone();
    forged.encoding = WeightEncoding::Dense {
        element_type: ElementType::U8,
    };
    assert!(source.component(&forged).is_err());
    let mut changed = checkpoint.config().projections[0].clone();
    changed.global_bits = 4.0f32.to_bits();
    assert!(source
        .component(&changed.alpha_component().unwrap())
        .is_err());
    changed = checkpoint.config().projections[0].clone();
    changed.source_file = "other.safetensors".into();
    assert!(source
        .component(&changed.alpha_component().unwrap())
        .is_err());
    drop(source);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 4);
    drop(retained);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    let too_small = host(31);
    assert!(checkpoint.managed_source(&too_small).is_err());
    assert_eq!(too_small.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn encoder_rejects_missing_prefix_limits_smoothing_and_noncanonical_physical_data() {
    assert!(open(&metadata(), tensors(), 3, limits()).1.is_err());
    let mut raw = metadata();
    raw.as_object_mut()
        .unwrap()
        .remove("num_hidden_layers_override");
    assert!(open(&raw, tensors(), 1, limits()).1.is_err());
    raw = metadata();
    raw["quantization_config"]["pre_quant_scale"] = true.into();
    assert!(open(&raw, tensors(), 1, limits()).1.is_err());
    assert!(open(
        &metadata(),
        tensors(),
        1,
        Qwen3VLTextNvfp4Limits {
            maximum_projections: 6,
            ..limits()
        }
    )
    .1
    .is_err());
    assert!(open(
        &metadata(),
        tensors(),
        1,
        Qwen3VLTextNvfp4Limits {
            maximum_source_bytes: 128,
            ..limits()
        }
    )
    .1
    .is_err());
    for (dtype, shape, bytes) in [
        (Dtype::BF16, vec![1], vec![0x40, 0x40]),
        (Dtype::F32, vec![], 3.0f32.to_le_bytes().to_vec()),
        (Dtype::F32, vec![1], f32::NAN.to_le_bytes().to_vec()),
    ] {
        let mut data = tensors();
        data.insert(
            "model.language_model.layers.0.self_attn.q_proj.weight_global_scale".into(),
            (dtype, shape, bytes),
        );
        assert!(open(&metadata(), data, 1, limits()).1.is_err());
    }
    let (_dir, checkpoint) = open(&metadata(), tensors(), 1, limits());
    let checkpoint = checkpoint.unwrap();
    let family = Qwen3VLTextNvfp4Family::new().unwrap();
    let mut raw = serde_json::to_value(checkpoint.config()).unwrap();
    raw["projections"][0]["global_bits"] = f32::INFINITY.to_bits().into();
    assert!(family.parse_config(&raw).is_err());
    raw = serde_json::to_value(checkpoint.config()).unwrap();
    raw["scalar_profile"] = "unqualified".into();
    assert!(family.parse_config(&raw).is_err());
}
