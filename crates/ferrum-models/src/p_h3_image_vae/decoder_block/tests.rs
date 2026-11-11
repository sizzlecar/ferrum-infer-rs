use super::*;
use ferrum_quantization::{
    comfy_convrot::{ComfyCanonicalWeightSource, ComfyQuantizationMetadata},
    SafetensorsArchive,
};
use half::bf16;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use serde_json::json;
#[cfg(feature = "cuda")]
mod cuda;
pub(in crate::p_h3_image_vae) mod oracle;
type Family = PH3ImageVaeDecoderBlockFamily;
type Preset = PH3ImageVaeDecoderBlockExecutionPreset;
type Tensors = BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>;
fn id<T>(s: impl Into<String>) -> T
where
    T: TryFrom<String>,
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn limits() -> PH3ImageVaeDecoderBlockLimits {
    PH3ImageVaeDecoderBlockLimits {
        max_metadata_bytes: 1 << 20,
        max_stored_bytes: 16 << 20,
        max_plan_nodes: 128,
    }
}
fn architecture(heads: u64) -> PH3ImageVaeDecoderConfig {
    PH3ImageVaeDecoderConfig::from_hf_json(&serde_json::to_vec(&json!({
        "_class_name":super::super::PH3_IMAGE_VAE_CLASS,"latent_channels":24,"out_channels":3,
        "spatial_downsample_factors":[2,2,2,2],"decoder_num_layers":2,"decoder_num_attention_heads":heads,
        "decoder_attention_head_dim":64,"decoder_num_register_tokens":4,"decoder_ffn_mult":4,
        "decoder_rope_dim_ratio":0.75,"decoder_rope_theta":100,"decoder_norm_eps":0.00001,
    })).unwrap()).unwrap()
}
fn config(batch: u64) -> PH3ImageVaeDecoderBlockConfig {
    // A nonzero block index catches hardcoded layer-zero weight loading. Four
    // heads preserve the released D64/rot48 layout with a single H256 group.
    PH3ImageVaeDecoderBlockConfig::new(architecture(4), 1, batch, 7, Preset::F32A8FullAttention)
        .unwrap()
}
fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}
fn bf16_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|v| bf16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}
fn original(c: &PH3ImageVaeDecoderBlockConfig) -> Tensors {
    let mut tensors = BTreeMap::new();
    for (name, shape) in weights::original_bf16_vectors(c).unwrap() {
        let count = shape.iter().product::<u64>() as usize;
        let values = (0..count).map(|i| {
            if name.contains("norm") {
                0.75 + (i % 9) as f32 / 16.0
            } else if name.starts_with("scale") {
                ((i % 7) as f32 - 3.0) / 8.0
            } else {
                ((i % 11) as f32 - 5.0) / 128.0
            }
        });
        tensors.insert(
            weights::external(c, &name),
            (
                Dtype::BF16,
                shape.into_iter().map(|v| v as usize).collect(),
                bf16_bytes(values),
            ),
        );
    }
    for (module, input, output) in weights::projections(c).unwrap() {
        let name = weights::external(c, module);
        let n = output as usize;
        let k = input as usize;
        // Signed INT8 remains signed, including -128, and rows vary across
        // heads/QKV roles. Constant or all-zero branches cannot mask a wrong
        // per-head permutation or an omitted vector/bias/scale.
        let bytes = (0..n * k)
            .map(|i| {
                let row = i / k;
                let col = i % k;
                if col == 0 && row % 17 == 0 {
                    (-128i8) as u8
                } else {
                    (((row * 7 + col * 3) % 19) as i8 - 9) as u8
                }
            })
            .collect();
        tensors.insert(format!("{name}.weight"), (Dtype::I8, vec![n, k], bytes));
        tensors.insert(
            format!("{name}.weight_scale"),
            (
                Dtype::F32,
                vec![n, 1],
                f32_bytes((0..n).map(|i| (1 + i % 3) as f32 / 32768.0)),
            ),
        );
        let metadata = serde_json::to_vec(&ComfyQuantizationMetadata::Int8Tensorwise {
            convrot: true,
            convrot_groupsize: 256,
        })
        .unwrap();
        tensors.insert(
            format!("{name}.comfy_quant"),
            (Dtype::U8, vec![metadata.len()], metadata),
        );
    }
    tensors
}
fn archive(tensors: &Tensors) -> (tempfile::TempDir, ComfyCanonicalWeightSource) {
    let directory = tempfile::tempdir().unwrap();
    let views = tensors
        .iter()
        .map(|(name, (dtype, shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, &None, &directory.path().join("weights.safetensors")).unwrap();
    let archive =
        SafetensorsArchive::open_single_file(directory.path(), "weights.safetensors").unwrap();
    (directory, ComfyCanonicalWeightSource::new(archive).unwrap())
}
fn input_values(c: &PH3ImageVaeDecoderBlockConfig) -> BTreeMap<ProgramValueId, Vec<f32>> {
    let h = c.architecture().hidden_size().unwrap() as usize;
    let rows = c.rows().unwrap() as usize;
    let rot = c.architecture().rotary_dim() as usize;
    let mut cosine = Vec::new();
    let mut sine = Vec::new();
    for row in 0..rows {
        for i in 0..rot {
            let phase = (row as f32 + 0.5) * (i % (rot / 2) + 1) as f32 / 64.0;
            cosine.push(libm::cosf(phase));
            sine.push(libm::sinf(phase));
        }
    }
    BTreeMap::from([
        (
            id(Family::INPUT_ID),
            (0..rows * h)
                .map(|i| ((i * 7 + i / h * 29) % 97) as f32 / 64.0 - 0.75 + 0.000123)
                .collect(),
        ),
        (id(Family::COSINE_ID), cosine),
        (id(Family::SINE_ID), sine),
    ])
}

#[test]
fn explicit_f32_preset_and_executable_geometry_are_distinct_from_metadata_inventory() {
    let c = config(2);
    let raw = serde_json::to_value(&c).unwrap();
    assert_eq!(
        serde_json::from_value::<PH3ImageVaeDecoderBlockConfig>(raw.clone()).unwrap(),
        c
    );
    assert_eq!(raw["execution_preset"], "f32_a8_full_attention");
    let mut missing = raw;
    missing.as_object_mut().unwrap().remove("execution_preset");
    assert!(serde_json::from_value::<PH3ImageVaeDecoderBlockConfig>(missing).is_err());
    assert!(PH3ImageVaeDecoderBlockConfig::new(
        architecture(4),
        2,
        1,
        7,
        Preset::F32A8FullAttention
    )
    .is_err());
    // Metadata-only source maps permit this H64 architecture; the executable
    // A8 route must reject its incomplete H256 input group before admission.
    assert!(architecture(1).expected_comfy_weights().is_ok());
    assert!(PH3ImageVaeDecoderBlockConfig::new(
        architecture(1),
        0,
        1,
        7,
        Preset::F32A8FullAttention
    )
    .is_err());
    assert!(PH3ImageVaeDecoderBlockConfig::new(
        architecture(4),
        0,
        0,
        7,
        Preset::F32A8FullAttention
    )
    .is_err());
    let specs = c.input_tensor_specs().unwrap();
    assert_eq!(
        specs[&id::<ProgramValueId>(Family::COSINE_ID)].dimensions,
        [14, 48]
    );
    assert_eq!(
        specs[&id::<ProgramValueId>(Family::INPUT_ID)].dimensions,
        [14, 256]
    );
    assert!(specs.values().all(|s| s.element_type == ElementType::F32));
}

#[test]
fn checkpoint_keeps_four_raw_w8_groups_and_promotes_only_real_bf16_vectors() {
    let c = config(2);
    let (_directory, source) = archive(&original(&c));
    let checkpoint = PH3ImageVaeDecoderBlockCheckpoint::new(&c, &source, limits()).unwrap();
    let family = Family::new(limits()).unwrap();
    let stored = family.weights(&c).unwrap();
    let schema = stored.stored_schema().unwrap();
    for component in &schema.components {
        let name = &component.external_names[0];
        let original = source.archive().tensor(name).unwrap();
        let payload = checkpoint.component(component).unwrap();
        assert!(name.starts_with("decoder.transformer_blocks.1."));
        if original.dtype() == Dtype::BF16 {
            assert_eq!(payload.element_type(), ElementType::F32);
            let values = original
                .bytes()
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32());
            assert_eq!(payload.bytes(), f32_bytes(values));
        } else {
            assert_eq!(payload.bytes().as_ptr(), original.bytes().as_ptr());
            assert!(payload.retained_host_memory().is_some());
        }
    }
    assert!(schema.components.iter().all(|w| w
        .external_names
        .iter()
        .all(|name| !name.contains("norm_q") && !name.contains("norm_k"))));
    let mut bad = original(&c);
    let name = weights::external(&c, "attn.to_qkv.bias");
    bad.get_mut(&name).unwrap().0 = Dtype::F32;
    bad.get_mut(&name).unwrap().2 = f32_bytes(std::iter::repeat_n(0.0, 768));
    let (_directory, source) = archive(&bad);
    assert!(PH3ImageVaeDecoderBlockCheckpoint::new(&c, &source, limits()).is_err());
    let mut bad = original(&c);
    bad.remove(&weights::external(&c, "scale2"));
    let (_directory, source) = archive(&bad);
    assert!(PH3ImageVaeDecoderBlockCheckpoint::new(&c, &source, limits()).is_err());
    let mut low = limits();
    low.max_stored_bytes = 1;
    assert!(family.weights(&c).is_ok());
    assert!(Family::new(low).unwrap().weights(&c).is_err());
}

#[test]
fn graph_preserves_per_head_qkv_partial_rotation_full_isolation_and_residual_order() {
    let c = config(2);
    let family = Family::new(limits()).unwrap();
    let prepared = TypedFamilyRegistration::new(family.clone())
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &id(Family::CUDA_F32_PROFILE_ID),
        )
        .unwrap();
    let graph = prepared.program();
    let options = c.compile_options().unwrap();
    for output in graph
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .flat_map(|n| &n.outputs)
    {
        let spec = options
            .tensor_specs()
            .get(output)
            .expect("every actual emitted output has concrete geometry");
        assert_eq!(spec.element_type, ElementType::F32);
        assert_eq!(spec.dimensions[0], c.rows().unwrap());
    }
    let nodes: Vec<_> = graph.blocks().iter().flat_map(|b| &b.nodes).collect();
    let producer = |name| {
        let output = program::value(name).unwrap();
        *nodes.iter().find(|n| n.outputs.contains(&output)).unwrap()
    };
    for (name, start) in [("query", 0), ("key", 64), ("value", 128)] {
        let n = producer(name);
        assert_eq!(
            n.operation_id.as_str(),
            GROUPED_COLUMN_SLICE_F32_OPERATION_ID
        );
        assert_eq!(
            n.attributes[&id::<AttributeId>("group_width")],
            SemanticValue::Unsigned(192)
        );
        assert_eq!(
            n.attributes[&id::<AttributeId>("column_start")],
            SemanticValue::Unsigned(start)
        );
        assert_eq!(
            n.attributes[&id::<AttributeId>("column_count")],
            SemanticValue::Unsigned(64)
        );
        assert_eq!(n.inputs, [program::value("qkv_biased").unwrap()]);
    }
    for name in ["query_normalized_rotated", "key_normalized_rotated"] {
        let n = producer(name);
        assert_eq!(
            n.operation_id.as_str(),
            UNIT_HEAD_RMS_PARTIAL_ROPE_F32_OPERATION_ID
        );
        assert_eq!(
            n.attributes[&id::<AttributeId>("head_dim")],
            SemanticValue::Unsigned(64)
        );
        assert_eq!(
            n.attributes[&id::<AttributeId>("rotary_dim")],
            SemanticValue::Unsigned(48)
        );
        assert!(!n.inputs.iter().any(|v| v.as_str().contains("weight")));
    }
    let attention = producer("attention");
    assert_eq!(
        attention.attributes[&id::<AttributeId>("sequence_length")],
        SemanticValue::Unsigned(7)
    );
    assert_eq!(
        attention.attributes[&id::<AttributeId>("segment_bidirectional")],
        SemanticValue::Integers(vec![1])
    );
    assert_eq!(
        attention.attributes[&id::<AttributeId>("key_run_counts")],
        SemanticValue::Integers(vec![1, 1])
    );
    assert_eq!(
        producer("attention_finite").operation_id.as_str(),
        NAN_TO_NUM_F32_OPERATION_ID
    );
    assert_eq!(
        producer("activation").operation_id.as_str(),
        SILU_MUL_F32_OPERATION_ID
    );
    let output = *nodes
        .iter()
        .find(|n| n.outputs.iter().any(|v| v.as_str() == Family::OUTPUT_ID))
        .unwrap();
    assert_eq!(
        output.operation_id.as_str(),
        CHANNEL_SCALED_RESIDUAL_F32_OPERATION_ID
    );
    assert_eq!(
        output.inputs[0],
        program::value("attention_residual").unwrap()
    );
    assert_eq!(output.inputs[1], program::value("w2_biased").unwrap());
    assert!(prepared
        .numerical_profile()
        .boundaries
        .values()
        .all(|v| *v == ElementType::F32));
    let mut low = limits();
    low.max_plan_nodes = nodes.len() - 1;
    assert!(Family::new(low).unwrap().weights(&c).is_err());
}
