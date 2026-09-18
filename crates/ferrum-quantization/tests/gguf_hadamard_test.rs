use std::collections::BTreeMap;
use std::io::{Cursor, Write};

use candle_core::quantized::gguf_file::{Content, Value};
use ferrum_interfaces::vnext::{
    ContractVersion, ElementType, PhysicalWeightLayout, WeightComponentRole, WeightComponentSource,
    WeightComponentSpec, WeightEncoding, WeightSchema, WeightTensorSpec,
};
use ferrum_quantization::gguf::source::{hadamard_sign_component, hadamard_transform_spec};
use ferrum_quantization::gguf::{
    GgufFile, GgufHadamardDirection, GgufHadamardGdnPermutation, GgufHadamardSigns, GgufInventory,
    GgufModelMetadata, GgufWeightComponentSource, NativeGgufFile,
};

fn metadata() -> BTreeMap<String, Value> {
    BTreeMap::from([
        (
            "general.architecture".into(),
            Value::String("qwen35".into()),
        ),
        ("prism.hadamard.version".into(), Value::U32(1)),
        ("prism.hadamard.block_size".into(), Value::U32(4)),
        (
            "prism.hadamard.transform".into(),
            Value::String("normalized-sylvester-walsh-hadamard".into()),
        ),
        (
            "prism.hadamard.axis".into(),
            Value::String("input-last-dimension".into()),
        ),
        (
            "prism.hadamard.sign_mode".into(),
            Value::String("identity".into()),
        ),
        (
            "prism.hadamard.weight_names".into(),
            strings(&["output.weight"]),
        ),
    ])
}

fn strings(values: &[&str]) -> Value {
    Value::Array(
        values
            .iter()
            .map(|value| Value::String((*value).into()))
            .collect(),
    )
}
fn ints(values: &[i32]) -> Value {
    Value::Array(values.iter().copied().map(Value::I32).collect())
}
fn explicit(meta: &mut BTreeMap<String, Value>, widths: &[i32], signs: &[i32]) {
    meta.insert(
        "prism.hadamard.sign_mode".into(),
        Value::String("explicit".into()),
    );
    meta.insert("prism.hadamard.sign_widths".into(), ints(widths));
    meta.insert("prism.hadamard.sign_values".into(), ints(signs));
}

// Small descriptor/payload fixtures retain raw type IDs unknown to Candle.
fn fixture(meta: &BTreeMap<String, Value>, tensors: &[(&str, u64, u32)]) -> Vec<u8> {
    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    fn kind(value: &Value) -> u32 {
        match value {
            Value::U32(_) => 4,
            Value::I32(_) => 5,
            Value::Bool(_) => 7,
            Value::String(_) => 8,
            Value::Array(_) => 9,
            _ => panic!("unsupported fixture scalar"),
        }
    }
    fn value(bytes: &mut Vec<u8>, item: &Value) {
        match item {
            Value::U32(n) => bytes.extend_from_slice(&n.to_le_bytes()),
            Value::I32(n) => bytes.extend_from_slice(&n.to_le_bytes()),
            Value::Bool(b) => bytes.push(u8::from(*b)),
            Value::String(s) => string(bytes, s),
            Value::Array(items) => {
                let element_type = items.first().map(kind).unwrap_or(5);
                assert!(items.iter().all(|item| kind(item) == element_type));
                bytes.extend_from_slice(&element_type.to_le_bytes());
                bytes.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for item in items {
                    value(bytes, item);
                }
            }
            _ => panic!("unsupported fixture value"),
        }
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    for (key, item) in meta {
        string(&mut bytes, key);
        bytes.extend_from_slice(&kind(item).to_le_bytes());
        value(&mut bytes, item);
    }
    let mut payload_size = 0_u64;
    for (name, width, dtype) in tensors {
        string(&mut bytes, name);
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&dtype.to_le_bytes());
        bytes.extend_from_slice(&payload_size.to_le_bytes());
        let size = match dtype {
            142 => width / 128 * 34,
            42 => width / 64 * 18,
            0 => width * 4,
            _ => panic!("unsupported fixture dtype"),
        };
        payload_size += size.div_ceil(32) * 32;
    }
    bytes.resize(bytes.len().div_ceil(32) * 32 + payload_size as usize, 0);
    bytes
}

fn inventory(
    meta: &BTreeMap<String, Value>,
    tensors: &[(&str, u64, u32)],
) -> candle_core::Result<GgufInventory> {
    let bytes = fixture(meta, tensors);
    GgufInventory::read(&mut Cursor::new(&bytes), bytes.len() as u64)
}
fn rejects(meta: &BTreeMap<String, Value>, tensors: &[(&str, u64, u32)], expected: &str) {
    let error = inventory(meta, tensors).unwrap_err().to_string();
    assert!(error.contains(expected), "expected {expected:?}: {error}");
}
fn file(bytes: &[u8]) -> tempfile::NamedTempFile {
    let mut file = tempfile::Builder::new().suffix(".gguf").tempfile().unwrap();
    file.write_all(bytes).unwrap();
    file
}

#[test]
fn pq2_without_transform_metadata_is_not_implicitly_rotated() {
    let meta = BTreeMap::from([(
        "general.architecture".into(),
        Value::String("qwen35".into()),
    )]);
    let bytes = fixture(&meta, &[("output.weight", 128, 142)]);
    assert!(
        GgufInventory::read(&mut Cursor::new(&bytes), bytes.len() as u64)
            .unwrap()
            .hadamard
            .is_none()
    );
    let artifact = file(&bytes);
    assert!(GgufWeightComponentSource::open(artifact.path())
        .unwrap()
        .file()
        .hadamard()
        .is_none());
}

#[test]
fn explicit_signs_span_the_full_width_and_preserve_inverse_direction() {
    let mut meta = metadata();
    meta.insert(
        "prism.hadamard.inverse_weight_names".into(),
        strings(&["token_embd.weight"]),
    );
    // The two blocks intentionally have different signs.
    explicit(&mut meta, &[8], &[1, 1, 1, 1, -1, 1, -1, 1]);
    let tensors = [("output.weight", 8, 0), ("token_embd.weight", 8, 0)];
    let bytes = fixture(&meta, &tensors);
    let parsed = inventory(&meta, &tensors).unwrap();
    let hadamard = parsed.hadamard.as_ref().unwrap();
    assert_eq!(hadamard.version(), 1);
    assert_eq!(hadamard.block_size(), 4);
    assert_eq!(
        hadamard.weight("output.weight").unwrap().direction(),
        GgufHadamardDirection::BeforeMatmul
    );
    assert_eq!(
        hadamard.weight("token_embd.weight").unwrap().direction(),
        GgufHadamardDirection::AfterEmbeddingLookup
    );
    assert_eq!(
        hadamard.weight("token_embd.weight").unwrap().input_width(),
        8
    );
    let GgufHadamardSigns::Explicit(signs) = hadamard.signs() else {
        panic!("explicit signs lost");
    };
    assert_eq!(signs[&8], [1, 1, 1, 1, -1, 1, -1, 1]);
    let content = Content::read(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(
        GgufInventory::from_content(&content, bytes.len() as u64).unwrap(),
        parsed
    );
    let prefix = &bytes[..parsed.tensor_data_offset as usize];
    assert_eq!(
        GgufInventory::read(&mut Cursor::new(prefix), bytes.len() as u64).unwrap(),
        parsed
    );
    assert_eq!(
        GgufModelMetadata::read(&mut Cursor::new(&bytes))
            .unwrap()
            .hadamard
            .as_ref(),
        Some(hadamard)
    );
    let artifact = file(&bytes);
    let native = NativeGgufFile::open(artifact.path()).unwrap();
    assert_eq!(native.hadamard(), Some(hadamard));
    let serialized = serde_json::to_value(hadamard).unwrap();
    explicit(&mut meta, &[8], &[-1; 8]);
    assert_ne!(
        serde_json::to_value(inventory(&meta, &tensors).unwrap().hadamard.unwrap()).unwrap(),
        serialized
    );
}

#[test]
fn gdn_output_permutation_does_not_rotate_unlisted_alpha_beta() {
    let mut meta = metadata();
    meta.insert(
        "prism.hadamard.weight_names".into(),
        strings(&[
            "blk.0.attn_qkv.weight",
            "blk.0.attn_gate.weight",
            "blk.0.ssm_out.weight",
        ]),
    );
    meta.insert("prism.hadamard.gdn_v_grouped".into(), Value::Bool(true));
    meta.insert("qwen35.ssm.group_count".into(), Value::U32(2));
    meta.insert("qwen35.ssm.time_step_rank".into(), Value::U32(8));
    let tensors = [
        ("blk.0.attn_qkv.weight", 16, 0),
        ("blk.0.attn_gate.weight", 16, 0),
        ("blk.0.ssm_out.weight", 16, 0),
        ("blk.0.ssm_alpha.weight", 16, 0),
        ("blk.0.ssm_beta.weight", 16, 0),
    ];
    let rotation = inventory(&meta, &tensors).unwrap().hadamard.unwrap();
    assert!(rotation.gdn_v_grouped());
    assert_eq!(
        rotation
            .weight("blk.0.ssm_out.weight")
            .unwrap()
            .gdn_permutation(),
        Some(GgufHadamardGdnPermutation {
            head_dim: 2,
            key_heads: 2,
            repeats: 4
        })
    );
    assert!(rotation
        .weight("blk.0.attn_qkv.weight")
        .unwrap()
        .gdn_permutation()
        .is_none());
    assert!(rotation.weight("blk.0.ssm_alpha.weight").is_none());
    assert!(rotation.weight("blk.0.ssm_beta.weight").is_none());
    meta.insert("qwen35.ssm.group_count".into(), Value::U32(3));
    rejects(&meta, &tensors, "incompatible GDN");
    meta.insert("qwen35.ssm.group_count".into(), Value::U32(0));
    rejects(&meta, &tensors, "positive uint32");
    meta.remove("qwen35.ssm.group_count");
    rejects(&meta, &tensors, "positive uint32");
}

#[test]
fn partial_unknown_or_mistyped_transform_metadata_is_rejected() {
    let tensors = [("output.weight", 8, 0)];
    let mut meta = metadata();
    meta.remove("prism.hadamard.version");
    rejects(&meta, &tensors, "missing prism.hadamard.version");
    for (key, value, expected) in [
        ("version", Value::U32(2), "unsupported version"),
        ("version", Value::I32(1), "must be uint32"),
        ("block_size", Value::U32(0), "power of two"),
        ("block_size", Value::U32(3), "power of two"),
        ("block_size", Value::U32(16), "not divisible"),
        (
            "transform",
            Value::String("other".into()),
            "unsupported transform",
        ),
        ("axis", Value::String("output".into()), "unsupported axis"),
        (
            "sign_mode",
            Value::String("other".into()),
            "unsupported sign_mode",
        ),
        ("gdn_v_grouped", Value::U32(1), "must be a boolean"),
        (
            "future_semantics",
            Value::Bool(true),
            "unknown prism.hadamard",
        ),
    ] {
        let mut meta = metadata();
        meta.insert(format!("prism.hadamard.{key}"), value);
        rejects(&meta, &tensors, expected);
    }
    let mut meta = metadata();
    meta.insert(
        "general.architecture".into(),
        Value::String("unknown-architecture".into()),
    );
    rejects(&meta, &tensors, "no verified transform roles");
}

#[test]
fn signs_reject_wrong_length_duplicate_width_missing_width_and_non_sign_values() {
    let tensors = [("output.weight", 8, 0)];
    for (widths, signs, expected) in [
        (vec![], vec![], "nonempty sign_widths"),
        (vec![0], vec![], "positive int32"),
        (vec![-4], vec![], "positive int32"),
        (vec![3], vec![1; 3], "divisible by block_size"),
        (vec![8, 8], vec![1; 16], "unique"),
        (vec![8], vec![1; 7], "shorter"),
        (vec![8], vec![1; 9], "unconsumed"),
        (vec![4], vec![1; 4], "no signs for width 8"),
        (vec![8], vec![0; 8], "+1 or -1"),
    ] {
        let mut meta = metadata();
        explicit(&mut meta, &widths, &signs);
        rejects(&meta, &tensors, expected);
    }
    let mut meta = metadata();
    meta.insert("prism.hadamard.sign_widths".into(), ints(&[8]));
    rejects(&meta, &tensors, "identity sign mode");
    explicit(&mut meta, &[8], &[1; 8]);
    meta.insert(
        "prism.hadamard.sign_values".into(),
        Value::Array(vec![Value::U32(1); 8]),
    );
    rejects(&meta, &tensors, "arrays must contain int32 or strings");
}

#[test]
fn tensor_roles_missing_names_and_duplicate_transforms_are_rejected() {
    let tensors = [("output.weight", 8, 0), ("blk.0.ssm_alpha.weight", 8, 0)];
    for (names, expected) in [
        (vec![], "must not be empty"),
        (
            vec!["output.weight", "output.weight"],
            "duplicate transform",
        ),
        (vec!["blk.0.attn_q.weight"], "is absent"),
        (
            vec!["blk.0.ssm_alpha.weight"],
            "not a verified projection role",
        ),
        (vec!["token_embd.weight"], "not a verified projection role"),
        (
            vec!["blk.not-a-layer.attn_q.weight"],
            "not a verified projection role",
        ),
    ] {
        let mut meta = metadata();
        meta.insert("prism.hadamard.weight_names".into(), strings(&names));
        rejects(&meta, &tensors, expected);
    }
    let mut meta = metadata();
    meta.insert(
        "prism.hadamard.inverse_weight_names".into(),
        strings(&["output.weight"]),
    );
    rejects(&meta, &tensors, "not a token embedding lookup");
    meta.insert(
        "prism.hadamard.inverse_weight_names".into(),
        strings(&["token_embd.weight", "token_embd.weight"]),
    );
    rejects(
        &meta,
        &[("output.weight", 8, 0), ("token_embd.weight", 8, 0)],
        "duplicate transform",
    );
}

#[test]
fn supported_raw_dtype_does_not_bypass_missing_execution_transforms() {
    let bytes = fixture(&metadata(), &[("output.weight", 8, 0)]);
    let artifact = file(&bytes);
    assert!(NativeGgufFile::open(artifact.path())
        .unwrap()
        .hadamard()
        .is_some());
    assert!(GgufWeightComponentSource::open(artifact.path())
        .err()
        .unwrap()
        .to_string()
        .contains("refusing to ignore"));
    assert!(GgufFile::open(artifact.path())
        .err()
        .unwrap()
        .to_string()
        .contains("refusing to ignore"));
}

#[test]
fn pq2_1024_rotation_is_recognized_but_q2_0_is_not_misidentified_as_pq2() {
    let mut meta = metadata();
    meta.insert("prism.hadamard.block_size".into(), Value::U32(1024));
    explicit(&mut meta, &[1024], &vec![1; 1024]);
    let parsed = inventory(&meta, &[("output.weight", 1024, 142)]).unwrap();
    assert_eq!(parsed.hadamard.unwrap().block_size(), 1024);
    assert_eq!(parsed.tensors[0].bytes_per_block, 34);
    rejects(&meta, &[("output.weight", 1024, 42)], "missing block ABIs");
}

#[test]
fn retained_transform_arrays_reject_nested_types_and_truncation() {
    let mut meta = metadata();
    meta.insert(
        "prism.hadamard.weight_names".into(),
        Value::Array(vec![strings(&["output.weight"])]),
    );
    rejects(
        &meta,
        &[("output.weight", 8, 0)],
        "arrays must contain int32 or strings",
    );
    let bytes = fixture(&metadata(), &[("output.weight", 8, 0)]);
    for length in [24, bytes.len() / 2] {
        assert!(
            GgufInventory::read(&mut Cursor::new(&bytes[..length]), bytes.len() as u64).is_err()
        );
    }
}

fn source_schema(native: &NativeGgufFile) -> WeightSchema {
    let mut components = Vec::new();
    let mut tensors = Vec::new();
    for (index, name) in native.tensor_names().enumerate() {
        let info = native.tensor_info(name).unwrap();
        assert_eq!(info.ggml_type, 0);
        let id: ferrum_interfaces::vnext::WeightId =
            format!("component.values.{index}").try_into().unwrap();
        let mut layout = PhysicalWeightLayout::Dense {
            component_id: id.clone(),
        };
        if let Some(transform) = native
            .hadamard()
            .and_then(|metadata| hadamard_transform_spec(metadata, name).unwrap())
        {
            layout = PhysicalWeightLayout::Hadamard {
                values: Box::new(layout),
                transform,
            };
        }
        components.push(WeightComponentSpec {
            id,
            role: WeightComponentRole::Values,
            external_names: vec![name.into()],
            dimensions: info.dimensions.clone(),
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        });
        tensors.push(WeightTensorSpec {
            id: format!("weight.{index}").try_into().unwrap(),
            dimensions: info.dimensions.clone(),
            logical_element_type: ElementType::F32,
            physical_layout: layout,
            required: true,
        });
    }
    if let Some(metadata) = native.hadamard() {
        if matches!(metadata.signs(), GgufHadamardSigns::Explicit(_)) {
            let widths: std::collections::BTreeSet<_> = metadata
                .weights()
                .values()
                .map(|weight| weight.input_width())
                .collect();
            for width in widths {
                components.push(hadamard_sign_component(width).unwrap());
            }
        }
    }
    WeightSchema {
        format_id: "format.fixture.gguf".to_owned().try_into().unwrap(),
        layout_id: "layout.fixture.gguf".to_owned().try_into().unwrap(),
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    }
}

#[test]
fn schema_binding_registers_read_only_f32_signs_once_per_width() {
    let mut meta = metadata();
    meta.insert(
        "prism.hadamard.weight_names".into(),
        strings(&["output.weight", "blk.0.attn_q.weight"]),
    );
    explicit(&mut meta, &[8], &[1, -1, 1, -1, -1, 1, 1, -1]);
    let artifact = file(&fixture(
        &meta,
        &[("output.weight", 8, 0), ("blk.0.attn_q.weight", 8, 0)],
    ));
    let native = NativeGgufFile::open(artifact.path()).unwrap();
    let schema = source_schema(&native);
    let source =
        GgufWeightComponentSource::open_with_schema(artifact.path(), &schema, native.hadamard())
            .unwrap();
    let signs: Vec<_> = schema
        .components
        .iter()
        .filter(|component| component.role == WeightComponentRole::TransformSigns)
        .collect();
    assert_eq!(signs.len(), 1);
    assert_eq!(signs[0].physical_bytes().unwrap(), 8 * 4);
    let first = source.component(signs[0]).unwrap();
    let second = source.component(signs[0]).unwrap();
    assert_eq!(first.bytes().as_ptr(), second.bytes().as_ptr());
    let values: Vec<_> = first
        .bytes()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(values, [1., -1., 1., -1., -1., 1., 1., -1.]);
    let mut forged = signs[0].clone();
    forged.external_names[0] = "output.weight".into();
    assert!(source.component(&forged).is_err());
}

#[test]
fn source_binding_rejects_omitted_transforms_wrong_direction_and_changed_sign_identity() {
    let mut meta = metadata();
    explicit(&mut meta, &[8], &[1; 8]);
    let artifact = file(&fixture(&meta, &[("output.weight", 8, 0)]));
    let native = NativeGgufFile::open(artifact.path()).unwrap();
    let schema = source_schema(&native);
    let mut omitted = schema.clone();
    let PhysicalWeightLayout::Hadamard { values, .. } = &omitted.tensors[0].physical_layout else {
        panic!("missing transform");
    };
    omitted.tensors[0].physical_layout = *values.clone();
    omitted
        .components
        .retain(|component| component.role != WeightComponentRole::TransformSigns);
    assert!(GgufWeightComponentSource::open_with_schema(
        artifact.path(),
        &omitted,
        native.hadamard()
    )
    .is_err());
    let mut wrong_direction = schema.clone();
    let PhysicalWeightLayout::Hadamard { transform, .. } =
        &mut wrong_direction.tensors[0].physical_layout
    else {
        unreachable!()
    };
    transform.application = ferrum_interfaces::vnext::HadamardApplication::AfterEmbeddingLookup;
    assert!(GgufWeightComponentSource::open_with_schema(
        artifact.path(),
        &wrong_direction,
        native.hadamard()
    )
    .is_err());
    let mut changed = serde_json::to_value(native.hadamard().unwrap()).unwrap();
    changed["signs"]["by_width"]["8"][0] = serde_json::json!(-1);
    let changed = serde_json::from_value(changed).unwrap();
    assert!(
        GgufWeightComponentSource::open_with_schema(artifact.path(), &schema, Some(&changed))
            .is_err()
    );
    assert!(GgufWeightComponentSource::open_with_schema(artifact.path(), &schema, None).is_err());
}

#[test]
fn restored_metadata_revalidates_geometry_widths_and_signs() {
    let mut meta = metadata();
    explicit(&mut meta, &[8], &[1; 8]);
    let parsed = inventory(&meta, &[("output.weight", 8, 0)])
        .unwrap()
        .hadamard
        .unwrap();
    let dimensions = [1, 8];
    parsed
        .validate_for_tensors(
            "qwen35",
            [("output.weight", dimensions.as_slice())].into_iter(),
            None,
        )
        .unwrap();
    for (field, value) in [
        ("version", serde_json::json!(2)),
        ("block_size", serde_json::json!(0)),
    ] {
        let mut changed = serde_json::to_value(&parsed).unwrap();
        changed[field] = value;
        let changed: ferrum_quantization::gguf::GgufHadamard =
            serde_json::from_value(changed).unwrap();
        assert!(changed
            .validate_for_tensors(
                "qwen35",
                [("output.weight", dimensions.as_slice())].into_iter(),
                None
            )
            .is_err());
    }
    let mut changed = serde_json::to_value(&parsed).unwrap();
    changed["weights"]["output.weight"]["input_width"] = serde_json::json!(16);
    let changed: ferrum_quantization::gguf::GgufHadamard = serde_json::from_value(changed).unwrap();
    assert!(changed
        .validate_for_tensors(
            "qwen35",
            [("output.weight", dimensions.as_slice())].into_iter(),
            None
        )
        .is_err());
}
