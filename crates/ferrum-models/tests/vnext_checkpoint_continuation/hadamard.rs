use super::*;
use candle_core::quantized::gguf_file::Value as GgufValue;
use ferrum_quantization::gguf::source::hadamard_sign_component;
use ferrum_quantization::gguf::{GgufWeightComponentSource, NativeGgufFile};
use std::io::Write;
use std::num::NonZeroU32;

fn transform(inverse: bool, output: bool) -> HadamardTransformSpec {
    HadamardTransformSpec {
        block_size: NonZeroU32::new(HIDDEN as u32).unwrap(),
        signs: HadamardSigns::Explicit(PhysicalWeightComponentBinding::exact_contiguous(
            hadamard_sign_component(HIDDEN).unwrap().id,
        )),
        application: if inverse {
            HadamardApplication::AfterEmbeddingLookup
        } else {
            HadamardApplication::BeforeMatmul {
                input_permutation: output.then_some(GroupedFeatureTranspose {
                    inner_extent: 64,
                    first_outer_extent: 2,
                    second_outer_extent: 2,
                }),
            }
        },
    }
}

fn pq2(component_id: &str, name: &str, rows: u64) -> WeightComponentSpec {
    WeightComponentSpec {
        id: id(component_id),
        role: WeightComponentRole::PackedValues,
        external_names: vec![name.into()],
        dimensions: vec![rows, HIDDEN / 128],
        encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
            format_id: id("quantization.gguf.pq2-0"),
            logical_values_per_block: 128,
            bytes_per_block: 34,
        }),
        required: true,
    }
}

fn packed_layout(component_id: WeightId, inverse: bool, output: bool) -> PhysicalWeightLayout {
    PhysicalWeightLayout::Hadamard {
        values: Box::new(PhysicalWeightLayout::BlockQuantized {
            blocks: PhysicalWeightComponentBinding::exact_contiguous(component_id),
            block_axis: 1,
            block_padding: PhysicalWeightPadding::Exact,
        }),
        transform: transform(inverse, output),
    }
}

pub(super) fn configure(schema: &mut WeightSchema) {
    for (logical, external, rows, inverse, output) in [
        ("embedding", "token_embd.weight", VOCAB, true, false),
        ("o", "blk.0.ssm_out.weight", HIDDEN, false, true),
    ] {
        let component_id = format!("component.{logical}");
        let component = schema
            .components
            .iter_mut()
            .find(|component| component.id.as_str() == component_id)
            .unwrap();
        *component = pq2(&component_id, external, rows);
        let tensor = schema
            .tensors
            .iter_mut()
            .find(|tensor| tensor.id.as_str() == format!("weight.{logical}"))
            .unwrap();
        tensor.physical_layout = packed_layout(component.id.clone(), inverse, output);
    }
    schema
        .components
        .retain(|component| component.id.as_str() != "component.qkvzba");
    let mut parts = Vec::new();
    let mut offset = 0;
    // Lexical component order deliberately differs from projection order, so
    // resolved component indexing cannot accidentally stand in for this ABI.
    for (component_id, name, rows, rotated) in [
        ("component.gdn.20-qkv", "blk.0.attn_qkv.weight", 512, true),
        ("component.gdn.10-z", "blk.0.attn_gate.weight", 256, true),
        ("component.gdn.40-b", "blk.0.ssm_beta.weight", 4, false),
        ("component.gdn.30-a", "blk.0.ssm_alpha.weight", 4, false),
    ] {
        let component = if rotated {
            pq2(component_id, name, rows)
        } else {
            WeightComponentSpec {
                id: id(component_id),
                role: WeightComponentRole::Values,
                external_names: vec![name.into()],
                dimensions: vec![rows, HIDDEN],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: true,
            }
        };
        let layout = if rotated {
            packed_layout(component.id.clone(), false, false)
        } else {
            PhysicalWeightLayout::Dense {
                component_id: component.id.clone(),
            }
        };
        parts.push(CompositeWeightPart {
            layout: Box::new(layout),
            logical_offsets: vec![offset, 0],
            extents: vec![rows, HIDDEN],
        });
        schema.components.push(component);
        offset += rows;
    }
    let tensor = schema
        .tensors
        .iter_mut()
        .find(|tensor| tensor.id.as_str() == "weight.qkvzba")
        .unwrap();
    assert_eq!(tensor.dimensions[0], offset);
    tensor.physical_layout = PhysicalWeightLayout::Composite { parts };
    schema
        .components
        .push(hadamard_sign_component(HIDDEN).unwrap());
    schema.version = ContractVersion::new(1, 1);
}

/// Generate a tiny actual GGUF source and bind it through the production source
/// registry. This exercises validated metadata F32 signs and packed retention,
/// rather than letting the fixture invent already-resolved device parameters.
pub(super) fn source(
    schema: &WeightSchema,
    values: &BTreeMap<WeightId, Vec<u8>>,
) -> (tempfile::NamedTempFile, GgufWeightComponentSource) {
    let names = |names: &[&str]| {
        GgufValue::Array(
            names
                .iter()
                .filter(|name| {
                    schema.components.iter().any(|component| {
                        component
                            .external_names
                            .iter()
                            .any(|external| external == **name)
                    })
                })
                .map(|name| GgufValue::String((*name).into()))
                .collect(),
        )
    };
    let metadata = [
        ("general.architecture", GgufValue::String("qwen35".into())),
        ("prism.hadamard.version", GgufValue::U32(1)),
        ("prism.hadamard.block_size", GgufValue::U32(HIDDEN as u32)),
        (
            "prism.hadamard.transform",
            GgufValue::String("normalized-sylvester-walsh-hadamard".into()),
        ),
        (
            "prism.hadamard.axis",
            GgufValue::String("input-last-dimension".into()),
        ),
        (
            "prism.hadamard.sign_mode",
            GgufValue::String("explicit".into()),
        ),
        (
            "prism.hadamard.weight_names",
            names(&[
                "blk.0.attn_qkv.weight",
                "blk.0.attn_gate.weight",
                "blk.0.ssm_out.weight",
            ]),
        ),
        (
            "prism.hadamard.inverse_weight_names",
            names(&["token_embd.weight"]),
        ),
        (
            "prism.hadamard.sign_widths",
            GgufValue::Array(vec![GgufValue::I32(HIDDEN as i32)]),
        ),
        (
            "prism.hadamard.sign_values",
            GgufValue::Array(
                (0..HIDDEN)
                    .map(|index| {
                        GgufValue::I32(if (index * 13 + index / 7) % 5 < 2 {
                            -1
                        } else {
                            1
                        })
                    })
                    .collect(),
            ),
        ),
        ("prism.hadamard.gdn_v_grouped", GgufValue::Bool(true)),
        ("qwen35.ssm.group_count", GgufValue::U32(2)),
        ("qwen35.ssm.time_step_rank", GgufValue::U32(4)),
    ];
    let components: Vec<_> = schema
        .components
        .iter()
        .filter(|component| component.role != WeightComponentRole::TransformSigns)
        .collect();
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&(components.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        string(&mut bytes, key);
        bytes.extend_from_slice(&kind(&value).to_le_bytes());
        value_bytes(&mut bytes, &value);
    }
    let mut offset = 0_u64;
    for component in &components {
        string(&mut bytes, &component.external_names[0]);
        bytes.extend_from_slice(&(component.dimensions.len() as u32).to_le_bytes());
        let mut dims = component.dimensions.clone();
        let dtype = match &component.encoding {
            WeightEncoding::Dense {
                element_type: ElementType::F32,
            } => 0_u32,
            WeightEncoding::Dense {
                element_type: ElementType::F16,
            } => 1,
            WeightEncoding::BlockQuantized(spec) => {
                let (dtype, values_per_block, bytes_per_block) = match spec.format_id.as_str() {
                    "quantization.gguf.pq2-0" => (142, 128, 34),
                    "quantization.gguf.q4-k" => (12, 256, 144),
                    "quantization.gguf.q6-k" => (14, 256, 210),
                    other => panic!("unsupported fixture GGUF block format: {other}"),
                };
                assert_eq!(spec.logical_values_per_block, values_per_block);
                assert_eq!(spec.bytes_per_block, bytes_per_block);
                let blocks = dims.iter().product::<u64>();
                assert_eq!(
                    values[&component.id].len() as u64,
                    blocks.checked_mul(u64::from(bytes_per_block)).unwrap(),
                    "GGUF descriptor and retained source bytes must describe the same blocks"
                );
                // Schema dimensions count blocks on the last axis. GGUF tensor
                // dimensions count decoded values, for PQ2 and K-quants alike.
                let columns = dims.last_mut().unwrap();
                *columns = columns.checked_mul(u64::from(values_per_block)).unwrap();
                dtype
            }
            other => panic!("unexpected fixture encoding: {other:?}"),
        };
        for dim in dims.iter().rev() {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&dtype.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
        offset += (values[&component.id].len() as u64).div_ceil(32) * 32;
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    for component in components {
        bytes.extend_from_slice(&values[&component.id]);
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    }
    let mut file = tempfile::Builder::new().suffix(".gguf").tempfile().unwrap();
    file.write_all(&bytes).unwrap();
    let native = NativeGgufFile::open(file.path()).unwrap();
    let source =
        GgufWeightComponentSource::open_with_schema(file.path(), schema, native.hadamard())
            .unwrap();
    (file, source)
}

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}
fn kind(value: &GgufValue) -> u32 {
    match value {
        GgufValue::U32(_) => 4,
        GgufValue::I32(_) => 5,
        GgufValue::Bool(_) => 7,
        GgufValue::String(_) => 8,
        GgufValue::Array(_) => 9,
        _ => panic!("unsupported fixture metadata"),
    }
}
fn value_bytes(bytes: &mut Vec<u8>, value: &GgufValue) {
    match value {
        GgufValue::U32(value) => bytes.extend_from_slice(&value.to_le_bytes()),
        GgufValue::I32(value) => bytes.extend_from_slice(&value.to_le_bytes()),
        GgufValue::Bool(value) => bytes.push(u8::from(*value)),
        GgufValue::String(value) => string(bytes, value),
        GgufValue::Array(values) => {
            bytes.extend_from_slice(&kind(&values[0]).to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                value_bytes(bytes, value);
            }
        }
        _ => panic!("unsupported fixture metadata"),
    }
}

#[test]
fn hadamard_gguf_fixture_preserves_q4k_and_q6k_descriptors_and_source_bytes() {
    let kind = AttentionKind::GatedDeltaHadamardF16;
    let mut schema = Family::new(kind).weight_schema(&kind).unwrap();
    for (suffix, format, bytes) in [
        ("q4", "quantization.gguf.q4-k", 144),
        ("q6", "quantization.gguf.q6-k", 210),
    ] {
        let component_id: WeightId = id(format!("component.fixture.{suffix}"));
        schema.components.push(WeightComponentSpec {
            id: component_id.clone(),
            role: WeightComponentRole::PackedValues,
            external_names: vec![format!("fixture.{suffix}.weight")],
            dimensions: vec![2, HIDDEN / 256],
            encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                format_id: id(format),
                logical_values_per_block: 256,
                bytes_per_block: bytes,
            }),
            required: true,
        });
        schema.tensors.push(WeightTensorSpec {
            id: id(format!("weight.fixture.{suffix}")),
            dimensions: vec![2, HIDDEN],
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::BlockQuantized {
                blocks: PhysicalWeightComponentBinding::exact_contiguous(component_id),
                block_axis: 1,
                block_padding: PhysicalWeightPadding::Exact,
            },
            required: true,
        });
    }
    // Reopen the actual GGUF file, including its Hadamard metadata. The
    // production source validates dtype and decoded tensor shape.
    let source = Weights::new(&schema);
    assert!(source.source.is_some());
    for component in &schema.components {
        if component.id.as_str().starts_with("component.fixture.") {
            let loaded = source.component(component).unwrap();
            assert_eq!(loaded.bytes(), source.values[&component.id]);
        }
    }
}
