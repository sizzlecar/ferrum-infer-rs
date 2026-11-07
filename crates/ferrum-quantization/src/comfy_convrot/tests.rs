use super::*;
use safetensors::tensor::{serialize_to_file, TensorView};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn tensor<'a>(dtype: Dtype, shape: &'a [u64], bytes: &'a [u8]) -> ComfyQuantTensor<'a> {
    ComfyQuantTensor::new(dtype, shape, bytes).unwrap()
}

fn w4_metadata() -> ComfyQuantizationMetadata {
    ComfyQuantizationMetadata::CodebookW4A8 {
        group_size: 16,
        convrot: true,
        convrot_groupsize: 16,
    }
}

struct W4Fixture {
    packed: Vec<u8>,
    relative: Vec<u8>,
    channel: Vec<u8>,
    codebook: Vec<u8>,
}
impl W4Fixture {
    fn new() -> Self {
        Self {
            packed: vec![0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe],
            relative: vec![0x38], // E4M3FN 1.
            channel: f32_bytes(&[0.5]),
            codebook: f32_bytes(&[
                -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -200.0, 200.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                11.0,
            ]),
        }
    }
    fn matrix(&self) -> Result<ComfyQuantizedMatrix<'_>> {
        ComfyQuantizedMatrix::new(
            w4_metadata(),
            tensor(Dtype::I8, &[1, 8], &self.packed),
            ComfyQuantizationComponents::CodebookW4A8 {
                relative_scale: tensor(Dtype::F8_E4M3, &[1, 1], &self.relative),
                channel_scale: tensor(Dtype::F32, &[1], &self.channel),
                codebook: tensor(Dtype::F32, &[16], &self.codebook),
            },
        )
    }
}

#[test]
fn metadata_matches_real_checkpoint_and_rejects_unsupported_or_ambiguous_fields() {
    let observed = br#"{"format": "asym_w4a8_int8", "group_size": 16, "convrot": true, "convrot_groupsize": 256}"#;
    let parsed = ComfyQuantizationMetadata::parse(observed).unwrap();
    assert_eq!(parsed.rotation_group_size(), Some(256));
    assert_eq!(
        ComfyQuantizationMetadata::parse(&serde_json::to_vec(&parsed).unwrap()).unwrap(),
        parsed,
    );
    for bad in [
        r#"{"format":"asym_w4a8_int8","group_size":16,"convrot":false,"convrot_groupsize":256}"#,
        r#"{"format":"asym_w4a8_int8","group_size":3,"convrot":true,"convrot_groupsize":256}"#,
        r#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":8}"#,
        r#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":0}"#,
        r#"{"format":"int8_tensorwise","transposed":true}"#,
        r#"{"format":"int8_tensorwise","convrot":true,"convrot":false}"#,
        r#"{"format":"int8_tensorwise","format":"asym_w4a8_int8"}"#,
        r#"{"format":"w6a8_int8","group_size":32}"#,
    ] {
        assert!(
            ComfyQuantizationMetadata::parse(bad.as_bytes()).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn nibble_order_int8_grid_rounding_and_saturation_are_explicit() {
    let fixture = W4Fixture::new();
    let matrix = fixture.matrix().unwrap();
    assert_eq!((matrix.rows(), matrix.columns()), (1, 16));
    assert_eq!(
        matrix.decoded_rotated_i8_reference().unwrap(),
        [-2, -2, 0, 0, 2, 2, -127, 127, 4, 5, 6, 7, 8, 9, 10, 11],
    );
    // Payload stays packed and borrowed; decoding does not rewrite the source.
    assert_eq!(matrix.weight().bytes().as_ptr(), fixture.packed.as_ptr());
    assert_eq!(matrix.weight().bytes()[7], 0xfe);
}

#[test]
fn relative_scale_and_codebook_payload_control_the_decoded_grid() {
    let mut fixture = W4Fixture::new();
    fixture.relative[0] = 0x40; // E4M3FN 2, not an exponent-only scale.
    fixture.codebook = f32_bytes(&[1.25; 16]);
    assert_eq!(
        fixture
            .matrix()
            .unwrap()
            .decoded_rotated_i8_reference()
            .unwrap(),
        vec![2; 16]
    );
    fixture.relative[0] = 0; // Representable scale underflow yields a zero group.
    assert_eq!(
        fixture
            .matrix()
            .unwrap()
            .decoded_rotated_i8_reference()
            .unwrap(),
        vec![0; 16]
    );
}

#[test]
fn observed_h3_codebook_and_first_packed_group_decode_through_int8_grid() {
    // A reduced first-group fixture from blocks.0.attn.out_proj, at the pinned
    // model revision in the module documentation. The full matrix is not a
    // fixture: [5376,3584] packed bytes, s_rel [5376,448], logical K=7168.
    // This test checks canonical decoding only, before inverse ConvRot.
    let fixture = W4Fixture {
        packed: vec![155, 90, 115, 184, 212, 108, 86, 112],
        relative: vec![103], // E4M3FN 60.
        channel: f32_bytes(&[0.0034406516]),
        codebook: f32_bytes(&[
            -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
            0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
        ]),
    };
    assert_eq!(
        fixture
            .matrix()
            .unwrap()
            .decoded_rotated_i8_reference()
            .unwrap(),
        [23, 9, 16, -16, -30, -3, 3, 23, -23, 38, 30, -9, -9, -16, -59, -3,]
    );
}

#[test]
fn e4m3fn_handles_subnormal_max_finite_and_nan_encodings() {
    assert_eq!(decode_e4m3fn(0x01), 1.0 / 512.0);
    assert_eq!(decode_e4m3fn(0x38), 1.0);
    assert_eq!(decode_e4m3fn(0x7e), 448.0);
    assert_eq!(decode_e4m3fn(0xfe), -448.0);
    for invalid_scale in [0x7f, 0xff, 0x80, 0xb8] {
        let mut fixture = W4Fixture::new();
        fixture.relative[0] = invalid_scale;
        assert!(fixture.matrix().is_err());
    }
}

#[test]
fn mismatched_component_shapes_formats_and_nonfinite_values_are_rejected() {
    assert!(ComfyQuantTensor::new(Dtype::I8, &[2, 2], &[0; 3]).is_err());
    assert!(ComfyQuantTensor::new(Dtype::F32, &[u64::MAX, 2], &[0; 4]).is_err());
    assert!(ComfyQuantTensor::new(Dtype::I8, &[0, 4], &[]).is_err());
    let fixture = W4Fixture::new();
    assert!(ComfyQuantizedMatrix::new(
        w4_metadata(),
        tensor(Dtype::I8, &[1, 8], &fixture.packed),
        ComfyQuantizationComponents::Int8 {
            scale: tensor(Dtype::F32, &[1], &fixture.channel)
        },
    )
    .is_err());
    assert!(ComfyQuantizedMatrix::new(
        w4_metadata(),
        tensor(Dtype::I8, &[1, 8], &fixture.packed),
        ComfyQuantizationComponents::CodebookW4A8 {
            relative_scale: tensor(Dtype::U8, &[1, 1], &fixture.relative),
            channel_scale: tensor(Dtype::F32, &[1], &fixture.channel),
            codebook: tensor(Dtype::F32, &[16], &fixture.codebook),
        },
    )
    .is_err());
    for bad in [0.0, -1.0, f32::INFINITY, f32::NAN] {
        let mut fixture = W4Fixture::new();
        fixture.channel = f32_bytes(&[bad]);
        assert!(fixture.matrix().is_err());
    }
    let mut fixture = W4Fixture::new();
    fixture.codebook = f32_bytes(&[f32::NAN; 16]);
    assert!(fixture.matrix().is_err());
}

fn direct_hadamard_reference(row: &[i8]) -> Vec<f32> {
    const H4: [[f32; 4]; 4] = [
        [1.0, 1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, 1.0],
    ];
    (0..row.len())
        .map(|column| {
            row.iter()
                .enumerate()
                .map(|(index, value)| {
                    let (mut i, mut j, mut sign) = (index, column, 1.0);
                    for _ in 0..row.len().trailing_zeros() / 2 {
                        sign *= H4[i % 4][j % 4];
                        i /= 4;
                        j /= 4;
                    }
                    f32::from(*value) * sign
                })
                .sum::<f32>()
                / (row.len() as f32).sqrt()
        })
        .collect()
}

#[test]
fn convrot_matches_regular_kronecker_matrix_and_preserves_channel_scales() {
    for group in [4, 16, 64, 256] {
        let values: Vec<i8> = (0..group).map(|index| (index % 7) as i8 - 3).collect();
        let bytes: Vec<u8> = values.iter().map(|value| *value as u8).collect();
        let scale = f32_bytes(&[1.0]);
        let shape = [1, group as u64];
        let matrix = ComfyQuantizedMatrix::new(
            ComfyQuantizationMetadata::Int8Tensorwise {
                convrot: true,
                convrot_groupsize: group,
            },
            tensor(Dtype::I8, &shape, &bytes),
            ComfyQuantizationComponents::Int8 {
                scale: tensor(Dtype::F32, &[], &scale),
            },
        )
        .unwrap();
        assert_eq!(
            matrix.dequantize_f32_reference().unwrap(),
            direct_hadamard_reference(&values)
        );
    }
    let bytes = [1, 0, 0, 0, 0, 1, 0, 0];
    let scales = f32_bytes(&[2.0, 0.5]);
    let matrix = ComfyQuantizedMatrix::new(
        ComfyQuantizationMetadata::Int8Tensorwise {
            convrot: true,
            convrot_groupsize: 4,
        },
        tensor(Dtype::I8, &[2, 4], &bytes),
        ComfyQuantizationComponents::Int8 {
            scale: tensor(Dtype::F32, &[2, 1], &scales),
        },
    )
    .unwrap();
    assert_eq!(
        matrix.dequantize_f32_reference().unwrap(),
        [1.0, 1.0, 1.0, -1.0, 0.25, 0.25, -0.25, 0.25]
    );
}

#[test]
fn w4_reference_applies_channel_scale_before_inverse_rotation() {
    let fixture = W4Fixture::new();
    let matrix = fixture.matrix().unwrap();
    let expected: Vec<f32> =
        direct_hadamard_reference(&[-2, -2, 0, 0, 2, 2, -127, 127, 4, 5, 6, 7, 8, 9, 10, 11])
            .into_iter()
            .map(|value| value * 0.5)
            .collect();
    assert_eq!(matrix.dequantize_f32_reference().unwrap(), expected);
}

#[test]
fn canonical_archive_suffixes_are_loaded_without_rewriting_source_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = W4Fixture::new();
    let native_metadata = ComfyQuantizationMetadata::CodebookW4A8 {
        group_size: 16,
        convrot: true,
        convrot_groupsize: 256,
    };
    let metadata = serde_json::to_vec(&native_metadata).unwrap();
    let packed = fixture.packed.repeat(16);
    let relative = fixture.relative.repeat(16);
    let tensors = [
        (
            "layer.comfy_quant",
            TensorView::new(Dtype::U8, vec![metadata.len()], &metadata).unwrap(),
        ),
        (
            "layer.weight",
            TensorView::new(Dtype::I8, vec![1, 128], &packed).unwrap(),
        ),
        (
            "layer.weight_s_rel",
            TensorView::new(Dtype::F8_E4M3, vec![1, 16], &relative).unwrap(),
        ),
        (
            "layer.weight_s_channel",
            TensorView::new(Dtype::F32, vec![1], &fixture.channel).unwrap(),
        ),
        (
            "layer.weight_codebook",
            TensorView::new(Dtype::F32, vec![16], &fixture.codebook).unwrap(),
        ),
    ];
    serialize_to_file(tensors, &None, &directory.path().join("model.safetensors")).unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    let parsed = ComfyQuantizedMatrix::from_archive(&archive, "layer").unwrap();
    assert_eq!(parsed.weight().bytes(), packed);
    assert_eq!(
        parsed.decoded_rotated_i8_reference().unwrap(),
        fixture
            .matrix()
            .unwrap()
            .decoded_rotated_i8_reference()
            .unwrap()
            .repeat(16)
    );
    assert!(ComfyQuantizedMatrix::from_archive(&archive, "missing").is_err());
    let source = ComfyCanonicalWeightSource::new(archive).unwrap();
    assert_eq!(source.modules().get("layer"), Some(&native_metadata));
    use ferrum_runtime::{
        ElementType, WeightComponentRole, WeightComponentSource, WeightComponentSpec,
        WeightEncoding, WeightId,
    };
    let mut component = WeightComponentSpec {
        id: WeightId::new("weight.layer.relative_scales").unwrap(),
        role: WeightComponentRole::Scales,
        external_names: vec!["layer.weight_s_rel".into()],
        dimensions: vec![1, 16],
        encoding: WeightEncoding::Dense {
            element_type: ElementType::U8,
        },
        required: true,
    };
    let payload = source.component(&component).unwrap();
    assert_eq!(payload.bytes(), relative.as_slice());
    assert!(payload.retained_host_memory().is_some());
    assert!(payload
        .retained_host_memory()
        .unwrap()
        .readonly_file_mapping()
        .is_some());
    // Standalone raw scale-carrier ports use a dense U8 logical tensor, whose
    // storage leaf requires Values. It still retains E4M3FN source bytes.
    component.role = WeightComponentRole::Values;
    let schema = ferrum_runtime::WeightSchema {
        format_id: ferrum_runtime::WeightFormatId::new(COMFY_CONVROT_CANONICAL_WEIGHT_FORMAT_ID)
            .unwrap(),
        layout_id: ferrum_runtime::WeightLayoutId::new("weight-layout.comfy.relative.test")
            .unwrap(),
        version: ferrum_runtime::ContractVersion::new(1, 0),
        components: vec![component.clone()],
        tensors: vec![ferrum_runtime::WeightTensorSpec {
            id: WeightId::new("weight.relative.port").unwrap(),
            dimensions: vec![1, 16],
            logical_element_type: ElementType::U8,
            physical_layout: ferrum_runtime::PhysicalWeightLayout::Dense {
                component_id: component.id.clone(),
            },
            required: true,
        }],
    };
    schema
        .validate(&ferrum_runtime::ModelFamilyId::new("family.comfy.relative.test").unwrap())
        .unwrap();
    assert_eq!(
        source.component(&component).unwrap().bytes(),
        relative.as_slice()
    );
    component.encoding = WeightEncoding::Dense {
        element_type: ElementType::Bf16,
    };
    assert!(source.component(&component).is_err());
    component.encoding = WeightEncoding::Dense {
        element_type: ElementType::U8,
    };
    component.dimensions = vec![1, 2];
    assert!(source.component(&component).is_err());
}

#[test]
fn canonical_int8_source_requires_exact_native_scale_scope_and_rotation() {
    use ferrum_runtime::{
        ElementType, WeightComponentRole, WeightComponentSource, WeightComponentSpec,
        WeightEncoding, WeightId,
    };
    for (convrot, group, shape, accepted) in [
        (true, 256, vec![2, 1], true),
        (true, 256, vec![2], false),
        (true, 256, vec![], false),
        (true, 256, vec![1], false),
        (false, 256, vec![2, 1], false),
        (true, 16, vec![2, 1], false),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let metadata = serde_json::to_vec(&ComfyQuantizationMetadata::Int8Tensorwise {
            convrot,
            convrot_groupsize: group,
        })
        .unwrap();
        let weight = vec![128_u8; 2 * 256];
        let scales: Vec<u8> = std::iter::repeat_n(0.5_f32, shape.iter().product::<usize>())
            .flat_map(f32::to_le_bytes)
            .collect();
        serialize_to_file(
            [
                (
                    "layer.comfy_quant",
                    TensorView::new(Dtype::U8, vec![metadata.len()], &metadata).unwrap(),
                ),
                (
                    "layer.weight",
                    TensorView::new(Dtype::I8, vec![2, 256], &weight).unwrap(),
                ),
                (
                    "layer.weight_scale",
                    TensorView::new(Dtype::F32, shape.clone(), &scales).unwrap(),
                ),
            ],
            &None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let archive = SafetensorsArchive::open(directory.path()).unwrap();
        // Inspection/reference keeps legal upstream broadcasts; the executable
        // canonical source deliberately admits only the declared [N,1] ABI.
        ComfyQuantizedMatrix::from_archive(&archive, "layer").unwrap();
        let source = ComfyCanonicalWeightSource::new(archive);
        assert_eq!(
            source.is_ok(),
            accepted,
            "rotation={convrot}/{group},scale={shape:?}"
        );
        if let Ok(source) = source {
            let component = WeightComponentSpec {
                id: WeightId::new("weight.layer.scale").unwrap(),
                role: WeightComponentRole::Values,
                external_names: vec!["layer.weight_scale".into()],
                dimensions: vec![2, 1],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F32,
                },
                required: true,
            };
            let payload = source.component(&component).unwrap();
            assert_eq!(payload.bytes(), scales);
            assert!(payload.retained_host_memory().is_some());
            assert!(payload
                .retained_host_memory()
                .unwrap()
                .readonly_file_mapping()
                .is_some());
        }
    }
}
