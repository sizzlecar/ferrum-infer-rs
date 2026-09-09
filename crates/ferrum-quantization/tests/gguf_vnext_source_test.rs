use std::io::{Cursor, Write};

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use ferrum_interfaces::vnext::{
    BlockQuantizationSpec, ElementType, WeightComponentRole, WeightComponentSource,
    WeightComponentSpec, WeightEncoding, WeightId,
};
use ferrum_quantization::GgufWeightComponentSource;
use half::f16;

struct NativeTensorFixture<'a> {
    name: &'a str,
    ggml_type: u32,
    dimensions: &'a [u64],
    payload: &'a [u8],
}

fn native_fixture(tensors: &[NativeTensorFixture<'_>]) -> tempfile::NamedTempFile {
    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    let mut header = b"GGUF".to_vec();
    header.extend_from_slice(&3_u32.to_le_bytes());
    header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&2_u64.to_le_bytes());
    string(&mut header, "general.architecture");
    header.extend_from_slice(&8_u32.to_le_bytes());
    string(&mut header, "fixture");
    string(&mut header, "general.quantization_version");
    header.extend_from_slice(&4_u32.to_le_bytes());
    header.extend_from_slice(&2_u32.to_le_bytes());
    let mut payload = Vec::new();
    for tensor in tensors {
        payload.resize(payload.len().div_ceil(32) * 32, 0);
        string(&mut header, tensor.name);
        header.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dimension in tensor.dimensions.iter().rev() {
            header.extend_from_slice(&dimension.to_le_bytes());
        }
        header.extend_from_slice(&tensor.ggml_type.to_le_bytes());
        header.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        payload.extend_from_slice(tensor.payload);
    }
    header.resize(header.len().div_ceil(32) * 32, 0);
    header.extend_from_slice(&payload);
    let mut file = tempfile::NamedTempFile::with_suffix(".gguf").unwrap();
    file.write_all(&header).unwrap();
    file.flush().unwrap();
    file
}

#[test]
fn native_iq_source_preserves_complete_physical_encodings_and_retained_payloads() {
    use ferrum_quantization::gguf::gguf_weight_encoding;

    let iq3 = (0..220).map(|n| n as u8).collect::<Vec<_>>();
    let iq4_nl = (0..36).map(|n| 255 - n as u8).collect::<Vec<_>>();
    let iq4_xs = (0..272).map(|n| (n * 7) as u8).collect::<Vec<_>>();
    let tensors = [
        NativeTensorFixture {
            name: "iq3.weight",
            ggml_type: 21,
            dimensions: &[2, 256],
            payload: &iq3,
        },
        NativeTensorFixture {
            name: "iq4_nl.weight",
            ggml_type: 20,
            dimensions: &[2, 32],
            payload: &iq4_nl,
        },
        NativeTensorFixture {
            name: "iq4_xs.weight",
            ggml_type: 23,
            dimensions: &[2, 256],
            payload: &iq4_xs,
        },
    ];
    let file = native_fixture(&tensors);
    let source = GgufWeightComponentSource::open(file.path()).unwrap();
    assert_eq!(source.file().quantization_version(), Some(2));
    assert_eq!(source.file().tensor_count(), tensors.len());
    let mut retained = Vec::new();
    for tensor in tensors {
        let info = source.file().tensor_info(tensor.name).unwrap();
        assert_eq!(info.ggml_type, tensor.ggml_type);
        assert_eq!(info.dimensions, tensor.dimensions);
        let component = component(
            "component.iq",
            tensor.name,
            vec![2, 1],
            gguf_weight_encoding(tensor.ggml_type).unwrap(),
        );
        let payload = source.component(&component).unwrap();
        assert_eq!(payload.element_type(), ElementType::U8);
        assert_eq!(payload.bytes(), tensor.payload);
        assert_eq!(
            payload.bytes().as_ptr(),
            source
                .file()
                .tensor_byte_slice(tensor.name)
                .unwrap()
                .as_ptr()
        );
        retained.push((
            payload.retained_host_memory().unwrap().clone(),
            tensor.payload.to_vec(),
        ));
        let mut wrong_abi = component.clone();
        if let WeightEncoding::BlockQuantized(spec) = &mut wrong_abi.encoding {
            spec.bytes_per_block += 1;
        }
        assert!(source.component(&wrong_abi).is_err());
    }
    drop(source);
    for (region, expected) in retained {
        assert_eq!(region.bytes(), expected);
    }
}

#[test]
fn native_source_rejects_truncated_payload_and_unknown_tensor_type() {
    let file = native_fixture(&[NativeTensorFixture {
        name: "truncated.weight",
        ggml_type: 23,
        dimensions: &[1, 256],
        payload: &[0; 135],
    }]);
    assert!(GgufWeightComponentSource::open(file.path()).is_err());
    let file = native_fixture(&[NativeTensorFixture {
        name: "unknown.weight",
        ggml_type: u32::MAX,
        dimensions: &[1, 256],
        payload: &[0; 136],
    }]);
    assert!(GgufWeightComponentSource::open(file.path()).is_err());
}

fn component(
    id: &str,
    external_name: &str,
    dimensions: Vec<u64>,
    encoding: WeightEncoding,
) -> WeightComponentSpec {
    WeightComponentSpec {
        id: WeightId::new(id).unwrap(),
        role: if matches!(encoding, WeightEncoding::BlockQuantized(_)) {
            WeightComponentRole::PackedValues
        } else {
            WeightComponentRole::Values
        },
        external_names: vec![external_name.to_owned()],
        dimensions,
        encoding,
        required: true,
    }
}

fn build_gguf() -> tempfile::NamedTempFile {
    let device = Device::Cpu;
    let dense =
        Tensor::from_vec((0..8).map(|value| value as f32).collect(), (2, 4), &device).unwrap();
    let dense = QTensor::quantize(&dense, GgmlDType::F32).unwrap();
    let quantized = Tensor::from_vec(
        (0..512)
            .map(|value| ((value % 17) as f32 - 8.0) * 0.125)
            .collect(),
        (2, 256),
        &device,
    )
    .unwrap();
    let quantized = QTensor::quantize(&quantized, GgmlDType::Q4K).unwrap();
    let architecture = Value::String("test".to_owned());
    let metadata = vec![("general.architecture", &architecture)];
    let tensors = vec![("dense.weight", &dense), ("quantized.weight", &quantized)];
    let mut bytes = Vec::new();
    gguf_file::write(&mut Cursor::new(&mut bytes), &metadata, &tensors).unwrap();
    let mut file = tempfile::NamedTempFile::with_suffix(".gguf").unwrap();
    file.write_all(&bytes).unwrap();
    file.flush().unwrap();
    file
}

#[test]
fn mmap_source_returns_exact_dense_and_q4_k_payloads() {
    let file = build_gguf();
    let source = GgufWeightComponentSource::open(file.path()).unwrap();

    let dense = component(
        "component.dense",
        "dense.weight",
        vec![2, 4],
        WeightEncoding::Dense {
            element_type: ElementType::F32,
        },
    );
    let dense_payload = source.component(&dense).unwrap();
    assert_eq!(dense_payload.dimensions(), [2, 4]);
    assert_eq!(dense_payload.element_type(), ElementType::F32);
    assert_eq!(dense_payload.bytes().len(), 32);
    assert!(dense_payload.retained_host_memory().is_some());

    let quantized = component(
        "component.q4-k",
        "quantized.weight",
        vec![2, 1],
        WeightEncoding::BlockQuantized(BlockQuantizationSpec {
            format_id: "quantization.gguf.q4-k".to_owned().try_into().unwrap(),
            logical_values_per_block: 256,
            bytes_per_block: 144,
        }),
    );
    let payload = source.component(&quantized).unwrap();
    assert_eq!(payload.dimensions(), [2, 1]);
    assert_eq!(payload.element_type(), ElementType::U8);
    assert_eq!(payload.bytes().len(), 288);
    assert!(payload.retained_host_memory().is_some());
    assert_eq!(
        payload.bytes().as_ptr(),
        source
            .file()
            .tensor_byte_slice("quantized.weight")
            .unwrap()
            .as_ptr()
    );
    assert_eq!(payload.external_name(), "quantized.weight");
    assert_eq!(payload.source_file(), source.source_file());
}

#[test]
fn dense_source_materializes_the_requested_typed_float_payload() {
    let file = build_gguf();
    let source = GgufWeightComponentSource::open(file.path()).unwrap();
    let dense = component(
        "component.dense-f16",
        "dense.weight",
        vec![2, 4],
        WeightEncoding::Dense {
            element_type: ElementType::F16,
        },
    );

    let payload = source.component(&dense).unwrap();
    let values = payload
        .bytes()
        .chunks_exact(2)
        .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
        .collect::<Vec<_>>();

    assert_eq!(payload.element_type(), ElementType::F16);
    assert_eq!(payload.dimensions(), [2, 4]);
    assert_eq!(payload.bytes().len(), 16);
    assert!(payload.retained_host_memory().is_none());
    assert_eq!(values, (0..8).map(|value| value as f32).collect::<Vec<_>>());
    assert_ne!(
        payload.bytes().as_ptr(),
        source
            .file()
            .tensor_byte_slice("dense.weight")
            .unwrap()
            .as_ptr()
    );
}

#[test]
fn mmap_source_rejects_dtype_abi_shape_and_tensor_fusion_drift() {
    let file = build_gguf();
    let source = GgufWeightComponentSource::open(file.path()).unwrap();
    let mut quantized = component(
        "component.q4-k",
        "quantized.weight",
        vec![2, 1],
        WeightEncoding::BlockQuantized(BlockQuantizationSpec {
            format_id: "quantization.gguf.q4-k".to_owned().try_into().unwrap(),
            logical_values_per_block: 256,
            bytes_per_block: 144,
        }),
    );

    if let WeightEncoding::BlockQuantized(spec) = &mut quantized.encoding {
        spec.format_id = "quantization.gguf.q6-k".to_owned().try_into().unwrap();
    }
    assert!(source.component(&quantized).is_err());

    if let WeightEncoding::BlockQuantized(spec) = &mut quantized.encoding {
        spec.format_id = "quantization.gguf.q4-k".to_owned().try_into().unwrap();
    }
    quantized.dimensions = vec![1, 1];
    assert!(source.component(&quantized).is_err());

    quantized.dimensions = vec![1, 2];
    assert!(source.component(&quantized).is_err());

    quantized.dimensions = vec![2, 1];
    quantized.external_names.push("dense.weight".to_owned());
    assert!(source.component(&quantized).is_err());
}

#[test]
#[ignore = "requires FERRUM_TEST_GGUF_PATH to point at Qwen3.5-4B-Q4_K_M.gguf"]
fn real_qwen35_q4_k_m_preserves_mixed_tensor_abis() {
    let path = std::env::var("FERRUM_TEST_GGUF_PATH").expect("FERRUM_TEST_GGUF_PATH");
    let source = GgufWeightComponentSource::open(path).unwrap();
    assert_eq!(source.file().architecture().unwrap(), "qwen35");
    assert_eq!(source.file().tensor_count(), 426);

    let cases = [
        (
            component(
                "component.attn-qkv",
                "blk.0.attn_qkv.weight",
                vec![8192, 10],
                WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                    format_id: "quantization.gguf.q5-k".to_owned().try_into().unwrap(),
                    logical_values_per_block: 256,
                    bytes_per_block: 176,
                }),
            ),
            14_417_920,
        ),
        (
            component(
                "component.ssm-beta",
                "blk.0.ssm_beta.weight",
                vec![32, 80],
                WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                    format_id: "quantization.gguf.q8-0".to_owned().try_into().unwrap(),
                    logical_values_per_block: 32,
                    bytes_per_block: 34,
                }),
            ),
            87_040,
        ),
        (
            component(
                "component.output-norm",
                "output_norm.weight",
                vec![2560],
                WeightEncoding::Dense {
                    element_type: ElementType::F32,
                },
            ),
            10_240,
        ),
    ];

    for (component, expected_bytes) in cases {
        let tensor_name = &component.external_names[0];
        let payload = source.component(&component).unwrap();
        let mmap_payload = source.file().tensor_byte_slice(tensor_name).unwrap();
        assert_eq!(payload.bytes().len(), expected_bytes);
        assert_eq!(payload.bytes().as_ptr(), mmap_payload.as_ptr());
    }
}
