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
fn dense_source_materializes_the_typed_float_payload_once() {
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
