use std::io::Cursor;

use candle_core::quantized::gguf_file::{self, Content, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Shape, Tensor};
use ferrum_quantization::gguf::GgufInventory;

fn fixture() -> Vec<u8> {
    let dense = Tensor::from_vec(vec![1_f32, -2., 3., 4.], (2, 2), &Device::Cpu).unwrap();
    let dense = QTensor::quantize(&dense, GgmlDType::F32).unwrap();
    let matrix = Tensor::from_vec(
        (0..512).map(|n| ((n % 19) as f32 - 9.) / 4.).collect(),
        (2, 256),
        &Device::Cpu,
    )
    .unwrap();
    let matrix = QTensor::quantize(&matrix, GgmlDType::Q4K).unwrap();
    let architecture = Value::String("fixture".into());
    let quantization_version = Value::U32(2);
    let mut bytes = Cursor::new(Vec::new());
    gguf_file::write(
        &mut bytes,
        &[
            ("general.architecture", &architecture),
            ("general.quantization_version", &quantization_version),
        ],
        &[("z_norm.weight", &dense), ("a_projection.weight", &matrix)],
    )
    .unwrap();
    bytes.into_inner()
}

#[test]
fn inventories_mixed_payloads_and_complete_header_prefix_identically() {
    let bytes = fixture();
    let full = GgufInventory::read(&mut Cursor::new(&bytes), bytes.len() as u64).unwrap();
    let prefix = &bytes[..full.tensor_data_offset as usize];
    let header = GgufInventory::read(&mut Cursor::new(prefix), bytes.len() as u64).unwrap();
    assert_eq!(header, full);
    assert_eq!(full.quantization_version, Some(2));
    assert_eq!(full.tensor_counts_by_dtype["F32"], 1);
    assert_eq!(full.tensor_counts_by_dtype["Q4K"], 1);
    assert_eq!(full.tensor_payload_bytes, 16 + 288);
    let quantized = &full.tensors[0];
    assert_eq!(quantized.name, "a_projection.weight");
    assert_eq!(quantized.dimensions, [2, 256]);
    assert_eq!(quantized.logical_values_per_block, 256);
    assert_eq!(quantized.bytes_per_block, 144);
    assert_eq!(quantized.block_axis, 1);
    assert_eq!(
        quantized.quantization_format.as_deref(),
        Some("quantization.gguf.q4-k")
    );
    assert!(full.tensors[1].quantization_format.is_none());
    // A header prefix alone is not a complete artifact: its physical file
    // length cannot satisfy the tensor bounds without an external declaration.
    assert!(GgufInventory::read(&mut Cursor::new(prefix), prefix.len() as u64).is_err());
}

#[test]
fn rejects_truncated_descriptors_and_payload_bounds() {
    let bytes = fixture();
    assert!(GgufInventory::read(&mut Cursor::new(&bytes[..24]), bytes.len() as u64).is_err());
    let content = Content::read(&mut Cursor::new(&bytes)).unwrap();
    let inventory = GgufInventory::from_content(&content, bytes.len() as u64).unwrap();
    let last_byte = inventory
        .tensors
        .iter()
        .map(|tensor| tensor.absolute_offset + tensor.bytes)
        .max()
        .unwrap();
    assert!(GgufInventory::from_content(&content, last_byte - 1).is_err());
}

#[test]
fn rejects_partial_row_blocks_even_when_total_elements_form_whole_blocks() {
    let bytes = fixture();
    let mut content = Content::read(&mut Cursor::new(&bytes)).unwrap();
    content
        .tensor_infos
        .get_mut("a_projection.weight")
        .unwrap()
        .shape = Shape::from((256, 2));
    let error = GgufInventory::from_content(&content, bytes.len() as u64).unwrap_err();
    assert!(error
        .to_string()
        .contains("incomplete row quantization block"));
}

#[test]
fn rejects_overlap_misalignment_and_offset_overflow() {
    let bytes = fixture();
    for bad_offset in [0, 1, u64::MAX - 31] {
        let mut content = Content::read(&mut Cursor::new(&bytes)).unwrap();
        content
            .tensor_infos
            .get_mut("a_projection.weight")
            .unwrap()
            .offset = bad_offset;
        assert!(GgufInventory::from_content(&content, u64::MAX).is_err());
    }
}

#[test]
fn rejects_overflowing_and_empty_shapes() {
    let bytes = fixture();
    for dimensions in [
        vec![usize::MAX, usize::MAX, 256],
        vec![0, 256],
        vec![],
        vec![1, 1, 1, 1, 256],
    ] {
        let mut content = Content::read(&mut Cursor::new(&bytes)).unwrap();
        content
            .tensor_infos
            .get_mut("a_projection.weight")
            .unwrap()
            .shape = Shape::from(dimensions);
        assert!(GgufInventory::from_content(&content, u64::MAX).is_err());
    }
}

#[test]
fn retains_split_identity_and_rejects_incomplete_or_inconsistent_metadata() {
    let bytes = fixture();
    let mut content = Content::read(&mut Cursor::new(&bytes)).unwrap();
    content.metadata.insert("split.no".into(), Value::U16(1));
    assert!(GgufInventory::from_content(&content, bytes.len() as u64).is_err());
    content.metadata.insert("split.count".into(), Value::U16(2));
    content
        .metadata
        .insert("split.tensors.count".into(), Value::I32(7));
    let split = GgufInventory::from_content(&content, bytes.len() as u64)
        .unwrap()
        .split
        .unwrap();
    assert_eq!((split.index, split.count, split.total_tensors), (1, 2, 7));
    content.metadata.insert("split.no".into(), Value::U16(2));
    assert!(GgufInventory::from_content(&content, bytes.len() as u64).is_err());
    content.metadata.insert("split.no".into(), Value::I32(-1));
    assert!(GgufInventory::from_content(&content, bytes.len() as u64).is_err());
}

fn raw_header(tensors: &[(&str, u32, u64)], alignment: u32) -> Vec<u8> {
    raw_header_version(tensors, alignment, 3)
}

fn raw_header_version(tensors: &[(&str, u32, u64)], alignment: u32, version: u32) -> Vec<u8> {
    fn length(bytes: &mut Vec<u8>, value: u64, version: u32) {
        if version == 1 {
            bytes.extend_from_slice(&u32::try_from(value).unwrap().to_le_bytes());
        } else {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    fn string(bytes: &mut Vec<u8>, value: &str, version: u32) {
        length(bytes, value.len() as u64, version);
        bytes.extend_from_slice(value.as_bytes());
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&version.to_le_bytes());
    length(&mut bytes, tensors.len() as u64, version);
    length(&mut bytes, 3, version);
    string(&mut bytes, "general.architecture", version);
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    string(&mut bytes, "fixture", version);
    string(&mut bytes, "general.alignment", version);
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&alignment.to_le_bytes());
    // Exercise skipping real string arrays, including valid empty tokens.
    string(&mut bytes, "tokenizer.ggml.tokens", version);
    bytes.extend_from_slice(&9_u32.to_le_bytes());
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    length(&mut bytes, 3, version);
    for token in ["", "hello", "世界"] {
        string(&mut bytes, token, version);
    }
    for (name, dtype, offset) in tensors {
        string(&mut bytes, name, version);
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        length(&mut bytes, 256, version);
        length(&mut bytes, 1, version);
        bytes.extend_from_slice(&dtype.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }
    bytes
}

#[test]
fn legacy_and_current_descriptor_versions_agree_with_candle() {
    for (version, alignment) in [(1, 64), (2, 24), (3, 32)] {
        let bytes = raw_header_version(&[("projection.weight", 12, 0)], alignment, version);
        let full_size =
            (bytes.len() as u64).div_ceil(alignment.into()) * u64::from(alignment) + 144;
        let candle = Content::read(&mut Cursor::new(&bytes)).unwrap();
        let inventory = GgufInventory::read(&mut Cursor::new(&bytes), full_size).unwrap();
        assert_eq!(
            inventory,
            GgufInventory::from_content(&candle, full_size).unwrap()
        );
        assert_eq!(inventory.tensors[0].dimensions, [1, 256]);
    }
}

#[test]
fn inventories_iq4_xs_without_claiming_runtime_decoder_support() {
    let bytes = raw_header(&[("projection.weight", 23, 0)], 32);
    let full_size = (bytes.len() as u64).div_ceil(32) * 32 + 136;
    // Candle cannot parse this code, but the file's declared ABI is known.
    assert!(Content::read(&mut Cursor::new(&bytes)).is_err());
    let inventory = GgufInventory::read(&mut Cursor::new(&bytes), full_size).unwrap();
    let tensor = &inventory.tensors[0];
    assert_eq!(tensor.dtype, "IQ4_XS");
    assert_eq!(tensor.ggml_type, 23);
    assert_eq!(
        (tensor.logical_values_per_block, tensor.bytes_per_block),
        (256, 136)
    );
    assert_eq!(tensor.bytes, 136);
    assert!(!tensor.candle_dtype_available);
    assert!(GgufInventory::read(&mut Cursor::new(&bytes), full_size - 1).is_err());
}

#[test]
fn iq4_nl_retains_its_distinct_block_abi() {
    let bytes = raw_header(&[("projection.weight", 20, 0)], 32);
    let full_size = (bytes.len() as u64).div_ceil(32) * 32 + 144;
    let inventory = GgufInventory::read(&mut Cursor::new(&bytes), full_size).unwrap();
    let tensor = &inventory.tensors[0];
    assert_eq!(tensor.dtype, "IQ4_NL");
    assert_eq!(
        (tensor.logical_values_per_block, tensor.bytes_per_block),
        (32, 18)
    );
    assert_eq!(tensor.bytes, 144);
    assert!(!tensor.candle_dtype_available);
    assert!(Content::read(&mut Cursor::new(&bytes)).is_err());
}

#[test]
fn iq3_s_counts_signs_and_scales_in_its_physical_bytes() {
    let bytes = raw_header(&[("projection.weight", 21, 0)], 32);
    let full_size = (bytes.len() as u64).div_ceil(32) * 32 + 110;
    let inventory = GgufInventory::read(&mut Cursor::new(&bytes), full_size).unwrap();
    let tensor = &inventory.tensors[0];
    assert_eq!(tensor.dtype, "IQ3_S");
    assert_eq!(
        (tensor.logical_values_per_block, tensor.bytes_per_block),
        (256, 110)
    );
    assert_eq!(tensor.bytes, 110);
    assert!(!tensor.candle_dtype_available);
    assert!(GgufInventory::read(&mut Cursor::new(&bytes), full_size - 1).is_err());
}

#[test]
fn header_reader_rejects_duplicate_names_zero_alignment_and_unknown_abi() {
    let cases = [
        raw_header(&[("weight", 23, 0), ("weight", 23, 160)], 32),
        raw_header(&[("weight", 23, 0)], 0),
        raw_header(&[("weight", 23, 0)], 4),
        raw_header(&[("weight", u32::MAX, 0)], 32),
    ];
    for bytes in cases {
        assert!(GgufInventory::read(&mut Cursor::new(bytes), u64::MAX).is_err());
    }
}

#[test]
fn header_reader_rejects_every_truncated_prefix_of_a_descriptor_table() {
    let bytes = raw_header(&[("weight", 23, 0)], 32);
    for length in 0..bytes.len() {
        assert!(
            GgufInventory::read(&mut Cursor::new(&bytes[..length]), u64::MAX).is_err(),
            "accepted {length}-byte prefix"
        );
    }
}
