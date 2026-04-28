//! Phase 1A smoke tests: build a tiny GGUF in a tempfile, then verify the
//! adapter can parse the header, look up metadata + tensor descriptors, and
//! materialise the tensor on CPU.
//!
//! No external model files needed — we use candle's `Content::write` to
//! synthesize a 2-tensor GGUF in memory and dump it to a tempfile.

use std::io::{Cursor, Write};

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use ferrum_quantization::gguf::GgufFile;

/// Build a minimal-but-realistic GGUF on disk:
///   - architecture string: "qwen3"
///   - one metadata u32 ("qwen3.block_count")
///   - one metadata f32 ("qwen3.attention.layer_norm_rms_epsilon")
///   - one F32 tensor "embed_tokens" shape (8, 4)
///   - one Q4_0 tensor "blk.0.attn_q" shape (32, 32)  — exercises K-quant path
fn build_test_gguf() -> tempfile::NamedTempFile {
    let device = Device::Cpu;

    // Tensor 1: small F32 (no quant) so we can round-trip values exactly.
    let raw_a = (0..32).map(|i| i as f32 * 0.5).collect::<Vec<_>>();
    let t_a = Tensor::from_vec(raw_a, (8, 4), &device).unwrap();
    let qt_a = QTensor::quantize(&t_a, GgmlDType::F32).unwrap();

    // Tensor 2: Q4_0 over a larger tensor so block math is non-trivial.
    // 32x32 = 1024 elements = 32 blocks of 32 elements each (Q4_0 block size).
    let raw_b = (0..1024)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect::<Vec<_>>();
    let t_b = Tensor::from_vec(raw_b, (32, 32), &device).unwrap();
    let qt_b = QTensor::quantize(&t_b, GgmlDType::Q4_0).unwrap();

    // metadata + tensors must outlive the write call (slice of refs)
    let arch_v = Value::String("qwen3".to_string());
    let block_count_v = Value::U32(28);
    let rms_eps_v = Value::F32(1.0e-6);
    let metadata: Vec<(&str, &Value)> = vec![
        ("general.architecture", &arch_v),
        ("qwen3.block_count", &block_count_v),
        ("qwen3.attention.layer_norm_rms_epsilon", &rms_eps_v),
    ];
    let tensors: Vec<(&str, &QTensor)> = vec![("embed_tokens", &qt_a), ("blk.0.attn_q", &qt_b)];

    // Write to in-memory buffer first, then dump to a NamedTempFile.
    // Going via Vec keeps Content::write happy (it wants Seek + Write).
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        gguf_file::write(&mut cursor, &metadata, &tensors).unwrap();
    }

    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(&buf).unwrap();
    tmp.flush().unwrap();
    tmp
}

#[test]
fn gguf_open_and_parse_header() {
    let tmp = build_test_gguf();

    let gguf = GgufFile::open(tmp.path()).expect("GgufFile::open");

    // Architecture is the well-known key
    assert_eq!(gguf.architecture().unwrap(), "qwen3");

    // Typed metadata accessors round-trip the values we wrote.
    assert_eq!(gguf.metadata_u32("qwen3.block_count").unwrap(), 28);
    assert!(
        (gguf
            .metadata_f32("qwen3.attention.layer_norm_rms_epsilon")
            .unwrap()
            - 1.0e-6)
            .abs()
            < 1.0e-12,
        "rms_eps round-trip mismatch"
    );

    // Wrong-type access returns Err rather than panicking.
    assert!(
        gguf.metadata_u32("general.architecture").is_err(),
        "u32 access on a string field should error"
    );

    // Missing key returns Err with a helpful message
    let err = gguf.metadata_u32("does.not.exist").unwrap_err().to_string();
    assert!(
        err.contains("does.not.exist"),
        "error mentions missing key: {err}"
    );

    // Tensor descriptor table
    assert_eq!(gguf.tensor_count(), 2);
    assert!(gguf.has_tensor("embed_tokens"));
    assert!(gguf.has_tensor("blk.0.attn_q"));
    assert!(!gguf.has_tensor("nope"));

    let names: std::collections::HashSet<&str> = gguf.tensor_names().collect();
    assert_eq!(names.len(), 2);
    assert!(names.contains("embed_tokens"));
    assert!(names.contains("blk.0.attn_q"));

    let info_a = gguf.tensor_info("embed_tokens").unwrap();
    assert_eq!(info_a.shape.dims(), &[8, 4]);
    assert_eq!(info_a.ggml_dtype, GgmlDType::F32);

    let info_b = gguf.tensor_info("blk.0.attn_q").unwrap();
    assert_eq!(info_b.shape.dims(), &[32, 32]);
    assert_eq!(info_b.ggml_dtype, GgmlDType::Q4_0);
}

#[test]
fn gguf_read_tensor_payload_round_trip() {
    let tmp = build_test_gguf();
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let device = Device::Cpu;

    // F32 tensor: round-trip should be bit-exact.
    let qt = gguf.read_tensor("embed_tokens", &device).unwrap();
    assert_eq!(qt.dtype(), GgmlDType::F32);
    assert_eq!(qt.shape().dims(), &[8, 4]);

    let dequant = qt.dequantize(&device).unwrap();
    let values: Vec<f32> = dequant.flatten_all().unwrap().to_vec1().unwrap();
    let expected: Vec<f32> = (0..32).map(|i| i as f32 * 0.5).collect();
    assert_eq!(values, expected, "F32 round-trip mismatch");

    // Q4_0 tensor: round-trip is lossy (~5% rel error tolerable for this test);
    // we only assert shape + dtype here, not values.
    let qt_b = gguf.read_tensor("blk.0.attn_q", &device).unwrap();
    assert_eq!(qt_b.dtype(), GgmlDType::Q4_0);
    assert_eq!(qt_b.shape().dims(), &[32, 32]);
    let _ = qt_b.dequantize(&device).unwrap(); // just verify dequant succeeds
}

#[test]
fn gguf_open_missing_file_returns_err() {
    let err = GgufFile::open("/no/such/path/to/model.gguf")
        .expect_err("opening non-existent file must error");
    let s = err.to_string();
    assert!(s.contains("/no/such/path"), "error includes path: {s}");
}
