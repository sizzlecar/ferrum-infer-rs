//! `LlamaFamilyConfig::from_gguf` — verify metadata parsing for the four
//! Llama-family architectures (qwen3 / qwen2 / llama / mistral) plus
//! fallback paths (missing optional fields, vocab inferred from embed).

use std::io::{Cursor, Write};

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use ferrum_models::models::llama_family::LlamaFamilyConfig;
use ferrum_quantization::gguf::GgufFile;

/// Build a GGUF that contains only metadata + a fake embed table — enough
/// to drive `from_gguf` end-to-end without needing real weights.
fn build_metadata_gguf(
    arch: &str,
    extra_keys: &[(&str, Value)],
    vocab_rows: usize,
    hidden_size: usize,
) -> tempfile::NamedTempFile {
    // Required: a token_embd.weight tensor — `from_gguf` falls back to it
    // if `<arch>.vocab_size` isn't set, and it has to dequantize without
    // shape errors. Use F32 so any size works.
    let device = Device::Cpu;
    let n = vocab_rows * hidden_size;
    let raw: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
    let t = Tensor::from_vec(raw, (vocab_rows, hidden_size), &device).unwrap();
    let embed = QTensor::quantize(&t, GgmlDType::F32).unwrap();

    let arch_v = Value::String(arch.to_string());
    let mut metadata: Vec<(&str, &Value)> = vec![("general.architecture", &arch_v)];
    for (k, v) in extra_keys {
        metadata.push((k, v));
    }
    let tensors: Vec<(&str, &QTensor)> = vec![("token_embd.weight", &embed)];

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
fn qwen3_config_from_gguf() {
    // Qwen3-0.6B-ish numbers: 28 layers, hidden 1024, ffn 3072, 16 heads,
    // 8 kv heads (GQA 2:1), rms eps 1e-6, rope 1e6, ctx 32768.
    let block_count = Value::U32(28);
    let hidden = Value::U32(1024);
    let ffn = Value::U32(3072);
    let heads = Value::U32(16);
    let kv_heads = Value::U32(8);
    let rms_eps = Value::F32(1.0e-6);
    let ctx = Value::U32(32768);
    let rope = Value::F32(1.0e6);
    let vocab = Value::U32(151_936);

    let extra = [
        ("qwen3.block_count", block_count),
        ("qwen3.embedding_length", hidden),
        ("qwen3.feed_forward_length", ffn),
        ("qwen3.attention.head_count", heads),
        ("qwen3.attention.head_count_kv", kv_heads),
        ("qwen3.attention.layer_norm_rms_epsilon", rms_eps),
        ("qwen3.context_length", ctx),
        ("qwen3.rope.freq_base", rope),
        ("qwen3.vocab_size", vocab),
    ];

    let tmp = build_metadata_gguf("qwen3", &extra, 32, 1024);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let cfg = LlamaFamilyConfig::from_gguf(&gguf).expect("from_gguf");

    assert_eq!(cfg.num_layers, 28);
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.intermediate_size, 3072);
    assert_eq!(cfg.num_heads, 16);
    assert_eq!(cfg.num_kv_heads, 8);
    assert_eq!(cfg.head_dim, 64); // 1024 / 16
    assert_eq!(cfg.max_seq_len, 32768);
    assert!((cfg.rms_norm_eps - 1.0e-6).abs() < 1e-12);
    assert!((cfg.rope_theta - 1.0e6).abs() < 1.0);
    assert_eq!(cfg.vocab_size, 151_936);
    assert!(cfg.has_qk_norm, "Qwen3 must enable QK-norm");
    assert_eq!(cfg.sliding_window, 0);
}

#[test]
fn llama_config_from_gguf_uses_default_rope() {
    // Llama-3.2-1B-ish, but omit rope_theta — helper should fall back to
    // the Llama default of 5e5.
    let extra = [
        ("llama.block_count", Value::U32(16)),
        ("llama.embedding_length", Value::U32(2048)),
        ("llama.feed_forward_length", Value::U32(8192)),
        ("llama.attention.head_count", Value::U32(32)),
        ("llama.attention.head_count_kv", Value::U32(8)),
        ("llama.attention.layer_norm_rms_epsilon", Value::F32(1.0e-5)),
        ("llama.context_length", Value::U32(131072)),
        ("llama.vocab_size", Value::U32(128_256)),
    ];

    let tmp = build_metadata_gguf("llama", &extra, 32, 2048);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let cfg = LlamaFamilyConfig::from_gguf(&gguf).unwrap();

    assert_eq!(cfg.num_layers, 16);
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.head_dim, 64); // 2048 / 32
    assert!(!cfg.has_qk_norm);
    assert!(
        (cfg.rope_theta - 500_000.0).abs() < 1.0,
        "Llama default rope is 5e5 when not in metadata"
    );
    assert_eq!(cfg.sliding_window, 0);
}

#[test]
fn mistral_config_from_gguf_reads_sliding_window() {
    let extra = [
        ("mistral.block_count", Value::U32(32)),
        ("mistral.embedding_length", Value::U32(4096)),
        ("mistral.feed_forward_length", Value::U32(14336)),
        ("mistral.attention.head_count", Value::U32(32)),
        ("mistral.attention.head_count_kv", Value::U32(8)),
        (
            "mistral.attention.layer_norm_rms_epsilon",
            Value::F32(1.0e-5),
        ),
        ("mistral.attention.sliding_window", Value::U32(4096)),
        ("mistral.context_length", Value::U32(32768)),
        ("mistral.vocab_size", Value::U32(32_000)),
    ];

    let tmp = build_metadata_gguf("mistral", &extra, 32, 4096);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let cfg = LlamaFamilyConfig::from_gguf(&gguf).unwrap();

    assert_eq!(cfg.sliding_window, 4096, "Mistral v0.1 sliding window read");
    assert!(!cfg.has_qk_norm);
    // Default mistral rope is 1e7
    assert!(
        (cfg.rope_theta - 10_000_000.0).abs() < 10.0,
        "Mistral default rope = 1e7"
    );
}

#[test]
fn vocab_falls_back_to_embed_table_rows() {
    // Don't set <arch>.vocab_size; from_gguf should infer from token_embd
    // shape (vocab_rows passed to build_metadata_gguf).
    let extra = [
        ("qwen3.block_count", Value::U32(4)),
        ("qwen3.embedding_length", Value::U32(64)),
        ("qwen3.feed_forward_length", Value::U32(128)),
        ("qwen3.attention.head_count", Value::U32(2)),
        ("qwen3.attention.layer_norm_rms_epsilon", Value::F32(1.0e-6)),
    ];
    let tmp = build_metadata_gguf("qwen3", &extra, 999, 64);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let cfg = LlamaFamilyConfig::from_gguf(&gguf).unwrap();
    assert_eq!(cfg.vocab_size, 999);
    // num_kv_heads also missing → falls back to num_heads
    assert_eq!(cfg.num_kv_heads, 2);
}

#[test]
fn missing_required_field_returns_err() {
    // No block_count → must error
    let extra = [
        ("qwen3.embedding_length", Value::U32(1024)),
        ("qwen3.feed_forward_length", Value::U32(3072)),
        ("qwen3.attention.head_count", Value::U32(16)),
        ("qwen3.attention.layer_norm_rms_epsilon", Value::F32(1.0e-6)),
    ];
    let tmp = build_metadata_gguf("qwen3", &extra, 32, 1024);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let result = LlamaFamilyConfig::from_gguf(&gguf);
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("block_count"),
        "error mentions missing field: {err}"
    );
}

#[test]
fn unknown_architecture_works_with_safe_defaults() {
    // Unknown arch — has_qk_norm should default to false, rope_theta to 10k.
    let extra = [
        ("custom.block_count", Value::U32(4)),
        ("custom.embedding_length", Value::U32(32)),
        ("custom.feed_forward_length", Value::U32(64)),
        ("custom.attention.head_count", Value::U32(2)),
        (
            "custom.attention.layer_norm_rms_epsilon",
            Value::F32(1.0e-6),
        ),
        ("custom.vocab_size", Value::U32(100)),
    ];
    let tmp = build_metadata_gguf("custom", &extra, 32, 32);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let cfg = LlamaFamilyConfig::from_gguf(&gguf).unwrap();
    assert!(!cfg.has_qk_norm, "unknown arch defaults to no QK-norm");
    assert!(
        (cfg.rope_theta - 10_000.0).abs() < 1.0,
        "unknown arch defaults to 10k rope"
    );
}
