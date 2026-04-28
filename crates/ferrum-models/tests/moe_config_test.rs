//! `Qwen3MoeConfig::from_gguf` — verify MoE metadata parsing for the
//! qwen3moe architecture, plus interaction with `LlamaFamilyConfig::from_gguf`
//! (which must reject MoE archs rather than silently lowering them).

use std::io::{Cursor, Write};

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use ferrum_models::models::llama_family::LlamaFamilyConfig;
use ferrum_models::moe_config::Qwen3MoeConfig;
use ferrum_quantization::gguf::GgufFile;

/// Build a metadata-only GGUF (with a placeholder embed table). Mirrors the
/// helper from `gguf_config_test.rs` but takes pre-built `Value` entries
/// and a custom architecture string.
fn build_gguf(
    arch: &str,
    extra_keys: &[(&str, Value)],
    vocab_rows: usize,
    hidden_size: usize,
) -> tempfile::NamedTempFile {
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
fn qwen3moe_config_from_gguf_full() {
    // Qwen3-30B-A3B numbers: 48 layers, hidden 2048, 32 heads, 4 kv heads,
    // 128 experts, top-8 routing, expert FFN 768.
    let extra = [
        ("qwen3moe.block_count", Value::U32(48)),
        ("qwen3moe.embedding_length", Value::U32(2048)),
        ("qwen3moe.feed_forward_length", Value::U32(6144)), // legacy field
        ("qwen3moe.attention.head_count", Value::U32(32)),
        ("qwen3moe.attention.head_count_kv", Value::U32(4)),
        (
            "qwen3moe.attention.layer_norm_rms_epsilon",
            Value::F32(1.0e-6),
        ),
        ("qwen3moe.context_length", Value::U32(32768)),
        ("qwen3moe.rope.freq_base", Value::F32(1.0e6)),
        ("qwen3moe.vocab_size", Value::U32(151_936)),
        ("qwen3moe.expert_count", Value::U32(128)),
        ("qwen3moe.expert_used_count", Value::U32(8)),
        ("qwen3moe.expert_feed_forward_length", Value::U32(768)),
        ("qwen3moe.expert_norm_topk_prob", Value::Bool(true)),
    ];

    let tmp = build_gguf("qwen3moe", &extra, 32, 2048);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let cfg = Qwen3MoeConfig::from_gguf(&gguf).expect("Qwen3MoeConfig::from_gguf");

    // MoE-specific
    assert_eq!(cfg.num_experts, 128);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert_eq!(cfg.expert_intermediate_size, 768);
    assert!(cfg.norm_topk_prob);
    assert!(cfg.is_truly_sparse());

    // Inherited dense fields
    assert_eq!(cfg.base.num_layers, 48);
    assert_eq!(cfg.base.hidden_size, 2048);
    assert_eq!(cfg.base.num_heads, 32);
    assert_eq!(cfg.base.num_kv_heads, 4);
    assert_eq!(cfg.base.head_dim, 64); // 2048 / 32
    assert_eq!(cfg.base.intermediate_size, 768); // mirrored from expert size
    assert_eq!(cfg.base.vocab_size, 151_936);
    assert_eq!(cfg.base.max_seq_len, 32768);
    assert!(cfg.base.has_qk_norm, "Qwen3-MoE has QK-norm");
    assert_eq!(cfg.base.sliding_window, 0);
    assert!((cfg.base.rope_theta - 1.0e6).abs() < 1.0);
    assert!((cfg.base.rms_norm_eps - 1.0e-6).abs() < 1e-12);
}

#[test]
fn qwen3moe_norm_topk_prob_defaults_true_when_missing() {
    // Some older GGUF dumps don't include `expert_norm_topk_prob` —
    // helper should default to true (Qwen3-MoE convention).
    let extra = [
        ("qwen3moe.block_count", Value::U32(4)),
        ("qwen3moe.embedding_length", Value::U32(64)),
        ("qwen3moe.attention.head_count", Value::U32(2)),
        (
            "qwen3moe.attention.layer_norm_rms_epsilon",
            Value::F32(1.0e-6),
        ),
        ("qwen3moe.expert_count", Value::U32(8)),
        ("qwen3moe.expert_used_count", Value::U32(2)),
        ("qwen3moe.expert_feed_forward_length", Value::U32(128)),
    ];
    let tmp = build_gguf("qwen3moe", &extra, 32, 64);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let cfg = Qwen3MoeConfig::from_gguf(&gguf).unwrap();
    assert!(cfg.norm_topk_prob);
}

#[test]
fn invalid_topk_count_returns_err() {
    // expert_used_count > expert_count is structurally invalid.
    let extra = [
        ("qwen3moe.block_count", Value::U32(4)),
        ("qwen3moe.embedding_length", Value::U32(64)),
        ("qwen3moe.attention.head_count", Value::U32(2)),
        (
            "qwen3moe.attention.layer_norm_rms_epsilon",
            Value::F32(1.0e-6),
        ),
        ("qwen3moe.expert_count", Value::U32(4)),
        ("qwen3moe.expert_used_count", Value::U32(8)), // > 4
        ("qwen3moe.expert_feed_forward_length", Value::U32(128)),
    ];
    let tmp = build_gguf("qwen3moe", &extra, 32, 64);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let result = Qwen3MoeConfig::from_gguf(&gguf);
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("expert_used_count"), "{err}");
}

#[test]
fn missing_expert_fields_returns_err() {
    let extra = [
        ("qwen3moe.block_count", Value::U32(4)),
        ("qwen3moe.embedding_length", Value::U32(64)),
        ("qwen3moe.attention.head_count", Value::U32(2)),
        (
            "qwen3moe.attention.layer_norm_rms_epsilon",
            Value::F32(1.0e-6),
        ),
        // No expert_count / expert_used_count / expert_feed_forward_length
    ];
    let tmp = build_gguf("qwen3moe", &extra, 32, 64);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let result = Qwen3MoeConfig::from_gguf(&gguf);
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("expert_count"),
        "should mention missing key: {err}"
    );
}

#[test]
fn wrong_architecture_returns_err() {
    // Dense qwen3 GGUF passed to MoE constructor should reject upfront.
    let extra = [
        ("qwen3.block_count", Value::U32(28)),
        ("qwen3.embedding_length", Value::U32(1024)),
        ("qwen3.attention.head_count", Value::U32(16)),
        ("qwen3.attention.layer_norm_rms_epsilon", Value::F32(1.0e-6)),
    ];
    let tmp = build_gguf("qwen3", &extra, 32, 1024);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let result = Qwen3MoeConfig::from_gguf(&gguf);
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("expected arch 'qwen3moe'"), "{err}");
}

#[test]
fn dense_from_gguf_rejects_moe_archs_with_redirect() {
    // LlamaFamilyConfig::from_gguf should NOT silently lower a qwen3moe
    // file to a dense config (would lose all MoE info). It must error
    // and point callers at the MoE constructor.
    let extra = [
        ("qwen3moe.block_count", Value::U32(48)),
        ("qwen3moe.embedding_length", Value::U32(2048)),
        ("qwen3moe.attention.head_count", Value::U32(32)),
        (
            "qwen3moe.attention.layer_norm_rms_epsilon",
            Value::F32(1.0e-6),
        ),
    ];
    let tmp = build_gguf("qwen3moe", &extra, 32, 2048);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let result = LlamaFamilyConfig::from_gguf(&gguf);
    assert!(result.is_err(), "dense from_gguf must reject MoE arch");
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("MoE") && err.contains("Qwen3MoeConfig::from_gguf"),
        "redirect points to MoE constructor: {err}"
    );

    // mixtral and deepseek2 should also be rejected.
    for arch in ["mixtral", "deepseek2"] {
        let arch_extra = [
            (format!("{arch}.block_count"), Value::U32(32)),
            (format!("{arch}.embedding_length"), Value::U32(4096)),
            (format!("{arch}.attention.head_count"), Value::U32(32)),
            (
                format!("{arch}.attention.layer_norm_rms_epsilon"),
                Value::F32(1.0e-5),
            ),
        ];
        // Convert to (&str, Value) tuples for build_gguf.
        let extras_owned: Vec<(String, Value)> = arch_extra.into_iter().collect();
        let extras_ref: Vec<(&str, Value)> = extras_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();
        let tmp = build_gguf(arch, &extras_ref, 32, 4096);
        let gguf = GgufFile::open(tmp.path()).unwrap();
        let result = LlamaFamilyConfig::from_gguf(&gguf);
        assert!(
            result.is_err(),
            "dense from_gguf must reject {arch}, got {:?}",
            result.is_ok()
        );
    }
}

#[test]
fn from_base_round_trips_correctly() {
    // Construct a dense LlamaFamilyConfig (any arch) then wrap in MoE.
    // Verifies the constructor does what it says without going through GGUF.
    let extra = [
        ("qwen3.block_count", Value::U32(28)),
        ("qwen3.embedding_length", Value::U32(1024)),
        ("qwen3.feed_forward_length", Value::U32(3072)),
        ("qwen3.attention.head_count", Value::U32(16)),
        ("qwen3.attention.layer_norm_rms_epsilon", Value::F32(1.0e-6)),
    ];
    let tmp = build_gguf("qwen3", &extra, 32, 1024);
    let gguf = GgufFile::open(tmp.path()).unwrap();
    let dense = LlamaFamilyConfig::from_gguf(&gguf).unwrap();

    let moe = Qwen3MoeConfig::from_base(dense.clone(), 64, 4, 256, true);
    assert_eq!(moe.base, dense);
    assert_eq!(moe.num_experts, 64);
    assert_eq!(moe.num_experts_per_tok, 4);
    assert!(moe.is_truly_sparse());

    // Edge: top_k == num_experts → not sparse.
    let dense_eq = Qwen3MoeConfig::from_base(dense, 4, 4, 256, true);
    assert!(!dense_eq.is_truly_sparse());
}
