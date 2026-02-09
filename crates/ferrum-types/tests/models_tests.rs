use ferrum_types::*;

fn demo_model_info() -> ModelInfo {
    ModelInfo {
        model_id: ModelId::from("demo"),
        model_type: ModelType::Llama,
        num_parameters: 1_000_000, // 1M
        hidden_size: 1024,
        num_layers: 16,
        num_heads: 16,
        num_kv_heads: 8,
        vocab_size: 32000,
        max_sequence_length: 2048,
        dtype: DataType::FP16,
        device: Device::CPU,
        version: None,
        license: None,
        metadata: Default::default(),
    }
}

#[test]
fn model_type_display() {
    assert_eq!(ModelType::Llama.to_string(), "llama");
    assert_eq!(ModelType::Code("rust".into()).to_string(), "code-rust");
}

#[test]
fn model_info_estimates() {
    let info = demo_model_info();
    let size = info.estimated_size_bytes();
    assert!(size > 0);
    assert!(info.supports_sequence_length(1024));
    assert!(!info.supports_sequence_length(4096));

    let mem = info.memory_requirements(2, 128);
    assert!(mem.parameter_memory > 0);
    assert!(
        mem.total_estimated >= mem.parameter_memory + mem.kv_cache_memory + mem.activation_memory
    );
}

#[test]
fn quant_config_helpers() {
    let q = QuantizationConfig::INT4 {
        symmetric: true,
        group_size: 128,
    };
    assert_eq!(q.bits(), 4);
    assert!(!q.is_high_accuracy());

    let q = QuantizationConfig::FP8 {
        e4m3: true,
        kv_cache: true,
    };
    assert!(q.is_high_accuracy());
}

#[test]
fn token_usage_helpers() {
    let mut u = TokenUsage::new(10, 5);
    assert_eq!(u.total_tokens, 15);
    u.add_completion_tokens(3);
    assert_eq!(u.total_tokens, 18);
}

#[test]
fn model_config_validate() {
    let mut c = ModelConfig::new("id", "/path/to/model");
    c.max_batch_size = 4;
    c.max_sequence_length = 128;
    c.validate().unwrap();
}
