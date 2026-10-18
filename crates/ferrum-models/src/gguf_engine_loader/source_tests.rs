use super::*;
use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::Tensor;
use ferrum_interfaces::vnext::{ModelSourceKind, OriginalModelSource, OriginalModelSources};
use std::io::Cursor;

fn semantic(architecture: &str) -> serde_json::Value {
    serde_json::json!({
        "architectures": [architecture],
        "hidden_size": 4, "intermediate_size": 8,
        "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 2,
        "vocab_size": 4, "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "hidden_act": "silu"
    })
}

fn fixture(
    root: &Path,
    definition: &serde_json::Value,
    arch: &str,
    extra: &[(&str, Value)],
    weights: bool,
) -> ProductionModelSourceBundle {
    fixture_with_biases(root, definition, arch, extra, weights, &[])
}

fn fixture_with_biases(
    root: &Path,
    definition: &serde_json::Value,
    arch: &str,
    extra: &[(&str, Value)],
    weights: bool,
    biases: &[(&str, usize)],
) -> ProductionModelSourceBundle {
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(definition).unwrap(),
    )
    .unwrap();
    std::fs::write(root.join("tokenizer.json"), b"{}").unwrap();
    let mut metadata = vec![(
        "general.architecture".to_owned(),
        Value::String(arch.into()),
    )];
    for (suffix, value) in [
        ("block_count", Value::U32(1)),
        ("embedding_length", Value::U32(4)),
        ("feed_forward_length", Value::U32(8)),
        ("attention.head_count", Value::U32(2)),
        ("attention.head_count_kv", Value::U32(1)),
        ("attention.key_length", Value::U32(2)),
        ("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        ("context_length", Value::U32(32)),
        ("vocab_size", Value::U32(4)),
        ("rope.freq_base", Value::F32(10000.0)),
    ] {
        metadata.push((format!("{arch}.{suffix}"), value));
    }
    for (suffix, value) in extra {
        let key = format!("{arch}.{suffix}");
        metadata.retain(|(existing, _)| existing != &key);
        metadata.push((key, value.clone()));
    }
    let mut tensors = if weights {
        [
            ("token_embd.weight", vec![4, 4]),
            ("output.weight", vec![4, 4]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![2, 4]),
            ("blk.0.attn_v.weight", vec![2, 4]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_gate.weight", vec![8, 4]),
            ("blk.0.ffn_up.weight", vec![8, 4]),
            ("blk.0.ffn_down.weight", vec![4, 8]),
        ]
        .into_iter()
        .map(|(name, shape)| {
            let tensor = Tensor::from_vec(
                vec![0.25_f32; shape.iter().product()],
                shape,
                &candle_core::Device::Cpu,
            )
            .unwrap();
            (name, QTensor::quantize(&tensor, GgmlDType::F32).unwrap())
        })
        .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    tensors.extend(biases.iter().map(|(name, length)| {
        let tensor = Tensor::from_vec(
            vec![0.125_f32; *length],
            vec![*length],
            &candle_core::Device::Cpu,
        )
        .unwrap();
        (*name, QTensor::quantize(&tensor, GgmlDType::F32).unwrap())
    }));
    let metadata = metadata
        .iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect::<Vec<_>>();
    let tensors = tensors
        .iter()
        .map(|(name, tensor)| (*name, tensor))
        .collect::<Vec<_>>();
    let mut bytes = Cursor::new(Vec::new());
    gguf_file::write(&mut bytes, &metadata, &tensors).unwrap();
    let path = root.join("weights.gguf");
    std::fs::write(&path, bytes.into_inner()).unwrap();
    let original = |kind, location: &Path| OriginalModelSource {
        kind,
        location: location.display().to_string(),
        requested_revision: None,
    };
    ProductionModelSourceBundle::open(
        root,
        root,
        ProductionWeightArtifact::gguf_file(&path),
        OriginalModelSources {
            semantic: original(ModelSourceKind::LocalDirectory, root),
            tokenizer: original(ModelSourceKind::LocalDirectory, root),
            weights: original(ModelSourceKind::LocalFile, &path),
        },
    )
    .unwrap()
}

fn validated(sources: &ProductionModelSourceBundle) -> Result<LlamaFamilyConfig> {
    let definition = ConfigManager::new().load_from_bytes(sources.config_json())?;
    let gguf = GgufFile::open(sources.weights().path()).unwrap();
    validate_dense_gguf_sources(&gguf, &definition)
}

#[test]
fn source_bound_gguf_rejects_wrong_architecture_before_weights() {
    let root = tempfile::tempdir().unwrap();
    let sources = fixture(
        root.path(),
        &semantic("LlamaForCausalLM"),
        "qwen2",
        &[],
        false,
    );
    let error =
        load_gguf_decoder_from_sources_with_info(&sources, &Device::CPU, ModelId::new("fixture"))
            .err()
            .unwrap();
    assert!(matches!(error, FerrumError::Unsupported { .. }));
    assert!(error.to_string().contains("architecture"));
}

#[test]
fn source_bound_gguf_rejects_dimensions_and_semantic_math_before_weights() {
    for (field, value) in [
        ("hidden_size", serde_json::json!(8)),
        ("intermediate_size", serde_json::json!(16)),
        ("num_hidden_layers", serde_json::json!(2)),
        ("num_attention_heads", serde_json::json!(4)),
        ("num_key_value_heads", serde_json::json!(2)),
        ("head_dim", serde_json::json!(4)),
        ("vocab_size", serde_json::json!(8)),
        ("rms_norm_eps", serde_json::json!(1e-5)),
        ("rope_theta", serde_json::json!(500000.0)),
        ("sliding_window", serde_json::json!(8)),
        ("hidden_act", serde_json::json!("relu")),
    ] {
        let root = tempfile::tempdir().unwrap();
        let mut definition = semantic("LlamaForCausalLM");
        definition[field] = value;
        let sources = fixture(root.path(), &definition, "llama", &[], false);
        let error = load_gguf_decoder_from_sources_with_info(
            &sources,
            &Device::CPU,
            ModelId::new("fixture"),
        )
        .err()
        .unwrap();
        assert!(
            error.to_string().contains("semantic mismatch"),
            "{field}: {error}"
        );
    }
}

#[test]
fn source_bound_gguf_loads_same_family_and_mistral_alias_with_frozen_config() {
    for architecture in ["LlamaForCausalLM", "MistralForCausalLM"] {
        let root = tempfile::tempdir().unwrap();
        let mut definition = semantic(architecture);
        definition["sliding_window"] = serde_json::json!(8);
        definition["rope_scaling"] = serde_json::json!({"rope_type": "linear", "factor": 2.0});
        let sources = fixture(
            root.path(),
            &definition,
            "llama",
            &[
                ("attention.sliding_window", Value::U32(8)),
                ("rope.scaling.type", Value::String("linear".into())),
                ("rope.scaling.factor", Value::F32(2.0)),
            ],
            true,
        );
        let config = validated(&sources).unwrap();
        assert!(config.rope_interleaved, "retain the physical Q/K layout");
        assert_eq!(config.sliding_window, 8);
        assert_eq!(config.rope_theta, 10000.0);
        assert_eq!(
            config.rope_scaling,
            Some(crate::models::llama_family::RopeScalingConfig::Linear { factor: 2.0 })
        );
        std::fs::write(root.path().join("config.json"), b"{}").unwrap();
        let (mut model, info) = load_gguf_decoder_from_sources_with_info(
            &sources,
            &Device::CPU,
            ModelId::new("fixture"),
        )
        .unwrap();
        assert_eq!(info.hidden_size, 4);
        let logits = model.prefill("fixture", &[0, 1]);
        assert_eq!(logits.len(), 4);
        assert!(logits.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn source_bound_gguf_rejects_missing_or_unsupported_rope_scaling() {
    let root = tempfile::tempdir().unwrap();
    let mut definition = semantic("MistralForCausalLM");
    definition["rope_scaling"] = serde_json::json!({"rope_type": "linear", "factor": 2.0});
    let sources = fixture(root.path(), &definition, "llama", &[], false);
    assert!(validated(&sources)
        .unwrap_err()
        .to_string()
        .contains("RoPE"));
    let sources = fixture(
        root.path(),
        &definition,
        "llama",
        &[("rope.scaling.type", Value::String("linear".into()))],
        false,
    );
    assert!(validated(&sources)
        .unwrap_err()
        .to_string()
        .contains("factor"));
    definition["rope_scaling"] = serde_json::json!({"rope_type": "yarn", "factor": 2.0});
    let sources = fixture(root.path(), &definition, "llama", &[], false);
    assert!(matches!(
        validated(&sources),
        Err(FerrumError::Unsupported { .. })
    ));
}

#[test]
fn source_bound_gguf_accepts_gemma_norm_conversion_but_not_changed_schedule() {
    let root = tempfile::tempdir().unwrap();
    let mut definition = semantic("Gemma3ForCausalLM");
    definition["hidden_act"] = serde_json::json!("gelu_pytorch_tanh");
    definition["sliding_window"] = serde_json::json!(8);
    let sources = fixture(
        root.path(),
        &definition,
        "gemma3",
        &[("attention.sliding_window", Value::U32(8))],
        false,
    );
    assert!(!validated(&sources).unwrap().rmsnorm_unit_offset);
    definition["sliding_window_pattern"] = serde_json::json!(4);
    let sources = fixture(
        root.path(),
        &definition,
        "gemma3",
        &[("attention.sliding_window", Value::U32(8))],
        false,
    );
    assert!(validated(&sources)
        .unwrap_err()
        .to_string()
        .contains("sliding_window_pattern"));
}

#[test]
fn public_dense_gguf_rejects_zero_dimensions_and_invalid_gqa_before_derived_math() {
    for (suffix, value) in [
        ("attention.head_count", 0),
        ("attention.head_count_kv", 0),
        ("attention.head_count_kv", 3),
        ("attention.key_length", 0),
        ("embedding_length", 0),
        ("feed_forward_length", 0),
        ("block_count", 0),
        ("vocab_size", 0),
        ("context_length", 0),
    ] {
        let root = tempfile::tempdir().unwrap();
        let sources = fixture(
            root.path(),
            &semantic("Gemma3ForCausalLM"),
            "gemma3",
            &[("block_count", Value::U32(62)), (suffix, Value::U32(value))],
            false,
        );
        // key_length is explicitly present in this real GGUF. Zero heads
        // must return Err before Gemma 27B's hidden_size / num_heads branch.
        let gguf = GgufFile::open(sources.weights().path()).unwrap();
        assert!(
            LlamaFamilyConfig::from_gguf(&gguf).is_err(),
            "{suffix}={value}"
        );
    }
}

#[test]
fn source_bound_gguf_rejects_nested_rope_and_malformed_explicit_pattern() {
    let root = tempfile::tempdir().unwrap();
    let mut definition = semantic("LlamaForCausalLM");
    definition["rope_parameters"] = serde_json::json!({
        "full_attention": {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}
    });
    let sources = fixture(root.path(), &definition, "llama", &[], false);
    assert!(validated(&sources)
        .unwrap_err()
        .to_string()
        .contains("nested semantic rope_parameters"));

    for pattern in [
        serde_json::json!("6"),
        serde_json::json!(6.0),
        serde_json::json!(6.5),
        serde_json::json!(0),
        serde_json::json!(-1),
        serde_json::Value::Null,
    ] {
        let mut definition = semantic("Gemma3ForCausalLM");
        definition["hidden_act"] = serde_json::json!("gelu_pytorch_tanh");
        definition["sliding_window_pattern"] = pattern;
        let sources = fixture(root.path(), &definition, "gemma3", &[], false);
        assert!(validated(&sources)
            .unwrap_err()
            .to_string()
            .contains("positive integer"));
    }
}

#[test]
fn source_bound_gguf_layer_types_must_match_the_executors_actual_schedule() {
    let root = tempfile::tempdir().unwrap();
    let mut definition = semantic("Gemma3ForCausalLM");
    definition["hidden_act"] = serde_json::json!("gelu_pytorch_tanh");
    definition["num_hidden_layers"] = serde_json::json!(6);
    definition["sliding_window"] = serde_json::json!(8);
    definition["sliding_window_pattern"] = serde_json::json!(6);
    definition["layer_types"] = serde_json::json!([
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention"
    ]);
    let extra = [
        ("block_count", Value::U32(6)),
        ("attention.sliding_window", Value::U32(8)),
    ];
    let sources = fixture(root.path(), &definition, "gemma3", &extra, false);
    validated(&sources).unwrap();
    for layers in [
        serde_json::json!([
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention"
        ]),
        serde_json::json!(["sliding_attention"]),
        serde_json::json!("full_attention"),
    ] {
        definition["layer_types"] = layers;
        let sources = fixture(root.path(), &definition, "gemma3", &extra, false);
        assert!(validated(&sources)
            .unwrap_err()
            .to_string()
            .contains("layer_types"));
    }
    // Uniform full attention remains valid on the non-patterned family.
    let mut definition = semantic("LlamaForCausalLM");
    definition["layer_types"] = serde_json::json!(["full_attention"]);
    validated(&fixture(root.path(), &definition, "llama", &[], false)).unwrap();
}

#[test]
fn source_bound_gguf_qwen2_default_biases_load_and_prefill() {
    let root = tempfile::tempdir().unwrap();
    let definition = semantic("Qwen2ForCausalLM");
    let biases = [
        ("blk.0.attn_q.bias", 4),
        ("blk.0.attn_k.bias", 2),
        ("blk.0.attn_v.bias", 2),
    ];
    let sources = fixture_with_biases(root.path(), &definition, "qwen2", &[], true, &biases);
    let (mut model, _) =
        load_gguf_decoder_from_sources_with_info(&sources, &Device::CPU, ModelId::new("fixture"))
            .unwrap();
    let logits = model.prefill("fixture", &[0, 1]);
    assert_eq!(logits.len(), 4);
    assert!(logits.iter().all(|value| value.is_finite()));
    model.release("fixture");
    drop(model);

    let sources = fixture_with_biases(root.path(), &definition, "qwen2", &[], false, &biases[..2]);
    assert!(validated(&sources)
        .unwrap_err()
        .to_string()
        .contains("attn_v.bias"));
    let extra_bias = [
        biases[0],
        biases[1],
        biases[2],
        ("blk.0.attn_output.bias", 4),
    ];
    let sources = fixture_with_biases(root.path(), &definition, "qwen2", &[], false, &extra_bias);
    assert!(validated(&sources)
        .unwrap_err()
        .to_string()
        .contains("attn_output.bias"));
}

#[test]
fn source_bound_gguf_rejects_explicit_bias_flags_against_fixed_family_contracts() {
    let qwen2_biases = [
        ("blk.0.attn_q.bias", 4),
        ("blk.0.attn_k.bias", 2),
        ("blk.0.attn_v.bias", 2),
    ];
    for (architecture, arch, fixed_attention) in [
        ("Qwen2ForCausalLM", "qwen2", Some(true)),
        ("MistralForCausalLM", "llama", Some(false)),
        ("Qwen3ForCausalLM", "qwen3", None),
        ("Gemma3ForCausalLM", "gemma3", None),
    ] {
        let root = tempfile::tempdir().unwrap();
        let mut definition = semantic(architecture);
        if arch == "gemma3" {
            definition["hidden_act"] = serde_json::json!("gelu_pytorch_tanh");
        }
        let biases: &[(&str, usize)] = if arch == "qwen2" { &qwen2_biases } else { &[] };
        // Omitted flags preserve the actual family defaults, including Qwen2's
        // mandatory Q/K/V bias. Explicit matching declarations remain valid.
        validated(&fixture_with_biases(
            root.path(),
            &definition,
            arch,
            &[],
            false,
            biases,
        ))
        .unwrap();
        definition["mlp_bias"] = serde_json::json!(false);
        if let Some(attention) = fixed_attention {
            definition["attention_bias"] = serde_json::json!(attention);
        }
        validated(&fixture_with_biases(
            root.path(),
            &definition,
            arch,
            &[],
            false,
            biases,
        ))
        .unwrap();

        for (field, conflicting) in [
            ("mlp_bias", Some(true)),
            ("attention_bias", fixed_attention.map(|value| !value)),
        ] {
            let Some(conflicting) = conflicting else {
                continue;
            };
            let mut invalid = definition.clone();
            invalid[field] = serde_json::json!(conflicting);
            let sources = fixture_with_biases(root.path(), &invalid, arch, &[], false, biases);
            let error = validated(&sources).unwrap_err().to_string();
            assert!(
                error.contains(field) && error.contains("fixed bias contract"),
                "{architecture}: {error}"
            );
        }
    }
}

#[test]
fn source_bound_gguf_llama_bias_flags_require_exact_presence_and_shape() {
    let root = tempfile::tempdir().unwrap();
    let mut definition = semantic("LlamaForCausalLM");
    definition["attention_bias"] = serde_json::json!(true);
    definition["mlp_bias"] = serde_json::json!(true);
    let biases = [
        ("blk.0.attn_q.bias", 4),
        ("blk.0.attn_k.bias", 2),
        ("blk.0.attn_v.bias", 2),
        ("blk.0.attn_output.bias", 4),
        ("blk.0.ffn_gate.bias", 8),
        ("blk.0.ffn_up.bias", 8),
        ("blk.0.ffn_down.bias", 4),
    ];
    validated(&fixture_with_biases(
        root.path(),
        &definition,
        "llama",
        &[],
        false,
        &biases,
    ))
    .unwrap();
    let missing = fixture(root.path(), &definition, "llama", &[], false);
    assert!(validated(&missing)
        .unwrap_err()
        .to_string()
        .contains("attn_q.bias"));
    let mut wrong_shape = biases;
    wrong_shape[0].1 = 5;
    let sources = fixture_with_biases(root.path(), &definition, "llama", &[], false, &wrong_shape);
    assert!(validated(&sources)
        .unwrap_err()
        .to_string()
        .contains("shape [4]"));
    let defaults = fixture_with_biases(
        root.path(),
        &semantic("LlamaForCausalLM"),
        "llama",
        &[],
        false,
        &biases,
    );
    assert!(validated(&defaults)
        .unwrap_err()
        .to_string()
        .contains("expected bias presence false"));
}
