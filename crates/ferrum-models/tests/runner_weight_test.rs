//! Real model weight loading test.
//!
//! Loads Qwen3-0.6B safetensors into ModelRunner<CpuBackend> and runs decode.
//! Requires model to be downloaded: `ferrum pull qwen3:0.6b`
//!
//! Ignored by default (needs ~1.4GB model files). Run explicitly:
//!   cargo test -p ferrum-models --test runner_weight_test -- --ignored

use std::path::PathBuf;

fn qwen3_model_path() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let path = home
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join("models--Qwen--Qwen3-0.6B")
        .join("snapshots");

    // Find first snapshot directory
    if path.exists() {
        for entry in std::fs::read_dir(&path).ok()? {
            let entry = entry.ok()?;
            if entry.file_type().ok()?.is_dir() {
                let config = entry.path().join("config.json");
                if config.exists() {
                    return Some(entry.path());
                }
            }
        }
    }
    None
}

#[test]
#[ignore] // Requires downloaded model
fn test_load_qwen3_weights_into_runner() {
    let model_path = match qwen3_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Qwen3-0.6B not found, skipping. Run: ferrum pull qwen3:0.6b");
            return;
        }
    };

    eprintln!("Loading from: {}", model_path.display());

    // Parse config
    let config_path = model_path.join("config.json");
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();

    // Build ModelDefinition manually from config.json
    let model_def = ferrum_models::definition::ModelDefinition {
        architecture: ferrum_models::registry::Architecture::Qwen3,
        hidden_size: config_json["hidden_size"].as_u64().unwrap() as usize,
        intermediate_size: config_json["intermediate_size"].as_u64().unwrap() as usize,
        vocab_size: config_json["vocab_size"].as_u64().unwrap() as usize,
        num_hidden_layers: config_json["num_hidden_layers"].as_u64().unwrap() as usize,
        num_attention_heads: config_json["num_attention_heads"].as_u64().unwrap() as usize,
        num_key_value_heads: config_json
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        max_position_embeddings: config_json["max_position_embeddings"].as_u64().unwrap() as usize,
        rope_theta: config_json.get("rope_theta").and_then(|v| v.as_f64()),
        ..Default::default()
    };

    eprintln!(
        "Config: hidden={}, layers={}, heads={}, kv_heads={:?}, vocab={}",
        model_def.hidden_size,
        model_def.num_hidden_layers,
        model_def.num_attention_heads,
        model_def.num_key_value_heads,
        model_def.vocab_size,
    );

    // Convert to TransformerConfig
    let cfg = ferrum_models::model_config::qwen3_config(&model_def);
    eprintln!(
        "TransformerConfig: layers={}, h={}, heads={}, kv={}, hd={}, vocab={}",
        cfg.num_layers,
        cfg.hidden_size,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.vocab_size,
    );

    // Load weights via Candle VarBuilder
    let loader = ferrum_models::SafeTensorsLoader::new(model_path.to_str().unwrap());
    let vb = loader
        .load_varbuilder(&candle_core::Device::Cpu, candle_core::DType::F32)
        .expect("Failed to load VarBuilder");

    eprintln!("VarBuilder loaded, extracting weights...");

    let weights = ferrum_models::model_config::weight_loader::load_model_weights(&vb, &cfg)
        .expect("Failed to load model weights");

    eprintln!(
        "Weights loaded: {} layers, embed={} floats, lm_head={} floats",
        weights.layers.len(),
        weights.embed.len(),
        weights.lm_head_w.len(),
    );

    // Verify dimensions
    assert_eq!(weights.layers.len(), cfg.num_layers);
    assert_eq!(weights.embed.len(), cfg.vocab_size * cfg.hidden_size);
    assert_eq!(weights.lm_head_w.len(), cfg.vocab_size * cfg.hidden_size);

    // Create ModelRunner and decode one token
    let mut runner = ferrum_kernels::backend::runner::ModelRunner::<
        ferrum_kernels::backend::cpu::CpuBackend,
    >::new(cfg.clone(), weights);

    eprintln!("ModelRunner created, running decode...");

    let logits = runner.decode("test", 1, 0); // token_id=1 (arbitrary), pos=0
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "logits should be finite"
    );
    assert!(
        logits.iter().any(|x| *x != 0.0),
        "logits should be non-zero"
    );

    // Find argmax
    let (max_idx, max_val) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    eprintln!("Decode result: argmax={max_idx}, max_logit={max_val:.4}");
    eprintln!("SUCCESS: ModelRunner decode with real Qwen3-0.6B weights");
}
