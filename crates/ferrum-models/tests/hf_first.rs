use tempfile::TempDir;
use std::fs;
use std::path::PathBuf;

use ferrum_models as fm;
use fm::ModelSourceResolver;
use fm::{ConfigManager};
use fm::source::{DefaultModelSourceResolver, ModelSourceConfig, ModelFormat};
use fm::registry::DefaultModelRegistry;

fn write_file(dir: &PathBuf, name: &str, content: &str) {
    fs::write(dir.join(name), content).unwrap();
}

#[tokio::test]
async fn load_hf_config_from_source() {
    let tmp = TempDir::new().unwrap();
    let model_dir = tmp.path().join("hf_model");
    fs::create_dir_all(&model_dir).unwrap();

    let config_json = r#"{
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": false
    }"#;
    write_file(&model_dir, "config.json", config_json);

    // tokenizer.json 不强制用于本测试

    let resolver = DefaultModelSourceResolver::with_defaults();
    let source = resolver
        .resolve(model_dir.to_str().unwrap(), None)
        .await
        .expect("resolve local path");

    assert_eq!(source.format, ModelFormat::HuggingFace);

    let mut cfg_mgr = ConfigManager::new();
    let cfg = cfg_mgr
        .load_from_source(&source)
        .await
        .expect("load hf config");

    assert!(matches!(cfg.architecture, fm::Architecture::Llama));
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.intermediate_size, 11008);
    assert_eq!(cfg.vocab_size, 32000);
    assert_eq!(cfg.num_hidden_layers, 32);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.max_position_embeddings, 4096);
    assert!(matches!(cfg.norm_type, fm::NormType::RMSNorm));
    assert!((cfg.norm_eps - 1e-6).abs() < 1e-12);
    assert!(matches!(cfg.activation, fm::Activation::SiLU));
    assert_eq!(cfg.attention_config.attention_bias, false);
    assert_eq!(cfg.rope_theta, Some(10000.0));
    assert!(cfg.rope_scaling.is_none());
}

#[test]
fn config_manager_supports_hf() {
    let cfg_mgr = ConfigManager::new();
    assert!(cfg_mgr.supports_format(&ModelFormat::HuggingFace));
}

#[tokio::test]
async fn resolver_cache_hit_offline() {
    let tmp = TempDir::new().unwrap();

    // 自定义解析器配置，强制使用本地 cache 目录且离线
    let model_id = "microsoft/DialoGPT-medium";
    let revision = "main";

    let cfg = ModelSourceConfig {
        cache_dir: Some(tmp.path().to_path_buf()),
        hf_token: None,
        offline_mode: true,
        max_retries: 1,
        download_timeout: 10,
        use_file_lock: false,
    };

    let resolver = DefaultModelSourceResolver::new(cfg);

    // 构造与解析器一致的缓存路径结构：<cache>/models/{org--model}/{revision}/
    let cache_path = tmp
        .path()
        .join("models")
        .join("microsoft--DialoGPT-medium")
        .join(revision);
    fs::create_dir_all(&cache_path).unwrap();

    // 只需存在 config.json 即可通过基本校验
    write_file(&cache_path, "config.json", "{}\n");

    let source = resolver
        .resolve(model_id, Some(revision))
        .await
        .expect("resolve from cache in offline mode");

    assert_eq!(source.local_path, cache_path);
    assert!(source.from_cache);
    assert_eq!(source.format, ModelFormat::HuggingFace);
}

#[tokio::test]
async fn resolver_offline_without_cache_errors() {
    let tmp = TempDir::new().unwrap();

    let cfg = ModelSourceConfig {
        cache_dir: Some(tmp.path().to_path_buf()),
        hf_token: None,
        offline_mode: true,
        max_retries: 1,
        download_timeout: 10,
        use_file_lock: false,
    };

    let resolver = DefaultModelSourceResolver::new(cfg);

    let err = resolver
        .resolve("unknown/Not-Exist-Model", Some("main"))
        .await
        .err()
        .expect("should error when offline and no cache");

    let msg = format!("{}", err);
    assert!(msg.to_lowercase().contains("offline"));
}

#[tokio::test]
async fn registry_alias_and_discovery() {
    // 别名解析
    let registry = DefaultModelRegistry::with_defaults();
    let resolved = registry.resolve_model_id("llama2-7b");
    assert_eq!(resolved, "meta-llama/Llama-2-7b-hf");

    // 模型发现
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().to_path_buf();

    let mdir = root.join("my_llama");
    fs::create_dir_all(&mdir).unwrap();

    // 基本文件：config、tokenizer、权重
    write_file(&mdir, "config.json", r#"{"model_type":"llama"}"#);
    write_file(&mdir, "tokenizer.json", "{}\n");
    write_file(&mdir, "model.safetensors", "placeholder");

    let mut registry = DefaultModelRegistry::new();
    let found = registry
        .discover_models(&root)
        .await
        .expect("discover models");

    assert_eq!(found.len(), 1);
    let m = &found[0];
    assert_eq!(m.format, ModelFormat::HuggingFace);
    assert!(m.is_valid);
    assert!(matches!(m.architecture, Some(fm::Architecture::Llama))); 
}

#[tokio::test]
async fn load_hf_config_defaults_when_missing_fields() {
    let tmp = TempDir::new().unwrap();
    let model_dir = tmp.path().join("hf_default_model");
    fs::create_dir_all(&model_dir).unwrap();

    // 缺失大部分字段，只留 model_type
    write_file(&model_dir, "config.json", r#"{"model_type":"llama"}"#);

    let resolver = DefaultModelSourceResolver::with_defaults();
    let source = resolver
        .resolve(model_dir.to_str().unwrap(), None)
        .await
        .unwrap();

    let mut cfg_mgr = ConfigManager::new();
    let cfg = cfg_mgr.load_from_source(&source).await.unwrap();

    // 检查默认填充值
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.intermediate_size, 11008);
    assert_eq!(cfg.vocab_size, 32000);
    assert_eq!(cfg.num_hidden_layers, 32);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.max_position_embeddings, 4096);
}
