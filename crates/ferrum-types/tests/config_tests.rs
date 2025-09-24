use ferrum_types::*;

#[test]
fn engine_config_default_sane() {
    let cfg = EngineConfig::default();
    assert_eq!(cfg.model.model_id.as_str(), "default");
    assert!(cfg.batching.max_batch_size >= 1);
    assert!(cfg.monitoring.enable_metrics);
}

#[test]
fn scheduler_config_default_sane() {
    let cfg = SchedulerConfig::default();
    assert!(matches!(cfg.policy, SchedulingPolicy::Priority));
    assert!(cfg.max_waiting_requests >= 1);
}

#[test]
fn kv_cache_config_default_sane() {
    let cfg = KvCacheConfig::default();
    assert!(matches!(cfg.cache_type, KvCacheType::Contiguous));
    assert!(cfg.block_size > 0);
}

#[test]
fn memory_config_default_sane() {
    let cfg = MemoryConfig::default();
    assert!(cfg.alignment >= 1);
}

#[test]
fn backend_config_default_sane() {
    let cfg = BackendConfig::default();
    assert!(matches!(cfg.backend_type, BackendType::Candle));
    assert!(cfg.enable_optimizations);
}

#[test]
fn sampling_presets_contains_expected() {
    let presets = SamplingPresets::default();
    assert!(presets.presets.contains_key("greedy"));
    assert!(presets.presets.get("creative").is_some());
    assert!(presets.presets.get("precise").is_some());
}
