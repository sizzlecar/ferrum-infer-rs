use ferrum_types::*;

#[test]
fn engine_config_default_sane() {
    let c = EngineConfig::default();
    assert!(c.max_concurrent_requests >= 1);
    assert!(c.enable_streaming);
}

#[test]
fn scheduler_config_default_sane() {
    let c = SchedulerConfig::default();
    assert!(matches!(c.policy, SchedulingPolicy::Priority));
    assert!(c.max_waiting_requests >= 1);
}

#[test]
fn kv_cache_config_default_sane() {
    let c = KvCacheConfig::default();
    assert!(matches!(c.cache_type, KvCacheType::Paged));
    assert!(c.block_size > 0);
}

#[test]
fn memory_config_default_sane() {
    let c = MemoryConfig::default();
    assert!(c.alignment >= 1);
}

#[test]
fn backend_config_default_sane() {
    let c = BackendConfig::default();
    assert!(matches!(c.backend_type, BackendType::Candle));
    assert!(c.enable_optimizations);
}

#[test]
fn sampling_presets_contains_expected() {
    let p = SamplingPresets::default();
    assert!(p.presets.contains_key("greedy"));
    assert!(p.presets.get("creative").is_some());
    assert!(p.presets.get("precise").is_some());
}
