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
    assert!(!cfg.prompt_token_estimate);
    assert_eq!(cfg.prefill_first_until_active, None);
    assert_eq!(cfg.active_decode_prefill_chunk, None);
}

#[test]
fn scheduler_config_applies_runtime_snapshot() {
    let snapshot = RuntimeConfigSnapshot::from_entries([
        RuntimeConfigEntry::new(
            "FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE",
            "1",
            RuntimeConfigSource::Env,
        ),
        RuntimeConfigEntry::new(
            "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE",
            "4",
            RuntimeConfigSource::Env,
        ),
        RuntimeConfigEntry::new(
            "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK",
            "64",
            RuntimeConfigSource::Env,
        ),
        RuntimeConfigEntry::new("FERRUM_SCHED_NONE_PROF", "", RuntimeConfigSource::Env),
    ]);
    let mut cfg = SchedulerConfig::default();

    cfg.apply_runtime_config_snapshot(&snapshot).unwrap();

    assert!(cfg.prompt_token_estimate);
    assert_eq!(cfg.prefill_first_until_active, Some(4));
    assert_eq!(cfg.active_decode_prefill_chunk, Some(64));
    assert!(cfg.scheduler_none_prof);
}

#[test]
fn scheduler_config_rejects_invalid_runtime_snapshot() {
    let snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
        "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK",
        "not-a-number",
        RuntimeConfigSource::Env,
    )]);
    let mut cfg = SchedulerConfig::default();

    let err = cfg.apply_runtime_config_snapshot(&snapshot).unwrap_err();

    assert!(err.contains("FERRUM_ACTIVE_DECODE_PREFILL_CHUNK"));
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
