use ferrum_types::*;

#[test]
fn engine_config_default_sane() {
    let cfg = EngineConfig::default();
    assert_eq!(cfg.model.model_id.as_str(), "default");
    assert!(cfg.batching.max_batch_size >= 1);
    assert_eq!(cfg.kv_cache.max_blocks, 2048);
    assert_eq!(cfg.batching.max_num_batched_tokens, 2048);
    assert!(cfg.scheduler.prompt_token_estimate);
    assert!(cfg.monitoring.enable_metrics);
    assert_eq!(cfg.memory.usable_capacity_bytes, None);
}

#[test]
fn memory_config_resolves_exact_usable_capacity_without_float_rounding() {
    let mut memory = MemoryConfig::default();
    memory.usable_capacity_bytes = Some(777);

    let budget = memory.resolve_capacity_budget(1_000).unwrap();

    assert_eq!(budget.capacity_bytes, 1_000);
    assert_eq!(budget.usable_capacity_bytes, 777);
    assert_eq!(budget.reserve_bytes, 223);
}

#[test]
fn memory_config_rejects_exact_usable_capacity_above_pool_ceiling() {
    let mut memory = MemoryConfig::default();
    memory.pool_size = Some(512);
    memory.usable_capacity_bytes = Some(513);

    let error = memory.resolve_capacity_budget(1_000).unwrap_err();

    assert!(error.contains("1..=512"));
}

#[test]
fn memory_config_preserves_threshold_based_default_budget() {
    let memory = MemoryConfig::default();

    let budget = memory.resolve_capacity_budget(1_000).unwrap();

    assert_eq!(budget.capacity_bytes, 1_000);
    // Preserve the historical f32 threshold behavior exactly. Callers that
    // need byte-exact capacity use `usable_capacity_bytes` instead.
    assert_eq!(budget.usable_capacity_bytes, 949);
    assert_eq!(budget.reserve_bytes, 51);
}

#[test]
fn memory_config_missing_exact_budget_uses_backward_compatible_default() {
    let mut value = serde_json::to_value(MemoryConfig::default()).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("usable_capacity_bytes");

    let memory: MemoryConfig = serde_json::from_value(value).unwrap();

    assert_eq!(memory.usable_capacity_bytes, None);
}

#[test]
fn engine_config_applies_runtime_snapshot() {
    let snapshot = RuntimeConfigSnapshot::from_entries([
        RuntimeConfigEntry::new("FERRUM_KV_MAX_BLOCKS", "4096", RuntimeConfigSource::Env),
        RuntimeConfigEntry::new(
            "FERRUM_MAX_BATCHED_TOKENS",
            "8192",
            RuntimeConfigSource::Env,
        ),
        RuntimeConfigEntry::new("FERRUM_PAGED_MAX_SEQS", "7", RuntimeConfigSource::Env),
        RuntimeConfigEntry::new(
            "FERRUM_RUNTIME_MEMORY_BUDGET_BYTES",
            "12345",
            RuntimeConfigSource::Cli,
        ),
        RuntimeConfigEntry::new(
            "FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE",
            "1",
            RuntimeConfigSource::Env,
        ),
        RuntimeConfigEntry::new("FERRUM_BATCHED_GRAPH", "1", RuntimeConfigSource::Cli),
        RuntimeConfigEntry::new("FERRUM_REUSABLE_EXECUTION", "0", RuntimeConfigSource::Cli),
        RuntimeConfigEntry::new(
            "FERRUM_SCHEDULER_TRACE_JSONL",
            "/tmp/sched.jsonl",
            RuntimeConfigSource::Env,
        ),
    ]);
    let mut cfg = EngineConfig::default();

    cfg.apply_runtime_config_snapshot(&snapshot).unwrap();

    assert_eq!(cfg.kv_cache.max_blocks, 4096);
    assert_eq!(cfg.batching.max_num_batched_tokens, 8192);
    assert_eq!(cfg.scheduler.max_running_requests, 7);
    assert_eq!(cfg.memory.usable_capacity_bytes, Some(12_345));
    assert!(cfg.scheduler.prompt_token_estimate);
    assert!(cfg.backend.enable_cuda_graphs);
    assert!(!cfg.backend.enable_reusable_execution);
    assert_eq!(
        cfg.runtime.scheduler_trace_jsonl.as_deref(),
        Some(std::path::Path::new("/tmp/sched.jsonl"))
    );
}

#[test]
fn engine_config_rejects_invalid_runtime_snapshot() {
    let snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
        "FERRUM_KV_MAX_BLOCKS",
        "0",
        RuntimeConfigSource::Env,
    )]);
    let mut cfg = EngineConfig::default();

    let err = cfg.apply_runtime_config_snapshot(&snapshot).unwrap_err();

    assert!(err.contains("FERRUM_KV_MAX_BLOCKS"));
}

#[test]
fn engine_config_applies_build_composition_knobs() {
    // FERRUM_MODEL_PATH / SPEC_* / DTYPE / METAL_DTYPE / TP used to be read
    // straight from env by builder.rs and registry.rs. They now land in
    // EngineConfig.runtime here; the builder/registry read the typed field.
    let snapshot = RuntimeConfigSnapshot::from_entries([
        RuntimeConfigEntry::new(
            "FERRUM_MODEL_PATH",
            "/models/target",
            RuntimeConfigSource::Env,
        ),
        RuntimeConfigEntry::new(
            "FERRUM_SPEC_DRAFT",
            "/models/draft",
            RuntimeConfigSource::Env,
        ),
        RuntimeConfigEntry::new("FERRUM_SPEC_N", "8", RuntimeConfigSource::Env),
        RuntimeConfigEntry::new("FERRUM_DTYPE", "fp32", RuntimeConfigSource::Env),
        RuntimeConfigEntry::new("FERRUM_METAL_DTYPE", "fp16", RuntimeConfigSource::Env),
        RuntimeConfigEntry::new("FERRUM_TP", "4", RuntimeConfigSource::Env),
    ]);
    let mut cfg = EngineConfig::default();

    cfg.apply_runtime_config_snapshot(&snapshot).unwrap();

    assert_eq!(cfg.runtime.model_path.as_deref(), Some("/models/target"));
    assert_eq!(cfg.runtime.spec_draft.as_deref(), Some("/models/draft"));
    assert_eq!(cfg.runtime.spec_n, Some(8));
    assert_eq!(cfg.runtime.dtype.as_deref(), Some("fp32"));
    assert_eq!(cfg.runtime.metal_dtype.as_deref(), Some("fp16"));
    assert_eq!(cfg.runtime.tp, Some(4));
}

#[test]
fn engine_config_build_knobs_ignore_empty_spec_draft() {
    // Empty FERRUM_SPEC_DRAFT must resolve to None (disabled), matching the
    // old builder env parse that treated "" as "no draft".
    let snapshot = RuntimeConfigSnapshot::from_entries([
        RuntimeConfigEntry::new("FERRUM_SPEC_DRAFT", "", RuntimeConfigSource::Env),
        RuntimeConfigEntry::new("FERRUM_SPEC_N", "not-a-number", RuntimeConfigSource::Env),
    ]);
    let mut cfg = EngineConfig::default();

    cfg.apply_runtime_config_snapshot(&snapshot).unwrap();

    assert_eq!(cfg.runtime.spec_draft, None);
    assert_eq!(cfg.runtime.spec_n, None);
}

#[test]
fn scheduler_config_default_sane() {
    let cfg = SchedulerConfig::default();
    assert!(matches!(cfg.policy, SchedulingPolicy::Priority));
    assert!(cfg.max_waiting_requests >= 1);
    assert!(cfg.prompt_token_estimate);
    assert_eq!(cfg.prefill_first_until_active, None);
    assert_eq!(cfg.active_decode_prefill_chunk, None);
}

#[test]
fn scheduler_config_missing_prompt_token_estimate_uses_default_true() {
    let mut value = serde_json::to_value(SchedulerConfig::default()).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("prompt_token_estimate");

    let cfg: SchedulerConfig = serde_json::from_value(value).unwrap();

    assert!(cfg.prompt_token_estimate);
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
    assert!(cfg.enable_reusable_execution);
}

#[test]
fn sampling_presets_contains_expected() {
    let presets = SamplingPresets::default();
    assert!(presets.presets.contains_key("greedy"));
    assert!(presets.presets.get("creative").is_some());
    assert!(presets.presets.get("precise").is_some());
}
