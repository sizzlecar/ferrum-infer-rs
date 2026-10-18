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
            "FERRUM_REUSABLE_EXECUTION_EXACT_DECODE_WIDTHS",
            "1,3,7",
            RuntimeConfigSource::ConfigFile,
        ),
        RuntimeConfigEntry::new(
            "FERRUM_REUSABLE_EXECUTION_MAX_AUTOMATIC_EXACT_DECODE_WIDTH",
            "24",
            RuntimeConfigSource::ConfigFile,
        ),
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
        cfg.backend.reusable_execution_capture.exact_decode_widths,
        Some(vec![1, 3, 7])
    );
    assert_eq!(
        cfg.backend
            .reusable_execution_capture
            .maximum_automatic_exact_decode_width,
        24
    );
    assert_eq!(
        cfg.runtime.scheduler_trace_jsonl.as_deref(),
        Some(std::path::Path::new("/tmp/sched.jsonl"))
    );
}

#[test]
fn engine_config_rejects_empty_reusable_exact_width_list() {
    let snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
        "FERRUM_REUSABLE_EXECUTION_EXACT_DECODE_WIDTHS",
        "",
        RuntimeConfigSource::ConfigFile,
    )]);
    let mut cfg = EngineConfig::default();

    let error = cfg.apply_runtime_config_snapshot(&snapshot).unwrap_err();

    assert!(error.contains("FERRUM_REUSABLE_EXECUTION_EXACT_DECODE_WIDTHS"));
}

#[test]
fn engine_config_rejects_reusable_capture_widths_above_independent_hard_bound() {
    for (key, value) in [
        ("FERRUM_REUSABLE_EXECUTION_EXACT_DECODE_WIDTHS", "1,32,33"),
        (
            "FERRUM_REUSABLE_EXECUTION_MAX_AUTOMATIC_EXACT_DECODE_WIDTH",
            "33",
        ),
    ] {
        let snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
            key,
            value,
            RuntimeConfigSource::ConfigFile,
        )]);
        let mut cfg = EngineConfig::default();

        let error = cfg.apply_runtime_config_snapshot(&snapshot).unwrap_err();

        assert!(error.contains(key));
        assert!(error.contains("1..=32"));
    }
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
    // FERRUM_MODEL_PATH / DTYPE / METAL_DTYPE / TP used to be read
    // straight from env by builder.rs and registry.rs. They now land in
    // EngineConfig.runtime here; the builder/registry read the typed field.
    let snapshot = RuntimeConfigSnapshot::from_entries([
        RuntimeConfigEntry::new(
            "FERRUM_MODEL_PATH",
            "/models/target",
            RuntimeConfigSource::Env,
        ),
        RuntimeConfigEntry::new("FERRUM_DTYPE", "fp32", RuntimeConfigSource::Env),
        RuntimeConfigEntry::new("FERRUM_METAL_DTYPE", "fp16", RuntimeConfigSource::Env),
        RuntimeConfigEntry::new("FERRUM_TP", "4", RuntimeConfigSource::Env),
    ]);
    let mut cfg = EngineConfig::default();

    cfg.apply_runtime_config_snapshot(&snapshot).unwrap();

    assert_eq!(cfg.runtime.model_path.as_deref(), Some("/models/target"));
    assert_eq!(cfg.runtime.dtype.as_deref(), Some("fp32"));
    assert_eq!(cfg.runtime.metal_dtype.as_deref(), Some("fp16"));
    assert_eq!(cfg.runtime.tp, Some(4));
}

#[test]
fn retired_execution_controls_reject_before_mutating_engine_config() {
    for key in [
        "FERRUM_WHOLE_PROMPT_PREFIX_CACHE",
        "FERRUM_SPEC_DRAFT",
        "FERRUM_SPEC_N",
    ] {
        for value in ["", "false", "0", "1", "/models/draft"] {
            for source in [
                RuntimeConfigSource::Env,
                RuntimeConfigSource::ConfigFile,
                RuntimeConfigSource::Cli,
            ] {
                let snapshot = RuntimeConfigSnapshot::from_entries([
                    RuntimeConfigEntry::new("FERRUM_KV_CAPACITY", "1024", source),
                    RuntimeConfigEntry::new(key, value, source),
                ]);
                let mut cfg = EngineConfig::default();
                let before = serde_json::to_value(&cfg).unwrap();
                let error = cfg.apply_runtime_config_snapshot(&snapshot).unwrap_err();
                assert!(error.contains(key) && error.contains("retired"), "{error}");
                assert_eq!(serde_json::to_value(&cfg).unwrap(), before);
            }
        }
    }
}

#[test]
fn retired_config_fields_are_rejected_even_when_disabled_or_null() {
    use serde_json::json;

    let baseline = serde_json::to_value(EngineConfig::default()).unwrap();
    serde_json::from_value::<EngineConfig>(baseline.clone()).unwrap();
    for (section, key, values) in [
        (
            "runtime",
            "prefix_cache_enabled",
            vec![json!(false), json!(true), json!(null)],
        ),
        (
            "runtime",
            "spec_draft",
            vec![json!(""), json!("/models/draft"), json!(null)],
        ),
        ("runtime", "spec_n", vec![json!(0), json!(4), json!(null)]),
        (
            "scheduler",
            "enable_load_balancing",
            vec![json!(false), json!(true)],
        ),
        (
            "scheduler",
            "fair_share_weights",
            vec![json!({}), json!({"client": 1.0})],
        ),
        (
            "scheduler",
            "enable_sla_enforcement",
            vec![json!(false), json!(true)],
        ),
        (
            "kv_cache",
            "enable_prefix_caching",
            vec![json!(false), json!(true)],
        ),
        ("kv_cache", "prefix_cache_size", vec![json!(0), json!(100)]),
        ("runtime", "unknown_runtime_option", vec![json!(false)]),
        ("scheduler", "unknown_scheduler_option", vec![json!(false)]),
        ("kv_cache", "unknown_cache_option", vec![json!(false)]),
    ] {
        for value in values {
            let mut wire = baseline.clone();
            wire[section]
                .as_object_mut()
                .unwrap()
                .insert(key.to_string(), value);
            let error = serde_json::from_value::<EngineConfig>(wire).unwrap_err();
            assert!(error.to_string().contains("unknown field"), "{error}");
            assert!(error.to_string().contains(key), "{error}");
        }
    }
}

#[test]
fn scheduler_policy_only_accepts_continuous_batch() {
    let baseline = serde_json::to_value(SchedulerConfig::default()).unwrap();
    assert_eq!(baseline["policy"], "ContinuousBatch");
    assert_eq!(
        serde_json::to_value(SchedulingPolicy::default()).unwrap(),
        "ContinuousBatch"
    );
    let restored: SchedulerConfig = serde_json::from_value(baseline.clone()).unwrap();
    assert!(matches!(restored.policy, SchedulingPolicy::ContinuousBatch));
    for policy in ["FCFS", "Priority", "FairShare", "SJF", "RoundRobin"] {
        let mut wire = baseline.clone();
        wire["policy"] = serde_json::Value::String(policy.to_string());
        let error = serde_json::from_value::<SchedulerConfig>(wire).unwrap_err();
        assert!(error.to_string().contains("unknown variant"), "{error}");
        assert!(error.to_string().contains(policy), "{error}");
    }
}

#[test]
fn scheduler_config_default_sane() {
    let cfg = SchedulerConfig::default();
    assert!(matches!(cfg.policy, SchedulingPolicy::ContinuousBatch));
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
    ]);
    let mut cfg = SchedulerConfig::default();

    cfg.apply_runtime_config_snapshot(&snapshot).unwrap();

    assert!(cfg.prompt_token_estimate);
    assert_eq!(cfg.prefill_first_until_active, Some(4));
    assert_eq!(cfg.active_decode_prefill_chunk, Some(64));
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
fn retired_profiler_snapshots_reject_before_engine_or_scheduler_mutation() {
    for key in [
        "FERRUM_BATCH_PREFILL_PROF",
        "FERRUM_RBD_PROF",
        "FERRUM_UNIFIED_POST_PROF",
        "FERRUM_NEXT_BATCH_PROF",
        "FERRUM_SCHED_NONE_PROF",
    ] {
        for value in ["", "0", "1"] {
            for source in [
                RuntimeConfigSource::Env,
                RuntimeConfigSource::ConfigFile,
                RuntimeConfigSource::Cli,
            ] {
                let snapshot = RuntimeConfigSnapshot::from_entries([
                    RuntimeConfigEntry::new("FERRUM_KV_CAPACITY", "1024", source),
                    RuntimeConfigEntry::new("FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE", "0", source),
                    RuntimeConfigEntry::new(key, value, source),
                ]);
                let mut engine = EngineConfig::default();
                let engine_before = serde_json::to_value(&engine).unwrap();
                let error = engine.apply_runtime_config_snapshot(&snapshot).unwrap_err();
                assert!(error.contains(key) && error.contains("retired"), "{error}");
                assert_eq!(serde_json::to_value(&engine).unwrap(), engine_before);

                // Library users may configure the scheduler without building
                // an Engine. Its own entry point must reject before mutation.
                let mut scheduler = SchedulerConfig::default();
                let scheduler_before = serde_json::to_value(&scheduler).unwrap();
                let error = scheduler
                    .apply_runtime_config_snapshot(&snapshot)
                    .unwrap_err();
                assert!(error.contains(key) && error.contains("retired"), "{error}");
                assert_eq!(serde_json::to_value(&scheduler).unwrap(), scheduler_before);
            }
        }
    }
}

#[test]
fn retired_profiler_fields_reject_in_typed_configuration() {
    for field in [
        "batch_prefill_prof",
        "rbd_prof",
        "unified_post_prof",
        "next_batch_prof",
    ] {
        for enabled in [false, true] {
            let mut wire = serde_json::to_value(EngineConfig::default()).unwrap();
            wire["runtime"][field] = serde_json::json!(enabled);
            let error = serde_json::from_value::<EngineConfig>(wire).unwrap_err();
            assert!(error.to_string().contains("unknown field"), "{error}");
            assert!(error.to_string().contains(field), "{error}");
        }
    }
    for enabled in [false, true] {
        let mut wire = serde_json::to_value(SchedulerConfig::default()).unwrap();
        wire["scheduler_none_prof"] = serde_json::json!(enabled);
        let error = serde_json::from_value::<SchedulerConfig>(wire).unwrap_err();
        assert!(error.to_string().contains("unknown field"), "{error}");
        assert!(error.to_string().contains("scheduler_none_prof"), "{error}");
    }
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
    assert_eq!(cfg.reusable_execution_capture.exact_decode_widths, None);
    assert_eq!(
        cfg.reusable_execution_capture
            .maximum_automatic_exact_decode_width,
        DEFAULT_MAXIMUM_AUTOMATIC_EXACT_DECODE_WIDTH
    );
    assert_eq!(
        DEFAULT_MAXIMUM_AUTOMATIC_EXACT_DECODE_WIDTH,
        MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH
    );
}

#[test]
fn backend_config_without_capture_policy_uses_backward_compatible_default() {
    let mut serialized = serde_json::to_value(BackendConfig::default()).unwrap();
    serialized
        .as_object_mut()
        .unwrap()
        .remove("reusable_execution_capture");

    let cfg: BackendConfig = serde_json::from_value(serialized).unwrap();

    assert_eq!(
        cfg.reusable_execution_capture,
        ReusableExecutionCaptureConfig::default()
    );
}

#[test]
fn reusable_execution_capture_partial_config_keeps_automatic_safety_default() {
    let capture: ReusableExecutionCaptureConfig = serde_json::from_value(serde_json::json!({
        "exact_decode_widths": [1, 2, 4, 8, 16, 24, 32]
    }))
    .unwrap();

    assert_eq!(
        capture.exact_decode_widths,
        Some(vec![1, 2, 4, 8, 16, 24, 32])
    );
    assert_eq!(
        capture.maximum_automatic_exact_decode_width,
        DEFAULT_MAXIMUM_AUTOMATIC_EXACT_DECODE_WIDTH
    );
}

#[test]
fn reusable_execution_capture_explicit_policy_round_trips() {
    let expected = ReusableExecutionCaptureConfig {
        exact_decode_widths: Some(vec![1, 3, 7, 32]),
        maximum_automatic_exact_decode_width: 16,
        ..Default::default()
    };

    let serialized = serde_json::to_value(&expected).unwrap();
    let actual: ReusableExecutionCaptureConfig = serde_json::from_value(serialized).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn reusable_execution_preparation_resolves_declared_capabilities() {
    use ReusableExecutionPreparationMode::{Auto, OnDemand, Startup};
    assert_eq!(Auto.resolve(true).unwrap(), OnDemand);
    assert_eq!(Auto.resolve(false).unwrap(), Startup);
    assert_eq!(Startup.resolve(true).unwrap(), Startup);
    assert_eq!(Startup.resolve(false).unwrap(), Startup);
    assert_eq!(OnDemand.resolve(true).unwrap(), OnDemand);
    assert!(OnDemand.resolve(false).is_err());
    for mode in [Auto, Startup, OnDemand] {
        let mut config = EngineConfig::default();
        let snapshot = RuntimeConfigSnapshot::from_entries(vec![RuntimeConfigEntry::new(
            "FERRUM_REUSABLE_EXECUTION_PREPARATION",
            mode.as_runtime_value(),
            RuntimeConfigSource::ConfigFile,
        )]);
        config.apply_runtime_config_snapshot(&snapshot).unwrap();
        assert_eq!(config.backend.reusable_execution_capture.preparation, mode);
    }
    assert!("sometimes"
        .parse::<ReusableExecutionPreparationMode>()
        .is_err());
}

#[test]
fn sampling_presets_contains_expected() {
    let presets = SamplingPresets::default();
    assert!(presets.presets.contains_key("greedy"));
    assert!(presets.presets.get("creative").is_some());
    assert!(presets.presets.get("precise").is_some());
}
