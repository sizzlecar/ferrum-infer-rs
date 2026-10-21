use super::*;

fn builder(entries: &[(&str, &str, RuntimeConfigSource)]) -> FerrumConfigBuilder {
    let runtime = RuntimeConfigSnapshot::from_entries(
        entries
            .iter()
            .map(|(key, value, source)| RuntimeConfigEntry::new(*key, *value, *source)),
    );
    FerrumConfigBuilder::new(runtime)
        .with_hardware_capabilities(HardwareCapabilities {
            backend: "metal".into(),
            ..HardwareCapabilities::unknown()
        })
        .with_model_capabilities(ModelCapabilities {
            max_context_len: Some(40960),
            num_hidden_layers: Some(28),
            head_dim: Some(128),
            kv_heads: Some(8),
            ..ModelCapabilities::unknown()
        })
        .with_selected_kv_cache_layout(SelectedKvCacheLayout::Paged {
            block_size_tokens: 16,
        })
        .with_selected_kv_cache_storage(Some(crate::KvCacheStorage::F32))
}

#[test]
fn legacy_selected_capacity_reaches_allocator_config() {
    let resolved = builder(&[
        ("FERRUM_KV_CAPACITY", "8192", RuntimeConfigSource::Default),
        ("FERRUM_MAX_MODEL_LEN", "2048", RuntimeConfigSource::Cli),
        ("FERRUM_PAGED_MAX_SEQS", "16", RuntimeConfigSource::Cli),
        ("FERRUM_MAX_BATCHED_TOKENS", "512", RuntimeConfigSource::Cli),
    ])
    .resolve()
    .unwrap();
    let mut engine = crate::EngineConfig::default();
    engine
        .apply_runtime_config_snapshot(&resolved.runtime_config)
        .unwrap();
    let execution = engine.llama_execution_config();
    assert_eq!(execution.kv_capacity, Some(2048));
    assert_eq!(execution.options.kv_max_blocks, Some(2048));
    assert_eq!(execution.paged_max_seqs, 16);
    assert_eq!(engine.runtime.max_model_len, Some(2048));
    assert_eq!(engine.batching.max_num_batched_tokens, 512);
    let document = resolved.effective_config_document();
    assert_eq!(document["selected_kv_capacity"], 2048);
    assert_eq!(document["admission"]["kv_block_count"], 2048);
    assert_eq!(
        document["admission"]["memory_estimate"]["kv_bytes_per_token"],
        229376
    );
    assert_eq!(
        document["admission"]["memory_estimate"]["kv_capacity_bytes"],
        7u64 * 1024 * 1024 * 1024
    );
    assert_eq!(
        document["admission"]["memory_estimate"]["is_resident_usage"],
        false
    );
}

#[test]
fn legacy_explicit_pool_and_larger_logical_capacity_preserve_provenance() {
    let capacity = RuntimeConfigEntry::new("FERRUM_KV_CAPACITY", "4096", RuntimeConfigSource::Cli);
    let blocks = RuntimeConfigEntry::new(
        "FERRUM_KV_MAX_BLOCKS",
        "1024",
        RuntimeConfigSource::ConfigFile,
    );
    let resolved = builder(&[
        ("FERRUM_KV_CAPACITY", "4096", RuntimeConfigSource::Cli),
        (
            "FERRUM_KV_MAX_BLOCKS",
            "1024",
            RuntimeConfigSource::ConfigFile,
        ),
        ("FERRUM_MAX_MODEL_LEN", "2048", RuntimeConfigSource::Cli),
    ])
    .resolve()
    .unwrap();
    assert!(resolved.runtime_config.entries.contains(&capacity));
    assert!(resolved.runtime_config.entries.contains(&blocks));
}

#[test]
fn legacy_explicit_logical_capacity_cannot_undercut_request_ceiling() {
    for source in [
        RuntimeConfigSource::Cli,
        RuntimeConfigSource::Env,
        RuntimeConfigSource::ConfigFile,
        RuntimeConfigSource::ScriptCase,
    ] {
        let error = builder(&[
            ("FERRUM_KV_CAPACITY", "1024", source),
            ("FERRUM_MAX_MODEL_LEN", "2048", RuntimeConfigSource::Cli),
        ])
        .resolve()
        .unwrap_err();
        assert!(
            matches!(error, AutoConfigError::InvalidOverride { key, .. } if key == "FERRUM_KV_CAPACITY")
        );
    }
}

#[test]
fn legacy_default_is_visible_and_compiled_plan_defaults_stay_owned_by_plan() {
    let resolved = builder(&[]).resolve().unwrap();
    assert_eq!(resolved.runtime_usize("FERRUM_KV_CAPACITY"), Some(512));
    assert_eq!(resolved.runtime_usize("FERRUM_KV_MAX_BLOCKS"), Some(32));
    let mut engine = crate::EngineConfig::default();
    engine
        .apply_runtime_config_snapshot(&resolved.runtime_config)
        .unwrap();
    assert_eq!(engine.llama_execution_config().paged_max_seqs, 1);

    let resolved = builder(&[("FERRUM_MAX_MODEL_LEN", "2048", RuntimeConfigSource::Cli)])
        .with_execution_planning_model(ExecutionPlanningModel::CompiledPlan)
        .with_selected_kv_cache_layout(SelectedKvCacheLayout::Unknown)
        .with_selected_kv_cache_storage(None)
        .resolve()
        .unwrap();
    assert_eq!(resolved.runtime_usize("FERRUM_KV_CAPACITY"), None);
    assert_eq!(resolved.runtime_usize("FERRUM_KV_MAX_BLOCKS"), None);
}

#[test]
fn legacy_automatic_pool_does_not_exceed_usable_sequence_capacity() {
    let resolved = builder(&[
        ("FERRUM_MAX_MODEL_LEN", "2048", RuntimeConfigSource::Cli),
        ("FERRUM_PAGED_MAX_SEQS", "1", RuntimeConfigSource::Cli),
        ("FERRUM_MAX_BATCHED_TOKENS", "512", RuntimeConfigSource::Cli),
    ])
    .resolve()
    .unwrap();
    assert_eq!(resolved.runtime_usize("FERRUM_KV_CAPACITY"), Some(2048));
    assert_eq!(resolved.runtime_usize("FERRUM_KV_MAX_BLOCKS"), Some(128));

    // An explicit pool can retain more shared/prefix blocks than one live sequence.
    let resolved = builder(&[
        ("FERRUM_MAX_MODEL_LEN", "2048", RuntimeConfigSource::Cli),
        ("FERRUM_PAGED_MAX_SEQS", "1", RuntimeConfigSource::Cli),
        ("FERRUM_KV_MAX_BLOCKS", "256", RuntimeConfigSource::Cli),
    ])
    .resolve()
    .unwrap();
    assert_eq!(resolved.runtime_usize("FERRUM_KV_MAX_BLOCKS"), Some(256));
}

#[test]
fn contiguous_storage_does_not_claim_a_paged_pool_and_pool_overflow_is_rejected() {
    let document = builder(&[])
        .with_selected_kv_cache_layout(SelectedKvCacheLayout::Contiguous)
        .resolve()
        .unwrap()
        .effective_config_document();
    assert_eq!(
        document["admission"]["memory_estimate"]["kv_bytes_per_token"],
        229376
    );
    assert!(document["admission"]["memory_estimate"]["kv_capacity_bytes"].is_null());
    assert_eq!(
        document["admission"]["memory_estimate"]["applies_to_selected_state_layout"],
        false
    );
    let overflowing_blocks = usize::MAX.to_string();
    assert!(
        matches!(builder(&[("FERRUM_KV_MAX_BLOCKS", &overflowing_blocks, RuntimeConfigSource::Cli)]).resolve(), Err(AutoConfigError::InvalidOverride { key, .. }) if key == "FERRUM_KV_MAX_BLOCKS")
    );
}

#[test]
fn inferred_batch_is_clamped_but_explicit_batch_is_not() {
    for source in [
        RuntimeConfigSource::Default,
        RuntimeConfigSource::MemoryProfile,
        RuntimeConfigSource::Cli,
        RuntimeConfigSource::ConfigFile,
        RuntimeConfigSource::Env,
        RuntimeConfigSource::ScriptCase,
    ] {
        let result = builder(&[
            ("FERRUM_MAX_MODEL_LEN", "1024", RuntimeConfigSource::Cli),
            ("FERRUM_PAGED_MAX_SEQS", "1", RuntimeConfigSource::Cli),
            ("FERRUM_MAX_BATCHED_TOKENS", "2048", source),
        ])
        .resolve();
        if matches!(
            source,
            RuntimeConfigSource::Default | RuntimeConfigSource::MemoryProfile
        ) {
            let resolved = result.unwrap();
            assert_eq!(
                resolved.runtime_usize("FERRUM_MAX_BATCHED_TOKENS"),
                Some(1024)
            );
            assert_eq!(
                resolved
                    .runtime_config
                    .entries
                    .iter()
                    .find(|entry| entry.key == "FERRUM_MAX_BATCHED_TOKENS")
                    .unwrap()
                    .source,
                source
            );
        } else {
            assert!(
                matches!(result, Err(AutoConfigError::InvalidOverride { key, .. }) if key == "FERRUM_MAX_BATCHED_TOKENS")
            );
        }
    }
}

#[test]
fn automatic_pool_can_fit_one_long_context_or_reports_available_capacity() {
    let entries = [
        ("FERRUM_MAX_MODEL_LEN", "40960", RuntimeConfigSource::Cli),
        ("FERRUM_PAGED_MAX_SEQS", "1", RuntimeConfigSource::Cli),
    ];
    let resolved = builder(&entries).resolve().unwrap();
    assert_eq!(resolved.runtime_usize("FERRUM_KV_CAPACITY"), Some(40960));
    assert_eq!(resolved.runtime_usize("FERRUM_KV_MAX_BLOCKS"), Some(2560));

    let error = builder(&entries)
        .with_model_capabilities(ModelCapabilities {
            max_context_len: Some(40960),
            num_hidden_layers: Some(28),
            kv_heads: Some(8),
            head_dim: Some(128),
            estimated_weight_bytes: Some(3 * 1024 * 1024 * 1024),
            ..ModelCapabilities::unknown()
        })
        .with_hardware_capabilities(HardwareCapabilities {
            backend: "metal".into(),
            vram_bytes: Some(8 * 1024 * 1024 * 1024),
            ..HardwareCapabilities::unknown()
        })
        .resolve()
        .unwrap_err();
    assert!(
        matches!(error, AutoConfigError::InvalidOverride { key, reason } if key == "FERRUM_MAX_MODEL_LEN" && reason.contains("cannot fit"))
    );
}
