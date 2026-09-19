use super::*;

fn request(bytes: u64) -> StartupMemoryRequest {
    StartupMemoryRequest::from_snapshot(
        DeviceMemorySnapshot {
            capacity_bytes: bytes,
            available_bytes: bytes,
            source: "test_device".into(),
        },
        1.0,
        &RuntimeConfigSnapshot::default(),
    )
    .unwrap()
}

fn costs(work: StartupWorkload) -> Result<u64, String> {
    let (context, sequences, tokens) = match work {
        StartupWorkload::Prefill {
            context_tokens,
            chunk_tokens,
        } => (context_tokens, 1, chunk_tokens),
        StartupWorkload::Decode {
            active_sequences, ..
        } => (1, active_sequences, active_sequences),
    };
    // A hybrid model: weights + fixed sequence state + token-scaled state +
    // a provider workspace that jumps at a supported shape bucket boundary.
    Ok(1000 + sequences as u64 * (100 + context as u64 * 10) + (tokens as u64).div_ceil(8) * 80)
}

#[test]
fn decode_reserves_request_ceiling_without_reserving_full_sequence_state() {
    let target = StartupResourceLimits {
        context_tokens: 100,
        max_sequences: 16,
        max_batch_tokens: 32,
    };
    let fitted = fit_startup_resources(&request(4000), target, |work| {
        let request_bytes = match work {
            StartupWorkload::Prefill { context_tokens, .. } => context_tokens,
            StartupWorkload::Decode {
                context_tokens,
                active_sequences,
            } => context_tokens * active_sequences,
        };
        Ok(costs(work)? + request_bytes as u64)
    })
    .unwrap();
    assert_eq!(fitted.selected.context_tokens, target.context_tokens);
    assert!(fitted.selected.max_sequences > 1);
    assert!(fitted.selected.max_sequences < target.max_sequences);
    assert!(fitted.decode_peak_bytes <= 4000);
}

#[test]
fn explicit_concurrency_can_reduce_automatic_request_ceiling() {
    let mut requested = request(600);
    requested.sequences_is_explicit = true;
    let plan = fit_startup_resources(
        &requested,
        StartupResourceLimits {
            context_tokens: 100,
            max_sequences: 4,
            max_batch_tokens: 32,
        },
        |work| {
            Ok(match work {
                StartupWorkload::Prefill { context_tokens, .. } => 100 + 5 * context_tokens as u64,
                StartupWorkload::Decode {
                    context_tokens,
                    active_sequences,
                } => 100 + active_sequences as u64 * (4 * context_tokens as u64 + 1),
            })
        },
    )
    .unwrap();
    assert_eq!(plan.selected.max_sequences, 4);
    assert_eq!(plan.selected.context_tokens, 31);
    assert!(plan.context_peak_bytes <= 600 && plan.decode_peak_bytes <= 600);
}

#[test]
fn hardware_budget_changes_context_without_multiplying_it_by_concurrency() {
    let target = StartupResourceLimits {
        context_tokens: 4096,
        max_sequences: 16,
        max_batch_tokens: 32,
    };
    let small = fit_startup_resources(&request(4000), target, costs).unwrap();
    let large = fit_startup_resources(&request(8000), target, costs).unwrap();
    assert!(small.selected.context_tokens < large.selected.context_tokens);
    assert!(small.selected.max_sequences > 1);
    assert_eq!(small.selected.max_sequences, target.max_sequences);
    assert!(small.context_peak_bytes <= 4000 && small.decode_peak_bytes <= 4000);
    assert!(
        costs(StartupWorkload::Prefill {
            context_tokens: small.selected.context_tokens + 1,
            chunk_tokens: 32
        })
        .unwrap()
            > 4000
    );
    assert!(small
        .reasons
        .iter()
        .any(|reason| reason.contains("context tokens")));
}

#[test]
fn explicit_context_reduces_only_automatic_batch_and_preserves_the_request() {
    let mut requested = request(2200);
    requested.context_is_explicit = true;
    let target = StartupResourceLimits {
        context_tokens: 100,
        max_sequences: 4,
        max_batch_tokens: 32,
    };
    let fitted = fit_startup_resources(&requested, target, costs).unwrap();
    assert_eq!(fitted.selected.context_tokens, 100);
    assert_eq!(fitted.selected.max_batch_tokens, 8);
    requested.batch_is_explicit = true;
    assert!(fit_startup_resources(&requested, target, costs)
        .unwrap_err()
        .contains("explicit batch/context"));
}

#[test]
fn explicit_concurrency_is_never_silently_reduced() {
    let mut requested = request(1600);
    let target = StartupResourceLimits {
        context_tokens: 50,
        max_sequences: 16,
        max_batch_tokens: 32,
    };
    let fitted = fit_startup_resources(&requested, target, costs).unwrap();
    assert!(fitted.selected.max_sequences < 16);
    requested.sequences_is_explicit = true;
    assert!(fit_startup_resources(&requested, target, costs).is_err());
}

#[test]
fn no_runnable_configuration_and_provider_errors_are_reported() {
    let target = StartupResourceLimits {
        context_tokens: 100,
        max_sequences: 4,
        max_batch_tokens: 32,
    };
    assert!(fit_startup_resources(&request(1000), target, costs)
        .unwrap_err()
        .contains("minimum runnable"));
    assert_eq!(
        fit_startup_resources(&request(8000), target, |_| Err(
            "unsupported provider".into()
        ))
        .unwrap_err(),
        "unsupported provider"
    );
}

#[test]
fn automatic_budget_uses_available_memory_and_explicit_budget_keeps_precedence() {
    let device = DeviceMemorySnapshot {
        capacity_bytes: 10000,
        available_bytes: 4000,
        source: "shared_device".into(),
    };
    let automatic =
        StartupMemoryRequest::from_snapshot(device.clone(), 0.5, &RuntimeConfigSnapshot::default())
            .unwrap();
    assert_eq!(automatic.usable_capacity_bytes, 2000);
    let explicit = RuntimeConfigSnapshot::from_env_vars([
        ("FERRUM_RUNTIME_MEMORY_BUDGET_BYTES", "3500"),
        ("FERRUM_MAX_MODEL_LEN", "2048"),
    ]);
    let planned = StartupMemoryRequest::from_snapshot(device.clone(), 0.5, &explicit).unwrap();
    assert_eq!(planned.usable_capacity_bytes, 3500);
    assert!(planned.context_is_explicit);
    assert!(!planned.batch_is_explicit && !planned.sequences_is_explicit);
    let too_large = explicit.with_entry(
        "FERRUM_RUNTIME_MEMORY_BUDGET_BYTES",
        "5000",
        RuntimeConfigSource::Cli,
    );
    assert!(StartupMemoryRequest::from_snapshot(device.clone(), 0.5, &too_large).is_err());
    for invalid in [0.0, -0.1, f32::NAN, f32::INFINITY, 1.1] {
        assert!(StartupMemoryRequest::from_snapshot(
            device.clone(),
            invalid,
            &RuntimeConfigSnapshot::default()
        )
        .is_err());
    }
}

#[test]
fn inferred_values_remain_automatic_and_zero_available_is_not_fabricated() {
    let inferred = RuntimeConfigSnapshot::default()
        .with_entry("FERRUM_MAX_MODEL_LEN", "100", RuntimeConfigSource::Default)
        .with_entry(
            "FERRUM_PAGED_MAX_SEQS",
            "4",
            RuntimeConfigSource::MemoryProfile,
        );
    let device = DeviceMemorySnapshot {
        capacity_bytes: 4096,
        available_bytes: 0,
        source: "busy_device".into(),
    };
    assert!(StartupMemoryRequest::from_snapshot(device.clone(), 0.9, &inferred).is_err());
    let mut free = device;
    free.available_bytes = 4096;
    let planned = StartupMemoryRequest::from_snapshot(free, 0.9, &inferred).unwrap();
    assert!(!planned.context_is_explicit && !planned.sequences_is_explicit);
}

#[test]
fn fitted_limits_reach_engine_and_effective_config_with_consistent_provenance() {
    let mut requested = request(4000);
    requested.sequences_is_explicit = true;
    let plan = fit_startup_resources(
        &requested,
        StartupResourceLimits {
            context_tokens: 4096,
            max_sequences: 4,
            max_batch_tokens: 32,
        },
        costs,
    )
    .unwrap();
    let snapshot = RuntimeConfigSnapshot::default().with_entry(
        "FERRUM_PAGED_MAX_SEQS",
        "4",
        RuntimeConfigSource::Cli,
    );
    let mut resolved = crate::FerrumConfigBuilder::new(snapshot)
        .with_execution_resource_authority(crate::ExecutionResourceAuthority::PlanRuntime)
        .resolve()
        .unwrap();
    resolved.apply_startup_memory_plan(&plan);
    let mut engine = crate::EngineConfig::default();
    plan.apply_to_engine_config(&mut engine).unwrap();
    assert_eq!(
        engine.runtime.max_model_len,
        Some(plan.selected.context_tokens)
    );
    assert_eq!(engine.scheduler.max_running_requests, 4);
    assert_eq!(engine.memory.usable_capacity_bytes, Some(4000));
    let document = resolved.effective_config_document();
    assert_eq!(
        document["selected_max_model_len"],
        plan.selected.context_tokens
    );
    assert_eq!(document["selected_max_sequences"], 4);
    assert_eq!(
        document["startup_memory_plan"]["selected"]["context_tokens"],
        plan.selected.context_tokens
    );
    assert_eq!(
        resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_PAGED_MAX_SEQS")
            .unwrap()
            .source,
        RuntimeConfigSource::Cli
    );
    assert_eq!(
        resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_MAX_MODEL_LEN")
            .unwrap()
            .source,
        RuntimeConfigSource::MemoryProfile
    );
}
