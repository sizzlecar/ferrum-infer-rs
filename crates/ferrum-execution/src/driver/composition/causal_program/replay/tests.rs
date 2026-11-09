use super::*;

fn policy(tokens: u64, mode: ReusableExecutionPreparationMode) -> ResolvedRuntimePolicy {
    let capture = ReusableExecutionCaptureConfig {
        preparation: mode,
        maximum_executables_per_lane: (mode == ReusableExecutionPreparationMode::OnDemand)
            .then_some(7),
        ..Default::default()
    };
    let reusable = resolve_reusable_execution_policy(
        2,
        tokens,
        64,
        &[],
        &capture,
        ReusableExecutionOnDemandAdmission::AfterWarmup,
        true,
    )
    .unwrap()
    .policy;
    ResolvedRuntimePolicy::new(
        "policy.causal.replay-test",
        ContractVersion::new(1, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            capacity_bytes: 1 << 30,
            reserve_bytes: 1 << 20,
            maximum_active_sequences: 2,
            dynamic_storage_profile_order: vec![DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )
            .unwrap()],
            checkpoint_capacity: None,
        },
        AdmissionPolicy {
            maximum_queue_depth: 2,
            maximum_scheduled_tokens: tokens,
            sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
            sequence_backing_growth: SequenceBackingGrowthPolicy::Exact,
            allow_defer: true,
            cancellation_check_interval_steps: 1,
        },
        AttentionExecutionPolicy::Portable,
        ExecutionDeterminismRequirement::BitwiseSameRuntimeWithReplay,
        Some(reusable),
    )
    .unwrap()
}

#[test]
fn causal_replay_reserves_two_rows_without_increasing_admission_or_catalog_limits() {
    let original = policy(4, ReusableExecutionPreparationMode::OnDemand);
    let resolved = resolve_policy(original.clone(), true).unwrap();
    assert_eq!(resolved.memory(), original.memory());
    assert_eq!(resolved.admission(), original.admission());
    assert_eq!(
        resolved.attention_execution(),
        original.attention_execution()
    );
    assert_eq!(
        resolved.execution_determinism(),
        original.execution_determinism()
    );
    assert_ne!(resolved.fingerprint_str(), original.fingerprint_str());
    let reusable = resolved.reusable_execution().unwrap();
    assert_eq!(reusable.maximum_reusable_lanes(), 1);
    assert_eq!(reusable.maximum_executables_per_lane(), Some(7));
    let class = ReusableExecutionClassId::new(TWO_ROW_CLASS).unwrap();
    assert!(reusable.smallest_covering_bucket(&class, 1, 2, 0).is_some());
    assert!(reusable.smallest_covering_bucket(&class, 2, 4, 0).is_some());
    assert!(reusable.smallest_covering_bucket(&class, 2, 5, 0).is_none());
    assert!(reusable.smallest_covering_bucket(&class, 3, 4, 0).is_none());
    for original in original.reusable_execution().unwrap().buckets() {
        assert!(reusable.buckets().contains(original));
    }
}

#[test]
fn causal_replay_keeps_two_row_width_within_the_real_token_ceiling() {
    let resolved =
        resolve_policy(policy(3, ReusableExecutionPreparationMode::OnDemand), true).unwrap();
    let reusable = resolved.reusable_execution().unwrap();
    let class = ReusableExecutionClassId::new(TWO_ROW_CLASS).unwrap();
    assert!(reusable.smallest_covering_bucket(&class, 1, 2, 0).is_some());
    assert!(reusable.smallest_covering_bucket(&class, 2, 4, 0).is_none());
    assert!(reusable
        .program_policy()
        .unwrap()
        .programs()
        .iter()
        .all(|program| program.shape().token_capacity() <= 3));
}

#[test]
fn causal_mixed_replay_preserves_repeatability_and_original_capacity() {
    let original = policy(4, ReusableExecutionPreparationMode::OnDemand);
    let strict = resolve_policy(original.clone(), true).unwrap();
    let mixed = resolve_policy_with_eager_boundaries(original.clone(), true).unwrap();
    assert_eq!(mixed.memory(), original.memory());
    assert_eq!(mixed.admission(), original.admission());
    assert_eq!(mixed.reusable_execution(), strict.reusable_execution());
    assert_eq!(
        mixed.execution_determinism(),
        ExecutionDeterminismRequirement::BitwiseSameRuntime
    );
    assert_ne!(mixed.fingerprint_str(), strict.fingerprint_str());
    assert_eq!(
        strict.execution_determinism(),
        ExecutionDeterminismRequirement::BitwiseSameRuntimeWithReplay
    );
}

#[test]
fn causal_replay_rejects_startup_state_reexecution_and_installation_only_programs() {
    assert!(resolve_policy(policy(4, ReusableExecutionPreparationMode::Startup), true).is_err());
    assert!(resolve_policy(policy(4, ReusableExecutionPreparationMode::OnDemand), false).is_err());
}
