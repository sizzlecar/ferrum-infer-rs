use super::*;

#[path = "../../../../../ferrum-interfaces/tests/vnext_sequence_checkpoint_contract/fixture.rs"]
mod checkpoint_fixture;
#[path = "../../../../../ferrum-interfaces/tests/vnext_core_contract/mod.rs"]
mod vnext_core_contract;

fn insert(
    index: &mut PrefixIndex<Arc<str>>,
    prefix: &[u32],
    input: &[u32],
    name: &str,
) -> Arc<str> {
    let owner: Arc<str> = Arc::from(name);
    drop(index.insert(Arc::from(prefix), Arc::from(input), Arc::clone(&owner)));
    owner
}

#[test]
fn longest_exact_tokens_leave_a_suffix_and_refresh_lru() {
    let mut index = PrefixIndex::default();
    insert(&mut index, &[1, 2], &[1, 2, 3], "short");
    insert(&mut index, &[1, 2, 3], &[1, 2, 3, 4], "long");
    insert(&mut index, &[9], &[9, 8], "other");
    assert_eq!(
        index.longest(&[1, 2, 3, 8], false, |_| true).as_deref(),
        Some("long")
    );
    assert_eq!(index.evict().as_deref(), Some("short"));
    assert_eq!(index.evict().as_deref(), Some("other"));
    assert_eq!(index.evict().as_deref(), Some("long"));
    assert!(index.longest(&[1, 2, 3], false, |_| true).is_none());
}

#[test]
fn input_dependency_and_suffix_legality_filter_before_longest_selection() {
    let mut index = PrefixIndex::default();
    insert(&mut index, &[1], &[1, 2, 3, 4], "short");
    insert(&mut index, &[1, 2], &[1, 2, 8], "long");
    assert_eq!(
        index.longest(&[1, 2, 3, 4], true, |_| true).as_deref(),
        Some("short")
    );
    assert_eq!(
        index.longest(&[1, 2, 3, 4], false, |n| n == 1).as_deref(),
        Some("short")
    );
    assert!(index.longest(&[1, 7], true, |_| true).is_none());
    assert!(index.longest(&[1, 2], false, |n| n == 2).is_none());
    assert!(index.longest(&[7, 2, 3, 4], false, |_| true).is_none());
}

#[test]
fn replacement_and_eviction_drop_index_ownership_but_not_restore_pins() {
    let mut index = PrefixIndex::default();
    let first = insert(&mut index, &[1], &[1, 2, 3], "first");
    let pin = index.longest(&[1, 2, 4], false, |_| true).unwrap();
    let replaced = index.insert(Arc::from([1, 2]), Arc::from([1, 2, 3]), Arc::from("second"));
    assert_eq!(Arc::strong_count(&first), 3);
    drop(replaced);
    assert_eq!(Arc::strong_count(&first), 2);
    drop(pin);
    assert_eq!(Arc::strong_count(&first), 1);
    assert_eq!(index.evict().as_deref(), Some("second"));
    assert!(index.evict().is_none());
}

#[test]
fn equal_length_prompts_and_partial_token_matches_are_not_prefix_hits() {
    let mut index = PrefixIndex::default();
    insert(&mut index, &[12, 3], &[12, 3, 4], "owner");
    assert!(index.longest(&[12, 3], false, |_| true).is_none());
    assert!(index.longest(&[1, 23, 4], false, |_| true).is_none());
    assert!(index.longest(&[12, 9, 4], false, |_| true).is_none());
    assert_eq!(
        index.longest(&[12, 3, 5], false, |_| true).as_deref(),
        Some("owner")
    );
}

#[test]
fn capture_selection_uses_completed_nonfinal_natural_boundary() {
    let chunks = [
        PrefillChunk::new(0, 4, 10).unwrap(),
        PrefillChunk::new(4, 4, 10).unwrap(),
        PrefillChunk::new(8, 2, 10).unwrap(),
    ];
    assert!(!capture_candidate(chunks[0]));
    assert!(capture_candidate(chunks[1]));
    assert!(!capture_candidate(chunks[2]));
    assert!(!capture_candidate(PrefillChunk::new(0, 10, 10).unwrap()));
}

#[test]
fn actual_plan_policy_and_provider_contract_gate_token_evidence() {
    use checkpoint_fixture::{Fixture, Spec};
    let span = TokenSpanWork::from_token_ids_with_fit(&[1, 2, 3], 0..2, 8).unwrap();
    let no_capacity = Fixture::build(Spec::default()).unwrap();
    assert!(usable_layout(&no_capacity.plan).is_none());
    // Mismatched evidence is never consulted on the disabled ordinary path.
    assert!(retain_token_evidence(&no_capacity.plan, span.clone(), &[9]).is_ok());
    let enabled_spec = || Spec {
        checkpoint_capacity: Some(CheckpointCapacityPolicy::new(1 << 20).unwrap()),
        ..Spec::default()
    };
    for spec in [
        Spec {
            declare_provider: false,
            ..enabled_spec()
        },
        Spec {
            conditioning: true,
            ..enabled_spec()
        },
        Spec {
            numerics: CheckpointPartitionNumerics::SamePartitionOnly,
            ..enabled_spec()
        },
    ] {
        let fixture = Fixture::build(spec).unwrap();
        assert!(usable_layout(&fixture.plan).is_none());
        assert!(retain_token_evidence(&fixture.plan, span.clone(), &[9]).is_ok());
    }
    let enabled = Fixture::build(enabled_spec()).unwrap();
    let captured = Fixture::build(Spec {
        numerics: CheckpointPartitionNumerics::CapturedExecutionContinuation,
        ..enabled_spec()
    })
    .unwrap();
    assert!(usable_layout(&captured.plan).is_some());
    assert!(retain_token_evidence(&captured.plan, span.clone(), &[1, 2, 9]).is_err());
    assert!(usable_layout(&enabled.plan).is_some());
    assert!(retain_token_evidence(&enabled.plan, span.clone(), &[1, 2, 9]).is_err());
    let tracked = retain_token_evidence(&enabled.plan, span.clone(), &[1, 2, 3]).unwrap();
    assert_eq!(tracked, span);
    assert_eq!(tracked.fingerprint(), span.fingerprint());
    assert_eq!(
        serde_json::to_value(&tracked).unwrap(),
        serde_json::to_value(&span).unwrap()
    );
}

#[test]
fn product_flag_resolves_to_existing_memory_budget_without_reserving_slots() {
    use ferrum_kernels::backend::reference::ReferenceVNextComposition;
    use ferrum_types::{DataType, ModelType};
    let composition =
        ReferenceVNextComposition::create(DeviceId::new("device.prefix-policy").unwrap()).unwrap();
    let info = ModelInfo {
        model_id: "policy-fixture".into(),
        model_type: ModelType::Custom("policy-fixture".into()),
        num_parameters: 0,
        hidden_size: 4,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 16,
        max_sequence_length: 64,
        dtype: DataType::FP16,
        device: Device::CPU,
        version: None,
        license: None,
        metadata: Default::default(),
    };
    let mut engine = EngineConfig::default();
    // Enabling the independent whole-prompt cache does not allocate vNext
    // sequence-state checkpoint capacity.
    engine.runtime.prefix_cache_enabled = true;
    engine.runtime.prefix_state_cache_enabled = false;
    let disabled =
        VNextExecutorConfig::from_engine_config(&engine, &info, composition.runtime().as_ref())
            .unwrap();
    assert!(disabled
        .runtime_policy
        .memory()
        .checkpoint_capacity
        .is_none());
    engine.runtime.prefix_state_cache_enabled = true;
    let enabled =
        VNextExecutorConfig::from_engine_config(&engine, &info, composition.runtime().as_ref())
            .unwrap();
    let memory = enabled.runtime_policy.memory();
    assert_eq!(
        memory.checkpoint_capacity.unwrap().maximum_retained_bytes(),
        memory.capacity_bytes - memory.reserve_bytes
    );
    assert_eq!(
        memory.maximum_active_sequences,
        disabled.runtime_policy.memory().maximum_active_sequences
    );
    assert_eq!(
        enabled.runtime_policy.admission(),
        disabled.runtime_policy.admission()
    );
}

#[test]
fn foreground_eviction_requires_a_capacity_source_checkpoint_release_can_change() {
    let checkpoint_domain = CapacityDomainId::new(11).unwrap();
    let unrelated_domain = CapacityDomainId::new(12).unwrap();
    let domains = BTreeSet::from([checkpoint_domain]);
    let wait = |sources: &[CapacityAvailabilitySource]| {
        CapacityWaitCondition::from_observation(
            1,
            sources
                .iter()
                .map(|source| CapacityAvailabilityEpoch::new(*source, 1).unwrap())
                .collect(),
        )
        .unwrap()
    };
    assert!(!checkpoint_release_can_help(
        &wait(&[CapacityAvailabilitySource::ActiveSequenceSlots]),
        &domains
    ));
    assert!(!checkpoint_release_can_help(
        &wait(&[
            CapacityAvailabilitySource::ActiveSequenceSlots,
            CapacityAvailabilitySource::Domain(checkpoint_domain)
        ]),
        &domains
    ));
    assert!(!checkpoint_release_can_help(
        &wait(&[CapacityAvailabilitySource::Domain(unrelated_domain)]),
        &domains
    ));
    assert!(checkpoint_release_can_help(
        &wait(&[CapacityAvailabilitySource::Domain(checkpoint_domain)]),
        &domains
    ));
    assert!(checkpoint_release_can_help(
        &wait(&[CapacityAvailabilitySource::PlanDeviceBudget]),
        &domains
    ));
    assert!(!checkpoint_release_can_help(
        &wait(&[CapacityAvailabilitySource::ProcessDeviceCapacity]),
        &BTreeSet::new()
    ));
}

#[test]
fn foreground_growth_preserves_checkpoint_and_requires_a_fresh_probe() {
    let mut index = PrefixIndex::default();
    let owner = insert(&mut index, &[1, 2], &[1, 2, 3], "captured");
    let mut recovery = PrefixPressureMaintenance::new(index.entries.len());
    let attempts = std::cell::Cell::new(0);
    let result = recovery
        .recover(
            true,
            || {
                attempts.set(attempts.get() + 1);
                Ok(Some(()))
            },
            || index.evict().is_some(),
        )
        .unwrap();
    assert!(matches!(result, PrefixPressureRecovery::Maintained(())));
    assert_eq!(attempts.get(), 1);
    assert_eq!(Arc::strong_count(&owner), 2);
    assert_eq!(
        index.longest(&[1, 2, 4], false, |_| true).as_deref(),
        Some("captured")
    );
    // A subsequent failed probe cannot turn one successful growth into an
    // unconditional eviction or an unbounded series of maintenance attempts.
    let repeated = recovery
        .recover::<()>(
            true,
            || panic!("one foreground call already attempted maintenance"),
            || index.evict().is_some(),
        )
        .unwrap();
    assert!(matches!(repeated, PrefixPressureRecovery::Unchanged));
    assert_eq!(Arc::strong_count(&owner), 2);
}

#[test]
fn foreground_capacity_limit_evicts_after_maintenance_and_preserves_restore_pin() {
    let mut index = PrefixIndex::default();
    let first = insert(&mut index, &[1], &[1, 2], "first");
    let pin = Arc::clone(&first);
    insert(&mut index, &[9], &[9, 8], "second");
    let mut recovery = PrefixPressureMaintenance::new(index.entries.len());
    let maintenance_finished = std::cell::Cell::new(false);
    let result = recovery
        .recover::<()>(
            true,
            || {
                maintenance_finished.set(true);
                Err(VNextError::DeviceCapacityUnavailable(
                    DeviceCapacityPressure::new(
                        DeviceCapacityPressureScope::PlanBudget,
                        "device.capacity-policy-test".into(),
                        32,
                        64,
                        80,
                        64,
                        80,
                    )
                    .unwrap(),
                ))
            },
            || {
                assert!(maintenance_finished.get());
                index.evict().is_some()
            },
        )
        .unwrap();
    assert!(matches!(result, PrefixPressureRecovery::Evicted));
    assert_eq!(Arc::strong_count(&first), 2);
    assert_eq!(&*pin, "first");
    assert_eq!(
        index.longest(&[9, 8, 7], false, |_| true).as_deref(),
        Some("second")
    );
    let next = recovery
        .recover::<()>(
            true,
            || panic!("pressure recovery must not repeatedly grow for each entry"),
            || index.evict().is_some(),
        )
        .unwrap();
    assert!(matches!(next, PrefixPressureRecovery::Evicted));
    assert!(index.entries.is_empty());
    assert_eq!(Arc::strong_count(&first), 2);
}

#[test]
fn unrelated_inapplicable_and_faulted_pressure_never_discards_checkpoint() {
    let mut index = PrefixIndex::default();
    let owner = insert(&mut index, &[1], &[1, 2], "captured");
    let mut recovery = PrefixPressureMaintenance::new(index.entries.len());
    let unrelated = recovery
        .recover::<()>(
            false,
            || panic!("non-checkpoint capacity must not trigger optional maintenance"),
            || index.evict().is_some(),
        )
        .unwrap();
    assert!(matches!(unrelated, PrefixPressureRecovery::Unchanged));
    let not_applicable = recovery
        .recover::<()>(true, || Ok(None), || index.evict().is_some())
        .unwrap();
    assert!(matches!(not_applicable, PrefixPressureRecovery::Unchanged));
    assert_eq!(Arc::strong_count(&owner), 2);

    let error = PrefixPressureMaintenance::new(index.entries.len())
        .recover::<()>(
            true,
            || {
                Err(VNextError::InvalidExecutionPlan {
                    reason: "foreign maintenance authority".into(),
                })
            },
            || index.evict().is_some(),
        )
        .err()
        .expect("a contract failure must reach the caller");
    assert!(matches!(error, VNextError::InvalidExecutionPlan { .. }));
    assert_eq!(Arc::strong_count(&owner), 2);
}

#[test]
fn last_regular_maintenance_attempt_eviction_permits_a_fresh_attempt_with_a_pin() {
    let mut index = PrefixIndex::default();
    let checkpoint = insert(&mut index, &[1], &[1, 2], "checkpoint");
    let pin = Arc::clone(&checkpoint);
    let mut budget = PrefixPressureMaintenance::new(index.entries.len());
    let mut actual_attempts = 0;
    assert!(budget.allows_backing_attempt(actual_attempts));
    actual_attempts += 1;
    assert!(budget.allows_backing_attempt(actual_attempts));
    actual_attempts += 1;
    assert!(!budget.allows_backing_attempt(actual_attempts));

    // The second actual maintenance returned device pressure. Removing an
    // index owner grants one fresh maintenance attempt, even when a native
    // restore still pins its bytes; it does not assert a capacity epoch change.
    assert!(budget.evict_after_pressure(|| index.evict().is_some()));
    assert_eq!(Arc::strong_count(&checkpoint), 2);
    assert_eq!(&*pin, "checkpoint");
    assert!(budget.allows_backing_attempt(actual_attempts));
    assert_eq!(actual_attempts, 2);
    actual_attempts += 1;
    assert!(!budget.allows_backing_attempt(actual_attempts));

    // A concurrent insertion cannot enlarge a running call's eviction budget.
    let later = insert(&mut index, &[9], &[9, 8], "later");
    assert!(!budget.evict_after_pressure(|| index.evict().is_some()));
    assert_eq!(Arc::strong_count(&later), 2);
    assert_eq!(
        index.longest(&[9, 8, 7], false, |_| true).as_deref(),
        Some("later")
    );
}

#[test]
fn empty_or_raced_index_removal_grants_no_extra_maintenance_attempt() {
    let mut index = PrefixIndex::default();
    let checkpoint = insert(&mut index, &[1], &[1, 2], "checkpoint");
    let mut budget = PrefixPressureMaintenance::new(index.entries.len());
    drop(index.evict());
    assert!(!budget.evict_after_pressure(|| index.evict().is_some()));
    assert!(!budget.allows_backing_attempt(MAX_BACKING_MAINTENANCE_ATTEMPTS));
    assert_eq!(Arc::strong_count(&checkpoint), 1);
    let mut empty = PrefixPressureMaintenance::new(0);
    assert!(!empty.evict_after_pressure(|| panic!("no starting entry can be removed")));
    assert!(!empty.allows_backing_attempt(MAX_BACKING_MAINTENANCE_ATTEMPTS));
}

#[test]
fn native_snapshot_reports_index_extents_after_replacement_and_pinned_eviction() {
    use checkpoint_fixture::{Fixture, Spec};
    struct RetainedCheckpoint {
        extent_bytes: u64,
    }
    let fixture = Fixture::build(Spec {
        checkpoint_capacity: Some(CheckpointCapacityPolicy::new(1 << 20).unwrap()),
        ..Spec::default()
    })
    .unwrap();
    let metrics = PrefixCacheMetrics::default();
    let mut index = PrefixIndex::default();
    let snapshot = |index: &PrefixIndex<Arc<RetainedCheckpoint>>| {
        index.snapshot(&fixture.plan, true, &metrics, |owner| owner.extent_bytes)
    };
    let first = Arc::new(RetainedCheckpoint {
        extent_bytes: 65536,
    });
    drop(index.insert(Arc::from([1]), Arc::from([1, 2, 3]), Arc::clone(&first)));
    let pin = index.longest(&[1, 2, 4], false, |_| true).unwrap();
    assert_eq!(snapshot(&index)["entries"], 1);
    assert_eq!(snapshot(&index)["bytes"], 65536);
    // A deeper capture replaces the same input, not an additional entry or a
    // pressure eviction. The old native owner can remain pinned by a restore.
    drop(index.insert(
        Arc::from([1, 2]),
        Arc::from([1, 2, 3]),
        Arc::new(RetainedCheckpoint {
            extent_bytes: 131072,
        }),
    ));
    let current = snapshot(&index);
    assert_eq!(current["entries"], 1);
    assert_eq!(current["bytes"], 131072);
    assert_eq!(current["evictions"], 0);
    drop(index.evict());
    let empty = snapshot(&index);
    assert_eq!(empty["entries"], 0);
    assert_eq!(empty["bytes"], 0);
    assert_eq!(empty["excludes_evicted_inflight_pins"], true);
    assert_eq!(Arc::strong_count(&first), 2);
    assert_eq!(pin.extent_bytes, 65536);
}

#[test]
fn native_snapshot_supplies_zero_health_fields_and_actual_capability_reasons() {
    use checkpoint_fixture::{Fixture, Spec};
    let index = PrefixIndex::<u64>::default();
    let metrics = PrefixCacheMetrics::default();
    let capacity = Some(CheckpointCapacityPolicy::new(1 << 20).unwrap());
    let disabled = Fixture::build(Spec::default()).unwrap();
    let snapshot = index.snapshot(&disabled.plan, false, &metrics, |bytes| *bytes);
    assert_eq!(snapshot["requested"], false);
    assert_eq!(snapshot["enabled"], false);
    // These are all scalar fields consumed by cache health / Prometheus. Even
    // disabled native execution must override server text-LCP observations.
    for key in [
        "entries",
        "bytes",
        "hits",
        "misses",
        "evictions",
        "saved_prefill_tokens",
    ] {
        assert_eq!(snapshot[key].as_u64(), Some(0), "missing native {key}");
    }
    assert_eq!(snapshot["source"], "vnext-native-sequence-checkpoint-cache");
    assert_eq!(snapshot["position"], "model-executor");
    let unsupported = Fixture::build(Spec {
        checkpoint_capacity: capacity,
        declare_provider: false,
        ..Spec::default()
    })
    .unwrap();
    let snapshot = index.snapshot(&unsupported.plan, true, &metrics, |bytes| *bytes);
    assert_eq!(snapshot["requested"], true);
    assert_eq!(snapshot["enabled"], false);
    assert_eq!(
        snapshot["unsupported_reasons"],
        serde_json::json!(unsupported.reasons())
    );
    for (spec, reason) in [
        (
            Spec {
                conditioning: true,
                ..Spec::default()
            },
            "conditioning_inputs",
        ),
        (
            Spec {
                numerics: CheckpointPartitionNumerics::SamePartitionOnly,
                ..Spec::default()
            },
            "same_partition_execution_trace",
        ),
    ] {
        let fixture = Fixture::build(Spec {
            checkpoint_capacity: capacity,
            ..spec
        })
        .unwrap();
        let snapshot = index.snapshot(&fixture.plan, true, &metrics, |bytes| *bytes);
        assert_eq!(snapshot["enabled"], false);
        assert_eq!(
            snapshot["missing_executor_evidence"],
            serde_json::json!([reason])
        );
        assert!(!snapshot.contains_key("unsupported_reasons"));
    }
    let enabled = Fixture::build(Spec {
        checkpoint_capacity: capacity,
        ..Spec::default()
    })
    .unwrap();
    let snapshot = index.snapshot(&enabled.plan, true, &metrics, |bytes| *bytes);
    assert_eq!(snapshot["enabled"], true);
    assert!(!snapshot.contains_key("unsupported_reasons"));
    assert!(!snapshot.contains_key("missing_executor_evidence"));
}

#[test]
fn restore_metrics_wait_for_consuming_ack_and_ignore_cancelled_or_dropped_outputs() {
    // This exercises the actual engine publication envelope and metrics
    // callback, not a simulated native copy or a numeric continuation claim.
    struct PublicationGuard {
        cancellations: Arc<AtomicU64>,
        acknowledged: bool,
    }
    impl PublicationGuard {
        fn acknowledge(mut self) {
            self.acknowledged = true;
        }
    }
    impl Drop for PublicationGuard {
        fn drop(&mut self) {
            if !self.acknowledged {
                self.cancellations.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    let metrics = Arc::new(PrefixCacheMetrics::default());
    let cancellations = Arc::new(AtomicU64::new(0));
    let cancelled = Arc::new(AtomicBool::new(false));
    let output = |restored: usize, prompt: usize| {
        let metrics = Arc::clone(&metrics);
        let cancelled = Arc::clone(&cancelled);
        let publication = PublicationGuard {
            cancellations: Arc::clone(&cancellations),
            acknowledged: false,
        };
        let request = RequestId::new();
        PlanRuntimePrefixRestoreOutput::new(
            request.clone(),
            restored,
            prompt,
            Arc::new(ferrum_testkit::MockKvCacheHandle::new(request, 1, restored)),
            move || {
                metrics.acknowledge_restore(restored, || {
                    if cancelled.load(Ordering::Acquire) {
                        return Err(FerrumError::cancelled("exact restore target was cancelled"));
                    }
                    publication.acknowledge();
                    Ok(())
                })
            },
        )
    };
    let dropped = output(3, 4).unwrap();
    assert_eq!(cancellations.load(Ordering::Relaxed), 0);
    drop(dropped);
    assert!(output(4, 4).is_err());
    let pending = output(3, 4).unwrap();
    assert_eq!(cancellations.load(Ordering::Relaxed), 2);
    cancelled.store(true, Ordering::Release);
    assert!(pending.acknowledge().is_err());
    assert_eq!(cancellations.load(Ordering::Relaxed), 3);
    assert_eq!(metrics.hits.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.saved_prefill_tokens.load(Ordering::Relaxed), 0);

    cancelled.store(false, Ordering::Release);
    let pending = output(3, 4).unwrap();
    assert_eq!(metrics.hits.load(Ordering::Relaxed), 0);
    pending.acknowledge().unwrap();
    output(2, 4).unwrap().acknowledge().unwrap();
    assert_eq!(metrics.hits.load(Ordering::Relaxed), 2);
    assert_eq!(metrics.saved_prefill_tokens.load(Ordering::Relaxed), 5);
    assert_eq!(metrics.misses.load(Ordering::Relaxed), 0);
    assert_eq!(cancellations.load(Ordering::Relaxed), 3);
    metrics.reset();
    assert_eq!(metrics.hits.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.saved_prefill_tokens.load(Ordering::Relaxed), 0);
}
