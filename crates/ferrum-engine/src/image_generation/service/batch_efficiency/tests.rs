use super::*;

fn context() -> ImageCapacitySearchContext {
    ImageCapacitySearchContext {
        shape: [1; 32],
        capacity_bytes: 100,
    }
}

fn observe(policy: &mut BatchEfficiency, count: usize, actual: u64, install: u64, retained: bool) {
    policy.observe(
        ExecutionContext {
            context: context(),
            adapter: None,
            capacity_upper: 3,
            batch_size: count,
        },
        Duration::from_micros(actual),
        ImageBatchExecutionFeedback {
            stages: vec![ImageBatchStageFeedback {
                stage: ImageStage::TextEncoder,
                initialization_us: install,
                weights_retained: retained,
            }],
        },
        16,
    );
}

fn select(policy: &mut BatchEfficiency, queued: usize) -> usize {
    policy
        .select(Some(context()), &None, queued.min(3), 3)
        .count
}

#[test]
fn recurring_reload_can_probe_one_adjacent_batch_without_learning_from_cold_install() {
    let mut policy = BatchEfficiency::default();
    observe(&mut policy, 1, 30_000, 20_000, true);
    assert!(policy.0.as_ref().unwrap().batches.is_empty());
    observe(&mut policy, 1, 3_300, 0, true);
    // A short successful queue is not a ceiling on future batching.
    assert_eq!(select(&mut policy, 3), 3);
    observe(&mut policy, 3, 8_500, 0, false);
    // This batch evicted TE, but its hot installation cost is still unknown.
    assert_eq!(select(&mut policy, 3), 3);
    assert!(!policy.select(Some(context()), &None, 3, 3).allow_growth);
    observe(&mut policy, 3, 10_200, 1_700, false);
    assert_eq!(select(&mut policy, 3), 2);
    assert!(!policy.select(Some(context()), &None, 3, 3).allow_growth);
    // The first B2 pays the previous B3's reload. Its sustained cost must not
    // keep that one-off penalty after B2 has proved the weights stay resident.
    observe(&mut policy, 2, 7_200, 1_700, true);
    assert_eq!(select(&mut policy, 3), 2);
    assert_eq!(policy.0.as_ref().unwrap().explored_from, Some(3));
    assert_eq!(
        policy
            .0
            .as_ref()
            .unwrap()
            .batches
            .iter()
            .find(|b| b.count == 2)
            .unwrap()
            .cost(&policy.0.as_ref().unwrap().stages),
        Some(5_500)
    );
}

#[test]
fn observed_savings_pay_for_larger_reassessment_instead_of_permanently_pinning_a_size() {
    let mut policy = BatchEfficiency::default();
    observe(&mut policy, 3, 8_000, 0, true);
    observe(&mut policy, 2, 4_000, 0, true);
    // Each smaller dispatch earns 8_000*2/3 - 4_000 us. Once the total
    // reaches one actual larger dispatch, spend it on a real larger batch.
    while policy.0.as_ref().unwrap().savings_us < 8_000 {
        assert_eq!(select(&mut policy, 3), 2);
        observe(&mut policy, 2, 4_000, 0, true);
    }
    assert_eq!(select(&mut policy, 3), 3);
    assert!(!policy.select(Some(context()), &None, 3, 3).allow_growth);
    observe(&mut policy, 3, 8_000, 0, true);
    assert_eq!(policy.0.as_ref().unwrap().savings_us, 0);
    assert_eq!(select(&mut policy, 3), 2);
}

#[test]
fn changed_shape_adapter_or_capacity_frontier_reopens_the_candidate_without_free_byte_jitter() {
    let mut policy = BatchEfficiency::default();
    observe(&mut policy, 3, 9_000, 0, true);
    observe(&mut policy, 2, 4_000, 0, true);
    assert_eq!(select(&mut policy, 3), 2);
    let mut changed = context();
    changed.capacity_bytes += 1;
    assert_eq!(policy.select(Some(changed), &None, 3, 3).count, 2);
    assert_eq!(policy.select(Some(changed), &None, 3, 4).count, 3);
    changed.shape = [2; 32];
    assert_eq!(policy.select(Some(changed), &None, 3, 3).count, 3);
    let adapter = Some(Arc::new(ImageAdapterBatchIdentity {
        effective_identity: "different".into(),
        strength_bits: 1.0_f32.to_bits(),
    }));
    assert_eq!(policy.select(Some(context()), &adapter, 3, 3).count, 3);
    assert_eq!(policy.select(None, &None, 3, 3).count, 3);
}

#[test]
fn an_ineffective_adjacent_observation_does_not_start_a_downward_probe_ladder() {
    let mut policy = BatchEfficiency::default();
    observe(&mut policy, 3, 30_000, 20_000, false);
    observe(&mut policy, 3, 10_000, 1_000, false);
    observe(&mut policy, 3, 10_000, 1_000, false);
    assert_eq!(select(&mut policy, 3), 2);
    observe(&mut policy, 2, 9_000, 1_000, false);
    assert_eq!(select(&mut policy, 3), 3);
    observe(&mut policy, 3, 10_000, 1_000, false);
    assert_eq!(select(&mut policy, 3), 3);
}
