use super::*;

#[test]
fn cohort_cost_prices_fragmented_tails_instead_of_only_per_image_execution() {
    let mut policy = BatchEfficiency::default();
    for (count, cost) in [(1, 7_000), (3, 17_000), (4, 23_000)] {
        observe(&mut policy, count, cost, 0, true);
    }
    // Three rows look cheaper per image, but 3+1 costs more than four.
    assert_eq!(policy.warm_start(Some(context()), &None, 4, 16), Some(4));
    assert_eq!(policy.warm_start(Some(context()), &None, 8, 16), Some(4));
    // A different cohort still benefits from 3+3: no fixed batch preference.
    assert_eq!(policy.warm_start(Some(context()), &None, 6, 16), Some(3));
    let domain = policy.0.as_mut().unwrap();
    domain.capacity_upper = 16;
    assert_eq!(policy.select(Some(context()), &None, 4, 16), 4);
}

#[test]
fn spilling_calibration_has_one_bootstrap_then_needs_earned_execution_savings() {
    let mut policy = BatchEfficiency::default();
    observe(&mut policy, 1, 30_000, 20_000, true);
    observe(&mut policy, 1, 4_000, 0, true);
    assert!(policy.allow_spilling_candidate(Some(context()), &None, 2, 1));
    assert!(!policy.allow_spilling_candidate(Some(context()), &None, 3, 1));
    observe_event(&mut policy, 2, 5_000, 0, false, true);
    assert!(policy.0.as_ref().unwrap().exploration_credit_us < 0);
    assert!(!policy.allow_spilling_candidate(Some(context()), &None, 3, 1));
    // Recover known work to price the previously unknown recurring reload.
    observe(&mut policy, 1, 5_000, 1_000, true);
    assert!(policy.allow_spilling_candidate(Some(context()), &None, 2, 1));
    let before = policy.0.as_ref().unwrap().exploration_credit_us;
    observe(&mut policy, 2, 6_000, 1_000, false);
    assert!(policy.0.as_ref().unwrap().exploration_credit_us > before);
    // Two known executions cover three rows, plus the conservative first
    // installation reserve. Test the cost boundary, not a repetition count.
    let reserve = 2 * 6_000 + 20_000;
    policy.0.as_mut().unwrap().exploration_credit_us = reserve - 1;
    assert!(!policy.allow_spilling_candidate(Some(context()), &None, 3, 1));
    policy.0.as_mut().unwrap().exploration_credit_us = reserve;
    assert!(policy.allow_spilling_candidate(Some(context()), &None, 3, 1));
}

#[test]
fn a_known_resident_batch_does_not_pay_an_unknown_spilling_probe_when_queue_grows() {
    let mut policy = BatchEfficiency::default();
    observe(&mut policy, 2, 12_000, 0, true);
    observe(&mut policy, 4, 23_500, 0, true);
    assert!(!policy.allow_spilling_candidate(Some(context()), &None, 5, 4));
    assert!(!policy.allow_spilling_candidate(Some(context()), &None, 16, 4));
    assert_eq!(policy.warm_start(Some(context()), &None, 8, 16), Some(4));
    assert_eq!(policy.warm_start(Some(context()), &None, 16, 16), Some(4));
}

fn context() -> ImageCapacitySearchContext {
    ImageCapacitySearchContext {
        shape: [1; 32],
        capacity_bytes: 100,
    }
}

fn observe(policy: &mut BatchEfficiency, count: usize, actual: u64, install: u64, retained: bool) {
    observe_event(policy, count, actual, install, retained, false);
}

fn observe_event(
    policy: &mut BatchEfficiency,
    count: usize,
    actual: u64,
    install: u64,
    retained: bool,
    exploration: bool,
) {
    policy.observe(
        ExecutionContext {
            context: context(),
            adapter: None,
            capacity_upper: 3,
            batch_size: count,
            exploration,
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
    policy.select(Some(context()), &None, queued.min(3), 3)
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
    observe(&mut policy, 3, 10_200, 1_700, false);
    assert_eq!(select(&mut policy, 3), 2);
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
    assert_eq!(policy.select(Some(changed), &None, 3, 3), 2);
    assert_eq!(policy.select(Some(changed), &None, 3, 4), 3);
    changed.shape = [2; 32];
    assert_eq!(policy.select(Some(changed), &None, 3, 3), 3);
    let adapter = Some(Arc::new(ImageAdapterBatchIdentity {
        effective_identity: "different".into(),
        strength_bits: 1.0_f32.to_bits(),
    }));
    assert_eq!(policy.select(Some(context()), &adapter, 3, 3), 3);
    assert_eq!(policy.select(None, &None, 3, 3), 3);
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

#[test]
fn pipeline_primes_observed_size_but_background_still_explores_unknown_larger_batch() {
    let mut policy = BatchEfficiency::default();
    observe(&mut policy, 1, 3_300, 0, true);
    observe(&mut policy, 3, 8_000, 0, true);
    assert_eq!(policy.warm_start(Some(context()), &None, 16, 16), Some(3));
    // Preparing the backlog still probes the larger available shape. A larger
    // device is not permanently constrained by the first small queue observed.
    assert_eq!(policy.select(Some(context()), &None, 16, 16), 16);
    let mut changed = context();
    changed.shape[0] += 1;
    assert_eq!(policy.warm_start(Some(changed), &None, 16, 16), None);
}

#[test]
fn ready_spilling_batch_is_priced_using_known_work_before_repeating_an_unknown_reload() {
    let mut policy = BatchEfficiency::default();
    observe(&mut policy, 1, 3_300, 0, true);
    observe(&mut policy, 2, 5_500, 0, true);
    // First B3 ran, but left the text encoder missing. Its cold-free reload
    // time is not known until another actual installation has completed.
    observe(&mut policy, 3, 8_000, 0, false);
    assert_eq!(policy.reconsider_ready(context(), &None, 3, 3), Some(2));
    observe(&mut policy, 2, 7_200, 1_700, true);
    // The background plan may already be ready when this execution prices
    // the reload. It must use that feedback too, not execute a known loser.
    assert_eq!(policy.reconsider_ready(context(), &None, 3, 3), Some(2));
    assert_eq!(policy.reconsider_ready(context(), &None, 3, 4), None);
    assert_eq!(select(&mut policy, 3), 2);
}
