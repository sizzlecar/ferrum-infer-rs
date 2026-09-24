use super::*;

fn p(n: u64) -> Option<NonZeroU64> {
    Some(NonZeroU64::new(n).unwrap())
}

#[test]
fn per_request_limit_is_not_an_aggregate_or_time_limit() {
    let policy = PlanningWorkPolicy {
        maximum_wave_tokens: 2048,
        active_decode_prefill_chunk: Some(128),
        ..Default::default()
    };
    let envelope = policy.for_ready_decoders(1);
    let mut used = WaveWorkUsage::default();
    assert!(envelope.include(&mut used, None));
    for n in [66, 128, 19, 28, 34, 53, 19] {
        assert!(envelope.include(&mut used, p(n)));
    }
    assert_eq!(used.prefill_tokens, 347);
    assert_eq!(used.tokens, 348);
    let before = used;
    assert!(!envelope.include(&mut used, p(129)));
    assert_eq!(used, before);
}

#[test]
fn aggregate_limit_is_independent_of_owner_count_and_split_membership() {
    let policy = PlanningWorkPolicy {
        maximum_wave_tokens: 2048,
        prefill_step_chunk: Some(64),
        active_decode_prefill_chunk: Some(128),
        active_decode_prefill_token_budget: Some(100),
        allow_mixed: false,
        ..Default::default()
    };
    let active = policy.for_ready_decoders(1);
    let mut used = WaveWorkUsage::default();
    assert!(active.include(&mut used, p(64)));
    assert!(active.include(&mut used, p(36)));
    assert!(!active.include(&mut used, p(1)));
    assert!(!active.include(&mut used, None));
    let mut reverse = WaveWorkUsage::default();
    assert!(active.include(&mut reverse, None));
    assert!(!active.include(&mut reverse, p(1)));
    let pure = policy.for_ready_decoders(0);
    let mut used = WaveWorkUsage::default();
    assert!(pure.include(&mut used, p(64)));
    assert!(pure.include(&mut used, p(64)));
    assert_eq!(
        used.prefill_tokens, 128,
        "pure-prefill has no active aggregate cap"
    );
}

#[test]
fn disabled_aggregate_and_successor_activation_preserve_step_limit() {
    let policy = PlanningWorkPolicy {
        prefill_step_chunk: Some(100),
        active_decode_prefill_chunk: Some(20),
        active_decode_prefill_token_budget: Some(0),
        ..Default::default()
    };
    assert_eq!(
        policy.for_ready_decoders(0).maximum_prefill_chunk,
        Some(100)
    );
    assert_eq!(policy.for_ready_decoders(1).maximum_prefill_chunk, Some(20));
    assert_eq!(policy.for_ready_decoders(1).maximum_prefill_tokens, None);
    assert_eq!(policy.for_ready_decoders(0).maximum_prefill_tokens, None);
}

#[test]
fn resolved_scheduler_defaults_and_pressure_are_shared() {
    let config = SchedulerConfig {
        prefill_step_chunk: Some(12),
        active_decode_prefill_token_budget: Some(7),
        ..Default::default()
    };
    let scheduler = ContinuousBatchScheduler::new(config);
    let mut hint = BatchHint::simple(32);
    hint.max_tokens = 99;
    let policy = scheduler.planning_work_policy(&hint, true, false);
    assert_eq!(policy.maximum_wave_tokens, 99);
    assert_eq!(policy.for_ready_decoders(1).maximum_prefill_chunk, Some(7));
    assert_eq!(policy.for_ready_decoders(0).maximum_prefill_chunk, Some(12));
    let pressured = scheduler.planning_work_policy(&hint, true, true);
    assert_eq!(
        pressured.for_ready_decoders(1).maximum_prefill_chunk,
        Some(12.min(scheduler.default_active_decode_prefill_chunk() as u64))
    );
    assert_eq!(
        pressured.for_ready_decoders(1).maximum_prefill_tokens,
        Some(7)
    );
}

#[test]
fn total_token_limit_and_arithmetic_reject_without_partial_charging() {
    let envelope = PlanningWorkPolicy {
        maximum_wave_tokens: 5,
        ..Default::default()
    }
    .for_ready_decoders(1);
    let mut used = WaveWorkUsage::default();
    assert!(envelope.include(&mut used, p(4)));
    assert!(envelope.include(&mut used, None));
    let before = used;
    assert!(!envelope.include(&mut used, None));
    assert_eq!(used, before);
    let unlimited = PlanningWorkPolicy::default().for_ready_decoders(0);
    let mut maximum = WaveWorkUsage::default();
    assert!(unlimited.include(&mut maximum, p(u64::MAX)));
    let before = maximum;
    assert!(!unlimited.include(&mut maximum, None));
    assert_eq!(maximum, before);
}
