use super::two_slot::{colliding_work, keyed_distribution};
use super::*;
use crate::joint_policy::scenarios::ConditionalSnapshot;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Costs {
    at: Instant,
    calls: AtomicUsize,
    unknown: Option<CostWork>,
}

impl CostSnapshot for Costs {
    fn revision(&self) -> u64 {
        1
    }

    fn query_limits(&self) -> CostQueryLimits {
        Default::default()
    }

    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("conditional consumer must request complete scenarios")
    }

    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.unknown == Some(query.work[0]) {
            CostScenarioPrediction::Unknown(CostUnknownReason::NoCoverage)
        } else {
            keyed_distribution(self.at, query.work[0], count)
        }
    }
}

fn expected(at: Instant, work: CostWork, rank: usize) -> CostPrediction {
    let time = Duration::from_micros(1000 + work.context_tokens as u64 + rank as u64);
    CostPrediction::Known {
        central_service_time: time,
        upper_service_time: time,
        risk: CostRisk::ConditionalScenario {
            dependence: CostScenarioDependence::SharedResidualRank,
        },
        support: CostSupport::ValidatedFeatureDomain,
        samples: 32,
        observed_at: at - Duration::from_secs(1),
        expires_at: at + Duration::from_secs(9),
    }
}

#[test]
fn ordinal_hints_actual_consumer_skips_hashing_but_keeps_rank_metadata_quota_and_time_gate() {
    let at = Instant::now();
    let keys = colliding_work(2, 3);
    let original = Costs {
        at,
        calls: AtomicUsize::new(0),
        unknown: None,
    };
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_queries = 2;
    config.planning.max_scenarios = 2;
    let cache = parking_lot::Mutex::new(ScenarioCache::new(at, 8192, 2));
    let queries = AtomicUsize::new(0);
    let valid_until = parking_lot::Mutex::new(at + Duration::from_secs(60));
    let diagnostics = crate::joint_policy::diagnostics::Recorder::new(2);
    let snapshot = |rank, started| ConditionalSnapshot {
        original: &original,
        rank,
        weights: &[0.5, 0.5],
        dependence: CostScenarioDependence::SharedResidualRank,
        query_limits: Default::default(),
        config: &config,
        started,
        queries: &queries,
        cache: &cache,
        valid_until: &valid_until,
        diagnostics: &diagnostics,
    };
    for rank in 0..2 {
        cache.lock().begin_rank();
        for key in &keys[..2] {
            assert_eq!(
                snapshot(rank, at).predict(&cost_query(&[*key]), at),
                expected(at, *key, rank)
            );
        }
    }
    assert_eq!(original.calls.load(Ordering::Relaxed), 2);
    assert_eq!(queries.load(Ordering::Relaxed), 2);
    {
        let cache = cache.lock();
        assert_eq!((cache.hint_hits, cache.hash_lookups), (2, 2));
    }
    assert_eq!(*valid_until.lock(), at + Duration::from_secs(9));
    let report = diagnostics.finish(None);
    assert_eq!((report.provider_calls, report.cache_hits), (2, 2));
    assert!(!report.query_budget_exhausted);

    cache.lock().begin_rank();
    assert_eq!(
        snapshot(1, at - Duration::from_secs(2)).predict(&cost_query(&[keys[0]]), at),
        CostPrediction::Unknown(CostUnknownReason::PlanningBudget)
    );
    assert_eq!(cache.lock().hint_hits, 2);
    assert_eq!(
        diagnostics.finish(None).first_failure.unwrap().origin,
        crate::joint_policy::FailureOrigin::PreQueryTime
    );
    // A wrong ordinal hint cannot turn a genuinely new query into a hit.
    assert_eq!(
        snapshot(1, at).predict(&cost_query(&[keys[2]]), at),
        CostPrediction::Unknown(CostUnknownReason::PlanningBudget)
    );
    assert!(diagnostics.finish(None).query_budget_exhausted);
    assert_eq!(original.calls.load(Ordering::Relaxed), 2);
}

#[test]
fn ordinal_hints_changed_query_order_and_extra_work_use_exact_fallback() {
    let at = Instant::now();
    let keys = colliding_work(4, 3);
    let mut cache = ScenarioCache::new(at, 16 * 1024, 4);
    for key in &keys[..2] {
        cache.insert(&cost_query(&[*key]), keyed_distribution(at, *key, 2));
    }
    assert!(
        cache.hints.iter().all(Option::is_none),
        "seed has no ordinal"
    );
    cache.begin_rank();
    for key in &keys[..2] {
        assert_eq!(
            cache.get(&cost_query(&[*key]), at, 0),
            Some(expected(at, *key, 0))
        );
    }
    cache.begin_rank();
    for key in keys[..2].iter().rev() {
        assert_eq!(
            cache.get(&cost_query(&[*key]), at, 1),
            Some(expected(at, *key, 1))
        );
    }
    assert_eq!((cache.hint_hits, cache.hash_lookups), (0, 4));
    let extra = [keys[2]];
    let (hit, pending) = cache.get_with_key(&cost_query(&extra), at, 1);
    assert!(hit.is_none());
    cache.insert_with_key(pending, keyed_distribution(at, keys[2], 2));
    // Same ordinal sequence is not required even after the new work is stored.
    cache.begin_rank();
    assert_eq!(
        cache.get(&cost_query(&extra), at, 1),
        Some(expected(at, keys[2], 1))
    );
    assert_eq!(
        cache.get(&cost_query(&[keys[1]]), at, 1),
        Some(expected(at, keys[1], 1))
    );
}

#[test]
fn ordinal_hints_evicted_slot_and_absent_rank_never_alias_another_distribution() {
    let at = Instant::now();
    let keys = colliding_work(2, 3);
    let mut cache = ScenarioCache::new(at, 8192, 2);
    for key in &keys[..2] {
        cache.insert(&cost_query(&[*key]), keyed_distribution(at, *key, 2));
    }
    cache.begin_rank();
    assert!(cache.get(&cost_query(&[keys[0]]), at, 0).is_some());
    assert!(cache.get(&cost_query(&[keys[1]]), at, 0).is_some());
    cache.insert(&cost_query(&[keys[2]]), keyed_distribution(at, keys[2], 3));
    cache.begin_rank();
    assert!(cache.get(&cost_query(&[keys[0]]), at, 1).is_none());
    assert_eq!(
        cache.get(&cost_query(&[keys[1]]), at, 1),
        Some(expected(at, keys[1], 1))
    );
    assert_eq!(cache.hint_hits, 1, "secondary survives primary eviction");
    cache.begin_rank();
    assert_eq!(
        cache.get(&cost_query(&[keys[2]]), at, 2),
        Some(expected(at, keys[2], 2))
    );
    let before = cache.hits();
    cache.begin_rank();
    assert!(cache.get(&cost_query(&[keys[2]]), at, 3).is_none());
    assert_eq!(cache.hits(), before);
    cache.begin_rank();
    assert!(cache
        .get(&cost_query(&[keys[2]]), at + Duration::from_nanos(1), 0)
        .is_none());
    assert_eq!(cache.hits(), before);
}

#[test]
fn ordinal_hints_interleaved_misses_remember_original_ordinal_and_actual_secondary() {
    let at = Instant::now();
    let keys = colliding_work(4, 2);
    let a = [keys[0]];
    let b = [keys[1]];
    let mut cache = ScenarioCache::new(at, 16 * 1024, 4);
    let fixed_bytes = cache.retained_bytes();
    cache.begin_rank();
    let (_, a_pending) = cache.get_with_key(&cost_query(&a), at, 0);
    let (_, b_pending) = cache.get_with_key(&cost_query(&b), at, 0);
    // Reverse completion order: B fills the primary, A must use its secondary.
    cache.insert_with_key(b_pending, keyed_distribution(at, keys[1], 3));
    cache.insert_with_key(a_pending, keyed_distribution(at, keys[0], 2));
    assert_eq!(
        cache.retained_bytes(),
        fixed_bytes + 2 * std::mem::size_of::<CostWork>() + 5 * std::mem::size_of::<Duration>()
    );
    cache.begin_rank();
    assert_eq!(
        cache.get(&cost_query(&a), at, 1),
        Some(expected(at, keys[0], 1))
    );
    assert_eq!(
        cache.get(&cost_query(&b), at, 2),
        Some(expected(at, keys[1], 2))
    );
    assert_eq!((cache.hint_hits, cache.hash_lookups), (2, 2));

    cache.begin_rank();
    let (_, pending) = cache.get_with_key(&cost_query(&a), at, 2);
    cache.insert(&cost_query(&a), keyed_distribution(at, keys[0], 4));
    cache.insert_with_key(pending, keyed_distribution(at, keys[0], 3));
    assert_eq!(
        cache.retained_bytes(),
        fixed_bytes + 2 * std::mem::size_of::<CostWork>() + 6 * std::mem::size_of::<Duration>()
    );
    assert_eq!(
        cache
            .slots
            .iter()
            .flatten()
            .filter(|entry| entry.matches(&cost_query(&a)))
            .count(),
        1
    );
    cache.begin_rank();
    assert_eq!(
        cache.get(&cost_query(&a), at, 2),
        Some(expected(at, keys[0], 2))
    );
}

#[test]
fn ordinal_hints_are_bounded_charged_and_do_not_wrap_or_evict_on_capacity_failure() {
    let at = Instant::now();
    let keys = colliding_work(2, 3);
    let fixed_bytes =
        2 * (std::mem::size_of::<Option<Entry>>() + std::mem::size_of::<Option<NonZeroUsize>>());
    let budget =
        fixed_bytes + 2 * (std::mem::size_of::<CostWork>() + 2 * std::mem::size_of::<Duration>());
    let mut cache = ScenarioCache::new(at, budget, 2);
    assert_eq!(cache.retained_bytes(), fixed_bytes);
    assert_eq!(cache.slots.len(), 2);
    assert_eq!(cache.hints.len(), 2);
    for key in &keys[..2] {
        cache.insert(&cost_query(&[*key]), keyed_distribution(at, *key, 2));
    }
    assert_eq!(cache.retained_bytes(), budget);
    cache.begin_rank();
    for key in &keys[..2] {
        assert!(cache.get(&cost_query(&[*key]), at, 0).is_some());
    }
    let original_hints = cache.hints.clone();
    assert!(cache.get(&cost_query(&[keys[0]]), at, 0).is_some());
    assert_eq!(
        cache.hints, original_hints,
        "no modulo reuse after capacity"
    );
    cache.next_hint = usize::MAX;
    assert!(cache.get(&cost_query(&[keys[0]]), at, 0).is_some());
    assert_eq!(cache.next_hint, usize::MAX);
    assert_eq!(cache.hint_hits, 0);
    cache.begin_rank();
    let third = [keys[2]];
    let (_, pending) = cache.get_with_key(&cost_query(&third), at, 0);
    cache.insert_with_key(pending, keyed_distribution(at, keys[2], 3));
    assert_eq!(cache.retained_bytes(), budget);
    assert_eq!(cache.hints, original_hints);
    for key in &keys[..2] {
        cache.begin_rank();
        assert_eq!(
            cache.get(&cost_query(&[*key]), at, 1),
            Some(expected(at, *key, 1))
        );
    }
    for (bytes, count) in [(0, 2), (1, 2), (8192, 0)] {
        let mut disabled = ScenarioCache::new(at, bytes, count);
        disabled.begin_rank();
        assert!(disabled.get(&cost_query(&third), at, 0).is_none());
        assert_eq!(disabled.retained_bytes(), 0);
        assert!(disabled.hints.is_empty());
    }
    let mut one = ScenarioCache::new(at, 8192, 1);
    one.insert(&cost_query(&third), keyed_distribution(at, keys[2], 2));
    assert!(one.get(&cost_query(&third), at, 0).is_some());
    one.begin_rank();
    assert!(one.get(&cost_query(&third), at, 1).is_some());
    assert_eq!((one.hits(), one.hint_hits, one.hash_lookups), (2, 1, 1));
}

#[test]
fn ordinal_hints_unknown_at_changed_ordinal_still_calls_provider_and_is_not_cached() {
    let at = Instant::now();
    let keys = colliding_work(4, 2);
    let original = Costs {
        at,
        calls: AtomicUsize::new(0),
        unknown: Some(keys[1]),
    };
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_queries = 4;
    config.planning.max_scenarios = 2;
    let cache = parking_lot::Mutex::new(ScenarioCache::new(at, 8192, 4));
    let queries = AtomicUsize::new(0);
    let valid_until = parking_lot::Mutex::new(at + Duration::from_secs(60));
    let diagnostics = crate::joint_policy::diagnostics::Recorder::new(4);
    let snapshot = ConditionalSnapshot {
        original: &original,
        rank: 0,
        weights: &[0.5, 0.5],
        dependence: CostScenarioDependence::SharedResidualRank,
        query_limits: Default::default(),
        config: &config,
        started: at,
        queries: &queries,
        cache: &cache,
        valid_until: &valid_until,
        diagnostics: &diagnostics,
    };
    cache.lock().begin_rank();
    assert_eq!(
        snapshot.predict(&cost_query(&[keys[0]]), at),
        expected(at, keys[0], 0)
    );
    for _ in 0..2 {
        cache.lock().begin_rank();
        assert_eq!(
            snapshot.predict(&cost_query(&[keys[1]]), at),
            CostPrediction::Unknown(CostUnknownReason::NoCoverage)
        );
    }
    assert_eq!(original.calls.load(Ordering::Relaxed), 3);
    assert_eq!(queries.load(Ordering::Relaxed), 3);
    assert_eq!(cache.lock().hint_hits, 0);
    cache.lock().begin_rank();
    assert_eq!(
        snapshot.predict(&cost_query(&[keys[0]]), at),
        expected(at, keys[0], 0)
    );
    assert_eq!(cache.lock().hint_hits, 1);
    assert_eq!(diagnostics.finish(None).cache_hits, 1);
}
