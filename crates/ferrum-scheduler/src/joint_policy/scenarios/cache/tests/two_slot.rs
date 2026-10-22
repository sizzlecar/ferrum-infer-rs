use super::*;
use crate::joint_policy::scenarios::ConditionalSnapshot;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Pigeonhole construction over legal decode work, independent of a particular
/// hash value or host word width. No distribution-quality threshold is assumed.
pub(super) fn colliding_work(slots: usize, count: usize) -> Vec<CostWork> {
    let mut buckets = vec![Vec::new(); slots];
    for context in 2..=slots * (count - 1) + 2 {
        let work = CostWork {
            requested: ExecutionRange::new(context - 1, 1).unwrap(),
            context_tokens: context,
            ..work()
        };
        let slot = index(&cost_query(&[work]), slots);
        buckets[slot].push(work);
        if buckets[slot].len() == count {
            return std::mem::take(&mut buckets[slot]);
        }
    }
    unreachable!("more lawful distinct keys than the bounded buckets can hold")
}

pub(super) fn keyed_distribution(
    at: Instant,
    work: CostWork,
    count: usize,
) -> CostScenarioPrediction {
    let mut prediction = distribution(at, count);
    if let CostScenarioPrediction::Known {
        central_service_time,
        scenarios,
        ..
    } = &mut prediction
    {
        let base = 1000 + work.context_tokens as u64;
        *central_service_time = Duration::from_micros(base);
        for scenario in scenarios {
            scenario.service_time = Duration::from_micros(base + scenario.rank as u64);
        }
    }
    prediction
}

#[test]
fn two_slot_collisions_keep_distinct_full_rank_distributions_without_growing_storage() {
    let at = Instant::now();
    // Explicit two-slot table: exercise exhaustion independently of the
    // maximum probe count used by larger tables.
    let keys = colliding_work(2, 3);
    let mut cache = ScenarioCache::new(at, 16 * 1024, 2);
    let slot_bytes = cache.retained_bytes();
    for work in &keys[..2] {
        cache.insert(&cost_query(&[*work]), keyed_distribution(at, *work, 3));
    }
    assert_eq!(cache.slots.len(), 2);
    assert_eq!(
        cache.retained_bytes(),
        slot_bytes + 2 * (std::mem::size_of::<CostWork>() + 3 * std::mem::size_of::<u64>())
    );
    for rank in 0..3 {
        for work in &keys[..2] {
            let prediction = cache.get(&cost_query(&[*work]), at, rank).unwrap();
            assert!(matches!(prediction, CostPrediction::Known {
                central_service_time,
                upper_service_time,
                observed_at,
                expires_at,
                samples: 32,
                support: CostSupport::ValidatedFeatureDomain,
                risk: CostRisk::ConditionalScenario {
                    dependence: CostScenarioDependence::SharedResidualRank
                },
            } if central_service_time == Duration::from_micros(1000 + work.context_tokens as u64 + rank as u64)
                && upper_service_time == central_service_time
                && observed_at == at - Duration::from_secs(1)
                && expires_at == at + Duration::from_secs(9)));
        }
    }
    let bytes = cache.retained_bytes();
    cache.insert(&cost_query(&[keys[2]]), keyed_distribution(at, keys[2], 3));
    assert!(cache.get(&cost_query(&[keys[0]]), at, 0).is_none());
    assert!(cache.get(&cost_query(&[keys[1]]), at, 2).is_some());
    assert!(cache.get(&cost_query(&[keys[2]]), at, 2).is_some());
    assert!(cache.get(&cost_query(&[keys[2]]), at, 3).is_none());
    assert_eq!(cache.retained_bytes(), bytes);
}

#[test]
fn two_slot_miss_token_rechecks_current_exact_key_before_empties_and_debits_actual_victim() {
    let at = Instant::now();
    let keys = colliding_work(4, 2);
    let query_work = [keys[1]];
    let query = cost_query(&query_work);
    let mut cache = ScenarioCache::new(at, 16 * 1024, 4);
    let slot_bytes = cache.retained_bytes();
    let (_, pending) = cache.get_with_key(&query, at, 0);
    // While this provider is outside the lock, another call fills the primary
    // and then stores this same key in its second slot.
    cache.insert(&cost_query(&[keys[0]]), keyed_distribution(at, keys[0], 2));
    cache.insert(&query, keyed_distribution(at, keys[1], 4));
    cache.insert_with_key(pending, keyed_distribution(at, keys[1], 3));
    assert_eq!(
        cache
            .slots
            .iter()
            .flatten()
            .filter(|entry| entry.matches(&query))
            .count(),
        1
    );
    assert_eq!(
        cache.retained_bytes(),
        slot_bytes + 2 * std::mem::size_of::<CostWork>() + 5 * std::mem::size_of::<u64>()
    );
    assert!(cache.get(&cost_query(&[keys[0]]), at, 1).is_some());
    assert!(cache.get(&query, at, 2).is_some());
    assert!(cache.get(&query, at, 3).is_none());

    // Explicit valid sparse layout: an empty primary must not hide an exact
    // secondary or cause duplicate insertion. This asserts a cache invariant,
    // without claiming a particular concurrent execution creates this layout.
    let primary = index(&query, cache.slots.len());
    let removed = cache.slots[primary].take().unwrap();
    cache.retained_bytes -= removed.payload_bytes();
    let (_, pending) = cache.get_with_key(&query, at, 3);
    cache.insert_with_key(pending, keyed_distribution(at, keys[1], 4));
    assert!(cache.slots[primary].is_none());
    assert_eq!(
        cache
            .slots
            .iter()
            .flatten()
            .filter(|entry| entry.matches(&query))
            .count(),
        1
    );
    assert_eq!(
        cache.retained_bytes(),
        slot_bytes + std::mem::size_of::<CostWork>() + 4 * std::mem::size_of::<u64>()
    );
}

#[test]
fn two_slot_capacity_failure_preserves_both_entries_and_one_slot_never_duplicates_a_probe() {
    let at = Instant::now();
    let keys = colliding_work(2, 3);
    // Enough compact ranks retain the unchanged conservative slot minimum;
    // the next rank exceeds this exact retained-payload budget.
    let ranks = std::mem::size_of::<CostScenario>()
        .saturating_sub(std::mem::size_of::<Option<NonZeroUsize>>())
        .div_ceil(std::mem::size_of::<u64>())
        .max(2);
    let budget = 2 * std::mem::size_of::<Option<Entry>>()
        + 2 * std::mem::size_of::<Option<NonZeroUsize>>()
        + 2 * std::mem::size_of::<CostWork>()
        + 2 * ranks * std::mem::size_of::<u64>();
    let mut cache = ScenarioCache::new(at, budget, 2);
    assert_eq!(cache.slots.len(), 2);
    for work in &keys[..2] {
        cache.insert(&cost_query(&[*work]), keyed_distribution(at, *work, ranks));
    }
    assert_eq!(cache.retained_bytes(), budget);
    let third_work = [keys[2]];
    let (_, pending) = cache.get_with_key(&cost_query(&third_work), at, 0);
    cache.insert_with_key(pending, keyed_distribution(at, keys[2], ranks + 1));
    assert_eq!(cache.retained_bytes(), budget);
    for work in &keys[..2] {
        assert!(cache.get(&cost_query(&[*work]), at, 1).is_some());
    }
    assert!(cache.get(&cost_query(&[keys[2]]), at, 0).is_none());
    let mut one = ScenarioCache::new(at, 8192, 1);
    assert_eq!(one.probe_slots(0).collect::<Vec<_>>(), vec![0]);
    one.insert(&cost_query(&[keys[0]]), keyed_distribution(at, keys[0], 2));
    assert!(one.get(&cost_query(&[keys[0]]), at, 1).is_some());
    assert_eq!(one.hits(), 1);
}

struct CountedCosts {
    at: Instant,
    calls: AtomicUsize,
}

impl CostSnapshot for CountedCosts {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        Default::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("the conditional consumer must use full scenarios")
    }
    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        self.calls.fetch_add(1, Ordering::Relaxed);
        keyed_distribution(self.at, query.work[0], count)
    }
}

#[test]
fn two_slot_actual_consumer_reuses_collisions_at_exact_provider_quota_but_never_bypasses_a_gate() {
    let at = Instant::now();
    let keys = colliding_work(2, 3);
    let original = CountedCosts {
        at,
        calls: AtomicUsize::new(0),
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
        query_limits: CostQueryLimits::default(),
        config: &config,
        started,
        queries: &queries,
        cache: &cache,
        valid_until: &valid_until,
        diagnostics: &diagnostics,
    };
    for rank in 0..2 {
        for work in &keys[..2] {
            assert!(matches!(
                snapshot(rank, at).predict(&cost_query(&[*work]), at),
                CostPrediction::Known { .. }
            ));
        }
    }
    assert_eq!(original.calls.load(Ordering::Relaxed), 2);
    assert_eq!(queries.load(Ordering::Relaxed), 2);
    let report = diagnostics.finish(None);
    assert_eq!((report.provider_calls, report.cache_hits), (2, 2));
    assert!(!report.query_budget_exhausted);
    assert_eq!(*valid_until.lock(), at + Duration::from_secs(9));
    // Even a true hit does not bypass the unchanged pre-query elapsed gate.
    assert_eq!(
        snapshot(0, at - Duration::from_secs(2)).predict(&cost_query(&[keys[0]]), at),
        CostPrediction::Unknown(CostUnknownReason::PlanningBudget)
    );
    assert_eq!(
        diagnostics.finish(None).first_failure.unwrap().origin,
        crate::joint_policy::FailureOrigin::PreQueryTime
    );
    // A genuinely new key still needs a provider call, and the quota is spent.
    assert_eq!(
        snapshot(0, at).predict(&cost_query(&[keys[2]]), at),
        CostPrediction::Unknown(CostUnknownReason::PlanningBudget)
    );
    assert!(diagnostics.finish(None).query_budget_exhausted);
    assert_eq!(original.calls.load(Ordering::Relaxed), 2);
    assert_eq!(queries.load(Ordering::Relaxed), 2);
}
