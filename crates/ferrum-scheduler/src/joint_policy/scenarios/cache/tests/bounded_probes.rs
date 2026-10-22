use super::two_slot::{colliding_work, keyed_distribution};
use super::*;
use crate::joint_policy::scenarios::ConditionalSnapshot;
use std::sync::atomic::{AtomicUsize, Ordering};

fn insert_handle(cache: &mut ScenarioCache, work: CostWork, ranks: usize) -> CacheHandle {
    let work = [work];
    let query = cost_query(&work);
    let (_, key, _) = cache.get_with_handle(&query, cache.at, 0, None);
    cache
        .insert_with_handle(key, keyed_distribution(cache.at, work[0], ranks))
        .unwrap()
}

fn assert_value(cache: &mut ScenarioCache, at: Instant, work: CostWork, rank: usize) {
    let expected = Duration::from_micros(1000 + work.context_tokens as u64 + rank as u64);
    assert!(matches!(cache.get(&cost_query(&[work]), at, rank),
        Some(CostPrediction::Known {
            central_service_time, upper_service_time, observed_at, expires_at,
            risk: CostRisk::ConditionalScenario { dependence: CostScenarioDependence::SharedResidualRank },
            support: CostSupport::ValidatedFeatureDomain, samples: 32,
        }) if central_service_time == expected && upper_service_time == expected
            && observed_at == at - Duration::from_secs(1)
            && expires_at == at + Duration::from_secs(9)));
}

#[test]
fn bounded_probes_use_later_empty_slots_then_replace_only_primary_when_full() {
    let at = Instant::now();
    let slots = MAX_PROBES * 2;
    let keys = colliding_work(slots, MAX_PROBES + 1);
    let mut cache = ScenarioCache::new(at, 128 * 1024, slots);
    assert_eq!(cache.slots.len(), slots);
    assert!(cache.enable_handles());
    let fixed = cache.retained_bytes();
    let mut handles = Vec::new();
    for key in &keys[..MAX_PROBES] {
        handles.push(insert_handle(&mut cache, *key, 3));
        // In particular, storing the third collision must retain the first
        // two when a later probe slot is still free.
        for earlier in &keys[..handles.len()] {
            assert_value(&mut cache, at, *earlier, 2);
        }
    }
    let primary = index(&cost_query(&[keys[0]]), slots);
    assert_eq!(
        handles.iter().map(|h| h.slot).collect::<Vec<_>>(),
        cache.probe_slots(primary).collect::<Vec<_>>()
    );
    assert_eq!(
        cache.slots.iter().filter(|slot| slot.is_none()).count(),
        slots - MAX_PROBES
    );
    let bytes =
        fixed + MAX_PROBES * (std::mem::size_of::<CostWork>() + 3 * std::mem::size_of::<u64>());
    assert_eq!(cache.retained_bytes(), bytes);
    let replacement = insert_handle(&mut cache, keys[MAX_PROBES], 3);
    assert_eq!(replacement.slot, primary);
    assert!(cache.handle_work(handles[0], at, 0).is_none());
    assert!(cache.get(&cost_query(&[keys[0]]), at, 0).is_none());
    for (key, handle) in keys[1..MAX_PROBES].iter().zip(&handles[1..]) {
        assert_eq!(
            cache.handle_work(*handle, at, 2),
            Some(std::slice::from_ref(key))
        );
        assert_value(&mut cache, at, *key, 2);
    }
    assert_value(&mut cache, at, keys[MAX_PROBES], 2);
    assert_eq!(cache.retained_bytes(), bytes);
    assert_eq!(
        cache.slots.iter().filter(|slot| slot.is_none()).count(),
        slots - MAX_PROBES
    );
}

#[test]
fn bounded_probes_refresh_last_exact_slot_before_early_empty_and_recheck_pending_tokens() {
    let at = Instant::now();
    let slots = MAX_PROBES * 2;
    let keys = colliding_work(slots, MAX_PROBES + 1);
    let mut cache = ScenarioCache::new(at, 128 * 1024, slots);
    assert!(cache.enable_handles());
    let handles = keys[..MAX_PROBES]
        .iter()
        .map(|key| insert_handle(&mut cache, *key, 3))
        .collect::<Vec<_>>();
    let last_work = [keys[MAX_PROBES - 1]];
    let query = cost_query(&last_work);
    let (_, pending, _) = cache.get_with_handle(&query, at, 3, None);
    assert!(pending.is_some());
    // A legal sparse layout: a former primary payload has disappeared while
    // the requested exact key remains at the eighth probe. This is an explicit
    // cache-state boundary, not a claim about how often production reaches it.
    let primary = handles[0].slot;
    let removed = cache.slots[primary].take().unwrap();
    cache.retained_bytes -= removed.payload_bytes();
    let fresh = cache
        .insert_with_handle(pending, keyed_distribution(at, last_work[0], 4))
        .unwrap();
    assert_eq!(fresh.slot, handles[MAX_PROBES - 1].slot);
    assert!(cache.slots[primary].is_none());
    assert!(cache.handle_work(handles[MAX_PROBES - 1], at, 0).is_none());
    assert_value(&mut cache, at, last_work[0], 3);
    assert_eq!(
        cache
            .slots
            .iter()
            .flatten()
            .filter(|entry| entry.matches(&query))
            .count(),
        1
    );

    // Two provider calls may hold tokens for the same absent query. The second
    // completion must reselect the CURRENT exact entry, not use a new empty slot.
    let new_work = [keys[MAX_PROBES]];
    let new_query = cost_query(&new_work);
    let (_, first) = cache.get_with_key(&new_query, at, 0);
    let (_, second) = cache.get_with_key(&new_query, at, 0);
    let first_handle = cache
        .insert_with_handle(first, keyed_distribution(at, new_work[0], 2))
        .unwrap();
    let second_handle = cache
        .insert_with_handle(second, keyed_distribution(at, new_work[0], 3))
        .unwrap();
    assert_eq!(first_handle.slot, primary);
    assert_eq!(second_handle.slot, primary);
    assert_ne!(first_handle.generation, second_handle.generation);
    assert!(cache.handle_work(first_handle, at, 0).is_none());
    assert_eq!(
        cache
            .slots
            .iter()
            .flatten()
            .filter(|entry| entry.matches(&new_query))
            .count(),
        1
    );
    assert_value(&mut cache, at, new_work[0], 2);
}

#[test]
fn bounded_probes_refused_payload_preserves_every_full_probe_handle_and_byte_ledger() {
    let at = Instant::now();
    let slots = MAX_PROBES;
    let keys = colliding_work(slots, MAX_PROBES + 1);
    let ranks = std::mem::size_of::<CostScenario>()
        .saturating_sub(std::mem::size_of::<Option<NonZeroUsize>>() + std::mem::size_of::<u64>())
        .div_ceil(std::mem::size_of::<u64>())
        .max(2);
    let payload = std::mem::size_of::<CostWork>() + ranks * std::mem::size_of::<u64>();
    let budget = slots
        * (std::mem::size_of::<Option<Entry>>()
            + std::mem::size_of::<Option<NonZeroUsize>>()
            + std::mem::size_of::<u64>()
            + payload);
    let mut cache = ScenarioCache::new(at, budget, slots);
    assert_eq!(cache.slots.len(), slots);
    assert!(cache.enable_handles());
    let handles = keys[..MAX_PROBES]
        .iter()
        .map(|key| insert_handle(&mut cache, *key, ranks))
        .collect::<Vec<_>>();
    assert_eq!(cache.retained_bytes(), budget);
    let generation = cache.last_generation;
    let work = [keys[MAX_PROBES]];
    let (_, pending, _) = cache.get_with_handle(&cost_query(&work), at, 0, None);
    assert!(cache
        .insert_with_handle(pending, keyed_distribution(at, work[0], ranks + 1))
        .is_none());
    assert_eq!(cache.last_generation, generation);
    assert_eq!(cache.retained_bytes(), budget);
    for (key, handle) in keys[..MAX_PROBES].iter().zip(handles) {
        assert_eq!(
            cache.handle_work(handle, at, ranks - 1),
            Some(std::slice::from_ref(key))
        );
        assert_value(&mut cache, at, *key, ranks - 1);
    }
    assert!(cache.get(&cost_query(&work), at, 0).is_none());
}

#[test]
fn bounded_probes_wrap_deterministically_without_duplicates_for_short_tables() {
    let at = Instant::now();
    // Boundary sizes exercise one slot, short tables, exactly the bound, and
    // larger tables. Every possible primary exercises both sides of wrap.
    for slots in [
        1,
        2,
        3,
        MAX_PROBES - 1,
        MAX_PROBES,
        MAX_PROBES + 1,
        MAX_PROBES * 2 + 1,
    ] {
        let cache = ScenarioCache::new(at, 128 * 1024, slots);
        assert_eq!(cache.slots.len(), slots);
        for primary in 0..slots {
            let probes = cache.probe_slots(primary).collect::<Vec<_>>();
            let expected = (primary..slots)
                .chain(0..primary)
                .take(slots.min(MAX_PROBES))
                .collect::<Vec<_>>();
            assert_eq!(probes, expected);
            assert_eq!(cache.probe_slots(primary).collect::<Vec<_>>(), probes);
            assert_eq!(
                probes
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                probes.len()
            );
        }
    }
    let mut disabled = ScenarioCache::new(at, 0, 8);
    let query_work = [work()];
    assert!(disabled.get(&cost_query(&query_work), at, 0).is_none());
    disabled.insert(&cost_query(&query_work), distribution(at, 2));
    assert_eq!(disabled.retained_bytes(), 0);
}

struct BoundedCosts {
    at: Instant,
    calls: AtomicUsize,
}

impl CostSnapshot for BoundedCosts {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        Default::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("conditional consumer must request complete distributions")
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
fn bounded_probes_real_conditional_consumer_reuses_third_collision_at_exact_provider_quota() {
    let at = Instant::now();
    let keys = colliding_work(3, 4);
    let original = BoundedCosts {
        at,
        calls: AtomicUsize::new(0),
    };
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_queries = 3;
    config.planning.max_scenarios = 2;
    let cache = parking_lot::Mutex::new(ScenarioCache::new(at, 8192, 3));
    let queries = AtomicUsize::new(0);
    let valid_until = parking_lot::Mutex::new(at + Duration::from_secs(60));
    let diagnostics = crate::joint_policy::diagnostics::Recorder::new(3);
    let snapshot = |rank| ConditionalSnapshot {
        original: &original,
        rank,
        weights: &[0.5, 0.5],
        dependence: CostScenarioDependence::SharedResidualRank,
        query_limits: CostQueryLimits::default(),
        config: &config,
        started: at,
        queries: &queries,
        cache: &cache,
        valid_until: &valid_until,
        diagnostics: &diagnostics,
    };
    for key in &keys[..3] {
        assert!(matches!(
            snapshot(0).predict(&cost_query(&[*key]), at),
            CostPrediction::Known { .. }
        ));
    }
    assert_eq!(queries.load(Ordering::Relaxed), 3);
    for key in &keys[..3] {
        assert!(matches!(snapshot(1).predict(&cost_query(&[*key]), at),
            CostPrediction::Known { central_service_time, upper_service_time, expires_at, .. }
            if central_service_time == Duration::from_micros(1001 + key.context_tokens as u64)
                && upper_service_time == central_service_time && expires_at == at + Duration::from_secs(9)));
    }
    assert_eq!(original.calls.load(Ordering::Relaxed), 3);
    assert_eq!(queries.load(Ordering::Relaxed), 3);
    assert!(!diagnostics.finish(None).query_budget_exhausted);
    assert_eq!(*valid_until.lock(), at + Duration::from_secs(9));
    assert_eq!(
        snapshot(0).predict(&cost_query(&[keys[3]]), at),
        CostPrediction::Unknown(CostUnknownReason::PlanningBudget)
    );
    assert!(diagnostics.finish(None).query_budget_exhausted);
    assert_eq!(original.calls.load(Ordering::Relaxed), 3);
}
