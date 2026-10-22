use super::*;
use crate::joint_policy::scenarios::ConditionalSnapshot;
use std::sync::atomic::{AtomicUsize, Ordering};

fn complete_distribution(at: Instant, count: usize) -> CostScenarioPrediction {
    let mut result = distribution(at, count);
    if let CostScenarioPrediction::Known { scenarios, .. } = &mut result {
        let denominator = (count * (count + 1) / 2) as f64;
        for scenario in scenarios {
            // Unequal weights and order-sensitive descending distinct values:
            // index preservation, not sorting or a single quantile, is required.
            scenario.weight = (scenario.rank + 1) as f64 / denominator;
            scenario.service_time = Duration::from_nanos(
                1000 + ((count - scenario.rank) * 17 + scenario.rank % 3) as u64,
            );
        }
    }
    result
}

struct CompleteCosts {
    at: Instant,
    calls: AtomicUsize,
}

impl CostSnapshot for CompleteCosts {
    fn revision(&self) -> u64 {
        1
    }

    fn query_limits(&self) -> CostQueryLimits {
        Default::default()
    }

    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("the conditional consumer must query complete scenarios")
    }

    fn cost_scenarios(
        &self,
        _: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        self.calls.fetch_add(1, Ordering::Relaxed);
        complete_distribution(self.at, count)
    }
}

#[test]
fn compact_payload_keeps_all_rank_values_weights_metadata_and_actual_query_quota() {
    let at = Instant::now();
    let count = 32;
    let expected = complete_distribution(at, count).validated(at, count);
    let CostScenarioPrediction::Known { scenarios, .. } = &expected else {
        panic!("fixture is a complete lawful distribution")
    };
    let weights = scenarios.iter().map(|s| s.weight).collect::<Vec<_>>();
    let original = CompleteCosts {
        at,
        calls: AtomicUsize::new(0),
    };
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_queries = 1;
    config.planning.max_scenarios = count;
    let cache = parking_lot::Mutex::new(ScenarioCache::new(at, 8192, 1));
    let queries = AtomicUsize::new(0);
    let valid_until = parking_lot::Mutex::new(at + Duration::from_secs(60));
    let diagnostics = crate::joint_policy::diagnostics::Recorder::new(1);
    let snapshot = |rank| ConditionalSnapshot {
        original: &original,
        rank,
        weights: &weights,
        dependence: CostScenarioDependence::SharedResidualRank,
        query_limits: CostQueryLimits::default(),
        config: &config,
        started: at,
        queries: &queries,
        cache: &cache,
        valid_until: &valid_until,
        diagnostics: &diagnostics,
    };
    let work = [work()];
    let query = cost_query(&work);
    for (rank, scenario) in scenarios.iter().enumerate() {
        assert_eq!(
            snapshot(rank).predict(&query, at),
            CostPrediction::Known {
                central_service_time: scenario.service_time,
                upper_service_time: scenario.service_time,
                risk: CostRisk::ConditionalScenario {
                    dependence: CostScenarioDependence::SharedResidualRank,
                },
                support: CostSupport::ValidatedFeatureDomain,
                samples: 32,
                observed_at: at - Duration::from_secs(1),
                expires_at: at + Duration::from_secs(9),
            }
        );
    }
    assert_eq!(original.calls.load(Ordering::Relaxed), 1);
    assert_eq!(queries.load(Ordering::Relaxed), 1);
    let report = diagnostics.finish(None);
    assert_eq!(
        (report.provider_calls, report.cache_hits),
        (1, u64::try_from(count - 1).unwrap())
    );
    assert!(!report.query_budget_exhausted);
    assert_eq!(*valid_until.lock(), at + Duration::from_secs(9));
    assert_eq!(
        cache.lock().retained_bytes(),
        std::mem::size_of::<Option<Entry>>()
            + std::mem::size_of::<Option<NonZeroUsize>>()
            + std::mem::size_of::<CostWork>()
            + count * std::mem::size_of::<Duration>()
    );
    // No ABI-size gate: print the actual compiled target layout for evidence.
    eprintln!(
        "compact scenario cache layout: scenario={} duration={} entry_slot={} work={} ranks={count}",
        std::mem::size_of::<CostScenario>(),
        std::mem::size_of::<Duration>(),
        std::mem::size_of::<Option<Entry>>(),
        std::mem::size_of::<CostWork>(),
    );
    let different = [CostWork {
        context_tokens: 9,
        ..work[0]
    }];
    assert_eq!(
        snapshot(0).predict(&cost_query(&different), at),
        CostPrediction::Unknown(CostUnknownReason::PlanningBudget)
    );
    assert!(diagnostics.finish(None).query_budget_exhausted);
    assert_eq!(original.calls.load(Ordering::Relaxed), 1);
    assert_eq!(queries.load(Ordering::Relaxed), 1);
}

#[test]
fn compact_payload_exact_byte_boundary_and_rejected_replacement_preserve_old_entry() {
    let at = Instant::now();
    let count = 32;
    let work = [work()];
    let query = cost_query(&work);
    let minimum = std::mem::size_of::<Option<Entry>>()
        + std::mem::size_of::<Option<NonZeroUsize>>()
        + std::mem::size_of::<CostWork>()
        + count * std::mem::size_of::<Duration>();
    let mut exact = ScenarioCache::new(at, minimum, 1);
    exact.insert(&query, complete_distribution(at, count));
    assert_eq!(exact.retained_bytes(), minimum);
    assert!(exact.get(&query, at, count - 1).is_some());
    assert!(exact.get(&query, at, count).is_none());
    let mut short = ScenarioCache::new(at, minimum - 1, 1);
    let slot_storage = short.retained_bytes();
    short.insert(&query, complete_distribution(at, count));
    assert_eq!(short.retained_bytes(), slot_storage);
    assert!(short.get(&query, at, 0).is_none());

    // One slot guarantees collision. An oversized payload cannot evict a
    // previously valid distribution before the checked capacity decision.
    let larger = [
        work[0],
        CostWork {
            context_tokens: 8,
            ..work[0]
        },
    ];
    let (_, pending) = exact.get_with_key(&cost_query(&larger), at, 0);
    exact.insert_with_key(pending, complete_distribution(at, count));
    assert_eq!(exact.retained_bytes(), minimum);
    for rank in 0..count {
        assert!(exact.get(&query, at, rank).is_some());
    }
    assert!(exact.get(&cost_query(&larger), at, 0).is_none());
    let (_, pending) = exact.get_with_key(&cost_query(&larger), at, 0);
    exact.insert_with_key(pending, complete_distribution(at, 2));
    assert_eq!(
        exact.retained_bytes(),
        std::mem::size_of::<Option<Entry>>()
            + std::mem::size_of::<Option<NonZeroUsize>>()
            + 2 * std::mem::size_of::<CostWork>()
            + 2 * std::mem::size_of::<Duration>()
    );
    assert!(exact.get(&query, at, 0).is_none());
    assert!(exact.get(&cost_query(&larger), at, 1).is_some());
    assert!(exact.retained_bytes() <= minimum);
}

#[test]
fn compact_payload_does_not_renew_expiry_or_reuse_another_query_time() {
    let at = Instant::now();
    let work = [work()];
    let query = cost_query(&work);
    let mut cache = ScenarioCache::new(at, 8192, 1);
    cache.insert(&query, complete_distribution(at, 32));
    let expiry = at + Duration::from_secs(9);
    let prediction = cache.get(&query, at, 31).unwrap();
    assert!(matches!(prediction, CostPrediction::Known { expires_at, .. } if expires_at == expiry));
    assert!(cache
        .get(&query, at + Duration::from_nanos(1), 31)
        .is_none());
    assert_eq!(cache.hits(), 1);
    // A later comparison must query its own snapshot at its own time. The
    // existing distribution validator rejects at the exact expiry boundary;
    // an Unknown cannot be converted into a compact cache entry.
    let expired = complete_distribution(at, 32).validated(expiry, 32);
    assert_eq!(
        expired,
        CostScenarioPrediction::Unknown(CostUnknownReason::Stale)
    );
    let mut later = ScenarioCache::new(expiry, 8192, 1);
    later.insert(&query, expired);
    assert!(later.get(&query, expiry, 31).is_none());
}
