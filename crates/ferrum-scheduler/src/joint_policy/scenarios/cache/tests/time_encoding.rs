use super::*;
use crate::joint_policy::scenarios::ConditionalSnapshot;
use std::sync::atomic::{AtomicUsize, Ordering};

fn full_range_times() -> Vec<Duration> {
    let boundary = Duration::from_nanos(u64::MAX);
    vec![
        Duration::from_nanos(1),
        boundary,
        boundary.checked_add(Duration::from_nanos(1)).unwrap(),
        Duration::MAX,
    ]
}

fn with_times(at: Instant, values: &[Duration]) -> CostScenarioPrediction {
    let mut result = distribution(at, values.len());
    if let CostScenarioPrediction::Known {
        central_service_time,
        scenarios,
        ..
    } = &mut result
    {
        *central_service_time = values[0];
        for (scenario, value) in scenarios.iter_mut().zip(values) {
            scenario.service_time = *value;
        }
    }
    result.validated(at, values.len())
}

#[test]
fn time_encoding_preserves_complete_duration_range_rank_order_and_checked_bounds() {
    let at = Instant::now();
    let values = full_range_times();
    for (values, expected) in [
        (values[..2].to_vec(), TimeEncoding::Nanoseconds),
        (values.clone(), TimeEncoding::SecondsNanoseconds),
        (
            vec![values[3], values[0], values[2], values[1]],
            TimeEncoding::SecondsNanoseconds,
        ),
    ] {
        let CostScenarioPrediction::Known { scenarios, .. } = with_times(at, &values) else {
            panic!("positive full-range durations remain valid cost values")
        };
        let encoding = TimeEncoding::select(&scenarios);
        assert_eq!(encoding, expected);
        let bytes = encoding.payload_bytes(scenarios.len()).unwrap();
        let encoded = encoding.pack(scenarios);
        assert_eq!(std::mem::size_of_val(encoded.as_ref()), bytes);
        for (rank, value) in values.iter().enumerate() {
            assert_eq!(encoding.get(&encoded, rank), Some(*value));
        }
        assert_eq!(encoding.get(&encoded, values.len()), None);
        assert_eq!(encoding.get(&encoded, usize::MAX), None);
        assert_eq!(encoding.payload_bytes(usize::MAX), None);
    }
    assert_eq!(TimeEncoding::SecondsNanoseconds.get(&[1], 0), None);
    assert_eq!(
        TimeEncoding::SecondsNanoseconds.get(&[u64::MAX, 1_000_000_000], 0),
        None
    );
    assert_eq!(
        TimeEncoding::SecondsNanoseconds.get(&[u64::MAX, u64::MAX], 0),
        None
    );
    eprintln!(
        "lossless time cache layout: entry={} option_entry={} word={} duration={} encoding={}",
        std::mem::size_of::<Entry>(),
        std::mem::size_of::<Option<Entry>>(),
        std::mem::size_of::<u64>(),
        std::mem::size_of::<Duration>(),
        std::mem::size_of::<TimeEncoding>(),
    );
}

#[test]
fn time_encoding_rejected_wide_refresh_preserves_handle_and_accepted_refresh_replaces_it() {
    let at = Instant::now();
    let values = full_range_times();
    let narrow = vec![Duration::from_nanos(7); values.len()];
    let narrow_bytes = TimeEncoding::Nanoseconds
        .payload_bytes(narrow.len())
        .unwrap();
    let wide_bytes = TimeEncoding::SecondsNanoseconds
        .payload_bytes(values.len())
        .unwrap();
    let trace = wide_bytes - narrow_bytes;
    let budget = std::mem::size_of::<Option<Entry>>()
        + std::mem::size_of::<Option<NonZeroUsize>>()
        + std::mem::size_of::<u64>()
        + std::mem::size_of::<CostWork>()
        + wide_bytes;
    let work = [work()];
    let query = cost_query(&work);
    let ((), capture) = super::super::accounting::capture(|| {
        let mut cache = ScenarioCache::new(at, budget, 1);
        assert!(cache.enable_handles());
        assert!(cache.set_trace_bytes(trace));
        cache.insert(&query, with_times(at, &narrow));
        let (_, _, old) = cache.get_with_handle(&query, at, 0, None);
        let old = old.unwrap();
        let generation = cache.last_generation;
        assert_eq!(cache.retained_bytes(), budget);
        cache.insert(&query, with_times(at, &values));
        assert_eq!(cache.last_generation, generation);
        assert_eq!(
            cache.handle_work(old, at, values.len() - 1),
            Some(work.as_slice())
        );
        assert!(matches!(cache.get_with_handle(&query, at, 3, Some(old)).0,
            Some(CostPrediction::Known { central_service_time, .. }) if central_service_time == narrow[3]));
        assert!(cache.set_trace_bytes(0));
        cache.insert(&query, with_times(at, &values));
        assert!(cache.handle_work(old, at, 0).is_none());
        let (_, _, fresh) = cache.get_with_handle(&query, at, 0, None);
        let fresh = fresh.unwrap();
        assert_ne!(fresh, old);
        for (rank, value) in values.iter().enumerate() {
            assert!(
                matches!(cache.get_with_handle(&query, at, rank, Some(fresh)).0,
                Some(CostPrediction::Known { central_service_time, upper_service_time, .. })
                if central_service_time == *value && upper_service_time == *value)
            );
        }
        assert!(cache.handle_work(fresh, at, values.len()).is_none());
        assert!(cache.handle_work(fresh, at, usize::MAX).is_none());
        assert_eq!(cache.retained_bytes(), budget);
    });
    let report = capture.reports[0].unwrap();
    assert_eq!(report.counts.byte_limit_rejections, 1);
    assert_eq!(
        report.at_cache_teardown_ledger.service_time_payload,
        wide_bytes
    );
    assert!(report.at_cache_teardown_ledger.sum_matches_ledger);
    assert_eq!(
        report
            .counts
            .first_byte_rejection
            .unwrap()
            .before
            .service_time_payload,
        narrow_bytes
    );
}

#[test]
fn time_encoding_wide_exact_byte_boundary_and_expiry_do_not_invent_extra_ranks() {
    let at = Instant::now();
    let values = full_range_times();
    let payload = TimeEncoding::SecondsNanoseconds
        .payload_bytes(values.len())
        .unwrap();
    let budget = std::mem::size_of::<Option<Entry>>()
        + std::mem::size_of::<Option<NonZeroUsize>>()
        + std::mem::size_of::<CostWork>()
        + payload;
    let work = [work()];
    let query = cost_query(&work);
    let mut exact = ScenarioCache::new(at, budget, 1);
    exact.insert(&query, with_times(at, &values));
    assert_eq!(exact.retained_bytes(), budget);
    assert!(exact.get(&query, at, values.len() - 1).is_some());
    for rank in values.len()..values.len() * 2 {
        assert!(exact.get(&query, at, rank).is_none());
    }
    assert!(exact.get(&query, at, usize::MAX).is_none());
    assert!(exact.get(&query, at + Duration::from_nanos(1), 0).is_none());
    let mut short = ScenarioCache::new(at, budget - 1, 1);
    let fixed = short.retained_bytes();
    short.insert(&query, with_times(at, &values));
    assert_eq!(short.retained_bytes(), fixed);
    assert!(short.get(&query, at, 0).is_none());
    let expiry = at + Duration::from_secs(9);
    let expired = with_times(at, &values).validated(expiry, values.len());
    assert_eq!(
        expired,
        CostScenarioPrediction::Unknown(CostUnknownReason::Stale)
    );
    let mut later = ScenarioCache::new(expiry, budget, 1);
    later.insert(&query, expired);
    assert!(later.get(&query, expiry, 0).is_none());
}

struct WideCosts {
    at: Instant,
    calls: AtomicUsize,
}

impl CostSnapshot for WideCosts {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        Default::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("conditional consumer must use complete distributions")
    }
    fn cost_scenarios(
        &self,
        _: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let values = full_range_times();
        assert_eq!(count, values.len());
        with_times(self.at, &values)
    }
}

#[test]
fn time_encoding_real_conditional_consumer_keeps_every_rank_at_spent_provider_quota() {
    let at = Instant::now();
    let values = full_range_times();
    let original = WideCosts {
        at,
        calls: AtomicUsize::new(0),
    };
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_queries = 1;
    config.planning.max_scenarios = values.len();
    let cache = parking_lot::Mutex::new(ScenarioCache::new(at, 8192, 1));
    assert!(cache.lock().enable_handles());
    let queries = AtomicUsize::new(0);
    let valid_until = parking_lot::Mutex::new(at + Duration::from_secs(60));
    let diagnostics = crate::joint_policy::diagnostics::Recorder::new(1);
    let snapshot = |rank| ConditionalSnapshot {
        original: &original,
        rank,
        weights: &[0.25; 4],
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
    let mut handle = None;
    for (rank, value) in values.iter().enumerate() {
        let (prediction, fresh) = snapshot(rank).predict_with_handle(&query, at, handle);
        assert_eq!(
            prediction,
            CostPrediction::Known {
                central_service_time: *value,
                upper_service_time: *value,
                risk: CostRisk::ConditionalScenario {
                    dependence: CostScenarioDependence::SharedResidualRank
                },
                support: CostSupport::ValidatedFeatureDomain,
                samples: 32,
                observed_at: at - Duration::from_secs(1),
                expires_at: at + Duration::from_secs(9),
            }
        );
        assert!(fresh.is_some());
        handle = fresh;
    }
    assert_eq!(original.calls.load(Ordering::Relaxed), 1);
    assert_eq!(queries.load(Ordering::Relaxed), 1);
    assert_eq!(
        diagnostics.finish(None).cache_hits,
        (values.len() - 1) as u64
    );
    assert_eq!(*valid_until.lock(), at + Duration::from_secs(9));
    let other = [CostWork {
        context_tokens: 9,
        ..work[0]
    }];
    assert_eq!(
        snapshot(0).predict(&cost_query(&other), at),
        CostPrediction::Unknown(CostUnknownReason::PlanningBudget)
    );
    assert!(diagnostics.finish(None).query_budget_exhausted);
    assert_eq!(original.calls.load(Ordering::Relaxed), 1);
}
