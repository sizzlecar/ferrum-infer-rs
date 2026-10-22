use super::two_slot::{colliding_work, keyed_distribution};
use super::*;
use crate::joint_policy::scenarios::ConditionalSnapshot;
use std::sync::atomic::AtomicUsize;

fn enabled(at: Instant, bytes: usize, queries: usize) -> ScenarioCache {
    let mut cache = ScenarioCache::new(at, bytes, queries);
    assert!(cache.enable_handles());
    cache
}

fn insert_handle(
    cache: &mut ScenarioCache,
    query: &CostQuery<'_>,
    prediction: CostScenarioPrediction,
) -> CacheHandle {
    let (_, key, _) = cache.get_with_handle(query, cache.at, 0, None);
    cache.insert_with_handle(key, prediction).unwrap()
}

#[test]
fn trace_handles_bind_owner_generation_secondary_slot_and_complete_rank_metadata() {
    let at = Instant::now();
    let keys = colliding_work(4, 3);
    let a = [keys[0]];
    let b = [keys[1]];
    let c = [keys[2]];
    let mut cache = enabled(at, 16 * 1024, 4);
    let first = insert_handle(&mut cache, &cost_query(&a), keyed_distribution(at, a[0], 3));
    let second = insert_handle(&mut cache, &cost_query(&b), keyed_distribution(at, b[0], 3));
    assert_ne!(first.slot, second.slot);
    assert_eq!(cache.handle_work(second, at, 2), Some(b.as_slice()));
    let before_hashes = cache.hash_lookups;
    for rank in 0..3 {
        let (prediction, key, handle) =
            cache.get_with_handle(&cost_query(&b), at, rank, Some(second));
        assert!(key.is_none());
        assert_eq!(handle, Some(second));
        assert_eq!(
            prediction,
            Some(CostPrediction::Known {
                central_service_time: Duration::from_micros(
                    1000 + b[0].context_tokens as u64 + rank as u64
                ),
                upper_service_time: Duration::from_micros(
                    1000 + b[0].context_tokens as u64 + rank as u64
                ),
                risk: CostRisk::ConditionalScenario {
                    dependence: CostScenarioDependence::SharedResidualRank
                },
                support: CostSupport::ValidatedFeatureDomain,
                samples: 32,
                observed_at: at - Duration::from_secs(1),
                expires_at: at + Duration::from_secs(9),
            })
        );
    }
    assert_eq!(cache.hash_lookups, before_hashes);
    assert_eq!(cache.handle_hits, 3);

    // Group one can replace an entry after a whole Split wave was preloaded.
    // The group-two prediction must reject its former identity and fall back.
    let third = insert_handle(&mut cache, &cost_query(&c), keyed_distribution(at, c[0], 3));
    assert_eq!(third.slot, first.slot);
    assert!(cache.handle_work(first, at, 0).is_none());
    assert_eq!(cache.handle_work(second, at, 0), Some(b.as_slice()));
    let (prediction, key, handle) = cache.get_with_handle(&cost_query(&a), at, 0, Some(first));
    assert!(prediction.is_none() && key.is_some() && handle.is_none());

    // Updating the same key also invalidates the old distribution identity.
    cache.insert(&cost_query(&b), keyed_distribution(at, b[0], 2));
    assert!(cache.handle_work(second, at, 0).is_none());
    let (prediction, _, refreshed) = cache.get_with_handle(&cost_query(&b), at, 1, Some(second));
    assert!(prediction.is_some());
    assert_ne!(refreshed, Some(second));
    assert!(cache.handle_work(refreshed.unwrap(), at, 2).is_none());
}

#[test]
fn trace_handles_reject_foreign_owner_time_rank_and_non_exact_execution() {
    let at = Instant::now();
    let work = [work()];
    let query = cost_query(&work);
    let mut first = enabled(at, 8192, 1);
    let mut second = enabled(at, 8192, 1);
    let a = insert_handle(&mut first, &query, distribution(at, 2));
    let b = insert_handle(&mut second, &query, distribution(at, 2));
    assert_eq!((a.slot, a.generation), (b.slot, b.generation));
    assert_ne!(a.owner, b.owner);
    assert!(second.handle_work(a, at, 0).is_none());
    let (prediction, _, handle) = second.get_with_handle(&query, at, 1, Some(a));
    assert!(prediction.is_some());
    assert_eq!(handle, Some(b));
    assert_eq!(
        second.handle_hits, 0,
        "foreign identity uses ordinary exact lookup"
    );
    assert!(second.handle_work(b, at, 2).is_none());
    assert!(second
        .handle_work(
            CacheHandle {
                slot: usize::MAX,
                ..b
            },
            at,
            0
        )
        .is_none());
    let ordinal = second.next_hint;
    let (prediction, key, handle) =
        second.get_with_handle(&query, at + Duration::from_nanos(1), 0, Some(b));
    assert!(prediction.is_none() && key.is_none() && handle.is_none());
    assert_eq!(second.next_hint, ordinal);
    assert!(second
        .handle_work(b, at + Duration::from_nanos(1), 0)
        .is_none());

    for query in [
        CostQuery {
            work_is_exact: false,
            ..query
        },
        CostQuery {
            execution: CostExecution::SequentialPrefillThenDecode,
            ..query
        },
    ] {
        let mut cache = enabled(at, 8192, 1);
        let (_, key, _) = cache.get_with_handle(&query, at, 0, None);
        assert!(cache.insert_with_handle(key, distribution(at, 2)).is_none());
        assert!(
            cache.get_with_key(&query, at, 1).0.is_some(),
            "ordinary cache still supports the original query kind"
        );
    }
}

#[test]
fn trace_handle_and_owner_exhaustion_never_wrap_or_reenable() {
    let owners = AtomicU64::new(u64::MAX - 1);
    assert_eq!(allocate_handle_owner(&owners).unwrap().get(), u64::MAX - 1);
    assert!(allocate_handle_owner(&owners).is_none());
    assert!(allocate_handle_owner(&owners).is_none());
    assert_eq!(owners.load(Ordering::Relaxed), u64::MAX);

    let at = Instant::now();
    let work = [work()];
    let query = cost_query(&work);
    let mut cache = enabled(at, 8192, 1);
    cache.last_generation = u64::MAX - 1;
    let old = insert_handle(&mut cache, &query, distribution(at, 2));
    assert_eq!(old.generation.get(), u64::MAX);
    let bytes = cache.retained_bytes();
    cache.insert(&query, distribution(at, 2));
    assert!(cache.handle_work(old, at, 0).is_none());
    assert!(!cache.enable_handles());
    assert_eq!(
        cache.retained_bytes(),
        bytes,
        "generation storage stays charged"
    );
    let (prediction, _, handle) = cache.get_with_handle(&query, at, 1, Some(old));
    assert!(prediction.is_some() && handle.is_none());
    cache.insert(&query, distribution(at, 2));
    assert!(cache.handle_work(old, at, 0).is_none());
}

#[test]
fn trace_generations_hints_and_payload_share_one_budget_with_atomic_failures() {
    let at = Instant::now();
    let base = std::mem::size_of::<Option<Entry>>() + std::mem::size_of::<Option<NonZeroUsize>>();
    let generation_bytes = std::mem::size_of::<u64>();
    let payload = std::mem::size_of::<CostWork>() + 2 * std::mem::size_of::<u64>();
    let trace_size = 64;
    let budget = base + generation_bytes + payload + trace_size;
    let mut cache = enabled(at, budget, 1);
    let ordinary = ScenarioCache::new(at, budget, 1);
    assert_eq!(cache.slots.len(), ordinary.slots.len());
    assert_eq!(cache.hints.len(), ordinary.hints.len());
    assert_eq!(
        cache.retained_bytes(),
        ordinary.retained_bytes() + generation_bytes
    );
    assert!(cache.set_trace_bytes(trace_size));
    let trace = vec![0u8; trace_size].into_boxed_slice();
    let work = [work()];
    let query = cost_query(&work);
    let old = insert_handle(&mut cache, &query, distribution(at, 2));
    assert_eq!(cache.retained_bytes(), budget);
    assert!(!cache.set_trace_bytes(trace_size + 1));
    assert!(!cache.set_trace_bytes(usize::MAX));
    assert_eq!(
        (cache.retained_bytes(), cache.trace_bytes),
        (budget, trace_size)
    );
    // A rank miss obtains a real insertion token for the same key. The larger
    // distribution cannot evict the old payload or consume its generation.
    let (_, key, _) = cache.get_with_handle(&query, at, 2, Some(old));
    let generation = cache.last_generation;
    assert!(cache.insert_with_handle(key, distribution(at, 3)).is_none());
    assert_eq!(cache.last_generation, generation);
    assert_eq!(cache.handle_work(old, at, 1), Some(work.as_slice()));
    assert_eq!(cache.retained_bytes(), budget);
    drop(trace);
    assert!(cache.set_trace_bytes(0));
    let (_, key, _) = cache.get_with_handle(&query, at, 2, None);
    let new = cache.insert_with_handle(key, distribution(at, 3)).unwrap();
    assert_ne!(old, new);
    assert!(cache.handle_work(old, at, 0).is_none());
    assert!(cache.retained_bytes() <= budget);

    for (bytes, queries) in [(0, 10), (8192, 0)] {
        let mut disabled = ScenarioCache::new(at, bytes, queries);
        assert!(!disabled.enable_handles());
        assert!(!disabled.set_trace_bytes(1));
        assert!(disabled.set_trace_bytes(0));
        assert_eq!(disabled.retained_bytes(), 0);
    }
    let mut seeded = ScenarioCache::new(at, 8192, 1);
    seeded.insert(&query, distribution(at, 2));
    let before = seeded.retained_bytes();
    assert!(
        !seeded.enable_handles(),
        "enable is a pre-seed capability only"
    );
    assert_eq!(seeded.retained_bytes(), before);
    assert!(seeded.get(&query, at, 1).is_some());
}

#[test]
fn trace_handle_fallback_and_interleaved_insertion_keep_one_ordinal_per_query() {
    let at = Instant::now();
    let keys = colliding_work(4, 3);
    let a = [keys[0]];
    let b = [keys[1]];
    let c = [keys[2]];
    let mut cache = enabled(at, 16 * 1024, 4);
    let a_handle = insert_handle(&mut cache, &cost_query(&a), keyed_distribution(at, a[0], 2));
    let b_handle = insert_handle(&mut cache, &cost_query(&b), keyed_distribution(at, b[0], 2));
    // A pending lookup for C occupies ordinal zero; a different lookup for B
    // interleaves before C's provider result fills the actual primary slot.
    cache.begin_rank();
    let (_, pending, _) = cache.get_with_handle(&cost_query(&c), at, 0, None);
    assert!(cache
        .get_with_handle(&cost_query(&b), at, 0, Some(b_handle))
        .0
        .is_some());
    let c_handle = cache
        .insert_with_handle(pending, keyed_distribution(at, c[0], 2))
        .unwrap();
    assert_eq!(cache.hints[0].unwrap().get() - 1, c_handle.slot);
    assert_eq!(cache.hints[1].unwrap().get() - 1, b_handle.slot);
    cache.begin_rank();
    // Trusted hit advances ordinal zero; stale A falls back at ordinal one,
    // succeeds for current C, and must not consume ordinal two as well.
    assert!(cache
        .get_with_handle(&cost_query(&b), at, 1, Some(b_handle))
        .0
        .is_some());
    let hashes = cache.hash_lookups;
    assert_eq!(
        cache
            .get_with_handle(&cost_query(&c), at, 1, Some(a_handle))
            .2,
        Some(c_handle)
    );
    assert_eq!(cache.next_hint, 2);
    assert_eq!(cache.hash_lookups, hashes + 1);
    cache.begin_rank();
    let hint_hits = cache.hint_hits;
    let hashes = cache.hash_lookups;
    assert!(cache.get_with_key(&cost_query(&b), at, 0).0.is_some());
    assert!(cache.get_with_key(&cost_query(&c), at, 0).0.is_some());
    assert_eq!(cache.next_hint, 2);
    assert_eq!(cache.hint_hits, hint_hits + 2);
    assert_eq!(cache.hash_lookups, hashes);
}

#[derive(Clone, Copy)]
enum Response {
    Valid,
    Unknown,
    Expired,
    InvalidRank,
    DifferentWeights,
}

struct CountedCosts {
    at: Instant,
    response: Response,
    calls: AtomicUsize,
}

impl CostSnapshot for CountedCosts {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("the real conditional consumer requires distributions")
    }
    fn cost_scenarios(
        &self,
        _: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if matches!(self.response, Response::Unknown) {
            return CostScenarioPrediction::Unknown(CostUnknownReason::NoCoverage);
        }
        let mut result = distribution(self.at, count);
        if let CostScenarioPrediction::Known {
            scenarios,
            expires_at,
            ..
        } = &mut result
        {
            match self.response {
                Response::Expired => *expires_at = self.at,
                Response::InvalidRank => scenarios[1].rank = 0,
                Response::DifferentWeights => {
                    scenarios[0].weight = 0.25;
                    scenarios[1].weight = 0.75;
                }
                Response::Valid | Response::Unknown => {}
            }
        }
        result
    }
}

#[test]
fn trace_handles_cannot_turn_provider_unknown_expiry_or_coupling_failure_into_replay() {
    for response in [
        Response::Unknown,
        Response::Expired,
        Response::InvalidRank,
        Response::DifferentWeights,
    ] {
        let at = Instant::now();
        let original = CountedCosts {
            at,
            response,
            calls: AtomicUsize::new(0),
        };
        let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
        config.planning.max_elapsed_us = 1_000_000;
        config.planning.max_queries = 1;
        config.planning.max_scenarios = 2;
        let cache = parking_lot::Mutex::new(enabled(at, 8192, 2));
        let bytes = cache.lock().retained_bytes();
        let queries = AtomicUsize::new(0);
        let valid_until = parking_lot::Mutex::new(at + Duration::from_secs(60));
        let diagnostics = crate::joint_policy::diagnostics::Recorder::new(1);
        let snapshot = ConditionalSnapshot {
            original: &original,
            rank: 0,
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
        let work = [work()];
        let (prediction, handle) = snapshot.predict_with_handle(&cost_query(&work), at, None);
        assert!(matches!(prediction, CostPrediction::Unknown(_)) && handle.is_none());
        assert_eq!(original.calls.load(Ordering::Relaxed), 1);
        let cache = cache.lock();
        assert!(cache.slots.iter().all(Option::is_none));
        assert_eq!(cache.retained_bytes(), bytes);
        assert_eq!(cache.last_generation, 0);
    }
}

#[test]
fn trace_handle_real_consumer_uses_rank_at_spent_provider_quota_but_obeys_time_and_misses() {
    let at = Instant::now();
    let original = CountedCosts {
        at,
        response: Response::Valid,
        calls: AtomicUsize::new(0),
    };
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_queries = 1;
    config.planning.max_scenarios = 2;
    let cache = parking_lot::Mutex::new(enabled(at, 8192, 2));
    let queries = AtomicUsize::new(0);
    let valid_until = parking_lot::Mutex::new(at + Duration::from_secs(60));
    let diagnostics = crate::joint_policy::diagnostics::Recorder::new(1);
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
    let work = [work()];
    let query = cost_query(&work);
    let (first, handle) = snapshot(0, at).predict_with_handle(&query, at, None);
    assert!(matches!(first, CostPrediction::Known { .. }));
    let handle = handle.unwrap();
    cache.lock().begin_rank();
    let (second, reused) = snapshot(1, at).predict_with_handle(&query, at, Some(handle));
    assert!(
        matches!(second, CostPrediction::Known { central_service_time, .. }
        if central_service_time == Duration::from_micros(1001))
    );
    assert_eq!(reused, Some(handle));
    assert_eq!(original.calls.load(Ordering::Relaxed), 1);
    assert_eq!(queries.load(Ordering::Relaxed), 1);
    assert_eq!(*valid_until.lock(), at + Duration::from_secs(9));
    let report = diagnostics.finish(None);
    assert_eq!((report.provider_calls, report.cache_hits), (1, 1));
    let ordinal = cache.lock().next_hint;
    let (late, late_handle) =
        snapshot(1, at - Duration::from_secs(2)).predict_with_handle(&query, at, Some(handle));
    assert_eq!(
        late,
        CostPrediction::Unknown(CostUnknownReason::PlanningBudget)
    );
    assert!(late_handle.is_none());
    assert_eq!(
        cache.lock().next_hint,
        ordinal,
        "the original time gate precedes even handle lookup"
    );
    let different = [CostWork {
        context_tokens: 99,
        ..work[0]
    }];
    let (miss, new_handle) = snapshot(1, at).predict_with_handle(&cost_query(&different), at, None);
    assert_eq!(
        miss,
        CostPrediction::Unknown(CostUnknownReason::PlanningBudget)
    );
    assert!(new_handle.is_none());
    assert!(diagnostics.finish(None).query_budget_exhausted);
    assert_eq!(original.calls.load(Ordering::Relaxed), 1);
}
