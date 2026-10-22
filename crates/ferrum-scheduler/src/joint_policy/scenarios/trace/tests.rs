use super::*;
use ferrum_execution::{execution_attempt::AttemptPrefixRestore, token_execution::ExecutionRange};
use std::cell::Cell;

struct NeverPredict;
impl CostSnapshot for NeverPredict {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("trace storage must never predict")
    }
}

struct Harness {
    now: Instant,
    config: SchedulerSloConfig,
    cache: parking_lot::Mutex<ScenarioCache>,
    queries: AtomicUsize,
    expiry: parking_lot::Mutex<Instant>,
    diagnostics: diagnostics::Recorder,
}

impl Harness {
    fn new(slots: usize, bytes: usize) -> Self {
        let now = Instant::now();
        let mut cache = ScenarioCache::new(now, bytes, slots);
        cache.enable_handles();
        Self {
            now,
            config: "ttft:1000,tpot:100,itl:100".parse().unwrap(),
            cache: parking_lot::Mutex::new(cache),
            queries: AtomicUsize::new(0),
            expiry: parking_lot::Mutex::new(now + Duration::from_secs(60)),
            diagnostics: diagnostics::Recorder::new(4096),
        }
    }

    fn snapshot(&self) -> ConditionalSnapshot<'_> {
        ConditionalSnapshot {
            original: &NeverPredict,
            rank: 0,
            weights: &[0.5, 0.5],
            dependence: CostScenarioDependence::SharedResidualRank,
            query_limits: CostQueryLimits::default(),
            config: &self.config,
            started: self.now,
            queries: &self.queries,
            cache: &self.cache,
            valid_until: &self.expiry,
            diagnostics: &self.diagnostics,
        }
    }

    fn insert(&self, work: &[CostWork]) -> CacheHandle {
        let query = CostQuery {
            work,
            work_is_exact: true,
            execution: CostExecution::SingleInvocation,
        };
        let mut cache = self.cache.lock();
        let (prediction, key, handle) = cache.get_with_handle(&query, self.now, 0, None);
        if prediction.is_some() {
            return handle.unwrap();
        }
        cache
            .insert_with_handle(
                key,
                CostScenarioPrediction::Known {
                    central_service_time: Duration::from_millis(1),
                    scenarios: vec![
                        CostScenario {
                            rank: 0,
                            weight: 0.5,
                            service_time: Duration::from_millis(1),
                        },
                        CostScenario {
                            rank: 1,
                            weight: 0.5,
                            service_time: Duration::from_millis(2),
                        },
                    ],
                    dependence: CostScenarioDependence::SharedResidualRank,
                    support: CostSupport::ValidatedFeatureDomain,
                    samples: 2,
                    observed_at: self.now,
                    expires_at: self.now + Duration::from_secs(60),
                },
            )
            .unwrap()
    }
}

fn requests() -> Vec<SimulatedRequest> {
    (0..2)
        .map(|_| SimulatedRequest {
            id: ferrum_types::RequestId::new(),
            prompt_tokens: 3,
            prefill_tokens: 0,
            output_tokens: 0,
            output_cap: 4,
            scenario_end_tokens: 4,
            age: Duration::ZERO,
            central_clock: RequestClock::default(),
            risk_clock: RequestClock::default(),
        })
        .collect()
}

fn groups() -> Vec<Vec<GroupMember>> {
    vec![
        vec![GroupMember {
            request: 0,
            work: CostWork {
                kind: AttemptWorkKind::Prefill,
                requested: ExecutionRange::new(0, 1).unwrap(),
                context_tokens: 3,
                prefix_restore: AttemptPrefixRestore::NoneObserved,
                internally_reused_tokens: None,
            },
        }],
        vec![GroupMember {
            request: 1,
            work: CostWork {
                kind: AttemptWorkKind::Decode,
                requested: ExecutionRange::new(1, 1).unwrap(),
                context_tokens: 2,
                prefix_restore: AttemptPrefixRestore::NoneObserved,
                internally_reused_tokens: None,
            },
        }],
    ]
}

fn actual_bytes(store: &TraceStore) -> usize {
    store.initial_bytes().unwrap()
        + store
            .actions
            .iter()
            .map(|action| {
                action.waves.capacity() * std::mem::size_of::<WaveTrace>()
                    + action
                        .waves
                        .iter()
                        .map(|wave| std::mem::size_of_val(wave.requests.as_ref()))
                        .sum::<usize>()
            })
            .sum::<usize>()
}

#[test]
fn trace_handles_account_descriptor_capacity_indexes_reset_and_no_storage() {
    let harness = Harness::new(8, 128 * 1024);
    let snapshot = harness.snapshot();
    let groups = groups();
    let recorded = recorded_groups(&groups);
    let handles = [
        Some(harness.insert(&[groups[0][0].work])),
        Some(harness.insert(&[groups[1][0].work])),
    ];
    let baseline = harness.cache.lock().retained_bytes();
    let mut store = TraceStore::default();
    let mut requests = requests();
    assert!(store.prepare(&requests, 1, &harness.cache, &harness.diagnostics));
    for round in 0..32 {
        store.publish(
            0,
            round,
            TracePhase::Continuation,
            recorded.groups(),
            handles,
            &snapshot,
            harness.now,
            false,
        );
        assert_eq!(store.actions[0].waves.len(), round + 1);
        assert_eq!(store.charged_bytes, actual_bytes(&store));
        assert_eq!(
            harness.cache.lock().retained_bytes(),
            baseline + store.charged_bytes
        );
    }
    let bytes = store.charged_bytes;
    store.publish(
        0,
        0,
        TracePhase::Continuation,
        recorded.groups(),
        handles,
        &snapshot,
        harness.now,
        true,
    );
    assert_eq!(
        store.charged_bytes, bytes,
        "unchanged replay does not republish storage"
    );
    assert_eq!(harness.queries.load(Ordering::Relaxed), 0);
    let peak = harness.diagnostics.finish(None).cache_peak_bytes;
    assert_eq!(peak, harness.cache.lock().retained_bytes());
    requests[0].scenario_end_tokens = 3;
    assert!(store.prepare(&requests, 1, &harness.cache, &harness.diagnostics));
    assert!(store.actions[0].waves.is_empty());
    assert_eq!(store.charged_bytes, actual_bytes(&store));
    assert_eq!(
        harness.cache.lock().retained_bytes(),
        baseline + store.charged_bytes
    );
    assert!(harness.diagnostics.finish(None).cache_peak_bytes >= peak);
    store.clear(&mut harness.cache.lock());
    assert_eq!(store.charged_bytes, 0);
    assert_eq!(harness.cache.lock().retained_bytes(), baseline);

    let tiny = Harness::new(8, 0);
    assert!(!store.prepare(&requests, 1, &tiny.cache, &tiny.diagnostics));
    assert_eq!(actual_bytes(&store), 0);
    assert_eq!(tiny.cache.lock().retained_bytes(), 0);
}

#[test]
fn trace_handles_require_complete_handles_and_fallback_before_any_replay_append() {
    let harness = Harness::new(2, 128 * 1024);
    let snapshot = harness.snapshot();
    let groups = groups();
    let recorded = recorded_groups(&groups);
    let handles = [
        Some(harness.insert(&[groups[0][0].work])),
        Some(harness.insert(&[groups[1][0].work])),
    ];
    let mut store = TraceStore::default();
    assert!(store.prepare(&requests(), 1, &harness.cache, &harness.diagnostics));
    let initial = store.charged_bytes;
    store.publish(
        0,
        0,
        TracePhase::First,
        recorded.groups(),
        [handles[0], None],
        &snapshot,
        harness.now,
        false,
    );
    assert!(store.actions[0].waves.is_empty());
    assert_eq!(store.charged_bytes, initial);
    store.publish(
        0,
        0,
        TracePhase::First,
        recorded.groups(),
        handles,
        &snapshot,
        harness.now,
        false,
    );
    assert_eq!(store.actions[0].waves.len(), 1);

    // Two occupied slots: the third distinct lawful key must replace one of
    // this wave's entries. Neither previously applied geometry nor raw slot
    // identity can authorize replay of the now-incomplete wave.
    let mut replacement = groups[0][0].work;
    replacement.context_tokens = 7;
    harness.insert(&[replacement]);
    let calls = Cell::new(0);
    let replay = |_plan: &PlanBuilder| {
        calls.set(calls.get() + 1);
    };
    let mut plan = PlanBuilder {
        work: Vec::new(),
        requests: Vec::new(),
        group_ends: [0; 2],
        members: Vec::new(),
        group_count: 0,
        participant_limit: 2,
        query_limits: CostQueryLimits::default(),
        execution: CostExecution::SequentialPrefillThenDecode,
        active: Some(vec![true; 2]),
    };
    let mut trace = TraceEvaluation::new(&mut store, &snapshot, &replay, harness.now);
    assert!(!trace.try_plan(0, 0, TracePhase::First, &mut plan).unwrap());
    assert_eq!(calls.get(), 0);
    assert_eq!(plan.group_count, 0);
    assert!(plan.members.is_empty());
    plan.push_group(&groups[0]).unwrap();
    plan.push_group(&groups[1]).unwrap();
    assert_eq!(
        plan.group_count, 2,
        "ordinary fallback sees no partial replay mutation"
    );
    assert_eq!(harness.queries.load(Ordering::Relaxed), 0);
    assert_eq!(trace.store.charged_bytes, actual_bytes(trace.store));
}

fn recorded_groups(groups: &[Vec<GroupMember>]) -> PlanBuilder {
    let mut plan = PlanBuilder {
        work: Vec::new(),
        requests: Vec::new(),
        group_ends: [0; 2],
        members: Vec::new(),
        group_count: 0,
        participant_limit: 2,
        query_limits: CostQueryLimits::default(),
        execution: CostExecution::SequentialPrefillThenDecode,
        active: Some(vec![true; 2]),
    };
    for group in groups {
        plan.push_group(group).unwrap();
    }
    plan
}
