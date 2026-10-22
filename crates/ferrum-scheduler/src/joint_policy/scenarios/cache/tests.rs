use super::*;
use ferrum_execution::{execution_attempt::AttemptWorkKind, token_execution::ExecutionRange};

mod two_slot;

fn work() -> CostWork {
    CostWork {
        kind: AttemptWorkKind::Decode,
        requested: ExecutionRange::new(3, 1).unwrap(),
        context_tokens: 4,
        prefix_restore: AttemptPrefixRestore::NoneObserved,
        internally_reused_tokens: None,
    }
}

fn cost_query(work: &[CostWork]) -> CostQuery<'_> {
    CostQuery {
        work,
        work_is_exact: true,
        execution: CostExecution::SingleInvocation,
    }
}

fn distribution(at: Instant, count: usize) -> CostScenarioPrediction {
    CostScenarioPrediction::Known {
        central_service_time: Duration::from_millis(1),
        scenarios: (0..count)
            .map(|rank| CostScenario {
                rank,
                weight: 1.0 / count as f64,
                service_time: Duration::from_micros(1000 + rank as u64),
            })
            .collect(),
        dependence: CostScenarioDependence::SharedResidualRank,
        support: CostSupport::ValidatedFeatureDomain,
        samples: 32,
        observed_at: at - Duration::from_secs(1),
        expires_at: at + Duration::from_secs(9),
    }
}

#[test]
fn colliding_keys_require_every_work_field_exactness_execution_length_and_order() {
    let at = Instant::now();
    let first = work();
    let second = CostWork {
        requested: ExecutionRange::new(8, 1).unwrap(),
        context_tokens: 9,
        ..first
    };
    let base = vec![first, second];
    let mut variants = vec![
        ("base", base.clone(), true, CostExecution::SingleInvocation),
        (
            "exact",
            base.clone(),
            false,
            CostExecution::SingleInvocation,
        ),
        (
            "execution",
            base.clone(),
            true,
            CostExecution::SequentialPrefillThenDecode,
        ),
        ("length", vec![first], true, CostExecution::SingleInvocation),
        (
            "order",
            vec![second, first],
            true,
            CostExecution::SingleInvocation,
        ),
    ];
    for (name, changed) in [
        (
            "kind",
            CostWork {
                kind: AttemptWorkKind::Prefill,
                ..first
            },
        ),
        (
            "start",
            CostWork {
                requested: ExecutionRange::new(2, 2).unwrap(),
                ..first
            },
        ),
        (
            "end",
            CostWork {
                requested: ExecutionRange::new(3, 2).unwrap(),
                ..first
            },
        ),
        (
            "context",
            CostWork {
                context_tokens: 5,
                ..first
            },
        ),
        (
            "restore1",
            CostWork {
                prefix_restore: AttemptPrefixRestore::Restored { tokens: 1 },
                ..first
            },
        ),
        (
            "restore2",
            CostWork {
                prefix_restore: AttemptPrefixRestore::Restored { tokens: 2 },
                ..first
            },
        ),
        (
            "reuse0",
            CostWork {
                internally_reused_tokens: Some(0),
                ..first
            },
        ),
        (
            "reuse1",
            CostWork {
                internally_reused_tokens: Some(1),
                ..first
            },
        ),
    ] {
        variants.push((
            name,
            vec![changed, second],
            true,
            CostExecution::SingleInvocation,
        ));
    }
    // Exactly one slot forces every pair to collide independently of the hash
    // algorithm. No fixed hash value or accidentally distinct hash is assumed.
    let mut cache = ScenarioCache::new(at, 16 * 1024, 1);
    for (stored_index, (stored_name, stored_work, exact, execution)) in variants.iter().enumerate()
    {
        let stored = CostQuery {
            work: stored_work,
            work_is_exact: *exact,
            execution: *execution,
        };
        cache.insert(&stored, distribution(at, 2));
        for (queried_index, (name, work, exact, execution)) in variants.iter().enumerate() {
            let query = CostQuery {
                work,
                work_is_exact: *exact,
                execution: *execution,
            };
            assert_eq!(
                cache.get(&query, at, 0).is_some(),
                stored_index == queried_index,
                "stored={stored_name}, queried={name}"
            );
        }
    }
}

#[test]
fn reused_miss_key_accounts_for_the_entry_present_at_insertion_and_keeps_bounds() {
    let at = Instant::now();
    let a = [work()];
    let b = [
        work(),
        CostWork {
            context_tokens: 8,
            ..work()
        },
    ];
    let a = cost_query(&a);
    let b = cost_query(&b);
    let slot_bytes = std::mem::size_of::<Option<Entry>>();
    let budget =
        slot_bytes + 2 * std::mem::size_of::<CostWork>() + 4 * std::mem::size_of::<CostScenario>();
    let mut cache = ScenarioCache::new(at, budget, 1);
    let (miss, key) = cache.get_with_key(&a, at, 0);
    assert!(miss.is_none());
    assert!(key.is_some());
    // The provider call is outside the lock. If another lookup/insertion has
    // replaced this collision slot, subtract its CURRENT payload, not an old
    // byte count captured in the token.
    cache.insert(&b, distribution(at, 4));
    assert_eq!(cache.retained_bytes(), budget);
    cache.insert_with_key(key, distribution(at, 2));
    assert_eq!(
        cache.retained_bytes(),
        slot_bytes + std::mem::size_of::<CostWork>() + 2 * std::mem::size_of::<CostScenario>()
    );
    assert!(cache.get(&a, at, 1).is_some());
    assert!(cache.get(&b, at, 0).is_none());
    let before = cache.retained_bytes();
    let (_, key) = cache.get_with_key(&b, at, 0);
    cache.insert_with_key(
        key,
        CostScenarioPrediction::Unknown(CostUnknownReason::NoCoverage),
    );
    assert_eq!(cache.retained_bytes(), before);
    assert!(cache.get(&a, at, 0).is_some());
    let oversized = vec![work(); 100];
    let (_, key) = cache.get_with_key(&cost_query(&oversized), at, 0);
    cache.insert_with_key(key, distribution(at, 32));
    assert_eq!(cache.retained_bytes(), before);
    assert!(cache.get(&a, at, 0).is_some());
    assert!(cache.retained_bytes() <= budget);
}

#[test]
fn cached_ranks_keep_prediction_metadata_and_tokens_cannot_bypass_time_or_disabled_cache() {
    let at = Instant::now();
    let work = [work()];
    let query = cost_query(&work);
    let mut cache = ScenarioCache::new(at, 8192, 1);
    let (_, key) = cache.get_with_key(&query, at, 0);
    cache.insert_with_key(key, distribution(at, 4));
    for rank in 0..4 {
        let (prediction, key) = cache.get_with_key(&query, at, rank);
        assert!(key.is_none(), "a hit has no insertion to perform");
        assert_eq!(
            prediction,
            Some(CostPrediction::Known {
                central_service_time: Duration::from_micros(1000 + rank as u64),
                upper_service_time: Duration::from_micros(1000 + rank as u64),
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
    let hits = cache.hits();
    let (prediction, key) = cache.get_with_key(&query, at + Duration::from_nanos(1), 0);
    assert!(prediction.is_none() && key.is_none());
    assert_eq!(cache.hits(), hits);
    assert!(cache.get(&query, at, 4).is_none());
    for (bytes, queries) in [(0, 10), (8192, 0)] {
        let mut disabled = ScenarioCache::new(at, bytes, queries);
        let (prediction, key) = disabled.get_with_key(&query, at, 0);
        assert!(prediction.is_none() && key.is_none());
        disabled.insert_with_key(key, distribution(at, 4));
        assert_eq!(disabled.retained_bytes(), 0);
        assert_eq!(disabled.hits(), 0);
    }
    // Tokens are private to the original predict call, with defensive checks
    // for accidental time/geometry misuse; the immutable query is in the token.
    let other = [CostWork {
        context_tokens: 9,
        ..work[0]
    }];
    let (_, key) = cache.get_with_key(&cost_query(&other), at, 0);
    let mut other_time = ScenarioCache::new(at + Duration::from_nanos(1), 8192, 1);
    other_time.insert_with_key(key, distribution(at, 4));
    assert!(other_time.slots.iter().all(Option::is_none));
    let (_, key) = cache.get_with_key(&cost_query(&other), at, 0);
    let mut other_geometry = ScenarioCache::new(at, 8192, 2);
    other_geometry.insert_with_key(key, distribution(at, 4));
    assert!(other_geometry.slots.iter().all(Option::is_none));
}
