use super::super::accounting::{capture, TraceClearReason};
use super::*;

#[test]
fn accounting_separates_replacement_from_refused_bytes_and_preserves_cached_values() {
    let at = Instant::now();
    let keys = two_slot::colliding_work(2, 3);
    let payload = std::mem::size_of::<CostWork>() + 2 * std::mem::size_of::<Duration>();
    let trace = 64;
    let budget = 2 * std::mem::size_of::<Option<Entry>>()
        + 2 * std::mem::size_of::<Option<NonZeroUsize>>()
        + 2 * std::mem::size_of::<u64>()
        + 2 * payload
        + trace;
    let ((), captured) = capture(|| {
        let mut cache = ScenarioCache::new(at, budget, 2);
        assert!(cache.enable_handles());
        assert!(cache.set_trace_bytes(trace));
        for key in &keys[..2] {
            cache.insert(&cost_query(&[*key]), distribution(at, 2));
        }
        assert_eq!(cache.retained_bytes(), budget);
        // Same-key refresh and a different-key eviction are distinct events.
        cache.insert(&cost_query(&[keys[0]]), distribution(at, 2));
        cache.insert(&cost_query(&[keys[2]]), distribution(at, 2));
        assert!(cache.get(&cost_query(&[keys[0]]), at, 0).is_none());
        assert!(cache.get(&cost_query(&[keys[1]]), at, 1).is_some());
        assert!(cache.get(&cost_query(&[keys[2]]), at, 1).is_some());
        // More retained ranks fail the same original byte guard, with no eviction.
        cache.insert(&cost_query(&[keys[0]]), distribution(at, 3));
        assert!(cache.get(&cost_query(&[keys[2]]), at, 1).is_some());
        cache.note_trace_clear(TraceClearReason::ActualDescriptorCapacityRefused);
        assert!(cache.set_trace_bytes(0));
    });
    assert_eq!(captured.dropped_reports, 0);
    let report = captured.reports[0].unwrap();
    assert!(captured.reports[1].is_none());
    let count = report.counts;
    assert_eq!(count.known_insert_attempts, 5);
    assert_eq!(count.empty_insertions, 2);
    assert_eq!(count.exact_key_replacements, 1);
    assert_eq!(count.collision_replacements, 1);
    assert_eq!(count.byte_limit_rejections, 1);
    assert_eq!(count.arithmetic_rejections, 0);
    assert_eq!(count.occupied_slots, 2);
    let first = count.first_byte_rejection.unwrap();
    assert_eq!(first.before.trace, trace);
    assert_eq!(first.before.retained, budget);
    assert!(first.before.sum_matches_ledger);
    assert_eq!(first.victim_payload_bytes, payload);
    assert_eq!(
        first.proposed_retained_bytes,
        budget + std::mem::size_of::<Duration>()
    );
    assert_eq!(
        count.last_byte_rejection.unwrap().proposed_retained_bytes,
        first.proposed_retained_bytes
    );
    assert_eq!(report.at_cache_teardown_ledger.trace, 0);
    assert_eq!(report.at_cache_teardown_ledger.retained, budget - trace);
    assert!(report.at_cache_teardown_ledger.sum_matches_ledger);
    assert_eq!(count.trace_clears.actual_descriptor_capacity_refused, 1);
    assert_eq!(count.trace_clears.with_positive_trace_charge, 1);
    assert_eq!(count.trace_clears.last_charge_before_clear, trace);
    assert_eq!(
        report.layout.option_cache_handle,
        std::mem::size_of::<Option<CacheHandle>>()
    );
    assert_eq!(
        report.layout.wave_trace,
        crate::joint_policy::scenarios::trace::wave_trace_size()
    );
}

#[test]
fn accounting_scopes_are_thread_local_restore_after_unwind_and_report_overflow() {
    let at = Instant::now();
    let ((), outer) = capture(|| {
        let cache = ScenarioCache::new(at, 1024, 2);
        // Both fresh threads start with their first capture identity. Only
        // ThreadId distinguishes them, independent of this test thread's past.
        let moved_cache = std::thread::spawn(move || capture(|| ScenarioCache::new(at, 1024, 1)).0)
            .join()
            .unwrap();
        std::thread::spawn(move || {
            let ((), other) = capture(|| drop(moved_cache));
            assert!(other.reports.iter().all(Option::is_none));
            assert_eq!(other.dropped_reports, 0);
        })
        .join()
        .unwrap();
        let ((), inner) = capture(|| drop(ScenarioCache::new(at, 1024, 1)));
        assert_eq!(inner.reports[0].unwrap().slots, 1);
        assert!(std::panic::catch_unwind(|| capture(|| panic!("test-only unwind"))).is_err());
        std::thread::spawn(move || {
            let ((), other) = capture(|| drop(ScenarioCache::new(at, 1024, 3)));
            assert_eq!(other.reports[0].unwrap().slots, 3);
        })
        .join()
        .unwrap();
        drop(cache);
        for _ in 0..3 {
            drop(ScenarioCache::new(at, 1024, 1));
        }
    });
    assert_eq!(outer.reports[0].unwrap().slots, 2);
    assert_eq!(outer.reports[1].unwrap().slots, 1);
    assert_eq!(outer.dropped_reports, 2);
    // A cache that outlives a capture cannot leak into a later capture.
    let (old_cache, old_capture) = capture(|| ScenarioCache::new(at, 1024, 1));
    assert!(old_capture.reports.iter().all(Option::is_none));
    let ((), later) = capture(|| drop(old_cache));
    assert!(later.reports.iter().all(Option::is_none));
    assert_eq!(later.dropped_reports, 0);
}
