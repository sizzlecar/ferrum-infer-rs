use super::*;

#[test]
fn unknown_physical_wave_retains_original_failure_at_its_raw_fifo_entry() {
    let (live, actual, _) = fixture(256);
    let queue = sink(2, 256);
    let ticket = live.reserve(Some(1)).unwrap();
    let (call, clock) = begin(&actual, &queue);
    let mut call = call.with_live_ticket(Some(ticket));
    {
        let mut context = call.context().unwrap();
        context.physical_wave(Err(ActualWaveEvidenceUnknown::GraphPath), Some(3));
        clock.set(6);
        context.terminal(ActualWaveOutcome::Completed, None);
        context.finish_call(ObservedCallOutcome::Completed);
    }
    assert_eq!(call.finish(), CostCallDisposition::Queued);
    assert!(queue.pop_numbered_with_ticket().is_none());
    let audit = live.audit();
    assert!(audit.population.failed && audit.population.closed);
    assert_eq!((audit.population.issued, audit.population.retired), (1, 1));
    assert!(audit.first_settlement_failure.is_none());
    assert_eq!(audit.retained_waves, 0);
    assert!(
        live.reserve(Some(7)).is_none(),
        "no replacement of failed offer"
    );
    let failed = serde_json::to_value(audit.population.first_ticket_failure.unwrap()).unwrap();
    assert_eq!(failed["ticket"], 1);
    assert_eq!(failed["accepted_fifo_ordinal"], 1);
    assert_eq!(failed["cause"]["reason"], "actual_evidence_unknown");
    assert_eq!(failed["cause"]["physical_waves"], 1);
    assert_eq!(failed["cause"]["retained_waves"], 1);
    assert_eq!(failed["cause"]["lost_observations"], 0);
    assert_eq!(failed["cause"]["dispatch_outcome"], "Completed");
    assert_eq!(failed["cause"]["dispatch_unknown"], "GraphPath");
    assert_eq!(failed["cause"]["first_unknown_wave_reason"], "GraphPath");
    assert_eq!(failed["cause"]["unknown_wave_count"], 1);
}

#[test]
fn not_submitted_and_truncated_composite_calls_keep_distinct_original_failure_causes() {
    for physical_waves in [0_usize, 2] {
        let (live, actual, _) = fixture(256);
        let queue = sink(2, 256);
        let ticket = live.reserve(Some(1)).unwrap();
        let (call, clock) = begin(&actual, &queue);
        let mut call = call.with_live_ticket(Some(ticket));
        {
            let mut context = call.context().unwrap();
            for _ in 0..physical_waves {
                context.physical_wave(Ok(actual.clone()), Some(3));
                clock.set(6);
                context.terminal(ActualWaveOutcome::Completed, None);
            }
            context.finish_call(if physical_waves == 0 {
                ObservedCallOutcome::NotSubmitted
            } else {
                ObservedCallOutcome::Completed
            });
        }
        assert_eq!(call.finish(), CostCallDisposition::Queued);
        assert!(queue.pop_numbered_with_ticket().is_none());
        let audit = live.audit();
        assert!(audit.population.failed);
        assert_eq!((audit.population.issued, audit.population.retired), (1, 1));
        let failed = serde_json::to_value(audit.population.first_ticket_failure.unwrap()).unwrap();
        assert_eq!(failed["cause"]["physical_waves"], physical_waves);
        // The original fixture's recorder retains only one physical wave.
        // Keep physical attempts, retained records, and loss distinct.
        assert_eq!(failed["cause"]["retained_waves"], physical_waves.min(1));
        assert_eq!(
            failed["cause"]["lost_observations"],
            physical_waves.saturating_sub(1)
        );
        assert!(failed["cause"]["first_unknown_wave_reason"].is_null());
        assert_eq!(
            failed["cause"]["dispatch_outcome"],
            if physical_waves == 0 {
                "NotSubmitted"
            } else {
                "Completed"
            }
        );
        // This diagnostic does not invent an actual sample or permit the
        // present source6 protocol to omit its originally offered ticket.
        assert_eq!(audit.retained_waves, 0);
    }
}

#[test]
fn a_later_ticket_drop_cannot_overwrite_the_original_call_failure() {
    let (live, actual, _) = fixture(256);
    let queue = sink(2, 256);
    let first = live.reserve(Some(1)).unwrap();
    let later = live.reserve(Some(1)).unwrap();
    let (call, _) = begin(&actual, &queue);
    let mut call = call.with_live_ticket(Some(first));
    call.context()
        .unwrap()
        .finish_call(ObservedCallOutcome::NotSubmitted);
    call.finish();
    assert!(queue.pop_numbered_with_ticket().is_none());
    let before = serde_json::to_value(live.audit().population.first_ticket_failure).unwrap();
    drop(later);
    let audit = live.audit();
    assert_eq!(
        serde_json::to_value(audit.population.first_ticket_failure).unwrap(),
        before
    );
    assert_eq!((audit.population.issued, audit.population.retired), (2, 2));
    assert!(audit.population.failed && audit.population.closed);
}
