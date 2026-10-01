//! Real private recorder/settlement -> original FIFO -> pre-issued population.
//! No deserialized or independently constructed numerical receipt is accepted.
use super::*;
#[path = "live_windows/ticket_failure.rs"]
mod ticket_failure;
use crate::continuous_engine::inner::cost_observation::live_calibration::LiveCalibration;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredInputV2;

fn fixture(
    offers: usize,
) -> (
    Arc<LiveCalibration>,
    ActualWaveShape,
    Vec<HostCostFeaturesV1>,
) {
    let (actual, hosts, exact) = algorithm_parts_graph(&[8], 8, None);
    let selected = actual.statistical_evidence.as_ref().unwrap();
    let owner = StructuredInputV2::owner_for(
        &exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
    )
    .unwrap();
    (LiveCalibration::fixture(owner, offers), actual, hosts)
}

fn record(
    live: &LiveCalibration,
    queue: &Arc<BoundedCostSampleSink>,
    actual: ActualWaveShape,
    hosts: &[HostCostFeaturesV1],
    terminal_reason: Option<FinishReason>,
) {
    // Issue first, before the executor call or its actual outcome exists.
    let ticket = live.reserve(Some(1)).unwrap();
    let (call, clock) = begin_with_retained_capacity(&actual, queue, 64);
    let mut call = call
        .with_live_ticket(Some(ticket))
        .with_structured_capture(true);
    for (participant, host) in call.participants.iter_mut().zip(hosts) {
        participant.host_features = Some(*host);
    }
    execute(&mut call, &clock, actual.clone());
    let outcome = terminal_reason.map(|reason| {
        let mut t = terminal();
        t.finish_reason = reason;
        t
    });
    settle(&mut call, &clock, &actual.rows[0], 10, outcome);
    call.reject(CostCallRejection::Composite);
    clock.set(20);
    call.finish();
}

#[test]
fn live_window_natural_eos_and_stop_preserve_private_terminal_and_fifo() {
    for reason in [FinishReason::EOS, FinishReason::Stop] {
        let (live, actual, hosts) = fixture(1);
        let queue = sink(2, 256);
        record(&live, &queue, actual, &hosts, Some(reason));
        let (fifo, entry, ticket) = queue.pop_numbered_with_ticket().unwrap();
        let CostEvidenceEntry::StagesOnly { stages, .. } = &entry else {
            panic!("real terminal stays auxiliary to legacy training")
        };
        assert_eq!(
            stages.rows[0].terminal.as_ref().unwrap().finish_reason,
            reason
        );
        assert!(stages.structured_evidence.as_ref().unwrap().is_ok());
        // The older Length-only numerical scope remains closed.
        assert!(
            super::super::super::super::trainer::structured_v2::structured_discovery_input_v2(
                stages
            )
            .is_err()
        );
        live.consume(ticket.unwrap(), fifo, &entry);
        let a = live.audit();
        assert_eq!(
            (a.population.issued, a.population.retired, a.retained_waves),
            (1, 1, 1)
        );
        assert!(!a.population.failed);
        assert!(a.failure.is_none());
        assert_eq!(a.qualified_publications, 0);
    }
}

#[test]
fn live_window_final_fifo_capacity_loss_cannot_be_replaced() {
    let (live, actual, hosts) = fixture(2);
    let queue = sink(1, 256);
    record(&live, &queue, actual.clone(), &hosts, None);
    record(&live, &queue, actual, &hosts, None);
    assert_eq!(queue.stats().entries_dropped_capacity, 1);
    let (fifo, entry, ticket) = queue.pop_numbered_with_ticket().unwrap();
    live.consume(ticket.unwrap(), fifo, &entry);
    assert!(queue.pop_numbered_with_ticket().is_none());
    assert!(live.reserve(Some(30)).is_none());
    let a = live.audit();
    assert!(a.population.failed && a.population.closed);
    assert_eq!((a.population.issued, a.population.retired), (2, 2));
}

#[test]
fn live_window_repeated_frontier_is_not_reclassified_as_new_open_boundary() {
    let (live, actual, hosts) = fixture(2);
    let queue = sink(2, 256);
    for _ in 0..2 {
        record(&live, &queue, actual.clone(), &hosts, None);
        let (fifo, entry, ticket) = queue.pop_numbered_with_ticket().unwrap();
        live.consume(ticket.unwrap(), fifo, &entry);
    }
    let a = live.audit();
    assert_eq!(a.retained_waves, 1);
    assert_eq!(a.failure, Some("within_window_frontier_discontinuity"));
    assert!(a.population.failed);
}

#[test]
fn live_window_first_settlement_failure_keeps_domain_and_binding_reasons() {
    let (live, actual, hosts) = fixture(2);
    let queue = sink(2, 256);
    // Both tickets precede the first failure; the second original entry still
    // has to retire without replacing the first generation diagnostic.
    for _ in 0..2 {
        record(&live, &queue, actual.clone(), &hosts, None);
    }
    let (first_fifo, mut first, first_ticket) = queue.pop_numbered_with_ticket().unwrap();
    let CostEvidenceEntry::StagesOnly { stages, .. } = &mut first else {
        unreachable!()
    };
    Arc::make_mut(stages)
        .actual_shape
        .as_mut()
        .unwrap()
        .row_multiset_features = None;
    live.consume(first_ticket.unwrap(), first_fifo, &first);
    let first_diagnostic = live.audit().first_settlement_failure.unwrap();
    let encoded = serde_json::to_value(first_diagnostic).unwrap();
    assert_eq!(encoded["accepted_ordinal"], first_fifo);
    assert_eq!(encoded["model_error"], "InvalidSample");
    assert_eq!(encoded["host_content_rejection"], "domain_missing");
    assert_eq!(
        encoded["private_settlement"]["binding_rejected"],
        "host_identity_mismatch"
    );
    assert_eq!(encoded["row_multiset_features_present"], false);
    assert_eq!(encoded["legacy_rejection"], "composite");

    let (next_fifo, mut next, next_ticket) = queue.pop_numbered_with_ticket().unwrap();
    let CostEvidenceEntry::StagesOnly { stages, .. } = &mut next else {
        unreachable!()
    };
    Arc::make_mut(stages).structured_evidence =
        Some(Err(StructuredSettlementUnknown::InvalidClock));
    live.consume(next_ticket.unwrap(), next_fifo, &next);
    let audit = live.audit();
    assert_eq!(audit.first_settlement_failure, Some(first_diagnostic));
    assert_eq!(audit.retained_waves, 0);
    assert_eq!((audit.population.issued, audit.population.retired), (2, 2));
    assert!(audit.population.failed);
}

#[test]
fn live_window_settlement_diagnostic_preserves_private_and_selected_typed_errors() {
    use crate::continuous_engine::inner::cost_observation::trainer::host_content::statistical::{
        complete_structured_observation_v2, SettlementFailureDiagnostic,
    };
    use ferrum_scheduler::implementations::continuous::cost_model::statistical::model::ModelUnknown;

    let original = stages(&[8], None, 8);
    let mut private_failure = original.as_ref().clone();
    private_failure.structured_evidence = Some(Err(StructuredSettlementUnknown::InvalidClock));
    let private_entry = entry(Arc::new(private_failure));
    let error = complete_structured_observation_v2(&private_entry)
        .err()
        .unwrap();
    assert_eq!(error, ModelUnknown::InvalidSample);
    let diagnostic = SettlementFailureDiagnostic::capture(&private_entry, 7, error);
    let encoded = serde_json::to_value(diagnostic).unwrap();
    assert!(encoded["host_content_rejection"].is_null());
    assert_eq!(
        encoded["private_settlement"]["producer_rejected"],
        "invalid_clock"
    );
    assert_eq!(encoded["recipe"], "present");

    let mut missing_selected = original.as_ref().clone();
    missing_selected.statistical_evidence = None;
    let selected_entry = entry(Arc::new(missing_selected));
    let error = complete_structured_observation_v2(&selected_entry)
        .err()
        .unwrap();
    assert_eq!(
        error,
        ModelUnknown::Evidence(StatisticalEvidenceUnknown::MissingProducer)
    );
    let diagnostic = SettlementFailureDiagnostic::capture(&selected_entry, 8, error);
    assert_eq!(diagnostic.model_error, error);
    let encoded = serde_json::to_value(diagnostic).unwrap();
    assert_eq!(encoded["model_error"], "Evidence(MissingProducer)");
    assert_eq!(encoded["selected_evidence_present"], false);
    assert_eq!(encoded["recipe"], "missing");
}

#[test]
fn live_window_channel_publication_keeps_ticket_acceptance_before_visibility() {
    use std::time::Duration;
    let (live, actual, hosts) = fixture(1);
    let queue = sink(2, 256);
    let (sent, waiting) = std::sync::mpsc::channel();
    let (release, resume) = std::sync::mpsc::channel();
    queue.on_next_send(move || {
        sent.send(()).unwrap();
        resume.recv_timeout(Duration::from_secs(3)).unwrap();
    });
    let producer_live = Arc::clone(&live);
    let producer_queue = Arc::clone(&queue);
    let producer = std::thread::spawn(move || {
        record(
            &producer_live,
            &producer_queue,
            actual,
            &hosts,
            Some(FinishReason::EOS),
        );
    });
    waiting.recv_timeout(Duration::from_secs(3)).unwrap();
    assert!(queue.pop_numbered_with_ticket().is_none());
    assert_eq!(live.audit().population.issued, 1);
    assert_eq!(live.audit().population.retired, 0);
    release.send(()).unwrap();
    producer.join().unwrap();
    let (fifo, entry, ticket) = queue.pop_numbered_with_ticket().unwrap();
    assert_eq!(fifo, 1);
    live.consume(ticket.unwrap(), fifo, &entry);
    let audit = live.audit();
    assert_eq!(audit.population.retired, 1);
    assert_eq!(audit.retained_waves, 1);
    assert!(!audit.population.failed);
    assert!(audit.failure.is_none());
    assert!(!queue.stats().has_lost_samples());
}
