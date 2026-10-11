use super::*;
use ferrum_types::{
    ProfileEntrypoint, ProfileEventKind, ProfileStatus, OBSERVABILITY_PROFILE_SCHEMA_VERSION,
};

const CLOCK: u64 = 7;

fn owner(wave: u64, canonical: u32, caller: u32) -> CycleParticipantOwner {
    CycleParticipantOwner {
        sequence_authority: CycleAuthority {
            sparse_id: canonical + 1,
            generation: 1,
        },
        request_authority: CycleAuthority {
            sparse_id: canonical + 1,
            generation: 1,
        },
        frame_id: wave * 10 + u64::from(canonical) + 1,
        request_id: format!("00000000-0000-0000-0000-{:012}", canonical + 1),
        request_identity: format!("request.fixture.{}", canonical + 1),
        canonical_ordinal: canonical,
        caller_ordinal: caller,
        context: Some(CycleCallContext {
            clock_id: CLOCK,
            iteration: wave,
            call: 0,
            caller_ordinal: caller,
        }),
    }
}

fn wave(id: u64) -> CycleWaveWitness {
    CycleWaveWitness {
        clock_id: CLOCK,
        kind: "decode".to_owned(),
        batch_step_id: id,
        batch_invocation_id: id + 10,
        lane_id: 1,
        device_id: "device.fixture".to_owned(),
        runtime_implementation_fingerprint: "runtime.fixture".to_owned(),
        prepared_wave_fingerprint: format!("prepared.{id}"),
        // Original caller ordinals may have gaps and need not be canonical order.
        owners: vec![owner(id, 0, 3), owner(id, 1, 1)],
    }
}

fn wave_scope(wave: &CycleWaveWitness) -> CycleScope {
    CycleScope::Wave {
        clock_id: wave.clock_id,
        batch_step_id: wave.batch_step_id,
        batch_invocation_id: wave.batch_invocation_id,
        prepared_wave_fingerprint: wave.prepared_wave_fingerprint.clone(),
    }
}

fn submission_scope(wave: &CycleWaveWitness) -> CycleScope {
    CycleScope::Submission {
        clock_id: wave.clock_id,
        batch_step_id: wave.batch_step_id,
        batch_invocation_id: wave.batch_invocation_id,
        prepared_wave_fingerprint: wave.prepared_wave_fingerprint.clone(),
        slot_id: wave.batch_step_id + 20,
        submission_fingerprint: format!("submission.{}", wave.batch_step_id),
    }
}

fn participant_scope(wave: &CycleWaveWitness, canonical: usize) -> CycleScope {
    CycleScope::Participant {
        clock_id: wave.clock_id,
        slot_id: wave.batch_step_id + 20,
        submission_fingerprint: format!("submission.{}", wave.batch_step_id),
        owner: wave.owners[canonical].clone(),
    }
}

fn event(nanos: u64, stage: &str, scope: CycleScope) -> CycleEvent {
    CycleEvent {
        at: CycleTimestamp {
            clock_id: CLOCK,
            nanos,
        },
        stage: stage.to_owned(),
        scope,
        outcome: matches!(
            stage,
            "sample_end"
                | "text_decode_end"
                | "stream_send_end"
                | "request_complete_end"
                | "call_returned"
                | "compute_graph_launch_end"
        )
        .then(|| "success".to_owned()),
        wave_witness: None,
    }
}

fn append_wave(events: &mut Vec<CycleEvent>, id: u64, offset: u64) {
    let wave = wave(id);
    let context = wave.owners[0].context.expect("fixture call context");
    for (at, stage) in [
        (0, "loop_resumed"),
        (1, "iteration_lock_start"),
        (2, "iteration_lock_end"),
        (2, "admission_start"),
        (3, "admission_end"),
        (3, "schedule_start"),
        (4, "schedule_end"),
        (400, "iteration_end"),
    ] {
        events.push(event(
            offset + at,
            stage,
            CycleScope::Iteration { iteration: id },
        ));
    }
    events.push(event(offset, "call_started", CycleScope::Call { context }));
    events.push(event(
        offset + 5,
        "prepare_start",
        CycleScope::Call { context },
    ));
    events.push(event(offset + 9, "prepare_end", wave_scope(&wave)));
    let mut prepared = event(offset + 10, "wave_prepared", wave_scope(&wave));
    prepared.wave_witness = Some(wave.clone());
    events.push(prepared);
    for (at, stage) in [
        (15, "encode_start"),
        (20, "command_enqueue_start"),
        (30, "compute_graph_launch_start"),
        (35, "compute_graph_launch_end"),
        (80, "command_enqueue_end"),
        (90, "encode_end"),
    ] {
        events.push(event(offset + at, stage, wave_scope(&wave)));
    }
    for (at, stage) in [
        (95, "submission_bound"),
        (100, "completion_queued"),
        (110, "completion_worker_started"),
        (120, "terminal_readback_start"),
        (140, "terminal_readback_end"),
        (150, "completion_result_ready"),
        (170, "completion_await_resumed"),
        (180, "retirement_start"),
        (190, "retirement_end"),
        (200, "product_return"),
    ] {
        events.push(event(offset + at, stage, submission_scope(&wave)));
    }
    events.push(event(
        offset + 210,
        "call_returned",
        CycleScope::Call { context },
    ));
    for canonical in 0..2 {
        let skew = canonical as u64 * 20;
        for (at, stage) in [
            (215, "request_lock_start"),
            (218, "request_lock_end"),
            (220, "sample_start"),
            (260, "sample_end"),
            (265, "token_commit"),
            (270, "stop_check_start"),
            (275, "stop_check_end"),
            (276, "stream_update_start"),
            (277, "text_lock_start"),
            (279, "text_lock_end"),
            (280, "text_decode_start"),
            (290, "text_decode_end"),
            (310, "stream_send_start"),
            (315, "stream_send_end"),
            (316, "stream_yield_start"),
            (318, "stream_yield_resumed"),
            (340, "stream_update_end"),
        ] {
            events.push(event(
                offset + at + skew,
                stage,
                participant_scope(&wave, canonical),
            ));
        }
    }
}

fn capture() -> DecodeCycleCapture {
    let mut events = Vec::new();
    append_wave(&mut events, 1, 0);
    append_wave(&mut events, 2, 450);
    let count = events.len() as u64;
    DecodeCycleCapture {
        clock: CycleClock {
            id: CLOCK,
            wall_anchor_unix_nanos: 1_000_000_000,
            wall_anchor_max_error_nanos: 1,
        },
        capacity: 256,
        active_producers_at_close: 0,
        stats: CycleStats {
            attempted: count,
            accepted: count,
            ..CycleStats::default()
        },
        events,
    }
}

fn remove_stage(capture: &mut DecodeCycleCapture, stage: &str, wave: u64) {
    capture.events.retain(|event| {
        event.stage != stage
            || event
                .scope
                .wave_key()
                .is_none_or(|key| key.batch_step_id != wave)
    });
    capture.stats.attempted = capture.events.len() as u64;
    capture.stats.accepted = capture.events.len() as u64;
}

#[test]
fn record_arrival_order_does_not_change_physical_wave_analysis() {
    let mut capture = capture();
    let ordered = analyze_capture(&capture, Some(2));
    assert!(ordered.usable, "{:?}", ordered.issues);
    capture.events.reverse();
    let reversed = analyze_capture(&capture, Some(2));
    assert!(reversed.usable, "{:?}", reversed.issues);
    assert_eq!(
        serde_json::to_value(&ordered.cohorts).unwrap(),
        serde_json::to_value(&reversed.cohorts).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&ordered.phase_metrics).unwrap(),
        serde_json::to_value(&reversed.phase_metrics).unwrap()
    );
    assert_eq!(ordered.cohorts[0].width, 2);
    assert_eq!(ordered.cohorts[0].next_wave.unwrap().batch_step_id, 2);
    assert_eq!(
        ordered.cohorts[0].boundaries["next_compute_graph_launch_start"],
        Some(480)
    );
    assert!(ordered.cohorts[0].missing_edges.is_empty());
    assert_eq!(ordered.cohorts[0].owner_coverage.expected, 2);
    assert_eq!(ordered.cohorts[0].owner_coverage.complete, 2);
    let chain = &ordered.chain_metrics["terminal_readback_end->next_host_graph_launch"];
    assert_eq!(chain.count, 1);
    assert_eq!(chain.median_ns, Some(340.0));
}

#[test]
fn overlapping_participant_spans_are_unioned_and_not_summed() {
    let analysis = analyze_capture(&capture(), Some(2));
    let sample = &analysis.phase_metrics["sample"];
    assert_eq!(sample.count, 4);
    assert_eq!(sample.median_ns, Some(40.0));
    assert_eq!(sample.p99_ns, Some(40.0));
    // Each wave contains [220,260] and [240,280], a 60 ns union.
    assert_eq!(sample.union_ns, 120);
    let sum_of_phase_unions: u64 = analysis.phase_metrics.values().map(|x| x.union_ns).sum();
    // encode encloses command enqueue and graph launch: nested spans are one
    // observed host interval when the overall clock union is reported.
    assert!(analysis.observed_interval_union_ns < sum_of_phase_unions);
}

#[test]
fn equal_timestamps_can_report_an_observed_zero_interval_in_either_arrival_order() {
    let mut capture = capture();
    capture.events.retain(|event| {
        !matches!(event.stage.as_str(), "sample_start" | "sample_end")
            || matches!(&event.scope, CycleScope::Participant { owner, .. } if owner.frame_id == 11)
    });
    for event in &mut capture.events {
        if event.stage == "sample_end" {
            event.at.nanos = 220;
        }
    }
    capture.stats.attempted = capture.events.len() as u64;
    capture.stats.accepted = capture.events.len() as u64;
    for _ in 0..2 {
        let analysis = analyze_capture(&capture, Some(2));
        assert!(analysis.usable, "{:?}", analysis.issues);
        let sample = &analysis.phase_metrics["sample"];
        assert_eq!(sample.count, 1);
        assert_eq!(sample.median_ns, Some(0.0));
        assert_eq!(sample.p99_ns, Some(0.0));
        assert_eq!(sample.union_ns, 0);
        assert!(!analysis.unmatched_boundaries.contains_key("sample"));
        capture.events.reverse();
    }
}

#[test]
fn missing_completion_boundary_is_explicit_and_never_zero_duration() {
    let mut capture = capture();
    remove_stage(&mut capture, "completion_result_ready", 1);
    let analysis = analyze_capture(&capture, Some(2));
    assert!(analysis.collector_complete);
    assert!(analysis.identity_valid);
    assert_eq!(
        analysis.cohorts[0].boundaries["completion_result_ready"],
        None
    );
    assert!(analysis.cohorts[0]
        .missing_edges
        .iter()
        .any(|edge| edge == "completion_result_ready"));
    let resume = &analysis.phase_metrics["completion_await_resume"];
    assert_eq!(resume.count, 1, "only the other complete wave contributes");
    assert_eq!(resume.median_ns, Some(20.0));
    assert!(!analysis.unmatched_boundaries.is_empty());
}

fn is_first_owner(event: &CycleEvent) -> bool {
    matches!(&event.scope, CycleScope::Participant { owner, .. } if owner.frame_id == 11)
}

fn sync_counts(capture: &mut DecodeCycleCapture) {
    capture.stats.attempted = capture.events.len() as u64;
    capture.stats.accepted = capture.events.len() as u64;
}

fn assert_no_chain_samples(analysis: &DecodeCycleAnalysis) {
    assert!(analysis
        .cohorts
        .iter()
        .all(|cohort| cohort.classification != "complete_decode_to_decode"));
    assert!(analysis.chain_metrics.values().all(|metric| {
        metric.count == 0 && metric.median_ns.is_none() && metric.p99_ns.is_none()
    }));
}

#[test]
fn next_wave_requires_a_bound_submission_before_a_complete_chain_is_qualified() {
    let mut capture = capture();
    remove_stage(&mut capture, "submission_bound", 2);
    let analysis = analyze_capture(&capture, Some(2));
    assert!(analysis.cohorts[0]
        .missing_edges
        .contains(&"next_submission_bound".to_owned()));
    assert_no_chain_samples(&analysis);
}

#[test]
fn every_owner_needs_observed_sampling_and_token_commit_boundaries() {
    for stage in ["sample_start", "token_commit"] {
        let mut capture = capture();
        capture
            .events
            .retain(|event| event.stage != stage || !is_first_owner(event));
        sync_counts(&mut capture);
        let analysis = analyze_capture(&capture, Some(2));
        assert!(analysis.collector_complete);
        assert!(analysis.identity_valid);
        assert_eq!(analysis.cohorts[0].classification, "incomplete_edges");
        assert!(analysis.cohorts[0]
            .missing_edges
            .contains(&format!("owner.0.{stage}")));
        assert_no_chain_samples(&analysis);
    }
}

#[test]
fn failed_calls_participants_and_next_launches_disqualify_the_whole_window() {
    for (stage, nanos) in [
        ("call_returned", 210),
        ("sample_end", 260),
        ("stream_send_end", 315),
        ("compute_graph_launch_end", 485),
        ("call_returned", 660),
    ] {
        let mut capture = capture();
        let failed = capture
            .events
            .iter_mut()
            .find(|event| event.stage == stage && event.at.nanos == nanos)
            .expect("fixture failure boundary");
        failed.outcome = Some("failed".to_owned());
        let analysis = analyze_capture(&capture, Some(2));
        assert_eq!(
            analysis.cohorts[0].classification, "failed_window",
            "failure at {stage} ({nanos} ns)"
        );
        assert_no_chain_samples(&analysis);
    }
}

#[test]
fn missing_declared_outcomes_are_unknown_and_never_implicit_success() {
    for (stage, nanos) in [
        ("call_returned", 210),
        ("sample_end", 260),
        ("text_decode_end", 290),
        ("stream_send_end", 315),
        ("request_complete_end", 350),
        ("compute_graph_launch_end", 485),
    ] {
        let mut capture = capture();
        if stage == "request_complete_end" {
            let wave = wave(1);
            capture.events.push(event(
                342,
                "request_complete_start",
                participant_scope(&wave, 0),
            ));
            capture.events.push(event(
                350,
                "request_complete_end",
                participant_scope(&wave, 0),
            ));
            sync_counts(&mut capture);
        }
        let unknown = capture
            .events
            .iter_mut()
            .find(|event| event.stage == stage && event.at.nanos == nanos)
            .expect("fixture outcome boundary");
        unknown.outcome = None;
        let analysis = analyze_capture(&capture, Some(2));
        assert_eq!(
            analysis.cohorts[0].classification, "incomplete_edges",
            "missing outcome at {stage} ({nanos} ns)"
        );
        assert_no_chain_samples(&analysis);
    }
}

#[test]
fn failure_stage_without_an_outcome_remains_failed_and_discloses_unknown_metadata() {
    let mut capture = capture();
    let wave = wave(1);
    capture
        .events
        .push(event(330, "failure_observed", participant_scope(&wave, 0)));
    sync_counts(&mut capture);
    let analysis = analyze_capture(&capture, Some(2));
    assert_eq!(analysis.cohorts[0].classification, "failed_window");
    assert!(analysis.cohorts[0]
        .missing_edges
        .contains(&"owner.0.failure_observed_outcome".to_owned()));
    assert_no_chain_samples(&analysis);
}

#[test]
fn output_children_need_the_update_wrapper_and_successful_send_needs_a_yield() {
    for missing_stages in [
        ["stream_update_start", "stream_update_end"],
        ["stream_yield_start", "stream_yield_resumed"],
    ] {
        let mut capture = capture();
        capture.events.retain(|event| {
            !is_first_owner(event) || !missing_stages.contains(&event.stage.as_str())
        });
        sync_counts(&mut capture);
        let analysis = analyze_capture(&capture, Some(2));
        assert_eq!(analysis.cohorts[0].classification, "incomplete_edges");
        assert_no_chain_samples(&analysis);
    }
}

#[test]
fn an_observed_stream_yield_requires_an_observed_successful_send() {
    let mut capture = capture();
    capture.events.retain(|event| {
        !is_first_owner(event)
            || !matches!(
                event.stage.as_str(),
                "stream_send_start" | "stream_send_end"
            )
    });
    sync_counts(&mut capture);
    let analysis = analyze_capture(&capture, Some(2));
    assert_eq!(analysis.cohorts[0].classification, "incomplete_edges");
    assert_no_chain_samples(&analysis);
}

#[test]
fn conditional_request_completion_follows_the_finished_stream_update() {
    let mut capture = capture();
    let wave = wave(1);
    for canonical in 0..2 {
        let skew = canonical as u64 * 20;
        capture.events.push(event(
            342 + skew,
            "request_complete_start",
            participant_scope(&wave, canonical),
        ));
        capture.events.push(event(
            350 + skew,
            "request_complete_end",
            participant_scope(&wave, canonical),
        ));
    }
    sync_counts(&mut capture);
    let healthy = analyze_capture(&capture, Some(2));
    assert_eq!(
        healthy.cohorts[0].classification,
        "complete_decode_to_decode"
    );
    assert_eq!(healthy.cohorts[0].owner_coverage.complete, 2);
    assert_eq!(
        healthy.chain_metrics["terminal_readback_end->next_host_graph_launch"].count,
        1
    );

    capture
        .events
        .iter_mut()
        .find(|event| event.stage == "request_complete_start" && is_first_owner(event))
        .expect("first owner completion start")
        .at
        .nanos = 339;
    let premature = analyze_capture(&capture, Some(2));
    assert_eq!(premature.cohorts[0].classification, "incomplete_edges");
    assert_no_chain_samples(&premature);
}

#[test]
fn absent_optional_output_branches_are_valid_but_partial_branches_are_incomplete() {
    let mut absent = capture();
    absent
        .events
        .retain(|event| !event.stage.starts_with("text_") && !event.stage.starts_with("stream_"));
    sync_counts(&mut absent);
    let analysis = analyze_capture(&absent, Some(2));
    assert!(analysis.usable, "{:?}", analysis.issues);
    assert_eq!(
        analysis.cohorts[0].classification,
        "complete_decode_to_decode"
    );
    assert!(analysis.cohorts[0].missing_edges.is_empty());
    assert_eq!(
        analysis.chain_metrics["terminal_readback_end->next_host_graph_launch"].count,
        1
    );

    let mut partial = capture();
    partial
        .events
        .retain(|event| event.stage != "text_decode_end" || !is_first_owner(event));
    sync_counts(&mut partial);
    let analysis = analyze_capture(&partial, Some(2));
    assert_eq!(analysis.cohorts[0].classification, "incomplete_edges");
    assert!(analysis.cohorts[0]
        .missing_edges
        .contains(&"owner.0.text_decode_end".to_owned()));
    assert_no_chain_samples(&analysis);
}

#[test]
fn collector_drops_and_live_producers_disqualify_the_capture() {
    let faults: [fn(&mut DecodeCycleCapture); 8] = [
        |capture| capture.stats.busy = 1,
        |capture| capture.stats.full = 1,
        |capture| capture.stats.closed = 1,
        |capture| capture.stats.poisoned = 1,
        |capture| capture.stats.invalid_metadata = 1,
        |capture| capture.stats.outstanding_producers = 1,
        |capture| capture.active_producers_at_close = 1,
        |capture| capture.stats.accepted -= 1,
    ];
    for fault in faults {
        let mut capture = capture();
        fault(&mut capture);
        let analysis = analyze_capture(&capture, Some(2));
        assert!(!analysis.collector_complete);
        assert!(!analysis.usable);
        assert!(!analysis.issues.is_empty());
        assert_no_chain_samples(&analysis);
    }
}

#[test]
fn foreign_clock_and_prepared_fingerprint_are_not_joined() {
    for corrupt_clock in [false, true] {
        let mut capture = capture();
        let event = capture
            .events
            .iter_mut()
            .find(|event| event.stage == "completion_result_ready")
            .unwrap();
        if corrupt_clock {
            event.at.clock_id += 1;
        } else if let CycleScope::Submission {
            prepared_wave_fingerprint,
            ..
        } = &mut event.scope
        {
            *prepared_wave_fingerprint = "foreign.prepared".to_owned();
        }
        let analysis = analyze_capture(&capture, Some(2));
        assert!(!analysis.identity_valid);
        assert!(!analysis.usable);
        assert!(!analysis.issues.is_empty());
        assert_no_chain_samples(&analysis);
    }
}

#[test]
fn participant_scope_must_match_the_declared_frame_and_caller() {
    for change_frame in [false, true] {
        let mut capture = capture();
        let event = capture
            .events
            .iter_mut()
            .find(|event| event.stage == "sample_start")
            .unwrap();
        if let CycleScope::Participant { owner, .. } = &mut event.scope {
            if change_frame {
                owner.frame_id += 100;
            } else {
                owner.caller_ordinal += 10;
            }
        }
        let analysis = analyze_capture(&capture, Some(2));
        assert!(!analysis.identity_valid);
        assert!(!analysis.usable);
        assert_no_chain_samples(&analysis);
    }
}

#[test]
fn reaper_local_slots_are_disambiguated_by_submission_fingerprint() {
    let mut capture = capture();
    for event in &mut capture.events {
        if event.at.nanos < 450 {
            continue;
        }
        match &mut event.scope {
            CycleScope::Submission { slot_id, .. } | CycleScope::Participant { slot_id, .. } => {
                *slot_id = 21
            }
            _ => {}
        }
    }
    let analysis = analyze_capture(&capture, Some(2));
    assert!(analysis.identity_valid, "{:?}", analysis.issues);
    assert!(analysis.usable);
    assert_eq!(analysis.phase_metrics["sample"].count, 4);

    // The same complete submission tuple cannot declare a second physical wave.
    for event in &mut capture.events {
        if event.at.nanos < 450 {
            continue;
        }
        match &mut event.scope {
            CycleScope::Submission {
                submission_fingerprint,
                ..
            }
            | CycleScope::Participant {
                submission_fingerprint,
                ..
            } => *submission_fingerprint = "submission.1".to_owned(),
            _ => {}
        }
    }
    let analysis = analyze_capture(&capture, Some(2));
    assert!(!analysis.identity_valid);
    assert!(!analysis.usable);
    assert_no_chain_samples(&analysis);
}

#[test]
fn physical_width_filter_does_not_turn_partial_waves_into_full_cohorts() {
    let analysis = analyze_capture(&capture(), Some(4));
    assert!(analysis.collector_complete);
    assert!(analysis.identity_valid);
    assert!(analysis.cohorts.is_empty());
    assert!(analysis.chain_metrics.values().all(|metric| {
        metric.count == 0 && metric.median_ns.is_none() && metric.p99_ns.is_none()
    }));
}

#[test]
fn natural_tail_has_no_invented_next_launch_or_zero_duration_sample() {
    let mut capture = capture();
    capture.events.retain(|event| event.at.nanos < 450);
    capture.stats.attempted = capture.events.len() as u64;
    capture.stats.accepted = capture.events.len() as u64;
    let analysis = analyze_capture(&capture, Some(2));
    assert!(analysis.usable, "{:?}", analysis.issues);
    assert_eq!(analysis.cohorts.len(), 1);
    assert_eq!(analysis.cohorts[0].classification, "incomplete_tail");
    assert_eq!(
        analysis.cohorts[0].boundaries["next_compute_graph_launch_start"],
        None
    );
    assert!(analysis
        .chain_metrics
        .values()
        .all(|metric| metric.count == 0 && metric.median_ns.is_none() && metric.p99_ns.is_none()));
}

fn envelope(capture: &DecodeCycleCapture, declared_complete: bool) -> FerrumProfileEvent {
    FerrumProfileEvent {
        schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
        ts_unix_nanos: 1_000_000_000,
        event_id: "fixture.capture".to_owned(),
        request_id: "engine".to_owned(),
        correlation_id: Some("decode-cycle".to_owned()),
        entrypoint: ProfileEntrypoint::Serve,
        backend: "fixture".to_owned(),
        runtime_preset_hash: "fixture.preset".to_owned(),
        phase: "engine.decode_cycle_capture".to_owned(),
        event_kind: ProfileEventKind::Instant,
        timestamp: "1970-01-01T00:00:01Z".parse().unwrap(),
        status: ProfileStatus::DiagnosticOnly,
        model: None,
        duration_us: None,
        memory: None,
        resource: None,
        error: None,
        replay: None,
        shape: BTreeMap::from([
            ("collected_events".to_owned(), capture.events.len().into()),
            ("event_capacity".to_owned(), capture.capacity.into()),
        ]),
        backend_detail: None,
        attributes: BTreeMap::from([
            ("clock_source".to_owned(), "rust_std_instant".into()),
            ("capture".to_owned(), serde_json::to_value(capture).unwrap()),
            ("collector_complete".to_owned(), declared_complete.into()),
        ]),
    }
}

#[test]
fn profile_parser_validates_the_envelope_and_independent_coverage() {
    let capture = capture();
    let event = envelope(&capture, true);
    let jsonl = serde_json::to_string(&event).unwrap();
    let parsed = analyze_profile_jsonl(&format!("\n{jsonl}\n"), Some(2)).unwrap();
    assert_eq!(parsed.len(), 1);
    assert!(parsed[0].usable);

    let misleading = serde_json::to_string(&envelope(&capture, false)).unwrap();
    let parsed = analyze_profile_jsonl(&misleading, Some(2)).unwrap();
    assert!(parsed[0].collector_complete);
    assert!(!parsed[0].usable);
    assert_no_chain_samples(&parsed[0]);

    let mut invalid = event;
    invalid.shape.clear();
    assert!(analyze_profile_jsonl(&serde_json::to_string(&invalid).unwrap(), Some(2)).is_err());
    assert!(analyze_profile_jsonl(&jsonl[..jsonl.len() - 1], Some(2)).is_err());
    assert!(analyze_profile_jsonl("", Some(2)).is_err());
}
