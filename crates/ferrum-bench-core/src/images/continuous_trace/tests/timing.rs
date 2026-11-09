use super::*;

const REQUEST: &str = "00000000-0000-0000-0000-000000000001";
const ATTEMPT: &str = "00000000-0000-0000-0000-000000000002";
fn correlation(request: &str, attempt: &str) -> Value {
    json!({"request_id":request,"client_attempt_id":attempt})
}
fn push(events: &mut Vec<(u64, Value)>, at: u64, event: Value) -> u64 {
    events.push((at, event));
    events.len() as u64
}
fn artifact_two(events: &[(u64, Value)], modify: impl FnOnce(&mut Value)) -> Vec<u8> {
    let mut out = Vec::new();
    let mut observed = 0u64;
    for (index, (at, event)) in events.iter().enumerate() {
        observed = observed.max(*at);
        serde_json::to_writer(
            &mut out,
            &json!({"schema_version":2,"session_uuid":SESSION,"sequence":index+1,
            "monotonic_ns":observed,"event_ns":at,"event":event}),
        )
        .unwrap();
        out.push(b'\n');
    }
    let mut footer = json!({"schema_version":2,"session_uuid":SESSION,"footer":{"sealed":true,"truncated":false,"recorded":events.len(),"dropped":0,
        "used_bytes":out.len(),"maximum_bytes":1<<20,"backing_bytes":1<<20}});
    modify(&mut footer);
    serde_json::to_writer(&mut out, &footer).unwrap();
    out.push(b'\n');
    out
}
fn history() -> Vec<(u64, Value)> {
    let mut e = vec![(0, header())];
    push(
        &mut e,
        10,
        json!({"kind":"request_queued","correlation":correlation(REQUEST, ATTEMPT)}),
    );
    push(
        &mut e,
        20,
        json!({"kind":"request_dequeued","request_id":REQUEST}),
    );
    push(
        &mut e,
        30,
        json!({"kind":"initialization_started","request_id":REQUEST}),
    );
    let initialization = push(
        &mut e,
        30,
        json!({"kind":"phase_started","request_id":REQUEST,"request":null,"phase":"initialization"}),
    );
    push(
        &mut e,
        35,
        json!({"kind":"request_linked","request_id":REQUEST,"request":"a"}),
    );
    push(&mut e, 40, initialized("a"));
    push(
        &mut e,
        45,
        json!({"kind":"phase_finished","start_sequence":initialization,"outcome":"completed"}),
    );
    push(&mut e, 50, publish(10, &["a"]));
    let start = push(
        &mut e,
        60,
        json!({"kind":"packed_step_started","cohort_job":10,"plan_hash":"plan.10","program_fingerprint":"program.original","members":[{"request":"a","step_index":0}]}),
    );
    let mut committed = commit(10, json!([0., 0., 0.999]), vec![member("a", 0, 0, 0)]);
    committed["start_sequence"] = json!(start);
    push(&mut e, 100, committed);
    let start = push(
        &mut e,
        150,
        json!({"kind":"packed_step_started","cohort_job":10,"plan_hash":"plan.10","program_fingerprint":"program.original","members":[{"request":"a","step_index":1}]}),
    );
    let mut committed = commit(10, json!([0.5, 0.5, 0.999]), vec![member("a", 0, 1, 0)]);
    committed["start_sequence"] = json!(start);
    push(&mut e, 200, committed);
    push(&mut e, 205, exited("a", 2, "completed"));
    let decoder = push(
        &mut e,
        210,
        json!({"kind":"phase_started","request_id":REQUEST,"request":"a","phase":"decoder"}),
    );
    push(
        &mut e,
        215,
        json!({"kind":"phase_finished","start_sequence":decoder,"outcome":"completed"}),
    );
    push(
        &mut e,
        220,
        json!({"kind":"request_service_finished","request_id":REQUEST,"outcome":"completed"}),
    );
    e
}

#[test]
fn main_vision_binding_keeps_cancelled_producer_identity_and_old_trace_compatibility() {
    let old = analyze(&artifact_two(&history(), |_| {})).unwrap();
    assert!(old
        .timing
        .phases
        .iter()
        .all(|p| p.main_vision_binding.is_none()));
    let mut events = history();
    let start = events
        .iter()
        .position(|(_, e)| e["phase"] == "decoder")
        .unwrap();
    events[start].1["phase"] = json!("conditioning_vision");
    let binding = json!({"cohort_first_job":4,"producer_job":4,
        "producer_request":"cancelled-before-request-linked","reused":true,"retained_output_bytes":4096});
    events[start + 1].1["main_vision_binding"] = binding.clone();
    let analysis = analyze(&artifact_two(&events, |_| {})).unwrap();
    let actual = analysis
        .timing
        .phases
        .last()
        .unwrap()
        .main_vision_binding
        .as_ref()
        .unwrap();
    assert!(actual.reused);
    assert_eq!(actual.producer_request, "cancelled-before-request-linked");
    // A producer records its own original identity; a cache hit need not
    // invent a successful terminal image or later service link for its source.
    events[start + 1].1["main_vision_binding"]["reused"] = json!(false);
    assert!(analyze(&artifact_two(&events, |_| {})).is_err());
    events[start + 1].1["main_vision_binding"]["producer_request"] = json!("a");
    assert!(analyze(&artifact_two(&events, |_| {})).is_ok());
    for (field, value) in [
        ("retained_output_bytes", json!(0)),
        ("producer_request", json!("")),
        ("unknown", json!(true)),
        ("reused", json!("yes")),
    ] {
        let mut wrong = events.clone();
        wrong[start + 1].1["main_vision_binding"][field] = value;
        assert!(analyze(&artifact_two(&wrong, |_| {})).is_err());
    }
    events[start].1["phase"] = json!("decoder");
    assert!(analyze(&artifact_two(&events, |_| {})).is_err());
}

#[test]
fn schema_two_uses_actual_queue_and_committed_step_endpoints_and_schema_one_has_no_timing() {
    let analysis = analyze(&artifact_two(&history(), |_| {})).unwrap();
    assert_eq!(analysis.schema_version, 2);
    assert_eq!(analysis.status, ImageStepEvidenceStatus::Observed);
    assert_eq!(analysis.timing.status, ImageStepEvidenceStatus::Observed);
    let request = &analysis.timing.requests[0];
    assert_eq!(request.request.as_deref(), Some("a"));
    assert_eq!(request.queue_total_ns, Some(10));
    assert_eq!(request.accepted_to_initialization_ns, Some(20));
    assert_eq!(analysis.timing.steps[0].interval.duration_ns, 40);
    assert_eq!(analysis.timing.steps[0].members[0].wait_ns, None);
    assert_eq!(analysis.timing.steps[1].members[0].wait_ns, Some(50));
    assert_eq!(analysis.timing.phases[1].interval.duration_ns, 5);
    assert_eq!(analysis.timing.phases[1].ready_ns, Some(205));
    assert_eq!(analysis.timing.phases[1].wait_ns, Some(5));
    let old = analyze(&artifact(&events(), |_| {})).unwrap();
    assert_eq!(old.status, ImageStepEvidenceStatus::Observed);
    assert_eq!(old.timing.status, ImageStepEvidenceStatus::NotEvaluated);
    assert!(old
        .timing
        .unavailable_reasons
        .contains(&ImageTraceTimingUnavailable::SchemaOne));
    assert!(old.timing.steps.is_empty());
}

#[test]
fn distinct_visual_phases_preserve_intervals_outcomes_and_strict_names() {
    for phase in ["conditioning_vision", "prompt_expansion_vision"] {
        for outcome in ["completed", "failed"] {
            let mut events = history();
            let start = events
                .iter()
                .position(|(_, event)| event["phase"] == "decoder")
                .unwrap();
            events[start].1["phase"] = json!(phase);
            events[start + 1].1["outcome"] = json!(outcome);
            let parsed = analyze(&artifact_two(&events, |_| {})).unwrap();
            let observed = parsed.timing.phases.last().unwrap();
            assert_eq!(observed.phase, phase);
            assert_eq!(observed.outcome, outcome);
            assert_eq!(observed.interval.duration_ns, 5);
            assert_eq!(observed.ready_ns, None);
            assert_eq!(observed.wait_ns, None);
            events[start].1["phase"] = json!(format!("{phase}_unknown"));
            assert!(analyze(&artifact_two(&events, |_| {})).is_err());
        }
    }
}

#[test]
fn nested_visual_phases_keep_original_identity_and_do_not_extend_parent_intervals() {
    for (parent, child, model_identity) in [
        ("conditioning", "conditioning_vision", true),
        ("prompt_expansion", "prompt_expansion_vision", false),
    ] {
        for outcome in ["completed", "failed"] {
            let mut e = vec![(0, header())];
            push(
                &mut e,
                10,
                json!({"kind":"request_queued","correlation":correlation(REQUEST, ATTEMPT)}),
            );
            push(
                &mut e,
                20,
                json!({"kind":"request_dequeued","request_id":REQUEST}),
            );
            push(
                &mut e,
                30,
                json!({"kind":"initialization_started","request_id":REQUEST}),
            );
            let initialization = push(
                &mut e,
                30,
                json!({"kind":"phase_started","request_id":REQUEST,"request":null,"phase":"initialization"}),
            );
            if model_identity {
                push(
                    &mut e,
                    30,
                    json!({"kind":"request_linked","request_id":REQUEST,"request":"a"}),
                );
            }
            // Main conditioning already owns its model identity. Expansion
            // still owns the original service correlation, before that link.
            let request_id = (!model_identity).then_some(REQUEST);
            let request = model_identity.then_some("a");
            let parent_start = push(
                &mut e,
                31,
                json!({"kind":"phase_started","request_id":request_id,"request":request,"phase":parent}),
            );
            let child_start = push(
                &mut e,
                32,
                json!({"kind":"phase_started","request_id":request_id,"request":request,"phase":child}),
            );
            push(
                &mut e,
                38,
                json!({"kind":"phase_finished","start_sequence":child_start,"outcome":outcome}),
            );
            push(
                &mut e,
                39,
                json!({"kind":"phase_finished","start_sequence":parent_start,"outcome":outcome}),
            );
            push(
                &mut e,
                45,
                json!({"kind":"phase_finished","start_sequence":initialization,"outcome":"failed"}),
            );
            push(
                &mut e,
                50,
                json!({"kind":"request_service_finished","request_id":REQUEST,"outcome":"failed"}),
            );
            let parsed = analyze(&artifact_two(&e, |_| {})).unwrap();
            assert_eq!(parsed.timing.status, ImageStepEvidenceStatus::Observed);
            let phase = |name| {
                parsed
                    .timing
                    .phases
                    .iter()
                    .find(|p| p.phase == name)
                    .unwrap()
            };
            let inner = phase(child);
            let outer = phase(parent);
            assert_eq!(inner.interval.duration_ns, 6);
            assert_eq!(outer.interval.duration_ns, 8);
            assert_eq!(phase("initialization").interval.duration_ns, 15);
            assert!(inner.interval.start_ns > outer.interval.start_ns);
            assert!(inner.interval.end_ns < outer.interval.end_ns);
            assert_eq!(inner.request_id.as_deref(), request_id);
            assert_eq!(inner.request.as_deref(), request);
            assert_eq!(inner.outcome, outcome);
            assert_eq!(inner.ready_ns, None);
            assert_eq!(inner.wait_ns, None);
            assert_eq!(parsed.timing.requests[0].queue_total_ns, Some(10));
            assert_eq!(parsed.timing.requests[0].service_finished_ns, Some(50));
        }
    }
}

#[test]
fn requeue_and_cancel_close_real_queue_intervals_and_never_queued_rejection_has_none() {
    let mut e = vec![(0, header())];
    push(
        &mut e,
        2,
        json!({"kind":"request_queued","correlation":correlation(REQUEST,ATTEMPT)}),
    );
    push(
        &mut e,
        5,
        json!({"kind":"request_dequeued","request_id":REQUEST}),
    );
    push(
        &mut e,
        7,
        json!({"kind":"request_requeued","request_id":REQUEST}),
    );
    push(
        &mut e,
        11,
        json!({"kind":"request_service_finished","request_id":REQUEST,"outcome":"cancelled"}),
    );
    push(
        &mut e,
        12,
        json!({"kind":"request_submission_rejected","correlation":correlation("00000000-0000-0000-0000-000000000003",ATTEMPT),"outcome":"rejected"}),
    );
    let a = analyze(&artifact_two(&e, |_| {})).unwrap();
    assert_eq!(a.timing.status, ImageStepEvidenceStatus::Observed);
    assert_eq!(a.timing.requests[0].queue_total_ns, Some(7));
    assert_eq!(a.timing.requests[0].queue_intervals.len(), 2);
    assert!(a.timing.requests[1].never_queued);
    assert_eq!(a.timing.requests[1].queue_total_ns, None);
    assert_eq!(a.timing.requests[1].accepted_ns, None);
}

#[test]
fn missing_endpoints_or_incomplete_prefix_never_acquire_timing_qualification() {
    let mut e = history();
    for (_, event) in &mut e {
        if event["kind"] == "packed_step_committed" {
            event.as_object_mut().unwrap().remove("start_sequence");
        }
    }
    let a = analyze(&artifact_two(&e, |_| {})).unwrap();
    assert_eq!(a.timing.status, ImageStepEvidenceStatus::NotEvaluated);
    assert!(a.timing.steps.is_empty());
    assert!(a
        .timing
        .unavailable_reasons
        .contains(&ImageTraceTimingUnavailable::MissingEndpoints));
    let a = analyze(&artifact_two(&history(), |footer| {
        footer["footer"]["truncated"] = json!(true)
    }))
    .unwrap();
    assert_eq!(a.timing.status, ImageStepEvidenceStatus::NotEvaluated);
    assert_eq!(a.timing.steps.len(), 2); // Diagnostic samples remain available, not qualified.
    let e = vec![
        (0, header()),
        (
            1,
            json!({"kind":"phase_started","request":"a","request_id":null,"phase":"decoder"}),
        ),
    ];
    let a = analyze(&artifact_two(&e, |_| {})).unwrap();
    assert!(a.timing.phases.is_empty());
    assert_eq!(a.timing.status, ImageStepEvidenceStatus::NotEvaluated);
}

#[test]
fn backwards_span_wrong_membership_and_duplicate_correlation_are_rejected() {
    let mut e = history();
    let commit = e
        .iter_mut()
        .find(|(_, event)| event["kind"] == "packed_step_committed")
        .unwrap();
    commit.0 = 59;
    assert!(analyze(&artifact_two(&e, |_| {}))
        .unwrap_err()
        .reason
        .contains("backwards"));
    let mut e = history();
    let start = e
        .iter_mut()
        .find(|(_, event)| event["kind"] == "packed_step_started")
        .unwrap();
    start.1["members"][0]["step_index"] = json!(1);
    assert!(analyze(&artifact_two(&e, |_| {}))
        .unwrap_err()
        .reason
        .contains("membership/progress"));
    let e = vec![
        (0, header()),
        (
            1,
            json!({"kind":"request_queued","correlation":correlation(REQUEST,ATTEMPT)}),
        ),
        (
            2,
            json!({"kind":"request_queued","correlation":correlation(REQUEST,ATTEMPT)}),
        ),
    ];
    assert!(analyze(&artifact_two(&e, |_| {}))
        .unwrap_err()
        .reason
        .contains("repeats an identity"));
}

#[test]
fn malformed_clock_and_overflow_are_not_zero_duration_samples() {
    let original = artifact_two(&history(), |_| {});
    let mut lines = original
        .split(|b| *b == b'\n')
        .filter(|b| !b.is_empty())
        .map(|line| serde_json::from_slice::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    lines[2]["monotonic_ns"] = json!(0);
    let mut bytes = Vec::new();
    for line in lines {
        serde_json::to_writer(&mut bytes, &line).unwrap();
        bytes.push(b'\n');
    }
    assert!(analyze(&bytes)
        .unwrap_err()
        .reason
        .contains("clock regresses"));
    let overflow = String::from_utf8(original).unwrap().replacen(
        "\"event_ns\":0",
        "\"event_ns\":18446744073709551616",
        1,
    );
    assert!(analyze(overflow.as_bytes()).is_err());
}

fn attempt() -> ImageTraceBenchAttempt {
    ImageTraceBenchAttempt {
        endpoint_index: 0,
        sample_index: 7,
        repetition: 0,
        phase: "measured".into(),
        client_attempt_id: Some(ATTEMPT.into()),
        server_request_id: None,
        trace_session_uuid: None,
    }
}
#[test]
fn missing_response_correlation_joins_by_unique_attempt_and_duplicates_are_explicit() {
    let a = analyze(&artifact_two(&history(), |_| {})).unwrap();
    let rows = join_image_trace_attempts(&a, &[attempt()], 8).unwrap();
    assert_eq!(rows[0].status, ImageTraceJoinStatus::Matched);
    assert_eq!(rows[0].queue_total_ns, Some(10));
    let rows = join_image_trace_attempts(&a, &[attempt(), attempt()], 8).unwrap();
    assert!(rows
        .iter()
        .all(|row| row.status == ImageTraceJoinStatus::Ambiguous));
    let mut with_server = attempt();
    with_server.server_request_id = Some(REQUEST.into());
    let rows = join_image_trace_attempts(&a, &[with_server.clone(), with_server], 8).unwrap();
    assert!(rows
        .iter()
        .all(|row| row.status == ImageTraceJoinStatus::Ambiguous));
    let mut wrong = attempt();
    wrong.trace_session_uuid = Some(REQUEST.into());
    assert_eq!(
        join_image_trace_attempts(&a, &[wrong], 8).unwrap()[0].status,
        ImageTraceJoinStatus::SessionMismatch
    );
    let mut e = history();
    push(
        &mut e,
        230,
        json!({"kind":"request_submission_rejected","correlation":correlation("00000000-0000-0000-0000-000000000003",ATTEMPT),"outcome":"rejected"}),
    );
    let a = analyze(&artifact_two(&e, |_| {})).unwrap();
    assert_eq!(
        join_image_trace_attempts(&a, &[attempt()], 8).unwrap()[0].status,
        ImageTraceJoinStatus::Ambiguous
    );
}

#[test]
fn completed_service_link_must_name_its_actual_initialized_completed_model_request() {
    let mut e = history();
    for (_, event) in &mut e {
        if event["kind"] == "request_linked"
            || event["kind"] == "phase_started" && event["request"] == "a"
        {
            event["request"] = json!("wrong");
        }
    }
    let failure = analyze(&artifact_two(&e, |_| {})).unwrap_err();
    assert!(
        failure.reason.contains("actual model"),
        "{}",
        failure.reason
    );
}

#[test]
fn preinitialization_cancel_rejection_and_failure_do_not_claim_a_model_execution_join() {
    for outcome in ["cancelled", "rejected", "failed"] {
        let e = vec![
            (0, header()),
            (
                1,
                json!({"kind":"request_queued","correlation":correlation(REQUEST,ATTEMPT)}),
            ),
            (2, json!({"kind":"request_dequeued","request_id":REQUEST})),
            (
                3,
                json!({"kind":"request_linked","request_id":REQUEST,"request":"planned.only"}),
            ),
            (
                4,
                json!({"kind":"request_service_finished","request_id":REQUEST,"outcome":outcome}),
            ),
        ];
        let a = analyze(&artifact_two(&e, |_| {})).unwrap();
        assert_eq!(a.timing.status, ImageStepEvidenceStatus::Observed);
        let rows = join_image_trace_attempts(&a, &[attempt()], 8).unwrap();
        assert_eq!(rows[0].status, ImageTraceJoinStatus::Matched);
        assert_eq!(rows[0].matched_model_request, None);
        assert!(!a.timing.requests[0].model_request_initialized);
    }
}

#[test]
fn ready_cancelled_model_identity_remains_joinable_without_inventing_a_committed_step() {
    let e = vec![
        (0, header()),
        (
            1,
            json!({"kind":"request_queued","correlation":correlation(REQUEST,ATTEMPT)}),
        ),
        (2, json!({"kind":"request_dequeued","request_id":REQUEST})),
        (
            3,
            json!({"kind":"request_linked","request_id":REQUEST,"request":"a"}),
        ),
        (4, initialized("a")),
        (5, exited("a", 0, "cancelled")),
        (
            6,
            json!({"kind":"request_service_finished","request_id":REQUEST,"outcome":"cancelled"}),
        ),
    ];
    let a = analyze(&artifact_two(&e, |_| {})).unwrap();
    let rows = join_image_trace_attempts(&a, &[attempt()], 8).unwrap();
    assert_eq!(rows[0].status, ImageTraceJoinStatus::Matched);
    assert_eq!(rows[0].matched_model_request.as_deref(), Some("a"));
    assert!(a.timing.requests[0].model_request_initialized);
    assert!(a.timing.steps.is_empty());
}

#[test]
fn service_completed_cannot_relabel_a_cancelled_model_exit() {
    let e = vec![
        (0, header()),
        (
            1,
            json!({"kind":"request_queued","correlation":correlation(REQUEST,ATTEMPT)}),
        ),
        (2, json!({"kind":"request_dequeued","request_id":REQUEST})),
        (
            3,
            json!({"kind":"request_linked","request_id":REQUEST,"request":"a"}),
        ),
        (4, initialized("a")),
        (5, exited("a", 0, "cancelled")),
        (
            6,
            json!({"kind":"request_service_finished","request_id":REQUEST,"outcome":"completed"}),
        ),
    ];
    let failure = analyze(&artifact_two(&e, |_| {})).unwrap_err();
    assert!(
        failure.reason.contains("actual model"),
        "{}",
        failure.reason
    );
}

#[test]
fn phase_time_cannot_precede_its_original_service_boundary() {
    let mut e = history();
    e.iter_mut()
        .find(|(_, event)| event["kind"] == "phase_started")
        .unwrap()
        .0 = 5;
    let failure = analyze(&artifact_two(&e, |_| {})).unwrap_err();
    assert!(failure.reason.contains("backwards"), "{}", failure.reason);
}

#[test]
fn phase_server_and_model_identities_must_match_the_original_link() {
    let mut e = history();
    e.iter_mut()
        .find(|(_, event)| event["kind"] == "phase_started" && event["phase"] == "decoder")
        .unwrap()
        .1["request"] = json!("other");
    let failure = analyze(&artifact_two(&e, |_| {})).unwrap_err();
    assert!(
        failure.reason.contains("original service link"),
        "{}",
        failure.reason
    );
}

#[test]
fn completed_fence_time_cannot_precede_the_same_models_last_commit() {
    let mut e = history();
    e.iter_mut()
        .find(|(_, event)| event["kind"] == "request_exited_at_fence")
        .unwrap()
        .0 = 50;
    let failure = analyze(&artifact_two(&e, |_| {})).unwrap_err();
    assert!(failure.reason.contains("backwards"), "{}", failure.reason);
}

#[test]
fn independent_service_requests_can_arrive_with_out_of_order_boundary_times() {
    let other = "00000000-0000-0000-0000-000000000003";
    let mut e = vec![
        (0, header()),
        (
            10,
            json!({"kind":"request_queued","correlation":correlation(REQUEST,ATTEMPT)}),
        ),
        (
            5,
            json!({"kind":"request_queued","correlation":correlation(other,ATTEMPT)}),
        ),
        (20, json!({"kind":"request_dequeued","request_id":REQUEST})),
        (15, json!({"kind":"request_dequeued","request_id":other})),
    ];
    let first = push(
        &mut e,
        22,
        json!({"kind":"phase_started","request_id":REQUEST,"request":null,"phase":"metadata_preparation"}),
    );
    push(
        &mut e,
        24,
        json!({"kind":"phase_finished","start_sequence":first,"outcome":"completed"}),
    );
    let second = push(
        &mut e,
        18,
        json!({"kind":"phase_started","request_id":other,"request":null,"phase":"metadata_preparation"}),
    );
    push(
        &mut e,
        19,
        json!({"kind":"phase_finished","start_sequence":second,"outcome":"completed"}),
    );
    push(
        &mut e,
        30,
        json!({"kind":"request_service_finished","request_id":REQUEST,"outcome":"cancelled"}),
    );
    push(
        &mut e,
        25,
        json!({"kind":"request_service_finished","request_id":other,"outcome":"cancelled"}),
    );
    let a = analyze(&artifact_two(&e, |_| {})).unwrap();
    assert_eq!(a.timing.status, ImageStepEvidenceStatus::Observed);
    assert_eq!(
        a.timing
            .phases
            .iter()
            .map(|phase| phase.interval.duration_ns)
            .collect::<Vec<_>>(),
        [2, 1]
    );
    assert_eq!(
        a.timing
            .requests
            .iter()
            .map(|request| request.queue_total_ns)
            .collect::<Vec<_>>(),
        [Some(10), Some(10)]
    );
}

#[test]
fn cancelled_service_time_cannot_precede_its_actual_model_commit_or_exit() {
    for finished in [190, 202] {
        let mut e = history();
        let decoder = e
            .iter()
            .position(|(_, event)| event["kind"] == "phase_started" && event["phase"] == "decoder")
            .unwrap();
        e.drain(decoder..decoder + 2);
        let terminal = e
            .iter_mut()
            .find(|(_, event)| event["kind"] == "request_service_finished")
            .unwrap();
        terminal.0 = finished;
        terminal.1["outcome"] = json!("cancelled");
        let failure = analyze(&artifact_two(&e, |_| {})).unwrap_err();
        assert!(failure.reason.contains("backwards"), "{}", failure.reason);
    }
}
