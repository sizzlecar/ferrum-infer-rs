use super::*;
use std::io::Cursor;

fn command(submission: &str, index: u64, tokens: u64, elapsed: u64) -> Event {
    Event {
        phase: "vnext.device_native_work".into(),
        shape: Shape {
            command_index: Some(index),
            token_count: Some(tokens),
            device_elapsed_ns: Some(elapsed),
            device_interval_count: Some(1),
            ..Shape::default()
        },
        attributes: Attributes {
            physical_submission_fingerprint: Some(submission.into()),
            native_op_id: Some("ffn".into()),
            command_phase: Some("compute".into()),
            device_timing_status: Some("measured".into()),
            ..Attributes::default()
        },
        backend_detail: Some(Detail {
            device_intervals: Some(vec![Interval {
                kind: "compute".into(),
                start_offset_ns: 0,
                end_offset_ns: elapsed,
                subwork_id: Some("ffn.gate.existing".into()),
            }]),
        }),
    }
}

fn physical(submission: &str, count: u64) -> Event {
    Event {
        phase: "vnext.device_physical_submission".into(),
        shape: Shape {
            command_count: Some(count),
            ..Shape::default()
        },
        attributes: Attributes {
            physical_submission_fingerprint: Some(submission.into()),
            device_timing_status: Some("measured".into()),
            ..Attributes::default()
        },
        ..Event::default()
    }
}

fn report(events: &[Event]) -> Report {
    let mut bytes = Vec::new();
    for event in events {
        serde_json::to_writer(&mut bytes, event).unwrap();
        bytes.push(b'\n');
    }
    summarize(Cursor::new(bytes)).unwrap()
}

#[test]
fn shuffled_participant_copies_count_physical_command_and_subwork_once() {
    let mut first = serde_json::to_value(command("shared", 0, 3, 100)).unwrap();
    first["request_id"] = Value::String("request-a".into());
    first["attributes"]["participant_request_ids"] =
        Value::Array(vec![Value::String("a".into()), Value::String("b".into())]);
    let mut copy = first.clone();
    copy["request_id"] = Value::String("request-b".into());
    copy["attributes"]["participant_request_ids"]
        .as_array_mut()
        .unwrap()
        .reverse();
    let second = serde_json::to_value(command("next", 0, 1, 75)).unwrap();
    let lines = [copy, second, first]
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    let result = summarize(Cursor::new(lines)).unwrap();
    assert_eq!(result.counts.native_records, 3);
    assert_eq!(result.counts.duplicate_native_records, 1);
    assert_eq!(result.native_work.timing.commands, 2);
    assert_eq!(result.native_work.timing.measured_ns, 175);
    assert_eq!(
        result.native_work.subwork["ffn.gate.existing"].measured_ns,
        175
    );
    assert_eq!(
        result.native_work.subwork["ffn.gate.existing"].interval_count,
        2
    );
}

#[test]
fn conflicting_duplicate_timings_or_provider_metadata_are_errors() {
    let original = serde_json::to_value(command("shared", 0, 1, 100)).unwrap();
    let mut time_conflict = original.clone();
    time_conflict["shape"]["device_elapsed_ns"] = Value::from(99);
    let mut provider_conflict = original.clone();
    provider_conflict["attributes"]["provider_implementation_fingerprint"] =
        Value::String("changed".into());
    for conflict in [time_conflict, provider_conflict] {
        let error = summarize(Cursor::new(format!("{original}\n{conflict}\n"))).unwrap_err();
        assert!(error.to_string().contains("conflicting physical evidence"));
    }
}

#[test]
fn unavailable_missing_and_invalid_evidence_do_not_become_zero_cost_samples() {
    let valid = command("wave", 0, 32, 100);
    let mut unavailable = command("wave", 1, 32, 100);
    unavailable.attributes.device_timing_status = Some("unavailable".into());
    unavailable.shape.device_elapsed_ns = None;
    unavailable.shape.device_interval_count = Some(0);
    unavailable.backend_detail = None;
    let mut missing_elapsed = command("wave", 2, 32, 100);
    missing_elapsed.shape.device_elapsed_ns = None;
    let mut invalid = command("wave", 3, 32, 100);
    invalid
        .backend_detail
        .as_mut()
        .unwrap()
        .device_intervals
        .as_mut()
        .unwrap()[0]
        .end_offset_ns = 99;
    let mut legacy = command("wave", 4, 32, 25);
    legacy.backend_detail = None;
    let result = report(&[missing_elapsed, unavailable, valid, invalid, legacy]);
    let timing = result.native_work.timing;
    assert_eq!(timing.commands, 5);
    assert_eq!(timing.measured_commands, 2);
    assert_eq!(timing.measured_ns, 125);
    assert_eq!(timing.timing_statuses["unavailable"], 1);
    assert_eq!(timing.missing_elapsed_commands, 1);
    assert_eq!(timing.invalid_timing_commands, 1);
    assert_eq!(timing.missing_intervals_commands, 1);
    assert_eq!(
        result.native_work.subwork["ffn.gate.existing"].measured_ns,
        100
    );
}

#[test]
fn interval_gaps_are_excluded_and_overlaps_or_reversed_ranges_are_invalid() {
    let mut split = command("split", 0, 256, 30);
    split.shape.device_interval_count = Some(2);
    split.backend_detail = Some(Detail {
        device_intervals: Some(vec![
            Interval {
                kind: "compute".into(),
                start_offset_ns: 5,
                end_offset_ns: 15,
                subwork_id: Some("gate".into()),
            },
            Interval {
                kind: "compute".into(),
                start_offset_ns: 25,
                end_offset_ns: 45,
                subwork_id: None,
            },
        ]),
    });
    let good = report(&[split.clone()]);
    assert_eq!(good.native_work.timing.measured_ns, 30);
    assert_eq!(good.native_work.timing.unlabeled_interval_ns, 20);
    assert_eq!(good.native_work.subwork["gate"].measured_ns, 10);
    for start in [10, 46] {
        let mut invalid = split.clone();
        invalid
            .backend_detail
            .as_mut()
            .unwrap()
            .device_intervals
            .as_mut()
            .unwrap()[1]
            .start_offset_ns = start;
        assert_eq!(
            report(&[invalid])
                .native_work
                .timing
                .invalid_timing_commands,
            1
        );
    }
}

#[test]
fn one_row_heads_and_prompt_tail_do_not_assign_a_decode_phase() {
    let full = command("prefill", 0, 256, 100);
    let head = command("prefill", 1, 1, 20);
    let tail = command("prompt-tail", 0, 1, 30);
    let result = report(&[tail, head, full]);
    let prefill = &result.submissions[0];
    assert_eq!(prefill.observed_command_token_counts, [1, 256]);
    assert_eq!(prefill.maximum_command_token_count, Some(256));
    assert!(result
        .submissions
        .iter()
        .all(|submission| submission.phase.is_none()));
}

#[test]
fn mixed_phase_evidence_is_preserved_instead_of_guessing_from_largest_shape() {
    let mut a = command("mixed", 0, 32, 100);
    a.attributes.wave_phase = Some("prefill".into());
    let mut b = command("mixed", 1, 1, 20);
    b.attributes.wave_phase = Some("decode".into());
    let mut enclosing = physical("explicit", 1);
    enclosing.attributes.wave_phase = Some("mixed".into());
    let result = report(&[b, enclosing, command("explicit", 0, 33, 90), a]);
    assert_eq!(result.submissions[0].phase.as_deref(), Some("mixed"));
    assert_eq!(
        result.submissions[1].explicit_wave_phases,
        ["decode", "prefill"]
    );
    assert!(result.submissions[1].phase.is_none());
}

#[test]
fn enclosing_command_count_detects_missing_terminal_commands_without_adding_time() {
    let result = report(&[command("wave", 1, 16, 100), physical("wave", 4)]);
    let submission = &result.submissions[0];
    assert_eq!(submission.declared_command_count, Some(4));
    assert_eq!(submission.missing_declared_commands, Some(3));
    assert_eq!(submission.command_index_gaps_below_maximum, 1);
    assert_eq!(result.native_work.timing.measured_ns, 100);
    let unknown = report(&[command("old", 0, 16, 100)]);
    assert_eq!(unknown.submissions[0].missing_declared_commands, None);
}

#[test]
fn absent_identity_is_reported_and_incomplete_json_is_an_error() {
    let mut missing = command("wave", 0, 1, 100);
    missing.attributes.physical_submission_fingerprint = None;
    let result = report(&[missing]);
    assert_eq!(result.counts.records_missing_identity, 1);
    assert!(result.submissions.is_empty());
    let error = summarize(Cursor::new("{\"phase\":")).unwrap_err();
    assert!(error.to_string().contains("invalid JSON on line 1"));
}

fn with_participants(mut event: Event, start: u64, count: u64) -> Event {
    event.shape.participant_count = Some(count);
    event.shape.participant_start = Some(start);
    event.shape.participant_end = Some(start + count);
    event.attributes.participant_request_ids = Some(
        (start..start + count)
            .map(|id| format!("request-{id}"))
            .collect(),
    );
    event
}

#[test]
fn compute_participants_are_verified_independently_of_housekeeping_and_tokens() {
    let mut a = with_participants(command("wave", 0, 32, 100), 3, 32);
    a.attributes.wave_phase = Some("decode".into());
    let mut b = a.clone();
    b.shape.command_index = Some(1);
    let mut housekeeping = command("wave", 2, 1, 20);
    housekeeping.attributes.command_phase = Some("readback".into());
    housekeeping.shape.participant_count = Some(0);
    housekeeping.attributes.participant_request_ids = a.attributes.participant_request_ids.clone();
    let result = report(&[a, b, housekeeping]);
    let wave = &result.submissions[0];
    assert_eq!(wave.phase.as_deref(), Some("decode"));
    assert_eq!(wave.uniform_verified_compute_participant_count, Some(32));
    assert_eq!(wave.observed_compute_participant_counts, [32]);
    assert_eq!(wave.compute_command_shapes.len(), 2);
    assert_eq!(wave.compute_commands_missing_participant_evidence, 0);
    assert_eq!(wave.compute_commands_invalid_participant_evidence, 0);
    let shape = &wave.compute_command_shapes[0];
    assert_eq!(shape.command_index, 0);
    assert_eq!(shape.participant_start, Some(3));
    assert_eq!(shape.participant_end, Some(35));
    assert_eq!(shape.participant_request_id_count, Some(32));
    assert_eq!(shape.unique_participant_request_id_count, Some(32));

    let mut prefill = with_participants(command("prefill", 0, 512, 100), 0, 4);
    prefill.attributes.wave_phase = Some("prefill".into());
    let result = report(&[prefill]);
    assert_eq!(result.submissions[0].phase.as_deref(), Some("prefill"));
    assert_eq!(
        result.submissions[0].uniform_verified_compute_participant_count,
        Some(4)
    );
    assert_eq!(
        result.submissions[0].compute_command_shapes[0].token_count,
        Some(512)
    );
}

#[test]
fn missing_or_heterogeneous_participant_evidence_never_proves_uniform_n32() {
    let valid = with_participants(command("wave", 0, 32, 100), 0, 32);
    let legacy = command("wave", 1, 32, 100);
    let mut partial = with_participants(command("wave", 2, 32, 100), 0, 32);
    partial.shape.participant_end = None;
    let result = report(&[valid.clone(), legacy, partial]);
    let wave = &result.submissions[0];
    assert_eq!(wave.maximum_command_token_count, Some(32));
    assert_eq!(wave.uniform_verified_compute_participant_count, None);
    assert_eq!(wave.compute_commands_missing_participant_evidence, 2);
    assert_eq!(wave.compute_commands_invalid_participant_evidence, 0);

    let narrow = with_participants(command("wave", 1, 8, 100), 0, 8);
    let result = report(&[valid, narrow]);
    assert_eq!(
        result.submissions[0].observed_compute_participant_counts,
        [8, 32]
    );
    assert_eq!(
        result.submissions[0].uniform_verified_compute_participant_count,
        None
    );
    let mut readback = command("empty", 0, 32, 100);
    readback.attributes.command_phase = Some("readback".into());
    assert_eq!(
        report(&[readback]).submissions[0].uniform_verified_compute_participant_count,
        None
    );
}

#[test]
fn invalid_participant_evidence_is_reported_without_invalidating_device_time() {
    let valid = with_participants(command("wave", 0, 2, 100), 0, 2);
    let mut range = valid.clone();
    range.shape.participant_end = Some(3);
    let mut reversed = valid.clone();
    reversed.shape.participant_start = Some(u64::MAX);
    let mut missing_id = valid.clone();
    missing_id
        .attributes
        .participant_request_ids
        .as_mut()
        .unwrap()
        .pop();
    let mut duplicate_id = valid.clone();
    duplicate_id.attributes.participant_request_ids = Some(vec!["same".into(); 2]);
    let mut empty_id = valid;
    empty_id
        .attributes
        .participant_request_ids
        .as_mut()
        .unwrap()[0]
        .clear();
    let mut partial_conflict = range.clone();
    partial_conflict.shape.participant_count = None;
    for (event, issue) in [
        (range, "participant_range_count_mismatch"),
        (reversed, "reversed_participant_range"),
        (missing_id, "participant_request_id_count_mismatch"),
        (duplicate_id, "duplicate_participant_request_ids"),
        (empty_id, "empty_participant_request_id"),
        (
            partial_conflict,
            "participant_range_request_id_count_mismatch",
        ),
    ] {
        let result = report(&[event]);
        let wave = &result.submissions[0];
        assert_eq!(wave.uniform_verified_compute_participant_count, None);
        assert_eq!(wave.compute_commands_invalid_participant_evidence, 1);
        assert!(wave.compute_command_shapes[0]
            .participant_evidence_issues
            .contains(&issue));
        assert_eq!(wave.native_work.timing.measured_ns, 100);
        assert_eq!(wave.native_work.timing.invalid_timing_commands, 0);
    }
}

#[test]
fn malformed_participant_types_and_conflicting_duplicate_shapes_are_errors() {
    let original =
        serde_json::to_value(with_participants(command("wave", 0, 2, 100), 0, 2)).unwrap();
    for (pointer, value) in [
        ("/shape/participant_count", serde_json::json!(-1)),
        ("/shape/participant_start", serde_json::json!("0")),
        ("/shape/participant_end", serde_json::json!(2.5)),
        (
            "/attributes/participant_request_ids",
            serde_json::json!([1, 2]),
        ),
    ] {
        let mut malformed = original.clone();
        *malformed.pointer_mut(pointer).unwrap() = value;
        let error = summarize(Cursor::new(malformed.to_string())).unwrap_err();
        assert!(error
            .to_string()
            .contains("invalid native schema on line 1"));
    }
    let changed =
        serde_json::to_value(with_participants(command("wave", 0, 2, 100), 0, 1)).unwrap();
    let mut repeated_id = original.clone();
    repeated_id["attributes"]["participant_request_ids"]
        .as_array_mut()
        .unwrap()
        .push(Value::from("request-0"));
    for conflict in [changed, repeated_id] {
        let error = summarize(Cursor::new(format!("{original}\n{conflict}\n"))).unwrap_err();
        assert!(error.to_string().contains("conflicting physical evidence"));
    }
}
