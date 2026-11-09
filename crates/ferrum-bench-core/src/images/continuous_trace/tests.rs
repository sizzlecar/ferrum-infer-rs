use super::*;
use serde_json::json;
mod expansion;
mod producer_extensions;
mod timing;

const SESSION: &str = "8bb9fb6e-2fa2-4e1a-8ddd-4a721d4d52ca";
fn header() -> Value {
    json!({"kind":"header", "device":{"runtime_implementation_fingerprint":"runtime.actual"},
        "device_opaque_allowance_bytes":128,
        "assets":{"metadata_sha256":{"transformer":"metadata-only"},"external_weight_lock_required":true},
        "prompt_expansion":{"identity":{"sources":{"weight":"original-identity"}},"max_new_tokens":2},
        "noise_profile":"noise.actual", "pipeline":{"maximum_batch_size":2,"device_capacity_bytes":1024},
        "host":{"total_bytes":4096},"context":{"entrypoint":"serve","composition":{"preset":"actual"},"service_config":{"max_num_seqs":2}}})
}
fn initialized(request: &str) -> Value {
    json!({"kind":"request_initialized","request":request,"job":7,"seed":17,"width":32,"height":32,"editing":true})
}
fn publish(job: u64, members: &[&str]) -> Value {
    json!({"kind":"cohort_published","cohort_job":job,"ordered_members":members,"plan_hash":format!("plan.{job}"),"program_fingerprint":"program.original"})
}
fn member(request: &str, start: u64, step: u64, base: u64) -> Value {
    let (sigma, next, time) = if step == 0 {
        (1., 0.5, 0.)
    } else {
        (0.5, 0., 0.5)
    };
    json!({"request":request,"packed_row_start":start,"packed_row_end":start+7,
        "step_index":step,"sigma":sigma,"next_sigma":next,"scheduler_timestep":time,
        "row_runs":[{"length":1,"time_index":base,"modality":1},
            {"length":1,"time_index":2,"modality":0},
            {"length":4,"time_index":base+1,"modality":2},
            {"length":1,"time_index":base,"modality":0}]})
}
fn commit(job: u64, basis: Value, members: Vec<Value>) -> Value {
    json!({"kind":"packed_step_committed","cohort_job":job,"plan_hash":format!("plan.{job}"),
        "program_fingerprint":"program.original","packed_request_count":members.len(),"time_basis":basis,"members":members})
}
fn exited(request: &str, steps: u64, reason: &str) -> Value {
    json!({"kind":"request_exited_at_fence","request":request,"completed_steps":steps,"reason":reason})
}
fn events() -> Vec<Value> {
    vec![
        header(),
        initialized("a"),
        initialized("b"),
        publish(10, &["a", "b"]),
        commit(
            10,
            json!([0., 0., 0.999]),
            vec![member("a", 0, 0, 0), member("b", 7, 0, 0)],
        ),
        exited("a", 1, "cancelled"),
        initialized("c"),
        publish(100, &["b", "c"]),
        commit(
            100,
            json!([0.5, 0.5, 0.999, 0., 0.]),
            vec![member("b", 0, 1, 0), member("c", 7, 0, 3)],
        ),
        exited("b", 2, "completed"),
        publish(130, &["c"]),
        commit(130, json!([0.5, 0.5, 0.999]), vec![member("c", 0, 1, 0)]),
        exited("c", 2, "completed"),
    ]
}
fn artifact(events: &[Value], modify: impl FnOnce(&mut Value)) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, event) in events.iter().enumerate() {
        serde_json::to_writer(
            &mut out,
            &json!({"schema_version":1,"session_uuid":SESSION,"sequence":i+1,"event":event}),
        )
        .unwrap();
        out.push(b'\n');
    }
    let mut footer = json!({"schema_version":1,"session_uuid":SESSION,"footer":{"sealed":true,"truncated":false,"recorded":events.len(),"dropped":0,
        "used_bytes":out.len(),"maximum_bytes":1<<20,"backing_bytes":1<<20}});
    modify(&mut footer);
    serde_json::to_writer(&mut out, &footer).unwrap();
    out.push(b'\n');
    out
}
fn analyze(bytes: &[u8]) -> Result<ImageContinuousTraceAnalysis, ImageContinuousTraceError> {
    analyze_image_continuous_trace_bytes(bytes, &ImageContinuousTraceReadLimits::default())
}
fn rejects(events: &[Value], reason: &str) {
    let error = analyze(&artifact(events, |_| {})).unwrap_err();
    assert_eq!(error.kind, ImageContinuousTraceErrorKind::Invalid);
    assert!(error.reason.contains(reason), "{error}");
}

#[test]
fn published_committed_history_proves_joint_mixed_progress_and_refill_only_from_actual_waves() {
    let bytes = artifact(&events(), |_| {});
    let a = analyze(&bytes).unwrap();
    assert_eq!(a.status, ImageStepEvidenceStatus::Observed);
    assert!(a.incomplete_reasons.is_empty());
    assert_eq!(a.records, 13);
    assert_eq!(a.input_bytes, bytes.len() as u64);
    assert_eq!(a.footer.as_ref().unwrap().used_bytes, a.event_bytes);
    assert_eq!(a.facts.multi_request_committed_waves, 2);
    assert_eq!(a.facts.distinct_step_index_waves, 1);
    assert_eq!(a.facts.distinct_scheduler_timestep_waves, 1);
    // Fixed edit augmentation makes every basis nonuniform. It does not
    // falsely mark the first equal-progress wave as asynchronous.
    assert_eq!(a.facts.nonuniform_time_basis_waves, 3);
    assert_eq!(
        a.facts
            .first_mixed_progress_commit
            .as_ref()
            .unwrap()
            .cohort_job,
        100
    );
    assert_eq!(a.facts.refill_continuation_waves, 1);
    assert_eq!(
        a.facts.first_refill_commit.as_ref().unwrap().members[0].step_index,
        1
    );
    assert_eq!(
        (
            a.facts.completed_exits,
            a.facts.cancelled_exits,
            a.facts.failed_exits
        ),
        (2, 1, 0)
    );
    let identity = a.identity.as_ref().unwrap();
    assert!(identity.serve_context_present);
    assert!(!identity.metadata_digests_authenticate_weights);
    assert_eq!(
        identity.prompt_expansion,
        Some(events()[0]["prompt_expansion"].clone())
    );
    assert_eq!(identity.context["service_config"]["max_num_seqs"], 2);
    assert_eq!(identity.assets["external_weight_lock_required"], true);
    let chunked = analyze_image_continuous_trace(
        io::BufReader::with_capacity(1, Cursor::new(&bytes)),
        &ImageContinuousTraceReadLimits::default(),
    )
    .unwrap();
    assert_eq!(
        chunked.identity.unwrap().header_event_sha256,
        identity.header_event_sha256
    );
    assert_eq!(chunked.facts.committed_waves, a.facts.committed_waves);
}

#[test]
fn publication_without_forward_never_counts_as_joint_execution() {
    let events = vec![
        header(),
        initialized("a"),
        initialized("b"),
        publish(80, &["a", "b"]),
        exited("a", 0, "cancelled"),
        exited("b", 0, "failed"),
    ];
    let result = analyze(&artifact(&events, |_| {})).unwrap();
    assert_eq!(result.status, ImageStepEvidenceStatus::Observed);
    assert_eq!(result.facts.committed_waves, 0);
    assert_eq!(result.facts.multi_request_committed_waves, 0);
    assert!(result.facts.first_joint_commit.is_none());
}

#[test]
fn component_only_commits_without_full_header_are_explicitly_not_evaluated() {
    let events = vec![commit(
        10,
        json!([0., 0., 0.999]),
        vec![member("a", 0, 0, 0), member("b", 7, 0, 0)],
    )];
    let result = analyze(&artifact(&events, |_| {})).unwrap();
    assert_eq!(result.status, ImageStepEvidenceStatus::NotEvaluated);
    assert!(result
        .incomplete_reasons
        .contains(&ImageTraceIncompleteReason::MissingHeader));
    assert!(result.identity.is_none());
    assert_eq!(result.facts.multi_request_committed_waves, 0);
    assert!(result.facts.first_joint_commit.is_none());
}

#[test]
fn altered_membership_lifecycle_step_and_row_routing_are_rejected() {
    let baseline = events();
    let mut e = baseline.clone();
    e.remove(3);
    rejects(&e, "without a published");
    let mut e = baseline.clone();
    e[4]["members"].as_array_mut().unwrap().swap(0, 1);
    rejects(&e, "membership/order");
    let mut e = baseline.clone();
    e[6]["request"] = json!("b");
    rejects(&e, "repeats an ID");
    let mut e = baseline.clone();
    e[3]["ordered_members"] = json!(["a", "a"]);
    rejects(&e, "cohort repeats");
    let mut e = baseline.clone();
    e[7]["ordered_members"] = json!(["a", "c"]);
    rejects(&e, "resurrects");
    let mut e = baseline.clone();
    e[8]["members"][0]["step_index"] = json!(2);
    rejects(&e, "does not continue");
    let mut e = baseline.clone();
    e[4]["members"][0]["step_index"] = json!(1);
    rejects(&e, "not zero");
    let mut e = baseline.clone();
    e[4]["members"][1]["packed_row_start"] = json!(8);
    rejects(&e, "member rows");
    let mut e = baseline.clone();
    e[4]["members"][0]["packed_row_end"] = json!(8);
    rejects(&e, "do not cover");
    let mut e = baseline.clone();
    e[4]["members"][0]["row_runs"][0]["time_index"] = json!(3);
    rejects(&e, "outside the actual time basis");
    let mut e = baseline.clone();
    e[4]["members"][0]["row_runs"][0]["length"] = json!(0);
    rejects(&e, "row run is empty");
    let mut e = baseline.clone();
    e[4]["time_basis"] = json!([]);
    rejects(&e, "basis is empty");
    let mut e = baseline.clone();
    e[5]["completed_steps"] = json!(0);
    rejects(&e, "committed steps");
    let mut e = baseline.clone();
    e[5]["reason"] = json!("completed");
    rejects(&e, "terminal committed sigma");
    let mut e = baseline.clone();
    e[8]["members"][0]["row_runs"][0]["modality"] = json!(0);
    rejects(&e, "changed its original");
    let mut e = baseline.clone();
    e[8]["members"][0]["sigma"] = json!(0.75);
    rejects(&e, "does not continue");
    let mut e = baseline.clone();
    e.insert(1, header());
    rejects(&e, "exactly once");
}

#[test]
fn exact_export_statistics_schema_session_and_sequence_are_required() {
    for (field, value) in [
        ("recorded", json!(12)),
        ("used_bytes", json!(1)),
        ("maximum_bytes", json!(0)),
        ("backing_bytes", json!(1)),
    ] {
        let e = analyze(&artifact(&events(), |f| f["footer"][field] = value)).unwrap_err();
        assert!(e.reason.contains("footer counters"));
    }
    let baseline = artifact(&events(), |_| {});
    let mut lines = std::str::from_utf8(&baseline)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .collect::<Vec<_>>();
    for (line, field, value) in [
        (2, "sequence", json!(4)),
        (2, "schema_version", json!(2)),
        (
            2,
            "session_uuid",
            json!("22bbfb6e-2fa2-4e1a-8ddd-4a721d4d52ca"),
        ),
    ] {
        let old = lines[line][field].clone();
        lines[line][field] = value;
        let bytes = lines
            .iter()
            .flat_map(|v| {
                let mut b = serde_json::to_vec(v).unwrap();
                b.push(b'\n');
                b
            })
            .collect::<Vec<_>>();
        assert_eq!(
            analyze(&bytes).unwrap_err().kind,
            ImageContinuousTraceErrorKind::Invalid
        );
        lines[line][field] = old;
    }
    let mut trailing = baseline.clone();
    trailing.extend_from_slice(b"{}\n");
    assert!(analyze(&trailing)
        .unwrap_err()
        .reason
        .contains("follow the export footer"));
    let contradictory = artifact(&events(), |f| f["footer"]["dropped"] = json!(1));
    assert!(analyze(&contradictory)
        .unwrap_err()
        .reason
        .contains("sticky truncation"));
    let nonfinite = String::from_utf8(baseline)
        .unwrap()
        .replacen("0.999", "1e999", 1);
    assert_eq!(
        analyze(nonfinite.as_bytes()).unwrap_err().kind,
        ImageContinuousTraceErrorKind::Invalid
    );
}

#[test]
fn valid_prefixes_unsealed_lost_records_and_missing_exits_stay_not_evaluated() {
    let bytes = artifact(&events(), |_| {});
    // Every possible byte truncation exercises actual JSON, newline, event
    // and footer boundaries rather than a hand-picked successful prefix.
    for end in 0..bytes.len() {
        let result =
            analyze(&bytes[..end]).unwrap_or_else(|e| panic!("valid prefix at byte {end}: {e}"));
        assert_eq!(
            result.status,
            ImageStepEvidenceStatus::NotEvaluated,
            "byte {end}"
        );
        assert!(!result.incomplete_reasons.is_empty());
    }
    let unsealed = analyze(&artifact(&events(), |f| {
        f["footer"]["sealed"] = json!(false)
    }))
    .unwrap();
    assert!(unsealed
        .incomplete_reasons
        .contains(&ImageTraceIncompleteReason::Unsealed));
    let lost = analyze(&artifact(&events(), |f| {
        f["footer"]["truncated"] = json!(true);
        f["footer"]["dropped"] = json!(1);
    }))
    .unwrap();
    assert_eq!(lost.status, ImageStepEvidenceStatus::NotEvaluated);
    assert!(lost
        .incomplete_reasons
        .contains(&ImageTraceIncompleteReason::ProducerDroppedRecords));
    let mut e = events();
    e.pop();
    let open = analyze(&artifact(&e, |_| {})).unwrap();
    assert!(open
        .incomplete_reasons
        .contains(&ImageTraceIncompleteReason::RequestsWithoutExit));
    // Invalid complete JSON remains invalid even without a terminating newline.
    assert_eq!(
        analyze(b"{\"schema_version\":2}").unwrap_err().kind,
        ImageContinuousTraceErrorKind::Invalid
    );
    for malformed in [
        br#"{"event":{"sigma":01"#.as_slice(),
        br#"{"event":{"sigma":0.x"#.as_slice(),
        br#"{"event":{"sigma":1e+z"#.as_slice(),
    ] {
        assert_eq!(
            analyze(malformed).unwrap_err().kind,
            ImageContinuousTraceErrorKind::Invalid
        );
    }
}

#[test]
fn reader_capacities_are_inclusive_and_fail_closed_without_reading_an_unbounded_record() {
    let bytes = artifact(&events(), |_| {});
    let longest = bytes
        .split_inclusive(|&b| b == b'\n')
        .map(<[u8]>::len)
        .max()
        .unwrap();
    let limits = ImageContinuousTraceReadLimits {
        maximum_bytes: bytes.len(),
        maximum_record_bytes: longest,
        maximum_records: 13,
        maximum_requests: 3,
        maximum_members: 2,
        maximum_row_runs_per_member: 4,
    };
    assert_eq!(
        analyze_image_continuous_trace_bytes(&bytes, &limits)
            .unwrap()
            .status,
        ImageStepEvidenceStatus::Observed
    );
    for denied in [
        ImageContinuousTraceReadLimits {
            maximum_bytes: bytes.len() - 1,
            ..limits
        },
        ImageContinuousTraceReadLimits {
            maximum_record_bytes: longest - 1,
            ..limits
        },
        ImageContinuousTraceReadLimits {
            maximum_records: 12,
            ..limits
        },
        ImageContinuousTraceReadLimits {
            maximum_requests: 2,
            ..limits
        },
        ImageContinuousTraceReadLimits {
            maximum_members: 1,
            ..limits
        },
        ImageContinuousTraceReadLimits {
            maximum_row_runs_per_member: 3,
            ..limits
        },
    ] {
        assert_eq!(
            analyze_image_continuous_trace_bytes(&bytes, &denied)
                .unwrap_err()
                .kind,
            ImageContinuousTraceErrorKind::Capacity
        );
    }
    let huge = vec![b'x'; longest + 1];
    assert_eq!(
        analyze_image_continuous_trace_bytes(&huge, &limits)
            .unwrap_err()
            .kind,
        ImageContinuousTraceErrorKind::Capacity
    );
}

#[test]
fn io_failure_is_an_error_instead_of_an_incomplete_execution_snapshot() {
    struct Failed;
    impl io::Read for Failed {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("fixture"))
        }
    }
    impl BufRead for Failed {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            Err(io::Error::other("fixture"))
        }
        fn consume(&mut self, _: usize) {}
    }
    assert_eq!(
        analyze_image_continuous_trace(Failed, &ImageContinuousTraceReadLimits::default())
            .unwrap_err()
            .kind,
        ImageContinuousTraceErrorKind::Io
    );
}
