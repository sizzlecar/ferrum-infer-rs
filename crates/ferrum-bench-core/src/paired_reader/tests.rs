use super::*;
use serde_json::json;
use std::fs;
use std::time::Duration;
use tempfile::TempDir;

fn hex(d: core::Digest) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}
fn artifact(dir: &TempDir, name: &str, bytes: &[u8]) -> Artifact {
    let path = dir.path().join(name);
    fs::write(&path, bytes).unwrap();
    Artifact {
        path,
        sha256: digest(bytes),
        bytes: bytes.len() as u64,
    }
}
fn json_artifact(dir: &TempDir, name: &str, value: Value) -> Artifact {
    artifact(dir, name, &serde_json::to_vec(&value).unwrap())
}
fn limits() -> ReaderLimits {
    ReaderLimits {
        max_runs: 4,
        max_requests_per_run: 16,
        max_events_per_run: 128,
        max_artifact_bytes: 65536,
        max_line_bytes: 8192,
        max_total_read_bytes: 1_000_000,
        max_issues: 32,
        max_raw_failures: 32,
    }
}
fn spec() -> core::Spec {
    core::Spec {
        absolute: core::AbsoluteLimits {
            quantile: core::Fraction {
                numerator: 95,
                denominator: 100,
            },
            ttft_ms: 5.0,
            tpot_ms: 20.0,
            visible_itl_ms: 20.0,
            request_deadline: Duration::from_secs(1),
        },
        tolerances: core::Tolerances {
            feasible_min_throughput_ratio: 0.95,
            feasible_max_latency_ratio: 1.05,
            rescue_min_throughput_ratio: 0.93,
        },
        kind: core::CandidateKind::Controller,
        bootstrap: core::Bootstrap {
            prng: core::Prng::SplitMix64V1,
            seed: 7,
            replicates: 32,
            max_rng_draws: 4096,
            confidence: core::Fraction {
                numerator: 95,
                denominator: 100,
            },
            episode_sensitivity: false,
        },
        budget: core::Budget {
            max_runs: 4,
            max_requests: 64,
            max_events: 1024,
            max_replicates: 64,
            max_sample_visits: 10000,
            max_scratch_bytes: 1_000_000,
        },
    }
}
fn source(i: usize) -> Value {
    json!({"source_id":format!("fixture-{i}"),"parent_sha256":hex([1;32])})
}
fn row(i: usize) -> Value {
    let start = (i * 10) as f64;
    json!({"index":i,"scenario":"episode-0","source":source(i),"requested_max_tokens":2,
        "scheduled_ms":i*10,"start_ms":start,"finish_ms":start+12.0,
        "first_visible_ms":start+2.0,"last_visible_ms":start+12.0,"ttft_ms":2.0,
        "ttft_from_scheduled_ms":2.0,"tpot_last_visible_usage_ms":10.0,
        "success":true,"timed_out":false,"error":null,"http_status":200,"content_type":"text/event-stream",
        "usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5},
        "prompt_tokens":3,"completion_tokens":2,"total_tokens":5,
        "visible_event_ms":[2.0,12.0],"visible_event_utf8_bytes":[1,1],
        "visible_event_transport_chunk":[1,2],"itl_sse_text_event_ms":[10.0],
        "transport_chunks":2,"all_sse_events":4,
        "finish_reason":"stop","done":true,"text":"ok"})
}
fn distribution(count: usize, value: f64) -> Value {
    json!({"count":count,"p50":value,"p95":value,"p99":value})
}
fn overall() -> Value {
    json!({"requests":2,"successes":2,"errors":0,"timeouts":0,
    "successful_output_tokens_known":4,"output_tok_s_successful_usage":4.0/0.022,
    "throughput_denominator_scheduled_start_to_last_terminal_seconds":0.022,
    "ttft_ms_successful":distribution(2,2.0),"tpot_last_visible_usage_ms_successful":distribution(2,10.0),
    "itl_sse_text_event_ms_all_observed":distribution(2,10.0)})
}
struct Fixture {
    dir: TempDir,
    manifest: Manifest,
}
impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let source_digest = [1; 32];
        let cell = [2; 32];
        let inputs = [3; 32];
        let identity = json_artifact(
            &dir,
            "frozen.json",
            json!({"cell":hex(cell),"inputs":hex(inputs)}),
        );
        let build = json_artifact(
            &dir,
            "build.json",
            json!({"source":"source-pin","binary":hex([4;32])}),
        );
        let requests: Vec<_> = (0..2)
            .map(|i| ExpectedRequest {
                key: core::RequestKey {
                    index: i,
                    episode: 0,
                    source: source_digest,
                    prompt: digest(format!("prompt-{i}").as_bytes()),
                    scheduled_ms: (i * 10) as u64,
                    output_cap: 2,
                },
                scenario: "episode-0".into(),
                source: source(i),
                source_digest_pointer: "/parent_sha256".into(),
            })
            .collect();
        let config = json_artifact(
            &dir,
            "config.json",
            json!({"model":"fixture-model","protocol":"chat_completions","deadline_ms":1000,
            "temperature":0.0,"request_policy":{"ignore_eos":false},"provenance":{"fixture":true},
            "requests":(0..2).map(|i|json!({"scenario":"episode-0","prompt":format!("prompt-{i}"),
                "max_tokens":2,"offset_ms":i*10,"source":source(i)})).collect::<Vec<_>>() }),
        );
        let summary = json_artifact(
            &dir,
            "summary.json",
            json!({"schema":SCHEMA,"epoch_unix_ms":1000000.0,
            "model":"fixture-model","protocol":"chat_completions","request_deadline_ms":1000,
            "request_policy":{"ignore_eos":false},"provenance":{"fixture":true},"overall":overall()}),
        );
        let records = artifact(
            &dir,
            "requests.jsonl",
            format!("{}\n{}\n", row(1), row(0)).as_bytes(),
        );
        let client_exit = artifact(&dir, "client.exit-code", b"0\n");
        let make_run = |pid: u32| {
            let server = ServerIdentity::Linux {
                boot_id: "01234567-89ab-cdef-0123-456789abcdef".into(),
                pid,
                start_ticks: 99,
            };
            let incarnation =
                format!("linux:boot=01234567-89ab-cdef-0123-456789abcdef:pid={pid}:start_ticks=99");
            let monitor = json_artifact(
                &dir,
                &format!("monitor-{pid}.jsonl"),
                json!({"event":"monitor_start","pid":pid,
                "incarnation":incarnation,"initial_host":{"pid":pid,"incarnation":incarnation}}),
            );
            let build = BuildEvidence {
                artifact: build.clone(),
                source_pointer: "/source".into(),
                source_identity: "source-pin".into(),
                binary_pointer: "/binary".into(),
                binary_sha256: [4; 32],
            };
            RunManifest {
                repeat_id: 0,
                config: config.clone(),
                records: records.clone(),
                summary: summary.clone(),
                client_exit: client_exit.clone(),
                monitor,
                server,
                server_build: build.clone(),
                client_build: build,
                epoch_unix_ms: 1000000.0,
                policy: InputPolicy {
                    model: "fixture-model".into(),
                    protocol: "chat_completions".into(),
                    deadline_ms: 1000,
                    temperature: 0.0,
                    request_policy: json!({"ignore_eos":false}),
                    provenance: json!({"fixture":true}),
                },
            }
        };
        let pair = PairManifest {
            baseline: make_run(11),
            candidate: make_run(12),
        };
        let workload = core::Workload {
            cell,
            inputs,
            repeat_ids: vec![0],
            requests: requests.iter().map(|r| r.key.clone()).collect(),
        };
        let manifest = Manifest {
            cell: DigestEvidence {
                artifact: identity.clone(),
                pointer: "/cell".into(),
                expected: cell,
            },
            inputs: DigestEvidence {
                artifact: identity,
                pointer: "/inputs".into(),
                expected: inputs,
            },
            workload,
            requests,
            pairs: vec![pair],
        };
        Self { dir, manifest }
    }
    fn candidate(&self) -> &RunManifest {
        &self.manifest.pairs[0].candidate
    }
    fn set_rows(&mut self, rows: &[Value]) {
        let text = rows.iter().map(|r| format!("{r}\n")).collect::<String>();
        self.manifest.pairs[0].candidate.records =
            artifact(&self.dir, "changed.jsonl", text.as_bytes());
    }
    fn set_overall(&mut self, overall: Value) {
        let mut v: Value =
            serde_json::from_slice(&fs::read(&self.candidate().summary.path).unwrap()).unwrap();
        v["overall"] = overall;
        self.manifest.pairs[0].candidate.summary =
            json_artifact(&self.dir, "changed-summary.json", v);
    }
}

#[test]
fn completion_order_maps_to_frozen_order_and_original_clock() {
    let f = Fixture::new();
    let audit = read_pairs(&f.manifest, spec(), limits());
    assert!(audit.identity_issues.is_empty());
    assert!(
        audit.runs.iter().all(|a| a.issues.is_empty()),
        "{:?}",
        audit.runs
    );
    let run = audit.runs[1].run.as_ref().unwrap();
    assert_eq!(
        run.requests.iter().map(|r| r.key.index).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(run.requests[1].terminal_ms, Some(22.0));
    assert_eq!(run.requests[1].ttft_from_scheduled_ms, Some(2.0));
    assert_eq!(run.requests[1].visible_span_ms, Some(10.0));
    let report = audit.report.unwrap().unwrap();
    assert_eq!(report.candidate_absolute, core::Gate::Pass);
    assert_eq!(
        report.primary.unwrap_err(),
        core::Issue::InsufficientRunClusters
    );
}

#[test]
fn missing_expected_record_stays_missing_and_never_gets_a_terminal() {
    let mut f = Fixture::new();
    f.set_rows(&[row(0)]);
    let mut s = overall();
    s["requests"] = json!(1);
    s["successes"] = json!(1);
    f.set_overall(s);
    let audit = read_run(&f.manifest, f.candidate(), limits());
    let run = audit.run.unwrap();
    assert_eq!(run.requests.len(), 2);
    assert_eq!(run.requests[1].outcome, core::Outcome::Missing);
    assert_eq!(run.requests[1].terminal_ms, None);
    assert_eq!(run.requests[1].visible_text_events, None);
}

#[test]
fn duplicate_and_wrong_source_block_pairing_without_erasing_failure_audit() {
    for duplicate in [false, true] {
        let mut f = Fixture::new();
        let mut failed = row(1);
        failed["success"] = json!(false);
        failed["error"] = json!("recorded SSE error");
        if !duplicate {
            failed["source"]["source_id"] = json!("different");
        }
        let mut rows = vec![row(0), failed];
        if duplicate {
            rows.push(row(1));
        }
        f.set_rows(&rows);
        let a = read_run(&f.manifest, f.candidate(), limits());
        assert!(a.run.is_none());
        assert_eq!(a.observed_failures, 1);
        assert_eq!(a.matched_failures, 0);
        assert!(!a.raw_failures[0].matched_input);
        assert_eq!(a.observations[1].outcome, core::Outcome::Missing);
    }
}

#[test]
fn malformed_other_field_does_not_hide_explicit_failure() {
    let mut f = Fixture::new();
    let mut failed = row(1);
    failed["success"] = json!(false);
    failed["error"] = json!("failed");
    failed["start_ms"] = json!("invalid");
    f.set_rows(&[row(0), failed.clone()]);
    let a = read_run(&f.manifest, f.candidate(), limits());
    assert!(a.run.is_none());
    assert_eq!(a.observed_failures, 1);
    assert_eq!(a.raw_failures[0].index, Some(1));
    assert_eq!(a.matched_failures, 1);
    assert!(a.raw_failures[0].matched_input);
    assert!(matches!(
        a.observations[1].outcome,
        core::Outcome::Failed(_)
    ));
    assert_eq!(a.observations[1].terminal_ms, None);
    f.set_rows(&[failed, row(1), row(0)]);
    let a = read_run(&f.manifest, f.candidate(), limits());
    assert_eq!(a.observations[1].outcome, core::Outcome::Missing);
    assert_eq!(a.matched_failures, 0);
}

#[test]
fn one_pair_known_failure_survives_unavailable_cluster_inference() {
    let mut f = Fixture::new();
    let mut failed = row(1);
    failed["success"] = json!(false);
    failed["error"] = json!("SSE error");
    f.set_rows(&[row(0), failed]);
    let mut s = overall();
    s["successes"] = json!(1);
    s["errors"] = json!(1);
    s["successful_output_tokens_known"] = json!(2);
    s["output_tok_s_successful_usage"] = json!(2.0 / 0.022);
    s["ttft_ms_successful"] = distribution(1, 2.0);
    s["tpot_last_visible_usage_ms_successful"] = distribution(1, 10.0);
    f.set_overall(s);
    let a = read_pairs(&f.manifest, spec(), limits());
    assert!(a.candidate_observed_fail);
    assert_eq!(a.runs[1].matched_failures, 1);
    let r = a.report.unwrap().unwrap();
    assert_eq!(r.candidate_absolute, core::Gate::Fail);
    assert_eq!(r.primary.unwrap_err(), core::Issue::InsufficientRunClusters);
}

#[test]
fn copied_monitor_file_is_not_a_fresh_server_incarnation() {
    let mut f = Fixture::new();
    let baseline = &f.manifest.pairs[0].baseline;
    let bytes = fs::read(&baseline.monitor.path).unwrap();
    let server = baseline.server.clone();
    f.manifest.pairs[0].candidate.monitor =
        artifact(&f.dir, "different-monitor-file.jsonl", &bytes);
    f.manifest.pairs[0].candidate.server = server;
    let a = read_pairs(&f.manifest, spec(), limits());
    assert!(matches!(
        a.report,
        Some(Err(core::Issue::DuplicateServerIncarnation))
    ));
}

#[test]
fn epoch_mismatch_and_duplicate_json_identity_fields_are_rejected() {
    let mut f = Fixture::new();
    f.manifest.pairs[0].candidate.epoch_unix_ms += 1000.0;
    let a = read_run(&f.manifest, f.candidate(), limits());
    assert!(a.run.is_none());
    assert!(a.issues.iter().any(|i| i.kind == Kind::Epoch));
    let input = br#"{"source":{"id":"a","id":"b"}}"#;
    assert_eq!(parse::<Value>(input).unwrap_err(), Kind::DuplicateJsonField);
}

#[test]
fn bounds_and_hash_failures_do_not_silently_read_or_accept_more() {
    let f = Fixture::new();
    let mut l = limits();
    l.max_total_read_bytes = 0;
    let a = read_run(&f.manifest, f.candidate(), l);
    assert!(a.run.is_none());
    assert!(a.issues.iter().any(|i| i.kind == Kind::Budget));
    let mut run = f.candidate().clone();
    run.records.sha256 = [0; 32];
    let a = read_run(&f.manifest, &run, limits());
    assert!(a.run.is_none());
    assert!(a.issues.iter().any(|i| i.kind == Kind::Hash));
    assert_eq!(a.observed_failures, 0); // unverified bytes are not trusted evidence
}

#[test]
fn failed_partial_stream_retains_valid_stall_and_zero_without_claiming_completeness() {
    let mut value = row(0);
    value["success"] = json!(false);
    value["error"] = json!("stream error");
    value["finish_ms"] = json!(3000.0);
    value["last_visible_ms"] = json!(2002.0);
    value["visible_event_ms"] = json!([2.0, null, 1002.0, 2002.0, 2002.0]);
    value["visible_event_utf8_bytes"] = json!([1, 1, 1, 1, 1]);
    value["visible_event_transport_chunk"] = json!([1, 2, 3, 4, 4]);
    value["itl_sse_text_event_ms"] = json!([null, null, 1000.0, 0.0]);
    value["transport_chunks"] = json!(4);
    value["all_sse_events"] = json!(7);
    value["text"] = json!("abcde");
    value["usage"] = Value::Null;
    value["completion_tokens"] = Value::Null;
    value["prompt_tokens"] = Value::Null;
    value["total_tokens"] = Value::Null;
    let r: wire::Record = serde_json::from_value(value).unwrap();
    let f = Fixture::new();
    let mut log = Log::new(limits());
    let (mapped, _) = map_record(&r, &f.manifest.requests[0].key, &mut log);
    assert_eq!(mapped.visible_gaps_ms, vec![1000.0, 0.0]);
    assert_eq!(mapped.visible_text_events, Some(5));
    assert!(log.audit.issues.iter().any(|i| i.kind == Kind::Events));
    assert_eq!(mapped.usage, None);
    assert!(matches!(mapped.outcome, core::Outcome::Failed(_)));
}

#[test]
fn missing_event_provenance_cannot_promote_a_raw_gap_to_visible_itl() {
    let f = Fixture::new();
    let mut value = row(0);
    value["visible_event_utf8_bytes"] = Value::Null;
    let r: wire::Record = serde_json::from_value(value).unwrap();
    let mut log = Log::new(limits());
    let (mapped, _) = map_record(&r, &f.manifest.requests[0].key, &mut log);
    assert_eq!(mapped.visible_text_events, None);
    assert!(mapped.visible_gaps_ms.is_empty());
}

#[test]
fn summary_throughput_uses_all_terminal_records_and_detects_wrong_denominator() {
    let mut f = Fixture::new();
    let mut s = overall();
    s["throughput_denominator_scheduled_start_to_last_terminal_seconds"] = json!(0.012);
    s["output_tok_s_successful_usage"] = json!(4.0 / 0.012);
    f.set_overall(s);
    let a = read_run(&f.manifest, f.candidate(), limits());
    assert!(a.run.is_none());
    assert!(a.issues.iter().any(|i| i.kind == Kind::Summary));
}

#[test]
fn numeric_policy_comparison_does_not_round_large_seed_identities() {
    assert!(semantic_eq(
        &json!({"temperature":1}),
        &json!({"temperature":1.0})
    ));
    assert!(!semantic_eq(
        &json!({"seed":9007199254740993u64}),
        &json!({"seed":9007199254740992u64})
    ));
    assert!(!semantic_eq(
        &json!({"seed":9007199254740993u64}),
        &json!({"seed":9007199254740992.0})
    ));
}

#[test]
fn invalid_protocol_fail_is_retained_even_when_summary_cannot_be_reconciled() {
    let mut f = Fixture::new();
    let mut invalid = row(1);
    invalid["finish_reason"] = json!("tool_calls");
    f.set_rows(&[row(0), invalid]);
    let a = read_pairs(&f.manifest, spec(), limits());
    assert!(a.candidate_observed_fail);
    assert_eq!(a.runs[1].matched_failures, 1);
    assert!(a.runs[1]
        .raw_failures
        .iter()
        .any(|f| f.kind == RawFailureKind::InvalidProtocol && f.matched_input));
    assert!(matches!(
        a.runs[1].observations[1].outcome,
        core::Outcome::Failed(core::FailureKind::InvalidOutput)
    ));
    assert!(a.report.is_none());
}

#[test]
fn original_deadline_cannot_be_relaxed_by_analysis_spec() {
    let f = Fixture::new();
    let mut s = spec();
    s.absolute.request_deadline = Duration::from_secs(10);
    let a = read_pairs(&f.manifest, s, limits());
    assert!(a.report.is_none());
    assert!(a.identity_issues.contains(&Kind::Input));
}

#[test]
fn epoch_scale_roundoff_is_not_misclassified_as_missing_ttft() {
    let f = Fixture::new();
    let mut v = row(0);
    let start = 1_000_000.1;
    let first = 0.125;
    v["scheduled_ms"] = json!(1_000_000);
    v["start_ms"] = json!(start);
    v["finish_ms"] = json!(start + 1.125);
    v["first_visible_ms"] = json!(start + first);
    v["last_visible_ms"] = json!(start + 1.125);
    v["ttft_ms"] = json!(first);
    v["ttft_from_scheduled_ms"] = json!(0.225);
    v["tpot_last_visible_usage_ms"] = json!(1.0);
    v["visible_event_ms"] = json!([first, 1.125]);
    v["itl_sse_text_event_ms"] = json!([1.0]);
    let r: wire::Record = serde_json::from_value(v).unwrap();
    let mut key = f.manifest.requests[0].key.clone();
    key.scheduled_ms = 1_000_000;
    let mut log = Log::new(limits());
    let (mapped, _) = map_record(&r, &key, &mut log);
    assert_eq!(mapped.ttft_from_scheduled_ms, Some(0.225));
    assert_eq!(mapped.visible_span_ms, Some(1.0));
    assert!(!log.audit.issues.iter().any(|i| i.kind == Kind::Timing));
}

#[test]
fn independent_failure_fields_survive_malformed_outcome_or_index() {
    for field in ["timed_out", "index", "success", "error"] {
        let mut f = Fixture::new();
        let mut value = row(1);
        value["success"] = json!(false);
        value["error"] = json!("recorded failure");
        value[field] = json!(["wrong type"]);
        f.set_rows(&[row(0), value]);
        let a = read_run(&f.manifest, f.candidate(), limits());
        assert!(a.run.is_none());
        assert_eq!(a.observed_failures, 1, "{field}");
        assert_eq!(a.matched_failures, usize::from(field != "index"), "{field}");
        if field != "index" {
            assert!(matches!(
                a.observations[1].outcome,
                core::Outcome::Failed(_)
            ));
            assert_eq!(a.observations[1].terminal_ms, None);
            assert_eq!(a.observations[1].usage, None);
        }
    }
}

#[test]
fn ambiguous_field_does_not_erase_a_different_unique_failure_field() {
    for nested_identity in [false, true] {
        let mut f = Fixture::new();
        let mut value = row(1);
        value["success"] = json!(false);
        value["error"] = json!("recorded failure");
        let text = if nested_identity {
            value.to_string().replace(
                "\"source_id\":\"fixture-1\"",
                "\"source_id\":\"fixture-1\",\"source_id\":\"other\"",
            )
        } else {
            value
                .to_string()
                .replace("\"success\":false", "\"success\":false,\"success\":true")
        };
        f.manifest.pairs[0].candidate.records = artifact(
            &f.dir,
            "ambiguous.jsonl",
            format!("{}\n{text}\n", row(0)).as_bytes(),
        );
        let a = read_run(&f.manifest, f.candidate(), limits());
        assert!(a.run.is_none());
        assert_eq!(a.observed_failures, 1);
        assert!(a.issues.iter().any(|i| i.kind == Kind::DuplicateJsonField));
        assert_eq!(a.matched_failures, usize::from(!nested_identity));
        assert_eq!(a.raw_failures[0].matched_input, !nested_identity);
    }
    let ambiguous = br#"{"index":1,"success":false,"success":true,"error":null,"timed_out":false}"#;
    let p: projection::Projection = serde_json::from_slice(ambiguous).unwrap();
    assert_eq!(p.failure(), None); // no unique failure field, so no invented fact
}

#[test]
fn retained_transport_and_text_contradictions_cannot_create_complete_itl() {
    let f = Fixture::new();
    for (field, value, retained_gap) in [
        ("transport_chunks", json!(1), false),
        ("all_sse_events", json!(1), false),
        ("text", json!("o"), false),
        ("text", json!("é"), false), // aggregate bytes equal; event UTF8 boundaries do not
        ("text", json!("okX"), true), // known prefix, unaccounted final text
        ("visible_event_utf8_bytes", json!([1, 2]), false),
    ] {
        let mut value_record = row(0);
        value_record[field] = value;
        let record: wire::Record = serde_json::from_value(value_record).unwrap();
        let mut log = Log::new(limits());
        let (mapped, _) = map_record(&record, &f.manifest.requests[0].key, &mut log);
        assert_eq!(mapped.visible_text_events, None, "{field}");
        assert_eq!(
            mapped.visible_gaps_ms,
            if retained_gap { vec![10.0] } else { vec![] },
            "{field}"
        );
        assert!(log.audit.issues.iter().any(|i| i.kind == Kind::Events));
    }
}

#[test]
fn later_bad_byte_length_preserves_prior_stall_without_resuming_unknown_offsets() {
    let f = Fixture::new();
    for bad_index in [1usize, 2] {
        let mut value = row(0);
        value["success"] = json!(false);
        value["error"] = json!("stream failed");
        value["finish_ms"] = json!(40.0);
        value["last_visible_ms"] = json!(32.0);
        value["tpot_last_visible_usage_ms"] = json!(30.0);
        value["visible_event_ms"] = json!([2.0, 12.0, 22.0, 32.0]);
        value["visible_event_utf8_bytes"] = json!([1, 1, 1, 1]);
        value["visible_event_utf8_bytes"][bad_index] = json!("bad");
        value["visible_event_transport_chunk"] = json!([1, 2, 3, 4]);
        value["itl_sse_text_event_ms"] = json!([10.0, 10.0, 10.0]);
        value["transport_chunks"] = json!(4);
        value["all_sse_events"] = json!(6);
        value["text"] = json!("abcd");
        let record: wire::Record = serde_json::from_value(value).unwrap();
        let mut log = Log::new(limits());
        let (mapped, _) = map_record(&record, &f.manifest.requests[0].key, &mut log);
        assert_eq!(mapped.visible_text_events, None);
        assert_eq!(
            mapped.visible_gaps_ms,
            if bad_index == 2 { vec![10.0] } else { vec![] }
        );
        assert!(matches!(mapped.outcome, core::Outcome::Failed(_)));
    }
    let mut value = row(0);
    value["finish_ms"] = json!(50.0);
    value["last_visible_ms"] = json!(42.0);
    value["tpot_last_visible_usage_ms"] = json!(40.0);
    value["visible_event_ms"] = json!([2.0, 12.0, 22.0, 32.0, 42.0]);
    value["visible_event_utf8_bytes"] = json!([1, 1, 1, 1, 1]);
    value["visible_event_transport_chunk"] = json!([1, 2, 3, 4, 5]);
    value["itl_sse_text_event_ms"] = json!([10.0, 10.0, 10.0, 10.0]);
    value["transport_chunks"] = json!(5);
    value["all_sse_events"] = json!(7);
    value["text"] = json!("éabc");
    let record: wire::Record = serde_json::from_value(value).unwrap();
    let mut log = Log::new(limits());
    let (mapped, _) = map_record(&record, &f.manifest.requests[0].key, &mut log);
    assert_eq!(mapped.visible_text_events, None);
    assert!(
        mapped.visible_gaps_ms.is_empty(),
        "invalid UTF8 segmentation must not resume at later ASCII bytes"
    );
}
