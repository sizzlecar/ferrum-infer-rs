use super::super::model_basic::{ARITHMETIC_PROMPT, MEMORY_PROMPT, RECALL_PROMPT};
use super::*;
use ferrum_types::ModelOutputProtocol;
use serde_json::json;

fn task(id: &str, backend: Backend, checks: Vec<ModelCheck>) -> ExpectedModelRun {
    ExpectedModelRun {
        profile: ModelProfile {
            reasoning_protocol: ferrum_types::ModelReasoningProtocol::PromptOpened,
            id: id.into(),
            model: format!("fixture/{id}"),
            target: ExecutionTarget {
                architecture: "dense".into(),
                protocol: ModelOutputProtocol::Text,
                precision: "bf16".into(),
                backend,
                execution_path: "production-plan-runtime".into(),
            },
            available: true,
            estimate: None,
        },
        binary_sha256: "a".repeat(64),
        version: "0.8.8".into(),
        checks,
        disable_thinking: true,
        use_default_backend: true,
        max_tokens: 128,
        runtime_capacity: None,
        reasoning_alias_replay: false,
        stop_prompt: "Write an original sentence about rain.".into(),
    }
}

/// Small runner-result fixtures exercise the shared semantic verifier through
/// its actual report consumer; protocol framing remains covered by runner tests.
struct ReportFixture {
    value: Value,
}

impl ReportFixture {
    fn passed(expected: &ExpectedModelRun) -> Self {
        let ready = json!({
            "event": "ready", "requested_model": expected.profile.model,
            "reasoning_protocol": expected.profile.reasoning_protocol,
            "backend": match expected.profile.target.backend {
                Backend::Cpu => "CPU", Backend::Metal => "Metal", Backend::Cuda => "CUDA(0)",
            }
        });
        let mut cases = vec![
            json!({"case": "binary-version", "status": "passed", "evidence": {"version": format!("ferrum {}", expected.version)}}),
            json!({"case": "serve-startup", "status": "passed", "evidence": {
                "version": expected.version, "status": "healthy", "reasoning_protocol": expected.profile.reasoning_protocol,
                "auto_config": {"hardware_capabilities": {"backend": backend_name(expected.profile.target.backend)},
                    "selected_max_model_len": expected.runtime_capacity.map(|c| c.context_tokens),
                    "selected_kv_capacity": expected.runtime_capacity.map(|c| c.context_tokens),
                    "selected_max_sequences": expected.runtime_capacity.map(|c| c.max_num_seqs)}
            }}),
        ];
        for check in &expected.checks {
            // Explicitly describe the existing runner's cases independently of
            // ModelCheck::cases, so changing that mapping can expose omissions.
            let entries: &[&str] = match check {
                ModelCheck::Basic => &["run-basic", "serve-basic"],
                ModelCheck::Stop => &["run-stop", "serve-stop"],
                ModelCheck::Structured => &["serve-structured"],
                ModelCheck::Tools => &["serve-tools"],
                ModelCheck::Reasoning => &["run-reasoning", "serve-reasoning"],
                ModelCheck::Length => &["run-length", "serve-length"],
                ModelCheck::AutoToolsJson => &["serve-auto-tools-json"],
            };
            for name in entries {
                let evidence = match *name {
                    "run-basic" => {
                        let answers: Vec<_> = ["OK", "42", "cobalt-731"]
                            .into_iter()
                            .map(|answer| {
                                let output = boundary_fixture(answer, "", "stop", 10);
                                json!({"content": output["message"]["content"], "reasoning": null,
                                "finish_reason": output["finish_reason"], "usage": output["usage"]})
                            })
                            .collect();
                        json!({"ready": ready.clone(), "answers": answers, "prompts": [MEMORY_PROMPT, ARITHMETIC_PROMPT, RECALL_PROMPT]})
                    }
                    "serve-basic" => {
                        let mut observations = json!({});
                        for (mode, answer) in [
                            ("memory_write", "OK"),
                            ("sync", "42"),
                            ("stream", "42"),
                            ("recall", "cobalt-731"),
                            ("stream_recall", "cobalt-731"),
                        ] {
                            observations[mode] = boundary_fixture(answer, "", "stop", 10);
                        }
                        json!({"requests": basic_requests(&observations), "observations": observations})
                    }
                    "run-stop" | "serve-stop" => {
                        stop_fixture(expected, &ready, *name == "run-stop")
                    }
                    "run-reasoning" | "serve-reasoning" => {
                        let output = boundary_fixture("42", "17 plus 25 equals 42.", "stop", 10);
                        json!({"ready": ready, "enable_thinking": true, "max_tokens": expected.max_tokens,
                            "output": output, "sync": output, "stream": output})
                    }
                    "run-length" | "serve-length" => {
                        let baseline = boundary_fixture("alpha beta gamma delta", "", "stop", 10);
                        let output = boundary_fixture("alpha beta gamma", "", "length", 8);
                        json!({"ready": ready, "baseline_ready": ready, "enable_thinking": false,
                            "baseline_max_tokens": expected.max_tokens, "budget": 8, "baseline": baseline,
                            "output": output, "sync": output, "stream": output})
                    }
                    "serve-tools" => {
                        let mut evidence = json!({"reasoning_alias_replayed": expected.reasoning_alias_replay, "tool_result": 579});
                        for (mode, id) in [("sync", "sync-call"), ("stream", "stream-call")] {
                            let message = json!({"role": "assistant", "content": null,
                                "reasoning": expected.reasoning_alias_replay.then_some("Use calc to add the two numbers."),
                                "tool_calls": [{"id": id, "type": "function", "function": {"name": "calc", "arguments": "{\"expression\":\"123+456\"}"}}]});
                            let mut replayed_assistant = message.clone();
                            if mode == "stream" && expected.reasoning_alias_replay {
                                let thought = replayed_assistant
                                    .as_object_mut()
                                    .unwrap()
                                    .remove("reasoning")
                                    .unwrap();
                                replayed_assistant["reasoning_content"] = thought;
                            }
                            evidence[format!("{mode}_call")] = json!({
                                "message": message,
                                "finish_reason": "tool_calls", "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
                            });
                            evidence[format!("{mode}_continuation")] = json!({
                                "message": {"role": "assistant", "content": "579", "reasoning": null},
                                "finish_reason": "stop", "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                                "tool_call_id": id, "replayed_assistant": replayed_assistant,
                                "tool_result_message": {"role": "tool", "tool_call_id": id, "content": "{\"result\":579}"}
                            });
                        }
                        evidence
                    }
                    _ => json!({}),
                };
                cases.push(json!({"case": name, "status": "passed", "evidence": evidence}));
            }
        }
        cases.push(json!({"case": "binary-unchanged", "status": "passed", "evidence": {"sha256": expected.binary_sha256}}));
        Self {
            value: json!({
                "schema_version": 2, "status": "passed", "profile_id": expected.profile.id,
                "target": expected.profile.target, "binary_sha256": expected.binary_sha256,
                "options": {
                    "profile_id": expected.profile.id, "model": expected.profile.model,
                    "backend": backend_name(expected.profile.target.backend), "checks": expected.checks,
                    "disable_thinking": expected.disable_thinking, "use_default_backend": expected.use_default_backend,
                    "max_tokens": expected.max_tokens, "reasoning_alias_replay": expected.reasoning_alias_replay,
                    "context_tokens": expected.runtime_capacity.map(|c| c.context_tokens),
                    "max_num_seqs": expected.runtime_capacity.map(|c| c.max_num_seqs),
                    "stop_prompt": expected.stop_prompt,
                    "ferrum_bin": "/staging/ferrum", "source_label": null, "precision_label": null,
                    "report_dir": "/reports/model", "startup_timeout_secs": 600,
                },
                "sampling": {"temperature": 0, "seed": 7, "max_tokens": expected.max_tokens},
                "environment_policy": if expected.use_default_backend {
                    "remove_inherited_ferrum_overrides"
                } else { "inherit" },
                "cases": cases,
            }),
        }
    }

    fn case_mut(&mut self, name: &str) -> &mut Value {
        self.value["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|case| case["case"] == name)
            .unwrap()
    }

    fn remove_case(&mut self, name: &str) {
        self.value["cases"]
            .as_array_mut()
            .unwrap()
            .retain(|case| case["case"] != name);
    }
}

fn basic_requests(observations: &Value) -> Value {
    let memory = json!({"role": "user", "content": MEMORY_PROMPT});
    let arithmetic = json!({"role": "user", "content": ARITHMETIC_PROMPT});
    let recall = json!({"role": "user", "content": RECALL_PROMPT});
    let history = vec![
        memory.clone(),
        observations["memory_write"]["message"].clone(),
        arithmetic,
    ];
    let mut sync_recall = history.clone();
    sync_recall.extend([observations["sync"]["message"].clone(), recall.clone()]);
    let mut stream_recall = history.clone();
    stream_recall.extend([observations["stream"]["message"].clone(), recall]);
    json!({"memory_write": [memory], "sync": history, "stream": history,
        "recall": sync_recall, "stream_recall": stream_recall})
}

fn stop_fixture(expected: &ExpectedModelRun, ready: &Value, is_run: bool) -> Value {
    use super::super::model_stop::{select_stop_boundary, StopMode};
    use sha2::{Digest, Sha256};
    let baseline_text = "alpha beta gamma delta epsilon and some more words";
    let mut baseline = boundary_fixture(baseline_text, "", "stop", 30);
    let boundary = select_stop_boundary(&baseline).unwrap();
    let mut output = boundary_fixture(&boundary.expected_prefix, "", "stop", 12);
    for (value, text) in [
        (&mut baseline, baseline_text),
        (&mut output, boundary.expected_prefix.as_str()),
    ] {
        value["raw_availability"] = json!(if is_run { "captured" } else { "not_exposed" });
        if is_run {
            value["ready"] = ready.clone();
            value["raw_text"] = json!(text);
            value["raw_text_sha256"] = json!(format!("{:x}", Sha256::digest(text.as_bytes())));
        }
    }
    let outputs = if is_run {
        json!({"run": output})
    } else {
        json!({"sync": output, "stream": output})
    };
    json!({"ready": ready, "probes": [{"mode": StopMode::TaskDefault,
        "inputs": {"prompt": expected.stop_prompt, "temperature": 0, "seed": 7,
            "max_tokens": expected.max_tokens, "runtime_capacity": expected.runtime_capacity,
            "enable_thinking": if expected.disable_thinking { Some(false) } else { None }},
        "boundary": boundary, "baseline": baseline, "outputs": outputs}]})
}

fn rejected(expected: &ExpectedModelRun, fixture: &ReportFixture, reason: &str) {
    let errors = verify_model_report(expected, &fixture.value).unwrap_err();
    assert!(
        errors.iter().any(|error| error.contains(reason)),
        "missing {reason:?} in {errors:?}"
    );
}

#[test]
fn pinned_model_reports_require_actual_sources_from_both_product_entrypoints() {
    let revision = "a".repeat(40);
    let mut expected = task("pinned", Backend::Cuda, vec![ModelCheck::Basic]);
    expected.profile.model = format!("fixture/model@{revision}");
    let mut identity = json!({"schema_version": 1, "resolved_model": "fixture/model", "requested_model": expected.profile.model,
        "original_sources": {}, "resolved_sources": {}});
    for role in ["weights", "semantic", "tokenizer"] {
        identity["original_sources"][role] = json!({
            "kind": "repository", "location": "fixture/model", "requested_revision": revision});
        identity["resolved_sources"][role] = json!({
            "canonical_location": "fixture/model", "resolved_revision": revision,
            "files": [{"relative_path": format!("{role}.bin"), "size_bytes": 16, "sha256": "c".repeat(64)}]});
    }
    let mut valid = ReportFixture::passed(&expected);
    for name in ["run-basic", "serve-startup"] {
        valid.case_mut(name)["evidence"]["source_identity"] = identity.clone();
    }
    verify_model_report(&expected, &valid.value).unwrap();
    for name in ["run-basic", "serve-startup"] {
        let mut changed = ReportFixture {
            value: valid.value.clone(),
        };
        changed.case_mut(name)["evidence"]["source_identity"] = Value::Null;
        rejected(&expected, &changed, "source evidence is missing");
        for (field, value) in [
            ("schema_version", Value::Null),
            ("schema_version", json!(999)),
            ("resolved_model", json!("fixture/other")),
        ] {
            let mut changed = ReportFixture {
                value: valid.value.clone(),
            };
            changed.case_mut(name)["evidence"]["source_identity"][field] = value;
            rejected(
                &expected,
                &changed,
                "source identity schema or resolved public model",
            );
        }
        for role in ["weights", "semantic", "tokenizer"] {
            for (parent, field, replacement) in [
                ("original_sources", "location", json!("fixture/other")),
                ("original_sources", "requested_revision", Value::Null),
                (
                    "resolved_sources",
                    "canonical_location",
                    json!("fixture/other"),
                ),
                (
                    "resolved_sources",
                    "resolved_revision",
                    json!("b".repeat(40)),
                ),
                ("resolved_sources", "files", json!([])),
                (
                    "resolved_sources",
                    "files",
                    json!([{"relative_path":"weights.bin","size_bytes":16}]),
                ),
            ] {
                let mut changed = ReportFixture {
                    value: valid.value.clone(),
                };
                changed.case_mut(name)["evidence"]["source_identity"][parent][role][field] =
                    replacement;
                assert!(
                    verify_model_report(&expected, &changed.value).is_err(),
                    "{name}/{role}/{field}"
                );
            }
        }
    }
}

#[test]
fn checks_have_strict_round_trip_names() {
    for check in [
        ModelCheck::Basic,
        ModelCheck::Stop,
        ModelCheck::Structured,
        ModelCheck::Tools,
        ModelCheck::Reasoning,
        ModelCheck::Length,
    ] {
        let text = check.to_string();
        assert_eq!(text.parse::<ModelCheck>().unwrap(), check);
        assert_eq!(serde_json::to_value(check).unwrap(), text);
        assert_eq!(
            serde_json::from_value::<ModelCheck>(json!(text)).unwrap(),
            check
        );
    }
    for text in ["", "Basic", " basic", "serve-basic", "all"] {
        assert!(text.parse::<ModelCheck>().is_err());
    }
}

#[test]
fn selected_checks_are_reusable_and_do_not_impose_unselected_model_work() {
    for checks in [
        vec![ModelCheck::Basic],
        vec![ModelCheck::Stop],
        vec![ModelCheck::Structured],
        vec![ModelCheck::Tools],
        vec![ModelCheck::Tools, ModelCheck::Basic],
    ] {
        let expected = task("selected", Backend::Metal, checks);
        let mut fixture = ReportFixture::passed(&expected);
        fixture.value["options"]["checks"]
            .as_array_mut()
            .unwrap()
            .reverse();
        // Descriptive labels and artifact locations are not evidence of execution.
        fixture.value["options"]["report_dir"] = json!("/another/report/location");
        fixture.value["options"]["precision_label"] = json!("descriptive only");
        assert_eq!(verify_model_report(&expected, &fixture.value), Ok(()));
    }
}

#[test]
fn basic_semantics_are_rechecked_for_every_declared_reasoning_capability() {
    for capability in [
        ModelReasoningProtocol::None,
        ModelReasoningProtocol::PromptOpened,
        ModelReasoningProtocol::ModelGenerated,
    ] {
        let mut expected = task("basic", Backend::Metal, vec![ModelCheck::Basic]);
        expected.profile.reasoning_protocol = capability;
        let valid = ReportFixture::passed(&expected);
        verify_model_report(&expected, &valid.value).unwrap();
        for (name, pointer, wrong) in [
            (
                "run-basic",
                "/evidence/answers/0/content",
                json!("not remembered"),
            ),
            ("run-basic", "/evidence/answers/1/content", json!("43")),
            (
                "run-basic",
                "/evidence/answers/2/content",
                json!("other-code"),
            ),
            (
                "serve-basic",
                "/evidence/observations/memory_write",
                Value::Null,
            ),
            (
                "serve-basic",
                "/evidence/observations/sync/message/content",
                json!("43"),
            ),
            (
                "serve-basic",
                "/evidence/observations/stream/message/content",
                json!("43"),
            ),
            (
                "serve-basic",
                "/evidence/observations/recall/message/content",
                json!("other-code"),
            ),
            (
                "serve-basic",
                "/evidence/observations/stream_recall/message/content",
                json!("other-code"),
            ),
        ] {
            let mut invalid = ReportFixture::passed(&expected);
            *invalid.case_mut(name).pointer_mut(pointer).unwrap() = wrong;
            assert!(
                verify_model_report(&expected, &invalid.value).is_err(),
                "accepted {capability:?} {name} {pointer}"
            );
        }
        for (name, pointer) in [
            ("run-basic", "/evidence/prompts/1"),
            ("serve-basic", "/evidence/requests/stream_recall/3/content"),
        ] {
            let mut invalid = ReportFixture::passed(&expected);
            *invalid.case_mut(name).pointer_mut(pointer).unwrap() = json!("rewritten request");
            assert!(
                verify_model_report(&expected, &invalid.value).is_err(),
                "accepted altered {name} input {pointer}"
            );
        }
        let mut missing = ReportFixture::passed(&expected);
        missing.case_mut("run-basic")["evidence"]["answers"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(verify_model_report(&expected, &missing.value).is_err());
    }
}

#[test]
fn missing_entrypoint_or_failed_case_cannot_hide_behind_a_passed_summary() {
    for (check, names) in [
        (ModelCheck::Basic, vec!["run-basic", "serve-basic"]),
        (ModelCheck::Stop, vec!["run-stop", "serve-stop"]),
        (ModelCheck::Structured, vec!["serve-structured"]),
        (ModelCheck::Tools, vec!["serve-tools"]),
    ] {
        let expected = task("selected", Backend::Cuda, vec![check]);
        for name in names {
            let mut missing = ReportFixture::passed(&expected);
            missing.remove_case(name);
            rejected(
                &expected,
                &missing,
                &format!("missing required model case {name}"),
            );
            for status in ["failed", "running", "not-run"] {
                let mut failed = ReportFixture::passed(&expected);
                failed.case_mut(name)["status"] = json!(status);
                failed.case_mut(name)["error"] = json!("existing case oracle rejected its output");
                rejected(
                    &expected,
                    &failed,
                    &format!("model case {name} is not passed"),
                );
            }
        }
    }
}

#[test]
fn duplicates_unexecuted_cases_and_nonterminal_reports_are_errors() {
    let expected = task("selected", Backend::Metal, vec![ModelCheck::Basic]);
    let mut duplicate = ReportFixture::passed(&expected);
    let second = duplicate.case_mut("run-basic").clone();
    duplicate.value["cases"]
        .as_array_mut()
        .unwrap()
        .push(second);
    rejected(&expected, &duplicate, "duplicate model case run-basic");
    let mut unfinished = ReportFixture::passed(&expected);
    unfinished.value["status"] = json!("running");
    rejected(&expected, &unfinished, "not terminal passed");
    let mut unexecuted = ReportFixture::passed(&expected);
    unexecuted.value["unexecuted_serve_checks"] = json!(["basic"]);
    rejected(&expected, &unexecuted, "unexecuted serve checks");
    let mut no_evidence = ReportFixture::passed(&expected);
    no_evidence
        .case_mut("serve-basic")
        .as_object_mut()
        .unwrap()
        .remove("evidence");
    rejected(&expected, &no_evidence, "missing oracle evidence");
}

#[test]
fn task_inputs_are_checked_before_any_process_is_started() {
    let expected = task("selected", Backend::Metal, vec![ModelCheck::Basic]);
    let fixture = ReportFixture::passed(&expected);
    for (field, wrong) in [
        ("profile_id", json!("another")),
        ("model", json!("smaller/replacement")),
        ("backend", json!("cpu")),
        ("disable_thinking", json!(false)),
        ("use_default_backend", json!(false)),
        ("max_tokens", json!(1)),
        ("reasoning_alias_replay", json!(true)),
        ("stop_prompt", json!("different prompt")),
        ("checks", json!(["structured"])),
        ("checks", json!(["basic", "basic"])),
        ("checks", json!(["unknown"])),
    ] {
        let mut options = fixture.value["options"].clone();
        options[field] = wrong;
        assert!(
            verify_model_options(&expected, &options).is_err(),
            "accepted mismatched {field}"
        );
    }
    let mut invalid = expected.clone();
    invalid.checks.clear();
    assert!(verify_model_options(&invalid, &fixture.value["options"]).is_err());
    invalid = expected.clone();
    invalid.reasoning_alias_replay = true;
    let mut options = fixture.value["options"].clone();
    options["reasoning_alias_replay"] = json!(true);
    assert!(verify_model_options(&invalid, &options)
        .unwrap_err()
        .iter()
        .any(|error| error.contains("requires the tools check")));
}

#[test]
fn backend_observations_are_independent_of_matching_declared_targets() {
    let expected = task(
        "selected",
        Backend::Cuda,
        vec![ModelCheck::Basic, ModelCheck::Stop],
    );
    for name in ["run-basic", "run-stop"] {
        for backend in [
            Value::Null,
            json!("CPU"),
            json!("unknown"),
            json!("CUDA()"),
            json!("CUDA(0) CPU fallback"),
        ] {
            let mut fixture = ReportFixture::passed(&expected);
            fixture.case_mut(name)["evidence"]["ready"]["backend"] = backend;
            rejected(&expected, &fixture, &format!("{name} backend"));
        }
    }
    for backend in [Value::Null, json!("cpu"), json!("unknown")] {
        let mut fixture = ReportFixture::passed(&expected);
        fixture.case_mut("serve-startup")["evidence"]["auto_config"]["hardware_capabilities"]
            ["backend"] = backend;
        rejected(&expected, &fixture, "serve-startup backend");
    }
    let mut valid = ReportFixture::passed(&expected);
    valid.case_mut("run-basic")["evidence"]["ready"]["backend"] = json!("CUDA(3)");
    assert_eq!(verify_model_report(&expected, &valid.value), Ok(()));
}

#[test]
fn binary_and_declared_execution_class_must_match_the_expected_task() {
    let expected = task("selected", Backend::Metal, vec![ModelCheck::Basic]);
    for (field, wrong) in [
        ("architecture", json!("other-architecture")),
        ("protocol", json!("harmony_gpt_oss")),
        ("precision", json!("int4")),
        ("backend", json!("cuda")),
        ("execution_path", json!("legacy-model-executor")),
    ] {
        let mut fixture = ReportFixture::passed(&expected);
        fixture.value["target"][field] = wrong;
        rejected(&expected, &fixture, "declared target");
    }
    let mut unbound = ReportFixture::passed(&expected);
    unbound.value["target"] = Value::Null;
    rejected(&expected, &unbound, "declared target");
    let mut replaced = ReportFixture::passed(&expected);
    replaced.value["binary_sha256"] = json!("b".repeat(64));
    rejected(&expected, &replaced, "report binary_sha256");
    let mut changed = ReportFixture::passed(&expected);
    changed.case_mut("binary-unchanged")["evidence"]["sha256"] = json!("b".repeat(64));
    rejected(&expected, &changed, "binary-unchanged");
    let mut version = ReportFixture::passed(&expected);
    version.case_mut("binary-version")["evidence"]["version"] = json!("ferrum 0.8.7");
    rejected(&expected, &version, "binary-version");
    version = ReportFixture::passed(&expected);
    version.case_mut("serve-startup")["evidence"]["version"] = json!("0.8.7");
    rejected(&expected, &version, "serve-startup version");
}

#[test]
fn declared_alias_replay_cannot_hide_a_case_that_did_not_replay_it() {
    let mut expected = task("selected", Backend::Cuda, vec![ModelCheck::Tools]);
    expected.reasoning_alias_replay = true;
    expected.disable_thinking = false;
    let mut fixture = ReportFixture::passed(&expected);
    assert_eq!(verify_model_report(&expected, &fixture.value), Ok(()));
    fixture.case_mut("serve-tools")["evidence"]["reasoning_alias_replayed"] = json!(false);
    rejected(&expected, &fixture, "reasoning alias");
}

#[test]
fn report_sets_match_profiles_without_order_or_pass_ratio_shortcuts() {
    let metal = task("readme-metal", Backend::Metal, vec![ModelCheck::Basic]);
    let cuda = task("readme-cuda", Backend::Cuda, vec![ModelCheck::Basic]);
    let metal_report = ReportFixture::passed(&metal).value;
    let cuda_report = ReportFixture::passed(&cuda).value;
    let expected = [metal, cuda];
    assert_eq!(
        verify_model_reports(&expected, &[cuda_report.clone(), metal_report.clone()]),
        Ok(())
    );
    assert!(verify_model_reports(&expected, &[metal_report.clone()])
        .unwrap_err()
        .iter()
        .any(|error| error.contains("missing report for profile readme-cuda")));
    assert!(verify_model_reports(
        &expected,
        &[
            metal_report.clone(),
            metal_report.clone(),
            cuda_report.clone()
        ]
    )
    .unwrap_err()
    .iter()
    .any(|error| error.contains("duplicate report")));
    let mut unexpected = cuda_report;
    unexpected["profile_id"] = json!("unselected");
    assert!(
        verify_model_reports(&expected, &[metal_report.clone(), unexpected])
            .unwrap_err()
            .iter()
            .any(|error| error.contains("unexpected report"))
    );
    assert!(
        verify_model_reports(&[expected[0].clone(), expected[0].clone()], &[metal_report])
            .unwrap_err()
            .iter()
            .any(|error| error.contains("duplicate expected profile"))
    );
    assert!(verify_model_reports(&[], &[]).is_err());
}

#[test]
fn default_backend_mode_cannot_claim_an_inherited_override_environment() {
    let expected = task("readme", Backend::Metal, vec![ModelCheck::Basic]);
    let mut fixture = ReportFixture::passed(&expected);
    fixture.value["environment_policy"] = json!("inherit");
    rejected(&expected, &fixture, "environment policy");
    fixture
        .value
        .as_object_mut()
        .unwrap()
        .remove("environment_policy");
    rejected(&expected, &fixture, "environment policy");

    let mut explicit = expected;
    explicit.use_default_backend = false;
    let fixture = ReportFixture::passed(&explicit);
    assert_eq!(verify_model_report(&explicit, &fixture.value), Ok(()));
}

#[test]
fn passed_summary_cannot_contradict_health_or_task_verification_errors() {
    let expected = task("selected", Backend::Metal, vec![ModelCheck::Basic]);
    for status in [json!("unhealthy"), Value::Null] {
        let mut fixture = ReportFixture::passed(&expected);
        fixture.case_mut("serve-startup")["evidence"]["status"] = status;
        rejected(&expected, &fixture, "health status");
    }
    let mut fixture = ReportFixture::passed(&expected);
    fixture.value["task_verification_errors"] = json!(["serve-basic did not run"]);
    rejected(&expected, &fixture, "task verification errors");
    fixture.value["task_verification_errors"] = json!([]);
    assert_eq!(verify_model_report(&expected, &fixture.value), Ok(()));
}

#[test]
fn tools_require_both_named_calls_and_their_actual_continuations() {
    let expected = task("selected", Backend::Cuda, vec![ModelCheck::Tools]);
    for name in [
        "sync_call",
        "stream_call",
        "sync_continuation",
        "stream_continuation",
    ] {
        let mut fixture = ReportFixture::passed(&expected);
        fixture.case_mut("serve-tools")["evidence"]
            .as_object_mut()
            .unwrap()
            .remove(name);
        rejected(&expected, &fixture, name);

        let mut fixture = ReportFixture::passed(&expected);
        fixture.case_mut("serve-tools")["evidence"][name]["finish_reason"] = json!("length");
        rejected(&expected, &fixture, name);
    }
    for mode in ["sync", "stream"] {
        for (field, wrong) in [("id", json!("")), ("type", json!("unknown"))] {
            let mut fixture = ReportFixture::passed(&expected);
            fixture.case_mut("serve-tools")["evidence"][format!("{mode}_call")]["message"]
                ["tool_calls"][0][field] = wrong;
            rejected(&expected, &fixture, &format!("{mode}_call"));
        }
        let mut fixture = ReportFixture::passed(&expected);
        fixture.case_mut("serve-tools")["evidence"][format!("{mode}_call")]["message"]
            ["tool_calls"][0]["function"]["name"] = json!("lookup_weather");
        rejected(&expected, &fixture, &format!("{mode}_call"));

        let mut fixture = ReportFixture::passed(&expected);
        fixture.case_mut("serve-tools")["evidence"][format!("{mode}_continuation")]
            ["tool_call_id"] = json!("another-call");
        rejected(&expected, &fixture, &format!("{mode}_continuation"));
    }
}

#[test]
fn tool_report_cannot_substitute_expression_result_or_actual_replay_messages() {
    let expected = task("tools", Backend::Cuda, vec![ModelCheck::Tools]);
    let valid = ReportFixture::passed(&expected);
    verify_model_report(&expected, &valid.value).unwrap();
    let mut wrong_result = ReportFixture::passed(&expected);
    wrong_result.case_mut("serve-tools")["evidence"]["tool_result"] = json!(580);
    assert!(verify_model_report(&expected, &wrong_result.value).is_err());
    for mode in ["sync", "stream"] {
        // A self-consistent replay of the wrong expression still violates the task.
        let mut invalid = ReportFixture::passed(&expected);
        let evidence = &mut invalid.case_mut("serve-tools")["evidence"];
        let wrong_arguments = json!("{\"expression\":\"123+457\"}");
        evidence[format!("{mode}_call")]["message"]["tool_calls"][0]["function"]["arguments"] =
            wrong_arguments.clone();
        evidence[format!("{mode}_continuation")]["replayed_assistant"]["tool_calls"][0]
            ["function"]["arguments"] = wrong_arguments;
        assert!(verify_model_report(&expected, &invalid.value).is_err());
        for (pointer, wrong) in [
            ("/message/content", json!("580")),
            ("/tool_call_id", json!("another-call")),
            ("/replayed_assistant/tool_calls/0/id", json!("another-call")),
            ("/tool_result_message/tool_call_id", json!("another-call")),
            ("/tool_result_message/content", json!("{\"result\":580}")),
            ("/replayed_assistant", Value::Null),
            ("/usage/completion_tokens", json!(0)),
        ] {
            let mut invalid = ReportFixture::passed(&expected);
            let output =
                &mut invalid.case_mut("serve-tools")["evidence"][format!("{mode}_continuation")];
            *output.pointer_mut(pointer).unwrap() = wrong;
            assert!(
                verify_model_report(&expected, &invalid.value).is_err(),
                "accepted {mode} {pointer}"
            );
        }
    }
}

fn boundary_fixture(content: &str, reasoning: &str, finish: &str, tokens: u64) -> Value {
    json!({"message": {"role": "assistant", "content": content, "reasoning": reasoning},
        "finish_reason": finish, "usage": {"prompt_tokens": 5, "completion_tokens": tokens, "total_tokens": 5 + tokens}})
}

#[test]
fn boundary_reports_cannot_replace_observations_with_a_passed_parent() {
    let expected = task(
        "boundaries",
        Backend::Metal,
        vec![ModelCheck::Reasoning, ModelCheck::Length],
    );
    let valid = ReportFixture::passed(&expected);
    verify_model_report(&expected, &valid.value).unwrap();
    for name in [
        "run-reasoning",
        "serve-reasoning",
        "run-length",
        "serve-length",
    ] {
        let mut fixture = ReportFixture::passed(&expected);
        fixture.case_mut(name)["evidence"] = json!({"passed": true});
        rejected(&expected, &fixture, name);
    }
    let mut fixture = ReportFixture::passed(&expected);
    fixture.case_mut("serve-reasoning")["evidence"]["stream"]["message"]["reasoning"] = Value::Null;
    rejected(&expected, &fixture, "no distinct nonempty thought");
    let mut fixture = ReportFixture::passed(&expected);
    fixture.case_mut("run-length")["evidence"]["output"]["finish_reason"] = json!("eos");
    rejected(&expected, &fixture, "length finish");
    let mut fixture = ReportFixture::passed(&expected);
    fixture.case_mut("serve-length")["evidence"]["stream"]["usage"]["completion_tokens"] = json!(7);
    rejected(&expected, &fixture, "token usage");
    let mut fixture = ReportFixture::passed(&expected);
    fixture.case_mut("run-length")["evidence"]["baseline_ready"]["backend"] = json!("CPU");
    rejected(&expected, &fixture, "baseline readiness");
    let mut fixture = ReportFixture::passed(&expected);
    fixture.case_mut("serve-reasoning")["evidence"]["enable_thinking"] = json!(false);
    rejected(&expected, &fixture, "explicitly enable thinking");
}

#[test]
fn truncation_budget_cannot_accept_empty_or_overflowing_baselines() {
    for tokens in [0, 1, 2, u64::MAX] {
        assert!(length_probe_budget(tokens).is_err());
    }
    assert_eq!(length_probe_budget(10).unwrap(), 8);
    let baseline = boundary_fixture("alpha beta", "", "stop", 10);
    for content in ["", "alpha beta", "different", "<|channel>final"] {
        assert!(verify_length_observations(
            &baseline,
            &boundary_fixture(content, "", "length", 8),
            8
        )
        .is_err());
    }
}

#[test]
fn reasoning_capability_cannot_be_faked_by_nullable_thought_or_declared_target() {
    let mut expected = task("non-thinking", Backend::Cpu, vec![ModelCheck::Basic]);
    expected.profile.reasoning_protocol = ModelReasoningProtocol::None;
    let valid = ReportFixture::passed(&expected);
    verify_model_report(&expected, &valid.value).unwrap();
    for capability in [
        serde_json::Value::Null,
        json!("unknown"),
        json!("prompt_opened"),
        json!("model_generated"),
    ] {
        for case in ["run-basic", "serve-startup"] {
            let mut bad = ReportFixture::passed(&expected);
            let observation = if case == "run-basic" {
                &mut bad.case_mut(case)["evidence"]["ready"]
            } else {
                &mut bad.case_mut(case)["evidence"]
            };
            observation["reasoning_protocol"] = capability.clone();
            assert!(
                verify_model_report(&expected, &bad.value).is_err(),
                "accepted {case}: {capability}"
            );
        }
    }
    for mode in ["memory_write", "sync", "stream", "recall", "stream_recall"] {
        let mut bad = ReportFixture::passed(&expected);
        bad.case_mut("serve-basic")["evidence"]["observations"][mode]["message"]["reasoning"] =
            json!("unexpected thought");
        assert!(verify_model_report(&expected, &bad.value).is_err());
    }
    let mut bad = ReportFixture::passed(&expected);
    bad.case_mut("run-basic")["evidence"]["answers"][1]["reasoning"] = json!("unexpected thought");
    assert!(verify_model_report(&expected, &bad.value).is_err());
    let mut bad = ReportFixture::passed(&expected);
    bad.case_mut("serve-basic")["evidence"]["observations"] = Value::Null;
    assert!(verify_model_report(&expected, &bad.value).is_err());
    expected.checks = vec![ModelCheck::Reasoning];
    let fabricated = ReportFixture::passed(&expected);
    assert!(verify_model_report(&expected, &fabricated.value).is_err());
    expected.profile.reasoning_protocol = ModelReasoningProtocol::Unknown;
    let unknown = ReportFixture::passed(&expected);
    assert!(verify_model_options(&expected, &unknown.value["options"]).is_err());
}

#[test]
fn absence_oracle_is_distinct_from_enabled_reasoning_and_rejects_control_leaks() {
    let absent = boundary_fixture("42", "", "stop", 10);
    verify_reasoning_absence_observation(&absent, "42").unwrap();
    assert!(verify_reasoning_observation(&absent).is_err());
    for (content, thought, finish) in [
        ("42", "private", "stop"),
        ("<think>42", "", "stop"),
        ("42", "", "length"),
        ("43", "", "stop"),
    ] {
        assert!(verify_reasoning_absence_observation(
            &boundary_fixture(content, thought, finish, 10),
            "42"
        )
        .is_err());
    }
    let mut malformed = absent;
    malformed["message"]["reasoning"] = json!(false);
    assert!(verify_reasoning_absence_observation(&malformed, "42").is_err());
}

#[test]
fn stop_report_replays_channel_prefix_usage_and_entrypoint_observations() {
    let expected = task("selected", Backend::Metal, vec![ModelCheck::Stop]);
    let valid = ReportFixture::passed(&expected);
    verify_model_report(&expected, &valid.value).unwrap();
    for (case, pointer, value) in [
        (
            "run-stop",
            "/evidence/probes/0/outputs/run/message/content",
            json!("wrong prefix"),
        ),
        (
            "run-stop",
            "/evidence/probes/0/outputs/run/raw_text",
            json!("invented raw text"),
        ),
        (
            "serve-stop",
            "/evidence/probes/0/outputs/stream/finish_reason",
            json!("length"),
        ),
        (
            "serve-stop",
            "/evidence/probes/0/outputs/stream/usage/completion_tokens",
            json!(30),
        ),
        (
            "serve-stop",
            "/evidence/probes/0/outputs/stream",
            Value::Null,
        ),
        (
            "serve-stop",
            "/evidence/probes/0/inputs/enable_thinking",
            json!(true),
        ),
        (
            "serve-stop",
            "/evidence/probes/0/mode",
            json!("disabled_thinking"),
        ),
        (
            "serve-stop",
            "/evidence/probes/0/boundary/channel",
            json!("reasoning"),
        ),
    ] {
        let mut changed = ReportFixture::passed(&expected);
        *changed.case_mut(case).pointer_mut(pointer).unwrap() = value;
        assert!(
            verify_model_report(&expected, &changed.value).is_err(),
            "accepted {case}{pointer}"
        );
    }
    let mut old = ReportFixture::passed(&expected);
    old.case_mut("serve-stop")["evidence"] = json!({"expected_prefix":"alpha", "outputs":[]});
    rejected(&expected, &old, "missing channel observations");
}

#[test]
fn a_reasoning_stop_requires_separate_final_evidence_even_when_false_still_reasons() {
    use super::super::model_stop::{select_stop_boundary, StopMode};
    let mut expected = task("selected", Backend::Metal, vec![ModelCheck::Stop]);
    expected.disable_thinking = false;
    let mut fixture = ReportFixture::passed(&expected);
    let mut final_probe = fixture.case_mut("serve-stop")["evidence"]["probes"][0].clone();
    final_probe["mode"] = json!(StopMode::DisabledThinking);
    final_probe["inputs"]["enable_thinking"] = json!(false);
    for pointer in ["/baseline", "/outputs/sync", "/outputs/stream"] {
        final_probe.pointer_mut(pointer).unwrap()["message"]["reasoning"] =
            json!("A retained thought.");
    }
    let mut baseline = boundary_fixture("", "I will examine the street and describe its reflections before composing my final paragraph.", "length", u64::from(expected.max_tokens));
    baseline["raw_availability"] = json!("not_exposed");
    let boundary = select_stop_boundary(&baseline).unwrap();
    let mut output = boundary_fixture("", &boundary.expected_prefix, "stop", 20);
    output["raw_availability"] = json!("not_exposed");
    let mut thought_probe = final_probe.clone();
    thought_probe["mode"] = json!(StopMode::TaskDefault);
    thought_probe["inputs"]["enable_thinking"] = Value::Null;
    thought_probe["boundary"] = json!(boundary);
    thought_probe["baseline"] = baseline;
    thought_probe["outputs"] = json!({"sync": output, "stream": output});
    fixture.case_mut("serve-stop")["evidence"]["probes"] = json!([thought_probe]);
    rejected(&expected, &fixture, "final-channel stop uncovered");
    fixture.case_mut("serve-stop")["evidence"]["probes"]
        .as_array_mut()
        .unwrap()
        .push(final_probe);
    verify_model_report(&expected, &fixture.value).unwrap();
    fixture.case_mut("serve-stop")["evidence"]["probes"][1]["outputs"]["stream"]["message"]
        ["reasoning"] = json!("discarded previous thought");
    assert!(verify_model_report(&expected, &fixture.value).is_err());
}

#[test]
fn only_buffered_harmony_can_record_unavailable_run_raw_prefix_evidence() {
    let mut expected = task("selected", Backend::Metal, vec![ModelCheck::Stop]);
    expected.profile.target.protocol = ModelOutputProtocol::HarmonyGptOss;
    let mut fixture = ReportFixture::passed(&expected);
    let probe = &mut fixture.case_mut("run-stop")["evidence"]["probes"][0];
    for pointer in ["/baseline", "/outputs/run"] {
        let observation = probe.pointer_mut(pointer).unwrap();
        observation["raw_availability"] = json!("unavailable");
        observation["raw_text"] = Value::Null;
    }
    verify_model_report(&expected, &fixture.value).unwrap();
    expected.profile.target.protocol = ModelOutputProtocol::Text;
    fixture.value["target"] = json!(expected.profile.target);
    rejected(&expected, &fixture, "actual raw delta evidence");
}

#[test]
fn explicit_functional_capacity_is_bound_before_execution_and_observed_at_startup() {
    let mut expected = task("functional", Backend::Metal, vec![ModelCheck::Stop]);
    let capacity = ModelRunCapacity {
        context_tokens: 2048,
        max_num_seqs: 1,
    };
    expected.runtime_capacity = Some(capacity);
    let valid = ReportFixture::passed(&expected);
    verify_model_report(&expected, &valid.value).unwrap();
    for (field, value) in [
        ("context_tokens", json!(1024)),
        ("context_tokens", Value::Null),
        ("max_num_seqs", json!(2)),
    ] {
        let mut changed = valid.value["options"].clone();
        changed[field] = value;
        assert!(verify_model_options(&expected, &changed).is_err());
    }
    for (field, value) in [
        ("selected_max_model_len", json!(512)),
        ("selected_max_model_len", json!(4096)),
        ("selected_kv_capacity", Value::Null),
        ("selected_max_sequences", json!(2)),
    ] {
        let mut changed = ReportFixture::passed(&expected);
        changed.case_mut("serve-startup")["evidence"]["auto_config"][field] = value;
        assert!(verify_model_report(&expected, &changed.value).is_err());
    }
    for tokens in [2048, 2049] {
        expected.max_tokens = tokens;
        let fixture = ReportFixture::passed(&expected);
        assert!(verify_model_options(&expected, &fixture.value["options"]).is_err());
    }
    assert!(ModelRunCapacity {
        context_tokens: 2048,
        max_num_seqs: 0
    }
    .validate(512)
    .is_err());
}
