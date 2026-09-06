use super::*;
use ferrum_types::ModelOutputProtocol;
use serde_json::json;

fn task(id: &str, backend: Backend, checks: Vec<ModelCheck>) -> ExpectedModelRun {
    ExpectedModelRun {
        profile: ModelProfile {
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
        reasoning_alias_replay: false,
        stop_prompt: "Write an original sentence about rain.".into(),
    }
}

/// These are runner-result fixtures, not substitutes for its answer/JSON/SSE
/// oracles. The consuming runner retains tests of those actual assertions.
struct ReportFixture {
    value: Value,
}

impl ReportFixture {
    fn passed(expected: &ExpectedModelRun) -> Self {
        let ready = json!({
            "event": "ready", "requested_model": expected.profile.model,
            "backend": match expected.profile.target.backend {
                Backend::Cpu => "CPU", Backend::Metal => "Metal", Backend::Cuda => "CUDA(0)",
            }
        });
        let mut cases = vec![
            json!({"case": "binary-version", "status": "passed", "evidence": {"version": format!("ferrum {}", expected.version)}}),
            json!({"case": "serve-startup", "status": "passed", "evidence": {
                "version": expected.version, "status": "healthy",
                "auto_config": {"hardware_capabilities": {"backend": backend_name(expected.profile.target.backend)}}
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
            };
            for name in entries {
                let evidence = match *name {
                    "run-basic" | "run-stop" => json!({"ready": ready.clone()}),
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
                        let mut evidence =
                            json!({"reasoning_alias_replayed": expected.reasoning_alias_replay});
                        for (mode, id) in [("sync", "sync-call"), ("stream", "stream-call")] {
                            evidence[format!("{mode}_call")] = json!({
                                "message": {"role": "assistant", "tool_calls": [{"id": id, "type": "function", "function": {"name": "calc", "arguments": "{\"expression\":\"123+456\"}"}}]},
                                "finish_reason": "tool_calls", "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
                            });
                            evidence[format!("{mode}_continuation")] = json!({
                                "message": {"role": "assistant", "content": "579"},
                                "finish_reason": "stop", "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}, "tool_call_id": id
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

fn rejected(expected: &ExpectedModelRun, fixture: &ReportFixture, reason: &str) {
    let errors = verify_model_report(expected, &fixture.value).unwrap_err();
    assert!(
        errors.iter().any(|error| error.contains(reason)),
        "missing {reason:?} in {errors:?}"
    );
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
    rejected(&expected, &fixture, "reasoning alias replay mode");
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
        rejected(
            &expected,
            &fixture,
            &format!("completed {name} oracle evidence"),
        );

        let mut fixture = ReportFixture::passed(&expected);
        fixture.case_mut("serve-tools")["evidence"][name]["finish_reason"] = json!("length");
        rejected(
            &expected,
            &fixture,
            &format!("completed {name} oracle evidence"),
        );
    }
    for mode in ["sync", "stream"] {
        for (field, wrong) in [("id", json!("")), ("type", json!("unknown"))] {
            let mut fixture = ReportFixture::passed(&expected);
            fixture.case_mut("serve-tools")["evidence"][format!("{mode}_call")]["message"]
                ["tool_calls"][0][field] = wrong;
            rejected(
                &expected,
                &fixture,
                &format!("{mode}_call is missing its named tool identity"),
            );
        }
        let mut fixture = ReportFixture::passed(&expected);
        fixture.case_mut("serve-tools")["evidence"][format!("{mode}_call")]["message"]
            ["tool_calls"][0]["function"]["name"] = json!("lookup_weather");
        rejected(
            &expected,
            &fixture,
            &format!("{mode}_call is missing its named tool identity"),
        );

        let mut fixture = ReportFixture::passed(&expected);
        fixture.case_mut("serve-tools")["evidence"][format!("{mode}_continuation")]
            ["tool_call_id"] = json!("another-call");
        rejected(
            &expected,
            &fixture,
            &format!("{mode}_continuation did not replay its actual call identity"),
        );
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
