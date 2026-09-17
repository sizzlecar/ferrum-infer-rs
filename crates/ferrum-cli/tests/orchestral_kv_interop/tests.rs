use super::*;
use std::collections::BTreeSet;

fn exchange(facts: &Value) -> Value {
    json!({"payload": {"type": "tool_exchange_committed", "assistant": {"content": [
        {"type": "tool_call", "call_id": "call-1", "name": "file_read", "arguments": {"path": DATA_FILE}}
    ]}, "tool": {"content": [{"type": "tool_result", "call_id": "call-1", "is_error": false,
        "result": {"path": DATA_FILE, "content": serde_json::to_string(facts).unwrap()}}]}}})
}

#[test]
fn exact_semantic_values_allow_prose_but_reject_substrings_and_wrong_values() {
    let expected = json!({"tracking_code": "cargo-012abc", "parcel_count": 17});
    journal::validate_answer(
        "The tracking code is cargo-012abc, and there are 17 parcels.",
        &expected,
    )
    .unwrap();
    journal::validate_answer(
        "{\"tracking_code\":\"cargo-012abc\",\"parcel_count\":17}",
        &expected,
    )
    .unwrap();
    for text in [
        "cargo-012abc has 117 parcels",
        "prefix-cargo-012abc has 17 parcels",
        "cargo-012abc-extra has 17 parcels",
        "cargo-012abd has 17 parcels",
    ] {
        assert!(journal::validate_answer(text, &expected).is_err(), "{text}");
    }
}

#[test]
fn tool_read_requires_native_identity_success_and_actual_file_content() {
    let facts = json!({"tracking_code": "cargo-abc", "parcel_count": 17});
    let mut snapshot = journal::Snapshot {
        session: vec![exchange(&facts)],
        ..Default::default()
    };
    snapshot.validate_file_read(DATA_FILE, &facts).unwrap();
    snapshot.session[0]["payload"]["tool"]["content"][0]["call_id"] = json!("different-call");
    assert!(snapshot.validate_file_read(DATA_FILE, &facts).is_err());
    snapshot.session[0] = exchange(&facts);
    snapshot.session[0]["payload"]["tool"]["content"][0]["is_error"] = json!(true);
    assert!(snapshot.validate_file_read(DATA_FILE, &facts).is_err());
    snapshot.session[0] = exchange(&facts);
    assert!(snapshot
        .validate_file_read(DATA_FILE, &json!({"tracking_code": "wrong"}))
        .is_err());
    snapshot.session = vec![
        json!({"payload": {"type": "run_output_committed", "message": "I called file_read and got cargo-abc, 17"}}),
    ];
    assert!(snapshot.validate_file_read(DATA_FILE, &facts).is_err());
}

fn delivery(text: &str) -> Value {
    json!({"event": {"run_id": "run-1", "payload": {"type": "delivery_committed", "delivery": {
        "final_response": {"media_type": "text/plain", "body": {"kind": "inline", "value": text}}
    }}}})
}

fn write_journal_fixture(directory: &Path) {
    fs::create_dir_all(directory).unwrap();
    fs::write(
        directory.join("session-fixture.json"),
        serde_json::to_vec(&vec![json!({
            "session_id": SESSION, "session_seq": 1,
            "payload": {"type": "run_input_committed"}
        })])
        .unwrap(),
    )
    .unwrap();
    fs::write(
        directory.join("run-fixture.json"),
        serde_json::to_vec(&json!({"run": {"records": [delivery("actual delivery")]}})).unwrap(),
    )
    .unwrap();
}

#[test]
fn isolated_journal_reader_accepts_flat_and_exact_session_shard_layouts() {
    for sharded in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let directory = if sharded {
            journal::session_directory(root.path())
        } else {
            root.path().to_path_buf()
        };
        write_journal_fixture(&directory);
        // An unrelated shard is never opened, even if its files are invalid.
        let unrelated = root.path().join("sessions/unrelated");
        fs::create_dir_all(&unrelated).unwrap();
        fs::write(unrelated.join("session-private.json"), "not JSON").unwrap();
        let snapshot = journal::Snapshot::read(root.path()).unwrap();
        assert_eq!(snapshot.session.len(), 1);
        assert_eq!(
            snapshot.new_delivery(&BTreeSet::new()).unwrap(),
            "actual delivery"
        );
    }
}

#[test]
fn isolated_journal_reader_rejects_mixed_layouts_and_wrong_session_identity() {
    let root = tempfile::tempdir().unwrap();
    let shard = journal::session_directory(root.path());
    write_journal_fixture(&shard);
    fs::write(
        shard.join("session-fixture.json"),
        serde_json::to_vec(&vec![json!({
            "session_id": "another-session", "session_seq": 1,
            "payload": {"type": "run_input_committed"}
        })])
        .unwrap(),
    )
    .unwrap();
    assert!(journal::Snapshot::read(root.path()).is_err());
    write_journal_fixture(&shard);
    write_journal_fixture(root.path());
    fs::remove_file(shard.join("run-fixture.json")).unwrap();
    fs::remove_file(root.path().join("session-fixture.json")).unwrap();
    assert!(journal::Snapshot::read(root.path())
        .err()
        .expect("a run and its session must share a storage layout")
        .to_string()
        .contains("multiple storage layouts"));
}

#[cfg(unix)]
#[test]
fn isolated_journal_reader_rejects_symlinked_layout_components_and_files() {
    use std::os::unix::fs::symlink;
    for component in ["root", "sessions", "shard", "file"] {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("journals");
        let outside = temporary.path().join("outside");
        write_journal_fixture(&outside);
        match component {
            "root" => symlink(&outside, &root).unwrap(),
            "sessions" => {
                fs::create_dir(&root).unwrap();
                symlink(&outside, root.join("sessions")).unwrap();
            }
            "shard" => {
                fs::create_dir_all(root.join("sessions")).unwrap();
                symlink(&outside, journal::session_directory(&root)).unwrap();
            }
            "file" => {
                let shard = journal::session_directory(&root);
                fs::create_dir_all(&shard).unwrap();
                symlink(
                    outside.join("session-fixture.json"),
                    shard.join("session-link.json"),
                )
                .unwrap();
            }
            _ => unreachable!(),
        }
        assert!(journal::Snapshot::read(&root)
            .err()
            .expect("symlink must be rejected")
            .to_string()
            .contains("symlink"));
    }
}

#[test]
fn final_delivery_rejects_partial_failure_and_duplicate_terminals() {
    let mut snapshot = journal::Snapshot::default();
    snapshot
        .runs
        .insert("run-1".into(), vec![delivery("cargo-abc")]);
    assert_eq!(
        snapshot.new_delivery(&BTreeSet::new()).unwrap(),
        "cargo-abc"
    );
    assert!(snapshot
        .new_delivery(&BTreeSet::from(["run-1".into()]))
        .is_err());
    snapshot
        .runs
        .get_mut("run-1")
        .unwrap()
        .push(delivery("cargo-abc"));
    assert!(snapshot.new_delivery(&BTreeSet::new()).is_err());
    snapshot.runs.insert(
        "run-1".into(),
        vec![json!({"event": {"run_id": "run-1", "payload": {"type": "run_incomplete"}}})],
    );
    assert!(snapshot.new_delivery(&BTreeSet::new()).is_err());
}

#[test]
fn actual_kv_resolution_and_failure_counters_are_required() {
    let format = ferrum_types::KvStorageFormat::Int8PerTokenHeadF32ScaleV1;
    let mut health = json!({"status": "healthy", "kv_storage": {
        "source": "resolved_model_plan", "requested": format, "selected": format
    }, "scheduler": {"failed_requests": 0}});
    health_evidence(&health, "int8").unwrap();
    assert!(health_evidence(&health, "fp16").is_err());
    health["scheduler"]["failed_requests"] = json!(1);
    assert!(health_evidence(&health, "int8").is_err());
    health["scheduler"]["failed_requests"] = Value::Null;
    assert!(health_evidence(&health, "int8").is_err());
}

#[test]
fn paired_commands_share_model_visible_workspace_and_recreate_facts() {
    let temporary = tempfile::tempdir().unwrap();
    let config: Configuration = serde_json::from_value(json!({
        "ferrum_bin": "/unused/ferrum", "orchestral_bin": "/unused/orchestral",
        "model": "/unused/model", "backend": "metal", "report_dir": temporary.path(),
        "context_tokens": 8192, "max_tokens": 512
    }))
    .unwrap();
    let facts = Facts::new();
    let sandbox = prepare_sandbox(&config, &facts).unwrap();
    let original = fs::read(sandbox.join(DATA_FILE)).unwrap();
    for dtype in ["fp16", "int8"] {
        let directory = config.report_dir.join(dtype);
        fs::create_dir(&directory).unwrap();
        let command = config.orchestral_command("http://127.0.0.1:1/v1", &directory, "same prompt");
        assert_eq!(command.get_current_dir(), Some(sandbox.as_path()));
        let args: Vec<_> = command.get_args().collect();
        let cwd_index = args.iter().position(|arg| *arg == "--cwd").unwrap();
        assert_eq!(args[cwd_index + 1], sandbox.as_os_str());
        assert_eq!(fs::read(sandbox.join(DATA_FILE)).unwrap(), original);
        // A failed agent could leave files behind: retain them as evidence,
        // but do not expose them to the other format's initial workspace.
        fs::write(sandbox.join("unexpected.txt"), dtype).unwrap();
        fs::rename(&sandbox, directory.join("sandbox-after")).unwrap();
        assert!(directory.join("sandbox-after/unexpected.txt").exists());
        if dtype == "fp16" {
            assert_eq!(prepare_sandbox(&config, &facts).unwrap(), sandbox);
            assert!(!sandbox.join("unexpected.txt").exists());
        }
    }
}

fn request_bundle(root: &Path, id: &str, generated_history: bool) -> PathBuf {
    let directory = root.join(id);
    fs::create_dir_all(&directory).unwrap();
    let mut messages = vec![
        json!({"role":"system", "content":"[redacted]", "content_chars":24}),
        json!({"role":"user", "content":"[redacted]", "content_chars":18}),
    ];
    if generated_history {
        messages.push(json!({"role":"assistant", "content":"[redacted]"}));
    }
    let body =
        json!({"model": MODEL_ALIAS, "messages": messages, "temperature":0.0, "stream":true});
    let schema = ferrum_types::OBSERVABILITY_PROFILE_SCHEMA_VERSION;
    write_json(directory.join("replay_body.json"), &body).unwrap();
    write_json(
        directory.join("request.json"),
        &json!({
            "schema_version":schema, "request_id":id, "entrypoint":"serve", "method":"POST",
            "endpoint":"/v1/chat/completions", "http":{"body":body}
        }),
    )
    .unwrap();
    write_json(
        directory.join("prompt_token_ids.json"),
        &json!({
            "schema_version":schema, "request_id":id, "token_ids":[11,22,33],
            "token_count":3, "unavailable_reason":null
        }),
    )
    .unwrap();
    write_json(
        directory.join("sampling_params.json"),
        &json!({
            "schema_version":schema, "request_id":id,
            "sampling_params":{"temperature":0.0, "max_tokens":512}, "unavailable_reason":null
        }),
    )
    .unwrap();
    directory
}

fn startup_bundle(root: &Path, id: &str) -> PathBuf {
    let directory = root.join(id);
    fs::create_dir_all(&directory).unwrap();
    let schema = ferrum_types::OBSERVABILITY_PROFILE_SCHEMA_VERSION;
    let request = json!({
        "schema_version":schema, "request_id":id, "entrypoint":"serve",
        "backend":"actual", "actual_model_smoke":true, "sanitized":true,
        "replay_command":"fixture serve command"
    });
    for path in [directory.join("request.json"), root.join("request.json")] {
        write_json(path, &request).unwrap();
    }
    fs::write(root.join("replay_command.txt"), "fixture serve command\n").unwrap();
    write_json(
        directory.join("prompt_token_ids.json"),
        &json!({
            "schema_version":schema, "request_id":id, "token_ids":null,
            "token_count":null,
            "unavailable_reason":"startup or non-run request has no rendered prompt token dump in WP9 L0"
        }),
    )
    .unwrap();
    write_json(
        directory.join("sampling_params.json"),
        &json!({
            "schema_version":schema, "request_id":id, "sampling_params":null,
            "unavailable_reason":"sampling params unavailable for this replay bundle kind in WP9 L0"
        }),
    )
    .unwrap();
    write_json(
        directory.join("output_token_ids.json"),
        &json!({
            "schema_version":schema, "request_id":id, "token_ids":[],
            "token_count":0, "finish_reason":null
        }),
    )
    .unwrap();
    directory
}

#[test]
fn initial_model_request_comparison_accepts_startup_bundles_and_root_aliases() {
    let temporary = tempfile::tempdir().unwrap();
    for dtype in ["fp16", "int8"] {
        let root = temporary.path().join(dtype).join("model-requests");
        // Neither directory has a role-bearing name. Only its product evidence
        // may distinguish startup from a real inference request.
        startup_bundle(&root, "observation");
        request_bundle(&root, "inference", false);
    }
    let comparison = request_capture::compare_first_requests(temporary.path()).unwrap();
    assert_eq!(comparison["equal"], true);
    assert_eq!(comparison["fp16"]["prompt_token_count"], 3);
}

#[test]
fn initial_model_request_comparison_rejects_malformed_startup_and_http_evidence() {
    let temporary = tempfile::tempdir().unwrap();
    let mut startup = PathBuf::new();
    for dtype in ["fp16", "int8"] {
        let root = temporary.path().join(dtype).join("model-requests");
        startup = startup_bundle(&root, "observation");
        request_bundle(&root, "inference", false);
    }
    let path = startup.join("request.json");
    let original: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    // An HTTP record cannot evade strict validation by carrying otherwise
    // valid startup metadata, including an explicitly null HTTP field.
    for (field, value) in [("method", json!("GET")), ("http", Value::Null)] {
        let mut malformed = original.clone();
        malformed[field] = value;
        write_json(&path, &malformed).unwrap();
        let error = request_capture::compare_first_requests(temporary.path()).unwrap_err();
        assert!(error.to_string().contains("HTTP request-dump"), "{error:#}");
    }
    write_json(&path, &original).unwrap();
    let tokens_path = startup.join("prompt_token_ids.json");
    let mut tokens: Value = serde_json::from_slice(&fs::read(&tokens_path).unwrap()).unwrap();
    tokens["token_ids"] = json!([11]);
    tokens["token_count"] = json!(1);
    write_json(tokens_path, &tokens).unwrap();
    let error = request_capture::compare_first_requests(temporary.path()).unwrap_err();
    assert!(error.to_string().contains("startup evidence"), "{error:#}");
}

#[test]
fn initial_model_request_comparison_validates_root_aliases() {
    let temporary = tempfile::tempdir().unwrap();
    let mut root = PathBuf::new();
    for dtype in ["fp16", "int8"] {
        root = temporary.path().join(dtype).join("model-requests");
        startup_bundle(&root, "observation");
        request_bundle(&root, "inference", false);
    }
    let command_path = root.join("replay_command.txt");
    fs::write(&command_path, "different command\n").unwrap();
    let error = request_capture::compare_first_requests(temporary.path()).unwrap_err();
    assert!(
        error.to_string().contains("replay command alias"),
        "{error:#}"
    );
    fs::remove_file(command_path).unwrap();
    let error = request_capture::compare_first_requests(temporary.path()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("incomplete request-dump root aliases"),
        "{error:#}"
    );
}

#[test]
fn initial_model_request_comparison_uses_actual_tokens_and_effective_sampling() {
    let temporary = tempfile::tempdir().unwrap();
    let mut candidate = PathBuf::new();
    for dtype in ["fp16", "int8"] {
        let root = temporary.path().join(dtype).join("model-requests");
        // Directory sorting would choose the later request; only the message
        // schema determines which input preceded all generated history.
        request_bundle(&root, "a-later", true);
        candidate = request_bundle(&root, "z-first", false);
    }
    let comparison = request_capture::compare_first_requests(temporary.path()).unwrap();
    assert_eq!(comparison["equal"], true);
    assert_eq!(
        comparison["fp16"]["prompt_token_ids_sha256_u32_le"],
        comparison["int8"]["prompt_token_ids_sha256_u32_le"]
    );
    let path = candidate.join("prompt_token_ids.json");
    let mut tokens: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    // Sanitized text can have equal length while the actual system/workspace
    // token differs. The token comparison must still reject this pair.
    tokens["token_ids"][1] = json!(44);
    write_json(&path, &tokens).unwrap();
    assert_eq!(
        request_capture::compare_first_requests(temporary.path()).unwrap()["equal"],
        false
    );
    tokens["token_ids"][1] = json!(22);
    write_json(&path, &tokens).unwrap();
    let path = candidate.join("sampling_params.json");
    let mut sampling: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    sampling["sampling_params"]["temperature"] = json!(0.5);
    write_json(&path, &sampling).unwrap();
    assert_eq!(
        request_capture::compare_first_requests(temporary.path()).unwrap()["equal"],
        false
    );
}

#[test]
fn initial_model_request_comparison_rejects_ambiguous_or_incomplete_evidence() {
    let temporary = tempfile::tempdir().unwrap();
    let mut candidate = PathBuf::new();
    for dtype in ["fp16", "int8"] {
        let root = temporary.path().join(dtype).join("model-requests");
        candidate = request_bundle(&root, "first", false);
    }
    let tokens_path = candidate.join("prompt_token_ids.json");
    let original: Value = serde_json::from_slice(&fs::read(&tokens_path).unwrap()).unwrap();
    for (key, wrong) in [
        ("token_ids", Value::Null),
        ("token_count", json!(2)),
        ("unavailable_reason", json!("not retained")),
        ("request_id", json!("wrong")),
        ("schema_version", json!(999)),
    ] {
        let mut mutated = original.clone();
        mutated[key] = wrong;
        write_json(&tokens_path, &mutated).unwrap();
        assert!(
            request_capture::compare_first_requests(temporary.path()).is_err(),
            "{key}"
        );
    }
    write_json(&tokens_path, &original).unwrap();
    let duplicate = request_bundle(candidate.parent().unwrap(), "ambiguous", false);
    assert!(request_capture::compare_first_requests(temporary.path()).is_err());
    fs::remove_dir_all(duplicate).unwrap();
    fs::remove_file(tokens_path).unwrap();
    assert!(request_capture::compare_first_requests(temporary.path()).is_err());
    fs::remove_dir_all(candidate).unwrap();
    assert!(request_capture::compare_first_requests(temporary.path()).is_err());
}
