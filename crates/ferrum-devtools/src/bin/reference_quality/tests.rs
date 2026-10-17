use super::*;
use input::{Case, Policy, Sample, Task};

fn policy() -> Policy {
    Policy {
        context_tokens: 1024,
        template_reserve_tokens: 128,
        math_max_tokens: 512,
        choice_max_tokens: 2,
        generation_seed: 42,
        concurrency: 2,
        timeout_secs: 30,
    }
}
fn sample() -> Sample {
    Sample {
        case: Case {
            id: "subject/example".into(),
            stratum: "subject".into(),
            source_record_index: 7,
            task: Task::Choice,
            prompt: "Choose an answer".into(),
            prompt_sha256: sha(b"Choose an answer"),
            public_input_tokens: 8,
        },
        prompt: vec![0, 19, 20],
        request_id: "original-request".into(),
        prompt_dump_sha256: "a".repeat(64),
        response_sha256: "b".repeat(64),
        answer: "C".into(),
    }
}
fn response() -> Value {
    json!({"content":"C","tokens":[42,99],"tokens_predicted":2,
    "tokens_evaluated":3,"stop":true,"truncated":false,"stop_type":"eos",
    "prompt_token_ids":[0,19,20]})
}

struct Fixture {
    _dir: tempfile::TempDir,
    plan: PathBuf,
    hash: String,
    strict: PathBuf,
    answers: PathBuf,
}
fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let strict = dir.path().join("strict");
    fs::create_dir(&strict).unwrap();
    let mut cases = vec![sample().case];
    let mut second = cases[0].clone();
    second.id = "another-subject/example".into();
    second.stratum = "another-subject".into();
    // Record indexes are local to each dataset stratum, not globally unique.
    cases.push(second);
    let case_file = dir.path().join("cases.json");
    save(&case_file, &cases).unwrap();
    let answers = dir.path().join("answers.json");
    save(
        &answers,
        &json!([{"id":cases[0].id,"answer":"C"},{"id":cases[1].id,"answer":"C"}]),
    )
    .unwrap();
    let p = json!({"schema":1,"frozen":true,"sample_count":2,
        "sampling":"sha256_rank_without_replacement_v1; eligible_subject_proportional_largest_remainder; pooled_dispatch_hash_shuffle",
        "scoring":"final_marker_exact_rational_v1; trimmed_uppercase_letter_v1; length_is_wrong; protocol_invalid_blocks_gate",
        "cases":{"path":case_file,"sha256":sha(&fs::read(&case_file).unwrap())},
        "answers_sha256":sha(&fs::read(&answers).unwrap()),"policy":policy(),
        "execution":{"weights_sha256":"1".repeat(64),"model_alias":"reference-test",
            "server_capacity":2,"strict_runtime_sha256":"2".repeat(64),"collector_binary_sha256":"3".repeat(64)}});
    let plan = dir.path().join("frozen-plan.json");
    save(&plan, &p).unwrap();
    let hash = sha(&fs::read(&plan).unwrap());
    save(
        &strict.join("manifest.json"),
        &json!({"schema":1,"arm":"strict","execute":true,
        "plan":p,"plan_sha256":hash,"concurrency":2,"runtime_sha256":"2".repeat(64),
        "collector_binary_sha256":"3".repeat(64)}),
    )
    .unwrap();
    let mut results = vec![];
    for (i, c) in cases.iter().enumerate() {
        let root = strict.join(format!("sample-{i:03}"));
        fs::create_dir(&root).unwrap();
        let id = format!("request-{i}");
        let result = json!({"success":true,"sample_index":i,"source_record_index":7,
            "response_id":id,"evidence":{"request_id":id},"completion_ordinal":1-i});
        save(&root.join("result.json"), &result).unwrap();
        results.push(result);
        save(&root.join("source-case.json"), c).unwrap();
        let request = json!({"model":"reference-test","messages":[{"role":"user","content":c.prompt}],
            "stream":false,"max_tokens":2,"temperature":0,"top_p":1,"repetition_penalty":1,"repeat_penalty":1,
            "seed":42,"ignore_eos":false,"stop":[],"chat_template_kwargs":{"enable_thinking":false},
            "return_tokens":true,"verbose":true});
        save(&root.join("request.json"), &request).unwrap();
        save(&root.join("request.body"), &request).unwrap();
        save(&root.join("http.json"), &json!({"status":200})).unwrap();
        save(
            &root.join("response.body"),
            &json!({"id":id,"usage":{"prompt_tokens":3}}),
        )
        .unwrap();
        save(
            &root.join("server-prompt-dump.json"),
            &json!({"request_id":id,"token_count":3,
            "token_ids":[0,19,20],"unavailable_reason":null}),
        )
        .unwrap();
        save(
            &root.join("prompt-tokens.json"),
            &json!({"request_id":id,"token_ids":[0,19,20],
            "source":"server_dump/prompt_token_ids.json","status":"captured"}),
        )
        .unwrap();
    }
    save(&strict.join("results.json"), &results).unwrap();
    Fixture {
        _dir: dir,
        plan,
        hash,
        strict,
        answers,
    }
}
fn replace_field(path: &Path, key: &str, value: Value) {
    let mut json: Value = read_json(path).unwrap();
    json[key] = value;
    fs::write(path, serde_json::to_vec(&json).unwrap()).unwrap();
}

#[test]
fn frozen_capture_uses_case_order_and_exact_source_response_identity() {
    let f = fixture();
    let source = input::load(&f.plan, &f.hash, &f.strict, &f.answers).unwrap();
    assert_eq!(source.samples.len(), 2);
    assert_eq!(source.samples[0].request_id, "request-0");
    assert_eq!(source.samples[1].case.id, "another-subject/example");
    assert_eq!(
        source.samples[0].case.source_record_index,
        source.samples[1].case.source_record_index
    );
    replace_field(
        &f.strict.join("sample-000/source-case.json"),
        "id",
        json!("another-subject/example"),
    );
    assert!(input::load(&f.plan, &f.hash, &f.strict, &f.answers).is_err());
}

#[test]
fn frozen_capture_rejects_changed_prompt_dump_count_identity_and_file_pin() {
    for (file, key, value) in [
        ("server-prompt-dump.json", "request_id", json!("request-1")),
        (
            "server-prompt-dump.json",
            "unavailable_reason",
            json!("unavailable"),
        ),
        ("server-prompt-dump.json", "token_count", json!(2)),
        ("prompt-tokens.json", "token_ids", json!([0, 19, 21])),
        ("response.body", "usage", json!({"prompt_tokens":2})),
        ("request.json", "ignore_eos", json!(true)),
    ] {
        let f = fixture();
        replace_field(&f.strict.join("sample-000").join(file), key, value);
        assert!(
            input::load(&f.plan, &f.hash, &f.strict, &f.answers).is_err(),
            "{file}/{key}"
        );
    }
    let f = fixture();
    fs::write(&f.answers, b"[]").unwrap();
    assert!(input::load(&f.plan, &f.hash, &f.strict, &f.answers).is_err());
}

fn guard() -> Value {
    json!({"serving_cells":[{"port":19082,"binary_sha256":"4".repeat(64),
    "client_sha256":"5".repeat(64),"args":["--model","/models/weights.gguf","--parallel","2",
        "--ctx-size","2048","--kv-unified-per-slot","1024","--host","127.0.0.1","--port","19082"]}],
    "input_files":[{"path":"/models/weights.gguf","sha256":"1".repeat(64)}]})
}

#[test]
fn reference_guard_binds_same_weights_capacity_endpoint_and_collector() {
    let f = fixture();
    let source = input::load(&f.plan, &f.hash, &f.strict, &f.answers).unwrap();
    let url = reqwest::Url::parse("http://127.0.0.1:19082/").unwrap();
    assert_eq!(
        input::validate_reference(&guard(), &source.plan, &url, &"5".repeat(64)).unwrap(),
        "http://127.0.0.1:19082/completion"
    );
    for (pointer, value) in [
        ("/input_files/0/sha256", json!("6".repeat(64))),
        ("/serving_cells/0/client_sha256", json!("6".repeat(64))),
        ("/serving_cells/0/args/3", json!("1")),
        ("/serving_cells/0/args/5", json!("4096")),
    ] {
        let mut changed = guard();
        *changed.pointer_mut(pointer).unwrap() = value;
        assert!(input::validate_reference(&changed, &source.plan, &url, &"5".repeat(64)).is_err());
    }
    for url in [
        "http://127.0.0.1:19083/",
        "http://127.0.0.1:19082/v1/",
        "https://127.0.0.1:19082/",
    ] {
        assert!(input::validate_reference(
            &guard(),
            &source.plan,
            &reqwest::Url::parse(url).unwrap(),
            &"5".repeat(64)
        )
        .is_err());
    }
}

#[test]
fn guard_runtime_pin_excludes_only_client_arguments_and_has_no_self_reference() {
    let original = guard();
    let pin = input::guard_runtime_pin(&original).unwrap();
    let reordered: Value = serde_json::from_str(&format!(
        "{{\"serving_cells\":{},\"input_files\":{}}}",
        original["serving_cells"], original["input_files"]
    ))
    .unwrap();
    assert_eq!(input::guard_runtime_pin(&reordered).unwrap(), pin);
    let mut bound = original.clone();
    bound["serving_cells"][0]["client_args"] = json!(["run", "--guard-runtime-sha256", pin]);
    assert_eq!(input::guard_runtime_pin(&bound).unwrap(), pin);
    assert_ne!(
        sha(&serde_json::to_vec(&bound).unwrap()),
        sha(&serde_json::to_vec(&original).unwrap())
    );
}

#[test]
fn guard_runtime_pin_changes_with_server_model_environment_and_executable_pins() {
    let original = guard();
    let pin = input::guard_runtime_pin(&original).unwrap();
    for (pointer, value) in [
        ("/input_files/0/sha256", json!("6".repeat(64))),
        ("/serving_cells/0/binary_sha256", json!("6".repeat(64))),
        ("/serving_cells/0/client_sha256", json!("6".repeat(64))),
        ("/serving_cells/0/args/3", json!("1")),
    ] {
        let mut changed = original.clone();
        *changed.pointer_mut(pointer).unwrap() = value;
        assert_ne!(
            input::guard_runtime_pin(&changed).unwrap(),
            pin,
            "{pointer}"
        );
    }
    let mut changed = original;
    changed["serving_cells"][0]["env"] = json!({"KEY":"value"});
    assert_ne!(input::guard_runtime_pin(&changed).unwrap(), pin);
}

#[test]
fn native_reply_preserves_terminal_ids_and_distinguishes_token_limit_from_invalid_context() {
    let s = sample();
    let p = policy();
    let valid = response();
    let observed = capture::observation(&valid, &s, &p).unwrap();
    assert_eq!(observed.tokens, [42, 99]);
    assert!(observed.prompt_confirmed);
    let mut limited = valid.clone();
    limited["stop_type"] = json!("limit");
    let observed = capture::observation(&limited, &s, &p).unwrap();
    assert_eq!(observed.finish, "length");
    assert!(!score::grade(
        &Task::Choice,
        &observed.text,
        &observed.finish,
        "C"
    ));
    for (key, value) in [
        ("tokens", json!([42, -1])),
        ("tokens", json!([42, 4294967296_u64])),
        ("tokens", json!([42, 1.5])),
        ("tokens_predicted", json!(1)),
        ("tokens_evaluated", json!(2)),
        ("prompt_token_ids", json!([0, 19, 21])),
        ("truncated", json!(true)),
        ("stop", json!(false)),
        ("stop_type", json!("unknown")),
        ("stop_type", json!("word")),
        ("tokens", json!([42, 99, 100])),
    ] {
        let mut changed = valid.clone();
        changed[key] = value;
        assert!(capture::observation(&changed, &s, &p).is_err(), "{key}");
    }
    let mut no_response_ids = valid;
    no_response_ids
        .as_object_mut()
        .unwrap()
        .remove("prompt_token_ids");
    assert!(
        !capture::observation(&no_response_ids, &s, &p)
            .unwrap()
            .prompt_confirmed
    );
}

#[test]
fn standard_answer_scoring_matches_frozen_exact_rules() {
    for text in ["Work\n#### 0.5", "#### 1/2", "\n#### 1,000/2,000\n"] {
        assert!(score::grade(&Task::Math, text, "stop", "1/2"));
    }
    for text in [
        "#### 0.5\nmore",
        "#### 0.5\n#### 0.5",
        "#### 1/0",
        "#### 12,34",
    ] {
        assert!(!score::grade(&Task::Math, text, "stop", "1/2"));
    }
    assert!(score::grade(&Task::Choice, " C\n", "stop", "C"));
    for text in ["c", "C.", "The answer is C"] {
        assert!(!score::grade(&Task::Choice, text, "stop", "C"));
    }
    assert!(!score::grade(&Task::Math, "#### 0.5", "length", "1/2"));
}

#[tokio::test]
async fn real_http_capture_retains_success_and_failed_raw_evidence() {
    use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
    use std::sync::{Arc, Mutex};
    #[derive(Clone)]
    struct StateData {
        status: StatusCode,
        body: Value,
        seen: Arc<Mutex<Vec<Value>>>,
    }
    async fn reply(
        State(s): State<StateData>,
        Json(request): Json<Value>,
    ) -> (StatusCode, Json<Value>) {
        s.seen.lock().unwrap().push(request);
        (s.status, Json(s.body))
    }
    for (status, body, valid) in [
        (StatusCode::OK, response(), true),
        (
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"error":"busy"}),
            false,
        ),
        (
            StatusCode::OK,
            json!({"stop":true,"truncated":false,"tokens_evaluated":3,"tokens":null}),
            false,
        ),
    ] {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/completion", post(reply))
            .with_state(StateData {
                status,
                body: body.clone(),
                seen: seen.clone(),
            });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/completion", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let dir = tempfile::tempdir().unwrap();
        let s = sample();
        let p = policy();
        let request = capture::request(&p, &s);
        let result = capture::collect(
            &reqwest::Client::new(),
            &endpoint,
            serde_json::to_vec(&request).unwrap(),
            dir.path(),
            &s,
            &p,
        )
        .await;
        assert_eq!(result.is_ok(), valid);
        assert_eq!(seen.lock().unwrap().as_slice(), [request]);
        assert_eq!(
            read_json::<Value>(&dir.path().join("response.body")).unwrap(),
            body
        );
        assert_eq!(
            read_json::<Value>(&dir.path().join("http.json")).unwrap()["status"],
            status.as_u16()
        );
        if valid {
            assert_eq!(
                fs::read_to_string(dir.path().join("visible-text.txt")).unwrap(),
                "C"
            );
        }
        server.abort();
    }
}
