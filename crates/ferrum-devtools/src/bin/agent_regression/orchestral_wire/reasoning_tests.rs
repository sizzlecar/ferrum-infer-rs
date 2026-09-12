use super::*;

fn with_reasoning(text: &str) -> Fixture {
    let mut fixture = Fixture::new();
    let journal = fixture.public.session_path.as_ref().unwrap().clone();
    let mut session: Vec<Value> = serde_json::from_slice(&fs::read(&journal).unwrap()).unwrap();
    for index in 0..2 {
        session[index + 1]["payload"]["assistant"]["content"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "type":"continuation",
                "namespace":"openai-compatible/reasoning-content/v1",
                "value":text,
            }));
        for record in fixture.records.iter_mut().skip(index + 1) {
            record.messages[2 + index * 2]["reasoning_content"] = text.into();
        }
        let path = fixture
            .dir
            .path()
            .join(format!("coding-{index}.response.sse"));
        let raw = fs::read_to_string(&path).unwrap();
        let mut frames = raw
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|line| *line != "[DONE]")
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        // Reasoning arrives both before and with the final Tool argument chunk.
        let midpoint = text
            .char_indices()
            .nth(text.chars().count() / 2)
            .map_or(text.len(), |(offset, _)| offset);
        frames[0]["choices"][0]["delta"]["reasoning"] = text[..midpoint].into();
        frames[1]["choices"][0]["delta"]["reasoning_content"] = text[midpoint..].into();
        write_frames(fixture.dir.path(), index as u32, &frames);
    }
    session.last_mut().unwrap()["payload"]["message"]["content"]
        .as_array_mut().unwrap().push(json!({
            "type":"continuation", "namespace":"openai-compatible/reasoning-content/v1", "value":"final private state",
        }));
    write_frames(
        fixture.dir.path(),
        2,
        &[
            json!({"id":"http-response-2","choices":[{"index":0,"delta":{"reasoning":"final private "}}]}),
            json!({"id":"http-response-2","choices":[{"index":0,"delta":{"reasoning_content":"state","content":"Changed and checked."},"finish_reason":"stop"}]}),
        ],
    );
    fs::write(&journal, serde_json::to_vec(&session).unwrap()).unwrap();
    fixture.public = orchestral_evidence::read(fixture.dir.path(), "session");
    assert!(fixture.public.complete(), "{:?}", fixture.public.errors);
    fixture
}

#[test]
fn reasoning_binds_raw_stream_to_durable_message_and_exact_following_history() {
    for text in ["", "  Ω\n\\n\t<tool_call>literal</tool_call>  "] {
        let fixture = with_reasoning(text);
        let evidence = fixture.bind();
        assert!(evidence.complete(), "{:?}", evidence.unproven);
        assert_eq!(evidence.receipts.len(), 2);
        assert_eq!(evidence.terminal.unwrap().successful_requests_bound, 3);
        assert_eq!(
            fixture.public.output.as_deref(),
            Some("Changed and checked.")
        );
    }
}

#[test]
fn missing_changed_or_misowned_reasoning_cannot_certify_tool_replay() {
    let mut omitted_public = with_reasoning("private");
    omitted_public.public.tool_exchanges[0].assistant["content"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(!omitted_public.bind().complete());

    let mut omitted_wire = with_reasoning("private");
    omitted_wire.records[1].messages[2]
        .as_object_mut()
        .unwrap()
        .remove("reasoning_content");
    assert!(!omitted_wire.bind().complete());

    let mut changed = with_reasoning("  private\n");
    changed.records[1].messages[2]["reasoning_content"] = "private".into();
    assert!(
        !changed.bind().complete(),
        "reasoning bytes must not be trimmed"
    );

    let mut foreign = with_reasoning("private");
    foreign.public.tool_exchanges[0].assistant["content"][2]["namespace"] = "foreign/state".into();
    assert!(!foreign.bind().complete());

    let mut duplicate = with_reasoning("private");
    let block = duplicate.public.tool_exchanges[0].assistant["content"][2].clone();
    duplicate.public.tool_exchanges[0].assistant["content"]
        .as_array_mut()
        .unwrap()
        .push(block);
    assert!(!duplicate.bind().complete());

    let mut injected = Fixture::new();
    injected.public.tool_exchanges[0].assistant["content"].as_array_mut().unwrap().push(json!({
        "type":"continuation", "namespace":"openai-compatible/reasoning-content/v1", "value":"invented",
    }));
    assert!(
        !injected.bind().complete(),
        "public metadata must originate in SSE"
    );
}

#[test]
fn malformed_conflicting_or_post_finish_reasoning_is_unproven() {
    for delta in [
        json!({"reasoning_content":12}),
        json!({"reasoning":"a","reasoning_content":"b"}),
    ] {
        let fixture = with_reasoning("private");
        write_frames(
            fixture.dir.path(),
            0,
            &[
                json!({"id":"http-response-0","choices":[{"index":0,"delta":delta,"finish_reason":"tool_calls"}]}),
            ],
        );
        assert!(!fixture.bind().complete());
    }
    let fixture = with_reasoning("private");
    let path = fixture.dir.path().join("coding-0.response.sse");
    let raw = fs::read_to_string(&path).unwrap();
    let late = json!({"id":"http-response-0","choices":[{"index":0,"delta":{"reasoning":"late"}}]});
    fs::write(
        path,
        raw.replace("data: [DONE]", &format!("data: {late}\r\n\r\ndata: [DONE]")),
    )
    .unwrap();
    assert!(!fixture.bind().complete());
}

#[test]
fn final_continuation_must_match_raw_sse_even_when_visible_delivery_is_unchanged() {
    for change in ["changed", "missing", "foreign", "duplicate", "non_string"] {
        let mut fixture = with_reasoning("tool private state");
        let journal = fixture.public.session_path.as_ref().unwrap();
        let mut session: Vec<Value> = serde_json::from_slice(&fs::read(journal).unwrap()).unwrap();
        let content = session.last_mut().unwrap()["payload"]["message"]["content"]
            .as_array_mut()
            .unwrap();
        match change {
            "changed" => content[1]["value"] = "different final private state".into(),
            "missing" => {
                content.pop();
            }
            "foreign" => content[1]["namespace"] = "other/state".into(),
            "duplicate" => content.push(content[1].clone()),
            "non_string" => content[1]["value"] = json!({"opaque":"state"}),
            _ => unreachable!(),
        }
        assert_eq!(
            content[0],
            json!({"type":"text","text":"Changed and checked."})
        );
        fs::write(journal, serde_json::to_vec(&session).unwrap()).unwrap();
        fixture.public = orchestral_evidence::read(fixture.dir.path(), "session");
        let evidence = fixture.bind();
        assert!(!evidence.complete(), "accepted {change} final continuation");
        assert!(evidence.terminal.is_none());
    }

    let fixture = with_reasoning("tool private state");
    write_frames(
        fixture.dir.path(),
        2,
        &[
            json!({"id":"http-response-2","choices":[{"index":0,"delta":{"content":"Changed and checked."},"finish_reason":"stop"}]}),
        ],
    );
    assert!(
        !fixture.bind().complete(),
        "journal reasoning needs a raw source"
    );
}

#[test]
fn saved_journals_and_wire_are_reaudited_without_changing_original_results() {
    let fixture = with_reasoning("  Ω\nprivate");
    let report = tempfile::tempdir().unwrap();
    let task_dir = report.path().join("coding");
    let journals = task_dir.join("journals");
    let requests = report.path().join("requests");
    fs::create_dir_all(&journals).unwrap();
    fs::create_dir(&requests).unwrap();
    fs::copy(
        fixture.public.run_path.as_ref().unwrap(),
        journals.join("run-saved.json"),
    )
    .unwrap();
    fs::copy(
        fixture.public.session_path.as_ref().unwrap(),
        journals.join("session-saved.json"),
    )
    .unwrap();
    let original = json!({"id":"coding", "orchestral":{"session_id":"session"},
        "completed":false, "validation":{"exit_code":0}})
    .to_string();
    fs::write(task_dir.join("result.json"), &original).unwrap();
    fs::write(
        report.path().join("manifest.json"),
        json!({"schema_version":2,
        "agent":{"kind":"orchestral","tool_result_format":"json"}})
        .to_string(),
    )
    .unwrap();
    for record in &fixture.records {
        let stem = format!("coding-{}", record.request_index);
        fs::write(
            requests.join(format!("{stem}.json")),
            serde_json::to_vec(record).unwrap(),
        )
        .unwrap();
        fs::write(
            requests.join(format!("{stem}.request.json")),
            json!({"messages":record.messages}).to_string(),
        )
        .unwrap();
        fs::copy(
            fixture.dir.path().join(format!("{stem}.response.sse")),
            requests.join(format!("{stem}.response.sse")),
        )
        .unwrap();
    }
    let output = report.path().join("audit.json");
    assert_eq!(audit_saved(report.path(), "coding", &output).unwrap(), 0);
    let audit: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(audit["closed_loop_evidence"], true);
    assert_eq!(audit["public"]["output"], "Changed and checked.");
    assert_eq!(audit["wire"]["receipts"].as_array().unwrap().len(), 2);
    assert!(audit_saved(report.path(), "coding", &task_dir.join("result.json")).is_err());
    assert_eq!(
        fs::read_to_string(task_dir.join("result.json")).unwrap(),
        original
    );
    let replay = requests.join("coding-1.request.json");
    let mut body: Value = serde_json::from_slice(&fs::read(&replay).unwrap()).unwrap();
    body["messages"][2]
        .as_object_mut()
        .unwrap()
        .remove("reasoning_content");
    fs::write(replay, body.to_string()).unwrap();
    assert_eq!(
        audit_saved(
            report.path(),
            "coding",
            &report.path().join("changed-audit.json")
        )
        .unwrap(),
        1
    );
}
