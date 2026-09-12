use super::*;

fn with_reasoning(text: &str) -> Fixture {
    let mut fixture = Fixture::new();
    for index in 0..2 {
        fixture.public.tool_exchanges[index].assistant["content"]
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
