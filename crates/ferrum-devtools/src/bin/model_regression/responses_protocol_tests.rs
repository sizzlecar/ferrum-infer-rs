use super::*;

fn thought() -> Value {
    json!({"id": "rs-1", "type": "reasoning", "status": "completed", "summary": [],
        "encrypted_content": null,
        "content": [{"type": "reasoning_text", "text": "用工具计算 123+456。"}]})
}

fn message(text: &str) -> Value {
    json!({"id": "msg-1", "type": "message", "status": "completed", "role": "assistant",
        "phase": "final_answer",
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}]})
}

fn call() -> Value {
    json!({"id": "fc-1", "type": "function_call", "status": "completed", "call_id": "call-1",
        "name": "calc", "arguments": "{\"expression\":\"123+456\"}"})
}

fn response(output: Vec<Value>) -> Value {
    json!({"id": "resp-1", "object": "response", "status": "completed", "model": "fixture",
        "created_at": 1, "completed_at": 2, "error": null, "incomplete_details": null,
        "tool_choice": "auto", "text": {"format": {"type": "json_schema"}},
        "output": output,
        "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}})
}

// A wire fixture expresses a complete caller-visible lifecycle, including
// fragmented Unicode and arguments. Mutations below alter independent pieces
// of that evidence, rather than obtaining expectations from the parser.
fn events(response: &Value) -> Vec<Value> {
    let mut initial = response.clone();
    initial["status"] = json!("in_progress");
    initial["completed_at"] = Value::Null;
    initial["usage"] = Value::Null;
    initial["output"] = json!([]);
    let mut result = vec![
        json!({"type": "response.created", "response": initial}),
        json!({"type": "response.in_progress", "response": initial}),
    ];
    for (output_index, item) in response["output"].as_array().unwrap().iter().enumerate() {
        let mut added = item.clone();
        added["status"] = json!("in_progress");
        added.as_object_mut().unwrap().remove("phase");
        let function = item["type"] == "function_call";
        added[if function { "arguments" } else { "content" }] =
            if function { json!("") } else { json!([]) };
        result.push(json!({"type": "response.output_item.added", "output_index": output_index, "item": added}));
        if function {
            for character in item["arguments"].as_str().unwrap().chars() {
                result.push(
                    json!({"type": "response.function_call_arguments.delta", "item_id": item["id"],
                    "output_index": output_index, "delta": character.to_string()}),
                );
            }
            result.push(json!({"type": "response.function_call_arguments.done", "item_id": item["id"],
                "output_index": output_index, "call_id": item["call_id"], "name": item["name"], "arguments": item["arguments"]}));
        } else {
            for (content_index, part) in item["content"].as_array().unwrap().iter().enumerate() {
                let mut empty = part.clone();
                empty["text"] = json!("");
                result.push(
                    json!({"type": "response.content_part.added", "item_id": item["id"],
                    "output_index": output_index, "content_index": content_index, "part": empty}),
                );
                let channel = part["type"].as_str().unwrap();
                for character in part["text"].as_str().unwrap().chars() {
                    result.push(json!({"type": format!("response.{channel}.delta"), "item_id": item["id"],
                        "output_index": output_index, "content_index": content_index, "delta": character.to_string()}));
                }
                result.push(json!({"type": format!("response.{channel}.done"), "item_id": item["id"],
                    "output_index": output_index, "content_index": content_index, "text": part["text"]}));
                result.push(
                    json!({"type": "response.content_part.done", "item_id": item["id"],
                    "output_index": output_index, "content_index": content_index, "part": part}),
                );
            }
        }
        result.push(json!({"type": "response.output_item.done", "output_index": output_index, "item": item}));
    }
    result.push(json!({"type": "response.completed", "response": response}));
    result
}

fn wire(events: &[Value], done: bool) -> String {
    let mut wire = String::new();
    for (sequence, event) in events.iter().enumerate() {
        let mut event = event.clone();
        event["sequence_number"] = json!(sequence);
        wire.push_str(&format!(
            "event: {}\ndata: {event}\n\n",
            event["type"].as_str().unwrap()
        ));
    }
    if done {
        wire.push_str("data: [DONE]\n\n");
    }
    wire
}

fn first<'a>(events: &'a mut [Value], kind: &str) -> &'a mut Value {
    events
        .iter_mut()
        .find(|event| event["type"] == kind)
        .unwrap()
}

#[test]
fn responses_sync_and_sse_preserve_native_replay_and_canonical_channels() {
    for native in [
        response(vec![thought(), call()]),
        response(vec![message("{\"answer\":579}")]),
    ] {
        let sync = responses_sync(&native.to_string()).unwrap();
        for done in [false, true] {
            let raw = wire(&events(&native), done);
            let streamed = responses_stream(&(raw.replace('\n', "\r\n") + ": keepalive")).unwrap();
            assert_eq!(streamed.response, native);
            assert_eq!(sync.response, native);
            assert_eq!(streamed.chat.message, sync.chat.message);
            assert_eq!(streamed.chat.finish, sync.chat.finish);
            assert_eq!(streamed.chat.usage, sync.chat.usage);
        }
        if native["output"][0]["type"] == "reasoning" {
            assert_eq!(sync.chat.message["reasoning"], "用工具计算 123+456。");
            assert_eq!(sync.chat.message["tool_calls"][0]["id"], "call-1");
            assert_eq!(
                sync.chat.message["tool_calls"][0]["function"]["arguments"],
                call()["arguments"]
            );
            assert_eq!(sync.chat.finish, "tool_calls");
            assert_eq!(sync.chat.content(), "");
        } else {
            assert_eq!(sync.chat.content(), "{\"answer\":579}");
            assert_eq!(sync.chat.finish, "stop");
        }
    }
}

#[test]
fn responses_sync_rejects_incomplete_errors_duplicate_identity_and_invalid_usage() {
    let native = response(vec![thought(), call()]);
    for (field, value) in [
        ("status", json!("incomplete")),
        ("error", json!({"message": "failed"})),
        ("incomplete_details", json!({"reason": "max_output_tokens"})),
        (
            "usage",
            json!({"input_tokens": 12, "output_tokens": 8, "total_tokens": 19}),
        ),
        ("usage", Value::Null),
        ("output", json!([])),
        ("output", json!([call(), call()])),
    ] {
        let mut invalid = native.clone();
        invalid[field] = value;
        assert!(
            responses_sync(&invalid.to_string()).is_err(),
            "accepted {invalid}"
        );
    }
    for (field, value) in [
        ("id", json!("rs-1")),
        ("status", json!("in_progress")),
        ("arguments", json!({})),
    ] {
        let mut invalid = native.clone();
        invalid["output"][1][field] = value;
        assert!(responses_sync(&invalid.to_string()).is_err());
    }
    let mut duplicate_call = call();
    duplicate_call["id"] = json!("fc-2");
    assert!(responses_sync(&response(vec![call(), duplicate_call]).to_string()).is_err());
    let mut wrong_channel = thought();
    wrong_channel["content"][0]["type"] = json!("output_text");
    assert!(responses_sync(&response(vec![wrong_channel, call()]).to_string()).is_err());
}

#[test]
fn responses_sse_requires_complete_ordered_lifecycle_and_one_terminal() {
    let valid = events(&response(vec![message("579")]));
    for kind in [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ] {
        let removed: Vec<_> = valid
            .iter()
            .filter(|event| event["type"] != kind)
            .cloned()
            .collect();
        assert!(
            responses_stream(&wire(&removed, true)).is_err(),
            "accepted missing {kind}"
        );
        let mut repeated = valid.clone();
        let index = repeated
            .iter()
            .position(|event| event["type"] == kind)
            .unwrap();
        repeated.insert(index, repeated[index].clone());
        assert!(
            responses_stream(&wire(&repeated, true)).is_err(),
            "accepted duplicate {kind}"
        );
    }
    for kind in ["error", "response.failed", "response.incomplete"] {
        let mut failed = valid.clone();
        failed.last_mut().unwrap()["type"] = json!(kind);
        assert!(responses_stream(&wire(&failed, true)).is_err());
    }
    let raw = wire(&valid, true);
    assert!(responses_stream(&(raw.clone() + "data: [DONE]\n\n")).is_err());
    assert!(
        responses_stream(&raw.replace("\"sequence_number\":1", "\"sequence_number\":0")).is_err()
    );
    assert!(responses_stream(&raw.replace(
        "event: response.output_text.done",
        "event: response.reasoning_text.done"
    ))
    .is_err());
    assert!(responses_stream(raw.trim_end()).is_err());
    let mut reordered = valid;
    reordered.swap(0, 1);
    assert!(responses_stream(&wire(&reordered, true)).is_err());
}

#[test]
fn responses_sse_rejects_changed_ids_indexes_and_payload_at_every_completion_level() {
    let valid = events(&response(vec![thought(), message("579")]));
    for (kind, field, value) in [
        (
            "response.output_text.delta",
            "item_id",
            json!("another-message"),
        ),
        ("response.output_text.delta", "output_index", json!(0)),
        ("response.output_text.delta", "content_index", json!(1)),
        ("response.output_text.delta", "delta", json!("wrong")),
        ("response.output_text.done", "text", json!("578")),
        (
            "response.reasoning_text.done",
            "text",
            json!("replacement thought"),
        ),
        ("response.content_part.added", "content_index", json!(1)),
        ("response.output_item.added", "output_index", json!(1)),
    ] {
        let mut invalid = valid.clone();
        first(&mut invalid, kind)[field] = value;
        assert!(
            responses_stream(&wire(&invalid, true)).is_err(),
            "accepted {kind}/{field}"
        );
    }
    let mut changed_part = valid.clone();
    first(&mut changed_part, "response.content_part.done")["part"]["text"] = json!("injected");
    assert!(responses_stream(&wire(&changed_part, true)).is_err());
    let mut changed_item = valid.clone();
    first(&mut changed_item, "response.output_item.done")["item"]["content"][0]["text"] =
        json!("injected");
    assert!(responses_stream(&wire(&changed_item, true)).is_err());
    for change in ["id", "content", "order", "control"] {
        let mut invalid = valid.clone();
        let response = &mut invalid.last_mut().unwrap()["response"];
        match change {
            "id" => response["id"] = json!("resp-other"),
            "content" => response["output"][1]["content"][0]["text"] = json!("injected"),
            "order" => response["output"].as_array_mut().unwrap().swap(0, 1),
            "control" => response["tool_choice"] = json!("none"),
            _ => unreachable!(),
        }
        assert!(
            responses_stream(&wire(&invalid, true)).is_err(),
            "accepted completed {change}"
        );
    }
}

#[test]
fn responses_sse_checks_fragmented_function_identity_arguments_and_terminal_items() {
    let valid = events(&response(vec![thought(), call()]));
    for (kind, field, value) in [
        (
            "response.function_call_arguments.delta",
            "item_id",
            json!("fc-other"),
        ),
        (
            "response.function_call_arguments.delta",
            "delta",
            json!("incorrect"),
        ),
        (
            "response.function_call_arguments.done",
            "arguments",
            json!("{}"),
        ),
        (
            "response.function_call_arguments.done",
            "call_id",
            json!("call-other"),
        ),
        (
            "response.function_call_arguments.done",
            "name",
            json!("another_tool"),
        ),
        (
            "response.function_call_arguments.done",
            "output_index",
            json!(0),
        ),
    ] {
        let mut invalid = valid.clone();
        first(&mut invalid, kind)[field] = value;
        assert!(
            responses_stream(&wire(&invalid, true)).is_err(),
            "accepted {kind}/{field}"
        );
    }
    let mut missing = valid.clone();
    missing.retain(|event| event["type"] != "response.function_call_arguments.done");
    assert!(responses_stream(&wire(&missing, true)).is_err());
    let mut late_delta = valid.clone();
    let delta = first(&mut late_delta, "response.function_call_arguments.delta").clone();
    let done_index = late_delta
        .iter()
        .position(|event| event["type"] == "response.function_call_arguments.done")
        .unwrap();
    late_delta.insert(done_index + 1, delta);
    assert!(responses_stream(&wire(&late_delta, true)).is_err());
    let mut injected = valid;
    for event in &mut injected {
        if event["type"] == "response.output_item.done" && event["item"]["type"] == "function_call"
        {
            event["item"]["arguments"] = json!("{\"expression\":\"1+1\"}");
        }
    }
    assert!(responses_stream(&wire(&injected, true)).is_err());
}

#[test]
fn responses_sse_accepts_multiple_content_parts_and_calls_without_index_confusion() {
    let mut text = message("One ");
    text["phase"] = json!("commentary");
    text["content"]
        .as_array_mut()
        .unwrap()
        .push(json!({"type": "output_text", "text": "尾", "annotations": [], "logprobs": []}));
    let mut second_call = call();
    second_call["id"] = json!("fc-2");
    second_call["call_id"] = json!("call-2");
    second_call["arguments"] = json!("{\"expression\":\"2+2\"}");
    let native = response(vec![thought(), text, call(), second_call]);
    let parsed = responses_stream(&wire(&events(&native), true)).unwrap();
    assert_eq!(parsed.response, native);
    assert_eq!(parsed.chat.content(), "One 尾");
    assert_eq!(
        parsed.chat.message["tool_calls"].as_array().unwrap().len(),
        2
    );
    assert_eq!(parsed.chat.message["tool_calls"][1]["id"], "call-2");
}
