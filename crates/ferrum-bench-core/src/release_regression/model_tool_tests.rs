use super::*;
use serde_json::json;

fn called(expression: &str, id: &str) -> Value {
    json!({"message": {"role": "assistant", "content": null,
        "reasoning": "I will use the requested calculator.",
        "tool_calls": [{"id": id, "type": "function", "function": {
            "name": "calc", "arguments": json!({"expression": expression}).to_string()}}]},
        "finish_reason": "tool_calls", "usage": {
            "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})
}

fn continuation(call: &Value, alias: bool) -> Value {
    let id = call["message"]["tool_calls"][0]["id"].clone();
    let mut assistant = call["message"].clone();
    if alias {
        let object = assistant.as_object_mut().unwrap();
        let thought = object.remove("reasoning").unwrap();
        object.insert("reasoning_content".into(), thought);
    }
    json!({"message": {"role": "assistant", "content": "**579**", "reasoning": null},
        "finish_reason": "stop", "usage": {
            "prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43},
        "tool_call_id": id, "replayed_assistant": assistant,
        "tool_result_message": {"role": "tool", "tool_call_id": id,
            "content": "{\"result\":579}"}})
}

fn evidence(alias: bool) -> Value {
    let sync = called("(123 + 456)", "sync-calculation");
    let stream = called("(456) + (123)", "stream-calculation");
    json!({"sync_continuation": continuation(&sync, false),
        "stream_continuation": continuation(&stream, alias),
        "sync_call": sync, "stream_call": stream, "tool_result": 579,
        "reasoning_alias_replayed": alias})
}

fn native_response(observation: &Value, id: &str) -> Value {
    let mut output = Vec::new();
    let message = &observation["message"];
    if let Some(reasoning) = message["reasoning"]
        .as_str()
        .filter(|text| !text.is_empty())
    {
        output.push(
            json!({"id": format!("{id}-reasoning"), "type": "reasoning", "status": "completed",
            "summary": [], "content": [{"type": "reasoning_text", "text": reasoning}]}),
        );
    }
    if let Some(content) = message["content"].as_str().filter(|text| !text.is_empty()) {
        output.push(json!({"id": format!("{id}-message"), "type": "message", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}]}));
    }
    for (index, call) in message["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        output.push(json!({"id": format!("{id}-item-{index}"), "type": "function_call", "status": "completed",
            "call_id": call["id"], "name": call["function"]["name"], "arguments": call["function"]["arguments"]}));
    }
    json!({"id": id, "object": "response", "status": "completed", "output": output,
        "usage": {"input_tokens": observation["usage"]["prompt_tokens"],
            "output_tokens": observation["usage"]["completion_tokens"], "total_tokens": observation["usage"]["total_tokens"]}})
}

pub(crate) fn auto_json_evidence() -> Value {
    let mut evidence = json!({});
    for (mode, stream) in [("sync", false), ("stream", true)] {
        let call = called("(123)+(456)", &format!("{mode}-calculation"));
        let mut first = auto_tools_json_controls();
        first["messages"] = json!([{"role": "user", "content": AUTO_TOOLS_JSON_PROMPT}]);
        first["max_tokens"] = json!(128);
        first["stream"] = json!(stream);
        first["temperature"] = json!(0);
        if stream {
            first["stream_options"] = json!({"include_usage": true});
        }
        let mut replay = first.clone();
        replay["messages"].as_array_mut().unwrap().extend([
            call["message"].clone(),
            json!({"role": "tool", "tool_call_id": call["message"]["tool_calls"][0]["id"], "content": "{\"result\":579}"}),
        ]);
        evidence[mode] = json!({
            "call_request": first, "call": call, "continuation_request": replay,
            "continuation": {"message": {"role": "assistant", "content": "{\"answer\":579}", "reasoning": null},
                "finish_reason": "stop", "usage": {"prompt_tokens": 40, "completion_tokens": 4, "total_tokens": 44}}
        });
    }
    evidence["responses"] = json!({});
    for (mode, stream) in [("sync", false), ("stream", true)] {
        let call = called("456+123", &format!("responses-{mode}-calculation"));
        let native = native_response(&call, &format!("resp-{mode}-call"));
        let final_observation = evidence[mode]["continuation"].clone();
        let final_native = native_response(&final_observation, &format!("resp-{mode}-final"));
        let mut first = auto_tools_json_responses_controls();
        first["input"] = json!([{"role": "user", "content": AUTO_TOOLS_JSON_PROMPT}]);
        first["max_output_tokens"] = json!(128);
        first["stream"] = json!(stream);
        first["temperature"] = json!(0);
        let mut replay = first.clone();
        let input = replay["input"].as_array_mut().unwrap();
        input.extend(native["output"].as_array().unwrap().iter().cloned());
        input.push(json!({"type": "function_call_output", "call_id": call["message"]["tool_calls"][0]["id"],
            "output": "{\"result\":579}"}));
        evidence["responses"][mode] = json!({"call_request": first, "call": call, "call_response": native,
            "continuation_request": replay, "continuation": final_observation, "continuation_response": final_native});
    }
    evidence
}

#[test]
fn auto_tools_json_requires_native_responses_and_unchanged_full_output_replay() {
    let good = auto_json_evidence();
    verify_auto_tools_json_case(&good, 128).unwrap();
    let mut missing = good.clone();
    missing.as_object_mut().unwrap().remove("responses");
    assert!(verify_auto_tools_json_case(&missing, 128)
        .unwrap_err()
        .contains("responses.sync"));
    for mode in ["sync", "stream"] {
        for field in [
            "call_response",
            "continuation_response",
            "call_request",
            "continuation_request",
        ] {
            let mut wrong = good.clone();
            wrong["responses"][mode]
                .as_object_mut()
                .unwrap()
                .remove(field);
            assert!(
                verify_auto_tools_json_case(&wrong, 128).is_err(),
                "missing {mode}.{field}"
            );
        }
        for (suffix, changed) in [
            (
                "/continuation_request/input/3/call_id",
                json!("not-the-actual-call"),
            ),
            (
                "/continuation_request/input/3/output",
                json!("{\"result\":580}"),
            ),
            (
                "/continuation_request/input/3/output",
                json!("{\"result\":579,\"result\":579}"),
            ),
            ("/continuation_request/input/3/type", json!("message")),
            (
                "/continuation_request/input/1/content/0/text",
                json!("invented reasoning"),
            ),
            (
                "/continuation_request/input/2/arguments",
                json!("{\"expression\":\"122+457\"}"),
            ),
            (
                "/continuation_request/input/2/id",
                json!("invented output-item identity"),
            ),
            (
                "/call_response/output/1/call_id",
                json!("changed-native-call"),
            ),
            ("/call_response/usage/input_tokens", json!(11)),
            (
                "/continuation_response/output/0/content/0/text",
                json!("{\"answer\":580}"),
            ),
            ("/continuation_response/status", json!("incomplete")),
            (
                "/continuation_response/output/0/status",
                json!("incomplete"),
            ),
        ] {
            let mut wrong = good.clone();
            let pointer = format!("/responses/{mode}{suffix}");
            *wrong.pointer_mut(&pointer).unwrap() = changed;
            assert!(
                verify_auto_tools_json_case(&wrong, 128).is_err(),
                "accepted {pointer}"
            );
        }
        let mut reordered = good.clone();
        reordered["responses"][mode]["continuation_request"]["input"]
            .as_array_mut()
            .unwrap()
            .swap(1, 2);
        assert!(verify_auto_tools_json_case(&reordered, 128).is_err());
        let mut dropped = good.clone();
        dropped["responses"][mode]["continuation_request"]["input"]
            .as_array_mut()
            .unwrap()
            .remove(1);
        assert!(verify_auto_tools_json_case(&dropped, 128).is_err());
        let mut wrong_id = good.clone();
        wrong_id["responses"][mode]["continuation_request"]["input"][3]["call_id"] =
            good["responses"][mode]["call_response"]["output"][1]["id"].clone();
        assert!(
            verify_auto_tools_json_case(&wrong_id, 128).is_err(),
            "output item id is not call_id"
        );
        let mut wrong_namespace = good.clone();
        wrong_namespace["responses"][mode]["call_response"]["output"][1]["namespace"] =
            json!("undeclared");
        wrong_namespace["responses"][mode]["continuation_request"]["input"][2]["namespace"] =
            json!("undeclared");
        assert!(verify_auto_tools_json_case(&wrong_namespace, 128).is_err());
        let mut summary = good.clone();
        let changed = json!([{"type": "summary_text", "text": "unobserved reasoning"}]);
        summary["responses"][mode]["call_response"]["output"][0]["summary"] = changed.clone();
        summary["responses"][mode]["continuation_request"]["input"][1]["summary"] = changed;
        assert!(verify_auto_tools_json_case(&summary, 128).is_err());
    }
}

#[test]
fn auto_tools_json_responses_rejects_weakened_controls_and_chat_only_fields() {
    let good = auto_json_evidence();
    for mode in ["sync", "stream"] {
        for (field, value) in [
            ("tool_choice", json!("none")),
            ("tools", json!([])),
            ("text", Value::Null),
            ("temperature", json!(1)),
            ("max_output_tokens", json!(127)),
            ("stream", json!(mode != "stream")),
        ] {
            let mut wrong = good.clone();
            wrong["responses"][mode]["continuation_request"][field] = value;
            assert!(
                verify_auto_tools_json_case(&wrong, 128).is_err(),
                "changed replay {field}"
            );
        }
        for choice in [
            json!("none"),
            json!("required"),
            json!({"type":"function", "name":"calc"}),
        ] {
            let mut wrong = good.clone();
            for turn in ["call_request", "continuation_request"] {
                wrong["responses"][mode][turn]["tool_choice"] = choice.clone();
            }
            assert!(verify_auto_tools_json_case(&wrong, 128).is_err());
        }
        for field in [
            "response_format",
            "messages",
            "stream_options",
            "max_tokens",
            "stop",
            "seed",
        ] {
            let mut wrong = good.clone();
            for turn in ["call_request", "continuation_request"] {
                wrong["responses"][mode][turn][field] = Value::Null;
            }
            assert!(
                verify_auto_tools_json_case(&wrong, 128).is_err(),
                "accepted Chat-only {field}"
            );
        }
        for (field, value) in [("temperature", json!(1)), ("seed", json!(7))] {
            let mut wrong = good.clone();
            for turn in ["call_request", "continuation_request"] {
                wrong["responses"][mode][turn][field] = value.clone();
            }
            assert!(verify_auto_tools_json_case(&wrong, 128).is_err());
        }
        let mut weakened = good.clone();
        for turn in ["call_request", "continuation_request"] {
            weakened["responses"][mode][turn]["text"]["format"]["strict"] = json!(false);
        }
        assert!(verify_auto_tools_json_case(&weakened, 128).is_err());
        let mut missing = good.clone();
        missing["responses"].as_object_mut().unwrap().remove(mode);
        assert!(verify_auto_tools_json_case(&missing, 128).is_err());
    }
}

#[test]
fn auto_tools_json_reports_each_failed_mode_and_preserves_repeated_call_failure() {
    let mut repeated = auto_json_evidence();
    for mode in ["sync", "stream"] {
        repeated[mode]["continuation"] = repeated[mode]["call"].clone();
        repeated["responses"][mode]["continuation"] = repeated["responses"][mode]["call"].clone();
        repeated["responses"][mode]["continuation_response"] =
            repeated["responses"][mode]["call_response"].clone();
    }
    let error = verify_auto_tools_json_case(&repeated, 128).unwrap_err();
    for label in [
        "chat.sync: final:",
        "chat.stream: final:",
        "responses.sync: final:",
        "responses.stream: final:",
    ] {
        assert!(error.contains(label), "{error}");
    }
    assert!(error.contains("another tool call"));
    repeated["sync"] = json!({"phase": "call_response", "error": "HTTP transport timed out", "call_request": auto_tools_json_controls()});
    let error = verify_auto_tools_json_case(&repeated, 128).unwrap_err();
    assert!(error.contains("chat.sync: call_response: HTTP transport timed out"));
    assert!(error.contains("responses.stream: final:"));
    for pointer in ["/sync/error", "/responses/stream/error"] {
        for malformed in [json!({"message": "transport failed"}), json!(true)] {
            let mut corrupted = auto_json_evidence();
            let parent = pointer.strip_suffix("/error").unwrap();
            corrupted.pointer_mut(parent).unwrap()["error"] = malformed;
            assert!(verify_auto_tools_json_case(&corrupted, 128)
                .unwrap_err()
                .contains("invalid non-string error field"));
        }
    }
}

#[test]
fn auto_tools_json_checks_real_calculation_schema_and_caller_owned_history() {
    let good = auto_json_evidence();
    verify_auto_tools_json_case(&good, 128).unwrap();
    for mode in ["sync", "stream"] {
        for (suffix, value) in [
            (
                "/call/message/tool_calls/0/function/arguments",
                json!("{\"expression\":\"122+457\"}"),
            ),
            ("/continuation/message/content", json!("{\"answer\":580}")),
            (
                "/continuation/message/content",
                json!("{\"answer\":\"579\"}"),
            ),
            (
                "/continuation/message/content",
                json!("{\"answer\":579,\"extra\":true}"),
            ),
            (
                "/continuation/message/content",
                json!("{\"answer\":579,\"answer\":579}"),
            ),
            ("/continuation/message/content", json!("[579]")),
            ("/continuation/finish_reason", json!("length")),
            (
                "/continuation_request/messages/1/content",
                json!("invented assistant text"),
            ),
            (
                "/continuation_request/messages/2/tool_call_id",
                json!("another-call"),
            ),
            (
                "/continuation_request/messages/2/content",
                json!("{\"result\":580}"),
            ),
        ] {
            let mut wrong = good.clone();
            let pointer = format!("/{mode}{suffix}");
            *wrong.pointer_mut(&pointer).unwrap() = value;
            assert!(
                verify_auto_tools_json_case(&wrong, 128).is_err(),
                "accepted {pointer}"
            );
        }
    }
}

#[test]
fn auto_tools_json_rejects_forced_calls_or_weakened_replay_controls() {
    let good = auto_json_evidence();
    for mode in ["sync", "stream"] {
        for choice in [
            json!("required"),
            json!("none"),
            json!({"type": "function", "function": {"name": "calc"}}),
        ] {
            let mut wrong = good.clone();
            wrong[mode]["call_request"]["tool_choice"] = choice.clone();
            wrong[mode]["continuation_request"]["tool_choice"] = choice;
            assert!(verify_auto_tools_json_case(&wrong, 128).is_err());
        }
        let mut dropped_schema = good.clone();
        for turn in ["call_request", "continuation_request"] {
            dropped_schema[mode][turn]
                .as_object_mut()
                .unwrap()
                .remove("response_format");
        }
        assert!(verify_auto_tools_json_case(&dropped_schema, 128).is_err());
        for (field, changed) in [
            ("tool_choice", json!("none")),
            ("tools", json!([])),
            ("response_format", Value::Null),
            ("temperature", json!(1)),
        ] {
            let mut wrong = good.clone();
            wrong[mode]["continuation_request"][field] = changed;
            assert!(
                verify_auto_tools_json_case(&wrong, 128).is_err(),
                "lost {field}"
            );
        }
    }
}

#[test]
fn auto_tools_json_requires_both_observed_wire_modes_and_request_budgets() {
    let good = auto_json_evidence();
    for mode in ["sync", "stream"] {
        let mut missing = good.clone();
        missing.as_object_mut().unwrap().remove(mode);
        assert!(verify_auto_tools_json_case(&missing, 128).is_err());
        let mut wrong_mode = good.clone();
        wrong_mode[mode]["call_request"]["stream"] = json!(mode != "stream");
        wrong_mode[mode]["continuation_request"]["stream"] = json!(mode != "stream");
        assert!(verify_auto_tools_json_case(&wrong_mode, 128).is_err());
    }
    assert!(verify_auto_tools_json_case(&good, 127).is_err());
}

#[test]
fn integer_addition_accepts_balanced_parentheses_without_joining_separate_numbers() {
    for expression in [
        "123+456",
        "(123+456)",
        "(123)+(456)",
        " ((123) + ((456))) ",
        "\n456\t+\u{2003}123\r",
        "(000123)+0456",
    ] {
        let addition = IntegerAddition::parse(expression).unwrap();
        assert_eq!(addition.result().unwrap(), 579, "{expression}");
    }
    assert_eq!(IntegerAddition::parse("0+0").unwrap().result().unwrap(), 0);
    assert_eq!(
        IntegerAddition::parse("18446744073709551615+0")
            .unwrap()
            .result()
            .unwrap(),
        u64::MAX
    );
    for expression in [
        "",
        "123",
        "123456",
        "12 3+456",
        "123+45 6",
        "123\n456",
        "123++456",
        "+123+456",
        "123+-456",
        "123-456",
        "123*456",
        "123/456",
        "123+456+0",
        "123+456;0+0",
        "123+456 trailing",
        "(123+456",
        "123+456)",
        "()123+456",
        "(123+)456",
        "123(+456)",
        "123+()456",
        "(123)(456)",
        "0x7b+456",
        "123.0+456",
        "１２３+456",
        "18446744073709551616+0",
        "18446744073709551615+1",
    ] {
        assert!(
            IntegerAddition::parse(expression).is_err(),
            "accepted {expression:?}"
        );
    }
    assert!(IntegerAddition {
        left: u64::MAX,
        right: 1
    }
    .result()
    .is_err());
}

#[test]
fn calc_call_checks_requested_operands_and_actual_argument_schema() {
    for expression in ["(123+456)", "(456)+(123)"] {
        let observation = called(expression, "actual-id");
        let validated = verify_calc_call(&observation, 20).unwrap();
        assert_eq!(validated.call, observation["message"]["tool_calls"][0]);
        assert_eq!(validated.result, 579);
    }
    for expression in ["122+457", "123+457", "12 3+456", "123+456+0"] {
        assert!(verify_calc_call(&called(expression, "call"), 128).is_err());
    }
    for arguments in [
        r#"{"expression":"123+456","extra":0}"#,
        r#"{"expression":579}"#,
        r#"{"expression":"123+456","expression":"456+123"}"#,
        r#"["123+456"]"#,
        "{invalid",
    ] {
        let mut observation = called("123+456", "call");
        observation["message"]["tool_calls"][0]["function"]["arguments"] = json!(arguments);
        assert!(verify_calc_call(&observation, 128).is_err(), "{arguments}");
    }
}

#[test]
fn calc_handoff_requires_canonical_identity_termination_and_consistent_bounded_usage() {
    let good = called("123+456", "call");
    for (pointer, value) in [
        ("/message/role", json!("tool")),
        ("/message/content", json!(579)),
        ("/message/reasoning", json!("<think>leaked")),
        ("/message/tool_calls/0/id", json!(" ")),
        ("/message/tool_calls/0/type", json!("other")),
        (
            "/message/tool_calls/0/function/name",
            json!("lookup_weather"),
        ),
        ("/finish_reason", json!("stop")),
        ("/usage/prompt_tokens", json!(0)),
        ("/usage/completion_tokens", json!(0)),
        ("/usage/total_tokens", json!(31)),
    ] {
        let mut wrong = good.clone();
        *wrong.pointer_mut(pointer).unwrap() = value;
        assert!(verify_calc_call(&wrong, 128).is_err(), "{pointer}");
    }
    for field in ["reasoning_content", "function_call"] {
        let mut wrong = good.clone();
        wrong["message"][field] = json!("unexpected legacy field");
        assert!(verify_calc_call(&wrong, 128).is_err());
    }
    let mut multiple = good.clone();
    multiple["message"]["tool_calls"]
        .as_array_mut()
        .unwrap()
        .push(good["message"]["tool_calls"][0].clone());
    assert!(verify_calc_call(&multiple, 128).is_err());
    assert!(verify_calc_call(&good, 19).is_err());
    assert!(verify_calc_call(&good, 0).is_err());
    let mut overflow = good;
    overflow["usage"] = json!({"prompt_tokens": u64::MAX, "completion_tokens": 1});
    assert!(verify_calc_call(&overflow, 128).is_err());
}

#[test]
fn tool_call_preamble_is_preserved_in_actual_continuation_history() {
    let mut document = evidence(false);
    for (mode, preamble) in [
        ("sync", "I will use the calculator."),
        ("stream", "Calculating that sum now."),
    ] {
        let mut call = document[format!("{mode}_call")].clone();
        call["message"]["content"] = json!(preamble);
        verify_calc_call(&call, 128).unwrap();
        document[format!("{mode}_continuation")] = continuation(&call, false);
        document[format!("{mode}_call")] = call;
    }
    verify_tool_case(&document, 128, false).unwrap();
    document["stream_continuation"]["replayed_assistant"]["content"] = Value::Null;
    assert!(verify_tool_case(&document, 128, false).is_err());
}

#[test]
fn tool_case_replays_each_actual_call_and_only_moves_observed_reasoning_for_alias_mode() {
    verify_tool_case(&evidence(false), 128, false).unwrap();
    let mut punctuation = evidence(false);
    punctuation["sync_continuation"]["message"]["content"] = json!("579.");
    punctuation["stream_continuation"]["message"]["content"] = json!("`579`");
    verify_tool_case(&punctuation, 128, false).unwrap();
    verify_tool_case(&evidence(true), 128, true).unwrap();
    assert!(verify_tool_case(&evidence(false), 128, true).is_err());
    assert!(verify_tool_case(&evidence(true), 128, false).is_err());
    for value in [Value::Null, json!(""), json!("   ")] {
        let mut wrong = evidence(true);
        wrong["stream_call"]["message"]["reasoning"] = value.clone();
        wrong["stream_continuation"]["replayed_assistant"]["reasoning_content"] = value;
        assert!(verify_tool_case(&wrong, 128, true).is_err());
    }
    let mut invented = evidence(true);
    invented["stream_continuation"]["replayed_assistant"]["reasoning_content"] =
        json!("Invented reasoning");
    assert!(verify_tool_case(&invented, 128, true).is_err());
}

#[test]
fn tool_case_rejects_wrong_computation_history_identity_and_missing_entrypoint() {
    let good = evidence(false);
    for field in [
        "sync_call",
        "stream_call",
        "sync_continuation",
        "stream_continuation",
        "tool_result",
        "reasoning_alias_replayed",
    ] {
        let mut wrong = good.clone();
        wrong.as_object_mut().unwrap().remove(field);
        assert!(verify_tool_case(&wrong, 128, false).is_err(), "{field}");
    }
    for mode in ["sync", "stream"] {
        let continuation = format!("/{mode}_continuation");
        for (suffix, value) in [
            ("/tool_call_id", json!("another-call")),
            ("/tool_result_message/tool_call_id", json!("another-call")),
            ("/tool_result_message/role", json!("assistant")),
            ("/tool_result_message/content", json!("{\"result\":580}")),
            ("/tool_result_message/content", json!("[579]")),
            (
                "/tool_result_message/content",
                json!("{\"result\":579,\"extra\":1}"),
            ),
            ("/replayed_assistant/reasoning", json!("fabricated")),
            (
                "/replayed_assistant/tool_calls/0/function/arguments",
                json!("{\"expression\":\"122+457\"}"),
            ),
            ("/message/content", json!("5 79")),
            ("/message/content", json!(".579")),
            ("/message/content", json!("580")),
            ("/message/content", json!("579 and another answer")),
            ("/finish_reason", json!("length")),
            ("/usage/completion_tokens", json!(0)),
            ("/usage/total_tokens", json!(44)),
        ] {
            let mut wrong = good.clone();
            let pointer = format!("{continuation}{suffix}");
            *wrong.pointer_mut(&pointer).unwrap() = value;
            assert!(verify_tool_case(&wrong, 128, false).is_err(), "{pointer}");
        }
        let mut new_tool = good.clone();
        new_tool[format!("{mode}_continuation")]["message"]["tool_calls"] =
            good[format!("{mode}_call")]["message"]["tool_calls"].clone();
        assert!(verify_tool_case(&new_tool, 128, false).is_err());
        let mut over_budget = good.clone();
        over_budget[format!("{mode}_continuation")]["usage"] =
            json!({"prompt_tokens": 40, "completion_tokens": 129, "total_tokens": 169});
        assert!(verify_tool_case(&over_budget, 128, false).is_err());
    }
    let mut wrong_result = good;
    wrong_result["tool_result"] = json!(580);
    assert!(verify_tool_case(&wrong_result, 128, false).is_err());
}
