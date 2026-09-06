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
