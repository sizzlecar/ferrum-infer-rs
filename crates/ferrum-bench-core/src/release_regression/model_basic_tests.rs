use super::*;

fn observation(answer: &str, reasoning: &str) -> Value {
    json!({
        "message": {"role": "assistant", "content": answer, "reasoning": reasoning},
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 24, "completion_tokens": 8, "total_tokens": 32}
    })
}

fn run_evidence(reasoning: &str) -> Value {
    let answers: Vec<_> = ["OK", "42", "cobalt-731"]
        .into_iter()
        .map(|answer| {
            let output = observation(answer, reasoning);
            json!({
                "event": "assistant", "content": output["message"]["content"],
                "reasoning": output["message"]["reasoning"], "finish_reason": "eos",
                "usage": output["usage"]
            })
        })
        .collect();
    json!({"prompts": [MEMORY_PROMPT, ARITHMETIC_PROMPT, RECALL_PROMPT], "answers": answers})
}

fn serve_evidence(reasoning: &str) -> Value {
    let observations = json!({
        "memory_write": observation("OK", reasoning),
        "sync": observation("42", reasoning),
        "stream": observation("42", reasoning),
        "recall": observation("cobalt-731", reasoning),
        "stream_recall": observation("cobalt-731", reasoning)
    });
    let memory = json!({"role": "user", "content": MEMORY_PROMPT});
    let arithmetic = json!({"role": "user", "content": ARITHMETIC_PROMPT});
    let recall = json!({"role": "user", "content": RECALL_PROMPT});
    let written = &observations["memory_write"]["message"];
    let requests = json!({
        "memory_write": [memory],
        "sync": [memory, written, arithmetic],
        "stream": [memory, written, arithmetic],
        "recall": [memory, written, arithmetic, observations["sync"]["message"], recall],
        "stream_recall": [memory, written, arithmetic, observations["stream"]["message"], recall]
    });
    json!({"observations": observations, "requests": requests})
}

#[test]
fn basic_requires_correct_real_answers_for_thinking_and_non_thinking_models() {
    for (protocol, reasoning) in [
        (ModelReasoningProtocol::None, ""),
        (
            ModelReasoningProtocol::PromptOpened,
            "I will remember cobalt-731.",
        ),
        (ModelReasoningProtocol::ModelGenerated, "17 plus 25 is 42."),
    ] {
        verify_basic_run(&run_evidence(reasoning), protocol, 8).unwrap();
        verify_basic_serve(&serve_evidence(reasoning), protocol, 8).unwrap();
    }
}

#[test]
fn basic_accepts_answer_formatting_without_changing_decimal_values_or_history() {
    for (protocol, reasoning) in [
        (ModelReasoningProtocol::None, ""),
        (
            ModelReasoningProtocol::PromptOpened,
            "I will remember cobalt-731.",
        ),
    ] {
        let mut run = run_evidence(reasoning);
        for (index, formatted) in ["OK.", "**42**", "cobalt-731."].into_iter().enumerate() {
            run["answers"][index]["content"] = json!(formatted);
        }
        verify_basic_run(&run, protocol, 8).unwrap();
        run["answers"][1]["content"] = json!(".42");
        assert!(verify_basic_run(&run, protocol, 8).is_err());

        let mut serve = serve_evidence(reasoning);
        for (step, formatted) in [
            ("memory_write", "OK."),
            ("sync", "**42**"),
            ("stream", "**42**"),
            ("recall", "cobalt-731."),
            ("stream_recall", "cobalt-731."),
        ] {
            serve["observations"][step]["message"]["content"] = json!(formatted);
        }
        for step in ["sync", "stream", "recall", "stream_recall"] {
            serve["requests"][step][1] = serve["observations"]["memory_write"]["message"].clone();
        }
        for (step, arithmetic) in [("recall", "sync"), ("stream_recall", "stream")] {
            serve["requests"][step][3] = serve["observations"][arithmetic]["message"].clone();
        }
        verify_basic_serve(&serve, protocol, 8).unwrap();
        for step in ["sync", "stream"] {
            let mut bad = serve.clone();
            bad["observations"][step]["message"]["content"] = json!(".42");
            assert!(verify_basic_serve(&bad, protocol, 8).is_err());
        }
    }
}

#[test]
fn older_two_turn_and_missing_http_steps_cannot_satisfy_basic() {
    let mut old = run_evidence("");
    old["answers"].as_array_mut().unwrap().remove(0);
    assert!(verify_basic_run(&old, ModelReasoningProtocol::PromptOpened, 8).is_err());
    for step in ["memory_write", "sync", "stream", "recall", "stream_recall"] {
        let mut missing = serve_evidence("");
        missing["observations"]
            .as_object_mut()
            .unwrap()
            .remove(step);
        assert!(verify_basic_serve(&missing, ModelReasoningProtocol::PromptOpened, 8).is_err());
    }
}

#[test]
fn basic_binds_the_actual_prompt_sequence_and_each_http_history() {
    let mut run = run_evidence("");
    run["prompts"][2] = json!("Repeat cobalt-731.");
    assert!(verify_basic_run(&run, ModelReasoningProtocol::None, 8).is_err());

    let mut valid = serve_evidence("I will remember cobalt-731.");
    for (mode, recall) in [("sync", "recall"), ("stream", "stream_recall")] {
        valid["observations"][mode]["message"]["reasoning"] =
            json!(format!("The {mode} calculation gives 42."));
        valid["requests"][recall][3] = valid["observations"][mode]["message"].clone();
    }
    verify_basic_serve(&valid, ModelReasoningProtocol::PromptOpened, 8).unwrap();
    for (recall, other) in [("recall", "stream"), ("stream_recall", "sync")] {
        let mut wrong = valid.clone();
        wrong["requests"][recall][3] = valid["observations"][other]["message"].clone();
        assert!(verify_basic_serve(&wrong, ModelReasoningProtocol::PromptOpened, 8).is_err());
    }
    for step in ["memory_write", "sync", "stream", "recall", "stream_recall"] {
        let mut missing = valid.clone();
        missing["requests"].as_object_mut().unwrap().remove(step);
        assert!(verify_basic_serve(&missing, ModelReasoningProtocol::PromptOpened, 8).is_err());
    }
    for step in ["recall", "stream_recall"] {
        let mut repeated = valid.clone();
        repeated["requests"][step][4]["content"] = json!("Repeat the code cobalt-731.");
        assert!(verify_basic_serve(&repeated, ModelReasoningProtocol::PromptOpened, 8).is_err());
    }
    for step in ["sync", "stream", "recall", "stream_recall"] {
        let mut altered = valid.clone();
        altered["requests"][step][1]["reasoning"] = Value::Null;
        assert!(verify_basic_serve(&altered, ModelReasoningProtocol::PromptOpened, 8).is_err());
        let mut dropped = valid.clone();
        dropped["requests"][step].as_array_mut().unwrap().remove(0);
        assert!(verify_basic_serve(&dropped, ModelReasoningProtocol::PromptOpened, 8).is_err());
    }
}

#[test]
fn basic_rejects_refusal_arithmetic_errors_and_failed_recall_in_every_mode() {
    for (index, wrong) in [(0, "I cannot remember that."), (1, "43"), (2, "cobalt-732")] {
        let mut bad = run_evidence("");
        bad["answers"][index]["content"] = json!(wrong);
        assert!(verify_basic_run(&bad, ModelReasoningProtocol::ModelGenerated, 8).is_err());
    }
    for (step, wrong) in [
        ("memory_write", "cobalt-731"),
        ("sync", "43"),
        ("stream", "43"),
        ("recall", "cobalt-732"),
        ("stream_recall", "cobalt-732"),
    ] {
        let mut bad = serve_evidence("");
        bad["observations"][step]["message"]["content"] = json!(wrong);
        assert!(verify_basic_serve(&bad, ModelReasoningProtocol::ModelGenerated, 8).is_err());
    }
}

#[test]
fn every_non_thinking_observation_must_remain_without_reasoning() {
    for index in 0..3 {
        let mut bad = run_evidence("");
        bad["answers"][index]["reasoning"] = json!("private thought");
        assert!(verify_basic_run(&bad, ModelReasoningProtocol::None, 8).is_err());
    }
    for step in ["memory_write", "sync", "stream", "recall", "stream_recall"] {
        let mut bad = serve_evidence("");
        bad["observations"][step]["message"]["reasoning"] = json!("private thought");
        assert!(verify_basic_serve(&bad, ModelReasoningProtocol::None, 8).is_err());
    }
}

#[test]
fn canonical_checks_reject_aliases_tool_calls_and_control_leaks() {
    for (field, invalid) in [
        ("reasoning_content", Value::Null),
        ("reasoning_content", json!("private thought")),
        ("tool_calls", json!([{"id": "unexpected-call"}])),
        ("reasoning", json!(false)),
        ("reasoning", json!("<think>unparsed thought")),
        ("content", json!("<|assistant|>OK")),
        ("content", json!("OK<")),
    ] {
        let mut run = run_evidence("");
        run["answers"][0][field] = invalid.clone();
        assert!(verify_basic_run(&run, ModelReasoningProtocol::PromptOpened, 8).is_err());
        let mut serve = serve_evidence("");
        serve["observations"]["memory_write"]["message"][field] = invalid;
        assert!(verify_basic_serve(&serve, ModelReasoningProtocol::PromptOpened, 8).is_err());
    }
    let mut bad_role = serve_evidence("");
    bad_role["observations"]["memory_write"]["message"]["role"] = json!("user");
    assert!(verify_basic_serve(&bad_role, ModelReasoningProtocol::None, 8).is_err());
}

#[test]
fn basic_requires_natural_finish_and_consistent_positive_usage_within_budget() {
    for (field, invalid) in [
        ("finish_reason", json!("length")),
        ("finish_reason", json!("tool_calls")),
        ("finish_reason", Value::Null),
        (
            "usage",
            json!({"prompt_tokens": 0, "completion_tokens": 8, "total_tokens": 8}),
        ),
        (
            "usage",
            json!({"prompt_tokens": 24, "completion_tokens": 0, "total_tokens": 24}),
        ),
        (
            "usage",
            json!({"prompt_tokens": 24, "completion_tokens": 8, "total_tokens": 33}),
        ),
        (
            "usage",
            json!({"prompt_tokens": u64::MAX, "completion_tokens": 1, "total_tokens": 0}),
        ),
        (
            "usage",
            json!({"prompt_tokens": u64::MAX, "completion_tokens": 1}),
        ),
        (
            "usage",
            json!({"prompt_tokens": 24, "completion_tokens": 9, "total_tokens": 33}),
        ),
    ] {
        let mut run = run_evidence("");
        run["answers"][1][field] = invalid.clone();
        assert!(verify_basic_run(&run, ModelReasoningProtocol::PromptOpened, 8).is_err());
        let mut serve = serve_evidence("");
        serve["observations"]["stream"][field] = invalid;
        assert!(verify_basic_serve(&serve, ModelReasoningProtocol::PromptOpened, 8).is_err());
    }
    assert!(verify_basic_run(&run_evidence(""), ModelReasoningProtocol::None, 0).is_err());
    assert!(verify_basic_serve(&serve_evidence(""), ModelReasoningProtocol::None, 0).is_err());
    let mut http_eos = serve_evidence("");
    http_eos["observations"]["sync"]["finish_reason"] = json!("eos");
    assert!(verify_basic_serve(&http_eos, ModelReasoningProtocol::None, 8).is_err());
}
