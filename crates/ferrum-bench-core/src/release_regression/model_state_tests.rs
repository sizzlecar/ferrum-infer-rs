use super::*;

fn observed(content: &str) -> Value {
    json!({"message": {"role": "assistant", "content": content, "reasoning": null}, "finish_reason": "stop", "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}})
}

fn turn(records: &mut Vec<Value>, epoch: u64, turn: u64, prompt: &str, content: &str) {
    let id = format!("request-{epoch}-{turn}");
    records.push(json!({"event": "user", "session_id": "one-session", "request_id": id, "history_epoch": epoch, "turn": turn, "content": prompt, "history_before": {"message_count": 2 * turn}}));
    records.push(json!({"event": "assistant", "session_id": "one-session", "request_id": id, "history_epoch": epoch, "turn": turn, "content": content, "reasoning": null, "finish_reason": "eos", "usage": observed(content)["usage"]}));
}

fn reset(records: &mut Vec<Value>, epoch: u64, before: u64) {
    records.push(json!({"event": "history_reset", "session_id": "one-session", "history_epoch": epoch, "turn": 0, "history_before": {"message_count": before}, "history_after": {"message_count": 0, "turn_count": 0}}));
}

pub(crate) fn run_fixture() -> Vec<Value> {
    run_fixture_with_controls("OK", "NONE")
}

fn run_fixture_with_controls(acknowledged: &str, no_memory: &str) -> Vec<Value> {
    let mut records = Vec::new();
    turn(&mut records, 0, 0, &remember("cobalt-731"), acknowledged);
    turn(&mut records, 0, 1, RECALL, "cobalt-731");
    reset(&mut records, 1, 4);
    turn(&mut records, 1, 0, EMPTY_RECALL, no_memory);
    turn(&mut records, 1, 1, &remember("amber-284"), acknowledged);
    turn(&mut records, 1, 2, RECALL, "amber-284");
    reset(&mut records, 2, 6);
    turn(&mut records, 2, 0, EMPTY_RECALL, no_memory);
    records
}

#[test]
fn run_reset_requires_forgotten_old_state_and_continuity_of_the_new_history() {
    verify_run(&run_fixture(), ModelReasoningProtocol::None, 16).unwrap();
    for mutate in [
        |r: &mut Vec<Value>| {
            r[6]["content"] = json!("cobalt-731");
        },
        |r: &mut Vec<Value>| {
            r[10]["content"] = json!("cobalt-731");
        },
        |r: &mut Vec<Value>| {
            r.remove(4);
        },
        |r: &mut Vec<Value>| {
            r[4]["history_after"]["message_count"] = json!(4);
        },
        |r: &mut Vec<Value>| {
            r[5]["history_before"]["message_count"] = json!(4);
        },
        |r: &mut Vec<Value>| {
            r[6]["history_epoch"] = json!(0);
        },
        |r: &mut Vec<Value>| {
            r[6]["session_id"] = json!("another-session");
        },
        |r: &mut Vec<Value>| {
            r[5]["request_id"] = json!("request-0-0");
            r[6]["request_id"] = json!("request-0-0");
        },
        |r: &mut Vec<Value>| {
            r[1]["usage"]["total_tokens"] = json!(26);
        },
        |r: &mut Vec<Value>| {
            r[1]["reasoning_content"] = Value::Null;
        },
        |r: &mut Vec<Value>| {
            r.push(json!({"event": "error"}));
        },
    ] {
        let mut bad = run_fixture();
        mutate(&mut bad);
        assert!(
            verify_run(&bad, ModelReasoningProtocol::None, 16).is_err(),
            "{bad:?}"
        );
    }
}

pub(crate) fn serve_fixture() -> ServeStateEvidence {
    serve_fixture_with_controls("OK", "NONE")
}

fn serve_fixture_with_controls(acknowledged: &str, no_memory: &str) -> ServeStateEvidence {
    let mut evidence = ServeStateEvidence {
        writes: Vec::new(),
        recall_rounds: Vec::new(),
        fresh: Vec::new(),
    };
    let mut histories = Vec::new();
    for (index, code) in ["cobalt-731", "amber-284"].into_iter().enumerate() {
        let mut messages = vec![json!({"role": "user", "content": remember(code)})];
        evidence.writes.push(StateExchange {
            request_id: format!("write-{index}"),
            messages: messages.clone(),
            stream: index == 1,
            observation: observed(acknowledged),
        });
        messages.push(observed(acknowledged)["message"].clone());
        histories.push(messages);
    }
    for round in 0..2 {
        let mut exchanges = Vec::new();
        for (index, code) in ["cobalt-731", "amber-284"].into_iter().enumerate() {
            histories[index].push(json!({"role": "user", "content": RECALL}));
            exchanges.push(StateExchange {
                request_id: format!("recall-{round}-{index}"),
                messages: histories[index].clone(),
                stream: (index + round) % 2 == 1,
                observation: observed(code),
            });
            histories[index].push(observed(code)["message"].clone());
        }
        evidence.recall_rounds.push(exchanges);
    }
    for stream in [false, true] {
        evidence.fresh.push(StateExchange {
            request_id: format!("fresh-{stream}"),
            messages: vec![json!({"role": "user", "content": EMPTY_RECALL})],
            stream,
            observation: observed(no_memory),
        });
    }
    evidence
}

#[test]
fn state_recall_and_reset_questions_do_not_supply_the_remembered_identifiers() {
    for code in CODES {
        assert!(remember(code).contains(code));
        assert!(!RECALL.contains(code));
        assert!(!EMPTY_RECALL.contains(code));
    }
    let mixed_case = "aB-7_x";
    assert!(remember(mixed_case).contains(mixed_case));
}

#[test]
fn state_recall_rejects_changed_case_and_truncated_identifiers() {
    for changed in ["Cobalt-731", "Cob"] {
        let mut records = run_fixture();
        records[3]["content"] = json!(changed);
        assert!(verify_run(&records, ModelReasoningProtocol::None, 16).is_err());

        for round in 0..2 {
            let mut evidence = serve_fixture();
            evidence.recall_rounds[round][0].observation["message"]["content"] = json!(changed);
            assert!(verify_serve(&evidence, ModelReasoningProtocol::None, 16).is_err());
        }
    }
}

#[test]
fn legacy_state_requests_cannot_be_relabelled_as_the_explicit_copying_probe() {
    // These are recorded requests from the previous probe, not an assertion
    // about the wording of the new contract. Even correct answers cannot turn
    // a different request/history into evidence for this probe.
    let legacy_write = "The code to remember is cobalt-731. Reply with only OK.";
    let legacy_recall = super::super::model_basic::RECALL_PROMPT;
    let legacy_empty = "What code did I ask you to remember in this conversation? Reply with only that code, or NONE if I have not given you a code.";
    for (index, prompt) in [(0, legacy_write), (2, legacy_recall), (5, legacy_empty)] {
        let mut records = run_fixture();
        records[index]["content"] = json!(prompt);
        assert!(verify_run(&records, ModelReasoningProtocol::None, 16).is_err());
    }
    let mutations: [fn(&mut ServeStateEvidence, &str); 3] = [
        |e: &mut ServeStateEvidence, prompt: &str| {
            e.writes[0].messages[0]["content"] = json!(prompt)
        },
        |e: &mut ServeStateEvidence, prompt: &str| {
            e.recall_rounds[0][0].messages[2]["content"] = json!(prompt)
        },
        |e: &mut ServeStateEvidence, prompt: &str| {
            e.fresh[0].messages[0]["content"] = json!(prompt)
        },
    ];
    for (mutate, prompt) in mutations
        .into_iter()
        .zip([legacy_write, legacy_recall, legacy_empty])
    {
        let mut evidence = serve_fixture();
        mutate(&mut evidence, prompt);
        assert!(verify_serve(&evidence, ModelReasoningProtocol::None, 16).is_err());
    }
}

#[test]
fn state_controls_ignore_ascii_case_but_remembered_codes_do_not() {
    for (acknowledged, no_memory) in [("ok", "None"), ("Ok.", "none"), ("**oK**", "`nOnE`.")] {
        verify_run(
            &run_fixture_with_controls(acknowledged, no_memory),
            ModelReasoningProtocol::None,
            16,
        )
        .unwrap();
        verify_serve(
            &serve_fixture_with_controls(acknowledged, no_memory),
            ModelReasoningProtocol::None,
            16,
        )
        .unwrap();
    }

    // An opaque code that happens to spell a control word is not a control.
    for (code, changed) in [("NONE", "None"), ("OK", "ok"), ("cobalt-731", "Cobalt-731")] {
        assert!(StateAnswer::Code(code).matches(code));
        assert!(!StateAnswer::Code(code).matches(changed));
    }
    for changed in ["cobalt-732", "cobalt 731", "cobalt- 731", "amber-284"] {
        assert!(!StateAnswer::Code("cobalt-731").matches(changed));
    }
}

#[test]
fn state_controls_reject_leaked_codes_extra_text_and_non_ascii_lookalikes() {
    for no_memory in [
        "",
        "cobalt-731",
        "amber-284",
        "None cobalt-731",
        "None of them",
        "None\nOK",
        "N O N E",
        "N\u{039f}NE", // Greek omicron is not the ASCII control word.
    ] {
        assert!(verify_run(
            &run_fixture_with_controls("ok", no_memory),
            ModelReasoningProtocol::None,
            16,
        )
        .is_err());
        assert!(verify_serve(
            &serve_fixture_with_controls("ok", no_memory),
            ModelReasoningProtocol::None,
            16,
        )
        .is_err());
    }
    for acknowledged in ["", "OK cobalt-731", "OK\nNONE", "\u{039f}K"] {
        assert!(!StateAnswer::Acknowledged.matches(acknowledged));
    }
}

#[test]
fn mixed_case_state_controls_preserve_protocol_history_and_completion_checks() {
    for mutate in [
        |records: &mut Vec<Value>| records[6]["finish_reason"] = json!("length"),
        |records: &mut Vec<Value>| records[6]["usage"]["total_tokens"] = json!(26),
        |records: &mut Vec<Value>| records[6]["reasoning_content"] = Value::Null,
        |records: &mut Vec<Value>| records[6]["history_epoch"] = json!(0),
        |records: &mut Vec<Value>| records[3]["content"] = json!("Cobalt-731"),
    ] {
        let mut records = run_fixture_with_controls("ok", "None");
        mutate(&mut records);
        assert!(verify_run(&records, ModelReasoningProtocol::None, 16).is_err());
    }
    for mutate in [
        |e: &mut ServeStateEvidence| e.fresh[0].observation["finish_reason"] = json!("length"),
        |e: &mut ServeStateEvidence| {
            e.fresh[0].observation["message"]["reasoning"] = json!("thought")
        },
        |e: &mut ServeStateEvidence| e.fresh[0].observation["usage"]["total_tokens"] = json!(26),
        |e: &mut ServeStateEvidence| {
            e.recall_rounds[0][0].messages[1] = observed("OK")["message"].clone()
        },
        |e: &mut ServeStateEvidence| e.fresh[0].request_id = e.writes[0].request_id.clone(),
        |e: &mut ServeStateEvidence| {
            e.recall_rounds[0][0].observation["message"]["content"] = json!("Cobalt-731")
        },
    ] {
        let mut evidence = serve_fixture_with_controls("ok", "None");
        mutate(&mut evidence);
        assert!(verify_serve(&evidence, ModelReasoningProtocol::None, 16).is_err());
    }
}

#[test]
fn serve_distinguishes_own_history_from_other_sessions_and_fresh_controls() {
    verify_serve(&serve_fixture(), ModelReasoningProtocol::None, 16).unwrap();
    for mutate in [
        |e: &mut ServeStateEvidence| {
            e.recall_rounds[0][1].observation["message"]["content"] = json!("cobalt-731");
        },
        |e: &mut ServeStateEvidence| {
            e.recall_rounds[1][0].messages = e.recall_rounds[1][1].messages.clone();
        },
        |e: &mut ServeStateEvidence| {
            e.recall_rounds[1][0].messages.pop();
        },
        |e: &mut ServeStateEvidence| {
            e.recall_rounds[1][0].stream = false;
        },
        |e: &mut ServeStateEvidence| {
            e.fresh[1].observation["message"]["content"] = json!("amber-284");
        },
        |e: &mut ServeStateEvidence| {
            e.fresh[0].messages = e.recall_rounds[1][0].messages.clone();
        },
        |e: &mut ServeStateEvidence| {
            e.fresh[0].request_id = e.writes[0].request_id.clone();
        },
        |e: &mut ServeStateEvidence| {
            e.fresh[0].observation["finish_reason"] = json!("length");
        },
        |e: &mut ServeStateEvidence| {
            e.fresh[0].observation["message"]["reasoning"] = json!("unexpected thought");
        },
        |e: &mut ServeStateEvidence| {
            e.recall_rounds.remove(1);
        },
    ] {
        let mut bad = serve_fixture();
        mutate(&mut bad);
        assert!(
            verify_serve(&bad, ModelReasoningProtocol::None, 16).is_err(),
            "{bad:?}"
        );
    }
    assert!(verify_serve(&serve_fixture(), ModelReasoningProtocol::None, 4).is_err());
}
