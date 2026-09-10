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
    let mut records = Vec::new();
    turn(&mut records, 0, 0, &remember("cobalt-731"), "OK");
    turn(&mut records, 0, 1, RECALL, "cobalt-731");
    reset(&mut records, 1, 4);
    turn(&mut records, 1, 0, EMPTY_RECALL, "NONE");
    turn(&mut records, 1, 1, &remember("amber-284"), "OK");
    turn(&mut records, 1, 2, RECALL, "amber-284");
    reset(&mut records, 2, 6);
    turn(&mut records, 2, 0, EMPTY_RECALL, "NONE");
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
            observation: observed("OK"),
        });
        messages.push(observed("OK")["message"].clone());
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
            observation: observed("NONE"),
        });
    }
    evidence
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
