use super::*;
use crate::protocol;
use sha2::{Digest, Sha256};

fn wire(message: &Value, finish: &str, tokens: u64, stream: bool) -> String {
    let usage =
        json!({"prompt_tokens": 5, "completion_tokens": tokens, "total_tokens": 5 + tokens});
    if !stream {
        return json!({"choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}).to_string();
    }
    [
        json!({"id": "stop", "choices": [{"index": 0, "delta": message, "finish_reason": null}]}),
        json!({"id": "stop", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}),
        json!({"id": "stop", "choices": [], "usage": usage}),
    ]
    .into_iter()
    .map(|value| format!("data: {value}\n\n"))
    .collect::<String>()
        + "data: [DONE]\n\n"
}

fn parse_wire(message: &Value, finish: &str, tokens: u64, stream: bool) -> Value {
    let text = wire(message, finish, tokens, stream);
    http_observation(
        &if stream {
            protocol::stream(&text)
        } else {
            protocol::sync(&text)
        }
        .unwrap(),
    )
}

#[test]
fn sync_and_sse_stop_oracles_accept_length_thought_baseline_but_not_a_final_claim() {
    let thought =
        "I will consider the quiet street and the reflections before writing the final paragraph.";
    for stream in [false, true] {
        let baseline = parse_wire(
            &json!({"role": "assistant", "content": "", "reasoning": thought}),
            "length",
            512,
            false,
        );
        let boundary = select_stop_boundary(&baseline).unwrap();
        assert_eq!(boundary.channel, StopChannel::Reasoning);
        let message =
            json!({"role": "assistant", "content": "", "reasoning": boundary.expected_prefix});
        let output = parse_wire(&message, "stop", 100, stream);
        verify_stop_observations(&baseline, &output, &boundary, 512).unwrap();
        for (field, value) in [
            ("content", json!("invented final answer")),
            ("reasoning", json!(thought)),
        ] {
            let mut invalid = message.clone();
            invalid[field] = value;
            assert!(verify_stop_observations(
                &baseline,
                &parse_wire(&invalid, "stop", 100, stream),
                &boundary,
                512
            )
            .is_err());
        }
        for (finish, tokens) in [("length", 512), ("stop", 512)] {
            assert!(verify_stop_observations(
                &baseline,
                &parse_wire(&message, finish, tokens, stream),
                &boundary,
                512
            )
            .is_err());
        }
        if stream {
            let truncated = wire(&message, "stop", 100, true).replace("data: [DONE]\n\n", "");
            assert!(protocol::stream(&truncated).is_err());
        }
    }
}

fn jsonl_run(content: &str, reasoning: &str, raw: &str, finish: &str, tokens: u64) -> Run {
    let values = [
        json!({"event": "ready"}),
        json!({"event": "assistant_delta", "request_id": "stop", "raw_text_delta": raw}),
        json!({"event": "assistant", "request_id": "stop", "content": content, "reasoning": reasoning,
            "finish_reason": finish, "usage": {"prompt_tokens": 5, "completion_tokens": tokens, "total_tokens": tokens + 5},
            "raw_text_sha256": format!("{:x}", Sha256::digest(raw.trim().as_bytes()))}),
        json!({"event": "exit"}),
    ];
    let records = protocol::run_records(
        &values
            .iter()
            .map(|value| format!("{value}\n"))
            .collect::<String>(),
    )
    .unwrap();
    Run {
        ready: records[0].clone(),
        assistants: records
            .iter()
            .filter(|value| value["event"] == "assistant")
            .cloned()
            .collect(),
        records,
    }
}

#[test]
fn run_raw_prefix_keeps_native_framing_and_detects_text_hidden_by_final_fields() {
    let thought =
        "The user wants exactly alpha beta gamma delta epsilon, so I will copy that whole answer.";
    let content = "alpha beta gamma delta epsilon";
    let raw = format!("<think>\n{thought}\n</think>\n{content}");
    let baseline = run_observation(&jsonl_run(content, thought, &raw, "eos", 100)).unwrap();
    let boundary = select_stop_boundary(&baseline).unwrap();
    assert_eq!(boundary.channel, StopChannel::Reasoning);
    let expected_raw = &raw[..raw.find(&boundary.stop).unwrap()];
    let output = run_observation(&jsonl_run(
        "",
        &boundary.expected_prefix,
        expected_raw,
        "stop",
        30,
    ))
    .unwrap();
    verify_stop_observations(&baseline, &output, &boundary, 512).unwrap();
    verify_stop_raw_observations(&baseline, &output, &boundary).unwrap();
    for alias in [Value::Null, json!(thought)] {
        let mut run = jsonl_run("", &boundary.expected_prefix, expected_raw, "stop", 30);
        for record in run.records.iter_mut().chain(run.assistants.iter_mut()) {
            if record["event"] == "assistant" {
                record["reasoning_content"] = alias.clone();
            }
        }
        let aliased = run_observation(&run).unwrap();
        assert!(verify_stop_observations(&baseline, &aliased, &boundary, 512).is_err());
    }
    let leaked_raw = format!("{expected_raw}{}", boundary.stop);
    let leaked = run_observation(&jsonl_run(
        "",
        &boundary.expected_prefix,
        &leaked_raw,
        "stop",
        30,
    ))
    .unwrap();
    assert!(verify_stop_raw_observations(&baseline, &leaked, &boundary).is_err());
}

#[test]
fn a_length_baseline_is_only_accepted_on_the_dedicated_stop_capture_path() {
    assert!(!RunCaptureMode::Natural.accepts(Some("length")));
    assert!(!RunCaptureMode::State.accepts(Some("length")));
    assert!(RunCaptureMode::Stop {
        disable_thinking: false
    }
    .accepts(Some("length")));
    for mode in [
        RunCaptureMode::Natural,
        RunCaptureMode::State,
        RunCaptureMode::Stop {
            disable_thinking: true,
        },
    ] {
        assert!(!mode.accepts(Some("error")));
        assert!(mode.accepts(Some("stop")));
        assert!(mode.accepts(Some("eos")));
    }
}
