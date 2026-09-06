use super::*;
use clap::Parser;

fn wire(message: Value, finish: &str, tokens: u64, stream: bool) -> String {
    let usage =
        json!({"prompt_tokens": 5, "completion_tokens": tokens, "total_tokens": 5 + tokens});
    if !stream {
        return json!({"choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}).to_string();
    }
    let mut text = String::new();
    for value in [
        json!({"id": "boundary", "choices": [{"index": 0, "delta": message, "finish_reason": null}]}),
        json!({"id": "boundary", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}),
        json!({"id": "boundary", "choices": [], "usage": usage}),
    ] {
        text.push_str(&format!("data: {value}\n\n"));
    }
    text.push_str("data: [DONE]\n\n");
    text
}

fn parse(text: &str, stream: bool) -> Result<Value> {
    Ok(observed(if stream {
        protocol::stream(text)?
    } else {
        protocol::sync(text)?
    }))
}

#[test]
fn actual_sync_and_sse_reasoning_oracles_require_thought_and_separated_final_answer() {
    for stream in [false, true] {
        let valid = json!({"role": "assistant", "content": "42", "reasoning": "17 plus 20 is 37; add the remaining 5."});
        verify_reasoning_observation(
            &parse(&wire(valid.clone(), "stop", 24, stream), stream).unwrap(),
        )
        .unwrap();
        for (field, value) in [
            ("reasoning", Value::Null),
            ("reasoning", json!("")),
            ("content", json!("17 plus 20 is 37. 42")),
            ("content", json!("<|channel>final\n42")),
        ] {
            let mut invalid = valid.clone();
            invalid[field] = value;
            assert!(verify_reasoning_observation(
                &parse(&wire(invalid, "stop", 24, stream), stream).unwrap()
            )
            .is_err());
        }
        let wire = wire(valid, "stop", 24, stream);
        if stream {
            assert!(parse(&wire.replace("data: [DONE]\n\n", ""), true).is_err());
            assert!(parse(&format!("{wire}data: [DONE]\n\n"), true).is_err());
        }
    }
}

#[test]
fn length_wire_oracle_rejects_natural_finish_wrong_budget_and_truncated_headers() {
    for stream in [false, true] {
        let baseline = parse(
            &wire(
                json!({"role": "assistant", "content": "alpha beta gamma delta"}),
                "stop",
                8,
                false,
            ),
            false,
        )
        .unwrap();
        let budget = length_probe_budget(8).unwrap();
        let partial = json!({"role": "assistant", "content": "alpha beta gamma"});
        let actual = parse(&wire(partial.clone(), "length", 6, stream), stream).unwrap();
        verify_length_observations(&baseline, &actual, budget).unwrap();
        for (message, finish, tokens) in [
            (partial, "stop", 6),
            (
                json!({"role": "assistant", "content": "alpha beta"}),
                "length",
                5,
            ),
            (
                json!({"role": "assistant", "content": "", "reasoning": "only thought so far"}),
                "length",
                6,
            ),
            (
                json!({"role": "assistant", "content": "<|channel>final"}),
                "length",
                6,
            ),
            (
                json!({"role": "assistant", "content": "alpha beta gamma delta"}),
                "length",
                6,
            ),
        ] {
            let actual = parse(&wire(message, finish, tokens, stream), stream).unwrap();
            assert!(verify_length_observations(&baseline, &actual, budget).is_err());
        }
    }
}

#[test]
fn explicit_probe_modes_preserve_quick_start_base_arguments() {
    let args = Args::try_parse_from([
        "model-regression",
        "--ferrum-bin",
        "fixture",
        "--model",
        "fixture:alias",
        "--backend",
        "metal",
        "--report-dir",
        "fixture",
        "--disable-thinking",
        "--use-default-backend",
    ])
    .unwrap();
    let argv = run_argv(&args, REASONING_PROMPT, true, 64);
    assert!(argv.iter().any(|arg| arg == "--enable-thinking"));
    assert!(!argv
        .iter()
        .any(|arg| arg == "--disable-thinking" || arg == "--backend"));
    assert!(argv.windows(2).any(|pair| pair == ["--max-tokens", "64"]));
    assert!(args
        .common_args("run")
        .iter()
        .any(|arg| arg == "--disable-thinking"));
    let argv = run_argv(&args, LENGTH_PROMPT, false, 12);
    assert!(argv.iter().any(|arg| arg == "--disable-thinking"));
    assert!(!argv.iter().any(|arg| arg == "--enable-thinking"));
}

#[test]
fn actual_wire_absence_is_not_a_reasoning_positive_and_rejects_thought_in_each_mode() {
    use ferrum_bench_core::release_regression::model_tasks::verify_reasoning_absence_observation;
    for stream in [false, true] {
        let clean = parse(
            &wire(
                json!({"role": "assistant", "content": "42"}),
                "stop",
                3,
                stream,
            ),
            stream,
        )
        .unwrap();
        verify_reasoning_absence_observation(&clean, "42").unwrap();
        assert!(verify_reasoning_observation(&clean).is_err());
        for message in [
            json!({"role": "assistant", "content": "42", "reasoning": "private thought"}),
            json!({"role": "assistant", "content": "<think>private</think>42"}),
        ] {
            let leaked = parse(&wire(message, "stop", 3, stream), stream).unwrap();
            assert!(verify_reasoning_absence_observation(&leaked, "42").is_err());
        }
    }
}
