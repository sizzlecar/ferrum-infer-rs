use super::*;
use serde_json::json;

fn observation(reasoning: &str, content: &str, finish: &str, tokens: u64) -> Value {
    json!({"message":{"role":"assistant","reasoning":reasoning,"content":content},
        "finish_reason":finish,"usage":{"prompt_tokens":17,"completion_tokens":tokens,"total_tokens":17+tokens}})
}
fn replay(baseline: &Value, boundary: &StopBoundary, tokens: u64) -> Value {
    match boundary.channel {
        StopChannel::Reasoning => observation(&boundary.expected_prefix, "", "stop", tokens),
        StopChannel::Final => observation(
            baseline["message"]["reasoning"].as_str().unwrap(),
            &boundary.expected_prefix,
            "stop",
            tokens,
        ),
    }
}

#[test]
fn a_copied_final_draft_is_truthfully_classified_as_reasoning() {
    let content = "alpha beta gamma delta epsilon";
    let baseline = observation(
        &format!("The requested draft is {content}. I will emit that exact draft now."),
        content,
        "eos",
        90,
    );
    let boundary = select_stop_boundary(&baseline).unwrap();
    assert_eq!(boundary.channel, StopChannel::Reasoning);
    verify_stop_observations(&baseline, &replay(&baseline, &boundary, 20), &boundary, 512).unwrap();
    let mut forged_final = boundary.clone();
    forged_final.channel = StopChannel::Final;
    assert!(verify_stop_observations(
        &baseline,
        &replay(&baseline, &boundary, 20),
        &forged_final,
        512
    )
    .is_err());
}

#[test]
fn a_reasoning_only_length_baseline_can_prove_an_earlier_reasoning_stop() {
    let mut baseline = observation("I need to consider the rolling waves and the lighthouse before drafting the requested paragraph.", "", "length", 512);
    baseline["message"]["content"] = Value::Null;
    let boundary = select_stop_boundary(&baseline).unwrap();
    assert_eq!(boundary.channel, StopChannel::Reasoning);
    let stopped = replay(&baseline, &boundary, 40);
    verify_stop_observations(&baseline, &stopped, &boundary, 512).unwrap();
    assert!(verify_stop_observations(&baseline, &stopped, &boundary, 513).is_err());
    assert!(verify_stop_observations(&baseline, &stopped, &boundary, 511).is_err());
}

#[test]
fn an_independent_final_segment_preserves_preceding_reasoning_exactly() {
    let baseline = observation(
        "\nI will describe the scene.\n",
        "Wind battered the lighthouse while a distant ship searched for the shore.\n",
        "stop",
        100,
    );
    let boundary = select_stop_boundary(&baseline).unwrap();
    assert_eq!(boundary.channel, StopChannel::Final);
    assert_eq!(
        boundary.expected_prefix.trim_end(),
        boundary.expected_prefix
    );
    assert!(!baseline["message"]["reasoning"]
        .as_str()
        .unwrap()
        .contains(&boundary.stop));
    let stopped = replay(&baseline, &boundary, 60);
    verify_stop_observations(&baseline, &stopped, &boundary, 512).unwrap();
    let mut changed_reasoning = stopped;
    changed_reasoning["message"]["reasoning"] = json!("I will describe the scene.");
    assert!(verify_stop_observations(&baseline, &changed_reasoning, &boundary, 512).is_err());
}

#[test]
fn short_unicode_answers_keep_exact_prefix_bytes_and_nonempty_trailing_output() {
    let baseline = observation("", "灯塔照亮风雨中的海岸。\n", "eos", 24);
    let boundary = select_stop_boundary(&baseline).unwrap();
    assert_eq!(boundary.channel, StopChannel::Final);
    verify_stop_observations(&baseline, &replay(&baseline, &boundary, 10), &boundary, 512).unwrap();
    assert!(select_stop_boundary(&observation("", "42", "stop", 2)).is_err());
    assert!(select_stop_boundary(&observation("", "   ", "stop", 2)).is_err());
}

#[test]
fn usage_finish_prefix_tool_and_sentinel_failures_cannot_pass() {
    let baseline = observation(
        "Earlier planning is complete.",
        "The bright lighthouse cast a steady beam across the harbor.",
        "stop",
        100,
    );
    let boundary = select_stop_boundary(&baseline).unwrap();
    let good = replay(&baseline, &boundary, 40);
    for (pointer, value) in [
        ("/finish_reason", json!("eos")),
        ("/finish_reason", json!("length")),
        ("/usage/completion_tokens", json!(100)),
        ("/usage/completion_tokens", json!(0)),
        ("/usage/total_tokens", json!(99)),
        ("/usage/prompt_tokens", json!(18)),
        (
            "/message/content",
            json!(format!("{} ", boundary.expected_prefix)),
        ),
        (
            "/message/content",
            json!(format!("{}{}", boundary.expected_prefix, boundary.stop)),
        ),
        ("/message/reasoning", json!("Different planning")),
        ("/message/role", json!("tool")),
    ] {
        let mut wrong = good.clone();
        *wrong.pointer_mut(pointer).unwrap() = value;
        assert!(
            verify_stop_observations(&baseline, &wrong, &boundary, 512).is_err(),
            "{pointer}: {wrong}"
        );
    }
    // Self-consistent usage still must prove earlier engine termination.
    let unchanged_usage = replay(&baseline, &boundary, 100);
    assert!(verify_stop_observations(&baseline, &unchanged_usage, &boundary, 512).is_err());
    let mut different_prompt = good.clone();
    different_prompt["usage"] =
        json!({"prompt_tokens":18,"completion_tokens":40,"total_tokens":58});
    assert!(verify_stop_observations(&baseline, &different_prompt, &boundary, 512).is_err());
    for field in ["tool_calls", "reasoning_content"] {
        let mut wrong = good.clone();
        wrong["message"][field] = json!([{"name":"unexpected"}]);
        assert!(verify_stop_observations(&baseline, &wrong, &boundary, 512).is_err());
    }
}

#[test]
fn reasoning_replay_cannot_emit_final_content_or_relabel_a_later_occurrence() {
    let baseline = observation(
        "The internal draft uses amber lanterns before proceeding to its conclusion.",
        "amber lanterns",
        "stop",
        90,
    );
    let boundary = select_stop_boundary(&baseline).unwrap();
    assert_eq!(boundary.channel, StopChannel::Reasoning);
    let mut stopped = replay(&baseline, &boundary, 20);
    stopped["message"]["content"] = json!("a later answer");
    assert!(verify_stop_observations(&baseline, &stopped, &boundary, 512).is_err());
    let false_final = StopBoundary {
        channel: StopChannel::Final,
        stop: "lantern".into(),
        expected_prefix: "amber ".into(),
    };
    assert!(verify_stop_observations(
        &baseline,
        &observation(
            baseline["message"]["reasoning"].as_str().unwrap(),
            "amber",
            "stop",
            40
        ),
        &false_final,
        512
    )
    .is_err());
}

#[test]
fn arbitrary_suffix_or_missing_prefix_is_not_an_internal_boundary() {
    let baseline = observation("", "abcdefabcdefXYZ", "stop", 40);
    for boundary in [
        StopBoundary {
            channel: StopChannel::Final,
            stop: "XYZ".into(),
            expected_prefix: "abcdefabcdef".into(),
        },
        StopBoundary {
            channel: StopChannel::Final,
            stop: "abc".into(),
            expected_prefix: "abcdef".into(),
        },
        StopBoundary {
            channel: StopChannel::Final,
            stop: "abc".into(),
            expected_prefix: "".into(),
        },
    ] {
        assert!(verify_stop_observations(
            &baseline,
            &replay(&baseline, &boundary, 10),
            &boundary,
            512
        )
        .is_err());
    }
    let mut overflowing = baseline.clone();
    overflowing["usage"] = json!({"prompt_tokens":u64::MAX,"completion_tokens":1,"total_tokens":0});
    assert!(select_stop_boundary(&overflowing).is_err());
}

#[test]
fn selection_work_limit_does_not_reject_a_long_legitimate_stop_sentinel() {
    let stop = "a long unique stop marker extending beyond the selector candidate width";
    let baseline = observation(
        "Unchanged preceding reasoning.",
        &format!("prefix{stop}remaining visible output"),
        "stop",
        100,
    );
    let boundary = StopBoundary {
        channel: StopChannel::Final,
        stop: stop.into(),
        expected_prefix: "prefix".into(),
    };
    verify_stop_observations(&baseline, &replay(&baseline, &boundary, 60), &boundary, 512).unwrap();
}
