//! The exact first-user Chat request used by serving workloads and replay.
use ferrum_bench_core::env::HttpRequestSampling;
use ferrum_types::ReasoningEffort;

pub(super) fn chat_completion_body(
    model: &str,
    prompt_text: &str,
    max_tokens: usize,
    ignore_eos: bool,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
    sampling: HttpRequestSampling,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let serde_json::Value::Object(sampling) =
        serde_json::to_value(sampling).expect("validated HTTP sampling must serialize")
    else {
        unreachable!("HTTP sampling serializes as an object");
    };
    body.as_object_mut()
        .expect("request body is an object")
        .extend(sampling);
    let mut options = serde_json::Map::new();
    if let Some(thinking) = enable_thinking {
        options.insert("enable_thinking".into(), thinking.into());
    }
    if let Some(effort) = reasoning_effort {
        options.insert("reasoning_effort".into(), serde_json::json!(effort));
    }
    if !options.is_empty() {
        body["chat_template_kwargs"] = serde_json::Value::Object(options);
    }
    if ignore_eos {
        body["ignore_eos"] = true.into();
    }
    body
}
