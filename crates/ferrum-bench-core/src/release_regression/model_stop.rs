//! Parsed-channel stop evidence. Reasoning and final text are never concatenated
//! into invented raw model output; each probe proves its actual stopped channel.
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopChannel {
    Reasoning,
    Final,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopMode {
    TaskDefault,
    DisabledThinking,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StopBoundary {
    pub channel: StopChannel,
    pub stop: String,
    pub expected_prefix: String,
}

const MAX_SENTINEL_CHARS: usize = 24;

fn text<'a>(message: &'a Value, field: &str) -> Result<&'a str, String> {
    match &message[field] {
        Value::Null => Ok(""),
        Value::String(text) => Ok(text),
        _ => Err(format!("stop observation has invalid {field}")),
    }
}

fn channels(observation: &Value) -> Result<(&str, &str), String> {
    let message = &observation["message"];
    if message["role"] != "assistant"
        || message.get("reasoning_content").is_some()
        || !(message["tool_calls"].is_null()
            || message["tool_calls"].as_array().is_some_and(Vec::is_empty))
    {
        return Err("stop observation must be a canonical assistant without tool calls".into());
    }
    Ok((text(message, "reasoning")?, text(message, "content")?))
}

fn usage(observation: &Value) -> Result<(u64, u64), String> {
    let usage = &observation["usage"];
    let prompt = usage["prompt_tokens"].as_u64().filter(|tokens| *tokens > 0);
    let completion = usage["completion_tokens"]
        .as_u64()
        .filter(|tokens| *tokens > 0);
    match (prompt, completion) {
        (Some(prompt), Some(completion))
            if prompt.checked_add(completion).is_some()
                && prompt.checked_add(completion) == usage["total_tokens"].as_u64() =>
        {
            Ok((prompt, completion))
        }
        _ => Err("stop observation has missing, zero, overflowing or inconsistent usage".into()),
    }
}

fn baseline_channels(baseline: &Value) -> Result<(&str, &str), String> {
    if !matches!(
        baseline["finish_reason"].as_str(),
        Some("stop" | "eos" | "length")
    ) {
        return Err("stop baseline has no supported completion reason".into());
    }
    usage(baseline)?;
    channels(baseline)
}

fn boundary_in(text: &str, earlier: &str, channel: StopChannel) -> Option<StopBoundary> {
    // Leave a non-whitespace tail even when HTTP preserves trailing newlines.
    // This only bounds selection; the recorded and compared prefix is unchanged.
    let content_end = text.trim_end().len();
    let positions: Vec<_> = text
        .char_indices()
        .map(|(position, _)| position)
        .take_while(|position| *position < content_end)
        .chain(std::iter::once(content_end))
        .collect();
    let count = positions.len() - 1;
    if count < 3 {
        return None;
    }
    // Start inside the answer and retain real output beyond the sentinel. This
    // is a bounded substring, not a suffix that might only match natural EOS.
    let preferred = (count / 3).max(1);
    for index in (preferred..count - 1).chain(1..preferred) {
        let prefix = &text[..positions[index]];
        if prefix.trim().is_empty() || prefix.trim_end() != prefix {
            continue;
        }
        let end = (index + MAX_SENTINEL_CHARS).min(count - 1);
        let stop = &text[positions[index]..positions[end]];
        if stop.trim().is_empty()
            || text[positions[end]..].trim().is_empty()
            || text.find(stop) != Some(positions[index])
            || earlier.contains(stop)
        {
            continue;
        }
        return Some(StopBoundary {
            channel,
            stop: stop.into(),
            expected_prefix: prefix.into(),
        });
    }
    None
}

pub fn select_stop_boundary(baseline: &Value) -> Result<StopBoundary, String> {
    let (reasoning, content) = baseline_channels(baseline)?;
    boundary_in(content, reasoning, StopChannel::Final)
        .or_else(|| boundary_in(reasoning, "", StopChannel::Reasoning))
        .ok_or_else(|| {
            "baseline has no observable internal stop boundary with a nonempty prefix and tail"
                .into()
        })
}

pub fn verify_stop_observations(
    baseline: &Value,
    replay: &Value,
    boundary: &StopBoundary,
    max_tokens: u32,
) -> Result<(), String> {
    let (baseline_reasoning, baseline_content) = baseline_channels(baseline)?;
    let (replay_reasoning, replay_content) = channels(replay)?;
    let (prompt, baseline_tokens) = usage(baseline)?;
    let (replay_prompt, replay_tokens) = usage(replay)?;
    if max_tokens == 0
        || baseline_tokens > u64::from(max_tokens)
        || (baseline["finish_reason"] == "length" && baseline_tokens != u64::from(max_tokens))
        || replay["finish_reason"] != "stop"
        || replay_prompt != prompt
        || replay_tokens >= baseline_tokens
        || replay_tokens > u64::from(max_tokens)
    {
        return Err(
            "stop replay must stop earlier with unchanged prompt usage and valid token budget"
                .into(),
        );
    }
    if boundary.stop.trim().is_empty()
        || boundary.expected_prefix.trim().is_empty()
        || boundary.expected_prefix.trim_end() != boundary.expected_prefix
    {
        return Err("stop boundary is empty or loses prefix bytes to display trimming".into());
    }
    let (domain, earlier) = match boundary.channel {
        StopChannel::Reasoning => (baseline_reasoning, ""),
        StopChannel::Final => (baseline_content, baseline_reasoning),
    };
    let start = domain
        .find(&boundary.stop)
        .ok_or("stop sentinel is absent from its declared channel")?;
    if domain[..start] != boundary.expected_prefix
        || domain[start + boundary.stop.len()..].trim().is_empty()
        || earlier.contains(&boundary.stop)
    {
        return Err("stop boundary does not match the first occurrence in its real channel".into());
    }
    match boundary.channel {
        StopChannel::Reasoning
            if replay_reasoning != boundary.expected_prefix || !replay_content.is_empty() =>
        {
            return Err(
                "reasoning stop must preserve its exact prefix and emit no later final content"
                    .into(),
            )
        }
        StopChannel::Final
            if replay_content != boundary.expected_prefix
                || replay_reasoning != baseline_reasoning =>
        {
            return Err(
                "final stop must preserve its exact prefix and all preceding reasoning".into(),
            )
        }
        _ => {}
    }
    if replay_reasoning.contains(&boundary.stop) || replay_content.contains(&boundary.stop) {
        return Err("stop sentinel leaked into replay output".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "model_stop_tests.rs"]
mod tests;
