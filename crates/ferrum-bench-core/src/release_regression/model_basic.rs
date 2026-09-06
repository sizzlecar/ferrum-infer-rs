//! Semantic observations for a real memory-write, arithmetic, and recall exchange.
//! These checks preserve model reasoning; they do not isolate which history field
//! supplied the recalled information or replace the runner's wire parsers.
use super::model_tasks::{boundary_observation, probe_answer_matches};
use ferrum_types::ModelReasoningProtocol;
use serde_json::{json, Value};

pub const MEMORY_PROMPT: &str = "The code to remember is cobalt-731. Reply with only OK.";
pub const ARITHMETIC_PROMPT: &str = "What is 17 + 25? Reply with only the number.";
pub const RECALL_PROMPT: &str = "What code did I ask you to remember? Reply with only that code.";

const RUN_STEPS: [(&str, &str); 3] = [
    ("memory_write", "OK"),
    ("arithmetic", "42"),
    ("recall", "cobalt-731"),
];
const SERVE_STEPS: [(&str, &str); 5] = [
    ("memory_write", "OK"),
    ("sync", "42"),
    ("stream", "42"),
    ("recall", "cobalt-731"),
    ("stream_recall", "cobalt-731"),
];

fn verify_answer(
    observation: &Value,
    expected_answer: &str,
    reasoning_protocol: ModelReasoningProtocol,
    max_tokens: u32,
    run: bool,
) -> Result<(), String> {
    let (content, reasoning, completion_tokens) = boundary_observation(observation)?;
    let finish = observation["finish_reason"].as_str();
    if finish != Some("stop") && !(run && finish == Some("eos")) {
        return Err("basic answer did not finish naturally".into());
    }
    if !probe_answer_matches(content, expected_answer) {
        return Err(format!(
            "basic expected {expected_answer:?}, received {content:?}"
        ));
    }
    if completion_tokens > u64::from(max_tokens) {
        return Err("basic completion exceeds the declared token budget".into());
    }
    if reasoning_protocol == ModelReasoningProtocol::None && !reasoning.is_empty() {
        return Err("non-thinking basic observation contains actual reasoning".into());
    }
    Ok(())
}

/// Check all three actual JSONL assistant records, including reasoning absence
/// when the loaded model declares that capability.
pub fn verify_basic_run(
    evidence: &Value,
    reasoning_protocol: ModelReasoningProtocol,
    max_tokens: u32,
) -> Result<(), String> {
    if evidence["prompts"] != json!([MEMORY_PROMPT, ARITHMETIC_PROMPT, RECALL_PROMPT]) {
        return Err("basic run prompts differ from the three actual task turns".into());
    }
    let answers = evidence["answers"]
        .as_array()
        .filter(|answers| answers.len() == RUN_STEPS.len())
        .ok_or("basic run requires memory-write, arithmetic, and recall observations")?;
    for (answer, (step, expected_answer)) in answers.iter().zip(RUN_STEPS) {
        let mut observation = json!({
            "message": {"role": "assistant", "content": answer["content"],
                "reasoning": answer["reasoning"], "tool_calls": answer["tool_calls"]},
            "finish_reason": answer["finish_reason"], "usage": answer["usage"]
        });
        // Preserve even a null alias so a malformed run record cannot become a
        // canonical observation merely through conversion by this consumer.
        if let Some(alias) = answer.get("reasoning_content") {
            observation["message"]["reasoning_content"] = alias.clone();
        }
        verify_answer(
            &observation,
            expected_answer,
            reasoning_protocol,
            max_tokens,
            true,
        )
        .map_err(|error| format!("basic run/{step}: {error}"))?;
    }
    Ok(())
}

/// Check the shared memory write and both actual HTTP arithmetic/recall paths.
pub fn verify_basic_serve(
    evidence: &Value,
    reasoning_protocol: ModelReasoningProtocol,
    max_tokens: u32,
) -> Result<(), String> {
    let observations = &evidence["observations"];
    for (step, expected_answer) in SERVE_STEPS {
        verify_answer(
            &observations[step],
            expected_answer,
            reasoning_protocol,
            max_tokens,
            false,
        )
        .map_err(|error| format!("basic serve/{step}: {error}"))?;
    }
    let memory_user = json!({"role": "user", "content": MEMORY_PROMPT});
    let memory_request = json!([memory_user]);
    if evidence["requests"]["memory_write"] != memory_request {
        return Err("basic memory-write request differs from its user instruction".into());
    }
    let arithmetic_history = vec![
        memory_user,
        observations["memory_write"]["message"].clone(),
        json!({"role": "user", "content": ARITHMETIC_PROMPT}),
    ];
    for (mode, recall_mode) in [("sync", "recall"), ("stream", "stream_recall")] {
        if evidence["requests"][mode] != json!(arithmetic_history) {
            return Err(format!(
                "basic {mode} did not replay its actual memory-write assistant"
            ));
        }
        let mut recall_history = arithmetic_history.clone();
        recall_history.push(observations[mode]["message"].clone());
        recall_history.push(json!({"role": "user", "content": RECALL_PROMPT}));
        if evidence["requests"][recall_mode] != json!(recall_history) {
            return Err(format!(
                "basic {recall_mode} did not replay its actual {mode} arithmetic assistant"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "model_basic_tests.rs"]
mod tests;
