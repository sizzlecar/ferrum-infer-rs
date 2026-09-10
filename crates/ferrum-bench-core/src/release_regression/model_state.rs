//! Observable model-state continuity, history reset and conversation isolation.
//! These product checks do not establish a cache hit, physical KV layout or
//! arithmetic tolerances; backend numerical and state-boundary tests remain separate.
use super::model_tasks::{boundary_observation, probe_answer_matches};
use ferrum_types::ModelReasoningProtocol;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;

pub const CODES: [&str; 2] = ["cobalt-731", "amber-284"];
pub const RECALL: &str = super::model_basic::RECALL_PROMPT;
pub const EMPTY_RECALL: &str = "What code did I ask you to remember in this conversation? Reply with only that code, or NONE if I have not given you a code.";

pub fn remember(code: &str) -> String {
    format!("The code to remember is {code}. Reply with only OK.")
}

#[derive(Debug, Clone)]
pub enum RunStep {
    Turn {
        prompt: String,
        answer: &'static str,
    },
    Reset,
}

pub fn run_steps() -> Vec<RunStep> {
    let mut steps = Vec::new();
    for code in CODES {
        steps.extend([
            RunStep::Turn {
                prompt: remember(code),
                answer: "OK",
            },
            RunStep::Turn {
                prompt: RECALL.into(),
                answer: code,
            },
            RunStep::Reset,
            RunStep::Turn {
                prompt: EMPTY_RECALL.into(),
                answer: "NONE",
            },
        ]);
    }
    steps
}

fn answer(
    observation: &Value,
    expected: &str,
    protocol: ModelReasoningProtocol,
    max_tokens: u32,
) -> Result<(), String> {
    let (content, reasoning, tokens) = boundary_observation(observation)?;
    if !matches!(observation["finish_reason"].as_str(), Some("stop" | "eos"))
        || !probe_answer_matches(content, expected)
        || tokens > u64::from(max_tokens)
        || (protocol == ModelReasoningProtocol::None && !reasoning.is_empty())
    {
        return Err(format!(
            "state probe expected a completed {expected:?}, received {content:?}"
        ));
    }
    Ok(())
}

pub fn verify_run(
    records: &[Value],
    protocol: ModelReasoningProtocol,
    max_tokens: u32,
) -> Result<(), String> {
    if records.iter().any(|record| record["event"] == "error") {
        return Err("state run contains a product error".into());
    }
    let mut records = records.iter().filter(|r| {
        matches!(
            r["event"].as_str(),
            Some("user" | "assistant" | "history_reset")
        )
    });
    let mut epoch = 0u64;
    let mut turn = 0u64;
    let mut session = None;
    let mut ids = BTreeSet::new();
    for step in run_steps() {
        match step {
            RunStep::Turn {
                prompt,
                answer: expected,
            } => {
                let user = records.next().ok_or("state run is missing a user turn")?;
                let assistant = records
                    .next()
                    .ok_or("state run is missing an assistant turn")?;
                let id = user["request_id"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or("state run has no request identity")?;
                let current_session = user["session_id"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or("state run has no session identity")?;
                let expected_session = *session.get_or_insert(current_session);
                if user["event"] != "user"
                    || user["content"] != prompt
                    || assistant["event"] != "assistant"
                    || assistant["request_id"] != id
                    || !ids.insert(id)
                    || [user, assistant].iter().any(|r| {
                        r["session_id"] != expected_session
                            || r["history_epoch"] != epoch
                            || r["turn"] != turn
                    })
                    || user["history_before"]["message_count"] != 2 * turn
                {
                    return Err("state run replay, identity, order or history epoch differs from the actual conversation".into());
                }
                let mut observation = json!({"message": {"role": "assistant", "content": assistant["content"], "reasoning": assistant["reasoning"], "tool_calls": assistant["tool_calls"]}, "finish_reason": assistant["finish_reason"], "usage": assistant["usage"]});
                if let Some(alias) = assistant.get("reasoning_content") {
                    observation["message"]["reasoning_content"] = alias.clone();
                }
                answer(&observation, expected, protocol, max_tokens)?;
                turn += 1;
            }
            RunStep::Reset => {
                let reset = records
                    .next()
                    .ok_or("state run did not record history reset")?;
                epoch += 1;
                if reset["event"] != "history_reset"
                    || reset["session_id"].as_str() != session
                    || reset["history_epoch"] != epoch
                    || reset["turn"] != 0
                    || reset["history_before"]["message_count"] != 2 * turn
                    || reset["history_after"]["message_count"] != 0
                    || reset["history_after"]["turn_count"] != 0
                {
                    return Err(
                        "state run did not clear the previous conversation before recall".into(),
                    );
                }
                turn = 0;
            }
        }
    }
    if records.next().is_some() {
        return Err("state run contains an unexpected extra turn or reset".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateExchange {
    pub request_id: String,
    pub messages: Vec<Value>,
    pub stream: bool,
    pub observation: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServeStateEvidence {
    pub writes: Vec<StateExchange>,
    /// Each round submits the two independent histories together. The second
    /// round swaps wire modes and replays each conversation's actual answer.
    pub recall_rounds: Vec<Vec<StateExchange>>,
    pub fresh: Vec<StateExchange>,
}

fn exchange(
    exchange: &StateExchange,
    expected_messages: &[Value],
    expected: &str,
    stream: bool,
    ids: &mut BTreeSet<String>,
    protocol: ModelReasoningProtocol,
    max_tokens: u32,
) -> Result<(), String> {
    if exchange.request_id.trim().is_empty()
        || !ids.insert(exchange.request_id.clone())
        || exchange.messages != expected_messages
        || exchange.stream != stream
    {
        return Err("state exchange has a reused request, wrong wire mode or another conversation's history".into());
    }
    answer(&exchange.observation, expected, protocol, max_tokens)
}

pub fn verify_serve(
    evidence: &ServeStateEvidence,
    protocol: ModelReasoningProtocol,
    max_tokens: u32,
) -> Result<(), String> {
    if evidence.writes.len() != CODES.len()
        || evidence.recall_rounds.len() != 2
        || evidence.fresh.len() != 2
    {
        return Err("state serve is missing the distinct histories, swapped wire modes or fresh conversation controls".into());
    }
    let mut ids = BTreeSet::new();
    let mut histories = Vec::new();
    for (index, code) in CODES.iter().enumerate() {
        let mut messages = vec![json!({"role": "user", "content": remember(code)})];
        exchange(
            &evidence.writes[index],
            &messages,
            "OK",
            index == 1,
            &mut ids,
            protocol,
            max_tokens,
        )?;
        messages.push(evidence.writes[index].observation["message"].clone());
        histories.push(messages);
    }
    for (round, exchanges) in evidence.recall_rounds.iter().enumerate() {
        if exchanges.len() != histories.len() {
            return Err("state serve lost an interleaved conversation".into());
        }
        for (index, history) in histories.iter_mut().enumerate() {
            history.push(json!({"role": "user", "content": RECALL}));
            exchange(
                &exchanges[index],
                history,
                CODES[index],
                (index + round) % 2 == 1,
                &mut ids,
                protocol,
                max_tokens,
            )?;
            history.push(exchanges[index].observation["message"].clone());
        }
    }
    for (index, fresh) in evidence.fresh.iter().enumerate() {
        exchange(
            fresh,
            &[json!({"role": "user", "content": EMPTY_RECALL})],
            "NONE",
            index == 1,
            &mut ids,
            protocol,
            max_tokens,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "model_state_tests.rs"]
pub(super) mod tests;
