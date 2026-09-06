//! Semantic checks for the runner's local integer-addition tool fixture.
//! Model arguments are parsed as a narrow grammar, never evaluated as code.
use super::model_tasks::{boundary_observation, probe_answer_matches};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegerAddition {
    pub left: u64,
    pub right: u64,
}

impl IntegerAddition {
    /// Exactly two unsigned decimal integers and one `+`, with optional
    /// whitespace and balanced parentheses around operands or the expression.
    pub fn parse(expression: &str) -> Result<Self, String> {
        let mut chars = expression.chars().peekable();
        let mut operands = Vec::new();
        let mut depth = 0usize;
        let mut expecting_operand = true;
        let mut saw_plus = false;
        while let Some(character) = chars.next() {
            match character {
                c if c.is_whitespace() => {}
                '(' if expecting_operand => depth += 1,
                ')' if !expecting_operand && depth > 0 => depth -= 1,
                '+' if !expecting_operand && !saw_plus => {
                    saw_plus = true;
                    expecting_operand = true;
                }
                '0'..='9' if expecting_operand && operands.len() < 2 => {
                    let mut number = u64::from(character.to_digit(10).unwrap());
                    while let Some(next @ '0'..='9') = chars.peek().copied() {
                        chars.next();
                        number = number
                            .checked_mul(10)
                            .and_then(|n| n.checked_add(u64::from(next.to_digit(10).unwrap())))
                            .ok_or("integer operand overflow")?;
                    }
                    operands.push(number);
                    expecting_operand = false;
                }
                _ => {
                    return Err(
                        "expected one unsigned integer addition with balanced parentheses".into(),
                    )
                }
            }
        }
        if depth != 0 || expecting_operand || !saw_plus || operands.len() != 2 {
            return Err("incomplete or unbalanced integer addition".into());
        }
        let addition = Self {
            left: operands[0],
            right: operands[1],
        };
        addition.result()?;
        Ok(addition)
    }

    pub fn result(self) -> Result<u64, String> {
        self.left
            .checked_add(self.right)
            .ok_or_else(|| "integer addition overflow".into())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedCalcCall {
    pub call: Value,
    pub result: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CalcArguments {
    expression: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolResult {
    result: u64,
}

fn verify_usage(observation: &Value, max_tokens: u32) -> Result<(), String> {
    let usage = &observation["usage"];
    let prompt = usage["prompt_tokens"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or("invalid prompt usage")?;
    let completion = usage["completion_tokens"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or("invalid completion usage")?;
    let total = prompt
        .checked_add(completion)
        .ok_or("tool observation usage overflow")?;
    if completion > u64::from(max_tokens) || usage["total_tokens"].as_u64() != Some(total) {
        return Err("tool observation exceeds its token budget or has inconsistent usage".into());
    }
    Ok(())
}

pub fn verify_calc_call(observation: &Value, max_tokens: u32) -> Result<ValidatedCalcCall, String> {
    verify_usage(observation, max_tokens)?;
    let message = &observation["message"];
    let content = match &message["content"] {
        Value::Null => "",
        Value::String(content) => content,
        _ => return Err("invalid calc handoff content".into()),
    };
    if observation["finish_reason"] != "tool_calls" || !message["function_call"].is_null() {
        return Err("calc handoff must terminate with canonical tool_calls".into());
    }
    let calls = message["tool_calls"]
        .as_array()
        .ok_or("missing calc tool call")?;
    if calls.len() != 1 {
        return Err("expected one calc tool call".into());
    }
    let call = &calls[0];
    if call["type"] != "function"
        || !call["id"].as_str().is_some_and(|id| !id.trim().is_empty())
        || call["function"]["name"] != "calc"
    {
        return Err("calc handoff has an invalid identity or selected the wrong tool".into());
    }
    // Reuse canonical text/framing validation without misclassifying the actual
    // tool call as an assistant answer. Reasoning remains the observed value.
    let mut text_observation = observation.clone();
    text_observation["message"]["content"] = Value::String(content.into());
    text_observation["message"]["tool_calls"] = Value::Null;
    boundary_observation(&text_observation)?;
    let arguments_json = call["function"]["arguments"]
        .as_str()
        .ok_or("missing calc arguments JSON")?;
    let arguments_value: Value = serde_json::from_str(arguments_json)
        .map_err(|error| format!("invalid calc arguments: {error}"))?;
    if !arguments_value.is_object() {
        return Err("calc arguments must be a JSON object".into());
    }
    // Deserialize the original bytes so duplicate fields are still rejected.
    // Serde's struct visitor alone also accepts a positional JSON array.
    let arguments: CalcArguments = serde_json::from_str(arguments_json)
        .map_err(|error| format!("invalid calc arguments: {error}"))?;
    let addition = IntegerAddition::parse(&arguments.expression)?;
    if !matches!((addition.left, addition.right), (123, 456) | (456, 123)) {
        return Err("calc expression did not add the requested operands 123 and 456".into());
    }
    Ok(ValidatedCalcCall {
        call: call.clone(),
        result: addition.result()?,
    })
}

fn verify_continuation(
    called: &Value,
    continuation: &Value,
    validated: &ValidatedCalcCall,
    max_tokens: u32,
    alias: bool,
) -> Result<(), String> {
    verify_usage(continuation, max_tokens)?;
    let (content, _, _) = boundary_observation(continuation)?;
    if continuation["finish_reason"] != "stop"
        || !continuation["message"]["function_call"].is_null()
        || !probe_answer_matches(content, &validated.result.to_string())
    {
        return Err(
            "tool continuation must naturally finish with the computed answer and no new tool call"
                .into(),
        );
    }
    if continuation["tool_call_id"] != validated.call["id"] {
        return Err("tool continuation did not replay its actual call identity".into());
    }
    let mut expected_assistant = called["message"].clone();
    if alias {
        let object = expected_assistant
            .as_object_mut()
            .ok_or("missing actual assistant history")?;
        let reasoning = object
            .remove("reasoning")
            .ok_or("alias replay needs actual reasoning")?;
        if !reasoning
            .as_str()
            .is_some_and(|text| !text.trim().is_empty())
        {
            return Err("alias replay needs nonempty actual tool-call reasoning".into());
        }
        object.insert("reasoning_content".into(), reasoning);
    }
    if continuation["replayed_assistant"] != expected_assistant {
        return Err(
            "replayed assistant history differs from the actual tool call or requested alias mode"
                .into(),
        );
    }
    let tool_message = &continuation["tool_result_message"];
    if tool_message["role"] != "tool" || tool_message["tool_call_id"] != validated.call["id"] {
        return Err("tool result message did not reference the actual call identity".into());
    }
    let result_json = tool_message["content"]
        .as_str()
        .ok_or("missing actual tool result message")?;
    let result_value: Value = serde_json::from_str(result_json)
        .map_err(|error| format!("invalid actual tool result: {error}"))?;
    if !result_value.is_object() {
        return Err("actual tool result must be a JSON object".into());
    }
    let result: ToolResult = serde_json::from_str(result_json)
        .map_err(|error| format!("invalid actual tool result: {error}"))?;
    if result.result != validated.result {
        return Err("actual tool result differs from the parsed calculation".into());
    }
    Ok(())
}

pub fn verify_tool_case(
    evidence: &Value,
    max_tokens: u32,
    reasoning_alias_replay: bool,
) -> Result<(), String> {
    if evidence["reasoning_alias_replayed"].as_bool() != Some(reasoning_alias_replay) {
        return Err("tool evidence does not match the requested reasoning alias mode".into());
    }
    for mode in ["sync", "stream"] {
        let called = &evidence[format!("{mode}_call")];
        let validated =
            verify_calc_call(called, max_tokens).map_err(|e| format!("{mode}_call: {e}"))?;
        if evidence["tool_result"].as_u64() != Some(validated.result) {
            return Err("recorded tool result differs from the parsed calculation".into());
        }
        verify_continuation(
            called,
            &evidence[format!("{mode}_continuation")],
            &validated,
            max_tokens,
            reasoning_alias_replay && mode == "stream",
        )
        .map_err(|e| format!("{mode}_continuation: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "model_tool_tests.rs"]
mod tests;
