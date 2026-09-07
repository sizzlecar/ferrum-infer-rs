//! Exercise the shared engine processor chain, including native handoff stops.
use super::*;
use ferrum_types::{
    ApiChatRequest, ApiFunction, ApiJsonSchema, ApiRequest, ApiResponseFormat, ApiTool,
    ApiToolCallProtocol, ModelOutputProtocol, ResponseFormat, StructuredOutputStart,
};
use serde_json::json;

const EOS: u32 = 3;
const ARGUMENTS: u32 = 9;
const FINAL: u32 = 8;
const MERGED_CALL: u32 = 10;
const MERGED_RETURN: u32 = 11;
const INVALID_ARGUMENTS: u32 = 12;
const INVALID_FINAL: u32 = 13;
const COMMENTARY: u32 = 14;
const FINAL_CHANNEL: u32 = 15;
const INCOMPLETE_ARGUMENTS: u32 = 19;
const INCOMPLETE_FINAL: u32 = 20;
const CHANNEL: u32 = 24;
const MESSAGE: u32 = 25;
const RETURN: u32 = 28;
const CALL: u32 = 29;
const UNRELATED_CONTROL: u32 = 31;
const MODEL_VOCAB: usize = 32;

fn tokenizer() -> Arc<dyn Tokenizer + Send + Sync> {
    let mut tokenizer = PolicyTokenizer::new(
        24,
        &[
            ("test", 0),
            (" ", 1),
            ("invalid prose", 2),
            ("<eos>", EOS),
            ("{", 4),
            ("}", 5),
            ("\"city\":\"Paris\"", 6),
            ("\"temperature\":21", 7),
            (r#"{"temperature":21}"#, FINAL),
            (r#"{"city":"Paris"}"#, ARGUMENTS),
            (r#"{"city":"Paris"}<|call|>"#, MERGED_CALL),
            (r#"{"temperature":21}<|return|>"#, MERGED_RETURN),
            (r#"{"city":7}"#, INVALID_ARGUMENTS),
            (r#"{"temperature":"warm"}"#, INVALID_FINAL),
            ("commentary to=functions.weather", COMMENTARY),
            ("final", FINAL_CHANNEL),
            ("assistant", 16),
            ("analysis", 17),
            ("reason", 18),
            (r#"{"city":""#, INCOMPLETE_ARGUMENTS),
            (r#"{"temperature":"#, INCOMPLETE_FINAL),
            ("Paris\"}", 21),
            ("21}", 22),
            ("json", 23),
            ("<|channel|>", CHANNEL),
            ("<|message|>", MESSAGE),
            ("<|start|>", 26),
            ("<|end|>", 27),
            ("<|return|>", RETURN),
            ("<|call|>", CALL),
            ("<|constrain|>", 30),
            ("<|fim_prefix|>", UNRELATED_CONTROL),
        ],
    );
    tokenizer.special.bos_token = None;
    tokenizer.special.unk_token = None;
    tokenizer.special.pad_token = None;
    tokenizer.special.extra_eos_tokens = vec![TokenId::new(CALL), TokenId::new(RETURN)];
    Arc::new(tokenizer)
}

fn state(tokenizer: &Arc<dyn Tokenizer + Send + Sync>, max_tokens: usize) -> SequenceState {
    let schema = json!({
        "type": "object", "properties": {"temperature": {"type": "integer"}},
        "required": ["temperature"], "additionalProperties": false
    });
    let mut request = policy_request();
    request.stream = true;
    request.sampling_params.max_tokens = max_tokens;
    request.sampling_params.model_output_protocol = ModelOutputProtocol::HarmonyGptOss;
    request.sampling_params.structured_output_start = StructuredOutputStart::HarmonyFinal;
    request.sampling_params.response_format = ResponseFormat::JsonSchema(schema.to_string());
    request.api_request = Some(ApiRequest::Chat(ApiChatRequest {
        messages: vec![],
        tools: vec![ApiTool {
            tool_type: "function".to_string(),
            function: ApiFunction {
                name: "weather".to_string(),
                description: None,
                parameters: Some(json!({
                    "type": "object", "properties": {"city": {"type": "string"}},
                    "required": ["city"], "additionalProperties": false
                })),
                strict: Some(true),
            },
        }],
        tool_choice: None,
        tool_call_protocol: ApiToolCallProtocol::Json,
        legacy_functions: vec![],
        legacy_function_call: None,
        response_format: Some(ApiResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(ApiJsonSchema {
                name: Some("answer".to_string()),
                schema,
                strict: Some(true),
            }),
        }),
        stream_options: None,
    }));
    SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(16)],
        Some(Arc::clone(tokenizer)),
        Some(MODEL_VOCAB),
    )
}

fn sample(
    state: &mut SequenceState,
    tokenizer: &(dyn Tokenizer + Send + Sync),
    expected: u32,
    invalid_payload: u32,
) -> Option<FinishReason> {
    let mut logits = vec![f32::NEG_INFINITY; MODEL_VOCAB];
    logits[expected as usize] = 1.0;
    logits[invalid_payload as usize] = 200.0;
    logits[EOS as usize] = 300.0;
    logits[UNRELATED_CONTROL as usize] = 400.0;
    let actual = state
        .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer))
        .unwrap();
    assert_eq!(actual, TokenId::new(expected));
    assert_eq!(logits[invalid_payload as usize], f32::NEG_INFINITY);
    assert_eq!(logits[EOS as usize], f32::NEG_INFINITY);
    assert_eq!(logits[UNRELATED_CONTROL as usize], f32::NEG_INFINITY);
    state.stop_reason(Some(tokenizer))
}

fn assert_complete_output(output: &str, is_call: bool) {
    let parsed = ferrum_types::parse_harmony_response(output).unwrap();
    if is_call {
        let call = parsed
            .tool_call
            .expect("native recipient must produce a tool call");
        assert_eq!(call.name, "weather");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&call.arguments_json).unwrap(),
            json!({"city": "Paris"})
        );
        assert!(parsed.content.is_empty());
    } else {
        assert!(parsed.tool_call.is_none());
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&parsed.content).unwrap(),
            json!({"temperature": 21})
        );
    }
}

#[test]
fn strict_auto_harmony_native_terminals_finish_and_remain_in_the_stream() {
    let tokenizer = tokenizer();
    for (channel, payload, terminal, invalid, is_call) in [
        (COMMENTARY, ARGUMENTS, CALL, INVALID_ARGUMENTS, true),
        (FINAL_CHANNEL, FINAL, RETURN, INVALID_FINAL, false),
    ] {
        let mut state = state(&tokenizer, 64);
        let mut streamed = String::new();
        for token in [CHANNEL, channel, MESSAGE, payload, terminal] {
            let stop = sample(&mut state, tokenizer.as_ref(), token, invalid);
            assert_eq!(stop, (token == terminal).then_some(FinishReason::EOS));
            if state.should_stream_generated_token(
                Some(tokenizer.as_ref()),
                TokenId::new(token),
                stop,
            ) {
                streamed.push_str(tokenizer.token_text(TokenId::new(token)).unwrap());
            }
        }
        let output = tokenizer.decode(&state.generated_tokens, true).unwrap();
        assert_eq!(
            streamed, output,
            "the stream must retain the protocol terminal"
        );
        assert!(state
            .structured_output_terminal_error(FinishReason::EOS)
            .is_none());
        assert_complete_output(&output, is_call);
    }
}

#[test]
fn strict_auto_harmony_ordinary_token_can_finish_payload_and_protocol_together() {
    let tokenizer = tokenizer();
    for (channel, combined, invalid, is_call) in [
        (COMMENTARY, MERGED_CALL, INVALID_ARGUMENTS, true),
        (FINAL_CHANNEL, MERGED_RETURN, INVALID_FINAL, false),
    ] {
        let mut state = state(&tokenizer, 64);
        assert!(
            !state.stop_token_ids.contains(&combined),
            "the regression requires an ordinary token"
        );
        let mut streamed = String::new();
        for token in [CHANNEL, channel, MESSAGE, combined] {
            let stop = sample(&mut state, tokenizer.as_ref(), token, invalid);
            assert_eq!(stop, (token == combined).then_some(FinishReason::EOS));
            if state.should_stream_generated_token(
                Some(tokenizer.as_ref()),
                TokenId::new(token),
                stop,
            ) {
                streamed.push_str(tokenizer.token_text(TokenId::new(token)).unwrap());
            }
        }
        let output = tokenizer.decode(&state.generated_tokens, true).unwrap();
        assert_eq!(
            streamed, output,
            "stopping must not drop the combined payload token"
        );
        assert!(state
            .structured_output_terminal_error(FinishReason::EOS)
            .is_none());
        assert_complete_output(&output, is_call);
    }
}

#[test]
fn strict_auto_harmony_incomplete_budget_is_an_error_for_either_branch() {
    let tokenizer = tokenizer();
    for (channel, incomplete, invalid) in [
        (COMMENTARY, INCOMPLETE_ARGUMENTS, INVALID_ARGUMENTS),
        (FINAL_CHANNEL, INCOMPLETE_FINAL, INVALID_FINAL),
    ] {
        let mut state = state(&tokenizer, 6);
        for token in [26, 16, CHANNEL, channel, MESSAGE, incomplete] {
            let stop = sample(&mut state, tokenizer.as_ref(), token, invalid);
            assert_eq!(stop, (token == incomplete).then_some(FinishReason::Length));
        }
        assert!(state
            .structured_output_terminal_error(FinishReason::Length)
            .is_some());
        let output = tokenizer.decode(&state.generated_tokens, true).unwrap();
        assert!(ferrum_types::parse_harmony_response(&output).is_err());
    }
}

#[test]
fn strict_auto_harmony_cannot_sample_premature_eos_as_a_successful_handoff() {
    let tokenizer = tokenizer();
    let mut state = state(&tokenizer, 64);
    for token in [CHANNEL, COMMENTARY, MESSAGE, INCOMPLETE_ARGUMENTS] {
        assert_eq!(
            sample(&mut state, tokenizer.as_ref(), token, INVALID_ARGUMENTS),
            None
        );
    }
    let prefix = state.generated_tokens.clone();
    let mut logits = vec![f32::NEG_INFINITY; MODEL_VOCAB];
    logits[EOS as usize] = 400.0;
    logits[CALL as usize] = 300.0;
    logits[RETURN as usize] = 200.0;
    assert!(state
        .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
        .is_err());
    assert_eq!(
        state.generated_tokens, prefix,
        "rejected EOS must not enter generated output"
    );
    assert!(state
        .structured_output_terminal_error(FinishReason::EOS)
        .is_some());
}
