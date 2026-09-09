use super::*;
use crate::structured_output::tests::ByteTokenizer;
use ferrum_interfaces::tokenizer::{ChatMessage, TokenizerInfo};
use ferrum_types::{
    ApiFunction, ApiJsonSchema, ApiResponseFormat, ApiTool, ApiToolCallProtocol, SpecialTokens,
};

const EOS: u32 = 256;

struct PacketTokenizer {
    base: ByteTokenizer,
    added: Vec<String>,
    special: SpecialTokens,
}

impl PacketTokenizer {
    fn new(packets: &[&str]) -> Self {
        let base = ByteTokenizer::new();
        let mut special = base.special_tokens().clone();
        let added: Vec<String> = [
            "<tool_call>",
            "</tool_call>",
            "<|channel|>",
            "<|message|>",
            "<|start|>",
            "<|end|>",
            "<|constrain|>",
            "<|call|>",
            "<|return|>",
            "<|channel>",
            "<channel|>",
        ]
        .into_iter()
        .chain(packets.iter().copied())
        .map(str::to_string)
        .collect();
        // These are deliberately rewritten by the factory's special-token
        // trie path. Merged ordinary tokens retain literal marker bytes.
        special.sep_token = Some(TokenId::new(257));
        special.cls_token = Some(TokenId::new(258));
        special.extra_eos_tokens = vec![TokenId::new(264), TokenId::new(265)];
        Self {
            base,
            added,
            special,
        }
    }

    fn id(&self, text: &str) -> u32 {
        self.token_id(text).expect("fixture token").get()
    }
    fn terminals(&self) -> HashSet<u32> {
        HashSet::from([EOS, self.id("<|call|>"), self.id("<|return|>")])
    }
}

impl Tokenizer for PacketTokenizer {
    fn encode(&self, text: &str, _special: bool) -> Result<Vec<TokenId>> {
        let mut remaining = text.as_bytes();
        let mut tokens = Vec::new();
        while !remaining.is_empty() {
            if let Some((index, token)) = self
                .added
                .iter()
                .enumerate()
                .filter(|(_, token)| remaining.starts_with(token.as_bytes()))
                .max_by_key(|(_, token)| token.len())
            {
                tokens.push(TokenId::new(257 + index as u32));
                remaining = &remaining[token.len()..];
            } else {
                tokens.push(TokenId::new(remaining[0] as u32));
                remaining = &remaining[1..];
            }
        }
        Ok(tokens)
    }
    fn decode(&self, tokens: &[TokenId], _skip: bool) -> Result<String> {
        Ok(String::from_utf8_lossy(
            &tokens
                .iter()
                .flat_map(|token| self.token_bytes(*token).unwrap_or_default())
                .collect::<Vec<_>>(),
        )
        .into_owned())
    }
    fn decode_incremental(&self, _prev: &[TokenId], next: TokenId) -> Result<String> {
        self.decode(&[next], false)
    }
    fn vocab_size(&self) -> usize {
        257 + self.added.len()
    }
    fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }
    fn token_id(&self, text: &str) -> Option<TokenId> {
        self.added
            .iter()
            .position(|value| value == text)
            .map(|index| TokenId::new(257 + index as u32))
            .or_else(|| self.base.token_id(text))
    }
    fn token_text(&self, token: TokenId) -> Option<&str> {
        if token.get() >= 257 {
            self.added
                .get(token.get() as usize - 257)
                .map(String::as_str)
        } else {
            self.base.token_text(token)
        }
    }
    fn token_bytes(&self, token: TokenId) -> Option<Vec<u8>> {
        if token.get() < 256 {
            Some(vec![token.get() as u8])
        } else {
            self.token_text(token).map(|text| text.as_bytes().to_vec())
        }
    }
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        self.base.apply_chat_template(messages)
    }
    fn info(&self) -> TokenizerInfo {
        let mut info = self.base.info();
        info.vocab_size = self.vocab_size();
        info.special_tokens = self.special.clone();
        info
    }
}

fn request(protocol: ApiToolCallProtocol) -> ApiChatRequest {
    ApiChatRequest {
        messages: Vec::new(),
        tools: vec![ApiTool {
            tool_type: "function".into(),
            function: ApiFunction {
                name: "weather".into(),
                description: None,
                parameters: Some(
                    json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}),
                ),
                strict: Some(true),
            },
        }],
        tool_choice: None,
        tool_call_protocol: protocol,
        legacy_functions: Vec::new(),
        legacy_function_call: None,
        response_format: Some(ApiResponseFormat {
            format_type: "json_schema".into(),
            json_schema: Some(ApiJsonSchema {
                name: None,
                schema: json!({"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false}),
                strict: Some(true),
            }),
        }),
        stream_options: None,
    }
}

fn processor(
    tokenizer: Arc<PacketTokenizer>,
    protocol: ApiToolCallProtocol,
    output: ModelOutputProtocol,
    start: StructuredOutputStart,
    max_tokens: usize,
) -> StructuredOutputProcessor {
    let terminals = tokenizer.terminals();
    let request = request(protocol);
    let schema = request
        .response_format
        .as_ref()
        .unwrap()
        .json_schema
        .as_ref()
        .unwrap()
        .schema
        .to_string();
    StructuredOutputFactory::new(tokenizer)
        .unwrap()
        .create_processor_with_chat_contract(
            &ResponseFormat::JsonSchema(schema),
            &start,
            max_tokens,
            &terminals,
            &[],
            Some(&request),
            output,
        )
        .unwrap()
        .unwrap()
}

fn append(
    processor: &StructuredOutputProcessor,
    tokenizer: &PacketTokenizer,
    generated: &mut Vec<TokenId>,
    text: &str,
) {
    for token in tokenizer.encode(text, false).unwrap() {
        let mut logits = vec![0.0; tokenizer.vocab_size()];
        processor
            .mask_logits_with_terminals(
                &mut logits,
                generated,
                &tokenizer.terminals(),
                &HashSet::new(),
            )
            .unwrap();
        assert!(
            logits[token.get() as usize].is_finite(),
            "rejected {:?} after {:?}",
            tokenizer.token_text(token),
            tokenizer.decode(generated, false)
        );
        generated.push(token);
    }
}

#[test]
fn explicit_json_call_and_final_schema_are_both_hard_grammar_branches() {
    let tokenizer = Arc::new(PacketTokenizer::new(&[]));
    let processor = processor(
        Arc::clone(&tokenizer),
        ApiToolCallProtocol::Json,
        ModelOutputProtocol::Text,
        StructuredOutputStart::Immediate,
        512,
    );
    let mut generated = Vec::new();
    let mut logits = vec![0.0; tokenizer.vocab_size()];
    processor.mask_logits(&mut logits, &generated).unwrap();
    assert!(logits[b'{' as usize].is_finite());
    assert!(logits[tokenizer.id("<tool_call>") as usize].is_finite());
    assert!(!logits[b'X' as usize].is_finite());
    append(&processor, &tokenizer, &mut generated, "<tool_call>{\n\"arguments\" : {\"city\":\"Paris\"},\n\"name\" : \"weather\"\n}</tool_call>");
    assert!(processor.is_accepting(&generated).unwrap());
    processor.reset().unwrap();
    generated.clear();
    append(&processor, &tokenizer, &mut generated, "{\"ok\":");
    logits.fill(0.0);
    processor.mask_logits(&mut logits, &generated).unwrap();
    assert!(logits[b't' as usize].is_finite());
    assert!(!logits[b'f' as usize].is_finite());
    append(&processor, &tokenizer, &mut generated, "true}");
    assert!(processor.is_accepting(&generated).unwrap());
    logits.fill(0.0);
    processor.mask_logits(&mut logits, &generated).unwrap();
    assert!(!logits[tokenizer.id("<tool_call>") as usize].is_finite());
}

#[test]
fn merged_marker_and_payload_tokens_cannot_hide_invalid_schema_bytes() {
    let valid = "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>";
    let invalid = "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":42}}</tool_call>";
    let tokenizer = Arc::new(PacketTokenizer::new(&[
        valid,
        invalid,
        "{\"ok\":true}",
        "{\"ok\":false}",
    ]));
    let processor = processor(
        Arc::clone(&tokenizer),
        ApiToolCallProtocol::Json,
        ModelOutputProtocol::Text,
        StructuredOutputStart::Immediate,
        64,
    );
    let mut logits = vec![0.0; tokenizer.vocab_size()];
    processor.mask_logits(&mut logits, &[]).unwrap();
    assert!(logits[tokenizer.id(valid) as usize].is_finite());
    assert!(!logits[tokenizer.id(invalid) as usize].is_finite());
    assert!(logits[tokenizer.id("{\"ok\":true}") as usize].is_finite());
    assert!(!logits[tokenizer.id("{\"ok\":false}") as usize].is_finite());
    assert!(processor
        .is_accepting(&[TokenId::new(tokenizer.id(valid))])
        .unwrap());
}

#[test]
fn xml_envelope_keeps_raw_parameters_and_allows_multiple_complete_calls() {
    let tokenizer = Arc::new(PacketTokenizer::new(&[]));
    let processor = processor(
        Arc::clone(&tokenizer),
        ApiToolCallProtocol::FunctionParameterXml,
        ModelOutputProtocol::Text,
        StructuredOutputStart::Immediate,
        512,
    );
    let mut generated = Vec::new();
    for city in ["Paris", "Oslo"] {
        append(&processor, &tokenizer, &mut generated, &format!("<tool_call>\n<function=weather>\n<parameter=city>\n{city}\n</parameter>\n</function>\n</tool_call>"));
    }
    assert!(processor.is_accepting(&generated).unwrap());
    processor.reset().unwrap();
    generated.clear();
    append(
        &processor,
        &tokenizer,
        &mut generated,
        "<tool_call><function=weather><parameter=city>Paris",
    );
    assert!(!processor.is_accepting(&generated).unwrap());
}

#[test]
fn forced_native_calls_mask_bare_arguments_and_unselected_names() {
    use ferrum_types::{ApiToolChoice, ApiToolChoiceFunction, StructuredOutputBranch};
    let tokenizer = Arc::new(PacketTokenizer::new(&[]));
    let mut chat = request(ApiToolCallProtocol::FunctionParameterXml);
    let mut other = chat.tools[0].clone();
    other.function.name = "clock".into();
    chat.tools.push(other);
    chat.response_format = None;
    for choice in [
        ApiToolChoice::Function {
            tool_type: "function".into(),
            function: ApiToolChoiceFunction {
                name: "weather".into(),
            },
        },
        ApiToolChoice::Mode("required".into()),
    ] {
        let named = matches!(&choice, ApiToolChoice::Function { .. });
        chat.tool_choice = Some(choice);
        let processor = StructuredOutputFactory::new(tokenizer.clone())
            .unwrap()
            .create_processor_with_chat_contract(
                &ResponseFormat::Text,
                &StructuredOutputStart::Immediate,
                512,
                &tokenizer.terminals(),
                &[],
                Some(&chat),
                ModelOutputProtocol::Text,
            )
            .unwrap()
            .unwrap();
        let mut generated = Vec::new();
        let mut logits = vec![0.0; tokenizer.vocab_size()];
        processor
            .mask_logits_with_terminals(
                &mut logits,
                &generated,
                &tokenizer.terminals(),
                &HashSet::new(),
            )
            .unwrap();
        assert!(!logits[b'{' as usize].is_finite());
        assert!(!logits[EOS as usize].is_finite());
        append(
            &processor,
            &tokenizer,
            &mut generated,
            "<tool_call><function=",
        );
        logits.fill(0.0);
        processor
            .mask_logits_with_terminals(
                &mut logits,
                &generated,
                &tokenizer.terminals(),
                &HashSet::new(),
            )
            .unwrap();
        assert!(logits[b'w' as usize].is_finite());
        assert_eq!(logits[b'c' as usize].is_finite(), !named);
        assert!(!logits[b'z' as usize].is_finite());
        append(
            &processor,
            &tokenizer,
            &mut generated,
            "weather><parameter=city>Paris</parameter></function>",
        );
        assert!(!processor.is_accepting(&generated).unwrap());
        append(&processor, &tokenizer, &mut generated, "</tool_call>");
        assert!(processor.is_accepting(&generated).unwrap());
        let (branch, payload) = processor
            .classified_result_with_terminals(&generated, &tokenizer.terminals())
            .unwrap()
            .unwrap();
        assert_eq!(branch, StructuredOutputBranch::ToolCall);
        assert_eq!(
            payload,
            "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>"
        );
    }
}

#[test]
fn forced_native_call_preserves_merged_reasoning_boundary_and_rejects_missing_selection() {
    use ferrum_types::{ApiToolChoice, ApiToolChoiceFunction, StructuredOutputBranch};
    let tokenizer = Arc::new(PacketTokenizer::new(&[
        "</think><tool_call><function=weather>",
    ]));
    let mut chat = request(ApiToolCallProtocol::FunctionParameterXml);
    chat.response_format = None;
    chat.tool_choice = Some(ApiToolChoice::Function {
        tool_type: "function".into(),
        function: ApiToolChoiceFunction {
            name: "weather".into(),
        },
    });
    let factory = StructuredOutputFactory::new(tokenizer.clone()).unwrap();
    let processor = factory
        .create_processor_with_chat_contract(
            &ResponseFormat::Text,
            &StructuredOutputStart::AfterDelimiter("</think>".into()),
            512,
            &tokenizer.terminals(),
            &[],
            Some(&chat),
            ModelOutputProtocol::Text,
        )
        .unwrap()
        .unwrap();
    let mut generated = Vec::new();
    append(&processor,&tokenizer,&mut generated,"Choose the weather tool.</think><tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>");
    let (branch, payload) = processor
        .classified_result_with_terminals(&generated, &tokenizer.terminals())
        .unwrap()
        .unwrap();
    assert_eq!(branch, StructuredOutputBranch::ToolCall);
    assert!(payload.starts_with("<tool_call>"));
    assert!(!payload.contains("Choose"));
    chat.tool_choice = Some(ApiToolChoice::Function {
        tool_type: "function".into(),
        function: ApiToolChoiceFunction {
            name: "undeclared".into(),
        },
    });
    assert!(factory
        .create_processor_with_chat_contract(
            &ResponseFormat::Text,
            &StructuredOutputStart::Immediate,
            512,
            &tokenizer.terminals(),
            &[],
            Some(&chat),
            ModelOutputProtocol::Text
        )
        .is_err());
}

#[test]
fn reasoning_budget_forces_closure_without_weakening_final_schema() {
    let tokenizer = Arc::new(PacketTokenizer::new(&[
        "</think>{\"ok\":true}",
        "</think>{\"ok\":false}",
        "</thi",
    ]));
    let processor = processor(
        Arc::clone(&tokenizer),
        ApiToolCallProtocol::Json,
        ModelOutputProtocol::Text,
        StructuredOutputStart::AfterDelimiter("</think>".into()),
        64,
    );
    let mut generated = Vec::new();
    let budget = processor
        .progress_with_terminals(&generated, &tokenizer.terminals())
        .unwrap()
        .budget
        .unwrap();
    append(
        &processor,
        &tokenizer,
        &mut generated,
        &"a".repeat(budget.reasoning_token_limit),
    );
    let mut logits = vec![0.0; tokenizer.vocab_size()];
    let outcome = processor
        .mask_logits_with_terminals(
            &mut logits,
            &generated,
            &tokenizer.terminals(),
            &HashSet::new(),
        )
        .unwrap();
    assert_eq!(outcome.phase, StructuredOutputPhase::ForcingDelimiter);
    assert!(!logits[b'b' as usize].is_finite());
    assert!(logits[tokenizer.id("</think>{\"ok\":true}") as usize].is_finite());
    assert!(!logits[tokenizer.id("</think>{\"ok\":false}") as usize].is_finite());
    append(
        &processor,
        &tokenizer,
        &mut generated,
        "</think>{\"ok\":true}",
    );
    assert!(processor.is_accepting(&generated).unwrap());
    processor.reset().unwrap();
    generated.clear();
    append(&processor, &tokenizer, &mut generated, "a</thi");
    assert_eq!(
        processor
            .progress_with_terminals(&generated, &tokenizer.terminals())
            .unwrap()
            .delimiter_prefix_token_count,
        1,
        "a five-byte delimiter prefix is one sampled token"
    );
}

#[test]
fn gemma_thought_can_transition_to_tool_or_final_in_one_token() {
    for result in [
        "{\"ok\":true}",
        "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>",
    ] {
        let packet = format!("<channel|>{result}");
        let tokenizer = Arc::new(PacketTokenizer::new(&[&packet]));
        let processor = processor(
            Arc::clone(&tokenizer),
            ApiToolCallProtocol::Json,
            ModelOutputProtocol::GemmaThought,
            StructuredOutputStart::AfterReasoningEnvelope {
                opening: "<|channel>thought\n".into(),
                closing: "<channel|>".into(),
                allow_reasoning: true,
            },
            512,
        );
        let mut generated = Vec::new();
        append(
            &processor,
            &tokenizer,
            &mut generated,
            "<|channel>thought\nNeed data.",
        );
        append(&processor, &tokenizer, &mut generated, &packet);
        assert!(processor.is_accepting(&generated).unwrap());
    }
}

#[test]
fn harmony_requires_the_selected_native_terminal_and_accepts_merged_handoff() {
    let call_body = "<|channel|>commentary to=functions.weather<|constrain|>json<|message|>{\"city\":\"Paris\"}";
    let merged = format!("{call_body}<|call|>");
    let tokenizer = Arc::new(PacketTokenizer::new(&[&merged]));
    let processor = processor(
        Arc::clone(&tokenizer),
        ApiToolCallProtocol::Json,
        ModelOutputProtocol::HarmonyGptOss,
        StructuredOutputStart::HarmonyFinal,
        512,
    );
    let mut generated = Vec::new();
    append(&processor, &tokenizer, &mut generated, call_body);
    assert!(!processor.is_accepting(&generated).unwrap());
    let mut logits = vec![0.0; tokenizer.vocab_size()];
    let outcome = processor
        .mask_logits_with_terminals(
            &mut logits,
            &generated,
            &tokenizer.terminals(),
            &HashSet::new(),
        )
        .unwrap();
    assert_eq!(
        outcome.grammar_owned_terminal_token_id,
        Some(tokenizer.id("<|call|>"))
    );
    assert!(!logits[tokenizer.id("<|return|>") as usize].is_finite());
    append(&processor, &tokenizer, &mut generated, "<|call|>");
    assert!(processor
        .is_protocol_complete_with_terminals(&generated, &tokenizer.terminals())
        .unwrap());
    processor.reset().unwrap();
    let merged_tokens = vec![TokenId::new(tokenizer.id(&merged))];
    assert!(processor
        .is_protocol_complete_with_terminals(&merged_tokens, &tokenizer.terminals())
        .unwrap());
    processor.reset().unwrap();
    generated.clear();
    append(&processor, &tokenizer, &mut generated, "<|channel|>analysis<|message|>Need data.<|end|><|start|>assistant to=functions.weather<|channel|>commentary<|message|>{\"city\":\"Paris\"}<|call|>");
    assert!(processor
        .is_protocol_complete_with_terminals(&generated, &tokenizer.terminals())
        .unwrap());
}

#[test]
fn hidden_reasoning_does_not_grant_unrelated_control_tokens() {
    for (output, start, prefix, closing) in [
        (
            ModelOutputProtocol::HarmonyGptOss,
            StructuredOutputStart::HarmonyFinal,
            "<|channel|>analysis<|message|>thinking",
            "<|end|>",
        ),
        (
            ModelOutputProtocol::GemmaThought,
            StructuredOutputStart::AfterReasoningEnvelope {
                opening: "<|channel>thought\n".into(),
                closing: "<channel|>".into(),
                allow_reasoning: true,
            },
            "<|channel>thought\nthinking",
            "<channel|>",
        ),
    ] {
        let tokenizer = Arc::new(PacketTokenizer::new(&["<|user|>", "<fim_prefix>"]));
        let processor = processor(
            Arc::clone(&tokenizer),
            ApiToolCallProtocol::Json,
            output,
            start,
            512,
        );
        let mut generated = Vec::new();
        append(&processor, &tokenizer, &mut generated, prefix);
        let unrelated = HashSet::from([tokenizer.id("<|user|>"), tokenizer.id("<fim_prefix>")]);
        let mut controls = unrelated.clone();
        controls.insert(tokenizer.id(closing));
        let mut logits = vec![0.0; tokenizer.vocab_size()];
        for id in &unrelated {
            logits[*id as usize] = 100.0;
        }
        processor
            .mask_logits_with_terminals(&mut logits, &generated, &tokenizer.terminals(), &controls)
            .unwrap();
        for id in unrelated {
            assert!(!logits[id as usize].is_finite());
        }
        assert!(logits[tokenizer.id(closing) as usize].is_finite());
    }
}

#[test]
fn composed_root_boundaries_exclude_external_eos_and_states_are_independent() {
    let tokenizer = Arc::new(PacketTokenizer::new(&[]));
    let first = processor(
        Arc::clone(&tokenizer),
        ApiToolCallProtocol::Json,
        ModelOutputProtocol::Text,
        StructuredOutputStart::Immediate,
        256,
    );
    let second = processor(
        Arc::clone(&tokenizer),
        ApiToolCallProtocol::Json,
        ModelOutputProtocol::Text,
        StructuredOutputStart::Immediate,
        256,
    );
    let text = "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>";
    let mut generated = Vec::new();
    append(&first, &tokenizer, &mut generated, text);
    generated.push(TokenId::new(EOS));
    assert_eq!(
        first
            .complete_root_text_len_with_terminals(&generated, &tokenizer.terminals())
            .unwrap(),
        Some(text.len())
    );
    assert!(!second.is_accepting(&[]).unwrap());
    let mut final_tokens = Vec::new();
    append(&second, &tokenizer, &mut final_tokens, "{\"ok\":true}");
    assert!(second.is_accepting(&final_tokens).unwrap());
    first.reset().unwrap();
    assert!(!first.is_accepting(&[]).unwrap());
    assert!(second.is_accepting(&final_tokens).unwrap());
}

fn configured_processor(
    factory: &StructuredOutputFactory,
    tokenizer: &PacketTokenizer,
    request: &ApiChatRequest,
    start: &StructuredOutputStart,
    output: ModelOutputProtocol,
) -> StructuredOutputProcessor {
    let schema = request
        .response_format
        .as_ref()
        .unwrap()
        .json_schema
        .as_ref()
        .unwrap()
        .schema
        .to_string();
    factory
        .create_processor_with_chat_contract(
            &ResponseFormat::JsonSchema(schema),
            start,
            512,
            &tokenizer.terminals(),
            &[],
            Some(request),
            output,
        )
        .unwrap()
        .unwrap()
}

#[test]
fn named_bare_calls_preserve_both_argument_field_names() {
    let tokenizer = Arc::new(PacketTokenizer::new(&[]));
    let processor = processor(
        Arc::clone(&tokenizer),
        ApiToolCallProtocol::Json,
        ModelOutputProtocol::Text,
        StructuredOutputStart::Immediate,
        512,
    );
    for payload in [
        "{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}",
        "{\"parameters\":{\"city\":\"Paris\"},\"name\":\"weather\"}",
    ] {
        processor.reset().unwrap();
        let mut generated = Vec::new();
        append(&processor, &tokenizer, &mut generated, payload);
        assert_eq!(
            processor
                .classified_result_with_terminals(&generated, &tokenizer.terminals())
                .unwrap(),
            Some((
                ferrum_types::StructuredOutputBranch::ToolCall,
                payload.to_string()
            ))
        );
    }
}

#[test]
fn final_schema_precedence_is_independent_of_call_field_order_and_spacing() {
    let tokenizer = Arc::new(PacketTokenizer::new(&[]));
    let factory = StructuredOutputFactory::new(tokenizer.clone()).unwrap();
    let mut request = request(ApiToolCallProtocol::Json);
    request
        .response_format
        .as_mut()
        .unwrap()
        .json_schema
        .as_mut()
        .unwrap()
        .schema = json!({
        "type":"object", "properties": {
            "arguments": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false},
            "name": {"const":"weather"}
        }, "required":["arguments","name"],"additionalProperties":false
    });
    let processor = configured_processor(
        &factory,
        &tokenizer,
        &request,
        &StructuredOutputStart::Immediate,
        ModelOutputProtocol::Text,
    );
    for payload in [
        "{\"arguments\":{\"city\":\"Paris\"},\"name\":\"weather\"}",
        "{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}",
        "{\n\"name\" : \"weather\",\n\"arguments\" : {\"city\":\"Paris\"}\n}",
    ] {
        processor.reset().unwrap();
        let mut generated = Vec::new();
        append(&processor, &tokenizer, &mut generated, payload);
        assert_eq!(
            processor
                .classified_result_with_terminals(&generated, &tokenizer.terminals())
                .unwrap(),
            Some((
                ferrum_types::StructuredOutputBranch::Final,
                payload.to_string()
            )),
            "payload: {payload}"
        );
    }
}

#[test]
fn gemma_classification_keeps_payload_bytes_in_the_closing_token() {
    let payload = "{\"ok\":true}";
    let packet = format!("<channel|>{payload}");
    let tokenizer = Arc::new(PacketTokenizer::new(&[&packet]));
    let processor = processor(
        Arc::clone(&tokenizer),
        ApiToolCallProtocol::Json,
        ModelOutputProtocol::GemmaThought,
        StructuredOutputStart::AfterReasoningEnvelope {
            opening: "<|channel>thought\n".into(),
            closing: "<channel|>".into(),
            allow_reasoning: true,
        },
        512,
    );
    let mut generated = Vec::new();
    append(
        &processor,
        &tokenizer,
        &mut generated,
        "<|channel>thought\nready",
    );
    append(&processor, &tokenizer, &mut generated, &packet);
    assert_eq!(
        processor
            .classified_result_with_terminals(&generated, &tokenizer.terminals())
            .unwrap(),
        Some((
            ferrum_types::StructuredOutputBranch::Final,
            payload.to_string()
        ))
    );
}

#[test]
fn local_schema_refs_remain_independent_and_tool_names_select_their_schema() {
    let valid = "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>";
    let wrong_name_schema =
        "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Tokyo\"}}</tool_call>";
    let other_tool =
        "<tool_call>{\"name\":\"other\",\"arguments\":{\"city\":\"Tokyo\"}}</tool_call>";
    let tokenizer = Arc::new(PacketTokenizer::new(&[
        valid,
        wrong_name_schema,
        other_tool,
        "{\"ok\":true}",
        "{\"ok\":\"Paris\"}",
    ]));
    let factory = StructuredOutputFactory::new(tokenizer.clone()).unwrap();
    let mut request = request(ApiToolCallProtocol::Json);
    request
        .response_format
        .as_mut()
        .unwrap()
        .json_schema
        .as_mut()
        .unwrap()
        .schema = json!({
        "$defs":{"item":{"const":true}},"type":"object","properties":{"ok":{"$ref":"#/$defs/item"}},"required":["ok"],"additionalProperties":false
    });
    request.tools[0].function.parameters = Some(json!({
        "$defs":{"item":{"const":"Paris"}},"type":"object","properties":{"city":{"$ref":"#/$defs/item"}},"required":["city"],"additionalProperties":false
    }));
    let mut other = request.tools[0].clone();
    other.function.name = "other".into();
    other.function.parameters.as_mut().unwrap()["$defs"]["item"]["const"] = json!("Tokyo");
    request.tools.push(other);
    let first = configured_processor(
        &factory,
        &tokenizer,
        &request,
        &StructuredOutputStart::Immediate,
        ModelOutputProtocol::Text,
    );
    let second = configured_processor(
        &factory,
        &tokenizer,
        &request,
        &StructuredOutputStart::Immediate,
        ModelOutputProtocol::Text,
    );
    let mut logits = vec![0.0; tokenizer.vocab_size()];
    first.mask_logits(&mut logits, &[]).unwrap();
    for text in [valid, other_tool, "{\"ok\":true}"] {
        assert!(logits[tokenizer.id(text) as usize].is_finite());
    }
    for text in [wrong_name_schema, "{\"ok\":\"Paris\"}"] {
        assert!(!logits[tokenizer.id(text) as usize].is_finite());
    }
    assert!(first
        .is_accepting(&[TokenId::new(tokenizer.id(valid))])
        .unwrap());
    assert!(!second.is_accepting(&[]).unwrap());
    let mut generated = Vec::new();
    append(&second, &tokenizer, &mut generated, "{\"ok\":true}");
    first.reset().unwrap();
    assert!(second.is_accepting(&generated).unwrap());
}

#[test]
fn fragmented_utf8_bytes_remain_valid_inside_final_and_arguments() {
    let tokenizer = Arc::new(PacketTokenizer::new(&[]));
    let factory = StructuredOutputFactory::new(tokenizer.clone()).unwrap();
    let mut request = request(ApiToolCallProtocol::Json);
    request
        .response_format
        .as_mut()
        .unwrap()
        .json_schema
        .as_mut()
        .unwrap()
        .schema = request.tools[0].function.parameters.clone().unwrap();
    let processor = configured_processor(
        &factory,
        &tokenizer,
        &request,
        &StructuredOutputStart::Immediate,
        ModelOutputProtocol::Text,
    );
    for payload in [
        "{\"city\":\"北京\"}",
        "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"北京\"}}</tool_call>",
    ] {
        processor.reset().unwrap();
        let mut generated = Vec::new();
        append(&processor, &tokenizer, &mut generated, payload);
        assert!(processor.is_accepting(&generated).unwrap());
        let (_, preserved) = processor
            .classified_result_with_terminals(&generated, &tokenizer.terminals())
            .unwrap()
            .unwrap();
        assert_eq!(preserved, payload);
    }
}

#[test]
fn final_json_strings_preserve_reasoning_tag_text_in_merged_and_split_tokens() {
    for city in ["<think>Paris</think>", "<think>Paris", "Paris</think>"] {
        let payload = json!({"city":city,"temperature":21}).to_string();
        for merged in [false, true] {
            let packets = [&payload[..]];
            let tokenizer = Arc::new(PacketTokenizer::new(if merged { &packets } else { &[] }));
            let factory = StructuredOutputFactory::new(tokenizer.clone()).unwrap();
            let mut request = request(ApiToolCallProtocol::Json);
            request
                .response_format
                .as_mut()
                .unwrap()
                .json_schema
                .as_mut()
                .unwrap()
                .schema = json!({
                "type":"object", "properties": {
                    "city":{"type":"string","const":city},
                    "temperature":{"type":"integer"}
                }, "required":["city","temperature"], "additionalProperties":false
            });
            let processor = configured_processor(
                &factory,
                &tokenizer,
                &request,
                &StructuredOutputStart::Immediate,
                ModelOutputProtocol::Text,
            );
            let mut generated = Vec::new();
            append(&processor, &tokenizer, &mut generated, &payload);
            assert_eq!(
                processor
                    .classified_result_with_terminals(&generated, &tokenizer.terminals())
                    .unwrap(),
                Some((ferrum_types::StructuredOutputBranch::Final, payload.clone()))
            );
        }
    }
}

#[test]
fn pretty_final_json_remains_valid_one_token_at_a_time_with_auto_tools() {
    let tokenizer = Arc::new(PacketTokenizer::new(&[]));
    for protocol in [
        ApiToolCallProtocol::Json,
        ApiToolCallProtocol::FunctionParameterXml,
    ] {
        for after_reasoning in [false, true] {
            let processor = processor(
                Arc::clone(&tokenizer),
                protocol,
                ModelOutputProtocol::Text,
                if after_reasoning {
                    StructuredOutputStart::AfterDelimiter("</think>".into())
                } else {
                    StructuredOutputStart::Immediate
                },
                512,
            );
            let mut generated = Vec::new();
            if after_reasoning {
                append(
                    &processor,
                    &tokenizer,
                    &mut generated,
                    "The tool result is known.</think>",
                );
            }
            // Each byte crosses the live grammar; newlines and indentation
            // outside strings are valid JSON, independent of tool availability.
            append(
                &processor,
                &tokenizer,
                &mut generated,
                "{\n  \"ok\": true\n}",
            );
            assert!(processor.is_accepting(&generated).unwrap());
            let (branch, text) = processor
                .classified_result_with_terminals(&generated, &tokenizer.terminals())
                .unwrap()
                .unwrap();
            assert_eq!(branch, ferrum_types::StructuredOutputBranch::Final);
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&text).unwrap(),
                json!({"ok": true})
            );
        }
    }
}

#[test]
fn pretty_final_opener_outscores_a_tool_when_both_branches_are_valid() {
    let invalid_final = "{\n  \"ok\": false\n}";
    for opener in ["{\n", " {\n"] {
        let tokenizer = Arc::new(PacketTokenizer::new(&[opener, invalid_final]));
        for protocol in [
            ApiToolCallProtocol::Json,
            ApiToolCallProtocol::FunctionParameterXml,
        ] {
            for after_reasoning in [false, true] {
                let processor = processor(
                    Arc::clone(&tokenizer),
                    protocol,
                    ModelOutputProtocol::Text,
                    if after_reasoning {
                        StructuredOutputStart::AfterDelimiter("</think>".into())
                    } else {
                        StructuredOutputStart::Immediate
                    },
                    512,
                );
                let mut generated = Vec::new();
                if after_reasoning {
                    append(
                        &processor,
                        &tokenizer,
                        &mut generated,
                        "Now return the final JSON.</think>",
                    );
                }
                let mut logits = vec![f32::NEG_INFINITY; tokenizer.vocab_size()];
                logits[tokenizer.id(invalid_final) as usize] = 100.0;
                logits[tokenizer.id(opener) as usize] = 50.0;
                logits[tokenizer.id("<tool_call>") as usize] = 10.0;
                processor
                    .mask_logits_with_terminals(
                        &mut logits,
                        &generated,
                        &tokenizer.terminals(),
                        &HashSet::new(),
                    )
                    .unwrap();
                assert!(
                    !logits[tokenizer.id(invalid_final) as usize].is_finite(),
                    "pretty output must still satisfy its schema"
                );
                assert!(
                    logits[tokenizer.id("<tool_call>") as usize].is_finite(),
                    "automatic tool branch must remain available"
                );
                let (selected, _) = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| left.total_cmp(right))
                    .unwrap();
                assert_eq!(selected as u32, tokenizer.id(opener),
                    "valid pretty JSON token {opener:?} was masked in favor of a lower-scoring tool, protocol={protocol:?}, after_reasoning={after_reasoning}");
                generated.push(TokenId::new(selected as u32));
                append(&processor, &tokenizer, &mut generated, "  \"ok\": true\n}");
                let (branch, text) = processor
                    .classified_result_with_terminals(&generated, &tokenizer.terminals())
                    .unwrap()
                    .unwrap();
                assert_eq!(branch, ferrum_types::StructuredOutputBranch::Final);
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(&text).unwrap(),
                    json!({"ok": true})
                );
            }
        }
    }
}
