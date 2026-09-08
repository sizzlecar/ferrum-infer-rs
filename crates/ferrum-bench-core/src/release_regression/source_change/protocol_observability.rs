//! Content proof for protocol changes that preserve observability code.
//! This does not prove the protocol or waive its model/runtime obligations.
use quote::ToTokens;
use syn::{visit_mut::VisitMut, Block, Expr, ImplItem, Item, Stmt};

#[derive(Clone, Copy, Debug)]
pub enum ProtocolObservabilityKind {
    Requests,
    Sequence,
    Completion,
}

fn tokens(value: &impl ToTokens) -> String {
    value.to_token_stream().to_string()
}

struct Normalize;
impl VisitMut for Normalize {
    fn visit_attributes_mut(&mut self, attrs: &mut Vec<syn::Attribute>) {
        attrs.retain(|attr| !attr.path().is_ident("doc"));
        for attr in attrs {
            self.visit_attribute_mut(attr);
        }
    }
    fn visit_signature_mut(&mut self, signature: &mut syn::Signature) {
        signature.inputs.pop_punct();
        syn::visit_mut::visit_signature_mut(self, signature);
    }
    fn visit_expr_call_mut(&mut self, expression: &mut syn::ExprCall) {
        expression.args.pop_punct();
        syn::visit_mut::visit_expr_call_mut(self, expression);
    }
    fn visit_expr_method_call_mut(&mut self, expression: &mut syn::ExprMethodCall) {
        expression.args.pop_punct();
        syn::visit_mut::visit_expr_method_call_mut(self, expression);
    }
    fn visit_expr_match_mut(&mut self, expression: &mut syn::ExprMatch) {
        for arm in &mut expression.arms {
            arm.comma = Some(Default::default());
        }
        syn::visit_mut::visit_expr_match_mut(self, expression);
    }
}

fn parse(source: &str) -> Result<syn::File, String> {
    let mut file = syn::parse_file(source).map_err(|error| error.to_string())?;
    Normalize.visit_file_mut(&mut file);
    Ok(file)
}
fn statements(source: &str) -> Vec<Stmt> {
    let mut block: Block =
        syn::parse_str(&format!("{{{source}}}")).expect("reviewed protocol statement contract");
    Normalize.visit_block_mut(&mut block);
    block.stmts
}
fn expression(source: &str) -> Expr {
    let mut expression = syn::parse_str(source).expect("reviewed protocol expression contract");
    Normalize.visit_expr_mut(&mut expression);
    expression
}

// Only the selected function's body is visited. Its signature, attributes and
// every unmatched statement remain in the final complete-file comparison.
fn body<'a>(file: &'a mut syn::File, owner: Option<&str>, name: &str) -> Option<&'a mut Block> {
    let mut found = None;
    for item in &mut file.items {
        match item {
            Item::Fn(function) if owner.is_none() && function.sig.ident == name => {
                if found.is_some() {
                    return None;
                }
                found = Some(function.block.as_mut());
            }
            Item::Impl(implementation)
                if owner.is_some_and(|owner| tokens(&implementation.self_ty) == owner) =>
            {
                for item in &mut implementation.items {
                    if let ImplItem::Fn(function) = item {
                        if function.sig.ident == name {
                            if found.is_some() {
                                return None;
                            }
                            found = Some(&mut function.block);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    found
}

struct StatementRewrite {
    after: Vec<Stmt>,
    before: Vec<Stmt>,
    count: usize,
}
impl VisitMut for StatementRewrite {
    fn visit_block_mut(&mut self, block: &mut Block) {
        let mut index = 0;
        while index + self.after.len() <= block.stmts.len() {
            if block.stmts[index..index + self.after.len()]
                .iter()
                .zip(&self.after)
                .all(|(a, b)| tokens(a) == tokens(b))
            {
                block
                    .stmts
                    .splice(index..index + self.after.len(), self.before.clone());
                self.count += 1;
                index += self.before.len();
            } else {
                index += 1;
            }
        }
        // Deliberately do not recurse into an unrelated conditional or closure.
    }
}
struct ExpressionRewrite {
    after: Expr,
    before: Expr,
    count: usize,
}
impl VisitMut for ExpressionRewrite {
    fn visit_expr_mut(&mut self, value: &mut Expr) {
        if tokens(value) == tokens(&self.after) {
            *value = self.before.clone();
            self.count += 1;
        } else {
            syn::visit_mut::visit_expr_mut(self, value);
        }
    }
}
fn rewrite_statements(
    file: &mut syn::File,
    owner: Option<&str>,
    name: &str,
    rules: &[(&str, &str)],
) -> bool {
    let Some(body) = body(file, owner, name) else {
        return true;
    };
    rules.iter().all(|(after, before)| {
        let mut rule = StatementRewrite {
            after: statements(after),
            before: statements(before),
            count: 0,
        };
        assert!(!rule.after.is_empty());
        rule.visit_block_mut(body);
        rule.count <= 1
    })
}

fn insert_window(
    file: &mut syn::File,
    owner: Option<&str>,
    name: &str,
    prefix: &str,
    addition: &str,
    suffix: &str,
) -> bool {
    rewrite_statements(
        file,
        owner,
        name,
        &[(
            &format!("{prefix} {addition} {suffix}"),
            &format!("{prefix} {suffix}"),
        )],
    )
}

// A new early return must not move across any other statement. These small
// protocol-only bodies are closed contracts, rather than arbitrary deletions.
fn pure_body(file: &mut syn::File, owner: &str, name: &str, before: &str, after: &str) -> bool {
    let Some(body) = body(file, Some(owner), name) else {
        return true;
    };
    let expected = statements(after);
    if tokens(body)
        == tokens(&Block {
            brace_token: Default::default(),
            stmts: expected,
        })
    {
        body.stmts = statements(before);
    }
    true
}
fn rewrite_expressions(
    file: &mut syn::File,
    owner: Option<&str>,
    name: &str,
    rules: &[(&str, &str)],
) -> bool {
    let Some(body) = body(file, owner, name) else {
        return true;
    };
    rules.iter().all(|(after, before)| {
        let mut rule = ExpressionRewrite {
            after: expression(after),
            before: expression(before),
            count: 0,
        };
        rule.visit_block_mut(body);
        rule.count <= 1
    })
}
fn remove_additions(file: &mut syn::File, contracts: &str) -> bool {
    let contracts = parse(contracts).expect("reviewed protocol item contracts");
    for contract in contracts.items {
        if let Item::Impl(contract) = contract {
            for item in contract.items {
                let mut count = 0;
                for candidate in &mut file.items {
                    if let Item::Impl(candidate) = candidate {
                        if tokens(&candidate.self_ty) == tokens(&contract.self_ty) {
                            candidate.items.retain(|candidate| {
                                let matches = tokens(candidate) == tokens(&item);
                                count += usize::from(matches);
                                !matches
                            });
                        }
                    }
                }
                if count > 1 {
                    return false;
                }
            }
        } else {
            let mut count = 0;
            file.items.retain(|item| {
                let matches = tokens(item) == tokens(&contract);
                count += usize::from(matches);
                !matches
            });
            if count > 1 {
                return false;
            }
        }
    }
    true
}

/// Recognize only reviewed protocol edits and compare every remaining AST node.
/// Request timing/metadata, execution evidence, collection and draining are
/// never projected out. Unknown syntax or side effects retain their difference.
pub fn protocol_observability_unchanged(
    before: &str,
    after: &str,
    kind: ProtocolObservabilityKind,
) -> Result<bool, String> {
    let before = parse(before)?;
    let mut after = parse(after)?;
    if tokens(&before) == tokens(&after) {
        return Ok(true);
    }
    let recognized = match kind {
        ProtocolObservabilityKind::Requests => requests(&mut after),
        ProtocolObservabilityKind::Sequence => sequence(&mut after),
        ProtocolObservabilityKind::Completion => completion(&mut after),
    };
    Ok(recognized && tokens(&before) == tokens(&after))
}

const REQUEST_ADDITIONS: &str = r#"
impl ApiChatRequest {
    pub fn automatic_tools_with_hard_response_format(&self) -> bool {
        !self.tools.is_empty()
            && self.tool_choice.as_ref().is_none_or(|choice| {
                matches!(choice, ApiToolChoice::Mode(mode) if mode.eq_ignore_ascii_case("auto"))
            })
            && self.response_format.as_ref().is_some_and(|format| {
                format.format_type == "json_object" || (format.format_type == "json_schema"
                    && format.json_schema.as_ref().is_some_and(|schema| schema.strict == Some(true)))
            })
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredOutputBranch { Final, ToolCall, }
pub fn api_response_from_classified_generated_text(
    request: &InferenceRequest, text: &str, finish_reason: FinishReason, branch: StructuredOutputBranch,
) -> crate::Result<Option<ApiResponse>> {
    let Some(ApiRequest::Chat(chat_request)) = request.api_request.as_ref() else { return Ok(None); };
    if !matches!(finish_reason, FinishReason::Stop | FinishReason::EOS) { return Ok(None); }
    if !chat_request.automatic_tools_with_hard_response_format() {
        return Err(crate::FerrumError::invalid_request(
            "classified structured output requires automatic tools with a hard response format",
        ));
    }
    let (content, tool_calls, wire_finish_reason) = match branch {
        StructuredOutputBranch::Final => (text.to_string(), Vec::new(), "stop"),
        StructuredOutputBranch::ToolCall => {
            let calls = parse_explicit_tool_call_envelopes(text, chat_request)
                .or_else(|| parse_classified_named_json_call(text, chat_request))
                .filter(|calls| {
                    calls.iter().all(|call| {
                        serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                            .is_ok_and(|arguments| arguments.is_object())
                    })
                }).ok_or_else(|| {
                    crate::FerrumError::invalid_format(
                        "classified tool output is not a complete declared function call",
                    )
                })?;
            (String::new(), calls, "tool_calls")
        }
    };
    Ok(Some(ApiResponse::Chat(ApiChatResponse {
        message: ApiChatMessage {
            role: ApiMessageRole::Assistant, content, name: None, tool_calls,
            tool_call_id: None, function_call: None,
        },
        finish_reason: Some(wire_finish_reason.to_string()),
    })))
}
fn parse_classified_named_json_call(text: &str, chat_request: &ApiChatRequest) -> Option<Vec<ApiToolCall>> {
    if chat_request.tool_call_protocol != ApiToolCallProtocol::Json { return None; }
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let name = value.get("name")?.as_str()?;
    if !api_tool_name_allowed(chat_request, name) { return None; }
    let arguments = match (value.get("arguments"), value.get("parameters")) {
        (Some(arguments), None) | (None, Some(arguments)) => arguments,
        _ => return None,
    };
    if !arguments.is_object() { return None; }
    let call = ApiToolCall {
        id: "call_0".to_string(), tool_type: "function".to_string(),
        function: ApiFunctionCall { name: name.to_string(), arguments: serde_json::to_string(arguments).ok()?, },
    };
    validate_parsed_tool_calls(vec![call])
}
fn parse_explicit_tool_call_envelopes(text: &str, chat_request: &ApiChatRequest) -> Option<Vec<ApiToolCall>> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut remaining = text.trim();
    let mut calls = Vec::new();
    while !remaining.is_empty() {
        if calls.len() >= MAX_PARALLEL_TOOL_CALLS_PER_RESPONSE { return None; }
        let payload = remaining.strip_prefix(OPEN)?.trim_start();
        let parsed = match chat_request.tool_call_protocol {
            ApiToolCallProtocol::Json => {
                let mut values = serde_json::Deserializer::from_str(payload).into_iter::<serde_json::Value>();
                let value = values.next()?.ok()?;
                remaining = payload[values.byte_offset()..].trim_start().strip_prefix(CLOSE)?.trim_start();
                parse_json_tool_call_value(&value, chat_request, calls.len(), false)?
            }
            ApiToolCallProtocol::FunctionParameterXml => {
                let end = remaining.find(CLOSE)? + CLOSE.len();
                let envelope = &remaining[..end];
                remaining = remaining[end..].trim_start();
                let mut parsed = parse_function_parameter_xml_tool_calls(envelope, chat_request, true)?;
                for (index, call) in parsed.iter_mut().enumerate() {
                    call.id = format!("call_{}", calls.len() + index);
                }
                parsed
            }
        };
        if calls.len() + parsed.len() > MAX_PARALLEL_TOOL_CALLS_PER_RESPONSE { return None; }
        for call in parsed {
            if calls.iter().any(|previous: &ApiToolCall| previous.id == call.id) { return None; }
            calls.push(call);
        }
    }
    validate_parsed_tool_calls(calls)
}
"#;

fn requests(file: &mut syn::File) -> bool {
    let control_guard = "if self.tools.is_empty() || api_tool_choice_is_none(self) { return &[]; }";
    let control_tail = "self.tool_call_protocol.generated_control_token_texts()";
    let envelope_guard =
        "if self.tools.is_empty() || api_tool_choice_is_none(self) { return None; }";
    let envelope_tail = "self.tool_call_protocol.generated_response_envelope()";
    if !remove_additions(file, REQUEST_ADDITIONS)
        || !pure_body(
            file,
            "ApiChatRequest",
            "generated_control_token_texts",
            &format!("{control_guard} {control_tail}"),
            &format!(
                r#"{control_guard} if self.automatic_tools_with_hard_response_format() {{ return &["<tool_call>", "</tool_call>"]; }} {control_tail}"#
            ),
        )
        || !pure_body(
            file,
            "ApiChatRequest",
            "generated_response_envelope",
            &format!("{envelope_guard} {envelope_tail}"),
            &format!(
                r#"{envelope_guard} if self.automatic_tools_with_hard_response_format() {{
                return Some(ResponseCompletionEnvelope {{
                    open_token_text: "<tool_call>".to_string(), close_token_text: "</tool_call>".to_string(),
                    max_envelopes: MAX_PARALLEL_TOOL_CALLS_PER_RESPONSE,
                }});
            }} {envelope_tail}"#
            ),
        )
        || !rewrite_expressions(
            file,
            None,
            "parse_tool_calls_from_generated_text",
            &[(
                "parse_function_parameter_xml_tool_calls(text, chat_request, false)",
                "parse_function_parameter_xml_tool_calls(text, chat_request)",
            )],
        )
    {
        return false;
    }

    if let Some(caller) = body(file, None, "parse_tool_calls_from_generated_text") {
        let addition = statements("if chat_request.automatic_tools_with_hard_response_format() { return parse_explicit_tool_call_envelopes(text, chat_request); }");
        let following = statements("if chat_request.tool_call_protocol == ApiToolCallProtocol::FunctionParameterXml { if let Some(calls) = parse_function_parameter_xml_tool_calls(text, chat_request) { return validate_parsed_tool_calls(calls); } }");
        if caller.stmts.first().map(tokens) == addition.first().map(tokens)
            && caller.stmts.get(1).map(tokens) == following.first().map(tokens)
        {
            caller.stmts.remove(0);
        }
    }

    // Reverse the extraction, preserving the entire parsed-call body. Only
    // index rebasing and explicit-argument policy are the reviewed changes.
    let helper_indices: Vec<_> = file
        .items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            matches!(item, Item::Fn(function) if function.sig.ident == "parse_json_tool_call_value")
                .then_some(index)
        })
        .collect();
    if helper_indices.len() > 1 {
        return false;
    }
    if let Some(index) = helper_indices.first().copied() {
        let Item::Fn(helper) = &file.items[index] else {
            return false;
        };
        let contract: syn::ItemFn = syn::parse_str("fn parse_json_tool_call_value(value: &serde_json::Value, chat_request: &ApiChatRequest, index_offset: usize, allow_unwrapped_arguments: bool) -> Option<Vec<ApiToolCall>> {}").unwrap();
        if tokens(&helper.sig) != tokens(&contract.sig)
            || !helper.attrs.is_empty()
            || !matches!(helper.vis, syn::Visibility::Inherited)
        {
            return false;
        }
        if !rewrite_expressions(file, None, "parse_json_tool_call_value", &[
            ("parse_tool_call_value(value, index_offset + index, chat_request)", "parse_tool_call_value(value, index, chat_request)"),
            ("parse_tool_call_value(tool_call, index_offset, chat_request)", "parse_tool_call_value(tool_call, 0, chat_request)"),
            ("parse_wrapped_tool_call_value(value, index_offset, chat_request)", "parse_wrapped_tool_call_value(&value, 0, chat_request)"),
            ("parse_tool_call_value(value, index_offset, chat_request)", "parse_tool_call_value(&value, 0, chat_request)"),
            ("|| { allow_unwrapped_arguments.then(|| parse_forced_tool_arguments_value(value, index_offset, chat_request)).flatten() }", "|| parse_forced_tool_arguments_value(&value, 0, chat_request)"),
        ]) { return false; }
        let Item::Fn(helper) = file.items.remove(index) else {
            return false;
        };
        let Some(caller) = body(file, None, "parse_tool_calls_from_generated_text") else {
            return false;
        };
        if caller.stmts.last().map(tokens)
            != statements("parse_json_tool_call_value(&value, chat_request, 0, true)")
                .last()
                .map(tokens)
        {
            return false;
        }
        caller.stmts.pop();
        caller.stmts.extend(helper.block.stmts);
    }
    for name in [
        "parse_function_parameter_xml_tool_calls",
        "parse_function_parameter_xml_arguments",
    ] {
        let functions: Vec<_> = file
            .items
            .iter_mut()
            .filter_map(|item| match item {
                Item::Fn(f) if f.sig.ident == name => Some(f),
                _ => None,
            })
            .collect();
        if functions.len() > 1 {
            return false;
        }
        for function in functions {
            if function
                .sig
                .inputs
                .last()
                .is_some_and(|input| tokens(input) == "require_complete_parameters : bool")
            {
                function.sig.inputs.pop();
                function.sig.inputs.pop_punct();
            }
        }
    }
    rewrite_expressions(file, None, "parse_function_parameter_xml_tool_calls", &[(
        "parse_function_parameter_xml_arguments(&function[name_end + 1..arguments_end], parameter_schema, require_complete_parameters)",
        "parse_function_parameter_xml_arguments(&function[name_end + 1..arguments_end], parameter_schema)"
    )]) && xml_parameter_guards(file)
}

fn xml_parameter_guards(file: &mut syn::File) -> bool {
    let Some(body) = body(file, None, "parse_function_parameter_xml_arguments") else {
        return true;
    };
    let mut matching_loops = 0;
    for statement in &mut body.stmts {
        if let Stmt::Expr(Expr::While(scan), _) = statement {
            if tokens(&scan.cond)
                == tokens(&expression(
                    "let Some(parameter_start) = remaining.find(PARAMETER_START)",
                ))
            {
                matching_loops += 1;
                let guard = statements("if require_complete_parameters && !remaining[..parameter_start].trim().is_empty() { return None; }");
                let next = statements(
                    "remaining = &remaining[parameter_start + PARAMETER_START.len()..];",
                );
                if scan.body.stmts.first().map(tokens) == guard.first().map(tokens)
                    && scan.body.stmts.get(1).map(tokens) == next.first().map(tokens)
                {
                    scan.body.stmts.remove(0);
                }
            }
        }
    }
    if matching_loops > 1 {
        return false;
    }
    let guard = statements("if require_complete_parameters && !remaining.trim().is_empty() { return None; } Some(arguments)");
    if body.stmts.len() >= guard.len()
        && body.stmts[body.stmts.len() - guard.len()..]
            .iter()
            .zip(&guard)
            .all(|(a, b)| tokens(a) == tokens(b))
    {
        body.stmts.remove(body.stmts.len() - guard.len());
    }
    true
}

fn sequence(file: &mut syn::File) -> bool {
    let factory_after = "factory.create_processor_with_chat_contract(&request.sampling_params.response_format, &request.sampling_params.structured_output_start, request.sampling_params.max_tokens, &stop_token_ids, &stop_text_seqs, request.api_request.as_ref().and_then(|request| match request { ferrum_types::ApiRequest::Chat(chat) => Some(chat), ferrum_types::ApiRequest::Completion(_) => None, }), request.sampling_params.model_output_protocol)";
    let factory_before = "factory.create_processor(&request.sampling_params.response_format, &request.sampling_params.structured_output_start, request.sampling_params.max_tokens, &stop_token_ids, &stop_text_seqs)";
    rewrite_expressions(
        file,
        Some("SequenceState"),
        "try_new_with_tokenizer_model_vocab_and_structured_factory",
        &[(factory_after, factory_before)],
    ) && pure_body(file, "SequenceState", "stop_reason",
        &format!("{USER_STOPS} {MODEL_STOPS}"),
        &format!("{USER_STOPS} {PROTOCOL_STOP} {MODEL_STOPS}"))
    && rewrite_expressions(
        file,
        Some("SequenceState"),
        "should_stream_generated_token",
        &[(
            r#"(tokenizer.is_some_and(|tokenizer| {
                ["<|call|>", "<|return|>"].into_iter().filter_map(|marker| tokenizer.token_id(marker)).any(|terminal| terminal == token)
            }) || (stop_reason == Some(FinishReason::EOS) && !self.stop_token_ids.contains(&token.get())
                && self.structured_output_processor.as_ref().is_some_and(|processor| {
                    processor.is_protocol_complete_with_terminals(&self.generated_tokens, &self.stop_token_ids).unwrap_or(false)
                })))"#,
            r#"tokenizer.is_some_and(|tokenizer| {
                ["<|call|>", "<|return|>"].into_iter().filter_map(|marker| tokenizer.token_id(marker)).any(|terminal| terminal == token)
            })"#,
        )],
    ) && rewrite_statements(file, Some("SequenceState"), "sample_and_commit_with_processors_and_tokenizer", &[(
        &format!("if let Some(processor) = &self.structured_output_processor {{ {GRAMMAR_PREFIX} {ACCEPTING_ROOT} if !constraint.accepting {{ {NATIVE_MASK} }} }}"),
        &format!("if let Some(processor) = &self.structured_output_processor {{ {GRAMMAR_PREFIX} if !constraint.accepting {{ mask_stop_token_logits(logits, &self.stop_token_ids); }} }}")
    )])
}

const USER_STOPS: &str = r#"
if let Some(&last_token) = self.generated_tokens.last() {
    if self.user_stop_token_ids.contains(&last_token.get()) { return Some(FinishReason::Stop); }
}
if !self.stop_text_seqs.is_empty() {
    if let Some(tok) = tokenizer {
        if let Ok(text) = tok.decode(&self.generated_tokens, true) {
            if self.stop_text_seqs.iter().any(|stop| !stop.is_empty() && text.contains(stop)) { return Some(FinishReason::Stop); }
        }
    }
}
"#;
const MODEL_STOPS: &str = r#"
if let Some(last_token) = self.generated_tokens.last() {
    if self.model_eos_token_ids.contains(&last_token.get()) { return Some(FinishReason::EOS); }
}
if self.generated_tokens.len() >= self.sampling_params.max_tokens { return Some(FinishReason::Length); }
None
"#;
const GRAMMAR_PREFIX: &str = r#"
let constraint = processor.mask_logits_with_terminals(logits, &self.generated_tokens, &self.stop_token_ids, &self.allowed_extended_token_ids)?;
required_structured_delimiter_token_id = constraint.required_delimiter_token_id;
grammar_start_token_index = constraint.grammar_start_token_index;
"#;
const ACCEPTING_ROOT: &str = r#"
if constraint.accepting && processor.complete_root_text_len_with_terminals(&self.generated_tokens, &self.stop_token_ids)?.is_some() {
    self.response_completion_state = ResponseCompletionState::Satisfied;
    if let Some(mask) = &mut self.argmax_token_mask { mask.set_tokens_validity(&self.model_eos_token_ids, true); }
    if let Some(mask) = &mut self.initial_argmax_token_mask { mask.set_tokens_validity(&self.model_eos_token_ids, true); }
}
"#;
const NATIVE_MASK: &str = r#"
if constraint.grammar_owned_terminal_token_id.is_some() {
    for &token_id in &self.stop_token_ids {
        if constraint.grammar_owned_terminal_token_id != Some(token_id) {
            if let Some(logit) = logits.get_mut(token_id as usize) { *logit = f32::NEG_INFINITY; }
        }
    }
} else { mask_stop_token_logits(logits, &self.stop_token_ids); }
"#;
const PROTOCOL_STOP: &str = r#"
if let Some(processor) = &self.structured_output_processor {
    match processor.is_protocol_complete_with_terminals(&self.generated_tokens, &self.stop_token_ids) {
        Ok(true) => return Some(FinishReason::EOS),
        Ok(false) => {}
        Err(_) => return Some(FinishReason::Error),
    }
}
"#;
const CLASSIFIED_RESPONSE: &str = r#"
impl SequenceState {
    fn classified_structured_api_response(&self, finish_reason: FinishReason) -> Result<Option<ferrum_types::ApiResponse>> {
        if !matches!(finish_reason, FinishReason::Stop | FinishReason::EOS) { return Ok(None); }
        let Some(processor) = self.structured_output_processor.as_ref() else { return Ok(None); };
        let Some((branch, payload)) = processor.classified_result_with_terminals(&self.generated_tokens, &self.stop_token_ids)? else {
            return Ok(None);
        };
        ferrum_types::api_response_from_classified_generated_text(&self.original_request, &payload, finish_reason, branch)
    }
}
"#;
const ROOT_STOP_BOUNDARY: &str = r#"
if let Some(root_end) = processor.complete_root_text_len_with_terminals(&self.generated_tokens, &self.stop_token_ids)? {
    if stop_end < root_end {
        return Err(FerrumError::model("stop sequence truncated the structured output before its complete tool or final result",));
    }
    return Ok(());
}
"#;
fn completion(file: &mut syn::File) -> bool {
    remove_additions(file, CLASSIFIED_RESPONSE)
        && insert_window(file, Some("SequenceState"), "validate_structured_stop_boundary",
            "if stop_end == full.len() { return Ok(()); }", ROOT_STOP_BOUNDARY,
            "let progress = processor.progress_with_terminals(&self.generated_tokens, &self.stop_token_ids)?;")
        && rewrite_statements(file, Some("EngineInner"), "complete_request_inner", &[("let mut classified_api_response = None;", "")])
        && rewrite_expressions(file, Some("EngineInner"), "complete_request_inner", &[
            (r#"seq.structured_output_terminal_error(finish_reason).or_else(|| {
                seq.validate_structured_stop_boundary(self.tokenizer.as_ref()).err()
            }).or_else(|| match seq.classified_structured_api_response(finish_reason) {
                Ok(response) => { classified_api_response = response; None }
                Err(error) => Some(error),
            })"#,
            r#"seq.structured_output_terminal_error(finish_reason).or_else(|| {
                seq.validate_structured_stop_boundary(self.tokenizer.as_ref()).err()
            })"#),
            (r#"classified_api_response.take().or_else(|| {
                ferrum_types::api_response_from_generated_text(&seq.original_request, &text, finish_reason)
            })"#,
            "ferrum_types::api_response_from_generated_text(&seq.original_request, &text, finish_reason)"),
        ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_preserve_timing_and_metadata_and_reject_unknown_protocol_effects() {
        let before = r#"
            struct RequestTokenTiming { enabled: bool }
            struct InferenceRequest { metadata: Metadata, timing: RequestTokenTiming }
            impl InferenceRequest {
                fn with_metadata(mut self, metadata: Metadata) -> Self { self.metadata = metadata; self }
            }
            impl ApiChatRequest {
                fn generated_control_token_texts(&self) -> &[&str] {
                    if self.tools.is_empty() || api_tool_choice_is_none(self) { return &[]; }
                    self.tool_call_protocol.generated_control_token_texts()
                }
            }
        "#;
        let insert = r#"if self.automatic_tools_with_hard_response_format() { return &["<tool_call>", "</tool_call>"]; }
            self.tool_call_protocol.generated_control_token_texts()"#;
        let after = before.replace(
            "self.tool_call_protocol.generated_control_token_texts()",
            insert,
        );
        assert!(protocol_observability_unchanged(
            before,
            &after,
            ProtocolObservabilityKind::Requests
        )
        .unwrap());
        for bad in [
            after.replace("enabled: bool", "enabled: Option<bool>"),
            after.replace("self.metadata = metadata", "self.metadata.clear()"),
            after.replace(
                "return &[\"<tool_call>\", \"</tool_call>\"]",
                "record_event(); return &[\"<tool_call>\", \"</tool_call>\"]",
            ),
            format!("{after} fn extra() {{ unknown_effect(); }}"),
        ] {
            assert!(!protocol_observability_unchanged(
                before,
                &bad,
                ProtocolObservabilityKind::Requests
            )
            .unwrap());
        }
        // Exact new item contracts do not hide an arbitrary call or attribute.
        let additions = parse(REQUEST_ADDITIONS).unwrap();
        let contract = additions.items.iter().find(|item| matches!(item, Item::Fn(f) if f.sig.ident == "parse_classified_named_json_call")).unwrap();
        let with_helper = format!("{after} {}", tokens(contract));
        assert!(protocol_observability_unchanged(
            before,
            &with_helper,
            ProtocolObservabilityKind::Requests
        )
        .unwrap());
        let mut effectful_helper = contract.clone();
        let Item::Fn(function) = &mut effectful_helper else {
            unreachable!()
        };
        function
            .block
            .stmts
            .insert(0, syn::parse_quote!(record_event();));
        for bad in [
            format!("{after} {}", tokens(&effectful_helper)),
            format!("{after} #[cfg(unix)] {}", tokens(contract)),
        ] {
            assert!(!protocol_observability_unchanged(
                before,
                &bad,
                ProtocolObservabilityKind::Requests
            )
            .unwrap());
        }
    }

    #[test]
    fn sequence_retains_capture_drain_and_lifecycle_on_both_sides_of_protocol_stop() {
        let before = format!(
            r#"
            struct SequenceState {{ timing: Option<Timing> }}
            impl SequenceState {{
                fn stop_reason(&self) -> Option<FinishReason> {{ {USER_STOPS} {MODEL_STOPS} }}
                fn record_generated_token_commit(&mut self) {{ self.timing.record(); }}
                fn take_execution_evidence(&mut self) -> Evidence {{ self.timing.take().finish() }}
            }}
            impl Drop for SequenceState {{ fn drop(&mut self) {{ self.release(); }} }}
        "#
        );
        let after = before.replace(MODEL_STOPS, &format!("{PROTOCOL_STOP} {MODEL_STOPS}"));
        assert!(protocol_observability_unchanged(
            &before,
            &after,
            ProtocolObservabilityKind::Sequence
        )
        .unwrap());
        for bad in [
            after.replace("self.timing.record();", ""),
            after.replace("self.timing.take()", "self.timing.clone()"),
            after.replace("self.release();", ""),
            after
                .replace(
                    "Ok(true) => return",
                    "Ok(true) => { self.timing.clear(); return",
                )
                .replace("Some(FinishReason::EOS),", "Some(FinishReason::EOS); },"),
            after.replace("self.release();", "unknown_effect(); self.release();"),
        ] {
            assert!(!protocol_observability_unchanged(
                &before,
                &bad,
                ProtocolObservabilityKind::Sequence
            )
            .unwrap());
        }
        let with_capture = "impl SequenceState { fn stop_reason(&self) -> Option<FinishReason> { self.timing.record(); None } }";
        let moved = with_capture.replace(
            "self.timing.record();",
            &format!("{PROTOCOL_STOP} self.timing.record();"),
        );
        assert!(!protocol_observability_unchanged(
            with_capture,
            &moved,
            ProtocolObservabilityKind::Sequence
        )
        .unwrap());
    }

    #[test]
    fn inserted_protocol_returns_and_errors_cannot_move_ahead_of_capture() {
        let request = r#"impl ApiChatRequest {
            fn generated_control_token_texts(&self) -> &[&str] {
                self.timing.record();
                if self.tools.is_empty() || api_tool_choice_is_none(self) { return &[]; }
                self.tool_call_protocol.generated_control_token_texts()
            }
        }"#;
        let moved = request.replace("self.timing.record();", r#"if self.automatic_tools_with_hard_response_format() { return &["<tool_call>", "</tool_call>"]; } self.timing.record();"#);
        assert!(!protocol_observability_unchanged(
            request,
            &moved,
            ProtocolObservabilityKind::Requests
        )
        .unwrap());
        let parser = "fn parse_tool_calls_from_generated_text(text: &str, chat_request: &ApiChatRequest) -> Option<Vec<ApiToolCall>> { record(); if chat_request.tool_call_protocol == ApiToolCallProtocol::FunctionParameterXml { if let Some(calls) = parse_function_parameter_xml_tool_calls(text, chat_request) { return validate_parsed_tool_calls(calls); } } None }";
        let moved = parser.replace("record();", "if chat_request.automatic_tools_with_hard_response_format() { return parse_explicit_tool_call_envelopes(text, chat_request); } record();");
        assert!(!protocol_observability_unchanged(
            parser,
            &moved,
            ProtocolObservabilityKind::Requests
        )
        .unwrap());
        let before = format!("impl SequenceState {{ fn sample_and_commit_with_processors_and_tokenizer(&mut self) {{ record(); if let Some(processor) = &self.structured_output_processor {{ {GRAMMAR_PREFIX} if !constraint.accepting {{ mask_stop_token_logits(logits, &self.stop_token_ids); }} }} commit(); }} }}");
        let after = before.replace(
            "if !constraint.accepting { mask_stop_token_logits(logits, &self.stop_token_ids); }",
            &format!("{ACCEPTING_ROOT} if !constraint.accepting {{ {NATIVE_MASK} }}"),
        );
        assert!(protocol_observability_unchanged(
            &before,
            &after,
            ProtocolObservabilityKind::Sequence
        )
        .unwrap());
        let moved = after
            .replace(ACCEPTING_ROOT, "")
            .replace("record();", &format!("{ACCEPTING_ROOT} record();"));
        assert!(!protocol_observability_unchanged(
            &before,
            &moved,
            ProtocolObservabilityKind::Sequence
        )
        .unwrap());
    }

    #[test]
    fn completion_keeps_terminal_evidence_and_unknown_control_flow() {
        let before = r#"
            impl SequenceState {
                fn validate_structured_stop_boundary(&self) -> Result<()> {
                    self.timing.record();
                    if stop_end == full.len() { return Ok(()); }
                    let progress = processor.progress_with_terminals(&self.generated_tokens, &self.stop_token_ids)?;
                    validate_json()?; Ok(())
                }
            }
            impl EngineInner {
                async fn complete_request_inner(&self) -> Result<()> {
                    let text = decode();
                    let api_response = ferrum_types::api_response_from_generated_text(&seq.original_request, &text, finish_reason);
                    let execution_evidence = seq.take_execution_evidence()?;
                    send(api_response, execution_evidence).await?;
                    release().await;
                    Ok(())
                }
            }
        "#;
        let after = before.replace("let progress =", &format!("{ROOT_STOP_BOUNDARY} let progress ="))
            .replace("let text = decode();", "let mut classified_api_response = None; let text = decode();")
            .replace("ferrum_types::api_response_from_generated_text(&seq.original_request, &text, finish_reason)",
                "classified_api_response.take().or_else(|| { ferrum_types::api_response_from_generated_text(&seq.original_request, &text, finish_reason) })");
        assert!(protocol_observability_unchanged(
            before,
            &after,
            ProtocolObservabilityKind::Completion
        )
        .unwrap());
        for bad in [
            after.replace("seq.take_execution_evidence()?", "None"),
            after.replace(
                "send(api_response, execution_evidence)",
                "send(api_response, None)",
            ),
            after.replace("release().await;", ""),
            after.replace("validate_json()?;", "return Ok(()); validate_json()?;"),
            after.replace("return Ok(());", "record_event(); return Ok(());"),
            after.replace(ROOT_STOP_BOUNDARY, "").replace(
                "self.timing.record();",
                &format!("{ROOT_STOP_BOUNDARY} self.timing.record();"),
            ),
        ] {
            assert!(!protocol_observability_unchanged(
                before,
                &bad,
                ProtocolObservabilityKind::Completion
            )
            .unwrap());
        }
    }
}
