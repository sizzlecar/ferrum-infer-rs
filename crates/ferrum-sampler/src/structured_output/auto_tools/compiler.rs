use super::*;
use ferrum_types::{ApiChatRequest, ApiToolCallProtocol, ModelOutputProtocol};
use llguidance::api::GrammarWithLexer;

pub(super) struct Compiled {
    pub grammar: TopLevelGrammar,
    pub plan: Plan,
    pub budget: Option<StructuredOutputBudgetPlan>,
}

struct Builder<'a> {
    factory: &'a StructuredOutputFactory,
    rules: Vec<String>,
    schemas: Vec<GrammarWithLexer>,
    marker_bytes: HashMap<u32, Vec<u8>>,
    boundary_controls: HashSet<u32>,
    markers: HashMap<String, String>,
}

impl Builder<'_> {
    fn marker(&mut self, text: &str) -> String {
        if let Some(rule) = self.markers.get(text) {
            return rule.clone();
        }
        let rule = format!("marker_{}", self.markers.len());
        let mut variants = vec![literal(text)];
        if let Some(token) = self.factory.tokenizer.token_id(text) {
            let id = token.get();
            if self
                .factory
                .defined_token_ids
                .get(id as usize)
                .copied()
                .unwrap_or(false)
            {
                if tokenizer_special_ids(self.factory.tokenizer.as_ref()).contains(&id) {
                    variants.push(format!("<[{id}]>"));
                }
                self.marker_bytes.insert(id, text.as_bytes().to_vec());
                self.boundary_controls.insert(id);
            }
        }
        self.rules.push(format!("{rule}: {}", variants.join(" | ")));
        self.markers.insert(text.to_string(), rule.clone());
        rule
    }

    fn schema(&mut self, name: &str, schema: serde_json::Value) -> Result<String> {
        let mut schema = compact_json_schema(schema)?;
        // Both branches must permit ordinary JSON formatting. Restricting
        // final JSON to compact separators can mask a model's intended answer
        // while leaving a lower-scoring tool call available. Request budgets
        // and the processor's liveness checks still bound generation; compiler
        // options remain owned by Ferrum rather than request x-guidance.
        JsonCompileOptions::default().apply_to(&mut schema);
        self.schemas.push(GrammarWithLexer {
            name: Some(name.to_string()),
            json_schema: Some(schema),
            lark_grammar: None,
        });
        Ok(format!("@{name}"))
    }

    fn wire(&mut self, text: &str) -> String {
        let mut result = Vec::new();
        let mut remaining = text;
        while let Some(start) = remaining.find('<') {
            if start > 0 {
                result.push(literal(&remaining[..start]));
            }
            let suffix = &remaining[start..];
            if let Some(end) = suffix.find('>') {
                result.push(self.marker(&suffix[..=end]));
                remaining = &suffix[end + 1..];
            } else {
                result.push(literal(suffix));
                remaining = "";
            }
        }
        if !remaining.is_empty() {
            result.push(literal(remaining));
        }
        result.join(" ")
    }

    fn reasoning(&mut self, name: &str, closing: &str) {
        // A lexical rule cannot reference a numeric-token grammar rule.
        // Instead, its byte language excludes framing, then the grammar
        // owns the complete closing marker (literal or atomic token).
        let body = format!("{}_BODY", name.to_ascii_uppercase());
        self.rules.push(format!(
            "{body}: /(?s:.*)/ & ~(/(?s:.*)/ ({} | {} | {}) /(?s:.*)/)",
            literal(closing),
            literal("<|"),
            literal("<tool_call>")
        ));
        let lexical = format!("{name}_text_close");
        self.rules
            .push(format!("{lexical}[lazy]: {body} {}", literal(closing)));
        self.marker(closing);
        let numeric = self.factory.tokenizer.token_id(closing).filter(|token| {
            tokenizer_special_ids(self.factory.tokenizer.as_ref()).contains(&token.get())
        });
        self.rules.push(if let Some(token) = numeric {
            format!("{name}: {lexical} | {body} <[{}]>", token.get())
        } else {
            format!("{name}: {lexical}")
        });
    }
}

fn literal(text: &str) -> String {
    serde_json::to_string(text).expect("string serialization")
}

pub(super) fn build(
    factory: &StructuredOutputFactory,
    response_format: &ResponseFormat,
    start: &StructuredOutputStart,
    max_tokens: usize,
    stop_ids: &HashSet<u32>,
    stop_texts: &[String],
    chat: &ApiChatRequest,
    protocol: ModelOutputProtocol,
) -> Result<Compiled> {
    let tool_only = chat.requires_native_tool_call();
    if tool_only && protocol == ModelOutputProtocol::HarmonyGptOss {
        return Err(FerrumError::invalid_request(
            "native XML tool framing cannot use the Harmony output protocol",
        ));
    }
    let mut builder = Builder {
        factory,
        rules: vec!["ws: /[ \\t\\r\\n]*/".into()],
        schemas: Vec::new(),
        marker_bytes: HashMap::new(),
        boundary_controls: HashSet::new(),
        markers: HashMap::new(),
    };
    if !tool_only {
        let final_schema = match response_format {
            ResponseFormat::JsonObject => json!({"type":"object"}),
            ResponseFormat::JsonSchema(schema) => {
                serde_json::from_str(schema).map_err(|error| {
                    FerrumError::invalid_request(format!(
                        "response_format.schema is not valid JSON: {error}"
                    ))
                })?
            }
            ResponseFormat::Text => {
                return Err(FerrumError::internal(
                    "automatic structured tools require a final schema",
                ))
            }
        };
        let final_rule = builder.schema("final_schema", final_schema)?;
        builder
            .rules
            .push(format!("final_value[capture]: {final_rule}"));
    }
    let mut calls = Vec::new();
    let mut named_calls = Vec::new();
    let mut argument_rules = Vec::new();
    let mut names = Vec::new();
    for (index, tool) in chat
        .tools
        .iter()
        .filter(|tool| {
            tool.tool_type == "function"
                && (!tool_only || chat.allows_tool_name(&tool.function.name))
        })
        .enumerate()
    {
        let name = &tool.function.name;
        let schema = tool
            .function
            .parameters
            .clone()
            .unwrap_or_else(|| json!({"type":"object"}));
        let arguments = if chat.tool_call_protocol == ApiToolCallProtocol::FunctionParameterXml
            && protocol != ModelOutputProtocol::HarmonyGptOss
        {
            // XML parameters are decoded and checked by the product's
            // complete schema validator. Do not compile an unused JSON
            // subgrammar and reject schemas this wire path never samples.
            String::new()
        } else {
            builder.schema(&format!("arguments_{index}"), schema)?
        };
        argument_rules.push(arguments.clone());
        names.push(name.clone());
        let call = format!("call_{index}");
        let open = builder.marker("<tool_call>");
        let close = builder.marker("</tool_call>");
        match chat.tool_call_protocol {
            ApiToolCallProtocol::Json => {
                // Keep each argument schema in its own root namespace so local
                // $ref values are not accidentally rebound by a JSON wrapper.
                let name_json = serde_json::to_string(name).expect("tool name serialization");
                let name_field = format!(
                    "{} ws {} ws {}",
                    literal("\"name\""),
                    literal(":"),
                    literal(&name_json)
                );
                let mut forms = Vec::new();
                for field in ["\"arguments\"", "\"parameters\""] {
                    let arguments_field =
                        format!("{} ws {} ws {arguments}", literal(field), literal(":"));
                    forms.push(format!(
                        "{} ws {name_field} ws {} ws {arguments_field} ws {}",
                        literal("{"),
                        literal(","),
                        literal("}")
                    ));
                    forms.push(format!(
                        "{} ws {arguments_field} ws {} ws {name_field} ws {}",
                        literal("{"),
                        literal(","),
                        literal("}")
                    ));
                }
                let named = format!("named_call_{index}");
                builder
                    .rules
                    .push(format!("{named}: {}", forms.join(" | ")));
                builder
                    .rules
                    .push(format!("{call}: {open} ws {named} ws {close}"));
                named_calls.push(named);
            }
            ApiToolCallProtocol::FunctionParameterXml => {
                let function = literal(&format!("<function={name}>"));
                builder.rules.push(format!(
                    "{call}: {open} ws {function} ws xml_parameter* {} ws {close}",
                    literal("</function>")
                ));
            }
        }
        calls.push(call);
    }
    if calls.is_empty() {
        return Err(FerrumError::invalid_request(
            "structured tools require an allowed declared function",
        ));
    }
    if chat.tool_call_protocol == ApiToolCallProtocol::FunctionParameterXml {
        // XML values intentionally retain the existing schema-aware terminal
        // decoder/validator. This grammar owns complete framing, not a new
        // JSON-Schema-to-XML translation.
        builder.rules.extend([
            format!(
                "xml_parameter: {} /[^<>\\r\\n=]+/ {} xml_value ws",
                literal("<parameter="),
                literal(">")
            ),
            format!("xml_value[lazy]: /(?s:.*)/ {}", literal("</parameter>")),
        ]);
    }
    builder
        .rules
        .push(format!("tool_call: {}", calls.join(" | ")));
    let bare = if named_calls.is_empty() {
        String::new()
    } else {
        format!(" | {}", named_calls.join(" | "))
    };
    builder.rules.push(format!(
        "tool_result[capture]: tool_call (ws tool_call){{0,31}}{bare}"
    ));
    builder.rules.push(if tool_only {
        "result: ws tool_result ws".into()
    } else {
        "result: ws (final_value | tool_result) ws".into()
    });

    let mut headers = Vec::new();
    let mut following_headers = Vec::new();
    let mut reasoning_close = Vec::new();
    let mut native_terminals = HashSet::new();
    let mut required_wire = Vec::new();
    if protocol != ModelOutputProtocol::HarmonyGptOss {
        required_wire.extend(["<tool_call>".to_string(), "</tool_call>".to_string()]);
        if chat.tool_call_protocol == ApiToolCallProtocol::FunctionParameterXml {
            required_wire.extend(["</function>".to_string(), "</parameter>".to_string()]);
            required_wire.extend(names.iter().map(|name| format!("<function={name}>")));
        }
    }
    let mut allow_reasoning = false;
    let mut max_initial = 0;
    let mut max_following = 0;
    let root;
    if protocol == ModelOutputProtocol::HarmonyGptOss {
        let return_rule = builder.marker("<|return|>");
        let call_rule = builder.marker("<|call|>");
        required_wire.extend([
            "<|end|>".to_string(),
            "<|call|>".to_string(),
            "<|return|>".to_string(),
        ]);
        for marker in ["<|return|>", "<|call|>"] {
            let id = factory
                .tokenizer
                .token_id(marker)
                .ok_or_else(|| {
                    FerrumError::config(format!(
                        "Harmony structured tools require token {marker:?}"
                    ))
                })?
                .get();
            if !stop_ids.contains(&id) {
                return Err(FerrumError::invalid_request(format!(
                    "Harmony structured tools require {marker} as a resolved terminal"
                )));
            }
            native_terminals.insert(id);
        }
        let mut direct_results = Vec::new();
        let mut follow_results = Vec::new();
        for prefix in ["", "<|start|>assistant"] {
            let text = format!("{prefix}<|channel|>final<|message|>");
            headers.push(Header::result(&text));
            max_initial = max_initial.max(token_count(factory, &text)?);
            required_wire.push(text.clone());
            let wire = builder.wire(&text);
            direct_results.push(format!("{wire} ws final_value ws {return_rule}"));
        }
        let final_follow = "<|start|>assistant<|channel|>final<|message|>";
        following_headers.push(Header::result(final_follow));
        required_wire.push(final_follow.to_string());
        max_following = token_count(factory, final_follow)?;
        let final_follow_wire = builder.wire(final_follow);
        follow_results.push(format!(
            "{final_follow_wire} ws final_value ws {return_rule}"
        ));
        for (name, arguments) in names.iter().zip(&argument_rules) {
            for constrain in ["", "<|constrain|>json"] {
                for header in [
                    format!("<|channel|>commentary to=functions.{name}{constrain}<|message|>"),
                    format!("<|start|>assistant<|channel|>commentary to=functions.{name}{constrain}<|message|>"),
                    format!("<|start|>assistant to=functions.{name}<|channel|>commentary{constrain}<|message|>"),
                ] {
                    let wire = builder.wire(&header);
                    direct_results.push(format!("{wire} ws {arguments} ws {call_rule}"));
                    headers.push(Header::result(&header));
                    max_initial = max_initial.max(token_count(factory, &header)?);
                    required_wire.push(header.clone());
                    if header.starts_with("<|start|>") {
                        follow_results.push(format!("{wire} ws {arguments} ws {call_rule}"));
                        following_headers.push(Header::result(&header));
                        max_following = max_following.max(token_count(factory, &header)?);
                    }
                }
            }
        }
        let mut analysis_headers = Vec::new();
        for text in [
            "<|channel|>analysis<|message|>",
            "<|start|>assistant<|channel|>analysis<|message|>",
        ] {
            analysis_headers.push(builder.wire(text));
            headers.push(Header::reasoning(text));
            max_initial = max_initial.max(token_count(factory, text)?);
            required_wire.push(text.to_string());
        }
        builder.reasoning("analysis", "<|end|>");
        builder
            .rules
            .push(format!("native_result: {}", direct_results.join(" | ")));
        builder
            .rules
            .push(format!("native_followup: {}", follow_results.join(" | ")));
        root = format!(
            "native_result | ({}) analysis native_followup",
            analysis_headers.join(" | ")
        );
        allow_reasoning = true;
        reasoning_close = b"<|end|>".to_vec();
    } else {
        match start {
            StructuredOutputStart::Immediate => {
                headers.push(Header::result(""));
                root = "result".into();
            }
            StructuredOutputStart::AfterDelimiter(closing) => {
                headers.push(Header::reasoning(""));
                reasoning_close = closing.as_bytes().to_vec();
                builder.reasoning("reasoning", closing);
                root = "reasoning result".into();
                allow_reasoning = true;
                required_wire.push(closing.clone());
            }
            StructuredOutputStart::AfterReasoningEnvelope {
                opening,
                closing,
                allow_reasoning: allow,
            } => {
                headers.push(if *allow {
                    Header::reasoning(opening)
                } else {
                    Header::result(&format!("{opening}{closing}"))
                });
                reasoning_close = closing.as_bytes().to_vec();
                let open = builder.wire(opening);
                let close = builder.wire(closing);
                max_initial = token_count(factory, opening)?;
                allow_reasoning = *allow;
                root = if *allow {
                    builder.reasoning("reasoning", closing);
                    format!("{open} reasoning result")
                } else {
                    format!("{open} {close} result")
                };
                required_wire.push(format!("{opening}{closing}"));
            }
            StructuredOutputStart::HarmonyFinal => {
                return Err(FerrumError::config(
                    "Harmony activation requires its declared output protocol",
                ))
            }
        }
    }
    // Deterministic framing conflicts fail before sampling. Dynamic stops in a
    // payload still terminate as incomplete under the engine's stop semantics.
    for wire in &required_wire {
        if let Some(stop) = stop_texts
            .iter()
            .find(|stop| !stop.is_empty() && wire.contains(stop.as_str()))
        {
            return Err(FerrumError::invalid_request(format!(
                "structured tool envelope conflicts with stop sequence {stop:?}"
            )));
        }
        for token in factory.tokenizer.encode(wire, false)? {
            if stop_ids.contains(&token.get()) && !native_terminals.contains(&token.get()) {
                return Err(FerrumError::invalid_request(format!(
                    "structured tool envelope conflicts with stop token {}",
                    token.get()
                )));
            }
        }
    }
    let close_count = if reasoning_close.is_empty() {
        0
    } else {
        token_count(
            factory,
            std::str::from_utf8(&reasoning_close).expect("protocol delimiter UTF-8"),
        )?
    };
    let budget = if !reasoning_close.is_empty() || protocol == ModelOutputProtocol::HarmonyGptOss {
        let mut budget = StructuredOutputBudgetPlan::automatic(
            max_tokens,
            (max_initial + close_count + max_following + usize::from(!native_terminals.is_empty()))
                .max(1),
        )?;
        if !allow_reasoning {
            budget.reasoning_token_limit = 0;
        }
        Some(budget)
    } else {
        None
    };
    builder.rules.insert(0, format!("start: {root}"));
    let mut grammars = vec![GrammarWithLexer {
        name: Some("root".into()),
        json_schema: None,
        lark_grammar: Some(builder.rules.join("\n")),
    }];
    grammars.extend(builder.schemas);
    Ok(Compiled {
        grammar: TopLevelGrammar {
            grammars,
            max_tokens: None,
        },
        plan: Plan {
            headers,
            following_headers,
            reasoning_close,
            wire_token_bytes: Arc::clone(&factory.wire_token_bytes),
            marker_bytes: builder.marker_bytes,
            boundary_controls: builder.boundary_controls,
            native_terminals,
            native: protocol == ModelOutputProtocol::HarmonyGptOss,
        },
        budget,
    })
}

fn token_count(factory: &StructuredOutputFactory, text: &str) -> Result<usize> {
    Ok(factory.tokenizer.encode(text, false)?.len())
}
