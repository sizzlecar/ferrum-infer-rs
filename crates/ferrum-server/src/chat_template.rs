use crate::openai::{
    AssistantMessagePhase, ChatFunction, ChatMessage, ChatTool, FunctionCallChoice, MessageRole,
    ToolChoice,
};
use ferrum_types::{
    has_unclosed_model_reasoning_block, model_reasoning_markers, ApiToolCallProtocol, FerrumError,
    ModelOutputProtocol,
};
use minijinja::Environment;
use serde::ser::SerializeStruct;
use serde::Serialize;
use serde_json::Value;
use std::fmt;
use std::str::FromStr;

/// Model-provided chat template, usually from GGUF `tokenizer.chat_template`
/// or HuggingFace `tokenizer_config.json`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelReasoningProtocol {
    None,
    PromptOpened,
    ModelGenerated,
}

#[derive(Clone, Debug)]
pub struct ModelChatTemplate {
    pub template: String,
    pub source: String,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub tool_call_protocol: ApiToolCallProtocol,
    pub output_protocol: ModelOutputProtocol,
    pub reasoning_protocol: ModelReasoningProtocol,
    pub reasoning_default_enabled: bool,
}

impl ModelChatTemplate {
    pub fn new(template: impl Into<String>, source: impl Into<String>) -> Self {
        let template = template.into();
        let mut model_template = Self {
            tool_call_protocol: tool_call_protocol_for_template(&template),
            output_protocol: ModelOutputProtocol::Text,
            template,
            source: source.into(),
            bos_token: None,
            eos_token: None,
            reasoning_protocol: ModelReasoningProtocol::None,
            reasoning_default_enabled: false,
        };
        let (reasoning_protocol, reasoning_default_enabled) =
            detect_model_reasoning_protocol(&model_template);
        model_template.reasoning_protocol = reasoning_protocol;
        model_template.reasoning_default_enabled = reasoning_default_enabled;
        model_template
    }

    pub fn reasoning_enabled(&self, requested: Option<bool>) -> bool {
        self.reasoning_protocol != ModelReasoningProtocol::None
            && requested.unwrap_or(self.reasoning_default_enabled)
    }

    /// Bind the resolved model capability after loading its unchanged template
    /// bytes, then probe reasoning with that protocol's actual delimiters.
    pub fn set_output_protocol(&mut self, protocol: ModelOutputProtocol) {
        self.output_protocol = protocol;
        let (reasoning_protocol, reasoning_default_enabled) = detect_model_reasoning_protocol(self);
        self.reasoning_protocol = reasoning_protocol;
        self.reasoning_default_enabled = reasoning_default_enabled;
    }
}

fn tool_call_protocol_for_template(template: &str) -> ApiToolCallProtocol {
    if template.contains("<tool_call>")
        && template.contains("<function=")
        && template.contains("<parameter=")
    {
        ApiToolCallProtocol::FunctionParameterXml
    } else {
        ApiToolCallProtocol::Json
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

impl ReasoningEffort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }
}

impl fmt::Display for ReasoningEffort {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ReasoningEffort {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "minimal" => Ok(Self::Minimal),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::XHigh),
            _ => Err(format!(
                "unsupported reasoning effort {value:?}; expected minimal, low, medium, high, or xhigh"
            )),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChatTemplateOptions {
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Clock seen by the template's `strftime_now` (Mistral-Small-3.2 and
    /// Llama-3.x inject "today's date" into the system prompt). `None` =
    /// local wall clock; golden tests pin the timestamp recorded at
    /// fixture-generation time so byte comparison survives the date
    /// changing.
    pub now_override: Option<chrono::NaiveDateTime>,
}

impl ChatTemplateOptions {
    pub fn default_for_template(_model_template: Option<&ModelChatTemplate>) -> Self {
        // Omission is a real third state: the model-owned template decides its
        // default. Only explicit product controls may force true or false.
        Self::default()
    }
}

/// Common prompt-message shape used by both CLI `run` and OpenAI `serve`.
#[derive(Clone, Debug)]
pub struct PromptMessage {
    pub role: String,
    pub content: String,
    pub reasoning_content: Option<String>,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<PromptToolCall>>,
    pub tool_call_id: Option<String>,
    pub function_call: Option<crate::openai::ChatFunctionCall>,
}

impl Serialize for PromptMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut len = 2;
        len += usize::from(self.reasoning_content.is_some());
        len += usize::from(self.name.is_some());
        len += usize::from(self.tool_calls.is_some());
        len += usize::from(self.tool_call_id.is_some());
        len += usize::from(self.function_call.is_some());
        let mut state = serializer.serialize_struct("PromptMessage", len)?;
        state.serialize_field("role", &self.role)?;
        let content = template_content_value(&self.content);
        state.serialize_field("content", &content)?;
        if let Some(reasoning_content) = &self.reasoning_content {
            state.serialize_field("reasoning_content", reasoning_content)?;
        }
        if let Some(name) = &self.name {
            state.serialize_field("name", name)?;
        }
        if let Some(tool_calls) = &self.tool_calls {
            state.serialize_field("tool_calls", tool_calls)?;
        }
        if let Some(tool_call_id) = &self.tool_call_id {
            state.serialize_field("tool_call_id", tool_call_id)?;
        }
        if let Some(function_call) = &self.function_call {
            state.serialize_field("function_call", function_call)?;
        }
        state.end()
    }
}

/// Tool-call shape exposed to model chat templates.
///
/// OpenAI's wire format serializes `function.arguments` as a JSON string, but
/// HuggingFace chat templates generally expect a parsed mapping so they can
/// apply `tojson`, `items`, and similar template operations. Keep that internal
/// shape separate from the API response type.
#[derive(Clone, Debug, Serialize)]
pub struct PromptToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: PromptFunctionCall,
}

#[derive(Clone, Debug, Serialize)]
pub struct PromptFunctionCall {
    pub name: String,
    pub arguments: Value,
}

impl From<&crate::openai::ChatToolCall> for PromptToolCall {
    fn from(call: &crate::openai::ChatToolCall) -> Self {
        Self {
            index: call.index,
            id: call.id.clone(),
            tool_type: call.tool_type.clone(),
            function: PromptFunctionCall {
                name: call.function.name.clone(),
                arguments: parse_template_arguments(&call.function.arguments),
            },
        }
    }
}

fn parse_template_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.to_string()))
}

fn template_content_value(content: &str) -> Value {
    Value::String(content.to_string())
}

impl PromptMessage {
    /// Content is passed to the chat template verbatim — including any
    /// `<think>...</think>` blocks in assistant history. Whether reasoning
    /// is kept or stripped from history is a per-template policy (DeepSeek
    /// strips it, Qwen3-Coder keeps it); pre-splitting here diverged from
    /// what `transformers.apply_chat_template` feeds the same template.
    /// `reasoning_content` is only set when the client supplies reasoning
    /// as a separate field.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            function_call: None,
        }
    }

    fn from_chat_message(message: &ChatMessage) -> Self {
        let mut prompt = Self::new(template_role(message), message.content.clone());
        if matches!(message.role, MessageRole::Assistant) && message.reasoning.is_some() {
            prompt.reasoning_content = message.reasoning.clone();
        }
        prompt.name = message.name.clone();
        prompt.tool_calls = message
            .tool_calls
            .as_ref()
            .map(|calls| calls.iter().map(PromptToolCall::from).collect());
        prompt.tool_call_id = message.tool_call_id.clone();
        prompt.function_call = message.function_call.clone();
        prompt
    }
}

/// Render common chat messages into the prompt string the model was trained
/// on. Prefer a model-provided chat template when available; otherwise use
/// a centralized legacy fallback for model families ferrum already supports.
///
/// A model-provided template that fails to render (or renders empty) is a
/// hard error: silently falling back to a generic prompt format feeds the
/// model a prompt it was not trained on and degrades output quality without
/// any visible failure.
pub fn render_prompt_messages(
    messages: &[PromptMessage],
    model_id: &str,
    model_template: Option<&ModelChatTemplate>,
) -> ferrum_types::Result<String> {
    render_prompt_messages_with_options(
        messages,
        model_id,
        model_template,
        &ChatTemplateOptions::default(),
    )
}

pub fn render_prompt_messages_with_options(
    messages: &[PromptMessage],
    model_id: &str,
    model_template: Option<&ModelChatTemplate>,
    options: &ChatTemplateOptions,
) -> ferrum_types::Result<String> {
    render_prompt_messages_with_options_and_compatibility(
        messages,
        model_id,
        model_template,
        options,
        true,
        &[],
    )
}

fn render_prompt_messages_with_options_and_compatibility(
    messages: &[PromptMessage],
    model_id: &str,
    model_template: Option<&ModelChatTemplate>,
    options: &ChatTemplateOptions,
    coalesce_interleaved_system_messages: bool,
    message_phases: &[Option<AssistantMessagePhase>],
) -> ferrum_types::Result<String> {
    if let Some(model_template) = model_template {
        return match render_model_template(
            messages,
            message_phases,
            model_template,
            options,
            coalesce_interleaved_system_messages,
            None,
            None,
            None,
            None,
        ) {
            Ok(prompt) if !prompt.trim().is_empty() => Ok(prompt),
            Ok(_) => Err(chat_template_render_error(
                model_template,
                "template rendered an empty prompt",
            )),
            Err(e) => Err(chat_template_render_error(model_template, e)),
        };
    }
    Ok(render_fallback_prompt(messages, model_id, None))
}

fn chat_template_render_error(
    template: &ModelChatTemplate,
    reason: impl std::fmt::Display,
) -> FerrumError {
    FerrumError::model(format!(
        "chat template from {} failed to render: {reason}. Refusing to fall back \
         to a generic prompt format because that silently degrades output quality; \
         fix the model's chat template or serve the model without one.",
        template.source
    ))
}

#[derive(Serialize)]
struct ModelTemplateContext<'a> {
    messages: Vec<Value>,
    add_generation_prompt: bool,
    bos_token: &'a str,
    eos_token: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<ReasoningEffort>,
    /// Pre-converted to `serde_json::Value`: minijinja serializes Rust
    /// *structs* with alphabetically sorted fields, but a JSON map (with
    /// serde_json `preserve_order`) keeps the OpenAI canonical key order
    /// that transformers' `tojson` renders.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    functions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<&'a FunctionCallChoice>,
}

fn model_template_message_values(
    messages: &[PromptMessage],
    phases: &[Option<AssistantMessagePhase>],
) -> std::result::Result<Vec<Value>, minijinja::Error> {
    if !phases.is_empty() && phases.len() != messages.len() {
        return Err(minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "assistant phase metadata did not match prompt messages",
        ));
    }
    messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            let mut value = serde_json::to_value(message).map_err(|error| {
                minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("failed to serialize prompt message: {error}"),
                )
            })?;
            if let Some(phase) = phases.get(index).copied().flatten() {
                value
                    .as_object_mut()
                    .expect("PromptMessage serializes as an object")
                    .insert("phase".to_string(), serde_json::json!(phase));
            }
            Ok(value)
        })
        .collect()
}

fn render_model_template(
    messages: &[PromptMessage],
    message_phases: &[Option<AssistantMessagePhase>],
    model_template: &ModelChatTemplate,
    options: &ChatTemplateOptions,
    coalesce_interleaved_system_messages: bool,
    tools: Option<&[ChatTool]>,
    tool_choice: Option<&ToolChoice>,
    functions: Option<&[ChatFunction]>,
    function_call: Option<&FunctionCallChoice>,
) -> std::result::Result<String, minijinja::Error> {
    let original = render_model_template_once(
        messages,
        message_phases,
        model_template,
        options,
        tools,
        tool_choice,
        functions,
        function_call,
    );
    if original.is_ok()
        || !coalesce_interleaved_system_messages
        || !has_nonleading_system_message(messages)
        || !original
            .as_ref()
            .err()
            .is_some_and(is_system_message_position_error)
    {
        return original;
    }

    // Preserve system messages in place for templates that support them. If a
    // model-owned template rejects that valid API history shape, retry with a
    // single leading system message while preserving system and conversation
    // order within their respective streams.
    let (adapted_messages, adapted_phases) = coalesce_system_messages(messages, message_phases);
    match render_model_template_once(
        &adapted_messages,
        &adapted_phases,
        model_template,
        options,
        tools,
        tool_choice,
        functions,
        function_call,
    ) {
        Ok(prompt) => Ok(prompt),
        Err(_) => original,
    }
}

fn is_system_message_position_error(error: &minijinja::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    let identifies_system_message =
        message.contains("system message") || message.contains("system role");
    let identifies_position = message.contains("beginning")
        || message.contains("must be first")
        || message.contains("must be the first")
        || message.contains("only be first")
        || message.contains("only be the first")
        || message.contains("must appear first")
        || message.contains("can only appear first")
        || message.contains("not allowed after")
        || message.contains("cannot appear after");
    identifies_system_message && identifies_position
}

fn has_nonleading_system_message(messages: &[PromptMessage]) -> bool {
    messages
        .iter()
        .enumerate()
        .any(|(index, message)| index > 0 && message.role == "system")
}

fn coalesce_system_messages(
    messages: &[PromptMessage],
    phases: &[Option<AssistantMessagePhase>],
) -> (Vec<PromptMessage>, Vec<Option<AssistantMessagePhase>>) {
    let mut first_system = None;
    let mut system_parts = Vec::new();
    let mut conversation = Vec::with_capacity(messages.len());
    let mut conversation_phases = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        if message.role == "system" {
            first_system.get_or_insert_with(|| message.clone());
            if !message.content.is_empty() {
                system_parts.push(message.content.clone());
            }
        } else {
            conversation.push(message.clone());
            conversation_phases.push(phases.get(index).copied().flatten());
        }
    }
    if let Some(mut system) = first_system {
        system.content = system_parts.join("\n\n");
        conversation.insert(0, system);
        conversation_phases.insert(0, None);
    }
    (conversation, conversation_phases)
}

fn render_model_template_once(
    messages: &[PromptMessage],
    message_phases: &[Option<AssistantMessagePhase>],
    model_template: &ModelChatTemplate,
    options: &ChatTemplateOptions,
    tools: Option<&[ChatTool]>,
    tool_choice: Option<&ToolChoice>,
    functions: Option<&[ChatFunction]>,
    function_call: Option<&FunctionCallChoice>,
) -> std::result::Result<String, minijinja::Error> {
    let mut env = Environment::new();
    // HF chat templates are written for Jinja2 and freely use Python string
    // methods (`.split()`, `.strip()`, `.startswith()`, ...). pycompat
    // resolves those at runtime; `normalize_hf_chat_template` below remains
    // for the exact spellings it already rewrote before pycompat landed.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    // transformers compiles chat templates with
    // `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`
    // and a plain-`json.dumps` `tojson` filter; match both so rendering is
    // byte-identical (verified by tests/chat_template_golden.rs).
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_filter("tojson", python_style_tojson);
    env.add_filter("trim_newlines", |s: String| {
        s.trim_matches('\n').to_string()
    });
    env.add_filter("trim_start_newlines", |s: String| {
        s.trim_start_matches('\n').to_string()
    });
    env.add_filter("trim_end_newlines", |s: String| {
        s.trim_end_matches('\n').to_string()
    });
    env.add_filter("starts_with", |s: String, prefix: String| {
        s.starts_with(&prefix)
    });
    env.add_filter("ends_with", |s: String, suffix: String| {
        s.ends_with(&suffix)
    });
    env.add_filter("after_think_end", |s: String| {
        s.split("</think>")
            .last()
            .unwrap_or("")
            .trim_start_matches('\n')
            .to_string()
    });
    env.add_filter("reasoning_from_think", |s: String| {
        s.split("</think>")
            .next()
            .unwrap_or("")
            .trim_end_matches('\n')
            .rsplit("<think>")
            .next()
            .unwrap_or("")
            .trim_start_matches('\n')
            .to_string()
    });
    // transformers exposes this helper to HuggingFace templates. Templates
    // use it to reject invalid message shapes with a useful model-authored
    // error instead of failing with an unrelated "unknown function" error.
    env.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    // transformers exposes `strftime_now(format)` = `datetime.now().strftime`
    // to templates (Mistral-Small-3.2 / Llama-3.x date their system prompts
    // with it). chrono's strftime covers the specifiers real templates use
    // (%d %m %Y %b %H %M %S); an unsupported one is a render error, not a
    // panic.
    let now = options
        .now_override
        .unwrap_or_else(|| chrono::Local::now().naive_local());
    env.add_function(
        "strftime_now",
        move |fmt: String| -> std::result::Result<String, minijinja::Error> {
            use std::fmt::Write as _;
            let mut out = String::new();
            write!(out, "{}", now.format(&fmt)).map_err(|_| {
                minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("strftime_now: unsupported format string {fmt:?}"),
                )
            })?;
            Ok(out)
        },
    );
    let template_messages = model_template_message_values(messages, message_phases)?;
    let template = normalize_hf_chat_template(&model_template.template);
    env.add_template("chat", &template)?;
    let tmpl = env.get_template("chat")?;
    tmpl.render(ModelTemplateContext {
        messages: template_messages,
        add_generation_prompt: true,
        bos_token: model_template.bos_token.as_deref().unwrap_or(""),
        eos_token: model_template.eos_token.as_deref().unwrap_or(""),
        enable_thinking: options.enable_thinking,
        reasoning_effort: options.reasoning_effort,
        tools: tools.and_then(|t| serde_json::to_value(t).ok()),
        tool_choice,
        functions: functions.and_then(|f| serde_json::to_value(f).ok()),
        function_call,
    })
}

fn detect_model_reasoning_protocol(
    model_template: &ModelChatTemplate,
) -> (ModelReasoningProtocol, bool) {
    let Some((opening, closing)) = model_reasoning_markers(model_template.output_protocol) else {
        return (ModelReasoningProtocol::None, false);
    };
    let messages = [PromptMessage {
        role: "user".to_string(),
        content: "reasoning protocol probe".to_string(),
        reasoning_content: None,
        name: None,
        tool_calls: None,
        tool_call_id: None,
        function_call: None,
    }];
    let now =
        chrono::NaiveDate::from_ymd_opt(2000, 1, 1).and_then(|date| date.and_hms_opt(0, 0, 0));
    let render = |enable_thinking| {
        render_model_template(
            &messages,
            &[],
            model_template,
            &ChatTemplateOptions {
                enable_thinking,
                reasoning_effort: None,
                now_override: now,
            },
            true,
            None,
            None,
            None,
            None,
        )
        .ok()
    };
    let Some(enabled) = render(Some(true)) else {
        return (ModelReasoningProtocol::None, false);
    };
    let default = render(None);
    let prompt_opened =
        |prompt: &str| has_unclosed_model_reasoning_block(model_template.output_protocol, prompt);
    if prompt_opened(&enabled) {
        let default_enabled = default.as_deref().is_some_and(prompt_opened);
        return (ModelReasoningProtocol::PromptOpened, default_enabled);
    }
    let Some(disabled) = render(Some(false)) else {
        return (ModelReasoningProtocol::None, false);
    };
    let completed_blocks = |prompt: &str| {
        prompt
            .matches(opening)
            .count()
            .min(prompt.matches(closing).count())
    };
    if completed_blocks(&disabled) > completed_blocks(&enabled) {
        return (
            ModelReasoningProtocol::ModelGenerated,
            default.as_deref() == Some(enabled.as_str()),
        );
    }
    (ModelReasoningProtocol::None, false)
}

/// `tojson` matching Python's `json.dumps(..., ensure_ascii=False)` as used
/// by transformers' chat-template environment: `", "` / `": "` separators and
/// insertion key order (hence the minijinja `preserve_order` feature).
/// minijinja's builtin emits compact separators, which breaks byte equality
/// with transformers-rendered tool definitions.
fn python_style_tojson(
    value: minijinja::value::Value,
    kwargs: minijinja::value::Kwargs,
) -> Result<String, minijinja::Error> {
    // transformers' `tojson` forwards kwargs to `json.dumps`; real templates
    // use `indent=N` (Llama-3.x tool specs). Python's indented output equals
    // serde_json's pretty formatter with an N-space indent. Anything else
    // (sort_keys, separators) is unimplemented — erroring beats silently
    // rendering a different prompt.
    let indent = kwargs.get::<Option<usize>>("indent")?;
    kwargs.assert_all_used()?;
    if let Some(indent) = indent {
        let indent_bytes = vec![b' '; indent];
        let mut out = Vec::new();
        let mut ser = serde_json::Serializer::with_formatter(
            &mut out,
            serde_json::ser::PrettyFormatter::with_indent(&indent_bytes),
        );
        serde::Serialize::serialize(&value, &mut ser).map_err(|e| {
            minijinja::Error::new(minijinja::ErrorKind::BadSerialization, e.to_string())
        })?;
        return String::from_utf8(out).map_err(|e| {
            minijinja::Error::new(minijinja::ErrorKind::BadSerialization, e.to_string())
        });
    }
    struct PyFormatter;
    impl serde_json::ser::Formatter for PyFormatter {
        fn begin_object_key<W: ?Sized + std::io::Write>(
            &mut self,
            writer: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if !first {
                writer.write_all(b", ")?;
            }
            Ok(())
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(
            &mut self,
            writer: &mut W,
        ) -> std::io::Result<()> {
            writer.write_all(b": ")
        }
        fn begin_array_value<W: ?Sized + std::io::Write>(
            &mut self,
            writer: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if !first {
                writer.write_all(b", ")?;
            }
            Ok(())
        }
    }

    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PyFormatter);
    serde::Serialize::serialize(&value, &mut ser).map_err(|e| {
        minijinja::Error::new(minijinja::ErrorKind::BadSerialization, e.to_string())
    })?;
    String::from_utf8(out)
        .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::BadSerialization, e.to_string()))
}

fn normalize_hf_chat_template(template: &str) -> String {
    template
        .replace(
            "message.content.split('</think>')[-1].lstrip('\\n')",
            "message.content|after_think_end",
        )
        .replace(
            "message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n')",
            "message.content|reasoning_from_think",
        )
        .replace(
            "content.split('</think>')[-1].lstrip('\\n')",
            "content|after_think_end",
        )
        .replace(
            "content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n')",
            "content|reasoning_from_think",
        )
        .replace(".startswith(", "|starts_with(")
        .replace(".endswith(", "|ends_with(")
        .replace(".strip('\\n')", "|trim_newlines")
        .replace(".lstrip('\\n')", "|trim_start_newlines")
        .replace(".rstrip('\\n')", "|trim_end_newlines")
}

/// Render OpenAI-style chat messages into the prompt string the model was
/// trained on.
///
/// Detects model family from the request's `model` field:
///   - qwen (Qwen2 / Qwen2.5 / Qwen3): ChatML with `<|im_start|>` / `<|im_end|>`
///   - llama 3: `<|start_header_id|>...<|end_header_id|>` + `<|eot_id|>`
///   - fallback: TinyLlama-style `<|system|>` / `<|user|>` / `<|assistant|>`
///     with `</s>` separators
///
/// All templates end with the assistant header so the first generated token
/// becomes the reply content (no extra role prefix).
pub fn render_chat_prompt(messages: &[ChatMessage], model_id: &str) -> String {
    let prompt_messages = messages
        .iter()
        .map(PromptMessage::from_chat_message)
        .collect::<Vec<_>>();
    render_fallback_prompt(&prompt_messages, model_id, None)
}

pub fn render_chat_prompt_with_model_template(
    messages: &[ChatMessage],
    model_id: &str,
    model_template: Option<&ModelChatTemplate>,
) -> ferrum_types::Result<String> {
    render_chat_prompt_with_model_template_options(
        messages,
        model_id,
        model_template,
        &ChatTemplateOptions::default(),
    )
}

pub fn render_chat_prompt_with_model_template_options(
    messages: &[ChatMessage],
    model_id: &str,
    model_template: Option<&ModelChatTemplate>,
    options: &ChatTemplateOptions,
) -> ferrum_types::Result<String> {
    render_chat_prompt_with_model_template_options_and_compatibility(
        messages,
        model_id,
        model_template,
        options,
        true,
        None,
    )
}

pub(crate) fn render_chat_prompt_with_model_template_options_and_compatibility(
    messages: &[ChatMessage],
    model_id: &str,
    model_template: Option<&ModelChatTemplate>,
    options: &ChatTemplateOptions,
    coalesce_interleaved_system_messages: bool,
    message_phases: Option<&[Option<AssistantMessagePhase>]>,
) -> ferrum_types::Result<String> {
    let prompt_messages = messages
        .iter()
        .map(PromptMessage::from_chat_message)
        .collect::<Vec<_>>();
    render_prompt_messages_with_options_and_compatibility(
        &prompt_messages,
        model_id,
        model_template,
        options,
        coalesce_interleaved_system_messages,
        message_phases.unwrap_or(&[]),
    )
}

fn render_fallback_prompt(
    messages: &[PromptMessage],
    model_id: &str,
    tool_spec: Option<String>,
) -> String {
    let model_lower = model_id.to_lowercase();

    if model_lower.contains("qwen") {
        let mut prompt = String::new();
        if let Some(tool_spec) = tool_spec {
            prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", tool_spec));
        }
        for msg in messages {
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, msg.content
            ));
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    } else if model_lower.contains("llama") && model_lower.contains("3") {
        // The engine encodes prompts with `add_special=true`, so do not
        // include `<|begin_of_text|>` here. Including it manually creates a
        // double-BOS prompt for Llama-3 tokenizers and degrades instruction
        // following.
        let mut prompt = String::new();
        if let Some(tool_spec) = tool_spec {
            prompt.push_str(&format!(
                "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                tool_spec
            ));
        }
        for msg in messages {
            prompt.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                msg.role, msg.content
            ));
        }
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        prompt
    } else {
        // TinyLlama / generic chat format. Promote the first system message
        // to the top; subsequent ones (rare) are emitted inline.
        let has_system = messages.iter().any(|m| m.role == "system");
        let mut prompt = String::new();
        if let Some(tool_spec) = tool_spec {
            prompt.push_str(&format!("<|system|>\n{}</s>\n", tool_spec));
        } else if !has_system {
            prompt.push_str("<|system|>\nYou are a helpful assistant.</s>\n");
        }
        for msg in messages {
            prompt.push_str(&format!("<|{}|>\n{}</s>\n", msg.role, msg.content));
        }
        prompt.push_str("<|assistant|>\n");
        prompt
    }
}

pub fn render_chat_prompt_with_tools(
    messages: &[ChatMessage],
    model_id: &str,
    tools: &[ChatTool],
    tool_choice: Option<&ToolChoice>,
    functions: &[ChatFunction],
    function_call: Option<&FunctionCallChoice>,
) -> String {
    let prompt_messages = messages
        .iter()
        .map(|msg| PromptMessage::new(template_role(msg), template_content(msg)))
        .collect::<Vec<_>>();
    render_fallback_prompt(
        &prompt_messages,
        model_id,
        render_tool_spec(tools, tool_choice, functions, function_call),
    )
}

pub fn render_chat_prompt_with_tools_and_model_template(
    messages: &[ChatMessage],
    model_id: &str,
    model_template: Option<&ModelChatTemplate>,
    options: &ChatTemplateOptions,
    tools: &[ChatTool],
    tool_choice: Option<&ToolChoice>,
    functions: &[ChatFunction],
    function_call: Option<&FunctionCallChoice>,
) -> ferrum_types::Result<String> {
    render_chat_prompt_with_tools_and_model_template_compatibility(
        messages,
        model_id,
        model_template,
        options,
        tools,
        tool_choice,
        functions,
        function_call,
        true,
        None,
    )
}

pub(crate) fn render_chat_prompt_with_tools_and_model_template_compatibility(
    messages: &[ChatMessage],
    model_id: &str,
    model_template: Option<&ModelChatTemplate>,
    options: &ChatTemplateOptions,
    tools: &[ChatTool],
    tool_choice: Option<&ToolChoice>,
    functions: &[ChatFunction],
    function_call: Option<&FunctionCallChoice>,
    coalesce_interleaved_system_messages: bool,
    message_phases: Option<&[Option<AssistantMessagePhase>]>,
) -> ferrum_types::Result<String> {
    if let Some(model_template) = model_template {
        if model_template_supports_tools(model_template) {
            let prompt_messages = messages
                .iter()
                .map(PromptMessage::from_chat_message)
                .collect::<Vec<_>>();
            return match render_model_template(
                &prompt_messages,
                message_phases.unwrap_or(&[]),
                model_template,
                options,
                coalesce_interleaved_system_messages,
                (!tools.is_empty()).then_some(tools),
                tool_choice,
                (!functions.is_empty()).then_some(functions),
                function_call,
            ) {
                Ok(prompt) if !prompt.trim().is_empty() => Ok(prompt),
                Ok(_) => Err(chat_template_render_error(
                    model_template,
                    "template rendered an empty prompt",
                )),
                Err(e) => Err(chat_template_render_error(model_template, e)),
            };
        }

        // The model ships a chat template with no `tools` support (e.g. the
        // DeepSeek-R1 distills). Inject the generic tool spec as a leading
        // system message and render it *through the model's own template*,
        // so tool definitions still reach the model in its native prompt
        // format instead of being silently dropped.
        let mut prompt_messages = Vec::with_capacity(messages.len() + 1);
        let tool_spec = render_tool_spec(tools, tool_choice, functions, function_call);
        if let Some(spec) = tool_spec.as_ref() {
            prompt_messages.push(PromptMessage::new("system", spec));
        }
        prompt_messages.extend(messages.iter().map(PromptMessage::from_chat_message));
        let prompt_phases = message_phases.map(|phases| {
            let mut prompt_phases = Vec::with_capacity(prompt_messages.len());
            if tool_spec.is_some() {
                prompt_phases.push(None);
            }
            prompt_phases.extend_from_slice(phases);
            prompt_phases
        });
        return match render_model_template(
            &prompt_messages,
            prompt_phases.as_deref().unwrap_or(&[]),
            model_template,
            options,
            coalesce_interleaved_system_messages,
            None,
            None,
            None,
            None,
        ) {
            Ok(prompt) if !prompt.trim().is_empty() => Ok(prompt),
            Ok(_) => Err(chat_template_render_error(
                model_template,
                "template rendered an empty prompt",
            )),
            Err(e) => Err(chat_template_render_error(model_template, e)),
        };
    }

    Ok(render_chat_prompt_with_tools(
        messages,
        model_id,
        tools,
        tool_choice,
        functions,
        function_call,
    ))
}

/// Whether a chat template references the `tools` variable as a standalone
/// identifier (substring matching alone would not distinguish a template
/// that only handles `message.tool_calls` history from one that renders
/// tool definitions).
fn model_template_supports_tools(template: &ModelChatTemplate) -> bool {
    let src = template.template.as_bytes();
    let needle = b"tools";
    let mut start = 0;
    while let Some(pos) = template.template[start..].find("tools") {
        let abs = start + pos;
        let before_ok = abs == 0 || {
            let c = src[abs - 1];
            !(c.is_ascii_alphanumeric() || c == b'_')
        };
        let after = abs + needle.len();
        let after_ok = after >= src.len() || {
            let c = src[after];
            !(c.is_ascii_alphanumeric() || c == b'_')
        };
        if before_ok && after_ok {
            return true;
        }
        start = after;
    }
    false
}

fn template_role(msg: &ChatMessage) -> &'static str {
    match msg.role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Function => "function",
        MessageRole::Tool => "tool",
    }
}

fn template_content(msg: &ChatMessage) -> String {
    let mut parts = Vec::new();
    if !msg.content.is_empty() {
        parts.push(msg.content.clone());
    }
    if let Some(tool_calls) = msg.tool_calls.as_deref().filter(|calls| !calls.is_empty()) {
        parts.push(json_line(serde_json::json!({ "tool_calls": tool_calls })));
    }
    if let Some(function_call) = msg.function_call.as_ref() {
        parts.push(json_line(
            serde_json::json!({ "function_call": function_call }),
        ));
    }
    parts.join("\n")
}

fn render_tool_spec(
    tools: &[ChatTool],
    tool_choice: Option<&ToolChoice>,
    functions: &[ChatFunction],
    function_call: Option<&FunctionCallChoice>,
) -> Option<String> {
    if tools.is_empty() && functions.is_empty() {
        return None;
    }

    let mut spec = serde_json::Map::new();
    spec.insert(
        "instruction".to_string(),
        serde_json::Value::String(
            "When a tool is needed, respond with JSON matching the provided tool/function schema; otherwise answer normally."
                .to_string(),
        ),
    );
    if !tools.is_empty() {
        spec.insert("tools".to_string(), serde_json::json!(tools));
    }
    if let Some(choice) = tool_choice {
        spec.insert("tool_choice".to_string(), serde_json::json!(choice));
    }
    if !functions.is_empty() {
        spec.insert("functions".to_string(), serde_json::json!(functions));
    }
    if let Some(choice) = function_call {
        spec.insert("function_call".to_string(), serde_json::json!(choice));
    }
    Some(json_line(serde_json::Value::Object(spec)))
}

fn json_line(value: serde_json::Value) -> String {
    serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: MessageRole, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: content.to_string(),
            reasoning: None,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            function_call: None,
        }
    }

    fn tool(name: &str) -> ChatTool {
        ChatTool {
            tool_type: "function".to_string(),
            function: ChatFunction {
                name: name.to_string(),
                description: Some("Get weather".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                })),
                strict: None,
            },
        }
    }

    #[test]
    fn qwen3_renders_chatml_without_forced_think_marker() {
        let out = render_chat_prompt(
            &[
                msg(MessageRole::System, "You are helpful."),
                msg(MessageRole::User, "Hi"),
            ],
            "qwen3:0.6b",
        );
        assert!(out.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
        assert!(!out.contains("<think>"));
    }

    #[test]
    fn qwen2_renders_chatml_without_think() {
        let out = render_chat_prompt(&[msg(MessageRole::User, "Hi")], "Qwen/Qwen2.5-7B-Instruct");
        assert!(out.ends_with("<|im_start|>assistant\n"));
        assert!(!out.contains("<think>"));
    }

    #[test]
    fn model_template_is_preferred_over_family_fallback() {
        let template = ModelChatTemplate::new(
            "{% for message in messages %}[{{ message.role }}]{{ message.content }}{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}",
            "test-template",
        );
        let out = render_chat_prompt_with_model_template(
            &[msg(MessageRole::User, "Hi")],
            "qwen3",
            Some(&template),
        )
        .unwrap();
        assert_eq!(out, "[user]Hi[assistant]");
    }

    #[test]
    fn model_template_is_used_for_tool_requests() {
        let template = ModelChatTemplate::new(
            "{% if tools %}<tools>{% for tool in tools %}{{ tool.function.name }}{% endfor %}</tools>{% endif %}{% for message in messages %}[{{ message.role }}]{{ message.content }}{% if message.tool_calls %}{% for tool_call in message.tool_calls %}<tool_call>{{ tool_call.function.name }}:{{ tool_call.function.arguments }}</tool_call>{% endfor %}{% endif %}{% if message.tool_call_id %}<tool_response id=\"{{ message.tool_call_id }}\">{{ message.content }}</tool_response>{% endif %}{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}",
            "tool-template",
        );
        let mut assistant = msg(MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![crate::openai::ChatToolCall {
            index: None,
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: crate::openai::ChatFunctionCall {
                name: "weather".to_string(),
                arguments: "{\"city\":\"Paris\"}".to_string(),
            },
        }]);
        let mut tool_result = msg(MessageRole::Tool, "sunny");
        tool_result.tool_call_id = Some("call_1".to_string());

        let out = render_chat_prompt_with_tools_and_model_template(
            &[
                msg(MessageRole::User, "Use weather."),
                assistant,
                tool_result,
            ],
            "served-hash-id",
            Some(&template),
            &ChatTemplateOptions::default(),
            &[tool("weather")],
            Some(&ToolChoice::Mode("auto".to_string())),
            &[],
            None,
        )
        .unwrap();

        assert!(out.contains("<tools>weather</tools>"));
        assert!(out.contains("<tool_call>weather:"), "{out}");
        assert!(out.contains("\"city\""), "{out}");
        assert!(out.contains("Paris"), "{out}");
        assert!(out.contains("<tool_response id=\"call_1\">sunny</tool_response>"));
        assert!(out.ends_with("[assistant]"));
        assert!(
            !out.contains("<|assistant|>"),
            "tool requests with model templates must not use generic fallback: {out}"
        );
    }

    #[test]
    fn model_template_tools_supports_qwen3_template_primitives() {
        let template = ModelChatTemplate::new(
            "{% if tools %}<tools>{% for tool in tools %}{{ tool | tojson }}{% endfor %}</tools>{% endif %}{% for message in messages[::-1] %}[{{ message.role }}]{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}",
            "qwen3-tool-primitives",
        );
        let out = render_chat_prompt_with_tools_and_model_template(
            &[
                msg(MessageRole::User, "Use weather."),
                msg(MessageRole::Assistant, "ok"),
            ],
            "served-hash-id",
            Some(&template),
            &ChatTemplateOptions::default(),
            &[tool("weather")],
            Some(&ToolChoice::Mode("auto".to_string())),
            &[],
            None,
        )
        .unwrap();

        // tojson renders Python-json.dumps style (", " / ": " separators,
        // insertion key order) for byte parity with transformers.
        assert!(out.contains("\"name\": \"weather\""), "{out}");
        assert!(
            out.contains("{\"type\": \"function\", \"function\": {"),
            "{out}"
        );
        assert!(out.contains("[assistant][user][assistant]"), "{out}");
    }

    #[test]
    fn model_template_tool_arguments_are_parsed_for_hf_templates() {
        let template = ModelChatTemplate::new(
            "{% for message in messages %}{% if message.tool_calls %}{% set tool_call = message.tool_calls[0].function %}{{ tool_call.arguments | tojson }}{% for name, value in tool_call.arguments | items %}[{{ name }}={{ value }}]{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}",
            "llama-tool-primitives",
        );
        let mut assistant = msg(MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![crate::openai::ChatToolCall {
            index: None,
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: crate::openai::ChatFunctionCall {
                name: "weather".to_string(),
                arguments: "{\"city\":\"Paris\",\"unit\":\"celsius\"}".to_string(),
            },
        }]);

        let out = render_chat_prompt_with_tools_and_model_template(
            &[msg(MessageRole::User, "Use weather."), assistant],
            "served-hash-id",
            Some(&template),
            &ChatTemplateOptions::default(),
            &[tool("weather")],
            Some(&ToolChoice::Mode("auto".to_string())),
            &[],
            None,
        )
        .unwrap();

        assert!(out.contains("\"city\""), "{out}");
        assert!(out.contains("\"Paris\""), "{out}");
        assert!(out.contains("[city=Paris]"), "{out}");
        assert!(out.contains("[unit=celsius]"), "{out}");
        assert!(out.ends_with("[assistant]"));
    }

    #[test]
    fn model_template_tool_result_content_stays_string_for_hf_templates() {
        let template = ModelChatTemplate::new(
            "{% for message in messages %}{% if message.role == 'tool' %}{% if message.content is string %}<tool_response>{{ message.content }}</tool_response>{% else %}not-string{% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}",
            "qwen-tool-result-primitives",
        );
        let mut tool_result = msg(
            MessageRole::Tool,
            "{\"city\":\"北京\",\"temp\":22,\"desc\":\"晴\"}",
        );
        tool_result.tool_call_id = Some("call_1".to_string());

        let out = render_chat_prompt_with_tools_and_model_template(
            &[msg(MessageRole::User, "Use weather."), tool_result],
            "served-hash-id",
            Some(&template),
            &ChatTemplateOptions::default(),
            &[tool("weather")],
            Some(&ToolChoice::Mode("auto".to_string())),
            &[],
            None,
        )
        .unwrap();

        assert!(out.contains("\"temp\""), "{out}");
        assert!(out.contains("22"), "{out}");
        assert!(out.contains("\"desc\":\"晴\""), "{out}");
        assert!(out.contains("<tool_response>"), "{out}");
        assert!(!out.contains("not-string"), "{out}");
        assert!(out.ends_with("[assistant]"));
    }

    #[test]
    fn qwen_style_model_template_does_not_force_empty_think() {
        let template = ModelChatTemplate::new(
            "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}",
            "qwen-template",
        );
        let out = render_chat_prompt_with_model_template(
            &[msg(MessageRole::User, "Hi")],
            "qwen3",
            Some(&template),
        )
        .unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(!out.contains("<think>"));
    }

    #[test]
    fn omitted_thinking_option_preserves_model_template_default() {
        let template = ModelChatTemplate::new(
            "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- else %}{{- '<think>\n' }}{%- endif %}{%- endif %}",
            "thinking-template",
        );
        let options = ChatTemplateOptions::default_for_template(Some(&template));
        assert_eq!(options.enable_thinking, None);
        let out = render_chat_prompt_with_model_template_options(
            &[msg(MessageRole::User, "Hi")],
            "served-model-alias",
            Some(&template),
            &options,
        )
        .unwrap();
        assert!(out.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn typed_reasoning_effort_is_exposed_to_model_template() {
        let template = ModelChatTemplate::new(
            "{% if reasoning_effort is defined %}Reasoning: {{ reasoning_effort }}{% else %}Reasoning: model-default{% endif %}",
            "reasoning-effort-template",
        );
        let default_prompt = render_chat_prompt_with_model_template_options(
            &[msg(MessageRole::User, "Hi")],
            "served-model-alias",
            Some(&template),
            &ChatTemplateOptions::default(),
        )
        .unwrap();
        assert_eq!(default_prompt, "Reasoning: model-default");

        let low_prompt = render_chat_prompt_with_model_template_options(
            &[msg(MessageRole::User, "Hi")],
            "served-model-alias",
            Some(&template),
            &ChatTemplateOptions {
                reasoning_effort: Some(ReasoningEffort::Low),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(low_prompt, "Reasoning: low");
    }

    #[test]
    fn explicit_thinking_options_override_model_template_default() {
        let template = ModelChatTemplate::new(
            "{% if add_generation_prompt %}<assistant>{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}{% endif %}",
            "thinking-template",
        );
        let enabled = render_chat_prompt_with_model_template_options(
            &[msg(MessageRole::User, "Hi")],
            "Qwen/Qwen3-0.6B",
            Some(&template),
            &ChatTemplateOptions {
                enable_thinking: Some(true),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(enabled, "<assistant><think>\n");

        let disabled = render_chat_prompt_with_model_template_options(
            &[msg(MessageRole::User, "Hi")],
            "Qwen/Qwen3-0.6B",
            Some(&template),
            &ChatTemplateOptions {
                enable_thinking: Some(false),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(disabled, "<assistant><think>\n\n</think>\n\n");
    }

    #[test]
    fn qwen3_model_generated_thinking_is_a_typed_template_capability() {
        let template = ModelChatTemplate::new(
            "{% if add_generation_prompt %}<assistant>{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}",
            "qwen3-model-generated-thinking-template",
        );

        assert_eq!(
            template.reasoning_protocol,
            ModelReasoningProtocol::ModelGenerated
        );
        assert!(template.reasoning_default_enabled);
        assert!(template.reasoning_enabled(None));
        assert!(template.reasoning_enabled(Some(true)));
        assert!(!template.reasoning_enabled(Some(false)));
    }

    #[test]
    fn declared_gemma_protocol_recomputes_template_reasoning_capability() {
        // Preserve the canonical template's generation-tail decisions without
        // copying its tool-schema formatting or other unrelated metadata.
        let source = concat!(
            "{% set enable_thinking = enable_thinking | default(false) %}",
            "{% if enable_thinking %}<|turn>system\n<|think|>\n<turn|>\n{% endif %}",
            "{% if add_generation_prompt %}",
            "{% if messages[-1].role == 'tool' %}",
            "<|turn>model\n<|tool_response>{{ messages[-1].content }}<tool_response|>",
            "{% if enable_thinking %}<|channel>thought\n{% endif %}",
            "{% else %}<|turn>model\n",
            "{% if not enable_thinking %}<|channel>thought\n<channel|>{% endif %}",
            "{% endif %}{% endif %}",
        );
        let mut template = ModelChatTemplate::new(source, "declared-thought-template");
        assert_eq!(template.reasoning_protocol, ModelReasoningProtocol::None);
        template.set_output_protocol(ModelOutputProtocol::GemmaThought);
        assert_eq!(template.template, source);
        assert_eq!(
            template.reasoning_protocol,
            ModelReasoningProtocol::ModelGenerated
        );
        assert!(!template.reasoning_default_enabled);
        assert!(!template.reasoning_enabled(None));
        assert!(!template.reasoning_enabled(Some(false)));
        assert!(template.reasoning_enabled(Some(true)));

        for (role, enabled, expected_tail, opened) in [
            (
                MessageRole::User,
                false,
                "<|turn>model\n<|channel>thought\n<channel|>",
                false,
            ),
            (MessageRole::User, true, "<|turn>model\n", false),
            (
                MessageRole::Tool,
                false,
                "<|tool_response>579<tool_response|>",
                false,
            ),
            (
                MessageRole::Tool,
                true,
                "<|tool_response>579<tool_response|><|channel>thought\n",
                true,
            ),
        ] {
            let prompt = render_chat_prompt_with_model_template_options(
                &[msg(role, "579")],
                "loaded-model-alias",
                Some(&template),
                &ChatTemplateOptions {
                    enable_thinking: Some(enabled),
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(prompt.ends_with(expected_tail), "{prompt:?}");
            assert_eq!(
                has_unclosed_model_reasoning_block(template.output_protocol, &prompt),
                opened
            );
        }

        template.set_output_protocol(ModelOutputProtocol::Text);
        assert_eq!(template.reasoning_protocol, ModelReasoningProtocol::None);
        assert!(!template.reasoning_enabled(Some(true)));
    }

    #[test]
    fn template_without_enable_thinking_does_not_get_thinking_default() {
        let template = ModelChatTemplate::new(
            "{% if add_generation_prompt %}<assistant>{% endif %}",
            "plain-template",
        );
        let options = ChatTemplateOptions::default_for_template(Some(&template));
        assert_eq!(options.enable_thinking, None);
        let out = render_chat_prompt_with_model_template_options(
            &[msg(MessageRole::User, "Hi")],
            "Qwen/Qwen3-0.6B",
            Some(&template),
            &options,
        )
        .unwrap();
        assert_eq!(out, "<assistant>");
    }

    #[test]
    fn assistant_think_history_exposes_reasoning_content_to_model_template() {
        let template = ModelChatTemplate::new(
            "{% for message in messages %}{% if message.reasoning_content is defined and message.reasoning_content is not none %}<r>{{ message.reasoning_content|trim_newlines }}</r>{{ message.content|trim_start_newlines }}{% else %}{{ message.content }}{% endif %}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}",
            "reasoning-template",
        );
        // reasoning_content is only present when the client supplied it
        // separately (OpenAI `message.reasoning`); raw `<think>` blocks in
        // content are the template's business (covered by golden tests).
        let mut assistant = PromptMessage::new("assistant", "answer");
        assistant.reasoning_content = Some("reason".to_string());
        let out = render_prompt_messages(
            &[assistant, PromptMessage::new("user", "next")],
            "qwen3",
            Some(&template),
        )
        .unwrap();
        assert_eq!(out, "<r>reason</r>answernext<assistant>");
    }

    #[test]
    fn assistant_phase_is_visible_to_model_template() {
        let template = ModelChatTemplate::new(
            "{% for message in messages %}[{{ message.role }}{% if message.phase is defined %}:{{ message.phase }}{% endif %}]{{ message.content }}{% endfor %}",
            "phase-template",
        );
        let messages = [
            msg(MessageRole::Assistant, "Still working"),
            msg(MessageRole::User, "Continue"),
        ];
        let phases = [Some(AssistantMessagePhase::Commentary), None];
        let out = render_chat_prompt_with_model_template_options_and_compatibility(
            &messages,
            "local-model",
            Some(&template),
            &ChatTemplateOptions::default(),
            true,
            Some(&phases),
        )
        .unwrap();
        assert_eq!(out, "[assistant:commentary]Still working[user]Continue");
    }

    #[test]
    fn hf_python_split_expressions_are_normalized_for_minijinja() {
        let template = ModelChatTemplate::new(
            "{% for message in messages %}{% set content = message.content.split('</think>')[-1].lstrip('\\n') %}{% set reasoning_content = message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}<r>{{ reasoning_content.strip('\\n') }}</r>{{ content.lstrip('\\n') }}{% endfor %}",
            "split-template",
        );
        let out = render_prompt_messages(
            &[PromptMessage {
                role: "assistant".to_string(),
                content: "<think>\nreason\n</think>\n\nanswer".to_string(),
                reasoning_content: None,
                name: None,
                tool_calls: None,
                tool_call_id: None,
                function_call: None,
            }],
            "qwen3",
            Some(&template),
        )
        .unwrap();
        assert_eq!(out, "<r>reason</r>answer");
    }

    #[test]
    fn qwen3_content_variable_split_expressions_are_normalized_for_minijinja() {
        let template = ModelChatTemplate::new(
            "{% for message in messages %}{% set content = message.content %}{% if '</think>' in content %}{% set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}{% set content = content.split('</think>')[-1].lstrip('\\n') %}{% endif %}<r>{{ reasoning_content.strip('\\n') }}</r>{{ content.lstrip('\\n') }}{% endfor %}",
            "qwen3-content-split-template",
        );
        let out = render_prompt_messages(
            &[PromptMessage {
                role: "assistant".to_string(),
                content: "<think>\nreason\n</think>\n\nanswer".to_string(),
                reasoning_content: None,
                name: None,
                tool_calls: None,
                tool_call_id: None,
                function_call: None,
            }],
            "qwen3",
            Some(&template),
        )
        .unwrap();
        assert_eq!(out, "<r>reason</r>answer");
    }

    #[test]
    fn qwen3_python_startswith_endswith_are_normalized_for_minijinja() {
        let template = ModelChatTemplate::new(
            "{% for message in messages %}{% if message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}plain{% else %}tool{% endif %}{% endfor %}",
            "qwen3-startswith-template",
        );
        let out = render_prompt_messages(
            &[
                PromptMessage::new("user", "hello"),
                PromptMessage::new("user", "<tool_response>ok</tool_response>"),
            ],
            "qwen3",
            Some(&template),
        )
        .unwrap();
        assert_eq!(out, "plaintool");
    }

    #[test]
    fn multi_turn_preserves_order() {
        let out = render_chat_prompt(
            &[
                msg(MessageRole::User, "A"),
                msg(MessageRole::Assistant, "B"),
                msg(MessageRole::User, "C"),
            ],
            "qwen3",
        );
        let a_idx = out.find("A").unwrap();
        let b_idx = out.find("B").unwrap();
        let c_idx = out.find("C").unwrap();
        assert!(a_idx < b_idx && b_idx < c_idx);
    }

    #[test]
    fn llama3_renders_header_format() {
        let out = render_chat_prompt(
            &[
                msg(MessageRole::System, "sys"),
                msg(MessageRole::User, "hi"),
            ],
            "meta-llama/Llama-3.2-1B-Instruct",
        );
        assert!(!out.starts_with("<|begin_of_text|>"));
        assert!(out.contains("<|start_header_id|>system<|end_header_id|>\n\nsys<|eot_id|>"));
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"));
        assert!(out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn unknown_model_uses_tinyllama_fallback() {
        let out = render_chat_prompt(&[msg(MessageRole::User, "hi")], "mystery-model");
        assert!(out.contains("<|system|>"));
        assert!(out.contains("<|user|>\nhi</s>"));
        assert!(out.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn fallback_preserves_legacy_function_and_tool_roles() {
        let out = render_chat_prompt(
            &[
                msg(MessageRole::Function, "{\"city\":\"Paris\"}"),
                msg(MessageRole::Tool, "sunny"),
            ],
            "mystery-model",
        );
        assert!(out.contains("<|function|>\n{\"city\":\"Paris\"}</s>"));
        assert!(out.contains("<|tool|>\nsunny</s>"));
    }

    #[test]
    fn qwen_renders_tool_definitions_and_assistant_tool_call_history() {
        let mut assistant = msg(MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![crate::openai::ChatToolCall {
            index: None,
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: crate::openai::ChatFunctionCall {
                name: "weather".to_string(),
                arguments: "{\"city\":\"Paris\"}".to_string(),
            },
        }]);

        let out = render_chat_prompt_with_tools(
            &[
                msg(MessageRole::User, "Use weather."),
                assistant,
                msg(MessageRole::Tool, "sunny"),
            ],
            "qwen3",
            &[tool("weather")],
            Some(&ToolChoice::Mode("auto".to_string())),
            &[],
            None,
        );

        assert!(out.contains("\"tools\":[{"));
        assert!(out.contains("\"type\":\"function\""));
        assert!(out.contains("\"tool_choice\":\"auto\""));
        assert!(out.contains("<|im_start|>assistant\n{"));
        assert!(out.contains("\"tool_calls\":[{"));
        assert!(out.contains("\"id\":\"call_1\""));
        assert!(out.contains("\"name\":\"weather\""));
        assert!(out.contains("<|im_start|>tool\nsunny<|im_end|>"));
    }

    #[test]
    fn pycompat_python_string_methods_render_without_normalization() {
        // Bracket subscripts plus bare `.strip()` / `.split(..)[-1]` are
        // spellings `normalize_hf_chat_template` does not rewrite — they must
        // work via minijinja-contrib pycompat (DeepSeek-R1 distill templates
        // use them).
        let template = ModelChatTemplate::new(
            "{% for message in messages %}{% if message['role'] == 'assistant' %}{% set content = message['content'].split('</think>')[-1] %}{{ content.strip() }}{% endif %}{% endfor %}",
            "pycompat-template",
        );
        let out = render_prompt_messages(
            &[PromptMessage {
                role: "assistant".to_string(),
                content: "<think>\nreason\n</think>\n\nanswer".to_string(),
                reasoning_content: None,
                name: None,
                tool_calls: None,
                tool_call_id: None,
                function_call: None,
            }],
            "deepseek-distill",
            Some(&template),
        )
        .unwrap();
        assert_eq!(out, "answer");
    }

    #[test]
    fn model_template_render_failure_is_an_error_not_a_silent_fallback() {
        let template =
            ModelChatTemplate::new("{{ messages | not_a_real_filter }}", "broken-template");
        let err = render_prompt_messages(
            &[PromptMessage::new("user", "hi")],
            "qwen3",
            Some(&template),
        )
        .unwrap_err();
        let message = format!("{err}");
        assert!(message.contains("broken-template"), "{message}");
        assert!(message.contains("failed to render"), "{message}");
    }

    #[test]
    fn hf_raise_exception_reports_the_template_error() {
        let template = ModelChatTemplate::new(
            "{{ raise_exception('System message must be at the beginning.') }}",
            "strict-template",
        );
        let err = render_prompt_messages(
            &[PromptMessage::new("user", "hi")],
            "qwen3",
            Some(&template),
        )
        .unwrap_err();
        let message = format!("{err}");
        assert!(
            message.contains("System message must be at the beginning."),
            "{message}"
        );
        assert!(!message.contains("unknown function"), "{message}");
    }

    #[test]
    fn permissive_template_preserves_interleaved_system_position() {
        let template = ModelChatTemplate::new(
            "{% for message in messages %}[{{ message.role }}]{{ message.content }}{% endfor %}",
            "system-in-place-template",
        );
        let out = render_prompt_messages(
            &[
                PromptMessage::new("system", "Initial"),
                PromptMessage::new("user", "Question"),
                PromptMessage::new("system", "Deferred"),
            ],
            "model-with-system-in-place",
            Some(&template),
        )
        .unwrap();
        assert_eq!(out, "[system]Initial[user]Question[system]Deferred");
    }

    #[test]
    fn interleaved_system_coalescing_can_be_disabled() {
        let template = ModelChatTemplate::new(
            "{% for message in messages %}{% if message.role == 'system' and not loop.first %}{{ raise_exception('System message must be at the beginning.') }}{% endif %}[{{ message.role }}]{{ message.content }}{% endfor %}",
            "strict-leading-system-template",
        );
        let error = render_prompt_messages_with_options_and_compatibility(
            &[
                PromptMessage::new("system", "Initial"),
                PromptMessage::new("user", "Question"),
                PromptMessage::new("system", "Deferred"),
            ],
            "strict-model",
            Some(&template),
            &ChatTemplateOptions::default(),
            false,
            &[],
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("System message must be at the beginning."),
            "{error}"
        );
    }

    #[test]
    fn strict_template_coalesces_consecutive_leading_system_messages() {
        let template = ModelChatTemplate::new(
            "{% for message in messages %}{% if message.role == 'system' and not loop.first %}{{ raise_exception('System message must be the first message.') }}{% endif %}[{{ message.role }}]{{ message.content }}{% endfor %}",
            "strict-single-system-template",
        );
        let out = render_prompt_messages(
            &[
                PromptMessage::new("system", "Initial"),
                PromptMessage::new("system", "Additional"),
                PromptMessage::new("user", "Question"),
            ],
            "strict-model",
            Some(&template),
        )
        .unwrap();
        assert_eq!(out, "[system]Initial\n\nAdditional[user]Question");
    }

    #[test]
    fn interleaved_system_retry_does_not_hide_an_unrelated_template_error() {
        let template = ModelChatTemplate::new(
            "{% if messages|length == 3 %}{{ raise_exception('tool schema rejected') }}{% endif %}ok",
            "unrelated-error-template",
        );
        let error = render_prompt_messages(
            &[
                PromptMessage::new("system", "Initial"),
                PromptMessage::new("user", "Question"),
                PromptMessage::new("system", "Deferred"),
            ],
            "strict-model",
            Some(&template),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("tool schema rejected"),
            "{error}"
        );
    }

    #[test]
    fn model_template_empty_render_is_an_error() {
        let template = ModelChatTemplate::new("{# renders nothing #}", "empty-template");
        let err = render_prompt_messages(
            &[PromptMessage::new("user", "hi")],
            "qwen3",
            Some(&template),
        )
        .unwrap_err();
        assert!(format!("{err}").contains("empty prompt"), "{err}");
    }

    #[test]
    fn tools_unaware_template_injects_tool_spec_through_model_template() {
        // e.g. DeepSeek-R1 distill templates have no `tools` support; tool
        // definitions must still reach the model in its native prompt format
        // instead of being dropped or routed to the generic fallback.
        let template = ModelChatTemplate::new(
            "{% for message in messages %}[{{ message.role }}]{{ message.content }}{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}",
            "no-tool-support-template",
        );
        let out = render_chat_prompt_with_tools_and_model_template(
            &[msg(MessageRole::User, "Use weather.")],
            "some-model",
            Some(&template),
            &ChatTemplateOptions::default(),
            &[tool("weather")],
            Some(&ToolChoice::Mode("auto".to_string())),
            &[],
            None,
        )
        .unwrap();
        assert!(out.starts_with("[system]"), "{out}");
        assert!(out.contains("\"tools\""), "{out}");
        assert!(out.contains("weather"), "{out}");
        assert!(out.ends_with("[assistant]"), "{out}");
        assert!(!out.contains("<|system|>"), "{out}");
    }

    #[test]
    fn template_tools_support_detection_requires_standalone_identifier() {
        let aware = ModelChatTemplate::new("{% if tools %}x{% endif %}", "t");
        assert!(model_template_supports_tools(&aware));
        let history_only = ModelChatTemplate::new(
            "{% for m in messages %}{% if m.tool_calls %}y{% endif %}{% endfor %}",
            "t",
        );
        assert!(!model_template_supports_tools(&history_only));
    }
}
