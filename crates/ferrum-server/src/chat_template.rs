use crate::openai::{
    ChatFunction, ChatMessage, ChatTool, FunctionCallChoice, MessageRole, ToolChoice,
};
use minijinja::{context, Environment};
use serde::Serialize;
use tracing::warn;

/// Model-provided chat template, usually from GGUF `tokenizer.chat_template`
/// or HuggingFace `tokenizer_config.json`.
#[derive(Clone, Debug)]
pub struct ModelChatTemplate {
    pub template: String,
    pub source: String,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

impl ModelChatTemplate {
    pub fn new(template: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            template: template.into(),
            source: source.into(),
            bos_token: None,
            eos_token: None,
        }
    }
}

/// Common prompt-message shape used by both CLI `run` and OpenAI `serve`.
#[derive(Clone, Debug, Serialize)]
pub struct PromptMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl PromptMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        let role = role.into();
        let content = content.into();
        if role == "assistant" {
            let (reasoning_content, content) = split_reasoning_content(content);
            return Self {
                role,
                content,
                reasoning_content,
            };
        }
        Self {
            role,
            content,
            reasoning_content: None,
        }
    }
}

fn split_reasoning_content(content: String) -> (Option<String>, String) {
    let Some(end_idx) = content.find("</think>") else {
        return (None, content);
    };
    let before_end = &content[..end_idx];
    let after_end = content[end_idx + "</think>".len()..]
        .trim_start_matches('\n')
        .to_string();
    let reasoning = before_end
        .rsplit_once("<think>")
        .map(|(_, reasoning)| reasoning)
        .unwrap_or(before_end)
        .trim_matches('\n')
        .to_string();
    (Some(reasoning), after_end)
}

/// Render common chat messages into the prompt string the model was trained
/// on. Prefer a model-provided chat template when available; otherwise use
/// a centralized legacy fallback for model families ferrum already supports.
pub fn render_prompt_messages(
    messages: &[PromptMessage],
    model_id: &str,
    model_template: Option<&ModelChatTemplate>,
) -> String {
    if let Some(model_template) = model_template {
        match render_model_template(messages, model_template) {
            Ok(prompt) if !prompt.trim().is_empty() => return prompt,
            Ok(_) => warn!(
                "model chat template rendered an empty prompt; falling back to legacy renderer"
            ),
            Err(e) => warn!(
                "failed to render model chat template from {}: {}; falling back to legacy renderer",
                model_template.source, e
            ),
        }
    }
    render_fallback_prompt(messages, model_id, None)
}

fn render_model_template(
    messages: &[PromptMessage],
    model_template: &ModelChatTemplate,
) -> std::result::Result<String, minijinja::Error> {
    let mut env = Environment::new();
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
    let template = normalize_hf_chat_template(&model_template.template);
    env.add_template("chat", &template)?;
    let tmpl = env.get_template("chat")?;
    tmpl.render(context! {
        messages => messages,
        add_generation_prompt => true,
        bos_token => model_template.bos_token.as_deref().unwrap_or(""),
        eos_token => model_template.eos_token.as_deref().unwrap_or(""),
    })
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
    render_chat_prompt_with_model_template(messages, model_id, None)
}

pub fn render_chat_prompt_with_model_template(
    messages: &[ChatMessage],
    model_id: &str,
    model_template: Option<&ModelChatTemplate>,
) -> String {
    let prompt_messages = messages
        .iter()
        .map(|msg| PromptMessage::new(template_role(msg), template_content(msg)))
        .collect::<Vec<_>>();
    render_prompt_messages(&prompt_messages, model_id, model_template)
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
        );
        assert_eq!(out, "[user]Hi[assistant]");
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
        );
        assert_eq!(
            out,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(!out.contains("<think>"));
    }

    #[test]
    fn assistant_think_history_exposes_reasoning_content_to_model_template() {
        let template = ModelChatTemplate::new(
            "{% for message in messages %}{% if message.reasoning_content is defined and message.reasoning_content is not none %}<r>{{ message.reasoning_content|trim_newlines }}</r>{{ message.content|trim_start_newlines }}{% else %}{{ message.content }}{% endif %}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}",
            "reasoning-template",
        );
        let out = render_prompt_messages(
            &[
                PromptMessage::new("assistant", "<think>\nreason\n</think>\n\nanswer"),
                PromptMessage::new("user", "next"),
            ],
            "qwen3",
            Some(&template),
        );
        assert_eq!(out, "<r>reason</r>answernext<assistant>");
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
            }],
            "qwen3",
            Some(&template),
        );
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
            }],
            "qwen3",
            Some(&template),
        );
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
        );
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
}
