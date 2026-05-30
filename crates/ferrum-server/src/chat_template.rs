use crate::openai::{ChatMessage, MessageRole};

/// Render OpenAI-style chat messages into the prompt string the model was
/// trained on. Mirrors the templates `ferrum-cli::commands::run` uses so
/// `/v1/chat/completions` produces the same behaviour as the interactive CLI.
///
/// Detects model family from the request's `model` field:
///   - qwen (Qwen2 / Qwen2.5 / Qwen3): ChatML with `<|im_start|>` / `<|im_end|>`
///     (Qwen3 adds the empty `<think></think>` marker to disable reasoning)
///   - llama 3: `<|start_header_id|>...<|end_header_id|>` + `<|eot_id|>`
///   - fallback: TinyLlama-style `<|system|>` / `<|user|>` / `<|assistant|>`
///     with `</s>` separators
///
/// All templates end with the assistant header so the first generated token
/// becomes the reply content (no extra role prefix).
pub(crate) fn render_chat_prompt(messages: &[ChatMessage], model_id: &str) -> String {
    let model_lower = model_id.to_lowercase();

    if model_lower.contains("qwen") {
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                template_role(msg),
                msg.content
            ));
        }
        prompt.push_str("<|im_start|>assistant\n");
        if model_lower.contains("qwen3") {
            // Qwen3: disable thinking mode by inserting an empty think block.
            prompt.push_str("<think>\n\n</think>\n\n");
        }
        prompt
    } else if model_lower.contains("llama") && model_lower.contains("3") {
        let mut prompt = String::from("<|begin_of_text|>");
        for msg in messages {
            prompt.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                template_role(msg),
                msg.content
            ));
        }
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        prompt
    } else {
        // TinyLlama / generic chat format. Promote the first system message
        // to the top; subsequent ones (rare) are emitted inline.
        let has_system = messages
            .iter()
            .any(|m| matches!(m.role, MessageRole::System));
        let mut prompt = String::new();
        if !has_system {
            prompt.push_str("<|system|>\nYou are a helpful assistant.</s>\n");
        }
        for msg in messages {
            let tag = match msg.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::Function => "function",
                MessageRole::Tool => "tool",
            };
            prompt.push_str(&format!("<|{}|>\n{}</s>\n", tag, msg.content));
        }
        prompt.push_str("<|assistant|>\n");
        prompt
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: MessageRole, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: content.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            function_call: None,
        }
    }

    #[test]
    fn qwen3_renders_chatml_with_think_marker() {
        let out = render_chat_prompt(
            &[
                msg(MessageRole::System, "You are helpful."),
                msg(MessageRole::User, "Hi"),
            ],
            "qwen3:0.6b",
        );
        assert!(out.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn qwen2_renders_chatml_without_think() {
        let out = render_chat_prompt(&[msg(MessageRole::User, "Hi")], "Qwen/Qwen2.5-7B-Instruct");
        assert!(out.ends_with("<|im_start|>assistant\n"));
        assert!(!out.contains("<think>"));
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
        assert!(out.starts_with("<|begin_of_text|>"));
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
}
