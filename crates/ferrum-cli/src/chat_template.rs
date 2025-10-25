//! Chat template formatting for different models

/// Format prompt using TinyLlama/ChatML template
pub fn format_tinyllama_prompt(messages: &[(String, String)], user_message: &str) -> String {
    let mut formatted = String::from("<|system|>\nYou are a friendly chatbot.</s>\n");

    // Add conversation history
    for (role, content) in messages {
        if role == "user" {
            formatted.push_str(&format!("<|user|>\n{}</s>\n", content));
        } else {
            formatted.push_str(&format!("<|assistant|>\n{}</s>\n", content));
        }
    }

    // Add current user message
    formatted.push_str(&format!("<|user|>\n{}</s>\n<|assistant|>\n", user_message));

    formatted
}

/// Format prompt for Llama models
pub fn format_llama_prompt(messages: &[(String, String)], user_message: &str) -> String {
    let mut formatted = String::from("[INST] ");

    for (role, content) in messages {
        if role == "user" {
            formatted.push_str(&format!("{} [/INST] ", content));
        } else {
            formatted.push_str(&format!("{} [INST] ", content));
        }
    }

    formatted.push_str(&format!("{} [/INST]", user_message));
    formatted
}

/// Auto-detect and format prompt based on model name
pub fn auto_format_prompt(
    model_name: &str,
    messages: &[(String, String)],
    user_message: &str,
) -> String {
    if model_name.to_lowercase().contains("tinyllama") {
        format_tinyllama_prompt(messages, user_message)
    } else if model_name.to_lowercase().contains("llama") {
        format_llama_prompt(messages, user_message)
    } else {
        // Default: simple format
        let mut formatted = String::new();
        for (role, content) in messages {
            formatted.push_str(&format!("{}: {}\n", role, content));
        }
        formatted.push_str(&format!("User: {}\nAssistant:", user_message));
        formatted
    }
}
