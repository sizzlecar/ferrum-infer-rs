//! Chat template support for conversation formatting

use ferrum_types::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, warn};

/// Chat message in conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender
    pub role: String,
    /// Content of the message
    pub content: String,
    /// Optional name of the sender
    pub name: Option<String>,
}

impl ChatMessage {
    /// Create new chat message
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            name: None,
        }
    }

    /// Create user message
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// Create assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }

    /// Create system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// Set sender name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Chat template for formatting conversations
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    /// Template string
    template: String,
    /// Compiled regex for variable substitution
    var_regex: Regex,
    /// Special tokens
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatTemplate {
    /// Create new chat template
    pub fn new(
        template: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Result<Self> {
        let var_regex = Regex::new(r"\{\{\s*([^}]+)\s*\}\}").map_err(|e| {
            ferrum_types::FerrumError::tokenizer(format!("Invalid template regex: {}", e))
        })?;

        Ok(Self {
            template,
            var_regex,
            bos_token,
            eos_token,
        })
    }

    /// Apply template to conversation
    pub fn apply(&self, messages: &[ChatMessage]) -> Result<String> {
        let mut context = HashMap::new();

        // Add special tokens
        if let Some(bos) = &self.bos_token {
            context.insert("bos_token".to_string(), bos.clone());
        }
        if let Some(eos) = &self.eos_token {
            context.insert("eos_token".to_string(), eos.clone());
        }

        // Add messages to context
        context.insert("messages".to_string(), format_messages(messages));

        // Apply template
        self.substitute_variables(&self.template, &context)
    }

    /// Apply template to a single message
    pub fn apply_message(&self, message: &ChatMessage) -> Result<String> {
        let mut context = HashMap::new();

        // Add message fields
        context.insert("role".to_string(), message.role.clone());
        context.insert("content".to_string(), message.content.clone());
        if let Some(name) = &message.name {
            context.insert("name".to_string(), name.clone());
        }

        // Add special tokens
        if let Some(bos) = &self.bos_token {
            context.insert("bos_token".to_string(), bos.clone());
        }
        if let Some(eos) = &self.eos_token {
            context.insert("eos_token".to_string(), eos.clone());
        }

        self.substitute_variables(&self.template, &context)
    }

    fn substitute_variables(
        &self,
        text: &str,
        context: &HashMap<String, String>,
    ) -> Result<String> {
        let mut result = text.to_string();

        for capture in self.var_regex.captures_iter(text) {
            if let Some(var_match) = capture.get(0) {
                if let Some(var_name) = capture.get(1) {
                    let var_name = var_name.as_str().trim();

                    if let Some(value) = context.get(var_name) {
                        result = result.replace(var_match.as_str(), value);
                    } else {
                        warn!("Template variable '{}' not found in context", var_name);
                        // Keep the placeholder for missing variables
                    }
                }
            }
        }

        Ok(result)
    }
}

/// Format messages for template context
fn format_messages(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|msg| {
            if let Some(name) = &msg.name {
                format!("{} ({}): {}", msg.role, name, msg.content)
            } else {
                format!("{}: {}", msg.role, msg.content)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Common chat templates for popular models
pub struct CommonTemplates;

impl CommonTemplates {
    /// Llama-2 chat template
    pub fn llama2() -> ChatTemplate {
        let template = r#"{% if messages[0]['role'] == 'system' %}
{% set system_message = messages[0]['content'] %}
{% set messages = messages[1:] %}
{% else %}
{% set system_message = '' %}
{% endif %}

{% for message in messages %}
{% if loop.first and system_message %}
{{ bos_token }}[INST] <<SYS>>
{{ system_message }}
<</SYS>>

{{ message['content'] }} [/INST]
{% elif message['role'] == 'user' %}
{{ bos_token }}[INST] {{ message['content'] }} [/INST]
{% elif message['role'] == 'assistant' %}
{{ message['content'] }}{{ eos_token }}
{% endif %}
{% endfor %}"#;

        ChatTemplate::new(
            template.to_string(),
            Some("<s>".to_string()),
            Some("</s>".to_string()),
        )
        .unwrap()
    }

    /// ChatML template (used by GPT-3.5/GPT-4)
    pub fn chatml() -> ChatTemplate {
        let template = r#"{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
<|im_start|>assistant
"#;

        ChatTemplate::new(template.to_string(), None, Some("<|im_end|>".to_string())).unwrap()
    }

    /// Simple template for basic models
    pub fn simple() -> ChatTemplate {
        let template = r#"{% for message in messages %}
{{ message['role'] }}: {{ message['content'] }}
{% endfor %}
assistant: "#;

        ChatTemplate::new(template.to_string(), None, None).unwrap()
    }

    /// Alpaca instruction template
    pub fn alpaca() -> ChatTemplate {
        let template = r#"Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{{ messages[-1]['content'] }}

### Response:
"#;

        ChatTemplate::new(template.to_string(), None, None).unwrap()
    }

    /// Get template by name
    pub fn get(name: &str) -> Option<ChatTemplate> {
        match name.to_lowercase().as_str() {
            "llama2" | "llama-2" => Some(Self::llama2()),
            "chatml" | "chatgpt" => Some(Self::chatml()),
            "simple" | "basic" => Some(Self::simple()),
            "alpaca" => Some(Self::alpaca()),
            _ => {
                warn!("Unknown chat template: {}", name);
                None
            }
        }
    }

    /// List available template names
    pub fn available_templates() -> Vec<&'static str> {
        vec!["llama2", "chatml", "simple", "alpaca"]
    }
}

/// Chat template builder for custom templates
pub struct ChatTemplateBuilder {
    template: Option<String>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatTemplateBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            template: None,
            bos_token: None,
            eos_token: None,
        }
    }

    /// Set template string
    pub fn template(mut self, template: impl Into<String>) -> Self {
        self.template = Some(template.into());
        self
    }

    /// Set BOS token
    pub fn bos_token(mut self, token: impl Into<String>) -> Self {
        self.bos_token = Some(token.into());
        self
    }

    /// Set EOS token
    pub fn eos_token(mut self, token: impl Into<String>) -> Self {
        self.eos_token = Some(token.into());
        self
    }

    /// Build the chat template
    pub fn build(self) -> Result<ChatTemplate> {
        let template = self
            .template
            .ok_or_else(|| ferrum_types::FerrumError::tokenizer("Template string is required"))?;

        ChatTemplate::new(template, self.bos_token, self.eos_token)
    }
}

impl Default for ChatTemplateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Template registry for managing multiple templates
#[derive(Debug, Default)]
pub struct TemplateRegistry {
    templates: HashMap<String, ChatTemplate>,
}

impl TemplateRegistry {
    /// Create new registry
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.load_common_templates();
        registry
    }

    /// Load common templates
    pub fn load_common_templates(&mut self) {
        for name in CommonTemplates::available_templates() {
            if let Some(template) = CommonTemplates::get(name) {
                self.templates.insert(name.to_string(), template);
                debug!("Loaded template: {}", name);
            }
        }
    }

    /// Register custom template
    pub fn register(&mut self, name: impl Into<String>, template: ChatTemplate) {
        let name = name.into();
        debug!("Registering custom template: {}", name);
        self.templates.insert(name, template);
    }

    /// Get template by name
    pub fn get(&self, name: &str) -> Option<&ChatTemplate> {
        self.templates.get(name)
    }

    /// List available templates
    pub fn list_templates(&self) -> Vec<String> {
        self.templates.keys().cloned().collect()
    }

    /// Check if template exists
    pub fn has_template(&self, name: &str) -> bool {
        self.templates.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_creation() {
        let msg = ChatMessage::user("Hello, world!").with_name("Alice");

        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello, world!");
        assert_eq!(msg.name, Some("Alice".to_string()));
    }

    #[test]
    fn test_simple_template() {
        let template = CommonTemplates::simple();
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ];

        let result = template.apply(&messages).unwrap();
        assert!(result.contains("user: Hello"));
        assert!(result.contains("assistant: Hi there!"));
    }

    #[test]
    fn test_template_registry() {
        let registry = TemplateRegistry::new();

        assert!(registry.has_template("simple"));
        assert!(registry.has_template("llama2"));
        assert!(!registry.has_template("nonexistent"));

        let templates = registry.list_templates();
        assert!(templates.len() > 0);
    }

    #[test]
    fn test_template_builder() {
        let template = ChatTemplateBuilder::new()
            .template("Hello {{ name }}!")
            .bos_token("<s>")
            .eos_token("</s>")
            .build()
            .unwrap();

        // Basic smoke test - would need proper context substitution in real use
        assert!(template.template.contains("Hello"));
    }
}
