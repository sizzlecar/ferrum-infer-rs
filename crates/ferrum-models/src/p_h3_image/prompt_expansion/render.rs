//! Render the supplied immutable HF template with the reference call context.
//! Image markers remain processor placeholders, not fabricated visual inputs.
use super::*;
use serde_json::{json, Value};
use std::io::{self, Write};

pub struct PH3ImagePromptExpansionInput {
    pub(super) text: String,
    pub(super) images: u8,
    pub(super) max_new_tokens: u32,
    pub(super) source_sha256: [String; 3],
}
impl PH3ImagePromptExpansionInput {
    pub fn rendered(&self) -> &str {
        &self.text
    }
    /// The caller must supply precisely these ordered real image payloads to
    /// its multimodal processor. Rendering does not expand their patch tokens.
    pub fn image_count(&self) -> u8 {
        self.images
    }
}
impl PH3ImagePromptExpansionProtocol {
    pub fn render(
        &self,
        request: &PH3ImagePromptExpansionRequest<'_>,
    ) -> Result<PH3ImagePromptExpansionInput> {
        self.check_request(request)?;
        let raw = request.request.trim_matches(python_space);
        let content = match request.task {
            PH3ImagePromptExpansionTask::TextToImage { width, height } => Value::String(format!(
                "Aspect ratio: {}\nRequest: {raw}",
                ph3_image_expansion_aspect_ratio(width, height)?
            )),
            PH3ImagePromptExpansionTask::Edit { image_count } => {
                let mut parts = Vec::with_capacity(usize::from(image_count) * 2 + 1);
                for index in 1..=image_count {
                    parts.push(json!({"type":"text","text":format!("<Picture {index}>: ")}));
                    parts.push(json!({"type":"image"}));
                }
                parts.push(json!({"type":"text","text":format!("Edit <Picture 1>. {raw}")}));
                Value::Array(parts)
            }
        };
        let mut context = self.specials.clone();
        context.insert(
            "messages".into(),
            json!([{"role":"user","content":content}]),
        );
        context.insert("add_generation_prompt".into(), Value::Bool(true));
        // No system message/tools, and no enable_thinking override: this is
        // the actual tokenizer.apply_chat_template call made by Diffusers.
        let mut env = minijinja::Environment::new();
        env.set_fuel(Some(self.limits.template_fuel));
        env.set_recursion_limit(64);
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        env.add_template("processor", &self.template)
            .map_err(template_error)?;
        let mut output = LimitedOutput {
            bytes: Vec::new(),
            maximum: self.limits.max_rendered_bytes,
        };
        env.get_template("processor")
            .map_err(template_error)?
            .render_captured_to(Value::Object(context), &mut output)
            .map_err(template_error)?;
        let text = String::from_utf8(output.bytes)
            .map_err(|e| PH3ImagePromptExpansionError::Template(e.to_string()))?;
        Ok(PH3ImagePromptExpansionInput {
            text,
            images: request.image_count(),
            max_new_tokens: request.max_new_tokens,
            source_sha256: self.source_sha256.clone(),
        })
    }
    /// T2I has no processor image expansion. Image requests deliberately fail
    /// this shortcut instead of masquerading one image-pad token as features.
    pub fn encode_text_input(&self, input: &PH3ImagePromptExpansionInput) -> Result<Vec<u32>> {
        if input.source_sha256 != self.source_sha256 {
            return Err(PH3ImagePromptExpansionError::Request(
                "rendered input belongs to another tokenizer/template source".into(),
            ));
        }
        if input.images != 0 {
            return Err(PH3ImagePromptExpansionError::Request(
                "edit requires the actual multimodal processor".into(),
            ));
        }
        let encoded = self
            .tokenizer
            .encode(input.text.as_str(), false)
            .map_err(|e| PH3ImagePromptExpansionError::Request(e.to_string()))?;
        if encoded
            .get_ids()
            .len()
            .checked_add(input.max_new_tokens as usize)
            .is_none_or(|n| n > self.context_tokens as usize)
        {
            return Err(PH3ImagePromptExpansionError::Request(
                "prompt plus complete generation exceeds the source context capacity".into(),
            ));
        }
        Ok(encoded.get_ids().to_vec())
    }
}
fn template_error(e: minijinja::Error) -> PH3ImagePromptExpansionError {
    PH3ImagePromptExpansionError::Template(e.to_string())
}
struct LimitedOutput {
    bytes: Vec<u8>,
    maximum: usize,
}
impl Write for LimitedOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.maximum.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other(
                "rendered processor template exceeds metadata capacity",
            ));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(|e| io::Error::other(e.to_string()))?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
