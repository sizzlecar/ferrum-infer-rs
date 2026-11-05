//! The official prompt-upsampler protocol, independently of native execution.
//! Request framing and validation follow Diffusers `c55ec27943462eeeb65c177d6ff7d8272d21d7a7`,
//! `modular_pipelines/p_h3_image/prompt_upsampler.py`. Rendering uses the
//! caller's original processor template, without substituting server framing.
//! Product enable/disable policy and model execution belong to the caller.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

mod edit_format;
mod images;
mod render;
mod rotary_inputs;
pub use edit_format::{ph3_image_edit_prompt_form, PH3ImageEditPromptForm};
pub use images::{PH3ImagePromptExpansionImageRun, PH3ImagePromptExpansionImageTokens};
pub use render::PH3ImagePromptExpansionInput;
pub use rotary_inputs::{PH3ImagePromptExpansionRotary, PH3_IMAGE_EXPANSION_ROTARY_PROFILE};

pub const PH3_IMAGE_PROMPT_EXPANSION_PROTOCOL: &str =
    "p_h3_image.diffusers.c55ec279.single_user.greedy.thinking.im_end.v1";
pub const PH3_IMAGE_T2I_EXPANSION_MAX_NEW_TOKENS: u32 = 1024;
pub const PH3_IMAGE_EDIT_EXPANSION_MAX_NEW_TOKENS: u32 = 1536;
pub const PH3_IMAGE_PROMPT_EXPANSION_MAX_IMAGES: u8 = 8;
const EOS: &str = "<|im_end|>";

#[derive(Debug, thiserror::Error)]
pub enum PH3ImagePromptExpansionError {
    #[error("invalid prompt-upsampler source: {0}")]
    Source(String),
    #[error("invalid prompt-upsampler request: {0}")]
    Request(String),
    #[error("prompt-upsampler template rendering failed: {0}")]
    Template(String),
    #[error("prompt-upsampler generation was not completed at its actual EOS token")]
    Incomplete,
    #[error("prompt-upsampler answer is invalid: {0}")]
    Answer(&'static str),
}
type Result<T> = std::result::Result<T, PH3ImagePromptExpansionError>;

/// Accepted serialized metadata/artifact limits, not parser/decoder peak heap
/// usage. Native inputs, outputs and KV still require their existing runtime
/// and Host budgets. The caller's generation ceiling cannot exceed the actual
/// tokenizer's declared context capacity.
#[derive(Clone, Copy, Debug)]
pub struct PH3ImagePromptExpansionLimits {
    pub max_tokenizer_bytes: usize,
    pub max_template_bytes: usize,
    pub max_request_bytes: usize,
    pub max_rendered_bytes: usize,
    pub max_answer_bytes: usize,
    pub max_generated_tokens: u32,
    pub template_fuel: u64,
}
impl PH3ImagePromptExpansionLimits {
    fn validate(self) -> Result<()> {
        if [
            self.max_tokenizer_bytes,
            self.max_template_bytes,
            self.max_request_bytes,
            self.max_rendered_bytes,
            self.max_answer_bytes,
        ]
        .iter()
        .any(|&n| n == 0 || n > isize::MAX as usize)
            || self.template_fuel == 0
            || self.max_generated_tokens == 0
        {
            return Err(PH3ImagePromptExpansionError::Source(
                "metadata capacities must be positive and addressable".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImagePromptExpansionTask {
    TextToImage { width: u32, height: u32 },
    Edit { image_count: u8 },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PH3ImagePromptExpansionGeneration {
    pub max_new_tokens: u32,
    pub do_sample: bool,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}
/// Borrows the unchanged user request. Image payloads stay with the caller;
/// `image_count` declares the ordered actual images supplied to the processor.
#[derive(Clone, Copy, Debug)]
pub struct PH3ImagePromptExpansionRequest<'a> {
    request: &'a str,
    task: PH3ImagePromptExpansionTask,
    max_new_tokens: u32,
}
impl<'a> PH3ImagePromptExpansionRequest<'a> {
    pub fn text_to_image(request: &'a str, width: u32, height: u32) -> Result<Self> {
        ph3_image_expansion_aspect_ratio(width, height)?;
        Ok(Self {
            request,
            task: PH3ImagePromptExpansionTask::TextToImage { width, height },
            max_new_tokens: PH3_IMAGE_T2I_EXPANSION_MAX_NEW_TOKENS,
        })
    }
    pub fn edit(request: &'a str, image_count: u8) -> Result<Self> {
        if !(1..=PH3_IMAGE_PROMPT_EXPANSION_MAX_IMAGES).contains(&image_count) {
            return Err(PH3ImagePromptExpansionError::Request(
                "edit requires 1..=8 ordered actual images".into(),
            ));
        }
        Ok(Self {
            request,
            task: PH3ImagePromptExpansionTask::Edit { image_count },
            max_new_tokens: PH3_IMAGE_EDIT_EXPANSION_MAX_NEW_TOKENS,
        })
    }
    pub fn with_max_new_tokens(mut self, tokens: u32) -> Result<Self> {
        if tokens == 0 {
            return Err(PH3ImagePromptExpansionError::Request(
                "max_new_tokens must be positive".into(),
            ));
        }
        self.max_new_tokens = tokens;
        Ok(self)
    }
    pub fn task(&self) -> PH3ImagePromptExpansionTask {
        self.task
    }
    pub fn max_new_tokens(&self) -> u32 {
        self.max_new_tokens
    }
    pub fn image_count(&self) -> u8 {
        match self.task {
            PH3ImagePromptExpansionTask::TextToImage { .. } => 0,
            PH3ImagePromptExpansionTask::Edit { image_count } => image_count,
        }
    }
    pub fn request(&self) -> &'a str {
        self.request
    }
}

/// The closest label in the reference's insertion order. Equal distances keep
/// the first label; minimizing linear aspect-ratio distance is a different rule.
pub fn ph3_image_expansion_aspect_ratio(width: u32, height: u32) -> Result<&'static str> {
    if width == 0 || height == 0 {
        return Err(PH3ImagePromptExpansionError::Request(
            "image dimensions must be positive".into(),
        ));
    }
    let aspect = f64::from(width) / f64::from(height);
    let mut best = ("1:1", f64::INFINITY);
    for (label, ratio) in [
        ("1:1", 1.),
        ("4:3", 4. / 3.),
        ("3:4", 3. / 4.),
        ("3:2", 3. / 2.),
        ("2:3", 2. / 3.),
        ("16:9", 16. / 9.),
        ("9:16", 9. / 16.),
    ] {
        let distance = (aspect / ratio).ln().abs();
        if distance < best.1 {
            best = (label, distance);
        }
    }
    Ok(best.0)
}

/// Checked immutable tokenizer/template identity. This object does not load an
/// upsampler model or claim an executable generation backend.
pub struct PH3ImagePromptExpansionProtocol {
    tokenizer: Tokenizer,
    template: String,
    specials: serde_json::Map<String, serde_json::Value>,
    eos: u32,
    pad: u32,
    context_tokens: u32,
    source_sha256: [String; 3],
    limits: PH3ImagePromptExpansionLimits,
}
impl PH3ImagePromptExpansionProtocol {
    pub fn from_source_bytes(
        tokenizer: &[u8],
        config: &[u8],
        template: &[u8],
        limits: PH3ImagePromptExpansionLimits,
    ) -> Result<Self> {
        limits.validate()?;
        if tokenizer.is_empty()
            || config.is_empty()
            || template.is_empty()
            || tokenizer.len() > limits.max_tokenizer_bytes
            || config.len() > limits.max_tokenizer_bytes
            || template.len() > limits.max_template_bytes
        {
            return Err(PH3ImagePromptExpansionError::Source(
                "source bytes exceed declared nonempty metadata capacity".into(),
            ));
        }
        let source_sha256 =
            [tokenizer, config, template].map(|bytes| format!("{:x}", Sha256::digest(bytes)));
        let raw: serde_json::Value = serde_json::from_slice(config)
            .map_err(|e| PH3ImagePromptExpansionError::Source(e.to_string()))?;
        let raw = raw.as_object().ok_or_else(|| {
            PH3ImagePromptExpansionError::Source("tokenizer config must be an object".into())
        })?;
        let content = |name: &str| -> Option<&str> {
            let v = raw.get(name)?;
            v.as_str().or_else(|| v.get("content")?.as_str())
        };
        if content("eos_token") != Some(EOS) || content("pad_token").is_none() {
            return Err(PH3ImagePromptExpansionError::Source(
                "requires the original declared im_end EOS and PAD token".into(),
            ));
        }
        let context_tokens = raw
            .get("model_max_length")
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| {
                PH3ImagePromptExpansionError::Source(
                    "requires a positive addressable declared model_max_length".into(),
                )
            })?;
        if limits.max_generated_tokens > context_tokens {
            return Err(PH3ImagePromptExpansionError::Source(
                "generation ceiling exceeds the source context capacity".into(),
            ));
        }
        let mut tokenizer = Tokenizer::from_bytes(tokenizer)
            .map_err(|e| PH3ImagePromptExpansionError::Source(e.to_string()))?;
        // The reference passes truncation/padding neither to apply_chat_template
        // nor the processor. A saved tokenizer's old options cannot silently
        // remove part of this request or insert a rectangular storage policy.
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| PH3ImagePromptExpansionError::Source(e.to_string()))?;
        let token = |text| -> Result<u32> {
            let id = tokenizer.token_to_id(text).ok_or_else(|| {
                PH3ImagePromptExpansionError::Source(format!("missing token {text}"))
            })?;
            let encoded = tokenizer
                .encode(text, false)
                .map_err(|e| PH3ImagePromptExpansionError::Source(e.to_string()))?;
            if encoded.get_ids() != [id] {
                return Err(PH3ImagePromptExpansionError::Source(format!(
                    "{text} must be one exact token"
                )));
            }
            Ok(id)
        };
        let eos = token(EOS)?;
        let pad = token(content("pad_token").unwrap())?;
        let template = std::str::from_utf8(template)
            .map_err(|e| PH3ImagePromptExpansionError::Source(e.to_string()))?
            .to_owned();
        if template.trim().is_empty() {
            return Err(PH3ImagePromptExpansionError::Source(
                "empty processor chat template".into(),
            ));
        }
        let specials = raw
            .iter()
            .filter_map(|(name, value)| {
                if !name.ends_with("_token") || value.is_null() {
                    return None;
                }
                value
                    .as_str()
                    .or_else(|| value.get("content")?.as_str())
                    .map(|v| (name.clone(), serde_json::Value::String(v.into())))
            })
            .collect();
        // Preserve hashes of original source bytes, not a reserialized config.
        Ok(Self {
            tokenizer,
            template,
            specials,
            eos,
            pad,
            context_tokens,
            source_sha256,
            limits,
        })
    }
    pub fn eos_token_id(&self) -> u32 {
        self.eos
    }
    pub fn pad_token_id(&self) -> u32 {
        self.pad
    }
    pub fn source_sha256(&self) -> &[String; 3] {
        &self.source_sha256
    }
    pub fn protocol(&self) -> &'static str {
        PH3_IMAGE_PROMPT_EXPANSION_PROTOCOL
    }
    pub fn context_tokens(&self) -> u32 {
        self.context_tokens
    }
    fn check_request(&self, request: &PH3ImagePromptExpansionRequest<'_>) -> Result<()> {
        if request.request.len() > self.limits.max_request_bytes
            || request.max_new_tokens > self.limits.max_generated_tokens
        {
            return Err(PH3ImagePromptExpansionError::Request(
                "request bytes or generation length exceed their declared capacity".into(),
            ));
        }
        Ok(())
    }
    pub fn generation(
        &self,
        request: &PH3ImagePromptExpansionRequest<'_>,
    ) -> Result<PH3ImagePromptExpansionGeneration> {
        self.check_request(request)?;
        Ok(PH3ImagePromptExpansionGeneration {
            max_new_tokens: request.max_new_tokens,
            do_sample: false,
            eos_token_id: self.eos,
            pad_token_id: self.pad,
        })
    }

    /// Consume the complete generated token history, including its actual EOS.
    /// A transport stop string, missing IDs or a Length/Cancelled completion
    /// cannot certify success. Reasoning tokens count toward max_new_tokens.
    pub fn extract(
        &self,
        request: &PH3ImagePromptExpansionRequest<'_>,
        new_ids: &[u32],
        finish: ferrum_types::FinishReason,
    ) -> Result<String> {
        self.check_request(request)?;
        if finish != ferrum_types::FinishReason::EOS
            || new_ids.len() > request.max_new_tokens as usize
        {
            return Err(PH3ImagePromptExpansionError::Incomplete);
        }
        let end = new_ids
            .iter()
            .position(|&id| id == self.eos)
            .ok_or(PH3ImagePromptExpansionError::Incomplete)?;
        if new_ids[..end]
            .iter()
            .any(|&id| self.tokenizer.id_to_token(id).is_none())
        {
            return Err(PH3ImagePromptExpansionError::Answer(
                "unknown token in generated history",
            ));
        }
        let answer = self
            .tokenizer
            .decode(&new_ids[..end], false)
            .map_err(|_| PH3ImagePromptExpansionError::Answer("token history cannot be decoded"))?;
        if answer.len() > self.limits.max_answer_bytes {
            return Err(PH3ImagePromptExpansionError::Answer(
                "decoded answer exceeds metadata capacity",
            ));
        }
        if answer.matches("</think>").count() != 1 {
            return Err(PH3ImagePromptExpansionError::Answer(
                "requires exactly one generated thinking terminator",
            ));
        }
        let prompt = answer
            .split_once("</think>")
            .unwrap()
            .1
            .trim_matches(python_space);
        if prompt
            .split(python_space)
            .filter(|word| !word.is_empty())
            .take(5)
            .count()
            < 5
        {
            return Err(PH3ImagePromptExpansionError::Answer(
                "expanded prompt requires at least five words",
            ));
        }
        if request.image_count() > 0 && ph3_image_edit_prompt_form(prompt).is_none() {
            return Err(PH3ImagePromptExpansionError::Answer(
                "edit output lacks an official brief or preservation instruction",
            ));
        }
        Ok(prompt.to_owned())
    }
}
// Python str.strip/split also classify the four C0 information separators.
fn python_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

#[cfg(test)]
mod tests;
