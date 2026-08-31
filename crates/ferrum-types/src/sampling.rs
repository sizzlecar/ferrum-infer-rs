//! Sampling and generation parameters

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{FerrumError, Result, TokenId};

/// Default repetition penalty for product chat entrypoints.
///
/// Chat defaults are greedy, and unpenalized greedy decoding can lock into
/// deterministic token loops on real models. Keep CLI `run` and OpenAI chat
/// serving on the same default unless an endpoint exposes an explicit override.
pub const DEFAULT_CHAT_REPETITION_PENALTY: f32 = 1.1;

/// Typed wire protocol emitted by the model after prompt rendering.
///
/// This is carried per request because output-token policy and response
/// parsing must agree on the exact model protocol. It must be selected from
/// resolved model metadata rather than inferred from a user-facing model name.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelOutputProtocol {
    /// Ordinary text output with no model-specific control-token protocol.
    #[default]
    Text,
    /// OpenAI Harmony output used by GPT-OSS checkpoints.
    HarmonyGptOss,
}

impl ModelOutputProtocol {
    /// Non-terminal control tokens that generation must be able to emit for
    /// this protocol. Terminal `<|call|>` / `<|return|>` tokens remain owned
    /// by the model's typed EOS configuration.
    pub const fn generated_control_token_texts(self) -> &'static [&'static str] {
        match self {
            Self::Text => &[],
            Self::HarmonyGptOss => &[
                "<|channel|>",
                "<|message|>",
                "<|start|>",
                "<|end|>",
                "<|constrain|>",
            ],
        }
    }

    /// Special-token text that skip-special decoding must preserve so the
    /// product parser can observe the complete protocol envelope.
    pub const fn preserved_special_token_texts(self) -> &'static [&'static str] {
        match self {
            Self::Text => &[],
            Self::HarmonyGptOss => &[
                "<|channel|>",
                "<|message|>",
                "<|start|>",
                "<|end|>",
                "<|constrain|>",
                "<|call|>",
                "<|return|>",
            ],
        }
    }
}

/// Sampling parameters for generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Temperature for randomness (0.0 = deterministic, higher = more random)
    pub temperature: f32,
    /// Nucleus sampling probability threshold
    pub top_p: f32,
    /// Top-k sampling - consider only top k tokens
    pub top_k: Option<usize>,
    /// Repetition penalty to reduce repetitive text
    pub repetition_penalty: f32,
    /// Presence penalty for token diversity
    pub presence_penalty: f32,
    /// Frequency penalty based on token frequency
    pub frequency_penalty: f32,
    /// Stop sequences to end generation
    pub stop_sequences: Vec<String>,
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
    /// Minimum probability threshold for tokens
    pub min_p: Option<f32>,
    /// Tail free sampling parameter
    pub tfs: Option<f32>,
    /// Typical sampling parameter
    pub typical_p: Option<f32>,
    /// Mirostat sampling parameters
    pub mirostat: Option<MirostatParams>,
    /// Response format constraint (JSON mode, schema-constrained, etc.)
    #[serde(default)]
    pub response_format: ResponseFormat,
    /// Point at which a structured-output grammar starts constraining model
    /// output. Thinking templates can open a reasoning block in the prompt;
    /// in that case the grammar activates only after the typed delimiter.
    #[serde(default)]
    pub structured_output_start: StructuredOutputStart,
    /// Product completion boundary that must be satisfied before model EOS
    /// tokens may terminate generation.
    ///
    /// This is independent of structured-output grammar activation: ordinary
    /// text and structured responses need a complete reasoning-to-payload
    /// transition when the rendered prompt opens one. A complete lexical
    /// response envelope, such as a tool call, is an alternate terminal path.
    #[serde(default)]
    pub response_completion_boundary: ResponseCompletionBoundary,
    /// Typed model-output protocol used for control-token sampling and
    /// response parsing.
    #[serde(default)]
    pub model_output_protocol: ModelOutputProtocol,
}

/// Typed activation boundary for constrained decoding.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(tag = "mode", content = "delimiter", rename_all = "snake_case")]
pub enum StructuredOutputStart {
    /// Constrain the first generated token.
    #[default]
    Immediate,
    /// Allow reasoning tokens until this exact tokenizer sequence is emitted,
    /// then constrain every subsequent token.
    AfterDelimiter(String),
}

/// A lexical response envelope that can complete an otherwise pending response.
///
/// This describes only token boundaries. The product/API layer remains
/// responsible for validating the enclosed payload after generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResponseCompletionEnvelope {
    pub open_token_text: String,
    pub close_token_text: String,
    pub max_envelopes: usize,
}

/// Typed model-EOS boundary for product response completion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ResponseCompletionBoundary {
    /// Model EOS may terminate any generated token.
    #[default]
    Immediate,
    /// Model EOS remains unavailable until the exact delimiter token sequence
    /// and at least one subsequent non-whitespace payload token are emitted,
    /// or until a configured alternate envelope is complete.
    AfterDelimiterAndPayload {
        delimiter: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        alternate_envelope: Option<ResponseCompletionEnvelope>,
    },
}

/// Response format for structured output. Mirrors OpenAI's
/// `response_format` API — no proprietary extensions.
///
/// - `Text`: no constraint (default)
/// - `JsonObject`: output must be a valid JSON object (matches OpenAI's
///   `{"type": "json_object"}`)
/// - `JsonSchema(schema)`: output must conform to the given JSON Schema
///   (matches OpenAI's `{"type": "json_schema", "json_schema": {...}}`).
///   Internally compiled to a regex FSM for per-token hard masking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "schema")]
#[derive(Default)]
pub enum ResponseFormat {
    /// No constraint — raw text output.
    #[default]
    Text,
    /// Output must be a valid JSON object.
    JsonObject,
    /// Output must conform to the given JSON schema (as a JSON string).
    JsonSchema(String),
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop_sequences: vec![],
            seed: None,
            min_p: None,
            tfs: None,
            typical_p: None,
            mirostat: None,
            response_format: ResponseFormat::default(),
            structured_output_start: StructuredOutputStart::default(),
            response_completion_boundary: ResponseCompletionBoundary::default(),
            model_output_protocol: ModelOutputProtocol::default(),
        }
    }
}

impl SamplingParams {
    /// Create greedy sampling parameters (deterministic)
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            ..Default::default()
        }
    }

    /// Create default sampling parameters with temperature
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            ..Default::default()
        }
    }

    /// Validate sampling parameters
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(FerrumError::invalid_request(
                "temperature must be finite and non-negative".to_string(),
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(FerrumError::invalid_request(
                "top_p must be in range (0, 1]".to_string(),
            ));
        }
        if let Some(top_k) = self.top_k {
            if top_k == 0 {
                return Err(FerrumError::invalid_request(
                    "top_k must be positive".to_string(),
                ));
            }
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(FerrumError::invalid_request(
                "repetition_penalty must be finite and positive".to_string(),
            ));
        }
        if !self.presence_penalty.is_finite() || !(-2.0..=2.0).contains(&self.presence_penalty) {
            return Err(FerrumError::invalid_request(
                "presence_penalty must be in range [-2, 2]".to_string(),
            ));
        }
        if !self.frequency_penalty.is_finite() || !(-2.0..=2.0).contains(&self.frequency_penalty) {
            return Err(FerrumError::invalid_request(
                "frequency_penalty must be in range [-2, 2]".to_string(),
            ));
        }
        if let Some(min_p) = self.min_p {
            if !min_p.is_finite() || min_p <= 0.0 || min_p > 1.0 {
                return Err(FerrumError::invalid_request(
                    "min_p must be in range (0, 1]".to_string(),
                ));
            }
        }
        if let Some(tfs) = self.tfs {
            if !tfs.is_finite() || tfs <= 0.0 || tfs > 1.0 {
                return Err(FerrumError::invalid_request(
                    "tfs must be in range (0, 1]".to_string(),
                ));
            }
        }
        if let Some(typical_p) = self.typical_p {
            if !typical_p.is_finite() || typical_p <= 0.0 || typical_p > 1.0 {
                return Err(FerrumError::invalid_request(
                    "typical_p must be in range (0, 1]".to_string(),
                ));
            }
        }
        if let ResponseCompletionBoundary::AfterDelimiterAndPayload {
            delimiter,
            alternate_envelope,
        } = &self.response_completion_boundary
        {
            if delimiter.is_empty() {
                return Err(FerrumError::invalid_request(
                    "response completion delimiter must not be empty".to_string(),
                ));
            }
            if let Some(envelope) = alternate_envelope {
                if envelope.open_token_text.is_empty() || envelope.close_token_text.is_empty() {
                    return Err(FerrumError::invalid_request(
                        "response completion envelope tokens must not be empty".to_string(),
                    ));
                }
                if envelope.max_envelopes == 0 {
                    return Err(FerrumError::invalid_request(
                        "response completion envelope limit must be greater than zero".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Mirostat sampling parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirostatParams {
    /// Mirostat mode (1 or 2)
    pub mode: u8,
    /// Target entropy
    pub tau: f32,
    /// Learning rate
    pub eta: f32,
}

/// Sampling presets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingPresets {
    pub presets: HashMap<String, SamplingParams>,
}

impl Default for SamplingPresets {
    fn default() -> Self {
        let mut presets = HashMap::new();
        presets.insert("greedy".to_string(), SamplingParams::greedy());
        presets.insert(
            "creative".to_string(),
            SamplingParams {
                temperature: 1.2,
                top_p: 0.9,
                top_k: Some(50),
                repetition_penalty: 1.1,
                ..Default::default()
            },
        );
        presets.insert(
            "precise".to_string(),
            SamplingParams {
                temperature: 0.3,
                top_p: 0.95,
                top_k: Some(20),
                repetition_penalty: 1.05,
                ..Default::default()
            },
        );
        Self { presets }
    }
}

/// Request priority levels
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub enum Priority {
    Low = 0,
    #[default]
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Reason for completion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// Hit maximum token limit
    Length,
    /// Hit stop sequence
    Stop,
    /// Hit end-of-sequence token
    EOS,
    /// Request was cancelled
    Cancelled,
    /// Error occurred during generation
    Error,
    /// Content filter triggered
    ContentFilter,
}

/// Special tokens configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpecialTokens {
    /// Beginning of sequence token
    pub bos_token: Option<TokenId>,
    /// End of sequence token
    pub eos_token: Option<TokenId>,
    /// Unknown token
    pub unk_token: Option<TokenId>,
    /// Padding token
    pub pad_token: Option<TokenId>,
    /// Separator token
    pub sep_token: Option<TokenId>,
    /// Classification token
    pub cls_token: Option<TokenId>,
    /// Mask token
    pub mask_token: Option<TokenId>,
    /// Additional end-of-sequence tokens. Models such as Llama-3 and GLM
    /// declare several `eos_token_id`s in `generation_config.json`;
    /// `eos_token` holds the primary one and the rest land here.
    #[serde(default)]
    pub extra_eos_tokens: Vec<TokenId>,
}
