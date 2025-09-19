//! Sampling and generation parameters

use serde::{Deserialize, Serialize};

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
    pub fn validate(&self) -> Result<(), String> {
        if self.temperature < 0.0 {
            return Err("Temperature must be non-negative".to_string());
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err("top_p must be in range (0, 1]".to_string());
        }
        if let Some(top_k) = self.top_k {
            if top_k == 0 {
                return Err("top_k must be positive".to_string());
            }
        }
        if self.repetition_penalty <= 0.0 {
            return Err("Repetition penalty must be positive".to_string());
        }
        if let Some(min_p) = self.min_p {
            if min_p <= 0.0 || min_p > 1.0 {
                return Err("min_p must be in range (0, 1]".to_string());
            }
        }
        if let Some(tfs) = self.tfs {
            if tfs <= 0.0 || tfs > 1.0 {
                return Err("tfs must be in range (0, 1]".to_string());
            }
        }
        if let Some(typical_p) = self.typical_p {
            if typical_p <= 0.0 || typical_p > 1.0 {
                return Err("typical_p must be in range (0, 1]".to_string());
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

/// Request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Normal
    }
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    /// Beginning of sequence token
    pub bos_token: Option<u32>,
    /// End of sequence token
    pub eos_token: Option<u32>,
    /// Unknown token
    pub unk_token: Option<u32>,
    /// Padding token
    pub pad_token: Option<u32>,
    /// Separator token
    pub sep_token: Option<u32>,
    /// Classification token
    pub cls_token: Option<u32>,
    /// Mask token
    pub mask_token: Option<u32>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token: None,
            eos_token: None,
            unk_token: None,
            pad_token: None,
            sep_token: None,
            cls_token: None,
            mask_token: None,
        }
    }
}
