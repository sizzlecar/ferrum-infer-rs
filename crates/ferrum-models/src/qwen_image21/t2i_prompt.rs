//! Pure T2I adapter for the fixed Qwen-Image 2.1 processor template.
//!
//! Reference: vllm-omni c0551d22674d0db81305c800b66ad83035fdc980,
//! pipeline_qwen_image_21.py:434-457,572-715; Transformers v5.3.0,
//! modeling_qwen3_vl.py:291-380,886-915,1223-1265. No image/video branch,
//! KV cache, final text RMSNorm, or CPU model fallback is supplied here.
//! Tokenizers 0.21 and libm 0.2.15 are explicit local implementations, not a
//! claim of deployed Tokenizers 0.23.3 / Torch bitwise parity. Source hashes
//! bind the supplied bytes, without authenticating a Hugging Face revision.
//!
//! Tokenizer/control objects and borrowed source mappings are outside the Host
//! backing-byte ledger. Gathered embeddings, rotary tables, and cropped hidden
//! payloads are admitted before allocation. No full embedding-table copy is made.

mod managed;
mod tokenizer;
pub use managed::{QwenImage21T2iManagedInputs, QwenImage21T2iTextFeatures};
pub use tokenizer::QwenImage21T2iTokenizer;

use crate::qwen3_vl_text::{Qwen3VLTextKeyRun, Qwen3VLTextLayout};
use anyhow::{ensure, Result};

pub const T2I_TEXT_ROTARY_PROFILE: &str = "qwen3vl.t2i.rope.f32_libm-0.2.15.to_bf16.v1";
pub const T2I_SYSTEM_PROMPT: &str = "Comprehend and analyze the provided prompt.";

/// Bounded control metadata. Padding validity is independent of token value:
/// a literal PAD token in a user prompt remains a valid sequence position.
#[derive(Debug, Clone)]
pub struct QwenImage21T2iPromptBatch {
    ids: Vec<u32>,
    lengths: Vec<usize>,
    drop_prefix: usize,
    layout: Qwen3VLTextLayout,
}
impl QwenImage21T2iPromptBatch {
    fn from_encoded(samples: Vec<Vec<u32>>, drop_prefix: usize, pad: u32) -> Result<Self> {
        ensure!(
            !samples.is_empty() && samples.len() <= 16,
            "T2I requires 1..=16 prompts"
        );
        ensure!(drop_prefix > 0, "system prefix must be nonempty");
        let sequence = samples.iter().map(Vec::len).max().unwrap();
        ensure!(
            samples.iter().all(|s| s.len() > drop_prefix),
            "system crop consumes a complete prompt"
        );
        let mut ids = Vec::new();
        ids.try_reserve_exact(
            sequence
                .checked_mul(samples.len())
                .ok_or_else(|| anyhow::anyhow!("token grid overflows"))?,
        )?;
        let mut lengths = Vec::with_capacity(samples.len());
        let mut runs = Vec::with_capacity(samples.len());
        for sample in samples {
            let left = sequence - sample.len();
            let mut sample_runs = Vec::with_capacity(2);
            if left != 0 {
                sample_runs.push(Qwen3VLTextKeyRun::new(u32::try_from(left)?, false));
            }
            sample_runs.push(Qwen3VLTextKeyRun::new(u32::try_from(sample.len())?, true));
            runs.push(sample_runs);
            lengths.push(sample.len());
            ids.extend(std::iter::repeat_n(pad, left));
            ids.extend(sample);
        }
        Ok(Self {
            ids,
            lengths,
            drop_prefix,
            layout: Qwen3VLTextLayout::new(u32::try_from(sequence)?, runs)?,
        })
    }
    pub fn layout(&self) -> &Qwen3VLTextLayout {
        &self.layout
    }
    pub fn input_ids(&self) -> &[u32] {
        &self.ids
    }
    pub fn valid_input_lengths(&self) -> &[usize] {
        &self.lengths
    }
    pub fn drop_prefix_tokens(&self) -> usize {
        self.drop_prefix
    }
    pub fn sequence_length(&self) -> usize {
        self.layout.sequence_length() as usize
    }
    pub fn batch(&self) -> usize {
        self.lengths.len()
    }
    pub fn valid_feature_lengths(&self) -> Vec<usize> {
        self.lengths.iter().map(|n| n - self.drop_prefix).collect()
    }
    pub fn feature_sequence_length(&self) -> usize {
        self.sequence_length() - self.drop_prefix
    }
    pub fn is_valid_key(&self, sample: usize, column: usize) -> Option<bool> {
        self.lengths
            .get(sample)
            .filter(|_| column < self.sequence_length())
            .map(|&length| column >= self.sequence_length() - length)
    }
    /// Fresh, cache-free T2I: all three axes use the complete padded column.
    /// No materialized B*S*3 position tensor is needed by this adapter.
    pub fn position_ids(&self, sample: usize, column: usize) -> Option<[u32; 3]> {
        (sample < self.batch() && column < self.sequence_length()).then_some([column as u32; 3])
    }
}

#[cfg(test)]
mod tests;
