//! Reproducible source and selection evidence for real conversation workloads.

use crate::BenchmarkPhase;
use serde::{Deserialize, Serialize};

pub mod records;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareGptFilter {
    /// All lengths use the frozen client tokenizer, without special tokens.
    pub min_input_tokens: u32,
    pub max_input_tokens: Option<u32>,
    pub min_output_tokens: u32,
    pub max_output_tokens: Option<u32>,
    /// Bounds raw prompt + requested output + the explicit template reserve.
    pub max_total_tokens: Option<u32>,
    /// An estimate, not a claim to have rendered the server's chat template.
    pub chat_template_reserve_tokens: u32,
    /// None uses the first assistant response's actual token count.
    pub fixed_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareGptCounts {
    pub records: u64,
    pub malformed_json: u64,
    pub missing_first_pair: u64,
    pub empty_text: u64,
    pub input_too_short: u64,
    pub input_too_long: u64,
    pub output_too_short: u64,
    pub output_too_long: u64,
    pub total_too_long: u64,
    pub eligible: u64,
    /// Informational; missing IDs do not exclude an otherwise valid pair.
    pub missing_original_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareGptSample {
    /// Zero-based source record (JSON array item or nonblank JSONL row).
    pub source_record_index: u64,
    pub original_id: Option<String>,
    pub phase: BenchmarkPhase,
    /// Index within this phase, matching the HTTP correlation header.
    pub request_index: u32,
    pub prompt_sha256: String,
    pub assistant_sha256: String,
    pub input_tokens: u32,
    pub reference_output_tokens: u32,
    pub requested_output_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareGptSelection {
    pub repeat_index: u32,
    pub rng_seed: u64,
    /// SHA-256 of the canonical serialized samples below, in dispatch order.
    pub selection_sha256: String,
    pub samples: Vec<ShareGptSample>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareGptDatasetEvidence {
    pub dataset: String,
    pub source_path: String,
    /// Exact raw file bytes, before parsing or filtering.
    pub source_sha256: String,
    pub source_format: String,
    pub tokenizer_sha256: String,
    pub filter: ShareGptFilter,
    pub counts: ShareGptCounts,
    pub prompt_seed: u64,
    pub sampling: String,
    pub ignore_eos: bool,
    pub enable_thinking: Option<bool>,
    pub repeats: Vec<ShareGptSelection>,
}
