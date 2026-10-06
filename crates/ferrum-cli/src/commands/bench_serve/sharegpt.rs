//! Streaming, non-truncating first-pair ShareGPT workload selection.

use super::{sha256_hex, BenchServeCommand, PromptCase};
use clap::Args;
use ferrum_bench_core::dataset::records::{
    sharegpt_first_pair, sharegpt_original_id, visit_sharegpt_records, ShareGptReadLimits,
    ShareGptRecord,
};
use ferrum_bench_core::dataset::{
    ShareGptCounts, ShareGptDatasetEvidence, ShareGptFilter, ShareGptSample, ShareGptSelection,
};
use ferrum_bench_core::BenchmarkPhase;
use ferrum_types::{FerrumError, Result};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use serde_json::Value;
use std::fs::File;

#[derive(Args, Clone, Debug)]
pub struct ShareGptArgs {
    /// Minimum raw first-human prompt tokens; records are filtered, never cut.
    #[arg(long, default_value_t = 4)]
    pub sharegpt_min_input_tokens: u32,
    /// Maximum raw first-human prompt tokens (before the server template).
    #[arg(long, default_value_t = 1024)]
    pub sharegpt_max_input_tokens: u32,
    /// Minimum raw first-assistant response tokens.
    #[arg(long, default_value_t = 4)]
    pub sharegpt_min_output_tokens: u32,
    /// Optional maximum raw first-assistant response tokens.
    #[arg(long)]
    pub sharegpt_max_output_tokens: Option<u32>,
    /// Maximum raw prompt + requested output + explicit template reserve.
    #[arg(long, default_value_t = 2048)]
    pub sharegpt_max_total_tokens: u32,
    /// Estimated template overhead for filtering; not an exact server render.
    #[arg(long, default_value_t = 0)]
    pub sharegpt_chat_template_reserve_tokens: u32,
    /// Explicitly replace reference-answer output lengths with a fixed budget.
    #[arg(long)]
    pub sharegpt_fixed_output_tokens: Option<u32>,
}

impl Default for ShareGptArgs {
    fn default() -> Self {
        Self {
            sharegpt_min_input_tokens: 4,
            sharegpt_max_input_tokens: 1024,
            sharegpt_min_output_tokens: 4,
            sharegpt_max_output_tokens: None,
            sharegpt_max_total_tokens: 2048,
            sharegpt_chat_template_reserve_tokens: 0,
            sharegpt_fixed_output_tokens: None,
        }
    }
}

impl ShareGptArgs {
    fn filter(&self) -> ShareGptFilter {
        ShareGptFilter {
            min_input_tokens: self.sharegpt_min_input_tokens,
            max_input_tokens: Some(self.sharegpt_max_input_tokens),
            min_output_tokens: self.sharegpt_min_output_tokens,
            max_output_tokens: self.sharegpt_max_output_tokens,
            max_total_tokens: Some(self.sharegpt_max_total_tokens),
            chat_template_reserve_tokens: self.sharegpt_chat_template_reserve_tokens,
            fixed_output_tokens: self.sharegpt_fixed_output_tokens,
        }
    }
}

pub(super) fn validate(cmd: &BenchServeCommand) -> Result<()> {
    if cmd.dataset != "sharegpt" {
        return Ok(());
    }
    if cmd.sharegpt_path.is_none() || cmd.seed.is_none() {
        return Err(FerrumError::model(
            "--dataset sharegpt requires --sharegpt-path and --seed",
        ));
    }
    let f = cmd.sharegpt.filter();
    if f.min_input_tokens == 0
        || f.min_output_tokens == 0
        || f.max_input_tokens.is_some_and(|n| n < f.min_input_tokens)
        || f.max_output_tokens.is_some_and(|n| n < f.min_output_tokens)
        || f.fixed_output_tokens == Some(0)
        || f.max_total_tokens
            .is_some_and(|n| n <= f.chat_template_reserve_tokens)
    {
        return Err(FerrumError::model(
            "invalid ShareGPT token bounds or output budget",
        ));
    }
    Ok(())
}

/// Deliberately independent of client concurrency and server/backend identity.
pub(super) fn repeat_seed(seed: u64, repeat: u32) -> u64 {
    seed ^ ((repeat as u64) << 32) ^ 0x5348_4152_4547_5054
}

pub(super) struct PreparedRepeat {
    pub prompts: Vec<PromptCase>,
    pub evidence: ShareGptDatasetEvidence,
}

/// Retain only the selected requests. Reusing this plan across cells both
/// freezes their workload and avoids retokenizing the source for every C.
pub(super) fn prepare(cmd: &BenchServeCommand) -> Result<Option<Vec<PreparedRepeat>>> {
    if cmd.dataset != "sharegpt" {
        return Ok(None);
    }
    let tokenizer = tokenizers::Tokenizer::from_file(cmd.tokenizer.join("tokenizer.json"))
        .map_err(|e| FerrumError::model(format!("load ShareGPT tokenizer: {e}")))?;
    let count =
        cmd.num_prompts
            .checked_add(cmd.warmup_requests)
            .ok_or_else(|| FerrumError::model("ShareGPT sample count overflow"))? as usize;
    let mut prepared = Vec::with_capacity(cmd.n_repeats as usize);
    let mut identity = None;
    for repeat in 0..cmd.n_repeats {
        let (prompts, evidence) = load(cmd, &tokenizer, repeat, count)?;
        append_evidence(&mut identity, evidence.clone())?;
        prepared.push(PreparedRepeat { prompts, evidence });
    }
    Ok(Some(prepared))
}

pub(super) fn load(
    cmd: &BenchServeCommand,
    tok: &tokenizers::Tokenizer,
    repeat: u32,
    count: usize,
) -> Result<(Vec<PromptCase>, ShareGptDatasetEvidence)> {
    validate(cmd)?;
    let mut unbounded_tokenizer = tok.clone();
    unbounded_tokenizer
        .with_truncation(None)
        .map_err(|e| FerrumError::model(format!("disable ShareGPT tokenizer truncation: {e}")))?;
    unbounded_tokenizer.with_padding(None);
    let tok = &unbounded_tokenizer;
    let path = cmd.sharegpt_path.as_ref().expect("validated ShareGPT path");
    let seed = cmd.seed.expect("validated ShareGPT seed");
    let file = File::open(path)
        .map_err(|e| FerrumError::model(format!("open ShareGPT {}: {e}", path.display())))?;
    let mut rng = StdRng::seed_from_u64(repeat_seed(seed, repeat));
    let filter = cmd.sharegpt.filter();
    let mut selector = Selector {
        tok,
        filter: &filter,
        rng: &mut rng,
        count,
        counts: ShareGptCounts::default(),
        selected: Vec::with_capacity(count),
    };
    let receipt = visit_sharegpt_records(file, ShareGptReadLimits::default(), |index, record| {
        if index != selector.counts.records {
            return Err(FerrumError::internal(
                "ShareGPT source record order differs",
            ));
        }
        match record {
            ShareGptRecord::Value(value) => selector.consider(value),
            ShareGptRecord::MalformedJsonLine => {
                selector.counts.records += 1;
                selector.counts.malformed_json += 1;
                Ok(())
            }
        }
    })?;
    let source_sha256 = receipt.source_sha256;
    if selector.selected.len() != count {
        return Err(FerrumError::model(format!(
            "ShareGPT has {} eligible records, but {count} distinct warmup+measured records are required; no repetition or truncation is allowed; source_sha256={source_sha256}; counts={}",
            selector.counts.eligible,
            serde_json::to_string(&selector.counts).expect("serialize counts")
        )));
    }
    selector.selected.shuffle(selector.rng);
    let mut prompts = Vec::with_capacity(count);
    let mut samples = Vec::with_capacity(count);
    for (index, (prompt, mut sample)) in selector.selected.into_iter().enumerate() {
        let warmup = index < cmd.warmup_requests as usize;
        sample.phase = if warmup {
            BenchmarkPhase::Warmup
        } else {
            BenchmarkPhase::Measured
        };
        sample.request_index = u32::try_from(if warmup {
            index
        } else {
            index - cmd.warmup_requests as usize
        })
        .map_err(|_| FerrumError::model("ShareGPT request index overflow"))?;
        prompts.push(prompt);
        samples.push(sample);
    }
    let selection_sha256 = sha256_hex(&serde_json::to_vec(&samples).expect("serialize selection"));
    let counts = selector.counts;
    let tokenizer_bytes = std::fs::read(cmd.tokenizer.join("tokenizer.json"))
        .map_err(|e| FerrumError::model(format!("hash ShareGPT tokenizer: {e}")))?;
    let evidence = ShareGptDatasetEvidence {
        dataset: "sharegpt".into(),
        source_path: path.display().to_string(),
        source_sha256,
        source_format: receipt.source_format.into(),
        tokenizer_sha256: sha256_hex(&tokenizer_bytes),
        filter,
        counts,
        prompt_seed: seed,
        sampling: "reservoir_without_replacement_then_shuffle_v1".into(),
        ignore_eos: cmd.ignore_eos,
        enable_thinking: cmd.enable_thinking,
        repeats: vec![ShareGptSelection {
            repeat_index: repeat,
            rng_seed: repeat_seed(seed, repeat),
            selection_sha256,
            samples,
        }],
    };
    Ok((prompts, evidence))
}

pub(super) fn append_evidence(
    all: &mut Option<ShareGptDatasetEvidence>,
    mut next: ShareGptDatasetEvidence,
) -> Result<()> {
    if let Some(first) = all {
        if first.source_sha256 != next.source_sha256
            || first.tokenizer_sha256 != next.tokenizer_sha256
            || first.counts != next.counts
            || first.filter != next.filter
        {
            return Err(FerrumError::model(
                "ShareGPT dataset/tokenizer changed between repeats",
            ));
        }
        first.repeats.append(&mut next.repeats);
    } else {
        *all = Some(next);
    }
    Ok(())
}

struct Selector<'a> {
    tok: &'a tokenizers::Tokenizer,
    filter: &'a ShareGptFilter,
    rng: &'a mut StdRng,
    count: usize,
    counts: ShareGptCounts,
    selected: Vec<(PromptCase, ShareGptSample)>,
}

impl Selector<'_> {
    fn consider(&mut self, value: Value) -> Result<()> {
        let index = self.counts.records;
        self.counts.records += 1;
        let original_id = sharegpt_original_id(&value);
        if original_id.is_none() {
            self.counts.missing_original_id += 1;
        }
        let Some(pair) = sharegpt_first_pair(&value) else {
            self.counts.missing_first_pair += 1;
            return Ok(());
        };
        let prompt = pair.prompt;
        let assistant = pair.assistant;
        if prompt.trim().is_empty() || assistant.trim().is_empty() {
            self.counts.empty_text += 1;
            return Ok(());
        }
        let input = token_length(self.tok, prompt)?;
        let output = token_length(self.tok, assistant)?;
        let requested = self.filter.fixed_output_tokens.unwrap_or(output);
        if input < self.filter.min_input_tokens {
            self.counts.input_too_short += 1;
        } else if self.filter.max_input_tokens.is_some_and(|n| input > n) {
            self.counts.input_too_long += 1;
        } else if output < self.filter.min_output_tokens {
            self.counts.output_too_short += 1;
        } else if self.filter.max_output_tokens.is_some_and(|n| output > n) {
            self.counts.output_too_long += 1;
        } else if self.filter.max_total_tokens.is_some_and(|n| {
            u64::from(input)
                + u64::from(requested)
                + u64::from(self.filter.chat_template_reserve_tokens)
                > u64::from(n)
        }) {
            self.counts.total_too_long += 1;
        } else {
            self.counts.eligible += 1;
            let slot = if self.selected.len() < self.count {
                self.selected.len() as u64
            } else {
                self.rng.random_range(0..self.counts.eligible)
            };
            if slot < self.count as u64 {
                let prompt_sha256 = sha256_hex(prompt.as_bytes());
                let sample = ShareGptSample {
                    source_record_index: index,
                    original_id,
                    phase: BenchmarkPhase::Measured,
                    request_index: 0,
                    prompt_sha256: prompt_sha256.clone(),
                    assistant_sha256: sha256_hex(assistant.as_bytes()),
                    input_tokens: input,
                    reference_output_tokens: output,
                    requested_output_tokens: requested,
                };
                let candidate = (
                    PromptCase {
                        text: prompt.to_owned(),
                        input_tokens: input,
                        sha256: prompt_sha256,
                        output_budget: Some(requested as usize),
                    },
                    sample,
                );
                if self.selected.len() < self.count {
                    self.selected.push(candidate);
                } else {
                    self.selected[slot as usize] = candidate;
                }
            }
        }
        Ok(())
    }
}

fn token_length(tok: &tokenizers::Tokenizer, text: &str) -> Result<u32> {
    let encoding = tok
        .encode(text, false)
        .map_err(|e| FerrumError::model(format!("tokenize ShareGPT text: {e}")))?;
    u32::try_from(encoding.len()).map_err(|_| FerrumError::model("ShareGPT token count overflow"))
}

#[cfg(test)]
mod tests;
