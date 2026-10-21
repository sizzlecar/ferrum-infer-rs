//! Optional, content-bound reuse of the existing prepared request/evidence pair.

use super::{
    repeat_seed, sha256_hex, unbounded_tokenizer, validate, BenchServeCommand, BenchmarkPhase,
    FerrumError, PreparedRepeat, PromptCase, Result, ShareGptDatasetEvidence, ShareGptFilter,
    ShareGptReadLimits,
};
use ferrum_bench_core::dataset::ShareGptExclusionEvidence;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

// Bump these contracts when parsing/selection or tokenization semantics change.
// Neither a build commit nor a machine identity is part of cache validity.
pub(super) const SELECTOR_CONTRACT: &str =
    "first-pair-history-excluded-reservoir-std-rng-shuffle-prefix-v2";
pub(super) const TOKENIZATION_CONTRACT: &str =
    "encode-no-special-tokens-no-truncation-no-padding-v1";
const CACHE_SCHEMA: u32 = 1;

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheIdentity {
    schema: u32,
    selector_contract: String,
    tokenization_contract: String,
    source_sha256: String,
    tokenizer_sha256: String,
    effective_tokenizer_sha256: String,
    reader_config_sha256: String,
    filter: ShareGptFilter,
    exclusions: Option<ShareGptExclusionEvidence>,
    selection_pool_size: Option<u32>,
    effective_pool_size: usize,
    prompt_seed: u64,
    repeat_index: u32,
    rng_seed: u64,
    request_count: usize,
    warmup_requests: u32,
    ignore_eos: bool,
    enable_thinking: Option<bool>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CachedPrompt {
    text: String,
    input_tokens: u32,
    sha256: String,
    output_budget: Option<usize>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CachedRepeat {
    prompts: Vec<CachedPrompt>,
    evidence: ShareGptDatasetEvidence,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheEntry {
    identity: CacheIdentity,
    payload_sha256: String,
    payload: CachedRepeat,
}

pub(super) struct PreparedCache {
    directory: PathBuf,
    source_path: String,
    identity: CacheIdentity,
}

impl PreparedCache {
    pub(super) fn for_command(
        cmd: &BenchServeCommand,
        tokenizer: &tokenizers::Tokenizer,
        count: usize,
    ) -> Result<Option<Self>> {
        let Some(directory) = &cmd.sharegpt.sharegpt_prepared_cache_dir else {
            return Ok(None);
        };
        validate(cmd)?;
        let source = cmd.sharegpt_path.as_ref().expect("validated ShareGPT path");
        let tokenizer_bytes = fs::read(cmd.tokenizer.join("tokenizer.json"))
            .map_err(|e| FerrumError::model(format!("hash ShareGPT tokenizer: {e}")))?;
        // Serialize the actual normalized runtime tokenizer, not just its file.
        let effective_tokenizer = unbounded_tokenizer(tokenizer)?;
        let tokenizer_value = serde_json::to_value(&effective_tokenizer)
            .map_err(|e| FerrumError::model(format!("serialize ShareGPT tokenizer: {e}")))?;
        let prompt_seed = cmd.seed.expect("validated ShareGPT seed");
        Ok(Some(Self {
            directory: directory.clone(),
            source_path: source.display().to_string(),
            identity: CacheIdentity {
                schema: CACHE_SCHEMA,
                selector_contract: SELECTOR_CONTRACT.into(),
                tokenization_contract: TOKENIZATION_CONTRACT.into(),
                source_sha256: hash_file(source)?,
                tokenizer_sha256: sha256_hex(&tokenizer_bytes),
                effective_tokenizer_sha256: json_hash(&tokenizer_value)?,
                reader_config_sha256: json_hash(&ShareGptReadLimits::default())?,
                filter: cmd.sharegpt.filter(),
                exclusions: super::exclusions::Exclusions::load(
                    &cmd.sharegpt.sharegpt_exclude_path,
                )?
                .evidence(),
                selection_pool_size: cmd.sharegpt.sharegpt_selection_pool_size,
                effective_pool_size: cmd.sharegpt.pool_size(count)?,
                prompt_seed,
                repeat_index: 0,
                rng_seed: repeat_seed(prompt_seed, 0),
                request_count: count,
                warmup_requests: cmd.warmup_requests,
                ignore_eos: cmd.ignore_eos,
                enable_thinking: cmd.enable_thinking,
            },
        }))
    }

    pub(super) fn get_or_prepare(
        &self,
        repeat: u32,
        prepare: impl FnOnce() -> Result<PreparedRepeat>,
    ) -> Result<PreparedRepeat> {
        let mut identity = self.identity.clone();
        identity.repeat_index = repeat;
        identity.rng_seed = repeat_seed(identity.prompt_seed, repeat);
        let path = self
            .directory
            .join(format!("{}.json", json_hash(&identity)?));
        if let Some(entry) = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<CacheEntry>(&bytes).ok())
            .filter(|entry| {
                entry.identity == identity
                    && json_hash(&entry.payload).ok().as_ref() == Some(&entry.payload_sha256)
                    && valid_payload(&entry.payload, &identity)
            })
        {
            eprintln!("ShareGPT prepared cache: hit for repeat {}", repeat + 1);
            let mut evidence = entry.payload.evidence;
            // Identical bytes may move; retain the actual input path in reports.
            evidence.source_path.clone_from(&self.source_path);
            return Ok(PreparedRepeat {
                prompts: entry
                    .payload
                    .prompts
                    .into_iter()
                    .map(|prompt| PromptCase {
                        text: prompt.text,
                        input_tokens: prompt.input_tokens,
                        sha256: prompt.sha256,
                        output_budget: prompt.output_budget,
                    })
                    .collect(),
                evidence,
            });
        }
        eprintln!("ShareGPT prepared cache: miss for repeat {}", repeat + 1);
        let prepared = prepare()?;
        let payload = CachedRepeat {
            prompts: prepared
                .prompts
                .iter()
                .map(|prompt| CachedPrompt {
                    text: prompt.text.clone(),
                    input_tokens: prompt.input_tokens,
                    sha256: prompt.sha256.clone(),
                    output_budget: prompt.output_budget,
                })
                .collect(),
            evidence: prepared.evidence.clone(),
        };
        if !valid_payload(&payload, &identity) {
            return Err(FerrumError::model(
                "ShareGPT inputs or preparation contract changed while preparing cache",
            ));
        }
        let entry = CacheEntry {
            payload_sha256: json_hash(&payload)?,
            identity,
            payload,
        };
        let bytes = serde_json::to_vec(&entry)
            .map_err(|e| FerrumError::model(format!("serialize ShareGPT prepared cache: {e}")))?;
        if let Err(error) = atomic_write(&path, &bytes) {
            // Cache availability must not change the selected workload.
            eprintln!(
                "ShareGPT prepared cache: could not publish {}: {error}",
                path.display()
            );
        }
        Ok(prepared)
    }
}

fn valid_payload(payload: &CachedRepeat, identity: &CacheIdentity) -> bool {
    let evidence = &payload.evidence;
    let sampling = if identity.selection_pool_size.is_some() {
        "reservoir_without_replacement_then_shuffle_prefix_v1"
    } else {
        "reservoir_without_replacement_then_shuffle_v1"
    };
    if evidence.dataset != "sharegpt"
        || evidence.source_sha256 != identity.source_sha256
        || evidence.tokenizer_sha256 != identity.tokenizer_sha256
        || evidence.filter != identity.filter
        || evidence.exclusions != identity.exclusions
        || evidence.selection_pool_size != identity.selection_pool_size
        || evidence.prompt_seed != identity.prompt_seed
        || evidence.ignore_eos != identity.ignore_eos
        || evidence.enable_thinking != identity.enable_thinking
        || evidence.sampling != sampling
        || !matches!(evidence.source_format.as_str(), "json_array" | "jsonl")
        || evidence.counts.eligible < identity.effective_pool_size as u64
        || evidence.counts.records < evidence.counts.eligible
        || evidence.repeats.len() != 1
        || payload.prompts.len() != identity.request_count
    {
        return false;
    }
    let selection = &evidence.repeats[0];
    if selection.repeat_index != identity.repeat_index
        || selection.rng_seed != identity.rng_seed
        || selection.samples.len() != identity.request_count
        || json_hash(&selection.samples).ok().as_ref() != Some(&selection.selection_sha256)
    {
        return false;
    }
    let mut seen = HashSet::with_capacity(identity.request_count);
    payload
        .prompts
        .iter()
        .zip(&selection.samples)
        .enumerate()
        .all(|(index, (prompt, sample))| {
            let warm = index < identity.warmup_requests as usize;
            let expected_index = if warm {
                index
            } else {
                index - identity.warmup_requests as usize
            };
            let filter = &identity.filter;
            sample.phase
                == if warm {
                    BenchmarkPhase::Warmup
                } else {
                    BenchmarkPhase::Measured
                }
                && sample.request_index as usize == expected_index
                && sample.source_record_index < evidence.counts.records
                && seen.insert(sample.source_record_index)
                && !prompt.text.trim().is_empty()
                && sha256_hex(prompt.text.as_bytes()) == prompt.sha256
                && prompt.sha256 == sample.prompt_sha256
                && prompt.input_tokens == sample.input_tokens
                && prompt.output_budget == Some(sample.requested_output_tokens as usize)
                && sample.requested_output_tokens
                    == filter
                        .fixed_output_tokens
                        .unwrap_or(sample.reference_output_tokens)
                && sample.input_tokens >= filter.min_input_tokens
                && filter
                    .max_input_tokens
                    .is_none_or(|n| sample.input_tokens <= n)
                && sample.reference_output_tokens >= filter.min_output_tokens
                && filter
                    .max_output_tokens
                    .is_none_or(|n| sample.reference_output_tokens <= n)
                && filter.max_total_tokens.is_none_or(|n| {
                    u64::from(sample.input_tokens)
                        + u64::from(sample.requested_output_tokens)
                        + u64::from(filter.chat_template_reserve_tokens)
                        <= u64::from(n)
                })
        })
}

fn json_hash(value: &impl Serialize) -> Result<String> {
    serde_json::to_vec(value)
        .map(|bytes| sha256_hex(&bytes))
        .map_err(|e| FerrumError::model(format!("serialize ShareGPT cache identity: {e}")))
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .map_err(|e| FerrumError::model(format!("open ShareGPT {}: {e}", path.display())))?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|e| FerrumError::model(format!("hash ShareGPT {}: {e}", path.display())))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let directory = path.parent().expect("cache file has a parent");
    fs::create_dir_all(directory)?;
    let temporary = directory.join(format!(".{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
pub(super) fn refresh_test_payload_checksum(value: &mut serde_json::Value) {
    let mut entry: CacheEntry = serde_json::from_value(value.clone()).unwrap();
    for selection in &mut entry.payload.evidence.repeats {
        selection.selection_sha256 = json_hash(&selection.samples).unwrap();
    }
    entry.payload_sha256 = json_hash(&entry.payload).unwrap();
    *value = serde_json::to_value(entry).unwrap();
}
