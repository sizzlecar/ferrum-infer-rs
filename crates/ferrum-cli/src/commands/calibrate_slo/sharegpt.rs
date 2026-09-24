//! Replay a frozen selection; never resample or turn raw human text into a
//! supposedly rendered prompt. Model Chat conversion lives in `inputs`.
use super::*;
use ferrum_bench_core::dataset::records::{
    sharegpt_first_pair, sharegpt_original_id, visit_sharegpt_records, ShareGptReadLimits,
    ShareGptRecord,
};
use ferrum_bench_core::dataset::{ShareGptDatasetEvidence, ShareGptSample};
use ferrum_bench_core::env::HttpRequestSampling;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Read, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FrozenShareGpt {
    pub report_path: PathBuf,
    pub report_sha256: [u8; 32],
    pub dataset_path: PathBuf,
    /// Exact tokenizer.json file used by the frozen client selection.
    pub tokenizer_path: PathBuf,
    pub repeat_index: u32,
    pub selection_sha256: [u8; 32],
    pub request_policy: RequestPolicy,
    #[serde(default)]
    pub read_limits: ShareGptReadLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RequestPolicy {
    /// Literal requested public model name used by the HTTP body and codec.
    /// It is not the internal model source or template lookup identity.
    pub requested_model_name: String,
    pub sampling: HttpRequestSampling,
    pub ignore_eos: bool,
    #[serde(deserialize_with = "required_option")]
    pub enable_thinking: Option<bool>,
    /// Required explicit null/value: historical dataset DTOs omit this field.
    /// This is a manifest declaration, not an attestation of an old HTTP body.
    #[serde(deserialize_with = "required_option")]
    pub reasoning_effort: Option<ferrum_types::ReasoningEffort>,
    #[serde(deserialize_with = "required_option")]
    pub server_default_enable_thinking: Option<bool>,
    pub interleaved_system_coalescing: bool,
}

fn required_option<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error> {
    Option::<T>::deserialize(deserializer)
}

impl FrozenShareGpt {
    pub(super) fn validate(&self) -> Result<()> {
        if self.report_path.as_os_str().is_empty()
            || self.dataset_path.as_os_str().is_empty()
            || self.tokenizer_path.as_os_str().is_empty()
            || self.report_sha256 == [0; 32]
            || self.selection_sha256 == [0; 32]
            || self.request_policy.requested_model_name.is_empty()
            || self.request_policy.requested_model_name.len() > 4096
            || self.request_policy.requested_model_name.trim()
                != self.request_policy.requested_model_name
        {
            return Err(FerrumError::config(
                "frozen ShareGPT needs paths and pinned digests",
            ));
        }
        self.request_policy.sampling.validate()?;
        self.read_limits.validate()
    }

    pub(super) fn resolve_paths(&mut self, parent: &Path) {
        for path in [
            &mut self.report_path,
            &mut self.dataset_path,
            &mut self.tokenizer_path,
        ] {
            if path.is_relative() {
                *path = parent.join(&*path);
            }
        }
    }
}

pub(super) struct RecoveredPrompt {
    pub sample: ShareGptSample,
    pub text: String,
}

pub(super) struct Recovered {
    pub prompts: Vec<RecoveredPrompt>,
    pub provenance: serde_json::Value,
}

#[derive(Deserialize)]
struct ReportInputs {
    // bench-serve can append an arbitrary --tag to this display label. It
    // cannot attest a wire name, and must not be split with name heuristics.
    model: Option<String>,
    dataset_evidence: Option<ShareGptDatasetEvidence>,
    env: ReportEnvironment,
}

#[derive(Deserialize)]
struct ReportEnvironment {
    http_request_sampling: Option<HttpRequestSampling>,
}

pub(super) fn load(source: &FrozenShareGpt) -> Result<Recovered> {
    source.validate()?;
    let report_bytes = read_small(&source.report_path)?;
    if <[u8; 32]>::from(Sha256::digest(&report_bytes)) != source.report_sha256 {
        return Err(FerrumError::config("frozen ShareGPT report digest differs"));
    }
    let report: ReportInputs = serde_json::from_slice(&report_bytes)
        .map_err(|error| FerrumError::config(format!("read frozen ShareGPT evidence: {error}")))?;
    let evidence = report
        .dataset_evidence
        .ok_or_else(|| FerrumError::config("report lacks frozen ShareGPT evidence"))?;
    if evidence.dataset != "sharegpt"
        || report.env.http_request_sampling != Some(source.request_policy.sampling)
        || evidence.ignore_eos != source.request_policy.ignore_eos
        || evidence.enable_thinking != source.request_policy.enable_thinking
    {
        return Err(FerrumError::config(
            "ShareGPT request policy differs or is unknown in report",
        ));
    }
    let mut matches = evidence
        .repeats
        .iter()
        .filter(|repeat| repeat.repeat_index == source.repeat_index);
    let selection = matches
        .next()
        .ok_or_else(|| FerrumError::config("ShareGPT repeat is absent"))?;
    if matches.next().is_some() || selection.samples.is_empty() || selection.samples.len() > 4096 {
        return Err(FerrumError::config(
            "ShareGPT selection is duplicate, empty or exceeds 4096 samples",
        ));
    }
    let selection_bytes = serde_json::to_vec(&selection.samples)
        .map_err(|error| FerrumError::config(format!("serialize frozen selection: {error}")))?;
    let selection_hash: [u8; 32] = Sha256::digest(&selection_bytes).into();
    if selection_hash != source.selection_sha256
        || hex(&selection_hash) != selection.selection_sha256
    {
        return Err(FerrumError::config(
            "frozen ShareGPT selection digest differs",
        ));
    }
    let tokenizer_bytes = read_small(&source.tokenizer_path)?;
    if hex(&Sha256::digest(&tokenizer_bytes)) != evidence.tokenizer_sha256 {
        return Err(FerrumError::config("ShareGPT tokenizer digest differs"));
    }
    let mut tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_bytes)
        .map_err(|error| FerrumError::config(format!("load frozen tokenizer: {error}")))?;
    tokenizer.with_truncation(None).map_err(|error| {
        FerrumError::config(format!("disable frozen tokenizer truncation: {error}"))
    })?;
    tokenizer.with_padding(None);
    let mut indices = BTreeMap::new();
    let mut phase_next = BTreeMap::new();
    for (position, sample) in selection.samples.iter().enumerate() {
        let next = phase_next.entry(sample.phase.as_str()).or_insert(0u32);
        if sample.request_index != *next
            || indices
                .insert(sample.source_record_index, position)
                .is_some()
        {
            return Err(FerrumError::config(
                "frozen selection has duplicate or unordered request identities",
            ));
        }
        *next += 1;
    }
    let mut recovered: Vec<Option<RecoveredPrompt>> = std::iter::repeat_with(|| None)
        .take(selection.samples.len())
        .collect();
    let mut retained_bytes = 0usize;
    let file = std::fs::File::open(&source.dataset_path)
        .map_err(|error| FerrumError::config(format!("open frozen dataset: {error}")))?;
    let receipt = visit_sharegpt_records(file, source.read_limits, |index, record| {
        let Some(&position) = indices.get(&index) else {
            return Ok(());
        };
        let sample = &selection.samples[position];
        let ShareGptRecord::Value(value) = record else {
            return Err(FerrumError::config("selected ShareGPT record is malformed"));
        };
        let pair = sharegpt_first_pair(&value)
            .ok_or_else(|| FerrumError::config("selected ShareGPT record lacks its first pair"))?;
        if sharegpt_original_id(&value) != sample.original_id
            || hex(&Sha256::digest(pair.prompt.as_bytes())) != sample.prompt_sha256
            || hex(&Sha256::digest(pair.assistant.as_bytes())) != sample.assistant_sha256
        {
            return Err(FerrumError::config(
                "selected ShareGPT text or identity differs",
            ));
        }
        let length = |text: &str| -> Result<u32> {
            let encoded = tokenizer.encode(text, false).map_err(|error| {
                FerrumError::config(format!("tokenize frozen ShareGPT pair: {error}"))
            })?;
            u32::try_from(encoded.len())
                .map_err(|_| FerrumError::config("ShareGPT token count overflow"))
        };
        let input = length(pair.prompt)?;
        let output = length(pair.assistant)?;
        let requested = evidence.filter.fixed_output_tokens.unwrap_or(output);
        if input != sample.input_tokens
            || output != sample.reference_output_tokens
            || requested != sample.requested_output_tokens
            || requested == 0
            || requested > 1_048_576
            || input < evidence.filter.min_input_tokens
            || output < evidence.filter.min_output_tokens
            || evidence
                .filter
                .max_input_tokens
                .is_some_and(|limit| input > limit)
            || evidence
                .filter
                .max_output_tokens
                .is_some_and(|limit| output > limit)
            || evidence.filter.max_total_tokens.is_some_and(|limit| {
                u64::from(input)
                    + u64::from(requested)
                    + u64::from(evidence.filter.chat_template_reserve_tokens)
                    > u64::from(limit)
            })
        {
            return Err(FerrumError::config(
                "selected ShareGPT lengths/filter/output policy differ",
            ));
        }
        retained_bytes = retained_bytes
            .checked_add(pair.prompt.len())
            .filter(|count| *count <= 64 * 1024 * 1024)
            .ok_or_else(|| {
                FerrumError::resource_exhausted("selected ShareGPT prompts exceed 64 MiB")
            })?;
        if pair.prompt.is_empty() || pair.prompt.len() > 1024 * 1024 {
            return Err(FerrumError::config(
                "selected ShareGPT prompt exceeds calibration bound",
            ));
        }
        recovered[position] = Some(RecoveredPrompt {
            sample: sample.clone(),
            text: pair.prompt.to_owned(),
        });
        Ok(())
    })?;
    if receipt.source_sha256 != evidence.source_sha256
        || receipt.source_format != evidence.source_format
        || receipt.records != evidence.counts.records
    {
        return Err(FerrumError::config(
            "ShareGPT dataset bytes, format or record count differ",
        ));
    }
    let prompts = recovered
        .into_iter()
        .map(|row| {
            row.ok_or_else(|| FerrumError::config("selected ShareGPT source record was not found"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Recovered {
        prompts,
        provenance: serde_json::json!({
            "kind":"frozen_sharegpt_v2", "report_sha256":hex(&source.report_sha256),
            "dataset_sha256":receipt.source_sha256,"tokenizer_sha256":evidence.tokenizer_sha256,
            "repeat_index":source.repeat_index,"selection_sha256":hex(&selection_hash),
            "request_policy":source.request_policy,
            "report_model_label":report.model,
            "requested_model_name_evidence":"explicit_manifest_declaration_report_label_may_include_tag",
            "reasoning_effort_evidence":"explicit_manifest_declaration_not_historical_body_attestation",
            "historical_server_token_digest":"unavailable_unless_separately_recorded",
        }),
    })
}

fn read_small(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|error| FerrumError::config(format!("open {}: {error}", path.display())))?
        .take(64 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| FerrumError::config(format!("read {}: {error}", path.display())))?;
    if bytes.len() > 64 * 1024 * 1024 {
        return Err(FerrumError::config(
            "frozen report/tokenizer exceeds 64 MiB",
        ));
    }
    Ok(bytes)
}

pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests;
