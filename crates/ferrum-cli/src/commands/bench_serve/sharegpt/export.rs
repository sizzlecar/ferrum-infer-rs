//! Offline export of a provenance-bound, explicitly new ShareGPT source pool.

use super::{
    load, sha256_hex, sharegpt_first_pair, sharegpt_original_id, unbounded_tokenizer,
    visit_sharegpt_records, BenchServeCommand, FerrumError, Result, ShareGptDatasetEvidence,
    ShareGptReadLimits, ShareGptSample,
};
use ferrum_bench_core::dataset::records::visit_sharegpt_raw_records;
use ferrum_bench_core::env::HttpRequestSampling;
use ferrum_server::chat_template::ReasoningEffort;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub(super) fn validate(cmd: &BenchServeCommand) -> Result<()> {
    let Some(_) = &cmd.sharegpt.export_sharegpt_pool else {
        if cmd.sharegpt.sharegpt_source_sha256.is_some() {
            return Err(FerrumError::config(
                "--sharegpt-source-sha256 is only valid with --export-sharegpt-pool",
            ));
        }
        return Ok(());
    };
    if cmd.dataset != "sharegpt"
        || cmd.sharegpt_path.is_none()
        || cmd.seed.is_none()
        || cmd.n_repeats != 1
        || !cmd
            .sharegpt
            .sharegpt_selection_pool_size
            .is_some_and(|n| n > 0)
    {
        return Err(FerrumError::config(
            "ShareGPT pool export requires --dataset sharegpt, --sharegpt-path, --seed, a positive explicit --sharegpt-selection-pool-size and --n-repeats 1",
        ));
    }
    if cmd.sharegpt.sharegpt_prepared_cache_dir.is_some() {
        return Err(FerrumError::config(
            "ShareGPT pool export reads the original source and cannot use a prepared cache",
        ));
    }
    let pin = cmd.sharegpt.sharegpt_source_sha256.as_deref().unwrap_or("");
    if pin.len() != 64 || !pin.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(FerrumError::config(
            "ShareGPT pool export requires a 64-digit --sharegpt-source-sha256",
        ));
    }
    Ok(())
}

#[derive(Serialize)]
struct TokenizerBinding {
    path: PathBuf,
    sha256: String,
    effective_tokenizer_sha256: String,
}

#[derive(Serialize)]
struct RequestPolicy {
    sampling: HttpRequestSampling,
    ignore_eos: bool,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
}

pub(super) struct SelectedPool {
    source: PathBuf,
    evidence: ShareGptDatasetEvidence,
    tokenizer: TokenizerBinding,
    request_policy: RequestPolicy,
}

pub(super) fn select_pool(cmd: &BenchServeCommand) -> Result<SelectedPool> {
    super::validate(cmd)?;
    let pool_size = cmd.sharegpt.sharegpt_selection_pool_size.ok_or_else(|| {
        FerrumError::config("ShareGPT pool export requires an explicit pool size")
    })?;
    let tokenizer_path = cmd.tokenizer.join("tokenizer.json");
    let tokenizer_bytes = fs::read(&tokenizer_path)
        .map_err(|error| io_error("read tokenizer", &tokenizer_path, error))?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|error| FerrumError::model(format!("load ShareGPT export tokenizer: {error}")))?;
    let tokenizer_sha256 = sha256_hex(&tokenizer_bytes);
    let normalized = serde_json::to_value(unbounded_tokenizer(&tokenizer)?)
        .map_err(|error| FerrumError::model(format!("serialize export tokenizer: {error}")))?;
    let effective_tokenizer_sha256 = sha256_hex(
        &serde_json::to_vec(&normalized)
            .map_err(|error| FerrumError::model(format!("hash export tokenizer: {error}")))?,
    );
    // Freeze the whole reservoir. Warmup/measured prefixes belong to the later
    // benchmark on the child source, not to this offline parent selection.
    let mut selection = cmd.clone();
    selection.num_prompts = pool_size;
    selection.warmup_requests = 0;
    let (_, evidence) = load(&selection, &tokenizer, 0, pool_size as usize)?;
    let expected = cmd
        .sharegpt
        .sharegpt_source_sha256
        .as_deref()
        .expect("validated export pin");
    if !evidence.source_sha256.eq_ignore_ascii_case(expected) {
        return Err(FerrumError::model(
            "ShareGPT export parent SHA256 differs from the required pin",
        ));
    }
    if evidence.tokenizer_sha256 != tokenizer_sha256 {
        return Err(FerrumError::model(
            "ShareGPT tokenizer changed during pool selection",
        ));
    }
    Ok(SelectedPool {
        source: cmd.sharegpt_path.clone().expect("validated source"),
        evidence,
        tokenizer: TokenizerBinding {
            path: tokenizer_path,
            sha256: tokenizer_sha256,
            effective_tokenizer_sha256,
        },
        request_policy: RequestPolicy {
            sampling: cmd.sampling.request_sampling(),
            ignore_eos: cmd.ignore_eos,
            enable_thinking: cmd.enable_thinking,
            reasoning_effort: cmd.reasoning_effort,
        },
    })
}

#[derive(Serialize)]
struct RecordMapping<'a> {
    child_record_index: u64,
    parent_source_record_index: u64,
    original_id: &'a Option<String>,
    prompt_sha256: &'a str,
    assistant_sha256: &'a str,
    input_tokens: u32,
    reference_output_tokens: u32,
    requested_output_tokens: u32,
}

#[derive(Serialize)]
struct ChildSource {
    file: &'static str,
    source_format: &'static str,
    source_sha256: String,
    source_bytes: u64,
    records: u64,
}

#[derive(Serialize)]
struct ExportReceipt<'a> {
    schema_version: u32,
    selection_semantics: &'static str,
    serialization_order: &'static str,
    selector_contract: &'static str,
    tokenization_contract: &'static str,
    reader_limits: ShareGptReadLimits,
    parent: &'a ShareGptDatasetEvidence,
    tokenizer: &'a TokenizerBinding,
    request_policy: &'a RequestPolicy,
    child: ChildSource,
    records: Vec<RecordMapping<'a>>,
}

pub(super) fn run(cmd: &BenchServeCommand) -> Result<()> {
    super::validate(cmd)?;
    let directory = cmd
        .sharegpt
        .export_sharegpt_pool
        .as_ref()
        .ok_or_else(|| FerrumError::config("ShareGPT export destination is missing"))?;
    // Reject existing directories (including symlinks) before scanning the
    // parent. create_dir below repeats the exclusive check at publication.
    match fs::symlink_metadata(directory) {
        Ok(_) => {
            return Err(FerrumError::config(
                "ShareGPT export destination already exists",
            ))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("inspect export destination", directory, error)),
    }
    let selected = select_pool(cmd)?;
    publish_pool(directory, &selected)?;
    eprintln!(
        "ShareGPT fixed derived pool exported to {}; re-selection uses the child source identity",
        directory.display()
    );
    Ok(())
}

pub(super) fn publish_pool(directory: &Path, selected: &SelectedPool) -> Result<()> {
    let mut publication = Publication::create(directory)?;
    match write_pool(directory, selected) {
        Ok(()) => {
            publication.complete = true;
            Ok(())
        }
        Err(error) => {
            if let Err(cleanup) = fs::remove_dir_all(directory) {
                return Err(FerrumError::model(format!(
                    "{error}; remove incomplete ShareGPT export {}: {cleanup}; no completed receipt was published",
                    directory.display()
                )));
            }
            publication.complete = true;
            Err(error)
        }
    }
}

fn write_pool(directory: &Path, selected: &SelectedPool) -> Result<()> {
    let samples = &selected.evidence.repeats[0].samples;
    let mut remaining = BTreeMap::new();
    for sample in samples {
        if remaining
            .insert(sample.source_record_index, sample)
            .is_some()
        {
            return Err(FerrumError::internal(
                "duplicate ShareGPT parent record index",
            ));
        }
    }
    let pool_path = directory.join(".pool.json.tmp");
    let mut output = BufWriter::new(create_file(&pool_path)?);
    output
        .write_all(b"[\n")
        .map_err(|error| io_error("write pool", &pool_path, error))?;
    let mut records = Vec::with_capacity(samples.len());
    let input = File::open(&selected.source)
        .map_err(|error| io_error("reopen parent source", &selected.source, error))?;
    let parent =
        visit_sharegpt_raw_records(input, ShareGptReadLimits::default(), |index, record| {
            let Some(sample) = remaining.remove(&index) else {
                return Ok(());
            };
            let raw = record
                .ok_or_else(|| FerrumError::model("selected ShareGPT record became malformed"))?;
            let value = serde_json::from_str(raw).map_err(|error| {
                FerrumError::model(format!("parse selected ShareGPT record: {error}"))
            })?;
            check_record(&value, sample)?;
            if !records.is_empty() {
                output
                    .write_all(b",\n")
                    .map_err(|error| io_error("write pool", &pool_path, error))?;
            }
            // Preserve the original record bytes, including exact numbers outside
            // i64/u64, all later turns, metadata and duplicate object fields.
            output
                .write_all(raw.as_bytes())
                .map_err(|error| io_error("write original pool record", &pool_path, error))?;
            records.push(RecordMapping {
                child_record_index: records.len() as u64,
                parent_source_record_index: index,
                original_id: &sample.original_id,
                prompt_sha256: &sample.prompt_sha256,
                assistant_sha256: &sample.assistant_sha256,
                input_tokens: sample.input_tokens,
                reference_output_tokens: sample.reference_output_tokens,
                requested_output_tokens: sample.requested_output_tokens,
            });
            Ok(())
        })?;
    if parent.source_sha256 != selected.evidence.source_sha256
        || parent.source_format != selected.evidence.source_format
        || parent.records != selected.evidence.counts.records
        || !remaining.is_empty()
        || records.len() != samples.len()
    {
        return Err(FerrumError::model(
            "ShareGPT parent changed between selection and export scans",
        ));
    }
    output
        .write_all(b"\n]\n")
        .map_err(|error| io_error("finish pool", &pool_path, error))?;
    output
        .flush()
        .map_err(|error| io_error("flush pool", &pool_path, error))?;
    output
        .get_ref()
        .sync_all()
        .map_err(|error| io_error("sync pool", &pool_path, error))?;
    drop(output);
    let child = visit_sharegpt_records(
        File::open(&pool_path).map_err(|error| io_error("verify pool", &pool_path, error))?,
        ShareGptReadLimits::default(),
        |_, _| Ok(()),
    )?;
    if child.records != samples.len() as u64 || child.source_format != "json_array" {
        return Err(FerrumError::internal(
            "ShareGPT exported record count differs",
        ));
    }
    let receipt = ExportReceipt {
        schema_version: 1,
        selection_semantics: "new_fixed_derived_pool",
        serialization_order: "parent_source_record_index_ascending",
        selector_contract: super::cache::SELECTOR_CONTRACT,
        tokenization_contract: super::cache::TOKENIZATION_CONTRACT,
        reader_limits: ShareGptReadLimits::default(),
        parent: &selected.evidence,
        tokenizer: &selected.tokenizer,
        request_policy: &selected.request_policy,
        child: ChildSource {
            file: "pool.json",
            source_format: child.source_format,
            source_sha256: child.source_sha256,
            source_bytes: child.source_bytes,
            records: child.records,
        },
        records,
    };
    let receipt_path = directory.join(".pool.receipt.json.tmp");
    let mut receipt_file = create_file(&receipt_path)?;
    serde_json::to_writer_pretty(&mut receipt_file, &receipt).map_err(|error| {
        FerrumError::model(format!("serialize ShareGPT export receipt: {error}"))
    })?;
    receipt_file
        .write_all(b"\n")
        .map_err(|error| io_error("write receipt", &receipt_path, error))?;
    receipt_file
        .sync_all()
        .map_err(|error| io_error("sync receipt", &receipt_path, error))?;
    drop(receipt_file);
    fs::rename(&pool_path, directory.join("pool.json"))
        .map_err(|error| io_error("publish pool", &pool_path, error))?;
    // The receipt is the final completion marker. A killed process with only
    // temporary files or an unreceipted pool has not completed an export.
    fs::rename(&receipt_path, directory.join("pool.receipt.json"))
        .map_err(|error| io_error("publish receipt", &receipt_path, error))?;
    Ok(())
}

fn check_record(value: &serde_json::Value, sample: &ShareGptSample) -> Result<()> {
    let pair = sharegpt_first_pair(value)
        .ok_or_else(|| FerrumError::model("selected ShareGPT record lost its first pair"))?;
    if sharegpt_original_id(value) != sample.original_id
        || sha256_hex(pair.prompt.as_bytes()) != sample.prompt_sha256
        || sha256_hex(pair.assistant.as_bytes()) != sample.assistant_sha256
    {
        return Err(FerrumError::model(
            "selected ShareGPT record changed during export",
        ));
    }
    Ok(())
}

fn create_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| io_error("create export file", path, error))
}

fn io_error(action: &str, path: &Path, error: impl std::fmt::Display) -> FerrumError {
    FerrumError::model(format!("{action} {}: {error}", path.display()))
}

struct Publication {
    directory: PathBuf,
    complete: bool,
}

impl Publication {
    fn create(directory: &Path) -> Result<Self> {
        fs::create_dir(directory)
            .map_err(|error| io_error("create exclusive export directory", directory, error))?;
        Ok(Self {
            directory: directory.to_owned(),
            complete: false,
        })
    }
}

impl Drop for Publication {
    fn drop(&mut self) {
        if !self.complete {
            // Unwind cleanup. Ordinary errors use the checked removal above.
            let _ = fs::remove_dir_all(&self.directory);
        }
    }
}
