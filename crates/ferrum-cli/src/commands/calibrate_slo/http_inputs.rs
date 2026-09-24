//! Original HTTP inputs, exported once independently of calibration repetitions.
//! This bundle records constructed requests, never claims they were executed.
use super::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs::OpenOptions, io::Write, path::Path};

const MAX_EXPORT_BYTES: u64 = 256 * 1024 * 1024;

#[cfg(test)]
mod tests;

#[derive(Debug, Serialize)]
pub(super) struct HttpInputExportReceipt {
    directory: PathBuf,
    manifest_file: &'static str,
    manifest_sha256: String,
    requests: usize,
    total_bytes: u64,
}

#[derive(Serialize)]
struct RequestFile<'a> {
    source_index: usize,
    sample: &'a ferrum_bench_core::dataset::ShareGptSample,
    body_file: String,
    body_bytes: u64,
    body_sha256: String,
}

#[derive(Serialize)]
struct HttpInputManifest<'a> {
    schema_version: u32,
    method: &'static str,
    endpoint: &'static str,
    media_type: &'static str,
    scope: &'static str,
    execution_evidence: &'static str,
    reference_policy_applied: bool,
    input_preprocessing_sha256: String,
    frozen_selection: serde_json::Value,
    protocol: &'a manifest::Protocol,
    training: &'a [manifest::Cohort],
    heldout: &'a [manifest::Cohort],
    requests: Vec<RequestFile<'a>>,
}

pub(super) fn export(
    inputs: &inputs::PreparedInputs,
    manifest: &manifest::Manifest,
    directory: &Path,
) -> Result<HttpInputExportReceipt> {
    export_with_limit(inputs, manifest, directory, MAX_EXPORT_BYTES)
}

fn export_with_limit(
    inputs: &inputs::PreparedInputs,
    manifest: &manifest::Manifest,
    directory: &Path,
    maximum_bytes: u64,
) -> Result<HttpInputExportReceipt> {
    manifest.validate()?;
    let inputs::PreparedInputs::ShareGpt { recovered, .. } = inputs else {
        return Err(FerrumError::config(
            "HTTP input export requires schema 2 frozen ShareGPT",
        ));
    };
    let indices: BTreeSet<_> = manifest
        .cohorts()
        .flat_map(|cohort| cohort.prompts.iter().copied())
        .collect();
    if indices.is_empty()
        || indices.len() > 4096
        || indices
            .iter()
            .any(|&index| index >= recovered.prompts.len())
        || maximum_bytes == 0
        || maximum_bytes > MAX_EXPORT_BYTES
    {
        return Err(FerrumError::config(
            "HTTP input export exceeds its frozen selection or byte bound",
        ));
    }
    // Atomic exclusive directory creation rejects even an existing empty dir or
    // symlink. Individual create_new writes defend against later interference.
    std::fs::create_dir(directory).map_err(|error| {
        FerrumError::config(format!(
            "create new HTTP input directory {}: {error}",
            directory.display()
        ))
    })?;
    let result: Result<HttpInputExportReceipt> = (|| {
        let mut bytes_written = 0;
        let mut requests = Vec::with_capacity(indices.len());
        for index in indices {
            let bytes = serde_json::to_vec(&inputs.original_http_body(index)?)
                .map_err(|error| FerrumError::config(format!("encode HTTP input: {error}")))?;
            let body_file = format!("request-{index:04}.json");
            write_new(
                &directory.join(&body_file),
                &bytes,
                &mut bytes_written,
                maximum_bytes,
            )?;
            requests.push(RequestFile {
                source_index: index,
                sample: &recovered.prompts[index].sample,
                body_file,
                body_bytes: bytes.len() as u64,
                body_sha256: sharegpt::hex(&Sha256::digest(&bytes)),
            });
        }
        let count = requests.len();
        let metadata = HttpInputManifest {
            schema_version: 1, method: "POST", endpoint: "/v1/chat/completions",
            media_type: "application/json",
            scope: "one_original_body_per_training_or_heldout_source_index; cohort_order_and_repetitions_preserved_separately",
            execution_evidence: "constructed_inputs_only; consult_actual_frontier_and_completion_records",
            reference_policy_applied: false,
            input_preprocessing_sha256: sharegpt::hex(&manifest.input_preprocessing_sha256),
            frozen_selection: inputs.provenance(), protocol: &manifest.protocol,
            training: &manifest.training, heldout: &manifest.validation, requests,
        };
        let bytes = serde_json::to_vec_pretty(&metadata)
            .map_err(|error| FerrumError::config(format!("encode HTTP input manifest: {error}")))?;
        // Written last: a complete manifest lists bodies already written and
        // synced. Partial failures keep their raw files for inspection.
        write_new(
            &directory.join("manifest.json"),
            &bytes,
            &mut bytes_written,
            maximum_bytes,
        )?;
        Ok(HttpInputExportReceipt {
            directory: directory.to_owned(),
            manifest_file: "manifest.json",
            manifest_sha256: sharegpt::hex(&Sha256::digest(&bytes)),
            requests: count,
            total_bytes: bytes_written,
        })
    })();
    result.map_err(|error| {
        FerrumError::config(format!(
            "HTTP input export failed; files created in {} are retained: {error}",
            directory.display()
        ))
    })
}

fn write_new(path: &Path, bytes: &[u8], total: &mut u64, limit: u64) -> Result<()> {
    let next = total
        .checked_add(bytes.len() as u64)
        .filter(|sum| *sum <= limit)
        .ok_or_else(|| {
            FerrumError::resource_exhausted("HTTP input bundle exceeds its byte limit")
        })?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| FerrumError::config(format!("create {}: {error}", path.display())))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| FerrumError::config(format!("write {}: {error}", path.display())))?;
    *total = next;
    Ok(())
}
