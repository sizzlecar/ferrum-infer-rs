//! Bounded, read-only loading of frozen comparison evidence. File hashes bind
//! original bytes; they do not attest hardware execution or statistical proof.

use super::*;
use std::path::Path;
use std::sync::Arc;

mod eligibility;
mod files;
mod macos_time;
mod memory;
mod types;
pub use types::*;

use files::{jsonl_rows, parse, Reader};

fn err(message: impl Into<String>) -> ArtifactError {
    ArtifactError(message.into())
}

struct LoadedArm {
    sidecar: Arc<SloSidecarCell>,
    repeat_index: u32,
    execution: ArmExecutionEvidence,
    memory: Option<PeakMemoryEvidence>,
}

impl LoadedArm {
    fn input(&self) -> ArmRepeatInput<'_> {
        ArmRepeatInput {
            legacy_benchmark: &self.sidecar.legacy_benchmark,
            report_repeat_index: self.repeat_index,
            evaluation: &self.sidecar.repeats[self.repeat_index as usize].evaluation,
            execution: &self.execution,
            memory: self.memory.as_ref(),
        }
    }
}

#[derive(Default)]
struct Sidecars {
    loaded_hashes: BTreeSet<String>,
    records: BTreeMap<(String, u32), Arc<SloSidecarCell>>,
    visible_gaps: usize,
}

fn validate_sidecar(
    cell: &SloSidecarCell,
    limits: &ArtifactLoadLimits,
    visible_gaps: &mut usize,
) -> Result<(), ArtifactError> {
    if cell.schema_version != 1
        || !valid_digest(&cell.config_sha256)
        || cell.repeats.is_empty()
        || cell.repeats.len() > limits.max_pairs_per_cell
        || cell.repeats.len() != cell.legacy_benchmark.n_repeats as usize
    {
        return Err(err(
            "sidecar version, config identity or repeat count is invalid",
        ));
    }
    let report = &cell.legacy_benchmark;
    if report.repeat_metrics.len() != cell.repeats.len() {
        return Err(err("sidecar and legacy repeat counts differ"));
    }
    if let Some(dataset) = &report.dataset_evidence {
        if dataset.repeats.len() > limits.max_pairs_per_cell
            || dataset
                .repeats
                .iter()
                .any(|selection| selection.samples.len() > limits.max_requests_per_repeat)
        {
            return Err(err("ShareGPT selection size limit exceeded"));
        }
    }
    for (index, repeat) in cell.repeats.iter().enumerate() {
        let records = &repeat.evaluation.request_evidence;
        if repeat.repeat_index as usize != index
            || repeat.evaluation.schema_version != crate::slo::SCHEMA_VERSION
            || records.len() > limits.max_requests_per_repeat
            || repeat.arrivals.len() != records.len()
        {
            return Err(err(
                "sidecar repeat index, version, request bound or arrival alignment is invalid",
            ));
        }
        for record in records {
            let gaps = record
                .visible_text
                .as_ref()
                .map_or(0, |text| text.gaps_ms.len());
            *visible_gaps = visible_gaps
                .checked_add(gaps)
                .filter(|&n| n <= limits.max_total_visible_gaps)
                .ok_or_else(|| err("total visible-gap resource limit exceeded"))?;
        }
        let offsets = repeat
            .arrivals
            .iter()
            .flat_map(|a| {
                [
                    a.scheduled_arrival_ms,
                    a.dispatched_ms,
                    a.request_started_ms,
                ]
            })
            .chain([
                repeat.request_start_window_s,
                repeat.observed_request_start_rate_rps,
            ]);
        if offsets.flatten().any(|n| !n.is_finite() || n < 0.0) {
            return Err(err("invalid arrival metadata"));
        }
    }
    Ok(())
}

impl Sidecars {
    fn load(
        &mut self,
        reader: &mut Reader<'_>,
        reference: &SidecarRef,
    ) -> Result<Arc<SloSidecarCell>, ArtifactError> {
        let bytes = reader.read(&reference.file)?;
        let hash = digest(&bytes);
        if self.loaded_hashes.insert(hash.clone()) {
            for (index, row) in jsonl_rows(&bytes, reader.limits.max_jsonl_records)?
                .into_iter()
                .enumerate()
            {
                let cell: SloSidecarCell = parse(row)?;
                validate_sidecar(&cell, reader.limits, &mut self.visible_gaps)?;
                self.records
                    .insert((hash.clone(), index as u32), Arc::new(cell));
            }
        }
        self.records
            .get(&(hash, reference.record_index))
            .cloned()
            .ok_or_else(|| err("sidecar JSONL record index is out of range"))
    }
}

fn load_arm(
    reader: &mut Reader<'_>,
    sidecars: &mut Sidecars,
    refs: &ArmArtifactRefs,
    contract: &FrozenComparisonContract,
    frozen_pair: &FrozenPair,
    concurrency: u32,
    candidate: bool,
) -> Result<LoadedArm, ArtifactError> {
    let sidecar = sidecars.load(reader, &refs.sidecar)?;
    let report = &sidecar.legacy_benchmark;
    let repeat = sidecar
        .repeats
        .get(refs.repeat_index as usize)
        .ok_or_else(|| err("sidecar repeat index is out of range"))?;
    let declaration: ExecutionArtifact = parse(&reader.read(&refs.execution)?)?;
    let server = if candidate {
        &contract.candidate
    } else {
        &contract.baseline
    };
    crate::BenchmarkRequestCorrelation::new(
        declaration.identity.benchmark_run_id.clone(),
        declaration.identity.cell_id.clone(),
        declaration.identity.repeat_index,
        crate::BenchmarkPhase::Measured,
        0,
    )
    .map_err(err)?;
    if declaration.schema_version != 1
        || declaration.identity.server_pid == 0
        || declaration.identity.repeat_index != refs.repeat_index
        || Some(&declaration.identity.benchmark_run_id) != report.benchmark_run_id.as_ref()
        || Some(&declaration.identity.cell_id) != report.cell_id.as_ref()
        || report.concurrency != Some(concurrency)
        || declaration.shared != contract.shared
        || declaration.server != *server
        || declaration.capacity != contract.capacity
        || declaration.measurement_started_unix_ns < contract.frozen_unix_ns
        || declaration.measurement_ended_unix_ns <= declaration.measurement_started_unix_ns
        || !same_digest(
            &sidecar.config_sha256,
            &contract.shared.client_slo_config_sha256,
        )
        || repeat.evaluation.config != contract.slo
    {
        return Err(err(
            "execution declaration, sidecar and frozen contract identity mismatch",
        ));
    }
    let selection = report
        .dataset_evidence
        .as_ref()
        .and_then(|dataset| {
            dataset
                .repeats
                .iter()
                .find(|selection| selection.repeat_index == refs.repeat_index)
        })
        .ok_or_else(|| err("referenced sidecar has no matching ordered ShareGPT selection"))?;
    if !valid_digest(&refs.selection_sha256)
        || !same_digest(&refs.selection_sha256, &selection.selection_sha256)
        || !same_digest(
            &refs.selection_sha256,
            &frozen_pair.selection.selection_sha256,
        )
        || !same_digest(
            &refs.selection_sha256,
            &json_digest(&selection.samples).map_err(|e| err(e.to_string()))?,
        )
    {
        return Err(err(
            "selection reference does not bind the frozen ordered samples",
        ));
    }
    let memory = memory::load(reader, &refs.memory, &declaration)?;
    let execution = ArmExecutionEvidence {
        shared: declaration.shared,
        server: declaration.server,
        capacity: declaration.capacity,
        source_manifest_sha256: digest(&reader.read(&refs.execution)?),
        measurement_started_unix_ns: declaration.measurement_started_unix_ns,
        measurement_ended_unix_ns: declaration.measurement_ended_unix_ns,
    };
    Ok(LoadedArm {
        sidecar,
        repeat_index: refs.repeat_index,
        execution,
        memory,
    })
}

/// Loads only bounded, local, hash-pinned artifacts and evaluates every frozen
/// cell/pair. Missing declared evidence remains Unknown through the comparator;
/// corrupt files or inconsistent provenance are loader errors, not slow runs.
/// Optional eligibility loads the original independent A/A pilot and planning
/// document; no caller-provided success certificate is accepted.
pub fn compare_manifest(
    path: &Path,
    limits: &ArtifactLoadLimits,
) -> Result<ArtifactComparisonReport, ArtifactError> {
    let (mut reader, manifest) = Reader::open(path, limits)?;
    let mut sidecars = Sidecars::default();
    let loaded = eligibility::load_comparison(&mut reader, &mut sidecars, &manifest)?;
    let inputs = loaded.inputs();
    let comparison = if let Some(refs) = &manifest.eligibility {
        eligibility::compare_using_pilot(
            &mut reader,
            &mut sidecars,
            refs,
            &loaded.contract,
            &inputs,
        )?
    } else {
        compare_slo_reports(&loaded.contract, &inputs).map_err(|error| err(error.to_string()))?
    };
    Ok(ArtifactComparisonReport { schema_version: 1, manifest_sha256: reader.manifest_hash().into(), verified_files: reader.verified_files(),
        evidence_boundary: "SHA-256 verifies original file bytes and declaration alignment, not physical hardware execution. Execution identity is an acquisition-time declaration. Native Metal samples are reconstructed; supplied macOS time output yields process lifetime peaks, never samples. Parsing does not establish actual process parentage; acquisition declarations and any command/ps evidence remain auditable sources. No OS sampling is performed by this loader. ProofPass requires raw independent-pilot eligibility and all approximate simultaneous bounds under declared assumptions; missing pilot evidence cannot produce proof. The verifier does not establish real-world IID or unconditional coverage.".into(), comparison })
}

#[cfg(test)]
mod tests;
