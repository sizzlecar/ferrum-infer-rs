//! Bounded retrospective ablation for legacy V3 recordings. This never emits
//! a deployable profile or invents the V2 per-row static capability. Pooling
//! under an unchanged old whole-wave host digest is an explicit diagnostic
//! assumption; new actual row features are required to validate production V2.
use anyhow::{bail, ensure, Context, Result};
use ferrum_interfaces::execution_cost::{CanonicalWaveCostFeatures, HostContentCostFeaturesV1};
use ferrum_scheduler::implementations::continuous::{
    cost_model::*,
    cost_profile::{self as profile, v3::CostProfileFileV3},
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    path::Path,
};

const FILE_LIMIT: u64 = 256 * 1024 * 1024;
const LINE_LIMIT: u64 = 4 * 1024 * 1024;
const QUERY_LIMIT: usize = 131_072;

#[derive(Deserialize)]
struct Clock {
    wall_unix_ns: u64,
    monotonic_ns: u64,
}
#[derive(Deserialize)]
struct Header {
    artifact_type: String,
    schema_version: u32,
    opening: Clock,
}
#[derive(Deserialize)]
struct Stages {
    actual_shape: ActualShape,
    finalized_at_ns: Option<u64>,
    full_wall_ns: Option<u64>,
    completeness: String,
    rows: Vec<Value>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ActualShape {
    exact: profile::ProfileWaveShape,
    numeric_features: Option<CanonicalWaveCostFeatures>,
    host_content_features: Option<HostContentCostFeaturesV1>,
}
impl ActualShape {
    fn into_shape(self) -> WaveExecutionShape {
        let mut shape: WaveExecutionShape = self.exact.into();
        shape.numeric_features = self.numeric_features;
        shape.host_content_features = self.host_content_features;
        shape
    }
}

fn bytes(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    ensure!(
        file.metadata()?.len() <= FILE_LIMIT,
        "artifact exceeds bounded input"
    );
    let mut out = Vec::new();
    file.take(FILE_LIMIT + 1).read_to_end(&mut out)?;
    ensure!(
        out.len() as u64 <= FILE_LIMIT,
        "artifact grew beyond bounded input"
    );
    Ok(out)
}
fn jsonl(path: &Path, mut visit: impl FnMut(usize, &[u8]) -> Result<()>) -> Result<[u8; 32]> {
    let file = File::open(path)?;
    ensure!(
        file.metadata()?.len() <= FILE_LIMIT,
        "JSONL exceeds bounded input"
    );
    let mut reader = BufReader::new(file);
    let mut line = Vec::new();
    let mut total = 0u64;
    let mut index = 0;
    let mut digest = Sha256::new();
    loop {
        line.clear();
        let length = Read::by_ref(&mut reader)
            .take(LINE_LIMIT + 1)
            .read_until(b'\n', &mut line)?;
        if length == 0 {
            break;
        }
        ensure!(length as u64 <= LINE_LIMIT, "JSONL line exceeds bound");
        total = total
            .checked_add(length as u64)
            .context("JSONL byte overflow")?;
        ensure!(total <= FILE_LIMIT, "JSONL grew beyond bound");
        digest.update(&line);
        visit(index, &line)?;
        index += 1;
    }
    Ok(digest.finalize().into())
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Deliberately only pure-row waves, with legacy host/provider hashes retained.
/// This DOES NOT recover per-row mask/policy/readback classes from their digest.
fn diagnostic_permute(shape: &mut WaveExecutionShape) -> Result<bool> {
    let Some(numeric) = shape.numeric_features.as_mut() else {
        return Ok(false);
    };
    match shape.kind {
        WaveKind::Decode => {
            ensure!(
                numeric.rows.len() == shape.decode_kv_tokens.len()
                    && shape.prefill_chunks.is_empty(),
                "misaligned legacy decode evidence"
            );
            let mut rows: Vec<_> = shape
                .decode_kv_tokens
                .iter()
                .copied()
                .zip(numeric.rows.iter().cloned())
                .collect();
            rows.sort_unstable();
            for (index, (kv, row)) in rows.into_iter().enumerate() {
                shape.decode_kv_tokens[index] = kv;
                numeric.rows[index] = row;
            }
        }
        WaveKind::Prefill => {
            ensure!(
                numeric.rows.len() == shape.prefill_chunks.len()
                    && shape.decode_kv_tokens.is_empty(),
                "misaligned legacy prefill evidence"
            );
            let mut rows: Vec<_> = shape
                .prefill_chunks
                .iter()
                .cloned()
                .zip(numeric.rows.iter().cloned())
                .collect();
            rows.sort_unstable();
            for (index, (chunk, row)) in rows.into_iter().enumerate() {
                shape.prefill_chunks[index] = chunk;
                numeric.rows[index] = row;
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}
fn result(value: CostPrediction, actual: u64) -> Value {
    match value {
        CostPrediction::Known(p) => {
            json!({"kind":"diagnostic_estimate","sample_count":p.sample_count,"planning_ns":p.planning_ns,"typical_ns":p.typical_ns,"valid_for_ns":p.valid_for_ns,"underestimate_ns":actual.saturating_sub(p.planning_ns)})
        }
        CostPrediction::Unknown(reason) => {
            json!({"kind":"diagnostic_unknown","reason":format!("{reason:?}")})
        }
    }
}

fn run(profile_path: &Path, source_path: &Path, raw_path: &Path, output: &Path) -> Result<()> {
    let original = bytes(profile_path)?;
    let mut file: CostProfileFileV3 = serde_json::from_slice(&original)?;
    ensure!(
        file.schema_version == 3
            && matches!(
                file.settings.feature_model,
                CostFeatureModel::EmpiricalHostContentV1 { .. }
            ),
        "requires explicit legacy empirical V3"
    );
    let mut header = None;
    let source_sha = jsonl(source_path, |index, line| {
        if index == 0 {
            header = Some(serde_json::from_slice::<Header>(line)?);
        }
        Ok(())
    })?;
    ensure!(
        source_sha == file.source.observation_artifact_sha256,
        "profile/source hash mismatch"
    );
    let header = header.context("source header missing")?;
    ensure!(
        header.artifact_type == "ferrum.cost-training-cut" && header.schema_version == 3,
        "unsupported legacy cut header"
    );
    let fingerprint: ExecutionFingerprint = file.fingerprint.clone().into();
    let mut settings: CostModelSettings = file.settings.exact.clone().into();
    settings.feature_model = file.settings.feature_model.clone();
    // Historical evaluation epoch, explicitly not a wall-clock freshness claim
    // at tool invocation. Every heldout timestamp is projected from the source
    // opening clock, and original profile measured_unix_ns is left unchanged.
    let clock = profile::ProfileLoadClock {
        wall_unix_ns: Some(file.generated_unix_ns),
        wall_max_error_ns: Some(0),
        monotonic_now_ns: 0,
    };
    let mut limits = profile::CostProfileLoadLimits::default();
    limits.max_file_bytes = std::num::NonZeroUsize::new(FILE_LIMIT as usize).unwrap();
    let baseline =
        profile::load_cost_profile_bytes(&original, &fingerprint, &settings, &limits, clock)?;
    let mut pure_samples = 0usize;
    for sample in &mut file.samples {
        let mut shape: WaveExecutionShape = sample.shape.exact.exact.clone().into();
        shape.numeric_features = sample.shape.exact.numeric_features.clone();
        shape.host_content_features = Some(sample.shape.host_content_features);
        if diagnostic_permute(&mut shape)? {
            pure_samples += 1;
        }
        sample.shape.exact = (&shape).into();
    }
    let diagnostic_bytes = serde_json::to_vec(&file)?;
    let candidate = profile::load_cost_profile_bytes(
        &diagnostic_bytes,
        &fingerprint,
        &settings,
        &limits,
        clock,
    )?;
    let mut queries = Vec::new();
    let mut excluded = 0usize;
    let raw_sha = jsonl(raw_path, |index, line| {
        let value: Value = serde_json::from_slice(line)?;
        if value["event"] != "wave" || value["phase"] != "validation" {
            return Ok(());
        }
        ensure!(
            queries.len() + excluded < QUERY_LIMIT,
            "heldout query count exceeds bound"
        );
        let Some(stages) = value.get("host_stages").filter(|s| !s.is_null()) else {
            excluded += 1;
            return Ok(());
        };
        let stages: Stages = serde_json::from_value(stages.clone())?;
        if stages.completeness != "complete_single_wave" || stages.rows.is_empty() {
            excluded += 1;
            return Ok(());
        }
        let actual = stages
            .full_wall_ns
            .filter(|n| *n > 0)
            .context("complete wave lacks measured wall")?;
        let observed = stages
            .finalized_at_ns
            .context("complete wave lacks receipt time")?;
        let wall = observed
            .checked_sub(header.opening.monotonic_ns)
            .and_then(|elapsed| header.opening.wall_unix_ns.checked_add(elapsed))
            .context("source clock projection overflow/reversal")?;
        let elapsed = wall
            .checked_sub(file.generated_unix_ns)
            .context("heldout precedes training cut")?;
        let mut shape = stages.actual_shape.into_shape();
        let original_prediction = result(
            baseline.snapshot.predict(
                &fingerprint,
                &shape,
                CostBoundary::PreparationToHostSettledV1,
                elapsed,
            ),
            actual,
        );
        if !diagnostic_permute(&mut shape)? {
            excluded += 1;
            return Ok(());
        }
        let alternative = result(
            candidate.snapshot.predict(
                &fingerprint,
                &shape,
                CostBoundary::PreparationToHostSettledV1,
                elapsed,
            ),
            actual,
        );
        queries.push(json!({"raw_line":index+1,"case":value["case"],"repetition":value["repetition"],"captured_prediction":value["host_content_frozen_prediction"],"baseline_replay":original_prediction,"same_legacy_hash_tuple_ablation":alternative,"actual_wall_ns":actual,"terminal_rows":stages.rows.iter().filter(|r|!r["terminal"].is_null()).count()}));
        Ok(())
    })?;
    let report = json!({"schema_version":1,"scope":"historical V3 same-host-digest tuple-permutation ablation; not production V2, not a fresh profile, not a deployment eligibility result; missing per-row categorical evidence remains unverified","assumption":"within the unchanged ordered legacy host digest and exact provider route, complete work/numeric tuple permutations are treated as exchangeable; this is not proven by the old recording","profile_sha256":hex(&Sha256::digest(&original)),"source_sha256":hex(&source_sha),"raw_sha256":hex(&raw_sha),"historical_epoch_unix_ns":file.generated_unix_ns,"baseline_recorded_samples":baseline.provenance.counts.recorded_samples,"pure_samples":pure_samples,"excluded_queries":excluded,"queries":queries});
    let mut target = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    serde_json::to_writer_pretty(&mut target, &report)?;
    target.write_all(b"\n")?;
    target.sync_all()?;
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 4 {
        bail!("usage: row_permutation_replay PROFILE_V3 CUT_SOURCE_JSONL COLLECTOR_RAW_JSONL NEW_OUTPUT_JSON");
    }
    run(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
    )
}

#[cfg(test)]
#[path = "row_permutation_replay/tests.rs"]
mod tests;
