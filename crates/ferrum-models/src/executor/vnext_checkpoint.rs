use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ferrum_interfaces::vnext::{
    CompletionReadbackBatchRequest, CompletionReadbackOutput, ExecutionPlan, HostTransferLayout,
    ProgramPlanCompileOptions, ProgramValueId, ResourceWorkShape, RetainedCompletionValue, RunId,
    TokenSpanWork,
};
use ferrum_types::{FerrumError, RequestId, Result, VNextCheckpointCaptureConfig};
use serde::Serialize;
use sha2::{Digest, Sha256};

const MAX_CHECKPOINT_VALUES: usize = 63;
const MAX_PREFILL_WAVES: usize = 16;

pub(super) struct VNextCheckpointSelection {
    output_dir: PathBuf,
    value_ids: Vec<ProgramValueId>,
    maximum_prefill_waves: usize,
}

impl VNextCheckpointSelection {
    pub(super) fn from_config(
        config: Option<&VNextCheckpointCaptureConfig>,
    ) -> Result<Option<Self>> {
        let Some(config) = config else {
            return Ok(None);
        };
        if config.output_dir.as_os_str().is_empty() {
            return Err(FerrumError::config(
                "vNext checkpoint output directory cannot be empty",
            ));
        }
        if config.value_ids.is_empty() || config.value_ids.len() > MAX_CHECKPOINT_VALUES {
            return Err(FerrumError::config(format!(
                "vNext checkpoint value count must be in 1..={MAX_CHECKPOINT_VALUES}"
            )));
        }
        if config.maximum_prefill_waves == 0 || config.maximum_prefill_waves > MAX_PREFILL_WAVES {
            return Err(FerrumError::config(format!(
                "vNext checkpoint prefill wave count must be in 1..={MAX_PREFILL_WAVES}"
            )));
        }
        let value_ids = config
            .value_ids
            .iter()
            .map(|value| {
                ProgramValueId::new(value.clone()).map_err(|error| {
                    FerrumError::config(format!(
                        "invalid vNext checkpoint value {value:?}: {error}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if value_ids.iter().collect::<BTreeSet<_>>().len() != value_ids.len() {
            return Err(FerrumError::config(
                "vNext checkpoint values contain duplicates",
            ));
        }
        let mut value_ids = value_ids;
        value_ids.sort();
        Ok(Some(Self {
            output_dir: config.output_dir.clone(),
            value_ids,
            maximum_prefill_waves: config.maximum_prefill_waves,
        }))
    }

    pub(super) fn retain_in(&self, options: &mut ProgramPlanCompileOptions) {
        for value_id in &self.value_ids {
            options.retain_completion_value(value_id.clone());
        }
    }

    pub(super) fn bind(
        self,
        plan: &ExecutionPlan,
        model_id: String,
        family_fingerprint: String,
        program_fingerprint: String,
        run_id: &RunId,
    ) -> Result<VNextCheckpointCapture> {
        prepare_empty_output_directory(&self.output_dir)?;
        let checkpoints = self
            .value_ids
            .iter()
            .map(|value_id| plan.completion_checkpoint(value_id).cloned())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| FerrumError::model(error.to_string()))?;
        let capture = VNextCheckpointCapture {
            output_dir: self.output_dir,
            checkpoints,
            maximum_prefill_waves: self.maximum_prefill_waves,
            next_prefill_wave: AtomicUsize::new(0),
            armed: AtomicBool::new(false),
            plan_id: plan.payload().plan_id().to_string(),
            plan_hash: plan.plan_hash().to_string(),
            model_id,
            family_fingerprint,
            program_fingerprint,
            run_id: run_id.to_string(),
        };
        capture.write_plan_manifest()?;
        Ok(capture)
    }
}

pub(super) struct VNextCheckpointCapture {
    output_dir: PathBuf,
    checkpoints: Vec<RetainedCompletionValue>,
    maximum_prefill_waves: usize,
    next_prefill_wave: AtomicUsize,
    armed: AtomicBool,
    plan_id: String,
    plan_hash: String,
    model_id: String,
    family_fingerprint: String,
    program_fingerprint: String,
    run_id: String,
}

impl VNextCheckpointCapture {
    pub(super) fn arm(&self) {
        self.armed.store(true, Ordering::Release);
    }

    pub(super) fn claim_prefill_wave(&self) -> Option<usize> {
        if !self.armed.load(Ordering::Acquire) {
            return None;
        }
        self.next_prefill_wave
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < self.maximum_prefill_waves).then_some(current + 1)
            })
            .ok()
    }

    pub(super) fn checkpoints(&self) -> &[RetainedCompletionValue] {
        &self.checkpoints
    }

    pub(super) fn readback_batches(
        &self,
        plan: &ExecutionPlan,
        token_spans: &[&TokenSpanWork],
    ) -> Result<Vec<CompletionReadbackBatchRequest>> {
        self.checkpoints
            .iter()
            .map(|checkpoint| {
                let requests = token_spans
                    .iter()
                    .enumerate()
                    .map(|(participant_index, token_span)| {
                        let participant_index = u32::try_from(participant_index).map_err(|_| {
                            FerrumError::backend("vNext checkpoint participant index exceeds u32")
                        })?;
                        let work = ResourceWorkShape::single((*token_span).clone())
                            .map_err(|error| FerrumError::backend(error.to_string()))?;
                        plan.completion_checkpoint_readback_for_work(
                            checkpoint.value_id(),
                            participant_index,
                            &work,
                        )
                        .map_err(|error| FerrumError::backend(error.to_string()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                CompletionReadbackBatchRequest::new(requests)
                    .map_err(|error| FerrumError::backend(error.to_string()))
            })
            .collect()
    }

    pub(super) fn checkpoint_for_output(
        &self,
        output: &CompletionReadbackOutput,
    ) -> Option<&RetainedCompletionValue> {
        self.checkpoints.iter().find(|checkpoint| {
            checkpoint.producer_node_id() == output.request().node_id()
                && checkpoint.resource_id() == output.request().resource_id()
        })
    }

    pub(super) fn write_output(
        &self,
        capture_index: usize,
        request_id: &RequestId,
        token_span: &TokenSpanWork,
        checkpoint: &RetainedCompletionValue,
        output: &CompletionReadbackOutput,
    ) -> Result<VNextCheckpointArtifactRecord> {
        if checkpoint.producer_node_id() != output.request().node_id()
            || checkpoint.resource_id() != output.request().resource_id()
            || checkpoint.logical_offset_bytes() != output.request().logical_offset_bytes()
            || checkpoint.tensor().element_type() != output.request().output_layout().element_type()
        {
            return Err(FerrumError::internal(
                "vNext checkpoint output does not match its retained semantic value",
            ));
        }
        let stem = checkpoint_file_stem(
            capture_index,
            output.request().participant_index(),
            checkpoint.value_id(),
        );
        let raw_file = format!("{stem}.bin");
        write_new_file(&self.output_dir.join(&raw_file), output.bytes())?;
        Ok(VNextCheckpointArtifactRecord {
            value: checkpoint.clone(),
            participant_index: output.request().participant_index(),
            request_id: request_id.to_string(),
            token_span: token_span.clone(),
            output_layout: output.request().output_layout(),
            raw_file,
            raw_bytes: u64::try_from(output.bytes().len()).unwrap_or(u64::MAX),
            raw_sha256: output.sha256().to_owned(),
        })
    }

    pub(super) fn finish_wave(
        &self,
        capture_index: usize,
        participant_count: usize,
        completion_fingerprint: &str,
        receipt_fingerprint: &str,
        mut records: Vec<VNextCheckpointArtifactRecord>,
    ) -> Result<()> {
        let expected_records = self
            .checkpoints
            .len()
            .checked_mul(participant_count)
            .ok_or_else(|| {
                FerrumError::internal("vNext checkpoint record count overflows usize")
            })?;
        if records.len() != expected_records {
            return Err(FerrumError::internal(format!(
                "vNext checkpoint wave produced {} records, expected {expected_records}",
                records.len()
            )));
        }
        let observed = records
            .iter()
            .map(|record| (record.value.value_id().clone(), record.participant_index))
            .collect::<BTreeSet<_>>();
        if observed.len() != expected_records {
            return Err(FerrumError::internal(
                "vNext checkpoint wave contains duplicate semantic participant records",
            ));
        }
        records.sort_by(|left, right| {
            left.value
                .value_id()
                .cmp(right.value.value_id())
                .then_with(|| left.participant_index.cmp(&right.participant_index))
        });
        let manifest = VNextCheckpointWaveManifest {
            schema_version: 1,
            capture_index,
            plan_id: &self.plan_id,
            plan_hash: &self.plan_hash,
            model_id: &self.model_id,
            family_fingerprint: &self.family_fingerprint,
            program_fingerprint: &self.program_fingerprint,
            run_id: &self.run_id,
            wave_kind: "prefill",
            participant_count,
            completion_fingerprint,
            receipt_fingerprint,
            records: &records,
        };
        let bytes = serde_json::to_vec_pretty(&manifest).map_err(|error| {
            FerrumError::internal(format!("serialize vNext checkpoint wave: {error}"))
        })?;
        write_new_file(
            &self
                .output_dir
                .join(format!("wave-{capture_index:04}.json")),
            &bytes,
        )
    }

    fn write_plan_manifest(&self) -> Result<()> {
        let manifest = VNextCheckpointPlanManifest {
            schema_version: 1,
            plan_id: &self.plan_id,
            plan_hash: &self.plan_hash,
            model_id: &self.model_id,
            family_fingerprint: &self.family_fingerprint,
            program_fingerprint: &self.program_fingerprint,
            run_id: &self.run_id,
            maximum_prefill_waves: self.maximum_prefill_waves,
            checkpoints: &self.checkpoints,
        };
        let bytes = serde_json::to_vec_pretty(&manifest).map_err(|error| {
            FerrumError::internal(format!("serialize vNext checkpoint plan: {error}"))
        })?;
        write_new_file(&self.output_dir.join("plan.json"), &bytes)
    }
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct VNextCheckpointArtifactRecord {
    value: RetainedCompletionValue,
    participant_index: u32,
    request_id: String,
    token_span: TokenSpanWork,
    output_layout: HostTransferLayout,
    raw_file: String,
    raw_bytes: u64,
    raw_sha256: String,
}

#[derive(Serialize)]
struct VNextCheckpointPlanManifest<'a> {
    schema_version: u32,
    plan_id: &'a str,
    plan_hash: &'a str,
    model_id: &'a str,
    family_fingerprint: &'a str,
    program_fingerprint: &'a str,
    run_id: &'a str,
    maximum_prefill_waves: usize,
    checkpoints: &'a [RetainedCompletionValue],
}

#[derive(Serialize)]
struct VNextCheckpointWaveManifest<'a> {
    schema_version: u32,
    capture_index: usize,
    plan_id: &'a str,
    plan_hash: &'a str,
    model_id: &'a str,
    family_fingerprint: &'a str,
    program_fingerprint: &'a str,
    run_id: &'a str,
    wave_kind: &'static str,
    participant_count: usize,
    completion_fingerprint: &'a str,
    receipt_fingerprint: &'a str,
    records: &'a [VNextCheckpointArtifactRecord],
}

fn prepare_empty_output_directory(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(FerrumError::config(
                    "vNext checkpoint output path must be a real directory",
                ));
            }
            if fs::read_dir(path)
                .map_err(|error| checkpoint_io_error("inspect output directory", error))?
                .next()
                .is_some()
            {
                return Err(FerrumError::config(
                    "vNext checkpoint output directory must be empty",
                ));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(path)
                .map_err(|error| checkpoint_io_error("create output directory", error))?;
        }
        Err(error) => return Err(checkpoint_io_error("inspect output directory", error)),
    }
    Ok(())
}

fn checkpoint_file_stem(
    capture_index: usize,
    participant_index: u32,
    value_id: &ProgramValueId,
) -> String {
    let slug = value_id
        .as_str()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    let digest = format!("{:x}", Sha256::digest(value_id.as_str().as_bytes()));
    format!(
        "capture-{capture_index:04}-participant-{participant_index:04}-{slug}-{}",
        &digest[..12]
    )
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| checkpoint_io_error("create evidence file", error))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| checkpoint_io_error("write evidence file", error))
}

fn checkpoint_io_error(context: &'static str, error: std::io::Error) -> FerrumError {
    FerrumError::internal(format!("vNext checkpoint {context}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture(maximum_prefill_waves: usize) -> VNextCheckpointCapture {
        VNextCheckpointCapture {
            output_dir: PathBuf::new(),
            checkpoints: Vec::new(),
            maximum_prefill_waves,
            next_prefill_wave: AtomicUsize::new(0),
            armed: AtomicBool::new(false),
            plan_id: "plan.test".to_owned(),
            plan_hash: "plan-hash".to_owned(),
            model_id: "model.test".to_owned(),
            family_fingerprint: "family-hash".to_owned(),
            program_fingerprint: "program-hash".to_owned(),
            run_id: "run.test".to_owned(),
        }
    }

    #[test]
    fn absent_capture_config_preserves_the_default_path() {
        assert!(VNextCheckpointSelection::from_config(None)
            .unwrap()
            .is_none());
    }

    #[test]
    fn capture_config_is_canonical_and_rejects_invalid_bounds() {
        let directory = tempfile::tempdir().unwrap();
        let config = VNextCheckpointCaptureConfig {
            output_dir: directory.path().join("capture"),
            value_ids: vec!["value.z".to_owned(), "value.a".to_owned()],
            maximum_prefill_waves: 2,
        };
        let selection = VNextCheckpointSelection::from_config(Some(&config))
            .unwrap()
            .unwrap();
        assert_eq!(
            selection
                .value_ids
                .iter()
                .map(ProgramValueId::as_str)
                .collect::<Vec<_>>(),
            ["value.a", "value.z"]
        );

        let duplicate = VNextCheckpointCaptureConfig {
            value_ids: vec!["value.a".to_owned(), "value.a".to_owned()],
            ..config.clone()
        };
        assert!(VNextCheckpointSelection::from_config(Some(&duplicate)).is_err());
        let zero_waves = VNextCheckpointCaptureConfig {
            maximum_prefill_waves: 0,
            ..config
        };
        assert!(VNextCheckpointSelection::from_config(Some(&zero_waves)).is_err());
    }

    #[test]
    fn capture_only_claims_bounded_waves_after_startup_arm() {
        let capture = capture(2);
        assert_eq!(capture.claim_prefill_wave(), None);
        capture.arm();
        assert_eq!(capture.claim_prefill_wave(), Some(0));
        assert_eq!(capture.claim_prefill_wave(), Some(1));
        assert_eq!(capture.claim_prefill_wave(), None);
    }

    #[test]
    fn evidence_directory_and_files_are_create_once() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("capture");
        prepare_empty_output_directory(&output).unwrap();
        let evidence = output.join("evidence.bin");
        write_new_file(&evidence, b"first").unwrap();
        assert!(write_new_file(&evidence, b"replacement").is_err());
        assert!(prepare_empty_output_directory(&output).is_err());
        assert_eq!(fs::read(evidence).unwrap(), b"first");
    }
}
