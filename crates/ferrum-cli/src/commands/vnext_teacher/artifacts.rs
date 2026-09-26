use super::*;
use ferrum_models::{vnext_teacher_token_digest, VNextTeacherEvidenceSink};
use ferrum_types::teacher_capture::*;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::Path,
};

pub(super) fn file_identity(path: &Path) -> Result<VNextTeacherFileIdentity> {
    let path = fs::canonicalize(path)
        .map_err(|error| FerrumError::io(format!("resolve teacher input file: {error}")))?;
    let mut file = File::open(&path).map_err(|error| FerrumError::io(error.to_string()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 65536];
    let mut bytes = 0_u64;
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| FerrumError::io(error.to_string()))?;
        if count == 0 {
            break;
        }
        bytes += count as u64;
        digest.update(&buffer[..count]);
    }
    Ok(VNextTeacherFileIdentity {
        path: path.to_string_lossy().into_owned(),
        bytes,
        sha256: format!("{:x}", digest.finalize()),
    })
}

pub(super) struct Artifacts {
    directory: PathBuf,
    pub(super) manifest: VNextTeacherCaptureManifest,
}

impl Artifacts {
    pub(super) fn create(
        cmd: &VNextTeacherCommand,
        spec: &VNextTeacherExecutionSpec,
    ) -> Result<Self> {
        fs::create_dir_all(&cmd.output_dir).map_err(|error| FerrumError::io(error.to_string()))?;
        if fs::read_dir(&cmd.output_dir)
            .map_err(|error| FerrumError::io(error.to_string()))?
            .next()
            .is_some()
        {
            return Err(FerrumError::config(
                "teacher output directory must be new or empty",
            ));
        }
        let manifest = VNextTeacherCaptureManifest {
            schema_version: REAL_HISTORY_TEACHER_CAPTURE_SCHEMA,
            artifact_type: REAL_HISTORY_TEACHER_CAPTURE_TYPE.into(),
            mode: cmd.mode.as_str().into(),
            output_policy: "unmodified_full_logits_before_sampling".into(),
            identity: None,
            vocabulary_size: 0,
            configuration: serde_json::Value::Null,
            owners: spec
                .owners
                .iter()
                .map(|owner| VNextTeacherOwnerRecord {
                    owner_id: owner.owner_id.clone(),
                    prompt_token_ids: owner.prompt_token_ids.clone(),
                    teacher_token_ids: owner.teacher_token_ids.clone(),
                    prompt_token_ids_sha256: vnext_teacher_token_digest(&owner.prompt_token_ids),
                    teacher_token_ids_sha256: vnext_teacher_token_digest(&owner.teacher_token_ids),
                })
                .collect(),
            waves: Vec::new(),
            decisions: Vec::new(),
            complete: false,
            errors: Vec::new(),
        };
        let artifact = Self {
            directory: cmd.output_dir.clone(),
            manifest,
        };
        let path = artifact.directory.join("manifest.json");
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| FerrumError::io(format!("reserve teacher manifest: {error}")))?;
        artifact.persist()?;
        Ok(artifact)
    }

    pub(super) fn persist(&self) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(&self.manifest).map_err(|error| {
            FerrumError::internal(format!("serialize teacher manifest: {error}"))
        })?;
        // This file was exclusively created in the dedicated empty directory.
        fs::write(self.directory.join("manifest.json"), bytes)
            .map_err(|error| FerrumError::io(format!("write teacher manifest: {error}")))
    }

    pub(super) fn finish(&mut self, error: Option<&FerrumError>) -> Result<()> {
        self.manifest.complete = error.is_none();
        if let Some(error) = error {
            self.manifest.errors.push(error.to_string());
        }
        self.persist()
    }
}

impl VNextTeacherEvidenceSink for Artifacts {
    fn wave(
        &mut self,
        evidence: &VNextTeacherWaveEvidence,
        raw_readbacks: &[Vec<u8>],
        completion_receipt: &serde_json::Value,
    ) -> Result<()> {
        if evidence.wave_index != self.manifest.waves.len() {
            return Err(FerrumError::internal(
                "teacher wave index is missing or repeated",
            ));
        }
        if raw_readbacks.len() != evidence.readbacks.len() {
            return Err(FerrumError::internal(
                "teacher raw readback inventory differs from its receipt",
            ));
        }
        let mut saved = evidence.clone();
        let receipt_file = format!("completion-{:06}.json", evidence.wave_index);
        let receipt_bytes = serde_json::to_vec(completion_receipt).map_err(|error| {
            FerrumError::internal(format!("serialize teacher completion: {error}"))
        })?;
        let mut receipt_file_handle = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.directory.join(&receipt_file))
            .map_err(|error| {
                FerrumError::io(format!("create teacher completion receipt: {error}"))
            })?;
        receipt_file_handle
            .write_all(&receipt_bytes)
            .map_err(|error| {
                FerrumError::io(format!("write teacher completion receipt: {error}"))
            })?;
        saved.completion_receipt = Some(VNextTeacherRawArtifact {
            file: receipt_file,
            bytes: receipt_bytes.len() as u64,
            sha256: format!("{:x}", Sha256::digest(&receipt_bytes)),
        });
        for (index, (readback, bytes)) in saved.readbacks.iter_mut().zip(raw_readbacks).enumerate()
        {
            let sha256 = format!("{:x}", Sha256::digest(bytes));
            if readback.byte_count != bytes.len() || readback.sha256 != sha256 {
                return Err(FerrumError::internal(
                    "teacher raw readback bytes differ from their physical receipt",
                ));
            }
            let file_name = format!("readback-{:06}-{index:03}.bin", evidence.wave_index);
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(self.directory.join(&file_name))
                .map_err(|error| {
                    FerrumError::io(format!("create teacher raw readback: {error}"))
                })?;
            file.write_all(bytes)
                .map_err(|error| FerrumError::io(format!("write teacher raw readback: {error}")))?;
            readback.raw_artifact = Some(VNextTeacherRawArtifact {
                file: file_name,
                bytes: bytes.len() as u64,
                sha256,
            });
        }
        self.manifest.waves.push(saved);
        self.persist()
    }

    fn decision(&mut self, evidence: &VNextTeacherDecisionEvidence, logits: &[f32]) -> Result<()> {
        if logits.len() != self.manifest.vocabulary_size
            || logits.iter().any(|value| !value.is_finite())
        {
            return Err(FerrumError::model(
                "teacher artifact requires exact finite vocabulary logits",
            ));
        }
        let file_name = format!("logits-{:06}.f32le", self.manifest.decisions.len());
        let bytes: Vec<_> = logits
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.directory.join(&file_name))
            .map_err(|error| FerrumError::io(format!("create teacher logits: {error}")))?;
        file.write_all(&bytes)
            .map_err(|error| FerrumError::io(format!("write teacher logits: {error}")))?;
        self.manifest.decisions.push(VNextTeacherDecisionRecord {
            evidence: evidence.clone(),
            logits: VNextTeacherLogitArtifact {
                file: file_name,
                encoding: "f32-le".into(),
                elements: logits.len(),
                bytes: bytes.len() as u64,
                sha256: format!("{:x}", Sha256::digest(&bytes)),
            },
        });
        self.persist()
    }
}
