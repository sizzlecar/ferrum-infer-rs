use super::*;
use std::{fs::OpenOptions, io::Write};

/// A loader-validated reference unit, not a bound on future latency or an
/// assertion of cost-model/heldout coverage. Hashes bind bytes, not hardware.
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationReferenceArtifact {
    pub schema_version: u32,
    pub reference_domain: Option<(NonZeroU32, NonZeroU32)>,
    pub path: PathBuf,
    pub sha256: [u8; 32],
    pub bytes: u64,
    pub protocol_sha256: [u8; 32],
    pub plan_sha256: [u8; 32],
    pub frozen_accepted_ordinal: u64,
    pub training_accepted_ordinal: u64,
    pub source_sha256: [u8; 32],
    pub discovery_records: Vec<ReferenceRecordId>,
    pub tau_ref_ns: u64,
    pub reference_revision: NonZeroU64,
}

pub(in crate::continuous_engine::inner::calibration) struct MemoryReferenceArtifact {
    pub loaded: Arc<LoadedPrefillReference>,
    pub bytes: Vec<u8>,
    pub plan_sha256: [u8; 32],
    pub frozen_accepted_ordinal: u64,
    pub training_accepted_ordinal: u64,
    pub source_sha256: [u8; 32],
    pub discovery_records: Vec<ReferenceRecordId>,
}

impl MemoryReferenceArtifact {
    fn publish(self, destination: &Path) -> Result<CalibrationReferenceArtifact> {
        let path = publish(destination, &self.bytes)?;
        let identity = self.loaded.identity();
        Ok(CalibrationReferenceArtifact {
            schema_version: if self.loaded.piecewise_domain().is_some() {
                PREFILL_REFERENCE_SCHEMA_V2
            } else {
                PREFILL_REFERENCE_SCHEMA_V1
            },
            reference_domain: self.loaded.piecewise_domain(),
            path,
            sha256: identity.artifact_sha256,
            bytes: self.bytes.len() as u64,
            protocol_sha256: identity.protocol_sha256,
            plan_sha256: self.plan_sha256,
            frozen_accepted_ordinal: self.frozen_accepted_ordinal,
            training_accepted_ordinal: self.training_accepted_ordinal,
            source_sha256: self.source_sha256,
            discovery_records: self.discovery_records,
            tau_ref_ns: self.loaded.tau_ref_ns().get(),
            reference_revision: identity.revision,
        })
    }
}

impl CalibrationReferenceCollector {
    /// Cold-path original-source join. The immutable cut must precede heldout
    /// validation; do not pass the later shutdown export. Every required
    /// discovery/preparation/trial receipt must still be retained and Recorded.
    pub fn finish(
        self,
        cut: &CalibrationProfileArtifact,
        destination: &Path,
    ) -> Result<CalibrationReferenceArtifact> {
        let requested = self.requested_witnesses(cut.accepted_ordinal)?;
        let joined = source::read(cut, &self.fingerprint, &requested)?;
        self.assemble_from_joined(joined, cut.accepted_ordinal, cut.source_digest, destination)
    }

    pub(super) fn requested_witnesses(
        &self,
        accepted_ordinal: u64,
    ) -> Result<BTreeMap<u64, &Witness>> {
        let required = self
            .plan
            .curves
            .len()
            .checked_add(1)
            .and_then(|n| n.checked_mul(self.plan.protocol.repetitions.get()))
            .ok_or_else(|| invalid("reference trial count overflow"))?;
        if self.trials.len() != required
            || self.trials.values().any(|trial| !trial.complete)
            || accepted_ordinal <= self.frozen_accepted_ordinal
        {
            return Err(invalid("reference trials/cut are incomplete"));
        }
        let mut requested = BTreeMap::new();
        for witness in self
            .discovery
            .iter()
            .map(|row| &row.witness)
            .chain(self.trials.values().flat_map(|trial| trial.samples.iter()))
        {
            if witness.accepted > accepted_ordinal
                || requested.insert(witness.accepted, witness).is_some()
            {
                return Err(invalid(
                    "reference source ordinal is duplicated or outside cut",
                ));
            }
        }
        Ok(requested)
    }

    pub(super) fn assemble_from_joined(
        self,
        joined: source::Joined,
        accepted_ordinal: u64,
        source_digest: [u8; 32],
        destination: &Path,
    ) -> Result<CalibrationReferenceArtifact> {
        self.assemble_memory(joined, accepted_ordinal, source_digest)?
            .publish(destination)
    }

    pub(super) fn assemble_memory(
        self,
        joined: source::Joined,
        accepted_ordinal: u64,
        source_digest: [u8; 32],
    ) -> Result<MemoryReferenceArtifact> {
        let record = |witness: &Witness| -> Result<ReferenceRecordId> {
            let sample = joined
                .samples
                .get(&witness.accepted)
                .ok_or_else(|| invalid("reference source join missing"))?;
            Ok(ReferenceRecordId {
                source_sha256: source_digest,
                ordinal: sample.source_record,
            })
        };
        let observed =
            |witness: &Witness, origin, previous_record| -> Result<ReferenceObservedSample> {
                let (prefix_before, prefix_after, generated_before, generated_after) =
                    match witness.commit.work {
                        CalibrationCommittedWork::Prefill {
                            start,
                            end,
                            generated_before,
                            generated_after,
                            ..
                        } => (start, end, generated_before, generated_after),
                        CalibrationCommittedWork::Decode {
                            kv_before,
                            kv_after,
                            generated_before,
                            generated_after,
                        } => (kv_before, kv_after, generated_before, generated_after),
                    };
                Ok(ReferenceObservedSample {
                    record: record(witness)?,
                    observation: joined.samples[&witness.accepted].clone(),
                    commit: ReferenceCommitReceipt {
                        owner_incarnation: NonZeroU64::new(witness.commit.owner_incarnation)
                            .ok_or_else(|| invalid("zero reference owner"))?,
                        work_generation: NonZeroU64::new(witness.commit.work_generation)
                            .ok_or_else(|| invalid("zero reference generation"))?,
                        origin,
                        previous_record,
                        prefix_before,
                        prefix_after,
                        generated_before: u32::try_from(generated_before)
                            .map_err(|_| invalid("reference generated count overflow"))?,
                        generated_after: u32::try_from(generated_after)
                            .map_err(|_| invalid("reference generated count overflow"))?,
                    },
                })
            };
        let mut builder = ReferenceCalibrationBuilder::new(
            self.plan.reference_revision,
            self.fingerprint.clone(),
            self.plan.protocol.clone(),
            joined.generated_unix_ns,
            self.plan.limits.clone(),
        )
        .map_err(reference_error)?;
        if let Some(spec) = &self.plan.piecewise {
            builder = builder
                .with_piecewise(spec.clone())
                .map_err(reference_error)?;
        }
        let mut decode = Vec::with_capacity(self.plan.protocol.repetitions.get());
        for repetition in 0..self.plan.protocol.repetitions.get() {
            let trial = &self.trials[&CalibrationReferenceTrial::Decode { repetition }];
            if trial.selected.len() != 1 {
                return Err(invalid("decode unit has multiple selected measurements"));
            }
            decode.push(observed(
                &trial.samples[trial.selected[0]],
                ReferenceStateOrigin::PreparedDecode,
                None,
            )?);
        }
        builder
            .set_decode_samples(decode)
            .map_err(reference_error)?;
        for (curve_index, curve) in self.plan.curves.iter().enumerate() {
            let mut trials = Vec::with_capacity(self.plan.protocol.repetitions.get());
            for repetition in 0..self.plan.protocol.repetitions.get() {
                let trial = &self.trials[&CalibrationReferenceTrial::Prefill {
                    curve: curve_index,
                    repetition,
                }];
                let mut samples = Vec::with_capacity(trial.selected.len());
                let mut previous = None;
                for (index, &selected) in trial.selected.iter().enumerate() {
                    let sample = observed(
                        &trial.samples[selected],
                        if index == 0 {
                            ReferenceStateOrigin::Fresh
                        } else {
                            ReferenceStateOrigin::CommittedContinuation
                        },
                        previous,
                    )?;
                    previous = Some(sample.record);
                    samples.push(sample);
                }
                trials.push(ReferencePrefillTrial {
                    trial_index: repetition,
                    input_tokens_sha256: curve.input_tokens_sha256,
                    samples,
                });
            }
            builder
                .add_curve(ReferenceCurveInput {
                    total_prompt_tokens: curve.total_prompt_tokens,
                    partition: curve.partition.clone(),
                    trials,
                })
                .map_err(reference_error)?;
        }
        let bytes = builder.finish_bytes().map_err(reference_error)?;
        let protocol_sha256 = self.plan.protocol_sha256().map_err(reference_error)?;
        // The exact product loader runs before publication; no duplicate
        // interpretation or fabricated runtime scoring curve is introduced.
        let loaded = load_prefill_reference_bytes(
            &bytes,
            &self.fingerprint.clone().into(),
            protocol_sha256,
            &self.plan.limits,
        )
        .map_err(reference_error)?;
        let discovery_records = self
            .discovery
            .iter()
            .map(|row| record(&row.witness))
            .collect::<Result<Vec<_>>>()?;
        Ok(MemoryReferenceArtifact {
            loaded,
            bytes,
            plan_sha256: self.plan_sha256,
            frozen_accepted_ordinal: self.frozen_accepted_ordinal,
            training_accepted_ordinal: accepted_ordinal,
            source_sha256: source_digest,
            discovery_records,
        })
    }
}

pub(super) fn publish(destination: &Path, bytes: &[u8]) -> Result<PathBuf> {
    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()
        .map_err(source::io_error)?;
    let leaf = destination
        .file_name()
        .ok_or_else(|| invalid("reference destination needs a filename"))?;
    let destination = parent.join(leaf);
    struct Temporary(PathBuf);
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let temp = Temporary(parent.join(format!(".ferrum-reference-{}.tmp", uuid::Uuid::new_v4())));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp.0)
        .map_err(source::io_error)?;
    file.write_all(bytes).map_err(source::io_error)?;
    file.sync_all().map_err(source::io_error)?;
    // Same-directory atomic no-replace publication, including alias races.
    std::fs::hard_link(&temp.0, &destination).map_err(source::io_error)?;
    Ok(destination)
}
