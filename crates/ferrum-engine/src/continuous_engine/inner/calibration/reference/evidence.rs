use super::*;

pub struct CalibrationReferenceDiscoverySample {
    pub(super) session: Arc<()>,
    pub(super) input: CalibrationRequestEvidence,
    pub(super) witness: Witness,
}
impl CalibrationReferenceDiscoverySample {
    pub fn shape(&self) -> ProfileWaveShapeV2 {
        ProfileWaveShapeV2::from(&self.witness.sample.actual_shape)
    }
    pub fn host_features(&self) -> HostCostFeaturesV1 {
        self.witness.host
    }
    pub fn request_evidence(&self) -> &CalibrationRequestEvidence {
        &self.input
    }
    pub fn accepted_ordinal(&self) -> u64 {
        self.witness.accepted
    }
}

pub(super) struct Witness {
    pub accepted: u64,
    pub sample: WaveCostObservation,
    pub host: HostCostFeaturesV1,
    pub commit: CalibrationCommittedRow,
}
impl Witness {
    pub fn capture(report: &CalibrationWaveReport) -> Result<Self> {
        let CalibrationObservation::Observed {
            sample,
            actual_rows,
            commits,
            host_features,
            accepted_ordinal: Some(accepted),
            disposition: CalibrationQueueDisposition::Published,
        } = &report.observation
        else {
            return Err(invalid("reference wave has no accepted actual observation"));
        };
        if report.submission != CalibrationSubmissionState::HostReconciled
            || report.error.is_some()
            || *accepted == 0
            || actual_rows.len() != 1
            || commits.len() != 1
            || host_features.len() != 1
            || sample.outcome != WaveObservationOutcome::Completed
            || sample.boundary != CostBoundary::PreparationToCommit
            || sample.timing.wall_total_ns == 0
            || sample.timing.stages.restore.is_some()
            || sample.timing.stages.maintenance.is_some()
        {
            return Err(invalid(
                "reference requires one successfully committed isolated actual row",
            ));
        }
        let actual = &actual_rows[0];
        let commit = &commits[0];
        let host = host_features[0].ok_or_else(|| invalid("reference host evidence is missing"))?;
        if actual.request_id != commit.request_id
            || actual.owner_incarnation != commit.owner_incarnation
            || actual.work_generation != commit.work_generation
            || actual.input_index != commit.input_index
            || commit.owner_incarnation == 0
            || commit.work_generation == 0
            || commit.committed_at_ns > sample.observed_at_ns
        {
            return Err(invalid(
                "reference actual/commit identity or clock mismatch",
            ));
        }
        let work = match commit.work {
            CalibrationCommittedWork::Prefill {
                start,
                end,
                total_prompt_tokens,
                generated_before,
                generated_after,
            } if start < end
                && end <= total_prompt_tokens
                && generated_before == 0
                && generated_after == u64::from(end == total_prompt_tokens) =>
            {
                ActualRowWork::Prefill {
                    offset: start,
                    count: end - start,
                    total_prompt_tokens,
                }
            }
            CalibrationCommittedWork::Decode {
                kv_before,
                kv_after,
                generated_before,
                generated_after,
            } if kv_before.checked_add(1) == Some(kv_after)
                && generated_before.checked_add(1) == Some(generated_after) =>
            {
                ActualRowWork::Decode {
                    kv_tokens: kv_before,
                }
            }
            _ => return Err(invalid("reference commit lacks exact successful progress")),
        };
        if actual.work != work
            || host.state.generated_tokens_before
                != match commit.work {
                    CalibrationCommittedWork::Prefill {
                        generated_before, ..
                    }
                    | CalibrationCommittedWork::Decode {
                        generated_before, ..
                    } => generated_before,
                }
        {
            return Err(invalid("reference work/host differs from actual commit"));
        }
        let shape = &sample.actual_shape;
        let numeric = shape
            .numeric_features
            .as_ref()
            .ok_or_else(|| invalid("reference requires actual numeric host evidence"))?;
        numeric
            .validate(1)
            .map_err(|_| invalid("reference numeric rows are invalid"))?;
        if shape.decode_kv_tokens.len() + shape.prefill_chunks.len() != 1 {
            return Err(invalid("reference requires a singleton shape"));
        }
        Ok(Self {
            accepted: *accepted,
            sample: (**sample).clone(),
            host,
            commit: commit.clone(),
        })
    }
    pub fn matches(&self, shape: &ProfileWaveShapeV2, host: HostCostFeaturesV1) -> bool {
        self.host == host && ProfileWaveShapeV2::from(&self.sample.actual_shape) == *shape
    }
}
