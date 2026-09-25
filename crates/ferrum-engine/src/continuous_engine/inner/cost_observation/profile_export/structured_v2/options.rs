use super::*;

#[derive(Debug, Clone)]
pub struct StructuredCalibrationOptionsV2 {
    pub observations_path: PathBuf,
    pub protocol_sha256: [u8; 32],
    pub scope: StructuredScopeV2,
    pub membership_rule: MembershipRuleV2,
    pub cohort_plan: CohortPlanV2,
    pub cohort_manifest_payload: serde_json::Value,
    pub settings: StructuredSettingsV2,
    pub phase_members: [usize; 3],
    pub maximum_offered_waves: NonZeroUsize,
    pub maximum_file_bytes: NonZeroU64,
}
impl StructuredCalibrationOptionsV2 {
    pub(super) fn validate(&self) -> Result<(), ExportError> {
        self.settings.validate().map_err(numeric_error)?;
        self.scope.validate().map_err(numeric_error)?;
        self.membership_rule.validate().map_err(numeric_error)?;
        self.cohort_plan.validate().map_err(numeric_error)?;
        self.cohort_plan
            .signature(&self.cohort_manifest_payload)
            .map_err(numeric_error)?;
        if self.protocol_sha256 == [0; 32]
            || self.scope.owner != self.membership_rule.owner
            || self.observations_path.as_os_str().is_empty()
            || self.maximum_offered_waves.get() > 65_536
            || self.phase_members.iter().any(|n| {
                *n < self.settings.min_phase_samples || *n > self.settings.max_phase_samples
            })
            || self.phase_members.iter().sum::<usize>() > self.maximum_offered_waves.get()
        {
            return Err(ExportError::Config(
                "invalid V2 structured population or bounds".into(),
            ));
        }
        let numeric_bytes = self
            .settings
            .max_phase_samples
            .checked_mul(self.settings.max_axes)
            .and_then(|n| n.checked_mul(12 * std::mem::size_of::<f64>()))
            .and_then(|n| {
                self.settings
                    .max_phase_samples
                    .checked_mul(128)
                    .and_then(|rows| {
                        rows.checked_mul(4 * std::mem::size_of::<StructuredHostRowV1>())
                    })
                    .and_then(|host| n.checked_add(host))
            });
        if numeric_bytes.is_none_or(|n| n > 128 * 1024 * 1024) {
            return Err(ExportError::Config(
                "V2 structured numeric storage exceeds 128 MiB".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn protocol_signature(
        &self,
        rule: [u8; 32],
        cohorts: [u8; 32],
    ) -> Result<[u8; 32], ExportError> {
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-live-source.v2\0");
        digest.update(MODEL_REVISION_V2.as_bytes());
        digest.update(self.protocol_sha256);
        digest.update(rule);
        digest.update(cohorts);
        digest.update(serde_json::to_vec(&self.scope)?);
        for value in self.phase_members.into_iter().map(|n| n as u64).chain([
            self.settings.min_phase_samples as u64,
            self.settings.min_fit_redundancy as u64,
            self.settings.max_phase_samples as u64,
            self.settings.max_axes as u64,
            self.settings.max_rank as u64,
            self.settings.max_wave_ns,
            self.settings.max_sample_age_ns,
            self.settings.static_margin_ns,
            self.maximum_offered_waves.get() as u64,
            self.maximum_file_bytes.get(),
        ]) {
            digest.update(value.to_le_bytes());
        }
        Ok(digest.finalize().into())
    }
}
