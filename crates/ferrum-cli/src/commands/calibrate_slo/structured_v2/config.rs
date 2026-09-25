use super::*;
use ferrum_engine::continuous_engine::{CalibrationSession, StructuredCalibrationOptionsV2};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    windows::{CohortPlanV2, CohortRequestV2, CohortV2, MembershipRuleV2},
    StructuredScopeV2,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::num::{NonZeroU64, NonZeroUsize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CaptureConfigV2 {
    /// Complete requests executed after options freeze and before source opening.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(in crate::commands::calibrate_slo) warmup: Vec<manifest::Cohort>,
    pub profile: PathBuf,
    pub source: PathBuf,
    pub scope: StructuredScopeV2,
    pub membership_rule: MembershipRuleV2,
    pub settings: structured::Settings,
    pub phase_members: [usize; 3],
    pub maximum_offered_waves: NonZeroUsize,
    pub maximum_file_bytes: NonZeroU64,
    pub declared_source_clock_error_ns: u64,
}
impl CaptureConfigV2 {
    pub(crate) fn validate(&self, manifest: &manifest::Manifest) -> Result<()> {
        let bad = |reason| FerrumError::config(format!("structured V2 declaration: {reason:?}"));
        self.settings.core().validate().map_err(bad)?;
        self.scope.validate().map_err(bad)?;
        self.membership_rule.validate().map_err(bad)?;
        if self.scope.owner != self.membership_rule.owner
            || self.scope.owner.rows as usize > manifest.protocol.maximum_requests.get()
            || self.maximum_offered_waves.get() > 65_536
            || self.maximum_offered_waves.get() as u64
                > manifest.protocol.maximum_wave_attempts.get()
            || self.maximum_file_bytes.get()
                > ferrum_types::SloCostProfileImportConfig::MAX_FILE_BYTES as u64
            || self.phase_members.iter().any(|n| {
                *n < self.settings.min_phase_samples || *n > self.settings.max_phase_samples
            })
            || self.phase_members.iter().sum::<usize>() > self.maximum_offered_waves.get()
            || manifest.reference.is_some()
            || manifest.validation_model.residual().is_empty()
            || manifest.validation_model.residual().len() > 256
        {
            return Err(FerrumError::config("V2 capture requires one frozen owner/window rule, independent complete cohorts and bounded fixed populations"));
        }
        Ok(())
    }
    pub(super) fn options(
        &self,
        manifest: &manifest::Manifest,
        inputs: &inputs::PreparedInputs,
        session: &CalibrationSession,
    ) -> Result<StructuredCalibrationOptionsV2> {
        self.validate(manifest)?;
        let plan = cohort_plan(manifest, |index, phase| {
            let (request, _) = inputs.request_for_phase(
                manifest,
                index,
                &session.configuration().model.model_id,
                session
                    .configuration()
                    .sampling
                    .default_params
                    .model_output_protocol,
                phase,
            )?;
            Ok(request.sampling_params.max_tokens as u64)
        })?;
        let payload = serde_json::to_value(manifest)
            .map_err(|e| FerrumError::config(format!("serialize V2 manifest: {e}")))?;
        // This declaration constructs no live owners. The real add_request hook
        // verifies every slot budget against the independently created request.
        let protocol = plan
            .signature(&payload)
            .map_err(|e| FerrumError::config(format!("V2 cohort manifest: {e:?}")))?;
        let mut hash = Sha256::new();
        hash.update(b"ferrum.cli.structured-source.v2\0");
        hash.update(protocol);
        Ok(StructuredCalibrationOptionsV2 {
            observations_path: self.source.clone(),
            protocol_sha256: hash.finalize().into(),
            scope: self.scope.clone(),
            membership_rule: self.membership_rule.clone(),
            cohort_plan: plan,
            cohort_manifest_payload: payload,
            settings: self.settings.core(),
            phase_members: self.phase_members,
            maximum_offered_waves: self.maximum_offered_waves,
            maximum_file_bytes: self.maximum_file_bytes,
        })
    }
}
/// Pure declaration shared by the real request factory and Rust protocol tests.
/// Each duplicate prompt and each repetition retains a distinct immutable slot.
pub(super) fn cohort_plan(
    manifest: &manifest::Manifest,
    mut maximum: impl FnMut(usize, super::super::report::Phase) -> Result<u64>,
) -> Result<CohortPlanV2> {
    use super::super::report::Phase;
    let mut phases: [Vec<CohortV2>; 3] = std::array::from_fn(|_| Vec::new());
    for (phase, (cases, label)) in [
        (&manifest.training[..], Phase::Training),
        (manifest.validation_model.residual(), Phase::Residual),
        (&manifest.validation[..], Phase::Qualification),
    ]
    .into_iter()
    .enumerate()
    {
        for (index, case) in cases.iter().enumerate() {
            for repetition in 0..case.repetitions.get() {
                let requests = case
                    .prompts
                    .iter()
                    .map(|&prompt| {
                        Ok(CohortRequestV2 {
                            manifest_prompt: prompt as u32,
                            maximum_output: maximum(prompt, label)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                phases[phase].push(CohortV2 {
                    manifest_case: index as u32,
                    repetition: repetition as u32,
                    requests,
                });
            }
        }
    }
    let plan = CohortPlanV2 { phases };
    plan.validate()
        .map_err(|e| FerrumError::config(format!("V2 complete cohort declaration: {e:?}")))?;
    Ok(plan)
}
