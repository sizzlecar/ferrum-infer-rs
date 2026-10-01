//! Controlled cumulative policy experiments, not historical binary aliases.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloExperimentStageV1 {
    /// Current adaptive scheduler with the same native operators and mandatory
    /// resource/identity/output correctness fixes as every subsequent stage.
    ControlledAdaptiveBaseline,
    SingleWave,
    OutputIsolation,
    DeadlineOnly,
    CostCandidates,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SloControllerPolicy {
    Legacy,
    ObserveCostCandidates,
    DeadlineOnly,
    CostCandidates,
}

/// One mapping from declared policy, shared by all engine entrypoints.
/// These switches are not execution/resource authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SloExecutionPolicy {
    pub single_wave: bool,
    pub controller: SloControllerPolicy,
    pub cost_observation: bool,
    pub time_admission: bool,
}

impl SloExperimentStageV1 {
    pub fn output_transport(self) -> SloOutputTransport {
        match self {
            Self::ControlledAdaptiveBaseline | Self::SingleWave => SloOutputTransport::Legacy,
            _ => SloOutputTransport::Credited,
        }
    }
}

impl SloConfig {
    pub fn execution_policy(&self) -> SloExecutionPolicy {
        use SloControllerPolicy as Controller;
        use SloExperimentStageV1 as Stage;
        match self.experiment_stage {
            None => SloExecutionPolicy {
                single_wave: self.mode == SloMode::Enforce,
                controller: match self.mode {
                    SloMode::Off => Controller::Legacy,
                    SloMode::Observe => Controller::ObserveCostCandidates,
                    SloMode::Enforce => Controller::CostCandidates,
                },
                cost_observation: self.mode != SloMode::Off,
                // Observe still audits the existing hypothetical time decision.
                time_admission: self.mode != SloMode::Off,
            },
            Some(stage) => SloExecutionPolicy {
                single_wave: stage != Stage::ControlledAdaptiveBaseline,
                controller: match stage {
                    Stage::ControlledAdaptiveBaseline
                    | Stage::SingleWave
                    | Stage::OutputIsolation => Controller::Legacy,
                    Stage::DeadlineOnly => Controller::DeadlineOnly,
                    Stage::CostCandidates | Stage::Complete => Controller::CostCandidates,
                },
                cost_observation: matches!(stage, Stage::CostCandidates | Stage::Complete),
                time_admission: stage == Stage::Complete,
            },
        }
    }

    pub(super) fn validate_experiment(&self) -> Result<(), String> {
        let Some(stage) = self.experiment_stage else {
            return Ok(());
        };
        if self.mode != SloMode::Enforce
            || self.admission.time_policy != SloTimeAdmissionPolicy::CompleteRequests
        {
            return Err("SLO stage experiments require Enforce and CompleteRequests".into());
        }
        if self.output.transport != stage.output_transport() {
            return Err(format!(
                "SLO stage {stage:?} requires {:?} output transport",
                stage.output_transport()
            ));
        }
        if !self.execution_policy().cost_observation {
            let cost = &self.cost_observation;
            if self.cost_profile.is_some()
                || self.prefill_reference.is_some()
                || self.required_query_observation.enabled()
                || cost.profile_export.is_some()
                || !cost.live_structured_calibration.is_disabled()
                || !cost.selected_feedback.is_disabled()
                || !cost.structured_feedback.is_disabled()
                || !cost.prospective_structured_capture.is_disabled()
                || !cost.structured_capture.is_disabled()
            {
                return Err("pre-cost SLO stages require no cost/reference artifacts, calibration, feedback or cost capture consumers".into());
            }
        }
        Ok(())
    }

    /// Prefix waiting belongs to the complete stage. Check the actual resolved
    /// scheduler value, including values supplied outside the SLO config file.
    pub fn validate_experiment_prefix(&self, wait: Option<NonZeroU64>) -> Result<(), String> {
        if wait.is_some()
            && self
                .experiment_stage
                .is_some_and(|stage| stage != SloExperimentStageV1::Complete)
        {
            return Err("prefix rendezvous waiting conflicts with a pre-complete SLO stage".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
