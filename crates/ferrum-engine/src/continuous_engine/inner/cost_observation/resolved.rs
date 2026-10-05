//! One private, immutable projection shared by this FIFO position's consumers.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::{
    statistical::model::ModelUnknown,
    structured_v2::{StructuredInputV2, StructuredQueryV2, StructuredUnknownV2},
    ExecutionFingerprint, WaveObservationOutcome,
};

#[derive(Debug, Clone, Copy)]
pub(super) enum ProjectionError {
    Settlement(ModelUnknown),
    MissingRecipe,
    Numeric(StructuredUnknownV2),
}
pub(super) struct StructuredActual {
    pub actual: Arc<trainer::host_content::statistical::CompleteSelectedObservation>,
    /// Source3's original Prepared numerical projection. Its meaning is not
    /// changed by source6's separately settled completion axes.
    pub base_input: StructuredInputV2,
    pub query: StructuredQueryV2,
    pub physical_scope: Result<(), StructuredUnknownV2>,
    pub feedback_scope: Result<(), StructuredUnknownV2>,
    observation_memory: Option<Arc<super::memory::ObservationBytePermit>>,
}
pub(super) struct ResolvedCostEntry {
    entry: CostEvidenceEntry,
    actual:
        Result<Arc<trainer::host_content::statistical::CompleteSelectedObservation>, ModelUnknown>,
    structured: Result<Arc<StructuredActual>, ProjectionError>,
    memory: Option<Arc<super::memory::ObservationBytePermit>>,
    /// Private same-call exclusion, minted while the original call still owns
    /// its complete settlement. Re-projecting a diagnostic DTO cannot mint it.
    preparation_feedback_observed_at: Option<u64>,
    producer_diagnostic: Option<ferrum_interfaces::vnext::DeviceObservationDiagnostic>,
}
impl ResolvedCostEntry {
    pub(super) fn with_producer_diagnostic(
        mut self,
        diagnostic: Option<ferrum_interfaces::vnext::DeviceObservationDiagnostic>,
    ) -> Self {
        self.producer_diagnostic = diagnostic;
        self
    }
    pub(super) fn producer_diagnostic(
        &self,
    ) -> Option<ferrum_interfaces::vnext::DeviceObservationDiagnostic> {
        self.producer_diagnostic
    }
    pub fn new(entry: CostEvidenceEntry) -> Self {
        Self::new_with_domain(entry, None)
    }
    pub fn new_with_domain(
        entry: CostEvidenceEntry,
        domain: Option<&CostWorkloadDomainV1>,
    ) -> Self {
        Self::new_with_domain_and_memory(entry, domain, None)
    }
    pub(super) fn new_with_domain_and_memory(
        mut entry: CostEvidenceEntry,
        domain: Option<&CostWorkloadDomainV1>,
        memory: Option<Arc<super::memory::ObservationBytePermit>>,
    ) -> Self {
        if let Some(memory) = &memory {
            let stages = match &mut entry {
                CostEvidenceEntry::Training { stages, .. } => stages.as_mut(),
                CostEvidenceEntry::StagesOnly { stages, .. } => Some(stages),
            };
            if let Some(stages) = stages {
                Arc::make_mut(stages).observation_memory = Some(Arc::clone(memory));
            }
        }
        let actual =
            trainer::host_content::statistical::complete_actual_observation(&entry).map(Arc::new);
        let structured = actual
            .clone()
            .map_err(ProjectionError::Settlement)
            .and_then(|actual| Self::project_actual(&entry, actual, domain))
            .map(Arc::new);
        Self {
            memory,
            entry,
            actual,
            structured,
            preparation_feedback_observed_at: None,
            producer_diagnostic: None,
        }
    }

    pub(super) fn with_original_preparation(
        mut self,
        proof: Option<&host_stages::CompletePrivateCalibrationSettlement>,
    ) -> Self {
        self.preparation_feedback_observed_at = proof.and_then(|proof| match &self.entry {
            CostEvidenceEntry::StagesOnly {
                stages,
                legacy_rejection: CostCallRejection::CalibrationPreparation,
            } => proof.prefix_observed_at(stages),
            CostEvidenceEntry::StagesOnly {
                stages,
                legacy_rejection: CostCallRejection::Composite,
            } => proof.readiness_observed_at(stages),
            CostEvidenceEntry::Training {
                sample,
                stages: Some(stages),
            } if sample.outcome == WaveObservationOutcome::Completed => {
                proof.readiness_observed_at(stages)
            }
            _ => None,
        });
        self
    }

    pub(super) fn preparation_feedback(&self) -> Option<(&ExecutionFingerprint, u64)> {
        let observed = self.preparation_feedback_observed_at?;
        let stages = match &self.entry {
            CostEvidenceEntry::StagesOnly {
                stages,
                legacy_rejection: CostCallRejection::CalibrationPreparation,
            } => stages,
            CostEvidenceEntry::StagesOnly {
                stages,
                legacy_rejection: CostCallRejection::Composite,
            } => stages,
            CostEvidenceEntry::Training {
                sample,
                stages: Some(stages),
            } if sample.outcome == WaveObservationOutcome::Completed => stages,
            _ => return None,
        };
        Some((stages.fingerprint.as_ref()?, observed))
    }
    pub fn project(entry: &CostEvidenceEntry) -> Result<StructuredActual, ProjectionError> {
        Self::project_with_domain(entry, None)
    }
    pub fn project_with_domain(
        entry: &CostEvidenceEntry,
        domain: Option<&CostWorkloadDomainV1>,
    ) -> Result<StructuredActual, ProjectionError> {
        let actual = trainer::host_content::statistical::complete_actual_observation(entry)
            .map(Arc::new)
            .map_err(ProjectionError::Settlement)?;
        Self::project_actual(entry, actual, domain)
    }
    fn project_actual(
        entry: &CostEvidenceEntry,
        actual: Arc<trainer::host_content::statistical::CompleteSelectedObservation>,
        domain: Option<&CostWorkloadDomainV1>,
    ) -> Result<StructuredActual, ProjectionError> {
        (|| {
            trainer::host_content::statistical::validate_structured_actual(entry, &actual)
                .map_err(ProjectionError::Settlement)?;
            let stages = match &entry {
                CostEvidenceEntry::Training { stages, .. } => stages.as_deref(),
                CostEvidenceEntry::StagesOnly { stages, .. } => Some(stages.as_ref()),
            }
            .ok_or(ProjectionError::Numeric(
                StructuredUnknownV2::MissingEvidence,
            ))?;
            let recipe = actual
                .selected
                .structured_capture()
                .and_then(Result::ok)
                .ok_or(ProjectionError::MissingRecipe)?;
            let terminal_positions = stages
                .rows
                .iter()
                .enumerate()
                .filter_map(|(index, row)| row.terminal.as_ref().map(|_| index as u32))
                .collect::<Vec<_>>();
            let (base_input, scoped_input) = match domain {
                Some(domain) => StructuredInputV2::from_actual_with_domain_projection(
                    &actual.exact,
                    &actual.selected,
                    recipe,
                    domain,
                ),
                None => StructuredInputV2::from_actual(&actual.exact, &actual.selected, recipe)
                    .map(|base| (base, Err(StructuredUnknownV2::MissingEvidence))),
            }
            .map_err(ProjectionError::Numeric)?;
            let scoped = if domain.is_some() {
                let causes = stages
                    .rows
                    .iter()
                    .enumerate()
                    .filter_map(|(position, row)| {
                        row.terminal
                            .as_ref()
                            .map(|terminal| (position as u32, terminal.finish_reason))
                    })
                    .collect::<Vec<_>>();
                scoped_input
                    .and_then(|input| input.with_cost_template_policy(
                        ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1))
                    .and_then(|input| input.with_settled_terminal_causes(&causes))
            } else {
                Err(StructuredUnknownV2::MissingEvidence)
            };
            let (input, physical_scope) = match scoped {
                Ok(input) => (input, Ok(())),
                Err(reason) => (
                    base_input
                        .clone()
                        .with_settled_completion(&terminal_positions)
                        .map_err(ProjectionError::Numeric)?,
                    Err(reason),
                ),
            };
            let feedback_scope = trainer::structured_v2::feedback_scope_for_actual(stages, &actual)
                .and_then(|()| {
                    if input.regression_axes().len() > 4096
                        || input.joint_support_coordinates().len() > 4096
                    {
                        Err(StructuredUnknownV2::Capacity)
                    } else {
                        Ok(())
                    }
                });
            Ok(StructuredActual {
                actual,
                base_input,
                query: StructuredQueryV2::exact(input),
                physical_scope,
                feedback_scope,
                observation_memory: stages.observation_memory.clone(),
            })
        })()
    }
    pub fn shared_actual(
        &self,
    ) -> Result<Arc<trainer::host_content::statistical::CompleteSelectedObservation>, ModelUnknown>
    {
        self.actual.clone()
    }
    pub fn selected_actual(
        &self,
    ) -> Result<Arc<trainer::host_content::statistical::CompleteSelectedObservation>, ModelUnknown>
    {
        let actual = self.shared_actual()?;
        trainer::host_content::statistical::require_legacy_route(&actual)?;
        Ok(actual)
    }
    pub(super) fn with_memory(mut self, memory: Arc<super::memory::ObservationBytePermit>) -> Self {
        self.memory = Some(memory);
        self
    }
    pub(super) fn memory(&self) -> Option<Arc<super::memory::ObservationBytePermit>> {
        self.memory.clone()
    }
    pub(super) fn retained_payload_bytes(&self) -> Option<usize> {
        let mut bytes = std::mem::size_of_val(self);
        match &self.entry {
            CostEvidenceEntry::Training { sample, stages } => {
                bytes = bytes.checked_add(super::memory::shape_bytes(&sample.actual_shape)?)?;
                if let Some(stages) = stages {
                    bytes = bytes.checked_add(stages.retained_payload_bytes()?)?;
                }
            }
            CostEvidenceEntry::StagesOnly { stages, .. } => {
                bytes = bytes.checked_add(stages.retained_payload_bytes()?)?;
            }
        }
        if let Ok(actual) = &self.actual {
            bytes = bytes
                .checked_add(std::mem::size_of_val(actual.as_ref()))?
                .checked_add(super::sealed::canonical_retained_bytes(&actual.exact)?)?
                .checked_add(super::memory::statistics_bytes(&actual.selected)?)?;
        }
        if let Ok(structured) = &self.structured {
            bytes = bytes
                .checked_add(std::mem::size_of_val(structured.as_ref()))?
                .checked_add(structured.base_input.retained_numeric_bytes()?)?
                .checked_add(structured.query.retained_payload_bytes()?)?;
        }
        bytes.checked_add(8 * std::mem::size_of::<usize>())
    }
    pub fn entry(&self) -> &CostEvidenceEntry {
        &self.entry
    }
    pub fn into_entry(self) -> CostEvidenceEntry {
        self.entry
    }
    pub fn structured(&self) -> Result<&StructuredActual, ProjectionError> {
        self.structured.as_deref().map_err(|e| *e)
    }
    pub fn shared_projection(&self) -> Result<Arc<StructuredActual>, ProjectionError> {
        self.structured.clone()
    }
}
