//! Owned numerical observations of the actual CUDA command. No executable,
//! stream, device memory, host transfer payload or lease is retained here.
use super::*;
use ferrum_interfaces::execution_cost::{
    SelectedCommandCostEvidenceV1, StatisticalEvidenceUnknown, StatisticalTransferKindV1,
};
use ferrum_interfaces::vnext::{
    DeviceObservationPacket, DeviceObservationTemplate, DeviceObservationTemplateBudget,
    FrozenObservationInput,
};

enum CommandMetadata {
    Compute(Arc<super::super::vnext_ops::CudaReplayCostRecipe>),
    Transfer {
        kind: StatisticalTransferKindV1,
        bytes: u64,
    },
    ProgramBinding(Box<[(u64, u64, u64)]>),
    StridedTransfer(StridedCopyRegion),
}
struct CommandObservation {
    metadata: CommandMetadata,
    retained: usize,
    projected: usize,
}
impl DeviceObservationTemplate for CommandObservation {
    fn command_count(&self) -> usize {
        1
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        Some(self.retained)
    }
    fn projection_retained_bytes_upper_bound(&self) -> Option<usize> {
        Some(self.projected)
    }
    fn project(
        &self,
        input: &FrozenObservationInput,
    ) -> Result<Vec<Option<SelectedCommandCostEvidenceV1>>, StatisticalEvidenceUnknown> {
        let capture = ferrum_types::SloStructuredCostCapture::HostSettledV1;
        let evidence = match &self.metadata {
            CommandMetadata::Compute(recipe) => recipe.project_observation(input),
            CommandMetadata::Transfer { kind, bytes } => {
                selected_cost::transfer(*kind, *bytes, input.tokens(), capture)
            }
            CommandMetadata::ProgramBinding(rows) => {
                selected_cost::program_binding(rows.iter().copied(), input.tokens(), capture)
            }
            CommandMetadata::StridedTransfer(region) => {
                selected_cost::strided_transfer(*region, input.tokens(), capture)
            }
        };
        Ok(vec![evidence])
    }
}
fn packet(
    metadata: CommandMetadata,
    input: FrozenObservationInput,
    occurrences: u64,
    budget: &Arc<DeviceObservationTemplateBudget>,
) -> Option<DeviceObservationPacket> {
    let retained = std::mem::size_of::<CommandObservation>().checked_add(match &metadata {
        CommandMetadata::Compute(recipe) => recipe.retained_payload_bytes()?,
        CommandMetadata::Transfer { .. } | CommandMetadata::StridedTransfer(_) => 0,
        CommandMetadata::ProgramBinding(rows) => {
            rows.len()
                .checked_mul(std::mem::size_of::<(u64, u64, u64)>())?
        }
    })?;
    let projected = SelectedCommandCostEvidenceV1::maximum_working_payload_bytes(occurrences)?
        .checked_add(std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>())?;
    let projected = projected.checked_add(match &metadata {
        CommandMetadata::Compute(recipe) => recipe.projection_scratch_bytes()?,
        _ => 0,
    })?;
    budget
        .reserve(retained)
        .ok()?
        .retain(Arc::new(CommandObservation {
            metadata,
            retained,
            projected,
        }))
        .ok()?
        .packet(input)
        .ok()
}
pub(super) fn compute(
    recipe: Arc<super::super::vnext_ops::CudaReplayCostRecipe>,
    occurrences: u64,
) -> Option<DeviceObservationPacket> {
    let input = recipe.captured_input().clone();
    let budget = Arc::clone(recipe.budget());
    packet(
        CommandMetadata::Compute(recipe),
        input,
        occurrences,
        &budget,
    )
}
pub(super) fn transfer(
    kind: StatisticalTransferKindV1,
    bytes: u64,
    tokens: u64,
    budget: &Arc<DeviceObservationTemplateBudget>,
) -> Option<DeviceObservationPacket> {
    packet(
        CommandMetadata::Transfer { kind, bytes },
        FrozenObservationInput::command(tokens),
        1,
        budget,
    )
}
pub(super) fn strided_transfer(
    region: StridedCopyRegion,
    tokens: u64,
    budget: &Arc<DeviceObservationTemplateBudget>,
) -> Option<DeviceObservationPacket> {
    packet(
        CommandMetadata::StridedTransfer(region),
        FrozenObservationInput::command(tokens),
        1,
        budget,
    )
}

pub(super) fn program_binding(
    rows: impl ExactSizeIterator<Item = (u64, u64, u64)>,
    tokens: u64,
    budget: &Arc<DeviceObservationTemplateBudget>,
) -> Option<DeviceObservationPacket> {
    let upper = std::mem::size_of::<CommandObservation>().checked_add(
        rows.len()
            .checked_mul(std::mem::size_of::<(u64, u64, u64)>())?,
    )?;
    let reservation = budget.reserve(upper.checked_mul(2)?).ok()?;
    let rows: Box<[_]> = rows.collect();
    let count = u64::try_from(rows.len()).ok()?;
    let projected = SelectedCommandCostEvidenceV1::maximum_working_payload_bytes(count)?
        .checked_add(std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>())?;
    reservation
        .retain(Arc::new(CommandObservation {
            metadata: CommandMetadata::ProgramBinding(rows),
            retained: upper,
            projected,
        }))
        .ok()?
        .packet(FrozenObservationInput::command(tokens))
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ledger(packet: DeviceObservationPacket, transfers: u64) -> DeviceSubmissionAttribution {
        DeviceSubmissionAttribution::new(vec![DeviceNativeWorkAttribution::new(
            0,
            Some(0),
            ferrum_interfaces::vnext::DeviceCommandPhase::DynamicBinding,
            DeviceNativeOperationId::new("actual.cuda.binding").unwrap(),
            DeviceExecutionPath::Eager,
            DeviceBatchingForm::ParticipantLoop,
            3,
            3,
            0,
            transfers,
            None,
        )
        .unwrap()
        .with_observation(packet)
        .unwrap()])
        .unwrap()
    }
    #[test]
    fn cuda_observation_owned_binding_snapshot_survives_later_source_mutation() {
        let mut source = vec![(16, 8, 2), (8, 8, 1)];
        let packet = program_binding(
            source.iter().copied(),
            3,
            &DeviceObservationTemplateBudget::new(4096).unwrap(),
        )
        .unwrap();
        let raw = ledger(packet, 2);
        source[0] = (4096, 4096, 32);
        source.clear();
        assert!(raw.has_unresolved_observation());
        assert!(raw.commands()[0].statistical_evidence().is_none());
        let bound = raw.maximum_resolved_bytes().unwrap();
        let resolved = raw.resolve_observation().unwrap();
        assert!(resolved.retained_payload_bytes().unwrap() <= bound);
        let selected = resolved.commands()[0].statistical_evidence().unwrap();
        selected.validate_command(3, 0, 2).unwrap();
        assert_eq!(selected.work().host_to_device_bytes, 24);
        selected
            .algorithm_work()
            .unwrap()
            .unwrap()
            .validate_command(selected)
            .unwrap();
    }
    #[test]
    fn cuda_observation_exact_actual_count_mismatch_cannot_become_known() {
        let packet = program_binding(
            [(16, 8, 2), (8, 8, 1)].into_iter(),
            3,
            &DeviceObservationTemplateBudget::new(4096).unwrap(),
        )
        .unwrap();
        assert_eq!(
            ledger(packet, 1).resolve_observation().unwrap_err(),
            StatisticalEvidenceUnknown::CommandMismatch
        );
        let packet = program_binding(
            [(4, 8, 2)].into_iter(),
            3,
            &DeviceObservationTemplateBudget::new(4096).unwrap(),
        )
        .unwrap();
        let unresolved = ledger(packet, 1).resolve_observation().unwrap();
        assert!(unresolved.commands()[0].statistical_evidence().is_none());
    }
}
