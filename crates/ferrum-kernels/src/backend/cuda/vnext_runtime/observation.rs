//! Owned numerical observations of the actual CUDA command. No executable,
//! stream, device memory, host transfer payload or lease is retained here.
use super::*;
use ferrum_interfaces::execution_cost::{
    SelectedCommandCostEvidenceV1, StatisticalEvidenceUnknown, StatisticalTransferKindV1,
};
use ferrum_interfaces::vnext::{
    DeviceObservationDiagnostic, DeviceObservationFailureStage, DeviceObservationPacket,
    DeviceObservationTemplate, DeviceObservationTemplateBudget, FrozenObservationInput,
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
        self.project_with_diagnostic(input, &mut None)
    }
    fn project_with_diagnostic(
        &self,
        input: &FrozenObservationInput,
        diagnostic: &mut Option<DeviceObservationDiagnostic>,
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
        if evidence.is_none() {
            let site = match &self.metadata {
                CommandMetadata::Compute(recipe) => recipe.projection_site(),
                CommandMetadata::Transfer { .. } => "command.transfer.project",
                CommandMetadata::ProgramBinding(_) => "command.program_binding.project",
                CommandMetadata::StridedTransfer(_) => "command.strided_transfer.project",
            };
            diagnostic.get_or_insert(DeviceObservationDiagnostic::new(
                DeviceObservationFailureStage::Projection,
                site,
                None,
            ));
        }
        Ok(vec![evidence])
    }
}
fn failure(
    stage: DeviceObservationFailureStage,
    site: &'static str,
    error: Option<StatisticalEvidenceUnknown>,
) -> DeviceObservationDiagnostic {
    DeviceObservationDiagnostic::new(stage, site, error)
}
fn packet(
    metadata: CommandMetadata,
    input: FrozenObservationInput,
    occurrences: u64,
    budget: &Arc<DeviceObservationTemplateBudget>,
) -> Result<DeviceObservationPacket, DeviceObservationDiagnostic> {
    let bounds = (|| {
        let retained = std::mem::size_of::<CommandObservation>().checked_add(match &metadata {
            CommandMetadata::Compute(recipe) => recipe.retained_payload_bytes()?,
            CommandMetadata::Transfer { .. } | CommandMetadata::StridedTransfer(_) => 0,
            CommandMetadata::ProgramBinding(rows) => rows
                .len()
                .checked_mul(std::mem::size_of::<(u64, u64, u64)>())?,
        })?;
        let projected = SelectedCommandCostEvidenceV1::maximum_working_payload_bytes(occurrences)?
            .checked_add(std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>())?
            .checked_add(match &metadata {
                CommandMetadata::Compute(recipe) => recipe.projection_scratch_bytes()?,
                _ => 0,
            })?;
        Some((retained, projected))
    })();
    let (retained, projected) = bounds.ok_or_else(|| {
        failure(
            DeviceObservationFailureStage::Retain,
            "command.packet.bounds",
            None,
        )
    })?;
    budget
        .reserve_with_diagnostic(retained, "command.packet.reserve")?
        .retain(Arc::new(CommandObservation {
            metadata,
            retained,
            projected,
        }))
        .map_err(|error| {
            failure(
                DeviceObservationFailureStage::Retain,
                "command.packet.retain",
                Some(error),
            )
        })?
        .packet(input)
        .map_err(|error| {
            failure(
                DeviceObservationFailureStage::Packet,
                "command.packet.freeze",
                Some(error),
            )
        })
}
pub(super) fn compute(
    recipe: Arc<super::super::vnext_ops::CudaReplayCostRecipe>,
    occurrences: u64,
) -> Result<DeviceObservationPacket, DeviceObservationDiagnostic> {
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
) -> Result<DeviceObservationPacket, DeviceObservationDiagnostic> {
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
) -> Result<DeviceObservationPacket, DeviceObservationDiagnostic> {
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
) -> Result<DeviceObservationPacket, DeviceObservationDiagnostic> {
    let upper = rows
        .len()
        .checked_mul(std::mem::size_of::<(u64, u64, u64)>())
        .and_then(|bytes| std::mem::size_of::<CommandObservation>().checked_add(bytes))
        .ok_or_else(|| {
            failure(
                DeviceObservationFailureStage::Retain,
                "command.program_binding.bounds",
                None,
            )
        })?;
    let construction = upper.checked_mul(2).ok_or_else(|| {
        failure(
            DeviceObservationFailureStage::Reserve,
            "command.program_binding.bound_overflow",
            None,
        )
    })?;
    let reservation =
        budget.reserve_with_diagnostic(construction, "command.program_binding.reserve")?;
    let rows: Box<[_]> = rows.collect();
    let projected = u64::try_from(rows.len())
        .ok()
        .and_then(SelectedCommandCostEvidenceV1::maximum_working_payload_bytes)
        .and_then(|bytes| {
            bytes.checked_add(std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>())
        })
        .ok_or_else(|| {
            failure(
                DeviceObservationFailureStage::Retain,
                "command.program_binding.projected_bound",
                None,
            )
        })?;
    reservation
        .retain(Arc::new(CommandObservation {
            metadata: CommandMetadata::ProgramBinding(rows),
            retained: upper,
            projected,
        }))
        .map_err(|error| {
            failure(
                DeviceObservationFailureStage::Retain,
                "command.program_binding.retain",
                Some(error),
            )
        })?
        .packet(FrozenObservationInput::command(tokens))
        .map_err(|error| {
            failure(
                DeviceObservationFailureStage::Packet,
                "command.program_binding.freeze",
                Some(error),
            )
        })
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
        let mut diagnostic = None;
        let unresolved = ledger(packet, 1)
            .resolve_observation_with_diagnostic(&mut diagnostic)
            .unwrap();
        assert!(unresolved.commands()[0].statistical_evidence().is_none());
        let failure = diagnostic.unwrap();
        assert_eq!(failure.stage, DeviceObservationFailureStage::Projection);
        assert_eq!(failure.error, None);
        assert_eq!(failure.command_index, Some(0));
        assert_eq!(
            failure.native_operation,
            Some(DeviceNativeOperationId::new("actual.cuda.binding").unwrap())
        );
    }
    #[test]
    fn cuda_observation_failed_reserve_reports_original_budget_and_releases() {
        let budget = DeviceObservationTemplateBudget::new(4096).unwrap();
        let lease = budget.reserve(0).unwrap();
        let overhead = budget.retained_payload_bytes();
        drop(lease);
        let held = budget.reserve(budget.maximum_bytes() - overhead).unwrap();
        let failure = program_binding([(16, 8, 2)].into_iter(), 3, &budget).unwrap_err();
        assert_eq!(failure.stage, DeviceObservationFailureStage::Reserve);
        assert_eq!(failure.error, Some(StatisticalEvidenceUnknown::Capacity));
        let snapshot = failure.budget.unwrap();
        assert_eq!(snapshot.current, 4096);
        assert_eq!(snapshot.maximum, 4096);
        assert!(snapshot.required.unwrap() > 0);
        assert_eq!(budget.retained_payload_bytes(), 4096);
        // Use the actual command producer and attribution fallback: a failed
        // numerical sidecar cannot erase the completed physical command.
        let mut command = super::super::tests::command("actual.cuda.transfer");
        command.compute_dispatch_count = 0;
        command.transfer_command_count = 1;
        command.core_transfer = Some((
            StatisticalTransferKindV1::HostToDevice,
            8,
            ferrum_types::SloStructuredCostCapture::HostSettledV1,
        ));
        command.prepare_core_observation(&budget);
        let raw = cuda_submission_attribution(
            &[DeviceCommandPhase::DynamicBinding],
            &[Some(0)],
            &[command],
            &[DeviceExecutionPath::Eager],
            None,
            Vec::new(),
        )
        .unwrap();
        let mut diagnostic = None;
        let resolved = raw
            .resolve_observation_with_diagnostic(&mut diagnostic)
            .unwrap();
        assert_eq!(resolved.commands().len(), 1);
        assert!(resolved.commands()[0].statistical_evidence().is_none());
        assert_eq!(
            diagnostic.unwrap().budget.unwrap().current,
            snapshot.current
        );
        drop(held);
        assert_eq!(budget.retained_payload_bytes(), 0);
        let packet = program_binding([(16, 8, 2)].into_iter(), 3, &budget).unwrap();
        assert!(budget.retained_payload_bytes() > 0);
        drop(packet);
        assert_eq!(budget.retained_payload_bytes(), 0);
    }
}
