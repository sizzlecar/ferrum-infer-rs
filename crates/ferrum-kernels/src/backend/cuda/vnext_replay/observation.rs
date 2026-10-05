//! CPU-only recipes sealed beside the actual successfully captured commands.
//! This object owns no CUDA context, graph, library handle, buffer or lease.
use super::*;
use ferrum_interfaces::execution_cost::{
    SelectedCommandCostEvidenceV1, StatisticalEvidenceUnknown,
};
use ferrum_interfaces::vnext::{
    DeviceObservationDiagnostic, DeviceObservationFailureStage, DeviceObservationTemplate,
    FrozenObservationInput,
};

pub(super) struct SegmentObservation {
    recipes: Box<[Option<ferrum_interfaces::vnext::DeviceObservationPacket>]>,
    retained: usize,
    projected: usize,
    first_failure: Option<DeviceObservationDiagnostic>,
}
impl SegmentObservation {
    pub(super) fn new(
        commands: &[CudaDeviceCommand],
        identity: Option<super::super::vnext_ops::CublasHandleApiIdentity>,
    ) -> Result<
        ferrum_interfaces::vnext::RetainedDeviceObservationTemplate,
        DeviceObservationDiagnostic,
    > {
        let failure = |site| {
            DeviceObservationDiagnostic::new(DeviceObservationFailureStage::Segment, site, None)
        };
        if commands.is_empty() || commands.len() > MAX_COST_COMMANDS {
            return Err(failure("segment.command_count"));
        }
        let eligible = |command: &CudaDeviceCommand| {
            command
                .cublas_cost_requirement()
                .matches_observed(identity)
                .then(|| command.observation_packet())
                .flatten()
        };
        let mut first_failure = None;
        let mut budget = None;
        let slots = commands
            .len()
            .checked_mul(std::mem::size_of::<
                Option<ferrum_interfaces::vnext::DeviceObservationPacket>,
            >())
            .ok_or_else(|| failure("segment.slots_bound"))?;
        let mut retained = std::mem::size_of::<Self>()
            .checked_add(slots)
            .ok_or_else(|| failure("segment.retained_bound"))?;
        let mut projected = commands
            .len()
            .checked_mul(std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>())
            .ok_or_else(|| failure("segment.projected_bound"))?;
        for (ordinal, command) in commands.iter().enumerate() {
            let mut reason = command.observation_diagnostic();
            if !command.cublas_cost_requirement().matches_observed(identity) {
                reason.get_or_insert(failure("segment.library_identity"));
            }
            if let Some(mut reason) = reason {
                reason.logical_command_ordinal = Some(ordinal as u32);
                first_failure.get_or_insert(reason);
            }
            if let Some(packet) = eligible(command) {
                let current = packet
                    .budget()
                    .ok_or_else(|| failure("segment.packet_budget_absent"))?;
                match &budget {
                    Some(expected) if !Arc::ptr_eq(expected, current) => {
                        return Err(first_failure.unwrap_or_else(|| failure("segment.mixed_budget")))
                    }
                    None => budget = Some(Arc::clone(current)),
                    _ => {}
                }
                // The checked private lease charges the child template to this
                // same budget. Keeping the complete packet below keeps that
                // lease alive; charge only its handle/input here, including
                // dynamic ranges, rather than charging the template again.
                retained = retained
                    .checked_add(packet.call_owned_payload_bytes())
                    .ok_or_else(|| failure("segment.retained_overflow"))?;
                projected = projected
                    .checked_add(packet.projection_retained_bytes_upper_bound())
                    .ok_or_else(|| failure("segment.projected_overflow"))?;
            } else if first_failure.is_none() {
                let mut reason = failure("segment.command_packet_absent");
                reason.logical_command_ordinal = Some(ordinal as u32);
                first_failure = Some(reason);
            }
        }
        let budget = budget
            .ok_or_else(|| first_failure.unwrap_or_else(|| failure("segment.budget_absent")))?;
        let upper = retained
            .checked_add(slots)
            .ok_or_else(|| failure("segment.reserve_bound"))?;
        let reservation = budget
            .reserve_with_diagnostic(upper, "segment.reserve")
            .map_err(|error| first_failure.unwrap_or(error))?;
        let recipes: Box<[_]> = commands.iter().map(eligible).collect();
        reservation
            .retain(Arc::new(Self {
                recipes,
                retained,
                projected,
                first_failure,
            }))
            .map_err(|error| {
                first_failure.unwrap_or_else(|| {
                    DeviceObservationDiagnostic::new(
                        DeviceObservationFailureStage::Retain,
                        "segment.retain",
                        Some(error),
                    )
                })
            })
    }
}

impl DeviceObservationTemplate for SegmentObservation {
    fn command_count(&self) -> usize {
        self.recipes.len()
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
        if diagnostic.is_none() {
            *diagnostic = self.first_failure;
        }
        self.recipes
            .iter()
            .enumerate()
            .map(|(ordinal, recipe)| {
                let Some(recipe) = recipe else {
                    return Ok(None);
                };
                let mut local = None;
                let result = recipe.template().project_with_diagnostic(input, &mut local);
                if let Some(mut failure) = local {
                    failure.logical_command_ordinal = Some(ordinal as u32);
                    diagnostic.get_or_insert(failure);
                }
                let mut values = result?;
                if values.len() != 1 {
                    return Err(StatisticalEvidenceUnknown::CommandMismatch);
                }
                Ok(values.pop().flatten())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::DeviceObservationTemplateBudget;

    struct CpuMetadata(Box<[u8]>);
    impl DeviceObservationTemplate for CpuMetadata {
        fn command_count(&self) -> usize {
            1
        }
        fn retained_payload_bytes(&self) -> Option<usize> {
            std::mem::size_of::<Self>().checked_add(self.0.len())
        }
        fn projection_retained_bytes_upper_bound(&self) -> Option<usize> {
            Some(std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>())
        }
        fn project(
            &self,
            _: &FrozenObservationInput,
        ) -> Result<Vec<Option<SelectedCommandCostEvidenceV1>>, StatisticalEvidenceUnknown>
        {
            Ok(vec![None])
        }
    }
    fn commands(budget: &Arc<DeviceObservationTemplateBudget>) -> Vec<CudaDeviceCommand> {
        let child = budget
            .retain(Arc::new(CpuMetadata(vec![0; 16 * 1024].into_boxed_slice())))
            .unwrap();
        let packet = child.packet(FrozenObservationInput::command(1)).unwrap();
        (0..2)
            .map(|_| {
                super::super::super::vnext_runtime::tests::command("actual.cuda.metadata")
                    .with_observation(packet.clone())
            })
            .collect()
    }

    #[test]
    fn cuda_segment_observation_charges_child_once_and_keeps_queued_lease() {
        let budget = DeviceObservationTemplateBudget::new(128 * 1024).unwrap();
        let original = commands(&budget);
        let child_bytes = budget.retained_payload_bytes();
        let child_projected = original[0]
            .observation_packet()
            .unwrap()
            .projection_retained_bytes_upper_bound();
        let resident = SegmentObservation::new(&original, None).unwrap();
        let total = budget.retained_payload_bytes();
        assert!(
            total - child_bytes < child_bytes,
            "parent handles must not reserve another copy of the leased child payload"
        );
        let queued = resident.packet(FrozenObservationInput::command(1)).unwrap();
        assert_eq!(
            queued.projection_retained_bytes_upper_bound(),
            2 * child_projected + 2 * std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>()
        );
        let copy = queued.clone();
        drop(original);
        drop(resident);
        drop(queued);
        assert_eq!(budget.retained_payload_bytes(), total);
        assert_eq!(copy.template().project(copy.input()).unwrap().len(), 2);
        drop(copy);
        assert_eq!(budget.retained_payload_bytes(), 0);

        for (limit, expected) in [(total, true), (total - 1, false)] {
            let budget = DeviceObservationTemplateBudget::new(limit).unwrap();
            let original = commands(&budget);
            let before = budget.retained_payload_bytes();
            let result = SegmentObservation::new(&original, None);
            assert_eq!(result.is_ok(), expected);
            if let Err(failure) = &result {
                assert_eq!(failure.stage, DeviceObservationFailureStage::Reserve);
                assert_eq!(failure.error, Some(StatisticalEvidenceUnknown::Capacity));
                let snapshot = failure.budget.unwrap();
                assert_eq!(snapshot.current, before);
                assert_eq!(snapshot.current + snapshot.required.unwrap(), total);
            }
            drop(result);
            assert_eq!(budget.retained_payload_bytes(), before);
            drop(original);
            assert_eq!(budget.retained_payload_bytes(), 0);
        }

        let other = DeviceObservationTemplateBudget::new(128 * 1024).unwrap();
        let mut mixed = commands(&budget);
        mixed.push(commands(&other).pop().unwrap());
        let before = (
            budget.retained_payload_bytes(),
            other.retained_payload_bytes(),
        );
        assert_eq!(
            SegmentObservation::new(&mixed, None).unwrap_err().site,
            "segment.mixed_budget"
        );
        assert_eq!(
            (
                budget.retained_payload_bytes(),
                other.retained_payload_bytes()
            ),
            before
        );
        drop(mixed);
        assert_eq!(
            (
                budget.retained_payload_bytes(),
                other.retained_payload_bytes()
            ),
            (0, 0)
        );
    }
}
