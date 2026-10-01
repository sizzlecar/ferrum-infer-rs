//! CPU-only recipes sealed beside the actual successfully captured commands.
//! This object owns no CUDA context, graph, library handle, buffer or lease.
use super::*;
use ferrum_interfaces::execution_cost::{
    SelectedCommandCostEvidenceV1, StatisticalEvidenceUnknown,
};
use ferrum_interfaces::vnext::{DeviceObservationTemplate, FrozenObservationInput};

pub(super) struct SegmentObservation {
    recipes: Box<[Option<ferrum_interfaces::vnext::DeviceObservationPacket>]>,
    retained: usize,
    projected: usize,
}
impl SegmentObservation {
    pub(super) fn new(
        commands: &[CudaDeviceCommand],
        identity: Option<super::super::vnext_ops::CublasHandleApiIdentity>,
    ) -> Option<ferrum_interfaces::vnext::RetainedDeviceObservationTemplate> {
        if commands.is_empty() || commands.len() > MAX_COST_COMMANDS {
            return None;
        }
        let eligible = |command: &CudaDeviceCommand| {
            command
                .cublas_cost_requirement()
                .matches_observed(identity)
                .then(|| command.observation_packet())
                .flatten()
        };
        let mut budget = None;
        let slots = commands.len().checked_mul(std::mem::size_of::<
            Option<ferrum_interfaces::vnext::DeviceObservationPacket>,
        >())?;
        let mut retained = std::mem::size_of::<Self>().checked_add(slots)?;
        let mut projected = commands
            .len()
            .checked_mul(std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>())?;
        // Only borrow/clone small handles while calculating a bound. Reserve
        // before allocating the owned slot catalogue.
        for command in commands {
            if let Some(packet) = eligible(command) {
                let current = packet.budget()?;
                match &budget {
                    Some(expected) if !Arc::ptr_eq(expected, current) => return None,
                    None => budget = Some(Arc::clone(current)),
                    _ => {}
                }
                retained = retained.checked_add(packet.retained_payload_bytes())?;
                projected =
                    projected.checked_add(packet.projection_retained_bytes_upper_bound())?;
            }
        }
        let budget = budget?;
        let reservation = budget.reserve(retained.checked_add(slots)?).ok()?;
        let recipes: Box<[_]> = commands.iter().map(eligible).collect();
        reservation
            .retain(Arc::new(Self {
                recipes,
                retained,
                projected,
            }))
            .ok()
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
        self.recipes
            .iter()
            .map(|recipe| {
                let Some(recipe) = recipe else {
                    return Ok(None);
                };
                let mut values = recipe.template().project(input)?;
                if values.len() != 1 {
                    return Err(StatisticalEvidenceUnknown::CommandMismatch);
                }
                Ok(values.pop().flatten())
            })
            .collect()
    }
}
