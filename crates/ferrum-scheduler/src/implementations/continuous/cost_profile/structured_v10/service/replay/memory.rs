//! One collector-wide budget: open-phase workspace plus all surviving states.
//! Encoded source bytes have their own unchanged limit. Payload accounting
//! excludes allocator bookkeeping and RSS; backing Vec capacities are counted.
use super::*;

#[derive(Default)]
pub(super) struct RetainedCalls(Vec<u64>);
impl RetainedCalls {
    pub(super) fn insert(&mut self, call: u64) -> bool {
        if self.0.last().is_none_or(|last| *last < call) {
            self.0.push(call);
            return true;
        }
        match self.0.binary_search(&call) {
            Ok(_) => false,
            Err(index) => {
                self.0.insert(index, call);
                true
            }
        }
    }
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }
    fn retained_heap_bytes(&self) -> Option<usize> {
        self.0.capacity().checked_mul(std::mem::size_of::<u64>())
    }
}

impl ChildState {
    fn retained_heap_bytes(&self) -> Option<usize> {
        match self {
            Self::Empty => Some(0),
            Self::Failed(reason) => Some(reason.capacity()),
            Self::Fitted(model) => model
                .retained_payload_bytes()?
                .checked_sub(std::mem::size_of::<FittedStructuredModelV2>()),
            Self::Calibrated(model) => model
                .retained_payload_bytes()?
                .checked_sub(std::mem::size_of::<CalibratedStructuredModelV2>()),
            Self::Qualified(model) => model
                .retained_payload_bytes()?
                .checked_sub(std::mem::size_of::<QualifiedStructuredModelV2>()),
        }
    }
}

impl StructuredServiceCollectorV6 {
    /// Retained numerical/population payload and the unchanged conservative
    /// workspace reservation for every currently open-phase member. Completed
    /// phases contribute their surviving model, never freed sample allocations.
    pub fn retained_numeric_bytes(&self) -> Option<usize> {
        let mut bytes = std::mem::size_of::<Self>()
            .checked_add(self.phase_workspace_bytes)?
            .checked_add(self.calls.retained_heap_bytes()?)?
            .checked_add(self.frontiers.retained_heap_bytes()?)?;
        for (capacity, item) in [
            (self.states.capacity(), std::mem::size_of::<ChildState>()),
            (
                self.samples.capacity(),
                std::mem::size_of::<Vec<StructuredNumericObservationV2>>(),
            ),
            (self.member_counts.capacity(), std::mem::size_of::<usize>()),
            (
                self.phases.capacity(),
                std::mem::size_of::<Vec<StructuredPhaseProvenanceV10>>(),
            ),
            (
                self.child_ages.capacity(),
                std::mem::size_of::<(u64, u64)>(),
            ),
            (
                self.header.declaration.scopes.capacity(),
                std::mem::size_of::<StructuredScopeV2>(),
            ),
        ] {
            bytes = bytes.checked_add(capacity.checked_mul(item)?)?;
        }
        // The inner sample allocations and their transient numerical work are
        // covered by the original input*12 + observation*4 reservation.
        for state in &self.states {
            bytes = bytes.checked_add(state.retained_heap_bytes()?)?;
        }
        for scope in &self.header.declaration.scopes {
            bytes = bytes.checked_add(scope.retained_heap_bytes()?)?;
        }
        for phases in &self.phases {
            bytes = bytes.checked_add(
                phases
                    .capacity()
                    .checked_mul(std::mem::size_of::<StructuredPhaseProvenanceV10>())?,
            )?;
        }
        Some(bytes)
    }

    pub(super) fn check_retained_capacity(
        &self,
        additional_workspace: usize,
    ) -> Result<(), CostProfileError> {
        self.retained_numeric_bytes()
            .and_then(|bytes| bytes.checked_add(additional_workspace))
            .filter(|bytes| *bytes <= self.header.declaration.maximum_retained_numeric_bytes)
            .map(|_| ())
            .ok_or(CostProfileError::Limit("source6 numeric capacity"))
    }
}
