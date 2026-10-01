//! Measurable owned numerical payload for both collecting and published states.
//! Phase workspace reservations are separate: these methods do not estimate
//! temporary fitting allocations or allocator bookkeeping/process RSS.
use super::*;

impl FittedStructuredModelV2 {
    fn retained_heap_bytes(&self) -> Option<usize> {
        self.scope
            .retained_heap_bytes()?
            .checked_add(
                self.exemplar
                    .retained_numeric_bytes()?
                    .checked_sub(std::mem::size_of::<StructuredInputV2>())?,
            )?
            .checked_add(self.numerical.retained_heap_bytes()?)?
            .checked_add(self.state.retained_heap_bytes()?)?
            .checked_add(
                self.service_window
                    .as_ref()
                    .and_then(|p| p.contract.nonnegative_envelope.as_ref())
                    .and_then(|c| c.algorithm_universe.as_ref())
                    .map_or(Some(0), DeclaredAlgorithmUniverseV1::retained_payload_bytes)?,
            )?
            .checked_add(
                self.owner_blocks
                    .as_ref()
                    .and_then(|p| p.contract.nonnegative_envelope.as_ref())
                    .and_then(|c| c.algorithm_universe.as_ref())
                    .map_or(Some(0), DeclaredAlgorithmUniverseV1::retained_payload_bytes)?,
            )?
            .checked_add(self.owner_blocks.as_ref().map_or(Some(0), |p| {
                p.contract
                    .input_target
                    .as_ref()
                    .map_or(0, OwnerInputTargetV1::retained_heap_bytes)
                    .checked_add(
                        p.input_target
                            .as_ref()
                            .map_or(0, OwnerInputTargetV1::retained_heap_bytes),
                    )
            })?)
        // Remaining source, completion coverage and service contracts are
        // inline values. Their inline storage is included by size_of below.
    }

    pub fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>().checked_add(self.retained_heap_bytes()?)
    }
}

impl CalibratedStructuredModelV2 {
    fn retained_heap_bytes(&self) -> Option<usize> {
        self.fitted
            .retained_heap_bytes()?
            .checked_add(self.joint_bank.as_ref().map_or(Some(0), |b| b.retained())?)?
            .checked_add(
                self.residual_support
                    .as_ref()
                    .map_or(Some(0), JointSupport::retained_heap_bytes)?,
            )
    }

    pub fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>().checked_add(self.retained_heap_bytes()?)
    }
}

impl QualifiedStructuredModelV2 {
    /// Owned model payload includes all retained numerical capacities and
    /// frozen call IDs. Encoded source bytes are not used as a RAM estimate.
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(self.calibrated.retained_heap_bytes()?)?
            .checked_add(
                self.qualification_support
                    .as_ref()
                    .map_or(Some(0), JointSupport::retained_heap_bytes)?,
            )
    }
}
