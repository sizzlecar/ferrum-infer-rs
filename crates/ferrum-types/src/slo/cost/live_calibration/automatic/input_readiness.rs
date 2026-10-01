//! Explicit collection work bounds, never numerical qualification authority.
use super::*;

#[cfg(test)]
mod tests;

const MAX_GEOMETRY_VISITS: u64 = 128_000_000;
const MAX_PHASE_MEMBERS: usize = 4096;

/// A complete original offer block remains the only phase transition boundary.
/// Input geometry may postpone that boundary, but observed costs, residuals and
/// qualification results never choose when to stop collecting a phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SloAutomaticCalibrationInputReadinessV1 {
    /// Historical source7 count-only stopping. This preserves its wire and
    /// complete-last-block member allowance for explicit compatibility use.
    CountOnlyV1 {},
    /// Requires checked physical input evidence. Missing directions or branches
    /// remain Unknown when the original block, age or resource limits expire.
    WorkAxesAndBranchesV1 {
        /// Per-phase total complete blocks, including all offers and exclusions.
        /// block_offered * maximum_phase_blocks must fit 4096 members; these
        /// limits are not reset after unsuccessful input-readiness checks.
        maximum_phase_blocks: [NonZeroUsize; 3],
        /// Cumulative geometry work visits per phase, across every readiness
        /// scan. This is independent from, and does not enlarge, retained bytes.
        maximum_geometry_visits: NonZeroU64,
    },
    /// Uses bounded residual caching for candidate order, with selected pivots
    /// and the final input span reverified from original rows. The numerical
    /// Fit/Residual/Qualification gates and total geometry allowance are unchanged.
    WorkAxesAndBranchesV2 {
        maximum_phase_blocks: [NonZeroUsize; 3],
        maximum_geometry_visits: NonZeroU64,
    },
    /// Scans every original coordinate, then runs the V2 geometry kernel on
    /// exactly nonzero columns. Near-zero and collinear columns are retained.
    /// Original rows and the numerical model keep their complete axis roster.
    WorkAxesAndBranchesV3 {
        maximum_phase_blocks: [NonZeroUsize; 3],
        maximum_geometry_visits: NonZeroU64,
    },
}

impl Default for SloAutomaticCalibrationInputReadinessV1 {
    fn default() -> Self {
        Self::WorkAxesAndBranchesV3 {
            maximum_phase_blocks: [NonZeroUsize::new(16).unwrap(); 3],
            maximum_geometry_visits: NonZeroU64::new(32_000_000).unwrap(),
        }
    }
}

impl SloAutomaticCalibrationInputReadinessV1 {
    pub(super) fn validate(&self) -> Result<(), String> {
        if let Self::WorkAxesAndBranchesV1 {
            maximum_phase_blocks,
            maximum_geometry_visits,
        }
        | Self::WorkAxesAndBranchesV2 {
            maximum_phase_blocks,
            maximum_geometry_visits,
        }
        | Self::WorkAxesAndBranchesV3 {
            maximum_phase_blocks,
            maximum_geometry_visits,
        } = self
        {
            if maximum_phase_blocks
                .iter()
                .any(|blocks| blocks.get() > MAX_PHASE_MEMBERS)
                || maximum_geometry_visits.get() > MAX_GEOMETRY_VISITS
            {
                return Err(
                    "automatic input readiness exceeds bounded block/geometry capacities".into(),
                );
            }
        }
        Ok(())
    }

    pub(super) fn validate_owner_blocks(
        &self,
        block: usize,
        phase_min_offered: [usize; 3],
    ) -> Result<(), String> {
        if let Self::WorkAxesAndBranchesV1 {
            maximum_phase_blocks,
            ..
        }
        | Self::WorkAxesAndBranchesV2 {
            maximum_phase_blocks,
            ..
        }
        | Self::WorkAxesAndBranchesV3 {
            maximum_phase_blocks,
            ..
        } = self
        {
            for (blocks, minimum) in maximum_phase_blocks.iter().zip(phase_min_offered) {
                block
                    .checked_mul(blocks.get())
                    .filter(|maximum| (minimum.max(8)..=MAX_PHASE_MEMBERS).contains(maximum))
                    .ok_or_else(|| "automatic input readiness must reserve complete phase blocks within 4096 members and the declared minimum offers".to_owned())?;
            }
        }
        Ok(())
    }
}
