//! Read-only requirements of one already-bound query. This is neither an
//! observation nor a qualification, and cannot be deserialized into either.
use super::*;

#[derive(Debug, Clone, Serialize)]
pub struct StructuredQueryDemandV2 {
    pub owner: StructuredOwnerKeyV2,
    pub domain_signature: [u8; 32],
    pub regression_axes: Vec<f64>,
    pub joint_support_coordinates: Vec<u64>,
    pub fixed_pending_positions: Vec<u32>,
    pub eligible_pending_positions: Vec<u32>,
    pub pending_constraint: Option<HostPendingConstraintV2>,
    pub length_positions: Vec<u32>,
    pub reachable_joint_counts: Vec<(u32, u32)>,
}

impl StructuredQueryV2 {
    /// Numerical requirements only. In particular, support coordinates retain
    /// their joint identity: coordinate extrema are not an authorized envelope.
    pub fn required_coverage(&self) -> Result<StructuredQueryDemandV2> {
        let (minimum, maximum) = self.pending_count_range()?;
        let length = u32::try_from(self.input.length_positions.len())
            .map_err(|_| StructuredUnknown::Capacity)?;
        let eligible = self
            .pending
            .as_ref()
            .map(|p| p.eligible.as_slice())
            .unwrap_or(&[]);
        Ok(StructuredQueryDemandV2 {
            owner: self.input.owner.clone(),
            domain_signature: self.input.domain,
            regression_axes: self.input.basis.clone(),
            joint_support_coordinates: self.input.support.clone(),
            fixed_pending_positions: self
                .input
                .pending_positions
                .iter()
                .copied()
                .filter(|p| eligible.binary_search(p).is_err())
                .collect(),
            eligible_pending_positions: eligible.to_vec(),
            pending_constraint: self.pending.as_ref().map(|p| p.constraint),
            length_positions: self.input.length_positions.clone(),
            reachable_joint_counts: (minimum..=maximum).map(|n| (n as u32, length)).collect(),
        })
    }

    // The authorization gate and diagnostics share this finite reachable set.
    // No Cartesian product over unrelated Length or pending counts is implied.
    pub(super) fn pending_count_range(&self) -> Result<(usize, usize)> {
        let rows = self.input.owner.rows as usize;
        if rows == 0 || rows > 128 || self.input.pending_positions.len() > rows {
            return Err(StructuredUnknown::InvalidInput);
        }
        let range = if let Some(pending) = &self.pending {
            let fixed = self
                .input
                .pending_positions
                .iter()
                .filter(|p| pending.eligible.binary_search(p).is_err())
                .count();
            (
                fixed + usize::from(pending.constraint == HostPendingConstraintV2::NonEmptySubset),
                fixed + pending.eligible.len(),
            )
        } else {
            let count = self.input.pending_positions.len();
            (count, count)
        };
        if range.0 > range.1 || range.1 > rows {
            return Err(StructuredUnknown::InvalidInput);
        }
        Ok(range)
    }
}
