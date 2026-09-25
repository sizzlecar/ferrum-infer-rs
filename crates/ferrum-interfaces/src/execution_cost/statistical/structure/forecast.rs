//! Planning-only host uncertainty. These values neither change an actual
//! recipe nor construct a settlement, execution permit, or fitted cost bound.
use super::*;
use std::sync::Arc;

/// Constraint on the selected subset of eligible positions. NonEmptySubset
/// applies only when unknown pending text is the reason for FullLogits; a fixed
/// FullLogits peer already covers the empty subset and must use AnySubset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostPendingConstraintV2 {
    AnySubset,
    NonEmptySubset,
}

/// Exact means that host state is still the captured state. Conditional-empty
/// Greedy branches remain Unresolved even with no eligible positions: future
/// token contents have not become an actual observed execution.
#[derive(Debug, Clone)]
pub enum HostContentForecastV2 {
    Exact,
    Unresolved(HostPendingSetV2),
}

/// Bounded uncertainty attached to one immutable selected recipe. Fixed
/// positions, including their pending bits, come from that recipe. Callers may
/// not supply a replacement fixed bitmap or deserialize a live forecast.
#[derive(Debug, Clone)]
pub struct HostPendingSetV2 {
    recipe: Arc<UnsettledStructuredWaveEvidenceV1>,
    eligible_positions: Vec<u32>,
    constraint: HostPendingConstraintV2,
}

impl HostPendingSetV2 {
    /// The caller derives eligible positions from request progress, before any
    /// future token is generated. It must establish that every represented
    /// subset uses the same selected physical route; host uncertainty alone
    /// does not grant provider route or mask-residency equivalence.
    pub fn new(
        exact: &CanonicalWaveCostShape,
        recipe: &Arc<UnsettledStructuredWaveEvidenceV1>,
        eligible_positions: &[u32],
        constraint: HostPendingConstraintV2,
    ) -> Result<Self, StatisticalEvidenceUnknown> {
        use StatisticalEvidenceUnknown::{Capacity, MissingHostDomain};
        recipe.validate_exact(exact)?;
        let rows = recipe.physical_host_rows();
        if rows.is_empty() || eligible_positions.len() > rows.len() {
            return Err(Capacity);
        }
        if eligible_positions.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(MissingHostDomain);
        }
        for &position in eligible_positions {
            let row = rows.get(position as usize).ok_or(MissingHostDomain)?;
            if row.physical_position != position
                || row.role != HostRowRoleV2::Decode
                || row.no_generated_history
                || row.decode_requires_full_logits != Some(row.pending_decoded_utf8)
            {
                return Err(MissingHostDomain);
            }
        }
        let mut next_eligible = eligible_positions.iter().copied().peekable();
        let mut fixed_forces_full = exact.kind == ActualWaveKind::Prefill;
        let mut anchor_has_eligible_pending = false;
        for (position, row) in rows.iter().enumerate() {
            if row.physical_position as usize != position
                || row.installed_policy.empirical_content_domain
                    != Some(HostContentDomainV1::PlainTextGreedyV1)
            {
                return Err(MissingHostDomain);
            }
            if next_eligible.peek().copied() == Some(row.physical_position) {
                next_eligible.next();
                anchor_has_eligible_pending |= row.pending_decoded_utf8;
            } else {
                fixed_forces_full |= row.decode_requires_full_logits == Some(true)
                    || (row.role == HostRowRoleV2::Prefill && row.final_prefill);
            }
        }
        match recipe.device().product() {
            StructuredCostProductV1::GreedyToken => {
                if !eligible_positions.is_empty()
                    || constraint != HostPendingConstraintV2::AnySubset
                    || fixed_forces_full
                    || rows.iter().any(|row| row.pending_decoded_utf8)
                {
                    return Err(MissingHostDomain);
                }
            }
            StructuredCostProductV1::FullLogits if fixed_forces_full => {
                if constraint != HostPendingConstraintV2::AnySubset {
                    return Err(MissingHostDomain);
                }
            }
            StructuredCostProductV1::FullLogits => {
                if constraint != HostPendingConstraintV2::NonEmptySubset
                    || !anchor_has_eligible_pending
                {
                    return Err(MissingHostDomain);
                }
            }
        }
        let mut eligible = Vec::new();
        eligible
            .try_reserve_exact(eligible_positions.len())
            .map_err(|_| Capacity)?;
        eligible.extend_from_slice(eligible_positions);
        Ok(Self {
            recipe: Arc::clone(recipe),
            eligible_positions: eligible,
            constraint,
        })
    }

    pub fn eligible_positions(&self) -> &[u32] {
        &self.eligible_positions
    }

    pub fn constraint(&self) -> HostPendingConstraintV2 {
        self.constraint
    }

    /// Additional allocation owned by this set. The retained immutable recipe
    /// is shared with the selected evidence and is accounted by that evidence.
    pub fn position_storage_bytes(&self) -> usize {
        self.eligible_positions.capacity() * std::mem::size_of::<u32>()
    }
}

impl HostContentForecastV2 {
    /// An equal-looking replacement recipe does not inherit this forecast.
    /// Arc identity keeps this check bounded without serializing/hashing the
    /// entire algorithm table or copying its backing storage for each query.
    pub fn validate(
        &self,
        exact: &CanonicalWaveCostShape,
        recipe: &UnsettledStructuredWaveEvidenceV1,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if let Self::Unresolved(pending) = self {
            if !std::ptr::eq(pending.recipe.as_ref(), recipe) {
                return Err(StatisticalEvidenceUnknown::ExactBindingMismatch);
            }
        }
        recipe.validate_exact(exact)
    }
}
