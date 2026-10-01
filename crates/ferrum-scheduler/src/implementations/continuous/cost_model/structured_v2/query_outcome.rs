//! Query availability is distinct from invalid original evidence. These
//! process-local results carry no new source, profile or parameter identity.
use super::StructuredUnknownV2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredQueryFailureV2 {
    /// A checked input needs an unobserved direction or physical branch.
    OutsideSupport(StructuredUnknownV2),
    /// A checked input has a finite prediction outside the planning range.
    /// Arithmetic overflow and inconsistent evidence remain Invalid.
    OutsidePredictionRange(StructuredUnknownV2),
    Invalid(StructuredUnknownV2),
}

impl StructuredQueryFailureV2 {
    /// Preserve the original public Unknown and replay/qualification behavior.
    pub const fn reason(self) -> StructuredUnknownV2 {
        match self {
            Self::OutsideSupport(reason)
            | Self::OutsidePredictionRange(reason)
            | Self::Invalid(reason) => reason,
        }
    }
}

impl From<StructuredUnknownV2> for StructuredQueryFailureV2 {
    fn from(reason: StructuredUnknownV2) -> Self {
        Self::Invalid(reason)
    }
}

pub(super) type QueryResult<T> = std::result::Result<T, StructuredQueryFailureV2>;

pub(super) fn planning_sum(
    upper: u64,
    residual: u64,
    static_margin: u64,
    learned_margin: u64,
    maximum_wave_ns: u64,
) -> QueryResult<u64> {
    let value = upper
        .checked_add(residual)
        .and_then(|v| v.checked_add(static_margin))
        .and_then(|v| v.checked_add(learned_margin))
        .ok_or(StructuredUnknownV2::Numerical)?;
    if value > maximum_wave_ns {
        return Err(StructuredQueryFailureV2::OutsidePredictionRange(
            StructuredUnknownV2::Numerical,
        ));
    }
    Ok(value)
}

impl super::StructuredQueryV2 {
    /// Check the complete query before any coverage exclusion. Constructors
    /// already check original physical evidence; this preserves the numerical
    /// and forecast invariants when an imported model evaluates the query.
    pub(super) fn validate_for_prediction(
        &self,
        settings: &super::StructuredSettingsV2,
    ) -> super::Result<()> {
        use super::{HostPendingConstraintV2, StructuredUnknownV2 as U};
        self.input.validate(settings)?;
        if let Some(pending) = &self.pending {
            if pending.eligible.iter().any(|p| *p >= self.input.owner.rows)
                || pending.eligible.windows(2).any(|p| p[0] >= p[1])
                || (pending.eligible.is_empty()
                    && pending.constraint == HostPendingConstraintV2::NonEmptySubset)
            {
                return Err(U::InvalidInput);
            }
        }
        self.pending_count_range()?;
        if let Some(upper) = self.repetition_upper_sum {
            let (basis, support) = self.input.repetition_offsets.ok_or(U::WrongDomain)?;
            let lower = *self.input.support.get(support).ok_or(U::InvalidInput)?;
            if upper < lower
                || upper > (1u64 << 53)
                || self.input.basis.get(basis) != Some(&(lower as f64))
            {
                return Err(U::InvalidInput);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detailed_query_range_cap_is_distinct_from_arithmetic_failure() {
        use StructuredQueryFailureV2::{Invalid, OutsidePredictionRange};
        use StructuredUnknownV2::Numerical;
        assert_eq!(planning_sum(7, 1, 1, 1, 10), Ok(10));
        assert_eq!(
            planning_sum(7, 1, 1, 2, 10),
            Err(OutsidePredictionRange(Numerical))
        );
        assert_eq!(planning_sum(u64::MAX - 3, 1, 1, 1, u64::MAX), Ok(u64::MAX));
        assert_eq!(
            planning_sum(u64::MAX - 3, 1, 1, 2, u64::MAX),
            Err(Invalid(Numerical))
        );
        assert_eq!(
            StructuredQueryFailureV2::from(Numerical),
            Invalid(Numerical)
        );
        for failure in [OutsidePredictionRange(Numerical), Invalid(Numerical)] {
            assert_eq!(failure.reason(), Numerical);
        }
    }
}
