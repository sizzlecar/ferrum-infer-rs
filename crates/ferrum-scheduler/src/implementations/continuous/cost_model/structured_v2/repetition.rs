//! A signed linear work interval, validated against the original future
//! recipe's lower anchor. No monotonic latency assumption is made.
use super::envelope::EnvelopeBounds;
use super::fit::RowSpaceFit;
use super::*;

pub(super) struct RepetitionGenerator {
    value: f64,
    identified: bool,
}
impl RepetitionGenerator {
    pub fn new(fit: &RowSpaceFit, input: &StructuredInputV2) -> Result<Option<Self>> {
        let Some((basis, _)) = input.repetition_offsets else {
            return Ok(None);
        };
        let mut direction = vec![0.; input.basis.len()];
        direction[basis] = 1.;
        let value = fit.linear_value(&direction)?;
        let identified = match fit.identify(&direction) {
            Ok(()) => true,
            Err(StructuredUnknown::UnidentifiedDirection) => false,
            Err(error) => return Err(error),
        };
        Ok(Some(Self { value, identified }))
    }
    pub fn extend(
        &self,
        input: &StructuredInputV2,
        upper: u64,
        mut bounds: EnvelopeBounds,
    ) -> QueryResult<EnvelopeBounds> {
        let (basis, support) = input
            .repetition_offsets
            .ok_or(StructuredUnknown::WrongDomain)?;
        let lower = input.support[support];
        if upper < lower || upper > (1u64 << 53) || input.basis[basis] != lower as f64 {
            return Err(StructuredUnknown::InvalidInput.into());
        }
        if upper == lower {
            return Ok(bounds);
        }
        if !self.identified {
            return Err(StructuredQueryFailureV2::OutsideSupport(
                StructuredUnknown::UnidentifiedDirection,
            ));
        }
        let delta = self.value * (upper - lower) as f64;
        let lo = bounds.lower_ns as f64 + delta.min(0.);
        let hi = bounds.upper_ns as f64 + delta.max(0.);
        if !lo.is_finite() || !hi.is_finite() || hi < lo {
            return Err(StructuredUnknown::Numerical.into());
        }
        if lo <= 0. || hi > (1u64 << 53) as f64 {
            return Err(StructuredQueryFailureV2::OutsidePredictionRange(
                StructuredUnknown::Numerical,
            ));
        }
        bounds.lower_ns = lo.ceil() as u64;
        bounds.upper_ns = hi.ceil() as u64;
        bounds.support_upper[support] = upper;
        Ok(bounds)
    }
}
