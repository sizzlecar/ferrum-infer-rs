//! Installed plain-text completion is observed only after the original host
//! settlement. Unknown future completion has a signed empirical envelope;
//! an early EOS/stop is never credited as a guaranteed reduction in work.
use super::input::position_moments;
use super::*;
use ferrum_interfaces::execution_cost::{HostContentDomainV1, HostTerminalExpectationV1};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct InputCompletion {
    pub positions: Vec<u32>,
    pub basis_offset: usize,
    pub support_offset: usize,
    pub settled: bool,
}

pub(super) fn installed(row: &StructuredHostRowV1) -> bool {
    matches!(
        row.installed_policy.empirical_content_domain,
        Some(HostContentDomainV1::PlainTextInstalledV2(_))
    )
}

fn early(row: &StructuredHostRowV1) -> bool {
    matches!(row.installed_policy.empirical_content_domain,
        Some(HostContentDomainV1::PlainTextInstalledV2(p)) if p.model_eos || p.user_stop)
        && row.terminal_expectation == HostTerminalExpectationV1::TokenMayTerminate
}

impl StructuredInputV2 {
    /// Pure numerical adapter. Live callers must first validate the original
    /// private settlement; source replay must validate the bound raw receipts.
    /// Positions are in physical row order, not host processing order.
    pub fn with_settled_completion(mut self, terminal_positions: &[u32]) -> Result<Self> {
        position_moments(terminal_positions)?;
        if terminal_positions.iter().any(|p| *p >= self.owner.rows) {
            return Err(StructuredUnknown::InvalidInput);
        }
        let Some(completion) = &mut self.completion else {
            return Ok(self);
        };
        let mut positions = Vec::new();
        for row in &self.physical_host_rows {
            if !installed(row) {
                continue;
            }
            let terminal = terminal_positions
                .binary_search(&row.physical_position)
                .is_ok();
            match row.terminal_expectation {
                HostTerminalExpectationV1::NoTokenProduced if terminal => {
                    return Err(StructuredUnknown::InvalidInput)
                }
                HostTerminalExpectationV1::LengthBoundary if !terminal => {
                    return Err(StructuredUnknown::MissingEvidence)
                }
                HostTerminalExpectationV1::TokenMayTerminate if terminal && !early(row) => {
                    return Err(StructuredUnknown::InvalidInput)
                }
                _ => {}
            }
            if terminal {
                positions.push(row.physical_position);
            }
        }
        let moments = position_moments(&positions)?;
        self.basis[completion.basis_offset..completion.basis_offset + 3]
            .copy_from_slice(&moments.map(|v| v as f64));
        self.support[completion.support_offset..completion.support_offset + 3]
            .copy_from_slice(&moments);
        completion.positions = positions;
        completion.settled = true;
        Ok(self)
    }

    pub(super) fn validate_actual_completion(&self) -> Result<()> {
        if self.completion.as_ref().is_some_and(|c| !c.settled) {
            return Err(StructuredUnknown::MissingEvidence);
        }
        Ok(())
    }

    pub(super) fn validate_completion(&self) -> Result<()> {
        let has_installed = self.physical_host_rows.iter().any(installed);
        let Some(c) = &self.completion else {
            return if has_installed {
                Err(StructuredUnknown::MissingEvidence)
            } else {
                Ok(())
            };
        };
        if !has_installed
            || c.basis_offset.checked_add(6) != Some(self.pending_basis_offset)
            || c.support_offset.checked_add(6) != Some(self.pending_support_offset)
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        let moments = position_moments(&c.positions)?;
        if self.basis[c.basis_offset..c.basis_offset + 3] != moments.map(|v| v as f64)
            || self.support[c.support_offset..c.support_offset + 3] != moments
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        for row in &self.physical_host_rows {
            let marked = c.positions.binary_search(&row.physical_position).is_ok();
            let forced = installed(row)
                && row.terminal_expectation == HostTerminalExpectationV1::LengthBoundary;
            if (forced && !marked) || (marked && !(forced || (c.settled && early(row)))) {
                return Err(StructuredUnknown::InvalidInput);
            }
        }
        if c.positions.iter().any(|p| *p >= self.owner.rows) {
            return Err(StructuredUnknown::InvalidInput);
        }
        Ok(())
    }
}

/// Per-physical-row actual early completion and continuation. A terminal at
/// capacity cannot stand in for a below-capacity EOS/stop challenge.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct CompletionCoverage {
    continued: u128,
    early_completed: u128,
}
impl CompletionCoverage {
    pub fn observed(samples: &[StructuredNumericObservationV2]) -> Result<Self> {
        let mut coverage = Self::default();
        for sample in samples {
            sample.input.validate_actual_completion()?;
            let Some(c) = &sample.input.completion else {
                continue;
            };
            for row in &sample.input.physical_host_rows {
                if early(row) {
                    let bit = 1u128 << row.physical_position;
                    if c.positions.binary_search(&row.physical_position).is_ok() {
                        coverage.early_completed |= bit;
                    } else {
                        coverage.continued |= bit;
                    }
                }
            }
        }
        Ok(coverage)
    }
    pub fn intersect(self, other: Self) -> Self {
        Self {
            continued: self.continued & other.continued,
            early_completed: self.early_completed & other.early_completed,
        }
    }
    pub fn authorize(self, input: &StructuredInputV2) -> Result<()> {
        self.authorize_detailed(input)
            .map_err(StructuredQueryFailureV2::reason)
    }
    pub fn authorize_detailed(self, input: &StructuredInputV2) -> QueryResult<()> {
        if input.completion.as_ref().is_some_and(|c| !c.settled) {
            let both = self.continued & self.early_completed;
            if input
                .physical_host_rows
                .iter()
                .any(|row| early(row) && both & (1u128 << row.physical_position) == 0)
            {
                return Err(StructuredQueryFailureV2::OutsideSupport(
                    StructuredUnknown::QualificationCoverage,
                ));
            }
        }
        Ok(())
    }
    pub fn bind(self, digest: &mut sha2::Sha256) {
        use sha2::Digest;
        digest.update(self.continued.to_le_bytes());
        digest.update(self.early_completed.to_le_bytes());
    }
}

pub(super) struct CompletionGenerators {
    values: Vec<f64>,
    identified: u128,
}
impl CompletionGenerators {
    pub fn new(fit: &super::fit::RowSpaceFit, input: &StructuredInputV2) -> Result<Option<Self>> {
        let Some(c) = &input.completion else {
            return Ok(None);
        };
        let mut values = Vec::with_capacity(input.owner.rows as usize);
        let mut identified = 0;
        let mut direction = vec![0.; input.basis.len()];
        for p in 0..input.owner.rows {
            direction[c.basis_offset..c.basis_offset + 3]
                .copy_from_slice(&position_moments(&[p])?.map(|v| v as f64));
            values.push(fit.linear_value(&direction)?);
            match fit.identify(&direction) {
                Ok(()) => identified |= 1u128 << p,
                Err(StructuredUnknown::UnidentifiedDirection) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(Some(Self { values, identified }))
    }
    pub fn retained_heap_bytes(&self) -> Option<usize> {
        self.values
            .capacity()
            .checked_mul(std::mem::size_of::<f64>())
    }
    pub fn extend(
        &self,
        input: &StructuredInputV2,
        mut bounds: super::envelope::EnvelopeBounds,
    ) -> QueryResult<super::envelope::EnvelopeBounds> {
        let c = input
            .completion
            .as_ref()
            .ok_or(StructuredUnknown::WrongDomain)?;
        if c.settled {
            return Ok(bounds);
        }
        let mut low_delta = 0f64;
        let mut high_delta = 0f64;
        let mut optional = Vec::new();
        for row in &input.physical_host_rows {
            if early(row) {
                let p = row.physical_position;
                if self.identified & (1u128 << p) == 0 {
                    return Err(StructuredQueryFailureV2::OutsideSupport(
                        StructuredUnknown::UnidentifiedDirection,
                    ));
                }
                let value = *self
                    .values
                    .get(p as usize)
                    .ok_or(StructuredUnknown::InvalidInput)?;
                low_delta += value.min(0.);
                high_delta += value.max(0.);
                optional.push(p);
            }
        }
        let lower = bounds.lower_ns as f64 + low_delta;
        let upper = bounds.upper_ns as f64 + high_delta;
        if !lower.is_finite() || !upper.is_finite() || upper < lower {
            return Err(StructuredUnknown::Numerical.into());
        }
        if lower <= 0. || upper > (1u64 << 53) as f64 {
            return Err(StructuredQueryFailureV2::OutsidePredictionRange(
                StructuredUnknown::Numerical,
            ));
        }
        bounds.lower_ns = lower.ceil() as u64;
        bounds.upper_ns = upper.ceil() as u64;
        for (i, extra) in position_moments(&optional)?.into_iter().enumerate() {
            bounds.support_upper[c.support_offset + i] = bounds.support_upper[c.support_offset + i]
                .checked_add(extra)
                .ok_or(StructuredUnknown::InvalidInput)?;
        }
        Ok(bounds)
    }
}
