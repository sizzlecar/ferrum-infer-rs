use super::super::super::input::position_moments;
use super::*;
use ferrum_interfaces::execution_cost::{HostContentDomainV1, HostTerminalExpectationV1};

pub(in super::super) fn axes(input: &StructuredInputV2) -> Result<Vec<u64>> {
    input
        .basis
        .iter()
        .map(|&x| {
            if !x.is_finite() || x < 0. || x > (1u64 << 53) as f64 || x.fract() != 0. {
                Err(StructuredUnknown::InvalidInput)
            } else {
                Ok(x as u64)
            }
        })
        .collect()
}

/// Shared pre-execution branch facts. Completion outcomes never enter here.
pub(in super::super) fn input_branches(input: &StructuredInputV2) -> [Option<bool>; 4] {
    [
        Some(!input.pending_positions.is_empty()),
        Some(!input.length_positions.is_empty()),
        Some(
            input
                .physical_host_rows
                .iter()
                .any(|row| row.mask_upload_required),
        ),
        input
            .repetition_offsets
            .map(|(basis, _)| input.basis[basis] != 0.),
    ]
}

pub(in super::super) fn early_capable(row: &StructuredHostRowV1) -> bool {
    matches!(row.installed_policy.empirical_content_domain,
        Some(HostContentDomainV1::PlainTextInstalledV2(policy)) if policy.model_eos || policy.user_stop)
        && row.terminal_expectation == HostTerminalExpectationV1::TokenMayTerminate
}

fn completion_upper(
    input: &StructuredInputV2,
    out: &mut [u64],
    before_settlement: bool,
) -> Result<()> {
    let Some(completion) = &input.completion else {
        return Ok(());
    };
    if completion.settled && !before_settlement {
        return Ok(());
    }
    let mut moments = [0u64; 3];
    for row in &input.physical_host_rows {
        if super::super::super::completion::installed(row)
            && (row.terminal_expectation == HostTerminalExpectationV1::LengthBoundary
                || early_capable(row))
        {
            for (sum, value) in moments
                .iter_mut()
                .zip(position_moments(&[row.physical_position])?)
            {
                *sum = sum.checked_add(value).ok_or(StructuredUnknown::Numerical)?;
            }
        }
    }
    out[completion.basis_offset..completion.basis_offset + 3].copy_from_slice(&moments);
    Ok(())
}

pub(in super::super) fn membership_axes(input: &StructuredInputV2) -> Result<Vec<u64>> {
    let mut out = axes(input)?;
    completion_upper(input, &mut out, true)?;
    Ok(out)
}

/// Shared Fit/Residual/Qualification/feedback coordinates for the empirical
/// global strategy. Completion opportunities are known before execution; the
/// three settled-outcome columns must never disclose EOS/Stop to this model.
/// This is an empirical regressor, not a bound on unseen terminal-tail latency.
pub(in super::super) fn model_axes(
    input: &StructuredInputV2,
    estimator: NonNegativePlanningEstimatorV1,
) -> Result<Vec<u64>> {
    if estimator.prospective_completion_work() {
        membership_axes(input)
    } else {
        axes(input)
    }
}

/// Componentwise work bounds form one common feasible-work superset. All
/// coefficients are nonnegative, so simultaneous expansion is conservative
/// even when these host branches cannot attain their maxima together.
pub(in super::super) fn query_upper(
    query: &StructuredQueryV2,
    domain: &CostWorkloadDomainV1,
) -> Result<Vec<u64>> {
    let input = &query.input;
    let mut out = axes(input)?;
    if let Some(pending) = &query.pending {
        position_moments(&pending.eligible)?;
        if pending.eligible.iter().any(|p| *p >= input.owner.rows)
            || (pending.constraint == HostPendingConstraintV2::NonEmptySubset
                && pending.eligible.is_empty())
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        let mut moments = [0u64; 3];
        for p in 0..input.owner.rows {
            if input.pending_positions.binary_search(&p).is_ok()
                || pending.eligible.binary_search(&p).is_ok()
            {
                for (sum, value) in moments.iter_mut().zip(position_moments(&[p])?) {
                    *sum = sum.checked_add(value).ok_or(StructuredUnknown::Numerical)?;
                }
            }
        }
        out[input.pending_basis_offset..input.pending_basis_offset + 3].copy_from_slice(&moments);
    }
    completion_upper(input, &mut out, false)?;
    if let Some(upper) = query.repetition_upper_sum {
        let (basis, support) = input
            .repetition_offsets
            .ok_or(StructuredUnknown::WrongDomain)?;
        let maximum = domain
            .limits()
            .repetition_slot_capacity
            .min(domain.limits().output_vocabulary_elements.get())
            .checked_mul(u64::from(input.owner.rows))
            .ok_or(StructuredUnknown::Capacity)?;
        if upper < input.support[support] || upper > maximum || upper > (1u64 << 53) {
            return Err(StructuredUnknown::InvalidInput);
        }
        out[basis] = upper;
    }
    Ok(out)
}

/// Catalog dispatch uses the same forecast range but never an observed
/// completion outcome to remove reachable work from the requested population.
pub(in super::super) fn prospective_query_upper(
    query: &StructuredQueryV2,
    domain: &CostWorkloadDomainV1,
) -> Result<Vec<u64>> {
    let mut out = query_upper(query, domain)?;
    completion_upper(&query.input, &mut out, true)?;
    Ok(out)
}

/// Estimator-derived projection only. Cold diagnostics can inspect rejected
/// inputs without accidentally acquiring or requiring authorization.
pub(in super::super) fn model_query_upper(
    query: &StructuredQueryV2,
    domain: &CostWorkloadDomainV1,
    estimator: NonNegativePlanningEstimatorV1,
    before_settlement: bool,
) -> Result<Vec<u64>> {
    if before_settlement || estimator.prospective_completion_work() {
        prospective_query_upper(query, domain)
    } else {
        query_upper(query, domain)
    }
}
