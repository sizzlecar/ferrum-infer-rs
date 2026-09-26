use super::*;

// Resource limits only: these do not establish statistical sample sufficiency.
const MAX_PAIRS: usize = 4096;
const MAX_RESAMPLES: usize = 1_000_000;
const MAX_PRIMARY_CELLS: usize = 64;
const MAX_STORED_VALUES: usize = 4_000_000; // 32 MB of f64 bootstrap results.
const MAX_ACCUMULATIONS: usize = 64_000_000;

impl FrozenStatisticalMethod {
    /// Freeze the implemented method and canonical configuration digest. The
    /// caller must still freeze the enclosing contract before measurements.
    pub fn paired_cluster_bootstrap(
        configuration: FrozenPairedBootstrap,
    ) -> Result<Self, ComparisonError> {
        validate_configuration(&configuration)?;
        Ok(Self {
            method_id: METHOD_ID.into(),
            analysis_unit: ANALYSIS_UNIT.into(),
            configuration_sha256: json_digest(&configuration)?,
            paired_bootstrap: Some(configuration),
        })
    }
}

fn validate_configuration(configuration: &FrozenPairedBootstrap) -> Result<(), ComparisonError> {
    let design = &configuration.declared_design;
    if configuration.schema_version != 1
        || configuration.resamples == 0
        || !configuration.family_alpha.is_finite()
        || configuration.family_alpha <= 0.0
        || configuration.family_alpha >= 1.0
        || !configuration.monte_carlo_error_budget.is_finite()
        || configuration.monte_carlo_error_budget <= 0.0
        || configuration.monte_carlo_error_budget >= configuration.family_alpha
        || configuration.maximum_relative_bound_width.len() != ComparisonMetric::ALL.len()
        || configuration
            .eligibility_plan_sha256
            .as_ref()
            .is_some_and(|hash| !valid_digest(hash))
        || ComparisonMetric::ALL.iter().any(|metric| {
            configuration
                .maximum_relative_bound_width
                .get(metric)
                .is_none_or(|width| !width.is_finite() || *width <= 0.0)
        })
        || !valid_digest(&design.pilot_source_sha256)
        || design.pilot_finished_unix_ns == 0
        || design.planned_pairs == 0
        || design.minimum_measured_requests_per_arm == 0
        || design.minimum_gap_bearing_requests_per_arm == 0
        || design.minimum_visible_gaps_per_arm == 0
        || design.independent_block_protocol.trim().is_empty()
    {
        return Err(ComparisonError(
            "invalid explicit bootstrap error/precision/design configuration".into(),
        ));
    }
    // Check even before a matrix is available, so enormous declarations do not
    // wait until after evidence traversal to fail a resource preflight.
    check_work(
        design.planned_pairs as usize,
        configuration.resamples as usize,
        1,
    )
}

pub(super) fn validate_method(
    contract: &FrozenComparisonContract,
    method: &FrozenStatisticalMethod,
) -> Result<(), ComparisonError> {
    let Some(configuration) = &method.paired_bootstrap else {
        return Ok(()); // External declarations remain unverified.
    };
    validate_configuration(configuration)?;
    if method.method_id != METHOD_ID || method.analysis_unit != ANALYSIS_UNIT {
        return Err(ComparisonError("bootstrap requires the implemented version and complete paired-repetition analysis unit".into()));
    }
    if !same_digest(&method.configuration_sha256, &json_digest(configuration)?) {
        return Err(ComparisonError(
            "bootstrap configuration digest mismatch".into(),
        ));
    }
    if configuration.declared_design.pilot_finished_unix_ns >= contract.frozen_unix_ns
        || configuration.declared_design.planned_pairs as usize != contract.pairs.len()
    {
        return Err(ComparisonError(
            "pilot must precede freeze and declared pair count must match the full frozen design"
                .into(),
        ));
    }
    check_work(
        contract.pairs.len(),
        configuration.resamples as usize,
        contract
            .cells
            .iter()
            .filter(|cell| cell.scope == CellScope::Primary)
            .count(),
    )
}

pub(super) fn check_work(
    pairs: usize,
    resamples: usize,
    cells: usize,
) -> Result<(), ComparisonError> {
    let values = cells
        .checked_mul(ComparisonMetric::ALL.len())
        .and_then(|metrics| metrics.checked_mul(resamples));
    let work = values.and_then(|values| values.checked_mul(pairs));
    if pairs == 0
        || pairs > MAX_PAIRS
        || resamples == 0
        || resamples > MAX_RESAMPLES
        || cells == 0
        || cells > MAX_PRIMARY_CELLS
        || values.is_none_or(|n| n > MAX_STORED_VALUES)
        || work.is_none_or(|n| n > MAX_ACCUMULATIONS)
    {
        return Err(ComparisonError("bootstrap CPU/memory limits exceeded (4096 pairs, 1M resamples, 64 primary cells, 4M stored values, 64M accumulations)".into()));
    }
    Ok(())
}
