use super::*;

// Whole-grid CPU ceiling, additional to the per-allocation bootstrap limits.
// This is a resource limit, never a statistical sample qualification threshold.
const MAX_GRID_ACCUMULATIONS: usize = 128_000_000;
const MAX_GRID_POINTS: usize = 128;

impl FrozenEligibilityPlan {
    pub fn configuration_sha256(&self) -> Result<String, ComparisonError> {
        json_digest(self)
    }
}

pub(super) fn validate(
    plan: &FrozenEligibilityPlan,
    configuration: &FrozenPairedBootstrap,
    primary_cells: usize,
) -> Result<(), EligibilityFailure> {
    let valid_positive = |value: f64| value.is_finite() && value > 0.0;
    if plan.schema_version != 1
        || plan.frozen_unix_ns == 0
        || plan.planning_pair_counts.is_empty()
        || plan.planning_pair_counts.len() > MAX_GRID_POINTS
        || plan.planning_pair_counts[0] < 2
        || plan
            .planning_pair_counts
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || plan.planning_resamples == 0
        || !valid_positive(plan.maximum_request_rank_step)
        || plan.maximum_request_rank_step > 0.01
        || !valid_positive(plan.maximum_visible_gap_rank_step)
        || plan.maximum_visible_gap_rank_step > 0.01
        || !valid_positive(plan.maximum_order_log_ratio_shift)
        || !valid_positive(plan.maximum_time_trend_log_ratio_shift)
        || !valid_positive(plan.maximum_absolute_lag_one_correlation)
        || plan.maximum_absolute_lag_one_correlation >= 1.0
        || plan.maximum_within_pair_idle_ns == 0
        || plan.independent_block_protocol.trim().is_empty()
        || plan.independent_block_protocol
            != configuration.declared_design.independent_block_protocol
    {
        return Err(failure(
            "invalid explicit eligibility planning/support/diagnostic configuration",
        ));
    }
    let mut total = 0_usize;
    for &pairs in &plan.planning_pair_counts {
        configuration::check_work(
            pairs as usize,
            plan.planning_resamples as usize,
            primary_cells,
        )?;
        let work = primary_cells
            .checked_mul(ComparisonMetric::ALL.len())
            .and_then(|n| n.checked_mul(plan.planning_resamples as usize))
            .and_then(|n| n.checked_mul(pairs as usize));
        total = work
            .and_then(|n| total.checked_add(n))
            .filter(|n| *n <= MAX_GRID_ACCUMULATIONS)
            .ok_or_else(|| failure("cumulative planning-grid CPU limit exceeded"))?;
    }
    Ok(())
}

pub(super) fn forecasts(
    plan: &FrozenEligibilityPlan,
    configuration: &FrozenPairedBootstrap,
    primary: &[&CellComparison],
) -> Result<Vec<PairAllocationForecast>, EligibilityFailure> {
    let columns: Vec<Vec<_>> = primary
        .iter()
        .flat_map(|cell| {
            ComparisonMetric::ALL.into_iter().map(|metric| {
                cell.pairs
                    .iter()
                    .map(|pair| pair.ratios[&metric].candidate_over_baseline)
                    .collect()
            })
        })
        .collect();
    let family = columns.len();
    let tail =
        (configuration.family_alpha - configuration.monte_carlo_error_budget) / family as f64;
    let rank = compute::tail_rank(
        plan.planning_resamples as usize,
        tail,
        configuration.monte_carlo_error_budget / family as f64,
    )
    .ok_or_else(|| {
        failure("pilot-planning simulation cannot resolve the full-family tail budget")
    })?;
    let mut result = Vec::with_capacity(plan.planning_pair_counts.len());
    for &pairs in &plan.planning_pair_counts {
        let mut distributions = compute::resample_means_for_size(
            &columns,
            plan.planning_resamples as usize,
            pairs as usize,
            plan.seed,
        )?;
        let mut metrics = Vec::with_capacity(family);
        for (index, (concurrency, metric)) in primary
            .iter()
            .flat_map(|cell| {
                ComparisonMetric::ALL
                    .into_iter()
                    .map(|metric| (cell.concurrency, metric))
            })
            .enumerate()
        {
            let column = &columns[index];
            let point = mean(column);
            let degenerate = column.iter().all(|ratio| *ratio == column[0]);
            let distribution = &mut distributions[index];
            let bound = if degenerate {
                None
            } else {
                distribution.sort_unstable_by(f64::total_cmp);
                Some(
                    distribution[if metric.is_throughput() {
                        rank - 1
                    } else {
                        distribution.len() - rank
                    }],
                )
            };
            let width = bound
                .map(|bound| {
                    if metric.is_throughput() {
                        point / bound - 1.0
                    } else {
                        bound / point - 1.0
                    }
                })
                .filter(|width| width.is_finite())
                .map(|width| width.max(0.0));
            metrics.push(PlanningMetricForecast {
                concurrency,
                metric,
                mean_pilot_ratio: point,
                one_sided_bound: bound,
                relative_bound_width: width,
                precision_target_met: width.is_some_and(|width| {
                    width <= configuration.maximum_relative_bound_width[&metric]
                }),
            });
        }
        let all_precision_targets_met = metrics.iter().all(|metric| metric.precision_target_met);
        result.push(PairAllocationForecast {
            planned_pairs: pairs,
            metrics,
            all_precision_targets_met,
        });
    }
    Ok(result)
}

pub(super) fn diagnostics(
    primary: &[&CellComparison],
    plan: &FrozenEligibilityPlan,
    first_order: PairedArmOrder,
) -> Result<Vec<PilotMetricDiagnostic>, EligibilityFailure> {
    let mut output = Vec::new();
    for cell in primary {
        for metric in ComparisonMetric::ALL {
            let ratios: Vec<_> = cell
                .pairs
                .iter()
                .map(|pair| pair.ratios[&metric].candidate_over_baseline)
                .collect();
            let logs: Vec<_> = ratios.iter().map(|ratio| ratio.ln()).collect();
            let average = mean(&logs);
            let center = (logs.len() - 1) as f64 / 2.0;
            let squares: f64 = logs.iter().map(|value| (value - average).powi(2)).sum();
            if !squares.is_finite() || squares <= 0.0 {
                return Err(failure(format!("pilot C{} / {metric:?} is degenerate; constant observations cannot certify planning precision", cell.concurrency)));
            }
            let lag = logs
                .windows(2)
                .map(|pair| (pair[0] - average) * (pair[1] - average))
                .sum::<f64>()
                / squares;
            let time_squares: f64 = (0..logs.len())
                .map(|index| (index as f64 - center).powi(2))
                .sum();
            let trend = logs
                .iter()
                .enumerate()
                .map(|(index, value)| (index as f64 - center) * (value - average))
                .sum::<f64>()
                / time_squares
                * (logs.len() - 1) as f64;
            let order = logs
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let baseline_first =
                        (first_order == PairedArmOrder::BaselineFirst) ^ (index % 2 == 1);
                    if baseline_first {
                        *value
                    } else {
                        -*value
                    }
                })
                .sum::<f64>()
                / logs.len() as f64;
            let reverse = mean(&ratios.iter().map(|ratio| 1.0 / ratio).collect::<Vec<_>>());
            if [lag, trend, order, reverse]
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(failure("non-finite pilot acquisition diagnostic"));
            }
            output.push(PilotMetricDiagnostic {
                concurrency: cell.concurrency,
                metric,
                mean_forward_ratio: mean(&ratios),
                mean_reverse_ratio: reverse,
                order_log_ratio_shift: order,
                time_trend_log_ratio_shift: trend,
                lag_one_correlation: lag,
                within_declared_limits: order.abs() <= plan.maximum_order_log_ratio_shift
                    && trend.abs() <= plan.maximum_time_trend_log_ratio_shift
                    && lag.abs() <= plan.maximum_absolute_lag_one_correlation,
            });
        }
    }
    Ok(output)
}
