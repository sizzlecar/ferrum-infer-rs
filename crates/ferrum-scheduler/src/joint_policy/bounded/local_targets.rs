//! Nearest-rank targets on a fixed, captured live-cohort service population.
//! These conditional prefixes do not predict future visible SSE events or
//! include requests that retired before capture. Rank weights describe the
//! supplied cost scenarios, not statistical confidence or service quantiles.
use super::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalTargetClass {
    Within,
    Outside,
    #[default]
    Undefined,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundedSelectionReason {
    #[default]
    StaticRetained,
    WithinTargetsFaster,
    WithinTargetsRecovery,
    ParetoTargetRepair,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct BoundedMetricTarget {
    pub classification: LocalTargetClass,
    /// Rank-weighted fixed-population counts, not an empirical run sample.
    pub samples: f64,
    pub unknown_samples: f64,
    pub violations: f64,
    pub allowed_violations: f64,
    /// Positive strict-bad count above N-ceil(q*N), divided by N, evaluated
    /// within each rank before weighting. This is NOT a latency magnitude.
    pub excess_fraction: f64,
    pub defined_mass: f64,
    pub within_mass: f64,
    pub undefined_mass: f64,
    pub outside_mass: f64,
}

impl BoundedMetricTarget {
    fn from_counts(
        samples: usize,
        unknown: usize,
        bad: usize,
        q: f64,
    ) -> Result<Self, Unsupported> {
        if unknown > samples || bad > samples - unknown || !q.is_finite() || q <= 0.0 || q > 1.0 {
            return Err(Unsupported::InvalidState);
        }
        // Same nearest-rank convention as the frozen Rust client. Samples at
        // the limit are good. No sort or future-output expansion is required.
        let rank = ((q * samples as f64).ceil() as usize).max(1).min(samples);
        let allowed = samples - rank;
        let defined = samples > 0 && unknown == 0;
        let within = defined && bad <= allowed;
        Ok(Self {
            classification: if !defined {
                LocalTargetClass::Undefined
            } else if within {
                LocalTargetClass::Within
            } else {
                LocalTargetClass::Outside
            },
            samples: samples as f64,
            unknown_samples: unknown as f64,
            violations: bad as f64,
            allowed_violations: allowed as f64,
            excess_fraction: if samples == 0 {
                0.0
            } else {
                bad.saturating_sub(allowed) as f64 / samples as f64
            },
            defined_mass: if defined { 1.0 } else { 0.0 },
            within_mass: if within { 1.0 } else { 0.0 },
            undefined_mass: if defined { 0.0 } else { 1.0 },
            outside_mass: if defined && !within { 1.0 } else { 0.0 },
        })
    }

    fn add(&mut self, value: Self, weight: f64) {
        self.samples += value.samples * weight;
        self.unknown_samples += value.unknown_samples * weight;
        self.violations += value.violations * weight;
        self.allowed_violations += value.allowed_violations * weight;
        self.excess_fraction += value.excess_fraction * weight;
        self.defined_mass += value.defined_mass * weight;
        self.within_mass += value.within_mass * weight;
        self.undefined_mass += value.undefined_mass * weight;
        self.outside_mass += value.outside_mass * weight;
        // Even a tiny positive supplied rank remains evidence. Probability
        // normalization tolerances must never erase a real unknown or failure.
        self.classification = if self.undefined_mass > 0.0 {
            LocalTargetClass::Undefined
        } else if self.outside_mass > 0.0 {
            LocalTargetClass::Outside
        } else {
            LocalTargetClass::Within
        };
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct BoundedLocalTargets {
    pub ttft: BoundedMetricTarget,
    pub tpot: BoundedMetricTarget,
    pub itl: BoundedMetricTarget,
}

impl BoundedLocalTargets {
    pub fn all_within(self) -> bool {
        self.metrics()
            .iter()
            .all(|metric| metric.classification == LocalTargetClass::Within)
    }

    fn metrics(self) -> [BoundedMetricTarget; 3] {
        [self.ttft, self.tpot, self.itl]
    }

    pub(super) fn add(&mut self, value: Self, weight: f64) {
        self.ttft.add(value.ttft, weight);
        self.tpot.add(value.tpot, weight);
        self.itl.add(value.itl, weight);
    }
}

fn increment(value: &mut usize, amount: usize) -> Result<(), Unsupported> {
    *value = value.checked_add(amount).ok_or(Unsupported::Arithmetic)?;
    Ok(())
}

pub(super) fn evaluate(
    initial: &[SimulatedRequest],
    baseline: &TrajectoryResult,
    candidate: &TrajectoryResult,
    config: &SchedulerSloConfig,
) -> Result<(BoundedLocalTargets, f64, BoundedScore, f64), Unsupported> {
    let baseline_timing = baseline
        .bounded_timing
        .as_ref()
        .ok_or(Unsupported::InvalidState)?;
    let timing = candidate
        .bounded_timing
        .as_ref()
        .ok_or(Unsupported::InvalidState)?;
    let wave = timing
        .reference_reached_at
        .ok_or(Unsupported::InvalidState)?;
    if wave > candidate.central_elapsed {
        return Err(Unsupported::InvalidState);
    }
    let ttft_limit = Duration::try_from_secs_f64(config.ttft_ms / 1000.0)
        .map_err(|_| Unsupported::InvalidState)?;
    let tpot_limit = Duration::try_from_secs_f64(config.tpot_ms / 1000.0)
        .map_err(|_| Unsupported::InvalidState)?;
    let itl_limit = Duration::try_from_secs_f64(config.itl_ms / 1000.0)
        .map_err(|_| Unsupported::InvalidState)?;
    let (mut ttft_bad, mut tpot_count, mut tpot_unknown, mut tpot_bad, mut itl_count, mut itl_bad) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut guard = BoundedScore::default();
    let mut unfinished_tpot_debt = 0.0;
    for (((before, baseline_request), base), service) in initial
        .iter()
        .zip(&baseline.requests)
        .zip(&baseline_timing.requests)
        .zip(&timing.requests)
    {
        let reference = service.reference.ok_or(Unsupported::InvalidState)?;
        let base_reference = base.reference.ok_or(Unsupported::InvalidState)?;
        let target = baseline_request.progress();
        let progress_at = reference
            .progress_reached_at
            .ok_or(Unsupported::InvalidState)?;
        let clock = reference
            .central
            .ok_or(Unsupported::InvalidState)?
            .projection;
        let base_clock = base_reference
            .central
            .ok_or(Unsupported::InvalidState)?
            .projection;
        if reference.target != target
            || base_reference.target != target
            || reference.central != reference.risk
            || base_reference.central != base_reference.risk
            || reference.following_central_output_at != reference.following_risk_output_at
            || reference.output_initially_satisfied
                != (target.output_tokens == before.output_tokens)
            || progress_at > wave
            || reference
                .progress_at_reach
                .is_none_or(|progress| !progress.covers(target))
            || clock.interval_count != base_clock.interval_count
            || clock.violating_interval_count > clock.interval_count
        {
            return Err(Unsupported::InvalidState);
        }

        // TTFT has one proxy value for every captured request. If the fixed
        // frontier has no first opportunity, use only that frontier's progress
        // and attainment time. A zero-progress waiter retains a shared exposure
        // floor, never a fabricated terminal first-text prediction.
        let ttft_value =
            if let Some(first) = before.central_clock.first_output.or(clock.first_output) {
                first.as_secs_f64()
            } else {
                let progress = target
                    .prefill_tokens
                    .checked_sub(before.prefill_tokens)
                    .ok_or(Unsupported::InvalidState)?;
                if progress == 0 {
                    before
                        .age
                        .checked_add(baseline.central_elapsed.max(candidate.central_elapsed))
                        .ok_or(Unsupported::Arithmetic)?
                        .as_secs_f64()
                } else {
                    before.age.as_secs_f64()
                        + progress_at.as_secs_f64()
                            * (1.0
                                + (before.prompt_tokens - target.prefill_tokens) as f64
                                    / progress as f64)
                }
            };
        if !ttft_value.is_finite() {
            return Err(Unsupported::Arithmetic);
        }
        increment(
            &mut ttft_bad,
            usize::from(ttft_value > ttft_limit.as_secs_f64()),
        )?;
        guard.ttft_pressure += (ttft_value / ttft_limit.as_secs_f64() - 1.0).max(0.0);

        // Eligibility and generated-token denominator are fixed by static.
        // Generated history without any initial timing anchor is not repaired
        // by a later simulated first opportunity.
        if target.output_tokens >= 2 {
            increment(&mut tpot_count, 1)?;
            match (clock.first_output, clock.last_output) {
                (Some(first), Some(last))
                    if clock.interval_count > 0
                        && (before.output_tokens == 0
                            || before.central_clock.first_output.is_some()) =>
                {
                    let elapsed = last.checked_sub(first).ok_or(Unsupported::InvalidState)?;
                    let average = elapsed.as_secs_f64() / (target.output_tokens - 1) as f64;
                    increment(
                        &mut tpot_bad,
                        usize::from(average > tpot_limit.as_secs_f64()),
                    )?;
                    // The quantile sample above is the fixed ordinal. A still
                    // unfinished reference additionally owes time through the
                    // actual common observation boundary. Reaching its ordinal
                    // early, or emitting extra uncredited outputs afterwards,
                    // must not erase that open debt in degradation mode.
                    let debt_end = if baseline_request.finished() {
                        last
                    } else {
                        before
                            .age
                            .checked_add(candidate.central_elapsed)
                            .ok_or(Unsupported::Arithmetic)?
                    };
                    let debt_elapsed = debt_end
                        .checked_sub(first)
                        .ok_or(Unsupported::InvalidState)?;
                    guard.tpot_time_debt += (debt_elapsed.as_secs_f64() / tpot_limit.as_secs_f64()
                        - (target.output_tokens - 1) as f64)
                        .max(0.0);
                }
                _ => increment(&mut tpot_unknown, 1)?,
            }
        }

        // A pending first interval can owe time even though TPOT itself is
        // still undefined. Only known anchors are used; silent generated
        // history is not turned into an observed first timestamp. Finished
        // static references contribute no open debt, even if an alternative
        // has executed more work beyond some other request's target.
        if !baseline_request.finished()
            && target.output_tokens > 0
            && (before.output_tokens == 0 || before.central_clock.first_output.is_some())
        {
            if let Some(first) = clock.first_output {
                let end = before
                    .age
                    .checked_add(candidate.central_elapsed)
                    .ok_or(Unsupported::Arithmetic)?;
                let elapsed = end.checked_sub(first).ok_or(Unsupported::InvalidState)?;
                unfinished_tpot_debt += (elapsed.as_secs_f64() / tpot_limit.as_secs_f64()
                    - (target.output_tokens - 1) as f64)
                    .max(0.0);
            }
        }

        // Captured historical intervals and the SAME reference-ordinal service
        // intervals form a conditional local proxy. Additional output can close
        // at most the one predeclared pending slot; it never adds good samples.
        increment(&mut itl_count, base_clock.interval_count)?;
        increment(&mut itl_bad, clock.violating_interval_count)?;
        if !baseline_request.finished()
            && base_clock.completed.is_none()
            && base_clock.last_output.is_some()
        {
            increment(&mut itl_count, 1)?;
            let last = clock.last_output.ok_or(Unsupported::InvalidState)?;
            let observed_until = before
                .age
                .checked_add(candidate.central_elapsed)
                .ok_or(Unsupported::Arithmetic)?;
            let end = reference
                .following_central_output_at
                .unwrap_or(observed_until);
            if end > observed_until {
                return Err(Unsupported::InvalidState);
            }
            let gap = end.checked_sub(last).ok_or(Unsupported::InvalidState)?;
            increment(&mut itl_bad, usize::from(gap > itl_limit))?;
        }
    }
    guard.ttft_pressure /= initial.len() as f64;
    guard.tpot_time_debt /= initial.len() as f64;
    guard.itl_pressure = itl_bad as f64 / initial.len() as f64;
    unfinished_tpot_debt /= initial.len() as f64;
    if !guard.total().is_finite() || !unfinished_tpot_debt.is_finite() {
        return Err(Unsupported::Arithmetic);
    }
    Ok((
        BoundedLocalTargets {
            ttft: BoundedMetricTarget::from_counts(
                initial.len(),
                0,
                ttft_bad,
                config.joint.ttft_quantile,
            )?,
            tpot: BoundedMetricTarget::from_counts(
                tpot_count,
                tpot_unknown,
                tpot_bad,
                config.joint.tpot_quantile,
            )?,
            itl: BoundedMetricTarget::from_counts(
                itl_count,
                0,
                itl_bad,
                config.joint.itl_quantile,
            )?,
        },
        wave.as_secs_f64(),
        guard,
        unfinished_tpot_debt,
    ))
}

fn supported(value: &BoundedActionEvidence) -> bool {
    value.unsupported_mass == 0.0
        && (value.supported_mass - 1.0).abs() <= 1e-9
        && value.prefill_progress_regressions == 0
        && value.overdue_prefill_regressions == 0
}

fn eligible(candidate: &BoundedActionEvidence, baseline: &BoundedActionEvidence) -> bool {
    supported(candidate)
        && candidate.unfinished_reference_tpot_debt <= baseline.unfinished_reference_tpot_debt
}

fn dominates(candidate: &BoundedActionEvidence, reference: &BoundedActionEvidence) -> bool {
    let mut strict = false;
    for (candidate, baseline) in candidate
        .local_targets
        .metrics()
        .iter()
        .zip(reference.local_targets.metrics())
    {
        if candidate.samples != baseline.samples
            || candidate.unknown_samples != baseline.unknown_samples
            || candidate.defined_mass != baseline.defined_mass
        {
            return false;
        }
        if baseline.classification == LocalTargetClass::Undefined {
            continue;
        }
        if baseline.classification == LocalTargetClass::Within
            && candidate.classification != LocalTargetClass::Within
        {
            return false;
        }
        if candidate.excess_fraction > baseline.excess_fraction {
            return false;
        }
        strict |= candidate.excess_fraction < baseline.excess_fraction;
    }
    // Fixed-reference pressures are conservative non-worsening guards only in
    // degradation mode. Final catchup scores remain diagnostic, never guards.
    // These are not summed or claimed to reconstruct a P95 latency magnitude.
    strict
        && candidate.reference_pressure.ttft_pressure <= reference.reference_pressure.ttft_pressure
        && candidate.reference_pressure.tpot_time_debt
            <= reference.reference_pressure.tpot_time_debt
        && candidate.reference_pressure.itl_pressure <= reference.reference_pressure.itl_pressure
}

pub(super) fn select(
    baseline: &BoundedActionEvidence,
    alternatives: &[BoundedActionEvidence],
) -> (Option<usize>, BoundedSelectionReason) {
    let static_within = baseline.local_targets.all_within();
    let mut best_time = static_within.then_some(baseline.reference_time_seconds);
    let mut selected = None;
    for (index, candidate) in alternatives.iter().enumerate() {
        if !eligible(candidate, baseline) || !candidate.local_targets.all_within() {
            continue;
        }
        if best_time.is_none_or(|time| candidate.reference_time_seconds < time) {
            selected = Some(index);
            best_time = Some(candidate.reference_time_seconds);
        }
    }
    if best_time.is_some() {
        return (
            selected,
            match selected {
                None => BoundedSelectionReason::StaticRetained,
                Some(_) if static_within => BoundedSelectionReason::WithinTargetsFaster,
                Some(_) => BoundedSelectionReason::WithinTargetsRecovery,
            },
        );
    }
    // A frontier with incomparable repairs has no declared preference. Retain
    // static rather than inventing another weighting or tie-breaking priority.
    let mut unique_best = None;
    for (index, candidate) in alternatives.iter().enumerate() {
        if !eligible(candidate, baseline) || !dominates(candidate, baseline) {
            continue;
        }
        let dominates_every_repair = alternatives.iter().enumerate().all(|(other_index, other)| {
            other_index == index
                || !eligible(other, baseline)
                || !dominates(other, baseline)
                || dominates(candidate, other)
        });
        if dominates_every_repair {
            unique_best = Some(index);
            break;
        }
    }
    (
        unique_best,
        if unique_best.is_some() {
            BoundedSelectionReason::ParetoTargetRepair
        } else {
            BoundedSelectionReason::StaticRetained
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_fixed_population_preserves_threshold_and_unknown_mass() {
        let p95 = BoundedMetricTarget::from_counts(20, 0, 1, 0.95).unwrap();
        assert_eq!(p95.classification, LocalTargetClass::Within);
        assert_eq!(p95.allowed_violations, 1.0);
        let maximum = BoundedMetricTarget::from_counts(20, 0, 1, 1.0).unwrap();
        assert_eq!(maximum.classification, LocalTargetClass::Outside);
        assert_eq!(maximum.excess_fraction, 0.05);
        assert_eq!(
            BoundedMetricTarget::from_counts(0, 0, 0, 0.95)
                .unwrap()
                .classification,
            LocalTargetClass::Undefined
        );

        for (tail, expected) in [
            (
                BoundedMetricTarget::from_counts(20, 1, 0, 0.95).unwrap(),
                LocalTargetClass::Undefined,
            ),
            (
                BoundedMetricTarget::from_counts(20, 0, 2, 0.95).unwrap(),
                LocalTargetClass::Outside,
            ),
        ] {
            let mut combined = BoundedMetricTarget::default();
            combined.add(p95, 1.0 - 5e-10);
            combined.add(tail, 5e-10);
            assert_eq!(combined.classification, expected);
            assert!(combined.undefined_mass + combined.outside_mass > 0.0);
        }
    }
}
