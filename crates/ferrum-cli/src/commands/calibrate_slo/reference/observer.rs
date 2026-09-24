use super::*;

/// One full singleton discovery request. Only declared target waves are kept.
pub(in crate::commands::calibrate_slo) struct DiscoveryObserver {
    pub(super) target: DiscoveryTarget,
    granule: NonZeroU32,
    piecewise: Option<PiecewiseReferenceSpec>,
    decode_before: u64,
    maximum_segments: usize,
    origin: Option<CalibrationFrontier>,
    generation: u64,
    prefix: usize,
    complete: bool,
    pub(super) samples: Vec<CalibrationReferenceDiscoverySample>,
}

impl DiscoveryObserver {
    pub(in crate::commands::calibrate_slo) fn new(
        config: &ReferenceConfig,
        target: DiscoveryTarget,
    ) -> Result<Self> {
        config.limits.validate().map_err(invalid)?;
        config.singleton_case(target)?;
        Ok(Self {
            target,
            granule: config.granule_tokens,
            piecewise: config.piecewise.clone(),
            decode_before: config.decode_unit.generated_before.get().into(),
            maximum_segments: (config.limits.max_points_per_curve.get() - 1)
                .min(config.limits.max_samples.get()),
            origin: None,
            generation: 0,
            prefix: 0,
            complete: false,
            samples: Vec::new(),
        })
    }

    /// Called on the first real frontier, before that request's first wave.
    pub(in crate::commands::calibrate_slo) fn frontier(
        &mut self,
        frontier: &CalibrationFrontier,
    ) -> Result<()> {
        if let Some(origin) = &self.origin {
            if origin.request_id() != frontier.request_id()
                || origin.owner_incarnation() != frontier.owner_incarnation()
            {
                return Err(invalid("singleton discovery changed owner"));
            }
            return Ok(());
        }
        let input = frontier.request_evidence().original_input_tokens;
        if frontier.prefill_progress() != Some((0, input))
            || input == 0
            || frontier.generated_tokens() != 0
            || frontier.kv_tokens() != 0
            || input > u32::MAX as usize
            || match &self.piecewise {
                Some(spec) => spec
                    .segment_count(u32::try_from(input).map_err(|_| invalid("input overflow"))?)
                    .map_err(|e| invalid(e.to_string()))?,
                None => input.div_ceil(self.granule.get() as usize),
            } > self.maximum_segments
        {
            return Err(invalid(
                "discovery needs a bounded complete original input and fresh zero-prefix owner",
            ));
        }
        self.generation = frontier.work_generation().get();
        self.origin = Some(frontier.clone());
        Ok(())
    }

    pub(in crate::commands::calibrate_slo) fn wave(
        &mut self,
        session: &CalibrationSession,
        before: &[CalibrationWork],
        report: &CalibrationWaveReport,
    ) -> Result<ObservationProgress> {
        if let Some(result) = skip_wave(self.complete, report)? {
            return Ok(result);
        }
        let [work] = before else {
            return Err(invalid(
                "reference discovery is singleton, not a mixed/batched wall",
            ));
        };
        let frontier = work.frontier();
        let origin = self
            .origin
            .as_ref()
            .ok_or_else(|| invalid("missing discovery initial frontier"))?;
        if origin.request_id() != frontier.request_id()
            || origin.owner_incarnation() != frontier.owner_incarnation()
            || frontier.work_generation().get() != self.generation
            || origin.request_evidence() != frontier.request_evidence()
        {
            return Err(invalid("discovery frontier chain changed or skipped work"));
        }
        let next_generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("discovery generation overflow"))?;
        let (selected, complete, next_prefix) = match work.work() {
            ActualRowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } => {
                let remaining = total_prompt_tokens
                    .checked_sub(offset)
                    .ok_or_else(|| invalid("invalid discovery prefill span"))?;
                let expected_count = match &self.piecewise {
                    Some(spec) => spec
                        .next_count(total_prompt_tokens, offset)
                        .map_err(|e| invalid(e.to_string()))?
                        .get(),
                    None => remaining.min(self.granule.get()),
                };
                if offset as usize != self.prefix
                    || count != expected_count
                    || total_prompt_tokens as usize
                        != origin.request_evidence().original_input_tokens
                {
                    return Err(invalid(
                        "discovery differs from the declared complete granule partition",
                    ));
                }
                let end = offset
                    .checked_add(count)
                    .ok_or_else(|| invalid("discovery prefix overflow"))?;
                let selected = matches!(self.target, DiscoveryTarget::Prefill { .. });
                (
                    selected,
                    selected && end == total_prompt_tokens,
                    end as usize,
                )
            }
            ActualRowWork::Decode { .. } => {
                if !matches!(self.target, DiscoveryTarget::Decode)
                    || frontier.generated_tokens() as u64 > self.decode_before
                {
                    return Err(invalid("discovery skipped its declared target"));
                }
                let selected = frontier.generated_tokens() as u64 == self.decode_before;
                (selected, selected, self.prefix)
            }
            ActualRowWork::Restore | ActualRowWork::Maintenance => {
                return Err(invalid(
                    "restore/maintenance is not a singleton reference work sample",
                ))
            }
        };
        if selected && self.samples.len() >= self.maximum_segments {
            return Err(invalid("discovery sample capacity exhausted"));
        }
        // The engine validates actual observation, host commit and queue
        // acceptance. Intent alone never supplies a shape or reference cost.
        let observed = session.capture_reference_discovery(frontier, report)?;
        if selected {
            self.samples
                .try_reserve_exact(1)
                .map_err(|_| invalid("discovery receipt allocation failed"))?;
            self.samples.push(observed);
        }
        self.generation = next_generation;
        self.prefix = next_prefix;
        self.complete = complete;
        Ok(if complete {
            ObservationProgress::TargetComplete
        } else if selected {
            ObservationProgress::Recorded
        } else {
            ObservationProgress::PreparingTarget
        })
    }

    pub(in crate::commands::calibrate_slo) fn finish(self) -> Result<Self> {
        if !self.complete {
            return Err(invalid("request terminated before its declared discovery target; do not move or retry the target"));
        }
        Ok(self)
    }
}

/// The engine collector owns all original receipts. This borrowed observer is
/// scoped to one full request and cannot make heldout rows reference trials.
pub(in crate::commands::calibrate_slo) struct TrialObserver<'a> {
    collector: &'a mut CalibrationReferenceCollector,
    key: CalibrationReferenceTrial,
    begun: bool,
    complete: bool,
}
impl<'a> TrialObserver<'a> {
    pub(in crate::commands::calibrate_slo) fn new(
        collector: &'a mut CalibrationReferenceCollector,
        key: CalibrationReferenceTrial,
    ) -> Self {
        Self {
            collector,
            key,
            begun: false,
            complete: false,
        }
    }
    pub(in crate::commands::calibrate_slo) fn frontier(
        &mut self,
        session: &mut CalibrationSession,
        frontier: &CalibrationFrontier,
    ) -> Result<()> {
        if !self.begun {
            session.begin_reference_trial(self.collector, self.key, frontier)?;
            self.begun = true;
        }
        Ok(())
    }
    pub(in crate::commands::calibrate_slo) fn wave(
        &mut self,
        report: &CalibrationWaveReport,
    ) -> Result<ObservationProgress> {
        if let Some(result) = skip_wave(self.complete, report)? {
            return Ok(result);
        }
        if !self.begun {
            return Err(invalid("reference trial has no fresh registration"));
        }
        self.complete = self.collector.observe(self.key, report)?;
        Ok(if self.complete {
            ObservationProgress::TargetComplete
        } else if matches!(self.key, CalibrationReferenceTrial::Decode { .. }) {
            ObservationProgress::PreparingTarget
        } else {
            ObservationProgress::Recorded
        })
    }
    pub(in crate::commands::calibrate_slo) fn finish(self) -> Result<()> {
        if !self.complete {
            return Err(invalid("request terminated before all predeclared reference targets; full output cannot be truncated to manufacture a trial"));
        }
        Ok(())
    }
}

pub(in crate::commands::calibrate_slo) trait CohortObserver:
    Send
{
    fn initial_frontier(
        &mut self,
        session: &mut CalibrationSession,
        frontier: &CalibrationFrontier,
    ) -> Result<()>;
    fn observe_wave(
        &mut self,
        session: &CalibrationSession,
        before: &[CalibrationWork],
        report: &CalibrationWaveReport,
    ) -> Result<ObservationProgress>;
}
impl CohortObserver for DiscoveryObserver {
    fn initial_frontier(
        &mut self,
        _session: &mut CalibrationSession,
        frontier: &CalibrationFrontier,
    ) -> Result<()> {
        self.frontier(frontier)
    }
    fn observe_wave(
        &mut self,
        session: &CalibrationSession,
        before: &[CalibrationWork],
        report: &CalibrationWaveReport,
    ) -> Result<ObservationProgress> {
        self.wave(session, before, report)
    }
}
impl CohortObserver for TrialObserver<'_> {
    fn initial_frontier(
        &mut self,
        session: &mut CalibrationSession,
        frontier: &CalibrationFrontier,
    ) -> Result<()> {
        self.frontier(session, frontier)
    }
    fn observe_wave(
        &mut self,
        _session: &CalibrationSession,
        _before: &[CalibrationWork],
        report: &CalibrationWaveReport,
    ) -> Result<ObservationProgress> {
        self.wave(report)
    }
}

fn skip_wave(
    complete: bool,
    report: &CalibrationWaveReport,
) -> Result<Option<ObservationProgress>> {
    skip_submission(complete, report.submission, report.error.is_some())
}

pub(super) fn skip_submission(
    complete: bool,
    submission: CalibrationSubmissionState,
    failed: bool,
) -> Result<Option<ObservationProgress>> {
    if failed
        || matches!(
            submission,
            CalibrationSubmissionState::Submitted | CalibrationSubmissionState::InFlightUnknown
        )
    {
        return Err(invalid(
            "reference request failed or has indeterminate reconciliation",
        ));
    }
    if submission == CalibrationSubmissionState::NotSubmitted {
        return Ok(Some(ObservationProgress::NotSubmitted));
    }
    Ok(complete.then_some(ObservationProgress::AfterTarget))
}
