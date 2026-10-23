use super::*;

/// Generated service opportunities, not predicted text events. The initial
/// text first/last offsets are anchors; each simulated generated token adds
/// an opportunity after them, even when decoding that token may emit no text.
/// This type deliberately has no conversion back to an observed RequestClock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServiceOpportunityClock {
    pub(in crate::joint_policy) projection: RequestClock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::joint_policy) struct RequestServiceTiming {
    pub central: ServiceOpportunityClock,
    pub risk: ServiceOpportunityClock,
    /// Latest successful progress update, relative to this traversal's start.
    last_progress_at: Duration,
    pub reference: Option<ReferenceServiceTiming>,
}

/// Fixed-size evidence at the static reference's progress and output ordinal.
/// Reaching a prefill frontier need not produce an output opportunity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::joint_policy) struct ReferenceServiceTiming {
    pub target: Progress,
    /// Elapsed traversal time at first coverage, not an ingress timestamp.
    pub progress_reached_at: Option<Duration>,
    /// Actual post-invocation progress; a prefill invocation may overshoot.
    pub progress_at_reach: Option<Progress>,
    /// The reference asks for no output beyond the initial generated ordinal.
    /// Its frozen clock may legitimately have no first/last text anchor.
    pub output_initially_satisfied: bool,
    /// Frozen exactly at target.output_tokens; later work cannot replace it.
    pub central: Option<ServiceOpportunityClock>,
    pub risk: Option<ServiceOpportunityClock>,
    /// Only the first service opportunity after the reference output ordinal,
    /// relative to request ingress. These are not observed text events.
    pub following_central_output_at: Option<Duration>,
    pub following_risk_output_at: Option<Duration>,
}

/// Only BoundedRounds produces this payload. `TrajectoryResult::requests`
/// preserves its input text clocks byte-for-byte, including no-visible-text
/// history with nonzero generated tokens and event/token count mismatches.
#[derive(Debug, Clone)]
pub struct BoundedTiming {
    pub(in crate::joint_policy) requests: Vec<RequestServiceTiming>,
    /// First whole-wave boundary covering every static reference target.
    /// Independent of TrajectoryResult's earlier invocation-boundary target
    /// time, and never shortened by extra work performed for the common wall.
    pub(in crate::joint_policy) reference_reached_at: Option<Duration>,
}

impl BoundedTiming {
    /// Test-only copy of a generated-service clock, never an observed text
    /// clock. Production consumers cannot convert this projection back.
    #[cfg(test)]
    pub(crate) fn service_clock(&self, index: usize) -> Option<RequestClock> {
        self.requests
            .get(index)
            .map(|timing| timing.central.projection)
    }

    /// Test-only reference-ordinal service clock, not observed text history.
    #[cfg(test)]
    pub(crate) fn reference_service_clock(&self, index: usize) -> Option<RequestClock> {
        self.requests
            .get(index)?
            .reference?
            .central
            .map(|clock| clock.projection)
    }

    #[cfg(test)]
    pub(crate) fn reference_wave_elapsed(&self) -> Option<Duration> {
        self.reference_reached_at
    }

    /// Dynamic request payload plus the added whole-wave reference timestamp,
    /// for all simultaneously retained static/alternative results. Ranks are
    /// visited and released sequentially, not retained.
    pub fn retained_bytes_bound(requests: usize, trajectories: usize) -> Option<usize> {
        requests
            .checked_mul(std::mem::size_of::<RequestServiceTiming>())?
            .checked_add(std::mem::size_of::<Option<Duration>>())?
            .checked_mul(trajectories)
    }

    pub(in crate::joint_policy) fn new(requests: &[SimulatedRequest]) -> Result<Self, Unsupported> {
        let mut clocks = Vec::new();
        clocks
            .try_reserve_exact(requests.len())
            .map_err(|_| Unsupported::ParticipantBudget)?;
        clocks.extend(requests.iter().map(|request| RequestServiceTiming {
            central: ServiceOpportunityClock {
                projection: request.central_clock,
            },
            risk: ServiceOpportunityClock {
                projection: request.risk_clock,
            },
            last_progress_at: Duration::ZERO,
            reference: None,
        }));
        Ok(Self {
            requests: clocks,
            reference_reached_at: None,
        })
    }

    pub(in crate::joint_policy) fn bind_reference(
        &mut self,
        initial: &[SimulatedRequest],
        targets: &[Progress],
    ) -> Result<(), Unsupported> {
        if initial.len() != self.requests.len() || targets.len() != initial.len() {
            return Err(Unsupported::InvalidState);
        }
        for (request, target) in initial.iter().zip(targets) {
            if !target.covers(request.progress())
                || target.prefill_tokens > request.prompt_tokens
                || target.output_tokens > request.scenario_end_tokens
                || (target.output_tokens > 0 && target.prefill_tokens != request.prompt_tokens)
            {
                return Err(Unsupported::InvalidState);
            }
        }
        for ((timing, request), &target) in self.requests.iter_mut().zip(initial).zip(targets) {
            let already_covered = request.progress().covers(target);
            let output_initially_satisfied = request.output_tokens == target.output_tokens;
            timing.reference = Some(ReferenceServiceTiming {
                target,
                progress_reached_at: already_covered.then_some(Duration::ZERO),
                progress_at_reach: already_covered.then_some(request.progress()),
                output_initially_satisfied,
                central: output_initially_satisfied.then_some(timing.central),
                risk: output_initially_satisfied.then_some(timing.risk),
                following_central_output_at: None,
                following_risk_output_at: None,
            });
        }
        Ok(())
    }

    /// Static learns its reference only at its actual terminal wave boundary.
    /// Each request's final ordinal clock is already retained; no replay or
    /// future query is necessary to freeze it.
    pub(in crate::joint_policy) fn freeze_static_reference(
        &mut self,
        initial: &[SimulatedRequest],
        requests: &[SimulatedRequest],
        elapsed: Duration,
    ) -> Result<(), Unsupported> {
        if initial.len() != self.requests.len() || requests.len() != initial.len() {
            return Err(Unsupported::InvalidState);
        }
        for ((timing, before), request) in self.requests.iter_mut().zip(initial).zip(requests) {
            timing.reference = Some(ReferenceServiceTiming {
                target: request.progress(),
                progress_reached_at: Some(timing.last_progress_at),
                progress_at_reach: Some(request.progress()),
                output_initially_satisfied: request.output_tokens == before.output_tokens,
                central: Some(timing.central),
                risk: Some(timing.risk),
                following_central_output_at: None,
                following_risk_output_at: None,
            });
        }
        self.reference_reached_at = Some(elapsed);
        Ok(())
    }

    /// Called after the whole wave and its phase-progress validation.
    pub(in crate::joint_policy) fn finish_wave(&mut self, elapsed: Duration) {
        if self.reference_reached_at.is_none()
            && self.requests.iter().all(|timing| {
                timing
                    .reference
                    .is_some_and(|reference| reference.progress_reached_at.is_some())
            })
        {
            self.reference_reached_at = Some(elapsed);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::joint_policy) fn apply_member(
        &mut self,
        index: usize,
        request: &mut SimulatedRequest,
        work: CostWork,
        central: Duration,
        risk: Duration,
        itl_limit: Duration,
        coupling: ClockCoupling,
    ) -> Result<(), Unsupported> {
        let timing = self
            .requests
            .get_mut(index)
            .ok_or(Unsupported::InvalidState)?;
        // The internal checked operation invokes no adapter/provider callbacks.
        // Swap only Copy clocks, preserving its token mutation/error order
        // without cloning RequestId per member. Restore on both Ok and Err.
        let observed_central = request.central_clock;
        let observed_risk = request.risk_clock;
        request.central_clock = timing.central.projection;
        request.risk_clock = timing.risk.projection;
        let output_before = request.output_tokens;
        let result = super::super::apply_member(request, work, central, risk, itl_limit, coupling);
        timing.central.projection = request.central_clock;
        timing.risk.projection = request.risk_clock;
        request.central_clock = observed_central;
        request.risk_clock = observed_risk;
        result?;
        timing.last_progress_at = central;
        if let Some(reference) = &mut timing.reference {
            if reference.progress_reached_at.is_none()
                && request.progress().covers(reference.target)
            {
                reference.progress_reached_at = Some(central);
                reference.progress_at_reach = Some(request.progress());
            }
            if reference.central.is_none()
                && request.output_tokens == reference.target.output_tokens
            {
                reference.central = Some(timing.central);
                reference.risk = Some(timing.risk);
            } else if request.output_tokens > reference.target.output_tokens
                && request.output_tokens > output_before
                && reference.following_central_output_at.is_none()
            {
                reference.following_central_output_at = timing.central.projection.last_output;
                reference.following_risk_output_at = timing.risk.projection.last_output;
            }
        }
        Ok(())
    }
}
