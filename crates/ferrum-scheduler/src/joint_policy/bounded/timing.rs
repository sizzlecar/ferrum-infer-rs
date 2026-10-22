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
}

/// Only BoundedRounds produces this payload. `TrajectoryResult::requests`
/// preserves its input text clocks byte-for-byte, including no-visible-text
/// history with nonzero generated tokens and event/token count mismatches.
#[derive(Debug, Clone)]
pub struct BoundedTiming {
    pub(in crate::joint_policy) requests: Vec<RequestServiceTiming>,
}

impl BoundedTiming {
    /// Dynamic payload for all simultaneously retained static/alternative
    /// results. Ranks are visited and released sequentially, not retained.
    pub fn retained_bytes_bound(requests: usize, trajectories: usize) -> Option<usize> {
        requests
            .checked_mul(trajectories)?
            .checked_mul(std::mem::size_of::<RequestServiceTiming>())
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
        }));
        Ok(Self { requests: clocks })
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
        let result = super::super::apply_member(request, work, central, risk, itl_limit, coupling);
        timing.central.projection = request.central_clock;
        timing.risk.projection = request.risk_clock;
        request.central_clock = observed_central;
        request.risk_clock = observed_risk;
        result
    }
}
