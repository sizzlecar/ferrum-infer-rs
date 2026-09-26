//! Measured search work only guides when to stop optional improvement. It is
//! neither a future runtime bound nor a certificate: replay uses its real clock.
use super::PlanningUnknownReason;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct MeasuredReplayWork(u64);

impl MeasuredReplayWork {
    pub(super) fn with_span(
        self,
        start_ns: u64,
        end_ns: u64,
    ) -> Result<Self, PlanningUnknownReason> {
        let elapsed = end_ns
            .checked_sub(start_ns)
            .ok_or(PlanningUnknownReason::ClockMovedBackwards)?;
        self.0
            .checked_add(elapsed)
            .map(Self)
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)
    }

    pub(super) fn ns(self) -> u64 {
        self.0
    }
}

/// The original phase endpoints are ceilings. A complete common plan can move
/// only the optional endpoint earlier; it never changes the transaction origin,
/// planner/publication endpoint, physical authority, or any request obligation.
pub(super) struct ReplayReserve {
    origin_ns: u64,
    configured_search_end_ns: u64,
    planner_end_ns: u64,
    configured_final_window_ns: u64,
    search_end_ns: u64,
    measured: MeasuredReplayWork,
    reserved_ns: u64,
}

impl ReplayReserve {
    pub(super) fn new(origin_ns: u64, search_end_ns: u64, planner_end_ns: u64) -> Self {
        // PlanningPhaseBudget checked the order. With an early outer cap the
        // optional endpoint can equal the origin, while construct still runs.
        Self {
            origin_ns,
            configured_search_end_ns: search_end_ns,
            planner_end_ns,
            configured_final_window_ns: planner_end_ns - search_end_ns,
            search_end_ns,
            measured: MeasuredReplayWork::default(),
            reserved_ns: 0,
        }
    }

    pub(super) fn observe_complete(
        &mut self,
        work: MeasuredReplayWork,
    ) -> Result<(), PlanningUnknownReason> {
        let measured = MeasuredReplayWork(self.measured.0.max(work.0));
        // F is configured slack for final ranking/checks and estimation error,
        // not a measured upper bound. R is only the complete path's observed
        // begin/advance wall time; no sibling or GPU prediction is accumulated.
        let reserved = self
            .configured_final_window_ns
            .checked_add(measured.0)
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        let deadline = self
            .planner_end_ns
            .checked_sub(reserved)
            .unwrap_or(self.origin_ns)
            .max(self.origin_ns)
            .min(self.configured_search_end_ns);
        self.search_end_ns = self.search_end_ns.min(deadline);
        self.measured = measured;
        self.reserved_ns = reserved;
        Ok(())
    }

    pub(super) fn deadline_ns(&self) -> u64 {
        self.search_end_ns
    }

    pub(super) fn is_early_stop(&self, now_ns: u64) -> bool {
        now_ns >= self.search_end_ns && now_ns < self.configured_search_end_ns
    }

    pub(super) fn measured_ns(&self) -> u64 {
        self.measured.ns()
    }

    pub(super) fn reserved_ns(&self) -> u64 {
        self.reserved_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_reserve_uses_original_origin_and_never_reopens_after_cheaper_plan() {
        let mut policy = ReplayReserve::new(10_000, 11_200, 11_600);
        let parent = MeasuredReplayWork::default()
            .with_span(10_300, 10_350)
            .unwrap();
        let left = parent.with_span(10_400, 10_900).unwrap();
        let right = parent.with_span(10_950, 11_000).unwrap();
        // Siblings each include the shared parent's 50ns once, not each other.
        assert_eq!(left.ns(), 550);
        assert_eq!(right.ns(), 100);
        policy.observe_complete(left).unwrap();
        assert_eq!(policy.deadline_ns(), 10_650);
        assert_eq!(policy.reserved_ns(), 950);
        policy.observe_complete(right).unwrap();
        assert_eq!(policy.deadline_ns(), 10_650);
        assert_eq!(policy.measured_ns(), 550);
    }

    #[test]
    fn replay_reserve_handles_overflow_and_closes_optional_window_without_wrapping() {
        let mut policy = ReplayReserve::new(0, 1_200, 1_600);
        assert_eq!(
            policy.observe_complete(MeasuredReplayWork(u64::MAX)),
            Err(PlanningUnknownReason::ArithmeticOverflow)
        );
        assert_eq!(
            policy.deadline_ns(),
            1_200,
            "failed update is not partially installed"
        );
        assert_eq!(
            MeasuredReplayWork(u64::MAX).with_span(0, 1),
            Err(PlanningUnknownReason::ArithmeticOverflow)
        );
        assert_eq!(
            MeasuredReplayWork::default().with_span(2, 1),
            Err(PlanningUnknownReason::ClockMovedBackwards)
        );
        policy.observe_complete(MeasuredReplayWork(1_500)).unwrap();
        assert_eq!(policy.deadline_ns(), 0);
        assert_eq!(policy.reserved_ns(), 1_900);
    }
}
