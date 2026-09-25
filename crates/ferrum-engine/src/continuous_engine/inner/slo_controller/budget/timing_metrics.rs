//! Public counters consume the existing once-only audit. No new clock reads or
//! device timing mode changes are needed to observe controller execution.
use super::*;
use ferrum_types::{ControllerTimingMetrics, ControllerWitnessAggregate, WallTimingAggregate};

fn add(target: &mut u64, value: u64) {
    *target = target.saturating_add(value);
}

fn timing(target: &mut WallTimingAggregate, wall_ns: u64, calls: u64) {
    add(&mut target.wall_ns_total, wall_ns);
    add(&mut target.calls, calls);
}

fn witness(
    target: &mut ControllerWitnessAggregate,
    value: ControllerWitnessAudit,
    audit: &ControllerAudit,
) {
    add(&mut target.samples, 1);
    add(&mut target.waves_total, value.waves);
    add(&mut target.tail_waves_total, value.tail_waves);
    add(
        &mut target.nonempty_tail_samples,
        u64::from(value.tail_waves > 0),
    );
    add(
        &mut target.enumeration_attempts,
        audit.search.enumeration_attempts as u64,
    );
    add(
        &mut target.generated_candidates,
        audit.search.generated_candidates as u64,
    );
    add(
        &mut target.expanded_candidates,
        audit.search.expanded_candidates as u64,
    );
    if !audit.clock_invalid {
        if audit.planning_wall_ns != u64::MAX {
            timing(&mut target.planning, audit.planning_wall_ns, 1);
        }
        let search = audit.stages[ControllerStage::SearchReplay as usize];
        timing(&mut target.search_replay, search.wall_ns, search.calls);
    }
}

pub(super) fn accumulate(totals: &mut ControllerTimingMetrics, audit: &ControllerAudit) {
    add(&mut totals.finalized_transactions, 1);
    add(
        &mut totals.invalid_clock_transactions,
        u64::from(audit.clock_invalid),
    );
    add(
        &mut totals.unknown_decisions,
        u64::from(audit.decision == "unknown"),
    );
    add(
        &mut totals.planner_phase_exhaustions,
        u64::from(audit.planner_budget_exhausted),
    );
    add(
        &mut totals.hard_budget_exhaustions,
        u64::from(audit.budget_exhausted),
    );
    add(
        &mut totals.backend_submitted,
        u64::from(audit.backend_submitted),
    );
    add(
        &mut totals.host_reconciled,
        u64::from(audit.host_reconciled),
    );
    if let Some(value) = audit.witness {
        witness(&mut totals.witnesses.decisions, value, audit);
        if audit.backend_submitted {
            witness(&mut totals.witnesses.backend_submitted, value, audit);
        }
        if audit.host_reconciled {
            witness(&mut totals.witnesses.host_reconciled, value, audit);
        }
    }
    match audit.outcome {
        "submitted" => add(&mut totals.submitted, 1),
        "observed" => add(&mut totals.observed, 1),
        "idle" => add(&mut totals.idle, 1),
        "failed" => add(&mut totals.failed, 1),
        "withdrawn" => add(&mut totals.withdrawn, 1),
        "calibration_blocked" => add(&mut totals.calibration_blocked, 1),
        _ => {}
    }
    if audit.planning_wall_ns == u64::MAX {
        add(&mut totals.unfinished_planning_transactions, 1);
    }
    // An invalid monotonic clock is diagnostic evidence, not a valid latency.
    if audit.clock_invalid {
        return;
    }
    if audit.planning_wall_ns != u64::MAX {
        timing(&mut totals.planning, audit.planning_wall_ns, 1);
    }
    timing(&mut totals.transaction, audit.transaction_wall_ns, 1);
    for stage in audit.stages {
        let target = match stage.stage {
            ControllerStage::Release => &mut totals.release,
            ControllerStage::Capture => &mut totals.capture,
            ControllerStage::SearchReplay => &mut totals.search_replay,
            ControllerStage::Publication => &mut totals.publication,
            ControllerStage::ReadyQueue => &mut totals.ready_queue,
            ControllerStage::IterationLock => &mut totals.iteration_lock_wait,
            ControllerStage::InputPreparation => &mut totals.input_preparation,
            ControllerStage::HostGuard => &mut totals.host_guard,
            ControllerStage::ExecutorAwait => &mut totals.executor_await,
            ControllerStage::Reconciliation => &mut totals.reconciliation,
        };
        timing(target, stage.wall_ns, stage.calls);
    }
}

impl EngineInner {
    pub(in crate::continuous_engine) fn controller_timing_snapshot(
        &self,
    ) -> Option<ControllerTimingMetrics> {
        self.slo_controller.lock().timing.clone()
    }
}
