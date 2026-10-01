//! Use the model produced by the original source8 population in the ordinary
//! SLO search and publication path. No witness or selected wave is fabricated.
use super::*;
use crate::continuous_engine::inner::slo_controller::SloIterationPlan;

pub(super) async fn submit(
    session: &CalibrationSession,
    executor: &ControlledExecutor,
    ids: &[RequestId],
) {
    submit_checked(session, executor, ids, ids.len(), false, None).await;
    assert!(session.frontiers().unwrap().is_empty());
}

pub(super) async fn submit_nonterminal(
    session: &CalibrationSession,
    executor: &ControlledExecutor,
    ids: &[RequestId],
    requires_forward_tail: bool,
) {
    // Nonfinal Prefill cannot commit the first output token in this wave.
    // Its real witness must reach that commit through an independently
    // checked tail. Final Prefill commits one token but leaves output work.
    let maximum_tokens = usize::try_from(
        session
            .test_engine_inner()
            .cost_runtime
            .as_ref()
            .unwrap()
            .workload_domain()
            .unwrap()
            .limits()
            .maximum_scheduled_tokens_per_wave
            .get(),
    )
    .expect("actual cost domain token capacity must fit the controller hint");
    assert!(maximum_tokens > 0);
    submit_checked(
        session,
        executor,
        ids,
        ids.len(),
        requires_forward_tail,
        Some(maximum_tokens),
    )
    .await;
    let frontiers = session.frontiers().unwrap();
    assert_eq!(frontiers.len(), ids.len());
    assert!(frontiers.iter().all(|row| ids.contains(row.request_id())));
}

pub(super) async fn submit_capacity_limited(
    session: &CalibrationSession,
    executor: &ControlledExecutor,
    ids: &[RequestId],
) {
    // Every ready decoder needs service in the common witness. With one row
    // per wave, multiple decoders require a genuine, independently replayed
    // tail even though only its first wave may execute.
    submit_checked(session, executor, ids, 1, false, None).await;
}

async fn submit_checked(
    session: &CalibrationSession,
    executor: &ControlledExecutor,
    ids: &[RequestId],
    maximum_rows: usize,
    requires_forward_tail: bool,
    maximum_tokens: Option<usize>,
) {
    let inner = session.test_engine_inner();
    let runtime = inner.cost_runtime.as_ref().unwrap();
    assert!(runtime.snapshot().is_some());
    assert!(!inner.slo_controller.lock().has_pending_calibration_work(),
        "the original driver must reap prior publications before an independent witness transaction");
    let before = inner.controller_timing_snapshot().unwrap_or_default();
    let physical_before = executor.physical.load(Ordering::Acquire);
    let mut hint = ferrum_interfaces::BatchHint::simple(maximum_rows);
    if let Some(maximum_tokens) = maximum_tokens {
        // BatchHint::simple otherwise offers 2048 tokens per row. This
        // controlled program's immutable cost domain declares its actual
        // smaller wave capacity; do not ask the planner for larger shapes.
        hint.max_tokens = maximum_tokens;
    }
    let prepared = match inner.prepare_slo_controller(&hint).unwrap() {
        SloIterationPlan::Selected(prepared) => prepared,
        SloIterationPlan::Idle => panic!(
            "original SLO search returned Idle: timing={:?}; route={:?}; resource={:?}",
            inner.controller_timing_snapshot(),
            *executor.cost_route_unknown.lock(),
            *executor.resource_planning_unknown.lock(),
        ),
        SloIterationPlan::Legacy => panic!("Enforce controller used legacy execution"),
        SloIterationPlan::Progressed
        | SloIterationPlan::PrefixMaintenance(_)
        | SloIterationPlan::PrefixSampling(_) => {
            panic!("ordinary request witness selected a prefix maintenance action")
        }
    };
    assert_eq!(executor.physical.load(Ordering::Acquire), physical_before);
    // A Selected result alone could be a safe CompleteRequests fallback.
    // Only the original once-only witness audit proves cost-guided execution.
    inner.execute_slo_controller_wave(prepared).await.unwrap();
    runtime.drain_calibration_fixture();
    let after = inner.controller_timing_snapshot().unwrap();
    assert_eq!(
        after.witnesses.decisions.samples,
        before.witnesses.decisions.samples + 1,
        "{after:?}"
    );
    assert_eq!(
        after.witnesses.backend_submitted.samples,
        before.witnesses.backend_submitted.samples + 1,
        "{after:?}"
    );
    assert_eq!(
        after.witnesses.host_reconciled.samples,
        before.witnesses.host_reconciled.samples + 1,
        "{after:?}"
    );
    assert_eq!(
        after.invalid_clock_transactions,
        before.invalid_clock_transactions
    );
    assert_eq!(
        after.hard_budget_exhaustions,
        before.hard_budget_exhaustions
    );
    if ids.len() <= maximum_rows && !requires_forward_tail {
        assert_eq!(
            after.witnesses.decisions.waves_total,
            before.witnesses.decisions.waves_total + 1,
            "all current token obligations fit the same witnessed wave"
        );
    } else {
        assert_eq!(
            after.witnesses.decisions.nonempty_tail_samples,
            before.witnesses.decisions.nonempty_tail_samples + 1,
            "satisfying uncommitted Prefill or omitted decoder obligations requires a real forward witness: {after:?}"
        );
        assert!(
            after.witnesses.decisions.tail_waves_total
                > before.witnesses.decisions.tail_waves_total
        );
        eprintln!("controlled CPU forward witness: {:?}", after.witnesses);
    }
    assert_eq!(
        executor.physical.load(Ordering::Acquire),
        physical_before + 1
    );
    let submitted = executor.submitted_requests.lock();
    let actual = submitted
        .last()
        .expect("actual submitted request identities");
    assert_eq!(actual.len(), ids.len().min(maximum_rows));
    assert!(actual.iter().all(|id| ids.contains(id)));
}
