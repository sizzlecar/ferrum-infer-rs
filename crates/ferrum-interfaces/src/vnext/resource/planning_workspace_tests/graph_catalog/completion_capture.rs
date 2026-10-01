use super::*;
use crate::model_executor::{ExecutorCompletionPlanningCapture, ExecutorPlanningCapture};

struct Budget(bool);
impl ResourcePlanningBudget for Budget {
    fn has_budget(&mut self) -> bool {
        self.0
    }
}
struct Observer {
    resource: Budget,
    optional: Budget,
    request_optional: bool,
    ready: usize,
    finished: usize,
}
impl Observer {
    fn new() -> Self {
        Self {
            resource: Budget(true),
            optional: Budget(true),
            request_optional: true,
            ready: 0,
            finished: 0,
        }
    }
}
impl ExecutorPlanningCapture for Observer {
    fn resource_budget(&mut self) -> &mut dyn ResourcePlanningBudget {
        &mut self.resource
    }
    fn completion_ready(&mut self, resources: &ResourcePlanningView) -> bool {
        assert_eq!(self.ready, 0, "completion is handed off exactly once");
        assert_eq!(resources.limits().maximum_projected_waves, 1);
        self.ready += 1;
        self.request_optional
    }
    fn forecast_budget(&mut self) -> &mut dyn ResourcePlanningBudget {
        &mut self.optional
    }
    fn forecast_finished(&mut self) {
        assert_eq!(self.ready, 1);
        assert!(self.request_optional);
        assert_eq!(
            self.finished, 0,
            "accepted optional phase finishes exactly once"
        );
        self.finished += 1;
    }
}
fn limits(waves: usize) -> ResourcePlanningLimits {
    ResourcePlanningLimits {
        maximum_projected_waves: waves,
        ..Default::default()
    }
}
fn captured(
    mut capture: impl FnMut() -> ResourcePlanningAvailability<ExecutorCompletionPlanningCapture>,
) -> ExecutorCompletionPlanningCapture {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match capture() {
            ResourcePlanningAvailability::Known(value) => return value,
            ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::ReadUnavailable(_))
                if std::time::Instant::now() < deadline =>
            {
                std::thread::yield_now()
            }
            ResourcePlanningAvailability::Unknown(reason) => {
                panic!("completion capture: {reason:?}")
            }
        }
    }
}

#[test]
fn shared_completion_forecast_uses_one_quiescent_lane_and_preserves_both_wave_limits() {
    let (h, bucket, lane) = setup(256);
    let sequence = admitted_sequence_with_ceiling(&h.root, "completion-shared", 4);
    let session = sequence.open_session().unwrap();
    let allocations = h.runtime.allocate_calls();
    let mut observer = Observer::new();
    let result = captured(|| {
        h.root.completion_planning_capture(
            &[&session],
            &lane,
            limits(1),
            limits(3),
            &mut observer,
            &mut |_| Ok((vec![0], ())),
            &mut |view, (): (), _| Ok(view),
        )
    });
    assert_eq!(observer.ready, 1);
    assert_eq!(observer.finished, usize::from(observer.request_optional));
    let Some(ExecutionCostRouteAvailability::Known(route)) = result.forecast else {
        panic!("same-lane optional route must be available without re-locking its lane");
    };
    assert_eq!(result.resources.limits(), limits(1));
    assert_eq!(route.resource_view().limits(), limits(3));
    assert!(result
        .resources
        .with_projection_limits(limits(3))
        .unwrap()
        .same_live_evidence(route.resource_view()));
    assert_eq!(allocations, h.runtime.allocate_calls());
    assert_eq!(
        route.resource_view().participants()[0].authority(),
        result.resources.participants()[0].authority()
    );
    // A state from completion must be accepted by the forecast view: equal
    // numbers from a second capture would carry a different private fence.
    let planned = known(h.root.project_resource_wave_with_bucket(
        route.resource_view(),
        &result.resources.initial_state(),
        &[row(0, 0, 1)],
        Some(bucket.bucket_id()),
        &mut || true,
    ));
    assert!(
        matches!(
            h.root.project_resource_wave_with_bucket(
                route.resource_view(),
                &planned.state,
                &[row(0, 1, 1)],
                None,
                &mut || true,
            ),
            ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::PhysicalCapacity)
        ),
        "retained physical capacity cannot be allocated a second time"
    );
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let actual = step(&batch, &lane, 1, Some(&bucket));
    assert_eq!(
        planned.selected_step_slot(),
        actual
            .claimed_backing()
            .lane_stable_slot_identity()
            .as_ref()
    );
    actual.try_retire_normal().unwrap();
    assert_eq!(allocations, h.runtime.allocate_calls());
    session.try_abort_if_quiescent().unwrap();
}

#[test]
fn optional_budget_and_policy_failure_keep_original_completion_resources() {
    let (h, _, lane) = setup(256);
    let sequence = admitted_sequence_with_ceiling(&h.root, "completion-optional", 4);
    let session = sequence.open_session().unwrap();
    let mut observer = Observer::new();
    observer.optional.0 = false;
    let result = captured(|| {
        h.root.completion_planning_capture(
            &[&session],
            &lane,
            limits(1),
            limits(3),
            &mut observer,
            &mut |_| panic!("exhausted optional budget cannot start forecast"),
            &mut |view, (): (), _| Ok(view),
        )
    });
    assert_eq!(observer.finished, 1);
    assert!(matches!(
        result.forecast,
        Some(ExecutionCostRouteAvailability::Unknown(
            ExecutionCostRouteUnknown::BudgetExhausted
        ))
    ));
    assert!(result.resources.participants()[0].matches_session_identity(&session));
    let mut observer = Observer::new();
    let result = captured(|| {
        h.root.completion_planning_capture(
            &[&session],
            &lane,
            limits(1),
            limits(3),
            &mut observer,
            &mut |_| Err(ExecutionCostRouteUnknown::Unsupported),
            &mut |view, (): (), _| Ok(view),
        )
    });
    assert_eq!(observer.finished, 1);
    assert!(matches!(
        result.forecast,
        Some(ExecutionCostRouteAvailability::Unknown(
            ExecutionCostRouteUnknown::Unsupported
        ))
    ));
    assert!(result.resources.participants()[0].matches_session_identity(&session));
    session.try_abort_if_quiescent().unwrap();
}

#[test]
fn graph_failure_and_incompatible_limits_cannot_erase_completion_or_widen_capture() {
    let (h, _, lane) = setup(256);
    let other_lane = h.root.create_execution_lane().unwrap();
    let sequence = admitted_sequence_with_ceiling(&h.root, "completion-graph", 4);
    let session = sequence.open_session().unwrap();
    *h.runtime.cost_graph_catalog.lock().unwrap() = Some(catalog(other_lane.id(), 1));
    let mut observer = Observer::new();
    let result = captured(|| {
        h.root.completion_planning_capture(
            &[&session],
            &lane,
            limits(1),
            limits(3),
            &mut observer,
            &mut |_| Ok((vec![0], ())),
            &mut |view, (): (), _| Ok(view),
        )
    });
    assert_eq!(observer.finished, 1);
    assert!(matches!(
        result.forecast,
        Some(ExecutionCostRouteAvailability::Unknown(
            ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::StaleIdentity)
        ))
    ));
    assert_eq!(result.resources.limits(), limits(1));
    *h.runtime.cost_graph_catalog.lock().unwrap() = None;
    let mut observer = Observer::new();
    let different = ResourcePlanningLimits {
        maximum_free_extents: 1,
        ..limits(3)
    };
    let result = captured(|| {
        h.root.completion_planning_capture(
            &[&session],
            &lane,
            limits(1),
            different,
            &mut observer,
            &mut |_| panic!("incompatible limits cannot request forecast metadata"),
            &mut |view, (): (), _| Ok(view),
        )
    });
    assert_eq!(observer.finished, 1);
    assert!(matches!(
        result.forecast,
        Some(ExecutionCostRouteAvailability::Unknown(
            ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::InvalidInput)
        ))
    ));
    assert_eq!(result.resources.limits(), limits(1));
    session.try_abort_if_quiescent().unwrap();
}

#[test]
fn declined_optional_work_never_reads_catalog_and_cancelled_session_never_calls_ready() {
    let (h, _, lane) = setup(256);
    let sequence = admitted_sequence_with_ceiling(&h.root, "completion-declined", 4);
    let session = sequence.open_session().unwrap();
    *h.runtime.cost_graph_probe.lock().unwrap() = Some(Box::new(|| {
        panic!("declined optional capture must not read graph catalog")
    }));
    let mut observer = Observer::new();
    observer.request_optional = false;
    let result = captured(|| {
        h.root.completion_planning_capture(
            &[&session],
            &lane,
            limits(1),
            limits(3),
            &mut observer,
            &mut |_| panic!("no optional metadata"),
            &mut |view, (): (), _| Ok(view),
        )
    });
    assert!(result.forecast.is_none());
    assert_eq!(observer.ready, 1);
    assert_eq!(observer.finished, usize::from(observer.request_optional));
    *h.runtime.cost_graph_probe.lock().unwrap() = None;
    session.try_abort_if_quiescent().unwrap();
    let mut observer = Observer::new();
    assert!(matches!(
        h.root.completion_planning_capture(
            &[&session],
            &lane,
            limits(1),
            limits(3),
            &mut observer,
            &mut |_| Ok((vec![0], ())),
            &mut |view, (): (), _| Ok(view),
        ),
        ResourcePlanningAvailability::Unknown(_)
    ));
    assert_eq!(observer.ready, 0);
    assert_eq!(observer.finished, 0);
}
