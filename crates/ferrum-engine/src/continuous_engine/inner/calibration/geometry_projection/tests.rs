use super::*;
use crate::continuous_engine::inner::{
    cost_observation::EngineCostRuntime,
    slo_controller::tests::fixture::{fixture_with_custom_config, ControlledExecutor},
};
use crate::{AutomaticCostProbeOutput, AutomaticCostProbeTemplate};
use ferrum_interfaces::execution_cost::{CostWorkloadLimitsV1, ExecutorCostIdentityAvailability};
use ferrum_types::{
    InferenceRequest, SloAutomaticCostProbeSamplingPresetV1, SloCostObservationConfig,
};
use std::num::{NonZeroU64, NonZeroUsize};
use std::time::Duration;

#[test]
fn prefill_chunk_for_width_separates_row_ceiling_from_wave_capacity() {
    for (whole, row, width, expected) in [
        (2, Some(1), 1, Some(1)),
        (2, Some(1), 2, Some(1)),
        (1, Some(1), 2, None),
        (2, None, 1, Some(2)),
        (2, None, 2, Some(1)),
        (5, Some(3), 2, Some(2)),
        (2, Some(1), 0, None),
    ] {
        let chunk = prefill_chunk_for_width(
            NonZeroU32::new(whole).unwrap(),
            row.and_then(NonZeroU32::new),
            width,
        );
        assert_eq!(chunk.map(NonZeroU32::get), expected);
        if let Some(chunk) = chunk {
            assert!(u64::from(chunk.get()) * u64::try_from(width).unwrap() <= u64::from(whole));
            assert!(row.is_none_or(|ceiling| chunk.get() <= ceiling));
        }
    }
    if usize::BITS > u32::BITS {
        assert!(prefill_chunk_for_width(
            NonZeroU32::new(u32::MAX).unwrap(),
            None,
            usize::try_from(u32::MAX).unwrap().checked_add(1).unwrap(),
        )
        .is_none());
    }
}

pub(in crate::continuous_engine::inner::calibration) async fn fixture(
    width: usize,
) -> (CalibrationSession, Arc<ControlledExecutor>) {
    fixture_with_domain(width, 32, width).await
}

pub(in crate::continuous_engine::inner::calibration) async fn fixture_with_domain(
    width: usize,
    context: usize,
    scheduled_tokens: usize,
) -> (CalibrationSession, Arc<ControlledExecutor>) {
    let (mut engine, _, executor) = fixture_with_custom_config(width, |config| {
        config.scheduler.slo.cost_observation =
            SloCostObservationConfig::structured_whole_wave_v2();
        config.batching.max_num_batched_tokens = scheduled_tokens;
    })
    .await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.bg_loop_spawned.store(false, Ordering::Release);
    inner.prefill_reference_runtime = None;
    let identity = inner.cost_runtime.as_ref().unwrap().identity.clone();
    let ExecutorCostIdentityAvailability::Known(known) = &identity else {
        panic!("CPU fixture identity")
    };
    let domain = CostWorkloadDomainV1::new_vnext(
        known,
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(width as u32).unwrap(),
            maximum_context_tokens: NonZeroU32::new(context as u32).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(scheduled_tokens as u64).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(
                inner.model_executor.info().vocab_size as u64,
            )
            .unwrap(),
            repetition_slot_capacity: 0,
            fixed_state_bytes_per_row: 0,
        },
    )
    .unwrap();
    inner.cost_runtime = Some(Arc::new(
        EngineCostRuntime::new_with_domain(
            identity,
            &inner.config.scheduler.slo.cost_observation,
            None,
            Some(domain),
        )
        .unwrap(),
    ));
    executor.enable_structured_query_route();
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    // Match the existing CPU future-query fixture's real resident Step and
    // whole-plan Invocation capacity. This admits/releases resources without
    // executing a token; pure projection must never grow the live arenas.
    executor.prepare_structured_query_resources();
    let session = CalibrationSession::from_fresh_engine(
        engine,
        CalibrationLimits::new(NonZeroUsize::new(width).unwrap()).unwrap(),
    )
    .unwrap();
    (session, executor)
}

pub(in crate::continuous_engine::inner::calibration) fn requests(
    session: &CalibrationSession,
    width: usize,
) -> Vec<ProbeRequest> {
    let mut request = InferenceRequest::new("v5", session.configuration().model.model_id.clone());
    request.stream = true;
    request.sampling_params.temperature = 1.0; // installed FullLogits CPU route
    request.sampling_params.repetition_penalty = 1.0;
    let template =
        AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText).unwrap();
    (0..width)
        .map(|seed| {
            let (request, contract) = template
                .instantiate(
                    NonZeroUsize::new(4).unwrap(),
                    seed as u64,
                    SloAutomaticCostProbeSamplingPresetV1::Configured,
                )
                .unwrap();
            ProbeRequest { request, contract }
        })
        .collect()
}

pub(in crate::continuous_engine::inner::calibration) fn limits() -> GeometryProjectionLimits {
    GeometryProjectionLimits {
        deadline: Instant::now() + Duration::from_secs(10),
        maximum_projections: 64,
        maximum_route_states: 8,
        maximum_retained_bytes: 4 * 1024 * 1024,
        prefill_chunk: NonZeroU32::new(3).unwrap(),
        prefill_row_ceiling: None,
    }
}

pub(in crate::continuous_engine::inner::calibration) fn assert_unsubmitted_clean(
    session: &CalibrationSession,
    executor: &ControlledExecutor,
) {
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert!(session.frontiers().unwrap().is_empty());
    assert_eq!(session.engine.inner.scheduler.active_count(), 0);
    assert_eq!(session.engine.inner.scheduler.waiting_count(), 0);
    assert!(session.engine.inner.prefill_reference_runtime.is_none());
}

#[tokio::test]
async fn geometry_projects_checked_integer_widths_without_prefill_submission_or_model() {
    let (mut session, executor) = fixture(3).await;
    let probes = requests(&session, 3);
    let points: Vec<_> = (1..=3)
        .map(|rows| GeometryProjectionPoint {
            rows,
            sequence_tokens: 2,
        })
        .collect();
    let report = session
        .project_geometry(probes, &points, limits())
        .await
        .unwrap();
    assert_eq!(report.admitted_requests, 3);
    assert!(report.projection_attempts > 0);
    assert_eq!(report.outcomes.len(), points.len());
    for outcome in &report.outcomes {
        assert_eq!(outcome.unknown, None);
        assert_eq!(outcome.branches.len(), 1);
        let branch = &outcome.branches[0];
        assert_eq!(branch.host_branch, GeometryHostBranch::FullLogits);
        assert_eq!(
            branch.query.input().owner().rows as usize,
            outcome.point.rows
        );
        assert_eq!(
            branch.query.input().numerical_family_key().unwrap(),
            branch.family
        );
    }
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_budget_exhaustion_cancels_all_admitted_roots() {
    let (mut session, executor) = fixture(1).await;
    let probes = requests(&session, 1);
    let mut budget = limits();
    budget.maximum_projections = 1;
    let report = session
        .project_geometry(
            probes,
            &[GeometryProjectionPoint {
                rows: 1,
                sequence_tokens: 2,
            }],
            budget,
        )
        .await
        .unwrap();
    assert_eq!(report.admitted_requests, 1);
    assert_eq!(report.projection_attempts, 1);
    assert_eq!(
        report.outcomes[0].unknown,
        Some(GeometryProjectionUnknown::BudgetExhausted)
    );
    assert!(report.outcomes[0].branches.is_empty());
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_does_not_invent_context_or_output_capacity() {
    let (mut session, executor) = fixture(1).await;
    let probes = requests(&session, 1);
    let points = [
        GeometryProjectionPoint {
            rows: 1,
            sequence_tokens: 1,
        },
        GeometryProjectionPoint {
            rows: 1,
            sequence_tokens: 5,
        },
    ];
    let report = session
        .project_geometry(probes, &points, limits())
        .await
        .unwrap();
    assert_eq!(report.projection_attempts, 0);
    for outcome in report.outcomes {
        assert_eq!(
            outcome.unknown,
            Some(GeometryProjectionUnknown::Unreachable)
        );
    }
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_reuses_cancelled_fresh_roots_without_submitting_work() {
    let (mut session, executor) = fixture(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    let mut previous_ids = Vec::new();
    for _ in 0..2 {
        let probes = requests(&session, 2);
        let ids: Vec<_> = probes
            .iter()
            .map(|probe| probe.request.id.clone())
            .collect();
        assert!(ids.iter().all(|id| !previous_ids.contains(id)));
        let report = session
            .project_geometry(
                probes,
                &[GeometryProjectionPoint {
                    rows: 2,
                    sequence_tokens: 2,
                }],
                limits(),
            )
            .await
            .unwrap();
        assert_eq!(report.admitted_requests, 2);
        assert_eq!(report.outcomes.len(), 1);
        assert_eq!(report.outcomes[0].unknown, None);
        assert_eq!(report.outcomes[0].branches.len(), 1);
        assert_unsubmitted_clean(&session, &executor);
        assert_eq!(executor.completion_calls.load(Ordering::Acquire), 0);
        previous_ids = ids;
    }
    session.shutdown().await.unwrap();
}
