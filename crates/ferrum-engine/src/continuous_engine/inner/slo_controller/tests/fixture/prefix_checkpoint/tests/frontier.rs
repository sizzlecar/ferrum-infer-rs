//! The no-cache-ID prefill route must still start at actual completed work.
use super::*;

fn project(
    executor: &ControlledExecutor,
    route: &ExecutionCostRouteView,
    offset: u32,
) -> ExecutionCostRouteAvailability<vnext::ExecutionCostRouteProjection> {
    executor.project_execution_cost_wave(
        route,
        &route.initial_state(),
        &vnext::FutureWaveCostQuery {
            kind: ActualWaveKind::Prefill,
            rows: &[vnext::FutureWaveCostRow {
                participant_index: 0,
                work: ActualRowWork::Prefill {
                    offset,
                    count: 2,
                    total_prompt_tokens: 4,
                },
                host_policy_signature: [0; 32],
                host_features: None,
                output: vnext::FutureCostOutput::Prefill {
                    final_logits: offset + 2 == 4,
                },
            }],
        },
        &mut || true,
    )
}

#[tokio::test]
async fn native_prefix_cpu_partial_prefill_without_cache_id_uses_completed_frontier() {
    let (_, executor) = startup_checkpoint_components(2).await;
    executor.prepare_structured_query_resources();
    let source = input();
    admit(&executor, &source);
    let fresh = view(&executor, &[(&source.request_id, None)]);
    let before = executor.native_structured_counts();
    known(project(&executor, &fresh, 0));
    assert!(matches!(
        project(&executor, &fresh, 2),
        ExecutionCostRouteAvailability::Unknown(vnext::ExecutionCostRouteUnknown::StaleView)
    ));
    assert_eq!(executor.native_structured_counts(), before);

    let output = partial(&executor, &source);
    assert_eq!(output.output().kv_cache().num_tokens(), 2);
    assert_eq!(
        executor.native_structured_history.lock()[&source.request_id].len(),
        4
    );
    let route = view(&executor, &[(&source.request_id, None)]);
    assert_eq!(
        route.resource_view().participants()[0]
            .completed_checkpoint_boundary()
            .unwrap()
            .completed_tokens(),
        2
    );
    let after = executor.native_structured_counts();
    known(project(&executor, &route, 2));
    assert!(matches!(
        project(&executor, &route, 0),
        ExecutionCostRouteAvailability::Unknown(vnext::ExecutionCostRouteUnknown::StaleView)
    ));
    assert_eq!(executor.native_structured_counts(), after);

    let untouched = input();
    admit(&executor, &untouched);
    let other = view(&executor, &[(&untouched.request_id, None)]);
    known(project(&executor, &other, 0));
    assert_eq!(executor.native_structured_counts(), after);

    // A missing retained frontier after a successful native frame is not a
    // new request. Do not guess zero from the absent decode cache ID.
    drop(output);
    assert!(matches!(
        executor.execution_cost_route_view(
            &[ExecutorResourcePlanningRequest {
                request_id: &source.request_id,
                cache_id: None,
            }],
            ResourcePlanningLimits::default(),
            &mut || true,
        ),
        ExecutionCostRouteAvailability::Unknown(vnext::ExecutionCostRouteUnknown::StaleView)
    ));
}
