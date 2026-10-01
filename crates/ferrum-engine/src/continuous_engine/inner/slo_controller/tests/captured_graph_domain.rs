//! Exercise the production ExecutorState through an actual captured lane view.
//! Numeric catalogs select a validation domain; they never invent a replay.
use super::*;
use ferrum_interfaces::vnext::{
    DeviceCostGraphCatalog, DeviceCostGraphCatalogBuilder, DeviceCostGraphCatalogLimits,
    DeviceCostGraphConfiguration as Configuration, DeviceCostGraphStreamState,
};
use ferrum_scheduler::implementations::continuous::cost_model::WaveGraphState;

fn state(configuration: Configuration) -> DeviceCostGraphStreamState {
    DeviceCostGraphStreamState::new(configuration, 0, 0, 0).unwrap()
}

fn empty_catalog(state: DeviceCostGraphStreamState) -> DeviceCostGraphCatalog {
    DeviceCostGraphCatalogBuilder::new(state, DeviceCostGraphCatalogLimits::new(1, 1, 1).unwrap())
        .unwrap()
        .finish(&mut || Ok(()))
        .unwrap()
}

#[tokio::test]
async fn captured_graph_domain_keeps_unsupported_and_unconfigured_exact() {
    let (engine, _, executor) = fixture().await;
    executor.enable_structured_query_route();
    let (_, session) = prefill::request(&engine, 4, 2).await;
    prefill::admit(&engine, 1).await;
    for stream in [None, Some(state(Configuration::Unconfigured))] {
        executor.set_cost_graph_evidence(stream, None);
        let captured = prefill::captured(&engine, &executor).await;
        assert_eq!(captured.route.graph_stream_state(), stream);
        assert!(captured.route.graph_catalog().is_none());
        assert_eq!(
            captured.snapshot.capabilities.graph_state,
            WaveGraphState::Disabled
        );
        let context = shape::ExecutorShape {
            engine: &engine.inner,
            captured: &captured,
        };
        let parent = context.begin(&captured.snapshot, &mut || Ok(())).unwrap();
        assert_eq!(
            parent.graph_domain(),
            Ok(PlanningGraphDomain::SnapshotExact)
        );
    }
    cleanup(engine, session).await;
}

#[tokio::test]
async fn captured_graph_domain_uses_ready_catalog_without_authorizing_missing_programs() {
    let (engine, _, executor) = fixture().await;
    executor.enable_structured_query_route();
    let (_, session) = prefill::request(&engine, 4, 2).await;
    prefill::admit(&engine, 1).await;
    for (configuration, domain) in [
        (
            Configuration::OnDemand,
            PlanningGraphDomain::ConfiguredPerWave,
        ),
        (
            Configuration::StartupReady,
            PlanningGraphDomain::ResidentReplayOnly,
        ),
    ] {
        let stream = state(configuration);
        executor.set_cost_graph_evidence(Some(stream), Some(empty_catalog(stream)));
        let before = prefill::UnsubmittedState::capture(&engine, &executor);
        let captured = prefill::captured(&engine, &executor).await;
        assert_eq!(captured.route.graph_stream_state(), Some(stream));
        assert_eq!(
            captured.route.graph_catalog().unwrap().stream_state(),
            stream
        );
        assert_eq!(
            captured.snapshot.capabilities.graph_state,
            WaveGraphState::Disabled
        );
        let context = shape::ExecutorShape {
            engine: &engine.inner,
            captured: &captured,
        };
        let parent = context.begin(&captured.snapshot, &mut || Ok(())).unwrap();
        assert_eq!(parent.graph_domain(), Ok(domain));
        let request = &captured.snapshot.requests[0];
        let work = [CandidateWork {
            key: request.key.clone(),
            action: WaveAction::Prefill {
                offset: 0,
                count: NonZeroU32::new(2).unwrap(),
            },
        }];
        let rows = [PlanningShapeRow {
            request,
            work: ActualRowWork::Prefill {
                offset: 0,
                count: 2,
                total_prompt_tokens: 4,
            },
        }];
        let input = PlanningExecutionInput {
            work: &work,
            requests: &captured.snapshot.requests,
            kind: ActualWaveKind::Prefill,
            rows: &rows,
            recurrent_state_bytes: request.recurrent_state_bytes,
        };
        // Empty catalogs carry no uploaded program. The real core must still
        // decline the projection, even though its graph label domain is valid.
        assert!(parent.project(&input, &mut || Ok(())).unwrap().is_none());
        assert_eq!(
            *executor.cost_route_unknown.lock(),
            Some(ferrum_interfaces::vnext::ExecutionCostRouteUnknown::ExecutionPolicy)
        );
        // A new backend setting does not mutate the captured immutable domain.
        executor.set_cost_graph_evidence(None, None);
        assert_eq!(parent.graph_domain(), Ok(domain));
        before.assert_unchanged(&engine, &executor);
    }
    cleanup(engine, session).await;
}

#[tokio::test]
async fn captured_graph_domain_rejects_preparing_and_ready_without_catalog() {
    let (engine, _, executor) = fixture().await;
    executor.enable_structured_query_route();
    let (_, session) = prefill::request(&engine, 4, 2).await;
    prefill::admit(&engine, 1).await;
    for (configuration, with_catalog) in [
        (Configuration::StartupPreparing, false),
        (Configuration::StartupPreparing, true),
        (Configuration::OnDemand, false),
        (Configuration::StartupReady, false),
    ] {
        let stream = state(configuration);
        executor.set_cost_graph_evidence(Some(stream), with_catalog.then(|| empty_catalog(stream)));
        let captured = prefill::captured(&engine, &executor).await;
        let context = shape::ExecutorShape {
            engine: &engine.inner,
            captured: &captured,
        };
        let parent = context.begin(&captured.snapshot, &mut || Ok(())).unwrap();
        assert_eq!(
            parent.graph_domain(),
            Err(PlanningUnknownReason::UnknownResourceEvidence)
        );
    }
    executor.set_cost_graph_evidence(None, None);
    cleanup(engine, session).await;
}
