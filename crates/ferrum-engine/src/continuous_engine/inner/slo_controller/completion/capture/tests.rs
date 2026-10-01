//! Compatibility uses actual queue/frontier/output ownership, not forged keys.
use super::*;
use crate::continuous_engine::inner::slo_controller::tests::{fixture, ready};
use crate::continuous_engine::ContinuousBatchEngine;
use ferrum_interfaces::{
    engine::InferenceEngine, output_flow::OutputProjectionContract, InferenceRequestContext,
};
use std::time::Duration;

async fn request(
    engine: &ContinuousBatchEngine,
) -> ferrum_interfaces::output_flow::CreditedOutputSession {
    let mut request = ferrum_types::InferenceRequest::new(
        "test test",
        engine.inner.config.model.model_id.clone(),
    );
    request.stream = true;
    request.sampling_params.max_tokens = 4;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    let id = request.id.clone();
    let session = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    ready(engine, &id).await;
    session
}

fn draft(engine: &ContinuousBatchEngine, width: usize) -> CompletionDraft {
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    loop {
        let budget = ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap();
        match engine
            .inner
            .capture_completion_draft(&ferrum_interfaces::BatchHint::simple(width), &budget)
        {
            Ok(draft) => return draft,
            Err(error)
                if matches!(
                    error.retry,
                    Some(retry::ControllerRetryReason::SnapshotBusy)
                ) && std::time::Instant::now() < deadline =>
            {
                std::thread::yield_now()
            }
            Err(error) => panic!("real completion draft: {error:?}"),
        }
    }
}

fn attach_original_route(engine: &ContinuousBatchEngine, draft: &mut CompletionDraft) {
    let requests = draft
        .selection
        .proof
        .fences
        .iter()
        .map(|f| ExecutorResourcePlanningRequest {
            request_id: &f.key.request_id,
            cache_id: f.resource_cache_id(),
        })
        .collect::<Vec<_>>();
    let route = engine.inner.model_executor.execution_cost_route_view(
        &requests,
        draft.forecast_limits,
        &mut || true,
    );
    assert!(matches!(route, ExecutionCostRouteAvailability::Known(_)));
    draft.forecast = Some(route);
}

#[tokio::test(start_paused = true)]
async fn original_matching_capture_is_consumed_once_and_changed_frontier_or_limits_are_not_shared()
{
    let (engine, _, _) = fixture::fixture_with_width(1).await;
    let session = request(&engine).await;
    assert!(engine.inner.prepare_slo_admission_turn(1).await.unwrap());
    let mut original = draft(&engine, 1);
    let mut current = draft(&engine, 1);
    attach_original_route(&engine, &mut original);
    let limits = original.forecast_limits;
    current.selection.proof.fences[0].generation += 1;
    assert!(original
        .take_compatible_forecast(
            &current.selection.proof.queue,
            &current.selection.proof.fences,
            limits
        )
        .is_none());
    assert!(original.forecast.is_some());
    current.selection.proof.fences[0].generation -= 1;
    let altered = ResourcePlanningLimits {
        maximum_participants: limits.maximum_participants - 1,
        ..limits
    };
    assert!(original
        .take_compatible_forecast(
            &current.selection.proof.queue,
            &current.selection.proof.fences,
            altered
        )
        .is_none());
    assert!(matches!(
        original.take_compatible_forecast(
            &current.selection.proof.queue,
            &current.selection.proof.fences,
            limits
        ),
        Some(ExecutionCostRouteAvailability::Known(_))
    ));
    assert!(original
        .take_compatible_forecast(
            &current.selection.proof.queue,
            &current.selection.proof.fences,
            limits
        )
        .is_none());
    drop(original);
    drop(current);
    fixture::cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn runnable_subset_and_reordered_owners_do_not_replace_complete_snapshot_evidence() {
    let (engine, _, _) = fixture::fixture_with_width(2).await;
    let first = request(&engine).await;
    let second = request(&engine).await;
    assert!(engine.inner.prepare_slo_admission_turn(2).await.unwrap());
    let mut subset = draft(&engine, 1);
    let mut complete = draft(&engine, 2);
    assert_eq!(subset.selection.proof.fences.len(), 1);
    assert_eq!(complete.selection.proof.fences.len(), 2);
    attach_original_route(&engine, &mut subset);
    let limits = subset.forecast_limits;
    assert!(subset
        .take_compatible_forecast(
            &complete.selection.proof.queue,
            &complete.selection.proof.fences,
            limits
        )
        .is_none());
    assert!(subset.forecast.is_some());
    let mut same = draft(&engine, 2);
    attach_original_route(&engine, &mut same);
    complete.selection.proof.fences.swap(0, 1);
    assert!(same
        .take_compatible_forecast(
            &complete.selection.proof.queue,
            &complete.selection.proof.fences,
            limits
        )
        .is_none());
    assert!(same.forecast.is_some());
    drop(subset);
    drop(complete);
    drop(same);
    drop(second);
    fixture::cleanup(engine, first).await;
}

#[tokio::test(start_paused = true)]
async fn absent_cost_model_keeps_completion_and_optional_deadline_cannot_be_restarted() {
    let (mut engine, _, _) = fixture::fixture_with_width(1).await;
    let previous = engine.inner.cost_runtime.as_ref().unwrap();
    let empty = crate::continuous_engine::inner::cost_observation::EngineCostRuntime::with_clock(
        previous.identity.clone(),
        Arc::clone(&previous.clock),
    )
    .unwrap();
    // No published cost model; retain the real owner/frontier allocator.
    Arc::get_mut(&mut engine.inner).unwrap().cost_runtime = Some(Arc::new(empty));
    assert!(engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    let session = request(&engine).await;
    assert!(engine.inner.prepare_slo_admission_turn(1).await.unwrap());
    let original = draft(&engine, 1);
    assert!(original.forecast.is_none());
    assert!(!original.selection.work.participants().is_empty());
    let started = slo_clock_now();
    let budget = ControllerBudget::new(started, Duration::from_millis(2)).unwrap();
    let mut state = DraftCapture {
        started,
        reserve_percent: 20,
        forecast_limits: original.forecast_limits,
        forecast_enabled: true,
        preparation_wall: None,
        optional_started: None,
        optional_elapsed: None,
        clock_invalid: false,
        phase: None,
        forecast: None,
    };
    let mut observer = CaptureObserver {
        state: &mut state,
        budget: Arc::clone(&budget),
        rows: Some(
            original
                .selection
                .work
                .participants()
                .iter()
                .map(|row| row.selection().clone())
                .collect(),
        ),
        kind: original.selection.work.kind(),
        work: None,
    };
    assert!(observer.completion_ready(&original.selection.resources));
    let first_elapsed = observer.state.preparation_wall;
    tokio::time::advance(Duration::from_micros(1_600)).await;
    assert!(!observer.forecast_budget().has_budget());
    observer.forecast_finished();
    assert_eq!(
        observer.state.optional_elapsed,
        Some(Duration::from_micros(1_600))
    );
    assert!(!observer.state.clock_invalid);
    assert_eq!(
        observer.state.completed_preparation_wall(slo_clock_now()),
        Some(Duration::ZERO),
        "the accepted optional interval is excluded from mandatory preparation"
    );
    tokio::time::advance(Duration::from_micros(100)).await;
    assert_eq!(
        observer.state.completed_preparation_wall(slo_clock_now()),
        Some(Duration::from_micros(100)),
        "teardown after optional completion is included"
    );
    assert!(!observer.completion_ready(&original.selection.resources));
    assert_eq!(observer.state.preparation_wall, first_elapsed);
    assert!(!observer.forecast_budget().has_budget());
    assert!(
        budget.poll(),
        "the unchanged hard budget retains completion publication time"
    );
    drop(observer);
    drop(original);
    fixture::cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn default_provider_pairs_accepted_optional_work_without_inventing_a_forecast() {
    let (engine, _, _) = fixture::fixture_with_width(1).await;
    let session = request(&engine).await;
    assert!(engine.inner.prepare_slo_admission_turn(1).await.unwrap());
    let original = draft(&engine, 1);
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap();
    let mut state = DraftCapture {
        started: slo_clock_now(),
        reserve_percent: 20,
        forecast_limits: original.forecast_limits,
        forecast_enabled: true,
        preparation_wall: None,
        optional_started: None,
        optional_elapsed: None,
        clock_invalid: false,
        phase: None,
        forecast: None,
    };
    let mut observer = CaptureObserver {
        state: &mut state,
        budget: Arc::clone(&budget),
        rows: Some(
            original
                .selection
                .work
                .participants()
                .iter()
                .map(|row| row.selection().clone())
                .collect(),
        ),
        kind: original.selection.work.kind(),
        work: None,
    };
    let requests = original
        .selection
        .proof
        .fences
        .iter()
        .map(|fence| ExecutorResourcePlanningRequest {
            request_id: &fence.key.request_id,
            cache_id: fence.resource_cache_id(),
        })
        .collect::<Vec<_>>();
    // ControlledExecutor intentionally uses the ModelExecutor default here.
    let captured = engine
        .inner
        .model_executor
        .execution_completion_planning_capture(
            &requests,
            original.selection.resources.limits(),
            original.forecast_limits,
            &mut observer,
        );
    let ResourcePlanningAvailability::Known(captured) = captured else {
        panic!("default provider must preserve its original real resources");
    };
    assert!(captured.forecast.is_none());
    assert!(observer.state.optional_started.is_none());
    assert_eq!(observer.state.optional_elapsed, Some(Duration::ZERO));
    assert!(!observer.state.clock_invalid);
    assert!(observer.work.as_ref().is_some_and(Result::is_ok));
    drop(observer);
    drop(requests);
    drop(original);
    fixture::cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn completed_preparation_only_tightens_the_original_optional_deadline() {
    let started = slo_clock_now();
    let budget = ControllerBudget::new(started, Duration::from_millis(2)).unwrap();
    let mut phase = budget.completion_optional_phase(Duration::from_micros(200), 20);
    tokio::time::advance(Duration::from_micros(100)).await;
    phase.tighten_for_completion_preparation(Duration::from_micros(600), 20);
    // A repeated smaller report cannot undo the already reserved 600us.
    phase.tighten_for_completion_preparation(Duration::from_micros(50), 20);
    tokio::time::advance(Duration::from_micros(1_299)).await;
    assert!(phase.poll());
    tokio::time::advance(Duration::from_micros(1)).await;
    assert!(!phase.poll());
    assert!(budget.poll());
    tokio::time::advance(Duration::from_micros(600)).await;
    assert!(!budget.poll(), "the hard deadline was never reset");
}

#[tokio::test(start_paused = true)]
async fn changed_sampling_policy_or_replaced_captured_inputs_do_not_share_forecast() {
    use ferrum_interfaces::model_executor::{
        GreedyRepetitionPenalty, LogitsReturnPolicy, TokenSelectionMask,
    };
    let (engine, _, _) = fixture::fixture_with_width(1).await;
    let session = request(&engine).await;
    assert!(engine.inner.prepare_slo_admission_turn(1).await.unwrap());
    let mut original = draft(&engine, 1);
    let mut current = draft(&engine, 1);
    attach_original_route(&engine, &mut original);
    let limits = original.forecast_limits;
    let mask = TokenSelectionMask::new(vec![1, 0, 1]);
    let repetition = GreedyRepetitionPenalty::new(1.2, vec![1, 2]);
    let captured = LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask.clone()),
        repetition_penalty: Some(repetition.clone()),
    };
    original.selection.proof.fences[0].logits_policy = captured.clone();
    let replacements = [
        LogitsReturnPolicy::FullLogits,
        LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: None,
        },
        LogitsReturnPolicy::GreedyArgmax {
            token_mask: Some(TokenSelectionMask::new(vec![1, 0, 1])),
            repetition_penalty: Some(repetition.clone()),
        },
        LogitsReturnPolicy::GreedyArgmax {
            token_mask: Some(mask.clone()),
            repetition_penalty: Some(GreedyRepetitionPenalty::new(1.2, vec![1, 2])),
        },
        LogitsReturnPolicy::GreedyArgmax {
            token_mask: Some(mask),
            repetition_penalty: Some(GreedyRepetitionPenalty::new(
                f32::from_bits(1.2_f32.to_bits() + 1),
                vec![1, 2],
            )),
        },
    ];
    for policy in replacements {
        current.selection.proof.fences[0].logits_policy = policy;
        assert!(original
            .take_compatible_forecast(
                &current.selection.proof.queue,
                &current.selection.proof.fences,
                limits,
            )
            .is_none());
        assert!(original.forecast.is_some());
    }
    current.selection.proof.fences[0].logits_policy = captured;
    assert!(matches!(
        original.take_compatible_forecast(
            &current.selection.proof.queue,
            &current.selection.proof.fences,
            limits,
        ),
        Some(ExecutionCostRouteAvailability::Known(_))
    ));
    drop(original);
    drop(current);
    fixture::cleanup(engine, session).await;
}
