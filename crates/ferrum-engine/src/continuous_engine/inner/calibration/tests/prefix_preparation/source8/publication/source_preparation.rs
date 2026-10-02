//! Real source preparation failure before any numerical collector opens.
use super::*;
use crate::continuous_engine::inner::calibration::prepared_owner::{
    PreparedProbeInputs, StartupSourcePreparation,
};
use crate::{AutomaticCostProbeOutput, AutomaticCostProbeTemplate};
use ferrum_interfaces::model_executor::{
    ExecutorPrefillAdmission, ExecutorPrefillAdmissionDecision, PrefixCapturePurpose,
    PrefixCaptureRequest, PrefixCaptureStatus,
};
use ferrum_interfaces::ModelExecutor;

fn assert_no_source_owners(session: &CalibrationSession) {
    assert!(session.prepared_owner_capture.is_none());
    assert!(!session.prefix_source8, "no numerical collector has opened");
    assert!(!session.startup_inventory_active());
    assert!(session.pending.is_none());
    assert!(session.frontiers().unwrap().is_empty());
    assert!(session.engine.inner.sequences.read().is_empty());
    assert_eq!(session.engine.inner.scheduler.active_count(), 0);
    assert_eq!(session.engine.inner.scheduler.waiting_count(), 0);
    assert!(session
        .engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
}

#[tokio::test]
async fn source8_native_interest_capacity_before_collection_keeps_spent_seed_and_next_source() {
    let (mut session, executor) = automatic_checkpoint_session(2).await;
    {
        let inner = Arc::get_mut(&mut session.engine.inner).unwrap();
        inner.config.scheduler.slo.planner.max_planning_us = NonZeroU64::new(1_000_000).unwrap();
        inner.config.scheduler.slo.planner.lookahead_waves = NonZeroUsize::new(4).unwrap();
        inner.prefill_reference_runtime = Some(
            crate::continuous_engine::inner::prefill_reference_runtime::test_piecewise_calibration_runtime(),
        );
    }
    executor.enable_structured_query_route();
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);
    let mut request = request(&session, 3);
    request.prompt = "test test test".into();
    if let Some(ferrum_types::ApiRequest::Completion(api)) = &mut request.api_request {
        api.prompt = request.prompt.clone();
    }
    request.sampling_params.top_k = Some(executor.info().vocab_size);
    request.sampling_params.stop_sequences.clear();
    let template =
        AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText).unwrap();
    let settings = ferrum_types::SloAutomaticCalibrationSettingsV1::default();
    let inputs = Box::pin(PreparedProbeInputs::new(
        &mut session,
        &settings,
        &[template],
    ))
    .await
    .unwrap();
    let mut budget = ProbeExecutionBudget::new_with_input_projection_limit(
        tokio::time::Instant::now()
            + Duration::from_millis(settings.cost_probe.maximum_duration_ms.get()),
        settings.cost_probe.maximum_probe_requests,
        settings.cost_probe.maximum_offered_waves,
        settings.cost_probe.maximum_input_projection_requests,
    );
    let deadline = budget.deadline();
    let mut cursor = inputs.into_cursor().unwrap();
    session
        .begin_startup_owner_series(cursor.maximum_sources(), deadline)
        .unwrap();
    session.begin_startup_inventory().unwrap();
    let series = Box::pin(cursor.next(&mut session, &mut budget))
        .await
        .unwrap()
        .unwrap();
    session.drain_startup_geometry().await.unwrap();
    session.end_startup_inventory().await.unwrap();
    assert_no_source_owners(&session);
    let native_sources: Vec<_> = (0..series.len())
        .filter(|&index| !series.source(index).unwrap().acquisitions().is_empty())
        .collect();
    let mut native_sources = native_sources.into_iter();
    let first = native_sources.next().expect("a declared native source");
    let next = native_sources
        .next()
        .expect("an independently reserved native successor");
    let failed_setup = series.source(first).unwrap().acquisitions()[0]
        .plan()
        .setup_work();

    // An independently admitted native owner supplies real interest pressure.
    // No interest runs a transfer or enters the engine's numerical collector.
    // Stop at the fixture's actual denial, rather than assuming its slot count.
    let blocker = RequestId::new();
    let tokens = vec![ferrum_types::TokenId::new(5); executor.capabilities().max_sequence_length];
    assert!(matches!(
        executor
            .try_admit_prefill(ExecutorPrefillAdmission::for_diagnostic(
                &blocker,
                &tokens,
                tokens.len()
            ))
            .unwrap(),
        ExecutorPrefillAdmissionDecision::Admitted(_)
    ));
    let mut interests = Vec::new();
    let mut denied = false;
    for boundary in 1..tokens.len() {
        match executor
            .retain_prefix_capture_interest(PrefixCaptureRequest {
                purpose: PrefixCapturePurpose::PrivateCalibration,
                source_request_id: &blocker,
                source_tokens: &tokens,
                maximum_sequence_tokens: tokens.len(),
                boundary,
                expires_at: deadline.into_std(),
            })
            .unwrap()
        {
            Some(lease) => {
                assert_eq!(lease.purpose(), PrefixCapturePurpose::PrivateCalibration);
                assert_eq!(lease.status(), PrefixCaptureStatus::Pending);
                interests.push(lease);
            }
            None => {
                denied = true;
                break;
            }
        }
    }
    assert!(
        denied && !interests.is_empty(),
        "the existing interest capacity must actually be full"
    );
    assert_eq!(
        executor.native_prefix_live_lease_counts(),
        (interests.len(), 0)
    );
    let original_requests = budget.requests_remaining();
    let original_actions = budget.attempts_remaining();
    let reserved_requests = budget.selection_requests_remaining();
    let reserved_actions = budget.selection_attempts_remaining();
    let physical = executor.physical.load(Ordering::Acquire);
    let transfers = executor.native_prefix_terminal_totals();
    let mut extra_rows = series.cold_offer_row_headroom().unwrap();
    let original_extra_rows = extra_rows;
    match session
        .prepare_startup_source(&series, first, &mut budget, &mut extra_rows)
        .await
        .unwrap()
    {
        StartupSourcePreparation::Ready { source, acquired } => {
            assert!(
                acquired.is_empty(),
                "capacity failure may only authorize rebuilt cold work"
            );
            assert!(source.acquisitions().is_empty());
            let (declaration, _, execution) = source.into_parts().unwrap();
            assert!(declaration.native_prefix_acquisition.is_none());
            for cohort in execution.cohorts() {
                assert_eq!(execution.acquisition_key(cohort).unwrap(), None);
                assert!(execution.requests_for(cohort).is_ok());
            }
        }
        StartupSourcePreparation::Skipped => {}
        StartupSourcePreparation::Interrupted { error, .. } => {
            panic!("capacity fallback failed to retire its seed: {error}")
        }
    }
    // Interest is armed after the real boundary Prefills. The denied capture
    // was never submitted, but those original owners/actions are not refunded.
    assert_eq!(
        original_requests - budget.requests_remaining(),
        failed_setup.requests
    );
    assert_eq!(
        original_actions - budget.attempts_remaining(),
        failed_setup.inference_waves
    );
    assert_eq!(
        executor.physical.load(Ordering::Acquire) - physical,
        failed_setup.inference_waves
    );
    assert_eq!(executor.native_prefix_terminal_totals(), transfers);
    assert_eq!(budget.selection_requests_remaining(), reserved_requests);
    assert!(budget.selection_attempts_remaining() <= reserved_actions);
    assert!(extra_rows <= original_extra_rows);
    assert_eq!(budget.deadline(), deadline);
    assert_no_source_owners(&session);
    session.completed_owner_boundary().unwrap();

    assert!(executor.cancel_prefill_admission(&blocker));
    drop(interests);
    assert_eq!(executor.native_prefix_live_lease_counts(), (0, 0));
    let requests_after_failure = budget.requests_remaining();
    let actions_after_failure = budget.attempts_remaining();
    let expected_setup = series
        .source(next)
        .unwrap()
        .acquisitions()
        .iter()
        .map(|key| key.plan().setup_work())
        .fold((0, 0), |(requests, actions), work| {
            (requests + work.requests, actions + work.actions().unwrap())
        });
    let StartupSourcePreparation::Ready { source, acquired } = session
        .prepare_startup_source(&series, next, &mut budget, &mut extra_rows)
        .await
        .unwrap()
    else {
        panic!("released interest capacity must permit the original successor");
    };
    assert!(!acquired.is_empty());
    assert!(acquired.iter().all(|prefix| prefix.ready()));
    source.ensure_acquisitions_bound().unwrap();
    assert_eq!(
        requests_after_failure - budget.requests_remaining(),
        expected_setup.0
    );
    assert_eq!(
        actions_after_failure - budget.attempts_remaining(),
        expected_setup.1
    );
    assert_eq!(budget.selection_requests_remaining(), reserved_requests);
    assert_eq!(budget.deadline(), deadline);
    assert_no_source_owners(&session);
    drop(source);
    drop(acquired);
    assert_eq!(executor.native_prefix_live_lease_counts(), (0, 0));
    session.finish_startup_owner_series().unwrap();
    session.shutdown().await.unwrap();
}
