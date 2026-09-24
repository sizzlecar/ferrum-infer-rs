use super::*;
use ferrum_interfaces::InferenceRequestContext;
use ferrum_types::{ServiceSloConfig, SloLatencyBudgets, SloMode};
use std::num::NonZeroU64;

fn observe_config() -> EngineConfig {
    let mut config = EngineConfig::default();
    config.scheduler.slo.mode = SloMode::Observe;
    config.scheduler.slo.default_service_class = Some("interactive".into());
    config.scheduler.slo.services.push(ServiceSloConfig {
        id: "interactive".into(),
        server_token_commit: SloLatencyBudgets {
            ttft_ms: NonZeroU64::new(500).unwrap(),
            tpot_ms: NonZeroU64::new(50).unwrap(),
            itl_ms: NonZeroU64::new(100).unwrap(),
        },
        attainment: Default::default(),
        client_visible: None,
    });
    config
}

#[tokio::test(start_paused = true)]
async fn slo_wait_timer_records_blocked_ttft_without_scheduler_activity() {
    let engine = test_continuous_engine_with_config(observe_config());
    let ingress = tokio::time::Instant::now().into_std();
    let mut sequence = SequenceState::new(policy_request(), vec![TokenId::new(1)]);
    let id = sequence.request_id.clone();
    sequence.slo = InferenceRequestContext::from_ingress(ingress)
        .resolve_slo(&engine.inner.config.scheduler.slo)
        .unwrap();
    engine.inner.sequences.write().insert(id.clone(), sequence);
    let iterations = engine.inner.iteration_count.load(Ordering::Relaxed);
    let mut wait = Box::pin(engine.inner.wait_for_slo_deadline());
    assert!(futures::poll!(&mut wait).is_pending());
    tokio::time::advance(Duration::from_millis(500)).await;
    assert!(futures::poll!(&mut wait).is_pending());
    assert!(!engine.inner.sequences.read()[&id]
        .slo
        .as_ref()
        .unwrap()
        .violations()
        .any());
    tokio::time::advance(Duration::from_millis(2)).await;
    // Observe remains in the wait. No capacity/admission wake is fabricated.
    assert!(futures::poll!(&mut wait).is_pending());
    assert!(
        engine.inner.sequences.read()[&id]
            .slo
            .as_ref()
            .unwrap()
            .violations()
            .ttft
    );
    assert!(engine.inner.next_slo_violation_wake().is_none());
    assert_eq!(
        engine.inner.iteration_count.load(Ordering::Relaxed),
        iterations
    );
    assert_eq!(engine.inner.scheduler.active_count(), 0);
    assert_eq!(engine.inner.scheduler.waiting_count(), 0);
    drop(wait);
    engine.inner.sequences.write().remove(&id);
    engine.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn slo_wait_timer_tracks_decode_budgets_without_mutating_token_progress() {
    let engine = test_continuous_engine_with_config(observe_config());
    let ingress = tokio::time::Instant::now().into_std();
    let mut sequence = SequenceState::new(policy_request(), vec![TokenId::new(1)]);
    let id = sequence.request_id.clone();
    sequence.sampling_params.max_tokens = 8;
    sequence.slo = InferenceRequestContext::from_ingress(ingress)
        .resolve_slo(&engine.inner.config.scheduler.slo)
        .unwrap();
    sequence.generated_tokens.push(TokenId::new(2));
    sequence
        .slo
        .as_mut()
        .unwrap()
        .record_commit(ingress)
        .unwrap();
    engine.inner.sequences.write().insert(id.clone(), sequence);
    let mut wait = Box::pin(engine.inner.wait_for_slo_deadline());
    assert!(futures::poll!(&mut wait).is_pending());
    tokio::time::advance(Duration::from_millis(52)).await;
    assert!(futures::poll!(&mut wait).is_pending());
    {
        let sequences = engine.inner.sequences.read();
        let timing = sequences[&id].slo.as_ref().unwrap();
        assert!(timing.violations().tpot);
        assert!(!timing.violations().itl);
        assert_eq!(timing.committed_tokens(), 1);
    }
    tokio::time::advance(Duration::from_millis(50)).await;
    assert!(futures::poll!(&mut wait).is_pending());
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        let timing = sequence.slo.as_ref().unwrap();
        assert!(timing.violations().tpot && timing.violations().itl);
        assert_eq!(timing.first_commit(), Some(ingress));
        assert_eq!(timing.last_commit(), Some(ingress));
        assert_eq!(sequence.generated_tokens, [TokenId::new(2)]);
    }
    assert!(engine.inner.next_slo_violation_wake().is_none());
    drop(wait);
    engine.inner.sequences.write().remove(&id);
    engine.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn slo_wait_timer_excludes_completed_token_budgets_and_off_mode() {
    let engine = test_continuous_engine_with_config(observe_config());
    let ingress = tokio::time::Instant::now().into_std();
    let mut sequence = SequenceState::new(policy_request(), vec![TokenId::new(1)]);
    let id = sequence.request_id.clone();
    sequence.sampling_params.max_tokens = 1;
    sequence.slo = InferenceRequestContext::from_ingress(ingress)
        .resolve_slo(&engine.inner.config.scheduler.slo)
        .unwrap();
    sequence.generated_tokens.push(TokenId::new(2));
    sequence
        .slo
        .as_mut()
        .unwrap()
        .record_commit(ingress)
        .unwrap();
    engine.inner.sequences.write().insert(id.clone(), sequence);
    assert!(engine.inner.next_slo_violation_wake().is_none());
    let _ = engine
        .inner
        .observe_slo_waits(|| ingress + Duration::from_secs(10));
    assert!(!engine.inner.sequences.read()[&id]
        .slo
        .as_ref()
        .unwrap()
        .violations()
        .any());
    let sequence = engine.inner.sequences.write().remove(&id).unwrap();
    engine.shutdown().await.unwrap();

    let off = test_continuous_engine_with_config(EngineConfig::default());
    off.inner.sequences.write().insert(id.clone(), sequence);
    let _ = off
        .inner
        .observe_slo_waits(|| panic!("Off must not read the observation clock"));
    assert!(off.inner.next_slo_violation_wake().is_none());
    off.inner.sequences.write().remove(&id);
    off.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn slo_wait_timer_revalidates_after_real_commit_moves_the_obligation() {
    let mut engine = test_continuous_engine_with_config(observe_config());
    // Exercise the typed timer branch without claiming Enforce startup support.
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .mode = SloMode::Enforce;
    let ingress = tokio::time::Instant::now().into_std();
    let mut sequence = SequenceState::new(policy_request(), vec![TokenId::new(1)]);
    sequence.sampling_params.max_tokens = 8;
    let id = sequence.request_id.clone();
    sequence.slo = InferenceRequestContext::from_ingress(ingress)
        .resolve_slo(&engine.inner.config.scheduler.slo)
        .unwrap();
    engine.inner.sequences.write().insert(id.clone(), sequence);
    let mut wait = Box::pin(engine.inner.wait_for_slo_deadline());
    assert!(futures::poll!(&mut wait).is_pending());
    tokio::time::advance(Duration::from_millis(490)).await;
    {
        let mut sequences = engine.inner.sequences.write();
        let sequence = sequences.get_mut(&id).unwrap();
        sequence.generated_tokens.push(TokenId::new(2));
        sequence.record_generated_token_commit();
    }
    tokio::time::advance(Duration::from_millis(12)).await;
    assert!(
        futures::poll!(&mut wait).is_pending(),
        "old TTFT wake must be revoked"
    );
    tokio::time::advance(Duration::from_millis(40)).await;
    assert_eq!(
        futures::poll!(&mut wait),
        std::task::Poll::Ready(ferrum_interfaces::SloTimingBoundary::TokenPrefix)
    );
    drop(wait);
    {
        let mut sequences = engine.inner.sequences.write();
        let sequence = sequences.get_mut(&id).unwrap();
        sequence.generated_tokens.push(TokenId::new(3));
        sequence.record_generated_token_commit_with_decode_postprocess(None);
        let timing = sequence.slo.as_ref().unwrap();
        assert!(timing.is_trusted());
        assert_eq!(timing.committed_tokens(), 2);
        assert!(timing.violations().tpot);
        assert!(!timing.violations().ttft);
    }
    engine.inner.sequences.write().remove(&id);
    engine.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn slo_wait_timer_revokes_stale_wake_after_cancellation_or_last_token() {
    for completed in [false, true] {
        let mut engine = test_continuous_engine_with_config(observe_config());
        Arc::get_mut(&mut engine.inner)
            .unwrap()
            .config
            .scheduler
            .slo
            .mode = SloMode::Enforce;
        let ingress = tokio::time::Instant::now().into_std();
        let mut sequence = SequenceState::new(policy_request(), vec![TokenId::new(1)]);
        sequence.sampling_params.max_tokens = 1;
        let id = sequence.request_id.clone();
        sequence.slo = InferenceRequestContext::from_ingress(ingress)
            .resolve_slo(&engine.inner.config.scheduler.slo)
            .unwrap();
        engine.inner.sequences.write().insert(id.clone(), sequence);
        let mut wait = Box::pin(engine.inner.wait_for_slo_deadline());
        assert!(futures::poll!(&mut wait).is_pending());
        if completed {
            let mut sequences = engine.inner.sequences.write();
            let sequence = sequences.get_mut(&id).unwrap();
            sequence.generated_tokens.push(TokenId::new(2));
            sequence.record_generated_token_commit();
        } else {
            engine.inner.sequences.write().remove(&id);
        }
        tokio::time::advance(Duration::from_millis(502)).await;
        assert!(futures::poll!(&mut wait).is_pending());
        assert!(engine.inner.next_slo_violation_wake().is_none());
        drop(wait);
        engine.inner.sequences.write().remove(&id);
        engine.shutdown().await.unwrap();
    }
}

#[test]
fn slo_runtime_records_commits_without_opt_in_diagnostic_history() {
    let config = observe_config();
    let ingress = Instant::now() - Duration::from_secs(1);
    let mut sequence = SequenceState::new(policy_request(), vec![TokenId::new(1)]);
    sequence.slo = InferenceRequestContext::from_ingress(ingress)
        .resolve_slo(&config.scheduler.slo)
        .unwrap();
    sequence.generated_tokens.push(TokenId::new(2));
    sequence.record_generated_token_commit();
    sequence.generated_tokens.push(TokenId::new(3));
    sequence.record_generated_token_commit_with_decode_postprocess(None);
    let timing = sequence.slo.as_ref().unwrap();
    assert_eq!(timing.ingress(), ingress);
    assert_eq!(timing.committed_tokens(), 2);
    assert!(timing.violations().ttft);
    assert!(timing.last_commit().unwrap() >= timing.first_commit().unwrap());
    assert!(sequence.take_execution_evidence().unwrap().is_none());
}

#[tokio::test]
async fn slo_runtime_stream_retains_ingress_from_before_engine_submission() {
    let engine = test_continuous_engine_with_config(observe_config());
    let ingress = Instant::now() - Duration::from_secs(1);
    let mut request = policy_request();
    request.sampling_params.max_tokens = 2;
    let request_id = request.id.clone();
    let mut stream = engine
        .infer_stream_with_context(request, InferenceRequestContext::from_ingress(ingress))
        .await
        .unwrap();
    {
        let sequences = engine.inner.sequences.read();
        let timing = sequences.get(&request_id).unwrap().slo.as_ref().unwrap();
        assert_eq!(timing.ingress(), ingress);
        assert_eq!(
            timing.first_deadline(),
            ingress + Duration::from_millis(500)
        );
    }
    while let Some(chunk) = futures::StreamExt::next(&mut stream).await {
        chunk.unwrap();
    }
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn slo_runtime_unconfigured_class_is_rejected_before_publication() {
    let engine = test_continuous_engine_with_config(observe_config());
    let context = InferenceRequestContext::capture().with_service_class("missing".into());
    assert!(engine
        .infer_with_context(policy_request(), context.clone())
        .await
        .is_err());
    assert!(engine
        .infer_stream_with_context(policy_request(), context)
        .await
        .is_err());
    assert!(engine.inner.sequences.read().is_empty());
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn slo_runtime_terminal_observation_keeps_commit_text_and_terminal_boundaries_separate() {
    let path = resource_trace_temp_path("slo-terminal-boundaries");
    let mut config = observe_config();
    config.runtime.scheduler_trace_jsonl = Some(path.clone());
    let engine = test_continuous_engine_with_config(config);
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;
    let request_id = request.id.to_string();
    let response = engine
        .infer_with_context(
            request,
            InferenceRequestContext::from_ingress(Instant::now() - Duration::from_secs(1)),
        )
        .await
        .unwrap();
    assert!(
        response.execution_evidence.is_none(),
        "SLO must not enable full token history"
    );
    flush_engine_profile_events(&engine);
    engine.shutdown().await.unwrap();
    let events = read_engine_profile_events(&path);
    let event = events
        .iter()
        .find(|event| event.phase == "engine_slo_terminal")
        .unwrap();
    assert_eq!(event.request_id, request_id);
    assert_eq!(event.status, ProfileStatus::DiagnosticOnly);
    let observation = &event.attributes["observation"];
    assert_eq!(observation["committed_tokens"], response.tokens.len());
    assert!(observation["first_token_commit_ms"].as_f64().unwrap() >= 1_000.0);
    assert_eq!(observation["token_tpot_ms"], serde_json::Value::Null);
    assert_eq!(observation["max_token_itl_ms"], serde_json::Value::Null);
    assert_eq!(observation["internal_timing_pass"], false);
    assert_eq!(observation["ttft_missed"], true);
    assert!(
        observation["engine_terminal_ms"].as_f64().unwrap()
            >= observation["last_token_commit_ms"].as_f64().unwrap()
    );
    assert_eq!(
        event.attributes["text_boundary"],
        "engine_text_prepared_before_channel_send_not_http_or_client_visible"
    );
    std::fs::remove_file(path).unwrap();
}
