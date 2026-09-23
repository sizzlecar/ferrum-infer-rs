use super::*;
use std::num::NonZeroUsize;

fn config(count: usize, tokens: usize, bytes: usize) -> SchedulerConfig {
    let mut config = SchedulerConfig::default();
    config.slo.mode = SloMode::Enforce;
    config.slo.admission.max_waiting_requests = NonZeroUsize::new(count).unwrap();
    config.slo.admission.max_waiting_prompt_tokens = NonZeroUsize::new(tokens).unwrap();
    config.slo.admission.max_waiting_prompt_bytes = NonZeroUsize::new(bytes).unwrap();
    config
}
fn request(text: &str, tokens: u64) -> InferenceRequest {
    InferenceRequest::new(text, "waiting-capacity-test")
        .with_metadata(PROMPT_TOKENS_METADATA_KEY, tokens.into())
}
fn failed(id: RequestId) -> InferenceResponse {
    InferenceResponse {
        request_id: id,
        text: String::new(),
        tokens: Vec::new(),
        finish_reason: ferrum_types::FinishReason::Error,
        usage: ferrum_types::TokenUsage::new(0, 0),
        latency_ms: 0,
        created_at: chrono::Utc::now(),
        metadata: Default::default(),
        api_response: None,
        execution_evidence: None,
    }
}

#[tokio::test]
async fn exact_waiting_bounds_use_token_counts_and_utf8_bytes_independently() {
    for limits in [(2, 10, 100), (10, 4, 100), (10, 100, 6)] {
        let scheduler = ContinuousBatchScheduler::new(config(limits.0, limits.1, limits.2));
        let first = scheduler.submit(request("中", 2)).await.unwrap();
        let second = scheduler.submit(request("文", 2)).await.unwrap();
        let rejected = request("a", 1);
        let id = rejected.id.clone();
        assert!(matches!(
            scheduler.submit(rejected).await,
            Err(FerrumError::ResourceExhausted { .. })
        ));
        assert_eq!(scheduler.waiting_count(), 2);
        assert_eq!(scheduler.trace_phase(&id), None);
        assert!(scheduler.cancel(first).await.unwrap());
        let replacement = scheduler.submit(request("a", 1)).await.unwrap();
        assert!(scheduler.cancel(second).await.unwrap());
        assert!(scheduler.cancel(replacement).await.unwrap());
        assert_eq!(scheduler.waiting_count(), 0);
    }
}

#[tokio::test]
async fn missing_or_invalid_token_counts_never_acquire_waiting_capacity() {
    let scheduler = ContinuousBatchScheduler::new(config(4, 4, 16));
    for count in [
        None,
        Some(serde_json::json!(-1)),
        Some(serde_json::json!(1.5)),
        Some(serde_json::json!("1")),
    ] {
        let mut input = InferenceRequest::new("test", "waiting-capacity-test");
        if let Some(count) = count {
            input
                .metadata
                .insert(PROMPT_TOKENS_METADATA_KEY.into(), count);
        }
        let id = input.id.clone();
        assert!(matches!(
            scheduler.submit(input).await,
            Err(FerrumError::RequestValidation { .. })
        ));
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.trace_phase(&id), None);
    }

    let retained = scheduler.submit(request("x", 1)).await.unwrap();
    let overflowing = request("y", usize::MAX as u64);
    let rejected = overflowing.id.clone();
    assert!(matches!(
        scheduler.submit(overflowing).await,
        Err(FerrumError::ResourceExhausted { .. })
    ));
    assert_eq!(scheduler.waiting_count(), 1);
    assert_eq!(
        scheduler.trace_phase(&retained),
        Some(RequestPhase::Waiting)
    );
    assert_eq!(scheduler.trace_phase(&rejected), None);
    assert!(scheduler.cancel(retained).await.unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simultaneous_submit_cannot_both_acquire_the_last_waiting_slot() {
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config(1, 2, 2)));
    let barrier = Arc::new(tokio::sync::Barrier::new(3));
    let mut tasks = Vec::new();
    for _ in 0..2 {
        let scheduler = scheduler.clone();
        let barrier = barrier.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            scheduler.submit(request("x", 1)).await
        }));
    }
    barrier.wait().await;
    let mut admitted = Vec::new();
    for task in tasks {
        match task.await.unwrap() {
            Ok(id) => admitted.push(id),
            Err(error) => assert!(matches!(error, FerrumError::ResourceExhausted { .. })),
        }
    }
    assert_eq!(admitted.len(), 1);
    assert_eq!(scheduler.waiting_count(), 1);
    assert!(scheduler.cancel(admitted.pop().unwrap()).await.unwrap());
}

#[tokio::test]
async fn start_prefill_and_error_release_waiting_charge_but_requeue_keeps_accepted_work() {
    let scheduler = ContinuousBatchScheduler::new(config(1, 2, 2));
    let first = scheduler.submit(request("a", 2)).await.unwrap();
    assert!(scheduler.promote_to_prefill_with_empty_retry(&first, None));
    assert_eq!(scheduler.waiting_count(), 0);
    let second = scheduler.submit(request("b", 2)).await.unwrap();
    assert!(scheduler.defer_prefill_to_waiting(&first));
    assert_eq!(
        scheduler.waiting_count(),
        2,
        "accepted work must survive a return to a full queue"
    );
    assert_eq!(scheduler.trace_phase(&first), Some(RequestPhase::Waiting));
    assert_eq!(scheduler.trace_phase(&second), Some(RequestPhase::Waiting));
    assert!(matches!(
        scheduler.submit(request("c", 1)).await,
        Err(FerrumError::ResourceExhausted { .. })
    ));
    scheduler
        .complete(first.clone(), &failed(first.clone()))
        .await
        .unwrap();
    assert_eq!(scheduler.trace_phase(&first), None);
    assert_eq!(scheduler.waiting_count(), 1);
    assert!(scheduler.cancel(second).await.unwrap());
    let third = scheduler.submit(request("c", 2)).await.unwrap();
    assert!(scheduler.cancel(third).await.unwrap());
}

#[test]
fn usage_overflow_is_an_error_in_each_dimension_instead_of_wrapping() {
    let one = WaitingPromptUsage {
        requests: 1,
        tokens: 1,
        bytes: 1,
    };
    for maximum in [
        WaitingPromptUsage {
            requests: usize::MAX,
            ..Default::default()
        },
        WaitingPromptUsage {
            tokens: usize::MAX,
            ..Default::default()
        },
        WaitingPromptUsage {
            bytes: usize::MAX,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            maximum.checked_add(one),
            Err(FerrumError::ResourceExhausted { .. })
        ));
    }
}

#[tokio::test]
async fn observe_and_off_keep_existing_queue_policy_without_requiring_token_metadata() {
    for mode in [SloMode::Off, SloMode::Observe] {
        let mut config = config(1, 1, 1);
        config.slo.mode = mode;
        config.max_waiting_requests = 2;
        let scheduler = ContinuousBatchScheduler::new(config);
        let first = scheduler
            .submit(InferenceRequest::new("larger text", "test"))
            .await
            .unwrap();
        let second = scheduler
            .submit(InferenceRequest::new("also larger", "test"))
            .await
            .unwrap();
        assert!(matches!(
            scheduler.submit(request("x", 1)).await,
            Err(FerrumError::Scheduler { .. })
        ));
        assert!(scheduler.cancel(first).await.unwrap());
        assert!(scheduler.cancel(second).await.unwrap());
    }
}
