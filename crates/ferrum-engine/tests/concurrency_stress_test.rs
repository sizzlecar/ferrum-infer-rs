//! Concurrency tests: queued requests, mixed lengths, and streaming completion.
//!
//! Uses mock executor with zero latency for fast execution.

use ferrum_engine::{ContinuousBatchEngine, LlmInferenceEngine};
use ferrum_scheduler::implementations::ContinuousBatchScheduler;
use ferrum_testkit::{MockModelExecutor, MockSampler, MockTokenizer};
use ferrum_types::{FinishReason, InferenceRequest, SchedulerConfig};
use std::sync::Arc;

const VOCAB: usize = 1000;

fn make_engine() -> ContinuousBatchEngine {
    let config = ferrum_types::EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let tokenizer = Arc::new(MockTokenizer::new(VOCAB));
    let sampler = Arc::new(MockSampler);
    let executor = MockModelExecutor::instant(VOCAB);

    ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        ferrum_execution::executor_lifecycle::ExecutorOwner::new(Box::new(executor)),
    )
    .expect("legacy engine composition must match executor authority")
}

fn make_request(id: usize, max_tokens: usize) -> InferenceRequest {
    let mut req = InferenceRequest::new(format!("Request {id}"), "mock-model");
    req.sampling_params.max_tokens = max_tokens;
    req.sampling_params.temperature = 0.0;
    req
}

#[tokio::test]
async fn concurrent_32_all_complete() {
    let engine = Arc::new(make_engine());
    let mut handles = Vec::new();

    for i in 0..32 {
        let eng = engine.clone();
        handles.push(tokio::spawn(
            async move { eng.infer(make_request(i, 5)).await },
        ));
    }

    let mut completed = 0;
    for h in handles {
        let resp = h.await.unwrap().unwrap();
        assert_eq!(resp.response().finish_reason, FinishReason::Length);
        completed += 1;
    }
    assert_eq!(completed, 32);
}

#[tokio::test]
async fn concurrent_64_all_complete() {
    let engine = Arc::new(make_engine());
    let mut handles = Vec::new();

    for i in 0..64 {
        let eng = engine.clone();
        handles.push(tokio::spawn(
            async move { eng.infer(make_request(i, 3)).await },
        ));
    }

    for h in handles {
        let resp = h.await.unwrap().expect("queued request must complete");
        assert_eq!(resp.response().finish_reason, FinishReason::Length);
        assert_eq!(resp.response().tokens.len(), 3);
    }
}

#[tokio::test]
async fn concurrent_mixed_lengths() {
    let engine = Arc::new(make_engine());
    let mut handles = Vec::new();

    for i in 0..16 {
        let eng = engine.clone();
        let max_tokens = if i % 2 == 0 { 3 } else { 10 };
        handles.push(tokio::spawn(async move {
            eng.infer(make_request(i, max_tokens)).await
        }));
    }

    for h in handles {
        let resp = h.await.unwrap().unwrap();
        assert!(!resp.response().tokens.is_empty());
    }
}

#[tokio::test]
async fn concurrent_streams_16() {
    use futures::StreamExt;

    let engine = Arc::new(make_engine());
    let mut handles = Vec::new();

    for i in 0..16 {
        let eng = engine.clone();
        handles.push(tokio::spawn(async move {
            let stream = eng.infer_stream(make_request(i, 5)).await.unwrap();
            let chunks: Vec<_> = stream.collect().await;
            chunks.len()
        }));
    }

    for h in handles {
        let chunk_count = h.await.unwrap();
        assert!(chunk_count > 0, "stream should produce chunks");
    }
}
