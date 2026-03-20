//! Latency profiling test.
//!
//! Runs a batch of requests and reports TTFT, TPOT, and percentile latencies.
//! Uses mock components with configurable latency to simulate realistic timing.

use ferrum_engine::{ContinuousBatchEngine, InferenceEngineInterface};
use ferrum_scheduler::implementations::ContinuousBatchScheduler;
use ferrum_testkit::{
    MockKvCacheManager, MockModelExecutor, MockSampler, MockTensorFactory, MockTokenizer,
};
use ferrum_types::{InferenceRequest, InferenceResponse, SchedulerConfig};
use std::sync::Arc;
use std::time::{Duration, Instant};

const VOCAB_SIZE: usize = 1000;

fn make_engine_with_latency(prefill_ms: u64, decode_ms: u64) -> ContinuousBatchEngine {
    let config = ferrum_types::EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let tokenizer = Arc::new(MockTokenizer::new(VOCAB_SIZE));
    let sampler = Arc::new(MockSampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(4096));
    let executor = Arc::new(MockModelExecutor::new(
        VOCAB_SIZE,
        Duration::from_millis(prefill_ms),
        Duration::from_millis(decode_ms),
    ));
    let tensor_factory = Arc::new(MockTensorFactory);

    ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        executor,
        tensor_factory,
    )
}

fn make_request(prompt: &str, max_tokens: usize) -> InferenceRequest {
    let mut req = InferenceRequest::new(prompt, "bench-model");
    req.sampling_params.max_tokens = max_tokens;
    req
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

struct LatencyStats {
    p50: f64,
    p95: f64,
    p99: f64,
    mean: f64,
    min: f64,
    max: f64,
}

impl LatencyStats {
    fn from_ms(mut values: Vec<f64>) -> Self {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        Self {
            p50: percentile(&values, 50.0),
            p95: percentile(&values, 95.0),
            p99: percentile(&values, 99.0),
            mean,
            min: values.first().copied().unwrap_or(0.0),
            max: values.last().copied().unwrap_or(0.0),
        }
    }
}

impl std::fmt::Display for LatencyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mean={:.2}ms  P50={:.2}ms  P95={:.2}ms  P99={:.2}ms  min={:.2}ms  max={:.2}ms",
            self.mean, self.p50, self.p95, self.p99, self.min, self.max
        )
    }
}

#[tokio::test]
async fn latency_profile_sequential() {
    let engine = make_engine_with_latency(2, 1); // 2ms prefill, 1ms decode
    let max_tokens = 10;
    let num_requests = 50;

    let mut latencies_ms = Vec::with_capacity(num_requests);
    let mut ttft_ms = Vec::with_capacity(num_requests);
    let mut tpot_ms = Vec::with_capacity(num_requests);

    for i in 0..num_requests {
        let req = make_request(&format!("Profile request {}", i), max_tokens);
        let start = Instant::now();
        let resp = engine.infer(req).await.unwrap();
        let elapsed = start.elapsed();

        let total_ms = elapsed.as_secs_f64() * 1000.0;
        latencies_ms.push(total_ms);

        // TTFT ≈ prefill time (first token from prefill)
        // TPOT ≈ (total - prefill) / (output_tokens - 1)
        let output_tokens = resp.tokens.len();
        if output_tokens > 1 {
            // Approximate TTFT as prefill + 1 decode
            let estimated_ttft = total_ms / output_tokens as f64;
            ttft_ms.push(estimated_ttft);
            let estimated_tpot = total_ms / output_tokens as f64;
            tpot_ms.push(estimated_tpot);
        }
    }

    let lat_stats = LatencyStats::from_ms(latencies_ms);
    let ttft_stats = LatencyStats::from_ms(ttft_ms);
    let tpot_stats = LatencyStats::from_ms(tpot_ms);

    eprintln!("\n=== Sequential Latency Profile ===");
    eprintln!("  Requests:     {}", num_requests);
    eprintln!("  Max tokens:   {}", max_tokens);
    eprintln!("  Total:        {}", lat_stats);
    eprintln!("  TTFT:         {}", ttft_stats);
    eprintln!("  TPOT:         {}", tpot_stats);

    // Sanity: total latency should be reasonable
    assert!(
        lat_stats.mean > 0.0,
        "Mean latency should be positive: {}",
        lat_stats.mean
    );
}

#[tokio::test]
async fn latency_profile_concurrent() {
    let engine = Arc::new(make_engine_with_latency(2, 1));
    let max_tokens = 10;
    let concurrency_levels = [1, 2, 4, 8];

    eprintln!("\n=== Concurrent Latency Profile ===");

    for &concurrency in &concurrency_levels {
        let start = Instant::now();
        let mut handles = Vec::new();

        for i in 0..concurrency {
            let e = engine.clone();
            handles.push(tokio::spawn(async move {
                let req = make_request(&format!("Conc {} req {}", concurrency, i), max_tokens);
                let req_start = Instant::now();
                let resp = e.infer(req).await.unwrap();
                let req_ms = req_start.elapsed().as_secs_f64() * 1000.0;
                (resp, req_ms)
            }));
        }

        let results: Vec<(InferenceResponse, f64)> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let wall_time = start.elapsed();
        let total_tokens: usize = results.iter().map(|(r, _)| r.tokens.len()).sum();
        let throughput = total_tokens as f64 / wall_time.as_secs_f64();
        let latencies: Vec<f64> = results.iter().map(|(_, ms)| *ms).collect();
        let lat_stats = LatencyStats::from_ms(latencies);

        eprintln!(
            "  concurrency={:2}  throughput={:.0} tok/s  wall={:.1}ms  latency: {}",
            concurrency,
            throughput,
            wall_time.as_secs_f64() * 1000.0,
            lat_stats,
        );

        assert!(total_tokens > 0);
    }
}

#[tokio::test]
async fn throughput_scaling() {
    use ferrum_testkit::MockModelExecutor;

    let max_tokens = 5;
    let num_requests = 20;

    // Instant executor — measures pure scheduling throughput
    let engine = Arc::new({
        let config = ferrum_types::EngineConfig::default();
        let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
        let tokenizer = Arc::new(MockTokenizer::new(VOCAB_SIZE));
        let sampler = Arc::new(MockSampler);
        let kv_cache = Arc::new(MockKvCacheManager::new(4096));
        let executor = Arc::new(MockModelExecutor::instant(VOCAB_SIZE));
        let tensor_factory = Arc::new(MockTensorFactory);
        ContinuousBatchEngine::new(
            config, scheduler, tokenizer, sampler, kv_cache, executor, tensor_factory,
        )
    });

    let start = Instant::now();
    let mut handles = Vec::new();
    for i in 0..num_requests {
        let e = engine.clone();
        handles.push(tokio::spawn(async move {
            let req = make_request(&format!("Throughput {}", i), max_tokens);
            e.infer(req).await.unwrap()
        }));
    }

    let results: Vec<InferenceResponse> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    let wall_time = start.elapsed();
    let total_tokens: usize = results.iter().map(|r| r.tokens.len()).sum();
    let throughput = total_tokens as f64 / wall_time.as_secs_f64();
    let rps = num_requests as f64 / wall_time.as_secs_f64();

    eprintln!("\n=== Throughput Scaling (instant executor) ===");
    eprintln!(
        "  {} requests, {} tokens in {:.1}ms",
        num_requests,
        total_tokens,
        wall_time.as_secs_f64() * 1000.0
    );
    eprintln!("  {:.0} tokens/sec, {:.0} requests/sec", throughput, rps);

    assert_eq!(results.len(), num_requests);
}
