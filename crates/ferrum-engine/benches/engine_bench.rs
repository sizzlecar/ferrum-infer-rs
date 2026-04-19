//! Engine benchmarks: throughput and latency.
//!
//! Measures TTFT (Time To First Token), TPOT (Time Per Output Token),
//! throughput (tokens/sec), and P50/P95/P99 latencies.
//!
//! Uses mock components — no GPU required.  Results measure the scheduling
//! and engine overhead, not model computation.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ferrum_engine::{ContinuousBatchEngine, InferenceEngineInterface};
use ferrum_scheduler::implementations::ContinuousBatchScheduler;
use ferrum_testkit::{
    MockKvCacheManager, MockModelExecutor, MockSampler, MockTensorFactory, MockTokenizer,
};
use ferrum_types::{InferenceRequest, SchedulerConfig};
use std::sync::Arc;
use std::time::Duration;

const VOCAB_SIZE: usize = 1000;

fn make_engine() -> ContinuousBatchEngine {
    let config = ferrum_types::EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let tokenizer = Arc::new(MockTokenizer::new(VOCAB_SIZE));
    let sampler = Arc::new(MockSampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(1024));
    let executor = Arc::new(MockModelExecutor::instant(VOCAB_SIZE));
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

// ────────────────────────────────────────────────────────────────────────────
// Single request latency
// ────────────────────────────────────────────────────────────────────────────

fn bench_single_request(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("single_request");
    group.measurement_time(Duration::from_secs(5));

    for max_tokens in [1, 5, 10, 20, 50] {
        group.throughput(Throughput::Elements(max_tokens as u64));
        group.bench_with_input(
            BenchmarkId::new("tokens", max_tokens),
            &max_tokens,
            |b, &max_tokens| {
                let engine = make_engine();
                b.iter(|| {
                    rt.block_on(async {
                        let req = make_request("Benchmark prompt", max_tokens);
                        engine.infer(req).await.unwrap()
                    })
                });
            },
        );
    }
    group.finish();
}

// ────────────────────────────────────────────────────────────────────────────
// Concurrent throughput
// ────────────────────────────────────────────────────────────────────────────

fn bench_concurrent_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("concurrent_throughput");
    group.measurement_time(Duration::from_secs(5));

    for concurrency in [1, 2, 4, 8, 16] {
        let total_tokens = concurrency * 5; // 5 tokens per request
        group.throughput(Throughput::Elements(total_tokens as u64));
        group.bench_with_input(
            BenchmarkId::new("concurrency", concurrency),
            &concurrency,
            |b, &concurrency| {
                let engine = Arc::new(make_engine());
                b.iter(|| {
                    rt.block_on(async {
                        let mut handles = Vec::new();
                        for i in 0..concurrency {
                            let e = engine.clone();
                            handles.push(tokio::spawn(async move {
                                let req = make_request(&format!("Concurrent {}", i), 5);
                                e.infer(req).await.unwrap()
                            }));
                        }
                        futures::future::join_all(handles).await
                    })
                });
            },
        );
    }
    group.finish();
}

// ────────────────────────────────────────────────────────────────────────────
// Scheduling overhead (instant executor)
// ────────────────────────────────────────────────────────────────────────────

fn bench_scheduling_overhead(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("scheduling_overhead");
    group.measurement_time(Duration::from_secs(5));

    // Measure pure scheduling overhead with instant executor
    group.bench_function("single_request_overhead", |b| {
        let engine = make_engine();
        b.iter(|| {
            rt.block_on(async {
                let req = make_request("Overhead test", 1);
                engine.infer(req).await.unwrap()
            })
        });
    });

    // Sequential requests to measure per-request overhead
    group.bench_function("sequential_10_requests", |b| {
        let engine = make_engine();
        b.iter(|| {
            rt.block_on(async {
                for i in 0..10 {
                    let req = make_request(&format!("Seq {}", i), 1);
                    engine.infer(req).await.unwrap();
                }
            })
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_request,
    bench_concurrent_throughput,
    bench_scheduling_overhead,
);
criterion_main!(benches);
