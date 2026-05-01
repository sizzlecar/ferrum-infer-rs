use ferrum_engine::{create_default_engine, simple_engine_config};
use ferrum_interfaces::InferenceEngine;
use ferrum_types::{Device, InferenceRequest, SamplingParams};
use std::{path::Path, sync::Arc, time::Duration};

const DEFAULT_QWEN_05B_PATH: &str = "/Users/chejinxuan/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775";
const QWEN_MODEL_ID: &str = "Qwen/Qwen2.5-0.5B-Instruct";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires local Qwen2.5-0.5B model files"]
async fn qwen_25_05b_requests_are_serialized_for_correctness() {
    let model_path = std::env::var("FERRUM_QWEN_TEST_MODEL_PATH")
        .unwrap_or_else(|_| DEFAULT_QWEN_05B_PATH.to_string());

    if !Path::new(&model_path).exists() {
        eprintln!(
            "Skip qwen concurrency test: model path does not exist: {}",
            model_path
        );
        return;
    }

    // Rust 2024 marks process-wide env mutation as unsafe.
    unsafe {
        std::env::set_var("FERRUM_MODEL_PATH", &model_path);
    }

    let mut config = simple_engine_config(QWEN_MODEL_ID, Device::CPU);
    config.scheduler.max_running_requests = 4;

    let engine: Arc<dyn InferenceEngine + Send + Sync> =
        Arc::from(create_default_engine(config).await.expect("create engine"));

    let monitor_engine = engine.clone();
    let monitor = tokio::spawn(async move {
        let mut max_active = 0usize;
        let mut max_queued = 0usize;
        for _ in 0..300 {
            let status = monitor_engine.status().await;
            max_active = max_active.max(status.active_requests);
            max_queued = max_queued.max(status.queued_requests);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        (max_active, max_queued)
    });

    let mut tasks = Vec::new();
    for i in 0..4 {
        let engine = engine.clone();
        tasks.push(tokio::spawn(async move {
            let mut params = SamplingParams::greedy();
            params.max_tokens = 12;
            let request = InferenceRequest::new(format!("Reply with req-{}", i), QWEN_MODEL_ID)
                .with_sampling_params(params);
            engine.infer(request).await
        }));
    }

    for task in tasks {
        let response = task.await.expect("join inference task").expect("inference");
        assert!(
            !response.text.trim().is_empty() || !response.tokens.is_empty(),
            "empty generation response"
        );
    }

    let (max_active, max_queued) = monitor.await.expect("join monitor");
    assert!(
        max_active <= 1,
        "Qwen executor should be serialized, observed max_active={}",
        max_active
    );
    assert!(
        max_queued >= 1,
        "expected at least one queued request under concurrent load"
    );

    unsafe {
        std::env::remove_var("FERRUM_MODEL_PATH");
    }
}
