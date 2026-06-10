//! Full-stack OpenAI wire-contract test (the server slice of the tiny_stack
//! suite). Spawns the real `AxumServer` over a stub engine — no model
//! download, no GPU — and asserts the `/v1/models` and chat-validation
//! contracts in-process.
//!
//! `tiny_stack_openai_wire_contract` is pinned by
//! `scripts/release/test_arch_goal_gate.py` (`REQUIRED_SCENARIO_TESTS`).
//! Kills hb-06: `/v1/models` returning an empty list and an empty `messages`
//! array being accepted instead of rejected with 400.

use async_trait::async_trait;
use ferrum_interfaces::engine::{InferenceEngine, LlmInferenceEngine};
use ferrum_server::{AxumServer, HttpServer, ServerConfig};
use ferrum_types::{
    EngineConfig, EngineMetrics, EngineStatus, FinishReason, HealthStatus, InferenceRequest,
    InferenceResponse, MemoryUsage, ModelId, StreamChunk, TokenId, TokenUsage,
};
use futures::{stream, Stream};
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::TcpListener;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct StubLlm {
    config: EngineConfig,
}

impl StubLlm {
    fn new() -> Self {
        let mut config = EngineConfig::default();
        config.model.model_id = ModelId::new("stub-model");
        Self { config }
    }
}

#[async_trait]
impl InferenceEngine for StubLlm {
    async fn status(&self) -> EngineStatus {
        EngineStatus {
            is_ready: true,
            loaded_models: vec![self.config.model.model_id.clone()],
            active_requests: 0,
            queued_requests: 0,
            memory_usage: MemoryUsage {
                total_bytes: 0,
                used_bytes: 0,
                free_bytes: 0,
                gpu_memory_bytes: None,
                cpu_memory_bytes: None,
                cache_memory_bytes: 0,
                utilization_percent: 0.0,
            },
            uptime_seconds: 0,
            last_heartbeat: chrono::Utc::now(),
            version: "test".to_string(),
        }
    }

    async fn shutdown(&self) -> ferrum_types::Result<()> {
        Ok(())
    }

    fn config(&self) -> &EngineConfig {
        &self.config
    }

    fn metrics(&self) -> EngineMetrics {
        EngineMetrics::default()
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus::healthy()
    }
}

#[async_trait]
impl LlmInferenceEngine for StubLlm {
    async fn infer(&self, request: InferenceRequest) -> ferrum_types::Result<InferenceResponse> {
        Ok(InferenceResponse {
            request_id: request.id,
            text: "ok".to_string(),
            tokens: vec![TokenId::new(11)],
            finish_reason: FinishReason::Stop,
            usage: TokenUsage::new(7, 1),
            latency_ms: 1,
            created_at: chrono::Utc::now(),
            metadata: HashMap::new(),
            api_response: None,
        })
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> ferrum_types::Result<Pin<Box<dyn Stream<Item = ferrum_types::Result<StreamChunk>> + Send>>>
    {
        let chunk = StreamChunk {
            request_id: request.id,
            text: "ok".to_string(),
            token: Some(TokenId::new(11)),
            finish_reason: Some(FinishReason::Stop),
            usage: Some(TokenUsage::new(7, 1)),
            created_at: chrono::Utc::now(),
            metadata: HashMap::new(),
            api_response: None,
        };
        Ok(Box::pin(stream::iter(vec![Ok(chunk)])))
    }
}

struct ServerFixture {
    base_url: String,
    task: tokio::task::JoinHandle<()>,
}

impl ServerFixture {
    async fn spawn() -> Self {
        let port = free_port();
        let base_url = format!("http://127.0.0.1:{port}");
        let mut config = ServerConfig::default();
        config.host = "127.0.0.1".to_string();
        config.port = port;
        let server = AxumServer::from_llm(Arc::new(StubLlm::new()));
        let task = tokio::spawn(async move {
            let _ = server.start(&config).await;
        });
        wait_health(&base_url).await;
        Self { base_url, task }
    }
}

impl Drop for ServerFixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local_addr").port()
}

async fn wait_health(base_url: &str) {
    let client = Client::new();
    let health_url = format!("{base_url}/health");
    let start = Instant::now();
    loop {
        if start.elapsed() > Duration::from_secs(10) {
            panic!("server did not become healthy");
        }
        let ok = client
            .get(&health_url)
            .timeout(Duration::from_secs(1))
            .send()
            .await
            .map(|response| response.status().is_success())
            .unwrap_or(false);
        if ok {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "current_thread")]
async fn tiny_stack_openai_wire_contract() {
    let fx = ServerFixture::spawn().await;
    let client = Client::new();

    // (1) /v1/models must list the loaded model, not an empty array.
    let resp = client
        .get(format!("{}/v1/models", fx.base_url))
        .send()
        .await
        .expect("models request");
    assert_eq!(resp.status(), 200, "/v1/models must be 200");
    let body: Value = resp.json().await.expect("models json");
    let data = body
        .get("data")
        .and_then(Value::as_array)
        .expect("models response has a data array");
    assert!(
        !data.is_empty(),
        "/v1/models must not be empty when a model is loaded: {body}"
    );
    assert!(
        data.iter()
            .any(|m| m.get("id").and_then(Value::as_str) == Some("stub-model")),
        "/v1/models must list the loaded model id: {body}"
    );

    // (2) An empty messages array must be rejected with 400, not accepted.
    let resp = client
        .post(format!("{}/v1/chat/completions", fx.base_url))
        .json(&json!({ "model": "stub-model", "messages": [] }))
        .send()
        .await
        .expect("empty-messages request");
    assert_eq!(
        resp.status(),
        400,
        "empty messages must be a 400 BadRequest, got {}",
        resp.status()
    );
}
