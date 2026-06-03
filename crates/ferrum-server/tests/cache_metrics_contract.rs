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

const REQUIRED_METRICS: &[&str] = &[
    "ferrum_prefix_cache_hits_total",
    "ferrum_prefix_cache_misses_total",
    "ferrum_prefix_cache_evictions_total",
    "ferrum_prefix_cache_saved_prefill_tokens_total",
    "ferrum_prefix_cache_entries",
    "ferrum_prefix_cache_bytes",
    "ferrum_session_cache_hits_total",
    "ferrum_session_cache_misses_total",
    "ferrum_session_cache_evictions_total",
    "ferrum_session_cache_entries",
    "ferrum_session_cache_tokens",
];

struct StubLlm {
    config: EngineConfig,
    text: String,
}

impl StubLlm {
    fn new(text: &str) -> Self {
        let mut config = EngineConfig::default();
        config.model.model_id = ModelId::new("stub-model");
        Self {
            config,
            text: text.to_string(),
        }
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
            text: self.text.clone(),
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
            text: self.text.clone(),
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
    async fn spawn(text: &str) -> Self {
        let port = free_port();
        let base_url = format!("http://127.0.0.1:{port}");
        let mut config = ServerConfig::default();
        config.host = "127.0.0.1".to_string();
        config.port = port;
        let server = AxumServer::from_llm(Arc::new(StubLlm::new(text)));
        let task = tokio::spawn(async move {
            let _ = server.start(&config).await;
        });
        wait_health(&base_url).await;
        Self { base_url, task }
    }

    fn chat_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
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

fn metric_value(metrics: &str, name: &str) -> f64 {
    metrics
        .lines()
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            (parts.next()? == name).then(|| parts.next()?.parse::<f64>().ok()).flatten()
        })
        .unwrap_or_else(|| panic!("missing metric {name}:\n{metrics}"))
}

#[tokio::test(flavor = "current_thread")]
async fn cache_metrics_contract_exposes_prefix_and_session_metrics() {
    std::env::set_var("FERRUM_PREFIX_CACHE", "1");
    std::env::set_var("FERRUM_SESSION_CACHE", "memory");
    std::env::set_var("FERRUM_SESSION_CACHE_MAX_ENTRIES", "8");
    std::env::set_var("FERRUM_SESSION_CACHE_MAX_TOKENS", "512");

    let fx = ServerFixture::spawn("ok").await;
    let client = Client::new();
    let payload = json!({
        "model": "stub-model",
        "messages": [{"role": "user", "content": "shared prefix prompt with stable suffix"}],
        "temperature": 0,
        "max_tokens": 8
    });
    for _ in 0..2 {
        let response = client
            .post(fx.chat_url())
            .header("X-Ferrum-Session", "session-a")
            .json(&payload)
            .send()
            .await
            .expect("chat post");
        assert_eq!(response.status(), 200);
    }

    let health: Value = client
        .get(format!("{}/health", fx.base_url))
        .send()
        .await
        .expect("health")
        .json()
        .await
        .expect("health json");
    assert_eq!(health["cache"]["prefix_cache"]["enabled"], true);
    assert_eq!(health["cache"]["session_cache"]["mode"], "memory");

    let metrics = client
        .get(format!("{}/metrics", fx.base_url))
        .send()
        .await
        .expect("metrics")
        .text()
        .await
        .expect("metrics text");
    for name in REQUIRED_METRICS {
        assert!(metrics.contains(name), "missing {name}:\n{metrics}");
    }
    assert!(metric_value(&metrics, "ferrum_prefix_cache_hits_total") > 0.0);
    assert!(metric_value(&metrics, "ferrum_prefix_cache_saved_prefill_tokens_total") > 0.0);
    assert!(metric_value(&metrics, "ferrum_session_cache_hits_total") > 0.0);
}
