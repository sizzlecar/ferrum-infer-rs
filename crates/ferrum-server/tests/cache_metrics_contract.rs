use async_trait::async_trait;
use ferrum_interfaces::engine::{InferenceEngine, LlmInferenceEngine};
use ferrum_server::{AxumServer, HttpServer, ServerConfig};
use ferrum_types::{
    EngineConfig, EngineMetrics, EngineStatus, FinishReason, HealthStatus, InferenceRequest,
    InferenceResponse, MemoryUsage, ModelId, StreamChunk, TokenId, TokenUsage,
};
use futures::stream;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const PREFIX_METRICS: &[&str] = &[
    "ferrum_prefix_cache_hits_total",
    "ferrum_prefix_cache_misses_total",
    "ferrum_prefix_cache_evictions_total",
    "ferrum_prefix_cache_saved_prefill_tokens_total",
    "ferrum_prefix_cache_entries",
    "ferrum_prefix_cache_bytes",
];

const SESSION_METRICS: &[&str] = &[
    "ferrum_session_cache_hits_total",
    "ferrum_session_cache_misses_total",
    "ferrum_session_cache_evictions_total",
    "ferrum_session_cache_entries",
    "ferrum_session_cache_tokens",
];

struct StubLlm {
    output_budget: ferrum_output::OutputBudget,
    config: EngineConfig,
    text: String,
    prompts: Mutex<Vec<String>>,
}

impl StubLlm {
    fn new(text: &str) -> Self {
        let mut config = EngineConfig::default();
        config.model.model_id = ModelId::new("stub-model");
        Self {
            output_budget: ferrum_output::OutputBudget::new(config.output.clone()).unwrap(),
            config,
            text: text.to_string(),
            prompts: Mutex::new(Vec::new()),
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
            memory_usage: ferrum_types::MemoryObservation::Known(MemoryUsage {
                total_bytes: 0,
                used_bytes: 0,
                free_bytes: 0,
                gpu_memory_bytes: None,
                cpu_memory_bytes: None,
                cache_memory_bytes: 0,
                utilization_percent: 0.0,
            }),
            uptime_seconds: 0,
            last_heartbeat: chrono::Utc::now(),
            version: "test".to_string(),
        }
    }

    async fn shutdown(&self) -> ferrum_types::Result<()> {
        Ok(())
    }

    async fn wait_for_shutdown_completion(&self) -> ferrum_types::Result<()> {
        // No background execution or journal producers in this wire fixture.
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
    async fn infer(
        &self,
        request: InferenceRequest,
    ) -> ferrum_types::Result<ferrum_output::InferenceOutput> {
        self.prompts.lock().unwrap().push(request.prompt.clone());
        ferrum_output::InferenceOutput::new(
            InferenceResponse {
                request_id: request.id,
                text: self.text.clone(),
                tokens: vec![TokenId::new(11)],
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::new(7, 1),
                latency_ms: 1,
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
                api_response: None,
                execution_evidence: None,
            },
            self.output_budget
                .try_open_session()?
                .try_reserve_semantic(0)?,
        )
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> ferrum_types::Result<ferrum_output::InferenceStream> {
        let chunk = StreamChunk {
            request_id: request.id,
            text: self.text.clone(),
            token: Some(TokenId::new(11)),
            finish_reason: Some(FinishReason::Stop),
            usage: Some(TokenUsage::new(7, 1)),
            created_at: chrono::Utc::now(),
            metadata: HashMap::new(),
            api_response: None,
            execution_evidence: None,
        };
        let session = self.output_budget.try_open_session()?;
        let chunk = session.wrap_chunk(chunk).await?;
        Ok(ferrum_output::InferenceStream::new(
            session,
            Box::pin(stream::iter(vec![Ok(chunk)])),
        ))
    }
}

struct ServerFixture {
    base_url: String,
    engine: Arc<StubLlm>,
    task: tokio::task::JoinHandle<()>,
}

impl ServerFixture {
    async fn spawn(text: &str) -> Self {
        let port = free_port();
        let base_url = format!("http://127.0.0.1:{port}");
        let mut config = ServerConfig::default();
        config.host = "127.0.0.1".to_string();
        config.port = port;
        let engine = Arc::new(StubLlm::new(text));
        let server = AxumServer::from_llm(engine.clone());
        let task = tokio::spawn(async move {
            let _ = server.start(&config).await;
        });
        wait_health(&base_url).await;
        Self {
            base_url,
            engine,
            task,
        }
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
            (parts.next()? == name)
                .then(|| parts.next()?.parse::<f64>().ok())
                .flatten()
        })
        .unwrap_or_else(|| panic!("missing metric {name}:\n{metrics}"))
}

#[tokio::test(flavor = "current_thread")]
async fn cache_metrics_contract_keeps_session_history_without_inventing_prefix_reuse() {
    // Requesting prefix caching does not establish that KV state was reused.
    std::env::set_var("FERRUM_PREFIX_CACHE", "1");
    std::env::set_var("FERRUM_SESSION_CACHE", "memory");
    std::env::set_var("FERRUM_SESSION_CACHE_MAX_ENTRIES", "8");
    std::env::set_var("FERRUM_SESSION_CACHE_MAX_TOKENS", "512");

    let fx = ServerFixture::spawn("remembered assistant answer").await;
    let client = Client::new();
    let first_prompt = "shared prefix prompt with first suffix";
    let second_prompt = "shared prefix prompt with second suffix";
    for prompt in [first_prompt, second_prompt] {
        let payload = json!({
            "model": "stub-model",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 8
        });
        let response = client
            .post(fx.chat_url())
            .header("X-Ferrum-Session", "session-a")
            .json(&payload)
            .send()
            .await
            .expect("chat post");
        assert_eq!(response.status(), 200);
        let body: Value = response.json().await.expect("chat json");
        assert_eq!(
            body["choices"][0]["message"]["content"],
            "remembered assistant answer"
        );
    }
    {
        let prompts = fx.engine.prompts.lock().unwrap();
        assert_eq!(prompts.len(), 2);
        assert!(prompts[0].contains(first_prompt));
        assert!(!prompts[0].contains("remembered assistant answer"));
        assert!(prompts[1].contains(first_prompt));
        assert!(prompts[1].contains("remembered assistant answer"));
        assert!(prompts[1].contains(second_prompt));
    }

    let health: Value = client
        .get(format!("{}/health", fx.base_url))
        .send()
        .await
        .expect("health")
        .json()
        .await
        .expect("health json");
    assert_eq!(health["cache"]["prefix_cache"], Value::Null);
    assert_eq!(health["cache"]["session_cache"]["mode"], "memory");

    let metrics = client
        .get(format!("{}/metrics", fx.base_url))
        .send()
        .await
        .expect("metrics")
        .text()
        .await
        .expect("metrics text");
    for name in SESSION_METRICS {
        assert!(metrics.contains(name), "missing {name}:\n{metrics}");
    }
    for name in PREFIX_METRICS {
        assert!(
            !metrics.contains(name),
            "prefix metric has no owner observation: {name}\n{metrics}"
        );
    }
    assert_eq!(
        metric_value(&metrics, "ferrum_session_cache_hits_total"),
        1.0
    );
    assert_eq!(
        metric_value(&metrics, "ferrum_session_cache_misses_total"),
        1.0
    );
    assert_eq!(metric_value(&metrics, "ferrum_session_cache_entries"), 1.0);
}
