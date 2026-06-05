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
            tokens: vec![TokenId::new(11), TokenId::new(12)],
            finish_reason: FinishReason::Stop,
            usage: TokenUsage::new(7, 2),
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
            usage: Some(TokenUsage::new(7, 2)),
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

fn parse_sse(body: &str) -> (Vec<Value>, usize) {
    let mut chunks = Vec::new();
    let mut done = 0usize;
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            done += 1;
        } else if !data.is_empty() {
            chunks.push(serde_json::from_str(data).expect("valid SSE JSON"));
        }
    }
    (chunks, done)
}

#[tokio::test(flavor = "current_thread")]
async fn structured_output_rejects_unsupported_strict_schema_subset() {
    let fx = ServerFixture::spawn("unused").await;
    let response = Client::new()
        .post(fx.chat_url())
        .json(&json!({
            "model": "stub-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Bad",
                    "strict": true,
                    "schema": {"oneOf": [{"type": "object"}, {"type": "array"}]}
                }
            }
        }))
        .send()
        .await
        .expect("post");
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.expect("error json");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "response_format.json_schema");
}

#[tokio::test(flavor = "current_thread")]
async fn structured_output_stream_buffers_until_valid_strict_json() {
    let fx = ServerFixture::spawn(r#"{"answer":"ok"}"#).await;
    let response = Client::new()
        .post(fx.chat_url())
        .json(&json!({
            "model": "stub-model",
            "messages": [{"role": "user", "content": "return json"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Answer",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"]
                    }
                }
            }
        }))
        .send()
        .await
        .expect("post");
    assert_eq!(response.status(), 200);
    let body = response.text().await.expect("sse body");
    let (chunks, done) = parse_sse(&body);
    assert_eq!(done, 1, "body: {body}");
    let content = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect::<String>();
    assert_eq!(
        serde_json::from_str::<Value>(&content).unwrap()["answer"],
        "ok"
    );
    let usage_chunks = chunks
        .iter()
        .filter(|chunk| chunk.get("usage").is_some_and(|usage| !usage.is_null()))
        .count();
    assert_eq!(usage_chunks, 1, "body: {body}");
}

#[tokio::test(flavor = "current_thread")]
async fn structured_output_tool_choice_required_returns_tool_calls() {
    let fx = ServerFixture::spawn(r#"{"expression":"123+456"}"#).await;
    let response = Client::new()
        .post(fx.chat_url())
        .json(&json!({
            "model": "stub-model",
            "messages": [{"role": "user", "content": "call calc"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "calc",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"]
                    }
                }
            }],
            "tool_choice": "required"
        }))
        .send()
        .await
        .expect("post");
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("json");
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "calc"
    );
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        r#"{"expression":"123+456"}"#
    );
}
