//! HTTP client for communicating with Ferrum servers

use ferrum_core::{Result, InferenceRequest, InferenceResponse};
use ferrum_server::openai::{ChatCompletionsRequest, ChatCompletionsResponse};
use crate::config::{CliConfig, ClientConfig};
use reqwest::Client;
use std::time::Duration;

/// Ferrum HTTP client
#[derive(Debug, Clone)]
pub struct FerrumClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    config: ClientConfig,
}

impl FerrumClient {
    /// Create a new client
    pub fn new(config: CliConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.client.timeout_seconds))
            .build()
            .map_err(|e| ferrum_core::Error::network(format!("Failed to create HTTP client: {}", e)))?;
        
        Ok(Self {
            client,
            base_url: config.client.base_url.clone(),
            api_key: config.client.api_key.clone(),
            config: config.client,
        })
    }
    
    /// Send inference request
    pub async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let url = format!("{}/v1/infer", self.base_url);
        
        let mut req_builder = self.client.post(&url).json(request);
        
        if let Some(api_key) = &self.api_key {
            req_builder = req_builder.bearer_auth(api_key);
        }
        
        let response = req_builder.send().await
            .map_err(|e| ferrum_core::Error::network(format!("Request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ferrum_core::Error::network(
                format!("Request failed with status {}: {}", status, body)
            ));
        }
        
        let inference_response: InferenceResponse = response.json().await
            .map_err(|e| ferrum_core::Error::deserialization(format!("Failed to parse response: {}", e)))?;
        
        Ok(inference_response)
    }
    
    /// Send chat completions request (OpenAI compatible)
    pub async fn chat_completions(&self, request: &ChatCompletionsRequest) -> Result<ChatCompletionsResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        
        let mut req_builder = self.client.post(&url).json(request);
        
        if let Some(api_key) = &self.api_key {
            req_builder = req_builder.bearer_auth(api_key);
        }
        
        let response = req_builder.send().await
            .map_err(|e| ferrum_core::Error::network(format!("Request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ferrum_core::Error::network(
                format!("Request failed with status {}: {}", status, body)
            ));
        }
        
        let chat_response: ChatCompletionsResponse = response.json().await
            .map_err(|e| ferrum_core::Error::deserialization(format!("Failed to parse response: {}", e)))?;
        
        Ok(chat_response)
    }
    
    /// Get server health
    pub async fn health(&self) -> Result<serde_json::Value> {
        let url = format!("{}/health", self.base_url);
        
        let response = self.client.get(&url).send().await
            .map_err(|e| ferrum_core::Error::network(format!("Health check failed: {}", e)))?;
        
        if !response.status().is_success() {
            let status = response.status();
            return Err(ferrum_core::Error::network(
                format!("Health check failed with status {}", status)
            ));
        }
        
        let health: serde_json::Value = response.json().await
            .map_err(|e| ferrum_core::Error::deserialization(format!("Failed to parse health response: {}", e)))?;
        
        Ok(health)
    }
    
    /// Get server metrics
    pub async fn metrics(&self) -> Result<serde_json::Value> {
        let url = format!("{}/metrics", self.base_url);
        
        let response = self.client.get(&url).send().await
            .map_err(|e| ferrum_core::Error::network(format!("Metrics request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let status = response.status();
            return Err(ferrum_core::Error::network(
                format!("Metrics request failed with status {}", status)
            ));
        }
        
        let metrics: serde_json::Value = response.json().await
            .map_err(|e| ferrum_core::Error::deserialization(format!("Failed to parse metrics response: {}", e)))?;
        
        Ok(metrics)
    }
    
    /// List available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/v1/models", self.base_url);
        
        let mut req_builder = self.client.get(&url);
        
        if let Some(api_key) = &self.api_key {
            req_builder = req_builder.bearer_auth(api_key);
        }
        
        let response = req_builder.send().await
            .map_err(|e| ferrum_core::Error::network(format!("Models request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let status = response.status();
            return Err(ferrum_core::Error::network(
                format!("Models request failed with status {}", status)
            ));
        }
        
        let models: ferrum_server::openai::ModelListResponse = response.json().await
            .map_err(|e| ferrum_core::Error::deserialization(format!("Failed to parse models response: {}", e)))?;
        
        Ok(models.data.into_iter().map(|m| m.id).collect())
    }
}
