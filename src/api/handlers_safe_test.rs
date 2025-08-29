//! Safe unit tests for API handlers without actix-web integration
//! 
//! These tests verify the logic without running the full web server

#[cfg(test)]
mod tests {
    use crate::api::handlers::*;
    use crate::api::types::*;
    use crate::config::Config;
    use crate::inference::{InferenceEngine, InferenceRequest, InferenceResponse};
    use std::sync::Arc;

    #[test]
    fn test_create_chat_completion_request() {
        let request = ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello!".to_string(),
                name: None,
            }],
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(1.0),
            n: Some(1),
            stream: Some(false),
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
        };

        // Convert to inference request
        let inference_req: InferenceRequest = request.clone().into();
        assert_eq!(inference_req.prompt, "Hello!");
        assert_eq!(inference_req.max_tokens, Some(100));
        assert_eq!(inference_req.temperature, Some(0.7));
    }

    #[test]
    fn test_create_completion_request() {
        let request = CompletionRequest {
            model: "test-model".to_string(),
            prompt: "Hello, world!".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(1.0),
            n: Some(1),
            stream: Some(false),
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
        };

        // Convert to inference request
        let inference_req: InferenceRequest = request.clone().into();
        assert_eq!(inference_req.prompt, "Hello, world!");
        assert_eq!(inference_req.max_tokens, Some(100));
    }

    #[test]
    fn test_app_state_creation() {
        let config = Config::default();
        let engine = InferenceEngine::new_for_test(config.clone());
        let app_state = AppState {
            engine: Arc::new(engine),
            config,
        };

        // Verify state is created correctly
        assert_eq!(app_state.config.server.port, 8080);
    }
}

