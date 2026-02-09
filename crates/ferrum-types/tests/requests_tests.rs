use ferrum_types::*;
use serde_json::json;

#[test]
fn inference_request_builder_defaults() {
    let req = InferenceRequest::new("hello", "llama");
    assert_eq!(req.prompt, "hello");
    assert_eq!(req.model_id.as_str(), "llama");
    assert!(!req.stream);
    assert_eq!(req.priority, Priority::Normal);
}

#[test]
fn inference_request_builder_setters() {
    let params = SamplingParams {
        max_tokens: 16,
        temperature: 0.7,
        ..Default::default()
    };
    let req = InferenceRequest::new("hi", "mistral")
        .with_sampling_params(params.clone())
        .with_stream(true)
        .with_priority(Priority::High)
        .with_client_id("client-1")
        .with_session_id(SessionId::new())
        .with_metadata("k", json!(1));

    assert_eq!(req.sampling_params.max_tokens, 16);
    assert!(req.stream);
    assert_eq!(req.priority, Priority::High);
    assert!(req.client_id.is_some());
    assert!(req.session_id.is_some());
    assert_eq!(req.metadata.get("k").unwrap(), &json!(1));
}

#[test]
fn batch_request_construction() {
    let r1 = InferenceRequest::new("a", "m");
    let r2 = InferenceRequest::new("b", "m").with_sampling_params(SamplingParams {
        max_tokens: 1024,
        ..Default::default()
    });
    let batch = BatchRequest::new(vec![r1, r2]);
    assert_eq!(batch.size(), 2);
    assert!(batch.max_sequence_length >= 512);
    assert!(!batch.is_empty());
}

#[test]
fn scheduled_request_progress_and_state() {
    let req = InferenceRequest::new("a", "m");
    let mut sreq = ScheduledRequest::new(req);
    sreq.update_progress(10);
    sreq.set_state(RequestState::Running);
    assert_eq!(sreq.tokens_processed, 10);
    assert_eq!(sreq.state, RequestState::Running);
}
