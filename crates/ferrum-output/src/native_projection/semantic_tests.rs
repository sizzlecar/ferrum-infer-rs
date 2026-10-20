use super::*;
use serde_json::json;

fn request() -> ApiChatRequest {
    serde_json::from_value(json!({"messages":[],"tool_call_protocol":"function_parameter_xml","tools":[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"content":{"type":"string"}}}}}]})).unwrap()
}

#[test]
fn native_projection_semantic_growth_rolls_back_raw_and_visible_prefix_together() {
    let budget = crate::OutputBudget::new(ferrum_types::OutputConfig {
        semantic_bytes_total: 16,
        semantic_bytes_per_session: 16,
        ..Default::default()
    })
    .unwrap();
    let session = budget.try_open_session().unwrap();
    let mut projector =
        NativeChatOutputProjector::new(&session, &request(), ModelOutputProtocol::Text, false)
            .unwrap()
            .unwrap();
    projector.push("abcd").unwrap();
    assert_eq!(
        session.semantic_usage(),
        8,
        "raw and visible buffers are distinct owners"
    );
    assert!(matches!(
        projector.push("efghi"),
        Err(FerrumError::ResourceExhausted { .. })
    ));
    assert_eq!(projector.raw(), "abcd");
    assert_eq!(projector.visible_prefix(), "abcd");
    assert_eq!(session.semantic_usage(), 8);
    let final_output = projector.finish(FinishReason::Stop).unwrap();
    assert_eq!(final_output.projection().visible.content, "abcd");
    assert!(session.semantic_usage() > 0);
    drop(final_output);
    assert_eq!(session.semantic_usage(), 0);
}

#[test]
fn native_projection_does_not_retain_unconsumed_request_history() {
    let session = crate::OutputBudget::new(Default::default())
        .unwrap()
        .try_open_session()
        .unwrap();
    let mut request = request();
    request.messages.push(ferrum_types::ApiChatMessage {
        role: ferrum_types::ApiMessageRole::Assistant,
        content: "prior answer".repeat(1024),
        name: None,
        tool_calls: vec![],
        tool_call_id: None,
        function_call: None,
    });
    let projector =
        NativeChatOutputProjector::new(&session, &request, ModelOutputProtocol::Text, false)
            .unwrap()
            .unwrap();
    assert!(
        projector.request.messages.is_empty(),
        "projection consumes tool policy, not conversation history"
    );
    assert_eq!(session.semantic_usage(), 0);
}

#[test]
fn native_projection_into_parts_preserves_allocations_and_credit() {
    let budget = crate::OutputBudget::new(Default::default()).unwrap();
    let session = budget.try_open_session().unwrap();
    let mut projector =
        NativeChatOutputProjector::new(&session, &request(), ModelOutputProtocol::Text, false)
            .unwrap()
            .unwrap();
    projector
        .push("<think>Private reasoning.</think>Visible 字节.<tool_call><function=write><parameter=content>Complete 参数</parameter></function></tool_call>")
        .unwrap();
    let output = projector.finish(FinishReason::Stop).unwrap();
    let content_ptr = output.projection().visible.content.as_ptr();
    let reasoning_ptr = output
        .projection()
        .visible
        .reasoning
        .as_ref()
        .unwrap()
        .as_ptr();
    let arguments_ptr = output
        .projection()
        .api_response
        .as_ref()
        .unwrap()
        .message
        .tool_calls[0]
        .function
        .arguments
        .as_ptr();
    let charged = session.semantic_usage();
    assert!(charged > 0);

    let (projection, lease) = output.into_parts();
    assert_eq!(projection.visible.content.as_ptr(), content_ptr);
    assert_eq!(
        projection.visible.reasoning.as_ref().unwrap().as_ptr(),
        reasoning_ptr
    );
    assert_eq!(
        projection.api_response.as_ref().unwrap().message.tool_calls[0]
            .function
            .arguments
            .as_ptr(),
        arguments_ptr
    );
    assert_eq!(projection.visible.content, "Visible 字节.");
    assert_eq!(
        projection.visible.reasoning.as_deref(),
        Some("Private reasoning.")
    );
    assert_eq!(lease.bytes(), charged);
    assert_eq!(session.semantic_usage(), charged);
    assert_eq!(lease.output_session().semantic_usage(), charged);
    drop(session);
    assert_eq!(budget.usage().sessions, 1);
    drop(projection);
    assert_eq!(budget.usage().semantic_bytes, charged);
    drop(lease);
    assert_eq!(budget.usage().semantic_bytes, 0);
    assert_eq!(budget.usage().sessions, 0);
}
