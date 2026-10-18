use super::*;

fn context() -> ResponseContext {
    let request: super::super::request::ResponsesRequest =
        serde_json::from_value(json!({"model":"test-model","input":"hello","store":false}))
            .unwrap();
    request.convert().unwrap().response
}

#[tokio::test]
async fn responses_semantic_terminal_borrows_history_when_no_full_copy_can_fit() {
    let budget = ferrum_output::OutputBudget::new(ferrum_types::OutputConfig {
        semantic_bytes_total: 16384,
        semantic_bytes_per_session: 16384,
        ..Default::default()
    })
    .unwrap();
    let session = budget.try_open_session().unwrap();
    let mut state = ResponsesStreamState::new(context(), session.clone()).unwrap();
    let expected = "字节 full output ".repeat(256);
    let mut sequence = 0;
    {
        let mut events = state.push_text(&expected);
        while let Some(event) = events.next().await {
            let (_, failed) = encode_event(&session, event, "resp_test", 0, &mut sequence)
                .await
                .unwrap();
            assert!(!failed);
        }
    }
    assert_eq!(state.text, expected);
    let held = session
        .try_reserve_semantic(16384 - session.semantic_usage() - 64)
        .unwrap();
    assert!(
        session.try_reserve_semantic(expected.len()).is_err(),
        "a second full text owner cannot fit"
    );
    let mut complete = false;
    let mut done = false;
    {
        let mut events = state.complete();
        while let Some(event) = events.next().await {
            let (bytes, failed) = encode_event(&session, event, "resp_test", 0, &mut sequence)
                .await
                .unwrap();
            assert!(!failed, "terminal fields borrow the existing owner");
            let data = sse_data(&bytes).unwrap();
            if data == "[DONE]" {
                done = true;
                continue;
            }
            let value: Value = serde_json::from_str(&data).unwrap();
            if value["type"] == "response.output_text.done" {
                assert_eq!(value["text"], expected);
            }
            if value["type"] == "response.completed" {
                complete = true;
                assert_eq!(
                    value["response"]["output"][0]["content"][0]["text"],
                    expected
                );
                assert_eq!(value["response"]["output"][0]["phase"], "final_answer");
            }
        }
    }
    assert!(complete && done);
    drop(held);
    drop(state);
    assert_eq!(session.semantic_usage(), 0);
}

#[test]
fn responses_semantic_tool_growth_rejection_preserves_existing_arguments_and_credit() {
    let budget = ferrum_output::OutputBudget::new(ferrum_types::OutputConfig {
        semantic_bytes_total: 512,
        semantic_bytes_per_session: 512,
        ..Default::default()
    })
    .unwrap();
    let session = budget.try_open_session().unwrap();
    let mut state = ResponsesStreamState::new(context(), session.clone()).unwrap();
    state
        .append_call(0, "call_1", "write", "{\"text\":")
        .unwrap();
    let charged = session.semantic_usage();
    assert!(matches!(
        state.append_call(0, "replacement", "changed", &"x".repeat(1024)),
        Err(ferrum_types::FerrumError::ResourceExhausted { .. })
    ));
    assert_eq!(state.tool_calls[&0].id, "call_1");
    assert_eq!(state.tool_calls[&0].name, "write");
    assert_eq!(state.tool_calls[&0].arguments, "{\"text\":");
    assert_eq!(session.semantic_usage(), charged);
    drop(state);
    assert_eq!(session.semantic_usage(), 0);
}
