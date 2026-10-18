use super::*;
use axum::body::to_bytes;
use ferrum_output::OutputBudget;
use ferrum_types::OutputConfig;
use futures::{stream, FutureExt, StreamExt};

#[tokio::test]
async fn output_frame_keeps_session_after_body_eof_and_drop() {
    let budget = OutputBudget::new(OutputConfig {
        max_sessions: 1,
        ..OutputConfig::default()
    })
    .unwrap();
    let session = budget.try_open_session().unwrap();
    let bytes = encode_done(&session).await.unwrap();
    let response = sse_response(session, stream::iter([Ok(bytes)]));
    let mut body = response.into_body().into_data_stream();
    let frame = body.next().await.unwrap().unwrap();
    assert_eq!(frame.as_ref(), b"data: [DONE]\n\n");
    let cloned = frame.clone();
    let sliced = frame.slice(1..4);
    assert!(body.next().await.is_none());
    assert!(budget.try_open_session().is_err());
    drop(body);
    drop(frame);
    drop(cloned);
    assert!(
        budget.try_open_session().is_err(),
        "last frame slice still owns session"
    );
    drop(sliced);
    assert!(budget.try_open_session().is_ok());
}

#[tokio::test]
async fn output_frame_retains_encoded_credit_through_last_clone() {
    let length = b"data: [DONE]\n\n".len();
    let budget = OutputBudget::new(OutputConfig {
        encoded_bytes_total: length,
        max_event_bytes: length,
        ..OutputConfig::default()
    })
    .unwrap();
    let session = budget.try_open_session().unwrap();
    let frame = encode_done(&session).await.unwrap();
    let last = frame.clone();
    let wait = session.reserve_encoded(length);
    futures::pin_mut!(wait);
    assert!(futures::poll!(&mut wait).is_pending());
    drop(frame);
    assert!(futures::poll!(&mut wait).is_pending());
    drop(last);
    let permit = wait.await.unwrap();
    drop(permit);
}

#[tokio::test]
async fn output_json_encoder_preserves_axum_sse_wire_and_checks_escaped_size() {
    use axum::response::{sse::Event, Sse};
    let value = serde_json::json!({"text": "中文\n\r\t\"\\\u{0001}"});
    let expected = Sse::new(stream::iter([Ok::<_, std::convert::Infallible>(
        Event::default()
            .event("response.delta")
            .json_data(&value)
            .unwrap(),
    )]))
    .into_response();
    let expected = to_bytes(expected.into_body(), usize::MAX).await.unwrap();
    let budget = OutputBudget::new(OutputConfig::default()).unwrap();
    let session = budget.try_open_session().unwrap();
    let actual = encode_json(&session, Some("response.delta"), &value)
        .await
        .unwrap();
    assert_eq!(actual, expected);

    let limited = OutputBudget::new(OutputConfig {
        encoded_bytes_total: expected.len(),
        max_event_bytes: expected.len() - 1,
        ..OutputConfig::default()
    })
    .unwrap();
    let limited = limited.try_open_session().unwrap();
    assert!(matches!(
        encode_json(&limited, Some("response.delta"), &value).await,
        Err(FerrumError::ResourceExhausted { .. })
    ));
    // Rejection happens before acquiring encoded space or allocating its buffer.
    let permit = limited.reserve_encoded(expected.len() - 1);
    futures::pin_mut!(permit);
    assert!(matches!(futures::poll!(permit), Poll::Ready(Ok(_))));
}

#[tokio::test]
async fn output_limit_error_bypasses_full_data_pool_and_retains_last_frame_owner() {
    use crate::axum_server::chat_stream::{self, ChatEventStream, ChatStreamEvent};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    struct Events {
        dropped: Arc<AtomicBool>,
    }
    impl Stream for Events {
        type Item = ChatStreamEvent;
        fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(Some(ChatStreamEvent::Done))
        }
    }
    impl Drop for Events {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    let budget = OutputBudget::new(OutputConfig {
        max_sessions: 1,
        encoded_bytes_total: 8,
        max_event_bytes: 8,
        ..OutputConfig::default()
    })
    .unwrap();
    let session = budget.try_open_session().unwrap();
    let normal = session.reserve_encoded(8).await.unwrap();
    let dropped = Arc::new(AtomicBool::new(false));
    let response = chat_stream::into_response(ChatEventStream {
        session,
        events: Box::pin(Events {
            dropped: dropped.clone(),
        }),
    });
    let mut body = response.into_body().into_data_stream();
    let frame = body
        .next()
        .now_or_never()
        .expect("control error cannot wait for data credit")
        .unwrap()
        .unwrap();
    assert!(
        dropped.load(Ordering::SeqCst),
        "upstream closes before the client polls again"
    );
    let wire = std::str::from_utf8(&frame).unwrap();
    assert!(wire.contains("output_limit_exceeded"));
    assert!(!wire.contains("[DONE]"));
    assert!(frame.len() <= ferrum_output::TERMINAL_ERROR_BYTES);
    assert_eq!(budget.usage().encoded_bytes, 8);
    assert_eq!(budget.usage().control_bytes, frame.len());
    let last = frame.clone();
    assert!(body.next().now_or_never().unwrap().is_none());
    drop(body);
    drop(normal);
    drop(frame);
    assert_eq!(budget.usage().sessions, 1);
    assert_eq!(budget.usage().control_bytes, last.len());
    assert!(budget.try_open_session().is_err());
    drop(last);
    assert_eq!(budget.usage().sessions, 0);
    assert_eq!(budget.usage().control_bytes, 0);
}

#[tokio::test]
async fn output_serialization_failure_stays_internal_and_control_cannot_repeat() {
    struct Invalid;
    impl Serialize for Invalid {
        fn serialize<S: serde::Serializer>(&self, _: S) -> std::result::Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("fixture serialization failure"))
        }
    }
    let budget = OutputBudget::new(OutputConfig::default()).unwrap();
    let session = budget.try_open_session().unwrap();
    let failure = encode_json(&session, None, &Invalid).await;
    assert!(failure.is_err());
    assert!(!matches!(
        &failure,
        Err(FerrumError::ResourceExhausted { .. })
    ));
    let (frame, terminal) = chat_encoding_result(&session, failure).unwrap();
    assert!(terminal);
    let wire = std::str::from_utf8(&frame).unwrap();
    assert!(wire.contains("internal_server_error"));
    assert!(!wire.contains("output_limit_exceeded"));
    drop(frame);
    assert_eq!(budget.usage().control_bytes, 0);
    assert!(chat_encoding_result(&session, Err(FerrumError::internal("again"))).is_err());
}

#[tokio::test]
async fn output_known_chat_error_uses_control_when_normal_pool_is_full() {
    use crate::axum_server::chat_stream::{self, ChatEventStream, ChatStreamEvent};
    let budget = OutputBudget::new(OutputConfig {
        encoded_bytes_total: 8,
        max_event_bytes: 8,
        ..OutputConfig::default()
    })
    .unwrap();
    let session = budget.try_open_session().unwrap();
    let normal = session.reserve_encoded(8).await.unwrap();
    let error = crate::openai::OpenAiError {
        error: crate::openai::OpenAiErrorDetail {
            message: "large upstream detail\n".repeat(1024),
            error_type: "invalid_request_error".into(),
            code: Some("invalid_schema".into()),
            param: Some("response_format".into()),
        },
    };
    let mut body = chat_stream::into_response(ChatEventStream {
        session,
        events: Box::pin(stream::iter([ChatStreamEvent::Error(error)])),
    })
    .into_body()
    .into_data_stream();
    let frame = body
        .next()
        .now_or_never()
        .expect("known failure cannot wait for data credit")
        .unwrap()
        .unwrap();
    let wire = std::str::from_utf8(&frame).unwrap();
    let payload = wire
        .split("\n\n")
        .next()
        .unwrap()
        .strip_prefix("data: ")
        .unwrap();
    let error: serde_json::Value = serde_json::from_str(payload).unwrap();
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert_eq!(error["error"]["code"], "invalid_schema");
    assert_eq!(error["error"]["param"], "response_format");
    assert_eq!(error["error"]["message"], OMITTED_UPSTREAM_DIAGNOSTIC);
    assert!(wire.ends_with("data: [DONE]\n\n"));
    assert!(!wire.contains("output_limit_exceeded"));
    assert!(frame.len() <= ferrum_output::TERMINAL_ERROR_BYTES);
    assert_eq!(budget.usage().encoded_bytes, 8);
    assert_eq!(budget.usage().control_bytes, frame.len());
    let last = frame.clone();
    assert!(body.next().now_or_never().unwrap().is_none());
    drop(body);
    drop(frame);
    drop(normal);
    assert_eq!(budget.usage().sessions, 1);
    drop(last);
    assert_eq!(budget.usage().sessions, 0);
    assert_eq!(budget.usage().control_bytes, 0);
}

#[test]
fn output_known_error_unbounded_classification_cannot_escape_control_limit() {
    let budget = OutputBudget::new(OutputConfig::default()).unwrap();
    let session = budget.try_open_session().unwrap();
    let error = crate::openai::OpenAiError {
        error: crate::openai::OpenAiErrorDetail {
            message: "detail".repeat(4096),
            error_type: "invalid_request_error".into(),
            code: Some("code".repeat(4096)),
            param: Some("param".repeat(4096)),
        },
    };
    let frame = encode_chat_failure(&session, &error).unwrap();
    let wire = std::str::from_utf8(&frame).unwrap();
    let payload = wire
        .split("\n\n")
        .next()
        .unwrap()
        .strip_prefix("data: ")
        .unwrap();
    let value: serde_json::Value = serde_json::from_str(payload).unwrap();
    assert_eq!(value["error"]["type"], "invalid_request_error");
    assert_eq!(value["error"]["code"], "upstream_error");
    assert_eq!(value["error"]["param"], serde_json::Value::Null);
    assert!(frame.len() <= ferrum_output::TERMINAL_ERROR_BYTES);
}
