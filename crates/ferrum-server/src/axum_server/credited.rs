//! Product routing for explicitly selected, already encoded credited output.
//! The engine owns framing; HTTP transfers the last byte owner without a queue
//! or another serialization pass. Other protocol codecs remain explicit gaps.

use super::*;
use ferrum_interfaces::output_flow::{
    CreditedOutputFrame, CreditedOutputSession, OutputProjectionContract,
};
mod chat;
mod evidence;
pub(super) use chat::stream as chat_stream;
use ferrum_types::SloOutputTransport;

pub(super) fn enabled(state: &AppState) -> bool {
    state.llm.as_ref().is_some_and(|engine| {
        engine.config().scheduler.slo.output.transport == SloOutputTransport::Credited
    })
}

pub(super) fn require_legacy_endpoint(
    state: &AppState,
    endpoint: &'static str,
) -> std::result::Result<(), ServerError> {
    if enabled(state) {
        return Err(ServerError::unsupported_feature(
            format!(
                "credited output does not yet support {endpoint}; supported HTTP projections are ordinary Text streaming /v1/completions and /v1/chat/completions"
            ),
            None,
        ));
    }
    Ok(())
}

pub(super) async fn completions_stream(
    state: AppState,
    openai_request: CompletionsRequest,
    mut inference_request: InferenceRequest,
    context: InferenceRequestContext,
) -> std::result::Result<Response, ServerError> {
    inference_request
        .evidence_request
        .capture_engine_token_timing = state.profile_detail.captures_engine_token_timing();
    inference_request.evidence_request.capture_prompt_token_ids = state.request_dump_dir.is_some();
    let observer = evidence::Observer::new(
        &state,
        openai_request.model.clone(),
        "/v1/completions",
        None,
    );
    let engine = state.llm.ok_or_else(|| {
        ServerError::ServiceUnavailable("LLM engine not loaded; completions unavailable".into())
    })?;
    // This endpoint currently always emits usage; unlike Chat, its request
    // type has no stream_options switch. Preserve that resolved wire rule.
    let contract = Arc::new(OutputProjectionContract::completions_sse(
        Uuid::new_v4().to_string(),
        openai_request.model,
        true,
    ));
    // Startup rejection precedes SSE headers. Later error and DONE frames are
    // encoded by the owner using its separately reserved terminal capacity.
    let session = engine
        .infer_credited_stream(inference_request, context, contract)
        .await
        .map_err(server_error_from_ferrum_error)?;
    Ok(stream_response(session, observer))
}

fn stream_response(
    session: CreditedOutputSession,
    observer: Option<evidence::Observer>,
) -> Response {
    // The optional observer owns only the terminal receiver. Frames remain the
    // cancellation authority; no second wire queue or text copy is introduced.
    if let Some(observer) = observer {
        observer.spawn(session.completion);
    } else {
        drop(session.completion);
    }
    let body =
        crate::credited_output::stream_body(session.frames.map(CreditedOutputFrame::into_wire));
    (
        [
            (axum::http::header::CONTENT_TYPE, "text/event-stream"),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
        ],
        body,
    )
        .into_response()
}
