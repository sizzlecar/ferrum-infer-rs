//! Chat routing after the ordinary registry, request and template conversion.
use super::*;
use ferrum_types::{ApiRequest, ApiToolChoice, ResponseCompletionBoundary};

pub(in crate::axum_server) async fn stream(
    state: AppState,
    request: ChatCompletionsRequest,
    inference_request: InferenceRequest,
    context: InferenceRequestContext,
    correlation: Option<BenchmarkRequestCorrelation>,
) -> std::result::Result<Response, ServerError> {
    let unsupported = || {
        ServerError::unsupported_feature(
        "credited output does not yet support this Chat projection; ordinary streaming Text without tool or structured contracts is supported".to_owned(), None)
    };
    let Some(ApiRequest::Chat(chat)) = inference_request.api_request.as_ref() else {
        return Err(unsupported());
    };
    let prompt_opened = inference_request
        .metadata
        .get(PROMPT_OPENED_REASONING_METADATA_KEY)
        .and_then(serde_json::Value::as_bool);
    let supported_boundary = match &inference_request
        .sampling_params
        .response_completion_boundary
    {
        ResponseCompletionBoundary::Immediate => true,
        ResponseCompletionBoundary::AfterDelimiterAndPayload {
            alternate_envelope, ..
        } => prompt_opened == Some(true) && alternate_envelope.is_none(),
    };
    if inference_request.sampling_params.model_output_protocol != ModelOutputProtocol::Text
        || !supported_boundary
        || inference_request.requires_structured_output()
        || !chat.tools.is_empty() || !chat.legacy_functions.is_empty() || chat.legacy_function_call.is_some()
        || chat.tool_choice.as_ref().is_some_and(|choice| !matches!(choice, ApiToolChoice::Mode(mode) if mode.eq_ignore_ascii_case("auto") || mode.eq_ignore_ascii_case("none")))
        || chat.response_format.as_ref().is_some_and(|format| format.format_type != "text" || format.json_schema.is_some())
        || prompt_opened.is_none()
    { return Err(unsupported()); }
    let include_usage = request
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let observer = evidence::Observer::new(
        &state,
        request.model.clone(),
        "/v1/chat/completions",
        correlation,
    )?;
    let contract = Arc::new(OutputProjectionContract::chat_sse(
        inference_request.id.to_string(),
        request.model,
        include_usage,
    ));
    let engine = state.llm.ok_or_else(|| {
        ServerError::ServiceUnavailable("LLM engine not loaded; chat unavailable".into())
    })?;
    let session = engine
        .infer_credited_stream(inference_request, context, contract)
        .await
        .map_err(server_error_from_ferrum_error)?;
    Ok(stream_response(session, observer))
}
