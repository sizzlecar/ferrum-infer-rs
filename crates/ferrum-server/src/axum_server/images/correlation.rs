//! Request metadata only. Service IDs come from the original job/error and
//! trace IDs from its actual collector; the caller only supplies an attempt ID.
use super::*;
use axum::http::{HeaderMap, HeaderValue};
use ferrum_types::image_trace::{
    parse_image_attempt_id, ImageRequestCorrelation, IMAGE_ATTEMPT_ID_HEADER,
    IMAGE_REQUEST_ID_HEADER, IMAGE_TRACE_SESSION_HEADER,
};
use uuid::Uuid;

pub(super) fn attempt(headers: &HeaderMap) -> Result<Option<Uuid>, ServerError> {
    if headers.contains_key(IMAGE_REQUEST_ID_HEADER)
        || headers.contains_key(IMAGE_TRACE_SESSION_HEADER)
    {
        return Err(ServerError::invalid_request(
            "image request and trace-session IDs are response-only metadata",
            None,
        ));
    }
    let all = headers.get_all(IMAGE_ATTEMPT_ID_HEADER);
    let mut values = all.iter();
    let value = values.next();
    if values.next().is_some() {
        return Err(ServerError::invalid_request(
            "image attempt ID must occur only once",
            None,
        ));
    }
    let value = value
        .map(|value| {
            value.to_str().map_err(|_| {
                ServerError::invalid_request("image attempt ID must be ASCII UUID metadata", None)
            })
        })
        .transpose()?;
    parse_image_attempt_id(value).map_err(|message| ServerError::invalid_request(message, None))
}

#[derive(Clone, Copy)]
pub(super) struct Context {
    correlation: Option<ImageRequestCorrelation>,
    trace_session: Option<Uuid>,
}
impl Context {
    pub(super) fn for_job(images: &ImageHttpService, correlation: ImageRequestCorrelation) -> Self {
        Self::new(images, Some(correlation))
    }

    fn new(images: &ImageHttpService, correlation: Option<ImageRequestCorrelation>) -> Self {
        Self {
            correlation,
            trace_session: images
                .service
                .continuous_trace_handle()
                .map(|handle| handle.session_uuid()),
        }
    }

    pub(super) fn response(self, mut response: Response) -> Response {
        let headers = response.headers_mut();
        let mut put = |name: &'static str, value: Uuid| {
            // Hyphenated UUIDs always fit an ASCII HTTP field value.
            headers.insert(name, HeaderValue::from_str(&value.to_string()).unwrap());
        };
        if let Some(correlation) = self.correlation {
            put(IMAGE_REQUEST_ID_HEADER, correlation.request_id);
            if let Some(attempt) = correlation.client_attempt_id {
                put(IMAGE_ATTEMPT_ID_HEADER, attempt);
            }
        }
        if let Some(session) = self.trace_session {
            put(IMAGE_TRACE_SESSION_HEADER, session);
        }
        response
    }

    pub(super) fn error(self, source: ServerError) -> HttpError {
        HttpError {
            source,
            context: Some(self),
        }
    }
}

pub(in crate::axum_server) struct HttpError {
    source: ServerError,
    context: Option<Context>,
}
impl From<ServerError> for HttpError {
    fn from(source: ServerError) -> Self {
        Self {
            source,
            context: None,
        }
    }
}
impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        let response = self.source.into_response();
        match self.context {
            Some(context) => context.response(response),
            None => response,
        }
    }
}

pub(super) fn service_failure(images: &ImageHttpService, source: ImageServiceError) -> HttpError {
    let context = Context::new(images, source.correlation());
    context.error(service_error(source))
}

#[cfg(test)]
mod tests;
