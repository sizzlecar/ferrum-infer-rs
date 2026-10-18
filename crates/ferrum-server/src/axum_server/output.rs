//! Pull-driven SSE encoding. The emitted bytes own their output credit.

use axum::{
    body::{Body, Bytes},
    http::header,
    response::{IntoResponse, Response},
};
use ferrum_output::{OutputBytesLease, OutputSession};
use ferrum_types::{FerrumError, Result};
use futures::Stream;
use http_body::{Body as HttpBody, Frame, SizeHint};
use serde::Serialize;
use std::{
    io::Write,
    pin::Pin,
    task::{Context, Poll},
};

mod semantic;
pub(super) use semantic::Retained;

pub(super) async fn json_response<T: Serialize + ?Sized>(
    session: &OutputSession,
    value: &T,
) -> Result<Response> {
    let bytes = encode(session, |writer| {
        serde_json::to_writer(writer, value).map_err(std::io::Error::other)
    })
    .await?;
    Ok((
        [(header::CONTENT_TYPE, "application/json")],
        Body::from(bytes),
    )
        .into_response())
}

struct EncodedFrame {
    bytes: Box<[u8]>,
    _encoded: OutputBytesLease,
    _session: OutputSession,
}

impl AsRef<[u8]> for EncodedFrame {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

// This outer owner survives an inner generator consuming its state on EOF.
// Bytes returned before body drop independently retain the same session.
struct OutputBody {
    inner: Body,
    _session: OutputSession,
}

impl HttpBody for OutputBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<std::result::Result<Frame<Bytes>, axum::Error>>> {
        Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

pub(super) fn sse_response(
    session: OutputSession,
    frames: impl Stream<Item = Result<Bytes>> + Send + 'static,
) -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        Body::new(OutputBody {
            inner: Body::from_stream(frames),
            _session: session,
        }),
    )
        .into_response()
}

struct CountWriter {
    len: usize,
    limit: usize,
    exceeded: bool,
}

impl Write for CountWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let Some(next) = self
            .len
            .checked_add(bytes.len())
            .filter(|n| *n <= self.limit)
        else {
            self.exceeded = true;
            return Err(std::io::Error::other(
                "SSE event exceeds configured output limit",
            ));
        };
        self.len = next;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn write_event(
    writer: &mut dyn Write,
    name: Option<&str>,
    write_data: impl FnOnce(&mut dyn Write) -> std::io::Result<()>,
) -> std::io::Result<()> {
    if let Some(name) = name {
        if name.bytes().any(|b| matches!(b, b'\n' | b'\r')) {
            return Err(std::io::Error::other("invalid SSE event name"));
        }
        writer.write_all(b"event: ")?;
        writer.write_all(name.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.write_all(b"data: ")?;
    write_data(writer)?;
    writer.write_all(b"\n\n")
}

async fn encode(
    session: &OutputSession,
    write: impl Fn(&mut dyn Write) -> std::io::Result<()>,
) -> Result<Bytes> {
    let count = encoded_size(&write, session.config().max_event_bytes)?;
    let lease = session.reserve_encoded(count).await?;
    encode_reserved(session, &write, count, lease)
}

fn encoded_size(
    write: &impl Fn(&mut dyn Write) -> std::io::Result<()>,
    limit: usize,
) -> Result<usize> {
    let mut counter = CountWriter {
        len: 0,
        limit,
        exceeded: false,
    };
    if let Err(error) = write(&mut counter) {
        return Err(if counter.exceeded {
            FerrumError::resource_exhausted("SSE event exceeds configured output limit")
        } else {
            FerrumError::internal(format!("SSE encoding: {error}"))
        });
    }
    Ok(counter.len)
}

fn encode_reserved(
    session: &OutputSession,
    write: &impl Fn(&mut dyn Write) -> std::io::Result<()>,
    count: usize,
    lease: OutputBytesLease,
) -> Result<Bytes> {
    let mut bytes = vec![0_u8; count].into_boxed_slice();
    let mut cursor = std::io::Cursor::new(bytes.as_mut());
    write(&mut cursor).map_err(|error| FerrumError::internal(format!("SSE encoding: {error}")))?;
    if cursor.position() != count as u64 {
        return Err(FerrumError::internal("SSE event changed during encoding"));
    }
    Ok(Bytes::from_owner(EncodedFrame {
        bytes,
        _encoded: lease,
        _session: session.clone(),
    }))
}

pub(super) async fn encode_json<T: Serialize + ?Sized>(
    session: &OutputSession,
    name: Option<&str>,
    value: &T,
) -> Result<Bytes> {
    encode(session, |writer| {
        write_event(writer, name, |writer| {
            serde_json::to_writer(writer, value).map_err(std::io::Error::other)
        })
    })
    .await
}

pub(super) async fn encode_done(session: &OutputSession) -> Result<Bytes> {
    encode(session, |writer| {
        write_event(writer, None, |writer| writer.write_all(b"[DONE]"))
    })
    .await
}

pub(super) fn encoding_error_details(error: &FerrumError) -> (&'static str, &'static str) {
    match error {
        FerrumError::ResourceExhausted { .. } => (
            "output_limit_exceeded",
            "Output event exceeds the configured byte limit.",
        ),
        _ => ("internal_server_error", "Output event encoding failed."),
    }
}

/// This one terminal frame never waits for normal data credit. Its fixed
/// allowance is consumed once, and its bytes still retain the original session.
pub(super) fn encode_terminal_error<T: Serialize + ?Sized>(
    session: &OutputSession,
    name: Option<&str>,
    value: &T,
) -> Result<Bytes> {
    let write = |writer: &mut dyn Write| {
        write_event(writer, name, |writer| {
            serde_json::to_writer(writer, value).map_err(std::io::Error::other)
        })
    };
    let count = encoded_size(&write, ferrum_output::TERMINAL_ERROR_BYTES)?;
    let lease = session.try_reserve_error(count)?;
    encode_reserved(session, &write, count, lease)
}

/// Known upstream failures use the control allowance directly, even when all
/// normal encoded credit is held by a slow client. The existing failure wire
/// includes its final DONE delimiter in this same bounded allocation.
pub(super) fn encode_upstream_failure<T: Serialize + ?Sized>(
    session: &OutputSession,
    name: Option<&str>,
    value: &T,
    compact: &T,
) -> Result<Bytes> {
    let write_value = |writer: &mut dyn Write, value: &T| {
        write_event(writer, name, |writer| {
            serde_json::to_writer(writer, value).map_err(std::io::Error::other)
        })?;
        write_event(writer, None, |writer| writer.write_all(b"[DONE]"))
    };
    let full = |writer: &mut dyn Write| write_value(writer, value);
    let compact = |writer: &mut dyn Write| write_value(writer, compact);
    let (write, count): (&dyn Fn(&mut dyn Write) -> std::io::Result<()>, _) =
        match encoded_size(&full, ferrum_output::TERMINAL_ERROR_BYTES) {
            Ok(count) => (&full, count),
            Err(FerrumError::ResourceExhausted { .. }) => {
                let count = encoded_size(&compact, ferrum_output::TERMINAL_ERROR_BYTES)?;
                (&compact, count)
            }
            Err(error) => return Err(error),
        };
    let lease = session.try_reserve_error(count)?;
    encode_reserved(session, &write, count, lease)
}

pub(super) const OMITTED_UPSTREAM_DIAGNOSTIC: &str =
    "Upstream stream failed; diagnostic message omitted.";

#[derive(Serialize, Clone, Copy)]
pub(super) struct UpstreamError<'a> {
    pub(super) message: &'a str,
    #[serde(rename = "type")]
    pub(super) error_type: &'a str,
    pub(super) param: Option<&'a str>,
    pub(super) code: Option<&'a str>,
}

pub(super) fn upstream_error(error: &crate::openai::OpenAiErrorDetail) -> UpstreamError<'_> {
    // Preserve the server's bounded protocol classifications verbatim. An
    // oversized/invalid classification remains an upstream internal failure,
    // never a fabricated output-limit error. Optional oversized params are
    // omitted; they cannot consume the entire terminal allowance.
    let field = |value: &str| {
        value.len() <= 64
            && !value.is_empty()
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_.-[]".contains(&byte))
    };
    UpstreamError {
        message: &error.message,
        error_type: if field(&error.error_type) {
            &error.error_type
        } else {
            "internal_server_error"
        },
        param: error.param.as_deref().filter(|value| field(value)),
        code: error.code.as_deref().map(|value| {
            if field(value) {
                value
            } else {
                "upstream_error"
            }
        }),
    }
}

pub(super) fn encode_chat_failure(
    session: &OutputSession,
    error: &crate::openai::OpenAiError,
) -> Result<Bytes> {
    #[derive(Serialize)]
    struct Envelope<'a> {
        error: UpstreamError<'a>,
    }
    let error = upstream_error(&error.error);
    let compact = UpstreamError {
        message: OMITTED_UPSTREAM_DIAGNOSTIC,
        ..error
    };
    encode_upstream_failure(
        session,
        None,
        &Envelope { error },
        &Envelope { error: compact },
    )
}

/// Returns whether the frame is a terminal failure; callers must then stop
/// pulling the source and must not append a successful terminal event.
pub(super) fn chat_encoding_result(
    session: &OutputSession,
    encoded: Result<Bytes>,
) -> Result<(Bytes, bool)> {
    match encoded {
        Ok(bytes) => Ok((bytes, false)),
        Err(error) => {
            let (code, message) = encoding_error_details(&error);
            let value = serde_json::json!({"error": {
                "message": message, "type": code, "param": null, "code": code
            }});
            Ok((encode_terminal_error(session, None, &value)?, true))
        }
    }
}

#[cfg(test)]
mod tests;
