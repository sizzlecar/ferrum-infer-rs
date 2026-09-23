//! The one serializer used both to prove budgets and to materialize frames.
use super::*;
use serde::Serialize;
use std::io::{self, Write};
mod chat;

const DONE: &[u8] = b"data: [DONE]\n\n";

#[derive(Serialize)]
struct Completion<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: &'a [Choice<'a>],
    usage: Option<Usage>,
}
#[derive(Serialize)]
struct Choice<'a> {
    text: &'a str,
    index: u32,
    finish_reason: Option<&'static str>,
}
#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}
#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    error: ErrorDetail<'a>,
}
#[derive(Serialize)]
struct ErrorDetail<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    param: Option<&'static str>,
    code: Option<&'static str>,
}

fn finish(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Length => "length",
        FinishReason::Stop | FinishReason::EOS => "stop",
        FinishReason::Cancelled => "cancelled",
        FinishReason::Error => "error",
        FinishReason::ContentFilter => "content_filter",
    }
}

fn sse(out: &mut impl Write, value: &impl Serialize) -> Result<(), OutputFlowError> {
    out.write_all(b"data: ")
        .map_err(|_| OutputFlowError::BoundExceeded)?;
    serde_json::to_writer(&mut *out, value).map_err(|_| OutputFlowError::Serialization)?;
    out.write_all(b"\n\n")
        .map_err(|_| OutputFlowError::BoundExceeded)
}

pub(super) fn data(
    out: &mut impl Write,
    contract: &OutputProjectionContract,
    text: &str,
    created: u64,
) -> Result<(), OutputFlowError> {
    match &contract.kind {
        ProjectionKind::ChatSse { .. } => {
            chat::data(out, contract, ChatOutputDelta::Content(text), created)
        }
        ProjectionKind::CliText => out
            .write_all(text.as_bytes())
            .map_err(|_| OutputFlowError::BoundExceeded),
        ProjectionKind::CompletionsSse {
            response_id, model, ..
        } => sse(
            out,
            &Completion {
                id: response_id,
                object: "text_completion",
                created,
                model,
                choices: &[Choice {
                    text,
                    index: 0,
                    finish_reason: None,
                }],
                usage: None,
            },
        ),
    }
}

pub(super) fn terminal(
    out: &mut impl Write,
    contract: &OutputProjectionContract,
    terminal: &OutputTerminal<'_>,
) -> Result<(), OutputFlowError> {
    match (&contract.kind, terminal) {
        (ProjectionKind::ChatSse { .. }, _) => chat::terminal(out, contract, terminal),
        (ProjectionKind::CliText, OutputTerminal::Success { .. }) => Ok(()),
        (ProjectionKind::CliText, OutputTerminal::Error(error)) => out
            .write_all(b"Error: ")
            .and_then(|_| out.write_all(error.message().as_bytes()))
            .and_then(|_| out.write_all(b"\n"))
            .map_err(|_| OutputFlowError::BoundExceeded),
        (
            ProjectionKind::CompletionsSse {
                response_id,
                model,
                include_usage,
            },
            OutputTerminal::Success {
                reason,
                usage,
                created,
            },
        ) => {
            sse(
                out,
                &Completion {
                    id: response_id,
                    object: "text_completion",
                    created: *created,
                    model,
                    choices: &[Choice {
                        text: "",
                        index: 0,
                        finish_reason: Some(finish(*reason)),
                    }],
                    usage: None,
                },
            )?;
            if *include_usage {
                let usage = Usage {
                    prompt_tokens: u32::try_from(usage.prompt_tokens)
                        .map_err(|_| OutputFlowError::Unsupported("OpenAI usage exceeds u32"))?,
                    completion_tokens: u32::try_from(usage.completion_tokens)
                        .map_err(|_| OutputFlowError::Unsupported("OpenAI usage exceeds u32"))?,
                    total_tokens: u32::try_from(usage.total_tokens)
                        .map_err(|_| OutputFlowError::Unsupported("OpenAI usage exceeds u32"))?,
                };
                sse(
                    out,
                    &Completion {
                        id: response_id,
                        object: "text_completion",
                        created: *created,
                        model,
                        choices: &[],
                        usage: Some(usage),
                    },
                )?;
            }
            out.write_all(DONE)
                .map_err(|_| OutputFlowError::BoundExceeded)
        }
        (ProjectionKind::CompletionsSse { .. }, OutputTerminal::Error(error)) => {
            sse(
                out,
                &ErrorEnvelope {
                    error: ErrorDetail {
                        message: error.message(),
                        kind: "internal_server_error",
                        param: None,
                        code: None,
                    },
                },
            )?;
            out.write_all(DONE)
                .map_err(|_| OutputFlowError::BoundExceeded)
        }
    }
}

pub(super) fn chat_data(
    out: &mut impl Write,
    contract: &OutputProjectionContract,
    delta: ChatOutputDelta<'_>,
    created: u64,
) -> Result<(), OutputFlowError> {
    chat::data(out, contract, delta, created)
}

pub(super) fn max_data_envelope(
    contract: &OutputProjectionContract,
) -> Result<usize, OutputFlowError> {
    let content = count(|out| data(out, contract, "", u64::MAX))?;
    if matches!(contract.kind, ProjectionKind::ChatSse { .. }) {
        Ok(content.max(count(|out| {
            chat::data(out, contract, ChatOutputDelta::Reasoning(""), u64::MAX)
        })?))
    } else {
        Ok(content)
    }
}

pub(super) fn plan_wire(
    contract: &OutputProjectionContract,
    text_bytes: usize,
    frames: usize,
    usage: &TokenUsage,
) -> Result<(usize, OutputCreditAmount), OutputFlowError> {
    let overhead = max_data_envelope(contract)?;
    let expansion = match contract.kind {
        ProjectionKind::CliText => 1,
        ProjectionKind::CompletionsSse { .. } | ProjectionKind::ChatSse { .. } => 6,
    };
    let wire_bytes = text_bytes
        .checked_mul(expansion)
        .and_then(|bytes| {
            overhead
                .checked_mul(frames)
                .and_then(|fixed| bytes.checked_add(fixed))
        })
        .ok_or(OutputFlowError::Overflow)?;
    let success = count(|out| {
        terminal(
            out,
            contract,
            &OutputTerminal::Success {
                reason: FinishReason::ContentFilter,
                usage,
                created: u64::MAX,
            },
        )
    })?;
    // A UTF-8 byte needs at most six JSON bytes. NUL achieves that bound;
    // counting the actual fixed serializer avoids guessed envelope constants.
    let error = BoundedOutputError::new(&"\0".repeat(MAX_OUTPUT_ERROR_BYTES));
    let failure = count(|out| terminal(out, contract, &OutputTerminal::Error(&error)))?;
    let events = match contract.kind {
        ProjectionKind::CliText => 1,
        ProjectionKind::CompletionsSse { include_usage, .. }
        | ProjectionKind::ChatSse { include_usage, .. } => 2 + usize::from(include_usage),
    };
    Ok((
        wire_bytes,
        OutputCreditAmount {
            events,
            bytes: success.max(failure),
            projection_bytes: 0,
        },
    ))
}

pub(super) struct Counter {
    used: usize,
}
impl Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.used = self
            .used
            .checked_add(bytes.len())
            .ok_or_else(|| io::Error::other("output count overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(super) fn count(
    write: impl FnOnce(&mut Counter) -> Result<(), OutputFlowError>,
) -> Result<usize, OutputFlowError> {
    let mut counter = Counter { used: 0 };
    write(&mut counter)?;
    Ok(counter.used)
}

pub(super) struct BoundedBuffer {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for BoundedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("output buffer limit exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(super) fn encode(
    limit: usize,
    write: impl FnOnce(&mut BoundedBuffer) -> Result<(), OutputFlowError>,
) -> Result<Vec<u8>, OutputFlowError> {
    // The caller already owns this byte grant. Avoid Vec's geometric growth:
    // the transport owner accounts actual retained capacity, not merely len.
    let bytes = Vec::with_capacity(limit);
    if bytes.capacity() > limit {
        return Err(OutputFlowError::BoundExceeded);
    }
    let mut buffer = BoundedBuffer { bytes, limit };
    write(&mut buffer)?;
    Ok(buffer.bytes)
}
