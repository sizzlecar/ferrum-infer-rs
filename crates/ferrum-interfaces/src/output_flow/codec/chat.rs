//! One borrowed serializer for both Chat SSE proof and actual wire bytes.
use super::*;

#[derive(Serialize)]
struct ChatChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: &'a [ChatChoice<'a>],
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}
#[derive(Serialize)]
struct ChatChoice<'a> {
    index: u32,
    delta: ChatDelta<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}
#[derive(Serialize)]
struct ChatDelta<'a> {
    role: &'static str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<&'a str>,
}

pub(super) fn data(
    out: &mut impl Write,
    contract: &OutputProjectionContract,
    delta: ChatOutputDelta<'_>,
    created: u64,
) -> Result<(), OutputFlowError> {
    let ProjectionKind::ChatSse {
        response_id, model, ..
    } = &contract.kind
    else {
        return Err(OutputFlowError::Unsupported(
            "Chat delta requires its Chat codec",
        ));
    };
    let delta = match delta {
        ChatOutputDelta::Content(content) => ChatDelta {
            role: "assistant",
            content,
            reasoning: None,
        },
        ChatOutputDelta::Reasoning(reasoning) => ChatDelta {
            role: "assistant",
            content: "",
            reasoning: Some(reasoning),
        },
    };
    sse(
        out,
        &ChatChunk {
            id: response_id,
            object: "chat.completion.chunk",
            created,
            model,
            choices: &[ChatChoice {
                index: 0,
                delta,
                finish_reason: None,
            }],
            usage: None,
        },
    )
}

pub(super) fn terminal(
    out: &mut impl Write,
    contract: &OutputProjectionContract,
    terminal: &OutputTerminal<'_>,
) -> Result<(), OutputFlowError> {
    let ProjectionKind::ChatSse {
        response_id,
        model,
        include_usage,
    } = &contract.kind
    else {
        return Err(OutputFlowError::Unsupported(
            "Chat terminal requires its Chat codec",
        ));
    };
    match terminal {
        OutputTerminal::Success {
            reason,
            usage,
            created,
        } => {
            sse(
                out,
                &ChatChunk {
                    id: response_id,
                    object: "chat.completion.chunk",
                    created: *created,
                    model,
                    choices: &[ChatChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: "assistant",
                            content: "",
                            reasoning: None,
                        },
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
                    &ChatChunk {
                        id: response_id,
                        object: "chat.completion.chunk",
                        created: *created,
                        model,
                        choices: &[],
                        usage: Some(usage),
                    },
                )?;
            }
        }
        OutputTerminal::Error(error) => sse(
            out,
            &ErrorEnvelope {
                error: ErrorDetail {
                    message: error.message(),
                    kind: "internal_server_error",
                    param: None,
                    code: None,
                },
            },
        )?,
    }
    out.write_all(DONE)
        .map_err(|_| OutputFlowError::BoundExceeded)
}
