//! Proven output envelopes and move-only lifetime budgets.
//!
//! This module is an integration foundation, not a credited engine or HTTP
//! adapter. Its plans apply only when consumers use the codec defined here and
//! retain the projection grant for all declared history/scratch storage.

use crate::{
    output_credit::*,
    tokenizer::{BoundedDecodeBound, BoundedIncrementalDecodePolicy},
    Tokenizer,
};
use ferrum_types::{
    ApiRequest, FinishReason, InferenceRequest, ModelOutputProtocol, RequestId,
    ResponseCompletionBoundary, TokenUsage,
};
use std::sync::Arc;

mod budget;
mod chat_projection;
mod codec;
mod completion_plan;
mod evidence;
mod evidence_profile;
pub use chat_projection::{BoundedChatProjection, ChatOutputDelta, TextReasoningPolicy};
pub use completion_plan::{CompletionTokenIds, ResponseCompletionPlan};
pub use evidence::EngineEvidenceRetentionPlan;
pub use evidence_profile::{CreditedExecutionProfile, CreditedPromptEvidence};
mod session;
pub use budget::{
    OutputFrameAttempt, OutputFramePermit, PrepaidOutputCapacityView, RequestOutputBudget,
};
pub use session::{
    CreditedFrameStream, CreditedOutputFrame, CreditedOutputSession, OutputCompletion,
    OutputConsumerControl, OutputFrameMetadata, OutputHistory,
};

#[cfg(test)]
mod tests;

pub const MAX_OUTPUT_ERROR_BYTES: usize = 512;

#[derive(Debug, thiserror::Error)]
pub enum OutputFlowError {
    #[error("unsupported output contract: {0}")]
    Unsupported(&'static str),
    #[error("output budget arithmetic overflow")]
    Overflow,
    #[error("output contract exceeded its declared bound")]
    BoundExceeded,
    #[error("output budget has an outstanding frame")]
    FrameInFlight,
    #[error("output budget is closed")]
    Closed,
    #[error("output admission lacks capacity")]
    AdmissionFull,
    #[error("output serialization failed")]
    Serialization,
    #[error(transparent)]
    Credit(#[from] OutputCreditError),
}

/// Immutable, in-process capability; no serde and no caller-supplied expansion
/// multipliers. Constructors bind the exact codec and its fixed wire fields.
#[derive(Debug, Clone)]
pub struct OutputProjectionContract {
    kind: ProjectionKind,
}

#[derive(Debug, Clone)]
enum ProjectionKind {
    CliText,
    CompletionsSse {
        response_id: String,
        model: String,
        include_usage: bool,
    },
    ChatSse {
        response_id: String,
        model: String,
        include_usage: bool,
    },
}

/// Actual codec family selected by an admitted output plan. This is a cold
/// cost-policy descriptor, not permission to construct or encode that plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum OutputCodecKind {
    CliText,
    CompletionsSse {
        include_usage: bool,
    },
    ChatSse {
        include_usage: bool,
        prompt_opened: bool,
    },
}

/// No request identity or model spelling is retained. Envelope bounds are
/// measured by the very same serializer as the wire, so JSON escaping is
/// included even when two fixed strings have equal UTF-8 lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct OutputCodecDescriptor {
    pub kind: OutputCodecKind,
    /// Bump when encoding/assembly work or its accounting contract changes.
    pub implementation_version: u32,
    pub max_data_envelope_bytes: usize,
    pub max_terminal_envelope_bytes: usize,
}

impl OutputProjectionContract {
    /// Chat SSE for a resolved ordinary Text request. Derivation still checks
    /// the actual request, template reasoning state and usage policy.
    pub fn chat_sse(response_id: String, model: String, include_usage: bool) -> Self {
        Self {
            kind: ProjectionKind::ChatSse {
                response_id,
                model,
                include_usage,
            },
        }
    }
    /// Raw UTF-8 stdout payload, without JSONL, terminal decorations, or a
    /// reasoning/native-protocol transformation. Those need separate codecs.
    pub fn cli_text() -> Self {
        Self {
            kind: ProjectionKind::CliText,
        }
    }

    /// OpenAI Completions SSE with the literal id/model held by this contract.
    /// Includes finish and DONE, and usage only when explicitly requested by
    /// the owning endpoint's resolved protocol. Chat/Responses are distinct.
    pub fn completions_sse(response_id: String, model: String, include_usage: bool) -> Self {
        Self {
            kind: ProjectionKind::CompletionsSse {
                response_id,
                model,
                include_usage,
            },
        }
    }
}

/// A small UTF-8-safe error representation. Construction copies at most the
/// documented limit and never retains the unbounded original error String.
#[derive(Debug)]
pub struct BoundedOutputError {
    message: Box<str>,
}

impl BoundedOutputError {
    pub fn new(message: &str) -> Self {
        let mut end = message.len().min(MAX_OUTPUT_ERROR_BYTES);
        while !message.is_char_boundary(end) {
            end -= 1;
        }
        Self {
            message: message[..end].into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// The final payload is fixed-size metadata or a bounded error. Final text must
/// travel through a data frame, including a UTF-8/stop-prefix terminal flush.
pub enum OutputTerminal<'a> {
    Success {
        reason: FinishReason,
        usage: &'a TokenUsage,
        created: u64,
    },
    Error(&'a BoundedOutputError),
}

/// Derived from the actual engine request after context handling. This does
/// not lower max_tokens or treat unknown decoder behavior as sufficient.
#[derive(Debug, Clone)]
pub struct RequestOutputPlan {
    request_id: RequestId,
    contract: Arc<OutputProjectionContract>,
    max_tokens: usize,
    prompt_tokens: usize,
    decoder: BoundedDecodeBound,
    token_bytes_bound: usize,
    decoded_bytes: usize,
    semantic_bytes: usize,
    text_reasoning: Option<TextReasoningPolicy>,
    incremental: Option<BoundedIncrementalDecodePolicy>,
    completion: ResponseCompletionPlan,
    evidence: EngineEvidenceRetentionPlan,
    data_frames: usize,
    wire_bytes: usize,
    projection_bytes: usize,
    terminal: OutputCreditAmount,
}

impl RequestOutputPlan {
    pub fn derive(
        contract: Arc<OutputProjectionContract>,
        tokenizer: &dyn Tokenizer,
        request: &InferenceRequest,
        prompt_tokens: usize,
    ) -> Result<Self, OutputFlowError> {
        if request.sampling_params.model_output_protocol != ModelOutputProtocol::Text
            || request.requires_structured_output()
            || matches!(request.api_request.as_ref(), Some(ApiRequest::Completion(completion))
                if completion.response_format.as_ref().is_some_and(|format|
                    format.format_type != "text" || format.json_schema.is_some()))
        {
            return Err(OutputFlowError::Unsupported(
                "native and structured projection are not proved by this codec",
            ));
        }
        let text_reasoning = match &contract.kind {
            ProjectionKind::ChatSse { include_usage, .. } => {
                let Some(ApiRequest::Chat(chat)) = request.api_request.as_ref() else {
                    return Err(OutputFlowError::Unsupported(
                        "Chat codec requires an actual Chat request",
                    ));
                };
                if !request.stream || !chat.tools.is_empty() || !chat.legacy_functions.is_empty()
                    || chat.legacy_function_call.is_some()
                    || chat.tool_choice.as_ref().is_some_and(|choice| !matches!(choice,
                        ferrum_types::ApiToolChoice::Mode(mode) if mode.eq_ignore_ascii_case("auto") || mode.eq_ignore_ascii_case("none")))
                    || chat.response_format.as_ref().is_some_and(|format| format.format_type != "text" || format.json_schema.is_some())
                {
                    return Err(OutputFlowError::Unsupported("Chat codec requires streaming plain Text without tool or structured contracts"));
                }
                let requested_usage = chat
                    .stream_options
                    .as_ref()
                    .and_then(|options| options.include_usage)
                    .unwrap_or(false);
                if *include_usage != requested_usage {
                    return Err(OutputFlowError::Unsupported(
                        "Chat codec usage policy differs from the request",
                    ));
                }
                Some(TextReasoningPolicy::from_request(request)?)
            }
            _ if matches!(request.api_request.as_ref(), Some(ApiRequest::Chat(_))) => {
                return Err(OutputFlowError::Unsupported(
                    "Chat requires its typed projection codec",
                ));
            }
            _ => None,
        };
        if let ResponseCompletionBoundary::AfterDelimiterAndPayload {
            alternate_envelope, ..
        } = &request.sampling_params.response_completion_boundary
        {
            // Raw CLI text can retain alternate envelopes verbatim. Chat must
            // wait for its own bounded tools projection before admitting them.
            if text_reasoning
                .is_some_and(|policy| !policy.prompt_opened() || alternate_envelope.is_some())
                || (text_reasoning.is_none() && !matches!(&contract.kind, ProjectionKind::CliText))
            {
                return Err(OutputFlowError::Unsupported(
                    "this codec has no delayed completion envelope projection",
                ));
            }
        }
        let completion = ResponseCompletionPlan::derive(
            tokenizer,
            &request.sampling_params.response_completion_boundary,
        )?;
        let incremental = if completion.is_delayed() {
            Some(tokenizer.bounded_incremental_decode_policy().ok_or(
                OutputFlowError::Unsupported(
                    "tokenizer has no proved bounded incremental equivalence",
                ),
            )?)
        } else {
            None
        };
        let completion_storage = completion.retained_storage_bytes();
        let max_tokens = request.sampling_params.max_tokens;
        if max_tokens == 0 {
            return Err(OutputFlowError::Unsupported("zero effective max_tokens"));
        }
        let evidence = EngineEvidenceRetentionPlan::derive(
            &request.evidence_request,
            prompt_tokens,
            max_tokens,
        )?;
        let decoder = tokenizer
            .bounded_decode_bound()
            .ok_or(OutputFlowError::Unsupported(
                "tokenizer has no bounded decoder with caller-owned workspace",
            ))?;
        let decode_storage = decoder
            .requirements(max_tokens)
            .map_err(|_| OutputFlowError::Overflow)?;
        let token_bytes_bound = tokenizer
            .bounded_token_bytes_bound()
            .ok_or(OutputFlowError::Unsupported(
                "tokenizer has no bounded context-free token byte surface",
            ))?
            .get();
        // UTF-8 validation holds the token bytes and their concatenation with
        // the prior incomplete scalar (at most three bytes), plus old/new
        // pending fragments. This work does not overlap full-history decode.
        let utf8_workspace = token_bytes_bound
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(9))
            .ok_or(OutputFlowError::Overflow)?;
        let decoded_bytes = decode_storage.text_bytes;
        // One raw command per committed token and one terminal stop/UTF-8
        // flush. Chat can project each into reasoning and content events.
        // These counts bound lifetime envelope bytes, not reserved queue slots.
        let commands = max_tokens.checked_add(1).ok_or(OutputFlowError::Overflow)?;
        let data_frames = commands
            .checked_mul(if text_reasoning.is_some() { 2 } else { 1 })
            .ok_or(OutputFlowError::Overflow)?;
        let semantic_bytes = decoded_bytes
            .checked_mul(if text_reasoning.is_some() {
                ferrum_types::TEXT_REASONING_MAX_PROJECTION_EPOCHS
            } else {
                1
            })
            .ok_or(OutputFlowError::Overflow)?;
        let total_tokens = prompt_tokens
            .checked_add(max_tokens)
            .ok_or(OutputFlowError::Overflow)?;
        let usage = TokenUsage {
            prompt_tokens,
            completion_tokens: max_tokens,
            total_tokens,
        };
        let (wire_bytes, terminal) =
            codec::plan_wire(&contract, semantic_bytes, data_frames, &usage)?;
        // Named, simultaneous payload high-water marks: decoded output,
        // engine pending text, consumer retained text, consumer transform
        // scratch; engine history, immutable snapshot, consumer token history.
        // The bounded decoder's raw workspace is a separate live allocation;
        // a decoded-text length bound alone cannot cover decoder internals.
        // Chat additionally retains one raw arena and one concatenation
        // scratch arena, each bounded by decoded_bytes. No further copies may
        // allocate under this grant without another explicit storage bound.
        let text_storage = decoded_bytes
            .checked_mul(if text_reasoning.is_some() { 6 } else { 4 })
            // A strict incremental payload check holds previous/full decoded
            // strings simultaneously. Its delta is borrowed, never a third
            // allocation; full-decode scratch is reused sequentially.
            .and_then(|bytes| {
                bytes.checked_add(if incremental.is_some() {
                    decoded_bytes.checked_mul(2)?
                } else {
                    0
                })
            })
            .and_then(|bytes| bytes.checked_add(decode_storage.scratch_bytes.max(utf8_workspace)))
            .ok_or(OutputFlowError::Overflow)?;
        let token_storage = max_tokens
            .checked_mul(std::mem::size_of::<ferrum_types::TokenId>())
            .and_then(|bytes| bytes.checked_mul(if incremental.is_some() { 4 } else { 3 }))
            .ok_or(OutputFlowError::Overflow)?;
        let fixed_storage = match &contract.kind {
            ProjectionKind::CliText => 0,
            ProjectionKind::CompletionsSse {
                response_id, model, ..
            }
            | ProjectionKind::ChatSse {
                response_id, model, ..
            } => response_id
                .len()
                .checked_add(model.len())
                .ok_or(OutputFlowError::Overflow)?,
        };
        let stop_storage = request
            .sampling_params
            .stop_sequences
            .iter()
            .try_fold(0usize, |bytes, stop| bytes.checked_add(stop.len()))
            .and_then(|bytes| bytes.checked_mul(3))
            .ok_or(OutputFlowError::Overflow)?;
        let projection_bytes = text_storage
            .checked_add(token_storage)
            .and_then(|bytes| bytes.checked_add(fixed_storage))
            .and_then(|bytes| bytes.checked_add(stop_storage))
            .and_then(|bytes| bytes.checked_add(completion_storage))
            .and_then(|bytes| bytes.checked_add(evidence.retained_bytes()))
            // BoundedOutputError's retained Box<str> can coexist with the
            // serialized terminal Vec, which owns separate wire credit.
            .and_then(|bytes| bytes.checked_add(MAX_OUTPUT_ERROR_BYTES))
            .ok_or(OutputFlowError::Overflow)?;
        Ok(Self {
            request_id: request.id.clone(),
            contract,
            max_tokens,
            prompt_tokens,
            decoder,
            token_bytes_bound,
            decoded_bytes,
            semantic_bytes,
            text_reasoning,
            incremental,
            completion,
            evidence,
            data_frames,
            wire_bytes,
            projection_bytes,
            terminal,
        })
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }
    pub fn effective_max_tokens(&self) -> usize {
        self.max_tokens
    }
    pub fn max_decoded_bytes(&self) -> usize {
        self.decoded_bytes
    }
    pub fn max_semantic_bytes(&self) -> usize {
        self.semantic_bytes
    }
    pub fn text_reasoning_policy(&self) -> Option<TextReasoningPolicy> {
        self.text_reasoning
    }
    pub fn bounded_decoder(&self) -> BoundedDecodeBound {
        self.decoder
    }
    pub fn bounded_incremental_policy(&self) -> Option<BoundedIncrementalDecodePolicy> {
        self.incremental
    }
    pub fn completion_plan(&self) -> ResponseCompletionPlan {
        self.completion
    }
    /// Separate engine evidence storage, covered by the retained projection lease.
    pub fn evidence_plan(&self) -> EngineEvidenceRetentionPlan {
        self.evidence
    }
    pub fn max_token_bytes(&self) -> usize {
        self.token_bytes_bound
    }
    /// Compute once during admission, then cache in the host cost policy.
    /// No output payload is serialized or allocated by this counting pass.
    pub fn codec_descriptor(&self) -> Result<OutputCodecDescriptor, OutputFlowError> {
        let kind = match &self.contract.kind {
            ProjectionKind::CliText => OutputCodecKind::CliText,
            ProjectionKind::CompletionsSse { include_usage, .. } => {
                OutputCodecKind::CompletionsSse {
                    include_usage: *include_usage,
                }
            }
            ProjectionKind::ChatSse { include_usage, .. } => OutputCodecKind::ChatSse {
                include_usage: *include_usage,
                prompt_opened: self
                    .text_reasoning
                    .ok_or(OutputFlowError::Unsupported("missing admitted Chat policy"))?
                    .prompt_opened(),
            },
        };
        Ok(OutputCodecDescriptor {
            kind,
            implementation_version: 1,
            max_data_envelope_bytes: codec::max_data_envelope(&self.contract)?,
            max_terminal_envelope_bytes: self.terminal.bytes,
        })
    }
    pub fn max_data_frames(&self) -> usize {
        self.data_frames
    }
    pub fn lifetime_wire_bytes(&self) -> usize {
        self.wire_bytes
    }
    pub fn retained_projection_bytes(&self) -> usize {
        self.projection_bytes
    }
    pub fn terminal_credit(&self) -> OutputCreditAmount {
        self.terminal
    }
    /// One recyclable data event plus fixed terminal events; independent of N.
    pub fn minimum_event_capacity(&self) -> usize {
        self.terminal.events + 1
    }

    fn validate_usage(&self, usage: &TokenUsage) -> Result<(), OutputFlowError> {
        if usage.prompt_tokens != self.prompt_tokens
            || usage.completion_tokens > self.max_tokens
            || usage.prompt_tokens.checked_add(usage.completion_tokens) != Some(usage.total_tokens)
        {
            return Err(OutputFlowError::BoundExceeded);
        }
        Ok(())
    }
}
