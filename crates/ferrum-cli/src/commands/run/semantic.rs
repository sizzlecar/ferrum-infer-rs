//! Owners for generated content retained beyond a streamed chunk.

use super::*;
use ferrum_output::{semantic_cost, InferenceOutput, OutputSemanticLease, OutputSession};
use ferrum_types::{InferenceExecutionEvidence, TokenId};
use serde::Serialize;

#[derive(Debug)]
pub(super) struct RunHistoryMessage {
    pub(super) prompt: PromptMessage,
    pub(super) raw_content: String,
    semantic: Option<OutputSemanticLease>,
}

impl RunHistoryMessage {
    /// Input history is distinct from an assistant result produced this session.
    pub(super) fn new(role: &str, content: &str) -> Self {
        Self {
            prompt: PromptMessage::new(role, content),
            raw_content: content.to_owned(),
            semantic: None,
        }
    }

    pub(super) fn assistant(
        session: &OutputSession,
        raw: &str,
        protocol: ModelOutputProtocol,
        parsed: &ParsedReasoningResponse,
    ) -> Result<Self> {
        let (content, reasoning) = match protocol {
            ModelOutputProtocol::Text => (raw, None),
            ModelOutputProtocol::HarmonyGptOss | ModelOutputProtocol::GemmaThought => {
                (parsed.content.as_str(), parsed.reasoning.as_deref())
            }
        };
        // Reserve both independent copies before cloning. The entry, rather
        // than a request-id side table, owns their credit until actual removal.
        let semantic = session.try_reserve_semantic(semantic_cost(&(
            "assistant",
            raw,
            content,
            reasoning,
        ))?)?;
        let mut prompt = PromptMessage::new("assistant", content);
        prompt.reasoning_content = reasoning.map(str::to_owned);
        Ok(Self {
            prompt,
            raw_content: raw.to_owned(),
            semantic: Some(semantic),
        })
    }

    pub(super) fn reserve_prompt_copy(&self) -> Result<Option<OutputSemanticLease>> {
        self.semantic
            .as_ref()
            .map(|lease| {
                lease.output_session().try_reserve_semantic(semantic_cost(&(
                    self.prompt.role.as_str(),
                    self.prompt.content.as_str(),
                    self.prompt.reasoning_content.as_deref(),
                ))?)
            })
            .transpose()
    }
}

#[derive(Serialize)]
pub(super) struct StreamGenerationData {
    pub(super) request_id: RequestId,
    pub(super) raw_text: String,
    pub(super) finish_reason: Option<FinishReason>,
    pub(super) usage: Option<TokenUsage>,
    pub(super) token_count: usize,
    pub(super) token_ids: Vec<TokenId>,
    pub(super) chunk_count: usize,
    pub(super) execution_evidence: Option<InferenceExecutionEvidence>,
}

pub(super) fn stream_reservation(
    session: &OutputSession,
    request_id: &RequestId,
) -> Result<OutputSemanticLease> {
    // Count fixed fields and empty containers once. Subsequent updates charge
    // only their own added content, never the already retained history.
    session.try_reserve_semantic(semantic_cost(&StreamGenerationData {
        request_id: request_id.clone(),
        raw_text: String::new(),
        finish_reason: None,
        usage: None,
        token_count: 0,
        token_ids: Vec::new(),
        chunk_count: 0,
        execution_evidence: None,
    })?)
}

pub(super) enum CollectedRunGeneration {
    Response(InferenceOutput),
    Stream {
        data: StreamGenerationData,
        semantic: OutputSemanticLease,
    },
}

pub(super) struct RunGenerationView<'a> {
    pub(super) request_id: &'a RequestId,
    pub(super) raw_text: &'a str,
    pub(super) finish_reason: Option<FinishReason>,
    pub(super) usage: Option<&'a TokenUsage>,
    pub(super) token_count: usize,
    pub(super) token_ids: &'a [TokenId],
    pub(super) chunk_count: usize,
    pub(super) execution_evidence: Option<&'a InferenceExecutionEvidence>,
}

impl CollectedRunGeneration {
    pub(super) fn from_response(output: InferenceOutput) -> Result<Self> {
        require_run_terminal_reason(Some(output.response().finish_reason))?;
        Ok(Self::Response(output))
    }

    pub(super) fn session(&self) -> &OutputSession {
        match self {
            Self::Response(output) => output.output_session(),
            Self::Stream { semantic, .. } => semantic.output_session(),
        }
    }

    pub(super) fn view(&self) -> RunGenerationView<'_> {
        match self {
            Self::Response(output) => {
                let response = output.response();
                RunGenerationView {
                    request_id: &response.request_id,
                    raw_text: &response.text,
                    finish_reason: Some(response.finish_reason),
                    usage: Some(&response.usage),
                    token_count: response.tokens.len(),
                    token_ids: &response.tokens,
                    chunk_count: 1,
                    execution_evidence: response.execution_evidence.as_ref(),
                }
            }
            Self::Stream { data, .. } => RunGenerationView {
                request_id: &data.request_id,
                raw_text: &data.raw_text,
                finish_reason: data.finish_reason,
                usage: data.usage.as_ref(),
                token_count: data.token_count,
                token_ids: &data.token_ids,
                chunk_count: data.chunk_count,
                execution_evidence: data.execution_evidence.as_ref(),
            },
        }
    }
}

pub(super) struct RetainedRunProjection {
    pub(super) parsed: ParsedReasoningResponse,
    _semantic: OutputSemanticLease,
}

impl RetainedRunProjection {
    pub(super) fn new(session: &OutputSession, parsed: ParsedReasoningResponse) -> Result<Self> {
        let semantic = session.try_reserve_semantic(semantic_cost(&(
            parsed.content.as_str(),
            parsed.reasoning.as_deref(),
        ))?)?;
        Ok(Self {
            parsed,
            _semantic: semantic,
        })
    }
}

/// Replace a retained optional field without cumulative overcharging.
pub(super) fn resize_field<T: Serialize + ?Sized>(
    lease: &mut OutputSemanticLease,
    old: &T,
    new: &T,
) -> Result<()> {
    let old = semantic_cost(old)?;
    let new = semantic_cost(new)?;
    let next = lease
        .bytes()
        .checked_sub(old)
        .and_then(|n| n.checked_add(new))
        .ok_or_else(|| FerrumError::internal("run semantic field size overflow"))?;
    lease.try_resize(next)
}
