//! Bounded, request-local Text reasoning projection. The owner keeps this
//! storage under its retained projection grant through cancellation/terminal.
use super::*;
use ferrum_types::{parse_text_reasoning_view, should_defer_model_reasoning_stream_delta};
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct TextReasoningPolicy {
    prompt_opened: bool,
}
impl TextReasoningPolicy {
    pub fn prompt_opened(self) -> bool {
        self.prompt_opened
    }
    pub(super) fn from_request(request: &InferenceRequest) -> Result<Self, OutputFlowError> {
        let prompt_opened = request
            .metadata
            .get(ferrum_types::PROMPT_OPENED_REASONING_METADATA_KEY)
            .and_then(serde_json::Value::as_bool)
            .ok_or(OutputFlowError::Unsupported(
                "Chat requires resolved template reasoning state",
            ))?;
        Ok(Self { prompt_opened })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatOutputDelta<'a> {
    Reasoning(&'a str),
    Content(&'a str),
}
impl ChatOutputDelta<'_> {
    pub fn text(&self) -> &str {
        match self {
            Self::Reasoning(text) | Self::Content(text) => text,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Channel {
    Reasoning,
    Content,
}

/// At most one prepared projection; the consumer must drain it before another
/// raw delta is accepted. No model/token frontier is owned by this helper.
pub struct BoundedChatProjection {
    policy: TextReasoningPolicy,
    maximum: usize,
    raw: String,
    scratch: String,
    sent_content: usize,
    sent_reasoning: usize,
    content: Range<usize>,
    reasoning: Range<usize>,
    pending: Option<Channel>,
    terminal_prepared: bool,
}
impl BoundedChatProjection {
    /// Allocate only after RequestOutputBudget::open succeeds. Both actual
    /// capacities are included separately in the plan's Chat projection bound.
    pub fn new(plan: &RequestOutputPlan) -> Result<Self, OutputFlowError> {
        let policy = plan
            .text_reasoning_policy()
            .ok_or(OutputFlowError::Unsupported("not a Chat projection plan"))?;
        let maximum = plan.max_decoded_bytes();
        let raw = String::with_capacity(maximum);
        let scratch = String::with_capacity(maximum);
        if raw.capacity() > maximum || scratch.capacity() > maximum {
            return Err(OutputFlowError::BoundExceeded);
        }
        Ok(Self {
            policy,
            maximum,
            raw,
            scratch,
            sent_content: 0,
            sent_reasoning: 0,
            content: 0..0,
            reasoning: 0..0,
            pending: None,
            terminal_prepared: false,
        })
    }

    pub fn raw_text(&self) -> &str {
        &self.raw
    }
    pub fn retained_capacity(&self) -> usize {
        self.raw.capacity() + self.scratch.capacity()
    }
    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Empty text can still finalize a previously deferred marker. Capacity
    /// rejection precedes mutation; a terminal may be prepared exactly once.
    pub fn prepare(&mut self, text: &str, terminal: bool) -> Result<(), OutputFlowError> {
        if self.terminal_prepared {
            return Err(OutputFlowError::Closed);
        }
        if self.has_pending() {
            return Err(OutputFlowError::FrameInFlight);
        }
        if text.len() > self.maximum.saturating_sub(self.raw.len()) {
            return Err(OutputFlowError::BoundExceeded);
        }
        self.raw.push_str(text);
        self.terminal_prepared = terminal;
        if !terminal
            && should_defer_model_reasoning_stream_delta(ModelOutputProtocol::Text, &self.raw)
        {
            return Ok(());
        }
        let view = parse_text_reasoning_view(&self.raw, self.policy.prompt_opened);
        if view
            .content_bytes()
            .checked_add(view.reasoning().map_or(0, str::len))
            .is_none_or(|bytes| bytes > self.maximum)
        {
            return Err(OutputFlowError::BoundExceeded);
        }
        self.scratch.clear();
        for part in view.content_parts() {
            self.scratch.push_str(part);
        }
        let content_end = self.scratch.len();
        if let Some(reasoning) = view.reasoning() {
            self.scratch.push_str(reasoning);
        }
        debug_assert!(self.scratch.len() <= self.raw.len());
        self.content = delta_range(&self.scratch[..content_end], 0, &mut self.sent_content);
        self.reasoning = delta_range(
            &self.scratch[content_end..],
            content_end,
            &mut self.sent_reasoning,
        );
        self.pending = if !self.reasoning.is_empty() {
            Some(Channel::Reasoning)
        } else if !self.content.is_empty() {
            Some(Channel::Content)
        } else {
            None
        };
        Ok(())
    }

    pub fn pending_delta(&self) -> Option<ChatOutputDelta<'_>> {
        match self.pending? {
            Channel::Reasoning => Some(ChatOutputDelta::Reasoning(
                &self.scratch[self.reasoning.clone()],
            )),
            Channel::Content => Some(ChatOutputDelta::Content(
                &self.scratch[self.content.clone()],
            )),
        }
    }
    /// Called only after the current frame has been encoded and accepted by
    /// its reserved wire slot. Event pressure leaves this cursor unchanged.
    pub fn advance(&mut self) -> Result<(), OutputFlowError> {
        self.pending = match self
            .pending
            .take()
            .ok_or(OutputFlowError::Unsupported("no pending Chat frame"))?
        {
            Channel::Reasoning if !self.content.is_empty() => Some(Channel::Content),
            _ => None,
        };
        Ok(())
    }
}

fn delta_range(text: &str, offset: usize, sent: &mut usize) -> Range<usize> {
    let start = if *sent <= text.len() && text.is_char_boundary(*sent) {
        *sent
    } else {
        text.len()
    };
    *sent = text.len();
    offset + start..offset + text.len()
}
