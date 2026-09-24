//! Move-only product output. The wire codec has already run; consumers move
//! the lease into their transport and must not reserialize or clone payloads.

use super::BoundedOutputError;
use crate::output_credit::LeasedOutput;
use ferrum_types::{FinishReason, TokenId, TokenUsage};
use futures::Stream;
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputFrameMetadata {
    /// Output command ordinal, not executor work generation.
    pub ordinal: u64,
    pub token: Option<TokenId>,
    pub generated_tokens: usize,
    pub terminal: bool,
}

pub struct CreditedOutputFrame {
    wire: LeasedOutput<Vec<u8>>,
    metadata: OutputFrameMetadata,
}

impl CreditedOutputFrame {
    pub fn new(wire: LeasedOutput<Vec<u8>>, metadata: OutputFrameMetadata) -> Self {
        Self { wire, metadata }
    }
    pub fn metadata(&self) -> OutputFrameMetadata {
        self.metadata
    }
    pub fn wire(&self) -> &LeasedOutput<Vec<u8>> {
        &self.wire
    }
    pub fn into_wire(self) -> LeasedOutput<Vec<u8>> {
        self.wire
    }
}

pub struct OutputHistory {
    pub text: String,
    pub tokens: Vec<TokenId>,
}

pub enum OutputCompletion {
    Succeeded {
        history: Option<OutputHistory>,
        reason: FinishReason,
        usage: TokenUsage,
        /// Engine evidence is not part of the wire payload. Its complete lifetime
        /// remains covered by this completion's retained projection lease.
        execution_evidence: Option<ferrum_types::InferenceExecutionEvidence>,
    },
    Failed(BoundedOutputError),
}

/// Small engine-owned cancellation bridge. Called only when a wire consumer
/// drops before observing the terminal frame; it must not block.
pub trait OutputConsumerControl: Send + Sync {
    fn consumer_dropped(&self);
}

pub struct CreditedFrameStream {
    receiver: mpsc::Receiver<CreditedOutputFrame>,
    control: Arc<dyn OutputConsumerControl>,
    terminal_seen: bool,
}

impl Stream for CreditedFrameStream {
    type Item = CreditedOutputFrame;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let result = self.receiver.poll_recv(cx);
        if let Poll::Ready(Some(frame)) = &result {
            self.terminal_seen |= frame.metadata.terminal;
        }
        result
    }
}

impl Drop for CreditedFrameStream {
    fn drop(&mut self) {
        if !self.terminal_seen {
            self.control.consumer_dropped();
        }
    }
}

pub struct CreditedOutputSession {
    pub frames: CreditedFrameStream,
    /// Dropping this receiver does not cancel inference. A consumer retaining
    /// final history must retain this lease for that history's whole lifetime.
    pub completion: oneshot::Receiver<LeasedOutput<OutputCompletion>>,
}

impl CreditedOutputSession {
    pub fn from_receivers(
        frames: mpsc::Receiver<CreditedOutputFrame>,
        completion: oneshot::Receiver<LeasedOutput<OutputCompletion>>,
        control: Arc<dyn OutputConsumerControl>,
    ) -> Self {
        Self {
            frames: CreditedFrameStream {
                receiver: frames,
                control,
                terminal_seen: false,
            },
            completion,
        }
    }
}
