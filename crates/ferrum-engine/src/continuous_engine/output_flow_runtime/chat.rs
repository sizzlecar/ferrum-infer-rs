//! One accepted output command can own several ordered Chat data frames.
//! The model frontier advances once; this cursor never authorizes model work.
use super::*;
use ferrum_interfaces::output_flow::BoundedChatProjection;

type FrameParts = (OutputFramePermit, mpsc::OwnedPermit<CreditedOutputFrame>);

pub(super) struct PendingProjection {
    metadata: OutputFrameMetadata,
    created: u64,
    first: Option<FrameParts>,
}

impl Owner {
    pub(super) fn initialize_chat_projection(&mut self) -> Result<(), OutputCloseReason> {
        if self.budget.plan().text_reasoning_policy().is_some() {
            self.chat_projection = Some(
                BoundedChatProjection::new(self.budget.plan())
                    .map_err(|_| OutputCloseReason::InvalidOutput)?,
            );
        }
        Ok(())
    }

    pub(super) fn chat_command(
        &mut self,
        ordinal: u64,
        frame: OutputFramePermit,
        wire_slot: mpsc::OwnedPermit<CreditedOutputFrame>,
        delta: OutputDelta,
    ) -> Result<(), OutputCloseReason> {
        if self.pending_projection.is_some() {
            return Err(OutputCloseReason::InvalidOutput);
        }
        self.chat_projection
            .as_mut()
            .ok_or(OutputCloseReason::InvalidOutput)?
            .prepare(&delta.text, false)
            .map_err(|_| OutputCloseReason::InvalidOutput)?;
        self.applied = ordinal;
        self.generated_tokens = delta.generated_tokens;
        self.pending_projection = Some(PendingProjection {
            metadata: OutputFrameMetadata {
                ordinal,
                token: delta.token,
                generated_tokens: delta.generated_tokens,
                terminal: false,
            },
            created: delta.created,
            first: Some((frame, wire_slot)),
        });
        Ok(())
    }

    /// None means progress: retry the owner loop (including terminal/cancel
    /// checks). Some(wait) leaves the cursor intact and services the ordinary
    /// owner control select while transport capacity is unavailable.
    pub(super) fn project_next_chat_frame(&mut self) -> Result<Option<WaitFor>, OutputCloseReason> {
        let projection = self
            .chat_projection
            .as_ref()
            .ok_or(OutputCloseReason::InvalidOutput)?;
        if !projection.has_pending() {
            let pending = self
                .pending_projection
                .take()
                .ok_or(OutputCloseReason::InvalidOutput)?;
            if let Some((frame, wire_slot)) = pending.first {
                self.budget
                    .finish_without_wire_frame(frame)
                    .map_err(|_| OutputCloseReason::InvalidOutput)?;
                drop(wire_slot);
            }
            return Ok(None);
        }
        let first = self
            .pending_projection
            .as_mut()
            .ok_or(OutputCloseReason::InvalidOutput)?
            .first
            .take();
        let (frame, wire_slot) = match first {
            Some(parts) => parts,
            None => match self.reserve_data_frame()? {
                FrameReservation::Reserved(frame, slot) => (frame, slot),
                FrameReservation::Wait(wait) => return Ok(Some(wait)),
            },
        };
        let delta = self
            .chat_projection
            .as_ref()
            .and_then(BoundedChatProjection::pending_delta)
            .ok_or(OutputCloseReason::InvalidOutput)?;
        let pending = self
            .pending_projection
            .as_mut()
            .ok_or(OutputCloseReason::InvalidOutput)?;
        let wire = self
            .budget
            .encode_chat_data_frame(frame, delta, pending.created)
            .map_err(|_| OutputCloseReason::InvalidOutput)?;
        let _ = wire_slot.send(CreditedOutputFrame::new(wire, pending.metadata));
        // A token is represented at most once in frame metadata even when both
        // semantic channels became visible in the same committed command.
        pending.metadata.token = None;
        self.chat_projection
            .as_mut()
            .expect("checked projection")
            .advance()
            .map_err(|_| OutputCloseReason::InvalidOutput)?;
        if !self
            .chat_projection
            .as_ref()
            .expect("checked projection")
            .has_pending()
        {
            self.pending_projection = None;
        }
        Ok(None)
    }

    pub(super) async fn flush_chat_terminal(
        &mut self,
        text: &str,
        created: u64,
    ) -> Result<(), OutputCloseReason> {
        if self.pending_projection.is_some() {
            return Err(OutputCloseReason::InvalidOutput);
        }
        self.chat_projection
            .as_mut()
            .ok_or(OutputCloseReason::InvalidOutput)?
            .prepare(text, true)
            .map_err(|_| OutputCloseReason::InvalidOutput)?;
        self.pending_projection = Some(PendingProjection {
            metadata: OutputFrameMetadata {
                ordinal: self.applied,
                token: None,
                generated_tokens: self.generated_tokens,
                terminal: false,
            },
            created,
            first: None,
        });
        while self.pending_projection.is_some() {
            if let Some(reason) = self.shared.cancel_reason() {
                return Err(reason);
            }
            if let Some(wait) = self.project_next_chat_frame()? {
                match self.event(wait).await {
                    Event::Capacity(slot) => self.acquired_wire_slot = slot,
                    Event::Close(reason) => return Err(reason),
                    // Terminal control was already consumed before this flush;
                    // no outstanding grant can produce another command now.
                    Event::Command(_) | Event::Terminal(_) => {
                        return Err(OutputCloseReason::InvalidOutput)
                    }
                }
            }
        }
        Ok(())
    }
}
