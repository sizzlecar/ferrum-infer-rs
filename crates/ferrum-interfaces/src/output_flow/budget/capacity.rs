//! Observational capacity backed by the admitted lifetime plan and real leases.
use super::*;

/// No reservation or execution authority. Only a budget can construct this
/// copy; lifecycle bytes are already charged and must not be charged again to
/// global free bytes for every simulated token. No consumer release is assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrepaidOutputCapacityView {
    remaining_token_commands: usize,
    remaining_wire_bytes: usize,
    no_drain_token_commands: usize,
}

impl PrepaidOutputCapacityView {
    pub fn remaining_token_commands(self) -> usize {
        self.remaining_token_commands
    }
    pub fn remaining_wire_bytes(self) -> usize {
        self.remaining_wire_bytes
    }
    pub fn no_drain_token_commands(self) -> usize {
        self.no_drain_token_commands
    }
}

impl RequestOutputBudget {
    pub fn data_event_capacity(&self) -> usize {
        let limits = self.account.snapshot().limits;
        limits.maximum.events - limits.terminal.events
    }

    /// Acquire actual event escrow for a bounded producer window. Existing
    /// output frames keep their charge; failed reservations grant no capacity.
    /// This is optional: ordinary single-frame users retain their old behavior.
    pub fn reserve_future_event_window(&mut self, target: usize) -> Result<(), OutputFlowError> {
        if self.closed {
            return Err(OutputFlowError::Closed);
        }
        let snapshot = self.account.snapshot();
        let held = self
            .future_events
            .as_ref()
            .map_or(0, |events| events.amount().events);
        let available = (snapshot.limits.maximum.events - snapshot.limits.terminal.events)
            .checked_sub(snapshot.data_used.events)
            .ok_or(OutputFlowError::BoundExceeded)?;
        let additional = target.saturating_sub(held).min(available);
        if additional == 0 {
            return Ok(());
        }
        let OutputCreditAttempt::Reserved(mut events) = self.account.try_reserve(
            OutputCreditLane::Data,
            OutputCreditAmount {
                events: additional,
                bytes: 0,
                projection_bytes: 0,
            },
        )?
        else {
            return Ok(());
        };
        if let Some(escrow) = self.future_events.as_mut() {
            escrow.try_merge(&mut events)?;
        } else {
            self.future_events = Some(events);
        }
        Ok(())
    }

    /// The owner supplies its actual processed-token frontier and additional
    /// currently free bounded wire slots (excluding the ready permit/terminal).
    /// This method cannot grant queue capacity: the transport owner must publish
    /// the result only alongside that same still-live ready permit.
    pub fn future_capacity_view(
        &self,
        ready: &OutputFramePermit,
        generated_tokens: usize,
        additional_wire_slots: usize,
    ) -> Result<PrepaidOutputCapacityView, OutputFlowError> {
        let credit = ready.credit();
        if self.closed || self.wire.is_some() {
            return Err(OutputFlowError::Closed);
        }
        if !self
            .projection
            .as_ref()
            .is_some_and(|projection| projection.same_account(&ready.credit))
            || ready.credit.lane() != OutputCreditLane::Data
            || credit.events != 1
            || credit.projection_bytes != 0
        {
            return Err(OutputCreditError::ForeignReservation.into());
        }
        let remaining_token_commands = self
            .plan
            .effective_max_tokens()
            .checked_sub(generated_tokens)
            .ok_or(OutputFlowError::BoundExceeded)?;
        let held_events = self
            .future_events
            .as_ref()
            .map_or(0, |events| events.amount().events);
        let additional_frames = held_events.min(additional_wire_slots);
        // Before another command can commit, the previous command must finish
        // projecting. Chat may emit both reasoning and content. The last command
        // can remain safely bounded in PendingProjection after its commit; its
        // first frame/event and raw arena are already guaranteed by Ready.
        let frames_per_command = if self.plan.text_reasoning_policy().is_some() {
            2
        } else {
            1
        };
        let no_drain_token_commands =
            (1 + additional_frames / frames_per_command).min(remaining_token_commands);
        Ok(PrepaidOutputCapacityView {
            remaining_token_commands,
            remaining_wire_bytes: credit.bytes,
            no_drain_token_commands,
        })
    }
}
