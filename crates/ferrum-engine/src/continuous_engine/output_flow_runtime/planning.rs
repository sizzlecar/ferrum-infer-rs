//! Nonblocking snapshots of already published output capacity.
use super::*;

impl OutputPlanningCreditView {
    pub(super) fn readiness_state(self) -> OutputReadinessState {
        match self {
            Self::Ready(view) if view.future_tokens > 0 && view.reserved.events > 0 => {
                // Bytes are the actual remaining reservoir, not a minimum
                // payload size. A zero-byte permit can still commit a token
                // without wire output via finish_without_wire_frame; codec
                // bounds remain enforced when a command actually emits text.
                OutputReadinessState::Ready
            }
            Self::Ready(_) | Self::ProjectionBusy => OutputReadinessState::ProjectionBusy,
            Self::OutputBlocked(reason) => OutputReadinessState::OutputBlocked(reason),
            Self::Closing(reason) => OutputReadinessState::Closing(reason),
        }
    }
}

impl OutputFlowPort {
    /// Diagnostic application frontier while the actor has published its next
    /// real credit. This does not assert client delivery or reserve any permit.
    pub(in crate::continuous_engine) fn applied_ordinal_while_ready(&self) -> Option<u64> {
        if self.shared.cancel_reason().is_some() {
            return None;
        }
        let mailbox = self.shared.mailbox.try_lock()?;
        if self.shared.cancel_reason().is_some() || !matches!(mailbox.state, PortState::Ready) {
            return None;
        }
        mailbox.ready.as_ref()?.ordinal.checked_sub(1)
    }
    /// Read only the bounded mailbox. Never takes a permit, reserves capacity,
    /// touches the codec/owner budget, or wakes the actor. `Ready` is a point-in-
    /// time observation, not permission to submit work or promise future refill.
    pub fn planning_credit_view(&self) -> OutputPlanningCreditView {
        let snapshot = self.planning_snapshot();
        if matches!(snapshot.readiness, OutputPlanningCreditView::Ready(_))
            && snapshot
                .future_capacity
                .is_none_or(|capacity| capacity.remaining_token_commands() == 0)
        {
            return OutputPlanningCreditView::ProjectionBusy;
        }
        snapshot.readiness
    }

    pub fn planning_snapshot(&self) -> OutputPlanningSnapshot {
        let unavailable = |readiness| OutputPlanningSnapshot {
            readiness,
            future_capacity: None,
        };
        if let Some(reason) = self.shared.cancel_reason() {
            return unavailable(OutputPlanningCreditView::Closing(reason));
        }
        let Some(mailbox) = self.shared.mailbox.try_lock() else {
            return unavailable(OutputPlanningCreditView::ProjectionBusy);
        };
        // Cancellation publishes its atomic reason before taking the mailbox
        // lock. Recheck after acquiring it so a waiting cancellation wins.
        if let Some(reason) = self.shared.cancel_reason() {
            return unavailable(OutputPlanningCreditView::Closing(reason));
        }
        match mailbox.state {
            PortState::Ready => match mailbox.ready.as_ref() {
                Some(parts) => OutputPlanningSnapshot {
                    readiness: OutputPlanningCreditView::Ready(ReadyOutputCreditView {
                        future_tokens: 1,
                        reserved: parts.frame.credit(),
                    }),
                    future_capacity: Some(parts.future_capacity),
                },
                None => unavailable(OutputPlanningCreditView::ProjectionBusy),
            },
            PortState::OutputBlocked(reason) => {
                unavailable(OutputPlanningCreditView::OutputBlocked(reason))
            }
            PortState::Closing(reason) => unavailable(OutputPlanningCreditView::Closing(reason)),
            PortState::ProjectionBusy | PortState::InFlight => {
                unavailable(OutputPlanningCreditView::ProjectionBusy)
            }
        }
    }
}
