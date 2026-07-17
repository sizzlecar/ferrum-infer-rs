use super::{
    LogicalWorkKind, ParticipantState, PressureEpisode, PressureEpisodeState,
    PressureTransitionOrdinal, PressureYieldSelection,
};
use ferrum_types::RequestId;
use std::collections::HashSet;
use std::fmt;

pub(super) trait PressureSelectionPolicy: fmt::Debug + Send + Sync {
    fn select_progress_owner(
        &self,
        episode: &PressureEpisode,
        requested: &HashSet<RequestId>,
    ) -> Option<RequestId>;

    fn select_yield(
        &self,
        episode: &PressureEpisode,
        requested: &HashSet<RequestId>,
        owner_id: &RequestId,
    ) -> Option<PressureYieldSelection>;
}

/// Keeps the oldest blocked frontier moving while minimizing avoidable
/// recompute among equivalent release candidates.
#[derive(Debug)]
pub(super) struct FairPressureSelectionPolicy;

impl PressureSelectionPolicy for FairPressureSelectionPolicy {
    fn select_progress_owner(
        &self,
        episode: &PressureEpisode,
        requested: &HashSet<RequestId>,
    ) -> Option<RequestId> {
        let prior_yielded = episode
            .yielded_request
            .as_ref()
            .and_then(|request_id| episode.participants.get(request_id))
            .filter(|participant| participant.work_kind != LogicalWorkKind::Terminal);
        if episode.state == PressureEpisodeState::Resumable {
            if let Some(participant) = prior_yielded {
                return Some(participant.request_id.clone());
            }
        }

        episode
            .participants
            .values()
            .filter(|participant| {
                matches!(participant.state, ParticipantState::Blocked { .. })
                    || requested.contains(&participant.request_id)
            })
            .min_by(|left, right| {
                let left_ordinal = match left.state {
                    ParticipantState::Blocked { ordinal } => ordinal,
                    _ => PressureTransitionOrdinal(u64::MAX),
                };
                let right_ordinal = match right.state {
                    ParticipantState::Blocked { ordinal } => ordinal,
                    _ => PressureTransitionOrdinal(u64::MAX),
                };
                left_ordinal
                    .cmp(&right_ordinal)
                    .then_with(|| right.priority.cmp(&left.priority))
                    .then_with(|| right.recompute_cost.cmp(&left.recompute_cost))
                    .then_with(|| left.request_id.0.cmp(&right.request_id.0))
            })
            .map(|participant| participant.request_id.clone())
    }

    fn select_yield(
        &self,
        episode: &PressureEpisode,
        requested: &HashSet<RequestId>,
        owner_id: &RequestId,
    ) -> Option<PressureYieldSelection> {
        let peer = episode
            .participants
            .values()
            .filter(|participant| {
                participant.request_id != *owner_id
                    && participant.advances_wait_source
                    && !matches!(participant.state, ParticipantState::Yielded)
            })
            .min_by(|left, right| {
                let left_requested = requested.contains(&left.request_id);
                let right_requested = requested.contains(&right.request_id);
                right_requested
                    .cmp(&left_requested)
                    .then_with(|| left.priority.cmp(&right.priority))
                    .then_with(|| left.recompute_cost.cmp(&right.recompute_cost))
                    .then_with(|| left.progress.cmp(&right.progress))
                    .then_with(|| left.request_id.0.cmp(&right.request_id.0))
            })
            .map(|participant| participant.request_id.clone());
        if let Some(peer) = peer {
            return Some(PressureYieldSelection::PeerHandoff(peer));
        }

        episode
            .participants
            .get(owner_id)
            .filter(|participant| {
                participant.advances_wait_source
                    && participant.work_kind == LogicalWorkKind::Decode
                    && matches!(participant.state, ParticipantState::Blocked { .. })
            })
            .map(|participant| {
                // Decode -> recompute strictly reduces resident state. A
                // recompute -> recompute transition discards partial replay
                // without logical progress and can only create a restart loop.
                PressureYieldSelection::SelfRecompute(participant.request_id.clone())
            })
    }
}
