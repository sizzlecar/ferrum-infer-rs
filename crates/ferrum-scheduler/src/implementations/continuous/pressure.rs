//! Phase-independent capacity-pressure coordination for continuous batching.
//!
//! The normal scheduler path never enters this module. An authoritative
//! allocator failure opens an episode; all subsequent liveness decisions for
//! the affected exact sources are made here, independent of the executor's
//! current prefill/decode work kind.

mod policy;

use ferrum_interfaces::vnext::{
    CapacityAvailabilitySource, CapacityWaitCondition, LogicalAdmissionCoordinatorId,
};
use ferrum_types::{Priority, RequestId};
use policy::{FairPressureSelectionPolicy, PressureSelectionPolicy};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;

const DEFAULT_JOURNAL_CAPACITY: usize = 512;

/// Executor work derived from one logical request frontier.
///
/// These values describe work; they are not independent resource owners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LogicalWorkKind {
    Waiting,
    Prefill,
    Recompute,
    Decode,
    Terminal,
}

/// Monotonic evidence that a request committed useful executor work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct LogicalWorkGeneration(u64);

impl LogicalWorkGeneration {
    pub(crate) const ZERO: Self = Self(0);

    pub const fn get(self) -> u64 {
        self.0
    }

    fn advance(self, delta: usize) -> Self {
        Self(
            self.0
                .saturating_add(u64::try_from(delta).unwrap_or(u64::MAX)),
        )
    }
}

/// Phase-independent logical and resident work owned by one request.
///
/// The value moves with the request when physical queue storage changes. It
/// intentionally contains no backend/model discriminator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct LogicalWorkFrontier {
    work_kind: LogicalWorkKind,
    computed_tokens: usize,
    resident_tokens: usize,
    scheduled_tokens: usize,
    committed_output_tokens: usize,
    progress_generation: LogicalWorkGeneration,
    recompute_origin: Option<LogicalWorkGeneration>,
    recompute_target_tokens: Option<usize>,
}

impl Default for LogicalWorkFrontier {
    fn default() -> Self {
        Self {
            work_kind: LogicalWorkKind::Waiting,
            computed_tokens: 0,
            resident_tokens: 0,
            scheduled_tokens: 0,
            committed_output_tokens: 0,
            progress_generation: LogicalWorkGeneration::ZERO,
            recompute_origin: None,
            recompute_target_tokens: None,
        }
    }
}

impl LogicalWorkFrontier {
    pub(crate) const fn work_kind(&self) -> LogicalWorkKind {
        self.work_kind
    }

    pub(crate) const fn progress_generation(&self) -> LogicalWorkGeneration {
        self.progress_generation
    }

    pub(crate) const fn recompute_cost(&self) -> usize {
        self.resident_tokens
    }

    pub(crate) fn begin_prefill(&mut self, recompute: bool) {
        self.work_kind = if recompute {
            LogicalWorkKind::Recompute
        } else {
            LogicalWorkKind::Prefill
        };
        if recompute && self.recompute_origin.is_none() {
            self.recompute_origin = Some(self.progress_generation);
        } else if !recompute {
            self.recompute_origin = None;
            self.recompute_target_tokens = None;
        }
        self.scheduled_tokens = self.computed_tokens;
    }

    pub(crate) fn begin_decode(&mut self) {
        self.work_kind = LogicalWorkKind::Decode;
        self.recompute_origin = None;
        self.recompute_target_tokens = None;
        self.scheduled_tokens = self.computed_tokens;
    }

    pub(crate) fn mark_scheduled(&mut self, tokens: usize) {
        self.scheduled_tokens = self.computed_tokens.saturating_add(tokens);
    }

    pub(crate) fn commit_prefill(&mut self, computed_tokens: usize, delta: usize) {
        self.computed_tokens = computed_tokens;
        self.resident_tokens = computed_tokens;
        self.scheduled_tokens = computed_tokens;
        // Recompute restores evicted resident state; replaying previously
        // committed tokens is not new logical progress and cannot release a
        // pressure hold. The next newly committed decode token advances the
        // generation after `begin_decode` clears the recompute target.
        if self.recompute_target_tokens.is_none() && delta > 0 {
            self.progress_generation = self.progress_generation.advance(delta);
        }
    }

    pub(crate) fn commit_decode(&mut self, committed_output_tokens: usize) {
        let delta = committed_output_tokens.saturating_sub(self.committed_output_tokens);
        self.committed_output_tokens = committed_output_tokens;
        self.computed_tokens = self.computed_tokens.saturating_add(delta);
        self.resident_tokens = self.resident_tokens.saturating_add(delta);
        self.scheduled_tokens = self.computed_tokens;
        if delta > 0 {
            self.progress_generation = self.progress_generation.advance(delta);
        }
    }

    pub(crate) fn yield_for_recompute(&mut self) {
        self.work_kind = LogicalWorkKind::Waiting;
        self.recompute_origin = Some(self.progress_generation);
        self.recompute_target_tokens = Some(self.computed_tokens);
        self.computed_tokens = 0;
        self.resident_tokens = 0;
        self.scheduled_tokens = 0;
    }

    pub(crate) fn finish(&mut self) {
        self.work_kind = LogicalWorkKind::Terminal;
        self.scheduled_tokens = self.computed_tokens;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct PressureEpisodeId(u64);

impl PressureEpisodeId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct PressureTransitionOrdinal(u64);

impl PressureTransitionOrdinal {
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureEpisodeState {
    Open,
    YieldPlanned,
    AwaitReleaseFence,
    Resumable,
    OwnerAdmissionPending,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureTransitionKind {
    Opened,
    FrontierBlocked,
    YieldPlanned,
    YieldAborted,
    ReleaseFenceArmed,
    ReleaseFenceCompleted,
    FrontierResumable,
    OwnerAdmissionPending,
    OwnerAdmitted,
    FrontierRetargeted,
    FrontierTerminal,
    Closed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PressureTransition {
    ordinal: PressureTransitionOrdinal,
    episode_id: PressureEpisodeId,
    kind: PressureTransitionKind,
    request_id: Option<RequestId>,
    peer_request_id: Option<RequestId>,
    state: PressureEpisodeState,
}

impl PressureTransition {
    pub const fn ordinal(&self) -> PressureTransitionOrdinal {
        self.ordinal
    }

    pub const fn episode_id(&self) -> PressureEpisodeId {
        self.episode_id
    }

    pub const fn kind(&self) -> PressureTransitionKind {
        self.kind
    }

    pub fn request_id(&self) -> Option<&RequestId> {
        self.request_id.as_ref()
    }

    pub fn peer_request_id(&self) -> Option<&RequestId> {
        self.peer_request_id.as_ref()
    }

    pub const fn state(&self) -> PressureEpisodeState {
        self.state
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PressureCandidate {
    pub(crate) request_id: RequestId,
    pub(crate) work_kind: LogicalWorkKind,
    pub(crate) priority: Priority,
    pub(crate) progress: LogicalWorkGeneration,
    pub(crate) recompute_cost: usize,
    pub(crate) advances_wait_source: bool,
    pub(crate) blocked_on: Option<CapacityWaitCondition>,
}

impl PressureCandidate {
    fn overlaps(&self, condition: &CapacityWaitCondition) -> bool {
        self.blocked_on
            .as_ref()
            .is_some_and(|blocked| waits_overlap(blocked, condition))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ParticipantState {
    Runnable,
    Blocked { ordinal: PressureTransitionOrdinal },
    YieldPlanned,
    Held,
    PendingResume,
    OwnerAdmissionPending,
    ProgressOwner,
    Terminal,
}

#[derive(Debug, Clone)]
struct PressureParticipant {
    request_id: RequestId,
    work_kind: LogicalWorkKind,
    priority: Priority,
    progress: LogicalWorkGeneration,
    recompute_cost: usize,
    advances_wait_source: bool,
    state: ParticipantState,
}

#[derive(Debug)]
struct PressureEpisode {
    state: PressureEpisodeState,
    observed: BTreeMap<CapacityAvailabilitySource, u64>,
    participants: HashMap<RequestId, PressureParticipant>,
    progress_owner: Option<RequestId>,
    last_transaction_victim: Option<RequestId>,
    owner_progress_baseline: LogicalWorkGeneration,
    owner_admission_pending_ordinal: Option<PressureTransitionOrdinal>,
    handoff_generation: u64,
    last_release_condition: Option<CapacityWaitCondition>,
}

impl PressureEpisode {
    fn new(condition: &CapacityWaitCondition) -> PressureEpisode {
        Self {
            state: PressureEpisodeState::Open,
            observed: observed_map(condition),
            participants: HashMap::new(),
            progress_owner: None,
            last_transaction_victim: None,
            owner_progress_baseline: LogicalWorkGeneration::ZERO,
            owner_admission_pending_ordinal: None,
            handoff_generation: 0,
            last_release_condition: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PressureYieldTransaction {
    episode_id: PressureEpisodeId,
    handoff_generation: u64,
    kind: PressureYieldKind,
    victim_request_id: RequestId,
    progress_owner_id: RequestId,
    progress_baseline: LogicalWorkGeneration,
    planned_ordinal: PressureTransitionOrdinal,
    wait_condition: CapacityWaitCondition,
}

/// Strategy selected at an authoritative execution-capacity boundary.
///
/// A peer handoff frees one frontier so another can make logical progress. A
/// self recompute is the bounded fallback when the only frontier proven to
/// advance the waited source is also the blocked owner; it releases that
/// physical state and queues the same logical frontier for reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureYieldKind {
    PeerHandoff,
    SelfRecompute,
}

impl PressureYieldKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PeerHandoff => "peer_handoff",
            Self::SelfRecompute => "self_recompute",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PressureYieldSelection {
    PeerHandoff(RequestId),
    SelfRecompute(RequestId),
}

impl PressureYieldSelection {
    fn kind(&self) -> PressureYieldKind {
        match self {
            Self::PeerHandoff(_) => PressureYieldKind::PeerHandoff,
            Self::SelfRecompute(_) => PressureYieldKind::SelfRecompute,
        }
    }

    fn into_victim(self) -> RequestId {
        match self {
            Self::PeerHandoff(request_id) | Self::SelfRecompute(request_id) => request_id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PressureReleaseFenceDisposition {
    Resumable(PressureTransitionOrdinal),
    OwnerAdmissionPending(PressureTransitionOrdinal),
    SelfRecomputeQueued(PressureTransitionOrdinal),
    Closed {
        ordinal: PressureTransitionOrdinal,
        reason: PressureHoldReleaseReason,
    },
}

impl PressureYieldTransaction {
    pub const fn episode_id(&self) -> PressureEpisodeId {
        self.episode_id
    }

    pub const fn handoff_generation(&self) -> u64 {
        self.handoff_generation
    }

    pub const fn kind(&self) -> PressureYieldKind {
        self.kind
    }

    pub fn victim_request_id(&self) -> &RequestId {
        &self.victim_request_id
    }

    pub fn progress_owner_id(&self) -> &RequestId {
        &self.progress_owner_id
    }

    pub const fn progress_baseline(&self) -> LogicalWorkGeneration {
        self.progress_baseline
    }

    pub const fn planned_ordinal(&self) -> PressureTransitionOrdinal {
        self.planned_ordinal
    }

    pub fn wait_condition(&self) -> &CapacityWaitCondition {
        &self.wait_condition
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PressureDecision {
    Deferred {
        episode_id: PressureEpisodeId,
        count: usize,
    },
    YieldPlanned(PressureYieldTransaction),
    InvariantViolation(PressureInvariantViolation),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureInvariantViolationClass {
    NoReleasableFrontier,
    OwnerRecomputeBlocked,
    YieldDidNotAdvanceSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PressureInvariantViolation {
    episode_id: PressureEpisodeId,
    class: PressureInvariantViolationClass,
    blocked_frontiers: usize,
}

impl PressureInvariantViolation {
    pub const fn episode_id(&self) -> PressureEpisodeId {
        self.episode_id
    }

    pub const fn class(&self) -> PressureInvariantViolationClass {
        self.class
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureHoldReleaseReason {
    OwnerTerminal,
}

impl PressureHoldReleaseReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OwnerTerminal => "owner_terminal",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PressureHoldStatus {
    None,
    Held {
        episode_id: PressureEpisodeId,
        progress_owner_id: RequestId,
        progress_baseline: LogicalWorkGeneration,
        progress_current: LogicalWorkGeneration,
    },
    OwnerAdmissionEligible {
        episode_id: PressureEpisodeId,
        progress_owner_id: RequestId,
        progress_baseline: LogicalWorkGeneration,
        progress_current: LogicalWorkGeneration,
        ordinal: PressureTransitionOrdinal,
    },
    Released {
        episode_id: PressureEpisodeId,
        progress_owner_id: RequestId,
        progress_baseline: LogicalWorkGeneration,
        progress_current: LogicalWorkGeneration,
        reason: PressureHoldReleaseReason,
        ordinal: PressureTransitionOrdinal,
        previous_wait_condition: Option<CapacityWaitCondition>,
        current_wait_condition: Option<CapacityWaitCondition>,
    },
}

#[derive(Debug, Clone)]
struct ReleasedHold {
    episode_id: PressureEpisodeId,
    progress_owner_id: RequestId,
    progress_baseline: LogicalWorkGeneration,
    progress_current: LogicalWorkGeneration,
    reason: PressureHoldReleaseReason,
    ordinal: PressureTransitionOrdinal,
    previous_wait_condition: Option<CapacityWaitCondition>,
    current_wait_condition: Option<CapacityWaitCondition>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub(crate) struct PressureCoordinatorStats {
    pub(crate) episodes_created: u64,
    pub(crate) candidate_scans: u64,
    pub(crate) active_episodes: usize,
    pub(crate) pending_release_fences: usize,
    pub(crate) last_transition_ordinal: u64,
    pub(crate) dropped_journal_entries: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PressureCoordinatorError {
    EpisodeIdentityExhausted,
    TransitionIdentityExhausted,
    OverlappingPendingEpisodes,
    UnknownEpisode(PressureEpisodeId),
    StaleTransaction {
        episode_id: PressureEpisodeId,
        expected: u64,
        actual: u64,
    },
    InvalidState {
        episode_id: PressureEpisodeId,
        expected: PressureEpisodeState,
        actual: PressureEpisodeState,
    },
}

impl fmt::Display for PressureCoordinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EpisodeIdentityExhausted => {
                formatter.write_str("pressure episode identity exhausted")
            }
            Self::TransitionIdentityExhausted => {
                formatter.write_str("pressure transition identity exhausted")
            }
            Self::OverlappingPendingEpisodes => formatter.write_str(
                "overlapping capacity sources belong to concurrent pending pressure episodes",
            ),
            Self::UnknownEpisode(id) => write!(formatter, "unknown pressure episode {}", id.get()),
            Self::StaleTransaction {
                episode_id,
                expected,
                actual,
            } => write!(
                formatter,
                "stale pressure transaction for episode {}: expected handoff {}, found {}",
                episode_id.get(),
                expected,
                actual
            ),
            Self::InvalidState {
                episode_id,
                expected,
                actual,
            } => write!(
                formatter,
                "pressure episode {} expected state {expected:?}, found {actual:?}",
                episode_id.get()
            ),
        }
    }
}

impl std::error::Error for PressureCoordinatorError {}

/// Cold-path coordinator for exact-source capacity pressure.
pub(crate) struct PressureCoordinator {
    next_episode_id: u64,
    next_transition_ordinal: u64,
    episodes: HashMap<PressureEpisodeId, PressureEpisode>,
    source_index:
        HashMap<(LogicalAdmissionCoordinatorId, CapacityAvailabilitySource), PressureEpisodeId>,
    request_index: HashMap<RequestId, PressureEpisodeId>,
    released_holds: HashMap<RequestId, ReleasedHold>,
    journal: VecDeque<PressureTransition>,
    journal_capacity: usize,
    stats: PressureCoordinatorStats,
    selection_policy: Box<dyn PressureSelectionPolicy>,
}

impl Default for PressureCoordinator {
    fn default() -> Self {
        Self::with_journal_capacity(DEFAULT_JOURNAL_CAPACITY)
    }
}

impl PressureCoordinator {
    pub(crate) fn with_journal_capacity(journal_capacity: usize) -> Self {
        Self {
            next_episode_id: 1,
            next_transition_ordinal: 1,
            episodes: HashMap::new(),
            source_index: HashMap::new(),
            request_index: HashMap::new(),
            released_holds: HashMap::new(),
            journal: VecDeque::new(),
            journal_capacity: journal_capacity.max(1),
            stats: PressureCoordinatorStats::default(),
            selection_policy: Box::new(FairPressureSelectionPolicy),
        }
    }

    pub(crate) fn stats(&self) -> PressureCoordinatorStats {
        PressureCoordinatorStats {
            active_episodes: self.episodes.len(),
            pending_release_fences: self
                .episodes
                .values()
                .filter(|episode| {
                    matches!(
                        episode.state,
                        PressureEpisodeState::YieldPlanned
                            | PressureEpisodeState::AwaitReleaseFence
                    )
                })
                .count(),
            ..self.stats
        }
    }

    pub(crate) fn has_records(&self) -> bool {
        !self.episodes.is_empty() || !self.released_holds.is_empty()
    }

    pub(crate) fn journal(&self) -> Vec<PressureTransition> {
        self.journal.iter().cloned().collect()
    }

    pub(crate) fn plan_failure(
        &mut self,
        request_ids: &[RequestId],
        condition: &CapacityWaitCondition,
        candidates: &[PressureCandidate],
    ) -> Result<PressureDecision, PressureCoordinatorError> {
        let requested = request_ids.iter().cloned().collect::<HashSet<_>>();
        self.stats.candidate_scans = self
            .stats
            .candidate_scans
            .saturating_add(u64::try_from(candidates.len()).unwrap_or(u64::MAX));

        let episode_id = self.find_or_open_episode_for(condition, request_ids)?;
        let current_observed = observed_map(condition);
        let previous_state = self
            .episodes
            .get(&episode_id)
            .expect("opened pressure episode remains registered")
            .state;

        if matches!(
            previous_state,
            PressureEpisodeState::YieldPlanned | PressureEpisodeState::AwaitReleaseFence
        ) {
            return Ok(PressureDecision::Deferred {
                episode_id,
                count: requested.len(),
            });
        }

        if previous_state == PressureEpisodeState::Resumable {
            let last_release_condition = self
                .episodes
                .get(&episode_id)
                .and_then(|episode| episode.last_release_condition.as_ref());
            if last_release_condition.is_some_and(|prior| {
                same_source_topology(prior, condition)
                    && !observed_advanced(&observed_map(prior), &current_observed)
            }) {
                let blocked_frontiers = candidates
                    .iter()
                    .filter(|candidate| {
                        requested.contains(&candidate.request_id) || candidate.overlaps(condition)
                    })
                    .count();
                return Ok(PressureDecision::InvariantViolation(
                    PressureInvariantViolation {
                        episode_id,
                        class: PressureInvariantViolationClass::YieldDidNotAdvanceSource,
                        blocked_frontiers,
                    },
                ));
            }
        }

        let mut newly_blocked = Vec::new();
        {
            let episode = self
                .episodes
                .get_mut(&episode_id)
                .expect("opened pressure episode remains registered");
            for (source, epoch) in &current_observed {
                episode.observed.insert(*source, *epoch);
            }
            for candidate in candidates {
                let relevant = requested.contains(&candidate.request_id)
                    || candidate.overlaps(condition)
                    || self.request_index.get(&candidate.request_id) == Some(&episode_id);
                if !relevant {
                    continue;
                }
                let participant = episode
                    .participants
                    .entry(candidate.request_id.clone())
                    .or_insert_with(|| PressureParticipant {
                        request_id: candidate.request_id.clone(),
                        work_kind: candidate.work_kind,
                        priority: candidate.priority,
                        progress: candidate.progress,
                        recompute_cost: candidate.recompute_cost,
                        advances_wait_source: candidate.advances_wait_source,
                        state: ParticipantState::Runnable,
                    });
                participant.work_kind = candidate.work_kind;
                participant.priority = candidate.priority;
                participant.progress = candidate.progress;
                participant.recompute_cost = candidate.recompute_cost;
                participant.advances_wait_source = candidate.advances_wait_source;
                self.request_index
                    .insert(candidate.request_id.clone(), episode_id);
                if requested.contains(&candidate.request_id)
                    && matches!(
                        participant.state,
                        ParticipantState::Runnable | ParticipantState::ProgressOwner
                    )
                {
                    newly_blocked.push(candidate.request_id.clone());
                } else if candidate.blocked_on.is_some()
                    && matches!(participant.state, ParticipantState::Runnable)
                {
                    newly_blocked.push(candidate.request_id.clone());
                }
            }
        }

        for request_id in newly_blocked {
            let ordinal = self.record_transition(
                episode_id,
                PressureTransitionKind::FrontierBlocked,
                Some(request_id.clone()),
                None,
                PressureEpisodeState::Open,
            )?;
            if let Some(participant) = self
                .episodes
                .get_mut(&episode_id)
                .and_then(|episode| episode.participants.get_mut(&request_id))
            {
                participant.state = ParticipantState::Blocked { ordinal };
            }
        }

        // A runnable peer may suppress self recompute only when the executor's
        // release footprint proves that terminating it advances this exact
        // wait source. Generic cache ownership is not causal evidence.
        let runnable_peer = candidates.iter().any(|candidate| {
            !requested.contains(&candidate.request_id)
                && candidate.advances_wait_source
                && candidate.blocked_on.is_none()
        });
        if runnable_peer {
            return Ok(PressureDecision::Deferred {
                episode_id,
                count: requested.len(),
            });
        }

        let owner_id = self.episodes.get(&episode_id).and_then(|episode| {
            self.selection_policy
                .select_progress_owner(episode, &requested)
        });
        let Some(owner_id) = owner_id else {
            return Ok(PressureDecision::Deferred {
                episode_id,
                count: requested.len(),
            });
        };
        let selection = self.episodes.get(&episode_id).and_then(|episode| {
            self.selection_policy
                .select_yield(episode, &requested, &owner_id)
        });
        let Some(selection) = selection else {
            let episode = self
                .episodes
                .get(&episode_id)
                .expect("opened pressure episode remains registered");
            let blocked_frontiers = episode
                .participants
                .values()
                .filter(|participant| {
                    matches!(
                        participant.state,
                        ParticipantState::Blocked { .. }
                            | ParticipantState::Held
                            | ParticipantState::OwnerAdmissionPending
                            | ParticipantState::PendingResume
                            | ParticipantState::ProgressOwner
                    )
                })
                .count();
            if blocked_frontiers > 1 {
                let owner_recompute_blocked = episode
                    .progress_owner
                    .as_ref()
                    .and_then(|owner| episode.participants.get(owner))
                    .is_some_and(|owner| {
                        owner.work_kind == LogicalWorkKind::Recompute
                            && matches!(owner.state, ParticipantState::Blocked { .. })
                    })
                    && episode
                        .participants
                        .values()
                        .any(|participant| participant.state == ParticipantState::Held);
                return Ok(PressureDecision::InvariantViolation(
                    PressureInvariantViolation {
                        episode_id,
                        class: if owner_recompute_blocked {
                            PressureInvariantViolationClass::OwnerRecomputeBlocked
                        } else {
                            PressureInvariantViolationClass::NoReleasableFrontier
                        },
                        blocked_frontiers,
                    },
                ));
            }
            return Ok(PressureDecision::Deferred {
                episode_id,
                count: requested.len(),
            });
        };
        let yield_kind = selection.kind();
        let victim_id = selection.into_victim();

        let (progress_baseline, handoff_generation) = {
            let episode = self
                .episodes
                .get_mut(&episode_id)
                .expect("opened pressure episode remains registered");
            let progress_baseline = episode
                .participants
                .get(&owner_id)
                .expect("selected progress owner remains registered")
                .progress;
            episode.handoff_generation = episode.handoff_generation.saturating_add(1);
            episode.state = PressureEpisodeState::YieldPlanned;
            if episode.progress_owner.is_none() {
                episode.progress_owner = Some(owner_id.clone());
                episode.owner_progress_baseline = progress_baseline;
            }
            episode.last_transaction_victim = Some(victim_id.clone());
            episode.owner_admission_pending_ordinal = None;
            episode.last_release_condition = Some(condition.clone());
            if let Some(owner) = episode.participants.get_mut(&owner_id) {
                owner.state = ParticipantState::PendingResume;
            }
            if let Some(victim) = episode.participants.get_mut(&victim_id) {
                victim.state = ParticipantState::YieldPlanned;
            }
            (progress_baseline, episode.handoff_generation)
        };
        let planned_ordinal = self.record_transition(
            episode_id,
            PressureTransitionKind::YieldPlanned,
            Some(victim_id.clone()),
            Some(owner_id.clone()),
            PressureEpisodeState::YieldPlanned,
        )?;
        Ok(PressureDecision::YieldPlanned(PressureYieldTransaction {
            episode_id,
            handoff_generation,
            kind: yield_kind,
            victim_request_id: victim_id,
            progress_owner_id: owner_id,
            progress_baseline,
            planned_ordinal,
            wait_condition: condition.clone(),
        }))
    }

    pub(crate) fn arm_release_fence(
        &mut self,
        transaction: &PressureYieldTransaction,
    ) -> Result<PressureTransitionOrdinal, PressureCoordinatorError> {
        self.validate_transaction(transaction, PressureEpisodeState::YieldPlanned)?;
        self.episodes
            .get_mut(&transaction.episode_id)
            .expect("validated pressure episode remains registered")
            .state = PressureEpisodeState::AwaitReleaseFence;
        self.record_transition(
            transaction.episode_id,
            PressureTransitionKind::ReleaseFenceArmed,
            Some(transaction.victim_request_id.clone()),
            Some(transaction.progress_owner_id.clone()),
            PressureEpisodeState::AwaitReleaseFence,
        )
    }

    pub(crate) fn complete_release_fence(
        &mut self,
        transaction: &PressureYieldTransaction,
        progress_owner_wait_condition: Option<&CapacityWaitCondition>,
    ) -> Result<
        (PressureTransitionOrdinal, PressureReleaseFenceDisposition),
        PressureCoordinatorError,
    > {
        self.validate_transaction(transaction, PressureEpisodeState::AwaitReleaseFence)?;
        let release_ordinal = self.record_transition(
            transaction.episode_id,
            PressureTransitionKind::ReleaseFenceCompleted,
            Some(transaction.victim_request_id.clone()),
            Some(transaction.progress_owner_id.clone()),
            PressureEpisodeState::AwaitReleaseFence,
        )?;
        let owner_terminal = self
            .episodes
            .get(&transaction.episode_id)
            .and_then(|episode| episode.participants.get(&transaction.progress_owner_id))
            .is_none_or(|owner| owner.state == ParticipantState::Terminal);
        let retargeted_wait_condition = progress_owner_wait_condition
            .filter(|current| !same_source_topology(transaction.wait_condition(), current))
            .cloned();
        let retained_peer_hold = transaction.kind == PressureYieldKind::SelfRecompute
            && self
                .episodes
                .get(&transaction.episode_id)
                .is_some_and(|episode| {
                    episode.participants.values().any(|participant| {
                        participant.request_id != transaction.progress_owner_id
                            && participant.state == ParticipantState::Held
                    })
                });
        {
            let episode = self
                .episodes
                .get_mut(&transaction.episode_id)
                .expect("validated pressure episode remains registered");
            if transaction.kind == PressureYieldKind::PeerHandoff || !retained_peer_hold {
                if let Some(victim) = episode.participants.get_mut(&transaction.victim_request_id) {
                    if victim.state != ParticipantState::Terminal {
                        victim.state = ParticipantState::Held;
                    }
                    victim.advances_wait_source = false;
                }
            }
            if !owner_terminal && transaction.kind == PressureYieldKind::PeerHandoff {
                episode.state = PressureEpisodeState::Resumable;
                if let Some(owner) = episode.participants.get_mut(&transaction.progress_owner_id) {
                    owner.state = ParticipantState::ProgressOwner;
                }
            } else if !owner_terminal && retained_peer_hold {
                episode.state = PressureEpisodeState::OwnerAdmissionPending;
                if let Some(owner) = episode.participants.get_mut(&transaction.progress_owner_id) {
                    owner.state = ParticipantState::OwnerAdmissionPending;
                    owner.advances_wait_source = false;
                }
            }
        }
        if owner_terminal {
            let closed_ordinal = self.close_episode(
                transaction.episode_id,
                Some(PressureHoldReleaseReason::OwnerTerminal),
            )?;
            return Ok((
                release_ordinal,
                PressureReleaseFenceDisposition::Closed {
                    ordinal: closed_ordinal,
                    reason: PressureHoldReleaseReason::OwnerTerminal,
                },
            ));
        }
        if transaction.kind == PressureYieldKind::SelfRecompute {
            if retained_peer_hold {
                let pending_ordinal = self.record_transition(
                    transaction.episode_id,
                    PressureTransitionKind::OwnerAdmissionPending,
                    Some(transaction.progress_owner_id.clone()),
                    None,
                    PressureEpisodeState::OwnerAdmissionPending,
                )?;
                self.episodes
                    .get_mut(&transaction.episode_id)
                    .expect("retained pressure episode remains registered")
                    .owner_admission_pending_ordinal = Some(pending_ordinal);
                return Ok((
                    release_ordinal,
                    PressureReleaseFenceDisposition::OwnerAdmissionPending(pending_ordinal),
                ));
            }
            let closed_ordinal = self.close_episode(transaction.episode_id, None)?;
            return Ok((
                release_ordinal,
                PressureReleaseFenceDisposition::SelfRecomputeQueued(closed_ordinal),
            ));
        }
        if let Some(current_wait_condition) = retargeted_wait_condition {
            let current_observed = observed_map(&current_wait_condition);
            if let Some(episode) = self.episodes.get_mut(&transaction.episode_id) {
                for (source, epoch) in &current_observed {
                    episode.observed.insert(*source, *epoch);
                }
            }
            for source in current_wait_condition
                .observed()
                .iter()
                .filter(|source| is_request_release_source(source.source()))
            {
                self.source_index.insert(
                    (current_wait_condition.coordinator_id(), source.source()),
                    transaction.episode_id,
                );
            }
            self.record_transition(
                transaction.episode_id,
                PressureTransitionKind::FrontierRetargeted,
                Some(transaction.progress_owner_id.clone()),
                Some(transaction.victim_request_id.clone()),
                PressureEpisodeState::Resumable,
            )?;
        }
        let resume_ordinal = self.record_transition(
            transaction.episode_id,
            PressureTransitionKind::FrontierResumable,
            Some(transaction.progress_owner_id.clone()),
            Some(transaction.victim_request_id.clone()),
            PressureEpisodeState::Resumable,
        )?;
        Ok((
            release_ordinal,
            PressureReleaseFenceDisposition::Resumable(resume_ordinal),
        ))
    }

    pub(crate) fn abort_yield(
        &mut self,
        transaction: &PressureYieldTransaction,
    ) -> Result<
        (
            PressureTransitionOrdinal,
            PressureTransitionOrdinal,
            Vec<RequestId>,
        ),
        PressureCoordinatorError,
    > {
        let episode = self.episodes.get(&transaction.episode_id).ok_or(
            PressureCoordinatorError::UnknownEpisode(transaction.episode_id),
        )?;
        if episode.handoff_generation != transaction.handoff_generation {
            return Err(PressureCoordinatorError::StaleTransaction {
                episode_id: transaction.episode_id,
                expected: episode.handoff_generation,
                actual: transaction.handoff_generation,
            });
        }
        if !matches!(
            episode.state,
            PressureEpisodeState::YieldPlanned | PressureEpisodeState::AwaitReleaseFence
        ) {
            return Err(PressureCoordinatorError::InvalidState {
                episode_id: transaction.episode_id,
                expected: PressureEpisodeState::YieldPlanned,
                actual: episode.state,
            });
        }
        let state = episode.state;
        let participants = episode.participants.keys().cloned().collect::<Vec<_>>();
        let aborted = self.record_transition(
            transaction.episode_id,
            PressureTransitionKind::YieldAborted,
            Some(transaction.victim_request_id.clone()),
            Some(transaction.progress_owner_id.clone()),
            state,
        )?;
        let closed = self.close_episode(transaction.episode_id, None)?;
        Ok((aborted, closed, participants))
    }

    pub(crate) fn record_progress(
        &mut self,
        request_id: &RequestId,
        progress: LogicalWorkGeneration,
    ) -> Result<(), PressureCoordinatorError> {
        let Some(episode_id) = self.request_index.get(request_id).copied() else {
            return Ok(());
        };
        let Some(episode) = self.episodes.get_mut(&episode_id) else {
            return Ok(());
        };
        if let Some(participant) = episode.participants.get_mut(request_id) {
            participant.progress = participant.progress.max(progress);
        }
        // Logical token progress consumes resident capacity; it does not prove
        // that a held peer can be re-admitted. Keep the peer held until the
        // stable owner reaches a terminal release.
        Ok(())
    }

    pub(crate) fn record_terminal(
        &mut self,
        request_id: &RequestId,
    ) -> Result<Option<PressureTransitionOrdinal>, PressureCoordinatorError> {
        self.released_holds.remove(request_id);
        let Some(episode_id) = self.request_index.get(request_id).copied() else {
            return Ok(None);
        };
        let is_owner = self
            .episodes
            .get(&episode_id)
            .and_then(|episode| episode.progress_owner.as_ref())
            == Some(request_id);
        let ordinal = self.record_transition(
            episode_id,
            PressureTransitionKind::FrontierTerminal,
            Some(request_id.clone()),
            None,
            self.episodes
                .get(&episode_id)
                .map_or(PressureEpisodeState::Closed, |episode| episode.state),
        )?;
        let release_transaction_pending = self.episodes.get(&episode_id).is_some_and(|episode| {
            matches!(
                episode.state,
                PressureEpisodeState::YieldPlanned | PressureEpisodeState::AwaitReleaseFence
            )
        });
        if release_transaction_pending {
            if let Some(participant) = self
                .episodes
                .get_mut(&episode_id)
                .and_then(|episode| episode.participants.get_mut(request_id))
            {
                participant.work_kind = LogicalWorkKind::Terminal;
                participant.advances_wait_source = false;
                participant.state = ParticipantState::Terminal;
            }
            return Ok(Some(ordinal));
        }
        if is_owner {
            self.close_episode(episode_id, Some(PressureHoldReleaseReason::OwnerTerminal))?;
        } else {
            let episode_drained = if let Some(episode) = self.episodes.get_mut(&episode_id) {
                episode.participants.remove(request_id);
                self.request_index.remove(request_id);
                episode.participants.is_empty()
            } else {
                false
            };
            if episode_drained {
                self.close_episode(episode_id, Some(PressureHoldReleaseReason::OwnerTerminal))?;
            }
        }
        Ok(Some(ordinal))
    }

    pub(crate) fn hold_status(&self, request_id: &RequestId) -> PressureHoldStatus {
        if let Some(released) = self.released_holds.get(request_id) {
            return PressureHoldStatus::Released {
                episode_id: released.episode_id,
                progress_owner_id: released.progress_owner_id.clone(),
                progress_baseline: released.progress_baseline,
                progress_current: released.progress_current,
                reason: released.reason,
                ordinal: released.ordinal,
                previous_wait_condition: released.previous_wait_condition.clone(),
                current_wait_condition: released.current_wait_condition.clone(),
            };
        }
        let Some(episode_id) = self.request_index.get(request_id).copied() else {
            return PressureHoldStatus::None;
        };
        let Some(episode) = self.episodes.get(&episode_id) else {
            return PressureHoldStatus::None;
        };
        let Some(participant) = episode.participants.get(request_id) else {
            return PressureHoldStatus::None;
        };
        if matches!(
            participant.state,
            ParticipantState::YieldPlanned | ParticipantState::Held
        ) || (participant.state == ParticipantState::PendingResume
            && episode.state != PressureEpisodeState::Resumable)
        {
            return PressureHoldStatus::Held {
                episode_id,
                progress_owner_id: episode
                    .progress_owner
                    .clone()
                    .unwrap_or_else(|| request_id.clone()),
                progress_baseline: episode.owner_progress_baseline,
                progress_current: episode
                    .progress_owner
                    .as_ref()
                    .and_then(|owner| episode.participants.get(owner))
                    .map_or(episode.owner_progress_baseline, |owner| owner.progress),
            };
        }
        if participant.state == ParticipantState::OwnerAdmissionPending
            && episode.state == PressureEpisodeState::OwnerAdmissionPending
            && episode.progress_owner.as_ref() == Some(request_id)
        {
            return PressureHoldStatus::OwnerAdmissionEligible {
                episode_id,
                progress_owner_id: request_id.clone(),
                progress_baseline: episode.owner_progress_baseline,
                progress_current: participant.progress,
                ordinal: episode
                    .owner_admission_pending_ordinal
                    .expect("owner admission pending state owns its transition ordinal"),
            };
        }
        PressureHoldStatus::None
    }

    pub(crate) fn consume_released_hold(
        &mut self,
        request_id: &RequestId,
    ) -> Result<Option<PressureTransitionOrdinal>, PressureCoordinatorError> {
        if self.released_holds.remove(request_id).is_some() {
            return Ok(None);
        }
        let Some(episode_id) = self.request_index.get(request_id).copied() else {
            return Ok(None);
        };
        let owner_admitted = self.episodes.get(&episode_id).is_some_and(|episode| {
            episode.state == PressureEpisodeState::OwnerAdmissionPending
                && episode.progress_owner.as_ref() == Some(request_id)
                && episode
                    .participants
                    .get(request_id)
                    .is_some_and(|participant| {
                        participant.state == ParticipantState::OwnerAdmissionPending
                    })
        });
        if !owner_admitted {
            return Ok(None);
        }
        let episode = self
            .episodes
            .get_mut(&episode_id)
            .expect("indexed pressure episode remains registered");
        episode.state = PressureEpisodeState::Resumable;
        episode.owner_admission_pending_ordinal = None;
        episode
            .participants
            .get_mut(request_id)
            .expect("progress owner remains registered")
            .state = ParticipantState::ProgressOwner;
        self.record_transition(
            episode_id,
            PressureTransitionKind::OwnerAdmitted,
            Some(request_id.clone()),
            None,
            PressureEpisodeState::Resumable,
        )
        .map(Some)
    }

    pub(crate) fn has_pending_release_for(&self, condition: &CapacityWaitCondition) -> bool {
        condition
            .observed()
            .iter()
            .filter(|source| is_request_release_source(source.source()))
            .any(|source| {
                self.source_index
                    .get(&(condition.coordinator_id(), source.source()))
                    .and_then(|episode_id| self.episodes.get(episode_id))
                    .is_some_and(|episode| {
                        matches!(
                            episode.state,
                            PressureEpisodeState::YieldPlanned
                                | PressureEpisodeState::AwaitReleaseFence
                        )
                    })
            })
    }

    pub(crate) fn all_blocked_without_release_for(
        &self,
        condition: &CapacityWaitCondition,
    ) -> bool {
        let Some(episode_id) = condition
            .observed()
            .iter()
            .filter(|source| is_request_release_source(source.source()))
            .find_map(|source| {
                self.source_index
                    .get(&(condition.coordinator_id(), source.source()))
                    .copied()
            })
        else {
            return false;
        };
        let Some(episode) = self.episodes.get(&episode_id) else {
            return false;
        };
        let live = episode
            .participants
            .values()
            .filter(|participant| participant.work_kind != LogicalWorkKind::Terminal);
        let (live_count, blocked_count) =
            live.fold((0usize, 0usize), |(live, blocked), participant| {
                (
                    live + 1,
                    blocked
                        + usize::from(matches!(
                            participant.state,
                            ParticipantState::Blocked { .. } | ParticipantState::Held
                        )),
                )
            });
        live_count > 1
            && live_count == blocked_count
            && !matches!(
                episode.state,
                PressureEpisodeState::YieldPlanned | PressureEpisodeState::AwaitReleaseFence
            )
    }

    fn find_or_open_episode_for(
        &mut self,
        condition: &CapacityWaitCondition,
        request_ids: &[RequestId],
    ) -> Result<PressureEpisodeId, PressureCoordinatorError> {
        let mut existing = condition
            .observed()
            .iter()
            .filter(|observed| is_request_release_source(observed.source()))
            .filter_map(|observed| {
                self.source_index
                    .get(&(condition.coordinator_id(), observed.source()))
                    .copied()
            })
            .collect::<Vec<_>>();
        existing.extend(
            request_ids
                .iter()
                .filter_map(|request_id| self.request_index.get(request_id).copied()),
        );
        existing.sort_unstable();
        existing.dedup();
        if existing.len() > 1 {
            return Err(PressureCoordinatorError::OverlappingPendingEpisodes);
        }
        if let Some(id) = existing.first().copied() {
            for source in condition
                .observed()
                .iter()
                .filter(|source| is_request_release_source(source.source()))
            {
                self.source_index
                    .insert((condition.coordinator_id(), source.source()), id);
            }
            return Ok(id);
        }

        let id = PressureEpisodeId(self.next_episode_id);
        self.next_episode_id = self
            .next_episode_id
            .checked_add(1)
            .ok_or(PressureCoordinatorError::EpisodeIdentityExhausted)?;
        let episode = PressureEpisode::new(condition);
        for source in condition
            .observed()
            .iter()
            .filter(|source| is_request_release_source(source.source()))
        {
            self.source_index
                .insert((condition.coordinator_id(), source.source()), id);
        }
        self.episodes.insert(id, episode);
        self.stats.episodes_created = self.stats.episodes_created.saturating_add(1);
        self.record_transition(
            id,
            PressureTransitionKind::Opened,
            None,
            None,
            PressureEpisodeState::Open,
        )?;
        Ok(id)
    }

    fn validate_transaction(
        &self,
        transaction: &PressureYieldTransaction,
        expected_state: PressureEpisodeState,
    ) -> Result<(), PressureCoordinatorError> {
        let episode = self.episodes.get(&transaction.episode_id).ok_or(
            PressureCoordinatorError::UnknownEpisode(transaction.episode_id),
        )?;
        if episode.handoff_generation != transaction.handoff_generation {
            return Err(PressureCoordinatorError::StaleTransaction {
                episode_id: transaction.episode_id,
                expected: episode.handoff_generation,
                actual: transaction.handoff_generation,
            });
        }
        if episode.state != expected_state {
            return Err(PressureCoordinatorError::InvalidState {
                episode_id: transaction.episode_id,
                expected: expected_state,
                actual: episode.state,
            });
        }
        Ok(())
    }

    fn close_episode(
        &mut self,
        episode_id: PressureEpisodeId,
        released_hold_reason: Option<PressureHoldReleaseReason>,
    ) -> Result<PressureTransitionOrdinal, PressureCoordinatorError> {
        let Some(mut episode) = self.episodes.remove(&episode_id) else {
            return Err(PressureCoordinatorError::UnknownEpisode(episode_id));
        };
        episode.state = PressureEpisodeState::Closed;
        let progress_current = episode
            .progress_owner
            .as_ref()
            .and_then(|owner| episode.participants.get(owner))
            .map_or(episode.owner_progress_baseline, |participant| {
                participant.progress
            });
        let close_ordinal = self.record_transition(
            episode_id,
            PressureTransitionKind::Closed,
            episode.progress_owner.clone(),
            episode.last_transaction_victim.clone(),
            PressureEpisodeState::Closed,
        )?;
        for participant in episode.participants.values() {
            self.request_index.remove(&participant.request_id);
            if let Some(reason) = released_hold_reason.filter(|_| {
                matches!(
                    participant.state,
                    ParticipantState::Held | ParticipantState::YieldPlanned
                )
            }) {
                self.released_holds.insert(
                    participant.request_id.clone(),
                    ReleasedHold {
                        episode_id,
                        progress_owner_id: episode
                            .progress_owner
                            .clone()
                            .unwrap_or_else(|| participant.request_id.clone()),
                        progress_baseline: episode.owner_progress_baseline,
                        progress_current,
                        reason,
                        ordinal: close_ordinal,
                        previous_wait_condition: None,
                        current_wait_condition: None,
                    },
                );
            }
        }
        self.source_index.retain(|_, id| *id != episode_id);
        Ok(close_ordinal)
    }

    fn record_transition(
        &mut self,
        episode_id: PressureEpisodeId,
        kind: PressureTransitionKind,
        request_id: Option<RequestId>,
        peer_request_id: Option<RequestId>,
        state: PressureEpisodeState,
    ) -> Result<PressureTransitionOrdinal, PressureCoordinatorError> {
        let ordinal = PressureTransitionOrdinal(self.next_transition_ordinal);
        self.next_transition_ordinal = self
            .next_transition_ordinal
            .checked_add(1)
            .ok_or(PressureCoordinatorError::TransitionIdentityExhausted)?;
        if self.journal.len() == self.journal_capacity {
            self.journal.pop_front();
            self.stats.dropped_journal_entries =
                self.stats.dropped_journal_entries.saturating_add(1);
        }
        self.journal.push_back(PressureTransition {
            ordinal,
            episode_id,
            kind,
            request_id,
            peer_request_id,
            state,
        });
        self.stats.last_transition_ordinal = ordinal.get();
        Ok(ordinal)
    }
}

/// Sources whose availability can be advanced by yielding one live request.
///
/// Device-budget generations remain part of the exact retry predicate, but a
/// request yield only returns logical pool extents; it does not shrink the
/// plan-owned backing pool or release process-wide device claims. Treating a
/// shared device-budget source as episode causality can therefore pair a
/// victim with an owner whose blocked domain the victim cannot advance.
fn is_request_release_source(source: CapacityAvailabilitySource) -> bool {
    matches!(
        source,
        CapacityAvailabilitySource::Domain(_) | CapacityAvailabilitySource::ActiveSequenceSlots
    )
}

fn waits_overlap(left: &CapacityWaitCondition, right: &CapacityWaitCondition) -> bool {
    left.coordinator_id() == right.coordinator_id()
        && left.observed().iter().any(|left_source| {
            is_request_release_source(left_source.source())
                && right.observed().iter().any(|right_source| {
                    is_request_release_source(right_source.source())
                        && left_source.source() == right_source.source()
                })
        })
}

fn observed_map(condition: &CapacityWaitCondition) -> BTreeMap<CapacityAvailabilitySource, u64> {
    condition
        .observed()
        .iter()
        .map(|entry| (entry.source(), entry.epoch()))
        .collect()
}

fn same_source_topology(left: &CapacityWaitCondition, right: &CapacityWaitCondition) -> bool {
    left.coordinator_id() == right.coordinator_id()
        && left.observed().len() == right.observed().len()
        && left
            .observed()
            .iter()
            .zip(right.observed())
            .all(|(left, right)| left.source() == right.source())
}

fn observed_advanced(
    prior: &BTreeMap<CapacityAvailabilitySource, u64>,
    current: &BTreeMap<CapacityAvailabilitySource, u64>,
) -> bool {
    prior.iter().any(|(source, epoch)| {
        current
            .get(source)
            .is_some_and(|current_epoch| current_epoch > epoch)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::{CapacityAvailabilityEpoch, CapacityDomainId};

    fn condition(epoch: u64) -> CapacityWaitCondition {
        CapacityWaitCondition::from_observation(
            41,
            vec![CapacityAvailabilityEpoch::new(
                CapacityAvailabilitySource::Domain(CapacityDomainId::new(4).unwrap()),
                epoch,
            )
            .unwrap()],
        )
        .unwrap()
    }

    fn candidate(
        request_id: &RequestId,
        work_kind: LogicalWorkKind,
        progress: u64,
        advances_wait_source: bool,
        blocked_on: Option<CapacityWaitCondition>,
    ) -> PressureCandidate {
        PressureCandidate {
            request_id: request_id.clone(),
            work_kind,
            priority: Priority::Normal,
            progress: LogicalWorkGeneration(progress),
            recompute_cost: usize::try_from(progress).unwrap(),
            advances_wait_source,
            blocked_on,
        }
    }

    #[test]
    fn no_pressure_path_creates_no_episode_or_journal_entry() {
        let coordinator = PressureCoordinator::default();
        assert_eq!(coordinator.stats(), PressureCoordinatorStats::default());
        assert!(coordinator.journal().is_empty());
    }

    #[test]
    fn recompute_replay_does_not_advance_logical_progress() {
        let mut frontier = LogicalWorkFrontier::default();
        frontier.begin_prefill(false);
        frontier.commit_prefill(4, 4);
        frontier.begin_decode();
        frontier.commit_decode(1);
        assert_eq!(frontier.progress_generation(), LogicalWorkGeneration(5));

        frontier.yield_for_recompute();
        frontier.begin_prefill(true);
        frontier.commit_prefill(5, 5);
        assert_eq!(frontier.progress_generation(), LogicalWorkGeneration(5));

        frontier.begin_decode();
        frontier.commit_decode(2);
        assert_eq!(frontier.progress_generation(), LogicalWorkGeneration(6));
    }

    #[test]
    fn overlapping_sources_join_one_episode_index() {
        let source_a = CapacityAvailabilitySource::Domain(CapacityDomainId::new(4).unwrap());
        let source_b = CapacityAvailabilitySource::ActiveSequenceSlots;
        let condition_for = |sources: &[(CapacityAvailabilitySource, u64)]| {
            CapacityWaitCondition::from_observation(
                41,
                sources
                    .iter()
                    .map(|(source, epoch)| CapacityAvailabilityEpoch::new(*source, *epoch).unwrap())
                    .collect(),
            )
            .unwrap()
        };
        let only_a = condition_for(&[(source_a, 1)]);
        let a_and_b = condition_for(&[(source_a, 1), (source_b, 1)]);
        let only_b = condition_for(&[(source_b, 1)]);
        let mut coordinator = PressureCoordinator::default();

        let episode = coordinator.find_or_open_episode_for(&only_a, &[]).unwrap();
        assert_eq!(
            coordinator.find_or_open_episode_for(&a_and_b, &[]).unwrap(),
            episode
        );
        assert_eq!(
            coordinator.find_or_open_episode_for(&only_b, &[]).unwrap(),
            episode
        );
        assert_eq!(coordinator.stats().active_episodes, 1);
    }

    #[test]
    fn shared_device_budget_does_not_join_disjoint_release_domains() {
        let wait_for = |domain: u32| {
            CapacityWaitCondition::from_observation(
                41,
                vec![
                    CapacityAvailabilityEpoch::new(
                        CapacityAvailabilitySource::Domain(CapacityDomainId::new(domain).unwrap()),
                        1,
                    )
                    .unwrap(),
                    CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::PlanDeviceBudget, 7)
                        .unwrap(),
                ],
            )
            .unwrap()
        };
        let domain_two = wait_for(2);
        let domain_four = wait_for(4);
        let mut coordinator = PressureCoordinator::default();

        assert!(!waits_overlap(&domain_two, &domain_four));
        let first = coordinator
            .find_or_open_episode_for(&domain_two, &[])
            .unwrap();
        let second = coordinator
            .find_or_open_episode_for(&domain_four, &[])
            .unwrap();

        assert_ne!(first, second);
        assert_eq!(coordinator.stats().active_episodes, 2);
        assert_eq!(coordinator.source_index.len(), 2);
        assert!(coordinator
            .source_index
            .keys()
            .all(|(_, source)| is_request_release_source(*source)));
    }

    #[test]
    fn one_frontier_joins_disjoint_sources_to_its_existing_episode() {
        let request_id = RequestId::new();
        let source_a = condition(1);
        let source_b = CapacityWaitCondition::from_observation(
            41,
            vec![CapacityAvailabilityEpoch::new(
                CapacityAvailabilitySource::ActiveSequenceSlots,
                1,
            )
            .unwrap()],
        )
        .unwrap();
        let mut coordinator = PressureCoordinator::default();

        coordinator
            .plan_failure(
                std::slice::from_ref(&request_id),
                &source_a,
                &[candidate(
                    &request_id,
                    LogicalWorkKind::Decode,
                    3,
                    true,
                    None,
                )],
            )
            .unwrap();
        let first_episode = coordinator.request_index[&request_id];
        coordinator
            .plan_failure(
                std::slice::from_ref(&request_id),
                &source_b,
                &[candidate(
                    &request_id,
                    LogicalWorkKind::Decode,
                    3,
                    true,
                    Some(source_a.clone()),
                )],
            )
            .unwrap();

        assert_eq!(coordinator.request_index[&request_id], first_episode);
        assert_eq!(coordinator.stats().active_episodes, 1);
        assert_eq!(coordinator.source_index.len(), 2);
    }

    #[test]
    fn repeated_failure_preserves_oldest_blocked_frontier() {
        let oldest = RequestId::new();
        let newer = RequestId::new();
        let wait = condition(7);
        let mut coordinator = PressureCoordinator::default();
        let initial = vec![
            candidate(&oldest, LogicalWorkKind::Decode, 3, true, None),
            candidate(&newer, LogicalWorkKind::Decode, 3, true, None),
        ];
        assert!(matches!(
            coordinator.plan_failure(std::slice::from_ref(&oldest), &wait, &initial),
            Ok(PressureDecision::Deferred { count: 1, .. })
        ));

        let repeated = vec![
            candidate(
                &oldest,
                LogicalWorkKind::Decode,
                3,
                true,
                Some(wait.clone()),
            ),
            candidate(&newer, LogicalWorkKind::Decode, 3, true, None),
        ];
        assert!(matches!(
            coordinator.plan_failure(std::slice::from_ref(&oldest), &wait, &repeated),
            Ok(PressureDecision::Deferred { count: 1, .. })
        ));
        assert_eq!(
            coordinator
                .journal()
                .iter()
                .filter(|transition| {
                    transition.kind == PressureTransitionKind::FrontierBlocked
                        && transition.request_id.as_ref() == Some(&oldest)
                })
                .count(),
            1
        );

        let all_blocked = vec![
            repeated[0].clone(),
            candidate(&newer, LogicalWorkKind::Decode, 3, true, None),
        ];
        let PressureDecision::YieldPlanned(transaction) = coordinator
            .plan_failure(std::slice::from_ref(&newer), &wait, &all_blocked)
            .unwrap()
        else {
            panic!("second blocked frontier must plan one yield");
        };
        assert_eq!(transaction.progress_owner_id(), &oldest);
        assert_eq!(transaction.victim_request_id(), &newer);
    }

    #[test]
    fn terminal_single_waiter_closes_its_empty_episode() {
        let request_id = RequestId::new();
        let wait = condition(7);
        let mut coordinator = PressureCoordinator::default();
        coordinator
            .plan_failure(
                std::slice::from_ref(&request_id),
                &wait,
                &[candidate(
                    &request_id,
                    LogicalWorkKind::Decode,
                    3,
                    false,
                    None,
                )],
            )
            .unwrap();
        assert_eq!(coordinator.stats().active_episodes, 1);

        coordinator.record_terminal(&request_id).unwrap();

        assert_eq!(coordinator.stats().active_episodes, 0);
        assert!(!coordinator.has_records());
    }

    #[test]
    fn single_releasable_frontier_self_recomputes_after_its_release_fence() {
        let request_id = RequestId::new();
        let wait = condition(73);
        let mut coordinator = PressureCoordinator::default();

        let PressureDecision::YieldPlanned(transaction) = coordinator
            .plan_failure(
                std::slice::from_ref(&request_id),
                &wait,
                &[candidate(
                    &request_id,
                    LogicalWorkKind::Decode,
                    33,
                    true,
                    None,
                )],
            )
            .unwrap()
        else {
            panic!("a self-blocked releasable frontier must plan recompute");
        };
        assert_eq!(transaction.kind(), PressureYieldKind::SelfRecompute);
        assert_eq!(transaction.victim_request_id(), &request_id);
        assert_eq!(transaction.progress_owner_id(), &request_id);

        let armed = coordinator.arm_release_fence(&transaction).unwrap();
        let (released, disposition) = coordinator
            .complete_release_fence(&transaction, None)
            .unwrap();
        let PressureReleaseFenceDisposition::SelfRecomputeQueued(closed) = disposition else {
            panic!("self release must queue recompute instead of resuming stale decode");
        };
        assert!(transaction.planned_ordinal() < armed);
        assert!(armed < released);
        assert!(released < closed);
        assert_eq!(coordinator.stats().active_episodes, 0);
        assert_eq!(coordinator.stats().pending_release_fences, 0);
        assert!(!coordinator.has_records());
    }

    #[test]
    fn retained_recompute_never_self_recomputes_without_logical_progress() {
        let request_id = RequestId::new();
        let wait = condition(73);
        let mut coordinator = PressureCoordinator::default();

        let decision = coordinator
            .plan_failure(
                std::slice::from_ref(&request_id),
                &wait,
                &[candidate(
                    &request_id,
                    LogicalWorkKind::Recompute,
                    33,
                    true,
                    None,
                )],
            )
            .unwrap();

        assert!(matches!(
            decision,
            PressureDecision::Deferred { count: 1, .. }
        ));
        assert_eq!(coordinator.stats().pending_release_fences, 0);
        assert!(coordinator
            .journal()
            .iter()
            .all(|transition| transition.kind() != PressureTransitionKind::YieldPlanned));
    }

    #[test]
    fn cuda_da9_cross_phase_terminal_state_plans_a_yield_before_park() {
        let mut coordinator = PressureCoordinator::default();
        let decode = RequestId::new();
        let recompute = RequestId::new();
        let wait = condition(73);

        let first = coordinator
            .plan_failure(
                std::slice::from_ref(&recompute),
                &wait,
                &[
                    candidate(&decode, LogicalWorkKind::Decode, 81, true, None),
                    candidate(&recompute, LogicalWorkKind::Recompute, 33, true, None),
                ],
            )
            .unwrap();
        assert!(matches!(first, PressureDecision::Deferred { count: 1, .. }));

        let decision = coordinator
            .plan_failure(
                std::slice::from_ref(&decode),
                &wait,
                &[
                    candidate(&decode, LogicalWorkKind::Decode, 81, true, None),
                    candidate(
                        &recompute,
                        LogicalWorkKind::Recompute,
                        33,
                        true,
                        Some(wait.clone()),
                    ),
                ],
            )
            .unwrap();
        let PressureDecision::YieldPlanned(transaction) = decision else {
            panic!("cross-phase all-blocked state must plan a release");
        };
        assert_eq!(transaction.progress_owner_id(), &recompute);
        assert_eq!(transaction.victim_request_id(), &decode);
        assert!(!coordinator.all_blocked_without_release_for(&wait));
        assert!(coordinator.has_pending_release_for(&wait));
        assert!(transaction.planned_ordinal().get() > 0);
    }

    #[test]
    fn release_fence_precedes_resumable_and_owner_terminal_releases_victim() {
        let mut coordinator = PressureCoordinator::default();
        let owner = RequestId::new();
        let victim = RequestId::new();
        let wait = condition(73);
        coordinator
            .plan_failure(
                std::slice::from_ref(&owner),
                &wait,
                &[
                    candidate(&owner, LogicalWorkKind::Recompute, 33, true, None),
                    candidate(&victim, LogicalWorkKind::Decode, 81, true, None),
                ],
            )
            .unwrap();
        let transaction = match coordinator
            .plan_failure(
                std::slice::from_ref(&victim),
                &wait,
                &[
                    candidate(
                        &owner,
                        LogicalWorkKind::Recompute,
                        33,
                        true,
                        Some(wait.clone()),
                    ),
                    candidate(&victim, LogicalWorkKind::Decode, 81, true, None),
                ],
            )
            .unwrap()
        {
            PressureDecision::YieldPlanned(transaction) => transaction,
            other => panic!("expected yield, got {other:?}"),
        };

        let armed = coordinator.arm_release_fence(&transaction).unwrap();
        assert!(matches!(
            coordinator.hold_status(&victim),
            PressureHoldStatus::Held { .. }
        ));
        let (released, disposition) = coordinator
            .complete_release_fence(&transaction, None)
            .unwrap();
        let PressureReleaseFenceDisposition::Resumable(resumable) = disposition else {
            panic!("live owner must become resumable");
        };
        assert!(transaction.planned_ordinal() < armed);
        assert!(armed < released);
        assert!(released < resumable);
        assert!(matches!(
            coordinator.hold_status(&victim),
            PressureHoldStatus::Held { .. }
        ));

        coordinator
            .record_progress(&owner, LogicalWorkGeneration(34))
            .unwrap();
        assert!(matches!(
            coordinator.hold_status(&victim),
            PressureHoldStatus::Held { .. }
        ));
        assert_eq!(coordinator.stats().active_episodes, 1);

        coordinator.record_terminal(&owner).unwrap();
        assert!(matches!(
            coordinator.hold_status(&victim),
            PressureHoldStatus::Released {
                reason: PressureHoldReleaseReason::OwnerTerminal,
                ..
            }
        ));
        assert_eq!(coordinator.stats().active_episodes, 0);
        let ordinals = coordinator
            .journal()
            .into_iter()
            .map(|event| event.ordinal().get())
            .collect::<Vec<_>>();
        assert!(ordinals.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn stable_owner_recomputes_without_promoting_held_peer() {
        let mut coordinator = PressureCoordinator::default();
        let owner = RequestId::new();
        let held_peer = RequestId::new();
        let first_wait = condition(73);
        coordinator
            .plan_failure(
                std::slice::from_ref(&owner),
                &first_wait,
                &[
                    candidate(&owner, LogicalWorkKind::Decode, 81, true, None),
                    candidate(&held_peer, LogicalWorkKind::Decode, 33, true, None),
                ],
            )
            .unwrap();
        let peer_yield = match coordinator
            .plan_failure(
                std::slice::from_ref(&held_peer),
                &first_wait,
                &[
                    candidate(
                        &owner,
                        LogicalWorkKind::Decode,
                        81,
                        true,
                        Some(first_wait.clone()),
                    ),
                    candidate(&held_peer, LogicalWorkKind::Decode, 33, true, None),
                ],
            )
            .unwrap()
        {
            PressureDecision::YieldPlanned(transaction) => transaction,
            other => panic!("expected peer yield, got {other:?}"),
        };
        assert_eq!(peer_yield.progress_owner_id(), &owner);
        assert_eq!(peer_yield.victim_request_id(), &held_peer);
        coordinator.arm_release_fence(&peer_yield).unwrap();
        assert!(matches!(
            coordinator
                .complete_release_fence(&peer_yield, None)
                .unwrap()
                .1,
            PressureReleaseFenceDisposition::Resumable(_)
        ));
        assert!(matches!(
            coordinator.hold_status(&held_peer),
            PressureHoldStatus::Held { .. }
        ));

        let advanced_wait = condition(154);
        let owner_recompute = match coordinator
            .plan_failure(
                std::slice::from_ref(&owner),
                &advanced_wait,
                &[
                    candidate(&owner, LogicalWorkKind::Decode, 145, true, None),
                    candidate(&held_peer, LogicalWorkKind::Waiting, 33, false, None),
                ],
            )
            .unwrap()
        {
            PressureDecision::YieldPlanned(transaction) => transaction,
            other => panic!("blocked stable owner must self recompute, got {other:?}"),
        };
        assert_eq!(owner_recompute.kind(), PressureYieldKind::SelfRecompute);
        assert_eq!(owner_recompute.progress_owner_id(), &owner);
        assert_eq!(owner_recompute.victim_request_id(), &owner);
        coordinator.arm_release_fence(&owner_recompute).unwrap();
        let (_, disposition) = coordinator
            .complete_release_fence(&owner_recompute, None)
            .unwrap();
        assert!(matches!(
            disposition,
            PressureReleaseFenceDisposition::OwnerAdmissionPending(_)
        ));
        assert_eq!(coordinator.stats().active_episodes, 1);
        assert!(matches!(
            coordinator.hold_status(&owner),
            PressureHoldStatus::OwnerAdmissionEligible { .. }
        ));
        assert!(matches!(
            coordinator.hold_status(&held_peer),
            PressureHoldStatus::Held { .. }
        ));

        let admitted = coordinator.consume_released_hold(&owner).unwrap();
        assert!(admitted.is_some());
        assert!(matches!(
            coordinator.hold_status(&owner),
            PressureHoldStatus::None
        ));
        assert!(matches!(
            coordinator.hold_status(&held_peer),
            PressureHoldStatus::Held { .. }
        ));

        coordinator.record_terminal(&owner).unwrap();
        assert!(matches!(
            coordinator.hold_status(&held_peer),
            PressureHoldStatus::Released {
                reason: PressureHoldReleaseReason::OwnerTerminal,
                ..
            }
        ));
        assert_eq!(coordinator.stats().active_episodes, 0);
    }

    #[test]
    fn release_fence_keeps_peer_held_when_owner_wait_source_retargets() {
        let wait_for = |domain: u32, domain_epoch: u64| {
            CapacityWaitCondition::from_observation(
                41,
                vec![
                    CapacityAvailabilityEpoch::new(
                        CapacityAvailabilitySource::Domain(CapacityDomainId::new(domain).unwrap()),
                        domain_epoch,
                    )
                    .unwrap(),
                    CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::PlanDeviceBudget, 1)
                        .unwrap(),
                ],
            )
            .unwrap()
        };
        let mut coordinator = PressureCoordinator::default();
        let owner = RequestId::new();
        let victim = RequestId::new();
        let original_wait = wait_for(4, 136);
        let retargeted_wait = wait_for(2, 178);

        coordinator
            .plan_failure(
                std::slice::from_ref(&owner),
                &original_wait,
                &[
                    candidate(&owner, LogicalWorkKind::Recompute, 33, true, None),
                    candidate(&victim, LogicalWorkKind::Decode, 81, true, None),
                ],
            )
            .unwrap();
        let transaction = match coordinator
            .plan_failure(
                std::slice::from_ref(&victim),
                &original_wait,
                &[
                    candidate(
                        &owner,
                        LogicalWorkKind::Recompute,
                        33,
                        true,
                        Some(original_wait.clone()),
                    ),
                    candidate(&victim, LogicalWorkKind::Decode, 81, true, None),
                ],
            )
            .unwrap()
        {
            PressureDecision::YieldPlanned(transaction) => transaction,
            other => panic!("expected yield, got {other:?}"),
        };

        let armed = coordinator.arm_release_fence(&transaction).unwrap();
        let (released, disposition) = coordinator
            .complete_release_fence(&transaction, Some(&retargeted_wait))
            .unwrap();
        let PressureReleaseFenceDisposition::Resumable(resumable) = disposition else {
            panic!("retargeted owner must remain the stable progress owner");
        };
        assert!(armed < released);
        assert!(released < resumable);
        assert_eq!(coordinator.stats().active_episodes, 1);
        assert!(matches!(
            coordinator.hold_status(&victim),
            PressureHoldStatus::Held { .. }
        ));
        let episode_events = coordinator
            .journal()
            .into_iter()
            .filter(|event| event.episode_id() == transaction.episode_id())
            .collect::<Vec<_>>();
        let completed = episode_events
            .iter()
            .find(|event| event.kind() == PressureTransitionKind::ReleaseFenceCompleted)
            .unwrap();
        let retargeted = episode_events
            .iter()
            .find(|event| event.kind() == PressureTransitionKind::FrontierRetargeted)
            .unwrap();
        assert!(episode_events
            .iter()
            .all(|event| event.kind() != PressureTransitionKind::Closed));
        assert!(completed.ordinal() < retargeted.ordinal());
        assert!(retargeted.ordinal() < resumable);

        coordinator.record_terminal(&owner).unwrap();
        assert!(matches!(
            coordinator.hold_status(&victim),
            PressureHoldStatus::Released {
                reason: PressureHoldReleaseReason::OwnerTerminal,
                ..
            }
        ));
        assert_eq!(coordinator.stats().active_episodes, 0);
    }

    #[test]
    fn owner_terminal_during_release_fence_closes_after_physical_release() {
        let mut coordinator = PressureCoordinator::default();
        let owner = RequestId::new();
        let victim = RequestId::new();
        let wait = condition(73);
        coordinator
            .plan_failure(
                std::slice::from_ref(&owner),
                &wait,
                &[
                    candidate(&owner, LogicalWorkKind::Recompute, 33, true, None),
                    candidate(&victim, LogicalWorkKind::Decode, 81, true, None),
                ],
            )
            .unwrap();
        let transaction = match coordinator
            .plan_failure(
                std::slice::from_ref(&victim),
                &wait,
                &[
                    candidate(
                        &owner,
                        LogicalWorkKind::Recompute,
                        33,
                        true,
                        Some(wait.clone()),
                    ),
                    candidate(&victim, LogicalWorkKind::Decode, 81, true, None),
                ],
            )
            .unwrap()
        {
            PressureDecision::YieldPlanned(transaction) => transaction,
            other => panic!("expected yield, got {other:?}"),
        };
        coordinator.arm_release_fence(&transaction).unwrap();

        coordinator.record_terminal(&owner).unwrap();

        assert_eq!(coordinator.stats().active_episodes, 1);
        assert_eq!(coordinator.stats().pending_release_fences, 1);
        let (released, disposition) = coordinator
            .complete_release_fence(&transaction, None)
            .unwrap();
        let PressureReleaseFenceDisposition::Closed {
            ordinal: closed,
            reason,
        } = disposition
        else {
            panic!("terminal owner must close the episode");
        };
        assert_eq!(reason, PressureHoldReleaseReason::OwnerTerminal);
        assert!(released < closed);
        assert_eq!(coordinator.stats().active_episodes, 0);
        assert!(matches!(
            coordinator.hold_status(&victim),
            PressureHoldStatus::Released {
                reason: PressureHoldReleaseReason::OwnerTerminal,
                ..
            }
        ));
    }

    #[test]
    fn aborted_yield_closes_pending_release_without_publishing_a_hold() {
        let mut coordinator = PressureCoordinator::default();
        let owner = RequestId::new();
        let victim = RequestId::new();
        let wait = condition(73);
        coordinator
            .plan_failure(
                std::slice::from_ref(&owner),
                &wait,
                &[
                    candidate(&owner, LogicalWorkKind::Decode, 3, true, None),
                    candidate(&victim, LogicalWorkKind::Decode, 3, true, None),
                ],
            )
            .unwrap();
        let transaction = match coordinator
            .plan_failure(
                std::slice::from_ref(&victim),
                &wait,
                &[
                    candidate(&owner, LogicalWorkKind::Decode, 3, true, Some(wait.clone())),
                    candidate(&victim, LogicalWorkKind::Decode, 3, true, None),
                ],
            )
            .unwrap()
        {
            PressureDecision::YieldPlanned(transaction) => transaction,
            other => panic!("expected yield, got {other:?}"),
        };
        coordinator.arm_release_fence(&transaction).unwrap();

        let (aborted, closed, participants) = coordinator.abort_yield(&transaction).unwrap();

        assert!(aborted < closed);
        assert_eq!(participants.len(), 2);
        assert_eq!(coordinator.stats().active_episodes, 0);
        assert!(!coordinator.has_records());
        assert!(matches!(
            coordinator.hold_status(&victim),
            PressureHoldStatus::None
        ));
    }

    #[test]
    fn blocked_recompute_owner_reports_explicit_invariant_without_releasing_peer() {
        let mut coordinator = PressureCoordinator::default();
        let owner = RequestId::new();
        let victim = RequestId::new();
        let wait_for = |domain: u32, domain_epoch: u64| {
            CapacityWaitCondition::from_observation(
                41,
                vec![
                    CapacityAvailabilityEpoch::new(
                        CapacityAvailabilitySource::Domain(CapacityDomainId::new(domain).unwrap()),
                        domain_epoch,
                    )
                    .unwrap(),
                    CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::PlanDeviceBudget, 1)
                        .unwrap(),
                ],
            )
            .unwrap()
        };
        let decode_wait = wait_for(4, 109);
        let recompute_wait = wait_for(2, 216);

        coordinator
            .plan_failure(
                std::slice::from_ref(&owner),
                &decode_wait,
                &[
                    candidate(&owner, LogicalWorkKind::Recompute, 33, true, None),
                    candidate(&victim, LogicalWorkKind::Decode, 81, true, None),
                ],
            )
            .unwrap();
        let transaction = match coordinator
            .plan_failure(
                std::slice::from_ref(&victim),
                &decode_wait,
                &[
                    candidate(
                        &owner,
                        LogicalWorkKind::Recompute,
                        33,
                        true,
                        Some(decode_wait.clone()),
                    ),
                    candidate(&victim, LogicalWorkKind::Decode, 81, true, None),
                ],
            )
            .unwrap()
        {
            PressureDecision::YieldPlanned(transaction) => transaction,
            other => panic!("expected yield, got {other:?}"),
        };
        assert_eq!(transaction.progress_owner_id(), &owner);
        assert_eq!(transaction.victim_request_id(), &victim);
        let episode_id = transaction.episode_id();
        coordinator.arm_release_fence(&transaction).unwrap();
        coordinator
            .complete_release_fence(&transaction, None)
            .unwrap();

        let decision = coordinator
            .plan_failure(
                std::slice::from_ref(&owner),
                &recompute_wait,
                &[
                    candidate(&owner, LogicalWorkKind::Recompute, 33, true, None),
                    candidate(&victim, LogicalWorkKind::Waiting, 81, false, None),
                ],
            )
            .unwrap();
        assert!(matches!(
            decision,
            PressureDecision::InvariantViolation(PressureInvariantViolation {
                episode_id: actual_episode,
                class: PressureInvariantViolationClass::OwnerRecomputeBlocked,
                blocked_frontiers: 2,
            }) if actual_episode == episode_id
        ));
        assert_eq!(coordinator.stats().active_episodes, 1);
        assert!(matches!(
            coordinator.hold_status(&victim),
            PressureHoldStatus::Held { .. }
        ));
        assert!(coordinator
            .journal()
            .iter()
            .all(|event| event.kind() != PressureTransitionKind::Closed));
    }

    #[test]
    fn unchanged_source_reports_blind_role_rotation_invariant() {
        let mut coordinator = PressureCoordinator::default();
        let owner = RequestId::new();
        let victim = RequestId::new();
        let wait = condition(73);
        let _ = coordinator
            .plan_failure(
                std::slice::from_ref(&owner),
                &wait,
                &[
                    candidate(
                        &owner,
                        LogicalWorkKind::Recompute,
                        33,
                        true,
                        Some(wait.clone()),
                    ),
                    candidate(&victim, LogicalWorkKind::Decode, 81, true, None),
                ],
            )
            .unwrap();
        let transaction = match coordinator
            .plan_failure(
                std::slice::from_ref(&victim),
                &wait,
                &[
                    candidate(
                        &owner,
                        LogicalWorkKind::Recompute,
                        33,
                        true,
                        Some(wait.clone()),
                    ),
                    candidate(&victim, LogicalWorkKind::Decode, 81, true, None),
                ],
            )
            .unwrap()
        {
            PressureDecision::YieldPlanned(transaction) => transaction,
            other => panic!("expected yield, got {other:?}"),
        };
        coordinator.arm_release_fence(&transaction).unwrap();
        coordinator
            .complete_release_fence(&transaction, None)
            .unwrap();

        let decision = coordinator
            .plan_failure(
                std::slice::from_ref(&owner),
                &wait,
                &[
                    candidate(&owner, LogicalWorkKind::Recompute, 33, true, None),
                    candidate(&victim, LogicalWorkKind::Waiting, 81, false, None),
                ],
            )
            .unwrap();
        assert!(matches!(
            decision,
            PressureDecision::InvariantViolation(PressureInvariantViolation {
                class: PressureInvariantViolationClass::YieldDidNotAdvanceSource,
                ..
            })
        ));
    }
}
