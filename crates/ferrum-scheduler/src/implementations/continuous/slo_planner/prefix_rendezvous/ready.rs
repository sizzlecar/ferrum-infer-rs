use super::*;

/// A currently retained immutable cache entry and its admitted untouched target.
/// There is no producer dependency or configurable waiting period. The engine
/// binds identity to the actual lease and current input/owner authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadyPrefixRestoreOffer {
    pub identity: [u8; 32],
    pub based_on_generation: u64,
    pub target: RequestWorkKey,
    pub boundary_tokens: NonZeroU32,
    /// Original proof lifetime on the request clock; never renewed after ack.
    pub expires_at_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadyPrefixPhase {
    Ready,
    Restored,
}

pub struct PlanningReadyPrefixInput<'a> {
    pub snapshot: &'a SchedulerSnapshot,
    pub offer: &'a ReadyPrefixRestoreOffer,
    pub phase: ReadyPrefixPhase,
}

pub struct PlanningReadyPrefixRestoreInput<'a> {
    pub snapshot: &'a SchedulerSnapshot,
    pub offer: &'a ReadyPrefixRestoreOffer,
    pub requests: &'a [RequestSchedulingView],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadyPrefixRestoreEvidence {
    pub cost_domain: PlanningShapeDomain<WaveExecutionShape>,
    pub restored_frontier: PrefixRestoredFrontier,
    pub model_version: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReadyPrefixAction {
    Restore(ReadyPrefixRestoreEvidence),
    Wave(SelectedWave),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReadyPrefixEvidence {
    pub(super) offer: ReadyPrefixRestoreOffer,
    pub(super) phase: ReadyPrefixPhase,
    pub(super) maintenance_model_version: u64,
    pub(super) validated_at_ns: u64,
    pub(super) valid_until_ns: u64,
    pub(super) first_action_cost_ns: u64,
    pub(super) protection: Arc<super::super::obligations::PlanningObligationSet>,
    pub(super) action: ReadyPrefixAction,
    pub(super) remaining: PrefixPathEvidence,
    pub(super) direct: Option<PrefixPathEvidence>,
}

impl ReadyPrefixEvidence {
    pub fn offer(&self) -> &ReadyPrefixRestoreOffer {
        &self.offer
    }
    pub fn phase(&self) -> ReadyPrefixPhase {
        self.phase
    }
    pub fn maintenance_model_version(&self) -> u64 {
        self.maintenance_model_version
    }
    pub fn validated_at_ns(&self) -> u64 {
        self.validated_at_ns
    }
    pub fn valid_until_ns(&self) -> u64 {
        self.valid_until_ns
    }
    pub fn first_action_cost_ns(&self) -> u64 {
        self.first_action_cost_ns
    }
    pub fn protection(&self) -> &Arc<super::super::obligations::PlanningObligationSet> {
        &self.protection
    }
    pub fn action(&self) -> &ReadyPrefixAction {
        &self.action
    }
    pub fn remaining(&self) -> &PrefixPathEvidence {
        &self.remaining
    }
    pub fn direct(&self) -> Option<&PrefixPathEvidence> {
        self.direct.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReadyPrefixDecision {
    Ready {
        evidence: ReadyPrefixEvidence,
        search: PlanningSearchStats,
    },
    PreferDirect {
        direct: PrefixPathEvidence,
        restoring: PrefixPathEvidence,
        search: PlanningSearchStats,
    },
    Unknown {
        reason: PlanningUnknownReason,
        search: PlanningSearchStats,
    },
}

/// Shared search and simulation keep the two public protocols distinct.
#[derive(Clone, Copy)]
pub(in super::super::super) enum PathOffer<'a> {
    Rendezvous(&'a PrefixRendezvousOffer),
    Ready(&'a ReadyPrefixRestoreOffer),
    CacheCapture(&'a PrefixCacheCaptureOffer),
}

impl<'a> PathOffer<'a> {
    pub(in super::super::super) fn target(self) -> Option<&'a RequestWorkKey> {
        match self {
            Self::Rendezvous(o) => Some(&o.target),
            Self::Ready(o) => Some(&o.target),
            Self::CacheCapture(_) => None,
        }
    }
    pub(in super::super::super) fn boundary(self) -> u32 {
        match self {
            Self::Rendezvous(o) => o.boundary_tokens.get(),
            Self::Ready(o) => o.boundary_tokens.get(),
            Self::CacheCapture(o) => o.boundary_tokens.get(),
        }
    }
    pub(in super::super::super) fn expires_at_ns(self) -> u64 {
        match self {
            Self::Rendezvous(o) => o.expires_at_ns,
            Self::Ready(o) => o.expires_at_ns,
            Self::CacheCapture(o) => o.expires_at_ns,
        }
    }
}
