//! A finite comparison of two complete queue trajectories. Checkpoint steps are
//! explicit physical projections, not prefill rows or execution permissions.
use super::{execution::PlanningExecutionState, types::*};
use crate::implementations::continuous::cost_model::{ExecutionFingerprint, WaveExecutionShape};
use std::{num::NonZeroU32, sync::Arc};

mod cache_capture;
mod ready;
mod search;
pub use cache_capture::{
    PlanningPrefixCacheCaptureBindingInput, PlanningPrefixCacheCaptureInput,
    PrefixCacheCaptureAction, PrefixCacheCaptureDecision, PrefixCacheCaptureEvidence,
    PrefixCacheCaptureOffer, PrefixCacheCapturePhase,
};
pub(in super::super) use ready::PathOffer;
pub use ready::{
    PlanningReadyPrefixInput, PlanningReadyPrefixRestoreInput, ReadyPrefixAction,
    ReadyPrefixDecision, ReadyPrefixEvidence, ReadyPrefixPhase, ReadyPrefixRestoreEvidence,
    ReadyPrefixRestoreOffer,
};

/// Runtime-originated offer. The execution adapter must bind the identity to
/// the captured plan, producer/target incarnations and exact prefix byte plan.
/// Constructing this numeric value grants neither a checkpoint nor a hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixRendezvousOffer {
    pub identity: [u8; 32],
    pub based_on_generation: u64,
    pub producer: RequestWorkKey,
    pub target: RequestWorkKey,
    pub boundary_tokens: NonZeroU32,
    /// Original configuration/retention deadline on the snapshot clock.
    pub expires_at_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixMaintenanceStage {
    Capture,
    Restore,
}

pub struct PlanningPrefixTransitionInput<'a> {
    pub snapshot: &'a SchedulerSnapshot,
    pub offer: &'a PrefixRendezvousOffer,
    pub stage: PrefixMaintenanceStage,
    /// Start of the last producer prefill span in this projected path. Capture
    /// validation must use this span, not an invented zero-based prompt.
    pub capture_span_start: u32,
    /// The entire current queue, including blocked and previously late owners.
    pub requests: &'a [RequestSchedulingView],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixRestoredFrontier {
    pub target: RequestWorkKey,
    pub previous_offset: u32,
    pub restored_tokens: u32,
}

pub struct ProjectedPrefixTransition<'epoch> {
    /// Complete alternatives, each including required initialization and all
    /// transfers/settlement. A likely subset is not a projection.
    pub cost_domain: PlanningShapeDomain<WaveExecutionShape>,
    /// Capture must return None. Restore must return the exact new frontier,
    /// backed by the successor's KV, recurrent state and byte-plan proof.
    pub restored_frontier: Option<PrefixRestoredFrontier>,
    /// Capture retains its private checkpoint; only that lineage may restore.
    /// No live reservation, assumed release or pointer may be fabricated here.
    pub successor: Arc<dyn PlanningExecutionState<'epoch> + 'epoch>,
}

/// Separately calibrated maintenance costs; inference's structured model is
/// not evidence for checkpoint copies or initialization. The adapter binds an
/// immutable model epoch and verifies the complete fingerprint/offer identity.
pub trait PlanningPrefixCostModel {
    fn model_version(&self) -> u64;
    fn predict(
        &self,
        fingerprint: &ExecutionFingerprint,
        offer: &PrefixRendezvousOffer,
        stage: PrefixMaintenanceStage,
        shape: &WaveExecutionShape,
        now_ns: u64,
    ) -> Option<PlanningCost>;

    fn predict_cache_capture(
        &self,
        _fingerprint: &ExecutionFingerprint,
        _offer: &PrefixCacheCaptureOffer,
        _shape: &WaveExecutionShape,
        _now_ns: u64,
    ) -> Option<PlanningCost> {
        None
    }

    fn predict_ready_restore(
        &self,
        _fingerprint: &ExecutionFingerprint,
        _offer: &ReadyPrefixRestoreOffer,
        _shape: &WaveExecutionShape,
        _now_ns: u64,
    ) -> Option<PlanningCost> {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMaintenanceEvidence {
    pub stage: PrefixMaintenanceStage,
    pub capture_span_start: u32,
    pub cost_domain: PlanningShapeDomain<WaveExecutionShape>,
    pub restored_frontier: Option<PrefixRestoredFrontier>,
    pub model_version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixPathStep {
    Wave(WaveCandidate),
    Maintenance(PrefixMaintenanceEvidence),
    ReadyRestore(ReadyPrefixRestoreEvidence),
}

/// Only the planner's independent replay can construct this evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixPathEvidence {
    steps: Vec<PrefixPathStep>,
    first_commit_at_ns: u64,
    completion_at_ns: u64,
    capture_ready_at_ns: Option<u64>,
    restore_ready_at_ns: Option<u64>,
}

impl PrefixPathEvidence {
    pub fn steps(&self) -> &[PrefixPathStep] {
        &self.steps
    }
    pub fn first_commit_at_ns(&self) -> u64 {
        self.first_commit_at_ns
    }
    pub fn completion_at_ns(&self) -> u64 {
        self.completion_at_ns
    }
    pub fn capture_ready_at_ns(&self) -> Option<u64> {
        self.capture_ready_at_ns
    }
    pub fn restore_ready_at_ns(&self) -> Option<u64> {
        self.restore_ready_at_ns
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixRendezvousComparison {
    offer: PrefixRendezvousOffer,
    inference_model_version: u64,
    maintenance_model_version: u64,
    validated_at_ns: u64,
    valid_until_ns: u64,
    direct: PrefixPathEvidence,
    waiting: PrefixPathEvidence,
}

impl PrefixRendezvousComparison {
    pub fn offer(&self) -> &PrefixRendezvousOffer {
        &self.offer
    }
    pub fn direct(&self) -> &PrefixPathEvidence {
        &self.direct
    }
    pub fn waiting(&self) -> &PrefixPathEvidence {
        &self.waiting
    }
    pub fn validated_at_ns(&self) -> u64 {
        self.validated_at_ns
    }
    pub fn valid_until_ns(&self) -> u64 {
        self.valid_until_ns
    }
    pub fn inference_model_version(&self) -> u64 {
        self.inference_model_version
    }
    pub fn maintenance_model_version(&self) -> u64 {
        self.maintenance_model_version
    }
    /// A comparison of discovered feasible trajectories, never an optimality
    /// proof. Engine identity/resource/clock revalidation is still required.
    pub fn should_hold(&self) -> bool {
        self.waiting.first_commit_at_ns < self.direct.first_commit_at_ns
            && self.validated_at_ns < self.valid_until_ns
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixRendezvousDecision {
    Compared {
        comparison: PrefixRendezvousComparison,
        search: PlanningSearchStats,
    },
    /// An enforcing caller must not create a hold from Unknown.
    Unknown {
        reason: PlanningUnknownReason,
        search: PlanningSearchStats,
    },
}

/// Current product phase of the original cohort. These values alone grant no
/// authority: the fresh execution state must bind them to actual completed
/// span, retained-checkpoint or restore-acknowledgement evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixContinuationPhase {
    HeldAwaitingProducer,
    AtCaptureBoundary { capture_span_start: u32 },
    CheckpointReady { capture_span_start: u32 },
    Restored,
}

pub struct PlanningPrefixContinuationInput<'a> {
    pub snapshot: &'a SchedulerSnapshot,
    pub offer: &'a PrefixRendezvousOffer,
    pub phase: PrefixContinuationPhase,
}

/// The first real action only. A model wave retains the ordinary independent
/// replay publication protocol. Maintenance still needs the engine's current
/// owner/resource/checkpoint/cancellation guards before any live operation.
#[derive(Debug, Clone, PartialEq)]
pub enum PrefixContinuationAction {
    Wave(SelectedWave),
    Maintenance(PrefixMaintenanceEvidence),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrefixContinuationEvidence {
    offer: PrefixRendezvousOffer,
    phase: PrefixContinuationPhase,
    maintenance_model_version: u64,
    validated_at_ns: u64,
    valid_until_ns: u64,
    first_action_cost_ns: u64,
    protection: Arc<super::obligations::PlanningObligationSet>,
    action: PrefixContinuationAction,
    remaining: PrefixPathEvidence,
}

impl PrefixContinuationEvidence {
    pub fn offer(&self) -> &PrefixRendezvousOffer {
        &self.offer
    }
    pub fn phase(&self) -> PrefixContinuationPhase {
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
    pub fn protection(&self) -> &Arc<super::obligations::PlanningObligationSet> {
        &self.protection
    }
    pub fn action(&self) -> &PrefixContinuationAction {
        &self.action
    }
    pub fn remaining(&self) -> &PrefixPathEvidence {
        &self.remaining
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrefixContinuationDecision {
    Ready {
        continuation: PrefixContinuationEvidence,
        search: PlanningSearchStats,
    },
    Unknown {
        reason: PlanningUnknownReason,
        search: PlanningSearchStats,
    },
}
