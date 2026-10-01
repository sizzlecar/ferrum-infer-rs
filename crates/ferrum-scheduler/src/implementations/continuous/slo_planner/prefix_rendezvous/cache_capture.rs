use super::*;

/// An optional cache publication at a model-declared source boundary.
/// It creates no follower, hold, restored frontier or future cache-hit claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixCacheCaptureOffer {
    pub identity: [u8; 32],
    pub based_on_generation: u64,
    pub source: RequestWorkKey,
    /// AtBoundary: actual completed span start. Preparing: actual initial offset;
    /// the eventual capture span comes from the last projected source wave.
    pub capture_span_start: u32,
    pub boundary_tokens: NonZeroU32,
    /// The original captured horizon; neither search nor replay may renew it.
    pub expires_at_ns: u64,
}

/// The provider must bind Preparing to its declared boundary without treating
/// the numeric initial offset as a completed checkpoint proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixCacheCapturePhase {
    AtBoundary,
    Preparing,
}

pub struct PlanningPrefixCacheCaptureBindingInput<'a> {
    pub snapshot: &'a SchedulerSnapshot,
    pub offer: &'a PrefixCacheCaptureOffer,
    pub phase: PrefixCacheCapturePhase,
}

pub struct PlanningPrefixCacheCaptureInput<'a> {
    pub snapshot: &'a SchedulerSnapshot,
    pub offer: &'a PrefixCacheCaptureOffer,
    pub capture_span_start: u32,
    pub requests: &'a [RequestSchedulingView],
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrefixCacheCaptureAction {
    Wave(SelectedWave),
    Capture(PrefixMaintenanceEvidence),
}

/// Only a complete independent replay can construct this first-action evidence.
/// Subsequent ordinary waves require their own fresh planning and final guards.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefixCacheCaptureEvidence {
    pub(super) offer: PrefixCacheCaptureOffer,
    pub(super) phase: PrefixCacheCapturePhase,
    pub(super) action: PrefixCacheCaptureAction,
    pub(super) inference_model_version: u64,
    pub(super) maintenance_model_version: u64,
    pub(super) validated_at_ns: u64,
    pub(super) valid_until_ns: u64,
    pub(super) first_action_cost_ns: u64,
    pub(super) protection: Arc<super::super::obligations::PlanningObligationSet>,
    pub(super) capture: PrefixMaintenanceEvidence,
    pub(super) steps: Vec<PrefixPathStep>,
    pub(super) completion_at_ns: u64,
}
impl PrefixCacheCaptureEvidence {
    pub fn phase(&self) -> PrefixCacheCapturePhase {
        self.phase
    }
    pub fn action(&self) -> &PrefixCacheCaptureAction {
        &self.action
    }
    pub fn offer(&self) -> &PrefixCacheCaptureOffer {
        &self.offer
    }
    pub fn inference_model_version(&self) -> u64 {
        self.inference_model_version
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
    pub fn capture(&self) -> &PrefixMaintenanceEvidence {
        &self.capture
    }
    pub fn steps(&self) -> &[PrefixPathStep] {
        &self.steps
    }
    pub fn completion_at_ns(&self) -> u64 {
        self.completion_at_ns
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrefixCacheCaptureDecision {
    Known {
        evidence: PrefixCacheCaptureEvidence,
        search: PlanningSearchStats,
    },
    Unknown {
        reason: PlanningUnknownReason,
        search: PlanningSearchStats,
    },
}
