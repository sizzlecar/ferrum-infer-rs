//! Observation-only projections of an actual submitted batch.
//!
//! These values grant no execution, retry, or resource-release authority. Work
//! is captured from the current admitted wave, independently of device timing
//! and executable qualification, and is bound to the original submission.

use serde::Serialize;

use crate::{
    BatchParticipantTokenRange, BoundDeviceSubmissionAttribution, OperationCompletionReceipt,
    SubmittedOperationReceipt, VNextError,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionFactsCapture {
    #[default]
    Off,
    On,
}

/// Current node-local participant order and packed token windows. Packed token
/// rows are not a claim about a provider's physical matrix or kernel geometry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutionNodeWork {
    node_index: u32,
    participants: Box<[BatchParticipantTokenRange]>,
}

impl ExecutionNodeWork {
    pub(crate) fn new(node_index: u32, participants: Box<[BatchParticipantTokenRange]>) -> Self {
        Self {
            node_index,
            participants,
        }
    }

    pub const fn node_index(&self) -> u32 {
        self.node_index
    }

    pub fn participants(&self) -> &[BatchParticipantTokenRange] {
        &self.participants
    }
}

/// Pure work evidence sharing the original submission receipt. The receipt
/// retains identity facts, not the submitted resources or completion owner.
#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct BoundSubmissionWork {
    submission: SubmittedOperationReceipt,
    nodes: Box<[ExecutionNodeWork]>,
}

impl BoundSubmissionWork {
    pub(crate) fn new(
        submission: SubmittedOperationReceipt,
        nodes: Box<[ExecutionNodeWork]>,
    ) -> Self {
        Self { submission, nodes }
    }

    pub fn submission(&self) -> &SubmittedOperationReceipt {
        &self.submission
    }

    pub fn nodes(&self) -> &[ExecutionNodeWork] {
        &self.nodes
    }
}

/// One submission's optional observations, retained on both normal submission
/// and post-submit contract failure. Missing attribution does not erase work.
#[derive(Debug, Default)]
pub struct SubmissionObservation {
    work: Option<BoundSubmissionWork>,
    attribution: Option<BoundDeviceSubmissionAttribution>,
}

impl SubmissionObservation {
    pub(crate) fn new(
        work: Option<BoundSubmissionWork>,
        attribution: Option<BoundDeviceSubmissionAttribution>,
    ) -> Self {
        Self { work, attribution }
    }

    pub fn work(&self) -> Option<&BoundSubmissionWork> {
        self.work.as_ref()
    }

    pub fn attribution(&self) -> Option<&BoundDeviceSubmissionAttribution> {
        self.attribution.as_ref()
    }

    pub fn into_parts(
        self,
    ) -> (
        Option<BoundSubmissionWork>,
        Option<BoundDeviceSubmissionAttribution>,
    ) {
        (self.work, self.attribution)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceExecutionFactsUnavailableReason {
    /// Current work is known, but actual provider leaf geometry and arithmetic
    /// routes were not captured. Logical attribution cannot fill this gap.
    NotCaptured,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "status", content = "reason", rename_all = "snake_case")]
pub enum DeviceExecutionFactsCoverage {
    Unavailable(DeviceExecutionFactsUnavailableReason),
}

/// Checked borrow of the actual terminal receipt and its submission work.
/// Success is determined solely by `completion().disposition()`. A drained or
/// quarantined indeterminate submission cannot construct this view.
#[derive(Debug, Clone, Copy)]
pub struct TerminalExecutionFacts<'a> {
    completion: &'a OperationCompletionReceipt,
    work: &'a BoundSubmissionWork,
    attribution: Option<&'a BoundDeviceSubmissionAttribution>,
}

impl<'a> TerminalExecutionFacts<'a> {
    pub fn new(
        completion: &'a OperationCompletionReceipt,
        work: &'a BoundSubmissionWork,
        attribution: Option<&'a BoundDeviceSubmissionAttribution>,
    ) -> Result<Self, VNextError> {
        let submission = completion.submission();
        if submission != work.submission()
            || attribution.is_some_and(|attribution| {
                attribution.batch_identity() != submission.batch_identity()
                    || attribution.submission_fingerprint() != submission.fingerprint()
            })
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "terminal execution facts differ from the original submission receipt"
                    .to_owned(),
            });
        }
        Ok(Self {
            completion,
            work,
            attribution,
        })
    }

    pub const fn completion(self) -> &'a OperationCompletionReceipt {
        self.completion
    }

    pub const fn work(self) -> &'a BoundSubmissionWork {
        self.work
    }

    pub const fn attribution(self) -> Option<&'a BoundDeviceSubmissionAttribution> {
        self.attribution
    }

    pub const fn device_coverage(self) -> DeviceExecutionFactsCoverage {
        DeviceExecutionFactsCoverage::Unavailable(
            DeviceExecutionFactsUnavailableReason::NotCaptured,
        )
    }
}
