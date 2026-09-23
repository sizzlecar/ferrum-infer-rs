//! Passive classification of the work performed by the existing completion
//! call. This is neither a submission permit nor evidence of client delivery.

use ferrum_types::Result;
use serde::Serialize;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutorCompletionWork {
    #[default]
    Unknown,
    NoAdditionalWork,
    AdditionalOrUnproven(ExecutorCompletionActivities),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ExecutorCompletionActivities {
    pub checkpoint_submitted: u32,
    pub submission_indeterminate: bool,
    pub maintenance_entered: bool,
    pub recovery_entered: bool,
    pub evidence_lost: bool,
}

#[derive(Debug)]
pub struct ExecutorCompletionObservation {
    pub result: Result<()>,
    pub work: ExecutorCompletionWork,
}

#[derive(Debug, Clone, Copy)]
pub struct ExecutorAdmissionCancellationObservation {
    pub released: bool,
    pub work: ExecutorCompletionWork,
}
