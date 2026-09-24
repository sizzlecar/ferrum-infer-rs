//! Explicit cold projection of the actual call's capture into public evidence.
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    CostCalibrationCapture, CostCalibrationResult, CostCalibrationStatus, CostCallDisposition,
    HostCommitOutcome, HostCommittedWork,
};
use ferrum_interfaces::execution_cost::{ActualWaveRow, HostCostFeaturesV1};
use ferrum_scheduler::implementations::continuous::cost_model::WaveCostObservation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationCommittedWork {
    Prefill {
        start: u32,
        end: u32,
        total_prompt_tokens: u32,
        generated_before: u64,
        generated_after: u64,
    },
    Decode {
        kv_before: u32,
        kv_after: u32,
        generated_before: u64,
        generated_after: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationCommittedRow {
    pub request_id: RequestId,
    pub owner_incarnation: u64,
    pub work_generation: u64,
    pub input_index: u32,
    pub committed_at_ns: u64,
    pub work: CalibrationCommittedWork,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationQueueDisposition {
    Published,
    /// Actual bounded sink rejection, independent of measurement eligibility.
    Dropped {
        reason: String,
    },
}

#[derive(Debug, Clone)]
pub enum CalibrationObservation {
    /// No finished eligible cost call attached to this wave; never a zero cost.
    PendingOrUnavailable,
    ConflictingCalls,
    Rejected {
        reason: String,
    },
    InvalidIdentityJoin,
    Observed {
        sample: Box<WaveCostObservation>,
        actual_rows: Vec<ActualWaveRow>,
        commits: Vec<CalibrationCommittedRow>,
        /// Captured before execution, in actual_rows' physical order.
        /// None is missing evidence, never a policy inferred from CLI intent.
        host_features: Vec<Option<HostCostFeaturesV1>>,
        /// Queue acceptance order. This is not the export source_record.
        accepted_ordinal: Option<u64>,
        disposition: CalibrationQueueDisposition,
    },
}

pub(super) fn project_capture(capture: &CostCalibrationCapture) -> CalibrationObservation {
    match capture.status() {
        CostCalibrationStatus::Pending => CalibrationObservation::PendingOrUnavailable,
        CostCalibrationStatus::ConflictingCalls => CalibrationObservation::ConflictingCalls,
        CostCalibrationStatus::Complete(value) => match value.as_ref() {
            CostCalibrationResult::Rejected(reason) => CalibrationObservation::Rejected {
                reason: format!("{reason:?}"),
            },
            CostCalibrationResult::Observed {
                sample,
                actual_rows,
                commits,
                host_features,
                accepted_ordinal,
                disposition,
            } => {
                if actual_rows.len() != commits.len()
                    || actual_rows.len() != host_features.len()
                    || !matches!(
                        (disposition, accepted_ordinal),
                        (CostCallDisposition::Published, Some(1..))
                            | (CostCallDisposition::Dropped(_), None)
                    )
                {
                    return CalibrationObservation::InvalidIdentityJoin;
                }
                let mut projected = Vec::with_capacity(commits.len());
                for (actual, commit) in actual_rows.iter().zip(commits) {
                    if actual.request_id != commit.request_id
                        || actual.owner_incarnation != commit.owner_incarnation
                        || actual.work_generation != commit.work_generation
                        || actual.input_index != commit.input_index
                    {
                        return CalibrationObservation::InvalidIdentityJoin;
                    }
                    let Some(committed_at_ns) = commit.committed_at_ns else {
                        return CalibrationObservation::InvalidIdentityJoin;
                    };
                    let work = match commit.outcome {
                        HostCommitOutcome::Committed(HostCommittedWork::Prefill {
                            start,
                            end,
                            total_prompt_tokens,
                            generated_tokens_before,
                            generated_tokens_after,
                        }) => CalibrationCommittedWork::Prefill {
                            start,
                            end,
                            total_prompt_tokens,
                            generated_before: generated_tokens_before,
                            generated_after: generated_tokens_after,
                        },
                        HostCommitOutcome::Committed(HostCommittedWork::Decode {
                            kv_tokens_before,
                            kv_tokens_after,
                            generated_tokens_before,
                            generated_tokens_after,
                        }) => CalibrationCommittedWork::Decode {
                            kv_before: kv_tokens_before,
                            kv_after: kv_tokens_after,
                            generated_before: generated_tokens_before,
                            generated_after: generated_tokens_after,
                        },
                        _ => return CalibrationObservation::InvalidIdentityJoin,
                    };
                    projected.push(CalibrationCommittedRow {
                        request_id: commit.request_id.clone(),
                        owner_incarnation: commit.owner_incarnation,
                        work_generation: commit.work_generation,
                        input_index: commit.input_index,
                        committed_at_ns,
                        work,
                    });
                }
                let disposition = match disposition {
                    CostCallDisposition::Published => CalibrationQueueDisposition::Published,
                    CostCallDisposition::Dropped(reason) => CalibrationQueueDisposition::Dropped {
                        reason: format!("{reason:?}"),
                    },
                    CostCallDisposition::Rejected(_) => {
                        return CalibrationObservation::InvalidIdentityJoin
                    }
                };
                CalibrationObservation::Observed {
                    sample: sample.clone(),
                    actual_rows: actual_rows.clone(),
                    commits: projected,
                    host_features: host_features.clone(),
                    accepted_ordinal: *accepted_ordinal,
                    disposition,
                }
            }
        },
    }
}
