//! Passive, GPU-free evidence for an executor's actual physical waves.
//!
//! This is observation, not resource or execution authority. All timestamps in
//! one recorder use the caller's single monotonic clock; no epoch is persisted.
//! Recorders never perform I/O, serialize, or change an execution decision.
//! The separate submission-guard contract validates this evidence at an actual
//! commit boundary; the evidence itself still grants no resource authority.

use std::{num::NonZeroU64, sync::Arc};

use ferrum_types::RequestId;

mod recorder;
pub use recorder::*;
mod context;
pub use context::*;
mod statistical;
pub use statistical::*;
mod canonical;
pub use canonical::*;
mod features;
pub use features::*;
mod submission_guard;
pub use submission_guard::*;
mod expected_work;
pub use expected_work::*;
#[cfg(test)]
mod tests;

pub const EXECUTOR_COST_IDENTITY_SCHEMA: u32 = 1;

/// Cold-path, domain-separated digests of actual execution-relevant inputs.
/// Request IDs, temporary paths, timestamps and requested Auto policies do not
/// belong here. Hardware identity must distinguish the real execution device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutorCostIdentity {
    pub schema_version: u32,
    pub model_weights: [u8; 32],
    pub numerical_policy: [u8; 32],
    pub device_runtime: [u8; 32],
    pub execution_config: [u8; 32],
}

/// Retains useful cold-path evidence without claiming a complete identity.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutorCostIdentityComponents {
    pub model_weights: Option<[u8; 32]>,
    pub numerical_policy: Option<[u8; 32]>,
    pub device_runtime: Option<[u8; 32]>,
    pub execution_config: Option<[u8; 32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostIdentityUnknownReason {
    Unsupported,
    MissingModelContent,
    MissingNumericalPolicy,
    MissingHardwareIdentity,
    MissingExecutionConfig,
    InvalidSourceEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutorCostIdentityAvailability {
    Known(Arc<ExecutorCostIdentity>),
    Unknown {
        reason: CostIdentityUnknownReason,
        partial: Option<Arc<ExecutorCostIdentityComponents>>,
    },
}

impl Default for ExecutorCostIdentityAvailability {
    fn default() -> Self {
        Self::Unknown {
            reason: CostIdentityUnknownReason::Unsupported,
            partial: None,
        }
    }
}

impl ExecutorCostIdentityComponents {
    pub fn into_availability(self) -> ExecutorCostIdentityAvailability {
        let reason = if self.model_weights.is_none() {
            Some(CostIdentityUnknownReason::MissingModelContent)
        } else if self.numerical_policy.is_none() {
            Some(CostIdentityUnknownReason::MissingNumericalPolicy)
        } else if self.device_runtime.is_none() {
            Some(CostIdentityUnknownReason::MissingHardwareIdentity)
        } else if self.execution_config.is_none() {
            Some(CostIdentityUnknownReason::MissingExecutionConfig)
        } else {
            None
        };
        match reason {
            Some(reason) => ExecutorCostIdentityAvailability::Unknown {
                reason,
                partial: Some(Arc::new(self)),
            },
            None => ExecutorCostIdentityAvailability::Known(Arc::new(ExecutorCostIdentity {
                schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
                model_weights: self.model_weights.unwrap(),
                numerical_policy: self.numerical_policy.unwrap(),
                device_runtime: self.device_runtime.unwrap(),
                execution_config: self.execution_config.unwrap(),
            })),
        }
    }
}

/// A getter must remain Unavailable until the actual observer is connected.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExecutorCostObservationCapability {
    #[default]
    Unavailable,
    SinglePhysicalWave,
    TracedComposite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActualWaveKind {
    Decode,
    Prefill,
    Mixed,
    Restore,
    Maintenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActualWavePath {
    PlanRuntime,
    NativeUnified,
    LegacySplit,
    UnsupportedFallback,
    CapacityFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActualWaveGraphState {
    Disabled,
    Cold,
    Warm,
    /// Configured OnDemand stream, proven eager without candidate preparation,
    /// capture, upload, replay or a cache-state transition. Not graph-disabled.
    ConfiguredEager,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActualWaveRowOrder {
    Ordered,
    /// Only a backend declaration, never inferred by sorting observation rows.
    IndependentRows,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActualRowWork {
    Decode {
        kv_tokens: u32,
    },
    Prefill {
        offset: u32,
        count: u32,
        total_prompt_tokens: u32,
    },
    Restore,
    Maintenance,
}

/// Correlation only: these values confer no right to use a request or its KV.
/// Preserve physical execution order, including mixed row interleaving.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActualWaveRow {
    pub request_id: RequestId,
    pub owner_incarnation: u64,
    pub work_generation: u64,
    pub input_index: u32,
    pub work: ActualRowWork,
}

#[derive(Debug, Clone)]
pub struct ActualWaveShape {
    pub kind: ActualWaveKind,
    pub path: ActualWavePath,
    pub graph: ActualWaveGraphState,
    pub row_order: ActualWaveRowOrder,
    pub provider_signature: [u8; 32],
    pub output_policy_signature: [u8; 32],
    pub numeric_features: Option<CanonicalWaveCostFeatures>,
    pub host_content_features: Option<HostContentCostFeaturesV1>,
    pub row_multiset_features: Option<HostRowMultisetCostFeaturesV2>,
    /// Passive sidecar from the same completed canonical receipt.
    pub statistical_evidence: Option<StatisticalWaveEvidenceV1>,
    pub rows: Vec<ActualWaveRow>,
    pub recurrent_state_bytes: u64,
    pub restore_bytes: u64,
    pub maintenance_bytes: u64,
    pub maintenance_units: u32,
}
// Equality retains the legacy exact contract. Passive statistics must be
// compared explicitly and can never alter an execution/route equality gate.
impl PartialEq for ActualWaveShape {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.path == other.path
            && self.graph == other.graph
            && self.row_order == other.row_order
            && self.provider_signature == other.provider_signature
            && self.output_policy_signature == other.output_policy_signature
            && self.numeric_features == other.numeric_features
            && self.host_content_features == other.host_content_features
            && self.row_multiset_features == other.row_multiset_features
            && self.rows == other.rows
            && self.recurrent_state_bytes == other.recurrent_state_bytes
            && self.restore_bytes == other.restore_bytes
            && self.maintenance_bytes == other.maintenance_bytes
            && self.maintenance_units == other.maintenance_units
    }
}
impl Eq for ActualWaveShape {}

impl ActualWaveShape {
    pub fn validate(&self, max_rows: usize) -> Result<(), CostRecorderError> {
        if self.rows.len() > max_rows {
            return Err(CostRecorderError::RowCapacity);
        }
        if let Some(features) = &self.numeric_features {
            features
                .validate(self.rows.len())
                .map_err(|_| CostRecorderError::InvalidShape)?;
        }
        if let Some(features) = &self.host_content_features {
            features
                .validate()
                .map_err(|_| CostRecorderError::InvalidShape)?;
            if self.numeric_features.is_none() {
                return Err(CostRecorderError::InvalidShape);
            }
        }
        if let Some(features) = &self.row_multiset_features {
            features
                .validate(self.rows.len())
                .map_err(|_| CostRecorderError::InvalidShape)?;
            if self.numeric_features.is_none()
                || features
                    .rows
                    .iter()
                    .zip(&self.rows)
                    .any(|(static_row, actual)| !static_row.role.matches_work(actual.work))
            {
                return Err(CostRecorderError::InvalidShape);
            }
        }
        let mut decode = 0;
        let mut prefill = 0;
        for (index, row) in self.rows.iter().enumerate() {
            if self.rows[..index].iter().any(|other| {
                other.input_index == row.input_index || other.request_id == row.request_id
            }) {
                return Err(CostRecorderError::InvalidShape);
            }
            match row.work {
                ActualRowWork::Decode { kv_tokens } if kv_tokens > 0 => decode += 1,
                ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                } if count > 0
                    && offset
                        .checked_add(count)
                        .is_some_and(|end| end <= total_prompt_tokens) =>
                {
                    prefill += 1
                }
                ActualRowWork::Restore if self.kind == ActualWaveKind::Restore => {}
                ActualRowWork::Maintenance if self.kind == ActualWaveKind::Maintenance => {}
                _ => return Err(CostRecorderError::InvalidShape),
            }
        }
        let valid = match self.kind {
            ActualWaveKind::Decode => decode > 0 && prefill == 0,
            ActualWaveKind::Prefill => prefill > 0 && decode == 0,
            ActualWaveKind::Mixed => prefill > 0 && decode > 0,
            ActualWaveKind::Restore => decode == 0 && prefill == 0 && self.restore_bytes > 0,
            ActualWaveKind::Maintenance => {
                decode == 0
                    && prefill == 0
                    && self.maintenance_bytes > 0
                    && self.maintenance_units > 0
            }
        };
        if !valid
            || (self.kind != ActualWaveKind::Restore && self.restore_bytes != 0)
            || (self.kind != ActualWaveKind::Maintenance
                && (self.maintenance_bytes != 0 || self.maintenance_units != 0))
        {
            return Err(CostRecorderError::InvalidShape);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActualWaveOutcome {
    Completed,
    /// Preserve evidence; this contract does not infer an isolated partial shape.
    PartiallyCompleted,
    NotSubmitted,
    Deferred,
    FailedAfterSubmit,
    SubmissionIndeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveObservationBoundary {
    IsolatedPreparationToCommit,
    CompositeDeferredCommit,
    ExecutorOnly,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActualWaveObservation {
    pub call_id: NonZeroU64,
    pub physical_wave_ordinal: u32,
    /// None retains a real physical-wave occurrence whose actual path could
    /// not be proven. Such an observation cannot train a guessed shape.
    pub shape: Option<ActualWaveShape>,
    pub shape_unknown: Option<ActualWaveEvidenceUnknown>,
    pub boundary: WaveObservationBoundary,
    pub prepare_started_at_ns: u64,
    /// Submission attempt started; a terminal outcome proves whether it ran.
    pub submission_started_at_ns: Option<u64>,
    pub terminal_at_ns: Option<u64>,
    pub host_committed_at_ns: Option<u64>,
    /// Only an independently measured actual device interval, never a sum.
    pub device_elapsed_ns: Option<NonZeroU64>,
    pub outcome: Option<ActualWaveOutcome>,
}
