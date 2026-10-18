use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

use super::{ContractVersion, VNextError};

pub const EXECUTION_IDENTITY_VERSION: ContractVersion = ContractVersion::new(3, 0);
pub const MAX_EXECUTION_EVENT_WIRE_BYTES: usize = 1024 * 1024;
pub const MAX_REPLAY_IDENTITY_WIRE_BYTES: usize = 1024 * 1024;
pub const MAX_RESOURCE_POOL_EVENT_WIRE_BYTES: usize = 16 * 1024 * 1024;

pub(super) fn invalid_event(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

pub(super) fn canonical_fingerprint(value: &impl Serialize) -> String {
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(value).expect("trusted event evidence must serialize"))
    )
}

pub(super) fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) fn validate_sha256(value: &str, label: &str) -> Result<(), VNextError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid_event(format!(
            "{label} must be a canonical lowercase SHA256"
        )));
    }
    Ok(())
}

macro_rules! nonzero_execution_id {
    ($name:ident, $label:literal) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(try_from = "u64", into = "u64")]
        pub struct $name(u64);

        impl $name {
            pub const fn get(self) -> u64 {
                self.0
            }
        }

        impl TryFrom<u64> for $name {
            type Error = VNextError;

            fn try_from(value: u64) -> Result<Self, Self::Error> {
                if value == 0 {
                    return Err(invalid_event(concat!($label, " must be non-zero")));
                }
                Ok(Self(value))
            }
        }

        impl From<$name> for u64 {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "{}", self.0)
            }
        }
    };
}

nonzero_execution_id!(ExecutionFrameId, "execution frame id");
nonzero_execution_id!(BatchStepId, "batch step id");
nonzero_execution_id!(BatchInvocationId, "batch invocation id");
nonzero_execution_id!(NodeInvocationId, "node invocation id");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionEventKind {
    RequestAccepted,
    PlanBuilt,
    FrameStarted,
    NodeStarted,
    OperationSubmitted,
    NodeRetired,
    FrameCompleted,
    FailureObserved,
    SequenceCompleted,
    SequenceAborted,
    RequestCompleted,
    RequestFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPhase {
    Resolution,
    Planning,
    Execution,
    Completion,
}

impl ExecutionPhase {
    pub(super) const fn rank(self) -> u8 {
        match self {
            Self::Resolution => 0,
            Self::Planning => 1,
            Self::Execution => 2,
            Self::Completion => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MonotonicTimestamp {
    pub nanos_since_run_start: u64,
}
