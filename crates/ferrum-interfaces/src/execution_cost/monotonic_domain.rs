//! Identity of a nanosecond cost clock across processes within one OS boot.
//! This is metadata, not evidence that a supplied timestamp is current. A
//! persisted declaration must equal the domain freshly read by the runtime.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::num::NonZeroU64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostMonotonicClockKindV1 {
    LinuxBoottimeV1,
    MacOsContinuousV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ClockWire {
    LinuxBoottimeV1 {
        boot_session_uuid: [u8; 16],
        namespace_device: u64,
        namespace_inode: NonZeroU64,
        boottime_offset_seconds: i64,
        boottime_offset_nanoseconds: u32,
    },
    MacOsContinuousV1 {
        boot_session_uuid: [u8; 16],
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DomainWire {
    schema_version: u32,
    clock: ClockWire,
}

/// A checked OS clock kind, boot identity and clock namespace. No process
/// creation time or wall clock enters this identity. Linux also binds the
/// namespace's immutable boot-time offset, so later inode reuse cannot equate
/// different clock origins after the previous process has exited.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "DomainWire", into = "DomainWire")]
pub struct CostMonotonicDomainV1 {
    wire: DomainWire,
    digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CostMonotonicDomainError {
    #[error("unsupported cost monotonic domain schema")]
    UnsupportedSchema,
    #[error("missing OS boot identity")]
    MissingBootIdentity,
    #[error("invalid boot-time namespace offset")]
    InvalidNamespaceOffset,
}

impl CostMonotonicDomainV1 {
    pub fn new_linux_boottime(
        boot_session_uuid: [u8; 16],
        namespace_device: u64,
        namespace_inode: NonZeroU64,
        boottime_offset_seconds: i64,
        boottime_offset_nanoseconds: u32,
    ) -> Result<Self, CostMonotonicDomainError> {
        DomainWire {
            schema_version: 1,
            clock: ClockWire::LinuxBoottimeV1 {
                boot_session_uuid,
                namespace_device,
                namespace_inode,
                boottime_offset_seconds,
                boottime_offset_nanoseconds,
            },
        }
        .try_into()
    }

    pub fn new_macos_continuous(
        boot_session_uuid: [u8; 16],
    ) -> Result<Self, CostMonotonicDomainError> {
        DomainWire {
            schema_version: 1,
            clock: ClockWire::MacOsContinuousV1 { boot_session_uuid },
        }
        .try_into()
    }

    pub fn validate(&self) -> Result<(), CostMonotonicDomainError> {
        self.wire.validate()
    }

    pub fn clock_kind(&self) -> CostMonotonicClockKindV1 {
        match self.wire.clock {
            ClockWire::LinuxBoottimeV1 { .. } => CostMonotonicClockKindV1::LinuxBoottimeV1,
            ClockWire::MacOsContinuousV1 { .. } => CostMonotonicClockKindV1::MacOsContinuousV1,
        }
    }

    /// Computed once from checked fields on cold construction/import.
    pub fn sha256(&self) -> &[u8; 32] {
        &self.digest
    }
}

impl DomainWire {
    fn validate(&self) -> Result<(), CostMonotonicDomainError> {
        if self.schema_version != 1 {
            return Err(CostMonotonicDomainError::UnsupportedSchema);
        }
        let boot = match self.clock {
            ClockWire::LinuxBoottimeV1 {
                boot_session_uuid,
                boottime_offset_nanoseconds,
                ..
            } => {
                if boottime_offset_nanoseconds >= 1_000_000_000 {
                    return Err(CostMonotonicDomainError::InvalidNamespaceOffset);
                }
                boot_session_uuid
            }
            ClockWire::MacOsContinuousV1 { boot_session_uuid } => boot_session_uuid,
        };
        if boot == [0; 16] {
            return Err(CostMonotonicDomainError::MissingBootIdentity);
        }
        Ok(())
    }
}

impl TryFrom<DomainWire> for CostMonotonicDomainV1 {
    type Error = CostMonotonicDomainError;

    fn try_from(wire: DomainWire) -> Result<Self, Self::Error> {
        wire.validate()?;
        let mut hash = Sha256::new();
        hash.update(b"ferrum.cost_monotonic_domain.v1\0");
        hash.update(wire.schema_version.to_le_bytes());
        match wire.clock {
            ClockWire::LinuxBoottimeV1 {
                boot_session_uuid,
                namespace_device,
                namespace_inode,
                boottime_offset_seconds,
                boottime_offset_nanoseconds,
            } => {
                hash.update([1]);
                hash.update(boot_session_uuid);
                hash.update(namespace_device.to_le_bytes());
                hash.update(namespace_inode.get().to_le_bytes());
                hash.update(boottime_offset_seconds.to_le_bytes());
                hash.update(boottime_offset_nanoseconds.to_le_bytes());
            }
            ClockWire::MacOsContinuousV1 { boot_session_uuid } => {
                hash.update([2]);
                hash.update(boot_session_uuid);
            }
        }
        Ok(Self {
            wire,
            digest: hash.finalize().into(),
        })
    }
}

impl From<CostMonotonicDomainV1> for DomainWire {
    fn from(value: CostMonotonicDomainV1) -> Self {
        value.wire
    }
}

#[cfg(test)]
mod tests;
