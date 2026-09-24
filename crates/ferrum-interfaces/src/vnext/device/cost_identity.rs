//! Hardware evidence for cost-profile compatibility, not performance promises.
//! A host-boot scope deliberately prevents reuse across machines or reboots.

use sha2::{Digest, Sha256};
use std::{num::NonZeroU64, sync::Arc};

mod cuda;
pub use cuda::CudaHostBootDeviceEvidence;

pub const DEVICE_COST_HARDWARE_IDENTITY_SCHEMA: u32 = 1;

/// Actual backend capability, independent of requested execution/timing policy.
/// Unknown cannot be promoted to Unsupported from an eager submission: it may
/// have captured a graph while executing eager work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DeviceCostGraphCaptureCapability {
    #[default]
    Unknown,
    /// This runtime implementation cannot execute graph capture at all.
    Unsupported,
    /// The runtime can capture graphs. An eager execution alone does not prove
    /// that the same submission performed no capture or graph preparation.
    Supported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceCostHardwareIdentityScope {
    /// The same actual physical device during one host boot. This is not an
    /// assertion that another device with the same marketing name is equivalent.
    HostBootDevice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceCostHardwareBackend {
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceCostHardwareIdentityField {
    BootSession,
    OsBuild,
    HostModel,
    DeviceRegistryId,
    DeviceName,
    UnifiedMemory,
    RuntimeImplementation,
    DeviceUuid,
    ComputeCapability,
    DeviceMemory,
    DriverBuild,
    DriverApiVersion,
    RuntimeApiVersion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceCostHardwareIdentityUnknown {
    Unsupported,
    QueryUnavailable(DeviceCostHardwareIdentityField),
    InvalidEvidence(DeviceCostHardwareIdentityField),
    EvidenceTooLarge(DeviceCostHardwareIdentityField),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceCostHardwareIdentityAvailability {
    Known(Arc<DeviceCostHardwareIdentity>),
    Unknown(DeviceCostHardwareIdentityUnknown),
}

impl Default for DeviceCostHardwareIdentityAvailability {
    fn default() -> Self {
        Self::Unknown(DeviceCostHardwareIdentityUnknown::Unsupported)
    }
}

/// Bounded, transient inputs read from the actual Metal device and host OS.
/// The constructor retains only a digest, not raw boot/device identifiers.
pub struct MetalHostBootDeviceEvidence<'a> {
    pub boot_session_uuid: [u8; 16],
    pub registry_id: NonZeroU64,
    pub device_name: &'a str,
    pub unified_memory: bool,
    pub os_build: &'a str,
    pub host_model: &'a str,
    pub runtime_implementation_fingerprint: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCostHardwareIdentity {
    schema_version: u32,
    scope: DeviceCostHardwareIdentityScope,
    backend: DeviceCostHardwareBackend,
    fingerprint: [u8; 32],
}

impl DeviceCostHardwareIdentity {
    pub fn metal_host_boot(
        evidence: MetalHostBootDeviceEvidence<'_>,
    ) -> Result<Self, DeviceCostHardwareIdentityUnknown> {
        use DeviceCostHardwareIdentityField as Field;
        if evidence.boot_session_uuid == [0; 16] {
            return Err(DeviceCostHardwareIdentityUnknown::InvalidEvidence(
                Field::BootSession,
            ));
        }
        validate_text(evidence.device_name, 256, Field::DeviceName)?;
        validate_text(evidence.os_build, 64, Field::OsBuild)?;
        validate_text(evidence.host_model, 128, Field::HostModel)?;
        let runtime = evidence.runtime_implementation_fingerprint;
        if runtime.len() != 64 || !runtime.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(DeviceCostHardwareIdentityUnknown::InvalidEvidence(
                Field::RuntimeImplementation,
            ));
        }
        let mut runtime_digest = [0; 32];
        for (index, output) in runtime_digest.iter_mut().enumerate() {
            *output = u8::from_str_radix(&runtime[index * 2..index * 2 + 2], 16).map_err(|_| {
                DeviceCostHardwareIdentityUnknown::InvalidEvidence(Field::RuntimeImplementation)
            })?;
        }
        let mut hash = Sha256::new();
        field(
            &mut hash,
            "domain",
            b"ferrum.device-cost.hardware.metal.host-boot-device",
        );
        field(
            &mut hash,
            "schema",
            &DEVICE_COST_HARDWARE_IDENTITY_SCHEMA.to_le_bytes(),
        );
        field(&mut hash, "boot-session", &evidence.boot_session_uuid);
        field(
            &mut hash,
            "registry-id",
            &evidence.registry_id.get().to_le_bytes(),
        );
        field(&mut hash, "device-name", evidence.device_name.as_bytes());
        field(
            &mut hash,
            "unified-memory",
            &[u8::from(evidence.unified_memory)],
        );
        field(&mut hash, "os-build", evidence.os_build.as_bytes());
        field(&mut hash, "host-model", evidence.host_model.as_bytes());
        field(&mut hash, "runtime-implementation", &runtime_digest);
        Ok(Self {
            schema_version: DEVICE_COST_HARDWARE_IDENTITY_SCHEMA,
            scope: DeviceCostHardwareIdentityScope::HostBootDevice,
            backend: DeviceCostHardwareBackend::Metal,
            fingerprint: hash.finalize().into(),
        })
    }

    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }
    pub const fn scope(&self) -> DeviceCostHardwareIdentityScope {
        self.scope
    }
    pub const fn backend(&self) -> DeviceCostHardwareBackend {
        self.backend
    }
    pub const fn fingerprint(&self) -> &[u8; 32] {
        &self.fingerprint
    }
}

fn validate_text(
    value: &str,
    max_bytes: usize,
    field: DeviceCostHardwareIdentityField,
) -> Result<(), DeviceCostHardwareIdentityUnknown> {
    if value.len() > max_bytes {
        return Err(DeviceCostHardwareIdentityUnknown::EvidenceTooLarge(field));
    }
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(DeviceCostHardwareIdentityUnknown::InvalidEvidence(field));
    }
    Ok(())
}

fn field(hash: &mut Sha256, name: &str, value: &[u8]) {
    hash.update((name.len() as u64).to_le_bytes());
    hash.update(name.as_bytes());
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence() -> MetalHostBootDeviceEvidence<'static> {
        MetalHostBootDeviceEvidence {
            boot_session_uuid: [1; 16],
            registry_id: NonZeroU64::new(17).unwrap(),
            device_name: "Injected device",
            unified_memory: true,
            os_build: "TestBuild1",
            host_model: "InjectedHost1",
            runtime_implementation_fingerprint:
                "abababababababababababababababababababababababababababababababab",
        }
    }

    #[test]
    fn scope_is_explicit_and_same_evidence_is_stable() {
        let identity = DeviceCostHardwareIdentity::metal_host_boot(evidence()).unwrap();
        assert_eq!(
            identity.scope(),
            DeviceCostHardwareIdentityScope::HostBootDevice
        );
        assert_eq!(identity.backend(), DeviceCostHardwareBackend::Metal);
        assert_eq!(
            identity,
            DeviceCostHardwareIdentity::metal_host_boot(evidence()).unwrap()
        );
    }

    #[test]
    fn boot_or_physical_device_change_invalidates_compatibility() {
        let baseline = DeviceCostHardwareIdentity::metal_host_boot(evidence()).unwrap();
        let mut rebooted = evidence();
        rebooted.boot_session_uuid = [2; 16];
        assert_ne!(
            baseline,
            DeviceCostHardwareIdentity::metal_host_boot(rebooted).unwrap()
        );
        let mut device = evidence();
        device.registry_id = NonZeroU64::new(18).unwrap();
        assert_ne!(
            baseline,
            DeviceCostHardwareIdentity::metal_host_boot(device).unwrap()
        );
    }

    #[test]
    fn environment_and_runtime_are_identity_inputs() {
        let baseline = DeviceCostHardwareIdentity::metal_host_boot(evidence()).unwrap();
        let changed_runtime = "cd".repeat(32);
        for variant in 0..5 {
            let mut changed = evidence();
            match variant {
                0 => changed.os_build = "TestBuild2",
                1 => changed.host_model = "InjectedHost2",
                2 => changed.device_name = "Other injected device",
                3 => changed.unified_memory = false,
                _ => {
                    changed.runtime_implementation_fingerprint = &changed_runtime;
                }
            }
            assert_ne!(
                baseline,
                DeviceCostHardwareIdentity::metal_host_boot(changed).unwrap()
            );
        }
    }

    #[test]
    fn missing_and_unbounded_inputs_cannot_create_known_identity() {
        let mut missing = evidence();
        missing.boot_session_uuid = [0; 16];
        assert!(DeviceCostHardwareIdentity::metal_host_boot(missing).is_err());
        let mut missing = evidence();
        missing.os_build = "";
        assert!(DeviceCostHardwareIdentity::metal_host_boot(missing).is_err());
        let large = "a".repeat(257);
        let mut missing = evidence();
        missing.device_name = &large;
        assert_eq!(
            DeviceCostHardwareIdentity::metal_host_boot(missing),
            Err(DeviceCostHardwareIdentityUnknown::EvidenceTooLarge(
                DeviceCostHardwareIdentityField::DeviceName
            ))
        );
        let mut missing = evidence();
        missing.runtime_implementation_fingerprint = "not-a-fingerprint";
        assert!(DeviceCostHardwareIdentity::metal_host_boot(missing).is_err());
    }
}
