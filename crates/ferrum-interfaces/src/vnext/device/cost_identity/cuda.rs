//! Inputs obtained from the actual CUDA context, installed driver/runtime and
//! host boot. A marketing name or an ordinal cannot establish compatibility.

use super::*;
use std::num::NonZeroU32;

pub struct CudaHostBootDeviceEvidence<'a> {
    pub boot_session_uuid: [u8; 16],
    pub device_uuid: [u8; 16],
    pub device_name: &'a str,
    pub compute_capability_major: NonZeroU32,
    pub compute_capability_minor: u32,
    pub total_memory_bytes: NonZeroU64,
    pub driver_build: &'a str,
    pub driver_api_version: NonZeroU32,
    pub runtime_api_version: NonZeroU32,
    pub os_build: &'a str,
    pub host_model: &'a str,
    pub runtime_implementation_fingerprint: &'a str,
}

impl DeviceCostHardwareIdentity {
    pub fn cuda_host_boot(
        evidence: CudaHostBootDeviceEvidence<'_>,
    ) -> Result<Self, DeviceCostHardwareIdentityUnknown> {
        use DeviceCostHardwareIdentityField as Field;
        for (value, name) in [
            (evidence.boot_session_uuid, Field::BootSession),
            (evidence.device_uuid, Field::DeviceUuid),
        ] {
            if value == [0; 16] {
                return Err(DeviceCostHardwareIdentityUnknown::InvalidEvidence(name));
            }
        }
        validate_text(evidence.device_name, 256, Field::DeviceName)?;
        validate_text(evidence.driver_build, 128, Field::DriverBuild)?;
        validate_text(evidence.os_build, 256, Field::OsBuild)?;
        validate_text(evidence.host_model, 256, Field::HostModel)?;
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
            b"ferrum.device-cost.hardware.cuda.host-boot-device",
        );
        field(
            &mut hash,
            "schema",
            &DEVICE_COST_HARDWARE_IDENTITY_SCHEMA.to_le_bytes(),
        );
        field(&mut hash, "boot-session", &evidence.boot_session_uuid);
        field(&mut hash, "device-uuid", &evidence.device_uuid);
        field(&mut hash, "device-name", evidence.device_name.as_bytes());
        field(
            &mut hash,
            "compute-major",
            &evidence.compute_capability_major.get().to_le_bytes(),
        );
        field(
            &mut hash,
            "compute-minor",
            &evidence.compute_capability_minor.to_le_bytes(),
        );
        field(
            &mut hash,
            "total-memory",
            &evidence.total_memory_bytes.get().to_le_bytes(),
        );
        field(&mut hash, "driver-build", evidence.driver_build.as_bytes());
        field(
            &mut hash,
            "driver-api",
            &evidence.driver_api_version.get().to_le_bytes(),
        );
        field(
            &mut hash,
            "runtime-api",
            &evidence.runtime_api_version.get().to_le_bytes(),
        );
        field(&mut hash, "os-build", evidence.os_build.as_bytes());
        field(&mut hash, "host-model", evidence.host_model.as_bytes());
        field(&mut hash, "runtime-implementation", &runtime_digest);
        Ok(Self {
            schema_version: DEVICE_COST_HARDWARE_IDENTITY_SCHEMA,
            scope: DeviceCostHardwareIdentityScope::HostBootDevice,
            backend: DeviceCostHardwareBackend::Cuda,
            fingerprint: hash.finalize().into(),
        })
    }
}

#[cfg(test)]
mod tests;
