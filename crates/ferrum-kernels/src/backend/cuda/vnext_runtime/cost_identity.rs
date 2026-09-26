//! Cold bounded queries of the actual context, driver/runtime and host boot.
//! Compiled CUDA targets and marketing names are not substitutes for queries.

use std::num::{NonZeroU32, NonZeroU64};
use std::sync::Arc;

use cudarc::driver::{sys, CudaContext};
use ferrum_interfaces::vnext::{
    CudaHostBootDeviceEvidence, DeviceCostHardwareIdentity,
    DeviceCostHardwareIdentityAvailability as Availability,
    DeviceCostHardwareIdentityField as Field, DeviceCostHardwareIdentityUnknown as Unknown,
};

use super::nvml::Nvml;

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaRuntimeGetVersion(version: *mut i32) -> i32;
}

pub(super) fn capture(context: &CudaContext, runtime_fingerprint: &str) -> Availability {
    let result = (|| {
        let uuid = context
            .uuid()
            .map_err(|_| Unknown::QueryUnavailable(Field::DeviceUuid))?;
        let uuid = uuid.bytes.map(|value| value as u8);
        let name = context
            .name()
            .map_err(|_| Unknown::QueryUnavailable(Field::DeviceName))?;
        let (major, minor) = context
            .compute_capability()
            .map_err(|_| Unknown::QueryUnavailable(Field::ComputeCapability))?;
        let major = u32::try_from(major)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or(Unknown::InvalidEvidence(Field::ComputeCapability))?;
        let minor =
            u32::try_from(minor).map_err(|_| Unknown::InvalidEvidence(Field::ComputeCapability))?;
        let memory = context
            .total_mem()
            .map_err(|_| Unknown::QueryUnavailable(Field::DeviceMemory))?;
        let memory = u64::try_from(memory)
            .ok()
            .and_then(NonZeroU64::new)
            .ok_or(Unknown::InvalidEvidence(Field::DeviceMemory))?;
        let mut driver_api = 0;
        let mut runtime_api = 0;
        // SAFETY: each version function receives one valid writable integer;
        // these query APIs enqueue no GPU work and the context is already live.
        let status = unsafe { sys::cuDriverGetVersion(&mut driver_api) };
        if status != sys::CUresult::CUDA_SUCCESS {
            return Err(Unknown::QueryUnavailable(Field::DriverApiVersion));
        }
        let status = unsafe { cudaRuntimeGetVersion(&mut runtime_api) };
        if status != 0 {
            return Err(Unknown::QueryUnavailable(Field::RuntimeApiVersion));
        }
        let driver_api = positive_version(driver_api, Field::DriverApiVersion)?;
        let runtime_api = positive_version(runtime_api, Field::RuntimeApiVersion)?;
        let nvml = Nvml::load().map_err(|_| Unknown::QueryUnavailable(Field::DriverBuild))?;
        let driver_build = nvml
            .driver_build()
            .map_err(|_| Unknown::QueryUnavailable(Field::DriverBuild))?;
        let host = host_evidence()?;
        DeviceCostHardwareIdentity::cuda_host_boot(CudaHostBootDeviceEvidence {
            boot_session_uuid: host.boot,
            device_uuid: uuid,
            device_name: &name,
            compute_capability_major: major,
            compute_capability_minor: minor,
            total_memory_bytes: memory,
            driver_build: &driver_build,
            driver_api_version: driver_api,
            runtime_api_version: runtime_api,
            os_build: &host.os,
            host_model: &host.cpu,
            runtime_implementation_fingerprint: runtime_fingerprint,
        })
    })();
    match result {
        Ok(identity) => Availability::Known(Arc::new(identity)),
        Err(reason) => Availability::Unknown(reason),
    }
}

fn positive_version(value: i32, field: Field) -> Result<NonZeroU32, Unknown> {
    u32::try_from(value)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or(Unknown::InvalidEvidence(field))
}

struct HostEvidence {
    boot: [u8; 16],
    os: String,
    cpu: String,
}

#[cfg(target_os = "linux")]
fn host_evidence() -> Result<HostEvidence, Unknown> {
    use std::io::{BufReader, Read};
    fn read(path: &str, field: Field, max: usize) -> Result<String, Unknown> {
        let file = std::fs::File::open(path).map_err(|_| Unknown::QueryUnavailable(field))?;
        let mut bytes = Vec::new();
        file.take(max as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| Unknown::QueryUnavailable(field))?;
        if bytes.len() > max {
            return Err(Unknown::EvidenceTooLarge(field));
        }
        let text = std::str::from_utf8(&bytes).map_err(|_| Unknown::InvalidEvidence(field))?;
        let text = text.trim_end_matches('\n');
        if text.is_empty() || text.chars().any(char::is_control) {
            return Err(Unknown::InvalidEvidence(field));
        }
        Ok(text.to_owned())
    }
    let boot = parse_uuid(&read(
        "/proc/sys/kernel/random/boot_id",
        Field::BootSession,
        64,
    )?)?;
    let release = read("/proc/sys/kernel/osrelease", Field::OsBuild, 128)?;
    let version = read("/proc/sys/kernel/version", Field::OsBuild, 128)?;
    let file = std::fs::File::open("/proc/cpuinfo")
        .map_err(|_| Unknown::QueryUnavailable(Field::HostModel))?;
    // Read only the first processor's bounded prefix. Never allocate according
    // to an untrusted/large file size or silently use a truncated model string.
    let cpu = parse_cpu_prefix(BufReader::new(file.take(4097)))?;
    Ok(HostEvidence {
        boot,
        os: format!("{release} {version}"),
        cpu,
    })
}

#[cfg(any(target_os = "linux", test))]
fn parse_cpu_prefix(mut input: impl std::io::BufRead) -> Result<String, Unknown> {
    let mut consumed = 0_usize;
    loop {
        let mut line = Vec::new();
        let count = input
            .read_until(b'\n', &mut line)
            .map_err(|_| Unknown::QueryUnavailable(Field::HostModel))?;
        if count == 0 {
            return Err(Unknown::QueryUnavailable(Field::HostModel));
        }
        consumed = consumed
            .checked_add(count)
            .ok_or(Unknown::EvidenceTooLarge(Field::HostModel))?;
        if consumed > 4096 || line.len() > 256 {
            return Err(Unknown::EvidenceTooLarge(Field::HostModel));
        }
        if line.pop() != Some(b'\n') {
            return Err(Unknown::InvalidEvidence(Field::HostModel));
        }
        let text =
            std::str::from_utf8(&line).map_err(|_| Unknown::InvalidEvidence(Field::HostModel))?;
        if let Some((key, value)) = text.split_once(':') {
            if key.trim() == "model name" {
                let value = value.trim();
                if value.is_empty() || value.chars().any(char::is_control) {
                    return Err(Unknown::InvalidEvidence(Field::HostModel));
                }
                return Ok(value.to_owned());
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn host_evidence() -> Result<HostEvidence, Unknown> {
    Err(Unknown::Unsupported)
}

fn parse_uuid(text: &str) -> Result<[u8; 16], Unknown> {
    let invalid = Unknown::InvalidEvidence(Field::BootSession);
    if text.len() != 36 {
        return Err(invalid);
    }
    let mut nibbles = Vec::with_capacity(32);
    for (index, byte) in text.bytes().enumerate() {
        if matches!(index, 8 | 13 | 18 | 23) {
            if byte != b'-' {
                return Err(invalid);
            }
        } else {
            nibbles.push((byte as char).to_digit(16).ok_or(invalid)? as u8);
        }
    }
    let mut uuid = [0; 16];
    for (byte, pair) in uuid.iter_mut().zip(nibbles.chunks_exact(2)) {
        *byte = pair[0] << 4 | pair[1];
    }
    if uuid == [0; 16] {
        return Err(invalid);
    }
    Ok(uuid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_boot_identity_and_api_versions_require_actual_valid_inputs() {
        let upper = parse_uuid("01234567-89AB-CDEF-0123-456789ABCDEF").unwrap();
        assert_eq!(
            upper,
            parse_uuid("01234567-89ab-cdef-0123-456789abcdef").unwrap()
        );
        for value in [
            "",
            "00000000-0000-0000-0000-000000000000",
            "01234567_89ab-cdef-0123-456789abcdef",
            "01234567-89ab-cdef-0123-456789abcdez",
        ] {
            assert!(parse_uuid(value).is_err());
        }
        assert!(positive_version(0, Field::RuntimeApiVersion).is_err());
        assert!(positive_version(-1, Field::DriverApiVersion).is_err());
        assert_eq!(
            positive_version(12080, Field::RuntimeApiVersion)
                .unwrap()
                .get(),
            12080
        );
    }

    #[test]
    fn cuda_host_cpu_query_rejects_truncated_or_missing_bounded_evidence() {
        assert_eq!(
            parse_cpu_prefix(&b"processor: 0\nmodel name: Injected CPU\n"[..]).unwrap(),
            "Injected CPU"
        );
        assert!(parse_cpu_prefix(&b"processor: 0\nmodel name: truncated"[..]).is_err());
        assert!(parse_cpu_prefix(&b"processor: 0\n"[..]).is_err());
        assert!(parse_cpu_prefix(format!("model name: {}\n", "x".repeat(256)).as_bytes()).is_err());
    }
}
