//! Cold, bounded host-boot identity queries. No GPU commands or subprocesses.

use ferrum_interfaces::vnext::{
    DeviceCostHardwareIdentity, DeviceCostHardwareIdentityAvailability,
    DeviceCostHardwareIdentityField as Field, DeviceCostHardwareIdentityUnknown as Unknown,
    MetalHostBootDeviceEvidence,
};
use metal::objc::runtime::{BOOL, YES};
use metal::objc::{msg_send, sel, sel_impl};
use std::{num::NonZeroU64, sync::Arc};

const MAX_SYSCTL_BYTES: usize = 256;

pub(super) fn capture(
    device: &metal::DeviceRef,
    runtime_fingerprint: &str,
) -> DeviceCostHardwareIdentityAvailability {
    // Older/unsupported implementations must not receive unavailable selectors.
    // SAFETY: respondsToSelector is an NSObject query; these selectors are fixed
    // protocol properties and no GPU work or mutation is requested.
    let available: [(Field, BOOL); 3] = unsafe {
        [
            (
                Field::DeviceRegistryId,
                msg_send![device, respondsToSelector: sel!(registryID)],
            ),
            (
                Field::DeviceName,
                msg_send![device, respondsToSelector: sel!(name)],
            ),
            (
                Field::UnifiedMemory,
                msg_send![device, respondsToSelector: sel!(hasUnifiedMemory)],
            ),
        ]
    };
    if let Some((field, _)) = available
        .into_iter()
        .find(|(_, supported)| *supported != YES)
    {
        return DeviceCostHardwareIdentityAvailability::Unknown(Unknown::QueryUnavailable(field));
    }
    // All values come from the exact retained device used for allocations.
    from_queries(
        device.registry_id(),
        device.name(),
        device.has_unified_memory(),
        runtime_fingerprint,
        read_sysctl,
    )
}

fn from_queries(
    registry_id: u64,
    device_name: &str,
    unified_memory: bool,
    runtime_fingerprint: &str,
    mut query: impl FnMut(Field) -> Result<Vec<u8>, Unknown>,
) -> DeviceCostHardwareIdentityAvailability {
    let result = (|| {
        let registry_id = NonZeroU64::new(registry_id)
            .ok_or(Unknown::InvalidEvidence(Field::DeviceRegistryId))?;
        let boot_raw = query(Field::BootSession)?;
        let os_raw = query(Field::OsBuild)?;
        let host_raw = query(Field::HostModel)?;
        let boot = parse_uuid(parse_sysctl_text(&boot_raw, Field::BootSession)?)?;
        let os_build = parse_sysctl_text(&os_raw, Field::OsBuild)?;
        let host_model = parse_sysctl_text(&host_raw, Field::HostModel)?;
        DeviceCostHardwareIdentity::metal_host_boot(MetalHostBootDeviceEvidence {
            boot_session_uuid: boot,
            registry_id,
            device_name,
            unified_memory,
            os_build,
            host_model,
            runtime_implementation_fingerprint: runtime_fingerprint,
        })
    })();
    match result {
        Ok(identity) => DeviceCostHardwareIdentityAvailability::Known(Arc::new(identity)),
        Err(reason) => DeviceCostHardwareIdentityAvailability::Unknown(reason),
    }
}

fn parse_sysctl_text(raw: &[u8], field: Field) -> Result<&str, Unknown> {
    if raw.len() > MAX_SYSCTL_BYTES {
        return Err(Unknown::EvidenceTooLarge(field));
    }
    // Require the exact returned C string, not a valid-looking truncated prefix.
    let bytes = raw
        .strip_suffix(&[0])
        .ok_or(Unknown::InvalidEvidence(field))?;
    if bytes.is_empty() || bytes.contains(&0) {
        return Err(Unknown::InvalidEvidence(field));
    }
    std::str::from_utf8(bytes).map_err(|_| Unknown::InvalidEvidence(field))
}

fn parse_uuid(text: &str) -> Result<[u8; 16], Unknown> {
    let invalid = Unknown::InvalidEvidence(Field::BootSession);
    if text.len() != 36 {
        return Err(invalid);
    }
    let mut nibbles = [0_u8; 32];
    let mut index = 0;
    for (position, byte) in text.bytes().enumerate() {
        if matches!(position, 8 | 13 | 18 | 23) {
            if byte != b'-' {
                return Err(invalid);
            }
        } else {
            nibbles[index] = (byte as char).to_digit(16).ok_or(invalid)? as u8;
            index += 1;
        }
    }
    let mut result = [0; 16];
    for (value, pair) in result.iter_mut().zip(nibbles.chunks_exact(2)) {
        *value = (pair[0] << 4) | pair[1];
    }
    if result == [0; 16] {
        return Err(invalid);
    }
    Ok(result)
}

#[cfg(target_os = "macos")]
fn read_sysctl(field: Field) -> Result<Vec<u8>, Unknown> {
    let name: &[u8] = match field {
        Field::BootSession => b"kern.bootsessionuuid\0",
        Field::OsBuild => b"kern.osversion\0",
        Field::HostModel => b"hw.model\0",
        _ => return Err(Unknown::QueryUnavailable(field)),
    };
    // A single fixed-capacity query cannot grow an allocation based on an OS
    // length reply. Unsupported keys, growth/races and oversized replies remain
    // Unknown; no fallback to marketing name plus memory is allowed.
    let mut bytes = [0_u8; MAX_SYSCTL_BYTES];
    let mut length = bytes.len();
    // SAFETY: names are fixed NUL-terminated constants, output points to the
    // complete writable array and its exact capacity; no new value is supplied.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr().cast(),
            bytes.as_mut_ptr().cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 {
        return Err(Unknown::QueryUnavailable(field));
    }
    if length > bytes.len() {
        return Err(Unknown::EvidenceTooLarge(field));
    }
    Ok(bytes[..length].to_vec())
}

#[cfg(not(target_os = "macos"))]
fn read_sysctl(_field: Field) -> Result<Vec<u8>, Unknown> {
    Err(Unknown::Unsupported)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUNTIME: &str = "abababababababababababababababababababababababababababababababab";

    fn query(field: Field) -> Result<Vec<u8>, Unknown> {
        Ok(match field {
            Field::BootSession => b"01234567-89AB-CDEF-0123-456789ABCDEF\0".to_vec(),
            Field::OsBuild => b"InjectedBuild1\0".to_vec(),
            Field::HostModel => b"InjectedHost1\0".to_vec(),
            _ => return Err(Unknown::QueryUnavailable(field)),
        })
    }

    fn known(value: DeviceCostHardwareIdentityAvailability) -> Arc<DeviceCostHardwareIdentity> {
        match value {
            DeviceCostHardwareIdentityAvailability::Known(value) => value,
            other => panic!("expected injected evidence to be known: {other:?}"),
        }
    }

    #[test]
    fn injected_host_boot_and_actual_device_make_a_stable_identity() {
        let first = known(from_queries(10, "Injected device", true, RUNTIME, query));
        let again = known(from_queries(10, "Injected device", true, RUNTIME, query));
        assert_eq!(first, again);
        assert_ne!(
            first,
            known(from_queries(11, "Injected device", true, RUNTIME, query))
        );
        let rebooted = known(from_queries(
            10,
            "Injected device",
            true,
            RUNTIME,
            |field| {
                if field == Field::BootSession {
                    Ok(b"11234567-89AB-CDEF-0123-456789ABCDEF\0".to_vec())
                } else {
                    query(field)
                }
            },
        ));
        assert_ne!(first, rebooted);
    }

    #[test]
    fn each_missing_required_query_stays_unknown() {
        for missing in [Field::BootSession, Field::OsBuild, Field::HostModel] {
            let identity = from_queries(10, "Injected device", true, RUNTIME, |field| {
                if field == missing {
                    Err(Unknown::QueryUnavailable(field))
                } else {
                    query(field)
                }
            });
            assert_eq!(
                identity,
                DeviceCostHardwareIdentityAvailability::Unknown(Unknown::QueryUnavailable(missing))
            );
        }
        assert_eq!(
            from_queries(0, "Injected device", true, RUNTIME, query),
            DeviceCostHardwareIdentityAvailability::Unknown(Unknown::InvalidEvidence(
                Field::DeviceRegistryId
            ))
        );
    }

    #[test]
    fn bounded_bytes_reject_truncation_embedded_nul_invalid_utf8_and_growth() {
        for raw in [
            b"build-without-terminator".to_vec(),
            b"a\0b\0".to_vec(),
            vec![255, 0],
            vec![],
            vec![0],
        ] {
            assert_eq!(
                parse_sysctl_text(&raw, Field::OsBuild),
                Err(Unknown::InvalidEvidence(Field::OsBuild))
            );
        }
        assert_eq!(
            parse_sysctl_text(&vec![0; MAX_SYSCTL_BYTES + 1], Field::OsBuild),
            Err(Unknown::EvidenceTooLarge(Field::OsBuild))
        );
    }

    #[test]
    fn uuid_is_canonicalized_and_invalid_boot_evidence_cannot_be_known() {
        assert_eq!(
            parse_uuid("01234567-89AB-CDEF-0123-456789ABCDEF"),
            parse_uuid("01234567-89ab-cdef-0123-456789abcdef")
        );
        for invalid in [
            "",
            "00000000-0000-0000-0000-000000000000",
            "0123456789AB-CDEF-0123-456789ABCDEF",
            "g1234567-89AB-CDEF-0123-456789ABCDEF",
        ] {
            assert!(parse_uuid(invalid).is_err());
        }
        let malformed = from_queries(10, "Injected device", true, RUNTIME, |field| {
            if field == Field::BootSession {
                Ok(b"not-a-boot-uuid\0".to_vec())
            } else {
                query(field)
            }
        });
        assert!(matches!(
            malformed,
            DeviceCostHardwareIdentityAvailability::Unknown(Unknown::InvalidEvidence(
                Field::BootSession
            ))
        ));
    }
}
