use super::*;

fn evidence() -> CudaHostBootDeviceEvidence<'static> {
    CudaHostBootDeviceEvidence {
        boot_session_uuid: [1; 16],
        device_uuid: [2; 16],
        device_name: "Injected CUDA device",
        compute_capability_major: NonZeroU32::new(8).unwrap(),
        compute_capability_minor: 9,
        total_memory_bytes: NonZeroU64::new(24 << 30).unwrap(),
        driver_build: "injected-driver-build",
        driver_api_version: NonZeroU32::new(12000).unwrap(),
        runtime_api_version: NonZeroU32::new(12000).unwrap(),
        os_build: "injected OS build",
        host_model: "injected host CPU",
        runtime_implementation_fingerprint:
            "abababababababababababababababababababababababababababababababab",
    }
}

#[test]
fn cuda_identity_is_stable_and_each_actual_input_changes_compatibility() {
    let baseline = DeviceCostHardwareIdentity::cuda_host_boot(evidence()).unwrap();
    assert_eq!(baseline.backend(), DeviceCostHardwareBackend::Cuda);
    assert_eq!(
        baseline.scope(),
        DeviceCostHardwareIdentityScope::HostBootDevice
    );
    assert_eq!(
        baseline,
        DeviceCostHardwareIdentity::cuda_host_boot(evidence()).unwrap()
    );
    let different_runtime = "cd".repeat(32);
    for change in 0..12 {
        let mut value = evidence();
        match change {
            0 => value.boot_session_uuid = [3; 16],
            1 => value.device_uuid = [4; 16],
            2 => value.device_name = "another CUDA device",
            3 => value.compute_capability_major = NonZeroU32::new(9).unwrap(),
            4 => value.compute_capability_minor = 0,
            5 => value.total_memory_bytes = NonZeroU64::new(16 << 30).unwrap(),
            6 => value.driver_build = "another driver build",
            7 => value.driver_api_version = NonZeroU32::new(12010).unwrap(),
            8 => value.runtime_api_version = NonZeroU32::new(12010).unwrap(),
            9 => value.os_build = "another OS build",
            10 => value.host_model = "another host CPU",
            _ => value.runtime_implementation_fingerprint = &different_runtime,
        }
        assert_ne!(
            baseline,
            DeviceCostHardwareIdentity::cuda_host_boot(value).unwrap()
        );
    }
}

#[test]
fn cuda_identity_rejects_missing_unbounded_or_malformed_required_evidence() {
    for missing in 0..7 {
        let mut value = evidence();
        match missing {
            0 => value.boot_session_uuid = [0; 16],
            1 => value.device_uuid = [0; 16],
            2 => value.device_name = "",
            3 => value.driver_build = "untrusted\ntruncated",
            4 => value.os_build = "",
            5 => value.host_model = "",
            _ => value.runtime_implementation_fingerprint = "not-a-runtime-hash",
        }
        assert!(DeviceCostHardwareIdentity::cuda_host_boot(value).is_err());
    }
    let oversized = "a".repeat(257);
    let mut value = evidence();
    value.device_name = &oversized;
    assert_eq!(
        DeviceCostHardwareIdentity::cuda_host_boot(value),
        Err(DeviceCostHardwareIdentityUnknown::EvidenceTooLarge(
            DeviceCostHardwareIdentityField::DeviceName
        ))
    );
}
