use super::*;

fn linux(
    boot: [u8; 16],
    device: u64,
    inode: u64,
    seconds: i64,
    nanos: u32,
) -> CostMonotonicDomainV1 {
    CostMonotonicDomainV1::new_linux_boottime(
        boot,
        device,
        NonZeroU64::new(inode).unwrap(),
        seconds,
        nanos,
    )
    .unwrap()
}

#[test]
fn cost_monotonic_domain_binds_boot_namespace_offset_and_clock_kind() {
    let first = linux([1; 16], 4, 5, -3, 1);
    assert_eq!(first, linux([1; 16], 4, 5, -3, 1));
    for other in [
        linux([2; 16], 4, 5, -3, 1),
        linux([1; 16], 6, 5, -3, 1),
        linux([1; 16], 4, 7, -3, 1),
        linux([1; 16], 4, 5, -2, 1),
        linux([1; 16], 4, 5, -3, 2),
        CostMonotonicDomainV1::new_macos_continuous([1; 16]).unwrap(),
    ] {
        assert_ne!(first, other);
        assert_ne!(first.sha256(), other.sha256());
    }
}

#[test]
fn cost_monotonic_domain_wire_is_checked_and_has_no_supplied_hash() {
    for value in [
        linux([1; 16], 4, 5, -3, 1),
        CostMonotonicDomainV1::new_macos_continuous([1; 16]).unwrap(),
    ] {
        let json = serde_json::to_value(&value).unwrap();
        assert!(json.get("digest").is_none());
        assert_eq!(
            serde_json::from_value::<CostMonotonicDomainV1>(json.clone()).unwrap(),
            value
        );
        let mut schema = json.clone();
        schema["schema_version"] = 2.into();
        assert!(serde_json::from_value::<CostMonotonicDomainV1>(schema).is_err());
        let mut extra = json.clone();
        extra["digest"] = serde_json::to_value([0_u8; 32]).unwrap();
        assert!(serde_json::from_value::<CostMonotonicDomainV1>(extra).is_err());
        let mut boot = json.clone();
        boot["clock"]["boot_session_uuid"] = serde_json::to_value([0_u8; 16]).unwrap();
        assert!(serde_json::from_value::<CostMonotonicDomainV1>(boot).is_err());
        let mut extra = json;
        extra["clock"]["invented_identity"] = true.into();
        assert!(serde_json::from_value::<CostMonotonicDomainV1>(extra).is_err());
    }
    let mut bad = serde_json::to_value(linux([1; 16], 0, 1, 0, 0)).unwrap();
    bad["clock"]["boottime_offset_nanoseconds"] = 1_000_000_000_u32.into();
    assert!(serde_json::from_value::<CostMonotonicDomainV1>(bad).is_err());
    assert!(CostMonotonicDomainV1::new_macos_continuous([0; 16]).is_err());
}

#[test]
fn cost_monotonic_domain_legacy_clock_defaults_to_no_restart_identity() {
    struct Legacy;
    impl crate::execution_cost::CostObservationClock for Legacy {
        fn now_ns(&self) -> Option<u64> {
            Some(7)
        }
    }
    let clock: &dyn crate::execution_cost::CostObservationClock = &Legacy;
    assert_eq!(clock.now_ns(), Some(7));
    assert!(clock.monotonic_domain().is_none());
}
