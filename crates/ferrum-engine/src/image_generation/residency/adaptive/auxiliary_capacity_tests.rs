use super::*;

#[test]
fn auxiliary_physical_availability_counts_only_the_new_peak_and_native_margin() {
    let memory = DeviceMemoryAvailability {
        total_bytes: 4096,
        free_bytes: 512,
        reusable_pool_bytes: 256,
    };
    // Existing owner allocations already reduced physical free. Requiring
    // them a second time would reject the exact available boundary.
    assert!(auxiliary_physical_fits(640, 128, Some(memory)));
    assert!(!auxiliary_physical_fits(641, 128, Some(memory)));
    assert!(!auxiliary_physical_fits(1, 769, Some(memory)));
    assert!(auxiliary_physical_fits(0, 769, Some(memory)));
}

#[test]
fn auxiliary_missing_physical_observation_is_not_an_invented_zero() {
    assert!(auxiliary_physical_fits(1024, 0, None));
    let exhausted = DeviceMemoryAvailability {
        total_bytes: 4096,
        free_bytes: 0,
        reusable_pool_bytes: 0,
    };
    assert!(!auxiliary_physical_fits(1024, 0, Some(exhausted)));
}
