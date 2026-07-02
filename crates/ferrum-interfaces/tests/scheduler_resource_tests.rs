use ferrum_interfaces::scheduler::{
    AllocatedResources, BatchResourceRequirements, ResourceConstraints, ResourceLimits,
};

#[test]
fn recurrent_state_resource_fields_default_to_empty() {
    let allocated = AllocatedResources::default();
    assert_eq!(allocated.recurrent_state_bytes, 0);
    assert_eq!(allocated.recurrent_state_slots, 0);

    let batch = BatchResourceRequirements::default();
    assert_eq!(batch.recurrent_state_bytes, 0);
    assert_eq!(batch.recurrent_state_slots, 0);

    let constraints = ResourceConstraints::default();
    assert_eq!(constraints.max_recurrent_state_bytes, None);
    assert_eq!(constraints.max_recurrent_state_slots, None);

    let limits = ResourceLimits::default();
    assert_eq!(limits.max_recurrent_state_bytes, None);
    assert_eq!(limits.max_recurrent_state_slots, None);
}
