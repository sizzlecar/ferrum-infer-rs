use super::*;

// Null input rejects after geometry validation but before any device enqueue.
// Thus -5 proves a valid plan reached the span check; -1 rejects a forged plan.
fn pack_status(plan: &UpstreamLinearPlanV1) -> i32 {
    unsafe {
        ffi::ferrum_upstream_mmq_pack_v2(
            plan,
            ptr::null(),
            plan.request.inputs,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    }
}

#[test]
#[ignore = "requires a source-builder-produced CUDA upstream linear artifact and GPU"]
fn cached_mmq_plan_rejects_mutation_and_preserves_span_checks() {
    let valid = plan(
        UpstreamLinearAlgorithm::Mmq,
        UpstreamLinearFormat::Q4K,
        UpstreamLinearLayout::Columns,
        64,
        256,
        128,
    );
    let mut same_address = valid;
    assert_eq!(pack_status(&same_address), -5);
    // Keep the request fixed and mutate every derived-plan byte, including
    // reserved fields. The unsigned-integer ABI has no padding.
    for byte in size_of::<UpstreamLinearRequestV1>()..size_of::<UpstreamLinearPlanV1>() {
        same_address = valid;
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(
                (&mut same_address as *mut UpstreamLinearPlanV1).cast::<u8>(),
                size_of::<UpstreamLinearPlanV1>(),
            );
            bytes[byte] ^= 0x80;
        }
        assert_eq!(
            pack_status(&same_address),
            -1,
            "accepted altered byte {byte}"
        );
    }
    same_address = valid;
    same_address.request.reserved = 1;
    assert_eq!(pack_status(&same_address), -1);
    same_address = valid;
    same_address.request.rows += 1;
    assert_eq!(pack_status(&same_address), -1);
    same_address = valid;
    assert_eq!(pack_status(&same_address), -5);
    // Equal values at a different address retain all per-launch span checks.
    assert_eq!(pack_status(&valid), -5);
}
