#[path = "support/addcmul_fma_bf16_qualification.rs"]
mod checks;
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
#[test]
fn managed_fma_bf16_preserves_fused_extremes_raw_gate_and_retirement() {
    let c = ReferenceVNextComposition::create(fixture::id("device.reference.addcmul_fma")).unwrap();
    checks::run(
        c.runtime().clone(),
        c.registry(),
        c.catalog(),
        c.weight_materializers(),
    );
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}

#[test]
fn fma_capacity_rejection_precedes_device_allocation_and_submission() {
    let c = ReferenceVNextComposition::create_with_capacity(
        fixture::id("device.reference.addcmul_fma.capacity"),
        128,
    )
    .unwrap();
    assert!(fixture::Config {
        rows: 3,
        inputs: 259,
        outputs: 259,
        kind: fixture::Kind::AddcmulFmaBf16
    }
    .try_compile(
        c.runtime().as_ref(),
        c.registry(),
        c.catalog(),
        c.weight_materializers()
    )
    .is_err());
    assert_eq!(c.runtime().snapshot().allocations, 0);
    assert_eq!(c.runtime().snapshot().submissions, 0);
}
