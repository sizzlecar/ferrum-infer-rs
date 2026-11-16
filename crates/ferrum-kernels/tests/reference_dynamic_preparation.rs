#[path = "support/dynamic_preparation_qualification.rs"]
mod checks;
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
#[test]
fn dynamic_preparation_releases_failed_b2_and_preserves_b1_without_submission() {
    let c = ReferenceVNextComposition::create_with_capacity(
        fixture::id("device.reference.dynamic_prepare"),
        128 << 20,
    )
    .unwrap();
    checks::run(
        c.runtime().clone(),
        c.registry(),
        c.catalog(),
        c.weight_materializers(),
        128 << 20,
        Some(&|| c.runtime().snapshot().submissions),
        None,
    );
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}
