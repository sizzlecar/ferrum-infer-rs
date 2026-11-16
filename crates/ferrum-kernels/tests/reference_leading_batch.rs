#[path = "support/leading_batch_qualification.rs"]
mod checks;
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
#[test]
fn leading_batch_managed_copy_preserves_all_encodings_axis_order_and_lifetime() {
    let c =
        ReferenceVNextComposition::create(fixture::id("device.reference.leading_batch")).unwrap();
    checks::run(
        c.runtime().clone(),
        c.registry(),
        c.catalog(),
        c.weight_materializers(),
    );
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}
