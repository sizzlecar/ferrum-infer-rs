#[path = "support/row_index_gather_qualification.rs"]
mod checks;
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;

#[test]
fn indexed_gather_preserves_bits_batches_and_total_index_semantics() {
    let c =
        ReferenceVNextComposition::create(fixture::id("device.reference.index_gather")).unwrap();
    checks::run(
        c.runtime().clone(),
        c.registry(),
        c.catalog(),
        c.weight_materializers(),
    );
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}
