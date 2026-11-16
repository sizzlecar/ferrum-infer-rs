#[path = "support/retained_host_input_qualification.rs"]
mod checks;
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;

#[test]
fn completed_output_slice_executes_with_exhausted_preprocess_capacity() {
    let composition =
        ReferenceVNextComposition::create(fixture::id("device.reference.retained_output")).unwrap();
    checks::run(
        composition.runtime().clone(),
        composition.registry(),
        composition.catalog(),
        composition.weight_materializers(),
    );
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
}
