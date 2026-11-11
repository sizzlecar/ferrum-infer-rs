//! Actual compact-table and original expanded graph execution on Reference.
#[path = "support/row_selected_bf16_component.rs"]
mod fixture;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
#[test]
fn row_selected_bf16_matches_original_expanded_managed_execution() {
    let composition = ReferenceVNextComposition::create(
        ferrum_runtime::DeviceId::new("device.reference.row_selected_bf16").unwrap(),
    )
    .unwrap();
    fixture::run(
        composition.runtime().clone(),
        composition.registry(),
        composition.catalog(),
        composition.weight_materializers(),
    );
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    assert!(composition.runtime().snapshot().submissions > 0);
}
