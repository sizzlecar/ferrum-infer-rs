#[path = "support/layer_norm_f32_component.rs"]
mod fixture;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
#[test]
fn stored_f32_layer_norm_centers_general_widths_and_preserves_separate_affine() {
    let c =
        ReferenceVNextComposition::create(fixture::id("device.reference.layer_norm_f32")).unwrap();
    fixture::run(
        c.runtime().clone(),
        c.registry(),
        c.catalog(),
        c.weight_materializers(),
    );
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}
