#[path = "support/gqa_f32_nonpower_component.rs"]
mod fixture;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
#[test]
fn f32_general_heads_execute_original_compact_kv_and_match_independent_oracle() {
    let c =
        ReferenceVNextComposition::create(fixture::id("device.reference.gqa_f32.general")).unwrap();
    fixture::run(
        c.runtime().clone(),
        c.registry(),
        c.catalog(),
        c.weight_materializers(),
    );
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
    assert!(c.runtime().snapshot().submissions > 0);
}
