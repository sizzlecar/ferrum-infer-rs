//! Managed Reference execution of both F32 GELU variants and their actual owners.
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
#[path = "support/gelu_f32_component.rs"]
mod gelu;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use fixture::{id, Kind};

#[test]
fn f32_gelu_executes_both_variants_without_half_rounding_and_retires_owners() {
    let c = ReferenceVNextComposition::create(id("device.reference.gelu_f32.fixture")).unwrap();
    gelu::exercise(
        c.runtime().clone(),
        c.registry(),
        c.catalog(),
        c.weight_materializers(),
    );
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
    assert!(c.runtime().snapshot().submissions > 0);
}
#[test]
fn f32_gelu_refuses_less_than_one_original_input_before_allocating() {
    let c = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.gelu_f32.capacity"),
        4 * 8 * 4 - 1,
    )
    .unwrap();
    for kind in [Kind::GeluTanhF32, Kind::GeluErfF32] {
        assert!(gelu::config(kind)
            .try_compile(
                c.runtime().as_ref(),
                c.registry(),
                c.catalog(),
                c.weight_materializers()
            )
            .is_err());
    }
    assert_eq!(c.runtime().snapshot().allocations, 0);
    assert_eq!(c.runtime().snapshot().submissions, 0);
}
