//! Real component execution of F32 visual replacement and DeepStack addition.
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
#[path = "support/row_scatter_f32_component.rs"]
mod scatter;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use fixture::{id, Kind};

#[test]
fn selected_f32_rows_preserve_other_requests_and_retire_real_managed_owners() {
    let c =
        ReferenceVNextComposition::create(id("device.reference.row_scatter_f32.fixture")).unwrap();
    scatter::exercise(
        c.runtime().clone(),
        c.registry(),
        c.catalog(),
        c.weight_materializers(),
    );
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
    assert!(c.runtime().snapshot().submissions > 0);
}
#[test]
fn f32_scatter_refuses_less_than_one_original_input_before_allocating() {
    let c = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.row_scatter_f32.capacity"),
        8 * 4 * 4 - 1,
    )
    .unwrap();
    for kind in [Kind::RowScatterReplaceF32, Kind::RowScatterAddF32] {
        assert!(scatter::config(kind)
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
