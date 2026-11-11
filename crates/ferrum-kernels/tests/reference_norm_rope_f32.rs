#[path = "support/norm_rope_f32_component.rs"]
mod checks;
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use fixture::*;
#[test]
fn managed_f32_head_rms_retains_gamma_precision_and_non_power_of_two_width() {
    run(Kind::HeadRmsF32);
}
#[test]
fn managed_f32_half_split_rope_preserves_separate_product_rounding() {
    run(Kind::HalfSplitRopeF32);
}
#[test]
fn managed_unit_rms_partial_rope_normalizes_tail_without_gamma_storage() {
    run(Kind::UnitHeadRmsPartialRopeF32);
}
fn run(kind: Kind) {
    let c = ReferenceVNextComposition::create(id("device.reference.norm_rope_f32")).unwrap();
    checks::run(
        c.runtime().clone(),
        c.registry(),
        c.catalog(),
        c.weight_materializers(),
        kind,
    );
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}
#[test]
fn norm_capacity_rejection_precedes_allocation() {
    let c = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.norm_f32.capacity"),
        1024,
    )
    .unwrap();
    assert!(Config {
        rows: 2,
        inputs: 2560,
        outputs: 5120,
        kind: Kind::HeadRmsF32
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
