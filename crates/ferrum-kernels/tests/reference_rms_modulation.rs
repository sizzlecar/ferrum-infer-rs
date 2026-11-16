#[path = "support/rms_modulation_component.rs"]
mod fixture;
#[path = "support/rms_modulation_qualification.rs"]
mod qualification;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
#[test]
fn fused_rms_modulation_uses_unrounded_residual_and_retained_owners() {
    let c = ReferenceVNextComposition::create_with_capacity(
        fixture::id("device.reference.rms_modulation"),
        2 << 20,
    )
    .unwrap();
    for (rows, width) in [(3, 2), (3, 5376), (3, 8191), (67, 7)] {
        for residual in [false, true] {
            qualification::qualify(
                c.runtime().clone(),
                c.registry(),
                fixture::Config {
                    rows,
                    width,
                    residual,
                },
            );
        }
    }
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}
