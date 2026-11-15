#[path = "support/packed_swiglu_component.rs"]
mod fixture;
#[path = "support/packed_swiglu_qualification.rs"]
mod qualification;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;

#[test]
fn packed_swiglu_runs_managed_reference_with_declared_rounding() {
    let composition = ReferenceVNextComposition::create_with_capacity(
        fixture::id("device.reference.packed_swiglu"),
        1 << 20,
    )
    .unwrap();
    for profile in [
        ferrum_runtime::PackedSwiGluBf16Profile::Exp2SigmoidF32SingleRound,
        ferrum_runtime::PackedSwiGluBf16Profile::SiluBf16ThenMultiply,
    ] {
        for (rows, width) in [(2, 17), (3, 129)] {
            qualification::qualify(
                composition.runtime().clone(),
                composition.registry(),
                fixture::Config {
                    rows,
                    width,
                    profile,
                },
            );
        }
    }
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
}
