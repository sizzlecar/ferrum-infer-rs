#[path = "support/head_fp8_roundtrip_component.rs"]
mod fixture;
#[path = "support/head_fp8_roundtrip_qualification.rs"]
mod qualification;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
#[test]
fn head_fp8_reference_runs_real_component_and_retires_owners() {
    let c = ReferenceVNextComposition::create_with_capacity(
        fixture::id("device.reference.head_fp8"),
        16 << 20,
    )
    .unwrap();
    for width in [7, 128, 129] {
        qualification::qualify(
            c.runtime().clone(),
            c.registry(),
            fixture::Config {
                batches: 2,
                heads: 2,
                rows: 3,
                width,
                selected_rows: None,
            },
        );
    }
    for width in [7, 128, 257] {
        qualification::qualify(
            c.runtime().clone(),
            c.registry(),
            fixture::Config {
                batches: 2,
                heads: 2,
                rows: 7,
                width,
                selected_rows: Some(11),
            },
        );
    }
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}
