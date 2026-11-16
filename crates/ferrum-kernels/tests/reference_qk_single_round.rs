#[path = "support/qk_single_round_component.rs"]
mod fixture;
#[path = "support/qk_single_round_qualification.rs"]
mod qualification;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
#[test]
fn qk_f32_intermediates_partial_tail_multihead_and_owned_execution() {
    let c = ReferenceVNextComposition::create_with_capacity(
        fixture::id("device.reference.qk_single_round"),
        4 << 20,
    )
    .unwrap();
    for (head, rot, heads) in [(2, 2, 2), (4, 2, 2), (128, 96, 56), (127, 96, 3)] {
        qualification::qualify(
            c.runtime().clone(),
            c.registry(),
            fixture::Config {
                rows: 3,
                width: head * heads,
                head,
                rot,
            },
        );
    }
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}
