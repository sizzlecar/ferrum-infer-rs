#[path = "support/channel_bias_bf16_qualification.rs"]
mod checks;
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
#[test]
fn channel_bias_bf16_managed_reference_matches_independent_oracle_and_retires() {
    let c = ReferenceVNextComposition::create(fixture::id("device.reference.channel_bias_bf16"))
        .unwrap();
    checks::run(
        c.runtime().clone(),
        c.registry(),
        c.catalog(),
        c.weight_materializers(),
    );
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}
