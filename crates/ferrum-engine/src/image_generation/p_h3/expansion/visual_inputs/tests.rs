use super::super::tests::fixture::{host, Fixture};
use super::*;
use ferrum_models::qwen3_vl_vision::Qwen3VLVisionGrid;

fn plan() -> VisualExpansionInputs {
    let fixture = Fixture::new();
    let expansion = fixture.expansion();
    let request = PH3ImagePromptExpansionRequest::edit("make the sky blue", 1)
        .unwrap()
        .with_max_new_tokens(2)
        .unwrap();
    let rendered = expansion.protocol.render(&request).unwrap();
    let tokens = expansion
        .protocol
        .encode_image_input(&rendered, &[Qwen3VLVisionGrid::new(1, 4, 6).unwrap()], 2)
        .unwrap();
    let raw = serde_json::from_slice(expansion.defined.sources().config_json()).unwrap();
    let mut text = Qwen35TextConfig::from_hf_config_value(&raw).unwrap();
    // A one-pair THW table for the original tiny hidden-width fixture. This
    // exercises physical backing and conversion, not full-model numerics.
    text.rope_parameters.partial_rotary_factor = 1.;
    text.rope_parameters.mrope_interleaved = true;
    text.rope_parameters.mrope_section = Some(vec![1, 0, 0]);
    VisualExpansionInputs::new(tokens, &text).unwrap()
}

fn output(
    host: &HostMemoryAccount,
    plan: &VisualExpansionInputs,
    value: f32,
    class: HostMemoryClass,
) -> ManagedHostBytes {
    let bytes =
        VisualExpansionInputs::extent(plan.tokens.image_runs()[0].rows(), plan.hidden, 4).unwrap();
    let mut result = host.allocate_zeroed(class, bytes).unwrap();
    for slot in result.as_mut().chunks_exact_mut(4) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    result.freeze()
}

#[test]
fn visual_bridge_exact_scope_retains_input_after_output_drop_and_short_scope_refunds() {
    let host = host();
    let plan = plan();
    let source = output(&host, &plan, 0.25, HostMemoryClass::Output);
    let peak = plan.retained_input_bytes(&host).unwrap();
    let too_small = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, peak - 1)])
        .unwrap();
    assert!(plan.bind_sources(&too_small.account(), &[&source]).is_err());
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::InputPreprocess),
        0
    );
    drop(too_small);
    let scope = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, peak)])
        .unwrap();
    let bound = plan.bind_sources(&scope.account(), &[&source]).unwrap();
    assert_eq!(bound.maximum_tokens(), plan.tokens.maximum_tokens());
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::InputPreprocess),
        peak
    );
    drop(source);
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::Output),
        0
    );
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::InputPreprocess),
        peak
    );
    drop(bound);
    drop(scope);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn visual_bridge_rejects_foreign_wrong_class_and_nonfinite_features_without_leaking_input() {
    let host = host();
    let foreign = super::super::tests::fixture::host();
    let plan = plan();
    let other = output(&foreign, &plan, 1., HostMemoryClass::Output);
    assert!(plan.bind_sources(&host, &[&other]).is_err());
    let wrong_class = output(&host, &plan, 1., HostMemoryClass::InputPreprocess);
    assert!(plan.bind_sources(&host, &[&wrong_class]).is_err());
    drop(wrong_class);
    for value in [f32::NAN, f32::INFINITY, f32::MAX] {
        let source = output(&host, &plan, value, HostMemoryClass::Output);
        let before = host.usage().unwrap().committed_bytes();
        assert!(plan.bind_sources(&host, &[&source]).is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), before);
        assert_eq!(
            host.usage()
                .unwrap()
                .class_allocated_bytes(HostMemoryClass::InputPreprocess),
            0
        );
    }
    assert!(plan.bind_sources(&host, &[]).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
