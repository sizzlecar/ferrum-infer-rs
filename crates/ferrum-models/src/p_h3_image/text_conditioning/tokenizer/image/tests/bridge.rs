use super::*;
use crate::{
    p_h3_image::text_conditioning::tests as text_fixture,
    qwen3_vl_vision::comfy::{tests::reference_outputs, Qwen3VLVisionComfyFamily},
};
use ferrum_execution::TensorExecutionOutput;

fn id<T: TryFrom<String>>(value: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}
fn host(alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            32 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 32 << 20,
                output_bytes: 32 << 20,
                transfer_staging_bytes: 32 << 20,
                weight_backing_bytes: 32 << 20,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap()
}
fn config(prompt: &PH3ImageEditPrompt) -> PH3ImageTextConditioningConfig {
    text_fixture::config(prompt.text().layout().clone())
        .with_visual_inputs(prompt.visual_rows().clone())
        .unwrap()
}
fn new_input_bytes(config: &PH3ImageTextConditioningConfig, h: &HostMemoryAccount) -> usize {
    config
        .input_tensor_specs()
        .unwrap()
        .values()
        .map(|s| h.backing_bytes(s.byte_len().unwrap() as usize).unwrap())
        .sum()
}
fn check_copied_ports(inputs: &[ManagedHostTensor], visual: &[TensorExecutionOutput]) {
    let source = std::iter::once(Qwen3VLVisionComfyFamily::OUTPUT_ID)
        .chain(Qwen3VLVisionComfyFamily::DEEPSTACK_IDS);
    let target = std::iter::once(Qwen3VLTextComfyFamily::VISUAL_EMBEDDINGS_ID)
        .chain(Qwen3VLTextComfyFamily::DEEPSTACK_IDS);
    for (source, target) in source.zip(target) {
        let output = visual
            .iter()
            .find(|o| o.value_id().as_str() == source)
            .unwrap();
        let input = inputs
            .iter()
            .find(|i| i.value_id().as_str() == target)
            .unwrap();
        assert_eq!(input.tensor(), output.tensor());
        assert_eq!(input.bytes().bytes(), output.bytes().bytes());
        assert_eq!(input.bytes().class(), HostMemoryClass::InputPreprocess);
        assert_eq!(output.bytes().class(), HostMemoryClass::Output);
        assert_ne!(
            input.bytes().bytes().as_ptr(),
            output.bytes().bytes().as_ptr()
        );
    }
}

#[test]
fn multiple_picture_readbacks_are_concatenated_in_order_with_shape_and_owner_validation() {
    let h = host(16);
    let first = reference_outputs(4, 6, 0, &h);
    let second = reference_outputs(6, 4, 1, &h);
    let prompt = tokenizer(512, 512)
        .prepare_images("cat after", &[grid(4, 6), grid(6, 4)], 2)
        .unwrap();
    let c = config(&prompt);
    let inputs = prompt
        .managed_image_inputs(&c, &h, &[&first, &second])
        .unwrap();
    for (source, target) in std::iter::once(Qwen3VLVisionComfyFamily::OUTPUT_ID)
        .chain(Qwen3VLVisionComfyFamily::DEEPSTACK_IDS)
        .zip(
            std::iter::once(Qwen3VLTextComfyFamily::VISUAL_EMBEDDINGS_ID)
                .chain(Qwen3VLTextComfyFamily::DEEPSTACK_IDS),
        )
    {
        let expected = [&first, &second]
            .into_iter()
            .flat_map(|outputs| {
                outputs
                    .iter()
                    .find(|output| output.value_id().as_str() == source)
                    .unwrap()
                    .bytes()
                    .bytes()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        let input = inputs
            .iter()
            .find(|input| input.value_id().as_str() == target)
            .unwrap();
        assert_eq!(input.bytes().bytes(), expected);
        assert_eq!(input.tensor().dimensions()[0], 12);
    }
    assert!(prompt.managed_image_inputs(&c, &h, &[&first]).is_err());
    let wrong_shape = reference_outputs(2, 4, 0, &h);
    assert!(prompt
        .managed_image_inputs(&c, &h, &[&first, &wrong_shape])
        .is_err());
    let foreign = reference_outputs(6, 4, 1, &host(16));
    assert!(prompt
        .managed_image_inputs(&c, &h, &[&first, &foreign])
        .is_err());
}

#[test]
fn actual_three_layer_visual_readbacks_copy_to_seven_exact_inputs_with_atomic_host_admission() {
    let h = host(16);
    let outputs = reference_outputs(4, 6, 0, &h);
    let prompt = tokenizer(128, 128)
        .prepare_single_image("cat after", &grid(4, 6), 2)
        .unwrap();
    let c = config(&prompt);
    let retained = h.usage().unwrap();
    assert_eq!(
        retained.allocated_bytes(),
        outputs
            .iter()
            .map(|o| o.bytes().backing_bytes())
            .sum::<usize>()
    );
    assert!(outputs
        .iter()
        .flat_map(|o| floats(o.bytes().bytes()))
        .any(|v| v.to_bits() & 0xffff != 0));
    let required = new_input_bytes(&c, &h);
    // The complete real tensors are unchanged. A grant one physical byte
    // below their separately padded sum must reject all seven atomically.
    let denied = h
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, required - 1)])
        .unwrap();
    let before = h.usage().unwrap();
    assert!(prompt
        .managed_inputs(&c, &denied.account(), &outputs)
        .is_err());
    assert_eq!(h.usage().unwrap(), before);
    denied.close().unwrap();
    let scope = h
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, required)])
        .unwrap();
    let view = scope.account();
    let inputs = prompt.managed_inputs(&c, &view, &outputs).unwrap();
    check_copied_ports(&inputs, &outputs);
    let ids = inputs
        .iter()
        .find(|i| i.value_id().as_str() == PH3ImageTextConditioningFamily::TOKEN_IDS_ID)
        .unwrap();
    assert_eq!(
        ids.bytes()
            .bytes()
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>(),
        prompt.text().token_ids()
    );
    assert_eq!(
        h.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::InputPreprocess),
        required
    );
    assert_eq!(
        h.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::Output),
        retained.allocated_bytes()
    );
    // Scope close returns only unused credit; both the new inputs and the
    // independent original visual readbacks remain valid under their owners.
    scope.close().unwrap();
    check_copied_ports(&inputs, &outputs);
    drop(inputs);
    assert_eq!(h.usage().unwrap(), retained);
    drop(outputs);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn foreign_incomplete_duplicate_and_wrong_shape_real_visual_outputs_reject_before_copy_and_retry() {
    let h = host(8);
    let outputs = reference_outputs(4, 6, 0, &h);
    let prompt = tokenizer(128, 128)
        .prepare_single_image("cat", &grid(4, 6), 2)
        .unwrap();
    let c = config(&prompt);
    let foreign = host(8);
    let before = h.usage().unwrap();
    assert!(prompt.managed_inputs(&c, &foreign, &outputs).is_err());
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
    assert!(prompt.managed_inputs(&c, &h, &outputs[..3]).is_err());
    let mut duplicate = outputs.clone();
    duplicate[1] = duplicate[0].clone();
    assert!(prompt.managed_inputs(&c, &h, &duplicate).is_err());
    // Select one real leading row from a completed visual output. Its physical
    // owner and output ID remain genuine, but its [1,256] port is incompatible
    // with this image's six expanded feature rows.
    let mut wrong_shape = outputs.clone();
    wrong_shape[0] = wrong_shape[0].batch_item(0).unwrap();
    assert!(prompt.managed_inputs(&c, &h, &wrong_shape).is_err());
    assert_eq!(h.usage().unwrap(), before);
    let retry = prompt.managed_inputs(&c, &h, &outputs).unwrap();
    check_copied_ports(&retry, &outputs);
    drop((retry, duplicate, wrong_shape, outputs));
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn h3_visual_wrapper_keeps_actual_stored_schema_and_budgets_all_seven_input_backings() {
    let h = host(64);
    let prompt = tokenizer(128, 128)
        .prepare_single_image("cat after", &grid(4, 6), 2)
        .unwrap();
    let plain = text_fixture::config(prompt.text().layout().clone());
    let visual = plain
        .clone()
        .with_visual_inputs(prompt.visual_rows().clone())
        .unwrap();
    let family = PH3ImageTextConditioningFamily::new(text_fixture::limits()).unwrap();
    assert_eq!(
        family.weights(&plain).unwrap(),
        family.weights(&visual).unwrap()
    );
    let prepared = TypedFamilyRegistration::new(family)
        .prepare_with_profile(
            &serde_json::to_value(&visual).unwrap(),
            &NumericalProfileId::new(PH3ImageTextConditioningFamily::CUDA_VISION_PROFILE_ID)
                .unwrap(),
        )
        .unwrap();
    let specs = visual.input_tensor_specs().unwrap();
    assert_eq!(
        prepared
            .program()
            .inputs()
            .iter()
            .collect::<std::collections::BTreeSet<_>>(),
        specs.keys().collect()
    );
    let operations = prepared
        .program()
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .map(|n| n.operation_id.as_str())
        .collect::<Vec<_>>();
    assert!(operations.contains(&ROW_SCATTER_REPLACE_F32_OPERATION_ID));
    assert_eq!(
        operations
            .iter()
            .filter(|&&op| op == ROW_SCATTER_ADD_F32_OPERATION_ID)
            .count(),
        3
    );
    let (_directory, source) = text_fixture::archive(&text_fixture::original(&plain));
    let adapter = PH3ImageTextWeightAdapter::new(
        &visual,
        std::sync::Arc::new(source),
        text_fixture::limits(),
    )
    .unwrap();
    let source = adapter.weight_source(&h);
    let mut promoted_norm = false;
    for component in &prepared.stored_weight_schema().unwrap().components {
        let payload = source.component(component).unwrap();
        assert_eq!(
            payload.bytes().len() as u64,
            component.physical_bytes().unwrap()
        );
        promoted_norm |= payload.retained_host_memory().is_some();
    }
    assert!(promoted_norm);
    let report = visual
        .host_requirements(&h, PH3ImageConditioningPrecision::F32)
        .unwrap();
    let encoder = report
        .phases
        .iter()
        .find(|p| p.phase == PH3ImageTextConditioningHostPhase::EncoderReadback)
        .unwrap();
    assert_eq!(encoder.input_bytes, new_input_bytes(&visual, &h));
    let old = plain
        .host_requirements(&h, PH3ImageConditioningPrecision::F32)
        .unwrap();
    let old_encoder = old
        .phases
        .iter()
        .find(|p| p.phase == PH3ImageTextConditioningHostPhase::EncoderReadback)
        .unwrap();
    let feature = h.backing_bytes(6 * 256 * 4).unwrap();
    assert_eq!(encoder.input_bytes, old_encoder.input_bytes + 4 * feature);
    assert!(h.usage().unwrap().committed_bytes() == 0);
    // The text-only adapter may not silently omit visual inputs for this mode.
    assert!(prompt.text().managed_inputs(&visual, &h).is_err());
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}
