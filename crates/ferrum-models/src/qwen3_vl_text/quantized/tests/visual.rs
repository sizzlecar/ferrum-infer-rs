use super::*;
use crate::qwen3_vl_text::{Qwen3VLTextDeepStack, Qwen3VLTextVisualRun};
fn runs() -> Qwen3VLTextDeepStack {
    Qwen3VLTextDeepStack::new(vec![
        Qwen3VLTextVisualRun::new(1, 2).unwrap(),
        Qwen3VLTextVisualRun::new(4, 1).unwrap(),
    ])
    .unwrap()
}
fn visual() -> Qwen3VLTextComfyConfig {
    config(
        text(5, 2)
            .with_hidden_state_indices(vec![1, 2, 3, 4, 5])
            .unwrap(),
        false,
    )
    .with_visual_inputs(runs())
    .unwrap()
}
#[test]
fn visual_mode_preserves_original_stored_schema_and_legacy_wire_identity() {
    let c = config(
        text(5, 2)
            .with_hidden_state_indices(vec![1, 2, 3, 4, 5])
            .unwrap(),
        false,
    );
    let raw = serde_json::to_value(&c).unwrap();
    assert!(raw.get("input_mode").is_none());
    assert_eq!(
        serde_json::from_value::<Qwen3VLTextComfyConfig>(raw.clone()).unwrap(),
        c
    );
    let v = c.clone().with_visual_inputs(runs()).unwrap();
    let f = Family::new(limits()).unwrap();
    assert_eq!(f.weights(&c).unwrap(), f.weights(&v).unwrap());
    assert_eq!(v.input_mode(), Qwen3VLTextComfyInputMode::VisualEmbeddings);
    let encoded = serde_json::to_value(&v).unwrap();
    assert_eq!(encoded["input_mode"], "visual_embeddings");
    assert_eq!(
        serde_json::from_value::<Qwen3VLTextComfyConfig>(encoded).unwrap(),
        v
    );
    let mut invalid = raw;
    invalid["input_mode"] = json!("visual_embeddings");
    assert!(serde_json::from_value::<Qwen3VLTextComfyConfig>(invalid).is_err());
    assert!(Qwen3VLTextComfyConfig::new(
        c.text().clone().with_deepstack(runs()).unwrap(),
        c.projections().clone()
    )
    .is_err());
    assert!(crate::p_h3_image::PH3ImageTextConditioningConfig::new(v, 4).is_err());
}
#[test]
fn main_rows_replace_before_first_decoder_and_three_scatter_adds_precede_their_taps() {
    let c = visual();
    let f = Family::new(limits()).unwrap();
    let profile = id(Family::CUDA_VISION_PROFILE_ID);
    let prepared = TypedFamilyRegistration::new(f)
        .prepare_with_profile(&serde_json::to_value(&c).unwrap(), &profile)
        .unwrap();
    let p = prepared.program();
    let replacement = &p.blocks()[0].nodes[0];
    assert_eq!(
        replacement.operation_id.as_str(),
        ROW_SCATTER_REPLACE_F32_OPERATION_ID
    );
    assert_eq!(
        replacement.inputs,
        [id(Family::EMBEDDINGS_ID), id(Family::VISUAL_EMBEDDINGS_ID)]
    );
    let spec = &p.input_tensor_specs()[&id(Family::VISUAL_EMBEDDINGS_ID)];
    assert_eq!(spec.dimensions, [3, 256]);
    assert_eq!(spec.element_type, ElementType::F32);
    for (layer, id) in Family::DEEPSTACK_IDS.iter().enumerate() {
        let spec = &p.input_tensor_specs()[&ProgramValueId::new(*id).unwrap()];
        assert_eq!(spec.dimensions, [3, 256]);
        assert_eq!(spec.element_type, ElementType::F32);
        let block = p
            .blocks()
            .iter()
            .find(|b| b.id == format!("block.qwen3_vl.text.comfy.layer{layer}"))
            .unwrap();
        let last = block.nodes.last().unwrap();
        assert_eq!(last.operation_id.as_str(), ROW_SCATTER_ADD_F32_OPERATION_ID);
        assert_eq!(last.inputs[1].as_str(), *id);
        assert_eq!(
            last.outputs,
            [Family::hidden_state_output_id(layer as u64 + 1).unwrap()]
        );
        assert_eq!(
            last.attributes[&AttributeId::new("destination_starts").unwrap()],
            SemanticValue::Integers(vec![1, 4])
        );
    }
    let first = p
        .blocks()
        .iter()
        .find(|b| b.id.ends_with(".layer0"))
        .unwrap();
    assert_eq!(first.nodes[0].inputs[0], replacement.outputs[0]);
    let final_norm = &p.blocks().last().unwrap().nodes[0];
    assert_eq!(
        final_norm.operation_id.as_str(),
        HEAD_RMS_NORM_F32_OPERATION_ID
    );
    assert_eq!(
        final_norm.outputs,
        [Family::hidden_state_output_id(5).unwrap()]
    );
    assert!(p
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .all(|n| n.operation_id.as_str() != ROW_SCATTER_ADD_BF16_OPERATION_ID));
    assert!(prepared
        .numerical_profile()
        .operations
        .iter()
        .any(
            |o| o.operation_id.as_str() == ROW_SCATTER_REPLACE_F32_OPERATION_ID
                && o.multiplication_type.is_none()
                && o.accumulation_type.is_none()
        ));
}
#[test]
fn visual_runs_reject_padding_and_cross_request_ranges_before_preparation() {
    let c = config(text(5, 2), false);
    for (start, length) in [(0, 1), (2, 2)] {
        let selection =
            Qwen3VLTextDeepStack::new(vec![Qwen3VLTextVisualRun::new(start, length).unwrap()])
                .unwrap();
        assert!(c.clone().with_visual_inputs(selection).is_err());
    }
    assert!(config(text(2, 2), false)
        .with_visual_inputs(runs())
        .is_err());
}
