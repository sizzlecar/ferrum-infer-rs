use super::fixture::*;
use super::*;
use ferrum_models::qwen3_vl_text::{Qwen3VLTextKeyRun, Qwen3VLTextLayout};
use safetensors::tensor::Dtype;

#[test]
fn selected_source_headers_reject_equal_numel_wrong_shape_and_wrong_dtype() {
    let family = projection();
    let schema = family.stored_weight_schema().unwrap();
    for (shape, dtype, valid) in [
        (vec![8, 64], Dtype::BF16, true),
        (vec![64, 8], Dtype::BF16, false),
        (vec![8, 64], Dtype::F16, false),
        (vec![8, 63], Dtype::BF16, false),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let source = archive(directory.path(), shape, dtype);
        let result = assets::validate_archive(schema, &source);
        if valid {
            assert_eq!(result.unwrap(), 8 * 64 * 2);
        } else {
            assert!(result.is_err());
        }
    }
}

#[test]
fn real_component_ports_interoperate_before_any_device_allocation() {
    let composition = composition("device.image.stage.typed-ports");
    let text = Qwen3VLTextConfig::new(
        text(),
        Qwen3VLTextLayout::new(3, vec![vec![Qwen3VLTextKeyRun::new(3, true)]]).unwrap(),
    )
    .unwrap();
    let transformer = QwenImage21TransformerConfig::new(
        transformer(),
        QwenImage21T2iLayout::new(3, 2, 2, vec![3]).unwrap(),
    )
    .unwrap();
    let decoder = decoder();
    let text_family = TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&text).unwrap(),
            &NumericalProfileId::new(Qwen3VLTextEncoderFamily::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let dit_family = TypedFamilyRegistration::new(QwenImage21TransformerFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&transformer).unwrap(),
            &NumericalProfileId::new(QwenImage21TransformerFamily::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let vae_family = TypedFamilyRegistration::new(QwenImage21VaeDecoderFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&decoder).unwrap(),
            &NumericalProfileId::new(QwenImage21VaeDecoderFamily::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let text_plan = compile(&composition, text_family, &text.compile_options().unwrap()).unwrap();
    let dit_options = transformer.compile_options().unwrap();
    let dit_plan = compile(&composition, dit_family.clone(), &dit_options).unwrap();
    let vae_plan = compile(
        &composition,
        vae_family,
        &decoder.compile_options().unwrap(),
    )
    .unwrap();
    let ports = |component: &CompiledTensorComponent<_>| {
        component
            .executable()
            .execution_plan()
            .payload()
            .tensor_ports()
            .unwrap()
            .clone()
    };
    let text_ports = ports(&text_plan);
    let dit_ports = ports(&dit_plan);
    let vae_ports = ports(&vae_plan);
    let find = |id: &str| {
        dit_ports
            .inputs()
            .iter()
            .find(|port| port.value_id().as_str() == id)
            .unwrap()
            .tensor()
    };
    assert_eq!(
        text_ports.outputs()[0].tensor(),
        find(QwenImage21TransformerFamily::TEXT_ID)
    );
    let shape = QwenImage21LatentShape::new(1, 2, 2).unwrap();
    let noise = NormalNoiseConfig::new(64, 2, 2).unwrap();
    assert_eq!(
        noise.output_tensor().unwrap(),
        shape.tensor(QwenImage21LatentLayout::Nchw).unwrap()
    );
    assert_eq!(
        find(QwenImage21TransformerFamily::LATENTS_ID),
        &shape.tensor(QwenImage21LatentLayout::Packed).unwrap()
    );
    assert_eq!(
        dit_ports.outputs()[0].tensor(),
        find(QwenImage21TransformerFamily::LATENTS_ID)
    );
    // Shape is preserved by explicit domain conversion; normalized bytes must
    // still pass through the separately tested managed two-BF16 adapter.
    assert_eq!(
        vae_ports.inputs()[0].tensor(),
        &noise.output_tensor().unwrap()
    );
    assert_eq!(vae_ports.outputs()[0].tensor().dimensions(), [1, 4, 32, 32]);

    let mut wrong = dit_options;
    wrong
        .insert_tensor_spec(
            id(QwenImage21TransformerFamily::FREQUENCIES_ID).unwrap(),
            ProgramTensorSpec {
                dimensions: vec![1, 4, 2],
                element_type: ElementType::F32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        )
        .unwrap();
    assert!(compile(&composition, dit_family, &wrong).is_err());
    let snapshot = composition.runtime().snapshot();
    assert_eq!(snapshot.allocations, 0);
    assert_eq!(snapshot.submissions, 0);
}

#[test]
fn reference_missing_real_philox_fails_during_compilation_not_as_fake_noise() {
    let composition = composition("device.image.stage.no-philox");
    let config = NormalNoiseConfig::new(64, 2, 2).unwrap();
    // A declared profile is a demand, not a fabricated Reference provider.
    let profile = NumericalProfileId::new("test.required.real-philox.v1").unwrap();
    let family = TypedFamilyRegistration::new(NormalNoiseFamily::new(profile.as_str()).unwrap())
        .prepare_with_profile(&serde_json::to_value(&config).unwrap(), &profile)
        .unwrap();
    assert!(matches!(family.weights(), ModelWeights::None));
    assert!(compile(&composition, family, &config.compile_options().unwrap()).is_err());
    assert_eq!(composition.runtime().snapshot().allocations, 0);
    assert_eq!(composition.runtime().snapshot().submissions, 0);
}
