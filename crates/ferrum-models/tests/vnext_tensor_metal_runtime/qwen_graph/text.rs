//! The real text family, including three DeepStack injections and a later layer.
use super::*;
use ferrum_models::qwen3_vl_text::*;

fn config(head: u64, editing: bool) -> Qwen3VLTextConfig {
    let architecture = Qwen3VLTextArchitecture::from_hf_value(&serde_json::json!({
        "architectures":["Qwen3VLForConditionalGeneration"], "model_type":"qwen3_vl",
        "text_config":{
            "model_type":"qwen3_vl_text", "dtype":"bfloat16", "attention_bias":false,
            "attention_dropout":0.0, "hidden_act":"silu", "num_hidden_layers":4,
            "hidden_size":128, "intermediate_size":192, "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":head, "vocab_size":256,
            "max_position_embeddings":128, "rms_norm_eps":0.000001, "rope_theta":5000000,
            "rope_scaling":{"rope_type":"default", "mrope_interleaved":true,
                "mrope_section":[head/4,head/8,head/8]}
        }
    }))
    .unwrap();
    // Different valid ranges, empty causal prefixes, an interior padding hole
    // and a fully padded sample. Rows beyond padding remain ordinary queries.
    let layout = Qwen3VLTextLayout::new(
        33,
        vec![
            vec![
                Qwen3VLTextKeyRun::new(2, false),
                Qwen3VLTextKeyRun::new(31, true),
            ],
            vec![
                Qwen3VLTextKeyRun::new(10, true),
                Qwen3VLTextKeyRun::new(3, false),
                Qwen3VLTextKeyRun::new(20, true),
            ],
            vec![Qwen3VLTextKeyRun::new(33, false)],
        ],
    )
    .unwrap();
    let mut config = Qwen3VLTextConfig::new(architecture, layout).unwrap();
    if editing {
        config = config
            .with_deepstack(
                Qwen3VLTextDeepStack::new(vec![
                    Qwen3VLTextVisualRun::new(3, 2).unwrap(),
                    Qwen3VLTextVisualRun::new(30, 3).unwrap(),
                    Qwen3VLTextVisualRun::new(34, 2).unwrap(),
                    Qwen3VLTextVisualRun::new(46, 4).unwrap(),
                ])
                .unwrap(),
            )
            .unwrap();
    }
    config
}

pub(super) fn maximum_command_count() -> usize {
    let prepared = TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(config(128, true)).unwrap(),
            &id(Qwen3VLTextEncoderFamily::METAL_PROFILE_ID),
        )
        .unwrap();
    prepared
        .program()
        .blocks()
        .iter()
        .map(|b| b.nodes.len())
        .sum::<usize>()
        + prepared.program().inputs().len()
}

pub(super) fn verify(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    host: &HostMemoryAccount,
) {
    for head in [32, 128] {
        for editing in [false, true] {
            let config = config(head, editing);
            let specs = config.input_tensor_specs().unwrap();
            let inputs = fixture::inputs(&specs);
            let options = config.compile_options().unwrap();
            let inspected = execute(
                runtime,
                registry,
                catalog,
                host,
                Qwen3VLTextEncoderFamily::new().unwrap(),
                &config,
                Qwen3VLTextEncoderFamily::METAL_PROFILE_ID,
                &options,
                &inputs,
                true,
            );
            let ordinary = execute(
                runtime,
                registry,
                catalog,
                host,
                Qwen3VLTextEncoderFamily::new().unwrap(),
                &config,
                Qwen3VLTextEncoderFamily::METAL_PROFILE_ID,
                &options,
                &inputs,
                false,
            );
            assert_eq!(ordinary.len(), 1);
            let output = id::<ProgramValueId>(Qwen3VLTextEncoderFamily::OUTPUT_ID);
            assert_eq!(ordinary[&output], inspected[&output]);
            eprintln!("Metal 4-layer text GQA head={head} editing={editing}: all boundaries and ordinary pre-final-norm output agree");
        }
    }
}
