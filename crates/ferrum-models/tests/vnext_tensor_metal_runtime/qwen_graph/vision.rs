//! Original family graph: three isolated images and all DeepStack mergers.
use super::*;
use ferrum_models::qwen3_vl_vision::*;

fn config(head: u64) -> Qwen3VLVisionConfig {
    let architecture = Qwen3VLVisionArchitecture::from_hf_value(&serde_json::json!({
        "architectures":["Qwen3VLForConditionalGeneration"], "model_type":"qwen3_vl",
        "vision_config": { "model_type":"qwen3_vl", "dtype":"bfloat16",
            "hidden_act":"gelu_pytorch_tanh", "depth":4, "hidden_size":head * 2,
            "intermediate_size":head * 3, "num_heads":2, "num_position_embeddings":16,
            "out_hidden_size":64, "deepstack_visual_indexes":[0,1,2],
            "in_channels":3, "patch_size":2, "temporal_patch_size":2, "spatial_merge_size":2 }
    }))
    .unwrap();
    Qwen3VLVisionConfig::new(
        architecture,
        vec![
            Qwen3VLVisionGrid::new(1, 2, 2).unwrap(),
            Qwen3VLVisionGrid::new(1, 4, 6).unwrap(),
            Qwen3VLVisionGrid::new(1, 2, 6).unwrap(),
        ],
    )
    .unwrap()
}

pub(super) fn maximum_command_count() -> usize {
    let prepared = TypedFamilyRegistration::new(Qwen3VLVisionFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(config(72)).unwrap(),
            &id(Qwen3VLVisionFamily::METAL_PROFILE_ID),
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
    // 72 is the trained architecture's non-power-of-two attention head width.
    for head in [4, 72] {
        let config = config(head);
        let inputs = fixture::inputs(&config.input_tensor_specs().unwrap());
        let options = config.compile_options().unwrap();
        let inspected = execute(
            runtime,
            registry,
            catalog,
            host,
            Qwen3VLVisionFamily::new().unwrap(),
            &config,
            Qwen3VLVisionFamily::METAL_PROFILE_ID,
            &options,
            &inputs,
            true,
        );
        let ordinary = execute(
            runtime,
            registry,
            catalog,
            host,
            Qwen3VLVisionFamily::new().unwrap(),
            &config,
            Qwen3VLVisionFamily::METAL_PROFILE_ID,
            &options,
            &inputs,
            false,
        );
        assert_eq!(ordinary.len(), 4);
        for name in std::iter::once(Qwen3VLVisionFamily::OUTPUT_ID)
            .chain(Qwen3VLVisionFamily::DEEPSTACK_IDS)
        {
            assert_eq!(
                ordinary[&id::<ProgramValueId>(name)],
                inspected[&id::<ProgramValueId>(name)]
            );
        }
        // Perturb only the first image's patch rows. Every output row belonging
        // to the other two images must remain bit-identical in the whole graph.
        let mut changed_inputs = inputs.clone();
        let patches = changed_inputs
            .get_mut(&id::<ProgramValueId>(Qwen3VLVisionFamily::PATCHES_ID))
            .unwrap();
        let length = (config.grids()[0].rows().unwrap()
            * config.architecture().patch_columns().unwrap()) as usize;
        for (i, x) in patches.values[..length].iter_mut().enumerate() {
            *x = rounds(*x + (i % 3) as f32 - 0.5);
        }
        let changed = execute(
            runtime,
            registry,
            catalog,
            host,
            Qwen3VLVisionFamily::new().unwrap(),
            &config,
            Qwen3VLVisionFamily::METAL_PROFILE_ID,
            &options,
            &changed_inputs,
            false,
        );
        let unchanged_start = config.architecture().output_size() as usize; // First image merges to one row.
        for name in ordinary.keys() {
            assert_eq!(
                &ordinary[name][unchanged_start..],
                &changed[name][unchanged_start..]
            );
        }
        assert_ne!(
            ordinary[&id::<ProgramValueId>(Qwen3VLVisionFamily::OUTPUT_ID)][..unchanged_start],
            changed[&id::<ProgramValueId>(Qwen3VLVisionFamily::OUTPUT_ID)][..unchanged_start]
        );
        eprintln!("Metal 4-layer vision head={head}: every boundary, main/DeepStack outputs, image isolation, repeat and retirement passed");
    }
}
