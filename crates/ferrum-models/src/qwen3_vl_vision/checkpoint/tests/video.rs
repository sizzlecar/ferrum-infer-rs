use super::*;

#[test]
fn complete_video_vision_matches_independent_temporal_components_without_cross_frame_attention() {
    let composition =
        ReferenceVNextComposition::create(id("device.reference.video_vision")).unwrap();
    qualify(
        composition.runtime().clone(),
        composition.registry(),
        composition.catalog(),
        composition.weight_materializers(),
        Qwen3VLVisionFamily::REFERENCE_PROFILE_ID,
        8,
    );
}

fn qualify<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
    profile: &str,
    hidden: u64,
) {
    let mut metadata = test_fixture::metadata();
    metadata["vision_config"]["num_heads"] = 1.into();
    metadata["vision_config"]["hidden_size"] = hidden.into();
    let architecture = Qwen3VLVisionArchitecture::from_hf_value(&metadata).unwrap();
    let original = test_fixture::Original::new(
        &crate::qwen3_vl_vision::comfy::Qwen3VLVisionComfyConfig::new(
            architecture.clone(),
            Qwen3VLVisionGrid::new(1, 2, 2).unwrap(),
        )
        .unwrap(),
    );
    let (_directory, archive) = original.archive();
    let archive = Arc::new(archive);
    let columns = architecture.patch_columns().unwrap() as usize;
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            16 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                output_bytes: 16 << 20,
                transfer_staging_bytes: 16 << 20,
                weight_backing_bytes: 16 << 20,
            },
        )
        .unwrap(),
    );
    let d = runtime.descriptor();
    let run = |which: &[usize]| {
        let c = Qwen3VLVisionConfig::new(
            architecture.clone(),
            vec![Qwen3VLVisionGrid::new(which.len() as u64, 2, 2).unwrap()],
        )
        .unwrap();
        let p = PreparedQwen3VLVisionCheckpoint::prepare(
            c,
            archive.clone(),
            &NumericalProfileId::new(profile).unwrap(),
            Qwen3VLVisionCheckpointLimits {
                maximum_metadata_bytes: 2 << 20,
                maximum_plan_nodes: 256,
                maximum_source_bytes: 4 << 20,
            },
        )
        .unwrap();
        let compiled = CompiledTensorComponent::compile(
            p.family().clone(),
            registry,
            catalog,
            &TensorComponentPolicy::new(
                d.total_memory_bytes,
                d.dynamic_storage_profiles.iter().copied().collect(),
            )
            .unwrap(),
            materializers,
            p.compile_options(),
        )
        .unwrap();
        let mut session = compiled
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::Stored(&p.managed_source(&host)),
                ComponentInitializationConfig {
                    run_id: id("run.video.vision.init"),
                    request_id: id("request.video.vision"),
                    transaction_id: id("transaction.video.vision.init"),
                    static_policy: StaticInitializationPolicy::new(4 << 20, 64).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let mut inputs = p.managed_position_inputs(&host).unwrap().to_vec();
        let mut patches = host
            .allocate_zeroed(
                HostMemoryClass::InputPreprocess,
                which.len() * 4 * columns * 2,
            )
            .unwrap();
        for (slot, &frame) in which.iter().enumerate() {
            for (j, bytes) in patches.as_mut()[slot * 4 * columns * 2..(slot + 1) * 4 * columns * 2]
                .chunks_exact_mut(2)
                .enumerate()
            {
                let value = (((j * (frame + 1) * 7 + frame * 13) % 41) as f32 - 20.) / 16.;
                bytes.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
            }
        }
        inputs.push(
            ManagedHostTensor::new(
                id(Qwen3VLVisionFamily::PATCHES_ID),
                ResolvedTensorSpec::new(
                    vec![which.len() as u64 * 4, columns as u64],
                    ElementType::Bf16,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                patches.freeze(),
            )
            .unwrap(),
        );
        let output = session
            .execute(
                id("run.video.vision.forward"),
                id("request.video.vision"),
                inputs,
            )
            .unwrap();
        let mut bytes = std::collections::BTreeMap::new();
        for tensor in output {
            bytes.insert(
                tensor.value_id().as_str().to_owned(),
                tensor.bytes().bytes().to_vec(),
            );
        }
        assert!(session.close().is_ok());
        bytes
    };
    let together = run(&[0, 1, 2]);
    let separate = [run(&[0]), run(&[1]), run(&[2])];
    let changed = run(&[0, 5, 2]);
    let mut changed_outputs = 0;
    for (name, bytes) in together {
        assert_eq!(
            bytes,
            separate
                .iter()
                .flat_map(|one| one[&name].iter().copied())
                .collect::<Vec<_>>()
        );
        let stride = bytes.len() / 3;
        assert_eq!(&changed[&name][..stride], &bytes[..stride]);
        assert_eq!(&changed[&name][2 * stride..], &bytes[2 * stride..]);
        changed_outputs +=
            usize::from(changed[&name][stride..2 * stride] != bytes[stride..2 * stride]);
    }
    assert_eq!(changed_outputs, 4);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[cfg(feature = "cuda")]
mod cuda;
