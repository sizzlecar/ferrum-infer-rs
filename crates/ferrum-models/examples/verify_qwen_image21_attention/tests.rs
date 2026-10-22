use super::*;

#[test]
fn compiler_binds_frequency_period_to_attention_layout_before_allocation() {
    use ferrum_kernels::backend::reference::ReferenceVNextComposition;
    use ferrum_runtime::*;
    let transformer=QwenImage21Config::from_hf_json(&serde_json::to_vec(&json!({"_class_name":"QwenImage21Transformer2DModel","num_layers":1,"num_attention_heads":2,"attention_head_dim":8,"context_in_dim":8,"in_channels":3,"out_channels":4,"patch_size":1,"mlp_ratio":3,"axes_dims_rope":[2,2,4],"causal_condition":true,"eps":0.75})).unwrap(),StructureLimits::default()).unwrap();
    let config = QwenImage21AttentionConfig::new(
        transformer,
        QwenImage21AttentionLayout::uniform(2, true).unwrap(),
    )
    .unwrap();
    let family = TypedFamilyRegistration::new(Family::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &NumericalProfileId::new(Family::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let composition = ReferenceVNextComposition::create(
        DeviceId::new("device.reference.attention-shape-test").unwrap(),
    )
    .unwrap();
    let descriptor = composition.runtime().descriptor();
    let policy = shared::TensorPolicy {
        capacity_bytes: descriptor.total_memory_bytes,
        storage_profiles: descriptor
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    };
    let frequency = ProgramValueId::new(Family::FREQUENCIES_ID).unwrap();
    for variant in 0..4 {
        let mut specs = family.program().input_tensor_specs().clone();
        let tensor = specs.get_mut(&frequency).unwrap();
        match variant {
            0 => {}
            // Divides R and was previously accepted by the standalone RoPE contract.
            1 => tensor.dimensions[0] = 1,
            2 => tensor.element_type = ElementType::Bf16,
            3 => {
                tensor.layout = ResolvedTensorLayout::Strided {
                    byte_strides: vec![32, 8, 4],
                }
            }
            _ => unreachable!(),
        }
        let options = ProgramPlanCompileOptions::new(specs).unwrap();
        let result = ProgramPlanCompiler::compile_with_weight_materializers(
            &family,
            composition.catalog(),
            &policy,
            &composition.registry().planning(),
            composition.weight_materializers(),
            &options,
        );
        if variant == 0 {
            assert!(result.is_ok());
        } else {
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("model-declared tensor constraint"));
        }
        assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    }
}

#[test]
fn cli_requires_an_explicit_probe_and_rejects_unknown_or_duplicate_options() {
    let base = ["--config", "c", "--model-dir", "m", "--artifact-dir", "a"];
    let parse = |tail: &[&str]| {
        Arguments::parse(
            base.into_iter()
                .chain(tail.iter().copied())
                .map(OsString::from),
        )
    };
    assert!(parse(&[]).is_err());
    for (name, probe) in [
        ("single", Probe::Single),
        ("causal-pair", Probe::CausalPair),
        ("full-pair", Probe::FullPair),
    ] {
        assert_eq!(parse(&["--probe", name]).unwrap().unwrap().probe, probe);
    }
    for tail in [
        vec!["--probe", "unknown"],
        vec!["--probe", "single", "--unknown", "1"],
        vec!["--probe", "single", "--probe", "full-pair"],
    ] {
        assert!(parse(&tail).is_err());
    }
}

#[test]
fn tiny_stored_attention_runs_nine_nodes_and_distinguishes_pair_visibility() {
    let directory = tempfile::tempdir().unwrap();
    let config = directory.path().join("config.json");
    std::fs::write(&config,serde_json::to_vec(&json!({"_class_name":"QwenImage21Transformer2DModel","num_layers":1,"num_attention_heads":2,"attention_head_dim":8,"context_in_dim":8,"in_channels":3,"out_channels":4,"patch_size":1,"mlp_ratio":3,"axes_dims_rope":[2,2,4],"causal_condition":true,"eps":0.75})).unwrap()).unwrap();
    let bytes = |values: Vec<f32>| {
        values
            .into_iter()
            .flat_map(|v| half::bf16::from_f32(v).to_bits().to_le_bytes())
            .collect::<Vec<_>>()
    };
    let projection = |kind: usize| {
        bytes(
            (0..256)
                .map(|i| {
                    if i % 16 != 0 {
                        0.0
                    } else if kind == 2 {
                        ((i / 16) % 8 + 1) as f32 / 8.0
                    } else if (i / 16) % 2 == 0 {
                        0.5
                    } else {
                        -0.5
                    }
                })
                .collect(),
        )
    };
    let weights = [
        projection(0),
        projection(1),
        bytes(vec![1.; 8]),
        bytes(vec![1.; 8]),
        projection(2),
        bytes(
            (0..256)
                .map(|i| {
                    if i / 16 == i % 16 {
                        1.0
                    } else if (i / 16 + 1) % 16 == i % 16 {
                        0.5
                    } else {
                        0.0
                    }
                })
                .collect(),
        ),
    ];
    let views = weights
        .iter()
        .enumerate()
        .map(|(index, bytes)| {
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::BF16,
                if index == 2 || index == 3 {
                    vec![8]
                } else {
                    vec![16, 16]
                },
                bytes,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    safetensors::tensor::serialize_to_file(
        Family::WEIGHT_NAMES.into_iter().zip(views),
        &None,
        &directory.path().join("weights.safetensors"),
    )
    .unwrap();
    let weight_map = Family::WEIGHT_NAMES
        .into_iter()
        .map(|name| (name, "weights.safetensors"))
        .collect::<BTreeMap<_, _>>();
    std::fs::write(
        directory.path().join("test.index.json"),
        serde_json::to_vec(&json!({"weight_map":weight_map})).unwrap(),
    )
    .unwrap();
    let mut attention_outputs = Vec::new();
    for (name, probe) in [
        ("single", Probe::Single),
        ("causal", Probe::CausalPair),
        ("full", Probe::FullPair),
    ] {
        let artifact_dir = directory.path().join(name);
        run(Arguments {
            config: config.clone(),
            model_dir: directory.path().into(),
            index_file: "test.index.json".into(),
            artifact_dir: artifact_dir.clone(),
            backend: backend::Backend::Reference,
            probe,
        })
        .unwrap();
        let result: serde_json::Value =
            serde_json::from_slice(&std::fs::read(artifact_dir.join("result.json")).unwrap())
                .unwrap();
        assert_eq!(result["comparison"]["passed"], true);
        assert_eq!(result["cleanup"]["managed_host_committed_bytes"], 0);
        let execution: serde_json::Value =
            serde_json::from_slice(&std::fs::read(artifact_dir.join("execution.json")).unwrap())
                .unwrap();
        assert_eq!(execution["nodes"].as_array().unwrap().len(), 9);
        assert_eq!(
            execution["initialized_weight_bytes"],
            4 * 16 * 16 * 2 + 2 * 8 * 2
        );
        assert_eq!(
            execution["managed_input_bytes"],
            probe.rows() * (16 * 2 + 8 * 4)
        );
        assert_eq!(execution["managed_output_bytes"], probe.rows() * 16 * 2 * 9);
        assert_eq!(
            execution["canonical_config"]["layout"]["sequence_length"],
            probe.rows()
        );
        for filename in [
            "query.bf16",
            "key.bf16",
            "query-normalized.bf16",
            "key-normalized.bf16",
            "query-rotated.bf16",
            "key-rotated.bf16",
            "value.bf16",
            "attention.bf16",
            "projected.bf16",
        ] {
            assert_eq!(
                std::fs::metadata(artifact_dir.join(filename))
                    .unwrap()
                    .len(),
                (probe.rows() * 16 * 2) as u64
            );
        }
        attention_outputs.push(std::fs::read(artifact_dir.join("attention.bf16")).unwrap());
    }
    let expected_value = bytes((0..16).map(|i| (i % 8 + 1) as f32 / 8.0).collect());
    assert_eq!(attention_outputs[0], expected_value);
    assert_eq!(&attention_outputs[1][..32], expected_value.as_slice());
    assert_ne!(&attention_outputs[2][..32], expected_value.as_slice());
    assert_eq!(&attention_outputs[1][32..], &attention_outputs[2][32..]);
    assert!(attention_outputs[2]
        .chunks_exact(2)
        .any(|v| u16::from_le_bytes(v.try_into().unwrap()) & 0x7fff != 0));
    // An inspection error must retire the static arena while the runtime
    // itself remains alive; dropping the entire process cannot hide a leak.
    use ferrum_runtime::*;
    let transformer = QwenImage21Config::from_hf_json(
        &std::fs::read(&config).unwrap(),
        StructureLimits::default(),
    )
    .unwrap();
    let checked = QwenImage21AttentionConfig::new(
        transformer,
        QwenImage21AttentionLayout::uniform(1, false).unwrap(),
    )
    .unwrap();
    let family = TypedFamilyRegistration::new(Family::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(checked).unwrap(),
            &NumericalProfileId::new(Family::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let source = SafetensorsArchive::open_with_index(directory.path(), "test.index.json").unwrap();
    let runtime = ferrum_kernels::backend::reference::ReferenceVNextComposition::create(
        DeviceId::new("device.reference.inspect-error").unwrap(),
    )
    .unwrap();
    let input = std::fs::read(directory.path().join("single/input-x.bf16")).unwrap();
    let frequencies = std::fs::read(directory.path().join("single/input-frequencies.f32")).unwrap();
    let inputs = [
        (Family::INPUT_ID, input.as_slice()),
        (Family::FREQUENCIES_ID, frequencies.as_slice()),
    ]
    .into_iter()
    .map(|(name, bytes)| {
        let value_id = ProgramValueId::new(name).unwrap();
        let spec = &family.program().input_tensor_specs()[&value_id];
        shared::ComponentInput {
            value_id,
            tensor: ResolvedTensorSpec::new(
                spec.dimensions.clone(),
                spec.element_type,
                spec.layout.clone(),
            )
            .unwrap(),
            bytes,
        }
    })
    .collect();
    let result = shared::execute_stored_component(
        &family,
        &source,
        shared::StoredComponentRuntime {
            runtime: runtime.runtime().clone(),
            registry: runtime.registry(),
            catalog: runtime.catalog(),
            materializers: runtime.weight_materializers(),
            capacity_bytes: runtime.runtime().descriptor().total_memory_bytes,
            staging_bytes: 65536,
            provenance: json!({}),
        },
        inputs,
        |_, _| bail!("injected inspection failure"),
    );
    assert_eq!(
        result.unwrap_err().to_string(),
        "injected inspection failure"
    );
    assert_eq!(runtime.runtime().snapshot().live_allocations, 0);
}
