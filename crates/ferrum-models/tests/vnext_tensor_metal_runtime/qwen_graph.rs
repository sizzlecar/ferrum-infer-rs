//! Complete block and multi-layer DiT execution, including native Q8 matrices.
//! Generated weights qualify arithmetic/composition, not trained image quality.
use super::*;
use ferrum_execution::component::{
    CompiledTensorComponent, ComponentWeightSource, TensorComponentPolicy,
};
use ferrum_models::qwen_image21::*;
use transforms::{decode, initialization, upload_values};
#[path = "qwen_graph/fixture.rs"]
mod fixture;
#[path = "qwen_graph/oracle.rs"]
mod oracle;
use fixture::{Data, Inspect, Source};

fn architecture(layers: u64) -> QwenImage21Config {
    QwenImage21Config::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
            "_class_name":"QwenImage21Transformer2DModel", "num_layers":layers,
            "num_attention_heads":2, "attention_head_dim":32, "context_in_dim":32,
            "in_channels":32, "out_channels":32, "patch_size":1, "mlp_ratio":3,
            "axes_dims_rope":[8,8,16], "causal_condition":true, "eps":0.000001
        }))
        .unwrap(),
        StructureLimits::default(),
    )
    .unwrap()
}

fn execute<P: ModelFamilyProvider>(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    host: &HostMemoryAccount,
    family: P,
    config: &P::Config,
    profile: &str,
    options: &ProgramPlanCompileOptions,
    inputs: &BTreeMap<ProgramValueId, Data>,
    inspect: bool,
) -> BTreeMap<ProgramValueId, Vec<f32>> {
    let prepared = TypedFamilyRegistration::new(Inspect(family, inspect))
        .prepare_with_profile(&serde_json::to_value(config).unwrap(), &id(profile))
        .unwrap();
    let graph = prepared.program().clone();
    let source = Source::new(&prepared);
    let mut data = source.data.clone();
    data.extend(inputs.clone());
    let policy = TensorComponentPolicy::new(
        CAPACITY,
        runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    let component = CompiledTensorComponent::compile(
        prepared,
        registry,
        catalog,
        &policy,
        &WeightMaterializerRegistry::identity_only().unwrap(),
        options,
    )
    .unwrap();
    let mut session = component
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(&source),
            initialization("metal.qwen.graph"),
        )
        .unwrap();
    drop(source);
    let mut previous: Option<Vec<ManagedHostBytes>> = None;
    let mut result = BTreeMap::new();
    for repeat in 0..2 {
        let outputs = session
            .execute(
                id(format!("run.metal.qwen.graph.{repeat}")),
                id(format!("request.metal.qwen.graph.{repeat}")),
                inputs
                    .iter()
                    .map(|(name, d)| upload_values(host, name.clone(), &d.spec, &d.values))
                    .collect(),
            )
            .unwrap();
        for output in &outputs {
            let values = decode(output.bytes().bytes());
            result.insert(output.value_id().clone(), values.clone());
            data.insert(
                output.value_id().clone(),
                Data {
                    spec: ProgramTensorSpec {
                        dimensions: output.tensor().dimensions().to_vec(),
                        element_type: output.tensor().element_type(),
                        layout: output.tensor().layout().clone(),
                    },
                    values,
                },
            );
        }
        if inspect {
            oracle::graph(&graph, &data);
        }
        if let Some(previous) = &previous {
            for (a, b) in previous.iter().zip(&outputs) {
                assert_eq!(a.bytes(), b.bytes().bytes());
            }
        }
        previous = Some(outputs.iter().map(|o| o.bytes().clone()).collect());
    }
    assert!(matches!(
        session.try_release_idle_dynamic_backing().unwrap(),
        IdleDynamicReleaseOutcome::Released(_)
    ));
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    for bytes in previous.as_ref().unwrap() {
        assert!(decode(bytes.bytes()).iter().all(|v| v.is_finite()));
    }
    drop(previous);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    eprintln!(
        "Metal {} inspect={inspect}: {} nodes, all outputs finite, repeat and retirement passed",
        graph.family_id(),
        graph.blocks().iter().map(|b| b.nodes.len()).sum::<usize>()
    );
    result
}

fn transformer_case(native: bool, editing: bool) -> (QwenImage21TransformerConfig, &'static str) {
    let arch = architecture(2);
    let mut config = QwenImage21TransformerConfig::new(
        arch.clone(),
        QwenImage21T2iLayout::new(4, 2, 2, vec![4, 3]).unwrap(),
    )
    .unwrap();
    if editing {
        config = config.with_reference_image(1, 1, 2, 2).unwrap();
    }
    if native {
        let expected = ExpectedWeights::from_config(&arch, StructureLimits::default()).unwrap();
        let encodings = expected
            .tensors()
            .values()
            .map(|tensor| {
                let encoding = if tensor.dimensions().len() == 2 {
                    WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                        format_id: id("quantization.gguf.q8-0"),
                        logical_values_per_block: 32,
                        bytes_per_block: 34,
                    })
                } else {
                    WeightEncoding::Dense {
                        element_type: ElementType::Bf16,
                    }
                };
                (tensor.external_name().to_owned(), encoding)
            })
            .collect();
        config = config.with_weight_encodings(encodings).unwrap();
    }
    (
        config,
        if native {
            QwenImage21TransformerFamily::METAL_NATIVE_BLOCK_PROFILE_ID
        } else {
            QwenImage21TransformerFamily::METAL_PROFILE_ID
        },
    )
}

pub(super) fn maximum_command_count() -> usize {
    // Resolve before constructing the runtime, like a product factory. The
    // largest full DiT includes its block operations and one upload command
    // per input, as in the existing CUDA composition's command calculation.
    [false, true]
        .into_iter()
        .flat_map(|native| {
            [false, true].into_iter().map(move |editing| {
                let (config, profile) = transformer_case(native, editing);
                let prepared =
                    TypedFamilyRegistration::new(QwenImage21TransformerFamily::new().unwrap())
                        .prepare_with_profile(&serde_json::to_value(config).unwrap(), &id(profile))
                        .unwrap();
                prepared
                    .program()
                    .blocks()
                    .iter()
                    .map(|b| b.nodes.len())
                    .sum::<usize>()
                    + prepared.program().inputs().len()
            })
        })
        .max()
        .unwrap()
}

pub(super) fn verify(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
) {
    // All-boundary diagnostics retain many page-aligned small tensors twice.
    let host = runtime
        .tensor_host_account(
            HostMemoryLimits::new(
                32 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 8 << 20,
                    output_bytes: 24 << 20,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        )
        .unwrap();
    for broadcast in [
        QwenImage21Block0Broadcast::Row,
        QwenImage21Block0Broadcast::Elementwise,
    ] {
        let config = QwenImage21Block0Config::new(
            QwenImage21AttentionConfig::new(
                architecture(1),
                QwenImage21AttentionLayout::new(
                    17,
                    vec![
                        QwenImage21AttentionSegment::new(1, false),
                        QwenImage21AttentionSegment::new(16, true),
                    ],
                    vec![
                        vec![QwenImage21AttentionKeyRun::new(17, true)],
                        vec![
                            QwenImage21AttentionKeyRun::new(1, false),
                            QwenImage21AttentionKeyRun::new(16, true),
                        ],
                    ],
                )
                .unwrap(),
            )
            .unwrap(),
            broadcast,
        )
        .unwrap();
        let specs = config.input_tensor_specs().unwrap();
        execute(
            runtime,
            registry,
            catalog,
            &host,
            QwenImage21Block0Family::new().unwrap(),
            &config,
            QwenImage21Block0Family::METAL_PROFILE_ID,
            &ProgramPlanCompileOptions::new(specs.clone()).unwrap(),
            &fixture::inputs(&specs),
            true,
        );
    }
    for native in [false, true] {
        for editing in [false, true] {
            let (config, profile) = transformer_case(native, editing);
            let input = fixture::inputs(&config.input_tensor_specs().unwrap());
            let options = config.compile_options().unwrap();
            let inspected = execute(
                runtime,
                registry,
                catalog,
                &host,
                QwenImage21TransformerFamily::new().unwrap(),
                &config,
                profile,
                &options,
                &input,
                true,
            );
            let ordinary = execute(
                runtime,
                registry,
                catalog,
                &host,
                QwenImage21TransformerFamily::new().unwrap(),
                &config,
                profile,
                &options,
                &input,
                false,
            );
            assert_eq!(ordinary.len(), 1);
            assert_eq!(
                ordinary[&id::<ProgramValueId>(QwenImage21TransformerFamily::VELOCITY_ID)],
                inspected[&id::<ProgramValueId>(QwenImage21TransformerFamily::VELOCITY_ID)]
            );
            eprintln!("Metal 2-layer DiT native_q8={native} editing={editing}: every numerical boundary and actual target-only product graph agree");
        }
    }
}
