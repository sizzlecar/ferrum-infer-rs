//! Analytic structured attention checks on admitted CUDA tensors.
use super::*;
use ferrum_kernels::backend::cuda::vnext_ops::{
    cuda_vnext_tensor_attention_providers, CUDA_STRUCTURED_ATTENTION_BF16_NUMERICAL_PROFILE_ID,
};
use half::bf16;
#[path = "attention/family.rs"]
mod family;
#[path = "attention/newfused.rs"]
mod newfused;
use family::{AttentionConfig, AttentionFamily};
const INPUTS: [&str; 3] = [
    "value.attention.query",
    "value.attention.key",
    "value.attention.value",
];

fn config(
    rows: u64,
    columns: u64,
    sequence: u64,
    head: u64,
    lists: [Vec<i64>; 5],
) -> AttentionConfig {
    let mut attributes = BTreeMap::from([
        (id("sequence_length"), SemanticValue::Unsigned(sequence)),
        (id("head_dim"), SemanticValue::Unsigned(head)),
        (
            id("scale"),
            SemanticValue::Rational(CanonicalRational::new(1, 1).unwrap()),
        ),
    ]);
    for (name, values) in [
        "segment_lengths",
        "segment_bidirectional",
        "key_run_counts",
        "key_run_lengths",
        "key_run_valid",
    ]
    .into_iter()
    .zip(lists)
    {
        attributes.insert(id(name), SemanticValue::Integers(values));
    }
    AttentionConfig {
        rows,
        columns,
        attributes,
    }
}

fn run_case(name: &str, config: AttentionConfig, inputs: [Vec<u16>; 3], expected: &[u16]) {
    let elements = (config.rows * config.columns) as usize;
    assert_eq!(expected.len(), elements);
    assert!(inputs.iter().all(|x| x.len() == elements));
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: USABLE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "experimental-unqualified-attention-smoke-v1".into(),
        })
        .unwrap(),
    );
    let contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(silu_bf16_contract().unwrap()),
        Box::new(silu_mul_bf16_contract().unwrap()),
        Box::new(tanh_bf16_contract().unwrap()),
        Box::new(modulation_scale_bf16_contract().unwrap()),
        Box::new(gated_residual_bf16_contract().unwrap()),
        Box::new(layer_norm_bf16_contract().unwrap()),
        Box::new(head_rms_norm_bf16_contract().unwrap()),
        Box::new(adjacent_rope_bf16_contract().unwrap()),
        Box::new(structured_attention_bf16_contract().unwrap()),
    ];
    let registry = OperationRuntimeRegistry::new(
        contracts,
        cuda_vnext_tensor_attention_providers(&runtime).unwrap(),
    )
    .unwrap();
    assert!(cuda_vnext_tensor_attention_providers(&runtime).is_err());
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.tensor.attention.fixture"),
        ContractVersion::new(1, 0),
        runtime
            .descriptor()
            .runtime_implementation_fingerprint
            .clone(),
        runtime.descriptor().id.clone(),
        runtime.descriptor().capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(runtime.descriptor().clone(), vec![engine])
        .unwrap();
    let family = TypedFamilyRegistration::new(AttentionFamily)
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &NumericalProfileId::new(CUDA_STRUCTURED_ATTENTION_BF16_NUMERICAL_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(
        INPUTS
            .into_iter()
            .map(|name| (id(name), config.spec()))
            .collect(),
    )
    .unwrap();
    options.retain_completion_value(id(OUTPUT_ID));
    let compiled = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy(runtime.as_ref()),
        &registry.planning(),
        &options,
    )
    .unwrap();
    let executable = compiled.executable();
    let resources = no_static(runtime.clone(), executable.execution_plan());
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            elements * 8,
            HostMemoryClassLimits {
                input_preprocess_bytes: elements * 6,
                output_bytes: elements * 2,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let mut executor = PreparedTensorExecutor::new(
        executable.clone(),
        registry.bind_plan(executable).unwrap(),
        resources.clone(),
        host.clone(),
        TensorExecutorConfig::default(),
    )
    .unwrap();
    let tensors = INPUTS
        .into_iter()
        .zip(inputs)
        .map(|(name, values)| {
            super::normalization::managed_input(
                &host,
                name,
                vec![config.rows, config.columns],
                &values,
            )
        })
        .collect();
    let outputs = executor
        .execute(
            id(format!("run.cuda.attention.{name}")),
            id(format!("request.cuda.attention.{name}")),
            tensors,
        )
        .unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(host.owns(outputs[0].bytes()));
    let actual: Vec<u16> = outputs[0]
        .bytes()
        .bytes()
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect();
    assert_eq!(actual, expected, "{name}");
    drop((outputs, executor, compiled, registry));
    close_resources(resources);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(cuda_vnext_tensor_attention_providers(&runtime).unwrap());
}

#[test]
#[ignore = "requires exclusive CUDA; managed structured attention analytic probes"]
fn managed_cuda_structured_attention_masks_online_softmax_and_empty_rows() {
    // Q=K=0 makes each visible softmax probability exactly 1/count. All V
    // sums and final answers below are small exact integers/binary fractions.
    // Adjacent bidirectional segments are deliberately distinct image blocks.
    let shape = config(
        20,
        256,
        10,
        128,
        [
            vec![2, 3, 2, 3],
            vec![0, 1, 1, 1],
            vec![3, 5],
            vec![3, 1, 6, 3, 1, 4, 1, 1],
            vec![1, 0, 1, 1, 0, 1, 0, 1],
        ],
    );
    let q = vec![0; 20 * 256];
    let mut k = q.clone();
    let mut v = q.clone();
    let mut expected = q.clone();
    let values = [0., 1., 2., 0., 4., 5., 6., 7., 8., 12.];
    for batch in 0..2 {
        for row in 0..10 {
            let invalid = row == 3 || (batch == 1 && row == 8);
            let mean = match row {
                0 => 0.,
                1 => 0.5,
                2..=4 => 1.75,
                5..=6 => 3.,
                _ => {
                    if batch == 0 {
                        5.
                    } else {
                        4.625
                    }
                }
            };
            for column in 0..256 {
                let coordinate = |base: f32| {
                    if column < 128 {
                        if column % 2 == 0 {
                            base
                        } else {
                            -base
                        }
                    } else if column % 2 == 0 {
                        2. * base
                    } else {
                        8.
                    }
                };
                let index = (batch * 10 + row) * 256 + column;
                v[index] = if invalid {
                    0x7fc0
                } else {
                    bf16::from_f32(coordinate(values[row])).to_bits()
                };
                if invalid {
                    k[index] = 0x7f80;
                }
                expected[index] = bf16::from_f32(coordinate(mean)).to_bits();
            }
        }
    }
    // IEEE +0 accumulation of negative zeros can yield +0; the first visible
    // row is copied before normalization, so -0 remains in its odd channels.
    run_case("segmented_padding", shape, [q, k, v], &expected);
    // Finite nonuniform scores 0,+/-1 produce 2*sigmoid(+/-1), doubled in
    // the second coordinate. Independent exp bounds put these far from BF16
    // midpoints: 1.462117..., 2.924234..., .537882..., 1.075765....
    let shape = config(2, 2, 2, 2, [vec![2], vec![1], vec![1], vec![2], vec![1]]);
    run_case(
        "nonuniform",
        shape,
        [
            vec![0x3f80, 0, 0xbf80, 0],
            vec![0, 0, 0x3f80, 0],
            vec![0, 0, 0x4000, 0x4080],
        ],
        &[0x3fbb, 0x403b, 0x3f0a, 0x3f8a],
    );
    // An empty visible set must produce +0 even with NaN/Inf in all inputs.
    let shape = config(2, 2, 2, 2, [vec![2], vec![0], vec![1], vec![2], vec![0]]);
    run_case(
        "empty",
        shape,
        [vec![0x7fc0; 4], vec![0x7f80; 4], vec![0x7fc0; 4]],
        &[0; 4],
    );
    eprintln!("CUDA managed structured attention: B2/S10/H2/D128 masks and hidden padding, nonuniform softmax, empty rows and retirement passed; no full-block or throughput claim");
}
