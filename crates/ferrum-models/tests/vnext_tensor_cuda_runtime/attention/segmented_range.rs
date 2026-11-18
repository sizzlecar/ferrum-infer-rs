//! Small BF16 range and segment-isolation baseline, independent of model weights.
use super::segmented_long::SegmentedFamily;
use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_vnext_cnn_f32_providers_with_segmented_sdpa, CudaTensorOperation,
        CUDA_CUDNN_SEGMENTED_ATTENTION_BF16_NUMERICAL_PROFILE_ID,
    },
    CudaCudnnConfig, CudnnSdpaPlanPolicy,
};

const LENGTHS: [usize; 2] = [2, 4];
const HEADS: usize = 2;
const DIM: usize = 72;
const COLUMNS: usize = HEADS * DIM;
const ROWS: usize = LENGTHS[0] + LENGTHS[1];
const BYTES: usize = ROWS * COLUMNS * 2;
const WORKSPACE_BYTES: u64 = 32 << 20;

fn input_tensor(host: &HostMemoryAccount, port: usize, values: [f32; 2]) -> ManagedHostTensor {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, BYTES)
        .unwrap();
    if port == 2 {
        let mut offset = 0;
        for (length, value) in LENGTHS.into_iter().zip(values) {
            let value = bf16::from_f32(value).to_bits().to_le_bytes();
            let end = offset + length * COLUMNS * 2;
            for scalar in bytes.as_mut()[offset..end].chunks_exact_mut(2) {
                scalar.copy_from_slice(&value);
            }
            offset = end;
        }
    }
    ManagedHostTensor::new(
        id(INPUTS[port]),
        input_spec(vec![ROWS as u64, COLUMNS as u64], ElementType::Bf16),
        bytes.freeze(),
    )
    .unwrap()
}

fn check_constant_means(bytes: &[u8], values: [f32; 2]) {
    assert_eq!(bytes.len(), BYTES);
    let mut offset = 0;
    for (segment, (length, expected)) in LENGTHS.into_iter().zip(values).enumerate() {
        // QK is zero, so softmax is uniform. The mean of a segment's constant
        // V is that constant. Lengths 2/4, probabilities 1/2 and 1/4, and these
        // power-of-two values are exactly representable in BF16 and F32.
        // This oracle needs no vendor output or candidate arithmetic replay.
        let end = offset + length * COLUMNS * 2;
        for (index, scalar) in bytes[offset..end].chunks_exact(2).enumerate() {
            let actual = bf16::from_bits(u16::from_le_bytes([scalar[0], scalar[1]])).to_f32();
            assert!(
                actual.is_finite(),
                "segment {segment}, scalar {index}: expected finite {expected}, got {actual}"
            );
            assert_eq!(actual, expected, "segment {segment}, scalar {index}");
        }
        offset = end;
    }
}

#[test]
#[ignore = "requires CUDA and explicit FERRUM_TEST_CUDNN_LIBRARY_PATH for cuDNN 9.20"]
fn managed_cuda_segmented_attention_d72_preserves_bf16_range_and_segment_isolation() {
    let values = [[131_072.0, -262_144.0], [524_288.0, -262_144.0]];
    for value in values.into_iter().flatten() {
        assert_eq!(bf16::from_f32(value).to_f32(), value);
        assert!(value.is_finite());
        assert!(half::f16::from_f32(value).is_infinite());
    }
    let library_path = std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY_PATH")
        .map(std::path::PathBuf::from)
        .expect("explicit cuDNN 9.20 library path");
    assert!(library_path.is_absolute());
    let cudnn = CudaCudnnConfig {
        library_path,
        dependency_libraries: std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCY_LIBRARIES")
            .map(|paths| std::env::split_paths(&paths).collect())
            .unwrap_or_default(),
        opaque_memory_bytes: 64 << 20,
        workspace_limit_bytes: WORKSPACE_BYTES,
        opaque_memory_profile: "segmented-attention-bf16-range-fixture-v1".into(),
    };
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: USABLE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "segmented-attention-bf16-range-fixture-v1".into(),
        })
        .unwrap(),
    );
    let providers = cuda_vnext_cnn_f32_providers_with_segmented_sdpa(
        &runtime,
        &cudnn,
        CudnnSdpaPlanPolicy {
            maximum_plans: LENGTHS.len(),
            maximum_live_bindings: LENGTHS.len(),
            opaque_memory_bytes: 128 << 20,
            workspace_limit_bytes: WORKSPACE_BYTES,
        },
        &[CudaTensorOperation::SegmentedAttention],
    )
    .unwrap();
    let selected: Vec<_> = providers
        .iter()
        .filter(|provider| {
            provider.descriptor().operation_id().as_str() == SEGMENTED_ATTENTION_BF16_OPERATION_ID
        })
        .collect();
    assert_eq!(selected.len(), 1);
    assert_eq!(
        selected[0].descriptor().provider_id().as_str(),
        "provider.cuda.segmented_attention.bf16.cudnn920"
    );
    drop(selected);
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(conv2d_f32_contract().unwrap()),
            Box::new(segmented_attention_bf16_contract().unwrap()),
        ],
        providers,
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.segmented.range.fixture"),
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
    let family = TypedFamilyRegistration::new(SegmentedFamily {
        head_dim: DIM,
        lengths: LENGTHS.to_vec(),
    })
    .prepare_with_profile(
        &serde_json::json!({}),
        &NumericalProfileId::new(CUDA_CUDNN_SEGMENTED_ATTENTION_BF16_NUMERICAL_PROFILE_ID).unwrap(),
    )
    .unwrap();
    let spec = ProgramTensorSpec {
        dimensions: vec![ROWS as u64, COLUMNS as u64],
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    };
    let mut options = ProgramPlanCompileOptions::new(
        INPUTS
            .into_iter()
            .map(|name| (id(name), spec.clone()))
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
    let nodes = executable.execution_plan().payload().nodes();
    assert_eq!(nodes.len(), 1);
    assert_eq!(
        nodes[0].operation_id().as_str(),
        SEGMENTED_ATTENTION_BF16_OPERATION_ID
    );
    let resources = no_static(runtime.clone(), executable.execution_plan());
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            5 * BYTES,
            HostMemoryClassLimits {
                input_preprocess_bytes: 3 * BYTES,
                output_bytes: 2 * BYTES,
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
    let mut previous: Option<ManagedHostBytes> = None;
    for (iteration, values) in values.into_iter().enumerate() {
        let inputs = (0..3)
            .map(|port| input_tensor(&host, port, values))
            .collect();
        let outputs = executor
            .execute(
                id(format!("run.segmented.range.{iteration}")),
                id(format!("request.segmented.range.{iteration}")),
                inputs,
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0].tensor().dimensions(),
            [ROWS as u64, COLUMNS as u64]
        );
        assert!(host.owns(outputs[0].bytes()));
        let bytes = outputs[0].bytes().clone();
        check_constant_means(bytes.bytes(), values);
        if let Some(previous) = previous.take() {
            let boundary = LENGTHS[0] * COLUMNS * 2;
            assert_eq!(
                &bytes.bytes()[boundary..],
                &previous.bytes()[boundary..],
                "changing the first segment must not affect the second segment"
            );
            assert_ne!(&bytes.bytes()[..boundary], &previous.bytes()[..boundary]);
        }
        drop(outputs);
        previous = Some(bytes);
        assert_eq!(host.usage().unwrap().allocated_bytes(), BYTES);
    }
    drop((executor, compiled, catalog, registry));
    close_resources(resources);
    drop(runtime);
    drop(previous);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
