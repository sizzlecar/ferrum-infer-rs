//! Vendor fused attention is tested independently of the legacy IEEE profile.
//! Native library selection is explicit; these ignored tests do not report
//! success without constructing the actual cuDNN plans and executing on CUDA.
use super::*;
use ferrum_kernels::backend::cuda::vnext_ops::{
    cuda_vnext_tensor_providers_with_cudnn_sdpa,
    CUDA_STRUCTURED_ATTENTION_FUSED_BF16_NUMERICAL_PROFILE_ID,
};
use ferrum_kernels::backend::cuda::{CudaCudnnConfig, CudnnSdpaPlanPolicy};

#[path = "newfused_family.rs"]
mod fused_family;
#[path = "newfused_oracle.rs"]
mod oracle;
use fused_family::FusedFamily;
use oracle::{Expected, Probe, COLUMNS};

const PROFILE: &str = CUDA_STRUCTURED_ATTENTION_FUSED_BF16_NUMERICAL_PROFILE_ID;
const WORKSPACE_LIMIT: u64 = 64 << 20;

fn cuda_config() -> (CudaCudnnConfig, CudnnSdpaPlanPolicy) {
    let library_path = std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY_PATH")
        .map(std::path::PathBuf::from)
        .expect(
            "set FERRUM_TEST_CUDNN_LIBRARY_PATH to an explicit absolute cuDNN 9.20 library path",
        );
    assert!(library_path.is_absolute());
    (
        CudaCudnnConfig {
            library_path,
            opaque_memory_bytes: 64 << 20,
            workspace_limit_bytes: WORKSPACE_LIMIT,
            opaque_memory_profile: "experimental-fused-attention-test-v1".into(),
        },
        CudnnSdpaPlanPolicy {
            maximum_plans: 16,
            maximum_live_bindings: 64,
            opaque_memory_bytes: 128 << 20,
            workspace_limit_bytes: WORKSPACE_LIMIT,
        },
    )
}

fn read_bits(bytes: &ManagedHostBytes) -> Vec<u16> {
    bytes
        .bytes()
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect()
}

fn run_case(name: &str, probe: Probe, expected: Expected) {
    let (cudnn, sdpa) = cuda_config();
    let elements = probe.inputs[0].len();
    let output_bytes = elements * 2;
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: USABLE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "experimental-fused-attention-test-v1".into(),
        })
        .unwrap(),
    );
    // The selected native set is empty. This factory still owns the shared
    // range/pack/zero module and returns the independent Conv + fused providers.
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(conv2d_bf16_contract().unwrap()),
            Box::new(structured_attention_fused_bf16_contract().unwrap()),
        ],
        cuda_vnext_tensor_providers_with_cudnn_sdpa(&runtime, &cudnn, sdpa, &[]).unwrap(),
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.tensor.fused_attention.fixture"),
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
    let family = TypedFamilyRegistration::new(FusedFamily)
        .prepare_with_profile(
            &serde_json::to_value(&probe.config).unwrap(),
            &NumericalProfileId::new(PROFILE).unwrap(),
        )
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(
        INPUTS
            .into_iter()
            .map(|name| (id(name), probe.config.spec()))
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
    let plan = executable.execution_plan();
    assert!(plan.payload().memory().static_allocations().is_empty());
    let nodes = plan.payload().nodes();
    assert_eq!(nodes.len(), 1);
    let node = &nodes[0];
    assert_eq!(
        node.operation_id().as_str(),
        STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID
    );
    let all_empty = expected.empty_rows.iter().all(|&x| x);
    if all_empty {
        assert!(node.provider_resources().scratch().is_none());
        assert!(node.scratch_resource().is_none());
    } else {
        // The final segment packs 3 batches with max 5 valid KV positions.
        // K and V plus two aligned i32 length arrays must be included before
        // the separately queried native workspace. Sequential segments share
        // this one admitted region; no full SxS score tensor is provisioned.
        let packing_bytes = 2 * 3 * 5 * COLUMNS as u64 * 2 + 2 * 256;
        let scratch = node.provider_resources().scratch().unwrap();
        let bytes = scratch.fixed_bytes().unwrap();
        assert!(bytes >= packing_bytes);
        assert!(bytes <= packing_bytes + WORKSPACE_LIMIT);
        assert!(node.scratch_resource().is_some());
        eprintln!("{name}: declared invocation scratch={bytes} bytes, packing minimum={packing_bytes}, workspace cap={WORKSPACE_LIMIT}; opaque allowances are declarations, not measured native peaks");
    }
    eprintln!(
        "{name}: profile={PROFILE}, provider implementation={}, estimator input={}",
        node.provider_implementation_fingerprint(),
        node.provider_resources().estimator_input_fingerprint()
    );
    let resources = no_static(runtime.clone(), plan);
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            4 * output_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: 3 * output_bytes,
                output_bytes,
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
    let sanitized = probe.sanitized_inputs();
    let mut previous = None;
    let mut retained = None;
    for (ordinal, inputs) in [&probe.inputs, &sanitized].into_iter().enumerate() {
        let tensors = INPUTS
            .into_iter()
            .zip(inputs)
            .map(|(name, values)| {
                super::super::normalization::managed_input(
                    &host,
                    name,
                    vec![probe.config.rows, probe.config.columns],
                    values,
                )
            })
            .collect();
        let outputs = executor
            .execute(
                id(format!("run.cuda.fused_attention.{name}.{ordinal}")),
                id(format!("request.cuda.fused_attention.{name}.{ordinal}")),
                tensors,
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(host.owns(outputs[0].bytes()));
        assert_eq!(
            outputs[0].tensor().dimensions(),
            [probe.config.rows, probe.config.columns]
        );
        let actual = read_bits(outputs[0].bytes());
        let stats = expected.check(&actual).unwrap();
        assert_eq!(stats.compared_elements, elements);
        eprintln!(
            "{name} execution {ordinal}: {stats:?}; tolerance abs={} + rel={}*abs(F64 reference)",
            oracle::ABSOLUTE_TOLERANCE,
            oracle::RELATIVE_TOLERANCE
        );
        if let Some(previous) = &previous {
            assert_eq!(
                &actual, previous,
                "masked K/V payload must not affect any output query"
            );
        }
        previous = Some(actual);
        let bytes = outputs[0].bytes().clone();
        drop(outputs);
        assert_eq!(host.usage().unwrap().allocated_bytes(), output_bytes);
        assert_eq!(host.usage().unwrap().committed_bytes(), output_bytes);
        if ordinal == 1 {
            retained = Some(bytes);
        } else {
            drop(bytes);
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        }
    }
    drop((executor, compiled, registry));
    close_resources(resources);
    // A fresh provider bundle and stream must fit after the old native plans,
    // scratch session and bindings retire, while the host output stays valid.
    drop(cuda_vnext_tensor_providers_with_cudnn_sdpa(&runtime, &cudnn, sdpa, &[]).unwrap());
    drop(runtime.create_stream().unwrap());
    drop(runtime);
    let retained = retained.unwrap();
    expected.check(&read_bits(&retained)).unwrap();
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
#[ignore = "requires exclusive CUDA and explicit FERRUM_TEST_CUDNN_LIBRARY_PATH for cuDNN 9.20"]
fn managed_cuda_fused_attention_independent_reference_masks_scratch_and_retirement() {
    let segmented = Probe::segmented();
    let empty = Probe::empty();
    // Prepare both independent expectations before constructing any candidate
    // runtime/provider. Unsupported vendor shapes remain real test failures.
    let expected_segmented = Expected::prepare(&segmented);
    let expected_empty = Expected::prepare(&empty);
    run_case("segmented", segmented, expected_segmented);
    run_case("empty", empty, expected_empty);
    eprintln!("Fused attention: bounded D128 samples checked against independent F64 softmax; no full-domain precision, legacy-profile equivalence, Torch parity or throughput claim");
}

#[test]
fn fused_fixture_prepares_distinct_profile_and_exact_structured_mask_without_cuda() {
    for probe in [Probe::segmented(), Probe::empty()] {
        let prepared = TypedFamilyRegistration::new(FusedFamily)
            .prepare_with_profile(
                &serde_json::to_value(&probe.config).unwrap(),
                &NumericalProfileId::new(PROFILE).unwrap(),
            )
            .unwrap();
        let operation = &prepared.program().blocks()[0].nodes[0].operation_id;
        assert_eq!(
            operation.as_str(),
            STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID
        );
        assert_ne!(operation.as_str(), STRUCTURED_ATTENTION_BF16_OPERATION_ID);
        let spec = ResolvedTensorSpec::new(
            vec![probe.config.rows, probe.config.columns],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        let geometry = StructuredAttentionBf16Geometry::new(
            &spec,
            &spec,
            &spec,
            &spec,
            &probe.config.attributes,
        )
        .unwrap();
        assert_eq!(geometry.mask().segment_ends(), [2, 5, 7]);
        assert_eq!(geometry.mask().segment_bidirectional(), [false, true, true]);
        for (batch, keys) in probe.valid_keys.iter().enumerate() {
            for (key, valid) in keys.iter().copied().enumerate() {
                assert_eq!(geometry.mask().key_is_valid(batch, key as u32), Some(valid));
            }
        }
        assert!(TypedFamilyRegistration::new(FusedFamily)
            .prepare_with_profile(
                &serde_json::to_value(&probe.config).unwrap(),
                &NumericalProfileId::new(CUDA_STRUCTURED_ATTENTION_BF16_NUMERICAL_PROFILE_ID)
                    .unwrap(),
            )
            .is_err());
    }
}
