//! Actual direct-F32 CUDA execution, native cuRAND mapping and managed owners.
//! This does not establish Torch CPU seed42, toolkit, or cross-device parity.
use super::*;
use ferrum_kernels::backend::cuda::vnext_ops::{
    cuda_vnext_tensor_providers, CudaPhiloxDeviceGeometry, CudaTensorOperation,
    CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_ID, CUDA_PHILOX_NORMAL_F32_NUMERICAL_PROFILE_ID,
};
use ferrum_models::normal_noise::{NormalNoiseConfig, NormalNoiseFamily, NormalNoisePrecision};

#[path = "../support/philox_normal_oracle.rs"]
mod oracle;

fn execute_states(
    runtime: &Arc<CudaDeviceRuntime>,
    registry: &OperationRuntimeRegistry<CudaDeviceRuntime>,
    catalog: &CapabilityCatalog,
    host: &HostMemoryAccount,
    precision: NormalNoisePrecision,
    count: u64,
    states: &[PhiloxSeedOffset],
) -> Vec<ManagedHostBytes> {
    let profile = match precision {
        NormalNoisePrecision::Bf16 => CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_ID,
        NormalNoisePrecision::F32 => CUDA_PHILOX_NORMAL_F32_NUMERICAL_PROFILE_ID,
    };
    let config = NormalNoiseConfig::new_with_precision(1, 1, count, precision).unwrap();
    let prepared = TypedFamilyRegistration::new(
        NormalNoiseFamily::new_with_precision(precision, profile).unwrap(),
    )
    .prepare_with_profile(
        &serde_json::to_value(&config).unwrap(),
        &NumericalProfileId::new(profile).unwrap(),
    )
    .unwrap();
    let compiled = ProgramPlanCompiler::compile(
        &prepared,
        catalog,
        &TensorPolicy {
            capacity_bytes: 1 << 30,
            ..policy(runtime.as_ref())
        },
        &registry.planning(),
        &config.compile_options().unwrap(),
    )
    .unwrap();
    let executable = compiled.executable();
    let plan = executable.execution_plan();
    assert!(plan.payload().memory().static_allocations().is_empty());
    let node = &plan.payload().nodes()[0];
    assert!(node.provider_resources().scratch().is_none());
    assert!(node.provider_resources().binding().is_none());
    assert!(node.provider_resources().persistent().is_none());
    let resources = no_static(runtime.clone(), plan);
    let mut executor = PreparedTensorExecutor::new(
        executable.clone(),
        registry.bind_plan(executable).unwrap(),
        resources.clone(),
        host.clone(),
        TensorExecutorConfig::default(),
    )
    .unwrap();
    let mut results = Vec::new();
    for (ordinal, &state) in states.iter().enumerate() {
        let mut bytes = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 16)
            .unwrap();
        bytes.as_mut().copy_from_slice(&state.little_endian_bytes());
        let input = ManagedHostTensor::new(
            id(NormalNoiseFamily::STATE_ID),
            NormalNoiseConfig::state_tensor().unwrap(),
            bytes.freeze(),
        )
        .unwrap();
        let output = executor
            .execute(
                id(format!(
                    "run.cuda.philox.f32.{precision:?}.{count}.{ordinal}"
                )),
                id(format!(
                    "request.cuda.philox.f32.{precision:?}.{count}.{ordinal}"
                )),
                vec![input.clone()],
            )
            .unwrap();
        assert_eq!(
            input.bytes().bytes(),
            state.little_endian_bytes(),
            "read-only seed/offset changed"
        );
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].tensor(), &config.output_tensor().unwrap());
        assert!(host.owns(output[0].bytes()));
        results.push(output[0].bytes().clone());
        drop((output, input));
        if ordinal == 1 {
            // Subsequent fresh inputs must reproduce independently of warm or
            // explicitly released dynamic backing, with no hidden RNG state.
            let retained = resources
                .try_retain_idle_dynamic_backing(
                    plan.payload().memory().minimum_runnable_request_bytes(),
                )
                .unwrap();
            assert!(matches!(retained, IdleDynamicBackingOutcome::Retained(_)));
            assert!(matches!(
                resources.try_release_idle_dynamic_backing().unwrap(),
                IdleDynamicReleaseOutcome::Released(_)
            ));
        }
    }
    drop((executor, compiled));
    close_resources(resources);
    results
}

#[test]
#[ignore = "requires CUDA; actual direct F32 cuRAND, native oracle and managed resource retirement"]
fn managed_cuda_philox_direct_f32_has_native_mantissa_and_preserves_bf16_sequence() {
    oracle::known_answers();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 8,
            opaque_memory_profile: "unqualified-direct-f32-philox-fixture".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(philox_normal_bf16_contract().unwrap()),
            Box::new(philox_normal_f32_contract().unwrap()),
        ],
        cuda_vnext_tensor_providers(
            &runtime,
            &[
                CudaTensorOperation::PhiloxNormal,
                CudaTensorOperation::PhiloxNormalF32,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.philox.f32.fixture"),
        ContractVersion::new(1, 0),
        descriptor.runtime_implementation_fingerprint.clone(),
        descriptor.id.clone(),
        descriptor.capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(descriptor.clone(), vec![engine])
        .unwrap();
    let geometry = CudaPhiloxDeviceGeometry::query(&runtime).unwrap();
    let first_round = u64::from(geometry.multiprocessors())
        * u64::from(geometry.max_threads_per_multiprocessor() / 256)
        * 1024;
    let states = [
        PhiloxSeedOffset::new(7, 3),
        PhiloxSeedOffset::new(7 | (1 << 40), 3),
        PhiloxSeedOffset::new(7, (1 << 40) + 3),
        PhiloxSeedOffset::new(7, u64::MAX),
        PhiloxSeedOffset::new(7, 3),
    ];
    // Both launch tails and a full four-wide round followed by another round.
    // The latter size comes from actual device properties, not a machine ID.
    for count in [771, first_round + 5] {
        let launch = geometry.launch_f32(count).unwrap();
        if count > first_round {
            assert_eq!(launch.rounds(), 2);
        }
        let output_capacity = usize::try_from(count * 6 * states.len() as u64).unwrap();
        let host = HostMemoryAccount::new(
            HostMemoryLimits::new(
                output_capacity + 16,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 16,
                    output_bytes: output_capacity,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        );
        let f32_outputs = execute_states(
            &runtime,
            &registry,
            &catalog,
            &host,
            NormalNoisePrecision::F32,
            count,
            &states,
        );
        let bf16_outputs = execute_states(
            &runtime,
            &registry,
            &catalog,
            &host,
            NormalNoisePrecision::Bf16,
            count,
            &states,
        );
        assert_eq!(f32_outputs[0].bytes(), f32_outputs[4].bytes());
        for changed in 1..4 {
            assert_ne!(f32_outputs[0].bytes(), f32_outputs[changed].bytes());
        }
        for ((&state, actual), bf16) in states.iter().zip(&f32_outputs).zip(&bf16_outputs) {
            let values: Vec<_> = actual
                .bytes()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            oracle::check(state, launch.blocks(), &values);
            assert!(
                values.iter().any(|x| x.to_bits() & 0xffff != 0),
                "F32 output was truncated to BF16"
            );
            for (index, (&x, pair)) in values.iter().zip(bf16.bytes().chunks_exact(2)).enumerate() {
                assert_eq!(
                    half::bf16::from_f32(x).to_bits(),
                    u16::from_le_bytes(pair.try_into().unwrap()),
                    "BF16 sequence changed at {index}"
                );
            }
        }
        // All device plan owners have retired; host readback remains valid.
        assert_eq!(host.usage().unwrap().allocated_bytes(), output_capacity);
        let tail = f32_outputs[0]
            .slice(f32_outputs[0].len() - 4..f32_outputs[0].len())
            .unwrap();
        let expected = tail.bytes().to_vec();
        drop((f32_outputs, bf16_outputs));
        assert_eq!(tail.bytes(), expected);
        assert_eq!(
            host.usage().unwrap().allocated_bytes(),
            usize::try_from(count * 4).unwrap()
        );
        drop(tail);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    }
    drop((catalog, registry, runtime));
}
