use super::*;
use ferrum_execution::component::{
    CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
    TensorComponentPolicy,
};
use ferrum_execution::TensorExecutorConfig;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config_with_cutlass_fp8,
        cuda_vnext_dense_linear_fp8_bf16_probe_provider, cuda_vnext_dense_linear_fp8_bf16_provider,
        CudaCutlassFp8Config, CudaCutlassFp8Probe, CUDA_FP8_QUANTIZATION_PROBE_PROFILE_ID,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};

const DEVICE_BYTES: u64 = 2 << 30;

pub(super) fn verify() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config_with_cutlass_fp8(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: DEVICE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "experimental-fp8-fixture-v1".into(),
        })
        .unwrap(),
    );
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            4 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                output_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let probe = CudaCutlassFp8Probe::new(host.clone());
    let native = CudaCutlassFp8Config {
        opaque_memory_bytes: 64 << 20,
        maximum_plans: 16,
        opaque_memory_profile: "experimental-fp8-native-fixture-v1".into(),
    };
    let provider =
        cuda_vnext_dense_linear_fp8_bf16_probe_provider(&runtime, &native, probe.clone()).unwrap();
    assert!(
        cuda_vnext_dense_linear_fp8_bf16_provider(&runtime, &native).is_err(),
        "one independently budgeted live provider cache per context"
    );
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(dense_linear_fp8_bf16_contract().unwrap())],
        vec![provider],
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.fp8.fixture"),
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
    let policy = TensorComponentPolicy::new(
        DEVICE_BYTES,
        runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    // Small M and M beyond one tile, unequal K/N, zero and tiny rows are semantic
    // probes. These shapes are not a hardware certification matrix.
    for (case, shape) in [
        Shape { m: 1, n: 32, k: 16 },
        Shape { m: 3, n: 32, k: 16 },
        Shape {
            m: 17,
            n: 16,
            k: 64,
        },
        Shape {
            m: 129,
            n: 64,
            k: 128,
        },
    ]
    .into_iter()
    .enumerate()
    {
        let payload = Payload::new(&shape);
        let inputs = input(&shape);
        let expected = oracle::expected(
            &inputs,
            &payload.weights,
            payload.scale(),
            shape.k as usize,
            shape.n as usize,
        );
        let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id(X),
            tensor(vec![shape.m, shape.k]),
        )]))
        .unwrap();
        let compiled = CompiledTensorComponent::compile(
            family::prepare(&shape),
            &registry,
            &catalog,
            &policy,
            &WeightMaterializerRegistry::identity_only().unwrap(),
            &options,
        )
        .unwrap();
        let plan = compiled.executable().execution_plan();
        let scratch = plan.payload().nodes()[0]
            .provider_resources()
            .scratch()
            .unwrap()
            .fixed_bytes()
            .unwrap();
        assert!(
            scratch >= shape.m * shape.k + shape.m * 4,
            "quantized activation and row scales must be charged"
        );
        eprintln!(
            "FP8 case{case} M={} K={} N={}: scratch={scratch}, static={}, plan={}",
            shape.m,
            shape.k,
            shape.n,
            plan.payload().memory().static_bytes(),
            plan.plan_hash()
        );
        let mut session = compiled
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::Stored(&payload),
                ComponentInitializationConfig {
                    request_id: id(format!("request.fp8.init.{case}")),
                    run_id: id(format!("run.fp8.init.{case}")),
                    transaction_id: id(format!("transaction.fp8.init.{case}")),
                    static_policy: StaticInitializationPolicy::new(1 << 20, 16).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let mut retained = None;
        for repetition in 0..2 {
            // Both capture destinations are charged to this same Host parent
            // before any device admission. The ordinary provider has no probe.
            let ticket = probe.arm(shape.m as u32, shape.k as u32).unwrap();
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, inputs.len() * 2)
                .unwrap();
            for (dst, value) in bytes.as_mut().chunks_exact_mut(2).zip(&inputs) {
                dst.copy_from_slice(&value.to_bits().to_le_bytes());
            }
            let input = ManagedHostTensor::new(
                id(X),
                ResolvedTensorSpec::new(
                    vec![shape.m, shape.k],
                    ElementType::Bf16,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                bytes.freeze(),
            )
            .unwrap();
            let outputs = session
                .execute(
                    id(format!("run.fp8.{case}.{repetition}")),
                    id(format!("request.fp8.{case}.{repetition}")),
                    vec![input],
                )
                .unwrap();
            assert_eq!(outputs.len(), 1);
            let quantized = ticket.take().unwrap();
            assert_eq!(
                (quantized.rows(), quantized.inputs()),
                (shape.m as u32, shape.k as u32)
            );
            assert!(
                host.owns(quantized.activation_bytes()) && host.owns(quantized.row_scale_bytes())
            );
            expected
                .check_quantization(
                    quantized.activation_bytes().bytes(),
                    quantized.row_scale_bytes().bytes(),
                )
                .unwrap();
            eprintln!("FP8 case{case} repeat{repetition}: actual quantization capture matched {} E4M3FN bytes and {} F32 scale bits; diagnostic={CUDA_FP8_QUANTIZATION_PROBE_PROFILE_ID}", expected.activation_bytes.len(), expected.row_scales.len());
            let bytes = outputs[0].bytes().clone();
            let (gemm_error, quantization_change) = expected.check(bytes.bytes(), shape.k as usize);
            eprintln!("FP8 case{case} repeat{repetition}: decoded-F64 GEMM max error={gemm_error}; activation quantization max dot change={quantization_change}; finite probe only");
            if let Some(previous) = &retained {
                assert_eq!(bytes.bytes(), ManagedHostBytes::bytes(previous));
            }
            drop(outputs);
            drop((quantized, ticket));
            retained = Some(bytes);
        }
        assert!(matches!(
            session.try_release_idle_dynamic_backing().unwrap(),
            IdleDynamicReleaseOutcome::Released(_)
        ));
        match session
            .close()
            .unwrap_or_else(|failure| panic!("FP8 session close failed: {:?}", failure.failure()))
        {
            PlanRuntimeCloseOutcome::Closed(receipt) => {
                assert!(receipt.released_static_resources() > 0)
            }
            PlanRuntimeCloseOutcome::Referenced { strong_count, .. } => {
                panic!("unexpected {strong_count} owners")
            }
        }
        drop(compiled);
        expected.check(retained.as_ref().unwrap().bytes(), shape.k as usize);
        drop(retained);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    drop(registry);
    // Policy changes cannot repurpose the already installed module allowance.
    let changed = CudaCutlassFp8Config {
        opaque_memory_bytes: 32 << 20,
        ..native.clone()
    };
    assert!(cuda_vnext_dense_linear_fp8_bf16_provider(&runtime, &changed).is_err());
    drop(cuda_vnext_dense_linear_fp8_bf16_provider(&runtime, &native).unwrap());
    drop(runtime);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
