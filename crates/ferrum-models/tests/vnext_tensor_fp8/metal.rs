use super::*;
use ferrum_execution::component::{
    CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
    TensorComponentPolicy,
};
use ferrum_execution::TensorExecutorConfig;
use ferrum_kernels::backend::metal::{
    vnext_ops::tensor::{
        metal_tensor_providers, metal_tensor_runtime_config, MetalTensorOperation,
    },
    vnext_runtime::{MetalDeviceRuntime, MetalTensorRuntimeConfig},
};

pub(super) fn verify() {
    let kind = MetalTensorOperation::Fp8Linear;
    assert_eq!(kind.profile(), METAL_PROFILE);
    let runtime = Arc::new(
        MetalDeviceRuntime::new_tensor_eager(MetalTensorRuntimeConfig {
            device: metal_tensor_runtime_config(id("device.metal.0")).unwrap(),
            usable_memory_bytes: 64 << 20,
            bootstrap_memory_bytes: 16 << 20,
            stream_memory_bytes: 1 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "metal.fp8.fixture.v1".into(),
        })
        .unwrap(),
    );
    let host = runtime
        .tensor_host_account(
            HostMemoryLimits::new(
                8 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 2 << 20,
                    output_bytes: 2 << 20,
                    transfer_staging_bytes: 2 << 20,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        )
        .unwrap();
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(kind.contract().unwrap())],
        metal_tensor_providers(&runtime, &[kind]).unwrap(),
    )
    .unwrap();
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.metal.fp8.fixture"),
        ContractVersion::new(1, 0),
        descriptor.runtime_implementation_fingerprint.clone(),
        descriptor.id.clone(),
        descriptor.capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(descriptor.clone(), vec![engine])
        .unwrap();
    let policy = |capacity| {
        TensorComponentPolicy::new(
            capacity,
            descriptor
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .unwrap()
    };
    // Full tiles, singleton, and tails on all axes; varying row scales/zero rows.
    for (case, shape) in [
        Shape { m: 1, n: 1, k: 1 },
        Shape { m: 3, n: 32, k: 16 },
        Shape {
            m: 33,
            n: 65,
            k: 31,
        },
        Shape {
            m: 129,
            n: 64,
            k: 128,
        },
        Shape { m: 3, n: 16, k: 16 }, // Identity weights expose quantization ties.
        Shape {
            m: 1,
            n: 128,
            k: 16,
        }, // All finite positive weight encodings.
        Shape {
            m: 1,
            n: 128,
            k: 16,
        }, // All finite negative weight encodings.
    ]
    .into_iter()
    .enumerate()
    {
        let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id(X),
            tensor(vec![shape.m, shape.k]),
        )]))
        .unwrap();
        let compile = |capacity| {
            CompiledTensorComponent::compile(
                TypedFamilyRegistration::new(family::Family)
                    .prepare_with_profile(
                        &serde_json::to_value(&shape).unwrap(),
                        &id(METAL_PROFILE),
                    )
                    .unwrap(),
                &registry,
                &catalog,
                &policy(capacity),
                &WeightMaterializerRegistry::identity_only().unwrap(),
                &options,
            )
        };
        let compiled = compile(64 << 20).unwrap();
        let plan = compiled.executable().execution_plan();
        let memory = plan.payload().memory();
        let minimum = memory.static_bytes() + memory.minimum_runnable_request_bytes();
        compile(minimum).unwrap();
        assert!(
            matches!(compile(minimum - 1), Err(VNextError::PlanCapacityExceeded {
            required_bytes, usable_capacity_bytes
        }) if required_bytes == minimum && usable_capacity_bytes == minimum - 1)
        );
        let scratch = plan.payload().nodes()[0]
            .provider_resources()
            .scratch()
            .unwrap()
            .fixed_bytes()
            .unwrap();
        assert!(scratch >= shape.m * shape.k + shape.m * 4);
        let mut payload = Payload::new(&shape);
        let mut inputs = input(&shape);
        if case == 4 {
            payload.weights.fill(0);
            payload.scale = 1_f32.to_le_bytes();
            for i in 0..16 {
                payload.weights[i * 16 + i] = 56;
            }
            for (dst, value) in inputs[..16].iter_mut().zip([
                448.,
                1.0625,
                1.1875,
                -1.0625,
                -1.1875,
                1. / 1024.,
                3. / 1024.,
                -1. / 1024.,
                15. / 1024.,
                -15. / 1024.,
                0.,
                -0.,
                0.015625,
                0.001953125,
                240.,
                -240.,
            ]) {
                *dst = bf16::from_f32(value);
            }
        } else if case >= 5 {
            payload.weights.fill(0);
            payload.scale = 1_f32.to_le_bytes();
            for n in 0..128 {
                payload.weights[n * 16] = (n.min(126) as u8) | if case == 6 { 128 } else { 0 };
            }
            inputs.fill(bf16::ZERO);
            inputs[0] = bf16::ONE;
            inputs[15] = bf16::from_f32(448.); // Row scale exactly one.
        }
        let expected = oracle::expected(
            &inputs,
            &payload.weights,
            payload.scale(),
            shape.k as usize,
            shape.n as usize,
        );
        let mut session = compiled
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::Stored(&payload),
                ComponentInitializationConfig {
                    request_id: id(format!("request.metal.fp8.init.{case}")),
                    run_id: id(format!("run.metal.fp8.init.{case}")),
                    transaction_id: id(format!("transaction.metal.fp8.init.{case}")),
                    static_policy: StaticInitializationPolicy::new(1 << 20, 16).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let mut previous = None;
        for repetition in 0..3 {
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, inputs.len() * 2)
                .unwrap();
            for (dst, value) in bytes.as_mut().chunks_exact_mut(2).zip(&inputs) {
                dst.copy_from_slice(&value.to_bits().to_le_bytes());
            }
            let poisoned_row = usize::from(shape.m > 1);
            if repetition == 1 {
                let offset = poisoned_row * shape.k as usize * 2;
                bytes.as_mut()[offset..offset + 2]
                    .copy_from_slice(&bf16::NAN.to_bits().to_le_bytes());
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
                    id(format!("run.metal.fp8.{case}.{repetition}")),
                    id(format!("request.metal.fp8.{case}.{repetition}")),
                    vec![input],
                )
                .unwrap();
            let bytes = outputs[0].bytes().clone();
            if repetition == 1 {
                for (row, data) in bytes.bytes().chunks_exact(shape.n as usize * 2).enumerate() {
                    for word in data.chunks_exact(2) {
                        let value = bf16::from_bits(u16::from_le_bytes(word.try_into().unwrap()));
                        assert_eq!(
                            value.is_nan(),
                            row == poisoned_row,
                            "nonfinite row leaked or was hidden"
                        );
                    }
                }
                continue;
            }
            expected.check(bytes.bytes(), shape.k as usize);
            if let Some(old) = previous {
                assert_eq!(bytes.bytes(), ManagedHostBytes::bytes(&old));
            }
            previous = Some(bytes);
        }
        assert!(matches!(
            session.try_release_idle_dynamic_backing().unwrap(),
            IdleDynamicReleaseOutcome::Released(_)
        ));
        match session
            .close()
            .unwrap_or_else(|f| panic!("Metal FP8 close: {:?}", f.failure()))
        {
            PlanRuntimeCloseOutcome::Closed(receipt) => {
                assert!(receipt.released_static_resources() > 0)
            }
            _ => panic!("retained Metal FP8 plan owner"),
        }
        drop(compiled);
        expected.check(previous.as_ref().unwrap().bytes(), shape.k as usize);
        drop(previous);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        eprintln!("Metal FP8 M={} N={} K={}: independent F64 oracle, admitted scratch={}, repeat and retirement passed",shape.m,shape.n,shape.k,scratch);
    }
}
