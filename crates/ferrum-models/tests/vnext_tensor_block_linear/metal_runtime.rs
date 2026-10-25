use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::metal::{
    vnext_ops::tensor::{
        metal_tensor_providers, metal_tensor_runtime_config, MetalTensorOperation,
    },
    vnext_runtime::{MetalDeviceRuntime, MetalTensorRuntimeConfig},
};
use std::sync::Arc;

pub(super) fn verify() {
    let capacity = 64 << 20;
    let operation = MetalTensorOperation::BlockLinear;
    let runtime = Arc::new(
        MetalDeviceRuntime::new_tensor_eager(MetalTensorRuntimeConfig {
            device: metal_tensor_runtime_config(id("device.metal.0")).unwrap(),
            usable_memory_bytes: capacity,
            bootstrap_memory_bytes: 16 << 20,
            stream_memory_bytes: 1 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "metal.block-linear.test.declared.v1".into(),
        })
        .unwrap(),
    );
    let host = runtime
        .tensor_host_account(
            HostMemoryLimits::new(
                8 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 4 << 20,
                    output_bytes: 2 << 20,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        )
        .unwrap();
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(operation.contract().unwrap())],
        metal_tensor_providers(&runtime, &[operation]).unwrap(),
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.metal.block.fixture"),
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
    let policy = |bytes| {
        TensorComponentPolicy::new(
            bytes,
            runtime
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .unwrap()
    };
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    for (format_index, format) in FORMATS.into_iter().enumerate() {
        for (case, (m, n, k, basis)) in [
            (512, 3, 512, true),
            (1, 7, 256, false),
            (33, 65, 512, false),
            (32, 64, 512, false),
            (256, 64, 256, true),
        ]
        .into_iter()
        .enumerate()
        {
            let shape = Shape {
                m,
                n,
                k,
                blocks: format.blocks(),
            };
            let fixture = Fixture::new(&shape, format);
            let values = input(&shape, basis);
            let options =
                ProgramPlanCompileOptions::new(BTreeMap::from([(id(X), tensor(vec![m, k]))]))
                    .unwrap();
            let compile = |bytes| {
                CompiledTensorComponent::compile(
                    family::prepare_with_profile(&shape, operation.profile()),
                    &registry,
                    &catalog,
                    &policy(bytes),
                    &materializers,
                    &options,
                )
            };
            let compiled = compile(capacity).unwrap();
            let memory = compiled.executable().execution_plan().payload().memory();
            let minimum = memory.static_bytes() + memory.minimum_runnable_request_bytes();
            let scratch = compiled.executable().execution_plan().payload().nodes()[0]
                .provider_resources()
                .scratch()
                .unwrap()
                .fixed_bytes()
                .unwrap();
            assert_eq!(
                scratch,
                n * k * 2,
                "only the current decoded matrix is invocation scratch"
            );
            compile(minimum).unwrap();
            assert!(
                matches!(compile(minimum - 1), Err(VNextError::PlanCapacityExceeded {
                required_bytes, usable_capacity_bytes
            }) if required_bytes == minimum && usable_capacity_bytes == minimum - 1)
            );
            let prefix = format!("metal.block.{format_index}.{case}");
            let mut session = compiled
                .initialize(
                    runtime.clone(),
                    host.clone(),
                    ComponentWeightSource::Stored(&fixture.source),
                    ComponentInitializationConfig {
                        request_id: id(format!("request.init.{prefix}")),
                        run_id: id(format!("run.init.{prefix}")),
                        transaction_id: id(format!("transaction.init.{prefix}")),
                        static_policy: StaticInitializationPolicy::new(1 << 20, 16).unwrap(),
                        executor: TensorExecutorConfig::default(),
                    },
                )
                .unwrap();
            let mut previous: Option<ManagedHostBytes> = None;
            for repetition in 0..2 {
                let mut bytes = host
                    .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 2)
                    .unwrap();
                for (dst, value) in bytes.as_mut().chunks_exact_mut(2).zip(&values) {
                    dst.copy_from_slice(&value.to_le_bytes());
                }
                let input = ManagedHostTensor::new(
                    id(X),
                    ResolvedTensorSpec::new(
                        vec![m, k],
                        ElementType::Bf16,
                        ResolvedTensorLayout::Contiguous,
                    )
                    .unwrap(),
                    bytes.freeze(),
                )
                .unwrap();
                let output = session
                    .execute(
                        id(format!("run.{prefix}.{repetition}")),
                        id(format!("request.{prefix}.{repetition}")),
                        vec![input],
                    )
                    .unwrap();
                assert_eq!(output.len(), 1);
                check(
                    &shape,
                    &values,
                    &fixture.decoded,
                    output[0].bytes().bytes(),
                    basis,
                );
                if let Some(previous) = &previous {
                    assert_eq!(output[0].bytes().bytes(), previous.bytes());
                }
                previous = Some(output[0].bytes().clone());
            }
            assert!(matches!(
                session.try_release_idle_dynamic_backing().unwrap(),
                IdleDynamicReleaseOutcome::Released(_)
            ));
            match session
                .close()
                .unwrap_or_else(|e| panic!("close: {:?}", e.failure()))
            {
                PlanRuntimeCloseOutcome::Closed(receipt) => {
                    assert!(receipt.released_static_resources() > 0)
                }
                PlanRuntimeCloseOutcome::Referenced { strong_count, .. } => {
                    panic!("{strong_count} remaining owners")
                }
            }
            drop(compiled);
            check(
                &shape,
                &values,
                &fixture.decoded,
                previous.as_ref().unwrap().bytes(),
                basis,
            );
            drop(previous);
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
            eprintln!("Metal {format:?} M={m} N={n} K={k} basis={basis}: oracle, repeat, scratch={scratch}, minimum={minimum}, retirement verified");
        }
    }
}
