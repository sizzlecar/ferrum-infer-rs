use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use std::sync::Arc;
const DEVICE_BYTES: u64 = 2 << 30;

pub(super) fn verify() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: DEVICE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "native-block-linear-fixture-v1".into(),
        })
        .unwrap(),
    );
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            8 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 4 << 20,
                output_bytes: 2 << 20,
                transfer_staging_bytes: 2 << 20,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(dense_linear_block_bf16_contract().unwrap())],
        cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::BlockLinear]).unwrap(),
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.block.fixture"),
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
        for (case, (m, n, k, basis)) in
            [(512, 3, 512, true), (1, 7, 256, false), (17, 7, 512, false)]
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
            let inputs = input(&shape, basis);
            let options =
                ProgramPlanCompileOptions::new(BTreeMap::from([(id(X), tensor(vec![m, k]))]))
                    .unwrap();
            let compile = |capacity| {
                CompiledTensorComponent::compile(
                    family::prepare(&shape),
                    &registry,
                    &catalog,
                    &policy(capacity),
                    &materializers,
                    &options,
                )
            };
            let compiled = compile(DEVICE_BYTES).unwrap();
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
                "only the active decoded projection is scratch"
            );
            assert!(memory.minimum_runnable_request_bytes() >= scratch);
            compile(minimum).unwrap();
            assert!(
                matches!(compile(minimum - 1), Err(VNextError::PlanCapacityExceeded {
                required_bytes, usable_capacity_bytes
            }) if required_bytes == minimum && usable_capacity_bytes == minimum - 1)
            );
            eprintln!("native {format:?} M={m} N={n} K={k}: scratch={scratch}, minimum_plan_bytes={minimum}, hash={}",
                compiled.executable().execution_plan().plan_hash());
            let prefix = format!("block.{format_index}.{case}");
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
            let mut retained: Option<ManagedHostBytes> = None;
            for repetition in 0..2 {
                let mut bytes = host
                    .allocate_zeroed(HostMemoryClass::InputPreprocess, inputs.len() * 2)
                    .unwrap();
                for (bytes, value) in bytes.as_mut().chunks_exact_mut(2).zip(&inputs) {
                    bytes.copy_from_slice(&value.to_le_bytes());
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
                let outputs = session
                    .execute(
                        id(format!("run.{prefix}.{repetition}")),
                        id(format!("request.{prefix}.{repetition}")),
                        vec![input],
                    )
                    .unwrap();
                assert_eq!(outputs.len(), 1);
                check(
                    &shape,
                    &inputs,
                    &fixture.decoded,
                    outputs[0].bytes().bytes(),
                    basis,
                );
                if let Some(previous) = &retained {
                    assert_eq!(outputs[0].bytes().bytes(), previous.bytes());
                }
                retained = Some(outputs[0].bytes().clone());
            }
            assert!(matches!(
                session.try_release_idle_dynamic_backing().unwrap(),
                IdleDynamicReleaseOutcome::Released(_)
            ));
            match session
                .close()
                .unwrap_or_else(|failure| panic!("close failed: {:?}", failure.failure()))
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
                &inputs,
                &fixture.decoded,
                retained.as_ref().unwrap().bytes(),
                basis,
            );
            drop(retained);
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        }
    }
}
