#![cfg(feature = "cuda")]
//! Managed CUDA execution of explicit F32 pointwise boundaries. The finite
//! fixture tolerance qualifies these operands and does not certify Torch math.
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
#[path = "support/pointwise_f32_component.rs"]
mod pointwise;
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use fixture::{host, id, Kind};
use std::sync::Arc;

#[test]
#[ignore = "requires configured CUDA host with admitted F32 pointwise fixture capacity"]
fn ordinary_f32_pointwise_preserves_ieee_boundaries_and_retires_real_cuda_work() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "pointwise-f32-fixture-unqualified-capacity".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(plain_residual_add_f32_contract().unwrap()),
            Box::new(silu_mul_f32_contract().unwrap()),
        ],
        cuda_vnext_tensor_providers(
            &runtime,
            &[
                CudaTensorOperation::PlainResidualAddF32,
                CudaTensorOperation::SiluMulF32,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.pointwise_f32.fixture"),
        ContractVersion::new(1, 0),
        runtime
            .descriptor()
            .runtime_implementation_fingerprint
            .clone(),
        runtime.descriptor().id.clone(),
        runtime.descriptor().capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(runtime.descriptor().clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    for kind in [Kind::F32ResidualAdd, Kind::F32SiluMul] {
        let component =
            pointwise::config(kind).compile(runtime.as_ref(), &registry, &catalog, &materializers);
        let host = host();
        let mut session = component
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id("run.cuda.pointwise_f32.init"),
                    request_id: id("request.cuda.pointwise_f32.init"),
                    transaction_id: id("transaction.cuda.pointwise_f32.init"),
                    static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        for trial in 0..2 {
            drop(pointwise::execute(&mut session, &host, kind, trial));
        }
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("CUDA pointwise close: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
