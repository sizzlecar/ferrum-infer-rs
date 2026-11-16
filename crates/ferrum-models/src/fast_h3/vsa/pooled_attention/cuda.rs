use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use std::sync::Arc;
#[test]
#[ignore = "requires CUDA; independent FP64 softmax/PV oracle, exact dyadic scores, managed retirement"]
fn pooled_attention_cuda_materializes_shared_scaled_scores_and_weighted_values() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0").unwrap()).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 16,
            opaque_memory_profile: "pooled-attention-qualification".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        BatchedF32Op::ALL
            .into_iter()
            .map(|op| Box::new(op.contract().unwrap()) as Box<dyn OperationContract>)
            .collect(),
        cuda_vnext_tensor_providers(
            &runtime,
            &[
                CudaTensorOperation::BatchedMatmulF32,
                CudaTensorOperation::ScaleRank4F32,
                CudaTensorOperation::RowSoftmaxRank4F32,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.pooled_attention").unwrap(),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    tests::qualify(runtime, &registry, &catalog, &materializers);
}
