#![cfg(feature = "cuda")]
#[path = "support/norm_rope_f32_component.rs"]
mod checks;
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use fixture::*;
use std::sync::{Arc, Mutex};
static GPU: Mutex<()> = Mutex::new(());
fn run(kind: Kind) {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "unqualified-f32-norm-rope-fixture-v1".into(),
        })
        .unwrap(),
    );
    let (contract, operation) = if matches!(kind, Kind::HeadRmsF32) {
        (
            head_rms_norm_f32_contract().unwrap(),
            CudaTensorOperation::HeadRmsNormF32,
        )
    } else if matches!(kind, Kind::UnitHeadRmsPartialRopeF32) {
        (
            unit_head_rms_partial_rope_f32_contract().unwrap(),
            CudaTensorOperation::UnitHeadRmsPartialRopeF32,
        )
    } else {
        (
            half_split_rope_f32_contract().unwrap(),
            CudaTensorOperation::HalfSplitRopeF32,
        )
    };
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(contract)],
        cuda_vnext_tensor_providers(&runtime, &[operation]).unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.norm_rope_f32"),
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
    checks::run(runtime, &registry, &catalog, &materializers, kind);
}
#[test]
#[ignore = "requires configured CUDA host; real F32 gamma/RMS on D=4/128/2560"]
fn managed_f32_head_rms_handles_non_power_of_two_hidden_width() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    run(Kind::HeadRmsF32);
}
#[test]
#[ignore = "requires configured CUDA host; real F32 RoPE with explicit noncontracted products"]
fn managed_f32_half_split_rope_retains_f32_boundaries() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    run(Kind::HalfSplitRopeF32);
}
#[test]
#[ignore = "requires CUDA; full-head unit RMS with a normalized unrotated tail"]
fn managed_unit_rms_partial_rope_keeps_f32_prefix_and_normalized_tail() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    run(Kind::UnitHeadRmsPartialRopeF32);
}
