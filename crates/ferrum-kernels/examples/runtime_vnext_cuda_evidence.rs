//! Actual context/identity/sampler smoke, without a model or inference workload.
use std::error::Error;

use ferrum_interfaces::vnext::{DeviceCostHardwareIdentityAvailability, DeviceId, DeviceRuntime};
use ferrum_kernels::backend::cuda::{
    vnext_ops::cuda_vnext_runtime_config, vnext_runtime::CudaDeviceRuntime,
};
use ferrum_types::{AttentionExecutionPolicy, DeviceMemorySamplingConfig};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args().collect();
    if arguments.len() != 3 {
        return Err("usage: runtime_vnext_cuda_evidence <cuda-ordinal> <new-memory-jsonl>".into());
    }
    let ordinal = arguments[1].parse()?;
    let config = cuda_vnext_runtime_config(
        ordinal,
        DeviceId::new(format!("cuda:{ordinal}"))?,
        AttentionExecutionPolicy::Portable,
    )?;
    let mut runtime = CudaDeviceRuntime::new(config)?;
    runtime.enable_device_memory_sampling(&DeviceMemorySamplingConfig {
        jsonl_path: arguments[2].clone().into(),
    })?;
    let identity = runtime.cost_hardware_identity();
    let repeated = runtime.cost_hardware_identity();
    if identity != repeated {
        return Err("cached CUDA cost identity changed within one runtime".into());
    }
    let known = matches!(identity, DeviceCostHardwareIdentityAvailability::Known(_));
    runtime.finish_device_memory_sampling()?;
    let memory = runtime
        .device_memory_snapshot()
        .ok_or("CUDA sampler missing")?;
    if !memory.complete || memory.error_count != 0 {
        return Err("CUDA memory capture incomplete".into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "hardware_identity_known": known,
            "hardware_identity": format!("{identity:?}"),
            "graph_capture_capability": format!("{:?}", runtime.cost_graph_capture_capability()),
            "core_execution_capabilities": format!("{:?}", runtime.cost_core_execution_capabilities()),
            "runtime_implementation_fingerprint": runtime.descriptor().runtime_implementation_fingerprint,
            "memory": memory,
            "model_loaded": false,
            "inference_or_performance_validation": false,
        }))?
    );
    if !known {
        return Err("actual CUDA identity remained unavailable; see reported reason".into());
    }
    Ok(())
}
