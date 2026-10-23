//! Cold execution identity: actual executor, immutable code and physical facts.

use ferrum_execution::ModelExecutor;
use ferrum_slo::model::ExecutionIdentity;
use ferrum_types::{Device, EngineConfig, FerrumError, Result};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::File, io::Read, sync::OnceLock};

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "scope")]
pub(super) enum BindingScope {
    Reusable,
    InstanceScoped { reason: String },
}

pub(super) struct BoundIdentity {
    pub identity: ExecutionIdentity,
    pub scope: BindingScope,
}

pub(super) fn bind(config: &EngineConfig, executor: &dyn ModelExecutor) -> Result<BoundIdentity> {
    // Unreported custom executors cannot become reusable by probing an
    // unrelated device that merely has the same requested enum/ordinal.
    if executor.resolved_model_plan().is_none() && executor.implementation_identity().is_none() {
        return bind_with_evidence(
            config,
            executor,
            Err("executor did not report its concrete implementation identity".into()),
            Err("implementation is instance scoped".into()),
        );
    }
    if executor.bound_resolved_sources().is_none() {
        return bind_with_evidence(
            config,
            executor,
            Err("executor did not report immutable model sources".into()),
            Err("model sources are instance scoped".into()),
        );
    }
    let physical = if config
        .backend
        .backend_options
        .get("selected_gpu_devices")
        .and_then(Value::as_array)
        .is_some_and(|devices| devices.len() > 1)
    {
        Err("stable multi-device execution identity is not yet qualified".into())
    } else {
        physical_identity(&executor.info().device)
    };
    let executable = executable_fingerprint();
    bind_with_evidence(config, executor, physical, executable)
}

fn bind_with_evidence(
    config: &EngineConfig,
    executor: &dyn ModelExecutor,
    physical: std::result::Result<Value, String>,
    executable: std::result::Result<String, String>,
) -> Result<BoundIdentity> {
    let implementation = executor.implementation_identity();
    let plan = executor.resolved_model_plan();
    let problem = physical
        .as_ref()
        .err()
        .cloned()
        .or_else(|| executable.as_ref().err().cloned())
        .or_else(|| {
            (implementation.is_none() && plan.is_none())
                .then(|| "executor did not report its concrete implementation identity".into())
        })
        .or_else(|| {
            executor
                .bound_resolved_sources()
                .is_none()
                .then(|| "executor did not report immutable model sources".into())
        });
    let scope = match problem {
        Some(reason) => BindingScope::InstanceScoped { reason },
        None => BindingScope::Reusable,
    };
    // A new assembly receives a new nonce if any proof is unavailable. The
    // product also reports SkippedUnbound and does not read/write a state file.
    let instance = matches!(scope, BindingScope::InstanceScoped { .. }).then(uuid::Uuid::new_v4);
    let mut identity = json!({
        "schema": 2,
        "binding": scope,
        "instance": instance,
        "physical": physical.ok(),
        "executable_sha256": executable.ok(),
        "native_artifacts": crate::commands::serve::compiled_kernel_features(),
        "implementation": implementation,
        "planning": executor.execution_planning_model(),
        "sources": executor.bound_resolved_sources(),
        "actual_model": executor.info(),
        "resolved_plan": plan.map(|plan| json!({
            "fingerprint": plan.fingerprint(),
            "device": plan.parts().device,
            "numerical_execution": plan.parts().numerical_execution,
            "engine": plan.parts().engine,
        })),
        "backend": config.backend,
        "numerical_execution": config.numerical_execution,
        "runtime": effective_runtime(config)?,
        "batching": config.batching,
        "kv_cache": config.kv_cache,
    });
    canonicalize(&mut identity);
    Ok(BoundIdentity {
        identity: ExecutionIdentity::new(
            serde_json::to_string(&identity)
                .map_err(|error| FerrumError::serialization(error.to_string()))?,
        )
        .map_err(|error| FerrumError::config(error.to_string()))?,
        scope,
    })
}

fn effective_runtime(config: &EngineConfig) -> Result<Value> {
    let mut value = serde_json::to_value(&config.runtime)
        .map_err(|error| FerrumError::serialization(error.to_string()))?;
    let fields = value.as_object_mut().expect("typed runtime object");
    // Sampled free memory and the requested search bounds are construction
    // evidence. The owner-selected limits and effective capacity are binding.
    fields.remove("startup_memory_request");
    fields.remove("startup_memory_plan");
    fields.insert(
        "selected_resource_limits".into(),
        json!(config
            .runtime
            .startup_memory_plan
            .as_ref()
            .map(|plan| plan.selected)),
    );
    fields.insert(
        "usable_capacity_bytes".into(),
        json!(config.memory.usable_capacity_bytes),
    );
    for key in [
        "profile_jsonl",
        "scheduler_trace_jsonl",
        "legacy_scheduler_trace_jsonl",
    ] {
        let enabled = fields.get(key).is_some_and(|value| !value.is_null());
        fields.insert(key.into(), Value::Bool(enabled));
    }
    Ok(value)
}

fn canonicalize(value: &mut Value) {
    match value {
        Value::Object(object) => {
            let mut entries: Vec<_> = std::mem::take(object).into_iter().collect();
            entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            for (key, mut value) in entries {
                canonicalize(&mut value);
                object.insert(key, value);
            }
        }
        Value::Array(array) => array.iter_mut().for_each(canonicalize),
        _ => {}
    }
}

fn executable_fingerprint() -> std::result::Result<String, String> {
    static FINGERPRINT: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    FINGERPRINT
        .get_or_init(|| {
            let path = std::env::current_exe().map_err(|e| format!("executable identity: {e}"))?;
            let mut file = File::open(path).map_err(|e| format!("executable identity: {e}"))?;
            let before = file.metadata().map_err(|e| e.to_string())?;
            if before.len() == 0 || before.len() > 2 * 1024 * 1024 * 1024 {
                return Err("executable identity exceeds its 2 GiB cold-read budget".into());
            }
            let mut hash = Sha256::new();
            let mut buffer = [0_u8; 64 * 1024];
            let mut bytes = 0_u64;
            loop {
                let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
                if count == 0 {
                    break;
                }
                bytes += count as u64;
                if bytes > before.len() {
                    return Err("executable changed during identity read".into());
                }
                hash.update(&buffer[..count]);
            }
            let after = file.metadata().map_err(|e| e.to_string())?;
            if bytes != before.len()
                || after.len() != before.len()
                || after.modified().ok() != before.modified().ok()
            {
                return Err("executable changed during identity read".into());
            }
            Ok(format!("{:x}", hash.finalize()))
        })
        .clone()
}

fn physical_identity(device: &Device) -> std::result::Result<Value, String> {
    let host = host_identity()?;
    let accelerator = match device {
        Device::CPU => Value::Null,
        #[cfg(feature = "cuda")]
        Device::CUDA(ordinal) => serde_json::to_value(
            ferrum_kernels::backend::probe_cuda_execution_device_identity(*ordinal as usize)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?,
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        Device::Metal => {
            let device = ferrum_kernels::backend::probe_metal_execution_device_identity();
            if device.registry_id == 0
                || device.name.is_empty()
                || device.recommended_working_set_bytes == 0
            {
                return Err("Metal physical identity is incomplete".into());
            }
            serde_json::to_value(device).map_err(|error| error.to_string())?
        }
        other => {
            return Err(format!(
                "physical identity unavailable for {other:?} in this build"
            ))
        }
    };
    Ok(json!({"host": host, "device": device, "accelerator": accelerator}))
}

fn host_identity() -> std::result::Result<Value, String> {
    let parallelism = std::thread::available_parallelism()
        .map_err(|e| e.to_string())?
        .get();
    Ok(json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "available_parallelism": parallelism,
        "facts": host_facts()?,
    }))
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn host_facts() -> std::result::Result<BTreeMap<String, String>, String> {
    let mut facts = BTreeMap::new();
    for name in [
        "hw.model",
        "machdep.cpu.brand_string",
        "hw.memsize",
        "hw.physicalcpu",
        "hw.logicalcpu",
        "kern.osrelease",
        "kern.osversion",
    ] {
        let key = std::ffi::CString::new(name).expect("constant sysctl key");
        let mut bytes = [0_u8; 4096];
        let mut length = bytes.len();
        // Fixed initialized output storage; sysctl writes at most length bytes.
        let result = unsafe {
            libc::sysctlbyname(
                key.as_ptr(),
                bytes.as_mut_ptr().cast(),
                &mut length,
                std::ptr::null_mut(),
                0,
            )
        };
        if result != 0 || length == 0 || length > bytes.len() {
            return Err(format!("required host identity fact {name} unavailable"));
        }
        // Canonical raw bytes retain numeric widths and avoid lossy decoding.
        let encoded: String = bytes[..length].iter().map(|b| format!("{b:02x}")).collect();
        facts.insert(name.into(), encoded);
    }
    Ok(facts)
}

#[cfg(target_os = "linux")]
fn host_facts() -> std::result::Result<BTreeMap<String, String>, String> {
    use std::collections::BTreeSet;
    fn read(path: &str, max: usize) -> std::result::Result<String, String> {
        let mut bytes = Vec::with_capacity(max + 1);
        File::open(path)
            .map_err(|e| e.to_string())?
            .take(max as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() > max {
            return Err(format!("host identity fact {path} exceeds budget"));
        }
        String::from_utf8(bytes).map_err(|e| e.to_string())
    }
    let cpu = read("/proc/cpuinfo", 256 * 1024)?;
    let mut attributes: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for line in cpu.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        if matches!(
            key,
            "vendor_id"
                | "model name"
                | "cpu family"
                | "model"
                | "stepping"
                | "flags"
                | "Features"
                | "CPU implementer"
                | "CPU architecture"
                | "CPU variant"
                | "CPU part"
                | "CPU revision"
                | "Hardware"
        ) {
            attributes.entry(key).or_default().insert(value.trim());
        }
    }
    if !attributes.contains_key("model name") && !attributes.contains_key("CPU part") {
        return Err("host CPU identity is incomplete".into());
    }
    let memory = read("/proc/meminfo", 16 * 1024)?;
    let total = memory
        .lines()
        .find(|line| line.starts_with("MemTotal:"))
        .ok_or_else(|| "host memory identity is incomplete".to_string())?;
    let mut facts = BTreeMap::from([
        (
            "cpu".into(),
            serde_json::to_string(&attributes).map_err(|e| e.to_string())?,
        ),
        ("memory".into(), total.to_owned()),
        (
            "kernel".into(),
            read("/proc/sys/kernel/osrelease", 4096)?.trim().to_owned(),
        ),
    ]);
    // Record the exact installed NVIDIA driver build when this host exposes it.
    if let Ok(version) = read("/proc/driver/nvidia/version", 4096) {
        facts.insert("nvidia_driver".into(), version);
    }
    Ok(facts)
}

#[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "linux")))]
fn host_facts() -> std::result::Result<BTreeMap<String, String>, String> {
    Err("stable host identity is not implemented on this platform".into())
}

#[cfg(test)]
mod tests;
