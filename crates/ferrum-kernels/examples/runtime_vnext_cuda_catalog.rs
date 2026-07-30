use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use ferrum_interfaces::vnext::DeviceId;
use ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition;
use ferrum_types::{AttentionExecutionPolicy, NativeOperatorBackend};
use serde::Serialize;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    if arguments.len() != 5 {
        return Err(format!(
            "usage: {} <cuda-ordinal> <attention-policy> <provider-catalog-out> <capability-catalog-out>",
            arguments.first().map(String::as_str).unwrap_or("runtime_vnext_cuda_catalog")
        )
        .into());
    }
    let ordinal = arguments[1].parse::<usize>()?;
    let policy = AttentionExecutionPolicy::parse_runtime_value(&arguments[2])?;
    let provider_catalog_path = Path::new(&arguments[3]);
    let capability_catalog_path = Path::new(&arguments[4]);
    let composition =
        CudaVNextComposition::create(ordinal, DeviceId::new(format!("cuda:{ordinal}"))?, policy)?;
    let capability_catalog = composition.catalog();
    let provider_catalog =
        capability_catalog.native_operator_provider_catalog(NativeOperatorBackend::Cuda)?;

    write_json_create_new(provider_catalog_path, &provider_catalog)?;
    write_json_create_new(capability_catalog_path, capability_catalog)?;
    println!(
        "FERRUM RUNTIME VNEXT CUDA LIVE CATALOG READY: provider={} capability={} capability_fingerprint={}",
        provider_catalog_path.display(),
        capability_catalog_path.display(),
        capability_catalog.fingerprint()?
    );
    Ok(())
}

fn write_json_create_new(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        return Err(format!("catalog output already exists: {}", path.display()).into());
    }
    let parent = path
        .parent()
        .ok_or_else(|| format!("catalog output has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("catalog"),
        std::process::id()
    ));
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path)?;
    Ok(())
}
