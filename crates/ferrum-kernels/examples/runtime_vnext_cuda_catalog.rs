use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use ferrum_interfaces::vnext::DeviceId;
use ferrum_kernels::backend::cuda::vnext_ops::cuda_validated_native_operator_catalog_input_with_causal_decode_preparation_mode;
use ferrum_types::{AttentionExecutionPolicy, CausalDecodePreparationMode, NativeOperatorBackend};
use serde::Serialize;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    if !matches!(arguments.len(), 6 | 7) {
        return Err(format!(
            "usage: {} <cuda-ordinal> <attention-policy> <provider-catalog-out> <capability-catalog-out> <compiled-native-operators-out> [causal-decode-preparation-mode]",
            arguments.first().map(String::as_str).unwrap_or("runtime_vnext_cuda_catalog")
        )
        .into());
    }
    let ordinal = arguments[1].parse::<usize>()?;
    let policy = AttentionExecutionPolicy::parse_runtime_value(&arguments[2])?;
    let provider_catalog_path = Path::new(&arguments[3]);
    let capability_catalog_path = Path::new(&arguments[4]);
    let compiled_native_operators_path = Path::new(&arguments[5]);
    let causal_decode_preparation = arguments
        .get(6)
        .map(|value| CausalDecodePreparationMode::parse_runtime_value(value))
        .transpose()?
        .unwrap_or_default();
    let catalog_input =
        cuda_validated_native_operator_catalog_input_with_causal_decode_preparation_mode(
            ordinal,
            DeviceId::new(format!("cuda:{ordinal}"))?,
            policy,
            causal_decode_preparation,
        )?;
    let capability_catalog = catalog_input.capability_catalog();
    let provider_catalog =
        capability_catalog.native_operator_provider_catalog(NativeOperatorBackend::Cuda)?;
    let compiled_native_operators =
        ferrum_kernels::native_ops::compiled_native_operator_artifacts();

    write_json_create_new(provider_catalog_path, &provider_catalog)?;
    write_json_create_new(capability_catalog_path, capability_catalog)?;
    write_json_create_new(compiled_native_operators_path, &compiled_native_operators)?;
    println!(
        "FERRUM RUNTIME VNEXT CUDA LIVE CATALOG READY: provider={} capability={} compiled_native_operators={} compiled_native_operator_count={} capability_fingerprint={}",
        provider_catalog_path.display(),
        capability_catalog_path.display(),
        compiled_native_operators_path.display(),
        compiled_native_operators.len(),
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
