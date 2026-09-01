//! Read-only first-run diagnostics.

use crate::config::CliConfig;
use clap::Args;
use ferrum_types::Result;
use std::process::Command;

#[derive(Args)]
pub struct DoctorCommand {
    /// Optional model alias, Hugging Face repository, or GGUF path to inspect.
    #[arg(value_name = "MODEL")]
    pub model: Option<String>,
}

pub async fn execute(cmd: DoctorCommand, config: CliConfig) -> Result<()> {
    println!("Ferrum {}", env!("CARGO_PKG_VERSION"));
    println!(
        "Platform: {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    let accelerators = compiled_accelerators();
    if accelerators.is_empty() {
        println!("Compiled acceleration: none (CPU/default build)");
    } else {
        println!("Compiled acceleration: {}", accelerators.join(", "));
    }

    if cfg!(feature = "cuda") {
        match cuda_devices() {
            Some(devices) => println!("CUDA devices: {devices}"),
            None => println!(
                "CUDA devices: not visible; check the NVIDIA driver and `nvidia-smi` before loading a model"
            ),
        }
    }

    let cache = crate::source_resolver::hf_cache_dir(&config);
    println!("Model cache: {}", cache.display());
    println!(
        "Cache status: {}",
        if cache.exists() {
            "present"
        } else {
            "not created yet"
        }
    );

    println!();
    if let Some(model) = cmd.model.as_deref() {
        let (source, format) = describe_model(model);
        println!("Requested model: {model}");
        println!("Resolved source: {source}");
        println!("Expected format: {format}");
        println!("No model was downloaded and no inference engine was started.");
        println!();
        println!("Next:");
        println!("  ferrum run {model} --disable-thinking");
        println!(
            "  ferrum serve --model {model} --served-model-name ferrum --disable-thinking --port 8000"
        );
    } else {
        println!("Recommended first model:");
        if cfg!(feature = "metal") {
            println!(
                "  Metal: ferrum run {} --disable-thinking",
                crate::source_resolver::METAL_FIRST_SUCCESS_MODEL
            );
        }
        if cfg!(feature = "cuda") {
            println!(
                "  CUDA:  ferrum run {} --disable-thinking",
                crate::source_resolver::CUDA_FIRST_SUCCESS_MODEL
            );
        }
        if accelerators.is_empty() {
            println!("  Install a Metal or CUDA build for the v0.8 accelerator paths.");
        }
        println!("Pass a model to inspect its source without downloading it.");
    }

    Ok(())
}

fn compiled_accelerators() -> Vec<&'static str> {
    let mut accelerators = Vec::new();
    if cfg!(feature = "metal") {
        accelerators.push("metal");
    }
    if cfg!(feature = "cuda") {
        accelerators.push("cuda");
    }
    accelerators
}

fn cuda_devices() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let devices = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!devices.is_empty()).then_some(devices)
}

fn describe_model(model: &str) -> (String, &'static str) {
    if let Some((repo, filename)) = crate::source_resolver::resolve_gguf_alias(model) {
        return (format!("{repo} / {filename}"), "GGUF");
    }

    let source = crate::source_resolver::resolve_model_alias(model);
    let format = if model.to_ascii_lowercase().ends_with(".gguf") {
        "GGUF"
    } else {
        "repository weights (resolved at startup)"
    };
    (source, format)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctor_reuses_curated_gguf_resolution() {
        let (source, format) = describe_model("qwen3.5:4b-q4_k_m");
        assert_eq!(format, "GGUF");
        assert!(source.contains("unsloth/Qwen3.5-4B-GGUF"));
        assert!(source.contains("Qwen3.5-4B-Q4_K_M.gguf"));
    }

    #[test]
    fn doctor_reuses_hf_alias_resolution() {
        let (source, format) = describe_model("qwen3.5:4b");
        assert_eq!(source, "Qwen/Qwen3.5-4B");
        assert_eq!(format, "repository weights (resolved at startup)");
    }
}
