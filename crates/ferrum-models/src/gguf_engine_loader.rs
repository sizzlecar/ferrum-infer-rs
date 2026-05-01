//! GGUF → engine glue. Lets `ferrum serve` / `ferrum bench` accept a
//! `.gguf` path (or an alias resolving to one) and produce a
//! `Box<dyn DecoderOnlyLLM>` that the existing `LlmExecutor` +
//! `ContinuousBatchEngine` can drive.
//!
//! Mirrors the runtime-construction half of `ferrum-cli`'s `run_gguf`
//! path so the ergonomic CLI flow and the engine's executor flow share
//! one code path for "load LlamaFamilyModel<B> from a GGUF file".

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferrum_quantization::gguf::{GgufFile, GgufLoader};
use ferrum_types::{DataType, Device, FerrumError, ModelId, ModelInfo, ModelType, Result};

use crate::common::DecoderOnlyLLM;
use crate::models::llama_family::{LlamaFamilyConfig, LlamaFamilyModel};
use crate::models::Qwen3MoeModel;
use crate::moe_config::Qwen3MoeConfig;

/// Returns `true` if the path looks like a GGUF model (ends in `.gguf`
/// case-insensitively). Used by the engine factory to branch model loading.
pub fn is_gguf_path(path: &str) -> bool {
    Path::new(path)
        .extension()
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
}

/// Best-effort tokenizer auto-discovery for a GGUF file.
///
/// Search order (first match wins):
/// 1. `<gguf-stem>.tokenizer.json` next to the GGUF (e.g.
///    `Qwen3-8B-Q4_K_M.tokenizer.json`).
/// 2. `tokenizer.json` next to the GGUF.
/// 3. `<stem-without-quant>.tokenizer.json` in a sibling
///    `../tokenizers/` directory (matches the `~/ferrum-bench/{models,tokenizers}/`
///    layout used by the Group A benchmark scripts).
///
/// Returns `None` if nothing is found — caller should surface a clear
/// error. We keep this in `ferrum-models` (not `ferrum-cli`) because both
/// the CLI and the engine's tokenizer factory need it.
pub fn auto_discover_tokenizer_path(gguf_path: &Path) -> Option<PathBuf> {
    let dir = gguf_path.parent()?;
    if let Some(stem) = gguf_path.file_stem() {
        let candidate = dir.join(format!("{}.tokenizer.json", stem.to_string_lossy()));
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    let bare = dir.join("tokenizer.json");
    if bare.is_file() {
        return Some(bare);
    }
    // Sibling tokenizers/ directory — strip a trailing -Q*_K_M / -Q*_0 / -Q*_1
    // quant suffix from the stem to find the canonical tokenizer.
    if let Some(stem) = gguf_path.file_stem().and_then(|s| s.to_str()) {
        let base = strip_quant_suffix(stem);
        if let Some(parent) = dir.parent() {
            let sibling = parent
                .join("tokenizers")
                .join(format!("{base}.tokenizer.json"));
            if sibling.is_file() {
                return Some(sibling);
            }
        }
    }
    None
}

fn strip_quant_suffix(stem: &str) -> &str {
    // Strip e.g. "-Q4_K_M", "-Q5_K_M", "-Q8_0", "-Q4_0", "-IQ4_NL", "-Q3_K_L"
    // Anything after the last `-Q` that matches the pattern. Conservative —
    // only strips when the suffix is short and looks quant-y.
    if let Some(idx) = stem.rfind("-Q") {
        let suffix = &stem[idx + 1..];
        let upper_alphanum = suffix
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_');
        if upper_alphanum && suffix.len() <= 8 {
            return &stem[..idx];
        }
    }
    if let Some(idx) = stem.rfind("-IQ") {
        let suffix = &stem[idx + 1..];
        let upper_alphanum = suffix
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_');
        if upper_alphanum && suffix.len() <= 8 {
            return &stem[..idx];
        }
    }
    stem
}

/// Load a GGUF model file as a `Box<dyn DecoderOnlyLLM>` ready for
/// `LlmExecutor` / `ContinuousBatchEngine`. Auto-detects MoE
/// (qwen3moe) vs dense (qwen3 / qwen2 / llama / mistral) from the
/// GGUF metadata.
///
/// `device` selects the backend:
///   - [`Device::CPU`] → `LlamaFamilyModel<CpuBackend>` / `Qwen3MoeModel<CpuBackend>`
///   - [`Device::Metal`] → same on `MetalBackend` (also registers the
///     mmap with the Metal weight cache so subsequent `B::load_quant*`
///     calls wrap weights as zero-copy `MTLBuffer`s — see
///     `register_gguf_mmap` in ferrum-kernels).
///
/// CUDA is not currently wired here — the GGUF Q4_K_M decode path on
/// CUDA still goes through candle's `quantized_*` modules in
/// `gguf_runtime.rs`. Add a CUDA branch below when we have native
/// CUDA Q4_K_M kernels in `ferrum-kernels`.
pub fn load_gguf_decoder(gguf_path: &Path, device: &Device) -> Result<Box<dyn DecoderOnlyLLM>> {
    let (llm, _info) = load_gguf_decoder_with_info(
        gguf_path,
        device,
        ModelId::new(gguf_path.display().to_string()),
    )?;
    Ok(llm)
}

/// Same as [`load_gguf_decoder`], but also returns a [`ModelInfo`] derived
/// from the GGUF metadata — needed by the engine's `LlmExecutor`.
pub fn load_gguf_decoder_with_info(
    gguf_path: &Path,
    device: &Device,
    model_id: ModelId,
) -> Result<(Box<dyn DecoderOnlyLLM>, ModelInfo)> {
    let gguf = GgufFile::open(gguf_path)
        .map_err(|e| FerrumError::model(format!("GgufFile::open {}: {e}", gguf_path.display())))?;
    let arch = gguf
        .architecture()
        .map_err(|e| FerrumError::model(format!("read GGUF arch: {e}")))?
        .to_string();
    let is_moe = arch == "qwen3moe";

    let dense_cfg = if is_moe {
        None
    } else {
        Some(LlamaFamilyConfig::from_gguf(&gguf)?)
    };
    let moe_cfg = if is_moe {
        Some(Qwen3MoeConfig::from_gguf(&gguf)?)
    } else {
        None
    };

    // Build a ModelInfo for the engine's LlmExecutor. The fields here drive
    // memory estimation + OpenAI /v1/models responses; the inference hot path
    // doesn't read them.
    let base_cfg = if let Some(c) = dense_cfg.as_ref() {
        c.clone()
    } else {
        moe_cfg.as_ref().unwrap().base.clone()
    };
    let model_type = match arch.as_str() {
        "qwen3" | "qwen3moe" | "qwen2" | "qwen" => ModelType::Qwen,
        "llama" => ModelType::Llama,
        "mistral" => ModelType::Mistral,
        other => ModelType::Custom(other.to_string()),
    };
    let model_info = ModelInfo {
        model_id,
        model_type,
        num_parameters: 0,
        hidden_size: base_cfg.hidden_size,
        num_layers: base_cfg.num_layers,
        num_heads: base_cfg.num_heads,
        num_kv_heads: base_cfg.num_kv_heads,
        vocab_size: base_cfg.vocab_size,
        max_sequence_length: base_cfg.max_seq_len,
        dtype: DataType::FP32,
        device: device.clone(),
        version: None,
        license: None,
        metadata: std::collections::HashMap::new(),
    };

    let gguf_arc = Arc::new(gguf);

    let llm: Box<dyn DecoderOnlyLLM> = match device {
        Device::CPU => {
            use ferrum_kernels::backend::cpu::CpuBackend;
            let loader = GgufLoader::<CpuBackend>::from_file(gguf_arc.clone());
            if let Some(mc) = moe_cfg {
                let model = Qwen3MoeModel::<CpuBackend>::new(mc, &loader, &gguf_arc)?;
                Box::new(model)
            } else {
                let dc = dense_cfg.unwrap();
                let model = LlamaFamilyModel::<CpuBackend>::new(dc, &loader)?;
                Box::new(model)
            }
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        Device::Metal => {
            #[cfg(feature = "metal")]
            {
                use ferrum_kernels::backend::metal::MetalBackend;
                // Wire the GGUF mmap into the Metal weight loader so subsequent
                // `B::load_quant_*` calls produce zero-copy MTLBuffers. CRITICAL
                // on a 32 GB Mac for 30B-A3B Q4_K_M (saves ~17 GB resident).
                ferrum_kernels::backend::metal::register_gguf_mmap(
                    gguf_arc.mmap_bytes(),
                    gguf_arc.clone(),
                )
                .map_err(|e| FerrumError::model(format!("register_gguf_mmap: {e}")))?;
                let loader = GgufLoader::<MetalBackend>::from_file(gguf_arc.clone());
                if let Some(mc) = moe_cfg {
                    let model = Qwen3MoeModel::<MetalBackend>::new(mc, &loader, &gguf_arc)?;
                    Box::new(model)
                } else {
                    let dc = dense_cfg.unwrap();
                    let model = LlamaFamilyModel::<MetalBackend>::new(dc, &loader)?;
                    Box::new(model)
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                return Err(FerrumError::device(
                    "Metal device requested but `metal` feature not enabled",
                ));
            }
        }
        Device::CUDA(_) => {
            return Err(FerrumError::unsupported(
                "GGUF decoder loading on CUDA is not yet wired through the engine — \
                 use --features cuda with the safetensors path, or fall back to \
                 `ferrum run <path.gguf>` which uses candle's quantized_* modules.",
            ));
        }
        other => {
            return Err(FerrumError::device(format!(
                "GGUF decoder loading does not support device {other:?}"
            )));
        }
    };

    Ok((llm, model_info))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_path_detected() {
        assert!(is_gguf_path("foo.gguf"));
        assert!(is_gguf_path("/abs/path/Qwen3-8B-Q4_K_M.GGUF"));
        assert!(!is_gguf_path("config.json"));
        assert!(!is_gguf_path("model.safetensors"));
        assert!(!is_gguf_path(""));
    }

    #[test]
    fn quant_suffix_stripped() {
        assert_eq!(strip_quant_suffix("Qwen3-8B-Q4_K_M"), "Qwen3-8B");
        assert_eq!(
            strip_quant_suffix("Meta-Llama-3.1-8B-Instruct-Q4_K_M"),
            "Meta-Llama-3.1-8B-Instruct"
        );
        assert_eq!(strip_quant_suffix("Qwen3-30B-A3B-Q5_K_M"), "Qwen3-30B-A3B");
        assert_eq!(strip_quant_suffix("Qwen3-8B-IQ4_NL"), "Qwen3-8B");
        // No quant suffix — unchanged
        assert_eq!(strip_quant_suffix("Qwen3-8B"), "Qwen3-8B");
        // Quirky names with `-Q` mid-string but no trailing quant pattern
        assert_eq!(strip_quant_suffix("My-Model"), "My-Model");
    }
}
