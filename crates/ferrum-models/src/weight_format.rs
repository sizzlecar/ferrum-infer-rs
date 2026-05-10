//! Dim 3 polymorphism point — weight-format detection for the executor
//! factory.
//!
//! Sibling of `source::ModelFormat`. The difference:
//!
//! - [`source::ModelFormat`](crate::source::ModelFormat) is a "what kind
//!   of files did we just download" hint used for cache classification
//!   and progress bars. It carries no path.
//! - [`WeightFormat`] is a "loader recipe" — it carries the resolved
//!   path AND tells the executor factory which `WeightLoader<B>` to
//!   instantiate. New formats (AWQ, EXL2, HQQ, ...) plug in by adding a
//!   variant + a `WeightLoader<B>` impl in `ferrum-quantization`, with
//!   no special-casing in `LlmExecutorFactory`.
//!
//! Replaces the `is_gguf_path` short-circuit in
//! `ferrum-engine::registry::CandleExecutorFactory` with a real
//! polymorphism point matching the 5-dim architecture (see
//! `docs/architecture-refactor-status.md`).

use ferrum_types::{FerrumError, Result};
use std::path::{Path, PathBuf};

/// Resolved weight format + path. Produced by [`WeightFormat::detect`]
/// from a user-supplied path (HF cache snapshot, local dir, or a
/// `.gguf` file).
#[derive(Debug, Clone)]
pub enum WeightFormat {
    /// HuggingFace safetensors directory: `config.json` + one or more
    /// `.safetensors` shards. The on-disk weights may be plain FP16/BF16
    /// **or** GPTQ-Int4 (`<name>.qweight` tensors); this is decided
    /// per-tensor by `NativeSafetensorsLoader::load_linear`.
    Safetensors { dir: PathBuf },

    /// GGUF single-file format (Llama-family / Qwen3-MoE quantized).
    /// Loaded by `ferrum_quantization::gguf::GgufLoader`.
    Gguf { path: PathBuf },
    // Future: Awq { dir }, Exl2 { dir }, Hqq { dir } …
}

impl WeightFormat {
    /// Detect the weight format from a user-supplied path.
    ///
    /// - If `path` is a file ending in `.gguf` (case-insensitive)
    ///   → [`WeightFormat::Gguf`].
    /// - If `path` is a directory containing `config.json`
    ///   → [`WeightFormat::Safetensors`].
    /// - Anything else returns a model error.
    pub fn detect(path: &Path) -> Result<Self> {
        if path.is_file()
            && path
                .extension()
                .map(|e| e.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
        {
            return Ok(Self::Gguf {
                path: path.to_owned(),
            });
        }
        if path.is_dir() && path.join("config.json").is_file() {
            return Ok(Self::Safetensors {
                dir: path.to_owned(),
            });
        }
        Err(FerrumError::model(format!(
            "Unrecognized weight format at {}: expected a `.gguf` file or \
             a HuggingFace safetensors directory containing `config.json`.",
            path.display()
        )))
    }

    /// The on-disk path this format resolved to.
    pub fn path(&self) -> &Path {
        match self {
            Self::Safetensors { dir } => dir,
            Self::Gguf { path } => path,
        }
    }

    /// Short label for logs / telemetry.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Safetensors { .. } => "safetensors",
            Self::Gguf { .. } => "gguf",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn detect_gguf_by_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Qwen3-0.6B-Q4_K_M.gguf");
        fs::write(&path, b"GGUF\0\0\0\0").unwrap();
        let fmt = WeightFormat::detect(&path).unwrap();
        assert!(matches!(fmt, WeightFormat::Gguf { .. }));
        assert_eq!(fmt.label(), "gguf");
    }

    #[test]
    fn detect_safetensors_dir() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        let fmt = WeightFormat::detect(dir.path()).unwrap();
        assert!(matches!(fmt, WeightFormat::Safetensors { .. }));
        assert_eq!(fmt.label(), "safetensors");
    }

    #[test]
    fn detect_unknown_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        // Empty dir, no config.json, not a .gguf file.
        let err = WeightFormat::detect(dir.path()).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("Unrecognized weight format"), "{msg}");
    }
}
