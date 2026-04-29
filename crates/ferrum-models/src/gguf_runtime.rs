//! Thin dispatch wrapper around candle-transformers' quantized GGUF model
//! loaders. Exists so the rest of ferrum can hand a `.gguf` path to
//! something that produces a `Tensor` from token ids without caring
//! whether the underlying arch is Qwen3 / Qwen3-MoE / Llama-3.x /
//! Mistral.
//!
//! ## Why this lives here, not in `models/`
//!
//! `models/llama_family.rs` is ferrum's hand-written transformer with
//! its own `Backend<B>` abstraction, GPTQ Marlin path, custom CUDA
//! kernels, etc. The Q4_K_M Apple Silicon benchmark target needs
//! candle's Metal Q4_K_M dequant kernels (which mistral.rs and
//! llama.cpp also rely on for parity), so we bypass `Backend<B>` for
//! that path entirely. The two runtimes coexist:
//!
//!   - **safetensors / GPTQ / dense workflows** → `LlamaFamilyModel<B>`
//!   - **GGUF Q4_K_M workflows (M1 Max bench)** → `GgufRuntime` (this file)
//!
//! When ferrum gains its own Metal Q4_K_M kernels, the GGUF runtime can
//! be swapped underneath without touching call sites. Until then we
//! lean on candle.

use std::fs::File;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_transformers::models::{quantized_llama, quantized_qwen3, quantized_qwen3_moe};

/// Supported GGUF architectures. The benchmark plan targets exactly these
/// three; everything else returns an error from [`GgufRuntime::open`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufArch {
    /// Dense Qwen3 (e.g. Qwen3-8B-Q4_K_M).
    Qwen3,
    /// Qwen3-MoE (e.g. Qwen3-30B-A3B-Q4_K_M).
    Qwen3Moe,
    /// Llama family (Llama-3.x, TinyLlama, Mistral via the same loader).
    Llama,
}

impl GgufArch {
    fn from_metadata_string(s: &str) -> Option<Self> {
        match s {
            "qwen3" => Some(Self::Qwen3),
            "qwen3moe" => Some(Self::Qwen3Moe),
            // Qwen2 GGUFs in the wild also use the llama-family loader;
            // the tokeniser and rope_theta differ but candle's quantized_llama
            // handles all of them via metadata-driven init.
            "llama" | "qwen2" | "mistral" | "tinyllama" => Some(Self::Llama),
            _ => None,
        }
    }
}

/// Owns one of candle's quantized model implementations and exposes a
/// uniform `forward(tokens, pos_offset) -> logits` API.
///
/// All variants keep their KV cache internally — call [`Self::reset_kv`]
/// between independent prompts.
pub enum GgufRuntime {
    Qwen3(quantized_qwen3::ModelWeights),
    Qwen3Moe(quantized_qwen3_moe::GGUFQWenMoE),
    Llama(quantized_llama::ModelWeights),
}

impl GgufRuntime {
    /// Open a `.gguf` file, dispatch on `general.architecture`, and load
    /// the matching candle model into device memory.
    ///
    /// `dtype_for_moe` — Qwen3-MoE's loader takes a dtype for the
    /// non-quantised tensors (norms, router projection). On Metal use
    /// `DType::F16`; on CPU use `DType::F32` (Metal F16 isn't always
    /// well-tuned on older Apple Silicon and CPU has no F16 fast path).
    pub fn open(
        path: impl AsRef<Path>,
        device: &Device,
        dtype_for_moe: DType,
    ) -> CandleResult<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = open_file(&path)?;
        let content = gguf_file::Content::read(&mut file).map_err(|e| {
            candle_core::Error::Msg(format!(
                "failed to parse GGUF header at {}: {e}",
                path.display()
            ))
        })?;

        let arch_str = content
            .metadata
            .get("general.architecture")
            .ok_or_else(|| candle_core::Error::Msg("GGUF missing general.architecture".into()))?
            .to_string()
            .map_err(|e| candle_core::Error::Msg(format!("general.architecture: {e}")))?
            .clone();

        let arch = GgufArch::from_metadata_string(&arch_str).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "GGUF arch '{arch_str}' unsupported by GgufRuntime — \
                 expected qwen3, qwen3moe, llama, qwen2, mistral, or tinyllama"
            ))
        })?;

        let runtime = match arch {
            GgufArch::Qwen3 => {
                let m = quantized_qwen3::ModelWeights::from_gguf(content, &mut file, device)?;
                GgufRuntime::Qwen3(m)
            }
            GgufArch::Qwen3Moe => {
                let m = quantized_qwen3_moe::GGUFQWenMoE::from_gguf(
                    content,
                    &mut file,
                    device,
                    dtype_for_moe,
                )?;
                GgufRuntime::Qwen3Moe(m)
            }
            GgufArch::Llama => {
                let m = quantized_llama::ModelWeights::from_gguf(content, &mut file, device)?;
                GgufRuntime::Llama(m)
            }
        };

        Ok(runtime)
    }

    /// Read a GGUF file's `general.architecture` without instantiating a
    /// model — useful for CLI arch detection before deciding what to do.
    pub fn detect_arch(path: impl AsRef<Path>) -> CandleResult<GgufArch> {
        let path = path.as_ref().to_path_buf();
        let mut file = open_file(&path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let s = content
            .metadata
            .get("general.architecture")
            .ok_or_else(|| candle_core::Error::Msg("GGUF missing general.architecture".into()))?
            .to_string()
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?
            .clone();
        GgufArch::from_metadata_string(&s)
            .ok_or_else(|| candle_core::Error::Msg(format!("unsupported arch '{s}'")))
    }

    /// Run one forward step. `input` is `[batch, seq_len]` (token ids as
    /// i64 / u32 depending on candle convention — the wrappers all accept
    /// the same Tensor). `offset` is the position the first input token
    /// occupies in the KV cache (0 for prompt prefill, prompt_len for
    /// the first decode step, etc.).
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> CandleResult<Tensor> {
        match self {
            Self::Qwen3(m) => m.forward(input, offset),
            Self::Qwen3Moe(m) => m.forward(input, offset),
            Self::Llama(m) => m.forward(input, offset),
        }
    }

    /// Drop any KV-cache state. Call between independent prompts so the
    /// next forward starts at offset 0 with a clean cache.
    ///
    /// `quantized_qwen3` exposes `clear_kv_cache`; the other two don't,
    /// so for those a "reset" requires re-opening the file. Surface
    /// that limitation rather than silently doing the wrong thing.
    pub fn reset_kv(&mut self) -> CandleResult<()> {
        match self {
            Self::Qwen3(m) => {
                m.clear_kv_cache();
                Ok(())
            }
            Self::Qwen3Moe(_) | Self::Llama(_) => Err(candle_core::Error::Msg(
                "reset_kv unsupported for this arch — re-open the GGUF for a fresh cache".into(),
            )),
        }
    }

    pub fn arch(&self) -> GgufArch {
        match self {
            Self::Qwen3(_) => GgufArch::Qwen3,
            Self::Qwen3Moe(_) => GgufArch::Qwen3Moe,
            Self::Llama(_) => GgufArch::Llama,
        }
    }
}

fn open_file(path: &PathBuf) -> CandleResult<File> {
    File::open(path).map_err(|e| {
        candle_core::Error::Msg(format!("failed to open GGUF '{}': {e}", path.display()))
    })
}

// Suppress unused-import warning when only one variant ends up referenced
// by callers (e.g. the bench tool) — Rust's dead-code analysis can't see
// through the `match` if the caller pins down the variant statically.
#[allow(dead_code)]
fn _force_link<R: Read + Seek>() {
    let _ = std::marker::PhantomData::<R>;
}
