//! Build a `LlamaFamilyConfig` from a GGUF file's metadata.
//!
//! GGUF model files store the architecture as `general.architecture` (string)
//! and namespace the actual hyperparameters under that prefix:
//!   `qwen3.block_count`, `qwen3.embedding_length`, `qwen3.attention.head_count`, …
//!   `llama.block_count`, `llama.attention.head_count_kv`, …
//!
//! This helper reads those metadata fields and produces the same
//! `LlamaFamilyConfig` you would otherwise build via `qwen3_from_def` /
//! `llama_from_def` from a HuggingFace `config.json`. Returning the same
//! type means downstream model construction (`LlamaFamilyModel::load`) is
//! unchanged regardless of source format.
//!
//! Phase 1C scope: dense Llama-family architectures (qwen3 / qwen2 / llama /
//! mistral / tinyllama). MoE-specific fields (`expert_count`, `expert_used_count`)
//! are deferred to Phase 2 alongside the MoE runtime.
//!
//! Architecture-specific notes:
//!   - **qwen3**: has QK-norm by convention; rope_theta default 1e6.
//!   - **qwen2 / qwen2.5**: no QK-norm; rope_theta default 1e6.
//!   - **llama**: no QK-norm; rope_theta default 5e5 (Llama-3.x).
//!   - **mistral**: no QK-norm; rope_theta default 1e7; reads
//!     `mistral.attention.sliding_window` if present.

use ferrum_quantization::gguf::GgufFile;
use ferrum_types::{FerrumError, Result};

use crate::models::llama_family::LlamaFamilyConfig;
use crate::moe_config::Qwen3MoeConfig;

/// Architectures that are known MoE — `LlamaFamilyConfig::from_gguf` rejects
/// them with a pointer to the appropriate MoE constructor rather than
/// silently lowering them to a dense config.
const KNOWN_MOE_ARCHS: &[&str] = &["qwen3moe", "mixtral", "deepseek2"];

impl LlamaFamilyConfig {
    /// Parse a `LlamaFamilyConfig` out of a GGUF file's metadata.
    ///
    /// Errors if `general.architecture` is missing or unrecognised, if
    /// any required architecture-scoped key is absent, or if the GGUF
    /// is a known MoE variant — those go through
    /// [`Qwen3MoeConfig::from_gguf`] instead so the MoE-specific
    /// hyperparameters aren't silently dropped.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf
            .architecture()
            .map_err(|e| FerrumError::model(format!("read general.architecture: {e}")))?
            .to_string();

        if KNOWN_MOE_ARCHS.contains(&arch.as_str()) {
            return Err(FerrumError::model(format!(
                "GGUF arch '{arch}' is MoE — use Qwen3MoeConfig::from_gguf or the matching MoE config builder, not LlamaFamilyConfig::from_gguf"
            )));
        }

        let block_count = read_u32(gguf, &format!("{arch}.block_count"))? as usize;
        let hidden_size = read_u32(gguf, &format!("{arch}.embedding_length"))? as usize;
        let intermediate_size = read_u32(gguf, &format!("{arch}.feed_forward_length"))? as usize;
        let num_heads = read_u32(gguf, &format!("{arch}.attention.head_count"))? as usize;
        // GQA models put kv-head count here; older ones omit it (= num_heads)
        let num_kv_heads = match read_u32(gguf, &format!("{arch}.attention.head_count_kv")) {
            Ok(v) => v as usize,
            Err(_) => num_heads,
        };
        let rms_norm_eps =
            read_f32(gguf, &format!("{arch}.attention.layer_norm_rms_epsilon"))? as f32;
        // Some GGUFs store context length, some don't — fall back to a sane
        // default rather than failing the whole config.
        let max_seq_len = read_u32(gguf, &format!("{arch}.context_length"))
            .map(|v| v as usize)
            .unwrap_or(4096);

        // rope_theta: optional; per-arch defaults.
        let default_rope = match arch.as_str() {
            "qwen3" | "qwen2" => 1_000_000.0_f64,
            "llama" => 500_000.0,
            "mistral" => 10_000_000.0,
            _ => 10_000.0,
        };
        let rope_theta = read_f32(gguf, &format!("{arch}.rope.freq_base"))
            .map(|v| v as f64)
            .unwrap_or(default_rope);

        // QK-norm: only Qwen3 has it among supported architectures.
        let has_qk_norm = matches!(arch.as_str(), "qwen3");

        // Sliding window: only Mistral v0.1 sets it; missing → 0 (disabled).
        let sliding_window = read_u32(gguf, &format!("{arch}.attention.sliding_window"))
            .map(|v| v as usize)
            .unwrap_or(0);

        // Vocab size: prefer arch-scoped, fall back to embed-table row count.
        let vocab_size = match read_u32(gguf, &format!("{arch}.vocab_size")) {
            Ok(v) => v as usize,
            Err(_) => infer_vocab_from_embed(gguf)?,
        };

        // head_dim: GGUF doesn't always store it; derive from hidden / heads.
        // (All known Llama-family checkpoints satisfy hidden_size % num_heads == 0.)
        if num_heads == 0 || hidden_size % num_heads != 0 {
            return Err(FerrumError::model(format!(
                "GGUF config: hidden_size {hidden_size} not divisible by num_heads {num_heads}"
            )));
        }
        let head_dim = hidden_size / num_heads;

        Ok(LlamaFamilyConfig {
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers: block_count,
            vocab_size,
            max_seq_len,
            rms_norm_eps,
            rope_theta,
            has_qk_norm,
            sliding_window,
        })
    }
}

impl Qwen3MoeConfig {
    /// Parse a `Qwen3MoeConfig` out of a GGUF file's metadata.
    ///
    /// Expects `general.architecture == "qwen3moe"`. Reads the dense fields
    /// from the `qwen3moe.*` namespace (same shape as `LlamaFamilyConfig`)
    /// plus the MoE-specific extras. Falls back to sane defaults for
    /// missing optional fields, matching `LlamaFamilyConfig::from_gguf`.
    ///
    /// Qwen3-MoE uses **QK-norm** like dense Qwen3 — that flag is set
    /// regardless of how `LlamaFamilyConfig::from_gguf` would treat the
    /// arch, because the dense path explicitly excludes MoE archs.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf
            .architecture()
            .map_err(|e| FerrumError::model(format!("read general.architecture: {e}")))?
            .to_string();
        if arch != "qwen3moe" {
            return Err(FerrumError::model(format!(
                "Qwen3MoeConfig::from_gguf: expected arch 'qwen3moe', got '{arch}'"
            )));
        }

        // Reuse the same key conventions as the dense path — qwen3moe.*
        // mirrors qwen3.* exactly for the shared transformer dims.
        let num_layers = read_u32(gguf, "qwen3moe.block_count")? as usize;
        let hidden_size = read_u32(gguf, "qwen3moe.embedding_length")? as usize;
        let num_heads = read_u32(gguf, "qwen3moe.attention.head_count")? as usize;
        let num_kv_heads = match read_u32(gguf, "qwen3moe.attention.head_count_kv") {
            Ok(v) => v as usize,
            Err(_) => num_heads,
        };
        let rms_norm_eps = read_f32(gguf, "qwen3moe.attention.layer_norm_rms_epsilon")?;
        let max_seq_len = read_u32(gguf, "qwen3moe.context_length")
            .map(|v| v as usize)
            .unwrap_or(32768);
        let rope_theta = read_f32(gguf, "qwen3moe.rope.freq_base")
            .map(|v| v as f64)
            .unwrap_or(1_000_000.0);
        let vocab_size = match read_u32(gguf, "qwen3moe.vocab_size") {
            Ok(v) => v as usize,
            Err(_) => infer_vocab_from_embed(gguf)?,
        };

        if num_heads == 0 || hidden_size % num_heads != 0 {
            return Err(FerrumError::model(format!(
                "GGUF Qwen3-MoE: hidden_size {hidden_size} not divisible by num_heads {num_heads}"
            )));
        }
        let head_dim = hidden_size / num_heads;

        // MoE-specific keys.
        let num_experts = read_u32(gguf, "qwen3moe.expert_count")? as usize;
        let num_experts_per_tok = read_u32(gguf, "qwen3moe.expert_used_count")? as usize;
        // Per-expert FFN length — distinct from the legacy `feed_forward_length`
        // (which most qwen3moe GGUFs leave as the dense reference value).
        let expert_intermediate_size =
            read_u32(gguf, "qwen3moe.expert_feed_forward_length")? as usize;
        // Whether the router normalises the top-K logits before combining.
        // Qwen3-MoE: yes. Some legacy GGUFs omit this key.
        let norm_topk_prob = match gguf.metadata_bool("qwen3moe.expert_norm_topk_prob") {
            Ok(v) => v,
            Err(_) => true,
        };

        if num_experts_per_tok == 0 || num_experts_per_tok > num_experts {
            return Err(FerrumError::model(format!(
                "GGUF Qwen3-MoE: invalid expert_used_count {num_experts_per_tok} (num_experts={num_experts})"
            )));
        }

        let base = LlamaFamilyConfig {
            hidden_size,
            // Qwen3-MoE has no shared dense FFN; mirror expert size into
            // base for any code that reads `intermediate_size`.
            intermediate_size: expert_intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            vocab_size,
            max_seq_len,
            rms_norm_eps,
            rope_theta,
            // Qwen3-MoE uses QK-norm exactly like dense Qwen3.
            has_qk_norm: true,
            // No sliding window in Qwen3-MoE.
            sliding_window: 0,
        };

        Ok(Self::from_base(
            base,
            num_experts,
            num_experts_per_tok,
            expert_intermediate_size,
            norm_topk_prob,
        ))
    }
}

fn read_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    gguf.metadata_u32(key)
        .map_err(|e| FerrumError::model(format!("GGUF {key}: {e}")))
}

fn read_f32(gguf: &GgufFile, key: &str) -> Result<f32> {
    gguf.metadata_f32(key)
        .map_err(|e| FerrumError::model(format!("GGUF {key}: {e}")))
}

/// Vocab size ≈ rows of the embedding table. Used when `<arch>.vocab_size`
/// isn't recorded in metadata (older GGUF dumps).
fn infer_vocab_from_embed(gguf: &GgufFile) -> Result<usize> {
    let info = gguf.tensor_info("token_embd.weight").ok_or_else(|| {
        FerrumError::model(
            "GGUF: cannot infer vocab — neither <arch>.vocab_size nor token_embd.weight present",
        )
    })?;
    let dims = info.shape.dims();
    if dims.is_empty() {
        return Err(FerrumError::model(
            "GGUF: token_embd.weight has empty shape",
        ));
    }
    Ok(dims[0])
}
