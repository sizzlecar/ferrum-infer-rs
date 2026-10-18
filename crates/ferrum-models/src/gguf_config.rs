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
//! The dense constructor accepts only the architectures implemented by the
//! Llama-family runtime. MoE families require their typed production definition.
//!
//! Architecture-specific notes:
//!   - **qwen3**: has QK-norm by convention; rope_theta default 1e6.
//!   - **qwen2 / qwen2.5**: no QK-norm; rope_theta default 1e6.
//!   - **llama**: no QK-norm; rope_theta default 5e5 (Llama-3.x).
//!   - **mistral**: no QK-norm; rope_theta default 1e7; reads
//!     `mistral.attention.sliding_window` if present.

use ferrum_quantization::gguf::GgufFile;
use ferrum_types::{Activation, FerrumError, NormType, Result};

use crate::definition::ModelDefinition;
use crate::models::llama_family::{
    llama_layer_attention_schedule, LlamaFamilyConfig, RopeScalingConfig,
};
use crate::registry::Architecture;

/// Architectures that are known MoE — `LlamaFamilyConfig::from_gguf` rejects
/// them with a pointer to typed production sources rather than
/// silently lowering them to a dense config.
const KNOWN_MOE_ARCHS: &[&str] = &["qwen3moe", "mixtral", "deepseek2"];

/// Shared eligibility check for the public dense config and product loader.
pub(crate) fn validate_dense_gguf_architecture(arch: &str) -> Result<()> {
    match arch {
        "qwen" | "qwen2" | "qwen3" | "llama" | "mistral" | "gemma3" => Ok(()),
        _ => Err(FerrumError::unsupported(format!(
            "GGUF architecture '{arch}' is not supported by the dense Llama-family runtime"
        ))),
    }
}

/// Checks semantic compatibility before any weight materialization. Physical
/// RoPE layout and Gemma norm offsets remain those of the converted GGUF.
pub(crate) fn validate_dense_gguf_sources(
    gguf: &GgufFile,
    definition: &ModelDefinition,
) -> Result<LlamaFamilyConfig> {
    let arch = gguf
        .architecture()
        .map_err(|error| FerrumError::model(format!("read GGUF arch: {error}")))?;
    let compatible_architecture = matches!(
        (definition.architecture, arch),
        (Architecture::Llama, "llama")
            | (Architecture::Mistral, "mistral" | "llama")
            | (Architecture::Qwen2, "qwen2")
            | (Architecture::Qwen3, "qwen3")
            | (Architecture::Gemma3, "gemma3")
    );
    if !compatible_architecture {
        return Err(FerrumError::unsupported(format!(
            "semantic architecture {:?} is incompatible with dense GGUF architecture '{arch}'",
            definition.architecture,
        )));
    }
    if definition.extra_params.get("architectures").is_none()
        && definition.extra_params.get("model_type").is_none()
    {
        return Err(FerrumError::model(
            "GGUF semantic source must declare its architecture",
        ));
    }
    if definition.num_attention_heads == 0
        || definition.hidden_size == 0
        || definition.norm_type != NormType::RMSNorm
    {
        return Err(FerrumError::model(
            "GGUF semantic source requires nonzero attention dimensions and RMSNorm",
        ));
    }
    if let Some(pattern) = definition.extra_params.get("sliding_window_pattern") {
        if definition.architecture != Architecture::Gemma3
            || pattern.as_u64().filter(|value| *value > 0).is_none()
        {
            return Err(FerrumError::unsupported(
                "semantic sliding_window_pattern requires a positive integer on the Gemma3 decoder",
            ));
        }
    }
    let mut semantic = match definition.architecture {
        Architecture::Llama => LlamaFamilyConfig::llama_from_def(definition),
        Architecture::Mistral => LlamaFamilyConfig::mistral_from_def(definition),
        Architecture::Qwen2 => LlamaFamilyConfig::qwen2_from_def(definition),
        Architecture::Qwen3 => LlamaFamilyConfig::qwen3_from_def(definition),
        Architecture::Gemma3 => LlamaFamilyConfig::gemma3_from_def(definition),
        _ => unreachable!("architecture checked above"),
    };
    semantic.activation = match definition.activation {
        Activation::Swish => Activation::SiLU,
        activation => activation,
    };
    for key in ["rope_scaling", "rope_parameters"] {
        if let Some(scaling) = definition
            .extra_params
            .get(key)
            .filter(|value| !value.is_null())
        {
            if !scaling.is_object() {
                return Err(FerrumError::model(format!(
                    "semantic {key} must be an object"
                )));
            }
            if key == "rope_parameters"
                && scaling
                    .as_object()
                    .unwrap()
                    .values()
                    .any(|value| value.is_object() || value.is_array())
            {
                return Err(FerrumError::unsupported(
                    "nested semantic rope_parameters are not implemented by the GGUF decoder",
                ));
            }
            for kind_key in ["rope_type", "type"] {
                if scaling
                    .get(kind_key)
                    .is_some_and(|value| !value.is_string())
                {
                    return Err(FerrumError::model(format!(
                        "semantic {key}.{kind_key} must be a string"
                    )));
                }
            }
            let kind = scaling
                .get("rope_type")
                .or_else(|| scaling.get("type"))
                .and_then(|value| value.as_str());
            if (key == "rope_scaling" && kind.is_none())
                || (!matches!(kind, None | Some("default" | "none"))
                    && (key != "rope_scaling" || semantic.rope_scaling.is_none()))
            {
                return Err(FerrumError::unsupported(format!(
                    "semantic {key} is not implemented by the GGUF decoder"
                )));
            }
        }
    }
    // Optional means absent, not malformed. Do not accept an invalid explicit
    // declaration through from_gguf's compatibility defaults.
    for suffix in [
        "attention.head_count_kv",
        "context_length",
        "attention.sliding_window",
        "vocab_size",
        "attention.key_length",
        "attention.value_length",
        "rope.dimension_count",
    ] {
        let key = format!("{arch}.{suffix}");
        if gguf.metadata(&key).is_some() {
            read_u32(gguf, &key)?;
        }
    }
    let theta_key = format!("{arch}.rope.freq_base");
    if gguf.metadata(&theta_key).is_some() {
        read_f32(gguf, &theta_key)?;
    }
    let physical = LlamaFamilyConfig::from_gguf(gguf)?;
    macro_rules! equal {
        ($field:ident) => {
            if semantic.$field != physical.$field {
                return Err(FerrumError::model(format!(
                    "GGUF semantic mismatch for {}: semantic={:?}, physical={:?}",
                    stringify!($field),
                    semantic.$field,
                    physical.$field,
                )));
            }
        };
    }
    equal!(hidden_size);
    equal!(intermediate_size);
    equal!(num_heads);
    equal!(num_kv_heads);
    equal!(head_dim);
    equal!(num_layers);
    equal!(vocab_size);
    equal!(max_seq_len);
    equal!(rms_norm_eps);
    equal!(has_qk_norm);
    equal!(sliding_window);
    equal!(activation);
    equal!(sliding_window_pattern);
    equal!(rope_local_theta);
    equal!(sandwich_norms);
    equal!(embed_scale);
    if !physical.rms_norm_eps.is_finite()
        || physical.rms_norm_eps <= 0.0
        || !physical.rope_theta.is_finite()
        || physical.rope_theta <= 0.0
    {
        return Err(FerrumError::model(
            "GGUF source has invalid dimensions or normalization parameters",
        ));
    }
    // GGUF stores this scalar as f32; compare after the format conversion.
    if semantic.rope_theta as f32 as f64 != physical.rope_theta
        || !equivalent_rope_scaling(&semantic.rope_scaling, &physical.rope_scaling)
        || semantic
            .query_pre_attn_scalar
            .unwrap_or(semantic.head_dim as f64)
            != physical
                .query_pre_attn_scalar
                .unwrap_or(physical.head_dim as f64)
    {
        return Err(FerrumError::model(
            "GGUF semantic mismatch for RoPE scaling/theta or attention scale",
        ));
    }
    for suffix in ["attention.value_length", "rope.dimension_count"] {
        let key = format!("{arch}.{suffix}");
        if gguf.metadata(&key).is_some() && read_u32(gguf, &key)? as usize != physical.head_dim {
            return Err(FerrumError::unsupported(format!(
                "GGUF {key} differs from the full head dimension supported by this decoder"
            )));
        }
    }
    validate_declared_layer_types(definition, &physical)?;
    validate_projection_biases(gguf, definition, &physical)?;
    // These two representation differences are intentional: llama.cpp permutes
    // Llama/Mistral Q/K for interleaved RoPE and folds Gemma's 1+w into weights.
    // They must not be overwritten by the semantic safetensors configuration.
    Ok(physical)
}

fn validate_declared_layer_types(
    definition: &ModelDefinition,
    physical: &LlamaFamilyConfig,
) -> Result<()> {
    let Some(declaration) = definition
        .extra_params
        .get("layer_types")
        .filter(|v| !v.is_null())
    else {
        return Ok(());
    };
    let layers = declaration
        .as_array()
        .ok_or_else(|| FerrumError::model("semantic layer_types must be an array"))?;
    if layers.len() != physical.num_layers {
        return Err(FerrumError::model(
            "semantic layer_types length differs from the GGUF layer count",
        ));
    }
    for (index, layer) in layers.iter().enumerate() {
        let schedule = llama_layer_attention_schedule(physical, index);
        let expected = if !schedule.is_global_layer || schedule.sliding_window > 0 {
            "sliding_attention"
        } else {
            "full_attention"
        };
        if layer.as_str() != Some(expected) {
            return Err(FerrumError::unsupported(format!(
                "semantic layer_types[{index}] does not match the GGUF decoder's {expected} schedule"
            )));
        }
    }
    Ok(())
}

fn validate_projection_biases(
    gguf: &GgufFile,
    definition: &ModelDefinition,
    physical: &LlamaFamilyConfig,
) -> Result<()> {
    let declared_bool = |key: &str| -> Result<Option<bool>> {
        definition
            .extra_params
            .get(key)
            .map(|value| {
                value
                    .as_bool()
                    .ok_or_else(|| FerrumError::model(format!("semantic {key} must be a boolean")))
            })
            .transpose()
    };
    let attention = declared_bool("attention_bias")?;
    let mlp = declared_bool("mlp_bias")?;
    // Family contracts: Qwen2 always has Q/K/V bias and no O/MLP bias;
    // Mistral has none. Llama's MLP and Llama/Qwen3/Gemma3 attention are
    // configured independently. These are architecture rules, not model IDs.
    let (fixed_attention, fixed_mlp) = match definition.architecture {
        Architecture::Qwen2 => (Some(true), Some(false)),
        Architecture::Mistral => (Some(false), Some(false)),
        Architecture::Llama => (None, None),
        Architecture::Qwen3 | Architecture::Gemma3 => (None, Some(false)),
        _ => unreachable!("dense source architecture checked above"),
    };
    for (key, declared, fixed) in [
        ("attention_bias", attention, fixed_attention),
        ("mlp_bias", mlp, fixed_mlp),
    ] {
        if let (Some(declared), Some(fixed)) = (declared, fixed) {
            if declared != fixed {
                return Err(FerrumError::model(format!(
                    "semantic {key}={declared} contradicts the {:?} fixed bias contract ({fixed})",
                    definition.architecture,
                )));
            }
        }
    }
    let attention = attention.unwrap_or(false);
    let mlp = mlp.unwrap_or(false);
    let (qkv_bias, output_bias, mlp_bias) = match definition.architecture {
        Architecture::Qwen2 => (true, false, false),
        Architecture::Mistral => (false, false, false),
        Architecture::Llama => (attention, attention, mlp),
        Architecture::Qwen3 | Architecture::Gemma3 => (attention, attention, false),
        _ => unreachable!("dense source architecture checked above"),
    };
    let width = |heads: usize| {
        heads
            .checked_mul(physical.head_dim)
            .ok_or_else(|| FerrumError::model("GGUF attention projection width overflows usize"))
    };
    let projections = [
        ("attn_q", qkv_bias, width(physical.num_heads)?),
        ("attn_k", qkv_bias, width(physical.num_kv_heads)?),
        ("attn_v", qkv_bias, width(physical.num_kv_heads)?),
        ("attn_output", output_bias, physical.hidden_size),
        ("ffn_gate", mlp_bias, physical.intermediate_size),
        ("ffn_up", mlp_bias, physical.intermediate_size),
        ("ffn_down", mlp_bias, physical.hidden_size),
    ];
    for layer in 0..physical.num_layers {
        for (projection, expected, length) in projections {
            let name = format!("blk.{layer}.{projection}.bias");
            let tensor = gguf.tensor_info(&name);
            if tensor.is_some() != expected {
                return Err(FerrumError::model(format!(
                    "GGUF semantic mismatch for {name}: expected bias presence {expected}"
                )));
            }
            if tensor.is_some_and(|tensor| tensor.shape.dims() != [length]) {
                return Err(FerrumError::model(format!(
                    "GGUF bias {name} must have shape [{length}]"
                )));
            }
        }
    }
    if gguf.has_tensor("output.bias") {
        return Err(FerrumError::model(
            "GGUF semantic mismatch: supported causal language heads have no output.bias",
        ));
    }
    Ok(())
}

fn equivalent_rope_scaling(
    left: &Option<RopeScalingConfig>,
    right: &Option<RopeScalingConfig>,
) -> bool {
    match (left, right) {
        (
            Some(RopeScalingConfig::Linear { factor: left }),
            Some(RopeScalingConfig::Linear { factor: right }),
        ) => *left as f32 == *right as f32,
        (None, Some(RopeScalingConfig::Linear { factor }))
        | (Some(RopeScalingConfig::Linear { factor }), None) => *factor == 1.0,
        _ => left == right,
    }
}

impl LlamaFamilyConfig {
    /// Parse a `LlamaFamilyConfig` out of a GGUF file's metadata.
    ///
    /// Errors if `general.architecture` is missing or unrecognised, if
    /// any required architecture-scoped key is absent, or if the GGUF
    /// is a known MoE variant. Those require typed semantic and tokenizer
    /// sources through the production registry so expert routing is preserved.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf
            .architecture()
            .map_err(|e| FerrumError::model(format!("read general.architecture: {e}")))?
            .to_string();

        if KNOWN_MOE_ARCHS.contains(&arch.as_str()) {
            return Err(FerrumError::unsupported(format!(
                "GGUF arch '{arch}' is MoE and requires typed semantic and tokenizer sources through a supported production model registry; it cannot use the dense Llama-family config"
            )));
        }
        validate_dense_gguf_architecture(&arch)?;

        let block_count = read_u32(gguf, &format!("{arch}.block_count"))? as usize;
        let hidden_size = read_u32(gguf, &format!("{arch}.embedding_length"))? as usize;
        let intermediate_size = read_u32(gguf, &format!("{arch}.feed_forward_length"))? as usize;
        let num_heads = read_u32(gguf, &format!("{arch}.attention.head_count"))? as usize;
        if [block_count, hidden_size, intermediate_size, num_heads].contains(&0) {
            return Err(FerrumError::model(
                "GGUF dense layer, embedding, FFN and attention dimensions must be nonzero",
            ));
        }
        // GQA models put kv-head count here; older ones omit it (= num_heads)
        let num_kv_heads = match read_u32(gguf, &format!("{arch}.attention.head_count_kv")) {
            Ok(v) => v as usize,
            Err(_) => num_heads,
        };
        if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
            return Err(FerrumError::model(
                "GGUF KV head count must be nonzero and divide the attention head count",
            ));
        }
        let rms_norm_eps =
            read_f32(gguf, &format!("{arch}.attention.layer_norm_rms_epsilon"))? as f32;
        // Some GGUFs store context length, some don't — fall back to a sane
        // default rather than failing the whole config.
        let max_seq_len = read_u32(gguf, &format!("{arch}.context_length"))
            .map(|v| v as usize)
            .unwrap_or(4096);

        // rope_theta: optional; per-arch defaults.
        let default_rope = match arch.as_str() {
            "qwen3" | "qwen2" | "gemma3" => 1_000_000.0_f64,
            "llama" => 500_000.0,
            "mistral" => 10_000_000.0,
            _ => 10_000.0,
        };
        let rope_theta = read_f32(gguf, &format!("{arch}.rope.freq_base"))
            .map(|v| v as f64)
            .unwrap_or(default_rope);
        let rope_scaling = read_rope_scaling(gguf, arch.as_str(), max_seq_len, rope_theta)?;

        // QK-norm: Qwen3 and Gemma3 among supported architectures.
        let has_qk_norm = matches!(arch.as_str(), "qwen3" | "gemma3");
        let rope_interleaved = matches!(arch.as_str(), "llama");

        // Sliding window: only Mistral v0.1 sets it; missing → 0 (disabled).
        let sliding_window = read_u32(gguf, &format!("{arch}.attention.sliding_window"))
            .map(|v| v as usize)
            .unwrap_or(0);

        // Vocab size: prefer arch-scoped, fall back to embed-table row count.
        let vocab_size = match read_u32(gguf, &format!("{arch}.vocab_size")) {
            Ok(v) => v as usize,
            Err(_) => infer_vocab_from_embed(gguf)?,
        };

        // head_dim: prefer the explicit `<arch>.attention.key_length` key
        // when present — Qwen3-MoE / Qwen3-30B-A3B has hidden_size=2048
        // but num_heads*head_dim=4096 (head_dim=128), so the
        // `hidden_size / num_heads` shortcut is WRONG for that family.
        // Fall back to the divide only for older GGUFs that omit the key.
        let head_dim = match read_u32(gguf, &format!("{arch}.attention.key_length")) {
            Ok(v) => v as usize,
            Err(_) => {
                if num_heads == 0 || hidden_size % num_heads != 0 {
                    return Err(FerrumError::model(format!(
                        "GGUF config: head_dim missing AND hidden_size {hidden_size} not divisible by num_heads {num_heads}"
                    )));
                }
                hidden_size / num_heads
            }
        };
        if [head_dim, vocab_size, max_seq_len].contains(&0) {
            return Err(FerrumError::model(
                "GGUF head, vocabulary and context dimensions must be nonzero",
            ));
        }

        let mut cfg = LlamaFamilyConfig {
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
            rope_scaling,
            rope_interleaved,
            has_qk_norm,
            sliding_window,
            ..Default::default()
        };

        if arch == "gemma3" {
            // Gemma 3 GGUF stores only window/θ_global/key_length; llama.cpp
            // hardcodes the rest per-arch — mirror those rules exactly.
            cfg.activation = ferrum_types::Activation::GeluTanh;
            cfg.sliding_window_pattern = 6;
            cfg.rope_local_theta = Some(10_000.0);
            cfg.sandwich_norms = true;
            // llama.cpp conversion pre-bakes the Gemma (1+w) into every
            // norm weight (verified against the safetensors values) —
            // never fold again on the GGUF lane.
            cfg.rmsnorm_unit_offset = false;
            cfg.embed_scale = Some(super::models::llama_family::bf16_round(
                (hidden_size as f64).sqrt() as f32,
            ));
            // query_pre_attn_scalar is not in the GGUF. HF configs: 27B
            // (62 blocks) uses hidden/num_heads (168); every other size
            // uses head_dim. Mirrors llama.cpp's per-size rule.
            cfg.query_pre_attn_scalar = Some(if block_count == 62 {
                (hidden_size / num_heads) as f64
            } else {
                head_dim as f64
            });
        }

        Ok(cfg)
    }
}

/// Read just the layer count from a GGUF header. Multi-GPU serve uses
/// this to materialize the even layer-split plan (the safetensors path
/// gets the count from ModelDefinition; GGUF skips that parse).
pub fn gguf_num_layers(path: &std::path::Path) -> Result<usize> {
    let gguf = GgufFile::open(path)
        .map_err(|e| FerrumError::model(format!("GgufFile::open {}: {e}", path.display())))?;
    let arch = gguf
        .architecture()
        .map_err(|e| FerrumError::model(format!("read GGUF arch: {e}")))?
        .to_string();
    Ok(read_u32(&gguf, &format!("{arch}.block_count"))? as usize)
}

fn read_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    gguf.metadata_u32(key)
        .map_err(|e| FerrumError::model(format!("GGUF {key}: {e}")))
}

fn read_f32(gguf: &GgufFile, key: &str) -> Result<f32> {
    gguf.metadata_f32(key)
        .map_err(|e| FerrumError::model(format!("GGUF {key}: {e}")))
}

fn read_rope_scaling(
    gguf: &GgufFile,
    arch: &str,
    max_seq_len: usize,
    rope_theta: f64,
) -> Result<Option<RopeScalingConfig>> {
    let key = format!("{arch}.rope.scaling.type");
    if gguf.metadata(&key).is_none() {
        if gguf
            .metadata(&format!("{arch}.rope.scaling.factor"))
            .is_some()
        {
            return Err(FerrumError::model(format!(
                "GGUF {key} is required with an explicit scaling factor"
            )));
        }
        return Ok(infer_llama3_rope_scaling(
            gguf,
            arch,
            max_seq_len,
            rope_theta,
        ));
    }
    let kind = gguf
        .metadata_string(&key)
        .map_err(|error| FerrumError::model(format!("GGUF {key}: {error}")))?;
    match kind {
        "none" => Ok(None),
        "linear" => {
            let factor = read_f32(gguf, &format!("{arch}.rope.scaling.factor"))? as f64;
            if !factor.is_finite() || factor <= 0.0 {
                return Err(FerrumError::model(
                    "GGUF linear RoPE scaling factor must be finite and positive",
                ));
            }
            Ok(Some(RopeScalingConfig::Linear { factor }))
        }
        _ => Err(FerrumError::unsupported(format!(
            "GGUF RoPE scaling '{kind}' is not implemented by the dense decoder"
        ))),
    }
}

fn infer_llama3_rope_scaling(
    gguf: &GgufFile,
    arch: &str,
    max_seq_len: usize,
    rope_theta: f64,
) -> Option<RopeScalingConfig> {
    if arch != "llama" || max_seq_len < 65_536 || (rope_theta - 500_000.0).abs() > 1.0 {
        return None;
    }
    let meta = [
        "general.basename",
        "general.name",
        "general.license",
        "general.finetune",
    ]
    .iter()
    .filter_map(|key| gguf.metadata_string(key).ok())
    .collect::<Vec<_>>()
    .join(" ")
    .to_ascii_lowercase();
    let is_llama3_long = [
        "llama-3.1",
        "llama 3.1",
        "llama3.1",
        "llama-3.2",
        "llama 3.2",
        "llama3.2",
        "llama-3.3",
        "llama 3.3",
        "llama3.3",
    ]
    .iter()
    .any(|needle| meta.contains(needle));
    if is_llama3_long {
        Some(RopeScalingConfig::llama3_default())
    } else {
        None
    }
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
