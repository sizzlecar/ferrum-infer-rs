//! CLI-level model source resolution.
//!
//! Centralises the lookup chain that `run` / `serve` / `bench` were
//! reinventing each in their own copy:
//!
//!   1. **Curated GGUF alias** — resolve an explicit quantized alias to one
//!      repository and filename.
//!   2. **GGUF file path** — if the user passed an existing `*.gguf` file,
//!      build a [`ResolvedModelSource`] directly without HF lookup.
//!   3. **Local model dir** — if the path is an existing directory with
//!      `config.json` + weights, treat it as a direct source.
//!   4. **HF cache hit** — `~/.cache/huggingface/hub/models--<owner>--<repo>/snapshots/<rev>`.
//!   5. **HF download** — fall back to [`HfDownloader`] (`run` / `serve`
//!      only; `bench` callers may opt out).
//!   6. **GPU-memory autosizing** — for GPU backends, run the chat
//!      autosizer once on the resolved snapshot so `FERRUM_KV_MAX_BLOCKS`
//!      etc. are populated before the engine starts.
//!
//! Before this module each command had its own `find_cached_model` /
//! `detect_format` (some forked, some `pub fn`-imported across files),
//! its own GGUF early-return, and its own autosize call site. The
//! duplication caused subtle drift (e.g. `serve` accepting a non-existent
//! `.gguf` path because it didn't call `looks_like_gguf_path` exactly the
//! same way as `run`). All callers now go through
//! [`resolve_model_source`].

use std::path::{Path, PathBuf};

use ferrum_models::source::{ModelFormat, ResolvedModelSource};
use ferrum_server::chat_template::ModelChatTemplate;
use ferrum_types::{
    FerrumError, Result, RuntimeConfigEntry, RuntimeConfigSnapshot, RuntimeConfigSource,
};

use crate::config::CliConfig;
use crate::gpu_mem_autosize::{apply_auto_size_with_profile, AutoSizeProfile};

/// Resolve the single Hugging Face cache root used by product entrypoints.
pub fn hf_cache_dir(config: &CliConfig) -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home);
    }
    PathBuf::from(shellexpand::tilde(&config.models.download.hf_cache_dir).as_ref())
}

/// Detect the on-disk format of a model directory or file.
pub fn detect_format(path: &Path) -> ModelFormat {
    if path.is_file()
        && path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
    {
        return ModelFormat::GGUF;
    }
    if path.join("model.safetensors").exists() || path.join("model.safetensors.index.json").exists()
    {
        ModelFormat::SafeTensors
    } else if path.join("pytorch_model.bin").exists() {
        ModelFormat::PyTorchBin
    } else {
        ModelFormat::Unknown
    }
}

/// True iff `model` is a path to an existing `*.gguf` file.
pub fn looks_like_gguf_path(model: &str) -> bool {
    let p = PathBuf::from(model);
    p.extension()
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
        && p.is_file()
}

/// Stable product-facing model id derived from one resolved source.
///
/// Repository models retain their canonical repository id. Direct local
/// directories use the directory name, while GGUF files use the file stem.
/// `run` and `serve` must use this helper rather than inventing entrypoint-
/// specific ids for the same local source.
pub fn public_model_id(source: &ResolvedModelSource) -> String {
    match source.format {
        ModelFormat::GGUF => source
            .local_path
            .file_stem()
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_else(|| source.original.clone()),
        _ if source.local_path == Path::new(&source.original) => source
            .local_path
            .file_name()
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_else(|| source.original.clone()),
        _ => source.original.clone(),
    }
}

/// Resolve an ergonomic model alias to its canonical Hugging Face model id.
pub fn resolve_model_alias(name: &str) -> String {
    match name.to_lowercase().as_str() {
        "tinyllama" | "tiny" => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        "qwen2.5:0.5b" | "qwen:0.5b" => "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        "qwen2.5:1.5b" | "qwen:1.5b" => "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
        "qwen2.5:3b" | "qwen:3b" => "Qwen/Qwen2.5-3B-Instruct".to_string(),
        "qwen2.5:7b" | "qwen:7b" => "Qwen/Qwen2.5-7B-Instruct".to_string(),
        "qwen3:0.6b" => "Qwen/Qwen3-0.6B".to_string(),
        "qwen3:1.7b" => "Qwen/Qwen3-1.7B".to_string(),
        "qwen3:4b" => "Qwen/Qwen3-4B".to_string(),
        "qwen3:14b" => "Qwen/Qwen3-14B".to_string(),
        "qwen3:32b" => "Qwen/Qwen3-32B".to_string(),
        "qwen3-coder:30b" | "qwen3-coder:30b-a3b" => {
            "Qwen/Qwen3-Coder-30B-A3B-Instruct".to_string()
        }
        "qwen3-coder:30b-gptq" => "jart25/Qwen3-Coder-30B-A3B-Instruct-Int4-gptq".to_string(),
        "qwen3:14b-gptq" => "JunHowie/Qwen3-14B-GPTQ-Int4".to_string(),
        "qwen3:32b-gptq" => "JunHowie/Qwen3-32B-GPTQ-Int4".to_string(),
        "deepseek-r1:8b" | "r1:8b" => "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B".to_string(),
        "deepseek-r1:14b" | "r1:14b" => "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B".to_string(),
        "deepseek-r1:32b" | "r1:32b" => "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B".to_string(),
        "deepseek-r1:32b-gptq" => "OPEA/DeepSeek-R1-Distill-Qwen-32B-int4-gptq-sym-inc".to_string(),
        "qwen2.5-coder:32b" => "Qwen/Qwen2.5-Coder-32B-Instruct".to_string(),
        "qwen2.5-coder:32b-gptq" => "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4".to_string(),
        "qwen2.5-coder:14b" => "Qwen/Qwen2.5-Coder-14B-Instruct".to_string(),
        "gemma3:1b" => "unsloth/gemma-3-1b-it".to_string(),
        "gemma3:4b" => "unsloth/gemma-3-4b-it".to_string(),
        "gemma3:27b" => "unsloth/gemma-3-27b-it".to_string(),
        "gemma3:27b-gptq" => "circulus/gemma-3-27b-it-gptq".to_string(),
        "mistral-small:24b" | "mistral-small:3.2" => {
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506".to_string()
        }
        "devstral:24b" | "devstral:2" => "mistralai/Devstral-Small-2-24B-Instruct-2512".to_string(),
        "magistral:24b" => "mistralai/Magistral-Small-2509".to_string(),
        "qwen2.5:3b-gptq" | "qwen2.5-3b-instruct-gptq-int4" => {
            "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4".to_string()
        }
        "llama3.2:1b" => "meta-llama/Llama-3.2-1B-Instruct".to_string(),
        "llama3.2:3b" => "meta-llama/Llama-3.2-3B-Instruct".to_string(),
        "whisper-tiny" | "whisper:tiny" => "openai/whisper-tiny".to_string(),
        "whisper-base" | "whisper:base" => "openai/whisper-base".to_string(),
        "whisper-small" | "whisper:small" => "openai/whisper-small".to_string(),
        "whisper-medium" | "whisper:medium" => "openai/whisper-medium".to_string(),
        "whisper-large-v3" | "whisper:large-v3" => "openai/whisper-large-v3".to_string(),
        "whisper-turbo" | "whisper:turbo" | "whisper-large-v3-turbo" => {
            "openai/whisper-large-v3-turbo".to_string()
        }
        "qwen3-tts" | "tts" | "tts:0.6b" => "Qwen/Qwen3-TTS-12Hz-0.6B-Base".to_string(),
        "tts:1.7b" | "qwen3-tts:1.7b" => "Qwen/Qwen3-TTS-12Hz-1.7B-Base".to_string(),
        _ => name.to_string(),
    }
}

struct GgufAliasEntry {
    aliases: &'static [&'static str],
    repo: &'static str,
    filename: &'static str,
    tokenizer_repo: Option<&'static str>,
}

const GGUF_ALIASES: &[GgufAliasEntry] = &[
    GgufAliasEntry {
        aliases: &["qwen3:8b-q4_k_m"],
        repo: "Qwen/Qwen3-8B-GGUF",
        filename: "Qwen3-8B-Q4_K_M.gguf",
        tokenizer_repo: None,
    },
    GgufAliasEntry {
        aliases: &["qwen3:4b-q4_k_m"],
        repo: "Qwen/Qwen3-4B-GGUF",
        filename: "Qwen3-4B-Q4_K_M.gguf",
        tokenizer_repo: None,
    },
    GgufAliasEntry {
        // Keep the unqualified `qwen3:1.7b` alias on safetensors. Quantized
        // aliases must name their format so every product entrypoint resolves
        // the same source.
        aliases: &["qwen3:1.7b-gguf", "qwen3:1.7b-q8_0"],
        repo: "Qwen/Qwen3-1.7B-GGUF",
        filename: "Qwen3-1.7B-Q8_0.gguf",
        tokenizer_repo: None,
    },
    GgufAliasEntry {
        aliases: &["qwen3:0.6b-gguf", "qwen3:0.6b-q8_0"],
        repo: "Qwen/Qwen3-0.6B-GGUF",
        filename: "Qwen3-0.6B-Q8_0.gguf",
        tokenizer_repo: None,
    },
    GgufAliasEntry {
        aliases: &["qwen3-moe:30b-a3b-q4_k_m", "qwen3:30b-a3b-q4_k_m"],
        repo: "Qwen/Qwen3-30B-A3B-GGUF",
        filename: "Qwen3-30B-A3B-Q4_K_M.gguf",
        tokenizer_repo: None,
    },
    GgufAliasEntry {
        aliases: &["gemma3:1b-q4_k_m"],
        repo: "unsloth/gemma-3-1b-it-GGUF",
        filename: "gemma-3-1b-it-Q4_K_M.gguf",
        tokenizer_repo: Some("unsloth/gemma-3-1b-it"),
    },
    GgufAliasEntry {
        aliases: &["gemma3:27b-q4_k_m"],
        repo: "unsloth/gemma-3-27b-it-GGUF",
        filename: "gemma-3-27b-it-Q4_K_M.gguf",
        tokenizer_repo: Some("unsloth/gemma-3-27b-it"),
    },
    GgufAliasEntry {
        aliases: &["qwen3:14b-q4_k_m"],
        repo: "Qwen/Qwen3-14B-GGUF",
        filename: "Qwen3-14B-Q4_K_M.gguf",
        tokenizer_repo: None,
    },
    GgufAliasEntry {
        aliases: &["qwen3:32b-q4_k_m"],
        repo: "Qwen/Qwen3-32B-GGUF",
        filename: "Qwen3-32B-Q4_K_M.gguf",
        tokenizer_repo: None,
    },
    GgufAliasEntry {
        aliases: &["qwen3-coder:30b-q4_k_m", "qwen3-coder:30b-a3b-q4_k_m"],
        repo: "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        filename: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        tokenizer_repo: None,
    },
    GgufAliasEntry {
        aliases: &["deepseek-r1:8b-q4_k_m", "r1:8b-q4_k_m"],
        repo: "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
        filename: "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
        tokenizer_repo: None,
    },
    GgufAliasEntry {
        aliases: &["deepseek-r1:32b-q4_k_m", "r1:32b-q4_k_m"],
        repo: "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        filename: "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        tokenizer_repo: None,
    },
    GgufAliasEntry {
        aliases: &["qwen2.5-coder:32b-q4_k_m"],
        repo: "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        filename: "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        tokenizer_repo: Some("Qwen/Qwen2.5-Coder-32B-Instruct"),
    },
    GgufAliasEntry {
        aliases: &["mistral-small:24b-q4_k_m"],
        repo: "bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF",
        filename: "mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
        tokenizer_repo: Some("unsloth/Mistral-Small-3.2-24B-Instruct-2506"),
    },
    GgufAliasEntry {
        aliases: &["devstral:24b-q4_k_m"],
        repo: "bartowski/mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF",
        filename: "mistralai_Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf",
        tokenizer_repo: Some("mistralai/Devstral-Small-2-24B-Instruct-2512"),
    },
    GgufAliasEntry {
        aliases: &["magistral:24b-q4_k_m"],
        repo: "bartowski/mistralai_Magistral-Small-2509-GGUF",
        filename: "mistralai_Magistral-Small-2509-Q4_K_M.gguf",
        tokenizer_repo: Some("unsloth/Magistral-Small-2509"),
    },
    GgufAliasEntry {
        aliases: &["llama3.1:8b-q4_k_m"],
        repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        tokenizer_repo: Some("unsloth/Meta-Llama-3.1-8B-Instruct"),
    },
    GgufAliasEntry {
        aliases: &["llama3.2:3b-q4_k_m"],
        repo: "bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        tokenizer_repo: Some("unsloth/Llama-3.2-3B-Instruct"),
    },
    GgufAliasEntry {
        aliases: &["llama3.2:1b-q4_k_m"],
        repo: "bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename: "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        tokenizer_repo: Some("unsloth/Llama-3.2-1B-Instruct"),
    },
];

/// Resolve a GGUF alias to its repository and exact quantized filename.
pub fn resolve_gguf_alias(name: &str) -> Option<(String, String)> {
    let name = name.to_lowercase();
    GGUF_ALIASES
        .iter()
        .find(|entry| entry.aliases.contains(&name.as_str()))
        .map(|entry| (entry.repo.to_string(), entry.filename.to_string()))
}

/// Resolve the tokenizer sidecar repository for a GGUF repository.
pub fn tokenizer_sibling_repo(gguf_repo: &str) -> Option<String> {
    if let Some(entry) = GGUF_ALIASES.iter().find(|entry| entry.repo == gguf_repo) {
        if let Some(repo) = entry.tokenizer_repo {
            return Some(repo.to_string());
        }
    }
    gguf_repo.strip_suffix("-GGUF").map(str::to_string)
}

/// Chat-profile runtime defaults for `ferrum run`. Sniffs the arch
/// (dense vs MoE — works for both GGUF files and safetensors snapshot
/// dirs) and materializes missing compatibility env vars for:
///
///   - `FERRUM_KV_CAPACITY`     — 8192 dense / 4096 MoE
///   - `FERRUM_PAGED_KV` / legacy `FERRUM_METAL_PAGED_KV` — 0 GGUF / 1 only
///     for Qwen3 dense and MoE safetensors.
///     The Metal Qwen3-MoE GGUF paged-KV decode path can repeat the first
///     generated token until `max_tokens`; keep GGUF on the contiguous path
///     until that kernel path is fixed. Qwen3 dense safetensors is validated
///     on paged KV; TinyLlama/Llama and Qwen2 dense safetensors produce token
///     noise on the Metal paged-KV path and default to contiguous KV.
///   - `FERRUM_PAGED_MAX_SEQS=2` dense / `1` MoE, `FERRUM_MAX_BATCH=1` — single-user REPL.
///     Keeps the paged pool at ~1.7 GB for `cap=8192` dense; without this
///     cap the default `max_seqs=32` makes the pool ~30 GB on a 32 GB Mac.
///   - `FERRUM_MOE_BATCHED=0`, `FERRUM_MOE_BATCHED_DECODE=0`,
///     `FERRUM_MOE_BATCH_THRESHOLD=2` — MoE only. `run` is an interactive
///     single-session path, so do not engage unneeded multi-sequence MoE
///     batching.
///
/// Idempotent: if a user explicitly sets one of these env vars before
/// invoking `ferrum run`, that value wins (we only set when unset).
/// Called automatically by `resolve_model_source` when the autosize
/// profile is `Chat`. Server/bench callers don't get these defaults —
/// they don't fit the multi-turn REPL pattern this profile is tuned for.
///
/// Without this, dense safetensors models (e.g. `Qwen/Qwen3-0.6B`)
/// inherit the model-level `DEFAULT_KV_CAPACITY=512` floor in
/// `llama_family.rs::ensure_kv`, which overflows after ~512 tokens on
/// a `max_tokens=2048` chat — manifesting as a `KV cache overflow on
/// layer 0` panic mid-response.
pub fn apply_chat_profile_env(snapshot_path: &Path) {
    let entries = chat_profile_runtime_entries(
        snapshot_path,
        &RuntimeConfigSnapshot::capture_current(),
        RuntimeConfigSource::Default,
    );
    crate::runtime_env::materialize_runtime_env_defaults(&entries);
}

pub fn chat_profile_runtime_entries(
    snapshot_path: &Path,
    current: &RuntimeConfigSnapshot,
    source: RuntimeConfigSource,
) -> Vec<RuntimeConfigEntry> {
    let is_gguf = snapshot_path.is_file()
        && snapshot_path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
    let is_moe = detect_moe_arch(snapshot_path);
    let model_family = detect_model_family(snapshot_path);
    chat_profile_runtime_entries_for_arch(is_gguf, is_moe, model_family.as_deref(), current, source)
}

fn chat_profile_runtime_entries_for_arch(
    is_gguf: bool,
    is_moe: bool,
    model_family: Option<&str>,
    current: &RuntimeConfigSnapshot,
    source: RuntimeConfigSource,
) -> Vec<RuntimeConfigEntry> {
    let mut entries = Vec::new();

    push_missing_entry(
        &mut entries,
        current,
        "FERRUM_KV_CAPACITY",
        if is_moe { "4096" } else { "8192" },
        source,
    );
    // GGUF: contiguous path is the correctness baseline. For safetensors,
    // keep paged KV only on families with current product evidence.
    let need_paged = !is_gguf
        && (is_moe
            || model_family.is_some_and(|family| {
                family.eq_ignore_ascii_case("qwen3") || family.eq_ignore_ascii_case("qwen3_5")
            }));
    push_paged_kv_compat_entries(
        &mut entries,
        current,
        if need_paged { "1" } else { "0" },
        source,
    );
    // Single-user REPL pool sizing — dense safetensors keeps a spare
    // sequence, while MoE uses one long interactive session. Keeping MoE at
    // 2048 tokens forces Qwen3-30B-A3B thinking-mode chats to shrink answers
    // after a few turns.
    if !is_gguf || is_moe {
        for (k, v) in [
            ("FERRUM_PAGED_MAX_SEQS", if is_moe { "1" } else { "2" }),
            ("FERRUM_MAX_BATCH", "1"),
        ] {
            push_missing_entry(&mut entries, current, k, v, source);
        }
    }
    if is_moe {
        for (k, v) in [
            ("FERRUM_MOE_BATCHED", "0"),
            ("FERRUM_MOE_BATCHED_DECODE", "0"),
            ("FERRUM_MOE_BATCH_THRESHOLD", "2"),
        ] {
            push_missing_entry(&mut entries, current, k, v, source);
        }
    }
    entries
}

fn push_missing_entry(
    entries: &mut Vec<RuntimeConfigEntry>,
    current: &RuntimeConfigSnapshot,
    key: &str,
    value: &str,
    source: RuntimeConfigSource,
) {
    if snapshot_value(current, key).is_none() {
        entries.push(RuntimeConfigEntry::new(key, value, source));
    }
}

fn push_paged_kv_compat_entries(
    entries: &mut Vec<RuntimeConfigEntry>,
    current: &RuntimeConfigSnapshot,
    value: &str,
    source: RuntimeConfigSource,
) {
    let effective_value = snapshot_value(current, "FERRUM_PAGED_KV")
        .or_else(|| snapshot_value(current, "FERRUM_METAL_PAGED_KV"))
        .unwrap_or(value);
    push_missing_entry(entries, current, "FERRUM_PAGED_KV", effective_value, source);
    push_missing_entry(
        entries,
        current,
        "FERRUM_METAL_PAGED_KV",
        effective_value,
        source,
    );
}

fn snapshot_value<'a>(snapshot: &'a RuntimeConfigSnapshot, key: &str) -> Option<&'a str> {
    snapshot
        .entries
        .iter()
        .find(|entry| entry.key == key)
        .map(|entry| entry.effective_value.as_str())
}

/// Detect whether `path` is a Mixture-of-Experts model. Handles both a
/// `.gguf` file (peek `general.architecture` from GGUF metadata) and a
/// safetensors snapshot directory (read `config.json` and match
/// `architectures` / `model_type` against `moe`, case-insensitive).
pub fn detect_moe_arch(path: &Path) -> bool {
    use ferrum_quantization::gguf::GgufFile;

    if path.is_file()
        && path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
    {
        return GgufFile::open(path)
            .ok()
            .and_then(|g| g.architecture().ok().map(|s| s.to_string()))
            .map(|a| a.to_lowercase().contains("moe"))
            .unwrap_or(false);
    }

    let config_path = path.join("config.json");
    let Ok(contents) = std::fs::read_to_string(&config_path) else {
        return false;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&contents) else {
        return false;
    };
    if let Some(archs) = json.get("architectures").and_then(|v| v.as_array()) {
        if archs
            .iter()
            .any(|a| a.as_str().is_some_and(|s| s.to_lowercase().contains("moe")))
        {
            return true;
        }
    }
    json.get("model_type")
        .and_then(|v| v.as_str())
        .is_some_and(|mt| mt.to_lowercase().contains("moe"))
}

pub fn detect_model_family(path: &Path) -> Option<String> {
    use ferrum_quantization::gguf::GgufFile;

    if path.is_file()
        && path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
    {
        return GgufFile::open(path)
            .ok()
            .and_then(|g| g.architecture().ok().map(|s| normalize_model_family(s)));
    }

    let config_path = path.join("config.json");
    let contents = std::fs::read_to_string(&config_path).ok()?;
    let json = serde_json::from_str::<serde_json::Value>(&contents).ok()?;
    if let Some(model_type) = json.get("model_type").and_then(|v| v.as_str()) {
        return Some(normalize_model_family(model_type));
    }
    json.get("architectures")
        .and_then(|v| v.as_array())
        .and_then(|archs| archs.iter().find_map(|arch| arch.as_str()))
        .map(normalize_model_family)
}

fn normalize_model_family(raw: &str) -> String {
    let lower = raw.to_ascii_lowercase();
    if lower.contains("qwen3_5_moe")
        || lower.contains("qwen3_5moe")
        || lower.contains("qwen35_moe")
        || lower.contains("qwen35moe")
    {
        "qwen3_5_moe".to_string()
    } else if lower.contains("qwen3_5") || lower.contains("qwen35") {
        "qwen3_5".to_string()
    } else if lower.contains("qwen3_moe")
        || lower.contains("qwen3moe")
        || lower.contains("qwen3_mo")
    {
        "qwen3_moe".to_string()
    } else if lower.contains("qwen3") {
        "qwen3".to_string()
    } else if lower.contains("qwen2") || lower == "qwen" {
        "qwen2".to_string()
    } else if lower.contains("mistral") {
        "mistral".to_string()
    } else if lower.contains("llama") || lower.contains("tinyllama") {
        "llama".to_string()
    } else {
        lower
    }
}

/// Correctness fallback for Metal GGUF MoE.
///
/// The device-side prefill MoE top-k/bucketing path currently produces
/// incorrect first-token logits for Qwen3-30B-A3B GGUF on Metal. The host
/// top-k path is slower, but is the validated product path until the Metal
/// GPU router is fixed. Keep this scoped to Metal + GGUF + MoE and let an
/// explicit `FERRUM_MOE_HOST_TOPK` env/config value win for diagnostics.
pub fn metal_gguf_moe_correctness_entries(
    snapshot_path: &Path,
    device: &ferrum_types::Device,
    current: &RuntimeConfigSnapshot,
    source: RuntimeConfigSource,
) -> Vec<RuntimeConfigEntry> {
    let is_gguf = snapshot_path.is_file()
        && snapshot_path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
    if !is_gguf || !device_is_metal(device) || !detect_moe_arch(snapshot_path) {
        return Vec::new();
    }

    let mut entries = Vec::new();
    push_missing_entry(&mut entries, current, "FERRUM_MOE_HOST_TOPK", "1", source);
    entries
}

/// Product defaults for `ferrum serve` on GGUF LLMs.
///
/// GGUF paths do not go through the HF-directory autosizer. Without an
/// explicit profile, Qwen3-30B-A3B falls back to the model default
/// `FERRUM_KV_CAPACITY=512`, which can make a normal sequence of OpenAI
/// requests (sync correctness, multi-turn, then stream) end the stream
/// immediately with an empty EOS. Keep this product path correct by default:
/// enough context for Qwen3 thinking-mode responses and a multi-request pool
/// for dense and MoE GGUF serving.
pub fn serve_profile_runtime_entries(
    snapshot_path: &Path,
    device: &ferrum_types::Device,
    current: &RuntimeConfigSnapshot,
    source: RuntimeConfigSource,
) -> Vec<RuntimeConfigEntry> {
    let is_gguf = snapshot_path.is_file()
        && snapshot_path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
    serve_profile_runtime_entries_for_arch(
        is_gguf,
        detect_moe_arch(snapshot_path),
        device_is_metal(device),
        current,
        source,
    )
}

fn device_is_metal(device: &ferrum_types::Device) -> bool {
    #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
    {
        matches!(device, ferrum_types::Device::Metal)
    }
    #[cfg(not(all(any(target_os = "macos", target_os = "ios"), feature = "metal")))]
    {
        let _ = device;
        false
    }
}

pub fn serve_profile_runtime_entries_for_arch(
    is_gguf: bool,
    is_moe: bool,
    is_metal: bool,
    current: &RuntimeConfigSnapshot,
    source: RuntimeConfigSource,
) -> Vec<RuntimeConfigEntry> {
    if !is_gguf {
        return Vec::new();
    }

    let mut entries = Vec::new();
    let kv_capacity = if is_moe && is_metal {
        // Metal Qwen3-MoE paged KV is stable for the README c16 path when
        // total pool blocks stay <= 1024. c16 × 1024 tokens gives exactly
        // that bound and avoids the repeated-token failure seen at
        // c16 × 2048.
        "1024"
    } else if is_moe {
        "2048"
    } else {
        "512"
    };
    push_missing_entry(
        &mut entries,
        current,
        "FERRUM_KV_CAPACITY",
        kv_capacity,
        source,
    );
    push_paged_kv_compat_entries(&mut entries, current, "1", source);
    for (k, v) in [
        (
            "FERRUM_PAGED_MAX_SEQS",
            if is_moe {
                "16"
            } else if is_metal {
                "16"
            } else {
                "32"
            },
        ),
        ("FERRUM_MAX_BATCH", "16"),
    ] {
        push_missing_entry(&mut entries, current, k, v, source);
    }
    if is_moe {
        for (k, v) in [
            ("FERRUM_MAX_BATCHED_TOKENS", "2048"),
            ("FERRUM_MOE_BATCHED", "1"),
            ("FERRUM_MOE_BATCHED_DECODE", "1"),
            ("FERRUM_MOE_BATCH_THRESHOLD", "2"),
        ] {
            push_missing_entry(&mut entries, current, k, v, source);
        }
    }
    entries
}

/// Load a model-provided chat template, if the source carries one.
///
/// GGUF stores this in `tokenizer.chat_template`; HuggingFace snapshots
/// commonly store it in `chat_template.jinja`, `chat_template.json`, or
/// `tokenizer_config.json`. The renderer may still fall back if a template
/// uses unsupported Jinja features, but callers should always prefer this
/// metadata over model-name heuristics.
pub fn load_model_chat_template(snapshot_path: &Path) -> Option<ModelChatTemplate> {
    let is_gguf = snapshot_path.is_file()
        && snapshot_path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
    if is_gguf {
        let gguf = ferrum_quantization::gguf::GgufFile::open(snapshot_path).ok()?;
        let template = gguf.metadata_string("tokenizer.chat_template").ok()?;
        return Some(ModelChatTemplate::new(
            template.to_string(),
            format!("{}:tokenizer.chat_template", snapshot_path.display()),
        ));
    }

    if !snapshot_path.is_dir() {
        return None;
    }

    let jinja_path = snapshot_path.join("chat_template.jinja");
    if let Ok(template) = std::fs::read_to_string(&jinja_path) {
        if !template.trim().is_empty() {
            return Some(ModelChatTemplate::new(
                template,
                jinja_path.display().to_string(),
            ));
        }
    }

    let json_path = snapshot_path.join("chat_template.json");
    if let Some(template) = read_template_json(&json_path) {
        return Some(template);
    }

    let tokenizer_config_path = snapshot_path.join("tokenizer_config.json");
    read_tokenizer_config_template(&tokenizer_config_path)
}

fn read_template_json(path: &Path) -> Option<ModelChatTemplate> {
    let text = std::fs::read_to_string(path).ok()?;
    if text.trim().is_empty() {
        return None;
    }
    match serde_json::from_str::<serde_json::Value>(&text).ok() {
        Some(serde_json::Value::String(template)) => {
            Some(ModelChatTemplate::new(template, path.display().to_string()))
        }
        Some(value) => template_value(&value).map(|template| {
            let mut t = ModelChatTemplate::new(template, path.display().to_string());
            t.bos_token = token_value(&value, "bos_token");
            t.eos_token = token_value(&value, "eos_token");
            t
        }),
        None => Some(ModelChatTemplate::new(text, path.display().to_string())),
    }
}

fn read_tokenizer_config_template(path: &Path) -> Option<ModelChatTemplate> {
    let text = std::fs::read_to_string(path).ok()?;
    let value = serde_json::from_str::<serde_json::Value>(&text).ok()?;
    let template = template_value(&value)?;
    let mut t = ModelChatTemplate::new(template, path.display().to_string());
    t.bos_token = token_value(&value, "bos_token");
    t.eos_token = token_value(&value, "eos_token");
    Some(t)
}

fn template_value(value: &serde_json::Value) -> Option<String> {
    match value.get("chat_template")? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(items) => items
            .iter()
            .find(|item| item.get("name").and_then(|v| v.as_str()) == Some("default"))
            .or_else(|| items.first())
            .and_then(|item| item.get("template").and_then(|v| v.as_str()))
            .map(ToString::to_string),
        serde_json::Value::Object(obj) => obj
            .get("template")
            .and_then(|v| v.as_str())
            .map(ToString::to_string),
        _ => None,
    }
}

fn token_value(value: &serde_json::Value, key: &str) -> Option<String> {
    match value.get(key)? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(obj) => obj
            .get("content")
            .and_then(|v| v.as_str())
            .map(ToString::to_string),
        _ => None,
    }
}

/// Look up `model_id` in the HF cache (`hub/models--owner--repo/snapshots/<rev>`).
/// Returns the resolved snapshot path + detected format, or `None` if not cached.
pub fn find_cached_model(cache_dir: &Path, model_id: &str) -> Option<ResolvedModelSource> {
    let repo_dir = cache_dir
        .join("hub")
        .join(format!("models--{}", model_id.replace('/', "--")));
    let snapshots_dir = repo_dir.join("snapshots");

    // Prefer the revision pointed to by refs/main.
    let ref_main = repo_dir.join("refs").join("main");
    if let Ok(rev) = std::fs::read_to_string(&ref_main) {
        let rev = rev.trim();
        if !rev.is_empty() {
            let snapshot = snapshots_dir.join(rev);
            if snapshot.exists() {
                let format = detect_format(&snapshot);
                if format != ModelFormat::Unknown {
                    return Some(ResolvedModelSource {
                        original: model_id.to_string(),
                        local_path: snapshot,
                        format,
                        from_cache: true,
                    });
                }
            }
        }
    }

    // Fallback: first snapshot directory with valid weights.
    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let format = detect_format(&path);
                if format != ModelFormat::Unknown {
                    return Some(ResolvedModelSource {
                        original: model_id.to_string(),
                        local_path: path,
                        format,
                        from_cache: true,
                    });
                }
            }
        }
    }

    None
}

/// Locate one exact GGUF artifact in the Hugging Face cache.
pub fn find_cached_gguf(cache_dir: &Path, repo: &str, filename: &str) -> Option<PathBuf> {
    let repo_dir = cache_dir
        .join("hub")
        .join(format!("models--{}", repo.replace('/', "--")));
    let snapshots_dir = repo_dir.join("snapshots");

    let ref_main = repo_dir.join("refs").join("main");
    if let Ok(revision) = std::fs::read_to_string(&ref_main) {
        let revision = revision.trim();
        if !revision.is_empty() {
            let candidate = snapshots_dir.join(revision).join(filename);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    std::fs::read_dir(&snapshots_dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path().join(filename))
        .find(|candidate| candidate.is_file())
}

/// Should the resolver attempt to download from HF if the model isn't
/// found locally? `run` / `serve` say yes; `bench` defaults to no
/// (caller handles per-bench-flow download policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadPolicy {
    /// Download from HF if not cached locally.
    AutoDownload,
    /// Error out if not cached.
    NoDownload,
}

/// Resolution outcome — a fully-resolved local source plus a flag the
/// caller can use to decide whether to apply GPU autosize.
pub struct Resolved {
    pub source: ResolvedModelSource,
    /// `true` when the resolver also ran the GPU-memory autosizer for
    /// this snapshot. Caller can skip a redundant call.
    pub autosized: bool,
}

/// One-stop model resolution. Caller passes the user's model arg
/// (alias / HF id / local dir / `.gguf` path), the HF cache dir, and a
/// download policy + autosize profile. Returns a resolved source.
///
/// Resolution order:
///   1. Explicit GGUF alias -> exact cached/downloaded GGUF file.
///   2. `*.gguf` file -> direct GGUF source.
///   3. Existing local directory with valid weights -> direct source.
///   4. HF cache hit -> cached source.
///   5. (if `AutoDownload`) HF download -> cached source.
///
/// On GPU backends the chat-profile autosizer fires once on the resolved
/// snapshot before returning, populating `FERRUM_KV_MAX_BLOCKS` etc.
/// `apply_autosize=false` skips it (used by `bench` which sets sizing
/// from `--max-tokens` etc. directly).
pub async fn resolve_model_source(
    model: &str,
    cache_dir: &Path,
    download: DownloadPolicy,
    autosize: Option<(AutoSizeProfile, f32)>,
) -> Result<Resolved> {
    // 1. Curated GGUF alias. Resolve this before the general HF alias table:
    // a GGUF alias names one exact file, not a safetensors repository.
    if let Some((repo, filename)) = resolve_gguf_alias(model) {
        let token = (download == DownloadPolicy::AutoDownload)
            .then(|| {
                std::env::var("HF_TOKEN")
                    .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
                    .ok()
            })
            .flatten();
        let (local_path, from_cache) = match find_cached_gguf(cache_dir, &repo, &filename) {
            Some(path) => (path, true),
            None if download == DownloadPolicy::AutoDownload => {
                let downloader =
                    ferrum_models::HfDownloader::new(cache_dir.to_path_buf(), token.clone())?;
                (
                    downloader.download_gguf(&repo, None, &filename).await?,
                    false,
                )
            }
            None => {
                return Err(FerrumError::model(format!(
                    "GGUF alias '{model}' is not cached and DownloadPolicy::NoDownload is set"
                )))
            }
        };
        if download == DownloadPolicy::AutoDownload {
            ensure_gguf_tokenizer_sidecars(cache_dir, token, &repo, &local_path).await?;
        }
        return Ok(finalize_resolution(
            ResolvedModelSource {
                original: model.to_string(),
                local_path,
                format: ModelFormat::GGUF,
                from_cache,
            },
            autosize,
        ));
    }

    // 2. GGUF file path.
    if looks_like_gguf_path(model) {
        let local_path = PathBuf::from(model);
        return Ok(finalize_resolution(
            ResolvedModelSource {
                original: model.to_string(),
                local_path,
                format: ModelFormat::GGUF,
                from_cache: false,
            },
            autosize,
        ));
    }

    // 3. Local directory.
    let direct = PathBuf::from(model);
    if direct.is_dir() {
        let format = detect_format(&direct);
        if format != ModelFormat::Unknown {
            return Ok(finalize_resolution(
                ResolvedModelSource {
                    original: model.to_string(),
                    local_path: direct,
                    format,
                    from_cache: false,
                },
                autosize,
            ));
        }
    }

    // 4. HF cache hit.
    let model_id = resolve_model_alias(model);
    if let Some(source) = find_cached_model(cache_dir, &model_id) {
        return Ok(finalize_resolution(source, autosize));
    }

    // 5. HF download.
    if download != DownloadPolicy::AutoDownload {
        return Err(FerrumError::model(format!(
            "model '{}' not found locally and DownloadPolicy::NoDownload set",
            model_id
        )));
    }

    let token = std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok();
    let downloader = ferrum_models::HfDownloader::new(cache_dir.to_path_buf(), token)?;
    let snapshot_path = downloader.download(&model_id, None).await?;
    let format = detect_format(&snapshot_path);
    if format == ModelFormat::Unknown {
        return Err(FerrumError::model(
            "downloaded model has unknown format (no safetensors / pytorch_model.bin)",
        ));
    }
    Ok(finalize_resolution(
        ResolvedModelSource {
            original: model_id,
            local_path: snapshot_path,
            format,
            from_cache: false,
        },
        autosize,
    ))
}

async fn ensure_gguf_tokenizer_sidecars(
    cache_dir: &Path,
    token: Option<String>,
    gguf_repo: &str,
    gguf_path: &Path,
) -> Result<()> {
    const SIDECAR_FILES: [&str; 6] = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.json",
        "chat_template.jinja",
        "generation_config.json",
    ];

    let snapshot_dir = gguf_path
        .parent()
        .ok_or_else(|| FerrumError::model("resolved GGUF path has no snapshot directory"))?;
    if snapshot_dir.join("tokenizer.json").is_file() {
        return Ok(());
    }
    let sibling = tokenizer_sibling_repo(gguf_repo).ok_or_else(|| {
        FerrumError::model(format!(
            "GGUF repository '{gguf_repo}' has no tokenizer sidecar source"
        ))
    })?;
    let downloader = ferrum_models::HfDownloader::new(cache_dir.to_path_buf(), token)?;
    let sibling_dir = downloader
        .download_sidecar_files(&sibling, None, &SIDECAR_FILES)
        .await?;
    for filename in SIDECAR_FILES {
        let source = sibling_dir.join(filename);
        if source.is_file() {
            std::fs::copy(&source, snapshot_dir.join(filename)).map_err(|error| {
                FerrumError::model(format!(
                    "copy GGUF tokenizer sidecar {filename} from {sibling}: {error}"
                ))
            })?;
        }
    }
    if !snapshot_dir.join("tokenizer.json").is_file() {
        return Err(FerrumError::model(format!(
            "GGUF tokenizer source '{sibling}' did not provide tokenizer.json"
        )));
    }
    Ok(())
}

fn finalize_resolution(
    source: ResolvedModelSource,
    autosize: Option<(AutoSizeProfile, f32)>,
) -> Resolved {
    let autosized = if let Some((profile, gpu_util)) = autosize {
        apply_auto_size_with_profile(&source.local_path, gpu_util, profile);
        if profile == AutoSizeProfile::Chat {
            apply_chat_profile_env(&source.local_path);
        }
        true
    } else {
        false
    };
    Resolved { source, autosized }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_model_dir(name: &str, config_json: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "ferrum-source-resolver-{name}-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config_json).unwrap();
        dir
    }

    fn value(entries: &[RuntimeConfigEntry], key: &str) -> Option<String> {
        entries
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| entry.effective_value.clone())
    }

    #[test]
    fn hf_and_gguf_aliases_are_disjoint() {
        for entry in GGUF_ALIASES {
            for alias in entry.aliases {
                assert_eq!(
                    resolve_model_alias(alias),
                    *alias,
                    "alias '{alias}' resolves to both an HF repository and a GGUF file"
                );
            }
        }
        assert_eq!(resolve_model_alias("qwen3:1.7b"), "Qwen/Qwen3-1.7B");
        assert!(resolve_gguf_alias("qwen3:1.7b").is_none());
        assert!(resolve_gguf_alias("qwen3:1.7b-gguf").is_some());
    }

    #[tokio::test]
    async fn resolves_local_model_directory_with_stable_product_id() {
        let dir = temp_model_dir(
            "local-product-id",
            r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5"}"#,
        );
        std::fs::write(dir.join("model.safetensors"), []).unwrap();

        let resolved = resolve_model_source(
            dir.to_str().unwrap(),
            &dir.join("unused-cache"),
            DownloadPolicy::NoDownload,
            None,
        )
        .await
        .unwrap();

        assert_eq!(resolved.source.local_path, dir);
        assert_eq!(resolved.source.format, ModelFormat::SafeTensors);
        assert!(!resolved.source.from_cache);
        assert_eq!(
            public_model_id(&resolved.source),
            dir.file_name().unwrap().to_string_lossy()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn resolves_direct_gguf_with_file_stem_product_id() {
        let dir = temp_model_dir("direct-gguf", r#"{}"#);
        let gguf = dir.join("Qwen3.5-4B-Instruct-Q4_K_M.gguf");
        std::fs::write(&gguf, []).unwrap();

        let resolved = resolve_model_source(
            gguf.to_str().unwrap(),
            &dir.join("unused-cache"),
            DownloadPolicy::NoDownload,
            None,
        )
        .await
        .unwrap();

        assert_eq!(resolved.source.local_path, gguf);
        assert_eq!(resolved.source.format, ModelFormat::GGUF);
        assert_eq!(
            public_model_id(&resolved.source),
            "Qwen3.5-4B-Instruct-Q4_K_M"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn resolves_cached_gguf_alias_without_entrypoint_short_circuit() {
        let cache = temp_model_dir("cached-gguf-alias", r#"{}"#);
        let (repo, filename) = resolve_gguf_alias("qwen3:4b-q4_k_m").unwrap();
        let repo_dir = cache
            .join("hub")
            .join(format!("models--{}", repo.replace('/', "--")));
        let revision = "fixture-revision";
        let snapshot = repo_dir.join("snapshots").join(revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::write(repo_dir.join("refs/main"), revision).unwrap();
        let gguf = snapshot.join(&filename);
        std::fs::write(&gguf, []).unwrap();

        let resolved =
            resolve_model_source("qwen3:4b-q4_k_m", &cache, DownloadPolicy::NoDownload, None)
                .await
                .unwrap();

        assert_eq!(resolved.source.local_path, gguf);
        assert_eq!(resolved.source.format, ModelFormat::GGUF);
        assert!(resolved.source.from_cache);
        assert_eq!(
            public_model_id(&resolved.source),
            Path::new(&filename).file_stem().unwrap().to_string_lossy()
        );
        let _ = std::fs::remove_dir_all(cache);
    }

    #[test]
    fn serve_profile_defaults_metal_gguf_moe_without_user_env() {
        let entries = serve_profile_runtime_entries_for_arch(
            true,
            true,
            true,
            &RuntimeConfigSnapshot::default(),
            RuntimeConfigSource::Default,
        );

        assert_eq!(
            value(&entries, "FERRUM_KV_CAPACITY").as_deref(),
            Some("1024")
        );
        assert_eq!(
            value(&entries, "FERRUM_PAGED_MAX_SEQS").as_deref(),
            Some("16")
        );
        assert_eq!(
            value(&entries, "FERRUM_METAL_PAGED_KV").as_deref(),
            Some("1")
        );
        assert_eq!(value(&entries, "FERRUM_PAGED_KV").as_deref(), Some("1"));
        assert_eq!(value(&entries, "FERRUM_MAX_BATCH").as_deref(), Some("16"));
        assert_eq!(value(&entries, "FERRUM_MOE_BATCHED").as_deref(), Some("1"));
        assert_eq!(
            value(&entries, "FERRUM_MOE_BATCHED_DECODE").as_deref(),
            Some("1")
        );
    }

    #[test]
    fn serve_profile_keeps_multi_seq_default_for_non_metal_moe() {
        let entries = serve_profile_runtime_entries_for_arch(
            true,
            true,
            false,
            &RuntimeConfigSnapshot::default(),
            RuntimeConfigSource::Default,
        );

        assert_eq!(
            value(&entries, "FERRUM_KV_CAPACITY").as_deref(),
            Some("2048")
        );
        assert_eq!(
            value(&entries, "FERRUM_PAGED_MAX_SEQS").as_deref(),
            Some("16")
        );
        assert_eq!(
            value(&entries, "FERRUM_METAL_PAGED_KV").as_deref(),
            Some("1")
        );
        assert_eq!(value(&entries, "FERRUM_PAGED_KV").as_deref(), Some("1"));
        assert_eq!(value(&entries, "FERRUM_MAX_BATCH").as_deref(), Some("16"));
        assert_eq!(value(&entries, "FERRUM_MOE_BATCHED").as_deref(), Some("1"));
        assert_eq!(
            value(&entries, "FERRUM_MOE_BATCHED_DECODE").as_deref(),
            Some("1")
        );
    }

    #[test]
    fn serve_profile_defaults_gguf_dense_without_moe_env() {
        let entries = serve_profile_runtime_entries_for_arch(
            true,
            false,
            true,
            &RuntimeConfigSnapshot::default(),
            RuntimeConfigSource::Default,
        );

        assert_eq!(
            value(&entries, "FERRUM_KV_CAPACITY").as_deref(),
            Some("512")
        );
        assert_eq!(
            value(&entries, "FERRUM_PAGED_MAX_SEQS").as_deref(),
            Some("16")
        );
        assert_eq!(
            value(&entries, "FERRUM_METAL_PAGED_KV").as_deref(),
            Some("1")
        );
        assert_eq!(value(&entries, "FERRUM_PAGED_KV").as_deref(), Some("1"));
        assert_eq!(value(&entries, "FERRUM_MOE_BATCHED"), None);
    }

    #[test]
    fn serve_profile_respects_explicit_user_env() {
        let current = RuntimeConfigSnapshot::from_entries(vec![RuntimeConfigEntry::new(
            "FERRUM_KV_CAPACITY",
            "4096",
            RuntimeConfigSource::Default,
        )]);
        let entries = serve_profile_runtime_entries_for_arch(
            true,
            true,
            true,
            &current,
            RuntimeConfigSource::Default,
        );

        assert_eq!(value(&entries, "FERRUM_KV_CAPACITY"), None);
        assert_eq!(
            value(&entries, "FERRUM_PAGED_MAX_SEQS").as_deref(),
            Some("16")
        );
    }

    #[test]
    fn chat_profile_defaults_dense_safetensors_as_typed_entries() {
        let dir = temp_model_dir(
            "dense",
            r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}"#,
        );
        let entries = chat_profile_runtime_entries(
            &dir,
            &RuntimeConfigSnapshot::default(),
            RuntimeConfigSource::Default,
        );

        assert_eq!(
            value(&entries, "FERRUM_KV_CAPACITY").as_deref(),
            Some("8192")
        );
        assert_eq!(
            value(&entries, "FERRUM_METAL_PAGED_KV").as_deref(),
            Some("1")
        );
        assert_eq!(value(&entries, "FERRUM_PAGED_KV").as_deref(), Some("1"));
        assert_eq!(
            value(&entries, "FERRUM_PAGED_MAX_SEQS").as_deref(),
            Some("2")
        );
        assert_eq!(value(&entries, "FERRUM_MAX_BATCH").as_deref(), Some("1"));
        assert_eq!(value(&entries, "FERRUM_MOE_BATCHED"), None);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_profile_recognizes_qwen35_dense_without_qwen3_fallback() {
        let dir = temp_model_dir(
            "qwen35_dense",
            r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5"}"#,
        );
        assert_eq!(detect_model_family(&dir).as_deref(), Some("qwen3_5"));
        assert!(!detect_moe_arch(&dir));

        let entries = chat_profile_runtime_entries(
            &dir,
            &RuntimeConfigSnapshot::default(),
            RuntimeConfigSource::Default,
        );

        assert_eq!(
            value(&entries, "FERRUM_METAL_PAGED_KV").as_deref(),
            Some("1")
        );
        assert_eq!(value(&entries, "FERRUM_PAGED_KV").as_deref(), Some("1"));
        assert_eq!(value(&entries, "FERRUM_MOE_BATCHED"), None);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_profile_disables_metal_paged_kv_for_llama_safetensors() {
        let dir = temp_model_dir(
            "llama",
            r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#,
        );
        let entries = chat_profile_runtime_entries(
            &dir,
            &RuntimeConfigSnapshot::default(),
            RuntimeConfigSource::Default,
        );

        assert_eq!(
            value(&entries, "FERRUM_METAL_PAGED_KV").as_deref(),
            Some("0")
        );
        assert_eq!(value(&entries, "FERRUM_PAGED_KV").as_deref(), Some("0"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_profile_disables_metal_paged_kv_for_qwen2_safetensors() {
        let dir = temp_model_dir(
            "qwen2",
            r#"{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2"}"#,
        );
        let entries = chat_profile_runtime_entries(
            &dir,
            &RuntimeConfigSnapshot::default(),
            RuntimeConfigSource::Default,
        );

        assert_eq!(
            value(&entries, "FERRUM_METAL_PAGED_KV").as_deref(),
            Some("0")
        );
        assert_eq!(value(&entries, "FERRUM_PAGED_KV").as_deref(), Some("0"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_profile_defaults_moe_safetensors_as_typed_entries() {
        let dir = temp_model_dir(
            "moe",
            r#"{"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe"}"#,
        );
        let entries = chat_profile_runtime_entries(
            &dir,
            &RuntimeConfigSnapshot::default(),
            RuntimeConfigSource::Default,
        );

        assert_eq!(
            value(&entries, "FERRUM_KV_CAPACITY").as_deref(),
            Some("4096")
        );
        assert_eq!(
            value(&entries, "FERRUM_PAGED_MAX_SEQS").as_deref(),
            Some("1")
        );
        assert_eq!(
            value(&entries, "FERRUM_METAL_PAGED_KV").as_deref(),
            Some("1")
        );
        assert_eq!(value(&entries, "FERRUM_PAGED_KV").as_deref(), Some("1"));
        assert_eq!(value(&entries, "FERRUM_MOE_BATCHED").as_deref(), Some("0"));
        assert_eq!(
            value(&entries, "FERRUM_MOE_BATCHED_DECODE").as_deref(),
            Some("0")
        );
        assert_eq!(
            value(&entries, "FERRUM_MOE_BATCH_THRESHOLD").as_deref(),
            Some("2")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_profile_recognizes_qwen35_moe_as_distinct_moe_family() {
        let dir = temp_model_dir(
            "qwen35_moe",
            r#"{"architectures":["Qwen3_5MoeForConditionalGeneration"],"model_type":"qwen3_5_moe"}"#,
        );
        assert_eq!(detect_model_family(&dir).as_deref(), Some("qwen3_5_moe"));
        assert!(detect_moe_arch(&dir));

        let entries = chat_profile_runtime_entries(
            &dir,
            &RuntimeConfigSnapshot::default(),
            RuntimeConfigSource::Default,
        );

        assert_eq!(
            value(&entries, "FERRUM_KV_CAPACITY").as_deref(),
            Some("4096")
        );
        assert_eq!(value(&entries, "FERRUM_MOE_BATCHED").as_deref(), Some("0"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn chat_profile_defaults_moe_gguf_as_typed_entries() {
        let entries = chat_profile_runtime_entries_for_arch(
            true,
            true,
            Some("qwen3_moe"),
            &RuntimeConfigSnapshot::default(),
            RuntimeConfigSource::Default,
        );

        assert_eq!(
            value(&entries, "FERRUM_KV_CAPACITY").as_deref(),
            Some("4096")
        );
        assert_eq!(
            value(&entries, "FERRUM_PAGED_MAX_SEQS").as_deref(),
            Some("1")
        );
        assert_eq!(
            value(&entries, "FERRUM_METAL_PAGED_KV").as_deref(),
            Some("0")
        );
        assert_eq!(value(&entries, "FERRUM_PAGED_KV").as_deref(), Some("0"));
        assert_eq!(value(&entries, "FERRUM_MOE_BATCHED").as_deref(), Some("0"));
        assert_eq!(
            value(&entries, "FERRUM_MOE_BATCHED_DECODE").as_deref(),
            Some("0")
        );
    }

    #[test]
    fn chat_profile_defaults_preserve_existing_snapshot_values() {
        let dir = temp_model_dir(
            "override",
            r#"{"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe"}"#,
        );
        let current = RuntimeConfigSnapshot::from_entries([
            RuntimeConfigEntry::new("FERRUM_KV_CAPACITY", "1234", RuntimeConfigSource::Env),
            RuntimeConfigEntry::new("FERRUM_MOE_BATCH_THRESHOLD", "7", RuntimeConfigSource::Env),
        ]);
        let entries = chat_profile_runtime_entries(&dir, &current, RuntimeConfigSource::Default);

        assert_eq!(value(&entries, "FERRUM_KV_CAPACITY"), None);
        assert_eq!(value(&entries, "FERRUM_MOE_BATCH_THRESHOLD"), None);
        assert_eq!(value(&entries, "FERRUM_MOE_BATCHED").as_deref(), Some("0"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn load_model_chat_template_reads_tokenizer_config() {
        let dir = temp_model_dir(
            "template",
            r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}"#,
        );
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":"{{ messages[0].content }}","bos_token":"<s>","eos_token":"</s>"}"#,
        )
        .unwrap();

        let template = load_model_chat_template(&dir).unwrap();
        assert_eq!(template.template, "{{ messages[0].content }}");
        assert_eq!(template.bos_token.as_deref(), Some("<s>"));
        assert_eq!(template.eos_token.as_deref(), Some("</s>"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
