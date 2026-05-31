//! CLI-level model source resolution.
//!
//! Centralises the lookup chain that `run` / `serve` / `bench` were
//! reinventing each in their own copy:
//!
//!   1. **GGUF file path** — if the user passed an existing `*.gguf` file,
//!      build a [`ResolvedModelSource`] directly without HF lookup.
//!   2. **Local model dir** — if the path is an existing directory with
//!      `config.json` + weights, treat it as a direct source.
//!   3. **HF cache hit** — `~/.cache/huggingface/hub/models--<owner>--<repo>/snapshots/<rev>`.
//!   4. **HF download** — fall back to [`HfDownloader`] (`run` / `serve`
//!      only; `bench` callers may opt out).
//!   5. **GPU-memory autosizing** — for GPU backends, run the chat
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
use ferrum_types::{
    FerrumError, Result, RuntimeConfigEntry, RuntimeConfigSnapshot, RuntimeConfigSource,
};

use crate::gpu_mem_autosize::{apply_auto_size_with_profile, AutoSizeProfile};

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

/// Chat-profile runtime defaults for `ferrum run`. Sniffs the arch
/// (dense vs MoE — works for both GGUF files and safetensors snapshot
/// dirs) and materializes missing compatibility env vars for:
///
///   - `FERRUM_KV_CAPACITY`     — 8192 dense / 512 MoE
///   - `FERRUM_METAL_PAGED_KV`  — 0 dense GGUF / 1 dense safetensors / 1 MoE.
///     Paged-KV is a hard requirement for the Qwen3-MoE GPU dispatch path
///     (without it decode emits ~0.1 tok/s of garbage on Metal). Dense
///     safetensors *also* requires paged on Metal: the contiguous-KV
///     attention path produces token noise from the very first decode step
///     for safetensors-loaded Qwen3. Dense GGUF (candle-transformers
///     loader) works on the contig path and keeps `0` to avoid allocating
///     a pool it doesn't need.
///   - `FERRUM_PAGED_MAX_SEQS=2`, `FERRUM_MAX_BATCH=1` — single-user REPL.
///     Keeps the paged pool at ~1.7 GB for `cap=8192` dense; without this
///     cap the default `max_seqs=32` makes the pool ~30 GB on a 32 GB Mac.
///   - `FERRUM_MOE_BATCHED=1`, `FERRUM_MOE_BATCHED_DECODE=1`,
///     `FERRUM_MOE_BATCH_THRESHOLD=2` — MoE only. Match the published
///     `docs/bench/macos-2026-05-02` c=1 30B-A3B → 42 tok/s defaults.
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
    let mut entries = Vec::new();

    push_missing_entry(
        &mut entries,
        current,
        "FERRUM_KV_CAPACITY",
        if is_moe { "512" } else { "8192" },
        source,
    );
    // Dense GGUF: contig path works, no need to allocate the pool.
    // Dense safetensors + MoE: paged required for correctness on Metal.
    let need_paged = is_moe || !is_gguf;
    push_missing_entry(
        &mut entries,
        current,
        "FERRUM_METAL_PAGED_KV",
        if need_paged { "1" } else { "0" },
        source,
    );
    // Single-user REPL pool sizing — both safetensors dense (avoids
    // 30 GB pool at default max_seqs=32) and MoE (per published bench
    // tuning). Skipped for GGUF dense which doesn't allocate a pool.
    if !is_gguf || is_moe {
        for (k, v) in [("FERRUM_PAGED_MAX_SEQS", "2"), ("FERRUM_MAX_BATCH", "1")] {
            push_missing_entry(&mut entries, current, k, v, source);
        }
    }
    if is_moe {
        for (k, v) in [
            ("FERRUM_MOE_BATCHED", "1"),
            ("FERRUM_MOE_BATCHED_DECODE", "1"),
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
fn detect_moe_arch(path: &Path) -> bool {
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
///   1. `*.gguf` file → direct GGUF source.
///   2. Existing local directory with valid weights → direct source.
///   3. HF cache hit → cached source.
///   4. (if `AutoDownload`) HF download → cached source.
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
    // 1. GGUF file path.
    if looks_like_gguf_path(model) {
        let local_path = PathBuf::from(model);
        let mut autosized = false;
        if let Some((profile, gpu_util)) = autosize {
            apply_auto_size_with_profile(&local_path, gpu_util, profile);
            autosized = true;
            if profile == AutoSizeProfile::Chat {
                apply_chat_profile_env(&local_path);
            }
        }
        return Ok(Resolved {
            source: ResolvedModelSource {
                original: model.to_string(),
                local_path,
                format: ModelFormat::GGUF,
                from_cache: false,
            },
            autosized,
        });
    }

    // 2. Local directory.
    let direct = PathBuf::from(model);
    if direct.is_dir() {
        let format = detect_format(&direct);
        if format != ModelFormat::Unknown {
            let mut autosized = false;
            if let Some((profile, gpu_util)) = autosize {
                apply_auto_size_with_profile(&direct, gpu_util, profile);
                autosized = true;
                if profile == AutoSizeProfile::Chat {
                    apply_chat_profile_env(&direct);
                }
            }
            return Ok(Resolved {
                source: ResolvedModelSource {
                    original: model.to_string(),
                    local_path: direct,
                    format,
                    from_cache: false,
                },
                autosized,
            });
        }
    }

    // 3. HF cache hit.
    let model_id = super::commands::run::resolve_model_alias(model);
    if let Some(source) = find_cached_model(cache_dir, &model_id) {
        let mut autosized = false;
        if let Some((profile, gpu_util)) = autosize {
            apply_auto_size_with_profile(&source.local_path, gpu_util, profile);
            autosized = true;
            if profile == AutoSizeProfile::Chat {
                apply_chat_profile_env(&source.local_path);
            }
        }
        return Ok(Resolved { source, autosized });
    }

    // 4. HF download.
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
    let mut autosized = false;
    if let Some((profile, gpu_util)) = autosize {
        apply_auto_size_with_profile(&snapshot_path, gpu_util, profile);
        autosized = true;
        if profile == AutoSizeProfile::Chat {
            apply_chat_profile_env(&snapshot_path);
        }
    }
    Ok(Resolved {
        source: ResolvedModelSource {
            original: model_id,
            local_path: snapshot_path,
            format,
            from_cache: false,
        },
        autosized,
    })
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
        assert_eq!(
            value(&entries, "FERRUM_PAGED_MAX_SEQS").as_deref(),
            Some("2")
        );
        assert_eq!(value(&entries, "FERRUM_MAX_BATCH").as_deref(), Some("1"));
        assert_eq!(value(&entries, "FERRUM_MOE_BATCHED"), None);
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
            Some("512")
        );
        assert_eq!(
            value(&entries, "FERRUM_METAL_PAGED_KV").as_deref(),
            Some("1")
        );
        assert_eq!(value(&entries, "FERRUM_MOE_BATCHED").as_deref(), Some("1"));
        assert_eq!(
            value(&entries, "FERRUM_MOE_BATCH_THRESHOLD").as_deref(),
            Some("2")
        );
        let _ = std::fs::remove_dir_all(dir);
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
        assert_eq!(value(&entries, "FERRUM_MOE_BATCHED").as_deref(), Some("1"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
