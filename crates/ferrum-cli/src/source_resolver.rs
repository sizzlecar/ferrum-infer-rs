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
use ferrum_types::{FerrumError, Result};

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

/// `ferrum run <gguf>` chat-profile env-var defaults. Sniffs the GGUF
/// arch (dense vs MoE) and sets:
///
///   - `FERRUM_KV_CAPACITY`     — 8192 dense / 512 MoE
///   - `FERRUM_METAL_PAGED_KV`  — 0 dense / 1 MoE (paged-KV is a hard
///     requirement for the Qwen3-MoE GPU dispatch path; without it
///     decode emits ~0.1 tok/s of garbage on Metal)
///   - `FERRUM_PAGED_MAX_SEQS=2`, `FERRUM_MAX_BATCH=1`,
///     `FERRUM_MOE_BATCHED=1`, `FERRUM_MOE_BATCHED_DECODE=1`,
///     `FERRUM_MOE_BATCH_THRESHOLD=2` — MoE only. Match the published
///     `docs/bench/macos-2026-05-02` c=1 30B-A3B → 42 tok/s defaults.
///
/// Idempotent: if a user explicitly sets one of these env vars before
/// invoking `ferrum run`, that value wins (we only set when unset).
/// Called automatically by `resolve_model_source` when the resolved
/// source is GGUF and the autosize profile is `Chat`. Server/bench
/// callers don't get these defaults — they don't fit the multi-turn
/// REPL pattern this profile is tuned for.
pub fn apply_gguf_chat_profile_env(gguf_path: &Path) {
    use ferrum_quantization::gguf::GgufFile;

    let is_moe = match GgufFile::open(gguf_path) {
        Ok(g) => g.architecture().map(|s| s.contains("moe")).unwrap_or(false),
        Err(_) => false,
    };

    // SAFETY: std::env::set_var is unsafe on Rust 2024+. We only call
    // this once at CLI startup before any other thread spawns.
    if std::env::var_os("FERRUM_KV_CAPACITY").is_none() {
        std::env::set_var("FERRUM_KV_CAPACITY", if is_moe { "512" } else { "8192" });
    }
    if std::env::var_os("FERRUM_METAL_PAGED_KV").is_none() {
        std::env::set_var("FERRUM_METAL_PAGED_KV", if is_moe { "1" } else { "0" });
    }
    if is_moe {
        for (k, v) in [
            ("FERRUM_PAGED_MAX_SEQS", "2"),
            ("FERRUM_MAX_BATCH", "1"),
            ("FERRUM_MOE_BATCHED", "1"),
            ("FERRUM_MOE_BATCHED_DECODE", "1"),
            ("FERRUM_MOE_BATCH_THRESHOLD", "2"),
        ] {
            if std::env::var_os(k).is_none() {
                std::env::set_var(k, v);
            }
        }
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
            // Chat profile + GGUF: also set the run-mode KV / MoE
            // env-var defaults that `run_gguf_one_shot` used to set
            // inline. Server / bench callers (profile=Server or
            // autosize=None) don't get these.
            if profile == AutoSizeProfile::Chat {
                apply_gguf_chat_profile_env(&local_path);
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
