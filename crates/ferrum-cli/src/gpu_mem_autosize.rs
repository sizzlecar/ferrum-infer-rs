//! GPU memory auto-tuning for the KV pool.
//!
//! Reads model config + on-disk weight file sizes + nvidia-smi reported
//! GPU total, then sets `FERRUM_KV_MAX_BLOCKS` so the KV pool fits inside
//! `total_mem * gpu_memory_utilization` after weights and a scratch reserve.
//! Mirrors vLLM's `gpu_memory_utilization` knob (default 0.9).
//!
//! Skipped when:
//! - nvidia-smi missing (Mac / CPU-only): keep static defaults.
//! - `config.json` not parseable: keep static defaults.
//! - User explicitly set `FERRUM_KV_MAX_BLOCKS`: respect their override.

use std::path::Path;

/// Bytes reserved for everything that's NOT weights or KV pool: cuBLAS
/// workspace, Marlin gather scratch, unified path scratch, embedding,
/// lm_head logits buffer, runtime allocator overhead. 4 GB is the
/// observed worst case at c=32 + chunked-prefill mixed batches.
const SCRATCH_RESERVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// PagedKvPool block_size — must match `PAGED_BLOCK_SIZE` in
/// `llama_family.rs`.
const PAGED_BLOCK_SIZE: u64 = 16;

/// Bytes per element of the KV cache. ferrum currently always uses FP16
/// for KV regardless of weight dtype (Marlin INT4 weights → FP16 KV).
const KV_DTYPE_BYTES: u64 = 2;

#[derive(Debug)]
pub struct AutoSizeResult {
    pub total_gpu_bytes: u64,
    pub free_gpu_bytes: u64,
    pub weight_bytes: u64,
    pub kv_block_bytes: u64,
    pub max_blocks: usize,
    pub reserved_for_scratch: u64,
}

impl AutoSizeResult {
    pub fn print_summary(&self) {
        let gb = |b: u64| (b as f64) / 1024.0 / 1024.0 / 1024.0;
        eprintln!(
            "[auto-size] gpu={:.1} GB total / {:.1} GB free | weights={:.1} GB | scratch reserve={:.1} GB | KV pool budget {:.1} GB → max_blocks={}",
            gb(self.total_gpu_bytes),
            gb(self.free_gpu_bytes),
            gb(self.weight_bytes),
            gb(self.reserved_for_scratch),
            gb((self.max_blocks as u64) * self.kv_block_bytes),
            self.max_blocks,
        );
    }
}

/// Compute target `FERRUM_KV_MAX_BLOCKS` from `gpu_memory_utilization`.
///
/// Returns None when any input is unavailable — caller leaves the
/// static default in place.
pub fn auto_size_kv_blocks(model_dir: &Path, gpu_util: f32) -> Option<AutoSizeResult> {
    let gpu_util = gpu_util.clamp(0.1, 1.0);

    // 1. Query GPU total + free via nvidia-smi (most portable across
    //    cudarc versions and works pre-cuda-driver-load).
    let nvsmi = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !nvsmi.status.success() {
        return None;
    }
    let s = String::from_utf8(nvsmi.stdout).ok()?;
    let line = s.lines().next()?.trim();
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    let total_mb: u64 = parts.first()?.parse().ok()?;
    let free_mb: u64 = parts.get(1)?.parse().ok()?;
    let total_bytes = total_mb * 1024 * 1024;
    let free_bytes = free_mb * 1024 * 1024;

    // 2. Parse config.json for model dims.
    let config_path = model_dir.join("config.json");
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path).ok()?).ok()?;
    let num_layers = config["num_hidden_layers"]
        .as_u64()
        .or_else(|| config["num_layers"].as_u64())?;
    let hidden_size = config["hidden_size"].as_u64()?;
    let num_attn_heads = config["num_attention_heads"].as_u64()?;
    let num_kv_heads = config["num_key_value_heads"]
        .as_u64()
        .unwrap_or(num_attn_heads);
    let head_dim = config["head_dim"]
        .as_u64()
        .unwrap_or_else(|| hidden_size / num_attn_heads.max(1));

    // 3. Sum .safetensors / .bin file sizes for weight estimate.
    let mut weight_bytes: u64 = 0;
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            let is_weight = p
                .extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext == "safetensors" || ext == "bin")
                .unwrap_or(false);
            if is_weight {
                if let Ok(meta) = entry.metadata() {
                    weight_bytes += meta.len();
                }
            }
        }
    }
    if weight_bytes == 0 {
        // Couldn't find weights — bail to static defaults.
        return None;
    }

    // 4. Compute KV budget. Reserve `(1 - util)` of total mem as host-
    //    bookkeeping margin, plus a fixed scratch reserve covering all
    //    transient buffers (cuBLAS workspace, Marlin scratch, unified
    //    forward intermediates, embedding, lm_head, etc).
    let target_used = (total_bytes as f64 * gpu_util as f64) as u64;
    let avail_for_kv = target_used
        .saturating_sub(weight_bytes)
        .saturating_sub(SCRATCH_RESERVE_BYTES);

    // KV per block: num_layers × num_kv_heads × block_size × head_dim
    //              × 2 (K and V) × dtype_bytes
    let block_bytes = num_layers * num_kv_heads * PAGED_BLOCK_SIZE * head_dim * 2 * KV_DTYPE_BYTES;
    if block_bytes == 0 {
        return None;
    }
    let max_blocks = (avail_for_kv / block_bytes) as usize;

    Some(AutoSizeResult {
        total_gpu_bytes: total_bytes,
        free_gpu_bytes: free_bytes,
        weight_bytes,
        kv_block_bytes: block_bytes,
        max_blocks,
        reserved_for_scratch: SCRATCH_RESERVE_BYTES,
    })
}

/// CLI usage profile: which presets the autosizer should consider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoSizeProfile {
    /// `ferrum serve` — many concurrent requests. Prioritise batch
    /// width (max_seqs) over per-seq context length. Bench scripts
    /// also use this profile.
    Server,
    /// `ferrum run` — single interactive user, multi-turn chat. Trade
    /// max_seqs for KV_CAPACITY so a long conversation doesn't crash
    /// after a few turns. With KV_CAPACITY=512 (server default), Qwen3
    /// thinking-mode replies hit overflow at the 4th turn.
    Chat,
}

/// Apply auto-sizing: read CLI flag, query nvidia-smi, set env vars.
/// Sets BOTH `FERRUM_KV_MAX_BLOCKS` (engine-side BlockPool) and
/// `FERRUM_PAGED_MAX_SEQS` (model-side paged_pools — sizes the GPU
/// KV pool as `max_seqs × max_blocks_per_seq`). Without bounding
/// max_seqs the GPU pool can grow far past the engine pool budget
/// — e.g. on Llama-3.1-8B FP16 with PAGED_MAX_SEQS=64 +
/// KV_CAPACITY=2048 the pool is 16 GB by itself, OOMing weight load
/// on a 24 GB card.
///
/// Idempotent — caller invokes once per CLI invocation before engine
/// init. Respects user overrides (no clobber if env already set).
///
/// Defaults to `AutoSizeProfile::Server`. Use `apply_auto_size_with_profile`
/// for chat (`ferrum run`) — it picks longer per-seq context.
pub fn apply_auto_size(model_dir: &Path, gpu_util: f32) {
    apply_auto_size_with_profile(model_dir, gpu_util, AutoSizeProfile::Server);
}

/// Apply auto-sizing with explicit usage profile. The chat profile
/// flips priority — long context per seq beats wide batch — because
/// the CLI REPL only ever has one active sequence and multi-turn
/// dialogues blow past the default 512-token cap fast.
pub fn apply_auto_size_with_profile(model_dir: &Path, gpu_util: f32, profile: AutoSizeProfile) {
    let kv_overridden = std::env::var("FERRUM_KV_MAX_BLOCKS").is_ok();
    let max_seqs_overridden = std::env::var("FERRUM_PAGED_MAX_SEQS").is_ok();
    if kv_overridden && max_seqs_overridden {
        return;
    }
    let Some(result) = auto_size_kv_blocks(model_dir, gpu_util) else {
        return;
    };
    result.print_summary();

    // Pool sizing: pool_blocks = max_seqs × (KV_CAPACITY / 16).
    // Picks the largest (max_seqs, KV_CAP) tuple from a fixed ladder
    // whose pool fits the budget.
    //   Server: prioritise wide batch (c=16/32) over context.
    //   Chat:   single user, multi-turn — pile budget into context.
    const SERVER_PRESETS: &[(usize, usize)] = &[
        // (max_seqs, KV_CAPACITY tokens)
        (32, 2048), // best — INT4 8B on 24 GB usually lands here
        (32, 1024), // FP16 8B at util=1.0
        (32, 512),  // FP16 8B at util=0.95
        (16, 2048), // long-context narrow batch
        (16, 1024),
        (16, 512), // last that supports c=16
        (8, 1024), // <c=16 only
        (8, 512),
    ];
    // 16384 fits a long thinking-mode reply + ~20 conversation turns
    // before /clear. max_seqs=2 leaves a slot for any internal use;
    // single-user CLI never needs more.
    const CHAT_PRESETS: &[(usize, usize)] = &[
        (2, 16384),
        (2, 8192),
        (2, 4096),
        (1, 16384),
        (1, 8192),
        (1, 4096),
        (1, 2048),
    ];
    let presets: &[(usize, usize)] = match profile {
        AutoSizeProfile::Server => SERVER_PRESETS,
        AutoSizeProfile::Chat => CHAT_PRESETS,
    };
    let (max_seqs_clamped, kv_capacity) = {
        let pick = presets.iter().copied().find(|(seqs, cap_tokens)| {
            let bps = (*cap_tokens / 16).max(1);
            seqs * bps <= result.max_blocks
        });
        if let Some((seqs, cap)) = pick {
            // Always override KV_CAPACITY explicitly so behaviour is
            // independent of the static default in llama_family.rs.
            (seqs, cap)
        } else {
            // Budget too tight for any preset — try min config.
            let bps = (result.max_blocks / 8).max(8);
            (8, bps * 16)
        }
    };

    let kv_capacity_overridden = std::env::var("FERRUM_KV_CAPACITY").is_ok();
    let max_batched_tokens_overridden = std::env::var("FERRUM_MAX_BATCHED_TOKENS").is_ok();
    // Phase 3 token budget. Server prioritises cohort prefill width
    // (4096 covers 32 × ~128-token prompts in one iter — the apples
    // workload — and the Qwen3MoE unified path activates because
    // m_total ≤ scratch.max_tokens). Chat is single-user with long
    // turns, no batch width to spend on; 2048 is plenty and keeps the
    // scratch (most notably `batch_logits` = t × vocab × 2 B) small.
    let max_batched_tokens = match profile {
        AutoSizeProfile::Server => 4096,
        AutoSizeProfile::Chat => 2048,
    };
    // SAFETY: set_var is unsafe on Rust 2024; runs once before threads spawn.
    unsafe {
        if !kv_overridden {
            std::env::set_var("FERRUM_KV_MAX_BLOCKS", result.max_blocks.to_string());
        }
        if !max_seqs_overridden {
            std::env::set_var("FERRUM_PAGED_MAX_SEQS", max_seqs_clamped.to_string());
        }
        if kv_capacity > 0 && !kv_capacity_overridden {
            std::env::set_var("FERRUM_KV_CAPACITY", kv_capacity.to_string());
        }
        if !max_batched_tokens_overridden {
            std::env::set_var("FERRUM_MAX_BATCHED_TOKENS", max_batched_tokens.to_string());
        }
    }
    eprintln!(
        "[auto-size] KV_MAX_BLOCKS={} PAGED_MAX_SEQS={} KV_CAPACITY={} MAX_BATCHED_TOKENS={}",
        if kv_overridden {
            "<user>".to_string()
        } else {
            result.max_blocks.to_string()
        },
        if max_seqs_overridden {
            "<user>".to_string()
        } else {
            max_seqs_clamped.to_string()
        },
        if kv_capacity_overridden {
            "<user>".to_string()
        } else if kv_capacity > 0 {
            kv_capacity.to_string()
        } else {
            "<default>".to_string()
        },
        if max_batched_tokens_overridden {
            "<user>".to_string()
        } else {
            max_batched_tokens.to_string()
        },
    );
}
