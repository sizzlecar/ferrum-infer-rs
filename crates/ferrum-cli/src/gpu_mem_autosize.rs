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
    let block_bytes =
        num_layers * num_kv_heads * PAGED_BLOCK_SIZE * head_dim * 2 * KV_DTYPE_BYTES;
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

/// Apply auto-sizing: read CLI flag, query nvidia-smi, set env var.
/// Idempotent — caller invokes once per CLI invocation before engine init.
/// Respects user-set FERRUM_KV_MAX_BLOCKS (no override).
pub fn apply_auto_size(model_dir: &Path, gpu_util: f32) {
    if std::env::var("FERRUM_KV_MAX_BLOCKS").is_ok() {
        // User explicitly set it — don't override.
        return;
    }
    if let Some(result) = auto_size_kv_blocks(model_dir, gpu_util) {
        result.print_summary();
        // SAFETY: set_var is unsafe on Rust 2024; runs once before threads spawn.
        unsafe {
            std::env::set_var("FERRUM_KV_MAX_BLOCKS", result.max_blocks.to_string());
        }
    }
}
