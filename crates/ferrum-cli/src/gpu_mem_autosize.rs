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

use ferrum_types::{RuntimeConfigEntry, RuntimeConfigSnapshot, RuntimeConfigSource};
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
    pub budgeted_weight_bytes: u64,
    pub weight_budget_shards: u64,
    pub budgeted_layer_count: u64,
    pub kv_block_bytes: u64,
    pub kv_pool_copies: u64,
    pub estimated_budget_blocks: usize,
    pub requested_min_blocks: usize,
    pub max_blocks: usize,
    pub reserved_for_scratch: u64,
}

impl AutoSizeResult {
    pub fn print_summary(&self) {
        let gb = |b: u64| (b as f64) / 1024.0 / 1024.0 / 1024.0;
        eprintln!(
            "[auto-size] gpu={:.1} GB total / {:.1} GB free | weights={:.1} GB budget / {:.1} GB total | layers={} budget | scratch reserve={:.1} GB | KV pool budget {:.1} GB → max_blocks={}",
            gb(self.total_gpu_bytes),
            gb(self.free_gpu_bytes),
            gb(self.budgeted_weight_bytes),
            gb(self.weight_bytes),
            self.budgeted_layer_count,
            gb(self.reserved_for_scratch),
            gb((self.max_blocks as u64) * self.kv_block_bytes * self.kv_pool_copies),
            self.max_blocks,
        );
        if self.weight_budget_shards > 1 {
            eprintln!(
                "[auto-size] weight budget shards={} (distributed strategy)",
                self.weight_budget_shards
            );
        }
        if self.kv_pool_copies > 1 {
            eprintln!(
                "[auto-size] KV pool copies={} (FA-compatible attention path)",
                self.kv_pool_copies
            );
        }
        if self.requested_min_blocks > self.estimated_budget_blocks {
            eprintln!(
                "[auto-size] requested runtime token floor requires KV_MAX_BLOCKS={} above estimated budget {}; honoring explicit runtime limits",
                self.requested_min_blocks, self.estimated_budget_blocks
            );
        }
    }
}

/// Compute target `FERRUM_KV_MAX_BLOCKS` from `gpu_memory_utilization`.
///
/// Returns None when any input is unavailable — caller leaves the
/// static default in place.
pub fn auto_size_kv_blocks(model_dir: &Path, gpu_util: f32) -> Option<AutoSizeResult> {
    auto_size_kv_blocks_with_pool_copies(model_dir, gpu_util, 1)
}

pub fn auto_size_kv_blocks_with_pool_copies(
    model_dir: &Path,
    gpu_util: f32,
    kv_pool_copies: u64,
) -> Option<AutoSizeResult> {
    let current = RuntimeConfigSnapshot::capture_current();
    auto_size_kv_blocks_with_pool_copies_for_snapshot(model_dir, gpu_util, kv_pool_copies, &current)
}

fn auto_size_kv_blocks_with_pool_copies_for_snapshot(
    model_dir: &Path,
    gpu_util: f32,
    kv_pool_copies: u64,
    runtime_config: &RuntimeConfigSnapshot,
) -> Option<AutoSizeResult> {
    let gpu_util = gpu_util.clamp(0.1, 1.0);
    let kv_pool_copies = kv_pool_copies.max(1);

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
                // HuggingFace snapshot files are commonly symlinks into the
                // blob cache. `DirEntry::metadata` reports the symlink itself;
                // `std::fs::metadata` follows it and gives the real shard size.
                if let Ok(meta) = std::fs::metadata(&p) {
                    weight_bytes += meta.len();
                }
            }
        }
    }
    if weight_bytes == 0 {
        // Couldn't find weights — bail to static defaults.
        return None;
    }
    let weight_budget_shards = weight_budget_shard_count(runtime_config);
    let budgeted_weight_bytes = ceil_div_u64(weight_bytes, weight_budget_shards);
    let budgeted_layer_count = layer_count_for_memory_budget(num_layers, runtime_config);

    // 4. Compute KV budget. Reserve `(1 - util)` of total mem as host-
    //    bookkeeping margin, plus a fixed scratch reserve covering all
    //    transient buffers (cuBLAS workspace, Marlin scratch, unified
    //    forward intermediates, embedding, lm_head, etc).
    let target_used = (total_bytes as f64 * gpu_util as f64) as u64;
    let avail_for_kv = target_used
        .saturating_sub(budgeted_weight_bytes)
        .saturating_sub(SCRATCH_RESERVE_BYTES);

    // KV per block: num_layers × num_kv_heads × block_size × head_dim
    //              × 2 (K and V) × dtype_bytes
    let block_bytes =
        budgeted_layer_count * num_kv_heads * PAGED_BLOCK_SIZE * head_dim * 2 * KV_DTYPE_BYTES;
    if block_bytes == 0 {
        return None;
    }
    let estimated_budget_blocks = (avail_for_kv / (block_bytes * kv_pool_copies)) as usize;
    let requested_min_blocks = requested_min_kv_blocks_from_snapshot(runtime_config);
    let max_blocks = estimated_budget_blocks.max(requested_min_blocks);

    Some(AutoSizeResult {
        total_gpu_bytes: total_bytes,
        free_gpu_bytes: free_bytes,
        weight_bytes,
        budgeted_weight_bytes,
        weight_budget_shards,
        budgeted_layer_count,
        kv_block_bytes: block_bytes,
        kv_pool_copies,
        estimated_budget_blocks,
        requested_min_blocks,
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
    let current = RuntimeConfigSnapshot::capture_current();
    let kv_overridden = snapshot_value(&current, "FERRUM_KV_MAX_BLOCKS").is_some();
    let max_seqs_overridden = snapshot_value(&current, "FERRUM_PAGED_MAX_SEQS").is_some();
    let max_batched_tokens_overridden =
        snapshot_value(&current, "FERRUM_MAX_BATCHED_TOKENS").is_some();
    let mut entries = Vec::new();
    // ALL three knobs covered by the user — nothing to set.
    if kv_overridden && max_seqs_overridden && max_batched_tokens_overridden {
        return;
    }
    // Set MAX_BATCHED_TOKENS first so it lands even when the user overrode
    // FERRUM_KV_MAX_BLOCKS + FERRUM_PAGED_MAX_SEQS (apples bench does both,
    // which used to silently skip the Phase 3 scratch budget alongside).
    if !max_batched_tokens_overridden {
        // 2048 is the safe default across both apples M2 (Llama-INT4 8B,
        // ~4 GB) and M3 (Qwen3-30B-A3B, ~17 GB weights + 1 GB scratch +
        // 6 GB KV pool on 24 GB cards). 4096 OOMs on M3 because the
        // Qwen3MoE scratch grows linearly with `t` (batch_logits =
        // t × vocab × 2 B alone is 1.2 GB at t=4096). The unified path
        // still activates at 2048 — scheduler admits up to 4 prefill
        // chunks (4 × 512 = 2048) per iter, m_total stays ≤ scratch.
        let mbt = match profile {
            AutoSizeProfile::Server => 2048,
            AutoSizeProfile::Chat => 2048,
        };
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_MAX_BATCHED_TOKENS",
            mbt.to_string(),
            RuntimeConfigSource::MemoryProfile,
        ));
        eprintln!(
            "[auto-size] MAX_BATCHED_TOKENS={} (profile={:?})",
            mbt, profile
        );
    }
    if kv_overridden && max_seqs_overridden {
        crate::runtime_env::materialize_runtime_env_defaults(&entries);
        return;
    }
    let kv_pool_copies = kv_pool_copies_from_snapshot(&current);
    let Some(result) = auto_size_kv_blocks_with_pool_copies(model_dir, gpu_util, kv_pool_copies)
    else {
        crate::runtime_env::materialize_runtime_env_defaults(&entries);
        return;
    };
    result.print_summary();

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
    // Pool sizing: pool_blocks = max_seqs x ceil(KV_CAPACITY / 16).
    // Picks the largest (max_seqs, KV_CAP) tuple from a fixed ladder whose
    // pool fits the budget.
    //   Server: prioritise wide batch (c=16/32) over context.
    //   Chat:   single user, multi-turn -- pile budget into context.
    let (max_seqs_clamped, kv_capacity) = select_paged_pool_preset(presets, result.max_blocks);

    let kv_capacity_overridden = snapshot_value(&current, "FERRUM_KV_CAPACITY").is_some();
    let mut forced_entries = Vec::new();
    if let Some((requested_seqs, requested_cap, requested_blocks)) =
        requested_paged_pool_blocks_from_snapshot(&current)
    {
        if requested_blocks > result.max_blocks {
            let (safe_seqs, safe_cap) = select_paged_pool_preset(presets, result.max_blocks);
            eprintln!(
                "[auto-size] requested paged KV pool requires {} blocks (max_seqs={} kv_capacity={}) above memory budget {}; clamping to max_seqs={} kv_capacity={}",
                requested_blocks, requested_seqs, requested_cap, result.max_blocks, safe_seqs, safe_cap
            );
            forced_entries.push(RuntimeConfigEntry::new(
                "FERRUM_PAGED_MAX_SEQS",
                safe_seqs.to_string(),
                RuntimeConfigSource::MemoryProfile,
            ));
            forced_entries.push(RuntimeConfigEntry::new(
                "FERRUM_KV_CAPACITY",
                safe_cap.to_string(),
                RuntimeConfigSource::MemoryProfile,
            ));
        }
    }
    // MAX_BATCHED_TOKENS already set above (it's independent of the KV pool
    // sizing logic, runs even when the user overrode KV_MAX_BLOCKS + SEQS).
    // FERRUM_MOE_GRAPH is resolved as a typed CLI startup default and
    // materialized outside the autosizer so it lands even when this function
    // early-returns on full-override.
    if !kv_overridden {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_KV_MAX_BLOCKS",
            result.max_blocks.to_string(),
            RuntimeConfigSource::MemoryProfile,
        ));
    }
    if !max_seqs_overridden {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_PAGED_MAX_SEQS",
            max_seqs_clamped.to_string(),
            RuntimeConfigSource::MemoryProfile,
        ));
    }
    if kv_capacity > 0 && !kv_capacity_overridden {
        entries.push(RuntimeConfigEntry::new(
            "FERRUM_KV_CAPACITY",
            kv_capacity.to_string(),
            RuntimeConfigSource::MemoryProfile,
        ));
    }
    entries.extend(forced_entries.iter().cloned());
    crate::runtime_env::materialize_runtime_env_defaults(&entries);
    for entry in &forced_entries {
        std::env::set_var(&entry.key, &entry.effective_value);
    }
    eprintln!(
        "[auto-size] KV_MAX_BLOCKS={} PAGED_MAX_SEQS={} KV_CAPACITY={}",
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
    );
}

fn ceil_div_u64(value: u64, divisor: u64) -> u64 {
    if divisor == 0 {
        return value;
    }
    value.div_ceil(divisor)
}

fn ceil_div_usize(value: usize, divisor: usize) -> usize {
    if divisor == 0 {
        return value;
    }
    value.div_ceil(divisor)
}

fn select_paged_pool_preset(presets: &[(usize, usize)], max_blocks: usize) -> (usize, usize) {
    if let Some((seqs, cap)) = presets.iter().copied().find(|(seqs, cap_tokens)| {
        let bps = ceil_div_usize(*cap_tokens, PAGED_BLOCK_SIZE as usize).max(1);
        seqs.saturating_mul(bps) <= max_blocks
    }) {
        return (seqs, cap);
    }

    // Budget too tight for any preset -- try min server-like config.
    let bps = (max_blocks / 8).max(8);
    (8, bps * PAGED_BLOCK_SIZE as usize)
}

fn weight_budget_shard_count(snapshot: &RuntimeConfigSnapshot) -> u64 {
    match snapshot_value(
        snapshot,
        crate::gpu_devices::SELECTED_DISTRIBUTED_STRATEGY_KEY,
    ) {
        Some("layer_split") => selected_gpu_device_count(snapshot).max(1) as u64,
        // Tensor-parallel support should extend this match with its own
        // weight/KV placement rules instead of treating all multi-GPU
        // strategies as identical.
        _ => 1,
    }
}

fn layer_count_for_memory_budget(num_layers: u64, snapshot: &RuntimeConfigSnapshot) -> u64 {
    match snapshot_value(
        snapshot,
        crate::gpu_devices::SELECTED_DISTRIBUTED_STRATEGY_KEY,
    ) {
        Some("layer_split") => {
            let shards = selected_gpu_device_count(snapshot).max(1) as u64;
            ceil_div_u64(num_layers, shards).max(1)
        }
        _ => num_layers,
    }
}

fn selected_gpu_device_count(snapshot: &RuntimeConfigSnapshot) -> usize {
    snapshot_value(snapshot, crate::gpu_devices::SELECTED_GPU_DEVICES_KEY)
        .map(|value| {
            value
                .split(',')
                .filter(|part| !part.trim().is_empty())
                .count()
        })
        .unwrap_or(1)
}

fn requested_min_kv_blocks_from_snapshot(snapshot: &RuntimeConfigSnapshot) -> usize {
    let max_model_len_blocks = snapshot_usize(snapshot, "FERRUM_MAX_MODEL_LEN")
        .map(|value| ceil_div_usize(value, PAGED_BLOCK_SIZE as usize))
        .unwrap_or(0);
    let max_batched_token_blocks = snapshot_usize(snapshot, "FERRUM_MAX_BATCHED_TOKENS")
        .map(|value| ceil_div_usize(value, PAGED_BLOCK_SIZE as usize))
        .unwrap_or(0);

    max_model_len_blocks.max(max_batched_token_blocks)
}

fn requested_paged_pool_blocks_from_snapshot(
    snapshot: &RuntimeConfigSnapshot,
) -> Option<(usize, usize, usize)> {
    match (
        snapshot_usize(snapshot, "FERRUM_PAGED_MAX_SEQS"),
        snapshot_usize(snapshot, "FERRUM_KV_CAPACITY"),
    ) {
        (Some(max_seqs), Some(kv_capacity)) => {
            let blocks =
                max_seqs.saturating_mul(ceil_div_usize(kv_capacity, PAGED_BLOCK_SIZE as usize));
            Some((max_seqs, kv_capacity, blocks))
        }
        _ => None,
    }
}

fn snapshot_value<'a>(snapshot: &'a RuntimeConfigSnapshot, key: &str) -> Option<&'a str> {
    snapshot
        .entries
        .iter()
        .find(|entry| entry.key == key)
        .map(|entry| entry.effective_value.as_str())
}

fn snapshot_usize(snapshot: &RuntimeConfigSnapshot, key: &str) -> Option<usize> {
    snapshot_value(snapshot, key).and_then(|value| value.parse::<usize>().ok())
}

fn snapshot_bool(snapshot: &RuntimeConfigSnapshot, key: &str) -> Option<bool> {
    snapshot_value(snapshot, key).map(|value| matches!(value, "1" | "true" | "TRUE" | "on" | "ON"))
}

fn kv_pool_copies_from_snapshot(snapshot: &RuntimeConfigSnapshot) -> u64 {
    let fa_layout = snapshot_bool(snapshot, "FERRUM_FA_LAYOUT_VARLEN").unwrap_or(false);
    let fa2_source = snapshot_bool(snapshot, "FERRUM_FA2_SOURCE").unwrap_or(false);
    let fa2_direct_ffi = snapshot_bool(snapshot, "FERRUM_FA2_DIRECT_FFI")
        .unwrap_or_else(|| snapshot_value(snapshot, "FERRUM_FA2_DIRECT_FFI_SHIM").is_some());

    if fa_layout || fa2_source || fa2_direct_ffi {
        2
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(vars: &[(&str, &str)]) -> RuntimeConfigSnapshot {
        RuntimeConfigSnapshot::from_env_vars(vars.iter().copied())
    }

    #[test]
    fn fa_compatible_attention_paths_count_two_kv_pool_copies() {
        assert_eq!(kv_pool_copies_from_snapshot(&snapshot(&[])), 1);
        assert_eq!(
            kv_pool_copies_from_snapshot(&snapshot(&[("FERRUM_FA_LAYOUT_VARLEN", "1")])),
            2
        );
        assert_eq!(
            kv_pool_copies_from_snapshot(&snapshot(&[("FERRUM_FA2_SOURCE", "1")])),
            2
        );
        assert_eq!(
            kv_pool_copies_from_snapshot(&snapshot(&[("FERRUM_FA2_DIRECT_FFI_SHIM", "/tmp/x.so")])),
            2
        );
        assert_eq!(
            kv_pool_copies_from_snapshot(&snapshot(&[
                ("FERRUM_FA2_DIRECT_FFI", "0"),
                ("FERRUM_FA2_DIRECT_FFI_SHIM", "/tmp/x.so"),
            ])),
            1
        );
    }

    #[test]
    fn layer_split_scopes_weight_and_layer_budget_to_selected_devices() {
        let snapshot = snapshot(&[
            (
                crate::gpu_devices::SELECTED_DISTRIBUTED_STRATEGY_KEY,
                "layer_split",
            ),
            (crate::gpu_devices::SELECTED_GPU_DEVICES_KEY, "0,1"),
        ]);

        assert_eq!(weight_budget_shard_count(&snapshot), 2);
        assert_eq!(layer_count_for_memory_budget(80, &snapshot), 40);
        assert_eq!(ceil_div_u64(37, weight_budget_shard_count(&snapshot)), 19);
    }

    #[test]
    fn unknown_multi_gpu_strategy_keeps_single_device_budget_until_wired() {
        let snapshot = snapshot(&[
            (
                crate::gpu_devices::SELECTED_DISTRIBUTED_STRATEGY_KEY,
                "tensor_parallel",
            ),
            (crate::gpu_devices::SELECTED_GPU_DEVICES_KEY, "0,1"),
        ]);

        assert_eq!(weight_budget_shard_count(&snapshot), 1);
        assert_eq!(layer_count_for_memory_budget(80, &snapshot), 80);
    }

    #[test]
    fn requested_runtime_token_limits_define_kv_block_floor() {
        let snapshot = snapshot(&[
            ("FERRUM_MAX_MODEL_LEN", "8192"),
            ("FERRUM_MAX_BATCHED_TOKENS", "1024"),
            ("FERRUM_PAGED_MAX_SEQS", "8"),
            ("FERRUM_KV_CAPACITY", "2048"),
        ]);

        assert_eq!(requested_min_kv_blocks_from_snapshot(&snapshot), 512);
        assert_eq!(
            requested_paged_pool_blocks_from_snapshot(&snapshot),
            Some((8, 2048, 1024))
        );
    }

    #[test]
    fn paged_pool_preset_fits_block_budget() {
        assert_eq!(
            select_paged_pool_preset(&[(32, 512), (8, 512)], 338),
            (8, 512)
        );
        assert_eq!(
            select_paged_pool_preset(&[(32, 512), (16, 512)], 2048),
            (32, 512)
        );
    }
}
