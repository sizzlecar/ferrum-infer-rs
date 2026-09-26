//! The production CUDA launch geometry, also consumed by selected evidence.
//! No allocation, device pointer, or command is constructed here.
use super::*;

fn config(grid: [u64; 3], block: u32, shared: u64) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    if grid.contains(&0) || block == 0 {
        return Err(CudaDeviceRuntimeError::contract("empty causal launch"));
    }
    Ok(LaunchConfig {
        grid_dim: (
            checked_u32_runtime(grid[0], "causal grid x")?,
            checked_u32_runtime(grid[1], "causal grid y")?,
            checked_u32_runtime(grid[2], "causal grid z")?,
        ),
        block_dim: (block, 1, 1),
        shared_mem_bytes: checked_u32_runtime(shared, "causal shared memory")?,
    })
}
pub(super) fn rms(tokens: u64, width: i32) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    config([tokens, 1, 1], super::super::rms_norm_threads(width), 0)
}
pub(super) fn flat(elements: u64) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    config(
        [elements.div_ceil(u64::from(THREADS_PER_BLOCK)), 1, 1],
        THREADS_PER_BLOCK,
        0,
    )
}
pub(super) fn prepare(
    shape: CudaCausalAttentionShape,
    tokens: u64,
    participants: u32,
) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    let heads = shape
        .key_value_heads
        .checked_mul(2)
        .and_then(|v| v.checked_add(shape.query_heads))
        .and_then(|v| u64::try_from(v).ok())
        .ok_or_else(|| CudaDeviceRuntimeError::contract("causal prepare head count overflows"))?;
    config([tokens, heads, u64::from(participants)], WARP_THREADS, 0)
}
pub(super) fn fallback(
    shape: CudaCausalAttentionShape,
    tokens: u64,
    participants: u32,
) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    config(
        [
            tokens,
            u64::try_from(shape.query_heads)
                .map_err(|_| CudaDeviceRuntimeError::contract("causal query heads negative"))?,
            u64::from(participants),
        ],
        WARP_THREADS,
        0,
    )
}
pub(super) fn grouped(
    shape: CudaCausalAttentionShape,
    tokens: u64,
    threads: u32,
) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    let tile = if shape.head_dim == 256 && shape.sliding_window_tokens == 1024 {
        GROUPED_FALLBACK_GEMMA_LOCAL_KV_TILE_TOKENS
    } else {
        GROUPED_FALLBACK_DEFAULT_KV_TILE_TOKENS
    };
    let shared = tile
        .checked_mul(
            u64::try_from(shape.head_dim)
                .map_err(|_| CudaDeviceRuntimeError::contract("causal head dimension negative"))?,
        )
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| CudaDeviceRuntimeError::contract("causal grouped shared memory overflow"))?;
    config(
        [
            tokens,
            u64::try_from(shape.key_value_heads)
                .map_err(|_| CudaDeviceRuntimeError::contract("causal KV heads negative"))?,
            1,
        ],
        threads,
        shared,
    )
}
pub(super) fn varlen(
    shape: CudaCausalAttentionShape,
    tokens: u64,
    sequence: u64,
    tiled: bool,
) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    if shape.sliding_window_tokens != 0 {
        return Err(CudaDeviceRuntimeError::contract(
            "varlen sliding window unsupported",
        ));
    }
    let rows = if tiled { VARLEN_TILED_QUERY_TOKENS } else { 1 };
    let shared = sequence
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| CudaDeviceRuntimeError::contract("varlen shared size overflow"))?;
    if shared > VARLEN_DYNAMIC_SHARED_BUDGET_BYTES {
        return Err(CudaDeviceRuntimeError::contract(
            "varlen shared budget exceeded",
        ));
    }
    config(
        [
            u64::try_from(shape.query_heads)
                .map_err(|_| CudaDeviceRuntimeError::contract("varlen head count negative"))?,
            tokens.div_ceil(rows),
            1,
        ],
        128,
        shared,
    )
}
