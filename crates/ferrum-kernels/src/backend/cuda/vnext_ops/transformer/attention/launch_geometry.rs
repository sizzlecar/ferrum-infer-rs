//! Pure launch descriptors shared by the real enqueue functions and evidence.
use super::*;

fn flat(elements: u64, label: &'static str) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    Ok(LaunchConfig {
        grid_dim: (checked_grid(elements, THREADS_PER_BLOCK, label)?, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

pub(super) fn rms(tokens: u64, hidden: i32) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    Ok(LaunchConfig {
        grid_dim: (checked_u32(tokens, "attention RMSNorm rows")?, 1, 1),
        block_dim: (super::super::rms_norm_threads(hidden), 1, 1),
        shared_mem_bytes: 0,
    })
}

pub(super) fn prepare(
    shape: AttentionShape,
    tokens: u64,
    batch: i32,
) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    let batch = u64::try_from(batch)
        .ok()
        .filter(|batch| *batch > 0)
        .ok_or_else(|| {
            CudaDeviceRuntimeError::contract("attention prepare batch is not positive")
        })?;
    let work = tokens
        .checked_mul(shape.qkv_features)
        .zip(tokens.checked_mul(shape.value_heads))
        .map(|(conv, gate)| conv.max(gate))
        .zip(
            shape
                .conv_state_elements()
                .ok()
                .and_then(|n| n.checked_mul(batch)),
        )
        .map(|(work, state)| work.max(state))
        .ok_or_else(|| CudaDeviceRuntimeError::contract("attention prepare work overflows"))?;
    flat(work, "attention prepare")
}

pub(super) fn conv(batch: i32, elements: i32) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    if batch <= 0 || elements <= 0 {
        return Err(CudaDeviceRuntimeError::contract(
            "attention convolution state commit is empty",
        ));
    }
    flat(
        (batch as u64) * (elements as u64),
        "attention convolution state commit",
    )
}

pub(super) fn qk(
    tokens: u64,
    shape: CudaAttentionShape,
) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    let rows = tokens
        .checked_mul(shape.key_heads as u64)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("attention QK rows overflow"))?;
    Ok(LaunchConfig {
        grid_dim: (checked_u32(rows, "attention QK rows")?, 1, 1),
        block_dim: (
            (shape.key_head_dim as u32).next_power_of_two().min(256),
            1,
            1,
        ),
        shared_mem_bytes: 0,
    })
}

pub(super) fn delta_entry(shape: CudaAttentionShape) -> &'static str {
    match (shape.tiled_delta, shape.value_head_mapping) {
        (false, GatedDeltaValueHeadMapping::GroupedByKeyHead) => DELTA_FUNCTION,
        (true, GatedDeltaValueHeadMapping::GroupedByKeyHead) => DELTA_TILED_FUNCTION,
        (false, GatedDeltaValueHeadMapping::InterleavedByKeyHead) => DELTA_INTERLEAVED_FUNCTION,
        (true, GatedDeltaValueHeadMapping::InterleavedByKeyHead) => {
            DELTA_TILED_INTERLEAVED_FUNCTION
        }
    }
}

pub(super) fn delta(
    batch: i32,
    shape: CudaAttentionShape,
) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    if batch <= 0 {
        return Err(CudaDeviceRuntimeError::contract(
            "attention delta batch is not positive",
        ));
    }
    Ok(LaunchConfig {
        grid_dim: if shape.tiled_delta {
            (
                (shape.value_head_dim as u32).div_ceil(16),
                shape.value_heads as u32,
                batch as u32,
            )
        } else {
            (shape.value_heads as u32, batch as u32, 1)
        },
        block_dim: (
            if shape.tiled_delta {
                256
            } else {
                shape.value_head_dim.min(256) as u32
            },
            1,
            1,
        ),
        shared_mem_bytes: 0,
    })
}

pub(super) fn gated(
    tokens: u64,
    shape: CudaAttentionShape,
) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    let rows = tokens
        .checked_mul(shape.value_heads as u64)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("attention gated rows overflow"))?;
    Ok(LaunchConfig {
        grid_dim: (checked_u32(rows, "attention gated rows")?, 1, 1),
        block_dim: (
            (shape.value_head_dim as u32).next_power_of_two().min(256),
            1,
            1,
        ),
        shared_mem_bytes: 0,
    })
}

pub(super) fn cast(elements: u64) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    flat(elements, "attention cast")
}
pub(super) fn residual(elements: u64) -> Result<LaunchConfig, CudaDeviceRuntimeError> {
    flat(elements, "attention residual")
}
