//! CUDA launcher for the recurrent gated DeltaNet update.
//!
//! This is the minimal backend-native W3 primitive: model code supplies
//! already-projected q/k/v plus g/beta gates, and the kernel updates the
//! recurrent state without a host round-trip.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use ferrum_types::{FerrumError, Result};

use super::CudaState;
use crate::backend::{CudaBuf, Dtype};
use crate::ptx;

const MODULE_NAME: &str = "gated_delta_rule";
const FUNC_NAME: &str = "recurrent_gated_delta_rule_f32";
const BATCH_FUNC_NAME: &str = "recurrent_gated_delta_rule_batch_f32";
const BATCH_INDEXED_FUNC_NAME: &str = "recurrent_gated_delta_rule_batch_indexed_f32";
const BATCH_INDEXED_TILED16_FUNC_NAME: &str =
    "recurrent_gated_delta_rule_batch_indexed_tiled16_f32";
const BATCH_INDEXED_PACKED_BA_F32_PARAMS_F32_FUNC_NAME: &str =
    "recurrent_gated_delta_rule_batch_indexed_packed_f32_ba_f32_params_f32";
const BATCH_INDEXED_PACKED_BA_F16_PARAMS_F16_FUNC_NAME: &str =
    "recurrent_gated_delta_rule_batch_indexed_packed_f32_ba_f16_params_f16";
const BATCH_INDEXED_PACKED_BA_F16_PARAMS_F32_FUNC_NAME: &str =
    "recurrent_gated_delta_rule_batch_indexed_packed_f32_ba_f16_params_f32";
const VARLEN_FUNC_NAME: &str = "recurrent_gated_delta_rule_varlen_f32";
const VARLEN_TILED16_FUNC_NAME: &str = "recurrent_gated_delta_rule_varlen_tiled16_f32";

#[allow(clippy::too_many_arguments)]
pub fn recurrent_gated_delta_rule_f32(
    ctx: &mut CudaState,
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    initial_state: &CudaBuf,
    out: &mut CudaBuf,
    final_state: &mut CudaBuf,
    tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    use_qk_l2norm: bool,
    scale: f32,
) -> Result<()> {
    validate_shape(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        out,
        final_state,
        tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    )?;

    let func = ctx.func(MODULE_NAME, ptx::GATED_DELTA_RULE, FUNC_NAME);
    let block = value_dim.min(256).max(1) as u32;
    let stream = ctx.stream.clone();
    let tokens_i32 = tokens as i32;
    let key_heads_i32 = key_heads as i32;
    let value_heads_i32 = value_heads as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    let use_qk_l2norm_i32 = i32::from(use_qk_l2norm);
    let mut builder = stream.launch_builder(&func);
    builder.arg(query.as_f32());
    builder.arg(key.as_f32());
    builder.arg(value.as_f32());
    builder.arg(g.as_f32());
    builder.arg(beta.as_f32());
    builder.arg(initial_state.as_f32());
    builder.arg(out.as_f32_mut());
    builder.arg(final_state.as_f32_mut());
    builder.arg(&tokens_i32);
    builder.arg(&key_heads_i32);
    builder.arg(&value_heads_i32);
    builder.arg(&key_dim_i32);
    builder.arg(&value_dim_i32);
    builder.arg(&use_qk_l2norm_i32);
    builder.arg(&scale);
    unsafe {
        builder
            .launch(LaunchConfig {
                grid_dim: (value_heads as u32, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|err| FerrumError::backend(format!("gated_delta_rule launch: {err}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn recurrent_gated_delta_rule_batch_f32(
    ctx: &mut CudaState,
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    initial_states: &CudaBuf,
    out: &mut CudaBuf,
    final_states: &mut CudaBuf,
    batch: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    use_qk_l2norm: bool,
    scale: f32,
) -> Result<()> {
    validate_batch_shape(
        query,
        key,
        value,
        g,
        beta,
        initial_states,
        out,
        final_states,
        batch,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    )?;

    let func = ctx.func(MODULE_NAME, ptx::GATED_DELTA_RULE, BATCH_FUNC_NAME);
    let block = value_dim.min(256).max(1) as u32;
    let stream = ctx.stream.clone();
    let batch_i32 = batch as i32;
    let key_heads_i32 = key_heads as i32;
    let value_heads_i32 = value_heads as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    let use_qk_l2norm_i32 = i32::from(use_qk_l2norm);
    let mut builder = stream.launch_builder(&func);
    builder.arg(query.as_f32());
    builder.arg(key.as_f32());
    builder.arg(value.as_f32());
    builder.arg(g.as_f32());
    builder.arg(beta.as_f32());
    builder.arg(initial_states.as_f32());
    builder.arg(out.as_f32_mut());
    builder.arg(final_states.as_f32_mut());
    builder.arg(&batch_i32);
    builder.arg(&key_heads_i32);
    builder.arg(&value_heads_i32);
    builder.arg(&key_dim_i32);
    builder.arg(&value_dim_i32);
    builder.arg(&use_qk_l2norm_i32);
    builder.arg(&scale);
    unsafe {
        builder
            .launch(LaunchConfig {
                grid_dim: (value_heads as u32, batch as u32, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|err| FerrumError::backend(format!("gated_delta_rule_batch launch: {err}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn recurrent_gated_delta_rule_batch_indexed_f32(
    ctx: &mut CudaState,
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    state_slots: &mut CudaBuf,
    slot_indices: &CudaBuf,
    out: &mut CudaBuf,
    batch: usize,
    max_slots: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    use_qk_l2norm: bool,
    scale: f32,
) -> Result<()> {
    validate_batch_indexed_shape(
        query,
        key,
        value,
        g,
        beta,
        state_slots,
        slot_indices,
        out,
        batch,
        max_slots,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    )?;

    let use_tiled = !use_qk_l2norm && key_dim == 128 && value_dim == 128;
    let func_name = if use_tiled {
        BATCH_INDEXED_TILED16_FUNC_NAME
    } else {
        BATCH_INDEXED_FUNC_NAME
    };
    let func = ctx.func(MODULE_NAME, ptx::GATED_DELTA_RULE, func_name);
    let block = if use_tiled {
        256
    } else {
        value_dim.min(256).max(1) as u32
    };
    let stream = ctx.stream.clone();
    let batch_i32 = batch as i32;
    let max_slots_i32 = max_slots as i32;
    let key_heads_i32 = key_heads as i32;
    let value_heads_i32 = value_heads as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    let use_qk_l2norm_i32 = i32::from(use_qk_l2norm);
    let mut builder = stream.launch_builder(&func);
    builder.arg(query.as_f32());
    builder.arg(key.as_f32());
    builder.arg(value.as_f32());
    builder.arg(g.as_f32());
    builder.arg(beta.as_f32());
    builder.arg(state_slots.as_f32_mut());
    builder.arg(slot_indices.as_u32());
    builder.arg(out.as_f32_mut());
    builder.arg(&batch_i32);
    builder.arg(&max_slots_i32);
    builder.arg(&key_heads_i32);
    builder.arg(&value_heads_i32);
    builder.arg(&key_dim_i32);
    builder.arg(&value_dim_i32);
    if use_tiled {
        builder.arg(&scale);
    } else {
        builder.arg(&use_qk_l2norm_i32);
        builder.arg(&scale);
    }
    let grid_dim = if use_tiled {
        (
            value_dim.div_ceil(16) as u32,
            value_heads as u32,
            batch as u32,
        )
    } else {
        (value_heads as u32, batch as u32, 1)
    };
    unsafe {
        builder
            .launch(LaunchConfig {
                grid_dim,
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|err| {
                FerrumError::backend(format!("gated_delta_rule_batch_indexed launch: {err}"))
            })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn recurrent_gated_delta_rule_batch_indexed_packed_f32(
    ctx: &mut CudaState,
    mixed_qkv: &CudaBuf,
    ba_raw: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    state_slots: &mut CudaBuf,
    slot_indices: &CudaBuf,
    out: &mut CudaBuf,
    batch: usize,
    max_slots: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    scale: f32,
) -> Result<()> {
    validate_batch_indexed_packed_shape(
        mixed_qkv,
        ba_raw,
        a_log,
        dt_bias,
        state_slots,
        slot_indices,
        out,
        batch,
        max_slots,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    )?;
    let op = "gated_delta_rule_batch_indexed_packed";
    require_dtype(op, "mixed_qkv", mixed_qkv.dtype(), Dtype::F32)?;
    require_dtype(op, "state_slots", state_slots.dtype(), Dtype::F32)?;
    require_dtype(op, "out", out.dtype(), Dtype::F32)?;
    require_dtype(op, "slot_indices", slot_indices.dtype(), Dtype::U32)?;
    let ba_dtype = ba_raw.dtype();
    let param_dtype = a_log.dtype();
    require_dtype(op, "dt_bias", dt_bias.dtype(), param_dtype)?;
    let func_name = match (ba_dtype, param_dtype) {
        (Dtype::F32, Dtype::F32) => BATCH_INDEXED_PACKED_BA_F32_PARAMS_F32_FUNC_NAME,
        (Dtype::F16, Dtype::F16) => BATCH_INDEXED_PACKED_BA_F16_PARAMS_F16_FUNC_NAME,
        (Dtype::F16, Dtype::F32) => BATCH_INDEXED_PACKED_BA_F16_PARAMS_F32_FUNC_NAME,
        _ => {
            return Err(FerrumError::model(format!(
                "{op} unsupported ba/param dtype combination: ba={} params={}",
                ba_dtype.name(),
                param_dtype.name()
            )));
        }
    };

    let func = ctx.func(MODULE_NAME, ptx::GATED_DELTA_RULE, func_name);
    let stream = ctx.stream.clone();
    let batch_i32 = batch as i32;
    let max_slots_i32 = max_slots as i32;
    let key_heads_i32 = key_heads as i32;
    let value_heads_i32 = value_heads as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    let mut builder = stream.launch_builder(&func);
    builder.arg(mixed_qkv.as_f32());
    match ba_dtype {
        Dtype::F32 => builder.arg(ba_raw.as_f32()),
        Dtype::F16 => builder.arg(ba_raw.as_f16()),
        _ => unreachable!("dtype checked above"),
    };
    match param_dtype {
        Dtype::F32 => {
            builder.arg(a_log.as_f32());
            builder.arg(dt_bias.as_f32());
        }
        Dtype::F16 => {
            builder.arg(a_log.as_f16());
            builder.arg(dt_bias.as_f16());
        }
        _ => unreachable!("dtype checked above"),
    };
    builder.arg(state_slots.as_f32_mut());
    builder.arg(slot_indices.as_u32());
    builder.arg(out.as_f32_mut());
    builder.arg(&batch_i32);
    builder.arg(&max_slots_i32);
    builder.arg(&key_heads_i32);
    builder.arg(&value_heads_i32);
    builder.arg(&key_dim_i32);
    builder.arg(&value_dim_i32);
    builder.arg(&scale);
    unsafe {
        builder
            .launch(LaunchConfig {
                grid_dim: (
                    value_dim.div_ceil(16) as u32,
                    value_heads as u32,
                    batch as u32,
                ),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|err| {
                FerrumError::backend(format!(
                    "gated_delta_rule_batch_indexed_packed launch: {err}"
                ))
            })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn recurrent_gated_delta_rule_varlen_f32(
    ctx: &mut CudaState,
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    initial_states: &CudaBuf,
    cu_seqlens: &CudaBuf,
    out: &mut CudaBuf,
    final_states: &mut CudaBuf,
    batch: usize,
    total_tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    use_qk_l2norm: bool,
    scale: f32,
) -> Result<()> {
    validate_varlen_shape(
        query,
        key,
        value,
        g,
        beta,
        initial_states,
        cu_seqlens,
        out,
        final_states,
        batch,
        total_tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    )?;

    let use_tiled = !use_qk_l2norm && key_dim == 128 && value_dim == 128;
    let func_name = if use_tiled {
        VARLEN_TILED16_FUNC_NAME
    } else {
        VARLEN_FUNC_NAME
    };
    let func = ctx.func(MODULE_NAME, ptx::GATED_DELTA_RULE, func_name);
    let block = if use_tiled {
        256
    } else {
        value_dim.min(256).max(1) as u32
    };
    let stream = ctx.stream.clone();
    let batch_i32 = batch as i32;
    let total_tokens_i32 = total_tokens as i32;
    let key_heads_i32 = key_heads as i32;
    let value_heads_i32 = value_heads as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    let use_qk_l2norm_i32 = i32::from(use_qk_l2norm);
    let mut builder = stream.launch_builder(&func);
    builder.arg(query.as_f32());
    builder.arg(key.as_f32());
    builder.arg(value.as_f32());
    builder.arg(g.as_f32());
    builder.arg(beta.as_f32());
    builder.arg(initial_states.as_f32());
    builder.arg(cu_seqlens.as_u32());
    builder.arg(out.as_f32_mut());
    builder.arg(final_states.as_f32_mut());
    builder.arg(&batch_i32);
    builder.arg(&total_tokens_i32);
    builder.arg(&key_heads_i32);
    builder.arg(&value_heads_i32);
    builder.arg(&key_dim_i32);
    builder.arg(&value_dim_i32);
    if use_tiled {
        builder.arg(&scale);
    } else {
        builder.arg(&use_qk_l2norm_i32);
        builder.arg(&scale);
    }
    let grid_dim = if use_tiled {
        (
            value_dim.div_ceil(16) as u32,
            value_heads as u32,
            batch as u32,
        )
    } else {
        (value_heads as u32, batch as u32, 1)
    };
    unsafe {
        builder
            .launch(LaunchConfig {
                grid_dim,
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|err| {
                FerrumError::backend(format!("gated_delta_rule_varlen launch: {err}"))
            })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_shape(
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    initial_state: &CudaBuf,
    out: &CudaBuf,
    final_state: &CudaBuf,
    tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<()> {
    if tokens == 0 || key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 {
        return Err(FerrumError::model(format!(
            "gated_delta_rule shape must be positive, got tokens={tokens} key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim}"
        )));
    }
    if value_heads % key_heads != 0 {
        return Err(FerrumError::model(format!(
            "gated_delta_rule value_heads {value_heads} must be divisible by key_heads {key_heads}"
        )));
    }

    for (label, actual, expected) in [
        ("query", query.len(), tokens * key_heads * key_dim),
        ("key", key.len(), tokens * key_heads * key_dim),
        ("value", value.len(), tokens * value_heads * value_dim),
        ("g", g.len(), tokens * value_heads),
        ("beta", beta.len(), tokens * value_heads),
        (
            "initial_state",
            initial_state.len(),
            value_heads * value_dim * key_dim,
        ),
        ("out", out.len(), tokens * value_heads * value_dim),
        (
            "final_state",
            final_state.len(),
            value_heads * value_dim * key_dim,
        ),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "gated_delta_rule {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_batch_shape(
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    initial_states: &CudaBuf,
    out: &CudaBuf,
    final_states: &CudaBuf,
    batch: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<()> {
    if batch == 0 || key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 {
        return Err(FerrumError::model(format!(
            "gated_delta_rule_batch shape must be positive, got batch={batch} key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim}"
        )));
    }
    if value_heads % key_heads != 0 {
        return Err(FerrumError::model(format!(
            "gated_delta_rule_batch value_heads {value_heads} must be divisible by key_heads {key_heads}"
        )));
    }

    for (label, actual, expected) in [
        ("query", query.len(), batch * key_heads * key_dim),
        ("key", key.len(), batch * key_heads * key_dim),
        ("value", value.len(), batch * value_heads * value_dim),
        ("g", g.len(), batch * value_heads),
        ("beta", beta.len(), batch * value_heads),
        (
            "initial_states",
            initial_states.len(),
            batch * value_heads * value_dim * key_dim,
        ),
        ("out", out.len(), batch * value_heads * value_dim),
        (
            "final_states",
            final_states.len(),
            batch * value_heads * value_dim * key_dim,
        ),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "gated_delta_rule_batch {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_batch_indexed_shape(
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    state_slots: &CudaBuf,
    slot_indices: &CudaBuf,
    out: &CudaBuf,
    batch: usize,
    max_slots: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<()> {
    if batch == 0
        || max_slots == 0
        || key_heads == 0
        || value_heads == 0
        || key_dim == 0
        || value_dim == 0
    {
        return Err(FerrumError::model(format!(
            "gated_delta_rule_batch_indexed shape must be positive, got batch={batch} max_slots={max_slots} key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim}"
        )));
    }
    if value_heads % key_heads != 0 {
        return Err(FerrumError::model(format!(
            "gated_delta_rule_batch_indexed value_heads {value_heads} must be divisible by key_heads {key_heads}"
        )));
    }

    for (label, actual, expected) in [
        ("query", query.len(), batch * key_heads * key_dim),
        ("key", key.len(), batch * key_heads * key_dim),
        ("value", value.len(), batch * value_heads * value_dim),
        ("g", g.len(), batch * value_heads),
        ("beta", beta.len(), batch * value_heads),
        (
            "state_slots",
            state_slots.len(),
            max_slots * value_heads * value_dim * key_dim,
        ),
        ("slot_indices", slot_indices.len(), batch),
        ("out", out.len(), batch * value_heads * value_dim),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "gated_delta_rule_batch_indexed {label} length {actual} < expected {expected}"
            )));
        }
    }
    for (label, buf) in [
        ("query", query),
        ("key", key),
        ("value", value),
        ("g", g),
        ("beta", beta),
        ("state_slots", state_slots),
        ("out", out),
    ] {
        validate_dtype(label, buf)?;
    }
    if slot_indices.dtype() != crate::backend::Dtype::U32 {
        return Err(FerrumError::model(format!(
            "gated_delta_rule_batch_indexed slot_indices dtype {} != u32",
            slot_indices.dtype().name()
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_batch_indexed_packed_shape(
    mixed_qkv: &CudaBuf,
    ba_raw: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    state_slots: &CudaBuf,
    slot_indices: &CudaBuf,
    out: &CudaBuf,
    batch: usize,
    max_slots: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<()> {
    if batch == 0
        || max_slots == 0
        || key_heads == 0
        || value_heads == 0
        || key_dim == 0
        || value_dim == 0
    {
        return Err(FerrumError::model(format!(
            "gated_delta_rule_batch_indexed_packed shape must be positive, got batch={batch} max_slots={max_slots} key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim}"
        )));
    }
    if value_heads % key_heads != 0 {
        return Err(FerrumError::model(format!(
            "gated_delta_rule_batch_indexed_packed value_heads {value_heads} must be divisible by key_heads {key_heads}"
        )));
    }
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    for (label, actual, expected) in [
        ("mixed_qkv", mixed_qkv.len(), batch * conv_channels),
        ("ba_raw", ba_raw.len(), batch * 2 * value_heads),
        ("a_log", a_log.len(), value_heads),
        ("dt_bias", dt_bias.len(), value_heads),
        (
            "state_slots",
            state_slots.len(),
            max_slots * value_heads * value_dim * key_dim,
        ),
        ("slot_indices", slot_indices.len(), batch),
        ("out", out.len(), batch * value_total),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "gated_delta_rule_batch_indexed_packed {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}

fn require_dtype(op: &str, label: &str, actual: Dtype, expected: Dtype) -> Result<()> {
    if actual != expected {
        return Err(FerrumError::model(format!(
            "{op} {label} dtype {} != expected {}",
            actual.name(),
            expected.name()
        )));
    }
    Ok(())
}

fn validate_dtype(label: &str, buf: &CudaBuf) -> Result<()> {
    if buf.dtype() != Dtype::F32 {
        return Err(FerrumError::model(format!(
            "gated_delta_rule_batch_indexed {label} dtype {} != f32",
            buf.dtype().name()
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_varlen_shape(
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    initial_states: &CudaBuf,
    cu_seqlens: &CudaBuf,
    out: &CudaBuf,
    final_states: &CudaBuf,
    batch: usize,
    total_tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<()> {
    if batch == 0
        || total_tokens == 0
        || key_heads == 0
        || value_heads == 0
        || key_dim == 0
        || value_dim == 0
    {
        return Err(FerrumError::model(format!(
            "gated_delta_rule_varlen shape must be positive, got batch={batch} total_tokens={total_tokens} key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim}"
        )));
    }
    if value_heads % key_heads != 0 {
        return Err(FerrumError::model(format!(
            "gated_delta_rule_varlen value_heads {value_heads} must be divisible by key_heads {key_heads}"
        )));
    }

    let state_len = value_heads * value_dim * key_dim;
    for (label, actual, expected) in [
        ("query", query.len(), total_tokens * key_heads * key_dim),
        ("key", key.len(), total_tokens * key_heads * key_dim),
        ("value", value.len(), total_tokens * value_heads * value_dim),
        ("g", g.len(), total_tokens * value_heads),
        ("beta", beta.len(), total_tokens * value_heads),
        ("initial_states", initial_states.len(), batch * state_len),
        ("cu_seqlens", cu_seqlens.len(), batch + 1),
        ("out", out.len(), total_tokens * value_heads * value_dim),
        ("final_states", final_states.len(), batch * state_len),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "gated_delta_rule_varlen {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}
