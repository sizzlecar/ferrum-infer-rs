//! CUDA launcher for the recurrent gated DeltaNet update.
//!
//! This is the minimal backend-native W3 primitive: model code supplies
//! already-projected q/k/v plus g/beta gates, and the kernel updates the
//! recurrent state without a host round-trip.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use ferrum_types::{FerrumError, Result};

use super::CudaState;
use crate::backend::CudaBuf;
use crate::ptx;

const MODULE_NAME: &str = "gated_delta_rule";
const FUNC_NAME: &str = "recurrent_gated_delta_rule_f32";
const BATCH_FUNC_NAME: &str = "recurrent_gated_delta_rule_batch_f32";

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
