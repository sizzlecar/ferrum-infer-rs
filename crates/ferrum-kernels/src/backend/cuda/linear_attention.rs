//! CUDA launchers for Qwen-style gated DeltaNet linear-attention helpers.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use ferrum_types::{FerrumError, Result};

use super::CudaState;
use crate::backend::CudaBuf;
use crate::ptx;

const MODULE_NAME: &str = "linear_attention";
const PREPARE_FUNC: &str = "linear_attention_prepare_f32";
const QK_L2NORM_FUNC: &str = "linear_attention_qk_l2norm_f32";
const GATED_RMS_NORM_FUNC: &str = "gated_rms_norm_f32";

#[allow(clippy::too_many_arguments)]
pub fn linear_attention_prepare_f32(
    ctx: &mut CudaState,
    mixed_qkv_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    a_raw: &CudaBuf,
    b_raw: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    query: &mut CudaBuf,
    key: &mut CudaBuf,
    value: &mut CudaBuf,
    g: &mut CudaBuf,
    beta: &mut CudaBuf,
    tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
    apply_qk_l2norm: bool,
) -> Result<()> {
    validate_prepare_shape(
        mixed_qkv_raw,
        conv_weight,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        query,
        key,
        value,
        g,
        beta,
        tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
    )?;

    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let conv_total = tokens * conv_channels;
    let gate_total = tokens * value_heads;
    let total = conv_total.max(gate_total);
    let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, PREPARE_FUNC);
    let stream = ctx.stream.clone();
    let tokens_i32 = tokens as i32;
    let key_heads_i32 = key_heads as i32;
    let value_heads_i32 = value_heads as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    let conv_kernel_i32 = conv_kernel as i32;
    let mut builder = stream.launch_builder(&func);
    builder.arg(mixed_qkv_raw.as_f32());
    builder.arg(conv_weight.as_f32());
    builder.arg(a_raw.as_f32());
    builder.arg(b_raw.as_f32());
    builder.arg(a_log.as_f32());
    builder.arg(dt_bias.as_f32());
    builder.arg(query.as_f32_mut());
    builder.arg(key.as_f32_mut());
    builder.arg(value.as_f32_mut());
    builder.arg(g.as_f32_mut());
    builder.arg(beta.as_f32_mut());
    builder.arg(&tokens_i32);
    builder.arg(&key_heads_i32);
    builder.arg(&value_heads_i32);
    builder.arg(&key_dim_i32);
    builder.arg(&value_dim_i32);
    builder.arg(&conv_kernel_i32);
    unsafe {
        builder
            .launch(LaunchConfig {
                grid_dim: (total.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|err| {
                FerrumError::backend(format!("linear_attention_prepare launch: {err}"))
            })?;
    }

    if apply_qk_l2norm {
        let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, QK_L2NORM_FUNC);
        let block = key_dim.next_power_of_two().min(256).max(1) as u32;
        let stream = ctx.stream.clone();
        let eps = 1e-6f32;
        let mut builder = stream.launch_builder(&func);
        builder.arg(query.as_f32_mut());
        builder.arg(key.as_f32_mut());
        builder.arg(&tokens_i32);
        builder.arg(&key_heads_i32);
        builder.arg(&key_dim_i32);
        builder.arg(&eps);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: ((tokens * key_heads) as u32, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|err| {
                    FerrumError::backend(format!("linear_attention_qk_l2norm launch: {err}"))
                })?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gated_rms_norm_f32(
    ctx: &mut CudaState,
    core: &CudaBuf,
    z: &CudaBuf,
    weight: &CudaBuf,
    out: &mut CudaBuf,
    tokens: usize,
    heads: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    validate_gated_rms_norm_shape(core, z, weight, out, tokens, heads, dim)?;

    let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, GATED_RMS_NORM_FUNC);
    let block = dim.next_power_of_two().min(256).max(1) as u32;
    let stream = ctx.stream.clone();
    let rows_i32 = (tokens * heads) as i32;
    let dim_i32 = dim as i32;
    let mut builder = stream.launch_builder(&func);
    builder.arg(core.as_f32());
    builder.arg(z.as_f32());
    builder.arg(weight.as_f32());
    builder.arg(out.as_f32_mut());
    builder.arg(&rows_i32);
    builder.arg(&dim_i32);
    builder.arg(&eps);
    unsafe {
        builder
            .launch(LaunchConfig {
                grid_dim: ((tokens * heads) as u32, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|err| FerrumError::backend(format!("gated_rms_norm launch: {err}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_prepare_shape(
    mixed_qkv_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    a_raw: &CudaBuf,
    b_raw: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
) -> Result<()> {
    if tokens == 0
        || key_heads == 0
        || value_heads == 0
        || key_dim == 0
        || value_dim == 0
        || conv_kernel == 0
    {
        return Err(FerrumError::model(format!(
            "linear_attention_prepare shape must be positive, got tokens={tokens} key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim} conv_kernel={conv_kernel}"
        )));
    }
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    for (label, actual, expected) in [
        ("mixed_qkv_raw", mixed_qkv_raw.len(), tokens * conv_channels),
        (
            "conv_weight",
            conv_weight.len(),
            conv_channels * conv_kernel,
        ),
        ("a_raw", a_raw.len(), tokens * value_heads),
        ("b_raw", b_raw.len(), tokens * value_heads),
        ("a_log", a_log.len(), value_heads),
        ("dt_bias", dt_bias.len(), value_heads),
        ("query", query.len(), tokens * qk_total),
        ("key", key.len(), tokens * qk_total),
        ("value", value.len(), tokens * value_total),
        ("g", g.len(), tokens * value_heads),
        ("beta", beta.len(), tokens * value_heads),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "linear_attention_prepare {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}

fn validate_gated_rms_norm_shape(
    core: &CudaBuf,
    z: &CudaBuf,
    weight: &CudaBuf,
    out: &CudaBuf,
    tokens: usize,
    heads: usize,
    dim: usize,
) -> Result<()> {
    if tokens == 0 || heads == 0 || dim == 0 {
        return Err(FerrumError::model(format!(
            "gated_rms_norm shape must be positive, got tokens={tokens} heads={heads} dim={dim}"
        )));
    }
    let expected = tokens * heads * dim;
    for (label, actual, expected) in [
        ("core", core.len(), expected),
        ("z", z.len(), expected),
        ("weight", weight.len(), dim),
        ("out", out.len(), expected),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "gated_rms_norm {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}
