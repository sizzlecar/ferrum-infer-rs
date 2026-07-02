//! CUDA launchers for Qwen-style gated DeltaNet linear-attention helpers.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use ferrum_types::{FerrumError, Result};

use super::CudaState;
use crate::backend::{CudaBuf, Dtype};
use crate::ptx;

const MODULE_NAME: &str = "linear_attention";
const PREPARE_FUNC: &str = "linear_attention_prepare_f32";
const PREPARE_F16_TO_F32_FUNC: &str = "linear_attention_prepare_f16_to_f32";
const PREPARE_F16_PARAMS_F32_FUNC: &str = "linear_attention_prepare_f16_params_f32";
const PREPARE_VARLEN_FUNC: &str = "linear_attention_prepare_varlen_f32";
const PREPARE_VARLEN_F16_TO_F32_FUNC: &str = "linear_attention_prepare_varlen_f16_to_f32";
const PREPARE_VARLEN_F16_PARAMS_F32_FUNC: &str = "linear_attention_prepare_varlen_f16_params_f32";
const PREPARE_VARLEN_PACKED_FUNC: &str = "linear_attention_prepare_varlen_packed_qkvz_ba_f32";
const PREPARE_VARLEN_PACKED_F16_TO_F32_FUNC: &str =
    "linear_attention_prepare_varlen_packed_qkvz_ba_f16_to_f32";
const PREPARE_VARLEN_PACKED_F16_PARAMS_F32_FUNC: &str =
    "linear_attention_prepare_varlen_packed_qkvz_ba_f16_params_f32";
const DECODE_PREPARE_FUNC: &str = "linear_attention_decode_prepare_f32";
const DECODE_PREPARE_F16_TO_F32_FUNC: &str = "linear_attention_decode_prepare_f16_to_f32";
const DECODE_PREPARE_F16_PARAMS_F32_FUNC: &str = "linear_attention_decode_prepare_f16_params_f32";
const DECODE_PREPARE_BATCH_FUNC: &str = "linear_attention_decode_prepare_batch_f32";
const DECODE_PREPARE_BATCH_F16_TO_F32_FUNC: &str =
    "linear_attention_decode_prepare_batch_f16_to_f32";
const DECODE_PREPARE_BATCH_F16_PARAMS_F32_FUNC: &str =
    "linear_attention_decode_prepare_batch_f16_params_f32";
const DECODE_PREPARE_BATCH_INDEXED_FUNC: &str = "linear_attention_decode_prepare_batch_indexed_f32";
const DECODE_PREPARE_BATCH_INDEXED_STATE_F16_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_f32_state_f16";
const DECODE_PREPARE_BATCH_INDEXED_F16_TO_F32_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_f16_to_f32";
const DECODE_PREPARE_BATCH_INDEXED_F16_TO_F32_STATE_F16_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_f16_to_f32_state_f16";
const DECODE_PREPARE_BATCH_INDEXED_F16_PARAMS_F32_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_f16_params_f32";
const DECODE_PREPARE_BATCH_INDEXED_F16_PARAMS_F32_STATE_F16_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_f16_params_f32_state_f16";
const DECODE_PREPARE_BATCH_INDEXED_PACKED_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f32";
const DECODE_PREPARE_BATCH_INDEXED_PACKED_STATE_F16_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f32_state_f16";
const DECODE_PREPARE_BATCH_INDEXED_PACKED_F16_TO_F32_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f16_to_f32";
const DECODE_PREPARE_BATCH_INDEXED_PACKED_F16_TO_F32_STATE_F16_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f16_to_f32_state_f16";
const DECODE_PREPARE_BATCH_INDEXED_PACKED_F16_PARAMS_F32_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f16_params_f32";
const DECODE_PREPARE_BATCH_INDEXED_PACKED_F16_PARAMS_F32_STATE_F16_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f16_params_f32_state_f16";
const DECODE_PREPARE_BATCH_INDEXED_PACKED_TO_MIXED_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_f32";
const DECODE_PREPARE_BATCH_INDEXED_PACKED_TO_MIXED_STATE_F16_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_f32_state_f16";
const DECODE_PREPARE_BATCH_INDEXED_PACKED_TO_MIXED_F16_TO_F32_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_f16_to_f32";
const DECODE_PREPARE_BATCH_INDEXED_PACKED_TO_MIXED_F16_TO_F32_STATE_F16_FUNC: &str =
    "linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_f16_to_f32_state_f16";
const QK_L2NORM_FUNC: &str = "linear_attention_qk_l2norm_f32";
const GATED_RMS_NORM_FUNC: &str = "gated_rms_norm_f32";
const GATED_RMS_NORM_F16_TO_F32_FUNC: &str = "gated_rms_norm_f16_to_f32";
const GATED_RMS_NORM_F16_Z_F32_WEIGHT_FUNC: &str = "gated_rms_norm_f16_z_f32_weight";

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

    let (input_dtype, param_dtype) = validate_prepare_dtype(
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
    )?;

    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let conv_total = tokens * conv_channels;
    let gate_total = tokens * value_heads;
    let total = conv_total.max(gate_total);
    let func_name = match (input_dtype, param_dtype) {
        (Dtype::F32, Dtype::F32) => PREPARE_FUNC,
        (Dtype::F16, Dtype::F16) => PREPARE_F16_TO_F32_FUNC,
        (Dtype::F16, Dtype::F32) => PREPARE_F16_PARAMS_F32_FUNC,
        _ => unreachable!("validate_prepare_dtype filters unsupported inputs"),
    };
    let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, func_name);
    let stream = ctx.stream.clone();
    let tokens_i32 = tokens as i32;
    let key_heads_i32 = key_heads as i32;
    let value_heads_i32 = value_heads as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    let conv_kernel_i32 = conv_kernel as i32;
    let mut builder = stream.launch_builder(&func);
    match input_dtype {
        Dtype::F32 => {
            builder.arg(mixed_qkv_raw.as_f32());
            builder.arg(conv_weight.as_f32());
            builder.arg(a_raw.as_f32());
            builder.arg(b_raw.as_f32());
        }
        Dtype::F16 => {
            builder.arg(mixed_qkv_raw.as_f16());
            builder.arg(conv_weight.as_f16());
            builder.arg(a_raw.as_f16());
            builder.arg(b_raw.as_f16());
        }
        _ => unreachable!("validate_prepare_dtype filters unsupported inputs"),
    }
    match param_dtype {
        Dtype::F32 => {
            builder.arg(a_log.as_f32());
            builder.arg(dt_bias.as_f32());
        }
        Dtype::F16 => {
            builder.arg(a_log.as_f16());
            builder.arg(dt_bias.as_f16());
        }
        _ => unreachable!("validate_prepare_dtype filters unsupported params"),
    }
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
pub fn linear_attention_prepare_varlen_f32(
    ctx: &mut CudaState,
    mixed_qkv_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    initial_conv_states: &CudaBuf,
    a_raw: &CudaBuf,
    b_raw: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    cu_seqlens: &CudaBuf,
    token_seq_indices: &CudaBuf,
    query: &mut CudaBuf,
    key: &mut CudaBuf,
    value: &mut CudaBuf,
    g: &mut CudaBuf,
    beta: &mut CudaBuf,
    final_conv_states: &mut CudaBuf,
    batch: usize,
    total_tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
    apply_qk_l2norm: bool,
) -> Result<()> {
    validate_prepare_varlen_shape(
        mixed_qkv_raw,
        conv_weight,
        initial_conv_states,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        cu_seqlens,
        token_seq_indices,
        query,
        key,
        value,
        g,
        beta,
        final_conv_states,
        batch,
        total_tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
    )?;

    let (input_dtype, param_dtype) = validate_prepare_varlen_dtype(
        mixed_qkv_raw,
        conv_weight,
        initial_conv_states,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        cu_seqlens,
        token_seq_indices,
        query,
        key,
        value,
        g,
        beta,
        final_conv_states,
    )?;

    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let state_len = conv_kernel.saturating_sub(1);
    let conv_total = total_tokens * conv_channels;
    let gate_total = total_tokens * value_heads;
    let state_total = batch * conv_channels * state_len;
    let total = conv_total.max(gate_total).max(state_total);
    let func_name = match (input_dtype, param_dtype) {
        (Dtype::F32, Dtype::F32) => PREPARE_VARLEN_FUNC,
        (Dtype::F16, Dtype::F16) => PREPARE_VARLEN_F16_TO_F32_FUNC,
        (Dtype::F16, Dtype::F32) => PREPARE_VARLEN_F16_PARAMS_F32_FUNC,
        _ => unreachable!("validate_prepare_varlen_dtype filters unsupported inputs"),
    };
    let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, func_name);
    let stream = ctx.stream.clone();
    let batch_i32 = batch as i32;
    let total_tokens_i32 = total_tokens as i32;
    let key_heads_i32 = key_heads as i32;
    let value_heads_i32 = value_heads as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    let conv_kernel_i32 = conv_kernel as i32;
    let mut builder = stream.launch_builder(&func);
    match input_dtype {
        Dtype::F32 => {
            builder.arg(mixed_qkv_raw.as_f32());
            builder.arg(conv_weight.as_f32());
        }
        Dtype::F16 => {
            builder.arg(mixed_qkv_raw.as_f16());
            builder.arg(conv_weight.as_f16());
        }
        _ => unreachable!("validate_prepare_varlen_dtype filters unsupported inputs"),
    }
    builder.arg(initial_conv_states.as_f32());
    match input_dtype {
        Dtype::F32 => {
            builder.arg(a_raw.as_f32());
            builder.arg(b_raw.as_f32());
        }
        Dtype::F16 => {
            builder.arg(a_raw.as_f16());
            builder.arg(b_raw.as_f16());
        }
        _ => unreachable!("validate_prepare_varlen_dtype filters unsupported inputs"),
    }
    match param_dtype {
        Dtype::F32 => {
            builder.arg(a_log.as_f32());
            builder.arg(dt_bias.as_f32());
        }
        Dtype::F16 => {
            builder.arg(a_log.as_f16());
            builder.arg(dt_bias.as_f16());
        }
        _ => unreachable!("validate_prepare_varlen_dtype filters unsupported params"),
    }
    builder.arg(cu_seqlens.as_u32());
    builder.arg(token_seq_indices.as_u32());
    builder.arg(query.as_f32_mut());
    builder.arg(key.as_f32_mut());
    builder.arg(value.as_f32_mut());
    builder.arg(g.as_f32_mut());
    builder.arg(beta.as_f32_mut());
    builder.arg(final_conv_states.as_f32_mut());
    builder.arg(&batch_i32);
    builder.arg(&total_tokens_i32);
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
                FerrumError::backend(format!("linear_attention_prepare_varlen launch: {err}"))
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
        builder.arg(&total_tokens_i32);
        builder.arg(&key_heads_i32);
        builder.arg(&key_dim_i32);
        builder.arg(&eps);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: ((total_tokens * key_heads) as u32, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|err| {
                    FerrumError::backend(format!("linear_attention_varlen_qk_l2norm launch: {err}"))
                })?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn linear_attention_prepare_varlen_packed_qkvz_ba_f32(
    ctx: &mut CudaState,
    mixed_qkvz_raw: &CudaBuf,
    ba_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    initial_conv_states: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    cu_seqlens: &CudaBuf,
    token_seq_indices: &CudaBuf,
    query: &mut CudaBuf,
    key: &mut CudaBuf,
    value: &mut CudaBuf,
    z: &mut CudaBuf,
    g: &mut CudaBuf,
    beta: &mut CudaBuf,
    final_conv_states: &mut CudaBuf,
    batch: usize,
    total_tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
    apply_qk_l2norm: bool,
) -> Result<()> {
    validate_prepare_varlen_packed_shape(
        mixed_qkvz_raw,
        ba_raw,
        conv_weight,
        initial_conv_states,
        a_log,
        dt_bias,
        cu_seqlens,
        token_seq_indices,
        query,
        key,
        value,
        z,
        g,
        beta,
        final_conv_states,
        batch,
        total_tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
    )?;

    let (input_dtype, param_dtype) = validate_prepare_varlen_packed_dtype(
        mixed_qkvz_raw,
        ba_raw,
        conv_weight,
        initial_conv_states,
        a_log,
        dt_bias,
        cu_seqlens,
        token_seq_indices,
        query,
        key,
        value,
        z,
        g,
        beta,
        final_conv_states,
    )?;

    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let state_len = conv_kernel.saturating_sub(1);
    let conv_total = total_tokens * conv_channels;
    let z_total = total_tokens * value_total;
    let gate_total = total_tokens * value_heads;
    let state_total = batch * conv_channels * state_len;
    let total = conv_total.max(z_total).max(gate_total).max(state_total);
    let func_name = match (input_dtype, param_dtype) {
        (Dtype::F32, Dtype::F32) => PREPARE_VARLEN_PACKED_FUNC,
        (Dtype::F16, Dtype::F16) => PREPARE_VARLEN_PACKED_F16_TO_F32_FUNC,
        (Dtype::F16, Dtype::F32) => PREPARE_VARLEN_PACKED_F16_PARAMS_F32_FUNC,
        _ => unreachable!("validate_prepare_varlen_packed_dtype filters unsupported inputs"),
    };
    let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, func_name);
    let stream = ctx.stream.clone();
    let batch_i32 = batch as i32;
    let total_tokens_i32 = total_tokens as i32;
    let key_heads_i32 = key_heads as i32;
    let value_heads_i32 = value_heads as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    let conv_kernel_i32 = conv_kernel as i32;
    let mut builder = stream.launch_builder(&func);
    match input_dtype {
        Dtype::F32 => {
            builder.arg(mixed_qkvz_raw.as_f32());
            builder.arg(ba_raw.as_f32());
            builder.arg(conv_weight.as_f32());
        }
        Dtype::F16 => {
            builder.arg(mixed_qkvz_raw.as_f16());
            builder.arg(ba_raw.as_f16());
            builder.arg(conv_weight.as_f16());
        }
        _ => unreachable!("validate_prepare_varlen_packed_dtype filters unsupported inputs"),
    }
    builder.arg(initial_conv_states.as_f32());
    match param_dtype {
        Dtype::F32 => {
            builder.arg(a_log.as_f32());
            builder.arg(dt_bias.as_f32());
        }
        Dtype::F16 => {
            builder.arg(a_log.as_f16());
            builder.arg(dt_bias.as_f16());
        }
        _ => unreachable!("validate_prepare_varlen_packed_dtype filters unsupported params"),
    }
    builder.arg(cu_seqlens.as_u32());
    builder.arg(token_seq_indices.as_u32());
    builder.arg(query.as_f32_mut());
    builder.arg(key.as_f32_mut());
    builder.arg(value.as_f32_mut());
    builder.arg(z.as_f32_mut());
    builder.arg(g.as_f32_mut());
    builder.arg(beta.as_f32_mut());
    builder.arg(final_conv_states.as_f32_mut());
    builder.arg(&batch_i32);
    builder.arg(&total_tokens_i32);
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
                FerrumError::backend(format!(
                    "linear_attention_prepare_varlen_packed launch: {err}"
                ))
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
        builder.arg(&total_tokens_i32);
        builder.arg(&key_heads_i32);
        builder.arg(&key_dim_i32);
        builder.arg(&eps);
        unsafe {
            builder
                .launch(LaunchConfig {
                    grid_dim: ((total_tokens * key_heads) as u32, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|err| {
                    FerrumError::backend(format!(
                        "linear_attention_varlen_packed_qk_l2norm launch: {err}"
                    ))
                })?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn linear_attention_decode_prepare_f32(
    ctx: &mut CudaState,
    mixed_qkv_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    conv_state: &CudaBuf,
    a_raw: &CudaBuf,
    b_raw: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    query: &mut CudaBuf,
    key: &mut CudaBuf,
    value: &mut CudaBuf,
    g: &mut CudaBuf,
    beta: &mut CudaBuf,
    next_conv_state: &mut CudaBuf,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
    apply_qk_l2norm: bool,
) -> Result<()> {
    validate_decode_prepare_shape(
        mixed_qkv_raw,
        conv_weight,
        conv_state,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        query,
        key,
        value,
        g,
        beta,
        next_conv_state,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
    )?;

    let (input_dtype, param_dtype) = validate_decode_prepare_dtype(
        mixed_qkv_raw,
        conv_weight,
        conv_state,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        query,
        key,
        value,
        g,
        beta,
        next_conv_state,
    )?;

    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let total = conv_channels.max(value_heads);
    let func_name = match (input_dtype, param_dtype) {
        (Dtype::F32, Dtype::F32) => DECODE_PREPARE_FUNC,
        (Dtype::F16, Dtype::F16) => DECODE_PREPARE_F16_TO_F32_FUNC,
        (Dtype::F16, Dtype::F32) => DECODE_PREPARE_F16_PARAMS_F32_FUNC,
        _ => unreachable!("validate_decode_prepare_dtype filters unsupported inputs"),
    };
    let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, func_name);
    let stream = ctx.stream.clone();
    let key_heads_i32 = key_heads as i32;
    let value_heads_i32 = value_heads as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    let conv_kernel_i32 = conv_kernel as i32;
    let mut builder = stream.launch_builder(&func);
    match input_dtype {
        Dtype::F32 => {
            builder.arg(mixed_qkv_raw.as_f32());
            builder.arg(conv_weight.as_f32());
        }
        Dtype::F16 => {
            builder.arg(mixed_qkv_raw.as_f16());
            builder.arg(conv_weight.as_f16());
        }
        _ => unreachable!("validate_decode_prepare_dtype filters unsupported inputs"),
    }
    builder.arg(conv_state.as_f32());
    match input_dtype {
        Dtype::F32 => {
            builder.arg(a_raw.as_f32());
            builder.arg(b_raw.as_f32());
        }
        Dtype::F16 => {
            builder.arg(a_raw.as_f16());
            builder.arg(b_raw.as_f16());
        }
        _ => unreachable!("validate_decode_prepare_dtype filters unsupported inputs"),
    }
    match param_dtype {
        Dtype::F32 => {
            builder.arg(a_log.as_f32());
            builder.arg(dt_bias.as_f32());
        }
        Dtype::F16 => {
            builder.arg(a_log.as_f16());
            builder.arg(dt_bias.as_f16());
        }
        _ => unreachable!("validate_decode_prepare_dtype filters unsupported params"),
    }
    builder.arg(query.as_f32_mut());
    builder.arg(key.as_f32_mut());
    builder.arg(value.as_f32_mut());
    builder.arg(g.as_f32_mut());
    builder.arg(beta.as_f32_mut());
    builder.arg(next_conv_state.as_f32_mut());
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
                FerrumError::backend(format!("linear_attention_decode_prepare launch: {err}"))
            })?;
    }

    if apply_qk_l2norm {
        let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, QK_L2NORM_FUNC);
        let block = key_dim.next_power_of_two().min(256).max(1) as u32;
        let stream = ctx.stream.clone();
        let tokens_i32 = 1i32;
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
                    grid_dim: (key_heads as u32, 1, 1),
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
pub fn linear_attention_decode_prepare_batch_f32(
    ctx: &mut CudaState,
    mixed_qkv_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    conv_states: &CudaBuf,
    a_raw: &CudaBuf,
    b_raw: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    query: &mut CudaBuf,
    key: &mut CudaBuf,
    value: &mut CudaBuf,
    g: &mut CudaBuf,
    beta: &mut CudaBuf,
    next_conv_states: &mut CudaBuf,
    batch: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
    apply_qk_l2norm: bool,
) -> Result<()> {
    validate_decode_prepare_batch_shape(
        mixed_qkv_raw,
        conv_weight,
        conv_states,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        query,
        key,
        value,
        g,
        beta,
        next_conv_states,
        batch,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
    )?;

    let (input_dtype, param_dtype) = validate_decode_prepare_dtype(
        mixed_qkv_raw,
        conv_weight,
        conv_states,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        query,
        key,
        value,
        g,
        beta,
        next_conv_states,
    )?;

    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let row_total = conv_channels.max(value_heads);
    let total = batch * row_total;
    let func_name = match (input_dtype, param_dtype) {
        (Dtype::F32, Dtype::F32) => DECODE_PREPARE_BATCH_FUNC,
        (Dtype::F16, Dtype::F16) => DECODE_PREPARE_BATCH_F16_TO_F32_FUNC,
        (Dtype::F16, Dtype::F32) => DECODE_PREPARE_BATCH_F16_PARAMS_F32_FUNC,
        _ => unreachable!("validate_decode_prepare_dtype filters unsupported inputs"),
    };
    let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, func_name);
    let stream = ctx.stream.clone();
    let batch_i32 = batch as i32;
    let key_heads_i32 = key_heads as i32;
    let value_heads_i32 = value_heads as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    let conv_kernel_i32 = conv_kernel as i32;
    let mut builder = stream.launch_builder(&func);
    match input_dtype {
        Dtype::F32 => {
            builder.arg(mixed_qkv_raw.as_f32());
            builder.arg(conv_weight.as_f32());
        }
        Dtype::F16 => {
            builder.arg(mixed_qkv_raw.as_f16());
            builder.arg(conv_weight.as_f16());
        }
        _ => unreachable!("validate_decode_prepare_dtype filters unsupported inputs"),
    }
    builder.arg(conv_states.as_f32());
    match input_dtype {
        Dtype::F32 => {
            builder.arg(a_raw.as_f32());
            builder.arg(b_raw.as_f32());
        }
        Dtype::F16 => {
            builder.arg(a_raw.as_f16());
            builder.arg(b_raw.as_f16());
        }
        _ => unreachable!("validate_decode_prepare_dtype filters unsupported inputs"),
    }
    match param_dtype {
        Dtype::F32 => {
            builder.arg(a_log.as_f32());
            builder.arg(dt_bias.as_f32());
        }
        Dtype::F16 => {
            builder.arg(a_log.as_f16());
            builder.arg(dt_bias.as_f16());
        }
        _ => unreachable!("validate_decode_prepare_dtype filters unsupported params"),
    }
    builder.arg(query.as_f32_mut());
    builder.arg(key.as_f32_mut());
    builder.arg(value.as_f32_mut());
    builder.arg(g.as_f32_mut());
    builder.arg(beta.as_f32_mut());
    builder.arg(next_conv_states.as_f32_mut());
    builder.arg(&batch_i32);
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
                FerrumError::backend(format!(
                    "linear_attention_decode_prepare_batch launch: {err}"
                ))
            })?;
    }

    if apply_qk_l2norm {
        let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, QK_L2NORM_FUNC);
        let block = key_dim.next_power_of_two().min(256).max(1) as u32;
        let stream = ctx.stream.clone();
        let tokens_i32 = batch as i32;
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
                    grid_dim: ((batch * key_heads) as u32, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|err| {
                    FerrumError::backend(format!("linear_attention_batch_qk_l2norm launch: {err}"))
                })?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn linear_attention_decode_prepare_batch_indexed_f32(
    ctx: &mut CudaState,
    mixed_qkv_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    conv_state_slots: &mut CudaBuf,
    slot_indices: &CudaBuf,
    a_raw: &CudaBuf,
    b_raw: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    query: &mut CudaBuf,
    key: &mut CudaBuf,
    value: &mut CudaBuf,
    g: &mut CudaBuf,
    beta: &mut CudaBuf,
    batch: usize,
    max_slots: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
    apply_qk_l2norm: bool,
) -> Result<()> {
    validate_decode_prepare_batch_indexed_shape(
        mixed_qkv_raw,
        conv_weight,
        conv_state_slots,
        slot_indices,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        query,
        key,
        value,
        g,
        beta,
        batch,
        max_slots,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
    )?;

    let (input_dtype, param_dtype) = validate_decode_prepare_dtype(
        mixed_qkv_raw,
        conv_weight,
        conv_state_slots,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        query,
        key,
        value,
        g,
        beta,
        conv_state_slots,
    )?;
    require_dtype(
        "linear_attention_decode_prepare_batch_indexed",
        "slot_indices",
        slot_indices.dtype(),
        Dtype::U32,
    )?;

    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let row_total = conv_channels.max(value_heads);
    let total = batch * row_total;
    let state_dtype = conv_state_slots.dtype();
    let func_name = match (input_dtype, param_dtype, state_dtype) {
        (Dtype::F32, Dtype::F32, Dtype::F32) => DECODE_PREPARE_BATCH_INDEXED_FUNC,
        (Dtype::F32, Dtype::F32, Dtype::F16) => DECODE_PREPARE_BATCH_INDEXED_STATE_F16_FUNC,
        (Dtype::F16, Dtype::F16, Dtype::F32) => DECODE_PREPARE_BATCH_INDEXED_F16_TO_F32_FUNC,
        (Dtype::F16, Dtype::F16, Dtype::F16) => {
            DECODE_PREPARE_BATCH_INDEXED_F16_TO_F32_STATE_F16_FUNC
        }
        (Dtype::F16, Dtype::F32, Dtype::F32) => DECODE_PREPARE_BATCH_INDEXED_F16_PARAMS_F32_FUNC,
        (Dtype::F16, Dtype::F32, Dtype::F16) => {
            DECODE_PREPARE_BATCH_INDEXED_F16_PARAMS_F32_STATE_F16_FUNC
        }
        _ => unreachable!("validate_decode_prepare_dtype filters unsupported inputs"),
    };
    let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, func_name);
    let stream = ctx.stream.clone();
    let batch_i32 = batch as i32;
    let max_slots_i32 = max_slots as i32;
    let key_heads_i32 = key_heads as i32;
    let value_heads_i32 = value_heads as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    let conv_kernel_i32 = conv_kernel as i32;
    let mut builder = stream.launch_builder(&func);
    match input_dtype {
        Dtype::F32 => {
            builder.arg(mixed_qkv_raw.as_f32());
            builder.arg(conv_weight.as_f32());
        }
        Dtype::F16 => {
            builder.arg(mixed_qkv_raw.as_f16());
            builder.arg(conv_weight.as_f16());
        }
        _ => unreachable!("validate_decode_prepare_dtype filters unsupported inputs"),
    }
    match state_dtype {
        Dtype::F32 => builder.arg(conv_state_slots.as_f32_mut()),
        Dtype::F16 => builder.arg(conv_state_slots.as_f16_mut()),
        _ => unreachable!("state dtype checked above"),
    };
    builder.arg(slot_indices.as_u32());
    match input_dtype {
        Dtype::F32 => {
            builder.arg(a_raw.as_f32());
            builder.arg(b_raw.as_f32());
        }
        Dtype::F16 => {
            builder.arg(a_raw.as_f16());
            builder.arg(b_raw.as_f16());
        }
        _ => unreachable!("validate_decode_prepare_dtype filters unsupported inputs"),
    }
    match param_dtype {
        Dtype::F32 => {
            builder.arg(a_log.as_f32());
            builder.arg(dt_bias.as_f32());
        }
        Dtype::F16 => {
            builder.arg(a_log.as_f16());
            builder.arg(dt_bias.as_f16());
        }
        _ => unreachable!("validate_decode_prepare_dtype filters unsupported params"),
    }
    builder.arg(query.as_f32_mut());
    builder.arg(key.as_f32_mut());
    builder.arg(value.as_f32_mut());
    builder.arg(g.as_f32_mut());
    builder.arg(beta.as_f32_mut());
    builder.arg(&batch_i32);
    builder.arg(&max_slots_i32);
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
                FerrumError::backend(format!(
                    "linear_attention_decode_prepare_batch_indexed launch: {err}"
                ))
            })?;
    }

    if apply_qk_l2norm {
        let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, QK_L2NORM_FUNC);
        let block = key_dim.next_power_of_two().min(256).max(1) as u32;
        let stream = ctx.stream.clone();
        let tokens_i32 = batch as i32;
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
                    grid_dim: ((batch * key_heads) as u32, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|err| {
                    FerrumError::backend(format!(
                        "linear_attention_batch_indexed_qk_l2norm launch: {err}"
                    ))
                })?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f32(
    ctx: &mut CudaState,
    mixed_qkvz_raw: &CudaBuf,
    ba_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    conv_state_slots: &mut CudaBuf,
    slot_indices: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    query: &mut CudaBuf,
    key: &mut CudaBuf,
    value: &mut CudaBuf,
    z: &mut CudaBuf,
    g: &mut CudaBuf,
    beta: &mut CudaBuf,
    batch: usize,
    max_slots: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
    apply_qk_l2norm: bool,
) -> Result<()> {
    validate_decode_prepare_batch_indexed_packed_shape(
        mixed_qkvz_raw,
        ba_raw,
        conv_weight,
        conv_state_slots,
        slot_indices,
        a_log,
        dt_bias,
        query,
        key,
        value,
        z,
        g,
        beta,
        batch,
        max_slots,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
    )?;

    let op = "linear_attention_decode_prepare_batch_indexed_packed";
    let input_dtype = mixed_qkvz_raw.dtype();
    require_supported_input_dtype(op, "mixed_qkvz_raw", input_dtype)?;
    for (label, actual) in [
        ("ba_raw", ba_raw.dtype()),
        ("conv_weight", conv_weight.dtype()),
    ] {
        require_dtype(op, label, actual, input_dtype)?;
    }
    let param_dtype = a_log.dtype();
    require_supported_input_dtype(op, "a_log", param_dtype)?;
    require_dtype(op, "dt_bias", dt_bias.dtype(), param_dtype)?;
    if input_dtype == Dtype::F32 && param_dtype != Dtype::F32 {
        return Err(FerrumError::model(format!(
            "{op} param dtype {} is unsupported for f32 input; expected f32",
            param_dtype.name()
        )));
    }
    for (label, actual) in [
        ("conv_state_slots", conv_state_slots.dtype()),
        ("query", query.dtype()),
        ("key", key.dtype()),
        ("value", value.dtype()),
        ("z", z.dtype()),
        ("g", g.dtype()),
        ("beta", beta.dtype()),
    ] {
        if label == "conv_state_slots" {
            if !matches!(actual, Dtype::F32 | Dtype::F16) {
                return Err(FerrumError::model(format!(
                    "{op} {label} dtype {} is unsupported",
                    actual.name()
                )));
            }
        } else {
            require_dtype(op, label, actual, Dtype::F32)?;
        }
    }
    require_dtype(op, "slot_indices", slot_indices.dtype(), Dtype::U32)?;

    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let row_total = conv_channels.max(value_total).max(value_heads);
    let total = batch * row_total;
    let state_dtype = conv_state_slots.dtype();
    let func_name = match (input_dtype, param_dtype, state_dtype) {
        (Dtype::F32, Dtype::F32, Dtype::F32) => DECODE_PREPARE_BATCH_INDEXED_PACKED_FUNC,
        (Dtype::F32, Dtype::F32, Dtype::F16) => DECODE_PREPARE_BATCH_INDEXED_PACKED_STATE_F16_FUNC,
        (Dtype::F16, Dtype::F16, Dtype::F32) => DECODE_PREPARE_BATCH_INDEXED_PACKED_F16_TO_F32_FUNC,
        (Dtype::F16, Dtype::F16, Dtype::F16) => {
            DECODE_PREPARE_BATCH_INDEXED_PACKED_F16_TO_F32_STATE_F16_FUNC
        }
        (Dtype::F16, Dtype::F32, Dtype::F32) => {
            DECODE_PREPARE_BATCH_INDEXED_PACKED_F16_PARAMS_F32_FUNC
        }
        (Dtype::F16, Dtype::F32, Dtype::F16) => {
            DECODE_PREPARE_BATCH_INDEXED_PACKED_F16_PARAMS_F32_STATE_F16_FUNC
        }
        _ => unreachable!("packed decode prepare dtype validation filters unsupported inputs"),
    };
    let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, func_name);
    let stream = ctx.stream.clone();
    let batch_i32 = batch as i32;
    let max_slots_i32 = max_slots as i32;
    let key_heads_i32 = key_heads as i32;
    let value_heads_i32 = value_heads as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    let conv_kernel_i32 = conv_kernel as i32;
    let mut builder = stream.launch_builder(&func);
    match input_dtype {
        Dtype::F32 => {
            builder.arg(mixed_qkvz_raw.as_f32());
            builder.arg(ba_raw.as_f32());
            builder.arg(conv_weight.as_f32());
        }
        Dtype::F16 => {
            builder.arg(mixed_qkvz_raw.as_f16());
            builder.arg(ba_raw.as_f16());
            builder.arg(conv_weight.as_f16());
        }
        _ => unreachable!("packed decode prepare dtype validation filters unsupported inputs"),
    }
    match state_dtype {
        Dtype::F32 => builder.arg(conv_state_slots.as_f32_mut()),
        Dtype::F16 => builder.arg(conv_state_slots.as_f16_mut()),
        _ => unreachable!("state dtype checked above"),
    };
    builder.arg(slot_indices.as_u32());
    match param_dtype {
        Dtype::F32 => {
            builder.arg(a_log.as_f32());
            builder.arg(dt_bias.as_f32());
        }
        Dtype::F16 => {
            builder.arg(a_log.as_f16());
            builder.arg(dt_bias.as_f16());
        }
        _ => unreachable!("packed decode prepare dtype validation filters unsupported params"),
    }
    builder.arg(query.as_f32_mut());
    builder.arg(key.as_f32_mut());
    builder.arg(value.as_f32_mut());
    builder.arg(z.as_f32_mut());
    builder.arg(g.as_f32_mut());
    builder.arg(beta.as_f32_mut());
    builder.arg(&batch_i32);
    builder.arg(&max_slots_i32);
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
                FerrumError::backend(format!(
                    "linear_attention_decode_prepare_batch_indexed_packed launch: {err}"
                ))
            })?;
    }

    if apply_qk_l2norm {
        let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, QK_L2NORM_FUNC);
        let block = key_dim.next_power_of_two().min(256).max(1) as u32;
        let stream = ctx.stream.clone();
        let tokens_i32 = batch as i32;
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
                    grid_dim: ((batch * key_heads) as u32, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|err| {
                    FerrumError::backend(format!(
                        "linear_attention_batch_indexed_packed_qk_l2norm launch: {err}"
                    ))
                })?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_f32(
    ctx: &mut CudaState,
    mixed_qkvz_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    conv_state_slots: &mut CudaBuf,
    slot_indices: &CudaBuf,
    mixed_qkv: &mut CudaBuf,
    z: &mut CudaBuf,
    batch: usize,
    max_slots: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
) -> Result<()> {
    validate_decode_prepare_batch_indexed_packed_to_mixed_shape(
        mixed_qkvz_raw,
        conv_weight,
        conv_state_slots,
        slot_indices,
        mixed_qkv,
        z,
        batch,
        max_slots,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
    )?;

    let op = "linear_attention_decode_prepare_batch_indexed_packed_to_mixed";
    let input_dtype = mixed_qkvz_raw.dtype();
    require_supported_input_dtype(op, "mixed_qkvz_raw", input_dtype)?;
    require_dtype(op, "conv_weight", conv_weight.dtype(), input_dtype)?;
    let state_dtype = conv_state_slots.dtype();
    if !matches!(state_dtype, Dtype::F32 | Dtype::F16) {
        return Err(FerrumError::model(format!(
            "{op} conv_state_slots dtype {} is unsupported",
            state_dtype.name()
        )));
    }
    require_dtype(op, "slot_indices", slot_indices.dtype(), Dtype::U32)?;
    require_dtype(op, "mixed_qkv", mixed_qkv.dtype(), Dtype::F32)?;
    require_dtype(op, "z", z.dtype(), Dtype::F32)?;

    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let row_total = conv_channels.max(value_total);
    let total = batch * row_total;
    let func_name = match (input_dtype, state_dtype) {
        (Dtype::F32, Dtype::F32) => DECODE_PREPARE_BATCH_INDEXED_PACKED_TO_MIXED_FUNC,
        (Dtype::F32, Dtype::F16) => DECODE_PREPARE_BATCH_INDEXED_PACKED_TO_MIXED_STATE_F16_FUNC,
        (Dtype::F16, Dtype::F32) => DECODE_PREPARE_BATCH_INDEXED_PACKED_TO_MIXED_F16_TO_F32_FUNC,
        (Dtype::F16, Dtype::F16) => {
            DECODE_PREPARE_BATCH_INDEXED_PACKED_TO_MIXED_F16_TO_F32_STATE_F16_FUNC
        }
        _ => unreachable!("dtype checked above"),
    };
    let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, func_name);
    let stream = ctx.stream.clone();
    let batch_i32 = batch as i32;
    let max_slots_i32 = max_slots as i32;
    let key_heads_i32 = key_heads as i32;
    let value_heads_i32 = value_heads as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    let conv_kernel_i32 = conv_kernel as i32;
    let mut builder = stream.launch_builder(&func);
    match input_dtype {
        Dtype::F32 => {
            builder.arg(mixed_qkvz_raw.as_f32());
            builder.arg(conv_weight.as_f32());
        }
        Dtype::F16 => {
            builder.arg(mixed_qkvz_raw.as_f16());
            builder.arg(conv_weight.as_f16());
        }
        _ => unreachable!("dtype checked above"),
    }
    match state_dtype {
        Dtype::F32 => builder.arg(conv_state_slots.as_f32_mut()),
        Dtype::F16 => builder.arg(conv_state_slots.as_f16_mut()),
        _ => unreachable!("state dtype checked above"),
    };
    builder.arg(slot_indices.as_u32());
    builder.arg(mixed_qkv.as_f32_mut());
    builder.arg(z.as_f32_mut());
    builder.arg(&batch_i32);
    builder.arg(&max_slots_i32);
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
                FerrumError::backend(format!(
                    "linear_attention_decode_prepare_batch_indexed_packed_to_mixed launch: {err}"
                ))
            })?;
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
    let (input_dtype, weight_dtype) = validate_gated_rms_norm_dtype(core, z, weight, out)?;

    let func_name = match (input_dtype, weight_dtype) {
        (Dtype::F32, Dtype::F32) => GATED_RMS_NORM_FUNC,
        (Dtype::F16, Dtype::F16) => GATED_RMS_NORM_F16_TO_F32_FUNC,
        (Dtype::F16, Dtype::F32) => GATED_RMS_NORM_F16_Z_F32_WEIGHT_FUNC,
        _ => unreachable!("validate_gated_rms_norm_dtype filters unsupported inputs"),
    };
    let func = ctx.func(MODULE_NAME, ptx::LINEAR_ATTENTION, func_name);
    let block = dim.next_power_of_two().min(256).max(1) as u32;
    let stream = ctx.stream.clone();
    let rows_i32 = (tokens * heads) as i32;
    let dim_i32 = dim as i32;
    let mut builder = stream.launch_builder(&func);
    builder.arg(core.as_f32());
    match input_dtype {
        Dtype::F32 => {
            builder.arg(z.as_f32());
        }
        Dtype::F16 => {
            builder.arg(z.as_f16());
        }
        _ => unreachable!("validate_gated_rms_norm_dtype filters unsupported inputs"),
    }
    match weight_dtype {
        Dtype::F32 => builder.arg(weight.as_f32()),
        Dtype::F16 => builder.arg(weight.as_f16()),
        _ => unreachable!("validate_gated_rms_norm_dtype filters unsupported weights"),
    };
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

fn validate_gated_rms_norm_dtype(
    core: &CudaBuf,
    z: &CudaBuf,
    weight: &CudaBuf,
    out: &CudaBuf,
) -> Result<(Dtype, Dtype)> {
    let op = "gated_rms_norm";
    require_dtype(op, "core", core.dtype(), Dtype::F32)?;
    require_dtype(op, "out", out.dtype(), Dtype::F32)?;
    let input_dtype = z.dtype();
    require_supported_input_dtype(op, "z", input_dtype)?;
    let weight_dtype = weight.dtype();
    require_supported_input_dtype(op, "weight", weight_dtype)?;
    if input_dtype == Dtype::F32 && weight_dtype != Dtype::F32 {
        return Err(FerrumError::model(format!(
            "{op} weight dtype {} is unsupported for f32 z; expected f32",
            weight_dtype.name()
        )));
    }
    Ok((input_dtype, weight_dtype))
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

fn require_supported_input_dtype(op: &str, label: &str, dtype: Dtype) -> Result<()> {
    match dtype {
        Dtype::F32 | Dtype::F16 => Ok(()),
        other => Err(FerrumError::model(format!(
            "{op} {label} dtype {} is unsupported; expected f32 or f16",
            other.name()
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_prepare_dtype(
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
) -> Result<(Dtype, Dtype)> {
    let op = "linear_attention_prepare";
    let input_dtype = mixed_qkv_raw.dtype();
    require_supported_input_dtype(op, "mixed_qkv_raw", input_dtype)?;
    for (label, actual) in [
        ("conv_weight", conv_weight.dtype()),
        ("a_raw", a_raw.dtype()),
        ("b_raw", b_raw.dtype()),
    ] {
        require_dtype(op, label, actual, input_dtype)?;
    }
    let param_dtype = a_log.dtype();
    require_supported_input_dtype(op, "a_log", param_dtype)?;
    require_dtype(op, "dt_bias", dt_bias.dtype(), param_dtype)?;
    if input_dtype == Dtype::F32 && param_dtype != Dtype::F32 {
        return Err(FerrumError::model(format!(
            "{op} param dtype {} is unsupported for f32 input; expected f32",
            param_dtype.name()
        )));
    }
    for (label, actual) in [
        ("query", query.dtype()),
        ("key", key.dtype()),
        ("value", value.dtype()),
        ("g", g.dtype()),
        ("beta", beta.dtype()),
    ] {
        require_dtype(op, label, actual, Dtype::F32)?;
    }
    Ok((input_dtype, param_dtype))
}

#[allow(clippy::too_many_arguments)]
fn validate_prepare_varlen_dtype(
    mixed_qkv_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    initial_conv_states: &CudaBuf,
    a_raw: &CudaBuf,
    b_raw: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    cu_seqlens: &CudaBuf,
    token_seq_indices: &CudaBuf,
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    final_conv_states: &CudaBuf,
) -> Result<(Dtype, Dtype)> {
    let op = "linear_attention_prepare_varlen";
    let input_dtype = mixed_qkv_raw.dtype();
    require_supported_input_dtype(op, "mixed_qkv_raw", input_dtype)?;
    for (label, actual) in [
        ("conv_weight", conv_weight.dtype()),
        ("a_raw", a_raw.dtype()),
        ("b_raw", b_raw.dtype()),
    ] {
        require_dtype(op, label, actual, input_dtype)?;
    }
    let param_dtype = a_log.dtype();
    require_supported_input_dtype(op, "a_log", param_dtype)?;
    require_dtype(op, "dt_bias", dt_bias.dtype(), param_dtype)?;
    if input_dtype == Dtype::F32 && param_dtype != Dtype::F32 {
        return Err(FerrumError::model(format!(
            "{op} param dtype {} is unsupported for f32 input; expected f32",
            param_dtype.name()
        )));
    }
    for (label, actual) in [
        ("initial_conv_states", initial_conv_states.dtype()),
        ("query", query.dtype()),
        ("key", key.dtype()),
        ("value", value.dtype()),
        ("g", g.dtype()),
        ("beta", beta.dtype()),
        ("final_conv_states", final_conv_states.dtype()),
    ] {
        require_dtype(op, label, actual, Dtype::F32)?;
    }
    require_dtype(op, "cu_seqlens", cu_seqlens.dtype(), Dtype::U32)?;
    require_dtype(
        op,
        "token_seq_indices",
        token_seq_indices.dtype(),
        Dtype::U32,
    )?;
    Ok((input_dtype, param_dtype))
}

#[allow(clippy::too_many_arguments)]
fn validate_prepare_varlen_packed_dtype(
    mixed_qkvz_raw: &CudaBuf,
    ba_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    initial_conv_states: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    cu_seqlens: &CudaBuf,
    token_seq_indices: &CudaBuf,
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    z: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    final_conv_states: &CudaBuf,
) -> Result<(Dtype, Dtype)> {
    let op = "linear_attention_prepare_varlen_packed";
    let input_dtype = mixed_qkvz_raw.dtype();
    require_supported_input_dtype(op, "mixed_qkvz_raw", input_dtype)?;
    for (label, actual) in [
        ("ba_raw", ba_raw.dtype()),
        ("conv_weight", conv_weight.dtype()),
    ] {
        require_dtype(op, label, actual, input_dtype)?;
    }
    let param_dtype = a_log.dtype();
    require_supported_input_dtype(op, "a_log", param_dtype)?;
    require_dtype(op, "dt_bias", dt_bias.dtype(), param_dtype)?;
    if input_dtype == Dtype::F32 && param_dtype != Dtype::F32 {
        return Err(FerrumError::model(format!(
            "{op} param dtype {} is unsupported for f32 input; expected f32",
            param_dtype.name()
        )));
    }
    for (label, actual) in [
        ("initial_conv_states", initial_conv_states.dtype()),
        ("query", query.dtype()),
        ("key", key.dtype()),
        ("value", value.dtype()),
        ("z", z.dtype()),
        ("g", g.dtype()),
        ("beta", beta.dtype()),
        ("final_conv_states", final_conv_states.dtype()),
    ] {
        require_dtype(op, label, actual, Dtype::F32)?;
    }
    require_dtype(op, "cu_seqlens", cu_seqlens.dtype(), Dtype::U32)?;
    require_dtype(
        op,
        "token_seq_indices",
        token_seq_indices.dtype(),
        Dtype::U32,
    )?;
    Ok((input_dtype, param_dtype))
}

#[allow(clippy::too_many_arguments)]
fn validate_decode_prepare_dtype(
    mixed_qkv_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    conv_state: &CudaBuf,
    a_raw: &CudaBuf,
    b_raw: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    next_conv_state: &CudaBuf,
) -> Result<(Dtype, Dtype)> {
    let op = "linear_attention_decode_prepare";
    let input_dtype = mixed_qkv_raw.dtype();
    require_supported_input_dtype(op, "mixed_qkv_raw", input_dtype)?;
    for (label, actual) in [
        ("conv_weight", conv_weight.dtype()),
        ("a_raw", a_raw.dtype()),
        ("b_raw", b_raw.dtype()),
    ] {
        require_dtype(op, label, actual, input_dtype)?;
    }
    let param_dtype = a_log.dtype();
    require_supported_input_dtype(op, "a_log", param_dtype)?;
    require_dtype(op, "dt_bias", dt_bias.dtype(), param_dtype)?;
    if input_dtype == Dtype::F32 && param_dtype != Dtype::F32 {
        return Err(FerrumError::model(format!(
            "{op} param dtype {} is unsupported for f32 input; expected f32",
            param_dtype.name()
        )));
    }
    for (label, actual) in [
        ("conv_state", conv_state.dtype()),
        ("query", query.dtype()),
        ("key", key.dtype()),
        ("value", value.dtype()),
        ("g", g.dtype()),
        ("beta", beta.dtype()),
        ("next_conv_state", next_conv_state.dtype()),
    ] {
        if matches!(label, "conv_state" | "next_conv_state") {
            if !matches!(actual, Dtype::F32 | Dtype::F16) {
                return Err(FerrumError::model(format!(
                    "{op} {label} dtype {} is unsupported",
                    actual.name()
                )));
            }
        } else {
            require_dtype(op, label, actual, Dtype::F32)?;
        }
    }
    Ok((input_dtype, param_dtype))
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

#[allow(clippy::too_many_arguments)]
fn validate_prepare_varlen_shape(
    mixed_qkv_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    initial_conv_states: &CudaBuf,
    a_raw: &CudaBuf,
    b_raw: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    cu_seqlens: &CudaBuf,
    token_seq_indices: &CudaBuf,
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    final_conv_states: &CudaBuf,
    batch: usize,
    total_tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
) -> Result<()> {
    if batch == 0
        || total_tokens == 0
        || key_heads == 0
        || value_heads == 0
        || key_dim == 0
        || value_dim == 0
        || conv_kernel == 0
    {
        return Err(FerrumError::model(format!(
            "linear_attention_prepare_varlen shape must be positive, got batch={batch} total_tokens={total_tokens} key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim} conv_kernel={conv_kernel}"
        )));
    }
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let conv_state_elements = conv_channels * conv_kernel.saturating_sub(1);
    for (label, actual, expected) in [
        (
            "mixed_qkv_raw",
            mixed_qkv_raw.len(),
            total_tokens * conv_channels,
        ),
        (
            "conv_weight",
            conv_weight.len(),
            conv_channels * conv_kernel,
        ),
        (
            "initial_conv_states",
            initial_conv_states.len(),
            batch * conv_state_elements,
        ),
        ("a_raw", a_raw.len(), total_tokens * value_heads),
        ("b_raw", b_raw.len(), total_tokens * value_heads),
        ("a_log", a_log.len(), value_heads),
        ("dt_bias", dt_bias.len(), value_heads),
        ("cu_seqlens", cu_seqlens.len(), batch + 1),
        ("token_seq_indices", token_seq_indices.len(), total_tokens),
        ("query", query.len(), total_tokens * qk_total),
        ("key", key.len(), total_tokens * qk_total),
        ("value", value.len(), total_tokens * value_total),
        ("g", g.len(), total_tokens * value_heads),
        ("beta", beta.len(), total_tokens * value_heads),
        (
            "final_conv_states",
            final_conv_states.len(),
            batch * conv_state_elements,
        ),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "linear_attention_prepare_varlen {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_prepare_varlen_packed_shape(
    mixed_qkvz_raw: &CudaBuf,
    ba_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    initial_conv_states: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    cu_seqlens: &CudaBuf,
    token_seq_indices: &CudaBuf,
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    z: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    final_conv_states: &CudaBuf,
    batch: usize,
    total_tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
) -> Result<()> {
    if batch == 0
        || total_tokens == 0
        || key_heads == 0
        || value_heads == 0
        || key_dim == 0
        || value_dim == 0
        || conv_kernel == 0
    {
        return Err(FerrumError::model(format!(
            "linear_attention_prepare_varlen_packed shape must be positive, got batch={batch} total_tokens={total_tokens} key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim} conv_kernel={conv_kernel}"
        )));
    }
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let qkvz_width = conv_channels + value_total;
    let ba_width = 2 * value_heads;
    let conv_state_elements = conv_channels * conv_kernel.saturating_sub(1);
    for (label, actual, expected) in [
        (
            "mixed_qkvz_raw",
            mixed_qkvz_raw.len(),
            total_tokens * qkvz_width,
        ),
        ("ba_raw", ba_raw.len(), total_tokens * ba_width),
        (
            "conv_weight",
            conv_weight.len(),
            conv_channels * conv_kernel,
        ),
        (
            "initial_conv_states",
            initial_conv_states.len(),
            batch * conv_state_elements,
        ),
        ("a_log", a_log.len(), value_heads),
        ("dt_bias", dt_bias.len(), value_heads),
        ("cu_seqlens", cu_seqlens.len(), batch + 1),
        ("token_seq_indices", token_seq_indices.len(), total_tokens),
        ("query", query.len(), total_tokens * qk_total),
        ("key", key.len(), total_tokens * qk_total),
        ("value", value.len(), total_tokens * value_total),
        ("z", z.len(), total_tokens * value_total),
        ("g", g.len(), total_tokens * value_heads),
        ("beta", beta.len(), total_tokens * value_heads),
        (
            "final_conv_states",
            final_conv_states.len(),
            batch * conv_state_elements,
        ),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "linear_attention_prepare_varlen_packed {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_decode_prepare_shape(
    mixed_qkv_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    conv_state: &CudaBuf,
    a_raw: &CudaBuf,
    b_raw: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    next_conv_state: &CudaBuf,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
) -> Result<()> {
    if key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 || conv_kernel == 0 {
        return Err(FerrumError::model(format!(
            "linear_attention_decode_prepare shape must be positive, got key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim} conv_kernel={conv_kernel}"
        )));
    }
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let conv_state_elements = conv_channels * conv_kernel.saturating_sub(1);
    for (label, actual, expected) in [
        ("mixed_qkv_raw", mixed_qkv_raw.len(), conv_channels),
        (
            "conv_weight",
            conv_weight.len(),
            conv_channels * conv_kernel,
        ),
        ("conv_state", conv_state.len(), conv_state_elements),
        ("a_raw", a_raw.len(), value_heads),
        ("b_raw", b_raw.len(), value_heads),
        ("a_log", a_log.len(), value_heads),
        ("dt_bias", dt_bias.len(), value_heads),
        ("query", query.len(), qk_total),
        ("key", key.len(), qk_total),
        ("value", value.len(), value_total),
        ("g", g.len(), value_heads),
        ("beta", beta.len(), value_heads),
        (
            "next_conv_state",
            next_conv_state.len(),
            conv_state_elements,
        ),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "linear_attention_decode_prepare {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_decode_prepare_batch_shape(
    mixed_qkv_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    conv_states: &CudaBuf,
    a_raw: &CudaBuf,
    b_raw: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    next_conv_states: &CudaBuf,
    batch: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
) -> Result<()> {
    if batch == 0
        || key_heads == 0
        || value_heads == 0
        || key_dim == 0
        || value_dim == 0
        || conv_kernel == 0
    {
        return Err(FerrumError::model(format!(
            "linear_attention_decode_prepare_batch shape must be positive, got batch={batch} key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim} conv_kernel={conv_kernel}"
        )));
    }
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let conv_state_elements = conv_channels * conv_kernel.saturating_sub(1);
    for (label, actual, expected) in [
        ("mixed_qkv_raw", mixed_qkv_raw.len(), batch * conv_channels),
        (
            "conv_weight",
            conv_weight.len(),
            conv_channels * conv_kernel,
        ),
        (
            "conv_states",
            conv_states.len(),
            batch * conv_state_elements,
        ),
        ("a_raw", a_raw.len(), batch * value_heads),
        ("b_raw", b_raw.len(), batch * value_heads),
        ("a_log", a_log.len(), value_heads),
        ("dt_bias", dt_bias.len(), value_heads),
        ("query", query.len(), batch * qk_total),
        ("key", key.len(), batch * qk_total),
        ("value", value.len(), batch * value_total),
        ("g", g.len(), batch * value_heads),
        ("beta", beta.len(), batch * value_heads),
        (
            "next_conv_states",
            next_conv_states.len(),
            batch * conv_state_elements,
        ),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "linear_attention_decode_prepare_batch {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_decode_prepare_batch_indexed_shape(
    mixed_qkv_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    conv_state_slots: &CudaBuf,
    slot_indices: &CudaBuf,
    a_raw: &CudaBuf,
    b_raw: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    batch: usize,
    max_slots: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
) -> Result<()> {
    if batch == 0
        || max_slots == 0
        || key_heads == 0
        || value_heads == 0
        || key_dim == 0
        || value_dim == 0
        || conv_kernel == 0
    {
        return Err(FerrumError::model(format!(
            "linear_attention_decode_prepare_batch_indexed shape must be positive, got batch={batch} max_slots={max_slots} key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim} conv_kernel={conv_kernel}"
        )));
    }
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let conv_state_elements = conv_channels * conv_kernel.saturating_sub(1);
    for (label, actual, expected) in [
        ("mixed_qkv_raw", mixed_qkv_raw.len(), batch * conv_channels),
        (
            "conv_weight",
            conv_weight.len(),
            conv_channels * conv_kernel,
        ),
        (
            "conv_state_slots",
            conv_state_slots.len(),
            max_slots * conv_state_elements,
        ),
        ("slot_indices", slot_indices.len(), batch),
        ("a_raw", a_raw.len(), batch * value_heads),
        ("b_raw", b_raw.len(), batch * value_heads),
        ("a_log", a_log.len(), value_heads),
        ("dt_bias", dt_bias.len(), value_heads),
        ("query", query.len(), batch * qk_total),
        ("key", key.len(), batch * qk_total),
        ("value", value.len(), batch * value_total),
        ("g", g.len(), batch * value_heads),
        ("beta", beta.len(), batch * value_heads),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "linear_attention_decode_prepare_batch_indexed {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_decode_prepare_batch_indexed_packed_shape(
    mixed_qkvz_raw: &CudaBuf,
    ba_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    conv_state_slots: &CudaBuf,
    slot_indices: &CudaBuf,
    a_log: &CudaBuf,
    dt_bias: &CudaBuf,
    query: &CudaBuf,
    key: &CudaBuf,
    value: &CudaBuf,
    z: &CudaBuf,
    g: &CudaBuf,
    beta: &CudaBuf,
    batch: usize,
    max_slots: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
) -> Result<()> {
    if batch == 0
        || max_slots == 0
        || key_heads == 0
        || value_heads == 0
        || key_dim == 0
        || value_dim == 0
        || conv_kernel == 0
    {
        return Err(FerrumError::model(format!(
            "linear_attention_decode_prepare_batch_indexed_packed shape must be positive, got batch={batch} max_slots={max_slots} key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim} conv_kernel={conv_kernel}"
        )));
    }
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let conv_state_elements = conv_channels * conv_kernel.saturating_sub(1);
    let qkvz_width = conv_channels + value_total;
    let ba_width = 2 * value_heads;
    for (label, actual, expected) in [
        ("mixed_qkvz_raw", mixed_qkvz_raw.len(), batch * qkvz_width),
        ("ba_raw", ba_raw.len(), batch * ba_width),
        (
            "conv_weight",
            conv_weight.len(),
            conv_channels * conv_kernel,
        ),
        (
            "conv_state_slots",
            conv_state_slots.len(),
            max_slots * conv_state_elements,
        ),
        ("slot_indices", slot_indices.len(), batch),
        ("a_log", a_log.len(), value_heads),
        ("dt_bias", dt_bias.len(), value_heads),
        ("query", query.len(), batch * qk_total),
        ("key", key.len(), batch * qk_total),
        ("value", value.len(), batch * value_total),
        ("z", z.len(), batch * value_total),
        ("g", g.len(), batch * value_heads),
        ("beta", beta.len(), batch * value_heads),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "linear_attention_decode_prepare_batch_indexed_packed {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_decode_prepare_batch_indexed_packed_to_mixed_shape(
    mixed_qkvz_raw: &CudaBuf,
    conv_weight: &CudaBuf,
    conv_state_slots: &CudaBuf,
    slot_indices: &CudaBuf,
    mixed_qkv: &CudaBuf,
    z: &CudaBuf,
    batch: usize,
    max_slots: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
) -> Result<()> {
    if batch == 0
        || max_slots == 0
        || key_heads == 0
        || value_heads == 0
        || key_dim == 0
        || value_dim == 0
        || conv_kernel == 0
    {
        return Err(FerrumError::model(format!(
            "linear_attention_decode_prepare_batch_indexed_packed_to_mixed shape must be positive, got batch={batch} max_slots={max_slots} key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim} conv_kernel={conv_kernel}"
        )));
    }
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let conv_state_elements = conv_channels * conv_kernel.saturating_sub(1);
    let qkvz_width = conv_channels + value_total;
    for (label, actual, expected) in [
        ("mixed_qkvz_raw", mixed_qkvz_raw.len(), batch * qkvz_width),
        (
            "conv_weight",
            conv_weight.len(),
            conv_channels * conv_kernel,
        ),
        (
            "conv_state_slots",
            conv_state_slots.len(),
            max_slots * conv_state_elements,
        ),
        ("slot_indices", slot_indices.len(), batch),
        ("mixed_qkv", mixed_qkv.len(), batch * conv_channels),
        ("z", z.len(), batch * value_total),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "linear_attention_decode_prepare_batch_indexed_packed_to_mixed {label} length {actual} < expected {expected}"
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
