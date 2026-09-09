use rayon::prelude::*;

use super::super::vnext_runtime::CpuRuntimeError;
use super::scalar::CpuFloat;

pub(super) fn rms_norm(
    input: &[u8],
    input_type: CpuFloat,
    weights: &[u8],
    output: &mut [u8],
    output_type: CpuFloat,
    width: usize,
    epsilon: f32,
) -> Result<(), CpuRuntimeError> {
    let row_bytes = input_type.byte_len(1, width)?;
    if input.is_empty()
        || !input.len().is_multiple_of(row_bytes)
        || output.len() != output_type.byte_len(input.len() / row_bytes, width)?
        || weights.len() != CpuFloat::F16.byte_len(1, width)?
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err(CpuRuntimeError::new(
            "CPU RMSNorm shape or epsilon differs from its contract",
        ));
    }
    let output_row_bytes = output_type.byte_len(1, width)?;
    output
        .par_chunks_exact_mut(output_row_bytes)
        .enumerate()
        .for_each(|(row, output)| {
            let input = &input[row * row_bytes..(row + 1) * row_bytes];
            let mut sum = 0.0_f32;
            for column in 0..width {
                let value = input_type.read(input, column);
                sum += value * value;
            }
            let scale = (sum / width as f32 + epsilon).sqrt().recip();
            for column in 0..width {
                output_type.write(
                    output,
                    column,
                    input_type.read(input, column) * scale * CpuFloat::F16.read(weights, column),
                );
            }
        });
    Ok(())
}

pub(super) fn residual_add(
    left: &[u8],
    left_type: CpuFloat,
    right: &[u8],
    output: &mut [u8],
) -> Result<(), CpuRuntimeError> {
    if left.is_empty()
        || !left.len().is_multiple_of(left_type.bytes())
        || right.len() != CpuFloat::F16.byte_len(1, left.len() / left_type.bytes())?
        || output.len() != left.len()
    {
        return Err(CpuRuntimeError::new(
            "CPU residual operands have incompatible ranges",
        ));
    }
    output
        .par_chunks_exact_mut(left_type.bytes())
        .enumerate()
        .for_each(|(index, output)| {
            left_type.write(
                output,
                0,
                left_type.read(left, index) + CpuFloat::F16.read(right, index),
            );
        });
    Ok(())
}

/// The residual contract permits the output to reuse the left input. Each
/// element is read before its write, without another allocation or aliased Rust
/// references. A repeated input may refer to that same complete F16 region.
pub(super) fn residual_add_in_place(
    output: &mut [u8],
    left_type: CpuFloat,
    right: Option<&[u8]>,
) -> Result<(), CpuRuntimeError> {
    if output.is_empty()
        || !output.len().is_multiple_of(left_type.bytes())
        || match right {
            Some(right) => {
                right.len() != CpuFloat::F16.byte_len(1, output.len() / left_type.bytes())?
            }
            None => left_type != CpuFloat::F16,
        }
    {
        return Err(CpuRuntimeError::new(
            "CPU in-place residual has incompatible operand types or lengths",
        ));
    }
    output
        .par_chunks_exact_mut(left_type.bytes())
        .enumerate()
        .for_each(|(index, output)| {
            let left = left_type.read(output, 0);
            let right = right.map_or(left, |right| CpuFloat::F16.read(right, index));
            left_type.write(output, 0, left + right);
        });
    Ok(())
}

pub(super) fn swiglu(
    gate_up: &[u8],
    output: &mut [u8],
    width: usize,
) -> Result<(), CpuRuntimeError> {
    let row_bytes = CpuFloat::F16.byte_len(1, width)?;
    let input_row_bytes = row_bytes
        .checked_mul(2)
        .ok_or_else(|| CpuRuntimeError::new("CPU SwiGLU row size overflows"))?;
    if gate_up.is_empty()
        || !gate_up.len().is_multiple_of(input_row_bytes)
        || output.len() != gate_up.len() / 2
    {
        return Err(CpuRuntimeError::new(
            "CPU SwiGLU operands have incompatible ranges",
        ));
    }
    output
        .par_chunks_exact_mut(row_bytes)
        .enumerate()
        .for_each(|(row, output)| {
            let input = &gate_up[row * input_row_bytes..(row + 1) * input_row_bytes];
            for column in 0..width {
                let gate = CpuFloat::F16.read(input, column);
                let up = CpuFloat::F16.read(input, width + column);
                let sigmoid = if gate >= 0.0 {
                    1.0 / (1.0 + (-gate).exp())
                } else {
                    let value = gate.exp();
                    value / (1.0 + value)
                };
                CpuFloat::F16.write(output, column, (gate * sigmoid) * up);
            }
        });
    Ok(())
}

/// Preserve logits while applying one repetition penalty per distinct token.
/// The caller supplies the admitted vocabulary-sized byte workspace.
#[allow(clippy::too_many_arguments)]
pub(super) fn masked_argmax(
    logits: &[u8],
    dtype: CpuFloat,
    valid_mask: &[u8],
    repeated_ids: &[u8],
    repeated_offsets: &[u8],
    penalty: f32,
    repetition_mask: &mut [u8],
) -> Result<u32, CpuRuntimeError> {
    let vocabulary = valid_mask.len();
    if vocabulary == 0
        || vocabulary > u32::MAX as usize
        || logits.len() != dtype.byte_len(1, vocabulary)?
        || repetition_mask.len() != vocabulary
        || !repeated_ids.len().is_multiple_of(4)
        || repeated_offsets.len() != 8
        || !penalty.is_finite()
        || penalty <= 0.0
    {
        return Err(CpuRuntimeError::new(
            "CPU masked argmax operands or repetition penalty are invalid",
        ));
    }
    let capacity = repeated_ids.len() / 4;
    let start =
        (u32::from_le_bytes(repeated_offsets[..4].try_into().unwrap()) as usize).min(capacity);
    let end =
        (u32::from_le_bytes(repeated_offsets[4..].try_into().unwrap()) as usize).min(capacity);
    repetition_mask.fill(0);
    if penalty != 1.0 && start < end {
        for entry in repeated_ids[start * 4..end * 4].chunks_exact(4) {
            let token = u32::from_le_bytes(entry.try_into().unwrap()) as usize;
            if token < vocabulary {
                repetition_mask[token] = 1;
            }
        }
    }
    let mut best = u32::MAX;
    let mut maximum = f32::NEG_INFINITY;
    for token in 0..vocabulary {
        if valid_mask[token] == 0 {
            continue;
        }
        let mut value = dtype.read(logits, token);
        if repetition_mask[token] != 0 {
            value = if value > 0.0 {
                value / penalty
            } else {
                value * penalty
            };
            if dtype == CpuFloat::F16 {
                value = half::f16::from_f32(value).to_f32();
            }
        }
        if value.is_finite() && (best == u32::MAX || value > maximum) {
            maximum = value;
            best = token as u32;
        }
    }
    Ok(best)
}
