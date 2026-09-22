//! Recurrent projection roles bound to the shared native matrix implementation.
use super::super::native_matrix;
use super::*;
use crate::backend::cuda::vnext_ops::native_blocks::q8_f32scale::{
    self, PackLayout, Q8F32ScaleKernels,
};
pub(super) use native_matrix::{dispatch_count, launch};
#[cfg(test)]
use native_matrix::{launch_parts, MAX_ROWS};

pub(super) fn uses_native(values: &[ResolvedValueBinding]) -> Result<bool, String> {
    native_matrix::uses_native(values, &[2, 7])
}

/// Both projection inputs share one invocation-owned pack span, overwritten
/// after the first projection is consumed. Unquantized native leaves stay strict.
pub(super) fn q8_workspace_per_token(values: &[ResolvedValueBinding]) -> Result<u64, String> {
    let mut bytes = 0;
    for ordinal in [2, 7] {
        let value = binding(values, ResolvedValueRole::Input, ordinal)?;
        let weight = value
            .weight()
            .ok_or("Q8 attention lacks physical matrix metadata")?;
        let parts = weights::matrix_parts(weight, value.tensor().dimensions())?;
        bytes = bytes.max(q8_part_workspace_per_token(&parts)?);
    }
    Ok(bytes)
}

fn q8_part_workspace_per_token(parts: &[weights::MatrixPart]) -> Result<u64, String> {
    if parts.is_empty() {
        return Err("Q8 attention projection is empty".into());
    }
    let mut bytes = 0;
    for part in parts {
        if part.transform.is_some() || part.signs_region.is_some() {
            return Err("Q8 attention does not support activation rotation".into());
        }
        if q8_f32scale::quantizes(part) {
            bytes = bytes.max(PackLayout::new(1, u64::from(part.columns))?.total_bytes);
        }
    }
    Ok(bytes)
}

pub(super) fn q8_dispatch_count(parts: &[weights::MatrixPart], rows: u64) -> Result<u64, String> {
    q8_part_workspace_per_token(parts)?;
    let launches = parts
        .len()
        .checked_add(usize::from(parts.iter().any(q8_f32scale::quantizes)))
        .ok_or("Q8 attention dispatch count overflows")?;
    // Exactly one pack per row chunk; a mixed matrix shares it across K leaves.
    dispatch_count(launches, rows)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_q8(
    stream: &CudaStream,
    native: &CudaNativeBlockKernels,
    q8: &Q8F32ScaleKernels,
    parts: &[weights::MatrixPart],
    regions: &[CudaBufferRegion],
    input: u64,
    output: u64,
    rows: i32,
    output_features: i32,
    input_features: i32,
    workspace: u64,
    workspace_bytes: u64,
) -> Result<(), CudaDeviceRuntimeError> {
    if regions.len() != parts.len() {
        return Err(CudaDeviceRuntimeError::contract(
            "Q8 attention projection regions disagree",
        ));
    }
    let pointers: Vec<_> = regions.iter().map(CudaBufferRegion::device_ptr).collect();
    launch_q8_parts(
        stream,
        native,
        q8,
        parts,
        &pointers,
        input,
        output,
        rows,
        output_features,
        input_features,
        workspace,
        workspace_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_q8_parts(
    stream: &CudaStream,
    native: &CudaNativeBlockKernels,
    q8: &Q8F32ScaleKernels,
    parts: &[weights::MatrixPart],
    pointers: &[u64],
    input: u64,
    output: u64,
    rows: i32,
    output_features: i32,
    input_features: i32,
    workspace: u64,
    workspace_bytes: u64,
) -> Result<(), CudaDeviceRuntimeError> {
    if rows <= 0 || output_features <= 0 || input_features <= 0 {
        return Err(CudaDeviceRuntimeError::contract(
            "invalid Q8 attention projection extent",
        ));
    }
    let required = q8_part_workspace_per_token(parts)
        .map_err(CudaDeviceRuntimeError::contract)?
        .checked_mul((rows as u64).min(native_matrix::MAX_ROWS))
        .ok_or_else(|| CudaDeviceRuntimeError::contract("Q8 attention pack size overflows"))?;
    if workspace_bytes < required || (required != 0 && workspace == 0) {
        return Err(CudaDeviceRuntimeError::contract(
            "Q8 attention pack exceeds admitted scratch",
        ));
    }
    let mut start = 0_u64;
    while start < rows as u64 {
        let count = (rows as u64 - start).min(native_matrix::MAX_ROWS) as u32;
        let input = input
            .checked_add(start * input_features as u64 * 2)
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("Q8 attention input pointer overflows")
            })?;
        let output = output
            .checked_add(start * output_features as u64 * 2)
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("Q8 attention output pointer overflows")
            })?;
        q8.launch(
            native,
            stream,
            parts,
            pointers,
            input,
            output,
            count,
            input_features as u32,
            output_features as u32,
            workspace,
        )?;
        start += u64::from(count);
    }
    Ok(())
}

pub(super) fn resolve_shared(
    regions: &mut Vec<CudaBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ordinal: u32,
    shape: &[u64],
) -> Result<SharedProjectionWeight, String> {
    let matrix = native_matrix::resolve_shared(regions, invocation, ordinal, shape)?;
    Ok(SharedProjectionWeight::Native {
        first_region: matrix.first_region,
        parts: matrix.parts,
    })
}

#[cfg(test)]
mod tests;
