//! Exact physical matrix partitions; projections allocate no expanded weights.
use super::*;
use crate::backend::cuda::vnext_ops::native_blocks::{weights, CudaNativeBlockKernels};
use ferrum_interfaces::vnext::PhysicalWeightLayout;
use std::sync::Arc;

pub(super) const MAX_ROWS: u64 = u16::MAX as u64;

pub(super) fn uses_native(
    values: &[ResolvedValueBinding],
    ordinals: &[u32],
) -> Result<bool, String> {
    let mut native = false;
    for &ordinal in ordinals {
        let value = binding(values, ResolvedValueRole::Input, ordinal)?;
        let weight = value
            .weight()
            .ok_or("attention projection lacks physical metadata")?;
        native |= !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. })
            && weights::matrix_parts(weight, value.tensor().dimensions()).is_ok();
    }
    if native {
        for &ordinal in ordinals {
            let value = binding(values, ResolvedValueRole::Input, ordinal)?;
            weights::matrix_parts(value.weight().unwrap(), value.tensor().dimensions())?;
        }
    }
    Ok(native)
}

#[derive(Debug, Clone)]
pub(super) struct SharedNativeMatrix {
    pub(super) first_region: usize,
    pub(super) parts: Arc<[weights::MatrixPart]>,
}

pub(super) fn resolve_shared(
    regions: &mut Vec<CudaBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ordinal: u32,
    shape: &[u64],
) -> Result<SharedNativeMatrix, String> {
    let first = &invocation.participants()[0];
    let matrix = weights::resolve(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, ordinal)?,
        shape,
    )?;
    for participant in &invocation.participants()[1..] {
        let candidate = weights::resolve(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
            shape,
        )?;
        if candidate.parts != matrix.parts
            || candidate.regions.len() != matrix.regions.len()
            || !candidate
                .regions
                .iter()
                .zip(&matrix.regions)
                .all(|(a, b)| super::same_physical_region(a, b))
        {
            return Err("native attention participants do not share physical matrices".into());
        }
    }
    let first_region = regions.len();
    regions.extend(matrix.regions);
    Ok(SharedNativeMatrix {
        first_region,
        parts: matrix.parts.into(),
    })
}

pub(super) fn dispatch_count(parts: usize, rows: u64) -> Result<u64, String> {
    if parts == 0 || rows == 0 {
        return Err("native attention projection is empty".into());
    }
    (parts as u64)
        .checked_mul(rows.div_ceil(MAX_ROWS))
        .ok_or_else(|| "native attention projection dispatch count overflows".into())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch(
    stream: &CudaStream,
    kernels: &CudaNativeBlockKernels,
    parts: &[weights::MatrixPart],
    regions: &[CudaBufferRegion],
    input: u64,
    output: u64,
    rows: i32,
    output_features: i32,
    input_features: i32,
) -> Result<(), CudaDeviceRuntimeError> {
    if rows <= 0 || output_features <= 0 || input_features <= 0 || regions.len() != parts.len() {
        return Err(CudaDeviceRuntimeError::contract(
            "invalid native attention projection extent",
        ));
    }
    launch_parts(
        stream,
        kernels,
        parts
            .iter()
            .zip(regions)
            .map(|(part, region)| (part, region.device_ptr())),
        input,
        output,
        rows,
        output_features,
        input_features,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_parts<'a>(
    stream: &CudaStream,
    kernels: &CudaNativeBlockKernels,
    parts: impl Iterator<Item = (&'a weights::MatrixPart, u64)> + Clone,
    input: u64,
    output: u64,
    rows: i32,
    output_features: i32,
    input_features: i32,
) -> Result<(), CudaDeviceRuntimeError> {
    if rows <= 0 || output_features <= 0 || input_features <= 0 || parts.clone().next().is_none() {
        return Err(CudaDeviceRuntimeError::contract(
            "native attention projection is empty",
        ));
    }
    let mut start = 0_u64;
    while start < rows as u64 {
        let count = (rows as u64 - start).min(MAX_ROWS) as u32;
        let input = offset_pointer(input, start * input_features as u64 * 2)?;
        let output = offset_pointer(output, start * output_features as u64 * 2)?;
        for (part, weight) in parts.clone() {
            kernels.linear(
                stream,
                input,
                weight,
                output,
                part,
                count,
                output_features as u32,
                ElementType::F16,
            )?;
        }
        start += u64::from(count);
    }
    Ok(())
}

fn offset_pointer(base: u64, offset: u64) -> Result<u64, CudaDeviceRuntimeError> {
    base.checked_add(offset).ok_or_else(|| {
        CudaDeviceRuntimeError::contract("native matrix activation pointer overflows")
    })
}
