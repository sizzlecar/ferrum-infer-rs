//! Recurrent projection roles bound to the shared native matrix implementation.
use super::super::native_matrix;
use super::*;
pub(super) use native_matrix::{dispatch_count, launch};
#[cfg(test)]
use native_matrix::{launch_parts, MAX_ROWS};

pub(super) fn uses_native(values: &[ResolvedValueBinding]) -> Result<bool, String> {
    native_matrix::uses_native(values, &[2, 7])
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
