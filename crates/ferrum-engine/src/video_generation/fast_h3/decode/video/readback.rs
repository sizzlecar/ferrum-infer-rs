//! A bounded optional phase grant on the existing Host ledger. No readback
//! credit is held while requests wait, condition, or denoise together.
use super::*;

pub struct FastH3VideoReadbackAdmission {
    pub(super) scope: HostMemoryScope,
    pub(super) maximum_readback_bytes: usize,
}

pub(super) fn try_admit(
    policy: FastH3VideoPostprocessing,
    parent: &HostMemoryAccount,
    maximum_readback_bytes: usize,
) -> Result<Option<FastH3VideoReadbackAdmission>, HostMemoryError> {
    if policy == FastH3VideoPostprocessing::Inline {
        return Ok(None);
    }
    // Both readbacks use this separate scope. The conservative base request
    // bound stays intact, so fallback requires no exchange or reacquisition.
    let bytes = maximum_readback_bytes
        .checked_mul(2)
        .ok_or(HostMemoryError::ArithmeticOverflow)?;
    match parent.reserve_scope(&[(HostMemoryClass::Output, bytes)]) {
        Ok(scope) => {
            tracing::debug!(bytes, "video decode overlap readback phase admitted");
            Ok(Some(FastH3VideoReadbackAdmission {
                scope,
                maximum_readback_bytes,
            }))
        }
        Err(HostMemoryError::Capacity { .. }) => {
            tracing::debug!(
                bytes,
                "video decode overlap capacity unavailable; using inline"
            );
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests;
