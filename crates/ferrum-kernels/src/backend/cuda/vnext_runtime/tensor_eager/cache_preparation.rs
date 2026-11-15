//! Optional startup preparation of the original private CUDA pool.
use super::*;
use std::num::NonZeroU64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaTensorCachePreparation {
    Prepared,
    CapacityUnavailable,
    BackingUnavailable,
    PrivatePoolUnavailable,
}

impl CudaDeviceRuntime {
    /// Temporarily admits, writes and retires `bytes` in the existing private
    /// pool. This is a cache hint, not a reservation for later requests; every
    /// tensor still needs its original Core admission and initialization.
    ///
    /// Call during startup after native dependencies are installed: later
    /// native-owner construction or memory pressure may reclaim cached pages.
    /// Insufficient capacity/backing skips the optional preparation. Unknown
    /// native outcomes retain the pool, context and entire temporary claim.
    pub fn prepare_tensor_allocation_cache(
        &self,
        bytes: NonZeroU64,
    ) -> Result<CudaTensorCachePreparation, CudaDeviceRuntimeError> {
        let count = usize::try_from(bytes.get())
            .ok()
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("CUDA cache preparation size is unaddressable")
            })?;
        let owner = self.tensor_eager.as_ref().ok_or_else(|| {
            CudaDeviceRuntimeError::contract("CUDA cache preparation requires a bounded runtime")
        })?;
        let Some(pool) = owner.allocation_pool.as_ref() else {
            return Ok(CudaTensorCachePreparation::PrivatePoolUnavailable);
        };
        let claim = match owner
            .capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, bytes.get())
        {
            Ok(claim) => claim,
            Err(VNextError::DeviceCapacityUnavailable(_)) => {
                return Ok(CudaTensorCachePreparation::CapacityUnavailable);
            }
            Err(error) => return Err(CudaDeviceRuntimeError::capacity(error)),
        };
        let start = Instant::now();
        let (prepared, retention) = create_native(Arc::new((Arc::clone(owner), claim)), || {
            pool.prepare_cache(count).map_err(|error| {
                CudaDeviceRuntimeError::driver("private allocation cache preparation", error)
            })
        })?;
        // Only completed free (or an allocation which returned OOM) reaches
        // here. The retained driver cache carries no logical tensor claim.
        drop(retention);
        let outcome = if prepared {
            CudaTensorCachePreparation::Prepared
        } else {
            CudaTensorCachePreparation::BackingUnavailable
        };
        tracing::info!(
            requested_bytes = bytes.get(),
            elapsed_seconds = start.elapsed().as_secs_f64(),
            ?outcome,
            pool_after = ?pool.snapshot(),
            "CUDA initial allocation cache preparation completed"
        );
        Ok(outcome)
    }
}
