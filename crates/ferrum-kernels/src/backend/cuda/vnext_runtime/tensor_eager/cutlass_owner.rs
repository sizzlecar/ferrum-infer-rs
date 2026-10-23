//! A CUDART static fatbin has no provider-scoped unload operation. Its separate
//! opaque claim is installed once on the tensor owner and conservatively kept
//! until process exit. Dropping cudarc's primary-context wrapper is not proof
//! that CUDART released its reference or that native destruction succeeded.

use super::*;

pub(super) struct ContextModuleQuota {
    identity: String,
    capacity: Option<DeviceCapacityRetention>,
}

impl Drop for ContextModuleQuota {
    fn drop(&mut self) {
        // cuDevicePrimaryCtxRelease (used by cudarc Drop) is not an explicit
        // successful fatbin unload/primary-context destruction receipt. This
        // first CUDART bridge cannot honestly return its opaque allowance.
        // Payload/stream resources still retire through their usual owners.
        if let Some(capacity) = self.capacity.take() {
            std::mem::forget(capacity);
        }
    }
}

pub(crate) struct CudaTensorCutlassRetention {
    // Keep the original context alive through every provider/command operation.
    context: Arc<CudaContext>,
    owner: Arc<TensorEagerState>,
}

impl CudaTensorCutlassRetention {
    pub(crate) fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }
}

impl Drop for CudaTensorCutlassRetention {
    fn drop(&mut self) {
        self.owner
            .cutlass_provider_live
            .store(false, Ordering::Release);
    }
}

impl CudaDeviceRuntime {
    /// One independently charged static FP8 module and one live bounded plan
    /// cache. Rebuilding an equal provider on this runtime does not recharge it;
    /// changing its declared native policy on the same context is rejected.
    /// Reserve before any native planning call, including failing calls.
    pub(crate) fn reserve_tensor_cutlass_module(
        &self,
        identity: &str,
        bytes: u64,
    ) -> Result<Arc<CudaTensorCutlassRetention>, CudaDeviceRuntimeError> {
        if bytes == 0 || identity.is_empty() {
            return Err(CudaDeviceRuntimeError::contract(
                "FP8 module requires a positive independent opaque allowance",
            ));
        }
        let owner = self.tensor_eager.as_ref().ok_or_else(|| {
            CudaDeviceRuntimeError::contract("FP8 module requires bounded tensor runtime")
        })?;
        let mut slot = owner
            .cutlass_module
            .lock()
            .map_err(|_| CudaDeviceRuntimeError::contract("FP8 module owner poisoned"))?;
        if let Some(installed) = slot.as_ref() {
            if installed.identity != identity
                || installed
                    .capacity
                    .as_ref()
                    .map(DeviceCapacityRetention::bytes)
                    != Some(bytes)
            {
                return Err(CudaDeviceRuntimeError::contract(
                    "FP8 module policy differs from this context's installed policy",
                ));
            }
        }
        if owner
            .cutlass_provider_live
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(CudaDeviceRuntimeError::contract(
                "FP8 provider is already live on this context",
            ));
        }
        if slot.is_none() {
            let capacity = match owner
                .capacity
                .reserve(DeviceOpaqueMemoryKind::Bootstrap, bytes)
            {
                Ok(value) => value,
                Err(error) => {
                    owner.cutlass_provider_live.store(false, Ordering::Release);
                    return Err(CudaDeviceRuntimeError::capacity(error));
                }
            };
            *slot = Some(ContextModuleQuota {
                identity: identity.to_owned(),
                capacity: Some(capacity),
            });
        }
        Ok(Arc::new(CudaTensorCutlassRetention {
            context: Arc::clone(&self.context),
            owner: Arc::clone(owner),
        }))
    }
}
