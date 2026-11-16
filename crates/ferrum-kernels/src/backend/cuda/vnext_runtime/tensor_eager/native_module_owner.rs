//! Context retention around the shared bounded static native module claims.
use super::*;
pub(crate) use crate::backend::cuda_static_native_module_capacity::StaticNativeModule;
use crate::backend::cuda_static_native_module_capacity::{
    StaticNativeModuleCapacityError, StaticNativeModuleLease, StaticNativeModuleSharing,
};

pub(crate) struct CudaTensorNativeModuleRetention {
    // Declaration order closes the lease before releasing the tensor owner,
    // then drops this retained context last.
    _lease: StaticNativeModuleLease,
    _owner: Arc<TensorEagerState>,
    context: Arc<CudaContext>,
}
impl CudaTensorNativeModuleRetention {
    pub(crate) fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }
}
impl CudaDeviceRuntime {
    /// Call before any CUDART planning/initialization, including failing calls.
    /// Existing policy cannot change; another live provider for the same typed
    /// module is rejected. Distinct modules charge the same runtime account.
    pub(crate) fn reserve_tensor_native_module(
        &self,
        module: StaticNativeModule,
        identity: &str,
        bytes: u64,
    ) -> Result<Arc<CudaTensorNativeModuleRetention>, CudaDeviceRuntimeError> {
        self.reserve_tensor_native_module_with_sharing(
            module,
            identity,
            bytes,
            StaticNativeModuleSharing::Exclusive,
        )
    }

    /// One source/configuration may have multiple immutable operation providers.
    /// Their leases share the existing installed module charge, and keep the
    /// context alive independently through their prepared submissions.
    pub(crate) fn share_tensor_native_module(
        &self,
        module: StaticNativeModule,
        identity: &str,
        bytes: u64,
    ) -> Result<Arc<CudaTensorNativeModuleRetention>, CudaDeviceRuntimeError> {
        self.reserve_tensor_native_module_with_sharing(
            module,
            identity,
            bytes,
            StaticNativeModuleSharing::Shared,
        )
    }

    fn reserve_tensor_native_module_with_sharing(
        &self,
        module: StaticNativeModule,
        identity: &str,
        bytes: u64,
        sharing: StaticNativeModuleSharing,
    ) -> Result<Arc<CudaTensorNativeModuleRetention>, CudaDeviceRuntimeError> {
        let owner = self.tensor_eager.as_ref().ok_or_else(|| {
            CudaDeviceRuntimeError::contract("static native module requires bounded tensor runtime")
        })?;
        let lease = owner
            .native_modules
            .claim_with_sharing(&owner.capacity, module, identity, bytes, sharing)
            .map_err(|error| match error {
                StaticNativeModuleCapacityError::Contract(reason) => {
                    CudaDeviceRuntimeError::contract(reason)
                }
                StaticNativeModuleCapacityError::Capacity(error) => {
                    CudaDeviceRuntimeError::capacity(error)
                }
            })?;
        Ok(Arc::new(CudaTensorNativeModuleRetention {
            context: Arc::clone(&self.context),
            _lease: lease,
            _owner: Arc::clone(owner),
        }))
    }
}
