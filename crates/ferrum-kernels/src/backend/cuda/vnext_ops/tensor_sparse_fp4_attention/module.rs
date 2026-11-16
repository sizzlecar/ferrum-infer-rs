//! One admitted native module shared by attention and its FP4 preparation.
use super::super::super::vnext_runtime::{CudaTensorNativeModuleRetention, StaticNativeModule};
use super::super::*;
use crate::native_ops::sage_vsa_fp4::{self, Device, SageVsaFp4Config};

pub(crate) fn source_fingerprint() -> Result<String, String> {
    if !sage_vsa_fp4::compiled() {
        return Err("VSA requires locked cuda-sage-vsa-fp4 artifact".into());
    }
    let mut artifacts = crate::native_ops::compiled_native_operator_artifacts()
        .iter()
        .filter(|a| a.operator == sage_vsa_fp4::SAGE_VSA_FP4_OPERATOR);
    let artifact = artifacts.next().ok_or("VSA artifact absent")?;
    if artifacts.next().is_some() {
        return Err("VSA artifact identity ambiguous".into());
    }
    Ok(implementation_fingerprint(&[
        sage_vsa_fp4::source_declaration_fingerprint().as_bytes(),
        &serde_json::to_vec(artifact).map_err(|e| e.to_string())?,
    ]))
}

pub(crate) struct Module {
    pub(crate) device: Device,
    pub(crate) owner: Arc<CudaTensorNativeModuleRetention>,
    pub(crate) fingerprint: String,
}
impl Module {
    pub(crate) fn acquire(
        runtime: &CudaDeviceRuntime,
        config: &SageVsaFp4Config,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        config
            .validate()
            .map_err(CudaDeviceRuntimeError::contract)?;
        let fingerprint = implementation_fingerprint(&[
            source_fingerprint()
                .map_err(CudaDeviceRuntimeError::contract)?
                .as_bytes(),
            config.opaque_memory_profile.as_bytes(),
            &config.opaque_memory_bytes.to_le_bytes(),
        ]);
        // The finite Core ledger is the only authority to acquire this module.
        // All consumers use the same source/config identity before a CUDART plan.
        let owner = runtime.share_tensor_native_module(
            StaticNativeModule::SageVsaFp4,
            &fingerprint,
            config.opaque_memory_bytes,
        )?;
        use cudarc::driver::sys::CUdevice_attribute::*;
        let attr = |key| -> Result<u32, CudaDeviceRuntimeError> {
            let n = owner
                .context()
                .attribute(key)
                .map_err(|e| CudaDeviceRuntimeError::driver("VSA device attribute", e))?;
            u32::try_from(n)
                .map_err(|_| CudaDeviceRuntimeError::contract("negative VSA device attribute"))
        };
        let major = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let device = Device {
            ordinal: runtime.descriptor().ordinal,
            architecture: major
                .checked_mul(100)
                .and_then(|v| minor.checked_mul(10).and_then(|m| v.checked_add(m)))
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("VSA device architecture overflow")
                })?,
            multiprocessors: attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            maximum_dynamic_shared_bytes: attr(
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )?
            .into(),
        };
        Ok(Self {
            device,
            owner,
            fingerprint,
        })
    }
}
