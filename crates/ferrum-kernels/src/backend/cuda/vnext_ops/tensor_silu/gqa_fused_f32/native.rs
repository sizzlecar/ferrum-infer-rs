//! Locked static native code; two dimension-specific preparations per provider.
use super::*;
use crate::backend::cuda::vnext_runtime::CudaTensorFusedF32Retention;
pub(super) use ferrum_native_ops::cutlass_fused_f32_attention::{
    CutlassFusedF32AttentionBindingsV1 as Bindings, CutlassFusedF32AttentionInfoV1 as Info,
    CutlassFusedF32AttentionRequestV1 as Request, CutlassFusedF32AttentionSpanV1 as Span,
    CUTLASS_FUSED_F32_ATTENTION_NUMERICAL_IDENTITY, CUTLASS_FUSED_F32_ATTENTION_OPERATOR,
};

pub(crate) fn source_fingerprint() -> Result<String, String> {
    if !cfg!(feature = "cuda-cutlass-fused-f32-attention") {
        return Err(
            "fused F32 attention requires the cuda-cutlass-fused-f32-attention artifact".into(),
        );
    }
    let mut artifacts = crate::native_ops::compiled_native_operator_artifacts()
        .iter()
        .filter(|artifact| artifact.operator == CUTLASS_FUSED_F32_ATTENTION_OPERATOR);
    let artifact = artifacts
        .next()
        .ok_or("fused F32 attention artifact identity absent")?;
    if artifacts.next().is_some() {
        return Err("fused F32 attention artifact identity is ambiguous".into());
    }
    let identity = serde_json::to_vec(artifact).map_err(|e| e.to_string())?;
    Ok(implementation_fingerprint(&[
        &identity,
        CUTLASS_FUSED_F32_ATTENTION_NUMERICAL_IDENTITY.as_bytes(),
        include_bytes!("native.rs"),
        include_bytes!("../../../vnext_runtime/tensor_eager/fused_f32_owner.rs"),
    ]))
}

pub(super) struct Native {
    pub(super) owner: Arc<CudaTensorFusedF32Retention>,
    pub(super) infos: [Info; 2],
    pub(super) artifact_identity: String,
}
impl Native {
    pub(super) fn new(
        runtime: &CudaDeviceRuntime,
        config: &CudaCutlassFusedF32Config,
    ) -> Result<Arc<Self>, CudaDeviceRuntimeError> {
        config.validate()?;
        let source = source_fingerprint().map_err(CudaDeviceRuntimeError::contract)?;
        let identity = implementation_fingerprint(&[
            source.as_bytes(),
            config.opaque_memory_profile.as_bytes(),
            &config.opaque_memory_bytes.to_le_bytes(),
        ]);
        // Claim before any CUDART call, including failed preparation. Its static
        // fatbin has no acknowledged unload, so the quota remains process-owned.
        let owner =
            runtime.reserve_tensor_fused_f32_module(&identity, config.opaque_memory_bytes)?;
        owner
            .context()
            .bind_to_thread()
            .map_err(|e| CudaDeviceRuntimeError::driver("fused F32 context bind", e))?;
        use cudarc::driver::sys::CUdevice_attribute::*;
        let attr = |key| {
            owner
                .context()
                .attribute(key)
                .map_err(|e| CudaDeviceRuntimeError::driver("fused F32 device attribute", e))
        };
        if attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? < 8 {
            return Err(CudaDeviceRuntimeError::contract(
                "fused F32 attention requires SM80 or newer",
            ));
        }
        let maximum_shared =
            u32::try_from(attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?)
                .map_err(|_| CudaDeviceRuntimeError::contract("negative shared-memory limit"))?;
        let mut infos = [Info::default(); 2];
        for (dimension, info) in [64, 72].into_iter().zip(&mut infos) {
            let request = Request::new(1, 1, 1, dimension, 1.0)
                .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
            prepare(&request, info)?;
            info.validate_for(&request)
                .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
            if info.shared_bytes > maximum_shared {
                return Err(CudaDeviceRuntimeError::contract(
                    "fused F32 native shared memory exceeds device limit",
                ));
            }
        }
        let artifact_identity =
            implementation_fingerprint(&[source.as_bytes(), numerical_fingerprint().as_bytes()]);
        Ok(Arc::new(Self {
            owner,
            infos,
            artifact_identity,
        }))
    }
    pub(super) fn fingerprint(&self, request: &Request) -> String {
        let info = self.infos[usize::from(request.dimension == 72)];
        implementation_fingerprint(&[
            self.artifact_identity.as_bytes(),
            &request.fingerprint_material(),
            &info.fingerprint_material(),
        ])
    }
    /// All exact spans and the owner remain retained by the enclosing Custom
    /// command, including nonzero status after possible enqueue. No retry.
    pub(super) unsafe fn run(
        &self,
        request: &Request,
        bindings: &Bindings,
        stream: *mut std::ffi::c_void,
    ) -> Result<(), CudaDeviceRuntimeError> {
        bindings
            .validate(request)
            .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
        if stream.is_null() {
            return Err(CudaDeviceRuntimeError::contract(
                "fused F32 requires an explicit stream",
            ));
        }
        #[cfg(feature = "cuda-cutlass-fused-f32-attention")]
        {
            status(unsafe { ferrum_cutlass_fused_f32_attention_run_v1(request, bindings, stream) })
        }
        #[cfg(not(feature = "cuda-cutlass-fused-f32-attention"))]
        {
            Err(CudaDeviceRuntimeError::contract(
                "fused F32 native artifact is not compiled",
            ))
        }
    }
}
fn prepare(request: &Request, info: &mut Info) -> Result<(), CudaDeviceRuntimeError> {
    #[cfg(feature = "cuda-cutlass-fused-f32-attention")]
    {
        status(unsafe { ferrum_cutlass_fused_f32_attention_prepare_v1(request, info) })
    }
    #[cfg(not(feature = "cuda-cutlass-fused-f32-attention"))]
    {
        let _ = (request, info);
        Err(CudaDeviceRuntimeError::contract(
            "fused F32 native artifact is not compiled",
        ))
    }
}
#[cfg(feature = "cuda-cutlass-fused-f32-attention")]
fn status(code: i32) -> Result<(), CudaDeviceRuntimeError> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaDeviceRuntimeError::contract(format!(
            "CUTLASS fused F32 native status {code}"
        )))
    }
}
#[cfg(feature = "cuda-cutlass-fused-f32-attention")]
unsafe extern "C" {
    fn ferrum_cutlass_fused_f32_attention_prepare_v1(
        request: *const Request,
        info: *mut Info,
    ) -> i32;
    fn ferrum_cutlass_fused_f32_attention_run_v1(
        request: *const Request,
        bindings: *const Bindings,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}
