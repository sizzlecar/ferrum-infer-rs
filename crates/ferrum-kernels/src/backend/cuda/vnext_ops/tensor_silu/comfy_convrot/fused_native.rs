//! Independent native authority; no payload allocation, cache or extra stream.
use super::*;
use crate::backend::cuda::vnext_runtime::CudaTensorConvRotFusedRetention;
use ferrum_native_ops::cutlass_int8_fused_output::{
    CutlassInt8FusedOutputBindingsV1 as Bindings, CutlassInt8FusedOutputInfoV1 as Info,
    CutlassInt8FusedOutputShapeV1 as NativeShape, CUTLASS_INT8_FUSED_OUTPUT_NUMERICAL_IDENTITY,
    CUTLASS_INT8_FUSED_OUTPUT_OPERATOR as OPERATOR,
};

#[derive(Clone, Debug)]
pub struct CudaConvRotFusedOutputConfig {
    pub opaque_memory_bytes: u64,
    pub opaque_memory_profile: String,
}
impl CudaConvRotFusedOutputConfig {
    pub fn validate(&self) -> Result<(), CudaDeviceRuntimeError> {
        if self.opaque_memory_bytes == 0
            || self.opaque_memory_profile.trim().is_empty()
            || self.opaque_memory_profile.len() > 256
            || !self.opaque_memory_profile.is_ascii()
            || self
                .opaque_memory_profile
                .bytes()
                .any(|b| b.is_ascii_control())
        {
            return Err(CudaDeviceRuntimeError::contract("fused INT8 output requires a positive opaque allowance and versioned ASCII profile"));
        }
        Ok(())
    }
}
pub(crate) fn source_fingerprint() -> Result<String, String> {
    if !cfg!(feature = "cuda-cutlass-int8-fused-output") {
        return Err(
            "fused INT8 output requires the cuda-cutlass-int8-fused-output artifact".into(),
        );
    }
    let mut artifacts = crate::native_ops::compiled_native_operator_artifacts()
        .iter()
        .filter(|a| a.operator == OPERATOR);
    let artifact = artifacts
        .next()
        .ok_or("fused INT8 output artifact identity absent")?;
    if artifacts.next().is_some() {
        return Err("fused INT8 output artifact identity is ambiguous".into());
    }
    Ok(implementation_fingerprint(&[
        &serde_json::to_vec(artifact).map_err(|e| e.to_string())?,
        crate::native_ops::cutlass_int8_fused_output::source_declaration_fingerprint().as_bytes(),
        CUTLASS_INT8_FUSED_OUTPUT_NUMERICAL_IDENTITY.as_bytes(),
    ]))
}

pub(super) struct Native {
    _owner: Arc<CudaTensorConvRotFusedRetention>,
    identity: String,
}
impl Native {
    pub(super) fn new(
        runtime: &CudaDeviceRuntime,
        config: &CudaConvRotFusedOutputConfig,
    ) -> Result<Arc<Self>, CudaDeviceRuntimeError> {
        config.validate()?;
        let source = source_fingerprint().map_err(CudaDeviceRuntimeError::contract)?;
        let identity = implementation_fingerprint(&[
            source.as_bytes(),
            config.opaque_memory_profile.as_bytes(),
            &config.opaque_memory_bytes.to_le_bytes(),
            numerical_fingerprint().as_bytes(),
            format!("cublas.version={}", dense::cublas_version()?).as_bytes(),
        ]);
        // Claim before this unit's first CUDART call. The static module cannot
        // acknowledge unload, so its opaque quota remains process-owned.
        let owner =
            runtime.reserve_tensor_convrot_fused_module(&identity, config.opaque_memory_bytes)?;
        owner
            .context()
            .bind_to_thread()
            .map_err(|e| CudaDeviceRuntimeError::driver("fused INT8 output context", e))?;
        let major = owner.context().attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| CudaDeviceRuntimeError::driver("fused INT8 output device attribute", e))?;
        if major < 8 {
            return Err(CudaDeviceRuntimeError::contract(
                "fused INT8 output requires SM80 or newer",
            ));
        }
        Ok(Arc::new(Self {
            _owner: owner,
            identity,
        }))
    }
    pub(super) fn prepare(
        self: &Arc<Self>,
        shape: Shape,
        storage: Storage,
    ) -> Result<Arc<Prepared>, CudaDeviceRuntimeError> {
        let request = crate::native_ops::cutlass_int8_fused_output::native_shape(
            if storage.f32() {
                ElementType::F32
            } else {
                ElementType::Bf16
            },
            shape.rows,
            shape.outputs,
            shape.inputs,
        );
        let mut info = Info::default();
        if let Some(request) = &request {
            prepare(request, &mut info)?;
            info.validate()
                .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
        }
        let fingerprint = implementation_fingerprint(&[
            self.identity.as_bytes(),
            storage.profile().as_bytes(),
            &shape.rows.to_le_bytes(),
            &shape.inputs.to_le_bytes(),
            &shape.outputs.to_le_bytes(),
            &[u8::from(request.is_some())],
            &info.workspace_bytes.to_le_bytes(),
            &info.tile_m.to_le_bytes(),
            &info.tile_n.to_le_bytes(),
            &info.tile_k.to_le_bytes(),
            &info.threads.to_le_bytes(),
        ]);
        Ok(Arc::new(Prepared {
            native: Arc::clone(self),
            request,
            geometry: [shape.rows, shape.outputs, shape.inputs],
            fingerprint,
        }))
    }
}

pub(super) struct Prepared {
    native: Arc<Native>,
    request: Option<NativeShape>,
    geometry: [u64; 3],
    pub(super) fingerprint: String,
}
impl Prepared {
    pub(super) fn fused(&self) -> bool {
        self.request.is_some()
    }
    pub(super) fn validate(
        &self,
        native: &Arc<Native>,
        shape: Shape,
        fingerprint: Option<&str>,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(native, &self.native) || fingerprint != Some(self.fingerprint.as_str()) {
            return Err("ConvRot prepared native owner or fingerprint differs".into());
        }
        if self.geometry != [shape.rows, shape.outputs, shape.inputs] {
            return Err("ConvRot prepared native geometry differs".into());
        }
        Ok(())
    }
    pub(super) unsafe fn run(
        &self,
        activation: u64,
        weight: u64,
        row_scale: u64,
        channel_scale: u64,
        output: u64,
        stream: *mut std::ffi::c_void,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let request = self.request.as_ref().ok_or_else(|| {
            CudaDeviceRuntimeError::contract("ConvRot native launch missing fused plan")
        })?;
        if stream.is_null() {
            return Err(CudaDeviceRuntimeError::contract(
                "ConvRot native launch requires an explicit stream",
            ));
        }
        let bindings = Bindings {
            activation,
            weight,
            row_scale,
            channel_scale,
            output,
        };
        bindings
            .validate(request)
            .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
        #[cfg(feature = "cuda-cutlass-int8-fused-output")]
        {
            status(unsafe { ferrum_int8_fused_output_run_v1(request, &bindings, stream) })
        }
        #[cfg(not(feature = "cuda-cutlass-int8-fused-output"))]
        {
            let _ = (request, bindings);
            Err(CudaDeviceRuntimeError::contract(
                "fused INT8 output artifact not compiled",
            ))
        }
    }
}
fn prepare(request: &NativeShape, info: &mut Info) -> Result<(), CudaDeviceRuntimeError> {
    #[cfg(feature = "cuda-cutlass-int8-fused-output")]
    {
        status(unsafe { ferrum_int8_fused_output_prepare_v1(request, info) })
    }
    #[cfg(not(feature = "cuda-cutlass-int8-fused-output"))]
    {
        let _ = (request, info);
        Err(CudaDeviceRuntimeError::contract(
            "fused INT8 output artifact not compiled",
        ))
    }
}
#[cfg(feature = "cuda-cutlass-int8-fused-output")]
fn status(code: i32) -> Result<(), CudaDeviceRuntimeError> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaDeviceRuntimeError::contract(format!(
            "CUTLASS fused INT8 output native status {code}"
        )))
    }
}
#[cfg(feature = "cuda-cutlass-int8-fused-output")]
unsafe extern "C" {
    fn ferrum_int8_fused_output_prepare_v1(request: *const NativeShape, info: *mut Info) -> i32;
    fn ferrum_int8_fused_output_run_v1(
        request: *const NativeShape,
        bindings: *const Bindings,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}
