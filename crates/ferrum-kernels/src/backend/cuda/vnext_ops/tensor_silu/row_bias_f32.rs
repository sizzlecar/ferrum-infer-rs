//! Borrow the already admitted native module for a single F32 row broadcast.
//! This handle never loads a module or creates a BLAS handle or allocation.
use super::*;
use cudarc::driver::CudaStream;

#[derive(Clone)]
pub(in crate::backend::cuda::vnext_ops) struct RetainedF32RowBias {
    // Retains the functions, unique module permit, original TensorEagerState
    // and its bootstrap charge. A command holds the same Arc as its provider.
    kernels: Arc<RetainedFunctions>,
}

impl RetainedF32RowBias {
    pub(super) fn new(kernels: &Arc<RetainedFunctions>) -> Self {
        Self {
            kernels: Arc::clone(kernels),
        }
    }

    pub(in crate::backend::cuda::vnext_ops) fn fingerprint() -> String {
        implementation_fingerprint(&[
            b"row-major-f32-bias;elements=R*O;channels=O;spatial=1;fadd.rn;ftz=false",
            numerical_fingerprint().as_bytes(),
            include_bytes!("row_bias_f32.rs"),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/kernels/vnext_cnn_f32.cuh"
            )),
        ])
    }

    pub(in crate::backend::cuda::vnext_ops) fn validate_regions(
        &self,
        regions: &[CudaBufferRegion],
    ) -> Result<(), String> {
        if regions.is_empty()
            || regions
                .iter()
                .any(|region| region.runtime_instance() != self.kernels.runtime_instance())
        {
            return Err("F32 row bias module and buffers belong to different CUDA runtimes".into());
        }
        Ok(())
    }

    pub(in crate::backend::cuda::vnext_ops) fn launch(
        &self,
        stream: &CudaStream,
        bias: &CudaBufferRegion,
        output: &CudaBufferRegion,
        rows: u64,
        channels: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        // Geometry/extent/noalias checks already ran before encoding SGEMM.
        let elements = rows
            .checked_mul(channels)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("F32 row bias extent overflows"))?;
        let spatial = 1u64;
        let output = output.device_ptr();
        let bias = bias.device_ptr();
        let mut launch = stream.launch_builder(self.kernels.cnn_f32_function(1));
        launch
            .arg(&output)
            .arg(&bias)
            .arg(&elements)
            .arg(&channels)
            .arg(&spatial);
        let config = LaunchConfig {
            grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|error| CudaDeviceRuntimeError::driver("F32 row bias broadcast", error))
    }
}
