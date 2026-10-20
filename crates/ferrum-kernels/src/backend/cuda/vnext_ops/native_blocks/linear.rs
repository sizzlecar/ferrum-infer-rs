//! Select each actual matrix launch once, then share it with execution facts.
use super::*;
use ferrum_runtime::{ElementType, HadamardApplication};

#[derive(Clone, Copy)]
enum LinearKernel {
    LinearQ4kF16,
    LinearQ4kTiledF16,
    LinearQ80F16,
    LinearQ5kF16,
    LinearQ5kTiledF16,
    LinearIq4xsF16,
    LinearIq4xsTiledF16,
    LinearQ6kF16,
    LinearQ6kTiledF16,
    LinearQ6kF32,
    LinearQ6kTiledF32,
    LinearF16,
    LinearF32,
    LinearTiledF16,
    LinearTiledF32,
    LinearF32F16,
    LinearTiledF32F16,
}
impl LinearKernel {
    fn function(self, kernels: &CudaNativeBlockKernels) -> &CudaFunction {
        match self {
            Self::LinearQ4kF16 => &kernels.linear_q4k_f16,
            Self::LinearQ4kTiledF16 => &kernels.linear_q4k_tiled_f16,
            Self::LinearQ80F16 => &kernels.linear_q8_0_f16,
            Self::LinearQ5kF16 => &kernels.linear_q5k_f16,
            Self::LinearQ5kTiledF16 => &kernels.linear_q5k_tiled_f16,
            Self::LinearIq4xsF16 => &kernels.linear_iq4xs_f16,
            Self::LinearIq4xsTiledF16 => &kernels.linear_iq4xs_tiled_f16,
            Self::LinearQ6kF16 => &kernels.linear_q6k_f16,
            Self::LinearQ6kTiledF16 => &kernels.linear_q6k_tiled_f16,
            Self::LinearQ6kF32 => &kernels.linear_q6k_f32,
            Self::LinearQ6kTiledF32 => &kernels.linear_q6k_tiled_f32,
            Self::LinearF16 => &kernels.linear_f16,
            Self::LinearF32 => &kernels.linear_f32,
            Self::LinearTiledF16 => &kernels.linear_tiled_f16,
            Self::LinearTiledF32 => &kernels.linear_tiled_f32,
            Self::LinearF32F16 => &kernels.linear_f32_f16,
            Self::LinearTiledF32F16 => &kernels.linear_tiled_f32_f16,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::LinearQ4kF16 => "vnext_gguf_linear_q4k_f16",
            Self::LinearQ4kTiledF16 => "vnext_gguf_linear_q4k_tiled_f16",
            Self::LinearQ80F16 => "vnext_gguf_linear_q8_0_f16",
            Self::LinearQ5kF16 => "vnext_gguf_linear_q5k_f16",
            Self::LinearQ5kTiledF16 => "vnext_gguf_linear_q5k_tiled_f16",
            Self::LinearIq4xsF16 => "vnext_gguf_linear_iq4xs_f16",
            Self::LinearIq4xsTiledF16 => "vnext_gguf_linear_iq4xs_tiled_f16",
            Self::LinearQ6kF16 => "vnext_gguf_linear_q6k_f16",
            Self::LinearQ6kTiledF16 => "vnext_gguf_linear_q6k_tiled_f16",
            Self::LinearQ6kF32 => "vnext_gguf_linear_q6k_f32",
            Self::LinearQ6kTiledF32 => "vnext_gguf_linear_q6k_tiled_f32",
            Self::LinearF16 => "vnext_gguf_linear_f16",
            Self::LinearF32 => "vnext_gguf_linear_f32",
            Self::LinearTiledF16 => "vnext_gguf_linear_tiled_f16",
            Self::LinearTiledF32 => "vnext_gguf_linear_tiled_f32",
            Self::LinearF32F16 => "vnext_gguf_linear_f32_f16",
            Self::LinearTiledF32F16 => "vnext_gguf_linear_tiled_f32_f16",
        }
    }
}

#[derive(Clone, Copy)]
pub(in crate::backend::cuda::vnext_ops) struct NativeLinearSelection {
    kernel: LinearKernel,
    parameters: [u32; 8],
    row_tile: u32,
}
impl NativeLinearSelection {
    pub(in crate::backend::cuda::vnext_ops) fn implementation(self) -> &'static str {
        self.kernel.name()
    }
    pub(in crate::backend::cuda::vnext_ops) fn row_tile(self) -> u32 {
        self.row_tile
    }
    pub(in crate::backend::cuda::vnext_ops) fn launch(self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (
                self.parameters[2].div_ceil(4),
                self.parameters[0].div_ceil(self.row_tile),
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub(in crate::backend::cuda::vnext_ops) struct TransformedLinearSelection {
    pub(in crate::backend::cuda::vnext_ops) linear: NativeLinearSelection,
    pub(in crate::backend::cuda::vnext_ops) transform: Option<hadamard::HadamardSelection>,
}

impl CudaNativeBlockKernels {
    pub(in crate::backend::cuda::vnext_ops) fn select_transformed_linear(
        &self,
        part: &weights::MatrixPart,
        rows: u32,
        output_stride: u32,
        activation: ElementType,
    ) -> Result<TransformedLinearSelection, CudaDeviceRuntimeError> {
        let transform = part
            .transform
            .as_ref()
            .map(|spec| {
                if !matches!(spec.application, HadamardApplication::BeforeMatmul { .. }) {
                    return Err(CudaDeviceRuntimeError::contract(
                        "native linear requires a forward transform",
                    ));
                }
                hadamard::HadamardSelection::select(
                    rows,
                    part.columns,
                    activation,
                    ElementType::F32,
                    spec,
                )
            })
            .transpose()?;
        let input_type = if transform.is_some() {
            ElementType::F32
        } else {
            activation
        };
        Ok(TransformedLinearSelection {
            linear: self.select_linear(part, rows, output_stride, input_type, activation)?,
            transform,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend::cuda::vnext_ops) fn launch_transformed_linear(
        &self,
        stream: &CudaStream,
        input: u64,
        weight: u64,
        output: u64,
        selection: TransformedLinearSelection,
        signs: u64,
        scratch: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let input = if let Some(transform) = selection.transform {
            if scratch == 0 {
                return Err(CudaDeviceRuntimeError::contract(
                    "native linear requires planned F32 scratch",
                ));
            }
            self.hadamard
                .launch_selected(stream, input, scratch, signs, transform)?;
            scratch
        } else {
            input
        };
        self.launch_linear(stream, input, weight, output, selection.linear)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend::cuda::vnext_ops) fn transformed_linear(
        &self,
        stream: &CudaStream,
        input: u64,
        weight: u64,
        output: u64,
        part: &weights::MatrixPart,
        rows: u32,
        output_stride: u32,
        activation: ferrum_runtime::ElementType,
        signs: u64,
        scratch: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let selection = self.select_transformed_linear(part, rows, output_stride, activation)?;
        self.launch_transformed_linear(stream, input, weight, output, selection, signs, scratch)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend::cuda::vnext_ops) fn linear_with_precision(
        &self,
        stream: &CudaStream,
        input: u64,
        weight: u64,
        output: u64,
        part: &weights::MatrixPart,
        rows: u32,
        output_stride: u32,
        input_type: ferrum_runtime::ElementType,
        output_type: ferrum_runtime::ElementType,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let selection = self.select_linear(part, rows, output_stride, input_type, output_type)?;
        self.launch_linear(stream, input, weight, output, selection)
    }

    fn select_linear(
        &self,
        part: &weights::MatrixPart,
        rows: u32,
        output_stride: u32,
        input_type: ElementType,
        output_type: ElementType,
    ) -> Result<NativeLinearSelection, CudaDeviceRuntimeError> {
        use ferrum_runtime::ElementType;
        if rows == 0
            || rows > u16::MAX as u32
            || part
                .output_offset
                .checked_add(part.rows)
                .is_none_or(|end| end > output_stride)
        {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA native linear launch extent is invalid",
            ));
        }
        let q4k =
            part.format == weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q4K);
        let q8_0 =
            part.format == weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q8_0);
        let q5k =
            part.format == weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q5K);
        let iq4xs =
            part.format == weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Iq4Xs);
        let q6k =
            part.format == weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q6K);
        let row_tile = if q8_0 && input_type == ElementType::F16 && output_type == ElementType::F16
        {
            q8_0_f16_row_tile(rows, part.columns, part.rows, self.multiprocessors)
        } else if rows > 1 {
            LINEAR_ROW_TILE
        } else {
            1
        };
        let kernel = match (input_type, output_type, row_tile > 1) {
            (ElementType::F16, ElementType::F16, false) if q4k => LinearKernel::LinearQ4kF16,
            (ElementType::F16, ElementType::F16, true) if q4k => LinearKernel::LinearQ4kTiledF16,
            (ElementType::F16, ElementType::F16, false) if q8_0 => LinearKernel::LinearQ80F16,
            (ElementType::F16, ElementType::F16, false) if q5k => LinearKernel::LinearQ5kF16,
            (ElementType::F16, ElementType::F16, true) if q5k => LinearKernel::LinearQ5kTiledF16,
            (ElementType::F16, ElementType::F16, false) if iq4xs => LinearKernel::LinearIq4xsF16,
            (ElementType::F16, ElementType::F16, true) if iq4xs => {
                LinearKernel::LinearIq4xsTiledF16
            }
            (ElementType::F16, ElementType::F16, false) if q6k => LinearKernel::LinearQ6kF16,
            (ElementType::F16, ElementType::F16, true) if q6k => LinearKernel::LinearQ6kTiledF16,
            (ElementType::F32, ElementType::F32, false) if q6k => LinearKernel::LinearQ6kF32,
            (ElementType::F32, ElementType::F32, true) if q6k => LinearKernel::LinearQ6kTiledF32,
            (ElementType::F16, ElementType::F16, false) => LinearKernel::LinearF16,
            (ElementType::F32, ElementType::F32, false) => LinearKernel::LinearF32,
            (ElementType::F16, ElementType::F16, true) => LinearKernel::LinearTiledF16,
            (ElementType::F32, ElementType::F32, true) => LinearKernel::LinearTiledF32,
            (ElementType::F32, ElementType::F16, false) => LinearKernel::LinearF32F16,
            (ElementType::F32, ElementType::F16, true) => LinearKernel::LinearTiledF32F16,
            _ => {
                return Err(CudaDeviceRuntimeError::contract(
                    "CUDA native linear activation dtype is unsupported",
                ))
            }
        };
        let [format, values, bytes] = part.format.parameters();
        let parameters = [
            rows,
            part.columns,
            part.rows,
            output_stride,
            part.output_offset,
            format,
            values,
            bytes,
        ];
        Ok(NativeLinearSelection {
            kernel,
            parameters,
            row_tile,
        })
    }

    fn launch_linear(
        &self,
        stream: &CudaStream,
        input: u64,
        weight: u64,
        output: u64,
        selection: NativeLinearSelection,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let mut launch = stream.launch_builder(selection.kernel.function(self));
        launch.arg(&input).arg(&weight).arg(&output);
        for parameter in &selection.parameters {
            launch.arg(parameter);
        }
        // SAFETY: The provider retains exact row-complete regions. Selection
        // preserves the original complete-warp and partial-tile guards.
        unsafe { launch.launch(selection.launch()) }
            .map(|_| ())
            .map_err(|error| CudaDeviceRuntimeError::driver("native matrix linear launch", error))
    }
}
