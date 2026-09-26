//! Exact, shape-limited fusion of two independent strict Q8_0 projections.
//! This changes launch topology only; NativeQ8 activation-quantized projections
//! do not consume this plan.
use super::*;
use ferrum_interfaces::vnext::ElementType;
use weights::{MatrixFormat, MatrixPart};

pub(in crate::backend::cuda::vnext_ops) const SELECTOR_VERSION: &str =
    "strict-q8-pair-f16-m4-m8-k4096-n32-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::backend::cuda::vnext_ops) struct PairPlan {
    rows: u32,
    columns: u32,
    outputs: u32,
    stride: u32,
    offsets: [u32; 2],
    format: [u32; 3],
}
impl PairPlan {
    pub(in crate::backend::cuda::vnext_ops) fn select(
        first: &MatrixPart,
        second: &MatrixPart,
        rows: u32,
        input_features: u32,
        output_stride: u32,
        input_type: ElementType,
        output_type: ElementType,
    ) -> Option<Self> {
        if !matches!(rows, 4 | 8)
            || input_type != ElementType::F16
            || output_type != ElementType::F16
            || input_features != 4096
        {
            return None;
        }
        for part in [first, second] {
            if part.format != MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q8_0)
                || part.columns != input_features
                || part.rows != 32
                || part.transform.is_some()
                || part.signs_region.is_some()
                || part.output_offset.checked_add(part.rows)? > output_stride
            {
                return None;
            }
        }
        let end0 = first.output_offset.checked_add(first.rows)?;
        let end1 = second.output_offset.checked_add(second.rows)?;
        if first.output_offset < end1 && second.output_offset < end0 {
            return None;
        }
        Some(Self {
            rows,
            columns: input_features,
            outputs: first.rows,
            stride: output_stride,
            offsets: [first.output_offset, second.output_offset],
            format: first.format.parameters(),
        })
    }

    pub(in crate::backend::cuda::vnext_ops) fn parameters(self) -> [u32; 9] {
        [
            self.rows,
            self.columns,
            self.outputs,
            self.stride,
            self.offsets[0],
            self.offsets[1],
            self.format[0],
            self.format[1],
            self.format[2],
        ]
    }

    pub(in crate::backend::cuda::vnext_ops) fn launch_config(self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.outputs.div_ceil(4), self.rows.div_ceil(8), 2),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// A single row chunk. The caller already checked the complete composite matrix
/// and retained both independently authorized physical weight regions.
pub(in crate::backend::cuda::vnext_ops) fn dispatches(
    parts: &[MatrixPart],
    rows: u32,
    input_features: u32,
    output_stride: u32,
) -> u64 {
    let mut index = 0;
    let mut count = 0;
    while index < parts.len() {
        let first = &parts[index];
        let paired = parts.get(index + 1).and_then(|second| {
            PairPlan::select(
                first,
                second,
                rows,
                input_features,
                output_stride,
                ElementType::F16,
                ElementType::F16,
            )
        });
        if paired.is_some() {
            count += 1;
            index += 2;
        } else {
            count += 1 + u64::from(first.transform.is_some());
            index += 1;
        }
    }
    count
}

impl CudaNativeBlockKernels {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend::cuda::vnext_ops) fn strict_q8_pair_parts<'a>(
        &self,
        stream: &CudaStream,
        parts: impl Iterator<Item = (&'a MatrixPart, u64, u64)>,
        input: u64,
        output: u64,
        rows: u32,
        input_features: u32,
        output_stride: u32,
        transform_scratch: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let mut pending = parts.peekable();
        while let Some((part, weight, signs)) = pending.next() {
            let pair = pending
                .peek()
                .and_then(|(second, _, _)| {
                    PairPlan::select(
                        part,
                        second,
                        rows,
                        input_features,
                        output_stride,
                        ElementType::F16,
                        ElementType::F16,
                    )
                })
                .filter(|_| self.q8_pair_enabled());
            if let Some(plan) = pair {
                let (_, second_weight, _) = pending.next().expect("selected second physical part");
                self.linear_q8_pair(stream, input, [weight, second_weight], output, plan)?;
            } else {
                self.transformed_linear(
                    stream,
                    input,
                    weight,
                    output,
                    part,
                    rows,
                    output_stride,
                    ElementType::F16,
                    signs,
                    transform_scratch,
                )?;
            }
        }
        Ok(())
    }

    pub(in crate::backend::cuda::vnext_ops) fn q8_pair_enabled(&self) -> bool {
        #[cfg(test)]
        if self.q8_pair_disabled {
            return false;
        }
        true
    }

    #[cfg(test)]
    pub(in crate::backend::cuda::vnext_ops) fn with_unpaired_q8_control(&self) -> Self {
        let mut control = self.clone();
        control.q8_pair_disabled = true;
        control
    }

    pub(in crate::backend::cuda::vnext_ops) fn linear_q8_pair(
        &self,
        stream: &CudaStream,
        input: u64,
        weights: [u64; 2],
        output: u64,
        plan: PairPlan,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let parameters = plan.parameters();
        let mut launch = stream.launch_builder(&self.linear_q8_pair_tiled_f16);
        launch
            .arg(&input)
            .arg(&weights[0])
            .arg(&weights[1])
            .arg(&output);
        for parameter in &parameters {
            launch.arg(parameter);
        }
        // Same original generic RowTile8 body per z-plane, distinct checked
        // output ranges, existing invocation-owned input/weight/output leases.
        unsafe { launch.launch(plan.launch_config()) }
            .map_err(|error| CudaDeviceRuntimeError::driver("native Q8 pair projection", error))?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "q8_pair/tests.rs"]
mod tests;
