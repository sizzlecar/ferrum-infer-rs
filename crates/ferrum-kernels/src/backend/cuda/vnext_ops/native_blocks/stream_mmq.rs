//! Explicit physical-B8 gate/up policy; all other geometries retain strict native math.
use super::weights;
use crate::backend::cuda::vnext_runtime::CudaDeviceRuntimeError;
use crate::gguf_blocks::GgufBlockFormat;
use cudarc::driver::{sys, CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

pub(in crate::backend::cuda::vnext_ops) const PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/vnext_q4_stream_mmq.ptx"));
pub(in crate::backend::cuda::vnext_ops) const OPERATION: &str =
    ferrum_interfaces::vnext::DENSE_SWIGLU_Q8_GATE_UP_STREAM_MMQ_OPERATION_ID;
const SHARED_BYTES: u32 = 45696;

#[derive(Clone)]
pub(in crate::backend::cuda::vnext_ops) struct StreamMmq {
    pack: CudaFunction,
    project: CudaFunction,
    fixup: CudaFunction,
    cta_budget: u32,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::backend::cuda::vnext_ops) struct Workspace {
    pub words_bytes: u64,
    pub scales_bytes: u64,
    pub partial_offset: u64,
    pub total_bytes: u64,
    pub ctas: u32,
}

fn invalid(message: &str) -> CudaDeviceRuntimeError {
    CudaDeviceRuntimeError::contract(message)
}

/// Validate the complete fused gate/up matrix, including both output spans.
pub(in crate::backend::cuda::vnext_ops) fn eligible(
    parts: &[weights::MatrixPart],
    rows: u32,
    hidden: u32,
    intermediate: u32,
) -> bool {
    rows == 8
        && hidden > 0
        && hidden % 256 == 0
        && intermediate > 0
        && parts.len() == 2
        && parts.iter().enumerate().all(|(index, part)| {
            part.format == weights::MatrixFormat::Block(GgufBlockFormat::Q4K)
                && part.columns == hidden
                && part.rows == intermediate
                && part.output_offset
                    == (index as u32).checked_mul(intermediate).unwrap_or(u32::MAX)
                && part.transform.is_none()
                && part.signs_region.is_none()
        })
}

impl Workspace {
    fn new(hidden: u32, intermediate: u32, cta_budget: u32) -> Result<Self, String> {
        if hidden == 0 || hidden % 256 != 0 || intermediate == 0 || cta_budget == 0 {
            return Err("Stream-K workspace requires complete K256 and nonzero extents".into());
        }
        let tiles = u64::from(intermediate).div_ceil(128);
        let units = tiles
            .checked_mul(u64::from(hidden / 256))
            .ok_or("Stream-K units overflow")?;
        let ctas =
            u32::try_from(units.min(u64::from(cta_budget))).map_err(|_| "Stream-K CTA overflow")?;
        let words_bytes = u64::from(hidden)
            .checked_mul(8)
            .ok_or("Stream-K pack overflow")?;
        let scales_bytes = words_bytes / 8;
        let partial_offset = words_bytes
            .checked_add(scales_bytes * 2)
            .ok_or("Stream-K metadata overflow")?;
        let partial_bytes = tiles
            .checked_add(u64::from(ctas))
            .and_then(|n| n.checked_mul(4096))
            .ok_or("Stream-K partial bytes overflow")?;
        let total_bytes = partial_offset
            .checked_add(partial_bytes)
            .ok_or("Stream-K workspace overflow")?;
        Ok(Self {
            words_bytes,
            scales_bytes,
            partial_offset,
            total_bytes,
            ctas,
        })
    }
}

impl StreamMmq {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self, CudaDeviceRuntimeError> {
        let module = ctx
            .load_module(Ptx::from_src(PTX))
            .map_err(|e| CudaDeviceRuntimeError::driver("Stream-MMQ module", e))?;
        let load = |name| {
            module
                .load_function(name)
                .map_err(|e| CudaDeviceRuntimeError::driver("Stream-MMQ function", e))
        };
        let project = load("vnext_q4_stream_mmq")?;
        project
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                SHARED_BYTES as i32,
            )
            .map_err(|e| CudaDeviceRuntimeError::driver("Stream-MMQ shared memory", e))?;
        let sm = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| CudaDeviceRuntimeError::driver("Stream-MMQ SM count", e))?;
        let active = project
            .occupancy_max_active_blocks_per_multiprocessor(256, SHARED_BYTES as usize, None)
            .map_err(|e| CudaDeviceRuntimeError::driver("Stream-MMQ occupancy", e))?;
        let cta_budget = u32::try_from(sm)
            .ok()
            .and_then(|sm| sm.checked_mul(active))
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid("Stream-MMQ has no resident CTA capacity"))?;
        Ok(Self {
            pack: load("vnext_q4_stream_pack")?,
            fixup: load("vnext_q4_stream_fixup")?,
            project,
            cta_budget,
        })
    }

    pub fn workspace(&self, hidden: u32, intermediate: u32) -> Result<Workspace, String> {
        Workspace::new(hidden, intermediate, self.cta_budget)
    }

    /// Caller owns the invocation scratch and retained input/weights/output regions.
    pub fn launch_gate_up(
        &self,
        stream: &CudaStream,
        parts: &[weights::MatrixPart],
        weights: &[u64],
        input: u64,
        output: u64,
        rows: u32,
        hidden: u32,
        intermediate: u32,
        scratch: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        if !eligible(parts, rows, hidden, intermediate) || weights.len() != parts.len() {
            return Err(invalid(
                "Stream-MMQ launch geometry differs from selected route",
            ));
        }
        let layout = self
            .workspace(hidden, intermediate)
            .map_err(CudaDeviceRuntimeError::contract)?;
        let address = |offset| {
            scratch
                .checked_add(offset)
                .ok_or_else(|| invalid("Stream-MMQ pointer overflow"))
        };
        let q = scratch;
        let d = address(layout.words_bytes)?;
        let sums = address(layout.words_bytes + layout.scales_bytes)?;
        let partial = address(layout.partial_offset)?;
        let stride = intermediate
            .checked_mul(2)
            .ok_or_else(|| invalid("Stream-MMQ stride overflow"))?;
        unsafe {
            stream
                .launch_builder(&self.pack)
                .arg(&input)
                .arg(&q)
                .arg(&d)
                .arg(&sums)
                .arg(&rows)
                .arg(&hidden)
                .launch(LaunchConfig {
                    grid_dim: ((rows * (hidden / 32)).div_ceil(8), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| CudaDeviceRuntimeError::driver("Stream-MMQ pack", e))?;
        for (part, weight) in parts.iter().zip(weights) {
            unsafe {
                stream
                    .launch_builder(&self.project)
                    .arg(&q)
                    .arg(&d)
                    .arg(&sums)
                    .arg(weight)
                    .arg(&partial)
                    .arg(&rows)
                    .arg(&hidden)
                    .arg(&intermediate)
                    .arg(&layout.ctas)
                    .launch(LaunchConfig {
                        grid_dim: (layout.ctas, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: SHARED_BYTES,
                    })
            }
            .map_err(|e| CudaDeviceRuntimeError::driver("Stream-MMQ projection", e))?;
            let values = rows
                .checked_mul(intermediate)
                .ok_or_else(|| invalid("Stream-MMQ output extent overflow"))?;
            unsafe {
                stream
                    .launch_builder(&self.fixup)
                    .arg(&partial)
                    .arg(&output)
                    .arg(&rows)
                    .arg(&hidden)
                    .arg(&intermediate)
                    .arg(&stride)
                    .arg(&part.output_offset)
                    .arg(&layout.ctas)
                    .launch(LaunchConfig::for_num_elems(values))
            }
            .map_err(|e| CudaDeviceRuntimeError::driver("Stream-MMQ fixup", e))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
