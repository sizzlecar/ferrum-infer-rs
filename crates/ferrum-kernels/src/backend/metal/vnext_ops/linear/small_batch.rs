use super::*;

pub(super) const SHADER_SOURCE: &str = include_str!("small_batch.metal");

pub(super) struct SmallBatchPipelines {
    q4: [ComputePipelineState; 3],
    q6: [ComputePipelineState; 3],
    q6_f32: [ComputePipelineState; 3],
}

impl SmallBatchPipelines {
    pub(super) fn new(device: &Device) -> Result<Self, MetalDeviceRuntimeError> {
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(|error| {
                MetalDeviceRuntimeError::contract(format!("compile small-batch GEMV: {error}"))
            })?;
        let pipeline = |name: &str| {
            let function = library.get_function(name, None).map_err(|error| {
                MetalDeviceRuntimeError::contract(format!("load {name}: {error}"))
            })?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| {
                    MetalDeviceRuntimeError::contract(format!("build {name}: {error}"))
                })
        };
        Ok(Self {
            q4: [
                pipeline("q4_shared_b2")?,
                pipeline("q4_shared_b3")?,
                pipeline("q4_shared_b4")?,
            ],
            q6: [
                pipeline("q6_shared_b2")?,
                pipeline("q6_shared_b3")?,
                pipeline("q6_shared_b4")?,
            ],
            q6_f32: [
                pipeline("q6_shared_f32_b2")?,
                pipeline("q6_shared_f32_b3")?,
                pipeline("q6_shared_f32_b4")?,
            ],
        })
    }

    pub(super) fn pipeline(
        &self,
        format: LinearPhysicalFormat,
        rows: u32,
    ) -> Option<&ComputePipelineState> {
        let index = rows.checked_sub(2)? as usize;
        match format {
            LinearPhysicalFormat::Q4K => self.q4.get(index),
            LinearPhysicalFormat::Q6K => self.q6.get(index),
            _ => None,
        }
    }

    pub(super) fn f32_pipeline(
        &self,
        format: LinearPhysicalFormat,
        rows: u32,
    ) -> Option<&ComputePipelineState> {
        let index = rows.checked_sub(2)? as usize;
        match format {
            LinearPhysicalFormat::Q6K => self.q6_f32.get(index),
            _ => None,
        }
    }
}
