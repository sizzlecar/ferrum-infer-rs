//! Reuse the real managed affine/copy bodies after independent F32 geometry
//! checks. Original rank-five shape remains in the provider's semantic proof.
use super::*;
use ferrum_runtime::ReshapeCopyF32Geometry;
pub(crate) struct ReferenceReshapeCopyF32Launch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: ReshapeCopyF32Geometry,
}
impl ReferenceReshapeCopyF32Launch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in [&self.input, &self.output] {
            region.validate_range()?;
            if region.element_type != ElementType::F32
                || region.length_bytes as u64 != self.geometry.bytes()
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 4 != 0
            {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "F32 reshape exact dtype/extent/alignment differs",
                ));
            }
        }
        if self.input.overlaps(&self.output) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "F32 reshape cannot alias its input",
            ));
        }
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        self.input.validate_runtime(runtime)?;
        self.output.validate_runtime(runtime)?;
        self.validate()
    }
}
impl ReferenceDeviceCommand {
    pub(crate) fn patch_projection_f32_tensor(
        launch: ReferenceAffineF32Launch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let mut command = Self::affine_f32_tensor(launch)?;
        command.operation = "vnext_patch_projection_f32";
        Ok(command)
    }
    pub(crate) fn reshape_copy_f32_tensor(
        launch: ReferenceReshapeCopyF32Launch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        Ok(Self {
            operation: "vnext_reshape_copy_f32",
            logical_work: Some(
                DeviceCommandLogicalWork::tensor(launch.geometry.elements(), 0)
                    .map_err(|e| ReferenceDeviceRuntimeError::contract(e.to_string()))?,
            ),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::ReshapeCopyF32 { launch },
        })
    }
}
pub(super) fn execute(launch: &ReferenceReshapeCopyF32Launch) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.output]);
    for i in 0..launch.geometry.elements() as usize {
        let value = storage.read_f32(0, i);
        storage.write_f32(1, i, value);
    }
}
