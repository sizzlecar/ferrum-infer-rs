//! Direct F32 pointwise work over retained physical owners.
use super::*;
use ferrum_runtime::{F32PointwiseOp, F32SameShapeGeometry};

pub(crate) struct ReferenceF32PointwiseLaunch {
    pub input: ReferenceBufferRegion,
    pub update: Option<ReferenceBufferRegion>,
    pub output: ReferenceBufferRegion,
    pub geometry: F32SameShapeGeometry,
    pub kind: F32PointwiseOp,
}
impl ReferenceF32PointwiseLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        if self.update.is_some() != (self.kind.input_count() == 2) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "F32 pointwise operand arity differs",
            ));
        }
        for region in [&self.input, &self.output]
            .into_iter()
            .chain(self.update.as_ref())
        {
            region.validate_range()?;
            if region.element_type != ElementType::F32
                || self.geometry.elements().checked_mul(4) != Some(region.length_bytes as u64)
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 4 != 0
            {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "F32 pointwise requires exact aligned F32 extents",
                ));
            }
        }
        if self.output.overlaps(&self.input)
            || self
                .update
                .as_ref()
                .is_some_and(|u| self.output.overlaps(u))
        {
            return Err(ReferenceDeviceRuntimeError::contract(
                "F32 pointwise output aliases an input",
            ));
        }
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        self.input.validate_runtime(runtime)?;
        if let Some(update) = &self.update {
            update.validate_runtime(runtime)?;
        }
        self.output.validate_runtime(runtime)?;
        self.validate()
    }
}
pub(super) fn execute(launch: &ReferenceF32PointwiseLaunch) {
    if let Some(update) = &launch.update {
        execute_locked(
            launch,
            AllocationLocks::new([&launch.input, &launch.output, update]),
        );
    } else {
        execute_locked(
            launch,
            AllocationLocks::new([&launch.input, &launch.output]),
        );
    }
}
fn execute_locked<const N: usize>(
    launch: &ReferenceF32PointwiseLaunch,
    mut storage: AllocationLocks<'_, N>,
) {
    for i in 0..launch.geometry.elements() as usize {
        let x = storage.read_f32(0, i);
        let y = match launch.kind {
            F32PointwiseOp::ResidualAdd => x + storage.read_f32(2, i),
            F32PointwiseOp::Multiply => x * storage.read_f32(2, i),
            F32PointwiseOp::SiluMul => (x / (1.0_f32 + libm::expf(-x))) * storage.read_f32(2, i),
            F32PointwiseOp::Silu => x / (1.0_f32 + libm::expf(-x)),
        };
        storage.write_f32(1, i, y);
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::*;
    use ferrum_runtime::{ResolvedTensorLayout, ResolvedTensorSpec};
    fn launch(runtime: &ReferenceDeviceRuntime) -> ReferenceF32PointwiseLaunch {
        let spec = ResolvedTensorSpec::new(
            vec![1, 2],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        let data: Vec<_> = [1.000244140625_f32, -1.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        ReferenceF32PointwiseLaunch {
            input: buffer(
                runtime,
                "resource.pointwise_f32.x",
                &data,
                16,
                ElementType::F32,
            )
            .region(0..8)
            .unwrap(),
            update: Some(
                buffer(
                    runtime,
                    "resource.pointwise_f32.u",
                    &data,
                    16,
                    ElementType::F32,
                )
                .region(0..8)
                .unwrap(),
            ),
            output: buffer(
                runtime,
                "resource.pointwise_f32.y",
                &[0xa5; 8],
                16,
                ElementType::F32,
            )
            .region(0..8)
            .unwrap(),
            geometry: F32SameShapeGeometry::new(&spec, &spec, &spec).unwrap(),
            kind: F32PointwiseOp::ResidualAdd,
        }
    }
    #[test]
    fn physical_dtype_extent_alignment_and_alias_fail_before_writing() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        for mutation in 0..6 {
            let mut l = launch(&runtime);
            let original = l.output.clone();
            match mutation {
                0 => l.input.element_type = ElementType::Bf16,
                1 => l.update.as_mut().unwrap().length_bytes -= 4,
                2 => l.output.length_bytes -= 4,
                3 => {
                    l.input = buffer(
                        &runtime,
                        "resource.pointwise_f32.unaligned",
                        &[0; 12],
                        16,
                        ElementType::F32,
                    )
                    .region(1..9)
                    .unwrap()
                }
                4 => l.output = l.input.clone(),
                5 => l.output = l.update.clone().unwrap(),
                _ => unreachable!(),
            }
            assert!(ReferenceDeviceCommand::f32_pointwise_tensor(l).is_err());
            assert_eq!(original.read(), [0xa5; 8]);
        }
        assert_eq!(runtime.snapshot().submissions, 0);
    }
    #[test]
    fn unary_arity_is_checked_before_execution() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let mut l = launch(&runtime);
        let output = l.output.clone();
        l.kind = F32PointwiseOp::Silu;
        assert!(l.validate().is_err());
        l.update = None;
        assert!(l.validate().is_ok());
        l.kind = F32PointwiseOp::SiluMul;
        assert!(ReferenceDeviceCommand::f32_pointwise_tensor(l).is_err());
        assert_eq!(output.read(), [0xa5; 8]);
        assert_eq!(runtime.snapshot().submissions, 0);
    }
    #[test]
    fn multiplication_keeps_full_f32_product_and_requires_both_operands() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let mut l = launch(&runtime);
        l.kind = F32PointwiseOp::Multiply;
        let output = l.output.clone();
        let mut stream = runtime.create_stream().unwrap();
        runtime
            .submit_entries(
                &mut stream,
                vec![(
                    DeviceCommandPhase::Compute,
                    Some(0),
                    ReferenceDeviceCommand::f32_pointwise_tensor(l).unwrap(),
                )],
                false,
            )
            .unwrap();
        let got: Vec<f32> = output
            .read()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(got, [(1.000244140625f64.powi(2)) as f32, 1.]);
        let mut invalid = launch(&runtime);
        invalid.kind = F32PointwiseOp::Multiply;
        invalid.update = None;
        assert!(ReferenceDeviceCommand::f32_pointwise_tensor(invalid).is_err());
    }
    #[test]
    fn a_late_foreign_update_is_rejected_by_runtime_preflight() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
        let mut l = launch(&runtime);
        let original = l.output.clone();
        l.update = launch(&foreign).update;
        let command = ReferenceDeviceCommand::f32_pointwise_tensor(l).unwrap();
        let mut stream = runtime.create_stream().unwrap();
        assert!(runtime
            .submit_entries(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                false
            )
            .is_err());
        assert_eq!(original.read(), [0xa5; 8]);
        assert_eq!(runtime.snapshot().submissions, 0);
    }
}
