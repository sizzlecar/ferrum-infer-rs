//! Bias is added only after the dense F32 result has completed.
use super::*;

pub(crate) struct ReferenceAffineF32Launch {
    pub(crate) dense: ReferenceDenseLinearLaunch,
    pub(crate) bias: ReferenceBufferRegion,
}
impl ReferenceAffineF32Launch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        self.dense.validate()?;
        self.bias.validate_range()?;
        if self.dense.scalar != ReferenceDenseScalar::F32
            || self.bias.element_type != ElementType::F32
            || self.dense.out_features.checked_mul(4) != Some(self.bias.length_bytes)
            || (self.bias.offset_bytes - self.bias.allocation.logical_offset_bytes) % 4 != 0
            || self.dense.output.overlaps(&self.bias)
        {
            return Err(ReferenceDeviceRuntimeError::contract(
                "F32 affine bias dtype, extent, alignment or output alias differs",
            ));
        }
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in [
            &self.dense.input,
            &self.dense.weight,
            &self.bias,
            &self.dense.output,
        ] {
            region.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(launch: &ReferenceAffineF32Launch) {
    execute_dense_linear(&launch.dense);
    let mut storage = AllocationLocks::new([&launch.dense.output, &launch.bias]);
    for index in 0..launch.dense.rows * launch.dense.out_features {
        let value =
            storage.read_f32(0, index) + storage.read_f32(1, index % launch.dense.out_features);
        storage.write_f32(0, index, value);
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::*;
    #[test]
    fn f32_affine_bias_preflight_rejects_alias_precision_and_extent_without_writes() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let input = buffer(
            &runtime,
            "resource.affine.input",
            &[0_u8; 8],
            16,
            ElementType::F32,
        );
        let weight = buffer(
            &runtime,
            "resource.affine.weight",
            &[0_u8; 8],
            16,
            ElementType::F32,
        );
        let output = buffer(
            &runtime,
            "resource.affine.output",
            &[0xa5_u8; 4],
            16,
            ElementType::F32,
        );
        let bad_type = buffer(
            &runtime,
            "resource.affine.bias_half",
            &[0_u8; 4],
            16,
            ElementType::Bf16,
        );
        for bias in [
            output.region(0..4).unwrap(),
            weight.region(0..8).unwrap(),
            bad_type.region(0..4).unwrap(),
        ] {
            let launch = ReferenceAffineF32Launch {
                dense: ReferenceDenseLinearLaunch {
                    input: input.region(0..8).unwrap(),
                    weight: weight.region(0..8).unwrap(),
                    output: output.region(0..4).unwrap(),
                    rows: 1,
                    in_features: 2,
                    out_features: 1,
                    scalar: ReferenceDenseScalar::F32,
                },
                bias,
            };
            assert!(ReferenceDeviceCommand::affine_f32_tensor(launch).is_err());
            assert_eq!(output.region(0..4).unwrap().read(), [0xa5_u8; 4]);
        }
        assert_eq!(runtime.snapshot().submissions, 0);
    }

    #[test]
    fn f32_affine_adds_bias_after_complete_f32_dot() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let f32_bytes = |values: &[f32]| {
            values
                .iter()
                .copied()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        };
        let input = buffer(
            &runtime,
            "resource.affine.order.input",
            &f32_bytes(&[1.0e10, 1.0, -1.0e10]),
            16,
            ElementType::F32,
        );
        let weight = buffer(
            &runtime,
            "resource.affine.order.weight",
            &f32_bytes(&[1.0, 1.0, 1.0]),
            16,
            ElementType::F32,
        );
        let bias = buffer(
            &runtime,
            "resource.affine.order.bias",
            &f32_bytes(&[1.000244140625]),
            16,
            ElementType::F32,
        );
        let output = buffer(
            &runtime,
            "resource.affine.order.output",
            &[0xa5_u8; 4],
            16,
            ElementType::F32,
        );
        let launch = ReferenceAffineF32Launch {
            dense: ReferenceDenseLinearLaunch {
                input: input.region(0..12).unwrap(),
                weight: weight.region(0..12).unwrap(),
                output: output.region(0..4).unwrap(),
                rows: 1,
                in_features: 3,
                out_features: 1,
                scalar: ReferenceDenseScalar::F32,
            },
            bias: bias.region(0..4).unwrap(),
        };
        launch.validate_runtime(runtime.runtime_instance).unwrap();
        execute(&launch);
        assert_eq!(
            output.region(0..4).unwrap().read(),
            f32_bytes(&[1.000244140625])
        );
    }
}
