//! Same F32 expression order as the CUDA contract; libm exp2 and RN32 div
//! are a declared CPU reference approximation, not a PTX transcendental oracle.
use super::transcendental::silu_mul_bf16_bits;
use super::*;
use ferrum_runtime::{PackedSwiGluBf16Geometry, PackedSwiGluBf16Profile};

pub(crate) struct ReferencePackedSwiGluBf16Launch {
    pub input: ReferenceBufferRegion,
    pub output: ReferenceBufferRegion,
    pub geometry: PackedSwiGluBf16Geometry,
    pub profile: PackedSwiGluBf16Profile,
}
impl ReferencePackedSwiGluBf16Launch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        for (region, bytes) in [
            (&self.input, self.geometry.input_bytes()),
            (&self.output, self.geometry.output_bytes()),
        ] {
            region.validate_range()?;
            if region.element_type != ElementType::Bf16
                || region.length_bytes as u64 != bytes
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "packed SwiGLU requires exact aligned BF16 extents",
                ));
            }
        }
        if self.input.overlaps(&self.output) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "packed SwiGLU output aliases its input",
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
fn value(value: f32, gate: f32) -> u16 {
    let negative = 0.0f32 - gate;
    let exponent = negative * f32::from_bits(0x3fb8aa3b);
    let e = libm::exp2f(exponent);
    let denominator = 1.0f32 + e;
    let sigmoid = 1.0f32 / denominator;
    let silu = gate * sigmoid;
    half::bf16::from_f32(value * silu).to_bits()
}
pub(super) fn execute(launch: &ReferencePackedSwiGluBf16Launch) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.output]);
    let width = launch.geometry.width() as usize;
    for index in 0..launch.geometry.elements() as usize {
        let source = (index / width) * width * 2 + index % width;
        let v = storage.read_bits(0, source);
        let gate = storage.read_bits(0, source + width);
        let output = match launch.profile {
            PackedSwiGluBf16Profile::Exp2SigmoidF32SingleRound => value(
                half::bf16::from_bits(v).to_f32(),
                half::bf16::from_bits(gate).to_f32(),
            ),
            PackedSwiGluBf16Profile::SiluBf16ThenMultiply => silu_mul_bf16_bits(gate, v),
        };
        storage.write_bits(1, index, output);
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::*;
    use ferrum_runtime::{ResolvedTensorLayout, ResolvedTensorSpec};
    fn launch(runtime: &ReferenceDeviceRuntime) -> ReferencePackedSwiGluBf16Launch {
        let x = ResolvedTensorSpec::new(
            vec![1, 4],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        let y = ResolvedTensorSpec::new(
            vec![1, 2],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        ReferencePackedSwiGluBf16Launch {
            input: buffer(
                runtime,
                "resource.packed_swiglu.x",
                &[0; 8],
                16,
                ElementType::Bf16,
            )
            .region(0..8)
            .unwrap(),
            output: buffer(
                runtime,
                "resource.packed_swiglu.y",
                &[0xa5; 4],
                16,
                ElementType::Bf16,
            )
            .region(0..4)
            .unwrap(),
            geometry: PackedSwiGluBf16Geometry::new(&x, &y).unwrap(),
            profile: PackedSwiGluBf16Profile::Exp2SigmoidF32SingleRound,
        }
    }
    #[test]
    fn single_round_witness_and_ieee_edges_are_not_staged_silu() {
        assert_eq!(value(1.0078125, 1.0), 0x3f3d);
        assert_ne!(value(1.0078125, 1.0), 0x3f3c);
        assert_eq!(value(2.0, 16.0), half::bf16::from_f32(32.0).to_bits());
        assert_eq!(value(1.0, -0.0), 0x8000);
        assert_eq!(value(1.0, -100.0), 0x8000);
        assert!(half::bf16::from_bits(value(0.0, f32::INFINITY)).is_nan());
    }
    #[test]
    fn malformed_or_foreign_physical_regions_fail_before_writing() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        for change in 0..4 {
            let mut l = launch(&runtime);
            let output = l.output.clone();
            match change {
                0 => l.input.element_type = ElementType::F32,
                1 => l.input.length_bytes -= 2,
                2 => l.output.length_bytes -= 2,
                3 => {
                    l.output = l.input.clone();
                    l.output.length_bytes = 4;
                }
                _ => unreachable!(),
            }
            assert!(ReferenceDeviceCommand::packed_swiglu_bf16_tensor(l).is_err());
            assert_eq!(output.read(), [0xa5; 4]);
        }
        let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
        let mut l = launch(&runtime);
        let output = l.output.clone();
        l.input = launch(&foreign).input;
        let command = ReferenceDeviceCommand::packed_swiglu_bf16_tensor(l).unwrap();
        let mut stream = runtime.create_stream().unwrap();
        assert!(runtime
            .submit_entries(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                false
            )
            .is_err());
        assert_eq!(output.read(), [0xa5; 4]);
        assert_eq!(runtime.snapshot().submissions, 0);
    }
}
