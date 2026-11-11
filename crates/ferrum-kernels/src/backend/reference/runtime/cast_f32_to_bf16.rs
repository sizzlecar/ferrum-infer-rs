//! Allocation-free, bit-defined IEEE conversion on the admitted CPU storage.
use super::*;
use ferrum_runtime::CastF32ToBf16Geometry;

pub(crate) struct ReferenceCastF32ToBf16Launch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: CastF32ToBf16Geometry,
}
impl ReferenceCastF32ToBf16Launch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        for (region, dtype, size) in [
            (&self.input, ElementType::F32, 4),
            (&self.output, ElementType::Bf16, 2),
        ] {
            region.validate_range()?;
            if region.element_type != dtype
                || self.geometry.elements().checked_mul(size) != Some(region.length_bytes as u64)
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % (size as usize)
                    != 0
            {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "cast requires exact aligned F32 input and BF16 output extents",
                ));
            }
        }
        if self.output.overlaps(&self.input) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "cast output aliases its input",
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
fn round_bits(bits: u32) -> u16 {
    if bits & 0x7fffffff > 0x7f800000 {
        return (bits >> 16) as u16 | 0x0040;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
pub(super) fn execute(launch: &ReferenceCastF32ToBf16Launch) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.output]);
    for i in 0..launch.geometry.elements() as usize {
        let bits = round_bits(storage.read_f32(0, i).to_bits());
        storage.write_bits(1, i, bits);
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::*;
    use ferrum_runtime::{ResolvedTensorLayout, ResolvedTensorSpec};
    fn geometry() -> CastF32ToBf16Geometry {
        CastF32ToBf16Geometry::new(
            &ResolvedTensorSpec::new(
                vec![1, 2],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            &ResolvedTensorSpec::new(
                vec![1, 2],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn ieee_rounding_preserves_ties_signed_zero_subnormal_and_overflow() {
        for (input, expected) in [
            (0x00000000, 0x0000),
            (0x80000000, 0x8000),
            (0x3f808000, 0x3f80),
            (0x3f818000, 0x3f82),
            (0xbf808000, 0xbf80),
            (0xbf818000, 0xbf82),
            (0x00008000, 0x0000),
            (0x00008001, 0x0001),
            (0x00010000, 0x0001),
            (0x00018000, 0x0002),
            (0x007fffff, 0x0080),
            (0x7f7f7fff, 0x7f7f),
            (0x7f7f8000, 0x7f80),
            (0xff7f8000, 0xff80),
            (0x7f800000, 0x7f80),
            (0xff800000, 0xff80),
        ] {
            assert_eq!(round_bits(input), expected);
        }
        for input in [0x7f800001, 0x7fc00000, 0xff800001] {
            let output = round_bits(input);
            assert_eq!(output & 0x7f80, 0x7f80);
            assert_ne!(output & 0x007f, 0);
            assert_ne!(output & 0x0040, 0);
            assert_eq!(output & 0x8000, (input >> 16) as u16 & 0x8000);
        }
    }
    #[test]
    fn cast_preflight_rejects_alias_and_inexact_extents_before_writes() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let input = buffer(
            &runtime,
            "resource.cast.input",
            &[0; 8],
            16,
            ElementType::F32,
        );
        let output = buffer(
            &runtime,
            "resource.cast.output",
            &[0xa5; 4],
            16,
            ElementType::Bf16,
        );
        let wrong_type = buffer(
            &runtime,
            "resource.cast.wrong_output",
            &[0xa5; 4],
            16,
            ElementType::F32,
        );
        let mut alias = input.region(0..4).unwrap();
        alias.element_type = ElementType::Bf16;
        for bad in [
            alias,
            output.region(0..2).unwrap(),
            wrong_type.region(0..4).unwrap(),
        ] {
            let launch = ReferenceCastF32ToBf16Launch {
                input: input.region(0..8).unwrap(),
                output: bad,
                geometry: geometry(),
            };
            assert!(ReferenceDeviceCommand::cast_f32_to_bf16_tensor(launch).is_err());
            assert_eq!(output.region(0..4).unwrap().read(), [0xa5; 4]);
        }
        assert_eq!(runtime.snapshot().submissions, 0);
    }
}
