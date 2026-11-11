//! Independent F32 operations; no staged half-precision boundary or workspace.
use super::*;
use ferrum_runtime::{
    HalfSplitRopeF32Geometry, HeadRmsNormF32Geometry, UnitHeadRmsPartialRopeF32Geometry,
};
#[derive(Clone, Copy)]
pub(crate) enum ReferenceF32NormRopeGeometry {
    HeadRms(HeadRmsNormF32Geometry),
    HalfSplitRope(HalfSplitRopeF32Geometry),
    UnitHeadRmsPartialRope(UnitHeadRmsPartialRopeF32Geometry),
}
impl ReferenceF32NormRopeGeometry {
    pub(crate) fn elements(self) -> u64 {
        match self {
            Self::HeadRms(g) => g.elements(),
            Self::HalfSplitRope(g) => g.elements(),
            Self::UnitHeadRmsPartialRope(g) => g.elements(),
        }
    }
    pub(crate) fn parameter_elements(self) -> u64 {
        match self {
            Self::HeadRms(g) => g.head_dim(),
            Self::HalfSplitRope(g) => g.rows() * g.head_dim(),
            Self::UnitHeadRmsPartialRope(g) => g.rows() * g.rotary_dim(),
        }
    }
    pub(crate) fn operations(self) -> u64 {
        match self {
            Self::HeadRms(g) => g.logical_arithmetic_operations(),
            Self::HalfSplitRope(g) => g.logical_arithmetic_operations(),
            Self::UnitHeadRmsPartialRope(g) => g.logical_arithmetic_operations(),
        }
    }
}
pub(crate) struct ReferenceF32NormRopeLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) parameter: ReferenceBufferRegion,
    pub(crate) sine: Option<ReferenceBufferRegion>,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: ReferenceF32NormRopeGeometry,
}
impl ReferenceF32NormRopeLaunch {
    pub(crate) fn name(&self) -> &'static str {
        match self.geometry {
            ReferenceF32NormRopeGeometry::HeadRms(_) => "vnext_head_rms_norm_f32",
            ReferenceF32NormRopeGeometry::HalfSplitRope(_) => "vnext_half_split_rope_f32",
            ReferenceF32NormRopeGeometry::UnitHeadRmsPartialRope(_) => {
                "vnext_unit_head_rms_partial_rope_f32"
            }
        }
    }
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        DeviceCommandLogicalWork::tensor(self.geometry.elements(), self.geometry.operations())
            .map_err(|e| invalid(e.to_string()))
    }
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let check =
            |r: &ReferenceBufferRegion, count: u64| -> Result<(), ReferenceDeviceRuntimeError> {
                r.validate_range()?;
                if r.element_type != ElementType::F32
                    || count.checked_mul(4) != Some(r.length_bytes as u64)
                    || (r.offset_bytes - r.allocation.logical_offset_bytes) % 4 != 0
                {
                    return Err(invalid("requires exact aligned F32 operands"));
                }
                Ok(())
            };
        check(&self.input, self.geometry.elements())?;
        check(&self.output, self.geometry.elements())?;
        check(&self.parameter, self.geometry.parameter_elements())?;
        match (self.geometry, &self.sine) {
            (ReferenceF32NormRopeGeometry::HeadRms(_), None) => {}
            (
                ReferenceF32NormRopeGeometry::HalfSplitRope(_)
                | ReferenceF32NormRopeGeometry::UnitHeadRmsPartialRope(_),
                Some(s),
            ) => check(s, self.geometry.parameter_elements())?,
            _ => return Err(invalid("sine presence differs from operation")),
        }
        if self.output.overlaps(&self.input)
            || self.output.overlaps(&self.parameter)
            || self.sine.as_ref().is_some_and(|s| self.output.overlaps(s))
        {
            return Err(invalid("output aliases a read input"));
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for r in [&self.input, &self.parameter, &self.output] {
            r.validate_runtime(runtime)?;
        }
        if let Some(s) = &self.sine {
            s.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(launch: &ReferenceF32NormRopeLaunch) {
    match launch.geometry {
        ReferenceF32NormRopeGeometry::HeadRms(g) => {
            let mut s = AllocationLocks::new([&launch.input, &launch.parameter, &launch.output]);
            let d = g.head_dim() as usize;
            for head in 0..(g.elements() / g.head_dim()) as usize {
                let start = head * d;
                let mut sums = [0.0_f32; 256];
                for (lane, sum) in sums.iter_mut().enumerate() {
                    for column in (lane..d).step_by(256) {
                        let x = s.read_f32(0, start + column);
                        *sum += x * x;
                    }
                }
                let mut stride = 128;
                while stride != 0 {
                    for lane in 0..stride {
                        sums[lane] += sums[lane + stride];
                    }
                    stride /= 2;
                }
                let mean = sums[0] / g.head_dim() as f32;
                let inverse = 1.0_f32 / (mean + g.effective_epsilon()).sqrt();
                for column in 0..d {
                    let value = (s.read_f32(0, start + column) * inverse) * s.read_f32(1, column);
                    s.write_f32(2, start + column, value);
                }
            }
        }
        ReferenceF32NormRopeGeometry::UnitHeadRmsPartialRope(g) => {
            let mut s = AllocationLocks::new([
                &launch.input,
                &launch.parameter,
                launch.sine.as_ref().expect("validated sine"),
                &launch.output,
            ]);
            let d = g.head_dim() as usize;
            let rot = g.rotary_dim() as usize;
            let columns = g.columns() as usize;
            for head in 0..(g.elements() / g.head_dim()) as usize {
                let start = head * d;
                let mut sums = [0.0_f32; 256];
                for (lane, sum) in sums.iter_mut().enumerate() {
                    for c in (lane..d).step_by(256) {
                        let x = s.read_f32(0, start + c);
                        *sum += x * x;
                    }
                }
                let mut stride = 128;
                while stride != 0 {
                    for lane in 0..stride {
                        sums[lane] += sums[lane + stride];
                    }
                    stride /= 2;
                }
                let inverse =
                    1.0_f32 / (sums[0] / g.head_dim() as f32 + g.effective_epsilon()).sqrt();
                for c in 0..d {
                    let value = s.read_f32(0, start + c) * inverse;
                    let result = if c < rot {
                        let partner = if c < rot / 2 {
                            -(s.read_f32(0, start + c + rot / 2) * inverse)
                        } else {
                            s.read_f32(0, start + c - rot / 2) * inverse
                        };
                        let f = start / columns * rot + c;
                        let p = value * s.read_f32(1, f);
                        let q = partner * s.read_f32(2, f);
                        p + q
                    } else {
                        value
                    };
                    s.write_f32(3, start + c, result);
                }
            }
        }
        ReferenceF32NormRopeGeometry::HalfSplitRope(g) => {
            let mut s = AllocationLocks::new([
                &launch.input,
                &launch.parameter,
                launch.sine.as_ref().expect("validated sine"),
                &launch.output,
            ]);
            let d = g.head_dim() as usize;
            let columns = g.columns() as usize;
            for i in 0..g.elements() as usize {
                let local = i % columns % d;
                let frequency = i / columns * d + local;
                let partner = if local < d / 2 {
                    -s.read_f32(0, i + d / 2)
                } else {
                    s.read_f32(0, i - d / 2)
                };
                let p = s.read_f32(0, i) * s.read_f32(1, frequency);
                let q = partner * s.read_f32(2, frequency);
                s.write_f32(3, i, p + q);
            }
        }
    }
}
fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("Reference F32 norm/RoPE: {}", reason.into()))
}
#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::*;
    use ferrum_runtime::{CanonicalRational, ResolvedTensorLayout, ResolvedTensorSpec};
    #[test]
    fn norm_rope_preflight_rejects_alias_dtype_and_extent_without_writes() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let x = buffer(
            &runtime,
            "resource.norm_rope.input",
            &[0; 16],
            16,
            ElementType::F32,
        );
        let p = buffer(
            &runtime,
            "resource.norm_rope.parameter",
            &[0; 16],
            16,
            ElementType::F32,
        );
        let y = buffer(
            &runtime,
            "resource.norm_rope.output",
            &[0xa5; 16],
            16,
            ElementType::F32,
        );
        let bad = buffer(
            &runtime,
            "resource.norm_rope.bad",
            &[0; 16],
            16,
            ElementType::Bf16,
        );
        let matrix = ResolvedTensorSpec::new(
            vec![1, 4],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        let gamma =
            ResolvedTensorSpec::new(vec![4], ElementType::F32, ResolvedTensorLayout::Contiguous)
                .unwrap();
        let norm = HeadRmsNormF32Geometry::new(
            &matrix,
            &gamma,
            &matrix,
            4,
            CanonicalRational::new(1, 4).unwrap(),
        )
        .unwrap();
        for parameter in [
            y.region(0..16).unwrap(),
            p.region(0..8).unwrap(),
            bad.region(0..16).unwrap(),
        ] {
            let launch = ReferenceF32NormRopeLaunch {
                input: x.region(0..16).unwrap(),
                parameter,
                sine: None,
                output: y.region(0..16).unwrap(),
                geometry: ReferenceF32NormRopeGeometry::HeadRms(norm),
            };
            assert!(ReferenceDeviceCommand::norm_rope_f32_tensor(launch).is_err());
            assert_eq!(y.region(0..16).unwrap().read(), [0xa5; 16]);
        }
        let rope = HalfSplitRopeF32Geometry::new(&matrix, &matrix, &matrix, &matrix, 4).unwrap();
        let launch = ReferenceF32NormRopeLaunch {
            input: x.region(0..16).unwrap(),
            parameter: p.region(0..16).unwrap(),
            sine: Some(p.region(0..16).unwrap()),
            output: x.region(0..16).unwrap(),
            geometry: ReferenceF32NormRopeGeometry::HalfSplitRope(rope),
        };
        assert!(ReferenceDeviceCommand::norm_rope_f32_tensor(launch).is_err());
        assert_eq!(runtime.snapshot().submissions, 0);
    }
}
