//! Two-pass centered F32 LayerNorm; fixed 256-lane reduction, no heap scratch.
use super::*;
use ferrum_runtime::LayerNormF32Geometry;
pub(crate) struct ReferenceLayerNormF32Launch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) gamma: ReferenceBufferRegion,
    pub(crate) beta: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: LayerNormF32Geometry,
}
impl ReferenceLayerNormF32Launch {
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        DeviceCommandLogicalWork::tensor(
            self.geometry.elements(),
            self.geometry.logical_arithmetic_operations(),
        )
        .map_err(|e| invalid(e.to_string()))
    }
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        for (r, elements) in [
            (&self.input, self.geometry.elements()),
            (&self.gamma, self.geometry.columns()),
            (&self.beta, self.geometry.columns()),
            (&self.output, self.geometry.elements()),
        ] {
            r.validate_range()?;
            if r.element_type != ElementType::F32
                || elements.checked_mul(4) != Some(r.length_bytes as u64)
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % 4 != 0
            {
                return Err(invalid("requires exact scalar-aligned F32 ranges"));
            }
        }
        if [&self.input, &self.gamma, &self.beta]
            .iter()
            .any(|r| self.output.overlaps(r))
        {
            return Err(invalid("output aliases a read operand"));
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for r in [&self.input, &self.gamma, &self.beta, &self.output] {
            r.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
fn tree(sums: &mut [f32; 256]) -> f32 {
    let mut stride = 128;
    while stride != 0 {
        for lane in 0..stride {
            sums[lane] += sums[lane + stride];
        }
        stride /= 2;
    }
    sums[0]
}
pub(super) fn execute(launch: &ReferenceLayerNormF32Launch) {
    let mut s = AllocationLocks::new([&launch.input, &launch.gamma, &launch.beta, &launch.output]);
    let width = launch.geometry.columns() as usize;
    for row in 0..launch.geometry.rows() as usize {
        let start = row * width;
        let mut sums = [0.0f32; 256];
        for (lane, sum) in sums.iter_mut().enumerate() {
            for c in (lane..width).step_by(256) {
                *sum += s.read_f32(0, start + c);
            }
        }
        let mean = tree(&mut sums) / width as f32;
        sums.fill(0.0);
        for (lane, sum) in sums.iter_mut().enumerate() {
            for c in (lane..width).step_by(256) {
                let centered = s.read_f32(0, start + c) - mean;
                let square = centered * centered;
                *sum += square;
            }
        }
        let variance = tree(&mut sums) / width as f32;
        let inverse = 1.0f32 / (variance + launch.geometry.effective_epsilon()).sqrt();
        for c in 0..width {
            let centered = s.read_f32(0, start + c) - mean;
            let normalized = centered * inverse;
            let scaled = normalized * s.read_f32(1, c);
            let y = scaled + s.read_f32(2, c);
            s.write_f32(3, start + c, y);
        }
    }
}
fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("Reference F32 LayerNorm: {}", reason.into()))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::*;
    use ferrum_runtime::{CanonicalRational, ResolvedTensorLayout, ResolvedTensorSpec};
    #[test]
    fn layer_norm_preflight_rejects_alias_wrong_physical_dtype_and_extent_before_writes() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let x = buffer(
            &runtime,
            "resource.layer_norm.x",
            &[0; 16],
            16,
            ElementType::F32,
        );
        let p = buffer(
            &runtime,
            "resource.layer_norm.p",
            &[0; 16],
            16,
            ElementType::F32,
        );
        let y = buffer(
            &runtime,
            "resource.layer_norm.y",
            &[0xa5; 16],
            16,
            ElementType::F32,
        );
        let spec = |d| {
            ResolvedTensorSpec::new(d, ElementType::F32, ResolvedTensorLayout::Contiguous).unwrap()
        };
        let g = LayerNormF32Geometry::new(
            &spec(vec![1, 4]),
            &spec(vec![4]),
            &spec(vec![4]),
            &spec(vec![1, 4]),
            CanonicalRational::new(1, 4).unwrap(),
        )
        .unwrap();
        let mut launch = ReferenceLayerNormF32Launch {
            input: x.region(0..16).unwrap(),
            gamma: p.region(0..16).unwrap(),
            beta: p.region(0..16).unwrap(),
            output: y.region(0..16).unwrap(),
            geometry: g,
        };
        launch.validate().unwrap();
        let original = launch.output.read();
        launch.output = launch.input.clone();
        assert!(launch.validate().is_err());
        launch.output = y.region(0..16).unwrap();
        launch.gamma.element_type = ElementType::Bf16;
        assert!(launch.validate().is_err());
        launch.gamma = p.region(0..12).unwrap();
        assert!(launch.validate().is_err());
        assert_eq!(launch.output.read(), original);
    }
}
