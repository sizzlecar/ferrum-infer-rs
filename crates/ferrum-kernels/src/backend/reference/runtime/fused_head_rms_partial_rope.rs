//! Warp-32 square reduction, precise sqrt/reciprocal, one norm BF16 boundary,
//! then separate F32 rotation products/addition and one final BF16 boundary.
use super::{
    bf16, AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};
use ferrum_runtime::FusedHeadRmsPartialRopeBf16Geometry;

pub(crate) struct ReferenceFusedHeadRmsPartialRopeLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) gamma: ReferenceBufferRegion,
    pub(crate) cosine: ReferenceBufferRegion,
    pub(crate) sine: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: FusedHeadRmsPartialRopeBf16Geometry,
}
impl ReferenceFusedHeadRmsPartialRopeLaunch {
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        DeviceCommandLogicalWork::tensor(
            self.geometry.rows() * self.geometry.columns(),
            self.geometry
                .logical_arithmetic_operations()
                .map_err(invalid)?,
        )
        .map_err(invalid)
    }
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let g = self.geometry;
        let extents = [
            g.rows() * g.columns() * 2,
            g.head_dim() * 2,
            g.rows() * g.rotary_dim() * 2,
            g.rows() * g.rotary_dim() * 2,
            g.rows() * g.columns() * 2,
        ];
        for (region, bytes) in [
            &self.input,
            &self.gamma,
            &self.cosine,
            &self.sine,
            &self.output,
        ]
        .into_iter()
        .zip(extents)
        {
            region.validate_range()?;
            if region.element_type != ElementType::Bf16
                || region.length_bytes as u64 != bytes
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(invalid(
                    "fused RMS/RoPE requires exact aligned BF16 operands",
                ));
            }
        }
        if [&self.input, &self.gamma, &self.cosine, &self.sine]
            .into_iter()
            .any(|r| self.output.overlaps(r))
        {
            return Err(invalid("fused RMS/RoPE output aliases a read input"));
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in [
            &self.input,
            &self.gamma,
            &self.cosine,
            &self.sine,
            &self.output,
        ] {
            region.validate_runtime(runtime)?;
        }
        self.validate()
    }
}

pub(super) fn execute(launch: &ReferenceFusedHeadRmsPartialRopeLaunch) {
    let mut memory = AllocationLocks::new([
        &launch.input,
        &launch.gamma,
        &launch.cosine,
        &launch.sine,
        &launch.output,
    ]);
    let g = launch.geometry;
    let (rows, columns, head, rot) = (
        g.rows() as usize,
        g.columns() as usize,
        g.head_dim() as usize,
        g.rotary_dim() as usize,
    );
    for row in 0..rows {
        for h in 0..columns / head {
            let start = row * columns + h * head;
            let mut lanes = [0.0_f32; 32];
            for (lane, sum) in lanes.iter_mut().enumerate() {
                for d in (lane..head).step_by(32) {
                    let x = bf16::from_bits(memory.read_bits(0, start + d)).to_f32();
                    *sum = x.mul_add(x, *sum);
                }
            }
            for offset in [16, 8, 4, 2, 1] {
                for lane in 0..32 - offset {
                    lanes[lane] = lanes[lane] + lanes[lane + offset];
                }
            }
            let mean = lanes[0] / head as f32;
            let inverse = 1.0_f32 / (mean + g.effective_epsilon()).sqrt();
            // No scratch matrix: each normalized pair is immediately rotated.
            for d in 0..rot / 2 {
                let normalized = |d, memory: &AllocationLocks<'_, 5>| {
                    let x = bf16::from_bits(memory.read_bits(0, start + d)).to_f32();
                    let gamma = bf16::from_bits(memory.read_bits(1, d)).to_f32();
                    bf16::from_f32((x * inverse) * gamma).to_f32()
                };
                let a = normalized(d, &memory);
                let b = normalized(d + rot / 2, &memory);
                for (channel, left, right) in [(d, a, -b), (d + rot / 2, b, a)] {
                    let c = bf16::from_bits(memory.read_bits(2, row * rot + channel)).to_f32();
                    let s = bf16::from_bits(memory.read_bits(3, row * rot + channel)).to_f32();
                    let p = left * c;
                    let q = right * s;
                    memory.write_bits(4, start + channel, bf16::from_f32(p + q).to_bits());
                }
            }
            for d in rot..head {
                let x = bf16::from_bits(memory.read_bits(0, start + d)).to_f32();
                let gamma = bf16::from_bits(memory.read_bits(1, d)).to_f32();
                memory.write_bits(
                    4,
                    start + d,
                    bf16::from_f32((x * inverse) * gamma).to_bits(),
                );
            }
        }
    }
}
fn invalid(reason: impl std::fmt::Display) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("fused head RMS/partial RoPE: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::super::{
        DeviceCommandPhase, DeviceRuntime, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    };
    use super::*;
    use ferrum_runtime::{CanonicalRational, ResolvedTensorLayout, ResolvedTensorSpec};
    fn fixture(runtime: &ReferenceDeviceRuntime) -> ReferenceFusedHeadRmsPartialRopeLaunch {
        let spec = |dimensions: Vec<u64>| {
            ResolvedTensorSpec::new(
                dimensions,
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap()
        };
        let x = spec(vec![1, 4]);
        let gamma = spec(vec![4]);
        let c = spec(vec![1, 2]);
        let region = |name, values: &[f32]| {
            let bytes: Vec<_> = values
                .iter()
                .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
                .collect();
            buffer(runtime, name, &bytes, 16, ElementType::Bf16)
                .region(0..bytes.len() as u64)
                .unwrap()
        };
        ReferenceFusedHeadRmsPartialRopeLaunch {
            input: region("resource.fused.x", &[1., 2., 3., 4.]),
            gamma: region("resource.fused.g", &[1.; 4]),
            cosine: region("resource.fused.c", &[1.; 2]),
            sine: region("resource.fused.s", &[0.; 2]),
            output: buffer(
                runtime,
                "resource.fused.y",
                &[0xa5; 8],
                16,
                ElementType::Bf16,
            )
            .region(0..8)
            .unwrap(),
            geometry: FusedHeadRmsPartialRopeBf16Geometry::new(
                &x,
                &gamma,
                &c,
                &c,
                &x,
                4,
                2,
                CanonicalRational::new(1, 100000).unwrap(),
            )
            .unwrap(),
        }
    }
    #[test]
    fn dtype_extent_alignment_and_alias_fail_without_touching_output() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        for mutation in 0..5 {
            let mut launch = fixture(&runtime);
            let original = launch.output.clone();
            match mutation {
                0 => launch.gamma.element_type = ElementType::F32,
                1 => launch.cosine.length_bytes -= 2,
                2 => launch.sine.offset_bytes += 1,
                3 => launch.output = launch.input.clone(),
                4 => launch.output = launch.gamma.clone(),
                _ => unreachable!(),
            }
            assert!(ReferenceDeviceCommand::fused_head_rms_partial_rope_tensor(launch).is_err());
            assert_eq!(original.read(), [0xa5; 8]);
        }
    }
    #[test]
    fn late_foreign_gamma_rejects_whole_submission_before_first_dispatch() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
        let first = fixture(&runtime);
        let output = first.output.clone();
        let mut second = fixture(&runtime);
        second.gamma = fixture(&foreign).gamma;
        let mut stream = runtime.create_stream().unwrap();
        let before = runtime.snapshot();
        assert!(runtime
            .submit_entries(
                &mut stream,
                vec![
                    (
                        DeviceCommandPhase::Compute,
                        Some(0),
                        ReferenceDeviceCommand::fused_head_rms_partial_rope_tensor(first).unwrap()
                    ),
                    (
                        DeviceCommandPhase::Compute,
                        Some(1),
                        ReferenceDeviceCommand::fused_head_rms_partial_rope_tensor(second).unwrap()
                    )
                ],
                false
            )
            .is_err());
        assert_eq!(output.read(), [0xa5; 8]);
        assert_eq!(runtime.snapshot().submissions, before.submissions);
        assert_eq!(
            runtime.snapshot().fused_head_rms_partial_rope_launches,
            before.fused_head_rms_partial_rope_launches
        );
    }
}
