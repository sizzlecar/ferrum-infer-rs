use super::*;
use ferrum_runtime::HeadRmsPartialRopeSingleRoundBf16Geometry;

pub(crate) struct ReferenceHeadRmsPartialRopeSingleRoundLaunch {
    pub input: ReferenceBufferRegion,
    pub gamma: ReferenceBufferRegion,
    pub cosine: ReferenceBufferRegion,
    pub sine: ReferenceBufferRegion,
    pub output: ReferenceBufferRegion,
    pub geometry: HeadRmsPartialRopeSingleRoundBf16Geometry,
}
impl ReferenceHeadRmsPartialRopeSingleRoundLaunch {
    pub fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let g = self.geometry;
        for (region, bytes) in [
            (&self.input, g.rows() * g.columns() * 2),
            (&self.gamma, g.head_dim() * 2),
            (&self.cosine, g.rows() * g.rotary_dim() * 2),
            (&self.sine, g.rows() * g.rotary_dim() * 2),
            (&self.output, g.rows() * g.columns() * 2),
        ] {
            region.validate_range()?;
            if region.element_type != ElementType::Bf16
                || region.length_bytes as u64 != bytes
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(invalid("requires exact aligned BF16 regions"));
            }
        }
        if [&self.input, &self.gamma, &self.cosine, &self.sine]
            .into_iter()
            .any(|r| self.output.overlaps(r))
        {
            return Err(invalid("output overlaps an input"));
        }
        Ok(())
    }
    pub fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
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
    pub fn logical_work(&self) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        DeviceCommandLogicalWork::tensor(
            self.geometry.rows() * self.geometry.columns(),
            self.geometry
                .logical_arithmetic_operations()
                .map_err(invalid)?,
        )
        .map_err(invalid)
    }
}
fn read(memory: &AllocationLocks<'_, 5>, slot: usize, index: usize) -> f32 {
    bf16::from_bits(memory.read_bits(slot, index)).to_f32()
}
fn butterfly(values: &mut [f32], offsets: &[usize]) {
    for &offset in offsets {
        for lane in 0..values.len() {
            let peer = lane ^ offset;
            if peer > lane {
                let sum = values[lane] + values[peer];
                values[lane] = sum;
                values[peer] = sum;
            }
        }
    }
}
pub(super) fn execute(launch: &ReferenceHeadRmsPartialRopeSingleRoundLaunch) {
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
            let x: [f32; 128] = std::array::from_fn(|d| {
                if d < head {
                    read(&memory, 0, start + d)
                } else {
                    0.
                }
            });
            let squares: [f32; 128] = x.map(|v| v * v);
            let mut lanes: [f32; 128] =
                std::array::from_fn(|lane| x[lane].mul_add(x[lane], squares[lane ^ 16]));
            let mut warps = [0f32; 4];
            for (warp, values) in lanes.chunks_exact_mut(32).enumerate() {
                butterfly(values, &[8, 4, 2, 1]);
                warps[warp] = values[0];
            }
            butterfly(&mut warps, &[2, 1]);
            let mean = warps[0] / head as f32;
            let inverse = 1.0f32 / libm::sqrtf(g.effective_epsilon() + mean);
            for d in 0..head {
                let normalized = (inverse * x[d]) * read(&memory, 1, d);
                let result = if d < rot {
                    let first = d < rot / 2;
                    let partner = if first { d + rot / 2 } else { d - rot / 2 };
                    let p = inverse * x[partner];
                    let gamma = read(&memory, 1, partner);
                    let rotated = if first {
                        (-p).mul_add(gamma, 0.0)
                    } else {
                        p * gamma
                    };
                    normalized.mul_add(
                        read(&memory, 2, row * rot + d),
                        rotated * read(&memory, 3, row * rot + d),
                    )
                } else {
                    normalized
                };
                memory.write_bits(4, start + d, bf16::from_f32(result).to_bits());
            }
        }
    }
}
fn invalid(reason: impl std::fmt::Display) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("single-round RMS/RoPE: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::*;
    use ferrum_runtime::*;
    fn launch(runtime: &ReferenceDeviceRuntime) -> ReferenceHeadRmsPartialRopeSingleRoundLaunch {
        let tensor = |shape| {
            ResolvedTensorSpec::new(shape, ElementType::Bf16, ResolvedTensorLayout::Contiguous)
                .unwrap()
        };
        let x = tensor(vec![2, 4]);
        let g = tensor(vec![2]);
        let t = tensor(vec![2, 2]);
        let mut attrs = HeadRmsPartialRopeSingleRoundBf16Profile::Block128Warp4Fma
            .attributes()
            .unwrap();
        for (name, value) in [
            ("head_dim", SemanticValue::Unsigned(2)),
            ("rotary_dim", SemanticValue::Unsigned(2)),
            (
                "epsilon",
                SemanticValue::Rational(CanonicalRational::new(1, 100000).unwrap()),
            ),
        ] {
            attrs.insert(AttributeId::new(name).unwrap(), value);
        }
        let region = |name, len| {
            buffer(runtime, name, &vec![0xa5; len], 16, ElementType::Bf16)
                .region(0..len as u64)
                .unwrap()
        };
        ReferenceHeadRmsPartialRopeSingleRoundLaunch {
            input: region("resource.qk.x", 16),
            gamma: region("resource.qk.gamma", 4),
            cosine: region("resource.qk.cos", 8),
            sine: region("resource.qk.sin", 8),
            output: region("resource.qk.output", 16),
            geometry: HeadRmsPartialRopeSingleRoundBf16Geometry::new(&x, &g, &t, &t, &x, &attrs)
                .unwrap(),
        }
    }
    #[test]
    fn output_alias_invalid_extent_dtype_and_foreign_owner_fail_before_submission() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        for change in 0..4 {
            let mut l = launch(&runtime);
            let original = l.output.clone();
            match change {
                0 => l.output = l.input.clone(),
                1 => l.gamma.element_type = ElementType::F32,
                2 => l.cosine.length_bytes -= 2,
                3 => l.sine.offset_bytes += 1,
                _ => unreachable!(),
            }
            assert!(ReferenceDeviceCommand::head_rms_partial_rope_single_round_tensor(l).is_err());
            assert_eq!(original.read(), [0xa5; 16]);
        }
        let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
        let mut l = launch(&runtime);
        let original = l.output.clone();
        l.gamma = launch(&foreign).gamma;
        let command = ReferenceDeviceCommand::head_rms_partial_rope_single_round_tensor(l).unwrap();
        let mut stream = runtime.create_stream().unwrap();
        assert!(runtime
            .submit_entries(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                false
            )
            .is_err());
        assert_eq!(original.read(), [0xa5; 16]);
        assert_eq!(runtime.snapshot().submissions, 0);
    }
}
