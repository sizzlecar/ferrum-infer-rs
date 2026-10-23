//! Ascending conceptual group order; FP32 additions/division and one BF16 RNE.
use super::{
    bf16, AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceCommandKind, ReferenceDeviceCommand, ReferenceDeviceRuntimeError,
};
use ferrum_runtime::SpatialGroupMeanBf16Geometry;

pub(crate) struct ReferenceSpatialGroupMeanLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) shape: SpatialGroupMeanBf16Geometry,
}
impl ReferenceSpatialGroupMeanLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let g = &self.shape;
        let input = g
            .batch()
            .checked_mul(g.input_channels())
            .and_then(|n| n.checked_mul(g.input_height()))
            .and_then(|n| n.checked_mul(g.input_width()))
            .ok_or_else(|| invalid("input extent overflow"))?;
        for (r, n) in [(&self.input, input), (&self.output, g.element_count())] {
            r.validate_range()?;
            let bytes = n
                .checked_mul(2)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n <= isize::MAX as usize)
                .ok_or_else(|| invalid("extent exceeds address space"))?;
            if r.element_type != ElementType::Bf16
                || r.length_bytes != bytes
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(invalid("BF16 region dtype/extent/alignment differs"));
            }
        }
        if self.output.overlaps(&self.input) {
            return Err(invalid("output overlaps input"));
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, id: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        self.input.validate_runtime(id)?;
        self.output.validate_runtime(id)?;
        self.validate()
    }
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        DeviceCommandLogicalWork::tensor(
            self.shape.element_count(),
            self.shape.logical_arithmetic_operations(),
        )
        .map_err(|e| invalid(e.to_string()))
    }
}
impl ReferenceDeviceCommand {
    pub(crate) fn spatial_group_mean_tensor(
        launch: ReferenceSpatialGroupMeanLaunch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        Ok(Self {
            operation: "vnext_spatial_group_mean_bf16",
            logical_work: Some(launch.logical_work()?),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::SpatialGroupMean { launch },
        })
    }
}
pub(super) fn execute(launch: &ReferenceSpatialGroupMeanLaunch) {
    let g = launch.shape;
    let mut storage = AllocationLocks::new([&launch.input, &launch.output]);
    let (ci, h, w, co, oh, ow, sh, sw, ft, group) = (
        g.input_channels() as usize,
        g.input_height() as usize,
        g.input_width() as usize,
        g.output_channels() as usize,
        g.output_height() as usize,
        g.output_width() as usize,
        g.scale_h() as usize,
        g.scale_w() as usize,
        g.frame_factor() as usize,
        g.group_size() as usize,
    );
    let spatial = sh * sw;
    for n in 0..g.batch() as usize {
        for o in 0..co {
            for y in 0..oh {
                for x in 0..ow {
                    let read = |j: usize| {
                        let index = o * group + j;
                        let c = index / (ft * spatial);
                        let phase = index % (ft * spatial);
                        if phase / spatial != ft - 1 {
                            return 0.0;
                        }
                        let dy = phase % spatial / sw;
                        let dx = phase % sw;
                        bf16::from_bits(
                            storage
                                .read_bits(0, ((n * ci + c) * h + y * sh + dy) * w + x * sw + dx),
                        )
                        .to_f32()
                    };
                    let mut sum = read(0);
                    for j in 1..group {
                        sum += read(j);
                    }
                    let value = sum / group as f32;
                    storage.write_bits(
                        1,
                        ((n * co + o) * oh + y) * ow + x,
                        bf16::from_f32(value).to_bits(),
                    );
                }
            }
        }
    }
}
fn invalid(message: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!(
        "reference spatial group mean: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::super::ReferenceDeviceRuntime;
    use super::*;
    use ferrum_runtime::*;
    use std::collections::BTreeMap;
    fn tensor(shape: &[u64]) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(
            shape.to_vec(),
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap()
    }
    fn region(r: &ReferenceDeviceRuntime, name: &str, bits: &[u16]) -> ReferenceBufferRegion {
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        buffer(r, name, &bytes, 16, ElementType::Bf16)
            .region(0..bytes.len() as u64)
            .unwrap()
    }
    #[test]
    fn spatial_group_mean_preserves_zero_frame_channel_order_and_real_batch_values() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let attrs = ["scale_h", "scale_w", "frame_factor"]
            .into_iter()
            .map(|s| (AttributeId::new(s).unwrap(), SemanticValue::Unsigned(2)))
            .collect::<BTreeMap<_, _>>();
        let geometry = SpatialGroupMeanBf16Geometry::new(
            &tensor(&[2, 1, 2, 2]),
            &tensor(&[2, 2, 1, 1]),
            &attrs,
        )
        .unwrap();
        let values = [1., 3., 5., 7., -2., -4., -6., -8.].map(|v| bf16::from_f32(v).to_bits());
        let output = region(&runtime, "resource.mean.output", &[0xa5a5; 4]);
        let launch = ReferenceSpatialGroupMeanLaunch {
            input: region(&runtime, "resource.mean.input", &values),
            output: output.clone(),
            shape: geometry,
        };
        let command = ReferenceDeviceCommand::spatial_group_mean_tensor(launch).unwrap();
        let mut stream = runtime.create_stream().unwrap();
        runtime
            .submit_entries(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                false,
            )
            .unwrap();
        let actual: Vec<_> = output
            .read()
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(actual, [0, 0x4080, 0, 0xc0a0]);
    }
    #[test]
    fn spatial_group_mean_rejects_alias_before_write() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let attrs = ["scale_h", "scale_w", "frame_factor"]
            .into_iter()
            .map(|s| (AttributeId::new(s).unwrap(), SemanticValue::Unsigned(1)))
            .collect::<BTreeMap<_, _>>();
        let shape = SpatialGroupMeanBf16Geometry::new(
            &tensor(&[1, 1, 1, 1]),
            &tensor(&[1, 1, 1, 1]),
            &attrs,
        )
        .unwrap();
        let input = region(&runtime, "resource.mean.alias", &[0x8000]);
        let launch = ReferenceSpatialGroupMeanLaunch {
            input: input.clone(),
            output: input.clone(),
            shape,
        };
        assert!(ReferenceDeviceCommand::spatial_group_mean_tensor(launch).is_err());
        assert_eq!(input.read(), [0, 0x80]);
    }
    #[test]
    fn spatial_group_mean_zero_frame_does_not_read_nan_and_singleton_keeps_negative_zero() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        for (frames, input, expected) in [(2, 0x7fc1, vec![0, 0x7fc1]), (1, 0x8000, vec![0x8000])] {
            let attrs = BTreeMap::from([
                (
                    AttributeId::new("scale_h").unwrap(),
                    SemanticValue::Unsigned(1),
                ),
                (
                    AttributeId::new("scale_w").unwrap(),
                    SemanticValue::Unsigned(1),
                ),
                (
                    AttributeId::new("frame_factor").unwrap(),
                    SemanticValue::Unsigned(frames),
                ),
            ]);
            let shape = SpatialGroupMeanBf16Geometry::new(
                &tensor(&[1, 1, 1, 1]),
                &tensor(&[1, frames, 1, 1]),
                &attrs,
            )
            .unwrap();
            let output = region(
                &runtime,
                "resource.mean.special.output",
                &vec![0xa5a5; frames as usize],
            );
            let command = ReferenceDeviceCommand::spatial_group_mean_tensor(
                ReferenceSpatialGroupMeanLaunch {
                    input: region(&runtime, "resource.mean.special.input", &[input]),
                    output: output.clone(),
                    shape,
                },
            )
            .unwrap();
            let mut stream = runtime.create_stream().unwrap();
            runtime
                .submit_entries(
                    &mut stream,
                    vec![(DeviceCommandPhase::Compute, Some(0), command)],
                    false,
                )
                .unwrap();
            let bits = output
                .read()
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect::<Vec<_>>();
            if frames == 2 {
                assert_eq!(bits[0], 0);
                assert!(bf16::from_bits(bits[1]).is_nan());
            } else {
                assert_eq!(bits, expected);
            }
        }
    }
}
