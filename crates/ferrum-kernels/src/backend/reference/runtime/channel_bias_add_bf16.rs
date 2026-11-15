use super::*;
use ferrum_runtime::ChannelBiasAddBf16Geometry;
pub(crate) struct ReferenceChannelBiasAddBf16Launch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) bias: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: ChannelBiasAddBf16Geometry,
}
impl ReferenceChannelBiasAddBf16Launch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        for (r, bytes) in [
            (&self.input, self.geometry.matrix_bytes()),
            (&self.bias, self.geometry.bias_bytes()),
            (&self.output, self.geometry.matrix_bytes()),
        ] {
            r.validate_range()?;
            if r.element_type != ElementType::Bf16
                || r.length_bytes as u64 != bytes
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "channel bias requires exact aligned BF16 extents",
                ));
            }
        }
        if self.output.overlaps(&self.input) || self.output.overlaps(&self.bias) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "channel bias output overlaps input",
            ));
        }
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for r in [&self.input, &self.bias, &self.output] {
            r.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(l: &ReferenceChannelBiasAddBf16Launch) {
    let mut storage = AllocationLocks::new([&l.input, &l.bias, &l.output]);
    for i in 0..l.geometry.elements() as usize {
        let x = bf16::from_bits(storage.read_bits(0, i)).to_f32();
        let bias =
            bf16::from_bits(storage.read_bits(1, i % l.geometry.columns() as usize)).to_f32();
        storage.write_bits(2, i, bf16::from_f32(x + bias).to_bits());
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::*;
    use ferrum_runtime::{ResolvedTensorLayout, ResolvedTensorSpec};
    fn launch(rt: &ReferenceDeviceRuntime) -> ReferenceChannelBiasAddBf16Launch {
        let t = |d| {
            ResolvedTensorSpec::new(d, ElementType::Bf16, ResolvedTensorLayout::Contiguous).unwrap()
        };
        let x = t(vec![1, 3]);
        let bias = t(vec![3]);
        let geometry = ChannelBiasAddBf16Geometry::new(&x, &bias, &x).unwrap();
        ReferenceChannelBiasAddBf16Launch {
            input: buffer(
                rt,
                "resource.channel_bias.x",
                &[0; 6],
                16,
                ElementType::Bf16,
            )
            .region(0..6)
            .unwrap(),
            bias: buffer(
                rt,
                "resource.channel_bias.b",
                &[0; 6],
                16,
                ElementType::Bf16,
            )
            .region(0..6)
            .unwrap(),
            output: buffer(
                rt,
                "resource.channel_bias.y",
                &[0xa5; 6],
                16,
                ElementType::Bf16,
            )
            .region(0..6)
            .unwrap(),
            geometry,
        }
    }
    #[test]
    fn channel_bias_bf16_rejects_same_extent_alias_wrong_storage_and_foreign_owner_before_writes() {
        let rt = ReferenceDeviceRuntime::new(config()).unwrap();
        let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
        for case in 0..4 {
            let mut l = launch(&rt);
            let y = l.output.clone();
            match case {
                0 => l.input = l.output.clone(),
                1 => l.bias = l.output.clone(),
                2 => l.bias.element_type = ElementType::F16,
                3 => l.bias.length_bytes -= 2,
                _ => unreachable!(),
            }
            assert!(ReferenceDeviceCommand::channel_bias_add_bf16_tensor(l).is_err());
            assert_eq!(y.read(), vec![0xa5; 6]);
        }
        let mut l = launch(&rt);
        let y = l.output.clone();
        l.bias = launch(&foreign).bias;
        let command = ReferenceDeviceCommand::channel_bias_add_bf16_tensor(l).unwrap();
        let mut stream = rt.create_stream().unwrap();
        assert!(rt
            .submit_entries(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                false
            )
            .is_err());
        assert_eq!(rt.snapshot().submissions, 0);
        assert_eq!(y.read(), vec![0xa5; 6]);
    }
}
