use super::*;
use ferrum_runtime::{HeadAxisPackGeometry, HeadAxisPackKind};
pub(crate) struct ReferenceHeadAxisPackLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: HeadAxisPackGeometry,
}
impl ReferenceHeadAxisPackLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        for (r, dtype, bytes, size) in [
            (
                &self.input,
                self.geometry.kind().input_type(),
                self.geometry.input_bytes(),
                self.geometry.kind().input_bytes(),
            ),
            (
                &self.output,
                ElementType::Bf16,
                self.geometry.output_bytes(),
                2,
            ),
        ] {
            r.validate_range()?;
            if r.element_type != dtype
                || r.length_bytes as u64 != bytes
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % size as usize != 0
            {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "head-axis pack requires exact aligned typed extents",
                ));
            }
        }
        if self.input.overlaps(&self.output) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "head-axis pack output aliases input",
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
pub(super) fn execute(l: &ReferenceHeadAxisPackLaunch) {
    let g = l.geometry;
    let mut storage = AllocationLocks::new([&l.input, &l.output]);
    for b in 0..g.batches() {
        for h in 0..g.heads() {
            for r in 0..g.rows() {
                for d in 0..g.head_dim() {
                    let src = (((b * g.heads() + h) * g.rows() + r) * g.head_dim() + d) as usize;
                    let dst = (((b * g.rows() + r) * g.heads() + h) * g.head_dim() + d) as usize;
                    let bits = match g.kind() {
                        HeadAxisPackKind::Bf16 => storage.read_bits(0, src),
                        HeadAxisPackKind::F32ToBf16 => {
                            super::float_cast::round_bits(storage.read_f32(0, src).to_bits())
                        }
                    };
                    storage.write_bits(1, dst, bits);
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::*;
    use ferrum_runtime::{ResolvedTensorLayout, ResolvedTensorSpec};
    fn launch(rt: &ReferenceDeviceRuntime, kind: HeadAxisPackKind) -> ReferenceHeadAxisPackLaunch {
        let input = ResolvedTensorSpec::new(
            vec![2, 3, 5, 7],
            kind.input_type(),
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        let output = ResolvedTensorSpec::new(
            vec![2, 5, 21],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        let geometry = HeadAxisPackGeometry::new(kind, &input, &output).unwrap();
        ReferenceHeadAxisPackLaunch {
            input: buffer(
                rt,
                "resource.head_pack.input",
                &vec![0; geometry.input_bytes() as usize],
                16,
                kind.input_type(),
            )
            .region(0..geometry.input_bytes())
            .unwrap(),
            output: buffer(
                rt,
                "resource.head_pack.output",
                &vec![0xa5; geometry.output_bytes() as usize],
                16,
                ElementType::Bf16,
            )
            .region(0..geometry.output_bytes())
            .unwrap(),
            geometry,
        }
    }
    #[test]
    fn head_axis_pack_rejects_exact_storage_alias_alignment_and_foreign_runtime_before_writes() {
        let rt = ReferenceDeviceRuntime::new(config()).unwrap();
        let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
        for kind in HeadAxisPackKind::ALL {
            for case in 0..4 {
                let mut l = launch(&rt, kind);
                let output = l.output.clone();
                match case {
                    0 => {
                        l.output = l.input.clone();
                        l.output.element_type = ElementType::Bf16;
                        l.output.length_bytes = l.geometry.output_bytes() as usize;
                    }
                    1 => l.input.element_type = ElementType::U8,
                    2 => l.output.length_bytes -= 2,
                    3 => l.input.offset_bytes += 1,
                    _ => unreachable!(),
                };
                assert!(ReferenceDeviceCommand::head_axis_pack_tensor(l).is_err());
                assert!(output.read().iter().all(|&b| b == 0xa5));
            }
            let mut l = launch(&rt, kind);
            l.input = launch(&foreign, kind).input;
            let output = l.output.clone();
            let command = ReferenceDeviceCommand::head_axis_pack_tensor(l).unwrap();
            let mut stream = rt.create_stream().unwrap();
            assert!(rt
                .submit_entries(
                    &mut stream,
                    vec![(DeviceCommandPhase::Compute, Some(0), command)],
                    false
                )
                .is_err());
            assert!(output.read().iter().all(|&b| b == 0xa5));
        }
        assert_eq!(rt.snapshot().submissions, 0);
    }
}
