use super::*;
use ferrum_runtime::MaskedTileMeanBf16F32Geometry;

pub(crate) struct ReferenceMaskedTileMeanLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) counts: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: MaskedTileMeanBf16F32Geometry,
}
impl ReferenceMaskedTileMeanLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        for (r, dtype, bytes, size) in [
            (
                &self.input,
                ElementType::Bf16,
                self.geometry.input_bytes(),
                2,
            ),
            (
                &self.counts,
                ElementType::I32,
                self.geometry.count_bytes(),
                4,
            ),
            (
                &self.output,
                ElementType::F32,
                self.geometry.output_bytes(),
                4,
            ),
        ] {
            r.validate_range()?;
            if r.element_type != dtype
                || r.length_bytes as u64 != bytes
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % size != 0
            {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "masked tile mean requires exact aligned physical extents",
                ));
            }
        }
        if self.output.overlaps(&self.input) || self.output.overlaps(&self.counts) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "masked tile mean output aliases input",
            ));
        }
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for r in [&self.input, &self.counts, &self.output] {
            r.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(launch: &ReferenceMaskedTileMeanLaunch) {
    let g = launch.geometry;
    let mut storage = AllocationLocks::new([&launch.input, &launch.counts, &launch.output]);
    for i in 0..g.output_elements() {
        let d = i % g.head_dim();
        let tile = i / g.head_dim() % g.tiles();
        let head = i / (g.head_dim() * g.tiles()) % g.heads();
        let batch = i / (g.head_dim() * g.tiles() * g.heads());
        let count = storage.read_i32(1, tile as usize);
        let value = if count <= 0 || count as u64 > g.tile_rows() {
            f32::from_bits(0x7fc00000)
        } else {
            let mut sum = 0.0f32;
            for row in 0..g.tile_rows() {
                let value = if row < count as u64 {
                    let offset = (batch * g.source_rows() + tile * g.tile_rows() + row)
                        * g.columns()
                        + head * g.head_dim()
                        + d;
                    f32::from_bits(u32::from(storage.read_bits(0, offset as usize)) << 16)
                } else {
                    0.0
                };
                sum += value;
            }
            sum / count as f32
        };
        storage.write_f32(2, i as usize, value);
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::*;
    use ferrum_runtime::{DeviceLogicalWork, ResolvedTensorLayout, ResolvedTensorSpec};
    fn shape(d: Vec<u64>, ty: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(d, ty, ResolvedTensorLayout::Contiguous).unwrap()
    }
    fn geometry(b: u64, t: u64, r: u64, h: u64, d: u64) -> MaskedTileMeanBf16F32Geometry {
        MaskedTileMeanBf16F32Geometry::new(
            &shape(vec![b, t * r, h * d], ElementType::Bf16),
            &shape(vec![t], ElementType::I32),
            &shape(vec![b, h, t, d], ElementType::F32),
            &MaskedTileMeanBf16F32Geometry::attributes(h).unwrap(),
        )
        .unwrap()
    }
    fn launch(
        runtime: &ReferenceDeviceRuntime,
        g: MaskedTileMeanBf16F32Geometry,
        x: &[u16],
        counts: &[i32],
    ) -> ReferenceMaskedTileMeanLaunch {
        let x = x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let c = counts
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        ReferenceMaskedTileMeanLaunch {
            input: buffer(runtime, "resource.masked_mean.x", &x, 16, ElementType::Bf16)
                .region(0..x.len() as u64)
                .unwrap(),
            counts: buffer(
                runtime,
                "resource.masked_mean.counts",
                &c,
                16,
                ElementType::I32,
            )
            .region(0..c.len() as u64)
            .unwrap(),
            output: buffer(
                runtime,
                "resource.masked_mean.y",
                &vec![0xa5; g.output_bytes() as usize],
                16,
                ElementType::F32,
            )
            .region(0..g.output_bytes())
            .unwrap(),
            geometry: g,
        }
    }
    fn submit(runtime: &ReferenceDeviceRuntime, command: ReferenceDeviceCommand) {
        let mut stream = runtime.create_stream().unwrap();
        runtime
            .submit_entries(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                false,
            )
            .unwrap();
    }
    #[test]
    fn masked_tile_mean_handles_tail_axes_invalid_counts_and_poisoned_padding_without_clamping() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let g = geometry(2, 6, 7, 3, 5);
        let counts = [7, 3, 0, -1, 8, i32::MAX];
        let mut x = vec![0x7fc1u16; (g.input_bytes() / 2) as usize];
        let mut expected = vec![0x7fc00000u32; g.output_elements() as usize];
        for b in 0..2 {
            for t in 0..2 {
                for h in 0..3 {
                    for d in 0..5 {
                        let mut sum = 0f64;
                        for r in 0..counts[t] as usize {
                            let value = bf16::from_f32(
                                (b * 3 + t * 2 + h) as f32 + d as f32 / 8. + r as f32 / 16.,
                            );
                            x[((b * 42 + t * 7 + r) * 15 + h * 5 + d) as usize] = value.to_bits();
                            sum += f64::from(value.to_f32());
                        }
                        expected[((b * 3 + h) * 6 + t) * 5 + d] =
                            ((sum / counts[t] as f64) as f32).to_bits();
                    }
                }
            }
        }
        let launch = launch(&runtime, g, &x, &counts);
        let output = launch.output.clone();
        let command = ReferenceDeviceCommand::masked_tile_mean_tensor(launch).unwrap();
        assert_eq!(
            command
                .attribution(0, Some(0), DeviceCommandPhase::Compute)
                .unwrap()
                .logical_work(),
            DeviceLogicalWork::Tensor {
                element_count: g.output_elements(),
                arithmetic_operations: g.arithmetic_operations()
            }
        );
        let before = runtime.snapshot().allocations;
        submit(&runtime, command);
        assert_eq!(runtime.snapshot().allocations, before);
        let actual = output
            .read()
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
    #[test]
    fn masked_tile_mean_preserves_declared_f32_rounding_and_gradual_underflow() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let g = geometry(1, 2, 4, 1, 2);
        let mut x = vec![0x7fc1; 16];
        for (r, v) in [16777216., 1., -16777216., 1.].into_iter().enumerate() {
            x[2 * r] = bf16::from_f32(v).to_bits();
            x[2 * r + 1] = if r == 0 { 1 } else { 0 };
        }
        x[8] = 0x8000;
        x[9] = 0x7f80;
        let launch = launch(&runtime, g, &x, &[4, 1]);
        let output = launch.output.clone();
        submit(
            &runtime,
            ReferenceDeviceCommand::masked_tile_mean_tensor(launch).unwrap(),
        );
        let actual = output
            .read()
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual, [0.25f32.to_bits(), 0x00004000, 0, 0x7f800000]);
    }
    #[test]
    fn masked_tile_mean_preflight_rejects_alias_wrong_storage_and_foreign_owners_before_writes() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
        let g = geometry(1, 2, 4, 1, 2);
        for case in 0..4 {
            let mut l = launch(&runtime, g, &[0; 16], &[4, 1]);
            let y = l.output.clone();
            match case {
                0 => l.counts.element_type = ElementType::F32,
                1 => l.output.length_bytes -= 4,
                2 => {
                    l.output = l.input.clone();
                    l.output.element_type = ElementType::F32;
                    l.output.length_bytes = g.output_bytes() as usize;
                }
                3 => l.input.offset_bytes += 1,
                _ => unreachable!(),
            }
            assert!(ReferenceDeviceCommand::masked_tile_mean_tensor(l).is_err());
            assert!(y.read().iter().all(|&b| b == 0xa5));
        }
        let mut l = launch(&runtime, g, &[0; 16], &[4, 1]);
        let y = l.output.clone();
        l.counts = launch(&foreign, g, &[0; 16], &[4, 1]).counts;
        let command = ReferenceDeviceCommand::masked_tile_mean_tensor(l).unwrap();
        let mut stream = runtime.create_stream().unwrap();
        assert!(runtime
            .submit_entries(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                false
            )
            .is_err());
        assert_eq!(runtime.snapshot().submissions, 0);
        assert!(y.read().iter().all(|&b| b == 0xa5));
    }
}
