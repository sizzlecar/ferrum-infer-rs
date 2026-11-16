use super::*;
use ferrum_runtime::RowCenterBf16Geometry;

pub(crate) struct ReferenceRowCenterBf16Launch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) mean: ReferenceBufferRegion,
    pub(crate) geometry: RowCenterBf16Geometry,
}
impl ReferenceRowCenterBf16Launch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        for (r, bytes) in [
            (&self.input, self.geometry.data_bytes()),
            (&self.output, self.geometry.data_bytes()),
            (&self.mean, self.geometry.mean_bytes()),
        ] {
            r.validate_range()?;
            if r.element_type != ElementType::Bf16
                || r.length_bytes as u64 != bytes
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "row center requires exact aligned BF16 extents",
                ));
            }
        }
        if self.input.overlaps(&self.output)
            || self.input.overlaps(&self.mean)
            || self.output.overlaps(&self.mean)
        {
            return Err(ReferenceDeviceRuntimeError::contract(
                "row center outputs must be disjoint from each other and input",
            ));
        }
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for r in [&self.input, &self.output, &self.mean] {
            r.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(l: &ReferenceRowCenterBf16Launch) {
    let g = l.geometry;
    let mut storage = AllocationLocks::new([&l.input, &l.output, &l.mean]);
    for b in 0..g.batches() {
        for c in 0..g.columns() {
            let mut sums = [0.0f32; 8];
            for (lane, sum) in sums.iter_mut().enumerate() {
                for r in (lane as u64..g.rows()).step_by(8) {
                    *sum += bf16::from_bits(
                        storage.read_bits(0, ((b * g.rows() + r) * g.columns() + c) as usize),
                    )
                    .to_f32();
                }
            }
            for stride in [4, 2, 1] {
                for lane in 0..stride {
                    sums[lane] += sums[lane + stride];
                }
            }
            let mean = bf16::from_f32(sums[0] * g.factor());
            storage.write_bits(2, (b * g.columns() + c) as usize, mean.to_bits());
        }
    }
    for i in 0..g.elements() {
        let b = i / (g.rows() * g.columns());
        let c = i % g.columns();
        let value = bf16::from_bits(storage.read_bits(0, i as usize)).to_f32();
        let mean = bf16::from_bits(storage.read_bits(2, (b * g.columns() + c) as usize)).to_f32();
        storage.write_bits(1, i as usize, bf16::from_f32(value - mean).to_bits());
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{buffer, config};
    use super::*;
    use ferrum_runtime::{ResolvedTensorLayout, ResolvedTensorSpec};
    fn launch(
        rt: &ReferenceDeviceRuntime,
        b: u64,
        s: u64,
        c: u64,
        x: &[u16],
    ) -> ReferenceRowCenterBf16Launch {
        let tensor = |d| {
            ResolvedTensorSpec::new(d, ElementType::Bf16, ResolvedTensorLayout::Contiguous).unwrap()
        };
        let input = tensor(vec![b, s, c]);
        let mean = tensor(vec![b, 1, c]);
        let geometry = RowCenterBf16Geometry::new(
            &input,
            &input,
            &mean,
            &RowCenterBf16Geometry::attributes().unwrap(),
        )
        .unwrap();
        assert_eq!(x.len() as u64, geometry.elements());
        let x = x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        ReferenceRowCenterBf16Launch {
            input: buffer(rt, "resource.center.x", &x, 16, ElementType::Bf16)
                .region(0..geometry.data_bytes())
                .unwrap(),
            output: buffer(
                rt,
                "resource.center.y",
                &vec![0xa5; geometry.data_bytes() as usize],
                16,
                ElementType::Bf16,
            )
            .region(0..geometry.data_bytes())
            .unwrap(),
            mean: buffer(
                rt,
                "resource.center.mean",
                &vec![0xa5; geometry.mean_bytes() as usize],
                16,
                ElementType::Bf16,
            )
            .region(0..geometry.mean_bytes())
            .unwrap(),
            geometry,
        }
    }
    fn run(rt: &ReferenceDeviceRuntime, l: ReferenceRowCenterBf16Launch) -> (Vec<u16>, Vec<u16>) {
        let output = l.output.clone();
        let mean = l.mean.clone();
        let command = ReferenceDeviceCommand::row_center_bf16_tensor(l).unwrap();
        let allocations = rt.snapshot().allocations;
        let mut stream = rt.create_stream().unwrap();
        rt.submit_entries(
            &mut stream,
            vec![(DeviceCommandPhase::Compute, Some(0), command)],
            false,
        )
        .unwrap();
        assert_eq!(rt.snapshot().allocations, allocations);
        let read = |r: ReferenceBufferRegion| {
            r.read()
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                .collect()
        };
        (read(output), read(mean))
    }
    #[test]
    fn row_center_includes_padding_and_isolates_batches_and_tail_columns() {
        let rt = ReferenceDeviceRuntime::new(config()).unwrap();
        let (b, s, c) = (2usize, 17usize, 35usize);
        let mut values = vec![0u16; b * s * c];
        let mut sums = vec![0f64; b * c];
        for batch in 0..b {
            for row in 0..s {
                for col in 0..c {
                    // Last three rows model the zero padding that must participate in the divisor.
                    let v = if row < 14 {
                        bf16::from_f32((batch * 3 + col) as f32 / 8. + row as f32 / 16.)
                    } else {
                        bf16::ZERO
                    };
                    values[(batch * s + row) * c + col] = v.to_bits();
                    sums[batch * c + col] += f64::from(v.to_f32());
                }
            }
        }
        let expected_mean = sums
            .iter()
            .map(|v| bf16::from_f32((v / s as f64) as f32).to_bits())
            .collect::<Vec<_>>();
        let expected = values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let m = expected_mean[(i / (s * c)) * c + i % c];
                bf16::from_f32(bf16::from_bits(v).to_f32() - bf16::from_bits(m).to_f32()).to_bits()
            })
            .collect::<Vec<_>>();
        let (actual, mean) = run(&rt, launch(&rt, b as u64, s as u64, c as u64, &values));
        assert_eq!(mean, expected_mean);
        assert_eq!(actual, expected);
    }
    #[test]
    fn row_center_keeps_bf16_mean_boundary_and_gradual_subnormals() {
        let rt = ReferenceDeviceRuntime::new(config()).unwrap();
        // Mean of first column rounds to 1.0 BF16 before subtraction; a fused
        // subtraction of the F32 mean would return -1/256 instead of zero.
        let x = [0x3f80, 1, 0x7f80, 0x3f81, 1, 0x7f80];
        let (actual, mean) = run(&rt, launch(&rt, 1, 2, 3, &x));
        assert_eq!(&mean[..2], &[0x3f80, 1]);
        assert_eq!(mean[2], 0x7f80);
        assert_eq!(&actual[..2], &[0, 0]);
        assert_eq!(&actual[3..5], &[0x3c00, 0]);
        assert!(bf16::from_bits(actual[2]).is_nan() && bf16::from_bits(actual[5]).is_nan());
    }
    #[test]
    fn row_center_preflight_checks_all_outputs_before_writes() {
        let rt = ReferenceDeviceRuntime::new(config()).unwrap();
        let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
        for case in 0..6 {
            let mut l = launch(&rt, 1, 2, 2, &[0; 4]);
            let y = l.output.clone();
            let m = l.mean.clone();
            match case {
                0 => l.output = l.input.clone(),
                1 => {
                    l.mean = l.input.clone();
                    l.mean.length_bytes = 4;
                }
                2 => {
                    l.mean = l.output.clone();
                    l.mean.length_bytes = 4;
                }
                3 => l.mean.element_type = ElementType::F16,
                4 => l.output.length_bytes -= 2,
                5 => {
                    l.input.offset_bytes += 1;
                    l.input.length_bytes -= 1;
                }
                _ => unreachable!(),
            }
            assert!(ReferenceDeviceCommand::row_center_bf16_tensor(l).is_err());
            assert!(y.read().iter().chain(m.read().iter()).all(|&v| v == 0xa5));
        }
        let mut l = launch(&rt, 1, 2, 2, &[0; 4]);
        let y = l.output.clone();
        let m = l.mean.clone();
        l.mean = launch(&foreign, 1, 2, 2, &[0; 4]).mean;
        let command = ReferenceDeviceCommand::row_center_bf16_tensor(l).unwrap();
        let mut stream = rt.create_stream().unwrap();
        assert!(rt
            .submit_entries(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                false
            )
            .is_err());
        assert_eq!(rt.snapshot().submissions, 0);
        assert!(y.read().iter().chain(m.read().iter()).all(|&v| v == 0xa5));
    }
}
