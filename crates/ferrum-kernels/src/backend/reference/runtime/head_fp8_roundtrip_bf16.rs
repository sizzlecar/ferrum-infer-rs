use super::*;
use ferrum_runtime::HeadFp8RoundtripBf16Geometry;
pub(crate) struct ReferenceHeadFp8RoundtripBf16Launch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: HeadFp8RoundtripBf16Geometry,
    pub(crate) indices: Option<ReferenceBufferRegion>,
}
impl ReferenceHeadFp8RoundtripBf16Launch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        for (r, bytes) in [
            (&self.input, self.geometry.source_bytes()),
            (&self.output, self.geometry.data_bytes()),
        ] {
            r.validate_range()?;
            if r.element_type != ElementType::Bf16
                || r.length_bytes as u64 != bytes
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "head FP8 roundtrip requires exact aligned BF16 extents",
                ));
            }
        }
        match (self.geometry.gather(), &self.indices) {
            (None, None) => {}
            (Some(g), Some(indices)) => {
                indices.validate_range()?;
                if indices.element_type != ElementType::I32
                    || indices.length_bytes as u64 != g.index_bytes()
                    || (indices.offset_bytes - indices.allocation.logical_offset_bytes) % 4 != 0
                    || indices.overlaps(&self.output)
                    || indices.overlaps(&self.input)
                {
                    return Err(ReferenceDeviceRuntimeError::contract(
                        "indexed head FP8 map extent, dtype or overlap differs",
                    ));
                }
            }
            _ => {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "indexed head FP8 map and geometry differ",
                ))
            }
        }
        if self.input.overlaps(&self.output) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "head FP8 roundtrip output aliases input",
            ));
        }
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        self.input.validate_runtime(runtime)?;
        self.output.validate_runtime(runtime)?;
        if let Some(indices) = &self.indices {
            indices.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
/// Exact E4M3FN RNE conversion followed by exact F32 expansion. Finite input
/// has already been clamped. Subnormal E4M3 spacing is 2^-9; normal spacing
/// keeps three significand fraction bits. Sign includes signed zero.
fn e4m3_roundtrip(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    let magnitude = bits & 0x7fff_ffff;
    let value = if magnitude < 0x3c80_0000 {
        (x.abs() * 512.).round_ties_even() / 512.
    } else {
        f32::from_bits((magnitude + 0x7_ffff + ((magnitude >> 20) & 1)) & 0x7ff0_0000)
    };
    f32::from_bits(value.to_bits() | sign)
}
pub(super) fn execute(l: &ReferenceHeadFp8RoundtripBf16Launch) {
    let g = l.geometry;
    let mut storage =
        AllocationLocks::new([&l.input, &l.output, l.indices.as_ref().unwrap_or(&l.input)]);
    for group in 0..g.groups() {
        let start = (group * g.head_dim()) as usize;
        let width = g.head_dim() as usize;
        let source_start = if let Some(gather) = g.gather() {
            let row = (start as u64) / gather.columns();
            let source = storage.read_i32(2, (row % gather.output_rows()) as usize);
            if source < 0 || source as u64 >= gather.source_rows() {
                for i in start..start + width {
                    storage.write_bits(1, i, 0);
                }
                continue;
            }
            (((row / gather.output_rows()) * gather.source_rows() + source as u64)
                * gather.columns()
                + start as u64 % gather.columns()) as usize
        } else {
            start
        };
        let mut amax = 0f32;
        for i in start..start + width {
            let x = bf16::from_bits(storage.read_bits(0, source_start + i - start))
                .to_f32()
                .abs();
            amax = if x.is_nan() || amax.is_nan() {
                f32::NAN
            } else {
                amax.max(x)
            };
        }
        let scale = amax / 448.;
        let inverse = if scale == 0. { 0. } else { 1. / scale };
        for i in start..start + width {
            let x = bf16::from_bits(storage.read_bits(0, source_start + i - start)).to_f32();
            let normalized = x * inverse;
            let quantized = e4m3_roundtrip(normalized.clamp(-448., 448.));
            storage.write_bits(1, i, bf16::from_f32(quantized * scale).to_bits());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Independent exhaustive-code nearest-neighbor oracle, including all ties.
    fn decode(code: u32) -> f64 {
        let e = code >> 3;
        let m = code & 7;
        if e == 0 {
            f64::from(m) / 512.
        } else {
            (1. + f64::from(m) / 8.) * 2f64.powi(e as i32 - 7)
        }
    }
    #[test]
    fn e4m3_rounding_checks_all_codes_midpoints_signs_and_neighbors() {
        for code in 0..127 {
            let x = decode(code) as f32;
            assert_eq!(e4m3_roundtrip(x).to_bits(), x.to_bits());
            assert_eq!(e4m3_roundtrip(-x).to_bits(), (-x).to_bits());
        }
        for code in 0..126 {
            let a = decode(code) as f32;
            let b = decode(code + 1) as f32;
            let mid = (a + b) / 2.;
            for x in [mid.next_down(), mid, mid.next_up()] {
                let expected = if x < mid || (x == mid && code % 2 == 0) {
                    a
                } else {
                    b
                };
                assert_eq!(e4m3_roundtrip(x), expected);
                assert_eq!(e4m3_roundtrip(-x).to_bits(), (-expected).to_bits());
            }
        }
    }
    use super::super::tests::{buffer, config};
    fn launch(
        rt: &ReferenceDeviceRuntime,
        head: u64,
        x: &[u16],
    ) -> ReferenceHeadFp8RoundtripBf16Launch {
        let t = ferrum_runtime::ResolvedTensorSpec::new(
            vec![1, 1, x.len() as u64],
            ElementType::Bf16,
            ferrum_runtime::ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        let geometry = HeadFp8RoundtripBf16Geometry::new(
            &t,
            &t,
            &HeadFp8RoundtripBf16Geometry::attributes(head).unwrap(),
        )
        .unwrap();
        ReferenceHeadFp8RoundtripBf16Launch {
            input: buffer(
                rt,
                "resource.fp8.x",
                &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
                16,
                ElementType::Bf16,
            )
            .region(0..geometry.data_bytes())
            .unwrap(),
            output: buffer(
                rt,
                "resource.fp8.y",
                &vec![0xa5; x.len() * 2],
                16,
                ElementType::Bf16,
            )
            .region(0..geometry.data_bytes())
            .unwrap(),
            geometry,
            indices: None,
        }
    }
    fn run(rt: &ReferenceDeviceRuntime, l: ReferenceHeadFp8RoundtripBf16Launch) -> Vec<u16> {
        let y = l.output.clone();
        let c = ReferenceDeviceCommand::head_fp8_roundtrip_bf16_tensor(l).unwrap();
        let mut stream = rt.create_stream().unwrap();
        rt.submit_entries(
            &mut stream,
            vec![(DeviceCommandPhase::Compute, Some(0), c)],
            false,
        )
        .unwrap();
        y.read()
            .chunks_exact(2)
            .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
            .collect()
    }
    #[test]
    fn fp8_roundtrip_keeps_zero_scale_subnormals_and_nonfinite_semantics() {
        let rt = ReferenceDeviceRuntime::new(config()).unwrap();
        let y = run(
            &rt,
            launch(&rt, 2, &[0, 0x8000, 1, 0, 0x7f80, 0x3f80, 0x7fc1, 0x3f80]),
        );
        assert_eq!(&y[..2], &[0, 0x8000]);
        assert_eq!(y[2], 1);
        assert!(y[3..].iter().all(|&b| bf16::from_bits(b).is_nan()));
        let x = [448., 1., 0., -448., 224., 0.5, 0., -224.].map(|x| bf16::from_f32(x).to_bits());
        assert_eq!(run(&rt, launch(&rt, 4, &x)), x);
    }
    #[test]
    fn fp8_roundtrip_preflight_rejects_alias_extent_dtype_and_foreign_owner() {
        let rt = ReferenceDeviceRuntime::new(config()).unwrap();
        let other = ReferenceDeviceRuntime::new(config()).unwrap();
        for case in 0..4 {
            let mut l = launch(&rt, 2, &[0; 4]);
            let y = l.output.clone();
            match case {
                0 => l.output = l.input.clone(),
                1 => l.input.length_bytes -= 2,
                2 => l.input.element_type = ElementType::F16,
                _ => l.input.offset_bytes += 1,
            };
            assert!(ReferenceDeviceCommand::head_fp8_roundtrip_bf16_tensor(l).is_err());
            assert!(y.read().iter().all(|&v| v == 0xa5));
        }
        let mut l = launch(&rt, 2, &[0; 4]);
        l.input = launch(&other, 2, &[0; 4]).input;
        let y = l.output.clone();
        let c = ReferenceDeviceCommand::head_fp8_roundtrip_bf16_tensor(l).unwrap();
        let mut stream = rt.create_stream().unwrap();
        assert!(rt
            .submit_entries(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), c)],
                false
            )
            .is_err());
        assert!(y.read().iter().all(|&v| v == 0xa5));
        assert_eq!(rt.snapshot().submissions, 0);
    }
}
