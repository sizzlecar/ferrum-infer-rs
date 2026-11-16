//! Pure BF16 activation quantization ABI; no weight or alpha argument.
use super::*;

pub const NVFP4_QUANTIZE_BF16_PROFILE_RCP_APPROX_FTZ: u32 = 1;
pub const NVFP4_QUANTIZE_BF16_EXPORTS: [&str; 3] = [
    "ferrum_nvfp4_quantize_bf16_plan_v1",
    "ferrum_nvfp4_quantize_bf16_validate_v1",
    "ferrum_nvfp4_quantize_bf16_run_v1",
];
pub const NVFP4_QUANTIZE_BF16_NUMERICAL_IDENTITY: &str = concat!(
    "nvfp4-bf16-activation-quantize-v1;sm120a;group16;",
    "bf16-abs-max;rcp.approx.ftz.f32;mul.rn.ftz.f32;setp.eq.ftz.f32;",
    "E4M3FN-RNE-satfinite;E2M1-RNE-satfinite;low-nibble-even;",
    "scale-layout128x4;zero-padding;zero-or-subnormal-amax-multiplier-zero;",
    "finite-input;positive-finite-global;no-alpha;no-dynamic-amax;no-pdl"
);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Nvfp4QuantizeBf16RequestV1 {
    pub abi: u32,
    pub size: u32,
    pub rows: u32,
    pub inputs: u32,
    pub device_ordinal: u32,
    pub architecture: u32,
    pub multiprocessors: u32,
    pub profile: u32,
}
impl Nvfp4QuantizeBf16RequestV1 {
    pub fn new(
        rows: u32,
        inputs: u32,
        device: CutlassNvfp4Device,
    ) -> Result<Self, CutlassNvfp4Error> {
        let request = Self {
            abi: 1,
            size: size_of::<Self>() as u32,
            rows,
            inputs,
            device_ordinal: device.ordinal,
            architecture: device.architecture,
            multiprocessors: device.multiprocessors,
            profile: NVFP4_QUANTIZE_BF16_PROFILE_RCP_APPROX_FTZ,
        };
        request.validate()?;
        Ok(request)
    }
    pub fn validate(&self) -> Result<(), CutlassNvfp4Error> {
        if self.abi != 1
            || self.size as usize != size_of::<Self>()
            || [self.rows, self.inputs, self.multiprocessors]
                .into_iter()
                .any(|v| v == 0 || v > i32::MAX as u32)
            || self.device_ordinal > i32::MAX as u32
        {
            return Err(CutlassNvfp4Error::InvalidArgument);
        }
        if self.architecture != 1200
            || self.inputs % 16 != 0
            || self.profile != NVFP4_QUANTIZE_BF16_PROFILE_RCP_APPROX_FTZ
        {
            return Err(CutlassNvfp4Error::UnsupportedGeometry);
        }
        self.input_bytes()?;
        self.packed_bytes()?;
        self.scale_bytes()?;
        Ok(())
    }
    pub fn input_bytes(self) -> Result<u64, CutlassNvfp4Error> {
        extent(self.rows.into(), u64::from(self.inputs) * 2)
    }
    pub fn packed_bytes(self) -> Result<u64, CutlassNvfp4Error> {
        extent(self.rows.into(), u64::from(self.inputs) / 2)
    }
    pub fn scale_bytes(self) -> Result<u64, CutlassNvfp4Error> {
        scale_extent(self.rows, self.inputs)
    }
    pub fn plan(self) -> Result<Nvfp4QuantizeBf16PlanV1, CutlassNvfp4Error> {
        self.validate()?;
        let blocks = self
            .scale_bytes()?
            .div_ceil(128)
            .min(u64::from(self.multiprocessors) * 8)
            .min(65535);
        Ok(Nvfp4QuantizeBf16PlanV1 {
            request: self,
            abi: 1,
            size: size_of::<Nvfp4QuantizeBf16PlanV1>() as u32,
            kernel_id: 1,
            blocks: blocks as u32,
            threads: 128,
            reserved: [0; 3],
        })
    }
}

/// No scratch or persistent plan state. Grid-stride work covers the complete
/// padded scale buffer, while writes to packed data cover only logical rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Nvfp4QuantizeBf16PlanV1 {
    pub request: Nvfp4QuantizeBf16RequestV1,
    pub abi: u32,
    pub size: u32,
    pub kernel_id: u32,
    pub blocks: u32,
    pub threads: u32,
    pub reserved: [u32; 3],
}
impl Nvfp4QuantizeBf16PlanV1 {
    pub fn validate(&self) -> Result<(), CutlassNvfp4Error> {
        if self.request.plan()? != *self {
            return Err(CutlassNvfp4Error::Plan);
        }
        Ok(())
    }
    pub fn fingerprint_material(self) -> [u8; 64] {
        let r = self.request;
        let words = [
            r.abi,
            r.size,
            r.rows,
            r.inputs,
            r.device_ordinal,
            r.architecture,
            r.multiprocessors,
            r.profile,
            self.abi,
            self.size,
            self.kernel_id,
            self.blocks,
            self.threads,
            self.reserved[0],
            self.reserved[1],
            self.reserved[2],
        ];
        let mut out = [0; 64];
        for (word, bytes) in words.into_iter().zip(out.chunks_exact_mut(4)) {
            bytes.copy_from_slice(&word.to_le_bytes());
        }
        out
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Nvfp4QuantizeBf16BindingsV1 {
    pub input: CutlassNvfp4Span,
    pub global_scale: CutlassNvfp4Span,
    pub packed: CutlassNvfp4Span,
    pub scales: CutlassNvfp4Span,
}
impl Nvfp4QuantizeBf16BindingsV1 {
    pub fn validate(&self, plan: &Nvfp4QuantizeBf16PlanV1) -> Result<(), CutlassNvfp4Error> {
        plan.validate()?;
        self.input.validate(plan.request.input_bytes()?, 16)?;
        self.global_scale.validate(4, 4)?;
        self.packed.validate(plan.request.packed_bytes()?, 16)?;
        self.scales.validate(plan.request.scale_bytes()?, 16)?;
        for write in [self.packed, self.scales] {
            for read in [self.input, self.global_scale] {
                disjoint(write, read)?;
            }
        }
        disjoint(self.packed, self.scales)
    }
}
const _: () = {
    assert!(size_of::<Nvfp4QuantizeBf16RequestV1>() == 32);
    assert!(size_of::<Nvfp4QuantizeBf16PlanV1>() == 64);
    assert!(size_of::<Nvfp4QuantizeBf16BindingsV1>() == 64);
    assert!(offset_of!(Nvfp4QuantizeBf16PlanV1, threads) == 48);
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pure_quantization_accepts_group_tail_and_pads_scales_without_padding_values() {
        let request = Nvfp4QuantizeBf16RequestV1::new(
            129,
            80,
            CutlassNvfp4Device {
                ordinal: 0,
                architecture: 1200,
                multiprocessors: 170,
                maximum_dynamic_shared_bytes: 0,
            },
        )
        .unwrap();
        assert_eq!(request.input_bytes().unwrap(), 129 * 80 * 2);
        assert_eq!(request.packed_bytes().unwrap(), 129 * 40);
        assert_eq!(request.scale_bytes().unwrap(), 256 * 8);
        let plan = request.plan().unwrap();
        plan.validate().unwrap();
        let mut forged = plan;
        forged.blocks += 1;
        assert!(forged.validate().is_err());
        let span = |address, bytes| CutlassNvfp4Span { address, bytes };
        let good = Nvfp4QuantizeBf16BindingsV1 {
            input: span(0x10000, request.input_bytes().unwrap()),
            global_scale: span(0x20000, 4),
            packed: span(0x30000, request.packed_bytes().unwrap()),
            scales: span(0x40000, request.scale_bytes().unwrap()),
        };
        good.validate(&plan).unwrap();
        for address in [
            good.input.address,
            good.global_scale.address,
            good.packed.address,
        ] {
            let mut bad = good;
            bad.scales.address = address;
            assert!(bad.validate(&plan).is_err());
        }
        let mut bad = good;
        bad.packed.bytes += 16;
        assert!(bad.validate(&plan).is_err());
    }
}
