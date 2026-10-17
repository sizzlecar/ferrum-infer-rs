//! Static-link adapter for the optional locked upstream CUDA artifact.
//!
//! This layer owns no GPU memory and performs no allocation, synchronization,
//! provider selection or fallback. Preparation queries the current device;
//! launch methods only validate retained geometry/spans and enqueue work.
//! Every unsafe method requires same-device live leases through stream completion.
pub use ferrum_native_ops::upstream_extra_linear::UpstreamExtraLinearFormat as ExtraFormat;
#[cfg(any(feature = "cuda-upstream-extra-linear", test))]
use ferrum_native_ops::upstream_extra_linear::UPSTREAM_EXTRA_LINEAR_OPERATOR;
use ferrum_native_ops::upstream_extra_linear::{
    UpstreamExtraLinearPrefillRequestV2, UpstreamExtraLinearRequestV1,
};
use ferrum_native_ops::upstream_linear::*;
pub use ferrum_native_ops::upstream_linear::{
    UpstreamLinearAlgorithm as Algorithm, UpstreamLinearArithmetic as Arithmetic,
    UpstreamLinearDevice as Device, UpstreamLinearError as Error, UpstreamLinearFormat as Format,
    UpstreamLinearLayout as Layout,
};
use std::ffi::c_void;
mod pdl;
pub use pdl::MmvqLaunchKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceSpan {
    pub address: u64,
    pub bytes: u64,
}
impl DeviceSpan {
    pub(super) fn checked(self, required: u64, alignment: u64) -> Result<*mut c_void, Error> {
        if required == 0 && self.bytes == 0 {
            return Ok(std::ptr::null_mut());
        }
        if self.address == 0
            || self.address % alignment != 0
            || self.bytes < required
            || self.address.checked_add(self.bytes).is_none()
            || usize::try_from(self.address).is_err()
        {
            return Err(Error::Span);
        }
        Ok(self.address as usize as *mut c_void)
    }
}
pub(super) fn distinct(spans: &[DeviceSpan]) -> Result<(), Error> {
    for (i, a) in spans.iter().enumerate() {
        for b in &spans[i + 1..] {
            if a.bytes != 0
                && b.bytes != 0
                && a.address < b.address.checked_add(b.bytes).ok_or(Error::Span)?
                && b.address < a.address.checked_add(a.bytes).ok_or(Error::Span)?
            {
                return Err(Error::Span);
            }
        }
    }
    Ok(())
}
pub(super) fn strided_bytes(
    rows: u32,
    columns: u32,
    stride: u32,
    element: u64,
) -> Result<u64, Error> {
    if stride < columns {
        return Err(Error::Span);
    }
    u64::from(rows - 1)
        .checked_mul(u64::from(stride))
        .and_then(|n| n.checked_add(u64::from(columns)))
        .and_then(|n| n.checked_mul(element))
        .ok_or(Error::Extent)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    #[cfg(any(feature = "cuda-upstream-linear", test))]
    Original,
    #[cfg(any(feature = "cuda-upstream-extra-linear", test))]
    Extra,
}
impl Family {
    const fn operator(self) -> &'static str {
        match self {
            #[cfg(any(feature = "cuda-upstream-linear", test))]
            Self::Original => UPSTREAM_LINEAR_OPERATOR,
            #[cfg(any(feature = "cuda-upstream-extra-linear", test))]
            Self::Extra => UPSTREAM_EXTRA_LINEAR_OPERATOR,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(any(feature = "cuda-upstream-linear", test)), allow(dead_code))]
enum SmallMmqPack {
    LegacyF32,
    DirectF16Memset,
    DirectF16KernelClear,
}
impl SmallMmqPack {
    #[cfg(any(feature = "cuda-upstream-linear", test))]
    fn from_exports<'a>(exports: impl IntoIterator<Item = &'a str>) -> Self {
        let mut capability = Self::LegacyF32;
        for export in exports {
            match export {
                "ferrum_upstream_mmq_pack_f16_small_v2" => return Self::DirectF16KernelClear,
                "ferrum_upstream_mmq_pack_f16_small_v1" => capability = Self::DirectF16Memset,
                _ => {}
            }
        }
        capability
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(any(feature = "cuda-upstream-linear", test)), allow(dead_code))]
enum SmallMmvqPack {
    F32Memset,
    F32KernelClear,
}
impl SmallMmvqPack {
    #[cfg(any(feature = "cuda-upstream-linear", test))]
    fn from_exports<'a>(exports: impl IntoIterator<Item = &'a str>) -> Self {
        if exports
            .into_iter()
            .any(|export| export == "ferrum_upstream_mmvq_pack_kernel_clear_small_v1")
        {
            Self::F32KernelClear
        } else {
            Self::F32Memset
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreparedUpstreamLinear {
    raw: UpstreamLinearPlanV1,
    arithmetic: Arithmetic,
    family: Family,
    small_mmq_pack: SmallMmqPack,
    #[cfg_attr(not(any(feature = "cuda-upstream-linear", test)), allow(dead_code))]
    small_mmvq_pack: SmallMmvqPack,
    q4_weight_decode: ferrum_types::Q4MmvqWeightDecode,
    q5_weight_decode: ferrum_types::Q5MmvqWeightDecode,
    mmvq_launch: MmvqLaunchKind,
}
impl PreparedUpstreamLinear {
    pub fn new(
        algorithm: Algorithm,
        arithmetic: Arithmetic,
        format: Format,
        layout: Layout,
        rows: u32,
        inputs: u32,
        outputs: u32,
        device: Device,
    ) -> Result<Self, Error> {
        let request = UpstreamLinearRequestV1::new(format, layout, rows, inputs, outputs, device)?;
        #[cfg(not(feature = "cuda-upstream-linear"))]
        {
            let _ = (algorithm, arithmetic, request);
            Err(Error::Unavailable)
        }
        #[cfg(feature = "cuda-upstream-linear")]
        {
            let mut raw = UpstreamLinearPlanV1::default();
            let call = match algorithm {
                Algorithm::Mmq if rows > 32 => ffi::ferrum_upstream_mmq_prefill_plan_v1,
                Algorithm::Mmq => ffi::ferrum_upstream_mmq_plan_v1,
                Algorithm::Mmvq => ffi::ferrum_upstream_mmvq_plan_v1,
            };
            // Native preparation checks the current device and configures MMQ
            // dynamic shared memory once, outside any CUDA graph capture.
            Error::from_status(unsafe { call(&request, &mut raw) })?;
            raw.validate_identity(&request, algorithm)?;
            Ok(Self {
                raw,
                arithmetic,
                family: Family::Original,
                small_mmq_pack: compiled_small_mmq_pack(),
                small_mmvq_pack: compiled_small_mmvq_pack(),
                q4_weight_decode: ferrum_types::Q4MmvqWeightDecode::Baseline,
                q5_weight_decode: ferrum_types::Q5MmvqWeightDecode::Baseline,
                mmvq_launch: MmvqLaunchKind::Ordered,
            })
        }
    }
    /// Prepare the independently qualified extra-format operator. Large rows
    /// are deliberately rejected here even if the archive exports a prefill primitive.
    pub fn new_extra(
        algorithm: Algorithm,
        arithmetic: Arithmetic,
        format: ExtraFormat,
        layout: Layout,
        rows: u32,
        inputs: u32,
        outputs: u32,
        device: Device,
    ) -> Result<Self, Error> {
        let request =
            UpstreamExtraLinearRequestV1::new(format, layout, rows, inputs, outputs, device)?;
        #[cfg(not(feature = "cuda-upstream-extra-linear"))]
        {
            let _ = (algorithm, arithmetic, request);
            Err(Error::Unavailable)
        }
        #[cfg(feature = "cuda-upstream-extra-linear")]
        {
            let mut raw = UpstreamLinearPlanV1::default();
            let call = match algorithm {
                Algorithm::Mmq => extra_ffi::ferrum_upstream_extra_mmq_plan_v1,
                Algorithm::Mmvq => extra_ffi::ferrum_upstream_extra_mmvq_plan_v1,
            };
            Error::from_status(unsafe { call(request.as_raw(), &mut raw) })?;
            request.validate_plan_identity(&raw, algorithm)?;
            Ok(Self {
                raw,
                arithmetic,
                family: Family::Extra,
                small_mmq_pack: SmallMmqPack::LegacyF32,
                small_mmvq_pack: SmallMmvqPack::F32Memset,
                q4_weight_decode: ferrum_types::Q4MmvqWeightDecode::Baseline,
                q5_weight_decode: ferrum_types::Q5MmvqWeightDecode::Baseline,
                mmvq_launch: MmvqLaunchKind::Ordered,
            })
        }
    }
    /// The artifact family is part of prepared/retained validation identity.
    pub fn operator(&self) -> &'static str {
        self.family.operator()
    }
    /// Independently qualified extra MMQ Columns M33..2048. This does not
    /// change `new_extra`, register a profile, or select a numerical fallback.
    pub fn new_extra_prefill(
        format: ExtraFormat,
        rows: u32,
        inputs: u32,
        outputs: u32,
        device: Device,
    ) -> Result<Self, Error> {
        let request =
            UpstreamExtraLinearPrefillRequestV2::new(format, rows, inputs, outputs, device)?;
        #[cfg(not(feature = "cuda-upstream-extra-linear"))]
        {
            let _ = request;
            Err(Error::Unavailable)
        }
        #[cfg(feature = "cuda-upstream-extra-linear")]
        {
            let mut raw = UpstreamLinearPlanV1::default();
            Error::from_status(unsafe {
                extra_ffi::ferrum_upstream_extra_mmq_prefill_plan_v2(request.as_raw(), &mut raw)
            })?;
            request.validate_plan_identity(&raw)?;
            Ok(Self {
                raw,
                arithmetic: Arithmetic::MarkerV2,
                family: Family::Extra,
                small_mmq_pack: SmallMmqPack::LegacyF32,
                small_mmvq_pack: SmallMmvqPack::F32Memset,
                q4_weight_decode: ferrum_types::Q4MmvqWeightDecode::Baseline,
                q5_weight_decode: ferrum_types::Q5MmvqWeightDecode::Baseline,
                mmvq_launch: MmvqLaunchKind::Ordered,
            })
        }
    }
    pub fn geometry(&self) -> &UpstreamLinearPlanV1 {
        &self.raw
    }
    pub fn arithmetic(&self) -> Arithmetic {
        self.arithmetic
    }
    /// Input-pack kernel and transfer counts for retained projection accounting.
    /// The linked archive capability is captured during cold preparation. Older
    /// archives and every other route retain their existing work accounting.
    pub fn pack_work_counts(&self) -> (u64, u64) {
        if self.mmvq_launch == MmvqLaunchKind::Programmatic {
            return (3, 0);
        }
        if self.arithmetic == Arithmetic::V1 {
            return (2, 0);
        }
        #[cfg(any(feature = "cuda-upstream-linear", test))]
        let small_original_mmq = self.family == Family::Original
            && self.arithmetic == Arithmetic::MarkerV2
            && self.raw.algorithm == Algorithm::Mmq as u32
            && (1..=32).contains(&self.raw.request.rows);
        #[cfg(not(any(feature = "cuda-upstream-linear", test)))]
        let small_original_mmq = false;
        if small_original_mmq {
            match self.small_mmq_pack {
                SmallMmqPack::DirectF16Memset => return (1, 1),
                SmallMmqPack::DirectF16KernelClear => return (2, 0),
                SmallMmqPack::LegacyF32 => {}
            }
        }
        #[cfg(any(feature = "cuda-upstream-linear", test))]
        if self.family == Family::Original
            && self.arithmetic == Arithmetic::MarkerV2
            && self.raw.algorithm == Algorithm::Mmvq as u32
            && (1..=32).contains(&self.raw.request.rows)
            && self.small_mmvq_pack == SmallMmvqPack::F32KernelClear
        {
            // Row clear, the retained F16-to-F32 conversion, and quantization.
            return (3, 0);
        }
        (2, 1 + u64::from(self.raw.guard_blocks != 0))
    }
    /// The dot implementation is selected once after the existing numerical
    /// route has supplied the exact native plan. Pack, cast and validation
    /// retain the original MarkerV2 ABI and resource extents.
    pub fn with_q4_mmvq_weight_decode(
        mut self,
        mode: ferrum_types::Q4MmvqWeightDecode,
    ) -> Result<Self, Error> {
        if mode == ferrum_types::Q4MmvqWeightDecode::Auto {
            return Err(Error::InvalidArgument);
        }
        if mode != ferrum_types::Q4MmvqWeightDecode::Baseline {
            if !cfg!(feature = "cuda-q4-shared-weight-decode") {
                return Err(Error::Unavailable);
            }
            let eligible = self.q4_shared_weight_decode_eligible()
                || (mode == ferrum_types::Q4MmvqWeightDecode::SharedM4M8
                    && self.shared_weight_decode_geometry(Format::Q4K, 4));
            if !eligible {
                return Err(Error::UnsupportedGeometry);
            }
        }
        self.q4_weight_decode = mode;
        self.mmvq_launch = MmvqLaunchKind::Ordered;
        Ok(self)
    }
    pub fn q4_mmvq_weight_decode(&self) -> ferrum_types::Q4MmvqWeightDecode {
        self.q4_weight_decode
    }
    pub fn q4_shared_weight_decode_eligible(&self) -> bool {
        self.shared_weight_decode_geometry(Format::Q4K, 8)
    }
    /// Cold policy resolution against the already validated native plan. The
    /// caller supplies cached feature/export qualification; device attributes
    /// are the actual values retained by native preparation, never GPU names.
    pub(crate) fn automatic_q4_mmvq_weight_decode(
        &self,
        native_available: bool,
    ) -> ferrum_types::Q4MmvqWeightDecode {
        if self.automatic_shared_weight_decode_device(native_available)
            && self.q4_shared_weight_decode_eligible()
        {
            ferrum_types::Q4MmvqWeightDecode::SharedM8
        } else {
            ferrum_types::Q4MmvqWeightDecode::Baseline
        }
    }
    pub(crate) fn automatic_q5_mmvq_weight_decode(
        &self,
        native_available: bool,
    ) -> ferrum_types::Q5MmvqWeightDecode {
        if self.automatic_shared_weight_decode_device(native_available)
            && self.q5_shared_weight_decode_eligible()
        {
            ferrum_types::Q5MmvqWeightDecode::SharedM8
        } else {
            ferrum_types::Q5MmvqWeightDecode::Baseline
        }
    }
    fn automatic_shared_weight_decode_device(&self, native_available: bool) -> bool {
        // Automatic promotion is currently qualified on SM120 only. Explicit
        // modes retain their existing native capability/geometry validation.
        native_available && self.raw.request.cc == 1200 && self.raw.request.sm_count > 0
    }
    fn shared_weight_decode_geometry(&self, format: Format, rows: u32) -> bool {
        #[cfg(any(feature = "cuda-upstream-linear", test))]
        {
            let p = &self.raw;
            self.family == Family::Original
                && self.arithmetic == Arithmetic::MarkerV2
                && p.algorithm == Algorithm::Mmvq as u32
                && p.request.format == format as u32
                && p.request.layout == Layout::Columns as u32
                && p.request.rows == rows
                && p.ncols == rows
                && p.channels == 1
                && matches!((rows, p.nwarps), (4, 4) | (8, 2))
                && p.rows_per_block == 2
                && p.small_k == 0
                && p.fixup_bytes == 0
        }
        #[cfg(not(any(feature = "cuda-upstream-linear", test)))]
        {
            let _ = (format, rows);
            false
        }
    }
    pub fn with_q5_mmvq_weight_decode(
        mut self,
        mode: ferrum_types::Q5MmvqWeightDecode,
    ) -> Result<Self, Error> {
        if mode == ferrum_types::Q5MmvqWeightDecode::Auto {
            return Err(Error::InvalidArgument);
        }
        if mode != ferrum_types::Q5MmvqWeightDecode::Baseline {
            if !cfg!(feature = "cuda-q5-shared-weight-decode") {
                return Err(Error::Unavailable);
            }
            let eligible = self.q5_shared_weight_decode_eligible()
                || (mode == ferrum_types::Q5MmvqWeightDecode::SharedM4M8
                    && self.shared_weight_decode_geometry(Format::Q5K, 4));
            if !eligible {
                return Err(Error::UnsupportedGeometry);
            }
        }
        self.q5_weight_decode = mode;
        self.mmvq_launch = MmvqLaunchKind::Ordered;
        Ok(self)
    }
    pub fn q5_mmvq_weight_decode(&self) -> ferrum_types::Q5MmvqWeightDecode {
        self.q5_weight_decode
    }
    /// Seal automatic scheduling after the actual ordinary/shared dot has been
    /// selected. Missing artifact capability or unsupported CC/PTX stays ordered.
    pub fn with_automatic_mmvq_launch(mut self) -> Result<Self, Error> {
        self.mmvq_launch = pdl::qualify(&self)?;
        Ok(self)
    }
    pub fn mmvq_launch_kind(&self) -> MmvqLaunchKind {
        self.mmvq_launch
    }
    pub fn q5_shared_weight_decode_eligible(&self) -> bool {
        self.shared_weight_decode_geometry(Format::Q5K, 8)
    }
    pub fn row_poison_bytes(&self) -> u64 {
        if self.arithmetic == Arithmetic::MarkerV2 {
            u64::from(self.raw.request.rows) * 4
        } else {
            0
        }
    }
    /// Enqueue a single static weight scan; its flag must remain associated
    /// with this exact physical weight view and leased for every dependent dot.
    /// Success means queued, not that the data passed the device check.
    /// # Safety
    /// `weights`/`flag` are valid disjoint same-device allocations. The caller
    /// preserves the flag, establishes stream dependencies, and never mutates W.
    pub unsafe fn check_weights(
        &self,
        weights: DeviceSpan,
        flag: DeviceSpan,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        if self.arithmetic != Arithmetic::MarkerV2 {
            return Err(Error::InvalidArgument);
        }
        let w = weights.checked(self.raw.weight_bytes, 4)?;
        let f = flag.checked(4, 4)?;
        distinct(&[weights, flag])?;
        #[cfg(feature = "cuda-upstream-linear")]
        {
            let api = dispatch::api(self.family, self.raw.algorithm)?;
            Error::from_status(unsafe { (api.check_weights)(&self.raw, w, f, stream) })
        }
        #[cfg(not(feature = "cuda-upstream-linear"))]
        {
            let _ = (w, f, stream);
            Err(Error::Unavailable)
        }
    }
    /// # Safety
    /// All spans are live on the prepared device/stream. Outputs remain leased
    /// through dot/cast. V1 additionally requires its qualified finite domain.
    pub unsafe fn pack(
        &self,
        input: DeviceSpan,
        stride: u32,
        converted: DeviceSpan,
        packed: DeviceSpan,
        row_poison: DeviceSpan,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        let r = &self.raw.request;
        let x = input.checked(strided_bytes(r.rows, r.inputs, stride, 2)?, 2)?;
        let c = converted.checked(self.raw.converted_bytes, 16)?;
        let q = packed.checked(self.raw.packed_bytes, 16)?;
        let poison = row_poison.checked(self.row_poison_bytes(), 16)?;
        distinct(&[input, converted, packed, row_poison])?;
        #[cfg(feature = "cuda-upstream-linear")]
        {
            #[cfg(ferrum_native_mmvq_pdl)]
            if self.mmvq_launch == MmvqLaunchKind::Programmatic {
                return Error::from_status(unsafe {
                    ffi::ferrum_upstream_mmvq_pdl_pack_v2(
                        &self.raw, x, stride, c, q, poison, stream,
                    )
                });
            }
            let api = dispatch::api(self.family, self.raw.algorithm)?;
            Error::from_status(unsafe {
                match self.arithmetic {
                    Arithmetic::V1 => (api.pack_v1)(&self.raw, x, stride, c, q, stream),
                    Arithmetic::MarkerV2 => {
                        (api.pack_v2)(&self.raw, x, stride, c, q, poison, stream)
                    }
                }
            })
        }
        #[cfg(not(feature = "cuda-upstream-linear"))]
        {
            let _ = (x, c, q, poison, stream);
            Err(Error::Unavailable)
        }
    }
    /// # Safety
    /// Weights include native-required zero row padding, and `packed` was
    /// produced by this plan's exact pack ABI. Scratch must not alias any input.
    pub unsafe fn dot(
        &self,
        weights: DeviceSpan,
        packed: DeviceSpan,
        output: DeviceSpan,
        fixup: DeviceSpan,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        let w = weights.checked(self.raw.weight_bytes, 4)?;
        let q = packed.checked(self.raw.packed_bytes, 16)?;
        let o = output.checked(self.raw.output_bytes, 4)?;
        let f = fixup.checked(self.raw.fixup_bytes, 4)?;
        if self.raw.fixup_bytes == 0 && fixup.bytes != 0 {
            return Err(Error::Span);
        }
        distinct(&[weights, packed, output, fixup])?;
        #[cfg(feature = "cuda-upstream-linear")]
        {
            #[cfg(ferrum_native_mmvq_pdl)]
            if self.mmvq_launch == MmvqLaunchKind::Programmatic {
                return Error::from_status(unsafe {
                    ffi::ferrum_upstream_mmvq_pdl_dot_v1(
                        &self.raw,
                        w,
                        q,
                        o,
                        f,
                        stream,
                        pdl::dot_kind(self) as u32,
                    )
                });
            }
            let api = dispatch::api(self.family, self.raw.algorithm)?;
            let dot = match self.q4_weight_decode {
                ferrum_types::Q4MmvqWeightDecode::Auto => return Err(Error::InvalidArgument),
                ferrum_types::Q4MmvqWeightDecode::Baseline => api.dot,
                #[cfg(feature = "cuda-q4-shared-weight-decode")]
                ferrum_types::Q4MmvqWeightDecode::SharedM4M8 if self.raw.request.rows == 4 => {
                    ffi::ferrum_upstream_mmvq_q4_shared_weight_decode_m4_dot_v1
                }
                #[cfg(feature = "cuda-q4-shared-weight-decode")]
                ferrum_types::Q4MmvqWeightDecode::SharedM8
                | ferrum_types::Q4MmvqWeightDecode::SharedM4M8 => {
                    ffi::ferrum_upstream_mmvq_q4_shared_weight_decode_dot_v1
                }
                #[cfg(not(feature = "cuda-q4-shared-weight-decode"))]
                ferrum_types::Q4MmvqWeightDecode::SharedM8
                | ferrum_types::Q4MmvqWeightDecode::SharedM4M8 => return Err(Error::Unavailable),
            };
            let dot = match self.q5_weight_decode {
                ferrum_types::Q5MmvqWeightDecode::Auto => return Err(Error::InvalidArgument),
                ferrum_types::Q5MmvqWeightDecode::Baseline => dot,
                #[cfg(feature = "cuda-q5-shared-weight-decode")]
                ferrum_types::Q5MmvqWeightDecode::SharedM4M8 if self.raw.request.rows == 4 => {
                    ffi::ferrum_upstream_mmvq_q5_shared_weight_decode_m4_dot_v1
                }
                #[cfg(feature = "cuda-q5-shared-weight-decode")]
                ferrum_types::Q5MmvqWeightDecode::SharedM8
                | ferrum_types::Q5MmvqWeightDecode::SharedM4M8 => {
                    ffi::ferrum_upstream_mmvq_q5_shared_weight_decode_dot_v1
                }
                #[cfg(not(feature = "cuda-q5-shared-weight-decode"))]
                ferrum_types::Q5MmvqWeightDecode::SharedM8
                | ferrum_types::Q5MmvqWeightDecode::SharedM4M8 => return Err(Error::Unavailable),
            };
            Error::from_status(unsafe { dot(&self.raw, w, q, o, f, stream) })
        }
        #[cfg(not(feature = "cuda-upstream-linear"))]
        {
            let _ = (w, q, o, f, stream);
            Err(Error::Unavailable)
        }
    }
    /// # Safety
    /// MarkerV2 requires the real retained weight flag and row flags from this
    /// invocation, with same-stream ordering or explicit event dependencies.
    /// No host claim may substitute for either device-produced flag.
    pub unsafe fn cast(
        &self,
        input: DeviceSpan,
        output: DeviceSpan,
        stride: u32,
        row_poison: DeviceSpan,
        weight_poison: DeviceSpan,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        let r = &self.raw.request;
        let i = input.checked(self.raw.output_bytes, 4)?;
        let o = output.checked(strided_bytes(r.rows, r.outputs, stride, 2)?, 2)?;
        let rp = row_poison.checked(self.row_poison_bytes(), 16)?;
        let wp = weight_poison.checked(
            if self.arithmetic == Arithmetic::MarkerV2 {
                4
            } else {
                0
            },
            4,
        )?;
        distinct(&[input, output, row_poison, weight_poison])?;
        #[cfg(feature = "cuda-upstream-linear")]
        {
            #[cfg(ferrum_native_mmvq_pdl)]
            if self.mmvq_launch == MmvqLaunchKind::Programmatic {
                return Error::from_status(unsafe {
                    ffi::ferrum_upstream_mmvq_pdl_cast_v2(&self.raw, i, o, stride, rp, wp, stream)
                });
            }
            let api = dispatch::api(self.family, self.raw.algorithm)?;
            Error::from_status(unsafe {
                match self.arithmetic {
                    Arithmetic::V1 => (api.cast_v1)(&self.raw, i, o, stride, stream),
                    Arithmetic::MarkerV2 => (api.cast_v2)(&self.raw, i, o, stride, rp, wp, stream),
                }
            })
        }
        #[cfg(not(feature = "cuda-upstream-linear"))]
        {
            let _ = (i, o, rp, wp, stream);
            Err(Error::Unavailable)
        }
    }
}

#[cfg(feature = "cuda-upstream-linear")]
fn compiled_small_mmq_pack() -> SmallMmqPack {
    crate::native_ops::compiled_native_operator_artifacts()
        .iter()
        .find(|artifact| {
            artifact.operator == UPSTREAM_LINEAR_OPERATOR
                && artifact.backend == ferrum_types::NativeOperatorBackend::Cuda
                && artifact.linkage == ferrum_types::NativeOperatorLinkage::Static
                && artifact.operator_abi_version == "1"
        })
        .map(|artifact| SmallMmqPack::from_exports(artifact.exports.iter().map(String::as_str)))
        .unwrap_or(SmallMmqPack::LegacyF32)
}

#[cfg(feature = "cuda-upstream-linear")]
fn compiled_small_mmvq_pack() -> SmallMmvqPack {
    crate::native_ops::compiled_native_operator_artifacts()
        .iter()
        .find(|artifact| {
            artifact.operator == UPSTREAM_LINEAR_OPERATOR
                && artifact.backend == ferrum_types::NativeOperatorBackend::Cuda
                && artifact.linkage == ferrum_types::NativeOperatorLinkage::Static
                && artifact.operator_abi_version == "1"
        })
        .map(|artifact| SmallMmvqPack::from_exports(artifact.exports.iter().map(String::as_str)))
        .unwrap_or(SmallMmvqPack::F32Memset)
}

#[cfg(feature = "cuda-upstream-linear")]
mod dispatch;

#[cfg(feature = "cuda-upstream-extra-linear")]
mod extra_ffi;
#[cfg(feature = "cuda-upstream-linear")]
mod ffi;
#[cfg(all(
    test,
    any(
        feature = "cuda-q4-shared-weight-decode",
        feature = "cuda-q5-shared-weight-decode"
    )
))]
#[path = "upstream_linear/shared_weight_tests.rs"]
mod shared_weight_tests;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn small_mmvq_pack_capability_requires_its_independent_export() {
        use SmallMmvqPack::{F32KernelClear, F32Memset};
        let capability = "ferrum_upstream_mmvq_pack_kernel_clear_small_v1";
        for (exports, expected) in [
            (vec![], F32Memset),
            (vec!["ferrum_upstream_mmvq_pack_v2"], F32Memset),
            (vec!["ferrum_upstream_mmq_pack_f16_small_v2"], F32Memset),
            (vec![capability], F32KernelClear),
            (
                vec!["ferrum_upstream_mmq_pack_f16_small_v2", capability],
                F32KernelClear,
            ),
        ] {
            assert_eq!(SmallMmvqPack::from_exports(exports), expected);
        }
    }
    #[test]
    fn small_mmvq_pack_work_counts_preserve_f32_and_archive_family_row_boundaries() {
        use Algorithm::{Mmq, Mmvq};
        use Arithmetic::{MarkerV2, V1};
        use Family::{Extra, Original};
        use SmallMmvqPack::{F32KernelClear, F32Memset};
        for (family, arithmetic, algorithm, rows, capability, guard, expected) in [
            (Original, MarkerV2, Mmvq, 1, F32KernelClear, 0, (3, 0)),
            (Original, MarkerV2, Mmvq, 3, F32KernelClear, 0, (3, 0)),
            (Original, MarkerV2, Mmvq, 8, F32KernelClear, 0, (3, 0)),
            (Original, MarkerV2, Mmvq, 32, F32KernelClear, 0, (3, 0)),
            (Original, MarkerV2, Mmvq, 0, F32KernelClear, 0, (2, 1)),
            (Original, MarkerV2, Mmvq, 33, F32KernelClear, 0, (2, 1)),
            (Original, MarkerV2, Mmvq, 32, F32Memset, 0, (2, 1)),
            (Original, MarkerV2, Mmvq, 32, F32Memset, 8, (2, 2)),
            (Extra, MarkerV2, Mmvq, 8, F32KernelClear, 0, (2, 1)),
            (Original, V1, Mmvq, 8, F32KernelClear, 0, (2, 0)),
            (Original, MarkerV2, Mmq, 32, F32KernelClear, 8, (1, 1)),
            (Original, MarkerV2, Mmq, 33, F32KernelClear, 8, (2, 2)),
        ] {
            let mut raw = UpstreamLinearPlanV1::default();
            raw.algorithm = algorithm as u32;
            raw.request.rows = rows;
            raw.guard_blocks = guard;
            let prepared = PreparedUpstreamLinear {
                raw,
                arithmetic,
                family,
                small_mmq_pack: SmallMmqPack::DirectF16Memset,
                small_mmvq_pack: capability,
                q4_weight_decode: ferrum_types::Q4MmvqWeightDecode::Baseline,
                q5_weight_decode: ferrum_types::Q5MmvqWeightDecode::Baseline,
                mmvq_launch: MmvqLaunchKind::Ordered,
            };
            assert_eq!(prepared.pack_work_counts(), expected,
                "{family:?}/{arithmetic:?}/{algorithm:?}/M{rows}/capability{capability:?}/guard{guard}");
        }
    }
    #[test]
    fn small_mmq_pack_capability_prefers_versioned_kernel_clear_without_requiring_it() {
        use SmallMmqPack::{DirectF16KernelClear, DirectF16Memset, LegacyF32};
        let v1 = "ferrum_upstream_mmq_pack_f16_small_v1";
        let v2 = "ferrum_upstream_mmq_pack_f16_small_v2";
        for (exports, expected) in [
            (vec![], LegacyF32),
            (vec!["ferrum_upstream_mmq_pack_v2"], LegacyF32),
            (vec!["ferrum_upstream_mmvq_pack_f16_small_v2"], LegacyF32),
            (vec![v1], DirectF16Memset),
            (vec![v2], DirectF16KernelClear),
            (vec![v1, v2], DirectF16KernelClear),
            (vec![v2, v1], DirectF16KernelClear),
        ] {
            assert_eq!(SmallMmqPack::from_exports(exports), expected);
        }
    }
    #[test]
    fn small_mmq_direct_pack_work_counts_preserve_archive_family_and_row_boundaries() {
        use Algorithm::{Mmq, Mmvq};
        use Arithmetic::{MarkerV2, V1};
        use Family::{Extra, Original};
        use SmallMmqPack::{DirectF16KernelClear, DirectF16Memset, LegacyF32};
        // Expected values are actual pack kernels and cudaMemset transfers,
        // excluding the unchanged dot/fixup/cast and cold static weight scan.
        for (family, arithmetic, algorithm, rows, capability, guard, expected) in [
            (Original, MarkerV2, Mmq, 1, DirectF16Memset, 8, (1, 1)),
            (Original, MarkerV2, Mmq, 3, DirectF16Memset, 8, (1, 1)),
            (Original, MarkerV2, Mmq, 32, DirectF16Memset, 8, (1, 1)),
            (Original, MarkerV2, Mmq, 32, DirectF16Memset, 0, (1, 1)),
            (Original, MarkerV2, Mmq, 32, LegacyF32, 8, (2, 2)),
            (Original, MarkerV2, Mmq, 32, LegacyF32, 0, (2, 1)),
            (Original, MarkerV2, Mmq, 33, DirectF16Memset, 8, (2, 2)),
            (Original, MarkerV2, Mmq, 75, DirectF16Memset, 8, (2, 2)),
            (Original, MarkerV2, Mmq, 0, DirectF16Memset, 8, (2, 2)),
            (Extra, MarkerV2, Mmq, 32, DirectF16Memset, 8, (2, 2)),
            (Original, V1, Mmq, 32, DirectF16Memset, 8, (2, 0)),
            (Original, V1, Mmvq, 32, LegacyF32, 0, (2, 0)),
            (Original, MarkerV2, Mmvq, 32, DirectF16Memset, 0, (2, 1)),
            (Original, MarkerV2, Mmq, 1, DirectF16KernelClear, 8, (2, 0)),
            (Original, MarkerV2, Mmq, 3, DirectF16KernelClear, 8, (2, 0)),
            (Original, MarkerV2, Mmq, 32, DirectF16KernelClear, 8, (2, 0)),
            (Original, MarkerV2, Mmq, 32, DirectF16KernelClear, 0, (2, 0)),
            (Original, MarkerV2, Mmq, 33, DirectF16KernelClear, 8, (2, 2)),
            (Original, MarkerV2, Mmq, 75, DirectF16KernelClear, 8, (2, 2)),
            (Original, MarkerV2, Mmq, 0, DirectF16KernelClear, 8, (2, 2)),
            (Extra, MarkerV2, Mmq, 32, DirectF16KernelClear, 8, (2, 2)),
            (Original, V1, Mmq, 32, DirectF16KernelClear, 8, (2, 0)),
            (Original, V1, Mmvq, 32, DirectF16KernelClear, 0, (2, 0)),
            (
                Original,
                MarkerV2,
                Mmvq,
                32,
                DirectF16KernelClear,
                0,
                (2, 1),
            ),
        ] {
            let mut raw = UpstreamLinearPlanV1::default();
            raw.algorithm = algorithm as u32;
            raw.request.rows = rows;
            raw.guard_blocks = guard;
            let prepared = PreparedUpstreamLinear {
                raw,
                arithmetic,
                family,
                small_mmq_pack: capability,
                small_mmvq_pack: SmallMmvqPack::F32Memset,
                q4_weight_decode: ferrum_types::Q4MmvqWeightDecode::Baseline,
                q5_weight_decode: ferrum_types::Q5MmvqWeightDecode::Baseline,
                mmvq_launch: MmvqLaunchKind::Ordered,
            };
            assert_eq!(prepared.pack_work_counts(), expected,
                "{family:?}/{arithmetic:?}/{algorithm:?}/M{rows}/capability{capability:?}/guard{guard}");
        }
    }

    #[test]
    fn shared_q4_weight_decode_requires_the_complete_prepared_mmvq_geometry() {
        let mut raw = UpstreamLinearPlanV1::default();
        raw.algorithm = Algorithm::Mmvq as u32;
        raw.request.format = Format::Q4K as u32;
        raw.request.layout = Layout::Columns as u32;
        raw.request.rows = 8;
        raw.request.cc = 1200;
        raw.request.sm_count = 16;
        raw.ncols = 8;
        raw.channels = 1;
        raw.nwarps = 2;
        raw.rows_per_block = 2;
        let plan = PreparedUpstreamLinear {
            raw,
            arithmetic: Arithmetic::MarkerV2,
            family: Family::Original,
            small_mmq_pack: SmallMmqPack::LegacyF32,
            small_mmvq_pack: SmallMmvqPack::F32Memset,
            q4_weight_decode: ferrum_types::Q4MmvqWeightDecode::Baseline,
            q5_weight_decode: ferrum_types::Q5MmvqWeightDecode::Baseline,
            mmvq_launch: MmvqLaunchKind::Ordered,
        };
        assert!(plan.q4_shared_weight_decode_eligible());
        assert_eq!(
            plan.automatic_q4_mmvq_weight_decode(true),
            ferrum_types::Q4MmvqWeightDecode::SharedM8
        );
        assert_eq!(
            plan.automatic_q4_mmvq_weight_decode(false),
            ferrum_types::Q4MmvqWeightDecode::Baseline
        );
        assert_eq!(
            plan.automatic_q5_mmvq_weight_decode(true),
            ferrum_types::Q5MmvqWeightDecode::Baseline
        );
        for (cc, sm_count) in [(0, 16), (890, 16), (1210, 16), (1200, 0)] {
            let mut changed = plan.clone();
            changed.raw.request.cc = cc;
            changed.raw.request.sm_count = sm_count;
            assert_eq!(
                changed.automatic_q4_mmvq_weight_decode(true),
                ferrum_types::Q4MmvqWeightDecode::Baseline
            );
        }
        let mutate = |f: fn(&mut PreparedUpstreamLinear)| {
            let mut changed = plan.clone();
            f(&mut changed);
            assert!(!changed.q4_shared_weight_decode_eligible());
            assert_eq!(
                changed.automatic_q4_mmvq_weight_decode(true),
                ferrum_types::Q4MmvqWeightDecode::Baseline
            );
        };
        mutate(|p| p.family = Family::Extra);
        mutate(|p| p.arithmetic = Arithmetic::V1);
        mutate(|p| p.raw.algorithm = Algorithm::Mmq as u32);
        mutate(|p| p.raw.request.format = Format::Q5K as u32);
        mutate(|p| p.raw.request.layout = Layout::Channels as u32);
        mutate(|p| p.raw.request.rows = 4);
        mutate(|p| p.raw.ncols = 4);
        mutate(|p| p.raw.channels = 8);
        mutate(|p| p.raw.nwarps = 4);
        mutate(|p| p.raw.rows_per_block = 1);
        mutate(|p| p.raw.small_k = 1);
        mutate(|p| p.raw.fixup_bytes = 4);
        assert_eq!(
            plan.q4_mmvq_weight_decode(),
            ferrum_types::Q4MmvqWeightDecode::Baseline
        );
        assert_eq!(
            plan.clone()
                .with_q4_mmvq_weight_decode(ferrum_types::Q4MmvqWeightDecode::Baseline)
                .unwrap()
                .q4_mmvq_weight_decode(),
            ferrum_types::Q4MmvqWeightDecode::Baseline
        );
        assert!(matches!(
            plan.clone()
                .with_q4_mmvq_weight_decode(ferrum_types::Q4MmvqWeightDecode::Auto),
            Err(Error::InvalidArgument)
        ));
        #[cfg(not(feature = "cuda-q4-shared-weight-decode"))]
        assert!(matches!(
            plan.with_q4_mmvq_weight_decode(ferrum_types::Q4MmvqWeightDecode::SharedM8),
            Err(Error::Unavailable)
        ));
    }
    #[test]
    fn shared_q5_weight_decode_requires_the_complete_prepared_mmvq_geometry() {
        let mut raw = UpstreamLinearPlanV1::default();
        raw.algorithm = Algorithm::Mmvq as u32;
        raw.request.format = Format::Q5K as u32;
        raw.request.layout = Layout::Columns as u32;
        raw.request.rows = 8;
        raw.request.cc = 1200;
        raw.request.sm_count = 16;
        raw.ncols = 8;
        raw.channels = 1;
        raw.nwarps = 2;
        raw.rows_per_block = 2;
        let plan = PreparedUpstreamLinear {
            raw,
            arithmetic: Arithmetic::MarkerV2,
            family: Family::Original,
            small_mmq_pack: SmallMmqPack::LegacyF32,
            small_mmvq_pack: SmallMmvqPack::F32Memset,
            q5_weight_decode: ferrum_types::Q5MmvqWeightDecode::Baseline,
            mmvq_launch: MmvqLaunchKind::Ordered,
            q4_weight_decode: ferrum_types::Q4MmvqWeightDecode::Baseline,
        };
        assert!(plan.q5_shared_weight_decode_eligible());
        assert_eq!(
            plan.automatic_q5_mmvq_weight_decode(true),
            ferrum_types::Q5MmvqWeightDecode::SharedM8
        );
        assert_eq!(
            plan.automatic_q5_mmvq_weight_decode(false),
            ferrum_types::Q5MmvqWeightDecode::Baseline
        );
        assert_eq!(
            plan.automatic_q4_mmvq_weight_decode(true),
            ferrum_types::Q4MmvqWeightDecode::Baseline
        );
        for (cc, sm_count) in [(0, 16), (890, 16), (1210, 16), (1200, 0)] {
            let mut changed = plan.clone();
            changed.raw.request.cc = cc;
            changed.raw.request.sm_count = sm_count;
            assert_eq!(
                changed.automatic_q5_mmvq_weight_decode(true),
                ferrum_types::Q5MmvqWeightDecode::Baseline
            );
        }
        let mutate = |f: fn(&mut PreparedUpstreamLinear)| {
            let mut changed = plan.clone();
            f(&mut changed);
            assert!(!changed.q5_shared_weight_decode_eligible());
            assert_eq!(
                changed.automatic_q5_mmvq_weight_decode(true),
                ferrum_types::Q5MmvqWeightDecode::Baseline
            );
        };
        mutate(|p| p.family = Family::Extra);
        mutate(|p| p.arithmetic = Arithmetic::V1);
        mutate(|p| p.raw.algorithm = Algorithm::Mmq as u32);
        mutate(|p| p.raw.request.format = Format::Q4K as u32);
        mutate(|p| p.raw.request.layout = Layout::Channels as u32);
        mutate(|p| p.raw.request.rows = 4);
        mutate(|p| p.raw.ncols = 4);
        mutate(|p| p.raw.channels = 8);
        mutate(|p| p.raw.nwarps = 4);
        mutate(|p| p.raw.rows_per_block = 1);
        mutate(|p| p.raw.small_k = 1);
        mutate(|p| p.raw.fixup_bytes = 4);
        assert_eq!(
            plan.q5_mmvq_weight_decode(),
            ferrum_types::Q5MmvqWeightDecode::Baseline
        );
        assert_eq!(
            plan.clone()
                .with_q5_mmvq_weight_decode(ferrum_types::Q5MmvqWeightDecode::Baseline)
                .unwrap()
                .q5_mmvq_weight_decode(),
            ferrum_types::Q5MmvqWeightDecode::Baseline
        );
        assert!(matches!(
            plan.clone()
                .with_q5_mmvq_weight_decode(ferrum_types::Q5MmvqWeightDecode::Auto),
            Err(Error::InvalidArgument)
        ));
        #[cfg(not(feature = "cuda-q5-shared-weight-decode"))]
        assert!(matches!(
            plan.with_q5_mmvq_weight_decode(ferrum_types::Q5MmvqWeightDecode::SharedM8),
            Err(Error::Unavailable)
        ));
    }
    #[test]
    fn shared_m4_weight_decode_requires_the_original_four_warp_geometry() {
        for format in [Format::Q4K, Format::Q5K] {
            let mut raw = UpstreamLinearPlanV1::default();
            raw.algorithm = Algorithm::Mmvq as u32;
            raw.request.format = format as u32;
            raw.request.layout = Layout::Columns as u32;
            raw.request.rows = 4;
            raw.ncols = 4;
            raw.channels = 1;
            raw.nwarps = 4;
            raw.rows_per_block = 2;
            let plan = PreparedUpstreamLinear {
                raw,
                arithmetic: Arithmetic::MarkerV2,
                family: Family::Original,
                small_mmq_pack: SmallMmqPack::LegacyF32,
                small_mmvq_pack: SmallMmvqPack::F32Memset,
                q4_weight_decode: ferrum_types::Q4MmvqWeightDecode::Baseline,
                q5_weight_decode: ferrum_types::Q5MmvqWeightDecode::Baseline,
                mmvq_launch: MmvqLaunchKind::Ordered,
            };
            assert!(plan.shared_weight_decode_geometry(format, 4));
            assert!(!plan.q4_shared_weight_decode_eligible());
            assert!(!plan.q5_shared_weight_decode_eligible());
            let mutations: [fn(&mut PreparedUpstreamLinear); 12] = [
                |p| p.family = Family::Extra,
                |p| p.arithmetic = Arithmetic::V1,
                |p| p.raw.algorithm = Algorithm::Mmq as u32,
                |p| p.raw.request.format = Format::Iq4Xs as u32,
                |p| p.raw.request.layout = Layout::Channels as u32,
                |p| p.raw.request.rows = 8,
                |p| p.raw.ncols = 8,
                |p| p.raw.channels = 4,
                |p| p.raw.nwarps = 2,
                |p| p.raw.rows_per_block = 1,
                |p| p.raw.small_k = 1,
                |p| p.raw.fixup_bytes = 4,
            ];
            for mutate in mutations {
                let mut changed = plan.clone();
                mutate(&mut changed);
                assert!(!changed.shared_weight_decode_geometry(format, 4));
            }
            if format == Format::Q4K {
                let selected =
                    plan.with_q4_mmvq_weight_decode(ferrum_types::Q4MmvqWeightDecode::SharedM4M8);
                if cfg!(feature = "cuda-q4-shared-weight-decode") {
                    assert!(selected.is_ok());
                } else {
                    assert!(matches!(selected, Err(Error::Unavailable)));
                }
            } else {
                let selected =
                    plan.with_q5_mmvq_weight_decode(ferrum_types::Q5MmvqWeightDecode::SharedM4M8);
                if cfg!(feature = "cuda-q5-shared-weight-decode") {
                    assert!(selected.is_ok());
                } else {
                    assert!(matches!(selected, Err(Error::Unavailable)));
                }
            }
        }
    }
    #[test]
    fn span_validation_rejects_overflow_aliasing_and_short_tails() {
        let a = DeviceSpan {
            address: 4096,
            bytes: 32,
        };
        assert!(a.checked(32, 16).is_ok());
        assert!(a.checked(33, 16).is_err());
        assert!(DeviceSpan {
            address: u64::MAX - 15,
            bytes: 32
        }
        .checked(1, 1)
        .is_err());
        assert!(distinct(&[
            a,
            DeviceSpan {
                address: 4112,
                bytes: 16
            }
        ])
        .is_err());
        assert!(distinct(&[
            a,
            DeviceSpan {
                address: 4128,
                bytes: 16
            }
        ])
        .is_ok());
        assert_eq!(strided_bytes(3, 17, 23, 2).unwrap(), 126);
    }
    #[cfg(not(feature = "cuda-upstream-linear"))]
    #[test]
    fn missing_optional_artifact_is_an_explicit_error() {
        assert!(matches!(
            PreparedUpstreamLinear::new(
                Algorithm::Mmq,
                Arithmetic::V1,
                Format::Iq4Xs,
                Layout::Columns,
                8,
                5120,
                1024,
                Device {
                    architecture: 800,
                    multiprocessors: 1,
                    maximum_dynamic_shared_bytes: 65536
                }
            ),
            Err(Error::Unavailable)
        ));
    }
    #[test]
    fn prepared_operator_identity_keeps_the_two_artifact_families_distinct() {
        for family in [Family::Original, Family::Extra] {
            let plan = PreparedUpstreamLinear {
                raw: Default::default(),
                arithmetic: Arithmetic::MarkerV2,
                family,
                small_mmq_pack: SmallMmqPack::LegacyF32,
                small_mmvq_pack: SmallMmvqPack::F32Memset,
                q4_weight_decode: ferrum_types::Q4MmvqWeightDecode::Baseline,
                q5_weight_decode: ferrum_types::Q5MmvqWeightDecode::Baseline,
                mmvq_launch: MmvqLaunchKind::Ordered,
            };
            assert_eq!(plan.operator(), family.operator());
        }
        assert_ne!(Family::Original.operator(), Family::Extra.operator());
    }
    #[cfg(not(feature = "cuda-upstream-extra-linear"))]
    #[test]
    fn missing_extra_artifact_does_not_fall_back_to_original_symbols() {
        assert!(matches!(
            PreparedUpstreamLinear::new_extra(
                Algorithm::Mmq,
                Arithmetic::MarkerV2,
                ExtraFormat::Iq4Nl,
                Layout::Columns,
                8,
                5120,
                1024,
                Device {
                    architecture: 800,
                    multiprocessors: 1,
                    maximum_dynamic_shared_bytes: 65536
                }
            ),
            Err(Error::Unavailable)
        ));
    }
}
