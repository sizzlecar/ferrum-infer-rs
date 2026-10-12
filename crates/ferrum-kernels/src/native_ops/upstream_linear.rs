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
    #[cfg(any(feature = "cuda-upstream-linear", test))]
    Q8,
    #[cfg(any(feature = "cuda-upstream-extra-linear", test))]
    Extra,
}
impl Family {
    const fn operator(self) -> &'static str {
        match self {
            #[cfg(any(feature = "cuda-upstream-linear", test))]
            Self::Original => UPSTREAM_LINEAR_OPERATOR,
            #[cfg(any(feature = "cuda-upstream-linear", test))]
            Self::Q8 => UPSTREAM_LINEAR_OPERATOR,
            #[cfg(any(feature = "cuda-upstream-extra-linear", test))]
            Self::Extra => UPSTREAM_EXTRA_LINEAR_OPERATOR,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreparedUpstreamLinear {
    raw: UpstreamLinearPlanV1,
    arithmetic: Arithmetic,
    family: Family,
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
        if format == Format::Q8_0
            && (algorithm != Algorithm::Mmq || arithmetic != Arithmetic::MarkerV2)
        {
            return Err(Error::UnsupportedGeometry);
        }
        #[cfg(not(feature = "cuda-upstream-linear"))]
        {
            let _ = (algorithm, arithmetic, request);
            Err(Error::Unavailable)
        }
        #[cfg(feature = "cuda-upstream-linear")]
        {
            let mut raw = UpstreamLinearPlanV1::default();
            let (family, call) = match (format, algorithm) {
                (Format::Q8_0, Algorithm::Mmq) => {
                    (Family::Q8, q8_ffi::ferrum_upstream_q8_mmq_plan_v1 as PlanFn)
                }
                (_, Algorithm::Mmq) if rows > 32 => (
                    Family::Original,
                    ffi::ferrum_upstream_mmq_prefill_plan_v1 as PlanFn,
                ),
                (_, Algorithm::Mmq) => {
                    (Family::Original, ffi::ferrum_upstream_mmq_plan_v1 as PlanFn)
                }
                (_, Algorithm::Mmvq) => (
                    Family::Original,
                    ffi::ferrum_upstream_mmvq_plan_v1 as PlanFn,
                ),
            };
            // Native preparation checks the current device and configures MMQ
            // dynamic shared memory once, outside any CUDA graph capture.
            Error::from_status(unsafe { call(&request, &mut raw) })?;
            raw.validate_identity(&request, algorithm)?;
            Ok(Self {
                raw,
                arithmetic,
                family,
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
            })
        }
    }
    pub fn geometry(&self) -> &UpstreamLinearPlanV1 {
        &self.raw
    }
    pub fn arithmetic(&self) -> Arithmetic {
        self.arithmetic
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
            if self.family == Family::Q8 {
                return Error::from_status(unsafe {
                    q8_ffi::ferrum_upstream_q8_mmq_check_weights_v2(&self.raw, w, f, stream)
                });
            }
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
            if self.family == Family::Q8 {
                return Error::from_status(unsafe {
                    q8_ffi::ferrum_upstream_q8_mmq_pack_v2(
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
            if self.family == Family::Q8 {
                return Error::from_status(unsafe {
                    q8_ffi::ferrum_upstream_q8_mmq_dot_v1(&self.raw, w, q, o, f, stream)
                });
            }
            let api = dispatch::api(self.family, self.raw.algorithm)?;
            Error::from_status(unsafe { (api.dot)(&self.raw, w, q, o, f, stream) })
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
            if self.family == Family::Q8 {
                return Error::from_status(unsafe {
                    q8_ffi::ferrum_upstream_q8_mmq_cast_v2(&self.raw, i, o, stride, rp, wp, stream)
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
type PlanFn =
    unsafe extern "C" fn(*const UpstreamLinearRequestV1, *mut UpstreamLinearPlanV1) -> i32;

// Q8 uses one independent prefix in the existing artifact. No old-format
// symbol, V1 arithmetic export, or MMVQ entry can be substituted for it.
#[cfg(feature = "cuda-upstream-linear")]
mod q8_ffi {
    use super::*;

    unsafe extern "C" {
        pub(super) fn ferrum_upstream_q8_mmq_plan_v1(
            request: *const UpstreamLinearRequestV1,
            plan: *mut UpstreamLinearPlanV1,
        ) -> i32;
        pub(super) fn ferrum_upstream_q8_mmq_pack_v2(
            plan: *const UpstreamLinearPlanV1,
            input: *const c_void,
            stride: u32,
            converted: *mut c_void,
            packed: *mut c_void,
            row_flags: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub(super) fn ferrum_upstream_q8_mmq_dot_v1(
            plan: *const UpstreamLinearPlanV1,
            weights: *const c_void,
            packed: *const c_void,
            output: *mut c_void,
            fixup: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub(super) fn ferrum_upstream_q8_mmq_check_weights_v2(
            plan: *const UpstreamLinearPlanV1,
            weights: *const c_void,
            weight_flag: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub(super) fn ferrum_upstream_q8_mmq_cast_v2(
            plan: *const UpstreamLinearPlanV1,
            input: *const c_void,
            output: *mut c_void,
            stride: u32,
            row_flags: *const c_void,
            weight_flag: *const c_void,
            stream: *mut c_void,
        ) -> i32;
    }
}

#[cfg(feature = "cuda-upstream-linear")]
mod dispatch;
#[cfg(feature = "cuda-upstream-extra-linear")]
mod extra_ffi;
#[cfg(feature = "cuda-upstream-linear")]
mod ffi;
#[cfg(test)]
mod tests {
    use super::*;
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
        for family in [Family::Original, Family::Q8, Family::Extra] {
            let plan = PreparedUpstreamLinear {
                raw: Default::default(),
                arithmetic: Arithmetic::MarkerV2,
                family,
            };
            assert_eq!(plan.operator(), family.operator());
        }
        assert_ne!(Family::Original.operator(), Family::Extra.operator());
        assert_eq!(Family::Q8.operator(), Family::Original.operator());
    }
    #[test]
    fn q8_constructor_rejects_unsupported_domain_before_native_dispatch() {
        let device = Device {
            architecture: 800,
            multiprocessors: 1,
            maximum_dynamic_shared_bytes: 65536,
        };
        for (algorithm, arithmetic, layout, rows) in [
            (Algorithm::Mmvq, Arithmetic::MarkerV2, Layout::Columns, 8),
            (Algorithm::Mmq, Arithmetic::V1, Layout::Columns, 8),
            (Algorithm::Mmq, Arithmetic::MarkerV2, Layout::Channels, 8),
            (Algorithm::Mmq, Arithmetic::MarkerV2, Layout::Columns, 1),
            (Algorithm::Mmq, Arithmetic::MarkerV2, Layout::Columns, 33),
        ] {
            assert!(matches!(
                PreparedUpstreamLinear::new(
                    algorithm,
                    arithmetic,
                    Format::Q8_0,
                    layout,
                    rows,
                    5120,
                    48,
                    device,
                ),
                Err(Error::UnsupportedGeometry)
            ));
        }
    }
    #[test]
    fn q8_projection_subspan_extent_keeps_real_output_stride() {
        for rows in [8, 32] {
            let required = strided_bytes(rows, 48, 16480, 2).unwrap();
            let base = 1 << 20;
            for offset in [16384, 16432] {
                let output = DeviceSpan {
                    address: base + offset * 2,
                    bytes: required,
                };
                assert!(output.checked(required, 2).is_ok());
                assert!(DeviceSpan {
                    bytes: required - 2,
                    ..output
                }
                .checked(required, 2)
                .is_err());
                // Scratch in a row gap is still within the checked output
                // envelope and cannot be substituted for disjoint storage.
                assert!(distinct(&[
                    output,
                    DeviceSpan {
                        address: output.address + 96,
                        bytes: 16
                    },
                ])
                .is_err());
            }
        }
        assert_eq!(strided_bytes(8, 48, 47, 2), Err(Error::Span));
    }
    #[cfg(not(feature = "cuda-upstream-linear"))]
    #[test]
    fn q8_missing_artifact_is_explicit_without_original_symbol_fallback() {
        assert!(matches!(
            PreparedUpstreamLinear::new(
                Algorithm::Mmq,
                Arithmetic::MarkerV2,
                Format::Q8_0,
                Layout::Columns,
                8,
                5120,
                48,
                Device {
                    architecture: 800,
                    multiprocessors: 1,
                    maximum_dynamic_shared_bytes: 65536,
                },
            ),
            Err(Error::Unavailable)
        ));
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
