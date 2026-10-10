//! Locked, independently typed F16-boundary Q6/D4 MMQ adapter. Preparation is cold;
//! enqueue methods neither allocate nor synchronize and never choose fallback.
use super::upstream_linear::{distinct, strided_bytes, Device, DeviceSpan, Error};
use ferrum_native_ops::upstream_q6_f16_linear::*;
use std::ffi::c_void;

#[cfg(feature = "cuda-upstream-q6-f32-linear")]
mod ffi;

#[derive(Debug, Clone)]
pub struct PreparedQ6F16Linear {
    raw: UpstreamQ6F16PlanV1,
}

impl PreparedQ6F16Linear {
    pub fn new(rows: u32, inputs: u32, outputs: u32, device: Device) -> Result<Self, Error> {
        let request = UpstreamQ6F16RequestV1::new(rows, inputs, outputs, device)?;
        #[cfg(not(feature = "cuda-upstream-q6-f32-linear"))]
        {
            let _ = request;
            Err(Error::Unavailable)
        }
        #[cfg(feature = "cuda-upstream-q6-f32-linear")]
        {
            let mut raw = UpstreamQ6F16PlanV1::default();
            Error::from_status(unsafe { ffi::ferrum_upstream_q6_f16_plan_v1(&request, &mut raw) })?;
            raw.validate_identity(&request)?;
            Ok(Self { raw })
        }
    }
    pub fn operator(&self) -> &'static str {
        UPSTREAM_Q6_F16_LINEAR_OPERATOR
    }
    pub fn geometry(&self) -> &UpstreamQ6F16PlanV1 {
        &self.raw
    }
    /// # Safety
    /// Exact immutable weights and the independent retained flag must remain
    /// live through completion. Success means queued, never GPU validation done.
    pub unsafe fn check_weights(
        &self,
        weights: DeviceSpan,
        flag: DeviceSpan,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        let w = weights.checked(self.raw.weight_bytes, 4)?;
        let f = flag.checked(4, 4)?;
        distinct(&[weights, flag])?;
        #[cfg(feature = "cuda-upstream-q6-f32-linear")]
        {
            Error::from_status(unsafe {
                ffi::ferrum_upstream_q6_f16_check_weights_v1(&self.raw, w, f, stream)
            })
        }
        #[cfg(not(feature = "cuda-upstream-q6-f32-linear"))]
        {
            let _ = (w, f, stream);
            Err(Error::Unavailable)
        }
    }
    /// # Safety
    /// All disjoint spans remain live on the prepared device until completion.
    pub unsafe fn pack(
        &self,
        input: DeviceSpan,
        stride: u32,
        converted: DeviceSpan,
        packed: DeviceSpan,
        rows: DeviceSpan,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        let r = &self.raw.request;
        let x = input.checked(strided_bytes(r.rows, r.inputs, stride, 2)?, 2)?;
        let c = converted.checked(self.raw.converted_bytes, 16)?;
        let q = packed.checked(self.raw.packed_bytes, 16)?;
        let f = rows.checked(u64::from(r.rows) * 4, 4)?;
        distinct(&[input, converted, packed, rows])?;
        if u64::from(stride) * u64::from(r.rows) > i32::MAX as u64 {
            return Err(Error::Extent);
        }
        #[cfg(feature = "cuda-upstream-q6-f32-linear")]
        {
            Error::from_status(unsafe {
                ffi::ferrum_upstream_q6_f16_pack_v1(&self.raw, x, stride, c, q, f, stream)
            })
        }
        #[cfg(not(feature = "cuda-upstream-q6-f32-linear"))]
        {
            let _ = (x, c, q, f, stream);
            Err(Error::Unavailable)
        }
    }
    /// # Safety
    /// Actual packed data uses this exact plan. Spans stay independently leased.
    pub unsafe fn dot(
        &self,
        weights: DeviceSpan,
        packed: DeviceSpan,
        raw: DeviceSpan,
        fixup: DeviceSpan,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        let w = weights.checked(self.raw.weight_bytes, 4)?;
        let q = packed.checked(self.raw.packed_bytes, 16)?;
        let y = raw.checked(self.raw.output_bytes, 4)?;
        let f = fixup.checked(self.raw.fixup_bytes, 4)?;
        if self.raw.fixup_bytes == 0 && fixup.bytes != 0 {
            return Err(Error::Span);
        }
        distinct(&[weights, packed, raw, fixup])?;
        #[cfg(feature = "cuda-upstream-q6-f32-linear")]
        {
            Error::from_status(unsafe {
                ffi::ferrum_upstream_q6_f16_dot_v1(&self.raw, w, q, y, f, stream)
            })
        }
        #[cfg(not(feature = "cuda-upstream-q6-f32-linear"))]
        {
            let _ = (w, q, y, f, stream);
            Err(Error::Unavailable)
        }
    }
    /// # Safety
    /// Flags have ordered producers; all ranges remain live through publication.
    pub unsafe fn cast(
        &self,
        raw: DeviceSpan,
        output: DeviceSpan,
        stride: u32,
        rows: DeviceSpan,
        flag: DeviceSpan,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        let r = &self.raw.request;
        let x = raw.checked(self.raw.output_bytes, 4)?;
        let y = output.checked(strided_bytes(r.rows, r.outputs, stride, 2)?, 2)?;
        let rf = rows.checked(u64::from(r.rows) * 4, 4)?;
        let wf = flag.checked(4, 4)?;
        distinct(&[raw, output, rows, flag])?;
        if u64::from(stride) * u64::from(r.rows) > i32::MAX as u64 {
            return Err(Error::Extent);
        }
        #[cfg(feature = "cuda-upstream-q6-f32-linear")]
        {
            Error::from_status(unsafe {
                ffi::ferrum_upstream_q6_f16_cast_v1(&self.raw, x, y, stride, rf, wf, stream)
            })
        }
        #[cfg(not(feature = "cuda-upstream-q6-f32-linear"))]
        {
            let _ = (x, y, rf, wf, stream);
            Err(Error::Unavailable)
        }
    }
}
