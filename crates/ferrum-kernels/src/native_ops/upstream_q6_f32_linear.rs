//! Locked, independently typed F32 Q6/D4 MMQ adapter. Preparation is cold;
//! enqueue methods neither allocate nor synchronize and never choose fallback.
use super::upstream_linear::{distinct, strided_bytes, Device, DeviceSpan, Error};
use ferrum_native_ops::upstream_q6_f32_linear::*;
use std::ffi::c_void;

#[cfg(feature = "cuda-upstream-q6-f32-linear")]
mod ffi;

#[derive(Debug, Clone)]
pub struct PreparedQ6F32Linear {
    raw: UpstreamQ6F32PlanV1,
    workspace: Q6F32Workspace,
}

/// Exact five independent, 16-byte-aligned ranges. Padding and the rounded
/// last MMQ loader footprint are already included in native `packed_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q6F32Workspace {
    pub converted: u64,
    pub packed: u64,
    pub raw: u64,
    pub fixup: u64,
    pub row_flags: u64,
    pub bytes: u64,
}
impl Q6F32Workspace {
    fn new(plan: &UpstreamQ6F32PlanV1) -> Result<Self, Error> {
        let mut cursor = 0u64;
        let mut take = |bytes: u64| -> Result<u64, Error> {
            cursor = cursor.checked_add(15).ok_or(Error::Extent)? & !15;
            let start = cursor;
            cursor = cursor.checked_add(bytes).ok_or(Error::Extent)?;
            Ok(start)
        };
        let converted = take(plan.converted_bytes)?;
        let packed = take(plan.packed_bytes)?;
        let raw = take(plan.output_bytes)?;
        let fixup = take(plan.fixup_bytes)?;
        let row_flags = take(u64::from(plan.request.rows) * 4)?;
        Ok(Self {
            converted,
            packed,
            raw,
            fixup,
            row_flags,
            bytes: cursor,
        })
    }
}
impl PreparedQ6F32Linear {
    pub fn new(rows: u32, inputs: u32, outputs: u32, device: Device) -> Result<Self, Error> {
        let request = UpstreamQ6F32RequestV1::new(rows, inputs, outputs, device)?;
        #[cfg(not(feature = "cuda-upstream-q6-f32-linear"))]
        {
            let _ = request;
            Err(Error::Unavailable)
        }
        #[cfg(feature = "cuda-upstream-q6-f32-linear")]
        {
            let mut raw = UpstreamQ6F32PlanV1::default();
            Error::from_status(unsafe { ffi::ferrum_upstream_q6_f32_plan_v1(&request, &mut raw) })?;
            raw.validate_identity(&request)?;
            Ok(Self {
                workspace: Q6F32Workspace::new(&raw)?,
                raw,
            })
        }
    }
    pub fn operator(&self) -> &'static str {
        UPSTREAM_Q6_F32_LINEAR_OPERATOR
    }
    pub fn geometry(&self) -> &UpstreamQ6F32PlanV1 {
        &self.raw
    }
    pub fn workspace(&self) -> Q6F32Workspace {
        self.workspace
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.workspace.bytes
    }
    fn piece(&self, workspace: DeviceSpan, offset: u64, bytes: u64) -> Result<DeviceSpan, Error> {
        workspace.checked(self.workspace.bytes, 16)?;
        if offset.checked_add(bytes).ok_or(Error::Extent)? > self.workspace.bytes {
            return Err(Error::Span);
        }
        Ok(DeviceSpan {
            address: workspace.address.checked_add(offset).ok_or(Error::Span)?,
            bytes,
        })
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
                ffi::ferrum_upstream_q6_f32_check_weights_v1(&self.raw, w, f, stream)
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
        let x = input.checked(strided_bytes(r.rows, r.inputs, stride, 4)?, 4)?;
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
                ffi::ferrum_upstream_q6_f32_pack_v1(&self.raw, x, stride, c, q, f, stream)
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
        distinct(&[weights, packed, raw, fixup])?;
        #[cfg(feature = "cuda-upstream-q6-f32-linear")]
        {
            Error::from_status(unsafe {
                ffi::ferrum_upstream_q6_f32_dot_v1(&self.raw, w, q, y, f, stream)
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
    pub unsafe fn publish(
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
        let y = output.checked(strided_bytes(r.rows, r.outputs, stride, 4)?, 4)?;
        let rf = rows.checked(u64::from(r.rows) * 4, 4)?;
        let wf = flag.checked(4, 4)?;
        distinct(&[raw, output, rows, flag])?;
        if u64::from(stride) * u64::from(r.rows) > i32::MAX as u64 {
            return Err(Error::Extent);
        }
        #[cfg(feature = "cuda-upstream-q6-f32-linear")]
        {
            Error::from_status(unsafe {
                ffi::ferrum_upstream_q6_f32_publish_v1(&self.raw, x, y, stride, rf, wf, stream)
            })
        }
        #[cfg(not(feature = "cuda-upstream-q6-f32-linear"))]
        {
            let _ = (x, y, rf, wf, stream);
            Err(Error::Unavailable)
        }
    }
    /// # Safety
    /// Spans belong to the prepared CUDA device, are independently leased until
    /// completion, and flag publication is ordered before this stream. Output
    /// starts at this leaf's offset; this adapter never adds that offset again.
    pub unsafe fn launch(
        &self,
        input: DeviceSpan,
        input_stride: u32,
        weights: DeviceSpan,
        output: DeviceSpan,
        output_stride: u32,
        workspace: DeviceSpan,
        flag: DeviceSpan,
        stream: *mut c_void,
    ) -> Result<(), Error> {
        let r = &self.raw.request;
        let x = input.checked(strided_bytes(r.rows, r.inputs, input_stride, 4)?, 4)?;
        let w = weights.checked(self.raw.weight_bytes, 4)?;
        let y = output.checked(strided_bytes(r.rows, r.outputs, output_stride, 4)?, 4)?;
        let f = flag.checked(4, 4)?;
        workspace.checked(self.workspace.bytes, 16)?;
        if u64::from(input_stride) * u64::from(r.rows) > i32::MAX as u64
            || u64::from(output_stride) * u64::from(r.rows) > i32::MAX as u64
        {
            return Err(Error::Extent);
        }
        // Validate all cross-stage aliasing before the first device write.
        distinct(&[input, weights, output, workspace, flag])?;
        let c = self
            .piece(
                workspace,
                self.workspace.converted,
                self.raw.converted_bytes,
            )?
            .checked(self.raw.converted_bytes, 16)?;
        let q = self
            .piece(workspace, self.workspace.packed, self.raw.packed_bytes)?
            .checked(self.raw.packed_bytes, 16)?;
        let raw = self
            .piece(workspace, self.workspace.raw, self.raw.output_bytes)?
            .checked(self.raw.output_bytes, 16)?;
        let fix = self
            .piece(workspace, self.workspace.fixup, self.raw.fixup_bytes)?
            .checked(self.raw.fixup_bytes, 16)?;
        let rows = self
            .piece(workspace, self.workspace.row_flags, u64::from(r.rows) * 4)?
            .checked(u64::from(r.rows) * 4, 4)?;
        #[cfg(feature = "cuda-upstream-q6-f32-linear")]
        {
            Error::from_status(unsafe {
                ffi::ferrum_upstream_q6_f32_pack_v1(&self.raw, x, input_stride, c, q, rows, stream)
            })?;
            Error::from_status(unsafe {
                ffi::ferrum_upstream_q6_f32_dot_v1(&self.raw, w, q, raw, fix, stream)
            })?;
            Error::from_status(unsafe {
                ffi::ferrum_upstream_q6_f32_publish_v1(
                    &self.raw,
                    raw,
                    y,
                    output_stride,
                    rows,
                    f,
                    stream,
                )
            })
        }
        #[cfg(not(feature = "cuda-upstream-q6-f32-linear"))]
        {
            let _ = (x, w, y, f, c, q, raw, fix, rows, stream);
            Err(Error::Unavailable)
        }
    }
}

#[cfg(test)]
mod tests;
