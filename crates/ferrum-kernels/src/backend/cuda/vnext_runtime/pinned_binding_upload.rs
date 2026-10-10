//! Submission-private pinned sources for the unchanged sparse binding copies.
//! Only host storage is pooled. No destination, authority, payload or graph is cached.

use super::*;

// One idle OR loaned allocation per stream. Oversized or concurrent submissions
// retain the pageable path; they never wait for or refill an in-flight loan.
const MAX_PINNED_UPLOAD_BYTES: usize = 1024 * 1024;

#[derive(Default)]
struct PoolState {
    loaned: bool,
    idle: Option<PinnedHostSlice<u8>>,
}

#[derive(Default)]
pub(super) struct PinnedUploadPool(Mutex<PoolState>);

impl PinnedUploadPool {
    fn borrow(
        self: &Arc<Self>,
        context: &Arc<CudaContext>,
        bytes: usize,
    ) -> Result<Option<(PinnedUploadLease, u64, bool)>, CudaDeviceRuntimeError> {
        if bytes == 0 || bytes > MAX_PINNED_UPLOAD_BYTES {
            return Ok(None);
        }
        let idle = {
            let Ok(mut pool) = self.0.lock() else {
                return Ok(None);
            };
            if pool.loaned {
                return Ok(None);
            }
            pool.loaned = true;
            pool.idle.take()
        };
        let mut lease = PinnedUploadLease {
            pool: Arc::clone(self),
            storage: idle,
            base: 0,
            submitted: false,
            reusable: false,
        };
        let reused = lease
            .storage
            .as_ref()
            .is_some_and(|host| host.len() >= bytes);
        let allocated_bytes = if reused {
            0
        } else {
            // Free a completed, undersized idle allocation before replacing it.
            drop(lease.storage.take());
            let capacity = bytes.next_power_of_two();
            let mut host = unsafe { context.alloc_pinned::<u8>(capacity) }.map_err(|error| {
                CudaDeviceRuntimeError::driver("binding pinned allocation", error)
            })?;
            host.as_mut_slice()
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("binding pinned initialization", error)
                })?
                .fill(0);
            lease.storage = Some(host);
            capacity as u64
        };
        Ok(Some((lease, allocated_bytes, reused)))
    }
}

struct PinnedUploadLease {
    pool: Arc<PinnedUploadPool>,
    storage: Option<PinnedHostSlice<u8>>,
    base: usize,
    submitted: bool,
    reusable: bool,
}

impl PinnedUploadLease {
    fn completed(&mut self, healthy: bool) {
        self.submitted = false;
        self.reusable = healthy;
    }
}

impl Drop for PinnedUploadLease {
    fn drop(&mut self) {
        let storage = self.storage.take();
        if self.submitted {
            // An ordinary Drop is not a completion receipt. Keep both memory
            // and the pool's outstanding-loan bit quarantined indefinitely.
            // The normal quarantine path calls completed(false) after a real
            // successful stream synchronization instead of taking this path.
            std::mem::forget(storage);
            return;
        }
        let mut retired = storage;
        if let Ok(mut pool) = self.pool.0.lock() {
            pool.loaned = false;
            if self.reusable {
                std::mem::swap(&mut pool.idle, &mut retired);
            }
        }
        // CUDA destruction never happens with the pool mutex held.
        drop(retired);
    }
}

pub(super) struct ProgramBindingPrelude {
    pub(super) shapes: Vec<(usize, usize, usize)>,
    pub(super) counters: Arc<ProgramBindingUploadCounters>,
    pub(super) plan: DeviceProgramBindingUploadSnapshot,
}

impl ProgramBindingPrelude {
    fn validate(
        &self,
        regions: &[CudaBufferRegion],
        payloads: &[Box<[u8]>],
    ) -> Result<(), CudaDeviceRuntimeError> {
        if regions.len() != payloads.len() || regions.len() != self.shapes.len() {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA sparse program binding transfer storage differs from its shape",
            ));
        }
        for ((region, payload), &(pitch, width, rows)) in
            regions.iter().zip(payloads).zip(&self.shapes)
        {
            if rows == 0
                || width == 0
                || width.checked_mul(rows) != Some(payload.len())
                || pitch
                    .checked_mul(rows - 1)
                    .and_then(|n| n.checked_add(width))
                    .is_none_or(|span| span as u64 > region.length_bytes)
            {
                return Err(CudaDeviceRuntimeError::contract(
                    "CUDA sparse program binding source or destination span changed",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn enqueue(
        &self,
        stream: &CudaStream,
        regions: &[CudaBufferRegion],
        payloads: &[Box<[u8]>],
        pinned: Option<PinnedSources<'_>>,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let mut attempt = self.counters.attempt(self.plan);
        self.validate(regions, payloads)?;
        for (index, ((region, payload), &(pitch, width, rows))) in
            regions.iter().zip(payloads).zip(&self.shapes).enumerate()
        {
            let source = match &pinned {
                Some(sources) => sources.pointer(index)?,
                None => payload.as_ptr(),
            };
            if rows == 1 {
                unsafe {
                    cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                        region.device_ptr,
                        source.cast(),
                        width,
                        stream.cu_stream(),
                    )
                }
                .result()
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("sparse program binding upload", error)
                })?;
            } else {
                let copy = cudarc::driver::sys::CUDA_MEMCPY2D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: source.cast(),
                    srcDevice: 0,
                    srcArray: std::ptr::null_mut(),
                    srcPitch: width,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: region.device_ptr,
                    dstArray: std::ptr::null_mut(),
                    dstPitch: pitch,
                    WidthInBytes: width,
                    Height: rows,
                };
                unsafe { cudarc::driver::sys::cuMemcpy2DAsync_v2(&copy, stream.cu_stream()) }
                    .result()
                    .map_err(|error| {
                        CudaDeviceRuntimeError::driver(
                            "strided sparse program binding upload",
                            error,
                        )
                    })?;
            }
            attempt.copied(payload.len() as u64, rows > 1);
        }
        attempt.succeeded();
        Ok(())
    }
}

pub(super) struct PinnedSources<'a> {
    base: usize,
    offsets: &'a [usize],
}

impl PinnedSources<'_> {
    fn pointer(&self, index: usize) -> Result<*const u8, CudaDeviceRuntimeError> {
        let offset = *self.offsets.get(index).ok_or_else(|| {
            CudaDeviceRuntimeError::contract("missing submission-private pinned source")
        })?;
        self.base
            .checked_add(offset)
            .map(|address| address as *const u8)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("pinned source address overflows"))
    }
}

/// This object is never stored in a shared command/executable. Exactly one
/// submit owns it, then moves it into that submission's fence or quarantine.
#[derive(Default)]
pub(super) struct PinnedUploadSubmission {
    lease: Option<PinnedUploadLease>,
    command_ranges: Vec<(usize, Range<usize>)>,
    offsets: Vec<usize>,
}

impl PinnedUploadSubmission {
    pub(super) fn prepare(
        context: &Arc<CudaContext>,
        pool: &Arc<PinnedUploadPool>,
        stream_quiescent: bool,
        commands: &[CudaDeviceCommand],
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let mut uploads = Vec::new();
        let mut bytes = 0_usize;
        for (index, command) in commands.iter().enumerate() {
            let Some(executable) = &command.executable else {
                continue;
            };
            let action = executable.enqueue.lock().map_err(|_| {
                CudaDeviceRuntimeError::contract("poisoned pinned upload declaration")
            })?;
            let CudaEnqueueAction::ProgramBindingPrelude(prelude) = &*action else {
                continue;
            };
            prelude.validate(&executable.regions, &executable.host_storage)?;
            let size = executable
                .host_storage
                .iter()
                .try_fold(0_usize, |total, payload| total.checked_add(payload.len()))
                .ok_or_else(|| CudaDeviceRuntimeError::contract("pinned upload size overflows"))?;
            bytes = bytes.checked_add(size).ok_or_else(|| {
                CudaDeviceRuntimeError::contract("pinned submission size overflows")
            })?;
            uploads.push((
                index,
                &executable.host_storage,
                Arc::clone(&prelude.counters),
                size,
            ));
        }
        if uploads.is_empty() {
            return Ok(Self::default());
        }
        let borrowed = if stream_quiescent {
            pool.borrow(context, bytes)?
        } else {
            None
        };
        let Some((mut lease, allocated_bytes, reused)) = borrowed else {
            for (_, _, counters, _) in uploads {
                counters.pinned_fallback();
            }
            return Ok(Self::default());
        };
        let host = lease
            .storage
            .as_mut()
            .ok_or_else(|| CudaDeviceRuntimeError::contract("missing pinned upload allocation"))?;
        let destination = host
            .as_mut_slice()
            .map_err(|error| CudaDeviceRuntimeError::driver("binding pinned staging", error))?;
        let mut command_ranges = Vec::with_capacity(uploads.len());
        let mut offsets = Vec::new();
        let mut cursor = 0_usize;
        for (index, payloads, _, _) in &uploads {
            let start = offsets.len();
            for payload in *payloads {
                let end = cursor.checked_add(payload.len()).ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("pinned upload staging size overflows")
                })?;
                let target = destination.get_mut(cursor..end).ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("pinned upload staging exceeds its lease")
                })?;
                target.copy_from_slice(payload);
                offsets.push(cursor);
                cursor = end;
            }
            command_ranges.push((*index, start..offsets.len()));
        }
        lease.base = host.as_ptr().map_err(|error| {
            CudaDeviceRuntimeError::driver("binding pinned source address", error)
        })? as usize;
        for (position, (_, _, counters, size)) in uploads.into_iter().enumerate() {
            counters.pinned_staged(
                if position == 0 { allocated_bytes } else { 0 },
                reused,
                size as u64,
            );
        }
        Ok(Self {
            lease: Some(lease),
            command_ranges,
            offsets,
        })
    }

    pub(super) fn sources_for(&mut self, command_index: usize) -> Option<PinnedSources<'_>> {
        let range = &self
            .command_ranges
            .iter()
            .find(|(index, _)| *index == command_index)?
            .1;
        let lease = self.lease.as_mut()?;
        // Mark conservatively before the first possible driver call. Any
        // partial error/unwind retains this submit's sources until real drain.
        lease.submitted = true;
        Some(PinnedSources {
            base: lease.base,
            offsets: &self.offsets[range.clone()],
        })
    }

    pub(super) fn completed(&mut self, healthy: bool) {
        if let Some(mut lease) = self.lease.take() {
            lease.completed(healthy);
            drop(lease);
        }
    }
}

#[cfg(test)]
mod tests;
