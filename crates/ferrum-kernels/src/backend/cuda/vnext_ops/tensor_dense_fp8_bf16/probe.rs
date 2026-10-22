//! Explicit diagnostic capture of the actual quantization intermediates.
//! No device allocation or alternate arithmetic path. The caller reserves two
//! Output backings before admission; the original command synchronously reads
//! valid scratch extents on its stream and publishes only a complete receipt.

use super::*;
use cudarc::driver::CudaStream;
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, ManagedHostBuffer, ManagedHostBytes};

pub const QUANTIZATION_PROBE_PROFILE_ID: &str =
    "diagnostic.cuda.cutlass_fp8.quantization_readback.v1";

pub struct CudaCutlassFp8Probe {
    host: HostMemoryAccount,
    inner: Mutex<Inner>,
}
struct Inner {
    next: u64,
    slot: Slot,
}
enum Slot {
    Idle,
    Armed {
        generation: u64,
        rows: u32,
        inputs: u32,
        buffers: Buffers,
    },
    InFlight(u64),
    Ready(u64, CudaCutlassFp8ProbeReceipt),
    Failed(u64),
}
struct Buffers {
    activations: ManagedHostBuffer,
    scales: ManagedHostBuffer,
}

/// One armed diagnostic invocation. Dropping it before encode cancels its
/// reservation. Dropping during native work never releases the command's owners.
pub struct CudaCutlassFp8ProbeTicket {
    probe: Arc<CudaCutlassFp8Probe>,
    generation: u64,
}

pub struct CudaCutlassFp8ProbeReceipt {
    rows: u32,
    inputs: u32,
    activations: ManagedHostBytes,
    scales: ManagedHostBytes,
}
impl CudaCutlassFp8ProbeReceipt {
    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn inputs(&self) -> u32 {
        self.inputs
    }
    pub fn activation_bytes(&self) -> &ManagedHostBytes {
        &self.activations
    }
    /// Exact little-endian F32 bit patterns, one scale per row.
    pub fn row_scale_bytes(&self) -> &ManagedHostBytes {
        &self.scales
    }
}

impl CudaCutlassFp8Probe {
    pub fn new(host: HostMemoryAccount) -> Arc<Self> {
        Arc::new(Self {
            host,
            inner: Mutex::new(Inner {
                next: 1,
                slot: Slot::Idle,
            }),
        })
    }

    pub fn arm(
        self: &Arc<Self>,
        rows: u32,
        inputs: u32,
    ) -> Result<CudaCutlassFp8ProbeTicket, CudaDeviceRuntimeError> {
        if rows == 0 || inputs == 0 {
            return Err(error("FP8 probe geometry must be nonzero"));
        }
        let bytes = |element: u64| {
            u64::from(rows)
                .checked_mul(element)
                .filter(|n| *n <= isize::MAX as u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| error("FP8 probe byte extent overflows"))
        };
        let mut inner = self.inner.lock().map_err(|_| error("FP8 probe poisoned"))?;
        if !matches!(inner.slot, Slot::Idle) {
            return Err(error("FP8 probe already armed, running or unread"));
        }
        let generation = inner.next;
        let next = generation
            .checked_add(1)
            .ok_or_else(|| error("FP8 probe generation exhausted"))?;
        let claims = self
            .host
            .reserve_many(&[
                (HostMemoryClass::Output, bytes(u64::from(inputs))?),
                (HostMemoryClass::Output, bytes(4)?),
            ])
            .map_err(|e| error(e.to_string()))?;
        let mut claims = claims.into_iter();
        let activations = claims
            .next()
            .ok_or_else(|| error("FP8 probe activation claim missing"))?
            .allocate_zeroed()
            .map_err(|e| error(e.to_string()))?;
        let scales = claims
            .next()
            .ok_or_else(|| error("FP8 probe scale claim missing"))?
            .allocate_zeroed()
            .map_err(|e| error(e.to_string()))?;
        inner.next = next;
        inner.slot = Slot::Armed {
            generation,
            rows,
            inputs,
            buffers: Buffers {
                activations,
                scales,
            },
        };
        Ok(CudaCutlassFp8ProbeTicket {
            probe: Arc::clone(self),
            generation,
        })
    }

    pub(super) fn claim(
        self: &Arc<Self>,
        rows: u32,
        inputs: u32,
    ) -> Result<Arc<Capture>, CudaDeviceRuntimeError> {
        let mut inner = self.inner.lock().map_err(|_| error("FP8 probe poisoned"))?;
        let generation = match &inner.slot {
            Slot::Armed {
                generation,
                rows: r,
                inputs: k,
                ..
            } if (*r, *k) == (rows, inputs) => *generation,
            _ => {
                return Err(error(
                    "FP8 probe must be armed for this exact invocation geometry",
                ))
            }
        };
        let Slot::Armed { buffers, .. } =
            std::mem::replace(&mut inner.slot, Slot::InFlight(generation))
        else {
            unreachable!()
        };
        Ok(Arc::new(Capture {
            probe: Arc::clone(self),
            generation,
            rows,
            inputs,
            buffers: Mutex::new(Some(buffers)),
        }))
    }
}

impl CudaCutlassFp8ProbeTicket {
    /// A compute/transfer failure cannot produce a partial success receipt.
    pub fn take(&self) -> Result<CudaCutlassFp8ProbeReceipt, CudaDeviceRuntimeError> {
        let mut inner = self
            .probe
            .inner
            .lock()
            .map_err(|_| error("FP8 probe poisoned"))?;
        match &inner.slot {
            Slot::Ready(generation, _) if *generation == self.generation => {}
            Slot::Failed(generation) if *generation == self.generation => {
                return Err(error(
                    "FP8 probe execution did not publish a complete capture",
                ))
            }
            _ => return Err(error("FP8 probe is not complete for this ticket")),
        }
        let Slot::Ready(_, receipt) = std::mem::replace(&mut inner.slot, Slot::Idle) else {
            unreachable!()
        };
        Ok(receipt)
    }
}
impl Drop for CudaCutlassFp8ProbeTicket {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.probe.inner.lock() {
            if matches!(&inner.slot, Slot::Armed { generation, .. } if *generation == self.generation)
            {
                inner.slot = Slot::Idle;
            }
        }
    }
}

pub(super) struct Capture {
    probe: Arc<CudaCutlassFp8Probe>,
    generation: u64,
    rows: u32,
    inputs: u32,
    buffers: Mutex<Option<Buffers>>,
}
impl Capture {
    pub(super) fn read(
        &self,
        stream: &CudaStream,
        scratch: &CudaBufferRegion,
        plan: &PreparedCutlassFp8,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| error("FP8 capture poisoned"))?;
        let destination = buffers
            .as_mut()
            .ok_or_else(|| error("FP8 capture cannot be executed twice"))?;
        let p = plan.plan();
        if (p.request.rows, p.request.inputs) != (self.rows, self.inputs)
            || destination.activations.len() as u64 != p.quantized_activation_bytes
            || destination.scales.len() as u64 != p.activation_scale_bytes
        {
            return Err(error("FP8 capture plan/backing geometry differs"));
        }
        let a = scratch.subregion(0, p.quantized_activation_bytes)?;
        let s = scratch.subregion(p.activation_scale_offset, p.activation_scale_bytes)?;
        stream
            .synchronize()
            .map_err(|e| CudaDeviceRuntimeError::driver("FP8 diagnostic stream completion", e))?;
        // Ordinary accounted Output backing, synchronous D2H: no async borrowed
        // host pointer, pageable-memory pinning claim, or hidden payload Vec.
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(
                destination.activations.as_mut(),
                a.device_ptr(),
            )
        }
        .map_err(|e| CudaDeviceRuntimeError::driver("FP8 diagnostic activation readback", e))?;
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(destination.scales.as_mut(), s.device_ptr())
        }
        .map_err(|e| CudaDeviceRuntimeError::driver("FP8 diagnostic scale readback", e))?;
        let mut inner = self
            .probe
            .inner
            .lock()
            .map_err(|_| error("FP8 probe poisoned"))?;
        if !matches!(inner.slot, Slot::InFlight(generation) if generation == self.generation) {
            return Err(error("FP8 diagnostic capture generation changed"));
        }
        let buffers = buffers
            .take()
            .ok_or_else(|| error("FP8 diagnostic output missing"))?;
        inner.slot = Slot::Ready(
            self.generation,
            CudaCutlassFp8ProbeReceipt {
                rows: self.rows,
                inputs: self.inputs,
                activations: buffers.activations.freeze(),
                scales: buffers.scales.freeze(),
            },
        );
        Ok(())
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.probe.inner.lock() {
            if matches!(inner.slot, Slot::InFlight(generation) if generation == self.generation) {
                inner.slot = Slot::Failed(self.generation);
            }
        }
    }
}
fn error(message: impl Into<String>) -> CudaDeviceRuntimeError {
    CudaDeviceRuntimeError::contract(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};
    fn host(bytes: usize) -> HostMemoryAccount {
        HostMemoryAccount::new(
            HostMemoryLimits::new(
                bytes,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 0,
                    output_bytes: bytes,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        )
    }
    #[test]
    fn reservations_are_atomic_and_unsubmitted_ticket_releases_them() {
        let small = host(23);
        let probe = CudaCutlassFp8Probe::new(small.clone());
        assert!(probe.arm(2, 8).is_err());
        assert_eq!(small.usage().unwrap().committed_bytes(), 0);
        let account = host(24);
        let probe = CudaCutlassFp8Probe::new(account.clone());
        let ticket = probe.arm(2, 8).unwrap();
        assert_eq!(account.usage().unwrap().committed_bytes(), 24);
        assert!(probe.arm(1, 1).is_err());
        assert!(ticket.take().is_err());
        drop(ticket);
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
        assert!(probe.arm(2, 8).is_ok());
    }
    #[test]
    fn wrong_geometry_does_not_consume_reservation_and_failed_capture_never_publishes() {
        let account = host(24);
        let probe = CudaCutlassFp8Probe::new(account.clone());
        let ticket = probe.arm(2, 8).unwrap();
        assert!(probe.claim(1, 16).is_err());
        assert_eq!(account.usage().unwrap().committed_bytes(), 24);
        let capture = probe.claim(2, 8).unwrap();
        assert!(ticket.take().is_err());
        assert!(probe.claim(2, 8).is_err());
        drop(capture);
        assert!(ticket.take().is_err());
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}
