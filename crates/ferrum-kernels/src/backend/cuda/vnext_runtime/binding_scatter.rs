//! Compact transport after fresh Core and typed-arena validation. Scratch is
//! runtime-owned, never a forged Core allocation or a reusable compute resource.

use super::{CudaDeviceRuntimeError, CudaProgramBindingTransfer};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

type Result<T> = std::result::Result<T, CudaDeviceRuntimeError>;
const DESCRIPTOR_BYTES: usize = 40;
const FUNCTION: &str = "ferrum_program_binding_scatter_v1";

#[derive(Clone, Copy)]
struct ScratchLimits {
    packet_bytes: usize,
    live_bytes: usize,
}

impl Default for ScratchLimits {
    fn default() -> Self {
        Self {
            packet_bytes: 8 * 1024 * 1024,
            live_bytes: 32 * 1024 * 1024,
        }
    }
}

struct ScratchBudget {
    limits: ScratchLimits,
    live: AtomicUsize,
}

impl ScratchBudget {
    fn reserve(self: &Arc<Self>, bytes: usize) -> Option<ScratchReservation> {
        if bytes == 0 || bytes > self.limits.packet_bytes {
            return None;
        }
        self.live
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                live.checked_add(bytes)
                    .filter(|total| *total <= self.limits.live_bytes)
            })
            .ok()?;
        Some(ScratchReservation {
            budget: Arc::clone(self),
            bytes,
        })
    }
}

struct ScratchReservation {
    budget: Arc<ScratchBudget>,
    bytes: usize,
}

impl Drop for ScratchReservation {
    fn drop(&mut self) {
        self.budget.live.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

pub(super) struct BindingScatter {
    function: CudaFunction,
    budget: Arc<ScratchBudget>,
}

pub(super) struct PreparedScatter {
    pub(super) bytes: Box<[u8]>,
    pub(super) owner: ScatterOwner,
}

// The fresh non-replayable prelude owns this value through its enqueue closure.
// Its commands are retained by the existing fence or partial-submit quarantine.
// Field order queues device deallocation before returning budget on terminal Drop.
pub(super) struct ScatterOwner {
    device: CudaSlice<u8>,
    _reservation: ScratchReservation,
    function: CudaFunction,
    count: u32,
    submitted: AtomicBool,
}

impl BindingScatter {
    pub(super) fn load(context: &Arc<CudaContext>) -> Result<Self> {
        let module = context
            .load_module(Ptx::from_src(crate::ptx::GATHER_COLUMNS.to_owned()))
            .map_err(|error| {
                CudaDeviceRuntimeError::driver("binding scatter module load", error)
            })?;
        let function = module.load_function(FUNCTION).map_err(|error| {
            CudaDeviceRuntimeError::driver("binding scatter function load", error)
        })?;
        Ok(Self {
            function,
            budget: Arc::new(ScratchBudget {
                limits: ScratchLimits::default(),
                live: AtomicUsize::new(0),
            }),
        })
    }

    pub(super) fn prepare(
        &self,
        allocation_stream: &Arc<CudaStream>,
        transfers: &[CudaProgramBindingTransfer],
        arena_bytes: u64,
    ) -> Result<Option<PreparedScatter>> {
        let Some(packet) =
            Packet::prepare(transfers, arena_bytes, self.budget.limits.packet_bytes)?
        else {
            return Ok(None);
        };
        let Some(reservation) = self.budget.reserve(packet.bytes.len()) else {
            return Ok(None);
        };
        // No H2D or destination writes occur during preparation. Allocation
        // failure can still safely choose the original sparse-copy prelude.
        let device = match unsafe { allocation_stream.alloc::<u8>(packet.bytes.len()) } {
            Ok(device) => device,
            Err(error) if error.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY => {
                return Ok(None);
            }
            Err(error) => {
                return Err(CudaDeviceRuntimeError::driver(
                    "binding scatter allocation",
                    error,
                ));
            }
        };
        let owner = ScatterOwner {
            device,
            _reservation: reservation,
            function: self.function.clone(),
            count: packet.count,
            submitted: AtomicBool::new(false),
        };
        // Event tracking is disabled by this runtime. Establish allocation
        // readiness explicitly before another stream can use its raw address.
        if let Err(error) = allocation_stream.synchronize() {
            // Unknown allocation completion is not a healthy fallback. Keep
            // both its allocation and budget charged instead of freeing it.
            std::mem::forget(owner);
            return Err(CudaDeviceRuntimeError::driver(
                "binding scatter allocation readiness",
                error,
            ));
        }
        Ok(Some(PreparedScatter {
            bytes: packet.bytes.into_boxed_slice(),
            owner,
        }))
    }
}

impl ScatterOwner {
    pub(super) fn upload(&self, stream: &CudaStream, payload: &[u8]) -> Result<()> {
        if payload.len() != self.device.len() {
            return Err(CudaDeviceRuntimeError::contract(
                "binding scatter source length changed",
            ));
        }
        if self.submitted.swap(true, Ordering::AcqRel) {
            return Err(CudaDeviceRuntimeError::contract(
                "binding scatter prelude was submitted twice",
            ));
        }
        let packet = self.device.device_ptr(stream).0;
        unsafe { cudarc::driver::result::memcpy_htod_async(packet, payload, stream.cu_stream()) }
            .map_err(|error| {
                CudaDeviceRuntimeError::driver("compact program binding upload", error)
            })
    }

    pub(super) fn scatter(&self, stream: &CudaStream, arena: u64) -> Result<()> {
        let packet = self.device.device_ptr(stream).0;
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(&packet).arg(&arena).arg(&self.count);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (self.count, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| {
            CudaDeviceRuntimeError::driver("compact program binding scatter", error)
        })?;
        Ok(())
    }
}

struct Packet {
    bytes: Vec<u8>,
    count: u32,
}

impl Packet {
    fn prepare(
        transfers: &[CudaProgramBindingTransfer],
        arena_bytes: u64,
        limit: usize,
    ) -> Result<Option<Self>> {
        let invalid = CudaDeviceRuntimeError::contract;
        if transfers.is_empty() || arena_bytes == 0 {
            return Err(invalid("binding scatter transfers or arena are empty"));
        }
        let count = u32::try_from(transfers.len())
            .map_err(|_| invalid("binding scatter descriptor count exceeds u32"))?;
        let header = transfers
            .len()
            .checked_mul(DESCRIPTOR_BYTES)
            .ok_or_else(|| invalid("binding scatter headers overflow"))?;
        let total = transfers.iter().try_fold(header, |total, transfer| {
            total
                .checked_add(transfer.payload.len())
                .ok_or_else(|| invalid("binding scatter packet bytes overflow"))
        })?;
        if total > limit {
            return Ok(None);
        }
        u64::try_from(total).map_err(|_| invalid("binding scatter packet bytes exceed u64"))?;
        let mut ranges = Vec::new();
        for transfer in transfers {
            let width = u64::try_from(transfer.row_bytes)
                .map_err(|_| invalid("binding scatter row width exceeds u64"))?;
            let rows = u64::try_from(transfer.row_count)
                .map_err(|_| invalid("binding scatter rows exceed u64"))?;
            if width == 0
                || rows == 0
                || transfer.row_bytes.checked_mul(transfer.row_count)
                    != Some(transfer.payload.len())
            {
                return Err(invalid(
                    "binding scatter payload differs from its nonempty rows",
                ));
            }
            let end = (rows - 1)
                .checked_mul(transfer.destination_stride_bytes)
                .and_then(|offset| transfer.destination_offset_bytes.checked_add(offset))
                .and_then(|start| start.checked_add(width))
                .ok_or_else(|| invalid("binding scatter destination extent overflows"))?;
            if end > arena_bytes {
                return Err(invalid("binding scatter destination exceeds arena"));
            }
            if ranges.try_reserve_exact(transfer.row_count).is_err() {
                return Ok(None);
            }
            // The validated final row bounds every preceding nonnegative row.
            for row in 0..rows {
                let start =
                    transfer.destination_offset_bytes + row * transfer.destination_stride_bytes;
                ranges.push((start, start + width));
            }
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(invalid("binding scatter destination rows overlap"));
        }
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(total).is_err() {
            return Ok(None);
        }
        let mut source = header;
        for transfer in transfers {
            for field in [
                source as u64,
                transfer.destination_offset_bytes,
                transfer.row_bytes as u64,
                transfer.row_count as u64,
                transfer.destination_stride_bytes,
            ] {
                bytes.extend_from_slice(&field.to_le_bytes());
            }
            source += transfer.payload.len();
        }
        for transfer in transfers {
            bytes.extend_from_slice(&transfer.payload);
        }
        Ok(Some(Self { bytes, count }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transfer(
        offset: u64,
        pitch: u64,
        width: usize,
        rows: usize,
        bytes: &[u8],
    ) -> CudaProgramBindingTransfer {
        CudaProgramBindingTransfer {
            destination_offset_bytes: offset,
            destination_stride_bytes: pitch,
            row_bytes: width,
            row_count: rows,
            payload: bytes.into(),
        }
    }

    #[test]
    fn binding_scatter_packet_preserves_interleaved_rows_and_wire_offsets() {
        let packet = Packet::prepare(
            &[transfer(4, 8, 4, 2, &[2; 8]), transfer(0, 8, 4, 2, &[1; 8])],
            16,
            96,
        )
        .unwrap()
        .unwrap();
        let fields = packet.bytes[..80]
            .chunks_exact(8)
            .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(fields, [80, 4, 4, 2, 8, 88, 0, 4, 2, 8]);
        assert_eq!(&packet.bytes[80..88], &[2; 8]);
        assert_eq!(&packet.bytes[88..], &[1; 8]);
        assert_eq!(packet.count, 2);
    }

    #[test]
    fn binding_scatter_packet_rejects_actual_overlap_bounds_and_overflow() {
        for transfers in [
            vec![transfer(0, 3, 4, 2, &[0; 8])],
            vec![transfer(0, 8, 4, 2, &[0; 8]), transfer(10, 0, 1, 1, &[0])],
            vec![transfer(16, 0, 1, 1, &[0])],
            vec![transfer(u64::MAX, 0, 1, 1, &[0])],
            vec![transfer(0, u64::MAX, 1, 3, &[0; 3])],
            vec![transfer(0, 8, 4, 2, &[0; 7])],
            vec![transfer(0, 8, usize::MAX, 2, &[0])],
            vec![transfer(0, 0, 0, 1, &[])],
            vec![],
        ] {
            assert!(Packet::prepare(&transfers, 16, 1024).is_err());
        }
    }

    #[test]
    fn binding_scatter_capacity_falls_back_and_releases_only_its_reservation() {
        let budget = Arc::new(ScratchBudget {
            limits: ScratchLimits {
                packet_bytes: 64,
                live_bytes: 96,
            },
            live: AtomicUsize::new(0),
        });
        assert!(budget.reserve(0).is_none());
        assert!(budget.reserve(65).is_none());
        let first = budget.reserve(64).unwrap();
        let second = budget.reserve(32).unwrap();
        assert!(budget.reserve(1).is_none());
        drop(first);
        assert_eq!(budget.live.load(Ordering::Acquire), 32);
        let third = budget.reserve(64).unwrap();
        drop((second, third));
        assert_eq!(budget.live.load(Ordering::Acquire), 0);
        let input = [transfer(0, 0, 1, 1, &[7])];
        assert!(Packet::prepare(&input, 1, 40).unwrap().is_none());
        assert!(Packet::prepare(&input, 1, 41).unwrap().is_some());
    }
}

#[cfg(test)]
#[path = "binding_scatter/tests.rs"]
mod cuda_tests;
