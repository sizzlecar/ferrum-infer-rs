//! Fresh terminal copies outside reusable compute graphs, before their fence.

use super::*;
use std::any::Any;

const CAPACITY: usize = 32;

enum Storage {
    Uninitialized,
    Available(PinnedHostSlice<u32>),
    CheckedOut,
}

pub(super) struct Pool {
    storage: Mutex<Storage>,
}

impl Pool {
    pub(super) fn new() -> Self {
        Self {
            storage: Mutex::new(Storage::Uninitialized),
        }
    }

    fn take(
        &self,
        context: &Arc<CudaContext>,
    ) -> Result<Option<PinnedHostSlice<u32>>, CudaDeviceRuntimeError> {
        let mut storage = self.storage.lock().map_err(|_| {
            CudaDeviceRuntimeError::contract("terminal readback pool mutex poisoned")
        })?;
        if matches!(*storage, Storage::CheckedOut) {
            return Ok(None);
        }
        if matches!(*storage, Storage::Uninitialized) {
            let host = unsafe { context.alloc_pinned::<u32>(CAPACITY) }
                .map_err(|e| CudaDeviceRuntimeError::driver("terminal readback pin", e))?;
            *storage = Storage::Available(host);
        }
        match std::mem::replace(&mut *storage, Storage::CheckedOut) {
            Storage::Available(host) => Ok(Some(host)),
            _ => unreachable!("checked available terminal storage"),
        }
    }
}

struct Packet {
    runtime_instance: u64,
    pool: Arc<Pool>,
    sources: Vec<CudaBufferRegion>,
    host: Mutex<PinnedHostSlice<u32>>,
    enqueued: AtomicBool,
    failed: AtomicBool,
    #[cfg(test)]
    fail_after: AtomicU64,
}

impl Packet {
    fn enqueue(&self, stream: &CudaStream) -> Result<(), CudaDeviceRuntimeError> {
        if self.enqueued.swap(true, Ordering::AcqRel) {
            self.failed.store(true, Ordering::Release);
            return Err(CudaDeviceRuntimeError::contract(
                "terminal packet enqueued twice",
            ));
        }
        // Start fail-closed: a panic or any prefix-copy error must never recycle.
        self.failed.store(true, Ordering::Release);
        let mut host = self
            .host
            .lock()
            .map_err(|_| CudaDeviceRuntimeError::contract("terminal packet host mutex poisoned"))?;
        let (values, _dma_guard) = unsafe { host.stream_synced_mut_slice(stream) };
        for (_ordinal, (value, source)) in values.iter_mut().zip(&self.sources).enumerate() {
            unsafe {
                cudarc::driver::result::memcpy_dtoh_async(
                    std::slice::from_mut(value),
                    source.device_ptr,
                    stream.cu_stream(),
                )
            }
            .map_err(|e| CudaDeviceRuntimeError::driver("terminal packet copy", e))?;
            #[cfg(test)]
            if self.fail_after.load(Ordering::Relaxed) == (_ordinal + 1) as u64 {
                return Err(CudaDeviceRuntimeError::contract(
                    "test injected terminal-copy prefix failure",
                ));
            }
        }
        self.failed.store(false, Ordering::Release);
        Ok(())
    }
}

/// The caller has resolved current Core authorities. Preparation queues no work.
pub(super) fn prepare_regions(
    runtime: &CudaDeviceRuntime,
    stream: &mut CudaDeviceStream,
    sources: Vec<CudaBufferRegion>,
) -> Result<Option<(CudaDeviceCommand, Arc<dyn Any + Send + Sync>)>, CudaDeviceRuntimeError> {
    runtime.validate_stream(stream)?;
    if sources.is_empty() || sources.len() > CAPACITY {
        return Ok(None);
    }
    if sources.iter().any(|source| {
        source.runtime_instance != runtime.runtime_instance
            || source.element_type != ElementType::U32
            || source.length_bytes != 4
            || source.device_ptr % 4 != 0
    }) {
        return Err(CudaDeviceRuntimeError::contract(
            "terminal packet source is not a current aligned U32",
        ));
    }
    if stream.state.snapshot() == StreamState::Failed {
        return Err(CudaDeviceRuntimeError::contract(
            "terminal packet stream failed",
        ));
    }
    let Some(mut host) = stream.terminal_readback_pool.take(&runtime.context)? else {
        return Ok(None);
    };
    // Initialize inactive slots too. This also makes short-after-long reuse
    // inspectable in the backend correctness test.
    host.as_mut_slice()
        .map_err(|e| CudaDeviceRuntimeError::driver("terminal packet initialize", e))?
        .fill(0xdeadbeef);
    let packet = Arc::new(Packet {
        runtime_instance: runtime.runtime_instance,
        pool: Arc::clone(&stream.terminal_readback_pool),
        sources,
        host: Mutex::new(host),
        enqueued: AtomicBool::new(false),
        failed: AtomicBool::new(false),
        #[cfg(test)]
        fail_after: AtomicU64::new(0),
    });
    let captured = Arc::clone(&packet);
    let mut command = CudaDeviceCommand::operation(
        "terminal.u32_readback",
        packet.sources.clone(),
        move |stream, _| captured.enqueue(stream),
    )?;
    command.compute_dispatch_count = 0;
    command.transfer_command_count = packet.sources.len() as u64;
    Ok(Some((command, packet)))
}

pub(super) fn prepare(
    runtime: &CudaDeviceRuntime,
    stream: &mut CudaDeviceStream,
    request: ferrum_runtime::DeviceScalarU32ReadbackBatch<'_, CudaDeviceBuffer>,
) -> Result<
    Option<ferrum_runtime::DevicePreparedTerminalReadback<CudaDeviceCommand>>,
    CudaDeviceRuntimeError,
> {
    let mut regions = Vec::with_capacity(request.sources().len());
    for source in request.sources() {
        let buffer = source.buffer();
        let copy = source.region();
        if buffer.runtime_instance != runtime.runtime_instance
            || source.descriptor() != &buffer.descriptor
            || copy.length_bytes() != 4
            || copy.destination_offset_bytes() != 0
            || copy.source_offset_bytes() % 4 != 0
        {
            return Err(CudaDeviceRuntimeError::contract(
                "terminal readback source differs from its current descriptor",
            ));
        }
        let end = checked_end(
            copy.source_offset_bytes(),
            4,
            buffer.descriptor.size_bytes,
            "terminal readback source",
        )?;
        regions.push(
            buffer.retained_region(copy.source_offset_bytes()..end, source.retention().clone())?,
        );
    }
    Ok(
        prepare_regions(runtime, stream, regions)?.map(|(command, payload)| {
            ferrum_runtime::DevicePreparedTerminalReadback::new(command, payload)
        }),
    )
}

/// Core calls this only after the original successful fence, releasing all
/// command owners first. No GPU work or scalar fallback is permitted here.
pub(super) fn finish(
    runtime: &CudaDeviceRuntime,
    payload: Arc<dyn Any + Send + Sync>,
) -> Result<Vec<u32>, CudaDeviceRuntimeError> {
    let packet = payload
        .downcast::<Packet>()
        .map_err(|_| CudaDeviceRuntimeError::contract("foreign terminal readback payload"))?;
    if packet.runtime_instance != runtime.runtime_instance
        || !packet.enqueued.load(Ordering::Acquire)
        || packet.failed.load(Ordering::Acquire)
    {
        return Err(CudaDeviceRuntimeError::contract(
            "terminal readback payload is not successfully submitted",
        ));
    }
    let packet = Arc::try_unwrap(packet).map_err(|_| {
        CudaDeviceRuntimeError::contract("terminal packet still has command owners")
    })?;
    let host = packet
        .host
        .into_inner()
        .map_err(|_| CudaDeviceRuntimeError::contract("terminal packet host mutex poisoned"))?;
    let values = host
        .as_slice()
        .map_err(|e| CudaDeviceRuntimeError::driver("terminal packet extract", e))?;
    let result = values[..packet.sources.len()].to_vec();
    let mut storage =
        packet.pool.storage.lock().map_err(|_| {
            CudaDeviceRuntimeError::contract("terminal readback pool mutex poisoned")
        })?;
    if !matches!(*storage, Storage::CheckedOut) {
        return Err(CudaDeviceRuntimeError::contract(
            "terminal readback pool already has storage",
        ));
    }
    *storage = Storage::Available(host);
    Ok(result)
}

#[cfg(test)]
pub(super) fn inject_failure(payload: &Arc<dyn Any + Send + Sync>, after: usize) {
    payload
        .downcast_ref::<Packet>()
        .unwrap()
        .fail_after
        .store(after as u64, Ordering::Relaxed);
}

#[cfg(test)]
pub(super) fn available_storage(stream: &CudaDeviceStream) -> Option<Vec<u32>> {
    let storage = stream.terminal_readback_pool.storage.lock().unwrap();
    match &*storage {
        Storage::Available(host) => Some(host.as_slice().unwrap().to_vec()),
        _ => None,
    }
}
