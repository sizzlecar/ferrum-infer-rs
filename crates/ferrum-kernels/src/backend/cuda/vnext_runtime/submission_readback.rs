use super::*;
use ferrum_interfaces::vnext::{
    DeviceReadbackStagingLease, DeviceReadbackStagingStorage, DeviceReadbackStagingStorageError,
};
use std::ops::Deref;

type PinnedHostStorage = Mutex<PinnedHostSlice<u8>>;

pub(super) const NATIVE_OPERATION: &str = "host.submission_readback";

/// Both the submission command and its terminal reader retain this snapshot.
/// Source retention is submission-local; the cached host allocation contains no
/// source/Step ownership. All command/reader owners must release their leases
/// before a later submission can reuse the host slot.
struct SubmissionReadback {
    snapshot: PinnedReadbackSnapshot<DeviceReadbackStagingStorage<PinnedHostStorage>>,
    _staging: DeviceReadbackStagingLease,
}

/// The CUDA snapshot mechanism is independent of core's staging-budget lease.
/// Production retains both together in `SubmissionReadback` until terminal read.
struct PinnedReadbackSnapshot<H> {
    host: H,
    source: CudaBufferRegion,
    output_bytes: usize,
}

fn allocate_host_storage(
    source: &CudaBufferRegion,
    capacity_bytes: usize,
) -> Result<PinnedHostStorage, CudaDeviceRuntimeError> {
    let host = unsafe {
        source
            ._allocation
            ._base
            .context()
            .alloc_pinned::<u8>(capacity_bytes)
    }
    .map_err(|error| {
        CudaDeviceRuntimeError::driver("submission readback host allocation", error)
    })?;
    Ok(Mutex::new(host))
}

impl<H: Deref<Target = PinnedHostStorage>> PinnedReadbackSnapshot<H> {
    fn new(
        source: CudaBufferRegion,
        host: H,
        output_bytes: usize,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        {
            let mut allocation = host
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if allocation.len() < output_bytes {
                return Err(CudaDeviceRuntimeError::contract(
                    "cached submission readback storage is smaller than its output layout",
                ));
            }
            // A cached slot is exclusively leased. Clear all prior payload and
            // padding, observing only this allocation's earlier DMA event.
            allocation
                .as_mut_slice()
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("submission readback host initialization", error)
                })?
                .fill(0);
        }
        Ok(Self {
            host,
            source,
            output_bytes,
        })
    }

    fn enqueue(
        &self,
        stream: &CudaStream,
        destination: Range<usize>,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let mut host = self
            .host
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // This guard records the allocation's own event even if enqueue fails.
        // The eager result binding's DMA and event precede its parent fence.
        let (bytes, _guard) = unsafe { host.stream_synced_mut_slice(stream) };
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(
                &mut bytes[destination],
                self.source.device_ptr,
                stream.cu_stream(),
            )
        }
        .map_err(|error| CudaDeviceRuntimeError::driver("submission readback snapshot", error))
    }

    fn read(&self) -> Result<Vec<u8>, CudaDeviceRuntimeError> {
        let host = self
            .host
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Core invokes this only for the successful exact parent terminal.
        // Observe this allocation's earlier event, never the whole stream.
        host.as_slice()
            .map(|bytes| bytes[..self.output_bytes].to_vec())
            .map_err(|error| {
                CudaDeviceRuntimeError::driver("submission readback host observation", error)
            })
    }
}

pub(super) fn prepare(
    runtime: &CudaDeviceRuntime,
    request: DeviceSubmissionReadbackRequest<'_, CudaDeviceBuffer>,
) -> Result<
    PreparedDeviceSubmissionReadback<CudaDeviceCommand, CudaDeviceRuntimeError>,
    CudaDeviceRuntimeError,
> {
    let (source, region, layout, retention, staging) = request.into_parts();
    runtime.validate_buffer(source)?;
    let output_bytes = validate_range(&source.descriptor, region, layout, staging.bytes())?;
    let source_end = region
        .source_offset_bytes()
        .checked_add(region.length_bytes())
        .ok_or_else(|| CudaDeviceRuntimeError::contract("submission readback source overflow"))?;
    let source = source.retained_region(region.source_offset_bytes()..source_end, retention)?;
    let capacity_bytes = checked_usize(staging.bytes(), "submission readback staging capacity")?;
    let host = staging
        .get_or_try_init_storage(|| allocate_host_storage(&source, capacity_bytes))
        .map_err(|error| match error {
            DeviceReadbackStagingStorageError::Initialization(error) => error,
            error => CudaDeviceRuntimeError::contract(error.to_string()),
        })?;
    let snapshot = Arc::new(SubmissionReadback {
        snapshot: PinnedReadbackSnapshot::new(source, host, output_bytes)?,
        _staging: staging,
    });
    let destination_start = checked_usize(
        region.destination_offset_bytes(),
        "submission readback offset",
    )?;
    let destination_end = checked_usize(
        region.destination_offset_bytes() + region.length_bytes(),
        "submission readback end",
    )?;
    let command_snapshot = Arc::clone(&snapshot);
    let command = CudaDeviceCommand::transfer(
        runtime.runtime_instance,
        NATIVE_OPERATION,
        vec![snapshot.snapshot.source.clone()],
        Vec::new(),
        Box::new(move |stream, _blas, _regions, _storage| {
            command_snapshot
                .snapshot
                .enqueue(stream, destination_start..destination_end)
        }),
    )
    .with_core_transfer(
        ferrum_interfaces::execution_cost::StatisticalTransferKindV1::DeviceToHost,
        region.length_bytes(),
        runtime.structured_capture(),
    );
    Ok(PreparedDeviceSubmissionReadback::new(
        Some(command),
        move || snapshot.snapshot.read(),
    ))
}

#[cfg(test)]
mod gpu_tests;

fn validate_range(
    source: &BufferDescriptor,
    region: CopyRegion,
    layout: HostTransferLayout,
    staging_bytes: u64,
) -> Result<usize, CudaDeviceRuntimeError> {
    let bytes = layout
        .byte_len()
        .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
    let element_bytes = layout.element_type().size_bytes();
    if layout.element_type() != source.element_type
        || region.source_offset_bytes() % element_bytes != 0
        || region.destination_offset_bytes() % element_bytes != 0
        || region.length_bytes() % element_bytes != 0
        || bytes > staging_bytes
    {
        return Err(CudaDeviceRuntimeError::contract(
            "submission readback layout, alignment or staging lease differs from its source",
        ));
    }
    checked_end(
        region.source_offset_bytes(),
        region.length_bytes(),
        source.size_bytes,
        "submission readback source",
    )?;
    checked_end(
        region.destination_offset_bytes(),
        region.length_bytes(),
        bytes,
        "submission readback output",
    )?;
    checked_usize(bytes, "submission readback output")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::{BufferUsage, ResourceId};

    fn source() -> BufferDescriptor {
        BufferDescriptor {
            resource_id: ResourceId::new("readback.range-test").unwrap(),
            size_bytes: 16,
            alignment_bytes: 4,
            usage: BufferUsage::Transfer,
            element_type: ElementType::U32,
        }
    }

    #[test]
    fn staging_lease_covers_the_host_layout_including_destination_offset() {
        let layout = HostTransferLayout::new(ElementType::U32, 4).unwrap();
        let region = CopyRegion::new(4, 4, 8).unwrap();
        assert_eq!(validate_range(&source(), region, layout, 16).unwrap(), 16);
        assert!(validate_range(&source(), region, layout, 8).is_err());
    }

    #[test]
    fn readback_requires_whole_elements_and_bounded_source_and_destination() {
        let layout = HostTransferLayout::new(ElementType::U32, 4).unwrap();
        for (source_offset, destination_offset, length) in
            [(2, 0, 4), (0, 2, 4), (0, 0, 2), (12, 0, 8), (0, 12, 8)]
        {
            let region = CopyRegion::new(source_offset, destination_offset, length).unwrap();
            assert!(validate_range(&source(), region, layout, 16).is_err());
        }
        let wrong_type = HostTransferLayout::new(ElementType::F32, 4).unwrap();
        assert!(validate_range(
            &source(),
            CopyRegion::new(0, 0, 16).unwrap(),
            wrong_type,
            16,
        )
        .is_err());
    }
}
