use super::*;
use ferrum_interfaces::vnext::DeviceReadbackStagingLease;

/// Both the submission command and its terminal reader retain this snapshot.
/// Drop pinned storage first: its own event guard must finish any outstanding
/// DMA before the source retention and staging-budget lease can be released.
struct SubmissionReadback {
    snapshot: PinnedReadbackSnapshot,
    _staging: DeviceReadbackStagingLease,
}

/// The CUDA snapshot mechanism is independent of core's staging-budget lease.
/// Production retains both together in `SubmissionReadback` until terminal read.
struct PinnedReadbackSnapshot {
    host: Mutex<PinnedHostSlice<u8>>,
    source: CudaBufferRegion,
}

impl PinnedReadbackSnapshot {
    fn new(source: CudaBufferRegion, output_bytes: usize) -> Result<Self, CudaDeviceRuntimeError> {
        let mut host = unsafe {
            source
                ._allocation
                ._base
                .context()
                .alloc_pinned::<u8>(output_bytes)
        }
        .map_err(|error| {
            CudaDeviceRuntimeError::driver("submission readback host allocation", error)
        })?;
        host.as_mut_slice()
            .map_err(|error| {
                CudaDeviceRuntimeError::driver("submission readback host initialization", error)
            })?
            .fill(0);
        Ok(Self {
            host: Mutex::new(host),
            source,
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
        host.as_slice().map(<[u8]>::to_vec).map_err(|error| {
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
    let snapshot = Arc::new(SubmissionReadback {
        snapshot: PinnedReadbackSnapshot::new(source, output_bytes)?,
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
        "host.submission_readback",
        vec![snapshot.snapshot.source.clone()],
        Vec::new(),
        Box::new(move |stream, _blas, _regions, _storage| {
            command_snapshot
                .snapshot
                .enqueue(stream, destination_start..destination_end)
        }),
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
