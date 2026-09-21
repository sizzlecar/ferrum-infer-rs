use super::*;

pub(super) fn prepare(
    runtime: &MetalDeviceRuntime,
    request: DeviceSubmissionReadbackRequest<'_, MetalDeviceBuffer>,
) -> Result<
    PreparedDeviceSubmissionReadback<MetalDeviceCommand, MetalDeviceRuntimeError>,
    MetalDeviceRuntimeError,
> {
    let (source, region, layout, retention, staging) = request.into_parts();
    runtime.validate_buffer(source)?;
    let output_bytes = validate_range(&source.descriptor, region, layout, staging.bytes())?;
    let source_end = region
        .source_offset_bytes()
        .checked_add(region.length_bytes())
        .ok_or_else(|| MetalDeviceRuntimeError::contract("submission readback source overflow"))?;
    let source = source.retained_region(region.source_offset_bytes()..source_end, retention)?;
    if source.buffer().storage_mode() != MTLStorageMode::Shared {
        return Err(MetalDeviceRuntimeError::contract(
            "submission readback requires shared Metal storage",
        ));
    }
    let destination_start = checked_usize(
        region.destination_offset_bytes(),
        "submission readback offset",
    )?;
    let destination_end = checked_usize(
        region.destination_offset_bytes() + region.length_bytes(),
        "submission readback end",
    )?;
    Ok(PreparedDeviceSubmissionReadback::new(None, move || {
        // The readback owns both the logical slot retention and the staging
        // lease through this copy. Core only invokes the reader after its exact
        // parent terminal and before releasing that parent's resource lease.
        let _staging = &staging;
        read_completed_region(&source, output_bytes, destination_start..destination_end)
    }))
}

/// The caller must own the source slot and have observed its writer's terminal
/// fence. No queue/stream is accepted here: later work must not delay this copy.
fn read_completed_region(
    source: &MetalBufferRegion,
    output_bytes: usize,
    destination: Range<usize>,
) -> Result<Vec<u8>, MetalDeviceRuntimeError> {
    if source.buffer().storage_mode() != MTLStorageMode::Shared
        || destination.start > destination.end
        || destination.end > output_bytes
        || u64::try_from(destination.len()).map_or(true, |length| length > source.length_bytes())
    {
        return Err(MetalDeviceRuntimeError::contract(
            "completed readback exceeds its shared source or host destination",
        ));
    }
    let source_start = checked_usize(source.offset_bytes(), "submission readback source offset")?;
    let mut output = vec![0_u8; output_bytes];
    unsafe {
        let pointer = source.buffer().contents().cast::<u8>().add(source_start);
        std::ptr::copy_nonoverlapping(
            pointer,
            output[destination.clone()].as_mut_ptr(),
            destination.len(),
        );
    }
    Ok(output)
}

fn validate_range(
    source: &BufferDescriptor,
    region: CopyRegion,
    layout: HostTransferLayout,
    staging_bytes: u64,
) -> Result<usize, MetalDeviceRuntimeError> {
    let bytes = layout
        .byte_len()
        .map_err(|error| MetalDeviceRuntimeError::contract(error.to_string()))?;
    let element_bytes = layout.element_type().size_bytes();
    if layout.element_type() != source.element_type
        || region.source_offset_bytes() % element_bytes != 0
        || region.destination_offset_bytes() % element_bytes != 0
        || region.length_bytes() % element_bytes != 0
        || bytes > staging_bytes
    {
        return Err(MetalDeviceRuntimeError::contract(
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
    use ferrum_interfaces::vnext::{
        DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageView, ResourceId,
    };

    #[test]
    fn completed_parent_bytes_are_readable_while_later_queue_work_is_gated() {
        let runtime = MetalDeviceRuntime::new(MetalDeviceRuntimeConfig {
            device_id: DeviceId::new("device/metal/submission-readback").unwrap(),
            runtime_implementation_fingerprint: "a".repeat(64),
            capabilities: BTreeSet::new(),
            dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )
            .unwrap()]),
        })
        .unwrap();
        let source = runtime
            .allocate_request(
                &BufferRequest::new(
                    ResourceId::new("resource/readback-parent").unwrap(),
                    4,
                    64,
                    BufferUsage::Activations,
                    ElementType::U32,
                )
                .unwrap(),
            )
            .unwrap();
        let bytes = 0x1234abcd_u32.to_le_bytes();
        let upload = runtime
            .encode_upload(
                &bytes,
                HostTransferLayout::new(ElementType::U32, 1).unwrap(),
                &source,
                0,
            )
            .unwrap();
        let mut stream = runtime.create_stream().unwrap();
        let parent = runtime
            .submit_commands(
                &mut stream,
                vec![(DeviceCommandPhase::DynamicBinding, None, upload)],
                DeviceTimingMode::Off,
                &DisabledDeviceSubmissionTimingSink,
            )
            .unwrap();
        let source_region = source.region(0..4).unwrap();
        let gate = runtime.device.new_shared_event();
        let child = stream.queue.new_command_buffer().to_owned();
        child.encode_wait_for_event(&gate, 1);
        let blit = child.new_blit_command_encoder();
        blit.fill_buffer(
            source_region.buffer(),
            NSRange {
                location: source_region.offset_bytes(),
                length: 4,
            },
            0,
        );
        blit.end_encoding();
        child.commit();
        struct ReleaseChild {
            gate: metal::SharedEvent,
            child: metal::CommandBuffer,
        }
        impl Drop for ReleaseChild {
            fn drop(&mut self) {
                self.gate.set_signaled_value(1);
                self.child.wait_until_completed();
            }
        }
        let child = ReleaseChild { gate, child };
        assert!(runtime
            .wait_fence(&parent)
            .unwrap()
            .terminal()
            .is_succeeded());
        assert_ne!(child.child.status(), MTLCommandBufferStatus::Completed);
        assert_ne!(child.child.status(), MTLCommandBufferStatus::Error);
        let result = read_completed_region(&source_region, 4, 0..4).unwrap();
        assert_eq!(result, bytes);
        assert_eq!(child.gate.signaled_value(), 0);
        drop(child);
        assert_eq!(
            read_completed_region(&source_region, 4, 0..4).unwrap(),
            [0; 4]
        );
        assert_eq!(result, bytes);
    }

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
