//! Managed tensor IO uses native no-copy views, never a hidden payload copy.
use super::*;
use ferrum_runtime::{HostMemoryClass, ManagedHostBuffer, ManagedHostBytes};

pub(super) fn upload(
    runtime: &MetalDeviceRuntime,
    source: ManagedHostBytes,
    layout: HostTransferLayout,
    destination: &MetalDeviceBuffer,
    offset: u64,
) -> Result<Option<MetalDeviceCommand>, MetalDeviceRuntimeError> {
    runtime.validate_buffer(destination)?;
    if source.class() != HostMemoryClass::InputPreprocess
        || destination.descriptor.usage != BufferUsage::Activations
        || layout.element_type() != destination.descriptor.element_type
    {
        return Err(MetalDeviceRuntimeError::contract(
            "Metal managed upload class, usage or dtype differs",
        ));
    }
    layout
        .validate_bytes(source.len())
        .map_err(|e| MetalDeviceRuntimeError::contract(e.to_string()))?;
    let (destination_start, destination_end) = transfer_range(
        layout,
        offset,
        source.len() as u64,
        destination.descriptor.size_bytes,
    )?;
    let destination = destination.region(destination_start..destination_end)?;
    let memory = source
        .retained_region()
        .map_err(|e| MetalDeviceRuntimeError::contract(e.to_string()))?;
    let native = crate::backend::metal::retained_memory::new_retained_buffer(
        runtime.device(),
        memory,
        source.bytes(),
        layout.element_type().size_bytes() as usize,
    )
    .map_err(|e| MetalDeviceRuntimeError::contract(e.to_string()))?;
    let Some(native) = native else {
        // The composition must supply a Host account aligned to the native
        // page size. Decline before submission instead of copying the payload.
        return Ok(None);
    };
    let byte_offset = native.byte_offset as u64;
    Ok(Some(MetalDeviceCommand::transfer(
        runtime.runtime_instance,
        HOST_UPLOAD_NATIVE_OPERATION_ID.as_str(),
        vec![destination],
        vec![native.buffer],
        Box::new(move |encoder, regions, staging| {
            encoder.with_blit(|blit| {
                blit.copy_from_buffer(
                    &staging[0],
                    byte_offset,
                    regions[0].buffer(),
                    regions[0].offset_bytes(),
                    regions[0].length_bytes(),
                );
            });
            Ok(())
        }),
    )))
}

pub(super) fn readback(
    runtime: &MetalDeviceRuntime,
    stream: &mut MetalDeviceStream,
    source: &MetalDeviceBuffer,
    region: CopyRegion,
    layout: HostTransferLayout,
    destination: &mut ManagedHostBuffer,
) -> Result<bool, MetalDeviceRuntimeError> {
    runtime.validate_stream(stream)?;
    runtime.validate_buffer(source)?;
    if layout.element_type() != source.descriptor.element_type
        || destination.class() != HostMemoryClass::Output
    {
        return Err(MetalDeviceRuntimeError::contract(
            "Metal managed readback class or dtype differs",
        ));
    }
    let (start, end) = transfer_range(
        layout,
        region.source_offset_bytes(),
        region.length_bytes(),
        source.descriptor.size_bytes,
    )?;
    let (out_start, out_end) = transfer_range(
        layout,
        region.destination_offset_bytes(),
        region.length_bytes(),
        destination.len() as u64,
    )?;
    let source = source.region(start..end)?;
    if source.buffer().storage_mode() != MTLStorageMode::Shared {
        return Err(MetalDeviceRuntimeError::contract(
            "Metal managed readback requires shared storage",
        ));
    }
    runtime.synchronize(stream)?;
    let offset = checked_usize(source.offset_bytes(), "Metal managed readback offset")?;
    let output = &mut destination.as_mut()[out_start as usize..out_end as usize];
    // SAFETY: synchronization proved all GPU work quiescent. The exact shared
    // source range is retained; destination is exclusively borrowed, accounted
    // Host memory. No reference or DMA access survives this synchronous copy.
    unsafe {
        std::ptr::copy(
            source.buffer().contents().cast::<u8>().add(offset),
            output.as_mut_ptr(),
            output.len(),
        );
    }
    Ok(true)
}

fn transfer_range(
    layout: HostTransferLayout,
    offset: u64,
    bytes: u64,
    capacity: u64,
) -> Result<(u64, u64), MetalDeviceRuntimeError> {
    if bytes
        != layout
            .byte_len()
            .map_err(|e| MetalDeviceRuntimeError::contract(e.to_string()))?
        || offset % layout.element_type().size_bytes() != 0
    {
        return Err(MetalDeviceRuntimeError::contract(
            "Metal managed transfer extent or scalar alignment differs",
        ));
    }
    let end = checked_end(offset, bytes, capacity, "Metal managed transfer")?;
    Ok((offset, end))
}

#[cfg(test)]
mod tests;
