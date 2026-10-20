//! Native no-copy views whose host owner follows every native buffer retain.

use ferrum_runtime::RetainedHostMemoryRegion;
use ferrum_types::{FerrumError, Result};
use metal::foreign_types::ForeignType;
use metal::MTLResourceOptions;
use std::ffi::c_void;

pub(super) struct RetainedMetalBuffer {
    pub(super) buffer: metal::Buffer,
    pub(super) byte_offset: usize,
}

pub(super) struct NativeMmapWindow {
    pub(super) start: usize,
    pub(super) len: usize,
    pub(super) byte_offset: usize,
}

/// Includes all native page padding, not merely the logical tensor bytes.
pub(super) fn checked_native_window(
    owner_start: usize,
    owner_len: usize,
    tensor_start: usize,
    tensor_len: usize,
    page_size: usize,
) -> Option<NativeMmapWindow> {
    if tensor_len == 0 || !page_size.is_power_of_two() {
        return None;
    }
    let owner_end = owner_start.checked_add(owner_len)?;
    let tensor_end = tensor_start.checked_add(tensor_len)?;
    let start = tensor_start & !(page_size - 1);
    let end = tensor_end.checked_add(page_size - 1)? & !(page_size - 1);
    if start < owner_start || end > owner_end || end <= start {
        return None;
    }
    Some(NativeMmapWindow {
        start,
        len: end - start,
        byte_offset: tensor_start - start,
    })
}

pub(super) fn native_page_size() -> Result<usize> {
    // SAFETY: sysconf has no pointer arguments and does not mutate host memory.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(size)
        .ok()
        .filter(|size| size.is_power_of_two())
        .ok_or_else(|| FerrumError::backend("Metal host page size is invalid"))
}

/// `None` requests a copying fallback. The native view must be entirely inside
/// the stable owner, and the payload must belong to the exact retained region.
#[allow(
    unexpected_cfgs,
    reason = "objc 0.2 selector macros use legacy cargo-clippy cfg"
)]
pub(super) fn new_retained_buffer(
    device: &metal::DeviceRef,
    memory: RetainedHostMemoryRegion,
    bytes: &[u8],
    required_alignment: usize,
) -> Result<Option<RetainedMetalBuffer>> {
    use metal::objc::{msg_send, sel, sel_impl};
    if !required_alignment.is_power_of_two() {
        return Err(FerrumError::backend(
            "Metal native-view alignment is invalid",
        ));
    }
    let tensor_start = bytes.as_ptr() as usize;
    let registered_start = memory.bytes().as_ptr() as usize;
    let registered_end = registered_start.checked_add(memory.length_bytes());
    let tensor_end = tensor_start.checked_add(bytes.len());
    if tensor_start < registered_start
        || !matches!((tensor_end, registered_end), (Some(end), Some(limit)) if end <= limit)
        || !tensor_start.is_multiple_of(required_alignment)
    {
        return Ok(None);
    }
    let owner = memory.owner_bytes();
    let Some(window) = checked_native_window(
        owner.as_ptr() as usize,
        owner.len(),
        tensor_start,
        bytes.len(),
        native_page_size()?,
    ) else {
        return Ok(None);
    };
    // Capture only stable host memory. Capturing a registration would cycle
    // through its residency set, which itself retains the native buffer.
    let deallocator =
        block::ConcreteBlock::new(move |_pointer: *const c_void, _length: metal::NSUInteger| {
            let _retained_until_native_release = &memory;
        })
        .copy();
    let callback: &block::Block<(*const c_void, metal::NSUInteger), ()> = &deallocator;
    // SAFETY: the checked native window lies within `memory`'s stable owner.
    // Metal copies the heap block and releases it after the final native buffer
    // retain. Inspect nil before constructing metal-rs's non-null owned type.
    let raw: *mut metal::MTLBuffer = unsafe {
        msg_send![device,
            newBufferWithBytesNoCopy: window.start as *const c_void
            length: window.len as metal::NSUInteger
            options: MTLResourceOptions::StorageModeShared
            deallocator: callback
        ]
    };
    if raw.is_null() {
        Ok(None)
    } else {
        let buffer = unsafe { metal::Buffer::from_ptr(raw) };
        Ok(
            (buffer.length() == window.len as u64).then_some(RetainedMetalBuffer {
                buffer,
                byte_offset: window.byte_offset,
            }),
        )
    }
}
