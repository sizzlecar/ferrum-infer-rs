use ferrum_types::{FerrumError, Result};
use metal::objc::runtime::{Class, Object, BOOL, YES};
use metal::objc::{msg_send, sel, sel_impl};
use std::ffi::CStr;
use std::ptr::NonNull;

const INITIAL_ALLOCATION_CAPACITY: metal::NSUInteger = 2_048;

/// Runtime wrapper for the macOS 15 `MTLResidencySet` API.
///
/// metal-rs 0.31 does not expose residency sets, so this narrowly wraps the
/// Objective-C selectors while keeping ownership and availability checks in
/// one place. The wrapper is private to the Metal backend.
pub(super) struct MetalResidencySet {
    raw: NonNull<Object>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MetalResidencyStats {
    pub(super) allocation_count: usize,
    pub(super) allocated_size: u64,
}

impl MetalResidencySet {
    pub(super) fn new(device: &metal::DeviceRef) -> Result<Option<Self>> {
        let Some(descriptor_class) = Class::get("MTLResidencySetDescriptor") else {
            return Ok(None);
        };

        let selector = sel!(newResidencySetWithDescriptor:error:);
        let supported: BOOL = unsafe { msg_send![device, respondsToSelector: selector] };
        if supported != YES {
            return Ok(None);
        }

        unsafe {
            let descriptor: *mut Object = msg_send![descriptor_class, new];
            let descriptor = NonNull::new(descriptor).ok_or_else(|| {
                FerrumError::device("Metal returned a null MTLResidencySetDescriptor")
            })?;
            let _: () =
                msg_send![descriptor.as_ptr(), setInitialCapacity: INITIAL_ALLOCATION_CAPACITY];

            let mut error: *mut Object = std::ptr::null_mut();
            let raw: *mut Object = msg_send![
                device,
                newResidencySetWithDescriptor: descriptor.as_ptr()
                error: &mut error
            ];
            let _: () = msg_send![descriptor.as_ptr(), release];

            if !error.is_null() {
                return Err(FerrumError::device(format!(
                    "failed to create Metal residency set: {}",
                    ns_error_description(error)
                )));
            }
            let raw = NonNull::new(raw)
                .ok_or_else(|| FerrumError::device("Metal returned a null residency set"))?;
            Ok(Some(Self { raw }))
        }
    }

    pub(super) fn add_allocation(&self, buffer: &metal::BufferRef) {
        unsafe {
            let _: () = msg_send![self.raw.as_ptr(), addAllocation: buffer];
        }
    }

    pub(super) fn commit_and_request(&self) -> MetalResidencyStats {
        unsafe {
            let _: () = msg_send![self.raw.as_ptr(), commit];
            let _: () = msg_send![self.raw.as_ptr(), requestResidency];
            let allocation_count: metal::NSUInteger = msg_send![self.raw.as_ptr(), allocationCount];
            let allocated_size: u64 = msg_send![self.raw.as_ptr(), allocatedSize];
            MetalResidencyStats {
                allocation_count: allocation_count as usize,
                allocated_size,
            }
        }
    }
}

impl Drop for MetalResidencySet {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.raw.as_ptr(), endResidency];
            let _: () = msg_send![self.raw.as_ptr(), removeAllAllocations];
            let _: () = msg_send![self.raw.as_ptr(), release];
        }
    }
}

// MTLResidencySet supports cross-thread use. Ferrum serializes mutation with
// the per-mapping mutex that owns this wrapper.
unsafe impl Send for MetalResidencySet {}
unsafe impl Sync for MetalResidencySet {}

unsafe fn ns_error_description(error: *mut Object) -> String {
    let description: *mut Object = msg_send![error, localizedDescription];
    if description.is_null() {
        return "unknown NSError".to_string();
    }
    let utf8: *const std::os::raw::c_char = msg_send![description, UTF8String];
    if utf8.is_null() {
        return "NSError has no UTF-8 description".to_string();
    }
    CStr::from_ptr(utf8).to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::MTLResourceOptions;

    #[test]
    fn residency_set_round_trip_when_supported() {
        let device = metal::Device::system_default().expect("Metal device");
        let Some(set) = MetalResidencySet::new(&device).expect("residency capability probe") else {
            return;
        };
        let buffer = device.new_buffer(4096, MTLResourceOptions::StorageModeShared);
        set.add_allocation(&buffer);
        let stats = set.commit_and_request();
        assert_eq!(stats.allocation_count, 1);
        assert!(stats.allocated_size >= buffer.length());
    }
}
