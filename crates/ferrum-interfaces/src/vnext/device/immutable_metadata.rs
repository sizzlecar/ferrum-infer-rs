//! Optional backend declarations, not resource-admission authority.
use std::sync::Arc;

use super::{BufferDescriptor, DeviceDescriptor, DeviceRuntime};
#[cfg(test)]
mod tests;

/// A backend promises that its original descriptor getter remains equal to
/// this descriptor for the lifetime of this exact runtime object. Core still
/// checks the initial getter and binds its private proof to the actual owner.
pub struct ImmutableRuntimeMetadata<'a, R: DeviceRuntime + ?Sized> {
    owner: &'a R,
    descriptor: &'a DeviceDescriptor,
}

impl<'a, R: DeviceRuntime + ?Sized> ImmutableRuntimeMetadata<'a, R> {
    pub fn declare(owner: &'a R, descriptor: &'a DeviceDescriptor) -> Self {
        Self { owner, descriptor }
    }

    pub(crate) fn for_owner(&self, owner: &R) -> Option<&DeviceDescriptor> {
        std::ptr::eq(self.owner, owner).then_some(self.descriptor)
    }
}

/// In addition to runtime metadata, the backend promises that the original
/// buffer getter remains equal for this exact buffer facade. Equal numeric
/// IDs or equal descriptors are not owner equivalence.
pub struct ImmutableBufferMetadata<'a, R: DeviceRuntime + ?Sized> {
    runtime: &'a R,
    buffer: &'a R::Buffer,
    descriptor: &'a BufferDescriptor,
}

impl<'a, R: DeviceRuntime + ?Sized> ImmutableBufferMetadata<'a, R> {
    pub fn declare(
        runtime: &'a R,
        buffer: &'a R::Buffer,
        descriptor: &'a BufferDescriptor,
    ) -> Self {
        Self {
            runtime,
            buffer,
            descriptor,
        }
    }

    pub(crate) fn for_owner(&self, runtime: &R, buffer: &R::Buffer) -> Option<&BufferDescriptor> {
        (std::ptr::eq(self.runtime, runtime) && std::ptr::eq(self.buffer, buffer))
            .then_some(self.descriptor)
    }
}

/// Non-owning identity of one backend resident entry. It must be replaced on
/// eviction, recapture or any change to that entry's segments. This marker
/// retains neither an executable nor any device allocation.
#[derive(Clone, Debug)]
pub struct DeviceReusableExecutionEntryIdentity(Arc<()>);

impl DeviceReusableExecutionEntryIdentity {
    pub fn new() -> Self {
        Self(Arc::new(()))
    }
    pub fn same_entry(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Default for DeviceReusableExecutionEntryIdentity {
    fn default() -> Self {
        Self::new()
    }
}
