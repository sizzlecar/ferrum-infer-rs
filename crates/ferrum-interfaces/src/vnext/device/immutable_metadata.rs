//! Optional backend declarations, not resource-admission authority.
use std::any::Any;
use std::sync::Arc;

use super::{BufferDescriptor, DeviceDescriptor, DeviceRuntime};
#[cfg(test)]
mod tests;

/// A backend promises that its original descriptor getter remains equal to
/// this descriptor for the lifetime of this exact runtime object. Core still
/// checks the initial getter and binds its private proof to the actual owner.
pub struct ImmutableRuntimeMetadata<'a> {
    owner: &'a (dyn Any + Send + Sync),
    descriptor: &'a DeviceDescriptor,
}

impl<'a> ImmutableRuntimeMetadata<'a> {
    pub fn declare<R: DeviceRuntime>(owner: &'a R, descriptor: &'a DeviceDescriptor) -> Self {
        Self { owner, descriptor }
    }

    pub(crate) fn for_owner<R: DeviceRuntime>(&self, owner: &R) -> Option<&DeviceDescriptor> {
        self.owner
            .downcast_ref::<R>()
            .filter(|actual| std::ptr::eq(*actual, owner))
            .map(|_| self.descriptor)
    }
}

/// In addition to runtime metadata, the backend promises that the original
/// buffer getter remains equal for this exact buffer facade. Equal numeric
/// IDs or equal descriptors are not owner equivalence.
pub struct ImmutableBufferMetadata<'a, B> {
    runtime: &'a (dyn Any + Send + Sync),
    buffer: &'a B,
    descriptor: &'a BufferDescriptor,
}

impl<'a, B> ImmutableBufferMetadata<'a, B> {
    pub fn declare<R: DeviceRuntime<Buffer = B>>(
        runtime: &'a R,
        buffer: &'a B,
        descriptor: &'a BufferDescriptor,
    ) -> Self {
        Self {
            runtime,
            buffer,
            descriptor,
        }
    }

    pub(crate) fn for_owner<R: DeviceRuntime<Buffer = B>>(
        &self,
        runtime: &R,
        buffer: &B,
    ) -> Option<&BufferDescriptor> {
        self.runtime
            .downcast_ref::<R>()
            .filter(|actual| std::ptr::eq(*actual, runtime))
            .filter(|_| std::ptr::eq(self.buffer, buffer))
            .map(|_| self.descriptor)
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
