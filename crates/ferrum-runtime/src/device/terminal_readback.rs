use std::any::Any;
use std::sync::Arc;

use super::{BufferDescriptor, DeviceBufferRetention};
use crate::CopyRegion;

/// One current, Core-validated scalar source. The retention belongs to this
/// wave; an equal address is not permission to retain an earlier wave's owner.
pub struct DeviceScalarU32ReadbackSource<'a, B> {
    buffer: &'a B,
    descriptor: &'a BufferDescriptor,
    region: CopyRegion,
    retention: DeviceBufferRetention,
}

impl<'a, B> DeviceScalarU32ReadbackSource<'a, B> {
    pub(crate) fn new(
        buffer: &'a B,
        descriptor: &'a BufferDescriptor,
        region: CopyRegion,
        retention: DeviceBufferRetention,
    ) -> Self {
        Self {
            buffer,
            descriptor,
            region,
            retention,
        }
    }

    pub fn buffer(&self) -> &B {
        self.buffer
    }
    pub fn descriptor(&self) -> &BufferDescriptor {
        self.descriptor
    }
    pub fn region(&self) -> &CopyRegion {
        &self.region
    }
    pub fn retention(&self) -> &DeviceBufferRetention {
        &self.retention
    }
}

/// Canonically ordered scalar outputs of exactly one freshly admitted wave.
pub struct DeviceScalarU32ReadbackBatch<'a, B> {
    sources: Vec<DeviceScalarU32ReadbackSource<'a, B>>,
}

impl<'a, B> DeviceScalarU32ReadbackBatch<'a, B> {
    pub(crate) fn new(sources: Vec<DeviceScalarU32ReadbackSource<'a, B>>) -> Self {
        Self { sources }
    }

    pub fn sources(&self) -> &[DeviceScalarU32ReadbackSource<'a, B>] {
        &self.sources
    }
}

/// Optional fresh submission tail and its completion payload. The tail is
/// outside reusable capture and must finish before the submission's one fence.
/// The payload must also be retained by the tail on any partial enqueue.
pub struct DevicePreparedTerminalReadback<C> {
    command: C,
    payload: Arc<dyn Any + Send + Sync>,
}

impl<C> DevicePreparedTerminalReadback<C> {
    pub fn new(command: C, payload: Arc<dyn Any + Send + Sync>) -> Self {
        Self { command, payload }
    }

    pub(crate) fn into_parts(self) -> (C, Arc<dyn Any + Send + Sync>) {
        (self.command, self.payload)
    }
}
