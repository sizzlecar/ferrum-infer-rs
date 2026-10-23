use super::{CopyRegion, DeviceBufferRetention};

/// A core-authorized copy from a completed tensor output into fixed Request
/// state. Both admitted backings remain owned through the producer fence.
/// Backends can inspect the sealed request but cannot mint one.
pub struct DeviceTensorStateCopy<'a, B> {
    source: &'a B,
    destination: &'a B,
    region: CopyRegion,
    source_retention: DeviceBufferRetention,
    destination_retention: DeviceBufferRetention,
}

impl<'a, B> DeviceTensorStateCopy<'a, B> {
    pub(crate) fn new(
        source: &'a B,
        destination: &'a B,
        region: CopyRegion,
        source_retention: DeviceBufferRetention,
        destination_retention: DeviceBufferRetention,
    ) -> Self {
        Self {
            source,
            destination,
            region,
            source_retention,
            destination_retention,
        }
    }

    pub fn source(&self) -> &'a B {
        self.source
    }
    pub fn destination(&self) -> &'a B {
        self.destination
    }
    pub fn region(&self) -> CopyRegion {
        self.region
    }
    pub fn source_retention(&self) -> &DeviceBufferRetention {
        &self.source_retention
    }
    pub fn destination_retention(&self) -> &DeviceBufferRetention {
        &self.destination_retention
    }
}
