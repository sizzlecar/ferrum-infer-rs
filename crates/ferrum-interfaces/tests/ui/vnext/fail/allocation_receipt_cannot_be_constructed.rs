use ferrum_interfaces::vnext::*;
use std::marker::PhantomData;

fn forge(
    resource_id: ResourceId,
    descriptor: BufferDescriptor,
) -> DeviceAllocationReceipt<'static> {
    DeviceAllocationReceipt {
        resource_id,
        generation: 1,
        descriptor,
        scope: PhantomData,
    }
}

fn main() {}
