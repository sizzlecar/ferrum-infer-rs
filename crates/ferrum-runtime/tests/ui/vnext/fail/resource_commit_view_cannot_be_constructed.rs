use ferrum_runtime::{BufferDescriptor, ResourceCommitView, ResourceId};

fn forge<'a>(
    resource_id: &'a ResourceId,
    descriptor: &'a BufferDescriptor,
    buffer: &'a Vec<u8>,
) -> ResourceCommitView<'a, Vec<u8>> {
    ResourceCommitView {
        resource_id,
        generation: 1,
        descriptor,
        buffer,
    }
}

fn main() {}
