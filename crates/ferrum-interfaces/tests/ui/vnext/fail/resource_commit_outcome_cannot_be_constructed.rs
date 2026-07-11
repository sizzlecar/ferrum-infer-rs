use ferrum_interfaces::vnext::*;

fn forge(buffer: Vec<u8>, descriptor: BufferDescriptor) {
    let _ = ResourceCommitOutcome::new(
        ResourceId::new("resource.forged").unwrap(),
        1,
        descriptor,
        buffer,
    );
}

fn main() {}
