use ferrum_interfaces::vnext::*;

fn move_stream<R: DeviceRuntime>(
    resources: &AdmittedSequenceResources<R>,
    mut stream: BoundExecutionStream<R>,
) {
    let permit = resources.activate(&mut stream).unwrap();
    drop(stream);
    drop(permit);
}

fn main() {}
