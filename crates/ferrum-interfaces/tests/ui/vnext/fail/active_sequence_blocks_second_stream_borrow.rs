use ferrum_interfaces::vnext::*;

fn borrow_stream_twice<R: DeviceRuntime>(
    resources: &AdmittedSequenceResources<R>,
    mut stream: BoundExecutionStream<R>,
) {
    let permit = resources.activate(&mut stream).unwrap();
    let second = &mut stream;
    drop(second);
    drop(permit);
}

fn main() {}
