use ferrum_interfaces::vnext::*;

fn replace_runtime<R: DeviceRuntime>(
    resources: &AdmittedSequenceResources<R>,
    mut stream: BoundExecutionStream<R>,
    replacement: BoundExecutionStream<R>,
) {
    let permit = resources.activate(&mut stream).unwrap();
    stream = replacement;
    drop(permit);
}

fn main() {}
