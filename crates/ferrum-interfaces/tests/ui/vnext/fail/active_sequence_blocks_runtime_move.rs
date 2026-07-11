use ferrum_interfaces::vnext::*;

fn raw_runtime_is_private<R: DeviceRuntime>(stream: &BoundExecutionStream<R>) {
    let _runtime = &stream.runtime;
}

fn main() {}
