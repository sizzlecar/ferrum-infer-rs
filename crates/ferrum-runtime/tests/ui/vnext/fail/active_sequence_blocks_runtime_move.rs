use ferrum_runtime::*;

fn raw_runtime_is_private<R: DeviceRuntime>(stream: &BoundExecutionStream<R>) {
    let _runtime = &stream.runtime;
}

fn main() {}
