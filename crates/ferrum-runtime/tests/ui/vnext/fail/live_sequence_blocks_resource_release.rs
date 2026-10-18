use ferrum_runtime::*;

fn release_live_sequence_capacity<R: DeviceRuntime>(sequence: &mut AdmittedSequenceResources<R>) {
    let _ = &mut sequence.logical_lease;
}

fn main() {}
