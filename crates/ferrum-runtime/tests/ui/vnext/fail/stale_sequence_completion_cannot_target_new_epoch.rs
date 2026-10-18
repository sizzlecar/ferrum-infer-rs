use ferrum_runtime::*;

fn attack<R: DeviceRuntime>(
    first: SynchronizedSequencePermit<'_, '_, R>,
    second: SynchronizedSequencePermit<'_, '_, R>,
) {
    let stale = first.complete().unwrap();
    let _ = second.complete(stale);
}

fn main() {}
