use ferrum_runtime::*;

fn duplicate(permit: DeviceAllocationPermit<'_>) {
    let _ = permit.clone();
}

fn main() {}
