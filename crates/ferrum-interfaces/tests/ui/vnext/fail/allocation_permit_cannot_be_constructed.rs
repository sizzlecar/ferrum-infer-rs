use ferrum_interfaces::vnext::*;

fn duplicate(permit: DeviceAllocationPermit<'_>) {
    let _ = permit.clone();
}

fn main() {}
