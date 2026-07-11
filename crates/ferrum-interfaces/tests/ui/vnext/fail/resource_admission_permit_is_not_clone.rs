use ferrum_interfaces::vnext::{DeviceRuntime, StaticProvisioningPermit};

fn duplicate<R: DeviceRuntime>(permit: StaticProvisioningPermit<R>) {
    let _duplicate = permit.clone();
}

fn main() {}
