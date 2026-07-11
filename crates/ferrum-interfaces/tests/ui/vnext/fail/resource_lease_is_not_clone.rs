use ferrum_interfaces::vnext::{DeviceRuntime, StaticProvisioningLease};

fn duplicate<R: DeviceRuntime>(lease: StaticProvisioningLease<R>) {
    let _duplicate = lease.clone();
}

fn main() {}
